import dis
import inspect
import types

import opcode
import torch  # # aehem.

import thunder
from thunder.langs.torch import _torch_to_thunder_complete_map

from .frontend import acquire_method, make_single_return, make_ssa, remove_unused_values
from .graph import Block, clone_blocks, Node, PhiValue, replace_values, Value
from .python_ir import get_instruction


def specify_inputs(gr, inps):
    inp_map = {p: v for p, v in zip(gr.local_variables_at_start, inps)}
    replace_values(gr, inp_map)


def split_block(gr, bl, n):
    # The admin involved:
    # - create a new "bottom block", the input block is the "top block"
    # - split the .nodes
    # - block_inputs of the top block and block_outputs of the bottom are the original
    #   block_inputs and block_outputs
    # - scan all the node inputs and block_outputs of the lower part to see
    #   which need to be block_inputs of the lower block and thus outputs of the top one
    # - define outputs of the "top block" to be the required inputs
    # - add the input PhiValues and replace the outputs of the top block with them in the
    #   uses in the bottom block
    # - add unconditional jump from top to bottom part

    i = 0
    while i < len(gr.blocks) and gr.blocks[i] is not bl:
        i += 1
    assert i < len(gr.blocks), "block not found"
    j = 0
    while j < len(bl.nodes) and bl.nodes[j] is not n:
        j += 1
    assert j < len(bl.nodes), "node not found"
    nbl = Block(is_ssa=True)
    nbl.nodes = bl.nodes[j:]
    del bl.nodes[j:]
    nbl.block_outputs = bl.block_outputs
    bl.block_outputs = set()
    nbl.block_inputs = []

    jump_ins = dis.Instruction(
        opname="JUMP_ABSOLUTE",
        opcode=opcode.opmap["JUMP_ABSOLUTE"],
        arg=None,
        argval=None,
        argrepr=None,
        offset=None,  # last_node_i.offset,
        starts_line=None,
        is_jump_target=False,
    )
    bl_jump_node = Node(i=jump_ins, inputs=[], outputs=[])
    bl_jump_node.jump_targets = [((0, 0), nbl)]
    bl.nodes.append(bl_jump_node)
    nbl.jump_sources.append(bl_jump_node)
    gr.blocks.insert(i + 1, nbl)

    potential_bl_outputs = {i for i in bl.block_inputs}
    for n in bl.nodes:
        for o in n.outputs:
            potential_bl_outputs.add(o)
    for i in bl.block_inputs:
        potential_bl_outputs.add(i)
    value_map = {}

    def get_or_create_phi(v):
        if v in value_map:
            return value_map[v]
        if v.is_const or v.is_global:
            return v
        if v in potential_bl_outputs:  # priority follow parent vs. phi_value?
            phi_value = PhiValue([v], [bl_jump_node], nbl)
            nbl.block_inputs.append(phi_value)
            bl.block_outputs.add(v)
            value_map[v] = phi_value
            return phi_value
        if v.parent is not None:
            # this adds v.parent to the value_map, so that is used
            # for the clone's parent
            get_or_create_phi(v.parent)
            return v.clone(translation_dict=value_map)
        raise ValueError(f"unknwn value {v}")

    for n in nbl.nodes:
        n.inputs = [get_or_create_phi(i) for i in n.inputs]
        for o in n.outputs:
            value_map[o] = o
        # for inplace ops, we also check the outputs (e.g. FOR_ITER)
        for idx_o, o in enumerate(n.outputs):
            if o in potential_bl_outputs:
                o.outputs[idx_o] = get_or_create_phi(o)
                bl.block_outputs.add(o)

    bl.block_outputs.update(nbl.block_outputs & potential_bl_outputs)
    nbl.block_outputs = {(get_or_create_phi(o) if o in potential_bl_outputs else o) for o in nbl.block_outputs}

    return nbl


def find_method_through_phi_parent(fn_value):
    # for inlining, we need to (reverse) traverse PhiValues and attribute
    # lookups to find the actual function we want to inline
    while isinstance(fn_value, PhiValue) and len(fn_value.values) == 1:
        fn_value = fn_value.values[0]
    if fn_value.parent is not None and fn_value.name is not None:
        parent_value, attr_lookups = find_method_through_phi_parent(fn_value.parent)
        attr_lookups.append(fn_value.name)
        return parent_value, attr_lookups
    if fn_value.node is not None and fn_value.node.i.opname == "BINARY_SUBSCR" and fn_value.node.inputs[1].is_const:
        parent_value, attr_lookups = find_method_through_phi_parent(fn_value.node.inputs[0])
        attr_lookups.append(f"[{fn_value.node.inputs[1].value}]")
        return parent_value, attr_lookups

    return fn_value, []


def find_and_evaluate_method_through_phi_parent(v):
    fn_parent_value, attr_lookups = find_method_through_phi_parent(v)
    if fn_parent_value.value is None:
        return None
    fn_value = fn_parent_value.value
    for al in attr_lookups:
        if al.startswith("["):
            fn_value = fn_value[int(al[1:-1])]  # non-int lookups?
        else:
            fn_value = getattr(fn_value, al)
    return fn_value


def inline_method_call(gr, n):  # criterion?
    found_block = False
    for i_bl, bl in enumerate(gr.blocks):
        for i_n, n1 in enumerate(bl.nodes):
            if n1 is n:  # is?
                found_block = True
                break
        if found_block:
            break
    assert found_block
    if n.i.opname == "CALL_METHOD":
        fn_value = find_and_evaluate_method_through_phi_parent(n.inputs[0])
        if fn_value is None:
            raise NotImplementedError("cannot inline non-explicit function")

        ## TODO: value for self arg in Method calls?
        ### in general: What is with callables here?
        if isinstance(fn_value, torch.nn.Module):
            mod1 = fn_value
            value_for_self1 = n.inputs[0]
            fn_value = fn_value.forward
        elif inspect.ismethod(fn_value):
            mod1 = fn_value.__self__
            value_for_self1 = n.inputs[1]
        else:
            mod1 = None
            value_for_self1 = None

        if inspect.isbuiltin(fn_value):
            raise NotImplementedError("cannot inline built-in (C-implemented) function")
    elif n.i.opname == "CALL_FUNCTION":
        fn_value = find_and_evaluate_method_through_phi_parent(n.inputs[0])
        if fn_value is None:
            raise NotImplementedError("cannot inline non-explicit function")

        if isinstance(fn_value, torch.nn.Module):
            mod1 = fn_value
            value_for_self1 = n.inputs[0]
            fn_value = fn_value.forward
        else:
            if not isinstance(fn_value, types.FunctionType):
                raise NotImplementedError(f"inlining {n}")
            mod1 = None
            value_for_self1 = None
    else:
        raise NotImplementedError(f"inlining {n}")

    nbl = split_block(gr, bl, bl.nodes[i_n + 1])
    n1 = bl.nodes.pop(i_n)
    assert n1 is n

    gr1 = acquire_method(fn_value, module=mod1, mro_klass=gr.mro_klass if mod1 == gr.module else None)
    make_ssa(gr1)
    make_single_return(gr1)

    # there should be exactly one
    (ret_bl,) = (bl for bl in gr1.blocks if len(bl.nodes) > 0 and bl.nodes[-1].i.opname == "RETURN_VALUE")

    ret_node = ret_bl.nodes[-1]
    ret_node.i = dis.Instruction(
        opname="JUMP_ABSOLUTE",
        opcode=opcode.opmap["JUMP_ABSOLUTE"],
        arg=None,
        argval=None,
        argrepr=None,
        offset=ret_node.i.offset,
        starts_line=ret_node.i.starts_line,
        is_jump_target=ret_node.i.is_jump_target,
    )
    bl.nodes[-1].jump_targets = [((0, 0), gr1.blocks[0])]
    gr1.blocks[0].jump_sources = [bl.nodes[-1]]
    ret_node.jump_targets = [((0, 0), nbl)]
    nbl.jump_sources = [ret_node if js == bl.nodes[-1] else js for js in nbl.jump_sources]

    gr.blocks[i_bl + 1 : i_bl + 1] = gr1.blocks

    # TODO Error checking parameters
    if gr1.ismethod and n.i.opname == "CALL_METHOD":
        call_args = [value_for_self1, *n.inputs[2:]]
    elif gr1.ismethod and n.i.opname == "CALL_FUNCTION":
        call_args = [value_for_self1, *n.inputs[1:]]
    elif n.i.opname == "CALL_METHOD":
        call_args = n.inputs[2:]
    elif n.i.opname == "CALL_FUNCTION":
        call_args = n.inputs[1:]
    else:
        raise NotImplementedError()

    assert len(n.outputs) == 1
    bl.block_outputs.remove(n.outputs[0])  # TODO: what with inplace!!
    bl.block_outputs.update(call_args)
    specify_inputs(gr1, call_args)

    # output values...
    rv = ret_node.inputs.pop()
    assert not ret_node.inputs
    (orv,) = n.outputs
    replace_values(gr, {orv: rv})
    ret_bl.block_outputs.add(rv)


def inline_submodule_calls(gr):
    # inlines submodule calls
    # TODO: recursively and not from nested structures (ModuleList etc.)
    changed = False
    gr.ensure_links()
    for bl in gr.blocks[:]:
        for n in bl.nodes[:]:
            if n.i.opname in {"CALL_METHOD", "CALL_FUNCTION"}:
                fn_value = find_and_evaluate_method_through_phi_parent(n.inputs[0])
                if isinstance(fn_value, torch.nn.Module):
                    inline_method_call(gr, n)
                    changed = True
    return changed


def strongly_inline_functions(gr):
    loop = True
    while loop:
        loop = False
        gr.ensure_links()
        for bl in gr.blocks[:]:
            for n in bl.nodes[:]:
                if n.i.opname in {"CALL_METHOD", "CALL_FUNCTION"}:
                    fn_value = find_and_evaluate_method_through_phi_parent(n.inputs[0])
                    if (
                        fn_value is not None
                        and not inspect.isbuiltin(fn_value)
                        and isinstance(fn_value, types.FunctionType)
                        and fn_value not in _torch_to_thunder_complete_map
                    ):
                        ## handle methods or nn.Modules / other classes?
                        inline_method_call(gr, n)
                        loop = True


def torch_to_thunder(gr, fallback=False):
    """replaces calls to torch.foo functions with calls into thunder's torch language."""

    def fill_in_value(v):
        # PhiValues ?
        if v.value is None and v.parent is not None:
            fill_in_value(v.parent)
        if v.value is None and v.parent is not None and v.parent.value is not None and v.name is not None:
            v.value = getattr(v.parent.value, v.name)

    for bl in gr.blocks:
        for n in bl.nodes:
            for i in n.inputs:
                done = False
                fill_in_value(i)
                i_or_parent = i
                while i_or_parent.value not in _torch_to_thunder_complete_map and i_or_parent.parent is not None:
                    i_or_parent = i_or_parent.parent

                if i_or_parent.value in _torch_to_thunder_complete_map:
                    i_or_parent.value = _torch_to_thunder_complete_map[i.value]
                    i_or_parent.typ = type(i_or_parent.value)
                    i_or_parent.parent = None
                    i_or_parent.is_const = True
                    i_or_parent.is_global = False
                    if n.i.opname == "CALL_METHOD":
                        # todo get others, too
                        n.i = get_instruction(opname="CALL_FUNCTION", arg=n.i.arg)
                        del n.inputs[1]
                    done = True

                if (not done) and fallback:  # fallback
                    # todo: change name?, deeper nesting?
                    if i.value == torch:
                        i.value = thunder.langs.torch
                    if i.parent is not None and i.parent.value == torch:
                        i.parent.value = thunder.langs.torch
                        i.value = getattr(thunder.langs.torch, i.name)

                    # replace other things by checking against torch module (make dict at startup?)
                    n = getattr(i.value, "__name__", None)
                    tf = None
                    if n is not None:
                        tf = getattr(torch, n, None)
                    if tf is not None and i.value == tf:
                        i.value = getattr(thunder.langs.torch, n)
                        i.is_global = False
                        i.is_const = True


def merge_two_blocks(gr, bl1):
    jt = bl1.nodes[-1].jump_targets
    if len(jt) != 1:
        raise RuntimeError("can only fuse blocks with deterministic connection")
    bl2 = jt[0][1]
    if len(bl2.jump_sources) != 1 or bl2.jump_sources[0] != bl1.nodes[-1]:
        raise RuntimeError("second block to be fused must only have first block as jump source")

    replacements = {}
    for i in bl2.block_inputs:
        assert isinstance(i, PhiValue) and len(i.values) == 1
        (iv,) = i.values
        if iv in bl1.block_outputs:
            replacements[i] = iv
        else:
            bl1.block_inputs.append(i)
            i.block = bl1

    replace_values(bl2, replacements, follow_phi_values=True)
    # TODO: Should this happen automatically in replace_values?
    #       Should we also replace values in bl1?
    for o in bl1.block_outputs:
        for pv in o.phi_values:
            if pv in replacements:
                pv.remove_value(o)
    bl1.block_outputs = {o for o in bl1.block_outputs if o.phi_values}
    bl1.block_outputs.update(bl2.block_outputs)

    bl1.nodes[-1:] = bl2.nodes
    gr.blocks.remove(bl2)


def merge_blocks_where_possible(gr):
    i_bl = 0
    while i_bl < len(gr.blocks):
        bl1 = gr.blocks[i_bl]
        jt = bl1.nodes[-1].jump_targets
        if len(jt) == 1:
            bl2 = jt[0][1]
        else:
            bl2 = None
        if bl2 is not None and len(bl2.jump_sources) == 1 and bl2.jump_sources[0] == bl1.nodes[-1]:
            merge_two_blocks(gr, bl1)
        else:
            i_bl += 1


def find_blocks_of_for(gr, for_block):
    assert for_block.nodes[-1].i.opname == "FOR_ITER"

    blocks_of_for_loop = {for_block}
    currently_looking_at = set()

    def find_blocks_of_for_rec(for_block, start_block):
        if for_block == start_block:
            return True
        if start_block in currently_looking_at:
            return False
        currently_looking_at.add(start_block)
        found = False
        for _, jt in start_block.nodes[-1].jump_targets:
            found |= find_blocks_of_for_rec(for_block, jt)
        currently_looking_at.remove(start_block)
        if found:
            blocks_of_for_loop.add(start_block)
        return found

    find_blocks_of_for_rec(for_block, for_block.nodes[-1].jump_targets[0][1])
    return blocks_of_for_loop


def unroll_for_over_modules(gr, for_iter_node):
    gr.ensure_links()
    get_iter_node = for_iter_node.inputs[0].values[0].node
    assert get_iter_node.i.opname == "GET_ITER"

    iterated_module_list_parent, attr_lookups = thunder.core.script.passes.find_method_through_phi_parent(
        get_iter_node.inputs[0]
    )
    assert iterated_module_list_parent.value is not None
    iterated_module_list = iterated_module_list_parent.value
    for al in attr_lookups:
        iterated_module_list = getattr(iterated_module_list, al)

    # what about more complex things?
    assert isinstance(iterated_module_list, (torch.nn.Sequential, torch.nn.ModuleList))

    for_loop_len = len(iterated_module_list)
    for_iter_block = for_iter_node.block
    get_iter_block = get_iter_node.block

    (iter_v,) = get_iter_node.outputs
    (iter_phi,) = for_iter_node.inputs

    assert isinstance(iter_phi, PhiValue)
    assert iter_v in iter_phi.values

    ### first we find the blocks of the for loop
    bls = find_blocks_of_for(gr, for_iter_block)

    jmp_nodes = {bl.nodes[-1] for bl in bls}
    assert all((v is iter_v or js in jmp_nodes) for v, js in zip(iter_phi.values, iter_phi.jump_sources))

    for_iter_node.i = get_instruction(opname="BINARY_SUBSCR", arg=None)
    iter_phi.remove_value(iter_v)
    assert len(iter_v.phi_values) == 0
    get_iter_block.block_outputs.remove(iter_v)

    get_iter_block.block_outputs.add(get_iter_node.inputs[0])

    seen = set()

    def delete_value_and_sources(v):
        # check that it is possible?
        if v in seen:
            return
        seen.add(v)
        if isinstance(v, PhiValue):
            for vv, js in zip(v.values, v.jump_sources):
                delete_value_and_sources(vv)
                js.block.block_outputs.remove(vv)
            v.block.block_inputs.remove(v)

    delete_value_and_sources(iter_phi)
    seq_phi = PhiValue(values=[get_iter_node.inputs[0]], jump_sources=[get_iter_block.nodes[-1]], block=for_iter_block)
    get_iter_block.nodes.remove(get_iter_node)
    for_iter_block.block_inputs.append(seq_phi)

    idx = Value(value=0, is_const=True)
    for_iter_node.inputs = [seq_phi, idx]
    for_iter_node.outputs = [for_iter_node.outputs[1]]

    for_iter_block_jmp = Node(i=get_instruction(opname="JUMP_ABSOLUTE", arg=None))
    for_iter_block.nodes.append(for_iter_block_jmp)
    for_iter_block_jmp.jump_targets = [((0, 0), for_iter_node.jump_targets[0][1])]
    _, for_iter_node_exit_jump_target = for_iter_node.jump_targets[1]
    for_iter_node.jump_targets = []
    for_iter_block_jmp.jump_targets[0][1].jump_sources = [
        (js if js is not for_iter_node else for_iter_block_jmp)
        for js in for_iter_block_jmp.jump_targets[0][1].jump_sources
    ]

    exit_block = Block()
    gr.blocks.append(exit_block)
    exit_node = Node(i=get_instruction(opname="JUMP_ABSOLUTE", arg=None))
    exit_node.jump_targets = [((0, 0), for_iter_node_exit_jump_target)]
    target_after_iter = exit_node.jump_targets[0][1]
    exit_node.jump_targets[0][1].jump_sources = [
        (js if js is not for_iter_node else exit_node) for js in exit_node.jump_targets[0][1].jump_sources
    ]
    exit_block.nodes.append(exit_node)
    for i in for_iter_block.block_inputs:
        exit_block.block_inputs.append(PhiValue([], [], exit_block))

    unroll_blocks = [(list(bls), {})] + [clone_blocks(bls) for _ in range(1, for_loop_len)]
    for i, (nbls, td) in enumerate(unroll_blocks):
        if i > 0:
            gr.blocks += nbls
            idx = Value(value=i, is_const=True)
            td[for_iter_node].inputs[1] = idx
            td[for_iter_node].outputs[0].name += f"_{i}"
        else:
            for_iter_node.outputs[0].name += "_0"

    gr.ensure_links()

    fixup_data = []
    for idx, (nbls, td) in enumerate(unroll_blocks):
        if idx == 0:
            fib_i = for_iter_block
            jump_sources_to_fix = [js for js in for_iter_block.jump_sources if js is not get_iter_block.nodes[-1]]
        else:
            fib_i = td[for_iter_block]
            jump_sources_to_fix = td[for_iter_block].jump_sources[:]
        if idx + 1 < len(unroll_blocks):
            _, td_next = unroll_blocks[idx + 1]
            fib_next = td_next[for_iter_block]
        else:
            fib_next = exit_block

        fixup_data.append((fib_i, jump_sources_to_fix, fib_next, nbls))

    for idx_it, (fib_i, jump_sources_to_fix, fib_next, nbls) in enumerate(fixup_data):
        for js in jump_sources_to_fix:
            for idx, (_, jt) in enumerate(js.jump_targets):
                if jt == fib_i:
                    js.set_jump_target(fib_next, idx=idx)

        for idx_i, i in enumerate(fib_i.block_inputs):
            if any((js.block in nbls) for js in i.jump_sources):
                ## if this is a variable updated in the loop:
                ##  - instead of looping back, point the update to the phi value of the next block (or the exit block)
                ##  - if idx > 0: remove external (before the loop) value
                for v, js in zip(i.values[:], i.jump_sources[:]):
                    assert js.block is not None
                    if js.block not in nbls and idx_it > 0:
                        i.remove_value(v)

    for idx_it, (fib_i, jump_sources_to_fix, fib_next, nbls) in enumerate(fixup_data):
        for idx_i, i in enumerate(fib_i.block_inputs):
            if any((js.block in nbls) for js in i.jump_sources):
                for v, js in zip(i.values[:], i.jump_sources[:]):
                    if js.block in nbls:
                        i.remove_value(v)
                        fib_next.block_inputs[idx_i].add_missing_value(v, jump_source=js)
                if idx_it == 0:
                    for pv in i.phi_values[:]:
                        if pv.block is target_after_iter:
                            pv.remove_value(i)
                            pv.add_missing_value(exit_block.block_inputs[idx_i], jump_source=exit_node)

    for i in exit_block.block_inputs[:]:
        if i.phi_values:
            exit_block.block_outputs.add(i)
        else:
            exit_block.block_inputs.remove(i)


def find_and_unroll_for_loop(gr):
    gr.ensure_links()

    for bl in gr.blocks[:]:
        for n in bl.nodes[:]:
            if n.i.opname == "FOR_ITER":
                for_iter_node = n
                get_iter_node = for_iter_node.inputs[0].values[0].node
                if get_iter_node.i.opname == "GET_ITER":
                    (
                        iterated_module_list_parent,
                        attr_lookups,
                    ) = thunder.core.script.passes.find_method_through_phi_parent(get_iter_node.inputs[0])
                    if iterated_module_list_parent.value is None:
                        continue
                    iterated_module_list = iterated_module_list_parent.value
                    for al in attr_lookups:
                        iterated_module_list = getattr(iterated_module_list, al)
                    # what about more complex things? in particular enumerate, but zip, ...
                    if isinstance(iterated_module_list, (torch.nn.Sequential, torch.nn.ModuleList)):
                        thunder.core.script.passes.unroll_for_over_modules(gr, for_iter_node)
                        thunder.core.script.passes.merge_blocks_where_possible(gr)
                        return True
    return False


def unroll_for_loops_and_inline_modules(gr):
    iterate = True
    while iterate:
        iterate = find_and_unroll_for_loop(gr)
        if not iterate:
            iterate = inline_submodule_calls(gr)
            if iterate:
                thunder.core.script.passes.merge_blocks_where_possible(gr)


def module_to_function(gr):
    attr_dict = {}
    attr_list = []
    gr.local_variables_at_start = [lv for lv in gr.local_variables_at_start if lv is not None]
    offset = len(gr.local_variables_at_start)
    for bl in gr.blocks:
        for n in bl.nodes:
            for idx_i, i in enumerate(n.inputs):
                # TODO: inefficient because it looks twice
                v = find_and_evaluate_method_through_phi_parent(i)
                maybe_self, attrs = find_method_through_phi_parent(i)

                if maybe_self.value is gr.module and isinstance(v, torch.Tensor):
                    attr_string = ".".join(attrs)
                    idx = attr_dict.setdefault(attr_string, offset + len(attr_dict))
                    if idx >= len(gr.local_variables_at_start):
                        func_arg = Value(name=attr_string, is_function_arg=True)
                        gr.local_variables_at_start.append(func_arg)
                        attr_list.append(attr_string)
                    else:
                        func_arg = gr.local_variables_at_start[idx]

                    pvs = [pv for pv in func_arg.phi_values if pv.block is bl]
                    if not pvs:
                        pv = PhiValue([func_arg], [None], bl)
                        bl.block_inputs.append(pv)
                    else:
                        (pv,) = pvs
                    n.inputs[idx_i] = pv
                    ## remove old input from phi_values etc?
                elif maybe_self.value is gr.module and (
                    n.i.opname not in {"BINARY_SUBSCR"} or not isinstance(v, torch.nn.Module)
                ):

                    ## inline to const...
                    i.value = v

                    i.typ = type(i.value)
                    i.parent = None
                    i.is_const = True
                    i.is_global = False

    remove_unused_values(gr)
    if gr.local_variables_at_start[0].phi_values:
        raise RuntimeError("could not eliminate self argument")
    del gr.local_variables_at_start[0]
    return attr_list
