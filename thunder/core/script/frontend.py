import collections
import dis
import inspect
import itertools
import opcode
import sys

import torch

from thunder.core.script.graph import Block, Graph, MROAwareObjectRef, Node, NULL, PhiValue, Value
from thunder.core.script.python_ir_data import jump_instructions, stack_effect_detail, unconditional_jump_names


class Super:
    pass


def acquire_method(method, module=None, mro_klass=None, verbose=False):
    if isinstance(method, torch.nn.Module):
        method = method.forward
    assert sys.version_info >= (3, 9) and sys.version_info < (3, 11)
    if verbose:
        print(inspect.getsource(method))
    sig = inspect.signature(method)
    if module is None and hasattr(method, "__self__"):
        module = method.__self__
    if mro_klass is None and module is not None:
        mro_klass = type(module)
    local_variables = []
    self_value = Value(value=module, name=method.__code__.co_varnames[0], is_function_arg=True)
    if inspect.ismethod(method):
        local_variables.append(self_value)
    else:
        self_value = NULL
    for p in sig.parameters.values():
        assert (
            p.name == method.__code__.co_varnames[len(local_variables)]
        ), f"mismatch {p.name} {method.__code__.co_varnames[len(local_variables)]}"
        local_variables.append(Value(typ=p.annotation, name=p.name, is_function_arg=True))
    # KWARGS?!
    for i in enumerate(method.__code__.co_varnames, start=len(local_variables)):
        local_variables.append(None)

    # bound_args = [module.forward.__self__]
    bc = list(dis.get_instructions(method))
    if verbose:
        print(dis.dis(method))
    # Map offset_start -> Block
    block_0 = Block(is_ssa=False)
    block_0.jump_sources.append(None)
    blocks_to_process = collections.OrderedDict({0: block_0})
    blocks = {}

    def get_or_make_block(offset_start, jump_source):
        for other_offset_start, other_bl in itertools.chain(blocks_to_process.items(), blocks.items()):
            if other_offset_start == offset_start:
                # take anything?
                # print("#oldbl##", offset_start, jump_source, other_bl.jump_sources)
                other_bl.jump_sources.append(jump_source)
                return other_bl
        # print("#newbl##", offset_start, jump_source, other_bl.jump_sources)
        bl = Block(is_ssa=False)
        blocks_to_process[offset_start] = bl
        bl.jump_sources.append(jump_source)
        return bl

    line_no = 0
    while blocks_to_process:
        offset_start, bl = blocks_to_process.popitem(last=False)
        blocks[offset_start] = bl

        ic = offset_start
        done = False
        while not done:
            i = bc[ic]
            if i.starts_line is not None:
                line_no = i.starts_line
            n = Node(i=i, line_no=line_no)

            # need to handle branching instructions here
            if i.opcode in jump_instructions:
                done = True
                if i.opcode in dis.hasjabs:
                    ic_target = i.arg
                elif "BACKWARD" in i.opname:
                    ic_target = ic + 1 - i.arg  # ?
                else:
                    ic_target = ic + 1 + i.arg

                if i.opname in unconditional_jump_names:
                    n.jump_targets = []
                else:
                    b1 = get_or_make_block(offset_start=ic + 1, jump_source=n)
                    n.jump_targets = [(stack_effect_detail(i.opname, i.arg, jump=False), b1)]

                b1 = get_or_make_block(offset_start=ic_target, jump_source=n)
                n.jump_targets.append((stack_effect_detail(i.opname, i.arg, jump=True), b1))
            elif i.opname in {"RETURN_VALUE", "RAISE_VARARGS"}:
                done = True
            else:
                if verbose:
                    print(i)

            bl.insert_node(n)
            ic += 1
            if ic < len(bc) and bc[ic].is_jump_target:
                # check if needed?
                if i.opname not in {"RETURN_VALUE", "RAISE_VARARGS"} and i.opcode not in jump_instructions:
                    # should insert jump absolute instead...
                    jump_ins = dis.Instruction(
                        opname="JUMP_ABSOLUTE",
                        opcode=None,  # the JUMP_ABSOLUTE is not in Python 3.11
                        arg=None,
                        argval=None,
                        argrepr=None,
                        offset=None,
                        starts_line=None,
                        is_jump_target=False,
                    )
                    jump_node = Node(i=jump_ins, inputs=[], outputs=[])
                    bl.insert_node(jump_node)
                    b1 = get_or_make_block(offset_start=ic, jump_source=jump_node)
                    jump_node.jump_targets = [
                        (
                            stack_effect_detail(jump_ins.opname, jump_ins.arg, jump=True),
                            b1,
                        )
                    ]
                done = True
    gr = Graph(list(blocks.values()))
    gr.local_variables_at_start = local_variables
    gr.ismethod = inspect.ismethod(method)
    gr.method = method
    gr.module = module
    gr.mro_klass = mro_klass
    gr.self_value = self_value
    return gr


def make_ssa(gr, verbose=False):
    for bl in gr.blocks:
        for n in bl.nodes:
            n.block = bl
        bl.all_stacks_at_start = [None if js is not None else [] for js in bl.jump_sources]
        bl.all_local_variables_at_start = [
            None if js is not None else gr.local_variables_at_start[:] for js in bl.jump_sources
        ]

    blocks_to_do = set(gr.blocks)
    while blocks_to_do:
        next_block = None
        for bl in blocks_to_do:
            all_deps_done = not any(js.block in blocks_to_do for js in bl.jump_sources if js is not None)
            if all_deps_done:
                next_block = bl
                break
        if next_block is None:
            # we need to break a cycle, so we choose one where we have variables for one branch
            for bl in blocks_to_do:
                any_deps_done = any(js.block not in blocks_to_do for js in bl.jump_sources)
                if any_deps_done:
                    next_block = bl
                    break

        assert next_block is not None
        bl = next_block
        blocks_to_do.remove(bl)
        assert not bl.is_ssa
        bl.is_ssa = True

        jump_sources = bl.jump_sources

        all_stacks_at_start = bl.all_stacks_at_start
        all_local_variables_at_start = bl.all_local_variables_at_start
        bl.jump_source_idxes_to_postprocess = [i for i, s in enumerate(all_stacks_at_start) if s is None]
        if bl.jump_source_idxes_to_postprocess:
            # TODO: Check what is going on with loops w.r.t. types
            all_complete_stacks_at_start = [s for s in bl.all_stacks_at_start if s is not None]
            all_complete_local_variables_at_start = [lv for lv in bl.all_local_variables_at_start if lv is not None]

            stack_depth_at_start = len(all_complete_stacks_at_start[0])
            num_lv_at_start = len(all_complete_local_variables_at_start[0])

            all_stacks_at_start = [
                (s if s is not None else [None for _ in range(stack_depth_at_start)]) for s in bl.all_stacks_at_start
            ]
            all_local_variables_at_start = [
                (lv if lv is not None else [None for _ in range(num_lv_at_start)])
                for lv in bl.all_local_variables_at_start
            ]

        stack = [PhiValue(v, jump_sources, bl) for v in zip(*all_stacks_at_start)]
        local_variables = [PhiValue(v, jump_sources, bl) for v in zip(*all_local_variables_at_start)]

        bl.block_inputs = stack + local_variables
        bl.stack_depth_at_start = len(stack)

        new_nodes = []
        for n_idx, n in enumerate(bl.nodes):
            i = n.i
            pop, push = stack_effect_detail(i.opname, i.arg)  # jump?
            inputs = stack[-pop:] if pop > 0 else []
            n.inputs = inputs[:]
            assert len(inputs) == pop, f"stack to shallow {len(inputs)=} {pop=} {i=}"
            if i.opname == "LOAD_FAST":
                outputs = [local_variables[i.arg]]
            elif i.opname == "STORE_FAST":
                outputs = []
                (local_variables[i.arg],) = inputs  # set name?
            elif i.opname == "DELETE_FAST":
                outputs = []
                local_variables[i.arg] = None
            elif i.opname == "LOAD_GLOBAL":
                if gr.method.__code__.co_names[i.arg] != "super":
                    if inspect.ismethod(gr.method):
                        func = gr.method.__func__
                    else:
                        func = gr.method
                    gn = gr.method.__code__.co_names[i.arg]
                    NOT = object()
                    gv = func.__globals__.get(gn, NOT)
                    if gv is NOT:
                        gv = func.__builtins__[gn]
                    outputs = [Value(name=gn, value=gv, is_global=True)]
                else:
                    outputs = [Value(name="super", value=Super())]
            elif i.opname == "LOAD_ATTR":
                an = gr.method.__code__.co_names[i.arg]
                ap = inputs[0]
                outputs = [Value(name=an, parent=ap)]
            elif i.opname == "CALL_FUNCTION" and i.arg == 0 and isinstance(inputs[0].value, Super):
                outputs = [Value(value=MROAwareObjectRef(gr.self_value, start_klass=gr.mro_klass))]
                print("##super#", outputs)
            elif i.opname == "LOAD_METHOD":  # also used for modules (callables)
                (obj,) = inputs
                mn = gr.method.__code__.co_names[i.arg]
                m = Value(parent=obj, name=mn)
                if obj.value is not None:
                    m.value = getattr(obj.value, mn)
                    m.typ = type(m.value)
                # error case
                # print("#lm###", type(m), type(obj), str(obj.value)[:100], m.value)
                if isinstance(obj.value, MROAwareObjectRef):
                    pass
                    # print("...###", obj.value.start_klass)
                #    obj = obj.obj
                outputs = [m, obj]
            elif i.opname == "LOAD_CONST":
                outputs = [Value(value=gr.method.__code__.co_consts[i.arg], is_const=True)]
            elif i.opname == "CALL_METHOD":
                outputs = [Value(node=n, nr=k) for k in range(push)]
                new_nodes.append(n)
            elif i.opname == "FOR_ITER":
                # JUMP TARGETS
                outputs = [inputs[0], Value(node=n, name=".for_iter_item")]
                new_nodes.append(n)
            elif i.opname in {
                "POP_JUMP_IF_FALSE",
                "POP_JUMP_IF_TRUE",
                "JUMP_FORWARD",
                "JUMP_ABSOLUTE",
            }:
                new_nodes.append(n)
                outputs = []
            # elif i.opname == "JUMP_FORWARD":
            # elif i.opname == "JUMP_ABSOLUTE":
            elif i.opname == "RETURN_VALUE":
                assert len(stack) == 1
                new_nodes.append(n)
                outputs = []
            else:
                if verbose:
                    print("unhandled", i)
                outputs = [Value(node=n, nr=k) for k in range(push)]
                new_nodes.append(n)
            if n.jump_targets is not None:
                all_block_outputs = set(local_variables)
                for (j_pop, j_push), jt in n.jump_targets:
                    idx_jt = jt.jump_sources.index(n)
                    j_stack = stack[:]
                    if j_pop > 0:
                        j_stack = j_stack[:-pop]
                    if j_push > 0:
                        j_stack.extend(outputs[:j_push])
                    jt.all_stacks_at_start[idx_jt] = j_stack
                    jt.all_local_variables_at_start[idx_jt] = local_variables[:]
                    all_block_outputs.update(j_stack)
                bl.block_outputs = all_block_outputs
            n.outputs = outputs
            ol = len(stack)
            if pop > 0:
                stack = stack[:-pop]
            stack.extend(outputs)
            assert (i.opname == "JUMP_ABSOLUTE" and i.arg is None and len(stack) == ol) or (
                len(stack) - ol == opcode.stack_effect(i.opcode, i.arg)
            )
        bl.nodes = new_nodes

    for bl in gr.blocks:
        for idx_js in bl.jump_source_idxes_to_postprocess:
            assert len(bl.all_stacks_at_start[idx_js]) == bl.stack_depth_at_start
            assert len(bl.block_inputs) == bl.stack_depth_at_start + len(bl.all_local_variables_at_start[idx_js])

            for idx_i, i in enumerate(bl.block_inputs):
                assert len(bl.jump_sources) == len(i.values)
                if idx_i < bl.stack_depth_at_start:
                    v = bl.all_stacks_at_start[idx_js][idx_i]
                else:
                    v = bl.all_local_variables_at_start[idx_js][idx_i - bl.stack_depth_at_start]
                i.add_missing_value(v, idx_js)

    for bl in gr.blocks:
        del bl.all_local_variables_at_start
        del bl.all_stacks_at_start
        del bl.stack_depth_at_start
        del bl.jump_source_idxes_to_postprocess

    remove_unused_values(gr)


def remove_unused_values(gr):
    gr.ensure_links()
    values_used = set()

    INDEX_OPS = {"BINARY_SUBSCR"}

    def mark_used(v):
        if v in values_used:
            return
        values_used.add(v)
        if v.node and v.node.i.opname in INDEX_OPS:
            for i in v.node.inputs:
                mark_used(i)
        if v.parent is not None:
            mark_used(v.parent)
        if isinstance(v, PhiValue):
            for w in v.values:
                mark_used(w)

    for bl in gr.blocks:
        for n in bl.nodes:
            if n.i.opname not in INDEX_OPS:
                for i in n.inputs:
                    mark_used(i)

    for bl in gr.blocks:
        for i in bl.block_inputs[:]:
            if i not in values_used:
                for v in i.values[:]:
                    if v is not None:
                        i.remove_value(v)
                bl.block_inputs.remove(i)
        bl.block_outputs = {o for o in bl.block_outputs if o in values_used}
        for n in bl.nodes[:]:
            if n.i.opname in INDEX_OPS and not any((o in values_used) for o in n.outputs):
                bl.nodes.remove(n)
    for i in gr.local_variables_at_start:
        if i is not None:
            i.phi_values = [pv for pv in i.phi_values if pv in values_used]

    for bl in gr.blocks:
        for n in bl.nodes:
            for o in n.outputs:
                o.phi_values = [pv for pv in o.phi_values if pv in values_used]

    # remove things only used in current block (and not in own phi) from outputs
    # TODO: think if this would obsolete the above
    outputs_used = set()
    for bl in gr.blocks:
        for i in bl.block_inputs:
            assert isinstance(i, PhiValue)
            for v in i.values:
                outputs_used.add(v)
    for bl in gr.blocks:
        bl.block_outputs = {o for o in bl.block_outputs if o in outputs_used}


def make_single_return(gr):
    bls = [b for b in gr.blocks if b.nodes[-1].i.opname == "RETURN_VALUE"]
    if len(bls) > 1:
        ret_bl = Block(is_ssa=True)
        ret_ins = dis.Instruction(
            opname="RETURN_VALUE",
            opcode=opcode.opmap["RETURN_VALUE"],
            arg=None,
            argval=None,
            argrepr=None,
            offset=None,
            starts_line=None,
            is_jump_target=False,
        )
        ret_input = PhiValue([], [], ret_bl)
        ret_node = Node(i=ret_ins, inputs=[ret_input], outputs=[])
        ret_bl.nodes = [ret_node]
        ret_bl.jump_sources = []
        ret_node.inputs = [ret_input]
        gr.blocks.append(ret_bl)
        ret_bl.block_outputs = {}
        ret_bl.block_inputs = [ret_input]

        for b in bls:
            # jump sources + unify!!!
            last_node_i = b.nodes[-1].i
            assert last_node_i.opname == "RETURN_VALUE"
            jump_ins = dis.Instruction(
                opname="JUMP_ABSOLUTE",
                opcode=opcode.opmap["JUMP_ABSOLUTE"],
                arg=None,
                argval=None,
                argrepr=None,
                offset=last_node_i.offset,
                starts_line=None,
                is_jump_target=last_node_i.is_jump_target,
            )
            jump_node = Node(i=jump_ins, inputs=[], outputs=[])
            jump_node.jump_targets = [((0, 0), ret_bl)]
            ret_bl.jump_sources.append(jump_node)
            # TODO: this should really be a method of PhiValue!
            ret_input.add_missing_value(b.nodes[-1].inputs[0], jump_source=jump_node)
            assert len(b.nodes[-1].inputs) == 1
            assert len(b.block_outputs) == 0
            b.block_outputs = {b.nodes[-1].inputs[0]}
            del b.nodes[-1]
            b.nodes.append(jump_node)
    return gr
