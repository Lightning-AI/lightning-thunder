import dis
import copy
import inspect
import opcode
import sys
import types
from typing import Any, Callable, Dict, List, Tuple, Union
from contextvars import ContextVar

import networkx as nx
import torch  # # aehem.

import thunder
from thunder.core.script.frontend import acquire_method, remove_unused_values
from thunder.core.script.graph import (
    assert_block,
    assert_node,
    assert_value,
    Graph,
    Block,
    clone_blocks,
    _generate_raises,
    GraphObject,
    Node,
    PhiValue,
    replace_values,
    SourceInformation,
    _Undefined,
    Value,
    repr_source_location,
)
from thunder.core.script.instrumentation import verbose_error, record
from thunder.core.script.python_ir_data import get_instruction, ThunderInstruction, JUMP_ABSOLUTE, X_THUNDER_STORE_ATTR
from thunder.torch import _torch_to_thunder_complete_map
from thunder.core.script.noinline import NOINLINE_METHODS
from thunder.core.utils import debug_asserts_enabled, OrderedSet

MAX_INLINE_ITERS = 50


def split_block(gr: "Graph", bl: "Block", n: "Node") -> Block:
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
    nbl = Block()
    nbl.nodes = bl.nodes[j:]
    del bl.nodes[j:]
    nbl.block_outputs = bl.block_outputs
    bl.block_outputs = OrderedSet()
    nbl.block_inputs = []

    bl_jump_node = Node(i=ThunderInstruction.make_jump_absolute(arg=None), inputs=[], outputs=[])
    bl_jump_node.jump_targets = [nbl]
    if bl.nodes:
        bl_jump_node.source_infos = copy.deepcopy(bl.nodes[-1].source_infos)
    else:
        bl_jump_node.source_infos = copy.deepcopy(nbl.nodes[0].source_infos)
    bl.nodes.append(bl_jump_node)
    nbl.jump_sources.append(bl_jump_node)
    gr.blocks.insert(i + 1, nbl)

    potential_bl_outputs = {i for i in bl.block_inputs}
    for n in bl.nodes:
        for o in n.outputs:
            potential_bl_outputs.add(o)
    for i in bl.block_inputs:
        potential_bl_outputs.add(i)
    value_map: dict[GraphObject, GraphObject] = {}

    def get_or_create_phi(v: Value) -> Value:
        if v in value_map:
            return assert_value(value_map[v])
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
                n.outputs[idx_o] = get_or_create_phi(o)
                bl.block_outputs.add(o)

    bl.block_outputs.update(nbl.block_outputs & potential_bl_outputs)
    nbl.block_outputs = OrderedSet(
        (get_or_create_phi(o) if o in potential_bl_outputs else o) for o in nbl.block_outputs
    )

    return nbl


@verbose_error
def find_method_through_phi_parent(fn_value: Value) -> tuple[Value, list[str]]:
    Point = tuple[Value, tuple[str, ...]]
    to_process: list[Point] = [(v, ()) for v in fn_value.resolve()]
    edges: OrderedSet[tuple[Point, Point]] = OrderedSet(((fn_value, ()), i) for i in to_process)
    while to_process:
        v, attr = to_process.pop()
        destination = (v, attr)
        if (parent := v.parent) is not None and (name := v.name) is not None:
            destination = (parent, (name, *attr))

        elif (node := v.node) is not None and node.i.opname == "BINARY_SUBSCR" and node.inputs[1].is_const:
            destination = (node.inputs[0], (repr(node.inputs[1].value), *attr))

        for vi in destination[0].resolve():
            edge = ((v, attr), (vi, destination[1]))
            if edge not in edges:
                edges.add(edge)
                to_process.append(edge[1])

    G = nx.from_edgelist(edges, nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))
    assert nx.is_connected(G.to_undirected())
    assert nx.is_directed_acyclic_graph(G)

    # A size one topological generation means all flow must pass through that node. Thus, the latest
    # generation with that property is the farthest we can resolve attributes.
    *_, (fn_value, attr_lookups) = (i for i, *other in nx.topological_generations(G) if not other)
    return fn_value, list(attr_lookups)


def find_and_evaluate_method_through_phi_parent(v: Value) -> Union[object, Callable]:
    fn_parent_value, attr_lookups = find_method_through_phi_parent(v)
    if fn_parent_value.value is None:
        return None
    fn_value = fn_parent_value.value
    for al in attr_lookups:
        value = getattr(fn_value, al, _Undefined)
        if value is _Undefined:
            return _Undefined(fn_value, al)
        fn_value = value
    return fn_value


class SkipInlineError(NotImplementedError):
    pass


@record(delegate_to="n")
def inline_method_call(gr: "Graph", n: "Node") -> None:
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
        fn_value: Callable = find_and_evaluate_method_through_phi_parent(n.inputs[0])  # type: ignore
        assert not isinstance(fn_value, _Undefined)
        if fn_value is None:
            raise NotImplementedError("cannot inline non-explicit function")

        ## TODO: value for self arg in Method calls?
        ### in general: What is with callables here?
        if isinstance(fn_value, torch.nn.Module):
            mod1: object = fn_value
            value_for_self1 = n.inputs[0]
            fn_value = fn_value.forward
        elif isinstance(fn_value, types.MethodType):
            mod1 = fn_value.__self__
            value_for_self1 = n.inputs[1]
        else:
            mod1 = None
            value_for_self1 = None

        if inspect.isbuiltin(fn_value):
            raise NotImplementedError("cannot inline built-in (C-implemented) function")
    elif n.i.opname in {"CALL_FUNCTION", "CALL_FUNCTION_KW"}:
        fn_value = find_and_evaluate_method_through_phi_parent(n.inputs[0])  # type: ignore
        assert not isinstance(fn_value, _Undefined)
        if fn_value is None:
            raise NotImplementedError("cannot inline non-explicit function")

        if isinstance(fn_value, torch.nn.Module):
            mod1 = fn_value
            value_for_self1 = n.inputs[0]
            fn_value = fn_value.forward
        else:
            if isinstance(fn_value, types.FunctionType):
                mod1 = None
                value_for_self1 = None
            elif isinstance(fn_value, types.MethodType):
                mod1 = fn_value.__self__
                value_for_self1 = n.inputs[1]
            else:
                source_str = repr_source_location(gr, n.source_infos)
                raise NotImplementedError(f"inlining {fn_value} in instruction {n} at\n{source_str}")
    else:
        raise NotImplementedError(f"inlining {n}")

    # splitting must be done before replacing values, but this is changed even if we don't inline...
    nbl = split_block(gr, bl, bl.nodes[i_n + 1])

    gr1 = acquire_method(fn_value, module=mod1, mro_klass=gr.mro_klass if mod1 == gr.module else None)
    for gr1_n in gr1.nodes():
        assert gr1_n.source_infos
        have_generated = False
        for si in gr1_n.source_infos:
            si.gen_line_no = si.gen_line_no + len(gr.source_lines) + 1
            si.gen_end_line_no = si.gen_end_line_no + len(gr.source_lines) + 1
        # prepend
        gr1_n.source_infos[:0] = copy.deepcopy(n.source_infos)
    gr.source_lines.append("\n")
    gr.source_lines += gr1.source_lines

    if gr1.ismethod:
        sig1 = inspect.signature(gr1.method.__func__)
    else:
        sig1 = inspect.signature(gr1.method)
    # transform defaults
    sig1 = sig1.replace(
        parameters=[
            p
            if p.default is inspect._empty
            else p.replace(default=Value(name=p.name, typ=type(p.default), value=p.default, is_const=True))
            for p in sig1.parameters.values()
        ]
    )

    if gr1.ismethod:
        call_args = [value_for_self1]
    else:
        call_args = []

    if n.i.opname == "CALL_METHOD":
        call_args += n.inputs[2:]
        call_kwargs: dict[str, Any] = {}
    elif n.i.opname == "CALL_FUNCTION":
        call_args += n.inputs[1:]
        call_kwargs = {}
    elif n.i.opname == "CALL_FUNCTION_KW":
        assert n.inputs[-1].is_const
        num_kwargs = len(n.inputs[-1].value)
        call_kwargs = {k: v for k, v in zip(n.inputs[-1].value, n.inputs[-1 - num_kwargs : -1])}
        call_args += n.inputs[1 : -1 - num_kwargs]
    else:
        raise NotImplementedError()

    # TODO: catch and translate error messages, check types(?)
    bound_args = sig1.bind(*call_args, **call_kwargs)
    bound_args.apply_defaults()

    gr1_varargs = [n for n, p in sig1.parameters.items() if p.kind == p.kind.VAR_POSITIONAL]
    gr1_varkwargs = [n for n, p in sig1.parameters.items() if p.kind == p.kind.VAR_KEYWORD]
    ## TODO: TRANSLATE args (=tuple of Values) and kwargs (=dict str->Value) to a Value to something Value of ... (probably needs at least BUILD_TUPLE etc)
    if gr1_varargs or gr1_varkwargs:
        raise SkipInlineError("varargs and kwargs are currently not implemented")

    n1 = bl.nodes.pop(i_n)
    assert n1 is n

    # there should be exactly one
    (ret_bl,) = (bl for bl in gr1.blocks if len(bl.nodes) > 0 and bl.nodes[-1].i.opname == "RETURN_VALUE")

    ret_node = ret_bl.nodes[-1]
    ret_node.i = ThunderInstruction.make(
        JUMP_ABSOLUTE,
        arg=-1,
        argrepr="None",
        offset=ret_node.i.offset,
        starts_line=ret_node.i.starts_line,
        is_jump_target=ret_node.i.is_jump_target,
    )
    bl.nodes[-1].jump_targets = [gr1.blocks[0]]
    gr1.blocks[0].jump_sources = [bl.nodes[-1]]
    ret_node.jump_targets = [nbl]
    nbl.jump_sources = [ret_node if js == bl.nodes[-1] else js for js in nbl.jump_sources]

    gr.blocks[i_bl + 1 : i_bl + 1] = gr1.blocks

    assert len(n.outputs) == 1
    inp_map = {p: bound_args.arguments[p.name] for p in gr1.local_variables_at_start if p.name in bound_args.arguments}
    bl.block_outputs.remove(n.outputs[0])  # TODO: what with inplace!!
    bl.block_outputs.update(inp_map.values())  # Note: This includes default args
    replace_values(gr1, inp_map)

    # output value
    rv = ret_node.inputs.pop()
    assert not ret_node.inputs
    (orv,) = n.outputs
    replace_values(gr, {orv: rv})
    ret_bl.block_outputs.add(rv)


def inline_submodule_calls(gr: "Graph") -> bool:
    # inlines submodule calls
    # returns whether something has changed
    # TODO: recursively and not from nested structures (ModuleList etc.)
    changed = False
    gr.ensure_links()
    for bl in gr.blocks[:]:
        for n in bl.nodes[:]:
            if n.i.opname in {"CALL_METHOD", "CALL_FUNCTION", "CALL_FUNCTION_KW"}:
                fn_value = find_and_evaluate_method_through_phi_parent(n.inputs[0])
                if isinstance(fn_value, _Undefined):
                    # TODO: We could insert a RAISE here if we then delete the return
                    # value and all (direct or indirect) uses.
                    methval = Value(
                        value=_generate_raises(
                            f"attribute error '{type(fn_value.value)}' object has no attribute '{fn_value.attr}'"
                        ),
                        is_const=True,
                    )
                    n.i = n.i.modify_copy(opname="CALL_FUNCTION", arg=0)
                    n.inputs = [methval]
                if isinstance(fn_value, torch.nn.Module) or (
                    inspect.ismethod(fn_value)
                    and isinstance(fn_value.__self__, torch.nn.Module)
                    and (fn_value not in NOINLINE_METHODS.get())
                ):
                    inline_method_call(gr, n)
                    changed = True

    return changed


def strongly_inline_functions(gr: "Graph") -> None:
    for _ in range(MAX_INLINE_ITERS):
        loop = False
        gr.ensure_links()
        for bl in gr.blocks[:]:
            for n in bl.nodes[:]:
                if n.i.opname in {"CALL_METHOD", "CALL_FUNCTION", "CALL_FUNCTION_KW"}:
                    fn_value = find_and_evaluate_method_through_phi_parent(n.inputs[0])
                    if (
                        fn_value is not None
                        and not inspect.isbuiltin(fn_value)
                        and isinstance(fn_value, types.FunctionType)
                        and fn_value not in _torch_to_thunder_complete_map
                        and fn_value not in NOINLINE_METHODS.get()
                    ):
                        ## handle methods or nn.Modules / other classes?
                        try:
                            inline_method_call(gr, n)
                            loop = True
                        except SkipInlineError:
                            pass
                        except RuntimeError as e:
                            (msg,) = e.args
                            source_str = repr_source_location(gr, n.source_infos)
                            msg = f"{msg}\nwhile inlining:\n{source_str}"
                            e.args = (msg,)
                            raise e
        if not loop:
            return

    raise AssertionError(f"Inlining did not complete after {MAX_INLINE_ITERS} passes.")


def torch_to_thunder(gr: "Graph", fallback: bool = False) -> None:
    """replaces calls to torch.foo functions with calls into thunder's torch language."""

    def fill_in_value(v: Value, seen: OrderedSet[Value]) -> None:
        if v in seen:
            return
        seen.add(v)
        parent = v.parent
        if parent is None and isinstance(v, PhiValue):
            for vv in v.values:
                fill_in_value(vv, seen)
            for vv in v.values[1:]:
                if vv.value is not v.values[0].value:
                    return
            v.value = v.values[0].value
        if v.value is None and parent is not None:
            fill_in_value(parent, seen)
        if v.name is None and isinstance(v, PhiValue) and parent is not None and parent.name is not None:
            v.name = parent.name
        if v.value is None and parent is not None and parent.value is not None and v.name is not None:
            v.value = getattr(parent.value, v.name, None)

    for bl in gr.blocks:
        for n in bl.nodes:
            for idx, i in enumerate(n.inputs):
                done = False
                fill_in_value(i, OrderedSet())
                i_or_parent = i
                while i_or_parent.value not in _torch_to_thunder_complete_map and i_or_parent.parent is not None:
                    i_or_parent = i_or_parent.parent

                if i_or_parent.value in _torch_to_thunder_complete_map:
                    i_or_parent.value = _torch_to_thunder_complete_map[i.value]
                    # we reinstantiate because we don't want a PhiValue here
                    i_new = Value(
                        value=i_or_parent.value,
                        typ=type(i_or_parent.value),
                        parent=None,
                        is_const=True,
                        is_global=False,
                        name=i_or_parent.name,
                    )
                    n.inputs[idx] = i_new
                    if n.i.opname == "CALL_METHOD" and idx == 0:
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
                        assert i.name is not None
                        i.value = getattr(thunder.langs.torch, i.name)

                    # replace other things by checking against torch module (make dict at startup?)
                    name = getattr(i.value, "__name__", None)
                    tf = None
                    if name is not None:
                        tf = getattr(torch, name, None)
                    if tf is not None and i.value == tf:
                        i.value = getattr(thunder.langs.torch, name)
                        i.is_global = False
                        i.is_const = True


def merge_two_blocks(gr: "Graph", bl1: "Block") -> None:
    jt = bl1.nodes[-1].jump_targets
    if len(jt) != 1:
        raise RuntimeError("can only fuse blocks with deterministic connection")
    bl2 = jt[0]
    if len(bl2.jump_sources) != 1 or bl2.jump_sources[0] != bl1.nodes[-1]:
        raise RuntimeError("second block to be fused must only have first block as jump source")

    replacements: dict[Value, Value] = {}
    for i in bl2.block_inputs:
        assert isinstance(i, PhiValue) and len(i.values) == 1, (i, getattr(i, "values", None))
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
    bl1.block_outputs = OrderedSet(o for o in bl1.block_outputs if o.phi_values)
    bl1.block_outputs.update(bl2.block_outputs)

    bl1.nodes[-1:] = bl2.nodes
    gr.blocks.remove(bl2)


def merge_blocks_where_possible(gr: "Graph") -> None:
    i_bl = 0
    while i_bl < len(gr.blocks):
        bl1 = gr.blocks[i_bl]
        jt = bl1.nodes[-1].jump_targets
        if len(jt) == 1:
            bl2 = jt[0]
        else:
            bl2 = None
        if bl2 is not None and len(bl2.jump_sources) == 1 and bl2.jump_sources[0] == bl1.nodes[-1]:
            merge_two_blocks(gr, bl1)
        else:
            i_bl += 1


def find_blocks_of_for(gr: "Graph", for_block: "Block") -> list[Block]:
    assert for_block.nodes[-1].i.opname == "FOR_ITER"

    blocks_of_for_loop = OrderedSet({for_block})
    currently_looking_at = set()

    def find_blocks_of_for_rec(for_block: "Block", start_block: "Block") -> bool:
        if for_block == start_block:
            return True
        if start_block in currently_looking_at:
            return False
        currently_looking_at.add(start_block)
        found = False
        for jt in start_block.nodes[-1].jump_targets:
            found |= find_blocks_of_for_rec(for_block, jt)
        currently_looking_at.remove(start_block)
        if found:
            blocks_of_for_loop.add(start_block)
        return found

    find_blocks_of_for_rec(for_block, for_block.nodes[-1].jump_targets[0])
    return list(blocks_of_for_loop)


def unroll_for_over_modules(gr: "Graph", for_iter_node: "Node") -> None:
    gr.ensure_links()
    get_iter_node = for_iter_node.inputs[0].values[0].node
    assert get_iter_node.i.opname == "GET_ITER"

    iterated_module_list_parent, attr_lookups = find_method_through_phi_parent(get_iter_node.inputs[0])
    assert iterated_module_list_parent.value is not None
    iterated_module_list = iterated_module_list_parent.value
    for al in attr_lookups:
        iterated_module_list = getattr(iterated_module_list, al)

    # what about more complex things?
    assert isinstance(iterated_module_list, (torch.nn.Sequential, torch.nn.ModuleList))

    for_loop_len = len(iterated_module_list)
    for_iter_block = for_iter_node.block
    assert for_iter_block is not None
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

    def delete_value_and_sources(v: Value) -> None:
        # check that it is possible?
        if v in seen:
            return
        seen.add(v)
        if isinstance(v, PhiValue):
            for vv, js in zip(v.values, v.jump_sources):
                delete_value_and_sources(vv)
                assert js is not None and js.block is not None
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
    for_iter_block_jmp.source_infos = copy.deepcopy(for_iter_node.source_infos)
    for_iter_block.nodes.append(for_iter_block_jmp)
    for_iter_block_jmp.jump_targets = [for_iter_node.jump_targets[0]]
    for_iter_node_exit_jump_target = for_iter_node.jump_targets[1]
    for_iter_node.jump_targets = []
    for_iter_block_jmp.jump_targets[0].jump_sources = [
        (js if js is not for_iter_node else for_iter_block_jmp)
        for js in for_iter_block_jmp.jump_targets[0].jump_sources
    ]

    exit_block = Block()
    gr.blocks.append(exit_block)
    exit_node = Node(i=get_instruction(opname="JUMP_ABSOLUTE", arg=None))
    exit_node.source_infos = copy.deepcopy(for_iter_node.source_infos)
    exit_node.jump_targets = [for_iter_node_exit_jump_target]
    target_after_iter = exit_node.jump_targets[0]
    exit_node.jump_targets[0].jump_sources = [
        (js if js is not for_iter_node else exit_node) for js in exit_node.jump_targets[0].jump_sources
    ]
    exit_block.nodes.append(exit_node)
    for i in for_iter_block.block_inputs:
        exit_block.block_inputs.append(PhiValue([], [], exit_block))

    unroll_blocks: list[tuple[list[Block], dict[GraphObject, GraphObject]]] = [(list(bls), {})]
    unroll_blocks += [clone_blocks(bls) for _ in range(1, for_loop_len)]
    for idx, (nbls, td) in enumerate(unroll_blocks):
        if idx > 0:
            gr.blocks += nbls
            v_idx = Value(value=idx, is_const=True)
            assert_node(td[for_iter_node]).inputs[1] = v_idx
            fin_o = assert_node(td[for_iter_node]).outputs[0]
            assert fin_o.name is not None
            fin_o.name += f"_{idx}"
        else:
            assert for_iter_node.outputs[0].name is not None
            for_iter_node.outputs[0].name += "_0"

    gr.ensure_links()

    fixup_data = []
    for idx, (nbls, td) in enumerate(unroll_blocks):
        if idx == 0:
            fib_i = for_iter_block
            jump_sources_to_fix = [js for js in for_iter_block.jump_sources if js is not get_iter_block.nodes[-1]]
        else:
            fib_i = assert_block(td[for_iter_block])
            jump_sources_to_fix = fib_i.jump_sources[:]
        if idx + 1 < len(unroll_blocks):
            _, td_next = unroll_blocks[idx + 1]
            fib_next = assert_block(td_next[for_iter_block])
        else:
            fib_next = exit_block

        fixup_data.append((fib_i, jump_sources_to_fix, fib_next, nbls))

    for idx_it, (fib_i, jump_sources_to_fix, fib_next, nbls) in enumerate(fixup_data):
        for js in jump_sources_to_fix:
            assert js is not None
            for idx, jt in enumerate(js.jump_targets):
                if jt == fib_i:
                    js.set_jump_target(fib_next, idx=idx)

        for idx_i, i in enumerate(fib_i.block_inputs):
            if any((js.block in nbls) for js in i.jump_sources):
                ## if this is a variable updated in the loop:
                ##  - instead of looping back, point the update to the phi value of the next block (or the exit block)
                ##  - if idx > 0: remove external (before the loop) value
                for v, js in zip(i.values[:], i.jump_sources[:]):
                    assert js is not None and js.block is not None
                    if js.block not in nbls and idx_it > 0:
                        i.remove_value(v)

    for idx_it, (fib_i, jump_sources_to_fix, fib_next, nbls) in enumerate(fixup_data):
        for idx_i, i in enumerate(fib_i.block_inputs):
            if any((js.block in nbls) for js in i.jump_sources):
                for v, js in zip(i.values[:], i.jump_sources[:]):
                    if assert_block(assert_node(js).block) in nbls:
                        i.remove_value(v)
                        assert_block(fib_next).block_inputs[idx_i].add_missing_value(v, jump_source=js)
                if idx_it == 0:
                    for pv in i.phi_values[:]:
                        if pv.block is target_after_iter:
                            pv.remove_value(i)
                            pv.add_missing_value(exit_block.block_inputs[idx_i], jump_source=exit_node)

    for i in exit_block.block_inputs[:]:
        if i.phi_values:
            exit_block.block_outputs.add(i)
        else:
            assert isinstance(i, PhiValue)
            for v in i.values[:]:
                i.remove_value(v)
            exit_block.block_inputs.remove(i)


def find_and_unroll_for_loop(gr: "Graph") -> bool:
    if debug_asserts_enabled():
        thunder.core.script.graph.check_graph(gr)
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
                    ) = find_method_through_phi_parent(get_iter_node.inputs[0])
                    if iterated_module_list_parent.value is None:
                        continue
                    iterated_module_list = iterated_module_list_parent.value
                    for al in attr_lookups:
                        iterated_module_list = getattr(iterated_module_list, al)
                    # what about more complex things? in particular enumerate, but zip, ...
                    if isinstance(iterated_module_list, (torch.nn.Sequential, torch.nn.ModuleList)):
                        thunder.core.script.passes.unroll_for_over_modules(gr, for_iter_node)
                        if debug_asserts_enabled():
                            thunder.core.script.graph.check_graph(gr)
                        thunder.core.script.passes.merge_blocks_where_possible(gr)
                        if debug_asserts_enabled():
                            thunder.core.script.graph.check_graph(gr)
                        thunder.core.script.graph.check_graph(gr)
                        return True
    if debug_asserts_enabled():
        thunder.core.script.graph.check_graph(gr)
    return False


def unroll_for_loops_and_inline_modules(gr: "Graph") -> None:
    if debug_asserts_enabled():
        thunder.core.script.graph.check_graph(gr)
    iterate = True
    while iterate:
        iterate = find_and_unroll_for_loop(gr)
        if not iterate:
            iterate = inline_submodule_calls(gr)
            if iterate:
                thunder.core.script.passes.merge_blocks_where_possible(gr)


def module_to_function(gr: "Graph") -> tuple[list[str], list[torch.Tensor]]:
    attr_dict: dict[str, int] = {}
    attr_list: list[str] = []
    attr_values = []
    return_values: dict[str, Value] = {}  # PhiValues in the return block

    if debug_asserts_enabled():
        thunder.core.script.graph.check_graph(gr)

    def functionalize_value_if_possible(i):
        # TODO: inefficient because it looks twice
        v = find_and_evaluate_method_through_phi_parent(i)
        # assert not isinstance(v, _Undefined), f"undefined: {v.value} {v.attr}"
        if isinstance(v, _Undefined):
            return Value(value=v, is_const=True)
        maybe_self, attrs = find_method_through_phi_parent(i)

        attr_string = ".".join(attrs)
        if maybe_self.value is gr.module and (isinstance(v, torch.Tensor) or (attr_string in return_values)):
            # the new attributes come directly after the self argument
            idx = attr_dict.setdefault(attr_string, len(attr_list) + 1)
            if idx == len(attr_list) + 1:
                func_arg = Value(name=attr_string, is_function_arg=True)
                gr.local_variables_at_start.insert(idx, func_arg)
                attr_list.append(attr_string)
                attr_values.append(v)
                gr.co_argcount += 1
                # we need a default argument to be able to put the things at the end (but this will have to change for *args, **kwargs anyway...
                # gr.func_defaults.append(None)
                if attr_string in return_values:
                    return_values[attr_string].add_missing_value(func_arg)
            else:
                func_arg = gr.local_variables_at_start[idx]

            pvs = [pv for pv in func_arg.phi_values if pv.block is bl]
            if not pvs:
                pv = PhiValue([func_arg], [None], bl)
                bl.block_inputs.append(pv)
            else:
                (pv,) = pvs
            ## remove old input from phi_values etc?
            return pv
        if maybe_self.value is gr.module and (
            n.i.opname not in {"BINARY_SUBSCR"} and not isinstance(v, torch.nn.Module)
        ):
            ## inline to const...
            i.value = v
            i.typ = type(i.value)
            i.parent = None
            i.is_const = True
            i.is_global = False
            return None
        return None

    return_block = None
    for bl in gr.blocks:
        if bl.nodes[-1].i.opname == "RETURN_VALUE":
            assert return_block is None, "multiple return statements should not happen here"
            return_block = bl
    assert return_block is not None, "could not find return block"

    for bl in gr.blocks:
        for n in bl.nodes:
            if n.i.opname == "STORE_ATTR":
                v = find_and_evaluate_method_through_phi_parent(n.inputs[1])
                if isinstance(v, _Undefined):
                    n.inputs[1] = Value(value=v, is_const=True)
                    continue
                    # assert not isinstance(v, _Undefined), f"undefined: {v.value} {v.attr}"
                maybe_self, attrs = find_method_through_phi_parent(n.inputs[1])
                attrs.append(n.i.argval)
                if maybe_self.value is gr.module:
                    attr_string = ".".join(attrs)
                    n.i = n.i.modify_copy(opname=X_THUNDER_STORE_ATTR, opcode=None, argval=attr_string)
                    pv = return_values.get(attr_string)
                    if pv is None:
                        pv = PhiValue([], [], return_block)
                        pv.name = attr_string
                        return_values[attr_string] = pv
                        return_block.block_inputs.append(pv)
                    v = Value(node=n, name=attr_string)  # disambiguate?
                    pv.add_missing_value(v, jump_source=bl.nodes[-1])
                    n.outputs = [v]
                    bl.block_outputs.add(v)
                    del n.inputs[1]

    for bl in gr.blocks:
        for n in bl.nodes:
            if n.i.opname == "CALL_METHOD":
                if n.inputs[0].parent == n.inputs[1]:
                    v = find_and_evaluate_method_through_phi_parent(n.inputs[0])
                    if not isinstance(v, types.MethodType) or v.__self__ != find_and_evaluate_method_through_phi_parent(
                        n.inputs[1]
                    ):
                        # this case (not a proper method call is usually handled in executing the LOAD_METHOD opcode)
                        n.i = n.i.modify_copy(opname="CALL_FUNCTION", opcode=None)
                        del n.inputs[1]

            for idx_i, i in enumerate(n.inputs):
                v = functionalize_value_if_possible(i)
                if v is not None:
                    n.inputs[idx_i] = v

        bl.block_outputs = OrderedSet(
            [v if (v := functionalize_value_if_possible(o)) is not None else o for o in bl.block_outputs]
        )

    if return_values:
        bt_extra = Node(
            i=get_instruction(opname="BUILD_TUPLE", arg=1 + len(return_values)),
            source_infos=copy.deepcopy(return_block.nodes[-1].source_infos),
        )
        bt_extra.inputs = return_block.nodes[-1].inputs + list(return_values.values())
        v_tuple_extra = Value(node=bt_extra)
        bt_extra.outputs = [v_tuple_extra]
        return_block.nodes.insert(-1, bt_extra)
        return_block.nodes[-1].inputs = [v_tuple_extra]

    remove_unused_values(gr)
    if gr.local_variables_at_start[0].phi_values:
        gr.summary(print_lines=True)
        raise RuntimeError(
            """could not eliminate self argument
    this most likely means that you are setting attributes in forward or using them
    in an unexpected way that thunder does not yet support.
    The problem lies in (indirect) uses of V_0 in the graph above."""
        )

    # check to avoid assignments for both a.b and a.b.c
    sorted_keys = sorted(return_values.keys())  # this uses that '.' sorts before other things
    for i in range(len(sorted_keys) - 1):
        kbase = sorted_keys[i]
        knext = sorted_keys[i + 1]
        if knext.startswith(kbase) and knext[len(kbase)] == ".":
            # N.B. we know that knext is longer if kbase is a prefix so the knext[len(kbase)] above will not be out of bounds.
            raise RuntimeError(f"Assigning to members of assigned members ('{kbase}' and '{knext}') is not supported.")

    del gr.local_variables_at_start[0]
    gr.co_argcount -= 1
    if gr.co_posonlyargcount > 0:
        gr.co_posonlyargcount -= 1

    # thunder.core.script.graph.check_graph(gr)
    # gr.summary(print_lines=True)

    return attr_list, attr_values, list(return_values.keys())
