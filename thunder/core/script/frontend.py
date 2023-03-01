import collections
import dis
import inspect
import itertools
import opcode
import sys
from typing import Callable, Dict, List, Optional

import torch

from thunder.core.script.graph import (
    assert_value,
    assert_node,
    assert_block,
    Block,
    Graph,
    MROAwareObjectRef,
    Node,
    PhiValue,
    Value,
)
from thunder.core.script.python_ir_data import jump_instructions, stack_effect_detail, unconditional_jump_names
from thunder.core.utils import OrderedSet


class Super:
    pass


def acquire_method(
    method: Callable, module: Optional[object] = None, mro_klass: Optional[type] = None, verbose: bool = False
) -> Graph:
    if isinstance(method, torch.nn.Module):
        method = method.forward
    assert sys.version_info >= (3, 9) and sys.version_info < (3, 11)
    source_lines, source_start_line = inspect.getsourcelines(method)
    if verbose:
        print("".join(source_lines))
    sig = inspect.signature(method)
    if module is None and hasattr(method, "__self__"):
        module = method.__self__
    if mro_klass is None and module is not None:
        mro_klass = type(module)
    local_variables: List[Optional[Value]] = []
    if inspect.ismethod(method):
        self_value = Value(value=module, name=method.__code__.co_varnames[0], is_function_arg=True)
        local_variables.append(self_value)
        self_offset = 1
    else:
        self_value = None
        self_offset = 0
    for vname in method.__code__.co_varnames[self_offset : len(sig.parameters.values()) + self_offset]:
        p = sig.parameters[vname]
        local_variables.append(Value(typ=p.annotation, name=p.name, is_function_arg=True))
    # KWARGS?!
    for _ in enumerate(method.__code__.co_varnames, start=len(local_variables)):
        local_variables.append(None)

    # bound_args = [module.forward.__self__]
    bc = list(dis.get_instructions(method))
    if verbose:
        dis.dis(method)
    # Map offset_start -> Block
    block_0 = Block(is_ssa=False)
    block_0.jump_sources.append(None)
    blocks_to_process = collections.OrderedDict({0: block_0})
    blocks: Dict[int, Block] = {}

    def get_or_make_block(offset_start: int, jump_source: Node) -> Block:
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

    line_no = 1
    while blocks_to_process:
        offset_start, bl = blocks_to_process.popitem(last=False)
        blocks[offset_start] = bl

        ic = offset_start
        done = False
        while not done:
            cur_ins = bc[ic]
            if cur_ins.starts_line is not None:
                line_no = cur_ins.starts_line - source_start_line + 1
            n = Node(i=cur_ins, line_no=line_no)

            # need to handle branching instructions here
            if cur_ins.opcode in jump_instructions:
                done = True
                if cur_ins.opcode in dis.hasjabs:
                    ic_target = cur_ins.arg
                elif "BACKWARD" in cur_ins.opname:
                    assert cur_ins.arg is not None
                    ic_target = ic + 1 - cur_ins.arg  # ?
                else:
                    assert cur_ins.arg is not None
                    ic_target = ic + 1 + cur_ins.arg

                if cur_ins.opname in unconditional_jump_names:
                    n.jump_targets = []
                else:
                    b1 = get_or_make_block(offset_start=ic + 1, jump_source=n)
                    n.jump_targets = [(stack_effect_detail(cur_ins.opname, cur_ins.arg, jump=False), b1)]

                assert ic_target is not None
                b1 = get_or_make_block(offset_start=ic_target, jump_source=n)
                n.jump_targets.append((stack_effect_detail(cur_ins.opname, cur_ins.arg, jump=True), b1))
            elif cur_ins.opname in {"RETURN_VALUE", "RAISE_VARARGS", "RERAISE"}:
                done = True
            else:
                if verbose:
                    print(cur_ins)

            bl.insert_node(n)
            ic += 1
            if ic < len(bc) and bc[ic].is_jump_target:
                # check if needed?
                if (
                    cur_ins.opname not in {"RETURN_VALUE", "RAISE_VARARGS", "RERAISE"}
                    and cur_ins.opcode not in jump_instructions
                ):
                    # should insert jump absolute instead...
                    jump_ins = dis.Instruction(
                        opname="JUMP_ABSOLUTE",
                        opcode=-1,  # the JUMP_ABSOLUTE is not in Python 3.11
                        arg=None,
                        argval=None,
                        argrepr="None",
                        offset=-999,
                        starts_line=None,
                        is_jump_target=False,
                    )
                    jump_node = Node(i=jump_ins, inputs=[], outputs=[], line_no=line_no)
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
    gr.all_local_variables_at_start = local_variables[:]
    gr.local_variables_at_start = [lv for lv in local_variables if lv is not None]
    gr.ismethod = inspect.ismethod(method)
    gr.co_argcount = 0 if not gr.ismethod else 1
    # deal with other flags?
    # NESTED, GENERATOR, NOFREE, COROUTINE, ITERABLE_COROUTINE, ASYNC_GENERATOR
    gr.co_flags = inspect.CO_OPTIMIZED | inspect.CO_NEWLOCALS
    gr.co_posonlyargcount = 0
    gr.co_kwonlyargcount = 0
    gr.func_defaults = []
    gr.func_kwdefaults = {}
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            gr.co_argcount += 1
            gr.co_posonlyargcount += 1
        elif p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            gr.co_argcount += 1
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            gr.co_kwonlyargcount += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            gr.co_flags |= inspect.CO_VARARGS
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            gr.co_flags |= inspect.CO_VARKEYWORDS
        else:
            assert False, f"unknown parameter kind {p.kind}"

        if p.default is not inspect._empty:
            if p.kind == inspect.Parameter.KEYWORD_ONLY:
                gr.func_kwdefaults[p.name] = p.default
            else:
                gr.func_defaults.append(p.default)
    gr.method = method
    gr.module = module
    gr.mro_klass = mro_klass
    gr.self_value = self_value
    gr.source_start_line = 1  # source_start_line
    gr.source_lines = source_lines
    return gr


def make_ssa(gr: "Graph", verbose: bool = False) -> None:
    for bl in gr.blocks:
        for n in bl.nodes:
            n.block = bl
        bl.all_stacks_at_start = [None if js is not None else [] for js in bl.jump_sources]
        bl.all_local_variables_at_start = [
            None if js is not None else gr.all_local_variables_at_start[:] for js in bl.jump_sources
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
                any_deps_done = any(assert_block(assert_node(js).block) not in blocks_to_do for js in bl.jump_sources)
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

        stack: List[Value] = [PhiValue(v, jump_sources, bl) for v in zip(*all_stacks_at_start)]
        local_variables: List[Optional[Value]] = [
            PhiValue(v, jump_sources, bl) for v in zip(*all_local_variables_at_start)
        ]

        bl.block_inputs = stack + [assert_value(v) for v in local_variables]
        bl.stack_depth_at_start = len(stack)

        new_nodes = []
        for n_idx, n in enumerate(bl.nodes):
            cur_ins = n.i
            pop, push = stack_effect_detail(cur_ins.opname, cur_ins.arg)  # jump?
            inputs = stack[-pop:] if pop > 0 else []
            n.inputs = inputs[:]
            assert len(inputs) == pop, f"stack to shallow {len(inputs)=} {pop=} {cur_ins=}"
            if cur_ins.opname == "LOAD_FAST":
                assert cur_ins.arg is not None
                outputs: List[Value] = [assert_value(local_variables[cur_ins.arg])]
            elif cur_ins.opname == "STORE_FAST":
                outputs = []
                assert cur_ins.arg is not None
                if len(local_variables) <= cur_ins.arg:
                    local_variables.extend(None for _ in range(len(local_variables), cur_ins.arg + 1))
                (local_variables[cur_ins.arg],) = inputs  # set name?
            elif cur_ins.opname == "DELETE_FAST":
                outputs = []
                assert isinstance(cur_ins.arg, int)
                local_variables[cur_ins.arg] = None
            elif cur_ins.opname == "LOAD_GLOBAL":
                if gr.method.__code__.co_names[cur_ins.arg] != "super":
                    if inspect.ismethod(gr.method):
                        func = gr.method.__func__
                    else:
                        func = gr.method
                    gn = gr.method.__code__.co_names[cur_ins.arg]
                    NOT = object()
                    gv = func.__globals__.get(gn, NOT)
                    if gv is NOT:
                        gv = func.__builtins__[gn]
                    outputs = [Value(name=gn, value=gv, is_global=True)]
                else:
                    outputs = [Value(name="super", value=Super())]
            elif cur_ins.opname == "LOAD_ATTR":
                an = gr.method.__code__.co_names[cur_ins.arg]
                ap = inputs[0]
                outputs = [Value(name=an, parent=ap)]
            elif cur_ins.opname == "CALL_FUNCTION" and cur_ins.arg == 0 and isinstance(inputs[0].value, Super):
                outputs = [Value(value=MROAwareObjectRef(gr.self_value, start_klass=gr.mro_klass))]
                print("##super#", outputs)
            elif cur_ins.opname == "LOAD_METHOD":  # also used for modules (callables)
                (obj,) = inputs
                mn = gr.method.__code__.co_names[cur_ins.arg]
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
            elif cur_ins.opname == "LOAD_CONST":
                outputs = [Value(value=gr.method.__code__.co_consts[cur_ins.arg], is_const=True)]
            elif cur_ins.opname == "CALL_METHOD":
                outputs = [Value(node=n, nr=k) for k in range(push)]
                new_nodes.append(n)
            elif cur_ins.opname == "FOR_ITER":
                # JUMP TARGETS
                outputs = [inputs[0], Value(node=n, name=".for_iter_item")]
                new_nodes.append(n)
            elif cur_ins.opname in {
                "POP_JUMP_IF_FALSE",
                "POP_JUMP_IF_TRUE",
                "JUMP_FORWARD",
                "JUMP_ABSOLUTE",
            }:
                new_nodes.append(n)
                outputs = []
            # elif cur_ins.opname == "JUMP_FORWARD":
            # elif cur_ins.opname == "JUMP_ABSOLUTE":
            elif cur_ins.opname == "RETURN_VALUE":
                assert len(stack) == 1
                new_nodes.append(n)
                outputs = []
            else:
                if verbose:
                    print("unhandled", cur_ins)
                outputs = [Value(node=n, nr=k) for k in range(push)]
                new_nodes.append(n)
            if n.jump_targets is not None:
                all_block_outputs = OrderedSet(local_variables)
                for (j_pop, j_push), jt in n.jump_targets:
                    idx_jt = jt.jump_sources.index(n)
                    j_stack = stack[:]
                    if j_pop > 0:
                        j_stack = j_stack[:-j_pop]
                    if j_push > 0:
                        # TODO: change to use output_jump / output_nojump or somesuch
                        if len(outputs) < j_push:
                            j_stack.extend([Value(node=n, nr=k) for k in range(j_push)])
                        else:
                            j_stack.extend(outputs[:j_push])
                    assert len(j_stack) == len(stack) + j_push - j_pop
                    jt.all_stacks_at_start[idx_jt] = j_stack
                    jt.all_local_variables_at_start[idx_jt] = local_variables[:]
                    all_block_outputs.update(j_stack)
                bl.block_outputs = all_block_outputs
            n.outputs = outputs
            ol = len(stack)
            if pop > 0:
                stack = stack[:-pop]
            stack.extend(outputs)
            assert (cur_ins.opname == "JUMP_ABSOLUTE" and cur_ins.arg is None and len(stack) == ol) or (
                len(stack) - ol == opcode.stack_effect(cur_ins.opcode, cur_ins.arg, jump=False)
            ), f"stack effect funnyness at {cur_ins}: {len(stack)} {ol} {opcode.stack_effect(cur_ins.opcode, cur_ins.arg, jump=False)}"
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

    del gr.all_local_variables_at_start
    for bl in gr.blocks:
        del bl.all_local_variables_at_start
        del bl.all_stacks_at_start
        del bl.stack_depth_at_start
        del bl.jump_source_idxes_to_postprocess

    remove_unused_values(gr)


def remove_unused_values(gr: Graph) -> None:
    gr.ensure_links()

    def remove_value(v: Value) -> None:
        for pv in v.phi_values:
            bl = pv.block
            pv.remove_value(v)
            if not pv.values:
                remove_value(pv)
                bl.block_inputs.remove(pv)
                if pv in bl.block_outputs:
                    bl.block_outputs.remove(pv)

    for i in gr.blocks[0].block_inputs:
        if len(i.values) == 1 and i.values[0] is None:
            remove_value(i)

    gr.blocks[0].block_inputs = [i for i in gr.blocks[0].block_inputs if len(i.values) != 1 or i.values[0] is not None]

    values_used = set()

    INDEX_OPS = {"BINARY_SUBSCR"}

    def mark_used(v: Value) -> None:
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
        bl.block_outputs = OrderedSet(o for o in bl.block_outputs if o in values_used)
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
        bl.block_outputs = OrderedSet(o for o in bl.block_outputs if o in outputs_used)


def make_single_return(gr: Graph) -> None:
    bls = [b for b in gr.blocks if b.nodes[-1].i.opname == "RETURN_VALUE"]
    if len(bls) > 1:
        ret_bl = Block(is_ssa=True)
        ret_ins = dis.Instruction(
            opname="RETURN_VALUE",
            opcode=opcode.opmap["RETURN_VALUE"],
            arg=None,
            argval=None,
            argrepr="None",
            offset=-999,
            starts_line=None,
            is_jump_target=False,
        )
        ret_input = PhiValue([], [], ret_bl)
        ret_node = Node(i=ret_ins, inputs=[ret_input], outputs=[], line_no=bls[-1].nodes[-1].line_no)
        ret_bl.nodes = [ret_node]
        ret_bl.jump_sources = []
        ret_node.inputs = [ret_input]
        gr.blocks.append(ret_bl)
        ret_bl.block_outputs = OrderedSet([])
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
                argrepr="None",
                offset=last_node_i.offset,
                starts_line=None,
                is_jump_target=last_node_i.is_jump_target,
            )
            jump_node = Node(i=jump_ins, inputs=[], outputs=[], line_no=b.nodes[-1].line_no)
            jump_node.jump_targets = [((0, 0), ret_bl)]
            ret_bl.jump_sources.append(jump_node)
            # TODO: this should really be a method of PhiValue!
            ret_input.add_missing_value(b.nodes[-1].inputs[0], jump_source=jump_node)
            assert len(b.nodes[-1].inputs) == 1
            assert len(b.block_outputs) == 0
            b.block_outputs = OrderedSet([b.nodes[-1].inputs[0]])
            del b.nodes[-1]
            b.nodes.append(jump_node)
