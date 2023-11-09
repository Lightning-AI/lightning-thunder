import collections
import functools
import dis
import inspect
import itertools
import sys
from typing import Optional, TypeVar
from collections.abc import Callable
from collections.abc import Iterable

import networkx as nx

from thunder.core.script.graph import (
    check_graph,
    replace_values,
    Block,
    Graph,
    MROAwareObjectRef,
    Node,
    NULL,
    PhiValue,
    SourceInformation,
    Value,
)
from thunder.core.script.instrumentation import record
from thunder.core.script import parse, values
from thunder.core.script.protograph import ProtoBlock, ProtoGraph
from thunder.core.script.protograph_passes import _get_missing_transitive, apply_protograph_passes, check_idempotent
from thunder.core.script.python_ir_data import get_instruction, SUPPORTS_PREPROCESSING
from thunder.core.utils import debug_asserts_enabled, OrderedSet

T = TypeVar("T")


class Super:
    pass


@check_idempotent
def _prune_epilogues(proto_graph: ProtoGraph) -> tuple[ProtoGraph, bool]:
    """Remove the `POP_TOP, ..., JUMP_ABSOLUTE` blocks introduced during parsing.

    NOTE: This is only for `_bind_to_graph`. The reason is that it produces a
          ProtoGraph with mismatched stacks. (Since we've pruned POP_TOP ops.)
          This isn't a problem since `_bind_to_graph` is value based, however
          it does make `_inter_block_edges` unsafe.
    """
    retain: dict[ProtoBlock, ProtoBlock] = {}
    for protoblock in proto_graph:
        instructions = tuple(i for i, _ in protoblock.flow.symbolic)
        if all(isinstance(i, parse.EpilogueFixup) for i in instructions):
            assert all(i.opname == parse.POP_TOP for i in instructions[:-1])
            assert instructions[-1].opname == parse.JUMP_ABSOLUTE, instructions[-1]
            continue

        retain[protoblock] = new_protoblock = ProtoBlock(protoblock.flow)
        new_protoblock.uses.update(protoblock.uses)

    for old, new in retain.items():
        for target, jump in old.jump_targets:
            if target not in retain:
                ((target, _),) = target.jump_targets
                assert target in retain
            new.add_jump_target(retain[target], jump)

    return ProtoGraph(retain.values()), len(retain) != len(tuple(proto_graph))


def _bind_to_graph(
    proto_graph: ProtoGraph,
    func: Callable,
    method_self: object | None = None,
    mro_klass: type | None = None,
) -> Graph:
    """Convert abstract value graph into a concrete Graph.

    The key nuance of this conversion is that the mapping from `AbstractValue`
    to `Value` is contextual. The first time we "see" an `AbstractValue` it
    maps to a `Value`. If we encounter it in any other block it maps to a
    PhiValue and we need to set the proper connectivity.

    This is perhaps clearer with an example. Suppose you have an argument `x`
    which is used by the root block and passed to the next block, and suppose
    you have another value `y` which is created in the root block and passed to
    the next block. In the abstract flow this is represented as:
           ________        ___________
    `x` -> | Root | -`x`-> | Block 1 | -> ...
           |  `y` | -`y`-> |         |
           --------        -----------

    On the other hand, `Graph` represents the same connectivity as:
                       ________                        ___________
    `x` ←┈┈→ `𝜙x_0` -> | Root | -`𝜙x_0` ←┈┈→ `𝜙x_1` -> | Block 1 | -> ...
                       |  `y` | -`y` ←┈┈┈┈┈→ `𝜙y_0` -> |         |
                       --------                        -----------

    (This diagram does not show the reason for PhiValues: to accept multiple inputs.)
    """
    # Peek at the signature and live objects to create Values. This is the
    # *only* region where this is permitted.
    # =========================================================================
    # TODO(robieta): Lazily generate specializations during runtime.
    signature = inspect.signature(func)
    func_globals = {**func.__builtins__, **func.__globals__, **{"super": Super()}}

    # NOTE:
    #   `inspect.signature` will expose parameters in intuitive order. However that
    #   is not necessarily how Python represents them internally. Specifically, varargs
    #   and varkwargs are moved to the end. This convention is load bearing (since it
    #   allows the interpreter index into a flat args array) so we must respect it
    #   here. (`func.__code__.co_varnames` is the canonical ordering.)
    arg_ordered_parameters = func.__code__.co_varnames[: len(signature.parameters)]
    source_file_name = inspect.getsourcefile(func)
    source_start_line = func.__code__.co_firstlineno
    if set(arg_ordered_parameters) != set(signature.parameters):
        assert hasattr(func, "__wrapped__")
        msg = f"({', '.join(arg_ordered_parameters)}) != ({', '.join(signature.parameters.keys())})"
        raise NotImplementedError(msg)

    co_name = func.__code__.co_name
    self_key: parse.VariableKey | None = None
    self_value: Value | None = None
    if method_self is not None:
        self_key = parse.VariableKey(arg_ordered_parameters[0], parse.VariableScope.LOCAL)
        self_value = Value(value=method_self, name=self_key.identifier, is_function_arg=True)

    get_initial_value_cache = {}

    def get_initial_value(key: parse.VariableKey, block: Block | None = None) -> Value:
        if key in get_initial_value_cache:
            v = get_initial_value_cache[key]
            assert not ((block is None or block != v.block) and not (v.is_global or v.is_const or v.is_function_arg))
            return v
        if key.is_const:
            v = Value(value=key.identifier, is_const=True)
            get_initial_value_cache[key] = v
            return v

        elif key == self_key:
            v = self_value
            get_initial_value_cache[key] = v
            return v

        name = key.identifier
        assert isinstance(name, str)
        if key.scope == parse.VariableScope.LOCAL:
            if (p := signature.parameters.get(name)) is not None:
                v = Value(typ=p.annotation, name=name, is_function_arg=True)
                get_initial_value_cache[key] = v
                return v
            v = Value(value=NULL, name=name, block=block)
            get_initial_value_cache[key] = v
            return v

        if key.scope == parse.VariableScope.NONLOCAL:
            msg = f"nonlocal variables are not supported but (key, name) = ({key}, {name}) found"
            raise RuntimeError(msg)

        if key.scope == parse.VariableScope.GLOBAL:
            try:
                val = func_globals[name]
            except KeyError:
                raise ValueError(f"Could not resolve global variable: {name=}.")
            v = Value(name=name, value=val, is_global=True)
            get_initial_value_cache[key] = v
            return v

        raise ValueError(f"Unhandled key: {key=}, name: {name=}")

    del func
    # End live inspection region.
    # =========================================================================
    assert not (missing_transitive := _get_missing_transitive(proto_graph)), missing_transitive
    proto_graph, _ = _prune_epilogues(proto_graph)
    blocks = {protoblock: Block() for protoblock in proto_graph}
    blocks[proto_graph.root].jump_sources.append(None)

    # Block inputs require special handling since we may need to create `PhiValue`s.
    input_conversions = {}
    for protoblock, block in blocks.items():
        for key, abstract_value in protoblock.begin_state:
            if protoblock is proto_graph.root:
                value = get_initial_value(key, block=block)
                if key.scope == parse.VariableScope.LOCAL and value.value is not NULL:
                    assert isinstance(abstract_value, values.ExternalRef), abstract_value
                    value = PhiValue([value], [None], block)

            elif key in protoblock.uses:
                value = PhiValue([], [], block)

            else:
                value = Value(value=NULL, block=block)

            input_conversions[(abstract_value, protoblock)] = value

    convert_cache = {}

    def convert(value: values.AbstractValue, protoblock: ProtoBlock, block: Block) -> Value:
        v = convert_cache.get((value, protoblock))
        if v is not None:
            if (
                v.block != block
                and block is not None
                and not (v.is_global or v.is_function_arg or v.is_const or v.value == NULL)
            ):
                raise AssertionError("ohoh, this should not happen")
            return v

        def _convert(value: values.AbstractValue, protoblock: ProtoBlock) -> Value:
            assert not value.is_detail, value
            if (out := input_conversions.get((value, protoblock), missing := object())) is not missing:
                return out

            if isinstance(value, values.NonPyObject):
                assert value.tag == values.NonPyObject.Tag.MISSING
                return Value(value=NULL)

            elif isinstance(value, (values.IntermediateValue, values.CompositeValue, values.AbstractPhiValue)):
                # For now we discard any information and just treat them as opaque.
                # TODO(robieta): refine
                return Value(block=block)

            elif isinstance(value, values.ExternalRef) and value.key.is_const:
                return get_initial_value(value.key, block=block)

            raise ValueError(f"Cannot convert abstract value: {value}, {protoblock} {protoblock is proto_graph.root=}")

        v = _convert(value, protoblock)
        convert_cache[(value, protoblock)] = v
        return v

    def make_nodes(protoblock: ProtoBlock, block: Block) -> Iterable[Node]:
        for instruction, node_flow in protoblock.node_flow:
            node = Node(
                i=instruction,
                inputs=[convert(v, protoblock, block) for v in node_flow.inputs],
                outputs=[convert(v, protoblock, block) for v in node_flow.outputs],
            )
            node.source_infos = [
                SourceInformation(
                    orig_file_name=source_file_name,
                    orig_line_no=instruction.line_no + source_start_line,
                    orig_end_line_no=instruction.line_no + source_start_line,
                    gen_line_no=instruction.line_no,
                    gen_end_line_no=instruction.line_no,
                    col_offset=0,
                    end_col_offset=999,
                ),
            ]

            for output in OrderedSet(node.outputs).difference(node.inputs):
                if not (output.node or output.is_const or output.is_global):
                    # output.node can be populated when we deconstruct a previously constructed value (e.g. binary_idx into a tuple from build_tuple)
                    output.node = node

            if node.i.opname in ("LOAD_ATTR", "LOAD_METHOD"):
                # Once we set `parent` (so PhiValue can traverse through it)
                # we can prune these just like all other load instructions.
                node.outputs[0].parent = node.inputs[0]
                node.outputs[0].name = node.i.argrepr
                continue

            elif node.i.opname == "CALL_FUNCTION":
                # Note: `super` handling is not currently generic. Corner cases
                #       such as `super(**{})` or `super_alias = super; super_alias()`
                #       will not be correctly handled.
                # TODO(robieta): handle `super` without load bearing names.
                if node.i.arg == 0 and isinstance(node.inputs[0].value, Super):
                    assert self_value is not None, "super() called in free context"
                    node.outputs[0].value = MROAwareObjectRef(self_value, start_klass=mro_klass)

            elif node.i.opname == "FOR_ITER":
                node.outputs[1].node = node
                node.outputs[1].name = ".for_item_iter"

            yield node

    # First pass: populate nodes and jump targets.
    for protoblock, block in blocks.items():
        block.nodes = list(make_nodes(protoblock, block))
        for target, _ in protoblock.jump_targets:
            jump_target = blocks[target]
            last_node = block.nodes[-1]
            jump_target.jump_sources.append(last_node)
            last_node.jump_targets.append(jump_target)

    # Second pass: link blocks.
    for protoblock, block in blocks.items():
        block_values = {
            k: v
            for k, abstract_v in protoblock.begin_state
            if isinstance(v := convert(abstract_v, protoblock, block), PhiValue)
        }

        block.block_inputs = list(OrderedSet(block_values.values()))
        for parent in proto_graph.parents[protoblock]:
            parent_state = dict(parent.end_state)
            for key, sink in block_values.items():
                source = convert(
                    parent_state.get(key, values.NonPyObject(values.NonPyObject.Tag.MISSING)), parent, block=None
                )
                if source.value is not NULL and source not in sink.values:
                    sink.add_missing_value(v=source, jump_source=blocks[parent].nodes[-1])

    # Third pass: specify block outputs once we know which Values are passed to another Block.
    for protoblock, block in blocks.items():
        outputs = (convert(abstract_value, protoblock, block) for k, abstract_value in protoblock.end_state)
        block.block_outputs.update(v for v in outputs if v.phi_values)

    param_keys = tuple(parse.VariableKey(p, parse.VariableScope.LOCAL) for p in arg_ordered_parameters)
    missing = {
        k: v
        for k in proto_graph.root.uses.difference(param_keys)
        if k.scope == parse.VariableScope.LOCAL and (v := get_initial_value(k)).value is not NULL
    }
    assert not missing, f"missing params {missing}"

    gr = Graph(list(blocks.values()))
    gr.local_variables_at_start = [get_initial_value(k) for k in param_keys]

    gr.co_name = co_name
    # bound_args = [module.forward.__self__]
    gr.self_value = self_value
    gr.ismethod = self_value is not None
    # deal with other flags?
    # NESTED, GENERATOR, NOFREE, COROUTINE, ITERABLE_COROUTINE, ASYNC_GENERATOR
    gr.co_flags = inspect.CO_OPTIMIZED | inspect.CO_NEWLOCALS
    gr.co_argcount = 0
    gr.co_posonlyargcount = 0
    gr.co_kwonlyargcount = 0
    gr.func_defaults = []
    gr.func_kwdefaults = {}
    for p in signature.parameters.values():
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
    return gr


def acquire_partial(
    pfunc: functools.partial,
    module: object | None = None,
    mro_klass: type | None = None,
) -> Graph:
    # This is complicated due to the semantics of calling Python functions.
    # The partial wrapper does the following:
    # def pfunc.__call__(*args, **kwargs):
    #    kw = pfunc.keywords.copy()
    #    kw.update(kwargs)
    #    return pfunc.func(*pfunc.args, *args, **kw)

    # This means:
    # - positional partial_args are applied from the front and once
    #   they are bound, they are removed from the signature,
    # - keyword only args get new defautls,
    # - binding a positional arg as a keyword arg effectively (i.e. in how
    #   it can be set in calls) makes that arg and all args to the right
    #   keyword only.
    # - things that cannot be bound to parameters may show up in varargs
    #   or kwargs parameters of the function.

    gr = acquire_method(pfunc.func, module, mro_klass)

    # first we shuffle positional args to kw only if they are in the kwargs of the partial
    pos_param_names = [v.name for v in gr.local_variables_at_start[: gr.co_argcount]]
    pos_param_names_to_idx = {n: i for i, n in enumerate(pos_param_names)}
    kw_pos_param_idx = [pos_param_names_to_idx[k] for k in pfunc.keywords if k in pos_param_names_to_idx]
    if kw_pos_param_idx:
        # convert positional default args to kw ones
        kw_pos_param_min = min(kw_pos_param_idx)
        if kw_pos_param_min < gr.co_posonlyargcount:
            raise TypeError(
                f"cannot bin positional-only argument {pos_param_names[kw_pos_param_min]} as keyword in partial"
            )

        num_to_kw = gr.co_argcount - kw_pos_param_min
        if gr.func_defaults:
            to_kw = gr.func_defaults[-num_to_kw:]
            del gr.func_defaults[-num_to_kw:]
            to_kw_names = pos_param_names[-num_to_kw:]
            gr.func_kwdefaults.update(zip(to_kw_names, to_kw))
        # convert positional args to kw only
        gr.co_kwonlyargcount += num_to_kw
        gr.co_argcount -= num_to_kw

    # deal with positional args. some will be mapped to concrete positional args, some might be added to varargs (*args)
    if gr.ismethod:
        arg_start = 1
        arg_count = gr.co_argcount - 1
    else:
        arg_start = 0
        arg_count = gr.co_argcount

    args_to_bind = pfunc.args[:arg_count]
    args_for_varargs = pfunc.args[arg_count:]

    # do we need to drop positional default args?
    posarg_default_start = gr.co_argcount - len(gr.func_defaults)
    posarg_default_to_delete = len(args_to_bind) + arg_start - posarg_default_start
    if posarg_default_to_delete > 0:
        gr.func_defaults = gr.func_defaults[posarg_default_to_delete:]

    bound_values = gr.local_variables_at_start[arg_start : arg_start + len(args_to_bind)]
    del gr.local_variables_at_start[arg_start : arg_start + len(args_to_bind)]

    for bound_value, arg in zip(bound_values, args_to_bind):
        bound_value.is_function_arg = False
        bound_value.is_const = True
        # TODO: check type?
        bound_value.value = arg
        gr.co_argcount -= 1
        if gr.co_posonlyargcount > 0:
            gr.co_posonlyargcount -= 1

    # handle keyword arguments to concrete parameters, collect in kwargs those for kw-varargs (**kwargs)
    param_names_to_idx = {
        v.name: i for i, v in enumerate(gr.local_variables_at_start[: gr.co_argcount + gr.co_kwonlyargcount])
    }
    kwargs = {}
    for argname, argvalue in pfunc.keywords.items():
        idx = param_names_to_idx.get(argname, -1)
        if idx == -1:
            kwargs[argname] = argvalue
            continue
        gr.func_kwdefaults[argname] = argvalue

    # for varargs and kwargs fed from partial we need the following prelude:
    # TODO: (but maybe we should just have a prelude always for the consts, too...)
    # if it has *varargs:
    #    TMP1 = LOAD_CONST partial_args_for_varargs (needs to be a tuple)
    #    varargs = TMP1 + varargs
    # if it has **kwargs:
    #    TMP2 = LOAD_CONST partial_kwargs
    #    kwargs = partial_kwargs | kwargs

    if args_for_varargs or kwargs:
        prelude = Block()
        jump_node = Node(i=parse.ThunderInstruction.make_jump_absolute(None), inputs=[], outputs=[])
        jump_node.source_infos = [
            SourceInformation(
                orig_file_name="",  # filename?
                orig_line_no=0,
                orig_end_line_no=0,
                gen_line_no=0,
                gen_end_line_no=0,
                col_offset=0,
                end_col_offset=999,
            ),
        ]

        prelude.nodes.append(jump_node)
        jump_target = gr.blocks[0]
        assert jump_target.jump_sources[0] is None
        jump_target.jump_sources[0] = jump_node
        jump_node.jump_targets.append(jump_target)
        prelude.jump_sources.append(None)
        for i in jump_target.block_inputs:
            assert i.jump_sources[0] is None
            i.jump_sources[0] = jump_node
    else:
        prelude = None

    # handle *args (varargs)
    if args_for_varargs:
        if kw_pos_param_idx:
            raise TypeError(
                f"partial tried to bind {len(pfunc.args)} positional arguments, but only {arg_count} are allowed after keyword binding"
            )
        if not (gr.co_flags & inspect.CO_VARARGS):
            raise TypeError(
                f"partial tried to bind {len(pfunc.args)} positional arguments, but only {arg_count} are allowed"
            )
        # the variable for varargs is at gr.co_argcount + gr.co_kwonlyargcount
        v_vararg_param = gr.local_variables_at_start[gr.co_argcount + gr.co_kwonlyargcount]
        v_partial_varargs = Value(name="partial_varargs", value=tuple(args_for_varargs), is_const=True)
        v_varargs_new = Value(name="varargs_with_partial", block=prelude)  # type is tuple
        pv = PhiValue([v_vararg_param], [None], block=prelude)
        new_n = Node(
            i=get_instruction(opname="BINARY_ADD", arg=None),
            inputs=[v_partial_varargs, pv],
            outputs=[v_varargs_new],
        )
        # line number?
        new_n.source_infos = [
            SourceInformation(
                orig_file_name="",  # filename?
                orig_line_no=0,
                orig_end_line_no=0,
                gen_line_no=0,
                gen_end_line_no=0,
                col_offset=0,
                end_col_offset=999,
            ),
        ]
        prelude.nodes.insert(0, new_n)
        prelude.block_outputs.add(v_varargs_new)
        # replace v_vararg_param with v_varargs_new in remainder
        replace_values(gr, {v_vararg_param: v_varargs_new})
        prelude.block_inputs.append(pv)

    # handle **kwargs
    if kwargs:
        if not (gr.co_flags & inspect.CO_VARKEYWORDS):
            raise TypeError(
                f"function does not have **kwargs but partial tries to bind unknown keywords {tuple(kwargs)}."
            )

        # the variable for varargs is at gr.co_argcount + gr.co_kwonlyargcount
        v_kwvararg_param = gr.local_variables_at_start[
            gr.co_argcount + gr.co_kwonlyargcount + (1 if gr.co_flags & inspect.CO_VARARGS else 0)
        ]
        v_partial_kwvarargs = Value(name="partial_kwvarargs", value=kwargs, is_const=True)
        v_kwvarargs_new = Value(name="kwvarargs_with_partial", block=prelude)  # type is dict
        pv = PhiValue([v_kwvararg_param], [None], block=prelude)
        new_n = Node(
            i=get_instruction(opname="BINARY_OR", arg=None),
            inputs=[v_partial_kwvarargs, pv],
            outputs=[v_kwvarargs_new],
        )
        # line number?
        new_n.source_infos = [
            SourceInformation(
                orig_file_name="",  # filename?
                orig_line_no=0,
                orig_end_line_no=0,
                gen_line_no=0,
                gen_end_line_no=0,
                col_offset=0,
                end_col_offset=999,
            ),
        ]
        prelude.nodes.insert(-1, new_n)
        prelude.block_outputs.add(v_kwvarargs_new)
        # replace v_vararg_param with v_varargs_new in remainder
        replace_values(gr, {v_kwvararg_param: v_kwvarargs_new})
        prelude.block_inputs.append(pv)

    if prelude:
        gr.blocks.insert(0, prelude)
    return gr


@functools.cache
def _construct_protograph(func):
    """Protoblocks are parse level constructs, so it is safe to reuse them."""
    return apply_protograph_passes(ProtoGraph.from_code(func.__code__))


@record
def acquire_method(
    method: Callable,
    module: object | None = None,
    mro_klass: type | None = None,
) -> Graph:
    assert SUPPORTS_PREPROCESSING, sys.version_info
    if isinstance(method, functools.partial):
        return acquire_partial(method, module, mro_klass)
    if callable(method) and not inspect.ismethod(method) and not inspect.isfunction(method):
        method = method.__call__

    method_self, func = (method.__self__, method.__func__) if inspect.ismethod(method) else (None, method)
    assert not inspect.ismethod(func)

    module = module or method_self
    if mro_klass is None and module is not None:
        mro_klass = type(module)

    gr = _bind_to_graph(_construct_protograph(func), func, method_self, mro_klass)
    gr.source_start_line = 1
    try:
        gr.source_lines, _ = inspect.getsourcelines(method)
    except OSError:
        gr.source_lines = ["# Failed to extract source."]

    gr.method = method
    gr.module = module
    gr.mro_klass = mro_klass
    if debug_asserts_enabled():
        check_graph(gr)
    return gr


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

    if debug_asserts_enabled():
        check_graph(gr)
