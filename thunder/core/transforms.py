from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from enum import auto, Enum
from itertools import chain, compress
from functools import lru_cache, partial, wraps
import math
from numbers import Number
from typing import Any, Callable, Dict, Union, Optional
from collections.abc import Sequence
import copy
import inspect

import thunder.core.utils as utils
from thunder.core import dtypes, prims
from thunder.clang import full, full_like, unsqueeze, squeeze, maybe_convert_to_dtype, slice_in_dim, sqrt, reciprocal
from thunder.core.devices import cpu, Device
from thunder.core.langctx import get_langctx, set_langctx, reset_langctx, get_default_langctx
from thunder.core.proxies import NumberProxy, Proxy, TensorProxy, variableify
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.symbol import BoundSymbolInterface, Symbol
from thunder.core.trace import TraceCtx as Trace, tracectx
from thunder.core.trace import VariableInterface as Variable
from thunder.core.trace import detached_trace, get_tracectx, set_tracectx, reset_tracectx, from_trace, TraceProvenance
from thunder.core.utils import (
    check,
    consumers,
    flatten_func,
    safe_map,
    safe_map_flat,
    safe_zip,
    unzip2,
    const_as,
    sequencify,
    canonicalize_dims,
    ProxyDict,
)
from thunder.executors.passes import dce
from thunder import clang

# from thunder.executors.torch import ops_to_torch_ops_map

import torch
import numpy as np


# TODO This should be a partial of thunder.trace, but that would cause a circular import
#   issue today. We should refactor so we dont have a circular import problem.
def construct_trace(inline_trace=False, **extra_kwargs):
    import thunder

    return thunder.trace(inline_trace=inline_trace, **extra_kwargs)


#
# Functions related to converting lists of bound symbols to and from DAGs, and operations on
#   DAGs of bound symbols
#
# TODO Consider adding more dag manipulation functions, currently these functions just support analysis
#   and toposorting


# A node in a DAG represents a bsym, with edges defined by the children (outgoing edges) and
#   parents (incoming edges) lists
# NOTE Don't compare nodes directly. Compare their bsyms.
class Node:
    def __init__(self, bsym: BoundSymbolInterface):
        self.bsym = bsym
        self.children: list[Node] = []
        self.parents: list[Node] = []

    # TODO Consider printing parents and children
    def __repr__(self) -> str:
        return str(self.bsym)

    def __hash__(self) -> int:
        utils.check(False, lambda: f"Trying to hash a Node. Hash its bsym instead.")

    def __eq__(self, other) -> bool:
        utils.check(False, lambda: f"Trying to compare Nodes for equality. Compare their bsyms' instead.")


# TODO Think about how to model nodes likes comments -- maybe comments should be associated with
#   other bound symbols so they are always printed above the bound symbol they refer to?
# Converts a sequence of bound symbols to a directed acyclic graph
# Returns a tuple of
#   - a list of all Nodes corresponding to bound symbols without parents
#   - an optional Node corresponding to the RETURN bound symbol, which must be unique or not in the list of bound symbols
def bsym_list_to_dag(bsyms: Sequence[BoundSymbolInterface]) -> tuple[list[Node], Node | None]:
    nodes_without_parents: list[Node] = []
    return_node: None | Node = None

    # Constructs dag
    producers, consumers = utils.producers_and_consumers(bsyms)

    bsym_to_node_map: dict[BoundSymbolInterface, Node] = {}
    for bsym in bsyms:
        node = Node(bsym)
        bsym_to_node_map[bsym] = node

        if bsym.sym.id is prims.PrimIDs.RETURN:
            utils.check(
                return_node is None,
                lambda: f"Found multiple RETURN nodes while converting a list of bound symbols to a dag",
            )
            return_node = node

    for bsym, node in bsym_to_node_map.items():
        has_parents: bool = False
        for inp in chain(bsym._flat_args, bsym._flat_kwargs):
            if not isinstance(inp, Proxy):
                continue

            producer = producers[inp]
            parent = bsym_to_node_map[producer]

            # Checks if the node was already parent to avoid multiple edges between two nodes
            already_has_parent: bool = False
            for pnode in node.parents:
                if producer is pnode.bsym:
                    already_has_parent = True
                    break

            if not already_has_parent:
                node.parents.append(parent)

            has_parents = True

        if not has_parents:
            nodes_without_parents.append(node)

        for out in bsym._flat_outs:
            if not isinstance(out, Proxy):
                continue

            # Checks that the output is actually produced by this function, and not an input to it
            if variableify(out) in chain(
                (variableify(x) for x in bsym._flat_args), (variableify(x) for x in bsym._flat_kwargs)
            ):
                continue

            children = consumers.get(out, [])
            for child in children:
                child_node = bsym_to_node_map[child]

                # Checks if the node was already a child to avoid multiple edges between two nodes
                already_has_child: bool = False
                for cnode in node.children:
                    if child is cnode.bsym:
                        already_has_child = True
                        break

                if not already_has_child:
                    node.children.append(child_node)

    return nodes_without_parents, return_node


class TOPOSORT_ORDER(Enum):
    TOP_DOWN = auto()
    BOTTOM_UP = auto()


def _default_toposort_selector(eligible_nodes: list[Node]) -> int:
    return 0


# Converts a dag of bound symbol nodes into a topologically sorted list of bound symbols
#   "selector" must be a function with signature fn(eligible_nodes: list[Node]) -> int, which
#       returns the index of the next node to add to the list of bound symbols
#       "eligible_nodes" will be a list of all nodes that can appear next in a valid topological
#       sorting of the dag (which is dependent on prevoius sorting choices)
#   If "toposort_order" is TOP_DOWN then the original nodes should be nodes without parents, and
#       eligible nodes will be the set of nodes who have all their parents sorted
#   If "toposort_order" is BOTTOM_UP then the original nodes should be a list with just the return node
#       (as returned from bsym_list_to_dag()) and eligible nodes will be the set of nodes who have
#       all their children sorted
#       NOTE Even though the sorting is BOTTOM_UP, the list of bound symbols will be returned in
#           a valid (top to bottom) order
def toposort_bsym_dag(
    start_nodes: list[Node], toposort_order: TOPOSORT_ORDER, selector: Callable = _default_toposort_selector
) -> list[BoundSymbolInterface]:
    sorted: set[BoundSymbolInterface] = set()
    bsyms: list[BoundSymbolInterface] = []

    eligible_nodes: list[Node] = copy.copy(start_nodes)
    while True:
        if len(eligible_nodes) == 0:
            break

        # Picks the next node
        idx: int = selector(eligible_nodes)
        node: Node = eligible_nodes.pop(idx)
        bsyms.append(node.bsym)
        sorted.add(node.bsym)

        # Identifies additional eligible nodes
        # NOTE This doesn't check that the possibly eligible mode wasn't previously eligible or sorted,
        #   because this is not possible since one of the nodes required for a parent or child to become
        #   eligible was just sorted.
        possibly_eligible = node.parents if toposort_order is TOPOSORT_ORDER.BOTTOM_UP else node.children

        for pe_node in possibly_eligible:
            required_nodes = pe_node.children if toposort_order is TOPOSORT_ORDER.BOTTOM_UP else pe_node.parents

            is_eligible: bool = True
            for req in required_nodes:
                if req.bsym not in sorted:
                    is_eligible = False
                    break

            if is_eligible:
                eligible_nodes.append(pe_node)

    if toposort_order is TOPOSORT_ORDER.BOTTOM_UP:
        bsyms.reverse()

    return bsyms


#
# Functions related to visitor transforms and modifying traces by tracing new functions
#


# TODO We should consider using alternative datastructures for bound symbols if we're manipulating them inplace.
#   Maybe we should be temporarily converting to a deque, or some intermediate datastructure that has to be
#   translated into a list.
# Helper function that extends a list with the values in "extension" from the specified starting index "start"
def _insert_extend_list(l: list, start: int, extension: Sequence[Any]) -> None:
    for offset, arg in enumerate(extension):
        l.insert(start + offset, arg)


# Calls the function fn (which must have the signature fn() -> Any), recording any
#   symbols called into the given trace, starting at the specified index into its
#   list of bound symbols.
# NOTE This operation is inplace. It will modify the trace's bound_symbols.
# NOTE Because this operation is explicitly inplace, it will disregard the trace being "complete".
def insert_inplace(
    trc: Trace,
    idx: int,
    fn: Callable,
) -> None:
    try:
        tracectx_tok = set_tracectx(trc)
        trc._complete = False

        # Creates a temporary scope to record these operations in
        old_scope = trc.scopes
        scope = []
        trc.scopes = [scope]

        fn()
        _insert_extend_list(trc.bound_symbols, idx, scope)

    finally:
        trc.scopes = old_scope
        trc._complete = True
        reset_tracectx(tracectx_tok)


# Removes the BoundSymbol at index from the given trace, then calls fn with it
#   (which must have the signature fn(bsym: BoundSymbol) -> Any) and records
#   any symbols called into the trace, starting at the specified index (the position
#   of the replaced BoundSymbol)
# NOTE This operation is inplace. It will modify the trace's bound_symbols.
# NOTE Because this operation is explicitly inplace, it will disregard the trace being "complete".
def replace_inplace(
    trc: Trace,
    idx: int,
    fn: Callable,
) -> None:
    try:
        tracectx_tok = set_tracectx(trc)
        trc._complete = False

        # Creates a temporary scope to record these operations in
        old_scope = trc.scopes
        scope = []
        trc.scopes = [scope]

        fn(trc.bound_symbols[idx])
        del trc.bound_symbols[idx]
        _insert_extend_list(trc.bound_symbols, idx, scope)

    finally:
        trc.scopes = old_scope
        trc._complete = True
        reset_tracectx(tracectx_tok)


# Specifies how to preserve or replace bound symbols when visiting them
class VISIT_TYPE(Enum):
    INSERT_AFTER = auto()
    INSERT_BEFORE = auto()
    REPLACE = auto()


# Creates a new trace from "trace_from" by calling "visit" on its bound symbols ("bsyms").
#   visit(bsym: BoundSymbolInterface) -> VISIT_TYPE should call operations
#   as if executing a program, and those operations will be recorded into the
#   new trace.
#   If visit() returns INSERT_AFTER for a bsym then that bsym will be copied
#   to the new trace before visit() is called. This is useful when augmenting the bound
#   symbols in an existing trace.
#   If visit() returns INSERT_BEFORE for a bsym then that bsym will be copied to the new trace
#   after visit() is called. This is also useful when augmenting the bound symbols in an existing
#   trace.
#   If visit() returns REPLACE for a bsym then that bsym will not be copied to the new trace.
# TODO Suggest a mechanism to preserve the original bound symbol with operations
#   recorded both before and after it. This could be done by passing the (sub)scope to visit() for
#   direct modification, acquiring the trace's current scope through the trace ctx and modifying it
#   directly (this can be done today), or adding a record() function that is a sugar for the previous
#   approach. Perhaps both passing the scope directly to visit() and adding record() would be helpful.
# TODO(crcrpar): Think about providing a guide how to let thunder "claim" if this is called after
# `thunder.executors.transform_for_execution`.
def visitor_transform(
    trace_from: Trace,
    provenance: str,
    visit: Callable,
) -> Trace:
    trc: Trace = from_trace(trace_from)

    try:
        tracectx_tok = set_tracectx(trc)

        for bsym in trace_from.bound_symbols:
            try:
                # Creates a temporary scope to support copying the original bsym BEFORE
                #   the operations performed by visit(), even though this doesn't know whether to
                #   copy the original bsym until after visit() completes
                old_scope = trc.scopes
                scope = []
                trc.scopes = [scope]

                visit_type = visit(bsym)

                if visit_type is VISIT_TYPE.INSERT_AFTER:
                    trc.bound_symbols.append(bsym)

                trc.bound_symbols.extend(scope)

                if visit_type is VISIT_TYPE.INSERT_BEFORE:
                    trc.bound_symbols.append(bsym)

            finally:
                # Restores the trc's scope
                trc.scopes = old_scope

        # Updates the trace's output
        return_bsym: BoundSymbolInterface = trc.bound_symbols[-1]
        check(
            return_bsym.sym.id is prims.PrimIDs.RETURN,
            lambda: f"Expected the last symbol of a inline visitor transformed trace to be RETURN, but it was {return_bsym.sym}",
        )
        trc.output = return_bsym.output

        trc.set_provenance(TraceProvenance(provenance))

        return trc

    finally:
        reset_tracectx(tracectx_tok)


# NOTE Associating fwd->bwd gradients
#   Consider a function f(inp) -> out, where inp and out are any objects. What should the signature
#   for its corresponding fwd and bwd functions be, and how do we understand how to call bwd and how to associate its
#   output with gradients?
#
#   We define fwd(inp) -> (out, some_saved_stuff) and bwd(some_saved_stuff, gout) -> ginp,
#   where gout and ginp are pytrees with the same structure as out and inp (respectively),
#   but with gradient values inplace of the fwd values. One caveat of this approach is that
#   bwd has to be able to infer the structure of inp from (some_saved_stuff, gout).
#
#   To determine how to associate grads, inp and ginp are flattened and corresponding tensors identified.
#   For example, if mul(a, b) -> c is called, then ginp is a tuple of two tensors, (ga, gb). When inp and ginp
#   are flattened and zipped together, this yields pairs (a, ga), (b, gb).
#
#   One thing to be aware of with this approach is that different calling conventions could confuse
#   this logic. For example, the input of sub(b=a, a=b) also flattens to (a, b), and sub_bwd would
#   naturally produce (gb, ga), and our desired association would be incorrect. This is addressed by
#   canonicalizing a BoundSymbol's args and kwargs so that BoundSymbols are always called in the same way. That is,
#   we canonicalize sub(b=a, a=b) to sub(b, a), and the association of gradients then works as expected.


def _unpack_trivial_fwd(x: Any, *, name: Optional[str] = None) -> Any:
    fwd_result = prims.unpack_trivial(x, name=name)

    return fwd_result, []


# NOTE Doesn't set the grad context -- RETURN doesn't differentiate like other ops
def _return_fwd(*args) -> Any:
    fwd_result = prims.python_return(*args)

    return fwd_result, []


def _mul_fwd(
    a: Number | TensorProxy, b: Number | TensorProxy
) -> tuple[Number | TensorProxy, list[Number | TensorProxy, Number | TensorProxy]]:
    fwd_result = (prims.mul(a, b),)

    return fwd_result, [a, b]


# NOTE Alternative implementation of augmented_forward_impls just to look at elaborating the code
# TODO Provide an extensibility mechanism for this map
# TODO Think about which of these entries should be symbols (if any)
_explicit_fwd_map = {
    # Unpack prims
    prims.PrimIDs.UNPACK_TRIVIAL: _unpack_trivial_fwd,
    # Utility prims
    prims.PrimIDs.RETURN: _return_fwd,
    # Elementwise binary prims
    prims.PrimIDs.MUL: _mul_fwd,
}

# def _mul_prim_bwd(output, saved)

_explicit_bwd_map = {
    # Elementwise binary prims
    # prims.PrimIDs.MUL: _mul_prim_bwd,
}


# WIP -- This will apply the grad transform to forward and create the appropriate grad backward
def pytorch_grad_transform(trc: Trace) -> Trace:
    remap = ProxyDict()

    def try_remap(x):
        if not isinstance(x, Proxy):
            return x

        return remap.get(x, x)

    def update_remap(old, new):
        for a, b in zip(old, new):
            if not isinstance(a, Proxy):
                continue

            remap[a] = b

    def fwd(remap: ProxyDict, bsym: BoundSymbolInterface) -> Any:
        sym = bsym.sym
        remapped_args = tree_map(try_remap, bsym.args)
        remapped_kwargs = tree_map(try_remap, bsym.kwargs)

        # Short circuits if the function has an explicit fwd impl
        explicit_fwd_grad = _explicit_fwd_map.get(sym.id, None)
        if explicit_fwd_grad is not None:
            result, saved = explicit_fwd_grad(*remapped_args, **remapped_kwargs)

            # Updates remap
            flat_outs, _ = tree_flatten(bsym.output)
            flat_results, _ = tree_flatten(result)
            update_remap(flat_outs, flat_results)

            return result, saved

        check(
            not sym.is_prim,
            lambda: f"Failed to find an explicit forward grad implementation for a primitive operation {sym.name}",
        )

        # NOTE In this case the symbol has no explicit fwd impl, so we create an implicit one
        # NOTE We intentionally pass the unused args and kwargs so that they are recorded as inputs in the trace
        def fn_(*args, **kwargs) -> Any:
            total_saved = []
            for sbsym in bsym.subsymbols:
                _, saved = fwd(remap, sbsym)
                total_saved.extend(saved)

            remapped_outs = tree_map(try_remap, bsym.output)
            return remapped_outs, total_saved

        implicit_fwd_name: str = f"{sym.name}_implicit_fwd"
        implicit_fwd = Symbol(name=implicit_fwd_name, meta=fn_, _module=None, _phantom=True)
        results, saved = implicit_fwd(*remapped_args, **remapped_kwargs)

        # Updates remap
        flat_outs, _ = tree_flatten(bsym.output)
        flat_results, _ = tree_flatten(results)
        update_remap(flat_outs, flat_results)

        return results, saved

    # Constructs the start of a new fwd->bwd trace, updating all calls in the original
    #   forward trace to be fwd calls instead (possibly construction an implicit forward)
    def visit(bsym: BoundSymbolInterface) -> VISIT_TYPE:
        result, saved = fwd(remap, bsym)
        return VISIT_TYPE.REPLACE

    ntrc = visitor_transform(
        trace_from=trc,
        provenance="Forward Grad Transform",
        visit=visit,
    )

    # A proxy -> [bsym, bsym, bsym...] mapping, where each bsym is a consumer of the proxy
    consumer_map = consumers(ntrc)

    # TODO Give grads more useful names, like "g0"
    # TODO Add a comment annotation (in the trace) to this exogenous_like call to explain
    #   it models the introduction of gradients
    # NOTE This looks a little weird, because we have
    #   RETURN
    #   EXOGENOUS_LIKE
    #   And of course functions don't usually continue after RETURN, but today we need the RETURN statement still
    #   because we want the trace to understand the data dependency. If we didn't keep RETURN then outputs that don't
    #   require grad might be DCEd, or their creation could be reordered after the EXOGENOUS_LIKE primitive.
    return_bsym: BoundSymbolInterface = ntrc.bound_symbols[-1]
    grad_outputs = []
    for x in return_bsym._flat_args:
        if not isinstance(x, TensorProxy) or not x.requires_grad:
            continue

        grad_outputs.append(x)

    # WIP
    # def grad_for_output(grad_map: ProxyDict, out: Any) -> Any:
    #     flat, spec = tree_flatten(out)

    #     grads = []
    #     for f in flat:
    #         g = None
    #         if isinstance(f, TensorProxy) and f.requires_grad:
    #             g = grad_map.get(f, None)

    #         if g is None:
    #             g.append(None)
    #             continue

    #         total_grad = g[0]
    #         for grad in g[1:]:
    #             total_grad = total_grad + grad

    #         grads.append(total_grad)

    #     return tree_unflatten(grads, spec)

    # def bwd(grad_map: ProxyDict, bsym: BoundSymbolInterface) -> Any:
    #     sym = bsym.sym
    #     out, saved = bsym.output
    #     inp = grad_for_output(grad_map, out)

    #     explicit_bwd_grad = _explicit_bwd_map.get(sym.id, None)

    #     pass

    # A proxy -> [grad, grad, grad...] mapping, where each grad is generated from a consumer of the proxy
    grad_map = ProxyDict()
    with tracectx(ntrc):
        grads = prims.exogenous_like(grad_outputs)
        for g, o in zip(grads, grad_outputs):
            grad_map.append(g, o)

        # Constructs backward
        # TODO NOTE About operations that produce multiple tensors and tensors in collections
        # TODO NOTE About tensors not requiring grad
        # TODO NOTE About outputs that are never differentiable

        # Iterates over the augmented forward bound symbols in reverse
        # NOTE That this is continuing to modify ntrc.bound_symbols (by appending more bound symbols to it),
        #   so the iteration has to independent of those mutation
        # NOTE range is (start, stop, step) (it doesn't accept keyword arguments), and the stop must be -1
        #   because the interval is [start, stop)
        # NOTE That start is -3 because -1 would be the end of the list of bound symbols, and this skips
        #   the EXOGENOUS_LIKE and RETURN statements that terminate fwd
        # num_bsyms = len(ntrc.bound_symbols)
        # for idx in range(num_bsyms - 3, -1, -1):
        #     bsym = ntrc.bound_symbols[idx]
        #     bwd(grad_map, bsym)

    # WIP Construct the full forward->backward trace

    return ntrc


class Transforms(Enum):
    IdentityOp = auto()
    VmapOp = auto()
    JvpOp = auto()
    VjpOp = auto()


# TODO: We are hitting a problem here that langctx might be set to None
# inside make_prim. This is because we are calling make_prim for transform call definition.
# We need to figure out a way to fix this.
# For now we are setting langctx to default langctx
# See https://github.com/Lightning-AI/lightning-thunder/issues/436
def make_transform_prim(
    id,
    name,
    *,
    meta,
):
    def wrapper_fixing_langctx(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            token = set_langctx(get_langctx() or get_default_langctx())
            try:
                return fn(*args, **kwargs)
            finally:
                reset_langctx(token)

        return wrapper

    return prims.make_prim(id, name, meta=wrapper_fixing_langctx(meta))


@lru_cache(maxsize=None)
def symbol_to_eval(bound_symbol):
    """Map a symbol to a function that evaluates it.

    Args:
        symbol: symbol to map
    """
    symbol = bound_symbol.sym
    if isinstance(symbol.id, prims.PrimIDs):
        prim_func = getattr(prims, symbol.name, None)
        if prim_func is not None:
            return prim_func

    # meta_func = prims.ops_to_meta_functions_map[symbol.op]
    # return prims.eval_meta_and_record_symbol_fn(meta_func, symbol.op, symbol.name)
    return symbol.__call__


# TODO: Currently we use trace.args and trace.kwargs to get the arguments
# Maybe we should use these instead
transform_skip_list = (
    prims.PrimIDs.UNPACK_EMPTY_DICT,
    prims.PrimIDs.UNPACK_KEY,
    prims.PrimIDs.UNPACK_SEQUENCE,
    prims.PrimIDs.UNPACK_TRIVIAL,
    prims.PrimIDs.RETURN,
)


def eval_trace(trace, *args, symbol_mapper=symbol_to_eval, with_env=False, **kwargs):
    """Evaluate a trace.

    Args:
        trace: trace to evaluate
        *args: arguments to evaluate the trace with
        symbol_mapper: function that maps a symbol to a function that evaluates it
        **kwargs: keyword arguments to evaluate the trace with

    Returns:
        result of evaluating the trace
    """
    env = {}

    def read(x: Variable):
        if isinstance(x, Variable):
            return env[x.name]
        else:
            return x

    def write(v: Variable, val: Any, allow_duplicates=False) -> None:
        if not isinstance(v, Variable):
            return
        # Duplicates are allowed and overwritten
        if v.name in env:
            if allow_duplicates:
                return
            raise ValueError(f"Variable {v.name} is being overwritten this is not allowed")
        env[v.name] = val

    safe_map_flat(write, list(trace.args), list(args))
    safe_map_flat(write, list(trace.kwargs.values()), list(kwargs.values()))

    # Duplicates are allowed for jvp_call symbols
    # Because the transformed trace is empty in this case and it's no-op
    allow_duplicates_list = (Transforms.JvpOp, Transforms.VjpOp, "torch.contiguous", "torch.Tensor.contiguous")
    write_with_duplicates = partial(write, allow_duplicates=True)

    for symbol in trace.bound_symbols:
        if symbol.sym.id in transform_skip_list:
            continue
        args = tree_map(read, symbol.args)
        kwargs = tree_map(read, symbol.kwargs)
        prim_func = symbol_mapper(symbol)
        if prim_func is None:
            continue
        result = prim_func(*args, **kwargs)
        if symbol.sym.id in allow_duplicates_list:
            safe_map_flat(write_with_duplicates, list(sequencify(symbol.output)), list(sequencify(result)))
            continue
        safe_map_flat(write, list(sequencify(symbol.output)), list(sequencify(result)))

    if with_env:
        return tree_map(read, trace.output), env

    return tree_map(read, trace.output)


def lower_to_prims_mapper(symbol: prims.Symbol):
    """For a given symbol, returns its prim decomposition.

    Args:
        symbol (prims.Symbol): The symbol to lower.

    Returns:
        Callable: The prim that implements the symbol
    """

    # If the symbol is a core primitive, then we don't need to do anything
    prim_func = getattr(prims, symbol.name, None)
    if prim_func is not None:
        return prim_func

    # SLICE primitive doesn't use `symbol.name` to avoid collisions with
    # Python's `slice``, so we need to explicitly map the symbol to its prim
    # function which is called `slice_prim`
    if symbol.sym.id == prims.PrimIDs.SLICE:
        return prims.slice_prim

    # All other symbols are treated as composite functions
    # We decompose them into primitives
    decomposed_fn = symbol.__call__
    return lower_to_prims(decomposed_fn)


def lower_to_prims(func):
    """Converts PyTorch functions to core Thunder primitives.

    Args:
        func (Callable): A Thunder function to be transformed.
    """

    def wrapper(*args, **kwargs):
        trace = construct_trace()(func, *args, **kwargs)
        return eval_trace(trace, *args, **kwargs, symbol_mapper=lower_to_prims_mapper)

    return wrapper


def _identity_call_metafunc(*args, trace: Trace, **kwargs):
    with detached_trace():
        return eval_trace(trace, *args, **kwargs)


identity_call = make_transform_prim(Transforms.IdentityOp, "identity_call", meta=_identity_call_metafunc)


def identity(func):
    """Identity transform for a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.
    """

    def wrapper(*args, **kwargs):
        trace = construct_trace()(func, *args, **kwargs)
        return identity_call(*args, **kwargs, trace=trace)

    return wrapper


def _identity_call_pytorch(*args, trace: Trace, **kwargs):
    import torch

    def symbol_mapper(op):
        if op.op == Transforms.IdentityOp:
            return _identity_call_pytorch

        torch_op = ops_to_torch_ops_map[op.op]
        if isinstance(torch_op, str):
            return getattr(torch, torch_op.strip("pytorch."))
        return torch_op

    with detached_trace():
        return eval_trace(trace, *args, **kwargs, symbol_mapper=symbol_mapper)


# Register the identity call for PyTorch executor.
# ops_to_torch_ops_map[Transforms.IdentityOp] = _identity_call_pytorch


# Inline transform
# ----------------
# The inline transform is a special case of the identity transform.
# It is used to inline the transformation of a function in the trace without
# removing separate transform primitives from the trace.
inline_transforms_map: dict[prims.Symbol, Callable] = dict()


def inline_symbol_mapper(bound_symbol):
    if bound_symbol.sym.id in inline_transforms_map:
        return inline_transforms_map[bound_symbol.sym.id]

    return symbol_to_eval(bound_symbol)


def _identity_call_inline(*args, trace: Trace, **kwargs):
    return eval_trace(trace, *args, **kwargs, symbol_mapper=inline_symbol_mapper)


inline_transforms_map[Transforms.IdentityOp] = _identity_call_inline


def inline(func):
    """Inline transform for a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.
    """

    def wrapper(*args, **kwargs):
        trace = construct_trace()(func, *args, **kwargs)
        return eval_trace(trace, *args, **kwargs, symbol_mapper=inline_symbol_mapper)

    return wrapper


# VMAP transform
# -------------


class NotMapped:
    """Represents a non-batched dimension."""

    def __repr__(self):
        return "not_mapped"


@dataclass(frozen=True)
class BatchedValue:
    """Batched value for the vmap transform.

    Attributes:
        value: Batched value.
        batch_dim: Batching dimension or not_mapped
    """

    value: Any
    batch_dim: Union[int, NotMapped]

    def __iter__(self):
        yield self.value
        yield self.batch_dim


def pair_to_batched_value(pair):
    """Converts a pair to a BatchedValue.

    Args:
        pair (Sequence): Pair to convert.

    Returns:
        BatchedValue: BatchedValue representation of the pair.
    """
    if isinstance(pair, BatchedValue):
        return pair
    else:
        assert isinstance(pair, Sequence) and len(pair) == 2
        return BatchedValue(*pair)


def vectorized_batcher(prim, axis_size, batched_values, **kwargs):
    batch_dim = batched_values[0].batch_dim
    assert all(
        batch_dim == bv.batch_dim for bv in batched_values[1:]
    ), f"`vectorized_batcher` got different batched dimensions {[bv.batch_dim for bv in batched_values]}"
    return BatchedValue(prim(*[bv.value for bv in batched_values], **kwargs), batch_dim)


not_mapped = NotMapped()


def movedim(x, src: int, dst: int):
    perm = [i for i in range(x.ndim) if i != src]
    perm.insert(dst, src)
    return prims.transpose(x, perm)


def move_batch_dim(axis_size, src, dst, x):
    if src is not_mapped:
        if isinstance(x, Number):
            return x
        target_shape = list(x.shape)
        target_shape.insert(dst, axis_size)
        bcast_dims = list(range(len(target_shape)))
        bcast_dims.pop(dst)
        return prims.broadcast_in_dim(x, target_shape, bcast_dims)
    elif src == dst:
        return x
    else:
        return movedim(x, src, dst)


def binary_op_batching_rule(op: prims.PrimIDs, axis_size: int, vals_in: BatchedValue):
    ((x, x_bdim), (y, y_bdim)) = vals_in
    if x_bdim != y_bdim:
        if x_bdim is not_mapped:
            x = move_batch_dim(axis_size, x_bdim, y_bdim, x)
            x_bdim = y_bdim
        else:
            y = move_batch_dim(axis_size, y_bdim, x_bdim, y)
    return BatchedValue(op(x, y), x_bdim)


def sin_vmap(axis_size: int, a: BatchedValue) -> BatchedValue:
    return vectorized_batcher(prims.sin, axis_size, (a,))


def cos_vmap(axis_size: int, a: BatchedValue) -> BatchedValue:
    return vectorized_batcher(prims.cos, axis_size, (a,))


def mul_vmap(axis_size: int, a: BatchedValue, b: BatchedValue) -> BatchedValue:
    return binary_op_batching_rule(prims.mul, axis_size, (a, b))


def add_vmap(axis_size: int, a: BatchedValue, b: BatchedValue) -> BatchedValue:
    return binary_op_batching_rule(prims.add, axis_size, (a, b))


def sum_vmap(axis_size: int, a: BatchedValue, dims: Sequence[int], **kwargs) -> BatchedValue:
    bdim = a.batch_dim
    # TODO: remove this when dims becomes a mandatory kwarg
    if len(dims) > 0:
        dims, _ = safe_zip(*dims)
    dims_before = tuple(el for el in dims if el < bdim)
    dims_after = tuple(el + 1 for el in dims if el >= bdim)
    batched_dims = dims_before + dims_after
    return vectorized_batcher(prims.sum, axis_size, (a,), dims=batched_dims, **kwargs)


# TODO: Please test this extensively
def broadcast_in_dim_vmap(
    axis_size: int, a: BatchedValue, shape: Sequence[BatchedValue], broadcast_dimensions: Sequence[BatchedValue]
) -> BatchedValue:
    bdim = a.batch_dim
    # TODO: remove this when shape and broadcast_dimensions become mandatory kwargs
    # See https://github.com/Lightning-AI/lightning-thunder/issues/181
    shape, _ = safe_zip(*shape)
    if len(broadcast_dimensions) > 0:
        broadcast_dimensions, _ = safe_zip(*broadcast_dimensions)
    if bdim is not_mapped:
        return BatchedValue(prims.broadcast_in_dim(a.value, shape, broadcast_dimensions), bdim)
    else:
        new_bdim = bdim + sum(1 for dim in broadcast_dimensions if dim < bdim)
        new_shape = list(shape)
        new_shape.insert(new_bdim, axis_size)
        new_broadcast_dimensions = (0,) + tuple(dim + 1 if dim >= bdim else dim for dim in broadcast_dimensions)
        if broadcast_dimensions == ():
            new_broadcast_dimensions = ()
        return BatchedValue(prims.broadcast_in_dim(a.value, new_shape, new_broadcast_dimensions), new_bdim)


vmap_impls: dict[prims.Symbol, Callable] = dict()


def unwrap_one_level_of_subsymbols(trace):
    new_symbols_iter = (
        bound_symbol.subsymbols if len(bound_symbol.subsymbols) > 0 else [bound_symbol]
        for bound_symbol in trace.bound_symbols
    )
    new_symbols = list(chain.from_iterable(new_symbols_iter))
    trace.bound_symbols = new_symbols
    return trace


def decomposed_fn_vmap_rule(axis_size, *args, fn, **kwargs):
    args, in_dims = unzip2(args)
    unbatched_args = tree_map(lambda x: remove_batch_dim(x) if isinstance(x, TensorProxy) else x, args)
    trace = construct_trace()(fn, *unbatched_args, **kwargs)
    trace = unwrap_one_level_of_subsymbols(trace)
    outs = _vmap_call_metafunc(False, args, in_dims, 0, axis_size, trace=trace, **kwargs)
    if isinstance(outs, Sequence):
        out_dims = (0,) * len(outs)
        return safe_map(pair_to_batched_value, safe_zip(outs, out_dims))
    return BatchedValue(outs, 0)


vmap_impls[prims.PrimIDs.SIN] = sin_vmap
vmap_impls[prims.PrimIDs.COS] = cos_vmap
vmap_impls[prims.PrimIDs.MUL] = mul_vmap
vmap_impls[prims.PrimIDs.ADD] = add_vmap
vmap_impls[prims.PrimIDs.SUM] = sum_vmap
vmap_impls[prims.PrimIDs.BROADCAST_IN_DIM] = broadcast_in_dim_vmap


def vmap_symbol_mapper(symbol: prims.Symbol, *, axis_size: int):
    """Maps a symbol to a vmap function that evaluates it.

    Args:
        symbol (prims.Symbol): Symbol to evaluate.

    Raises:
        NotImplementedError: If the vmap for the symbol is not implemented.

    Returns:
        Callable: vmap function that evaluates the symbol.
    """

    def wrap_arg(x):
        if isinstance(x, BatchedValue):
            return x
        elif isinstance(x, Number):
            return BatchedValue(x, not_mapped)
        else:
            raise ValueError(f"vmap wrap_arg got an unsupported type {type(x)}")

    if symbol.are_all_args_constant:

        def _vmap_impl_const(symbol, *args, **kwargs):
            out = symbol_to_eval(symbol)(*args, **kwargs)

            if isinstance(out, Sequence):
                return safe_map(pair_to_batched_value, safe_zip(args, [not_mapped] * len(out)))

            return BatchedValue(out, not_mapped)

        return partial(_vmap_impl_const, symbol)

    vmap_impl = vmap_impls.get(symbol.sym.id)
    if vmap_impl is None:
        if len(symbol.subsymbols) > 0:
            vmap_impl = partial(decomposed_fn_vmap_rule, fn=symbol.sym.__call__)
        else:
            raise NotImplementedError(f"vmap for {symbol.sym.id} is not implemented")

    def _vmap_impl(*args, **kwargs):
        args = tree_map(wrap_arg, args)
        assert all(isinstance(arg, BatchedValue) for arg in tree_flatten(args)[0])
        return vmap_impl(axis_size, *args, **kwargs)

    return _vmap_impl


def remove_batch_dim(tensor: TensorProxy, batch_dim: int = 0) -> TensorProxy:
    """Removes the batch dimension from a tensor.

    Args:
        tensor (TensorProxy): Tensor to remove the batch dimension from.

    Returns:
        TensorProxy: Tensor with the batch dimension removed.
    """
    new_shape = tensor.shape[:batch_dim] + tensor.shape[batch_dim + 1 :]
    return TensorProxy(like=tensor, shape=new_shape)


# TODO: in JAX args, in_dims are flattened the same way
# TODO: in JAX out_dims are flattened as well
def _vmap_call_metafunc(detached: bool, args, in_dims, out_dims, axis_size, trace: Trace, **kwargs):
    """Metafunction for vmap call.

    Args:
        detached (bool): Whether to detach the trace.
        args (Tuple[Proxy]): Arguments to the function.
        in_dims (Tuple[int]): Batch dimension for each argument.
        out_dims (Tuple[int]): Batch dimension for return values.
        trace (Trace): Trace to use for the function.
        kwargs: Keyword arguments.

    Raises:
        AssertionError: If the vmap for keyword arguments is not implemented.

    Returns:
        Result of the vmap transform.
    """
    common_device = {x.device for x in args if isinstance(x, TensorProxy)}
    assert len(common_device) <= 1, "vmap for multiple devices is not implemented"
    (common_device,) = common_device if len(common_device) == 1 else (cpu,)

    if axis_size is None:
        (axis_size,) = {x.shape[ax] for x, ax in zip(args, in_dims) if ax is not not_mapped}
    in_dims = in_dims if isinstance(in_dims, Sequence) else (in_dims,)
    in_dims = tuple(not_mapped if isinstance(a, Number) else d for a, d in safe_zip(args, in_dims))
    out_dims = out_dims if isinstance(out_dims, Sequence) else (out_dims,)

    ctx = detached_trace() if detached else nullcontext()
    with ctx:
        # We propagate the BatchValue through the trace, and then unwrap it at the end
        batched_args = safe_map(pair_to_batched_value, safe_zip(args, in_dims))
        result = eval_trace(
            trace, *batched_args, symbol_mapper=partial(vmap_symbol_mapper, axis_size=axis_size), **kwargs
        )
        # Unwrapping the BatchedValue's
        if isinstance(result, Sequence):
            flat_result, spec = tree_flatten(result)
            assert all(isinstance(x, BatchedValue) for x in flat_result)
            outs, bdims = unzip2(flat_result)
            # TODO: handle the case where out_dims is a single value better
            if len(out_dims) == 1:
                out_dims = out_dims * len(outs)
            outs = safe_map(partial(move_batch_dim, axis_size), bdims, out_dims, outs)
            return tree_unflatten(outs, spec)
        if isinstance(result, Number) and axis_size is not None:
            # TODO: fetch the default device from the context
            result = full(shape=(), fill_value=result, device=common_device)
            result = BatchedValue(result, not_mapped)
        elif isinstance(result, BatchedValue) and isinstance(result.value, Number) and axis_size is not None:
            result = BatchedValue(full(shape=(), fill_value=result.value, device=common_device), result.batch_dim)
        assert isinstance(result, BatchedValue)
        out = move_batch_dim(axis_size, result.batch_dim, out_dims[0], result.value)
        return out


vmap_call = make_transform_prim(Transforms.VmapOp, "vmap_call", meta=partial(_vmap_call_metafunc, True))
inline_transforms_map[Transforms.VmapOp] = partial(_vmap_call_metafunc, False)


# TODO: how should we handle out_dims here?
# although here we are calling vmap of identity, so we should know from the call to vmap
# This should be fine. If we have vmap(identity(func), out_dims=N) then this rule is first used
# to get the vmapped result of identity(func) in vmap_symbol_mapper, and out_dims is handled
# after that in the outer _vmap_call_metafunc.
def _identity_call_vmap(axis_size, *batched_args, trace: Trace, **kwargs):
    args, in_dims = unzip2(batched_args)
    out_dims = 0  # Fixme
    outs, out_dims = _vmap_call_metafunc(False, args, in_dims, out_dims, axis_size, trace=trace, **kwargs)
    if isinstance(outs, Sequence):
        return safe_map(pair_to_batched_value, safe_zip(outs, out_dims))
    return BatchedValue(outs, out_dims)


vmap_impls[Transforms.IdentityOp] = _identity_call_vmap


def _jvp_call_vmap(axis_size, batched_primals, batched_tangents, trace: Trace, **kwargs):
    primals, primals_bdims = safe_zip(*batched_primals)
    tangents, tangents_bdims = safe_zip(*batched_tangents)
    jvp_func = inline(partial(jvp_call, trace=trace))
    vmapped_jvp_func = inline(vmap(jvp_func, in_dims=(primals_bdims, tangents_bdims), axis_size=axis_size))
    result = vmapped_jvp_func(primals, tangents, **kwargs)
    return tree_map(lambda x: BatchedValue(x, 0), result)


vmap_impls[Transforms.JvpOp] = _jvp_call_vmap


def vmap(func, in_dims=0, out_dims=0, axis_size=None):
    """Vectorizing transform for a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.

    Returns:
        Callable: A vmapped version of the function.
    """

    # TODO: flatten
    # In JAX flattening of in_dims is rather complicated because it can optionally be
    # specified as a “prefix” pytree, meaning that a single leaf value can be applied
    # to an entire sub-pytree.

    def flatten_func_for_vmap(func, args, kwargs):
        flat_args, spec = tree_flatten((args, kwargs))

        def flat_func(*flat_args):
            fn_args, fn_kwargs = tree_unflatten(flat_args, spec)
            return func(*fn_args, **fn_kwargs)

        return flat_func, flat_args, spec

    def wrapper(*args, **kwargs):
        func_flat, args_flat, args_spec = flatten_func_for_vmap(func, args, kwargs)
        if isinstance(in_dims, int):
            in_dims_flat = (in_dims,) * len(args_flat)
        else:
            in_dims_flat, in_dims_spec = tree_flatten(in_dims)
            assert len(in_dims_flat) == len(args_flat), "in_dims must have the same length as args, kwargs"
        unbatched_args_flat = [remove_batch_dim(arg) if isinstance(arg, TensorProxy) else arg for arg in args_flat]
        trace = construct_trace()(func_flat, *unbatched_args_flat)
        outs = vmap_call(args_flat, in_dims_flat, out_dims, axis_size=axis_size, trace=trace)
        return outs

    return wrapper


# TODO This function commented out because it calls make_traced, which does not exist
# def vmap_eager(func, args, in_dims=0, out_dims=0, axis_size=None, executor="torch"):
#     """Computes the vmap of a Thunder function.

#     Args:
#         func (Callable): A Thunder function to be transformed.
#         args (_type_): Args of the function.
#         executor (str, optional): Executor to use. Defaults to "torch".

#     Returns:
#         The result of the vmapped function.
#     """
#     # TODO: fix this - not all args may be batched
#     # TODO: here we assume batch axis is 0
#     vmap_trace = make_trace(
#         inline(vmap(func, in_dims=in_dims, out_dims=out_dims, axis_size=axis_size)), executor=executor,
#         *args)
#     vmap_traced = make_traced(partial(eval_trace, vmap_trace), executor=executor)
#     return vmap_traced(*args)


# JVP transform
# -------------


@dataclass(frozen=True)
class JVPDual:
    """Dual number for the JVP transform.

    Attributes:
        primal: Primal value.
        tangent: Tangent value.
    """

    primal: Any
    tangent: Any

    def __iter__(self):
        yield self.primal
        yield self.tangent


def pair_to_jvp_dual(pair):
    """Converts a pair to a JVPDual.

    Args:
        pair (Sequence): Pair to convert.

    Returns:
        JVPDual: JVPDual representation of the pair.
    """
    if isinstance(pair, JVPDual):
        return pair
    else:
        assert isinstance(pair, Sequence) and len(pair) == 2
        return JVPDual(*pair)


def sin_jvp(a: JVPDual):
    x, xd = a
    return JVPDual(prims.sin(x), prims.cos(x) * xd)


def mul_jvp(a: JVPDual, b: JVPDual):
    x, xd = a
    y, yd = b
    return JVPDual(x * y, x * yd + y * xd)


def add_jvp(a: JVPDual, b: JVPDual):
    x, xd = a
    y, yd = b
    return JVPDual(x + y, xd + yd)


def broadcast_in_dim_jvp(a: JVPDual, shape: tuple[JVPDual, ...], broadcast_dimensions: tuple[JVPDual, ...]) -> JVPDual:
    x, xd = a
    # TODO: shape and broadcast_dimensions should be tuples of ints
    # but for now it's a tuple of JVPDuals
    # See https://github.com/Lightning-AI/lightning-thunder/issues/181
    if len(shape) > 0 and isinstance(shape[0], JVPDual):
        shape, _ = safe_zip(*shape)
    if len(broadcast_dimensions) > 0 and isinstance(broadcast_dimensions[0], JVPDual):
        broadcast_dimensions, _ = safe_zip(*broadcast_dimensions)
    return JVPDual(
        prims.broadcast_in_dim(x, shape, broadcast_dimensions), prims.broadcast_in_dim(xd, shape, broadcast_dimensions)
    )


def unpack_sequence_jvp(sequence: JVPDual, length: JVPDual) -> JVPDual:
    x = tree_map(lambda x: x.primal, sequence)
    xd = tree_map(lambda x: x.tangent, sequence)
    length, _ = length
    primals = prims.unpack_sequence(x, length)
    tangents = prims.unpack_sequence(xd, length)
    return safe_map(pair_to_jvp_dual, safe_zip(primals, tangents))


def unpack_trivial_jvp(x: JVPDual) -> JVPDual:
    return x


jvp_impls: dict[prims.Symbol, Callable] = dict()

jvp_impls[prims.PrimIDs.SIN] = sin_jvp
jvp_impls[prims.PrimIDs.MUL] = mul_jvp
jvp_impls[prims.PrimIDs.ADD] = add_jvp
jvp_impls[prims.PrimIDs.BROADCAST_IN_DIM] = broadcast_in_dim_jvp
# jvp_impls[prims.PrimIDs.UNPACK_SEQUENCE] = unpack_sequence_jvp
# jvp_impls[prims.PrimIDs.UNPACK_TRIVIAL] = unpack_trivial_jvp


def jvp_symbol_mapper(symbol: prims.Symbol):
    """Maps a symbol to a JVP function that evaluates it.

    Args:
        symbol (prims.Symbol): Symbol to evaluate.

    Raises:
        NotImplementedError: If the JVP for the symbol is not implemented.

    Returns:
        Callable: JVP function that evaluates the symbol.
    """

    def wrap_arg(x):
        if isinstance(x, JVPDual):
            return x
        elif isinstance(x, Number):
            return JVPDual(x, type(x)(0))
        else:
            raise ValueError(f"JVP wrap_arg got an unsupported type {type(x)}")

    # If symbol.args doesn't have subclasses of Variable, then we need to return a zero tangent
    # TODO: there may be a better way to detect constants in the trace
    if symbol.are_all_args_constant:

        def zeros_like(x):
            if isinstance(x, TensorProxy):
                return full_like(x, fill_value=0)
            elif isinstance(x, NumberProxy):
                return type(x.value)(0)
            elif isinstance(x, Number):
                return type(x)(0)
            else:
                raise ValueError(f"zeros_like inside JVP got an unsupported type {type(x)}")

        def jvp_impl_const(symbol, *args, **kwargs):
            primals = symbol_to_eval(symbol)(*args, **kwargs)
            if isinstance(primals, Sequence):
                tangents = tuple(zeros_like(p) for p in primals)
                return safe_map(pair_to_jvp_dual, safe_zip(primals, tangents))
            return JVPDual(primals, zeros_like(primals))

        return partial(jvp_impl_const, symbol)

    # Normal case, we have a proxy tangent
    jvp_impl = jvp_impls.get(symbol.sym.id)
    if jvp_impl is None:
        raise NotImplementedError(f"JVP for {symbol.sym.id} is not implemented")

    def _jvp_impl(*args, **kwargs):
        args = tree_map(wrap_arg, args)
        # Expecting JVPDuals wrapping pairs of primals and tangents
        assert all(isinstance(arg, JVPDual) for arg in tree_flatten(args)[0])
        return jvp_impl(*args, **kwargs)

    return _jvp_impl


def _jvp_call_metafunc(detached: bool, primals, tangents, trace: Trace, **kwargs):
    """Metafunction for the JVP transform.

    Args:
        detached (bool): Whether to detach the trace.
        primals (Tuple[Proxy]): Primal values.
        tangents (Tuple[Proxy]): Tangent values.
        trace (Trace): Trace of the function to be transformed.
        kwargs: Keyword arguments.

    Raises:
        AssertionError: If the JVP for keyword arguments is not implemented.

    Returns:
        Result of the JVP transform.
    """
    assert len(kwargs) == 0, "JVP for kwargs is not implemented"

    ctx = detached_trace() if detached else nullcontext()
    with ctx:
        # Wrapping the primals and tangents in JVPDuals is not strictly necessary, but it makes
        # the code more readable
        # We propagate the JVPDuals through the trace, and then unwrap them at the end
        primals_tangents_duals = safe_map(pair_to_jvp_dual, safe_zip(primals, tangents))
        result = eval_trace(trace, *primals_tangents_duals, symbol_mapper=jvp_symbol_mapper)
        # Unwrapping the JVPDuals
        if isinstance(result, Sequence):
            assert all(isinstance(x, JVPDual) for x in result)
            primals, tangents = unzip2(result)
            return primals, tangents
        assert isinstance(result, JVPDual)
        return result.primal, result.tangent


jvp_call = make_transform_prim(Transforms.JvpOp, "jvp_call", meta=partial(_jvp_call_metafunc, True))
inline_transforms_map[Transforms.JvpOp] = partial(_jvp_call_metafunc, False)


def _identity_call_jvp(*args: JVPDual, trace: Trace, **kwargs):
    primals, tangents = unzip2(args)
    out_primals, out_tangents = _jvp_call_metafunc(False, primals, tangents, trace, **kwargs)
    if isinstance(out_primals, Sequence):
        return safe_map(pair_to_jvp_dual, safe_zip(out_primals, out_tangents))
    return JVPDual(out_primals, out_tangents)


jvp_impls[Transforms.IdentityOp] = _identity_call_jvp


def _vmap_call_jvp(args: JVPDual, in_dims, out_dims, axis_size, trace: Trace, **kwargs):
    primals, tangents = safe_zip(*args)
    in_dims, _ = safe_zip(*in_dims)
    out_dims, _ = safe_zip(*out_dims)
    vmapped_trace = construct_trace()(
        inline(vmap(partial(eval_trace, trace), in_dims=in_dims, out_dims=out_dims, axis_size=axis_size)), *primals
    )
    vmapped_func = partial(eval_trace, vmapped_trace)
    out_primals, out_tangents = inline(jvp(vmapped_func))(primals, tangents, **kwargs)
    if isinstance(out_primals, Sequence):
        return safe_map(pair_to_jvp_dual, safe_zip(out_primals, out_tangents))
    return JVPDual(out_primals, out_tangents)


jvp_impls[Transforms.VmapOp] = _vmap_call_jvp


def jvp(func):
    """Jacobian-vector product transform for a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.

    Returns:
        Callable: A function that computes the Jacobian-vector product
            taking primals and tangents as arguments.
    """

    def wrapper(primals, tangents):
        trace = construct_trace()(func, *primals)
        return jvp_call(primals, tangents, trace=trace)

    return wrapper


# TODO This function commented out because it calls make_traced, which does not exist
# def jvp_eager(func, primals, tangents, executor="torch"):
#     """Computes the Jacobian-vector product of a Thunder function.

#     Args:
#         func (Callable): A Thunder function to be transformed.
#         primals (_type_): Primals of the function.
#         tangents (_type_): Tangents of the function.
#         executor (str, optional): Executor to use. Defaults to "torch".

#     Returns:
#         The result of the Jacobian-vector product.
#     """
#     trace = make_trace(func, executor=executor, *primals)

#     def jvp_func(*primals_and_tangents):
#         _primals, _tangents = primals_and_tangents[: len(primals)], primals_and_tangents[len(primals) :]
#         return _jvp_call_metafunc(_primals, _tangents, trace, detached=False)

#     jvp_trace = make_trace(jvp_func, executor=executor)(*primals, *tangents)
#     jvp_traced = make_traced(partial(eval_trace, jvp_trace), executor=executor)
#     return jvp_traced(*primals, *tangents)


# VJP transform
# =============
@dataclass(frozen=True)
class VJPDual:
    """A pair of primal and saved information for backward (residuals).

    Args:
        primal (Union[Proxy, Number]): Primal value, i.e., the value being differentiated.
        residuals (Tuple[Proxy, ...]): Residuals, i.e., the values that are
            saved for the backward.

    Yields:
        Tuple[Variable, Tuple[Variable, ...], Callable]: Primal and residuals
    """

    primal: Union[Proxy, Number]
    residuals: tuple[Proxy, ...]

    def __iter__(self):
        yield self.primal
        yield self.residuals


class NoPullback:
    """A dummy pullback function that returns None or raises an error."""

    def __init__(self, num_args=0):
        self.num_args = num_args

    def __call__(self, *args, **kwargs):
        if self.num_args > 0:
            return (None,) * self.num_args
        raise RuntimeError("Pullback called on a non-differentiable symbol or a constant.")


class ZeroBackward:
    """A helper backward function that returns zeros."""

    def __init__(self, num_args):
        self.num_args = num_args

    def __call__(self, *args, **kwargs):
        # Assuming that the first arguments are the forward arguments
        forward_args = args[: self.num_args]

        def zeros_like(x):
            if isinstance(x, TensorProxy):
                return full_like(x, fill_value=0)
            elif isinstance(x, NumberProxy):
                return type(x.value)(0)
            elif isinstance(x, Number):
                return type(x)(0)
            else:
                raise ValueError(f"zeros_like inside ZeroBackward got an unsupported type {type(x)}")

        return tuple(zeros_like(arg) for arg in forward_args)


# Mapping from symbols to augmented primal (forward) functions used in VJP
# The augmented_primal function takes the primal values and returns the primal
# result and the residuals (saved values for the backward).
augmented_forward_impls = {
    prims.PrimIDs.ABS: lambda x: (prims.abs(x), (x,)),
    prims.PrimIDs.ACOS: lambda x: (prims.acos(x), (x,)),
    prims.PrimIDs.ACOSH: lambda x: (prims.acosh(x), (x,)),
    prims.PrimIDs.ADD: lambda x, y: (prims.add(x, y), tuple()),
    prims.PrimIDs.ASIN: lambda x: (prims.asin(x), (x,)),
    prims.PrimIDs.ASINH: lambda x: (prims.asinh(x), (x,)),
    prims.PrimIDs.ATAN: lambda x: (prims.atan(x), (x,)),
    prims.PrimIDs.ATANH: lambda x: (prims.atanh(x), (x,)),
    prims.PrimIDs.COS: lambda x: (prims.cos(x), (x,)),
    prims.PrimIDs.COSH: lambda x: (prims.cosh(x), (x,)),
    prims.PrimIDs.DIV: lambda x, y: (prims.div(x, y), (x, y)),
    prims.PrimIDs.ERF: lambda x: (prims.erf(x), (x,)),
    prims.PrimIDs.ERFC: lambda x: (prims.erfc(x), (x,)),
    prims.PrimIDs.ERFINV: lambda x: (prims.erfinv(x), (prims.erfinv(x),)),
    prims.PrimIDs.ERFCINV: lambda x: (prims.erfcinv(x), (prims.erfcinv(x),)),
    prims.PrimIDs.EXP2: lambda x: (prims.exp2(x), (prims.exp2(x),)),
    prims.PrimIDs.EXPM1: lambda x: (prims.expm1(x), (prims.expm1(x),)),
    prims.PrimIDs.MUL: lambda x, y: (prims.mul(x, y), (x, y)),
    prims.PrimIDs.NDTRI: lambda x: (prims.ndtri(x), (prims.ndtri(x),)),
    prims.PrimIDs.SIN: lambda x: (prims.sin(x), (x,)),
    prims.PrimIDs.SINH: lambda x: (prims.sinh(x), (x,)),
    prims.PrimIDs.SUB: lambda x, y: (prims.sub(x, y), tuple()),
    prims.PrimIDs.SQRT: lambda x: (prims.sqrt(x), (prims.sqrt(x),)),
    prims.PrimIDs.EQ: lambda x, y: (prims.eq(x, y), (x, y)),
    prims.PrimIDs.GE: lambda x, y: (prims.ge(x, y), (x, y)),
    prims.PrimIDs.GT: lambda x, y: (prims.gt(x, y), (x, y)),
    prims.PrimIDs.LT: lambda x, y: (prims.lt(x, y), (x, y)),
    prims.PrimIDs.LOG: lambda x: (prims.log(x), (x,)),
    prims.PrimIDs.LOG10: lambda x: (prims.log10(x), (x,)),
    prims.PrimIDs.LOG1P: lambda x: (prims.log1p(x), (x,)),
    prims.PrimIDs.LOG2: lambda x: (prims.log2(x), (x,)),
    prims.PrimIDs.NEG: lambda x: (prims.neg(x), tuple()),
}

# Mapping from symbols to backward functions used in VJP
# The backward function takes the residuals and cotangents and returns the
# vector-Jacobian products for each argument.
backward_impls = {
    prims.PrimIDs.ABS: lambda x, g: g * prims.sign(x),
    prims.PrimIDs.ACOS: lambda x, g: -g / prims.sqrt(1.0 - x * x),
    prims.PrimIDs.ACOSH: lambda x, g: g * prims.rsqrt(x * x - 1.0),
    # Duplicates are not allowed in the backward_impls
    # Therefore, we multiply the gradient by 1.0 to make it a different tensor
    prims.PrimIDs.ADD: lambda g: (1.0 * g, 1.0 * g),
    prims.PrimIDs.ASIN: lambda x, g: g / prims.sqrt(1.0 - x * x),
    prims.PrimIDs.ASINH: lambda x, g: g * prims.rsqrt(1.0 + x * x),
    prims.PrimIDs.ATAN: lambda x, g: g / (1.0 + x * x),
    prims.PrimIDs.ATANH: lambda x, g: g / (1.0 - x * x),
    prims.PrimIDs.COS: lambda x, g: prims.mul(g, -prims.sin(x)),
    prims.PrimIDs.COSH: lambda x, g: prims.mul(g, prims.sinh(x)),
    prims.PrimIDs.DIV: lambda x, y, g: (g / y, -g * x / (y**2)),
    prims.PrimIDs.ERF: lambda x, g: g * 2.0 / prims.sqrt(math.pi) * prims.exp(-x * x),
    prims.PrimIDs.ERFC: lambda x, g: -g * 2.0 / prims.sqrt(math.pi) * prims.exp(-x * x),
    prims.PrimIDs.ERFINV: lambda result, g: g * 0.5 * prims.sqrt(math.pi) * prims.exp(result**2),
    prims.PrimIDs.ERFCINV: lambda result, g: -g * 0.5 * prims.sqrt(math.pi) * prims.exp(result**2),
    prims.PrimIDs.EXP2: lambda result, g: g * result * math.log(2.0),
    prims.PrimIDs.EXPM1: lambda result, g: g * (result + 1.0),
    prims.PrimIDs.MUL: lambda x, y, g: (g * y, g * x),
    prims.PrimIDs.NDTRI: lambda result, g: g * prims.exp(0.5 * result**2) * prims.sqrt(2.0 * math.pi),
    prims.PrimIDs.SIN: lambda x, g: prims.mul(g, prims.cos(x)),
    prims.PrimIDs.SINH: lambda x, g: prims.mul(g, prims.cosh(x)),
    prims.PrimIDs.SUB: lambda g: (g, -g),
    prims.PrimIDs.SQRT: lambda result, g: g / (2.0 * result),
    prims.PrimIDs.FULL: NoPullback(num_args=2),
    prims.PrimIDs.EQ: ZeroBackward(num_args=2),
    prims.PrimIDs.GE: ZeroBackward(num_args=2),
    prims.PrimIDs.LT: ZeroBackward(num_args=2),
    prims.PrimIDs.LOG: lambda x, g: g / x,
    prims.PrimIDs.LOG10: lambda x, g: g / (x * 2.302585092994046),
    prims.PrimIDs.LOG1P: lambda x, g: g / (x + 1),
    prims.PrimIDs.LOG2: lambda x, g: g / (x * 0.6931471805599453),
    prims.PrimIDs.NEG: lambda g: -g,
}


def register_augmented_forward(op):
    """Decorator to register an augmented forward implementation for a symbol.

    Args:
        op (Ops): Symbol for which to register the augmented forward implementation.

    Returns:
        Callable: Decorator function.
    """

    def decorator(func):
        augmented_forward_impls[op] = func
        return func

    return decorator


def register_backward(op):
    """Decorator to register a backward implementation for a symbol.

    Args:
        op (Ops): Symbol for which to register the backward implementation.

    Returns:
        Callable: Decorator function.
    """

    def decorator(func):
        backward_impls[op] = func
        return func

    return decorator


def restore_reduced_dims(x, reduced_dims, original_shape):
    """Restores the reduced dimensions of a tensor.

    Args:
        x (Variable): Tensor to be reshaped.
        reduced_dims (Tuple[int, ...]): Tuple of reduced dimensions.
        original_shape (Tuple[int, ...]): Original shape of the tensor.

    Returns:
        Variable: Tensor with the reduced dimensions restored.
    """
    if original_shape == ():  # scalar
        return x

    unsqueezed = clang.unsqueeze(x, reduced_dims)
    return clang.expand(unsqueezed, original_shape)


@register_augmented_forward(prims.PrimIDs.RSQRT)
def rsqrt_augmented(x):
    """Augmented rsqrt operation.

    Args:
        x (Variable): input tensor.

    Returns:
        VJPDual: Primal and residuals.
    """
    primal = prims.rsqrt(x)
    residuals = (primal,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.RSQRT)
def rsqrt_backward(result, g):
    # An alternative derivation used by JAX is -0.5 * g * rsqrt(x) / x
    # where rsqrt(x) and x are saved for the backwards pass.
    # This derivation was selected because it avoids saving the input tensor.
    return -0.5 * g * result**3.0


@register_augmented_forward(prims.PrimIDs.SUM)
def sum_aug_fwd(x, dims, output_dtype=None):
    """Augmented sum operation.

    Args:
        x (Variable): Tensor to be summed.
        dims (Tuple[int, ...]): Dimensions to be summed.
        output_dtype (str, optional): Output data type. Defaults to None.

    Returns:
        VJPDual: Primal and residuals.
    """
    primal = prims.sum(x, dims, output_dtype=output_dtype)
    residuals = (
        x.shape,
        dims,
    )

    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.SUM)
def sum_backward(x_shape, reduced_dims, g):
    # One return per positional argument of prims.sum
    return restore_reduced_dims(g, reduced_dims, x_shape), None


@register_augmented_forward(prims.PrimIDs.VAR)
def var_aug_fwd(a, dim, *, correction):
    v = prims.var(a, dim, correction=correction)
    return VJPDual((v,), (a, dim, correction, v))


# TODO: fix division by zero when n_elem_reduced == 0 or when v.numel == 0
# by returning zeros_like(a) or similar.
# TODO: fix grad when correction > n_elem_reduced.
@register_backward(prims.PrimIDs.VAR)
def var_backward(a, dim, correction, v, g):
    n_elem_reduced = a.numel // v.numel if a.numel != 0 else 1
    normalization_scalar = n_elem_reduced - correction
    g = restore_reduced_dims(g, dim, a.shape)
    mean = prims.sum(a, dim, output_dtype=v.dtype) / n_elem_reduced
    mean = restore_reduced_dims(mean, dim, a.shape)
    return ((2 * g * (a - mean)) / normalization_scalar, None)


@register_augmented_forward(prims.PrimIDs.VAR_MEAN)
def _var_mean_aug_fwd(a, dim, *, correction):
    v, m = prims.var_mean(a, dim, correction=correction)

    return (v, m), (a, dim, correction, m)


# TODO: fix division by zero when n_elem_reduced == 0 or when mean.numel == 0
# by returning zeros_like(a) or similar.
# TODO: fix grad when correction > n_elem_reduced.
@register_backward(prims.PrimIDs.VAR_MEAN)
def _var_mean_bwd(a, dim, correction, mean, grad_v, grad_m):
    n_elem_reduced = a.numel // mean.numel if a.numel != 0 else 1

    def mean_backward(a, dims, grad):
        mean_scale = 1.0 / n_elem_reduced
        grad = restore_reduced_dims(grad, dims, a.shape)
        return mean_scale * grad

    def var_backward(a, dims, correction, mean, grad):
        normalization_scalar = n_elem_reduced - correction
        grad = restore_reduced_dims(grad, dims, a.shape)
        mean = restore_reduced_dims(mean, dims, a.shape)
        return (2.0 * grad * (a - mean)) / normalization_scalar

    return (
        var_backward(a, dim, correction, mean, grad_v) + mean_backward(a, dim, grad_m),
        None,
    )


@register_augmented_forward(prims.PrimIDs.PAD)
def pad_aug_fwd(a, padding_value, padding_config):
    return VJPDual((prims.pad(a, padding_value, padding_config),), (a, padding_config))


@register_backward(prims.PrimIDs.PAD)
def pad_backward(a, padding_config, g):
    # Short circuit on empty input.
    if any(dim == 0 for dim in a.shape):
        return full_like(a, fill_value=0), None, None

    # Un-pad by padding with zero values
    zero_padding_config = [(-lo, -hi, 0) for lo, hi, _ in padding_config]

    g = prims.pad(g, 0.0, zero_padding_config)

    # Un-slice by slicing with a stride of value (dilation + 1)
    for dim, (_, _, d) in enumerate(padding_config):
        g = slice_in_dim(g, 0, g.shape[dim], stride=d + 1, dim=dim)

    return g, None, None


@register_augmented_forward(prims.PrimIDs.PROD)
def prod_aug_fwd(x, dims):
    """Augmented prod operation.

    Args:
        x (Variable): Tensor to be multiplied.
        dims (Tuple[int, ...]): Dimensions to be multiplied.

    Returns:
        VJPDual: Primal and residuals.
    """
    primal = prims.prod(x, dims)

    residuals = (
        primal,
        x,
        x.shape,
        dims,
    )
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.PROD)
def prod_pullback(primal, x, x_shape, reduced_dims, g):
    # One return per positional argument of prims.prod
    return prims.div(restore_reduced_dims(primal * g, reduced_dims, x_shape), x), None


def keepdim_reduction(reduction_fn, x, dims):
    """Applies reduction and fixes output to conform to keepdim=True"""
    out = reduction_fn(x, dims)
    argmax_sum_out_shape = [x.shape[i] if i not in dims else 1 for i in range(x.ndim)]
    broadcast_dims = [i for i in range(x.ndim) if i not in dims]
    return prims.broadcast_in_dim(out, argmax_sum_out_shape, broadcast_dims)


# Inspired from https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py#L353
def grad_chooser_pullback(primal, x, x_shape, reduced_dims, g):
    """Builds gradient of functions that choose a single item, such as min or max."""
    g_repeated = restore_reduced_dims(g, reduced_dims, x_shape)
    primal_repeated = restore_reduced_dims(primal, reduced_dims, x_shape)
    argmax_locations = x == primal_repeated
    argmax_sum = keepdim_reduction(prims.sum, argmax_locations, reduced_dims)
    out = g_repeated * argmax_locations / argmax_sum
    return out, None


register_backward(prims.PrimIDs.AMAX)(grad_chooser_pullback)
register_backward(prims.PrimIDs.AMIN)(grad_chooser_pullback)


# TODO: exact same for amin, argmax, argmin
@register_augmented_forward(prims.PrimIDs.AMAX)
def amax_aug_fwd(x, dims):
    """Augmented amax operation.

    Args:
        x (Variable): Tensor to compute amax on.
        dims (Tuple[int, ...]): Dimensions to compute amax over.

    Returns:
        VJPDual: Primal and residuals.
    """
    primal = prims.amax(x, dims)

    residuals = (
        primal,
        x,
        x.shape,
        dims,
    )

    return VJPDual(primal, residuals)


@register_augmented_forward(prims.PrimIDs.AMIN)
def amin_aug_fwd(x, dims):
    """Augmented amin operation.

    Args:
        x (Variable): Tensor to compute amin on.
        dims (Tuple[int, ...]): Dimensions to compute amin over.

    Returns:
        VJPDual: Primal and residuals.
    """
    primal = prims.amin(x, dims)

    residuals = (
        primal,
        x,
        x.shape,
        dims,
    )

    return VJPDual(primal, residuals)


@register_augmented_forward(prims.PrimIDs.EXP)
def exp_aug_fwd(x):
    """Augmented exp operation.

    Args:
        x (Variable): Tensor to be exponentiated.

    Returns:
        VJPDual: Primal and residuals.
    """
    primal = prims.exp(x)
    residuals = (primal,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.EXP)
def exp_backward(result, g):
    return g * result


@register_augmented_forward(prims.PrimIDs.POW)
def pow_aug_fed(x, y):
    """Augmented the pow operation.

    Args:
        x (Variable): Tensor with the base to be exponentiated.
        y (Variable): Tensor with power to raise to.

    Returns:
        VJPDual: Primal and residuals.
    """
    primal = prims.pow(x, y)
    residuals = (primal, x, y)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.POW)
def pow_backward(result, x, y, g):
    import thunder.clang as tlang

    gresult = g * result  # reuse common factor
    dx = g * y * x ** (y - 1)
    dy = gresult * tlang.log(x)
    return dx, dy


@register_augmented_forward(prims.PrimIDs.TAN)
def tan_aug_fwd(x):
    """Augmented tan operation.

    Args:
        x (Variable): Tensor to be passed to tan.
    """

    primal = prims.tan(x)
    residuals = (primal,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.TAN)
def tan_backward(result, g):
    return g * (1 + result * result)


@register_augmented_forward(prims.PrimIDs.TANH)
def tanh_aug_fwd(x):
    """Augmented tanh operation.

    Args:
        x (Variable): Tensor to be passed to tanh.

    Returns:
        VJPDual: Primal and residuals.
    """
    primal = prims.tanh(x)
    residuals = (primal,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.TANH)
def tanh_backward(result, g):
    return g * (1.0 - result * result)


# NOTE: Jax uses np.argsort in its transpose vjp computation
def _argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


@register_augmented_forward(prims.PrimIDs.TRANSPOSE)
def transpose_aug_fwd(a, permutation):
    primal = prims.transpose(a, permutation)
    residuals = (permutation,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.TRANSPOSE)
def transpose_backward(permutation, g):
    undo = _argsort(permutation)
    return prims.transpose(g, undo), None


@register_augmented_forward(prims.PrimIDs.RESHAPE)
def reshape_aug_fwd(a, shape):
    primal = prims.reshape(a, shape)
    residuals = (a.shape,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.RESHAPE)
def reshape_backward(orig_shape, g):
    return prims.reshape(g, orig_shape), None


@register_augmented_forward(prims.PrimIDs.SLICE)
def slice_aug_fwd(a, start_indices, end_indices, strides):
    primal = prims.slice_prim(a, start_indices, end_indices, strides)
    residuals = (a.shape, start_indices, end_indices, strides)
    return VJPDual(primal, residuals)


# Adapted from https://github.com/google/jax/blob/main/jax/_src/lax/slicing.py#L768
@register_backward(prims.PrimIDs.SLICE)
def pullback(shape, start_indices, end_indices, strides, g):
    padding = None
    if strides is None or np.all(np.equal(strides, 1)):
        padding = tuple(zip(start_indices, np.subtract(shape, end_indices), (0,) * len(start_indices)))
    else:
        real_limits = np.add(
            start_indices,
            np.where(np.equal(g.shape, 0), 0, np.add(1, np.multiply(np.subtract(g.shape, 1), strides))),
        )
        padding = tuple(zip(start_indices, np.subtract(shape, real_limits), np.subtract(strides, 1)))

    # We used NumPy arithmetics above, but the current infra expects Python ints.
    padding = tree_map(int, padding)
    result = prims.pad(g, const_as(0, g.dtype), padding)

    return result, None, None, None


@register_augmented_forward(prims.PrimIDs.BROADCAST_IN_DIM)
def broadcast_in_dim_aug_fwd(a: Proxy, shape: Sequence[int], broadcast_dimensions: Sequence[int]) -> VJPDual:
    primal = prims.broadcast_in_dim(a, shape, broadcast_dimensions)
    residuals = (a, shape, broadcast_dimensions)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.BROADCAST_IN_DIM)
def broadcast_in_dim_backward(a, shape, broadcast_dimensions, g):
    from thunder.torch import sum

    # If g is None, then the primal was a constant and the pullback is zero.
    # TODO: implement None propagation in the VJP infrastructure so that we don't need to do this.
    if g is None:
        return None, None, None
    unit_dims = tuple(i for i, s in enumerate(a.shape) if s == 1)
    bcast_dims = tuple(b for i, b in enumerate(broadcast_dimensions) if i not in unit_dims)
    reduce_dims = tuple(s for i, s in enumerate(range(len(shape))) if i not in bcast_dims)
    g = sum(g, reduce_dims)
    g = unsqueeze(g, unit_dims)
    # One return per positional argument of prims.broadcast_in_dim
    return g, None, None


@register_augmented_forward(prims.PrimIDs.CONVERT_ELEMENT_TYPE)
def convert_element_type_aug_fwd(a: Proxy, dtype: dtypes.dtype) -> VJPDual:
    primal = prims.convert_element_type(a, dtype)
    residuals = (a.dtype if isinstance(a, TensorProxy) else (a.python_type if isinstance(a, NumberProxy) else type(a)),)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.CONVERT_ELEMENT_TYPE)
def convert_element_type_backward(a_dtype, g):
    # perform cast back to input type during backward
    return prims.convert_element_type(g, a_dtype), None


@register_augmented_forward("torch.nn.functional.cross_entropy")
def cross_entropy_aug_fwd(
    input: Proxy,
    target: Proxy,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
) -> VJPDual:
    from thunder.torch import cross_entropy

    primal = cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)
    residuals = (input, target, weight, reduction, ignore_index, label_smoothing)
    return VJPDual(primal, residuals)


@register_backward("torch.nn.functional.cross_entropy")
def cross_entropy_backward(input, target, weight, reduction, ignore_index, label_smoothing, g):
    from thunder.torch import cross_entropy_backward

    ginput = cross_entropy_backward(g, input, target, weight, reduction, ignore_index, label_smoothing)
    return ginput, *((None,) * 7)


@register_augmented_forward("torch.nn.functional.embedding")
def embedding_aug_fwd(
    a: Proxy,
    weight: Proxy,
    padding_idx: Optional[int],
    max_norm: Optional[float],
    norm_type: float,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> VJPDual:
    from thunder.torch import embedding

    primal = embedding(
        a,
        weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )
    residuals = (a, weight.shape[0], padding_idx, scale_grad_by_freq, sparse)
    return VJPDual(primal, residuals)


@register_backward("torch.nn.functional.embedding")
def embedding_backward(a, num_weights, padding_idx, scale_grad_by_freq, sparse, g):
    from thunder.torch import embedding_backward

    padding_idx = -1 if padding_idx is None else padding_idx
    gweight = embedding_backward(g, a, num_weights, padding_idx, scale_grad_by_freq, sparse)
    return None, gweight, None, None, None, None, None


@register_augmented_forward("torch.softmax")
def softmax_aug_fwd(a: Proxy, dim: int, dtype: Optional[dtypes.dtype] = None) -> VJPDual:
    from thunder.torch import softmax

    primal = softmax(a, dim, dtype=dtype)
    residuals = (primal, dim)
    return VJPDual(primal, residuals)


@register_backward("torch.softmax")
def softmax_backward(primal, dim, g):
    return primal * (g - (primal * g).sum(dim, keepdim=True)), None, None


@register_augmented_forward(prims.PrimIDs.MATMUL)
def matmul_aug_fwd(a: TensorProxy, b: TensorProxy) -> VJPDual:
    primal = prims.matmul(a, b)
    residuals = (a, b)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.MATMUL)
def matmul_backward(a, b, g):
    from thunder.torch import sum

    last_dim = (-1,)
    first_dim = (-2,)
    if a.ndim == 1 and b.ndim == 1:
        return g * b, g * a

    if b.ndim == 1:
        ga = unsqueeze(g, last_dim) @ unsqueeze(b, last_dim).mT
        gb = a.mT @ unsqueeze(g, last_dim)
        if g.ndim > 1:
            gb = squeeze(gb, last_dim)
            gb = sum(gb, tuple(range(gb.ndim - 1)))
        return ga, gb

    if a.ndim == 1:
        ga = unsqueeze(g, first_dim) @ b.mT
        if g.ndim > 1:
            ga = sum(ga, tuple(range(ga.ndim - 1)))
        gb = unsqueeze(a, first_dim).mT @ unsqueeze(g, first_dim)
        return ga, gb

    return g @ b.mT, a.mT @ g


@register_augmented_forward(prims.PrimIDs.LINEAR)
def linear_aug_fwd(a: TensorProxy, b: TensorProxy, c: Optional[TensorProxy]) -> VJPDual:
    primal = prims.linear(a, b, c)
    residuals = (a, b, c)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.LINEAR)
def linear_backward(a, b, c, g):
    from thunder.torch import matmul, sum

    first_dim = (-2,)
    ga = matmul(g.reshape(-1, g.shape[-1]), b).reshape(a.shape)
    if a.ndim == 1:
        gb = matmul(unsqueeze(g, first_dim).mT, unsqueeze(a, first_dim))
    else:
        gb = matmul(g.reshape(-1, g.shape[-1]).mT, a.reshape(-1, a.shape[-1]))
        assert list(gb.shape) == list(b.shape), f"linear_backward: {gb.shape} != {b.shape}"
    if c is None:
        return ga, gb, None
    gc = sum(g, tuple(range(g.ndim - 1))) if g.ndim > 1 else g
    return ga, gb, gc


def iter_bound_symbols(bound_symbols):
    """Iterate over bound symbols, skipping symbols that are not supported by
    the transforms infrastructure.

    Args:
        bound_symbols (List[BoundSymbol]): List of bound symbols

    Yields:
        BoundSymbol: Bound symbols that are supported by the transforms
        infrastructure
    """
    for symbol in bound_symbols:
        if symbol.sym.id in transform_skip_list:
            continue
        else:
            yield symbol


def deconstruct_forward_env_for_backward(trace, env):
    # Note [Saving the forward environment in the backward rule]
    # We cannot save the trace object in the residuals because executors may not
    # be able to return it for the splitted forward/backward passes. Instead, we
    # save args and kwargs of the original function call and reconstruct the
    # trace object and its environment dict in the backward rule. Here we rely
    # on the fact that the order of the symbols in the trace is deterministic
    # and always the same for the same function call and the same set of
    # arguments. See test_grad.py:test_torch_autograd_function for an example
    # where this is tested.
    bound_symbols = iter_bound_symbols(trace.bound_symbols)
    saved_for_backward = tuple(env[sequencify(symbol.output)[0].name].residuals for symbol in bound_symbols)
    return saved_for_backward


def reconstruct_forward_env_for_backward(trace, saved_for_backward):
    bound_symbols = iter_bound_symbols(trace.bound_symbols)
    reconstructed_env = {
        sequencify(symbol.output)[0].name: VJPDual(None, saved_for_backward[i])
        for i, symbol in enumerate(bound_symbols)
    }
    return reconstructed_env


def decomposed_fn_aug_fwd_rule(*args, decomposed_fn, **kwargs):
    """Augmented forward rule for composite functions implemented in terms of other functions that are
    supposed to be supported by the VJP infrastructure.

    Args:
        decomposed_fn (Callable): decomposed version of the function

    Returns:
        Callable: Augmented forward rule for the composite function
    """
    trace = construct_trace()(decomposed_fn, *args, **kwargs)
    trace = unwrap_one_level_of_subsymbols(trace)
    # There may be a dead node like "_ = prims.convert_element_type(0, float)"
    # in the trace. We need to remove it before we can use the trace for
    # augmented_forward_pass.
    trace = dce(trace)[0]
    result, env = augmented_forward_pass(*args, trace=trace, **kwargs)
    saved_for_backward = deconstruct_forward_env_for_backward(trace, env)
    # Static caching does not with with kwargs dicts, so we're converting them
    # to None for now when possible.
    kwargs = None if not kwargs else kwargs
    residuals = (args, kwargs, saved_for_backward)
    return VJPDual(result, residuals)


def decomposed_fn_backward_rule(decomposed_fn, args, kwargs, saved_for_backward, *grads):
    kwargs = {} if kwargs is None else kwargs
    trace = construct_trace()(decomposed_fn, *args, **kwargs)
    trace = unwrap_one_level_of_subsymbols(trace)
    trace = dce(trace)[0]
    # bound_symbols = iter_bound_symbols(trace.bound_symbols)
    # reconstructed_env = {
    #     sequencify(symbol.output)[0].name: VJPDual(None, saved_for_backward[i])
    #     for i, symbol in enumerate(bound_symbols)
    # }
    reconstructed_env = reconstruct_forward_env_for_backward(trace, saved_for_backward)
    result = backward_pass(reconstructed_env, trace, grads)
    if len(args) == 1:
        return result[0]
    # Backward pass might return a dict with grads but current interface of
    # backward rule does not support it. So we just drop it for now.
    elif isinstance(result[-1], dict):
        return result[:-1]
    return result


@register_augmented_forward(prims.PrimIDs.CAT)
def cat_aug_fwd(tensors: list[TensorProxy], dim: int) -> VJPDual:
    primal = prims.cat(tensors, dim)
    residuals = (
        type(tensors),
        [t.shape[dim] for t in tensors],
        dim,
    )

    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.CAT)
def cat_backward(
    tensors_seq_type: Sequence, tensor_dim_lens: list[int], dim: int, g: TensorProxy
) -> (list[TensorProxy], None):
    grads = []

    slice_start = 0
    for dim_len in tensor_dim_lens:
        grads.append(slice_in_dim(g, slice_start, slice_start + dim_len, dim=dim))
        slice_start += dim_len
    return tensors_seq_type(grads), None


@register_augmented_forward("torch.Tensor.contiguous")
@register_augmented_forward("torch.contiguous")
def contiguous_aug_fwd(x: TensorProxy, /, *, memory_format: torch.memory_format = torch.contiguous_format) -> VJPDual:
    from thunder.torch import contiguous

    return VJPDual(contiguous(x, memory_format=memory_format), tuple())


@register_backward("torch.Tensor.contiguous")
@register_backward("torch.contiguous")
def contiguous_backward(*residuals_and_grad) -> TensorProxy:
    # Residuals is not empty because contiguous symbol has the same output as its input
    g = residuals_and_grad[-1]
    return g * 1.0


@register_augmented_forward(prims.PrimIDs.WHERE)
def where_aug_fwd(condition: TensorProxy, x: TensorProxy, y: TensorProxy) -> VJPDual:
    primal = prims.where(condition, x, y)
    residuals = (condition,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.WHERE)
def where_backward(condition, g):
    return None, prims.where(condition, g, 0.0), prims.where(condition, 0.0, g)


@register_augmented_forward(prims.PrimIDs.RECIPROCAL)
def reciprocal_aug_fwd(a: TensorProxy) -> VJPDual:
    primal = reciprocal(a)
    return VJPDual(primal, (primal,))


@register_backward(prims.PrimIDs.RECIPROCAL)
def reciprocal_backward(primal, g):
    return -g * primal * primal


@register_augmented_forward(prims.PrimIDs.SQUEEZE)
def squeeze_aug_fwd(a: TensorProxy, dims: Sequence[int]) -> VJPDual:
    primal = squeeze(a, dims)
    residuals = (dims,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.SQUEEZE)
def squeeze_backward(dims: Sequence[int], g: TensorProxy) -> (TensorProxy, None):
    return unsqueeze(g, dims), None


@register_augmented_forward(prims.PrimIDs.TAKE)
def take_aug_fwd(x: TensorProxy, index: TensorProxy, dim: int) -> VJPDual:
    primal = prims.take(x, index, dim)
    residuals = (
        x.shape,
        x.device,
        x.dtype,
        index,
        dim,
    )
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.TAKE)
def take_backward(
    shape: Sequence[int], device: Device, dtype: dtypes.dtype, index: TensorProxy, dim: int, g: TensorProxy
):
    return prims.index_add(prims.full(shape, fill_value=0, device=device, dtype=dtype), index, g, dim), None, None


@register_augmented_forward(prims.PrimIDs.TAKE_ALONG_AXIS)
def take_along_axis_aug_fwd(x: TensorProxy, index: TensorProxy, dim: int) -> VJPDual:
    primal = prims.take_along_axis(x, index, dim)
    residuals = (
        x.shape,
        x.device,
        x.dtype,
        index,
        dim,
    )
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.TAKE_ALONG_AXIS)
def take_along_axis_backward(
    shape: Sequence[int], device: Device, dtype: dtypes.dtype, index: TensorProxy, dim: int, g: TensorProxy
):
    return prims.scatter_add(prims.full(shape, fill_value=0, device=device, dtype=dtype), index, g, dim), None, None


@register_augmented_forward(prims.PrimIDs.UNIFORM)
def uniform_aug_fwd(shape, minval, maxval, *, device, dtype):
    primal = prims.uniform(shape, minval, maxval, device=device, dtype=dtype)
    return VJPDual(primal, (primal, minval, maxval))


@register_backward(prims.PrimIDs.UNIFORM)
def uniform_backward(primal, minval, maxval, g):
    # uniform is implemented as (maxval - minval) * uniform(shape, 0, 1) + minval
    unscaled_primal = (primal - minval) / (maxval - minval)
    reduce_all_dims = tuple(range(g.ndim))
    sum = partial(prims.sum, dims=reduce_all_dims)
    return None, sum(g * (1 - unscaled_primal)), sum(g * unscaled_primal)


nondifferentiable_vjp_symbols = (prims.PrimIDs.BITWISE_AND, prims.PrimIDs.FULL)


def vjp_symbol_mapper(symbol: prims.Symbol, *args, **kwargs):
    """Symbol mapper for the VJP transform.

    Args:
        symbol (prims.Symbol): Symbol to be mapped.
        args (Tuple[Variable]): Arguments to the symbol.
        kwargs (Dict[str, Variable]): Keyword arguments to the symbol.

    Returns:
        Callable: A function that computes the VJP of the symbol.
    """
    # Constant case
    if symbol.are_all_args_constant or symbol.sym.id in nondifferentiable_vjp_symbols:

        def vjp_impl_const(symbol, *args, **kwargs):
            args, kwargs = tree_map(lambda x: x.primal if isinstance(x, VJPDual) else x, (args, kwargs))
            primals = symbol_to_eval(symbol)(*args, **kwargs)
            if isinstance(primals, Sequence):
                return tree_map(lambda x: VJPDual(x, tuple()), primals)
            return VJPDual(primals, tuple())

        return partial(vjp_impl_const, symbol)

    # Normal case, we have a proxy tangent
    vjp_impl = augmented_forward_impls.get(symbol.sym.id)
    if vjp_impl is None:
        # We could not find a VJP for this symbol, so we try to decompose it
        if len(symbol.subsymbols) > 0 and not isinstance(symbol.sym.id, prims.PrimIDs):
            vjp_impl = partial(decomposed_fn_aug_fwd_rule, decomposed_fn=symbol.sym.__call__)
            register_backward(symbol.sym.id)(partial(decomposed_fn_backward_rule, symbol.sym.__call__))
        else:
            # We could not find a VJP for this symbol and we could not decompose it
            # It could be a torch.dropout with 0.0 probability, so we skip it
            if symbol.sym.id == "torch.nn.functional.dropout":
                return None
            raise NotImplementedError(f"VJP for {symbol.sym.id} is not implemented")

    def _vjp_impl(*args, **kwargs):
        primals, kwargs = tree_map(lambda x: x.primal if isinstance(x, VJPDual) else x, (args, kwargs))
        out_primal, out_residuals = vjp_impl(*primals, **kwargs)

        # We are saving the residuals and pullback only in the first output
        # backward_pass then retrieves the residuals and pullback from the first output
        if isinstance(out_primal, Sequence):
            return (VJPDual(out_primal[0], out_residuals), *(VJPDual(o, tuple()) for o in out_primal[1:]))

        return (VJPDual(out_primal, out_residuals),)

    return _vjp_impl


def augmented_forward_pass(*args, trace: Trace, **kwargs):
    """Augmented forward pass for the VJP transform.

    The augmented forward pass is a forward pass that returns the residuals
    of the forward pass.
    These residuals are used in the backward pass to compute the VJP and they
    are recorded in the environment dictionary for each variable.

    Args:
        args (Tuple[Variable]): Arguments to the function.
        trace (Trace): Trace of the function.
        kwargs (Dict[str, Variable]): Keyword arguments to the function.

    Returns:
        Tuple[Any, Dict[str, Any]]: Tuple of the primal outputs and the environment.
    """
    args, kwargs = tree_map(lambda x: VJPDual(x, tuple()), (args, kwargs))
    result, env = eval_trace(
        trace,
        *args,
        **kwargs,
        with_env=True,
        symbol_mapper=vjp_symbol_mapper,
    )
    result = tree_map(lambda x: x.primal if isinstance(x, VJPDual) else x, result)
    return result, env


# TODO: Instead of using the environment dictionary, we could use the trace
# symbols order (that should be deterministic) to retrieve the residuals needed
# for the backward pass.
def backward_pass(forward_env, trace, init_cotangents):
    """Backward pass for the VJP transform.

    The backward pass is a reverse mode automatic differentiation pass that
    computes the vector-Jacobian product (VJP) of the function.

    Args:
        forward_env (Dict[str, Any]): Environment of the forward pass.
        trace (Trace): Trace of the function.
        init_cotangents (Tuple[Variable]): Initial cotangents.

    Returns:
        Tuple[Proxy, ...]: Tuple of the results of the backward pass for each input.
    """
    env = {}

    def read_with_none(x: Variable):
        if isinstance(x, Variable):
            # Return None if the variable was not used in the computation and
            # hence not in the env
            return env.get(x.name, None)
        else:
            return x

    def write(v: Variable, val: Any) -> None:
        if isinstance(v, Variable):
            if v.name in env:
                if val is None:
                    return
                # Accumulate cotangents
                env[v.name] = clang.add(env[v.name], val) if env[v.name] is not None else val
                return
            env[v.name] = val
        elif isinstance(v, Sequence) and all(isinstance(x, int) for x in v):
            # TODO: remove when we move dims to kwargs
            pass
        elif isinstance(v, Sequence) and val is None:
            # broadcast None to the right shape
            safe_map(write, v, [0] * len(v))
        else:
            # Skip writing to constants
            pass

    if isinstance(init_cotangents, Sequence) and len(init_cotangents) == 1 and not isinstance(trace.output, Sequence):
        init_cotangents = init_cotangents[0]
    safe_map_flat(write, trace.output, init_cotangents)

    for symbol in reversed(trace.bound_symbols):
        if symbol.sym.id in transform_skip_list:
            continue
        symbol_output = sequencify(symbol.output)
        cotangents = safe_map(read_with_none, symbol_output)
        # Having a single cotangent is a common case, so we flatten it
        # Otherwise, we will need to rewrite the pullback functions
        cotangents = tree_flatten(cotangents)[0]
        residuals = forward_env[symbol_output[0].name].residuals
        if symbol.are_all_args_constant or symbol.sym.id in nondifferentiable_vjp_symbols:
            # We can skip the pullback if all the arguments are constant
            continue

        if len(cotangents) == 1 and cotangents[0] is None:
            # We can skip the pullback if the cotangent is None
            safe_map(write, symbol.args, (None,) * len(symbol.args))
            continue

        if symbol.sym.id == "torch.nn.functional.dropout" and not symbol.subsymbols:
            # We can skip the pullback if the dropout probability is 0.0
            # Assuming that the dropout symbol has the same output and argument
            # https://github.com/Lightning-AI/lightning-thunder/issues/906
            assert symbol.output.name == symbol.args[0].name, "Dropout symbol has a different output and argument"
            if symbol.args[1] == 0.0 or symbol.args[2] is False:
                continue

        pullback = backward_impls[symbol.sym.id]
        result = pullback(*residuals, *cotangents)
        if not isinstance(result, Sequence):
            result = (result,)
        check(
            len(result) == len(symbol.args),
            lambda: f"Pullback for {symbol.sym.id} returned {len(result)} values, but expected {len(symbol.args)}",
        )

        # See https://github.com/Lightning-AI/lightning-thunder/issues/977.
        # This is a temporary workaround.
        if symbol.sym.id in (prims.PrimIDs.CAT, "torch.cat"):
            safe_map_flat(write, symbol.args, result)
        else:
            safe_map(write, symbol.args, result)

    gargs = tree_map(read_with_none, tuple(trace.args))
    gkwargs = tree_map(read_with_none, trace.kwargs)
    gkwargs = {k: v for k, v in gkwargs.items() if v is not None}
    return gargs + (gkwargs,) if len(gkwargs) != 0 else gargs


def vjp_call_metafunc(detached: bool, primals, cotangents, trace: Trace, **kwargs):
    # Assuming primals is flat

    if not isinstance(primals, Sequence):
        primals = (primals,)

    ctx = detached_trace() if detached else nullcontext()
    with ctx:
        result, env = augmented_forward_pass(*primals, trace=trace, **kwargs)
        check(
            len(result) == len(cotangents) if isinstance(result, Sequence) else True,
            lambda: f"Expected cotangents to be a sequence of length {len(result)}, got a sequence of length {len(cotangents)}",
        )
        return result, backward_pass(env, trace, cotangents)


vjp_call = make_transform_prim(Transforms.VjpOp, "vjp_call", meta=partial(vjp_call_metafunc, True))
inline_transforms_map[Transforms.VjpOp] = partial(vjp_call_metafunc, False)


def vjp(func):
    """Computes the VJP of a function.

    Args:
        func (Callable): Function to be differentiated.
    """

    def _vjp(primals, cotangents, **kwargs):
        flat_func, flat_args, spec = flatten_func(func, primals, kwargs)
        trace = construct_trace()(flat_func, *flat_args)
        result, vjp_result = vjp_call(flat_args, cotangents, trace=trace)
        gprimals, gkwargs = tree_unflatten(vjp_result, spec)
        grads = gprimals + (gkwargs,) if len(gkwargs) != 0 else gprimals
        return result, grads

    return _vjp


def value_and_grad(func):
    """Computes the value and gradient of a function.

    This is a convenience function that combines the functionality of
    `vjp_call` with implicit initialization of the cotangent to 1.

    Args:
        func (Callable): Function to be differentiated.
    """

    def ones_like(x):
        if isinstance(x, TensorProxy):
            return full_like(x, fill_value=1)
        elif isinstance(x, NumberProxy):
            return type(x.value)(1)
        else:
            raise ValueError(f"ones_like inside value_and_grad got an unsupported type {type(x)}")

    def _value_and_grad(*args, **kwargs):
        trace = construct_trace()(func, *args, **kwargs)
        cotangents = tree_map(lambda v: ones_like(v), trace.output)
        return vjp(func)(args, cotangents, **kwargs)

    return _value_and_grad


ForwardBackwardTraces = namedtuple("ForwardBackwardTraces", ["forward_trace", "backward_trace"])


def _split_saved_for_backward_into_tensors_and_other(
    saved_for_backward: Sequence[Variable],
) -> tuple[Sequence[Variable], Sequence[Variable]]:
    """Splits saved_for_backward into tensors and other.

    Args:
        saved_for_backward (Sequence[Variable]): Saved_for_backward to split.

    Returns:
        tuple[Sequence[Variable], Sequence[Variable]]: Tuple of tensors and other.
    """
    is_tensor = lambda x: isinstance(x, TensorProxy)
    other, tensors = utils.partition(is_tensor, saved_for_backward)
    return tuple(tensors), tuple(other)


def _update_forward_with_new_saved_for_backward(forward_trace: Trace, saved_for_backward: Sequence[Variable]) -> None:
    """Updates the forward trace with new saved_for_backward.

    This is necessary because the updated saved_for_backward is not available
    when the forward and backward traces are constructed.

    Args:
        forward_trace (Trace): Forward trace to update.
        saved_for_backward (Sequence[Variable]): Saved_for_backward to use to
            update the forward trace.
    """
    saved_for_backward = tree_map(lambda x: x.value if isinstance(x, NumberProxy) else x, saved_for_backward)
    saved_tensors, saved_other = _split_saved_for_backward_into_tensors_and_other(saved_for_backward)
    forward_return_bsym = next(x for x in reversed(forward_trace.bound_symbols) if x.sym.id == prims.PrimIDs.RETURN)
    forward_return_bsym.args = (forward_trace.output[0], (saved_tensors, saved_other))
    forward_trace.output = forward_return_bsym.args


def _update_backward_with_new_saved_for_backward(backward_trace: Trace, saved_for_backward: Sequence[Variable]) -> None:
    """Updates the backward trace with new saved_for_backward.

    This is necessary because the updated saved_for_backward is
    not available when the backward trace is constructed.

    Args:
        backward_trace (Trace): Backward trace to update.
        saved_for_backward (Sequence[Variable]): Saved_for_backward to use to
            update the backward trace.
    """

    def unpacking_fn(saved_for_backward, cotangents):
        pass

    cotangents = backward_trace.args[1]
    saved_tensors, saved_other = _split_saved_for_backward_into_tensors_and_other(saved_for_backward)
    unpacking_trace = construct_trace(rename_proxies=False)(unpacking_fn, (saved_tensors, saved_other), cotangents)
    assert unpacking_trace.bound_symbols[-1].sym.id == prims.PrimIDs.RETURN

    backward_trace.args = unpacking_trace.args
    backward_trace_bsyms_without_unpacking = (
        bsym
        for bsym in backward_trace.bound_symbols
        if bsym.sym.id
        not in (
            prims.PrimIDs.UNPACK_EMPTY_DICT,
            prims.PrimIDs.UNPACK_KEY,
            prims.PrimIDs.UNPACK_SEQUENCE,
            prims.PrimIDs.UNPACK_TRIVIAL,
        )
    )
    backward_trace.bound_symbols = tuple((*unpacking_trace.bound_symbols[:-1], *backward_trace_bsyms_without_unpacking))


# NOTE: Returning namedtuples from compiled functions doesn't work. See:
# https://github.com/Lightning-AI/lightning-thunder/issues/881
# Note [Grad forward output spec]
# If it did work it would be nice to use this namedtuple
# instead of the plain tuple or dict that we're using now.
TorchAutogradForwardData = namedtuple(
    "TorchAutogradForwardData",
    ["output", "flat_args", "flat_output"],
)


def forward_and_backward_from_trace(trace: Trace, torch_autograd=False) -> ForwardBackwardTraces:
    """Generates the forward and backward passes from a trace.

    This is a convenience function that combines the functionality of
    `augmented_forward_pass` and `backward_pass`. The main difference is that
    this function does not require the user to provide new inputs for the
    trace evaluation. Instead it uses the inputs that were used to construct
    the trace.

    Args:
        trace (Trace): Trace to generate the forward and backward passes from.

    Returns:
        ForwardBackwardTraces: A named tuple containing the forward and backward
            traces.

    Example:
        >>> import torch
        >>> from thunder import compile, last_traces
        >>> from thunder.core.transforms import forward_and_backward_from_trace
        >>> def f(x):
        ...     return torch.sin(x)
        >>> x = torch.tensor(3.0)
        >>> cf = compile(f)
        >>> out = cf(x)
        >>> trace = last_traces(cf)[0]
        >>> forward_and_backward_from_trace(trace)
        ... ForwardBackwardTraces(
        ... forward_trace=# import thunder as thunder
        ... # import thunder.core.prims as prims
        ... import torch
        ...
        ... @torch.no_grad()
        ... def augmented_forward_fn(*args):
        ...   # args: "Collection" =  (t0,)  (trivial unpack)
        ...   t0, \
        ...   = args
        ...   t1 = prims.sin(t0)  # t1: "cpu f32[]"
        ...   return t1, (t0,),
        ... backward_trace=# import thunder as thunder
        ... # import thunder.core.prims as prims
        ... import torch
        ...
        ... @torch.no_grad()
        ... def backward_fn(saved_for_backward, cotangents):
        ...   # saved_for_backward: "Collection" =  (t0,)  (trivial unpack)
        ...   # cotangents: "cpu f32[]" =  cotangents  (trivial unpack)
        ...   t0, \
        ...   = saved_for_backward
        ...   t1 = prims.cos(t0)  # t1: "cpu f32[]"
        ...   t2 = prims.mul(cotangents, t1)  # t2: "cpu f32[]"
        ...   return (t2,))
    """

    output_spec = None

    def augmented_forward_fn(*args, **kwargs):
        result, env = augmented_forward_pass(*args, trace=trace, **kwargs)
        saved_for_backward = deconstruct_forward_env_for_backward(trace, env)
        if torch_autograd:
            nonlocal output_spec
            flat_args, _ = tree_flatten((args, kwargs))
            flat_output, output_spec = tree_flatten(result)
            flat_output = tuple(flat_output)
            # See Note [Grad forward output spec]
            for_autograd = TorchAutogradForwardData(
                result,
                flat_args,
                flat_output,
            )._asdict()
            return (for_autograd, saved_for_backward)
        return result, saved_for_backward

    # Copy the signature of the original function so that the arguments are
    # named correctly in the augmented forward pass instead of being named
    # "args" and "kwargs".
    augmented_forward_fn.__signature__ = inspect.signature(trace.fn)

    def ones_like(x):
        if isinstance(x, TensorProxy):
            return full_like(x, fill_value=1)
        elif isinstance(x, NumberProxy):
            return type(x.value)(1)
        else:
            return None

    forward_trace = construct_trace()(augmented_forward_fn, *trace.args, **trace.kwargs)
    # We set forward trace to construct proxies because we need these proxies to
    # have different names than the ones in the forward trace.
    try:
        tracectx_token = set_tracectx(forward_trace)
        # We don't want to record those ones_like calls in the forward trace.
        with detached_trace():
            if torch_autograd:
                # It's assumed that forward_trace.output[0] is a dict from TorchAutogradForwardData
                flat_output = forward_trace.output[0]["flat_output"]
                cotangents = utils.sequencify(tree_map(lambda v: ones_like(v), flat_output))
            else:
                cotangents = utils.sequencify(tree_map(lambda v: ones_like(v), trace.output))
    finally:
        reset_tracectx(tracectx_token)

    def backward_fn(saved_for_backward, cotangents):
        env = reconstruct_forward_env_for_backward(trace, saved_for_backward)
        if torch_autograd:
            cotangents = tree_unflatten(cotangents, output_spec)
        out = backward_pass(env, trace, cotangents)
        if torch_autograd:
            out = tree_flatten(out)[0]
        return out

    saved_for_backward = forward_trace.output[1]
    backward_trace = construct_trace(rename_proxies=False)(backward_fn, saved_for_backward, cotangents)

    # We are done with constructing the forward and backward passes at this
    # stage. The following is not strictly necessary, but it's good to filter
    # out the unused elements of the saved_for_backward and flatten it for more
    # compact backward trace.

    # Now we can determine exactly what's used in the backward pass from the
    # saved_for_backward. We can flatten and filter the saved_for_backward
    consumers = utils.consumers(backward_trace)

    # Forward's and backward's "saved_for_backward" are not necessarily the same
    # as the saved_for_backward, because some or all elements of the
    # saved_for_backward results might be re-proxified.
    bw_flat_saved_for_backward, spec = tree_flatten(backward_trace.args[0])
    fw_flat_saved_for_backward, _ = tree_flatten(forward_trace.output[1])
    used_mask = list((len(consumers.get(x, ())) > 0 for x in bw_flat_saved_for_backward))

    # Don't use the same variable twice in the backward pass
    seen = set()
    from thunder.core.proxies import Variable

    for i, x in enumerate(fw_flat_saved_for_backward):
        x = variableify(x)
        if not isinstance(x, Variable):
            continue
        if x in seen:
            used_mask[i] = False
        else:
            seen.add(x)

    only_used_fw_saved_for_backward = tuple(compress(fw_flat_saved_for_backward, used_mask))
    only_used_bw_saved_for_backward = tuple(compress(bw_flat_saved_for_backward, used_mask))

    # We need to update the traces with the new saved_for_backward
    _update_forward_with_new_saved_for_backward(forward_trace, only_used_fw_saved_for_backward)
    _update_backward_with_new_saved_for_backward(backward_trace, only_used_bw_saved_for_backward)
    forward_trace.set_provenance(TraceProvenance("Augmented forward pass"))
    backward_trace.set_provenance(TraceProvenance("Backward pass"))
    return ForwardBackwardTraces(forward_trace, backward_trace)


# do we happen to want to register `ltorch` ops such as `ltorch.layer_norm` as well?
autocast_impls: dict[prims.PrimIDs, Callable] = {}


def register_autocast_rule(op):
    def decorator(func):
        autocast_impls[op] = func
        return func

    return decorator


def maybe_downcast_to(dtype, args):
    allowed_downcast_types = (dtypes.float16, dtypes.bfloat16, dtypes.float32)
    if all(tree_map(lambda a: a.dtype in allowed_downcast_types, args)):
        return tree_map(lambda a: maybe_convert_to_dtype(a, dtype), args)
    else:
        return args


@register_autocast_rule(prims.PrimIDs.MATMUL)
def autocast_matmul_rule(a, b, dtype):
    """Autocast rule for matmul"""
    return prims.matmul(*(maybe_downcast_to(dtype, (a, b))))


@register_autocast_rule(prims.PrimIDs.LINEAR)
def autocast_linear_rule(a, w, bias, dtype):
    return prims.linear(*maybe_downcast_to(dtype, (a, w, bias)))


def decomposed_fn_autocast_rule(*args, fn, dtype, **kwargs):
    trace = construct_trace()(fn, *args, **kwargs)
    trace = unwrap_one_level_of_subsymbols(trace)
    return eval_trace(trace, *args, **kwargs, symbol_mapper=partial(autocast_symbol_mapper, dtype=dtype))


def autocast_symbol_mapper(bound_symbol: BoundSymbolInterface, dtype: dtypes.dtype):
    """Return the callable implementing the autocast rule for the symbol.

    Args:
        bound_symbol: Mapped to its autocast rule.

    Returns:
        Callable: The callable implementing the autocast rule for the symbol.
    """
    autocast_impl: Optional[Callable] = autocast_impls.get(bound_symbol.sym.id)
    if autocast_impl is None and bound_symbol.subsymbols:
        return partial(decomposed_fn_autocast_rule, fn=bound_symbol.sym.__call__, dtype=dtype)
    return bound_symbol.sym.__call__ if autocast_impl is None else partial(autocast_impl, dtype=dtype)


def autocast(func: Callable, dtype: dtypes.dtype):
    """Transforms a function to autocast certain operations.

    Args:
        func: The function to be transformed.
        dtype: The data type to which arguments of the function or sub functions could get cast if
            they are `dtypes.float32`.

    Returns:
        Callable: The transformed function
    """

    if not isinstance(dtype, dtypes.dtype):
        raise ValueError(f"`dtype` is expected to be `thunder.dtype.dtype` but {type(dtype)}")
    if dtype not in {dtypes.float16, dtypes.bfloat16}:
        raise ValueError(f"`dtype` is expected to be either `thunder.float16` or `thunder.bfloat16`, but {dtype}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        trace = construct_trace()(func, *args, **kwargs)
        return eval_trace(trace, *args, **kwargs, symbol_mapper=partial(autocast_symbol_mapper, dtype=dtype))

    return wrapper
