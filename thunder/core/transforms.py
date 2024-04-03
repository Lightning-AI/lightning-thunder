from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import auto, Enum
from itertools import chain, compress
from functools import lru_cache, partial, wraps
import math
from numbers import Number
from typing import Any, Dict, Union, Optional
from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Sequence
import copy
import inspect
import time
from collections import deque
import os

import thunder.core.utils as utils
from thunder.core import dtypes, prims
import thunder.core.devices as devices
from thunder.core.devices import cpu, Device
from thunder.core.proxies import (
    NumberProxy,
    Proxy,
    TensorProxy,
    FloatProxy,
    variableify,
    unvariableify,
    CollectionProxy,
    FutureTensorProxy,
)
from thunder.core.baseutils import default_dataclass_params
from thunder.core.compile_data import get_compile_data
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.symbol import BoundSymbol, BoundSymbolInterface, Symbol
from thunder.core.trace import TraceCtx as Trace, tracectx
from thunder.core.trace import VariableInterface as Variable
from thunder.core.trace import detached_trace, get_tracectx, set_tracectx, reset_tracectx, from_trace, TraceProvenance
from thunder.core.utils import (
    check,
    flatten_func,
    safe_map,
    safe_map_flat,
    safe_zip,
    unzip2,
    const_as,
    sequencify,
    ProxyDict,
)
import thunder.clang as clang
from thunder.clang import (
    full,
    full_like,
    unsqueeze,
    squeeze,
    maybe_convert_to_dtype,
    slice_in_dim,
    reciprocal,
    convolution,
)
from thunder.core.transform_common import dce
from thunder.core.vjp_utils import make_aug_forward_and_backward
from thunder.extend import Executor
import thunder.torch as ltorch

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
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
        self.number: None | int = None

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
#   - a list of all Nodes of bound symbols without children
# Note that nodes without parents or children may be in either list -- running a DCE pass
#   before toposorting should remove all nodes without children EXCEPT FOR the return node
def bsym_list_to_dag(
    bsyms: Sequence[BoundSymbolInterface], *, producers: None | ProxyDict = None, consumers: None | ProxyDict = None
) -> tuple[list[Node], list[Node]]:
    roots: list[Node] = []
    leaves: list[Node] = []
    return_node: None | Node = None

    # Note, we use "line number" as ids for consumers/producers
    producers = producers if producers is not None else utils.producers(bsyms, _map_to_numbers=True)
    consumers = consumers if consumers is not None else utils.consumers(bsyms, _map_to_numbers=True)

    # Constructs a node per bsym, and a bsym id -> node mapping
    bsym_id_to_node_map: dict[int, Node] = {}
    for bsym_id, bsym in enumerate(bsyms):
        node = Node(bsym)
        bsym_id_to_node_map[bsym_id] = node

        if bsym.sym.id is prims.PrimIDs.RETURN:
            utils.check(
                return_node is None,
                lambda: f"Found multiple RETURN nodes while converting a list of bound symbols to a dag",
            )
            return_node = node

    # Adds edges between nodes
    for bsym_id, node in bsym_id_to_node_map.items():
        has_parents: bool = False
        for inp in node.bsym.flat_proxy_args:
            producer = producers[inp]
            producer_node = bsym_id_to_node_map[producer]
            parent = bsym_id_to_node_map[producer]

            # Checks if the node was already parent to avoid multiple edges between two nodes
            already_has_parent: bool = False
            for pnode in node.parents:
                if producer_node.bsym is pnode.bsym:
                    already_has_parent = True
                    break

            if not already_has_parent:
                node.parents.append(parent)

            has_parents = True

        if not has_parents:
            roots.append(node)

        has_children: bool = False
        vargs = node.bsym.flat_variableified_proxy_args
        for out in node.bsym.flat_proxy_outs:
            # Checks that the output is actually produced by this function, and not an input to it
            vout = variableify(out)
            if vout in vargs:
                continue

            children = consumers.get(out, [])
            for child in children:
                has_children = True
                child_node = bsym_id_to_node_map[child]

                # Checks if the node was already a child to avoid multiple edges between two nodes
                already_has_child: bool = False
                for cnode in node.children:
                    if child_node.bsym is cnode.bsym:
                        already_has_child = True
                        break

                if not already_has_child:
                    node.children.append(child_node)

        if not has_children:
            leaves.append(node)

    return roots, leaves


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
    NO_OP = auto()


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
def visitor_transform(trace_from: Trace, visit: Callable, *, provenance: None | str = None) -> Trace:
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

                if visit_type is not VISIT_TYPE.NO_OP:
                    trc.bound_symbols.extend(scope)
                else:
                    trc.bound_symbols.append(bsym)

                if visit_type is VISIT_TYPE.INSERT_BEFORE:
                    trc.bound_symbols.append(bsym)

            finally:
                # Restores the trc's scope
                trc.scopes = old_scope

        if provenance is not None:
            trc.set_provenance(TraceProvenance(provenance))

        return trc

    finally:
        reset_tracectx(tracectx_tok)


#
# Composable transforms
#


# Helper function to add a transform
def add_transform(cfn: Callable, transform: Callable) -> Callable:
    from thunder.common import _create_callable, CompileData, CompileStats

    cd: None | Any = getattr(cfn, "_lc_cd", None)

    utils.check(cd is not None, lambda: f"Can only transform compiled thunder functions")
    utils.check(isinstance(cd, CompileData), lambda: f"Found an unknown compile data attribute {cd}")

    if cd.using_jit:
        from thunder import jit

        return jit(
            cd.fn,
            langctx=cd.langctx,
            executors=cd.executors_list,
            sharp_edges=cd.sharp_edges,
            # cache, interpretation?
            additional_transforms=cfn._lc_transforms + [transform],
            disable_torch_autograd=True,  # cd.disable_torch_autograd_support,
            **cd.compile_options,
        )

    cs = CompileStats()
    transforms = cfn._lc_transforms + [transform]
    potransforms = cfn._lc_post_optimization_transforms
    using_grad_transform = cfn._using_grad_transform

    ncfn = _create_callable(
        cd,
        cs,
        transforms=transforms,
        post_optimization_transforms=potransforms,
        _using_grad_transform=using_grad_transform,
    )
    return ncfn


# TODO Consider refactoring this with the above
# Helper function to add a post-optimization transform
def add_post_optimization_transform(cfn: Callable, transform: Callable) -> Callable:
    from thunder.common import _create_callable, CompileData, CompileStats

    cd: None | Any = getattr(cfn, "_lc_cd", None)

    utils.check(cd is not None, lambda: f"Can only transform compiled thunder functions")
    utils.check(isinstance(cd, CompileData), lambda: f"Found an unknown compile data attribute {cd}")

    cs = CompileStats()
    transforms = cfn._lc_transforms
    potransforms = cfn._lc_post_optimization_transforms + [transform]

    ncfn = _create_callable(cd, cs, transforms=transforms, post_optimization_transforms=potransforms)
    return ncfn


# The no-op transform. A trivial composable transform, only useful as an example.
def _noop_transform(trace: Trace, **kwargs) -> Trace:
    start_time_ns = time.time_ns()
    noop_trace = from_trace(trace)

    tracectx_tok: Any
    try:
        tracectx_tok = set_tracectx(noop_trace)
        prims.comment("This comment added by the no-op transform")
    finally:
        reset_tracectx(tracectx_tok)

    noop_trace.bound_symbols.extend(trace.bound_symbols)

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    noop_trace.set_provenance(TraceProvenance(f"No-op Transform (took {elapsed_time_millis} milliseconds)"))

    return noop_trace


def noop(cfn: Callable) -> Callable:
    return add_transform(cfn, _noop_transform)


# The comment fusions transform. Just adds a comment before and after each fusion.
#   This is an example of a post-optimization transform.
def _comment_fusions_transform(trace: Trace, **kwargs) -> Trace:
    start_time_ns = time.time_ns()
    commented_trace = from_trace(trace)

    nbsyms: list[BoundSymbol] = []
    for bsym in trace.bound_symbols:
        if bsym.sym.is_fusion:
            fusion_name = bsym.sym.name
            pre_comment_bsym = prims.comment.bind(f"Before {fusion_name}", output=None)
            post_comment_bsym = prims.comment.bind(f"After {fusion_name}", output=None)

            nbsyms.extend([pre_comment_bsym, bsym, post_comment_bsym])
        else:
            nbsyms.append(bsym)

    commented_trace.bound_symbols = nbsyms
    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000

    commented_trace.set_provenance(TraceProvenance(f"Comment Fusions (took {elapsed_time_millis} milliseconds)"))

    return commented_trace


def comment_fusions(cfn: Callable) -> Callable:
    return add_post_optimization_transform(cfn, _comment_fusions_transform)


#
# Helper functions for composable transforms
#


# Flattens a list of bound symbols, returning the flattened list
def flatten_for_transform(should_flatten: Callable, bsyms: list[BoundSymbol]) -> list[BoundSymbol]:
    flattened: list[BoundSymbol] = []

    def _flatten(bsym: BoundSymbol):
        if should_flatten(bsym):
            check(
                len(bsym.subsymbols) > 0,
                lambda: f"Trying to flatten {bsym} to create a grad formula, but it has no subsymbols",
            )
            for sbsym in bsym.subsymbols:
                _flatten(sbsym)
        else:
            flattened.append(bsym)

    for bsym in bsyms:
        _flatten(bsym)

    return flattened


#
# Phantom grad transform
#

#
# Functions related to functionalizing ThunderOptimizedModules
#


# TODO Test with buffers
def populate_grads(grads: list[TensorProxy], tom: None | torch.nn.Module = None, args=None, kwargs=None) -> None:
    idx: int = 0
    from thunder import ThunderModule, compile_data

    if isinstance(tom, ThunderModule) or thunder.compile_data(tom).using_jit:
        assert args is not None, "populate grad needs args (and possibly kwargs) to work with ThunderModules"
        if kwargs is None:
            kwargs = {}
        _, computation_inputs, _ = compile_data(tom).get_computation_and_inputs(*args, **kwargs)
        for p in computation_inputs:
            if isinstance(p, torch.Tensor) and p.requires_grad:
                # Supports grad accumulation (like when weight tying)
                if p.grad is not None:
                    p.grad += grads[idx]
                else:
                    p.grad = grads[idx]
                idx += 1
        return

    # Short-circuits if there are no args or kwargs
    if args is None and kwargs is None:
        return

    flats, _ = tree_flatten((args, kwargs))
    for f in flats:
        if isinstance(f, torch.Tensor) and f.requires_grad:
            f.grad = grads[idx]
            idx += 1


def extract_grads(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    grads = tuple(
        f.grad
        for f in chain(module.parameters(), module.buffers())
        if isinstance(f, torch.Tensor) and f.requires_grad and f.grad is not None
    )
    return grads


def clear_grads(module: torch.nn.Module) -> None:
    if not isinstance(module, torch.nn.Module):
        return

    for p in module.parameters():
        p.grad = None
    for b in module.buffers():
        b.grad = None


from thunder.core.interpreter import make_opaque
from thunder.core.langctxs import langctx, Languages


# TODO RC1 Replace with langctx
def torchctx(fn):
    _fn = langctx(Languages.TORCH)(fn)
    return make_opaque(_fn)


_grad_fn_map: dict[Any, Callable] = {}


def register_grad(sym_or_id: Symbol | Any, gradfn: Callable) -> None:
    id: Any = sym_or_id
    if isinstance(sym_or_id, Symbol):
        id = sym_or_id.id
    _grad_fn_map[id] = gradfn


# Grad functions for prims
from thunder.core.prims import PrimIDs as pids, get_grad, put_grad


# A generalization of prims.put_grad to pytrees
# TODO Consider validating that the specs are the same
# TODO Consider validating that that object requires grad (and filtering o.w.)
def put_grads(a, g):
    flats, _ = tree_flatten(a)
    flatgrads, _ = tree_flatten(g)

    for f, fg in zip(flats, flatgrads):
        if isinstance(f, TensorProxy) and isinstance(fg, TensorProxy):
            put_grad(f, fg)


#
# Unpacking operator grads
#

# NOTE prims.unpack_empty_dict creates no grad associations
register_grad(pids.UNPACK_EMPTY_DICT, prims.unpack_empty_dict)

# NOTE prims.unpack_key creates no grad associations
register_grad(pids.UNPACK_KEY, prims.unpack_key)

# NOTE prims.unpack_sequence creates no grad associations
register_grad(pids.UNPACK_SEQUENCE, prims.unpack_sequence)


#
# Data movement and transformation operator grads
#
@torchctx
def _convert_element_type_prim_grad(a: Number | TensorProxy, dtype: type | dtypes.dtype) -> Number | TensorProxy:
    fwd = prims.convert_element_type(a, dtype)

    g = get_grad(fwd)
    g_converted = prims.convert_element_type(g, dtypes.to_dtype(a))
    put_grad(a, g_converted)

    return fwd


register_grad(pids.CONVERT_ELEMENT_TYPE, _convert_element_type_prim_grad)

#
# Tensor creation operator grads
#

# NOTE prims.full creates no grad associations
register_grad(pids.FULL, prims.full)

# NOTE prims.iota creates no grad associations
register_grad(pids.IOTA, prims.iota)


def _uniform_grad(shape, minval, maxval, *, device, dtype):
    fwd, saved = uniform_aug_fwd(shape, minval, maxval, device=device, dtype=dtype)
    g = get_grad(fwd)
    _, gminval, gmaxval = uniform_backward(*saved, g)
    put_grads((minval, maxval), (gminval, gmaxval))
    return fwd


register_grad(pids.UNIFORM, _uniform_grad)

#
# Reshaping and permuting operator grads
#


@torchctx
def _broadcast_in_dim_prim_grad(
    a: TensorProxy, shape: Sequence[int], broadcast_dimensions: Sequence[int]
) -> TensorProxy:
    fwd = prims.broadcast_in_dim(a, shape, broadcast_dimensions)

    g = get_grad(fwd)

    unit_dims = tuple(i for i, s in enumerate(a.shape) if s == 1)
    bcast_dims = tuple(b for i, b in enumerate(broadcast_dimensions) if i not in unit_dims)
    reduce_dims = tuple(s for i, s in enumerate(range(len(shape))) if i not in bcast_dims)

    g = ltorch.sum(g, reduce_dims)

    # NOTE This must be clang.unsqueeze because torch.unsqueeze, unlike clang.unsqueeze, only accepts an integer
    #   (put another way, torch only allows one unsqueeze at a time)
    g = clang.unsqueeze(g, unit_dims)

    put_grad(a, g)

    return fwd


register_grad(pids.BROADCAST_IN_DIM, _broadcast_in_dim_prim_grad)


@torchctx
def _cat_prim_grad(tensors: list[TensorProxy], /, dim: int) -> TensorProxy:
    fwd = prims.cat(tensors, dim)

    g = get_grad(fwd)

    slice_start: int = 0
    t: TensorProxy
    for t in tensors:
        dim_len: int = t.shape[dim]
        slice_end: int = slice_start + dim_len
        g_slice: TensorProxy = clang.slice_in_dim(g, slice_start, slice_end, dim=dim)
        slice_start = slice_end
        put_grad(t, g_slice)

    return fwd


register_grad(pids.CAT, _cat_prim_grad)


def _reshape_prim_grad(a: TensorProxy, shape: tuple[int, ...]) -> TensorProxy:
    fwd = prims.reshape(a, shape)

    g = get_grad(fwd)

    a_grad = prims.reshape(g, a.shape)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.RESHAPE, _reshape_prim_grad)


@torchctx
def _slice_prim_grad(
    a: TensorProxy, start_indices: Sequence[int], end_indices: Sequence[int], strides: None | Sequence[int] = None
) -> TensorProxy:
    fwd = prims.slice_prim(a, start_indices, end_indices, strides)

    g = get_grad(fwd)

    padding = None
    if strides is None or np.all(np.equal(strides, 1)):
        padding = tuple(zip(start_indices, np.subtract(a.shape, end_indices), (0,) * len(start_indices)))
    else:
        real_limits = np.add(
            start_indices,
            np.where(np.equal(g.shape, 0), 0, np.add(1, np.multiply(np.subtract(g.shape, 1), strides))),
        )
        padding = tuple(zip(start_indices, np.subtract(a.shape, real_limits), np.subtract(strides, 1)))

    # Converts NumPy numbers to Python ints
    # TODO Should we support NumPy numbers better?
    padding = tree_map(int, padding)
    a_grad = prims.pad(g, const_as(0, g.dtype), padding)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.SLICE, _slice_prim_grad)


@torchctx
def _squeeze_prim_grad(a: TensorProxy, /, dims: tuple[int, ...]) -> TensorProxy:
    fwd = prims.squeeze(a, tuple(dims))

    g = get_grad(fwd)
    # NOTE This calls clang.unsqueeze, and not torch.unsqueeze, because torch.unsqueeze only supports
    #   unsqueezing a single dimension
    a_grad = clang.unsqueeze(g, dims)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.SQUEEZE, _squeeze_prim_grad)


@torchctx
def _take_prim_grad(a: TensorProxy, index: TensorProxy, dim: int) -> TensorProxy:
    fwd = prims.take(a, index, dim)

    g = get_grad(fwd)

    # TODO Switch to ltorch.index_add?
    # NOTE Intentionally not calling zeros_like to avoid preserving a
    # TODO Update to call ltorch.zeros
    zeros = prims.full(a.shape, fill_value=0, device=a.device, dtype=a.dtype)
    a_grad = prims.index_add(zeros, index, g, dim)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.TAKE, _take_prim_grad)


@torchctx
def _take_along_axis_prim_grad(a: TensorProxy, index: TensorProxy, dim: int) -> TensorProxy:
    fwd = prims.take_along_axis(a, index, dim)

    g = get_grad(fwd)
    # NOTE Intentionally not calling zeros_like to avoid preserving a
    # TODO Update to call ltorch.zeros
    zeros = prims.full(a.shape, fill_value=0, device=a.device, dtype=a.dtype)
    a_grad = prims.scatter_add(zeros, index, g, dim)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.TAKE_ALONG_AXIS, _take_along_axis_prim_grad)


@torchctx
def _transpose_prim_grad(a: TensorProxy, permutation: tuple[int, ...]) -> TensorProxy:
    fwd = prims.transpose(a, tuple(permutation))

    g = get_grad(fwd)
    undo = _argsort(permutation)
    a_grad = prims.transpose(g, tuple(undo))
    put_grad(a, a_grad)

    return fwd


register_grad(pids.TRANSPOSE, _transpose_prim_grad)

#
# Memory layout operator grads
#


@torchctx
def _stride_order_prim_grad(a: TensorProxy, /, order: Sequence[int]) -> TensorProxy:
    fwd = prims.stride_order(a, order)

    g = get_grad(fwd)
    put_grad(a, g)

    return fwd


register_grad(pids.STRIDE_ORDER, _stride_order_prim_grad)


#
# Elementwise unary operator grads
#
@torchctx
def _abs_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.abs(a)

    g = get_grad(fwd)
    put_grad(a, g * ltorch.sign(a))

    return fwd


register_grad(pids.ABS, _abs_prim_grad)


def _cos_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.cos(a)

    g = get_grad(fwd)
    put_grad(a, g * (-prims.sin(a)))

    return fwd


register_grad(pids.COS, _cos_prim_grad)


@torchctx
def _erf_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.erf(a)

    g = get_grad(fwd)
    a_grad = 2 / math.sqrt(math.pi) * ltorch.exp(-(a**2)) * g
    put_grad(a, a_grad)

    return fwd


register_grad(pids.ERF, _erf_prim_grad)


@torchctx
def _exp_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.exp(a)

    g = get_grad(fwd)
    a_grad = g * fwd
    put_grad(a, a_grad)

    return fwd


register_grad(pids.EXP, _exp_prim_grad)


@torchctx
def _log_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.log(a)

    g = get_grad(fwd)
    a_grad = g / a
    put_grad(a, a_grad)

    return fwd


register_grad(pids.LOG, _log_prim_grad)


@torchctx
def _neg_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.neg(a)

    g = get_grad(fwd)
    put_grad(a, -g)

    return fwd


register_grad(pids.NEG, _neg_prim_grad)


@torchctx
def _rsqrt_prim_grad(a: Number | TensorProxy, /) -> Number | TensorProxy:
    fwd = prims.rsqrt(a)

    g = get_grad(fwd)
    # An alternative derivation used by JAX is -0.5 * g * rsqrt(x) / x
    # where rsqrt(x) and x are saved for the backwards pass.
    # This derivation was selected because it avoids saving the input tensor.
    a_grad = -0.5 * g * fwd**3.0
    put_grad(a, a_grad)

    return fwd


register_grad(pids.RSQRT, _rsqrt_prim_grad)


def _sin_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.sin(a)

    g = get_grad(fwd)
    put_grad(a, g * prims.cos(a))

    return fwd


register_grad(pids.SIN, _sin_prim_grad)


@torchctx
def _tanh_prim_grad(a: Number | TensorProxy, /) -> Number | TensorProxy:
    fwd = prims.tanh(a)

    g = get_grad(fwd)
    a_grad = g * (1 - fwd * fwd)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.TANH, _tanh_prim_grad)

#
# Elementwise binary operator grads
#


@torchctx
def _add_prim_grad(a: Number | TensorProxy, b: Number | TensorProxy, /) -> Number | TensorProxy:
    fwd = a + b

    g = get_grad(fwd)
    a_grad = g
    b_grad = g
    put_grads((a, b), (a_grad, b_grad))

    return fwd


register_grad(pids.ADD, _add_prim_grad)


# NOTE The following grad definition relies on the fact that only inexact dtypes are differentiable,
#   and torch's true division operator and the division primitive agree on those types
@torchctx
def _div_prim_grad(a: Number | TensorProxy, b: Number | TensorProxy, /) -> Number | TensorProxy:
    fwd = a / b

    g = get_grad(fwd)
    a_grad = g / b
    b_grad = -g * ((a / b) / b)
    put_grads((a, b), (a_grad, b_grad))

    return fwd


register_grad(pids.DIV, _div_prim_grad)

# Comparison operators -- these create no grad associations
register_grad(pids.EQ, prims.eq)
register_grad(pids.GE, prims.ge)
register_grad(pids.LT, prims.lt)


@torchctx
def _mul_prim_grad(a: Number | TensorProxy, b: Number | TensorProxy, /) -> Number | TensorProxy:
    fwd = a * b

    g = get_grad(fwd)
    a_grad = b * g
    b_grad = a * g
    put_grads((a, b), (a_grad, b_grad))

    return fwd


register_grad(pids.MUL, _mul_prim_grad)


@torchctx
def _sub_prim_grad(a: Number | TensorProxy, b: Number | TensorProxy) -> Number | TensorProxy:
    fwd = a - b

    g = get_grad(fwd)
    a_grad = g
    b_grad = -g
    put_grads((a, b), (a_grad, b_grad))

    return fwd


register_grad(pids.SUB, _sub_prim_grad)


#
# Conditional operator grads
#


def _where_prim_grad(pred: Number | TensorProxy, a: Number | TensorProxy, b: Number | TensorProxy) -> TensorProxy:
    fwd = prims.where(pred, a, b)

    g = get_grad(fwd)
    a_grad = ltorch.where(pred, g, 0)
    b_grad = ltorch.where(pred, 0, g)
    put_grads((a, b), (a_grad, b_grad))

    return fwd


register_grad(pids.WHERE, _where_prim_grad)

#
# Reduction operator grads
#


# TODO Review tweaking grad_chooser_pullback interface
def _amax_prim_grad(a: TensorProxy, /, dims: Sequence[int]) -> TensorProxy:
    fwd = prims.amax(a, dims)

    g = get_grad(fwd)
    a_grad = grad_chooser_backward(fwd, a, a.shape, dims, g)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.AMAX, _amax_prim_grad)


def _sum_prim_grad(a: TensorProxy, /, dims: Sequence[int]) -> TensorProxy:
    fwd = prims.sum(a, dims)

    g = get_grad(fwd)
    a_grad = restore_reduced_dims(g, dims, a.shape)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.SUM, _sum_prim_grad)


@torchctx
def _topk_prim_grad(
    a: TensorProxy, /, k: int, dim: None | int = None, largest: bool = True, sorted: bool = True, *, out=None
):
    fwd = prims.topk(a, k, dim, largest, sorted, out=out)
    val, idx = fwd

    val_grad = get_grad(val)

    a_grad = ltorch.zeros_like(a)
    # TODO: replace with scatter once we have it.
    # scatter_add is a prim and it relies on atomic ops.
    a_grad = ltorch.scatter_add(a_grad, dim, idx, val_grad)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.TOPK, _topk_prim_grad)


# TODO Fix division by zero when n_elem_reduced == 0 or when mean.numel == 0
#   by returning zeros_like(a) or similar.
# TODO Fix grad when correction > n_elem_reduced.
@torchctx
def _var_mean_prim_grad(a: TensorProxy, /, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    v, m = prims.var_mean(a, dims, correction=correction)

    gv = get_grad(v)
    gm = get_grad(m)

    n_elem_reduced = a.numel // m.numel if a.numel != 0 else 1

    # Computes mean bwd
    mean_scale = 1.0 / n_elem_reduced
    mean_grad = mean_scale * restore_reduced_dims(gm, dims, a.shape)

    # Computes var bwd
    normalization_scalar = n_elem_reduced - correction
    restored_gv = restore_reduced_dims(gv, dims, a.shape)
    restored_mean = restore_reduced_dims(m, dims, a.shape)
    var_grad = (2 * restored_gv * (a - restored_mean)) / normalization_scalar

    put_grad(a, mean_grad + var_grad)

    return v, m


register_grad(pids.VAR_MEAN, _var_mean_prim_grad)


#
# Linear algebra operator grads
#
@torchctx
def _linear_prim_grad(a: TensorProxy, w: TensorProxy, bias: None | TensorProxy) -> TensorProxy:
    fwd = prims.linear(a, w, bias)

    g = get_grad(fwd)

    first_dim = -2
    grad_a = ltorch.matmul(g.reshape(-1, g.shape[-1]), w).reshape(a.shape)

    grad_w: TensorProxy
    if a.ndim == 1:
        grad_w = ltorch.matmul(g.unsqueeze(first_dim).mT, a.unsqueeze(first_dim))
    else:
        grad_w = ltorch.matmul(g.reshape(-1, g.shape[-1]).mT, a.reshape(-1, a.shape[-1]))

    put_grads((a, w), (grad_a, grad_w))

    if bias is not None:
        if g.ndim > 1:
            grad_bias = ltorch.sum(g, tuple(range(g.ndim - 1)))
        else:
            grad_bias = g
        put_grad(bias, grad_bias)

    return fwd


register_grad(pids.LINEAR, _linear_prim_grad)


# TODO Add explicit ltorch vs clang module to the tensor operations below
# TODO could we get rid of the final squeezes in the b.ndim == 1 case and the a.ndim == 1 case?
@torchctx
def _matmul_prim_grad(a: TensorProxy, b: TensorProxy, /) -> TensorProxy:
    fwd = prims.matmul(a, b)
    g = get_grad(fwd)

    last_dim = (-1,)
    first_dim = (-2,)
    if a.ndim == 1 and b.ndim == 1:
        put_grads((a, b), (g * b, g * a))
    elif b.ndim == 1:
        ga = unsqueeze(g, last_dim) @ unsqueeze(b, last_dim).mT
        gb = a.mT @ unsqueeze(g, last_dim)
        if g.ndim > 1:
            gb = squeeze(gb, last_dim)
            gb = ltorch.sum(gb, tuple(range(gb.ndim - 1)))
        put_grads((a, b), (ga, gb.squeeze()))
    elif a.ndim == 1:
        ga = unsqueeze(g, first_dim) @ b.mT
        if g.ndim > 1:
            ga = ltorch.sum(ga, tuple(range(ga.ndim - 1)))
        gb = unsqueeze(a, first_dim).mT @ unsqueeze(g, first_dim)
        put_grads((a, b), (ga.squeeze(), gb))
    else:
        put_grads((a, b), (g @ b.mT, a.mT @ g))

    return fwd


register_grad(pids.MATMUL, _matmul_prim_grad)

#
# NN operator grads
#


@torchctx
def _embedding_prim_grad(
    a: TensorProxy, /, weight, *, padding_idx=-1, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
) -> TensorProxy:
    fwd = prims.embedding(
        a,
        weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )

    g = get_grad(fwd)
    a_grad = prims.embedding_backward(g, a, weight.shape[0], padding_idx, scale_grad_by_freq, sparse)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.EMBEDDING, _embedding_prim_grad)

#
# Phantom grad transform helpers
#


def _get_gradfn(bsym: BoundSymbol, *, executors_list: Sequence[Any] = tuple()) -> None | Callable:
    cd = get_compile_data()
    executors_list = cd.executors_list if cd is not None else executors_list
    # Checks if the executor which has priority for this operation has a specific grad transform for it
    for ex in executors_list:
        if ex.can_execute_or_fuse(bsym):
            ex_grad_transform: None | Callable = ex.get_grad_transform(bsym.sym)
            if ex_grad_transform is not None:
                return ex_grad_transform
            break

    # If the executor doesn't define its own grad transform, this just returns the default grad transform for the bsym
    gradfn = _grad_fn_map.get(bsym.sym.id, None)
    return gradfn


# The default grad specifier for the grad transform
#   For each tensor output "o" requiring grad, specifies a grad of ones_like(o)
def _grad_specifier_default(pytree: Any) -> None:
    flats, _ = tree_flatten(pytree)

    for f in flats:
        if isinstance(f, TensorProxy) and f.requires_grad:
            # NOTE This uses clang.ones_like for now because ltorch.ones_like will extend the
            #   lifetime of f, while clang.ones_like will extract the metadata of f at trace time
            prims.put_grad(f, clang.full_like(f, 1.0))


# TODO Test with buffers
def _grad_out_specifier_default(pytree: Any) -> list[TensorProxy]:
    flats, _ = tree_flatten(pytree)
    grads = list([prims.get_grad(f) for f in flats if isinstance(f, TensorProxy) and f.requires_grad])
    return grads


# TODO Formally define the "gradient" of a tensor
# TODO If none of the inputs to an operation require grad, then we could leave that operation alone
#   This could actually improve performance if the operation performs extra computations in its grad transform
#   A workaround for this would be for the operation itself to check if its input requires grad or not
# Transforms a function to compute the "gradient" of its inputs requiring grad, possibly
#   modified by the grad specifiers (see below).
# The optional grad_specifier argument accepts the function's output, runs in a no-grad context,
#   and can put grads (using prims.put_grad()) for tensors. By default, for every tensor output "o"
#   that requires grad, a grad of ones_like(o) is put.
# The optional grad_out_specifier accepts the function's input and can call prims.get_grad() to
#   acquire and return gradients as desired. By default, the input is flattened
#   (tree_flatten(args, kwargs)) and a flat list of grads for all tensors requiring grad
#   and None for other inputs is returned.
# The algorithm for modifying the program has the following steps:
#   1) Flattens the original trace for the grad transform -- ensuing that all top-level symbols have
#       a grad function.
def grad(
    cfn, grad_specifier: Callable = _grad_specifier_default, grad_out_specifier: Callable = _grad_out_specifier_default
) -> Callable:
    # Creates a custom transform callable that binds the additional arguments to the grad transform
    @langctx(Languages.CLANG)
    def _grad_transform(trc: Trace, *, executors_list: Sequence[Any]) -> Trace:
        start_time_ns = time.time_ns()

        # STEP ONE -- Flattens and dces

        # Returns True if the bsym has no grad function, and so must be flattened for the grad transform
        never_flatten: set[Hashable] = {prims.PrimIDs.RETURN, prims.PrimIDs.UNPACK_TRIVIAL}

        def should_flatten_for_grad(bsym: BoundSymbol) -> bool:
            if bsym.sym.id in never_flatten:
                return False

            gradfn: None | Callable = _get_gradfn(bsym, executors_list=executors_list)
            return gradfn is None

        # TODO RC1: maybe move to produce these always on creation
        if trc.kwargs is None:
            trc.kwargs = {}
        if trc.fn is None:
            trc.fn = trc.python_callable()

        gradtrc = from_trace(trc)
        flattened_bsyms: list[BoundSymbol] = flatten_for_transform(should_flatten_for_grad, trc.bound_symbols)
        gradtrc.bound_symbols = flattened_bsyms
        gradtrc = dce(gradtrc)

        # STEP TWO -- Replaces original calls with grad calls

        # Defines the visitor pattern for the first pass of the grad transform,
        #   which swaps BoundSymbols with their grad functions
        def visit_(bsym: BoundSymbol) -> Callable:
            gradfn: None | Callable = _get_gradfn(bsym, executors_list=executors_list)
            check(
                gradfn is not None,
                lambda: f"Failed to find a gradfn for {bsym=} after flattening",
                exception_type=AssertionError,
            )
            return gradfn

        @wraps(trc.fn.meta if isinstance(trc.fn, Symbol) else trc.fn)
        def interpreting_fn(*args, **kwargs):
            result = eval_trace(gradtrc, *args, symbol_mapper=visit_, **kwargs)
            # Sets grads for output
            # TODO This effectively runs in a no grad context -- should it run in a grad context,
            #   or should there be the option to run it in a grad context?
            grad_specifier(result)
            # Constructs return value
            flat_grads = grad_out_specifier((args, kwargs))
            return flat_grads

        # NOTE After this call the gradtrc is invalid, because:
        #   1) The non-executable put_grad operations are still in the trace, and multiple calls to put grad are not accumulated
        #   2) The non-executable get_grad operations are still in the trace, and they may produce different proxies when
        #       called on the same inputs
        #   3) The operations are not in a valid order
        gradtrc = construct_trace(
            rename_proxies=False,
            use_dce=False,
        )(interpreting_fn, *gradtrc.args, **gradtrc.kwargs)
        gradtrc.scopes = [gradtrc.bound_symbols]
        gradtrc._complete = False

        # STEP THREE -- Handles put_grad and get_grad operations and accumulates gradients

        # Accumulates gradients, creating the primal -> accumulated grad mapping
        # Sets the tracing context so accumulation operations are recorded into the trace
        try:
            tracectx_tok = set_tracectx(gradtrc)

            # Maps primals to their gradients (which are returned from calling get_grad on the primal)
            primals_to_ginput_map: dict[Variable, TensorProxy] = {}

            # Handles put_grad operations
            # NOTE This modifies a list that is being enumerated over, but it's OK because we're only
            #   appending to the end of the list
            bsym: BoundSymbol
            for bsym in gradtrc.bound_symbols:
                id: Hashable = bsym.sym.id
                if id is pids.PUT_GRAD:
                    primal: Number | TensorProxy
                    grad: Number | TensorProxy
                    primal, grad = bsym.flat_args

                    # TODO Support autograd on numbers
                    # Filters calls to put_grad that would put a grad on a non-tensor or a tensor that doesn't require grad
                    if not isinstance(primal, TensorProxy) or not primal.requires_grad:
                        continue

                    # TODO Support complex autograd
                    check(
                        not dtypes.is_complex_dtype(dtypes.to_dtype(primal)),
                        lambda: f"Complex grad is not supported yet",
                    )

                    vprimal: Variable = variableify(primal)
                    accum: None | TensorProxy = primals_to_ginput_map.get(vprimal, None)

                    if accum is None:
                        primals_to_ginput_map[vprimal] = grad
                    else:
                        accum = accum + grad
                        primals_to_ginput_map[vprimal] = accum

            # Handles get_grad operations

            # Resets the swapmap for grads
            swapmap: dict[Variable, Proxy] = {}

            for bsym in gradtrc.bound_symbols:
                id: Hashable = bsym.sym.id
                if id is pids.GET_GRAD:
                    primal: Number | TensorProxy
                    (primal,) = bsym.flat_args
                    grad: Number | TensorProxy = bsym.output

                    vprimal: Variable
                    vgrad: Variable
                    vprimal, vgrad = variableify(primal), variableify(grad)

                    actual_grad: None | TensorProxy = primals_to_ginput_map.get(vprimal, None)

                    # NOTE If actual_grad is None then there's a get_grad() request for a tensor which
                    #   has no put_grad(), so we return zeros for the grad
                    # TODO Revisit this -- can we remove these computations, or use a special ZeroTensor
                    #   to simplify them?
                    if actual_grad is None:
                        actual_grad = ltorch.zeros_like(primal)
                        primals_to_ginput_map[vprimal] = actual_grad

                    # Updates the grad alias map
                    vactual_grad: Variable = variableify(actual_grad)
                    if vactual_grad != vgrad:
                        swapmap[vgrad] = actual_grad

        finally:
            # Restores scope
            reset_tracectx(tracectx_tok)

        # Filters put_grad and get_grad operations and applies the swapmap
        def _filter(bsym: BoundSymbol) -> bool:
            id: Hashable = bsym.sym.id
            return id in (pids.PUT_GRAD, pids.GET_GRAD)

        gradtrc.bound_symbols = [
            bsym.from_bsym_swap_proxies(swapmap) for bsym in gradtrc.bound_symbols if not _filter(bsym)
        ]

        # STEP FOUR --- Orders the operations
        # TODO Consider alternative scheduling algorithms, maybe by using graph features
        #   Some ideas are looking at memory-saving nodes, and nodes with a high in-degree or high out-degree

        # Creates an initial valid ordering to DCE, so that additional ordering analysis doesn't need to consider all symbols
        roots, leaves = bsym_list_to_dag(gradtrc.bound_symbols)
        ordered_bsyms = toposort_bsym_dag(roots, TOPOSORT_ORDER.TOP_DOWN)
        gradtrc.bound_symbols = ordered_bsyms
        gradtrc = dce(gradtrc)

        # Identifies the order of BoundSymbols using a "DFS and chain" algorithm
        # TODO Elaborate on this algorithm and the implementation
        roots, leaves = bsym_list_to_dag(gradtrc.bound_symbols)
        check(
            len(leaves) == 1,
            lambda: f"Expected only one leaf node when sorting for grad, found there were {len(leaves)} leaves",
        )
        (leaf,) = leaves

        # Computes the tensor memory used by the given pytree
        def memory_used(pytree) -> int:
            memory_use: int = 0

            def count_memory_use(x):
                nonlocal memory_use
                if isinstance(x, TensorProxy):
                    memory_use += x.dtype._bytes * x.numel

            tree_map(count_memory_use, pytree)
            return memory_use

        # True when a node is a "link" -- a node with only one child and at most one parent
        #   Conceptually the node is a "link" in a chain of nodes like A -> B -> C
        def is_link(n: Node) -> bool:
            _is_link = len(n.children) == 1 and len(n.parents) <= 1
            return _is_link

        counter: int = 0

        def _chain(c: list[Node], end: Node) -> list[Node]:
            nonlocal counter

            # TODO Experiment with not always fusing this into the consumer -- there could be an
            #   interesting mincut
            # NOTE In this case the chain cannot be extended, and its origin is input to the function
            #   so this just fuses it into the consumer
            if len(end.parents) == 0:
                for n in c:
                    n.number = counter
                    counter += 1
                return []

            # NOTE len(end.parents) == 1 on this path
            (parent,) = end.parents

            # Extends the chain if the parent is a link
            if is_link(parent):
                c.append(parent)
                return _chain(c, parent)

            # NOTE In this case the chain has a parent that is not a link
            # Finds the mincut
            mincut_idx: int = len(c)
            min_memory_usage: int = memory_used(parent.bsym.output)

            for idx, n in enumerate(c):
                mem = memory_used(n.bsym.output)
                if mem < min_memory_usage:
                    mincut_idx = idx
                    min_memory_usage = mem

            for idx, n in enumerate(c):
                if idx < mincut_idx:
                    n.number = counter
                    counter += 1

            # Returns the producer-side chain cut if it exists
            if mincut_idx < len(c):
                return [c[mincut_idx]]
            else:
                assert mincut_idx == len(c)
                return end.parents

        def chain_out(n: Node, stack: list[Node]) -> None:
            # Short-circuits if the original node n cannot be part of a chain
            if not is_link(n):
                stack.append(n)
                return

            # NOTE is_link(n) == True
            post_chain: list[Node] = _chain([n], n)

            for pc in post_chain:
                stack.append(pc)

        stack: list[Node] = [leaf]
        while len(stack) > 0:
            n: Node = stack.pop()

            if n.number is not None:
                continue

            n.number = counter
            counter += 1

            for p in n.parents:
                chain_out(p, stack)

        def _selector(eligible_nodes: list[Node]) -> int:
            min_idx: int = 0
            min_number: int = eligible_nodes[0].number

            # NOTE Although we've already processed the first element here, this still
            #   enumerates the entire list c to align the enumeration idx properly
            for idx, n in enumerate(eligible_nodes):
                check(n.number is not None, lambda: f"Expected each node to be numbered, but {n} was not")

                if n.number < min_number:
                    min_idx = idx
                    min_number = n.number

            return min_idx

        sorted_bsyms = toposort_bsym_dag(leaves, toposort_order=TOPOSORT_ORDER.BOTTOM_UP, selector=_selector)
        gradtrc.bound_symbols = sorted_bsyms
        gradtrc = dce(gradtrc)

        # TODO Consider how to handle grad w.r.t. only some outputs -- should all grad
        #   operations using the other output be removed? What kind of assumption does that
        #   make about the grad structure? Probably some kind of independence assumption? Are
        #   there practical examples where that's an issue?

        end_time_ns = time.time_ns()
        elapsed_time_ns = end_time_ns - start_time_ns
        elapsed_time_millis = elapsed_time_ns // 1000000
        gradtrc.set_provenance(TraceProvenance(f"Grad (took {elapsed_time_millis} milliseconds)"))

        return gradtrc

    # NOTE This is a kludge to indicate that we shouldn't use PyTorch's autograd because
    #   we're using our own autograd transform
    cfn._using_grad_transform = True

    return add_transform(cfn, _grad_transform)


def grad_v1(
    cfn,
) -> Callable:
    def grad(func):
        def grad_func(*args, **kwargs):
            _, grads = value_and_grad(func)(*args, **kwargs)
            grads = [g for g in grads if g is not None]
            return grads

        return grad_func

    def _grad_transform(trc: Trace, *, executors_list: Sequence[Any]) -> Trace:
        gradtrc = construct_trace()(grad(trc.python_callable()), *trc.args, **trc.kwargs)
        return gradtrc

    cfn._using_grad_transform = True
    return add_transform(cfn, _grad_transform)


class Transforms(Enum):
    IdentityOp = auto()
    VmapOp = auto()
    JvpOp = auto()
    VjpOp = auto()


@lru_cache(maxsize=None)
def symbol_to_eval(bound_symbol):
    """Map a BoundSymbol to a function that evaluates it.

    Args:
        bound_symbol: BoundSymbol to map
    """
    # Symbol is callable
    return bound_symbol.sym


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

    for symbol in trace.bound_symbols:
        if symbol.sym.id in transform_skip_list:
            continue
        args = tree_map(read, symbol.args)
        kwargs = tree_map(read, symbol.kwargs)
        prim_func = symbol_mapper(symbol)
        if prim_func is None:
            continue
        result = prim_func(*args, **kwargs)
        safe_map_flat(write, list(sequencify(symbol.output)), list(sequencify(result)))

    if with_env:
        return tree_map(read, trace.output), env

    return tree_map(read, trace.output)


def _identity_call_metafunc(*args, trace: Trace, **kwargs):
    with detached_trace():
        return eval_trace(trace, *args, **kwargs)


identity_call = Symbol(id=Transforms.IdentityOp, name="identity_call", meta=_identity_call_metafunc)


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
    batch_dim: int | NotMapped

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
    return prims.transpose(x, tuple(perm))


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
    outs = _vmap_call_metafunc(False, args, in_dims, 0, axis_size, function_trace=trace, **kwargs)
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
            vmap_impl = partial(decomposed_fn_vmap_rule, fn=symbol.sym)
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
def _vmap_call_metafunc(detached: bool, args, in_dims, out_dims, axis_size, function_trace: Trace, **kwargs):
    """Metafunction for vmap call.

    Args:
        detached (bool): Whether to detach the trace.
        args (Tuple[Proxy]): Arguments to the function.
        in_dims (Tuple[int]): Batch dimension for each argument.
        out_dims (Tuple[int]): Batch dimension for return values.
        function_trace (Trace): Trace to use for the function.
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
            function_trace, *batched_args, symbol_mapper=partial(vmap_symbol_mapper, axis_size=axis_size), **kwargs
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


vmap_call = Symbol(id=Transforms.VmapOp, name="vmap_call", meta=partial(_vmap_call_metafunc, False))


# TODO: how should we handle out_dims here?
# although here we are calling vmap of identity, so we should know from the call to vmap
# This should be fine. If we have vmap(identity(func), out_dims=N) then this rule is first used
# to get the vmapped result of identity(func) in vmap_symbol_mapper, and out_dims is handled
# after that in the outer _vmap_call_metafunc.
def _identity_call_vmap(axis_size, *batched_args, trace: Trace, **kwargs):
    args, in_dims = unzip2(batched_args)
    out_dims = 0  # Fixme
    outs, out_dims = _vmap_call_metafunc(False, args, in_dims, out_dims, axis_size, function_trace=trace, **kwargs)
    if isinstance(outs, Sequence):
        return safe_map(pair_to_batched_value, safe_zip(outs, out_dims))
    return BatchedValue(outs, out_dims)


vmap_impls[Transforms.IdentityOp] = _identity_call_vmap


def _jvp_call_vmap(axis_size, batched_primals, batched_tangents, *, function_trace: Trace, **kwargs):
    primals, primals_bdims = safe_zip(*batched_primals)
    tangents, tangents_bdims = safe_zip(*batched_tangents)
    jvp_func = partial(_jvp_call_metafunc, False, function_trace=function_trace)
    vmapped_jvp_func = vmap(jvp_func, in_dims=(primals_bdims, tangents_bdims), axis_size=axis_size)
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
        outs = vmap_call(args_flat, in_dims_flat, out_dims, axis_size=axis_size, function_trace=trace)
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
#         vmap(func, in_dims=in_dims, out_dims=out_dims, axis_size=axis_size), executor=executor,
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


def _jvp_call_metafunc(detached: bool, primals, tangents, *, function_trace: Trace, **kwargs):
    """Metafunction for the JVP transform.

    Args:
        detached (bool): Whether to detach the trace.
        primals (Tuple[Proxy]): Primal values.
        tangents (Tuple[Proxy]): Tangent values.
        function_trace (Trace): Trace of the function to be transformed.
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
        result = eval_trace(function_trace, *primals_tangents_duals, symbol_mapper=jvp_symbol_mapper)
        # Unwrapping the JVPDuals
        if isinstance(result, Sequence):
            assert all(isinstance(x, JVPDual) for x in result)
            primals, tangents = unzip2(result)
            return primals, tangents
        assert isinstance(result, JVPDual)
        return result.primal, result.tangent


jvp_call = Symbol(id=Transforms.JvpOp, name="jvp_call", meta=partial(_jvp_call_metafunc, False))


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
        vmap(partial(eval_trace, trace), in_dims=in_dims, out_dims=out_dims, axis_size=axis_size), *primals
    )
    vmapped_func = partial(eval_trace, vmapped_trace)
    out_primals, out_tangents = jvp(vmapped_func)(primals, tangents, **kwargs)
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
        return jvp_call(primals, tangents, function_trace=trace)

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

    primal: Proxy | Number
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
    prims.PrimIDs.ACOS: lambda x: (prims.acos(x), (x,)),
    prims.PrimIDs.ACOSH: lambda x: (prims.acosh(x), (x,)),
    prims.PrimIDs.ASIN: lambda x: (prims.asin(x), (x,)),
    prims.PrimIDs.ASINH: lambda x: (prims.asinh(x), (x,)),
    prims.PrimIDs.ATAN: lambda x: (prims.atan(x), (x,)),
    prims.PrimIDs.ATANH: lambda x: (prims.atanh(x), (x,)),
    prims.PrimIDs.ATAN2: lambda x, y: (prims.atan2(x, y), (x, y)),
    prims.PrimIDs.COSH: lambda x: (prims.cosh(x), (x,)),
    prims.PrimIDs.DIGAMMA: lambda x: (prims.digamma(x), (x,)),
    prims.PrimIDs.ERFC: lambda x: (prims.erfc(x), (x,)),
    prims.PrimIDs.ERFINV: lambda x: (prims.erfinv(x), (prims.erfinv(x),)),
    prims.PrimIDs.ERFCINV: lambda x: (prims.erfcinv(x), (prims.erfcinv(x),)),
    prims.PrimIDs.EXP2: lambda x: (prims.exp2(x), (prims.exp2(x),)),
    prims.PrimIDs.EXPM1: lambda x: (prims.expm1(x), (prims.expm1(x),)),
    prims.PrimIDs.LGAMMA: lambda x: (prims.lgamma(x), (x,)),
    prims.PrimIDs.NDTRI: lambda x: (prims.ndtri(x), (prims.ndtri(x),)),
    prims.PrimIDs.SINH: lambda x: (prims.sinh(x), (x,)),
    prims.PrimIDs.SQRT: lambda x: (prims.sqrt(x), (prims.sqrt(x),)),
    prims.PrimIDs.NE: lambda x, y: (prims.ne(x, y), (x, y)),
    prims.PrimIDs.GT: lambda x, y: (prims.gt(x, y), (x, y)),
    prims.PrimIDs.LE: lambda x, y: (prims.le(x, y), (x, y)),
    prims.PrimIDs.LOG10: lambda x: (prims.log10(x), (x,)),
    prims.PrimIDs.LOG1P: lambda x: (prims.log1p(x), (x,)),
    prims.PrimIDs.LOG2: lambda x: (prims.log2(x), (x,)),
    prims.PrimIDs.ZETA: lambda x, y: (prims.zeta(x, y), (x, y)),
    prims.PrimIDs.FMOD: lambda x, y: (prims.fmod(x, y), (x, y)),
}


# Mapping from symbols to backward functions used in VJP
# The backward function takes the residuals and cotangents and returns the
# vector-Jacobian products for each argument.
backward_impls = {
    prims.PrimIDs.ACOS: lambda x, g: -g / prims.sqrt(1.0 - x * x),
    prims.PrimIDs.ACOSH: lambda x, g: g * prims.rsqrt(x * x - 1.0),
    prims.PrimIDs.ASIN: lambda x, g: g / prims.sqrt(1.0 - x * x),
    prims.PrimIDs.ASINH: lambda x, g: g * prims.rsqrt(1.0 + x * x),
    prims.PrimIDs.ATAN: lambda x, g: g / (1.0 + x * x),
    prims.PrimIDs.ATANH: lambda x, g: g / (1.0 - x * x),
    prims.PrimIDs.COSH: lambda x, g: prims.mul(g, prims.sinh(x)),
    prims.PrimIDs.ERFC: lambda x, g: -g * 2.0 / math.sqrt(math.pi) * prims.exp(-x * x),
    prims.PrimIDs.ERFINV: lambda result, g: g * 0.5 * math.sqrt(math.pi) * prims.exp(result**2),
    prims.PrimIDs.ERFCINV: lambda result, g: -g * 0.5 * math.sqrt(math.pi) * prims.exp(result**2),
    prims.PrimIDs.EXP2: lambda result, g: g * result * math.log(2.0),
    prims.PrimIDs.EXPM1: lambda result, g: g * (result + 1.0),
    prims.PrimIDs.LGAMMA: lambda x, g: g * prims.digamma(x),
    prims.PrimIDs.NDTRI: lambda result, g: g * prims.exp(0.5 * result**2) * math.sqrt(2.0 * math.pi),
    prims.PrimIDs.SINH: lambda x, g: prims.mul(g, prims.cosh(x)),
    prims.PrimIDs.SQRT: lambda result, g: g / (2.0 * result),
    prims.PrimIDs.NE: ZeroBackward(num_args=2),
    prims.PrimIDs.GT: ZeroBackward(num_args=2),
    prims.PrimIDs.LE: ZeroBackward(num_args=2),
    prims.PrimIDs.LOG10: lambda x, g: g / (x * 2.302585092994046),
    prims.PrimIDs.LOG1P: lambda x, g: g / (x + 1),
    prims.PrimIDs.LOG2: lambda x, g: g / (x * 0.6931471805599453),
    prims.PrimIDs.FMOD: lambda x, y, g: (g, -g * prims.trunc(x / y)),
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


@register_backward(prims.PrimIDs.ZETA)
def zeta_backward(x, y, g):
    # The derivative wrt the first argument is not expressible in terms of zeta or
    # other special functions
    # Therefore, we compute only the derivative wrt the second argument
    gy = g * -x * prims.zeta(x + 1.0, y)

    # Return a mappping from the forward arguments to the gradients
    return {"y": gy}


@register_backward(prims.PrimIDs.DIGAMMA)
def digamma_backward(a: Proxy, g):
    from thunder.torch import polygamma

    return g * polygamma(1, a)


@register_augmented_forward("torch.polygamma")
def polygamma_aug_fwd(n: int, a: Proxy):
    from thunder.torch import polygamma

    primal = polygamma(n, a)
    residuals = (n, a)
    return VJPDual(primal, residuals)


@register_backward("torch.polygamma")
def polygamma_backward(n: int, a: Proxy, g):
    from thunder.torch import polygamma

    return None, g * polygamma(n + 1, a)


@register_backward(prims.PrimIDs.ATAN2)
def atan2_backward(x, y, g):
    alpha = 1.0 / (x * x + y * y)
    grad_x = g * y * alpha
    grad_y = g * -x * alpha
    return grad_x, grad_y


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
    if a.dtype != v.dtype:
        a = prims.convert_element_type(a, v.dtype)
    mean = prims.sum(a, dim) / n_elem_reduced
    mean = restore_reduced_dims(mean, dim, a.shape)
    return (2 * g * (a - mean)) / normalization_scalar


def n_elem_reduced(a_ndim, a_shape, dims):
    dims = utils.canonicalize_dims(a_ndim, dims)
    reduction_size = 1
    for idx, size in enumerate(a_shape):
        if idx in dims:
            reduction_size *= size
    return reduction_size


def mean_backward(a_ndim, a_shape, dims, grad):
    mean_local_grad = 1.0 / n_elem_reduced(a_ndim, a_shape, dims)
    return restore_reduced_dims(grad, dims, a_shape) * mean_local_grad


@register_augmented_forward(prims.PrimIDs.PAD)
def pad_aug_fwd(a, padding_value, padding_config):
    return VJPDual((prims.pad(a, padding_value, padding_config),), (a, padding_config))


@register_backward(prims.PrimIDs.PAD)
def pad_backward(a, padding_config, g):
    # Short circuit on empty input.
    if any(dim == 0 for dim in a.shape):
        return full_like(a, fill_value=0)

    # Un-pad by padding with zero values
    zero_padding_config = [(-lo, -hi, 0) for lo, hi, _ in padding_config]

    g = prims.pad(g, 0.0, zero_padding_config)

    # Un-slice by slicing with a stride of value (dilation + 1)
    for dim, (_, _, d) in enumerate(padding_config):
        g = slice_in_dim(g, 0, g.shape[dim], stride=d + 1, dim=dim)

    return g


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
    return prims.div(restore_reduced_dims(primal * g, reduced_dims, x_shape), x)


def keepdim_reduction(reduction_fn, x, dims):
    """Applies reduction and fixes output to conform to keepdim=True"""
    out = reduction_fn(x, dims)
    argmax_sum_out_shape = [x.shape[i] if i not in dims else 1 for i in range(x.ndim)]
    broadcast_dims = [i for i in range(x.ndim) if i not in dims]
    return prims.broadcast_in_dim(out, argmax_sum_out_shape, broadcast_dims)


# Inspired from https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py#L353
def grad_chooser_backward(primal, x, x_shape, reduced_dims, g):
    """Builds gradient of functions that choose a single item, such as min or max."""
    g_repeated = restore_reduced_dims(g, reduced_dims, x_shape)
    primal_repeated = restore_reduced_dims(primal, reduced_dims, x_shape)
    argmax_locations = x == primal_repeated
    argmax_sum = keepdim_reduction(prims.sum, argmax_locations, reduced_dims)
    out = g_repeated * argmax_locations / argmax_sum
    return out


register_backward(prims.PrimIDs.AMIN)(grad_chooser_backward)


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


# NOTE: Jax uses np.argsort in its transpose vjp computation
def _argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


@register_augmented_forward(prims.PrimIDs.DEVICE_PUT)
def device_put_aug_fwd(a: TensorProxy, device: Device) -> TensorProxy:
    primal = prims.device_put(a, device)
    residuals = (a.device,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.DEVICE_PUT)
def device_put_backward(orig_device, g):
    return prims.device_put(g, orig_device), None


@register_augmented_forward(prims.PrimIDs.CONVOLUTION)
def convolution_aug_fwd(
    a: Proxy,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    primal = convolution(a, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
    residuals = (primal, a, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.CONVOLUTION)
def convolution_backward(
    output: Proxy,
    input: Proxy,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    grad,
):
    # Transposed convolution is not supported!
    assert transposed == 0

    input_grad = None
    weight_grad = None
    bias_grad = None

    # Short circuit on zero-dim grad
    if any(s == 0 for s in grad.shape):
        input_grad = full_like(input, fill_value=0)
        weight_grad = full_like(weight, fill_value=0)
        if bias is not None:
            bias_grad = full_like(bias, fill_value=0)
        return (input_grad, weight_grad, bias_grad) + ((None,) * 6)

    batch, in_channels, *spatial_dims = input.shape
    out_channels, gin_channels, *kernel_dims = weight.shape
    dim = len(spatial_dims)

    def maybe_expand_seq(s, dim):
        if len(s) == 1:
            return (s[0],) * dim
        else:
            return s

    stride = maybe_expand_seq(stride, dim)
    padding = maybe_expand_seq(padding, dim)
    dilation = maybe_expand_seq(dilation, dim)

    def conv_transpose(t):
        return prims.transpose(t, (1, 0) + tuple(range(2, t.ndim)))

    # input_grad = {
    def transpose_and_flip_weight(weight):
        # The lines below are transposing the channels dims.
        # We also need to extract the group information and merge it
        # with the dimension corresponding to the "out_channels" dim.
        # (out_channels, gin_channels) -> (gin_channels, out_channels)
        weight = conv_transpose(weight)
        # Split (out_channels,) -> (groups, out_channels // groups)
        weight = weight.reshape([gin_channels, groups, out_channels // groups] + kernel_dims)
        # Moving groups to the left-most position.
        # (gin_channels, groups, out_channels // groups) -> (groups, gin_channels, out_channels // groups)
        weight = conv_transpose(weight)
        # Squash (groups, gin_channels) -> (in_channels)
        weight = weight.reshape([in_channels, out_channels // groups] + kernel_dims)

        # Flip spatial dimensions
        weight = prims.flip(weight, tuple(range(2, weight.ndim)))
        return weight

    # We need to pad the gradient to be able to fit kernel windows.
    initial_grad_padding = [d * (k - 1) for d, k in zip(dilation, kernel_dims)]

    input_grad = convolution(
        prims.pad(
            grad,
            0.0,
            # The pixes are stride away from each other in the original input.
            # Hence we need to dilate the gradient by dilation=stride - 1
            # so that there are stride - 1 zeros between the pixels.
            [(0, 0, 0), (0, 0, 0)] + [(0, 0, s - 1) for s in stride],
        ),
        transpose_and_flip_weight(weight),
        None,
        # Setting stride to 1 as the distance between pixels is taken
        # care by the pad right above.
        (1,),
        initial_grad_padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )

    def pad_to_input(grad):
        # We need to unpad the padding done to the input prior to the convolution.
        # Note that low and high padding are not necessarily equal, so we cannot
        # absorb it into the convolution just yet, unless the API is modified.
        pad_config = [(0, 0, 0), (0, 0, 0)]
        for o, i, g, p in zip(grad.shape[2:], spatial_dims, input_grad.shape[2:], padding):
            lo = -p
            # Note that (i + 2 * p) is the size of the padded input,
            # so the quantity (i + 2 * p) - g tells us by how much
            # we need to pad the gradient so that the elements outside
            # of the convolution receive zero gradient.
            # p is additionally subtracted to negate the input's pad
            # in forward.
            hi = (i + 2 * p) - g - p
            pad_config.append((lo, hi, 0))
        return prims.pad(grad, 0.0, pad_config)

    input_grad = pad_to_input(input_grad)
    # }

    # bias_grad = {
    if bias is not None:
        import thunder.torch as ltorch

        bias_grad = ltorch.sum(grad, [d for d in range(grad.ndim) if d != 1])
    # }

    # weight grad = {
    def pad_transpose_and_push_groups_into_batches(t):
        # First pad,...
        # Pad as necessary so that in convolutions we never advance
        # past relevant inputs.
        pad_config = [(0, 0, 0), (0, 0, 0)]
        for o, i, k, p, s, d in zip(grad.shape[2:], spatial_dims, kernel_dims, padding, stride, dilation):
            # Padding from below is the same.
            lo = p
            # There is always the max index idx in the input
            # the value of which, input[idx], the kernel touches.
            # The kernel reach, or k_reach, is exactly idx + 1.
            # We use it to decide by how much we need to pad from above
            # as to not move past relevant values.
            k_reach = (o - 1) * s + d * (k - 1) + 1
            # The pad from above equals k_reach minus the length
            # of the input padded from below.
            hi = k_reach - (i + p)
            pad_config.append((lo, hi, 0))
        t = prims.pad(t, 0.0, pad_config)

        _, _, *t_spatial_dims = t.shape

        # ... then do the rest.
        # t.shape == (batch, in_channels, ...)
        # The result of this function has shape
        # (in_channels, batch * groups)
        # (batch, in_channels) -> (in_channels, batch)
        t = conv_transpose(t)
        # Split (in_channels,) -> (groups, gin_channels)
        t = t.reshape([groups, gin_channels, batch] + t_spatial_dims)
        # Transpose (groups, gin_channels, batch) -> (gin_channels, groups, batch)
        t = conv_transpose(t)
        # Flatten (groups, batch) -> (groups * batch,)
        t = t.reshape([gin_channels, groups * batch] + t_spatial_dims)
        return t

    # input will have shape (gin_channels, groups * batch)
    input = pad_transpose_and_push_groups_into_batches(input)
    # grad will have shape (out_channels, batch)
    # Note that these shapes are compatible for a group convolution.
    grad = conv_transpose(grad)

    # Why do we flip stride and dilation?
    # kernel[i] and kernel[i + 1] are dilation apart from each other,
    # so dilation becomes the new stride.
    # All the elements that kernel[i] touches are stride away from each other,
    # hence stride becomes the new dilation.
    weight_grad = convolution(
        input,
        grad,
        None,
        dilation,  # set stride=dilation
        (0,),
        stride,  # set dilation=stride
        transposed,
        output_padding,
        groups,
    )

    # The result of the convolution has shape (gin_channels, out_channels),
    # so transposition is required.
    weight_grad = conv_transpose(weight_grad)
    # }

    return (input_grad, weight_grad, bias_grad)


@register_augmented_forward("torch.log_softmax")
def log_softmax_aug_fwd(input: TensorProxy, dim: int, *, dtype=None) -> VJPDual:
    from thunder.torch import log_softmax

    primal = log_softmax(input, dim=dim, dtype=dtype)
    residuals = (primal, dim, input.dtype)
    return VJPDual(primal, residuals)


@register_backward("torch.log_softmax")
def log_softmax_backward(primal, dim, dtype, g):
    from thunder.torch import log_softmax_backward

    return log_softmax_backward(g, primal, dim, dtype)


@register_augmented_forward("torch.nn.functional.nll_loss")
def nll_loss_aug_fwd(
    input: Proxy,
    target: Proxy,
    weight: None | Proxy,
    ignore_index: int,
    reduction: str,
) -> VJPDual:
    from thunder.torch import _nll_loss_helper

    primal, total_weight = _nll_loss_helper(
        input,
        target,
        weight,
        ignore_index,
        reduction,
    )
    residuals = (input, target, weight, reduction, ignore_index, total_weight)
    return VJPDual(primal, residuals)


@register_backward("torch.nn.functional.nll_loss")
def nll_loss_backward(input, target, weight, reduction, ignore_index, total_weight, g):
    from thunder.torch import nll_loss_backward

    ginput = nll_loss_backward(g, input, target, weight, reduction, ignore_index, total_weight)
    return ginput, *((None,) * 4)


@register_augmented_forward("torch.split")
def split_aug_fwd(a: TensorProxy, split_size_or_sections: int | Sequence[int], dim: int = 0) -> VJPDual:
    from thunder.torch import split

    primal = split(a, split_size_or_sections, dim)
    residuals = (dim,)
    return VJPDual(primal, residuals)


@register_backward("torch.split")
def split_backward(dim, *grads):
    from thunder.torch import cat

    return cat(grads, dim)


@register_augmented_forward("torch.nn.functional.embedding")
def embedding_aug_fwd(
    a: Proxy,
    weight: Proxy,
    padding_idx: int | None,
    max_norm: float | None,
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
    return gweight


@register_augmented_forward(prims.PrimIDs.BATCH_NORM)
def batch_norm_aug_fwd(
    a: TensorProxy,
    weight: None | TensorProxy,
    bias: None | TensorProxy,
    running_mean: None | TensorProxy,
    running_var: None | TensorProxy,
    training: bool,
    momentum: Number,
    eps: Number,
) -> VJPDual:
    primal = prims.batch_norm(
        a,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )
    output_mask = [x is not None for x in (a, weight, bias)]
    output, save_mean, save_invstd = primal
    residuals = (a, weight, running_mean, running_var, save_mean, save_invstd, training, eps, output_mask)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.BATCH_NORM)
def batch_norm_backward(a, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, *grads):
    from thunder.torch import batch_norm_backward

    result = batch_norm_backward(
        grads[0], a, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask
    )
    return *result, None, None


@register_augmented_forward("torch.cumsum")
def cumsum_aug_fwd(a: Proxy, dim: int, *, dtype: None | dtypes.dtype = None) -> VJPDual:
    from thunder.torch import cumsum

    primal = cumsum(a, dim, dtype=dtype)
    residuals = (
        a.dtype,
        dim,
    )
    return VJPDual(primal, residuals)


@register_backward("torch.cumsum")
def cumsum_backward(a_dtype, dim, g):
    g = g.to(a_dtype)
    if g.numel <= 1 or g.shape[dim] == 1:
        return g
    return g.flip(dim).cumsum(dim).flip(dim)


@register_augmented_forward("torch.softmax")
def softmax_aug_fwd(a: Proxy, dim: int, dtype: dtypes.dtype | None = None) -> VJPDual:
    from thunder.torch import softmax

    primal = softmax(a, dim, dtype=dtype)
    residuals = (primal, dim)
    return VJPDual(primal, residuals)


@register_backward("torch.softmax")
def softmax_backward(primal, dim, g):
    return primal * (g - (primal * g).sum(dim, keepdim=True))


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

    reconstructed_env = {}

    for idx, sym in enumerate(bound_symbols):
        k = sequencify(sym.output)[0].name
        v = VJPDual(None, saved_for_backward[idx])
        reconstructed_env[k] = v

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
    trace = dce(trace)
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
    trace = dce(trace)
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
    return g


def reciprocal_aug_fwd(a: TensorProxy) -> VJPDual:
    primal = reciprocal(a)
    return VJPDual(primal, (primal,))


def reciprocal_backward(primal, g):
    return -g * primal * primal


@partial(register_grad, prims.PrimIDs.RECIPROCAL)
def reciprocal_joint_forward_backward_rule(a: TensorProxy) -> TensorProxy:
    result, saved = reciprocal_aug_fwd(a)
    g = get_grad(result)
    ga = reciprocal_backward(*saved, g)
    put_grad(a, ga)
    return result


@register_augmented_forward("torch.index_put")
def index_put_aug_fwd(
    a: TensorProxy, /, indices: Sequence[TensorProxy], values: TensorProxy, accumulate: bool = False
) -> VJPDual:
    primal = clang.index_put(a, indices, values, accumulate)
    residuals = (
        indices,
        values,
        accumulate,
    )
    return VJPDual(primal, residuals)


def sum_to(a: TensorProxy, shape: Sequence[int]) -> TensorProxy:
    if not shape:
        return a.sum()
    leading_dims = a.ndim - len(shape)
    reduce_dims = tuple(range(leading_dims)) + tuple(
        i for i in range(leading_dims, a.ndim) if shape[i - leading_dims] == 1 and a.shape[i] != 1
    )
    a = ltorch.sum(a, dim=reduce_dims, keepdim=True)
    if leading_dims > 0:
        return ltorch.view(a, shape)
    return a


@register_backward("torch.index_put")
def index_put_backward(indices: Sequence[TensorProxy], values: TensorProxy, accumulate: bool, g: TensorProxy):
    g_values = g[indices]
    # torch has extra logic to handle the expanded values
    if not utils.same_shape(g_values.shape, values.shape):
        if clang.compute_broadcast_shape(g_values.shape, values.shape):
            g_values = sum_to(g_values, values.shape)
    if accumulate:
        return g, g_values
    return clang.index_put(g, indices, ltorch.zeros_like(values), False), g_values


def uniform_aug_fwd(shape, minval, maxval, *, device, dtype):
    primal = prims.uniform(shape, minval, maxval, device=device, dtype=dtype)
    return VJPDual(primal, (primal, minval, maxval))


def uniform_backward(primal, minval, maxval, g):
    # uniform is implemented as (maxval - minval) * uniform(shape, 0, 1) + minval
    unscaled_primal = (primal - minval) / (maxval - minval)
    reduce_all_dims = tuple(range(g.ndim))
    sum = partial(prims.sum, dims=reduce_all_dims)
    return None, sum(g * (1 - unscaled_primal)), sum(g * unscaled_primal)


nondifferentiable_vjp_symbols = (prims.PrimIDs.BITWISE_AND, prims.PrimIDs.SIGNBIT, prims.PrimIDs.FULL)


def is_constant_for_vjp(symbol: prims.Symbol) -> bool:
    """Check if a symbol is constant for the VJP transform.

    Args:
        symbol (prims.Symbol): Symbol to check.

    Returns:
        bool: True if the symbol is constant, False otherwise.
    """
    are_all_args_non_differentiable = not any(isinstance(arg, (FloatProxy, TensorProxy)) for arg in symbol.flat_args)
    return (
        are_all_args_non_differentiable
        or symbol.are_all_args_constant
        or symbol.sym.id in nondifferentiable_vjp_symbols
    )


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
    if is_constant_for_vjp(symbol):

        def vjp_impl_const(symbol, *args, **kwargs):
            args, kwargs = tree_map(lambda x: x.primal if isinstance(x, VJPDual) else x, (args, kwargs))
            primals = symbol_to_eval(symbol)(*args, **kwargs)
            if isinstance(primals, Sequence):
                return tree_map(lambda x: VJPDual(x, tuple()), primals)
            return VJPDual(primals, tuple())

        return partial(vjp_impl_const, symbol)

    # Normal case, we have a proxy tangent
    vjp_impl = augmented_forward_impls.get(symbol.sym.id)

    if _get_gradfn(symbol) is not None:
        vjp_impl, backward_fn = make_aug_forward_and_backward(symbol)

    if vjp_impl is None:
        # We could not find a VJP for this symbol, so we try to decompose it
        if len(symbol.subsymbols) > 0 and not isinstance(symbol.sym.id, prims.PrimIDs):
            vjp_impl = partial(decomposed_fn_aug_fwd_rule, decomposed_fn=symbol.sym)
        else:
            # We could not find a VJP for this symbol and we could not decompose it
            # It could be a torch.dropout with 0.0 probability, so we skip it
            if symbol.sym.id == "torch.nn.functional.dropout":
                return None
            print(f"VJP for {symbol} is not implemented")
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


def check_bsym_for_vjp(bsym):
    """
    Check if a bound symbol is supported by vjp.

    Args:
        bsym (BoundSymbol): The bound symbol to check.

    Returns:
        bool: True if the bound symbol is supported by vjp, False otherwise.
    """

    if bsym.sym.id in transform_skip_list:
        return True

    if bsym.sym.id in backward_impls and bsym.sym.id in augmented_forward_impls:
        return True

    if bsym.sym.id in _grad_fn_map:
        return True

    # We could not find a VJP for this symbol, so we try to decompose it
    # into sub-symbols and check if they are supported
    if len(bsym.subsymbols) > 0 and not bsym.sym.is_prim:
        subtrace = construct_trace()(bsym.sym, *bsym.args, **bsym.kwargs)
        subtrace = unwrap_one_level_of_subsymbols(subtrace)
        all_supported = all(check_bsym_for_vjp(subbsym) for subbsym in subtrace.bound_symbols)
        return all_supported

    return False


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

    def get_grad(x: Variable):
        if isinstance(x, Variable):
            # Return None if the variable was not used in the computation and
            # hence not in the env
            return env.get(x.name, None)
        else:
            return x

    def put_grad(v: Variable, val: Any) -> None:
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
        elif isinstance(v, str):
            env[v] = val
        elif isinstance(v, Sequence) and val is None:
            # broadcast None to the right shape
            safe_map(put_grad, v, [None] * len(v))
        elif isinstance(v, Sequence) and isinstance(val, Sequence):
            safe_map_flat(put_grad, v, val)
        else:
            # Skip writing to constants
            pass

    if isinstance(init_cotangents, Sequence) and len(init_cotangents) == 1 and not isinstance(trace.output, Sequence):
        init_cotangents = init_cotangents[0]
    safe_map_flat(put_grad, trace.output, init_cotangents)

    for symbol in reversed(trace.bound_symbols):
        if symbol.sym.id in transform_skip_list:
            continue
        symbol_output = sequencify(symbol.output)

        cotangents = tree_map(get_grad, symbol_output)
        # Having a single cotangent is a common case, so we flatten it
        # Otherwise, we will need to rewrite the pullback functions
        cotangents = tree_flatten(cotangents)[0]
        residuals = forward_env[symbol_output[0].name].residuals
        if is_constant_for_vjp(symbol):
            # We can skip the pullback if all the arguments are constant
            continue

        if all(cotangent is None for cotangent in cotangents):
            # We can skip the pullback if the cotangent is None
            safe_map(put_grad, symbol.args, (None,) * len(symbol.args))
            continue

        if symbol.sym.id == "torch.nn.functional.dropout" and not symbol.subsymbols:
            # We can skip the pullback if the dropout probability is 0.0
            # Assuming that the dropout symbol has the same output and argument
            assert symbol.output.name == symbol.args[0].name, "Dropout symbol has a different output and argument"
            if symbol.args[1] == 0.0 or symbol.args[2] is False:
                continue

        backward = backward_impls.get(symbol.sym.id)
        aug_forward = augmented_forward_impls.get(symbol.sym.id)

        if _get_gradfn(symbol) is not None:
            aug_forward, backward = make_aug_forward_and_backward(symbol)

        if backward is None:
            if len(symbol.subsymbols) > 0 and not isinstance(symbol.sym.id, prims.PrimIDs):
                # We could not find a backward for this symbol, so we try to decompose it
                backward = partial(decomposed_fn_backward_rule, symbol.sym)
            else:
                # We could not find a backward for this symbol and we could not decompose it
                raise NotImplementedError(f"Backward for {symbol.sym.id} is not implemented")

        result = backward(*residuals, *cotangents)
        if isinstance(result, dict):
            # If the backward returns a dict, we assume that it is a dict of
            # forward arguments to the corresponding
            # gradients/cotangents/adjoints/sensitivities.
            used_names = set()
            for i, (k, v) in enumerate(inspect.signature(aug_forward).parameters.items()):
                if v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    put_grad(symbol.args[i], result.get(k, None))
                    used_names.add(k)

            # For developer convenience, we allow using the name from the
            # forward meta in addition to the name from the augmented forward
            # signature.
            # If both names are used, the one from the forward meta takes
            # precedence.
            for i, (k, v) in enumerate(inspect.signature(symbol.sym.meta).parameters.items()):
                if v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    if k not in used_names:
                        put_grad(symbol.args[i], result.get(k, None))
            continue

        if not isinstance(result, Sequence):
            result = (result,)

        def is_differentiable(arg):
            match arg:
                case TensorProxy():
                    return dtypes.is_inexact_dtype(arg.dtype)
                case Sequence():
                    return arg and all(isinstance(x, TensorProxy) and dtypes.is_inexact_dtype(x.dtype) for x in arg)
                case _:
                    return False

        if len(symbol.args) != (orig_res_len := len(result)):
            check(
                orig_res_len <= len(symbol.args),
                lambda: f"Backward for {symbol.sym.id} returned {orig_res_len} values, "
                + f"but expected at most {len(symbol.args)}",
            )
            # Assuming that the non-differentiable arguments were dropped from
            # the backward function, we are going to append None to the result
            # to match the number of arguments. Alternatively, we could just
            # have a for-loop with a conditional when writing to the
            # environment.

            iter_result = iter(result)
            n_differentiable_args = sum(bool(is_differentiable(arg)) for arg in symbol.args)
            check(
                n_differentiable_args <= orig_res_len,
                lambda: f"Backward for {symbol.sym.id} returned {orig_res_len} value(s), "
                + f"but expected {n_differentiable_args}",
            )

            result = tuple(next(iter_result) if is_differentiable(arg) else None for arg in symbol.args)

        # See "Backward impl for ops of the type Sequence[TensorProxy], ... -> ... results in None grads."
        # This is a temporary workaround.
        if symbol.sym.id in (prims.PrimIDs.CAT, "torch.cat", "torch.stack"):
            safe_map_flat(put_grad, symbol.args, result)
        else:
            safe_map(put_grad, symbol.args, result)

    def get_inexact_dtype_or_none(x):
        if isinstance(x, (TensorProxy, FutureTensorProxy)) and dtypes.is_inexact_dtype(x.dtype):
            return x
        else:
            return None

    gargs = tree_map(get_grad, tuple(trace.args))
    gkwargs = tree_map(get_grad, trace.kwargs)
    gkwargs = {k: v for k, v in gkwargs.items() if v is not None}
    gargs, gkwargs = tree_map(get_inexact_dtype_or_none, (gargs, gkwargs))
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


# TODO: Can't use a Symbol here because mixed executor sybsymbols seem to be
# unsupported. See issue "Could not find an executor for bound symbol when its subsymbols
# are not fully supported by a single executor"
vjp_call = partial(
    vjp_call_metafunc, False
)  # Symbol(id=Transforms.VjpOp, name="vjp_call", meta=partial(vjp_call_metafunc, False))


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
    assert forward_trace.bound_symbols[-1].sym.id == prims.PrimIDs.RETURN
    new_return = (forward_trace.output[0], (saved_tensors, saved_other))
    forward_trace.bound_symbols[-1] = replace(forward_trace.bound_symbols[-1], args=new_return)


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
    unpacking_trace = construct_trace(rename_proxies=False, use_dce=False)(
        unpacking_fn, (saved_tensors, saved_other), cotangents
    )
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
    backward_trace.bound_symbols = list((*unpacking_trace.bound_symbols[:-1], *backward_trace_bsyms_without_unpacking))


# NOTE: Returning namedtuples from compiled functions doesn't work. See:
# "Allow returning namedtuples from compiled functions"
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
    augmented_forward_fn.__signature__ = inspect.signature(trace.fn or trace.python_callable())

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
            gkwargs = out[-1] if isinstance(out[-1], dict) else {}
            gargs = out[:-1] if isinstance(out[-1], dict) else out
            gkwargs = {k: gkwargs.get(k, None) for k in trace.kwargs}
            out = (*gargs, gkwargs)
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
    used_mask = list(len(consumers.get(x, ())) > 0 for x in bw_flat_saved_for_backward)

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

    def map_fn(a):
        if isinstance(a, TensorProxy) and a.dtype in allowed_downcast_types:
            return maybe_convert_to_dtype(a, dtype)
        return a

    return tree_map(map_fn, args)


@register_autocast_rule("torch.matmul")
@register_autocast_rule(prims.PrimIDs.MATMUL)
def autocast_matmul_rule(a, b, dtype):
    """Autocast rule for matmul"""
    return prims.matmul(*(maybe_downcast_to(dtype, (a, b))))


@register_autocast_rule("torch.nn.functional.linear")
@register_autocast_rule(prims.PrimIDs.LINEAR)
def autocast_linear_rule(a, w, bias, dtype):
    if bias is None:
        # Don't pass `bias` to maybe_downcast_to.
        downcast_args = maybe_downcast_to(dtype, (a, w)) + (bias,)
    else:
        downcast_args = maybe_downcast_to(dtype, (a, w, bias))

    return prims.linear(*downcast_args)


@register_autocast_rule("torch.nn.functional.scaled_dot_product_attention")
def autocast_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    *,
    dtype,
    scale,
):
    from thunder.torch import scaled_dot_product_attention

    q, k, v = maybe_downcast_to(dtype, (query, key, value))
    return scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale)


def autocast_symbol_mapper(bound_symbol: BoundSymbolInterface, dtype: dtypes.dtype):
    """Return the callable implementing the autocast rule for the symbol.

    Args:
        bound_symbol: Mapped to its autocast rule.

    Returns:
        Callable: The callable implementing the autocast rule for the symbol.
    """
    autocast_impl: Callable | None = autocast_impls.get(bound_symbol.sym.id)
    return bound_symbol.sym if autocast_impl is None else partial(autocast_impl, dtype=dtype)


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
