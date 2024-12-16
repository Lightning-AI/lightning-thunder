from __future__ import annotations
from collections import namedtuple
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass, replace
from enum import auto, Enum
from itertools import chain, compress
from functools import lru_cache, partial, wraps
import math
from numbers import Number
from typing import Any, TYPE_CHECKING
from collections.abc import Callable
from collections.abc import Sequence
import copy
import inspect
import time
import dataclasses

import thunder
import thunder.core.utils as utils
from thunder.core import dtypes, prims
from thunder.core.devices import cpu, Device
from thunder.core.trace_interpreter import (
    interpret_trace as eval_trace,
    interpret_trace_to_trace,
    trace_interpreter_skip_list,
)
from thunder.core.proxies import (
    CollectionProxy,
    NumberProxy,
    Proxy,
    TensorProxy,
    FloatProxy,
    variableify,
    unvariableify,
    FutureTensorProxy,
    ProxyTag,
)
from thunder.core.compile_data import get_compile_data, get_compile_option
from thunder.core.langctxs import langctx, Languages
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten, tree_flatten_with_dataclass
from thunder.core.symbol import BoundSymbol, BoundSymbolInterface, Symbol
from thunder.core.trace import TraceCtx as Trace
from thunder.core.trace import VariableInterface as Variable
from thunder.core.trace import (
    detached_trace,
    tracectx,
    set_tracectx,
    reset_tracectx,
    from_trace,
    TraceProvenance,
    TraceTag,
)
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
    find_producer_symbols,
)
import thunder.clang as clang
from thunder.clang import (
    empty,
    full,
    full_like,
    unsqueeze,
    squeeze,
    maybe_convert_to_dtype,
    slice_in_dim,
    reciprocal,
    convolution,
)
from thunder.core.transform_common import (
    dce,
    Transform,
    wrap_return_value_together_with_arguments,
    unwrap_return_value,
    VJPDual,
)
from thunder.core.vjp_utils import make_aug_forward_and_backward, get_saved_for_backward_tensors
from thunder.extend import Executor
import thunder.torch as ltorch

import torch

# from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
import numpy as np


TraceTag.register_tag("AUGMENTED_FORWARD")


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
            producer = producers.get(inp, None)
            if producer is None:
                continue
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


# NOTE This operation is inplace. It will modify the trace's bound_symbols.
# NOTE Because this operation is explicitly inplace, it will disregard the trace being "complete".
def insert_inplace(
    trc: Trace,
    idx: int,
    fn: Callable[[], Any],
) -> None:
    r"""Calls ``fn`` and record any symbols called into ``trc``, starting at ``idx``.

    Args:
        trc: Trace to insert :class:`~thunder.core.symbol.BoundSymbol`\s representing ``fn``.
        idx: Starting index of ``trc.bound_symbols`` to insert :class:`~thunder.core.symbol.BoundSymbol`\s representing ``fn``.
        fn:

    .. note::
        This operation is inplace. It will modify the given ``trc``'s :attr:`~thunder.core.trace.TraceCtx.bound_symbols`.

    .. note::
        Because this operation is explicitly inplace, it will disregard whether or not :func:`~thunder.core.trace.TraceCtx.mark_complete` has been called on ``trc`` already.
    """
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


# NOTE This operation is inplace. It will modify the trace's bound_symbols.
# NOTE Because this operation is explicitly inplace, it will disregard the trace being "complete".
def replace_inplace(
    trc: Trace,
    idx: int,
    fn: Callable[[BoundSymbol], Any],
) -> None:
    r"""Removes ``idx``-th :class:`~thunder.core.symbol.BoundSymbol` of ``trc`` and replace it ``bsyms`` representing ``fn``.

    Args:
        trc: Trace to insert :class:`~thunder.core.symbol.BoundSymbol`\s representing ``fn``.
        idx: Index of :class:`~thunder.core.symbol.BoundSymbol` of ``trc``.
        fn: Callable to bake into ``trc``, instead of ``idx``-th :class:`~thunder.core.symbol.BoundSymbol`.

    .. note::
        This operation is inplace. It will modify the given ``trc``'s :attr:`~thunder.core.trace.TraceCtx.bound_symbols`.

    .. note::
        Because this operation is explicitly inplace, it will disregard whether or not :func:`~thunder.core.trace.TraceCtx.mark_complete` has been called on ``trc`` already.
    """
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
def add_transform(
    cfn: Callable,
    *,
    transform: Transform | list[Transform],
    disable_torch_autograd_support=False,
) -> Callable:
    from thunder.common import CompileData

    cd: None | Any = getattr(cfn, "_lc_cd", None)

    utils.check(cd is not None, lambda: f"Can only transform compiled thunder functions")
    utils.check(isinstance(cd, CompileData), lambda: f"Found an unknown compile data attribute {cd}")
    if isinstance(transform, Transform):
        transform = [transform]
    else:
        utils.check(
            all(isinstance(t, Transform) for t in transform),
            lambda: "transform must be an instance of Transform or a list of Transform instances.",
        )

    assert cd.using_jit

    from thunder import jit

    # todo: move _lc_transforms to compile_data
    transforms = cfn._lc_transforms + transform
    jfn = jit(
        cd.fn,
        langctx=cd.langctx,
        executors=cd.executors_list,
        sharp_edges=cd.sharp_edges,
        # cache, interpretation?
        transforms=transforms,
        disable_torch_autograd=cd.disable_torch_autograd_support or disable_torch_autograd_support,
        **cd.compile_options,
    )
    return jfn


# The no-op transform. A trivial composable transform, only useful as an example.
class _NoopTransform(Transform):
    def transform_trace_pre_prologue(
        self, prologue_trace: Trace, computation_trace: Trace, epilogue_trace: Trace | None, **kwargs
    ) -> Trace:
        start_time_ns = time.perf_counter_ns()
        noop_trace = from_trace(computation_trace)

        tracectx_tok: Any
        try:
            tracectx_tok = set_tracectx(noop_trace)
            prims.comment("This comment added by the no-op transform")
        finally:
            reset_tracectx(tracectx_tok)

        noop_trace.bound_symbols.extend(computation_trace.bound_symbols)

        end_time_ns = time.perf_counter_ns()
        elapsed_time_ns = end_time_ns - start_time_ns
        elapsed_time_millis = elapsed_time_ns // 1000000
        noop_trace.set_provenance(TraceProvenance(f"No-op Transform (took {elapsed_time_millis} milliseconds)"))

        return prologue_trace, noop_trace, computation_trace


def noop(cfn: Callable) -> Callable:
    _noop_transform = _NoopTransform()
    return add_transform(cfn, transform=_noop_transform)


# The comment fusions transform. Just adds a comment before and after each fusion.
#   This is an example of a post-optimization transform.
class _CommentFusionsTransform(Transform):
    def transform_trace_post_optimization(self, trace: Trace, **kwargs) -> Trace:
        start_time_ns = time.perf_counter_ns()
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
        end_time_ns = time.perf_counter_ns()
        elapsed_time_ns = end_time_ns - start_time_ns
        elapsed_time_millis = elapsed_time_ns // 1000000

        commented_trace.set_provenance(TraceProvenance(f"Comment Fusions (took {elapsed_time_millis} milliseconds)"))

        return commented_trace


def comment_fusions(cfn: Callable) -> Callable:
    return add_transform(cfn, _CommentFusionsTransform)


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
                lambda: f"No grad rule found for {bsym} and no subsymbols inside it to create a grad formula",
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

    if isinstance(tom, ThunderModule) or compile_data(tom).using_jit:
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


_grad_fn_map: dict[Any, Callable] = {}


def register_grad(sym_or_id: Symbol | Any, gradfn: Callable) -> None:
    id: Any = sym_or_id
    if isinstance(sym_or_id, Symbol):
        id = sym_or_id.id

    # The gradfn are expected to be written in terms of torch functions by
    # default even if the original forward function could be written in terms of
    # other languages. We don't want to have developers worry about the language
    # context when writing grad functions. If the grad function is written in
    # terms of another language, developers can always wrap the gradfn in an
    # appropriate language context that will take precedence over the default
    # torch language context.
    _grad_fn_map[id] = langctx(Languages.TORCH)(gradfn)


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


def _broadcast_in_dim_prim_grad(
    a: TensorProxy, shape: Sequence[int], broadcast_dimensions: Sequence[int]
) -> TensorProxy:
    fwd = prims.broadcast_in_dim(a, shape, broadcast_dimensions)

    g = get_grad(fwd)

    unit_dims = tuple(i for i, s in enumerate(a.shape) if s == 1)
    bcast_dims = tuple(b for i, b in enumerate(broadcast_dimensions) if i not in unit_dims)
    reduce_dims = tuple(s for i, s in enumerate(range(len(shape))) if i not in bcast_dims)

    # NOTE When the reduce_dims tuple is empty, pytorch reduces all dimensions.
    # In this case, we do not want to reduce any dimensions, so skip this sum.
    if len(reduce_dims) > 0:
        g = ltorch.sum(g, reduce_dims)

    # NOTE This must be clang.unsqueeze because torch.unsqueeze, unlike clang.unsqueeze, only accepts an integer
    #   (put another way, torch only allows one unsqueeze at a time)
    g = clang.unsqueeze(g, unit_dims)

    put_grad(a, g)

    return fwd


register_grad(pids.BROADCAST_IN_DIM, _broadcast_in_dim_prim_grad)


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


def _squeeze_prim_grad(a: TensorProxy, /, dims: tuple[int, ...]) -> TensorProxy:
    fwd = prims.squeeze(a, tuple(dims))

    g = get_grad(fwd)
    # NOTE This calls clang.unsqueeze, and not torch.unsqueeze, because torch.unsqueeze only supports
    #   unsqueezing a single dimension
    a_grad = clang.unsqueeze(g, dims)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.SQUEEZE, _squeeze_prim_grad)


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


def _gather_prim_grad(a: TensorProxy, index: TensorProxy, dim: int) -> TensorProxy:
    fwd = prims.gather(a, index, dim)

    g = get_grad(fwd)
    # NOTE Intentionally not calling zeros_like to avoid preserving TensorProxy a.
    # TODO Update to call ltorch.zeros
    zeros = prims.full(a.shape, fill_value=0, device=a.device, dtype=a.dtype)
    a_grad = prims.scatter_add(zeros, index, g, dim)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.GATHER, _gather_prim_grad)


def _scatter_prim_grad(a: TensorProxy, /, index: TensorProxy, src: TensorProxy | Number, dim: int) -> TensorProxy:
    fwd = prims.scatter(a, index, src, dim)

    grad = get_grad(fwd)
    a_grad = prims.scatter(grad, index, 0, dim)
    put_grad(a, a_grad)

    if isinstance(src, TensorProxy):
        # NOTE: this is exactly what PyTorch is doing.
        # As such, it has the very same limitations. I.e.
        # the grad is not going to be correct unless the index list
        # (..., index[...], ...) does not have repeated elements
        src_grad = prims.gather(grad, index, dim)
        put_grad(src, src_grad)

    return fwd


register_grad(pids.SCATTER, _scatter_prim_grad)


def _index_copy_grad(a: TensorProxy, /, index: TensorProxy, src: TensorProxy, dim: int) -> TensorProxy:
    fwd = prims.index_copy(a, index, src, dim)

    grad = get_grad(fwd)

    # a_grad = grad.index_fill(dim, index, 0)
    # Unfortunately, we do not have `index_fill` for now
    # TODO: replace with `index_fill`
    grad_dim = utils.canonicalize_dim(grad.ndim, dim)
    index_len = len(index)
    index_unsqueeze_shape = [1] * grad.ndim
    index_unsqueeze_shape[grad_dim] = index_len
    index_expand_shape = list(grad.shape)
    index_expand_shape[grad_dim] = index_len
    a_grad = prims.scatter(grad, index.reshape(index_unsqueeze_shape).expand(*index_expand_shape), 0, dim)
    put_grad(a, a_grad)

    if src.ndim > 0:
        src_grad = prims.take(grad, index, dim).expand_as(src)
    else:
        src_grad = prims.take(grad, index.squeeze(0))
    put_grad(src, src_grad)

    return fwd


register_grad(pids.INDEX_COPY, _index_copy_grad)


def _scatter_add_prim_grad(a: TensorProxy, /, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    utils.check(
        not value._requires_grad or value.shape == index.shape,
        lambda: f"The gradient for the value Tensor is implemented only when value.shape == index.shape. "
        "value shape is {value.shape} while index shape is {index.shape}",
    )

    fwd = prims.scatter_add(a, index, value, dim)

    g = get_grad(fwd)
    # NOTE The value gradient is only correct when src.shape == index.shape.
    # See https://github.com/pytorch/pytorch/issues/27614#issuecomment-564648819
    value_grad = prims.gather(g, index, dim)
    put_grads((a, value), (g, value_grad))

    return fwd


register_grad(pids.SCATTER_ADD, _scatter_add_prim_grad)


def _take_along_axis_prim_grad(a: TensorProxy, index: TensorProxy, dim: int) -> TensorProxy:
    fwd = prims.take_along_axis(a, index, dim)

    g = get_grad(fwd)
    # NOTE Intentionally not calling zeros_like to avoid preserving TensorProxy a.
    # TODO Update to call ltorch.zeros
    zeros = prims.full(a.shape, fill_value=0, device=a.device, dtype=a.dtype)
    a_grad = prims.scatter_add(zeros, index, g, dim)
    put_grad(a, a_grad)

    return fwd


register_grad(pids.TAKE_ALONG_AXIS, _take_along_axis_prim_grad)


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


def _stride_order_prim_grad(a: TensorProxy, /, order: Sequence[int]) -> TensorProxy:
    fwd = prims.stride_order(a, order)

    g = get_grad(fwd)
    put_grad(a, g)

    return fwd


register_grad(pids.STRIDE_ORDER, _stride_order_prim_grad)


#
# Elementwise unary operator grads
#
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


def _erf_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.erf(a)

    g = get_grad(fwd)
    a_grad = 2 / math.sqrt(math.pi) * ltorch.exp(-(a**2)) * g
    put_grad(a, a_grad)

    return fwd


register_grad(pids.ERF, _erf_prim_grad)


def _exp_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.exp(a)

    g = get_grad(fwd)
    a_grad = g * fwd
    put_grad(a, a_grad)

    return fwd


register_grad(pids.EXP, _exp_prim_grad)


def _log_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.log(a)

    g = get_grad(fwd)
    a_grad = g / a
    put_grad(a, a_grad)

    return fwd


register_grad(pids.LOG, _log_prim_grad)


def _neg_prim_grad(a: Number | TensorProxy) -> Number | TensorProxy:
    fwd = prims.neg(a)

    g = get_grad(fwd)
    put_grad(a, -g)

    return fwd


register_grad(pids.NEG, _neg_prim_grad)


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
register_grad(pids.NE, prims.ne)
register_grad(pids.GE, prims.ge)
register_grad(pids.GT, prims.gt)
register_grad(pids.LE, prims.le)
register_grad(pids.LT, prims.lt)


def _mul_prim_grad(a: Number | TensorProxy, b: Number | TensorProxy, /) -> Number | TensorProxy:
    fwd = a * b

    g = get_grad(fwd)
    a_grad = b * g
    b_grad = a * g
    put_grads((a, b), (a_grad, b_grad))

    return fwd


register_grad(pids.MUL, _mul_prim_grad)


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


def _sort_prim_grad(
    a: TensorProxy, /, dim: None | int = None, descending: bool = False, stable: bool = False, *, out=None
) -> (TensorProxy, TensorProxy):
    dim = -1 if dim is None else dim
    sorted_a, sort_idx = prims.sort(a, dim, descending, stable, out=out)

    sorted_a_grad = get_grad(sorted_a)

    if a.ndim != 0:
        # TODO(nikitaved): replace with scatter once we have it.
        # scatter_add uses atomic ops which are slow!
        a_grad = ltorch.zeros_like(a)
        a_grad = ltorch.scatter_add(a_grad, dim, sort_idx, sorted_a_grad)
    else:
        a_grad = sorted_a_grad
    put_grad(a, a_grad)

    return sorted_a, sort_idx


register_grad(pids.SORT, _sort_prim_grad)


# TODO Fix division by zero when n_elem_reduced == 0 or when mean.numel == 0
#   by returning zeros_like(a) or similar.
# TODO Fix grad when correction > n_elem_reduced.
def _var_mean_prim_grad(a: TensorProxy, /, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    v, m = prims.var_mean(a, dims, correction=correction)

    gv = get_grad(v)
    gm = get_grad(m)

    n_elem_reduced = a.numel() // m.numel() if a.numel() != 0 else 1

    # Computes mean bwd
    mean_scale = 1.0 / n_elem_reduced
    mean_grad = mean_scale * restore_reduced_dims(gm, dims, a.shape)

    # Computes var bwd
    normalization_scalar = n_elem_reduced - correction
    restored_gv = restore_reduced_dims(gv, dims, a.shape)
    # Inserting a conversion to the same dtype to disable nvFuser executors's
    # bookend optimization (nv_enable_bookend), which can cause the backward
    # pass to generate two kernels
    mean_mdtype = prims.convert_element_type(m, m.dtype)
    restored_mean = restore_reduced_dims(mean_mdtype, dims, a.shape)
    var_grad = (2 * restored_gv * (a - restored_mean)) / normalization_scalar

    put_grad(a, mean_grad + var_grad)

    return v, m


register_grad(pids.VAR_MEAN, _var_mean_prim_grad)


#
# Linear algebra operator grads
#
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


def _maximum_grad(a: TensorProxy, b: TensorProxy, /):
    fwd = prims.maximum(a, b)

    g = get_grad(fwd)

    # NOTE: NaN propagation if either `a` or `b` is a NaN, then both elements receive `g` as gradient.
    # This is because comparison in presence of NaN is False (except for Not Equal which is always True).
    # Eg. Here `a` = NaN and `b` = 42
    # sub_g = where(NaN == 42 i.e. False, g / 2, g)  # sub_grad = g
    # grad_a = where(NaN < 42 i.e. False, 0., sub_g)  # grad_a = sub_g = g
    # grad_b = where(42 < NaN i.e. False, 0., sub_g)  # grad_b = sub_g = g
    # NOTE: If `g` is `NaN` then it will be propagated as the gradient of max element between a and b
    # and if both are equal, then as we evenly distributing the gradients, `NaN` will propagate through
    # the gradients of both `a` and `b`.

    # Compute sub-gradient if `a == b`
    # NOTE: We evenly distribute the gradient where the values are equal.
    sub_grad = prims.where(a == b, g / 2, g)

    a_grad = prims.where(a < b, 0.0, sub_grad)
    b_grad = prims.where(b < a, 0.0, sub_grad)

    put_grad(a, a_grad)
    put_grad(b, b_grad)
    return fwd


register_grad(pids.MAXIMUM, _maximum_grad)

# This operation creates no grad associations
register_grad(pids.ARGMAX, prims.argmax)

# This operation creates no grad associations
register_grad(pids.SHAPE, prims.shape)


def _copy_with_setitem_grad(a: TensorProxy, index, value: Number | TensorProxy):
    fwd = prims.copy_with_setitem(a, index, value)
    g = get_grad(fwd)

    a_grad = prims.copy_with_setitem(g, index, 0)
    put_grad(a, a_grad)

    if isinstance(value, TensorProxy):
        value_grad = g[index]
        expanded_dims = value_grad.ndim - value.ndim
        if expanded_dims > 0:
            value_grad = prims.sum(value_grad, tuple(range(expanded_dims)))
        put_grad(value, value_grad)

    return fwd


register_grad(pids.COPY_WITH_SETITEM, _copy_with_setitem_grad)


def _log_sigmoid_grad(
    a: TensorProxy,
) -> TensorProxy:
    from thunder.torch import abs, exp, log_sigmoid_backward, logsigmoid

    fwd = logsigmoid(a)

    g = get_grad(fwd)
    if a.device.type == "cpu":
        # NOTE PyTorch's CPU computation for logsigmoid's grad uses an additional "buffer" tensor, see
        # https://github.com/pytorch/pytorch/blob/7667235a23e2ffca4d32e6e16aa60a683418e159/torch/_decomp/decompositions.py#L332
        buffer = exp(-abs(a))
        a_grad = log_sigmoid_backward(g, a, buffer)
    else:
        # Here a placeholder tensor is provided.
        placeholder_buffer = empty((0,), device=a.device, dtype=a.dtype)
        a_grad = log_sigmoid_backward(g, a, placeholder_buffer)
    put_grad(a, a_grad)

    return fwd


register_grad("torch.nn.functional.logsigmoid", _log_sigmoid_grad)


#
# Phantom grad transform helpers
#


def _get_gradfn_and_executor(
    bsym: BoundSymbol, *, executors_list: Sequence[Any] = tuple()
) -> tuple[Callable | None, Executor | None]:
    cd = get_compile_data()
    executors_list = cd.executors_list if cd is not None else executors_list
    # Checks if the executor which has priority for this operation has a specific grad transform for it
    for ex in executors_list:
        if ex.can_execute_or_fuse(bsym):
            ex_grad_transform: None | Callable = ex.get_grad_transform(bsym.sym)
            if ex_grad_transform is not None:
                return ex_grad_transform, ex
            break

    # If the executor doesn't define its own grad transform, this just returns the default grad transform for the bsym
    gradfn = _grad_fn_map.get(bsym.sym.id, None)
    return gradfn, None


def grad(
    cfn,
) -> Callable:
    def grad(func):
        @wraps(func)
        def grad_func(*args, **kwargs):
            _, grads = value_and_grad(func)(*args, **kwargs)
            grads = tree_flatten(grads)[0]
            grads = [g for g in grads if g is not None]
            return grads

        return grad_func

    class _GradTransform(Transform):
        def transform_traces_pre_prologue(
            self,
            prologue_trc: Trace,
            computation_trc: Trace,
            epilogue_trc: Trace | None,
            *,
            executors_list: Sequence[Any],
        ) -> Trace:
            # Using trc.python_callable() makes it impossible to retrace the
            # function because the python_callable uses python_ctx which replaces
            # symbol occurrences with its symbol._call_ctx function
            computation_trc = dce(computation_trc)

            @wraps(computation_trc.python_callable())
            def python_callable(*args, **kwargs):
                return eval_trace(computation_trc, *args, **kwargs)["output"]

            # Don't DCE yet to keep argument unpackings
            gradtrc = construct_trace(use_dce=False)(
                grad(python_callable), *computation_trc.args, **computation_trc.kwargs
            )

            gradtrc = wrap_return_value_together_with_arguments(gradtrc)
            gradtrc = dce(gradtrc)
            grad_output = gradtrc.output
            pro_to_epi = prologue_trc.output[1]
            if type(grad_output) == dict:
                grad_output = grad_output["output"]

            def new_epilogue(*args):
                return args

            new_epilogue_trc = construct_trace()(new_epilogue, *pro_to_epi, *grad_output)

            return prologue_trc, gradtrc, new_epilogue_trc

    cfn._using_grad_transform = True
    _grad_transform = _GradTransform()
    return add_transform(cfn, transform=_grad_transform, disable_torch_autograd_support=True)


@lru_cache(maxsize=None)
def symbol_to_eval(bound_symbol):
    """Map a BoundSymbol to a function that evaluates it.

    Args:
        bound_symbol: BoundSymbol to map
    """
    # Symbol is callable
    return bound_symbol.sym


def unwrap_one_level_of_subsymbols(trace):
    new_symbols_iter = (
        bound_symbol.subsymbols if len(bound_symbol.subsymbols) > 0 else [bound_symbol]
        for bound_symbol in trace.bound_symbols
    )
    new_symbols = list(chain.from_iterable(new_symbols_iter))
    trace.bound_symbols = new_symbols
    return trace


# VJP transform
# =============
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
    prims.PrimIDs.LOG10: lambda x: (prims.log10(x), (x,)),
    prims.PrimIDs.LOG1P: lambda x: (prims.log1p(x), (x,)),
    prims.PrimIDs.LOG2: lambda x: (prims.log2(x), (x,)),
    prims.PrimIDs.ZETA: lambda x, y: (prims.zeta(x, y), (x, y)),
    prims.PrimIDs.FMOD: lambda x, y: (prims.fmod(x, y), (x, y)),
    prims.PrimIDs.COPY_: lambda x, y: (prims.copy_(x, y), tuple()),
    prims.PrimIDs.CLONE: lambda x: (prims.clone(x), tuple()),
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
    prims.PrimIDs.LOG10: lambda x, g: g / (x * 2.302585092994046),
    prims.PrimIDs.LOG1P: lambda x, g: g / (x + 1),
    prims.PrimIDs.LOG2: lambda x, g: g / (x * 0.6931471805599453),
    prims.PrimIDs.FMOD: lambda x, y, g: (g, -g * prims.trunc(x / y)),
    # The copy should not be differentiable. We return None to enable the generation of the backward graph through them.
    prims.PrimIDs.COPY_: lambda g: (None, None),
    prims.PrimIDs.CLONE: lambda g: g,
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

    # if x is CUDA tensor and y is CPU scalar tensor, gy should be on CPU
    if gy is not None and gy.device.type != y.device.type:
        gy = gy.to(device=y.device)
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
    n_elem_reduced = a.numel() // v.numel() if a.numel() != 0 else 1
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

    primals = split(a, split_size_or_sections, dim)
    # save `dtype, device and output shape` as few output can be unused user function
    # leading to incoming gradients being `None`
    # in which case we will create zeros as gradient to be passed to `cat`
    residuals = (dim, a.dtype, a.device, tuple(primal.shape for primal in primals))
    return VJPDual(primals, residuals)


@register_backward("torch.split")
def split_backward(dim, dtype, device, out_shapes, *grads):
    from thunder.torch import cat, zeros

    assert len(out_shapes) == len(grads)

    def make_zeros_like(shape):
        return zeros(shape, dtype=dtype, device=device)

    grads = tuple(
        grad if grad is not None else make_zeros_like(out_shape) for grad, out_shape in zip(grads, out_shapes)
    )

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
    if g.numel() <= 1 or g.shape[dim] == 1:
        return g
    return g.flip(dim).cumsum(dim).flip(dim)


@register_augmented_forward("torch.softmax")
def softmax_aug_fwd(a: Proxy, dim: int, dtype: dtypes.dtype | None = None) -> VJPDual:
    from thunder.torch import softmax

    primal = softmax(a, dim, dtype=dtype)
    residuals = (primal, dim, a.dtype)
    return VJPDual(primal, residuals)


@register_backward("torch.softmax")
def softmax_backward(primal, dim, input_dtype, g):
    grad = primal * (g - (primal * g).sum(dim, keepdim=True))
    return grad.to(input_dtype) if grad.dtype != input_dtype else grad


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
        if symbol.sym.id in trace_interpreter_skip_list:
            continue
        elif symbol.output is None:
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


if torch.distributed.is_available():
    from torch.distributed import ReduceOp
    from torch.distributed import distributed_c10d as c10d
    from torch._C._distributed_c10d import _resolve_process_group

    if TYPE_CHECKING:
        from torch.distributed import ProcessGroup
        from thunder.distributed.prims import DistributedReduceOps

    @register_augmented_forward("torch.ops._c10d_functional.all_reduce")
    def functional_all_reduce_augmented_forward(
        a: TensorProxy,
        /,
        op: str | ReduceOp | DistributedReduceOps = ReduceOp.SUM,
        group: None | ProcessGroup | str = None,
        async_op: bool = False,
        **kwargs,
    ) -> VJPDual:
        from thunder.torch import all_reduce

        if isinstance(group, str):
            group = _resolve_process_group(group)
        primal = all_reduce(a, op=op, group=group)
        residuals = (op, group)
        return VJPDual(primal, residuals)

    @register_backward("torch.ops._c10d_functional.all_reduce")
    def functional_all_backward(op, group, g) -> TensorProxy:
        from thunder.torch import all_reduce

        return all_reduce(g, op=op, group=group)


def sum_to(a: TensorProxy, shape: Sequence[int]) -> TensorProxy:
    if utils.same_shape(a.shape, shape):
        return a
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


nondifferentiable_vjp_symbols: set[prims.PrimIDs] = {
    prims.PrimIDs.BITWISE_AND,
    prims.PrimIDs.BITWISE_OR,
    prims.PrimIDs.BITWISE_NOT,
    prims.PrimIDs.BITWISE_XOR,
    prims.PrimIDs.SIGNBIT,
    prims.PrimIDs.FULL,
}


def is_constant_for_vjp(symbol: prims.Symbol) -> bool:
    """Check if a symbol is constant for the VJP transform.

    Args:
        symbol (prims.Symbol): Symbol to check.

    Returns:
        bool: True if the symbol is constant, False otherwise.
    """
    are_all_args_non_differentiable = not any(isinstance(arg, (FloatProxy, TensorProxy)) for arg in symbol.flat_args)
    # Symbol's tag their output in `torch.no_grad` regions with `DETACHED_AUTOGRAD_GRAPH`.
    # These are treated as constant for VJP.
    # NOTE - `any(()) is False`
    output_disconnected_from_graph = any(
        ProxyTag.DETACHED_AUTOGRAD_GRAPH in o.tags for o in symbol.flat_outs if isinstance(o, TensorProxy)
    )
    return (
        are_all_args_non_differentiable
        or symbol.are_all_args_constant
        or symbol.sym.id in nondifferentiable_vjp_symbols
        or output_disconnected_from_graph
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

    if _get_gradfn_and_executor(symbol)[0] is not None:
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

    if bsym.sym.id in trace_interpreter_skip_list:
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


def augmented_forward_pass_trace(trace: Trace, /, *args, **kwargs):
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
    trace, result, env = interpret_trace_to_trace(
        trace,
        *args,
        **kwargs,
        with_env=True,
        symbol_mapper=vjp_symbol_mapper,
    )
    result = tree_map(lambda x: x.primal if isinstance(x, VJPDual) else x, result)
    return trace, result, env


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
        elif dataclasses.is_dataclass(v) and dataclasses.is_dataclass(val):
            safe_map_flat(put_grad, tree_flatten_with_dataclass(v), tree_flatten_with_dataclass(val))
        else:
            # Skip writing to constants
            pass

    if isinstance(init_cotangents, Sequence) and len(init_cotangents) == 1 and not isinstance(trace.output, Sequence):
        init_cotangents = init_cotangents[0]
    safe_map_flat(put_grad, trace.output, init_cotangents)

    for symbol in reversed(list(iter_bound_symbols(trace.bound_symbols))):
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

        if _get_gradfn_and_executor(symbol)[0] is not None:
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


def vjp_call(primals, cotangents, trace: Trace, **kwargs):
    # Assuming primals is flat

    if not isinstance(primals, Sequence):
        primals = (primals,)

    result, env = augmented_forward_pass(*primals, trace=trace, **kwargs)
    check(
        len(result) == len(cotangents) if isinstance(result, Sequence) else True,
        lambda: f"Expected cotangents to be a sequence of length {len(result)}, got a sequence of length {len(cotangents)}",
    )
    return result, backward_pass(env, trace, cotangents)


def vjp(func):
    """Computes the VJP of a function.

    Args:
        func (Callable): Function to be differentiated.
    """

    def _vjp(primals, cotangents, **kwargs):
        flat_func, flat_args, spec = flatten_func(func, primals, kwargs)
        trace = construct_trace()(flat_func, *flat_args)
        result, vjp_result = vjp_call(flat_args, cotangents, trace=trace)
        # If the argument is a CPU scalar tensor, its gradient needs to be summed into a scalar tensor.
        vjp_result = tuple(
            (
                sum_to(grad, arg._shape)
                if (grad is not None and isinstance(arg, TensorProxy) and arg.device.type == "cpu")
                else grad
            )
            for grad, arg in zip(vjp_result, flat_args)
        )
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
            return None

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
    forward_trace_producers = utils.producers(forward_trace)
    saved_for_backward = tree_map(
        lambda x: x.value if isinstance(x, NumberProxy) and x not in forward_trace_producers else x, saved_for_backward
    )
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

    # When thunder.executors.torch_autograd.ThunderFunction.backward calls backward_fn, it copies
    # collections into mutable ones, so that the tensors will be deallocated when deleted.
    # See ThunderFunction.backward's notes for details
    saved_tensors = list(saved_tensors)
    unpacking_trace = construct_trace(rename_proxies=False, use_dce=False)(
        unpacking_fn, [saved_tensors, saved_other], cotangents
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
        >>> from thunder import jit, last_traces
        >>> from thunder.core.transforms import forward_and_backward_from_trace
        >>> def f(x):
        ...     return torch.sin(x)
        >>> x = torch.tensor(3.0)
        >>> cf = jit(f)
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

    forward_trace, result, env = augmented_forward_pass_trace(trace, *trace.args, **trace.kwargs)
    forward_trace.tags.add(TraceTag.AUGMENTED_FORWARD)
    saved_for_backward = deconstruct_forward_env_for_backward(trace, env)

    # The custom torch.autograd.Function only considers Tensors in the input/output (not ones that are nested inside python data structures)
    flat_output, output_spec = tree_flatten_with_dataclass(result["output"])
    if not torch_autograd:  # needed?
        output_spec = None
    result["flat_output"] = tuple(flat_output)

    assert forward_trace.bound_symbols.pop(-1).sym is prims.python_return
    with tracectx(forward_trace):
        prims.python_return((result, saved_for_backward))

    def ones_like(x):
        if isinstance(x, TensorProxy):
            return full_like(x, fill_value=1)
        elif isinstance(x, NumberProxy):
            return type(x.value)(1)
        else:
            return None

    # We set forward trace to construct proxies because we need these proxies to
    # have different names than the ones in the forward trace.
    try:
        tracectx_token = set_tracectx(forward_trace)
        # We don't want to record those ones_like calls in the forward trace.
        with detached_trace():
            if torch_autograd:
                flat_output = forward_trace.output[0]["flat_output"]
                cotangents = utils.sequencify(tree_map(lambda v: ones_like(v), flat_output))
            else:
                cotangents = utils.sequencify(tree_map(lambda v: ones_like(v), trace.output["output"]))
    finally:
        reset_tracectx(tracectx_token)

    saved_for_backward = forward_trace.output[1]
    flat_saves, _ = tree_flatten(saved_for_backward)
    trace_with_unwrapped_return = unwrap_return_value(trace)

    def backward_fn(saved_for_backward, cotangents):
        # trace converts all saved_for_backward into proxy, we want to restore number scalars afterwards.
        flat_saves_proxified, saves_spec = tree_flatten(saved_for_backward)
        flat_filtered = [
            proxified if isinstance(entry, Proxy) else entry
            for proxified, entry in zip(flat_saves_proxified, flat_saves)
        ]
        saved_for_backward = tree_unflatten(flat_filtered, saves_spec)
        env = reconstruct_forward_env_for_backward(trace_with_unwrapped_return, saved_for_backward)

        if torch_autograd:
            cotangents = tree_unflatten(cotangents, output_spec)
        out = backward_pass(env, trace_with_unwrapped_return, cotangents)
        if torch_autograd:
            gkwargs = out[-1] if isinstance(out[-1], dict) else {}
            gargs = out[:-1] if isinstance(out[-1], dict) else out
            gkwargs = {k: gkwargs.get(k, None) for k in trace_with_unwrapped_return.kwargs}
            out = (*gargs, gkwargs)
            out = tree_flatten(out)[0]
        return out

    backward_trace = construct_trace(rename_proxies=False, _used_names=forward_trace.names)(
        backward_fn, saved_for_backward, cotangents
    )

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

    enable_saved_for_backward_recomputation: None | bool = get_compile_option(
        "enable_saved_for_backward_recomputation", "Enable save for backward tensors recomputation."
    )
    if enable_saved_for_backward_recomputation:
        forward_trace, backward_trace = recompute_saved_for_backward(forward_trace, backward_trace)

    return ForwardBackwardTraces(forward_trace, backward_trace)


def recompute_saved_for_backward(fwd_trace: Trace, bwd_trace: Trace) -> tuple[Trace, Trace]:
    """Generates the pair of traces with rematerializaion of the saved-for-backward tensors.
    Args:
        fwd_trace (Trace): forward trace where to get the saved for backward from.
        bwd_trace (Trace): backward trace where to recompute the saved for backward to.

    Returns:
        tuple[Trace, Trace]: A tuple containing the new forward and backward traces.
    """

    start_time_ns = time.perf_counter_ns()

    saved_for_bw = get_saved_for_backward_tensors(fwd_trace)
    fwd_trace_args = {variableify(j) for j in fwd_trace.args}
    old_saved_for_bwd = {variableify(j) for j in saved_for_bw}

    all_rematerializable = old_saved_for_bwd - fwd_trace_args

    remat_policy: None | Callable[[set[Variable]], set[Variable]] = get_compile_option(
        "recomputation_policy",
        "A callable that accepts a set of variables and returns a set of the variables that are allowed to be recomputed from the forward in the backward trace. The compile option `enable_saved_for_backward_recomputation` needs to be true for this policy to take effect.",
    )

    if remat_policy:
        rematerializable = remat_policy(all_rematerializable)
    else:
        rematerializable = all_rematerializable

    producers = find_producer_symbols(fwd_trace, tuple(unvariableify(i) for i in rematerializable), fwd_trace.args)

    required_fw_args = fwd_trace_args & old_saved_for_bwd
    recomputed_tensors_from_producers = set()
    for prod in producers:
        for prod_arg in prod.flat_args:
            prod_arg = variableify(prod_arg)
            if prod_arg in fwd_trace_args:
                required_fw_args.add(prod_arg)
        for prod_out in prod.flat_outs:
            recomputed_tensors_from_producers.add(variableify(prod_out))

    required_saved_for_bwd = all_rematerializable - rematerializable - recomputed_tensors_from_producers
    new_saved_for_backward = tuple(unvariableify(i) for i in required_fw_args | required_saved_for_bwd)

    new_fwd_trace = from_trace(fwd_trace)
    new_fwd_trace.bound_symbols = fwd_trace.bound_symbols.copy()
    new_return_args = (fwd_trace.output[0], (new_saved_for_backward, fwd_trace.output[1][1]))
    new_fwd_trace.bound_symbols[-1] = prims.python_return.bind(*new_return_args, output=None)

    new_bwd_trace = from_trace(bwd_trace)
    # In cases where C0 name is carried from previous trace it must be removed
    # as the proxy needs to register with that specific name to follow the backward
    # trace standard signature.
    new_bwd_trace.names.discard("C0")

    with tracectx(new_bwd_trace):
        unpack_args = (CollectionProxy(new_saved_for_backward, name="C0"), len(new_saved_for_backward))

    # Here we make sure that the signature of the backward trace is the same as the one we expect.
    # This part of the trace is the unpacking of the tuple passed from the forward trace,
    # more specifically, C0 unpacks into the saved for backward tensors and C1 into the cotangents
    # used to compute the vector-Jacobian product.
    assert bwd_trace.bound_symbols[4].sym.id == prims.PrimIDs.UNPACK_SEQUENCE
    assert bwd_trace.bound_symbols[4].args[0].name == "C0"
    assert bwd_trace.bound_symbols[5].sym.id == prims.PrimIDs.UNPACK_SEQUENCE
    assert bwd_trace.bound_symbols[5].args[0].name == "C1"

    for idx, bsym in enumerate(bwd_trace.bound_symbols):
        if idx == 4:
            new_unpack = prims.unpack_sequence.bind(*unpack_args, output=new_saved_for_backward)
            new_bwd_trace.bound_symbols.append(new_unpack)
        elif idx == 6:
            new_bwd_trace.bound_symbols.extend(producers)
            new_bwd_trace.bound_symbols.append(bsym)
        else:
            new_bwd_trace.bound_symbols.append(bsym)

    new_bwd_trace.args = [(new_saved_for_backward, fwd_trace.output[1][1]), *bwd_trace.args[1:]]

    elapsed_time_ns = time.perf_counter_ns() - start_time_ns
    new_bwd_trace.set_provenance(
        TraceProvenance(f"Saved for backward remat trace (took {elapsed_time_ns * 1e-6:.2f} milliseconds)")
    )
    new_fwd_trace.set_provenance(
        TraceProvenance(f"Saved for backward remat trace (took {elapsed_time_ns * 1e-6:.2f} milliseconds)")
    )

    return new_fwd_trace, new_bwd_trace
