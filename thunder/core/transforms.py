from contextlib import nullcontext
from dataclasses import dataclass
from enum import auto, Enum
from itertools import chain
from functools import lru_cache, partial, wraps
from numbers import Number
from typing import Any, Callable, Dict, Tuple, Union, Optional
from collections.abc import Sequence

from thunder import _make_trace as make_trace
from thunder.core import dtypes, prims
from thunder.clang import full, full_like, unsqueeze, squeeze
from thunder.core.devices import cpu, Device
from thunder.core.langctx import get_langctx, set_langctx, reset_langctx, get_default_langctx
from thunder.core.proxies import NumberProxy, Proxy, TensorProxy
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.trace import TraceCtx as Trace
from thunder.core.trace import VariableInterface as Variable
from thunder.core.trace import detached_trace, get_tracectx
from thunder.core.utils import check, flatten_func, safe_map, safe_map_flat, safe_zip, unzip2, const_as, sequencify

# from thunder.executors.torch import ops_to_torch_ops_map

import torch

import numpy as np


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
    allow_duplicates_list = (Transforms.JvpOp, Transforms.VjpOp)
    write_with_duplicates = partial(write, allow_duplicates=True)

    for symbol in trace.bound_symbols:
        if symbol.sym.id in transform_skip_list:
            continue
        args = tree_map(read, symbol.args)
        kwargs = tree_map(read, symbol.kwargs)
        prim_func = symbol_mapper(symbol)
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
        trace = make_trace(func)(*args, **kwargs)
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
        trace = make_trace(func)(*args, **kwargs)
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
        trace = make_trace(func)(*args, **kwargs)
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
    trace = make_trace(fn)(*unbatched_args, **kwargs)
    trace = unwrap_one_level_of_subsymbols(trace)
    outs = _vmap_call_metafunc(args, in_dims, 0, axis_size, trace=trace, detached=False, **kwargs)
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


def remove_batch_dim(tensor: TensorProxy, batch_dim: int = 0):
    """Removes the batch dimension from a tensor.

    Args:
        tensor (TensorProxy): Tensor to remove the batch dimension from.

    Returns:
        TensorProxy: Tensor with the batch dimension removed.
    """
    trace = get_tracectx()
    new_shape = tensor.shape[:batch_dim] + tensor.shape[batch_dim + 1 :]
    return TensorProxy(shape=new_shape, dtype=tensor.dtype, device=tensor.device)


# TODO: in JAX args, in_dims are flattened the same way
# TODO: in JAX out_dims are flattened as well
def _vmap_call_metafunc(args, in_dims, out_dims, axis_size, trace: Trace, detached: bool, **kwargs):
    """Metafunction for vmap call.

    Args:
        args (Tuple[Proxy]): Arguments to the function.
        in_dims (Tuple[int]): Batch dimension for each argument.
        out_dims (Tuple[int]): Batch dimension for return values.
        trace (Trace): Trace to use for the function.
        detached (bool): Whether to detach the trace.
        kwargs: Keyword arguments.

    Raises:
        AssertionError: If the vmap for keyword arguments is not implemented.

    Returns:
        Result of the vmap transform.
    """
    assert len(kwargs) == 0, "vmap for kwargs is not implemented"

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
        result = eval_trace(trace, *batched_args, symbol_mapper=partial(vmap_symbol_mapper, axis_size=axis_size))
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


vmap_call = make_transform_prim(Transforms.VmapOp, "vmap_call", meta=partial(_vmap_call_metafunc, detached=True))
inline_transforms_map[Transforms.VmapOp] = partial(_vmap_call_metafunc, detached=False)


# TODO: how should we handle out_dims here?
# although here we are calling vmap of identity, so we should know from the call to vmap
# This should be fine. If we have vmap(identity(func), out_dims=N) then this rule is first used
# to get the vmapped result of identity(func) in vmap_symbol_mapper, and out_dims is handled
# after that in the outer _vmap_call_metafunc.
def _identity_call_vmap(axis_size, *batched_args, trace: Trace, **kwargs):
    args, in_dims = unzip2(batched_args)
    out_dims = 0  # Fixme
    outs, out_dims = _vmap_call_metafunc(args, in_dims, out_dims, axis_size, trace=trace, detached=False, **kwargs)
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
        trace = make_trace(func_flat)(*unbatched_args_flat)
        outs = vmap_call(args_flat, in_dims_flat, out_dims, axis_size=axis_size, trace=trace)
        return outs

    return wrapper


def vmap_eager(func, args, in_dims=0, out_dims=0, axis_size=None, executor="torch"):
    """Computes the vmap of a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.
        args (_type_): Args of the function.
        executor (str, optional): Executor to use. Defaults to "torch".

    Returns:
        The result of the vmapped function.
    """
    # TODO: fix this - not all args may be batched
    # TODO: here we assume batch axis is 0
    vmap_trace = make_trace(
        inline(vmap(func, in_dims=in_dims, out_dims=out_dims, axis_size=axis_size)), executor=executor
    )(*args)
    vmap_traced = make_traced(partial(eval_trace, vmap_trace), executor=executor)
    return vmap_traced(*args)


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


def _jvp_call_metafunc(primals, tangents, trace: Trace, detached: bool, **kwargs):
    """Metafunction for the JVP transform.

    Args:
        primals (Tuple[Proxy]): Primal values.
        tangents (Tuple[Proxy]): Tangent values.
        trace (Trace): Trace of the function to be transformed.
        detached (bool): Whether to detach the trace.
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


jvp_call = make_transform_prim(Transforms.JvpOp, "jvp_call", meta=partial(_jvp_call_metafunc, detached=True))
inline_transforms_map[Transforms.JvpOp] = partial(_jvp_call_metafunc, detached=False)


def _identity_call_jvp(*args: JVPDual, trace: Trace, **kwargs):
    primals, tangents = unzip2(args)
    out_primals, out_tangents = _jvp_call_metafunc(primals, tangents, trace, detached=False, **kwargs)
    if isinstance(out_primals, Sequence):
        return safe_map(pair_to_jvp_dual, safe_zip(out_primals, out_tangents))
    return JVPDual(out_primals, out_tangents)


jvp_impls[Transforms.IdentityOp] = _identity_call_jvp


def _vmap_call_jvp(args: JVPDual, in_dims, out_dims, axis_size, trace: Trace, **kwargs):
    primals, tangents = safe_zip(*args)
    in_dims, _ = safe_zip(*in_dims)
    out_dims, _ = safe_zip(*out_dims)
    vmapped_trace = make_trace(
        inline(vmap(partial(eval_trace, trace), in_dims=in_dims, out_dims=out_dims, axis_size=axis_size))
    )(*primals)
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
        trace = make_trace(func)(*primals)
        return jvp_call(primals, tangents, trace=trace)

    return wrapper


def jvp_eager(func, primals, tangents, executor="torch"):
    """Computes the Jacobian-vector product of a Thunder function.

    Args:
        func (Callable): A Thunder function to be transformed.
        primals (_type_): Primals of the function.
        tangents (_type_): Tangents of the function.
        executor (str, optional): Executor to use. Defaults to "torch".

    Returns:
        The result of the Jacobian-vector product.
    """
    trace = make_trace(func, executor=executor)(*primals)

    def jvp_func(*primals_and_tangents):
        _primals, _tangents = primals_and_tangents[: len(primals)], primals_and_tangents[len(primals) :]
        return _jvp_call_metafunc(_primals, _tangents, trace, detached=False)

    jvp_trace = make_trace(jvp_func, executor=executor)(*primals, *tangents)
    jvp_traced = make_traced(partial(eval_trace, jvp_trace), executor=executor)
    return jvp_traced(*primals, *tangents)


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
            else:
                raise ValueError(f"zeros_like inside JVP got an unsupported type {type(x)}")

        return tuple(zeros_like(arg) for arg in forward_args)


# Mapping from symbols to augmented primal (forward) functions used in VJP
# The augmented_primal function takes the primal values and returns the primal
# result and the residuals (saved values for the backward).
augmented_forward_impls = {
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
    prims.PrimIDs.MUL: lambda x, y: (prims.mul(x, y), (x, y)),
    prims.PrimIDs.SIN: lambda x: (prims.sin(x), (x,)),
    prims.PrimIDs.SINH: lambda x: (prims.sinh(x), (x,)),
    prims.PrimIDs.SUB: lambda x, y: (prims.sub(x, y), tuple()),
    prims.PrimIDs.EQ: lambda x, y: (prims.eq(x, y), (x, y)),
    prims.PrimIDs.GE: lambda x, y: (prims.ge(x, y), (x, y)),
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
    prims.PrimIDs.MUL: lambda x, y, g: (g * y, g * x),
    prims.PrimIDs.SIN: lambda x, g: prims.mul(g, prims.cos(x)),
    prims.PrimIDs.SINH: lambda x, g: prims.mul(g, prims.cosh(x)),
    prims.PrimIDs.SUB: lambda g: (g, -g),
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
    from thunder import clang

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
    dx = gresult * y / x
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
def slice_aug_fwd(a, start_indices, end_indices, strides=None):
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

    if strides is None:
        return result, None, None
    return result, None, None, None


@register_augmented_forward(prims.PrimIDs.BROADCAST_IN_DIM)
def broadcast_in_dim_aug_fwd(a: Proxy, shape: Sequence[int], broadcast_dimensions: Sequence[int]) -> VJPDual:
    primal = prims.broadcast_in_dim(a, shape, broadcast_dimensions)
    residuals = (a, shape, broadcast_dimensions)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.BROADCAST_IN_DIM)
def broadcast_in_dim_backward(a, shape, broadcast_dimensions, g):
    # If g is None, then the primal was a constant and the pullback is zero.
    # TODO: implement None propagation in the VJP infrastructure so that we don't need to do this.
    if g is None:
        return None, None, None
    unit_dims = tuple(i for i, s in enumerate(a.shape) if s == 1)
    bcast_dims = tuple(b for i, b in enumerate(broadcast_dimensions) if i not in unit_dims)
    reduce_dims = tuple(s for i, s in enumerate(range(len(shape))) if i not in bcast_dims)
    g = prims.sum(g, reduce_dims)
    g = unsqueeze(g, unit_dims)
    # One return per positional argument of prims.broadcast_in_dim
    return g, None, None


@register_augmented_forward(prims.PrimIDs.CONVERT_ELEMENT_TYPE)
def convert_element_type_aug_fwd(a: Proxy, dtype: dtypes.dtype) -> VJPDual:
    primal = prims.convert_element_type(a, dtype)
    residuals = (primal,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.CONVERT_ELEMENT_TYPE)
def convert_element_type_backward(a, g):
    # perform cast back to input type during backward
    return prims.convert_element_type(g, a.dtype), None


@register_augmented_forward("torch.nn.functional.embedding")
def embedding_aug_fwd(
    a: Proxy,
    weight: Proxy,
    *,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
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
    return None, gweight


@register_augmented_forward(prims.PrimIDs.MATMUL)
def matmul_aug_fwd(a: TensorProxy, b: TensorProxy) -> VJPDual:
    primal = prims.matmul(a, b)
    residuals = (a, b)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.MATMUL)
def matmul_backward(a, b, g):
    last_dim = (-1,)
    first_dim = (-2,)
    if a.ndim == 1 and b.ndim == 1:
        return g * b, g * a

    if b.ndim == 1:
        ga = unsqueeze(g, last_dim) @ unsqueeze(b, last_dim).mT
        gb = a.mT @ unsqueeze(g, last_dim)
        if g.ndim > 1:
            gb = squeeze(gb, last_dim)
            gb = prims.sum(gb, tuple(range(gb.ndim - 1)))
        return ga, gb

    if a.ndim == 1:
        ga = unsqueeze(g, first_dim) @ b.mT
        if g.ndim > 1:
            ga = prims.sum(ga, tuple(range(ga.ndim - 1)))
        gb = unsqueeze(a, first_dim).mT @ unsqueeze(g, first_dim)
        return ga, gb

    return g @ b.mT, a.mT @ g


# TODO: Remove registration against torch.nn.functional.linear once we have a
#       better way to handle VJPs for composite functions that are not in the prims/corelang
@register_augmented_forward(torch.nn.functional.linear)
def linear_aug_fwd(a: TensorProxy, b: TensorProxy, c: Optional[TensorProxy] = None) -> VJPDual:
    primal = prims.linear(a, b, c)
    residuals = (a, b, c)
    return VJPDual(primal, residuals)


@register_backward(torch.nn.functional.linear)
def linear_backward(a, b, c, g):
    from thunder.langs.torch import matmul

    first_dim = (-2,)
    ga = matmul(g, b)
    if a.ndim == 1:
        gb = matmul(unsqueeze(a, first_dim).mT, unsqueeze(g, first_dim)).mT
    else:
        gb = matmul(a.mT, g).mT
    if c is None:
        return ga, gb
    gc = prims.sum(g, tuple(range(g.ndim - 1))) if g.ndim > 1 else g
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


def decomposed_fn_aug_fwd_rule(*args, decomposed_fn, **kwargs):
    """Augmented forward rule for composite functions implemented in terms of other functions that are
    supposed to be supported by the VJP infrastructure.

    Args:
        decomposed_fn (Callable): decomposed version of the function

    Returns:
        Callable: Augmented forward rule for the composite function
    """
    trace = make_trace(decomposed_fn)(*args, **kwargs)
    trace = unwrap_one_level_of_subsymbols(trace)
    result, env = augmented_forward_pass(*args, trace=trace, **kwargs)
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
    residuals = (args, kwargs, saved_for_backward)
    return VJPDual(result, residuals)


def decomposed_fn_backward_rule(decomposed_fn, args, kwargs, saved_for_backward, *grads):
    trace = make_trace(decomposed_fn)(*args, **kwargs)
    trace = unwrap_one_level_of_subsymbols(trace)
    bound_symbols = iter_bound_symbols(trace.bound_symbols)
    reconstructed_env = {
        sequencify(symbol.output)[0].name: VJPDual(None, saved_for_backward[i])
        for i, symbol in enumerate(bound_symbols)
    }
    result = backward_pass(reconstructed_env, trace, grads)
    if len(args) == 1:
        return result[0]
    return result


@register_augmented_forward(prims.PrimIDs.WHERE)
def where_aug_fwd(condition: TensorProxy, x: TensorProxy, y: TensorProxy) -> VJPDual:
    primal = prims.where(condition, x, y)
    residuals = (condition,)
    return VJPDual(primal, residuals)


@register_backward(prims.PrimIDs.WHERE)
def where_backward(condition, g):
    return None, prims.where(condition, g, 0.0), prims.where(condition, 0.0, g)


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
    if symbol.are_all_args_constant:

        def vjp_impl_const(symbol, *args, **kwargs):
            primals = symbol_to_eval(symbol)(*args, **kwargs)
            n_args = len(args)
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
    result = tree_map(lambda x: x.primal, result)
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

    def read(x: Variable):
        if isinstance(x, Variable):
            return env[x.name]
        else:
            return x

    def write(v: Variable, val: Any) -> None:
        if isinstance(v, Variable):
            if v.name in env:
                # Accumulate cotangents
                env[v.name] = prims.add(env[v.name], val)
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

    if isinstance(trace.output, Sequence):
        safe_map(write, tuple(trace.output), init_cotangents)
    else:
        write(trace.output, init_cotangents)

    for symbol in reversed(trace.bound_symbols):
        if symbol.sym.id in transform_skip_list:
            continue
        symbol_output = sequencify(symbol.output)
        cotangents = safe_map(read, symbol_output)
        # Having a single cotangent is a common case, so we flatten it
        # Otherwise, we will need to rewrite the pullback functions
        cotangents = tree_flatten(cotangents)[0]
        residuals = forward_env[symbol_output[0].name].residuals
        if symbol.are_all_args_constant:
            # We can skip the pullback if all the arguments are constant
            continue
        pullback = backward_impls[symbol.sym.id]
        result = pullback(*residuals, *cotangents)
        if not isinstance(result, Sequence):
            result = (result,)
        check(
            len(result) == len(symbol.args),
            lambda: f"Pullback for {symbol.sym.id} returned {len(result)} values, but expected {len(symbol.args)}",
        )
        safe_map(write, symbol.args, result)

    def read_with_none(x: Variable):
        if isinstance(x, Variable):
            # Return None if the variable was not used in the computation and
            # hence not in the env
            return env.get(x.name, None)
        else:
            return x

    return tree_map(read_with_none, tuple(trace.args))


def vjp_call_metafunc(primals, cotangents, trace: Trace, detached, **kwargs):
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


vjp_call = make_transform_prim(Transforms.VjpOp, "vjp_call", meta=partial(vjp_call_metafunc, detached=True))
inline_transforms_map[Transforms.VjpOp] = partial(vjp_call_metafunc, detached=False)


def vjp(func):
    """Computes the VJP of a function.

    Args:
        func (Callable): Function to be differentiated.
    """

    def _vjp(primals, cotangents, **kwargs):
        flat_func, flat_args, spec = flatten_func(func, primals, kwargs)
        trace = make_trace(flat_func)(*flat_args)
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
        trace = make_trace(func)(*args, **kwargs)
        cotangents = tree_map(lambda v: ones_like(v), trace.output)
        return vjp(func)(args, cotangents, **kwargs)

    return _value_and_grad
