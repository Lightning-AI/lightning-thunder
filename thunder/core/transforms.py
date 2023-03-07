from contextlib import nullcontext
from dataclasses import dataclass
from enum import auto, Enum
from functools import lru_cache, partial
from numbers import Number
from typing import Any, Callable, Dict, Sequence, Tuple, Union

from thunder import make_trace, make_traced
from thunder.core import prims
from thunder.core.lang import full, full_like
from thunder.core.proxies import NumberProxy, Proxy, TensorProxy
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.trace import detached_trace, get_trace, Trace, Variable
from thunder.core.utils import check, flatten_func, safe_map, safe_map_flat, safe_zip, unzip2
from thunder.executors.torch import ops_to_torch_ops_map


class Transforms(Enum):
    IdentityOp = auto()
    VmapOp = auto()
    JvpOp = auto()
    VjpOp = auto()


@lru_cache(maxsize=None)
def symbol_to_eval(symbol: prims.Symbol):
    """Map a symbol to a function that evaluates it.

    Args:
        symbol: symbol to map
    """
    meta_func = prims.ops_to_meta_functions_map[symbol.op]

    prim_func = getattr(prims, symbol.name, None)
    if prim_func is not None:
        return prim_func

    return prims.eval_meta_and_record_symbol_fn(meta_func, symbol.op, symbol.name)


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

    def write(v: Variable, val: Any) -> None:
        if not isinstance(v, Variable):
            return
        # Duplicates are allowed but ignored
        if v.name in env:
            return
        env[v.name] = val

    safe_map_flat(write, list(trace.args), list(args))
    safe_map_flat(write, list(trace.kwargs.values()), list(kwargs.values()))

    for symbol in trace.symbols:
        args = safe_map_flat(read, symbol.args)
        kwargs = {k: read(v) for k, v in symbol.kwargs.items()}
        prim_func = symbol_mapper(symbol)
        result = prim_func(*args, **kwargs)
        if not isinstance(result, Sequence):
            result = (result,)
        safe_map_flat(write, list(symbol.outputs), list(result))

    if with_env:
        return safe_map_flat(read, trace.outputs), env

    return safe_map_flat(read, trace.outputs)


def _identity_call_metafunc(*args, trace: Trace, **kwargs):
    with detached_trace():
        return eval_trace(trace, *args, **kwargs)


identity_call = prims.make_prim(Transforms.IdentityOp, "identity_call", _identity_call_metafunc)


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
ops_to_torch_ops_map[Transforms.IdentityOp] = _identity_call_pytorch


# Inline transform
# ----------------
# The inline transform is a special case of the identity transform.
# It is used to inline the transformation of a function in the trace without
# removing separate transform primitives from the trace.
inline_transforms_map: Dict[prims.Symbol, Callable] = dict()


def inline_symbol_mapper(symbol: prims.Symbol):
    if symbol.op in inline_transforms_map:
        return inline_transforms_map[symbol.op]

    return symbol_to_eval(symbol)


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


def binary_op_batching_rule(op: prims.Ops, axis_size: int, vals_in: BatchedValue):
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
    if broadcast_dimensions != ():
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


vmap_impls: Dict[prims.Symbol, Callable] = dict()

vmap_impls[prims.Ops.SIN] = sin_vmap
vmap_impls[prims.Ops.COS] = cos_vmap
vmap_impls[prims.Ops.MUL] = mul_vmap
vmap_impls[prims.Ops.ADD] = add_vmap
vmap_impls[prims.Ops.SUM] = sum_vmap
vmap_impls[prims.Ops.BROADCAST_IN_DIM] = broadcast_in_dim_vmap


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

    vmap_impl = vmap_impls.get(symbol.op)
    if vmap_impl is None:
        raise NotImplementedError(f"vmap for {symbol.op} is not implemented")

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
    trace = get_trace()
    name = trace.make_proxy_name()
    new_shape = tensor.shape[:batch_dim] + tensor.shape[batch_dim + 1 :]
    return TensorProxy(name=name, shape=new_shape, dtype=tensor.dtype, device=tensor.device)


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
    (common_device,) = common_device if len(common_device) == 1 else ("cpu",)

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


vmap_call = prims.make_prim(Transforms.VmapOp, "vmap_call", partial(_vmap_call_metafunc, detached=True))
inline_transforms_map[Transforms.VmapOp] = partial(_vmap_call_metafunc, detached=False)


# TODO: how should we handle out_dims here?
# although here we are calling vmap of identity, so we should know from the call to vmap
# This should be fine. If we have vmap(identity(func), out_dims=N) then this rule is first used
# to get the vmapped result of identity(func) in vmap_symbol_mapper, and out_dims is handled
# after that in the outer _vmap_call_metafunc.
def _identity_call_vmap(*batched_args, trace: Trace, **kwargs):
    half = len(batched_args) // 2
    args, in_dims = batched_args[:half], batched_args[half:]
    out_dims = 0  # Fixme
    outs, out_dims = _vmap_call_metafunc(args, in_dims, out_dims, trace=trace, detached=False, **kwargs)
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


jvp_impls: Dict[prims.Symbol, Callable] = dict()

jvp_impls[prims.Ops.SIN] = sin_jvp
jvp_impls[prims.Ops.MUL] = mul_jvp
jvp_impls[prims.Ops.ADD] = add_jvp


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
    jvp_impl = jvp_impls.get(symbol.op)
    if jvp_impl is None:
        raise NotImplementedError(f"JVP for {symbol.op} is not implemented")

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


jvp_call = prims.make_prim(Transforms.JvpOp, "jvp_call", partial(_jvp_call_metafunc, detached=True))
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
class VJPTriple:
    """A triple of primal, residuals, and pullback.

    Args:
        primal (Union[Proxy, Number]): Primal value, i.e., the value being differentiated.
        residuals (Tuple[Proxy, ...]): Residuals, i.e., the values that are
            saved for the pullback.
        pullback (Callable): Pullback function, i.e., the function that computes
            the vector-Jacobian products given the cotangents.

    Yields:
        Tuple[Variable, Tuple[Variable, ...], Callable]: Primal, residuals, and pullback.
    """

    primal: Union[Proxy, Number]
    residuals: Tuple[Proxy, ...]
    pullback: Callable

    def __iter__(self):
        yield self.primal
        yield self.residuals
        yield self.pullback


class NoPullback:
    """A dummy pullback function that raises an error when called."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Pullback called on a non-differentiable symbol")


no_pullback = NoPullback()


# This is a dummy class that is used to represent empty residuals
# instead of an empty tuple to make the pullback function more readable
class NoResidual:
    """A dummy residual value."""

    pass


no_residual = NoResidual()
no_residuals = (no_residual,)

# Mapping from symbols to VJP implementations
# The VJP implementation is a function that takes the primal values and returns
# a triple of primal result, residuals, and pullback function.
vjp_impls = {
    prims.Ops.SIN: lambda x: (prims.sin(x), (x,), lambda x, g: prims.mul(g, prims.cos(x))),
    prims.Ops.ASIN: lambda x: (prims.asin(x), (x,), lambda x, g: g / prims.sqrt(1 - x**2)),
    prims.Ops.ADD: lambda x, y: (prims.add(x, y), no_residuals, lambda _, g: (g, g)),
}


def restore_reduced_dims(x, reduced_dims, original_shape):
    """Restores the reduced dimensions of a tensor.

    Args:
        x (Variable): Tensor to be reshaped.
        reduced_dims (Tuple[int, ...]): Tuple of reduced dimensions.
        original_shape (Tuple[int, ...]): Original shape of the tensor.

    Returns:
        Variable: Tensor with the reduced dimensions restored.
    """
    import thunder.core.lang as tlang

    unsqueezed = tlang.unsqueeze(x, reduced_dims)
    return tlang.expand(unsqueezed, original_shape)


def sum_vjp(x, dims, output_dtype=None):
    """VJP of the sum operation.

    Args:
        x (Variable): Tensor to be summed.
        dims (Tuple[int, ...]): Dimensions to be summed.
        output_dtype (str, optional): Output data type. Defaults to None.

    Returns:
        VJPTriple: Primal, residuals, and pullback.
    """
    primal = prims.sum(x, dims, output_dtype=output_dtype)
    residuals = (
        x.shape,
        dims,
    )

    def pullback(x_shape, reduced_dims, g):
        # One return per positional argument of prims.sum
        return restore_reduced_dims(g, reduced_dims, x_shape), None

    return VJPTriple(primal, residuals, pullback)


vjp_impls[prims.Ops.SUM] = sum_vjp


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
            if isinstance(primals, Sequence):
                return safe_map(lambda x: VJPTriple(x, (), no_pullback), primals)
            return VJPTriple(primals, (), no_pullback)

        return partial(vjp_impl_const, symbol)

    # Normal case, we have a proxy tangent
    vjp_impl = vjp_impls.get(symbol.op)
    if vjp_impl is None:
        raise NotImplementedError(f"VJP for {symbol.op} is not implemented")

    def wrap_arg(arg):
        if isinstance(arg, Number):
            return VJPTriple(arg, tuple(), no_pullback)
        elif isinstance(arg, VJPTriple):
            return arg
        else:
            raise TypeError(f"Unexpected type {type(arg)}")

    def _vjp_impl(*args, **kwargs):
        args = tree_map(wrap_arg, args)

        # Expecting VJPTriple wrapping pairs of primals and residuals
        assert all(isinstance(arg, VJPTriple) for arg in tree_flatten(args)[0])

        primals = tree_map(lambda x: x.primal, args)
        out_primal, out_residuals, out_pullback = vjp_impl(*primals, **kwargs)

        assert not isinstance(out_primal, Sequence), "Not implemented for multiple outputs"
        return (VJPTriple(out_primal, out_residuals, out_pullback),)

    return _vjp_impl


def augmented_primal_metafunc(*primals, trace: Trace, **kwargs):
    primals_residuals_pullbacks = [VJPTriple(primal, tuple(), no_pullback) for primal in primals]
    result, env = eval_trace(
        trace,
        *primals_residuals_pullbacks,
        **kwargs,
        with_env=True,
        symbol_mapper=vjp_symbol_mapper,
    )
    return result, env


def backward_pass(forward_env, trace, init_cotangents):
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
        elif isinstance(v, Number):
            pass
        elif isinstance(v, Sequence) and all(isinstance(x, int) for x in v):
            # TODO: remove when we move dims to kwargs
            pass
        else:
            raise TypeError(f"Unexpected type {type(v)}")

    if isinstance(trace.outputs, Sequence):
        safe_map(write, tuple(trace.outputs), init_cotangents)
    else:
        write(trace.outputs, init_cotangents)

    for symbol in reversed(trace.symbols):
        cotangents = safe_map(read, symbol.outputs)
        # Having a single cotangent is a common case, so we flatten it
        # Otherwise, we will need to rewrite the pullback functions
        cotangents = tree_flatten(cotangents)[0]
        assert len(cotangents) == 1, "Not implemented for multiple outputs"
        residuals = forward_env[symbol.outputs[0].name].residuals
        pullback = forward_env[symbol.outputs[0].name].pullback
        result = pullback(*residuals, *cotangents)
        if not isinstance(result, Sequence):
            result = (result,)
        safe_map(write, symbol.args, result)

    return tree_map(read, tuple(trace.args))


def vjp_call_metafunc(primals, cotangents, trace: Trace, detached, **kwargs):
    # Assuming primals is flat

    if not isinstance(primals, Sequence):
        primals = (primals,)

    ctx = detached_trace() if detached else nullcontext()
    with ctx:
        primals_residuals_pullbacks = [VJPTriple(primal, tuple(), no_pullback) for primal in primals]
        result, env = eval_trace(
            trace, *primals_residuals_pullbacks, with_env=True, **kwargs, symbol_mapper=vjp_symbol_mapper
        )
        # Unwrap the VJPTriple
        result = tree_map(lambda x: x.primal, result)
        return result, backward_pass(env, trace, cotangents)


vjp_call = prims.make_prim(Transforms.VjpOp, "vjp_call", partial(vjp_call_metafunc, detached=True))
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
        cotangents = tree_map(lambda v: ones_like(v.proxy), trace.outputs)
        return vjp(func)(args, cotangents, **kwargs)

    return _value_and_grad
