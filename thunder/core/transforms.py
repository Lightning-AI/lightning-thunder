from contextlib import nullcontext
from dataclasses import dataclass
from enum import auto, Enum
from functools import lru_cache, partial
from numbers import Number
from typing import Any, Callable, Dict, Sequence, Union

from thunder import make_trace, make_traced
from thunder.core import prims
from thunder.core.lang import full_like
from thunder.core.proxies import NumberProxy, TensorProxy
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.trace import detached_trace, get_trace, Trace, Variable
from thunder.core.utils import safe_map, safe_map_flat, safe_zip, unzip2, check
from thunder.executors.torch import ops_to_torch_ops_map


class Transforms(Enum):
    IdentityOp = auto()
    VmapOp = auto()
    JvpOp = auto()


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


def eval_trace(trace, *args, symbol_mapper=symbol_to_eval, **kwargs):
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
        check(v.name not in env, lambda: f"Found v={v} in env={env}!")
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
    pass


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


def vectorized_batcher(prim, axis_size, batched_args, batch_dims, **kwargs):
    assert all(batch_dims[0] == bd for bd in batch_dims[1:]), batch_dims
    return prim(*batched_args, **kwargs), batch_dims[0]


not_mapped = NotMapped()


def movedim(x, src: int, dst: int):
    perm = [i for i in range(x.ndim) if i != src]
    perm.insert(dst, src)
    return prims.transpose(x, perm)


def move_batch_dim(axis_size, src, dst, x):
    if src is not_mapped:
        if isinstance(x, (int, float)):
            return x
        target_shape = list(x.shape)
        target_shape.insert(dst, axis_size)
        return prims.broadcast_in_dim(x, target_shape, [dst])
    elif src == dst:
        return x
    else:
        return movedim(x, src, dst)


def binary_op_batching_rule(op, axis_size, vals_in, dims_in):
    (x, y), (x_bdim, y_bdim) = vals_in, dims_in
    if x_bdim != y_bdim:
        if x_bdim is not_mapped:
            x = move_batch_dim(axis_size, x_bdim, y_bdim, x)
            x_bdim = y_bdim
        else:
            y = move_batch_dim(axis_size, y_bdim, x_bdim, y)
    return op(x, y), x_bdim


def sin_vmap(axis_size, a, batched_dim):
    return vectorized_batcher(prims.sin, axis_size, (a,), (batched_dim,))


def cos_vmap(axis_size, a, batched_dim):
    return vectorized_batcher(prims.cos, axis_size, (a,), (batched_dim,))


def mul_vmap(axis_size, a, b, a_batched_dim, b_batched_dim):
    return binary_op_batching_rule(prims.mul, axis_size, (a, b), (a_batched_dim, b_batched_dim))


def add_vmap(axis_size, a, b, a_batched_dim, b_batched_dim):
    return binary_op_batching_rule(prims.add, axis_size, (a, b), (a_batched_dim, b_batched_dim))


vmap_impls: Dict[prims.Symbol, Callable] = dict()

vmap_impls[prims.Ops.SIN] = sin_vmap
vmap_impls[prims.Ops.COS] = cos_vmap
vmap_impls[prims.Ops.MUL] = mul_vmap
vmap_impls[prims.Ops.ADD] = add_vmap


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

    if not any(isinstance(arg, Variable) for arg in symbol.args):

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
        assert len(kwargs) == 0, "vmap for kwargs is not implemented"
        args = safe_map(wrap_arg, args)
        assert all(isinstance(arg, BatchedValue) for arg in args)
        _args, _bdims = unzip2(args)
        out_val, out_dim = vmap_impl(axis_size, *_args, *_bdims, **kwargs)
        if isinstance(out_val, Sequence):
            return safe_map(pair_to_batched_value, safe_zip(out_val, out_dim))
        return BatchedValue(out_val, out_dim)

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
def _vmap_call_metafunc(args, in_dims, out_dims, trace: Trace, detached: bool, **kwargs):
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

    (axis_size,) = {x.shape[ax] for x, ax in zip(args, in_dims) if ax is not not_mapped}
    in_dims = in_dims if isinstance(in_dims, Sequence) else (in_dims,)
    out_dims = out_dims if isinstance(out_dims, Sequence) else (out_dims,)

    ctx = detached_trace() if detached else nullcontext()
    with ctx:
        # We propagate the BatchValue through the trace, and then unwrap it at the end
        batched_args = safe_map(pair_to_batched_value, safe_zip(args, in_dims))
        result = eval_trace(trace, *batched_args, symbol_mapper=partial(vmap_symbol_mapper, axis_size=axis_size))
        # Unwrapping the BatchedValue's
        if isinstance(result, Sequence):
            assert all(isinstance(x, BatchedValue) for x in result)
            outs, bdims = unzip2(result)
            outs = safe_map(partial(move_batch_dim, axis_size), bdims, out_dims, outs)
            return outs, out_dims
        assert isinstance(result, BatchedValue)
        out = move_batch_dim(axis_size, result.batch_dim, out_dims[0], result.value)
        return out, out_dims[0]


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
    return _vmap_call_metafunc(args, in_dims, out_dims, trace=trace, detached=False, **kwargs)


vmap_impls[Transforms.IdentityOp] = _identity_call_vmap


def vmap(func, in_dims=0, out_dims=0):
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
            assert args_spec == in_dims_spec, "in_dims must have the same structure as args, kwargs"
        unbatched_args_flat = [remove_batch_dim(arg) if isinstance(arg, TensorProxy) else arg for arg in args_flat]
        trace = make_trace(func_flat)(*unbatched_args_flat)
        outs, bdims = vmap_call(args_flat, in_dims_flat, out_dims, trace=trace)
        return outs

    return wrapper


def vmap_eager(func, args, in_dims=0, out_dims=0, executor="torch"):
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
    vmap_trace = make_trace(inline(vmap(func, in_dims=in_dims, out_dims=out_dims)), executor=executor)(*args)
    vmap_traced = make_traced(partial(eval_trace, vmap_trace), executor=executor)
    return vmap_traced(*args)


# JVP transform
# -------------


def sin_jvp(a, ȧ):
    return prims.sin(a), prims.cos(a) * ȧ


def mul_jvp(a, b, ȧ, ḃ):
    return a * b, a * ḃ + b * ȧ


def add_jvp(a, b, ȧ, ḃ):
    return a + b, ȧ + ḃ


jvp_impls: Dict[prims.Symbol, Callable] = dict()

jvp_impls[prims.Ops.SIN] = sin_jvp
jvp_impls[prims.Ops.MUL] = mul_jvp
jvp_impls[prims.Ops.ADD] = add_jvp


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
    if not any(isinstance(arg, Variable) for arg in symbol.args):

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
        args = safe_map(wrap_arg, args)
        # Expecting JVPDuals wrapping pairs of primals and tangents
        assert all(isinstance(arg, JVPDual) for arg in args)
        primals, tangents = unzip2(args)
        out_primal, out_tangent = jvp_impl(*primals, *tangents, **kwargs)
        if isinstance(out_primal, Sequence):
            return safe_map(pair_to_jvp_dual, safe_zip(out_primal, out_tangent))
        return JVPDual(out_primal, out_tangent)

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


def _identity_call_jvp(*primals_and_tangents, trace: Trace, **kwargs):
    half = len(primals_and_tangents) // 2
    primals, tangents = primals_and_tangents[:half], primals_and_tangents[half:]
    return _jvp_call_metafunc(primals, tangents, trace, detached=False, **kwargs)


jvp_impls[Transforms.IdentityOp] = _identity_call_jvp


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
