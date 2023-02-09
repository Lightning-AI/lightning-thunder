from contextlib import nullcontext
from dataclasses import dataclass
from enum import auto, Enum
from functools import lru_cache, partial
from numbers import Number
from typing import Any, Callable, Dict, Sequence

from .. import make_trace, make_traced
from ..executors.torch import ops_to_torch_ops_map
from . import prims
from .lang import full_like
from .proxies import NumberProxy, Proxy, TensorProxy
from .trace import detached_trace, get_trace, Trace
from .utils import safe_map, safe_zip, unzip2, check


class Transforms(Enum):
    IdentityOp = auto()
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

    def read(x: Proxy):
        if isinstance(x, Proxy):
            return env[x]
        else:
            return x

    def write(v: Proxy, val: Any) -> None:
        check(v not in env, lambda: f"Found v={v} in env={env}!")
        env[v] = val

    safe_map(write, trace.args, args)
    safe_map(write, trace.kwargs.values(), kwargs.values())

    for symbol in trace.symbols:
        args = safe_map(read, symbol.args)
        kwargs = {k: read(v) for k, v in symbol.kwargs.items()}
        prim_func = symbol_mapper(symbol)
        result = prim_func(*args, **kwargs)
        if not isinstance(result, Sequence):
            result = (result,)
        safe_map(write, symbol.outputs, result)

    if not isinstance(trace.outputs, Sequence):
        return read(trace.outputs)
    return safe_map(read, trace.outputs)


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

    # If symbol.args doesn't have subclasses of Proxy, then we need to return a zero tangent
    # TODO: there may be a better way to detect constants in the trace
    if not any(isinstance(arg, Proxy) for arg in symbol.args):

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
