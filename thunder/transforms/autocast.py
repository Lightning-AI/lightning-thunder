from contextlib import contextmanager
from functools import partial, wraps
from collections.abc import Callable
from collections.abc import Sequence

from thunder.core import dtypes, prims, devices
from thunder.core.pytree import tree_map, tree_flatten
from thunder.core.proxies import TensorProxy
from thunder.core.symbol import BoundSymbolInterface, Symbol
from thunder.core.proxies import TensorProxy
from thunder.core.transforms import construct_trace, eval_trace
from thunder.clang import (
    maybe_convert_to_dtype,
)
import thunder.torch as ltorch


autocast_impls: dict[prims.PrimIDs, Callable] = {}


# NOTE: Rules which are registered ltorch symbols should match the type signature
#       of those symbols as we use this rule for translating from `torch` -> `thunder.torch`
#       if autocast is enabled while jitting. See also `NOTE: torch.autocast support`.
def register_autocast_rule(op):
    def decorator(func):
        autocast_impls[op] = func
        return func

    return decorator


_allowed_downcast_types = {dtypes.float16, dtypes.bfloat16}
_allowed_downcast_types_str_to_dtype_map = {str(dtype): dtype for dtype in _allowed_downcast_types}


def _check_valid_autocast_dtype(dtype):
    if dtype not in _allowed_downcast_types:
        raise ValueError(
            f"autocast: `dtype` is expected to be either `thunder.float16` or `thunder.bfloat16`, but {dtype}"
        )


def _get_downcast_dtype_from_str(str_dtype):
    return _allowed_downcast_types_str_to_dtype_map[str_dtype]


def maybe_downcast_to(dtype, args):
    allowed_downcast_types = (dtypes.float16, dtypes.bfloat16, dtypes.float32)

    def map_fn(a):
        if isinstance(a, TensorProxy) and a.dtype in allowed_downcast_types:
            return maybe_convert_to_dtype(a, dtype)
        return a

    return tree_map(map_fn, args)


@register_autocast_rule("torch.matmul")
def autocast_torch_matmul_rule(a, b, dtype):
    """Autocast rule for matmul"""
    return ltorch.matmul(*(maybe_downcast_to(dtype, (a, b))))


@register_autocast_rule(prims.PrimIDs.MATMUL)
def autocast_matmul_rule(a, b, dtype):
    """Autocast rule for matmul"""
    return prims.matmul(*(maybe_downcast_to(dtype, (a, b))))


def _linear_autocast_impl(a, w, bias, dtype):
    if bias is None:
        # Don't pass `bias` to maybe_downcast_to.
        downcast_args = maybe_downcast_to(dtype, (a, w)) + (bias,)
    else:
        downcast_args = maybe_downcast_to(dtype, (a, w, bias))

    return prims.linear(*downcast_args)


@register_autocast_rule("torch.nn.functional.linear")
def autocast_ltorch_linear_rule(a, w, bias=None, *, dtype):
    return _linear_autocast_impl(a, w, bias, dtype)


@register_autocast_rule(prims.PrimIDs.LINEAR)
def autocast_linear_rule(a, w, bias, dtype):
    return _linear_autocast_impl(a, w, bias, dtype)


def _convolution_autocast_impl(a, w, bias, *other_args, dtype):
    if bias is None:
        # Don't pass `bias` to maybe_downcast_to.
        downcast_args = maybe_downcast_to(dtype, (a, w)) + (bias,)
    else:
        downcast_args = maybe_downcast_to(dtype, (a, w, bias))

    return prims.convolution(*downcast_args, *other_args)


@register_autocast_rule("torch.nn.functional.conv1d")
def autocast_ltorch_conv1d_rule(
    a: TensorProxy,
    /,
    weight: TensorProxy,
    bias: TensorProxy | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int = 1,
    groups: int = 1,
    *,
    dtype,
) -> TensorProxy:
    from thunder.torch import _conv_helper

    return _conv_helper(
        1,
        a,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        conv_function=partial(_convolution_autocast_impl, dtype=dtype),
    )


@register_autocast_rule("torch.nn.functional.conv2d")
def autocast_ltorch_conv2d_rule(
    a: TensorProxy,
    /,
    weight: TensorProxy,
    bias: TensorProxy | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int = 1,
    groups: int = 1,
    *,
    dtype,
) -> TensorProxy:
    from thunder.torch import _conv_helper

    return _conv_helper(
        2,
        a,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        conv_function=partial(_convolution_autocast_impl, dtype=dtype),
    )


@register_autocast_rule("torch.nn.functional.conv3d")
def autocast_ltorch_conv3d_rule(
    a: TensorProxy,
    /,
    weight: TensorProxy,
    bias: TensorProxy | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int = 1,
    groups: int = 1,
    *,
    dtype,
) -> TensorProxy:
    from thunder.torch import _conv_helper

    return _conv_helper(
        3,
        a,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        conv_function=partial(_convolution_autocast_impl, dtype=dtype),
    )


@register_autocast_rule("torch.nn.functional.scaled_dot_product_attention")
def autocast_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    dtype,
    scale=None,
):
    from thunder.torch import scaled_dot_product_attention

    q, k, v = maybe_downcast_to(dtype, (query, key, value))
    return scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale)


def _maybe_get_autocast_rule_for_symbol(sym: Symbol):
    return autocast_impls.get(sym.id)


def autocast_symbol_mapper(bound_symbol: BoundSymbolInterface, dtype: dtypes.dtype):
    """Return the callable implementing the autocast rule for the symbol.

    Args:
        bound_symbol: Mapped to its autocast rule.

    Returns:
        Callable: The callable implementing the autocast rule for the symbol.
    """
    autocast_impl: Callable | None = _maybe_get_autocast_rule_for_symbol(bound_symbol.sym)
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
    _check_valid_autocast_dtype(dtype)

    @wraps(func)
    def wrapper(*args, **kwargs):
        trace = construct_trace()(func, *args, **kwargs)
        return eval_trace(trace, *args, **kwargs, symbol_mapper=partial(autocast_symbol_mapper, dtype=dtype))

    return wrapper


# State to enable and disable autocast (while interpreter is generating the computation trace).
_autocast_enabled = True


def is_autocast_enabled():
    return _autocast_enabled


# NOTE: Disable Autocast
# Once the autocast rule has been applied, we disable autocast as the autocast_rule usually converts the inputs
# and calls the same symbol.
# Without disabling autocast we will have infinite recursion.
@contextmanager
def disable_autocast():
    global _autocast_enabled
    default = _autocast_enabled
    _autocast_enabled = False
    try:
        yield
    finally:
        _autocast_enabled = default


def maybe_apply_autocast(sym):
    # NOTE: torch.autocast support
    # In PyTorch eager, enabling autocast allows mixed inputs to operator like `linear`
    # which expect dtypes to be same. This works in PyTorch eager, as dispatcher applies the
    # conversion first and then passes the converted input to the operator.
    # To mimick similar behavior, here we replace the `sym` for all operators which have
    # autocast rule, to apply the autocast conversion rule if autocast was enabled.

    from thunder.core.compile_data import get_compile_data

    # If user has actually enabled autocast in their code
    # AND we are not under `disable_autocast`
    cd = get_compile_data()
    if cd is None:
        # This would be `None` with the deprecated `thunder.compile` path
        user_enabled_autocast = False
    else:
        user_enabled_autocast = not cd.autocast_stack.is_empty()

    if (
        user_enabled_autocast
        and is_autocast_enabled()
        # symbol has autocast impl.
        and (autocast_impl := _maybe_get_autocast_rule_for_symbol(sym)) is not None
    ):

        @wraps(sym)
        def wrapper(*args, **kwargs):
            # See NOTE: Disable Autocast
            with disable_autocast():
                # Helper to determine which device should be queried for active
                # autocast context.
                def is_cpu_tensor(p):
                    if isinstance(p, TensorProxy):
                        return p.device.devicetype is devices.DeviceType.CPU
                    return False

                any_cpu = any(tree_flatten(tree_map(is_cpu_tensor, (args, kwargs)))[0])
                thunder_autocast_dtype = cd.autocast_stack.get_dtype_for_device_if_enabled("cpu" if any_cpu else "cuda")
                if thunder_autocast_dtype is None:  # We take this path if autocast was enabled for another device.
                    return sym(*args, **kwargs)
                return partial(autocast_impl, dtype=dtypes.to_dtype(thunder_autocast_dtype))(*args, **kwargs)

        return wrapper

    return None
