import math
from functools import reduce
from numbers import Number
from typing import Sequence, Union, List, Optional

import thunder.core.dtypes as dtypes

# TODO: remove prims import
from thunder.core import utils
import thunder.core.prims as prims
from thunder.core.proxies import TensorProxy
from thunder.core.langctx import langctx
import thunder.core.devices as devices

# This file defines the operations in lightning.compile's "core" language.
#
# These operators are intended to be used when defining user-facing languages, like the torch or NumPy
# languages.

__all__ = []


def _module_extractor() -> None:
    pass


def clang_ctx(fn):
    module_name = clang_ctx.__module__
    module = utils.get_module(module_name)
    _fn = langctx(module)(fn)
    return _fn


#
# Data movement and transformation operations
#


# TODO Review revising enforce_safe_casting to be more like NumPy's
@clang_ctx
def maybe_convert_to_dtype(a, dtype, *, enforce_safe_casting=False):
    """If a has the same dtype as the given dtype, returns a unmodified.

    Otherwise returns a converted to the given dtype.
    """

    utils.check(utils.is_dtype(dtype), lambda: f"Unknown dtype {dtype}!")

    if isinstance(a, Sequence):
        return tuple(maybe_convert_to_dtype(x, dtype) for x in a)
    if isinstance(a, TensorProxy):
        # Translates numbertypes to dtypes
        if dtypes.is_numbertype(dtype):
            dtype = dtypes.numbertype_to_dtype(dtype)
    elif isinstance(a, Number):
        # NOTE This allows conversions like (5, float32) -> 5., which is a little odd
        dtype = utils.dtype_to_numbertype(dtype)
    else:
        raise ValueError(
            f"Trying to convert the type of the data of an unknown object {a} of {type(a)} that is neither a tensor, number, or sequence!"
        )

    if not utils.are_same_dtypes(a, dtype):
        if enforce_safe_casting:
            utils.check(
                utils.can_safe_cast_to(cast_from=utils.to_dtype(a), cast_to=dtype),
                lambda: f"Can't safe case from a={a} with dtype {utils.to_dtype(a)} to {dtype}!",
            )

        return prims.convert_element_type(a, dtype)

    return a


# TODO Consider maybe_device_put analogous to maybe_convert_to_dtype above
@clang_ctx
def device_put(a, device):
    if device is not None and a.device != device:
        return prims.device_put(a, device)
    return a


#
# Tensor creation operations
#
# TODO Is there a good helper/wrapper for _like functions?


# TODO Add type annotations
def arange(*, start, step, stop, device, dtype=None):
    # Validates inputs
    # Checks that start, step, and stop are finite
    # TODO: semantically an infinite step seems fine?
    utils.check(math.isfinite(start), lambda: f"start={start} was non-finite")
    utils.check(math.isfinite(step), lambda: f"step={step} was non-finite")
    utils.check(math.isfinite(stop), lambda: f"stop={stop} was non-finite")

    # Checks that start, step, and stop are not complex
    utils.check(not isinstance(start, complex), lambda: f"start={start} was complex")
    utils.check(not isinstance(step, complex), lambda: f"step={step} was complex")
    utils.check(not isinstance(stop, complex), lambda: f"stop={stop} was complex")

    # Checks that step makes progress
    utils.check(
        (step < 0 and stop < start) or (step > 0 and stop > start),
        lambda: f"step={step} must make progress from start={start} to stop={stop}",
    )

    # (Optionally) infers dtype
    # TODO: replace with default datatypes for integer and float
    if dtype is None:
        if all(tuple(isinstance(x, int) for x in (start, step, stop))):
            dtype = dtypes.int64
        else:
            dtype = dtypes.float32

    length = math.ceil((stop - start) / step)

    if utils.is_exact_dtype(dtype):
        return prims.iota(
            length,
            start=start,
            step=step,
            device=device,
            dtype=dtype,
        )

    index = prims.iota(
        length,
        start=0,
        step=1,
        dtype=dtypes.int64,
        device=device,
    )

    result = start + index * step
    result = maybe_convert_to_dtype(result, dtype)
    return result


@clang_ctx
def full(shape, fill_value, *, device, dtype=None):
    fill_value_dtype = dtypes.numbertype_to_dtype(dtypes.to_dtype(fill_value))
    dtype = dtype if dtype is not None else fill_value_dtype
    device = devices.to_device(device)

    return prims.full(shape, fill_value, device=device, dtype=dtype)


# TODO Handle a being a number
@clang_ctx
def full_like(a, fill_value, *, device=None, dtype=None):
    # if isinstance(a, Number):
    #     dtype = type(fill_value) if dtype is None else dtypes.dtype_to_numbertype(dtype)
    #     utils.check(
    #         device is None or device == "cpu",
    #         "Numbers can only be created on the CPU, but found a request for device={device}",
    #     )
    #     return dtype(fill_value)

    device = devices.to_device(device) if device is not None else a.device
    dtype = dtype if dtype is not None else a.true_dtype

    return full(a.shape, fill_value, device=device, dtype=dtype)


# TODO Restore device and dtype
@clang_ctx
def uniform(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: Union[str, devices.Device],
    dtype: dtypes.dtype,
) -> TensorProxy:
    device = devices.to_device(device)

    return prims.uniform(shape, minval, maxval, device=device, dtype=dtype)


# TODO Handle a being a number
@clang_ctx
def uniform_like(
    a: TensorProxy,
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: Optional[Union[str, devices.Device]] = None,
    dtype: Optional[dtypes.dtype] = None,
):
    device = devices.to_device(device) if device is not None else a.device
    dtype = dtype if dtype is not None else a.true_dtype

    return prims.uniform(a.shape, minval, maxval, device=device, dtype=dtype)


#
# Shape operations
#


# Expands a to the specified shape, possibly adding new dimensions and expanding
#   dimensions of length 1 to any length
@clang_ctx
def expand(a, *shape):
    shape = utils.extract_shape_from_varargs(shape)

    # TODO: improve this error message with error context
    utils.check(
        len(shape) >= len(a.shape),
        lambda: "expand: the requested shape has too few dimensions!",
    )

    offset = len(shape) - len(a.shape)
    shape_ = list(shape)
    for idx, x in enumerate(a.shape):
        offset_idx = idx + offset
        requested_length = shape[offset_idx]
        utils.check(
            requested_length == x or x == 1 or requested_length == -1,
            lambda: f"expand: attempting to expand a dimension of length {x}!",
        )

        shape_[offset_idx] = requested_length if requested_length != -1 else x

    # At this point shape must be valid
    # utils.check_valid_shape(shape_)

    return prims.broadcast_in_dim(a, shape_, tuple(range(offset, len(a.shape) + offset)))


# NOTE: shape may have a single -1 value, which is a marker that the length of that dimension
#   should be inferred
@clang_ctx
def reshape(a, shape):
    # Checks for -1 marker value
    numel = 1
    neg_one_idx = None
    for idx, l in enumerate(shape):
        if l >= 0:
            numel *= l
        else:
            utils.check(l == -1, "Found a negative dimension length {l} in shape={shape}!")
            utils.check(neg_one_idx is None, "Found two -1 markers in shape={shape}!")
            neg_one_idx = idx

    # Short-circuits if no shape inference is needed
    if neg_one_idx is None:
        return prims.reshape(a, shape)

    # Constructs the inferred shape, replacing -1 with the necessary length
    utils.check(a.numel % numel == 0, lambda: f"Trying to reshape, but can't infer how to reshape {a.shape} to {shape}")
    remaining = a.numel // numel
    shape = list(shape)
    shape[neg_one_idx] = remaining
    # NOTE: alternatively a new tuple could be constructed as follows:
    # shape = shape[:neg_one_idx] + (remaining,) + shape[neg_one_idx + 1:]
    return prims.reshape(a, shape)


# https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice_in_dim.html
# NOTE: this implementation derived from
#   https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/slicing.html#slice_in_dim
@clang_ctx
def slice_in_dim(a, start_index, limit_index, stride=1, dim=0):
    len_dim = a.shape[dim]
    start_index = utils.canonicalize_dim_idx(len_dim, start_index)
    limit_index = utils.canonicalize_dim_idx(len_dim, limit_index)

    # Handles the start idx being greater than the dimension length by returning
    #   a tensor with no elements
    if start_index >= len_dim:
        shape = list(a.shape)
        shape[dim] = 0
        return full(shape, 0, device=a.device, dtype=a.dtype)

    # Handles the limit idx being greater than the dimension length by clamping it
    if limit_index >= len_dim:
        limit_index = len_dim

    # Constructs args for the slice prim
    start_indices = [0] * a.ndim
    limit_indices = list(a.shape)
    strides = [1] * a.ndim

    start_indices[dim] = start_index
    limit_indices[dim] = limit_index
    strides[dim] = stride

    return prims.slice_prim(a, start_indices, limit_indices, strides)


@clang_ctx
def squeeze(a, dims):
    dims = utils.canonicalize_dims(a.ndim, dims)
    result = prims.squeeze(a, dims)
    return result


@clang_ctx
def transpose(a, permutation):
    permutation = utils.canonicalize_dims(a.ndim, permutation)
    return prims.transpose(a, permutation)


@clang_ctx
def take(a, indices, axis):
    axis = utils.canonicalize_dim(a.ndim, axis)
    return prims.take(a, indices, axis)


@clang_ctx
def take_along_axis(arr, indices, axis):
    axis = utils.canonicalize_dim(arr.ndim, axis)
    return prims.take_along_axis(arr, indices, axis)


# Unsqueezes a, adding zero or more dimensions of length 1
# Added dimensions are specified by their position in the final tensor
# Based on https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.expand_dims.html
# NOTE: the dimensions do not have to be specified in any order
@clang_ctx
def unsqueeze(a, dims):
    # Short-circuits if dims is empty
    if len(dims) == 0:
        return a

    final_rank = a.ndim + len(dims)
    # Canonicalizes and sorts dimensions
    dims = sorted(utils.canonicalize_dims(final_rank, dims))

    # Validates that (canonicalized) dimensions are unique
    utils.check_no_duplicates(dims)

    # Constructs expanded (unsqueezed) shape and determines final position of original dims
    shape = []
    broadcast_dims = []
    dims_idx = 0
    a_idx = 0
    for idx in range(final_rank):
        if dims_idx < len(dims) and dims[dims_idx] == idx:
            shape.append(1)
            dims_idx += 1
        else:
            shape.append(a.shape[a_idx])
            broadcast_dims.append(a_idx + dims_idx)
            a_idx += 1

    return prims.broadcast_in_dim(a, shape, broadcast_dims)


@clang_ctx
def cat(tensors: List[TensorProxy], dim: int):
    """Concatenates the given sequence of tensors in the given dimension."""
    return prims.cat(tensors, dim)


@clang_ctx
def stack(tensors: List[TensorProxy], dim: int):
    """Concatenates the given sequence of tensors in a new (the given) dimension."""
    shapes = tuple(t.shape for t in tensors)
    utils.check(shapes, lambda: f"list of tensors cannot be empty")
    for i, s in enumerate(shapes[1:], start=1):
        utils.check(
            s == shapes[0], lambda: f"tensors must be of the same shape, tensor at {i} is {s} instead of {shapes[0]}"
        )
    tensors_ = [t.unsqueeze(dim) for t in tensors]
    return prims.cat(tensors_, dim)


@clang_ctx
def compute_broadcast_shape(*_shapes):
    """Computes the common shape with the fewest dimensions that all input shapes can be broadcast to."""
    shapes = tuple(x for x in filter(lambda x: x is not None, _shapes))

    # Short-circuits if there are no inputs shapes
    #   This might happen in calls like add(2, 3)
    if len(shapes) == 0:
        return None

    common_shape = [
        1,
    ] * reduce(max, (len(shape) for shape in shapes))

    for shape in shapes:
        for idx in range(-1, -1 - len(shape), -1):
            if common_shape[idx] == 1:
                common_shape[idx] = shape[idx]

            utils.check(
                (shape[idx] == 1) or (common_shape[idx] == shape[idx]),
                lambda: f"Attempting to broadcast a dimension of length {shape[idx]}!",
            )

    return tuple(common_shape)


@clang_ctx
def matrix_transpose(a: TensorProxy) -> TensorProxy:
    """Transposes the last two dimensions of a tensor.

    This function is used to implement the `.mT` attribute.

    Args:
        a (TensorProxy): The tensor to transpose.

    Returns:
        TensorProxy: The transposed tensor.

    Examples:
        >>> a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> def func(x): return x.mT
        >>> traced_func = thunder.make_traced(func, executor="torch")
        >>> traced_func(a)
        tensor([[1, 4],
                [2, 5],
                [3, 6]])
    """
    dim0, dim1 = -2, -1
    dim0, dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))
    permutation = list(range(a.ndim))
    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
    return transpose(a, permutation)


# TODO: add scalar support
# TODO: review hasattr pattern
@clang_ctx
def maybe_broadcast(*args):
    """Returns tensors with the same shape, possibly broadcasting inputs to the result shape."""

    # Computes common shape
    common_shape = compute_broadcast_shape(*map(lambda t: t.shape if hasattr(t, "shape") else None, args))

    def _maybe_broadcast(x, shape):
        if hasattr(x, "shape"):
            if not utils.same_shape(x.shape, common_shape):
                return expand(x, common_shape)

        return x

    return tuple(_maybe_broadcast(x, common_shape) for x in args)


#
# Elementwise unary operations
#
# TODO Consider annotating these operators with kind and type promotion information


# TODO Add supported dtypes
def _elementwise_unary_wrapper(a, *, prim, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, type_promotion_kind=type_promotion_kind)

    a = maybe_convert_to_dtype(a, computation_dtype)

    result = prim(a)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


# TODO Return self for bool and uint datatypes?
@clang_ctx
def abs(a: Union[TensorProxy, Number]):
    # Short-circuits for unsigned types like bool and int8
    if dtypes.is_unsigned_dtype(dtypes.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a, prim=prims.abs, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    )


@clang_ctx
def acos(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.acos, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def acosh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.acosh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def asin(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.asin, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def asinh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.asinh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def atan(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.atan, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def atanh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.atanh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def bitwise_not(a):
    return _elementwise_unary_wrapper(
        a,
        prim=prims.bitwise_not,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE,
    )


@clang_ctx
def ceil(a):
    if dtypes.is_exact_dtype(dtypes.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a,
        prim=prims.ceil,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


@clang_ctx
def cos(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.cos, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def cosh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.cosh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def erf(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.erf, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def erfc(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.erfc, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def erfcinv(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.erfcinv, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def erfinv(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.erfinv, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def exp(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.exp, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def exp2(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.exp2, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def expm1(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.expm1, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def floor(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a,
        prim=prims.floor,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


@clang_ctx
def isfinite(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return full_like(a, True, dtype=dtypes.bool8)

    return _elementwise_unary_wrapper(
        a,
        prim=prims.isfinite,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    )


@clang_ctx
def lgamma(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.lgamma, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def log(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.log, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def log10(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.log10, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def log1p(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.log1p, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def log2(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.log2, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def ndtri(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.ndtri, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


# TODO Should this have PRESERVE as its type promotion kind?
@clang_ctx
def neg(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.neg, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@clang_ctx
def reciprocal(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.reciprocal, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def round(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a, prim=prims.round, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@clang_ctx
def rsqrt(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.rsqrt, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def sigmoid(a):
    # We manually promote a, then determine type of the constant 1.0 value
    computation_dtype, result_dtype = utils.elementwise_type_promotion(
        a, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )
    a = maybe_convert_to_dtype(a, computation_dtype)
    one = 1 + 0j if isinstance(computation_dtype, dtypes.complexfloating) else 1.0
    result = reciprocal(add(one, exp(-a)))
    result = maybe_convert_to_dtype(result, result_dtype)
    return result


# TODO Review type promotionkind for sign
def sign(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sign, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
    )


# TODO Add supported dtypes to exclude complex
def signbit(a):
    if dtypes.is_unsigned_dtype(dtypes.to_dtype(a)):
        return full_like(a, False, dtype=dtypes.bool8)

    return _elementwise_unary_wrapper(
        a, prim=prims.signbit, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clang_ctx
def silu(a):
    return a * sigmoid(a)


@clang_ctx
def sin(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sin, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def sinh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sinh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def sqrt(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sqrt, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def tan(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.tan, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


def tanh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.tanh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


def trunc(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a, prim=prims.trunc, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


#
# Elementwise binary operations
#
# TODO Consider annotating these operators with kind and type promotion information


# TODO Add supported dtypes
def _elementwise_binary_wrapper(a, b, *, prim, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, b, type_promotion_kind=type_promotion_kind)

    a, b = maybe_broadcast(a, b)
    a, b = maybe_convert_to_dtype(a, computation_dtype), maybe_convert_to_dtype(b, computation_dtype)

    result = prim(a, b)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


@clang_ctx
def add(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.add)


@clang_ctx
def atan2(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.atan2, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clang_ctx
def bitwise_and(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.bitwise_and, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@clang_ctx
def bitwise_xor(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.bitwise_xor, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


# NOTE Python's math.copysign has int to float type promotion, and PyTorch's copysign does, too
# NOTE copysign is not defined for complex types, and this must be explicitly checked for since the
#   following definition would be valid for complex numbers, too
@clang_ctx
def copysign(a, b):
    utils.check(
        not dtypes.is_complex_dtype(dtypes.to_dtype(a)) and not dtypes.is_complex_dtype(dtypes.to_dtype(b)),
        lambda: f"copysign is not defined for complex dtypes",
    )

    computation_dtype, result_dtype = utils.elementwise_type_promotion(
        a, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )

    compute_a = maybe_convert_to_dtype(a, computation_dtype)

    result = where(signbit(b), -abs(compute_a), abs(compute_a))
    return maybe_convert_to_dtype(result, result_dtype)


@clang_ctx
def eq(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.eq, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


# NOTE Floor division in Python is defined to complement its modulus operator, s.t.

#   let a and b be numbers
#   q = a // b
#   r = a % b
#   b * q + r = a

#   This is NOT equivalent to floor(a/b). For example:

#   import math
#   a = .25
#   b = .001

#   # Compares flooring true division vs. floor division
#   math.floor(a / b)  # 250
#   a // b  # 249.

#   # Tests the invariant
#   q = a // b
#   r = a % b
#   b * q + r   # .25 == a

# See CPython's implementation here:
# https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636


# NOTE This is distinct from true_divide, which also wraps prims.div, because it doesn't promote
#   integers to floating point values
def _c_div(a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> Union[TensorProxy, Number]:
    return _elementwise_binary_wrapper(
        a, b, prim=prims.div, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


def _floor_divide_integer(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, computation_dtype
) -> Union[TensorProxy, Number]:
    # Converts truncation division to floor division

    offset = logical_and(signbit(a) != signbit(b), fmod(a, b) != 0)
    return _c_div(a, b) - offset


def _floor_divide_float(a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> Union[TensorProxy, Number]:
    mod = fmod(a, b)
    div = (a - mod) / b

    # Ensures that the remainder has the same sign as denominator
    different_signed_inputs = (a < 0) ^ (b < 0)
    non_zero_remainder = mod != 0
    mask = non_zero_remainder & different_signed_inputs
    div = where(mask, div - 1, div)

    # Maps quotient to nearest integer value
    floor_div = floor(div)
    mask = (div - floor_div) > 0.5
    floor_div = where(mask, floor_div + 1, floor_div)

    true_div = a / b

    # Copies signbit where floor division is zero
    floor_div = where(div != 0, floor_div, copysign(0, true_div))

    # Follows true divide behavior when the denominator is zero
    return where(b == 0, true_div, floor_div)


# Dispatches floor division to integer or floating point specializations
@clang_ctx
def floor_divide(a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> Union[TensorProxy, Number]:
    computation_dtype, _ = utils.elementwise_type_promotion(
        a, b, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )

    utils.check(not dtypes.is_complex_dtype(computation_dtype), lambda: f"Complex floor division is not supported")

    if dtypes.is_float_dtype(computation_dtype):
        return _floor_divide_float(a, b)

    # NOTE At this point the datatypes are neither complex nor floating point, so they are exact types
    return _floor_divide_integer(a, b, computation_dtype=computation_dtype)


@clang_ctx
def fmod(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.fmod)


# TODO Review mod vs fmod
def mod(a, b):
    return fmod(a, b)


@clang_ctx
def ge(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.ge, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clang_ctx
def gt(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.gt, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clang_ctx
def logical_and(a, b):
    if not utils.is_boolean_dtype(dtypes.to_dtype(a)):
        a = a != 0
    if not utils.is_boolean_dtype(dtypes.to_dtype(b)):
        b = b != 0

    return a & b


@clang_ctx
def lt(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.lt, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clang_ctx
def mul(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.mul)


@clang_ctx
def ne(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.ne, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clang_ctx
def nextafter(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.nextafter, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
    )


@clang_ctx
def pow(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.pow, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    )


@clang_ctx
def remainder(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.remainder)


@clang_ctx
def sub(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.sub)


@clang_ctx
def true_divide(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.div, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


#
# Conditional operators
#


@clang_ctx
def where(pred, a, b):
    # Performs type promotion
    promotiontype, _ = utils.elementwise_type_promotion(
        a, b, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
    )

    a, b = maybe_convert_to_dtype(a, promotiontype), maybe_convert_to_dtype(b, promotiontype)

    # Broadcasts
    pred, a, b = maybe_broadcast(pred, a, b)

    return prims.where(pred, a, b)
