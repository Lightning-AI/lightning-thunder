import math
from functools import reduce
from numbers import Number
from typing import Sequence

import thunder.core.dtypes as dtypes

# TODO: remove prims import
from thunder.core import prims, utils
from thunder.core.proxies import TensorProxy

# This file defines Thunder's "core" language.
#
# These operators are intended to be used when defining user-facing languages, like the torch or NumPy
# languages.
#
# This file depends on all other files in core.

__all__ = [
    # Data movement and transformation operations
    "maybe_convert_to_dtype",
    # Tensor creation operations
    "arange",
    "full",
    "full_like",
    "uniform",
    # Shape operations
    "compute_broadcast_shape",
    "expand",
    "maybe_broadcast",
    "reshape",
    "slice_in_dim",
    "squeeze",
    "transpose",
    "index_select",
    "unsqueeze",
    # Elemenwise unary operations
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bitwise_not",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "erfcinv",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "isfinite",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "ndtri",
    "reciprocal",
    "round",
    "rsqrt",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    # Elementwise binary operations
    "add",
    "atan2",
    "bitwise_and",
    "eq",
    "ge",
    "lt",
    "mul",
    "nextafter",
    "pow",
    "sub",
    "true_divide",
    # Elementwise ternary operations
    "where",
    # Language context
    "CoreLangCtx",
]

#
# Data movement and transformation operations
#


# TODO: may want to revise enforce_safe_casting to be more like NumPy's
def maybe_convert_to_dtype(a, dtype, *, enforce_safe_casting=False):
    """If a has the same dtype as the given dtype, returns a unmodified.

    Otherwise returns a converted to the given dtype.
    """

    utils.check(utils.is_dtype(dtype), lambda: f"Unknown dtype {dtype}!")

    if isinstance(a, Sequence):
        return tuple(maybe_convert_to_dtype(x, dtype) for x in a)
    if isinstance(a, TensorProxy):
        pass
    elif isinstance(a, Number):
        # NOTE: this allows conversions like (5, float32) -> 5., which is a little odd
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


#
# Tensor creation operations
#


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


# TODO: add error checking
def full(shape, fill_value, *, device, dtype=None):
    fill_value_dtype = dtypes.to_dtype(fill_value)
    dtype = dtype if dtype is not None else fill_value_dtype

    # Ensures the requested fill_value can be safely cast to the dtype
    # NOTE: this is always true if the dtype is inferred
    utils.check(
        utils.can_safe_cast_number_to(fill_value, fill_value_dtype),
        lambda: f"Can't safely cast fill_value of numbertype {fill_value_dtype} to dtype {dtype}!",
    )

    return prims.full(shape, fill_value, device=device, dtype=dtype)


def full_like(a, fill_value, *, device=None, dtype=None):
    if isinstance(a, Number):
        dtype = type(fill_value) if dtype is None else dtypes.dtype_to_numbertype(dtype)
        utils.check(
            device is None or device == "cpu",
            "Numbers can only be created on the CPU, but found a request for device={device}",
        )
        return dtype(fill_value)

    device = device if device is not None else a.device
    dtype = dtype if dtype is not None else a.true_dtype

    return full(a.shape, fill_value, device=device, dtype=dtype)


def uniform(shape, minval=0.0, maxval=1.0, *, dtype, device):
    return prims.uniform(shape, minval, maxval, dtype=dtype, device=device)


#
# Shape operations
#


# Expands a to the specified shape, possibly adding new dimensions and expanding
#   dimensions of length 1 to any length
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
    # TODO: this error message could probably be improved
    utils.check(a.numel() % numel == 0, "Can't infer length of dimension {neg_one_idx}!")
    remaining = a.numel() // numel
    shape = list(shape)
    shape[neg_one_idx] = remaining
    # NOTE: alternatively a new tuple could be constructed as follows:
    # shape = shape[:neg_one_idx] + (remaining,) + shape[neg_one_idx + 1:]
    return prims.reshape(a, shape)


# https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice_in_dim.html
# NOTE: this implementation derived from
#   https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/slicing.html#slice_in_dim
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


def squeeze(a, dims):
    dims = utils.canonicalize_dims(a.ndim, dims)
    result = prims.squeeze(a, dims)
    return result


def transpose(a, permutation):
    permutation = utils.canonicalize_dims(a.ndim, permutation)
    return prims.transpose(a, permutation)


def index_select(a, dim, index):
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.index_select(a, dim, index)


# Unsqueezes a, adding zero or more dimensions of length 1
# Added dimensions are specified by their position in the final tensor
# Based on https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.expand_dims.html
# NOTE: the dimensions do not have to be specified in any order
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


# TODO: add scalar support
# TODO: review hasattr pattern
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
def _elementwise_unary_helper(prim, type_promotion_kind, a, *, supported_dtypes=None):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, type_promotion_kind=type_promotion_kind)
    if supported_dtypes is not None:
        utils.check(
            computation_dtype in supported_dtypes,
            lambda: f"Unsupported dtype {computation_dtype}!",
        )

    a = maybe_convert_to_dtype(a, computation_dtype)

    result = prim(a)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


def abs(a):
    if dtypes.is_unsigned_dtype(dtypes.to_dtype(a)):
        return a

    return _elementwise_unary_helper(prims.abs, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT, a)


def acos(a):
    return _elementwise_unary_helper(prims.acos, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def acosh(a):
    return _elementwise_unary_helper(prims.acosh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def asin(a):
    return _elementwise_unary_helper(prims.asin, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def asinh(a):
    return _elementwise_unary_helper(prims.asinh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def atan(a):
    return _elementwise_unary_helper(prims.atan, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def atanh(a):
    return _elementwise_unary_helper(prims.atanh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def bitwise_not(a):
    return _elementwise_unary_helper(prims.bitwise_not, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


def ceil(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_helper(prims.ceil, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


def cos(a):
    return _elementwise_unary_helper(prims.cos, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def cosh(a):
    return _elementwise_unary_helper(prims.cosh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def erf(a):
    return _elementwise_unary_helper(prims.erf, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def erfc(a):
    return _elementwise_unary_helper(prims.erfc, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def erfcinv(a):
    return _elementwise_unary_helper(prims.erfcinv, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def erfinv(a):
    return _elementwise_unary_helper(prims.erfinv, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def exp(a):
    return _elementwise_unary_helper(prims.exp, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def exp2(a):
    return _elementwise_unary_helper(prims.exp2, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def expm1(a):
    return _elementwise_unary_helper(prims.expm1, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def floor(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_helper(prims.floor, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


def isfinite(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return full_like(a, True, dtype=dtypes.bool8)

    return _elementwise_unary_helper(prims.isfinite, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL, a)


def rsqrt(a):
    return _elementwise_unary_helper(prims.rsqrt, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def sign(a):
    return _elementwise_unary_helper(prims.sign, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


def sin(a):
    return _elementwise_unary_helper(prims.sin, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def sinh(a):
    return _elementwise_unary_helper(prims.sinh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def sqrt(a):
    return _elementwise_unary_helper(prims.sqrt, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def tan(a):
    return _elementwise_unary_helper(prims.tan, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def tanh(a):
    return _elementwise_unary_helper(prims.tanh, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def lgamma(a):
    return _elementwise_unary_helper(prims.lgamma, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def log(a):
    return _elementwise_unary_helper(prims.log, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def log10(a):
    return _elementwise_unary_helper(prims.log10, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def log1p(a):
    return _elementwise_unary_helper(prims.log1p, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def log2(a):
    return _elementwise_unary_helper(prims.log2, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def ndtri(a):
    return _elementwise_unary_helper(prims.ndtri, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def ndtri(a):
    return _elementwise_unary_helper(prims.ndtri, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def reciprocal(a):
    return _elementwise_unary_helper(prims.reciprocal, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a)


def round(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_helper(prims.round, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


def trunc(a):
    return _elementwise_unary_helper(prims.trunc, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a)


#
# Elementwise binary operations
#


# Helper function that implements broadcasting and type promotion for elementwise binary operations
# TODO: consider making type promotion kind an annotation on operations so it can be queried
#   programmatically
def _elementwise_binary_helper(prim, type_promotion_kind, a, b, *, supported_dtypes=None):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, b, type_promotion_kind=type_promotion_kind)

    a, b = maybe_broadcast(a, b)

    if supported_dtypes is not None:
        utils.check(
            computation_dtype in dtypes.resolve_dtypes(supported_dtypes),
            lambda: f"Unsupported dtype {computation_dtype}!",
        )

    a, b = maybe_convert_to_dtype(a, computation_dtype), maybe_convert_to_dtype(b, computation_dtype)

    result = prim(a, b)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


def add(a, b):
    return _elementwise_binary_helper(prims.add, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a, b)


def atan2(a, b):
    return _elementwise_binary_helper(prims.atan2, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a, b)


def bitwise_and(a, b):
    return _elementwise_binary_helper(
        prims.bitwise_and,
        utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        a,
        b,
        supported_dtypes=(dtypes.exact,),
    )


def eq(a, b):
    return _elementwise_binary_helper(prims.eq, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL, a, b)


def ge(a, b):
    return _elementwise_binary_helper(prims.ge, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL, a, b)


def lt(a, b):
    return _elementwise_binary_helper(prims.lt, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL, a, b)


def mul(a, b):
    return _elementwise_binary_helper(prims.mul, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a, b)


def nextafter(a, b):
    return _elementwise_binary_helper(prims.nextafter, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE, a, b)


def pow(a, b):
    return _elementwise_binary_helper(prims.pow, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG, a, b)


def sub(a, b):
    return _elementwise_binary_helper(prims.sub, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, a, b)


def true_divide(a, b):
    return _elementwise_binary_helper(prims.div, utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, a, b)


#
# Elementwise ternary operation
#


def where(pred, a, b):
    # Performs type promotion
    promotiontype, _ = utils.elementwise_type_promotion(
        a, b, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
    )
    a, b = maybe_convert_to_dtype(a, promotiontype), maybe_convert_to_dtype(b, promotiontype)

    # Broadcasts
    pred, a, b = maybe_broadcast(pred, a, b)

    return prims.where(pred, a, b)


class CoreLangCtx:
    def __init__(self):
        pass

    def add(self, a, b):
        return add(a, b)

    def sub(self, a, b):
        return sub(a, b)

    def true_divide(self, a, b):
        return true_divide(a, b)

    def intercept(self, op, *args, **kwargs):
        return None
