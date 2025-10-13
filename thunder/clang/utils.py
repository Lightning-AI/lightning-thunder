from numbers import Number
from collections.abc import Sequence
from collections.abc import Callable
from functools import reduce

from thunder.core import utils
import thunder.core.dtypes as dtypes
from thunder.core.symbol import Symbol

from thunder.core.proxies import (
    NumberProxy,
    TensorProxy,
)

TensorLike = TensorProxy


def create_maybe_convert_to_dtype_with_prim(conversion_prim: Symbol):
    assert isinstance(conversion_prim, Symbol)

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
        elif isinstance(a, (Number, NumberProxy)):
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

            return conversion_prim(a, dtype)

        return a

    return maybe_convert_to_dtype


# TODO Add supported dtypes
def _elementwise_unary_wrapper(
    a,
    *,
    prim,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    dtype_conversion_fn: Callable[[TensorProxy | NumberProxy, dtypes.dtype], TensorProxy | NumberProxy],
):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, type_promotion_kind=type_promotion_kind)

    a = dtype_conversion_fn(a, computation_dtype)
    result = prim(a)
    result = dtype_conversion_fn(result, result_dtype)

    return result


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


def expand_impl(a: TensorLike, *shape: int, broadcast_prim: Symbol) -> TensorLike:
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

    # NOTE: Converting shape_ to tuple makes it possible to apply CSE
    return broadcast_prim(a, tuple(shape_), tuple(range(offset, len(a.shape) + offset)))


# TODO: add scalar support
# TODO: review hasattr pattern
# NOTE: the tensor is not broadcasted if it is a CPU scalar tensor and treat_cpu_scalar_tensors_as_numbers=True
def maybe_broadcast_impl(*args, treat_cpu_scalar_tensors_as_numbers=True, expand_fn: Callable):
    """Returns tensors with the same shape, possibly broadcasting inputs to the result shape."""

    # Computes common shape
    common_shape = compute_broadcast_shape(*map(lambda t: t.shape if hasattr(t, "shape") else None, args))

    def _maybe_broadcast(x, shape):
        if treat_cpu_scalar_tensors_as_numbers and utils.is_cpu_scalar_tensor(x):
            return x
        if hasattr(x, "shape"):
            if not utils.same_shape(x.shape, common_shape):
                return expand_fn(x, common_shape)

        return x

    return tuple(_maybe_broadcast(x, common_shape) for x in args)
