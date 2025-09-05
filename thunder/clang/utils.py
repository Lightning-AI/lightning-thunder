from numbers import Number
from collections.abc import Sequence
from collections.abc import Callable

from thunder.core import utils
import thunder.core.dtypes as dtypes
from thunder.core.symbol import Symbol

from thunder.core.proxies import (
    NumberProxy,
    TensorProxy,
)


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
