from enum import Enum
from functools import reduce, wraps
from numbers import Number
from typing import Callable, Sequence, Type

import thunder.core.dtypes as dtypes
import thunder.core.trace as trace
from thunder.core.proxies import NumberProxy, TensorProxy

# This file defines utilities that can be used when defining primitive operations.

# This file depends on proxies.py and the dtypes submodule.

__all__ = [
    # Error checking helpers
    "check",
    # dtype and Python type-related functions
    "to_dtype",
    "is_boolean_dtype",
    "is_unsigned_dtype",
    "is_signedinteger_dtype",
    "is_integer_dtype",
    "is_exact_dtype",
    "is_low_precision_dtype",
    "is_float_dtype",
    "is_complex_dtype",
    "is_inexact_dtype",
    "is_numbertype",
    "is_dtype",
    "is_weak_dtype",
    "corresponding_real_dtype",
    "corresponding_complex_dtype",
    "dtype_to_numbertype",
    "are_same_dtypes",
    "corresponding_real_dtype",
    "corresponding_complex_dtype",
    "can_safe_cast_to",
    "check_same_dtype",
    "get_numberlike_type",
    "get_numberlike_value",
    "ELEMENTWISE_TYPE_PROMOTION_KIND",
    "get_computation_dtype",
    "elementwise_type_promotion",
    # Shape-related functions
    "extract_shape_from_varargs",
    "is_numbertensor",
    "same_shape",
    "check_same_shape",
    "canonicalize_dim",
    "canonicalize_dims",
    "check_valid_length",
    "check_valid_permutation",
    "check_valid_shape",
    "check_no_duplicates",
    # Device-related functions
    "check_same_device",
    # Context-related functions and decorators
    "langctx",
]

#
# Error checking helpers
#


# TODO: maybe put check in a dependency-free base_utils that's also imported here (so it can be used by proxies.py)
def check(cond: bool, s: Callable[[], str], exception_type: Type[Exception] = RuntimeError) -> None:
    """Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.

    s is a callable producing a string to avoid string construction if the error check is passed.
    """
    if not cond:
        raise exception_type(s())


#
# dtype-related functions
#

to_dtype = dtypes.to_dtype
is_boolean_dtype = dtypes.is_boolean_dtype
is_unsigned_dtype = dtypes.is_unsigned_dtype
is_signedinteger_dtype = dtypes.is_signedinteger_dtype
is_integer_dtype = dtypes.is_integer_dtype
is_exact_dtype = dtypes.is_exact_dtype
is_low_precision_dtype = dtypes.is_low_precision_dtype
is_float_dtype = dtypes.is_float_dtype
is_complex_dtype = dtypes.is_complex_dtype
is_inexact_dtype = dtypes.is_inexact_dtype
is_numbertype = dtypes.is_numbertype
is_dtype = dtypes.is_dtype
is_weak_dtype = dtypes.is_weak_dtype
dtype_to_numbertype = dtypes.dtype_to_numbertype
are_same_dtypes = dtypes.are_same_dtypes
corresponding_real_dtype = dtypes.corresponding_real_dtype
corresponding_complex_dtype = dtypes.corresponding_complex_dtype


def higher_dtype(a, b):
    for fn in (
        is_complex_dtype,
        is_float_dtype,
        is_signedinteger_dtype,
        # NOTE: checking that x is not boolean here signifies that x is a non-boolean unsigned integer dtype
        #   (checking for its being an unsigned integer dtype directly wouldn't work, because bools are an
        #    unsigned integer dtype!)
        lambda x: not is_boolean_dtype(x),
        is_boolean_dtype,
    ):
        if fn(a):
            return a
        if fn(b):
            return b

    raise ValueError(f"Trying to determine the higher dtype of unknown inputs {a} and {b}!")


def can_safe_cast_to(*, cast_to, cast_from) -> bool:
    return higher_dtype(cast_to, cast_from) == cast_to


def can_safe_cast_number_to(num, dtype):
    if is_complex_dtype(dtype):
        return True
    if is_float_dtype(dtype) and not isinstance(num, complex):
        return True
    if is_integer_dtype(dtype) and not isinstance(num, float) and not isinstance(num, complex):
        return True

    return False


def get_numberlike_type(x):
    if isinstance(x, NumberProxy):
        return x.python_type

    if isinstance(x, Number):
        return type(x)

    raise ValueError(f"Trying to extract the number type of unknown object {x} with type {type(x)}!")


def get_numberlike_value(x):
    if isinstance(x, NumberProxy):
        return x.value

    if isinstance(x, Number):
        return x

    raise ValueError(f"Trying to acquire the value of unknown object {x} with type {type(x)}!")


def check_same_dtype(*args):
    """Accepts multiple dtypes, TensorProxies, and numbers.

    Checks that ...
        - all numbers have the same numbertype (bool, int, float or complex)
        - all non-numbers convert to the same (strong) dtype
        - if both a numbertype and dtype are identified,
            that the dtype converts to the same numbertype as the numbertype

    Returns the common dtype and numbertype, if any.

    Raises a RuntimeError if the check fails.
    """

    if len(args) == 0:
        return None, None

    numbertype = None
    dtype = None
    for a in args:
        if isinstance(a, Number):
            typ = to_dtype(a)
            if numbertype is None:
                numbertype = typ

            check(
                typ is numbertype,
                lambda: f"Expected numbertype {numbertype} but found {typ}!",
            )
        else:
            typ = to_dtype(a, true_dtype=True)
            if dtype is None:
                dtype = typ

            check(
                are_same_dtypes(dtype, typ),
                lambda: f"Expected dtype {dtype} but found {typ}!",
            )

            # Biases towards strong dtypes
            if is_weak_dtype(dtype):
                dtype = typ

    # Reconciles the numbertype and dtype
    if numbertype is not None and dtype is not None:
        expected_numbertype = dtype_to_numbertype(dtype)
        check(
            numbertype is expected_numbertype,
            lambda: (
                f"Expected the numbertype {expected_numbertype}, corresponding to the dtype {dtype}, but found the numbertype {numbertype}!"
            ),
        )

    return numbertype, dtype


b8_, b8 = dtypes.bool8_, dtypes.bool8
u8_, u8 = dtypes.uint8_, dtypes.uint8
i8_, i8 = dtypes.int8_, dtypes.int8
i16_, i16 = dtypes.int16_, dtypes.int16
i32_, i32 = dtypes.int32_, dtypes.int32
i64_, i64 = dtypes.int64_, dtypes.int64

_exact_dtype_to_number_map = {
    bool: 0,
    b8_: 1,
    b8: 2,
    int: 3,
    u8_: 4,
    u8: 5,
    i8_: 6,
    i16_: 7,
    i32_: 8,
    i64_: 9,
    i8: 10,
    i16: 11,
    i32: 12,
    i64: 13,
}

# fmt: off
# Exact type lattice
#    b8_ -> b8 \
# b /-> i -> i8_ -> i16_ -> i32_ -> i64_ -> i8 -> i16 -> i32 -> i64
#                `-> u8_ -> u8 ----------------------------^
# TODO REVIEW: it's a little odd that u8_ + i64_ -> i16
_elementwise_exact_promotion_table = [
    #    b     b8_     b8      i    u8_    u8    i8_  i16_  i32_  i64_  i8   i16  i32  i64
    [ bool,    b8_,   b8,    int,  u8_,   u8,   i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # b
    [  b8_,    b8_,   b8,    i8_,  u8_,   u8,   i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # b8_
    [   b8,     b8,   b8,    i8_,  u8_,   u8,   i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # b8
    [  int,    i8_,  i8_,    int,  u8_,   u8,   i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # i
    [  u8_,    u8_,  u8_,    u8_,  u8_,   u8,   i16,  i16,  i16,  i16, i16, i16, i32, i64], # u8_
    [   u8,     u8,   u8,     u8,   u8,   u8,   i16,  i16,  i16,  i16, i16, i16, i32, i64], # u8
    [  i8_,    i8_,  i8_,    i8_,  i16,   i16,  i8_, i16_, i32_, i64_,  i8, i16, i32, i64], # i8_
    [ i16_,   i16_, i16_,   i16_,  i16,   i16, i16_, i16_, i32_, i64_,  i8, i16, i32, i64], # i16_
    [ i32_,   i32_, i32_,   i32_,  i16,   i16, i32_, i32_, i32_, i64_,  i8, i16, i32, i64], # i32_
    [ i64_,   i64_, i64_,   i64_,  i16,   i16, i64_, i64_, i64_, i64_,  i8, i16, i32, i64], # i64_
    [   i8,     i8,   i8,     i8,  i16,   i16,   i8,   i8,   i8,   i8,  i8, i16, i32, i64], # i8
    [  i16,    i16,  i16,    i16,  i16,   i16,  i16,  i16,  i16,  i16, i16, i16, i32, i64], # i16
    [  i32,    i32,  i32,    i32,  i32,   i32,  i32,  i32,  i32,  i32, i32, i32, i32, i64], # i32
    [  i64,    i64,  i64,    i64,  i64,   i64,  i64,  i64,  i64,  i64, i64, i64, i64, i64], # i64
]



bf_,     bf =  dtypes.bfloat16_,   dtypes.bfloat16
f16_,   f16 =  dtypes.float16_,    dtypes.float16
f32_,   f32 =  dtypes.float32_,    dtypes.float32
f64_,   f64 =  dtypes.float64_,    dtypes.float64
c32_,   c32 =  dtypes.complex32_,  dtypes.complex32
c64_,   c64 =  dtypes.complex64_,  dtypes.complex64
c128_, c128 =  dtypes.complex128_, dtypes.complex128

_inexact_dtype_to_number_map = {
    float   : 0,
    bf_     : 1,
    f16_    : 2,
    f32_    : 3,
    f64_    : 4,
    bf      : 5,
    f16     : 6,
    f32     : 7,
    f64     : 8,
    complex : 9,
    c32_    : 10,
    c64_    : 11,
    c128_   : 12,
    c32     : 13,
    c64     : 14,
    c128    : 15,
}

# Inexact type lattice
#    c* -> c32* -> c64* -> c128* -> c32 ----> c64 ----> c128
#   /    /        /       /       /          /        /
#  /    /        /       /   ,-> float16 -> fp32 -> fp64
# f -> fp16* -> fp32* -> fp64* -> bfloat16 --^
#  `-> bfloat16* -^
_elementwise_inexact_promotion_table = [
    #       f    bf_   f16_   f32_    f64_   bf   f16   f32   f64  complex   c32_   c64_  c128_   c32   c64  c128
    [   float,   bf_,  f16_,  f32_,  f64_,   bf,  f16,  f32,  f64, complex,  c32_,  c64_, c128_,  c32,  c64, c128], # f
    [     bf_,   bf_,  f32_,  f32_,  f64_,   bf,  f16,  f32,  f64,    c64_,  c32_,  c64_, c128_,  c32,  c64, c128], # bf_
    [    f16_,  f32_,  f16_,  f32_,  f64_,   bf,  f16,  f32,  f64,    c32_,  c32_,  c64_, c128_,  c32,  c64, c128], # f16_
    [    f32_,  f32_,  f32_,  f32_,  f64_,   bf,  f16,  f32,  f64,    c64_,  c32_,  c64_, c128_,  c32,  c64, c128], # f32_
    [    f64_,  f64_,  f64_,  f64_,  f64_,   bf,  f16,  f32,  f64,   c128_,  c32_,  c64_, c128_,  c32,  c64, c128], # f64_
    [      bf,    bf,    bf,    bf,    bf,   bf,  f32,  f32,  f64,     c64,   c64,   c64,   c64,  c64,  c64, c128], # bf
    [     f16,   f16,   f16,   f16,   f16,  f32,  f16,  f32,  f64,     c32,   c32,   c32,   c32,  c32,  c64, c128], # f16
    [     f32,   f32,   f32,   f32,   f32,  f32,  f32,  f32,  f64,     c64,   c64,   c64,   c64,  c64,  c64, c128], # f32
    [     f64,   f64,   f64,   f64,   f64,  f64,  f64,  f64,  f64,    c128,  c128,  c128,  c128, c128, c128, c128], # f64
    [ complex,  c64_,  c32_,  c64_, c128_,  c64,  c32,  c64, c128, complex,  c32_,  c64_, c128_,  c32,  c64, c128], # complex
    [    c32_,  c64_,  c32_,  c64_, c128_,  c64,  c32,  c64, c128,    c32_,  c32_,  c64_, c128_,  c32,  c64, c128], # c32_
    [    c64_,  c64_,  c64_,  c64_, c128_,  c64,  c32,  c64, c128,    c64_,  c64_,  c64_, c128_,  c32,  c64, c128], # c64_
    [   c128_, c128_, c128_, c128_, c128_,  c64,  c32,  c64, c128,   c128_, c128_, c128_, c128_,  c32,  c64, c128], # c128_
    [     c32,   c32,   c32,   c32,   c32,  c64,  c32,  c64, c128,     c32,   c32,   c32,   c32,  c32,  c64, c128], # c32
    [     c64,   c64,   c64,   c64,   c64,  c64,  c64,  c64, c128,     c64,   c64,   c64,   c64,  c64,  c64, c128], # c64
    [    c128,  c128,  c128,  c128,  c128, c128, c128, c128, c128,    c128,  c128,  c128,  c128, c128, c128, c128], # c128
]



def _elementwise_type_promotion(a, b):
    # Inexact x exact and exact x inexact cases
    # Inexact dtypes take preference over exact dtypes
    if is_inexact_dtype(a) and is_exact_dtype(b):
        return a
    if is_exact_dtype(a) and is_inexact_dtype(b):
        return b

    # Exact x Exact case
    # b -> i -> i8* -> i16* -> i32* -> i64* -> i8 -> i16 -> i32 -> i64
    #       `-> u8* -> u8 ----------------------------^
    if is_exact_dtype(a):
        a_idx, b_idx = _exact_dtype_to_number_map[a], _exact_dtype_to_number_map[b]
        return _elementwise_exact_promotion_table[a_idx][b_idx]

    # Inexact x Inexact case
    # c* -> c32* -> c64* -> c128* -> c32 ----> c64 ----> c128
    #       /        /       /       /          /        /
    #      /        /       /   ,-> float16 -> fp32 -> fp64
    # fp16* ---> fp32* -> fp64* -> bfloat16 --^
    # bfloat16* -^
    a_idx, b_idx = _inexact_dtype_to_number_map[a], _inexact_dtype_to_number_map[b]
    return _elementwise_inexact_promotion_table[a_idx][b_idx]


# Maps dtypes to their computation types for elementwise operations
_computation_dtype_map = {
    dtypes.float16_   : dtypes.float32_,
    dtypes.float16    : dtypes.float32,
    dtypes.bfloat16_  : dtypes.float32_,
    dtypes.bfloat16   : dtypes.float32,
    dtypes.complex32_ : dtypes.complex64_,
    dtypes.complex32  : dtypes.complex64,
}
# fmt: on


def get_computation_dtype(typ):
    return _computation_dtype_map.get(typ, typ)


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    PRESERVE = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)


# TODO: allow dtypes as arguments, too
def elementwise_type_promotion(*args, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND):
    """Computes the computation and result types for elementwise type promotion on the given arguments and with the
    given elementwise type promotion kind.

    Type promotion in Thunder uses a lattice similar to JAX's.
    See https://jax.readthedocs.io/en/latest/type_promotion.html.

    Reviewing the inputs determines a "promotion dtype", but this function returns a
    "computation dtype" and a "result dtype."

    The "type_promotion_kind" argument determines how the promotion dtype is mapped to a
    computation and result dtype.

    PRESERVE preserves the promotion dtype as the computation and result dtype.
    It's appropriate for kernels that perform no mathematical operations on their tensors.

    DEFAULT type promotion selects a computation dtype by mapping low precision promotion dtypes to their
    higher precision counterparts:

      float16   -> float32
      bfloat16  -> float32
      complex32 -> complex64

    The result dtype is the same as the promotion dtype.

    INT_TO_FLOAT is like DEFAULT, except integer promotion dtypes map to float for their
    computation and result dtypes.

    COMPLEX_TO_FLOAT is like DEFAULT, except complex promotion dtypes have their corresponding
    float dtypes as a return dtype:

        complex32  -> float16
        complex64  -> float32
        complex128 -> float64

    BOOL_TO_LONG is like DEFAULT, except boolean promotion dtypes use int64 for their computation
    and result dtypes.

    ALWAYS_BOOL is like PRESERVE, except the result dtype is always bool.

    Example operators for each type promotion option:

      DEFAULT                 : add
      PRESERVE                : where, nextafter, cat
      INT_TO_FLOAT            : sin
      COMPLEX_TO_FLOAT        : abs
      BOOL_TO_LONG            : pow
      ALWAYS_BOOL             : eq
    """

    # Type checks inputs
    assert all(isinstance(a, (TensorProxy, Number)) for a in args)
    assert len(args) > 0

    # Computes the promotion type
    extracted = (to_dtype(x, true_dtype=True) for x in args)
    promotiontype = reduce(_elementwise_type_promotion, extracted, bool)

    # Applies the different kinds of type promotion
    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE:
        return promotiontype, promotiontype

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        return promotiontype, bool

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT and is_integer_dtype(promotiontype):
        return float, float

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT and is_complex_dtype(promotiontype):
        if is_low_precision_dtype(promotiontype):
            return get_computation_dtype(promotiontype), dtypes.corresponding_real_dtype(promotiontype)
        return promotiontype, dtypes.corresponding_real_dtype(promotiontype)

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG and is_boolean_dtype(promotiontype):
        return int, int

    # Falls through to DEFAULT
    if is_low_precision_dtype(promotiontype):
        return get_computation_dtype(promotiontype), promotiontype
    return promotiontype, promotiontype


#
# Shape-related functions
#


def extract_shape_from_varargs(shape):
    """
    Returns a shape from varargs.
    In PyTorch, operations that accept shapes often accept them as varargs, like
    foo(*shape). However a user can pass the shape as a sequence of integers,
    like this:
      foo(1, 2, 3)
    or as a sequence of integers
      foo((1, 2, 3))
    In the first case shape will be a tuple of integers, and in the second case it's a tuple
    containing a tuple of integers. This validates those inputs and canonicalizes them
    to a tuple of integers.
    """

    # Handles tuple unwrapping
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]

    return shape


def is_numbertensor(t):
    """True if the input is a "number tensor" -- a single element tensor with an empty shape.

    False otherwise.
    """
    if len(t.shape) == 0:
        return True

    return False


# TODO: maybe generalize to *args like check_same_dtype
# TODO: change to check_same_shape or add check_same_shape variant and make check_same_dtype use the same pattern
def same_shape(a, b):
    return tuple(a) == tuple(b)


# TODO: improve error message
def check_same_shape(*args):
    shapes = tuple(x.shape for x in args if isinstance(x, TensorProxy))
    if len(shapes) > 1:
        shape = shapes[0]
        for othershape in shapes[1:]:
            check(
                same_shape(shape, othershape),
                lambda: f"Shapes were expected to be the same, but got shapes {shape} and {othershape}!",
            )


# "Wraps" a dim (up to one time) for the given rank, allowing dims to be
# specified using negative indices. For scalar tensors with rank 0, then idx
# must be in the range [-1, 0]. Otherwise, idx should be in the range [-rank, rank-1].
def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool = True) -> int:
    check(rank >= 0, lambda: f"Rank cannot be negative but got {rank}!")

    if rank == 0:
        check(
            wrap_scalar,
            lambda: f"Dimension specified as {idx} but tensor has no dimensions!",
            exception_type=IndexError,
        )
        rank = 1

    if idx >= 0 and idx < rank:
        return idx

    if idx < 0:
        _idx = idx + rank
    else:
        _idx = idx

    check(
        _idx >= 0 and _idx < rank,
        lambda: f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {idx})",
        exception_type=IndexError,
    )

    return _idx


def canonicalize_dim_idx(dim_length, idx):
    check(dim_length >= 0, lambda: f"The length of a dimension ({dim_length}) cannot be negative!")

    # NOTE: consider adding a flag for when idx >= dim_length
    #   Ops like torch.tensor_split allow indices greater than the length of a dimension to be specified
    if idx >= 0:
        return idx

    if idx < 0:
        return idx + dim_length

    return idx


def canonicalize_dims(rank, indices, wrap_scalar=True):
    if isinstance(indices, int):
        return canonicalize_dim(rank, indices, wrap_scalar)

    return tuple(canonicalize_dim(rank, x, wrap_scalar) for x in indices)


def check_valid_length(length: int):
    """Validates that an object represents a valid dimension length."""

    check(length >= 0, lambda: f"Found invalid length {length}!")


def check_valid_permutation(rank: int, perm):
    """
    Validates that perm is a permutation of length rank.
    """

    check(isinstance(perm, Sequence), lambda: f"Expected perm={perm} to be a Sequence!")
    check(tuple(sorted(perm)) == tuple(range(0, rank)), lambda: f"Expected perm={perm} to be a valid permutation!")


def check_valid_shape(shape):
    """Validates that a sequence represents a valid shape."""

    for l in shape:
        check_valid_length(l)


def validate_idx(rank: int, idx: int):
    """Validates that idx is a valid index for the given shape.

    Assumes the index is already canonicalized.
    """

    check(
        idx >= 0 and (idx < rank or idx == 0),
        lambda: f"Found invalid index {idx} for rank {rank}!",
    )


def check_no_duplicates(dims: Sequence):
    def _reify(x):
        if isinstance(x, NumberProxy):
            return x.value

        return x

    dims = tuple(_reify(x) for x in dims)

    check(len(dims) == len(set(dims)), lambda: f"Duplicate value in list of dimensions {dims}!")


#
# Device-related functions
#

# TODO: improve device handling
def check_same_device(*args):
    devices = tuple(x.device for x in args if isinstance(x, TensorProxy))
    if len(devices) > 1:
        device = devices[0]
        for otherdevice in devices[1:]:
            check(
                same_shape(device, otherdevice),
                lambda: f"Devices were expected to be the same, but got devices {device} and {otherdevice}!",
            )


#
# Context-related functions and decorators
#

# TODO: think about preserving the original function's signature
class langctx:
    """A decorator that calls the decorated function in the given language context, resetting to the caller's language
    context when the function is done."""

    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, fn_):
        @wraps(fn_)
        def fn(*args, **kwargs):
            tok = trace.set_language_context(self.ctx)
            result = fn_(*args, **kwargs)
            trace.reset_language_context(tok)
            return result

        return fn


def safe_map(f, *args):
    """Apply f to each element of args, which must all have the same length.

    Args:
        f: function to apply
        *args: arguments to apply f to

    Returns:
        list of results of applying f to each element of args
    """
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))


def safe_zip(*args):
    """Zip args, which must all have the same length.

    Args:
        *args: arguments to zip

    Returns:
        list of zipped results
    """
    return safe_map(lambda *x: x, *args)


def unzip2(pairs):
    """Unzip a list of pairs.

    Args:
        pairs (list): list of pairs

    Returns:
        list of first elements of pairs, list of second elements of pairs
    """
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2
