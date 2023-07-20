import sys
from enum import Enum
from functools import reduce, wraps
import itertools
from itertools import zip_longest, chain
from numbers import Number
from typing import Callable, Dict, Generic, List, Optional, Type, TypeVar, Tuple
from collections.abc import Iterable, Sequence

from looseversion import LooseVersion

import thunder.core.dtypes as dtypes
from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
from thunder.core.proxies import Proxy, NumberProxy, TensorProxy
from thunder.core.baseutils import *
from thunder.core.codeutils import *
from thunder.core.trace import TraceCtx
import thunder.core.prims as prims

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
    "can_safe_cast_to",
    "check_same_dtype",
    "get_numberlike_type",
    "get_numberlike_value",
    "ELEMENTWISE_TYPE_PROMOTION_KIND",
    "get_computation_dtype",
    "elementwise_type_promotion",
    "const_as",
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
    # Helpful classes
    "OrderedSet",
    # Context-related functions and decorators
    "langctx",
]

T = TypeVar("T")
T1 = TypeVar("T1")

#
# Error checking helpers
#

# This file defines utilities that can be used when defining primitive operations.

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


def get_numberlike_value(values):
    def _extract(x):
        check_type(x, (Number, NumberProxy))
        if isinstance(x, NumberProxy):
            return x.value
        return x

    flat, spec = tree_flatten(values)
    modified = tuple(_extract(x) for x in flat)
    return tree_unflatten(modified, spec)


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
    check(len(args) > 0, lambda: f"Execpted one or more arguments for type promotion, but got {args=}")
    for a in args:
        check_type(a, (TensorProxy, Number))

    # Computes the promotion type
    extracted = tuple(to_dtype(x, true_dtype=True) for x in args)
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


def const_as(number, dtype):
    """
    Returns the value with the typed with the numbertype corresponding to the dtype
    """

    return dtype_to_numbertype(dtype)(number)


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


def check_valid_permutation(rank: int, perm):
    """
    Validates that perm is a permutation of length rank.
    """

    check(isinstance(perm, Sequence), lambda: f"Expected perm={perm} to be a Sequence!")
    check(tuple(sorted(perm)) == tuple(range(0, rank)), lambda: f"Expected perm={perm} to be a valid permutation!")


def validate_idx(rank: int, idx: int):
    """Validates that idx is a valid index for the given shape.

    Assumes the index is already canonicalized.
    """

    check(
        isinstance(idx, int) and idx >= 0 and (idx < rank or idx == 0),
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


# TODO: improve device handling by canonicalizing devices and expressing them per langctx
# TODO: should the comparison between devices be ==?
def check_same_device(*args):
    devices = tuple(x.device for x in args if isinstance(x, TensorProxyInterface))
    if len(devices) > 1:
        device = devices[0]
        for otherdevice in devices[1:]:
            check(
                device == otherdevice,
                lambda: f"Devices were expected to be the same, but got devices {device} and {otherdevice}!",
            )


#
# Helpful classes
#


# Mimics a set (see https://docs.python.org/3/library/stdtypes.html#set) but is
# ordered
# NOTE dicts in Python are ordered since Python 3.7
# TODO Implement additional methods as needed
class OrderedSet(Generic[T]):
    # TODO: allow construction of an empty ordered set without requiring an empty sequence be specified
    def __init__(self, args: Optional[Iterable[T]] = None):
        self.d = {self.canonicalize(k): None for k in args or ()}

    def canonicalize(self, v: T) -> T:
        """Subclasses can override to coerce to a common type."""
        return v

    def __repr__(self) -> str:
        contents = ", ".join(repr(i) for i in self)
        return f"{self.__class__.__name__}{{ {contents} }}"

    def __contains__(self, x: T) -> bool:
        return self.canonicalize(x) in self.d

    def __bool__(self) -> bool:
        return bool(self.d)

    def __iter__(self) -> Iterable[T]:
        return iter(self.d.keys())

    def __len__(self) -> int:
        return len(self.d)

    # -
    def __sub__(self, other: "OrderedSet") -> "OrderedSet":
        return self.__class__(k for k in self if k not in other)

    def __and__(self, other: "OrderedSet") -> "OrderedSet":
        return self.__class__(k for k in self if k in other)

    def __or__(self, other: "OrderedSet") -> "OrderedSet":
        return self.__class__(itertools.chain(self, other))

    # NOTE: actual set signature is (self, *others)
    def difference(self, other):
        return self - other

    def add(self, x):
        self.d[self.canonicalize(x)] = None

    def update(self, x):
        for i in x:
            self.d.setdefault(self.canonicalize(i), None)

    def remove(self, x):
        del self.d[self.canonicalize(x)]

    def copy(self) -> "OrderedSet":
        return self.__class__(self)


#
# Context-related functions and decorators
#


def flatten_func(func, args, kwargs):
    """Returns a flattened version of func.

    Flattened functions accept flattened arguments. The flattened arguments are
    the flattened version of (args, kwargs).

    Args:
        func: function to flatten args: positional arguments to pass to func
        kwargs: keyword arguments to pass to func

    Returns:
        tuple of (flattened function, flattened arguments, spec)
    """
    flat_args, spec = tree_flatten((args, kwargs))

    def flat_func(*flat_args):
        fn_args, fn_kwargs = tree_unflatten(flat_args, spec)
        return func(*fn_args, **fn_kwargs)

    return flat_func, flat_args, spec


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


def safe_map_flat(f, *args):
    args_flat_spec = safe_map(tree_flatten, args)
    _, spec = args_flat_spec[0]
    for i, (_, s) in enumerate(args_flat_spec[1:], start=1):
        assert s == spec, f"argument layout mismatch: {args[0]} {args[i]}"
    out_flat = list(map(f, *[a for a, _ in args_flat_spec]))
    return tree_unflatten(out_flat, spec)


def _safe_zip_gen(*args):
    # It has to be a separate function because it's a generator.
    null = object()
    for zipped in itertools.zip_longest(*args, fillvalue=null):
        if null in zipped:
            raise ValueError(f"length mismatch: {list(map(len, args))}")
        yield zipped


def safe_zip(*args):
    """Zip args, which must all have the same length.

    Args:
        *args: arguments to zip

    Returns:
        generator of zipped results

    Raises:
        ValueError: if the lengths of the arguments do not match
    """
    if sys.version_info >= (3, 10):
        return zip(*args, strict=True)
    return _safe_zip_gen(*args)


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


def dict_join(*args: list[dict[T, T1]]) -> dict[T, T1]:
    """Combine multiple dictionaries.

    Args:
        args: Dictionaries to concatenate. If there are collisions then *later*
              arguments take precidence.

    Returns:
        A single dictionary with all of the keys and values in args.
    """
    return dict(itertools.chain(*(i.items() for i in args)))


#
# Utilities related to traces
#


def get_name(trace: TraceCtx, x: Any) -> str:
    if isinstance(x, Proxy):
        return x.name

    to = trace.get_tracked_object(x)
    if isinstance(to, TrackedObject):
        return to.name

    check(False, lambda: f"Could not find name for {x}")


# A dictionary-like class with proxies as keys
# NOTE Why ProxyDict?
#   NumberProxies are hashed on their value, like Python numbers,
#   but this means that distinct NumberProxies whose values compare
#   equal would be hashed to the same value, and it's sometimes important
#   to distinguish the type and history of different NumberProxies.
#   (You can see this behavior by trying to use both 5 and 5. as keys in a dict.)
#   ProxyDict changes the hashing behavior of proxies to hash using their name,
#   which preserves type and history since names are unique.
class ProxyDict:
    def __init__(self, trace: TraceCtx):
        self._trace = trace
        self._dict = {}

    def __setitem__(self, key: Proxy, val: Any):
        key_ = key.name
        self._dict[key_] = val

    def __getitem__(self, key: Proxy) -> Any:
        check_type(key, Proxy)
        key_ = key.name
        return self._dict[key_]

    def __contains__(self, key: Proxy) -> bool:
        check_type(key, Proxy)
        key_ = key.name
        return key_ in self._dict

    # Helper when values are lists
    def append(self, key: Proxy, val: Any) -> None:
        key_ = key.name
        vals = self._dict.get(key_, [])
        check_type(vals, list)
        vals.append(val)
        self._dict[key_] = vals

    def remove(self, key: Proxy, val: Any) -> None:
        raise NotImplementedError

    def get(self, key: Proxy, default: Any) -> Any:
        try:
            return self.__getitem__(key)
        except Exception:
            pass

        return default

    # Acquires the proxy by name
    def get_by_name(self, name: str) -> Any:
        check_type(name, str)
        return self._dict[name]

    def __repr__(self) -> str:
        return str(self._dict)


# Returns a proxy -> producer mapping
def producers(trace: TraceCtx) -> ProxyDict:
    producers = ProxyDict(trace)

    # Skips symbols that never produce anything
    skip = {
        prims.PrimIDs.COMMENT,
        prims.PrimIDs.PRINT,
        prims.PrimIDs.RETURN,
    }

    for bsym in trace.bound_symbols:
        if bsym.sym in skip:
            continue

        flat = bsym._flat_outs
        for out in flat:
            if not isinstance(out, Proxy):
                continue

            # NOTE out may already be in producers because some operations return their input,
            #   which might look like the same output is being produced multiple times
            #   The first production is what we're interested in
            if out not in producers:
                producers[out] = bsym

    return producers


def consumers(trace: TraceCtx) -> ProxyDict:
    consumers = ProxyDict(trace)

    # Skips symbols that never consume anything
    skip = {
        prims.PrimIDs.COMMENT,
        prims.PrimIDs.UNPACK_TRIVIAL,
    }

    for bsym in trace.bound_symbols:
        if bsym.sym in skip:
            continue

        flatargs = bsym._flat_args
        flatkwargs = bsym._flat_kwargs

        for x in chain(flatargs, flatkwargs):
            if not isinstance(x, Proxy):
                continue

            consumers.append(x, bsym)

    return consumers


# TODO This could be optimized by computing producers and consumers at the same time
# Returns two ProxyDicts, the first mapping proxies to the bound symbol that produced them,
#   and the second mapping proxies to the bound symbols that consume them (if any)
# NOTE This only returns things that are produced and consumed by "top level" bound symbols
#   in the trace. It does not recurse into the bound symbols.
def producers_and_consumers(trace: TraceCtx) -> tuple[ProxyDict, ProxyDict]:
    return producers(trace), consumers(trace)


def find_producer_symbols(trace: TraceCtx, proxies: Sequence[Proxy], stop_proxies: Sequence[Proxy]) -> Tuple[Any, ...]:
    """Find the symbols that produce the given proxies.

    This function is useful for finding a set of symbols that can be used to
    compute the given proxies.

    Args:
        trace: trace context
        proxies: proxies to find producers for
        stop_proxies: proxies to stop at

    Returns:
        tuple of symbols that produce the given proxies

    Example:
        >>> import torch
        >>> import thunder
        >>> from thunder.core import utils
        >>> x = torch.randn(3, 4)
        >>> y = torch.randn(3, 4)
        >>> def f(x, y):
        ...     return (x + y) * (x - y)
        >>> compiled_f = thunder.compile(f)
        >>> _ = compiled_f(x, y)
        >>> trace = thunder.last_traces(compiled_f)[0]
        >>> x_proxy = trace.args[0]
        >>> y_proxy = trace.args[1]
        >>> intermediate = trace.bound_symbols[-3].output
        >>> utils.find_producer_symbols(trace, [intermediate], [x_proxy, y_proxy])
        (__b = ltorch.sub(x, y)
        # __b = prims.sub(x, y),)
    """
    trace_producers = producers(trace)
    result = set()
    queue = list(proxies)
    while queue:
        proxy = queue.pop()
        p = trace_producers.get(proxy, None)
        if p is not None:
            result.add(p)
            for arg in p._flat_args:
                arg_name = arg.name if isinstance(arg, Proxy) else None
                if arg_name not in map(lambda x: x.name, stop_proxies):
                    queue.append(arg)
    original_order = {bsym: i for i, bsym in enumerate(trace.bound_symbols)}
    return tuple(sorted(result, key=lambda x: original_order[x]))


def get_symbols_to_last_used_variables(symbols, ignore):
    """Get a mapping from symbols to the last used variables.

    Mark last used intermediates to be deleted. This is necessary to avoid memory leaks.

    Args:
        symbols: list of symbols
        ignore: list of variables to be ignored, they will not be marked as last used

    Returns:
        dict: mapping from symbols to the last used variables
    """
    ignore = (ignore,) if not isinstance(ignore, Sequence) else ignore
    ignore = tree_flatten(ignore)[0]
    variable_to_last_symbol = {}
    symbol_to_last_variables = {}

    def _mark_last_use(symbol, variable):
        if variable in ignore:
            return
        if not variable in variable_to_last_symbol:
            variable_to_last_symbol[variable] = symbol
            symbol_to_last_variables.setdefault(symbol, []).append(variable)

    for symbol in reversed(symbols):
        # If this function is used in the combined nvfuser+torch executor, there are no symbols but regions.
        # Regions do not have args, kwargs
        if hasattr(symbol, "inputs"):
            variables = tuple(symbol.inputs)
        else:
            variables = (symbol.args, symbol.kwargs)
        tree_map(lambda x: _mark_last_use(symbol, x) if isinstance(x, trace.Variable) else None, variables)
    return symbol_to_last_variables
