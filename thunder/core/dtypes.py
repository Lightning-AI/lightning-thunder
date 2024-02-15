from typing import Any
from collections.abc import Iterable
from numbers import Number

import torch
import numpy as np

import thunder.core.baseutils as baseutils
from thunder.core.baseutils import NumberProxyInterface, TensorProxyInterface

# This file defines Thunder's dtypes (dtypes) and numbertypes and offers utilities for
#   working with them.
#
# Terminology:
#
#  - "type". In Thunder, the word "type" means what it does in Python, too -- a Python type.
#  - "dtype"/"dtype". "dtype" -- which is often abbreviated to "dtype" -- refers to the data format of
#      data in a tensor. Every tensor has a dtype.
#  - "numbertype". A Python type that Thunder recognizes as describing a number. Currently one of
#      bool, int, float or complex. In the future this may be generalized to allow more subclasses.
#
# In Thunder each dtype has a "strong" and "weak" variant, and these
#   variants affect type promotion. This is similar to JAX, except in JAX the strong/weak metadata
#   is on the tensor object, not the dtype itself.
# This mechanism for representing dtypes may change in the future.
#
# The numbertypes are also considered dtypes, with the following aliasing relationship:
#
# bool -> bool8
# int -> int64_
# float -> float32_
# complex -> complex64_
#
# The class hierarchy for Thunder's dtypes follows NumPy's, see
#   https://numpy.org/doc/stable/reference/arrays.scalars.html.
#
# NOTE: dtypes in Thunder are objects and not classes today. This may change in the future. Use
#   the helpers defined here for consistency and code maintainability.

# DEPENDENCIES:
#   - TensorProxy, from proxies.py, but not the entire file to avoid a circular dependency structure
#       This dependency could be eliminated by testing for the "dtype" attribute, instead of
#       querying the type

# TODO: maybe add nicknames for dtypes, like how torch.long is torch.int64
# TODO: support extensible dtype registration


_abstract_classes = set()


# TODO consider bytes per element to better identify low precision dtypes
class dtype:
    # TODO: in the future might want to use ABCMeta to prevent this and the
    #   abstract classes from being instantiated
    def __new__(cls, *args, **kwargs):
        if cls in _abstract_classes:
            raise TypeError(f"{cls} is abstract and cannot be instantiated!")

        return object.__new__(cls)

    def __init__(self, *, python_type, name, shortname, bytes, is_weak):
        self._python_type = python_type
        self._name = name
        self._shortname = shortname
        self._bytes = bytes
        self._is_weak = is_weak

    # NOTE: these are properties so they appear as read-only
    @property
    def python_type(self):
        return self._python_type

    @property
    def bytes(self):
        return self._bytes

    @property
    def is_weak(self):
        return self._is_weak

    def shortname(self):
        return f"{self._shortname}{8 * self._bytes}"

    # TODO Fix name printing
    def __repr__(self):
        return f"{self._name}{8 * self._bytes}{'_' if self._is_weak else ''}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self) -> int:
        return hash((self._name, self._bytes, self._is_weak))

    def __eq__(self, other) -> bool:
        if not isinstance(other, dtype):
            return False

        return self._name == other._name and self._bytes == other._bytes and self._is_weak == other._is_weak


class exact(dtype):
    """Abstract base class for the signedinteger, unsignedinteger and bool_ dtypes."""


class signedinteger(exact):
    """Base class for the signed integer dtypes: int8, int16, int32, int64."""

    def __init__(self, name, shortname, *, bytes, is_weak):
        super().__init__(python_type=int, name=name, shortname=shortname, bytes=bytes, is_weak=is_weak)


int8 = signedinteger("int", "i", bytes=1, is_weak=False)
int8_ = signedinteger("int", "i", bytes=1, is_weak=True)
int16 = signedinteger("int", "i", bytes=2, is_weak=False)
int16_ = signedinteger("int", "i", bytes=2, is_weak=True)
int32 = signedinteger("int", "i", bytes=4, is_weak=False)
int32_ = signedinteger("int", "i", bytes=4, is_weak=True)
int64 = signedinteger("int", "i", bytes=8, is_weak=False)
int64_ = signedinteger("int", "i", bytes=8, is_weak=True)


class unsignedinteger(exact):
    """Base class for the unsigned integer dtypes: uint8."""

    def __init__(self, name, shortname, *, bytes, is_weak):
        super().__init__(python_type=int, name=name, shortname=shortname, bytes=bytes, is_weak=is_weak)


uint8 = unsignedinteger("uint", "ui", bytes=1, is_weak=False)
uint8_ = unsignedinteger("uint", "ui", bytes=1, is_weak=True)


class bool_(exact):
    """Base class for the boolean dtype: bool8."""

    def __init__(self, name, shortname, *, is_weak):
        super().__init__(python_type=bool, name=name, shortname=shortname, bytes=1, is_weak=is_weak)


# NOTE: bool has a weak variant for completeness, but the boolean dtype could always
#   be considered strong or weak without any effect
# TODO: review bool_ vs bool8 naming -- are we suggesting a byte per bool?
bool8 = bool_("bool", "b", is_weak=False)
bool8_ = bool_("bool", "b", is_weak=True)


class inexact(dtype):
    """Abstract base class for the floating and complexfloating dtypes."""

    pass


class floating(inexact):
    """Base class for the floating dtypes: bfloat16, float16, float32, float64."""

    def __init__(self, name, shortname, *, bytes, is_weak):
        super().__init__(python_type=float, name=name, shortname=shortname, bytes=bytes, is_weak=is_weak)


bfloat16 = floating("bfloat", "bf", bytes=2, is_weak=False)
bfloat16_ = floating("bfloat", "bf", bytes=2, is_weak=True)
float16 = floating("float", "f", bytes=2, is_weak=False)
float16_ = floating("float", "f", bytes=2, is_weak=True)
float32 = floating("float", "f", bytes=4, is_weak=False)
float32_ = floating("float", "f", bytes=4, is_weak=True)
float64 = floating("float", "f", bytes=8, is_weak=False)
float64_ = floating("float", "f", bytes=8, is_weak=True)


class complexfloating(inexact):
    """Base class for the complex floating dtypes: complex32, complex64, complex128."""

    def __init__(self, name, shortname, *, bytes, is_weak):
        super().__init__(python_type=complex, name=name, shortname=shortname, bytes=bytes, is_weak=is_weak)


complex32 = complexfloating("complex", "c", bytes=4, is_weak=False)
complex32_ = complexfloating("complex", "c", bytes=4, is_weak=True)
complex64 = complexfloating("complex", "c", bytes=8, is_weak=False)
complex64_ = complexfloating("complex", "c", bytes=8, is_weak=True)
complex128 = complexfloating("complex", "c", bytes=16, is_weak=False)
complex128_ = complexfloating("complex", "c", bytes=16, is_weak=True)


_abstract_classes.update((dtype, exact, inexact))

all_dtypes = {
    bool8,
    bool8_,
    uint8,
    uint8_,
    int8,
    int8_,
    int16,
    int16_,
    int32,
    int32_,
    int64,
    int64_,
    bfloat16,
    bfloat16_,
    float16,
    float16_,
    float32,
    float32_,
    float64,
    float64_,
    complex32,
    complex32_,
    complex64,
    complex64_,
    complex128,
    complex128_,
}

all_numbertypes = {bool, int, float, complex}

all_dtypes_and_numbertypes = all_dtypes | all_numbertypes

_numbertype_to_dtype_map = {
    bool: bool8_,
    int: int64_,
    complex: complex64_,
    float: float32_,
}

boolean_dtypes = {bool8, bool8_, bool}

integer_dtypes = {d for d in all_dtypes if isinstance(d, exact)} | {bool, int}

nonboolean_integer_dtypes = {d for d in integer_dtypes if (not isinstance(d, bool_) and d is not bool)}

# NOTE alias for the above
exact_dtypes = integer_dtypes

low_precision_dtypes = {
    d
    for d in all_dtypes
    if (isinstance(d, inexact) and d.bytes <= 2) or (isinstance(d, complexfloating) and d.bytes <= 4)
}

float_dtypes = {d for d in all_dtypes if isinstance(d, floating)} | {float}

complex_dtypes = {d for d in all_dtypes if isinstance(d, complexfloating)} | {complex}

inexact_dtypes = float_dtypes | complex_dtypes

weak_dtypes = {d for d in all_dtypes if d.is_weak} | all_numbertypes

strong_dtypes = {d for d in all_dtypes if not d.is_weak}


def is_weak_dtype(dtype):
    if dtype in all_numbertypes:
        return True

    return dtype.is_weak


def _numberclass_to_numbertype(cls):
    if issubclass(cls, bool):
        return bool
    if issubclass(cls, int):
        return int
    if issubclass(cls, complex):
        return complex
    if issubclass(cls, float):
        return float

    raise ValueError(f"Trying to convert unknown type {cls} to a numbertype!")


def to_dtype(x: Any, /, *, true_dtype: bool = False) -> None | dtype:
    """Exctracts a dtype from an object or class."""

    if x is None or isinstance(x, dtype):
        return x
    if isinstance(x, np.ndarray):
        return _numpy_to_thunder_dtype_map[x.dtype]
    if isinstance(x, np.dtype):
        return _numpy_to_thunder_dtype_map[x.type]
    if isinstance(x, torch.Tensor):
        return _torch_to_thunder_dtype_map[x.dtype]
    if isinstance(x, torch.dtype):
        return _torch_to_thunder_dtype_map[x]
    if isinstance(x, TensorProxyInterface):
        if true_dtype:
            return x.true_dtype
        return x.dtype
    if isinstance(x, dtype):
        return x
    if isinstance(x, NumberProxyInterface):
        return x.python_type
    if isinstance(x, Number):
        return _numberclass_to_numbertype(type(x))
    if isinstance(x, type) and issubclass(x, Number):
        return _numberclass_to_numbertype(x)

    raise ValueError(f"Trying to extract a dtype from object {x} with unknown type {type(x)}!")


def has_subdtype(x, cls):
    dtype = to_dtype(x)
    return isinstance(dtype, cls)


# Translates a sequence of dtypes and dtype classes into a concrete set of corresponding (strong) dtypes
def resolve_dtypes(args):
    dtypes = set()
    for arg in args:
        if isinstance(arg, dtype):
            dtypes.add(arg)
            continue

        if isinstance(arg, Iterable):
            for a in arg:
                baseutils.check(
                    isinstance(a, dtype),
                    lambda: f"Iterables passed to resolve_dtypes must only contain dtypes, but found an Iterable with {a}",
                    exception_type=NotImplementedError,
                )
                dtypes.add(a)

        baseutils.check(
            arg in (dtype, exact, signedinteger, unsignedinteger, bool_, inexact, floating, complexfloating),
            lambda: f"Excepted arguments to resolve_dtypes to be dtypes, sets of dtypes, or a dtype (sub)class, but got {arg}",
            exception_type=AssertionError,
        )

        updates = tuple(dtype for dtype in all_dtypes if isinstance(dtype, arg) and not dtype.is_weak)
        dtypes.update(updates)

    return dtypes


_complex_to_real_dtype_map = {
    complex128_: float64_,
    complex128: float64,
    complex64_: float32_,
    complex64: float32,
    complex32_: float16_,
    complex32: float16,
    complex: float,
}

_real_to_complex_dtype_map = {
    bfloat16_: complex64_,
    bfloat16: complex64,
    float16_: complex32_,
    float16: complex32,
    float32_: complex64_,
    float32: complex64,
    float64_: complex128_,
    float64: complex128,
    float: complex,
}


def corresponding_real_dtype(dtype):
    return _complex_to_real_dtype_map[dtype]


def corresponding_complex_dtype(dtype):
    return _real_to_complex_dtype_map[dtype]


_strong_dtype_to_weak_dtype_map = {
    bool8: bool8_,
    uint8: uint8_,
    int8: int8_,
    int16: int16_,
    int32: int32_,
    int64: int64_,
    bfloat16: bfloat16_,
    float16: float16_,
    float32: float32_,
    float64: float64_,
    complex32: complex32_,
    complex64: complex64_,
    complex128: complex128_,
}

_weak_dtype_to_strong_dtype_map = {v: k for k, v in _strong_dtype_to_weak_dtype_map.items()}
_weak_dtype_to_strong_dtype_map.update(
    {
        bool: bool8,
        int: int64,
        float: float32,
        complex: complex64,
    }
)


def to_weak_dtype(dtype):
    dtype = to_dtype(dtype)
    if is_weak_dtype(dtype):
        return dtype
    return _strong_dtype_to_weak_dtype_map[dtype]


def to_strong_dtype(dtype):
    dtype = to_dtype(dtype)
    if not is_weak_dtype(dtype):
        return dtype
    return _weak_dtype_to_strong_dtype_map[dtype]


def is_boolean_dtype(dtype) -> bool:
    return dtype in boolean_dtypes


def is_unsigned_dtype(dtype):
    return is_boolean_dtype(dtype) or isinstance(dtype, unsignedinteger)


def is_signedinteger_dtype(dtype):
    if is_unsigned_dtype(dtype):
        return False

    return dtype in integer_dtypes


def is_integer_dtype(dtype) -> bool:
    return dtype in integer_dtypes


# Alias for is_integer_dtype
is_exact_dtype = is_integer_dtype


def is_nonboolean_integer_dtype(dtype) -> bool:
    return dtype in nonboolean_integer_dtypes


def is_low_precision_dtype(dtype) -> bool:
    return dtype in low_precision_dtypes


def is_float_dtype(dtype) -> bool:
    return dtype in float_dtypes


def is_complex_dtype(dtype) -> bool:
    return dtype in complex_dtypes


def is_inexact_dtype(dtype):
    return is_float_dtype(dtype) or is_complex_dtype(dtype)


# TODO: we could consider a more general notion of number defined by issubclass(typ, Number)
def is_numbertype(x):
    # Note: the first argument to issubclass must be a type
    if not type(x) == type:
        return False
    return issubclass(x, tuple(all_numbertypes))


def is_dtype(x):
    return x in all_dtypes or is_numbertype(x)


def dtype_to_numbertype(dtype):
    dtype = to_dtype(dtype)

    if is_boolean_dtype(dtype):
        return bool
    if is_integer_dtype(dtype):
        return int
    if is_float_dtype(dtype):
        return float
    if is_complex_dtype(dtype):
        return complex

    raise ValueError(f"Trying to extract a numbertype from non-dtype object {dtype}!")


def numbertype_to_dtype(dtype):
    if not is_numbertype(dtype):
        raise ValueError(f"Trying to extract a dtype from a non-numbertype object {dtype}!")

    dtype = to_dtype(dtype)
    return _numbertype_to_dtype_map[dtype]


def are_same_dtypes(a, b, *, weak_and_strong_are_equivalent=True):
    a, b = to_dtype(a), to_dtype(b)

    # Handles float -> float32_ aliasing by canonicalizing both a and b
    if is_numbertype(a):
        a = numbertype_to_dtype(a)
    if is_numbertype(b):
        b = numbertype_to_dtype(b)

    # Handles float32_ vs float32 by canonicalizing the dtypes to their strong variants
    if weak_and_strong_are_equivalent:
        a, b = to_strong_dtype(a), to_strong_dtype(b)

    return a is b


# Converts PyTorch dtypes to and from thunder dtypes
_thunder_to_torch_dtype_map = {
    bool: torch.bool,
    int: torch.int32,
    float: torch.float32,
    complex: torch.complex64,
    bool8_: torch.bool,
    bool8: torch.bool,
    uint8_: torch.uint8,
    uint8: torch.uint8,
    int8_: torch.int8,
    int8: torch.int8,
    int16_: torch.int16,
    int16: torch.int16,
    int32_: torch.int32,
    int32: torch.int32,
    int64_: torch.int64,
    int64: torch.int64,
    bfloat16_: torch.bfloat16,
    bfloat16: torch.bfloat16,
    float16_: torch.float16,
    float16: torch.float16,
    float32_: torch.float32,
    float32: torch.float32,
    float64_: torch.float64,
    float64: torch.float64,
    complex32_: torch.complex32,
    complex32: torch.complex32,
    complex64_: torch.complex64,
    complex64: torch.complex64,
    complex128_: torch.complex128,
    complex128: torch.complex128,
}

_torch_to_thunder_dtype_map = {
    v: k
    for k, v in _thunder_to_torch_dtype_map.items()
    if not is_weak_dtype(k) and not (type(k) is type and issubclass(k, Number))
}


def to_torch_dtype(x: None | torch.dtype | dtype) -> None | torch.dtype:
    if x is None or isinstance(x, torch.dtype):
        return x

    baseutils.check_type(x, dtype)
    return _thunder_to_torch_dtype_map[x]


# Converts NumPy dtypes to and from thunder dtypes

# NOTE NumPy does not support the bfloat16 or complexhalf (complex32) datatypes
_thunder_to_numpy_dtype_map = {
    bool: np.bool_,
    int: np.int_,
    float: np.float_,
    complex: np.cfloat,
    bool8_: np.bool_,
    bool8: np.bool_,
    uint8_: np.uint8,
    uint8: np.uint8,
    int8_: np.int8,
    int8: np.int8,
    int16_: np.int16,
    int16: np.int16,
    int32_: np.int32,
    int32: np.int32,
    int64_: np.int64,
    int64: np.int64,
    float16_: np.float16,
    float16: np.float16,
    float32_: np.float32,
    float32: np.float32,
    float64_: np.float64,
    float64: np.float64,
    complex64_: np.complex64,
    complex64: np.complex64,
    complex128_: np.complex128,
    complex128: np.complex128,
}

_numpy_to_thunder_dtype_map = {
    v: k
    for k, v in _thunder_to_numpy_dtype_map.items()
    if not is_weak_dtype(k) and not (type(k) is type and issubclass(k, Number))
}


def to_numpy_dtype(x: None | np.dtype | dtype) -> None | np.dtype:
    if x is None or isinstance(x, np.dtype):
        return x

    baseutils.check_type(x, dtype)
    return _thunder_to_numpy_dtype_map[x]
