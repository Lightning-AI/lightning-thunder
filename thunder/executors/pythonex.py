from typing import List, Any, Dict, Tuple, Union, Type
from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Sequence
from numbers import Number
import math
import operator
import builtins
import cmath
from types import ModuleType
import platform

import torch

import thunder.core.prims as prims
from thunder.core.prims import PrimIDs
from thunder.core.proxies import TensorProxy, CollectionProxy
from thunder.core.symbol import Symbol, BoundSymbol
from thunder.core import baseutils
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
import thunder.core.utils as utils

from thunder.extend import OperatorExecutor, register_executor, add_always_executor

# NOTE The criterion for being an "always" executor is that this does not introduce symbols that
#   it does not execute in its execution transforms
ex = OperatorExecutor("python", version=platform.python_version())
register_executor(ex)
add_always_executor(ex)

#
# Helper functions
#


def _always_executable(*args, **kwargs) -> bool:
    return True


def _never_executable(*args, **kwargs) -> bool:
    return False


#
# Unpacking primitives
#
def _check_tensor_shape_and_metadata_impl(
    t: torch.Tensor, shape: tuple[int, ...], device: str, dtype: torch.dtype, requires_grad: bool
) -> None:
    assert isinstance(t, torch.Tensor), f"expected Tensor, got {type(t).__name__}"
    assert (
        tuple(t.shape) == shape and str(t.device) == device and t.dtype == dtype and t.requires_grad == requires_grad
    ), f"expected tensor with {shape}, {device}, {dtype}, {requires_grad=}, got {tuple(t.shape)}, {str(t.device)}, {t.dtype}, {requires_grad}"


check_tensor_shape_and_metadata = ex.register_operator(
    "check_tensor_metadata", like=prims.check_tensor_shape_and_metadata, fn=_check_tensor_shape_and_metadata_impl
)
ex.register_implementation(
    prims.check_tensor_shape_and_metadata, check_tensor_shape_and_metadata, checker=_always_executable
)


def _check_literal_like_impl(p: Any, v: Any, /) -> None:
    utils.check(p == v, lambda: f"Expected {p} to be equal to {v}")


check_literal_like = ex.register_operator(
    "check_literal_like", like=prims.check_literal_like, fn=_check_literal_like_impl
)
ex.register_implementation(prims.check_literal_like, check_literal_like, checker=_always_executable)


def _check_none_impl(n: None, /) -> None:
    utils.check(n is None, lambda: f"Expected {n} to be None")


check_none = ex.register_operator("check_none", like=prims.check_none, fn=_check_none_impl)
ex.register_implementation(prims.check_none, check_none, checker=_always_executable)


def _check_type_impl(x: Any, typ: type, /) -> None:
    utils.check(type(x) is typ, lambda: f"Expected {x} to be have the type {typ}")


check_type = ex.register_operator("check_type", like=prims.check_type, fn=_check_type_impl)
ex.register_implementation(prims.check_type, check_type, checker=_always_executable)


def _check_instance_impl(x: Any, types: tuple[type], /) -> None:
    utils.check(isinstance(x, types), lambda: f"Expected {x} to be an instance of one of {types}")


check_instance = ex.register_operator("check_instance", like=prims.check_instance, fn=_check_instance_impl)
ex.register_implementation(prims.check_instance, check_instance, checker=_always_executable)


def _check_number_type_and_value_impl(n: Number, v: Number) -> None:
    utils.check(
        type(n) == type(v) and (n == v or (n != n and v != v)),
        lambda: f"Expected {n} to be equal to and have the type of {v}",
    )


check_number_type_and_value = ex.register_operator(
    "check_number_type_and_value", like=prims.check_number_type_and_value, fn=_check_number_type_and_value_impl
)
ex.register_implementation(prims.check_number_type_and_value, check_number_type_and_value, checker=_always_executable)


def _check_bool_conversion_impl(n: Number, b: bool, /) -> None:
    assert bool(n) is b


check_bool_conversion = ex.register_operator(
    "check_bool_conversion", like=prims.check_bool_conversion, fn=_check_bool_conversion_impl
)
ex.register_implementation(prims.check_bool_conversion, check_bool_conversion, checker=_always_executable)


def _check_string_value_impl(s: str, value: str) -> None:
    utils.check(s == value, lambda: f"Expected '{s}' to be equal to '{value}'")


check_string_value = ex.register_operator(
    "check_string_value", like=prims.check_string_value, fn=_check_string_value_impl
)
ex.register_implementation(prims.check_string_value, check_string_value, checker=_always_executable)


def _unpack_tuple_impl(tup: tuple, /) -> tuple:
    return tup


unpack_tuple = ex.register_operator("unpack_tuple", like=prims.unpack_tuple, fn=_unpack_tuple_impl)
ex.register_implementation(prims.unpack_tuple, unpack_tuple, checker=_always_executable)


def _unpack_list_impl(lst: list, /) -> list:
    return lst


unpack_list = ex.register_operator("unpack_list", like=prims.unpack_list, fn=_unpack_list_impl)
ex.register_implementation(prims.unpack_list, unpack_list, checker=_always_executable)


def _check_empty_impl(seq: tuple | list, /) -> None:
    utils.check(not len(seq), lambda: f"Expected {seq=} to be empty")


check_empty = ex.register_operator("check_empty", like=prims.check_empty, fn=_check_empty_impl)
ex.register_implementation(prims.check_empty, check_empty, checker=_always_executable)


def _check_len_impl(seq: tuple | list, length: int, /) -> None:
    utils.check(len(seq) == length, lambda: f"Expected {seq=} to be of length {length}")


check_len = ex.register_operator("check_len", like=prims.check_len, fn=_check_len_impl)
ex.register_implementation(prims.check_len, check_len, checker=_always_executable)


def _construct_tuple_impl(tup: tuple, /) -> None:
    return tup


construct_tuple = ex.register_operator("construct_tuple", like=prims.construct_tuple, fn=_construct_tuple_impl)
ex.register_implementation(prims.construct_tuple, construct_tuple, checker=_always_executable)


#
# Data movement and transformation prims
#


def _convert_element_type_prim_checker(a: Number | TensorProxy, /, dtype: type | dtypes.dtype) -> bool:
    return isinstance(a, Number) and dtype in (bool, int, float, complex)


def _convert_element_type_prim_impl(a: Number, dtype: type) -> Number:
    return dtype(a)


convert_element_type = ex.register_operator(
    "convert_element_type_prim", like=prims.convert_element_type, fn=_convert_element_type_prim_impl
)

ex.register_implementation(prims.convert_element_type, convert_element_type, checker=_convert_element_type_prim_checker)


#
# Elementwise unary primitives
#
# TODO Review math vs cmath and datatype support
# TODO Use NumPy to fill-in operations
# TODO Maybe exclusively use NumPy? Otherwise have to be more careful about cmath vs math and datatypes?
# TODO Review differences in type promotion (for example round(2.3))


def _elementwise_unary_checker(x: Number | TensorProxy) -> bool:
    return isinstance(x, Number)


# NOTE Tensor abs is distinct from Python's builtin abs because it preserves the dtype of booleans
#   i.e. tensor_abs(True) -> True, while builtins.abs(True) -> 1
def _tensor_abs_prim_impl(a: Number) -> Number:
    if type(a) is bool:
        return a

    return abs(a)


def _real_prim_impl(a: complex) -> float:
    return a.real


def _signbit_prim_impl(a: Number) -> bool:
    return a < 0


def _clear_collection_meta(coll: CollectionProxy) -> None:
    baseutils.check_type(coll, CollectionProxy)
    baseutils.check_type(coll.coll, Sequence)
    return None


def _clear_collection_prim_impl(a: Sequence) -> None:
    if isinstance(a, list):
        a.clear()


acos = ex.register_operator("acos", like=prims.acos, module=math)
acosh = ex.register_operator("acosh", like=prims.acosh, module=math)
asin = ex.register_operator("asin", like=prims.asin, module=math)
asinh = ex.register_operator("asinh", like=prims.asinh, module=math)
atan = ex.register_operator("atan", like=prims.atan, module=math)
atanh = ex.register_operator("atanh", like=prims.atanh, module=math)
py_abs = ex.register_operator("abs", like=prims.py_abs, module=builtins)
tensor_abs = ex.register_operator("tensor_abs", like=prims.abs, fn=_tensor_abs_prim_impl)
neg = ex.register_operator("neg", like=prims.neg, module=operator)
real = ex.register_operator("real", like=prims.real, fn=_real_prim_impl)
signbit = ex.register_operator("signbit", like=prims.signbit, fn=_signbit_prim_impl)
clear_collection = ex.register_operator("clear_collection", meta=_clear_collection_meta, fn=_clear_collection_prim_impl)

ex.register_implementation(prims.acos, acos, checker=_elementwise_unary_checker)
ex.register_implementation(prims.acosh, acosh, checker=_elementwise_unary_checker)
ex.register_implementation(prims.asin, asin, checker=_elementwise_unary_checker)
ex.register_implementation(prims.asinh, asinh, checker=_elementwise_unary_checker)
ex.register_implementation(prims.atan, atan, checker=_elementwise_unary_checker)
ex.register_implementation(prims.atanh, atanh, checker=_elementwise_unary_checker)
ex.register_implementation(prims.py_abs, py_abs, checker=_elementwise_unary_checker)
ex.register_implementation(prims.abs, tensor_abs, checker=_elementwise_unary_checker)
ex.register_implementation(prims.neg, neg, checker=_elementwise_unary_checker)
ex.register_implementation(prims.real, real, checker=_elementwise_unary_checker)
ex.register_implementation(prims.signbit, signbit, checker=_elementwise_unary_checker)


# # bitwise_not = _elementwise_unary_factory("invert", operator)
# # ceil = _elementwise_unary_factory("ceil", math)
# # cos = _elementwise_unary_factory("cos", math)
# # cosh = _elementwise_unary_factory("cosh", math)
# # erf = _elementwise_unary_factory("erf", math)
# # erfc = _elementwise_unary_factory("erfc", math)
# # erfcinv = None
# # erfinv = None
# # exp = _elementwise_unary_factory("exp", math)
# # exp2 = None
# # expm1 = _elementwise_unary_factory("expm1", math)
# # floor = _elementwise_unary_factory("floor", math)
# # isfinite = _elementwise_unary_factory("isfinite", cmath)
# # lgamma = _elementwise_unary_factory("lgamma", math)
# # log = _elementwise_unary_factory("log", math)
# # log10 = _elementwise_unary_factory("log10", math)
# # log1p = _elementwise_unary_factory("log1p", math)
# # log2 = _elementwise_unary_factory("log2", math)
# # ndtri = None
# # reciprocal = None
# # # NOTE pythonex_round to avoid a name conflict with the builtin round
# # pythonex_round = _elementwise_unary_factory("round", builtins)
# # rsqrt = None
# # sign = None
# # sin = _elementwise_unary_factory("sin", math)
# # sinh = _elementwise_unary_factory("sinh", math)
# # sqrt = _elementwise_unary_factory("sqrt", math)
# # tan = _elementwise_unary_factory("tan", math)
# # tanh = _elementwise_unary_factory("tanh", math)
# # trunc = _elementwise_unary_factory("trunc", math)

#
# Elementwise binary primitives
#


def _elementwise_binary_checker(a: Number | TensorProxy, b: Number | TensorProxy) -> bool:
    return isinstance(a, Number) and isinstance(b, Number)


add = ex.register_operator("add", like=prims.add, module=operator)
atan2 = ex.register_operator("atan2", like=prims.atan2, module=operator)
bitwise_and = ex.register_operator("bitwise_and", like=prims.bitwise_and, module=operator)
bitwise_or = ex.register_operator("bitwise_or", like=prims.bitwise_or, module=operator)
bitwise_xor = ex.register_operator("bitwise_xor", like=prims.bitwise_xor, module=operator)
eq = ex.register_operator("eq", like=prims.eq, module=operator)
py_floordiv = ex.register_operator("floordiv", like=prims.py_floordiv, module=operator)
fmod = ex.register_operator("fmod", like=prims.fmod, module=operator)
ge = ex.register_operator("ge", like=prims.ge, module=operator)
gt = ex.register_operator("gt", like=prims.gt, module=operator)
le = ex.register_operator("le", like=prims.le, module=operator)
lt = ex.register_operator("lt", like=prims.lt, module=operator)
mul = ex.register_operator("mul", like=prims.mul, module=operator)
ne = ex.register_operator("ne", like=prims.ne, module=operator)
# NOTE pythonex_pow to avoid a name conflict with the builtin pow
pythonex_pow = ex.register_operator("pow", like=prims.pow, module=operator)
sub = ex.register_operator("sub", like=prims.sub, module=operator)

# TODO: Restore truediv once we find it...
# truediv = ex.register_operator("truediv", like=prims.truediv, module=operator)

ex.register_implementation(prims.add, add, checker=_elementwise_binary_checker)
ex.register_implementation(prims.atan2, atan2, checker=_elementwise_binary_checker)
ex.register_implementation(prims.bitwise_and, bitwise_and, checker=_elementwise_binary_checker)
ex.register_implementation(prims.bitwise_or, bitwise_or, checker=_elementwise_binary_checker)
ex.register_implementation(prims.bitwise_xor, bitwise_xor, checker=_elementwise_binary_checker)
ex.register_implementation(prims.eq, eq, checker=_elementwise_binary_checker)
ex.register_implementation(prims.py_floordiv, py_floordiv, checker=_elementwise_binary_checker)
ex.register_implementation(prims.fmod, fmod, checker=_elementwise_binary_checker)
ex.register_implementation(prims.ge, ge, checker=_elementwise_binary_checker)
ex.register_implementation(prims.gt, gt, checker=_elementwise_binary_checker)
ex.register_implementation(prims.le, le, checker=_elementwise_binary_checker)
ex.register_implementation(prims.lt, lt, checker=_elementwise_binary_checker)
ex.register_implementation(prims.mul, mul, checker=_elementwise_binary_checker)
ex.register_implementation(prims.ne, ne, checker=_elementwise_binary_checker)
# NOTE pythonex_pow to avoid a name conflict with the builtin pow
ex.register_implementation(prims.pow, pythonex_pow, checker=_elementwise_binary_checker)
ex.register_implementation(prims.sub, sub, checker=_elementwise_binary_checker)

# TODO: Restore truediv once we find it...
# ex.register_implementation(prims.truediv, truediv, checker=_elementwise_binary_checker)
