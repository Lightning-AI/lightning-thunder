from typing import List, Callable, Any, Dict, Tuple, Union, Type, Hashable
from collections.abc import Sequence
from numbers import Number
import math
import operator
import builtins
import cmath
from types import ModuleType
import platform

import thunder.core.prims as prims
from thunder.core.prims import PrimIDs
from thunder.core.proxies import TensorProxy
from thunder.core.symbol import Symbol, BoundSymbol
import thunder.core.dtypes as dtypes

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

# TODO Restore the operations below

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

ex.register_implementation(prims.add, add, checker=_elementwise_binary_checker)

# TODO Restore the operations below (updating them to the new style)

# atan2 = _elementwise_binary_factory("atan2", math)
# bitwise_and = _elementwise_binary_factory("and_", operator)
# bitwise_or = _elementwise_binary_factory("or_", operator)
# bitwise_xor = _elementwise_binary_factory("xor", operator)
# eq = _elementwise_binary_factory("eq", operator)
# fmod = _elementwise_binary_factory("fmod", math)
# ge = _elementwise_binary_factory("ge", operator)
# gt = _elementwise_binary_factory("gt", operator)
# le = _elementwise_binary_factory("le", operator)
# lt = _elementwise_binary_factory("lt", operator)
# sub = _elementwise_binary_factory("sub", operator)
# mul = _elementwise_binary_factory("mul", operator)
# ne = _elementwise_binary_factory("ne", operator)
# truediv = _elementwise_binary_factory("truediv", operator)
# # NOTE pythonex_pow to avoid a name conflict with the builtin pow
# pythonex_pow = _elementwise_binary_factory("pow", operator)
