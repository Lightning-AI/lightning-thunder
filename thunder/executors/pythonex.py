from typing import List, Callable, Any, Dict, Tuple, Union, Type
from collections.abc import Sequence
from numbers import Number
import math
import operator
import builtins
import cmath

import thunder.core.prims as prims
from thunder.core.prims import PrimIDs
from thunder.core.proxies import TensorProxy
from thunder.core.trace import TraceCtx
from thunder.core.symbol import Symbol, BoundSymbol
import thunder.core.dtypes as dtypes

# NOTE _ops_map is declared here and defined after the callables have been defined
#   below
_ops_map: dict[Any, tuple[Callable, Callable]] = {}


def _never_executable(*args, **kwargs) -> bool:
    return False


#
# Data movement and transformation prims
#


def _convert_element_type_check(a: Union[TensorProxy, Number], dtype: Union[type, dtypes.dtype]) -> bool:
    return isinstance(a, Number) and dtype in (bool, int, float, complex)


def convert_element_type(bsym: BoundSymbol, a: Union[TensorProxy, Number], dtype: Union[type, dtypes.dtype]) -> Number:
    name = {bool: "bool", int: "int", float: "float", complex: "complex"}[dtype]

    sym = Symbol(name=name, meta=None, _module=builtins)
    tbsym = BoundSymbol(sym, args=(a,), kwargs={}, output=bsym.output)

    return tbsym


#
# Elementwise unary operations
#


def _elementwise_unary_check(a: Union[TensorProxy, Number]) -> bool:
    return isinstance(a, Number)


def _elementwise_unary_factory(name: str, module) -> Callable:
    def fn(bsym: BoundSymbol, a: Union[TensorProxy, Number]) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=module)
        tbsym = BoundSymbol(sym, args=(a,), kwargs={}, output=bsym.output)

        return tbsym

    return fn


# TODO Review math vs cmath and datatype support
# TODO Use NumPy to fill-in operations
# TODO Maybe exclusively use NumPy? Otherwise have to be more careful about cmath vs math and datatypes?
# TODO Review differences in type promotion (for example round(2.3))

# NOTE pythonex_abs to avoid a name conflict with the builtin abs
pythonex_abs = _elementwise_unary_factory("abs", builtins)
acos = _elementwise_unary_factory("acos", math)
acosh = _elementwise_unary_factory("acosh", math)
asin = _elementwise_unary_factory("asin", math)
asinh = _elementwise_unary_factory("asinh", math)
atan = _elementwise_unary_factory("atan", math)
atanh = _elementwise_unary_factory("atanh", math)
bitwise_not = _elementwise_unary_factory("invert", operator)
ceil = _elementwise_unary_factory("ceil", math)
cos = _elementwise_unary_factory("cos", math)
cosh = _elementwise_unary_factory("cosh", math)
erf = _elementwise_unary_factory("erf", math)
erfc = _elementwise_unary_factory("erfc", math)
erfcinv = None
erfinv = None
exp = _elementwise_unary_factory("exp", math)
exp2 = None
expm1 = _elementwise_unary_factory("expm1", math)
floor = _elementwise_unary_factory("floor", math)
isfinite = _elementwise_unary_factory("isfinite", cmath)
lgamma = _elementwise_unary_factory("lgamma", math)
log = _elementwise_unary_factory("log", math)
log10 = _elementwise_unary_factory("log10", math)
log1p = _elementwise_unary_factory("log1p", math)
log2 = _elementwise_unary_factory("log2", math)
ndtri = None
neg = _elementwise_unary_factory("neg", operator)
reciprocal = None
# NOTE pythonex_round to avoid a name conflict with the builtin round
pythonex_round = _elementwise_unary_factory("round", builtins)
rsqrt = None
sign = None
sin = _elementwise_unary_factory("sin", math)
sinh = _elementwise_unary_factory("sinh", math)
sqrt = _elementwise_unary_factory("sqrt", math)
tan = _elementwise_unary_factory("tan", math)
tanh = _elementwise_unary_factory("tanh", math)
trunc = _elementwise_unary_factory("trunc", math)

#
# Elementwise binary operations
#


def _elementwise_binary_check(a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> bool:
    return isinstance(a, Number) and isinstance(b, Number)


def _elementwise_binary_factory(name: str, module) -> Callable:
    def fn(bsym: BoundSymbol, a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=module)
        tbsym = BoundSymbol(sym, args=(a, b), kwargs={}, output=bsym.output)

        return tbsym

    return fn


add = _elementwise_binary_factory("add", operator)
sub = _elementwise_binary_factory("sub", operator)
atan2 = _elementwise_binary_factory("atan2", math)
mul = _elementwise_binary_factory("mul", operator)
truediv = _elementwise_binary_factory("truediv", operator)
# NOTE pythonex_pow to avoid a name conflict with the builtin pow
pythonex_pow = _elementwise_binary_factory("pow", operator)


# Maps from symbol ids to a tuple of (is_fusable, translate) callables
_ops_map.update(
    {
        # Data movement and transformation prims
        PrimIDs.CONVERT_ELEMENT_TYPE: (_convert_element_type_check, convert_element_type),
        # Elementwise unary prims
        PrimIDs.ABS: (_elementwise_unary_check, pythonex_abs),
        PrimIDs.ACOS: (_elementwise_unary_check, acos),
        PrimIDs.ACOSH: (_elementwise_unary_check, acosh),
        PrimIDs.ASIN: (_elementwise_unary_check, asin),
        PrimIDs.ASINH: (_elementwise_unary_check, asinh),
        PrimIDs.ATAN: (_elementwise_unary_check, atan),
        PrimIDs.ATANH: (_elementwise_unary_check, atanh),
        PrimIDs.BITWISE_NOT: (_elementwise_unary_check, bitwise_not),
        PrimIDs.CEIL: (_elementwise_unary_check, ceil),
        PrimIDs.COS: (_elementwise_unary_check, cos),
        PrimIDs.COSH: (_elementwise_unary_check, cosh),
        PrimIDs.ERF: (_elementwise_unary_check, erf),
        PrimIDs.ERFC: (_elementwise_unary_check, erfc),
        PrimIDs.ERFCINV: (_never_executable, erfcinv),
        PrimIDs.ERFINV: (_never_executable, erfinv),
        PrimIDs.EXP: (_elementwise_unary_check, exp),
        PrimIDs.EXP2: (_never_executable, exp2),
        PrimIDs.EXPM1: (_elementwise_unary_check, expm1),
        PrimIDs.FLOOR: (_elementwise_unary_check, floor),
        PrimIDs.ISFINITE: (_elementwise_unary_check, isfinite),
        PrimIDs.LGAMMA: (_elementwise_unary_check, lgamma),
        PrimIDs.LOG: (_elementwise_unary_check, log),
        PrimIDs.LOG10: (_elementwise_unary_check, log10),
        PrimIDs.LOG1P: (_elementwise_unary_check, log1p),
        PrimIDs.LOG2: (_elementwise_unary_check, log2),
        PrimIDs.NDTRI: (_never_executable, ndtri),
        PrimIDs.NEG: (_elementwise_unary_check, neg),
        PrimIDs.RECIPROCAL: (_never_executable, reciprocal),
        PrimIDs.ROUND: (_elementwise_unary_check, pythonex_round),
        PrimIDs.RSQRT: (_never_executable, rsqrt),
        PrimIDs.SIGN: (_never_executable, sign),
        PrimIDs.SIN: (_elementwise_unary_check, sin),
        PrimIDs.SINH: (_elementwise_unary_check, sinh),
        PrimIDs.SQRT: (_elementwise_unary_check, sqrt),
        PrimIDs.TAN: (_elementwise_unary_check, tan),
        PrimIDs.TANH: (_elementwise_unary_check, tanh),
        PrimIDs.TRUNC: (_elementwise_unary_check, trunc),
        # Elementwise binary prims
        PrimIDs.ADD: (_elementwise_binary_check, add),
        PrimIDs.SUB: (_elementwise_binary_check, sub),
        PrimIDs.ATAN2: (_elementwise_binary_check, atan2),
        PrimIDs.MUL: (_elementwise_binary_check, mul),
        PrimIDs.DIV: (_elementwise_binary_check, truediv),
        PrimIDs.POW: (_elementwise_binary_check, pythonex_pow),
    }
)

#
# Executor interface functions
#


# NOTE This is part of the executor interface
def is_supported(bsym: BoundSymbol, *, prims_only: bool = False) -> bool:
    sym = bsym.sym

    if prims_only and not sym.is_prim:
        return False

    # NOTE Symbols with Python implementations are supported
    if sym.python_impl is not None:
        return True

    check, _ = _ops_map.get(sym.id, (None, None))
    if check is None:
        return False
    return check(*bsym.args, **bsym.kwargs)


# NOTE This is part of the executor interface
def can_execute(bsym: BoundSymbol, *, prims_only: bool = False) -> bool:
    sym = bsym.sym

    # Some prims have explicit pure Python implementations which can be used
    if sym.python_impl is not None:
        return True

    if is_supported(bsym, prims_only=prims_only):
        return True

    if len(bsym.subsymbols) == 0:
        return False

    # Checks if all the operations this calls are executable
    can_execute_ = True
    for ssym in bsym.subsymbols:
        if not can_execute(ssym, prims_only=prims_only):
            can_execute_ = False
            break

    return can_execute_


def get_translator(bsym: BoundSymbol) -> Callable:
    return _ops_map[bsym.sym.id][1]


# NOTE This is part of the executor interface
def fuse(
    trace: TraceCtx, producers, consumers, bound_symbols: Sequence[BoundSymbol], counter: int
) -> list[BoundSymbol]:
    bsyms: list[BoundSymbol] = []

    for bsym in bound_symbols:
        # Symbols with Python implementations don't need to be translated
        if bsym.sym.python_impl is not None:
            bsyms.append(bsym)
        else:
            translator = get_translator(bsym)
            tbsym = translator(bsym, *bsym.args, **bsym.kwargs)
            bsyms.append(tbsym)

    return bsyms
