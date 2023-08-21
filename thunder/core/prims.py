from enum import auto, Enum
from numbers import Number
from functools import reduce
import operator
import builtins
import math
from typing import Union, Type, Any, List, Dict, Tuple, Optional, Callable, Hashable
from collections.abc import Sequence

import torch
import numpy as np

from thunder.core.symbol import Symbol, BoundSymbol, default_python_printer
from thunder.core.proxies import (
    CollectionProxy,
    TensorProxy,
    NumberProxy,
    is_proxyable,
    proxy,
    numberproxy,
    pytype,
    FutureTensorProxy,
)
import thunder.core.codeutils as codeutils
from thunder.core.codeutils import Printable
import thunder.core.utils as utils
import thunder.core.baseutils as baseutils
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes
from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
from thunder.core.trace import get_tracectx
from thunder.core.langctx import langctx

#
# Primitives and helpers for defining them
#


class PrimIDs(Enum):
    # Unpacking prims
    UNPACK_EMPTY_DICT = auto()
    UNPACK_KEY = auto()
    UNPACK_SEQUENCE = auto()
    UNPACK_TRIVIAL = auto()
    # TODO: UNPACK_SET
    # Utility prims
    COMMENT = auto()
    DEL = auto()
    PRINT = auto()
    RETURN = auto()
    # Data movement and transformation prims
    CONVERT_ELEMENT_TYPE = auto()
    DEVICE_PUT = auto()
    NUMPY_ARRAY_TO_TORCH_TENSOR = auto()
    # Tensor creation prims
    EXOGENOUS_LIKE = auto()
    FULL = auto()
    IOTA = auto()
    UNIFORM = auto()
    # Reshaping and permuting prims
    BROADCAST_IN_DIM = auto()
    CAT = auto()
    PAD = auto()
    RESHAPE = auto()
    SLICE = auto()
    SQUEEZE = auto()
    TRANSPOSE = auto()
    TAKE = auto()
    INDEX_ADD = auto()
    TAKE_ALONG_AXIS = auto()
    SCATTER_ADD = auto()
    VIEW = auto()
    # Memory layout prims (Experimental)
    STRIDE_ORDER = auto()
    # Elementwise unary prims
    PY_ABS = auto()
    ABS = auto()
    ACOS = auto()
    ACOSH = auto()
    ASIN = auto()
    ASINH = auto()
    ATAN = auto()
    ATANH = auto()
    BITWISE_NOT = auto()
    CEIL = auto()
    COS = auto()
    COSH = auto()
    ERF = auto()
    ERFC = auto()
    ERFCINV = auto()
    ERFINV = auto()
    EXP = auto()
    EXP2 = auto()
    EXPM1 = auto()
    FLOOR = auto()
    ISFINITE = auto()
    LGAMMA = auto()
    LOG = auto()
    LOG10 = auto()
    LOG1P = auto()
    LOG2 = auto()
    NDTRI = auto()
    NEG = auto()
    RECIPROCAL = auto()
    ROUND = auto()
    RSQRT = auto()
    SIGN = auto()
    SIGNBIT = auto()
    SIN = auto()
    SINH = auto()
    SQRT = auto()
    TAN = auto()
    TANH = auto()
    TRUNC = auto()
    # Elementwise binary prims
    ADD = auto()
    ATAN2 = auto()
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    DIV = auto()
    EQ = auto()
    FMOD = auto()
    GE = auto()
    GT = auto()
    LE = auto()
    LT = auto()
    MUL = auto()
    NE = auto()
    NEXTAFTER = auto()
    POW = auto()
    REMAINDER = auto()
    SUB = auto()
    # Conditional prims
    WHERE = auto()
    # Reduction prims
    AMAX = auto()
    AMIN = auto()
    PROD = auto()
    SUM = auto()
    VAR = auto()
    VAR_MEAN = auto()
    # Linear algebra prims (Mostly experimental)
    LINEAR = auto()
    MATMUL = auto()
    # NN prims (Experimental!)
    EMBEDDING = auto()
    EMBEDDING_BACKWARD = auto()
    # Distributed prims (Experimental!)
    ALL_REDUCE = auto()
    WAIT = auto()


# NOTE The primitive context is actually the lack of a context for interpreting operations
# TODO Maybe we should represent it as an actual ctx?
def prim_ctx(fn):
    _fn = langctx(None)(fn)
    return _fn


# TODO Document this function and describe the parts of a primitive
def make_prim(
    id,
    name,
    *,
    meta,
    python_printer=default_python_printer,
    python_impl=None,
    _bind_postprocess: None | Callable = None,
):
    sym = Symbol(
        name=name,
        meta=prim_ctx(meta),
        python_impl=python_impl,
        id=id,
        is_prim=True,
        python_printer=python_printer,
        _bind_postprocess=_bind_postprocess,
    )
    return sym


#
# Unpacking prims
#


def _collectify(x: Any, *, name: Optional[str] = None) -> Any:
    if baseutils.is_collection(x):
        return CollectionProxy(x, name=name)

    return x


def unpack_trivial_impl(x: Any, /, *, name: Optional[str] = None) -> Any:
    return x


def unpack_trivial_meta(x: Any, /, *, name: Optional[str] = None) -> Any:
    return _collectify(x, name=name)


def unpack_trivial_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str:
    utils.check(
        len(arg_printables) == 0,
        lambda: f"Expected zero arguments for unpack_trivial but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) <= 1,
        lambda: f"Expected at most one kwarg for unpack_trivial but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    result_str = "_" if bsym.output is None else f"{codeutils.prettyprint(out_printables, with_type=True)}"
    s = f"# {result_str} {'(unused)' if bsym.output is None else ''}"
    return s


# Removes the inputs from unpack_trivial, so it appears to have no input
def _unpack_trivial_bind_postprocess(bsym: BoundSymbol) -> None:
    bsym.args = ()


unpack_trivial = make_prim(
    PrimIDs.UNPACK_TRIVIAL,
    "unpack_trivial",
    meta=unpack_trivial_meta,
    python_printer=unpack_trivial_printer,
    python_impl=unpack_trivial_impl,
    _bind_postprocess=_unpack_trivial_bind_postprocess,
)


# TODO Restore const criteria
def unpack_sequence_meta(x: Union[Sequence, CollectionProxy], l: int) -> list:
    if isinstance(x, CollectionProxy):
        x = x.collection()

    utils.check_type(x, Sequence)
    utils.check_type(l, int)
    baseutils.check(len(x) == l, lambda x=x, l=l: f"Expected the length of {x=} to be {l=}")

    return list(_collectify(y) for y in x)


# TODO Review using multi-line unpacks more cleverly
# TODO Possibly put the length in the code to show the requirement
def unpack_sequence_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_sequence but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_sequence but got {kwarg_printables}",
        exception_type=AssertionError,
    )
    utils.check_type(bsym.output, Sequence)

    x, l = arg_printables
    call_str = f"{codeutils.prettyprint(x)}"

    # Short-circuits if there's nothing to unpack:
    if len(bsym.output) == 0:
        return f"# {call_str} (empty sequence)"

    lines = []
    for out in out_printables:
        line = f"{codeutils.prettyprint(out, literals_as_underscores=True)}, \\"
        lines.append(line)

    lines.append(f"= {call_str}")
    return lines


def unpack_sequence_impl(x: Sequence, l: int) -> list:
    return list(x)


unpack_sequence = make_prim(
    PrimIDs.UNPACK_SEQUENCE,
    "unpack_sequence",
    meta=unpack_sequence_meta,
    python_printer=unpack_sequence_printer,
    python_impl=unpack_sequence_impl,
)


def unpack_key_meta(d: Union[dict, CollectionProxy], key: Hashable) -> Any:
    if isinstance(d, CollectionProxy):
        d = d.collection()
    baseutils.check_type(d, dict)

    return _collectify(d[key])


def unpack_key_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_key but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_key but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    d, key = arg_printables
    dstr = codeutils.prettyprint(d)
    keystr = codeutils.prettyprint(key)
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {dstr}[{keystr}]"


def unpack_key_impl(d: dict, key: Hashable) -> Any:
    return d[key]


unpack_key = make_prim(
    PrimIDs.UNPACK_KEY,
    "unpack_key",
    meta=unpack_key_meta,
    python_printer=unpack_key_printer,
    python_impl=unpack_key_impl,
)


def unpack_empty_dict_meta(d: Union[dict, CollectionProxy]) -> tuple:
    baseutils.check_type(d, (dict, CollectionProxy))
    if isinstance(d, CollectionProxy):
        baseutils.check_type(d.collection(), dict)

    d = d if isinstance(d, dict) else d.collection()

    baseutils.check(len(d) == 0, lambda: f"unpack_empty_dict_meta expected an empty dict but received {d=}")
    return ()


def unpack_empty_dict_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected one argument for unpack_empty_dict but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_empty_dict but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    arg_str = codeutils.prettyprint(arg_printables[0])
    return f"# {arg_str} (empty dict)"


def unpack_empty_dict_impl(d: Union[dict, CollectionProxy]) -> tuple:
    assert len(d) == 0
    return ()


unpack_empty_dict = make_prim(
    PrimIDs.UNPACK_EMPTY_DICT,
    "unpack_empty_dict",
    meta=unpack_empty_dict_meta,
    python_printer=unpack_empty_dict_printer,
    python_impl=unpack_empty_dict_impl,
)


def unpack_dict(d: Union[dict, CollectionProxy]) -> tuple[Any, ...]:
    l = []

    baseutils.check_type(d, (dict, CollectionProxy))
    if isinstance(d, CollectionProxy):
        baseutils.check_type(d.collection(), dict)

    keys = d.keys if isinstance(d, dict) else d.collection().keys()

    # Short-circuits if the dict is empty
    # TODO We may want to make an explicit "unpack empty dict"
    if len(keys) == 0:
        return unpack_empty_dict(d)

    for k in keys:
        v = unpack_key(d, k)
        l.append(v)

    return tuple(l)


def unpack(x: Any) -> Any:
    if baseutils.is_collection(x) or isinstance(x, CollectionProxy):
        coll = x.collection() if isinstance(x, CollectionProxy) else x
        if isinstance(coll, Sequence):
            return unpack_sequence(x, len(coll))
        if isinstance(coll, dict):
            return unpack_dict(x)
        baseutils.check(False, lambda: f"unpack encountered an unsupported collection type {type(coll)}")

    return unpack_trivial(x)


#
# Utility prims
#


def _print_meta(x: Any) -> None:
    pass


def python_print_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str:
    utils.check(
        out_printables is None or len(out_printables) == 0,
        lambda: f"Expected no out printables when printing python_print, but got {out_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwarg printables when printing python_print, but got {kwarg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected only one arg printable when printing python_print, but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    return f"print({codeutils.prettyprint(arg_printables[0])})"


python_print = make_prim(
    PrimIDs.PRINT,
    "print",
    meta=_print_meta,
    python_printer=python_print_printer,
    python_impl=print,
)


def _comment_meta(s: str, /) -> None:
    return None


def comment_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str:
    utils.check(
        out_printables is None or len(out_printables) == 0,
        lambda: f"Expected no out printables when printing a comment, but got {out_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwarg printables when printing a comment, but got {kwarg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected only one arg printable when printing a comment, but got {arg_printables}",
        exception_type=AssertionError,
    )

    (s,) = arg_printables
    return f"# {s}"


comment = make_prim(
    PrimIDs.COMMENT,
    "comment",
    meta=_comment_meta,
    python_printer=comment_printer,
    python_impl=_comment_meta,
)


def _del_meta(*args):
    pass


def del_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for del but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    arg_string = ", ".join(codeutils.prettyprint(x, literals_allowed=False) for x in arg_printables)

    return f"del {arg_string}"


# NOTE This wrapper for del is necessary because python_impl=del is invalid syntax (del is not a regular function)
def _del_impl(x: Any) -> None:
    del x


python_del = make_prim(
    PrimIDs.DEL,
    "del",
    meta=_del_meta,
    python_printer=del_printer,
    python_impl=_del_impl,
)


def _return_meta(*args) -> Any:
    return args


def return_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for del but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    arg_str = (
        ""
        if (arg_printables is None or len(arg_printables) == 0)
        else ", ".join(codeutils.prettyprint(x) for x in arg_printables)
    )

    return f"return {arg_str}"


# NOTE This wrapper for del is necessary because python_impl=del is invalid syntax (del is not a regular function)
def _return_impl(*args) -> Any:
    return args


python_return = make_prim(
    PrimIDs.RETURN,
    "return",
    meta=_return_meta,
    python_printer=return_printer,
    python_impl=_return_impl,
)

#
# Data movement and transformation prims
#
# TODO create an expected type helper for consistent error formatting
# TODO: consider supporting number subclasses


# TODO Require the datatype of the conversion be constant
def _convert_element_type_meta(
    a: Union[TensorProxy, Number], dtype: Union[type, dtypes.dtype]
) -> Union[TensorProxy, NumberProxy, Number]:
    utils.check_type(a, (Number, TensorProxy))
    utils.check_type(dtype, (type, dtypes.dtype))

    # NOTE Python numbers are constants, and this will return another Python number when given one because
    #   The conversion is constant
    if isinstance(a, Number):
        utils.check(utils.is_numbertype(dtype), lambda: f"Trying to convert a number to non-numbertype object {dtype}")

        if isinstance(a, NumberProxy):
            return numberproxy(dtype, dtype(utils.get_numberlike_value(a)))

        number_result = dtype(a)
        return number_result

    return TensorProxy(like=a, dtype=dtype)


convert_element_type = make_prim(
    PrimIDs.CONVERT_ELEMENT_TYPE,
    "convert_element_type",
    meta=_convert_element_type_meta,
)


def _device_put_meta(a: TensorProxy, device: devices.Device) -> TensorProxy:
    # NOTE The TensorProxy constructor will validate that a is a TensorProxy
    #   and device is a devices.Device
    return TensorProxy(like=a, device=device)


device_put = make_prim(
    PrimIDs.DEVICE_PUT,
    "device_put",
    meta=_device_put_meta,
)


def _numpy_array_to_torch_tensor_meta(a: TensorProxy) -> TensorProxy:
    return TensorProxy(like=a)


numpy_array_to_torch_tensor = make_prim(
    PrimIDs.NUMPY_ARRAY_TO_TORCH_TENSOR,
    "numpy_array_to_torch_tensor",
    meta=_numpy_array_to_torch_tensor_meta,
)

#
# Helpers for elementwise primitive dtype handling
#


# NOTE Elementwise primitives always accept inputs with a common datatype, and they
#   usually produce an output with that same datatype (SAME).
#   Sometimes, however, elementwise operations can produce an output with a different
#   datatype than the inputs. For example, comparison operations like eq and lt always
#   produce boolean results (ALWAYS_BOOL), and other operations, like abs, map
#   complex numbers to floats (COMPLEX_TO_FLOAT).
#   The ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND enum describes these three behaviors so that
#   elementwise operations can rely on helper functions to implement this behavior.
class ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND(Enum):
    SAME = auto()
    ALWAYS_BOOL = auto()
    COMPLEX_TO_FLOAT = auto()


math_dtypes = dtypes.all_dtypes_and_numbertypes - dtypes.low_precision_dtypes
fp_math_dtypes = math_dtypes - dtypes.exact_dtypes
comparison_dtypes = dtypes.all_dtypes_and_numbertypes - dtypes.complex_dtypes

#
# Elementwise unary prims
#
# TODO Maybe make a helper to construct elementwise unary prims?


def _elementwise_unary_meta_factory(
    *,
    name,
    number_fn,
    supported_input_dtypes,
    output_dtype_kind,
    numbers_only: bool,
    number_type_map: Optional[dict[Type, Type]],
):
    def meta(a: Union[TensorProxy, Number]) -> Union[TensorProxy, Number]:
        # Checks that inputs have an expected type
        utils.check_type(a, (TensorProxy, Number))

        if isinstance(a, Number):
            # Checks that the numbertype is supported
            typ = utils.get_numberlike_type(a)
            val = utils.get_numberlike_value(a)

            allowed_types = number_type_map.keys() if number_type_map is not None else supported_input_dtypes

            utils.check(typ in allowed_types, lambda: f"Unsupported input dtype {typ}")

            output_type = None
            if number_type_map is not None:
                output_type = number_type_map[typ]
            elif output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME:
                output_type = typ
            elif output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL:
                output_type = bool
            elif output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.COMPLEX_TO_FLOAT:
                if dtypes.is_complex_dtype(typ):
                    output_type = float
                else:
                    output_type = typ
            else:
                utils.check(False, lambda: f"Unknown {output_dtype_kind=}")

            if val is None or number_fn is None:
                return numberproxy(output_typ, None)

            value = number_fn(a)
            result = numberproxy(type(value), value)
            utils.check(
                type(value) is output_type,
                lambda: f"Unexpected number output type {type(value)}, expected {output_type}, for input type {typ} (value={val})",
            )
            return result

        # NOTE a is a TensorProxy
        utils.check(
            not numbers_only,
            lambda: f"Trying to call a primitive ({name}) that only supports numbers with a tensor input",
        )

        # Checks that dtype is supported
        utils.check(a.dtype in supported_input_dtypes, lambda: f"Unsupported input dtype {a.dtype}")

        if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME:
            return TensorProxy(like=a)
        if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL:
            return TensorProxy(like=a, dtype=dtypes.bool8)
        if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.COMPLEX_TO_FLOAT:
            if dtypes.is_complex_dtype(a.dtype):
                return TensorProxy(like=a, dtype=dtypes.corresponding_real_dtype(a.true_dtype))
            return TensorProxy(like=a)

        utils.check(False, lambda: f"Unknown {output_dtype_kind=}", exception_type=AssertionError)

    return meta


def _make_elementwise_unary_prim(
    id: PrimIDs,
    name: str,
    number_fn: Optional[Callable] = None,
    python_printer: Callable = default_python_printer,
    supported_input_dtypes=dtypes.all_dtypes_and_numbertypes,
    output_dtype_kind: ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND = ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME,
    numbers_only: bool = False,
    number_type_map: Optional[dict[Type, Type]] = None,
):
    return make_prim(
        id,
        name,
        meta=_elementwise_unary_meta_factory(
            name=name,
            number_fn=number_fn,
            supported_input_dtypes=supported_input_dtypes,
            output_dtype_kind=output_dtype_kind,
            numbers_only=numbers_only,
            number_type_map=number_type_map,
        ),
        python_printer=python_printer,
    )


py_abs = _make_elementwise_unary_prim(
    PrimIDs.PY_ABS,
    "py_abs",
    number_fn=operator.abs,
    numbers_only=True,
    number_type_map={
        bool: int,
        int: int,
        float: float,
        complex: float,
    },
)

abs = _make_elementwise_unary_prim(
    PrimIDs.ABS,
    "abs",
    number_fn=operator.abs,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.COMPLEX_TO_FLOAT,
)

acos = _make_elementwise_unary_prim(
    PrimIDs.ACOS,
    "acos",
    supported_input_dtypes=fp_math_dtypes,
)

acosh = _make_elementwise_unary_prim(PrimIDs.ACOSH, "acosh", supported_input_dtypes=fp_math_dtypes)

asin = _make_elementwise_unary_prim(
    PrimIDs.ASIN,
    "asin",
    supported_input_dtypes=fp_math_dtypes,
)

asinh = _make_elementwise_unary_prim(
    PrimIDs.ASINH,
    "asinh",
    supported_input_dtypes=fp_math_dtypes,
)

atan = _make_elementwise_unary_prim(
    PrimIDs.ATAN,
    "atan",
    supported_input_dtypes=fp_math_dtypes,
)

atanh = _make_elementwise_unary_prim(
    PrimIDs.ATANH,
    "atanh",
    supported_input_dtypes=fp_math_dtypes,
)

bitwise_not = _make_elementwise_unary_prim(
    PrimIDs.BITWISE_NOT,
    "bitwise_not",
    number_fn=operator.inv,
    supported_input_dtypes=dtypes.exact_dtypes,
)

# TODO Should ceil accept float16 and bfloat16 types?
# NOTE This primitive preserves the input's dtype for tensors
#   but returns numbers as integers to be consistent with
#   Python's math.ceil
ceil = _make_elementwise_unary_prim(
    PrimIDs.CEIL,
    "ceil",
    number_fn=math.ceil,
    supported_input_dtypes=dtypes.float_dtypes,
)

cos = _make_elementwise_unary_prim(
    PrimIDs.COS,
    "cos",
    supported_input_dtypes=fp_math_dtypes,
)

cosh = _make_elementwise_unary_prim(
    PrimIDs.COSH,
    "cosh",
    supported_input_dtypes=fp_math_dtypes,
)

erf = _make_elementwise_unary_prim(
    PrimIDs.ERF,
    "erf",
    supported_input_dtypes=fp_math_dtypes,
)

erfc = _make_elementwise_unary_prim(
    PrimIDs.ERFC,
    "erfc",
    supported_input_dtypes=fp_math_dtypes,
)

erfcinv = _make_elementwise_unary_prim(
    PrimIDs.ERFCINV,
    "erfcinv",
    supported_input_dtypes=fp_math_dtypes,
)

erfinv = _make_elementwise_unary_prim(
    PrimIDs.ERFINV,
    "erfinv",
    supported_input_dtypes=fp_math_dtypes,
)

exp = _make_elementwise_unary_prim(
    PrimIDs.EXP,
    "exp",
    supported_input_dtypes=fp_math_dtypes,
)

exp2 = _make_elementwise_unary_prim(
    PrimIDs.EXP2,
    "exp2",
    supported_input_dtypes=fp_math_dtypes,
)

expm1 = _make_elementwise_unary_prim(
    PrimIDs.EXPM1,
    "expm1",
    supported_input_dtypes=fp_math_dtypes,
)

# TODO Should floor accept float16 and bfloat16 types?
# NOTE This preserves the input's dtype for tensors, but is consistent
#   with math.floor for numbers (always returning an integer)
floor = _make_elementwise_unary_prim(
    PrimIDs.FLOOR,
    "floor",
    number_fn=math.floor,
    supported_input_dtypes=dtypes.float_dtypes,
)

isfinite = _make_elementwise_unary_prim(
    PrimIDs.ISFINITE,
    "isfinite",
    supported_input_dtypes=dtypes.inexact_dtypes,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
)

lgamma = _make_elementwise_unary_prim(
    PrimIDs.LGAMMA,
    "lgamma",
    supported_input_dtypes=fp_math_dtypes,
)

log = _make_elementwise_unary_prim(
    PrimIDs.LOG,
    "log",
    supported_input_dtypes=fp_math_dtypes,
)

log10 = _make_elementwise_unary_prim(
    PrimIDs.LOG10,
    "log10",
    supported_input_dtypes=fp_math_dtypes,
)

log1p = _make_elementwise_unary_prim(
    PrimIDs.LOG1P,
    "log1p",
    supported_input_dtypes=fp_math_dtypes,
)

log2 = _make_elementwise_unary_prim(
    PrimIDs.LOG2,
    "log2",
    supported_input_dtypes=fp_math_dtypes,
)

ndtri = _make_elementwise_unary_prim(
    PrimIDs.NDTRI,
    "ndtri",
    supported_input_dtypes=fp_math_dtypes,
)

neg = _make_elementwise_unary_prim(
    PrimIDs.NEG,
    "neg",
    number_fn=operator.neg,
)

reciprocal = _make_elementwise_unary_prim(
    PrimIDs.RECIPROCAL,
    "reciprocal",
    supported_input_dtypes=fp_math_dtypes,
)

# Rounds to nearest even
# NOTE This round produces an output with the same dtype as its input
round = _make_elementwise_unary_prim(
    PrimIDs.ROUND,
    "round",
    number_fn=builtins.round,
    supported_input_dtypes=fp_math_dtypes,
)

rsqrt = _make_elementwise_unary_prim(
    PrimIDs.RSQRT,
    "rsqrt",
    supported_input_dtypes=fp_math_dtypes,
)

# NOTE jax.lax.sign and torch.sgn differ from numpy.sign in complex support
#       nump.sign: x / sqrt(x * x)
#       jax.lax.sign & torch.sgn: x / abs(x)
# NOTE PyTorch's sign and sgn differ in that sgn includes complex support
# NOTE This follows the convention of jax.lax.sign and torch.sgn
# jax.lax.sign: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.sign.html
# numpy.sign: https://numpy.org/doc/stable/reference/generated/numpy.sign.html
# torch.sgn: https://pytorch.org/docs/stable/generated/torch.sgn.html
# torch.sign: https://pytorch.org/docs/stable/generated/torch.sign.html
sign = _make_elementwise_unary_prim(
    PrimIDs.SIGN,
    "sign",
)


def _signbit_number(a: Number) -> bool:
    return a < 0


signbit = _make_elementwise_unary_prim(
    PrimIDs.SIGNBIT,
    "signbit",
    number_fn=_signbit_number,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
)

sin = _make_elementwise_unary_prim(
    PrimIDs.SIN,
    "sin",
    supported_input_dtypes=fp_math_dtypes,
)

sinh = _make_elementwise_unary_prim(
    PrimIDs.SINH,
    "sinh",
    supported_input_dtypes=fp_math_dtypes,
)

sqrt = _make_elementwise_unary_prim(
    PrimIDs.SQRT,
    "sqrt",
    supported_input_dtypes=fp_math_dtypes,
)

tan = _make_elementwise_unary_prim(
    PrimIDs.TAN,
    "tan",
    supported_input_dtypes=fp_math_dtypes,
)

tanh = _make_elementwise_unary_prim(
    PrimIDs.TANH,
    "tanh",
    supported_input_dtypes=fp_math_dtypes,
)

# NOTE This trunc preserves the dtype of its input
trunc = _make_elementwise_unary_prim(
    PrimIDs.TRUNC,
    "trunc",
    supported_input_dtypes=fp_math_dtypes,
    number_fn=math.trunc,
)

#
# Elementwise binary prims
#


# TODO add stride logic
# TODO Improve error messages for mismatched dtypes (using an error context)
def _elementwise_binary_meta_factory(
    *,
    name,
    number_fn,
    output_dtype_kind,
    supported_input_dtypes,
):
    def meta(
        a: Union[TensorProxy, Number],
        b: Union[TensorProxy, Number],
    ) -> Union[TensorProxy, Number]:
        # Checks that inputs have an expected type
        utils.check_type(a, (TensorProxy, Number))
        utils.check_type(b, (TensorProxy, Number))

        # Checks same dtype
        numbertype, dtype = utils.check_same_dtype(a, b)

        # Checks that dtype is supported
        utils.check(
            numbertype is None or numbertype in supported_input_dtypes, lambda: f"Unsupported number type {numbertype}"
        )
        utils.check(dtype is None or dtype in supported_input_dtypes, lambda: f"Unsupported input dtype {dtype}")

        # Special-cases number x number inputs
        if isinstance(a, Number) and isinstance(b, Number):
            aval, bval = utils.get_numberlike_value(a), utils.get_numberlike_value(b)

            # Handles the case where a number has an indeterminate value, or the operation has
            #   no number handler, by returning another indeterminate value
            if aval is None or bval is None or number_fn is None:
                return numberproxy(numbertype, None)

            value = number_fn(aval, bval)
            return numberproxy(type(value), value)

        # Checks same shape
        # NOTE: this doesn't verify a common shape if one or more inputs is a number
        utils.check_same_shape(a, b)

        # Checks same device
        utils.check_same_device(a, b)

        tensor = a if isinstance(a, TensorProxy) else b
        requires_grad = (isinstance(a, TensorProxy) and a.requires_grad) or (
            isinstance(b, TensorProxy) and b.requires_grad
        )

        if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME:
            # NOTE that this is not just like=tensor, because one tensor could have a weak dtype
            #   and the other a strong dtype, and these are the "same"
            return TensorProxy(like=tensor, dtype=dtype, requires_grad=requires_grad)
        if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL:
            return TensorProxy(like=tensor, dtype=dtypes.bool8, requires_grad=requires_grad)
        if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.COMPLEX_TO_FLOAT and dtypes.is_complex_dtype(dtype):
            return TensorProxy(like=tensor, dtype=dtypes.corresponding_real_dtype(dtype), requires_grad=requires_grad)

        raise AssertionError(f"Unknown {output_dtype_kind=}")

    return meta


# number_fn should be a function that handles Number x Number inputs,
#   it should only depend on Python and standard Python libraries
# torch_fn should be a PyTorch operation or composition of PyTorch
#   operations that implements the primitive
# TODO Maybe wrap number number functions in the prim ctx?
def _make_elementwise_binary_prim(
    id,
    name,
    number_fn=None,
    python_printer=default_python_printer,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME,
    supported_input_dtypes=dtypes.all_dtypes_and_numbertypes,
):
    return make_prim(
        id,
        name,
        meta=_elementwise_binary_meta_factory(
            name=name,
            number_fn=number_fn,
            output_dtype_kind=output_dtype_kind,
            supported_input_dtypes=supported_input_dtypes,
        ),
        python_printer=python_printer,
    )


add = _make_elementwise_binary_prim(
    PrimIDs.ADD,
    "add",
    number_fn=operator.add,
    supported_input_dtypes=math_dtypes,
)

atan2 = _make_elementwise_binary_prim(
    PrimIDs.ATAN2,
    "atan2",
    number_fn=math.atan2,
    supported_input_dtypes=fp_math_dtypes,
)

bitwise_and = _make_elementwise_binary_prim(
    PrimIDs.BITWISE_AND,
    "bitwise_and",
    number_fn=operator.and_,
    supported_input_dtypes=dtypes.exact_dtypes,
)

bitwise_or = _make_elementwise_binary_prim(
    PrimIDs.BITWISE_OR,
    "bitwise_or",
    number_fn=operator.or_,
    supported_input_dtypes=dtypes.exact_dtypes,
)

bitwise_xor = _make_elementwise_binary_prim(
    PrimIDs.BITWISE_XOR,
    "bitwise_xor",
    number_fn=operator.xor,
    supported_input_dtypes=dtypes.exact_dtypes,
)


def _div_numbers(a: Number, b: Number) -> Number:
    if dtypes.is_exact_dtype(type(a)) and dtypes.is_exact_dtype(type(b)):
        # Accounts for rounding towards zero instead of flooring
        if (a >= 0) != (b >= 0) and a % b:
            return a // b + 1
        else:
            return a // b

    return a / b


# NOTE This div is defined as equivalent to C-style division,
#   which is true division when computed for floating inputs
#   and rtz division for exact inputs
# See https://en.cppreference.com/w/cpp/numeric/math/div
div = _make_elementwise_binary_prim(
    PrimIDs.DIV,
    "div",
    number_fn=_div_numbers,
    supported_input_dtypes=math_dtypes,
)

eq = _make_elementwise_binary_prim(
    PrimIDs.EQ,
    "eq",
    number_fn=operator.eq,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
)

# NOTE fmod vs remainder
#   fmod is defined as fmod(a, b) == a - trunc_div(a, b) * b
#   It is the complement to truncation dividing a and b, unlike
#   remainder, which is the complement to floor dividing a and b
# NOTE This definition of fmod is consistent with C++'s std::fmod
#   See https://en.cppreference.com/w/cpp/numeric/math/fmod
# NOTE This definition of fmod is consistent with Python's
#   See https://docs.python.org/3/library/math.html#math.fmod
fmod = _make_elementwise_binary_prim(
    PrimIDs.FMOD,
    "fmod",
    number_fn=math.fmod,
    supported_input_dtypes=math_dtypes,
)

ge = _make_elementwise_binary_prim(
    PrimIDs.GE,
    "ge",
    number_fn=operator.ge,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
    supported_input_dtypes=comparison_dtypes,
)

gt = _make_elementwise_binary_prim(
    PrimIDs.GT,
    "gt",
    number_fn=operator.gt,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
    supported_input_dtypes=comparison_dtypes,
)

le = _make_elementwise_binary_prim(
    PrimIDs.LE,
    "le",
    number_fn=operator.le,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
    supported_input_dtypes=comparison_dtypes,
)

lt = _make_elementwise_binary_prim(
    PrimIDs.LT,
    "lt",
    number_fn=operator.lt,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
    supported_input_dtypes=comparison_dtypes,
)

mul = _make_elementwise_binary_prim(
    PrimIDs.MUL,
    "mul",
    number_fn=operator.mul,
    supported_input_dtypes=math_dtypes,
)

ne = _make_elementwise_binary_prim(
    PrimIDs.NE,
    "ne",
    number_fn=operator.ne,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
)

# TODO Review supported dtypes
nextafter = _make_elementwise_binary_prim(
    PrimIDs.NEXTAFTER,
    "nextafter",
    supported_input_dtypes=comparison_dtypes,
)

pow = _make_elementwise_binary_prim(
    PrimIDs.POW,
    "pow",
    number_fn=operator.pow,
    supported_input_dtypes=math_dtypes,
)


# NOTE What remainder is this?
#   There are several "remainder" functions of interest.
#   PyTorch's torch.remainder (https://pytorch.org/docs/master/generated/torch.remainder.html),
#       which is defined a remainder(a, b) = a - (a // b) * b
#   PyTorch's prims.remainder, which is defined the same way
#   NumPy's numpy.remainder (https://numpy.org/doc/stable/reference/generated/numpy.remainder.html),
#       which is defined as equivalent to Python's modulus operation and is the
#       remainder complement to floor divide
#   Python's math.remainder (https://docs.python.org/3/library/math.html#math.remainder),
#       which is the IEEE 754-style remainder x - n*y where n = rtne(x/y) (rtne being "round to nearest even")
#       when x and y are finite and y is nonzero. math.remainder complements rtne(x/y).
#   JAX's lax.rem (https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.rem.html),
#       which is defined as x mod y with the sign from the dividend and the absolute value
#       always being less than the divisor's absolute value. This is NOT consistent
#       with torch.remainder, numpy.remainder, math.remainder or Python's modulus operator.
# This prim is defined as equivalent to Python's modulus operator, and is consistent with
#   torch.remainder and numpy.remainder.
remainder = _make_elementwise_binary_prim(
    PrimIDs.REMAINDER,
    "remainder",
    number_fn=operator.mod,
    supported_input_dtypes=math_dtypes,
)

sub = _make_elementwise_binary_prim(
    PrimIDs.SUB,
    "sub",
    number_fn=operator.sub,
    supported_input_dtypes=math_dtypes,
)

#
# Conditional prims
#


# TODO Restore Number x Number x Number support
def _where_meta(pred: Number | TensorProxy, a: Number | TensorProxy, b: Number | TensorProxy) -> TensorProxy:
    # Checks types
    # NOTE pred must be a bool tensor or bool (this is checked later)
    utils.check_type(pred, (TensorProxy, Number))
    utils.check_type(a, (TensorProxy, Number))
    utils.check_type(b, (TensorProxy, Number))

    if isinstance(pred, Number) and isinstance(a, Number) and isinstance(b, Number):
        raise NotImplementedError

    # Checks pred dtype (bool or bool tensor)
    if isinstance(pred, Number):
        utils.check(
            pytype(pred) is bool,
            lambda: f"Expected pred to be a boolean number, but found a number of type {pytype(pred)}",
        )

    if isinstance(pred, TensorProxy):
        utils.check(
            pred.dtype is dtypes.bool8,
            lambda: f"Expected pred to be a tensor with dtype bool, but found dtype {pred.dtype}",
        )

    # Checks devices and determines result device
    utils.check_same_device(pred, a, b)
    resultdevice = devices.cpu
    devices_ = tuple(x.device for x in (pred, a, b) if isinstance(x, TensorProxy))
    if len(devices_) > 0:
        resultdevice = devices_[0]

    # Determines result dtype
    numbertype, tensordtype = utils.check_same_dtype(a, b)
    dtype = tensordtype if tensordtype is not None else numbertype

    # Checks shapes
    utils.check_same_shape(pred, a, b)

    # Determines output shape
    # NOTE Assumes at least one of pred, a, and b is a TensorProxy because of prior check for Number x Number x Number
    shapes = tuple(x.shape for x in (pred, a, b) if isinstance(x, TensorProxy))
    resultshape = shapes[0]

    requires_grad = (isinstance(a, TensorProxy) and a.requires_grad) or (isinstance(b, TensorProxy) and b.requires_grad)
    return TensorProxy(shape=resultshape, device=resultdevice, dtype=dtype, requires_grad=requires_grad)


where = make_prim(
    PrimIDs.WHERE,
    "where",
    meta=_where_meta,
)

#
# Tensor creation prims
#
# TODO Add some architecture for constructing tensor creation prims


# NOTE exogenous_like is an "intermediate" primitive intended to exist only while modifying or optimizing a
#   program. Its first intended use was to construct separate forward -> backward traces suitable
#   for use with PyTorch's autograd. The grad values in these traces are introduced as
#   "exogenous" values, and then later transforms remove the "exogenous" introductions and
#   separate each trace into its own function.
#   Because exogenous_like takes all outputs requiring grad as inputs and produces the grad representatives,
#   it cannot be reordered to locations where the trace could not be split between forward and backward.
# NOTE Why not just insert zeros_like calls into the trace?
#   This is a reasonable option as of this writing, but there are two issues with this:
#   1) Clarity. A developer would have to know these zeros_like calls were intended to model the
#       introduction of exogenous values, and practitioners would have to understand that while reviewing
#       traces. One option to improve that clarity would be to augment bound symbols with more comments
#       about their derivation.
#   2) Preventing optimizations. We don't want executors to try and fuse exogenous values, and we don't
#       want to optimize traces by assuming their values (for example, we might in the future add special
#       logic to optimize zerotensors).
#   3) The way the zeros_like calls get flattened could break it apart, but we want this primitive to atomically
#       produce multiple grads from multiple inputs.
def _exogenous_like_meta(likes: Sequence[TensorProxy], /) -> tuple[TensorProxy]:
    # NOTE Inputs are validated by the TensorProxy constructor

    return tuple([TensorProxy(like=x) for x in likes])


exogenous_like = make_prim(
    PrimIDs.EXOGENOUS_LIKE,
    "exogenous_like",
    meta=_exogenous_like_meta,
)


# TODO Review always setting requires_grad=False
#   Logically these tensors are constructed intermediate to a trace, so there's no mechanism for a user to
#   extract their grad, but we could support compiling forward and backward and accessing grad attributes
#   in the future
def _full_meta(shape: Sequence[int], fill_value: Number, *, device: devices.Device, dtype: dtypes.dtype):
    # Checks inputs
    utils.check_type(fill_value, Number)

    # Ensures the requested fill_value can be safely cast to the dtype
    fill_value_dtype = dtypes.to_dtype(fill_value)
    utils.check(
        utils.can_safe_cast_number_to(fill_value, fill_value_dtype),
        lambda: f"Can't safely cast fill_value of numbertype {fill_value_dtype} to dtype {dtype}",
    )

    return TensorProxy(shape=shape, device=device, dtype=dtype, requires_grad=False)


full = make_prim(
    PrimIDs.FULL,
    "full",
    meta=_full_meta,
)


def _iota_meta(
    length: Number, *, start: Number, step: Number, device: devices.Device, dtype: dtypes.dtype
) -> TensorProxy:
    # Checks types
    # NOTE that device and dtype types will be checked by TensorProxy, below
    utils.check_type(length, Number)
    utils.check_type(start, Number)
    utils.check_type(step, Number)

    # Checks input properties
    utils.check(utils.is_exact_dtype(dtype), lambda: f"dtype={dtype} was not an exact dtype")
    utils.check(not utils.is_boolean_dtype(dtype), lambda: f"dtype={dtype} was not a non-boolean dtype")
    utils.check(length >= 0, lambda: f"length={length} was not weakly positive")

    shape = () if length == 0 else (length,)

    return TensorProxy(shape=shape, device=device, dtype=dtype, requires_grad=False)


iota = make_prim(PrimIDs.IOTA, "iota", meta=_iota_meta)


# TODO Should the uniform prim include minval maxval or always be [0, 1)?
# TODO Review always setting requires_grad=False
#   Logically these tensors are constructed intermediate to a trace, so there's no mechanism for a user to
#   extract their grad, but we could support compiling forward and backward and accessing grad attributes
#   in the future
def _uniform_meta(
    shape: Sequence[int], minval: Number, maxval: Number, *, device: devices.Device, dtype: dtypes.dtype
) -> TensorProxy:
    # Checks inputs
    utils.check_type(minval, Number)
    utils.check_type(maxval, Number)
    utils.check_type(device, devices.Device)
    utils.check_type(dtype, dtypes.dtype)

    return TensorProxy(shape=shape, device=device, dtype=dtype, requires_grad=False)


uniform = make_prim(
    PrimIDs.UNIFORM,
    "uniform",
    meta=_uniform_meta,
)

#
# Shape prims
#


# NOTE broadcast_dimensions is a sequence with length equal to a.shape (which is not necessarily equal to shape)
def broadcast_in_dim_meta(a: TensorProxy, shape: Sequence[int], broadcast_dimensions: Sequence[int]) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(shape, Sequence)
    utils.check_type(broadcast_dimensions, Sequence)

    utils.check(
        len(a.shape) == len(broadcast_dimensions),
        lambda: f"Expected one broadcast dimension (broadcast_dimensions={broadcast_dimensions}) for each dimension of a={a.shape}",
    )

    # Checks that dimensions are strictly increasing and valid
    prev_idx = -1
    for original_length, idx in zip(a.shape, broadcast_dimensions):
        utils.check(
            idx > prev_idx,
            lambda: f"Expected the dimensions in broadcast_dimensions={broadcast_dimensions} to be strictly increasing",
        )
        prev_idx = idx

        utils.check(
            idx < len(shape),
            lambda: f"One of the broadcast_dimensions={broadcast_dimensions} was {idx}, which is out-of-bounds for a tensor with {len(shape)} dimensions",
        )
        utils.check(
            original_length == 1 or shape[idx] == original_length,
            lambda: f"A dimension of length {original_length} cannot be broadcast to a dimension of length {shape[idx]}",
        )

    return TensorProxy(like=a, shape=shape)


broadcast_in_dim = make_prim(
    PrimIDs.BROADCAST_IN_DIM,
    "broadcast_in_dim",
    meta=broadcast_in_dim_meta,
)


def cat_meta(tensors: list[TensorProxy], dim: int):
    utils.check(len(tensors) > 0, lambda: "Cat expects a non-empty list of tensors")
    utils.check_same_device(*tensors)
    utils.check_same_dtype(*tensors)

    ndim = tensors[0].ndim
    utils.check(
        dim >= -ndim and dim < ndim,
        lambda: f"Expected dimension in inclusive range of {-ndim} and {ndim-1}: got {dim}.",
        IndexError,
    )

    dim = utils.canonicalize_dim_idx(ndim, dim)

    shape = [s for s in tensors[0].shape]
    for i, ai in enumerate(tensors[1:]):
        utils.check(
            isinstance(ai, TensorProxy),
            lambda: f"First argument to cat must be a list of tensors: found type {type(ai)}",
        )
        utils.check(
            ai.ndim == ndim,
            lambda: f"Attempted to concatenate tensors of different dimension: got {ndim} and {ai.ndim}",
        )
        for d, (sd, sad) in enumerate(zip(shape, ai.shape)):
            utils.check(
                sd == sad or d == dim,
                lambda: f"Sizes of tensors must match except in dimension {dim}. "
                f"Expected size {sd} but got size {sad} for tensor number {i+1} in the list.",
            )
        shape[dim] += ai.shape[dim]

    requires_grad = any(list([t.requires_grad for t in tensors]))
    return TensorProxy(like=tensors[0], shape=shape, requires_grad=requires_grad)


cat = make_prim(
    PrimIDs.CAT,
    "cat",
    meta=cat_meta,
)


def pad_meta(a: TensorProxy, padding_value: Number, padding_config: Sequence[tuple[int, int, int]]) -> TensorProxy:
    # Validates types
    utils.check_type(a, TensorProxy)
    utils.check_type(padding_value, Number)
    utils.check_type(padding_config, Sequence)

    # Validates input properties
    utils.check(a.ndim == len(padding_config), lambda: f"Expected {a.ndim=} to equal {len(padding_config)=}")
    utils.check_same_dtype(a, padding_value)

    shape = []
    for l, (lo, hi, dilation) in zip(a.shape, padding_config):
        utils.check(dilation >= 0, lambda: f"Expected {dilation=} to be weakly positive")
        final_length = l + max(0, l - 1) * dilation + lo + hi
        utils.check(final_length >= 0, lambda: f"The length of a dimension after padding would be {final_length=} < 0")
        shape.append(final_length)

    return TensorProxy(like=a, shape=shape)


pad = make_prim(
    PrimIDs.PAD,
    "pad",
    meta=pad_meta,
)


def reshape_meta(a: TensorProxy, shape: Sequence[int]) -> TensorProxy:
    # Validates inputs
    utils.check_type(a, TensorProxy)
    utils.check_valid_shape(shape)

    numel = reduce(operator.mul, shape, 1)
    utils.check(
        numel == a.numel,
        lambda: f"Attempting to reshape a.shape={a.shape} to shape={shape}, but a.numel={a.numel} is different from the number of elements in shape, {numel}",
    )

    return TensorProxy(like=a, shape=shape)


reshape = make_prim(
    PrimIDs.RESHAPE,
    "reshape",
    meta=reshape_meta,
)


# TODO Be clear about what the prim can handle and what it can't
# TODO Validate that start_indices, end_indices, and strides are sequences of ints
# TODO Update the prim to not accept optional strides
# NOTE The stride parameter here refers to the stride of the slice, not the tensor's strides
def slice_meta(
    a: TensorProxy, start_indices: Sequence[int], end_indices: Sequence[int], strides: Optional[Sequence[int]] = None
) -> TensorProxy:
    if strides is None:
        strides = [1] * a.ndim

    # Checks types
    utils.check_type(a, TensorProxy)
    utils.check_type(start_indices, Sequence)
    utils.check_type(end_indices, Sequence)

    # NOTE This check doesn't use check type to inform callers that None is valid, too
    utils.check(isinstance(strides, Sequence), lambda: f"Expected strides={strides} to be None or a Sequence")

    # Checks all same length
    utils.check(
        a.ndim == len(start_indices) == len(end_indices) == len(strides),
        lambda: f"Expected the tensor's rank ({a.ndim}) to be equal to the length of start_indices ({len(start_indices)}), the length of end_indices ({len(end_indices)}), and the length of strides ({len(strides)})",
    )

    # Validates start, end, and stride values, and computes the new shape
    new_shape = []
    for start, stop, shape, stride in zip(start_indices, end_indices, a.shape, strides):
        utils.check(
            start >= 0, lambda: f"Expected all the indices in start_indices={start_indices} to be weakly positive!"
        )
        utils.check(
            start <= shape,
            lambda: f"Expected all the indices in start_indices={start_indices} to be weakly less than the length of the corresponding dimension in a.shape={a.shape}",
        )
        utils.check(
            start <= stop,
            lambda: f"Expected all the indices in start_indices={start_indices} to be weakly less than the indices in end_indices={end_indices}",
        )
        utils.check(
            stop <= shape,
            lambda: f"Expected all the indices in end_indices={end_indices} to be weakly less than the length of the corresponding dimension in a.shape={a.shape}",
        )
        utils.check(stride >= 1, lambda: f"Expected all the strides in strides={strides} to be strictly positive!")

        new_shape.append(math.floor((stop - start) / stride))

    return TensorProxy(like=a, shape=new_shape)


# NOTE: slice is named "slice_prim" and not "slice" because it conflicts with Python's "slice" builtin
slice_prim = make_prim(PrimIDs.SLICE, "slice", meta=slice_meta)


def squeeze_meta(a: TensorProxy, dims: Sequence[int]) -> TensorProxy:
    # Checks types
    utils.check_type(a, TensorProxy)
    utils.check_type(dims, Sequence)

    # Checks that no dims are redundant
    utils.check_no_duplicates(dims)

    # Checks that dims are valid
    for x in dims:
        utils.check(
            x >= 0 and x < len(a.shape), lambda: f"dims={dims} contained an invalid dimension {x} for a.shape={a.shape}"
        )

    shape = []
    for idx, l in enumerate(a.shape):
        # Checks that squeezed dims have length one
        if idx in dims:
            utils.check(l == 1, lambda: f"Cannot squeeze dimension {idx} of length {l} in a.shape={a.shape}")
            continue

        shape.append(l)

    return TensorProxy(like=a, shape=shape)


squeeze = make_prim(PrimIDs.SQUEEZE, "squeeze", meta=squeeze_meta)


def take_meta(a: TensorProxy, index: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(dim, int)
    utils.check_same_device(a, index)
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"index dtype={index.dtype} was not an integer dtype")
    utils.check(index.ndim <= 1, lambda: f"Expected index to a 1-D or 0-D tensor, but index.ndim={index.ndim}!")
    utils.validate_idx(a.ndim, dim)

    l = index.shape[0] if index.ndim == 1 else 1
    new_shape = a.shape[:dim] + (l,) + a.shape[dim + 1 :]

    return TensorProxy(like=a, shape=new_shape)


take = make_prim(PrimIDs.TAKE, "take", meta=take_meta)


def index_add_meta(a: TensorProxy, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(value, TensorProxy)
    utils.check_type(dim, int)
    utils.check_same_device(a, index, value)
    utils.check_same_dtype(a, value)
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"index dtype={index.dtype} was not an integer dtype")
    utils.check(
        value.ndim == a.ndim, lambda: f"Expected index (rank={value.ndim}) to have the same rank as a (rank={a.ndim})"
    )
    utils.check(index.ndim <= 1, lambda: f"Expected index to a 1-D or 0-D tensor, but index.ndim={index.ndim}!")
    utils.validate_idx(a.ndim, dim)
    utils.check(
        index.numel == value.shape[dim],
        lambda: f"Expected index={index} to have size equal to value.shape[dim]={value.shape[dim]}!",
    )

    utils.check(
        utils.same_shape(a.shape[:dim] + a.shape[dim + 1 :], value.shape[:dim] + value.shape[dim + 1 :]),
        lambda: f"Expected the all dimensions of a ({a.shape}) and value ({value.shape}) to be the same, except for dim ({dim})",
    )

    return TensorProxy(like=a)


index_add = make_prim(PrimIDs.INDEX_ADD, "index_add", meta=index_add_meta)


def take_along_axis_meta(a: TensorProxy, index: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(dim, int)
    utils.check_same_device(a, index)
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"index dtype={dtype} was not an integer dtype")
    utils.check(
        index.ndim == a.ndim, lambda: f"Expected index (rank={index.ndim}) to have the same rank as a (rank={a.ndim})"
    )
    utils.validate_idx(a.ndim, dim)

    utils.check(
        utils.same_shape(a.shape[:dim] + a.shape[dim + 1 :], index.shape[:dim] + index.shape[dim + 1 :]),
        lambda: f"Expected the all dimensions of a ({a.shape}) and index ({index.shape}) to be the same, except for dim ({dim})",
    )

    return TensorProxy(like=a, shape=index.shape)


take_along_axis = make_prim(PrimIDs.TAKE_ALONG_AXIS, "take_along_axis", meta=take_along_axis_meta)


def scatter_add_meta(a: TensorProxy, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(value, TensorProxy)
    utils.check_type(dim, int)
    utils.check_same_device(a, index, value)
    utils.check_same_dtype(a, value)
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"index dtype={index.dtype} was not an integer dtype")
    utils.check(
        index.ndim == a.ndim, lambda: f"Expected index (rank={index.ndim}) to have the same rank as a (rank={a.ndim})"
    )
    utils.check(
        index.ndim == value.ndim,
        lambda: f"Expected index (rank={index.ndim}) to have the same rank as value (rank={value.ndim})",
    )
    utils.validate_idx(a.ndim, dim)

    for idx, l in enumerate(index.shape):
        if idx != dim:
            utils.check(
                index.shape[idx] <= a.shape[idx],
                lambda: f"Expected 'index' size on all dimensions to be <= 'a', except `dim`. Found dim {idx}, where 'index' has {index.shape[idx]} and 'a' has {a.shape[idx]}",
            )
        utils.check(
            index.shape[idx] <= value.shape[idx],
            lambda: f"Expected 'index' size on all dimensions to be <= 'value'. Found dim {idx}, where 'index' has {index.shape[idx]} and 'value' has {value.shape[idx]}",
        )

    return TensorProxy(like=a)


scatter_add = make_prim(PrimIDs.SCATTER_ADD, "scatter_add", meta=scatter_add_meta)


def transpose_meta(a: TensorProxy, permutation: Sequence[int]) -> TensorProxy:
    utils.check(isinstance(a, TensorProxy), lambda: f"Expected a={a} to be a TensorProxy!")
    utils.check(
        a.ndim == len(permutation),
        lambda: f"Expected the length ({len(permutation)}) of the permutation={permutation} to be the number of dimensions ({a.ndim}) of a={a}",
    )
    utils.check_valid_permutation(a.ndim, permutation)

    new_shape = [0] * a.ndim
    for idx, dim in enumerate(permutation):
        new_shape[idx] = a.shape[dim]

    return TensorProxy(like=a, shape=new_shape)


transpose = make_prim(PrimIDs.TRANSPOSE, "transpose", meta=transpose_meta)


view = make_prim(PrimIDs.VIEW, "view", meta=reshape_meta)

#
# Memory format prims (Experimental)
#


def stride_order_meta(a: TensorProxy, order: Sequence[int]) -> TensorProxy:
    # Validates inputs
    utils.check_type(a, TensorProxy)
    utils.check_valid_permutation(a.ndim, order)

    return TensorProxy(like=a)


# TODO Consider a more general stride manipulation primitive, like PyTorch's
#   as_strided or set_strided operations
# See clang.stride_order for this prim's documentation
stride_order = make_prim(PrimIDs.STRIDE_ORDER, "stride_order", meta=stride_order_meta)

#
# Reduction prims
#
# TODO Review output_dtype modeling


# TODO Add type annotations
def _compute_reduction_output_shape(shape: Sequence[int], dims: Sequence[int]) -> Sequence[int]:
    for idx in dims:
        utils.validate_idx(len(shape), idx)

    new_shape = []
    for idx in range(len(shape)):
        if idx in dims:
            continue

        new_shape.append(shape[idx])

    return tuple(new_shape)


def _reduction_meta(a: TensorProxy, dims: Sequence[int], *, output_dtype: Optional[dtypes.dtype] = None) -> TensorProxy:
    """Meta function for single output reduction operations."""
    if output_dtype is None:
        output_dtype = a.true_dtype

    # Validates types
    utils.check_type(a, TensorProxy)
    utils.check_type(dims, Sequence)
    utils.check_type(output_dtype, dtypes.dtype)

    utils.check(
        len(dims) > 0 or len(a.shape) == 0,
        lambda: f"Expected {dims=} to be a non-empty sequence when the tensor's shape {a.shape} is non-empty",
    )

    output_shape = _compute_reduction_output_shape(a.shape, dims)

    return TensorProxy(like=a, shape=output_shape, dtype=output_dtype)


# TODO: review if reduction meta is OK for amax
amax = make_prim(PrimIDs.AMAX, "amax", meta=_reduction_meta)

amin = make_prim(PrimIDs.AMIN, "amin", meta=_reduction_meta)

prod = make_prim(PrimIDs.PROD, "prod", meta=_reduction_meta)

sum = make_prim(PrimIDs.SUM, "sum", meta=_reduction_meta)


# TODO Add comment for why var doesn't use _reduction_meta
# TODO Add output_dtype?
# TODO Check that dims is a sequence of integers
def _var_meta(a: TensorProxy, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    # Checks input types
    utils.check_type(a, TensorProxy)
    utils.check_type(dims, Sequence)
    utils.check_type(correction, Number)

    output_dtype = None
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.true_dtype)
    else:
        output_dtype = a.true_dtype

    return _reduction_meta(a, dims, output_dtype=output_dtype)


def _var_mean_meta(a: TensorProxy, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    output_dtype = None
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.true_dtype)
    else:
        output_dtype = a.true_dtype
    var = _reduction_meta(a, dims, output_dtype=output_dtype)
    mean = _reduction_meta(a, dims, output_dtype=a.true_dtype)
    return (var, mean)


var = make_prim(PrimIDs.VAR, "var", meta=_var_meta)
var_mean = make_prim(PrimIDs.VAR_MEAN, "var_mean", meta=_var_mean_meta)

#
# Linear algebra prims
#
# NOTE linear algebra prims are highly experimental and will almost definitely change


# out = a @ w.transpose() + bias
def linear_meta(a: TensorProxy, w: TensorProxy, bias: None | TensorProxy) -> TensorProxy:
    # a's shape is (batch dims..., in)
    # w's shape is (out x in)
    # if bias is not None, bias's shape is (out)
    # the output shape is (batch dims..., out)

    # Checks types of the required arguments
    utils.check(isinstance(a, TensorProxy), lambda: f"a={a} was not a TensorProxy!")
    utils.check(isinstance(w, TensorProxy), lambda: f"w={w} was not a TensorProxy!")

    # Checks that required arguments are on the same device
    utils.check(a.device == w.device, lambda: f"Expected a.device={a.device} and w.device={w.device} to be the same!")

    # Acquires the computation dtype and checks that a and w have the same dtype
    dtype = a.dtype
    utils.check(
        dtypes.are_same_dtypes(a, w), lambda: f"Expected a.dtype={a.dtype} and w.dtype={w.dtype} to be the same!"
    )

    # Acquires the shape information and validates the shapes of the required arguments
    batch_dims = a.shape[:-1]
    in_length = a.shape[-1]

    # Validates w's shape
    utils.check(
        len(w.shape) == 2, lambda: f"Expected w.shape={w.shape} to have length 2, but found length {len(w.shape)}!"
    )
    utils.check(
        w.shape[1] == in_length,
        lambda: f"Expected w.shape={w.shape} to have an innermost dimension of length {in_length}, the same length as the innermost dimension of a.shape={a.shape}!",
    )

    out_length = w.shape[0]

    # Validates bias shape
    if bias is not None:
        utils.check(isinstance(bias, TensorProxy), lambda: f"bias={bias} was not None or a TensorProxy!")
        utils.check(
            a.device == bias.device,
            lambda: f"Expected a.device={a.device} and bias.device={bias.device} to be the same!",
        )
        utils.check(
            len(bias.shape) == 1,
            lambda: f"Expected bias.shape={bias.shape} to have length 1, but found length {len(bias.shape)}!",
        )
        utils.check(
            bias.shape[0] == out_length,
            lambda: f"Expected bias.shape={bias.shape} to have an innermost dimension of length {out_length}, the same length as the outermost dimension of w.shape={w.shape}!",
        )
        utils.check(
            dtypes.are_same_dtypes(bias, a),
            lambda: f"Expected a.dtype={a.dtype} and bias.dtype={bias.dtype} to be the same!",
        )

    out_shape = batch_dims + (out_length,)

    requires_grad = any((a.requires_grad, w.requires_grad, False if bias is None else bias.requires_grad))
    return TensorProxy(shape=out_shape, device=a.device, dtype=dtype, requires_grad=requires_grad)


linear = make_prim(PrimIDs.LINEAR, "linear", meta=linear_meta)


def matmul_meta(a: TensorProxy, b: TensorProxy) -> TensorProxy:
    # Checks types
    utils.check(isinstance(a, TensorProxy), lambda: f"a={a} was not a TensorProxy")
    utils.check(isinstance(b, TensorProxy), lambda: f"b={b} was not a TensorProxy")

    if a.ndim < 1 or b.ndim < 1:
        raise NotImplementedError

    utils.check(a.device == b.device, lambda: f"Expected a.device={a.device} and b.device={b.device} to be the same")

    utils.check(
        dtypes.are_same_dtypes(a, b), lambda: f"Expected a.dtype={a.dtype} and b.dtype={b.dtype} to be the same"
    )

    if a.ndim == 1 and b.ndim == 1:
        utils.check(
            a.shape[0] == b.shape[0],
            lambda: f"Expected a.shape={a.shape} and b.shape={b.shape} to have the same length",
        )
        return TensorProxy(like=a, shape=())

    if a.ndim == 1:
        utils.check(
            a.shape[0] == b.shape[-2],
            lambda: f"Expected a.shape={a.shape} to be matrix multipiable with b.shape={b.shape}",
        )
        shape = list(b.shape[:-2])
        shape.append(b.shape[-1])
        return TensorProxy(like=a, shape=shape)

    if b.ndim == 1:
        utils.check(
            a.shape[-1] == b.shape[0],
            lambda: f"Expected a.shape={a.shape} to be matrix multipiable with b.shape={b.shape}",
        )
        shape = list(a.shape[:-2])
        shape.append(a.shape[-2])
        return TensorProxy(like=a, shape=shape)

    utils.check(
        utils.same_shape(a.shape[:-2], b.shape[:-2]),
        lambda: f"Expected the batch dimensions of a ({a.shape[:-2],}) and the batch dimensions of b ({b.shape[:-2]}) to be the same",
    )

    utils.check(
        a.shape[-1] == b.shape[-2],
        lambda: f"Expected the the last two dimensions of a ({a.shape[-2:]}) be matrix multipiable with the last two dimensions of b ({b.shape[-2:]})",
    )

    shape = list(a.shape[:-2])
    shape.append(a.shape[-2])
    shape.append(b.shape[-1])

    return TensorProxy(like=a, shape=shape)


matmul = make_prim(PrimIDs.MATMUL, "matmul", meta=matmul_meta)

#
# NN prims
#


def embedding_meta(a, weight, *, padding_idx=-1, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    # TODO: canonicalize and validating padding idx with weight.shape[0]

    if max_norm is not None:
        raise NotImplementedError

    utils.check(a.dtype == dtypes.int64, lambda: f"Expected a.dtype={a.dtype} to be int64")
    utils.check(weight.ndim == 2, lambda: f"Expected weight (weight.shape={weight.shape} to be a matrix)")

    shape = list(a.shape)
    shape.append(weight.shape[1])

    return TensorProxy(like=weight, shape=shape)


embedding = make_prim(PrimIDs.EMBEDDING, "embedding", meta=embedding_meta)


# TODO Add annotations
# TODO Review requires_grad=False -- what about double backward?
# TODO Once we have fusible index_put we can implement it using primitives
# For now we just use the PyTorch implementation
def embedding_backward_meta(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    shape = (num_weights, grad.shape[-1])
    return TensorProxy(shape=shape, device=grad.device, dtype=grad.dtype, requires_grad=False)


embedding_backward = make_prim(PrimIDs.EMBEDDING_BACKWARD, "embedding_backward", meta=embedding_backward_meta)

#
# Distributed prims
#
import torch.distributed


# This enum describes what all_reduce (below) will actually do
#   These operations are performed elementwise on all the "versions" of
#   the tensor across processes.
class DistributedReduceOps(Enum):
    SUM = auto()
    # AVG = auto()
    # PRODUCT = auto()
    # MIN = auto()
    # MAX = auto()
    # BAND = auto()
    # BOR = auto()
    # BXOR = auto()
    # PREMUL_SUM = auto()


# NOTE DISTRIBUTED AVAILABILITY
# PyTorch is often built without distributed support, which can be queried for using
#   torch.distributed.is_available(). When PyTorch is built without distributed then we
#   want to avoid accessing any parts of the torch.distributed module except
#   the is_available() function. Prims that depend on torch.distributed should
#   define stubs that throw unsupported errors here, and the actual prim implementations
#   in the else branch below.

if not torch.distributed.is_available():

    def all_reduce_meta(
        a: TensorProxy, op: DistributedReduceOps, group: torch.distributed.ProcessGroup, do_async: Number
    ) -> None:
        utils.check(False, lambda: f"PyTorch distributed is not available, {torch.distributed.is_available()=}")

    def wait_meta(a: FutureTensorProxy) -> None:
        utils.check(False, lambda: f"PyTorch distributed is not available, {torch.distributed.is_available()=}")

else:
    # NOTE This is essentially a wrapper around
    #   https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce
    #   that models the operation as a functional one.
    # TODO Support additional reduction operations
    # TODO Support do_async=True (maybe by adding the idea of a TensorProxy being a future?)
    # TODO Consider our own distributed calls that don't just wrap PyTorch's
    def all_reduce_meta(
        a: TensorProxy, op: DistributedReduceOps, group: torch.distributed.ProcessGroup, do_async: Number
    ) -> TensorProxy | FutureTensorProxy:
        # Checks types
        utils.check_type(a, TensorProxy)
        utils.check_type(op, DistributedReduceOps)
        utils.check_type(group, torch.distributed.ProcessGroup)
        utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")

        if do_async:
            return FutureTensorProxy(like=a)

        return TensorProxy(like=a)

    # NOTE This is a very particular implementation of wait that may need to be
    #   generalized in the future
    def wait_meta(a: FutureTensorProxy) -> TensorProxy:
        # Checks types
        utils.check_type(a, FutureTensorProxy)

        return TensorProxy(like=a)


all_reduce = make_prim(PrimIDs.ALL_REDUCE, "all_reduce", meta=all_reduce_meta)
wait = make_prim(PrimIDs.WAIT, "wait", meta=wait_meta)
