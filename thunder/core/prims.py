from enum import auto, Enum
from numbers import Number
from functools import partial, reduce
import operator
import math
from typing import Union, Type, Any, List, Sequence, Dict, Tuple

import torch
import numpy as np

from thunder.core.symbol import Symbol, BoundSymbol, default_python_printer
from thunder.core.proxies import TensorProxy, NumberProxy, is_proxyable, proxy, numberproxy, pyval
import thunder.core.codeutils as codeutils
from thunder.core.codeutils import Printable
import thunder.core.utils as utils
import thunder.core.baseutils as baseutils
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.trace import get_tracectx
from thunder.core.langctx import langctx

#
# Primitives and helpers for defining them
#


class PrimIDs(Enum):
    # Unpacking prims
    UNPACK_TRIVIAL = auto()
    UNPACK_SEQUENCE = auto()
    UNPACK_DICT = auto()
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
    FULL = auto()
    IOTA = auto()
    UNIFORM = auto()
    # Shape prims
    BROADCAST_IN_DIM = auto()
    CAT = auto()
    PAD = auto()
    RESHAPE = auto()
    SLICE = auto()
    SQUEEZE = auto()
    TRANSPOSE = auto()
    TAKE = auto()
    TAKE_ALONG_AXIS = auto()
    VIEW = auto()
    # Elementwise unary prims
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
    BITWISE_XOR = auto()
    DIV = auto()
    EQ = auto()
    FMOD = auto()
    GE = auto()
    GT = auto()
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
    # Matmul prims (Experimental!)
    LINEAR = auto()
    MATMUL = auto()
    # NN prims (Experimental!)
    EMBEDDING = auto()


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
):
    sym = Symbol(
        name=name, meta=prim_ctx(meta), python_impl=python_impl, id=id, is_prim=True, python_printer=python_printer
    )
    return sym


#
# Unpacking prims
#


def unpack_trivial_impl(x: Any) -> Any:
    return x


def unpack_trivial_meta(x: Any) -> Any:
    return x


def unpack_trivial_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: Dict[str, Printable]
) -> str:
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected one argument for unpack_trivial but got {arg_strings}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_trivial but got {kwarg_strings}",
        exception_type=AssertionError,
    )

    result_str = "" if bsym.output is None else f"{codeutils.prettyprint(out_printables)} = "
    arg_str = codeutils.prettyprint(arg_printables)

    s = f"# {result_str} {arg_str}"
    return s


unpack_trivial = make_prim(
    PrimIDs.UNPACK_TRIVIAL,
    "unpack_trivial",
    meta=unpack_trivial_meta,
    python_printer=unpack_trivial_printer,
    python_impl=unpack_trivial_impl,
)


# TODO Restore const criteria
def unpack_sequence_meta(x, l):
    utils.check_type(x, Sequence)

    assert len(x) == l
    # _ensure_const(l)

    return x


# TODO Review using multi-line unpacks more cleverly
# TODO Possibly put the length in the code to show the requirement
def unpack_sequence_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: Dict[str, Printable]
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

    call_str = f"{codeutils.prettyprint(arg_printables[0])}"

    # Short-circuits if there's nothing to unpack:
    if len(bsym.output) == 0:
        return f"# {call_str} (empty sequence)"

    trace = get_tracectx()

    result_str: str
    # Special-cases unpacking one item
    if not codeutils.is_collection(bsym.output):
        out = trace.get_tracked_object(bsym.output)
        return f"{codeutils.prettyprint(out)}, = {call_str}"

    lines = []
    for out in bsym.output:
        out = trace.get_tracked_object(out)
        line = f"{codeutils.prettyprint(out)}, \\"
        lines.append(line)

    lines.append(f"= {call_str}")
    return lines


def unpack_sequence_impl(x, l):
    assert len(x) == l
    return x


unpack_sequence = make_prim(
    PrimIDs.UNPACK_SEQUENCE,
    "unpack_sequence",
    meta=unpack_sequence_meta,
    python_printer=unpack_sequence_printer,
    python_impl=unpack_sequence_impl,
)


def unpack_dict_impl(d: dict, keys):
    return d


# TODO: instead of sorting the keys, should this return the key->value mapping? Or set it privately for printing?
# TODO: is sorting sufficient?
def unpack_dict_meta(d: dict, keys: Tuple) -> dict:
    utils.check_type(d, dict)
    utils.check_type(keys, tuple)
    utils.check(
        tuple(d.keys()) == keys,
        lambda: f"unpack_dict received a dict {d} and nonmatching keys {keys}",
        exception_type=AssertionError,
    )

    for k in d.keys():
        is_printable, _ = codeutils.is_printable(k)
        utils.check(
            is_printable,
            lambda: f"Trying to unpack a dict with unprintable key {k}, but currently only dicts with printable keys are supported",
        )

    return d


def unpack_dict_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: Dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_dict but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_dict but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    lines = []
    d = bsym.output
    dprintable, keyprintables = arg_printables
    dname = codeutils.prettyprint(dprintable)

    trace = get_tracectx()
    for key in keyprintables:
        out = d[key]
        out = trace.get_tracked_object(out)

        keystr = codeutils.prettyprint(key)
        s = f"{codeutils.prettyprint(out, with_type=True)} = {dname}[{keystr}]"
        lines.append(s)

    return lines


unpack_dict = make_prim(
    PrimIDs.UNPACK_DICT,
    "unpack_dict",
    meta=unpack_dict_meta,
    python_printer=unpack_dict_printer,
    python_impl=unpack_dict_impl,
)

#
# Utility prims
#


def _print_meta(x):
    pass


def python_print_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: Dict[str, Printable]
):
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
        lambda: f"Expected only one arg printablewhen printing python_print, but got {kwarg_printables}",
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


def _comment_meta(s: str) -> None:
    return None


def comment_printer(s: str) -> str:
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
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: Dict[str, Printable]
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
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: Dict[str, Printable]
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
    a: Union[TensorProxy, Number], dtype: Union[Type, dtypes.dtype]
) -> Union[TensorProxy, NumberProxy, Number]:
    utils.check_type(a, (Number, TensorProxy))
    utils.check_type(dtype, (Type, dtypes.dtype))

    # NOTE Python numbers are constants, and this will return another Python number when given one because
    #   The conversion is constant
    if isinstance(a, Number):
        utils.check(utils.is_numbertype(dtype), lambda: f"Trying to convert a number to non-numbertype object {dtype}")

        if isinstance(a, NumberProxy):
            return numberproxy(dtype, utils.get_numberlike_value(a))

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


def _elementwise_unary_meta(
    a: Union[TensorProxy, Number],
    *,
    number_handler=None,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME,
    supported_input_dtypes=dtypes.all_dtypes_and_numbertypes,
):
    # Checks that inputs have an expected type
    utils.check_type(a, (TensorProxy, Number))

    if isinstance(a, Number):
        # Checks that the numbertype is supported
        typ = utils.get_numberlike_type(a)
        val = utils.get_numberlike_value(a)

        if val is None or number_handler is None:
            return numberproxy(typ, None)

        utils.check(typ in supported_input_dtypes, lambda: f"Unsupported input dtype {typ}")

        value = number_handler(a)
        typ = a.python_type if isinstance(a, NumberProxy) else type(a)
        return numberproxy(typ, value)

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


abs = make_prim(
    PrimIDs.ABS,
    "abs",
    meta=partial(
        _elementwise_unary_meta,
        output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.COMPLEX_TO_FLOAT,
    ),
)

acos = make_prim(
    PrimIDs.ACOS,
    "acos",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

acosh = make_prim(
    PrimIDs.ACOSH,
    "acosh",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

asin = make_prim(
    PrimIDs.ASIN,
    "asin",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

asinh = make_prim(
    PrimIDs.ASINH,
    "asinh",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

atan = make_prim(
    PrimIDs.ATAN,
    "atan",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

atanh = make_prim(
    PrimIDs.ATANH,
    "atanh",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

bitwise_not = make_prim(
    PrimIDs.BITWISE_NOT,
    "bitwise_not",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=dtypes.exact_dtypes),
)

# TODO Should ceil accept float16 and bfloat16 types?
ceil = make_prim(
    PrimIDs.CEIL,
    "ceil",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=dtypes.float_dtypes),
)

cos = make_prim(
    PrimIDs.COS,
    "cos",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

cosh = make_prim(
    PrimIDs.COSH,
    "cosh",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

erf = make_prim(
    PrimIDs.ERF,
    "erf",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

erfc = make_prim(
    PrimIDs.ERFC,
    "erfc",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

erfcinv = make_prim(
    PrimIDs.ERFCINV,
    "erfcinv",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

erfinv = make_prim(
    PrimIDs.ERFINV,
    "erfinv",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

exp = make_prim(
    PrimIDs.EXP,
    "exp",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

exp2 = make_prim(
    PrimIDs.EXP2,
    "exp2",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

expm1 = make_prim(
    PrimIDs.EXPM1,
    "expm1",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

# TODO Should floor accept float16 and bfloat16 types?
floor = make_prim(
    PrimIDs.FLOOR,
    "floor",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=dtypes.float_dtypes),
)

isfinite = make_prim(
    PrimIDs.ISFINITE,
    "isfinite",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=dtypes.inexact_dtypes),
)

lgamma = make_prim(
    PrimIDs.LGAMMA,
    "lgamma",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

log = make_prim(
    PrimIDs.LOG,
    "log",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

log10 = make_prim(
    PrimIDs.LOG10,
    "log10",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

log1p = make_prim(
    PrimIDs.LOG1P,
    "log1p",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

log2 = make_prim(
    PrimIDs.LOG2,
    "log2",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

ndtri = make_prim(
    PrimIDs.NDTRI,
    "ndtri",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

neg = make_prim(
    PrimIDs.NEG,
    "neg",
    meta=_elementwise_unary_meta,
)

reciprocal = make_prim(
    PrimIDs.RECIPROCAL,
    "reciprocal",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

# TODO Review dtypes for round
round = make_prim(
    PrimIDs.ROUND,
    "round",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

rsqrt = make_prim(
    PrimIDs.RSQRT,
    "rsqrt",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
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
sign = make_prim(
    PrimIDs.SIGN,
    "sign",
    meta=_elementwise_unary_meta,
)

signbit = make_prim(
    PrimIDs.SIGNBIT,
    "signbit",
    meta=partial(_elementwise_unary_meta, output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL),
)

sin = make_prim(
    PrimIDs.SIN,
    "sin",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

sinh = make_prim(
    PrimIDs.SINH,
    "sinh",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

sqrt = make_prim(
    PrimIDs.SQRT,
    "sqrt",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

tan = make_prim(
    PrimIDs.TAN,
    "tan",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

tanh = make_prim(
    PrimIDs.TANH,
    "tanh",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

# TODO Review dtypes for trunc (with round and ceil and floor)
trunc = make_prim(
    PrimIDs.TRUNC,
    "trunc",
    meta=partial(_elementwise_unary_meta, supported_input_dtypes=fp_math_dtypes),
)

#
# Elementwise binary prims
#


# TODO add stride logic
# TODO Improve error messages for mismatched dtypes (using an error context)
def _elementwise_binary_meta(
    a: Union[TensorProxy, Number],
    b: Union[TensorProxy, Number],
    *,
    number_handler=None,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME,
    supported_input_dtypes=dtypes.all_dtypes_and_numbertypes,
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
        if aval is None or bval is None or number_handler is None:
            return numberproxy(numbertype, None)

        value = number_handler(aval, bval)
        return numberproxy(numbertype, value)

    # Checks same shape
    # NOTE: this doesn't verify a common shape if one or more inputs is a number
    utils.check_same_shape(a, b)

    # Checks same device
    utils.check_same_device(a, b)

    tensor = a if isinstance(a, TensorProxy) else b

    if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME:
        # NOTE that this is not just like=tensor, because one tensor could have a weak dtype
        #   and the other a strong dtype, and these are the "same"
        return TensorProxy(like=tensor, dtype=dtype)
    if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL:
        return TensorProxy(like=tensor, dtype=dtypes.bool8)
    if output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.COMPLEX_TO_FLOAT and dtypes.is_complex_dtype(dtype):
        return TensorProxy(like=tensor, dtype=dtypes.corresponding_real_dtype(dtype))

    raise AssertionError(f"Unknown {output_dtype_kind=}")


# number_fn should be a function that handles Number x Number inputs,
#   it should only depend on Python and standard Python libraries
# torch_fn should be a PyTorch operation or composition of PyTorch
#   operations that implements the primitive
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
        meta=partial(
            _elementwise_binary_meta,
            number_handler=number_fn,
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
    supported_input_dtypes=dtypes.exact_dtypes,
)

bitwise_xor = _make_elementwise_binary_prim(
    PrimIDs.BITWISE_XOR,
    "bitwise_and",
    supported_input_dtypes=dtypes.exact_dtypes,
)


def _div_numbers(a: Number, b: Number) -> Number:
    if type(a) in dtypes.exact_dtypes and type(b) in dtypes.exact_dtypes:
        # Accounts for rounding towards zero instead of flooring
        if (a >= 0) != (b >= 0) and a % b:
            return a // b + 1
        else:
            return a // b

    return a / b


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

fmod = _make_elementwise_binary_prim(
    PrimIDs.FMOD,
    "fmod",
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

remainder = _make_elementwise_binary_prim(
    PrimIDs.REMAINDER,
    "remainder",
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


# TODO restore type promotion
# TODO add stride logic
# TODO revise number handling to account for numbers with unknown values
# TODO extract some of this logic into helpers
# TODO use device properly
def _where_meta(pred, a, b):
    # Checks types
    # NOTE pred must be a tensor or bool
    utils.check_type(pred, (TensorProxy, bool))
    utils.check_type(a, (TensorProxy, Number))
    utils.check_type(b, (TensorProxy, Number))

    # Checks devices and determines result device
    utils.check_same_device(pred, a, b)
    resultdevice = devices.cpu
    devices_ = tuple(x.device for x in (pred, a, b) if isinstance(x, TensorProxy))
    if len(devices_) > 0:
        resultdevice = devices_[0]

    # Checks pred dtype and determines result dtype
    if isinstance(pred, TensorProxy):
        utils.check(
            pred.dtype is dtypes.bool8,
            lambda: f"Expected pred to be a tensor with dtype bool, but found dtype {pred.dtype}",
        )

    numbertype, tensordtype = utils.check_same_dtype(a, b)
    dtype = tensordtype if tensordtype is not None else numbertype

    # Checks shapes
    utils.check_same_shape(pred, a, b)

    # Constructs return meta

    # FIXME
    # Handles all number cases with custom number handler
    if isinstance(pred, Number) and isinstance(a, Number) and isinstance(b, Number):
        raise NotImplementedError
        # result = a if pred else b
        # result = resulttype(result)
        # return proxy(result, name=proxyname)

    # Determines output shape
    # NOTE Assumes at least one of pred, a, and b is a TensorProxy because of prior shortcircuit
    shapes = tuple(x.shape for x in (pred, a, b) if isinstance(x, TensorProxy))
    resultshape = shapes[0]

    return TensorProxy(shape=resultshape, device=resultdevice, dtype=dtype)


where = make_prim(
    PrimIDs.WHERE,
    "where",
    meta=_where_meta,
)

#
# Tensor creation prims
#
# TODO: add some architecture for constructing tensor creation prims


def _iota_meta(length, *, start, step, device, dtype):
    utils.check(utils.is_exact_dtype(dtype), lambda: f"dtype={dtype} was not an exact dtype")
    utils.check(not utils.is_boolean_dtype(dtype), lambda: f"dtype={dtype} was not a non-boolean dtype")
    utils.check(length >= 0, lambda: f"length={length} was not weakly positive")

    shape = () if length == 0 else (length,)

    return TensorProxy(shape=shape, device=device, dtype=dtype)


iota = make_prim(PrimIDs.IOTA, "iota", meta=_iota_meta)


def _full_meta(shape: Sequence[int], fill_value: Number, *, device: devices.Device, dtype: dtypes.dtype):
    # Checks inputs
    utils.check_type(fill_value, Number)

    # Ensures the requested fill_value can be safely cast to the dtype
    # NOTE This is always true if the dtype is inferred
    fill_value_dtype = dtypes.to_dtype(fill_value)
    utils.check(
        utils.can_safe_cast_number_to(fill_value, fill_value_dtype),
        lambda: f"Can't safely cast fill_value of numbertype {fill_value_dtype} to dtype {dtype}",
    )

    return TensorProxy(shape=shape, device=device, dtype=dtype)


full = make_prim(
    PrimIDs.FULL,
    "full",
    meta=_full_meta,
)


# TODO Should the uniform prim include minval maxval or always be [0, 1)?
def _uniform_meta(
    shape: Sequence[int], minval: Number, maxval: Number, *, device: devices.Device, dtype: dtypes.dtype
) -> TensorProxy:
    # Checks inputs
    utils.check_type(minval, Number)
    utils.check_type(maxval, Number)
    utils.check_type(device, devices.Device)
    utils.check_type(dtype, dtypes.dtype)

    return TensorProxy(shape=shape, device=device, dtype=dtype)


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


def cat_meta(tensors: List[TensorProxy], dim: int):
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

    return TensorProxy(like=tensors[0], shape=shape)


cat = make_prim(
    PrimIDs.CAT,
    "cat",
    meta=cat_meta,
)


def cat_meta(tensors: List[TensorProxy], dim: int):
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

    return TensorProxy(like=tensors[0], shape=shape)


cat = make_prim(
    PrimIDs.CAT,
    "cat",
    meta=cat_meta,
)


def pad_meta(a, padding_value, padding_config):
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


def reshape_meta(a, shape):
    # Validates inputs
    utils.check(isinstance(a, TensorProxy), lambda: f"a={a} was not a TensorProxy!")
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


# TODO: be clear about what the prim can handle and what it can't
# NOTE: the stride parameter here refers to the stride of the slice, not the tensor's
#   strides
def slice_meta(a, start_indices, end_indices, strides=None):
    if strides is None:
        strides = [1] * a.ndim

    # Checks types
    utils.check(isinstance(a, TensorProxy), lambda: f"Expected a={a} to be a TensorProxy!")
    utils.check(isinstance(start_indices, Sequence), lambda: f"Expected start_indices={start_indices} to be a Sequence")
    utils.check(isinstance(end_indices, Sequence), lambda: f"Expected end_indices={end_indices} to be a Sequence")
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
        print(f"{start=}, {stop=} {type(stop)=} {pyval(stop)=}")
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


def squeeze_meta(a, dims):
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


def transpose_meta(a, permutation):
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


def take_along_axis_meta(a, index, dim):
    utils.check(isinstance(a, TensorProxy), lambda: f"Expected a={a} to be a TensorProxy!")
    utils.check(isinstance(index, TensorProxy), lambda: f"Expected index={index} to be a TensorProxy!")
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"index dtype={dtype} was not an integer dtype")
    utils.check(index.ndim == a.ndim, lambda: f"Expected index={index} to a 1-D or scalar TensorProxy!")
    utils.validate_idx(a.ndim, dim)
    for idx, l in enumerate(a.shape):
        if idx != dim:
            utils.check(
                index.shape[idx] == l or index.shape[idx] == 1,
                lambda: f"take_along_axis 'index' size on all dimensions to be broadcast against 'a', except `dim`. Found incompatible sizes on dim {dim}, where 'a' has {l} and 'index' has {index.shape[idx]}",
            )

    return TensorProxy(like=a, shape=index.shape)


take_along_axis = make_prim(PrimIDs.TAKE_ALONG_AXIS, "take_along_axis", meta=take_along_axis_meta)


def take_meta(a, index, dim):
    utils.check(isinstance(a, TensorProxy), lambda: f"Expected a={a} to be a TensorProxy!")
    utils.check(isinstance(index, TensorProxy), lambda: f"Expected index={index} to be a TensorProxy!")
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"index dtype={dtype} was not an integer dtype")
    utils.check(index.ndim <= 1, lambda: f"Expected index={index} to a 1-D or number TensorProxy!")
    utils.validate_idx(a.ndim, dim)

    l = index.shape[0] if index.ndim == 1 else 1
    new_shape = a.shape[:dim] + (l,) + a.shape[dim + 1 :]

    return TensorProxy(like=a, shape=new_shape)


take = make_prim(PrimIDs.TAKE, "take", meta=take_meta)

view = make_prim(PrimIDs.VIEW, "view", meta=reshape_meta)

#
# Reduction prims
#
# TODO Review output_dtype modeling


# TODO Add type annotations
def _compute_reduction_output_shape(shape, dims):
    for idx in dims:
        utils.validate_idx(len(shape), idx)

    new_shape = []
    for idx in range(len(shape)):
        if idx in dims:
            continue

        new_shape.append(shape[idx])

    return tuple(new_shape)


# TODO Add type annotations
# TODO Validate input types
def _reduction_meta(a, dims, *, output_dtype=None):
    """Meta function for single output reduction operations."""
    if output_dtype is None:
        output_dtype = a.true_dtype

    output_shape = _compute_reduction_output_shape(a.shape, dims)

    return TensorProxy(like=a, shape=output_shape, dtype=output_dtype)


# TODO: review if reduction meta is OK for amax
amax = make_prim(PrimIDs.AMAX, "amax", meta=_reduction_meta)

amin = make_prim(PrimIDs.AMIN, "amin", meta=_reduction_meta)

prod = make_prim(PrimIDs.PROD, "prod", meta=_reduction_meta)

sum = make_prim(PrimIDs.SUM, "sum", meta=_reduction_meta)


# TODO Add type annotations
# TODO Add comment for why var doesn't use _reduction_meta
# TODO Validate input types
def _var_meta(a: TensorProxy, dims, *, correction):
    output_dtype = None
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.true_dtype)
    else:
        output_dtype = a.true_dtype

    return _reduction_meta(a, dims, output_dtype=output_dtype)


var = make_prim(PrimIDs.VAR, "var", meta=_var_meta)

#
# Matmul prims
#
# NOTE: matmul prims are highly experimental and will almost definitely change


# out = a @ w.transpose() + bias
def linear_meta(a, w, bias):
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

    return TensorProxy(shape=out_shape, device=a.device, dtype=dtype)


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
