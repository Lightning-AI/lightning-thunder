from enum import auto, Enum
from numbers import Number
from functools import reduce
import operator
import builtins
import math
from types import NoneType
from typing import Any
from collections.abc import Callable
from collections.abc import Hashable, Sequence

import torch

from thunder.core.langctxs import LanguageContext, register_langctx, Languages, langctx

#
# Creates and registers the torch language context
#
# NOTE That this is done separately from the definition of thunder.torch operations, because the
#   language context must be available before those operations are defined

_method_name_to_fn_map: dict[str, Callable] = {}


# Creates and registers the primitive language context
# TODO RC1 Register additinoal methods and record operations on numbers
class PrimCtx(LanguageContext):
    def __init__(self):
        super().__init__("primitive")

    def has_method(self, id: str) -> bool:
        return id in _method_name_to_fn_map

    def get_method(self, id: Any, *args, **kwargs) -> Callable:
        # Note: concrete implmenetations should only raise AttributeError or
        #       return None for "missing" methods as the proxies will
        #       route __getattr__ to here and hasattr relies on __getattr__
        #       throwing AttributeError (only) when the attribute does
        #       not exist.

        # Verifies that the method is not being called on any tensor proxies
        inps, _ = tree_flatten((args, kwargs))
        for x in inps:
            if isinstance(x, TensorProxy):
                raise ValueError(f"Attempting to call {id} from the primitive language context on a tensor proxy")

        method: None | Callable = _method_name_to_fn_map.get(id, None)

        if method is None:
            raise ValueError(f"The {self.name} language context has no method {id}")

        return method


primctx = PrimCtx()
register_langctx(Languages.PRIMS, primctx)


# Registers a method with the torch language context
def register_method(method_name: str, method: Callable, /) -> None:
    _method_name_to_fn_map[method_name] = method


from thunder.core.symbol import Symbol, BoundSymbol, default_python_printer
from thunder.core.proxies import (
    CollectionProxy,
    ListProxy,
    TensorProxy,
    NumberProxy,
    proxy,
    numberproxy,
    pytype,
    pyval,
    Proxy,
    StringProxy,
    TupleProxy,
    AnyProxy,
    IntegerProxy,
)
import thunder.core.codeutils as codeutils
from thunder.core.codeutils import Printable
import thunder.core.utils as utils
import thunder.core.baseutils as baseutils
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes
from thunder.core.pytree import tree_flatten
from thunder.core.langctxs import LanguageContext, register_langctx, Languages

#
# Primitives and helpers for defining them
#


class PrimIDs(Enum):
    # Unpacking and input validation prims
    ASSERT_TENSOR_METADATA = auto()
    CHECK_TENSOR_SHAPE_AND_METADATA = auto()
    CHECK_NONE = auto()
    CHECK_EMPTY = auto()
    CHECK_LITERAL_LIKE = auto()
    CHECK_TYPE = auto()
    CHECK_INSTANCE = auto()
    CHECK_NUMBER_TYPE_AND_VALUE = auto()
    CHECK_BOOL_CONVERSION = auto()
    CHECK_STRING_VALUE = auto()
    CHECK_SLICE_VALUE = auto()
    CHECK_LEN = auto()
    ASSERT_COMPARE = auto()
    PYTHON_VARS = auto()
    UNPACK_FUNCTION_OBJ = auto()
    UNPACK_CACHE_INFO = auto()
    UNPACK_ATTR = auto()
    UNPACK_GETITEM = auto()
    UNPACK_EMPTY_DICT = auto()
    UNPACK_ITER = auto()
    UNPACK_NEXT = auto()
    UNPACK_KEY = auto()
    UNPACK_SEQUENCE = auto()
    UNPACK_TRIVIAL = auto()
    UNPACK_TUPLE = auto()
    UNPACK_LIST = auto()
    UNPACK_DICT_KEY = auto()
    UNPACK_PARAMETER = auto()
    UNPACK_BUFFER = auto()
    UNPACK_SUBMODULE = auto()
    UNPACK_THUNDER_MODULE = auto()
    CONSTRUCT_TUPLE = auto()
    PACK_LIST = auto()
    PACK_BUFFER = auto()
    PACK_ATTR = auto()
    PACK_SETITEM = auto()
    DATACLASS_NEW = auto()
    SHAPE = auto()
    # TODO: UNPACK_SET
    # Utility prims
    COMMENT = auto()
    DEL = auto()
    PRINT = auto()
    RETURN = auto()
    # Prims related to transforms (like grad)
    GET_GRAD = auto()
    PUT_GRAD = auto()
    # Data movement and transformation prims
    CONVERT_ELEMENT_TYPE = auto()
    DEVICE_PUT = auto()
    NUMPY_ARRAY_TO_TORCH_TENSOR = auto()  # Experimental
    # Tensor creation prims
    EXOGENOUS_LIKE = auto()
    FULL = auto()
    IOTA = auto()
    UNIFORM = auto()
    UNIFORM_PHILOX = auto()
    RANDINT = auto()
    RANDN = auto()
    EMPTY = auto()
    TENSOR_FROM_SEQUENCE = auto()
    CLONE = auto()
    UPDATE_ALIASES = auto()
    # Probability distribution-related ops
    MULTINOMIAL = auto()
    GET_AND_UPDATE_RNG_STATE = auto()
    # Reshaping and permuting prims
    BROADCAST_IN_DIM = auto()
    CAT = auto()
    FLIP = auto()
    RESHAPE = auto()
    SLICE = auto()
    SQUEEZE = auto()
    TRANSPOSE = auto()
    UNFOLD = auto()
    VIEW = auto()
    SHALLOW_COPY = auto()  # a view copy
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
    DIGAMMA = auto()
    ERF = auto()
    ERFC = auto()
    ERFCINV = auto()
    ERFINV = auto()
    EXP = auto()
    EXP2 = auto()
    EXPM1 = auto()
    FLOOR = auto()
    FREXP = auto()
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
    REAL = auto()
    IMAG = auto()
    # Elementwise binary prims
    ADD = auto()
    ATAN2 = auto()
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    DIV = auto()
    EQ = auto()
    PY_FLOORDIV = auto()
    FMOD = auto()
    GE = auto()
    GT = auto()
    LE = auto()
    LT = auto()
    MAXIMUM = auto()
    MINIMUM = auto()
    MUL = auto()
    NE = auto()
    NEXTAFTER = auto()
    POW = auto()
    REMAINDER = auto()
    SUB = auto()
    ZETA = auto()
    BITWISE_LEFT_SHIFT = auto()
    BITWISE_RIGHT_SHIFT = auto()
    # Elementwise ternary prims
    LERP = auto()
    WHERE = auto()
    # Reduction prims
    AMAX = auto()
    AMIN = auto()
    PROD = auto()
    SUM = auto()
    VAR = auto()
    VAR_MEAN = auto()
    STD = auto()
    ARGMAX = auto()
    ARGMIN = auto()
    TOPK = auto()
    # Sort and dim permutations prims
    SORT = auto()
    # Scatter and gather prims (Experimental!)
    GATHER = auto()
    SCATTER = auto()
    INDEX_ADD = auto()
    INDEX_COPY = auto()
    INDEX_PUT = auto()
    SCATTER_ADD = auto()
    TAKE = auto()
    TAKE_ALONG_AXIS = auto()
    COPY_WITH_SETITEM = auto()
    # Linear algebra prims (Mostly experimental)
    MATMUL = auto()
    _GROUPED_MM = auto()  # Used for grouped matmuls
    # NN prims (Experimental!)
    CONVOLUTION = auto()
    EMBEDDING = auto()
    EMBEDDING_BACKWARD = auto()
    LINEAR = auto()
    PAD = auto()
    # Memory access methods
    ITEM = auto()
    COPY_ = auto()
    BITCAST = auto()
    #
    SINK = auto()


class OpTags(Enum):
    # TODO -- Consider renaming this tag
    # The operation manipulates a tensor's shape
    #   These operations may return a view or a new tensor in PyTorch
    #   e.g. slice, squeeze, transpose, view, reshape
    SHAPE_OP = auto()
    REDUCTION_OP = auto()
    RANDOM_OP = auto()
    MATMUL_OP = auto()
    # Ops that might cause a device sync
    DEVICE_SYNC_OP = auto()
    # Labels operations that should not be removed by the dead code elimination (DCE) pass
    DONT_DCE = auto()
    IN_PLACE = auto()
    AUTO_REGISTERED = auto()
    # Label for operations representing enter/exit of context managers.
    CTX_MANAGER_ENTER_EXIT_OP = auto()
    # Label to explicitly disable an operation from recomputing in backward.
    DONT_RECOMPUTE_IN_BACKWARD = auto()
    # Don't automatically tag operation to be recomputed in backward
    DONT_AUTO_RECOMPUTE_IN_BACKWARD = auto()


# TODO RC1 Document this function and describe the parts of a primitive
# NOTE: See extend.single_op_executor
def make_prim(
    id,
    name,
    *,
    meta,
    python_printer=default_python_printer,
    python_impl: None | Callable = None,
    tags: None | Sequence[OpTags] = None,
    method_name: None | str = None,
    _bind_postprocess: None | Callable = None,
    _print_as_impl: bool = False,
):
    sym = Symbol(
        name=name,
        meta=langctx(Languages.PRIMS)(meta),
        id=id,
        is_prim=True,
        tags=None if tags is None else list(tags),
        python_printer=python_printer,
        python_impl=python_impl,
        _bind_postprocess=_bind_postprocess,
        _print_as_impl=_print_as_impl,
    )

    if method_name is not None:
        register_method(method_name, sym)

    return sym


#
# Unpacking and input validation prims
#


# TODO Add a tag for these assertions
# TODO Review performance (comparisons and string construction)
# TODO Change shape check into a rank check once constraints are generated on dimension lengths separately
# TODO The conversions to torch devices and dtypes could be done at compile time instead of runtime
# Checks type, shape, device, and dtype
def assert_tensor_metadata_impl(
    t: torch.Tensor, /, shape: tuple[int], device: devices.Device, dtype: dtypes.dtype, requires_grad: bool
) -> None:
    dtype = dtypes.to_torch_dtype(dtype)

    if (
        type(t) in (torch.Tensor, torch.nn.Parameter)
        and tuple(t.shape) == shape
        and str(t.device) == device.device_str()
        and t.dtype == dtype
        and t.requires_grad == requires_grad
    ):
        return

    raise AssertionError(
        f"Object had unexpected metadata. Expected type Tensor/nn.Parameter (without subclass), shape {shape}, device {str(device.device_str())}, dtype {dtype}, and {requires_grad=}, but found type {type(t)}, shape {tuple(t.shape)}, device {str(t.device)}, and requires_grad {t.requires_grad}"
    )


def assert_tensor_metadata_meta(
    t: torch.Tensor, /, shape: tuple[int], device: devices.Device, dtype: dtypes.dtype, requires_grad: bool
) -> None:
    return


assert_tensor_metadata = make_prim(
    PrimIDs.ASSERT_TENSOR_METADATA,
    "assert_tensor_metadata",
    meta=assert_tensor_metadata_meta,
    python_impl=assert_tensor_metadata_impl,
    _print_as_impl=True,
)


# TODO: Or maybe assert op to include comparisons?
def assert_compare_impl(v: Any, /, op: str, other: Any) -> None:
    cmp_impls = {
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
    }
    if cmp_impls[op](v, other):
        return
    raise AssertionError(f"Comparison constraint violated: {v} {op} {other}")


def assert_compare_meta(v: Any, /, op: str, other: Any) -> None:
    return


assert_compare = make_prim(
    PrimIDs.ASSERT_COMPARE,
    "assert_compare",
    meta=assert_compare_meta,
    python_impl=assert_compare_impl,
    _print_as_impl=True,
)


def python_vars_impl(arg: None | str = None, /) -> dict:
    if arg is None:
        return vars()
    return vars(arg)


def python_vars_meta(arg: None | str = None, /) -> Proxy:
    return Proxy()


def python_vars_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str:
    utils.check(
        len(arg_printables) <= 1,
        lambda: f"Expected at most one argument for vars but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for vars but got {kwarg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        type(out_printables) is Proxy,
        lambda: f"Expected a proxy output for vars, but got {out_printables=}",
        exception_type=AssertionError,
    )

    out_str = codeutils.prettyprint(out_printables)
    (args,) = arg_printables
    arg_str = codeutils.prettyprint(args)
    utils.check(
        type(arg_str) is str,
        lambda: f"Expected a string argument for vars, but got {arg_str}",
        exception_type=AssertionError,
    )

    # NOTE Implementing vars (mruberry)
    # thunder programs are compiled by Python's builtin compile
    #   this means that their globals() is defined by the
    #   dictionary passed to compile, and we want to be able
    #   to access a variety of global dicts, including the
    #   globals dict of the "main" module (which is not directly
    #   accessible, as best I can tell). To do this, for now
    #   at least, we assume that vars() is always being called
    #   in one of our "prologue" traces, where we add the key
    #   "__globals_dicts" to the dict that we pass to compile,
    #   and it contains the various global dictionaries indexed
    #   by their module names. In the future the prologue
    #   trace could import modules (other than the main module)
    #   and call vars directly on those module names, which
    #   would be more readable.

    # Special cases dunder main
    # TODO Review this -- maybe we should acquire the globals by
    #   getting dunder globals on the compiled function itself

    s = f"{out_str} = globals()['__global_dicts'][{arg_str}]"
    return s


python_vars = make_prim(
    PrimIDs.PYTHON_VARS,
    "python_vars",
    meta=python_vars_meta,
    python_printer=python_vars_printer,
    python_impl=python_vars_impl,
)


def _collectify(x: Any, *, name: str | None = None) -> Any:
    if isinstance(x, Proxy):
        return x
    if baseutils.is_collection(x):
        return CollectionProxy(x, name=name)
    if isinstance(x, slice):
        return CollectionProxy((x.start, x.stop, x.step), name=name)

    return x


# TODO RC1 Align with ASSERT_TENSOR_METADATA
# NOTE The device is stored as a string for easier, more readable comparisons
def _check_tensor_shape_and_metadata_meta(
    t: TensorProxy, shape: tuple[int, NumberProxy, ...], device: str, dtype: torch.dtype, requires_grad: bool
) -> None:
    # Validates types
    baseutils.check_type(t, TensorProxy)
    baseutils.check_valid_shape(shape)
    baseutils.check_type(device, str)
    baseutils.check_type(dtype, torch.dtype)
    baseutils.check_type(requires_grad, bool)


check_tensor_shape_and_metadata = make_prim(
    PrimIDs.CHECK_TENSOR_SHAPE_AND_METADATA,
    "check_tensor_shape_and_metadata",
    meta=_check_tensor_shape_and_metadata_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_none_meta(p: AnyProxy, /) -> None:
    # Validates types
    baseutils.check_type(p, AnyProxy)
    baseutils.check(pytype(p) is NoneType, lambda: f"Expected {p} to be None")


check_none = make_prim(
    PrimIDs.CHECK_NONE,
    "check_none",
    meta=_check_none_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_empty_meta(seq: tuple | list | dict, /) -> None:
    # Validates types
    baseutils.check_type(seq, (tuple, list, dict))
    baseutils.check(len(seq) == 0, lambda: f"Expected an empty sequence, but found {seq=}")


check_empty = make_prim(
    PrimIDs.CHECK_EMPTY,
    "check_empty",
    meta=_check_empty_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_len_meta(seq: tuple | list | dict, length: int, /) -> None:
    # Validates types
    # baseutils.check_type(seq, (tuple, list, dict))
    baseutils.check_type(length, (int,))


check_len = make_prim(
    PrimIDs.CHECK_LEN,
    "check_len",
    meta=_check_len_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_literal_like_meta(p: AnyProxy, v: Any, /) -> None:
    # Validates types
    baseutils.check_type(p, AnyProxy)
    baseutils.check(pyval(p) == v, lambda: f"Expected {p} to be equal to {v}")


check_literal_like = make_prim(
    PrimIDs.CHECK_LITERAL_LIKE,
    "check_literal_like",
    meta=_check_literal_like_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_type_meta(x: Any, typ: type, /) -> None:
    # Validates types
    baseutils.check(isinstance(typ, type), lambda: f"Expected a type for check_type, but found {typ}")
    baseutils.check(pytype(x) is typ, lambda: f"Different types for {pytype(x)} and {typ}")


check_type = make_prim(
    PrimIDs.CHECK_TYPE,
    "check_type",
    meta=_check_type_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_instance_meta(x: Any, types: tuple[type], /) -> None:
    # Validates types
    baseutils.check(types, tuple, lambda: f"Expected a tuple of types for check_instance, but found {types}")

    for typ in types:
        baseutils.check(
            type(typ) is type,
            lambda: f"Expected a tuple of types for check_instance, but found an object of type {type(typ)} in the tuple",
        )

    baseutils.check(any(map(lambda y: issubclass(pytype(x), y), types)), lambda: f"Type {pytype(x)} was not in {types}")


check_instance = make_prim(
    PrimIDs.CHECK_INSTANCE,
    "check_instance",
    meta=_check_instance_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_number_type_and_value_meta(n: NumberProxy, value: Number, /) -> None:
    # Validates types
    baseutils.check_type(n, NumberProxy)
    baseutils.check_type(value, (Number, NumberProxy))
    baseutils.check(pytype(n) == pytype(value), lambda: f"Different types for {n} and {value}")


check_number_type_and_value = make_prim(
    PrimIDs.CHECK_NUMBER_TYPE_AND_VALUE,
    "check_number_type_and_value",
    meta=_check_number_type_and_value_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_bool_conversion_meta(n: NumberProxy, b: bool, /) -> None:
    # Validates types
    baseutils.check_type(n, NumberProxy)
    baseutils.check_type(b, bool)


check_bool_conversion = make_prim(
    PrimIDs.CHECK_BOOL_CONVERSION,
    "check_bool_conversion",
    method_name="check_bool_conversion",
    meta=_check_bool_conversion_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_string_value_meta(s: StringProxy, value: str) -> None:
    # Validates types
    baseutils.check_type(s, StringProxy)
    baseutils.check_type(value, str)


check_string_value = make_prim(
    PrimIDs.CHECK_STRING_VALUE,
    "check_string_value",
    meta=_check_string_value_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_slice_value_meta(s: AnyProxy, value: slice) -> None:
    baseutils.check_type(s, AnyProxy)
    baseutils.check_type(value, slice)


check_slice_value = make_prim(
    PrimIDs.CHECK_SLICE_VALUE,
    "check_slice_value",
    meta=_check_slice_value_meta,
    tags=(OpTags.DONT_DCE,),
)


def unpack_trivial_impl(x: Any, /, *, name: str | None = None) -> Any:
    return x


def unpack_trivial_meta(x: Any, /, *, name: str | None = None) -> Any:
    utils.check(name is not None, lambda: "Expected name argmument to not be None")
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
    s = f"# {result_str}{' (unused)' if bsym.output is None else ''}"
    return s


# Removes the inputs from unpack_trivial, so it appears to have no input
def _unpack_trivial_bind_postprocess(bsym: BoundSymbol) -> None:
    utils.check(
        bsym.kwargs["name"] is not None,
        lambda: "Expected name keyword argument not to be None for unpack_trivial.bind().",
    )
    bsym.args = ()


unpack_trivial = make_prim(
    PrimIDs.UNPACK_TRIVIAL,
    "unpack_trivial",
    meta=unpack_trivial_meta,
    python_printer=unpack_trivial_printer,
    python_impl=unpack_trivial_impl,
    _bind_postprocess=_unpack_trivial_bind_postprocess,
)


def unpack_function_obj_impl(x: Any, /, *, name: str | None = None) -> Any:
    return x


def unpack_function_obj_meta(x: Any, /, *, name: str | None = None) -> Any:
    return x


def unpack_function_obj_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str:
    utils.check(
        len(arg_printables) == 0,
        lambda: f"Expected zero arguments for unpack_function_obj but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) <= 1,
        lambda: f"Expected at most one kwarg for unpack_function_obj but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    result_str = "_" if bsym.output is None else f"{codeutils.prettyprint(out_printables, with_type=True)}"
    s = f"{result_str} = globals()['__function_obj']"
    return s


def _unpack_function_obj_bind_postprocess(bsym: BoundSymbol) -> None:
    bsym.args = ()


unpack_function_obj = make_prim(
    PrimIDs.UNPACK_FUNCTION_OBJ,
    "unpack_function_obj",
    meta=unpack_function_obj_meta,
    python_printer=unpack_function_obj_printer,
    python_impl=unpack_function_obj_impl,
    _bind_postprocess=_unpack_function_obj_bind_postprocess,
)


def unpack_thunder_module_impl(x: Any, /) -> Any:
    return x


def unpack_thunder_module_meta(x: Any, /) -> Any:
    return x


def unpack_thunder_module_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str:
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected one argument for unpack_thunder_module but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_thunder_module but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    result_str = "_" if bsym.output is None else f"{codeutils.prettyprint(out_printables, with_type=True)}"
    arg_str = codeutils.prettyprint(arg_printables[0])
    s = f"{result_str} = thunder.core.module.get_thunder_module({arg_str})"
    return s


unpack_thunder_module = make_prim(
    PrimIDs.UNPACK_THUNDER_MODULE,
    "unpack_thunder_module",
    meta=unpack_thunder_module_meta,
    python_printer=unpack_thunder_module_printer,
    python_impl=unpack_thunder_module_impl,
)


def unpack_cache_info_impl(x: Any, /, *, name: str | None = None) -> Any:
    return x


def unpack_cache_info_meta(x: Any, /, *, name: str | None = None) -> Any:
    return x


def unpack_cache_info_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str:
    utils.check(
        len(arg_printables) == 0,
        lambda: f"Expected zero arguments for unpack_cache_info but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) <= 1,
        lambda: f"Expected at most one kwarg for unpack_cache_info but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    result_str = "_" if bsym.output is None else f"{codeutils.prettyprint(out_printables, with_type=True)}"
    s = f"{result_str} = thunder._get_cache_info()"
    return s


def _unpack_cache_info_bind_postprocess(bsym: BoundSymbol) -> None:
    bsym.args = ()


unpack_cache_info = make_prim(
    PrimIDs.UNPACK_CACHE_INFO,
    "unpack_cache_info",
    meta=unpack_cache_info_meta,
    python_printer=unpack_cache_info_printer,
    python_impl=unpack_cache_info_impl,
    _bind_postprocess=_unpack_cache_info_bind_postprocess,
)


# TODO Restore const criteria
def unpack_sequence_meta(x: Sequence | CollectionProxy, l: int, /) -> list:
    if isinstance(x, CollectionProxy):
        x = x.collection()

    utils.check_type(x, Sequence)
    utils.check_type(l, (int, IntegerProxy))
    baseutils.check(len(x) == l, lambda x=x, l=l: f"Expected the length of {x=} to be {l=}")

    return list(_collectify(y) for y in x)


def _make_parts_into_line_or_lines(parts: list[str], out: list[str] | None = None) -> list[str]:
    if out is None:
        lines = []
    else:
        lines = out
    line_parts = []
    pos = 0
    for p in parts:
        if pos and pos + len(p) > 80:
            lines.append("".join(line_parts) + "\\")
            line_parts = []
            pos = 0
        line_parts.append(p)
        pos += len(p)

    lines.append("".join(line_parts))
    return lines


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

    parts = [f"{codeutils.prettyprint(out, literals_as_underscores=True)}, " for out in out_printables]
    parts.append(f"= {call_str}")

    lines = _make_parts_into_line_or_lines(parts)
    # Add info about the unpacked elements as comments
    for out in out_printables:
        details = _make_parts_into_line_or_lines([f"# {codeutils.prettyprint(out, with_type=True)}"])
        lines.extend(details)
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


# NOTE This actually returns a new tuple of the elements, which allows the elements of tup
#   to appear in the symbol's output. If the original tuple is a proxy and it were just
#   returned directly then the output would just be the tuple
def _unpack_tuple_meta(tup: tuple, /) -> tuple:
    utils.check_type(tup, tuple)

    def _proxy(x: Any):
        if isinstance(x, Proxy):
            return x
        return proxy(x)

    return tuple(_proxy(x) for x in tup)


def _unpack_tuple_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected one argument for unpack_tuple but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_tuple but got {kwarg_printables}",
        exception_type=AssertionError,
    )
    utils.check_type(bsym.output, tuple)

    (x,) = arg_printables
    call_str = f"{codeutils.prettyprint(x)}"

    # Short-circuits if there's nothing to unpack:
    if len(bsym.output) == 0:
        return f"# {call_str} (empty tuple)"

    parts = [f"{codeutils.prettyprint(out, literals_as_underscores=True)}, " for out in out_printables]
    parts.append(f"= {call_str}")

    lines = _make_parts_into_line_or_lines(parts)
    return lines


unpack_tuple = make_prim(
    PrimIDs.UNPACK_TUPLE,
    "unpack_tuple",
    meta=_unpack_tuple_meta,
    python_printer=_unpack_tuple_printer,
)


# NOTE This actually returns a new tuple of the elements, which allows the elements of tup
#   to appear in the symbol's output. If the original tuple is a proxy and it were just
#   returned directly then the output would just be the tuple
def _unpack_list_meta(lst: list, /) -> list:
    utils.check_type(lst, list)

    def _proxy(x: Any):
        if isinstance(x, Proxy):
            return x
        return proxy(x)

    return list(_proxy(x) for x in lst)


def _unpack_list_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected one argument for unpack_list but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_list but got {kwarg_printables}",
        exception_type=AssertionError,
    )
    utils.check_type(bsym.output, list)

    (x,) = arg_printables
    call_str = f"{codeutils.prettyprint(x)}"

    # Short-circuits if there's nothing to unpack:
    if len(bsym.output) == 0:
        return f"# {call_str} (empty list)"

    parts = [f"{codeutils.prettyprint(out, literals_as_underscores=True)}, " for out in out_printables]
    parts.append(f"= {call_str}")

    lines = _make_parts_into_line_or_lines(parts)
    return lines


unpack_list = make_prim(
    PrimIDs.UNPACK_LIST,
    "unpack_list",
    meta=_unpack_list_meta,
    python_printer=_unpack_list_printer,
)


def _construct_tuple_meta(tup: tuple, /) -> tuple:
    utils.check_type(tup, tuple)
    return TupleProxy(tup)


construct_tuple = make_prim(PrimIDs.CONSTRUCT_TUPLE, "construct_tuple", meta=_construct_tuple_meta)


# NOTE UNPACK_ATTR is intended only to be bound to directly, and not called
def unpack_attr_meta(o: Any, key: str) -> Any:
    raise NotImplementedError


def unpack_attr_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_attr but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_attr but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    origin, key = arg_printables
    origin_str = codeutils.prettyprint(origin)
    keystr = key
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {origin_str}.{keystr}"


def unpack_attr_impl(o: Any, key: str) -> Any:
    return getattr(o, key)


unpack_attr = make_prim(
    PrimIDs.UNPACK_ATTR,
    "unpack_attr",
    meta=unpack_attr_meta,
    python_printer=unpack_attr_printer,
    python_impl=unpack_attr_impl,
)


# NOTE UNPACK_PARAMETER is intended only to be bound to directly, and not called
def unpack_parameter_meta(o: Any, key: str, /) -> Any:
    raise NotImplementedError


def unpack_parameter_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_parameter but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_parameter but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    origin, key = arg_printables
    origin_str = codeutils.prettyprint(origin)
    keystr = codeutils.prettyprint(key)
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {origin_str}.get_parameter({keystr})"


def unpack_parameter_impl(o: Any, key: str, /) -> Any:
    return o.get_parameter(key)


unpack_parameter = make_prim(
    PrimIDs.UNPACK_PARAMETER,
    "unpack_parameter",
    meta=unpack_parameter_meta,
    python_printer=unpack_parameter_printer,
    python_impl=unpack_parameter_impl,
)


# NOTE UNPACK_SUBMODULE is intended only to be bound to directly, and not called
def unpack_submodule_meta(o: Any, key: str, /) -> Any:
    raise NotImplementedError


def unpack_submodule_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_submodule but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_submodule but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    origin, key = arg_printables
    origin_str = codeutils.prettyprint(origin)
    keystr = codeutils.prettyprint(key)
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {origin_str}.get_submodule({keystr})"


def unpack_submodule_impl(o: Any, key: str, /) -> Any:
    return o.get_submodule(key)


unpack_submodule = make_prim(
    PrimIDs.UNPACK_SUBMODULE,
    "unpack_submodule",
    meta=unpack_submodule_meta,
    python_printer=unpack_submodule_printer,
    python_impl=unpack_submodule_impl,
)


# NOTE UNPACK_BUFFER is intended only to be bound to directly, and not called
def unpack_buffer_meta(o: Any, key: str, /) -> Any:
    raise NotImplementedError


def unpack_buffer_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_buffer but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_buffer but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    origin, key = arg_printables
    origin_str = codeutils.prettyprint(origin)
    keystr = codeutils.prettyprint(key)
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {origin_str}.get_buffer({keystr})"


def unpack_buffer_impl(o: Any, key: str, /) -> Any:
    return o.get_buffer(key)


unpack_buffer = make_prim(
    PrimIDs.UNPACK_BUFFER,
    "unpack_buffer",
    meta=unpack_buffer_meta,
    python_printer=unpack_buffer_printer,
    python_impl=unpack_buffer_impl,
)


def pack_list_meta(*args: Any) -> Any:
    def _proxy(x: Any):
        if isinstance(x, Proxy):
            return x
        return proxy(x)

    a = ListProxy([_proxy(x) for x in args])
    return a


def pack_list_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for pack_list but got {kwarg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        isinstance(out_printables, ListProxy),
        lambda: f"Expected out of type ListProxy, but got {type(out_printables)}",
        exception_type=AssertionError,
    )

    l = out_printables.name
    call_str = f"{codeutils.prettyprint(l)}"

    parts = [f"{codeutils.prettyprint(arg, literals_as_underscores=True)}, " for arg in arg_printables]
    final_str = call_str.strip("'") + f" = [{''.join(parts)}]"

    return final_str


def pack_list_impl(*args: Any) -> Any:
    return list(args)


pack_list = make_prim(
    PrimIDs.PACK_LIST,
    "pack_list",
    meta=pack_list_meta,
    python_printer=pack_list_printer,
    python_impl=pack_list_impl,
    tags=(OpTags.DONT_DCE,),
)


# NOTE PACK_BUFFER is intended only to be bound to directly, and not called
def pack_buffer_meta(o: Any, key: Any, value: Any) -> Any:
    raise NotImplementedError


def pack_buffer_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 3,
        lambda: f"Expected three arguments for pack_buffer but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for pack_buffer but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    obj, key, value = arg_printables
    obj_str = codeutils.prettyprint(obj)
    key_str = codeutils.prettyprint(key)
    value_str = codeutils.prettyprint(value)
    return f"{obj_str}.set_buffer({key_str}, {value_str})"


def pack_buffer_impl(o: Any, key: Any, v: Any) -> None:
    # o[key] = v
    return None


pack_buffer = make_prim(
    PrimIDs.PACK_BUFFER,
    "pack_buffer",
    meta=pack_buffer_meta,
    python_printer=pack_buffer_printer,
    python_impl=pack_buffer_impl,
    tags=(OpTags.DONT_DCE,),
)


# NOTE PACK_ATTR is intended only to be bound to directly, and not called
def pack_attr_meta(o: Any, key: Any, value: Any) -> Any:
    raise NotImplementedError


def pack_attr_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 3,
        lambda: f"Expected three arguments for pack_attr but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for pack_attr but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    obj, key, value = arg_printables
    obj_str = codeutils.prettyprint(obj)
    key_str = key
    value_str = codeutils.prettyprint(value)
    return f"{obj_str}.{key_str} = {value_str}"


def pack_attr_impl(o: Any, key: Any, v: Any) -> None:
    o[key] = v
    return None


pack_attr = make_prim(
    PrimIDs.PACK_ATTR,
    "pack_attr",
    meta=pack_attr_meta,
    python_printer=pack_attr_printer,
    python_impl=pack_attr_impl,
    tags=(OpTags.DONT_DCE,),
)


# NOTE PACK_SETITEM is intended only to be bound to directly, and not called
def pack_setitem_meta(o: Any, key: Any, value: Any) -> Any:
    raise NotImplementedError


def pack_setitem_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 3,
        lambda: f"Expected three arguments for pack_setitem but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for pack_setitem but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    obj, key, value = arg_printables
    obj_str = codeutils.prettyprint(obj)
    key_str = codeutils.prettyprint(key)
    value_str = codeutils.prettyprint(value)
    return f"{obj_str}[{key_str}] = {value_str}"


def pack_setitem_impl(o: Any, key: Any, v: Any) -> None:
    o[key] = v
    return None


pack_setitem = make_prim(
    PrimIDs.PACK_SETITEM,
    "pack_setitem",
    meta=pack_setitem_meta,
    python_printer=pack_setitem_printer,
    python_impl=pack_setitem_impl,
    tags=(OpTags.DONT_DCE,),
)


def python_dataclass_new_meta(typ, **kwargs):
    # we are cheating here, but instantiating a dataclass can be wild. typ(**kwargs)
    return AnyProxy(None)


def python_dataclass_new_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    (typ,) = arg_printables
    outstr = codeutils.prettyprint(out_printables, literals_as_underscores=True)
    typ_str = codeutils._generate_dataclass_class_name(typ)
    kwarg_str = ", ".join(f"{k}={codeutils.prettyprint(v)}" for k, v in kwarg_printables.items())
    return f"{outstr} = {typ_str}({kwarg_str})"


def python_dataclass_new_impl(typ, **kwargs):
    return typ(**kwargs)


python_dataclass_new = make_prim(
    PrimIDs.DATACLASS_NEW,
    "python_dataclass_new",
    meta=python_dataclass_new_meta,
    python_printer=python_dataclass_new_printer,
    python_impl=python_dataclass_new_impl,
)


def shape_meta(t: TensorProxy) -> Sequence[int | NumberProxy]:
    return t._shape


shape = make_prim(
    PrimIDs.SHAPE,
    "shape",
    meta=shape_meta,
)


# NOTE UNPACK_GETITEM is intended only to be bound to directly, and not called
def unpack_getitem_meta(o: Any, key: Any) -> Any:
    raise NotImplementedError


def unpack_getitem_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_getitem but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_getitem but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    origin, key = arg_printables
    origin_str = codeutils.prettyprint(origin)
    keystr = codeutils.prettyprint(key)
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {origin_str}[{keystr}]"


def unpack_getitem_impl(o: Any, key: Any) -> Any:
    return o[key]


unpack_getitem = make_prim(
    PrimIDs.UNPACK_GETITEM,
    "unpack_getitem",
    meta=unpack_getitem_meta,
    python_printer=unpack_getitem_printer,
    python_impl=unpack_getitem_impl,
)


# NOTE UNPACK_ITER is intended only to be bound to directly, and not called
def unpack_iter_meta(o: Any, /) -> Any:
    raise NotImplementedError


def unpack_iter_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected one argument for unpack_iter but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_iter but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    (origin,) = arg_printables
    origin_str = codeutils.prettyprint(origin)
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {origin_str}.__iter__()"


def unpack_iter_impl(o: Any, /) -> Any:
    return o.__iter__()


unpack_iter = make_prim(
    PrimIDs.UNPACK_ITER,
    "unpack_iter",
    meta=unpack_iter_meta,
    python_printer=unpack_iter_printer,
    python_impl=unpack_iter_impl,
)


# NOTE UNPACK_NEXT is intended only to be bound to directly, and not called
def unpack_next_meta(o: Any, /) -> Any:
    raise NotImplementedError


def unpack_next_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 1,
        lambda: f"Expected one argument for unpack_next but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_next but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    (origin,) = arg_printables
    origin_str = codeutils.prettyprint(origin)
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {origin_str}.__next__()"


def unpack_next_impl(o: Any, /) -> Any:
    return o.__next__()


unpack_next = make_prim(
    PrimIDs.UNPACK_NEXT,
    "unpack_next",
    meta=unpack_next_meta,
    python_printer=unpack_next_printer,
    python_impl=unpack_next_impl,
)


def unpack_key_meta(d: dict | CollectionProxy, key: Hashable) -> Any:
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


def _unpack_dict_key_meta(d: dict, key: int | str, /) -> Proxy:
    baseutils.check_type(d, dict)
    baseutils.check_type(key, (int, str))

    def _proxy(x: Any):
        if isinstance(x, Proxy):
            return x
        return proxy(x)

    return _proxy(d[key])


def _unpack_dict_key_impl(d: dict, key: int | str, /) -> Any:
    return d[key]


def _unpack_dict_key_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(arg_printables) == 2,
        lambda: f"Expected two arguments for unpack_dict_key but got {arg_printables}",
        exception_type=AssertionError,
    )
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for unpack_dict_key but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    # Converts printables to strings
    d, key = arg_printables
    dstr = codeutils.prettyprint(d)
    keystr = codeutils.prettyprint(key)
    outstr = codeutils.prettyprint(out_printables, with_type=True, literals_as_underscores=True)

    return f"{outstr} = {dstr}[{keystr}]"


unpack_dict_key = make_prim(
    PrimIDs.UNPACK_DICT_KEY,
    "unpack_dict_key",
    meta=_unpack_dict_key_meta,
    python_printer=_unpack_dict_key_printer,
    python_impl=_unpack_dict_key_impl,
)


def unpack_empty_dict_meta(d: dict | CollectionProxy) -> tuple:
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


def unpack_empty_dict_impl(d: dict | CollectionProxy) -> tuple:
    assert len(d) == 0
    return ()


unpack_empty_dict = make_prim(
    PrimIDs.UNPACK_EMPTY_DICT,
    "unpack_empty_dict",
    meta=unpack_empty_dict_meta,
    python_printer=unpack_empty_dict_printer,
    python_impl=unpack_empty_dict_impl,
)


def unpack_dict(d: dict | CollectionProxy) -> tuple[Any, ...]:
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

    return unpack_trivial(x, name=x.name)


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
    "python_print",
    meta=_print_meta,
    python_printer=python_print_printer,
    python_impl=print,
    tags=(OpTags.DONT_DCE,),
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
def _del_impl(x: Any, /) -> None:
    del x


python_del = make_prim(
    PrimIDs.DEL,
    "python_del",
    meta=_del_meta,
    python_printer=del_printer,
    python_impl=_del_impl,
)


def _return_meta(*args) -> None:
    return None


def return_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
):
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for return but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    arg_str = (
        ""
        if (arg_printables is None or len(arg_printables) == 0)
        else ", ".join(codeutils.prettyprint(x) for x in arg_printables)
    )

    return f"return {arg_str}"


def _return_impl(*args) -> None:
    return None


python_return = make_prim(
    PrimIDs.RETURN,
    "python_return",
    meta=_return_meta,
    python_printer=return_printer,
    python_impl=_return_impl,
    tags=(OpTags.DONT_DCE,),
)

#
# Prims related to transforms (like grad)
#


# TODO Review number grad handling with dynamic constraints
def _get_grad_meta(a: Number | NumberProxy | TensorProxy, /) -> Number | TensorProxy:
    utils.check_type(a, (Number, NumberProxy, TensorProxy))

    if isinstance(a, TensorProxy):
        # NOTE: `a` could be a TensorProxy subclass and it's type should be preserved.
        return type(a)(like=a)

    # NOTE a is a Number in this branch
    return numberproxy(pytype(a), 0)


get_grad = make_prim(
    PrimIDs.GET_GRAD,
    "get_grad",
    meta=_get_grad_meta,
)


def _put_grad_meta(grad_for: Number | NumberProxy | TensorProxy, grad: Number | NumberProxy | TensorProxy) -> None:
    utils.check_type(grad_for, (Number, NumberProxy, TensorProxy))
    utils.check_type(grad, (Number, NumberProxy, TensorProxy))

    # Attempts to put a grad for a number or tensor with an exact dtype are ignored
    if dtypes.is_exact_dtype(dtypes.to_dtype(grad_for)):
        return None

    if isinstance(grad, TensorProxy):
        utils.check_type(grad_for, TensorProxy)
        utils.check_same_shape(grad_for, grad)
        utils.check_same_device(grad_for, grad)
        utils.check_same_dtype(grad_for, grad)
    else:
        # NOTE isinstance(grad, (Number, NumberProxy)) == True in this branch
        utils.check_type(grad_for, (Number, NumberProxy))
        # TODO Add number grad support

    return None


# PUT_GRAD is a sink node with side effects that updates Tensor.grad. It needs
# DONT_DCE tag to avoid removal in DCE pass.
put_grad = make_prim(
    PrimIDs.PUT_GRAD,
    "put_grad",
    meta=_put_grad_meta,
    tags=(OpTags.DONT_DCE,),
)

#
# Data movement and transformation prims
#
# TODO create an expected type helper for consistent error formatting
# TODO: consider supporting number subclasses


# TODO Require the datatype of the conversion be constant
def _convert_element_type_meta(a: Number | TensorProxy, /, dtype: type | dtypes.dtype) -> Number | TensorProxy:
    utils.check_type(a, (Number, NumberProxy, TensorProxy))
    utils.check_type(dtype, (type, dtypes.dtype))

    # NOTE Python numbers are constants, and this will return another Python number when given one because
    #   The conversion is constant
    if isinstance(a, (Number, NumberProxy)):
        utils.check(utils.is_numbertype(dtype), lambda: f"Trying to convert a number to non-numbertype object {dtype}")

        if isinstance(a, NumberProxy):
            return numberproxy(dtype, dtype(utils.get_numberlike_value(a)), constraint=a.constraint)

        number_result = dtype(a)
        return number_result

    return TensorProxy(like=a, dtype=dtype)


convert_element_type = make_prim(
    PrimIDs.CONVERT_ELEMENT_TYPE,
    "convert_element_type",
    meta=_convert_element_type_meta,
)


def _device_put_meta(a: TensorProxy, /, device: devices.Device) -> TensorProxy:
    # NOTE The TensorProxy constructor will validate that a is a TensorProxy
    #   and device is a devices.Device
    return TensorProxy(like=a, device=device)


device_put = make_prim(
    PrimIDs.DEVICE_PUT,
    "device_put",
    meta=_device_put_meta,
)


def _numpy_array_to_torch_tensor_meta(a: TensorProxy, /) -> TensorProxy:
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
#   produce boolean results (ALWAYS_BOOL), math.ceil/math.floor produces integer outputs for number inputs while preserves datatype for tensor inputs, and other operations, like abs, map
#   complex numbers to floats (COMPLEX_TO_FLOAT).
#   The ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND enum describes these three behaviors so that
#   elementwise operations can rely on helper functions to implement this behavior.
class ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND(Enum):
    SAME = auto()
    ALWAYS_BOOL = auto()
    INT_FOR_NUMBER = auto()
    COMPLEX_TO_FLOAT = auto()


math_dtypes = dtypes.all_dtypes_and_numbertypes - dtypes.low_precision_dtypes
fp_math_dtypes = math_dtypes - dtypes.exact_dtypes
comparison_dtypes = dtypes.all_dtypes_and_numbertypes - dtypes.complex_dtypes
ceil_floor_math_dtypes = dtypes.float_dtypes | dtypes.all_numbertypes - {complex}

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
    number_type_map: dict[type, type] | None,
):
    def meta(a: Number | TensorProxy, /) -> Number | TensorProxy:
        # Checks that inputs have an expected type
        utils.check_type(a, (TensorProxy, Number, NumberProxy))

        if isinstance(a, (Number, NumberProxy)):
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
            elif output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.INT_FOR_NUMBER:
                output_type = int
            elif output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.COMPLEX_TO_FLOAT:
                if dtypes.is_complex_dtype(typ):
                    output_type = float
                else:
                    output_type = typ
            else:
                utils.check(False, lambda: f"Unknown {output_dtype_kind=}")

            if val is None or number_fn is None:
                utils.check(
                    isinstance(a, NumberProxy),
                    lambda: f"Trying to call an elementwise unary operation {name} on a number, but the operation is not eagerly defined",
                )
                return numberproxy(output_type, None, a.constraint)

            # need to cast val to python_type in order to properly propagate output dtype.
            value = number_fn(typ(val))
            utils.check(
                type(value) is output_type,
                lambda: f"Unexpected number output type {type(value)}, expected {output_type}, for input type {typ} (value={val})",
            )

            # Only returns a proxy if the input is a proxy
            if isinstance(a, NumberProxy):
                return numberproxy(type(value), value, a.constraint)
            return value

        # NOTE a is a TensorProxy
        utils.check(
            not numbers_only,
            lambda: f"Trying to call a primitive ({name}) that only supports numbers with a tensor input",
        )

        # Checks that dtype is supported
        utils.check(a.dtype in supported_input_dtypes, lambda: f"Unsupported input dtype {a.dtype}")

        if (
            output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME
            or output_dtype_kind == ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.INT_FOR_NUMBER
        ):
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
    number_fn: Callable | None = None,
    python_printer: Callable = default_python_printer,
    supported_input_dtypes=dtypes.all_dtypes_and_numbertypes,
    output_dtype_kind: ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND = ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.SAME,
    numbers_only: bool = False,
    number_type_map: dict[type, type] | None = None,
    method_name: None | str = None,
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
        method_name=method_name,
    )


py_abs = _make_elementwise_unary_prim(
    PrimIDs.PY_ABS,
    "py_abs",
    method_name="abs",
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

digamma = _make_elementwise_unary_prim(
    PrimIDs.DIGAMMA,
    "digamma",
    supported_input_dtypes=fp_math_dtypes,
)


asin = _make_elementwise_unary_prim(
    PrimIDs.ASIN,
    "asin",
    number_fn=math.asin,
    supported_input_dtypes=fp_math_dtypes,
)

asinh = _make_elementwise_unary_prim(
    PrimIDs.ASINH,
    "asinh",
    number_fn=math.asinh,
    supported_input_dtypes=fp_math_dtypes,
)

atan = _make_elementwise_unary_prim(
    PrimIDs.ATAN,
    "atan",
    number_fn=math.atan,
    supported_input_dtypes=fp_math_dtypes,
)

atanh = _make_elementwise_unary_prim(
    PrimIDs.ATANH,
    "atanh",
    number_fn=math.atanh,
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
    supported_input_dtypes=ceil_floor_math_dtypes,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.INT_FOR_NUMBER,
)

cos = _make_elementwise_unary_prim(
    PrimIDs.COS,
    "cos",
    number_fn=math.cos,
    supported_input_dtypes=fp_math_dtypes,
)

cosh = _make_elementwise_unary_prim(
    PrimIDs.COSH,
    "cosh",
    number_fn=math.cosh,
    supported_input_dtypes=fp_math_dtypes,
)

erf = _make_elementwise_unary_prim(
    PrimIDs.ERF,
    "erf",
    number_fn=math.erf,
    supported_input_dtypes=fp_math_dtypes,
)

erfc = _make_elementwise_unary_prim(
    PrimIDs.ERFC,
    "erfc",
    number_fn=math.erfc,
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
    number_fn=math.exp,
    supported_input_dtypes=fp_math_dtypes,
)


def _exp2_number(a: Number) -> Number:
    if hasattr(math, "exp2"):
        return math.exp2(a)
    return 2**a


exp2 = _make_elementwise_unary_prim(
    PrimIDs.EXP2,
    "exp2",
    number_fn=_exp2_number,
    supported_input_dtypes=fp_math_dtypes,
)

expm1 = _make_elementwise_unary_prim(
    PrimIDs.EXPM1,
    "expm1",
    number_fn=math.expm1,
    supported_input_dtypes=fp_math_dtypes,
)

# TODO Should floor accept float16 and bfloat16 types?
# NOTE This preserves the input's dtype for tensors, but is consistent
#   with math.floor for numbers (always returning an integer)
floor = _make_elementwise_unary_prim(
    PrimIDs.FLOOR,
    "floor",
    number_fn=math.floor,
    supported_input_dtypes=ceil_floor_math_dtypes,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.INT_FOR_NUMBER,
)


def frexp_meta(a: TensorProxy, /) -> (TensorProxy, TensorProxy):
    utils.check_type(a, TensorProxy)
    return TensorProxy(like=a), TensorProxy(like=a, dtype=dtypes.int32)


frexp = make_prim(PrimIDs.FREXP, "frexp", meta=frexp_meta)

isfinite = _make_elementwise_unary_prim(
    PrimIDs.ISFINITE,
    "isfinite",
    supported_input_dtypes=dtypes.inexact_dtypes,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
)

lgamma = _make_elementwise_unary_prim(
    PrimIDs.LGAMMA,
    "lgamma",
    number_fn=math.lgamma,
    supported_input_dtypes=fp_math_dtypes,
)

log = _make_elementwise_unary_prim(
    PrimIDs.LOG,
    "log",
    number_fn=math.log,
    supported_input_dtypes=fp_math_dtypes,
)

log10 = _make_elementwise_unary_prim(
    PrimIDs.LOG10,
    "log10",
    number_fn=math.log10,
    supported_input_dtypes=fp_math_dtypes,
)

log1p = _make_elementwise_unary_prim(
    PrimIDs.LOG1P,
    "log1p",
    number_fn=math.log1p,
    supported_input_dtypes=fp_math_dtypes,
)

log2 = _make_elementwise_unary_prim(
    PrimIDs.LOG2,
    "log2",
    number_fn=math.log2,
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
    method_name="neg",
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
    number_fn=math.sin,
    supported_input_dtypes=fp_math_dtypes,
)

sinh = _make_elementwise_unary_prim(
    PrimIDs.SINH,
    "sinh",
    number_fn=math.sinh,
    supported_input_dtypes=fp_math_dtypes,
)

sqrt = _make_elementwise_unary_prim(
    PrimIDs.SQRT,
    "sqrt",
    number_fn=math.sqrt,
    supported_input_dtypes=fp_math_dtypes,
)

tan = _make_elementwise_unary_prim(
    PrimIDs.TAN,
    "tan",
    number_fn=math.tan,
    supported_input_dtypes=fp_math_dtypes,
)

tanh = _make_elementwise_unary_prim(
    PrimIDs.TANH,
    "tanh",
    number_fn=math.tanh,
    supported_input_dtypes=fp_math_dtypes,
)

# NOTE This trunc preserves the dtype of its input
trunc = _make_elementwise_unary_prim(
    PrimIDs.TRUNC,
    "trunc",
    supported_input_dtypes=fp_math_dtypes,
    number_fn=math.trunc,
)


def real_meta(a: complex | TensorProxy) -> float | TensorProxy:
    # Validates inputs
    utils.check_type(a, (TensorProxy, complex))
    dtyp = dtypes.to_dtype(a, true_dtype=True)
    utils.check(
        dtypes.is_complex_dtype(dtyp),
        lambda: f"real expected a complex tensor or number, but receive a tensor or number with dtype {dtyp}",
    )
    output_dtype = dtypes.corresponding_real_dtype(dtyp)

    if isinstance(a, complex):
        result = utils.get_numberlike_value(a).real
        return numberproxy(float, result)

    # NOTE a is a TensorProxy
    return TensorProxy(like=a, dtype=output_dtype)


real = make_prim(
    PrimIDs.REAL,
    "real",
    meta=real_meta,
)


def imag_meta(a: complex | TensorProxy) -> float | TensorProxy:
    utils.check_type(a, (TensorProxy, complex))
    dtyp = dtypes.to_dtype(a, true_dtype=True)
    utils.check(
        dtypes.is_complex_dtype(dtyp),
        lambda: f"imag expected a complex tensor or number, but receive a tensor or number with dtype {dtyp}",
    )
    output_dtype = dtypes.corresponding_real_dtype(dtyp)

    if isinstance(a, complex):
        result = utils.get_numberlike_value(a).imag
        return numberproxy(float, result)

    return TensorProxy(like=a, dtype=output_dtype)


imag = make_prim(
    PrimIDs.IMAG,
    "imag",
    meta=imag_meta,
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
    numbers_only: bool,
    output_dtype_kind,
    supported_input_dtypes,
):
    def meta(
        a: Number | TensorProxy,
        b: Number | TensorProxy,
        /,
    ) -> Number | TensorProxy:
        # Checks that inputs have an expected type
        utils.check_type(a, (TensorProxy, Number, NumberProxy))
        utils.check_type(b, (TensorProxy, Number, NumberProxy))

        # Checks same dtype
        numbertype, dtype = utils.check_same_dtype(a, b)

        # Checks that dtype is supported
        utils.check(
            numbertype is None or numbertype in supported_input_dtypes, lambda: f"Unsupported number type {numbertype}"
        )
        utils.check(dtype is None or dtype in supported_input_dtypes, lambda: f"Unsupported input dtype {dtype}")

        # Special-cases number x number inputs
        if isinstance(a, (Number, NumberProxy)) and isinstance(b, (Number, NumberProxy)):
            aval, bval = utils.get_numberlike_value(a), utils.get_numberlike_value(b)

            # Handles the case where a number has an indeterminate value, or the operation has
            #   no number handler, by returning another indeterminate value
            if aval is None or bval is None or number_fn is None:
                utils.check(
                    isinstance(a, NumberProxy) or isinstance(b, NumberProxy),
                    lambda: f"Trying to call an elementwise binary operation {name} on two numbers, but the operation is not eagerly defined",
                )
                return numberproxy(numbertype, None, constraint=utils.resolve_constraints(a, b))

            value = number_fn(aval, bval)
            # Only returns a NumberProxy if at least one input is a number proxy
            if isinstance(a, NumberProxy) or isinstance(b, NumberProxy):
                return numberproxy(type(value), value, constraint=utils.resolve_constraints(a, b))
            return value

        else:
            # NOTE a or b is a TensorProxy
            utils.check(
                not numbers_only,
                lambda: f"Trying to call a primitive ({name}) that only supports numbers with a tensor input",
            )

        # Checks same shape
        # NOTE: this doesn't verify a common shape if one or more inputs is a number
        utils.check_same_shape(a, b)

        # Checks same device
        utils.check_same_device(a, b)

        # If both inputs are tensors, choose the one that is not a CPU scalar tensor.
        if isinstance(a, TensorProxy) and isinstance(b, TensorProxy):
            tensor = a if (isinstance(a, TensorProxy) and not utils.is_cpu_scalar_tensor(a)) else b
        else:
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
    method_name: None | str = None,
    numbers_only: bool = False,
    constraint_function: None | Callable = None,
):
    return make_prim(
        id,
        name,
        meta=_elementwise_binary_meta_factory(
            name=name,
            number_fn=number_fn,
            numbers_only=numbers_only,
            output_dtype_kind=output_dtype_kind,
            supported_input_dtypes=supported_input_dtypes,
        ),
        python_printer=python_printer,
        method_name=method_name,
    )


add = _make_elementwise_binary_prim(
    PrimIDs.ADD,
    "add",
    number_fn=operator.add,
    supported_input_dtypes=math_dtypes,
    method_name="add",
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

# We currently do not support floordiv on tensors.
py_floordiv = _make_elementwise_binary_prim(
    PrimIDs.PY_FLOORDIV,
    "py_floordiv",
    method_name="floor_divide",
    number_fn=operator.floordiv,
    numbers_only=True,
    supported_input_dtypes={bool, int, float},
)

eq = _make_elementwise_binary_prim(
    PrimIDs.EQ,
    "eq",
    method_name="eq",
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
    method_name="ge",
    number_fn=operator.ge,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
    supported_input_dtypes=comparison_dtypes,
)

gt = _make_elementwise_binary_prim(
    PrimIDs.GT,
    "gt",
    method_name="gt",
    number_fn=operator.gt,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
    supported_input_dtypes=comparison_dtypes,
)

le = _make_elementwise_binary_prim(
    PrimIDs.LE,
    "le",
    method_name="le",
    number_fn=operator.le,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
    supported_input_dtypes=comparison_dtypes,
)

lt = _make_elementwise_binary_prim(
    PrimIDs.LT,
    "lt",
    method_name="lt",
    number_fn=operator.lt,
    output_dtype_kind=ELEMENTWISE_PRIM_OUTPUT_DTYPE_KIND.ALWAYS_BOOL,
    supported_input_dtypes=comparison_dtypes,
)

maximum = _make_elementwise_binary_prim(PrimIDs.MAXIMUM, "maximum", supported_input_dtypes=comparison_dtypes)

minimum = _make_elementwise_binary_prim(PrimIDs.MINIMUM, "minimum", supported_input_dtypes=comparison_dtypes)

mul = _make_elementwise_binary_prim(
    PrimIDs.MUL,
    "mul",
    method_name="mul",
    number_fn=operator.mul,
    supported_input_dtypes=math_dtypes,
)

ne = _make_elementwise_binary_prim(
    PrimIDs.NE,
    "ne",
    method_name="ne",
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
    method_name="mod",
    number_fn=operator.mod,
    supported_input_dtypes=math_dtypes,
)

sub = _make_elementwise_binary_prim(
    PrimIDs.SUB,
    "sub",
    method_name="sub",
    number_fn=operator.sub,
    supported_input_dtypes=math_dtypes,
)

zeta = _make_elementwise_binary_prim(
    PrimIDs.ZETA,
    "zeta",
    supported_input_dtypes=fp_math_dtypes,
)


bitwise_left_shift = _make_elementwise_binary_prim(
    PrimIDs.BITWISE_LEFT_SHIFT,
    "bitwise_left_shift",
    supported_input_dtypes=dtypes.integer_dtypes,
)


bitwise_right_shift = _make_elementwise_binary_prim(
    PrimIDs.BITWISE_RIGHT_SHIFT,
    "bitwise_right_shift",
    supported_input_dtypes=dtypes.integer_dtypes,
)

#
# Elementwise ternary prims
#


def _lerp_meta(start: TensorProxy, end: TensorProxy, weight: Number | TensorProxy, /) -> TensorProxy:
    utils.check_type(start, TensorProxy)
    utils.check_type(end, TensorProxy)
    utils.check_type(weight, (TensorProxy, Number, NumberProxy))

    numbertype, dtype = utils.check_same_dtype(start, end, weight)

    utils.check(numbertype is None or numbertype in fp_math_dtypes, lambda: f"Unsupported number type {numbertype}")
    utils.check(dtype is None or dtype in fp_math_dtypes, lambda: f"Unsupported input dtype {dtype}")

    utils.check_same_shape(start, end, weight)
    utils.check_same_device(start, end, weight)

    return TensorProxy(like=start, dtype=dtype)


lerp = make_prim(
    PrimIDs.LERP,
    "lerp",
    method_name="lerp",
    meta=_lerp_meta,
)


# TODO Restore Number x Number x Number support
def _where_meta(pred: Number | TensorProxy, a: Number | TensorProxy, b: Number | TensorProxy, /) -> TensorProxy:
    # Checks types
    # NOTE pred must be a bool tensor or bool (this is checked later)
    utils.check_type(pred, (TensorProxy, Number, NumberProxy))
    utils.check_type(a, (TensorProxy, Number, NumberProxy))
    utils.check_type(b, (TensorProxy, Number, NumberProxy))

    if (
        isinstance(pred, (Number, NumberProxy))
        and isinstance(a, (Number, NumberProxy))
        and isinstance(b, (Number, NumberProxy))
    ):
        raise NotImplementedError

    # Checks pred dtype (bool or bool tensor)
    if isinstance(pred, (Number, NumberProxy)):
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
    devices_ = tuple(x.device for x in (pred, a, b) if isinstance(x, TensorProxy) and not utils.is_cpu_scalar_tensor(x))
    if len(devices_) > 0:
        resultdevice = devices_[0]

    # Determines result dtype
    numbertype, tensordtype = utils.check_same_dtype(a, b)
    dtype = tensordtype if tensordtype is not None else numbertype

    # Checks shapes
    utils.check_same_shape(pred, a, b)

    # Determines output shape
    # NOTE Assumes at least one of pred, a, and b is a TensorProxy because of prior check for Number x Number x Number
    shapes = tuple(x.shape for x in (pred, a, b) if isinstance(x, TensorProxy) and not utils.is_cpu_scalar_tensor(x))
    if not shapes:
        shapes = (pred.shape,)
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


def _full_meta(
    shape: tuple[int, ...], fill_value: Number, *, device: devices.Device, dtype: dtypes.dtype
) -> TensorProxy:
    # Checks inputs
    utils.check_type(fill_value, (Number, NumberProxy))

    # Ensures the requested fill_value can be safely cast to the dtype
    fill_value_dtype = dtypes.to_dtype(fill_value)
    utils.check(
        utils.can_safe_cast_number_to(fill_value, dtype),
        lambda: f"Can't safely cast fill_value of numbertype {fill_value_dtype} to dtype {dtype}",
    )

    utils.check_type(shape, tuple)
    utils.check_valid_shape(shape)
    return TensorProxy(shape=shape, device=device, dtype=dtype)


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
    utils.check_type(length, (Number, NumberProxy))
    utils.check_type(start, (Number, NumberProxy))
    utils.check_type(step, (Number, NumberProxy))

    # Checks input properties
    utils.check(utils.is_exact_dtype(dtype), lambda: f"dtype={dtype} was not an exact dtype")
    utils.check(not utils.is_boolean_dtype(dtype), lambda: f"dtype={dtype} was not a non-boolean dtype")
    utils.check(length >= 0, lambda: f"length={length} was not weakly positive")

    shape = () if length == 0 else (length,)

    return TensorProxy(shape=shape, device=device, dtype=dtype)


iota = make_prim(PrimIDs.IOTA, "iota", meta=_iota_meta)


# TODO Should the uniform prim include minval maxval or always be [0, 1)?
def _uniform_meta(
    shape: Sequence[int], minval: Number, maxval: Number, *, device: devices.Device, dtype: dtypes.dtype
) -> TensorProxy:
    # Checks inputs
    utils.check_type(minval, (Number, NumberProxy))
    utils.check_type(maxval, (Number, NumberProxy))
    utils.check_type(device, devices.Device)
    utils.check_type(dtype, dtypes.dtype)

    return TensorProxy(shape=shape, device=device, dtype=dtype)


uniform = make_prim(
    PrimIDs.UNIFORM,
    "uniform",
    meta=_uniform_meta,
    tags=(OpTags.RANDOM_OP,),
)


# Follow the nvFuser way to get the seed/offset used in nvFuser's philox RNG from the Torch's RNG state and
# update the Torch's RNG state
def _get_and_update_rng_state_meta(
    seed: IntegerProxy | int | None, offset: IntegerProxy | int | None, device: devices.Device
) -> tuple[IntegerProxy, IntegerProxy]:
    seed_is_none = seed is None
    offset_is_none = offset is None
    utils.check(
        not (seed_is_none ^ offset_is_none),
        lambda: f"seed and offset must be given in pair (they are both None only used on first use of get_and_update_rng_statebut), but got {seed}, {offset}",
    )
    if not seed_is_none:
        utils.check_type(seed, (IntegerProxy, int))
    if not offset_is_none:
        utils.check_type(offset, (IntegerProxy, int))
    utils.check(
        device.devicetype is devices.DeviceType.CUDA,
        lambda: "get_and_update_rng_state is supported for CUDA only",
        exception_type=NotImplementedError,
    )
    return numberproxy(int, None), numberproxy(int, None)


get_and_update_rng_state = make_prim(
    PrimIDs.GET_AND_UPDATE_RNG_STATE,
    "get_and_update_rng_state",
    meta=_get_and_update_rng_state_meta,
    tags=(OpTags.RANDOM_OP,),  # RANDOM_OP here is a short hand for "uses / modifies the random state"
)


def _uniform_philox_meta(
    shape: Sequence[int],
    minval: float,
    maxval: float,
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
    seed: int | IntegerProxy | TensorProxy,
    offset: int | IntegerProxy | TensorProxy,
) -> TensorProxy:
    # Checks inputs
    utils.check_type(minval, float)
    utils.check_type(maxval, float)
    utils.check_type(device, devices.Device)
    utils.check_type(dtype, dtypes.dtype)
    utils.check_type(seed, (int, TensorProxy, IntegerProxy))
    utils.check_type(offset, (int, TensorProxy, IntegerProxy))
    utils.check(
        isinstance(seed, (int, IntegerProxy)) or seed.dtype is dtypes.int64,
        lambda: f"Expected {seed=} to be an integer or an int64 tensor",
    )
    utils.check(
        isinstance(offset, (int, IntegerProxy)) or seed.dtype is dtypes.int64,
        lambda: f"Expected {offset=} to be an integer or an int64 tensor",
    )
    utils.check(minval < maxval, lambda: f"`minval` must be less than `maxval` but {minval=}, {maxval=}")
    utils.check_valid_shape(shape)

    utils.check_same_shape(shape, seed, offset)
    utils.check_same_device(device, seed, offset)

    return TensorProxy(shape=shape, device=device, dtype=dtype)


uniform_philox = make_prim(
    PrimIDs.UNIFORM_PHILOX,
    "uniform_philox",
    meta=_uniform_philox_meta,
)


def _randn_meta(
    shape: tuple[int, ...],
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
):
    utils.check_type(device, devices.Device)
    utils.check_type(dtype, dtypes.dtype)
    utils.check_type(shape, tuple)
    utils.check_valid_shape(shape)
    return TensorProxy(shape=shape, device=device, dtype=dtype)


randn = make_prim(PrimIDs.RANDN, "randn", meta=_randn_meta)


def _randint_meta(
    low: int,
    high: int,
    shape: tuple[int, ...],
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
):
    utils.check_type(low, int)
    utils.check_type(high, int)
    utils.check(low < high, lambda: f"`low` must be less than `high` but {low=}, {high=}")
    utils.check_type(device, devices.Device)
    utils.check_type(dtype, dtypes.dtype)
    utils.check_type(shape, tuple)
    utils.check_valid_shape(shape)
    return TensorProxy(shape=shape, device=device, dtype=dtype)


randint = make_prim(PrimIDs.RANDINT, "randint", meta=_randint_meta)


def _empty_meta(
    shape: tuple[int, ...],
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
):
    utils.check_type(device, devices.Device)
    utils.check_type(dtype, dtypes.dtype)
    utils.check_type(shape, tuple)
    utils.check_valid_shape(shape)
    return TensorProxy(shape=shape, device=device, dtype=dtype)


empty = make_prim(PrimIDs.EMPTY, "empty", meta=_empty_meta)


# TODO(crcrpar): Cover `memory_format` kwarg
def _clone_meta(a: TensorProxy, **kwargs) -> TensorProxy:
    return TensorProxy(like=a)


clone = make_prim(PrimIDs.CLONE, "clone", meta=_clone_meta)


def _update_aliases_meta(aliases: tuple[TensorProxy], /) -> tuple[TensorProxy]:
    return tuple(TensorProxy(like=a, requires_grad=a.requires_grad) for a in aliases)


update_aliases = make_prim(
    PrimIDs.UPDATE_ALIASES,
    "update_aliases",
    meta=_update_aliases_meta,
)


# Prim to construct a Tensor from sequence/nested sequence of Numbers.
def _tensor_from_sequence_meta(
    seq: Sequence[Number | Sequence], *, dtype: None | dtypes.dtype, device: devices.Device
) -> TensorProxy:
    utils.check_type(dtype, (dtypes.dtype, NoneType))
    utils.check_type(device, devices.Device)
    utils.check_type(seq, Sequence)

    # str is treated as Sequence but we don't want to treat it as such.
    def is_sequence_not_str(obj):
        return isinstance(obj, Sequence) and not isinstance(obj, str)

    # Compute shape and validate that we have homogenous sequence.
    shape = []
    sequences = seq
    dim_len = len(sequences)
    shape.append(dim_len)
    while dim_len > 0 and is_sequence_not_str(first := sequences[0]):
        dim_len = len(first)
        next_sequences = []
        for s in sequences:
            # Check for homogenous sequence.
            utils.check(len(s) == dim_len, lambda: f"Expected seq of len={dim_len} at dim {len(shape)}")
            next_sequences.extend(s)

        shape.append(dim_len)
        sequences = next_sequences

    # Infer the dtype
    types = set()
    for element in sequences:
        utils.check(
            isinstance(element, (bool, int, float, complex)),
            lambda: f"Expected sequences of numbers, but found type {type(element)} when constructing a tensor from a sequence",
            ValueError,
        )
        types.add(pytype(element))

    # NOTE: inferred_dtype will stay None if sequence was empty.
    inferred_dtype = None
    if complex in types:
        inferred_dtype = complex
    elif float in types:
        inferred_dtype = float
    elif int in types:
        inferred_dtype = int
    elif bool in types:
        inferred_dtype = bool

    # user specified a dtype and we could infer the dtype
    if dtype is not None and inferred_dtype is not None:
        # verify that the inferred dtype can be safely cast to user requested dtype.
        # NOTE: We can't use `utils.can_safe_cast_to` as it is more fine-grained and it differentiates between
        #       unsigned and signed integers.
        #       But for Python Number types we can only infer `bool`, `int`, `float`, `complex`.
        utils.check(
            utils.can_safe_cast_number_to(inferred_dtype(0), dtype),
            lambda: f"Can't safely cast sequence with numbertype {inferred_dtype} to dtype {dtype}",
        )
    # user specified a dtype and we couldn't infer the dtype (empty sequence)
    elif dtype is not None and inferred_dtype is None:
        pass
    else:  # user didn't specify the dtype.
        # use inferred_dtype if available else default to float
        # NOTE: In future, we should rely on something like [thunder/torch].get_default_dtype.
        dtype = inferred_dtype if inferred_dtype is not None else float

    return TensorProxy(shape=shape, device=device, dtype=dtype)


# Prim to construct a Tensor from sequence/nested sequence of Numbers.
tensor_from_sequence = make_prim(PrimIDs.TENSOR_FROM_SEQUENCE, "tensor_from_sequence", meta=_tensor_from_sequence_meta)


def _multinomial_meta(
    input: TensorProxy,
    num_samples: int,
    replacement: bool,
    seed: int | None = None,
) -> TensorProxy:
    utils.check_type(input, TensorProxy)
    utils.check_type(num_samples, (int, IntegerProxy))
    utils.check(pytype(replacement) is bool, f"Expected boolean {replacement=}")
    utils.check_type(seed, (int, type(None)))

    utils.check(
        input.numel != 0,
        lambda: "Expected probability weights to be non-empty",
    )
    utils.check(
        0 < input.ndim <= 2,
        lambda: f"Expected {input.ndim=} to be 1 or 2",
    )
    utils.check(
        isinstance(input.dtype, dtypes.floating),
        lambda: f"Expected {input.dtype=} to be of a floating type",
    )
    n_categories = input.shape[-1]

    utils.check(num_samples > 0, lambda: f"Expected {num_samples=} to be greater than 0")
    utils.check(
        bool(replacement) or (num_samples <= n_categories),
        lambda: f"Cannot sample {num_samples} > {input.shape[-1]=} without replacement",
    )

    # TODO: PyTorch restriction. Could be removed once different ref is used.
    max_n_categories = 2**24
    utils.check(n_categories <= max_n_categories, lambda: f"Expected {n_categories=} to not exceed {max_n_categories}")

    if seed is not None:
        seed_lo = -0x8000_0000_0000_0000
        seed_hi = 0xFFFF_FFFF_FFFF_FFFF
        utils.check(seed_lo <= seed <= seed_hi, lambda: f"Expected {seed_lo} <= {seed=} <= {seed_hi}")

    shape = (*input.shape[:-1], num_samples)

    return TensorProxy(shape=shape, device=input.device, dtype=dtypes.int64)


multinomial = make_prim(
    PrimIDs.MULTINOMIAL,
    "multinomial",
    meta=_multinomial_meta,
    tags=(OpTags.RANDOM_OP,),
)


#
# Shape prims
#


# NOTE broadcast_dimensions is a sequence with length equal to a.shape (which is not necessarily equal to shape)
def broadcast_in_dim_meta(a: TensorProxy, /, shape: Sequence[int], broadcast_dimensions: Sequence[int]) -> TensorProxy:
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
    tags=(OpTags.SHAPE_OP,),
)


def cat_meta(tensors: list[TensorProxy], /, dim: int) -> TensorProxy:
    utils.check(len(tensors) > 0, lambda: "Cat expects a non-empty list of tensors")
    utils.check_types(tensors, TensorProxy)
    utils.check_same_device(*tensors)
    utils.check_same_dtype(*tensors)

    ndim = tensors[0].ndim
    utils.check(
        dim >= -ndim and dim < ndim,
        lambda: f"Expected dimension in inclusive range of {-ndim} and {ndim - 1}: got {dim}.",
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
                f"Expected size {sd} but got size {sad} for tensor number {i + 1} in the list.",
            )
        shape[dim] = shape[dim] + ai.shape[dim]

    return TensorProxy(like=tensors[0], shape=shape)


cat = make_prim(
    PrimIDs.CAT,
    "cat",
    meta=cat_meta,
)


def item_meta(a: TensorProxy, /) -> NumberProxy:
    utils.check_type(a, TensorProxy)

    utils.check(a.numel == 1, lambda: f"Expects input with numel=1 but got {a.numel=} instead", ValueError)

    numbertype = dtypes.dtype_to_numbertype(a.dtype)
    return numberproxy(numbertype, value=None)


item = make_prim(PrimIDs.ITEM, "item", meta=item_meta, tags=(OpTags.DEVICE_SYNC_OP,))


def flip_meta(a: TensorProxy, /, dims: Sequence[int]) -> TensorProxy:
    # Check types
    utils.check_type(a, TensorProxy)
    utils.check_type(dims, Sequence)
    utils.check(
        all(
            (
                0 <= d < a.ndim
                if isinstance(d, (int, IntegerProxy))
                else isinstance(d, IntegerProxy) and 0 <= pyval(d) < a.ndim
            )
            for d in dims
        ),
        lambda: f"Expected {dims=} to be a sequence of integers in [0, {a.ndim} - 1]",
    )

    utils.check_no_duplicates(dims)

    return TensorProxy(like=a)


flip = make_prim(PrimIDs.FLIP, "flip", meta=flip_meta)


def pad_meta(a: TensorProxy, /, padding_value: Number, padding_config: Sequence[tuple[int, int, int]]) -> TensorProxy:
    # Validates types
    utils.check_type(a, TensorProxy)
    utils.check_type(padding_value, (Number, NumberProxy))
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


def reshape_meta(a: TensorProxy, /, shape: tuple[int, NumberProxy, ...]) -> TensorProxy:
    # Validates inputs
    utils.check_type(a, TensorProxy)
    utils.check_valid_shape(shape)
    # Requires `shape` to a tuple so CSE can hash it properly. `list` is not a
    # hashable type. See #1789.
    utils.check_type(shape, tuple)

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
    tags=(OpTags.SHAPE_OP,),
)


# TODO Be clear about what the prim can handle and what it can't
# TODO Validate that start_indices, end_indices, and strides are sequences of ints
# TODO Update the prim to not accept optional strides
# NOTE The stride parameter here refers to the stride of the slice, not the tensor's strides
def slice_meta(
    a: TensorProxy, /, start_indices: Sequence[int], end_indices: Sequence[int], strides: None | Sequence[int] = None
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

        new_shape.append((stop - start + stride - 1) // stride)

    return TensorProxy(like=a, shape=new_shape)


# NOTE: slice is named "slice_prim" and not "slice" because it conflicts with Python's "slice" builtin
slice_prim = make_prim(PrimIDs.SLICE, "slice_prim", meta=slice_meta, tags=(OpTags.SHAPE_OP,))


def squeeze_meta(a: TensorProxy, /, dims: tuple[int, ...]) -> TensorProxy:
    # Checks types
    utils.check_type(a, TensorProxy)
    utils.check_type(dims, tuple)

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


squeeze = make_prim(PrimIDs.SQUEEZE, "squeeze", meta=squeeze_meta, tags=(OpTags.SHAPE_OP,))


def take_meta(a: TensorProxy, /, index: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(dim, (int, IntegerProxy))
    utils.check_same_device(a, index)
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"index dtype={index.dtype} was not an integer dtype")
    utils.check(index.ndim <= 1, lambda: f"Expected index to a 1-D or 0-D tensor, but index.ndim={index.ndim}!")
    utils.validate_idx(a.ndim, dim)

    utils.check(
        not (a.shape[dim] == 0 and index.numel > 0),
        lambda: "Attempting to index a 0-length dimension {dim=} with a non-empty index",
    )

    l = index.shape[0] if index.ndim == 1 else 1
    new_shape = a.shape[:dim] + (l,) + a.shape[dim + 1 :]

    return TensorProxy(like=a, shape=new_shape)


take = make_prim(PrimIDs.TAKE, "take", meta=take_meta)


# TODO We should be consistent using 'index' or 'indices'
def index_add_meta(a: TensorProxy, /, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(value, TensorProxy)
    utils.check_type(dim, (int, IntegerProxy))
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


def index_copy_meta(a: TensorProxy, /, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    utils.check(
        dtypes.to_dtype(index) is dtypes.int64,
        lambda: "index_copy: only indices of type int64 are supported",
    )
    return index_add_meta(a, index, value, dim)


index_copy = make_prim(PrimIDs.INDEX_COPY, "index_copy", meta=index_add_meta)


def index_put_meta(
    a: TensorProxy, /, indices: Sequence[TensorProxy], values: TensorProxy, accumulate: bool
) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(values, TensorProxy)
    utils.check_type(indices, Sequence)
    utils.check_types(indices, TensorProxy)
    utils.check(pytype(accumulate) is bool, f"Expected boolean {accumulate=}")

    supported_index_dtype = {dtypes.int32, dtypes.int64}
    for index in indices:
        utils.check(dtypes.to_dtype(index) in supported_index_dtype, lambda: f"Unsupported input dtype {index.dtype}")

    utils.check_same_device(a, values, *indices)
    utils.check_same_dtype(a, values)

    utils.check(
        len(indices) <= a.ndim, lambda: f"Too many indices for tensor of dimension {a.ndim} (got {len(indices)} )"
    )

    if indices:
        utils.check_same_shape(*indices)
        expanded_shape = indices[0].shape + a.shape[len(indices) :]
        utils.check(
            utils.same_shape(values.shape, expanded_shape),
            lambda: f"Expected 'values' to have the shape of {expanded_shape} (got {values.shape} )",
        )
    return TensorProxy(like=a)


index_put = make_prim(PrimIDs.INDEX_PUT, "index_put", meta=index_put_meta)


def take_along_axis_meta(a: TensorProxy, /, index: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(dim, (int, IntegerProxy))
    utils.check_same_device(a, index)
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"{index.dtype=} was not an integer dtype")
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


def copy_with_setitem_meta(a: TensorProxy, index, value: TensorProxy) -> TensorProxy:
    # TODO: port checks from clang, currently there  because of the utilities they need
    return TensorProxy(like=a)


copy_with_setitem = make_prim(PrimIDs.COPY_WITH_SETITEM, "copy_with_setitem", meta=copy_with_setitem_meta)


def gather_meta(a: TensorProxy, /, index: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(dim, (int, IntegerProxy))
    utils.check_same_device(a, index)
    utils.check(utils.is_integer_dtype(index.dtype), lambda: f"index dtype={index.dtype} was not an integer dtype")
    utils.check(
        index.ndim == a.ndim, lambda: f"Expected index (rank={index.ndim}) to have the same rank as a (rank={a.ndim})"
    )
    utils.validate_idx(a.ndim, dim)

    for idx, l in enumerate(index.shape):
        if idx != dim:
            utils.check(
                index.shape[idx] <= a.shape[idx],
                lambda: f"Expected 'index' size on all dimensions to be <= 'a', except `dim`. Found dim {idx}, where 'index' has {index.shape[idx]} and 'a' has {a.shape[idx]}",
            )
    return TensorProxy(like=a, shape=index.shape)


gather = make_prim(PrimIDs.GATHER, "gather", meta=gather_meta)


def scatter_add_meta(a: TensorProxy, /, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(index, TensorProxy)
    utils.check_type(value, TensorProxy)
    utils.check_type(dim, (int, IntegerProxy))
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


def scatter_meta(a: TensorProxy, /, index: TensorProxy, src: TensorProxy | Number, dim: int) -> TensorProxy:
    utils.check_type(src, (TensorProxy, Number, NumberProxy))

    if isinstance(src, TensorProxy):
        return scatter_add_meta(a, index, src, dim)

    # scatter_add_meta reuse when `src` is a scalar,
    # which is being replaced with a TensorProxy(like=a) and
    # shape[dim] = index.shape[dim
    utils.validate_idx(a.ndim, dim)
    utils.check(
        index.ndim == a.ndim, lambda: f"Expected index (rank={index.ndim}) to have the same rank as a (rank={a.ndim})"
    )

    dummy_src_shape = list(a.shape)
    dummy_src_shape[dim] = index.shape[dim]
    dummy_src = TensorProxy(like=a, shape=dummy_src_shape)

    return scatter_add_meta(a, index, dummy_src, dim)


scatter = make_prim(PrimIDs.SCATTER, "scatter", meta=scatter_meta)


def topk_meta(a: TensorProxy, /, k: int, dim: int, largest: Number, sorted: Number) -> (TensorProxy, TensorProxy):
    utils.check_type(a, TensorProxy)
    utils.check_type(k, (int, IntegerProxy))
    utils.check_type(dim, (int, IntegerProxy))
    utils.check(pytype(largest) is bool, lambda: f"Expected {largest=} to be a boolean value")
    utils.check(pytype(sorted) is bool, lambda: f"Expected {sorted=} to be a boolean value")

    utils.check(k >= 0 and k <= (a.shape[dim] if a.ndim > 0 else 1), lambda: f"selected index {k=} is out of range")

    new_shape = a.shape
    if a.ndim > 0:
        new_shape = list(new_shape)
        new_shape[dim] = k

    return TensorProxy(like=a, shape=new_shape), TensorProxy(like=a, shape=new_shape, dtype=dtypes.int64)


topk = make_prim(PrimIDs.TOPK, "topk", meta=topk_meta, tags=(OpTags.REDUCTION_OP,))


def sort_meta(a: TensorProxy, /, dim: int, descending: Number, sorted: Number) -> (TensorProxy, TensorProxy):
    utils.check_type(a, TensorProxy)
    utils.check_type(dim, (int, IntegerProxy))
    utils.check(pytype(descending) is bool, lambda: f"Expected {descending=} to be a boolean type")
    utils.check(pytype(sorted) is bool, lambda: f"Expected {sorted=} to be a boolean type")

    return TensorProxy(like=a), TensorProxy(like=a, dtype=dtypes.int64)


sort = make_prim(PrimIDs.SORT, "sort", meta=sort_meta)


def _grouped_mm_meta(a: TensorProxy, b: TensorProxy, offsets: TensorProxy) -> TensorProxy:
    """Meta function for _grouped_mm primitive.

    Accepts the following shape combinations:
    1. (m, k) x (k, n) -> (groups, m, n)
    2. (groups, m, k) x (k, n) -> (m, n)
    3. (m, k) x (groups, k, n) -> (m, n)

    Args:
        a: Input tensor of shape (groups, m, k) or (m, k)
        b: Input tensor of shape (groups, k, n) or (k, n)
        offsets: Offset tensor of shape (groups,)

    Returns:
        TensorProxy with shape (groups, m, n) or (m, n)
    """
    # Validate types
    utils.check_type(a, TensorProxy)
    utils.check_type(b, TensorProxy)
    utils.check_type(offsets, TensorProxy)

    # Accept 2D or 3D tensors
    utils.check(a.ndim in (2, 3), lambda: f"Expected a to have 2 or 3 dimensions, got {a.ndim}")
    utils.check(b.ndim in (2, 3), lambda: f"Expected b to have 2 or 3 dimensions, got {b.ndim}")

    utils.check(offsets.ndim == 1, lambda: f"`offsets` must be a vector, got shape {offsets.shape}")
    if a.ndim == 2 and b.ndim == 2:
        utils.check(a.shape[1] == b.shape[0], lambda: f"Inner dimension mismatch: {a.shape} vs {b.shape}")
        out_shape = (offsets.shape[0], a.shape[0], b.shape[1])
    if a.ndim == 3 and b.ndim == 2:
        utils.check(a.shape[2] == b.shape[1], lambda: f"Inner dimension mismatch: {a.shape} vs {b.shape}")
        utils.check(a.shape[0] == offsets.shape[0], lambda: f"Group count mismatch: {a.shape} vs {offsets.shape}")
        out_shape = (a.shape[1], b.shape[1])
    elif a.ndim == 2 and b.ndim == 3:
        utils.check(a.shape[1] == b.shape[1], lambda: f"Inner dimension mismatch: {a.shape} vs {b.shape}")
        utils.check(b.shape[0] == offsets.shape[0], lambda: f"Group count mismatch: {b.shape} vs {offsets.shape}")
        out_shape = (a.shape[0], b.shape[2])
    else:
        utils.check(False, lambda: f"Unexpected shape combination: {a.shape} and {b.shape}")

    utils.check_same_dtype(a, b)
    utils.check(a.dtype in dtypes.float_math_dtypes, lambda: f"`a` must be 16-bit float or higher, got {a.dtype}")
    utils.check(utils.is_integer_dtype(offsets.dtype), lambda: f"`offsets` must be integers, got {offsets.dtype}")

    utils.check_same_device(a, b)

    return TensorProxy(like=a, shape=out_shape)


_grouped_mm = make_prim(
    PrimIDs._GROUPED_MM,
    "_grouped_mm",
    meta=_grouped_mm_meta,
)


def transpose_meta(a: TensorProxy, /, permutation: tuple[int, ...]) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(permutation, tuple)
    utils.check(
        a.ndim == len(permutation),
        lambda: f"Expected the length ({len(permutation)}) of the permutation={permutation} to be the number of dimensions ({a.ndim}) of a={a}",
    )
    utils.check_valid_permutation(a.ndim, permutation)

    new_shape = [0] * a.ndim
    for idx, dim in enumerate(permutation):
        new_shape[idx] = a.shape[dim]

    return TensorProxy(like=a, shape=new_shape)


transpose = make_prim(PrimIDs.TRANSPOSE, "transpose", meta=transpose_meta, tags=(OpTags.SHAPE_OP,))


view = make_prim(PrimIDs.VIEW, "view", meta=reshape_meta, tags=(OpTags.SHAPE_OP,))


def shallow_copy_meta(a: TensorProxy, /) -> TensorProxy:
    return TensorProxy(like=a)


shallow_copy = make_prim(PrimIDs.SHALLOW_COPY, "shallow_copy", meta=shallow_copy_meta, tags=(OpTags.SHAPE_OP,))


def unfold_meta(a: TensorProxy, /, dim: int, size: int, step: int) -> TensorProxy:
    dim = utils.canonicalize_dim(a.ndim, dim)
    max_size = 1 if a.ndim == 0 else a.shape[dim]

    utils.check(
        size <= max_size, lambda: f"Maximum size for tensor at dimension {dim} is {max_size} but size is {size}"
    )
    utils.check(size >= 0, lambda: f"Size is {size} but must be >= 0")
    utils.check(step > 0, lambda: f"Step is {step} but must be > 0")

    shape = list(a.shape)
    shape.append(size)
    shape[dim] = (shape[dim] - size) // step + 1

    return TensorProxy(like=a, shape=shape)


unfold = make_prim(PrimIDs.UNFOLD, "unfold", meta=unfold_meta, tags=(OpTags.SHAPE_OP,))

#
# Memory format prims (Experimental)
#


def stride_order_meta(a: TensorProxy, /, order: Sequence[int]) -> TensorProxy:
    # Validates inputs
    utils.check_type(a, TensorProxy)
    utils.check_valid_permutation(a.ndim, order)

    return TensorProxy(like=a)


# TODO Consider a more general stride manipulation primitive, like PyTorch's
#   as_strided or set_strided operations
# See clang.stride_order for this prim's documentation
stride_order = make_prim(PrimIDs.STRIDE_ORDER, "stride_order", meta=stride_order_meta, tags=(OpTags.SHAPE_OP,))

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


def _reduction_meta(a: TensorProxy, /, dims: Sequence[int]) -> TensorProxy:
    """Meta function for single output reduction operations."""

    # Validates types
    utils.check_type(a, TensorProxy)
    utils.check_type(dims, Sequence)

    utils.check(
        len(dims) > 0 or len(a.shape) == 0,
        lambda: f"Expected {dims=} to be a non-empty sequence when the tensor's shape {a.shape} is non-empty",
    )

    output_shape = _compute_reduction_output_shape(a.shape, dims)

    return TensorProxy(like=a, shape=output_shape)


# TODO: review if reduction meta is OK for amax
amax = make_prim(PrimIDs.AMAX, "amax", meta=_reduction_meta, tags=(OpTags.REDUCTION_OP,))

amin = make_prim(PrimIDs.AMIN, "amin", meta=_reduction_meta, tags=(OpTags.REDUCTION_OP,))

prod = make_prim(PrimIDs.PROD, "prod", meta=_reduction_meta, tags=(OpTags.REDUCTION_OP,))

sum = make_prim(PrimIDs.SUM, "sum", meta=_reduction_meta, tags=(OpTags.REDUCTION_OP,))


# Note: We have seperate meta function for `argmin/argmax` instead of
#       reusing `_reduction_meta` as these operations expect Optional[int] for `dim`
#       and return output with integer dtype.
#
#       When `dim=None`, this operation returns linear index of `max/min` value
#       on the flattened version of the array.
#       These operators don't support reducing over multiple dimensions.
#       We can support multiple dimensions by flattening the passed dimension
#       and returning the linear index of `max/min`. Other approach could be
#       to return co-ordinate indices which can actually be used for indexing.
#       For now, we stick to reducing over only 1 dimension like other frameworks.
def _argmin_argmax_meta(a: TensorProxy, /, dim: int | None) -> TensorProxy:
    """Meta function for argmax and argmin."""

    # Validates types
    utils.check_type(a, TensorProxy)
    utils.check_type(dim, (int, IntegerProxy, NoneType))

    if a.numel == 0:
        utils.check(dim is not None, lambda: "Expected reduction dim to be specified for a.numel() == 0.")

    if dim is not None and a.ndim > 0:
        utils.check(a.shape[dim], lambda: f"Expected reduction dim {dim} to have non-zero size.")

    # `_compute_reduction_output_shape` expects sequence of ints.
    dims = range(a.ndim) if dim is None else (dim,)

    output_shape = _compute_reduction_output_shape(a.shape, dims)

    return TensorProxy(like=a, shape=output_shape, dtype=dtypes.int64)


argmax = make_prim(PrimIDs.ARGMAX, "argmax", meta=_argmin_argmax_meta, tags=(OpTags.REDUCTION_OP,))

argmin = make_prim(PrimIDs.ARGMIN, "argmin", meta=_argmin_argmax_meta, tags=(OpTags.REDUCTION_OP,))


# NOTE var doesn't use _reduction_meta because it has the correction parameter
# TODO Add output_dtype?
# TODO Check that dims is a sequence of integers
def _var_meta(a: TensorProxy, /, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    # Checks input types
    utils.check_type(a, TensorProxy)
    utils.check_type(dims, Sequence)
    utils.check_type(correction, (Number, NumberProxy))

    output_dtype = None
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.true_dtype)
    else:
        output_dtype = a.true_dtype

    reduced: TensorProxy = _reduction_meta(a, dims)
    return TensorProxy(like=reduced, dtype=output_dtype)


def _var_mean_meta(a: TensorProxy, /, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    output_dtype = None
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.true_dtype)
    else:
        output_dtype = a.true_dtype

    var_result: TensorProxy = _reduction_meta(a, dims)
    mean_result: TensorProxy = _reduction_meta(a, dims)

    var: TensorProxy = TensorProxy(like=var_result, dtype=output_dtype)
    mean: TensorProxy = TensorProxy(like=mean_result, dtype=a.true_dtype)

    return (var, mean)


var = make_prim(PrimIDs.VAR, "var", meta=_var_meta, tags=(OpTags.REDUCTION_OP,))
var_mean = make_prim(PrimIDs.VAR_MEAN, "var_mean", meta=_var_mean_meta, tags=(OpTags.REDUCTION_OP,))


def _std_meta(a: TensorProxy, /, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(dims, Sequence)
    utils.check_type(correction, (Number, NumberProxy))

    output_dtype = None
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.true_dtype)
    else:
        output_dtype = a.true_dtype

    reduced: TensorProxy = _reduction_meta(a, dims)
    return TensorProxy(like=reduced, dtype=output_dtype)


std = make_prim(PrimIDs.STD, "std", meta=_std_meta, tags=(OpTags.REDUCTION_OP,))


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
    utils.check_same_device(a, w)

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


linear = make_prim(
    PrimIDs.LINEAR, "linear", meta=linear_meta, tags=(OpTags.MATMUL_OP, OpTags.DONT_AUTO_RECOMPUTE_IN_BACKWARD)
)


def matmul_meta(a: TensorProxy, b: TensorProxy, /) -> TensorProxy:
    # Checks types
    utils.check(isinstance(a, TensorProxy), lambda: f"a={a} was not a TensorProxy")
    utils.check(isinstance(b, TensorProxy), lambda: f"b={b} was not a TensorProxy")

    if a.ndim < 1 or b.ndim < 1:
        raise NotImplementedError

    utils.check_same_device(a, b)

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
        lambda: f"Expected the batch dimensions of a {a.shape[:-2]} and the batch dimensions of b {b.shape[:-2]} to be the same",
    )

    utils.check(
        a.shape[-1] == b.shape[-2],
        lambda: f"Expected the the last two dimensions of a ({a.shape[-2:]}) be matrix multipiable with the last two dimensions of b ({b.shape[-2:]})",
    )

    shape = list(a.shape[:-2])
    shape.append(a.shape[-2])
    shape.append(b.shape[-1])

    return TensorProxy(like=a, shape=shape)


matmul = make_prim(
    PrimIDs.MATMUL, "matmul", meta=matmul_meta, tags=(OpTags.MATMUL_OP, OpTags.DONT_AUTO_RECOMPUTE_IN_BACKWARD)
)

#
# NN prims
#


# TODO: model transpose and layout (channels last and alike)
def convolution_meta(
    a: TensorProxy,
    /,
    weight: TensorProxy,
    bias: TensorProxy | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: Number,
    output_padding: Sequence[int],
    groups: int,
) -> TensorProxy:
    # Validate types {
    utils.check_type(a, TensorProxy)
    utils.check_type(weight, TensorProxy)
    utils.check_type(bias, (TensorProxy, type(None)))
    utils.check_same_dtype(a, weight, *([bias] if bias is not None else []))
    utils.check(pytype(transposed) is bool, lambda: f"Expected {transposed=} to be a boolean value")
    utils.check_type(groups, (int, IntegerProxy))
    # }

    # Validate device {
    utils.check_same_device(a, weight, *([bias] if bias is not None else []))
    # }

    # TODO: add support for these later {
    utils.check(transposed == 0, lambda: f"{transposed=} is not supported", exception_type=NotImplementedError)
    # }

    # Validate inputs {
    utils.check(groups > 0, lambda: f"{groups=} should be greater than 0")

    # Check the ranks of `a` and `weight`
    features_rank = weight.ndim - 2
    utils.check(features_rank > 0, lambda: f"{weight.ndim=} should be at least 3")
    utils.check(a.ndim == weight.ndim, lambda: f"{a.ndim=} should be equal to {weight.ndim}")

    # Check out_channels/in_channels/groups/bias relationships
    minibatch, in_channels, *features_size = a.shape
    out_channels, in_channels_grouped, *kernel_size = weight.shape

    # Check features_size and kernel_size contain no empty dims.
    # This limitation comes from PyTorch.
    # It is possible to relax this requirement on input and
    # error only if the padded input dim is empty.
    utils.check(
        all(fs != 0 for fs in features_size),
        lambda: f"Input's shape {a.shape=} can be zero only "
        "in the batch (i.e. a.shape[0]) and/or "
        "in the channel dimension (i.e. a.shape[1])",
    )
    utils.check(
        all(ks != 0 for ks in kernel_size),
        lambda: f"{kernel_size=} (i.e. weight.shape[2:]) should not contain zero dimensions",
    )

    utils.check(
        in_channels_grouped * groups == in_channels,
        lambda: f"{weight.shape[1]=} should be equal to "
        f"(in_channels / groups)={in_channels // groups} "
        f"(i.e. {a.shape[1]=} / {groups=})",
    )
    utils.check(
        out_channels % groups == 0, lambda: f"out_channels (i.e. {weight.shape[0]=}) should be divisible by {groups=}"
    )
    utils.check(
        bias is None or (bias.ndim == 1 and bias.numel == out_channels),
        lambda: f"{bias.ndim=} should be 1 and {bias.numel=} should match out_channels, (i.e. {weight.shape[0]=})",
    )

    # Check sequences (stride, padding, dilation, output_padding)
    # for correct type, length, and that the elements are from proper domains.
    def check_sequence(seq, seq_str_name, rank, *, min_val):
        utils.check_type(seq, Sequence)

        utils.check(len(seq) == 1 or len(seq) == rank, lambda: f"len({seq_str_name}) should be either 1 or {rank}")

        # Check all elements are >= min_val
        for i, e in enumerate(seq):
            utils.check(
                isinstance(e, (int, IntegerProxy)) and e >= min_val,
                lambda: f"all elements in {seq_str_name} should be integers at least {min_val}, "
                f"but {seq_str_name}[{i}]={seq[i]} does not satisfy these requirements",
            )

    # stride and dilation should be at least 1
    check_sequence(stride, "stride", features_rank, min_val=1)
    check_sequence(dilation, "dilation", features_rank, min_val=1)
    # paddings should be non-negative
    check_sequence(padding, "padding", features_rank, min_val=0)
    check_sequence(output_padding, "output_padding", features_rank, min_val=0)

    # Expand sequences to features_rank len if needed.
    def maybe_expand_seq(seq, ndim):
        if isinstance(seq, (int, IntegerProxy)):
            return (seq,) * ndim
        elif len(seq) == 1:
            return (seq[0],) * ndim
        else:
            return tuple(seq)

    # Let's expand sequence inputs to match features_rank
    # for easier homogeneous processing
    stride = maybe_expand_seq(stride, features_rank)
    dilation = maybe_expand_seq(dilation, features_rank)
    padding = maybe_expand_seq(padding, features_rank)
    output_padding = maybe_expand_seq(output_padding, features_rank)

    # Check a.shape[2:]/weight.shape[2:] consistency
    # in the presence of padding/dilation, i.e.
    # having padded input shape smaller than the dilated kernel shape
    # at any dimension is an error.
    for dim, (f, p, k, d) in enumerate(zip(features_size, padding, kernel_size, dilation)):
        padded_a_dim = f + 2 * p
        dilated_weight_dim = d * (k - 1) + 1
        tensor_dim = dim + 2
        utils.check(
            padded_a_dim >= dilated_weight_dim,
            lambda: f"Inconsistent shapes at dimension {tensor_dim} between `a` and `weight`. "
            f"The padded `a` dimension {tensor_dim} is equal to {padded_a_dim} "
            f"(i.e. a.shape[{tensor_dim}] + 2 * padding[{dim}] = "
            f"{a.shape[tensor_dim]} + 2 * {padding[dim]}) "
            "and should be greater or equal to the dilated `weight` shape at the same dimension "
            f"which is equal to {dilated_weight_dim} "
            f"(i.e. dilation[{dim}] * (weight.shape[{tensor_dim}] - 1) + 1 = "
            f"{dilation[dim]} * ({weight.shape[tensor_dim]} - 1) + 1)",
        )
    # }

    # Output shape {
    output_shape = [minibatch, out_channels if in_channels > 0 else 0]
    for f, p, k, d, s, op in zip(features_size, padding, kernel_size, dilation, stride, output_padding):
        # Padded features
        pf = f + 2 * p
        # Dilated kernel
        dk = d * (k - 1) + 1
        dim_len = (pf - dk) // s + 1 + op
        output_shape.append(dim_len)

    return TensorProxy(like=a, shape=output_shape)
    # }


convolution = make_prim(
    PrimIDs.CONVOLUTION,
    "convolution",
    meta=convolution_meta,
)


def embedding_meta(
    a: TensorProxy, /, weight, *, padding_idx=-1, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
) -> TensorProxy:
    # TODO: canonicalize and validating padding idx with weight.shape[0]

    if max_norm is not None:
        raise NotImplementedError

    utils.check(a.dtype == dtypes.int64, lambda: f"Expected a.dtype={a.dtype} to be int64")
    utils.check(weight.ndim == 2, lambda: f"Expected weight (weight.shape={weight.shape} to be a matrix)")

    shape = list(a.shape)
    shape.append(weight.shape[1])

    return TensorProxy(like=weight, shape=shape)


embedding = make_prim(PrimIDs.EMBEDDING, "embedding", meta=embedding_meta)


# TODO Update this so it's not a prim
# TODO Add annotations
# TODO Once we have fusible index_put we can implement it using primitives
# For now we just use the PyTorch implementation
def embedding_backward_meta(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    shape = (num_weights, grad.shape[-1])
    return TensorProxy(shape=shape, device=grad.device, dtype=grad.dtype)


embedding_backward = make_prim(PrimIDs.EMBEDDING_BACKWARD, "embedding_backward", meta=embedding_backward_meta)


def copy__meta(
    copy_from: TensorProxy,
    copy_to: TensorProxy,
    *,
    grad_enabled: bool,
):
    utils.check_type(copy_from, TensorProxy)
    utils.check_type(copy_to, TensorProxy)
    utils.check_same_device(copy_from, copy_to)
    utils.check_same_shape(copy_from, copy_to)
    utils.check_same_dtype(copy_from, copy_to)
    return TensorProxy(like=copy_to)


copy_ = make_prim(PrimIDs.COPY_, "copy_", meta=copy__meta, tags=(OpTags.DONT_DCE,))


def bitcast_meta(
    src: TensorProxy,
    dtype: dtypes.dtype,
) -> TensorProxy:
    shape = list(src.shape)
    src_itemsize = src.dtype.bytes
    dst_itemsize = dtype.bytes
    if src_itemsize != dst_itemsize:
        factor = dst_itemsize / src_itemsize
        if factor > 1:
            utils.check(
                shape[-1] > factor and shape[-1] % factor == 0,
                lambda: f"{src.shape[-1]=} is not divisible by {factor=}. Viewing {src.dtype=} as {dtype=}",
            )
        shape[-1] = int(shape[-1] / factor)
    return TensorProxy(shape=tuple(shape), device=src.device, dtype=dtype)


bitcast = make_prim(PrimIDs.BITCAST, "bitcast", meta=bitcast_meta)


def sink_meta(*args, **kwargs):
    return


# TODO do we want another tag to remove this after prologue is constructed?
sink = make_prim(PrimIDs.SINK, "sink", meta=sink_meta, tags=(OpTags.DONT_DCE,))
