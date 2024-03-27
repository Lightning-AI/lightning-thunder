from functools import partial
from typing import Any
from types import EllipsisType, NoneType
from collections.abc import Callable
from collections.abc import Sequence
from numbers import Number
import optree

from thunder.core.options import (
    INTERPRETATION_OPTIONS,
    CACHE_OPTIONS,
    SHARP_EDGES_OPTIONS,
)
from thunder.core.trace import (
    TraceCtx,
    tracectx,
)

import thunder.core.prims as prims

from thunder.extend import Executor
from thunder.core.compile_data import get_cache_option
from thunder.core.langctxs import LanguageContext
from thunder.core.baseutils import is_base_printable
from thunder.core.codeutils import get_siginfo, SigInfo, is_simple_printable_collection
from thunder.core.proxies import (
    proxy,
    Proxy,
    TensorProxy,
    pyval,
    pytype,
    NumberProxy,
    StringProxy,
    IntegerProxy,
    FloatProxy,
    ComplexProxy,
    TupleProxy,
    ListProxy,
    DictProxy,
    AnyProxy,
)

import thunder.clang as clang
import thunder

# NOTE This import is intentionally pytorch so that it thunder.torch doesn't import this
import torch as pytorch


def _eager_validate_tensor(p: TensorProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(f"Trying to unpack a number with symbolic values, but this is not supported yet")

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_tensor_shape_and_metadata(p)

    return ([p], [p])


def _eager_unpack_tensor(
    t: pytorch.Tensor, /, name: None | str, *, co: CACHE_OPTIONS
) -> tuple[TensorProxy, TensorProxy]:
    p = proxy(t, name=name)
    return p, p


def _eager_validate_literal_like(p: AnyProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(
            f"Trying to unpack a literal-like value of type {pytype(p)} with symbolic values, but this is not supported yet"
        )

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_literal_like(p, pyval(p))

    return ([], [pyval(p)])


def _eager_unpack_literal_like(x: Any, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[AnyProxy, None]:
    p = proxy(x, name=name)
    return p, x


def _eager_validate_none(p: AnyProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(f"Trying to unpack a None with symbolic values, but this is not supported yet")

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_none(p)

    return ([], [None])


def _eager_unpack_none(n: None, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[AnyProxy, None]:
    assert n is None
    p = proxy(None, name=name)
    return p, None


def _eager_validate_number(p: NumberProxy, /, *, co: CACHE_OPTIONS):
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(f"Trying to unpack a number with symbolic values, but this is not supported yet")

    # When not using symbolic values, numbers are compile-time constants, so an actual
    #   Python number is used when interpreting the function, and no number is passed
    #   from the prologue to the computation in the eventual thunder program
    val = pyval(p)

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_number_type_and_value(p, val)

    return ([], [val])


def _eager_unpack_number(num: Number, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[NumberProxy, Number]:
    p = proxy(num, name=name)
    return p, num


def _eager_validate_string(p: StringProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        raise NotImplementedError(f"Trying to unpack a string with symbolic values, but this is not supported yet")

    # When not using symbolic values, strings are compile-time constants, so an actual
    #   Python string is used when interpreting the function, and no string is passed
    #   from the prologue to the computation in the eventual thunder program
    val = pyval(p)

    if co is CACHE_OPTIONS.CONSTANT_VALUES:
        clang.check_string_value(p, val)

    return ([], [val])


def _eager_unpack_string(
    s: str,
    /,
    name: None | str,
    *,
    co: CACHE_OPTIONS,
) -> tuple[StringProxy, str]:
    p = proxy(s, name=name)
    return p, s


# NOTE When unpacking a tuple...
#   - the values in the TupleProxy are the interpreter values
#   - the interpreter is given the tuple, the computation is given the tuple and a flat list of its proxied elements
#   - non-tuple values within the tuple are temporarily assigned proxies by clang.unpack_tuple so they can be
#       validated
def _eager_validate_tuple(p: TupleProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    unpacked = clang.unpack_tuple(p)

    computation_args = [p]

    clang.check_instance(p, (tuple, pytorch.Size))
    if len(p) == 0:
        clang.check_empty(p)

    for x in unpacked:
        cargs, iargs = _eager_validate(x, co=co)
        computation_args.extend(cargs)

    return (computation_args, [p])


def _eager_unpack_tuple(tup: tuple, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[TupleProxy, TupleProxy]:
    unpack = partial(_eager_unpack, name=None, co=co)

    values = []
    for x in tup:
        p, a = unpack(x)
        values.append(a)

    p = proxy(tuple(values), name=name)
    return p, p


def _eager_validate_list(p: ListProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    unpacked = clang.unpack_list(p)

    computation_args = [p]

    clang.check_type(p, list)
    if len(p) == 0:
        clang.check_empty(p)

    for x in unpacked:
        cargs, iargs = _eager_validate(x, co=co)
        computation_args.extend(cargs)

    return (computation_args, [p])


def _eager_unpack_list(lst: list, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[TupleProxy, TupleProxy]:
    unpack = partial(_eager_unpack, name=None, co=co)

    values = []
    for x in lst:
        p, a = unpack(x)
        values.append(a)

    p = proxy(list(values), name=name)
    return p, p


def _eager_validate_dict(p: DictProxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    clang.check_type(p, dict)

    if len(p) == 0:
        clang.check_empty(p)

    computation_args = [p]
    for k, v in p.items():
        pv = clang.unpack_dict_key(p, k)
        cargs, iargs = _eager_validate(pv, co=co)
        computation_args.extend(cargs)

    return (computation_args, [p])


def _eager_unpack_dict(d: dict, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[TupleProxy, TupleProxy]:
    unpack = partial(_eager_unpack, name=None, co=co)
    proxied = {}
    for k, v in d.items():
        if not isinstance(k, (int, str)):
            raise ValueError(f"Unsupported input dict key type {type(k)}. Supported types are int and str.")

        vp, a = unpack(v)
        proxied[k] = a

    p = proxy(proxied, name=name)
    return p, p


def _eager_validate_any(p: Proxy, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    typ: type = pytype(p)
    if typ is NoneType:
        return _eager_validate_none(p, co=co)
    if typ in (pytorch.dtype, pytorch.device, slice, EllipsisType):
        return _eager_validate_literal_like(p, co=co)

    raise NotImplementedError("Trying to validate an object with type {typ}, but this is not implemented")


_type_to_unpack_map: dict[type, Callable] = {
    pytorch.Tensor: _eager_unpack_tensor,
    bool: _eager_unpack_number,
    int: _eager_unpack_number,
    float: _eager_unpack_number,
    complex: _eager_unpack_number,
    str: _eager_unpack_string,
    tuple: _eager_unpack_tuple,
    pytorch.Size: _eager_unpack_tuple,
    list: _eager_unpack_list,
    dict: _eager_unpack_dict,
    slice: _eager_unpack_literal_like,
    EllipsisType: _eager_unpack_literal_like,
    NoneType: _eager_unpack_none,
    pytorch.dtype: _eager_unpack_literal_like,
    pytorch.device: _eager_unpack_literal_like,
}

_type_to_validation_map: dict[type, Callable] = {
    TensorProxy: _eager_validate_tensor,
    IntegerProxy: _eager_validate_number,
    FloatProxy: _eager_validate_number,
    ComplexProxy: _eager_validate_number,
    StringProxy: _eager_validate_string,
    TupleProxy: _eager_validate_tuple,
    ListProxy: _eager_validate_list,
    DictProxy: _eager_validate_dict,
    AnyProxy: _eager_validate_any,
}


def _eager_validate(x: Any, /, *, co: CACHE_OPTIONS) -> tuple[list, list]:
    typ: type = type(x)
    unpack_fn = _type_to_validation_map.get(typ, None)
    if unpack_fn is None:
        raise ValueError(f"Cannot validate object of type {typ}. Please file an issue requesting support.")

    return unpack_fn(x, co=co)


def _eager_unpack(x: Any, /, name: None | str, *, co: CACHE_OPTIONS) -> tuple[Proxy, Any]:
    typ: type = type(x)
    unpack_fn = _type_to_unpack_map.get(typ, None)
    if unpack_fn is None:
        raise ValueError(f"Cannot unpack object of type {typ}. Please file an issue requesting support.")

    return unpack_fn(x, name, co=co)


# A helper for "eager unpacking" interpreters that eagerly unpack their arguments as inputs
# An interpreter must do two things:
#   1) Create a prologue function with the same signature as the original function that
#       acquires all supported inputs and validates them according to the caching option
#   2) Creates a computation function that accepts the output of the prologue function and
#       returns what the original function did
def _eager_unpacking_interpreter(
    interpreter: Callable, fn: Callable, args, kwargs, /, *, interpreter_name: str
) -> tuple[TraceCtx, TraceCtx]:
    # Unpacks the inputs
    si: SigInfo = get_siginfo(fn, args, kwargs)

    prologue_trc: TraceCtx = TraceCtx(si.unwrapped_fn)
    computation_trc: TraceCtx = TraceCtx()

    # Constructs the prologue trace (which just trivially unpacks the tensor arguments for now)
    # TODO RC1 Remove the no_grad and no_autocast context managers from this trace
    # TODO RC1 Provide a mechanism to add context managers to the prologue and computation functions
    # TODO RC1 Don't always import torch in traces (particularly the prologue trace)
    csi = SigInfo("computation")
    csi.args = []
    prologue_args = []  # Arguments to the prologue
    prologue_kwargs = {}  # Kwargs to the prologue
    computation_args = []  # Arguments to the computation
    interpretation_args = []  # Arguments to interpret with
    interpretation_kwargs = {}  # Kwargs to interpret with
    co: CACHE_OPTIONS = get_cache_option()
    with tracectx(prologue_trc):
        # Unpacks args
        for name, x in si.args:
            p: Proxy
            p, _ = _eager_unpack(x, name, co=co)
            prims.unpack_trivial(p)
            cargs, iargs = _eager_validate(p, co=co)
            computation_args.extend(cargs)

            prologue_args.append(p)
            interpretation_args.extend(iargs)

        # Unpacks varargs (if present)
        # NOTE varargs must follow other positional args
        if si.varargs is not None:
            name, x = si.varargs

            p: Proxy
            p, _ = _eager_unpack(x, name, co=co)
            prims.unpack_trivial(p)
            cargs, iargs = _eager_validate(p, co=co)
            computation_args.extend(cargs)

            prologue_args.append(p)
            (iarg,) = iargs
            interpretation_args.extend(iarg)

        # Unpacks kwargs
        for name, x in si.kwargs.items():
            p: Proxy
            p, _ = _eager_unpack(x, name, co=co)
            prims.unpack_trivial(p)
            cargs, iargs = _eager_validate(p, co=co)
            computation_args.extend(cargs)

            prologue_kwargs[name] = p
            (iarg,) = iargs
            interpretation_kwargs[name] = iarg

        if si.varkwargs is not None:
            name, x = si.varkwargs

            p: Proxy
            p, _ = _eager_unpack(x, name, co=co)
            prims.unpack_trivial(p)
            cargs, iargs = _eager_validate(p, co=co)
            computation_args.extend(cargs)

            prologue_kwargs[name] = p
            (iarg,) = iargs

            for k, v in iarg.items():
                interpretation_kwargs[k] = v

    prologue_trc.args = prologue_args
    prologue_trc.kwargs = prologue_kwargs

    # Constructs the computation trace
    # TODO RC1 Only unpack what's used in the computation
    with tracectx(computation_trc):
        p: Proxy
        for p in computation_args:
            prims.unpack_trivial(p)
            csi.args.append((p.name, None))
            computation_trc.add_name(p.name)

        result = interpreter(si.unwrapped_fn)(*interpretation_args, **interpretation_kwargs)

        # Validates that the returned items are proxies or printable values
        def leaf_test(x: Any) -> bool:
            if isinstance(x, Proxy):
                return True
            if is_base_printable(x):
                return True
            if is_simple_printable_collection(x):
                return False

            raise RuntimeError(
                f"Trying to return object of type {type(x)}, but only proxies, strings, torch.device objects, numbers, tuples, lists, and dicts can be returned."
            )

        optree.tree_flatten(result, is_leaf=leaf_test)

        prims.python_return(result)

    # Creates hand-off from prologue to computation
    with tracectx(prologue_trc):
        prims.python_return(tuple(computation_args))

    # Constructs the computation trace's signature
    computation_trc._siginfo = csi
    computation_trc.args = computation_args

    return prologue_trc, computation_trc


# Translates the Python function a thunder program using the Python interpreter
def _python_interpreter(
    fn: Callable, args, kwargs, /, *, sharp_edges: SHARP_EDGES_OPTIONS
) -> tuple[TraceCtx, TraceCtx]:
    if sharp_edges is not SHARP_EDGES_OPTIONS.ALLOW:
        raise ValueError(
            f"Detecting sharp edges is not supported when using the Python interpreter. To detect sharp edges use another interpretation option."
        )

    def _interpreter(fn_):
        return fn_

    return _eager_unpacking_interpreter(_interpreter, fn, args, kwargs, interpreter_name="Python")


# Translates the Python function to a thunder program using the thunder interpreter
def _translate_functions_interpreter(
    fn: Callable, args, kwargs, /, *, sharp_edges: SHARP_EDGES_OPTIONS
) -> tuple[TraceCtx, TraceCtx]:
    from thunder.core.jit_ext import minimal_thunder_jit

    pjit = partial(minimal_thunder_jit, sharp_edges=sharp_edges)
    return _eager_unpacking_interpreter(pjit, fn, args, kwargs, interpreter_name="translate functions")


# note: keep this roughly in sync with thunder.jit
def jit(
    fn: Callable,
    /,
    *,
    langctx: None | str | Any | LanguageContext = None,
    executors: None | Sequence[Executor] = None,
    sharp_edges: None | SHARP_EDGES_OPTIONS | str = None,
    interpretation: None | INTERPRETATION_OPTIONS | str = None,
    cache: None | CACHE_OPTIONS | str = None,
    disable_torch_autograd: bool = False,  # TODO Revisit this UX for RC1
    **compile_options,  # TODO RC1 Make this explicit -- dict of options
) -> Callable:
    """Just-in-time compile a function.

    Args:
        fn: A function to compile.
    Keyword Args:
        langctx: the language context, which language / library to emulate. default: "torch" for PyTorch compatibility.
        executors: list of executors to use. Defaults to the executors returned by `thunder.get_default_executors()` and always amened by `thunder.get_always_executors()`.
                   You can get a list of all available executors with `thunder.get_all_executors()`.
        sharp_edges: sharp edge detection action. What to do when thunder detects a construct that is likely to lead to errors. Can be ``"allow"``, ``"warn"``, ``"error"``. Defaults to ``"allow"``.
        cache: caching mode. default: ``"constant values"```

               - ``"no caching"`` - disable caching and always recompute,
               - ``"constant values"`` - require Tensors to be of the same shape, device, dtype etc., and integers and strings to match exactly,
               - ``"same input"`` - don't check, but just assume that a cached function works if it exists.
        interpretation: default: ``"translate functions"``

               - ``"python interpreter"`` run in the cpython interpreter, you need to program thunder explicitly,
               - ``"translate functions"`` use the thunder interpreter to translate torch functions to thunder and (optionally) detect sharp edges
    """
    if interpretation is None:
        interpretation = INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS
    return thunder.jit(
        fn,
        langctx=langctx,
        executors=executors,
        sharp_edges=sharp_edges,
        interpretation=interpretation,
        cache=cache,
        disable_torch_autograd=disable_torch_autograd,
        **compile_options,
    )
