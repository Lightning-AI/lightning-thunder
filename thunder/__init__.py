from functools import wraps, partial
from typing import Dict, Set, Optional, Any, List, Tuple, Type
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
import os
import dis
from enum import Enum, auto
import time
from numbers import Number
from itertools import chain
from types import NoneType
import optree

from looseversion import LooseVersion

from thunder.core.options import (
    INTERPRETATION_OPTIONS,
    resolve_interpretation_option,
    CACHE_OPTIONS,
    resolve_cache_option,
    SHARP_EDGES_OPTIONS,
    resolve_sharp_edges_option,
)
from thunder.core.trace import (
    TraceCtx,
    from_trace,
    set_tracectx,
    reset_tracectx,
    tracectx,
)

import thunder.core.prims as prims
import thunder.core.dtypes as dtypes
from thunder.core.symbol import BoundSymbol
import thunder.core.devices as devices
from thunder.common import (
    CompileData,
    CompileStats,
    _create_callable,
    trace,
    preprocess,
    transform_for_execution,
)
import thunder.extend as extend
from thunder.extend import Executor, add_default_executor
from thunder.core.compile_data import compile_data_and_stats, get_cache_option, using_symbolic_values
from thunder.core.langctxs import LanguageContext, resolve_language, Languages
import thunder.core.langctxs as langctxs
from thunder.core.codeutils import get_siginfo, SigInfo, is_simple_printable_collection, is_simple_printable_value
from thunder.core.proxies import (
    is_proxyable,
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
from thunder.core.jit_ext import minimal_thunder_jit, thunder_general_jit
from thunder.core.pytree import tree_flatten
from thunder.executors.torch_autograd import thunder_backward

# NOTE This import is intentionally pytorch so that it thunder.torch doesn't import this
import torch as pytorch

import thunder.clang as clang

# Imports executors (to populate default executors and make them accessible)
import thunder.executors.pythonex
import thunder.executors.torchex
import thunder.executors.nvfuserex

pythonex = extend.get_executor("python")


_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

# TODO GTC Review exposed names
__all__ = [
    # dtype aliases
    "bool8",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex32",
    "complex64",
    "complex128",
    # language aliases
    "torch",
    "numpy",
    "prims"
    # interface functions
    # TODO Extend this
    # TODO Add device aliases
    # TODO Add executor aliases
    "sdpa_executor",
    "nvfuser_executor",
    "pytorch_executor",
    # debugging functions
    "set_execution_callback_file",
]


def __version__():
    return LooseVersion("0.0.1")


# TODO maybe move these aliases to the core language?
#
# dtype aliases
#
bool8 = dtypes.bool8
uint8 = dtypes.uint8
int8 = dtypes.int8
int16 = dtypes.int16
int32 = dtypes.int32
int64 = dtypes.int64
bfloat16 = dtypes.bfloat16
float16 = dtypes.float16
float32 = dtypes.float32
float64 = dtypes.float64
complex32 = dtypes.complex32
complex64 = dtypes.complex64
complex128 = dtypes.complex128

#
# Module aliases
#

# NOTE this allows clang.foo() to be called directly as thunder.foo()
from thunder.clang import *

#
# Promoted executor-related functions and objects
#

# TODO Add more of these functions
get_all_executors = extend.get_all_executors
get_default_executors = extend.get_default_executors
get_always_executors = extend.get_always_executors

sdpa_executor: None | extend.Executor = extend.get_executor("sdpa")
nvfuser_executor: None | extend.Executor = extend.get_executor("nvfuser")
pytorch_executor: None | extend.Executor = extend.get_executor("torch")

# Default executor list is [sdpa -> nvfuser -> torch -> python]
if nvfuser_executor:
    add_default_executor(nvfuser_executor)

if sdpa_executor:
    add_default_executor(sdpa_executor)

#
# Promoted debugging functions
#

# If set, Python programs will be written to this file before being executed, and if the
#   the file is modified then the modified version of the program will be compiled and executed, instead.
from thunder.core.trace import _set_execution_file

set_execution_callback_file = _set_execution_file


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
    if typ is pytorch.dtype:
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
    NoneType: _eager_unpack_none,
    pytorch.dtype: _eager_unpack_literal_like,
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
    prologue_trc: TraceCtx = TraceCtx(fn)
    computation_trc: TraceCtx = TraceCtx()

    # Unpacks the inputs
    # TODO GTC Support PyTorch dtypes
    # TODO GTC Support sequences of numbers, tensors, arrays, and strings
    # TODO GTC Support mappings from strings and numbers to numbers, tensors, arrays, strings
    # TODO GTC Consider supporting nested sequences of mappings that have these properties
    # TODO GTC Consider supporting arbitrary literal inputs
    # TODO GTC Consider supporiting arbitrary object inputs
    supported_input_types = tuple(_type_to_unpack_map.keys())

    si: SigInfo = get_siginfo(fn, args, kwargs)

    # Constructs the prologue trace (which just trivially unpacks the tensor arguments for now)
    # TODO GTC Remove the no_grad and no_autocast context managers from this trace
    # TODO GTC Provide a mechanism to add context managers to the prologue and computation functions
    # TODO GTC Don't always import torch in traces (particularly the prologue trace)
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
    # TODO GTC Only unpack what's used in the computation
    with tracectx(computation_trc):
        p: Proxy
        for p in computation_args:
            prims.unpack_trivial(p)
            csi.args.append((p.name, None))
            computation_trc.add_name(p.name)

        result = interpreter(fn)(*interpretation_args, **interpretation_kwargs)

        # Validates that the returned items are proxies or printable values
        def leaf_test(x: Any) -> bool:
            if isinstance(x, Proxy):
                return True
            if is_simple_printable_value(x):
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
    pjit = partial(minimal_thunder_jit, sharp_edges=sharp_edges)
    return _eager_unpacking_interpreter(pjit, fn, args, kwargs, interpreter_name="translate functions")


# Translates the Python function to a thunder program using the thunder interpreter
def _general_frontend(fn: Callable, args, kwargs, /, *, sharp_edges: SHARP_EDGES_OPTIONS) -> tuple[TraceCtx, TraceCtx]:
    return thunder_general_jit(fn, args, kwargs, sharp_edges=sharp_edges)


# This function will replace compile() (below) before gtc
# TODO GTC Consider adding a debug_log parameter to control debug printing
# TODO GTC Consider renaming compile_options to additional_compile_options
def jit(
    fn: Callable,
    /,
    *,
    langctx: None | str | Any | LanguageContext = None,
    executors: None | Sequence[Executor] = None,
    sharp_edges: None | SHARP_EDGES_OPTIONS | str = None,
    interpretation: None | INTERPRETATION_OPTIONS | str = None,
    cache: None | CACHE_OPTIONS | str = None,
    **compile_options,
) -> Callable:
    # Resolves langctx
    if langctx is None:
        langctx = Languages.TORCH
    langctx: LanguageContext = resolve_language(langctx)

    # Resolves executors
    # TODO GTC Review exposed executor names
    if executors is None:
        executors = tuple(get_default_executors() + get_always_executors())
    else:
        executors = tuple(executors)

        for ex in executors:
            if not isinstance(ex, Executor):
                raise ValueError(f"Value {ex} passed in 'executors' was not an executor")

        # Extends with always executors (ensuring they are always present and in the correct order)
        #   if not already present and in the correct order
        always_executors: tuple[Executor] = get_always_executors()
        if executors[-len(always_executors) :] != always_executors:
            executors = executors + always_executors

    # Resolves options
    sharp_edges = resolve_sharp_edges_option(sharp_edges)
    interpretation = resolve_interpretation_option(interpretation)
    cache = resolve_cache_option(cache)

    # TODO GTC Refine the compile data option to remove unused options
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors,
        cache_option=cache,
        using_jit=True,
        use_cudagraphs=False,
        use_torch_compile=False,
        disable_torch_autograd_support=False,
        use_rematerialization=False,
        only_execute_prims=False,
        disable_preprocessing=True,
        compile_options=compile_options,
    )
    cs = CompileStats()

    @wraps(fn)
    def fn_(*args, **kwargs) -> Any:
        # TODO GTC Support being called from another jitted function by just calling fn with
        #   distinct compile data (test this)

        # Updats call statistics
        cs.last_trace_host_start = time.time_ns()
        cs.calls += 1

        # TODO GTC Add autocast checks to prologue (make it a compile option)
        # TODO GTC Add module and function checks to prologue (make it a compile option)

        # Checks cache
        # TODO GTC Record prologue time vs computation (not execution) time
        # TODO GTC Set interpreted_instructions and history
        cs.last_trace_cache_start = time.time_ns()
        if (cd.cache_option is CACHE_OPTIONS.CONSTANT_VALUES) or (cd.cache_option is CACHE_OPTIONS.SYMBOLIC_VALUES):
            for pro, pro_traces, comp, comp_traces in cs.interpreter_cache:
                try:
                    inps = pro(*args, **kwargs)
                except Exception as ex:
                    continue

                cs.last_trace_host_tracing_start = time.time_ns()
                cs.last_trace_host_tracing_stop = time.time_ns()

                cs.last_trace_host_execution_start = time.time_ns()
                result = comp(*inps)
                cs.last_trace_host_execution_stop = time.time_ns()
                # Updates cache statistics
                cs.cache_hits += 1
                cs.last_executed = comp
                cs.last_traces = comp_traces
                cs.last_interpreted_instructions = None
                cs.last_interpreted_history = None
                cs.last_prologue_traces = pro_traces
                cs.last_prologue = pro
                cs.last_trace_cache_stop = time.time_ns()
                cs.last_trace_host_stop = time.time_ns()
                return result

        if cd.cache_option is CACHE_OPTIONS.SAME_INPUT:
            if len(cs.interpreter_cache):
                pro, pro_traces, comp, comp_traces = cs.interpreter_cache[0]
                inps = pro(*args, **kwargs)

                cs.last_trace_host_tracing_start = time.time_ns()
                cs.last_trace_host_tracing_stop = time.time_ns()

                cs.last_trace_host_execution_start = time.time_ns()
                result = comp(*inps)
                cs.last_trace_host_execution_stop = time.time_ns()

                # Updates cache statistics
                cs.cache_hits += 1
                cs.last_executed = comp
                cs.last_traces = comp_traces
                cs.last_interpreted_instructions = None
                cs.last_interpreted_history = None
                cs.last_prologue_traces = pro_traces
                cs.last_prologue = pro
                cs.last_trace_cache_stop = time.time_ns()
                cs.last_trace_host_stop = time.time_ns()
                return result

        cs.cache_misses += 1
        cs.last_trace_cache_stop = time.time_ns()

        # Resets use of compile flags
        cs.last_compile_reasons = defaultdict(list)

        # TODO GTC Acquires the interpreter
        # TODO GTC Implement all INTERPRETATION_OPTIONS
        # TODO GTC Acquire the interpretation option from the compile data
        interpreter: Callable
        if interpretation is INTERPRETATION_OPTIONS.PYTHON_INTERPRETER:
            interpreter = _python_interpreter
        elif interpretation is INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS:
            interpreter = _translate_functions_interpreter
        elif interpretation is INTERPRETATION_OPTIONS.TRANSLATE_PYTHON:
            interpreter = _general_frontend
        else:
            raise NotImplementedError(
                f"Only the 'python interpreter' and 'translate functions' interpretation options are currently implemented."
            )

        with compile_data_and_stats(cd, cs):
            # Acquires the trace OR inlines the trace into an existing trace and
            #   returns the (proxied) result of the operation
            cs.last_trace_tracing_start = time.time_ns()

            with langctxs.langctx(cd.langctx):
                prologue_trc: TraceCtx
                computation_trc: TraceCtx
                # TODO GTC Review if sharp_edges should come from a CompileOptions object
                prologue_trc, computation_trc = interpreter(fn, args, kwargs, sharp_edges=sharp_edges)

            cs.last_trace_tracing_stop = time.time_ns()

            # Makes the prologue callable
            protraces = transform_for_execution(
                prologue_trc,
                executors_list=(pythonex,),
                use_del_last_used=False,
            )
            protrace = protraces[-1]
            pro = protrace.python_callable()

            # TODO GTC Apply transforms
            # note: prologue runtime is not in the measurement below
            prologue_outputs = pro(*args, **kwargs)

            tensor_cls = (pytorch.Tensor, TensorProxy)
            requires_grad = any(isinstance(arg, tensor_cls) and arg.requires_grad for arg in prologue_outputs)
            if not cd.disable_torch_autograd_support and requires_grad:
                # thunder_backward may recursively call compile and wraps the result in a
                # torch.autograd.Function to support embedding of Thunder-compiled
                # functions in torch's Autograd
                cs.last_trace_host_execution_start = time.time_ns()
                comp = thunder_backward(compile_data=cd, compile_stats=cs)(computation_trc.python_callable())
                computation_result = comp(*prologue_outputs)
                cs.last_trace_host_execution_stop = time.time_ns()
                cs.last_executed = comp
                extraces = []  # todo
            else:
                # TODO GTC Update this transform's parameters to take 'executors' instead of 'executors_list'
                extraces = transform_for_execution(
                    computation_trc,
                    executors_list=cd.executors_list,
                )
                extrace = extraces[-1]

                comp = extrace.python_callable()

                # Executes the traced program
                cs.last_trace_host_execution_start = time.time_ns()
                computation_result = comp(*prologue_outputs)
                cs.last_trace_host_execution_stop = time.time_ns()

            # TODO GTC Update the cache
            if cd.cache_option is not CACHE_OPTIONS.NO_CACHING:
                cs.interpreter_cache.append((pro, protraces, comp, extraces))

            # Updates statistics
            cs.last_traces = [computation_trc] + extraces
            cs.last_executed = comp
            cs.last_prologue_traces = [prologue_trc] + protraces
            cs.last_prologue = pro

            cs.last_trace_host_stop = time.time_ns()

        return computation_result

    # Sets compile options and statistics attributes
    fn_._lc_cd = cd
    fn_._lc_cs = cs

    return fn_


def compile(
    fn: Callable,
    *,
    langctx: None | Any = None,
    executors_list: None | Sequence[Executor] = None,
    cache_mode: None | str | CACHE_OPTIONS = None,
    use_cudagraphs: bool = False,
    use_torch_compile: bool = False,
    disable_torch_autograd_support: bool = False,
    use_rematerialization: bool = False,
    only_execute_prims: bool = False,
    disable_preprocessing: bool = False,
    **kwargs,
) -> Callable:
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors_list,
        cache_option=cache_mode,
        use_cudagraphs=use_cudagraphs,
        use_torch_compile=use_torch_compile,
        disable_torch_autograd_support=disable_torch_autograd_support,
        use_rematerialization=use_rematerialization,
        only_execute_prims=only_execute_prims,
        disable_preprocessing=disable_preprocessing,
        compile_options=kwargs,
    )

    cs = CompileStats()
    _fn = _create_callable(cd, cs)
    return _fn


def compile_data(fn) -> CompileData | None:
    return getattr(fn, "_lc_cd", None)


def compile_stats(fn) -> CompileStats | None:
    return getattr(fn, "_lc_cs", None)


# TODO We should remove compiledata.last_traces in favor of forward_last_traces and backward_last_traces
def last_traces(fn) -> list[TraceCtx] | tuple[list[TraceCtx], list[TraceCtx]]:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.forward_last_traces is not None and cs.backward_last_traces is not None:
        return cs.forward_last_traces, cs.backward_last_traces
    if cs.last_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_traces


def last_prologue_traces(fn) -> TraceCtx:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_prologue_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_prologue_traces


def cache_option(fn) -> CACHE_OPTIONS:
    cd = compile_data(fn)
    if cd is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cd.cache_option


def cache_hits(fn) -> int:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cs.cache_hits


def cache_misses(fn) -> int:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cs.cache_misses


def list_transforms(fn) -> list:
    return fn._lc_transforms


def last_interpreted_instructions(fn: Callable) -> list[dis.Instruction]:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_interpreted_instructions is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_interpreted_instructions


def last_interpreted_history(fn: Callable) -> list[dis.Instruction | str]:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_interpreted_history is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_interpreted_history


# Prints how compiled options were used (or not)
def last_compile_options(fn: Callable, /) -> None:
    cd = compile_data(fn)
    cs = compile_stats(fn)

    # NOTE Different categories of compile options
    # Specified and Queried --- in cs.last_compile_reasons and cd.compile_options
    # Queried but not Specified --- in cs.last_compile_reasons but not in cd.compile_options (not printed)
    # Specified but not Queried --- in cd.compile_options but not in cs.last_compile_reasons

    specified: set = set(cd.compile_options.keys())
    queried: set = set(cs.last_compile_reasons.keys())

    # Prints used options
    print("Used compile options:")
    used = specified & queried

    if len(used) == 0:
        print("\tNo used options")

    for option in used:
        reasons = set(cs.last_compile_reasons[option])

        for reason in reasons:
            print(f"\t{option}. {reason}")

    # Prints unused options
    print("Unused compile options:")
    unused: set = specified - queried

    if len(unused) == 0:
        print("\tNo unused options")

    for option in unused:
        print(f"\t{option}")


# TODO (mruberry) Update this
def _grad_transform(trace):
    grad_fwd_trace = from_trace(trace)
    trace_tok = set_tracectx(grad_fwd_trace)
    all_residuals = []

    # Constructs grad fwd and records info
    # TODO: make recursive (or iterative, whatever)
    current_inputs = grad_fwd_trace.args
    for bsym in trace.bound_symbols:
        grad_defined = bsym.sym.grad_defined
        grad_ignored = bsym.sym.grad_ignored
        grad_fwd, grad_bwd = bsym.sym.grad_fwd, bsym.sym.grad_bwd

        if not grad_defined:
            raise NotImplementedError

        # Constructs the new grad_fwd symbol, which returns the primals and residuals
        if grad_fwd is None:
            fw_result = bsym.sym(*current_inputs)
            residuals = None
            all_residuals.append(residuals)
            current_inputs = fw_result if isinstance(fw_result, Sequence) else (fw_result,)
            continue

        fw_result, residuals = grad_fwd(*current_inputs)
        all_residuals.append(residuals)
        current_inputs = fw_result if isinstance(fw_result, Sequence) else (fw_result,)

    # Constructs bwd part of the program
    current_grads = (prims.full(o.shape, 1.0, device=o.device, dtype=o.dtype) for o in fw_result)

    for bsym, residuals in zip(reversed(trace.bound_symbols), reversed(all_residuals)):
        grad_fwd = bsym.sym.grad_fwd
        grad_bwd = bsym.sym.grad_bwd
        grad_defined = bsym.sym.grad_defined
        if not grad_defined:
            raise NotImplementedError(f"grad_bwd not defined for {bsym.sym}")

        if grad_fwd is None:
            continue
        current_grads = grad_bwd(*current_grads, *residuals)
        current_grads = (current_grads,) if not isinstance(current_grads, Sequence) else current_grads

    grad_fwd_trace.output = current_grads

    # Resets tracing context
    reset_tracectx(trace_tok)

    return grad_fwd_trace


# TODO Test nesting of grad and grad and grad and grad
# TODO Test nesting of a regular function + calling grad
def grad(fn):
    cfn = compile(fn)

    @wraps(cfn)
    def _fn(*args, **kwargs):
        original_result, original_trace = cfn(*args, **kwargs)
        original_trace = last_traces(cfn)

        gradir = _grad_transform(original_trace)

        return original_result, original_trace

    return _fn
