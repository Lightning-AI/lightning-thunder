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

from looseversion import LooseVersion

from thunder.core.trace import (
    TraceCtx,
    from_trace,
    set_tracectx,
    reset_tracectx,
    tracectx,
)

import thunder.core.prims as prims
import thunder.core.dtypes as dtypes
import thunder.core.symbol as symbol
import thunder.core.devices as devices
from thunder.common import (
    CACHE_OPTIONS,
    CompileData,
    CompileStats,
    _create_callable,
    trace,
    preprocess,
    _string_to_cache_option,
    transform_for_execution,
    SHARP_EDGES_OPTIONS,
    _str_to_sharp_edges_option,
)
import thunder.extend as extend
from thunder.extend import Executor, add_default_executor
from thunder.core.compile_data import compile_data_and_stats
from thunder.core.langctxs import LanguageContext, resolve_language, Languages
import thunder.core.langctxs as langctxs
from thunder.core.codeutils import get_siginfo, SigInfo
from thunder.core.proxies import is_proxyable, proxy, Proxy, TensorProxy
from thunder.core.jit_ext import minimal_thunder_jit

# The following executors are always available, and so are unconditionally imported
from thunder.executors import pythonex, torchex, nvfuserex, sdpaex

# NOTE This import is intentionally pytorch so that it thunder.torch doesn't import this
import torch as pytorch
import thunder.torch


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

#
# Interpretation modes
#
# These modes control how the function will be interpreted
# PYTHON_INTERPRETER means that the Python interpreter is used
# TRANSLATE_FUNCTIONS means that the thunder interpreter is used, and it translates PyTorch and NumPy operations to thunder operations
# TRANSLATE_EVERYTHING means that the thunder interpreter is used, and it translates the entire function a thunder program


class INTERPRETATION_OPTIONS(Enum):
    PYTHON_INTERPRETER = auto()
    TRANSLATE_FUNCTIONS = auto()
    TRANSLATE_EVERYTHING = auto()


_str_to_interpretation_option_map: dict[str, INTERPRETATION_OPTIONS] = {
    "python interpreter": INTERPRETATION_OPTIONS.PYTHON_INTERPRETER,
    "translate functions": INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS,
    "translate everything": INTERPRETATION_OPTIONS.TRANSLATE_EVERYTHING,
}


def _unknown_interpretation_option(x: Any) -> None:
    raise ValueError(
        f"Unknown interpretation option {x}. Allowed modes are 'python interpreter', 'translate functions', and 'translate everything'."
    )


def _str_to_interpretation_option(s: str, /) -> INTERPRETATION_OPTIONS:
    interpretation_option: None | INTERPRETATION_OPTIONS = _str_to_interpretation_option_map.get(s.lower(), None)

    if interpretation_option is None:
        _unknown_interpretation_option(s)

    return interpretation_option


# A helper for "eager unpacking" interpreters that eagerly unpack their arguments as inputs
# An interpreter must do two things:
#   1) Create a prologue function with the same signature as the original function that
#       acquires all supported inputs and validates them according to the caching option
#   2) Creates a computation function that accepts the output of the prologue function and
#       returns what the original function did
def _eager_unpacking_interpreter(
    interpreter: Callable, fn: Callable, args, kwargs, /, *, interpreter_name: str
) -> tuple[TraceCtx, TraceCtx]:
    # TODO GTC Update using_interpreter to using_prologue until it's removed when all traces have prologues
    prologue_trc: TraceCtx = TraceCtx(fn)
    computation_trc: TraceCtx = TraceCtx()

    # Unpacks the inputs

    if len(kwargs) > 0:
        raise NotImplementedError(f"kwargs are not yet supported")

    # TODO GTC Support strings
    # TODO GTC Support PyTorch and NumPy dtypes
    # TODO GTC Support sequences of numbers, tensors, arrays, and strings
    # TODO GTC Support mappings from strings and numbers to numbers, tensors, arrays, strings
    # TODO GTC Consider supporting nested sequences of mappings that have these properties
    # TODO GTC Consider supporting arbitrary literal inputs
    # TODO GTC Consider supporiting arbitrary object inputs
    supported_input_types = (
        Number,
        pytorch.Tensor,
        str,
    )

    for arg in args:
        if not isinstance(arg, supported_input_types):
            raise NotImplementedError(
                f"Inputs with {type(arg)} are not supported when using the {interpreter_name} interpreter. Supports input types are {supported_input_types}"
            )

    si: SigInfo = get_siginfo(fn, args, kwargs)

    if si.varargs is not None:
        raise NotImplementedError("varargs are not yet supported")

    # Constructs the prologue trace (which just trivially unpacks the tensor arguments for now)
    # TODO GTC Remove the no_grad and no_autocast context managers from this trace
    # TODO GTC Provide a mechanism to add context managers to the prologue and computation functions
    # TODO GTC Don't always import torch in traces (particularly the prologue trace)
    with tracectx(prologue_trc):
        proxyargs = []
        for name, x in si.args:
            p = proxy(x, name=name)
            proxyargs.append(p)

        p: Proxy
        for p in proxyargs:
            prims.unpack_trivial(p)

        prims.python_return(proxyargs)

    prologue_trc.args = proxyargs

    with tracectx(computation_trc):
        # TODO GTC Only unpack what's used in the computation
        p: Proxy
        for p in proxyargs:
            prims.unpack_trivial(p)

        result = interpreter(fn)(*proxyargs)
        prims.python_return(result)

    # Constructs the computation trace's signature
    si = SigInfo("computation")
    si.args = list((p.name, None) for p in proxyargs)
    computation_trc._siginfo = si
    computation_trc.args = proxyargs

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
        if executors[-len(always_executors)] != always_executors:
            executors = executors + always_executors

    # Resolves sharp edges option
    if sharp_edges is None:
        # TODO GTC Change the default to WARN
        sharp_edges = SHARP_EDGES_OPTIONS.ALLOW
    if isinstance(sharp_edges, str):
        sharp_edges = _str_to_sharp_edges_option(sharp_edges)
    if not isinstance(sharp_edges, SHARP_EDGES_OPTIONS):
        raise ValueError(f"Unknown sharp edges option {sharp_edges}. Allowed options are 'allow', 'warn', and 'error'.")

    # Resolves interpretation option
    if interpretation is None:
        # TODO GTC Change the default to TRANSLATE_FUNCTIONS
        interpretation = INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS
    if isinstance(interpretation, str):
        interpretation = _str_to_interpretation_option_map(interpretation)
    if not isinstance(interpretation, INTERPRETATION_OPTIONS):
        _unknown_interpretation_option(interpretation)

    # Resolves cache option
    if cache is None:
        # TODO GTC Change the default to DYNAMIC_STRIDES
        cache = CACHE_OPTIONS.NO_CACHING
    if isinstance(cache, str):
        cache = _string_to_cache_option(cache)
    if not isinstance(cache, CACHE_OPTIONS):
        raise ValueError(
            f"Unknown cache option {cache}. Allowed options are 'no caching', 'assume same inputs', and 'dynamic strides'."
        )

    # TODO GTC Refine the compile data option to remove unused options
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors,
        cache_option=cache,
        use_cudagraphs=False,
        use_torch_compile=False,
        disable_torch_autograd_support=True,
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
        # TODO GTC Add caching (with prologues)
        # TODO GTC Add DYNAMIC_SHAPES caching option

        # Checks cache
        cs.last_trace_cache_start = time.time_ns()
        if cd.cache_option in (CACHE_OPTIONS.ASSUME_SAME_INPUTS, CACHE_OPTIONS.DYNAMIC_STRIDES):
            raise NotImplementedError(f"Only the 'no caching' cache mode is currently supported.")

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

            # TODO GTC Apply transforms

            # TODO GTC Update this transform's parameters to take 'executors' instead of 'executors_list'
            extraces = transform_for_execution(
                computation_trc,
                executors_list=cd.executors_list,
            )

            extrace = extraces[-1]

            pro = prologue_trc.python_callable()
            comp = extrace.python_callable()

            # Executes the traced program
            cs.last_trace_host_execution_start = time.time_ns()
            computation_result = comp(*pro(*args, **kwargs))
            cs.last_trace_host_execution_stop = time.time_ns()

            # TODO GTC Update the cache
            # Updates metadata
            # TODO GTC Populate last_traces properly
            cs.last_traces = extraces
            # TODO GTC Populate last executed properly (prologue + computation + consider backward)
            cs.last_executed = comp
            cs.last_prologue = prologue_trc

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


def last_prologue(fn) -> TraceCtx:
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_prologue is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_prologue


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
