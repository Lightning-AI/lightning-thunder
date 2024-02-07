from functools import wraps, partial
from typing import Dict, Set, Optional, Any, List, Tuple, Type
from collections.abc import Callable
from collections.abc import Sequence
import os
import dis
from enum import Enum, auto

from looseversion import LooseVersion

from thunder.core.trace import (
    TraceCtx,
    from_trace,
    set_tracectx,
    reset_tracectx,
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
)
import thunder.extend as extend
from thunder.extend import Executor, add_default_executor

# The following executors are always available, and so are unconditionally imported
from thunder.executors import pythonex, torchex, nvfuserex, sdpaex

import thunder.torch as ltorch
import thunder.numpy as lnumpy

torchlangctx = ltorch
numpylangctx = lnumpy


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
# Language context options
#
# The language context controls how methods on tensors are interpreted.
# TODO Allow a language context object to be specified directly

_str_to_lang_ctx_map: dict[str, Any] = {
    "torch": torchlangctx,
    "numpy": numpylangctx,
}


def _str_to_lang_ctx(s: str, /) -> Any:
    lang_ctx: None | Any = _str_to_lang_ctx_map.get(s.lower(), None)

    if lang_ctx is None:
        raise ValueError(f"Unknown language ctx {s}. Allowed modes are 'torch' or 'numpy'.")

    return lang_ctx


#
# Sharp edges modes
#
# A "sharp edge" is part of the original program which may not be captured
#   in the generated thunder program.
# ALLOW means that sharp edges are unchecked. (Sharp edges are allowed.)
# WARN means that when a sharp edge is identified a warning is thrown.
# ERROR means that when a sharp edge is identified an error is thrown.


class SHARP_EDGES_OPTIONS(Enum):
    ALLOW = auto()
    WARN = auto()
    ERROR = auto()


_str_to_sharp_edges_options_map: dict[str, SHARP_EDGES_OPTIONS] = {
    "allow": SHARP_EDGES_OPTIONS.ALLOW,
    "warn": SHARP_EDGES_OPTIONS.WARN,
    "error": SHARP_EDGES_OPTIONS.ERROR,
}


def _str_to_sharp_edges_option(s: str, /) -> SHARP_EDGES_OPTIONS:
    sharp_edges_option: None | SHARP_EDGES_OPTIONS = _str_to_sharp_edges_options_map.get(s.lower(), None)

    if sharp_edges_option is None:
        raise ValueError(f"Unknown sharp edges option {s}. Allowed modes are 'allow', 'warn', and 'error'.")

    return sharp_edges_option


#
# Interpretation modes
#
# These modes control how the function will be interpreted
# NONE means that no interpretation is performed -- the function is just traced.
# FUNCTIONS means that the interpreter only translates PyTorch and NumPy operations to thunder operations
# EVERYTHING means that the interpreter translates the entire function a thunder program


class INTERPRETATION_OPTIONS(Enum):
    NONE = auto()
    TRANSLATE_FUNCTIONS = auto()
    TRANSLATE_EVERYTHING = auto()


_str_to_interpretation_option_map: dict[str, INTERPRETATION_OPTIONS] = {
    "none": INTERPRETATION_OPTIONS.NONE,
    "translate functions": INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS,
    "translate everything": INTERPRETATION_OPTIONS.TRANSLATE_EVERYTHING,
}


def _str_to_interpretation_option(s: str, /) -> INTERPRETATION_OPTIONS:
    interpretation_option: None | INTERPRETATION_OPTIONS = _str_to_interpretation_option_map.get(s.lower(), None)

    if interpretation_option is None:
        raise ValueError(
            f"Unknown interpretation option {s}. Allowed modes are 'none', 'translate functions', and 'translate everything'."
        )

    return interpretation_option


# This function will replace compile() (below) before gtc
# TODO GTC Consider adding a debug_log parameter to control debug printing
def jit(
    fn: Callable,
    /,
    *,
    langctx: None | str | Any = None,
    executors: None | Sequence[Executor] = None,
    sharp_edges: None | SHARP_EDGES_OPTIONS | str = None,
    interpretation: None | INTERPRETATION_OPTIONS | str = None,
    cache: None | CACHE_OPTIONS | str = None,
    **compile_options,
) -> Callable:
    # Resolves langctx
    # TODO GTC Allow directly passing a LanguageContext class
    if langctx is None:
        langctx = torchlangctx
    if isinstance(langctx, str):
        langctx = _str_to_lang_ctx(langctx)
    if langctx is not torchlangctx and langctx is not numpylangctx:
        raise NotImplementedError(f"Only 'torch' and 'numpy' are currently implemented as language contexts.")

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

    # TODO GTC Implememt these options
    if sharp_edges in (SHARP_EDGES_OPTIONS.WARN, SHARP_EDGES_OPTIONS.ERROR):
        raise NotImplementedError(f"Only the 'allow' option of sharp edges is currently implemented.")

    # Resolves interpretation option
    if interpretation is None:
        # TODO GTC Change the default to FUNCTIONS
        interpretation = INTERPRETATION_OPTIONS.NONE
    if isinstance(interpretation, str):
        interpretation = _str_to_interpretation_option_map(interpretation)
    if not isinstance(interpretation, INTERPRETATION_OPTIONS):
        raise ValueError(
            f"Unknown interpretation option {interpretation}. Allowed options are 'none', 'translate functions', and 'translate everything'."
        )

    # TODO GTC Implement these modes
    if interpretation in (INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS, INTERPRETATION_OPTIONS.TRANSLATE_EVERYTHING):
        raise NotImplementedError(f"Only the 'none' interpretation option is currently implemented.")

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

    if cache in (CACHE_OPTIONS.ASSUME_SAME_INPUTS, CACHE_OPTIONS.DYNAMIC_STRIDES):
        raise NotImplementedError(f"Only the 'no caching' cache mode is currently supported.")

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
    _fn = _create_callable(cd, cs)
    return _fn


def compile(
    fn: Callable,
    *,
    langctx: Any | None = None,
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
