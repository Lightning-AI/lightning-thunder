from functools import wraps, partial
from typing import Dict, Set, Optional, Any, List, Callable, Tuple, Type
import os

from looseversion import LooseVersion

from thunder.core.trace import (
    TraceCtx,
    from_trace,
    set_tracectx,
    reset_tracectx,
)

import thunder.core.prims as prims
import thunder.core.dtypes as dtypes
import thunder.executors as executors
import thunder.core.symbol as symbol
import thunder.core.devices as devices
from thunder.common import CACHE_MODES, CompileData, CompileStats, _create_callable, trace, preprocess

import thunder.torch as ltorch
torchlangctx = ltorch


_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

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
    "trace",
    "preprocess",
    # TODO Add device aliases
    # TODO Add executor aliases
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
# Promoted executor-related functions
#

list_executors = executors.list_executors
list_default_executors = executors.list_default_executors
set_default_executors = executors.set_default_executors
add_operator_executor = executors.add_operator_executor

def compile(
    fn: Callable,
    *,
    langctx: Optional[Any] = None,
    executors_list: Optional[list[executors.Executor]] = None,
    cache_mode: None | str | CACHE_MODES = None,
    use_cudagraphs: bool = False,
    disable_torch_autograd_support: bool = False,
    use_rematerialization: bool = False,
    only_execute_prims: bool = False,
    disable_preprocessing: bool = False,
) -> Callable:
    
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors_list,
        cache_mode=cache_mode,
        use_cudagraphs=use_cudagraphs,
        disable_torch_autograd_support=disable_torch_autograd_support,
        use_rematerialization=use_rematerialization,
        only_execute_prims=only_execute_prims,
        disable_preprocessing=disable_preprocessing,
    )

    cs = CompileStats()
    _fn = _create_callable(cd, cs)
    return _fn

def compile_data(fn) -> Optional[CompileData]:
    return getattr(fn, "_lc_cd", None)

def compile_stats(fn) -> Optional[CompileStats]:
    return getattr(fn, "_lc_cs", None)


# TODO We should remove compiledata.last_traces in favor of forward_last_traces and backward_last_traces
def last_traces(fn) -> None | list[TraceCtx] | tuple[list[TraceCtx], list[TraceCtx]]:
    cs = compile_stats(fn)
    
    if cs.forward_last_traces is not None and cs.backward_last_traces is not None:
        return cs.forward_last_traces, cs.backward_last_traces
    return cs.last_traces


def cache_mode(fn) -> CACHE_MODES:
    return compile_data(fn).cache_mode


def cache_hits(fn) -> int:
    return compile_stats(fn).cache_hits


def cache_misses(fn) -> int:
    return compile_stats(fn).cache_misses

def list_transforms(fn) -> list:
    return fn._lc_transforms


# TODO There is probably a better way to do this
symbol.set_eagerctx(
    (partial(compile, executors_list=[executors.TORCH], only_execute_prims=True, disable_preprocessing=True), ltorch)
)


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
