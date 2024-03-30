from functools import wraps
from typing import Any
from collections import defaultdict, namedtuple
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import os
import dis
import time
import warnings

from looseversion import LooseVersion

from thunder.core.options import (
    INTERPRETATION_OPTIONS,
    resolve_interpretation_option,
    resolve_sharp_edges_option,
    CACHE_OPTIONS,
    SHARP_EDGES_OPTIONS,
)
from thunder.core.trace import (
    TraceCtx,
    from_trace,
    set_tracectx,
    reset_tracectx,
    is_tracing,
)

from thunder import functional as functional
import thunder.core.prims as prims
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
from thunder.core.transform_common import dce
from thunder.common import (
    CompileData,
    CompileStats,
    _create_callable,
    trace,
    transform_for_execution,
)
import thunder.extend as extend
from thunder.extend import Executor, add_default_executor
from thunder.core.compile_data import compile_data_and_stats
from thunder.core.langctxs import LanguageContext
import thunder.core.langctxs as langctxs
from thunder.core.baseutils import run_once, check
from thunder.core.proxies import (
    Proxy,
    TensorProxy,
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
from thunder.core.jit_ext import thunder_general_jit
from thunder.executors.torch_autograd import split_forward_backward, ThunderFunction
from thunder.cudagraphs import CUDAGraphExecutor

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

# TODO RC1 Review exposed names
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
    "prims",
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


# Translates the Python function to a thunder program using the thunder interpreter
def _general_frontend(fn: Callable, args, kwargs, /, *, sharp_edges: SHARP_EDGES_OPTIONS) -> tuple[TraceCtx, TraceCtx]:
    return thunder_general_jit(fn, args, kwargs, sharp_edges=sharp_edges)


class ThunderModule(pytorch.nn.Module):
    """A wrapper nn.Module subclass.

    This wrapper is returned by ``thunder.jit``, you would typically not
    instantiate it manually.

    """

    def __init__(self, model, compiled_model_call):
        """"""
        super().__init__()
        self._model = model

        self._forward_fn = compiled_model_call

    def forward(self, *args, **kwargs):
        res = self._forward_fn(*args, **kwargs)
        return res

    @contextmanager
    def no_sync(self):
        r"""Context manager to disable gradient synchronization in data parallel mode.

        This context manager is intended to be used in conjunction with
        :class:`torch.nn.parallel.DistributedDataParallel` to disable gradient
        synchronization in the backward pass. It will not have any effect when
        used with other modules.

        .. note::

            This could lead to different accumulated gradients with ``torch.nn.parallel.distributed.DistributedDataParallel.no_sync``.
            PyTorch's gradient synchronization is implemented by applying all-reduce to gradient buckets of ``torch.nn.Parameter.grad``.
            Thus the ``no_sync`` context leads to :math:`\text{AllReduce} \left( \sum_{i = 0}^{\rm{num_grad_accum_steps}} g_i \right)`.
            In contrast, this synchronizes accumulated gradients when exiting, leading to
            :math:`\text{AllReduce} \left( \sum_{i = 0}^{\rm{num_grad_accum_steps - 1}} g_i \right) + \text{AllReduce}(g_{\rm{num_grad_accum_steps}})`.

        .. warning::

            You must reuse this context manager in each group of gradient accumulation iterations since gradients will get synchronized
            on context manager exit.

            .. code-block:: python

                with model.no_sync():
                    for _ in range(len(gradient_accumulation_iters)):
                        loss(model(x)).backward()  # uses no-sync-backward trace
                loss(model(x)).backward()  # uses the regular backward trace
                optimizer.step()

        """
        from thunder.distributed import (
            set_skip_data_parallel_grad_sync,
            reset_skip_data_parallel_grad_sync,
            _sync_grads,
        )

        token = set_skip_data_parallel_grad_sync(True)
        try:
            yield
        finally:
            reset_skip_data_parallel_grad_sync(token)
            _sync_grads(self)

    def __getattr__(self, name: str) -> Any:
        if name == "_model":
            return self._modules["_model"]
        return getattr(self._model, name)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.load_state_dict(*args, **kwargs)


# this captures the information needed to decide whether a cached function
# matches (e.g. ddp and autocast state)
_cache_info_ctx = ContextVar("cache_info_ctx")


def _with_cache_info_ctx(fn):
    def cache_info_wrapper(*args, **kwargs):
        tok = _cache_info_ctx.set({})
        try:
            res = fn(*args, **kwargs)
        finally:
            _cache_info_ctx.reset(tok)
        return res

    return cache_info_wrapper


def _get_cache_info():
    return _cache_info_ctx.get()


@run_once
def _recursive_jit_call_warning() -> None:
    warnings.warn(
        "Calling a jitted function from a jitted function currently uses all settings from the caller. In the future this behavior may change."
    )


CacheEntry = namedtuple(
    "CacheEntry",
    [
        "prologue_fn",
        "prologue_traces",
        "computation_fn",
        "computation_traces",
        "epilogue_fn",
        "epilogue_traces",
        "backward_fn",
        "backward_traces",
    ],
)


# This function will replace compile() (below) before RC1
# TODO RC1 Consider adding a debug_log parameter to control debug printing
# TODO RC1 Consider renaming compile_options to additional_compile_options
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
    additional_transforms: list | None = None,
    **compile_options,  # TODO RC1 Make this explicit -- dict of options
) -> Callable:
    """Just-in-time compile a callable (function or model).

    Args:
        fn: A :class:`~torch.nn.Module` or a function to compile.
    Keyword Args:
        langctx: the language context, which language / library to emulate. default: "torch" for PyTorch compatibility.
        executors: list of executors to use. Defaults to the executors returned by `thunder.get_default_executors()` and always amened by `thunder.get_always_executors()`.
                   You can get a list of all available executors with `thunder.get_all_executors()`.
        sharp_edges: sharp edge detection action. What to do when thunder detects a construct that is likely to lead to errors. Can be ``"allow"``, ``"warn"``, ``"error"``. Defaults to ``"allow"``.
        cache: caching mode. default: ``"constant values"```

               - ``"no caching"`` - disable caching and always recompute,
               - ``"constant values"`` - require Tensors to be of the same shape, device, dtype etc., and integers and strings to match exactly,
               - ``"same input"`` - don't check, but just assume that a cached function works if it exists.
        interpretation: (deprecated: don't use this, use the thunder.functional.jit entry point to get the functional jit)
    """

    if "executors_list" in compile_options:
        warnings.warn("outdated argument executors_list= in call, please use executors=")
        if executors is None:
            executors = compile_options.pop("executors_list")

    # Resolves interpreter option
    interpretation = resolve_interpretation_option(interpretation)
    interpreter: Callable
    if interpretation is INTERPRETATION_OPTIONS.PYTHON_INTERPRETER:
        interpreter = functional._python_interpreter
    elif interpretation is INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS:
        interpreter = functional._translate_functions_interpreter
    elif interpretation is INTERPRETATION_OPTIONS.TRANSLATE_PYTHON:
        interpreter = _general_frontend

    if additional_transforms is None:
        additional_transforms = []

    # TODO: verify that tutorials don't have false positives and enable warning by default
    # # Make sharp_edges == warn default if not supplied and if in the general jit
    # if interpretation is INTERPRETATION_OPTIONS.TRANSLATE_PYTHON and sharp_edges is None:
    #     sharp_edges = SHARP_EDGES_OPTIONS.WARN

    executor_lookasides = {}
    for ex in executors or []:
        # TODO: sharp edge if lookasides are shadowed?
        executor_lookasides.update(ex._lookasides)

    # TODO RC1 Refine the compile data option to remove unused options
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors,
        cache_option=cache,
        sharp_edges=sharp_edges,
        using_jit=True,
        use_cudagraphs=False,
        use_torch_compile=False,
        disable_torch_autograd_support=disable_torch_autograd,
        use_rematerialization=False,
        only_execute_prims=False,
        disable_preprocessing=True,
        compile_options=compile_options,
        executor_lookasides=executor_lookasides,
    )
    cs = CompileStats()

    @_with_cache_info_ctx
    def get_computation_and_inputs(*args, **kwargs):
        # set up a record of things in the current environment that impact caching / prologues
        # this could be replaced by the respective querying in the prologues
        cache_info = _get_cache_info()

        # autocast related operations
        is_autocast_enabled = False
        if pytorch.is_autocast_enabled() or pytorch.is_autocast_cpu_enabled():
            if pytorch.is_autocast_enabled() and pytorch.is_autocast_cpu_enabled():
                raise NotImplementedError(
                    "thunder.autocast does not support torch.is_autocast_enabled() and torch.is_autocast_cpu_enabled() simultaneously at this moment."
                )
            is_autocast_enabled = True
            autocast_gpu_dtype = dtypes.to_dtype(pytorch.get_autocast_gpu_dtype())
            autocast_cpu_dtype = dtypes.to_dtype(pytorch.get_autocast_cpu_dtype())
            cache_info.update(
                autocast_config_torch_enabled=pytorch.is_autocast_enabled(),
                autocast_config_torch_cpu_enabled=pytorch.is_autocast_cpu_enabled(),
                autocast_gpu_dtype=str(autocast_gpu_dtype),
                autocast_cpu_dtype=str(autocast_cpu_dtype),
            )
            autocast_thunder_dtype = autocast_cpu_dtype if pytorch.is_autocast_cpu_enabled() else autocast_gpu_dtype

        cache_info["is_autocast_enabled"] = is_autocast_enabled

        # TODO(crcrpar): support FSDP as well
        is_ddp_enabled = getattr(fn, "use_ddp", False)
        no_grad_sync = False
        if is_ddp_enabled:
            from thunder.distributed import get_skip_data_parallel_grad_sync

            no_grad_sync = get_skip_data_parallel_grad_sync()
        cache_info["no_grad_sync"] = no_grad_sync

        # TODO RC1 Add module and function checks to prologue (make it a compile option)

        # Checks cache
        cs.last_trace_cache_start = time.time_ns()
        if (cd.cache_option is CACHE_OPTIONS.CONSTANT_VALUES) or (cd.cache_option is CACHE_OPTIONS.SYMBOLIC_VALUES):
            for cache_entry in reversed(cs.interpreter_cache):
                (
                    pro,
                    pro_traces,
                    comp,
                    comp_traces,
                    epilogue,
                    epilogue_traces,
                    backward_fn,
                    backward_traces,
                ) = cache_entry
                try:
                    cs.last_prologue_execution_start = time.time_ns()
                    if epilogue:
                        inps, pro_to_epi = pro(*args, **kwargs)
                    else:
                        inps = pro(*args, **kwargs)
                        pro_to_epi = None
                    cs.last_prologue_execution_stop = time.time_ns()
                except Exception as _:
                    continue

                cs.last_trace_host_tracing_start = time.time_ns()
                cs.last_trace_host_tracing_stop = time.time_ns()

                # Updates cache statistics
                cs.cache_hits += 1
                cs.last_traces = comp_traces
                cs.last_interpreted_instructions = None
                cs.last_interpreted_history = None
                cs.last_prologue_traces = pro_traces
                cs.last_prologue = pro
                cs.last_prologue_transformation_start = 0
                cs.last_prologue_transformation_stop = 0
                cs.last_computation_transformation_start = 0
                cs.last_computation_transformation_stop = 0

                return cache_entry, inps, pro_to_epi

        if cd.cache_option is CACHE_OPTIONS.SAME_INPUT:
            if len(cs.interpreter_cache):
                cache_entry = cs.interpreter_cache[0]
                (
                    pro,
                    pro_traces,
                    comp,
                    comp_traces,
                    epilogue,
                    epilogue_traces,
                    backward_fn,
                    backward_traces,
                ) = cache_entry

                cs.last_prologue_execution_start = time.time_ns()
                if epilogue:
                    inps, pro_to_epi = pro(*args, **kwargs)
                else:
                    inps = pro(*args, **kwargs)
                    pro_to_epi = None
                cs.last_prologue_execution_stop = time.time_ns()

                cs.last_trace_host_tracing_start = time.time_ns()
                cs.last_trace_host_tracing_stop = time.time_ns()

                # Updates cache statistics
                cs.cache_hits += 1
                cs.last_traces = comp_traces
                cs.last_interpreted_instructions = None
                cs.last_interpreted_history = None
                cs.last_prologue_traces = pro_traces
                cs.last_prologue = pro

                return cache_entry, inps, pro_to_epi

        cs.cache_misses += 1
        cs.last_trace_cache_stop = time.time_ns()

        # Resets use of compile flags
        cs.last_compile_reasons = defaultdict(list)

        with compile_data_and_stats(cd, cs):
            # Acquires the trace OR inlines the trace into an existing trace and
            #   returns the (proxied) result of the operation
            cs.last_trace_tracing_start = time.time_ns()

            with langctxs.langctx(cd.langctx):
                prologue_trc: TraceCtx
                computation_trc: TraceCtx
                prologue_trc, computation_trc, *maybe_epilogue = interpreter(
                    fn, args, kwargs, sharp_edges=cd.sharp_edges
                )

            if maybe_epilogue:
                epilogue_traces = maybe_epilogue
                if epilogue_traces[-1] is not None:
                    epilogue = epilogue_traces[-1].python_callable()
                else:
                    epilogue_traces = None
                    epilogue = None
            else:
                epilogue_traces = None
                epilogue = None

            cs.last_trace_tracing_stop = time.time_ns()

            # Makes the prologue callable
            cs.last_prologue_transformation_start = time.time_ns()
            protraces = transform_for_execution(
                prologue_trc,
                executors_list=(pythonex,),
                use_del_last_used=False,
            )
            protrace = protraces[-1]
            pro = protrace.python_callable()

            cs.last_prologue_transformation_stop = time.time_ns()

            cs.last_prologue_execution_start = time.time_ns()
            if epilogue:
                inps, pro_to_epi = pro(*args, **kwargs)
            else:
                inps = pro(*args, **kwargs)
                pro_to_epi = None
            cs.last_prologue_execution_stop = time.time_ns()

            computation_traces = [computation_trc]
            cs.last_traces = computation_traces
            backward_traces = []
            cs.last_backward_traces = backward_traces

            computation_trc = dce(computation_trc)
            computation_traces.append(computation_trc)

            if is_autocast_enabled:
                from thunder.core.transforms import autocast

                computation_trc = trace(compile_data=cd)(
                    autocast(computation_trc.python_callable(), dtype=autocast_thunder_dtype), *inps
                )
                computation_traces.append(computation_trc)

            backward_trc = None
            if not cd.disable_torch_autograd_support:
                tensor_cls = (pytorch.Tensor, TensorProxy)
                requires_grad = any(isinstance(arg, tensor_cls) and arg.requires_grad for arg in inps)

                if requires_grad:
                    # thunder_backward may recursively call compile and wraps the result in a
                    # torch.autograd.Function to support embedding of Thunder-compiled
                    # functions in torch's Autograd

                    # Currently split_forward_backward also includes
                    # transform_for_execution and various sorting of symbols,
                    # applying transform_for_execution after this would be
                    # breaking the order of operations
                    computation_trc, backward_trc = split_forward_backward(computation_trc, cd, cs, *inps)
                    # Note computation_trc and backward_trc have been appended to cs.last_(backward_)traces
                    # by split_forward_backward
                    extraces = cs.last_traces
                    check(
                        not additional_transforms,
                        lambda: "Specifying additional_transforms is not supported with PyTorch Autograd integration",
                    )

            if backward_trc is None:
                cs.last_computation_transformation_start = time.time_ns()

                ## EPILOGUE and TRANSFORMS should not mix...
                # applies transforms
                for transform in additional_transforms:
                    computation_trc = transform(computation_trc, executors_list=cd.executors_list)
                    computation_traces.append(computation_trc)

                with langctxs.langctx(cd.langctx):
                    extraces = transform_for_execution(
                        computation_trc,
                        executors_list=cd.executors_list,
                    )
                computation_trc = extraces[-1]
                cs.last_computation_transformation_stop = time.time_ns()

            comp = computation_trc.python_callable()

            if backward_trc is not None:
                backward_fn = backward_trc.python_callable()
            else:
                backward_fn = None

            # TODO RC1 Update the cache
            cache_entry = CacheEntry(
                pro, protraces, comp, extraces, epilogue, epilogue_traces, backward_fn, backward_traces
            )
            if cd.cache_option is not CACHE_OPTIONS.NO_CACHING:
                cs.interpreter_cache.append(cache_entry)

            cs.last_traces += extraces
            cs.last_prologue_traces = [prologue_trc] + protraces
            cs.last_prologue = pro

        return cache_entry, inps, pro_to_epi

    cd.get_computation_and_inputs = get_computation_and_inputs

    @wraps(fn)
    def fn_(*args, **kwargs) -> Any:
        if is_tracing():
            _recursive_jit_call_warning()
            return fn(*args, **kwargs)

        # Updats call statistics
        cs.last_trace_host_start = time.time_ns()
        cs.calls += 1

        cache_entry, inps, pro_to_epi = get_computation_and_inputs(*args, **kwargs)
        cs.last_trace_host_execution_start = time.time_ns()

        result = cache_entry.computation_fn(*inps)

        if cache_entry.backward_fn:
            # Run the compiled forward function
            data_for_autograd, (saved_tensors, saved_other) = result

            # Connect produced tensors with PyTorch's autograd graph
            ThunderFunction.apply(
                cache_entry.backward_fn,
                saved_tensors,
                saved_other,
                data_for_autograd["flat_output"],
                *data_for_autograd["flat_args"],
            )
            result = data_for_autograd["output"]

        if cache_entry.epilogue_fn:
            result, comp_to_epi = result
            cache_entry.epilogue_fn(*pro_to_epi, *comp_to_epi)

        cs.last_trace_host_execution_stop = time.time_ns()
        cs.last_computation_execution_stop = cs.last_trace_host_execution_stop

        cs.last_executed = cache_entry.computation_fn
        cs.last_trace_cache_stop = time.time_ns()
        cs.last_trace_host_stop = time.time_ns()

        return result

    if isinstance(fn, pytorch.nn.Module):
        fn_ = ThunderModule(fn, fn_)

    # Sets compile options and statistics attributes
    fn_._lc_cd = cd
    fn_._lc_cs = cs
    fn_._lc_transforms = additional_transforms[:]  ## transforms
    fn_._lc_post_optimization_transforms = []  ## post_optimization_transforms

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
    """Obtains the compilation data from a JITed function.

    The compile data (:class:`CompileData`) contains information about how the JIT has been configured
    for compilation (including referencing the function or module that is being compiled).
    """
    return getattr(fn, "_lc_cd", None)


def compile_stats(fn) -> CompileStats | None:
    """Obtains the compilation statistics from a JITed function.

    The compilation statistics (:class:`CompileStats`) contain information about each compilation run -
    collected when a JITed function is called for the first time or with previously unseen state.
    This includes the cache of traces (pologues, computation, possibly backward and epilogue) and
    how they have been transformed and information about cache hits and misses and timings.
    """
    return getattr(fn, "_lc_cs", None)


def last_traces(fn) -> list[TraceCtx]:
    """Obtains the list of computation traces that have been produced for the last run of the function. This is a list
    of traces mirroring the progression of transformations being applied to the trace (at index 0) that has
    been acquired from interpreting the user program.

    If the function has forward and backward, the forward is returned.
    """
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_traces


def last_backward_traces(fn) -> list[TraceCtx]:
    """Obtains the list of backward traces that have been produced for the last run of the function and the selected prologue."""
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_backward_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_backward_traces


def last_prologue_traces(fn) -> TraceCtx:
    """Obtains the list of prologue traces that have been produced for the last run of the function and the selected prologue."""
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_prologue_traces is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_prologue_traces


def cache_option(fn) -> CACHE_OPTIONS:
    """Returns the cache options set when JITting the function."""
    cd = compile_data(fn)
    if cd is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cd.cache_option


def cache_hits(fn) -> int:
    """Returns the number of cache hits we found when running the function."""
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cs.cache_hits


def cache_misses(fn) -> int:
    """Returns the number of cache misses we found when running the function."""
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    return cs.cache_misses


def list_transforms(fn) -> list:
    """Returns the list of (explicit) transforms applied to the JITed function."""
    return fn._lc_transforms


def last_interpreted_instructions(fn: Callable) -> list[dis.Instruction]:
    """Returns the list of instructions the interpreter encountered while tracing through the
    user program (on the last cache miss).
    """
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_interpreted_instructions is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_interpreted_instructions


def last_interpreted_history(fn: Callable) -> list[dis.Instruction | str]:
    """Returns the list of instructions and other information the interpreter encountered while tracing through the
    user program (on the last cache miss).
    """
    cs = compile_stats(fn)
    if cs is None:
        raise TypeError(f"{fn} doesn't seem to be a thunder compiled function.")
    if cs.last_interpreted_history is None:
        raise TypeError(f"{fn} doesn't seem to have been called yet.")
    return cs.last_interpreted_history


def last_compile_options(fn: Callable, /) -> None:
    """Prints how compiled options were used (or not)"""

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
