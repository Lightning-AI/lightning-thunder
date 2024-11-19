import dis
from typing import Any
from collections.abc import Callable, Generator, Hashable, Sequence
from collections import deque, defaultdict
import time
from functools import wraps
from io import StringIO
from weakref import WeakValueDictionary

from thunder.core.options import (
    CACHE_OPTIONS,
    resolve_cache_option,
    SHARP_EDGES_OPTIONS,
    resolve_sharp_edges_option,
    DebugOptions,
)
from thunder.core.utils import check, is_collection, AutocastStack
from thunder.core.pytree import tree_flatten, tree_map
from thunder.core.compile_data import compile_data_and_stats
import thunder.core.langctxs as langctxs
from thunder.core.langctxs import set_langctx, reset_langctx, LanguageContext, resolve_language, Languages
from thunder.core.codeutils import get_siginfo
from thunder.core.trace import (
    TraceCtx,
    get_tracectx,
    set_tracectx,
    reset_tracectx,
)
from thunder.core.transform_common import dce
from thunder.core.proxies import (
    is_proxyable,
    proxy,
    Proxy,
    CollectionProxy,
    TensorProxy,
    DistParallelType,
    FutureTensorProxy,
)
import thunder.core.prims as prims
import thunder.distributed as dist
import thunder.torch as ltorch  # we need to import thunder.torch before the below for import ordering...
from thunder.extend import Executor, get_default_executors, get_always_executors, OperatorExecutor, add_executor_lists
import thunder.executors as executors
from thunder.transforms.autocast import autocast
from thunder.core.dtypes import to_dtype

import torch as torch
import numpy as np
import thunder


__all__ = [
    "CompileStats",
    "CompileData",
    "cache_put",
    "cache_get",
    "trace",
    "transform_for_execution",
    "transform_to_torch_types",
]

#
# Datastructures for compiled functions
#


# Holds statistics and caches for a compiled function
# TODO RC1 Review how autograd traces are presented
class CompileStats:
    """A class holding statistics and caches for a compiled function.

    .. note::
        It is highly recommended that some attributes such as :attr:`CompileStats.last_traces` and
        :attr:`CompileStats.last_backward_traces` via :func:`thunder.last_traces` and
        :func:`thunder.last_backward_traces`, respectively.
        See :mod:`thunder` for more of such utility functions.

    Attributes:
        last_computation (Callable):
        last_traces (Sequence[TraceCtx]):
        last_prologue (TraceCtx):
        last_prologue_traces (Sequence[TraceCtx]):
        last_interpreted_instructions (Generator[dist.Instruction, None, None] | None):
        last_interpreter_log (list[InterpreterLogItem] | None):
        last_backward_traces (Sequence[TraceCtx]):
        last_trace_host_start (int):
        last_trace_host_stop (int):
        last_trace_cache_start (int):
        last_trace_cache_stop (int):
        last_trace_tracing_start (int):
        last_trace_tracing_stop (int):
        last_trace_host_execution_start (int):
        last_trace_host_execution_stop (int):
        last_prologue_transformation_start (int):
        last_prologue_transformation_stop (int):
        last_prologue_execution_start (int):
        last_prologue_execution_stop (int):
        last_computation_execution_start (int):
        last_computation_execution_stop (int):
        cache (dict):
        interpreter_cache (list):
        calls (int):
        cache_hits (int):
        cache_misses (int):
        last_compile_reasons (dict):
    """

    def __init__(self):
        # Callables and traces
        self.last_computation = None
        self.last_traces = None
        self.last_prologue = None
        self.last_prologue_traces = None
        self.last_interpreted_instructions: Generator[dis.Instruction, None, None] | None = None
        self.last_interpreter_log: list[InterpreterLogItem] | None = None

        # torch.autograd.Function specific data
        self.last_backward_traces = None

        # Timing stats
        self.last_trace_host_start: int = -1
        self.last_trace_host_stop: int = -1
        self.last_trace_cache_start: int = -1
        self.last_trace_cache_stop: int = -1
        self.last_trace_tracing_start: int = -1
        self.last_trace_tracing_stop: int = -1
        self.last_trace_host_execution_start: int = -1
        self.last_trace_host_execution_stop: int = -1

        self.last_prologue_transformation_start: int = -1
        self.last_prologue_transformation_stop: int = -1
        self.last_prologue_execution_start: int = -1
        self.last_prologue_execution_stop: int = -1
        self.last_computation_execution_start: int = -1
        self.last_computation_execution_stop: int = -1

        # Cache stats
        self.cache = {}
        self.interpreter_cache: list = []
        self.calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Compiler option stats
        self.last_compile_reasons: dict = defaultdict(list)

    def _time_template(self, start: int, stop: int, desc: str, /) -> int:
        if start < 0 or stop < 0 or stop < start:
            if start == -1 and stop == -1:
                raise AssertionError(f"Querying for {desc} time, but it seems that the function hasn't been called")
            raise AssertionError(f"The {desc} times {start=} and {stop=} were not recorded correctly")
        return stop - start

    def last_cache_lookup_time(self, /) -> int:
        start: int = self.last_trace_cache_start
        stop: int = self.last_trace_cache_stop
        return self._time_template(start, stop, "cache lookup")

    def last_trace_construction_time(self, /) -> int:
        start: int = self.last_trace_host_start
        stop: int = self.last_trace_host_stop
        return self._time_template(start, stop, "trace construction")

    def last_prologue_transformation_time(self, /) -> int:
        start: int = self.last_prologue_transformation_start
        stop: int = self.last_prologue_transformation_stop
        return self._time_template(start, stop, "prologue construction")

    def last_prologue_execution_time(self, /) -> int:
        start: int = self.last_prologue_execution_start
        stop: int = self.last_prologue_execution_stop
        return self._time_template(start, stop, "prologue execution")

    def last_computation_execution_time(self, /) -> int:
        start: int = self.last_computation_execution_start
        stop: int = self.last_computation_execution_stop
        return self._time_template(start, stop, "computation execution")


# A class that holds data about the compiled object, including statistics about how it's been called
# TODO Better document the module-related data the preprocessing harvests,
#   like additional_param_names
# TODO RC1 Rename this to CompileOptions
class CompileData:
    """A class holding data about the compiled object.

    Data include statistics about how it's been called.
    """

    def __init__(
        self,
        *,
        fn: Callable,
        langctx: None | LanguageContext = None,
        executors_list: None | tuple[Executor, ...] = None,
        cache_option: None | str | CACHE_OPTIONS = None,
        sharp_edges: None | SHARP_EDGES_OPTIONS | str = None,
        using_jit: bool = False,
        only_execute_prims: bool = False,
        disable_preprocessing: bool = False,
        disable_torch_autograd_support: bool = False,
        debug_log: None | StringIO = None,
        compile_options: dict[str, Any] = {},
        get_computation_and_inputs: Callable | None = None,
        executor_lookasides: dict[Callable, Callable] | None = None,
        debug_options: DebugOptions | None = None,
    ):
        # Records whether we're using the thunder.jit() entrypoint or not
        #   The thunder.jit() entrypoint introduces important architectural updates,
        #   but some components are still designed to work with the older entrypoint
        #   and are being temporarily maintained to facilitate their development.
        self.using_jit = using_jit

        # runs prologues to get the compute/backward/epilogue function and inputs
        self.get_computation_and_inputs = get_computation_and_inputs

        # lookasides provided by the executors
        self.executor_lookasides = executor_lookasides

        # Resolves cache option
        self.cache_option = resolve_cache_option(cache_option)

        # State for pytorch autocast context managers.
        self.autocast_stack: AutocastStack = AutocastStack()

        # State to query whether grad is enabled or disabled using
        # torch.no_grad/torch.enable_grad/torch._C._set_grad_enabled
        self.is_grad_enabled: bool = True

        #
        # Gathers additional metadata
        #

        # Constructs executors list
        # NOTE The executors list is fixed at compile time
        always_executors: tuple[Executor] = get_always_executors()
        if executors_list is None:
            self.executors_list = get_default_executors() + always_executors
        else:
            self.executors_list = tuple(executors_list)
            # Adds always executors (if not present)
            if self.executors_list[-len(always_executors) :] != always_executors:
                self.executors_list = add_executor_lists(self.executors_list, always_executors)

        # Validates executors list
        for ex in self.executors_list:
            assert isinstance(
                ex, Executor
            ), f"Expected all elements of the executors list to be executors, but found {ex}"

        # Resolves language context (defaulting to the torch language)
        self.langctx = langctx if langctx is not None else resolve_language(Languages.TORCH)
        if not isinstance(self.langctx, LanguageContext):
            raise ValueError(
                f"Attempting to construct a CompileData object with an invalid language context type {type(self.langctx)}"
            )

        # Resolves sharp edges option
        self.sharp_edges = resolve_sharp_edges_option(sharp_edges)

        self.fn = fn
        self.only_execute_prims = only_execute_prims
        self.disable_preprocessing = disable_preprocessing
        self.disable_torch_autograd_support = disable_torch_autograd_support
        self.debug_log = debug_log
        self.debug_options = DebugOptions() if debug_options is None else debug_options

        # TODO Consider validating that this dict has exclusively string keys
        self.compile_options = compile_options

        self.is_module = isinstance(self.fn, torch.nn.Module)

        # to not introduce (more) ref cycles, use WeakValueDictionary and map from int->ThunderModule
        # but the accessor has to check if tmm[id(module)]._module is module
        self._thunder_module_map = WeakValueDictionary()

        # We set the process_group_for_ddp attribute on the module when
        # thunder.distributed.ddp(module) is called.
        self.process_group_for_ddp = getattr(self.fn, "process_group_for_ddp", None)

        #
        # Possibly processes the function
        #
        self.additional_param_names = None
        self.additional_param_values = None
        self.additional_return_names = None
        self.num_constant_args = 0

        assert disable_preprocessing, "please use thunder.compile if you need preprocessing"


# Common UX functions
def _unpack_inputs(fn, tracectx: TraceCtx, args, kwargs, *, rename_proxies: bool):
    tracectx.unpacking()

    # Translates tensors, arrays, and dtypes to thunder.jit types
    # TODO Translate NumPy dtypes
    def translate(x: Any, *, name: str | None = None) -> Any:
        # NOTE Unpacking proxies
        # When we encounter a proxy, we need to make sure that it's name is the
        # same as the name that the unpack is requesting. If it's not, we need to
        # create a new proxy with the requested name.
        # TODO: There might be better ways to do this, but this is the simplest
        #   way to get it working correctly now.
        #   One alternative would be to modify the function's signature to include
        #   the name of the proxy, but that might require a lot of changes to the
        #   codebase.
        if is_proxyable(x):
            return proxy(x, name=name)

        if isinstance(x, Proxy):
            # register proxy name used by NumberProxies in TensorProxy.shape
            if isinstance(x, TensorProxy):
                for s_p in filter(lambda s: isinstance(s, Proxy), x._shape):
                    # TODO need to avoid name conflict here, since s_p.name
                    # could have conflicted with something defined earlier in
                    # the trace.
                    get_tracectx().names.add(s_p.name)
            if not rename_proxies:
                get_tracectx().names.add(x.name)
                return x
            return x.replace_name(name)
        if isinstance(x, torch.dtype):
            return to_dtype(x)
        if is_collection(x):
            return tree_map(translate, x)

        return x

    # Construct proxy args and kwargs by parsing signature and analyzing inputs
    si = get_siginfo(fn, args, kwargs)

    # Constructs args
    cq = deque()
    proxyargs = []
    for name, x in si.args:
        translated = translate(x, name=name)
        proxyargs.append(translated)

        unpacked = prims.unpack_trivial(translated, name=name)
        if isinstance(unpacked, CollectionProxy):
            cq.append(unpacked)

    # Handles varargs
    if si.varargs is not None:
        varargs_name, x = si.varargs

        translated = translate(x)
        proxyargs.extend(translated)

        unpacked = prims.unpack_trivial(translated, name=varargs_name)
        cq.append(unpacked)

    proxykwargs = {}
    for name, x in si.kwargs.items():
        translated = translate(x, name=name)
        proxykwargs[name] = translated

        unpacked = prims.unpack_trivial(translated, name=name)
        if isinstance(unpacked, CollectionProxy):
            cq.append(unpacked)

    if si.varkwargs is not None:
        varkwargs_name, x = si.varkwargs

        translated = translate(x)
        proxykwargs.update(translated)

        unpacked = prims.unpack_trivial(translated, name=varkwargs_name)
        cq.append(unpacked)

    # Unpacks collections to introduce proxy names to the trace
    while True:
        try:
            c = cq.popleft()
            unpacked = prims.unpack(c)
            for u in unpacked:
                if isinstance(u, CollectionProxy):
                    cq.append(u)
        except IndexError:
            break

    tracectx.unpacked()
    return proxyargs, proxykwargs


#
# Caching objects and functions
#
# TODO We could look at supporting non-hashable inputs, like dicts


# TODO Update cacheable types
def _make_subkey_for(x: Any) -> tuple[bool, None | tuple]:
    if isinstance(x, (torch.Tensor, TensorProxy)):
        return True, (type(x), x.shape, x.device, x.dtype, x.requires_grad)

    # TODO Add NumPy ndarray support
    if isinstance(x, np.ndarray):
        return False, None

    # NOTE Special cases strings because strings are Sequences, but we want to treat them like non-Sequence objects
    if isinstance(x, str):
        return True, (str, x)

    if isinstance(x, Sequence):
        key = [None] * len(x)
        for idx, v in enumerate(x):
            is_hashable, subkey = _make_subkey_for(v)
            if not is_hashable:
                return None, False
            key[idx] = subkey
        return True, tuple(key)

    # TODO Add support for additional collections (like dicts)
    if is_collection(x):
        return False, None

    if isinstance(x, Hashable):
        return True, (type(x), x)

    return False, None


# Private class just to separate objects in the cache
class _key_value_separator:
    pass


# Returns a hashable key or None if the given args and kwargs are not hashable
def _make_cache_key(
    args,
    kwargs,
    autocast_key=None,
    distributed_key=None,
) -> None | tuple:
    key = [None] * (len(args) + len(kwargs))

    # Constructs arg portion of key
    for idx, arg in enumerate(args):
        is_hashable, subkey = _make_subkey_for(arg)
        if not is_hashable:
            return None
        key[idx] = subkey

    # Constructs kwarg portion of key

    def kwarg_helper(key, value) -> tuple[bool, None | tuple]:
        is_key_hashable, key_key = _make_subkey_for(key)
        is_value_hashable, value_key = _make_subkey_for(value)

        return is_key_hashable and is_value_hashable, (key_key, _key_value_separator, value_key)

    offset = len(args)
    for idx, (k, v) in enumerate(kwargs.items()):
        is_hashable, subkey = kwarg_helper(k, v)
        if not is_hashable:
            return None

        key[offset + idx] = subkey

    if autocast_key is not None:
        key += autocast_key
    if distributed_key is not None:
        key += distributed_key

    return tuple(key)


# construct cache key for autocast operations
def _make_autocast_cache_key(
    is_autocast_enabled, is_autocast_cpu_enabled, autocast_gpu_dtype, autocast_cpu_dtype
) -> list:
    return [is_autocast_enabled, is_autocast_cpu_enabled, autocast_gpu_dtype, autocast_cpu_dtype]


def _make_distributed_cache_key(
    no_grad_sync: bool,
) -> list[bool]:
    return [no_grad_sync]


# Returns True if successfully cached, false otherwise
def cache_put(
    cache: dict,
    fn,
    traces: Sequence[TraceCtx],
    args,
    kwargs,
    autocast_key=None,
    distributed_key=None,
) -> bool:
    key = _make_cache_key(args, kwargs, autocast_key, distributed_key)

    if key is None:
        return False

    cache[key] = (fn, traces)
    return True


def cache_get(
    cache: dict,
    args,
    kwargs,
    autocast_key=None,
    distributed_key=None,
) -> tuple[None | Callable, None | Sequence[TraceCtx]]:
    key = _make_cache_key(args, kwargs, autocast_key, distributed_key)
    return cache.get(key, (None, None))


# Produces a trace of the given function with the given args and kwargs
# If inline_trace is True and this is called while tracing then
#   the trace will be inlined into the current trace, and instead of a trace
#   the results of the function will be returned
# If inline_trace is False then this will always produce a new trace.
#   If this is called while already tracing then the tracing context that
#   calls this will not observe those calls
# If rename_proxies is True then new proxy inputs are generated when the function is called.
# If rename_proxies is False then proxy inputs are passed to the function unmodified.
#   This can be useful when trace() is called in a context where proxies have already
#   been constructed.
# If include_return_statement is True then the trace will terminate with a RETURN operation
# If include_return_statement is False then the trace will end without an explicit RETURN
# If `_used_names` is provided, then these names will not appear in the newly constructed trace.
# This arguments is useful, if we don't want certain names to appear in the new trace
# Example: We don't want names from forward trace to appear in backward trace unless passed as input.
# TODO Consider modeling additional calls to trace()
# TODO RC1 Change the way this is called to be trace(langctx, inline_trace, rename_proxies...)(fn, *args, **kwargs)
#   to separate the traced function's args and kwargs from this function's kwargs
from thunder.core.interpreter import InterpreterLogItem, make_opaque


def trace(
    compile_data: None | CompileData = None,
    inline_trace: bool = True,
    rename_proxies: bool = True,
    include_return_statement: bool = True,
    use_dce: bool = True,
    insert_ddp_syncs: bool = False,
    _used_names: set[str] | None = None,
) -> Callable:
    @make_opaque
    def _trace(
        fn,
        *args,
        **kwargs,
    ) -> Any | TraceCtx:
        # Resolves language context
        # TODO RC1 Don't require the isinstance check here -- compile data should do this (and make langctx a property)
        langctx_: LanguageContext = resolve_language(Languages.TORCH)
        if compile_data is not None and isinstance(compile_data.langctx, LanguageContext):
            langctx_ = compile_data.langctx

        try:
            langctx_tok = set_langctx(langctx_)
            current_trace = get_tracectx()
            tracectx_tok = None

            if current_trace is not None and inline_trace:
                return fn(*args, **kwargs)

            trace = TraceCtx(fn)

            # Add `_used_names` to the trace, so that
            # they won't be used for proxies generated
            # while tracing.
            if _used_names is not None:
                for name in _used_names:
                    trace.add_name(name)

            tracectx_tok = set_tracectx(trace)

            proxyargs, proxykwargs = args, kwargs
            proxyargs, proxykwargs = _unpack_inputs(fn, trace, args, kwargs, rename_proxies=rename_proxies)
            trace.args, trace.kwargs = proxyargs, proxykwargs

            if insert_ddp_syncs:
                from thunder.core import utils
                from thunder.distributed import get_skip_data_parallel_grad_sync

                no_sync = get_skip_data_parallel_grad_sync()
                utils.check(
                    not (no_sync and getattr(compile_data, "use_fsdp", False)),
                    lambda: "`thunder.distributed.fsdp` does not support `no_sync`",
                )

                def ddp_sync(arg: Any | TensorProxy) -> Any | TensorProxy:
                    if isinstance(arg, TensorProxy) and arg.distparallel_type in (
                        DistParallelType.REPLICATED,
                        DistParallelType.FULLY_SHARDED,
                    ):
                        return dist.prims.synchronize(arg, compile_data.process_group_for_ddp)
                    else:
                        return arg

                if not no_sync:
                    proxyargs, proxykwargs = tree_map(ddp_sync, (proxyargs, proxykwargs))

            result = fn(*proxyargs, **proxykwargs)

            if include_return_statement:

                def wait_for_future(f: FutureTensorProxy) -> TensorProxy:
                    if isinstance(f, FutureTensorProxy):
                        return f.wait()
                    return f

                # It's a safety check to make sure that we don't return a future
                # tensor from a traced function.
                if trace._any_future_tensors:
                    result = tree_map(wait_for_future, result)

                prims.python_return(result)

            trace.mark_complete()

            # TODO Stop calling this here and make it a separate trace in the sequence
            #   of traces
            if use_dce:
                trace = dce(trace)

        finally:
            # Resets contexts
            reset_langctx(langctx_tok)

            if tracectx_tok is not None:
                reset_tracectx(tracectx_tok)

        return trace

    return _trace


# TODO Remove executor-specific passes
# TODO Constraint generation based off executor requirements
# TODO Consider making this faster by reusing more data
# TODO Create a general mechanism for running traces that produces reproducible provenance and the
#   appropriate error checks
def transform_for_execution(
    trace: TraceCtx,
    executors_list: Sequence[Executor],
    *,
    only_execute_prims=False,
    use_del_last_used=True,
) -> list[TraceCtx]:
    traces: list[TraceCtx] = []

    # TODO If only_execute_prims, then flatten to prims here

    # Runs passes that are generally useful
    dce_trace = dce(trace)
    traces.append(dce_trace)

    # cse_trace = cse(dce_trace)
    # traces.append(cse_trace)

    extrace = executors.passes.transform_for_execution(dce_trace, executors_list)

    traces.append(extrace)

    if use_del_last_used:
        lifetime_trace = executors.passes.del_last_used(extrace)
        traces.append(lifetime_trace)

    return traces


# NOTE: Do not use this function and do not update it.
# Use `thunder.jit` instead.
def _execute_trace(
    trc: TraceCtx,
    *,
    args,
    kwargs,
    compile_data: CompileData,
    transforms: list[Callable] = [],
) -> tuple[Any, Callable, list[TraceCtx]]:
    # Transforms the trace for execution
    # TODO Add the capability to recover from pass failures

    with langctxs.langctx(compile_data.langctx):
        extraces = transform_for_execution(
            trc,
            executors_list=compile_data.executors_list,
            only_execute_prims=compile_data.only_execute_prims,
        )
    extrace = extraces[-1]

    # Applies post-optimization transforms
    for transform in transforms:
        extrace = transform.transform_trace_post_optimization(extrace)
        extraces.append(extrace)

    # Constructs the Python callable
    c = extrace.python_callable()

    # Executes the operation
    result: Any = c(*args, **kwargs)

    return result, c, extraces


# Constructs a function that returns its output + the trace for further analysis
# TODO probably a better name for this?
# TODO review functions which compute large objects unrelated to tensors and how
#   they're handled
# TODO can the language context be detected from the inputs?
# TODO:
#   Today all tensor outputs will be torch tensors, even if the input was NumPy arrays
#   provided in the NumPy language ctx -- what should the outputs be?  Should we provide
#   a helper to convert torch tensors to NumPy arrays on output?


def transform_to_torch_types(trace: TraceCtx):
    # convert the thunder types to torch types if any
    def map_to_torch(x: Any) -> Any:
        if isinstance(x, thunder.dtypes.dtype):
            return thunder.dtypes.to_torch_dtype(x)
        elif isinstance(x, thunder.devices.Device):
            return thunder.devices.to_torch_device(x)
        return x

    last = trace.bound_symbols[-1]
    assert last.sym.id == prims.PrimIDs.RETURN
    new_args = tree_map(map_to_torch, last.args)
    new_bsym = prims.python_return.bind(*new_args, output=None)
    trace.bound_symbols[-1] = new_bsym
    return trace
