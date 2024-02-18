from typing import Any, Optional
from collections.abc import Callable
from enum import Enum, auto
from collections import deque, defaultdict
from contextlib import contextmanager
import time
from collections.abc import Hashable, Sequence
from functools import wraps
import os
from io import StringIO

from thunder.core.options import CACHE_OPTIONS, resolve_cache_option
from thunder.core.utils import check, is_collection
from thunder.core.pytree import tree_flatten, tree_map
from thunder.cudagraphs import CUDAGraphExecutor
from thunder.core.compile_data import compile_data_and_stats
from thunder.core.langctxs import set_langctx, reset_langctx, LanguageContext, resolve_language, Languages
from thunder.core.codeutils import get_siginfo
from thunder.core.trace import (
    TraceCtx,
    get_tracectx,
    set_tracectx,
    reset_tracectx,
)
from thunder.core.transform_common import dce, cse
from thunder.core.proxies import is_proxyable, proxy, Proxy, CollectionProxy, TensorProxy, DDPType, FutureTensorProxy
import thunder.core.prims as prims
import thunder.distributed as dist
import thunder.torch as ltorch
from thunder.extend import Executor, get_default_executors, get_always_executors, OperatorExecutor
import thunder.executors as executors
from thunder.executors.torch_autograd import thunder_backward
from thunder.core.transforms import autocast
from thunder.core.dtypes import to_dtype

import torch as torch
import numpy as np

#
# Datastructures for compiled functions
#


# Holds statistics and caches for a compiled function
# TODO GTC Update last_executed to last_computation
# TODO GTC Review how autograd traces are presented
class CompileStats:
    def __init__(self):
        # Callables and traces
        self.last_executed = None
        self.last_traces = None
        self.last_prologue = None
        self.last_prologue_traces = None
        self.last_interpreted_instructions = None
        self.last_interpreted_history = None

        # torch.autograd.Function specific data
        self.primal_trace = None
        self.forward_last_traces = None
        self.backward_last_traces = None

        # Timing stats
        self.last_trace_host_start: int = -1
        self.last_trace_host_stop: int = -1
        self.last_trace_cache_start: int = -1
        self.last_trace_cache_stop: int = -1
        self.last_trace_tracing_start: int = -1
        self.last_trace_tracing_stop: int = -1
        self.last_trace_host_execution_start: int = -1
        self.last_trace_host_execution_stop: int = -1

        # Cache stats
        self.cache = {}
        self.interpreter_cache: list = []
        self.calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Compiler option stats
        self.last_compile_reasons: dict = defaultdict(list)


import thunder.core.script.frontend as script_frontend
import thunder.core.script.instrumentation as script_instrumentation
import thunder.core.script.passes as passes
import thunder.core.script.python_ir as python_ir


# Preprocesses function
# Currently tries to map torch.foo lookups to thunder.torch.foo lookups
@script_instrumentation.record
def preprocess(fn, is_module):
    gr = script_frontend.acquire_method(fn.forward if is_module else fn)
    passes.unroll_for_loops_and_inline_modules(gr)
    if is_module:
        (
            additional_param_names,
            additional_param_values,
            additional_return_names,
        ) = passes.module_to_function(gr)
    passes.strongly_inline_functions(gr)
    passes.torch_to_thunder(gr)

    thunder_fn = python_ir.generate_function(gr)
    if is_module:
        thunder_fn._additional_param_names = additional_param_names
        thunder_fn._additional_param_values = additional_param_values
        thunder_fn._additional_return_names = additional_return_names
    else:
        thunder_fn._additional_param_names = None
        thunder_fn._additional_param_values = None
        thunder_fn._additional_return_names = None

    return thunder_fn


# A class that holds data about the compiled object, including statistics about how it's been called
# TODO Better document the module-related data the preprocessing harvests,
#   like additional_param_names
# TODO GTC Rename this to CompileOptions
class CompileData:
    def __init__(
        self,
        *,
        fn: Callable,
        langctx: None | LanguageContext = None,
        executors_list: None | tuple[Executor, ...] = None,
        cache_option: None | str | CACHE_OPTIONS = None,
        only_execute_prims: bool = False,
        disable_preprocessing: bool = False,
        use_cudagraphs: bool = False,
        use_torch_compile: bool = False,
        disable_torch_autograd_support: bool = False,
        use_rematerialization: bool = False,
        debug_log: None | StringIO = None,
        compile_options: dict[str, Any] = {},
    ):
        # Resolves cache option
        self.cache_option = resolve_cache_option(cache_option)

        #
        # Gathers additional metadata
        #

        # Constructs executors list
        # NOTE The executors list is fixed at compile time
        if executors_list is None:
            self.executors_list = tuple(get_default_executors() + get_always_executors())
        else:
            self.executors_list = tuple(executors_list) + tuple(get_always_executors())

        # Validates executors list
        for ex in self.executors_list:
            assert isinstance(
                ex, Executor
            ), f"Expected all elements of the executors list to be executors, but found {ex}"

        self.fn = fn
        self.langctx = langctx if langctx is not None else resolve_language(Languages.TORCH)
        if not isinstance(self.langctx, LanguageContext):
            raise ValueError(
                f"Attempting to construct a CompileData object with an invalid language context type {type(self.langctx)}"
            )

        self.only_execute_prims = only_execute_prims
        self.disable_preprocessing = disable_preprocessing
        self.use_rematerialization = use_rematerialization
        self.use_cudagraphs = use_cudagraphs
        self.use_torch_compile = use_torch_compile
        self.disable_torch_autograd_support = disable_torch_autograd_support
        self.debug_log = debug_log

        # TODO Consider validating that this dict has exclusively string keys
        self.compile_options = compile_options

        self.is_module = isinstance(self.fn, torch.nn.Module)

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

        self._processed_function: Callable
        if disable_preprocessing:
            self._processed_function = fn
        else:
            self._processed_function = preprocess(fn, is_module=self.is_module)

            # TODO Revisit assuming parameters are const
            if self.is_module:
                self.additional_param_names = self.processed_function._additional_param_names
                self.additional_param_values = self.processed_function._additional_param_values
                self.additional_return_names = self.processed_function._additional_return_names
                self.num_constant_args = len(self.additional_param_values)

    # Disallows overwriting processed_function
    @property
    def processed_function(self):
        return self._processed_function


# Common UX functions
def _unpack_inputs(fn, tracectx: TraceCtx, args, kwargs, *, rename_proxies: bool):
    tracectx.unpacking()

    # Translates tensors, arrays, and dtypes to lightning.compile types
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


class ThunderOptimizedModule(torch.nn.Module):  # TOM
    # todo: subclass nn.Module or forward things like .state_dict() to the
    #       model
    def __init__(self, model, fn, tfn, additional_param_names, additional_param_values, additional_return_names):
        super().__init__()
        self._model = model
        self._forward_fn = fn
        self._tfn = tfn

        # Note [DistributedDataParallel and ddp_type]
        # If model was wrapped with thunder.distributed.ddp it would have a
        # .use_ddp attribute set to True and all parameters would be already
        # broadcasted to all other processes. So that our tracing is aware of
        # this we need to mark the ddp_type of model's parameters as
        # thunder.proxies.DDPType.REPLICATED
        if getattr(model, "use_ddp", False):
            for v in additional_param_values:
                v.ddp_type = DDPType.REPLICATED

        # If model was wrapped with thunder.distributed.fsdp it would have a
        # .use_fsdp attribute set to True and all parameters would be already
        # sharded across all other processes. So that our tracing is aware of
        # this we need to mark the ddp_type of model's parameters as
        # thunder.proxies.DDPType.FULLY_SHARDED
        if getattr(model, "use_fsdp", False):
            for v in additional_param_values:
                v.ddp_type = DDPType.FULLY_SHARDED

        self._additional_param_values = additional_param_values
        self._additional_param_names = additional_param_names
        self._additional_return_names = additional_return_names
        d = {k: i for i, k in enumerate(additional_param_names)}
        self._additional_return_param_idxes = [d[k] for k in additional_return_names]

    def __call__(self, *args, **kwargs):
        all_args = (*self._additional_param_values, *args)
        res = self._forward_fn(*all_args, **kwargs)
        if self._additional_return_names:
            res, *additional_returns = res
            assert len(self._additional_return_names) == len(
                additional_returns
            ), f"Number of expected additional return args {len(self._additional_return_names)=} does not match the actual number {len(additional_returns)=}"
            for k, v, idx in zip(
                self._additional_return_names, additional_returns, self._additional_return_param_idxes
            ):
                m = self._model
                parts = k.split(".")
                for p in parts[:-1]:
                    m = getattr(m, p)
                setattr(m, parts[-1], v)
                self._additional_param_values[idx] = v
        return res

    @contextmanager
    def no_sync(self):
        """Context manager to disable gradient synchronization in data parallel mode.

        This context manager is intended to be used in conjunction with
        :class:`torch.nn.parallel.DistributedDataParallel` to disable gradient
        synchronization in the backward pass. It will not have any effect when
        used with other modules.

        .. note::

            This could lead to different accumulated gradients with ``torch.nn.parallel.distributed.DistributedDataParallel.no_sync``.
            PyTorch's gradient synchronization is implemented by applying all-reduce to gradient buckets of ``torch.nn.Parameter.grad``.
            Thus the ``no_sync`` context leads to :math:`\text{AllReduce} \\left( \\sum_{i = 0}^{\rm{num_grad_accum_steps}} g_i \right)`.
            In contrast, this synchronizes accumulated gradients when exiting, leading to
            :math:`\text{AllReduce} \\left( \\sum_{i = 0}^{\rm{num_grad_accum_steps - 1}} g_i \right) + \text{AllReduce}(g_{\rm{num_grad_accum_steps}})`.

        """
        from torch.distributed import distributed_c10d as c10d

        from thunder.distributed import set_skip_data_parallel_grad_sync
        from thunder.distributed import reset_skip_data_parallel_grad_sync

        token = set_skip_data_parallel_grad_sync(True)
        try:
            yield
        finally:
            reset_skip_data_parallel_grad_sync(token)

            params_with_grad = list(filter(lambda p: p.grad is not None, self.parameters()))
            grads = [p.grad for p in params_with_grad]
            process_group = self._lc_cd.process_group_for_ddp
            torch._foreach_div_(grads, process_group.size())
            with c10d._coalescing_manager(
                group=process_group,
                async_ops=True,
            ) as cm:
                for p in params_with_grad:
                    c10d.all_reduce(p.grad)
            cm.wait()


#
# Caching objects and functions
#
# TODO We could look at supporting non-hashable inputs, like dicts


# TODO Update cacheable types
def _make_subkey_for(x: Any) -> tuple[bool, None | tuple]:
    if isinstance(x, torch.Tensor):
        return True, (torch.Tensor, x.shape, x.device, x.dtype, x.requires_grad)

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
# TODO Consider modeling additional calls to trace()
# TODO GTC Change the way this is called to be trace(langctx, inline_trace, rename_proxies...)(fn, *args, **kwargs)
#   to separate the traced function's args and kwargs from this function's kwargs


def trace(
    compile_data: None | CompileData = None,
    inline_trace: bool = True,
    rename_proxies: bool = True,
    include_return_statement: bool = True,
    use_dce: bool = True,
    insert_ddp_syncs: bool = False,
) -> Callable:
    def _trace(
        fn,
        *args,
        **kwargs,
    ) -> Any | TraceCtx:
        # Resolves language context
        # TODO GTC Don't require the isinstance check here -- compile data should do this (and make langctx a property)
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
                    if isinstance(arg, TensorProxy) and arg.ddp_type in (DDPType.REPLICATED, DDPType.FULLY_SHARDED):
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
    use_rematerialization=True,
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


# Executes the trace with the given args and kwargs and returns the result,
#   the callable executed, and the series of traces constructed to produce
#   that callable from the trace
def _execute_trace(
    trc: TraceCtx,
    *,
    args,
    kwargs,
    compile_data: CompileData,
    post_optimization_transforms: list[Callable] = [],
) -> tuple[Any, Callable, list[TraceCtx]]:
    # Transforms the trace for execution
    # TODO Add the capability to recover from pass failures

    extraces = transform_for_execution(
        trc,
        executors_list=compile_data.executors_list,
        only_execute_prims=compile_data.only_execute_prims,
        use_rematerialization=compile_data.use_rematerialization,
    )
    extrace = extraces[-1]

    # Applies post-optimization transforms
    for transform in post_optimization_transforms:
        extrace = transform(extrace)
        extraces.append(extrace)

    # Constructs the Python callable
    c = extrace.python_callable()

    # TODO GTC Remove this option (by modeling torch.compile as another executor)
    if compile_data.use_torch_compile:
        c = torch.compile(c)

    # TODO GTC Mark this option as experimental
    if compile_data.use_cudagraphs:
        c = CUDAGraphExecutor(c, num_constant_args=compile_data.num_constant_args)

    # Executes the operation
    result: Any = c(*args, **kwargs)

    return result, c, extraces


# Constructs a function that returns its output + the trace for further analysis
# TODO probably a better name for this?
# TODO review functions which compute large objects unrelated to tensors and how
#   they're handled
# TODO can the language context be detected from the inputs?
# TODO  https://github.com/Lightning-AI/lightning-thunder/issues/316
#   Today all tensor outputs will be torch tensors, even if the input was NumPy arrays
#   provided in the NumPy language ctx -- what should the outputs be?  Should we provide
#   a helper to convert torch tensors to NumPy arrays on output?
# TODO Provide an option to not preprocess (for debugging)


def _create_callable(
    cd: CompileData,
    cs: CompileStats,
    *,
    transforms: list[Callable] = [],
    post_optimization_transforms: list[Callable] = [],
    _using_grad_transform: bool = False,
) -> Callable:
    @wraps(cd.processed_function)
    def _fn(*args, **kwargs) -> tuple[Any, list[TraceCtx]]:
        cs.last_trace_host_start = time.time_ns()
        cs.calls += 1

        # autocast related operations
        is_autocast_enabled = False
        autocast_key = None
        if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
            if torch.is_autocast_enabled() and torch.is_autocast_cpu_enabled():
                raise NotImplementedError(
                    "thunder.autocast does not support torch.is_autocast_enabled() and torch.is_autocast_cpu_enabled() simultaneously at this moment."
                )
            is_autocast_enabled = True
            autocast_gpu_dtype = to_dtype(torch.get_autocast_gpu_dtype())
            autocast_cpu_dtype = to_dtype(torch.get_autocast_cpu_dtype())
            autocast_key = _make_autocast_cache_key(
                torch.is_autocast_enabled(), torch.is_autocast_cpu_enabled(), autocast_gpu_dtype, autocast_cpu_dtype
            )
            autocast_thunder_dtype = autocast_cpu_dtype if torch.is_autocast_cpu_enabled() else autocast_gpu_dtype

        # TODO(crcrpar): support FSDP as well
        is_ddp_enabled = getattr(cd.fn, "use_ddp", False)
        no_grad_sync = False
        if is_ddp_enabled:
            from thunder.distributed import get_skip_data_parallel_grad_sync

            no_grad_sync = get_skip_data_parallel_grad_sync()
        distributed_key = _make_distributed_cache_key(no_grad_sync)

        # Tries to lookup a callable in a cache
        # TODO Return the previous traces when caching
        cs.last_trace_cache_start = time.time_ns()
        if cd.cache_option is CACHE_OPTIONS.SAME_INPUT and cs.last_executed is not None:
            # TODO Update _last_traces?
            # Updates statistics before early termination
            cs.cache_hits += 1
            cs.last_trace_cache_stop = time.time_ns()
            cs.last_trace_tracing_start = -1
            cs.last_trace_tracing_stop = -1
            cs.last_trace_host_execution_start = time.time_ns()
            result = cs.last_executed(*args, **kwargs)
            cs.last_trace_host_execution_stop = time.time_ns()
            cs.last_trace_host_stop = cs.last_trace_host_execution_stop
            return result
        if cd.cache_option is CACHE_OPTIONS.CONSTANT_VALUES:
            c, _ = cache_get(cs.cache, args[cd.num_constant_args :], kwargs, autocast_key, distributed_key)
            if c is not None:
                # Updates statistics before early termination
                cs.cache_hits += 1
                cs.last_executed = c
                cs.last_trace_cache_stop = time.time_ns()
                cs.last_trace_tracing_start = -1
                cs.last_trace_tracing_stop = -1
                cs.last_trace_host_execution_start = time.time_ns()
                result = c(*args, **kwargs)
                cs.last_trace_host_execution_stop = time.time_ns()
                cs.last_trace_host_stop = cs.last_trace_host_execution_stop
                return result
        cs.cache_misses += 1
        cs.last_trace_cache_stop = time.time_ns()

        # Applies the autocast transform if PyTorch's autocast behavior is enabled
        processed_function = (
            cd.processed_function
            if not is_autocast_enabled
            else autocast(cd.processed_function, dtype=autocast_thunder_dtype)
        )

        # Resets use of compile flags
        cs.last_compile_reasons = defaultdict(list)
        with compile_data_and_stats(cd, cs):
            # Determines whether to use autograd.Function or not
            # autograd.Function (which supports calling .backward() in PyTorch) is used when:
            #   1) The grad() transform is not applied
            #   2) At least one input tensor (or tensor proxy) requires grad
            if not _using_grad_transform:
                flat_args, _ = tree_flatten((args, kwargs))
                tensor_cls = (torch.Tensor, TensorProxy)
                requires_grad = any(isinstance(arg, tensor_cls) and arg.requires_grad for arg in flat_args)
                if not cd.disable_torch_autograd_support and requires_grad:
                    compile_config = {
                        "langctx": cd.langctx,
                        "executors_list": cd.executors_list,
                        "only_execute_prims": cd.only_execute_prims,
                        "cache_option": cd.cache_option,
                        "use_rematerialization": cd.use_rematerialization,
                        "use_cudagraphs": cd.use_cudagraphs,
                        **cd.compile_options,
                    }

                    # thunder_backward may recursively call compile and wraps the result in a
                    # torch.autograd.Function to support embedding of Thunder-compiled
                    # functions in torch's Autograd
                    cs.last_trace_host_execution_start = time.time_ns()
                    c = thunder_backward(compile_data=cd, compile_stats=cs, **compile_config)(processed_function)
                    result = c(*args, **kwargs)
                    cs.last_trace_host_execution_stop = time.time_ns()
                    cs.last_executed = c
                    if cd.cache_option is CACHE_OPTIONS.CONSTANT_VALUES:
                        cache_put(
                            cs.cache,
                            c,
                            None,
                            args[cd.num_constant_args :],
                            kwargs,
                            autocast_key=None,
                            distributed_key=distributed_key,
                        )
                    cs.last_trace_host_stop = time.time_ns()
                    return result

            # TODO Revisit compile() behavior when hit in a trace ctx
            #   This will inline the invocation of compile into the current
            #   trace (UNLESS there was a cache hit, per above)
            #   This interaction between the cache and tracing seems odd
            # TODO Support a practitioner who wants to explicitly and separately compile
            #   part of the program

            # Acquires the trace OR inlines the trace into an existing trace and
            #   returns the (proxied) result of the operation
            cs.last_trace_tracing_start = time.time_ns()
            trc_or_result = trace(compile_data=cd)(processed_function, *args, **kwargs)
            cs.last_trace_tracing_stop = time.time_ns()

            # Checks for inlined transforms
            current_trace = get_tracectx()
            check(
                current_trace is None or len(transforms) == 0,
                lambda: f"Inlining transformed traces is not yet supported",
                exception_type=NotImplementedError,
            )

            # Returns the (proxied) result if this call to compile was inlined
            if current_trace is not None:
                result = trc_or_result
                return result

            # Starts recording a sequence of traces (this is not inlined)
            trc: TraceCtx = trc_or_result
            traces: list[TraceCtx] = [trc]

            # Applies transforms
            for transform in transforms:
                trc = transform(trc, executors_list=cd.executors_list)
                traces.append(trc)

            #
            # Executes the trace, then updates the CompiledData and possibly
            #   updates a cache
            #
            cs.last_trace_host_execution_start = time.time_ns()
            result, c, extraces = _execute_trace(
                trc,
                args=args,
                kwargs=kwargs,
                compile_data=cd,
                post_optimization_transforms=post_optimization_transforms,
            )
            cs.last_trace_host_execution_stop = time.time_ns()

            traces.extend(extraces)
            cs.last_traces = traces
            cs.last_executed = c

            # (Possibly) Updates the cache
            if cd.cache_option is CACHE_OPTIONS.CONSTANT_VALUES:
                cache_put(
                    cs.cache,
                    c,
                    traces,
                    args[cd.num_constant_args :],
                    kwargs,
                    autocast_key=None,
                    distributed_key=distributed_key,
                )

            cs.last_trace_host_stop = time.time_ns()
            return result

    if cd.is_module:
        _fn = ThunderOptimizedModule(
            cd.fn,
            _fn,
            cd.processed_function,
            cd.additional_param_names,
            cd.additional_param_values,
            cd.additional_return_names,
        )

    # NOTE is_module is False
    _fn._pfn = cd.processed_function
    _fn._lc_cd = cd
    _fn._lc_cs = cs
    _fn._lc_transforms = transforms
    _fn._lc_post_optimization_transforms = post_optimization_transforms
    _fn._using_grad_transform = _using_grad_transform

    return _fn
