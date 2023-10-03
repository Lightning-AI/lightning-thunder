from functools import wraps, partial
from numbers import Number
from typing import Dict, Set, Optional, Any, List, Callable, Tuple, Type
from collections.abc import Hashable
from collections.abc import Sequence
from collections import deque
from enum import auto, Enum
import os
import traceback
from dataclasses import dataclass
import argparse
import time

from looseversion import LooseVersion

import torch as pytorch
import numpy as np

from thunder.core.proxies import is_proxyable, proxy, Proxy, CollectionProxy, TensorProxy
from thunder.core.trace import (
    TraceCtx,
    from_trace,
    get_tracectx,
    set_tracectx,
    reset_tracectx,
    wrap_in_trace_variable,
    maybe_start_trace,
    maybe_reset_trace,
)
from thunder.core.langctx import get_langctx, set_langctx, reset_langctx, get_default_langctx
import thunder.core.utils as utils
from thunder.core.codeutils import get_siginfo, is_collection
import thunder.core.prims as prims
import thunder.core.dtypes as dtypes
import thunder.executors as executors
import thunder.core.symbol as symbol
import thunder.core.devices as devices
from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
from thunder.cudagraphs import CUDAGraphExecutor
from thunder.executors.torchex import thunder_backward
import thunder.executors as executors

import thunder.core.script as script
import thunder.core.script.frontend
import thunder.core.script.instrumentation
import thunder.core.script.passes
import thunder.core.script as script
import thunder.core.script.python_ir

import thunder.core.transforms as transforms
from thunder.core.transforms import pytorch_grad_transform

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


# Common UX functions


def _unpack_inputs(fn, tracectx: TraceCtx, args, kwargs, *, rename_proxies: bool):
    tracectx.unpacking()

    # Translates tensors, arrays, and dtypes to lightning.compile types
    # TODO Translate NumPy dtypes
    def translate(x: Any, *, name: Optional[str] = None) -> Any:
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
        if isinstance(x, pytorch.dtype):
            return ltorch.to_thunder_dtype(x)
        if utils.is_collection(x):
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


# Preprocesses function
# Currently tries to map torch.foo lookups to thunder.torch.foo lookups
@thunder.core.script.instrumentation.record
def preprocess(fn, is_module):
    gr = script.frontend.acquire_method(fn.forward if is_module else fn)
    thunder.core.script.passes.unroll_for_loops_and_inline_modules(gr)
    if is_module:
        (
            additional_param_names,
            additional_param_values,
            additional_return_names,
        ) = thunder.core.script.passes.module_to_function(gr)
    script.passes.strongly_inline_functions(gr)
    script.passes.torch_to_thunder(gr)

    thunder_fn = thunder.core.script.python_ir.generate_function(gr)
    if is_module:
        thunder_fn._additional_param_names = additional_param_names
        thunder_fn._additional_param_values = additional_param_values
        thunder_fn._additional_return_names = additional_return_names
    else:
        thunder_fn._additional_param_names = None
        thunder_fn._additional_param_values = None
        thunder_fn._additional_return_names = None
    return thunder_fn


class ThunderOptimizedModule(pytorch.nn.Module):  # TOM
    # todo: subclass nn.Module or forward things like .state_dict() to the
    #       model
    def __init__(self, model, fn, tfn, additional_param_names, additional_param_values, additional_return_names):
        super().__init__()
        self._model = model
        self._forward_fn = fn
        self._tfn = tfn
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


#
# Caching objects and functions
#
# TODO We could look at supporting non-hashable inputs, like dicts


class CACHE_MODES(Enum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()
    LAST_EXECUTED = auto()


# TODO Update cacheable types
def _make_subkey_for(x: Any) -> tuple[bool, None | tuple]:
    if isinstance(x, pytorch.Tensor):
        return True, (pytorch.Tensor, x.shape, x.device, x.dtype, x.requires_grad)

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
    if utils.is_collection(x):
        return False, None

    if isinstance(x, Hashable):
        return True, (type(x), x)

    return False, None


# Private class just to separate objects in the cache
class _key_value_separator:
    pass


# Returns a hashable key or None if the given args and kwargs are not hashable
def _make_cache_key(args, kwargs) -> None | tuple:
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

    return tuple(key)


# Returns True if successfully cached, false otherwise
def cache_put(cache: dict, fn, traces: Sequence[TraceCtx], args, kwargs) -> bool:
    key = _make_cache_key(args, kwargs)

    if key is None:
        return False

    cache[key] = (fn, traces)
    return True


def cache_get(cache: dict, args, kwargs) -> tuple[None | Callable, None | Sequence[TraceCtx]]:
    key = _make_cache_key(args, kwargs)
    return cache.get(key, (None, None))


# A class that holds data about the compiled object, including statistics about how it's been called
# TODO Better document the module-related data the preprocessing harvests,
#   like additional_param_names
class CompileData:
    def __init__(
        self,
        *,
        fn: Callable,
        langctx: Optional[Any] = None,
        executors_list: Optional[list[executors.Executor]] = None,
        only_execute_prims: bool = False,
        disable_preprocessing: bool = False,
        always_trace: Optional[bool] = None,
        use_dynamic_caching: Optional[bool] = None,
        use_static_caching: Optional[bool] = None,
        use_last_executed: Optional[bool] = None,
        use_cudagraphs: bool = False,
        disable_torch_autograd_support: bool = False,
        use_rematerialization: bool = False,
    ):
        #
        # Determines the cache mode
        #

        # Sets a default tracing mode if one wasn't specified
        if (always_trace, use_dynamic_caching, use_static_caching, use_last_executed) == ((None,) * 4):
            use_static_caching = True

        # Checks that only one tracing mode is set
        always_trace = always_trace if always_trace is not None else False
        use_dynamic_caching = use_dynamic_caching if use_dynamic_caching is not None else False
        use_static_caching = use_static_caching if use_static_caching is not None else False
        use_last_executed = use_last_executed if use_last_executed is not None else False
        utils.check(
            always_trace ^ use_static_caching ^ use_last_executed ^ use_dynamic_caching,
            lambda: f"Exactly one caching mode must be specified, but more than one of {always_trace=} (default), {use_static_caching=}, and {use_last_executed=} was set",
        )

        # Identifies cache mode
        self.cache_mode: CACHE_MODES
        if always_trace:
            self.cache_mode = CACHE_MODES.NONE
        elif use_dynamic_caching:
            self.cache_mode = CACHE_MODES.DYNAMIC
        elif use_static_caching:
            self.cache_mode = CACHE_MODES.STATIC
        elif use_last_executed:
            self.cache_mode = CACHE_MODES.LAST_EXECUTED

        # TODO Implement dynamic caching
        if self.cache_mode is CACHE_MODES.DYNAMIC:
            raise NotImplementedError

        #
        # Gathers additional metadata
        #

        self.fn = fn
        self.langctx = langctx
        self.executors_list = executors_list
        self.only_execute_prims = only_execute_prims
        self.disable_preprocessing = disable_preprocessing
        self.use_rematerialization = use_rematerialization
        self.use_cudagraphs = use_cudagraphs
        self.disable_torch_autograd_support = disable_torch_autograd_support

        self.is_module = isinstance(self.fn, pytorch.nn.Module)

        #
        # Possibly processes the function
        #
        self.additional_param_names = None
        self.additional_param_values = None
        self.additional_return_names = None
        self.num_constant_args = 0

        self.post_processed_function: Callable
        if disable_preprocessing:
            self.post_processed_function = fn
        else:
            self.post_processed_function = preprocess(fn, is_module=self.is_module)

            # TODO Revisit assuming parameters are const
            if self.is_module:
                self.additional_param_names = self.post_processed_function._additional_param_names
                self.additional_param_values = self.post_processed_function._additional_param_values
                self.additional_return_names = self.post_processed_function._additional_return_names
                self.num_constant_args = len(self.additional_param_values)

        #
        # Initializes execution statistics
        #
        self.last_executed = None
        self.last_traces = None
        self.last_trace_host_start: int = -1
        self.last_trace_host_stop: int = -1
        self.last_trace_cache_start: int = -1
        self.last_trace_cache_stop: int = -1
        self.last_trace_tracing_start: int = -1
        self.last_trace_tracing_stop: int = -1
        self.last_trace_host_execution_start: int = -1
        self.last_trace_host_execution_stop: int = -1

        # torch.autograd.Function specific data
        self.primal_trace = None
        self.forward_last_traces = None
        self.backward_last_traces = None

        self.cache = {}
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0


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
# TODO Change the way this is called to be trace(langctx, inline_trace, rename_proxies...)(fn, *args, **kwargs)
#   to separate the traced function's args and kwargs from this function's kwargs


def trace(
    langctx: None | Any = None,
    inline_trace: bool = True,
    rename_proxies: bool = True,
    include_return_statement: bool = True,
    use_dce: bool = True,
) -> Callable:
    def _trace(
        fn,
        *args,
        **kwargs,
    ) -> Any | TraceCtx:
        langctx_ = langctx if langctx is not None else get_default_langctx()

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

            result = fn(*proxyargs, **proxykwargs)

            if include_return_statement:
                prims.python_return(result)

            trace.set_output(result)
            if use_dce:
                trace, _ = transforms.dce(trace)

        finally:
            # Resets contexts
            reset_langctx(langctx_tok)

            if tracectx_tok is not None:
                reset_tracectx(tracectx_tok)

        return trace

    return _trace


def _wrap_in_tom(
    *,
    original_module,
    possibly_processed_function,
    compiled_function,
    compile_data: CompileData,
) -> ThunderOptimizedModule:
    tom = ThunderOptimizedModule(
        original_module,
        compiled_function,
        possibly_processed_function,
        compile_data.additional_param_names,
        compile_data.additional_param_values,
        compile_data.additional_return_names,
    )

    tom._pfn = possibly_processed_function
    tom._lc_cd = compile_data
    return tom


# Executes the trace with the given args and kwargs and returns the result,
#   the callable executed, and the series of traces constructed to produce
#   that callable from the trace
def execute_trace(
    trc: TraceCtx,
    *,
    args,
    kwargs,
    compile_data: CompileData,
) -> tuple[Any, Callable, list[TraceCtx]]:
    # Transforms the trace for execution
    # TODO Add the capability to recover from pass failures
    extrace, extraces = executors.transform_for_execution(
        trc,
        executors_list=compile_data.executors_list,
        only_execute_prims=compile_data.only_execute_prims,
        use_rematerialization=compile_data.use_rematerialization,
    )

    # Constructs the Python callable
    c = extrace.python_callable()

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
def compile(
    fn: Callable,
    *,
    langctx: Optional[Any] = None,
    executors_list: Optional[list[executors.Executor]] = None,
    always_trace: Optional[bool] = None,
    use_dynamic_caching: Optional[bool] = None,
    use_static_caching: Optional[bool] = None,
    use_last_executed: Optional[bool] = None,
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
        always_trace=always_trace,
        use_dynamic_caching=use_dynamic_caching,
        use_static_caching=use_static_caching,
        use_last_executed=use_last_executed,
        use_cudagraphs=use_cudagraphs,
        disable_torch_autograd_support=disable_torch_autograd_support,
        use_rematerialization=use_rematerialization,
        only_execute_prims=only_execute_prims,
        disable_preprocessing=disable_preprocessing,
    )

    @wraps(fn)
    def _fn(*args, **kwargs) -> tuple[Any, list[TraceCtx]]:
        cd.last_trace_host_start = time.time_ns()
        cd.calls += 1

        # Tries to lookup a callable in a cache
        # TODO Return the previous traces when caching
        cd.last_trace_cache_start = time.time_ns()
        if cd.cache_mode is CACHE_MODES.LAST_EXECUTED and cd.last_executed is not None:
            if cd.last_executed is not None:
                # TODO Update _last_traces?
                # Updates statistics before early termination
                cd.cache_hits += 1
                cd.last_trace_cache_stop = time.time_ns()
                cd.last_trace_tracing_start = -1
                cd.last_trace_tracing_stop = -1
                cd.last_trace_host_execution_start = time.time_ns()
                result = cd.last_executed(*args, **kwargs)
                cd.last_trace_host_execution_stop = time.time_ns()
                cd.last_trace_host_stop = cd.last_trace_host_execution_stop
                return result
        if cd.cache_mode is CACHE_MODES.STATIC:
            c, _ = cache_get(cd.cache, args[cd.num_constant_args :], kwargs)
            if c is not None:
                # Updates statistics before early termination
                cd.cache_hits += 1
                cd.last_executed = c
                cd.last_trace_cache_stop = time.time_ns()
                cd.last_trace_tracing_start = -1
                cd.last_trace_tracing_stop = -1
                cd.last_trace_host_execution_start = time.time_ns()
                result = c(*args, **kwargs)
                cd.last_trace_host_execution_stop = time.time_ns()
                cd.last_trace_host_stop = cd.last_trace_host_execution_stop
                return result
        cd.cache_misses += 1
        cd.last_trace_cache_stop = time.time_ns()

        # Determines whether to use autograd.Function or not
        flat_args, _ = tree_flatten((args, kwargs))
        tensor_cls = (pytorch.Tensor, TensorProxy)
        requires_grad = any(isinstance(arg, tensor_cls) and arg.requires_grad for arg in flat_args)
        if not disable_torch_autograd_support and requires_grad:
            compile_config = {
                "langctx": langctx,
                "executors_list": executors_list,
                "only_execute_prims": only_execute_prims,
                "always_trace": always_trace,
                "use_dynamic_caching": use_dynamic_caching,
                "use_static_caching": use_static_caching,
                "use_last_executed": use_last_executed,
                "use_rematerialization": use_rematerialization,
                "use_cudagraphs": use_cudagraphs,
            }

            # thunder_backward may recursively call compile and wraps the result in a
            # torch.autograd.Function to support embedding of Thunder-compiled
            # functions in PyTorch's Autograd
            cd.last_trace_host_execution_start = time.time_ns()
            c = thunder_backward(compile_data=cd, **compile_config)(cd.post_processed_function)
            result = c(*args, **kwargs)
            cd.last_trace_host_execution_stop = time.time_ns()
            cd.last_executed = c
            if cd.cache_mode is CACHE_MODES.STATIC:
                cache_put(cd.cache, c, None, args[cd.num_constant_args :], kwargs)
            cd.last_trace_host_stop = time.time_ns()
            return result

        # TODO Revisit compile() behavior when hit in a trace ctx
        #   This will inline the invocation of compile into the current
        #   trace (UNLESS there was a cache hit, per above)
        #   This interaction between the cache and tracing seems odd
        # TODO Support a practitioner who wants to explicitly and separately compile
        #   part of the program

        # Acquires the trace OR inlines the trace into an existing trace and
        #   returns the (proxied) result of the operation
        cd.last_trace_tracing_start = time.time_ns()
        trc_or_result = trace(langctx=langctx)(cd.post_processed_function, *args, **kwargs)
        cd.last_trace_tracing_stop = time.time_ns()

        # Returns the (proxied) result if this call to compile was inlined
        current_trace = get_tracectx()
        if current_trace is not None:
            result = trc_or_result
            return result

        #
        # Executes the trace, then updates the CompiledData and possibly
        #   updates a cache
        #

        trc: TraceCtx = trc_or_result
        traces: list[TraceCtx] = [trc]

        cd.last_trace_host_execution_start = time.time_ns()
        result, c, extraces = execute_trace(
            trc,
            args=args,
            kwargs=kwargs,
            compile_data=cd,
        )
        cd.last_trace_host_execution_stop = time.time_ns()

        traces.extend(extraces)
        cd.last_traces = traces
        cd.last_executed = c

        # (Possibly) Updates the cache
        if cd.cache_mode is CACHE_MODES.STATIC:
            cache_put(cd.cache, c, traces, args[cd.num_constant_args :], kwargs)

        cd.last_trace_host_stop = time.time_ns()
        return result

    if cd.is_module:
        return _wrap_in_tom(
            original_module=fn,
            possibly_processed_function=cd.post_processed_function,
            compiled_function=_fn,
            compile_data=cd,
        )

    # NOTE not is_module
    _fn._pfn = cd.post_processed_function
    _fn._lc_cd = cd
    return _fn


# WIP Autograd function implementation to support compile_torch experimentation
# TODO Tensors returned within collections -- like with return ((c, d),) -- will not requires_grad
#   when using autograd.Function -- probably need to return those tensors directly and then just not pass them
#   on from the wrapper
# TODO Look at constructing this class using a metaclass that can:
#   - set the name of this class to something like "{fn.__name__}" so the backward
#       of everything isn't "LCFunctionBackward"
class LCFunction(pytorch.autograd.Function):
    # TODO Test swapping order of tensors, tensors that require grad in keywords
    @staticmethod
    def forward(
        ctx: pytorch.autograd.function.FunctionCtx,
        _: pytorch.Tensor,
        fn,
        compiledata: CompileData,
        args_and_kwargs: tuple[Any, Any],
    ) -> Any:
        args, kwargs = args_and_kwargs

        # TODO Add caching (inputs -> forward callable, backward compiled object)
        # TODO Add module support

        # TODO Transform for grad
        trc = trace()(fn, *args, **kwargs)

        grad_forward = pytorch_grad_transform(trc)

        # TODO Check if currently tracing?

        # TODO Support cudagraphs
        # TODO Support rematerialization
        # TODO Actually execute the grad_forward, not the original trc
        result, c, extraces = execute_trace(
            trc,
            args=args,
            kwargs=kwargs,
            executors_list=compiledata.executors_list,
            only_execute_prims=compiledata.only_execute_prims,
            use_cudagraphs=False,
            use_rematerialization=False,
        )

        traces: list[TraceCtx] = [trc, grad_forward]
        traces.extend(extraces)

        compiledata.last_traces = traces
        compiledata.last_executed = c

        # TODO Update cache (maps input -> (forward callable, compiled backward function))
        # TODO Update ctx with backward function

        # TODO Save necessary tensors for backward
        # ctx.save_for_backward(())

        # Returns tensors requiring grad separately from the other results
        # NOTE GRAD OUTS
        #   This is done because returning tensors in collections -- like ((c, d),) -- will not properly
        #   set the tensors c and d as requiring grad. For this case, this logic will alter the function's
        #   return to be ((c, d),), c, d, and then the wrapper that calls the function will ignore the
        #   c and d results.
        #   Note further that we have to inspect the proxy's requires_grad property to know if grad
        #   should be set on the tensor.
        result_and_grad_outs = [result]
        flat_results, _ = tree_flatten(result)
        flat_proxy_results = trc.bound_symbols[-1]._flat_outs
        for x, p in zip(flat_results, flat_proxy_results):
            if isinstance(x, pytorch.Tensor):
                if p.requires_grad:
                    result_and_grad_outs.append(x)
                else:
                    ctx.mark_non_differentiable(x)

        return tuple(result_and_grad_outs)

    # TODO Test double backwards
    # NOTE When a function produces multiple outputs requiring grad, the gradient of inputs w.r.t. every output
    #   does not always need to be computed. For example, let's say that a function produces tensors a, b, and c,
    #   and a practitioner calls a.backward(), or .backward() on a later tensor that is derived exclusively from
    #   a. In this case the gradient of the inputs is only computed w.r.t. the computation for a. PyTorch models
    #   this by passing tensors of zeros for each output not involved in the computation.
    #   In this example, the input would be grad_a, zeros_like(b), zeros_like(c).
    #   In these cases, then, excessive computation on the zero tensors may be performed.
    # TODO Consider checking for these cases, or requesting PyTorch provide a quick way to check if the tensors
    #   are just zero tensors (like by passing None instead of zero tensors). Maybe PyTorch does have SOME
    #   way of detecting this, and I (mruberry) am just ignorant of it.
    #   One natural thing would be for these tensors to be "zerotensors" (a PyTorch implementation detail) but
    #   they are not.
    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError


# WIP
# Returns a callable compatible with PyTorch's autograd
def compile_torch(
    fn: Callable,
    *,
    langctx: Optional[Any] = None,
    executors_list: Optional[list[executors.Executor]] = None,
    always_trace: Optional[bool] = None,
    use_dynamic_caching: Optional[bool] = None,
    use_static_caching: Optional[bool] = None,
    use_last_executed: Optional[bool] = None,
    use_cudagraphs: bool = False,
    only_execute_prims: bool = False,
    disable_preprocessing: bool = False,
) -> Callable:
    cd = CompileData(
        fn=fn,
        langctx=langctx,
        executors_list=executors_list,
        always_trace=always_trace,
        use_dynamic_caching=use_dynamic_caching,
        use_static_caching=use_static_caching,
        use_last_executed=use_last_executed,
        use_cudagraphs=use_cudagraphs,
        use_rematerialization=False,
        disable_torch_autograd_support=False,
        only_execute_prims=only_execute_prims,
        disable_preprocessing=disable_preprocessing,
    )

    # TODO Add timings
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # TODO Add caching

        # TODO Review recursive trace support
        current_trace = get_tracectx()
        if current_trace is not None:
            raise NotImplementedError(
                "Recursively tracing is not yet supported. Please file an issue requesting this feature."
            )

        # trc = trace(cd.post_processed_function, *args, langctx=langctx, **kwargs)

    return wrapper
    # @wraps(fn)
    # def wrapper(*args, **kwargs):
    #     # NOTE pytorch.autograd.Function.apply() does not accept keyword arguments
    #     # NOTE This call to apply() packs the args and kwargs in a tuple.
    #     #   This would typically prevent autograd.Function.apply() from marking inexact tensor
    #     #   outputs as requiring grad, so this passes a "dummy" tensor that requires_grad to let
    #     #   apply() know to mark inexact tensor outputs as requiring grad (and having the correct)
    #     #   autograd history.
    #     # NOTE Another option would be to flatten and unflatten the args and kwargs, but that can be
    #     #   slow.
    #     # NOTE See the GRAD OUTS note above
    #     result = LCFunction.apply(_dummy, fn, cd, (args, kwargs))
    #     return result[0]

    # wrapper._lc_cd = cd

    # return wrapper


def compile_data(fn) -> Optional[CompileData]:
    return getattr(fn, "_lc_cd", None)


# TODO We should remove compiledata.last_traces in favor of forward_last_traces and backward_last_traces
def last_traces(fn) -> None | list[TraceCtx] | tuple[list[TraceCtx], list[TraceCtx]]:
    if compile_data(fn).forward_last_traces is not None and compile_data(fn).backward_last_traces is not None:
        return compile_data(fn).forward_last_traces, compile_data(fn).backward_last_traces
    return compile_data(fn).last_traces


def cache_mode(fn) -> CACHE_MODES:
    return compile_data(fn).cache_mode


def cache_hits(fn) -> int:
    return compile_data(fn).cache_hits


def cache_misses(fn) -> int:
    return compile_data(fn).cache_misses


# TODO There is probably a better way to do this
symbol.set_eagerctx(
    (partial(compile, executors_list=[executors.TORCH], only_execute_prims=True, disable_preprocessing=True), ltorch)
)


# TODO Actually implement this
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
