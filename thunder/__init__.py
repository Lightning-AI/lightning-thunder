from functools import wraps, partial
from numbers import Number
from typing import Dict, Set, Optional, Any, List, Callable, Tuple, Type, Hashable
from collections.abc import Sequence
from collections import deque
from enum import auto, Enum
import os
import torch as pytorch
import traceback

from looseversion import LooseVersion

from thunder.core.proxies import is_proxyable, proxy, Proxy, CollectionProxy
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
from thunder.core.langctx import get_langctx, set_langctx, reset_langctx
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
import thunder.core.script.passes
import thunder.core.script as script
import thunder.core.script.python_ir

import thunder.torch as ltorch


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


def _unpack_inputs(fn, tracectx: TraceCtx, args, kwargs):
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


def _make_subkey_for(x: Any) -> Any:
    if isinstance(x, pytorch.Tensor):
        return (pytorch.Tensor, x.shape, x.device, x.dtype)

    return type(x), x


def _make_cache_key(args, kwargs) -> Any:
    def _cache_kwargkey_helper(x: Any, *, key: Hashable) -> Any:
        return type(key), key, _make_subkey_for(x)

    arg_key = tuple(_make_subkey_for(arg) for arg in args)
    kwarg_key = tuple(_cache_kwargkey_helper(v, key=key) for key, v in kwargs.items())
    return arg_key + kwarg_key


def cache_put(cache, fn, traces, args, kwargs) -> None:
    key = _make_cache_key(args, kwargs)
    cache[key] = (fn, traces)


def cache_get(cache, args, kwargs) -> Optional[Callable]:
    key = _make_cache_key(args, kwargs)
    return cache.get(key, (None, None))


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
    only_execute_prims: bool = False,
    disable_preprocessing: bool = False,
    always_trace: Optional[bool] = None,
    use_dynamic_caching: Optional[bool] = None,
    use_static_caching: Optional[bool] = None,
    use_last_executed: Optional[bool] = None,
    use_rematerialization: bool = False,
    use_cudagraphs: bool = False,
    use_generated_backward: bool = False,
) -> Callable:
    pfn: Callable

    # Sets a default tracing mode if one wasn't specified
    if (always_trace, use_dynamic_caching, use_static_caching, use_last_executed) == ((None,) * 4):
        always_trace = True

    # Checks that only one tracing mode is set
    always_trace = always_trace if always_trace is not None else False
    use_dynamic_caching = use_dynamic_caching if use_dynamic_caching is not None else False
    use_static_caching = use_static_caching if use_static_caching is not None else False
    use_last_executed = use_last_executed if use_last_executed is not None else False
    utils.check(
        always_trace ^ use_static_caching ^ use_last_executed ^ use_dynamic_caching,
        lambda: f"Only one caching mode can be specified, but more than one of {always_trace=} (default), {use_static_caching=}, and {use_last_executed=} was set",
    )

    # (Optionally) preprocesses
    if disable_preprocessing:
        pfn = fn
    else:
        pfn = preprocess(fn, is_module=isinstance(fn, pytorch.nn.Module))

    # Constructs function metadata
    # Identifies cache mode
    cache_mode: CACHE_MODES
    if always_trace:
        cache_mode = CACHE_MODES.NONE
    elif use_dynamic_caching:
        cache_mode = CACHE_MODES.DYNAMIC
    elif use_static_caching:
        cache_mode = CACHE_MODES.STATIC
    elif use_last_executed:
        cache_mode = CACHE_MODES.LAST_EXECUTED

    # TODO Implement dynamic caching
    if cache_mode is CACHE_MODES.DYNAMIC:
        raise NotImplementedError

    @wraps(fn)
    def _fn(*args, **kwargs) -> tuple[Any, list[TraceCtx]]:
        # TODO Return the previous traces when caching
        if use_last_executed and _fn._last_executed is not None:
            if _fn._last_executed is not None:
                result = c(*args, **kwargs)
                # TODO Update _last_traces
                _fn._cache_hits += 1
                return result
        if use_static_caching:
            c, traces = cache_get(_fn._cache, args[_fn._num_constant_args :], kwargs)
            if c is not None:
                # TODO Update _last_traces
                _fn._cache_hits += 1
                _fn._last_executed = c
                _fn._last_traces = traces
                return c(*args, **kwargs)

        _fn._cache_misses += 1

        # Sets the language and tracing context
        nonlocal langctx
        langctx_tok = None
        if langctx is not None:
            langctx_tok = set_langctx(langctx)

        started, tok, trace = maybe_start_trace(pfn)

        try:
            proxyargs = args
            proxykwargs = kwargs
            if started:
                proxyargs, proxykwargs = _unpack_inputs(pfn, trace, args, kwargs)
                trace.args = proxyargs
                trace.kwargs = proxykwargs

            result = pfn(*proxyargs, **proxykwargs)

            if started:
                prims.python_return(result)

            # TODO review this with nested traces
            trace.set_output(result)

            traces: list[TraceCtx] = [trace]

            if started:
                # TODO Add the capability to recover from pass failures
                extrace, extraces = executors.transform_for_execution(
                    trace,
                    executors_list=executors_list,
                    only_execute_prims=only_execute_prims,
                    use_rematerialization=use_rematerialization,
                )
                traces.extend(extraces)

                # Constructs callable and updates cache data
                _fn._last_traces = traces
                c = extrace.python_callable()

                if use_cudagraphs and not isinstance(fn, pytorch.nn.Module):
                    c = CUDAGraphExecutor(c)

                _fn._last_executed = c

                if use_static_caching:
                    cache_put(_fn._cache, c, traces, args[_fn._num_constant_args :], kwargs)

                result: Any = c(*args, **kwargs)
        finally:
            # Ensures the language and tracing contexts are reset
            maybe_reset_trace(started, tok)
            if langctx_tok is not None:
                reset_langctx(langctx_tok)

        return result

    if not isinstance(fn, pytorch.nn.Module) and use_generated_backward:
        raise NotImplementedError(
            "Generated backward is only supported for nn.Modules for now. ",
            "Please wrap your function in a nn.Module and try again. ",
            "Alternatively, you can use the @thunder_backward decorator instead of thunder.compile.",
        )

    if isinstance(fn, pytorch.nn.Module):
        if use_cudagraphs:
            _fn = CUDAGraphExecutor(_fn, num_constant_args=len(pfn._additional_param_values))

        if use_generated_backward:
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
            _fn = thunder_backward(**compile_config)(pfn)

        _fn = ThunderOptimizedModule(
            fn, _fn, pfn, pfn._additional_param_names, pfn._additional_param_values, pfn._additional_return_names
        )
        # TODO Revisit assuming these are const
        _fn._num_constant_args = len(pfn._additional_param_values)

    _fn._pfn = pfn
    if not hasattr(_fn, "_num_constant_args"):
        _fn._num_constant_args = 0
    _fn._cache_mode = cache_mode
    _fn._last_executed = None
    _fn._last_traces = None
    _fn._cache_hits = 0
    _fn._cache_misses = 0
    _fn._cache = {}

    return _fn


def last_traces(fn) -> Optional[List[TraceCtx]]:
    return fn._last_traces


def cache_mode(fn) -> CACHE_MODES:
    return fn._cache_mode


def cache_hits(fn) -> int:
    return fn._cache_hits


def cache_misses(fn) -> int:
    return fn._cache_misses


# NOTE A sugar for compile_with_info that just returns the output of the program
def compile_with_info(fn, **compile_kwargs) -> Callable:
    cfn = compile(fn, **compile_kwargs)

    @wraps(cfn)
    def _fn(*args, **kwargs):
        result = cfn(*args, **kwargs)
        return result, last_traces(cfn)

    _fn._fn = cfn
    return _fn


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
    cfn = compile_with_info(fn)

    @wraps(cfn)
    def _fn(*args, **kwargs):
        original_result, original_trace = cfn(*args, **kwargs)

        gradir = _grad_transform(original_trace)

        return original_result, original_trace

    return _fn


#
# old stuff below here
#

#
# tracing functions
#


# def _get_executor(executor=None):
#     if executor is None:
#         ex = get_executor_context()
#         if ex is None:
#             raise ValueError("No executor specified!")
#         return ex

#     if executor == "torch":
#         try:
#             from thunder.executors.torch import torchCtx

#             return torchCtx()
#         except ModuleNotFoundError:
#             raise RuntimeError(
#                 "The 'torch' executor was requested, but the `torch` package "
#                 "is not available. Please make sure the `torch` package is installed"
#                 "in the environment."
#             )

#     if executor == "nvfuser":
#         try:
#             from thunder.executors.nvfuser import nvFuserCtx

#             return nvFuserCtx()
#         except ModuleNotFoundError:
#             raise RuntimeError(
#                 "The 'nvfuser' executor was requested, but NVFuser is not available. "
#                 "Please make sure the `torch` package is installed and CUDA is available."
#             )

#     if hasattr(executor, "get_executor_context"):
#         return executor.get_executor_context()

#     raise ValueError(f"Trying to acquire an executor from unknown object {executor}!")


# # TODO: consider how subclasses could be supported
# # TODO: consider how proxies are extensible (review JAX's proxy extension mechanism)
# # TODO: harvest arg and kwargn names upfront to avoid name collisions with proxies
# def _make_proxies(fn, trace, langctx, *args, **kwargs):
#     """Proxying rules:

#     1. All number and tensor inputs are proxied, including if they're in a container.
#     2. All other inputs are passed unmodified.
#     3. If a proxy is passed in as an input, its name is regenerated to avoid name collisions.
#     """

#     sig = inspect.signature(fn)
#     bound_args = sig.bind_partial(*args)
#     varargs_name = inspect.getfullargspec(fn).varargs

#     def _convert(x):
#         if isinstance(x, Proxy):
#             # Regenerates proxy names to avoid name collisions
#             name = trace.make_proxy_name()
#             p = x.replace_name(name)
#             return p

#         if isinstance(x, (int, float, complex)) or isinstance(x, langctx.tensor_cls):
#             # Proxies numbers and tensors
#             name = trace.make_proxy_name()
#             p = langctx.proxy(x, name=name)
#             return p

#         if isinstance(x, langctx.dtype_cls):
#             # Converts dtypes
#             thunder_dtype = langctx.thunder_dtype(x)
#             return thunder_dtype

#         return x

#     proxyargs = []
#     for name, arg in bound_args.arguments.items():
#         if not isinstance(arg, Proxy) and (isinstance(arg, (int, float)) or isinstance(arg, langctx.tensor_cls)):
#             # NOTE: for numbers or tensors that are passed as positional args,
#             #   this just gives them the name of the positional argument
#             #   Numbers or tensors in a collection (like a list or dict) are
#             #   just given generic names (in the else-block, below)
#             p = langctx.proxy(arg, name=name)
#             proxyargs.append(p)
#         else:
#             values, structure = tree_flatten(arg)
#             converted_values = list(_convert(v) for v in values)

#             packed = tree_unflatten(converted_values, structure)

#             # Handles varargs
#             if name == varargs_name:
#                 proxyargs.extend(packed)
#             else:
#                 proxyargs.append(packed)

#     proxykwargs = {}
#     for name, kwarg in kwargs.items():
#         if isinstance(kwarg, (int, float)) or isinstance(kwarg, langctx.tensor_cls):
#             # NOTE: for numbers or tensors that are passed as keyword arguments,
#             #   this just gives them the name of the argument
#             #   Numbers or tensors in a collection (like a list or dict) are
#             #   just given generic names (in the else-block, below)
#             p = langctx.proxy(kwarg, name=name)
#             proxykwargs[name] = p
#         else:
#             values, structure = tree_flatten(kwarg)
#             converted_values = list(_convert(v) for v in values)
#             packed = tree_unflatten(converted_values, structure)
#             proxykwargs[name] = packed

#     return proxyargs, proxykwargs


# TODO TEMPORARY TEST FUNCTION -- NEEDS UX REVIEW
def construct_trace(fn, trace, proxyargs, proxykwargs):
    trace.args = proxyargs
    trace.kwargs = proxykwargs
    proxyresult = fn(*proxyargs, **proxykwargs)
    trace.set_output(proxyresult)
    return trace


# TODO TEMPORARY TEST FUNCTION -- NEEDS UX REVIEW
def _make_trace(
    fn: Callable,
    *,
    langctx=None,
    disable_preprocessing=True,
) -> Callable:
    """Converts a callable into a callable that will be traced and the trace returned.

    Args:
        fn: The callable to be traced.
        langctx: The language context to use for the trace. If None, the default language context is used.
        disable_preprocessing: If True, preprocessing is disabled. If False, preprocessing is enabled. Defaults to True.

    Example:
        >>> import torch
        >>> import thunder
        >>> import thunder.clang as lang
        >>> def func(a, b):
        ...     return lang.add(a, b)
        >>> tracing_func = thunder.make_trace(func)
        >>> a = torch.randn(2, 2, device='cuda')
        >>> b = torch.randn(2, 1, device='cuda')
        >>> trace = tracing_func(a, b)
    """

    if disable_preprocessing:
        pfn = fn
    else:
        pfn = preprocess(fn, is_module=isinstance(fn, pytorch.nn.Module))

    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            # Sets the proper tracing context
            if langctx is not None:
                langctx_tok = set_langctx(langctx)
            trace = TraceCtx(fn)
            trace_token = set_tracectx(trace)
            proxyargs, proxykwargs = _unpack_inputs(pfn, trace, args, kwargs)
            trace = construct_trace(pfn, trace, proxyargs, proxykwargs)
        finally:
            # Resets the tracing context
            reset_tracectx(trace_token)
            if langctx is not None:
                reset_langctx(langctx_tok)
        return trace

    if isinstance(fn, pytorch.nn.Module):
        wrapped = ThunderOptimizedModule(fn, wrapped, pfn, pfn._additional_param_names, pfn._additional_param_values)

    return wrapped


# import torch  # oops


# class ThunderOptimizedModule(torch.nn.Module):  # TOM
#     # todo: subclass nn.Module or forward things like .state_dict() to the
#     #       model
#     def __init__(self, model, fn, tfn, additional_param_names, additional_param_values):
#         super().__init__()
#         self._model = model
#         self._forward_fn = fn
#         self._tfn = tfn
#         self._additional_param_values = additional_param_values
#         self._additional_param_names = additional_param_names

#     def __call__(self, *args, **kwargs):
#         all_args = (*self._additional_param_values, *args)
#         res = self._forward_fn(*all_args, **kwargs)
#         return res


# def make_traced(
#     fn: Callable,
#     executor: Optional[str] = None,
#     language_ctx=langs.torch,
#     *,
#     mode=None,
#     _info=False,
#     _return_fusion=False,
#     _preprocess=False,
#     _profile_info=False,
#     _static=False,
# ) -> Callable:
#     """Converts a callable in a callable that will be traced and then executed.

#     Example usage:

#       def foo(a, b):
#         return tlang.add(a, b)

#       traced_foo = thunder.make_traced(foo)

#       a = torch.randn(2, 2, device='cuda')
#       b = torch.randn(2, 1, device='cuda')

#       result = traced_foo(a, b)
#     """
#     from thunder.core.transforms import inline

#     ex = _get_executor(executor)
#     langctx = language_ctx.ctx()

#     if _preprocess:
#         tfn = preprocess(fn, is_module=isinstance(fn, langctx.module_cls))
#     else:
#         tfn = fn

#     @wraps(fn)
#     def _fn(*args, **kwargs):
#         acquisition_start = time.time_ns()
#         acqusition_end = None
#         result = None
#         if _fn._thunder_cache is not None:
#             cached = _fn._thunder_cache(*args, **kwargs)
#             if cached is not None:
#                 result = cached(*args, **kwargs)
#                 acquisition_end = time.time_ns()

#         translation_start = translation_end = 0
#         invocation_start = invocation_end = 0
#         fusion = None
#         profile_info = None
#         if result is None:
#             trace = make_trace(inline(tfn), executor, language_ctx)(*args, **kwargs)
#             acquisition_end = time.time_ns()

#             translation_start = time.time_ns()
#             # TODO: probably refactor this to not be such an ugly variadic return
#             #   (maybe by returning a dict)
#             profile_info = None
#             fusion = None
#             if _profile_info:
#                 fusion, profile_info = ex.fuse(
#                     trace,
#                     profile_info=_profile_info,
#                     mode=mode,
#                     args=args,
#                     kwargs=kwargs,
#                     static_inputs=tfn._additional_param_values if hasattr(tfn, "_additional_param_values") else None,
#                 )
#             else:
#                 fusion = ex.fuse(
#                     trace,
#                     mode=mode,
#                     args=args,
#                     kwargs=kwargs,
#                     static_inputs=tfn._additional_param_names if hasattr(tfn, "_additional_param_names") else None,
#                 )
#             translation_end = time.time_ns()

#             invocation_start = time.time_ns()
#             result = fusion(*args, **kwargs)
#             invocation_end = time.time_ns()

#             if _static:
#                 _fn._thunder_cache = cache.build_cache(tfn, args, kwargs, fusion)

#         meta = None
#         if _info:
#             meta = {
#                 "acquisition_time": acquisition_end - acquisition_start,
#                 "invocation_time": invocation_end - invocation_start,
#                 "translation_time": translation_end - translation_start,
#             }

#         if _info or _return_fusion or _profile_info:
#             return {
#                 "result": result,
#                 "meta": meta,
#                 "fusion": fusion,
#                 "profile_info": profile_info,
#             }

#         return result

#     if isinstance(fn, langctx.module_cls):
#         _fn = ThunderOptimizedModule(fn, _fn, tfn, tfn._additional_param_names, tfn._additional_param_values)

#     _fn._tfn = tfn
#     setattr(_fn, "_thunder_cache", None)

#     return _fn
