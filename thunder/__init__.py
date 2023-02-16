import inspect
import os
import time
from functools import wraps
from typing import Callable, Optional

import thunder.core.cache as cache
import thunder.core.dtypes as dtypes
import thunder.core.proxies as proxies
import thunder.core.script as script
import thunder.core.script.frontend
import thunder.core.script.passes
import thunder.core.script.python_ir
import thunder.langs as langs
from thunder.__about__ import *
from thunder.core.proxies import Proxy
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.trace import (
    get_executor_context,
    get_trace,
    new_trace,
    reset_executor_context,
    reset_language_context,
    reset_trace,
    set_executor_context,
    set_language_context,
)

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
    # tracing functions
    "make_trace",
    "make_traced",
]

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
# tracing functions
#


def _get_executor(executor=None):
    if executor is None:
        ex = get_executor_context()
        if ex is None:
            raise ValueError("No executor specified!")
        return ex

    if executor == "torch":
        try:
            from thunder.executors.torch import torchCtx

            return torchCtx()
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'torch' executor was requested, but the `torch` package "
                "is not available. Please make sure the `torch` package is installed"
                "in the environment."
            )

    if executor == "nvfuser":
        try:
            from thunder.executors.nvfuser import nvFuserCtx

            return nvFuserCtx()
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'nvfuser' executor was requested, but NVFuser is not available. "
                "Please make sure the `torch` package is installed and CUDA is available."
            )

    if hasattr(executor, "get_executor_context"):
        return executor.get_executor_context()

    raise ValueError(f"Trying to acquire an executor from unknown object {executor}!")


# TODO: consider how subclasses could be supported
# TODO: consider how proxies are extensible (review JAX's proxy extension mechanism)
# TODO: harvest arg and kwargn names upfront to avoid name collisions with proxies
def _make_proxies(fn, trace, langctx, *args, **kwargs):
    """Proxying rules:

    1. All number and tensor inputs are proxied, including if they're in a container.
    2. All other inputs are passed unmodified.
    3. If a proxy is passed in as an input, its name is regenerated to avoid name collisions.
    """

    sig = inspect.signature(fn)
    bound_args = sig.bind_partial(*args)
    varargs_name = inspect.getfullargspec(fn).varargs

    def _convert(x):
        if isinstance(x, (int, float, complex)) or isinstance(x, langctx.tensor_cls):
            # Proxies numbers and tensors
            name = trace.make_proxy_name()
            p = langctx.proxy(x, name=name)
            return p

        if isinstance(x, langctx.dtype_cls):
            # Converts dtypes
            thunder_dtype = langctx.thunder_dtype(x)
            return thunder_dtype

        if isinstance(x, Proxy):
            # Regenerates proxy names to avoid name collisions
            name = trace.make_proxy_name()
            p = x.replace_name(name)
            return p

        return x

    proxyargs = []
    for name, arg in bound_args.arguments.items():
        if isinstance(arg, (int, float)) or isinstance(arg, langctx.tensor_cls):
            # NOTE: for numbers or tensors that are passed as positional args,
            #   this just gives them the name of the positional argument
            #   Numbers or tensors in a collection (like a list or dict) are
            #   just given generic names (in the else-block, below)
            p = langctx.proxy(arg, name=name)
            proxyargs.append(p)
        else:
            values, structure = tree_flatten(arg)
            converted_values = list(_convert(v) for v in values)

            packed = tree_unflatten(converted_values, structure)

            # Handles varargs
            if name == varargs_name:
                proxyargs.extend(packed)
            else:
                proxyargs.append(packed)

    proxykwargs = {}
    for name, kwarg in kwargs.items():
        if isinstance(kwarg, (int, float)) or isinstance(kwarg, langctx.tensor_cls):
            # NOTE: for numbers or tensors that are passed as keyword arguments,
            #   this just gives them the name of the argument
            #   Numbers or tensors in a collection (like a list or dict) are
            #   just given generic names (in the else-block, below)
            p = langctx.proxy(kwarg, name=name)
            proxykwargs[name] = p
        else:
            values, structure = tree_flatten(kwarg)
            converted_values = list(_convert(v) for v in values)
            packed = tree_unflatten(converted_values, structure)
            proxykwargs[name] = packed

    return proxyargs, proxykwargs


def _construct_trace(fn, trace, proxyargs, proxykwargs):
    trace.add_args(proxyargs)
    trace.add_kwargs(proxykwargs)
    proxyresult = fn(*proxyargs, **proxykwargs)
    trace.add_outputs(proxyresult)
    return trace


def make_trace(fn: Callable, executor: Optional[str] = None, language_ctx=langs.torch):
    """Converts a callable into a callable that will be traced and the trace returned.

    Args:
        fn: The callable to be traced.
        executor: The executor to use for the trace. If None, the default executor is used.
        language_ctx: The language context to use for the trace. If None, the default language context is used.

    Example:
        >>> import thunder
        >>> from thunder.core import lang
        >>> def foo(a, b):
        ...     return lang.add(a, b)
        >>> tracing_foo = thunder.make_trace(foo, executor="torch")
        >>> a = torch.randn(2, 2, device='cuda')
        >>> b = torch.randn(2, 1, device='cuda')
        >>> trace = tracing_foo(a, b)
    """
    ex = _get_executor(executor)
    langctx = language_ctx.ctx()

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            # Sets the proper tracing context
            trace_token = new_trace()
            executor_token = set_executor_context(ex)
            lang_token = set_language_context(langctx)
            trace = get_trace()
            proxyargs, proxykwargs = _make_proxies(fn, trace, langctx, *args, **kwargs)
            trace = _construct_trace(fn, trace, proxyargs, proxykwargs)
        finally:
            # Resets the tracing context
            reset_trace(trace_token)
            reset_language_context(lang_token)
            if executor_token is not None:
                reset_executor_context(executor_token)
        return trace

    return wrapper


# Preprocesses function
# Currently tries to map torch.foo lookups to thunder.torch.foo lookups
def preprocess(fn, is_module):
    gr = script.frontend.acquire_method(fn.forward if is_module else fn, verbose=False)

    script.frontend.make_ssa(gr)
    script.frontend.make_single_return(gr)
    thunder.core.script.passes.unroll_for_loops_and_inline_modules(gr)
    if is_module:
        additional_param_names = thunder.core.script.passes.module_to_function(gr)
    script.passes.strongly_inline_functions(gr)
    script.passes.torch_to_thunder(gr)

    thunder_fn = script.python_ir.generate_function(gr)
    if is_module:
        thunder_fn._additional_param_names = additional_param_names
    return thunder_fn


import torch  # oops


class ThunderOptimizedModule(torch.nn.Module):  # TOM
    # todo: subclass nn.Module or forward things like .state_dict() to the
    #       model
    def __init__(self, model, fn, tfn, additional_param_names):
        super().__init__()
        self._model = model
        self._forward_fn = fn
        self._tfn = tfn
        self._additional_param_names = additional_param_names
        self._fn_param_names = [
            n
            for n, p in inspect.signature(self._tfn).parameters.items()
            if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
        ][-len(additional_param_names) :]

    def __call__(self, *args, **kwargs):
        sd = self._model.state_dict()
        params = {
            pn: sd[n.replace("[", "").replace("]", "")]
            for pn, n in zip(self._fn_param_names, self._additional_param_names)
        }
        kwargs_params = {**params, **kwargs}
        if len(kwargs_params) != len(params) + len(kwargs):
            # We could allow this, but once we do, people will rely on it.
            # As long as we don't we can still discuss.
            raise RuntimeError("passing parameter values is not supported")
        res = self._forward_fn(*args, **kwargs_params)
        return res


def make_traced(
    fn: Callable,
    executor: Optional[str] = None,
    language_ctx=langs.torch,
    *,
    _info=False,
    _return_fusion=False,
    _preprocess=False,
    _profile_info=False,
    _static=False,
) -> Callable:
    """Converts a callable in a callable that will be traced and then executed.

    Example usage:

      def foo(a, b):
        return tlang.add(a, b)

      traced_foo = thunder.make_traced(foo)

      a = torch.randn(2, 2, device='cuda')
      b = torch.randn(2, 1, device='cuda')

      result = traced_foo(a, b)
    """

    ex = _get_executor(executor)
    langctx = language_ctx.ctx()

    if _preprocess:
        tfn = preprocess(fn, is_module=isinstance(fn, langctx.module_cls))
    else:
        tfn = fn

    @wraps(fn)
    def _fn(*args, **kwargs):
        acquisition_start = time.time_ns()
        acqusition_end = None
        result = None
        if _fn._thunder_cache is not None:
            cached = _fn._thunder_cache(*args, **kwargs)
            if cached is not None:
                result = cached(*args, **kwargs)
                acquisition_end = time.time_ns()

        translation_start = translation_end = 0
        invocation_start = invocation_end = 0
        fusion = None
        profile_info = None
        if result is None:
            trace = make_trace(tfn, executor, language_ctx)(*args, **kwargs)
            acquisition_end = time.time_ns()

            translation_start = time.time_ns()
            # TODO: probably refactor this to not be such an ugly variadic return
            #   (maybe by returning a dict)
            profile_info = None
            fusion = None
            if _profile_info:
                fusion, profile_info = ex.fuse(trace, profile_info=_profile_info)
            else:
                fusion = ex.fuse(trace)
            translation_end = time.time_ns()

            invocation_start = time.time_ns()
            result = fusion(*args, **kwargs)
            invocation_end = time.time_ns()

            if _static:
                _fn._thunder_cache = cache.build_cache(tfn, args, kwargs, fusion)

        meta = None
        if _info:
            meta = {
                "acquisition_time": acquisition_end - acquisition_start,
                "invocation_time": invocation_end - invocation_start,
                "translation_time": translation_end - translation_start,
            }

        if _info or _return_fusion or _profile_info:
            return {
                "result": result,
                "meta": meta,
                "fusion": fusion,
                "profile_info": profile_info,
            }

        return result

    if isinstance(fn, langctx.module_cls):
        _fn = ThunderOptimizedModule(fn, _fn, tfn, tfn._additional_param_names)

    setattr(_fn, "_thunder_cache", None)

    return _fn
