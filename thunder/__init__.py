from functools import wraps, partial
from numbers import Number
from typing import Sequence, Dict, Set, Optional, Any, List, Callable, Tuple, Type
from collections import deque
from enum import auto, Enum
import os
import torch as pytorch
import traceback

from looseversion import LooseVersion

from thunder.core.proxies import is_proxyable, proxy, Proxy
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
from thunder.core.pytree import tree_flatten, tree_unflatten

import thunder.core.script as script
import thunder.core.script.frontend
import thunder.core.script.passes
import thunder.core.script as script

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

# Common UX functions


# TODO Restore caching and ensure inputs are tracked symbolically and not as constants
# https://github.com/Lightning-AI/lightning-thunder/issues/328
#   Need to add arg, vararg, kwarg, and varkwargs names to trace names to avoid conflicts
#   fix this when putting the unpacking into traces
def _unpack_inputs(fn, tracectx: TraceCtx, args, kwargs):
    si = get_siginfo(fn, args, kwargs)

    def proxy_or_track(x: Any, *, name: Optional[str] = None):
        # TODO (mruberry) One idea would be to just unpack the proxy with the new name, but how this interaction works
        #   with nested traces is unclear
        if isinstance(x, Proxy) and name is not None:
            utils.check(
                x.name == name,
                lambda: f"An existing proxy {x} is being passed as an input, but its name is not the same name ({name}) as the unpack is requesting",
                exception_type=NotImplementedError,
            )

        if is_proxyable(x):
            return proxy(x, name=name)

        return tracectx.track(x, name=name)

    def _unpack(x: Any, *, name: str = None) -> List:
        unpacked = None
        colls = []

        # TODO Probably kill collection proxies in favor of tracking
        # TODO Add support for dicts and sets
        # TODO Consider reviewing dict values
        if is_collection(x):
            flat, spec = tree_flatten(x)
            pot_flat = tuple(proxy_or_track(f) for f in flat)
            pot_collection = tree_unflatten(pot_flat, spec)

            # NOTE It's important that the collection is tracked after the unflattening -- because it's now a different
            #   collection that will actually be passed through the program!
            # NOTE At this point the subcollections are still not tracked -- that's handled below
            tracectx.track(pot_collection, name=name)

            items: Sequence
            if isinstance(pot_collection, Sequence):
                unpacked = prims.unpack_sequence(pot_collection, len(pot_collection))
                items = unpacked
            elif isinstance(pot_collection, Dict):
                # NOTE The use of tuple on the keys of the dictionary is important
                #   keys() returns a dict_keys, which is a dictionary view object
                #   (see https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects)
                #   This object is a collection, but treemap will not properly flatten it, resulting
                #   in an infinite recursion when attempting to flatten collections
                # TODO We could consider extending treemap to properly flatten dictionary views
                # TODO We should definitely improve our use of tree_flatten to avoid infinite recursion
                #   when trying to flatten collections recursively
                unpacked = prims.unpack_dict(pot_collection, tuple(pot_collection.keys()))
                items = unpacked.values()
            else:
                utils.check(
                    False, lambda: f"Found an unsupported collection type {type(x)}", exception_type=NotImplementedError
                )

            for o in items:
                if is_collection(o):
                    colls.append(o)
        else:
            pot = proxy_or_track(x, name=name)
            unpacked = prims.unpack_trivial(pot)

        return unpacked, colls

    tracectx.unpacking()

    # Constructs args
    proxyargs = []
    collection_queue = deque()
    for name, x in si.args:
        unpacked, colls = _unpack(x, name=name)
        proxyargs.append(unpacked)
        collection_queue.extend(colls)

    # Handles varargs
    if si.varargs is not None:
        varargs_name, x = si.varargs
        unpacked, colls = _unpack(x, name=varargs_name)
        proxyargs.extend(unpacked)
        collection_queue.extend(colls)

    proxykwargs = {}
    for name, x in si.kwargs.items():
        unpacked, colls = _unpack(x, name=name)
        proxykwargs[name] = unpacked
        collection_queue.extend(colls)

    if si.varkwargs is not None:
        varkwargs_name, x = si.varkwargs
        unpacked, colls = _unpack(x, name=varkwargs_name)
        proxykwargs.update(unpacked)
        collection_queue.extend(colls)

    # NOTE This code is very similar to the above -- maybe they can be refactored together?
    def unpack_recursive(x, q):
        if is_collection(x):
            # NOTE This proxy_or_track is required because the initial track above didn't include sub-collections
            proxy_or_track(x)

            items: Sequence
            if isinstance(x, Sequence):
                items = prims.unpack_sequence(x, len(x))
            elif isinstance(x, Dict):
                # NOTE See note above about why it's important to tuple the dict_keys object
                d = prims.unpack_dict(x, tuple(x.keys()))
                items = d.values()
            else:
                raise NotImplementedError

            for o in items:
                if is_collection(o):
                    q.append(o)

    while True:
        try:
            x = collection_queue.popleft()
            unpack_recursive(x, collection_queue)
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
        additional_param_names, additional_param_values = thunder.core.script.passes.module_to_function(gr)
    script.passes.strongly_inline_functions(gr)
    script.passes.torch_to_thunder(gr)

    thunder_fn = script.python_ir.generate_function(gr)
    if is_module:
        thunder_fn._additional_param_names = additional_param_names
        thunder_fn._additional_param_values = additional_param_values
    else:
        thunder_fn._additional_param_names = None
        thunder_fn._additional_param_values = None
    return thunder_fn


class ThunderOptimizedModule(pytorch.nn.Module):  # TOM
    # todo: subclass nn.Module or forward things like .state_dict() to the
    #       model
    def __init__(self, model, fn, tfn, additional_param_names, additional_param_values):
        super().__init__()
        self._model = model
        self._forward_fn = fn
        self._tfn = tfn
        self._additional_param_values = additional_param_values
        self._additional_param_names = additional_param_names

    def __call__(self, *args, **kwargs):
        all_args = (*self._additional_param_values, *args)
        res = self._forward_fn(*all_args, **kwargs)
        return res


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
def compile_with_info(
    fn: Callable,
    *,
    langctx: Optional[Any] = None,
    executors_list: Optional[List[executors.Executor]] = None,
    only_execute_prims: bool = False,
    disable_preprocessing: bool = False,
) -> Callable:
    pfn: Callable

    import time

    start = time.time_ns()
    if disable_preprocessing:
        pfn = fn
    else:
        pfn = preprocess(fn, is_module=isinstance(fn, pytorch.nn.Module))
    elapsed = time.time_ns()

    @wraps(fn)
    def _fn(*args, **kwargs) -> Tuple[Any, List[TraceCtx]]:
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

            traces: List[TraceCtx] = [trace]

            if started:
                # TODO Add the capability to recover from pass failures
                extrace, extraces = executors.transform_for_execution(
                    trace, executors_list=executors_list, only_execute_prims=only_execute_prims
                )
                traces.extend(extraces)

                c = extrace.python_callable()

                # Attempts to execute the trace, returning the traces
                # TODO Review the pattern of returning the exception
                result: Any
                try:
                    result = c(*args, **kwargs)
                except Exception as e:

                    def print_traces(traces):
                        for trace in traces:
                            print(trace)
                            print("\n\n")

                    print_traces(traces)
                    traceback.print_exception(e)
                    return e, traces
        finally:
            # Ensures the language and tracing contexts are reset
            maybe_reset_trace(started, tok)
            if langctx_tok is not None:
                reset_langctx(langctx_tok)

        return result, traces

    _fn._pfn = pfn

    if isinstance(fn, pytorch.nn.Module):
        _fn = ThunderOptimizedModule(fn, _fn, pfn, pfn._additional_param_names, pfn._additional_param_values)

    return _fn


# NOTE A sugar for compile_with_info that just returns the output of the program
def compile(fn, **compile_kwargs) -> Callable:
    cfn = compile_with_info(fn, **compile_kwargs)

    @wraps(cfn)
    def _fn(*args, **kwargs):
        result, _ = cfn(*args, **kwargs)
        return result

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
def _make_trace(fn: Callable, *, langctx=None) -> Callable:
    """Converts a callable into a callable that will be traced and the trace returned.

    Args:
        fn: The callable to be traced.
        langctx: The language context to use for the trace. If None, the default language context is used.

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

    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            # Sets the proper tracing context
            if langctx is not None:
                langctx_tok = set_langctx(langctx)
            trace = TraceCtx(fn)
            trace_token = set_tracectx(trace)
            proxyargs, proxykwargs = _unpack_inputs(fn, trace, args, kwargs)
            trace = construct_trace(fn, trace, proxyargs, proxykwargs)
        finally:
            # Resets the tracing context
            reset_tracectx(trace_token)
            if langctx is not None:
                reset_langctx(langctx_tok)
        return trace

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
