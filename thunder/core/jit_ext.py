import thunder
import math
from typing import Any, Optional, Dict, Tuple, Literal
import builtins
import collections
from collections.abc import ValuesView, Iterable, Iterator
from collections.abc import Callable, Sequence
import weakref
import random
from functools import partial, wraps, reduce
import linecache
import operator
import copy
import contextvars
from contextlib import contextmanager
import dis
import warnings
from enum import Enum, auto
from io import StringIO
import inspect
import time

from thunder.core.compile_data import compile_data_and_stats, get_cache_option, get_compile_data
import thunder.clang as clang
import thunder.core.transforms
from thunder.core.baseutils import run_once

from types import (
    CellType,
    ClassMethodDescriptorType,
    CodeType,
    CoroutineType,
    FrameType,
    FunctionType,
    MethodType,
    MethodDescriptorType,
    ModuleType,
    NoneType,
    BuiltinFunctionType,
    BuiltinMethodType,
    MethodWrapperType,
    WrapperDescriptorType,
    TracebackType,
    CellType,
    ModuleType,
    CodeType,
    BuiltinFunctionType,
    FunctionType,
    MethodType,
    GetSetDescriptorType,
    UnionType,
)

import torch
import torch.utils.checkpoint
from thunder.core.proxies import (
    DistParallelType,
    proxy,
    Proxy,
    ProxyTag,
    AnyProxy,
    NumberProxy,
    StringProxy,
    TensorProxy,
    FutureTensorProxy,
    make_proxy_name,
    Variable,
    variableify,
    unvariableify,
    is_proxy_name_available,
)
from thunder.core.trace import set_tracectx, reset_tracectx, tracectx, from_trace
from thunder.core.interpreter import (
    InterpreterLogItem,
    InterpreterFrame,
    interpret,
    _interpret_call,
    CapsuleType,
    default_callbacks,
    INTERPRETER_CALLBACKS,
    INTERPRETER_SIGNALS,
    default_opcode_interpreter,
    _default_lookaside_map,
    default_lookaside,
    do_raise,
    get_interpreterruntimectx,
    InterpreterRuntimeCtx,
    is_opaque,
    Py_NULL,
    member_descriptor,
    WrappedValue,
    unwrap,
    wrap,
    wrap_const,
    PseudoInst,
    ProvenanceRecord,
    interpreter_needs_wrap,
)
from thunder.core.langctxs import set_langctx, reset_langctx, Languages, resolve_language
from thunder.core.baseutils import extract_callable_name
from thunder.core.codeutils import get_siginfo, SigInfo
import thunder.core.prims as prims
from thunder.common import transform_for_execution
from thunder.core.options import CACHE_OPTIONS, SHARP_EDGES_OPTIONS, DebugOptions
from thunder.core.symbol import Symbol, BoundSymbol, is_traceable

from thunder.extend import Executor
from thunder.common import CompileData, CompileStats
from thunder.core.trace import TraceCtx, TraceResults
from thunder.torch import _torch_to_thunder_function_map
from thunder.clang import _clang_fn_set
from thunder.core.pytree import tree_map, tree_iter
from thunder.core.compile_data import compile_data_and_stats

#
# jit_ext.py implements extensions of thunder's interpreter
#
ProxyTag.register_tag("STATIC_MEMORY_LOCATION")


EXT_FLAG_IS_PROXY_DERIVED = 1
EXT_FLAG_IS_TENSOR_PROXY = 2
EXT_FLAG_IS_MODULE_MEMBER_DICT = 4
EXT_FLAG_IS_MODULE = 8
EXT_FLAG_IS_CALLABLE = 16
EXT_FLAG_IS_CONSTRAINABLE_INPUT = 32
MODULE_MEMBER_DICT_ATTRS = {
    "_parameters",
    "_modules",
    "_buffers",
    "__dict__",
}


class JITSharpEdgeError(RuntimeError):
    """
    Thrown when the program cannot be safely translated to a thunder program,
    even with interpretation=INTERPRETATION_OPTIONS.TRANSLATE_PYTHON.
    Such cases are referred to as JIT "sharp edges".
    """

    pass


def _general_jit_sharp_edge(desc: str, value: Any, /) -> Any | INTERPRETER_SIGNALS:
    sharp_edges: SHARP_EDGES_OPTIONS = get_jit_ctx().sharp_edges

    s: str = (
        f"{desc} This is currently considered a sharp edge even with interpretation=INTERPRETATION_OPTIONS.TRANSLATE_PYTHON. For cases in which we are overly strict, please file an issue. Thank you!"
    )

    if sharp_edges is SHARP_EDGES_OPTIONS.ERROR:
        return do_raise(JITSharpEdgeError(s))

    # Warn and return value anyway
    if sharp_edges is SHARP_EDGES_OPTIONS.WARN:
        warnings.warn(s)

    return value


def _infer_name_postfix_from_provenance(pr: ProvenanceRecord) -> str:
    # Instructions that are considered terminal for recursions below
    terminal_instructions = {PseudoInst.INPUT_ARGS, PseudoInst.INPUT_FN}

    def get_postfix(pr: ProvenanceRecord):
        if pr.inst in terminal_instructions:
            return [""]
        elif pr.inst == PseudoInst.BINARY_SUBSCR or pr.inst == PseudoInst.LOAD_ATTR:
            # These we recurse over
            assert len(pr.inputs) == 2
            lhs, rhs = pr.inputs
            postfix = get_postfix(lhs)

            if rhs.inst == PseudoInst.CONSTANT:
                rhs_postfix = str(rhs.value)

                if lhs.ext_flag & EXT_FLAG_IS_MODULE:
                    if rhs_postfix not in MODULE_MEMBER_DICT_ATTRS:
                        postfix.append(rhs_postfix)
                else:
                    postfix.append(rhs_postfix)

            return postfix
        else:
            # Skip as if terminal for now
            # TODO: improve this later
            return [""]

    return "_".join(get_postfix(pr))


class JitCtx:
    def __init__(
        self,
        prologue_trace,
        computation_trace,
        *,
        sharp_edges: SHARP_EDGES_OPTIONS,
        process_group_for_ddp=None,
        executor_lookasides,
        ad_hoc_executor,
    ):
        self._sharp_edges: SHARP_EDGES_OPTIONS = sharp_edges
        self._prologue_trace = prologue_trace
        self._computation_trace: TraceCtx = computation_trace
        self._constraints = []
        self._process_group_for_ddp = process_group_for_ddp
        self._additional_outputs = collections.defaultdict(list)
        self._proxy_swapmap: dict[Variable, Proxy] = {}
        self._executor_lookasides: dict[Callable, Callable] = executor_lookasides
        self._ad_hoc_executor = ad_hoc_executor

    @property
    def ad_hoc_executor(self):
        return self._ad_hoc_executor

    @property
    def sharp_edges(self) -> SHARP_EDGES_OPTIONS:
        return self._sharp_edges

    @property
    def prologue_trace(self) -> TraceCtx:
        return self._prologue_trace

    @property
    def computation_trace(self) -> TraceCtx:
        return self._computation_trace

    def add_constraint(self, constraint):
        self._constraints.append(constraint)

    def proxify(self, value: WrappedValue) -> Any:
        assert isinstance(value, WrappedValue)
        uvalue = value.value
        # Sequence / dict is not registered as Proxy
        # avoid double registration by skipping if value has a registered proxy.
        if isinstance(uvalue, Proxy) or value.original_value is not value.nothing:
            return uvalue
        elif isinstance(uvalue, torch.device):
            co: CACHE_OPTIONS = get_cache_option()
            p: AnyProxy = proxy(uvalue, history=value.provenance)
            if co in (CACHE_OPTIONS.CONSTANT_VALUES, CACHE_OPTIONS.SYMBOLIC_VALUES):
                # NOTE: Even with SYMBOLIC_VALUES, we want to strictly constraint the device as
                # the computation trace may utilize device specific executors.
                self.add_constraint((clang.check_literal_like, p, uvalue))
            elif co in (CACHE_OPTIONS.SAME_INPUT,):
                raise NotImplementedError(f"Unsupported cache option {co}")
            else:  # co is CACHE_OPTIONS.NO_CACHING
                pass
        elif isinstance(uvalue, torch.Tensor):
            # we always want to proxy torch.Tensor, even const

            name_postfix = _infer_name_postfix_from_provenance(value.provenance)
            if name_postfix:
                name = f"t{name_postfix}"
            else:
                name = None

            p = proxy(uvalue, name=name, history=value.provenance)

            # TensorProxy attributes should be considered derived quantities, so we flag TensorProxies here
            value.provenance.ext_flag |= EXT_FLAG_IS_TENSOR_PROXY

            if isinstance(p, TensorProxy) and p.distparallel_type in (
                DistParallelType.REPLICATED,
                DistParallelType.FULLY_SHARDED,
            ):
                p_new = thunder.distributed.prims.synchronize(
                    p,
                    self._process_group_for_ddp,
                )
                if isinstance(p.thunder_fsdp_padding_size, int):
                    p_new = p_new[: (p_new.shape[0] - p.thunder_fsdp_padding_size)]
                p_orig = p
                p = p_new
            else:
                p_orig = p
            if p is not uvalue:
                value.register_proxy(p)
            # TODO: other caching modes
            co: CACHE_OPTIONS = get_cache_option()
            if co is CACHE_OPTIONS.CONSTANT_VALUES:
                self.add_constraint((clang.check_tensor_shape_and_metadata, p_orig))
            elif co is CACHE_OPTIONS.SYMBOLIC_VALUES:
                # TODO: establish guarding logic to allow non-broadcast shape change
                self.add_constraint((clang.check_tensor_shape_and_metadata, p_orig))
            elif co not in (CACHE_OPTIONS.SAME_INPUT, CACHE_OPTIONS.NO_CACHING):
                raise NotImplementedError(f"Unsupported cache option {co}")
            return p

        elif isinstance(uvalue, (float, int, complex, str, slice)):
            assert should_register_for_prologue(value.provenance)
            value.provenance.ext_flag |= EXT_FLAG_IS_PROXY_DERIVED
            # we follow the caching mechanisms of the eager_unpack_interpreter
            p = proxy(uvalue, history=value.provenance)
            if value.provenance.ext_flag & EXT_FLAG_IS_CONSTRAINABLE_INPUT and hasattr(p, "make_constrainable"):
                p.make_constrainable()
            assert p.history is not None, f"{p.history}, {value.provenance} {type(p)}"

            co: CACHE_OPTIONS = get_cache_option()
            if co is CACHE_OPTIONS.CONSTANT_VALUES:
                if isinstance(uvalue, str):
                    self.add_constraint((clang.check_string_value, p, uvalue))
                elif isinstance(uvalue, slice):
                    self.add_constraint((clang.check_slice_value, p, uvalue))
                else:
                    self.add_constraint((clang.check_number_type_and_value, p, uvalue))
            elif co is CACHE_OPTIONS.SYMBOLIC_VALUES:
                if p is not uvalue:
                    value.register_proxy(p)
            elif co not in (CACHE_OPTIONS.SAME_INPUT, CACHE_OPTIONS.NO_CACHING):
                raise NotImplementedError(f"Unsupported cache option {co}")
            return p
        elif isinstance(uvalue, dict):
            value.track_items()
            proxy_d = type(uvalue)((k, i.value) for k, i in value.item_wrappers.items())
            value.register_proxy(proxy_d)
            for an, av in value.attribute_wrappers.items():
                if callable(av.value):
                    av.register_proxy(getattr(proxy_d, an))
                else:
                    raise NotImplementedError(
                        f"proxify {type(uvalue).__name__} with attribute {an} of type {type(av.value).__name__}"
                    )
            return proxy_d
        elif isinstance(uvalue, Sequence):
            value.track_items()
            proxy_s = type(uvalue)(i.value for i in value.item_wrappers)
            value.register_proxy(proxy_s)
            for an, av in value.attribute_wrappers.items():
                if callable(av.value):
                    av.register_proxy(getattr(proxy_s, an))
                else:
                    raise NotImplementedError(
                        f"proxify {type(uvalue).__name__} with attribute {an} of type {type(av.value).__name__}"
                    )
            return proxy_s
        else:
            raise ValueError("cannot proxify value of {type(uvalue).__type} objects")


_jit_ctx = contextvars.ContextVar("jitctx")


def set_jit_ctx(ctx: JitCtx) -> Any:
    return _jit_ctx.set(ctx)


def get_jit_ctx() -> JitCtx:
    return _jit_ctx.get()


def reset_jit_ctx(token) -> None:
    _jit_ctx.reset(token)


general_jit_callbacks: dict[INTERPRETER_CALLBACKS, Callable] = {}


def register_general_jit_callback(key: INTERPRETER_CALLBACKS) -> Callable:
    def decorator(fn: Callable):
        assert key not in general_jit_callbacks
        general_jit_callbacks[key] = fn
        return fn

    return decorator


#
# general_jit lookasides
#

_general_jit_lookaside_map = {}


def ensure_recursive_proxies(fn):  # shortcut for things we already processed?
    @wraps(fn)
    def wrapper(*args, **kwargs):
        recursively_proxy(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


def record_source_loc_in_symbol_header(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
        filename, positions = runtimectx.get_current_user_source_location()
        ctx: JitCtx = get_jit_ctx()
        ctx._computation_trace.set_current_source_location(filename, positions)
        return fn(*args, **kwargs)

    return wrapper


_general_jit_lookaside_map.update(
    {
        k: ensure_recursive_proxies(interpreter_needs_wrap(record_source_loc_in_symbol_header(v)))
        for k, v in _torch_to_thunder_function_map.items()
    }
)


def register_general_jit_lookaside(diverted_fn):
    def lookaside_wrapper(lookaside):
        _general_jit_lookaside_map[diverted_fn] = lookaside
        return lookaside

    return lookaside_wrapper


# PyTorch moved this to torch.compiler.is_compiling as official API
# we are compiling
if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
    is_compiling = torch.compiler.is_compiling
else:
    is_compiling = torch._dynamo.is_compiling


@register_general_jit_lookaside(is_compiling)
@interpreter_needs_wrap
def jit_is_compiling_lookaside():
    return True


# lookaside for getattr. We record the provenance of the attribute but for the core attribute getting, we
# rely on the default JIT getattr lookaside (as returned from default_lookaside)
@register_general_jit_lookaside(getattr)
def _general_jit_getattr_lookaside(obj: Any, name: str, *maybe_default: Any):
    getattr_lookaside = default_lookaside(getattr)
    assert getattr_lookaside is not None

    value = getattr_lookaside(obj, name, *maybe_default)
    if value is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return value

    assert isinstance(value, WrappedValue)
    assert isinstance(name, WrappedValue)

    if (not maybe_default) and (value is not INTERPRETER_SIGNALS.EXCEPTION_RAISED):
        if isinstance(unwrap(obj), torch.nn.Module) and (unwrap(name) in MODULE_MEMBER_DICT_ATTRS):
            value.provenance.ext_flag |= EXT_FLAG_IS_MODULE_MEMBER_DICT

    return value


@register_general_jit_lookaside(isinstance)
def _general_jit_isinstance_lookaside(obj: Any, cls: type | UnionType | tuple[type | UnionType]):
    uobj = unwrap(obj)
    ucls = unwrap(cls)
    if isinstance(uobj, TensorProxy):
        res = issubclass(torch.Tensor, ucls)
        # We represent `nn.Parameters` with `TensorProxy`,
        # so to support `isinstance(t, torch.nn.Parameter)`
        # we peek at the original python_type of the wrapped object.
        if not isinstance(ucls, (tuple, list)):
            ucls = (ucls,)
        if torch.nn.Parameter in ucls:
            res = issubclass(obj.python_typ, ucls)
    else:
        res = isinstance(uobj, ucls)

    pr = ProvenanceRecord(
        PseudoInst.LOOKASIDE, inputs=[wrap_const(isinstance).provenance, obj.provenance, cls.provenance]
    )
    return wrap(res, provenance=pr)


# PyTorch >= 2.4 uses dict, <=2.3 uses OrderedDict
@register_general_jit_lookaside(dict.__setitem__)
def _general_jit_dict_setitem(d, key, value):
    dict_setitem_lookaside = default_lookaside(dict.__setitem__)
    assert dict_setitem_lookaside is not None

    if d.provenance.ext_flag & EXT_FLAG_IS_MODULE_MEMBER_DICT:
        ctx: JitCtx = get_jit_ctx()
        if d.original_value is d.nothing:
            ctx.proxify(d)
        ctx._additional_outputs[d].append((PseudoInst.STORE_SUBSCR, d, key, value))

    return dict_setitem_lookaside(d, key, value)


@register_general_jit_lookaside(collections.OrderedDict.__setitem__)
def _general_jit_ordered_dict_setitem(d, key, value):
    dict_setitem_lookaside = default_lookaside(collections.OrderedDict.__setitem__)
    assert dict_setitem_lookaside is not None

    if d.provenance.ext_flag & EXT_FLAG_IS_MODULE_MEMBER_DICT:
        ctx: JitCtx = get_jit_ctx()
        if d.original_value is d.nothing:
            ctx.proxify(d)
        ctx._additional_outputs[d].append((PseudoInst.STORE_SUBSCR, d, key, value))

    return dict_setitem_lookaside(d, key, value)


@register_general_jit_lookaside(setattr)
def _general_jit_setattr_lookaside(obj: Any, name: str, value: Any):
    setattr_lookaside = default_lookaside(setattr)
    assert setattr_lookaside is not None

    uobj = unwrap(obj)
    uname = unwrap(name)
    if isinstance(uobj, torch.nn.Module):
        # 1) modify the inner thing
        # 2) divert the actual setattr...
        for n in MODULE_MEMBER_DICT_ATTRS:
            member_dict = _interpret_call(getattr, obj, wrap_const(n))
            member_dict.provenance.ext_flag |= EXT_FLAG_IS_MODULE_MEMBER_DICT

    # check if it is an "outside value"?
    res = setattr_lookaside(obj, name, value)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res
    return res


@register_general_jit_lookaside(torch.compile)
def _jit_torch_compile_lookaside(*args, **kwargs):
    return do_raise(
        NotImplementedError(
            "Using torch.compile within a function to be JIT-compiled by Thunder is not supported. "
            "Please remove the call to torch.compile or apply it outside the function."
        )
    )


# TODO Expand on this
@interpreter_needs_wrap
def _general_jit_hasattr_lookaside(obj: Any, name: str):
    hasattr_lookaside = default_lookaside(hasattr) or hasattr
    return hasattr_lookaside(obj, name)


_general_jit_lookaside_map[hasattr] = _general_jit_hasattr_lookaside


# We want to record a constraint when we go from proxy -> value here.
# At the same time Python expects to (but we might think to loosen the requirement
# to return a bool for the JIT, return a proxy with origin informaiton and postpone
# recording the constraint to conditional jumps and such.
def _general_jit_bool_lookaside(wrapped_x: Any) -> bool | INTERPRETER_SIGNALS:
    assert isinstance(wrapped_x, WrappedValue)
    # It doesn't feel right to insert constraints in bool lookaside, constraints here only applies when the bool value is used in control flow.
    if isinstance(wrapped_x.value, NumberProxy):
        if wrapped_x.value.is_dynamic():
            raise NotImplementedError(f"conversion to bool is not allowed on dynamic proxy={wrapped_x.value}")
        wrapped_x.value.make_static_constrained()
    bool_lookaside = default_lookaside(bool) or bool
    return bool_lookaside(wrapped_x)


_general_jit_lookaside_map[bool] = _general_jit_bool_lookaside


def _get_torch_nn_module_named_members_lookaside(
    model: torch.nn.Module, named_member_method, get_member_method, *unwrapped_args, **unwrapped_kwargs
):
    assert isinstance(model, torch.nn.Module)

    # Get the value of `prefix` if it was passed.
    prefix = ""
    if len(unwrapped_args) > 1:  # as positional arg
        prefix = unwrapped_args[1]
    elif "prefix" in unwrapped_kwargs:  # as kwarg
        prefix = unwrapped_kwargs["prefix"]

    # Here we call the unwrapped model with unwrapped arguments
    # to get the names of parameter or buffer that we are interested in.
    # We then use these names in `_get_named_member_impl` to get the corresponding proxies.
    # NOTE: If prefix was passed, these names will be qualified with prefix.
    member_names = {name for name, _ in named_member_method(*unwrapped_args, **unwrapped_kwargs)}

    # NOTE: This will be interpreted.
    def _get_named_member_impl(members):
        for member in members:
            org_name = member
            if prefix:
                # Undo prefix if it was passed to correctly get the corresponding
                # parameter/buffer.
                org_name = org_name.replace(prefix + ".", "")
            b = get_member_method(org_name)
            yield member, b

    pr = ProvenanceRecord(PseudoInst.LOOKASIDE, inputs=[wrap_const(named_member_method).provenance])

    return _interpret_call(_get_named_member_impl, wrap(member_names, provenance=pr))


@register_general_jit_lookaside(torch.nn.Module.named_parameters)
def _general_jit_named_parameters_lookaside(obj: Any, *args, **kwargs):
    model = unwrap(obj)
    unwrapped_args = tuple(unwrap(arg) for arg in args)
    unwrapped_kwargs = {unwrap(k): unwrap(v) for k, v in kwargs.items()}
    return _get_torch_nn_module_named_members_lookaside(
        model, model.named_parameters, model.get_parameter, *unwrapped_args, **unwrapped_kwargs
    )


@register_general_jit_lookaside(torch.nn.Module.named_buffers)
def _general_jit_named_buffers_lookaside(obj: Any, *args, **kwargs):
    model = unwrap(obj)
    unwrapped_args = tuple(unwrap(arg) for arg in args)
    unwrapped_kwargs = {unwrap(k): unwrap(v) for k, v in kwargs.items()}
    return _get_torch_nn_module_named_members_lookaside(
        model, model.named_buffers, model.get_buffer, *unwrapped_args, **unwrapped_kwargs
    )


def _convert_pytorchfunc_to_thundertrace(
    func: Callable[[Any], Any],
    shallow_copy_output: bool,
    *args,
    **kwargs,
) -> tuple[TraceCtx | INTERPRETER_SIGNALS, ProvenanceRecord | None]:
    """Converts pytorch function to thunder trace.

    Note that the generated trace would not have _siginfo and args set.

    Args:
        func: A callable composed of pytorch functions.
        shallow_copy_output: Needs to be :obj:`True` only if func is `torch.autograd.Function.apply` as
            it produces views of the tensor to attach the autograd node to.
        *args:
        **kwargs
    """
    from thunder.core.baseutils import sequencify

    active_jit_ctx: JitCtx = get_jit_ctx()
    active_jit_ctx.computation_trace.push_scope([])
    wrapped_func_result = _interpret_call(func, *args, **kwargs)
    if wrapped_func_result is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return wrapped_func_result, None

    trace = TraceCtx()
    bsyms = active_jit_ctx.computation_trace.pop_scope()
    trace.bound_symbols.extend(bsyms)
    func_result = unwrap(wrapped_func_result)
    if shallow_copy_output and not bsyms:
        from thunder.core.baseutils import sequencify

        out_to_shallow_copy: dict[Variable, TensorProxy] = {}
        for a in sequencify(func_result):
            shallow_copy_of_a = prims.shallow_copy.meta(a)
            bsym = prims.shallow_copy.bind(a, output=shallow_copy_of_a)
            trace.add_bound_symbol(bsym)
            out_to_shallow_copy[variableify(a)] = shallow_copy_of_a
        func_result = tree_map(lambda t: out_to_shallow_copy.get(variableify(t), t), func_result)
    with tracectx(trace):
        prims.python_return(func_result)
    return trace, sequencify(wrapped_func_result)[0].provenance


@register_general_jit_lookaside(torch.autograd.function.Function.apply.__func__)
def _general_jit_torch_autograd_function_apply_lookaside(obj: Any, *args, **kwargs):
    """Encapsulate forward into a bsym, define and register augmented fwd and bwd.

    This lookaside does three things:
        1. Encapsulate ``MyFunc(torch.autograd.Function).forward`` into :class:`~thunder.core.symbol.BoundSymbol`
        2. Define an augmented forward which is different from ``MyFunc.forward`` in its return values: returning
           ``ctx.saved_tensors`` as residuals.
        3. Trace ``MyFunc.backward``, define :class:`~thunder.core.trace.TraceCtx` whose args are ``(*residuals, *grads)``.
           So far, non-tensor ``ctx`` attributes seem to be folded into a trace.
    """
    from thunder.core.baseutils import check, sequencify

    custom_autograd_function_cls = unwrap(obj)
    custom_forward = custom_autograd_function_cls.forward
    ctx = torch.autograd.function.FunctionCtx()
    ctx_proxy = proxy(ctx, name=None, history=None)
    wrapped_ctx = wrap_const(ctx_proxy)
    trace_of_fwd, fwd_output_provenance = _convert_pytorchfunc_to_thundertrace(
        custom_forward, True, wrapped_ctx, *args, **kwargs
    )
    if trace_of_fwd is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return trace_of_fwd

    # Forward.
    unwrapped_custom_forward_args = tree_map(lambda a: unwrap(a), args)
    trace_of_fwd._siginfo = SigInfo.from_name_and_args(
        custom_autograd_function_cls.__name__,
        unwrapped_custom_forward_args,
    )
    trace_of_fwd.args = unwrapped_custom_forward_args
    unpack_bsyms = [
        prims.unpack_trivial.bind(a, name=a.name, output=a)
        for a in filter(lambda a: isinstance(a, Proxy), trace_of_fwd.args)
    ]
    trace_of_fwd.bound_symbols = unpack_bsyms + trace_of_fwd.bound_symbols

    @wraps(trace_of_fwd.python_callable())
    def core_of_forward(*args, **kwargs):
        return thunder.core.trace_interpreter.interpret_trace(trace_of_fwd, *args, **kwargs)

    custom_fwd_sym = get_jit_ctx().ad_hoc_executor.register_operator(
        trace_of_fwd._siginfo.name,
        like=core_of_forward,
    )
    unwrapped_forward_result = custom_fwd_sym(*unwrapped_custom_forward_args)
    forward_result = wrap(
        unwrapped_forward_result,
        provenance=ProvenanceRecord(PseudoInst.LOOKASIDE, inputs=[obj.provenance, fwd_output_provenance]),
    )

    augmented_bsym_output: tuple[tuple[TensorProxy, ...], tuple[TensorProxy, ...]] = (
        tuple(sequencify(trace_of_fwd.output)),
        ctx_proxy.saved_tensors,
    )
    trace_of_augmented_fwd = TraceCtx()
    trace_of_augmented_fwd.bound_symbols.extend(trace_of_fwd.bound_symbols[:-1])
    with tracectx(trace_of_augmented_fwd):
        prims.python_return(augmented_bsym_output)
    trace_of_augmented_fwd._siginfo = SigInfo.from_name_and_args(custom_fwd_sym.name, unwrapped_custom_forward_args)
    trace_of_augmented_fwd.args = unwrapped_custom_forward_args

    # Backward definition
    custom_backward = custom_autograd_function_cls.backward
    grads = tree_map(
        lambda a: a.replace_name(f"grad_{a.name}"),
        sequencify(trace_of_fwd.output),
    )
    wrapped_grads = tree_map(lambda g: wrap(g, provenance=fwd_output_provenance), grads)
    trace_of_backward, _ = _convert_pytorchfunc_to_thundertrace(custom_backward, False, wrapped_ctx, *wrapped_grads)
    if trace_of_backward is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return trace_of_backward
    trace_of_backward._siginfo = SigInfo.from_name_and_args(
        f"{custom_fwd_sym.name}_backward",
        ctx_proxy.saved_tensors + grads,
    )
    trace_of_backward.args = tuple(ctx_proxy.saved_tensors + grads)
    bwd_unpack_bsyms = [
        prims.unpack_trivial.bind(a, name=a.name, output=a)
        for a in filter(lambda a: isinstance(a, Proxy), trace_of_backward.args)
    ]
    trace_of_backward.bound_symbols = bwd_unpack_bsyms + trace_of_backward.bound_symbols

    bwd_trace_impl = TraceCtx()
    bwd_trace_impl.bound_symbols.extend(trace_of_backward.bound_symbols)
    bwd_trace_impl._siginfo = SigInfo.from_name_and_args(
        "backward_impl",
        ctx_proxy.saved_consts + ctx_proxy.saved_tensors + grads,
    )
    bwd_trace_impl.args = tuple(ctx_proxy.saved_consts + ctx_proxy.saved_tensors + grads)

    @wraps(bwd_trace_impl.python_callable())
    def bwd_impl_callable(*args, **kwargs):
        return thunder.core.trace_interpreter.interpret_trace(bwd_trace_impl, *args, **kwargs)

    @wraps(core_of_forward)
    def grad_transform(*args, **kwargs):
        from thunder.core.transforms import get_grad
        from thunder.core.transforms import put_grads
        from thunder.core.trace_interpreter import interpret_trace

        check(not kwargs, lambda: f"{kwargs=} should be empty")
        primal, residuals = interpret_trace(trace_of_augmented_fwd, *args, **kwargs)
        check(len(primal) == 1, lambda: f"{primal=} has {len(primal)} proxies but expected 1")
        grads = tree_map(lambda t: get_grad(t), primal)
        bwd_args = ctx_proxy.saved_consts + tuple(sequencify(residuals)) + grads
        result = bwd_impl_callable(*bwd_args)
        put_grads(args, result)
        return primal

    get_jit_ctx().ad_hoc_executor.register_implementation(
        custom_fwd_sym,
        execution_transform=core_of_forward,
        grad_transform=grad_transform,
    )
    return forward_result


# ref: https://github.com/pytorch/pytorch/blob/38114ec/torch/_functorch/autograd_function.py#L715-L752
@register_general_jit_lookaside(torch.ops.higher_order.autograd_function_apply)
def _general_jit_torch_ops_higher_order_autograd_function_apply(fwd, bwd, *fwd_args, **fwd_kwargs):
    from thunder.core.baseutils import sequencify
    from thunder.core.pytree import tree_map
    from thunder.core.trace_interpreter import interpret_trace

    def _generate_random_str_id() -> str:
        import secrets
        import string

        length = 5
        return "".join(secrets.choice(string.ascii_lowercase) for _ in range(length))

    args_tensor_mask = unwrap(fwd_kwargs["args_tensor_mask"])
    # TODO(crcrpar): Think about making use of `non_differentiable_idx`
    # note that this key is quite new: https://github.com/pytorch/pytorch/pull/134087
    # non_differentiable_idx = fwd_kwargs.get("non_differentiable_idx")
    length_of_tensor_args = sum(args_tensor_mask)
    new_fwd_args = (wrap_const(None),) + fwd_args[:length_of_tensor_args]

    aug_fwd_trace, aug_fwd_provenance = _convert_pytorchfunc_to_thundertrace(fwd, False, *new_fwd_args)
    if aug_fwd_trace is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return aug_fwd_trace
    aug_fwd_result = aug_fwd_trace.output
    output, saved_values = unwrap(aug_fwd_result)
    unwrapped_fwd_args = tree_map(lambda t: unwrap(t), new_fwd_args)

    tmp_name = _generate_random_str_id()
    aug_fwd_trace.args = unwrapped_fwd_args
    aug_fwd_trace._siginfo = SigInfo.from_name_and_args(
        f"higher_order_autograd_function_apply_{tmp_name}",
        aug_fwd_trace.args,
    )

    trace_of_forward = from_trace(aug_fwd_trace)
    for bsym in aug_fwd_trace.bound_symbols:
        if bsym.sym.id == prims.PrimIDs.RETURN:
            continue
        trace_of_forward.bound_symbols.append(bsym)
    with tracectx(trace_of_forward):
        prims.python_return(*(sequencify(output)))

    @wraps(aug_fwd_trace.python_callable())
    def forward(*args, **kwargs):
        return interpret_trace(trace_of_forward, *args, **kwargs)

    grads = sequencify(tree_map(lambda t: TensorProxy(like=t), sequencify(output)))
    bwd_tensor_args = grads + tuple(saved_values)
    bwd_args = (None,) + bwd_tensor_args
    wrapped_bwd_args = tree_map(lambda t: wrap(t, provenance=aug_fwd_provenance), bwd_args)
    bwd_trace, bwd_trace_provenance = _convert_pytorchfunc_to_thundertrace(
        bwd,
        False,
        *wrapped_bwd_args,
    )
    if bwd_trace is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return bwd_trace
    bwd_trace.args = bwd_args
    bwd_unpack_bsyms = [
        prims.unpack_trivial.bind(a, name=a.name, output=a)
        for a in filter(lambda a: isinstance(a, Proxy), bwd_trace.args)
    ]
    bwd_trace.bound_symbols = bwd_unpack_bsyms + bwd_trace.bound_symbols
    bwd_trace._siginfo = SigInfo.from_name_and_args(f"bwd_{tmp_name}", bwd_trace.args)

    @wraps(forward)
    def grad_transform(*args, **kwargs):
        from thunder.core.transforms import get_grad, put_grads

        primal, residuals = interpret_trace(aug_fwd_trace, *args, **kwargs)
        grads = tree_map(lambda t: get_grad(t), sequencify(primal))
        bwd_args = (None,) + tuple(grads) + tuple(sequencify(residuals))
        result = interpret_trace(bwd_trace, *bwd_args)
        put_grads(args[1:], result)

        return primal

    forward_op = get_jit_ctx().ad_hoc_executor.register_operator(trace_of_forward._siginfo.name, like=forward)
    unwrapped_output = forward_op(*unwrapped_fwd_args)
    output = wrap(
        unwrapped_output, provenance=ProvenanceRecord(PseudoInst.LOOKASIDE, inputs=[fwd.provenance, aug_fwd_provenance])
    )
    get_jit_ctx().ad_hoc_executor.register_implementation(
        forward_op,
        execution_transform=forward,
        grad_transform=grad_transform,
    )
    return output


@register_general_jit_lookaside(torch.autocast.__enter__)
def autocast_enter(autocast_obj):
    unwrap_autocast_obj = unwrap(autocast_obj)
    device = unwrap_autocast_obj.device
    dtype = unwrap_autocast_obj.fast_dtype
    enabled = unwrap_autocast_obj._enabled
    # NOTE - We manually map `torch.autocast.__enter__` to `torch.amp.autocast_mode._enter_autocast`
    #        as it is functional variant of the same. This also allows the symbol to appear in trace
    #        for better inspectibility.
    thunder_fn = _torch_to_thunder_function_map[torch.amp.autocast_mode._enter_autocast]
    thunder_fn(device, dtype, enabled)
    return wrap(None, provenance=ProvenanceRecord(PseudoInst.LOOKASIDE, inputs=[autocast_obj.provenance]))


@register_general_jit_lookaside(torch.autocast.__exit__)
def autocast_exit(autocast_obj, exc_type, exc_val, exc_tb):
    thunder_fn = _torch_to_thunder_function_map[torch.amp.autocast_mode._exit_autocast]
    thunder_fn()
    # NOTE - We manually map `torch.autocast.__exit__` to `torch.amp.autocast_mode._exit_autocast`
    #        as it is functional variant of the same. This also allows the symbol to appear in trace
    #        for better inspectibility.
    return wrap(None, provenance=ProvenanceRecord(PseudoInst.LOOKASIDE, inputs=[autocast_obj.provenance]))


@register_general_jit_lookaside(torch.finfo)
@interpreter_needs_wrap
def _general_jit_torch_finfo_lookaside(dtype: thunder.dtypes.dtype):
    torch_dtype = thunder.dtypes.to_torch_dtype(dtype)
    res = torch.finfo(torch_dtype)
    return res


@register_general_jit_lookaside(torch.utils.checkpoint.checkpoint)
def _general_jit_torch_checkpoint_lookaside(
    function: Callable,
    *args,
    **kwargs: Any,
):
    """
    This function does preprocessing of the `function` argument before
    dispatching the call to `thunder.torch.checkpoint`. This is necessary
    because the `function` is potentially calling into PyTorch functions that
    are not yet translated to Thunder. `thunder.torch.checkpoint` is a Thunder
    function that can handle only Thunder functions as input.

    Args:
        function: The function to be checkpointed.
        args: Arguments to the function.
        kwargs: Keyword arguments to the function.

    Returns:
        The result of calling `thunder.torch.checkpoint` with the preprocessed
        `function` and its arguments.
    """
    from thunder.torch import checkpoint

    # It should be possible to call the general_thunder_jit here to handle the
    # conversion from torch to thunder but it doesn't work now
    # See https://github.com/Lightning-AI/lightning-thunder/issues/1126
    # TODO: Convert the function to a Thunder function
    def thunder_function(*args, **kwargs):
        return unwrap(function)(*args, **kwargs)

    wrapped_thunder_function = wrap_const(thunder_function)
    return interpreter_needs_wrap(checkpoint)(wrapped_thunder_function, *args, **kwargs)


# Adds proxy methods
# NOTE These methods map to themselves, which prevents the interpreter from looking into them
#   This is OK because these methods are written in a tracing-safe manner, and trying to
#   interpreter their internals is unnecessary and would just add complexity at this time


@interpreter_needs_wrap
def prop_lookaside_helper(meth, /, *args, **kwargs):
    res = meth(*args, **kwargs)
    return res


def prop_lookaside_wrap(attr_getter):
    def fn(obj, /, *args, **kwargs):
        attr = attr_getter(obj)

        if callable(attr):

            def fn_(*args, **kwargs):
                return prop_lookaside_helper(attr, *args, **kwargs)

        else:
            return attr

        return fn_

    return fn


def get_methods_properties(typ):
    for meth_name in dir(typ):
        meth = getattr(typ, meth_name)
        if isinstance(meth, (MethodType, BuiltinMethodType, MethodDescriptorType, WrapperDescriptorType)) and (
            getattr(meth, "__objclass__", None) == typ or (getattr(meth, "__self__", None) == typ)
        ):
            yield meth, meth
        elif isinstance(meth, FunctionType):
            yield meth, meth  # __getattr__
        elif isinstance(meth, property):
            if meth.fget is not None:
                yield meth.fget, prop_lookaside_wrap(meth.fget)


_general_jit_lookaside_map.update(
    {
        **{
            fn: interpreter_needs_wrap(record_source_loc_in_symbol_header(la))
            for fn, la in get_methods_properties(NumberProxy)
        },
        **{
            fn: ensure_recursive_proxies(interpreter_needs_wrap(record_source_loc_in_symbol_header(la)))
            for fn, la in get_methods_properties(TensorProxy)
        },
        prop_lookaside_helper: prop_lookaside_helper,
    }
)


# when we pass containers to the computation trace, we want these to be using the proxies
def recursively_proxy(*args, **kwargs):
    def proxy_recursion(v):
        if isinstance(v.value, str):
            need_proxy = False
        elif isinstance(v.value, (Sequence, dict)):
            v.track_items()
            need_proxy = any(proxy_recursion(i) for i in v.item_wrappers)
        else:
            need_proxy = isinstance(v.value, torch.Tensor)
        if need_proxy:
            ctx: JitCtx = get_jit_ctx()
            ctx.proxify(v)
        is_proxied = v.original_value is not v.nothing
        return is_proxied

    for a in args:
        proxy_recursion(a)
    for v in kwargs.values():
        proxy_recursion(v)


# TODO Document this function (with steps)
def general_jit_lookaside(fn, *args, **kwargs) -> None | Callable:
    # Identifies the lookaside
    lookaside: None | Callable

    ctx: JitCtx = get_jit_ctx()

    if thunder.core.utils.is_hashable(fn):  # see issue #889
        if (executor_lookaside := ctx._executor_lookasides.get(fn, None)) is not None:
            lookaside = executor_lookaside
        # the ad hoc executor may be extended during compilation
        elif (executor_lookaside := ctx.ad_hoc_executor._lookasides.get(fn, None)) is not None:
            lookaside = interpreter_needs_wrap(executor_lookaside)
        elif isinstance(fn, Symbol) or fn in _clang_fn_set:
            # Performs symbol lookasides
            # NOTE Symbols "lookaside" to themselves; this just prevents their internals from being jitted
            # NOTE clang operations are not symbols, but we still prevent their internals from being jitted
            recursively_proxy(*args, **kwargs)
            lookaside = interpreter_needs_wrap(record_source_loc_in_symbol_header(fn))
        elif (general_jit_lookaside := _general_jit_lookaside_map.get(fn, None)) is not None:
            lookaside = general_jit_lookaside
        else:
            # Falls through to the interpreter's default lookaside
            lookaside = default_lookaside(fn, *args, **kwargs)
    else:  # non-hashable
        lookaside = None

    if lookaside is None:

        def is_from_torch(fn):
            module = getattr(fn, "__module__", None)
            if module is None:
                module = getattr(getattr(fn, "__objclass__", None), "__module__", None)
            if module is None:
                return False
            if module.startswith("torch"):
                return module
            return False

        has_tensor_arg = False
        for a in args:
            if isinstance(a.value, TensorProxy):
                has_tensor_arg = True
                break
            if isinstance(a.value, Sequence):
                if any(isinstance(i, TensorProxy) for i in a.value):
                    has_tensor_arg = True
                    break

        if is_opaque(fn) and (torch_module_name := is_from_torch(fn)) and has_tensor_arg:
            if torch_module_name.startswith("torch._C"):
                return lookaside

            # Torch functions have __name__ defined
            fn_name = f"{fn.__module__}.{fn.__name__}"

            # Probably merge with sharp edges
            calling_opaque_torch_msg = (
                f"Trying to call function {fn_name}, but it is not yet supported. "
                "Please file an issue requesting support. "
                "To find out which operations are not yet recognized by `thunder.jit`, "
                "please run `examine` as per:\n\n"
                "from thunder.examine import examine\n"
                "examine(<your thunder.jit callable argument>, ...)\n"
            )

            return do_raise(NotImplementedError(calling_opaque_torch_msg))

        elif is_opaque(fn) and has_tensor_arg:
            # We whitelist a few built-in things
            if fn.__name__ == "__new__":
                objectclass = fn.__self__
            else:
                objectclass = getattr(fn, "__objclass__", None)
            if objectclass is not None:
                module = getattr(objectclass, "__module__", None)
            else:
                module = getattr(fn, "__module__", None)
            if module is not None:
                module, *_ = module.split(".", 1)
            if module not in {"builtins", "_operator", "optree"}:
                # TODO: warn?
                ctx.ad_hoc_executor.register_operator_for_opaque_function(fn)
                return interpreter_needs_wrap(ctx.ad_hoc_executor._lookasides[fn])

    return lookaside


#
# callbacks and callback utilities
#


@contextmanager
def jit_ctx(ctx: JitCtx):
    token = set_jit_ctx(ctx)
    try:
        yield
    finally:
        reset_jit_ctx(token)


def _general_jit_const_callback(value: Any) -> WrappedValue:
    return value


# TODO(nikitaved): maybe call it upon Frame creation
def _maybe_update_proxy_name(orig_value: Any, name: str, is_internal: bool | None = None):
    # Names that we do not re-name proxies into as these are reserved
    proxy_rename_ignore_names = {
        "fn",  # For example, `fn = globals()['__function_obj']` in prologue
        "obj",  # For example, `obj = fn.forward` in prologue
    }

    if is_internal is None:
        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
        frame = runtimectx.peek_frame_stack()
        assert frame is not None  # pass is_internal if you call this before the frame is set up
        is_internal = frame.module in {"thunder.core.interpreter", "thunder.core.jit_ext"}

    uvalue = unwrap(orig_value)

    if (
        isinstance(uvalue, Proxy)
        and (name not in proxy_rename_ignore_names)
        and is_proxy_name_available(name)
        and not is_internal
    ):
        uvalue_var = variableify(uvalue)
        rename_proxy_swapmap = get_jit_ctx()._proxy_swapmap
        if uvalue_var not in rename_proxy_swapmap:
            uvalue_renamed = uvalue.replace_name(name)
            rename_proxy_swapmap[uvalue_var] = uvalue_renamed


def _apply_trace_proxy_rename(
    trace: TraceCtx, rename_proxy_swapmap: None | dict[Variable, Proxy] = None, name: str | None = None
) -> TraceCtx:
    if rename_proxy_swapmap is None:
        rename_proxy_swapmap = get_jit_ctx()._proxy_swapmap

    new_trace = from_trace(trace)

    # Rename args/kwargs {
    def proxy_name_replacer(arg: Any):
        if isinstance(arg, Proxy):
            return rename_proxy_swapmap.get(variableify(arg), arg)
        else:
            return arg

    new_trace.args = tree_map(proxy_name_replacer, new_trace.args)
    new_trace.kwargs = tree_map(proxy_name_replacer, new_trace.kwargs)
    # }

    # Rename proxies in bound symbols {
    new_bsyms = []
    for bsym in trace.bound_symbols:
        new_bsym = bsym.from_bsym_swap_proxies(rename_proxy_swapmap)
        new_bsyms.append(new_bsym)

    new_trace.bound_symbols = new_bsyms
    # }

    # Update signature {
    if name is not None:
        si = SigInfo(name)
        si.args = [(p.name, None) for p in new_trace.args]
        new_trace._siginfo = si
    # }

    return new_trace


# TODO Do we need to warn here? It would find its way in the wrap callback
def _general_jit_global_callback(orig_value: Any, name: str) -> Any:
    _maybe_update_proxy_name(orig_value, name)

    return orig_value


_safe_provenance_inst = {
    "INPUT_ARGS",
    "INPUT_KWARGS",
    "INPUT_FN",
    "LOAD_ATTR",
    "CONSTANT",
    "BINARY_SUBSCR",
}


def should_register_for_prologue(pr):
    inst = pr.inst
    if pr.ext_flag & EXT_FLAG_IS_TENSOR_PROXY:
        return False
    if isinstance(inst, dis.Instruction):
        inst = inst.opname
    else:
        inst = inst.value
    if inst not in _safe_provenance_inst:
        return False
    if inst == "CONSTANT" and callable(pr.value):
        if pr.value.__name__ != "__getitem__" and pr.value != GetSetDescriptorType.__get__:
            return False
    return all(should_register_for_prologue(i) for i in pr.inputs)


def _general_jit_wrap_callback(value):
    ctx: JitCtx = get_jit_ctx()

    uvalue = value.value
    # for modules, rewrite m.__dict__["key"] to m.key
    if (
        value.provenance.inst is PseudoInst.BINARY_SUBSCR
        and value.provenance.inputs[0].inst is PseudoInst.LOAD_ATTR
        and value.provenance.inputs[0].inputs[0].ext_flag & EXT_FLAG_IS_MODULE
        and value.provenance.inputs[0].inputs[1].inst is PseudoInst.CONSTANT
        and value.provenance.inputs[0].inputs[1].value == "__dict__"
    ):
        value.provenance = ProvenanceRecord(
            PseudoInst.LOAD_ATTR,
            inputs=[value.provenance.inputs[0].inputs[0], value.provenance.inputs[1]],
            ext_flag=value.provenance.ext_flag,
        )
    if isinstance(uvalue, torch.nn.Module):
        value.provenance.ext_flag |= EXT_FLAG_IS_MODULE
    elif isinstance(uvalue, torch.Tensor):
        # we always want to proxy torch.Tensor, even const
        p = ctx.proxify(value)
    elif value.provenance.inst is PseudoInst.CONSTANT:
        value.provenance.ext_flag |= EXT_FLAG_IS_PROXY_DERIVED
    elif callable(uvalue):
        value.provenance.ext_flag |= EXT_FLAG_IS_CALLABLE
    elif type(uvalue) in (tuple, list, dict, CellType, ModuleType, set):
        pass  # basic containers are OK, too, subclasses?
    elif isinstance(uvalue, Proxy):
        value.provenance.ext_flag |= EXT_FLAG_IS_PROXY_DERIVED
    elif isinstance(uvalue, (float, int, complex, str, slice, torch.device)) and not isinstance(uvalue, Proxy):
        if value.provenance.ext_flag & EXT_FLAG_IS_PROXY_DERIVED:  # we already have seen this
            pass
        elif should_register_for_prologue(value.provenance):
            value.provenance.ext_flag |= EXT_FLAG_IS_PROXY_DERIVED
            value.provenance.ext_flag |= EXT_FLAG_IS_CONSTRAINABLE_INPUT
            # we follow the caching mechanisms of the eager_unpack_interpreter
            p = ctx.proxify(value)
        else:
            return _general_jit_sharp_edge(
                f"We are using a (non-const) value of type {type(uvalue).__name__}, which is not identified as an input.",
                value,
            )
    else:
        return _general_jit_sharp_edge(
            f"We are using a (non-const) value of unknown type {type(uvalue).__name__}, which may or may not be safe.",
            value,
        )


def _general_jit_load_fast_callback(orig_value: Any, name: str) -> Any:
    _maybe_update_proxy_name(orig_value, name)

    return orig_value


def _general_jit_load_deref_callback(orig_value: Any, name: str) -> Any:
    _maybe_update_proxy_name(orig_value, name)

    return orig_value


def _general_jit_store_deref_callback(
    orig_value: Any, name: str, co_cellsvars: tuple[str], co_freevars: tuple[str]
) -> Any:
    _maybe_update_proxy_name(orig_value, name)

    return orig_value


def _general_jit_store_fast_callback(orig_value: Any, name: str) -> Any:
    _maybe_update_proxy_name(orig_value, name)

    return orig_value


def _general_jit_local_callback(name: str, value: Any, *, module: str) -> None:
    is_internal = module in {"thunder.core.interpreter", "thunder.core.jit_ext"}

    _maybe_update_proxy_name(value, name, is_internal=is_internal)
    return value


general_jit_callbacks: dict[INTERPRETER_CALLBACKS, Callable] = {
    INTERPRETER_CALLBACKS.CONST_CALLBACK: _general_jit_const_callback,
    INTERPRETER_CALLBACKS.GLOBAL_CALLBACK: _general_jit_global_callback,
    INTERPRETER_CALLBACKS.WRAP_CALLBACK: _general_jit_wrap_callback,
    INTERPRETER_CALLBACKS.LOAD_FAST_CALLBACK: _general_jit_load_fast_callback,
    INTERPRETER_CALLBACKS.LOAD_DEREF_CALLBACK: _general_jit_load_deref_callback,
    INTERPRETER_CALLBACKS.STORE_DEREF_CALLBACK: _general_jit_store_deref_callback,
    INTERPRETER_CALLBACKS.STORE_FAST_CALLBACK: _general_jit_store_fast_callback,
    INTERPRETER_CALLBACKS.LOCAL_CALLBACK: _general_jit_local_callback,
}
general_jit_callbacks = default_callbacks | general_jit_callbacks


# This pass identifies NumberProxy that's marked as statically constrained and propagate the constraints to inputs to the trace.
# The logic is that, if all inputs that produces a NumberProxy is marked statically constrained, then the value of the NumberProxy is statically constrained.
# This pass currently only does backward propagation to insert constraints in prologue trace
# TODO: We should be able to apply constant-folding and simplify computation_trace.
# TODO: If we allow symbolic constraints, we would be able to get more cache re-use. i.e. rather than requiring a NumberProxy to be static, we can have a finer grained constraints as `check_number_gt`.
def propagate_constraints(ctx, inputs, intermediates, computation_trace):
    import thunder.core.utils as utils

    # set of NumberProxy variables that has already been traversed and marked as statically constrained.
    static_np_set = set()

    # add static constraints for inputs
    for inp in inputs:
        u_inp = unvariableify(inp)
        if not isinstance(u_inp, NumberProxy):
            continue
        if u_inp.is_static_constrained():
            ctx.add_constraint((clang.check_number_type_and_value, u_inp, u_inp.value))
            static_np_set.add(inp)

    producers = utils.producers(computation_trace.bound_symbols, _map_to_numbers=False)
    # add static constraints propagated from intermediates.
    for intermediate in intermediates:
        u_intermediate = unvariableify(intermediate)
        if not isinstance(u_intermediate, NumberProxy) or not u_intermediate.is_static_constrained():
            continue

        # DFS traversal along producers, starting from seed `intermediate`
        front = [intermediate]
        while len(front) != 0:
            v = front.pop()
            if v in static_np_set:
                continue
            static_np_set.add(v)

            uv = unvariableify(v)
            if v in inputs:
                ctx.add_constraint((clang.check_number_type_and_value, uv, uv.value))
            else:
                producer = producers[uv]
                for inp in producer.flat_proxy_args:
                    if not isinstance(inp, NumberProxy):
                        continue
                    front.append(variableify(inp))


def get_computation_inputs_and_intermediates(computation_trace):
    inputs_list = []
    inputs_set = set()
    intermediates_set = set()

    for bsym in computation_trace.bound_symbols:
        v: Variable
        for v in bsym.flat_variableified_proxy_args:
            if v not in inputs_set and v not in intermediates_set:
                inputs_list.append(v)
                inputs_set.add(v)
        for v in bsym.flat_variableified_proxy_outs:
            intermediates_set.add(v)

    return inputs_list, inputs_set, intermediates_set


def get_parameter_or_buffer_or_submodule_name_and_root(provenance):
    assert provenance.inputs[0].inst is PseudoInst.LOAD_ATTR
    assert provenance.inputs[0].inputs[0].ext_flag & EXT_FLAG_IS_MODULE
    typ = provenance.inputs[0].inputs[1].value
    name = [provenance.inputs[1].value]
    mprovenance = provenance.inputs[0].inputs[0]

    while (
        mprovenance.inst is PseudoInst.BINARY_SUBSCR
        and mprovenance.inputs[1].inst is PseudoInst.CONSTANT
        and mprovenance.inputs[0].inst is PseudoInst.LOAD_ATTR
        and mprovenance.inputs[0].inputs[0].ext_flag & EXT_FLAG_IS_MODULE
    ):
        assert (
            mprovenance.inputs[0].inputs[1].inst is PseudoInst.CONSTANT
            and mprovenance.inputs[0].inputs[1].value == "_modules"
        )

        name_component = mprovenance.inputs[1].value
        name.insert(0, name_component)
        mprovenance = mprovenance.inputs[0].inputs[0]
    return typ, name, mprovenance


def unpack_inputs(ctx, prologue_trace, pro_to_comp_inps, pro_to_epi_inps, args, kwargs):
    already_unpacked: dict[int, Proxy] = {}
    orig_modules: dict[int, Proxy] = {}

    # param_ordering[id(proxy] is a list that contains either finite numbers or (strings preceded by math.inf)
    param_ordering: dict[int, list] = {}

    # Unpacks the inputs in the prologue trace
    # TODO Generate unpacking constraints
    def unpack(v: Variable | Proxy) -> Proxy:
        p: Proxy
        if isinstance(v, Proxy):
            p = v
        else:
            p = v.proxy

        assert p.history is not None, f"{p} has history None"
        if id(p) in already_unpacked:
            return p

        # Adds the name to the prologue trace
        if not prologue_trace.has_name(p.name):
            prologue_trace.add_name(p.name)

        def from_input(provenance, *, new_output=False):
            if provenance.inst == PseudoInst.INPUT_ARGS:
                assert new_output
                param_ordering[id(pro_args_proxy)] = (pro_args_proxy, [0])
                return pro_args_proxy
            elif provenance.inst == PseudoInst.INPUT_KWARGS:
                assert new_output
                param_ordering[id(pro_kwargs_proxy)] = (pro_kwargs_proxy, [1])
                return pro_kwargs_proxy
            elif provenance.inst == PseudoInst.INPUT_FN:
                if provenance.ext_flag & EXT_FLAG_IS_MODULE:
                    name = "module"
                else:
                    name = "fn"
                if new_output:
                    output = Proxy(name=name)
                else:
                    output = p
                param_ordering[id(output)] = (output, [3])
                provenance.proxy = output
                bsym = prims.unpack_function_obj.bind(output, output=output)
                prologue_trace.bound_symbols.append(bsym)
                return output
            assert False

        def from_load_attr(provenance, *, new_output=False):
            obj, name = (from_provenance(i, new_output=True) for i in provenance.inputs)
            orig_obj = obj
            if provenance.inputs[0].ext_flag & EXT_FLAG_IS_MODULE and provenance.ext_flag & EXT_FLAG_IS_CALLABLE:
                obj = orig_modules.get(id(obj), obj)

            if new_output:
                output = Proxy("obj")
            else:
                output = p
            param_ordering[id(output)] = (output, param_ordering[id(orig_obj)][1] + [math.inf, "." + str(name)])
            bsym = prims.unpack_attr.bind(obj, name, output=output)
            prologue_trace.bound_symbols.append(bsym)
            return output

        def from_constant(provenance, *, new_output=False):
            if isinstance(provenance.value, (int, str)):
                return provenance.value
            else:
                raise NotImplementedError(f"constant of type {type(provenance.value)} {provenance.value}")

        def unpack_parameter_or_buffer_or_submodule(provenance, *, new_output=False):
            typ, name, root_module_provenance = get_parameter_or_buffer_or_submodule_name_and_root(provenance)
            root_module = from_provenance(root_module_provenance, new_output=True)
            if new_output:
                output = Proxy("m")  # name? collectify?
            else:
                output = p

            param_ordering[id(output)] = (
                output,
                param_ordering[id(root_module)][1]
                + [
                    i
                    for name_component in name
                    for i in (
                        [int(name_component)] if name_component.isnumeric() else [math.inf, ("." + name_component)]
                    )
                ],
            )

            name = ".".join(name)
            if typ == "_parameters":
                bsym = prims.unpack_parameter.bind(root_module, name, output=output)
                output.tags.add(ProxyTag.STATIC_MEMORY_LOCATION)
            elif typ == "_buffers":
                bsym = prims.unpack_buffer.bind(root_module, name, output=output)
                output.tags.add(ProxyTag.STATIC_MEMORY_LOCATION)
            elif typ == "_modules":
                bsym = prims.unpack_submodule.bind(root_module, name, output=output)
            else:
                assert False
            prologue_trace.bound_symbols.append(bsym)
            return output

        def from_binary_subscr(provenance, *, new_output=False):
            # special case tensors, todo: do this via pattern matching after unpacking?
            if (
                provenance.inst is PseudoInst.BINARY_SUBSCR
                and provenance.inputs[0].inst is PseudoInst.LOAD_ATTR
                and provenance.inputs[0].inputs[1].inst is PseudoInst.CONSTANT
                and provenance.inputs[0].inputs[1].value in {"_parameters", "_buffers", "_modules"}
                and provenance.inputs[0].inputs[0].ext_flag & EXT_FLAG_IS_MODULE
            ):
                return unpack_parameter_or_buffer_or_submodule(provenance, new_output=new_output)

            inputs = [from_provenance(i, new_output=True) for i in provenance.inputs]
            obj, idx = inputs
            if new_output:
                output = Proxy("subscr")  # name? collectify?
            else:
                output = p
            if isinstance(idx, (int, str, Proxy)):
                if isinstance(idx, int):
                    idx = int(idx)
                elif isinstance(idx, str):
                    idx = str(idx)
                param_ordering[id(output)] = (output, param_ordering[id(obj)][1] + [math.inf, "[", idx])
                bsym = prims.unpack_getitem.bind(obj, idx, output=output)
                prologue_trace.bound_symbols.append(bsym)
            else:
                raise NotImplementedError(f"Unpacking from BINARY_SUBSCR with elaborate inputs {inputs=} {provenance}")
            return output

        def from_opaque(provenance, *, new_output=False):
            fn = provenance.inputs[0]
            args = provenance.inputs[1]
            if fn.inst != PseudoInst.CONSTANT:
                raise NotImplementedError(f"unpacking from nonconstant opaque function")
            if fn.value.__name__ == "__getitem__":
                idx, obj = args.inputs
                # This should be solved in the JIT...
                return from_provenance(
                    ProvenanceRecord(PseudoInst.BINARY_SUBSCR, inputs=[obj, idx]), new_output=new_output
                )
            elif fn.value == GetSetDescriptorType.__get__:
                # todo: find a more elegant way?
                # Arg 1 is the object we want to get the attribute from
                # Arg 2 is the GetSetDescriptor, which contains the arrgument name as .__name__
                assert len(args.inputs) == 3
                assert args.inputs[2].inst == PseudoInst.CONSTANT and isinstance(
                    args.inputs[2].value, GetSetDescriptorType
                )
                return from_provenance(
                    ProvenanceRecord(
                        PseudoInst.LOAD_ATTR,
                        inputs=[
                            args.inputs[1],
                            ProvenanceRecord(PseudoInst.CONSTANT, inputs=[], value=args.inputs[2].value.__name__),
                        ],
                    )
                )
            raise NotImplementedError(f"unpacking from OPAQUE {fn.value} {provenance}")

        def from_provenance(provenance, *, new_output=False):
            p = getattr(provenance, "proxy", None)
            if p is not None:
                return p

            inst = provenance.inst
            if isinstance(inst, dis.Instruction):
                inst = inst.opname

            d = {
                "INPUT_ARGS": from_input,
                "INPUT_KWARGS": from_input,
                "INPUT_FN": from_input,
                "LOAD_ATTR": from_load_attr,
                "CONSTANT": from_constant,
                "BINARY_SUBSCR": from_binary_subscr,
                "OPAQUE": from_opaque,
            }

            unpack_fn = d.get(inst)
            if unpack_fn is None:
                raise NotImplementedError(f"Unpacking from {inst} {provenance}")
            res = unpack_fn(provenance, new_output=new_output)

            if provenance.ext_flag & EXT_FLAG_IS_MODULE:
                assert prologue_trace.bound_symbols[-1].output is res
                if prologue_trace.bound_symbols[-1].sym != prims.unpack_submodule:
                    orig_module = Proxy("module")
                    prologue_trace.bound_symbols[-1].output = orig_module
                    bsym = prims.unpack_thunder_module.bind(orig_module, output=res)
                    orig_modules[id(res)] = orig_module
                    prologue_trace.bound_symbols.append(bsym)

            provenance.proxy = res
            return res

        assert isinstance(p.history, ProvenanceRecord), p.history

        # We reset p.history.proxy to make sure from_provenance(p.history) calls unpack_fn on p.history.
        # This is necessary because past recursive unpackings may have unpacked p.history and associated it
        # with a different proxy.
        # For example, if p is a TensorProxy, the history of p.grad is
        #     ProvenanceRecord(LOAD_ATTR, inputs=[p.history, <CONSTANT "grad">])
        # and unpack(p.grad) attaches new proxies to all its sub-histories, including p.history
        p.history.proxy = None

        with tracectx(prologue_trace):
            try:
                from_provenance(p.history)
            except Exception as e:
                raise NotImplementedError(f"Exception occured unpacking object from {p.history}") from e

        already_unpacked[id(p)] = p

        return p

    with tracectx(prologue_trace):
        for n, l in (("args", len(args)), ("kwargs", len(kwargs))):
            output = Proxy(name=n)
            bsym = prims.unpack_trivial.bind(output, output=output, name=n)
            prologue_trace.bound_symbols.append(bsym)
            bsym = prims.check_len.bind(output, l, output=None)
            prologue_trace.bound_symbols.append(bsym)
            if n == "args":
                pro_args_proxy = output
            else:
                assert n == "kwargs"
                pro_kwargs_proxy = output

    def is_variableified_tensorproxy(v: Variable | Proxy) -> Proxy:
        p: Proxy
        if isinstance(v, Proxy):
            p = v
        else:
            p = v.proxy
        return not isinstance(p, TensorProxy)

    # TODO: This is just a WAR to get things working. We'll revisit this when
    # we deal with constraints in prologue trace.
    #
    # We sort variables to before `unpack` to put TensorProxy before others.
    # Because we could have TensorProxy.shape be part of `pro_to_xxx` along with
    # the TensorProxy. If we unpack the shape first, we'll ended up unpack the
    # tensor with a wrong name. e.g. A shape would have a history as:
    #   ProvenanceRecord(
    #     i1 = INPUT_ARGS()
    #     i2 = BINARY_SUBSCR(i1, 0)    # This is the TensorProxy
    #     i3 = LOAD_ATTR(i2, 'shape')
    #     i4 = BINARY_SUBSCR(i3, 1)
    #   )
    pro_to_epi_inps = sorted(pro_to_epi_inps, key=is_variableified_tensorproxy)
    pro_to_comp_inps = sorted(pro_to_comp_inps, key=is_variableified_tensorproxy)

    pro_to_epi = tuple(sorted((unpack(v) for v in pro_to_epi_inps), key=lambda x: param_ordering[id(x)][1]))
    pro_to_comp = tuple(sorted((unpack(v) for v in pro_to_comp_inps), key=lambda x: param_ordering[id(x)][1]))

    with tracectx(prologue_trace):
        for prim, *args in ctx._constraints:
            for a in args:
                if isinstance(a, Proxy):
                    unpack(a)
            # unpacking Proxy in TensorProxy.shape which is used in `check_tensor_shape_and_metadata`
            if prim == clang.check_tensor_shape_and_metadata:
                for s in a.shape:
                    if isinstance(s, Proxy):
                        unpack(s)

            prim(*args)

        cache_info = thunder._get_cache_info()
        # assert len of cache info to ensure that we're not missing anything?
        if cache_info:
            cache_info_p = Proxy(name="cache_info")
            bsym = prims.unpack_cache_info.bind(cache_info_p, output=cache_info_p)
            prologue_trace.bound_symbols.append(bsym)
            for k, v in cache_info.items():
                p = proxy(v, name=f"cache_info_{k}", history=None)
                bsym = prims.unpack_getitem.bind(cache_info_p, k, output=p)
                prologue_trace.bound_symbols.append(bsym)

                if isinstance(v, str):
                    clang.check_string_value(p, v)
                elif isinstance(v, (int, bool, float)):
                    clang.check_number_type_and_value(p, v)
                elif isinstance(v, (torch.dtype, torch.device)):
                    clang.check_literal_like(p, v)
                else:
                    raise NotImplementedError(f"cache info of type {type(v).__name__}")

        prims.python_return((pro_to_comp, pro_to_epi))

    return pro_to_comp, pro_to_epi


def process_recorded_modifications(ctx, epilogue_trace):
    root_for_provenances = {}
    for modified_object, modifications in ctx._additional_outputs.items():
        umodified_object = modified_object.value

        if isinstance(umodified_object, dict):
            last_modification = {}
            for inst, *args in modifications:
                if inst == PseudoInst.STORE_SUBSCR:
                    _, key, value = args
                    # should we warn if we have multiple assignments?
                    last_modification[key.value] = (inst, value)
                else:
                    raise NotImplementedError(f"Modifications {inst} on dicts are not supported")
            for k, (inst, *args) in last_modification.items():
                if inst == PseudoInst.STORE_SUBSCR:
                    (value,) = args
                    assert isinstance(value.value, Proxy)

                    assert modified_object.provenance.inst is PseudoInst.LOAD_ATTR
                    assert modified_object.provenance.inputs[1].inst is PseudoInst.CONSTANT
                    assert modified_object.provenance.inputs[1].value == "_buffers"

                    typ, name, root_module_provenance = get_parameter_or_buffer_or_submodule_name_and_root(
                        modified_object.provenance.inputs[0]
                    )
                    assert typ == "_modules"
                    root_module_proxy = root_for_provenances.get(root_module_provenance)
                    if root_module_proxy is None:
                        ## we want this to created in the compute trace context for namespace...
                        root_module_proxy = Proxy(history=root_module_provenance)
                        epilogue_trace.add_name(root_module_proxy.name)
                        root_for_provenances[root_module_provenance] = root_module_proxy

                    name = ".".join(name + [k])
                    with tracectx(epilogue_trace):
                        bsym = prims.pack_buffer.bind(root_module_proxy, name, value.value, output=None)
                        epilogue_trace.bound_symbols.append(bsym)
                else:
                    raise NotImplementedError(f"Modifications {inst} on dicts are not supported")
        else:
            raise NotImplementedError(f"Modifications of {type(uvalue).__name__} objects are not supported")


def bind_inputs(name, trace, input_vars, input_proxies):
    # restore `scopes` so the unpack below would be appended to the trace
    trace.scopes = [trace.bound_symbols]
    # Unpacks inputs into the computation trace
    # TODO This currently does the unpacks at the end of the trace, then moves them to the beginning, there's
    #   almost certainly a more elegant way to do this
    with tracectx(trace):
        p: Proxy
        for p in input_proxies:
            prims.unpack_trivial(p, name=p.name)

    bsyms = trace.bound_symbols
    trace.bound_symbols = bsyms[-len(input_proxies) :] + bsyms[: -len(input_proxies)]

    si = SigInfo(name)
    si.args = [(v.proxy.name, None) for v in input_vars]
    trace._siginfo = si
    trace.args = input_proxies


def _get_process_group_from(*fn_and_args) -> Optional["ProcessGroup"]:
    # `ddp` and `fsdp` transforms add attribute `procses_group_for_ddp`
    # on the Module that they wrap. This module could be passed to `thunder.jit`
    # as the function to be jitted or as an argument of the function to be jitted.
    found_pg = None
    for fn_or_arg in fn_and_args:
        pg = getattr(fn_or_arg, "process_group_for_ddp", None)
        if pg is not None and found_pg is None:
            found_pg = pg
        elif pg is not None and pg != found_pg:
            raise NotImplementedError("jitting modules with different ProcessGroup is not supported currently.")
    return found_pg


def update_tags(proxy_swapmap: dict[Variable, Proxy]) -> None:
    for old, new in proxy_swapmap.items():
        new.tags.update(unvariableify(old).tags)


DebugOptions.register_option(
    "record_interpreter_history", bool, False, "record interpreter history (use thunder.last_interpreter_log to access)"
)


def thunder_general_jit(
    fn: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    /,
    *,
    sharp_edges: SHARP_EDGES_OPTIONS,
    ad_hoc_executor,
) -> TraceResults:
    # TODO: move into wrap_callback or so
    if isinstance(fn, torch.nn.parallel.DistributedDataParallel):
        raise NotImplementedError(
            f"jitting DistributedDataParallel modules is not supported compile the module and then wrap in DDP"
        )

    co: CACHE_OPTIONS = get_cache_option()
    if co not in {CACHE_OPTIONS.CONSTANT_VALUES, CACHE_OPTIONS.NO_CACHING, CACHE_OPTIONS.SYMBOLIC_VALUES}:
        raise NotImplementedError(f"cache option {co.name} is not supported")

    prologue_trace: TraceCtx = TraceCtx(fn)
    computation_trace: TraceCtx = TraceCtx()
    epilogue_trace: TraceCtx | None = TraceCtx()

    si = SigInfo("prologue")
    si.varargs = ("args", None)
    si.varkwargs = ("kwargs", None)
    prologue_trace._siginfo = si

    compile_data = get_compile_data()
    executor_lookasides = {k: interpreter_needs_wrap(v) for k, v in compile_data.executor_lookasides.items()}

    process_group_for_ddp: Optional["ProcessGroup"] = _get_process_group_from(fn, *args, *kwargs.values())
    ctx: JitCtx = JitCtx(
        prologue_trace,
        computation_trace,
        sharp_edges=sharp_edges,
        process_group_for_ddp=process_group_for_ddp,
        executor_lookasides=executor_lookasides,
        ad_hoc_executor=ad_hoc_executor,
    )
    jfn = interpret(
        fn,
        fn_lookaside=general_jit_lookaside,
        callbacks=general_jit_callbacks,
        with_provenance_tracking=True,
        uncacheable_classes=(torch.Tensor, int, float, str, NoneType),
        record_history=compile_data.debug_options.record_interpreter_history,
    )

    with jit_ctx(ctx):
        with tracectx(computation_trace):
            result = jfn(*args, **kwargs)
            computation_trace.set_current_source_location(None, None)
            process_recorded_modifications(ctx, epilogue_trace)
            last_interpreter_log = jfn._last_interpreter_log
            result_proxies = tuple(p for p in tree_iter(result) if isinstance(p, (TensorProxy, NumberProxy)))
            prims.python_return(result_proxies)
        with tracectx(epilogue_trace):
            prims.python_return(result)

    pro_to_comp, pro_to_comp_set, computation_intermediates = get_computation_inputs_and_intermediates(
        computation_trace
    )
    epilogue_inputs, _, _ = get_computation_inputs_and_intermediates(epilogue_trace)

    comp_to_epi = []
    pro_to_epi = []

    # propagate static constrained intermediates to inputs
    propagate_constraints(ctx, pro_to_comp, computation_intermediates, computation_trace)

    # we want tensors to go through the computation trace even for noops because
    # we may need to propagate gradients etc.
    comp_available = computation_intermediates | pro_to_comp_set
    for i in epilogue_inputs:
        if i in comp_available:
            comp_to_epi.append(i)
        else:
            pro_to_epi.append(i)
    comp_to_epi = tuple(comp_to_epi)
    comp_to_epi_proxies = tuple(v.proxy for v in comp_to_epi)
    pro_to_epi = tuple(pro_to_epi)
    pro_to_comp = tuple(pro_to_comp)

    with tracectx(computation_trace):
        last = computation_trace.bound_symbols.pop(-1)
        assert last.sym.id == prims.PrimIDs.RETURN
        prims.python_return(comp_to_epi_proxies)
    # restore `scopes` so the return would be appended to the trace
    computation_trace.scopes = [computation_trace.bound_symbols]

    pro_to_comp_proxies, pro_to_epi_proxies = unpack_inputs(ctx, prologue_trace, pro_to_comp, pro_to_epi, args, kwargs)

    proxy_order = {id(p): i for i, p in enumerate(pro_to_comp_proxies)}
    pro_to_comp = tuple(sorted(pro_to_comp, key=lambda v: proxy_order[id(v.proxy)]))

    bind_inputs("computation", computation_trace, pro_to_comp, pro_to_comp_proxies)
    for p in pro_to_epi_proxies + comp_to_epi_proxies:
        epilogue_trace.names.add(p.name)
    bind_inputs("epilogue", epilogue_trace, pro_to_epi + comp_to_epi, pro_to_epi_proxies + comp_to_epi_proxies)

    # Returns a new swapmap dictionary which has the keys (ctx._proxy_swapmap.key() & variableify(proxies))
    def restrict_proxy_swapmap(proxies: tuple[Proxy]) -> dict[Variable, Proxy]:
        proxy_swapmap = ctx._proxy_swapmap
        proxy_vars = {variableify(p) for p in proxies}
        common_vars = proxy_swapmap.keys() & proxy_vars
        restricted_proxy_swapmap = {v: proxy_swapmap[v] for v in common_vars}
        return restricted_proxy_swapmap

    # Update prologue trace by renaming proxies which are passed from prologue to the computation trace
    prologue_trace = _apply_trace_proxy_rename(prologue_trace, restrict_proxy_swapmap(pro_to_comp_proxies))

    update_tags(ctx._proxy_swapmap)

    # Update computation trace by renaming proxies which are in the ctx._proxy_swapmap
    computation_trace = _apply_trace_proxy_rename(computation_trace, ctx._proxy_swapmap, "computation")

    # Update epilogue trace by renaming proxies which are passed to the epilogue trace from prologue and computation traces
    epilogue_trace = _apply_trace_proxy_rename(
        epilogue_trace, restrict_proxy_swapmap(pro_to_epi_proxies + comp_to_epi_proxies), "epilogue"
    )

    return TraceResults(prologue_trace, computation_trace, epilogue_trace, last_interpreter_log)
