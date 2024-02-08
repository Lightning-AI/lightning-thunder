from typing import Any
from collections.abc import ValuesView, Iterable, Iterator
from collections.abc import Callable, Sequence
import weakref
import random
from functools import partial, wraps
import copy
import contextvars
import warnings
from enum import Enum, auto
from io import StringIO
import time

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
    MethodDescriptorType,
    MethodWrapperType,
    WrapperDescriptorType,
    TracebackType,
    CellType,
    ModuleType,
    CodeType,
    BuiltinFunctionType,
    FunctionType,
    MethodType,
)

import torch
from thunder.core.proxies import (
    proxy,
    Proxy,
    NumberProxy,
    StringProxy,
    TensorProxy,
    make_proxy_name,
    variableify,
    unvariableify,
)
from thunder.core.trace import set_tracectx, reset_tracectx, tracectx
from thunder.core.jit import (
    jit,
    _jit,
    default_callbacks,
    JIT_CALLBACKS,
    JIT_SIGNALS,
    default_opcode_interpreter,
    _default_lookaside_map,
    default_lookaside,
    JITFrame,
    do_raise,
    get_jitcompilectx,
    JitCompileCtx,
    is_opaque,
    Py_NULL,
    member_descriptor,
    WrappedValue,
    unwrap,
    wrap,
)
from thunder.core.langctx import set_langctx, reset_langctx, get_default_langctx
from thunder.core.baseutils import extract_callable_name
from thunder.core.codeutils import get_siginfo, SigInfo
import thunder.core.prims as prims
from thunder.common import transform_for_execution, CACHE_OPTIONS
from thunder.core.symbol import Symbol, BoundSymbol

from thunder.extend import Executor
from thunder.common import CompileData, CompileStats
from thunder.core.trace import TraceCtx
from thunder.torch import _torch_to_thunder_function_map
from thunder.clang import _clang_fn_set
from thunder.core.proxies import proxy, Variable
from thunder.core.pytree import tree_map

#
# jit_ext.py implements extensions of thunder's interpreter
#


#
# Functions and objects related to type properties
#

_atomic_copy_types = {
    type(None),
    type(Ellipsis),
    type(NotImplemented),
    int,
    float,
    bool,
    complex,
    bytes,
    str,
    CodeType,
    type,
    range,
    BuiltinFunctionType,
    weakref.ref,
    property,
}

_immutable_types = {
    type(None),
    type(Ellipsis),
    type(NotImplemented),
    int,
    float,
    bool,
    complex,
    bytes,
    str,
    type,
    range,
    BuiltinFunctionType,
    weakref.ref,
    property,
    FunctionType,
    tuple,
    frozenset,
    slice,
}


def is_immutable(val: Any, /) -> bool:
    return type(val) in _immutable_types


_uncopyable_types = {
    ModuleType,
    contextvars.ContextVar,
}


def is_uncopyable(val: Any, /) -> bool:
    return type(val) in _uncopyable_types


#
# Minimal thunder extension
#
# This extension remaps operations to thunder operations and prevents the interpreter from tracing
#   into symbols
# TODO GTC Add all symbols + methods
# TODO GTC Reuse minimal objects in other executors

_minimal_lookaside_map = {}
_minimal_lookaside_map.update(_torch_to_thunder_function_map)

# Adds proxy methods
# NOTE These methods map to themselves, which prevents the interpreter from looking into them
#   This is OK because these methods are written in a tracing-safe manner, and trying to
#   interpreter their internals is unnecessary and would just add complexity at this time
_minimal_lookaside_map.update(
    {
        NumberProxy.__add__: NumberProxy.__add__,
        NumberProxy.__bool__: NumberProxy.__bool__,  # TODO Review returning a BoolProxy from this
        NumberProxy.__neg__: NumberProxy.__neg__,
        NumberProxy.__sub__: NumberProxy.__sub__,
        TensorProxy.__add__: TensorProxy.__add__,
        TensorProxy.__mul__: TensorProxy.__mul__,
        TensorProxy.__sub__: TensorProxy.__sub__,
    }
)


def _minimal_lookaside(fn, *args, **kwargs) -> None | Callable:
    # Identifies the lookaside
    lookaside: None | Callable
    if isinstance(fn, Symbol) or fn in _clang_fn_set:
        # Performs symbol lookasides
        # NOTE Symbols "lookaside" to themselves; this just prevents their internals from being jitted
        # NOTE clang operations are not symbols, but we still prevent their internals from being jitted
        lookaside = fn
    elif (minimal_lookaside := _minimal_lookaside_map.get(fn, None)) is not None:
        lookaside = minimal_lookaside
    else:
        # Falls through to the interpreter's default lookaside
        lookaside = default_lookaside(fn, *args, **kwargs)

    return lookaside


# TODO GTC Add debug_log
def minimal_thunder_jit(
    fn: Callable,
) -> Callable:
    return jit(fn, fn_lookaside=_minimal_lookaside)


#
# Objects and functions related to the literpreter context
#


class LitCtx:
    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()

        self.fn = fn

        self._prologue_trc: TraceCtx = TraceCtx(fn, using_interpreter=True)
        self._prologue_trc.args = args
        self._prologue_trc.kwargs = kwargs

        self._computation_trc: TraceCtx = TraceCtx(using_interpreter=True)

    @property
    def prologue_trace(self) -> TraceCtx:
        return self._prologue_trc

    @property
    def computation_trace(self) -> TraceCtx:
        return self._computation_trc

    # NOTE All proxies are constructed in the context of the computation trace, and their
    #   names must be added to the prologue trace (this is done when constructing the prologue trace)
    def proxify(self, val: Any, /, *, name: None | str = None, history: tuple, **kwargs) -> Any:
        # NOTE This marker indicates that the local has not yet been created, and so this skips them
        if val is Py_NULL():
            return val

        # Short-circuits if the val is a WrappedValue (in which case it's a constant that doesn't need to be proxied)
        if isinstance(val, WrappedValue):
            return val

        # Short-circuits if val is already a proxy
        # TODO Check for distinct provenances for types that care about that (mutable collections)
        if isinstance(val, Proxy):
            return val

        if isinstance(val, str):
            return proxy(val, name=name, history=history)

        # TODO Add history
        if isinstance(val, torch.Tensor):
            return proxy(val, name=name, history=history)

        return proxy(val, name=name, history=history)


_litctx = contextvars.ContextVar("litctx")


# Sets the phantom ctx
def set_litctx(ctx: LitCtx) -> Any:
    return _litctx.set(ctx)


# Returns the current phantom ctx
def get_litctx() -> LitCtx:
    return _litctx.get()


# Resets the phantom ctx
def reset_litctx(token) -> None:
    _litctx.reset(token)


lit_callbacks: dict[JIT_CALLBACKS, Callable] = {}


def register_lit_callback(key: JIT_CALLBACKS) -> Callable:
    def decorator(fn: Callable):
        assert key not in lit_callbacks
        lit_callbacks[key] = fn
        return fn

    return decorator


#
# lit lookasides
#

# TODO Add all lit operation translations (see https://github.com/Lightning-AI/lightning-thunder/issues/1804)
_lit_lookaside_map = {}
_lit_lookaside_map.update(_torch_to_thunder_function_map)


# lookaside for getattr. We record the provenance of the attribute but for the core attribute getting, we
# rely on the default JIT getattr lookaside (as returned from default_lookaside).
def _lit_getattr_lookaside(obj: Any, name: str, *maybe_default: Any):
    getattr_lookaside = default_lookaside(getattr) or getattr
    res = getattr_lookaside(obj, name, *maybe_default)
    if not isinstance(res, Proxy):
        ctx: LitCtx = get_litctx()
        return ctx.proxify(res, name=name, history=(UNPACK_ACTION.FROM_GETATTR, name, obj))
    return res


_lit_lookaside_map[getattr] = _lit_getattr_lookaside


# TODO Expand on this
def _lit_hasattr_lookaside(obj: Any, name: str):
    hasattr_lookaside = default_lookaside(hasattr) or hasattr
    return hasattr_lookaside(obj, name)


_lit_lookaside_map[hasattr] = _lit_hasattr_lookaside


# We want to record a constraint when we go from proxy -> value here.
# At the same time Python expects to (but we might think to loosen the requirement
# to return a bool for the JIT, return a proxy with origin informaiton and postpone
# recording the constraint to conditional jumps and such.
def _lit_bool_lookaside(x: Any) -> bool | JIT_SIGNALS:
    if isinstance(x, NumberProxy) and (x.value is True or x.value is False):
        # TODO: what if x is from the computational trace?
        lit_ctx = get_litctx()
        prologue_trc = lit_ctx.prologue_trace
        with tracectx(prologue_trc):
            prims.assert_compare(x, "==", x.value)
        return x.value

    if isinstance(x, NumberProxy):
        lit_ctx = get_litctx()
        prologue_trc = lit_ctx.prologue_trace
        res = x.value != 0
        with tracectx(prologue_trc):
            prims.assert_compare(x, "!=" if res else "==", 0)
        return res

    bool_lookaside = default_lookaside(bool) or bool
    return bool_lookaside(x)


_lit_lookaside_map[bool] = _lit_bool_lookaside

# Adds proxy methods
# NOTE These methods map to themselves, which prevents the interpreter from looking into them
#   This is OK because these methods are written in a tracing-safe manner, and trying to
#   interpreter their internals is unnecessary and would just add complexity at this time
_lit_lookaside_map.update(
    {
        NumberProxy.__add__: NumberProxy.__add__,
        NumberProxy.__bool__: NumberProxy.__bool__,  # TODO Review returning a BoolProxy from this
        NumberProxy.__neg__: NumberProxy.__neg__,
        NumberProxy.__sub__: NumberProxy.__sub__,
        TensorProxy.__add__: TensorProxy.__add__,
        TensorProxy.__mul__: TensorProxy.__mul__,
        TensorProxy.__sub__: TensorProxy.__sub__,
    }
)

# TODO Implement safety --- UNSAFE, PERMISSIVE, SAFE
_safe_functions: set = {
    dict.get,  # TODO Review safety of this
    FunctionType.__new__,
    isinstance,
    member_descriptor.__get__,  # TODO Review the safety of this
    MethodDescriptorType.__get__,  # TODO Review the safety of this
    type,
    tuple.__len__,
}


# TODO Document this function (with steps)
def lit_lookaside(fn, *args, **kwargs) -> None | Callable:
    # Identifies the lookaside
    lookaside: None | Callable
    if isinstance(fn, Symbol) or fn in _clang_fn_set:
        # Performs symbol lookasides
        # NOTE Symbols "lookaside" to themselves; this just prevents their internals from being jitted
        # NOTE clang operations are not symbols, but we still prevent their internals from being jitted
        lookaside = fn
    elif (lit_lookaside := _lit_lookaside_map.get(fn, None)) is not None:
        lookaside = lit_lookaside
    else:
        # Falls through to the interpreter's default lookaside
        lookaside = default_lookaside(fn, *args, **kwargs)

    if lookaside is None:
        if is_opaque(fn) and fn not in _safe_functions:
            raise NotImplementedError(
                f"Trying to call opaque function {extract_callable_name(fn)}, but it's unsupported. Please file an issue requesting supporting."
            )
        return None

    # NOTE lookaside is not None
    # Wraps the lookaside to unwrap WrappedValues
    @wraps(lookaside)
    def unwrapper(*args, **kwargs):
        args, kwargs = tree_map(unwrap, (args, kwargs))
        return lookaside(*args, **kwargs)

    return unwrapper


#
# lit callbacks
#


class UNPACK_ACTION(Enum):
    FROM_SIGNATURE = auto()
    FROM_CLOSURE = auto()
    FROM_GETATTR = auto()


def _lit_const_callback(value: Any) -> WrappedValue:
    return wrap(value)


def _lit_freevar_callback(name: str, value: Any, /, *, fn: Callable, idx: int) -> Any:
    if not isinstance(value, Proxy):
        ctx: LitCtx = get_litctx()
        return ctx.proxify(value, name=name, history=(UNPACK_ACTION.FROM_CLOSURE, name, fn, idx))
    return value


# TODO Support additional global loads
def _lit_global_callback(globals_dict: dict, name: str) -> Any:
    # Allows loading the torch module
    value = globals_dict[name]
    if value is torch:
        return value

    raise NotImplementedError(f"Tried to load global {name}, but global loads are currently unsupported")


def _lit_local_callback(name: str, value: Any, /) -> Any:
    ctx: LitCtx = get_litctx()
    return ctx.proxify(value, name=name, history=(UNPACK_ACTION.FROM_SIGNATURE, name))


lit_callbacks: dict[JIT_CALLBACKS, Callable] = {
    JIT_CALLBACKS.CONST_CALLBACK: _lit_const_callback,
    JIT_CALLBACKS.FREEVAR_CALLBACK: _lit_freevar_callback,
    JIT_CALLBACKS.GLOBAL_CALLBACK: _lit_global_callback,
    JIT_CALLBACKS.LOCAL_CALLBACK: _lit_local_callback,
}
lit_callbacks = default_callbacks | lit_callbacks


# TODO Add support for transforms
# TODO Introduce caching
# TODO Support other langctx
def _create_callable(cd: CompileData, cs: CompileStats) -> Callable:
    @wraps(cd.fn)
    def fn_(*args, **kwargs) -> tuple[Any, list[TraceCtx]]:
        cs.last_trace_host_start = time.time_ns()
        cs.calls += 1

        # TODO Implement distinct cache modes
        if cd.cache_option is not CACHE_OPTIONS.NO_CACHING:
            for prologue, computation in cs.interpreter_cache:
                try:
                    inps = prologue(*args, **kwargs)
                    cs.cache_hits += 1
                    return computation(*inps)
                except Exception as ex:
                    pass
            cs.cache_misses += 1

        # Currently executes the program eagerly as a placeholder
        jfn: Callable
        lit_ctx = LitCtx(cd.fn, *args, **kwargs)
        set_litctx(lit_ctx)
        lang = get_default_langctx()
        try:
            lang_tok = set_langctx(lang)
            trace_tok = set_tracectx(lit_ctx.computation_trace)
            cs.last_trace_tracing_start = time.time_ns()
            jfn = jit(cd.fn, fn_lookaside=lit_lookaside, callbacks=lit_callbacks, debug_log=cd.debug_log)
            result = jfn(*args, **kwargs)

            # Translates wrapped values to actual values
            # TODO Review this with collections
            result = tree_map(unwrap, result)

            prims.python_return(result)
            cs.last_trace_tracing_stop = time.time_ns()
        finally:
            reset_tracectx(trace_tok)
            reset_langctx(lang_tok)
            cs.last_interpreted_instructions = jfn._last_interpreted_instructions
            cs.last_interpreted_history = jfn._last_interpreted_history

        # Constructs the prologue
        #   The prologue ...
        #   - Accepts the original function's parameters
        #   - Acquires all inputs to the computation, including closures and globals
        #   - Unpacks all inputs
        #   - Validates that the input is valid for the computational trace it's associated with
        #   - Returns the flattened inputs
        # TODO Validate the inputs in the prologue, currently it just unpacks
        prologue_trc = lit_ctx.prologue_trace
        computation_trc = lit_ctx.computation_trace
        already_unpacked: set[int] = set()
        inps: set[Variable] = set()

        # Identifies inputs to computation trace (by looking for proxies with history)
        bsym: BoundSymbol
        for bsym in lit_ctx.computation_trace.bound_symbols:
            v: Variable
            for v in bsym.flat_variableified_proxy_args:
                if v.proxy.history is not None:
                    inps.add(v)

        # Unpacks the inputs in the prologue trace
        # TODO Generate unpacking constraints
        def unpack(v: Variable) -> Proxy:
            p: Proxy = v.proxy
            assert p.history is not None

            if v in already_unpacked:
                return p

            # Adds the name to the prologue trace
            if not prologue_trc.has_name(p.name):
                prologue_trc.add_name(p.name)

            def from_signature(name: str):
                bsym = prims.unpack_trivial.bind(p, output=p)
                prologue_trc.bound_symbols.append(bsym)

            def from_closure(name: str, fn: Callable, idx: int):
                # if fn is the function being compiled, we need to acquire it,
                # else it will come from our scope and be available
                if fn == cd.fn:
                    bsym_fn = prims.unpack_function_obj.bind(fn, output=fn)
                    prologue_trc.bound_symbols.append(bsym_fn)
                bsym_closure = prims.unpack_attr.bind(fn, "__closure__", output=fn.__closure__)
                prologue_trc.bound_symbols.append(bsym_closure)
                bsym = prims.unpack_attr.bind(fn.__closure__[idx], "cell_contents", output=p)
                prologue_trc.bound_symbols.append(bsym)

            def from_getattr(name: str, obj: Any):
                bsym = prims.unpack_attr.bind(obj, name, output=p)
                prologue_trc.bound_symbols.append(bsym)

            d = {
                UNPACK_ACTION.FROM_SIGNATURE: from_signature,
                UNPACK_ACTION.FROM_CLOSURE: from_closure,
                UNPACK_ACTION.FROM_GETATTR: from_getattr,
            }

            action, *args = p.history
            d[action](*args)
            already_unpacked.add(v)

            # Adds cache constraints
            # TODO Consider refactoring these contraints
            # TODO Constrain on rank, device, and dtype
            if isinstance(p, TensorProxy):
                with tracectx(prologue_trc):
                    prims.assert_tensor_metadata(p, p.shape, p.device, p.dtype, p.requires_grad)

            return p

        v: Variable
        for v in inps:
            unpack(v)

        # Returns the inputs from the prologue trace
        prologue_rvals: tuple[Proxy]
        with tracectx(prologue_trc):
            prologue_rvals = tuple(unvariableify(x) for x in inps)
            prims.python_return(prologue_rvals)

        # Constructs the computation trace's signature
        # TODO Only handles args at the moment
        si = SigInfo("computation")
        si.args = list((p.name, None) for p in prologue_rvals)
        computation_trc._siginfo = si
        computation_trc.args = prologue_rvals

        # Unpacks inputs into the computation trace
        # TODO This currently does the unpacks at the end of he trace, then moves them to the beginning, there's
        #   almost certainly a more elegant way to do this
        with tracectx(computation_trc):
            p: Proxy
            for p in prologue_rvals:
                prims.unpack_trivial(p)

        bsyms = computation_trc.bound_symbols
        computation_trc.bound_symbols = bsyms[-len(prologue_rvals) :] + bsyms[: -len(prologue_rvals)]

        # TODO Apply transforms like grad

        extraces = transform_for_execution(
            computation_trc,
            executors_list=cd.executors_list,
        )

        extrace = extraces[-1]

        pro = prologue_trc.python_callable()
        c = extrace.python_callable()

        # Executes the traced program
        cs.last_trace_host_execution_start = time.time_ns()
        computation_result = c(*pro(*args, **kwargs))
        cs.last_trace_host_execution_stop = time.time_ns()

        # Updates the cache
        if cd.cache_option is not CACHE_OPTIONS.NO_CACHING:
            cs.interpreter_cache.append((pro, c))

        # Updates metadata
        # TODO What should the last_traces be in this case?
        cs.last_traces = extraces
        # TODO What should the last executed be in this case?
        cs.last_executed = c
        cs.last_prologue = prologue_trc

        cs.last_trace_host_stop = time.time_ns()
        return computation_result

    fn_._lc_cd = cd
    fn_._lc_cs = cs
    return fn_


# TODO Support recursive litjiting
# NOTE This is an analogue to lit.compile, because how it handles trace generation
#   is sufficiently distinct that merging the two would be quite tricky
def litjit(
    fn: Callable,
    /,
    executors_list: None | Sequence[Executor] = None,
    debug_log: None | StringIO = None,
    cache_option: None | str | CACHE_OPTIONS = None,
) -> Callable:
    cd = CompileData(
        fn=fn,
        langctx=None,
        executors_list=executors_list,
        cache_option=cache_option,
        use_cudagraphs=False,
        use_torch_compile=False,
        disable_torch_autograd_support=True,
        use_rematerialization=False,
        only_execute_prims=False,
        disable_preprocessing=True,
        debug_log=debug_log,
    )

    cs = CompileStats()
    fn_ = _create_callable(cd, cs)
    return fn_
