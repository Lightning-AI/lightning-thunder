import thunder
from typing import Any
from collections.abc import ValuesView, Iterable, Iterator
from collections.abc import Callable, Sequence
import weakref
import random
from functools import partial, wraps
import copy
import contextvars
import dis
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
    GetSetDescriptorType,
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
    _jit_no_unwrap,
    CapsuleType,
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
    WRAPPED_VALUE_TYPE,
    unwrap,
    wrap,
    wrap_const,
    PseudoInst,
    ProvenanceRecord,
    jit_needs_wrap,
)
from thunder.core.langctxs import set_langctx, reset_langctx, Languages, resolve_language
from thunder.core.baseutils import extract_callable_name
from thunder.core.codeutils import get_siginfo, SigInfo
import thunder.core.prims as prims
from thunder.common import transform_for_execution
from thunder.core.options import CACHE_OPTIONS, SHARP_EDGES_OPTIONS
from thunder.core.symbol import Symbol, BoundSymbol, is_traceable

from thunder.extend import Executor
from thunder.common import CompileData, CompileStats
from thunder.core.trace import TraceCtx
from thunder.torch import _torch_to_thunder_function_map
from thunder.clang import _clang_fn_set
from thunder.core.proxies import proxy, Variable
from thunder.core.pytree import tree_map
from thunder.core.compile_data import compile_data_and_stats

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
# This extension supports detecting and warning or erroring on "sharp edges" -- behavior in the
#   original Python program that cannot be translated to the thunder program

# TODO GTC Add all symbols + methods
# TODO GTC Reuse minimal objects in other executors
# TODO GTC Detect additional sharp edges
#   - inputs that are not function arguments (or their derivatives)
#   - modifying an input
#   - calling a function with a side effect (e.g. randn, print)
# TODO GTC What kind of error should a sharp edge raise?
# TODO GTC Improve sharp edges warnings and errors to show the source line
#   https://github.com/Lightning-AI/lightning-thunder/issues/2099


# Context for the minimal interpreter
class MinimalCtx:
    def __init__(self, *, sharp_edges: SHARP_EDGES_OPTIONS):
        self._sharp_edges: SHARP_EDGES_OPTIONS = sharp_edges

    @property
    def sharp_edges(self) -> SHARP_EDGES_OPTIONS:
        return self._sharp_edges


_minimal_ctx = contextvars.ContextVar("minimalctx")


def set_minimal_ctx(ctx: MinimalCtx) -> Any:
    return _minimal_ctx.set(ctx)


def get_minimal_ctx() -> MinimalCtx:
    return _minimal_ctx.get()


def reset_minimal_ctx(token) -> None:
    _minimal_ctx.reset(token)


# Minimal lookasides

_minimal_lookaside_map = {}

# Translates actual torch functions to their corresponding thunder functions
_minimal_lookaside_map.update(_torch_to_thunder_function_map)


def _minimal_lookaside(fn, *args, **kwargs) -> None | Callable:
    # Identifies the lookaside
    lookaside: None | Callable
    if is_traceable(fn):
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


# Minimal callbacks (necessary for sharp edges)


def _sharp_edge(desc: str, /) -> None:
    sharp_edges: SHARP_EDGES_OPTIONS = get_minimal_ctx().sharp_edges

    s: str = f"{desc} is a sharp edge that cannot be translated to a thunder program unless using interpretation=INTERPRETATION_OPTIONS.TRANSLATE_EVERYTHING."

    if sharp_edges is SHARP_EDGES_OPTIONS.ERROR:
        raise AssertionError(s)

    if sharp_edges is SHARP_EDGES_OPTIONS.WARN:
        warnings.warn(s)


def _minimal_global_callback(globals_dict: dict, name: str) -> Any:
    value: Any = globals_dict[name]

    # Allows loading global modules.
    #   Some global loads, like these, are so essential that they have to be part of any Python program
    #   translation scheme.
    # TODO GTC Review this check. There may be other types we want to allow. This essentially assumes that
    #   the module is captured at interpretation time, or that global module names will not change for
    #   the lifetime of the program.
    #   We could consider adding a check that the name refers to the same module as it did previously.
    if not isinstance(value, ModuleType):
        _sharp_edge("Loading a global that is not a module")

    return value


_minimal_callbacks: dict[JIT_CALLBACKS, Callable] = {
    JIT_CALLBACKS.GLOBAL_CALLBACK: _minimal_global_callback,
}
_minimal_callbacks = default_callbacks | _minimal_callbacks


# TODO GTC Add debug_log
def minimal_thunder_jit(fn: Callable, /, *, sharp_edges: SHARP_EDGES_OPTIONS) -> Callable:
    ctx: MinimalCtx = MinimalCtx(sharp_edges=sharp_edges)
    jfn = jit(fn, fn_lookaside=_minimal_lookaside, callbacks=_minimal_callbacks)

    def fn_(*args, **kwargs):
        try:
            tok = set_minimal_ctx(ctx)
            return jfn(*args, **kwargs)
        finally:
            reset_minimal_ctx(tok)

    return fn_


#
# Objects and functions related to the literpreter context
#


class LitCtx:
    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()

        def fn_(*args, **kwargs):
            self.fn(*args, **kwargs)

        self.fn = fn

        self._prologue_trc: TraceCtx = TraceCtx(fn)
        self._prologue_trc._siginfo = get_siginfo(fn_, args, kwargs)
        self._prologue_trc.args = args
        self._prologue_trc.kwargs = kwargs
        self._constraints = []

        self._computation_trc: TraceCtx = TraceCtx(prologue=self._prologue_trc)

    @property
    def prologue_trace(self) -> TraceCtx:
        return self._prologue_trc

    @property
    def computation_trace(self) -> TraceCtx:
        return self._computation_trc

    def add_comparison_constraint(self, lhs, typ, rhs):
        self._constraints.append((typ, lhs, rhs))

    @property
    def comparison_constraints(self) -> list:
        return self._constraints

    # NOTE All proxies are constructed in the context of the computation trace, and their
    #   names must be added to the prologue trace (this is done when constructing the prologue trace)
    def proxify(self, val: Any, /, *, name: None | str = None, history: tuple, **kwargs) -> Any:
        # NOTE This marker indicates that the local has not yet been created, and so this skips them
        if val is Py_NULL():
            return val

        # Short-circuits if the val is a WrappedValue (in which case it's a constant that doesn't need to be proxied)
        if isinstance(val, WrappedValue) and val.typ == WRAPPED_VALUE_TYPE.CONSTANT:
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


def set_litctx(ctx: LitCtx) -> Any:
    return _litctx.set(ctx)


def get_litctx() -> LitCtx:
    return _litctx.get()


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

_lit_lookaside_map.update({k: jit_needs_wrap(v) for k, v in _torch_to_thunder_function_map.items()})


# lookaside for getattr. We record the provenance of the attribute but for the core attribute getting, we
# rely on the default JIT getattr lookaside (as returned from default_lookaside)


def _lit_getattr_lookaside(obj: Any, name: str, *maybe_default: Any):
    getattr_lookaside = default_lookaside(getattr)
    assert getattr_lookaside is not None

    value = getattr_lookaside(obj, name, *maybe_default)
    if value is JIT_SIGNALS.EXCEPTION_RAISED:
        return value

    assert isinstance(value, WrappedValue)
    assert isinstance(name, WrappedValue)

    if not isinstance(value.value, Proxy):
        ctx: LitCtx = get_litctx()
        p = ctx.proxify(value.value, name=name.value, history=(UNPACK_ACTION.FROM_PROVENANCE, value.provenance))
        if value.value is not p:
            value.value = p
            # this does not work yet:
            # res = _jit_no_unwrap(setattr, obj, name, value)
            # if isinstance(res, JIT_SIGNALS):
            #    return res

        return value

    return value


_lit_lookaside_map[getattr] = _lit_getattr_lookaside


# TODO Expand on this
@jit_needs_wrap
def _lit_hasattr_lookaside(obj: Any, name: str):
    hasattr_lookaside = default_lookaside(hasattr) or hasattr
    return hasattr_lookaside(obj, name)


_lit_lookaside_map[hasattr] = _lit_hasattr_lookaside


# We want to record a constraint when we go from proxy -> value here.
# At the same time Python expects to (but we might think to loosen the requirement
# to return a bool for the JIT, return a proxy with origin informaiton and postpone
# recording the constraint to conditional jumps and such.
def _lit_bool_lookaside(wrapped_x: Any) -> bool | JIT_SIGNALS:
    assert isinstance(wrapped_x, WrappedValue)
    x = unwrap(wrapped_x)
    if isinstance(x, NumberProxy) and (x.value is True or x.value is False):
        # TODO: what if x is from the computational trace?
        lit_ctx = get_litctx()
        lit_ctx.add_comparison_constraint(x, "==", x.value)
        return wrap_const(x.value)

    if isinstance(x, NumberProxy):
        lit_ctx = get_litctx()
        res = x.value != 0
        lit_ctx.add_comparison_constraint(x, "!=" if res else "==", 0)
        return wrap_const(res)

    bool_lookaside = default_lookaside(bool) or bool
    return bool_lookaside(wrapped_x)


_lit_lookaside_map[bool] = _lit_bool_lookaside

# Adds proxy methods
# NOTE These methods map to themselves, which prevents the interpreter from looking into them
#   This is OK because these methods are written in a tracing-safe manner, and trying to
#   interpreter their internals is unnecessary and would just add complexity at this time


def get_methods_properties(typ):
    for meth_name in dir(typ):
        meth = getattr(typ, meth_name)
        if isinstance(meth, (MethodType, BuiltinMethodType, MethodDescriptorType, WrapperDescriptorType)) and (
            getattr(meth, "__objclass__", None) == typ or (getattr(meth, "__self__", None) == typ)
        ):
            yield meth
        elif isinstance(meth, FunctionType):
            yield meth  # __getattr__
        elif isinstance(meth, property):
            if meth.fget is not None:
                yield meth.fget
            if meth.fset is not None:
                yield meth.fset
            if meth.fdel is not None:
                yield meth.fdel


_lit_lookaside_map.update(
    {
        **{fn: jit_needs_wrap(fn) for fn in get_methods_properties(NumberProxy)},
        **{fn: jit_needs_wrap(fn) for fn in get_methods_properties(TensorProxy)},
        NumberProxy.__add__: jit_needs_wrap(NumberProxy.__add__),
        NumberProxy.__bool__: jit_needs_wrap(NumberProxy.__bool__),  # TODO Review returning a BoolProxy from this
        NumberProxy.__neg__: jit_needs_wrap(NumberProxy.__neg__),
        NumberProxy.__sub__: jit_needs_wrap(NumberProxy.__sub__),
        NumberProxy.__floordiv__: jit_needs_wrap(NumberProxy.__floordiv__),
        NumberProxy.__le__: jit_needs_wrap(NumberProxy.__ge__),
        NumberProxy.__ge__: jit_needs_wrap(NumberProxy.__le__),
        TensorProxy.__add__: jit_needs_wrap(TensorProxy.__add__),
        TensorProxy.__mul__: jit_needs_wrap(TensorProxy.__mul__),
        TensorProxy.__sub__: jit_needs_wrap(TensorProxy.__sub__),
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
    tuple.__getitem__,
    FunctionType.__get__,  # TODO: review safety
    torch._C._get_tracing_state,  # TODO: review safety
    object.__new__,
    object.__init__,
    callable,
    NoneType.__bool__,
    dict.__len__,
    dict.__contains__,
    dict.__getitem__,
    contextvars.ContextVar.get,
    type.__or__,
    list.__new__,
    list.__init__,
    list.__getitem__,
    reversed.__new__,
    CellType.__new__,
    GetSetDescriptorType.__get__,
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
        needs_wrap = getattr(lookaside, "__jit_needs_wrap", False)
        if needs_wrap:
            args, kwargs = tree_map(unwrap, (args, kwargs))
        return lookaside(*args, **kwargs)

    return unwrapper


#
# lit callbacks
#


# TODO: remove this field from history and just use provenance records?
class UNPACK_ACTION(Enum):
    FROM_PROVENANCE = auto()


def _lit_const_callback(value: Any) -> WrappedValue:
    return value


def _lit_freevar_callback(name: str, wrapped_cell: Any, /, *, fn: Callable, idx: int) -> Any:
    assert isinstance(wrapped_cell, WrappedValue)
    cell = wrapped_cell.value

    if cell == CellType():
        return wrapped_cell

    contents = cell.cell_contents
    if isinstance(contents, Proxy):
        return wrapped_cell

    ctx: LitCtx = get_litctx()

    provenance = ProvenanceRecord(
        PseudoInst.LOAD_ATTR, inputs=[wrapped_cell.provenance, wrap_const("cell_contents").provenance]
    )
    proxy = ctx.proxify(contents, name=name, history=(UNPACK_ACTION.FROM_PROVENANCE, provenance))

    if proxy is not contents:
        # TODO replacing cells is EVIL!, but we do not want to leak proxy, so we would need a cell proxy that diverts the write.
        wrapped_proxy = wrap(proxy, typ=WRAPPED_VALUE_TYPE.INTERMEDIATE, provenance=provenance)
        wrapped_cell.value = CellType(proxy)
        wrapped_cell.attribute_wrappers["cell_contents"] = wrapped_proxy
        # this would leak proxies:
        # cell.cell_contents = wrap(proxy, typ=WRAPPED_VALUE_TYPE.INTERMEDIATE, provenance=provenance)

    return wrapped_cell


# TODO Support additional global loads
def _lit_global_callback(globals_dict: dict, name: str) -> Any:
    # Allows loading the torch module
    value = globals_dict[name]
    if (
        value is torch
        or (value is torch.nn.modules.module._global_backward_pre_hooks)
        or (value is torch.nn.modules.module._global_backward_hooks)
        or (value is torch.nn.modules.module._global_forward_hooks)
        or (value is torch.nn.modules.module._global_forward_pre_hooks)
        or (value is torch.nn.functional)
        or (value is thunder.core.proxies.get_langctx)
    ):
        return value

    raise NotImplementedError(f"Tried to load global {name}, but global loads are currently unsupported")
    return value


def _lit_local_callback(name: str, value: Any, /) -> Any:
    ctx: LitCtx = get_litctx()
    if isinstance(value, WrappedValue) and value.typ == WRAPPED_VALUE_TYPE.CONSTANT:
        return value

    if not isinstance(value, WrappedValue):
        # TODO: consider making the an error once we wrap everything
        value = wrap_const(value)

    assert isinstance(value, WrappedValue)
    provenance = value.provenance
    return wrap(
        ctx.proxify(value.value, name=name, history=(UNPACK_ACTION.FROM_PROVENANCE, provenance)),
        provenance=provenance,
        typ=value.typ,
    )


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
        with compile_data_and_stats(cd, cs):
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
            lang = resolve_language(Languages.TORCH)
            try:
                lang_tok = set_langctx(lang)
                trace_tok = set_tracectx(lit_ctx.computation_trace)
                cs.last_trace_tracing_start = time.time_ns()
                jfn = jit(
                    cd.fn,
                    fn_lookaside=lit_lookaside,
                    callbacks=lit_callbacks,
                    debug_log=cd.debug_log,
                    with_provenance_tracking=True,
                    uncacheable_classes=(torch.Tensor, int, float, str),
                )
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
            already_unpacked: dict[int, Proxy] = {}
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
            def unpack(v: Variable | Proxy) -> Proxy:
                p: Proxy
                if isinstance(v, Proxy):
                    p = v
                else:
                    p = v.proxy

                assert p.history is not None
                if id(p) in already_unpacked:
                    return p

                # Adds the name to the prologue trace
                if not prologue_trc.has_name(p.name):
                    prologue_trc.add_name(p.name)

                def from_input(provenance, *, new_output=False):
                    if new_output:
                        if provenance.inst == PseudoInst.INPUT_ARGS:
                            name = "args"
                        elif provenance.inst == PseudoInst.INPUT_KWARGS:
                            name = "kwargs"
                        elif provenance.inst == PseudoInst.INPUT_FN:
                            name = "fn"

                        output = Proxy(name=name)
                        provenance.proxy = output
                    else:
                        output = p
                        provenance.proxy = output
                    if provenance.inst == PseudoInst.INPUT_FN:
                        bsym = prims.unpack_function_obj.bind(output, output=output)
                    else:
                        bsym = prims.unpack_trivial.bind(output, output=output)
                    prologue_trc.bound_symbols.append(bsym)
                    return output

                def from_load_attr(provenance, *, new_output=False):
                    inputs = [from_provenance(i, new_output=True) for i in provenance.inputs]
                    if new_output:
                        output = Proxy("obj")
                    else:
                        output = p
                    bsym = prims.unpack_attr.bind(inputs[0], inputs[1], output=output)
                    prologue_trc.bound_symbols.append(bsym)
                    return output

                def from_constant(provenance, *, new_output=False):
                    if isinstance(provenance.value, (int, str)):
                        return provenance.value
                    else:
                        raise NotImplementedError(f"constant of type {type(provenance.value)} {provenance.value}")

                def from_binary_subscr(provenance, *, new_output=False):
                    inputs = [from_provenance(i, new_output=True) for i in provenance.inputs]
                    idx, obj = inputs
                    if new_output:
                        output = Proxy("subscr")  # name? collectify?
                    else:
                        output = p
                    if isinstance(idx, (int, str)):
                        if isinstance(idx, int):
                            idx = int(idx)
                        elif isinstance(idx, str):
                            idx = str(idx)
                        bsym = prims.unpack_getitem.bind(obj, idx, output=output)
                        prologue_trc.bound_symbols.append(bsym)
                    else:
                        raise NotImplementedError(
                            f"Unpacking from BINARY_SUBSCR with elaborate inputs {inputs=} {provenance}"
                        )
                    return output

                def from_opaque(provenance, *, new_output=False):
                    fn = provenance.inputs[0]
                    args = provenance.inputs[1]
                    if fn.inst != PseudoInst.CONSTANT:
                        raise NotImplementedError(f"unpacking from nonconstant opaque function")
                    if fn.value.__name__ == "__getitem__":
                        idx, obj = args.inputs
                        return from_provenance(
                            ProvenanceRecord(PseudoInst.BINARY_SUBSCR, inputs=[idx, obj]), new_output=new_output
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
                                    ProvenanceRecord(
                                        PseudoInst.CONSTANT, inputs=[], value=args.inputs[2].value.__name__
                                    ),
                                ],
                            )
                        )
                    raise NotImplementedError(f"unpacking from OPAQUE {fn.value} {provenance}")

                def from_provenance(provenance, *, new_output=False):
                    if hasattr(provenance, "proxy"):
                        return provenance.proxy  # bind?

                    def collect_inst(pr):
                        inst = pr.inst
                        if isinstance(inst, dis.Instruction):
                            inst = inst.opname
                        else:
                            inst = inst.value
                        res = {inst}
                        for i in pr.inputs:
                            res |= collect_inst(i)
                        return res

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
                    provenance.proxy = res
                    return res

                action, *args = p.history
                assert action is UNPACK_ACTION.FROM_PROVENANCE
                with tracectx(prologue_trc):
                    from_provenance(*args)
                already_unpacked[id(p)] = p

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
            for typ, lhs, rhs in lit_ctx.comparison_constraints:
                unpack(lhs)
                with tracectx(prologue_trc):
                    prims.assert_compare(lhs, typ, rhs)

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
