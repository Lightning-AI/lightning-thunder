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

from thunder.core.compile_data import compile_data_and_stats, get_cache_option, using_symbolic_values, get_compile_data
import thunder.clang as clang
import thunder.core.transforms

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
    UnionType,
)

import torch
from thunder.core.proxies import (
    DDPType,
    proxy,
    Proxy,
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
from thunder.core.options import CACHE_OPTIONS, SHARP_EDGES_OPTIONS
from thunder.core.symbol import Symbol, BoundSymbol, is_traceable

from thunder.extend import Executor
from thunder.common import CompileData, CompileStats
from thunder.core.trace import TraceCtx
from thunder.torch import _torch_to_thunder_function_map
from thunder.clang import _clang_fn_set
from thunder.core.pytree import tree_map
from thunder.core.compile_data import compile_data_and_stats

#
# jit_ext.py implements extensions of thunder's interpreter
#


EXT_FLAG_IS_PROXY_DERIVED = 1
EXT_FLAG_IS_TENSOR_PROXY = 2
EXT_FLAG_IS_MODULE_MEMBER_DICT = 4
MODULE_MEMBER_DICT_ATTRS = {
    "_parameters",
    "_modules",
    "_buffers",
    "__dict__",
}


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

# TODO RC1 Add all symbols + methods
# TODO RC1 Reuse minimal objects in other executors
# TODO RC1 Detect additional sharp edges
#   - inputs that are not function arguments (or their derivatives)
#   - modifying an input
#   - calling a function with a side effect (e.g. randn, print)
# TODO RC1 What kind of error should a sharp edge raise?
# TODO RC1 Improve sharp edges warnings and errors to show the source line
#   See issue "jit: Improve "sharp edges" errors and warnings to show the sharp
#       edge's source location"


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


def _lookaside_sharp_edge(lookaside: Callable, lookaside_name: str):
    def wrapped_lookaside(*args, **kwargs):
        res = _sharp_edge(f"Calling {lookaside_name}()", lookaside)
        if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return res
        else:
            return res(*args, **kwargs)

    return wrapped_lookaside


# Accessing these attributes for these specific types are sharp edges
_attr_sharp_edge_map = {
    "__globals__": (FunctionType,),
}


def _getattr_lookaside_sharp_edge(obj: Any, attr_name: str, *default):
    default_getattr_lookaside = default_lookaside(getattr)
    getattr_res = default_getattr_lookaside(obj, attr_name, *default)

    types_with_name_attr = _attr_sharp_edge_map.get(unwrap(attr_name), ())
    for py_type in types_with_name_attr:
        if isinstance(unwrap(obj), py_type):
            return _sharp_edge(f"Accessing attribute '{attr_name}' of type {py_type}", getattr_res)

    return getattr_res


_minimal_lookaside_map = {
    globals: _lookaside_sharp_edge(globals, "globals"),
    locals: _lookaside_sharp_edge(locals, "locals"),
    vars: _lookaside_sharp_edge(vars, "vars"),
    input: _lookaside_sharp_edge(input, "input"),
    print: _lookaside_sharp_edge(print, "print"),
    getattr: _getattr_lookaside_sharp_edge,
    open: _lookaside_sharp_edge(open, "open"),
}

# Translates actual torch functions to their corresponding thunder functions
_minimal_lookaside_map.update(_torch_to_thunder_function_map)


# NOTE: [SharpEdge - random]
# We want to mark functions from `random` module as Sharp Edges.
# These functions are actually method of a hidden/global `random.Random` object
# which manages the required state. Also, note that we want to allow these methods
# on an user instantiated `random.Random` object.
# So, we need to know the method being called and the object it is bound to, to figure out
# if that call is valid or not (within SharpEdge context).
#
# By the time, we get the function to lookaside in `_minimal_lookaside`,
# we get the function and `self` is passed as the first argument.
# We check if `self` is same as the id of the hidden/global object, in which case,
# we wrap it as a SharpEdge otherwise it is from a user's instance which is ok.
#
# Reference:
# 1. Defintion of random.Random : https://github.com/python/cpython/blob/418e72041349dccdd2bf6ad58643fec3314b1691/Lib/random.py#L110
# 2. Instantiation of global random.Random: https://github.com/python/cpython/blob/418e72041349dccdd2bf6ad58643fec3314b1691/Lib/random.py#L913-L920
_RANDOM_MODULE_DETAILS: dict[str, Any] = {}


# Populate the functions present in `random` module and also get reference to the hidden/global `random.Random`
# object.
# See NOTE: [SharpEdge - random] for more information
def _populate_random_module_details() -> None:
    random_classes_and_members = filter(lambda x: not x.startswith("_"), dir(random))
    random_functions = []
    random_global_obj = None
    for attr in random_classes_and_members:
        random_attr = getattr(random, attr)
        if hasattr(random_attr, "__func__"):
            random_functions.append(random_attr.__func__)

            # `__self__` on method points to bound object.
            if hasattr(random_attr, "__self__"):
                if random_global_obj is None:
                    random_global_obj = random_attr.__self__
                assert random_global_obj == random_attr.__self__

    _RANDOM_MODULE_DETAILS["random_global_obj"] = random_global_obj
    _RANDOM_MODULE_DETAILS["random_functions"] = frozenset(random_functions)


_populate_random_module_details()


# Calling functions from random is a sharp edge.
# See NOTE: [SharpEdge - random] for more information
def _random_module_lookaside(
    lookaside: Callable, args: tuple[Any, ...]
) -> Callable | None | Literal[INTERPRETER_SIGNALS.EXCEPTION_RAISED]:
    # Calls for `random` function will always have the global object or
    # user's `random.Random` object passed as first argument
    # (except for something like `random.Random.__init__` which is ok)
    if len(args) == 0:
        return None

    # These methods are resolved to method of parent `_random.Random` class
    SPECIAL_METHODS = ["getrandbits", "random"]

    # grab the bound object for the passed method.
    bound_obj = args[0]
    # if we can match the bound object to be the global object
    # then any calls to it's method is a sharp edge.
    # NOTE: Use `is` for checking the global object, if we use `==` and bound_obj is a Proxy
    #       then we take a path where `random.Random` global object shouldn't
    #       show up and it errors with found `random.Random` but expected
    #       `TensorProxy` or `NumberProxy`.
    if bound_obj is _RANDOM_MODULE_DETAILS["random_global_obj"]:
        # Sanity assertion.
        if not (lookaside in _RANDOM_MODULE_DETAILS["random_functions"] or lookaside.__name__ in SPECIAL_METHODS):
            return do_raise(AssertionError(f"Found an unexpected function {lookaside} in random."))
        return _lookaside_sharp_edge(lookaside, lookaside_name=f"random.{lookaside.__qualname__}")


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
    elif (random_lookaside := _random_module_lookaside(fn, args)) is not None:
        lookaside = random_lookaside
    else:
        # Falls through to the interpreter's default lookaside
        lookaside = default_lookaside(fn, *args, **kwargs)

    return lookaside


# Minimal callbacks (necessary for sharp edges)


class ThunderSharpEdgeError(RuntimeError):
    """
    Thrown when the user program cannot be safely translated to a thunder program,
    unless using interpretation=INTERPRETATION_OPTIONS.TRANSLATE_PYTHON.
    Such cases are referred to as "sharp edges".
    """

    pass


def _sharp_edge(desc: str, value: Any, /) -> Any | INTERPRETER_SIGNALS:
    sharp_edges: SHARP_EDGES_OPTIONS = get_minimal_ctx().sharp_edges

    s: str = (
        f"{desc} is a sharp edge that cannot be translated to a thunder program unless using interpretation=INTERPRETATION_OPTIONS.TRANSLATE_PYTHON."
    )

    if sharp_edges is SHARP_EDGES_OPTIONS.ERROR:
        return do_raise(ThunderSharpEdgeError(s))

    # Warn and return value anyway
    if sharp_edges is SHARP_EDGES_OPTIONS.WARN:
        warnings.warn(s)

    return value


def _minimal_store_global_callback(orig_value: Any, name: str) -> Any:
    return _sharp_edge(f"Storing a global variable `{name}`", orig_value)


def _minimal_store_deref_callback(orig_value: Any, name: str, co_cellsvars: tuple[str], co_freevars: tuple[str]) -> Any:
    # ShapeEdge Description: Writing a non-local outside of current scope is a SharpEdge.
    # code_obj.co_freevars contains the names of free variables in a function.
    # Free variables are non-local variables defined in an outer function
    # and used in an inner function.
    # So if the STORE_DEREF is updating a variable in `co_freevars`,
    # it means we are mutating a captured variable.
    if name in co_freevars:
        return _sharp_edge(f"Storing into a nonlocal variable `{name}`", orig_value)
    return orig_value


# TODO: we need the builtins for impl functions...
safe_builtins = {id(bi): bi for bi in builtins.__dict__.values()}


def _minimal_global_callback(orig_value: Any, name: str) -> Any:
    value: Any = orig_value

    # Allows loading global modules.
    #   Some global loads, like these, are so essential that they have to be part of any Python program
    #   translation scheme.
    # TODO RC1 Review this check. There may be other types we want to allow. This essentially assumes that
    #   the module is captured at interpretation time, or that global module names will not change for
    #   the lifetime of the program.
    #   We could consider adding a check that the name refers to the same module as it did previously.
    if id(value) in safe_builtins:
        return value

    if not isinstance(value, ModuleType):
        return _sharp_edge("Loading a global that is not a module", value)

    return value


_minimal_callbacks: dict[INTERPRETER_CALLBACKS, Callable] = {
    INTERPRETER_CALLBACKS.STORE_GLOBAL_CALLBACK: _minimal_store_global_callback,
    INTERPRETER_CALLBACKS.GLOBAL_CALLBACK: _minimal_global_callback,
    INTERPRETER_CALLBACKS.STORE_DEREF_CALLBACK: _minimal_store_deref_callback,
}
_minimal_callbacks = default_callbacks | _minimal_callbacks


# TODO RC1 Add debug_log
def minimal_thunder_jit(fn: Callable, /, *, sharp_edges: SHARP_EDGES_OPTIONS) -> Callable:
    ctx: MinimalCtx = MinimalCtx(sharp_edges=sharp_edges)
    jfn = interpret(fn, fn_lookaside=_minimal_lookaside, callbacks=_minimal_callbacks)

    def fn_(*args, **kwargs):
        try:
            tok = set_minimal_ctx(ctx)
            return jfn(*args, **kwargs)
        finally:
            reset_minimal_ctx(tok)

    return fn_


#
# Objects and functions related to the general thunder jit
#


class JITSharpEdgeError(RuntimeError):
    """
    Thrown when the program cannot be safely translated to a thunder program,
    even with interpretation=INTERPRETATION_OPTIONS.TRANSLATE_PYTHON.
    Such cases are referred to as JIT "sharp edges".
    """

    pass


def _general_jit_sharp_edge(desc: str, value: Any, /) -> Any | INTERPRETER_SIGNALS:
    sharp_edges: SHARP_EDGES_OPTIONS = get_minimal_ctx().sharp_edges

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

                if pr.inst == PseudoInst.BINARY_SUBSCR:
                    maybe_module_pr = lhs
                else:
                    # LOAD_ATTR
                    maybe_module_pr = pr

                if maybe_module_pr.ext_flag & EXT_FLAG_IS_MODULE_MEMBER_DICT:
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


class GeneralJitCtx(MinimalCtx):
    def __init__(
        self,
        prologue_trace,
        computation_trace,
        *,
        sharp_edges: SHARP_EDGES_OPTIONS,
        process_group_for_ddp=None,
        executor_lookasides,
    ):
        super().__init__(sharp_edges=sharp_edges)

        self._prologue_trace = prologue_trace
        self._computation_trace: TraceCtx = computation_trace
        self._constraints = []
        self._process_group_for_ddp = process_group_for_ddp
        self._additional_outputs = collections.defaultdict(list)
        self._proxy_swapmap: dict[Variable, Proxy] = {}
        self._executor_lookasides: dict[Callable, Callable] = executor_lookasides

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
        if isinstance(uvalue, Proxy):
            return p
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

            from thunder.core import utils
            from thunder.distributed import get_skip_data_parallel_grad_sync

            no_sync = get_skip_data_parallel_grad_sync()
            compile_data = get_compile_data()
            utils.check(
                not (no_sync and getattr(compile_data, "use_fsdp", False)),
                lambda: "`thunder.distributed.fsdp` does not support `no_sync`",
            )

            if not no_sync and isinstance(p, TensorProxy) and p.ddp_type in (DDPType.REPLICATED, DDPType.FULLY_SHARDED):
                p_new = thunder.distributed.prims.synchronize(p, self._process_group_for_ddp)
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
            elif co not in (CACHE_OPTIONS.SAME_INPUT, CACHE_OPTIONS.NO_CACHING):
                raise NotImplementedError(f"Unsupported cache option {co}")
            return p

        elif isinstance(uvalue, (float, int, complex, str)):
            assert should_register_for_prologue(value.provenance)
            value.provenance.ext_flag |= EXT_FLAG_IS_PROXY_DERIVED
            # we follow the caching mechanisms of the eager_unpack_interpreter
            p = proxy(uvalue, history=value.provenance)
            assert p.history is not None, f"{p.history}, {value.provenance} {type(p)}"

            co: CACHE_OPTIONS = get_cache_option()
            if co is CACHE_OPTIONS.CONSTANT_VALUES:
                if isinstance(uvalue, str):
                    self.add_constraint((clang.check_string_value, p, uvalue))
                else:
                    self.add_constraint((clang.check_number_type_and_value, p, uvalue))
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
        else:
            raise ValueError("cannot proxify value of {type(uvalue).__type} objects")


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


_general_jit_lookaside_map.update(
    {k: ensure_recursive_proxies(interpreter_needs_wrap(v)) for k, v in _torch_to_thunder_function_map.items()}
)


def general_jit_lookaside(diverted_fn):
    def lookaside_wrapper(lookaside):
        _general_jit_lookaside_map[diverted_fn] = lookaside
        return lookaside

    return lookaside_wrapper


# lookaside for getattr. We record the provenance of the attribute but for the core attribute getting, we
# rely on the default JIT getattr lookaside (as returned from default_lookaside)
@general_jit_lookaside(getattr)
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


@general_jit_lookaside(isinstance)
def _general_jit_isinstance_lookaside(obj: Any, cls: type | UnionType | tuple[type | UnionType]):
    uobj = unwrap(obj)
    ucls = unwrap(cls)
    if isinstance(uobj, TensorProxy):
        res = issubclass(torch.Tensor, ucls)
    else:
        res = isinstance(uobj, ucls)

    pr = ProvenanceRecord(
        PseudoInst.LOOKASIDE, inputs=[wrap_const(isinstance).provenance, obj.provenance, cls.provenance]
    )
    return wrap(res, provenance=pr)


@general_jit_lookaside(collections.OrderedDict.__setitem__)
def _general_jit_dict_setitem(d, key, value):
    dict_setitem_lookaside = default_lookaside(collections.OrderedDict.__setitem__)
    assert dict_setitem_lookaside is not None

    if d.provenance.ext_flag & EXT_FLAG_IS_MODULE_MEMBER_DICT:
        ctx: GeneralJitCtx = get_general_jit_ctx()
        if d.original_value is d.nothing:
            ctx.proxify(d)
        ctx._additional_outputs[d].append((PseudoInst.STORE_SUBSCR, d, key, value))

    return dict_setitem_lookaside(d, key, value)


@general_jit_lookaside(setattr)
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
    bool_lookaside = default_lookaside(bool) or bool
    return bool_lookaside(wrapped_x)


_general_jit_lookaside_map[bool] = _general_jit_bool_lookaside

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
        **{fn: interpreter_needs_wrap(la) for fn, la in get_methods_properties(NumberProxy)},
        **{fn: ensure_recursive_proxies(interpreter_needs_wrap(la)) for fn, la in get_methods_properties(TensorProxy)},
        prop_lookaside_helper: prop_lookaside_helper,
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
    Exception.__new__,
    StopIteration.__init__,
}


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
            ctx: GeneralJitCtx = get_general_jit_ctx()
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

    ctx: GeneralJitCtx = get_general_jit_ctx()

    if (executor_lookaside := ctx._executor_lookasides.get(fn, None)) is not None:
        lookaside = executor_lookaside
    elif isinstance(fn, Symbol) or fn in _clang_fn_set:
        # Performs symbol lookasides
        # NOTE Symbols "lookaside" to themselves; this just prevents their internals from being jitted
        # NOTE clang operations are not symbols, but we still prevent their internals from being jitted
        recursively_proxy(*args, **kwargs)
        lookaside = interpreter_needs_wrap(fn)
    elif (general_jit_lookaside := _general_jit_lookaside_map.get(fn, None)) is not None:
        lookaside = general_jit_lookaside
    else:
        # Falls through to the interpreter's default lookaside
        lookaside = default_lookaside(fn, *args, **kwargs)

    if lookaside is None:

        def is_from_torch(fn):
            return hasattr(fn, "__module__") and fn.__module__ and fn.__module__.startswith("torch")

        has_tensor_arg = False
        for a in args:
            if isinstance(a.value, TensorProxy):
                has_tensor_arg = True
                break
            if isinstance(a.value, Sequence):
                if any(isinstance(i, TensorProxy) for i in a.value):
                    has_tensor_arg = True
                    break

        if is_opaque(fn) and is_from_torch(fn) and has_tensor_arg:
            if fn.__module__.startswith("torch._C"):
                return lookaside

            # Torch functions have __name__ defined
            fn_name = f"{fn.__module__}.{fn.__name__}"

            # Probably merge with sharp edges
            calling_opaque_torch_msg = (
                f"Trying to call function {fn_name}, but it is not yet supported. "
                "Please file an issue requesting support. "
                "To find out which operations are not yet recongnized by `thunder.jit`, "
                "please run `examine` as per:\n\n"
                "from thunder.examine import examine\n"
                "examine(<your thunder.jit callable argument>, ...)\n"
            )

            return do_raise(NotImplementedError(calling_opaque_torch_msg))

    return lookaside


#
# general_jit callbacks and callback utilities
#

get_general_jit_ctx = get_minimal_ctx


@contextmanager
def general_jit_ctx(ctx: MinimalCtx):
    token = set_minimal_ctx(ctx)
    try:
        yield
    finally:
        reset_minimal_ctx(token)


def _general_jit_const_callback(value: Any) -> WrappedValue:
    return value


# TODO(nikitaved): maybe call it upon Frame creation
def _maybe_update_proxy_name(orig_value: Any, name: str):
    # Names that we do not re-name proxies into as these are reserved
    proxy_rename_ignore_names = {
        "fn",  # For example, `fn = globals()['__function_obj']` in prologue
        "obj",  # For example, `obj = fn.forward` in prologue
    }

    uvalue = unwrap(orig_value)

    if isinstance(uvalue, Proxy) and (name not in proxy_rename_ignore_names) and is_proxy_name_available(name):
        uvalue_var = variableify(uvalue)
        rename_proxy_swapmap = get_general_jit_ctx()._proxy_swapmap
        if uvalue_var not in rename_proxy_swapmap:
            uvalue_renamed = uvalue.replace_name(name)
            rename_proxy_swapmap[uvalue_var] = uvalue_renamed


def _apply_trace_proxy_rename(
    trace: TraceCtx, rename_proxy_swapmap: None | dict[Variable, Proxy] = None, name: str | None = None
) -> TraceCtx:
    if rename_proxy_swapmap is None:
        rename_proxy_swapmap = get_general_jit_ctx()._proxy_swapmap

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
    ctx: GeneralJitCtx = get_general_jit_ctx()

    uvalue = value.value
    if isinstance(uvalue, torch.Tensor):
        # we always want to proxy torch.Tensor, even const
        p = ctx.proxify(value)
    elif value.provenance.inst is PseudoInst.CONSTANT:
        value.provenance.ext_flag |= EXT_FLAG_IS_PROXY_DERIVED
    elif callable(uvalue):
        pass  # we only care if it is called
    elif type(uvalue) in (tuple, list, dict, CellType, ModuleType, set):
        pass  # basic containers are OK, too, subclasses?
    elif isinstance(uvalue, Proxy):
        value.provenance.ext_flag |= EXT_FLAG_IS_PROXY_DERIVED
    elif isinstance(uvalue, (float, int, complex, str)) and not isinstance(uvalue, Proxy):
        if value.provenance.ext_flag & EXT_FLAG_IS_PROXY_DERIVED:  # we already have seen this
            pass
        elif should_register_for_prologue(value.provenance):
            value.provenance.ext_flag |= EXT_FLAG_IS_PROXY_DERIVED
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


general_jit_callbacks: dict[INTERPRETER_CALLBACKS, Callable] = {
    INTERPRETER_CALLBACKS.CONST_CALLBACK: _general_jit_const_callback,
    INTERPRETER_CALLBACKS.GLOBAL_CALLBACK: _general_jit_global_callback,
    INTERPRETER_CALLBACKS.WRAP_CALLBACK: _general_jit_wrap_callback,
    INTERPRETER_CALLBACKS.LOAD_FAST_CALLBACK: _general_jit_load_fast_callback,
    INTERPRETER_CALLBACKS.LOAD_DEREF_CALLBACK: _general_jit_load_deref_callback,
    INTERPRETER_CALLBACKS.STORE_DEREF_CALLBACK: _general_jit_store_deref_callback,
}
general_jit_callbacks = default_callbacks | general_jit_callbacks


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

    return inputs_list, intermediates_set


def unpack_inputs(ctx, prologue_trace, pro_to_comp_inps, pro_to_epi_inps, args, kwargs, *, has_epilogue: bool):
    already_unpacked: dict[int, Proxy] = {}
    is_pure = True

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
            assert new_output
            if provenance.inst == PseudoInst.INPUT_ARGS:
                param_ordering[id(p)][1].insert(0, 0)
                return pro_args_proxy
            elif provenance.inst == PseudoInst.INPUT_KWARGS:
                param_ordering[id(p)][1].insert(0, 1)
                is_pure = False
                return pro_kwargs_proxy
            elif provenance.inst == PseudoInst.INPUT_FN:
                param_ordering[id(p)][1].insert(0, 3)
                is_pure = False
                name = "fn"
                output = Proxy(name=name)
                provenance.proxy = output
                bsym = prims.unpack_function_obj.bind(output, output=output)
                prologue_trace.bound_symbols.append(bsym)
                return output
            assert False

        def from_load_attr(provenance, *, new_output=False):
            is_pure = False
            inputs = [from_provenance(i, new_output=True) for i in provenance.inputs]
            if new_output:
                output = Proxy("obj")
            else:
                output = p
            param_ordering[id(p)][1][:0] = [math.inf, "." + str(inputs[1])]
            bsym = prims.unpack_attr.bind(inputs[0], inputs[1], output=output)
            prologue_trace.bound_symbols.append(bsym)
            return output

        def from_constant(provenance, *, new_output=False):
            if isinstance(provenance.value, (int, str)):
                return provenance.value
            else:
                raise NotImplementedError(f"constant of type {type(provenance.value)} {provenance.value}")

        def from_binary_subscr(provenance, *, new_output=False):
            inputs = [from_provenance(i, new_output=True) for i in provenance.inputs]
            obj, idx = inputs
            if new_output:
                output = Proxy("subscr")  # name? collectify?
            else:
                output = p
            if isinstance(idx, (int, str)):
                if isinstance(idx, int):
                    idx = int(idx)
                elif isinstance(idx, str):
                    idx = str(idx)
                param_ordering[id(p)][1][:0] = [math.inf, "[" + str(idx) + "]"]
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
            if hasattr(provenance, "proxy"):
                return provenance.proxy  # bind?

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

        assert isinstance(p.history, ProvenanceRecord), p.history
        param_ordering[id(p)] = (p, [])
        with tracectx(prologue_trace):
            try:
                from_provenance(p.history)
            except Exception as e:
                raise NotImplementedError(f"Exception occured unpacking object from {p.history}") from e

        already_unpacked[id(p)] = p

        # Adds cache constraints
        # TODO Consider refactoring these contraints
        # TODO Constrain on rank, device, and dtype
        if isinstance(p, TensorProxy):
            with tracectx(prologue_trace):
                prims.assert_tensor_metadata(p, p.shape, p.device, p.dtype, p.requires_grad)

        return p

    with tracectx(prologue_trace):
        for n, l in (("args", len(args)), ("kwargs", len(kwargs))):
            output = Proxy(name=n)
            bsym = prims.unpack_trivial.bind(output, output=output)
            prologue_trace.bound_symbols.append(bsym)
            bsym = prims.check_len.bind(output, l, output=None)
            prologue_trace.bound_symbols.append(bsym)
            if n == "args":
                pro_args_proxy = output
            else:
                assert n == "kwargs"
                pro_kwargs_proxy = output

    pro_to_epi = tuple(sorted((unpack(v) for v in pro_to_epi_inps), key=lambda x: param_ordering[id(x)][1]))
    pro_to_comp = tuple(sorted((unpack(v) for v in pro_to_comp_inps), key=lambda x: param_ordering[id(x)][1]))

    with tracectx(prologue_trace):
        for prim, *args in ctx._constraints:
            for a in args:
                if isinstance(a, Proxy):
                    unpack(a)
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
                else:
                    raise NotImplementedError(f"cache info of type {type(v).__name__}")

        if has_epilogue:
            prims.python_return((pro_to_comp, pro_to_epi))
        else:
            prims.python_return(pro_to_comp)

    return pro_to_comp, pro_to_epi


def process_recorded_modifications(ctx, epilogue_trace):
    for modified_object, modifications in ctx._additional_outputs.items():
        umodified_object = modified_object.value
        ## we want this to created in the compute trace context for namespace...
        modified_object_proxy = Proxy(history=modified_object.provenance)
        epilogue_trace.add_name(modified_object_proxy.name)

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

                    with tracectx(epilogue_trace):
                        bsym = prims.pack_setitem.bind(modified_object_proxy, k, value.value, output=None)
                        epilogue_trace.bound_symbols.append(bsym)
                else:
                    raise NotImplementedError(f"Modifications {inst} on dicts are not supported")
        else:
            raise NotImplementedError(f"Modifications of {type(uvalue).__name__} objects are not supported")


def bind_inputs(name, trace, input_vars, input_proxies):
    # Unpacks inputs into the computation trace
    # TODO This currently does the unpacks at the end of he trace, then moves them to the beginning, there's
    #   almost certainly a more elegant way to do this
    with tracectx(trace):
        p: Proxy
        for p in input_proxies:
            prims.unpack_trivial(p)

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


def thunder_general_jit(
    fn: Callable, args, kwargs, /, *, sharp_edges: SHARP_EDGES_OPTIONS
) -> tuple[TraceCtx, TraceCtx]:
    # TODO: move into wrap_callback or so
    if isinstance(fn, torch.nn.parallel.DistributedDataParallel):
        raise NotImplementedError(
            f"jitting DistributedDataParallel modules is not supported compile the module and then wrap in DDP"
        )

    co: CACHE_OPTIONS = get_cache_option()
    if co not in {CACHE_OPTIONS.CONSTANT_VALUES, CACHE_OPTIONS.NO_CACHING}:
        raise NotImplementedError(f"Only constant constraints is supported")

    prologue_trace: TraceCtx = TraceCtx(fn)
    computation_trace: TraceCtx = TraceCtx()
    epilogue_trace: TraceCtx = TraceCtx()

    si = SigInfo("prologue")
    si.varargs = ("args", None)
    si.varkwargs = ("kwargs", None)
    prologue_trace._siginfo = si

    compile_data = get_compile_data()
    executor_lookasides = {k: interpreter_needs_wrap(v) for k, v in compile_data.executor_lookasides.items()}

    process_group_for_ddp: Optional["ProcessGroup"] = _get_process_group_from(fn, *args, *kwargs.values())
    ctx: GeneralJitCtx = GeneralJitCtx(
        prologue_trace,
        computation_trace,
        sharp_edges=sharp_edges,
        process_group_for_ddp=process_group_for_ddp,
        executor_lookasides=executor_lookasides,
    )
    jfn = interpret(
        fn,
        fn_lookaside=general_jit_lookaside,
        callbacks=general_jit_callbacks,
        with_provenance_tracking=True,
        uncacheable_classes=(torch.Tensor, int, float, str, NoneType),
    )

    with general_jit_ctx(ctx):
        with tracectx(computation_trace):
            result = jfn(*args, **kwargs)
            prims.python_return(result)
            process_recorded_modifications(ctx, epilogue_trace)

    pro_to_comp, computation_intermediates = get_computation_inputs_and_intermediates(computation_trace)

    epilogue_inputs, _ = get_computation_inputs_and_intermediates(epilogue_trace)

    comp_to_epi = []
    pro_to_epi = []

    for i in epilogue_inputs:
        if i in computation_intermediates:
            comp_to_epi.append(i)
        else:
            pro_to_epi.append(i)
    comp_to_epi = tuple(comp_to_epi)
    comp_to_epi_proxies = tuple(v.proxy for v in comp_to_epi)
    pro_to_epi = tuple(pro_to_epi)

    if epilogue_trace.bound_symbols:
        with tracectx(computation_trace):
            last = computation_trace.bound_symbols.pop(-1)
            assert last.sym.id == prims.PrimIDs.RETURN
            prims.python_return((result, comp_to_epi_proxies))

        with tracectx(epilogue_trace):
            prims.python_return(None)
    else:
        epilogue_trace = None

    pro_to_comp_proxies, pro_to_epi_proxies = unpack_inputs(
        ctx, prologue_trace, pro_to_comp, pro_to_epi, args, kwargs, has_epilogue=epilogue_trace is not None
    )

    proxy_order = {id(p): i for i, p in enumerate(pro_to_comp_proxies)}
    pro_to_comp = tuple(sorted(pro_to_comp, key=lambda v: proxy_order[id(v.proxy)]))

    bind_inputs("computation", computation_trace, pro_to_comp, pro_to_comp_proxies)
    if epilogue_trace:
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

    # Update computation trace by renaming proxies which are in the ctx._proxy_swapmap
    computation_trace = _apply_trace_proxy_rename(computation_trace, ctx._proxy_swapmap, "computation")

    # Update epilogue trace by renaming proxies which are passed to the epilogue trace from prologue and computation traces
    if epilogue_trace:
        epilogue_trace = _apply_trace_proxy_rename(
            epilogue_trace, restrict_proxy_swapmap(pro_to_epi_proxies + comp_to_epi_proxies), "epilogue"
        )

    return prologue_trace, computation_trace, epilogue_trace
