from typing import Any
from collections.abc import ValuesView
from types import ModuleType
from collections.abc import Callable, Sequence

from functools import partial, wraps
import copy
import contextvars
import warnings

import torch
from thunder.core.proxies import proxy, Proxy, TensorProxy
from thunder.core.trace import set_tracectx, reset_tracectx
from thunder.core.jit import (
    jit,
    default_callbacks,
    JIT_CALLBACKS,
    default_opcode_interpreter,
    _default_lookaside_map,
    default_lookaside,
    JITFrame,
    do_raise,
)
from thunder.core.langctx import set_langctx, reset_langctx, get_default_langctx
from thunder.core.codeutils import get_siginfo

#
# jit_ext.py implements extensions of thunder's interpreter
#

#
# phantom mode (no side effects)
#


# TODO Track deleted to deal with duplicate names
# TODO Make proxy construction extensible
# TODO Probably don't want to track what's stored or proxied based on name alone
# TODO The current stored/proxify relationship is far from correct
class PhantomInterpreterRuntimeCtx:
    def __init__(self):
        # Tracks stores to the localsplus list of JITFrame objects
        self.localsplus_stores: set = set()

        # Maps from ids to input objects
        # NOTE This extends the lifetime of all inputs to be at least the lifetime of the interpreter,
        #   this prevents Python from reusing the id of one of the inputs, which is how we track them
        # NOTE This means that the proxies of inputs have lifetimes that are at least the
        #   lifetime of the interpreter, too
        self.input_map: dict[int, Any] = {}

        # Maps from the ids of input objects to their corresponding proxy objects
        # NOTE This dict is compatible with copy.deepcopy()'s memo dict
        self.input_proxy_map: dict[int, Any] = {}

        # ids of all proxies
        self.proxy_id_set: set[int] = set()

        # Maps from lookups into global dicts to proxied values
        # NOTE Handling global dicts are probably a good use case for a dict proxy object
        # NOTE This assumes that global dicts persist for the lifetime our interpreter
        #   (in fact, we're going to assume that they persist for the lifetime of the
        #   Python interpreter)
        self.global_lookups: dict[tuple[int, str], Any] = {}

        # Records deleted values
        self.global_deletions: set[tuple[int, str]] = set()

    @property
    def inputs(self) -> ValuesView[tuple[str, Any]]:
        return self.input_map.values()

    # TODO Extend to consider other immutable types
    def is_immutable(self, val: Any) -> bool:
        if isinstance(val, int):
            return True

        return False

    # Returns the object's proxy (itself, if it is a proxy)
    def proxify(self, name: str, val: Any, /) -> Any:
        val_id = id(val)

        # Checks if val itself is a proxy
        if val_id in self.proxy_id_set:
            return val

        # Checks to see if val is associated with an existing proxy
        p: None | Any = self.input_proxy_map.get(val_id, None)
        if p is not None:
            return p

        # Adds the object to the input map to ensure it lives as long as the interpreter does
        self.input_map[val_id] = (name, val)

        # Immutable objects are there own proxies
        if self.is_immutable(val):
            return val

        # Copies the input_proxy_map because deepcopy might mutate it but then fail, and
        #   if the deepcopy fails then we don't want to modify the input_proxy_map
        memo = copy.copy(self.input_proxy_map)

        try:
            p = copy.deepcopy(val, memo=memo)
        except TypeError as te:
            warnings.warn(f"Couldn't proxy {name} of type {type(val)}; modifications to it will not be prevented")
            return val

        # Updates the proxy id set
        self.proxy_id_set.add(id(p))

        # Some objects, like types, return themselves when deepcopied
        if p is val:
            warnings.warn(f"Couldn't proxy {name} of type {type(val)}; modifications to it will not be prevented")
            return val

        # Removes the id(memo) entry (if deepcopy added it) and updates input_proxy_map with the new proxies
        # NOTE deepcopy can add an id(memo) entry to store references to objects whose lifetimes it wants to
        #   extend until the deepcopy finishes
        memo_id = id(memo)
        memo.pop(memo_id, None)
        self.input_proxy_map = memo

        return p


_phantomctx = contextvars.ContextVar("phantomctx")


# Sets the phantom ctx
def set_phantomctx(ctx: PhantomInterpreterRuntimeCtx) -> Any:
    return _phantomctx.set(ctx)


# Returns the current phantom ctx
def get_phantomctx() -> PhantomInterpreterRuntimeCtx:
    return _phantomctx.get()


# Resets the phantom ctx
def reset_phantomctx(token) -> None:
    _phantomctx.reset(token)


phantom_callbacks: dict[JIT_CALLBACKS, Callable] = {}


# TODO Handle deleting globals
# TODO Use something like inspect.getmodule() and vars() to acquire the module of these globals, and then
#   to acquire its globals in the future
# Handles global loads and stores, essentially keeping an additional dictionary
#   that tracks modifications to all the global dictionaries
def _load_global_callback(globals_dict: dict, name: str, /) -> Any:
    ctx: PhantomInterpreterRuntimeCtx = get_phantomctx()

    gid: int = id(globals_dict)
    key: tuple[int, str] = (gid, name)

    if key in ctx.global_deletions:
        return do_raise(NameError(f"name '{name}' is not defined"))

    p: None | Any = ctx.global_lookups.get(key, None)

    if p is not None:
        return p

    val: Any = globals_dict[name]
    p = ctx.proxify(name, val)
    ctx.global_lookups[key] = p

    return p


phantom_callbacks[JIT_CALLBACKS.LOAD_GLOBAL_CALLBACK] = _load_global_callback


# TODO Consider if this should suppress the population of frame.globals[name] completely
def _store_global_callback(globals_dict: dict, name: str, val: Any, /) -> Any:
    ctx: PhantomInterpreterRuntimeCtx = get_phantomctx()

    # Records the store
    gid: int = id(globals_dict)
    key: tuple[int, str] = (gid, name)
    ctx.global_lookups[key] = val

    # Records that this object is no longer deleted (if it was)
    ctx.global_deletions.discard(key)

    # Returns the existing value (so it's unmodified)
    return globals_dict[name]


phantom_callbacks[JIT_CALLBACKS.STORE_GLOBAL_CALLBACK] = _store_global_callback


def _delete_global_callback(globals_dict: dict, name: str, /) -> None:
    ctx: PhantomInterpreterRuntimeCtx = get_phantomctx()

    assert name in globals_dict

    # Records the deletion
    gid: int = id(globals_dict)
    key: tuple[int, str] = (gid, name)
    ctx.global_deletions.add(key)


phantom_callbacks[JIT_CALLBACKS.DELETE_GLOBAL_CALLBACK] = _delete_global_callback


def phantom_jit(
    fn: Callable,
    *,
    opcode_interpreter: Callable = default_opcode_interpreter,
    fn_lookaside: Callable = default_lookaside,
    callbacks: dict[JIT_CALLBACKS, Callable] = phantom_callbacks,
    ctx_cls: type = PhantomInterpreterRuntimeCtx,
) -> Callable:
    jfn = jit(fn, opcode_interpreter=opcode_interpreter, fn_lookaside=fn_lookaside, callbacks=callbacks)

    @wraps(jfn)
    def fn(*args, **kwargs) -> Callable:
        try:
            ctx: PhantomInterpreterRuntimeCtx = ctx_cls()
            tok: Any = set_phantomctx(ctx)

            si = get_siginfo(fn, args, kwargs)

            pargs = []
            for name, x in si.args:
                p = ctx.proxify(name, x)
                pargs.append(p)

            if si.varargs is not None:
                varargs_name, x = si.varargs
                pvarargs = ctx.proxify(varargs_name, x)
                pargs.extend(pvarargs)

            pkwargs = {}
            for name, x in si.kwargs.items():
                p = ctx.proxify(name, x)
                pkwargs[name] = x

            if si.varkwargs is not None:
                varkwargs_name, x = si.varkwargs
                pvarkwargs = ctx.proxify(varkwargs_name, x)
                pkwargs.update(pvarkwargs)

            result = jfn(*pargs, **pkwargs)

            # Propagates metadata
            # TODO Find a better way to do this? -- why doesn't wraps propagate these?
            fn._last_interpreted_instructions = jfn._last_interpreted_instructions
            fn._last_interpreted_history = jfn._last_interpreted_history
            return result
        finally:
            reset_phantomctx(tok)

    return fn


#
# thunder mode (no side effects + creates a thunder program to execute)
#
# WIP. This currently is just a scaffolding to report compilation statistics.

import time

from thunder.extend import Executor
from thunder.common import CompileData, CompileStats
from thunder.core.trace import TraceCtx

# NOTE Calls into symbols MUST use this lookaside -- we don't want to jit into them
_thunder_lookaside_map = {
    TensorProxy.__add__: TensorProxy.__add__,
}

_thunder_lookaside_map = _default_lookaside_map | _thunder_lookaside_map


def thunder_lookaside(fn, *args, **kwargs) -> None | Callable:
    return _thunder_lookaside_map.get(fn, None)


class ThunderInterpreterRuntimeCtx(PhantomInterpreterRuntimeCtx):
    def __init__(self):
        self._stored: dict[str, Any] = {}
        self._proxies: dict[str, Any] = {}

    @property
    def stored(self) -> dict[str, Any]:
        return self._stored

    @property
    def proxies(self) -> dict[str, Any]:
        return self._proxies

    def record_stored(self, name: str, val: Any) -> None:
        self._stored[name] = val

    # TODO Extend proxies beyond torch tensors
    # TODO Some objects are not copyable, like ModuleTypes, but we still
    #   want to prevent mutation to them -- for now we just don't proxy them
    # TODO Call the phantom interpreter's proxy method on non-thunder proxies
    def proxify(self, name: str, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return proxy(x)

        return x


def thunder_jit(fn: Callable, *args, **kwargs) -> Callable:
    return phantom_jit(fn, ctx_cls=ThunderInterpreterRuntimeCtx, fn_lookaside=thunder_lookaside)


# TODO Add support for transforms
# TODO Introduce caching
# TODO Support other langctx
def _create_callable(cd: CompileData, cs: CompileStats) -> Callable:
    jfn = thunder_jit(cd.fn)

    @wraps(cd.fn)
    def fn_(*args, **kwargs) -> tuple[Any, list[TraceCtx]]:
        cs.last_trace_host_start = time.time_ns()
        cs.calls += 1

        # TODO Caching goes here

        # Currently executes the program eagerly as a placeholder
        computation_trace = TraceCtx()
        lang = get_default_langctx()
        try:
            lang_tok = set_langctx(lang)
            trace_tok = set_tracectx(computation_trace)
            cs.last_trace_tracing_start = time.time_ns()
            result = jfn(*args, **kwargs)
            cs.last_trace_tracing_stop = time.time_ns()
        finally:
            reset_tracectx(trace_tok)
            reset_langctx(lang_tok)

        # TODO Apply transforms

        # Executes the traced program
        cs.last_trace_host_execution_start = time.time_ns()
        # TODO Execute the traced program (currently it's executed eagerly)
        cs.last_trace_host_execution_stop = time.time_ns()

        # TODO Update cache

        # Updates metadata
        cs.last_interpreted_instructions = jfn._last_interpreted_instructions
        cs.last_interpreted_history = jfn._last_interpreted_history

        cs.last_trace_host_stop = time.time_ns()
        return result

    fn_._lc_cd = cd
    fn_._lc_cs = cs
    return fn_


# TODO Support recursive litjiting
# NOTE This is an analogue to thunder.compile, because how it handles trace generation
#   is sufficiently distinct that merging the two would be quite tricky
def litjit(
    fn: Callable,
    executors_list: None | Sequence[Executor] = None,
) -> Callable:
    cd = CompileData(
        fn=fn,
        langctx=None,
        executors_list=executors_list,
        cache_mode=None,
        use_cudagraphs=False,
        use_torch_compile=False,
        disable_torch_autograd_support=True,
        use_rematerialization=False,
        only_execute_prims=False,
        disable_preprocessing=False,
    )

    cs = CompileStats()
    fn_ = _create_callable(cd, cs)
    return fn_
