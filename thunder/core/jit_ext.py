from typing import Any
from collections.abc import Callable, Sequence

from functools import partial, wraps
import copy
import contextvars

#
# jit_ext.py implements extensions of thunder's interpreter
#

#
# phantom mode (no side effects)
#


# TODO Track deleted to deal with duplicate names
# TODO Make proxy construction extensible
# TODO Probably don't want to track what's stored or proxied based on name alone
class PhantomJitRuntimeCtx:
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

    def proxify(self, name: str, x: Any) -> Any:
        if name in self.proxies:
            return self.proxies[name]
        p = copy.deepcopy(x)
        self._proxies[name] = p
        return p


_phantomctx = contextvars.ContextVar("phantomctx")


# Sets the phantom ctx
def set_phantomctx(ctx: PhantomJitRuntimeCtx) -> Any:
    return _phantomctx.set(ctx)


# Returns the current phantom ctx
def get_phantomctx() -> PhantomJitRuntimeCtx:
    return _phantomctx.get()


# Resets the phantom ctx
def reset_phantomctx(token) -> None:
    _phantomctx.reset(token)


from thunder.core.jit import jit, default_callbacks, JIT_CALLBACKS

phantom_callbacks: dict[JIT_CALLBACKS, Callable] = {}


def load_callback(name: str, val: Any):
    ctx: PhantomJitRuntimeCtx = get_phantomctx()

    if name in ctx.stored:
        return val

    return ctx.proxify(name, val)


phantom_callbacks[JIT_CALLBACKS.LOAD_CALLBACK] = load_callback


def store_callback(name: str, val: Any):
    ctx: PhantomJitRuntimeCtx = get_phantomctx()
    ctx.record_stored(name, val)
    return val


phantom_callbacks[JIT_CALLBACKS.STORE_CALLBACK] = store_callback


def phantom_jit(fn: Callable, *args, **kwargs):
    assert "callbacks" not in kwargs
    kwargs["callbacks"] = phantom_callbacks
    jfn = jit(fn, *args, **kwargs)

    @wraps(jfn)
    def fn(*args, **kwargs) -> Callable:
        try:
            ctx: PhantomJitRuntimeCtx = PhantomJitRuntimeCtx()
            tok: Any = set_phantomctx(ctx)
            result = jfn(*args, **kwargs)

            # Propagates metadata
            # TODO Find a better way to do this?
            fn._last_interpreted_instructions = jfn._last_interpreted_instructions
            fn._last_history = jfn._last_history
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


def thunder_jit(fn: Callable, *args, **kwargs) -> Callable:
    return phantom_jit(fn, *args, **kwargs)


# TODO Add support for transforms
# TODO Introduce caching
def _create_callable(cd: CompileData, cs: CompileStats) -> Callable:
    jfn = thunder_jit(cd.fn)

    @wraps(cd.fn)
    def fn_(*args, **kwargs) -> tuple[Any, list[TraceCtx]]:
        cs.last_trace_host_start = time.time_ns()
        cs.calls += 1

        # TODO Caching goes here

        # Currently executes the program eagerly as a placeholder
        # TODO Construct the initial trace
        cs.last_trace_tracing_start = time.time_ns()
        result = jfn(*args, **kwargs)
        cs.last_trace_tracing_stop = time.time_ns()

        # TODO Apply transforms

        # Executes the traced program
        cs.last_trace_host_execution_start = time.time_ns()
        # TODO Execute the traced program (currently it's executed eagerly)
        cs.last_trace_host_execution_stop = time.time_ns()

        # TODO Update cache

        # Updates metadata

        cs.last_interpreted_instructions = jfn._last_interpreted_instructions
        cs.last_interpreted_history = jfn._last_history

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
