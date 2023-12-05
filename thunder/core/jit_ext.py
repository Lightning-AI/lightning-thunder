from typing import Any
from collections.abc import Callable

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

def phantom_jit(*args, **kwargs):
    assert 'callbacks' not in kwargs
    kwargs['callbacks'] = phantom_callbacks
    jfn = jit(*args, **kwargs)

    @wraps(jfn)
    def fn(*args, **kwargs):
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