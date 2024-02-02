from contextvars import ContextVar
from typing import Optional, Any
from functools import wraps
from contextlib import contextmanager


# TODO Create a langctx interface
# TODO make the default language context configurable
def get_default_langctx() -> Any:
    import thunder.torch as torchlangctx

    return torchlangctx


#
# Functions related to setting, getting, and resetting the current tracing context
#
_langctx = ContextVar("langctx")


def set_langctx(ctx):
    """Sets the current language context."""

    return _langctx.set(ctx)


# TODO allowing setting a different default language context?
def get_langctx():
    """Gets the current language context, returning the default language context (for PyTorch)
    if there is no current language context."""

    try:
        t = _langctx.get()
        return t
    except LookupError:
        pass

    import thunder.torch as torchlangctx

    return torchlangctx


def reset_langctx(token):
    """Resets the language context."""

    _langctx.reset(token)


# TODO allow 3rd parties to extend the lang ctx by setting an environment variable pointing to their langctx
# TODO document the langctx contract
# TODO refactor this so that all available language contexts are available for review
#   Then enumerate through them
# TODO guard imports and include device type thinking
def langctx_for(x: Any) -> Any:
    import thunder.torch as torchlangctx
    import thunder.numpy as nplangctx

    if isinstance(x, torchlangctx.tensor_cls):
        return torchlangctx
    if isinstance(x, nplangctx.tensor_cls):
        return nplangctx

    return get_langctx()


# TODO allow the prim fwd language context to be set?
#   Instead of allowing it to be set, we may want to preserve tensor mediums better
#   (instead of converting everything to a common medium, which is torch tensors today) and
#   select the prim fwd ctx based on the tensor medium
def get_prim_fwd_langctx() -> Any:
    import thunder.torch as torchlangctx

    return torchlangctx


def get_numberctx() -> Any:
    import thunder.core.prims as numberctx

    return numberctx


# Decorator for setting the language context local to a function
#   Ex. @langctx(torchlangctx)
class langctx:
    def __init__(self, _langctx, /):
        self.langctx = _langctx

    def __call__(self, fn):
        @wraps(fn)
        def _fn(*args, **kwargs):
            try:
                tok = set_langctx(self.langctx)
                result = fn(*args, **kwargs)
                return result
            finally:
                reset_langctx(tok)

        return _fn


@contextmanager
def lang(ctx: Any) -> None:
    tok = set_langctx(ctx)
    try:
        yield
    finally:
        reset_langctx(tok)
