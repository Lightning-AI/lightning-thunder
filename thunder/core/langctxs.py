from contextvars import ContextVar
from typing import Optional, Any
from collections.abc import Callable
from functools import wraps
from contextlib import contextmanager
from enum import Enum, auto

#
# Context variables, context managers, and helpers related to setting the language context.
#   The language context is a context variable that determines how methods on proxies are resolved.
#   For example, in NumPy, ndarray.size returns the number of elements in the array. In PyTorch,
#   torch.Tensor.size(dim=None) returns the tensor's shape when dim is None, and the length of the
#   specified dimension when dim specifies a dimension (using an integer offset).
#


class LanguageContext:
    def __init__(self, name: str, /):
        self._name: str = name

    @property
    def name(self, /) -> str:
        return self._name

    # TODO RC1 Update this signature to include args and kwargs?
    def has_method(self, id: str) -> bool:
        raise NotImplementedError("Abstract base class")

    # Finds the appropriate method for the given arguments
    # Should raise an exception if the language doesn't have the method
    def get_method(self, id: str, *args, **kwargs) -> Callable:
        # Note: concrete implementations should only raise AttributeError or
        #       return None for "missing" methods as the proxies will
        #       route __getattr__ to here and hasattr relies on __getattr__
        #       throwing AttributeError (only) when the attribute does
        #       not exist.
        raise NotImplementedError("Abstract base class")


#
# Functions related to setting, getting, and resetting the current tracing context
#
_langctx = ContextVar("langctx")


def set_langctx(ctx: LanguageContext, /) -> Any:
    """Sets the current language context."""
    if not isinstance(ctx, LanguageContext):
        raise ValueError(f"Cannot set type {type(ctx)} as a language context")

    return _langctx.set(ctx)


def get_langctx() -> LanguageContext:
    """Gets the current language context"""
    t = _langctx.get()
    return t


def reset_langctx(token: Any, /) -> None:
    """Resets the language context."""
    _langctx.reset(token)


# A helper for acquiring a method
def resolve_method(id: Any, *args, **kwargs) -> Callable:
    ctx: LanguageContext = get_langctx()
    method: Callable = ctx.get_method(id, *args, **kwargs)
    return method


_langctx_registry: dict[Any, LanguageContext] = {}


def register_langctx(id: Any, ctx: LanguageContext) -> None:
    if not isinstance(ctx, LanguageContext):
        raise ValueError(f"Cannot register type {type(ctx)} as a LanguageContext")
    _langctx_registry[id] = ctx


def resolve_language(id: Any, /) -> LanguageContext:
    if isinstance(id, LanguageContext):
        return id

    # Tries to look up the language context
    lang: None | LanguageContext = _langctx_registry.get(id, None)

    if lang is None:
        raise ValueError(f"Unknown language context {id}")

    return lang


# IDs for first-party languages
class Languages(Enum):
    NUMPY = auto()
    TORCH = auto()
    CLANG = auto()
    PRIMS = auto()


# Decorator and context manager for setting the language context with a function
#   or region
#
#   Ex. @langctx(torchlangctx)
#       def foo(...):
#
#   Ex. with langctx(torchlangctx):
#           ...
class langctx:
    def __init__(self, _langctx: Any | LanguageContext, /):
        if not isinstance(_langctx, LanguageContext):
            _langctx = _langctx_registry.get(_langctx, None)
            if _langctx is None:
                raise ValueError(f"Unknown language {_langctx}")

        self.langctx: LanguageContext = _langctx

    def __call__(self, fn: Callable, /) -> Callable:
        @wraps(fn)
        def _fn(*args, **kwargs):
            try:
                tok = set_langctx(self.langctx)
                result = fn(*args, **kwargs)
                return result
            finally:
                reset_langctx(tok)

        return _fn

    def __enter__(self):
        self.tok = set_langctx(self.langctx)

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_langctx(self.tok)
