from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from thunder.core.options import CACHE_OPTIONS

#
# Setting and querying the "compile_data_and_stats" context variable, which contains
#   a tuple of (CompileData, CompileStats) objects for the current trace.
#

_compile_data = ContextVar("compile_data", default=(None, None))


#
# Setting, getting, and resetting the context variable
#


# NOTE This just acquires the compile data part of the context var's tuple
def get_compile_data() -> None | Any:
    """Returns the current compile data."""
    cd, cs = _compile_data.get()

    return cd


def set_compile_data_and_stats(cd, cs, /):
    """Sets the current compile data.

    This is used to pass compile data to functions that are called during compilation.
    """
    token = _compile_data.set((cd, cs))
    return token


def reset_compile_data_and_stats(token, /):
    """Resets the compile data."""
    _compile_data.reset(token)


@contextmanager
def compile_data_and_stats(cd, cs, /):
    """Sets the current compile data for the duration of the context."""
    token = set_compile_data_and_stats(cd, cs)
    try:
        yield
    finally:
        reset_compile_data_and_stats(token)


#
# Query helpers
#


def get_compile_option(option: str, description: str, /) -> None | Any:
    cd, cs = _compile_data.get()

    if cd is None or cs is None:
        return None

    # See NOTE Different categories of compile options in thunder/__init__.py
    cs.last_compile_reasons[option].append(description)
    return cd.compile_options.get(option, None)


# Whether or not the caching option uses symbolic values
def get_cache_option() -> CACHE_OPTIONS:
    cd = get_compile_data()
    return cd.cache_option


# TODO RC1 Remove the try (hack for when operating outside of this contextvar being set)
def using_symbolic_values() -> bool:
    try:
        return get_cache_option() is CACHE_OPTIONS.SYMBOLIC_VALUES
    except:
        return False


def using_jit() -> bool:
    try:
        cd, cs = _compile_data.get()
        return cd.using_jit
    except:
        return False
