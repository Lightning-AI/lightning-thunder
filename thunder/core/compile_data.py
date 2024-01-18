from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_compile_data = ContextVar("compile_data", default=(None, None))


def get_compile_option(option: str, description: str, /) -> None | Any:
    cd, cs = _compile_data.get()

    if cs or cs is None:
        return None

    # See NOTE Different categories of compile options in thunder/__init__.py
    cs.last_compile_reasons[option].append(description)
    return cd.compile_options.get(option, None)


def get_compile_data():
    """Returns the current compile data.

    Returns None if there is no compile data.
    """
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
