from contextlib import contextmanager
from contextvars import ContextVar

_compile_data = ContextVar("compile_data", default=None)


def get_compile_data():
    """Returns the current compile data.

    Returns None if there is no compile data.
    """
    return _compile_data.get()


def set_compile_data(cd):
    """Sets the current compile data.

    This is used to pass compile data to functions that are called during compilation.
    """
    token = _compile_data.set(cd)
    return token


def reset_compile_data(token):
    """Resets the compile data."""
    _compile_data.reset(token)


@contextmanager
def compile_data(cd):
    """Sets the current compile data for the duration of the context."""
    token = set_compile_data(cd)
    try:
        yield
    finally:
        reset_compile_data(token)
