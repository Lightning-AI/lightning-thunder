from contextvars import ContextVar
from typing import Callable


NOINLINE_METHODS: ContextVar[set[Callable]] = ContextVar("NOINLINE_METHODS", default=set())


def noinline(f: Callable) -> Callable:
    """
    Function/Decorator to prevent preprocessing from inlining the function.

    Example:
    >>> @noinline
    >>> def foo(x):
    >>>     return x + 1
    >>> def bar(x):
    >>>     return foo(x) + 1
    >>> thunder.compile(bar)
    """

    NOINLINE_METHODS.get().add(f)
    return f


@noinline
def invoke_noinline(f: Callable) -> Callable:
    """
    Function to prevent preprocessing from inlining a single invocation of a function.

    Example:
    >>> def foo(x):
    >>>     return x + 1
    >>> def bar(x):
    >>>     return invoke_noinline(foo)(x) + 1
    >>> thunder.compile(bar)
    """

    return f
