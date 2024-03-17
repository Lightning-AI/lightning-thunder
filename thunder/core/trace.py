# import contextvars  # TODO: review this (vs threadlocal?) -- currently used to set the current trace
import string
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar

# This file defines the tracing context, methods for acquiring it, and related classes.
# This file is the base of Thunder's file hierarchy. It is always safe to import.
# In the future, the tracing context will be elaborated on with more structure, like
#   regions and transitions between regions.
# Currently, traces are represented as single regions with no transitions. All their
#   constraints are evaluated before entry, and not incrementally as transitions
#   between distinct traces.

__all__ = [
    "Constraint",
    "Trace",
    "new_trace",
    "get_trace",
    "reset_trace",
    "set_language_context",
    "get_language_context",
    "reset_language_context",
    "set_executor_context",
    "get_executor_context",
    "reset_executor_context",
]

#
# ContextVars
#

# Holds the current Trace object
_trace = ContextVar("trace")


def new_trace():
    """Sets the current trace."""

    return _trace.set(Trace())


def get_trace():
    """Gets the current trace, returning None if there is no current trace."""

    try:
        t = _trace.get()
        return t
    except LookupError:
        pass

    return None


def reset_trace(token):
    """Resets the tracing state."""

    _trace.reset(token)


@contextmanager
def detached_trace():
    """Context manager that detaches the current trace.

    This is useful for code that should not be traced, but that is called from
    traced code. For example, if you have a function that is traced, and that
    function calls a function that should not be traced, you can use this context
    manager to detach the current trace before calling the function that should
    not be traced.
    """
    trace_token = new_trace()
    yield
    reset_trace(trace_token)


# Holds the current language context
# TODO: unused
# NOTE: this file does not depend on the definition of the language context,
#   so it's an opaque object from the perspective of this file
_language_ctx = ContextVar("language_ctx")


def set_language_context(ctx):
    """Sets the current trace."""

    return _language_ctx.set(ctx)


# TODO: add ability to get a temporary "anonymous" trace
# TODO: possibly add a kwarg to control this behavior
def get_language_context():
    """Gets the current trace, returning None if there is no current trace."""

    try:
        ctx = _language_ctx.get()
        return ctx
    except LookupError:
        pass

    return None


def reset_language_context(token):
    """Resets the tracing state."""

    _language_ctx.reset(token)


# Holds the current execution context
# NOTE: this file does not depend on the definition of the execution context,
#   so it's an opaque object from the perspective of this file
_executor_ctx = ContextVar("executor_ctx")


def set_executor_context(ctx):
    """Sets the current trace."""

    return _executor_ctx.set(ctx)


# TODO: add ability to get a temporary "anonymous" trace
# TODO: possibly add a kwarg to control this behavior
def get_executor_context():
    """Gets the current execution context, returning None if there is no current trace."""

    try:
        ctx = _executor_ctx.get()
        return ctx
    except LookupError:
        pass

    return None


def reset_executor_context(token):
    """Resets the tracing state."""

    _executor_ctx.reset(token)


class Trace:
    """The tracing context.

    Contains common datastructures to track the trace.
    """

    def __init__(self):
        self.args = None
        self.kwargs = None
        self.outputs = None

        self.symbols = deque()

        self.name_ctr = 0
        self.names = set()

    def __repr__(self):
        symbol_string = "\n".join(str(sym) for sym in self.symbols)
        return (
            f"[Trace"
            f"\nArgs:\n{self.args}"
            f"\nKwargs:\n{self.kwargs}"
            f"\nSymbols:\n{symbol_string}"
            f"\nOutputs:\n{self.outputs}"
            f"\n]"
        )

    def add_args(self, args):
        self.args = args

    def add_kwargs(self, kwargs):
        self.kwargs = kwargs

    def add_outputs(self, outputs):
        self.outputs = outputs

    def add_symbol(self, sym):
        self.symbols.append(sym)
        return sym

    def _make_proxy_name(self, generated_name_counter, names):
        chars = tuple(string.ascii_lowercase)

        def _gen_name(ctr):
            place = 0
            s = ""
            while ctr >= 0:
                if place > 0:
                    ctr = ctr // (place * len(chars))
                idx = ctr % (len(chars))
                c = chars[idx]
                s = c + s
                ctr = ctr - (ord(c) + place * len(chars))
                place += 1

            # NOTE: adds "__" to avoid collision with keywords
            # TODO: improve naming to avoid conflicts
            return "__" + s

        ctr = generated_name_counter
        name = None
        while True:
            name = _gen_name(ctr)
            ctr += 1
            if name not in names:
                break

        names.add(name)
        return name, ctr, names

    def make_proxy_name(self):
        name, self.name_ctr, self.names = self._make_proxy_name(self.name_ctr, self.names)
        return name
