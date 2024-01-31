# NOTE This delays annotation evaluation, allowing a class to annotate itself
#   This feature is available in Python 3.7 and later.
#   This import (like all __future__ imports) must be at the beginning of the file.
from __future__ import annotations
from enum import Enum
import functools
import os

import sys
import collections.abc
from numbers import Number
from typing import Any, Type, Union, Optional, Tuple, List
from collections.abc import Callable
from collections.abc import Sequence
from types import MappingProxyType, ModuleType, CodeType
import re
import inspect

#
# Common utils importable by any other file
#
# TODO: make all of these callable through utils (the idea is that this can be imported even if utils cannot)

# Python 3.10 introduces a new dataclass parameter, `slots`, which we'd like to use
# by default. However, we still want to support Python 3.9, so we need to
# conditionally set the default dataclass parameters.
# MappingProxyType is used to make the configuration immutable.
default_dataclass_params = MappingProxyType({"frozen": True, "slots": True})

#
# Functions and classes (and metaclasses) related to class management
#


# Use this a metaclass to get a singleton pattern
#
# Ex.
# class SingletonClass(BaseClasses..., metaclass=Singleton):
#   ...
#
# When lazy initialization is not required, an alternative to a singleton class
#   is just using a new file and importing it.
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


#
# Interfaces to support type annotation and instance checks without
#   creating circular dependencies.
#
class ProxyInterface:
    @property
    def name(self):
        pass

    def type_string(self):
        pass


class NumberProxyInterface:
    pass


class TensorProxyInterface:
    pass


class SymbolInterface:
    name: str
    is_prim: bool
    id: Any | None


class BoundSymbolInterface:
    sym: SymbolInterface
    args: Sequence
    kwargs: dict
    output: Any
    subsymbols: Sequence[BoundSymbolInterface]


#
# Functions related to error handling
#


def check(cond: bool, s: Callable[[], str], exception_type: type[Exception] = RuntimeError) -> None:
    """Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.

    s is a callable producing a string to avoid string construction if the error check is passed.
    """
    if not cond:
        raise exception_type(s())


def check_type(x: Any, types: type | Sequence[type]):
    check(
        isinstance(x, types),
        lambda: f"{x} had an unexpected type {type(x)}. Supported types are {types}",
        exception_type=ValueError,
    )


def check_types(xs: Sequence[Any], types: type | Sequence[type]):
    """
    Checks that all elements in xs have one of the types in types.

    Raises a ValueError if this is not the case.
    """
    for i, x in enumerate(xs):
        check(
            isinstance(x, types),
            lambda: f"Element {i} ({x}) had an unexpected type {type(x)}. Supported types are {types}",
            exception_type=ValueError,
        )


#
# Functions related to Python object queries and manipulation
#


# TODO Review these imports -- not a big deal since we depend on both
#   But if we want to be extensible in the future we probably need to use langctx_for here
#   Which means that this function needs to move to codeutils (probably fine) or
#   this needs to take a dependency on langctx.py (probably not great)
import torch
import numpy as np


# A somewhat hacky way to check if an object is a collection but not a string or
#   tensor object
def is_collection(x: Any) -> bool:
    return isinstance(x, collections.abc.Collection) and not isinstance(x, (str, torch.Tensor, np.ndarray))


def sequencify(x: Any) -> Sequence:
    # NOTE strings in Python are sequences, which requires this special handling
    #   to avoid "hello" being treated as "h", "e", "l", "l", "o"
    if not isinstance(x, Sequence) or isinstance(x, str):
        return (x,)

    return x


def get_module(name: str) -> Any:
    return sys.modules[name]


#
# Functions related to common checks
#


def check_valid_length(length: int):
    """Validates that an object represents a valid dimension length."""

    check_type(length, int)
    check(length >= 0, lambda: f"Found invalid length {length}!")


def check_valid_shape(shape: tuple[int, ...] | list[int]):
    """Validates that a sequence represents a valid shape."""

    check_type(shape, (tuple, list))

    for l in shape:
        check_valid_length(l)


#
# Functions related to printing and debugging
#


def extract_callable_name(fn: Callable | CodeType) -> str:
    if isinstance(fn, CodeType):
        return fn.co_qualname if hasattr(fn, "co_qualname") else fn.co_name
    elif hasattr(fn, "__qualname__"):
        return fn.__qualname__
    elif hasattr(fn, "__name__"):
        return fn.__name__
    elif hasattr(fn, "__class__"):
        return fn.__class__.__name__
    elif isinstance(fn, functools.partial):
        return f"<partial with inner type {extract_callable_name(fn.func)}>"
    else:
        assert isinstance(fn, Callable), (fn, type(fn))
        return f"<callable of type {type(fn).__name__}>"


#
# Functions related to printing code
#

tab = "  "


def indent(level):
    return f"{tab * level}"


_type_to_str_map = {
    bool: "bool",
    int: "int",
    float: "float",
    complex: "complex",
    str: "str",
}


def is_printable_type(typ: type) -> bool:
    return typ in _type_to_str_map


# TODO Document this function and ensure it's used consistently
# TODO Add more basic Python types
def print_type(typ: type, with_quotes: bool = True) -> str:
    # Special cases basic Python types

    if typ in _type_to_str_map:
        return _type_to_str_map[typ]

    # Handles the general case of where types are printed like
    #   <class 'float'>
    #   Does this by capturing the name in quotes with a regex
    s = str(typ)
    result = re.search(".+'(.+)'.*", s)
    if with_quotes:
        return f"'{result.group(1)}'"

    s = result.group(1).replace(".", "_")
    return s


#
# Functions related to constructing callables from Python strings
#

_exec_ctr = 0


def compile_and_exec(fn_name: str, python_str: str, program_name: str, ctx: dict) -> Callable:
    global _exec_ctr

    program_name = f"{program_name}_{_exec_ctr}"

    # simple cache hack
    mtime = None  # this signals that the cache should not be invalidated(!)
    lines = python_str.splitlines(keepends=True)
    size = len(python_str)
    inspect.linecache.cache[program_name] = size, mtime, lines, program_name

    try:
        code = compile(python_str, program_name, mode="exec")
        exec(code, ctx)
        return ctx[fn_name]
    except Exception as e:
        print("Encountered an exception while trying to compile the following program:")
        print(python_str)
        raise e
    finally:
        _exec_ctr += 1


#
# Other utility functions without dependencies on the rest of the codebase
#


class TermColors(Enum):
    BLACK = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


def run_once(f: Callable[[], Any]) -> Callable[[], Any]:
    """
    Wraps a function with no arguments so that it caches the output and only runs once.
    This is similar to functools.cache, but unlike the standard wrapper does not cache the arguments,
    and provides a strong guarantee that the function will only be run once.
    """

    def wrapper(*args, **kwargs):
        if not wrapper._has_run:
            wrapper._result = f(*args, **kwargs)
            wrapper._has_run = True
            return wrapper._result
        else:
            return wrapper._result

    wrapper._has_run = False
    wrapper._result = None
    return wrapper


supported_terms = ["ansi", "linux", "cygwin", "screen", "eterm", "konsole", "rxvt", "kitty", "tmux"]
startswith_terms = ["vt1", "vt2", "xterm"]


@run_once
def warn_term_variable_once() -> None:
    import warnings

    warnings.warn(
        f"Could not determine the terminal type, terminal colors are disabled. To enable colors, set the TERM environment variable to a supported value.{os.sep}"
        f"Supported values are: {supported_terms}{os.sep}"
        f"Or any value starting with: {startswith_terms}{os.sep}"
        f"To disable this message, set TERM to 'dumb'."
    )


@run_once
def init_windows_terminal() -> None:
    """Initializes the Windows terminal to support ANSI colors by calling the command processor."""
    os.system("")


@functools.cache
def init_colors(force_enable: bool | None = None) -> dict[str, str]:
    """
    Returns a dictionary mapping color names to the sequences required to switch to that color.
    See TermColors for the list of color names.

    If force_enable is None or not specified, then we attempt to discern if the environment supports colors.
    If force_enable is True, then we return the sequences anyway, even if we detect it is not supported.
    If force_enable is False, then we return empty strings for the colors.

    Regardless of the value of force_enable, if the TERM environment variable is set to 'dumb', we return empty strings for the colors.
    """

    windows = os.name == "nt"
    windows_terminal = windows and (os.environ.get("WT_SESSION", False) is False)
    term = os.environ.get("TERM", None)

    # Check if colors should be enabled
    if term == "dumb":
        colors_enabled = False
    elif force_enable is None:
        if windows_terminal:
            colors_enabled = True
        elif term is None:
            colors_enabled = False
            warn_term_variable_once()
        elif term not in supported_terms and not any(term.startswith(t) for t in startswith_terms):
            colors_enabled = False
            warn_term_variable_once()
        else:  # Terminal supported
            colors_enabled = True
    else:
        colors_enabled = force_enable

    # Do initialization for windows terminal (and potentially other terminals on windows that go through it)
    if colors_enabled and windows:
        init_windows_terminal()

    return {k.name: k.value if colors_enabled else "" for k in TermColors}
