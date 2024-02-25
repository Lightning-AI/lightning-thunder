# NOTE This delays annotation evaluation, allowing a class to annotate itself
#   This feature is available in Python 3.7 and later.
#   This import (like all __future__ imports) must be at the beginning of the file.
from __future__ import annotations
from enum import Enum
import functools
import os
import dis

import sys
import collections.abc
from numbers import Number
from typing import Any, Type, Union, Optional, Tuple, List
from collections.abc import Callable
from collections.abc import Sequence
from types import MappingProxyType, ModuleType, CodeType, EllipsisType, FunctionType, MethodType
import re
import inspect

import torch
import numpy as np


#
# Common utilities importable by any other file
#


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
# Functions related to Python object queries and manipulation
#


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


# Function objects have a lot of stuff in them, that may not be relevant when we
# just want to preview bytecode. Here, we make a list of the things we don't need to see.
fnprint_exclude_attrs = {
    "__class__",
    "__doc__",
    "co_code",
    "co_lnotab",
    "co_name",
    "co_firstlineno",
    "co_filename",
    "co_kwonlyargcount",
    "co_stacksize",
    "co_flags",
    "co_nlocals",
    "co_linetable",
}


# View the bytecode and code object of a function or code object and any nested functions.
# This is meant to be useful for figuring out what's going on in a function.
# Example:
#     @fnprint
#     def foo(a, b):
#         return a + b
def fnprint(fn: FunctionType | MethodType | CodeType, first=True) -> Callable:
    if isinstance(fn, FunctionType):
        x = fn.__code__
    elif isinstance(fn, MethodType):
        x = fn.__func__.__code__  # type: ignore
    else:
        x = fn

    if first:
        try:
            source = inspect.getsource(x)
        except:
            source = "Source could not be found."
    else:
        source = f"SUBFUNCTION {x.co_name}:"

    print(source)
    for k in dir(x):
        v = getattr(x, k)
        if hasattr(v, "__call__"):
            continue
        if k in fnprint_exclude_attrs:
            continue
        print(f"{k}: {v}")

    print("co_code:")
    dis.dis(x, depth=0)
    print()

    # Recurse for nested functions
    for f in x.co_consts:
        if f.__class__ == x.__class__:
            fnprint(f, False)
            print()

    if first:
        print("=" * 50)
        print()

    def error_fn(*args, **kwargs):
        raise ValueError("A code object was passed to fnprint(). Cannot return a callable of it.")

    return fn if isinstance(fn, Callable) else error_fn


def _print_float_number(f: float) -> str:
    if f != f:
        return "float('NaN')"

    if f == float("inf"):
        return "float('inf')"

    if f == -float("inf"):
        return "-float('inf')"

    return str(f)


def _print_complex_number(c: complex) -> str:
    real_str: str = _print_float_number(c.real)
    imag_str: str = _print_float_number(c.imag)

    return f"complex({real_str}, {imag_str})"


def print_number(n: Number) -> str:
    if isinstance(n, complex):
        return _print_complex_number(n)
    if isinstance(n, float):
        return _print_float_number(n)

    return str(n)


#
# Functions related to printing code
#

tab = "  "


def indent(level):
    return f"{tab * level}"


_torch_dtype_to_str_map = {
    torch.bool: "torch.bool",
    torch.uint8: "torch.uint8",
    torch.int8: "torch.int8",
    torch.int16: "torch.int16",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.bfloat16: "torch.bfloat16",
    torch.float16: "torch.float16",
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.complex32: "torch.complex32",
    torch.complex64: "torch.complex64",
    torch.complex128: "torch.complex128",
}

_type_to_str_map = {
    bool: "bool",
    int: "int",
    float: "float",
    complex: "complex",
    str: "str",
    tuple: "tuple",
    list: "list",
    dict: "dict",
    slice: "slice",
    EllipsisType: "ellipsis",
    torch.Size: "torch.Size",
    torch.strided: "torch.strided",
    torch.device: "torch.device",
    torch.dtype: "torch.dtype",
}


def is_base_printable_type(typ: type, /) -> bool:
    try:
        return typ in _type_to_str_map
    except:
        return False


def print_base_type(typ: type, /) -> str:
    return _type_to_str_map[typ]


# TODO Document this function and ensure it's used consistently
# TODO Add more basic Python types
def print_type(typ: type, /, *, with_quotes: bool = True) -> str:
    # Special cases basic Python types
    s: str
    if is_base_printable_type(typ):
        s = print_base_type(typ)

        if with_quotes:
            return f"'{s}'"
        return s

    # NOTE not is_base_printable_type(typ)
    # Handles the general case of where types are printed like
    #   <class 'float'>
    #   Does this by capturing the name in quotes with a regex
    s = str(typ)
    result = re.search(".+'(.+)'.*", s)

    if with_quotes:
        return f"'{result.group(1)}'"

    return result.group(1).replace(".", "_")


_printable_literals = {
    None: "None",
    Ellipsis: "...",
    torch.strided: "torch.strided",
}

_printable_value_types = {
    str: lambda s: f'"{s}"',
    torch.device: lambda d: f'torch.device("{str(d)}")',
    torch.dtype: lambda d: _torch_dtype_to_str_map[d],
    bool: lambda b: str(b),
    int: lambda b: str(b),
    float: _print_float_number,
    complex: _print_complex_number,
    slice: lambda slc: str(slc),
}


def is_base_printable_literal(x: Any, /) -> bool:
    try:
        return x in _printable_literals
    except:
        return False


def is_base_printable_value(x: Any, /) -> bool:
    return type(x) in _printable_value_types


# True if the object can be printed by print_base_printable; False o.w.
def is_base_printable(x: Any, /) -> bool:
    if isinstance(x, ProxyInterface):
        return True

    if is_base_printable_type(x):
        return True

    if is_base_printable_literal(x):
        return True

    if is_base_printable_value(x):
        return True

    return False


# Returns a string representing the value. Throws a ValueError if the object
#   cannot be converted to a string.
def print_base_printable(x: Any, /) -> str:
    if isinstance(x, ProxyInterface):
        return x.name

    if is_base_printable_type(x):
        return print_base_type(x)

    if is_base_printable_literal(x):
        return _printable_literals[x]

    fn: None | Callable = _printable_value_types.get(type(x), None)
    if fn is not None:
        return fn(x)

    raise ValueError(f"print_base_printable was called with type {type(x)} that it doesn't know how to print")


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
# Functions related to printing with colors
#


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


class TermColors(Enum):
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"


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
