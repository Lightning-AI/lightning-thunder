from typing import Any
from collections.abc import Sequence
from enum import Enum, auto
import warnings

#
# Options that can be accessed by all parts of the system
#

#
# Common helpers
#


def _unknown_option(option_name: str, allowed: Sequence[str], default: str, unknown: Any, /) -> None:
    allowed_options: str = ", ".join(allowed)
    raise ValueError(
        f"Unknown {option_name} option {unknown}. Allowed options are {allowed_options}. The default option is {default}."
    )


#
# Interpretation options
#
# These options control how the function will be interpreted.
# PYTHON_INTERPRETER uses the actual Python interpreter to construct the thunder program.
# TRANSLATE_FUNCTIONS is the default option. It uses the thunder interpreter to translate PyTorch operation
#   to thunder operations. For example, torch.add becomes thunder.torch.add.
# TRANSLATE_PYTHON is an experimental option. It lets the thunder interpreter translate
#   the entire function a thunder program.


class INTERPRETATION_OPTIONS(Enum):
    PYTHON_INTERPRETER = auto()
    TRANSLATE_FUNCTIONS = auto()
    TRANSLATE_PYTHON = auto()


_str_to_interpretation_option_map: dict[str, INTERPRETATION_OPTIONS] = {
    "python interpreter": INTERPRETATION_OPTIONS.PYTHON_INTERPRETER,
    "translate functions": INTERPRETATION_OPTIONS.TRANSLATE_FUNCTIONS,
    "translate python": INTERPRETATION_OPTIONS.TRANSLATE_PYTHON,
}


def _str_to_interpretation_option(s: str, /) -> None | INTERPRETATION_OPTIONS:
    return _str_to_interpretation_option_map.get(s.lower(), None)


# Resolves a specified interpretation option, defaulting to TRANSLATE_FUNCTIONS
def resolve_interpretation_option(x: Any, /) -> INTERPRETATION_OPTIONS:
    io: None | INTERPRETATION_OPTIONS

    if x is None:
        io = INTERPRETATION_OPTIONS.TRANSLATE_PYTHON
    elif isinstance(x, INTERPRETATION_OPTIONS):
        io = x
    elif isinstance(x, str):
        io = _str_to_interpretation_option(x)

    if io is None:
        _unknown_option("interpretation", _str_to_interpretation_option_map.keys(), "translate functions", x)

    # if io is INTERPRETATION_OPTIONS.TRANSLATE_PYTHON:
    #     warnings.warn(
    #         "The 'translate python' interpretation option is experimental and still in development. It may not work as expected."
    #     )

    return io


#
# Cache options
#
# These options control how thunder caches programs (quickly mapping inputs to thunder programs)
# NO_CACHING is useful for debugging and experiment. It constructs a new thunder program each time the jitted
#   callable is called. This can be very slow.
# SAME_INPUT is an unsafe option not intended for general use. It attempts to call the first thunder program constructed
#   on all inputs, conceptually assuming that the inputs are the "same". Whether inputs actually are the "same" as
#   other inputs depends on the program.
# CONSTANT_VALUES is the default cache option. It treats values like numbers and strings as compile-time constants.
#   Calling the jitted callable with different numbers or strings will cause a new thunder program to be compiled.
#   Note that the lengths of a tensor's dimensions are considered like other numbers for caching, so passing tensors with
#   different shapes will also cause a new program to be compiled.
#   This option is very efficient if input values are mostly constant because some operations -- like
#   checking that two tensors have the same shape, or adding numbers together -- can be computed at compile-time
#   instead of run-time.
# SYMBOLIC_VALUES is currently experimental and for development only.
#   It treats values like numbers and strings as dynamic symbols whose values may vary across calls to the jitted callable.
#   This can reduce compile-time if the value inputs to the jitted callable vary over time, but it also increases
#   the time it takes to evaluate the cache, as operations that could be performed at compile-time
#   must now be performed at run-time.


class CACHE_OPTIONS(Enum):
    NO_CACHING = auto()
    SAME_INPUT = auto()
    CONSTANT_VALUES = auto()
    SYMBOLIC_VALUES = auto()


_string_to_cache_option_map = {
    "no caching": CACHE_OPTIONS.NO_CACHING,
    "same input": CACHE_OPTIONS.SAME_INPUT,
    "constant values": CACHE_OPTIONS.CONSTANT_VALUES,
    "symbolic values": CACHE_OPTIONS.SYMBOLIC_VALUES,
}


def _string_to_cache_option(s: str, /) -> None | CACHE_OPTIONS:
    return _string_to_cache_option_map.get(s.lower(), None)


# Resolves a specified cache option, defaulting to CONSTANT_VALUES
def resolve_cache_option(x: Any, /) -> CACHE_OPTIONS:
    co: None | CACHE_OPTIONS
    if x is None:
        co = CACHE_OPTIONS.CONSTANT_VALUES
    elif isinstance(x, CACHE_OPTIONS):
        co = x
    elif isinstance(x, str):
        co = _string_to_cache_option(x)

    if co is None:
        _unknown_option("cache", _string_to_cache_option_map.keys(), "constant values", x)

    if co is CACHE_OPTIONS.SYMBOLIC_VALUES:
        warnings.warn("The 'symbolic values' cache option is highly experimental and for development only.")

    return co


#
# Sharp edges options
#
# These options control how thunder handles "sharp edges" -- parts of the original program that
#   may not be captured by a thunder program.
# ALLOW is the default sharp edges options. It ignores sharp edges. This puts the burden of handling sharp
#   edges on practitioners.
# WARN is an experimental option that will emit a warning when a sharp edge is identified.
#   This option can only be set if the thunder interpreter is used. (See interpretation options.)
# ERROR is an experimental option that will raise an exception when a sharp edge is identified.
#   This option can only be set if the thunder interpreter is used. (See interpretation options.)


class SHARP_EDGES_OPTIONS(Enum):
    ALLOW = auto()
    WARN = auto()
    ERROR = auto()


_str_to_sharp_edges_options_map: dict[str, SHARP_EDGES_OPTIONS] = {
    "allow": SHARP_EDGES_OPTIONS.ALLOW,
    "warn": SHARP_EDGES_OPTIONS.WARN,
    "error": SHARP_EDGES_OPTIONS.ERROR,
}


def _str_to_sharp_edges_option(s: str, /) -> None | SHARP_EDGES_OPTIONS:
    return _str_to_sharp_edges_options_map.get(s.lower(), None)


def resolve_sharp_edges_option(x: Any, /) -> SHARP_EDGES_OPTIONS:
    seo: None | SHARP_EDGES_OPTIONS

    if x is None:
        seo = SHARP_EDGES_OPTIONS.ALLOW
    elif isinstance(x, SHARP_EDGES_OPTIONS):
        seo = x
    elif isinstance(x, str):
        seo = _str_to_sharp_edges_option(x)

        if seo is None:
            _unknown_option("sharp edges", _str_to_sharp_edges_options_map.keys(), "allow", x)

        if seo is SHARP_EDGES_OPTIONS.WARN:
            warnings.warn(
                f"The 'warn' sharp edges option is experimental and still in development. It may not work as expected."
            )
        if seo is SHARP_EDGES_OPTIONS.ERROR:
            warnings.warn(
                f"The 'error' sharp edges option is experimental and still in development. It may not work as expected."
            )

    return seo
