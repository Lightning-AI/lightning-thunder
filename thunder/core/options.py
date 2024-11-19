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


class DebugOptions:
    _defaults = {}
    _docs = {}

    def __init__(self, **kwargs):
        cls = self.__class__
        for k, default in self._defaults.items():
            v = kwargs.pop(k, default)
            typ = cls.__annotations__[k]
            if not isinstance(v, typ):
                raise TypeError(f"{cls.__name__}.{k} needs to be of type {typ.__name__}")
            setattr(self, k, v)
        if kwargs:
            unknown_args = ", ".join(f"{k}" for k in kwargs)
            raise TypeError(f"unknown argument(s) for {cls.__name__}: {unknown_args}")

    @classmethod
    def register_option(cls, name, typ, default, doc=""):
        if hasattr(cls, name):
            raise AttributeError(f"{cls.__name__}.{name} is already registered")

        assert isinstance(default, typ)
        cls._defaults[name] = default
        cls.__annotations__[name] = typ
        cls._docs[name] = doc
        setattr(cls, name, default)
        cls._set_docstring()

    @classmethod
    def _set_docstring(cls):
        cls.__doc__ = f"""{cls.__name__}(**options)
    options can be dynamically registered, currently registered ones are below

    Keyword Args:
        {cls.list_options(docstr=True)}
        """

    @classmethod
    def list_options(cls, docstr=False):
        lines = []
        cls.__annotations__  # initialize annotations in cls.__dict__
        for name, default in sorted(cls._defaults.items()):
            typ = cls.__annotations__[name]
            doc = cls._docs[name]
            lines.append(f"{name}: {typ.__name__}={default}   {doc}")

        sep = "\n" if not docstr else "\n\n        "
        return sep.join(lines)

    def __repr__(self):
        cls = self.__class__
        repr = [f"{cls.__name__}("]
        for k, default in cls._defaults.items():
            v = getattr(self, k, default)
            if v != default:
                repr.append(f"  {k}={v},")
        repr.append(")")
        if len(repr) <= 3:
            return "".join(r.lstrip().rstrip(",") for r in repr)
        return "\n".join(repr)


DebugOptions._set_docstring()
