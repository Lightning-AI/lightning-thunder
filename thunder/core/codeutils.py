from types import CodeType, FunctionType, MethodType, EllipsisType
from typing import List, Dict, Tuple, Set, Deque, Any
from numbers import Number
from collections import deque
from collections.abc import Mapping, Sequence, Iterable
import inspect
from inspect import Parameter
import string
import functools
from functools import partial
import dis

import torch

import thunder.core.baseutils as baseutils
from thunder.core.baseutils import ProxyInterface, check
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.baseutils import *

#
# Functions related to analyzing and printing functions and arguments
#


# TODO This can be a frozen dataclass
class ContextObject:
    def __init__(self, name: str, obj: Any):
        self.name = name
        self.obj = obj


Printable = str | ContextObject | ProxyInterface

_modules_to_shortnames_map = {
    "thunder.torch": "ltorch",
    "thunder.numpy": "lnp",
    "thunder.core.prims": "prims",
}


def module_shortname(module):
    return _modules_to_shortnames_map.get(module, module)


def indent_string(indent):
    tab = "  "
    return f"{tab * indent}"


def is_simple_printable_collection(x: Any) -> bool:
    simple_printable_collection_types = (
        tuple,
        list,
        dict,
    )

    return type(x) in simple_printable_collection_types


def is_printable(x: Any) -> tuple[bool, None | tuple[str, Any]]:
    if baseutils.is_base_printable(x):
        return True, None

    if isinstance(x, ContextObject):
        return True, None
    if is_collection(x):
        # TODO RC1 Fix collection printing by testing if each item is printable and gathering the imports
        #   required (if any)
        flat, _ = tree_flatten(x)
        return True, None
        # return all((is_printable(f) for f in flat)), None
    if isinstance(x, dtypes.dtype):
        return True, ("dtypes", dtypes)
    if isinstance(x, devices.Device):
        return True, ("devices", devices)

    return False, None


def is_literal(x: Any) -> bool:
    if isinstance(x, (ContextObject, ProxyInterface)):
        return False

    if is_collection(x):
        flat, _ = tree_flatten(x)
        for f in flat:
            if is_literal(f):
                return True
        return False

    return True


def _to_printable(tracectx: Optional, x: Any) -> tuple[Any, Optional[tuple[str, Any]]]:
    can_print, module_info = is_printable(x)
    if can_print:
        return x, module_info

    # NOTE Non-printable objects are serialized in the Python context, if a trace context is available
    if tracectx is not None:
        co = tracectx.add_object(x)
        return co, None

    # NOTE In this case we don't understand how to serialize the object as a string, and
    #   there's no trace ctx, so this will be printed as an unknown object
    return x, None


# TODO Improve type annotations
def to_printable(
    trace: Optional,
    x: Any,
    *,
    import_ctx: Optional[dict] = None,
    object_ctx: Optional[dict] = None,
) -> Printable:
    # Short-circuits if x is a Proxy
    if isinstance(x, ProxyInterface):
        return x

    if is_collection(x):
        flat, spec = tree_flatten(x)

        printables = []
        for f in flat:
            printables.append(to_printable(trace, f, import_ctx=import_ctx, object_ctx=object_ctx))

        printable = tree_unflatten(printables, spec)
        return printable

    # NOTE In this case the object is not a collection, and
    #   it may require an import or additional context to print

    # TODO Instead of constant names, maybe "context names"?
    printable, module_info = _to_printable(trace, x)

    if module_info is not None and import_ctx is not None:
        module_name, module = module_info
        import_ctx[module_name] = module

    if isinstance(printable, ContextObject) and object_ctx is not None:
        object_ctx[printable.name] = x

    return printable


# NOTE This quote marker allows for removal of quotation marks when printing collections
_quote_marker = "_@_"


def _qm(s: str, quote_markers: bool) -> str:
    if not quote_markers:
        return s

    return f"{_quote_marker}{s}{_quote_marker}"


# TODO Review prettyprinting other map types like dict -- these need to print strings in a particular way
# TODO Make a passthrough for types known to be serializable using their __repr__
def prettyprint(
    x: Any,
    *,
    with_type: bool = False,
    literals_allowed: bool = True,
    literals_as_underscores: bool = False,
    _quote_markers: bool = False,
) -> str:
    check(
        literals_allowed or not is_literal(x),
        lambda: f"Attempting to print a literal {x} where literals are not allowed",
        exception_type=AssertionError,
    )

    m = partial(_qm, quote_markers=_quote_markers)

    if literals_as_underscores and is_literal(x) and not is_collection(x):
        return m("_")

    if type(x) is str:
        return m(repr(x))

    # ProxyInterface is base-printable, but we treat it with special care
    if isinstance(x, ProxyInterface):
        # NOTE This doesn't need quote markers because it can't
        #   occur in a collection
        if with_type:
            return f'{x.name}: "{x.type_string()}"'
        return m(x.name)

    if baseutils.is_base_printable(x):
        return m(baseutils.print_base_printable(x))

    if isinstance(x, ContextObject):
        return m(x.name)
    if is_collection(x):
        flat, spec = tree_flatten(x)
        printed = tuple(
            prettyprint(x, with_type=False, literals_as_underscores=literals_as_underscores, _quote_markers=True)
            for x in flat
        )
        unflattened = tree_unflatten(printed, spec)
        unflattened_str = str(unflattened)
        # NOTE Collections of strings (so collections of names) print like this --
        #   ('a', 'b') -- but we want them to print like this -- (a, b) --
        #   so this just removes all the single quotes -- this seems super hacky
        unflattened_str = unflattened_str.replace(f"{_quote_marker}'", "")
        unflattened_str = unflattened_str.replace(f"'{_quote_marker}", "")
        return unflattened_str
    if isinstance(x, dtypes.dtype):
        return m(f"dtypes.{str(x)}")
    if isinstance(x, devices.Device):
        return m(f'devices.Device("{str(x)}")')
    if type(x) is type:
        return m(f"{baseutils.print_type(x, with_quotes=False)}")

    # Handles objects that this doesn't know how to serialize as a string
    return m(f"(object of type {print_type(type(x), with_quotes=False)})")


# TODO Make this a frozen dataclass?
class SigInfo:
    def __init__(self, name, /):
        self.name = name
        self.args = []
        self.varargs = None
        self.kwargs = {}
        self.varkwargs = None
        self.defaultdict = {}
        self.unwrapped_fn = None

    def __repr__(self):
        return f"[SigInfo args={self.args}, varargs={self.varargs}, kwargs={self.kwargs}, varkwargs={self.varkwargs}]"

    # NOTE This prints the original signature, not the bound signature
    # TODO Print the original signature's type annotations
    # TODO Maybe be clear about what inputs are const and what aren't?
    # TODO Improve this signature's type annotations
    def prettyprint(self, *, trace: Optional = None, import_ctx: Optional = None, object_ctx=None) -> str:
        def _arg_printer(name: str, has_default: bool, default: Any = None) -> str:
            # NOTE In this case the argument has a default value, like 'a' in foo(a=5)
            if has_default:
                printable = to_printable(trace, default, import_ctx=import_ctx, object_ctx=object_ctx)

                return f"{name}={prettyprint(printable)}"

            # NOTE In this case the value has no default, like 'a' in foo(a)
            return name

        args = []

        for name, _ in self.args:
            printed = _arg_printer(name, name in self.defaultdict, self.defaultdict.get(name, None))
            args.append(printed)

        if self.varargs is not None:
            varargs_name, _ = self.varargs
            args.append(f"*{varargs_name}")

        # Writes the keyword-only marker
        if self.varargs is None and len(self.kwargs.items()) > 0:
            args.append("*")

        for name, _ in self.kwargs.items():
            printed = _arg_printer(name, name in self.defaultdict, self.defaultdict.get(name, None))
            args.append(printed)

        if self.varkwargs is not None:
            varkwargs_name, _ = self.varkwargs
            args.append(f"**{varkwargs_name}")

        arg_str = ", ".join(args)

        return f"def {self.name}({arg_str}):"


# Creates a SigInfo object from a function and the inputs to it
# The SigInfo object contains name and value information for the args, varargs, kwargs, and varkwargs
#   given to a function.
# To call a function foo from its SigInfo, you can do the following:
#
# arg_values = tuple(x[1] for x in si.args)
# if si.varargs is not None:
#     arg_values = arg_values + si.varargs[1]
# kwarg_values = si.kwargs
# if si.varkwargs is not None:
#     kwarg_values.update(si.varkwargs[1])
# foo(*arg_values, **kwarg_values)
#
# This removes the name information and combines the args and varargs into arg_values,
#   and the kwargs and varkwargs into kwarg_values


# TODO RC1 Review errors and improve message quality (ex. too many arguments error)
# TODO RC1 Have this always return a SigInfo or another type (maybe by wrapping in another function)
def get_siginfo(fn: Callable, args, kwargs, *, _make_named_inputs: bool = False) -> SigInfo | Any:
    # Unwraps partials and records their arguments
    partials = []
    partial_kwargs = {}

    args = args if args is not None else ()
    kwargs = kwargs if kwargs is not None else {}

    fn_ = fn
    unwrapped: Callable = fn
    while True:
        if not isinstance(fn_, functools.partial):
            break

        partials.append(fn_)

        check(
            len(fn_.args) == 0,
            lambda: f"Support for partials with positional args (like {fn_.args}) is not implemented yet",
            exception_type=NotImplementedError,
        )

        fn_ = fn_.func
        unwrapped = fn_

    # NOTE That the partials are iterated over in REVERSE order because the keywords from later partials override
    #   the keywords from earlier partials
    for p in reversed(partials):
        partial_kwargs.update(p.keywords)

    # TODO Hacky way to extract meta function from Symbol objects
    #   This should probably use a SymbolInterface, or Symbol should define __name__
    if hasattr(fn_, "meta"):
        fn_ = fn_.meta

    # Binds args and kwargs to signature
    sig = inspect.signature(fn_)
    kwargs.update(partial_kwargs)
    ba = sig.bind(*args, **kwargs)

    # Augments arguments with default values
    # NOTE A default value is, for example alpha=1. in a function's signature
    #   These default values are not included in ba.arguments
    # NOTE Partial kwargs take precedence as defaults
    default_dict = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}
    default_dict.update(partial_kwargs)
    args_dict = default_dict | ba.arguments

    # Augments the parameters with positional information
    params_with_indices = {k: (v, idx) for idx, (k, v) in enumerate(sig.parameters.items())}

    # Constructs an object with properties named after the names in the function's signature,
    #   that return their corresponding arguments when accessed
    if _make_named_inputs:

        class NamedBindings:
            pass

        for name, x in args_dict.items():
            setattr(NamedBindings, name, property(lambda self, x=x: x))

        return NamedBindings()

    # Constructs signature information
    # NOTE On this path _make_named_inputs == False, so a SigInfo is created

    # Acquires the name of the function
    # NOTE Not all callables define __name__, including objects that define __call__ and
    #   objects created with functools.partial

    match fn_:
        case functools.partial():
            name = fn_.func.__name__
        case _:
            if hasattr(fn_, "__name__"):
                name = fn_.__name__
            elif hasattr(fn_, "__call__"):
                raise NotImplementedError(f"Can't yet create a signature for a callable object, like {type(fn_)}")
            else:
                raise ValueError(f"Don't know how to extract a signature from type {type(fn_)}")

    si = SigInfo(name)

    for name, x in args_dict.items():
        p, idx = params_with_indices[name]
        pkind = p.kind

        if pkind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            si.args.append((x, idx, name))
        elif pkind is Parameter.VAR_POSITIONAL:
            si.varargs = (name, x)
        elif pkind is Parameter.KEYWORD_ONLY:
            si.kwargs[name] = x
        elif pkind is Parameter.VAR_KEYWORD:
            si.varkwargs = (name, x)
        else:
            raise ValueError(f"Unexpected parameter kind {pkind}")

    si.args = sorted(si.args, key=lambda x: x[1])
    si.args = tuple((x[2], x[0]) for x in si.args)

    si.defaultdict = default_dict
    si.unwrapped_fn = unwrapped
    return si
