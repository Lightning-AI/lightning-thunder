from __future__ import annotations

import functools
import inspect
import sys
from contextvars import ContextVar
from contextlib import contextmanager
from itertools import chain
from types import ModuleType

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Type, Tuple, TYPE_CHECKING
from collections.abc import Callable, Iterable
from collections.abc import Sequence

import thunder.core.baseutils as baseutils
import thunder.core.codeutils as codeutils
from thunder.core.codeutils import Printable
from thunder.core.baseutils import BoundSymbolInterface, ProxyInterface
from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
from thunder.core.proxies import Proxy, NumberProxy, variableify, CollectionProxy

from thunder.core.trace import (
    get_tracectx,
    maybe_reset_trace,
    maybe_start_trace,
    VariableInterface,
    wrap_in_trace_variable,
)

#
# Support for querying "traceable" functions
#   A "traceable" function is one that doesn't require interpretation for thunder to translate
#   to a thunder program. Put another way, these functions have no "sharp edges." Every Symbol
#   must be a traceable function.
# TODO RC1 Consider if we need to provide a mechanism to register operations as traceable
#   (like clang operations and methods on proxies)


def is_traceable(fn: Callable) -> bool:
    return isinstance(fn, Symbol)


if TYPE_CHECKING:
    from thunder.core.prims import OpTags


_bsym_header = ContextVar("bsym_header", default="")


@contextmanager
def bsym_header(header: str):
    """Sets the current header for BoundSymbols."""

    token = _bsym_header.set(header)
    try:
        yield
    finally:
        _bsym_header.reset(token)


# TODO THIS IS BROKEN. We need to pass the args and kwargs here, but with placeholders pointing to the Python ctx
#   where appropriate. That way the other objects can be printed correctly. Otherwise we can't disambiguate between
#   actual strings, which must be enclosed in quotes, and the placeholders pointing to the ctx, which are currently
#   passed as strings.
# NOTE Assumes the outputs of symbols are proxies or collections of proxies
def default_python_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str | Iterable[str]:
    arg_str = (
        ""
        if (arg_printables is None or len(arg_printables) == 0)
        else ", ".join(codeutils.prettyprint(x) for x in arg_printables)
    )
    kwarg_str: str

    if len(kwarg_printables) == 0:
        kwarg_str = ""
    else:
        kwarg_str = ", ".join(f"{k}={codeutils.prettyprint(v)}" for k, v in kwarg_printables.items())

    result_str: str
    if bsym.output is None or (codeutils.is_collection(bsym.output) and len(bsym.output) == 0):
        result_str = ""
    else:
        result_str = f"{codeutils.prettyprint(out_printables, literals_as_underscores=True)} = "

    # Creates a comment describing the output
    comment_str = ""
    if isinstance(bsym.output, Proxy):
        comment_str = f"  # {codeutils.prettyprint(out_printables, with_type=True)}"

    s = f"{result_str}{bsym.name_with_module()}({arg_str}{', ' if (len(arg_str) > 0 and len(kwarg_str) > 0) else ''}{kwarg_str}){comment_str}"

    if bsym.header:
        header_lines = (
            bsym.header
            if isinstance(bsym.header, Sequence) and not isinstance(bsym.header, str)
            else bsym.header.splitlines()
        )
        header_lines = (f"# {line}" for line in header_lines)
        return chain(header_lines, [s])

    return s


# A symbol represents a function and how it can be transformed

# name is a string name for the operation
# meta should use thunder.jit functions to evaluate the function;
#   it will be called with thunder.jit proxies
# id is an optional value to use when translating the function to executors
# is_prim should be True if the Symbol represents a thunder.jit primitive
# python_printer is a function that will produce valid Python for calling the
#   operation; this can usually be set to None, in which case the default python
#   printer will be used for the Symbol. Symbols that control their own printing
#   are typically unpacking datastructures, and producing particular, structured
#   outputs.
# python_impl defines the Python implementation of the operation, if
#   available


# TODO Improve annotations
@dataclass(**baseutils.default_dataclass_params)
class Symbol:
    name: str
    meta: Callable | None = None
    id: None | Any = None
    is_prim: bool = False
    tags: None | list[Any] = None
    is_fusion: bool = False
    python_printer: Callable = default_python_printer
    _module: None | type | ModuleType = None
    _hash: int | None = None
    executor: None | Any = None
    python_impl: None | Callable = None
    _print_as_impl: bool = False  # If not None, w

    # An optional postprocessing function to modify the bound symbol resulting from bind()
    _bind_postprocess: None | Callable = None

    @property
    def __name__(self):
        return self.name

    # Relies on the assumption that symbols with the same non-None ids are equal.
    # If both IDs are none, then symbols with the same name and module are equal.
    def __hash__(self) -> int:
        if self._hash is None:
            object.__setattr__(self, "_hash", hash((self.name, self._module, self.id)))
        return self._hash

    # Symbols are equal if they have the same id (if present),
    #   or name and module if id is not present.
    # NOTE This does not check if the two Symbol objects have the same meta, impls,
    #   if they are prims, if they have the same executor, or if they have the same printer, etc.
    # TODO Catch and warn when people construct symbols violating these assumptions
    def __eq__(self, other: Symbol) -> int:
        if not isinstance(other, Symbol):
            return False
        if self.id is None and other.id is None:
            return (self.name, self._module) == (other.name, other._module)
        return self.id == other.id

    @property
    def module(self) -> None | ModuleType:
        # Identifies the module needed to call the function
        #   The module can be specified explicitly by setting the _module attribute,
        #   or it can be inferred by inspecting the module of the meta function,
        #   which is assumed to be defined in the correct module.
        # NOTE Functions created through partial would report originating from the
        #   functools module, so those are unwrapped
        # TODO Is there a better way to identify modules?
        # TODO Maybe let the module be specified directly, too?
        # TODO Test that this works with decorators -- only decorators with @wraps?
        #   Decorators defined in different modules?
        # TODO This aggressively unwraps partials and wrapped functions, but if the user is calling
        #   an operation that is itself a partial or a wrapper they may expect to see that call

        module = self._module
        if self.is_fusion:
            return None
        if module is not None:
            result = module
        elif self.meta is None:
            result = None
        else:
            fn_ = self.meta
            while isinstance(fn_, functools.partial) or hasattr(fn_, "__wrapped__"):
                if isinstance(fn_, functools.partial):
                    fn_ = fn_.func
                else:
                    fn_ = fn_.__wrapped__
            result = inspect.getmodule(fn_)
        return result

    def __repr__(self) -> str:
        return f"[Symbol name={self.name}]"

    def normalize(self, *args, **kwargs):
        si = inspect.signature(self.meta)
        ba = si.bind(*args, **kwargs)
        ba.apply_defaults()
        return ba.args, ba.kwargs

    # TODO Remove _call_ctx
    def bind(self, *args, output, subsymbols=(), _call_ctx: None | dict = None, **kwargs) -> BoundSymbol:
        if self.meta is not None:
            args, kwargs = self.normalize(*args, **kwargs)

        b = BoundSymbol(
            self,
            args=args,
            kwargs=kwargs,
            output=output,
            subsymbols=subsymbols,
            header=_bsym_header.get(),
            _call_ctx=_call_ctx,
        )
        if self._bind_postprocess:
            self._bind_postprocess(b)
        return b

    def __call__(self, *args, **kwargs):
        trace = get_tracectx()

        baseutils.check(
            trace is not None,
            lambda: f"Attempting to execute outside of a tracing context, which is not supported",
            exception_type=NotImplementedError,
        )

        baseutils.check(not trace._complete, lambda: f"Trying to add {self} to a trace that is complete!")
        result: Any
        subsymbols = []
        if self.is_prim:
            symbols_list = trace.peek_scope()
            # NOTE Operations within primitives are not recorded
            # TODO Revisit this when adding constraints -- meta operations can originate constraints
            if symbols_list is None:
                return self.meta(*args, **kwargs)

            trace.push_scope(None)
            result = self.meta(*args, **kwargs)
            trace.pop_scope()
        else:
            trace.push_scope(subsymbols)
            result = self.meta(*args, **kwargs)
            trace.pop_scope()

        bsym = self.bind(*args, **kwargs, output=result, subsymbols=subsymbols)
        symbols_list = trace.peek_scope()

        baseutils.check(
            symbols_list is not None,
            lambda: f"A symbol {self} was called while processing a primitive",
            exception_type=AssertionError,
        )

        symbols_list.append(bsym)
        return result


# A symbol, arguments (and kwarguments), output, and sub-symbols
# args is a sequence of the arguments
# kwargs is a dict of the kwargs
# outputs is a sequence of the outputs
# subsymbols is a sequence of symbols that compose this particular symbol
# _object_ctx can be provided to specify a name -> Python object binding required to compile
#   the BoundSymbol
# NOTE This could also provide an _import_ctx to specify the import context directly
# TODO: We use @functools.cached_property extensively here, but it only works because
# BoundSymbolInterface causes our dataclass to inherit a __dict__ property.
# This gives us the additional flexibility, but probably isn't what we want, because it
# sort of defeats the purpose of having a dataclass in the first place.
# NOTE It is assumed that the (non-cached) properties of a BoundSymbol are immutable, except for the _executor annotation
@dataclass
class BoundSymbol(BoundSymbolInterface):
    sym: Symbol
    args: tuple
    kwargs: dict
    output: Any
    subsymbols: Sequence[BoundSymbol] = ()

    # Header is a string that may be printed before the symbol
    header: str | list[str] = ""

    _call_ctx: None | dict[str, Any] = None

    _import_ctx: dict = field(default_factory=dict)
    _object_ctx: dict = field(default_factory=dict)
    _executor: None | Any = None

    # The line number of the bound symbol
    # NOTE This is only intended for internal use, and should be set explicitly on all BoundSymbols
    #   being analyzed before use, since it may have been set by previous passes, too
    _line_no: None | int = None

    # TODO: Should we do input validation in post_init?
    # For example, making sure kwargs is empty dict instead on None.
    def __post_init__(self):
        self.args = tuple(self.args)

    # Constructs a new BoundSymbol with default values taken from this BoundSymbol
    #   Override values can be specified as kwargs
    # Issue -- Provide a pattern for updating subsymbols when swapping outputs
    #   Maybe this can also just swap one set of symbols for another?
    #   Consider adding verification that the new and old output have the same metadata
    def from_bsym(self, **kwargs) -> BoundSymbol:
        self_kwargs = {
            "sym": self.sym,
            "args": self.args,
            "kwargs": self.kwargs,
            "output": self.output,
            "subsymbols": self.subsymbols,
            "header": self.header,
            "_call_ctx": self._call_ctx,
            "_import_ctx": self._import_ctx,
            "_object_ctx": self._object_ctx,
            "_executor": self._executor,
        }

        self_kwargs.update(kwargs)

        return BoundSymbol(**self_kwargs)

    # NOTE coll must be a Container of "variableified" proxies
    def has_input(self, coll) -> bool:
        for x in self.flat_args:
            if not isinstance(x, Proxy):
                continue

            vx = variableify(x)
            if vx in coll:
                return True

        return False

    # Produces a new BoundSymbol with the inputs swapped as described by swap_map,
    #   which maps from variablified proxies to swap to the proxies to swap them with
    # TODO Consider adding a check that the swap is to another proxy with the same metadata
    # TODO Consider preserving the BoundSymbol if does not use any of the proxies in the swapmap
    def from_bsym_swap_proxies(
        self,
        swap_map: dict[VariableInterface, Proxy],
        skip_inputs: bool = False,
        skip_output: bool = False,
        skip_subsymbols: bool = False,
    ) -> BoundSymbol:
        """Create a new :class:`BoundSymbol` with its inputs, output, and subsymbols updated with ``swap_map``.

        This replaces :class:``Proxy``s, e.g. :class:`TensorProxy`, of inputs and output
        with another ones already seen recorded in ``swap_map`` (``swap_map`` maps variableified
        :class:``Proxy`` to an existing one generated by the same expression), and do the same to subsymbols.

        Turning on some of positional arguments following ``swap_map`` can be useful, for example,
        when querying right-hand side expression of a :class:`BoundSymbol` in :func:`thunder.executors.passes.CSE`.

        Args:
            skip_inputs:
            skip_output:
            skip_subsymbols:

        Returns:
            :class:`BoundSymbol`
        """
        if len(swap_map) == 0:
            return self

        def swap(c):
            flats, spec = tree_flatten(c)

            swapped = []
            for fa in flats:
                visited: set[VariableInterface] = set()
                if isinstance(fa, CollectionProxy):
                    fa.coll = tree_map(swap, fa.collection())
                if isinstance(fa, Proxy):
                    ovfa = variableify(fa)
                    vfa = ovfa
                    while vfa in swap_map:
                        baseutils.check(
                            vfa not in visited, lambda: f"Detected a cycle while swapping; the cycle includes {visited}"
                        )
                        visited.add(vfa)

                        fa = swap_map[vfa]
                        vfa = variableify(fa)

                swapped.append(fa)

            return tree_unflatten(swapped, spec)

        nargs = swap(self.args) if not skip_inputs else self.args
        nkwargs = swap(self.kwargs) if not skip_inputs else self.kwargs

        new_output = swap(self.output) if not skip_output else self.output

        subsymbols: list[BoundSymbol]
        if not skip_subsymbols:
            subsymbols = []
            for bsym in self.subsymbols:
                subsymbols.append(bsym.from_bsym_swap_proxies(swap_map, skip_inputs, skip_output, skip_subsymbols))
        else:
            subsymbols = self.subsymbols

        return self.from_bsym(args=nargs, kwargs=nkwargs, output=new_output, subsymbols=subsymbols)

    # NOTE Making these cached properties relies on the assumption that the inputs to and output of a BoundSymbol
    #   are immutable

    @functools.cached_property
    def flat_args_and_spec(self):
        return tree_flatten((self.args, self.kwargs))

    @functools.cached_property
    def flat_args(self):
        flatargs, _ = self.flat_args_and_spec
        return flatargs

    @functools.cached_property
    def flat_proxy_args(self) -> tuple[Proxy, ...]:
        return tuple(x for x in self.flat_args if isinstance(x, Proxy))

    @functools.cached_property
    def flat_variableified_proxy_args(self) -> tuple[Proxy, ...]:
        return tuple(variableify(x) for x in self.flat_args if isinstance(x, Proxy))

    # TODO The performance of these _var_* properties could be improved by reusing the tree spec
    @functools.cached_property
    def _var_args(self):
        return tree_map(variableify, self.args)

    @functools.cached_property
    def _var_kwargs(self):
        return tree_map(variableify, self.kwargs)

    @functools.cached_property
    def flat_outs_and_spec(self):
        return tree_flatten(self.output)

    @functools.cached_property
    def flat_outs(self):
        flatouts, _ = self.flat_outs_and_spec
        return flatouts

    @functools.cached_property
    def flat_proxy_outs(self) -> tuple[Proxy, ...]:
        return tuple(x for x in self.flat_outs if isinstance(x, Proxy))

    @functools.cached_property
    def flat_variableified_proxy_outs(self) -> tuple[Proxy, ...]:
        return tuple(variableify(x) for x in self.flat_outs if isinstance(x, Proxy))

    @functools.cached_property
    def _var_output(self):
        return tree_map(variableify, self.output)

    @property
    def _out_printables(self):
        trace = get_tracectx()
        return codeutils.to_printable(trace, self.output, import_ctx=self._import_ctx, object_ctx=self._object_ctx)

    @property
    def _arg_printables(self):
        trace = get_tracectx()
        return tuple(
            codeutils.to_printable(trace, x, import_ctx=self._import_ctx, object_ctx=self._object_ctx)
            for x in self.args
        )

    @property
    def _kwarg_printables(self):
        trace = get_tracectx()
        return codeutils.to_printable(trace, self.kwargs, import_ctx=self._import_ctx, object_ctx=self._object_ctx)

    # BoundSymbols are hashable and comparable by identity
    # This is necessary for using them as keys in a dict or set members
    # See dce in thunder/executors/passes.py for the usage.
    # TODO: if kwargs were hashable (frozendict), we could use a tuple of (sym, args, kwargs, output) as the key
    #       and avoid the need for this. It would also mean we need to deep convert all the kwargs' contents to be hashable as well.
    @functools.cached_property
    def _hash(self) -> int:
        try:
            return hash((self.sym, self._var_args, self._var_output, len(self.kwargs)))
        except:
            # Since args / output may contain unhashable types
            return id(self)

    def __hash__(self):
        return self._hash

    # TODO Deal with the contents of kwargs in __eq__ and __hash__.
    def __eq__(self, other):
        if not isinstance(other, BoundSymbol):
            return False
        if self is other:
            return True
        if len(self.kwargs) > 0 or len(other.kwargs) > 0:
            return False

        return (self.sym, self._var_args, self._var_output) == (other.sym, other._var_args, other._var_output)

    def rhs(self):
        return BoundSymbolRHS(self)

    # TODO Document contexts
    def import_ctx(self):
        # NOTE This initializes the context
        self._out_printables, self._arg_printables, self._kwarg_printables
        if self.sym is not None and self.sym.python_impl is not None:
            # NOTE BoundSymbols of Symbols with a python_impl defined are run in Python, and are assumed
            #   to not need any imports to run properly, unless _import_prims is True
            if self.sym._print_as_impl:
                module_name = self.sym.module.__name__
                import_ctx = {module_name: self.sym.module}
            else:
                import_ctx = {}
        elif self._call_ctx is not None:
            # NOTE If the call ctx was specified directly, then no import is needed to call the function
            import_ctx = {}
        else:
            # BoundSymbols of Symbols without Python implementations (either because they
            #   have Python implementations or defined call ctxs) are assumed to need
            #   a module import to run properly
            module_name = self.sym.module.__name__
            import_ctx = {module_name: self.sym.module}

            # TODO Include the other modules on the path?
            # Also includes the root module of this (potential) submodule
            if "." in module_name:
                root_name = module_name.split(".")[0]
                import_ctx[root_name] = sys.modules[root_name]

        self._import_ctx.update(import_ctx)
        return self._import_ctx

    def object_ctx(self):
        # NOTE This initializes the context
        self._out_printables, self._arg_printables, self._kwarg_printables

        return self._object_ctx

    # def set_output(self, output: Any):
    #     self.output = output

    def name_with_module(self):
        # Short-circuits if the symbol has no associated module
        if self.sym.module is None or self._call_ctx is not None:
            return f"{self.sym.name}"

        module_name = codeutils.module_shortname(self.sym.module.__name__)
        fn_name = self.sym.name

        # TODO There is almost certainly a better way to model this -- maybe by passing the
        #   function?
        if self.sym._print_as_impl:
            fn_name = f"{fn_name}_impl"
        return f"{module_name}.{fn_name}"

    def _get_call_ctx(self):
        return self._call_ctx if self._call_ctx is not None else {}

    # TODO Consider if this should gather contexts recursively
    #   Currently this means that imports required for the subsymbols won't
    #   be printed, even though they're used in the comments
    def gather_ctxs(self) -> tuple[dict, dict, dict]:
        return self.import_ctx(), self._get_call_ctx(), self.object_ctx()

    def _get_lines(self, indent: int, commented: bool = False):
        lines = []

        s = self.sym.python_printer(self, self._out_printables, self._arg_printables, self._kwarg_printables)

        comment = "# " if commented else ""

        if isinstance(s, str):
            lines.append(f"{codeutils.indent_string(indent)}{comment}{s}")
        else:
            for line in s:
                lines.append(f"{codeutils.indent_string(indent)}{comment}{line}")
        return lines

    def python(self, indent: int, commented: bool = False, print_depth: int = 1) -> list[str]:
        lines = []

        # Checks if this symbol is too "deep" to be printed
        if print_depth == 0:
            return lines

        my_lines = self._get_lines(indent, commented=commented)
        lines.extend(my_lines)

        for ssym in self.subsymbols:
            ssym_lines = ssym.python(indent + 1, commented=True, print_depth=(print_depth - 1))
            lines.extend(ssym_lines)

        return lines

    # TODO Revisit this in the broader context of what's const
    @functools.cached_property
    def are_all_args_constant(self):
        """Returns True if all arguments are constant (i.e. not Variables)."""
        return not any(isinstance(arg, VariableInterface) for arg in self.flat_args)

    def __repr__(self) -> str:
        return "\n".join(self.python(indent=0, print_depth=-1))


def gather_tags(bsym: BoundSymbol) -> set[OpTags]:
    tags = set(bsym.sym.tags) if bsym.sym.tags is not None else set()

    for sbsym in bsym.subsymbols:
        tags |= gather_tags(sbsym)

    return tags


def has_tags(bsym: BoundSymbol, tags: set[OpTags]) -> bool:
    """:obj:`True` if `bsym` and its subsymbols has any of ``tags``."""
    return not tags.isdisjoint(gather_tags(bsym))


# NOTE: A wrapper class that hashes and equates only the right hand side of a BoundSymbol.
# That is to say, its symbol, args, and kwargs, but not its output.
# The intent is that this will be useful in writing a common subexpression elimination pass, beacuse
# it will allow dictionary lookups to find equivalent BoundSymbols.
@dataclass(**baseutils.default_dataclass_params)
class BoundSymbolRHS:
    parent: BoundSymbol
    _hash: int | None = None

    def _do_hash(self) -> int:
        if self.parent.kwargs and len(self.parent.kwargs) > 0:
            return id(self)
        try:
            return hash((self.parent.sym, self.parent._var_args))
        except:
            return id(self)

    def __hash__(self) -> int:
        if not self._hash:
            object.__setattr__(self, "_hash", self._do_hash())
        return self._hash

    # TODO: Deal with kwargs, in __eq__ and __hash__, just like with BoundSymbol.
    def __eq__(self, other: BoundSymbolRHS) -> bool:
        if not isinstance(other, BoundSymbolRHS):
            return False
        if self.parent is other.parent:
            return True
        if len(self.parent.kwargs) > 0 or len(other.parent.kwargs) > 0:
            return False
        return (self.parent.sym, self.parent._var_args) == (other.parent.sym, other.parent._var_args)
