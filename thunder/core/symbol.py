from __future__ import annotations

import functools
import inspect
import sys
from contextvars import ContextVar
from itertools import chain

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, List, Type, Tuple

import thunder.core.baseutils as baseutils
import thunder.core.codeutils as codeutils
from thunder.core.codeutils import Printable, TrackedObject
from thunder.core.baseutils import BoundSymbolInterface, ProxyInterface
from thunder.core.langctx import get_langctx, get_prim_fwd_langctx
from thunder.core.pytree import tree_flatten, tree_unflatten
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
from thunder.core.proxies import NumberProxy

from thunder.core.trace import (
    get_tracectx,
    maybe_reset_trace,
    maybe_start_trace,
    VariableInterface,
    wrap_in_trace_variable,
)

# NOTE Context variables for eager execution
#   Expected to be set only once
_eagetctx = ContextVar("eagerctx")


def set_eagerctx(ctx):
    """Sets the current eager execution context."""

    return _eagetctx.set(ctx)


# TODO THIS IS BROKEN. We need to pass the args and kwargs here, but with placeholders pointing to the Python ctx
#   where appropriate. That way the other objects can be printed correctly. Otherwise we can't disambiguate between
#   actual strings, which must be enclosed in quotes, and the placeholders pointing to the ctx, which are currently
#   passed as strings.
# NOTE Assumes the outputs of symbols are proxies or collections of proxies
# TODO Review printing names of tracked objects
def default_python_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: Dict[str, Printable]
):
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

    s = f"{result_str}{bsym.name_with_module()}({arg_str}{', ' if (len(arg_str) > 0 and len(kwarg_str) > 0) else ''}{kwarg_str})"
    return s


# A symbol represents a function and how it can be transformed

# name is a string name for the operation
# meta should use lightning.compile functions to evaluate the function;
#   it will be called with lightning.compile proxies
# python_impl defines the Python implementation of the operation, if
#   available
# id is an optional value to use when translating the function to executors
# is_prim should be True if the Symbol represents a lightning.compile primitive
# python_printer is a function that will produce valid Python for calling the
#   operation; this can usually be set to None, in which case the default python
#   printer will be used for the Symbol. Symbols that control their own printing
#   are typically unpacking datastructures, and producing particular, structured
#   outputs.


# TODO Improve annotations
# TODO: add the method define_grad_transform(grad_fwd, grad_bwd)
#   which will accept functions or symbols (and translate functions to symbols automatically)
# TODO: consider other forwards, like number forward and NumPy forward
#   think about how to support those things without importing torch or NumPy
#   --> define eager transform to do this? -- won't faciliate debugging clang or prims, but
#   maybe that's fine? -- Could faciliate debugging prims because invoking a symbol
#   outside a tracing ctx query for the appropriate torch or NumPy operation to run?
#   Maybe expose torch.prims and numpy.prims to deal with this?
@dataclass(**baseutils.default_dataclass_params)
class Symbol:
    name: str
    meta: Optional[Callable]
    python_impl: Optional[Callable] = None
    id: Optional[Any] = None
    is_prim: bool = False
    python_printer: Callable = default_python_printer
    _module: Optional[Any] = None

    @property
    def __name__(self):
        return self.name

    @property
    def module(self):
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
        module = self._module
        if module is not None:
            result = module
        elif self.meta is None:
            result = None
        else:
            module_func = self.meta
            while isinstance(module_func, functools.partial):
                module_func = module_func.func
            result = inspect.getmodule(module_func)
        return result

        # Properties used in transforms (defined later)
        # TODO https://github.com/Lightning-AI/lightning-thunder/issues/326
        #   Remove this from here (think how symbols could be extended with transforms)
        # self.grad_defined = False
        # self.grad_ignored = False
        # self.grad_fwd = None
        # self.grad_bwd = None

    def __repr__(self) -> str:
        return f"[Symbol name={self.name}]"

    def _symbolify(self, x):
        # Preserves None
        if x is None:
            return x

        if isinstance(x, Symbol):
            return x

        # NOTE: assumes x is a callable
        return Symbol(name=x.__name__, meta=x)

    def name_with_module(self):
        # Short-circuits if the symbol has no associated module
        if self.module is None:
            return f"{self.name}"

        module_name = codeutils.module_shortname(self.module.__name__)
        return f"{module_name}.{self.name}"

    # TODO: pick which function goes into the expression
    def bind(self, args, kwargs={}, output=[], subsymbols=[]) -> BoundSymbol:
        b = BoundSymbol(self, args, kwargs, output, tuple(subsymbols))
        return b

    # TODO Restore eager dispatch by tracing and executing a trace
    def __call__(self, *args, **kwargs):
        trace = get_tracectx()

        # NOTE This signals an eager invocation
        if trace is None:
            compile_eager, prims_eager = _eagetctx.get()
            if self.is_prim:
                peager = prims_eager.get_eager_implementation_for(self.id)
                baseutils.check(
                    peager is not None,
                    lambda: f"Couldn't find an eager implementation for {self.name}",
                    exception_type=NotImplementedError,
                )
                return peager(*args, **kwargs)

            ceager = compile_eager(self.meta)
            return ceager(*args, **kwargs)

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

        # TODO Consider a way of expressing the name of the output here
        bsym = self.bind(args, kwargs, result, subsymbols)
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
# TODO Can we make this frozen?
@dataclass(**({} if sys.version_info < (3, 10) else {"slots": True}))
class BoundSymbol(BoundSymbolInterface):
    sym: Symbol
    args: Sequence
    kwargs: dict
    output: Any
    subsymbols: Sequence["BoundSymbol"] = ()
    # TODO Add flat_outs
    _flat_args: Optional[Sequence] = None
    _flat_kwargs: Optional[Sequence] = None
    _flat_outs: Optional[Sequence] = None
    _arg_strings: Optional[str] = None
    _kwarg_strings: Optional[str] = None
    _import_ctx: Optional[Dict[str, Any]] = None
    _call_ctx: Optional[Dict[str, Any]] = None
    _object_ctx: Optional[Dict[str, Any]] = None

    # NOTE _executor, being an implementation detail, must be set after
    #   the BoundSymbol is constructed
    # TODO Maybe the _executor annotation should be modeled differently?
    _executor: Optional = None

    # TODO Document contexts
    def __post_init__(self):
        trace = get_tracectx()

        # Adds the needed import to call the function and initializes the import ctx
        if self._import_ctx is None:
            # BoundSymbols of Symbols with a python_impl defined are run in Python, and are assumed
            #   to not need any imports to run properly
            if self.sym is not None and self.sym.python_impl is not None:
                self._import_ctx = {}
            # NOTE If the call ctx was specified directly, then no import is needed to call the function
            elif self._call_ctx is not None:
                self._import_ctx = {}
            else:
                # BoundSymbols of Symbols without Python implementations are assumed to need
                #   a module import to run properly
                module_name = self.sym.module.__name__
                self._import_ctx = {module_name: self.sym.module}

                # TODO Include the other modules on the path?
                # Also includes the root module of this (potential) submodule
                if "." in module_name:
                    root_name = module_name.split(".")[0]
                    self._import_ctx[root_name] = sys.modules[root_name]

        # Constructs the call ctx
        self._call_ctx = self._call_ctx if self._call_ctx is not None else {}

        # Constructs the object ctx and printables
        # NOTE The object ctx may update the import ctx
        self._object_ctx = {}

        self._flat_args, arg_spec = tree_flatten(self.args)
        self._flat_kwargs, kwarg_spec = tree_flatten(self.kwargs)
        self._flat_outs, out_spect = tree_flatten(self.output)

        # TODO Update this to be "ctx_name" or similar
        def _to_printable(x: Any) -> Any:
            to = trace.get_tracked_object(x)
            is_tracked = isinstance(to, TrackedObject)

            if is_tracked:
                return to

            # TODO This is unsafe if the collection cannot be flattened by tree_flatten
            #   We should probably have tree_flatten throw an exception when that occurs
            if codeutils.is_collection(x):
                flat, spec = tree_flatten(x)
                printables = tuple(_to_printable(f) for f in flat)
                printable = tree_unflatten(printables, spec)
                return printable

            # NOTE In this case the object is not tracked nor a collection, and
            #   it may require an import or additional context to print

            printable, module_info = codeutils.to_printable(trace.make_const_name, to)

            if module_info is not None:
                module_name, module = module_info
                self._import_ctx[module_name] = module

            if isinstance(printable, codeutils.ContextObject):
                self._object_ctx[printable.name] = x

            return printable

        self._out_printables = _to_printable(self.output)
        self._arg_printables = tuple(_to_printable(x) for x in self.args)
        self._kwarg_printables = _to_printable(self.kwargs)

    # BoundSymbols are hashable and comparable by identity
    # This is necessary for using them as keys in a dict or set members
    # See dce in thunder/executors/passes.py for the usage.
    # TODO: if kwargs were hashable (frozendict), we could use a tuple of (sym, args, kwargs, output) as the key
    #       and avoid the need for this.
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    # def set_output(self, output: Any):
    #     self.output = output

    def name_with_module(self):
        return self.sym.name_with_module()

    # TODO Consider if this should gather contexts recursively
    #   Currently this means that imports required for the subsymbols won't
    #   be printed, even though they're used in the comments
    def _gather_ctxs(self) -> Tuple[dict, dict, dict]:
        return self._import_ctx, self._call_ctx, self._object_ctx

    def _get_lines(self, indent: int, commented: bool = False):
        lines = []

        s = self.sym.python_printer(self, self._out_printables, self._arg_printables, self._kwarg_printables)
        commented = "# " if commented else ""
        if isinstance(s, str):
            lines.append(f"{codeutils.indent_string(indent)}{commented}{s}")
        else:
            for line in s:
                lines.append(f"{codeutils.indent_string(indent)}{commented}{line}")

        return lines

    def python(self, indent: int, commented: bool = False, print_depth: int = 1) -> List[str]:
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
    @property
    @functools.lru_cache(maxsize=None)
    def are_all_args_constant(self):
        """Returns True if all arguments are constant (i.e. not Variables)."""
        return not any(isinstance(arg, VariableInterface) for arg in self._flat_args)

    def __repr__(self):
        trace = get_tracectx()
        return "\n".join(self.python(indent=0, print_depth=-1))
