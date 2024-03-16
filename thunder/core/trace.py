import os
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Any, Tuple, Type, Dict, List, Union
from collections.abc import Callable
from collections.abc import Sequence, Hashable
import string
from numbers import Number
import inspect
import functools
from types import ModuleType

import thunder
import thunder.core.codeutils as codeutils
import thunder.core.baseutils as baseutils
from thunder.core.baseutils import ProxyInterface, BoundSymbolInterface
import thunder.core.devices as devices
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.codeutils import ContextObject


# TODO see issue "Improve TraceProvenance"
#   Make this more interesting / printer better -- maybe let
#   practitioners acquire the pass callable so they can replicate the pass?
# This class is intended to describe how the trace was constructed
#   In particular, it's intended to describe the pass that created the trace,
#   and maybe have some data about the what the trace did. For example,
#   it might contain data about how many operations a dce pass removed.
class TraceProvenance:
    # NOTE "pss" and not "pass" because "pass" is a keyword
    def __init__(self, pss: str):
        self.pss = pss

    def __repr__(self) -> str:
        return f"# Constructed by {self.pss}"


# TODO Should traces be BoundSymbols?
# TODO issue "Create a mechanism for freezing TraceCtx objects"
#   Add validation that a constant is never assigned to / reassigned
#   Possibly separate the ideas of a trace -- a series of scopes containing bound symbols --
#   and a TraceCtx, which can produce new traces
#   In particular, when running passes the trace is effectively reduced to a list of bound_symbols
#   ... but maybe we still need the naming context?
# TODO Allow the function signature to be modified by transforms
class TraceCtx:
    def __init__(self, fn: None | Callable = None, *, prologue=None, is_prologue: bool = False):
        self.fn: None | Callable = fn

        self._prologue = prologue
        self._is_prologue: bool = is_prologue

        self.args = None
        self.kwargs = None

        self.bound_symbols: list[BoundSymbolInterface] = []
        self.scopes = [self.bound_symbols]

        self.name_ctr = 0
        self.obj_name_ctr = 0
        self.names = set()

        self._object_ctx: dict[int, ContextObject] = {}

        self._provenance: TraceProvenance | None = None

        # Objects related to unpacking
        # NOTE While unpacking inputs we also sometimes want to convert those inputs
        #   For example, when converting NumPy arrays we want to convert them to torch tensors
        #   Having this happen while unpacking is tricky to read, so we want the conversion
        #   to happen when the unpacking is finished.
        #   There are different ways this could be accomplished. One idea is that
        #   an association between proxies and the original inputs could be maintained
        #   so that the proxies could be reviewed and conversions determined later.
        #   Another idea (implemented here) is that we can maintain a list of
        #   deferred conversions to be handled when unpacking is finished.
        #   This requires the trace having the state of unpacking, and it requires
        #   conversions which may occur when unpacking to state that they should
        #   be deferred.
        # TODO Feel free to revisit this pattern
        self._unpacking = False
        self._post_unpack = []

        # NOTE SigInfo is here because we only want to construct one SigInfo for the trace
        self._siginfo = None

        # TODO Improve "freezing" traces
        self._complete = False

        self._any_future_tensors = False

    @property
    def prologue(self):
        return self._prologue

    @property
    def is_prologue(self):
        return self._is_prologue

    #
    # Methods related to the trace's signature
    #
    def siginfo(self) -> codeutils.SigInfo:
        if self._siginfo is not None:
            return self._siginfo

        assert self.fn is not None, "Can't provide siginfo for a trace without a function signature"
        self._siginfo = codeutils.get_siginfo(self.fn, self.args, self.kwargs)
        return self._siginfo

    #
    # Methods for getting and setting trace metadata (like the provenance)
    #

    def get_provenance(self) -> None | TraceProvenance:
        return self._provenance

    def set_provenance(self, provenance: TraceProvenance) -> None:
        self._provenance = provenance

    #
    # Methods related to name construction
    #

    def add_name(self, name: str) -> None:
        baseutils.check(
            name not in self.names,
            lambda: f"Trying to add the name {name} to a trace, but that name is already used",
        )
        self.names.add(name)

    def has_name(self, name: str) -> bool:
        return name in self.names

    def _gen_name(self, ctr):
        return ctr

    def _make_name(self, *, prefix: str | None = None, is_object_name: bool = False, obj: Any | None = None) -> str:
        name: str
        while True:
            if is_object_name:
                baseutils.check(
                    prefix is None,
                    lambda: f"prefix was {prefix}, but prefixes are not supported when {is_object_name=}",
                )
                name = self._gen_name(self.obj_name_ctr)
                self.obj_name_ctr += 1
                if obj is None:
                    name = f"_object_{name}"
                else:
                    name = f"_{baseutils.print_type(type(obj), with_quotes=False)}_{name}"
            else:
                ctr = self._gen_name(self.name_ctr)
                prefix = "_" if prefix is None else prefix
                name = f"{prefix}{ctr}"
                self.name_ctr += 1

            if name not in self.names:
                break

        self.names.add(name)
        return name

    # Constructs and records new name -- or, if name is not None --
    #   just records the given name
    def make_name(self, name: str | None = None, *, prefix: str | None = None) -> str:
        if name is not None:
            self.names.add(name)
            return name

        return self._make_name(prefix=prefix)

    def make_object_name(self, x: Any) -> str:
        return self._make_name(is_object_name=True, obj=x)

    #
    # Methods related to adding inputs, outputs, bound symbols, and manipulating scopes
    #

    # NOTE This "unwraps" singleton outputs
    # TODO Consider revisiting this behavior
    @property
    def output(self) -> Any:
        from thunder.core.prims import PrimIDs

        for bsym in self.bound_symbols:
            if bsym.sym.id is PrimIDs.RETURN:
                rval = bsym.args

                if len(rval) == 1:
                    return rval[0]

                return rval

        assert False, "Trace has no return, and so no output"

    def mark_complete(self) -> None:
        self._complete = True

    def add_bound_symbol(self, bsym: BoundSymbolInterface) -> None:
        self.bound_symbols.append(bsym)

    def push_scope(self, scope: list) -> None:
        self.scopes.append(scope)

    def pop_scope(self) -> list:
        return self.scopes.pop()

    def peek_scope(self) -> list | None:
        if len(self.scopes) == 0:
            return None

        return self.scopes[-1]

    #
    # Unpacking methods
    #

    def unpacking(self) -> None:
        self._unpacking = True

    # Calls all deferred functions
    def unpacked(self) -> None:
        self._unpacking = False
        for fn in self._post_unpack:
            fn()
        self._post_unpack = []

    # Defers the function if unpacking, and calls it immediately if not unpacking
    def post_unpack(self, fn: Callable) -> None:
        if not self._unpacking:
            fn()
        else:
            self._post_unpack.append(fn)

    #
    # Methods for querying objects
    #

    # TODO Revise this -- currently all non-proxies are considered const
    def is_constant(self, x: Any) -> bool:
        if isinstance(x, ProxyInterface):
            return False

        return True

    #
    # Methods related to constructing a Python program representing the trace
    #

    def add_object(self, x: Any) -> ContextObject:
        key = id(x)

        prev = self._object_ctx.get(key, None)

        if prev is None:
            val = ContextObject(self.make_object_name(x), x)
            self._object_ctx[key] = val
            return val

        return prev

    # TODO Ensure no names are clobbered with different values
    # Gathers the import and object contexts
    # The import context maps expected module names to the actual modules; these must be imported with those names
    #   to run the program properly
    # The object context maps names to Python objects; these are required to represent Python objects which
    #   cannot be readily imported or serialized as (short, readable) strings -- for example an arbitrary
    #   user-defined object
    def _gather_ctxs(self) -> tuple[dict, dict, dict]:
        import_ctx = {}
        call_ctx = {}
        object_ctx = {}

        # Gathers from BoundSymbols
        for bsym in self.bound_symbols:
            bsym_import_ctx, bsym_call_ctx, bsym_object_ctx = bsym.gather_ctxs()
            import_ctx.update(bsym_import_ctx)
            call_ctx.update(bsym_call_ctx)
            object_ctx.update(bsym_object_ctx)

        # Updates the import context to use nicer shortnames
        import_ctx_shortnames = {codeutils.module_shortname(k): v for k, v in import_ctx.items()}

        return import_ctx_shortnames, call_ctx, object_ctx

    # TODO Ensure no names are clobbered with different values
    # Practitioner-facing command that just returns the a single Python context,
    #   combining both the import and object contexts
    def python_ctx(self) -> dict:
        # Gathers the BoundSymbol's import, call, and object contexts
        import_ctx, call_ctx, object_ctx = self._gather_ctxs()

        # Gathers the signature's import and object contexts
        # NOTE The result of the prettyprint is ignored, this is just interested in the context updates
        si = self.siginfo()
        _ = si.prettyprint(trace=self, import_ctx=import_ctx, object_ctx=object_ctx)

        # Creates common ctx
        import_ctx.update(call_ctx)
        import_ctx.update(object_ctx)

        return import_ctx

    # TODO Account for multi-line signatures
    # TODO issue "Add type annotations to Python function produced by traces"
    #   Consider extending the signature with type information, in particular the
    #   the type information of the return value might be interesting
    def python(self, *, print_depth: int = 1) -> str:
        token = set_tracectx(self)

        try:
            # Acquires ctx and imports from the BoundSymbols...
            import_ctx, call_ctx, object_ctx = self._gather_ctxs()

            # ... and from the signature
            if self._siginfo is None and self.fn is None:
                signature_str = f"# No signature available"
            else:
                si = self.siginfo()
                signature_str = si.prettyprint(trace=self, import_ctx=import_ctx, object_ctx=object_ctx)

            # Constructs program strings
            program = []

            # Prints provenance (if any) first
            if self._provenance is not None:
                provenance_str = f"{str(self._provenance)}"
                program.append(provenance_str)

            # NOTE torch is explicitly imported because we always run in the no_grad() ctx (see below)
            import torch

            import_ctx["torch"] = torch

            # Prints imports, sorted by name

            def keyfn(class_or_module: type | ModuleType) -> str:
                if isinstance(class_or_module, ModuleType):
                    return class_or_module.__name__
                return class_or_module.__module__

            name: str
            class_or_module: type | ModuleType
            for name, class_or_module in sorted(import_ctx.items(), key=lambda x: keyfn(x[1])):
                import_str: str

                # Handles class imports
                if not isinstance(class_or_module, ModuleType):
                    cls: type = class_or_module
                    import_str = f"from {cls.__module__} import {cls.__name__}"
                else:
                    # class_or_module is a module
                    module: ModuleType = class_or_module
                    if module.__name__ == name:
                        import_str = f"import {module.__name__}"
                    else:
                        import_str = f"import {module.__name__} as {name}"
                program.append(import_str)

            program.append("from thunder.executors.torchex import no_autocast")
            # Separates imports from the function for readability
            if len(import_ctx) > 0:
                program.append("")

            # Prints the signature and the no_grad context (for when calling torch operations)
            program.append("@torch.no_grad()")
            # Prints the signature and the no_autocast context
            program.append("@no_autocast()")
            program.append(signature_str)

            # TODO Print objects from context
            # Prints constants (if any) upfront
            # constants = tuple(om for om in self._object_meta_map.values() if om.is_constant)
            # if len(constants) > 0:
            #     const_comment_str = f"{indent}# Initializes constants"
            #     program.append(const_comment_str)
            # for c in constants:
            #     constant_python = c.python(indent=1)
            #     program.extend(constant_python)

            # Separates constants from operations
            # if len(constants) > 0:
            #     program.append("")

            # Prints operations
            for bsym in self.bound_symbols:
                lines = bsym.python(indent=1, print_depth=print_depth)
                program.extend(lines)

            python = "\n".join(program)

            return python
        finally:
            reset_tracectx(token)

    # Returns a Python callable that executes the trace
    # TODO issue "Create a mechanism for freezing TraceCtx objects"
    #   Create a mechanism for freezing traces and cache the compilation
    def python_callable(self, *, global_dicts: None | dict = None) -> Callable:
        python_str: str

        # Writes the program to allow it to be edited before execution
        path: None | str = _get_execution_file()
        if path is not None:
            f = open(path, "w")
            f.write(self.python())
            f.close()

            input(f"Trace written to {os.path.realpath(path)}  Press Any key to execute it")

            with open(path) as file:
                python_str = file.read()
        else:
            python_str = self.python()

        ctx = self.python_ctx()
        if global_dicts is not None:
            ctx["__global_dicts"] = global_dicts
        ctx["__function_obj"] = self.fn
        ctx["thunder"] = thunder

        callable = baseutils.compile_and_exec(
            self.siginfo().name, python_str=python_str, program_name=f"thunder.{self.siginfo().name}", ctx=ctx
        )
        return callable

    def __repr__(self) -> str:
        return self.python(print_depth=-1)


# Constructs a new trace by shallow copying parts of an existing trace
# NOTE Bound symbols and provenance are not copied
def from_trace(trace: TraceCtx) -> TraceCtx:
    t = TraceCtx(trace.fn, prologue=trace.prologue)
    t.args = trace.args
    t.kwargs = trace.kwargs

    t.name_ctr = trace.name_ctr
    t.obj_name_ctr = trace.obj_name_ctr
    t.names = trace.names

    t._siginfo = trace._siginfo
    return t


#
# Functions related to setting, getting, and resetting the current tracing context
#
_tracectx = ContextVar("tracectx")


def set_tracectx(ctx):
    """Sets the current trace context."""

    return _tracectx.set(ctx)


def get_tracectx() -> None | TraceCtx:
    """Gets the current trace context, returning None if there is no current trace context."""

    try:
        t = _tracectx.get()
        return t
    except LookupError:
        pass

    return None


def reset_tracectx(token):
    """Resets the tracing context."""

    _tracectx.reset(token)


def is_tracing() -> bool:
    return get_tracectx() is not None


def maybe_start_trace(fn) -> tuple[bool, Any | None, TraceCtx]:
    trace = get_tracectx()
    if trace is None:
        trace = TraceCtx(fn)
        tok = set_tracectx(trace)
        return True, tok, trace

    return False, None, trace


def maybe_reset_trace(started: bool, tok: Any) -> None:
    if not started:
        return

    reset_tracectx(tok)


@contextmanager
def tracectx(trace: None | TraceCtx):
    tok = set_tracectx(trace)
    try:
        yield
    finally:
        reset_tracectx(tok)


@contextmanager
def detached_trace():
    """Context manager that detaches the current trace.

    This is useful for code that should not be traced, but that is called from
    traced code. For example, if you have a function that is traced, and that
    function calls a function that should not be traced, you can use this context
    manager to detach the current trace before calling the function that should
    not be traced.
    """
    outer_trace = get_tracectx()
    outer_names = set()
    if outer_trace is not None:
        outer_names = outer_trace.names
    trace = TraceCtx(None)
    trace.names.update(outer_names)
    trace_token = set_tracectx(trace)
    yield
    reset_tracectx(trace_token)


#
# Helpers for querying properties of the trace
#


def get_prologue() -> None | TraceCtx:
    return get_tracectx().prologue


#
# Variable helpers
#


class VariableInterface:
    @property
    def name(self):
        pass


class TraceVariable:
    def __init__(self, obj: VariableInterface):
        self.proxy = obj
        self.name = obj.name


def wrap_in_trace_variable(obj):
    if isinstance(obj, VariableInterface):
        return TraceVariable(obj)
    return obj


#
# Functions related to the execution file callback
#
# When set, this will write traces to a file, then load the source of that file, before executing them

_execution_file = ContextVar("_execution_file", default=None)


# TODO Consider generalizing these to take a file object, too
def _set_execution_file(path: str) -> None:
    _execution_file.set(path)


def _get_execution_file() -> None | str:
    return _execution_file.get()
