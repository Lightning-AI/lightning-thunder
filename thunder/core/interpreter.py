from __future__ import annotations

import builtins
import contextlib
import contextvars
import dataclasses
import datetime
import dis
import enum
import functools
import linecache
import inspect
import os
import re
import sys
import traceback
import weakref
import torch
from typing import Any, Literal, NamedTuple, TypedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence, Set, Sized
import collections
import operator

from io import StringIO

from types import (
    CellType,
    ClassMethodDescriptorType,
    CodeType,
    CoroutineType,
    FrameType,
    GetSetDescriptorType,
    FunctionType,
    MethodType,
    MethodDescriptorType,
    ModuleType,
    NoneType,
    BuiltinFunctionType,
    BuiltinMethodType,
    MethodDescriptorType,
    MethodWrapperType,
    WrapperDescriptorType,
    TracebackType,
)

from thunder.core.baseutils import Singleton, init_colors, extract_callable_name


#
# interpreter.py implements a Python interpreter in Python.
#
#   The interpreter is still being developed.
#   See thunder/tests/test_interpreter.py for its current capabilities.
#
#   Python opcodes are executed by "handlers" that manipulate the stack and other
#   components of the interpreter state, and can return a new instruction pointer
#   for the interpreter to jump to.
#
#   The interpreter has two fundamental modes. By default it just tries to emulate the
#   Python interpreter. This is useful when adding opcodes, and for verifying
#   that the Python updates are correct. It also is used by the functional jit.
#
#   When using provenance tracking, the interpreter additionally tries to keep track
#   of where values originated. This mode is used by the general jit. This is done by
#   wrapping all values in WrappedValues.
#
#   Both thunder.jit and thunder.functional.jit use extensions (in jit_ext.py) to
#   create Thunder Programs. They use callbacks and additional lookasides to
#   add their functionality.
#
#   The Thunder program constructed has three parts, a "prologue trace", a
#   "computation trace", and (optionally) an "epilogue trace". The prologue trace has
#   the original function's signature and it:
#
#       - Gathers all the "inputs" to the function, including inputs like globals
#           accessed by the function
#       - Flattens the "inputs" to their computational "leaves" -- the objects that
#           the computation operates on and/or modifies
#       - Validates the inputs, ensuring they are valid inputs for the trace
#           (if the inputs are invalid, the prologue returns a special INVALID_INPUTS symbol)
#
#   The prologue returns the flattened computational leaves, which are then
#   passed to the computation trace for execution. The computation trace
#   records Python operations with side effects and functional computations.
#   Because the prologue has flattened the inputs, the computation trace
#   never performs a container access. Python operations with side effects
#   occur in the same order they did in the original program.
#
#   The epilogue trace applies modifications that were recorded in interpretation.
#   (Initially only setattrs to nn.Module values.)
#

__all__ = [
    "interpret",
]

#
# Types, collections and functions related to understanding objects constructed by the interpreter (like literal numbers)
#


# WrappedValues

# In provenance tracking mode (set in the compile ctx), the interpreter wraps the
# values it handles in this wrapper.
# For now, lookasides will get unwrapped inputs by default and their
# result will be wrapped as originating from the lookaside.
# If a lookaside does not want to handle WrappedValues but needs unwrapping
# of the inputs and wrapping of the results, it needs to
# be decorated with @interpreter_needs_wrap.

# The core difference to proxies is that the code running in the interpreter will
# *not* see the wrapper but will only ever see the original values
# while proxies are intended to be run inside the interpreter.
# Thus we expect wrappers and proxies to complement each other and not
# one to replace the other.

# The .value member should always be a plain Python object without wrappers
# inside, those need to be tracked in item_wrappers and attribute_wrappers

# The introduction strategy for WrappedValues is a bit incremental in
# that we have the following regions where everything is wrapped
# - all things in localsplus are wrapped
# - all things on the stack are wrapped
# - all things in _interpret_call are wrapped
# In contrast for opcode handlers, wrapping is "opt-in" (by popping
# values with .pop_wrapped() and wrapping what they push on the stack.
# For lookasides, it currently is also opt-in, see above.


class WrappedValue:
    def __init__(self, value, /, *, provenance: ProvenanceRecord):
        assert isinstance(provenance, ProvenanceRecord)

        ctx: InterpreterCompileCtx = get_interpretercompilectx()
        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

        self.python_typ = type(value)

        self.nothing = object()
        self.original_value = self.nothing

        self.has_item_tracking = False

        self.value = value

        if isinstance(value, (list, tuple)):
            self.item_wrappers = len(value) * [None]
        elif isinstance(value, Sequence):
            self.item_wrappers = None
        else:
            self.item_wrappers: None | dict | list = {}  # TODO: wrappers for things via getitem/setiem
            self.key_wrappers = {}

        self.attribute_wrappers = {}  # TODO: wrappers for things via getattr/setattr

        self.provenance = provenance
        if self.provenance.inst == PseudoInst.CONSTANT:
            self.provenance.value = value

        runtimectx.cache_wrapper(self)

        cb: None | Callable = ctx.callback(INTERPRETER_CALLBACKS.WRAP_CALLBACK)
        if cb is not None:
            cb(self)

    def track_items(self):
        if self.has_item_tracking:
            return

        populate_item_wrappers(self)

        ctx: InterpreterCompileCtx = get_interpretercompilectx()
        cb: None | Callable = ctx.callback(INTERPRETER_CALLBACKS.PROXIFY_CALLBACK)

        if cb is not None:
            cb(self)

        self.has_item_tracking = True

    def register_proxy(self, proxy):
        # note: the proxy is responsible for capturiing all the existing attributes/values
        assert (
            self.original_value is self.nothing
        ), "cannot proxy multiple times, please file an issue to discuss your use-case"
        self.original_value = self.value
        self.value = proxy

        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
        # we need to keep a strong ref on things that have been proxied to recover the proxy via the cache or we could structure the cache to do this for us
        runtimectx.register_proxied_value(self)

    def unwrap(self):
        return self.value


# Note: Use these with care!
#       In some situations - in particular *args/**kwargs, Python creates tuples and dicts for us,
#       these functions are intended to do the appropriate wrapping for them.
def wrap_args_from_list(l):  # returns a new list!
    res = [_interpret_call(lambda l, i: l[i], l, wrap_const(i)) for i in range(len(unwrap(l)))]
    return res


def wrap_kwargs_from_dict(d):  # returns a new dict
    return {k: _interpret_call(lambda d, k: d[k], d, wrap_const(k)) for k in unwrap(d)}


def wrapped_build_tuple(l: Sequence[WrappedValue]) -> WrappedValue:
    assert all(isinstance(v, WrappedValue) for v in l)
    if l:
        pr = ProvenanceRecord(PseudoInst.BUILD_TUPLE, inputs=[v.provenance for v in l][::-1])  # other inst?
        out = wrap(tuple(v.value for v in l), provenance=pr)
        out.item_wrappers = list(l)
    else:
        # Note: if we revisit returning const here instead of an empty tuple from BUILD_TUPLE, we need to add this to wrap_aergs
        out = wrap_const(())
    return out


def wrap_args(tup):
    assert isinstance(tup, tuple)  # allow other sequences?
    assert all(isinstance(v, WrappedValue) for v in tup)

    return wrapped_build_tuple(tup)


def wrap_kwargs(d):
    assert isinstance(d, dict)
    if not d:
        return wrap_const({})
    # todo: relax k condition?
    assert all(isinstance(k, str) and isinstance(v, WrappedValue) for k, v in d.items())

    inputs = []
    uout = {}
    wrap_key_dict = {}
    wrap_value_dict = {}
    for k, v in d.items():
        wk = wrap_const(k)
        inputs.append(wk.provenance)
        inputs.append(v.provenance)
        uout[wk.value] = v.value
        wrap_value_dict[wk.value] = v
        wrap_key_dict[wk.value] = wk

    pr = ProvenanceRecord(PseudoInst.BUILD_DICT, inputs=inputs)
    out = wrap(uout, provenance=pr)
    out.item_wrappers = wrap_value_dict
    out.key_wrappers = wrap_key_dict
    return out


def wrap(value: Any, /, *, provenance: ProvenanceRecord) -> WrappedValue:
    if isinstance(value, WrappedValue):
        if isinstance(value.value, list):
            assert isinstance(value.item_wrappers, Sized)
            assert len(value.value) == len(
                value.item_wrappers
            ), f"{len(value.value)} {len(value.item_wrappers)} {value.provenance}"
        if isinstance(value.value, dict):
            assert value.item_wrappers is not None
            assert len(value.item_wrappers) == len(value.key_wrappers), f"{value.value}"
        return value

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

    if ctx._with_provenance_tracking:
        cached = runtimectx._known_wrappers.get(id(value))
        if cached is not None:
            potential_wrap = cached[0]()
            if potential_wrap is not None:
                # Note: we want to cache mutuable objects to not run into trouble
                #       with multiple accesses to the same.
                #       As the cache only holds a weakref to the WrappedValue instance
                #       one must persist WrappedValues once things are starting to be modified.
                if isinstance(value, list):
                    assert len(value) == len(potential_wrap.item_wrappers)
                return potential_wrap
            else:
                del runtimectx._known_wrappers[id(value)]
        return WrappedValue(value, provenance=provenance)
    return value


def wrap_const(value: Any, /, *, provenance: ProvenanceRecord | None = None) -> WrappedValue:
    if provenance is None:
        provenance = ProvenanceRecord(inst=PseudoInst.CONSTANT, inputs=[], value=value)
    return wrap(value, provenance=provenance)


def wrap_consts(*values):
    return tuple(wrap_const(v) for v in values)


def wrapped_isinstance(v, c):
    if isinstance(v, WrappedValue):
        return isinstance(v.value, c)
    return isinstance(v, c)


def unwrap(value: Any, /) -> Any:
    if isinstance(value, WrappedValue):
        return value.unwrap()
    return value


def populate_single_dict_item_wrapper(uvalue, obj, key):
    if not isinstance(key, WrappedValue):
        key = wrap_const(key)
    if key.value not in obj.item_wrappers:
        value = wrap_binary_subscr(uvalue, obj, key)
        obj.item_wrappers[key.value] = value
    obj.key_wrappers.setdefault(key.value, key)


def populate_attribute_wrapper(wrapped_object, name, wrapped_attribute):
    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if not ctx._with_provenance_tracking:
        return

    assert isinstance(wrapped_object, WrappedValue)
    assert isinstance(wrapped_attribute, WrappedValue)

    assert (
        getattr(wrapped_object.value, name) is wrapped_attribute.value
    ), f"{getattr(wrapped_object.value, name)}, {wrapped_attribute.value}"

    wrapped_object.attribute_wrappers[name] = wrapped_attribute


def wrap_binary_subscr(uvalue, obj, key):
    if not isinstance(key, WrappedValue):
        key = wrap_const(key)
    if obj.provenance.inst is PseudoInst.CONSTANT and key.provenance.inst is PseudoInst.CONSTANT:
        return wrap_const(uvalue)
    return wrap(uvalue, provenance=ProvenanceRecord(PseudoInst.BINARY_SUBSCR, inputs=[obj.provenance, key.provenance]))


def populate_item_wrappers(l):
    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if not ctx._with_provenance_tracking:
        return

    assert isinstance(l, WrappedValue)
    # to do: generalize
    if wrapped_isinstance(l, (list, tuple)):
        if l.item_wrappers is None:
            l.item_wrappers = [None for _ in range(len(l.value))]
        assert isinstance(l.item_wrappers, list)
        assert len(l.value) == len(l.item_wrappers), f"{len(l.value)=} {len(l.item_wrappers)=}"

        for i, v in enumerate(l.value):
            if l.item_wrappers[i] is None:
                wv = wrap_binary_subscr(v, l, i)
                l.item_wrappers[i] = wv
        return

    if wrapped_isinstance(l, dict):
        assert isinstance(l.item_wrappers, dict)
        for k, v in l.value.items():
            if k not in l.item_wrappers:
                wk = wrap_const(k)
                wv = wrap_binary_subscr(v, l, wk)
                l.item_wrappers[k] = wv
                l.key_wrappers[k] = wk  # or have those from an iteration of the input?
        return

    raise NotImplementedError(f"populate item wrapppers for {type(l.value)}")


#
# interpreter context
#

# Constructs the builtins dictionary
builtins_dict: dict[str, Any] = {k: getattr(builtins, k) for k in dir(builtins)}

# https://docs.python.org/3/library/inspect.html#inspect.getattr_static
getset_descriptor = type(type(open(__file__)).name)
member_descriptor = type(inspect.getattr_static(ModuleType, "__dict__"))


# The interpreter's compile context, which handles compilation directives
# See the comment for interpret() for how these functions work
class InterpreterCompileCtx:
    def __init__(
        self,
        *,
        opcode_interpreter: Callable,
        fn_lookaside: Callable,
        callbacks: dict[INTERPRETER_CALLBACKS, Callable],
        with_provenance_tracking: bool = False,
        uncacheable_classes: Sequence[type] | None = None,
    ):
        self._opcode_interpreter: Callable = opcode_interpreter
        self._fn_lookaside: Callable = fn_lookaside
        self._callbacks: dict[INTERPRETER_CALLBACKS, Callable] = callbacks
        self._with_provenance_tracking = with_provenance_tracking
        if with_provenance_tracking:
            assert isinstance(uncacheable_classes, (list, tuple))
            uncacheable_classes = tuple(set(uncacheable_classes) | {NoneType, int, str, float, bool})

        self._uncacheable_classes = uncacheable_classes

    @property
    def with_provenance_tracking(self):
        return self._with_provenance_tracking

    def interpret(self, inst: dis.Instruction, /, **interpreter_state) -> None | int | INTERPRETER_SIGNALS:
        return self._opcode_interpreter(inst, **interpreter_state)

    def lookaside(self, fn: Callable, /, *args, **kwargs) -> None | Callable:
        return self._fn_lookaside(fn, *args, **kwargs)

    def callback(self, id: INTERPRETER_CALLBACKS) -> None | Callable:
        cb: None | Callable = self._callbacks.get(id, None)
        return cb


_interpretercompilectx = contextvars.ContextVar("interpretercompilectx")


# Sets the interpreter ctx
def set_interpretercompilectx(ctx) -> Any:
    return _interpretercompilectx.set(ctx)


# Returns the current interpreter ctx
def get_interpretercompilectx() -> InterpreterCompileCtx:
    return _interpretercompilectx.get()


def get_interpretercompilectx_if_available() -> InterpreterCompileCtx | None:
    return _interpretercompilectx.get(None)


# Resets the interpreter ctx
def reset_interpretercompilectx(token) -> None:
    _interpretercompilectx.reset(token)


# Context manager for setting the interpreter ctx
@contextlib.contextmanager
def interpretercompilectx(_interpretercompilectx: InterpreterCompileCtx):
    tok: Any = set_interpretercompilectx(_interpretercompilectx)
    try:
        yield
    finally:
        reset_interpretercompilectx(tok)


class LineHistoryItem(TypedDict):
    kind: Literal["Line"]
    fn: Callable | CodeType
    filename: str
    position: Positions | None


class OpaqueHistoryItem(TypedDict):
    kind: Literal["Opaque"]
    fn: Callable


class LookasideHistoryItem(TypedDict):
    kind: Literal["Lookaside"]
    fn: Callable


class CallHistoryItem(TypedDict):
    kind: Literal["InterpreterCall"]
    fn: Callable
    prev_frame: str


class ReturnHistoryItem(TypedDict):
    kind: Literal["InterpreterReturn"]
    fn: Callable
    is_signal: bool
    rval: type | INTERPRETER_SIGNALS


InterpreterHistoryItem = (
    dis.Instruction
    | str
    | LineHistoryItem
    | OpaqueHistoryItem
    | LookasideHistoryItem
    | CallHistoryItem
    | ReturnHistoryItem
)


# How the Interpreter deals with exceptions:
# Conceptually, there are two sources of exceptions that we want to distinguish:
# - Things arising from user code ("User Exceptions").
# - Things that stem from the interpreter itself ("Interpreter Errors").

# To the interpreter User Exceptions are part of its normal operations. So we usually
# don't use Python's exception mechanism to handle these. Instead the
# interpreter mimics CPython's implementation of exception handling by
# defining analoguous structures (curexec and exception_stack) set by do_raise and returns
# INTERPRETER_SIGNALS.EXCEPTION_RAISED when running things that raised exceptions.
# Here in particular:
# - Handlers are "inside the interpreter",
# - _interpret_call_with_unwrapping(...) is "inside the interpreter",
# - lookasides are "inside the interpreter",
# - opaque functions are necessarily outside the interpreter,
# - function objects, iterators, generators,... created in the interpreter are NOT per se in inside the interpreter,
#   so they should raise as usual. When called with _interpret_call_with_unwrapping(...) that will be handled appropriately
#   in the RAISE_VALUE handler.
# Note that for user exceptions, we need to manually amend the traceback as we unwrap. We do so in
# _run_frame.
# Interpreter Errors are raised wrapped in a InterpreterError exception to signal that we have run into
# something with the Interpreter itself or how it was called.


# The interpreter's runtime context, which tracks stack changes in Python mode
class InterpreterRuntimeCtx:
    def __init__(self, *, debug_log: None | StringIO = None):
        self.frame_stack: list[InterpreterFrame] = []
        self._globals_dict: dict[str, Any] | None = None
        self._history: list[InterpreterHistoryItem] = []
        self._interpreted_instructions: list[dis.Instruction] = []
        self._curexc: BaseException | None = None
        # The exception_stack mirrors the exc_info/exc_state from PyThreadState
        # exception_stack is the stack of exceptions currently being handled, we only have exceptions here
        self.exception_stack = [None]
        # ts.exc_info is the top of the stack (self.exception_stack[-1]).
        # Note that most of the time, we are changing the self.exception_stack[-1] instead of popping/pushing exceptions.
        # `exception_stack[-1] = ...` is the equivalent of assiging ts.exc_info->exc_type/value/traceback.
        # ts.exc_state is exc_info (the bottom-most element of the stack(?))
        # ts.curexc (in Python 3.10 as _type / _value / _traceback) is the exception currently being raised

        self.debug_log = debug_log
        self._original_callsite: inspect.FrameInfo = inspect.stack()[2]  # [__init__, fn_, callsite, ...]
        self._prev_filename: None | str = None
        self._prev_position: Positions | None = None
        self._known_wrappers = {}
        self._proxied_values = set()

    def register_proxied_value(self, v):
        self._proxied_values.add(v)

    def cache_wrapper(self, wrapped_value: WrappedValue):
        compilectx: InterpreterCompileCtx = get_interpretercompilectx()
        # we don't know what wrapped_value does for eq, so "is not ()" is correct here but python does not like "is ()"
        _empty_tup = ()
        if not isinstance(wrapped_value.value, compilectx._uncacheable_classes) and (
            wrapped_value.value is not _empty_tup
        ):
            self._known_wrappers[id(wrapped_value.value)] = (weakref.ref(wrapped_value),)

    @property
    def curexc(self) -> BaseException | None:
        return self._curexc

    @curexc.setter
    def curexc(self, value):
        if value is not None:
            assert isinstance(value, BaseException), value
        self._curexc = value

    @property
    def globals_dict(self) -> dict[str, Any]:
        assert self._globals_dict is not None, self._globals_dict
        return self._globals_dict

    # The operations encountered while interpreting
    @property
    def interpreted_instructions(self) -> list[dis.Instruction]:
        return self._interpreted_instructions

    # The operations and opaque calls encountered while interpreting
    @property
    def history(self) -> list[InterpreterHistoryItem]:
        return self._history

    def record(self, val: InterpreterHistoryItem, /) -> None:
        self._history.append(val)

        if self.debug_log is not None:
            self.debug_log.write(f"Appended to history: {val}" + os.linesep)

    def peek_interpreter_stack(self) -> InterpreterStack:
        return self.frame_stack[-1].interpreter_stack

    # get current top of stack
    def peek_frame_stack(self) -> InterpreterFrame | None:
        return self.frame_stack[-1] if len(self.frame_stack) != 0 else None

    def _push_frame_stack(self, frame: InterpreterFrame):
        self.frame_stack.append(frame)

    # for returning from method calls
    def _pop_frame_stack(self):
        assert self.frame_stack, self
        return self.frame_stack.pop()

    @contextlib.contextmanager
    def push_frame_stack(self, frame: InterpreterFrame):
        self._push_frame_stack(frame)
        try:
            yield
        finally:
            pf = self._pop_frame_stack()
            assert pf is frame, "Frame stack inconsistency"

    # TODO Instead of appending to both history and and interpreted_instructions we could
    #   consider just appending to history and then filtering to only instructions when
    #   interpreted_instructions is accessed
    def record_interpreted_instruction(self, inst: dis.Instruction, /) -> InterpreterRuntimeCtx:
        self._interpreted_instructions.append(inst)
        self.record(inst)
        return self

    def record_interpreter_call(self, fn: Callable) -> InterpreterRuntimeCtx:
        frame: InterpreterFrame | None = self.peek_frame_stack()

        # If frame is None, that means that this is the first call to _interpret_call, in _run_frame.
        # In that case we should also print out what line we're starting on, since
        # no line number changes have happened yet.
        if frame is not None:
            self.record({"kind": "InterpreterCall", "fn": fn, "prev_frame": frame.qualname})
        else:
            if hasattr(self._original_callsite, "positions"):
                pos = self._original_callsite.positions
            else:
                pos = Positions(self._original_callsite.lineno, self._original_callsite.lineno, 0, 999)
            # self.record_position(fn, self._original_callsite.filename, pos)
            self.record(
                {
                    "kind": "InterpreterCall",
                    "fn": fn,
                    "prev_frame": self._original_callsite.function,
                }
            )
        return self

    def record_interpreter_return(self, fn: Callable, rval: Any | INTERPRETER_SIGNALS, /) -> InterpreterRuntimeCtx:
        is_signal: bool = isinstance(rval, INTERPRETER_SIGNALS)
        rv: type | INTERPRETER_SIGNALS = rval if is_signal else type(rval)
        self.record(ReturnHistoryItem(kind="InterpreterReturn", fn=fn, is_signal=is_signal, rval=rv))
        return self

    def record_opaque_call(self, fn: Callable) -> InterpreterRuntimeCtx:
        self.record(OpaqueHistoryItem(kind="Opaque", fn=fn))
        return self

    def record_lookaside(self, fn: Callable) -> InterpreterRuntimeCtx:
        self.record(LookasideHistoryItem(kind="Lookaside", fn=fn))
        return self

    def record_position(
        self, fn: Callable | CodeType, filename: str, position: Positions | None, /
    ) -> InterpreterRuntimeCtx:
        # Only record a change in the Python line
        if filename == self._prev_filename and _positions_equal(position, self._prev_position):
            return self

        if position is not None and position.lineno is None:
            position = None

        self._prev_position = position
        self._prev_filename = filename
        line = LineHistoryItem(kind="Line", fn=fn, filename=filename, position=position)
        self.record(line)
        return self

    def format_traceback(self):
        return os.linesep.join(f.format_with_source() for f in self.frame_stack)


def print_to_history(*objects, sep=" ", end=os.linesep):
    if sep is None:
        sep = " "
    if end is None:
        end = os.linesep

    ctx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    ctx._history.append(str(sep).join(str(o) for o in objects) + str(end))


_interpreterruntimectx = contextvars.ContextVar("interpreterruntimectx")


# Sets the interpreter ctx
def set_interpreterruntimectx(ctx) -> Any:
    return _interpreterruntimectx.set(ctx)


# Returns the current interpreter ctx
def get_interpreterruntimectx() -> InterpreterRuntimeCtx:
    return _interpreterruntimectx.get()


# Resets the interpreter ctx
def reset_interpreterruntimectx(token) -> None:
    _interpreterruntimectx.reset(token)


# Context manager for setting the interpreter ctx
@contextlib.contextmanager
def interpreterruntimectx(_interpreterruntimectx: InterpreterRuntimeCtx):
    tok: Any = set_interpreterruntimectx(_interpreterruntimectx)
    try:
        yield
    finally:
        reset_interpreterruntimectx(tok)


# A convenience helper for setting both the interpreter compile and runtime ctx
@contextlib.contextmanager
def interpreter_ctx(_interpretercompilectx: InterpreterCompileCtx, _interpreterruntimectx: InterpreterRuntimeCtx):
    compile_tok: Any = set_interpretercompilectx(_interpretercompilectx)
    runtime_tok: Any = set_interpreterruntimectx(_interpreterruntimectx)
    try:
        yield
    finally:
        reset_interpretercompilectx(compile_tok)
        reset_interpreterruntimectx(runtime_tok)


#
# Helpers
#


def is_opaque(fn: Callable) -> bool:
    if isinstance(
        fn, (BuiltinFunctionType, BuiltinMethodType, MethodDescriptorType, MethodWrapperType, WrapperDescriptorType)
    ):
        return True

    # NOTE builtins.type has type type, but type() is an opaque function
    if fn is type:
        return True

    return False


# Acquires the code object from a function or method (converting methods to functions)
# TODO Print a nicer error message
def extract_code(fn) -> CodeType:
    assert isinstance(fn, Callable), fn
    if isinstance(fn, MethodType):
        return extract_code(fn.__func__)

    if not hasattr(fn, "__code__"):
        raise ValueError(f"Cannot interpret object {repr(fn)} of type {type(fn)}")

    code: CodeType = fn.__code__

    return code


# TODO There may be a better way to determine if an object is a PyCapsule, like
#   importing a known PyCapsule and acquiring its type

CapsuleType = type(datetime.datetime_CAPI)  # type: ignore (Undocumented)


def is_pycapsule(x: Any) -> bool:
    typ: type = type(x)
    return isinstance(typ, CapsuleType)


def is_generalized_method(fn):
    return inspect.ismethod(fn) or isinstance(fn, (BuiltinMethodType, MethodWrapperType))


# Our own exception class for reporting compilation problems
class InterpreterError(RuntimeError):
    pass


# Python doesn't expose the builtin iterator classes, of the iterable or the callable variety.
# This is a helper class, and should not be accessible from user code.
class _CallableIterator:
    def __init__(self, fn: Callable, s: Any):
        self._fn = fn
        self._sentinel = s

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        # TODO: This call to _fn() needs to be interpreted.
        # We should decide how to accomplish this, because
        # this object could outlive the interpreter/runtime contexts.
        # Perhaps all calls to builtin iterators by _interpret_call should be looked aside?
        if (n := self._fn()) == self._sentinel:
            raise StopIteration
        return n


class Py_NULL(metaclass=Singleton):
    pass


# Python <= 3.10 keeps a stack of TryBlocks in the frame. When an exception happens, it unwinds the try block
#                looking for handlers.
# Python >= 3.11 does not use this and has an map from instruction (offsets) to exception handlers instead
#                (code.co_excepttiontable), this is handled in _run_frame
class PyTryBlock:
    # These constants are from Python
    SETUP_FINALLY_TYPE = dis.opmap.get("SETUP_FINALLY", 122)  # 122 is the opcode from 3.10
    EXCEPT_HANDLER_TYPE = 257  # "implicit opcode from Include/opcode.h

    def __init__(self, typ: int, handler: int, level: int):
        self.typ = typ
        self.handler = handler
        self.level = level

    def __str__(self):
        return f"<{type(self).__name__} typ={self.typ} handler={self.handler} level={self.level}>"

    def __repr__(self):
        return self.__str__()


# Use dis.Positions in 3.11+ and make it up in <3.11
if sys.version_info < (3, 11):

    class Positions(NamedTuple):
        lineno: int = None
        end_lineno: int = None
        col_offset: int = None
        end_col_offset: int = None

else:
    Positions = dis.Positions


def _positions_equal(p1: Positions | None, p2: Positions | None):
    if p1 is None or p2 is None:
        return p1 is p2  # both are None
    return (
        p1.lineno == p2.lineno
        and p1.col_offset == p2.col_offset
        and p1.end_lineno == p2.end_lineno
        and p1.end_col_offset == p2.end_col_offset
    )


DUNDER_PATTERN = re.compile(r"^__[a-z_]+__$")


class PythonFrameWrapper:
    def __init__(self, frame: FrameType):
        self.frame: FrameType = frame
        self.code = frame.f_code
        # co_qualname is Python 3.11+
        self.qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        self.positions: Positions
        if sys.version_info >= (3, 11):
            codepos = traceback._get_code_position(frame.f_code, frame.f_lasti)  # type: ignore (_get_code_position is undocumented)
            self.positions = Positions(*codepos)
        else:
            self.positions = Positions(frame.f_lineno, frame.f_lineno, 0, 999)

    def format_with_source(self):
        assert self.positions is not None, self
        l = []
        l.append(f"  in {self.qualname} in file: {self.code.co_filename}, line {self.positions.lineno}:")
        if self.code.co_filename:
            ls = linecache.getlines(self.code.co_filename)
            lineno = self.positions.lineno
            if lineno is None:
                lineno = self.code.co_firstlineno
            l.append("  " + ls[max(lineno - 1, 0)].rstrip())
        return os.linesep.join(l)

    def get_or_make_python_frame(self) -> FrameType:
        return self.frame


def get_python_tb(tb: list | TracebackType | None) -> list:
    if isinstance(tb, list):
        return tb

    res = []
    while tb != None:
        res.append(PythonFrameWrapper(tb.tb_frame))
        tb = tb.tb_next
    return res


class PseudoInst(str, enum.Enum):
    BINARY_SUBSCR = "BINARY_SUBSCR"
    BUILD_DICT = "BUILD_DICT"
    BUILD_TUPLE = "BUILD_TUPLE"
    BUILD_NAMEDTUPLE = "BUILD_NAMEDTUPLE"
    CONSTANT = "CONSTANT"
    EXCEPTION_HANDLER = "EXCEPTION_HANDLER"
    INPUT_ARGS = "INPUT_ARGS"
    INPUT_KWARGS = "INPUT_KWARGS"
    INPUT_FN = "INPUT_FN"
    LOAD_ATTR = "LOAD_ATTR"
    LOOKASIDE = "LOOKASIDE"
    OPAQUE = "OPAQUE"
    SEND = "SEND"
    GET_LEN = "GET_LEN"
    BINARY_ADD = "BINARY_ADD"
    LIST_APPEND = "LIST_APPEND"
    LIST_EXTEND = "LIST_EXTEND"
    GET_ITER = "GET_ITER"
    CONTAINS_OP = "CONTAINS_OP"
    SUPER = "SUPER"
    BUILTINS = "BUILTINS"
    STORE_SUBSCR = "STORE_SUBSCR"


@dataclasses.dataclass
class ProvenanceRecord:
    inst: dis.Instruction | PseudoInst  # or save opname, argval?
    inputs: list  # should we record this relative to the original stack top?
    output_idx: int = 0
    output_key: int | slice | None = None
    value: Any | None = None
    ext_flag: int = 0  # for use by extensions

    def __post_init__(self):
        assert isinstance(self.inst, (dis.Instruction, PseudoInst)), f"{self.inst} is not Instruction or PseudoInst"
        assert isinstance(self.inputs, list)
        assert all(isinstance(i, ProvenanceRecord) for i in self.inputs)
        self.inputs = self.inputs.copy()
        if self.inputs:
            self.ext_flag = functools.reduce(lambda a, b: a & b, (i.ext_flag for i in self.inputs))
        else:
            self.ext_flag = 0

    def __hash__(self):
        return hash((self.inst, *self.inputs))

    def __str__(self):
        counter = 0
        out = ["ProvenanceRecord("]
        known = {}

        def recurse_str(self):
            if self in known:
                return known[self]
            if self.inst == PseudoInst.CONSTANT:
                if isinstance(self.value, (int, str, bool, NoneType)):
                    return repr(self.value)
                elif isinstance(self.value, type):
                    return self.value.__name__
                else:
                    s = repr(self.value)
                    if len(s) < 80 and "\n" not in s:
                        return f"{self.inst}({s})"
                    return f"{self.inst}(<{type(self.value).__name__} object>)"
            nonlocal counter
            inputs = [recurse_str(i) for i in self.inputs]
            inputs_str = ", ".join(inputs)
            i = counter
            counter += 1
            l = f"  i{counter} = {self.inst}({inputs_str})"
            if self.output_idx != 0 or self.output_key is not None:
                l += "# with output spec"
            out.append(l)
            res = f"i{counter}"
            known[self] = res
            return res

        res = recurse_str(self)
        if len(out) == 1:
            return f"ProvenanceRecord({res})"
        out.append(")")
        return "\n".join(out)


class InterpreterStack:
    def __init__(self):
        self._stack = []
        self.provenance_inst = None
        self.provenance_inputs = None

    @contextlib.contextmanager
    def set_cur_instruction(self, inst: dis.Instruction | PseudoInst):
        assert self.provenance_inst is None and self.provenance_inputs is None
        assert isinstance(inst, (dis.Instruction, PseudoInst))
        self.provenance_inst = inst
        self.provenance_inputs = []
        self.output_idx = 0
        try:
            yield
        finally:
            self.provenance_inst = None
            self.provenance_inputs = None

    def get_provenance_record(self, value, key: int | slice | None = None):
        if not isinstance(key, (int, NoneType)):
            raise NotImplementedError("sorry, todo")
        # key=None means append, other key means setitem
        pr = ProvenanceRecord(
            inst=self.provenance_inst,
            inputs=self.provenance_inputs,
            output_idx=self.output_idx,
            output_key=key,
        )
        value = wrap(value, provenance=pr)
        self.output_idx += 1
        return value

    def unpack_provenance_record(self, wrapped_value, key: int | slice | None = None):
        ctx: InterpreterCompileCtx = get_interpretercompilectx()
        if not ctx._with_provenance_tracking:
            return wrapped_value
        # key=None is pop
        if isinstance(key, slice):
            l = len(self._stack)
            if key.start is not None:
                start = key.start
            else:
                start = -l
            assert start < 0
            if key.step is not None:
                step = key.step
            else:
                step = 1
            assert step > 0
            if key.stop is not None:
                stop = key.stop
                assert stop < 0
            else:
                stop = 0
            idxes = list(range(start, stop, step))
            assert len(idxes) == len(wrapped_value)
            return [self.unpack_provenance_record(wv_i, key=i) for i, wv_i in zip(idxes, wrapped_value)]
        if not isinstance(key, (int, NoneType)):
            raise NotImplementedError("sorry, todo")

        assert key is None or key < 0
        assert self.provenance_inputs is not None
        self.provenance_inputs.append(wrapped_value.provenance)
        return unwrap(wrapped_value)

    def delete_provenance_record(self, wrapped_value, key: int | slice):
        # TODO
        pass

    # NOTE push is an alias for append
    def push(self, val: Any, /) -> None:
        return self.append(val)

    # NOTE Append is a helper for dunder setitem
    def append(self, val: Any, /) -> None:
        if isinstance(val, WrappedValue):
            if isinstance(val.value, tuple):
                val.track_items()
                assert val.item_wrappers is not None
                for u, v in zip(val.value, val.item_wrappers):
                    assert u is v.value or u is v.original_value, f"{u}, {v.value}"
        self._stack.append(self.get_provenance_record(val))

    def extend(self, vals: Iterable[Any], /) -> None:
        for v in vals:
            self.append(v)

    def pop_wrapped(self) -> Any:
        return self._stack.pop()

    def pop(self) -> Any:
        return self.unpack_provenance_record(self.pop_wrapped())

    def __len__(self) -> int:
        return len(self._stack)

    def getitem_wrapped(self, key: int | slice, /) -> Any:
        return self._stack[key]

    def __getitem__(self, key: int | slice, /) -> Any:
        return self.unpack_provenance_record(self.getitem_wrapped(key), key=key)

    def __setitem__(self, key: int, val: Any, /) -> None:
        self._stack[key] = self.get_provenance_record(val, key=key)

    def __delitem__(self, key: int | slice, /) -> None:
        self.delete_provenance_record(self._stack[key], key)
        del self._stack[key]


# This is an interpreter frame, similar to Python's for use in our interpreter
# It contains all information needed to execute the current code
# so for generators (which need to suspend and resume) one only needs to
# have the InterpreterFrame around to continue execution.
@dataclasses.dataclass
class InterpreterFrame:
    code: CodeType
    qualname: str
    globals: dict[str, Any] | WrappedValue
    # Name storage, for LOAD_NAME, STORE_NAME, and DELETE_NAME
    # TODO Is this the best way to model this?
    names: dict[str, Any] | WrappedValue
    positions: Positions | None = None
    inst: dis.Instruction | None = None
    call_shape_kwnames: tuple[str] | None = None  # for KW_NAMES opcode in 3.11+
    interpreter_stack: InterpreterStack = dataclasses.field(default_factory=InterpreterStack)
    try_stack: list[PyTryBlock] = dataclasses.field(default_factory=list)
    inst_ptr: int = 0
    lasti: int = 0  # this may deviate from inst_ptr due to RERAISE

    def __repr__(self):
        return f"<{type(self).__name__} {self.code.co_name} at {self.code.co_filename}:{self.positions.lineno if self.positions else None}>"

    # in Python 3.11+ the slots are not split by local/cell/free any more
    localsplus: list[Any] = dataclasses.field(default_factory=list)

    # This is not used for name lookup, it's the return value of locals(), as provided by the locals lookaside.
    # If you want to modify the current value of locals, modify localsplus instead.
    _locals: dict[str, Any] = dataclasses.field(default_factory=lambda: wrap_const({}))

    # advance to the given instruction
    def nexti(self, inst: dis.Instruction):
        self.inst = inst
        if (3, 9) <= sys.version_info < (3, 11):
            if inst.starts_line is not None:
                self.positions = Positions(inst.starts_line, inst.starts_line, 0, 999)
        elif (3, 11) <= sys.version_info < (3, 12):
            if inst.positions is not None:
                self.positions = inst.positions
        else:
            raise NotImplementedError(f"Python {sys.version_info} not supported")

        ctx: InterpreterRuntimeCtx = get_interpreterruntimectx()
        file_name: None | str = self.code.co_filename
        ctx.record_position(self.code, file_name, self.positions)

    def format_with_source(self) -> str:
        # todo: multiple lines in positions, underline, indent
        assert self.positions is not None, self
        l = []
        l.append(f"  in {self.qualname} in file: {self.code.co_filename}, line {self.positions.lineno}:")
        if self.code.co_filename:
            ls = linecache.getlines(self.code.co_filename)
            if ls:
                lineno = self.positions.lineno
                if lineno is None:
                    lineno = self.code.co_firstlineno
                l.append("  " + ls[max(lineno - 1, 0)].rstrip())
            else:
                l.append("  <unavailable>")
        return os.linesep.join(l)

    def get_localsplus_name(self, idx: int) -> str:
        if sys.version_info < (3, 11):
            if idx < self.code.co_nlocals:
                return self.code.co_varnames[idx]
            idx -= self.code.co_nlocals
            if idx < len(self.code.co_cellvars):
                return self.code.co_cellvars[idx]
            idx -= len(self.code.co_cellvars)
            return self.code.co_freevars[idx]
        else:
            # _varname_from_oparg is not documented
            return self.code._varname_from_oparg(idx)  # type: ignore

    def get_or_make_python_frame(self) -> FrameType:
        def fn():
            pass

        assert self.positions is not None
        lineno = self.positions.lineno
        if lineno is None:
            lineno = self.code.co_firstlineno

        rel_lineno = lineno - self.code.co_firstlineno + 1

        # we prefer this code object over fn.__code__ to get the first lineno and the current lineno right,
        # which the following does by inserting so many empty lines that relative to the start line
        # the exception is raised at the right line
        code = compile((rel_lineno - 1) * "\n" + "raise ValueError()", self.code.co_filename, "exec")

        replacements = dict(
            co_filename=self.code.co_filename, co_firstlineno=self.code.co_firstlineno, co_name=self.code.co_name
        )

        if hasattr(fn.__code__, "co_qualname"):
            replacements["co_qualname"] = self.qualname

        fn.__code__ = code.replace(**replacements)  # type: ignore (The replaced fields are the correct types)

        try:
            fn()
            assert False, "Unreachable."
        except ValueError as e:
            tb = e.__traceback__

        assert tb is not None
        while tb.tb_next is not None:
            tb = tb.tb_next
        return tb.tb_frame


#
# Handler registration
#

_default_opcode_handler_map: dict[str, Callable] = {}


def default_opcode_interpreter(inst: dis.Instruction, /, **interpreter_state) -> None | int | INTERPRETER_SIGNALS:
    handler: None | Callable = _default_opcode_handler_map.get(inst.opname, None)
    if handler is None:
        return INTERPRETER_SIGNALS.UNHANDLED_OPCODE

    with interpreter_state["stack"].set_cur_instruction(inst):
        return handler(inst, **interpreter_state)


class register_opcode_handler:
    def __init__(self, name: str, *, min_ver: tuple[int, int] | None = None, max_ver: tuple[int, int] | None = None):
        self.name: str = name
        self.min_ver = min_ver
        self.max_ver = max_ver

    def __call__(self, fn: Callable) -> Callable:
        # TODO: Create a list of opcodes per version, and assert that they're all registered.
        if (self.min_ver is None or self.min_ver <= sys.version_info) and (
            self.max_ver is None or sys.version_info < (*self.max_ver[:-1], self.max_ver[-1] + 1)
        ):
            assert self.name not in _default_opcode_handler_map, self.name
            assert self.name in dis.opmap, self.name
            _default_opcode_handler_map[self.name] = fn
            return fn
        return _default_opcode_handler_map.get(self.name, fn)


#
# Lookaside logic
#


# Returns a new (lookaside) callable that unwraps inputs and wraps its result.
# If the original callable raises an exception, this exception is
# wrapped into INTERPRETER_SIGNALS
def interpreter_needs_wrap(fn):
    def wrapping_wrapper(*args, **kwargs):
        if isinstance(fn, WrappedValue):
            wrapped_fn = fn
        else:
            wrapped_fn = wrap_const(fn)
        ufn = unwrap(fn)

        ctx: InterpreterCompileCtx = get_interpretercompilectx()

        if ctx._with_provenance_tracking:
            uargs = tuple(unwrap(arg) for arg in args)
            ukwargs = {unwrap(k): unwrap(v) for k, v in kwargs.items()}
        else:
            uargs = args
            ukwargs = kwargs

        try:
            res = ufn(*uargs, **ukwargs)

            # If result is a WrappedValue, we trust its provenance record
            if isinstance(res, WrappedValue):
                return res

            # Pass along detected exceptions
            if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                return res

            if ctx._with_provenance_tracking:
                pr = ProvenanceRecord(
                    inst=PseudoInst.OPAQUE,
                    inputs=[wrapped_fn.provenance, wrap_args(args).provenance, wrap_kwargs(kwargs).provenance],
                )
                res = wrap(res, provenance=pr)

            return res

        except Exception as e:
            # Any exceptions from opaque calls are being wrapped with `do_raise`
            res = do_raise(e)
            return res

    return wrapping_wrapper


# Calling a function as an opaque function makes the interpeter not trace into it
@interpreter_needs_wrap
def call_opaque(fn, /, *args, **kwargs):
    return fn(*args, **kwargs)


# Decorator for call opaque
def make_opaque(fn):
    @functools.wraps(fn)
    def fn_(*args, **kwargs):
        if is_jitting():
            return call_opaque(fn, *args, **kwargs)
        else:
            return fn(*args, **kwargs)

    return fn_


#
# Jit lookasides
#
def is_jitting_with_raise():
    """Allow code to behave differently under `@jit`. (For testing.)"""

    # Guard against opaque functions which interrupt jitting.
    if (ctx := get_interpretercompilectx_if_available()) is not None:
        raise InterpreterError(f"Lookaside was not triggered, but there is an active compile context: {ctx}")

    return False


def is_jitting():
    return False


def _is_jitting_lookaside():
    return wrap_const(True)


#
# Python builtin lookasides (ordered alphabetically)
#


# https://docs.python.org/3/library/functions.html?highlight=any#any
def _any_lookaside(obj: Iterable):
    def impl(obj):
        for element in obj:
            if element:
                return True
        return False

    return _interpret_call(impl, obj)


# Implements a less opaque function than bool() that can interpret into dunder bool and dunder len calls
def _bool_lookaside(x: Any) -> bool | INTERPRETER_SIGNALS:
    def impl(x):
        # Handles objects that define __bool__
        null = object()
        dunder_bool = getattr(type(x), "__bool__", null)
        if dunder_bool is not null:
            assert callable(dunder_bool)
            return dunder_bool(x)

        # Handles objects that do not define __bool__ but define __len__
        if hasattr(x, "__len__"):
            return len(x) != 0

        # NOTE By default, objects evaluate to True
        return True

    ux = unwrap(x)  # make the shortcut also work for WrappedValue True or False
    if ux is True or ux is False:
        return x
    return _interpret_call(impl, x)


# https://docs.python.org/3/library/functions.html#enumerate
def _enumerate_lookaside(obj: Iterable, start: int = 0):
    if not wrapped_isinstance(start, int):
        return do_raise(TypeError(f"{type(start)} object cannot be interpreted as an integer"))

    def impl(obj, start):
        n = start
        for elem in obj:
            yield n, elem
            n += 1

    return _interpret_call(impl, obj, wrap_const(start))


@interpreter_needs_wrap
def eval_lookaside(
    source: str | bytes | bytearray | CodeType,  # A python expression
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    /,
) -> Any:
    """Emulate the builtin `eval` function, but evaluate the code in the interpreter."""
    return eval_exec_helper(source, globals, locals, closure=None, mode="eval")


# https://docs.python.org/3/library/functions.html#exec
def exec_lookaside(
    source: str | bytes | bytearray | CodeType,  # A python statement
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    /,
    *,
    closure: tuple[CellType, ...] | None = None,
):
    """Emulate the builtin `exec` function, but evaluate the code in the interpreter."""
    return eval_exec_helper(source, globals, locals, closure=closure, mode="exec")


def eval_exec_helper(
    source: str | bytes | bytearray | CodeType,  # A python statement
    globals: dict[str, Any] | None = None,
    locals: Mapping[str, object] | None = None,
    /,
    *,
    closure: tuple[CellType, ...] | None = None,
    mode: str,
):
    if unwrap(globals) is None and unwrap(locals) is None:
        globals = _globals_lookaside()
        locals = _locals_lookaside()
    elif unwrap(globals) is None:
        globals = _globals_lookaside()
    elif unwrap(locals) is None:
        locals = globals  # !

    if not wrapped_isinstance(source, CodeType):
        ## TODO: revisit str here!
        ucode = compile(str(unwrap(source)), "<string>", mode)
    else:
        ucode = unwrap(source)

    if unwrap(closure) is not None:
        raise NotImplementedError("the closure argument is not yet supported in eval")
    else:
        closure = wrap_const(None)

    uglobals = unwrap(globals)

    compilectx: InterpreterCompileCtx = get_interpretercompilectx()
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

    if "__builtins__" not in uglobals:

        def set_builtins(globals, builtins_dict):
            globals["__builtins__"] = builtins_dict

        # note that it always is *the* (same) builtins_dict
        if compilectx._with_provenance_tracking:
            bd = wrap(builtins_dict, provenance=ProvenanceRecord(inst=PseudoInst.BUILTINS, inputs=[]))
        else:
            bd = builtins_dict

        res = _interpret_call(set_builtins, globals, bd)
        assert res is not INTERPRETER_SIGNALS.EXCEPTION_RAISED

    # execed code has no LOCALSPLUS but only NAMES...
    frame = InterpreterFrame(code=ucode, localsplus=[], globals=globals, names=locals, qualname="<string>")

    if ucode.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR):
        # we should split the preparation from _setup_frame_and_run_python_function
        raise NotImplementedError("exec / eval with generator / coroutine / async generator flags")

    try:
        res, status = _run_frame(frame, compilectx, runtimectx)
    except Exception as e:
        # We need to cheat a bit to get a Python frame here...
        python_frame = frame.get_or_make_python_frame()
        tb = TracebackType(e.__traceback__, python_frame, python_frame.f_lasti, python_frame.f_lineno)
        raise e.with_traceback(tb)

    if mode == "eval":
        return res

    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res
    return wrap_const(None)  # exec does not return anything


# The `__get__`, `__set__`, and `__delete__` methods on a property are implemented in C
# so we have to help the interpreter find its way to the underlying user defined methods.
_PROPERTY_ALIASES = {"__get__": "fget", "__set__": "fset", "__delete__": "fdel"}


# TODO: audit the error messages, in Python some(or all?) things print the class name
# NOT to be put directly in lookasides(!)
def _object_getattribute_lookaside(obj: Any, name: str):
    """Implements the `object.__getattribute__` portion of `getattr`.

    https://docs.python.org/3/howto/descriptor.html#invocation-from-an-instance
    """
    # Some of the handling below does not deal with WrappedValues,
    # so we have an unwrapped copy.
    uobj = unwrap(obj)
    objtype = type(uobj)
    null = cls_var = descr_get = object()

    name = unwrap(name)
    if not isinstance(name, str):
        return do_raise(TypeError("getattr(): attribute name must be string"))

    # TODO: classes and super have a slightly different resolution behavior
    #   https://docs.python.org/3/howto/descriptor.html#invocation-from-a-class
    #   https://docs.python.org/3/howto/descriptor.html#invocation-from-super
    if isinstance(uobj, (type, super)):
        result = getattr(uobj, name, null)
        if result is null:
            return do_raise(AttributeError(name))
        else:
            return result

    # This is too coarse grained, but there is a lot of nuance in the dunder methods
    # for fundamental types, so for now we just bail out. Specifically:
    #   1)  Some builtin C types have `__get__` methods but act like simple namespaces.
    #   2)  If `obj` has a metaclass, the dunder methods might be dynamic.
    # So for now we just fall back to the builtin `getattr` for these bedrock lookups.
    if DUNDER_PATTERN.match(name) or isinstance(uobj, (type, super)):
        return (
            do_raise(AttributeError(f"'{type(uobj).__name__}' object has no attribute '{name}'"))
            if (result := getattr(uobj, name, null)) is null
            else result
        )

    def lookup_descriptor_field(field_name):
        # Bypass the C portions of `property` so we don't break the `_interpret_call` chain
        if type(cls_var) is property and (shortcut := _PROPERTY_ALIASES.get(field_name)):
            # `property` will define `__set__` / `__delete__` even if `fget` / `fdelete` are null
            # However in those cases we should not go through the data descriptor path.
            if (method := getattr(cls_var, shortcut)) is None:
                return null

            # We need to emulate a pure Python version of `property.__get__ / __set__ / __delete__`
            return lambda _, obj, __: method(obj)
        result = _interpret_call_with_unwrapping(getattr, type(cls_var), field_name, null)

        # TODO: For now we can't handle custom `__getattr__`s which raise when we check for
        #       __get__, __set__, or __delete__.
        assert result is not INTERPRETER_SIGNALS.EXCEPTION_RAISED
        return result

    # Check for class variables.
    for base in objtype.__mro__:
        if (cls_var := vars(base).get(name, null)) is not null:
            descr_get = lookup_descriptor_field("__get__")
            break

    if descr_get is not null:
        assert cls_var is not null
        if lookup_descriptor_field("__set__") is not null or lookup_descriptor_field("__delete__") is not null:
            assert callable(descr_get)
            compilectx = get_interpretercompilectx()

            # if it is opaque, don't _interpret_call here, to avoid a wrap/unwrap dance
            if is_opaque(descr_get):
                return descr_get(cls_var, uobj, objtype)
            result = _interpret_call_with_unwrapping(descr_get, cls_var, obj, objtype)
            return result

    # NOTE: `__dict__` is somewhat special, since we can't look inside `__dict__` when we call `obj.__dict__`.
    #       Instead there is a `tp_dict` field in the C struct which controls `__dict__`. (Note that calling
    #       `obj.__dict__` may not return the dict in `tp_dict`, but rather a view on it.) It is, however,
    #       possible to assign to the `__dict__` field of an object. (Including `dict` subclasses.)
    obj_dict = _interpret_call(getattr, obj, wrap_const("__dict__"), wrap_const(null))
    uobj_dict = unwrap(obj_dict)
    if uobj_dict is not null:
        # TODO: error return from _interpret_call?
        assert isinstance(uobj_dict, dict), uobj_dict  # This should be enforced by `PyObject`

        # Even if `obj_dict` is a subclass (which only happens in the corner case that `__dict__` has
        # been manually assigned) Python appears to reinterpret it as a simple dict for the purpose of
        # attribute resolution.
        # we avoid interpreting into dict.get if obj_dict is a plain dict to avoid creating a wrapper for it.
        if type(uobj_dict) == dict:
            instance_value = uobj_dict.get(name, null)
        else:
            instance_value = _interpret_call_with_unwrapping(dict.get, obj_dict, name, null)
        if instance_value is not null:
            return instance_value

    if descr_get is not null:
        assert callable(descr_get)
        return _interpret_call_with_unwrapping(descr_get, cls_var, obj, objtype)

    if cls_var is not null:
        return cls_var

    return do_raise(AttributeError(name))


def check_self(obj, potential_method):
    if is_generalized_method(potential_method.value) or wrapped_isinstance(potential_method, super):
        uslf = getattr(potential_method.value, "__self__", None)
        if uslf is obj.value:
            populate_attribute_wrapper(potential_method, "__self__", obj)
        if type(obj.value) is super:
            superself = obj.attribute_wrappers.get("__self__")
            # super might not have self (when used with classes) or things
            # can happen with types:
            # tng = torch.no_grad()
            # s = super(torch.utils._contextlib._NoParamDecoratorContextManager, tng)
            # s.__self__ is tng while s.__new__.__self__ is object
            # This means that there the __self__ will appear to come out of
            # thin air, but we suspect that for types it is not that
            # terrible.
            if superself is not None and potential_method.value.__self__ is superself.value:
                populate_attribute_wrapper(potential_method, "__self__", superself)


def plausibly_wrapper_of(wrapper, value):
    if wrapper.value is value or wrapper.original_value is value:
        return True
    if callable(value) or True:
        if wrapper.value == value or wrapper.original_value == value:
            return True
    return False


def wrap_attribute(plain_result, obj, name):
    compilectx: InterpreterCompileCtx = get_interpretercompilectx()
    if not compilectx._with_provenance_tracking:
        return plain_result

    known_wrapper = obj.attribute_wrappers.get(name.value)
    # note: there are cases where "is" will always fail (e.g. BuiltinMethods
    #       are recreated every time)
    if known_wrapper is not None:
        assert plausibly_wrapper_of(
            known_wrapper, plain_result
        ), f"attribute {name.value} of {type(obj.value).__name__} object out of sync: {known_wrapper.value} vs. {plain_result}"
        return known_wrapper

    pr = ProvenanceRecord(PseudoInst.LOAD_ATTR, inputs=[obj.provenance, name.provenance])
    result = wrap(plain_result, provenance=pr)

    obj.attribute_wrappers[name.value] = result

    check_self(obj, result)

    return result


def _setattr_lookaside(obj: Any, name: str, value: Any):
    uobj = unwrap(obj)
    uname = unwrap(name)
    uvalue = unwrap(value)
    compilectx: InterpreterCompileCtx = get_interpretercompilectx()

    res = _interpret_call(lambda o, n, v: o.__setattr__(n, v), obj, name, value)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res

    if compilectx._with_provenance_tracking:
        obj.attribute_wrappers[uname] = value

        obj_dict = obj.attribute_wrappers.get("__dict__")
        if obj_dict is not None:
            obj_dict.item_wrappers[uname] = value
            obj_dict.key_wrappers[uname] = name

    return wrap_const(None)


def _getattr_lookaside(obj: Any, name: str, *maybe_default: Any):
    """Emulate slot_tp_getattr_hook()."""
    result = _object_getattribute_lookaside(obj, name)

    ctx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    compilectx: InterpreterCompileCtx = get_interpretercompilectx()

    assert not isinstance(result, WrappedValue)
    if result is not INTERPRETER_SIGNALS.EXCEPTION_RAISED or not isinstance(ctx.curexc, AttributeError):
        if result is not INTERPRETER_SIGNALS.EXCEPTION_RAISED and compilectx._with_provenance_tracking:
            result = wrap_attribute(result, obj, name)
        return result

    # `__getattr__` is only triggered if `__getattribute__` fails.
    # TODO: this should be `_interpret_call_with_unwrapping(getattr, obj, "__getattr__", null := object())`, but that would require multiple current exceptions.
    null = object()
    obj_getattr = getattr(unwrap(obj), "__getattr__", null)

    if obj_getattr is not null:
        ctx.curexc = None
        assert callable(obj_getattr)
        if compilectx._with_provenance_tracking:
            obj_getattr = wrap_attribute(obj_getattr, obj, wrap_const("__getattr__"))
        result = _interpret_call(obj_getattr, name)
        # which provenances to cache here?
        # result = wrap_attribute(unwrap(result), obj, name)

    # And finally if all else fails apply the default. (If provided.)
    if result is INTERPRETER_SIGNALS.EXCEPTION_RAISED and isinstance(ctx.curexc, AttributeError) and maybe_default:
        ctx.curexc = None
        (default,) = maybe_default
        return default

    return result


# TODO: Implement setattr() and delattr() lookasides.
# def _setattr_lookaside(obj: Any, name: str, value: Any) -> None:
#     def impl(o, n, v):
#         o.n = v
#
#     return _interpret_call_with_unwrapping(impl, obj, name, value)


# def _delattr_lookaside(obj: Any, name: str) -> None:
#     def impl(o, n):
#         del o.n
#
#     return _interpret_call_with_unwrapping(impl, obj, name)


def _getitem_lookaside(obj, key, /):
    def impl(obj, key):
        return obj[key]

    return _interpret_call(impl, obj, key)


def _globals_lookaside() -> dict[str, Any]:
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    frame = runtimectx.frame_stack[-1]
    return frame.globals


# https://docs.python.org/3/library/functions.html?highlight=len#len
# Calls https://docs.python.org/3/reference/datamodel.html?highlight=__len__#object.__len__
def _len_lookaside(obj: Any) -> int | INTERPRETER_SIGNALS:
    def impl(obj):
        if not hasattr(obj, "__len__"):
            raise TypeError(f"object of type '{type(obj).__name__}' has no len()")

        lenattr = getattr(obj, "__len__")
        result = lenattr()

        if not isinstance(result, int):
            raise TypeError(f"'{type(result).__name__}' object cannot be interpreted as an integer")

        if result < 0:
            raise ValueError("__len__() should return >= 0")

        return result

    result = _interpret_call(impl, obj)
    if result is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return result

    assert wrapped_isinstance(result, int)
    return result


def _iter_lookaside(obj, *sentinel) -> Iterator | INTERPRETER_SIGNALS:
    # If sentinel is provided
    if len(sentinel) != 0:
        assert len(sentinel) == 1, f"Too many arguments to iter(): {sentinel}"
        (sentinel,) = sentinel

        def iter_sentinel_impl(obj, sentinel):
            if not callable(obj):
                raise TypeError(f"obj must be callable, not {type(obj).__name__}")
            return _CallableIterator(obj, sentinel)

        ret = _interpret_call(iter_sentinel_impl, obj, sentinel)
        if ret is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return ret
        assert isinstance(unwrap(ret), _CallableIterator)
        return ret
        # This used to be a trick to return the same type as the builtin iter
        # return iter(lambda: next(unwrap(ret)), sentinel)

    # If sentinel is not provided
    def nosentinel_impl(obj):
        if hasattr(obj, "__iter__"):
            return obj.__iter__()
        elif hasattr(obj, "__getitem__"):
            # Python linters don't have an issue converting to Sequence, although the language may
            # say that isinstance(obj, Sequence) is false for some objects with a __getitem__.
            seq: Sequence = obj

            # Create an iterator for the iterable.
            # We don't check __len__ here, because cpython doesn't check __len__ here, it just
            # keeps blindly calling getitem with increasing numbers until it hits an IndexError.
            it_idx = 0
            sentinel = object()

            def next_item():
                try:
                    nonlocal it_idx
                    ret = seq[it_idx]
                    it_idx += 1
                    return ret
                except IndexError:
                    return sentinel

            return _CallableIterator(next_item, sentinel)
        else:
            raise TypeError(f"{type(object)} object is not iterable")

    ret = _interpret_call(nosentinel_impl, obj)
    # This used to be a trick to return the same type as the iter.

    # uret = unwrap(ret)
    # if isinstance(uret, _CallableIterator):

    #     # Trick iter() into constructing a new iterator of the correct type.
    #     class _IteratorSequenceWrapper:
    #         def __init__(self, it: _CallableIterator):
    #             self._it = it

    #         def __getitem__(self, i):
    #             r = self._it.__next__()
    #             if r is self._it._sentinel:
    #                 raise IndexError
    #             return r

    #     ret = _interpret_call(lambda cls, ret: iter(cls(ret)), _IteratorSequenceWrapper, ret)

    return ret


def _locals_lookaside() -> dict[str, Any]:
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    frame = runtimectx.frame_stack[-1]
    _locals = frame._locals
    u_locals = unwrap(_locals)
    for i, v in enumerate(frame.localsplus):
        name = frame.get_localsplus_name(i)
        if v is Py_NULL():
            # Only delete identifiers to avoid breaking pytest, which
            # rewrites assertions using locals() for some reason.
            if name.isidentifier() and name in u_locals.keys():

                def delitem(d, k):
                    del d[k]

                res = _interpret_call(delitem, _locals, wrap_const(name))
                assert not isinstance(res, INTERPRETER_SIGNALS)
            continue
        elif isinstance(v, CellType):  # sketchy, we should keep this info
            v = v.cell_contents

        def setitem(d, k, v):
            d[k] = v

        res = _interpret_call(setitem, _locals, wrap_const(name), v)

    return _locals


# https://docs.python.org/3.13/library/functions.html#next
_nil = []


def _next_lookaside(iterator, default=_nil):
    def impl(iterator):
        return iterator.__next__()

    res = _interpret_call(impl, iterator)

    if default is not _nil and res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
        if isinstance(runtimectx._curexc, StopIteration):
            res = default
    return res


def _reversed_lookaside(seq):
    def impl(seq):
        reversed_meth = getattr(seq, "__reversed__", None)
        if reversed_meth is not None:
            return reversed_meth()

        if not isinstance(seq, Sequence):
            raise TypeError(f"'{type(seq)}' object is not reversible")

        return SequenceIter(seq, is_reversed=True)

    return _interpret_call(impl, seq)


# we do like the built-in super (and in fact it is impossible to implement
# it in Python when it comes to builtin methods (torch.autograd.Function.apply
# has a __self__ but not a __func__, so we cannot fill in the superclass self
# in the call to the func), but super() without arguments needs to inspect
# frames, so we do this inspection here and then instantiate the builtin super
# with parameters
def _super_lookaside(cls=Py_NULL(), obj=None):
    # cls Py_NULL vs. obj None this is on purpose

    # magic for super() per frame inspection. Note that we do not currently
    # do frame entries for lookasides, so the thing calling super is at the top
    if cls is Py_NULL() and obj is None:
        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

        frame = runtimectx.frame_stack[-1]
        self_name = frame.code.co_varnames[0]  # is this guaranteed to be self?
        self_idx = None
        class_idx = None
        for i in range(len(frame.localsplus)):
            if frame.get_localsplus_name(i) == self_name:
                self_idx = i
                # we cannot break because it might be put in a cell in Python 3.10
            if frame.get_localsplus_name(i) == "__class__":
                class_idx = i
        if class_idx is None or not wrapped_isinstance(frame.localsplus[class_idx], CellType):
            return do_raise(RuntimeError("super(): __class__ cell not found"))
        assert self_idx is not None
        cls = _interpret_call(lambda c: c.cell_contents, frame.localsplus[class_idx])
        obj = frame.localsplus[self_idx]
        if wrapped_isinstance(obj, CellType):  # this is a bit fishy, Python knows in advance
            obj = _interpret_call(lambda c: c.cell_contents, obj)

    # now cls and obj are set
    ucls = unwrap(cls)
    if not isinstance(ucls, type):
        return do_raise(TypeError(f"super() argument 1 must be a type, not {type(ucls).__name__}"))

    uobj = unwrap(obj)
    if obj is None:
        obj = wrap_const(obj)

    usup = super(ucls, uobj)  # type: ignore

    compilectx: InterpreterCompileCtx = get_interpretercompilectx()
    if not compilectx._with_provenance_tracking:
        return usup

    pr = ProvenanceRecord(PseudoInst.SUPER, inputs=[cls.provenance, obj.provenance])
    sup = wrap(usup, provenance=pr)

    check_self(obj, sup)

    return sup


@interpreter_needs_wrap
def _type_lookaside(obj):
    return type(unwrap(obj))


@interpreter_needs_wrap
def _isinstance_lookaside(obj, cls):
    obj = unwrap(obj)
    cls = unwrap(cls)
    if isinstance(cls, tuple):
        cls = tuple(unwrap(c) for c in cls)
    return isinstance(obj, cls)


def _functools_reduce_lookaside(
    fn: Callable, iterable: Iterable, initializer: Py_NULL | Any = Py_NULL(), /
) -> Any | INTERPRETER_SIGNALS:
    null = wrap_const(object())
    if initializer is Py_NULL():
        initializer = null

    def impl(fn, iterable, initializer, null):
        it = iter(iterable)

        # No, default is not None, it is absence of value.
        if initializer is null:
            try:
                res = next(it)
            except StopIteration:
                raise TypeError("reduce() of empty iterable with no initial value")
        else:
            res = initializer

        for e in it:
            res = fn(res, e)

        return res

    return _interpret_call(impl, fn, iterable, initializer, null)


# An iterator to be returned from Sequence.__iter__ lookasides below. This will be run in the interpreter
# Note: this potentially might imitate a list_iterator / tuple_iterator more...
class SequenceIter:
    def __init__(self, s, is_reversed=False):
        self.s = s
        self.next_pos = 0 if not is_reversed else len(s) - 1
        self.is_reversed = is_reversed

    def __iter__(self):
        return self

    def __length_hint__(self):
        return len(self.s)

    def __next__(self):
        if (not self.is_reversed and (self.next_pos >= len(self.s))) or self.is_reversed and (self.next_pos < 0):
            raise StopIteration()
        res = self.s[self.next_pos]
        self.next_pos += 1 if not self.is_reversed else -1
        return res


# wrapper-handling lookasides for sequences and mutuable sequences.
# note:
# - these are only for use when wrapping is enabled
# - the methods (or the corresponding functions) will be registered
#   as lookasides for tuple and list. This means that they will be
#   called with wrapped values also for self and self.value will point
#   to the actual object...
#
# TODO: maybe make these generic for sequences / mutuable sequence
# https://docs.python.org/3/library/stdtypes.html#common-sequence-operations
class SequenceWrapperMethods(WrappedValue):
    # NOTE! This is not actually a WrappedValue. However,

    def __init__(self, iterable=(), /):
        if iterable == ():
            iterable = wrap_const(())
        l = wrap_const([])
        assert l.item_wrappers is not None

        res = _interpret_call(list.extend, l, iterable)
        if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return res
        self.value = self.python_typ(l.value)
        self.item_wrappers = l.item_wrappers[:]
        return wrap_const(None)

    def __getitem__(self, idx, /):
        self.track_items()
        assert self.item_wrappers is not None

        uidx = idx.value
        uself = self.value
        if isinstance(uidx, int):
            uidx = int(uidx)  # might have been IntProxy, this should insert condition if not const
            if uidx < -len(uself) or uidx >= len(uself):
                return do_raise(IndexError(f"{type(uself)} index out of range"))

            if uidx < 0:
                # TODO: callback to allow asserting length of list/tuple
                #       either only for uidx < 0 or for any uid
                uidx += len(uself)
            assert (
                self.item_wrappers[uidx].value is self.value[uidx]
                or self.item_wrappers[uidx].original_value is self.value[uidx]
            )
            return self.item_wrappers[uidx]

        if isinstance(uidx, slice):
            ures = uself[uidx]  # errors?
            res = wrap_binary_subscr(ures, self, idx)
            res.item_wrappers = self.item_wrappers[uidx]
            return res

        return do_raise(TypeError(f"{type(uself)} indices must be integers or slices, not {type(idx.value)}"))

    def __contains__(self, key, /):
        self.track_items()
        return _interpret_call(lambda s, k: any((i == k) for i in s), self, key)

    def __add__(self, value, /):
        self.track_items()
        populate_item_wrappers(value)

        try:
            ures = self.value + value.value
        except Exception as e:
            return do_raise(e)

        pr = ProvenanceRecord(PseudoInst.BINARY_ADD, inputs=[self.provenance, value.provenance])
        res = wrap(ures, provenance=pr)
        res.item_wrappers = self.item_wrappers + value.item_wrappers
        return res

    def __mul__(self, n, /):
        self.track_items()

        def impl(self, n):
            l = []
            for _ in range(n):
                l.extend(self)
            return l

        return _interpret_call(impl, self, n)

    def __rmul__(self, n, /):
        self.track_items()
        return self.__mul__(n)

    def __len__(self):
        self.track_items()
        # TODO: record length check
        pr = ProvenanceRecord(PseudoInst.GET_LEN, inputs=[self.provenance])
        return wrap(len(self.value), provenance=pr)

    def index(self, value, start=0, stop=2**63 - 1, /):
        self.track_items()

        # this is the actual python signature for list.index
        if start == 0:
            start = wrap_const(start)
        if stop == 2**63 - 1:
            stop = wrap_const(stop)

        # assert len?

        def impl(seq, value, start, stop):
            for idx, item in enumerate(seq):
                if item == value:
                    return idx
            raise ValueError(f"{value} is not in {type(seq)}")

        return _interpret_call(impl, self, value, start, stop)

    def count(self, value, /):
        self.track_items()
        raise NotImplementedError("Sequence.count, please file an issue")

    def __iter__(self):
        self.track_items()
        return _interpret_call(SequenceIter, self)

    def __reversed__(self):
        self.track_items()
        return _interpret_call(SequenceIter, self, wrap_const(True))


class MutSequenceWrapperMethods(SequenceWrapperMethods):
    def __new__(cls, iterable=()):
        assert isinstance(cls.value, type)
        return wrap_const(cls.value())

    def __init__(self, iterable=()):
        # We need to propagate the return value because it could be JIT_SIGNALS
        res = SequenceWrapperMethods.__init__(self, iterable)
        return res

    def __setitem__(self, key, value, /):
        self.track_items()
        uself = self.value
        ukey = key.value

        if isinstance(ukey, slice):
            value = _interpret_call(list, value)
            if value is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                return value
            assert isinstance(value, WrappedValue)
            assert isinstance(value.value, list)
            populate_item_wrappers(value)

        uvalue = value.value

        assert isinstance(uself, list)
        self.value[ukey] = uvalue
        assert self.item_wrappers is not None
        assert value.item_wrappers is not None
        if isinstance(ukey, slice):
            self.item_wrappers[ukey] = value.item_wrappers[:]
        else:
            self.item_wrappers[ukey] = value
            assert self.item_wrappers[ukey].value is self.value[ukey]

        assert len(self.value) == len(self.item_wrappers)
        return wrap_const(None)

    def __delitem__(self, key, /):
        self.track_items()
        assert self.item_wrappers is not None

        ukey = key.value
        try:
            del self.value[ukey]
            del self.item_wrappers[ukey]
        except Exception as e:
            return do_raise(e)
        return wrap_const(None)

    def append(self, value, /):
        self.track_items()
        assert self.item_wrappers is not None

        pr = ProvenanceRecord(PseudoInst.LIST_APPEND, inputs=[self.provenance, value.provenance])
        self.provenance = pr  # should have an update method
        self.value.append(value.value)
        assert type(self.item_wrappers) is list
        self.item_wrappers.append(value)
        assert len(self.value) == len(self.item_wrappers)
        return wrap_const(None)

    def clear(self, /):
        self.track_items()
        raise NotImplementedError("Sequence.clear, please file an issue")

    def copy(self, /):
        self.track_items()
        raise NotImplementedError("Sequence.copy, please file an issue")

    def extend(self, iterable, /):
        self.track_items()
        assert self.item_wrappers is not None

        if not isinstance(iterable.value, (tuple, list)):

            def impl(l, iterable):
                for i in iterable:
                    l.append(i)

            res = _interpret_call(impl, self, iterable)
            assert len(self.value) == len(self.item_wrappers)
            return res

        populate_item_wrappers(iterable)
        pr = ProvenanceRecord(PseudoInst.LIST_EXTEND, inputs=[self.provenance, iterable.provenance])
        self.provenance = pr  # should have an update method
        assert len(iterable.value) == len(iterable.item_wrappers)
        # also includes l.value is iterable.value
        self.value.extend(iterable.value)
        assert type(self.item_wrappers) is list
        self.item_wrappers.extend(iterable.item_wrappers)
        assert len(self.value) == len(self.item_wrappers)
        return wrap_const(None)

    def __iadd__(self, iterable, /):
        self.track_items()
        res = _interpret_call(list.extend, self, iterable)
        return self

    def __imul__(self, n, /):
        self.track_items()
        raise NotImplementedError("Sequence.__imul__, please file an issue")

    def insert(self, i, x, /):
        self.track_items()
        raise NotImplementedError("Sequence.insert, please file an issue")

    def pop(self, index=-1, /):
        self.track_items()

        if index == -1:
            index = wrap_const(-1)

        uself = self.value
        uindex = index.value

        assert isinstance(uself, list)

        if not isinstance(uindex, int):
            return do_raise(TypeError(f"'{type(uindex)}' object cannot be interpreted as an integer"))

        uindex = int(uindex)  # if it was subclass like IntProxy

        # assert len

        if not uself:
            return do_raise(IndexError(f"pop on empty {type(uself)}"))

        if uindex < -len(uself) or uindex >= len(uself):
            return do_raise(IndexError(f"pop index out of range"))

        res = _interpret_call(lambda l, i: l[i], self, index)

        assert res is not INTERPRETER_SIGNALS.EXCEPTION_RAISED

        assert self.item_wrappers is not None
        assert len(self.value) == len(self.item_wrappers)
        del uself[uindex]
        del self.item_wrappers[uindex]
        assert len(self.value) == len(self.item_wrappers)

        return res

    def remove(self, x, /):
        self.track_items()
        raise NotImplementedError("Sequence.remove, please file an issue")

    def reverse(self, /):
        self.track_items()
        self.value.reverse()
        assert type(self.item_wrappers) is list
        self.item_wrappers.reverse()
        return wrap_const(None)


class MappingKeysIterator(Iterator):
    # note: the __init__ will be executed by Python itself, and
    #       the caller needs to set up the wrapped_attribute for _mapping
    # The other methods are called through the interpreter mechanism.
    def __init__(self, mapping, underlying_key_iterator):
        self._mapping = mapping
        self._underlying_key_iterator = underlying_key_iterator

    # This is called as a lookaside!
    def __next__(self):
        try:
            uk = self.value._underlying_key_iterator.__next__()
        except Exception as e:
            return do_raise(e)
        k = self.attribute_wrappers["_mapping"].key_wrappers[uk]
        return k


class MappingKeysView:
    def __init__(self, mapping):
        self._mapping = mapping

    @property
    def mapping(self):
        return self._mapping  # a should be a MappingProxy...

    def isdisjoint(self, other):
        return all((k not in self.mapping) for k in other)

    # This is called as a lookaside!
    def __iter__(self):
        raw_mapping_iter = self.value._mapping.__iter__()
        pr = ProvenanceRecord(PseudoInst.GET_ITER, inputs=[self.attribute_wrappers["_mapping"].provenance])
        u_mapping_iter = MappingKeysIterator(self.value._mapping, raw_mapping_iter)
        mapping_iter = wrap(u_mapping_iter, provenance=pr)
        populate_attribute_wrapper(mapping_iter, "_mapping", self.attribute_wrappers["_mapping"])
        return mapping_iter

    # This is called as a lookaside!
    def __reversed__(self):
        raw_mapping_iter = self.value._mapping.__reversed__()
        pr = ProvenanceRecord(PseudoInst.GET_ITER, inputs=[self.attribute_wrappers["_mapping"].provenance])
        u_mapping_iter = MappingKeysIterator(self.value._mapping, raw_mapping_iter)
        mapping_iter = wrap(u_mapping_iter, provenance=pr)
        populate_attribute_wrapper(mapping_iter, "_mapping", self.attribute_wrappers["_mapping"])
        return mapping_iter


class MappingValuesIterator:
    def __init__(self, mapping, is_reversed=False):
        self._mapping = mapping
        if is_reversed:
            self._key_iter = reversed(mapping)
        else:
            self._key_iter = iter(mapping)

    def __iter__(self):
        return self

    def __next__(self):
        return self._mapping[next(self._key_iter)]


class MappingValuesWrapper:
    def __init__(self, mapping):
        self._mapping = mapping

    def __iter__(self):
        return MappingValuesIterator(self._mapping)


class MappingItemsIterator:
    def __init__(self, mapping, is_reversed=False):
        self._mapping = mapping
        if is_reversed:
            self._key_iter = mapping.__reversed__()
        else:
            self._key_iter = mapping.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        k = next(self._key_iter)
        return k, self._mapping[k]


class MappingItemsWrapper:
    def __init__(self, mapping):
        self._mapping = mapping

    def __iter__(self):
        return MappingItemsIterator(self._mapping)


class MutMappingWrapperMethods(WrappedValue):
    def __new__(cls, /, *args, **kwds):
        uvalue = unwrap(cls)()
        # todo: for subclasses, better record the call to the constructor
        return wrap_const(uvalue)

    def __init__(self, *other, **kwds):
        MutMappingWrapperMethods.update(self, *other, **kwds)
        return wrap_const(None)

    def __setitem__(self, key, value):
        self.track_items()
        assert self.item_wrappers is not None

        self.value[key.value] = value.value
        self.key_wrappers[key.value] = key
        self.item_wrappers[key.value] = value
        return wrap_const(None)

    def __delitem__(self, key):
        self.track_items()
        assert self.item_wrappers is not None

        try:
            del self.value[key.value]
        except Exception as e:
            return do_raise(e)

        del self.key_wrappers[key.value]
        del self.item_wrappers[key.value]
        return wrap_const(None)

    def __getitem__(self, key):
        # Calling self.track_items() here breaks things.
        assert self.item_wrappers is not None

        try:
            uv = self.value[key.value]
        except Exception as e:
            return do_raise(e)

        populate_single_dict_item_wrapper(uv, self, key.value)
        v = self.item_wrappers[key.value]
        assert uv is v.value or uv is v.original_value, f"value for {key.value} out of sync {uv} {v.value}"
        return v

    def __iter__(self):
        def impl(self):
            return self.keys().__iter__()

        return _interpret_call(impl, self)

    def __reversed__(self):
        def impl(self):
            return self.keys().__reversed__()

        return _interpret_call(impl, self)

    def __len__(self):
        self.track_items()
        # TODO: record length check
        pr = ProvenanceRecord(PseudoInst.GET_LEN, inputs=[self.provenance])
        return wrap(len(self.value), provenance=pr)

    def clear(self):
        self.track_items()
        assert self.item_wrappers is not None

        self.value.clear()
        self.key_wrappers.clear()
        self.item_wrappers.clear()

    # note: popitem with last is only for ordered dict
    def popitem(self, last=Py_NULL()):
        self.track_items()
        assert self.item_wrappers is not None

        if last is Py_NULL():
            last_d = {}
        else:
            last_d = {"last": last.value}

        try:
            uk, uv = self.value.popitem(last=last)
        except Exception as e:
            return do_raise(e)

        k = self.key_wrappers.pop(uk)
        v = self.item_wrappers.pop(uk)
        assert k.value is uk
        assert v.value is uv
        return k, v

    def __contains__(self, key):
        self.track_items()
        # TODO: assert presence
        pr = ProvenanceRecord(PseudoInst.CONTAINS_OP, inputs=[self.provenance, key.provenance])
        return wrap(key.value in self.value, provenance=pr)

    def get(self, key, default=None):
        res = _interpret_call(lambda d, k: d[k], self, key)
        if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
            if isinstance(runtimectx._curexc, KeyError):
                runtimectx._curexc = None
                if default is None:
                    res = wrap_const(None)
                else:
                    res = default
        return res

    # move to end is odict specific, does not affect metadata but will raise
    # unimplemented if we comment it out
    # def move_to_end(self, key, last=True):

    # TODO
    # def __sizeof__(self):

    def update(self, *other, **other_kw):
        # This looks like it could nicely go into one big impl, but dicts
        # are quite omnipresent and we need to avoid infinite recursions
        # between iters (which have DICT_MERGE somewhere in the iter
        # lookaside apparently) and dicts.
        self.track_items()

        if other:
            (other,) = other
            # testing bool(other.value) is a bit touchy, but
            # we do this to avoid infinite recursions
            if hasattr(other.value, "keys") and other.value:

                def impl_other_keys(self, other):
                    for k in other.keys():
                        self[k] = other[k]

                res = _interpret_call(impl_other_keys, self, other)
                if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                    return res
            elif other.value:

                def impl_other_nokeys(self, other):
                    for k, v in other:
                        self[k] = v

                res = _interpret_call(impl_other_nokeys, self, other)
                if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                    return res

        for uk, v in other_kw.items():
            k = wrap_const(uk)

            def setitem(self, k, v):
                self[k] = v

            res = _interpret_call(setitem, self, k, v)
            if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                return res

        return wrap_const(None)

    def keys(self):
        self.track_items()
        return _interpret_call(MappingKeysView, self)

    def items(self):
        self.track_items()
        return _interpret_call(MappingItemsWrapper, self)

    def values(self):
        self.track_items()
        return _interpret_call(MappingValuesWrapper, self)

    # __ne__ = _collections_abc.MutableMapping.__ne__

    # __marker = object()

    def pop(self, key, default=Py_NULL()):
        if default is Py_NULL():

            def impl_no_default(self, key):
                v = self[key]
                del self[key]
                return v

            return _interpret_call(impl_no_default, self, key)

        def impl(self, key, default):
            if key in self:
                v = self[key]
                del self[key]
                return v
            return default

        return _interpret_call(impl, self, key, default)

    def setdefault(self, key, default=None):
        def impl(self, key, default):
            if key in self:
                return self[key]
            self[key] = default
            return default

        return _interpret_call(impl, self, key, default)

    # def __repr__(self):
    # def __reduce__(self):

    def copy(self):
        return _interpret_call(self.value.__class__, self)

    # @classmethod
    # def fromkeys(cls, iterable, value=None):

    # def __eq__(self, other):

    def __ior__(self, other):
        def impl(self, other):
            self.update(other)
            return self

        return _interpret_call(impl, self, other)

    def __or__(self, other):
        def impl(self, other):
            if not isinstance(other, dict):
                return NotImplemented
            new = self.copy()
            new.update(other)
            return new

        return _interpret_call(impl, self, other)

    def __ror__(self, other):
        def impl(self, other):
            if not isinstance(other, dict):
                return NotImplemented
            new = self.__class__(other)
            new.update(self)
            return new

        return _interpret_call(impl, self, other)


def _collections_namedtuple_lookaside(
    typename: str,
    field_names: Iterable[str],
    *,
    rename: bool = False,
    defaults: None | Iterable[Any] = None,
    module: None | str = None,
):
    # Type checks {
    assert wrapped_isinstance(typename, str)
    assert wrapped_isinstance(field_names, Iterable)
    assert wrapped_isinstance(rename, bool)
    if defaults is not None:
        assert wrapped_isinstance(defaults, Iterable)
    if module is not None:
        assert wrapped_isinstance(module, str)
    # }

    # Wrap defaults {
    if not isinstance(rename, WrappedValue):
        rename = wrap_const(rename)

    if defaults is None:
        defaults = wrap_const(defaults)

    if module is None:
        # To prevent taking module from the direct caller,
        # we use the module's name from the active frame
        curr_frame = get_interpreterruntimectx().frame_stack[-1]
        module = unwrap(curr_frame.globals).get("__name__", None)
        module = wrap_const(module)
    # }

    # Run opaque namedtuple {
    @interpreter_needs_wrap
    def create_namedtuple(typename: str, field_names: str, **kwargs):
        namedtuple_type = collections.namedtuple(typename, field_names, **kwargs)
        return namedtuple_type

    namedtuple_type = create_namedtuple(typename, field_names, rename=rename, defaults=defaults, module=module)
    if namedtuple_type is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return namedtuple_type

    assert wrapped_isinstance(namedtuple_type, type)
    # }

    return namedtuple_type


_default_lookaside_map: dict[Callable, Callable] = {
    # Jit lookasides
    is_jitting: _is_jitting_lookaside,
    is_jitting_with_raise: _is_jitting_lookaside,
    call_opaque: call_opaque,
    # Python builtin lookasides
    any: _any_lookaside,
    bool: _bool_lookaside,
    enumerate: _enumerate_lookaside,
    exec: exec_lookaside,
    eval: eval_lookaside,
    getattr: _getattr_lookaside,
    setattr: _setattr_lookaside,
    globals: _globals_lookaside,
    iter: _iter_lookaside,
    len: _len_lookaside,
    locals: _locals_lookaside,
    next: _next_lookaside,
    reversed: _reversed_lookaside,
    super: _super_lookaside,
    type.__call__: _type_lookaside,
    isinstance: _isinstance_lookaside,
    functools.reduce: _functools_reduce_lookaside,
    operator.getitem: _getitem_lookaside,
    collections.namedtuple: _collections_namedtuple_lookaside,
}


# While mutuable sequences (lists) are created empty in __new__ and populated in __init__,
# immutuable sequences (tuples) are created with contents in __new__ and __init__ is a nop
# (object.__init__, actually).
def _tuple_new_provenance_tracking_lookaside(cls, iterable=(), /):
    new_tuple_type = cls.value
    assert issubclass(new_tuple_type, tuple)

    if iterable == ():
        iterable = wrap_const(())

    if isinstance(iterable.value, (list, tuple)):
        # special case to avoid infinite recursion
        iterable.track_items()
        item_wrappers = []
        # TODO: investigate why just taking the wrappers will break test_interpreter.py::test_module_hooks
        for i in range(len(iterable.value)):
            item_wrappers.append(_interpret_call(lambda l, i: l[i], iterable, wrap_const(i)))
    else:
        iterator = _interpret_call(iter, iterable)
        if iterator is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return INTERPRETER_SIGNALS.EXCEPTION_RAISED

        item_wrappers = []
        done = False
        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
        while not done:
            wv = _interpret_call(lambda iterator: iterator.__next__(), iterator)
            if wv is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                if isinstance(runtimectx.curexc, StopIteration):
                    done = True
                else:
                    return INTERPRETER_SIGNALS.EXCEPTION_RAISED
            else:
                item_wrappers.append(wv)

    def is_likely_from_collections_namedtuple(tuple_type):
        from collections import namedtuple

        # Check if tuple_type code object is coming from namedtuple
        return (
            hasattr(tuple_type, "__repr__")
            and hasattr(tuple_type.__repr__, "__code__")
            and tuple_type.__repr__.__code__ in namedtuple.__code__.co_consts
        )

    # Construction of namedtuples may raise
    try:
        ures = tuple(w.value for w in item_wrappers)
        # Named tuples expect varargs, not iterables at new/init
        if is_likely_from_collections_namedtuple(new_tuple_type):
            if hasattr(new_tuple_type, "__bases__") and new_tuple_type.__bases__ == (tuple,):
                ures = new_tuple_type(*ures)
                build_inst = PseudoInst.BUILD_NAMEDTUPLE
            else:
                return do_raise(
                    NotImplementedError(
                        f"The type {new_tuple_type} is likely a subclassed named tuple. "
                        "Subclassing the types returned by `collections.namedtuple` "
                        "is currently not supported! Please, file an issue requesting this support."
                    )
                )
        else:
            ures = new_tuple_type(ures)
            build_inst = PseudoInst.BUILD_TUPLE
    except Exception as e:
        return do_raise(e)

    pr = ProvenanceRecord(build_inst, inputs=[w.provenance for w in item_wrappers])
    res = wrap(ures, provenance=pr)
    res.item_wrappers = item_wrappers

    for u, v in zip(res.value, res.item_wrappers):
        assert u is v.value, f"{u}, {v.value}"
    return res


def _cell_new_provenance_tracking_lookaside(typ, *contents):
    assert typ.value is CellType
    if contents:
        (contents,) = contents
        ucell = CellType(contents.value)
        pr = ProvenanceRecord(
            PseudoInst.OPAQUE, inputs=[wrap_const(CellType).provenance, typ.provenance, contents.provenance]
        )
        cell = wrap(ucell, provenance=pr)
        populate_attribute_wrapper(cell, "cell_contents", contents)
    else:
        ucell = CellType()
        pr = ProvenanceRecord(PseudoInst.OPAQUE, inputs=[wrap_const(CellType).provenance, typ.provenance])
        cell = wrap(ucell, provenance=pr)
    return cell


_default_provenance_tracking_lookaside_map = {
    CellType.__new__: _cell_new_provenance_tracking_lookaside,
    tuple.__new__: _tuple_new_provenance_tracking_lookaside,
    MappingKeysView.__iter__: MappingKeysView.__iter__,
    MappingKeysView.__reversed__: MappingKeysView.__reversed__,
    MappingKeysIterator.__next__: MappingKeysIterator.__next__,
}


def _register_provenance_tracking_lookasides(typ, wrapper):
    for meth_name in dir(typ):
        meth = getattr(typ, meth_name)
        if isinstance(meth, (BuiltinMethodType, MethodDescriptorType, WrapperDescriptorType)) and (
            getattr(meth, "__objclass__", None) == typ or (getattr(meth, "__self__", None) == typ)
        ):
            if meth in _default_provenance_tracking_lookaside_map:
                pass
            elif hasattr(wrapper, meth_name):
                _default_provenance_tracking_lookaside_map[meth] = getattr(wrapper, meth_name)
            elif is_opaque(meth):

                def get_unimplemented_fn(meth_name):
                    def unimplemented(*args, **kwargs):
                        raise NotImplementedError(f"{typ}.{meth_name} is not yet supported, please file an issue.")

                    return unimplemented

                _default_provenance_tracking_lookaside_map[meth] = get_unimplemented_fn(meth_name)


# _register_provenance_tracking_lookasides(Sequence, SequenceWrapperMethods)
_register_provenance_tracking_lookasides(tuple, SequenceWrapperMethods)
_register_provenance_tracking_lookasides(list, MutSequenceWrapperMethods)
_register_provenance_tracking_lookasides(dict, MutMappingWrapperMethods)
_register_provenance_tracking_lookasides(collections.OrderedDict, MutMappingWrapperMethods)


# The default function lookaside -- currently it doesn't intercept anything
def default_lookaside(fn, /, *args, **kwargs) -> None | Callable:
    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    try:
        if ctx._with_provenance_tracking:
            lookaside = _default_provenance_tracking_lookaside_map.get(fn, None)
            if lookaside is not None:
                return lookaside
        return _default_lookaside_map.get(fn, None)
    except TypeError:
        # unhashable fn, e.g. weakref to a WeakSet
        return None


#
# Callback registration
#


# To register a callback, map from this enum to a callable. The args and kwargs for each
#   event may be different, as documented below.
class INTERPRETER_CALLBACKS(enum.Enum):
    # Called when a constant (in a CodeType's co_consts) is loaded
    #   (value: Any, /) -> Any
    #   The returned value is used in place of the original value
    # TODO Consider passing the code object and index into co_consts to the callback
    CONST_CALLBACK = enum.auto()

    # Called when a freevar is created
    #   (name: str, value: Any, /, * fn: Callable, idx: int) -> Any
    #   The returned value is used in place of the original value
    FREEVAR_CALLBACK = enum.auto()

    # Called when a global is loaded
    #   (orig_value: Any | WrappedValue, name: str) -> Any | INTERPRETER_SIGNALS
    #   The returned value is loaded instead of the original value
    GLOBAL_CALLBACK = enum.auto()

    # Called when storing into a global variable
    #   (orig_value: Any | WrappedValue, name: str) -> Any
    #   The returned value is stored instead of the original value
    STORE_GLOBAL_CALLBACK = enum.auto()

    # Called when a local variable is loaded
    #   (orig_value: Any | WrappedValue, name: str) -> Any | INTERPRETER_SIGNALS
    #   The returned value is loaded instead of the original value
    LOAD_FAST_CALLBACK = enum.auto()

    # Called when a cell variable is loaded
    #   (orig_value: Any | WrappedValue, name: str) -> Any | INTERPRETER_SIGNALS
    #   The returned value is loaded instead of the original value
    LOAD_DEREF_CALLBACK = enum.auto()

    # Called when storing into a nonlocal variable
    #   (orig_value: Any | WrappedValue, name: str, co_cellvars: tuple[str], co_freevars: tuple[str]) -> Any
    #   The returned value is stored instead of the original value
    STORE_DEREF_CALLBACK = enum.auto()

    # Called when a locals (in localsplus) is created
    #   (name: str, value: Any, /) -> Any
    #   The returned value is used in place of the original value
    LOCAL_CALLBACK = enum.auto()

    # Called when a new WrappedValue is created
    #   (wrapped_value: WrappedValue, value: Any) -> Any
    #   The returned value is recorded as wrapped_value.value
    #   other WrappedValue fields (provenance, ...) are populated
    WRAP_CALLBACK = enum.auto()

    # Called when a WrappedValue might need a proxy
    #   (wrapped_value: WrappedValue) -> None
    # For container proxies this is when it is modified for the first time.
    PROXIFY_CALLBACK = enum.auto()


default_callbacks: dict[INTERPRETER_CALLBACKS, Callable] = {}


def const_callback(value: Any, /) -> Any:
    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    cb: None | Callable = ctx.callback(INTERPRETER_CALLBACKS.CONST_CALLBACK)

    if cb is None:
        return value

    return cb(value)


def get_cell_contents(c):
    # n.b. this will not get the metainformation for wrapped values (should it?)
    c = unwrap(c)
    if c == CellType():
        return Py_NULL()
    else:
        return c.cell_contents


def register_cell_proxy(cell, new_contents):
    ucell = unwrap(cell)
    if ucell == CellType() or new_contents is Py_NULL():
        # cells are a bit tricky to deal with when empty unfortunately, as empty
        # cells raise ValueError when trying to access the contents
        # we might solve it with a custom cell proxy class
        raise NotImplementedError(
            "cannot handle empty cells in proxying, please file an issue to discuss your use case"
        )

    wcontents = cell.attribute_wrappers.get("cell_contents")
    if wcontents is None:
        pr = ProvenanceRecord(PseudoInst.LOAD_ATTR, inputs=[cell.provenance, wrap_const("cell_contents").provenance])
        wcontents = wrap(cell.value.cell_contents, provenance=pr)
        cell.attribute_wrappers["cell_contents"] = wcontents

    wcontents.register_proxy(new_contents)


def freevar_callback(name: str, cell: CellType, /, *, fn: Callable, idx: int) -> CellType:
    assert isinstance(name, str)
    assert wrapped_isinstance(cell, CellType)

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    cb: None | Callable = ctx.callback(INTERPRETER_CALLBACKS.FREEVAR_CALLBACK)

    if cb is None:
        return cell

    old_contents = get_cell_contents(cell)

    # nb. this callback will get a (optionally wrapped) cell and returns a plain value
    new_contents: Any = cb(name, cell, fn=fn, idx=idx)
    if not ctx._with_provenance_tracking:
        return new_contents

    assert not isinstance(
        new_contents, (WrappedValue, CellType)
    ), "freevar_callback should return a plain value, not a WrappedValue or a CellType"

    if new_contents is not old_contents:
        register_cell_proxy(cell, new_contents)
    return cell


def globals_lookup(globals_dict: dict | WrappedValue, name: Any) -> Any | INTERPRETER_SIGNALS:
    # TODO: extend to arbitrary non wrap_const'able types
    assert wrapped_isinstance(name, str)
    assert wrapped_isinstance(globals_dict, dict)

    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

    # this cannot be well implemented in an impl because we would get infinite recursions

    if unwrap(name) in unwrap(globals_dict):
        return _interpret_call(lambda d, k: d[k], globals_dict, name)

    builtins = _interpret_call(lambda d, k: d[k], globals_dict, wrap_const("__builtins__"))

    if isinstance(unwrap(builtins), ModuleType):
        res = _interpret_call(getattr, builtins, name)
        if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return do_raise(NameError(f"name '{unwrap(name)}' is not defined"))
        return res

    # here CPython subscripts without checking that it's a dict, so do we
    res = _interpret_call(lambda d, k: d[k], builtins, name)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED and isinstance(runtimectx._curexc, KeyError):
        return do_raise(NameError(f"name '{unwrap(name)}' is not defined"))

    return res


def global_callback(
    orig_val: Any | WrappedValue,
    name: str,
    callback_type: INTERPRETER_CALLBACKS = INTERPRETER_CALLBACKS.GLOBAL_CALLBACK,
) -> Any | WrappedValue | INTERPRETER_SIGNALS:
    assert isinstance(name, str)

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    cb: None | Callable = ctx.callback(callback_type)

    if cb is None:
        return orig_val
    else:
        return cb(orig_val, name)


def store_deref_callback(value: Any, name: str, co_cellvars: tuple[str], co_freevars: tuple[str]) -> Any:
    assert isinstance(name, str)
    assert isinstance(co_cellvars, tuple)
    assert isinstance(co_freevars, tuple)

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    cb: None | Callable = ctx.callback(INTERPRETER_CALLBACKS.STORE_DEREF_CALLBACK)

    if cb is None:
        return value

    return cb(value, name, co_cellvars, co_freevars)


def load_fast_callback(value: Any, name: str):
    assert isinstance(name, str)

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    cb: None | Callable = ctx.callback(INTERPRETER_CALLBACKS.LOAD_FAST_CALLBACK)

    if cb is None:
        return value

    return cb(value, name)


def load_deref_callback(value: Any, name: str):
    assert isinstance(name, str)

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    cb: None | Callable = ctx.callback(INTERPRETER_CALLBACKS.LOAD_DEREF_CALLBACK)

    if cb is None:
        return value

    return cb(value, name)


def local_callback(name: str, value: Any, /) -> Any:
    assert isinstance(name, str)

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    cb: None | Callable = ctx.callback(INTERPRETER_CALLBACKS.LOCAL_CALLBACK)

    if cb is None:
        return value

    return cb(name, value)


def check_and_append(stack, val):
    if val is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return val
    stack.append(val)


def check_signal(val):
    if val is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return val
    return None


#
# Python opcode handlers (sorted alphabetically)
#


# https://docs.python.org/3.11/library/dis.html#opcode-ASYNC_GEN_WRAP
@register_opcode_handler("ASYNC_GEN_WRAP", min_ver=(3, 11))
def _async_gen_wrap_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    # the next thing will be to yield the value, but we delegate this along with the wrapping to thunder_interpreter_async_generator
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-BEFORE_ASYNC_WITH
@register_opcode_handler("BEFORE_ASYNC_WITH")
def _before_async_with_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

    mgr = stack.pop()

    # python does a "special lookup"
    enter_method = _interpret_call_with_unwrapping(getattr, mgr, "__aenter__")
    if enter_method is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return do_raise(
            TypeError(
                "{'type(mgr).__name__}' object does not support the async context manager protocol (missed __aenter__ method)"
            )
        )
    exit_method = _interpret_call_with_unwrapping(getattr, mgr, "__aexit__")
    if exit_method is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return do_raise(
            TypeError(
                "{'type(mgr).__name__}' object does not support the context manager protocol (missed __aexit__ method)"
            )
        )

    assert callable(enter_method)
    assert callable(exit_method)

    stack.append(exit_method)

    return check_and_append(stack, _interpret_call_with_unwrapping(enter_method))


# https://docs.python.org/3.11/library/dis.html#opcode-BEFORE_WITH
@register_opcode_handler("BEFORE_WITH", min_ver=(3, 11))
def _before_with_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

    mgr = stack.pop()

    # python does a "special lookup"
    enter_method = _interpret_call_with_unwrapping(getattr, mgr, "__enter__")
    if enter_method is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return do_raise(
            TypeError(
                "{'type(mgr).__name__}' object does not support the context manager protocol (missed __enter__ method)"
            )
        )
    exit_method = _interpret_call_with_unwrapping(getattr, mgr, "__exit__")
    if exit_method is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return do_raise(
            TypeError(
                "{'type(mgr).__name__}' object does not support the context manager protocol (missed __exit__ method)"
            )
        )

    assert callable(enter_method)
    assert callable(exit_method)

    stack.append(exit_method)

    return check_and_append(stack, _interpret_call_with_unwrapping(enter_method))


class BINARY_OP(enum.Enum):
    ADD = 0
    AND = 1
    FLOORDIV = 2
    LSHIFT = 3
    MATMUL = 4
    MUL = 5
    MOD = 6
    OR = 7
    POW = 8
    RSHIFT = 9
    SUB = 10
    TRUEDIV = 11
    XOR = 12
    IADD = 13
    IAND = 14
    IFLOORDIV = 15
    ILSHIFT = 16
    IMATMUL = 17
    IMUL = 18
    IMOD = 19
    IOR = 20
    IPOW = 21
    IRSHIFT = 22
    ISUB = 23
    ITRUEDIV = 24
    IXOR = 25


def _binary_op(stack: InterpreterStack, op: BINARY_OP, a, b):
    ops = [
        ("+", "__add__", "__radd__"),
        ("&", "__and__", "__rand__"),
        ("//", "__floordiv__", "__rfloordiv__"),
        ("<<", "__lshift__", "__rlshift__"),
        ("@", "__matmul__", "__rmatmul__"),
        ("*", "__mul__", "__rmul__"),
        ("%", "__mod__", "__rmod__"),
        ("|", "__or__", "__ror__"),
        ("**", "__pow__", "__rpow__"),
        (">>", "__rshift__", "__rrshift__"),
        ("-", "__sub__", "__rsub__"),
        ("/", "__truediv__", "__rtruediv__"),
        ("^", "__xor__", "__rxor__"),
        ("+=", "__iadd__"),
        ("&=", "__iand__"),
        ("//=", "__ifloordiv__"),
        ("<<=", "__ilshift__"),
        ("@=", "__imatmul__"),
        ("*=", "__imul__"),
        ("%=", "__imod__"),
        ("|=", "__ior__"),
        ("**=", "__ipow__"),
        (">>=", "__irshift__"),
        ("-=", "__isub__"),
        ("/=", "__itruediv__"),
        ("^=", "__ixor__"),
    ]

    assert type(op) is BINARY_OP
    idx: int = op.value

    res = Py_NULL()
    binop_name, *_ = ops[idx]
    _, left_method, right_method = ops[idx % BINARY_OP.IADD.value]
    _, inplace_method = ops[idx % BINARY_OP.IADD.value + BINARY_OP.IADD.value]

    left_method, right_method, inplace_method = wrap_consts(left_method, right_method, inplace_method)

    # If the operator is an inplace operator, try to call the inplace method
    if idx >= BINARY_OP.IADD.value:

        def inplace_impl(a, b, inplace_method):
            if hasattr(type(a), inplace_method):
                return getattr(type(a), inplace_method)(a, b)
            return NotImplemented

        res = _interpret_call_with_unwrapping(inplace_impl, a, b, inplace_method)

    # Otherwise, if the method is inplace and not defined, or is an
    # out of place operator, call the out of place operator (__add__/__radd__).
    if idx < BINARY_OP.IADD.value or (res is NotImplemented):

        def outofplace_impl(a, b, left_method, right_method, binop_name):
            if (not hasattr(type(a), left_method)) or (
                (result := getattr(type(a), left_method)(a, b)) is NotImplemented
            ):
                if (not hasattr(type(b), right_method)) or (
                    (result := getattr(type(b), right_method)(b, a)) is NotImplemented
                ):
                    err: TypeError = TypeError(
                        f"unsupported operand type(s) for {binop_name}: '{type(a)}' and '{type(b)}'"
                    )
                    raise err
            return result

        res = _interpret_call_with_unwrapping(outofplace_impl, a, b, left_method, right_method, binop_name)

    # Either one or the other should have been called, and stored to res.
    assert res is not Py_NULL()
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res
    stack.append(res)


def _binary_op_helper(stack: InterpreterStack, op: BINARY_OP):
    b = stack.pop_wrapped()
    a = stack.pop_wrapped()
    return _binary_op(stack, op, a, b)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_ADD
@register_opcode_handler("BINARY_ADD", max_ver=(3, 10))
def _binary_add_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.ADD)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_AND
@register_opcode_handler("BINARY_AND", max_ver=(3, 10))
def _binary_and_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.AND)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_FLOOR_DIVIDE
@register_opcode_handler("BINARY_FLOOR_DIVIDE", max_ver=(3, 10))
def _binary_floor_divide_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.FLOORDIV)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_LSHIFT
@register_opcode_handler("BINARY_LSHIFT", max_ver=(3, 10))
def _binary_lshift_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.LSHIFT)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_MATRIX_MULTIPLY
@register_opcode_handler("BINARY_MATRIX_MULTIPLY", max_ver=(3, 10))
def _binary_matrix_multiply_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.MATMUL)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_MULTIPLY
@register_opcode_handler("BINARY_MULTIPLY", max_ver=(3, 10))
def _binary_multiply_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.MUL)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_MODULO
@register_opcode_handler("BINARY_MODULO", max_ver=(3, 10))
def _binary_modulo_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.MOD)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_OR
@register_opcode_handler("BINARY_OR", max_ver=(3, 10))
def _binary_or_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.OR)


# https://docs.python.org/3.11/library/dis.html#opcode-BINARY_OP
@register_opcode_handler("BINARY_OP", min_ver=(3, 11))
def _binary_op_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    return _binary_op_helper(stack, BINARY_OP(inst.arg))


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_POWER
@register_opcode_handler("BINARY_POWER", max_ver=(3, 10))
def _binary_power_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.POW)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_RSHIFT
@register_opcode_handler("BINARY_RSHIFT", max_ver=(3, 10))
def _binary_rshift_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.RSHIFT)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_SUBTRACT
@register_opcode_handler("BINARY_SUBTRACT", max_ver=(3, 10))
def _binary_subtract_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.SUB)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_TRUE_DIVIDE
@register_opcode_handler("BINARY_TRUE_DIVIDE", max_ver=(3, 10))
def _binary_true_divide_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.TRUEDIV)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_SUBTRACT
@register_opcode_handler("BINARY_XOR", max_ver=(3, 10))
def _binary_xor_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.XOR)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_ADD
@register_opcode_handler("INPLACE_ADD", max_ver=(3, 10))
def _inplace_add_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IADD)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_AND
@register_opcode_handler("INPLACE_AND", max_ver=(3, 10))
def _inplace_and_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IAND)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_FLOOR_DIVIDE
@register_opcode_handler("INPLACE_FLOOR_DIVIDE", max_ver=(3, 10))
def _inplace_floor_divide_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IFLOORDIV)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_LSHIFT
@register_opcode_handler("INPLACE_LSHIFT", max_ver=(3, 10))
def _inplace_lshift_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.ILSHIFT)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_MATRIX_MULTIPLY
@register_opcode_handler("INPLACE_MATRIX_MULTIPLY", max_ver=(3, 10))
def _inplace_matrix_multiply_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IMATMUL)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_MULTIPLY
@register_opcode_handler("INPLACE_MULTIPLY", max_ver=(3, 10))
def _inplace_multiply_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IMUL)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_MODULO
@register_opcode_handler("INPLACE_MODULO", max_ver=(3, 10))
def _inplace_modulo_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IMOD)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_OR
@register_opcode_handler("INPLACE_OR", max_ver=(3, 10))
def _inplace_or_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IOR)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_POWER
@register_opcode_handler("INPLACE_POWER", max_ver=(3, 10))
def _inplace_power_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IPOW)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_RSHIFT
@register_opcode_handler("INPLACE_RSHIFT", max_ver=(3, 10))
def _inplace_rshift_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IRSHIFT)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_SUBTRACT
@register_opcode_handler("INPLACE_SUBTRACT", max_ver=(3, 10))
def _inplace_subtract_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.ISUB)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_TRUE_DIVIDE
@register_opcode_handler("INPLACE_TRUE_DIVIDE", max_ver=(3, 10))
def _inplace_true_divide_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.ITRUEDIV)


# https://docs.python.org/3.10/library/dis.html#opcode-INPLACE_SUBTRACT
@register_opcode_handler("INPLACE_XOR", max_ver=(3, 10))
def _inplace_xor_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    return _binary_op_helper(stack, BINARY_OP.IXOR)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_SUBSCR
@register_opcode_handler("BINARY_SUBSCR")
def _binary_subscr_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop_wrapped()
    tos1 = stack.pop_wrapped()

    def impl(tos1, tos):
        return tos1.__getitem__(tos)

    res = _interpret_call(impl, tos1, tos)

    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res

    return check_and_append(stack, res)


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_CONST_KEY_MAP
@register_opcode_handler("BUILD_CONST_KEY_MAP")
def _build_const_key_map_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int
    count: int = inst.arg

    keys = stack.pop()
    assert len(keys) == count
    values = reversed([stack.pop() for _ in range(count)])
    d: dict = dict(zip(keys, values))
    stack.append(d)


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_LIST
@register_opcode_handler("BUILD_LIST")
def _build_list_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int
    values: list[Any] = list(reversed([stack.pop_wrapped() for _ in range(inst.arg)]))

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        pr = ProvenanceRecord(inst, inputs=[v.provenance for v in values])
        result = wrap([v.value for v in values], provenance=pr)
        result.item_wrappers = values
    else:
        result = values
    stack.append(result)


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_MAP
@register_opcode_handler("BUILD_MAP")
def _build_map_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int
    count: int = inst.arg

    # NOTE The reversed() call below is necessary to handle key collisions properly
    d: dict = {k: v for v, k in reversed(tuple((stack.pop(), stack.pop()) for _ in range(count)))}
    stack.append(d)


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_SET
@register_opcode_handler("BUILD_SET")
def _build_set_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int
    result: set = set(reversed([stack.pop() for _ in range(inst.arg)]))
    stack.append(result)


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_SLICE
@register_opcode_handler("BUILD_SLICE")
def _build_slice_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int

    tos = stack.pop()
    tos1 = stack.pop()
    if inst.arg == 2:
        stack.append(slice(tos1, tos))
    else:
        assert inst.arg == 3, f"Unexpected argument value for build tuple handler {inst.arg=}"
        tos2 = stack.pop()
        stack.append(slice(tos2, tos1, tos))


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_STRING
@register_opcode_handler("BUILD_STRING")
def _build_string_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int

    count: int = inst.arg

    strings: tuple[str, ...] = tuple(reversed(tuple(stack.pop() for _ in range(count))))
    stack.append("".join(strings))


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_TUPLE
@register_opcode_handler("BUILD_TUPLE")
def _build_tuple_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int
    data: list[Any] = [stack.pop_wrapped() for _ in range(inst.arg)][::-1]

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        result = wrapped_build_tuple(data)
    else:
        result = tuple(data)
    stack.append(result)


# https://docs.python.org/3.11/library/dis.html#opcode-KW_NAMES
@register_opcode_handler("KW_NAMES", min_ver=(3, 11))
def _kw_names_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None:
    assert inst.arg is not None
    frame.call_shape_kwnames = co.co_consts[inst.arg]


# NOTE This only accepts positional args
# https://docs.python.org/3.11/library/dis.html#opcode-CALL
@register_opcode_handler("CALL", min_ver=(3, 11))
def _call_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)
    argc: int = inst.arg
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop_wrapped() for _ in range(argc))))
    func_or_self = stack.pop_wrapped()
    func_or_null = stack.pop_wrapped()
    if frame.call_shape_kwnames is not None:
        kwnames = frame.call_shape_kwnames
        assert len(args) >= len(kwnames)
        kwargs = dict(zip(kwnames, args[-len(kwnames) :]))
        args = args[: -len(kwnames)]
        frame.call_shape_kwnames = None
    else:
        kwargs = {}
    if unwrap(func_or_null) is not Py_NULL():
        func = func_or_null
        args = (func_or_self, *args)
    else:
        func = func_or_self

    res = _interpret_call(func, *args, **kwargs)
    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        assert isinstance(res, (WrappedValue, INTERPRETER_SIGNALS)), f"{res} unexpected"

    return check_and_append(stack, res)


# NOTE This only accepts positional args
# https://docs.python.org/3.10/library/dis.html#opcode-CALL_FUNCTION
@register_opcode_handler("CALL_FUNCTION", max_ver=(3, 10))
def _call_function_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)
    argc: int = inst.arg
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop_wrapped() for _ in range(argc))))
    func: Callable = stack.pop_wrapped()
    return check_and_append(stack, _interpret_call(func, *args))


# https://docs.python.org/3.10/library/dis.html#opcode-CALL_FUNCTION_EX
@register_opcode_handler("CALL_FUNCTION_EX")
def _call_function_ex_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    kwargs = stack.pop_wrapped() if inst.arg & 0x01 else {}
    args = stack.pop_wrapped()
    func = stack.pop_wrapped()
    assert wrapped_isinstance(kwargs, Mapping)
    assert wrapped_isinstance(args, Iterable)
    assert wrapped_isinstance(func, Callable)

    if (3, 11) <= sys.version_info:
        null = stack.pop_wrapped()
        assert wrapped_isinstance(null, Py_NULL)

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        args = wrap_args_from_list(args)
        kwargs = wrap_kwargs_from_dict(kwargs)
    return check_and_append(stack, _interpret_call(func, *args, **kwargs))


# https://docs.python.org/3.10/library/dis.html#opcode-CALL_FUNCTION_KW
@register_opcode_handler("CALL_FUNCTION_KW", max_ver=(3, 10))
def _call_function_kw_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    kw_names: tuple[str, ...] = stack.pop_wrapped()

    kwarg_length: INTERPRETER_SIGNALS | int = _interpret_call(len, kw_names)
    if kwarg_length is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return kwarg_length
    kwarg_length = unwrap(kwarg_length)
    assert type(kwarg_length) is int

    kwargs_flat: tuple[Any, ...] = tuple(reversed(tuple(stack.pop_wrapped() for _ in range(kwarg_length))))
    fn_kwargs: dict[str, Any] = {k: v for k, v in zip(unwrap(kw_names), kwargs_flat)}
    assert type(inst.arg) is int
    arg_length: int = inst.arg - kwarg_length
    args = tuple(reversed(tuple(stack.pop_wrapped() for _ in range(arg_length))))
    func: Callable = stack.pop_wrapped()

    return check_and_append(stack, _interpret_call_with_unwrapping(func, *args, **fn_kwargs))


# https://docs.python.org/3.10/library/dis.html#opcode-CALL_METHOD
@register_opcode_handler("CALL_METHOD", max_ver=(3, 10))
def _call_method_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop_wrapped() for _ in range(inst.arg))))
    second_lm = stack.pop_wrapped()
    first_lm = stack.pop_wrapped()
    if unwrap(first_lm) is not Py_NULL():
        meth = first_lm
        args = (second_lm, *args)
    else:
        meth = second_lm

    res = _interpret_call(meth, *args)
    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        assert isinstance(res, (WrappedValue, INTERPRETER_SIGNALS)), f"{res} unexpected"
    return check_and_append(stack, res)


# https://docs.python.org/3.10/library/dis.html#opcode-CONTAINS_OP
# https://docs.python.org/3.10/reference/expressions.html#membership-test-operations
@register_opcode_handler("CONTAINS_OP")
def _contains_op_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop_wrapped()
    tos1 = stack.pop_wrapped()

    assert isinstance(inst.arg, int)
    invert: bool = inst.arg == 1

    def impl(tos, tos1):
        if hasattr(tos, "__contains__"):
            return getattr(tos, "__contains__")(tos1)

        if hasattr(tos, "__iter__"):
            return any(v is tos1 or v == tos1 for v in tos)

        if hasattr(tos, "__getitem__") and hasattr(tos, "__len__"):
            return any(tos[i] is tos1 or tos[i] == tos1 for i in range(len(tos)))

        err: NotImplementedError = NotImplementedError(
            f"__contains__, __iter__, __getitem__, and __len__ are not implemented for input {type(tos)}'"
        )
        raise err

    result = _interpret_call(impl, tos, tos1)
    if result is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return result

    if invert:
        result = _interpret_call(lambda result: not result, result)
        if result is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return result

    stack.append(result)


# https://docs.python.org/3.11/library/dis.html#opcode-CHECK_EXC_MATCH
@register_opcode_handler("CHECK_EXC_MATCH", min_ver=(3, 11))
def _check_exc_match_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    right = stack.pop()
    left = stack[-1]
    assert isinstance(left, BaseException)
    # TODO: raise type error if right is  not an exception
    stack.append(isinstance(left, right))


# TODO See issue "Fix COMPARE_OP handler"
# https://docs.python.org/3.10/library/dis.html#opcode-COMPARE_OP
@register_opcode_handler("COMPARE_OP")
def _compare_op_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    cmp_impls = {
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
    }
    b = stack.pop()
    a = stack.pop()
    assert type(inst.arg) is int
    assert inst.arg < len(dis.cmp_op), f"{inst}, {dis.cmp_op}"

    op = cmp_impls[dis.cmp_op[inst.arg]]
    res: bool = op(unwrap(a), unwrap(b))
    stack.append(res)


# https://docs.python.org/3.11/library/dis.html#opcode-COPY
@register_opcode_handler("COPY", min_ver=(3, 11))
def _copy_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    assert inst.arg >= 1
    stack.append(stack[-inst.arg])


# https://docs.python.org/3.10/library/dis.html#opcode-COPY_DICT_WITHOUT_KEYS
@register_opcode_handler("COPY_DICT_WITHOUT_KEYS", max_ver=(3, 10))
def _copy_dict_without_keys_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    keys = stack.pop()
    assert isinstance(keys, Iterable)
    match_subject = stack[-1]
    assert isinstance(match_subject, MutableMapping)

    # This could be better expressed as:
    # stack.append({k: v for k, v in subject.items() if k not in keys})
    # However in cpython the instruction is implemented with
    # PyDict_DelItem, which calls PyObject_Hash, which can have side effects.
    # So, we begrudgingly follow cpython.
    def impl(keys, match_subject):
        rest = {}
        rest.update(match_subject)
        for k in keys:
            del rest[k]
        return rest

    res = _interpret_call_with_unwrapping(impl, keys, match_subject)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res
    stack.append(res)


# https://docs.python.org/3.11/library/dis.html#opcode-COPY_FREE_VARS
@register_opcode_handler("COPY_FREE_VARS", min_ver=(3, 11))
def _copy_free_vars_handler(inst: dis.Instruction, /, **kwargs) -> None:
    # we already do this when setting up the function call in _interpret_call
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-DELETE_ATTR
@register_opcode_handler("DELETE_ATTR")
def _delete_attr_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    namei: int = inst.arg

    name: str = co.co_names[namei]

    tos: Any = stack.pop()

    def impl():
        delattr(tos, name)

    return _interpret_call_with_unwrapping(impl)


# https://docs.python.org/3.10/library/dis.html#opcode-DELETE_DEREF
@register_opcode_handler("DELETE_DEREF")
def _delete_deref_handler(
    inst: dis.Instruction,
    /,
    stack: InterpreterStack,
    co: CodeType,
    frame: InterpreterFrame,
    **kwargs,
) -> None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)
    i: int = inst.arg
    if sys.version_info < (3, 11):
        i += co.co_nlocals

    assert i >= 0 and i < len(frame.localsplus)

    def impl(cell):
        del cell.cell_contents

    res = _interpret_call(impl, frame.localsplus[i])
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res


# https://docs.python.org/3/library/dis.html#opcode-DELETE_FAST
@register_opcode_handler("DELETE_FAST")
def _delete_fast_handler(inst: dis.Instruction, /, co: CodeType, frame: InterpreterFrame, **kwargs) -> None:
    assert type(inst.arg) is int
    var_num: int = inst.arg
    assert var_num >= 0 and var_num < co.co_nlocals

    # NOTE The deletion just sets the reference in localsplus to an instance of Py_NULL
    frame.localsplus[var_num] = Py_NULL()


# https://docs.python.org/3/library/dis.html#opcode-DELETE_GLOBAL
@register_opcode_handler("DELETE_GLOBAL")
def _delete_global_handler(
    inst: dis.Instruction, /, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    namei: int = inst.arg
    name: str = co.co_names[namei]

    res = _interpret_call(
        lambda frame_globals, name: frame_globals.__delitem__(name),
        frame.globals,
        wrap_const(name),
    )
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res


# https://docs.python.org/3.11/library/dis.html#opcode-DELETE_NAME
@register_opcode_handler("DELETE_NAME")
def _delete_name_handler(
    inst: dis.Instruction, /, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    namei: int = inst.arg
    name: str = co.co_names[namei]

    def impl(names_dict, name):
        if name not in names_dict:
            raise NameError(f"name '{name}' is not defined")
        del names_dict[name]

    return check_signal(_interpret_call(impl, frame.names, wrap_const(name)))


# https://docs.python.org/3.10/library/dis.html#opcode-DELETE_SUBSCR
@register_opcode_handler("DELETE_SUBSCR")
def _delete_subscr_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop_wrapped()
    tos1 = stack.pop_wrapped()

    def impl(tos1, tos):
        tos1.__delitem__(tos)

    res = _interpret_call(impl, tos1, tos)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res


# https://docs.python.org/3.10/library/dis.html#opcode-DICT_MERGE
@register_opcode_handler("DICT_MERGE")
def _dict_merge_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, **kwargs
) -> None | INTERPRETER_SIGNALS:
    a = stack.pop_wrapped()
    b = stack.getitem_wrapped(-1)
    # TODO: Raise inside interpreter
    assert wrapped_isinstance(b, MutableMapping), b
    assert wrapped_isinstance(a, Mapping), a
    if overlap := unwrap(b).keys() & unwrap(a):
        return do_raise(KeyError(f"{co.co_name} got multiple values for keyword argument {next(iter(overlap))}"))
    res = _interpret_call(lambda a, b: b.update(a), a, b)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res


@register_opcode_handler("DICT_UPDATE")
def _dict_update_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int
    a = stack.pop()
    b = stack[-inst.arg]
    assert isinstance(b, MutableMapping), b
    assert isinstance(a, Mapping), a
    b.update(a)


# https://docs.python.org/3.10/library/dis.html#opcode-DUP_TOP
@register_opcode_handler("DUP_TOP", max_ver=(3, 10))
def _dup_top_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    stack.append(stack[-1])


# https://docs.python.org/3.10/library/dis.html#opcode-DUP_TOP_TWO
@register_opcode_handler("DUP_TOP_TWO", max_ver=(3, 10))
def _dup_top_two_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    stack.extend(stack[-2:])


# The exception representation and handling changed between 3.10 and 3.11
# https://docs.python.org/3.10/library/dis.html#opcode-END_ASYNC_FOR
@register_opcode_handler("END_ASYNC_FOR", max_ver=(3, 10))
def _end_async_for_handler_3_10(
    inst: dis.Instruction,
    /,
    stack: InterpreterStack,
    try_stack: list[PyTryBlock],
    inst_ptr: int,
    frame: InterpreterFrame,
    exception_stack: list,
    **kwargs,
) -> None | INTERPRETER_SIGNALS:
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    assert inst.arg is None

    exc = stack.pop()
    assert issubclass(exc, BaseException)

    if issubclass(exc, StopAsyncIteration):
        try_block = try_stack.pop()
        assert try_block.typ == PyTryBlock.EXCEPT_HANDLER_TYPE
        assert try_block.level + 3 <= len(stack)
        assert exception_stack

        assert len(stack) >= try_block.level + 3
        del stack[try_block.level + 3 :]
        exc_type = frame.interpreter_stack.pop()  # we ignore that and asume == type(exc_value)
        exc_value = frame.interpreter_stack.pop()
        exc_traceback = frame.interpreter_stack.pop()
        if exc_value != None:
            exc_value.__traceback__ = exc_traceback
        assert runtimectx.exception_stack
        # CPython sets exc_info->exc_type/value/traceback
        # see RuntimeCtx inititalization of exception_stack for more info
        runtimectx.exception_stack[-1] = exc_value  # replace the exc_info
        # Python 3.10 has `continue` here, but there is no code except the else
        stack.pop()
        return
    else:
        val = stack.pop()
        tb = stack.pop()
        assert isinstance(val, BaseException)
        val.__traceback__ = tb
        runtimectx.curexc = val
        return INTERPRETER_SIGNALS.EXCEPTION_RAISED


# https://docs.python.org/3.11/library/dis.html#opcode-END_ASYNC_FOR
@register_opcode_handler("END_ASYNC_FOR", min_ver=(3, 11))
def _end_async_for_handler_3_11(
    inst: dis.Instruction,
    /,
    stack: InterpreterStack,
    try_stack: list[PyTryBlock],
    inst_ptr: int,
    frame: InterpreterFrame,
    exception_stack: list,
    **kwargs,
) -> None | INTERPRETER_SIGNALS:
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    assert inst.arg is None

    val = stack.pop()
    assert isinstance(val, BaseException)

    if isinstance(val, StopAsyncIteration):
        stack.pop()
        return
    else:
        runtimectx.curexc = val
        return INTERPRETER_SIGNALS.EXCEPTION_RAISED


# https://docs.python.org/3.10/library/dis.html#opcode-EXTENDED_ARG
@register_opcode_handler("EXTENDED_ARG")
def _extended_arg_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-FORMAT_VALUE
# TODO Extend the implementation to
@register_opcode_handler("FORMAT_VALUE")
def _format_value_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    FVC_MASK: int = 0x3
    FVC_NONE: int = 0x0
    FVC_STR: int = 0x1
    FVC_REPR: int = 0x2
    FVC_ASCII: int = 0x3
    FVS_MASK: int = 0x4
    FVS_HAVE_SPEC: int = 0x4

    assert type(inst.arg) is int
    flags: int = inst.arg
    assert isinstance(flags, int)

    value = stack.pop()
    fmt_spec = None
    if (flags & FVS_MASK) == FVS_HAVE_SPEC:
        fmt_spec = value
        value = stack.pop()

    _case: int = flags & FVC_MASK

    def impl(value, fmt_spec):
        # NOTE `match` was only introduced in Python 3.10, but we support Python 3.9
        if _case == FVC_NONE:
            pass
        elif _case == FVC_STR:
            value = str(value)
        elif _case == FVC_REPR:
            value = repr(value)
        else:
            assert _case == FVC_ASCII, f"Unknown FVC_MASK in FORMAT_VALUE"
            value = ascii(value)

        formatted: str = format(value, fmt_spec) if fmt_spec is not None else format(value)
        return formatted

    return check_and_append(stack, _interpret_call_with_unwrapping(impl, value, fmt_spec))


# https://docs.python.org/3.10/library/dis.html#opcode-FOR_ITER
@register_opcode_handler("FOR_ITER")
def _for_iter_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    delta: int = inst.arg

    tos: Iterator = stack.getitem_wrapped(-1)
    assert wrapped_isinstance(tos, Iterator), f"got {type(unwrap(tos))} instead of Iterator"

    def _next_impl(tos):
        return next(tos)

    v: Any
    r = _interpret_call(_next_impl, tos)

    if r is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        ctx = get_interpreterruntimectx()
        if isinstance(ctx.curexc, StopIteration):
            stack.pop_wrapped()
            return inst_ptr + delta + 1
        return r

    stack.append(r)


# https://docs.python.org/3.10/library/dis.html#opcode-GET_AITER
@register_opcode_handler("GET_AITER")
def _get_aiter_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop()

    def impl():
        if not hasattr(tos, "__aiter__"):
            raise TypeError(f"'async for' requires an object with __aiter__ method, got {type(tos).__name__}")
        ait = tos.__aiter__()
        if not hasattr(ait, "__anext__"):
            raise TypeError(
                f"'async for' received an object from __aiter__ that does not implement __anext__: {type(ait).__name__}"
            )
        return ait

    return check_and_append(stack, _interpret_call_with_unwrapping(impl))


def is_coro_or_iter(o):
    if type(o) is CoroutineType:
        return True
    codeobj = getattr(o, "gi_code", None)
    if isinstance(codeobj, CodeType) and codeobj.co_flags & inspect.CO_COROUTINE:
        return True
    return False


# PyCoro_GetAwaitableIter:
def get_awaitable_iter(tos):
    if is_coro_or_iter(tos):
        return tos

    if not hasattr(tos, "__await__"):
        raise TypeError(f"object {type(tos).__name__} can't be used in 'await' expression")

    res = tos.__await__()

    if is_coro_or_iter(res):
        raise TypeError("__await__() returned a coroutine")

    # check for iterator
    if not hasattr(res, "__next__"):
        raise TypeError(f"__await__() returned non-iterator of type '{type(res).__name__}'")

    return res


# https://docs.python.org/3.10/library/dis.html#opcode-GET_ANEXT
@register_opcode_handler("GET_ANEXT")
def _get_anext_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack[-1]

    def impl():
        an = tos.__anext__()
        return get_awaitable_iter(an)

    return check_and_append(stack, _interpret_call_with_unwrapping(impl))


# https://docs.python.org/3.10/library/dis.html#opcode-GET_AWAITABLE
@register_opcode_handler("GET_AWAITABLE")
def _get_awaitable_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop()
    return check_and_append(stack, _interpret_call_with_unwrapping(get_awaitable_iter, tos))


def _iter_impl(obj):
    def impl():
        return obj.__iter__()

    return impl


# https://docs.python.org/3.10/library/dis.html#opcode-GET_ITER
@register_opcode_handler("GET_ITER")
def _get_iter_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop_wrapped()
    return check_and_append(stack, _interpret_call(iter, tos))


# https://docs.python.org/3.10/library/dis.html#opcode-GET_LEN
@register_opcode_handler("GET_LEN")
def _get_len_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    def impl(tos):
        return len(tos)

    ret = _interpret_call_with_unwrapping(impl, stack[-1])
    if ret is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return ret
    stack.append(ret)


# NOTE (mruberry) The actual implementation of IMPORT_FROM is quite complicated, and there doesn't appear
#   to be a Python exposure for the operation (unlike __import__ for IMPORT_NAME)
#   There may be a better way to model this, including by just calling "from module import name"
#   directly -- are we really worried that programs will put tensor operations in import hooks?
# https://docs.python.org/3.10/library/dis.html#opcode-IMPORT_FROM
@register_opcode_handler("IMPORT_FROM")
def _import_from_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)
    namei: int = inst.arg

    # NOTE The stack is peeked, not popped
    module = stack.getitem_wrapped(-1)
    name: WrappedValue = wrap_const(co.co_names[namei])

    def impl(module, name):
        if hasattr(module, name):
            return getattr(module, name)
        # TODO: the below needs a test
        # CPython links to https://bugs.python.org/issue17636
        # TODO: check that module.__name__ is a valid name
        fullname = f"{module.__name__}.{name}"
        return __import__(fullname)

    return check_and_append(stack, _interpret_call(impl, module, name))


# https://docs.python.org/3.10/library/dis.html#opcode-IMPORT_NAME
@register_opcode_handler("IMPORT_NAME")
def _import_name_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)
    namei: int = inst.arg

    module_name: str = co.co_names[namei]

    fromlist = stack.pop()
    level = stack.pop()

    # relative imports rely on the the current module's name (from the frame stac?)
    # but that isn't available if we use impl, so we resolve it here.
    if level > 0:  # relative import
        # cannot do this in impl easily, but error handling?
        # TODO: model this more after resove_name in CPython's Python/import.c
        def get_current_name(globals):
            package = globals.get("__package__")
            if package is None:
                spec = globals.get("__spec__")
                if spec is not None:
                    package = spec.parent
            if package is None:
                package = globals["__name__"]
            return package

        current_name = _interpret_call(get_current_name, frame.globals)
        if current_name is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return do_raise(KeyError("'__name__' not in globals"))
        current_name = unwrap(current_name)
        name_parts = current_name.split(".")
        module_parts = name_parts[: len(name_parts) - level + 1]
        if module_name:  # from . import foo will have '' as module_name
            module_parts.append(module_name)
        module_name = ".".join(module_parts)
        level = 0
    else:
        current_name = "n/a"

    def impl(module_name, fromlist, level):
        module = __import__(module_name, fromlist=fromlist, level=level)
        return module

    return check_and_append(stack, _interpret_call_with_unwrapping(impl, module_name, fromlist, level))


# https://docs.python.org/3.10/library/dis.html#opcode-IMPORT_STAR
@register_opcode_handler("IMPORT_STAR")
def _import_star_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    # The module is actually imported from another instruction.
    # This instruction can only be parsed at top level and modify globals,
    # since localsplus is of fixed length/positions. It can't be parsed inside a function.

    # `from operator import *` compiles as
    #  0 LOAD_CONST    0 (0)
    #  2 LOAD_CONST    1 (('*',))
    #  4 IMPORT_NAME   0 (operator)
    #  6 IMPORT_STAR

    module = stack.pop()
    assert isinstance(module, ModuleType)

    # Get the locals of the current frame, not the frame created by interpreted impl() below.
    _locals = _interpret_call_with_unwrapping(locals)
    if _locals is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return _locals
    assert isinstance(_locals, dict)

    # For every name in __all__ if present in the module, or every name in __dict__ not
    # starting with _ if __all__ is not present, add the name to the current locals() dict,
    # and produce the same exceptions as cpython would.
    def impl():
        skip_leading_underscores = False
        all_names = getattr(module, "__all__", None)
        if all_names is None:
            if not hasattr(module, "__dict__") or not hasattr(module.__dict__, "keys"):
                raise ImportError("from-import-* object has no __dict__ and no __all__")
            skip_leading_underscores = True
            all_names = module.__dict__.keys()

        assert all_names is not None
        for name in all_names:
            if not isinstance(name, str):
                modname = module.__name__
                if not isinstance(modname, str):
                    raise TypeError(f"module __name__ must be a string, not {modname}")
                # NOTE import * has different error messages if trying to acquire from __dict__ vs. __all__
                if skip_leading_underscores:
                    raise TypeError(f"Key in {modname}.__dict__ must be str, not {type(name)}")
                raise TypeError(f"Item in {modname}.__all__ must be str, not {type(name)}")

            if skip_leading_underscores and name.startswith("_"):
                continue
            _locals[name] = getattr(module, name)

    res = _interpret_call_with_unwrapping(impl)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res

    return None


# https://docs.python.org/3.10/library/dis.html#opcode-IS_OP
@register_opcode_handler("IS_OP")
def _is_op_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()
    stack.append(a is not b if inst.arg == 1 else a is b)


# https://docs.python.org/3.10/library/dis.html#opcode-JUMP_ABSOLUTE
@register_opcode_handler("JUMP_ABSOLUTE", max_ver=(3, 10))
def _jump_absolute_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    assert type(inst.arg) is int
    target: int = inst.arg
    return target


# https://docs.python.org/3.10/library/dis.html#opcode-JUMP_FORWARD
@register_opcode_handler("JUMP_FORWARD")
def _jump_forward_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    assert type(inst.arg) is int
    delta: int = inst.arg
    return inst_ptr + delta + 1


# https://docs.python.org/3.11/library/dis.html#opcode-JUMP_BACKWARD
@register_opcode_handler("JUMP_BACKWARD", min_ver=(3, 11))
def _jump_backward_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    assert type(inst.arg) is int
    delta: int = inst.arg
    return inst_ptr - delta + 1


# https://docs.python.org/3.11/library/dis.html#opcode-JUMP_BACKWARD_NO_INTERRUPT
@register_opcode_handler("JUMP_BACKWARD_NO_INTERRUPT", min_ver=(3, 11))
def _jump_backward_no_interrupt_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    assert type(inst.arg) is int
    delta: int = inst.arg
    return inst_ptr - delta + 1


# https://docs.python.org/3.10/library/dis.html#opcode-JUMP_IF_NOT_EXC_MATCH
@register_opcode_handler("JUMP_IF_NOT_EXC_MATCH", max_ver=(3, 10))
def _jump_if_not_exc_match_handler(
    inst: dis.Instruction, /, inst_ptr: int, stack: InterpreterStack, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    target: int = inst.arg

    right = stack.pop()
    left = stack.pop()
    if not isinstance(right, tuple):
        right = (right,)
    if any((isinstance(left, aright) or issubclass(left, aright)) for aright in right):
        return None
    return target


# https://docs.python.org/3.10/library/dis.html#opcode-JUMP_TRUE_OR_POP
# https://docs.python.org/3.11/library/dis.html#opcode-JUMP_TRUE_OR_POP
@register_opcode_handler("JUMP_IF_TRUE_OR_POP")
def _jump_if_true_or_pop_handler(
    inst: dis.Instruction, /, inst_ptr: int, stack: InterpreterStack, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    target: int = inst.arg

    tos = stack.getitem_wrapped(-1)

    cnd: bool | INTERPRETER_SIGNALS = _interpret_call(bool, tos)
    if cnd is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return cnd

    if not unwrap(cnd):
        stack.pop_wrapped()
        return None

    if sys.version_info >= (3, 11):
        target += inst_ptr + 1
    return target


# https://docs.python.org/3.10/library/dis.html#opcode-JUMP_FALSE_OR_POP
# https://docs.python.org/3.11/library/dis.html#opcode-JUMP_FALSE_OR_POP
@register_opcode_handler("JUMP_IF_FALSE_OR_POP")
def _jump_if_false_or_pop_handler(
    inst: dis.Instruction, /, inst_ptr: int, stack: InterpreterStack, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    target: int = inst.arg

    tos = stack.getitem_wrapped(-1)

    cnd: bool | INTERPRETER_SIGNALS = _interpret_call(bool, tos)
    if cnd is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return cnd

    if unwrap(cnd):
        stack.pop_wrapped()
        return None

    if sys.version_info >= (3, 11):
        target += inst_ptr + 1
    return target


# https://docs.python.org/3.10/library/dis.html#opcode-LIST_APPEND
@register_opcode_handler("LIST_APPEND")
def _list_append_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    i: int = inst.arg

    # NOTE Doesn't pop the list that's extended
    tos = stack.pop_wrapped()
    l: list = stack.getitem_wrapped(-i)

    assert wrapped_isinstance(l, list)

    def impl(l, tos):
        l.append(tos)

    res = _interpret_call(impl, l, tos)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res


# https://docs.python.org/3.10/library/dis.html#opcode-LIST_EXTEND
@register_opcode_handler("LIST_EXTEND")
def _list_extend_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    i: int = inst.arg

    # NOTE Doesn't pop the list that's extended
    tos = stack.pop_wrapped()
    l: list = stack.getitem_wrapped(-i)

    # NOTE tos does not have to be a list
    assert wrapped_isinstance(l, list)
    res = _interpret_call(lambda l1, l2: l1.extend(l2), l, tos)

    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res


# https://docs.python.org/3.10/library/dis.html#opcode-LIST_TO_TUPLE
@register_opcode_handler("LIST_TO_TUPLE")
def _list_to_tuple_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    tos = stack.pop_wrapped()
    assert wrapped_isinstance(tos, list)
    populate_item_wrappers(tos)

    res = tuple(unwrap(tos))

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        pr = ProvenanceRecord(inst, inputs=[tos.provenance])
        res = wrap(res, provenance=pr)
        res.item_wrappers = tos.item_wrappers[:]

    stack.append(res)


# https://docs.python.org/3.13/library/dis.html#opcode-LOAD_ASSERTION_ERROR
@register_opcode_handler("LOAD_ASSERTION_ERROR")
def _load_assertion_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    stack.append(wrap_const(AssertionError))


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_ATTR
@register_opcode_handler("LOAD_ATTR")
def _load_attr_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int

    a = stack.pop_wrapped()
    name: WrappedValue = wrap_const(co.co_names[inst.arg])

    return check_and_append(stack, _interpret_call(getattr, a, name))


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_BUILD_CLASS
@register_opcode_handler("LOAD_BUILD_CLASS")
def _load_build_class_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    build_class = globals_lookup(frame.globals, wrap_const("__build_class__"))
    return check_and_append(stack, build_class)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_CLOSURE
@register_opcode_handler("LOAD_CLOSURE")
def _load_closure_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None:
    assert type(inst.arg) is int
    i: int = inst.arg

    if sys.version_info < (3, 11):
        i += co.co_nlocals

    assert i >= 0 and i < len(frame.localsplus)
    val = frame.localsplus[i]

    stack.append(val)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_CONST
@register_opcode_handler("LOAD_CONST")
def _load_const_handler(inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, **kwargs) -> None:
    assert type(inst.arg) is int

    constant = co.co_consts[inst.arg]
    constant = wrap_const(constant)
    constant = const_callback(constant)
    stack.append(constant)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_DEREF
@register_opcode_handler("LOAD_DEREF")
def _load_deref_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    i: int = inst.arg

    if sys.version_info < (3, 11):
        i += co.co_nlocals

    assert i >= 0 and i < len(frame.localsplus)
    cell = frame.localsplus[i]
    name: str = frame.get_localsplus_name(i)

    # it seems that the only way to check for an empty cell (short of
    # try... except) is comparison to another empty cell
    if unwrap(cell) == CellType():
        return do_raise(
            NameError(f"free variable '{frame.get_localsplus_name(i)}' referenced before assignment in enclosing scope")
        )

    val = _interpret_call(getattr, cell, wrap_const("cell_contents"))
    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        assert isinstance(val, WrappedValue), f"{val}"
        if isinstance(val.value, list):
            assert isinstance(val.item_wrappers, Sized)
            assert len(val.value) == len(val.item_wrappers)

    val = load_deref_callback(val, name)

    return check_and_append(stack, val)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_FAST
@register_opcode_handler("LOAD_FAST")
def _load_fast_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)
    var_num: int = inst.arg
    assert var_num >= 0 and var_num < len(frame.localsplus)

    val: Any = frame.localsplus[var_num]
    name: str = frame.get_localsplus_name(var_num)

    # empty local variable slots are initialized to Py_NULL()
    if isinstance(val, Py_NULL):
        return do_raise(UnboundLocalError(f"local variable '{name}' referenced before assignment"))

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        assert isinstance(val, WrappedValue), f"unexpected value of type {type(val)}, {val}, {inst}"

    val = load_fast_callback(val, name)

    return check_and_append(stack, val)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_GLOBAL
@register_opcode_handler("LOAD_GLOBAL")
def _load_global_handler(
    inst: dis.Instruction,
    /,
    stack: InterpreterStack,
    co: CodeType,
    globals_dict: dict[str, Any],
    **kwargs,
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    idx = inst.arg
    if (3, 11) <= sys.version_info:
        idx = idx // 2
    co_name: str = co.co_names[idx]

    obj: Any | WrappedValue = globals_lookup(globals_dict, wrap_const(co_name))
    if obj is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return do_raise(NameError(f"name '{co_name}' is not defined"))
    else:
        obj = global_callback(obj, co_name)
        if obj is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return obj

    if (3, 11) <= sys.version_info:
        # for 3.11+, the lowest bit indicates whether a NULL should be pushed
        if inst.arg & 1:
            stack.append(wrap_const(Py_NULL()))

    return check_and_append(stack, obj)


# https://docs.python.org/3.11/library/dis.html#opcode-LOAD_METHOD
@register_opcode_handler("LOAD_METHOD")
def _load_method_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    name = wrap_const(co.co_names[inst.arg])
    obj = stack.pop_wrapped()

    meth = _interpret_call(getattr, obj, name)
    if meth is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return meth

    umeth = unwrap(meth)

    if hasattr(umeth, "__self__") and unwrap(obj) is umeth.__self__:
        populate_attribute_wrapper(meth, "__self__", obj)

    if inspect.ismethod(umeth):
        func_attr = _interpret_call(getattr, meth, wrap_const("__func__"))
        if func_attr is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return func_attr
        # meth.__self__ is obj for regular methods but cls for class methods
        self_attr = _interpret_call(getattr, meth, wrap_const("__self__"))
        if self_attr is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return self_attr
        stack.append(func_attr)
        stack.append(self_attr)
    else:
        stack.append(wrap_const(Py_NULL()))
        stack.append(meth)


# https://docs.python.org/3.11/library/dis.html#opcode-LOAD_NAME
@register_opcode_handler("LOAD_NAME")
def _load_name_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)
    namei: int = inst.arg
    name: str = co.co_names[namei]

    value: Any

    if name in unwrap(frame.names):
        value = _interpret_call(lambda d, k: d[k], frame.names, wrap_const(name))
    else:
        # Look up globals, then builtins.
        value = globals_lookup(frame.globals, wrap_const(name))
        if value is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return do_raise(NameError(f"{name=} is not defined"))

    return check_and_append(stack, value)


# https://docs.python.org/3.11/library/dis.html#opcode-MAKE_CELL
@register_opcode_handler("MAKE_CELL", min_ver=(3, 11))
def _make_cell_handler(inst: dis.Instruction, /, frame: InterpreterFrame, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg
    assert i >= 0 and i < len(frame.localsplus)
    val = frame.localsplus[i]

    if isinstance(val, Py_NULL):
        # empty local variable slots (Py_NULL()) produce an empty cell
        c = _interpret_call(CellType)
        assert c is not INTERPRETER_SIGNALS.EXCEPTION_RAISED
    else:
        # wrap the current val into a cell
        c = _interpret_call(CellType, val)
        assert c is not INTERPRETER_SIGNALS.EXCEPTION_RAISED

    frame.localsplus[i] = c


# https://docs.python.org/3.10/library/dis.html#opcode-MAKE_FUNCTION
@register_opcode_handler("MAKE_FUNCTION")
def _make_function_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, globals_dict: dict[str, Any], **kwargs
) -> None:
    assert type(inst.arg) is int

    if sys.version_info < (3, 11):
        name = stack.pop()
    else:
        name = ""

    fn_co: CodeType = unwrap(stack.pop_wrapped())
    name = fn_co.co_name

    ctx: InterpreterCompileCtx = get_interpretercompilectx()

    if inst.arg & 0x08:
        # Python will have built at tuple of cell vars
        # (via STORE_DEREF, LOAD_CLOSURE)
        closure = _interpret_call(tuple, stack.pop_wrapped())
        if ctx._with_provenance_tracking:
            for i, cell in enumerate(closure.value):
                assert isinstance(cell, CellType)
                if cell != CellType():
                    cell_wrapper = closure.item_wrappers[i]
                    # TODO: investigate: the store_deref and load_closure does this to us, apparently
                    wrapped_contents = cell.cell_contents
                    if isinstance(wrapped_contents, WrappedValue):
                        cell.cell_contents = cell.cell_contents.value
                        populate_attribute_wrapper(cell_wrapper, "cell_contents", wrapped_contents)
    else:
        closure = None

    if inst.arg & 0x04:
        annotations = stack.pop()
        assert type(annotations) is tuple and len(annotations) % 2 == 0
        annotations = dict(zip(annotations[::2], annotations[1::2]))
    else:
        annotations = None

    if inst.arg & 0x02:
        kwdefaults = stack.pop()
        assert type(kwdefaults) == dict
    else:
        kwdefaults = None

    if inst.arg & 0x01:
        argdefs = stack.pop()
        assert type(argdefs) == tuple
    else:
        argdefs = None

    fn = FunctionType(fn_co, unwrap(globals_dict), name, argdefs=argdefs, closure=unwrap(closure))

    if kwdefaults is not None:
        fn.__kwdefaults__ = kwdefaults

    if annotations is not None:
        fn.__annotations__ = annotations

    stack.append(fn)


# https://docs.python.org/3.10/library/dis.html#opcode-MAP_ADD
@register_opcode_handler("MAP_ADD")
def _map_add_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg

    # NOTE Doesn't pop the dict that's extended
    tos = stack.pop()
    tos1 = stack.pop()
    d: dict = stack[-i]

    assert type(d) is dict, type(d)
    d[tos1] = tos


# NOTE: The behavior should match match_class() in cpython.
def _match_class_impl(kw_names, typ, subject, count) -> tuple | None:
    seen = []
    attrs = []

    if not isinstance(subject, typ):
        return None

    if hasattr(typ, "__match_args__"):
        match_self = False
        match_args = typ.__match_args__
    else:
        match_self = True
        match_args = ()

    if not type(match_args) is tuple:
        raise TypeError(f"{typ.__name__}.__match_args__ must be a tuple (got {type(match_args)})")

    allowed = 1 if match_self else len(match_args)
    if allowed < count:
        plural = "" if allowed == 1 else "s"
        raise TypeError(f"{typ.__name__}() accepts {allowed} positional sub-pattern{plural} ({count} given)")

    if not match_self:
        # Match positional sub-patterns
        for attr_name in match_args:
            if not isinstance(attr_name, str):
                raise TypeError(f"__match_args__ elements must be strings (got {type(attr_name).__name__})")

            if attr_name in seen:
                raise TypeError(f"{typ.__name__}() got multiple sub-patterns for attribute {attr_name}")
            seen.append(attr_name)

            try:
                attrs.append(getattr(subject, attr_name))
            except AttributeError:
                return None

    # Match keyword subpatterns
    for attr_name in kw_names:
        assert isinstance(attr_name, str)

        if attr_name in seen:
            raise TypeError(f"{typ.__name__}() got multiple sub-patterns for attribute {attr_name}")
        seen.append(attr_name)

        try:
            attrs.append(getattr(subject, attr_name))
        except AttributeError:
            return None

    return tuple(attrs)


# https://docs.python.org/3.10/library/dis.html#opcode-MATCH_CLASS
@register_opcode_handler("MATCH_CLASS")
def _match_class_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    count: int = inst.arg

    kw_attr_names = stack.pop()
    typ = stack.pop()
    subject = stack.pop()
    assert type(kw_attr_names) is tuple

    ret = _interpret_call_with_unwrapping(_match_class_impl, kw_attr_names, typ, subject, count)
    if ret is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return ret

    stack.append(ret if ret is not None else stack[-2])
    if sys.version_info < (3, 11):
        stack.append(True if ret is not None else False)


def _match_keys_impl(keys, subject):
    marker = object()
    values_or_none = []
    for k in keys:
        v = subject.get(k, marker)
        if v is not marker:
            values_or_none.append(v)
        else:
            values_or_none = None
            break

    return tuple(values_or_none) if values_or_none is not None else None


# https://docs.python.org/3.10/library/dis.html#opcode-MATCH_KEYS
@register_opcode_handler("MATCH_KEYS")
def _match_keys_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    keys = stack[-1]
    subject = stack[-2]
    assert isinstance(keys, tuple)
    assert isinstance(subject, Mapping)

    ret = _interpret_call_with_unwrapping(_match_keys_impl, keys, subject)
    if ret is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return ret

    stack.append(ret)
    if sys.version_info < (3, 11):
        stack.append(ret is not None)


# https://docs.python.org/3.10/library/dis.html#opcode-MATCH_MAPPING
@register_opcode_handler("MATCH_MAPPING")
def _match_mapping_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    # NOTE: We cannot check tp_flags but this is, according to the docs, close enough.
    # tp_flags is a bitfield containing on the type containing information about what protocols the type supports. We
    # do not model it, because it constantly changes from version to version, and inheritance of each flag is complicated.
    # Thankfully, somebody else seems to have had this conversation with the cpython devs before us, and the following
    # is the documented workaround.
    stack.push(isinstance(stack[-1], Mapping))


# https://docs.python.org/3.10/library/dis.html#opcode-MATCH_SEQUENCE
@register_opcode_handler("MATCH_SEQUENCE")
def _match_sequence_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    # NOTE: We cannot check tp_flags but this is, according to the docs, close enough.
    # See the comment on MATCH_MAPPING.
    supported_sequence: bool = isinstance(stack[-1], Sequence) and not isinstance(stack[-1], (str, bytes, bytearray))
    stack.push(supported_sequence)


# https://docs.python.org/3.10/library/dis.html#opcode-NOP
@register_opcode_handler("NOP")
def _nop_handler(inst: dis.Instruction, /, **kwargs) -> None:
    pass


# https://docs.python.org/3.11/library/dis.html#opcode-RESUME
@register_opcode_handler("RESUME", min_ver=(3, 11))
def _resume_handler(inst: dis.Instruction, /, **kwargs) -> None:
    pass


# https://docs.python.org/3.11/library/dis.html#opcode-PRECALL
@register_opcode_handler("PRECALL", min_ver=(3, 11), max_ver=(3, 11))
def _precall_handler(inst: dis.Instruction, /, co: CodeType, **kwargs) -> None:
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-POP_BLOCK
@register_opcode_handler("POP_BLOCK", max_ver=(3, 10))
def _pop_block_handler(inst: dis.Instruction, /, try_stack: list[PyTryBlock], **kwargs) -> None:
    try_stack.pop()


# https://docs.python.org/3.10/library/dis.html#opcode-POP_EXCEPT
# Note that Python <= 3.10 always has type/value/traceback on the stack as three items
#           Python >= 3.11 only has one (but has the split in other places)
@register_opcode_handler("POP_EXCEPT", max_ver=(3, 10))
def _pop_except_handler_3_10(
    inst: dis.Instruction, /, stack: InterpreterStack, try_stack: list[PyTryBlock], exception_stack: list, **kwargs
) -> None:
    try_block = try_stack.pop()
    assert try_block.typ == PyTryBlock.EXCEPT_HANDLER_TYPE
    assert try_block.level + 3 <= len(stack) <= try_block.level + 4
    assert exception_stack
    exc_type = stack.pop()
    exc_value = stack.pop()
    exc_traceback = stack.pop()
    # we assume that type and traceback are set on exc_value already (check?)
    # CPython sets exc_info->exc_type/value/traceback, see RuntimeCtx inititalization of exception_stack for more info
    exception_stack[-1] = exc_value


# https://docs.python.org/3.11/library/dis.html#opcode-POP_EXCEPT
@register_opcode_handler("POP_EXCEPT", min_ver=(3, 11))
def _pop_except_handler_3_11(
    inst: dis.Instruction, /, stack: InterpreterStack, try_stack: list[PyTryBlock], exception_stack: list, **kwargs
) -> None:
    exc_value = stack.pop()
    # CPython sets exc_info->exc_type/value/traceback, see RuntimeCtx inititalization of exception_stack for more info
    exception_stack[-1] = exc_value


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_FALSE
@register_opcode_handler("POP_JUMP_BACKWARD_IF_FALSE", min_ver=(3, 11))
def _pop_jump_backward_if_false_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)

    tos = stack.pop()

    cnd: bool | INTERPRETER_SIGNALS = _interpret_call_with_unwrapping(bool, tos)
    if cnd is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return cnd

    if not cnd:
        return inst_ptr - inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_NONE
@register_opcode_handler("POP_JUMP_BACKWARD_IF_NONE", min_ver=(3, 11))
def _pop_jump_backward_if_none_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()

    if tos is None:
        return inst_ptr - inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_NOT_NONE
@register_opcode_handler("POP_JUMP_BACKWARD_IF_NOT_NONE", min_ver=(3, 11))
def _pop_jump_backward_if_not_none_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()

    if tos is not None:
        return inst_ptr - inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_TRUE
@register_opcode_handler("POP_JUMP_BACKWARD_IF_TRUE", min_ver=(3, 11))
def _pop_jump_backward_if_true_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)

    tos = stack.pop()

    cnd: bool | INTERPRETER_SIGNALS = _interpret_call_with_unwrapping(bool, tos)
    if cnd is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return cnd

    if cnd:
        return inst_ptr - inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_FALSE
@register_opcode_handler("POP_JUMP_FORWARD_IF_FALSE", min_ver=(3, 11))
def _pop_jump_forward_if_false_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)

    tos = stack.pop_wrapped()

    cnd: bool | INTERPRETER_SIGNALS = _interpret_call(bool, tos)
    if cnd is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return cnd

    if not unwrap(cnd):
        return inst_ptr + inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_TRUE
@register_opcode_handler("POP_JUMP_FORWARD_IF_TRUE", min_ver=(3, 11))
def _pop_jump_forward_if_true_handler_3_11(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert isinstance(inst.arg, int)

    tos = stack.pop_wrapped()

    cnd: bool | INTERPRETER_SIGNALS = _interpret_call(bool, tos)
    if cnd is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return cnd

    if unwrap(cnd):
        return inst_ptr + inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_NONE
@register_opcode_handler("POP_JUMP_FORWARD_IF_NONE", min_ver=(3, 11))
def _pop_jump_forward_if_none_handler_3_11(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()
    if tos is None:
        return inst_ptr + inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_NOT_NONE
@register_opcode_handler("POP_JUMP_FORWARD_IF_NOT_NONE", min_ver=(3, 11))
def _pop_jump_forward_if_none_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()
    if tos is not None:
        return inst_ptr + inst.arg + 1

    return None


@register_opcode_handler("POP_JUMP_IF_FALSE", max_ver=(3, 10))
def _pop_jump_if_false_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int

    tos = stack.pop_wrapped()

    res = _interpret_call(bool, tos)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res
    ures = unwrap(res)
    assert ures is False or ures is True
    cnd: bool = ures

    if not cnd:
        return inst.arg
    return None


@register_opcode_handler("POP_JUMP_IF_TRUE", max_ver=(3, 10))
def _pop_jump_if_true_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> int | None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int

    tos = stack.pop()

    def impl():
        return bool(tos)

    # Acquires the condition, only calling bool() if tos requires conversion
    # NOTE Unconditionally calling bool() would cause an infinite recursion with the bool lookaside
    cnd: bool
    if tos is False or tos is True:
        cnd = tos
    else:
        tmp = _interpret_call_with_unwrapping(impl)
        if tmp is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return tmp
        else:
            assert tmp is False or tmp is True
            cnd = tmp

    if cnd:
        return inst.arg
    return None


# https://docs.python.org/3.10/library/dis.html#opcode-POP_TOP
@register_opcode_handler("POP_TOP")
def _pop_top_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    stack.pop_wrapped()


# Returns either
def do_raise(exc: Any = Py_NULL(), cause: Any = Py_NULL()) -> Literal[INTERPRETER_SIGNALS.EXCEPTION_RAISED]:
    # Get the type and exception being raised
    typ: Any = Py_NULL()
    value: Any = Py_NULL()
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    if exc is Py_NULL():
        # Re-raise
        assert runtimectx.exception_stack
        value = runtimectx.exception_stack[0]
        if value == None:
            return do_raise(RuntimeError("No active exception to reraise"))
        assert isinstance(value, BaseException)
        # check for cause being PY_NULL? Python does not do this, but it would seem to be a bug
        runtimectx.curexc = value
        return INTERPRETER_SIGNALS.EXCEPTION_RAISED

    if isinstance(exc, type) and issubclass(exc, BaseException):
        typ = exc
        value = unwrap(_interpret_call_with_unwrapping(exc))  # TODO: handle elsewhere?
        if value is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return value
        if not isinstance(value, BaseException):  # TODO: maybe drop: this is from CPython, but can it happen in Python?
            return do_raise(
                TypeError(f"calling {typ} should have returned an instance of BaseException, not {type(value)}")
            )

    elif isinstance(exc, BaseException):
        value = exc
        typ = type(exc)
    else:
        return do_raise(TypeError("exceptions must derive from BaseException"))

    # Attach the cause
    if cause is not Py_NULL():
        fixed_cause: BaseException | None = None
        if isinstance(cause, type) and issubclass(cause, BaseException):
            jret = _interpret_call_with_unwrapping(cause)
            assert isinstance(jret, BaseException)
            if jret is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                return jret
            fixed_cause = jret

            # NOTE: This was a bug in cpython, fixed by cpython PR #112216
            if not isinstance(fixed_cause, BaseException):
                return do_raise(
                    TypeError(
                        f"calling {cause} should have returned an instance of BaseException, not {type(fixed_cause)}"
                    )
                )
        elif isinstance(cause, BaseException):  # PyExceptionInstance_Check
            fixed_cause = cause
        elif cause is None:
            fixed_cause = None
        else:
            return do_raise(TypeError(f"exception causes must derive from BaseException"))

        value.__cause__ = fixed_cause

    runtimectx.curexc = value
    return INTERPRETER_SIGNALS.EXCEPTION_RAISED


# https://docs.python.org/3.11/library/dis.html#opcode-PRINT_EXPR
@register_opcode_handler("PRINT_EXPR")
def _print_expr_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    def impl(tos):
        # NOTE: There is no other way to obtain the display hook, other
        #       than writing a C extension, so we mangle.
        # NOTE: The display hook's return value is ignored by cpython.
        # NOTE: By default, type(sys.__displayhook__) is <class 'builtin_function_or_method'>.
        from sys import displayhook as __thunder_sys_displayhook

        __thunder_sys_displayhook(tos)
        return None

    tos = stack.pop()
    val = _interpret_call_with_unwrapping(impl, tos)
    if val is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return val
    return None


# https://docs.python.org/3.11/library/dis.html#opcode-PUSH_EXC_INFO
@register_opcode_handler("PUSH_EXC_INFO", min_ver=(3, 11))
def _push_exc_info_handler(inst: dis.Instruction, /, stack: InterpreterStack, exception_stack: list, **kwargs) -> None:
    assert exception_stack
    top = stack.pop()
    # CPython reads exc_info->exc_type/value/traceback, see RuntimeCtx inititalization of exception_stack for more info
    stack.append(exception_stack[-1])
    stack.append(top)
    assert isinstance(top, BaseException)
    exception_stack[-1] = top


# https://docs.python.org/3.11/library/dis.html#opcode-PUSH_NULL
@register_opcode_handler("PUSH_NULL", min_ver=(3, 11))
def _push_null_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    stack.append(wrap_const(Py_NULL()))


# https://docs.python.org/3.10/library/dis.html#opcode-RAISE_VARARGS
@register_opcode_handler("RAISE_VARARGS")
def _raise_varargs_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, try_stack: list[PyTryBlock], **kwargs
) -> None | INTERPRETER_SIGNALS:
    cause: Any = Py_NULL()
    exc: Any = Py_NULL()
    assert type(inst.arg) is int
    if inst.arg == 2:
        cause = stack.pop()
        exc = stack.pop()
    elif inst.arg == 1:
        exc = stack.pop()
    else:
        assert inst.arg == 0
    return do_raise(exc, cause)


# https://docs.python.org/3.10/library/dis.html#opcode-RERAISE
@register_opcode_handler("RERAISE", max_ver=(3, 10))
def _reraise_handler_3_10(
    inst: dis.Instruction, /, stack: InterpreterStack, try_stack: list[PyTryBlock], frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert try_stack
    assert type(inst.arg) is int

    if inst.arg != 0:
        frame.lasti = try_stack[-1].handler

    exc = stack.pop()
    val = stack.pop()
    tb = stack.pop()
    assert isinstance(val, BaseException)
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    val.__traceback__ = tb
    runtimectx.curexc = val
    return INTERPRETER_SIGNALS.EXCEPTION_RAISED


# https://docs.python.org/3.11/library/dis.html#opcode-RERAISE
@register_opcode_handler("RERAISE", min_ver=(3, 11))
def _reraise_handler_3_11(
    inst: dis.Instruction, /, stack: InterpreterStack, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int

    val = stack.pop()
    if inst.arg != 0:
        # Note: The documentation is wrong here, this is from the ceval.c
        lasti = stack[-inst.arg]
        assert isinstance(lasti, int)
        frame.lasti = lasti

    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
    assert isinstance(val, BaseException)
    runtimectx.curexc = val
    return INTERPRETER_SIGNALS.EXCEPTION_RAISED


# https://docs.python.org/3.10/library/dis.html#opcode-RETURN_VALUE
@register_opcode_handler("RETURN_VALUE")
def _return_value_handler(inst: dis.Instruction, /, **kwargs) -> int | None | INTERPRETER_SIGNALS:
    return INTERPRETER_SIGNALS.RETURN_VALUE


# https://docs.python.org/3.10/library/dis.html#opcode-ROT_N
@register_opcode_handler("ROT_N", max_ver=(3, 10))
def _rot_n_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int
    assert len(stack) >= inst.arg
    # stack[-inst.arg :] = (stack[-1], *stack[-inst.arg : -1])
    rhs = (stack[-1], *stack[-inst.arg : -1])
    for i in range(-inst.arg, 0):
        stack[i] = rhs[i]


# https://docs.python.org/3.10/library/dis.html#opcode-ROT_THREE
@register_opcode_handler("ROT_THREE", max_ver=(3, 10))
def _rot_three_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    top = stack[-1]
    second = stack[-2]
    third = stack[-3]

    stack[-1] = second
    stack[-2] = third
    stack[-3] = top


# https://docs.python.org/3.10/library/dis.html#opcode-ROT_TWO
@register_opcode_handler("ROT_TWO", max_ver=(3, 10))
def _rot_two_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    top = stack[-1]
    second = stack[-2]

    stack[-1] = second
    stack[-2] = top


# https://docs.python.org/3.10/library/dis.html#opcode-ROT_FOUR
@register_opcode_handler("ROT_FOUR", max_ver=(3, 10))
def _rot_four_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    # stack[-4:] = (stack[-1], *stack[-4:-1])
    top = stack[-1]
    second = stack[-2]
    third = stack[-3]
    fourth = stack[-4]

    stack[-1] = second
    stack[-2] = third
    stack[-3] = fourth
    stack[-4] = top


# https://docs.python.org/3.10/library/dis.html#opcode-SET_ADD
@register_opcode_handler("SET_ADD")
def _set_add_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg

    # NOTE Doesn't pop the set that's extended
    tos = stack.pop()
    s: set = stack[-i]

    assert type(s) is set, type(s)
    s.add(tos)


# https://docs.python.org/3.10/library/dis.html#opcode-SET_UPDATE
@register_opcode_handler("SET_UPDATE")
def _set_update_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg

    # NOTE Doesn't pop the set that is updated
    tos = stack.pop()
    s: set = stack[-i]

    # NOTE tos does not have to be a set
    assert isinstance(s, set)
    s.update(tos)


# https://docs.python.org/3.10/library/dis.html#opcode-SETUP_ASYNC_WITH
@register_opcode_handler("SETUP_ASYNC_WITH", max_ver=(3, 10))
def _setup_async_with_handler(
    inst: dis.Instruction, *, inst_ptr: int, stack: InterpreterStack, try_stack: list[PyTryBlock], **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    instr_offset = inst_ptr + inst.arg + 1

    aenter_res = stack.pop()
    try_stack.append(PyTryBlock(PyTryBlock.SETUP_FINALLY_TYPE, instr_offset, len(stack)))
    stack.append(aenter_res)


# https://docs.python.org/3.10/library/dis.html#opcode-SETUP_FINALLY
@register_opcode_handler("SETUP_FINALLY", max_ver=(3, 10))
def _setup_finally_handler(
    inst: dis.Instruction, *, inst_ptr: int, stack: InterpreterStack, try_stack: list[PyTryBlock], **kwargs
) -> None:
    assert inst.arg is not None
    instr_offset = inst_ptr + inst.arg + 1
    try_stack.append(PyTryBlock(PyTryBlock.SETUP_FINALLY_TYPE, instr_offset, len(stack)))


# https://docs.python.org/3.10/library/dis.html#opcode-SETUP_WITH
@register_opcode_handler("SETUP_WITH", max_ver=(3, 10))
def _setup_with_handler(
    inst: dis.Instruction, *, inst_ptr: int, stack: InterpreterStack, try_stack: list[PyTryBlock], **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    instr_offset = inst_ptr + inst.arg + 1

    mgr = stack.pop()

    # python does a "special lookup"
    enter_method = _interpret_call_with_unwrapping(getattr, mgr, "__enter__")
    if enter_method is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return enter_method
    exit_method = _interpret_call_with_unwrapping(getattr, mgr, "__exit__")
    if exit_method is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return exit_method

    assert callable(enter_method)
    assert callable(exit_method)

    stack.append(exit_method)

    res = _interpret_call_with_unwrapping(enter_method)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res

    try_stack.append(PyTryBlock(PyTryBlock.SETUP_FINALLY_TYPE, instr_offset, len(stack)))

    stack.append(res)


# https://docs.python.org/3.11/library/dis.html#opcode-SWAP
@register_opcode_handler("SWAP", min_ver=(3, 11))
def _swap_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    i = inst.arg
    assert isinstance(i, int)
    assert 0 < i <= len(stack)

    top = stack[-1]
    other = stack[-i]

    stack[-1] = other
    stack[-i] = top


# https://docs.python.org/3.10/library/dis.html#opcode-STORE_ATTR
@register_opcode_handler("STORE_ATTR")
def _store_attr_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    namei: int = inst.arg

    name: str = wrap_const(co.co_names[namei])

    tos: Any = stack.pop_wrapped()
    tos1: Any = stack.pop_wrapped()

    def impl(tos, name, tos1):
        setattr(tos, name, tos1)

    res = _interpret_call(impl, tos, name, tos1)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res


# https://docs.python.org/3.10/library/dis.html#opcode-STORE_DEREF
@register_opcode_handler("STORE_DEREF")
def _store_deref_handler(
    inst: dis.Instruction,
    /,
    stack: InterpreterStack,
    co: CodeType,
    frame: InterpreterFrame,
    **kwargs,
) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg
    if sys.version_info < (3, 11):
        i += co.co_nlocals

    assert i >= 0 and i < len(frame.localsplus)

    tos = stack.pop_wrapped()

    tos = store_deref_callback(tos, inst.argval, co.co_cellvars, co.co_freevars)

    if tos is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return tos

    # TODO: we would like to do this, but the extensions currently block all of setattr

    # def impl(cell, v):
    #    cell.cell_contents = v
    # res = _interpret_call(impl, frame.localsplus[i], tos)
    # assert res is not INTERPRETER_SIGNALS.EXCEPTION_RAISED

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        frame.localsplus[i].value.cell_contents = tos.value
        populate_attribute_wrapper(frame.localsplus[i], "cell_contents", tos)
    else:
        frame.localsplus[i].cell_contents = tos


# https://docs.python.org/3.10/library/dis.html#opcode-STORE_GLOBAL
@register_opcode_handler("STORE_GLOBAL")
def _store_global_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    namei: int = inst.arg

    name: str = co.co_names[namei]
    tos = stack.pop_wrapped()

    # Our behavior is a bit different from CPython here.
    # Namely, we error with KeyError when `name` is not present in the globals dict.
    # CPython, on the other hand, when a callable is created with types.FunctionType
    # ignores exceptions. This might create a callable which runs successfully even
    # with the missing `name`. See the example courtesy of @t-vi below.
    #
    # def fn():
    #   global t
    #   t = "I am t"
    #
    # class D(dict):
    #   def __setitem__(self, x, v):
    #       raise RuntimeError("Nobody expects...")
    #
    # >>> import types
    # >>> new_fn = types.FunctionType(fn.__code__, d)
    # >>> new_fn()
    # >>> t
    # NameError: name 't' is not defined
    tos = global_callback(tos, name, INTERPRETER_CALLBACKS.STORE_GLOBAL_CALLBACK)
    if tos is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return tos

    res = _interpret_call(
        lambda frame_globals, name, value: frame_globals.__setitem__(name, value), frame.globals, wrap_const(name), tos
    )
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return res


# https://docs.python.org/3.10/library/dis.html#opcode-STORE_FAST
@register_opcode_handler("STORE_FAST")
def _store_fast_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None:
    tos = stack.pop_wrapped()
    assert type(inst.arg) is int
    var_num: int = inst.arg

    name: str = co.co_varnames[var_num]
    frame.localsplus[var_num] = tos


# https://docs.python.org/3.10/library/dis.html#opcode-STORE_NAME
@register_opcode_handler("STORE_NAME")
def _store_name_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, co: CodeType, frame: InterpreterFrame, **kwargs
) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    namei: int = inst.arg
    name: str = co.co_names[namei]

    tos: Any = stack.pop_wrapped()

    def impl(names_dict, name, value):
        names_dict[name] = value

    return check_signal(_interpret_call(impl, frame.names, wrap_const(name), tos))


# https://docs.python.org/3.10/library/dis.html#opcode-STORE_SUBSCR
@register_opcode_handler("STORE_SUBSCR")
def _store_subscr_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop_wrapped()
    tos1 = stack.pop_wrapped()
    tos2 = stack.pop_wrapped()

    def impl(tos, tos1, tos2):
        return tos1.__setitem__(tos, tos2)

    return _interpret_call_with_unwrapping(impl, tos, tos1, tos2)


# https://docs.python.org/3.10/library/dis.html#opcode-UNARY_INVERT
@register_opcode_handler("UNARY_INVERT")
def _unary_invert_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop()

    def impl():
        if hasattr(tos, "__invert__"):
            result = tos.__invert__()
            if result is not NotImplemented:
                return result

        raise TypeError(f"bad operand type for unary ~: '{type(tos).__name__}'")

    return check_and_append(stack, _interpret_call_with_unwrapping(impl))


# https://docs.python.org/3.10/library/dis.html#opcode-UNARY_NOT
@register_opcode_handler("UNARY_NOT")
def _unary_not_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop()

    def impl():
        if bool(tos):
            return False
        return True

    return check_and_append(stack, _interpret_call_with_unwrapping(impl))


# https://docs.python.org/3.10/library/dis.html#opcode-UNARY_NEGATIVE
@register_opcode_handler("UNARY_NEGATIVE")
def _unary_negative_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop()

    def impl():
        if hasattr(tos, "__neg__"):
            result = tos.__neg__()
            if result is not NotImplemented:
                return result

        raise TypeError(f"bad operand type for unary -: '{type(tos).__name__}'")

    return check_and_append(stack, _interpret_call_with_unwrapping(impl))


# https://docs.python.org/3.10/library/dis.html#opcode-UNARY_POSITIVE
@register_opcode_handler("UNARY_POSITIVE")
def _unary_positive_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    tos = stack.pop()

    def impl():
        if hasattr(tos, "__pos__"):
            result = tos.__pos__()
            if result is not NotImplemented:
                return result

        raise TypeError(f"bad operand type for unary +: '{type(tos).__name__}'")

    return check_and_append(stack, _interpret_call_with_unwrapping(impl))


# https://docs.python.org/3.10/library/dis.html#opcode-UNPACK_EX
@register_opcode_handler("UNPACK_EX")
def _unpack_ex_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    assert type(inst.arg) is int
    counts: int = inst.arg
    before_list: WrappedValue = wrap_const(counts & 0xFF)
    after_list: WrappedValue = wrap_const(counts >> 8)

    seq: Iterable = stack.pop_wrapped()
    assert wrapped_isinstance(seq, Iterable)

    def impl(seq, before_list, after_list):
        results: list = []
        it: Iterator = iter(seq)

        for _ in range(before_list):
            results.append(next(it))

        list_result: list = list(it)
        results.append(list_result)

        if after_list > 0:
            for x in list_result[-after_list:]:
                results.append(x)

            del list_result[-after_list:]

        return results

    results = _interpret_call(impl, seq, before_list, after_list)
    if results is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        return results

    ctx: InterpreterCompileCtx = get_interpretercompilectx()
    if ctx._with_provenance_tracking:
        populate_item_wrappers(results)
        assert type(results) is WrappedValue
        assert results.item_wrappers is not None
        results = results.item_wrappers[:]

    assert type(results) is list

    for x in reversed(results):
        stack.append(x)


# https://docs.python.org/3.10/library/dis.html#opcode-UNPACK_SEQUENCE
@register_opcode_handler("UNPACK_SEQUENCE")
def _unpack_sequence_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    # contrary to the opname, seq is an Iterable, not necessarily a sequence
    seq: Iterable = stack.pop_wrapped()

    assert type(inst.arg) is int
    count: int = inst.arg

    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

    seq_iter = _interpret_call(iter, seq)

    values = []
    for _ in range(count):
        v = _interpret_call(next, seq_iter)
        if v is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            if isinstance(runtimectx._curexc, StopIteration):
                return do_raise(ValueError(f"not enough values to unpack (expected {count}, got {len(values)})"))
            else:
                return v

        values.append(v)

    v = _interpret_call(next, seq_iter)
    if v is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        if isinstance(runtimectx._curexc, StopIteration):
            runtimectx._curexc = None
        else:
            return v
    else:
        return do_raise(ValueError(f"too many values to unpack (expected {count})"))

    for x in values[::-1]:
        stack.append(x)


# Generator handling
# https://docs.python.org/3.10/library/dis.html#opcode-GEN_START
@register_opcode_handler("GEN_START", max_ver=(3, 10))
def _gen_start_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None:
    assert type(inst.arg) is int
    assert 0 <= inst.arg < 3  # yeah, we are not doing anything with it
    stack.pop_wrapped()  # this should be None (sent to the generator), but Python does not check


# https://docs.python.org/3.10/library/dis.html#opcode-GET_YIELD_FROM_ITER
@register_opcode_handler("GET_YIELD_FROM_ITER")
def _get_yield_from_iter_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, **kwargs
) -> None | INTERPRETER_SIGNALS:
    tos = stack.getitem_wrapped(-1)
    utos = unwrap(tos)
    if not (inspect.isgenerator(utos) or inspect.iscoroutine(utos)):
        return check_and_append(stack, _interpret_call(iter, stack.pop_wrapped()))


# https://docs.python.org/3.11/library/dis.html#opcode-RETURN_GENERATOR
@register_opcode_handler("RETURN_GENERATOR", min_ver=(3, 11))
def _return_generator_handler(inst: dis.Instruction, /, **kwargs) -> None | INTERPRETER_SIGNALS:
    return INTERPRETER_SIGNALS.RETURN_GENERATOR


# https://docs.python.org/3.11/library/dis.html#opcode-SEND
@register_opcode_handler("SEND", min_ver=(3, 11))
def _send_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, inst_ptr: int, **kwargs
) -> None | int | INTERPRETER_SIGNALS:
    # SEND(delta)
    # Equivalent to STACK[-1] = STACK[-2].send(STACK[-1]). Used in yield from and await statements.
    # If the call raises StopIteration, pop the top value from the stack, push the exception's value attribute, and increment the bytecode counter by delta.
    assert isinstance(inst.arg, int)
    send_value = stack.pop()
    generator = stack[-1]

    if send_value is None and hasattr(generator, "__next__"):
        # iterators don't have a .send method
        def impl():
            return generator.__next__()

    else:

        def impl():
            return generator.send(send_value)

    res = _interpret_call_with_unwrapping(impl)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
        if isinstance(runtimectx.curexc, StopIteration):
            stack.pop()  # remove generator
            stack.append(runtimectx.curexc.value)
            runtimectx.curexc = None
            return inst_ptr + inst.arg + 1
        else:
            return res  # propagate exception

    stack.append(res)


# https://docs.python.org/3.10/library/dis.html#opcode-WITH_EXCEPT_START
@register_opcode_handler("WITH_EXCEPT_START", max_ver=(3, 10))
def _with_except_start_handler_3_10(
    inst: dis.Instruction, *, inst_ptr: int, stack: InterpreterStack, try_stack: list[PyTryBlock], **kwargs
) -> None | INTERPRETER_SIGNALS:
    exc = stack[-1]
    val = stack[-2]
    tb = stack[-3]
    assert exc is not None
    assert not isinstance(exc, int)  # funny but from Python
    exit_func = stack[-7]
    return check_and_append(stack, _interpret_call_with_unwrapping(exit_func, exc, val, tb))


# https://docs.python.org/3.11/library/dis.html#opcode-WITH_EXCEPT_START
@register_opcode_handler("WITH_EXCEPT_START", min_ver=(3, 11))
def _with_except_start_handler_3_11(
    inst: dis.Instruction, *, inst_ptr: int, stack: InterpreterStack, try_stack: list[PyTryBlock], **kwargs
) -> None | INTERPRETER_SIGNALS:
    # in 3.11 the exception representation changed to only val
    val = stack[-1]
    exc = type(val)
    tb = val.__traceback__

    assert isinstance(stack[-3], int)
    exit_func = stack[-4]
    return check_and_append(stack, _interpret_call_with_unwrapping(exit_func, exc, val, tb))


# https://docs.python.org/3.10/library/dis.html#opcode-YIELD_FROM
@register_opcode_handler("YIELD_FROM", max_ver=(3, 10))
def _yield_from_handler(
    inst: dis.Instruction, /, stack: InterpreterStack, frame: InterpreterFrame, inst_ptr: int, **kwargs
) -> None | INTERPRETER_SIGNALS:
    send_value = stack.pop()
    generator = stack[-1]

    if send_value is None and hasattr(generator, "__next__"):
        # iterators don't have a .send method
        def impl():
            return generator.__next__()

    else:

        def impl():
            return generator.send(send_value)

    res = _interpret_call_with_unwrapping(impl)
    if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()
        if isinstance(runtimectx.curexc, StopIteration):
            stack.pop()  # remove generator
            stack.append(runtimectx.curexc.value)
            runtimectx.curexc = None
            return None
        else:
            return res  # propagate exception

    # this is a gross hack, this will be incremented so the inst_ptr is at this YIELD_FROM again
    # cleaner would be to introduce another INTERPRETER_SIGNALS
    frame.inst_ptr -= 1
    # this will be yielded
    stack.append(res)
    return INTERPRETER_SIGNALS.YIELD_VALUE


# https://docs.python.org/3.10/library/dis.html#opcode-YIELD_VALUE
@register_opcode_handler("YIELD_VALUE")
def _yield_value_handler(inst: dis.Instruction, /, stack: InterpreterStack, **kwargs) -> None | INTERPRETER_SIGNALS:
    # note that the popping from the stack is done in the _run_frame loop
    return INTERPRETER_SIGNALS.YIELD_VALUE


# In order to support "colored functions" that suspend and resume execution
# (generator, async generator, coroutine), we define generic equivalents here
# that take the interpreter frame and execute until the next yield point.
# The way these functions work in Python is that objects are created and
# retruned either on invocation (Python <=3.10) or by the RETURN_GENERATOR
# opcode (Python >= 3.11).
def make_generator(
    frame: InterpreterFrame,
    compilectx: InterpreterCompileCtx,
    runtimectx: InterpreterRuntimeCtx,
):
    # TODO: detect whether the send is from the interpreter? if so, how? Currently the tracking will miss the connection between send and here. :(
    def thunder_interpreter_generator():
        send_value: Any = None  # the value gotten from from <generator>.send
        while True:  # or maybe have return?
            with interpreter_ctx(compilectx, runtimectx):
                try:
                    res, status = _run_frame(frame, compilectx, runtimectx, send_value=send_value)
                except Exception as e:
                    msg = f"Encountered exception {type(e).__name__}: {e}"
                    raise InterpreterError(msg) from e
                if status is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                    e = runtimectx.curexc
                    assert isinstance(e, BaseException)
                    runtimectx.curexc = None
                    if isinstance(e, StopIteration):
                        return unwrap(e.value)
                    raise e
            if status == INTERPRETER_SIGNALS.RETURN_VALUE:
                return  # TODO: should this return res?
            assert status == INTERPRETER_SIGNALS.YIELD_VALUE
            send_value = yield unwrap(res)

    return wrap_const(thunder_interpreter_generator())


# factory to create an async generator for a given interpreter frame, see comment for make_generator
def make_async_generator(
    frame: InterpreterFrame,
    compilectx: InterpreterCompileCtx,
    runtimectx: InterpreterRuntimeCtx,
):
    async def thunder_interpreter_async_generator():
        send_value: Any = None  # the value gotten from from <generator>.send
        while True:  # or maybe have return?
            with interpreter_ctx(compilectx, runtimectx):
                try:
                    res, status = _run_frame(frame, compilectx, runtimectx, send_value=send_value)
                except Exception as e:
                    msg = f"Encountered exception {type(e).__name__}: {e}"
                    raise InterpreterError(msg) from e
                if status is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                    e = runtimectx.curexc
                    assert isinstance(e, BaseException)
                    runtimectx.curexc = None
                    if isinstance(e, StopIteration):
                        return
                    raise e
            if status == INTERPRETER_SIGNALS.RETURN_VALUE:
                return  # TODO: should this return res?
            assert status == INTERPRETER_SIGNALS.YIELD_VALUE
            send_value = yield unwrap(res)

    return wrap_const(thunder_interpreter_async_generator())  # TODO: better than wrap_const


# factory to create a coroutine for a given interpreter frame, see comment for make_generator
def make_coroutine(
    frame: InterpreterFrame,
    compilectx: InterpreterCompileCtx,
    runtimectx: InterpreterRuntimeCtx,
):
    async def thunder_interpreter_coroutine():
        send_value: Any = None  # the value gotten from from <generator>.send
        while True:  # or maybe have return?
            with interpreter_ctx(compilectx, runtimectx):
                try:
                    res, status = _run_frame(frame, compilectx, runtimectx, send_value=send_value)
                except Exception as e:
                    msg = f"Encountered exception {type(e).__name__}: {e}"
                    raise InterpreterError(msg) from e
                if status is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                    e = runtimectx.curexc
                    assert isinstance(e, BaseException)
                    runtimectx.curexc = None
                    if isinstance(e, StopIteration):
                        return unwrap(e.value)
                    raise e
            if status == INTERPRETER_SIGNALS.RETURN_VALUE:
                return unwrap(res)
            assert status == INTERPRETER_SIGNALS.YIELD_VALUE
            raise NotImplementedError("not implemented")

    return wrap_const(thunder_interpreter_coroutine())


def _interpret_call_with_unwrapping(fn: Callable, /, *args, **kwargs) -> Any | INTERPRETER_SIGNALS:
    args = wrap_consts(*args)
    kwargs = {k: wrap_const(v) for k, v in kwargs.items()}
    res = _interpret_call(fn, *args, **kwargs)
    if isinstance(res, INTERPRETER_SIGNALS):
        return res

    compilectx: InterpreterCompileCtx = get_interpretercompilectx()
    if compilectx._with_provenance_tracking:
        assert all(isinstance(a, WrappedValue) for a in args)
        assert all(isinstance(a, WrappedValue) for a in kwargs.values())
        if isinstance(res.value, list):
            assert len(res.value) == len(
                res.item_wrappers
            ), f"{len(res.value)} {len(res.item_wrappers)} {res.value} {res.item_wrappers} {fn}"
        if isinstance(res.value, dict):
            assert len(res.key_wrappers) == len(
                res.item_wrappers
            ), f"{len(res.value)} {len(res.item_wrappers)} {len(res.key_wrappers)} {res.value} {res.item_wrappers} {fn}"
        for a in args:
            if isinstance(a.value, list):
                assert isinstance(a.item_wrappers, Sized)
                assert len(a.value) == len(
                    a.item_wrappers
                ), f"{len(a.value)} {len(a.item_wrappers)} {a.value} {a.item_wrappers} {fn}"
            if isinstance(a.value, dict):
                assert isinstance(a.item_wrappers, Sized)
                assert len(a.key_wrappers) == len(
                    a.item_wrappers
                ), f"{len(a.value)} {len(a.item_wrappers)} {len(a.key_wrappers)} {a.value} {a.item_wrappers} {fn}"

    return unwrap(res)


def _interpret_call(fn: Callable | WrappedValue, /, *args, **kwargs) -> Any | INTERPRETER_SIGNALS:
    compilectx: InterpreterCompileCtx = get_interpretercompilectx()
    runtimectx: InterpreterRuntimeCtx = get_interpreterruntimectx()

    # TODO: Implement generics and fix WrappedValue[T] everywhere.
    runtimectx.record_interpreter_call(fn)  # type: ignore
    rval = _call_dispatch(compilectx, runtimectx, fn, *args, **kwargs)  # type: ignore
    if compilectx._with_provenance_tracking:
        assert isinstance(rval, (INTERPRETER_SIGNALS, WrappedValue)), f"return {rval} unexpected calling {unwrap(fn)}"
    runtimectx.record_interpreter_return(fn, rval)  # type: ignore

    return rval


# Interprets the callable with the given args and kwargs
# NOTE There are (currently) 8 cases for interpretation:
#
#   (0) For already interpret()-ed functions, we use the original function
#       to trace through rather than trying to have the interpreter trace through
#       itself (which neither does not work nor would it be useful).
#   (1) Bound methods are unbound
#       (1a) The callable is a bound method (implemented in Python), in which case it's canonicalized
#       (1b) The callable is a builtin method, in which case we try to unbind it
#   (2) The callable has a lookaside, in which case it's used to execute the operation
#   (3) The callable is a partial object, in which case it's recursively unwrapped
#           Note that this case is after (1), which allows for lookasides on partial objects
#   (4) The callable is opaque, in which case it's executed by the CPython interpreter and its
#           result returned
#   (5) The callable is a type object, in which case it is instantiated with __new__ and initialized with __init__
#   (6) The callable is a callable object, in which case its __call__ attribute is called recursively
#   (7) The callable is a FunctionType, in which case it's recursively interpretered by the interpreter
#
# TODO Consider refactoring this into one function for each case
def _call_dispatch(
    compilectx: InterpreterCompileCtx, runtimectx: InterpreterRuntimeCtx, fn: Callable, /, *args, **kwargs
) -> Any | INTERPRETER_SIGNALS:
    if isinstance(fn, WrappedValue):
        wrapped_fn = fn
    else:
        # TODO: think about whether this is a good choice in all circumstances
        wrapped_fn = wrap_const(fn)
    fn = unwrap(fn)
    assert not isinstance(fn, WrappedValue)

    if compilectx._with_provenance_tracking:
        assert all(isinstance(a, WrappedValue) for a in args)
        assert all(isinstance(a, WrappedValue) for a in kwargs.values())
        for a in args:
            if isinstance(a.value, list):
                assert len(a.value) == len(
                    a.item_wrappers
                ), f"{len(a.value)} {len(a.item_wrappers)} {a.value} {a.item_wrappers}"

    # (1) Already (interpreter wrapped)
    if hasattr(fn, "__thunder_interpreter_orig_fn"):
        fn = fn.__thunder_interpreter_orig_fn
        assert isinstance(fn, Callable)
        wrapped_fn = wrap_const(fn)
    # (0) Bound methods are unbound
    # (1a) The callable is a bound method (implemented in Python), in which case it's unwrapped
    if inspect.ismethod(fn):

        def _impl(fn, *args, **kwargs):
            return fn.__func__(fn.__self__, *args, **kwargs)

        return _interpret_call(_impl, wrapped_fn, *args, **kwargs)  # type: ignore

    # (1b) The callable is a builtin method, in which case it's canonicalized
    #   A builtin is canonicalized when it's a call on a type or a module (or their equivalent)
    #   Ex. The append method of a list
    #       l = [0, 1, 2]
    #       type(l.append)  # builtin_function_or_method
    #       isinstance(l.append, BuiltinMethodType)  # True
    #       l.append.__self__  # [0, 1, 2]
    #
    #   However, l.append is a method specific to the list l, and we'd like to canonicalize
    #   it to list.append.
    #   We do this by acquiring the function's name from the __self__ object's type. In this case
    #   getattr(type(l), "append").
    #
    #   Ex. The __new__ method of a type
    #       type(list.__new__)  # builtin_function_or_method
    #       isinstance(list.__new__, BuiltinMethodType)  # True
    #       list.__new__.__self__  # list
    #       type(list).__new__  # <function type.__new__(*args, **kwargs)>
    #
    #   We don't want to unwrap calls on types, because these calls are already
    #   canonicalized.
    #
    #   Ex. A method on a module
    #       import builtins
    #       type(builtins.hasattr) # builtin_function_or_method
    #
    #   Like with types in the above example, we don't want to unwrap method calls
    #   on modules, because they are already canonicalized.
    #
    #   Ex. Builtin methods with __self__ = None
    #       import torch
    #       torch.relu.__self__  # None
    #
    #   Builtin methods with __self__ = None cannot be further canonicalized
    #
    #   Ex. Builtin methods on PyCapsules (see https://docs.python.org/3/c-api/capsule.html)
    #       import torch
    #       torch._C._are_functorch_transforms_active.__self__  # <capsule object NULL at 0x7bcf01e69f20>
    #       type(torch._C._are_functorch_transforms_active.__self__)  # PyCapsule
    #
    #   Like with types and modules, the base PyCapsule class does not have the methods we're looking for.
    #
    #   Ex.
    # NOTE Builtin Methods
    #   Builtin methods are not considered methods by inspect.ismethod
    if isinstance(fn, (BuiltinMethodType, MethodWrapperType)):
        assert is_opaque(fn)
        slf = fn.__self__

        if slf is not None and not isinstance(slf, (type, ModuleType)) and not is_pycapsule(slf):
            # NOTE: we need to walk the mro because we need to deal with super().foo
            #       using the qualname is not good here because Python qualnames come with no
            #       guarantees (e.g. random._random.Random.seed and random.Random.seed are distinct
            #       but have the same qualname (Random.seed).
            #       Binding (using <unbound_method>.__get__) is what super does to get a bound method-
            #       While it will be a new object every time __get__ is called (so no `is`), equality
            #       works.
            #       The next trouble is that types that are not subclassable might not implement __get__
            #       for their methods (e.g. re.Pattern.match). But those can only appear as the 0th item in mro.
            #       Also, if there is no unbound method defined on the type, it might be a bound method on the
            #       type itself (e.g. object.__or__, which will be checked e.g. from int.__or__ when traversing
            #       the mro).
            unbound_fn = None
            for t in type(slf).mro()[1:]:
                unbound_fn_candidate = getattr(t, fn.__name__, None)
                if (
                    unbound_fn_candidate is not None
                    and isinstance(unbound_fn_candidate, (WrapperDescriptorType, MethodDescriptorType))
                    and unbound_fn_candidate.__get__(slf) == fn
                ):
                    unbound_fn = unbound_fn_candidate
                    break

            # this gets the method from the "top" type
            if unbound_fn is None:
                unbound_fn = getattr(type(slf), fn.__name__, None)

            # TODO: The above is our best attempt to get the unbound function.
            #       If it fails for whatever reason (and unbound_fn is None here),
            #       we have two options:
            #       - (currently done) fall through and treat this as an opaque
            #         function call (maybe we could warn)
            #       - refuse to work and raise an exception
            #       Given that the main reason to do this unwrapping is to be able
            #       to define lookasides on methods as <class>.method and have
            #       them work when called on objects, the harm done by the first
            #       approach seems limited (as long as people test their lookasides).
            if unbound_fn is not None:
                assert not isinstance(unbound_fn, BuiltinFunctionType)
                slf = _interpret_call(getattr, wrapped_fn, wrap_const("__self__"))
                unbound_fn = wrap_const(unbound_fn)  # TODO!
                return _interpret_call(unbound_fn, slf, *args, **kwargs)

    # (2) Handles lookasides
    lookaside_fn: INTERPRETER_SIGNALS | None | Callable = compilectx.lookaside(fn, *args, **kwargs)
    if lookaside_fn is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
        # Happens with sharp edges, for example
        return lookaside_fn
    if lookaside_fn:
        runtimectx.record_lookaside(lookaside_fn)
        res = lookaside_fn(*args, **kwargs)
        return res

    # TODO: disabled as partial is just like any other class
    # (3) Handles partial objects
    if isinstance(fn, functools.partial):

        def partial_call_impl(partial_function, /, *args, **kwargs):
            return partial_function.func(*(partial_function.args + args), **(partial_function.keywords | kwargs))

        return _interpret_call(partial_call_impl, wrapped_fn, *args, **kwargs)

    if isinstance(fn, functools.partialmethod):
        raise NotImplementedError(
            "functools.partialmethod objects like {fn} are not currently supported, please file an issue requesting support"
        )

    # (4) Handles opaque functions
    if is_opaque(fn):
        runtimectx.record_opaque_call(fn)
        args_ = [unwrap(a) for a in args]
        kwargs_ = {unwrap(k): unwrap(v) for k, v in kwargs.items()}
        try:
            opaque_result: Any = fn(*args_, **kwargs_)
        except Exception as e:
            runtimectx.curexc = e
            return INTERPRETER_SIGNALS.EXCEPTION_RAISED

        if compilectx._with_provenance_tracking:
            pr = ProvenanceRecord(
                inst=PseudoInst.OPAQUE,
                inputs=[wrapped_fn.provenance, wrap_args(args).provenance, wrap_kwargs(kwargs).provenance],
            )
            opaque_result = wrap(opaque_result, provenance=pr)
        return opaque_result

    # (5) Handle types
    if isinstance(fn, type):
        if not hasattr(fn, "__new__"):
            raise NotImplementedError(
                f"Don't know how to interpret a callable with type {type(fn)} without a __new__ method"
            )
        obj = _interpret_call(fn.__new__, wrapped_fn, *args, **kwargs)
        if obj is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return obj

        wrapped_init = _interpret_call(getattr, obj, wrap_const("__init__"))
        assert not isinstance(wrapped_init, INTERPRETER_SIGNALS)
        populate_attribute_wrapper(wrapped_init, "__self__", obj)
        res = _interpret_call(wrapped_init, *args, **kwargs)
        if res is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
            return res
        return obj

    # (6) Handles callable objects (with a dunder call method)
    if not isinstance(fn, (FunctionType, MethodType)):
        if not hasattr(fn, "__call__"):
            raise NotImplementedError(
                f"Don't know how to interpret a callable with type {type(fn)} without a __call__ method"
            )

        wrapped_call = _interpret_call(getattr, wrapped_fn, wrap_const("__call__"))
        assert not isinstance(wrapped_call, INTERPRETER_SIGNALS)
        populate_attribute_wrapper(wrapped_call, "__self__", wrapped_fn)
        return _interpret_call(wrapped_call, *args, **kwargs)

    # (7) interprets into the function
    assert isinstance(fn, FunctionType), f"{fn=} had an unexpected type ({type(fn)}"
    return _setup_frame_and_run_python_function(compilectx, runtimectx, wrapped_fn, *args, **kwargs)


def _setup_frame_and_run_python_function(
    compilectx: InterpreterCompileCtx, runtimectx: InterpreterRuntimeCtx, wrapped_fn, /, *args, **kwargs
):
    fn = unwrap(wrapped_fn)

    # adjustments for "hidden" instructions (EXTENDED_ARGS, CACHE, ...)
    # TODO: use the code object as the authorative source
    sig = inspect.signature(fn, follow_wrapped=False)
    params = []
    for p in sig.parameters.values():
        if p.default is not inspect._empty:
            p = p.replace(default=wrap_const(p.default))
        params.append(p)
    sig = sig.replace(parameters=params)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    locals_dict: dict[str, Any] = dict(bound.arguments)

    if compilectx._with_provenance_tracking:
        variable_args = [n for n, p in sig.parameters.items() if p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD]
        for name in variable_args:
            val = locals_dict[name]
            if isinstance(val, tuple):
                val = wrap_args(val)
            else:
                assert isinstance(val, dict)
                val = wrap_kwargs(val)
            locals_dict[name] = val

    code: CodeType = extract_code(fn)

    # Comprehensions:
    #   CPython prefixes protected variables with a period. (Since it's not
    #   legal syntax, so they cannot collide with user variable names.) However,
    #   `inspect` helpfully renames them. We need to undo this to properly reflect
    #   what is in the actual bytecode.
    for name in code.co_varnames:
        if name.startswith("."):
            locals_dict[name] = locals_dict.pop(f"implicit{name[1:]}")

    # in Python 3.10: local vars is (var_names, co_cellvars, co_freevars), also the cellvars/freevars are set up on call
    # in Python 3.11+, these are not separated and cells will be set up by the MAKE_CELL instruction
    # in Thunder, we adopt the Python 3.11 way of allocating a single array for all localplus vars,
    #             but for 3.10 we do need to do the 3.10-style setting-up-at-entry here.
    localsplus: list[Any] = []
    if code.co_freevars:
        assert fn.__closure__
        assert len(code.co_freevars) == len(fn.__closure__)
        closure = _interpret_call(getattr, wrapped_fn, wrap_const("__closure__"))
        # we need to use __getiem__ directly to avoid infinite recursion
        closure = [
            _interpret_call(lambda x, i: x.__getitem__(i), closure, wrap_const(i)) for i in range(len(fn.__closure__))
        ]
    else:
        closure = []
        assert not fn.__closure__

    if (3, 10) <= sys.version_info < (3, 11):
        assert len(code.co_varnames) == code.co_nlocals
        for n in code.co_varnames:
            local = locals_dict.get(n, Py_NULL())
            local = local_callback(n, local)
            localsplus.append(local)
            # NOTE Updates locals_dict on Python 3.10, so co_cellvars has the same replacements
            #   as localsplus does (this is not necessary on Python 3.11, because there cellvars are
            #   constructed using the MAKE_CELL instruction)
            locals_dict[n] = local
        for n in code.co_cellvars:
            if n in locals_dict:
                c = _interpret_call(CellType, locals_dict[n])
                assert c is not INTERPRETER_SIGNALS.EXCEPTION_RAISED
                localsplus.append(c)
                localsplus[code.co_varnames.index(n)] = Py_NULL()
            else:
                localsplus.append(wrap_const(CellType()))
        for i, (name, value) in enumerate(zip(code.co_freevars, closure)):
            local = freevar_callback(name, value, fn=wrapped_fn, idx=i)
            localsplus.append(local)
    elif (3, 11) <= sys.version_info < (3, 12):
        assert len(code.co_varnames) == code.co_nlocals
        for n in code.co_varnames:
            local = locals_dict.get(n, Py_NULL())
            local = local_callback(n, local)
            localsplus.append(local)
        for n in code.co_cellvars:
            # those in locals_dict will use that index but will
            # see MAKE_CELL called for them for the conversion
            if n not in locals_dict:
                localsplus.append(Py_NULL())
        for i, (name, value) in enumerate(zip(code.co_freevars, closure)):
            local = freevar_callback(name, value, fn=wrapped_fn, idx=i)
            localsplus.append(local)
    else:
        raise NotImplementedError(
            f"Python version {sys.version_info.major}.{sys.version_info.minor} is not supported at this moment."
        )

    if compilectx._with_provenance_tracking:
        frame_globals = wrap_attribute(wrapped_fn.value.__globals__, wrapped_fn, wrap_const("__globals__"))
        frame_builtins = wrap(builtins_dict, provenance=ProvenanceRecord(inst=PseudoInst.BUILTINS, inputs=[]))
    else:
        frame_globals = fn.__globals__
        frame_builtins = builtins_dict

    # Creates the current ready to run stack frame for the current function
    frame = InterpreterFrame(
        code=code, localsplus=localsplus, globals=frame_globals, names=wrap_const({}), qualname=fn.__qualname__
    )

    # Python 3.10 deals with creating the generator on call,
    # 3.11+ use the RETURN_GENERATOR opcode
    if sys.version_info < (3, 11):
        if code.co_flags & inspect.CO_GENERATOR:
            return make_generator(frame, compilectx, runtimectx)
        if code.co_flags & inspect.CO_COROUTINE:
            return make_coroutine(frame, compilectx, runtimectx)
        if code.co_flags & inspect.CO_ASYNC_GENERATOR:
            return make_async_generator(frame, compilectx, runtimectx)

    try:
        res, status = _run_frame(frame, compilectx, runtimectx)
    except Exception as e:
        # We need to cheat a bit to get a Python frame here...
        python_frame = frame.get_or_make_python_frame()
        tb = TracebackType(e.__traceback__, python_frame, python_frame.f_lasti, python_frame.f_lineno)
        raise e.with_traceback(tb)
    return res


def _run_frame(
    frame: InterpreterFrame,
    compilectx: InterpreterCompileCtx,
    runtimectx: InterpreterRuntimeCtx,
    *,
    send_value: Any = Py_NULL(),
):
    # Pushes the current stack frame for the current function
    with runtimectx.push_frame_stack(frame):
        stack: InterpreterStack = frame.interpreter_stack
        if send_value != Py_NULL():
            with stack.set_cur_instruction(PseudoInst.SEND):
                stack.append(send_value)

        insts: tuple[dis.Instruction, ...] = tuple(dis.get_instructions(frame.code))
        inst_ptr_to_idx = {inst.offset // 2: idx for idx, inst in enumerate(insts)}
        max_inst_ptr = max(inst_ptr_to_idx.keys())
        while True:
            # we might have jumped or advanced to a "hidden" instruction such as cache,
            # so move forward until we have something to look at.
            # N.B.: For Python 3.12 there is a change coming up that changes how to
            #       consider CACHE items in computing relative jumps.
            #       This is discussed at the top of
            #       https://docs.python.org/3.12/library/dis.html
            while frame.inst_ptr not in inst_ptr_to_idx:
                assert frame.inst_ptr <= max_inst_ptr
                frame.inst_ptr += 1
            inst: dis.Instruction = insts[inst_ptr_to_idx[frame.inst_ptr]]

            # Updates the stack frame to the current position
            # TODO maybe also have inst_ptr?
            frame.nexti(inst)
            runtimectx.record_interpreted_instruction(inst)
            skip_stack_effect_check: bool = False  # the exception handling will change the stack wildly
            stack_size_before_handler: int = len(stack)

            frame.lasti = frame.inst_ptr  # ???
            interpretation_result: None | int | INTERPRETER_SIGNALS = compilectx.interpret(
                inst,
                inst_ptr=frame.inst_ptr,
                stack=frame.interpreter_stack,
                globals_dict=frame.globals,
                try_stack=frame.try_stack,
                exception_stack=runtimectx.exception_stack,
                co=frame.code,
                frame=frame,
            )
            if interpretation_result is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                e = runtimectx.curexc
                runtimectx.curexc = None
                assert isinstance(e, BaseException)
                current_exception = e

                if sys.version_info >= (3, 11):
                    exception_table = dis._parse_exception_table(frame.code)  # type: ignore (_parse_exception_table is undocumented)

                    found = False
                    et_start = et_end = et_handler = et_level = et_lasti = 0  # For type checker
                    for et_start, et_end, et_handler, et_level, et_lasti in exception_table:
                        found = et_start <= frame.inst_ptr * 2 < et_end
                        if found:
                            break
                    if found:
                        assert len(frame.interpreter_stack) >= et_level
                        del frame.interpreter_stack[et_level:]
                        if et_lasti:
                            with frame.interpreter_stack.set_cur_instruction(PseudoInst.EXCEPTION_HANDLER):
                                frame.interpreter_stack.append(frame.lasti)
                        with frame.interpreter_stack.set_cur_instruction(PseudoInst.EXCEPTION_HANDLER):
                            frame.interpreter_stack.append(current_exception)
                        current_exception = None
                        skip_stack_effect_check = True
                        interpretation_result = et_handler // 2
                else:
                    # This is Python 3.10-style unwinding
                    skip_stack_effect_check = True  # or only do this in ifs below?
                    while frame.try_stack:
                        try_block = frame.try_stack.pop()
                        if try_block.typ == PyTryBlock.EXCEPT_HANDLER_TYPE:
                            assert len(frame.interpreter_stack) >= try_block.level + 3
                            with frame.interpreter_stack.set_cur_instruction(PseudoInst.EXCEPTION_HANDLER):
                                del frame.interpreter_stack[try_block.level + 3 :]
                                exc_type = frame.interpreter_stack.pop()  # we ignore that and assume == type(exc_value)
                                exc_value = frame.interpreter_stack.pop()
                                exc_traceback = frame.interpreter_stack.pop()
                            if exc_value != None:
                                exc_value.__traceback__ = exc_traceback
                            assert runtimectx.exception_stack
                            # CPython sets exc_info->exc_type/value/traceback
                            # see RuntimeCtx inititalization of exception_stack for more info
                            runtimectx.exception_stack[-1] = exc_value  # replace the exc_info
                            # Python 3.10 has `continue` here, but there is no code except the else
                        else:
                            # There are actually only these two PyTryBlock types in 3.10
                            assert try_block.typ == PyTryBlock.SETUP_FINALLY_TYPE

                            # Python 3.10 UNWIND_BLOCK
                            assert len(frame.interpreter_stack) >= try_block.level
                            with frame.interpreter_stack.set_cur_instruction(PseudoInst.EXCEPTION_HANDLER):
                                del frame.interpreter_stack[try_block.level :]

                            # Python 3.10 handling of SETUP_FINALLY blocks
                            frame.try_stack.append(
                                PyTryBlock(PyTryBlock.EXCEPT_HANDLER_TYPE, frame.lasti, len(frame.interpreter_stack))
                            )
                            assert runtimectx.exception_stack
                            # CPython sreads exc_info->exc_type/value/traceback
                            # see RuntimeCtx inititalization of exception_stack for more info
                            exc = runtimectx.exception_stack[-1]
                            with frame.interpreter_stack.set_cur_instruction(PseudoInst.EXCEPTION_HANDLER):
                                frame.interpreter_stack.append(exc.__traceback__ if exc is not None else None)
                                frame.interpreter_stack.append(exc)
                                # Python distinguishes explicit exc_type present or NULL/None
                                frame.interpreter_stack.append(type(exc))
                            current_exception = e
                            # NormalizeException ?

                            # CPython sets exc_info->exc_type/value/traceback here
                            # see RuntimeCtx inititalization of exception_stack for more info
                            runtimectx.exception_stack[-1] = current_exception
                            with frame.interpreter_stack.set_cur_instruction(PseudoInst.EXCEPTION_HANDLER):
                                frame.interpreter_stack.append(
                                    current_exception.__traceback__ if exc is not None else None
                                )
                                frame.interpreter_stack.append(current_exception)
                                # Python distinguishes explicit exc_type present or NULL/None
                                frame.interpreter_stack.append(type(current_exception))
                            current_exception = None
                            interpretation_result = try_block.handler
                            # f->f_state = FRAME_EXECUTING;  /* Resume normal execution */
                            break  # continue with handler
                if current_exception is not None:
                    e = current_exception
                    # We need to cheat a bit to get a Python frame here...
                    python_frame = frame.get_or_make_python_frame()
                    tb = TracebackType(e.__traceback__, python_frame, python_frame.f_lasti, python_frame.f_lineno)
                    e = e.with_traceback(tb)
                    runtimectx.curexc = e
                    return INTERPRETER_SIGNALS.EXCEPTION_RAISED, INTERPRETER_SIGNALS.EXCEPTION_RAISED

            # TODO Improve this error message
            if interpretation_result is INTERPRETER_SIGNALS.UNHANDLED_OPCODE:
                raise NotImplementedError(f"Encountered unimplemented opcode {inst.opname} while tracing.")

            elif interpretation_result in (INTERPRETER_SIGNALS.RETURN_VALUE, INTERPRETER_SIGNALS.YIELD_VALUE):
                # advance the inst_ptr, needed in particular for YIELD
                frame.inst_ptr += 1
                # get the result from the current stack

                result = frame.interpreter_stack.pop_wrapped()
                # Restores the previous stack, the caller needs to put the value on it
                return result, interpretation_result
            elif interpretation_result is INTERPRETER_SIGNALS.RETURN_GENERATOR:
                frame.inst_ptr += 1
                if frame.code.co_flags & inspect.CO_GENERATOR:
                    return make_generator(frame, compilectx, runtimectx), interpretation_result
                if frame.code.co_flags & inspect.CO_COROUTINE:
                    return make_coroutine(frame, compilectx, runtimectx), interpretation_result
                if frame.code.co_flags & inspect.CO_ASYNC_GENERATOR:
                    return make_async_generator(frame, compilectx, runtimectx), interpretation_result
                raise NotImplementedError(
                    f"Not implemented: RETURN_GENERATOR from code of {frame.qualname} with flags {dis.pretty_flags(frame.code.co_flags)}"
                )
            elif interpretation_result is None:
                frame.inst_ptr += 1
            else:
                assert isinstance(interpretation_result, int), interpretation_result
                frame.inst_ptr = interpretation_result

            if not skip_stack_effect_check:  # the exception handling will change the stack wildly
                # Verifies the handler had the expected stack effect (delta on stack sie)
                actual_stack_effect: int = len(stack) - stack_size_before_handler
                jumped: bool = isinstance(interpretation_result, int) and interpretation_result != -1
                expected_stack_effect: int = dis.stack_effect(inst.opcode, inst.arg, jump=jumped)

                if (3, 11) <= sys.version_info < (3, 12):
                    # PRECALL stack effect (3.11) has a -inst.arg stack effect in the function that we only see during CALL
                    if inst.opname == "PRECALL":
                        assert type(inst.arg) is int
                        assert (
                            expected_stack_effect == -inst.arg
                        ), f"precall with stack effect {expected_stack_effect}, {inst}"
                        expected_stack_effect = 0
                    elif inst.opname == "CALL":
                        assert type(inst.arg) is int
                        assert expected_stack_effect == -1, f"call with stack effect {expected_stack_effect}, {inst}"
                        expected_stack_effect = -inst.arg - 1
                assert (
                    actual_stack_effect == expected_stack_effect
                ), f"Unexpected stack effect from {inst.opname}: expected {expected_stack_effect}, but the actual effect was {actual_stack_effect} at {inst}"


# Special signals for the interpreter
# TODO Consider a different name for this class
class INTERPRETER_SIGNALS(enum.Enum):
    UNHANDLED_OPCODE = enum.auto()
    UNSAFE_FUNCTION = enum.auto()
    RETURN_VALUE = enum.auto()
    RETURN_GENERATOR = enum.auto()
    YIELD_VALUE = enum.auto()
    EXCEPTION_RAISED = enum.auto()


#
# Defines interpreter ux
#


# Interprets the Python program
# The interpretation can be extended by specifying one or more of the following:
#   (1) The opcode_interpreter function, which has the signature
#
#   opcode_interpreter(inst: dist.Instruction, /, **interpreter_state) -> None | int | INTERPRETER_SIGNALS
#
#   The opcode handler is called to interpret an opcode, and is an opportunity to
#   implement custom opcode handling.
#
#   If the opcode is unhandled, then INTERPRETER_SIGNALS.UNHANDLED_OPCODE should be returned.
#
#   Otherwise the function will be called with the following keyword arguments:
#
#       - stack, the interpreter stack
#       - inst_ptr, the current instruction pointer
#       - co, the code object
#       - globals_dict, the globals dictionary
#       - builtins_dict, the builtins dictionary
#       - frame: interpreter frame containing local variables, source loc etc.
#       - try_stack, a "try block" stack to facilitate handling exceptions
#       - exception_stack, the stack of currently handled exceptions
#
#   The handler can then return None, -1 to indicate a return statement, or a weakly positive integer
#   to indicate that the interpreter should jump absolute to that instruction.
#
#   The arguments passed to the handler are very likely to change in the near future, but most
#   handlers only need to consume a small subset of the above arguments.
#
#   (2) The fn_lookaside function, which has the signature
#
#   fn_lookaside(fn, *args, **kwargs) -> None | Callable
#
#   The function 'lookaside' is an opportunity to intercept functions and either provide custom
#   implementations of them or raise exceptions if they are "unsafe".
#   It is called whenever a function is interpreterd.
#
#   If there is no lookaside, then None should be returned.
#
#   If the function is unsafe, then a callable that raises UnsafeOperator (to be implemented)
#       should be returned.
#
#   Otherwise, the function should implement the lookaside when called with the same args and kwargs.
def interpret(
    fn: Callable,
    *,
    opcode_interpreter: Callable = default_opcode_interpreter,
    fn_lookaside: Callable = default_lookaside,
    callbacks: dict[INTERPRETER_CALLBACKS, Callable] = default_callbacks,
    debug_log: None | StringIO = None,
    with_provenance_tracking: bool = False,
    uncacheable_classes: list[type] | None = None,
) -> Callable:
    compilectx: InterpreterCompileCtx = InterpreterCompileCtx(
        opcode_interpreter=opcode_interpreter,
        fn_lookaside=fn_lookaside,
        callbacks=callbacks,
        with_provenance_tracking=with_provenance_tracking,
        uncacheable_classes=uncacheable_classes,
    )
    if hasattr(fn, "__thunder_interpreter_orig_fn"):
        fn = fn.__thunder_interpreter_orig_fn

    @functools.wraps(fn)
    def fn_(*args, **kwargs) -> Any:
        runtimectx: InterpreterRuntimeCtx = InterpreterRuntimeCtx(debug_log=debug_log)

        with interpreter_ctx(compilectx, runtimectx):
            try:
                # we normalize the outmost function to be interpreted to take
                # args and kwargs as arguments (not *args and **kwargs).
                # We thus have three special INPUTs for the entry function: INPUT_ARGS, INPUT_KWARGS, INPUT_FN
                args = wrap(
                    args,
                    provenance=ProvenanceRecord(inst=PseudoInst.INPUT_ARGS, inputs=[]),
                )

                kwargs = wrap(
                    kwargs,
                    provenance=ProvenanceRecord(inst=PseudoInst.INPUT_KWARGS, inputs=[]),
                )

                fn_wrapped = wrap(
                    fn,
                    provenance=ProvenanceRecord(inst=PseudoInst.INPUT_FN, inputs=[]),
                )

                def getfn():
                    def fn_2(args, kwargs):
                        return fn(*args, **kwargs)

                    return fn_2

                wrapped_fn_2 = wrap_const(getfn())
                if compilectx._with_provenance_tracking:
                    wrapped_closure = wrap_attribute(
                        wrapped_fn_2.value.__closure__, wrapped_fn_2, wrap_const("__closure__")
                    )
                    wrapped_cell = wrap_binary_subscr(wrapped_closure.value[0], wrapped_closure, 0)
                    assert isinstance(wrapped_closure.item_wrappers, list)
                    wrapped_closure.item_wrappers[0] = wrapped_cell
                    populate_attribute_wrapper(wrapped_cell, "cell_contents", fn_wrapped)

                interpretation_result: Any = _interpret_call(wrapped_fn_2, args, kwargs)
                interpretation_result = unwrap(interpretation_result)
            except Exception as e:
                # TODO Highlight the portion of the line that originated the opcode on Python versions that include
                #      the line offset information in the instruction
                traceback_str = os.linesep.join(f.format_with_source() for f in runtimectx.frame_stack)
                msg = (
                    f"Encountered exception {type(e).__name__}: {e} while tracing {fn}:{os.linesep}" f"{traceback_str}"
                )
                raise InterpreterError(msg) from e
            finally:
                # NOTE: Wrapped functions are valid to assign new attributes to.
                fn_._last_interpreted_instructions = runtimectx.interpreted_instructions  # type: ignore
                fn_._last_interpreted_history = runtimectx.history  # type: ignore

            # # NOTE: Wrapped functions are valid to assign new attributes to.
            # fn_._last_interpreted_instructions = runtimectx.interpreted_instructions  # type: ignore
            # fn_._last_interpreted_history = runtimectx.history  # type: ignore

            if interpretation_result is INTERPRETER_SIGNALS.EXCEPTION_RAISED:
                e = runtimectx.curexc
                assert isinstance(e, BaseException), e
                runtimectx.curexc = None
                raise e

            return interpretation_result

    fn_.__thunder_interpreter_orig_fn = fn  # type: ignore
    return fn_


def last_interpreted_instructions(fn: Callable) -> None | list[dis.Instruction]:
    return getattr(fn, "_last_interpreted_instructions", None)


def last_interpreted_history(fn: Callable) -> None | list[InterpreterHistoryItem]:
    return getattr(fn, "_last_interpreted_history", None)


def print_history(
    history: list[InterpreterHistoryItem],
    /,
    print_fn: Callable = print,
    use_colors: bool = True,
    indent: bool = True,
    max_depth: int | None = None,
    color_internals: bool = False,
    print_source_code: bool = True,
) -> None:
    colors = init_colors(use_colors)
    interpreter_path = os.path.join("thunder", "core", "interpreter.py")

    c_indent = -1
    inside_inner_interpreter = False
    for item in history:
        linecolor = ""
        nl = ""
        deindent = False
        source_line = None

        # Match each kind of history item. The history items are instructions, strings,
        # or typed dicts, with "kind" describing what kind of entry it is.
        match item:
            case dis.Instruction():
                if color_internals or not inside_inner_interpreter:
                    linecolor = colors["MAGENTA"]
                history_line = f"Instruction('{item.opname}', arg={item.arg}, argrepr={repr(item.argrepr)})"

            case str():
                # Print the string as-is, indented, without colors.
                linecolor = colors["RESET"]
                history_line = item

            case {"kind": "Line", "fn": _fn, "filename": filename, "position": position}:
                # LineHistoryItem
                inside_inner_interpreter = interpreter_path in filename
                if color_internals or not inside_inner_interpreter:
                    linecolor = colors["YELLOW"]
                nl = os.linesep
                fnname = extract_callable_name(_fn)
                if position:
                    history_line = f"# Line {filename}:{position.lineno} in {fnname}()"
                else:
                    history_line = f"# {filename} in {fnname}()"

                if not print_source_code or not position:
                    continue

                first_lineno = position.lineno
                assert first_lineno
                linestr = linecache.getline(filename, first_lineno)
                if linestr.endswith(os.linesep):
                    linestr = linestr[: -len(os.linesep)]
                source_line = linestr

            case {"kind": "InterpreterCall", "fn": fn, "prev_frame": prev_frame}:
                # CallHistoryItem
                if color_internals or not inside_inner_interpreter:
                    linecolor = colors["GREEN"]
                c_indent += 1
                history_line = f"Interpreting call to {extract_callable_name(fn)}() from {prev_frame}{'()' if not prev_frame.endswith('>') else ''}"

            case {"kind": "InterpreterReturn", "fn": fn, "is_signal": is_signal, "rval": rval}:
                # ReturnHistoryItem
                if color_internals or not inside_inner_interpreter:
                    linecolor = colors["RED"]
                deindent = True
                meaning = "signal" if is_signal else "value of type"
                val = rval if is_signal else rval.__qualname__
                history_line = f"Returning from call to {extract_callable_name(fn)}() with {meaning} {val}"

            case {"kind": "Lookaside", "fn": fn}:
                # LookasideHistoryItem
                if color_internals or not inside_inner_interpreter:
                    linecolor = colors["BLUE"]
                history_line = f"Lookaside to {extract_callable_name(fn)}()"

            case {"kind": "Opaque", "fn": fn}:
                # OpaqueHistoryItem
                if color_internals or not inside_inner_interpreter:
                    linecolor = colors["CYAN"]
                history_line = f"Opaque call to {fn} with name {extract_callable_name(fn)}"

            case _:
                raise NotImplementedError(f"Unexpected history item {item}")

        if max_depth is None or c_indent <= max_depth:
            print_fn(f"{nl}{' ' * c_indent if indent else ''}{linecolor}{history_line}{colors['RESET']}")

            if source_line:
                print_fn(f"{' ' * c_indent if indent else ''}{linecolor}{source_line}{colors['RESET']}")

        if deindent:
            c_indent -= 1


def print_last_interpreted_history(
    fn: Callable,
    /,
    print_fn: Callable = print,
    use_colors: bool = True,
    indent: bool = True,
    max_depth: int | None = None,
    color_internals: bool = False,
    print_source_code: bool = True,
) -> None:
    if (history := last_interpreted_history(fn)) is None:
        print("No history could be found.")
        return
    print_history(
        history,
        print_fn=print_fn,
        use_colors=use_colors,
        indent=indent,
        max_depth=max_depth,
        color_internals=color_internals,
        print_source_code=print_source_code,
    )
