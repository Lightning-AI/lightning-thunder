from __future__ import annotations

import dis
import sys
import collections
from collections.abc import Iterator, Sequence, Callable, Iterable, Mapping
from dataclasses import dataclass, field
import inspect
import linecache
from typing import Any
from types import (
    CellType,
    CodeType,
    FunctionType,
    MethodType,
    BuiltinFunctionType,
    BuiltinMethodType,
    MethodWrapperType,
    WrapperDescriptorType,
)
import functools
from functools import partial
from enum import Enum, auto
from numbers import Number
from contextvars import ContextVar
from contextlib import contextmanager
import builtins

import torch

from thunder.core.trace import TraceCtx, tracectx
import thunder.core.prims as prims
from thunder.common import _execute_trace, CompileData, CompileStats, CACHE_MODES
from thunder.extend import Executor
import thunder.torch as ltorch
from thunder.torch import _torch_to_thunder_function_map
from thunder.core.proxies import Proxy
from thunder.core.baseutils import Singleton

#
# jit.py implements a Python interpreter in Python.
#
#   The jit is still being developed.
#   See thunder/tests/test_jit.py for its current capabilities.
#
#   Python opcodes are executed by "handlers" that manipulate the stack and other
#   components of the interpreter state, and can return a new instruction pointer
#   for the interpreter to jump to.
#
#   The jit has a JitMode.PYTHON mode, where it just tries to emulate the
#   Python interpreter. This is useful when adding opcodes, and for verifying
#   that the Python updates are correct.
#
#   A work in progress is JitMode.THUNDER. When this mode is set the jit
#   avoids Python side effects and constructs a Thunder program to be executed.
#
#   The Thunder program constructed has two parts, a "prologue trace" and a
#   "computation trace". The prologue trace has the original function's signature
#   and it:
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

__all__ = [
    "jit",
]

#
# Jit context
#

# Constructs the builtins dictionary
builtins_dict: dict[str, Any] = {k: getattr(builtins, k) for k in dir(builtins)}


# The Jit's compile context, which handles compilation directives
# See the comment for jit() for how these functions work
class JitCompileCtx:
    def __init__(self, *, opcode_interpreter: Callable, fn_lookaside: Callable):
        self._opcode_interpreter: Callable = opcode_interpreter
        self._fn_lookaside: Callable = fn_lookaside

    def interpret(self, inst: dis.Instruction, /, **interpreter_state) -> None | int | JIT_SIGNALS:
        return self._opcode_interpreter(inst, **interpreter_state)

    def lookaside(self, fn: Callable, *args, **kwargs) -> tuple[bool, None | Any] | JIT_SIGNALS:
        return self._fn_lookaside(fn)


_jitcompilectx = ContextVar("jitcompilectx")


# Sets the jit ctx
def set_jitcompilectx(ctx) -> Any:
    return _jitcompilectx.set(ctx)


# Returns the current jitctx
def get_jitcompilectx() -> JitCompileCtx:
    return _jitcompilectx.get()


# Resets the jitctx
def reset_jitcompilectx(token) -> None:
    _jitcompilectx.reset(token)


# Context manager for setting the jitctx
@contextmanager
def jitcompilectx(_jitcompilectx: JitCompileCtx):
    tok: Any = set_jitcompilectx(_jitcompilectx)
    try:
        yield
    finally:
        reset_jitcompilectx(tok)


# The Jit's runtime context, which tracks stack changes in Python mode
# TODO Merge interpreter stack into frame stack?
class JitRuntimeCtx:
    def __init__(self):
        self.frame_stack = []
        self._interpreter_stacks = [[]]

    @property
    def globals_dict(self) -> dict[str, Any]:
        return self._globals_dict

    def push_interpreter_stack(self) -> list:
        interpreter_stack: list = []
        self._interpreter_stacks.append(interpreter_stack)
        return interpreter_stack

    def peek_interpreter_stack(self) -> list:
        return self._interpreter_stacks[-1]

    def pop_interpreter_stack(self) -> list:
        return self._interpreter_stacks.pop()

    # advance to the given instruction
    def frame_stack_change_top(self, inst: dis.Instruction):
        assert self.frame_stack
        self.frame_stack[-1].nexti(inst)

    # get current top of stack
    def peek_frame_stack(self) -> list:
        return self.frame_stack[-1]

    # for method calls. There is a bit of a trick because the filename is
    # part of the code object, but not the instruction's position information.
    def push_frame_stack(self, code: CodeType, localsplus: list[Any], qualname: str):
        self.frame_stack.append(JITFrame(code=code, localsplus=localsplus, qualname=qualname))

    # for returning from method calls
    def pop_frame_stack(self):
        assert self.frame_stack
        del self.frame_stack[-1]


_jitruntimectx = ContextVar("jitruntimectx")


# Sets the jit ctx
def set_jitruntimectx(ctx) -> Any:
    return _jitruntimectx.set(ctx)


# Returns the current jitctx
def get_jitruntimectx() -> JitRuntimeCtx:
    return _jitruntimectx.get()


# Resets the jitctx
def reset_jitruntimectx(token) -> None:
    _jitruntimectx.reset(token)


# Context manager for setting the jitctx
@contextmanager
def jitruntimectx(_jitruntimectx: JitRuntimeCtx):
    tok: Any = set_jitruntimectx(_jitruntimectx)
    try:
        yield
    finally:
        reset_jitruntimectx(tok)


# A convenience helper for setting both the jit compile and runtime ctx
@contextmanager
def jitctx(_jitcompilectx: JitCompileCtx, _jitruntimectx: JitRuntimeCtx):
    compile_tok: Any = set_jitcompilectx(_jitcompilectx)
    runtime_tok: Any = set_jitruntimectx(_jitruntimectx)
    try:
        yield
    finally:
        reset_jitcompilectx(compile_tok)
        reset_jitruntimectx(runtime_tok)


#
# Helpers
#


def is_opaque(fn: Callable) -> bool:
    if isinstance(fn, (BuiltinFunctionType, BuiltinMethodType, MethodWrapperType, WrapperDescriptorType)):
        return True

    # NOTE builtins.type has type type, but type() is an opaque function
    if fn is type:
        return True

    return False


# Acquires the code object from a function or method (converting methods to functions)
# TODO Print a nicer error message
def extract_code(fn: Callable) -> CodeType:
    if isinstance(fn, MethodType):
        return extract_code(fn.__func__)

    if not hasattr(fn, "__code__"):
        raise ValueError(f"Cannot JIT object {repr(fn)} of type {type(fn)}")

    code: CodeType = fn.__code__

    return code


# Our own exception class for reporting compilation problems
class JITError(RuntimeError):
    pass


class Py_NULL(metaclass=Singleton):
    pass


@dataclass
class TryBlock:
    def __init__(self):
        pass


@dataclass
class PyErr_StackItem:
    def __init__(self, exc_type, exc_value, exc_traceback):
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.exc_traceback = exc_traceback


# Use dis.Positions in 3.11+ and make it up in <3.11
if sys.version_info < (3, 11):
    Positions = collections.namedtuple(
        "Positions",
        [
            "lineno",
            "end_lineno",
            "col_offset",
            "end_col_offset",
        ],
        defaults=[None] * 4,
    )
else:
    Positions = dis.Positions


# This is an interpreter frame, similar to Python's for use fo the JIT
# currently, we mainly use this to provide exception tracebacks,
# so in contrast to Python we don't (yet) store globals / locals etc. here.
@dataclass
class JITFrame:
    code: CodeType
    qualname: str
    positions: Positions | None = None
    inst: dis.Instruction | None = None
    call_shape_kwnames: tuple[str] | None = None  # for KW_NAMES opcode in 3.11+

    # in Python 3.11+ the slots are not split by local/cell/free any more
    localsplus: list[Any] = field(default_factory=list)

    def nexti(self, inst: dis.Instruction):
        self.inst = inst
        if (3, 9) <= sys.version_info < (3, 11):
            if inst.starts_line is not None:
                self.positions = Positions(inst.starts_line, inst.starts_line, 0, 999)
        elif (3, 11) <= sys.version_info < (3, 12):
            self.positions = inst.positions
        else:
            raise NotImplementedError(f"Python {sys.version_info} not supported")

    def format_with_source(self):
        # todo: multiple lines in positions, underline, indent
        l = []
        l.append(f"  in {self.qualname} in file: {self.code.co_filename}, line {self.positions.lineno}:")
        if self.code.co_filename:
            ls = linecache.getlines(self.code.co_filename)
            lineno = self.positions.lineno
            if lineno is None:
                lineno = self.code.co_firstlineno
            l.append("  " + ls[max(lineno - 1, 0)].rstrip())
        return "\n".join(l)

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
            return self.code._varname_from_oparg(idx)


#
# Handler registration
#

_default_opcode_handler_map: dict[str, Callable] = {}


def default_opcode_interpreter(inst: dis.Instruction, /, **interpreter_state) -> None | int | JIT_SIGNALS:
    handler: None | Callable = _default_opcode_handler_map.get(inst.opname, None)
    if handler is None:
        return JIT_SIGNALS.UNHANDLED_OPCODE

    return handler(inst, **interpreter_state)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1528
class register_opcode_handler:
    def __init__(self, name: str, *, min_ver: tuple[int, int] | None = None, max_ver: tuple[int, int] | None = None):
        self.name: str = name
        self.min_ver = min_ver
        self.max_ver = max_ver

    def __call__(self, fn: Callable) -> Callable:
        if (self.min_ver is None or self.min_ver <= sys.version_info) and (
            self.max_ver is None or sys.version_info < (*self.max_ver[:-1], self.max_ver[-1] + 1)
        ):
            assert self.name not in _default_opcode_handler_map
            _default_opcode_handler_map[self.name] = fn
            return fn
        return _default_opcode_handler_map.get(self.name)


# The default function lookaside -- currently it doesn't intercept anything
def default_fn_lookaside(fn, *args, **kwargs) -> tuple[bool, None | Any] | JIT_SIGNALS:
    return (False, None)


#
# Python opcode handlers (sorted alphabetically)
#


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1529
# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1530
# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_ADD
@register_opcode_handler("BINARY_ADD", max_ver=(3, 10))
def _binary_add_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()

    def impl():
        if (not hasattr(a, "__add__")) or ((result := a.__add__(b)) is NotImplemented):
            if (not hasattr(b, "__radd__")) or ((result := b.__radd__(a)) is NotImplemented):
                # TODO Restore formatting once FORMAT_VALUE is implemented
                # raise TypeError(f"Unsupported operand type(s) for +: '{type(a)}' and '{type(b)}'")
                err: TypeError = TypeError("Unsupported operand types for binary add")
                raise err

        return result

    _jit(impl)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_MULTIPLY
@register_opcode_handler("BINARY_MULTIPLY", max_ver=(3, 10))
def _binary_multiply_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()

    def impl():
        if (not hasattr(a, "__mul__")) or ((result := a.__mul__(b)) is NotImplemented):
            if (not hasattr(b, "__rmul__")) or ((result := b.__rmul__(a)) is NotImplemented):
                # TODO Restore formatting once FORMAT_VALUE is implemented
                # raise TypeError(f"Unsupported operand type(s) for *: '{type(a)}' and '{type(b)}'")
                err: TypeError = TypeError("Unsupported operand types for binary multiply")
                raise err

        return result

    _jit(impl)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_SUBTRACT
@register_opcode_handler("BINARY_SUBTRACT", max_ver=(3, 10))
def _binary_subtract_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()

    def impl():
        if (not hasattr(a, "__sub__")) or ((result := a.__sub__(b)) is NotImplemented):
            if (not hasattr(b, "__rsub__")) or ((result := b.__rsub__(a)) is NotImplemented):
                # TODO Restore formatting once FORMAT_VALUE is implemented
                # raise TypeError(f"Unsupported operand type(s) for -: '{type(a)}' and '{type(b)}'")
                err: TypeError = TypeError("Unsupported operand types for binary subtract")
                raise err

        return result

    _jit(impl)


# https://docs.python.org/3.11/library/dis.html#opcode-BINARY_OP
@register_opcode_handler("BINARY_OP", min_ver=(3, 11))
def _binary_op_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()

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

    binop_name, *method_names = ops[inst.arg]
    if len(method_names) == 2:
        left_method, right_method = method_names

        def impl():
            if (not hasattr(a, left_method)) or ((result := getattr(a, left_method)(b)) is NotImplemented):
                if (not hasattr(b, right_method)) or ((result := getattr(b, right_method)(a)) is NotImplemented):
                    # TODO Restore formatting once FORMAT_VALUE is implemented
                    # raise TypeError(f"Unsupported operand type(s) for +: '{type(a)}' and '{type(b)}'")
                    err: TypeError = TypeError("Unsupported operand types for " + binop_name)
                    raise err

            return result

    else:
        (method,) = method_names

        def impl():
            if (not hasattr(a, method)) or ((result := getattr(a, method)(b)) is NotImplemented):
                # TODO Restore formatting once FORMAT_VALUE is implemented
                # raise TypeError(f"Unsupported operand type(s) for +: '{type(a)}' and '{type(b)}'")
                err: TypeError = TypeError("Unsupported operand types for " + binop_name)
                raise err

            return result

    _jit(impl)


# TODO Review if there's a better way to perform the subscription
# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_SUBSCR
@register_opcode_handler("BINARY_SUBSCR")
def _binary_subscr_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    tos = stack.pop()
    tos1 = stack.pop()

    # NOTE This cannot be implemented by jitting tos1[tos], since that would call this handler again
    stack.append(tos1[tos])


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_LIST
@register_opcode_handler("BUILD_LIST")
def _build_list_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    result: list[Any] = list(reversed([stack.pop() for _ in range(inst.arg)]))
    stack.append(result)


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_MAP
@register_opcode_handler("BUILD_MAP")
def _build_map_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    count: int = inst.arg

    # NOTE The reversed() call below is necessary to handle key collisions properly
    d: dict = {k: v for v, k in reversed(tuple((stack.pop(), stack.pop()) for _ in range(count)))}
    stack.append(d)


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_SLICE
@register_opcode_handler("BUILD_SLICE")
def _build_slice_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
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
def _build_string_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int

    count: int = inst.arg

    strings: tuple[str, ...] = reversed(tuple(stack.pop() for _ in range(count)))
    stack.append("".join(strings))


# https://docs.python.org/3.10/library/dis.html#opcode-BUILD_TUPLE
@register_opcode_handler("BUILD_TUPLE")
def _build_tuple_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    result: tuple[Any, ...] = tuple(reversed([stack.pop() for _ in range(inst.arg)]))
    stack.append(result)


# https://docs.python.org/3.11/library/dis.html#opcode-KW_NAMES
@register_opcode_handler("KW_NAMES", min_ver=(3, 11))
def _kw_names_handler(inst: dis.Instruction, /, stack: list, co: CodeType, frame: JITFrame, **kwargs) -> None:
    assert inst.arg is not None
    frame.call_shape_kwnames = co.co_consts[inst.arg]


# NOTE This only accepts positional args
# https://docs.python.org/3.11/library/dis.html#opcode-CALL
@register_opcode_handler("CALL", min_ver=(3, 11))
def _call_handler(inst: dis.Instruction, /, stack: list, frame: JITFrame, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    argc: int = inst.arg
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(argc))))
    func_or_self = stack.pop()
    func_or_null = stack.pop()
    if frame.call_shape_kwnames is not None:
        kwnames = frame.call_shape_kwnames
        assert len(args) >= len(kwnames)
        kwargs = dict(zip(kwnames, args[-len(kwnames) :]))
        args = args[: -len(kwnames)]
        frame.call_shape_kwnames = None
    else:
        kwargs = {}
    if func_or_null is not Py_NULL():
        func = func_or_null
        args = (func_or_self, *args)
    else:
        func = func_or_self
    _jit(func, *args, **kwargs)


# NOTE This only accepts positional args
# https://docs.python.org/3.10/library/dis.html#opcode-CALL_FUNCTION
@register_opcode_handler("CALL_FUNCTION")
def _call_function_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    argc: int = inst.arg
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(argc))))
    func: Callable = stack.pop()

    _jit(func, *args)


# https://docs.python.org/3.10/library/dis.html#opcode-CALL_FUNCTION_EX
@register_opcode_handler("CALL_FUNCTION_EX")
def _call_function_ex_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    kwargs = stack.pop() if inst.arg & 0x01 else {}
    args = stack.pop()
    func = stack.pop()
    assert isinstance(kwargs, Mapping)
    assert isinstance(args, Iterable)
    assert isinstance(func, Callable)

    if (3, 11) <= sys.version_info:
        null = stack.pop()
        assert isinstance(null, Py_NULL)

    _jit(func, *args, **kwargs)


# https://docs.python.org/3.10/library/dis.html#opcode-CALL_FUNCTION_KW
@register_opcode_handler("CALL_FUNCTION_KW")
def _call_function_kw_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    kw_names: tuple[str, ...] = stack.pop()
    kwarg_length: int = len(kw_names)
    kwargs_flat: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(kwarg_length))))
    fn_kwargs: dict[str, Any] = {k: v for k, v in zip(kw_names, kwargs_flat)}
    assert type(inst.arg) is int
    arg_length: int = inst.arg - kwarg_length
    args = tuple(reversed(tuple(stack.pop() for _ in range(arg_length))))
    func: Callable = stack.pop()

    _jit(func, *args, **fn_kwargs)


# https://docs.python.org/3.10/library/dis.html#opcode-CALL_METHOD
@register_opcode_handler("CALL_METHOD")
def _call_method_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(inst.arg))))
    second_lm = stack.pop()
    first_lm = stack.pop()
    if first_lm is not Py_NULL():
        meth = first_lm
        args = (second_lm, *args)
    else:
        meth = second_lm

    _jit(meth, *args)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1523
# https://docs.python.org/3.10/library/dis.html#opcode-COMPARE_OP
@register_opcode_handler("COMPARE_OP")
def _compare_op_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
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
    result: Any = cmp_impls[dis.cmp_op[inst.arg]](a, b)
    stack.append(result)


# https://docs.python.org/3.10/library/dis.html#opcode-DICT_MERGE
@register_opcode_handler("DICT_MERGE")
def _dict_merge_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    a = stack.pop()
    b = stack[-1]
    # TODO: Raise inside interpreter
    assert isinstance(b, Mapping), b
    assert isinstance(a, Mapping), a
    if overlap := b.keys() & a:
        # TODO: Raise inside interpreter
        raise KeyError(f"{co.co_name} got multiple values for keyword argument {next(iter(overlap))}")
    b.update(a)


@register_opcode_handler("DICT_UPDATE")
def _dict_update_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    a = stack.pop()
    b = stack[-inst.arg]
    assert isinstance(b, Mapping), b
    assert isinstance(a, Mapping), a
    b.update(a)


# https://docs.python.org/3.10/library/dis.html#opcode-DUP_TOP
@register_opcode_handler("DUP_TOP")
def _dup_top_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    stack.append(stack[-1])


# https://docs.python.org/3.11/library/dis.html#opcode-MAKE_CELL
@register_opcode_handler("MAKE_CELL", min_ver=(3, 11))
def _make_cell_handler(inst: dis.Instruction, /, frame: JITFrame, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg
    assert i >= 0 and i < len(frame.localsplus)
    val = frame.localsplus[i]

    if isinstance(val, Py_NULL):
        # empty local variable slots (Py_NULL()) produce an empty cell
        frame.localsplus[i] = CellType()
    else:
        # wrap the current val into a cell
        frame.localsplus[i] = CellType(val)


# https://docs.python.org/3.11/library/dis.html#opcode-COPY
@register_opcode_handler("COPY", min_ver=(3, 11))
def _copy_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    assert inst.arg >= 1
    stack.append(stack[-inst.arg])


# https://docs.python.org/3.11/library/dis.html#opcode-PUSH_NULL
@register_opcode_handler("PUSH_NULL", min_ver=(3, 11))
def _push_null_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    stack.append(Py_NULL())


# https://docs.python.org/3.10/library/dis.html#opcode-FORMAT_VALUE
# TODO Extend the jitted implementation to
@register_opcode_handler("FORMAT_VALUE")
def _format_value_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    FVC_MASK: int = 0x3
    FVC_NONE: int = 0x0
    FVC_STR: int = 0x1
    FVC_REPR: int = 0x2
    FVC_ASCII: int = 0x3
    FVS_MASK: int = 0x4
    FVS_HAVE_SPEC: int = 0x4

    flags: int = inst.arg
    assert isinstance(flags, int)

    value = stack.pop()
    fmt_spec = None
    if (flags & FVS_MASK) == FVS_HAVE_SPEC:
        fmt_spec = value
        value = stack.pop()

    case: int = flags & FVC_MASK

    def impl():
        nonlocal value
        # NOTE `match` was only introduced in Python 3.10, but we support Python 3.9
        if case == FVC_NONE:
            pass
        elif case == FVC_STR:
            value = str(value)
        elif case == FVC_REPR:
            value = repr(value)
        else:
            assert case == FVC_ASCII, f"Unknown FVC_MASK in FORMAT_VALUE"
            value = ascii(value)

        formatted: str = format(value, fmt_spec) if fmt_spec is not None else format(value)
        return formatted

    _jit(impl)


# https://docs.python.org/3.10/library/dis.html#opcode-GET_LEN
@register_opcode_handler("GET_LEN")
def _get_len_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    a = stack.pop()
    stack.append(len(a))


# NOTE (mruberry) The actual implementation of IMPORT_FROM is quite complicated, and there doesn't appear
#   to be a Python exposure for the operation (unlike __import__ for IMPORT_NAME)
#   There may be a better way to model this, including by just calling "from module import name"
#   directly -- are we really worried that programs will put tensor operations in import hooks?
# https://docs.python.org/3.10/library/dis.html#opcode-IMPORT_FROM
@register_opcode_handler("IMPORT_FROM")
def _import_from_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    namei: int = inst.arg

    # NOTE The stack is peeked, not popped
    module = stack[-1]
    name: str = co.co_names[namei]

    def impl():
        return getattr(module, name)

    _jit(impl)


# https://docs.python.org/3.10/library/dis.html#opcode-IMPORT_NAME
@register_opcode_handler("IMPORT_NAME")
def _import_name_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    namei: int = inst.arg

    module_name: str = co.co_names[namei]

    fromlist = stack.pop()
    level = stack.pop()

    def impl():
        module = __import__(module_name, fromlist=fromlist, level=level)
        return module

    _jit(impl)


# https://docs.python.org/3.10/library/dis.html#opcode-IS_OP
@register_opcode_handler("IS_OP")
def _is_op_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()
    stack.append(a is not b if inst.arg == 1 else a is b)


# https://docs.python.org/3.10/library/dis.html#opcode-JUMP_FORWARD
@register_opcode_handler("JUMP_FORWARD")
def _jump_forward_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    delta: int = inst.arg
    assert isinstance(delta, int)
    return inst_ptr + delta + 1


# https://docs.python.org/3.10/library/dis.html#opcode-LIST_EXTEND
@register_opcode_handler("LIST_EXTEND")
def _list_extend_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg

    # NOTE Doesn't pop the list that's extended
    tos = stack.pop()
    l: list = stack[-i]

    # NOTE tos does not have to be a list
    assert isinstance(l, list)
    l.extend(tos)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_ATTR
@register_opcode_handler("LOAD_ATTR")
def _load_attr_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert type(inst.arg) is int

    a = stack.pop()
    name: str = co.co_names[inst.arg]

    def impl():
        return getattr(a, name)

    _jit(impl)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_CLOSURE
@register_opcode_handler("LOAD_CLOSURE")
def _load_closure_handler(inst: dis.Instruction, /, stack: list, co: CodeType, frame: JITFrame, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg

    if sys.version_info < (3, 11):
        i += co.co_nlocals

    assert i >= 0 and i < len(frame.localsplus)
    val = frame.localsplus[i]
    stack.append(val)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_CONST
@register_opcode_handler("LOAD_CONST")
def _load_const_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert type(inst.arg) is int
    stack.append(co.co_consts[inst.arg])


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_DEREF
@register_opcode_handler("LOAD_DEREF")
def _load_deref_handler(inst: dis.Instruction, /, stack: list, co: CodeType, frame: JITFrame, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg

    if sys.version_info < (3, 11):
        i += co.co_nlocals

    assert i >= 0 and i < len(frame.localsplus)
    cell = frame.localsplus[i]

    # it seems that the only way to check for an empty cell (short of
    # try... except) is comparison to another empty cell
    if cell == CellType():
        do_raise(
            NameError(f"free variable '{frame.get_localsplus_name(i)}' referenced before assignment in enclosing scope")
        )
        return -1
    val = cell.cell_contents
    stack.append(val)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_FAST
@register_opcode_handler("LOAD_FAST")
def _load_fast_handler(inst: dis.Instruction, /, stack: list, co: CodeType, frame: JITFrame, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg
    assert i >= 0 and i < len(frame.localsplus)
    val = frame.localsplus[i]

    # empty local variable slots are initialized to Py_NULL()
    if isinstance(val, Py_NULL):
        do_raise(UnboundLocalError(f"local variable '{frame.get_localsplus_name(i)}' referenced before assignment"))
        return -1

    stack.append(val)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1524
# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_GLOBAL
@register_opcode_handler("LOAD_GLOBAL")
def _load_global_handler(
    inst: dis.Instruction,
    /,
    stack: list,
    co: CodeType,
    globals_dict: dict[str, Any],
    builtins_dict: dict[str, Any],
    **kwargs,
) -> None:
    assert type(inst.arg) is int
    idx = inst.arg
    if (3, 11) <= sys.version_info:
        idx = idx // 2
    co_name: str = co.co_names[idx]

    try:
        obj = globals_dict[co_name]
    except KeyError:
        try:
            obj = builtins_dict[co_name]
        except KeyError as e:
            # TODO: UndefVariableError
            raise e
    if (3, 11) <= sys.version_info:
        # for 3.11+, the lowest bit indicates whether a NULL should be pushed
        if inst.arg & 1:
            stack.append(Py_NULL())
    stack.append(obj)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1525
@register_opcode_handler("LOAD_METHOD")
def _load_method_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert type(inst.arg) is int
    name = co.co_names[inst.arg]
    obj = stack.pop()
    try:
        meth = getattr(obj, name)
    except AttributeError as e:
        raise e

    if inspect.ismethod(meth):
        stack.append(meth.__func__)
        # meth.__self__ ihis is obj for regular methods but cls for class methods
        stack.append(meth.__self__)
    else:
        stack.append(Py_NULL())
        stack.append(meth)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1526
# https://docs.python.org/3.10/library/dis.html#opcode-MAKE_FUNCTION
@register_opcode_handler("MAKE_FUNCTION")
def _make_function_handler(inst: dis.Instruction, /, stack: list, globals_dict: dict[str, Any], **kwargs) -> None:
    if sys.version_info < (3, 11):
        name = stack.pop()
    else:
        name = ""

    fn_co: CodeType = stack.pop()
    name = fn_co.co_name

    if inst.arg != 0 and inst.arg != 0x08:
        raise NotImplementedError("Annotations on functions compiled inline are not yet supported")

    if inst.arg & 0x08:
        # Python will have built at tuple of cell vars
        # (via STORE_DEREF, LOAD_CLOSURE)
        closure = tuple(stack.pop())
        assert all(isinstance(v, CellType) for v in closure)
    else:
        closure = None

    fn = FunctionType(fn_co, globals_dict, name, closure=closure)
    stack.append(fn)


# https://docs.python.org/3.10/library/dis.html#opcode-NOP
@register_opcode_handler("NOP")
def _nop_handler(inst: dis.Instruction, /, **kwargs) -> None:
    pass


# https://docs.python.org/3.11/library/dis.html#opcode-RESUME
@register_opcode_handler("RESUME", min_ver=(3, 11))
def _resume_handler(inst: dis.Instruction, /, **kwargs) -> None:
    pass


# https://docs.python.org/3.11/library/dis.html#opcode-COPY_FREE_VARS
@register_opcode_handler("COPY_FREE_VARS", min_ver=(3, 11))
def _copy_free_vars_handler(inst: dis.Instruction, /, **kwargs) -> None:
    # we already do this when setting up the function call in _jit
    pass


# https://docs.python.org/3.11/library/dis.html#opcode-PRECALL
@register_opcode_handler("PRECALL", min_ver=(3, 11), max_ver=(3, 11))
def _precall_handler(inst: dis.Instruction, /, co: CodeType, **kwargs) -> None:
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-POP_BLOCK
@register_opcode_handler("POP_BLOCK")
def _pop_block_handler(inst: dis.Instruction, /, try_stack: list[TryBlock], **kwargs) -> None:
    try_stack.pop()


# https://docs.python.org/3.10/library/dis.html#opcode-POP_EXCEPT
@register_opcode_handler("POP_EXCEPT")
def _pop_except_handler(inst: dis.Instruction, /, try_stack: list[TryBlock], **kwargs) -> None:
    try_stack.pop()


@register_opcode_handler("POP_JUMP_FORWARD_IF_FALSE", min_ver=(3, 11))
def _pop_jump_forward_if_false_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()

    def impl():
        return bool(tos)

    _jit(impl)

    cnd: bool = stack.pop()
    if not cnd:
        return inst_ptr + inst.arg + 1

    return None


@register_opcode_handler("POP_JUMP_FORWARD_IF_TRUE", min_ver=(3, 11))
def _pop_jump_forward_if_true_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()

    def impl():
        return bool(tos)

    _jit(impl)

    cnd: bool = stack.pop()
    if cnd:
        return inst_ptr + inst.arg + 1

    return None


@register_opcode_handler("POP_JUMP_FORWARD_IF_NONE", min_ver=(3, 11))
def _pop_jump_forward_if_none_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()
    if tos is None:
        return inst_ptr + inst.arg + 1

    return None


@register_opcode_handler("POP_JUMP_IF_FALSE")
def _pop_jump_if_false_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> int | None:
    assert type(inst.arg) is int

    tos = stack.pop()

    def impl():
        return bool(tos)

    _jit(impl)

    cnd: bool = stack.pop()
    if not cnd:
        return inst.arg
    return None


@register_opcode_handler("POP_JUMP_IF_TRUE")
def _pop_jump_if_true_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> int | None:
    assert type(inst.arg) is int

    tos = stack.pop()

    def impl():
        return bool(tos)

    _jit(impl)

    cnd: bool = stack.pop()
    if cnd:
        return inst.arg
    return None


# https://docs.python.org/3.10/library/dis.html#opcode-POP_TOP
@register_opcode_handler("POP_TOP")
def _pop_top_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    stack.pop()


def do_raise(exc: Any = Py_NULL(), cause: Any = Py_NULL(), **kwargs):
    # Get the type and exception being raised
    _type: Any = Py_NULL()
    _value: Any = Py_NULL()
    if exc is Py_NULL():
        # Re-raise
        # Get topmost exception, pull type, value, and traceback from it
        # Call _PyErr_Restore()
        pass
    elif isinstance(exc, type):  # TODO review this, it's PyExceptionClass_Check
        _type = exc
        _value = _jit(exc)
        if not isinstance(_value, BaseException):  # PyExceptionInstance_Check
            do_raise(
                TypeError(f"calling {_type} should have returned an instance of BaseException, not {type(_value)}")
            )
            return
    elif isinstance(exc, BaseException):  # PyExceptionInstance_Check
        _value = exc
        _type = type(exc)  # TODO review this line
        pass
    else:
        do_raise(TypeError("exceptions must derive from BaseException"), Py_NULL())
        return

    # Attach the cause
    if cause is not Py_NULL():
        fixed_cause: Py_NULL | BaseException = Py_NULL()
        if isinstance(cause, type):  # Another PyExceptionClass_Check
            fixed_cause = _jit(cause)
            # NOTE: This check is missing from cpython, seems like a bug in cpython.
            if not isinstance(fixed_cause, BaseException):
                do_raise(
                    TypeError(
                        f"calling {cause} should have returned an instance of BaseException, not {type(fixed_cause)}"
                    )
                )
                return
        elif not isinstance(cause, BaseException):  # PyExceptionInstance_Check
            fixed_cause = cause
        elif cause is None:
            fixed_cause = Py_NULL()
        else:
            do_raise(TypeError(f"exception causes must derive from BaseException"))
            return

        _ex: BaseException = _value
        __cause: None | BaseException = None if isinstance(fixed_cause, Py_NULL) else fixed_cause
        _ex.__cause__ = __cause

    # Call PyErr_SetObject() to update the thread's state
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-RAISE_VARARGS
@register_opcode_handler("RAISE_VARARGS")
def _raise_varargs_handler(inst: dis.Instruction, /, stack: list, try_stack: list[TryBlock], **kwargs) -> None:
    cause: Any = Py_NULL()
    exc: Any = Py_NULL()
    assert type(inst.arg) is int
    if inst.arg == 2:
        cause = stack.pop()
    elif inst.arg == 1:
        exc = stack.pop()
    else:
        assert inst.arg == 0
    do_raise(exc, cause)


# https://docs.python.org/3.10/library/dis.html#opcode-RERAISE
@register_opcode_handler("RERAISE")
def _reraise_handler(inst: dis.Instruction, /, stack: list, try_stack: list[TryBlock], **kwargs) -> None:
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-RETURN_VALUE
@register_opcode_handler("RETURN_VALUE")
def _return_value_handler(inst: dis.Instruction, /, **kwargs) -> int | None:
    return -1


# https://docs.python.org/3.10/library/dis.html#opcode-ROT_N
@register_opcode_handler("ROT_N")
def _rot_n_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    assert len(stack) >= inst.arg
    stack[-inst.arg :] = (stack[-1], *stack[-inst.arg : -1])


# https://docs.python.org/3.10/library/dis.html#opcode-ROT_TWO
@register_opcode_handler("ROT_TWO")
def _rot_two_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    a = stack.pop()
    b = stack.pop()
    c = stack.pop()
    stack.append(a)
    stack.append(c)
    stack.append(b)


# https://docs.python.org/3.10/library/dis.html#opcode-SETUP_WITH
@register_opcode_handler("SETUP_WITH")
def _setup_with_handler(inst: dis.Instruction, /, try_stack: list[TryBlock], **kwargs) -> None:
    assert type(inst.arg) is int
    try_stack.append(TryBlock())


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1552
# https://docs.python.org/3.10/library/dis.html#opcode-STORE_DEREF
@register_opcode_handler("STORE_DEREF")
def _store_deref_handler(
    inst: dis.Instruction,
    /,
    stack: list,
    co: CodeType,
    frame: JITFrame,
    **kwargs,
) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg
    if sys.version_info < (3, 11):
        i += co.co_nlocals

    assert i >= 0 and i < len(frame.localsplus)

    tos = stack.pop()
    frame.localsplus[i].cell_contents = tos


# https://docs.python.org/3.10/library/dis.html#opcode-DELETE_DEREF
@register_opcode_handler("DELETE_DEREF")
def _delete_deref_handler(
    inst: dis.Instruction,
    /,
    stack: list,
    co: CodeType,
    frame: JITFrame,
    **kwargs,
) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg
    if sys.version_info < (3, 11):
        i += co.co_nlocals

    assert i >= 0 and i < len(frame.localsplus)

    del frame.localsplus[i].cell_contents


# https://docs.python.org/3.10/library/dis.html#opcode-STORE_FAST
@register_opcode_handler("STORE_FAST")
def _store_fast_handler(inst: dis.Instruction, /, stack: list, co: CodeType, frame: JITFrame, **kwargs) -> None:
    a = stack.pop()
    assert type(inst.arg) is int
    i: int = inst.arg
    frame.localsplus[i] = a


# https://docs.python.org/3.10/library/dis.html#opcode-UNPACK_SEQUENCE
@register_opcode_handler("UNPACK_SEQUENCE")
def _unpack_sequence_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    seq: Sequence = stack.pop()

    for x in reversed(seq):
        stack.append(x)


# Interprets the callable with the given args and kwargs
# NOTE There are (currently) 6 cases for interpretation:
#
#   (1) The callable has a lookaside, in which case it executes the operation or signals the operation
#           is unsafe
#   (2) The callable is opaque, in which case it's executed by the CPython interpreter and its
#           result returned
#   (3) The callable is a partial object, in which case it's recursively unwrapped
#   (4) The callable is a callable class, in which case its __call__ attribute is called recursively
#   (5) The callable is a method, in which calls its __func__ attribute is called recursively
#   (6) The callable is a FunctionType, in which case it's recursively interpretered by the jit
#
# NOTE _jit both inserts the result of what's called onto the stack it's called with and returns the result
# TODO Consider refactoring this into one function for each case
def _jit(fn: Callable, *args, **kwargs) -> Any:
    compilectx: JitCompileCtx = get_jitcompilectx()
    runtimectx: JitRuntimeCtx = get_jitruntimectx()

    # (1) Handles lookasides
    lookaside_result: tuple[bool, None | Any] | JIT_SIGNALS = compilectx.lookaside(fn, *args, **kwargs)

    # TODO Improve this error message
    if lookaside_result is JIT_SIGNALS.UNSAFE_FUNCTION:
        raise RuntimeError(f"Attempted to execute unsafe function {fn=}")

    did_lookaside: bool
    lookaside_retval: Any
    did_lookaside, lookaside_retval = lookaside_result

    # Adds the returned value to the current stack
    if did_lookaside:
        runtimectx.peek_interpreter_stack().append(lookaside_retval)
        return lookaside_retval

    # (2) Handles opaque functions, adding the returned value to the current stack
    if is_opaque(fn):
        opaque_result: Any = fn(*args, **kwargs)
        runtimectx.peek_interpreter_stack().append(opaque_result)
        return opaque_result

    # (3) Handles partial objects
    if isinstance(fn, functools.partial):
        p: functools.partial = fn
        return _jit(p.func, *(p.args + args), **(p.keywords | kwargs))

    if isinstance(fn, functools.partialmethod):
        raise NotImplementedError(
            "functools.partialmethod objects like {fn} are not currently supported, please file an issue requesting support"
        )

    # (4) Handles callable classes
    if not isinstance(fn, (FunctionType, MethodType)):
        if not hasattr(fn, "__call__"):
            raise NotImplementedError(
                f"Don't know how to jit a callable with type {type(fn)} without a __call__ method"
            )
        return _jit(fn.__call__, *args, **kwargs)

    # (5) Handles (bound) methods
    #     while we unwrap methods to __func__ when processing LOAD_METHOD (so those will not run into this)
    #     methods may also originate from LOAD_ATTR e.g. assign to variable and calling that
    if inspect.ismethod(fn):
        return _jit(fn.__func__, fn.__self__, *args, **kwargs)

    assert isinstance(fn, FunctionType), f"{fn=} had an unexpected type ({type(fn)}"

    # (6) Jits into the function
    insts: tuple[dis.Instruction, ...] = tuple(dis.get_instructions(fn))
    # adjustments for "hidden" instructiions (EXTENDED_ARGS, CACHE, ...)
    inst_ptr_to_idx = {inst.offset // 2: idx for idx, inst in enumerate(insts)}
    bound = inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    locals_dict: dict[str, Any] = dict(bound.arguments)
    globals_dict: dict[str, Any] = fn.__globals__
    try_stack: list[TryBlock] = []
    stack: list = runtimectx.push_interpreter_stack()

    code: CodeType = extract_code(fn)

    # in Python 3.10: local vars is (var_names, co_cellvars, co_freevars)
    # in Python 3.11+, these are not separated, in Python 3.10 we need to create cell vars here and add them to closures in Python 3.11 they will be dealt with though MAKE_CELL...
    localsplus: list[Any] = []

    if (3, 10) <= sys.version_info < (3, 11):
        assert len(code.co_varnames) == code.co_nlocals
        for n in code.co_varnames:
            localsplus.append(locals_dict.get(n, Py_NULL()))
        for n in code.co_cellvars:
            if n in locals_dict:
                localsplus.append(CellType(locals_dict[n]))
                localsplus[code.co_varnames.index(n)] = Py_NULL()
            else:
                localsplus.append(CellType())
        if code.co_freevars:
            assert len(code.co_freevars) == len(fn.__closure__)
            localsplus.extend(fn.__closure__)
        else:
            assert not fn.__closure__
    elif (3, 11) <= sys.version_info < (3, 12):
        assert len(code.co_varnames) == code.co_nlocals
        for n in code.co_varnames:
            localsplus.append(locals_dict.get(n, Py_NULL()))
        for n in code.co_cellvars:
            # those in locals_dict will use that index but will
            # see MAKE_CELL called for them for the conversion
            if n not in locals_dict:
                localsplus.append(CellType())
        if code.co_freevars:
            assert len(code.co_freevars) == len(fn.__closure__)
            localsplus.extend(fn.__closure__)
    else:
        raise NotImplementedError(
            f"Python version {sys.version_info.major}.{sys.version_info.minor} is not supported at this moment."
        )

    # Pushes a stack frame for the current function
    runtimectx.push_frame_stack(code, localsplus, fn.__qualname__)

    inst_ptr: int = 0
    max_inst_ptr = max(inst_ptr_to_idx.keys())
    while True:
        # we might have jumped or advanced to a "hidden" instruction such as cache,
        # so move forward until we have something to look at.
        while inst_ptr not in inst_ptr_to_idx:
            assert inst_ptr <= max_inst_ptr
            inst_ptr += 1
        inst: dis.Instruction = insts[inst_ptr_to_idx[inst_ptr]]
        # Updates the stack frame to the current position
        # TODO maybe also have inst_ptr?
        runtimectx.frame_stack_change_top(inst)
        stack_size_before_handler: int = len(stack)
        interpretation_result: None | int | JIT_SIGNALS = compilectx.interpret(
            inst,
            inst_ptr=inst_ptr,
            stack=stack,
            globals_dict=globals_dict,
            builtins_dict=builtins_dict,
            try_stack=try_stack,
            co=code,
            frame=runtimectx.peek_frame_stack(),
        )

        # TODO Improve this error message
        if interpretation_result is JIT_SIGNALS.UNHANDLED_OPCODE:
            raise NotImplementedError(f"Encountered unimplemented opcode {inst.opname} while tracing.\n")

        if interpretation_result == -1:
            # Restores the previous stack and puts the returned value onto it
            result: Any = stack.pop()
            runtimectx.pop_interpreter_stack()
            runtimectx.peek_interpreter_stack().append(result)
            runtimectx.pop_frame_stack()
            return result
        elif interpretation_result is None:
            inst_ptr += 1
        else:
            assert isinstance(interpretation_result, int)
            inst_ptr = interpretation_result

        # Verifies the handler had the expected stack effect (delta on stack size)
        actual_stack_effect: int = len(stack) - stack_size_before_handler
        jumped: bool = isinstance(interpretation_result, int) and interpretation_result != -1
        expected_stack_effect: int = dis.stack_effect(inst.opcode, inst.arg, jump=jumped)

        if (3, 11) <= sys.version_info < (3, 12):
            # PRECALL stack effect (3.11) has a -2 stck effect in the function that we only see during CALL
            if inst.opname == "PRECALL":
                assert expected_stack_effect == -inst.arg, f"precall with stack effect {expected_stack_effect}, {inst}"
                expected_stack_effect = 0
            elif inst.opname == "CALL":
                assert expected_stack_effect == -1, f"call with stack effect {expected_stack_effect}, {inst}"
                expected_stack_effect = -inst.arg - 1
        assert (
            actual_stack_effect == expected_stack_effect
        ), f"Unexpected stack effect from {inst.opname}: expected {expected_stack_effect}, but the actual effect was {actual_stack_effect} at {inst}"


# Special signals for the interpreter
# TODO Consider a different name for this class
class JIT_SIGNALS(Enum):
    UNHANDLED_OPCODE = auto()
    UNSAFE_FUNCTION = auto()


# Interprets the Python program
# The interpretation can be extended by specifying one or more of the following:
#   (1) The opcode_interpreter function, which has the signature
#
#   opcode_interpreter(inst: dist.Instruction, /, **interpreter_state) -> None | int | JIT_SIGNALS
#
#   The opcode handler is called to interpret an opcode, and is an opportunity to
#   implement custom opcode handling.
#
#   If the opcode is unhandled, then JIT_SIGNALS.UNHANDLED_OPCODE should be returned.
#
#   Otherwise the function will be called with the following keyword arguments:
#
#       - stack, the interpreter stack
#       - inst_ptr, the current instruction pointer
#       - co, the code object
#       - globals_dict, the globals dictionary
#       - builtins_dict, the builtins dictionary
#       - frame: JIT frame containing local variables, source loc etc.
#       - try_stack, a stack to facilitate handling exceptions
#
#   The handler can then return None, -1 to indicate a return statement, or a weakly positive integer
#   to indicate that the interpreter should jump absolute to that instruction.
#
#   The arguments passed to the handler are very likely to change in the near future, but most
#   handlers only need to consume a small subset of the above arguments.
#
#   (2) The fn_lookaside function, which has the signature
#
#   fn_lookaside(fn, *args, **kwargs) -> tuple[bool, None | Any] | JIT_SIGNALS
#
#   The function 'lookaside' is an opportunity to intercept functions and either provide custom
#   implementations of them or raise exceptions if they are "unsafe".
#   It is called whenever a function is jitted.
#
#   If the function is unsafe, then JIT_SIGNALS.UNSAFE_FUNCTION should be returned.
#
#   Otherwise, the function should return a tuple with two values:
#
#       - The first value is a boolean that is True if a lookaside occurred, and False otherwise
#       - The second value should be the result of the lookaside, if it happened, or None
#
#   If a lookaside did not occur, then the second value in the tuple is ignored.
def jit(
    fn: Callable,
    *,
    opcode_interpreter: Callable = default_opcode_interpreter,
    fn_lookaside: Callable = default_fn_lookaside,
) -> Callable:
    compilectx: JitCompileCtx = JitCompileCtx(
        opcode_interpreter=opcode_interpreter,
        fn_lookaside=fn_lookaside,
    )

    @functools.wraps(fn)
    def fn_(*args, **kwargs) -> Any:
        runtimectx: JitRuntimeCtx = JitRuntimeCtx()

        with jitctx(compilectx, runtimectx):
            try:
                return _jit(fn, *args, **kwargs)
            except Exception as e:
                # TODO Highlight the portion of the line that originated the opcode on Python versions that include
                #   the line offset information in the instruction
                traceback_str = "\n".join(f.format_with_source() for f in runtimectx.frame_stack)
                msg = f"Encountered exception {type(e).__name__}: {e} while tracing {fn}:\n" f"{traceback_str}"
                raise JITError(msg) from e

    return fn_
