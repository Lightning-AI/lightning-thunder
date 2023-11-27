from __future__ import annotations

import dis
import sys
import collections
from collections.abc import Iterator, Sequence, Callable
from dataclasses import dataclass
import inspect
import linecache
from typing import Any
from types import CellType, CodeType, FunctionType, MethodType, BuiltinFunctionType
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

    # for method calls. There is a bit of a trick because the filename is
    # part of the code object, but not the instruction's position information.
    def push_frame_stack(self, code: CodeType):
        self.frame_stack.append(JITFrame(code=code))

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

# Acquires the method-wrapper type indirectly by getting the type of a known
#   method-wrapper
# TODO Extend to "slot wrapper"?
# TODO What is the best way to test for method-wrapper?
# NOTE method-wrapper is a CPython implementation detail. It represents
#   an opaque method, and it has a __self__ attribute populated with
#   its first argument
_a: int = 2
MethodWrapperType: type = type(_a.__add__)


def is_opaque(fn: Callable) -> bool:
    if isinstance(fn, (BuiltinFunctionType, MethodWrapperType)):
        return True

    # NOTE builtins.type has type type, but type() is an opaque function
    if fn is type:
        return True

    return False


# Acquires the code object from a function or method (converting methods to functions)
# TODO FIXME This won't acquire the code object for all callables...
#   ... what about partial objects, callable classes, functools.wraps, @contextmanager, etc.
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
    positions: Positions | None = None
    inst: dis.Instruction | None = None

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
        l.append(f"  in {self.code.co_name} in file: {self.code.co_filename}, line {self.positions.lineno}:")
        if self.code.co_filename:
            ls = linecache.getlines(self.code.co_filename)
            l.append("  " + ls[max(self.positions.lineno - 1, 0)].rstrip())
        return "\n".join(l)


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
    def __init__(self, name: str):
        self.name: str = name

    def __call__(self, fn: Callable) -> Callable:
        _default_opcode_handler_map[self.name] = fn
        return fn


# The default function lookaside -- currently it doesn't intercept anything
def default_fn_lookaside(fn, *args, **kwargs) -> tuple[bool, None | Any] | JIT_SIGNALS:
    return (False, None)


#
# Python opcode handlers (sorted alphabetically)
#


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1529
# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1530
@register_opcode_handler("BINARY_ADD")
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


@register_opcode_handler("BINARY_MULTIPLY")
def _binary_multiply_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()

    def impl():
        if (not hasattr(a, "__mul__")) or ((result := a.__mul__(b)) is NotImplemented):
            if (not hasattr(b, "__rmul__")) or ((result := b.__rmul__(a)) is NotImplemented):
                # TODO Restore formatting once FORMAT_VALUE is implemented
                # raise TypeError(f"Unsupported operand type(s) for +: '{type(a)}' and '{type(b)}'")
                err: TypeError = TypeError("Unsupported operand types for binary add")
                raise err

        return result

    _jit(impl)


@register_opcode_handler("BINARY_SUBTRACT")
def _binary_subtract_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()

    def impl():
        if (not hasattr(a, "__sub__")) or ((result := a.__sub__(b)) is NotImplemented):
            if (not hasattr(b, "__rsub__")) or ((result := b.__rsub__(a)) is NotImplemented):
                # TODO Restore formatting once FORMAT_VALUE is implemented
                # raise TypeError(f"Unsupported operand type(s) for +: '{type(a)}' and '{type(b)}'")
                err: TypeError = TypeError("Unsupported operand types for binary add")
                raise err

        return result

    _jit(impl)


@register_opcode_handler("BUILD_MAP")
def _build_map_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert inst.arg is not None
    count: int = inst.arg

    # NOTE The reversed() call below is necessary to handle key collisions properly
    d: dict = {k: v for v, k in reversed(tuple((stack.pop(), stack.pop()) for _ in range(count)))}
    stack.append(d)


@register_opcode_handler("BUILD_TUPLE")
def _build_tuple_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert inst.arg is not None
    result: tuple[Any, ...] = tuple(reversed([stack.pop() for _ in range(inst.arg)]))
    stack.append(result)


# TODO Test how this handles functions, callable classes, and lambdas
#   Functions
#   Callable classes
#   Lambdas
# NOTE This only accepts positional args
@register_opcode_handler("CALL_FUNCTION")
def _call_function_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert inst.arg is not None
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(inst.arg))))
    func: Callable = stack.pop()
    _jit(func, *args)


# https://docs.python.org/id/3.5/library/dis.html#opcode-CALL_FUNCTION_KW
@register_opcode_handler("CALL_FUNCTION_KW")
def _call_function_kw_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    kw_names: tuple[str, ...] = stack.pop()
    kwarg_length: int = len(kw_names)
    kwargs_flat: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(kwarg_length))))
    fn_kwargs: dict[str, Any] = {k: v for k, v in zip(kw_names, kwargs_flat)}
    assert inst.arg is not None
    arg_length: int = inst.arg - kwarg_length
    args = tuple(reversed(tuple(stack.pop() for _ in range(arg_length))))
    func: Callable = stack.pop()

    _jit(func, *args, **fn_kwargs)


# https://docs.python.org/3.10/library/dis.html#opcode-CALL_FUNCTION_EX
# @register_opcode_handler("CALL_FUNCTION_EX")
# def _call_function_ex_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
#     pass


@register_opcode_handler("CALL_METHOD")
def _call_method_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert inst.arg is not None
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
    assert inst.arg is not None
    assert inst.arg < len(dis.cmp_op), f"{inst}, {dis.cmp_op}"
    result: Any = cmp_impls[dis.cmp_op[inst.arg]](a, b)
    stack.append(result)


# https://docs.python.org/3.10/library/dis.html#opcode-DICT_MERGE
# @register_opcode_handler("DICT_MERGE")
# def _dict_merge_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
#     pass


# https://docs.python.org/3.10/library/dis.html#opcode-DUP_TOP
@register_opcode_handler("DUP_TOP")
def _dup_top_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    stack.append(stack[-1])


@register_opcode_handler("GET_LEN")
def _get_len_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    a = stack.pop()
    stack.append(len(a))


@register_opcode_handler("IS_OP")
def _is_op_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()
    stack.append(a is not b if inst.arg == 1 else a is b)


# https://docs.python.org/3.10/library/dis.html?highlight=dis#opcode-LOAD_ATTR
@register_opcode_handler("LOAD_ATTR")
def _load_attr_handler(
    inst: dis.Instruction, /, stack: list, locals_dict: dict[str, Any], co: CodeType, **kwargs
) -> None:
    assert inst.arg is not None

    a = stack.pop()
    name: str = co.co_names[inst.arg]

    def impl():
        return getattr(a, name)

    _jit(impl)


@register_opcode_handler("LOAD_CLOSURE")
def _load_closure_handler(
    inst: dis.Instruction, /, stack: list, locals_dict: dict[str, Any], co: CodeType, **kwargs
) -> None:
    assert inst.arg is not None
    var_name: str = co.co_cellvars[inst.arg]
    actual: Any = locals_dict[var_name]
    stack.append(actual)


@register_opcode_handler("LOAD_CONST")
def _load_const_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert inst.arg is not None
    stack.append(co.co_consts[inst.arg])


@register_opcode_handler("LOAD_DEREF")
def _load_deref_handler(inst: dis.Instruction, /, stack: list, closures: tuple[CellType, ...], **kwargs) -> None:
    assert inst.arg is not None
    stack.append(closures[inst.arg].cell_contents)


@register_opcode_handler("LOAD_FAST")
def _load_fast_handler(
    inst: dis.Instruction, /, stack: list, locals_dict: dict[str, Any], co: CodeType, **kwargs
) -> None:
    assert inst.arg is not None
    name: str = co.co_varnames[inst.arg]
    actual: Any = locals_dict[name]
    stack.append(actual)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1524
# https://docs.python.org/3.13/library/dis.html#opcode-LOAD_GLOBAL
# NOTE This version is the 3.10 handler
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
    assert inst.arg is not None
    co_name: str = co.co_names[inst.arg]
    try:
        obj = globals_dict[co_name]
    except KeyError:
        try:
            obj = builtins_dict[co_name]
        except KeyError as e:
            # TODO: UndefVariableError
            raise e
    stack.append(obj)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1525
@register_opcode_handler("LOAD_METHOD")
def _load_method_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert inst.arg is not None
    name = co.co_names[inst.arg]
    obj = stack.pop()
    try:
        meth = getattr(obj, name)
    except AttributeError as e:
        raise e

    if isinstance(meth, MethodType):
        stack.append(meth)
        stack.append(obj)
    else:
        stack.append(Py_NULL())
        stack.append(meth)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1526
# https://docs.python.org/id/3.5/library/dis.html#opcode-MAKE_FUNCTION
@register_opcode_handler("MAKE_FUNCTION")
def _make_function_handler(inst: dis.Instruction, /, stack: list, locals_dict: dict[str, Any], **kwargs) -> None:
    name: str = stack.pop()
    fn_co: CodeType = stack.pop()

    if inst.arg != 0 and inst.arg != 0x08:
        raise NotImplementedError("Annotations on functions compiled inline are not yet supported")

    if inst.arg & 0x08:
        closure = tuple(CellType(v) for v in stack.pop())
    else:
        closure = None

    fn = FunctionType(fn_co, locals_dict, name, closure=closure)
    stack.append(fn)


@register_opcode_handler("NOP")
def _nop_handler(inst: dis.Instruction, /, **kwargs) -> None:
    pass


@register_opcode_handler("POP_BLOCK")
def _pop_block_handler(inst: dis.Instruction, /, try_stack: list[TryBlock], **kwargs) -> None:
    try_stack.pop()


@register_opcode_handler("POP_EXCEPT")
def _pop_except_handler(inst: dis.Instruction, /, try_stack: list[TryBlock], **kwargs) -> None:
    try_stack.pop()


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1527
@register_opcode_handler("POP_JUMP_IF_FALSE")
def _pop_jump_if_false_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> int | None:
    a = stack.pop()
    cnd = bool(a)
    assert inst.arg is not None

    if cnd is False:
        return inst.arg
    return None


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1527
@register_opcode_handler("POP_JUMP_IF_TRUE")
def _pop_jump_if_true_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> int | None:
    a = stack.pop()
    cnd = bool(a)
    assert inst.arg is not None

    if cnd is True:
        return inst.arg
    return None


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


@register_opcode_handler("RAISE_VARARGS")
def _raise_varargs_handler(inst: dis.Instruction, /, stack: list, try_stack: list[TryBlock], **kwargs) -> None:
    cause: Any = Py_NULL()
    exc: Any = Py_NULL()
    assert inst.arg is not None
    if inst.arg == 2:
        cause = stack.pop()
    elif inst.arg == 1:
        exc = stack.pop()
    else:
        assert inst.arg == 0
    do_raise(exc, cause)


@register_opcode_handler("RERAISE")
def _reraise_handler(inst: dis.Instruction, /, stack: list, try_stack: list[TryBlock], **kwargs) -> None:
    pass


@register_opcode_handler("RETURN_VALUE")
def _return_value_handler(inst: dis.Instruction, /, **kwargs) -> int | None:
    return -1


@register_opcode_handler("ROT_N")
def _rot_n_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert inst.arg is not None
    assert len(stack) >= inst.arg
    stack[-inst.arg :] = (stack[-1], *stack[-inst.arg : -1])


@register_opcode_handler("ROT_TWO")
def _rot_two_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    a = stack.pop()
    b = stack.pop()
    c = stack.pop()
    stack.append(a)
    stack.append(c)
    stack.append(b)


@register_opcode_handler("SETUP_WITH")
def _setup_with_handler(inst: dis.Instruction, /, try_stack: list[TryBlock], **kwargs) -> None:
    assert inst.arg is not None
    try_stack.append(TryBlock())


# https://docs.python.org/3/library/dis.html#opcode-STORE_FAST
@register_opcode_handler("STORE_FAST")
def _store_fast_handler(
    inst: dis.Instruction, /, stack: list, locals_dict: dict[str, Any], co: CodeType, **kwargs
) -> None:
    a = stack.pop()
    assert inst.arg is not None
    var_name: str = co.co_varnames[inst.arg]
    locals_dict[var_name] = a


# https://docs.python.org/id/3.5/library/dis.html#opcode-UNPACK_SEQUENCE
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
        return _jit(fn.__call__, *((fn.__call__.__self__,) + args), **kwargs)

    # (5) Handles methods
    if isinstance(fn, MethodType):
        return _jit(fn.__func__, *args, **kwargs)

    assert isinstance(fn, FunctionType), f"{fn=} had an unexpected type ({type(fn)}"

    # (6) Jits into the function
    insts: tuple[dis.Instruction, ...] = tuple(dis.get_instructions(fn))
    locals_dict: dict[str, Any] = dict(inspect.signature(fn).bind(*args, **kwargs).arguments)
    globals_dict: dict[str, Any] = fn.__globals__
    closures = fn.__closure__
    try_stack: list[TryBlock] = []
    stack: list = runtimectx.push_interpreter_stack()

    code: CodeType = extract_code(fn)

    # Pushes a stack frame for the current function
    runtimectx.push_frame_stack(code)

    inst_ptr: int = 0
    while True:
        inst: dis.Instruction = insts[inst_ptr]

        # Updates the stack frame to the current position
        # TODO maybe also have inst_ptr?
        runtimectx.frame_stack_change_top(inst)
        stack_size_before_handler: int = len(stack)

        interpretation_result: None | int | JIT_SIGNALS = compilectx.interpret(
            inst,
            inst_ptr=inst_ptr,
            stack=stack,
            locals_dict=locals_dict,
            globals_dict=globals_dict,
            builtins_dict=builtins_dict,
            try_stack=try_stack,
            closures=closures,
            co=code,
        )

        # TODO Improve this error message
        if interpretation_result is JIT_SIGNALS.UNHANDLED_OPCODE:
            raise NotImplementedError(f"Encountered unimplemented opcode {inst.opname} while tracing.\n")

        if interpretation_result == -1:
            # Restores the previous stack and puts the returned value onto it
            result: Any = stack.pop()
            runtimectx.pop_interpreter_stack()
            runtimectx.peek_interpreter_stack().append(result)
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
        assert (
            actual_stack_effect == expected_stack_effect
        ), f"Unexpected stack effect from {inst.opname}: expected {expected_stack_effect}, but the actual effect was {actual_stack_effect}"


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
#       - locals_dict, the locals dictionary
#       - globals_dict, the globals dictionary
#       - builtins_dict, the builtins dictionary
#       - closures, the closures object
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

                # TODO Enable this version when CALL_FUNCTION_EX is implemented
                # return _jit(lambda: fn(*args, **kwargs))
            except Exception as e:
                # TODO Highlight the portion of the line that originated the opcode on Python versions that include
                #   the line offset information in the instruction
                traceback_str = "\n".join(f.format_with_source() for f in runtimectx.frame_stack)
                msg = f"Encountered exception {type(e).__name__}: {e} while tracing {fn}:\n" f"{traceback_str}"
                raise JITError(msg) from e

    return fn_
