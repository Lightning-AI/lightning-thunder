from __future__ import annotations

import builtins
import contextlib
import contextvars
import dataclasses
import dis
import enum
import functools
import linecache
import inspect
import sys
from typing import Any, NamedTuple
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

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

    def lookaside(self, fn: Callable, *args, **kwargs) -> None | Callable:
        return self._fn_lookaside(fn, *args, **kwargs)


_jitcompilectx = contextvars.ContextVar("jitcompilectx")


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
@contextlib.contextmanager
def jitcompilectx(_jitcompilectx: JitCompileCtx):
    tok: Any = set_jitcompilectx(_jitcompilectx)
    try:
        yield
    finally:
        reset_jitcompilectx(tok)


# The Jit's runtime context, which tracks stack changes in Python mode
class JitRuntimeCtx:
    def __init__(self):
        self.frame_stack: list[JITFrame] = []
        self._globals_dict: dict[str, Any] | None = None
        self._history: list[dis.Instruction | str] = []
        self._interpreted_instructions: list[dis.Instruction] = []

    @property
    def globals_dict(self) -> dict[str, Any]:
        assert self._globals_dict is not None
        return self._globals_dict

    # The operations encountered while interpreting
    @property
    def interpreted_instructions(self) -> list[dis.Instruction]:
        return self._interpreted_instructions

    # The operations and opaque calls encountered while interpreting
    @property
    def history(self) -> list[dis.Instruction | str]:
        return self._history

    def peek_interpreter_stack(self) -> list:
        return self.frame_stack[-1].interpreter_stack

    # get current top of stack
    def peek_frame_stack(self) -> JITFrame | None:
        return self.frame_stack[-1] if len(self.frame_stack) != 0 else None

    def push_frame_stack(self, frame: JITFrame):
        self.frame_stack.append(frame)

    # for returning from method calls
    def pop_frame_stack(self):
        assert self.frame_stack
        del self.frame_stack[-1]

    # TODO Instead of appending to both history and and interpreted_instructions we could
    #   consider just appending to history and then filtering to only instructions when
    #   interpreted_instructions is accessed
    def record_interpreted_instruction(self, inst: dis.Instruction) -> JitRuntimeCtx:
        self._interpreted_instructions.append(inst)
        self._history.append(inst)
        return self

    def record_opaque_call(self, fn: Callable) -> JitRuntimeCtx:
        self._history.append(f"Opaque call to {fn} with name {getattr(fn, '__name__', 'None')}")
        return self

    def record_lookaside(self, fn: Callable) -> JitRuntimeCtx:
        self._history.append(f"Lookaside to {fn.__name__}")


_jitruntimectx = contextvars.ContextVar("jitruntimectx")


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
@contextlib.contextmanager
def jitruntimectx(_jitruntimectx: JitRuntimeCtx):
    tok: Any = set_jitruntimectx(_jitruntimectx)
    try:
        yield
    finally:
        reset_jitruntimectx(tok)


# A convenience helper for setting both the jit compile and runtime ctx
@contextlib.contextmanager
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


# SETUP_FINALLY is Python <= 3.10
SETUP_FINALLY_TYPE = dis.opmap.get("SETUP_FINALLY")


class PyTryBlock:
    def __init__(self, typ: int, handler: int, level: int):
        self.typ = typ
        self.handler = handler
        self.level = level


class PyErr_StackItem:
    def __init__(self, exc_type: Any, exc_value: Any, exc_traceback: Any, previous: None | PyErr_StackItem = None):
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.exc_traceback = exc_traceback
        self.previous = previous


# Use dis.Positions in 3.11+ and make it up in <3.11
if sys.version_info < (3, 11):

    class Positions(NamedTuple):
        lineno: int = None
        end_lineno: int = None
        col_offset: int = None
        end_col_offset: int = None

else:
    Positions = dis.Positions


# This is an interpreter frame, similar to Python's for use fo the JIT
# It contains all information needed to execute the current code
# so for generators (which need to suspend and resume) one only needs to
# have the JITFrame around to continue execution.
@dataclasses.dataclass
class JITFrame:
    code: CodeType
    qualname: str
    globals: dict[str, Any]
    builtins: dict[str, Any]
    positions: Positions | None = None
    inst: dis.Instruction | None = None
    call_shape_kwnames: tuple[str] | None = None  # for KW_NAMES opcode in 3.11+
    interpreter_stack: list[Any] = dataclasses.field(default_factory=list)
    try_stack: list[PyTryBlock] = dataclasses.field(default_factory=list)
    inst_ptr: int = 0

    # in Python 3.11+ the slots are not split by local/cell/free any more
    localsplus: list[Any] = dataclasses.field(default_factory=list)

    # advance to the given instruction
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
        assert self.positions is not None
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
            # _varname_from_oparg is not documented
            return self.code._varname_from_oparg(idx)  # type: ignore


#
# Handler registration
#

_default_opcode_handler_map: dict[str, Callable] = {}


def default_opcode_interpreter(inst: dis.Instruction, /, **interpreter_state) -> None | int | JIT_SIGNALS:
    handler: None | Callable = _default_opcode_handler_map.get(inst.opname, None)
    if handler is None:
        return JIT_SIGNALS.UNHANDLED_OPCODE

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
            assert self.name not in _default_opcode_handler_map
            _default_opcode_handler_map[self.name] = fn
            return fn
        return _default_opcode_handler_map.get(self.name, fn)


# Implements a less opaque function than bool() that can interpret into dunder bool and dunder len calls
def _bool_lookaside(x: Any) -> bool:
    def impl():
        # Handles objects that define __bool__
        if hasattr(x, "__bool__"):
            return getattr(x, "__bool__")()

        # Handles objects that do not define __bool__ but define __len__
        # TODO Add a len() lookaside
        if hasattr(x, "__len__"):
            return len(x) != 0

        # NOTE By default, objects evaluate to True
        return True

    return _jit(impl)


_default_lookaside_map: dict[Callable, Callable] = {
    bool: _bool_lookaside,
}


# The default function lookaside -- currently it doesn't intercept anything
def default_lookaside(fn, *args, **kwargs) -> None | Callable:
    return _default_lookaside_map.get(fn, None)


#
# Python opcode handlers (sorted alphabetically)
#


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


def _binary_op(stack: list, op: BINARY_OP, a, b):
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

    binop_name, *method_names = ops[idx]
    if len(method_names) == 2:
        left_method, right_method = method_names

        def impl():
            if (not hasattr(a, left_method)) or ((result := getattr(a, left_method)(b)) is NotImplemented):
                if (not hasattr(b, right_method)) or ((result := getattr(b, right_method)(a)) is NotImplemented):
                    err: TypeError = TypeError(
                        f"Unsupported operand type(s) for {binop_name}: '{type(a)}' and '{type(b)}'"
                    )
                    raise err

            return result

    else:
        (method,) = method_names

        def impl():
            if (not hasattr(a, method)) or ((result := getattr(a, method)(b)) is NotImplemented):
                err: TypeError = TypeError(f"Unsupported operand type(s) for {binop_name}: '{type(a)}' and '{type(b)}'")
                raise err

            return result

    stack.append(_jit(impl))


def _binary_op_helper(stack: list, op: BINARY_OP):
    b = stack.pop()
    a = stack.pop()
    return _binary_op(stack, op, a, b)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_ADD
@register_opcode_handler("BINARY_ADD", max_ver=(3, 10))
def _binary_add_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.ADD)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_AND
@register_opcode_handler("BINARY_AND", max_ver=(3, 10))
def _binary_and_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.AND)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_FLOOR_DIVIDE
@register_opcode_handler("BINARY_FLOOR_DIVIDE", max_ver=(3, 10))
def _binary_floor_divide_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.FLOORDIV)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_LSHIFT
@register_opcode_handler("BINARY_LSHIFT", max_ver=(3, 10))
def _binary_lshift_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.LSHIFT)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_MATRIX_MULTIPLY
@register_opcode_handler("BINARY_MATRIX_MULTIPLY", max_ver=(3, 10))
def _binary_matrix_multiply_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.MATMUL)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_MULTIPLY
@register_opcode_handler("BINARY_MULTIPLY", max_ver=(3, 10))
def _binary_multiply_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.MUL)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_MODULO
@register_opcode_handler("BINARY_MODULO", max_ver=(3, 10))
def _binary_modulo_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.MOD)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_OR
@register_opcode_handler("BINARY_OR", max_ver=(3, 10))
def _binary_or_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.OR)


# https://docs.python.org/3.11/library/dis.html#opcode-BINARY_OP
@register_opcode_handler("BINARY_OP", min_ver=(3, 11))
def _binary_op_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    return _binary_op_helper(stack, BINARY_OP(inst.arg))


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_POWER
@register_opcode_handler("BINARY_POWER", max_ver=(3, 10))
def _binary_power_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.POW)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_RSHIFT
@register_opcode_handler("BINARY_RSHIFT", max_ver=(3, 10))
def _binary_rshift_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.RSHIFT)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_SUBTRACT
@register_opcode_handler("BINARY_SUBTRACT", max_ver=(3, 10))
def _binary_subtract_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.SUB)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_TRUE_DIVIDE
@register_opcode_handler("BINARY_TRUE_DIVIDE", max_ver=(3, 10))
def _binary_true_divide_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.TRUEDIV)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_SUBTRACT
@register_opcode_handler("BINARY_XOR", max_ver=(3, 10))
def _binary_xor_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    return _binary_op_helper(stack, BINARY_OP.XOR)


# https://docs.python.org/3.10/library/dis.html#opcode-BINARY_SUBSCR
@register_opcode_handler("BINARY_SUBSCR")
def _binary_subscr_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    tos = stack.pop()
    tos1 = stack.pop()

    def impl():
        return tos1.__getitem__(tos)

    stack.append(_jit(impl))


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

    strings: tuple[str, ...] = tuple(reversed(tuple(stack.pop() for _ in range(count))))
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
    stack.append(_jit(func, *args, **kwargs))


# NOTE This only accepts positional args
# https://docs.python.org/3.10/library/dis.html#opcode-CALL_FUNCTION
@register_opcode_handler("CALL_FUNCTION")
def _call_function_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    argc: int = inst.arg
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(argc))))
    func: Callable = stack.pop()
    stack.append(_jit(func, *args))


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

    stack.append(_jit(func, *args, **kwargs))


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

    stack.append(_jit(func, *args, **fn_kwargs))


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

    stack.append(_jit(meth, *args))


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


# https://docs.python.org/3.11/library/dis.html#opcode-COPY
@register_opcode_handler("COPY", min_ver=(3, 11))
def _copy_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    assert inst.arg >= 1
    stack.append(stack[-inst.arg])


# https://docs.python.org/3.11/library/dis.html#opcode-COPY_FREE_VARS
@register_opcode_handler("COPY_FREE_VARS", min_ver=(3, 11))
def _copy_free_vars_handler(inst: dis.Instruction, /, **kwargs) -> None:
    # we already do this when setting up the function call in _jit
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-DICT_MERGE
@register_opcode_handler("DICT_MERGE")
def _dict_merge_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    a = stack.pop()
    b = stack[-1]
    # TODO: Raise inside interpreter
    assert isinstance(b, MutableMapping), b
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
    assert isinstance(b, MutableMapping), b
    assert isinstance(a, Mapping), a
    b.update(a)


# https://docs.python.org/3.10/library/dis.html#opcode-DUP_TOP
@register_opcode_handler("DUP_TOP")
def _dup_top_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    stack.append(stack[-1])


# https://docs.python.org/3/library/dis.html#opcode-DELETE_FAST
@register_opcode_handler("DELETE_FAST")
def _delete_fast_handler(inst: dis.Instruction, /, co: CodeType, frame: JITFrame, **kwargs) -> None:
    assert type(inst.arg) is int
    var_num: int = inst.arg
    assert var_num >= 0 and var_num < co.co_nlocals

    # NOTE The deletion just sets the reference in localsplus to an instance of Py_NULL
    frame.localsplus[var_num] = Py_NULL()


# https://docs.python.org/3.10/library/dis.html#opcode-DELETE_SUBSCR
@register_opcode_handler("DELETE_SUBSCR")
def _delete_subscr_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    tos = stack.pop()
    tos1 = stack.pop()
    del tos1[tos]


# https://docs.python.org/3.10/library/dis.html#opcode-EXTENDED_ARG
@register_opcode_handler("EXTENDED_ARG")
def _extended_arg_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    pass


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

    assert type(inst.arg) is int
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

    stack.append(_jit(impl))


# TODO
# https://docs.python.org/3.10/library/dis.html#opcode-FOR_ITER
@register_opcode_handler("FOR_ITER")
def _for_iter_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> int | None:
    assert type(inst.arg) is int
    delta: int = inst.arg

    tos: Iterator = stack[-1]
    assert isinstance(tos, Iterator)

    v: Any
    try:
        v = next(tos)
        stack.append(v)
    except StopIteration:
        stack.pop()
        return inst_ptr + delta + 1


def _iter_impl(obj):
    def impl():
        return obj.__iter__()

    return impl


# https://docs.python.org/3.10/library/dis.html#opcode-GET_ITER
@register_opcode_handler("GET_ITER")
def _get_iter_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    tos = stack.pop()
    stack.append(_jit(_iter_impl(tos)))


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

    stack.append(_jit(impl))


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

    stack.append(_jit(impl))


# https://docs.python.org/3.10/library/dis.html#opcode-IS_OP
@register_opcode_handler("IS_OP")
def _is_op_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()
    stack.append(a is not b if inst.arg == 1 else a is b)


# https://docs.python.org/3.10/library/dis.html#opcode-JUMP_ABSOLUTE
@register_opcode_handler("JUMP_ABSOLUTE")
def _jump_absolute_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    assert type(inst.arg) is int
    target: int = inst.arg
    return target


# https://docs.python.org/3.10/library/dis.html#opcode-JUMP_FORWARD
@register_opcode_handler("JUMP_FORWARD")
def _jump_forward_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    assert type(inst.arg) is int
    delta: int = inst.arg
    assert isinstance(delta, int)
    return inst_ptr + delta + 1


# https://docs.python.org/3.11/library/dis.html#opcode-JUMP_BACKWARD
@register_opcode_handler("JUMP_BACKWARD", min_ver=(3, 11))
def _jump_backward_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    delta: int = inst.arg
    assert isinstance(delta, int)
    return inst_ptr - delta + 1


# https://docs.python.org/3.11/library/dis.html#opcode-JUMP_BACKWARD
# TODO: we currently ignore the NO_INTERRUPT part,
#       https://github.com/Lightning-AI/lightning-thunder/issues/1631
@register_opcode_handler("JUMP_BACKWARD_NO_INTERRUPT", min_ver=(3, 11))
def _jump_backward_no_interrupt_handler(inst: dis.Instruction, /, inst_ptr: int, **kwargs) -> int:
    delta: int = inst.arg
    assert isinstance(delta, int)
    return inst_ptr - delta + 1


# https://docs.python.org/3.10/library/dis.html#opcode-LIST_APPEND
@register_opcode_handler("LIST_APPEND")
def _list_append_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg

    # NOTE Doesn't pop the list that's extended
    tos = stack.pop()
    l: list = stack[-i]

    assert isinstance(l, list)
    l.append(tos)


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


# https://docs.python.org/3.10/library/dis.html#opcode-LIST_TO_TUPLE
@register_opcode_handler("LIST_TO_TUPLE")
def _list_to_tuple_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    tos = stack.pop()
    assert isinstance(tos, list)
    stack.append(tuple(tos))


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_ATTR
@register_opcode_handler("LOAD_ATTR")
def _load_attr_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert type(inst.arg) is int

    a = stack.pop()
    name: str = co.co_names[inst.arg]

    def impl():
        return getattr(a, name)

    stack.append(_jit(impl))


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
        return
    val = cell.cell_contents
    stack.append(val)


# https://docs.python.org/3.10/library/dis.html#opcode-LOAD_FAST
@register_opcode_handler("LOAD_FAST")
def _load_fast_handler(inst: dis.Instruction, /, stack: list, co: CodeType, frame: JITFrame, **kwargs) -> None:
    assert isinstance(inst.arg, int)
    i: int = inst.arg
    assert i >= 0 and i < len(frame.localsplus)

    val: Any = frame.localsplus[i]

    # empty local variable slots are initialized to Py_NULL()
    if isinstance(val, Py_NULL):
        do_raise(UnboundLocalError(f"local variable '{frame.get_localsplus_name(i)}' referenced before assignment"))
        return

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


# https://docs.python.org/3.11/library/dis.html#opcode-PRECALL
@register_opcode_handler("PRECALL", min_ver=(3, 11), max_ver=(3, 11))
def _precall_handler(inst: dis.Instruction, /, co: CodeType, **kwargs) -> None:
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-POP_BLOCK
@register_opcode_handler("POP_BLOCK")
def _pop_block_handler(inst: dis.Instruction, /, try_stack: list[PyTryBlock], **kwargs) -> None:
    try_stack.pop()


# https://docs.python.org/3.10/library/dis.html#opcode-POP_EXCEPT
@register_opcode_handler("POP_EXCEPT")
def _pop_except_handler(inst: dis.Instruction, /, try_stack: list[PyTryBlock], **kwargs) -> None:
    try_stack.pop()


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_FALSE
@register_opcode_handler("POP_JUMP_FORWARD_IF_FALSE", min_ver=(3, 11))
def _pop_jump_forward_if_false_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()

    def impl():
        return bool(tos)

    # Acquires the condition, only calling bool() if tos requires conversion
    # NOTE Unconditionally calling bool() would cause an infinite recursion with the bool lookaside
    cnd: bool
    if tos is False or tos is True:
        cnd = tos
    else:
        cnd = _jit(impl)

    if not cnd:
        return inst_ptr + inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_TRUE
@register_opcode_handler("POP_JUMP_FORWARD_IF_TRUE", min_ver=(3, 11))
def _pop_jump_forward_if_true_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()

    def impl():
        return bool(tos)

    # Acquires the condition, only calling bool() if tos requires conversion
    # NOTE Unconditionally calling bool() would cause an infinite recursion with the bool lookaside
    cnd: bool
    if tos is False or tos is True:
        cnd = tos
    else:
        cnd = _jit(impl)

    if cnd:
        return inst_ptr + inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_NONE
@register_opcode_handler("POP_JUMP_FORWARD_IF_NONE", min_ver=(3, 11))
def _pop_jump_forward_if_none_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()
    if tos is None:
        return inst_ptr + inst.arg + 1

    return None


# https://docs.python.org/3.11/library/dis.html#opcode-POP_JUMP_FORWARD_IF_NOT_NONE
@register_opcode_handler("POP_JUMP_FORWARD_IF_NOT_NONE", min_ver=(3, 11))
def _pop_jump_forward_if_none_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> int | None:
    assert isinstance(inst.arg, int)

    tos = stack.pop()
    if tos is not None:
        return inst_ptr + inst.arg + 1

    return None


@register_opcode_handler("POP_JUMP_IF_FALSE")
def _pop_jump_if_false_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> int | None:
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
        cnd = _jit(impl)

    if not cnd:
        return inst.arg
    return None


@register_opcode_handler("POP_JUMP_IF_TRUE")
def _pop_jump_if_true_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> int | None:
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
        cnd = _jit(impl)

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


# https://docs.python.org/3.11/library/dis.html#opcode-PUSH_NULL
@register_opcode_handler("PUSH_NULL", min_ver=(3, 11))
def _push_null_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    stack.append(Py_NULL())


# https://docs.python.org/3.10/library/dis.html#opcode-RAISE_VARARGS
@register_opcode_handler("RAISE_VARARGS")
def _raise_varargs_handler(inst: dis.Instruction, /, stack: list, try_stack: list[PyTryBlock], **kwargs) -> None:
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
def _reraise_handler(inst: dis.Instruction, /, stack: list, try_stack: list[PyTryBlock], **kwargs) -> None:
    pass


# https://docs.python.org/3.10/library/dis.html#opcode-RETURN_VALUE
@register_opcode_handler("RETURN_VALUE")
def _return_value_handler(inst: dis.Instruction, /, **kwargs) -> int | None:
    return JIT_SIGNALS.RETURN_VALUE


# https://docs.python.org/3.10/library/dis.html#opcode-ROT_N
@register_opcode_handler("ROT_N")
def _rot_n_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    assert len(stack) >= inst.arg
    stack[-inst.arg :] = (stack[-1], *stack[-inst.arg : -1])


# https://docs.python.org/3.10/library/dis.html#opcode-ROT_TWO
@register_opcode_handler("ROT_TWO")
def _rot_two_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    top = stack[-1]
    second = stack[-2]

    stack[-1] = second
    stack[-2] = top


# https://docs.python.org/3.10/library/dis.html#opcode-SETUP_FINALLY
@register_opcode_handler("SETUP_FINALLY")
def _setup_finally_handler(inst: dis.Instruction, stack: list, try_stack: list[PyTryBlock], **kwargs) -> None:
    assert inst.arg is not None
    instr_offset = stack.index(inst) + inst.arg
    try_stack.append(PyTryBlock(SETUP_FINALLY_TYPE, instr_offset, 0))


# https://docs.python.org/3.10/library/dis.html#opcode-SETUP_WITH
@register_opcode_handler("SETUP_WITH")
def _setup_with_handler(inst: dis.Instruction, /, try_stack: list[PyTryBlock], **kwargs) -> None:
    assert type(inst.arg) is int
    try_stack.append(PyTryBlock())


# https://docs.python.org/3.11/library/dis.html#opcode-SWAP
@register_opcode_handler("SWAP", min_ver=(3, 11))
def _swap_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    i = inst.arg
    assert isinstance(i, int)
    assert 0 < i <= len(stack)

    top = stack[-1]
    other = stack[-i]

    stack[-1] = other
    stack[-i] = top


# https://docs.python.org/3.10/library/dis.html#opcode-STORE_ATTR
@register_opcode_handler("STORE_ATTR")
def _store_attr_handler(inst: dis.Instruction, /, stack: list, co: CodeType, **kwargs) -> None:
    assert type(inst.arg) is int
    namei: int = inst.arg

    name: str = co.co_names[namei]

    tos: Any = stack.pop()
    tos1: Any = stack.pop()

    def impl():
        setattr(tos, name, tos1)

    _jit(impl)


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


# https://docs.python.org/3.10/library/dis.html#opcode-UNARY_NOT
@register_opcode_handler("UNARY_NOT")
def _unary_not_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    tos = stack.pop()

    def impl():
        if bool(tos):
            return False
        return True

    stack.append(_jit(impl))


# https://docs.python.org/id/3.5/library/dis.html#opcode-UNPACK_SEQUENCE
# https://docs.python.org/3.10/library/dis.html#opcode-UNARY_NEGATIVE
@register_opcode_handler("UNARY_NEGATIVE")
def _unary_negative_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    tos = stack[-1]
    stack[-1] = -tos


# https://docs.python.org/3.10/library/dis.html#opcode-UNPACK_EX
@register_opcode_handler("UNPACK_EX")
def _unpack_ex_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert type(inst.arg) is int
    counts: int = inst.arg
    before_list: int = counts & 0xFF
    after_list: int = counts >> 8

    seq: Iterable = stack.pop()
    assert isinstance(seq, Iterable)

    def impl():
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

    results = _jit(impl)

    for x in reversed(results):
        stack.append(x)


# https://docs.python.org/3.10/library/dis.html#opcode-UNPACK_SEQUENCE
@register_opcode_handler("UNPACK_SEQUENCE")
def _unpack_sequence_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    seq: Sequence = stack.pop()

    for x in reversed(seq):
        stack.append(x)


# Generator handling
# https://docs.python.org/3.10/library/dis.html#opcode-GEN_START
@register_opcode_handler("GEN_START", max_ver=(3, 10))
def _gen_start_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    assert 0 <= inst.arg < 3  # yeah, we are not doing anything with it
    stack.pop()  # this should be None (sent to the generator), but Python does not check


# https://docs.python.org/3.10/library/dis.html#opcode-GET_YIELD_FROM_ITER
@register_opcode_handler("GET_YIELD_FROM_ITER")
def _get_yield_from_iter_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    tos = stack[-1]
    if not (inspect.isgenerator(tos) or inspect.iscoroutine(tos)):
        stack.append(_jit(_iter_impl(stack.pop())))


# https://docs.python.org/3.11/library/dis.html#opcode-RETURN_GENERATOR
@register_opcode_handler("RETURN_GENERATOR", min_ver=(3, 11))
def _return_generator_handler(inst: dis.Instruction, /, **kwargs) -> None:
    return JIT_SIGNALS.RETURN_GENERATOR


# https://docs.python.org/3.11/library/dis.html#opcode-SEND
@register_opcode_handler("SEND", min_ver=(3, 11))
def _send_handler(inst: dis.Instruction, /, stack: list, inst_ptr: int, **kwargs) -> None:
    # SEND(delta)
    # Equivalent to STACK[-1] = STACK[-2].send(STACK[-1]). Used in yield from and await statements.
    # If the call raises StopIteration, pop the top value from the stack, push the exception’s value attribute, and increment the bytecode counter by delta.
    assert isinstance(inst.arg, int)
    send_value = stack.pop()
    generator = stack[-1]

    if send_value is None:
        # iterators don't have a .send method
        def impl():
            return generator.__next__()

    else:

        def impl():
            return generator.send(send_value)

    try:
        res = _jit(impl)
    except StopIteration as si:
        stack.pop()  # remove generator
        stack.append(si.value)
        return inst_ptr + inst.arg + 1

    stack.append(res)


# https://docs.python.org/3.10/library/dis.html#opcode-YIELD_FROM
@register_opcode_handler("YIELD_FROM", max_ver=(3, 10))
def _yield_from_handler(inst: dis.Instruction, /, stack: list, frame: JITFrame, inst_ptr: int, **kwargs) -> None:
    send_value = stack.pop()
    generator = stack[-1]

    if send_value is None:
        # iterators don't have a .send method
        def impl():
            return generator.__next__()

    else:

        def impl():
            return generator.send(send_value)

    try:
        res = _jit(impl)
    except StopIteration as si:
        stack.pop()  # remove generator
        stack.append(si.value)
        return

    # this is a gross hack, this will be incremented so the inst_ptr is at this YIELD_FROM again
    # cleaner would be to introduce another JIT_SIGNALS
    frame.inst_ptr -= 1
    # this will be yielded
    stack.append(res)
    return JIT_SIGNALS.YIELD_VALUE


# https://docs.python.org/3.10/library/dis.html#opcode-YIELD_VALUE
@register_opcode_handler("YIELD_VALUE")
def _yield_value_handler(inst: dis.Instruction, /, stack: list, **kwargs) -> None:
    # note that the popping from the stack is done in the _jit_run loop
    return JIT_SIGNALS.YIELD_VALUE


def make_generator(
    frame: JITFrame,
    compilectx: JitCompileCtx,
    runtimectx: JitRuntimeCtx,
):
    def thunder_jit_generator():
        send_value: Any = None  # the value gotten from from <generator>.send
        while True:  # or maybe have return?
            with jitctx(compilectx, runtimectx):
                res, status = _jit_run(frame, compilectx, runtimectx, send_value=send_value)
            if status == JIT_SIGNALS.RETURN_VALUE:
                return
            assert status == JIT_SIGNALS.YIELD_VALUE
            sent_value = yield res

    return thunder_jit_generator()


# Interprets the callable with the given args and kwargs
# NOTE There are (currently) 6 cases for interpretation:
#
#   (1) The callable has a lookaside, in which case it's used to execute the operation
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
    lookaside_fn: None | Callable = compilectx.lookaside(fn, *args, **kwargs)
    if lookaside_fn:
        runtimectx.record_lookaside(lookaside_fn)
        return lookaside_fn(*args, **kwargs)

    # (2) Handles opaque functions
    if is_opaque(fn):
        opaque_result: Any = fn(*args, **kwargs)
        runtimectx.record_opaque_call(fn)
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
    # adjustments for "hidden" instructiions (EXTENDED_ARGS, CACHE, ...)
    bound = inspect.signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    locals_dict: dict[str, Any] = dict(bound.arguments)

    code: CodeType = extract_code(fn)

    # Comprehensions:
    #   CPython prefixes protected variables with a period. (Since it's not
    #   legal syntax, so they cannot collide with user variable names.) However,
    #   `inspect` helpfully renames them. We need to undo this to properly reflect
    #   what is in the actual bytecode.
    for name in code.co_varnames:
        if name.startswith("."):
            locals_dict[name] = locals_dict.pop(f"implicit{name[1:]}")

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

    # Creates the current ready to run stack frame for the current function
    frame = JITFrame(
        code=code, localsplus=localsplus, globals=fn.__globals__, builtins=builtins_dict, qualname=fn.__qualname__
    )

    # Python 3.10 deals with creating the generator on call,
    # 3.11+ use the RETURN_GENERATOR opcode
    if sys.version_info < (3, 11):
        if code.co_flags & inspect.CO_GENERATOR:
            return make_generator(frame, compilectx, runtimectx)

    res, status = _jit_run(frame, compilectx, runtimectx)
    return res


def _jit_run(
    frame: JITFrame,
    compilectx: JitCompileCtx,
    runtimectx: JitRuntimeCtx,
    *,
    send_value: Any = Py_NULL(),
):
    # Pushes the current stack frame for the current function
    runtimectx.push_frame_stack(frame)
    stack: list = frame.interpreter_stack
    if send_value != Py_NULL():
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
        runtimectx.record_interpreted_instruction(inst)
        # Updates the stack frame to the current position
        # TODO maybe also have inst_ptr?
        frame.nexti(inst)
        stack_size_before_handler: int = len(stack)
        interpretation_result: None | int | JIT_SIGNALS = compilectx.interpret(
            inst,
            inst_ptr=frame.inst_ptr,
            stack=frame.interpreter_stack,
            globals_dict=frame.globals,
            builtins_dict=frame.builtins,
            try_stack=frame.try_stack,
            co=frame.code,
            frame=frame,
        )

        # TODO Improve this error message
        if interpretation_result is JIT_SIGNALS.UNHANDLED_OPCODE:
            raise NotImplementedError(f"Encountered unimplemented opcode {inst.opname} while tracing.\n")

        elif interpretation_result in (JIT_SIGNALS.RETURN_VALUE, JIT_SIGNALS.YIELD_VALUE):
            # advance the inst_ptr, needed in particular for YIELD
            frame.inst_ptr += 1
            # get rhe result from the current stack
            result = frame.interpreter_stack.pop()
            # Restores the previous stack, the caller needs to put the value on it
            runtimectx.pop_frame_stack()
            return result, interpretation_result
        elif interpretation_result is JIT_SIGNALS.RETURN_GENERATOR:
            frame.inst_ptr += 1
            assert frame.code.co_flags & inspect.CO_GENERATOR
            runtimectx.pop_frame_stack()
            return make_generator(frame, compilectx, runtimectx), interpretation_result
        elif interpretation_result is JIT_SIGNALS.EXCEPTION_RAISED:
            raise Unimplemented("exception handling")
        elif interpretation_result is None:
            frame.inst_ptr += 1
        else:
            assert isinstance(interpretation_result, int)
            frame.inst_ptr = interpretation_result

        # Verifies the handler had the expected stack effect (delta on stack sie)
        actual_stack_effect: int = len(stack) - stack_size_before_handler
        jumped: bool = isinstance(interpretation_result, int) and interpretation_result != -1
        expected_stack_effect: int = dis.stack_effect(inst.opcode, inst.arg, jump=jumped)

        if (3, 11) <= sys.version_info < (3, 12):
            # PRECALL stack effect (3.11) has a -inst.arg stack effect in the function that we only see during CALL
            if inst.opname == "PRECALL":
                assert type(inst.arg) is int
                assert expected_stack_effect == -inst.arg, f"precall with stack effect {expected_stack_effect}, {inst}"
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
class JIT_SIGNALS(enum.Enum):
    UNHANDLED_OPCODE = enum.auto()
    UNSAFE_FUNCTION = enum.auto()
    RETURN_VALUE = enum.auto()
    RETURN_GENERATOR = enum.auto()
    YIELD_VALUE = enum.auto()
    EXCEPTION_RAISED = enum.auto()


#
# Defines jit ux
#


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
#   fn_lookaside(fn, *args, **kwargs) -> None | Callable
#
#   The function 'lookaside' is an opportunity to intercept functions and either provide custom
#   implementations of them or raise exceptions if they are "unsafe".
#   It is called whenever a function is jitted.
#
#   If there is no lookaside, then None should be returned.
#
#   If the function is unsafe, then a callable that raises UnsafeOperator (to be implemented)
#       should be returned.
#
#   Otherwise, the function should implement the lookaside when called with the same args and kwargs.
def jit(
    fn: Callable,
    *,
    opcode_interpreter: Callable = default_opcode_interpreter,
    fn_lookaside: Callable = default_lookaside,
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
                jit_result: Any = _jit(fn, *args, **kwargs)
                fn_._last_interpreted_instructions = runtimectx.interpreted_instructions
                fn_._last_history = runtimectx.history
                return jit_result
            except Exception as e:
                # TODO Highlight the portion of the line that originated the opcode on Python versions that include
                #   the line offset information in the instruction
                traceback_str = "\n".join(f.format_with_source() for f in runtimectx.frame_stack)
                msg = f"Encountered exception {type(e).__name__}: {e} while tracing {fn}:\n" f"{traceback_str}"
                raise JITError(msg) from e

    return fn_


def last_interpreted_instructions(fn: Callable) -> None | list[dis.Instruction]:
    return getattr(fn, "_last_interpreted_instructions", None)


def last_history(fn: Callable) -> None | list[dis.Instruction | str]:
    return getattr(fn, "_last_history", None)
