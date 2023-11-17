import dis
from collections.abc import Iterator, Sequence, Callable
from dataclasses import dataclass
import inspect
from typing import Any
from types import CellType, CodeType, FunctionType, MethodType
import functools
from functools import partial
from enum import Enum, auto
from numbers import Number

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
    "pjit",
]

#
# Jit globals
#

# Maps function names to their overrides that should be called instead of the original function
#   For example, PyTorch operations are renamed
_function_override_map: dict[str | Callable, Callable] = {}
_function_override_map.update(_torch_to_thunder_function_map)


def _fn_lookaside(fn: Callable, *args, **kwargs) -> tuple[bool, Any]:
    fn_: None | Callable = _function_override_map.get(fn, None)

    if fn_ is None:
        return False, None

    return True, fn_(*args, **kwargs)


#
# Helpers
#


# TODO Give this class a better name
# TODO Add more provenance information
# TODO Handle number proxies correctly
# TODO Handle aliasing correctly -- do we want tensors that are the same
#   to be turned into one proxy, or two, based on the function's parameters?
# Holds Proxy context to facilitate jitting Python
class JitProxyMap:
    def __init__(self):
        self.proxy_map: dict[int, Proxy] = {}

    def proxify(self, x: Any, *, name: str | None = None) -> Any:
        # Returns existing proxies, if available
        p: None | Proxy
        p = self.proxy_map.get(id(x), None)

        if p is not None:
            return p

        # Creates a proxy
        if isinstance(x, torch.Tensor):
            p = ltorch.tensorproxy(name, x)
        elif isinstance(x, Number):
            raise NotImplementedError("Support for number proxies is not yet implemented")
        else:
            p = Proxy(name=name)

        self.proxy_map[id(x)] = p
        return p


from typing import Literal


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


class JitMode(Enum):
    PYTHON = auto()
    THUNDER = auto()


#
# Per-Jit globals
#
# TODO There may be a better way to represent these -- like with context vars, or passing them directly

prologue_trace: TraceCtx
computation_trace: TraceCtx
jpm: JitProxyMap
jit_mode: JitMode

#
# Handler registration
#

_inst_to_python_opcode_handler_map: dict[str, Callable] = {}
_inst_to_thunder_opcode_handler_map: dict[str, Callable] = {}
_mode_to_opcode_map_map: dict = {
    JitMode.PYTHON: _inst_to_python_opcode_handler_map,
    JitMode.THUNDER: _inst_to_thunder_opcode_handler_map,
}


def _get_handler(opname: str) -> None | Callable:
    return _mode_to_opcode_map_map[jit_mode].get(opname, None)


# TODO Extend with the ability to indicate supported Python versions
class register_opcode_handler:
    def __init__(self, name: str, supported_modes: Sequence[JitMode]):
        self.name: str = name
        self.supported_modes = supported_modes

    def __call__(self, fn: Callable) -> Callable:
        mode: JitMode
        for mode in self.supported_modes:
            _mode_to_opcode_map_map[mode][self.name] = fn

        return fn


@register_opcode_handler("POP_BLOCK", supported_modes=[JitMode.PYTHON])
def _pop_block_handler(try_stack: list[TryBlock], **kwargs) -> None:
    try_stack.pop()


@register_opcode_handler("POP_EXCEPT", supported_modes=[JitMode.PYTHON])
def _pop_except_handler(try_stack: list[TryBlock], **kwargs) -> None:
    try_stack.pop()


@register_opcode_handler("SETUP_WITH", supported_modes=[JitMode.PYTHON])
def _setup_with_handler(try_stack: list[TryBlock], inst: dis.Instruction, **kwargs) -> None:
    assert inst.arg is not None
    try_stack.append(TryBlock())


@register_opcode_handler("RERAISE", supported_modes=[JitMode.PYTHON])
def _reraise_handler(stack: list, try_stack: list[TryBlock], **kwargs) -> None:
    pass


@register_opcode_handler("RAISE_VARARGS", supported_modes=[JitMode.PYTHON])
def _raise_varargs_handler(inst: dis.Instruction, stack: list, try_stack: list[TryBlock], **kwargs) -> None:
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
        _value = do_call(exc)
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
            fixed_cause = do_call(cause)
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


def do_call(fn: Callable, *args, **kwargs):
    return fn(*args, **kwargs)


# TODO Handle UndefVariableError
# https://docs.python.org/3.13/library/dis.html#opcode-LOAD_GLOBAL
# NOTE This version is the 3.10 handler
@register_opcode_handler("LOAD_GLOBAL", supported_modes=[JitMode.PYTHON])
def _load_global_handler(
    stack: list,
    inst: dis.Instruction,
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


@register_opcode_handler("LOAD_CONST", supported_modes=[JitMode.PYTHON])
def _load_const_handler(stack: list, inst: dis.Instruction, co: CodeType, **kwargs) -> None:
    assert inst.arg is not None
    stack.append(co.co_consts[inst.arg])


@register_opcode_handler("LOAD_FAST", supported_modes=[JitMode.PYTHON, JitMode.THUNDER])
def _load_fast_handler(stack: list, locals_dict: dict[str, Any], inst: dis.Instruction, co: CodeType, **kwargs) -> None:
    assert inst.arg is not None
    name: str = co.co_varnames[inst.arg]
    actual: Any = locals_dict[name]

    # Tells the Thunder logic about this name
    if jit_mode is JitMode.THUNDER:
        # NOTE The choice of trace doesn't matter here, since both traces use the same names
        with tracectx(computation_trace):
            jpm.proxify(actual, name=name)

    stack.append(actual)


@register_opcode_handler("LOAD_CLOSURE", supported_modes=[JitMode.PYTHON])
def _load_closure_handler(
    stack: list, locals_dict: dict[str, Any], inst: dis.Instruction, co: CodeType, **kwargs
) -> None:
    assert inst.arg is not None
    var_name: str = co.co_cellvars[inst.arg]
    actual: Any = locals_dict[var_name]
    stack.append(actual)


@register_opcode_handler("LOAD_DEREF", supported_modes=[JitMode.PYTHON])
def _load_deref_handler(stack: list, closures: tuple[CellType, ...], inst: dis.Instruction, **kwargs) -> None:
    assert inst.arg is not None
    stack.append(closures[inst.arg].cell_contents)


# https://docs.python.org/id/3.5/library/dis.html#opcode-UNPACK_SEQUENCE
@register_opcode_handler("UNPACK_SEQUENCE", supported_modes=[JitMode.PYTHON])
def _unpack_sequence_handler(stack: list, **kwargs) -> None:
    seq: Sequence = stack.pop()

    for x in reversed(seq):
        stack.append(x)


@register_opcode_handler("BUILD_TUPLE", supported_modes=[JitMode.PYTHON])
def _build_tuple_handler(stack: list, inst: dis.Instruction, **kwargs) -> None:
    assert inst.arg is not None
    result: tuple[Any, ...] = tuple(reversed([stack.pop() for _ in range(inst.arg)]))
    stack.append(result)


# TODO Handle annotations
# TODO Review the context passed to FunctionType
# https://docs.python.org/id/3.5/library/dis.html#opcode-MAKE_FUNCTION
@register_opcode_handler("MAKE_FUNCTION", supported_modes=[JitMode.PYTHON])
def _make_function_handler(stack: list, locals_dict: dict[str, Any], inst: dis.Instruction, **kwargs) -> None:
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


# https://docs.python.org/3/library/dis.html#opcode-STORE_FAST
@register_opcode_handler("STORE_FAST", supported_modes=[JitMode.PYTHON])
def _store_fast_handler(
    stack: list, locals_dict: dict[str, Any], inst: dis.Instruction, co: CodeType, **kwargs
) -> None:
    a = stack.pop()
    assert inst.arg is not None
    var_name: str = co.co_varnames[inst.arg]
    locals_dict[var_name] = a


# TODO Review exception handling
# TODO Recursively call jit on the dunder add (and dunder radd) calls
@register_opcode_handler("BINARY_ADD", supported_modes=[JitMode.PYTHON])
def _binary_add_handler(stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()

    if (not hasattr(a, "__add__")) or ((result := a.__add__(b)) is NotImplemented):
        if (not hasattr(b, "__radd__")) or ((result := b.__radd__(a)) is NotImplemented):
            raise TypeError(f"Unsupported operand type(s) for +: '{type(a)}' and '{type(b)}'")

    stack.append(result)


# TODO Record these operations into the trace to generate constraints
# TODO Record these operations when between tensors because they actually launch kernels
# TODO Call the proper dunder methods
# TODO The reverse of gt is le! May check for TypeError instead of NotImplementedError
@register_opcode_handler("COMPARE_OP", supported_modes=[JitMode.PYTHON])
def _compare_op_handler(stack: list, inst: dis.Instruction, **kwargs) -> None:
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


# TODO Model the call to __bool__ on the object properly
@register_opcode_handler("POP_JUMP_IF_FALSE", supported_modes=[JitMode.PYTHON])
def _pop_jump_if_false_handler(stack: list, inst: dis.Instruction, **kwargs) -> int | None:
    a = stack.pop()
    cnd = bool(a)
    assert inst.arg is not None

    if cnd is False:
        return inst.arg
    return None


@register_opcode_handler("POP_JUMP_IF_TRUE", supported_modes=[JitMode.PYTHON])
def _pop_jump_if_true_handler(stack: list, inst: dis.Instruction, **kwargs) -> int | None:
    a = stack.pop()
    cnd = bool(a)
    assert inst.arg is not None

    if cnd is True:
        return inst.arg
    return None


@register_opcode_handler("RETURN_VALUE", supported_modes=[JitMode.PYTHON])
def _return_value_handler(**kwargs) -> int | None:
    return -1


@register_opcode_handler("GET_LEN", supported_modes=[JitMode.PYTHON])
def _get_len_handler(stack: list, **kwargs) -> None:
    a = stack.pop()
    stack.append(len(a))


@register_opcode_handler("NOP", supported_modes=[JitMode.PYTHON])
def _nop_handler(**kwargs) -> None:
    pass


@register_opcode_handler("IS_OP", supported_modes=[JitMode.PYTHON])
def _is_op_handler(instr: dis.Instruction, stack: list, **kwargs) -> None:
    b = stack.pop()
    a = stack.pop()
    stack.append(a is not b if instr.arg == 1 else a is b)


@register_opcode_handler("POP_TOP", supported_modes=[JitMode.PYTHON])
def _pop_top_handler(stack: list, **kwargs) -> None:
    stack.pop()


@register_opcode_handler("ROT_N", supported_modes=[JitMode.PYTHON])
def _rot_n_handler(stack: list, inst: dis.Instruction, **kwargs) -> None:
    assert inst.arg is not None
    assert len(stack) >= inst.arg
    stack[-inst.arg :] = (stack[-1], *stack[-inst.arg : -1])


@register_opcode_handler("ROT_TWO", supported_modes=[JitMode.PYTHON])
def _rot_two_handler(stack: list, **kwargs) -> None:
    a = stack.pop()
    b = stack.pop()
    c = stack.pop()
    stack.append(a)
    stack.append(c)
    stack.append(b)


# TODO Test how this handles functions, callable classes, and lambdas
#   Functions
#   Callable classes
#   Lambdas
# NOTE This only accepts positional args
@register_opcode_handler("CALL_FUNCTION", supported_modes=[JitMode.PYTHON])
def _call_function_handler(stack: list, inst: dis.Instruction, **kwargs) -> None:
    assert inst.arg is not None
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(inst.arg))))
    func: Callable = stack.pop()
    _, retval = _jit(func, *args)
    stack.append(retval)


# https://docs.python.org/id/3.5/library/dis.html#opcode-CALL_FUNCTION_KW
@register_opcode_handler("CALL_FUNCTION_KW", supported_modes=[JitMode.PYTHON])
def _call_function_kw_handler(stack: list, inst: dis.Instruction, **kwargs) -> None:
    kw_names: tuple[str, ...] = stack.pop()
    kwarg_length: int = len(kw_names)
    kwargs_flat: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(kwarg_length))))
    fn_kwargs: dict[str, Any] = {k: v for k, v in zip(kw_names, kwargs_flat)}
    assert inst.arg is not None
    arg_length: int = inst.arg - kwarg_length
    args = tuple(reversed(tuple(stack.pop() for _ in range(arg_length))))
    func: Callable = stack.pop()

    _, retval = _jit(func, *args, **fn_kwargs)
    stack.append(retval)


# TODO handle raising exception from user code
@register_opcode_handler("LOAD_METHOD", supported_modes=[JitMode.PYTHON])
def _load_method_handler(stack: list, inst: dis.Instruction, co: CodeType, **kwargs) -> None:
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


@register_opcode_handler("CALL_METHOD", supported_modes=[JitMode.PYTHON])
def _call_method_handler(stack: list, inst: dis.Instruction, **kwargs) -> None:
    assert inst.arg is not None
    args: tuple[Any, ...] = tuple(reversed(tuple(stack.pop() for _ in range(inst.arg))))
    second_lm = stack.pop()
    first_lm = stack.pop()
    if first_lm is not Py_NULL():
        meth = first_lm
        args = (second_lm, *args)
    else:
        meth = second_lm

    # TODO Restore lookaside depending on the mode
    # We should move this inside a common do_call() function.
    # lookaside: bool
    # result: Any
    # lookaside, result = _fn_lookaside(meth, *args)

    # if lookaside:
    #     stack.append(result)
    #     return

    # NOTE In this branch there was no function lookaside, so this traces into the function
    _, retval = _jit(meth, args)
    stack.append(retval)


# TODO Should the way this is called be like _jit(fn)(*args, **kwargs) instead of the current signature?
def _jit(fn, *args, **kwargs) -> tuple[bool, Any]:
    # TODO FIXME This won't acquire the code object for all callables
    #   What about partial objects, callable classes, callable modules, functools.wraps and @contextmanager...
    code: CodeType = fn.__code__
    insts: tuple[dis.Instruction, ...] = tuple(dis.get_instructions(fn))
    locals_dict: dict[str, Any] = dict(inspect.signature(fn).bind(*args, **kwargs).arguments)
    globals_dict: dict[str, Any] = globals()
    builtins_dict: dict[str, Any] = {k: getattr(__builtins__, k) for k in dir(__builtins__)}
    closures = fn.__closure__
    try_stack: list[TryBlock] = []
    stack: list[Any] = []

    inst_ptr: int = 0
    while True:
        inst: dis.Instruction = insts[inst_ptr]
        handler: None | Callable = _get_handler(inst.opname)

        # TODO Highlight the portion of the line that originated the opcode on Python versions that include
        #   the line offset information in the instruction
        if handler is None:
            line_no: int | None = None
            for i in reversed(insts[: insts.index(inst) + 1]):
                if i.starts_line is not None:
                    line_no = i.starts_line
                    break
            with open(code.co_filename) as f:
                lines = f.readlines()
            line_msg = f":{line_no}" if line_no else ""
            line_contents = lines[line_no - 1] if line_no else ""
            raise NotImplementedError(
                f"Encountered unimplemented opcode {inst.opname} while tracing.\n"
                f"Encountered in {fn.__name__}() at {code.co_filename if code.co_filename else 'Unknown'}{line_msg}\n"
                f"{line_contents}"
            )

        # NOTE On this path handler is not None

        stack_size_before_handler: int = len(stack)

        ptr_update: None | int = handler(
            inst_ptr=inst_ptr,
            stack=stack,
            locals_dict=locals_dict,
            globals_dict=globals_dict,
            builtins_dict=builtins_dict,
            try_stack=try_stack,
            closures=closures,
            inst=inst,
            co=code,
        )
        if ptr_update == -1:
            return (ret := inst.opname == "RETURN_VALUE"), stack.pop() if ret else None
        elif ptr_update is None:
            inst_ptr += 1
        else:
            inst_ptr = ptr_update

        # Verifies the handler had the expected stack effect (delta on stack size)
        actual_stack_effect: int = len(stack) - stack_size_before_handler
        jumped: bool = ptr_update is not None and ptr_update != -1
        expected_stack_effect: int = dis.stack_effect(inst.opcode, inst.arg, jump=jumped)
        assert (
            actual_stack_effect == expected_stack_effect
        ), f"Unexpected stack effect from {inst.opname}: expected {expected_stack_effect}, but the actual effect was {actual_stack_effect}"


# TODO Support remaining compilation parameters
def jit(fn: Callable, *, executors_list: None | Sequence[Executor] = None, mode: None | JitMode = None) -> Callable:
    cd: CompileData = CompileData(
        fn=fn,
        langctx=None,
        executors_list=executors_list,
        cache_mode=None,
        use_cudagraphs=None,
        use_torch_compile=None,
        disable_torch_autograd_support=None,
        use_rematerialization=None,
        only_execute_prims=None,
        disable_preprocessing=True,
    )
    cs: CompileStats = CompileStats()

    @functools.wraps(fn)
    def fn_(*args, **kwargs):
        # TODO Implement caching

        # Creates global datastructures
        global prologue_trace
        prologue_trace = TraceCtx(fn)
        prologue_trace.args = args
        prologue_trace.kwargs = kwargs

        # NOTE The computation_trace doesn't have a signature yet, its signature is determined
        #   by the computation "leaves" it uses
        global computation_trace
        computation_trace = TraceCtx()

        # Ties the prologue trace's and the computation trace's name logic together
        prologue_trace.names = computation_trace.names

        global jpm
        jpm = JitProxyMap()

        global jit_mode
        jit_mode = mode if mode is not None else JitMode.THUNDER

        should_return, result = _jit(fn, *args, **kwargs)

        # Python mode computes the actual result here
        if mode is JitMode.PYTHON:
            return result

        raise NotImplementedError(f"Modes other than JitMode.PYTHON are not yet supported")

    return fn_


pjit = partial(jit, mode=JitMode.PYTHON)
