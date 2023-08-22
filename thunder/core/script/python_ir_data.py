import dis
import enum
import logging
import sys
from types import CodeType
from typing import Callable, Dict, Optional, Tuple, Union
from collections.abc import Iterable

try:
    from types import EllipsisType
except ImportError:
    # EllipsisType was introduced in 3.10
    EllipsisType = type(...)

from thunder.core.utils import OrderedSet

logger = logging.getLogger(__name__)
SUPPORTS_PREPROCESSING = (3, 9) <= sys.version_info < (3, 11)
EXTENDED_ARG = "EXTENDED_ARG"
JUMP_ABSOLUTE = "JUMP_ABSOLUTE"
RETURN_VALUE = "RETURN_VALUE"
X_THUNDER_STORE_ATTR = "X_THUNDER_STORE_ATTR"


class VariableScope(enum.Enum):
    CONST = enum.auto()
    LOCAL = enum.auto()
    NONLOCAL = enum.auto()
    GLOBAL = enum.auto()
    STACK = enum.auto()


class InstructionSet(OrderedSet[str]):
    """Convenience class for checking opcode properties."""

    def canonicalize(self, i: str | int | dis.Instruction) -> str:
        if isinstance(i, str):
            return i

        elif isinstance(i, int):
            return dis.opname[i]

        else:
            assert isinstance(i, dis.Instruction)
            return i.prefix if isinstance(i, SyntheticInstruction) else i.opname


UNCONDITIONAL_JUMP_INSTRUCTIONS = InstructionSet(
    (JUMP_ABSOLUTE, "JUMP_FORWARD", "JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT")
)
JUMP_INSTRUCTIONS = InstructionSet((*dis.hasjabs, *dis.hasjrel, *UNCONDITIONAL_JUMP_INSTRUCTIONS))

RAISE_RETURN_INSTRUCTIONS = InstructionSet(("RAISE_VARARGS", "RERAISE"))
RETURN_INSTRUCTIONS = InstructionSet((RETURN_VALUE, *RAISE_RETURN_INSTRUCTIONS))


class SyntheticInstruction(dis.Instruction):
    def __new__(cls, i: dis.Instruction) -> "SyntheticInstruction":
        opname, _, *args = i
        i_is_synthetic = isinstance(i, SyntheticInstruction)
        if not i_is_synthetic:
            opname = f"{opname}__{cls.__name__}"
            assert opname not in dis.opmap, opname

        self = super(SyntheticInstruction, cls).__new__(cls, opname, -1, *args)
        self.prefix: str = i.prefix if i_is_synthetic else i.opname
        return self

    def __copy__(self) -> "SyntheticInstruction":
        return self.__class__(self)


#   One quirk of the CPython interpreter is that some jump instructions will
#   push different arguments onto the stack depending on which branch is taken.
#   This can add complexity and make analysis and manipulation more cumbersome.
#   To simplify things we split off the jump dependent behavior into "epilogue"
#   pseudo-Instructions and put them in separate blocks. That means that only
#   the very first (parsing) and very last (binding) stages need to handle this
#   behavior; everything in between just sees a simple Graph.
class NoJumpEpilogue(SyntheticInstruction):
    pass


class JumpEpilogue(SyntheticInstruction):
    pass


# (pop, push)
StackEffect = Tuple[int, Tuple[int, ...]]

# Common stack effects to make `_STACK_EFFECTS_SPEC` more readable and less bug prone.
PushNew = (0,)
NoStackEffect = (0, ())
TOS = -1  # Top of stack.
PeekTOS = (1, (TOS,))
PeekTOS_andPushNew = (1, (TOS, 0))
PopTOS = (1, ())
ReplaceTOS = (1, PushNew)


def rotate_N(oparg: int) -> StackEffect:
    return oparg, (TOS,) + tuple(range(-oparg, -1))


def unpack_N(oparg: int) -> StackEffect:
    return 1, tuple(range(oparg))


def format_value(oparg: int) -> StackEffect:
    return (2 if ((oparg & 0x04) != 0) else 1), PushNew


def function_detail(*args: list[int]):
    flags = tuple(args)

    def effect(oparg: int) -> StackEffect:
        return 2 + sum((oparg & flag) != 0 for flag in flags), PushNew

    return effect


_STACK_EFFECTS_SPEC: Dict[str, Union[StackEffect, EllipsisType, Callable[[int], StackEffect]]] = {
    "NOP": NoStackEffect,  #                            ∅           -> ∅
    "EXTENDED_ARG": NoStackEffect,
    #
    # Stack manipulation
    "POP_TOP": PopTOS,  #                               A           -> ∅
    "ROT_TWO": rotate_N(2),  #                          A,B         -> B,A
    "ROT_THREE": rotate_N(3),  #                        A,B,C       -> C,A,B
    "ROT_FOUR": rotate_N(4),  #                         A,B,C,D     -> D,A,B,C
    "ROT_N": rotate_N,  #                               A,B,...,Z   -> Z,A,B,...
    "DUP_TOP": (1, (TOS, TOS)),  #                      A           -> A,A
    "DUP_TOP_TWO": (2, (-2, -1, -2, -1)),  #            A,B         -> A,B,A,B
    "UNPACK_SEQUENCE": unpack_N,  #                     A           -> B,C,...
    #
    # Jumps & return
    "JUMP_FORWARD": NoStackEffect,  #                   ∅           -> ∅
    JUMP_ABSOLUTE: ...,
    "POP_JUMP_IF_FALSE": PopTOS,  #                     A           -> ∅
    "POP_JUMP_IF_TRUE": ...,
    RETURN_VALUE: ...,
    "JUMP_IF_NOT_EXC_MATCH": (2, ()),  #                A,B         -> ∅
    #
    # Exceptions and context managers:
    "POP_BLOCK": NoStackEffect,  #                      ∅           -> ∅
    "POP_EXCEPT": (3, ()),  #                           A, B, C     -> ∅
    "RERAISE": ...,
    "RAISE_VARARGS": lambda oparg: (oparg, ()),  #      A,B,...     -> ∅
    "WITH_EXCEPT_START": (7, tuple(range(8))),  #       ??!?
    "LOAD_ASSERTION_ERROR": (0, PushNew),  #            ∅           -> A
    #
    # Variable manipulation
    "LOAD_CONST": (0, PushNew),  #                      ∅           -> A
    "LOAD_FAST": ...,
    "LOAD_GLOBAL": ...,
    "LOAD_NAME": ...,
    "LOAD_METHOD": (1, (0, TOS)),  #                    A           -> B,A
    "STORE_FAST": PopTOS,  #                            A           -> ∅
    "STORE_GLOBAL": ...,
    "STORE_NAME": ...,
    "DELETE_FAST": NoStackEffect,  #                    ∅           -> ∅
    "DELETE_GLOBAL": ...,
    "DELETE_NAME": ...,
    #
    # Attributes
    "LOAD_ATTR": ReplaceTOS,  #                         A           -> B
    "STORE_ATTR": (2, ()),  #                           A, B        -> ∅
    "DELETE_ATTR": PopTOS,  #                           A           -> ∅
    #
    # Closures
    "LOAD_CLOSURE": (0, PushNew),  #                    ∅           -> A
    "LOAD_DEREF": ...,
    "LOAD_CLASSDEREF": ...,
    "STORE_DEREF": PopTOS,  #                           A           -> ∅
    "DELETE_DEREF": NoStackEffect,  #                   ∅           -> ∅
    #
    # Functions and calls                               A,B,...     -> Z
    "CALL_FUNCTION": lambda x: (x + 1, PushNew),
    "CALL_METHOD": lambda x: (x + 2, PushNew),
    "CALL_FUNCTION_KW": ...,
    "CALL_FUNCTION_EX": function_detail(0x01),
    "MAKE_FUNCTION": function_detail(0x01, 0x02, 0x04, 0x08),
    #
    # Build containers                                  A,B,...     -> Z
    "BUILD_TUPLE": lambda oparg: (oparg, PushNew),
    "BUILD_LIST": ...,
    "BUILD_SET": ...,
    "BUILD_STRING": ...,
    "BUILD_MAP": lambda oparg: (oparg * 2, PushNew),
    "BUILD_CONST_KEY_MAP": lambda x: (x + 1, PushNew),
    "LIST_TO_TUPLE": ReplaceTOS,  #                     A           -> B
    #
    # Insertion leaves container on the stack           A,B         -> A
    "SET_ADD": (2, (-2,)),
    "SET_UPDATE": ...,
    "LIST_APPEND": ...,
    "LIST_EXTEND": ...,
    "MAP_ADD": ...,
    "DICT_MERGE": ...,
    "DICT_UPDATE": ...,
    "COPY_DICT_WITHOUT_KEYS": (2, (-2, 0)),  #          A,B         -> A,C  (I am unsure...)
    #
    # Unary operators                                   A           -> B
    "UNARY_POSITIVE": ReplaceTOS,
    "UNARY_NEGATIVE": ...,
    "UNARY_NOT": ...,
    "UNARY_INVERT": ...,
    #
    # Binary operators                                  A,B         -> C
    "BINARY_POWER": (2, PushNew),
    "BINARY_MULTIPLY": ...,
    "BINARY_MATRIX_MULTIPLY": ...,
    "BINARY_MODULO": ...,
    "BINARY_ADD": ...,
    "BINARY_SUBTRACT": ...,
    "BINARY_SUBSCR": ...,
    "BINARY_FLOOR_DIVIDE": ...,
    "BINARY_TRUE_DIVIDE": ...,
    "INPLACE_FLOOR_DIVIDE": ...,
    "INPLACE_TRUE_DIVIDE": ...,
    "INPLACE_ADD": ...,
    "INPLACE_SUBTRACT": ...,
    "INPLACE_MULTIPLY": ...,
    "INPLACE_MATRIX_MULTIPLY": ...,
    "INPLACE_MODULO": ...,
    "BINARY_LSHIFT": ...,
    "BINARY_RSHIFT": ...,
    "BINARY_AND": ...,
    "BINARY_XOR": ...,
    "BINARY_OR": ...,
    "COMPARE_OP": ...,
    "IS_OP": ...,
    "CONTAINS_OP": ...,
    #
    # Binary operators (inplace)
    #   https://docs.python.org/3/reference/datamodel.html?highlight=iadd#object.__iadd__
    #   "... and return the result (which could be, but does not have to be, self)."
    "INPLACE_POWER": (2, PushNew),
    "INPLACE_LSHIFT": ...,
    "INPLACE_RSHIFT": ...,
    "INPLACE_AND": ...,
    "INPLACE_XOR": ...,
    "INPLACE_OR": ...,
    #
    # Indexing operators
    "STORE_SUBSCR": (3, ()),  #                         A,B,C       -> ∅
    "DELETE_SUBSCR": (2, ()),  #                        A,B         -> ∅
    "BUILD_SLICE": lambda x: (x, PushNew),  #           A,B,...     -> Z
    "UNPACK_EX": lambda x: (1, tuple(range((x & 0xFF) + (x >> 8) + 1))),  # A   -> B,C,...
    #
    # Iterators
    "GET_ITER": ReplaceTOS,  #                          A           -> B
    "GET_YIELD_FROM_ITER": ReplaceTOS,
    #
    # Misc.
    "FORMAT_VALUE": format_value,  #                    (A?),B      -> C
    "PRINT_EXPR": PopTOS,  #                            A           -> ∅
    "IMPORT_STAR": ...,
    "LOAD_BUILD_CLASS": (0, PushNew),
    "SETUP_ANNOTATIONS": NoStackEffect,
    "GET_LEN": (1, (-1, 0)),
    "IMPORT_NAME": (2, PushNew),
    "IMPORT_FROM": (1, (-1, 0)),
    "MATCH_CLASS": (3, PushNew),
    "MATCH_MAPPING": (1, (TOS, 0)),
    "MATCH_SEQUENCE": ...,
    "MATCH_KEYS": (2, (-1, -2, 0) + () if sys.version_info >= (3, 11) else (1,)),
    #
    # TODO(robieta, t-vi): Iterators and generators
    # "GEN_START": PopTOS,  #   Where does TOS for this come from?
    # "YIELD_VALUE": ReplaceTOS,  # I think
    # "YIELD_FROM": (2, PushNew),  #  I am very unsure
    # "GET_AWAITABLE": (1, 1),
    # "BEFORE_ASYNC_WITH": (1, 2),
    # "GET_AITER": (1, 1),
    # "GET_ANEXT": (1, 2),
    # "END_ASYNC_FOR": (7, 0),
}


_JUMP_DEPENDENT_SPEC: Dict[str, Tuple[StackEffect, StackEffect, StackEffect]] = {
    "FOR_ITER": (PeekTOS, PeekTOS_andPushNew, PopTOS),
    "SETUP_WITH": ((1, range(2)), NoStackEffect, (2, range(-2, 7 - 2))),
    "SETUP_FINALLY": (NoStackEffect, NoStackEffect, (0, range(6))),
    "SETUP_ASYNC_WITH": (ReplaceTOS, NoStackEffect, (1, range(-1, 6 - 1))),
    #
    # NOTE: These instructions have been removed since they are extraneous special cases.
    #       https://github.com/faster-cpython/ideas/issues/567
    #       https://github.com/python/cpython/issues/102859
    "JUMP_IF_TRUE_OR_POP": (PeekTOS, PopTOS, NoStackEffect),
    "JUMP_IF_FALSE_OR_POP": (PeekTOS, PopTOS, NoStackEffect),
}


FIXED_STACK_EFFECTS_DETAIL: Dict[str, StackEffect] = {}
SIMPLE_VARIABLE_STACK_EFFECTS_DETAIL: Dict[str, Callable[[int], StackEffect]] = {}
JUMP_DEPENDENT_DETAIL: Dict[Tuple[str, bool], Tuple[str, StackEffect]] = {}


def __build_maps():
    assert not FIXED_STACK_EFFECTS_DETAIL
    assert not SIMPLE_VARIABLE_STACK_EFFECTS_DETAIL
    assert not JUMP_DEPENDENT_DETAIL

    for opname, effects in _JUMP_DEPENDENT_SPEC.items():
        shared_effect, nojump_effect, jump_effect = ((pop, tuple(stack)) for pop, stack in effects)
        assert opname in JUMP_INSTRUCTIONS, opname
        assert opname not in UNCONDITIONAL_JUMP_INSTRUCTIONS, opname
        assert opname not in _STACK_EFFECTS_SPEC
        FIXED_STACK_EFFECTS_DETAIL[opname] = shared_effect
        for is_jump, epilogue in enumerate((nojump_effect, jump_effect)):
            JUMP_DEPENDENT_DETAIL[(opname, bool(is_jump))] = epilogue

    prior_effect = Ellipsis
    for opname, effect in _STACK_EFFECTS_SPEC.items():
        if effect is Ellipsis:
            effect = prior_effect

        if effect is Ellipsis:
            logger.warn("Invalid effect for opname %s", opname)

        elif isinstance(effect, tuple):
            FIXED_STACK_EFFECTS_DETAIL[opname] = effect

        elif callable(effect):
            SIMPLE_VARIABLE_STACK_EFFECTS_DETAIL[opname] = effect

        else:
            logger.warn("Unhandled opname %s: %s", opname, effect)

        prior_effect = effect


# Unpack `_STACK_EFFECTS_SPEC` into constituent maps.
__build_maps()
del __build_maps


def stack_effect_adjusted(instruction: dis.Instruction) -> StackEffect:
    opname: str = instruction.opname
    if isinstance(instruction, NoJumpEpilogue):
        return JUMP_DEPENDENT_DETAIL[(instruction.prefix, False)]

    elif isinstance(instruction, JumpEpilogue):
        return JUMP_DEPENDENT_DETAIL[(instruction.prefix, True)]

    elif (effect := FIXED_STACK_EFFECTS_DETAIL.get(opname)) is not None:
        return effect

    elif (effect_fn := SIMPLE_VARIABLE_STACK_EFFECTS_DETAIL.get(opname)) is not None:
        oparg: Optional[int] = instruction.arg
        assert oparg is not None
        return effect_fn(oparg)

    raise ValueError(f"Invalid opname {opname}")


def get_epilogue(instruction: dis.Instruction, *, jump: bool = False) -> Optional[dis.Instruction]:
    if (instruction.opname, jump) in JUMP_DEPENDENT_DETAIL:
        epilogue = (JumpEpilogue if jump else NoJumpEpilogue)(instruction)
        epilogue.line_no = getattr(instruction, "line_no", None)
        return epilogue


def stack_effect_detail(instruction: dis.Instruction, *, jump: bool = False) -> tuple[int, int]:
    assert type(instruction) is dis.Instruction, type(instruction)
    pop, push = stack_effect_adjusted(instruction)
    if epilogue := get_epilogue(instruction, jump=jump):
        epilogue_pop, epilogue_push = stack_effect_adjusted(epilogue)
        push = push[:-epilogue_pop] + epilogue_push
    return pop, len(push)


def make_jump_absolute(arg: int) -> dis.Instruction:
    return dis.Instruction(
        opname=JUMP_ABSOLUTE,
        opcode=dis.opmap.get(JUMP_ABSOLUTE, -1),
        arg=arg,
        argval=None,
        argrepr=f"{arg}",
        offset=-999,
        starts_line=None,
        is_jump_target=False,
    )


def make_return(is_jump_target: bool) -> dis.Instruction:
    return dis.Instruction(
        opname=RETURN_VALUE,
        opcode=dis.opmap[RETURN_VALUE],
        arg=None,
        argval=None,
        argrepr="",
        offset=-999,
        starts_line=None,
        is_jump_target=is_jump_target,
    )


def get_instruction(opname: str, arg: Optional[int], **kwargs) -> dis.Instruction:
    ctor_kwargs = dict(
        opname=opname,
        opcode=dis.opmap.get(opname, -1),
        arg=arg,
        argval=None,
        argrepr="None",
        offset=-999,
        starts_line=None,
        is_jump_target=False,
    )
    ctor_kwargs.update(kwargs)
    return dis.Instruction(**ctor_kwargs)


def modify_copy_instruction(i: dis.Instruction, **kwargs) -> dis.Instruction:
    # todo: or make a mutuable Instruction?
    return dis.Instruction(**{**i._asdict(), **kwargs})


def compute_jump(instruction: dis.Instruction, position: int) -> Optional[int]:
    if instruction.opcode in dis.hasjabs:
        assert instruction.arg is not None
        return instruction.arg

    elif instruction.opname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
        assert instruction.arg is not None
        return position + 1 - instruction.arg

    elif "BACKWARD" in instruction.opname:
        # TODO: POP_JUMP_BACKWARD_IF_... variants
        raise NotImplementedError(instruction.opname)

    elif instruction.opcode in dis.hasjrel:
        assert instruction.arg is not None
        return position + 1 + instruction.arg

    return None


def debug_compare_functions_print(diffs: dict[str, Tuple[list, list]]):
    for k, (v1, v2) in diffs.items():
        if not (v1 == None and v2 == None):
            print(f"Differences in: {k}")
            print(f"  CodeObject 1: {v1}")
            print(f"  CodeObject 2: {v2}")


def debug_compare_functions(
    code1: Union[CodeType, Callable], code2: Union[CodeType, Callable], *, show=False
) -> dict[str, Tuple[list, list]]:
    if not isinstance(code1, CodeType):
        code1 = code1.__code__
    if not isinstance(code2, CodeType):
        code2 = code2.__code__

    attrs = [
        "co_argcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_filename",
        "co_name",
        "co_freevars",
        "co_cellvars",
    ]

    diffs = {}
    for attr in attrs:
        v1 = getattr(code1, attr)
        v2 = getattr(code2, attr)

        if v1 != v2:
            if isinstance(v1, dict) and isinstance(v2, dict):
                diffs[attr] = (v1 - v2, v2 - v1)
            if isinstance(v1, str) and isinstance(v2, str):
                diffs[attr] = (v1, v2)
            elif isinstance(v1, Iterable) and isinstance(v2, Iterable):
                diffs[attr] = (set(v1) - set(v2), set(v2) - set(v1))
            else:
                diffs[attr] = (v1, v2)

    if show:
        debug_compare_functions_print(diffs)

    return diffs


load_opcodes = {
    "LOAD_CONST": VariableScope.CONST,
    "LOAD_FAST": VariableScope.LOCAL,
    "LOAD_DEREF": VariableScope.NONLOCAL,
    "LOAD_GLOBAL": VariableScope.GLOBAL,
}

store_opcodes = {
    "STORE_FAST": VariableScope.LOCAL,
    "STORE_DEREF": VariableScope.NONLOCAL,
    "STORE_GLOBAL": VariableScope.GLOBAL,
}

del_opcodes = {
    "DELETE_FAST": VariableScope.LOCAL,
    "DELETE_DEREF": VariableScope.NONLOCAL,
    "DELETE_GLOBAL": VariableScope.GLOBAL,
}
