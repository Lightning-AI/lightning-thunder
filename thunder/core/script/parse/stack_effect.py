import dis
import opcode
import sys
from typing import NewType, TypeAlias, TypeVar
from collections.abc import Callable
from collections.abc import Iterable

from types import EllipsisType

from thunder.core.utils import FrozenDict

__all__ = ("stack_effect_detail", "fill_ellipses")

T = TypeVar("T")
Pop = NewType("Pop", int)
Push = NewType("Push", int)
StackEffect: TypeAlias = tuple[Pop, Push] | tuple[Pop, tuple[Push, Push]]

# Aliases for common cases
NoStackEffect = (Pop(0), Push(0))
PushTOS = (Pop(0), Push(1))
PopTOS = (Pop(1), Push(0))
ReplaceTOS = (Pop(1), Push(1))
BinaryOp = (Pop(2), Push(1))


def make_function_detail(*args: int) -> Callable[[int], StackEffect]:
    return lambda oparg: (Pop(2 + sum((oparg & flag) != 0 for flag in args)), Push(1))


def fill_ellipses(**kwargs: T | EllipsisType) -> Iterable[tuple[str, T]]:
    prior_effect: T | EllipsisType = Ellipsis
    for opname, effect in kwargs.items():
        if effect is Ellipsis:
            effect = prior_effect
        assert effect is not Ellipsis
        prior_effect = effect
        yield opname, effect


__EFFECTS = dict[str, StackEffect | Callable[[int], StackEffect] | EllipsisType](
    NOP=NoStackEffect,  #                                   ∅           -> ∅
    EXTENDED_ARG=NoStackEffect,
    #
    # Stack manipulation
    POP_TOP=PopTOS,  #                                      A           -> ∅
    ROT_TWO=(Pop(2), Push(2)),  #                           A,B         -> B,A
    ROT_THREE=(Pop(3), Push(3)),  #                         A,B,C       -> C,A,B
    ROT_FOUR=(Pop(4), Push(4)),  #                          A,B,C,D     -> D,A,B,C
    ROT_N=lambda oparg: (Pop(oparg), Push(oparg)),  #       A,B,...,Z   -> Z,A,B,...
    DUP_TOP=(Pop(1), Push(2)),  #                           A           -> A,A
    DUP_TOP_TWO=(Pop(2), Push(4)),  #                       A,B         -> A,B,A,B
    UNPACK_SEQUENCE=lambda oparg: (Pop(1), Push(oparg)),  # A           -> B,C,...
    #
    # Jumps & return
    JUMP_FORWARD=NoStackEffect,  #                          ∅           -> ∅
    JUMP_ABSOLUTE=...,
    POP_JUMP_IF_FALSE=PopTOS,  #                            A           -> ∅
    POP_JUMP_IF_TRUE=...,
    RETURN_VALUE=...,
    JUMP_IF_NOT_EXC_MATCH=BinaryOp,  #                      A,B         -> ∅
    #
    # Exceptions and context managers:
    POP_BLOCK=NoStackEffect,  #                             ∅           -> ∅
    POP_EXCEPT=(Pop(3), Push(0)),  #                        A, B, C     -> ∅
    RERAISE=...,
    RAISE_VARARGS=lambda oparg: (Pop(oparg), Push(0)),  #   A,B,...     -> ∅
    WITH_EXCEPT_START=(Pop(7), Push(8)),  #       ??!?
    LOAD_ASSERTION_ERROR=PushTOS,  #                        ∅           -> A
    #
    # Variable manipulation
    LOAD_CONST=PushTOS,  #                                  ∅           -> A
    LOAD_FAST=...,
    LOAD_GLOBAL=...,
    LOAD_NAME=...,
    STORE_FAST=PopTOS,  #                                   A           -> ∅
    STORE_GLOBAL=...,
    STORE_NAME=...,
    DELETE_FAST=NoStackEffect,  #                           ∅           -> ∅
    DELETE_GLOBAL=...,
    DELETE_NAME=...,
    #
    # Attributes
    LOAD_METHOD=(Pop(1), Push(2)),  #                       A           -> B,A
    LOAD_ATTR=ReplaceTOS,  #                                A           -> B
    STORE_ATTR=(Pop(2), Push(0)),  #                        A, B        -> ∅
    DELETE_ATTR=PopTOS,  #                                  A           -> ∅
    #
    # Closures
    LOAD_CLOSURE=PushTOS,  #                                ∅           -> A
    LOAD_DEREF=...,
    LOAD_CLASSDEREF=...,
    STORE_DEREF=PopTOS,  #                                  A           -> ∅
    DELETE_DEREF=NoStackEffect,  #                          ∅           -> ∅
    #
    # Functions and calls                                   A,B,...     -> Z
    CALL_FUNCTION=lambda x: (Pop(x + 1), Push(1)),
    CALL_METHOD=lambda x: (Pop(x + 2), Push(1)),
    CALL_FUNCTION_KW=...,
    CALL_FUNCTION_EX=make_function_detail(0x01),
    MAKE_FUNCTION=make_function_detail(0x01, 0x02, 0x04, 0x08),
    #
    # Build containers                                      A,B,...     -> Z
    BUILD_TUPLE=lambda oparg: (Pop(oparg), Push(1)),
    BUILD_LIST=...,
    BUILD_SET=...,
    BUILD_STRING=...,
    BUILD_MAP=lambda oparg: (Pop(oparg * 2), Push(1)),
    BUILD_CONST_KEY_MAP=lambda x: (Pop(x + 1), Push(1)),
    LIST_TO_TUPLE=ReplaceTOS,  #                            A           -> B
    #
    # Insertion leaves container on the stack               A,B         -> A
    SET_ADD=BinaryOp,
    SET_UPDATE=...,
    LIST_APPEND=...,
    LIST_EXTEND=...,
    DICT_MERGE=...,
    DICT_UPDATE=...,
    MAP_ADD=(Pop(3), Push(1)),  #                           A,B,C       -> A
    COPY_DICT_WITHOUT_KEYS=(Pop(2), Push(2)),  #            A,B         -> A,C  (I am unsure...)
    #
    # Unary operators                                       A           -> B
    UNARY_POSITIVE=ReplaceTOS,
    UNARY_NEGATIVE=...,
    UNARY_NOT=...,
    UNARY_INVERT=...,
    #
    # Binary operators                                      A,B         -> C
    BINARY_POWER=BinaryOp,
    BINARY_MULTIPLY=...,
    BINARY_MATRIX_MULTIPLY=...,
    BINARY_MODULO=...,
    BINARY_ADD=...,
    BINARY_SUBTRACT=...,
    BINARY_SUBSCR=...,
    BINARY_FLOOR_DIVIDE=...,
    BINARY_TRUE_DIVIDE=...,
    INPLACE_FLOOR_DIVIDE=...,
    INPLACE_TRUE_DIVIDE=...,
    INPLACE_ADD=...,
    INPLACE_SUBTRACT=...,
    INPLACE_MULTIPLY=...,
    INPLACE_MATRIX_MULTIPLY=...,
    INPLACE_MODULO=...,
    BINARY_LSHIFT=...,
    BINARY_RSHIFT=...,
    BINARY_AND=...,
    BINARY_XOR=...,
    BINARY_OR=...,
    COMPARE_OP=...,
    IS_OP=...,
    CONTAINS_OP=...,
    #
    # Binary operators (inplace)
    #   https://docs.python.org/3/reference/datamodel.html?highlight=iadd#object.__iadd__
    #   "... and return the result (which could be, but does not have to be, self)."
    INPLACE_POWER=BinaryOp,
    INPLACE_LSHIFT=...,
    INPLACE_RSHIFT=...,
    INPLACE_AND=...,
    INPLACE_XOR=...,
    INPLACE_OR=...,
    #
    # Indexing operators
    STORE_SUBSCR=(Pop(3), Push(0)),  #                      A,B,C       -> ∅
    DELETE_SUBSCR=(Pop(2), Push(0)),  #                     A,B         -> ∅
    BUILD_SLICE=lambda x: (Pop(x), Push(1)),  #             A,B,...     -> Z
    UNPACK_EX=lambda x: (Pop(1), Push((x & 0xFF) + (x >> 8) + 1)),  #  A   -> B,C,...
    #
    # Iterators
    GET_ITER=ReplaceTOS,  #                                 A           -> B
    GET_YIELD_FROM_ITER=ReplaceTOS,
    #
    # Misc.
    FORMAT_VALUE=lambda oparg: (Pop(1 + bool(oparg & 0x04)), Push(1)),  # (A?),B      -> C
    PRINT_EXPR=PopTOS,  #                                   A           -> ∅
    IMPORT_STAR=...,
    LOAD_BUILD_CLASS=PushTOS,
    SETUP_ANNOTATIONS=NoStackEffect,
    GET_LEN=(Pop(1), Push(2)),
    IMPORT_NAME=BinaryOp,
    IMPORT_FROM=(Pop(1), Push(2)),
    MATCH_CLASS=(Pop(3), Push(1)),
    MATCH_MAPPING=(Pop(1), Push(2)),
    MATCH_SEQUENCE=...,
    MATCH_KEYS=(Pop(2), Push(3 + bool(sys.version_info < (3, 11)))),
    #
    # Jump dependent
    FOR_ITER=(Pop(1), (Push(2), Push(0))),
    SETUP_WITH=(Pop(1), (Push(2), Push(7))),
    SETUP_FINALLY=(Pop(0), (Push(0), Push(6))),
    SETUP_ASYNC_WITH=(Pop(0), (Push(0), Push(6))),
    #
    # NOTE: These instructions have been removed since they are extraneous special cases.
    #       https://github.com/faster-cpython/ideas/issues/567
    #       https://github.com/python/cpython/issues/102859
    JUMP_IF_TRUE_OR_POP=(Pop(1), (Push(0), Push(1))),
    JUMP_IF_FALSE_OR_POP=...,
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
)


# Split so MyPy can type check `__EFFECTS` without having to go through `fill_ellipses`.
_RAW_STACK_EFFECTS = FrozenDict[str, StackEffect | Callable[[int], StackEffect]](fill_ellipses(**__EFFECTS))
del __EFFECTS


def stack_effect_detail(instruction: dis.Instruction) -> tuple[Pop, tuple[Push, Push]]:
    assert isinstance(instruction, dis.Instruction), instruction
    if callable(effect := _RAW_STACK_EFFECTS[instruction.opname]):
        assert instruction.arg is not None
        effect = effect(instruction.arg)

    assert isinstance(effect, tuple) and len(effect) == 2 and isinstance(effect[0], int)
    if isinstance(effect[1], int):
        effect = (effect[0], (effect[1],) * 2)

    # Python exposes a method to compute stack effect, so while it's not part
    # of the public API we may as well use it to check our bookkeeping.
    pop, (push_nojump, push_jump) = effect
    for jump, push in ((False, push_nojump), (True, push_jump)):
        expected = opcode.stack_effect(instruction.opcode, instruction.arg, jump=jump)
        assert expected == push - pop, (expected, push, pop, jump)

    return Pop(pop), (Push(push_nojump), Push(push_jump))
