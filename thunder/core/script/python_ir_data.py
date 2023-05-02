import dis
import enum
import logging
import sys
from types import CodeType
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

try:
    from types import EllipsisType
except ImportError:
    # EllipsisType was introduced in 3.10
    EllipsisType = type(...)

from thunder.core.utils import dict_join

logger = logging.getLogger(__name__)
SUPPORTS_PREPROCESSING = ((3, 9) <= sys.version_info < (3, 11))

# this is Python 3.10 specific for the time being.

#  *  0 -- when not jump
#  *  1 -- when jump
#  * -1 -- maximal

class VariableScope(enum.Enum):
    CONST = enum.auto()
    LOCAL = enum.auto()
    NONLOCAL = enum.auto()
    GLOBAL = enum.auto()
    STACK = enum.auto()


class ArgScope(NamedTuple):
    arg: int
    scope: VariableScope

    def __repr__(self) -> str:
        return f"ArgScope(arg={self.arg}, scope={self.scope.name})"

    def __lt__(self, other: "ArgScope") -> bool:
        return (self.scope.value, self.arg) < (other.scope.value, other.arg)


FixedEffect = Tuple[int, Tuple[int, ...]]
TOS = -1  # Top of stack.
PushNew = (0,)
NoBranchDependent = (False, ())

def rotate_N(oparg: int) -> FixedEffect:
    return oparg, (TOS,) + tuple(range(-oparg, -1))

def unpack_N(oparg: int) -> FixedEffect:
    return 1, tuple(range(oparg))

def format_value(oparg: int) -> FixedEffect:
    return (2 if ((oparg & 0x04) != 0) else 1), PushNew

def function_detail(*args: List[int]):
    flags = tuple(args)
    def effect(oparg: int) -> FixedEffect:
        return 2 + sum((oparg & flag) != 0 for flag in flags), PushNew
    return effect


_STACK_EFFECTS_SPEC: Dict[str, Union[FixedEffect, EllipsisType, Callable[[int], FixedEffect]]] = {
    "NOP": (0, ()),  #                                  ∅ -> ∅
    "EXTENDED_ARG": (0, ()),  #                         ∅ -> ∅

    # Stack manipulation
    "POP_TOP": (1, ()),  #                              A           -> ∅
    "ROT_TWO":   rotate_N(2),  #                        A,B         -> B,A
    "ROT_THREE": rotate_N(3),  #                        A,B,C       -> C,A,B
    "ROT_FOUR":  rotate_N(4),  #                        A,B,C,D     -> D,A,B,C
    "ROT_N":     rotate_N,  #                           A,B,...,Z   -> Z,A,B,...
    "DUP_TOP": (1, (TOS, TOS)),  #                      A           -> A,A
    "DUP_TOP_TWO": (2, (-2, -1, -2, -1)),  #            A,B         -> A,B,A,B
    "UNPACK_SEQUENCE": unpack_N,  #                     A           -> B,C,...

    # Jumps & return
    "JUMP_FORWARD": (0, ()),  #                         ∅           -> ∅
    "JUMP_ABSOLUTE": ...,
    "POP_JUMP_IF_FALSE": (1, ()),  #                    A           -> ∅
    "POP_JUMP_IF_TRUE": ...,
    "RETURN_VALUE": ...,
    "JUMP_IF_NOT_EXC_MATCH": (2, ()),  #                A,B         -> ∅

    # Exceptions and context managers:
    "POP_BLOCK": (0, ()),  #                            ∅           -> ∅
    "POP_EXCEPT": (3, ()),  #                           A, B, C     -> ∅
    "RERAISE": ...,
    "RAISE_VARARGS": lambda oparg: (oparg, ()),  #      A,B,...     -> ∅
    "WITH_EXCEPT_START": (7, tuple(range(8))),  #       ??!?
    "LOAD_ASSERTION_ERROR": (0, PushNew),  #            ∅           -> A

    # Variable manipulation
    "LOAD_CONST": (0, PushNew),  #                      ∅           -> A
    "LOAD_FAST": ...,
    "LOAD_GLOBAL": ...,
    "LOAD_NAME": ...,
    "LOAD_METHOD": (1, (0, TOS)),  #                    A           -> B,A

    "STORE_FAST": (1, ()),  #                           A           -> ∅
    "STORE_GLOBAL": ...,
    "STORE_NAME": ...,
 
    "DELETE_FAST": (0, ()),  #                          ∅           -> ∅
    "DELETE_GLOBAL": ...,
    "DELETE_NAME": ...,

    # Attributes
    "LOAD_ATTR": (1, PushNew),  #                       A           -> B
    "STORE_ATTR": (2, ()),  #                           A, B        -> ∅
    "DELETE_ATTR": (1, ()),  #                          A           -> ∅

    # Closures
    "LOAD_CLOSURE": (0, PushNew),  #                    ∅           -> A
    "LOAD_DEREF": ...,
    "LOAD_CLASSDEREF": ...,
    "STORE_DEREF": (1, ()),  #                          A           -> ∅
    "DELETE_DEREF": (0, ()),  #                         ∅           -> ∅

    # Functions and calls                               A,B,...     -> Z
    "CALL_FUNCTION": lambda x: (x + 1, PushNew),
    "CALL_METHOD": lambda x: (x + 2, PushNew),
    "CALL_FUNCTION_KW": ...,
    "CALL_FUNCTION_EX": function_detail(0x01),
    "MAKE_FUNCTION": function_detail(0x01, 0x02, 0x04, 0x08),

    # Build containers                                  A,B,... -> Z
    "BUILD_TUPLE": lambda oparg: (oparg, PushNew),
    "BUILD_LIST":   ...,
    "BUILD_SET":    ...,
    "BUILD_STRING": ...,
    "BUILD_MAP": lambda oparg: (oparg * 2, PushNew),
    "BUILD_CONST_KEY_MAP": lambda x: (x + 1, PushNew),
    "LIST_TO_TUPLE": (1, PushNew),  #                   A       -> B

    # Insertion leaves container on the stack           A,B     -> A
    "SET_ADD": (2, (-2,)),
    "SET_UPDATE": ...,
    "LIST_APPEND": ...,
    "LIST_EXTEND": ...,
    "MAP_ADD": ...,
    "DICT_MERGE": ...,
    "DICT_UPDATE": ...,
    "COPY_DICT_WITHOUT_KEYS": (2, (-2, 0)),  #          A,B     -> A,C  (I am unsure...)

    # Unary operators                                   A       -> B
    "UNARY_POSITIVE": (1, PushNew),
    "UNARY_NEGATIVE": ...,
    "UNARY_NOT": ...,
    "UNARY_INVERT": ...,

    # Binary operators                                  A,B     -> C
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

    # Binary operators (inplace)
    #   https://docs.python.org/3/reference/datamodel.html?highlight=iadd#object.__iadd__
    #   "... and return the result (which could be, but does not have to be, self)."
    "INPLACE_POWER": (2, PushNew),
    "INPLACE_LSHIFT": ...,
    "INPLACE_RSHIFT": ...,
    "INPLACE_AND": ...,
    "INPLACE_XOR": ...,
    "INPLACE_OR": ...,

    # Indexing operators
    "STORE_SUBSCR": (3, ()),  #                 A,B,C   -> ∅
    "DELETE_SUBSCR": (2, ()),  #                A,B     -> ∅
    "BUILD_SLICE": lambda x: (x, PushNew),  #   A,B,... -> Z
    "UNPACK_EX": lambda x: (1, tuple(range((x & 0xFF) + (x >> 8) + 1))),  # A   -> B,C,...

    # Iterators                                 A       -> B
    "GET_ITER": (1, PushNew),
    "GET_YIELD_FROM_ITER": (1, PushNew),

    # Misc.
    "FORMAT_VALUE": format_value,  #            (A?),B  -> C
    "PRINT_EXPR": (1, ()),  #                   A       -> ∅
    "IMPORT_STAR": ...,
    "LOAD_BUILD_CLASS": (0, PushNew),
    "SETUP_ANNOTATIONS": (0, ()),
    "GET_LEN": (1, (-1, 0)),
    "IMPORT_NAME": (2, PushNew),
    "IMPORT_FROM": (1, (-1, 0)),

    "MATCH_CLASS": (3, PushNew),
    "MATCH_MAPPING": (1, (TOS, 0)),
    "MATCH_SEQUENCE": ...,
    "MATCH_KEYS": (2, (-1, -2, 0) + () if sys.version_info >= (3, 11) else (1,)),

    # TODO(robieta, t-vi): Iterators and generators
    "GEN_START": (1, ()),  #   Where does TOS for this come from?
    "YIELD_VALUE": (1, PushNew),  # I think
    "YIELD_FROM": (2, PushNew),  #  I am very unsure
    # "GET_AWAITABLE": (1, 1),
    # "BEFORE_ASYNC_WITH": (1, 2),
    # "GET_AITER": (1, 1),
    # "GET_ANEXT": (1, 2),
    # "END_ASYNC_FOR": (7, 0),
}

FIXED_STACK_EFFECTS_DETAIL: Dict[str, FixedEffect] = {}
SIMPLE_VARIABLE_STACK_EFFECTS_DETAIL: Dict[str, Callable[[int], FixedEffect]] = {}

# Unpack `_STACK_EFFECTS_SPEC` into constituent maps.
__prior_effect = Ellipsis
for __opname, __effect in _STACK_EFFECTS_SPEC.items():
    if __effect is Ellipsis:
        __effect = __prior_effect
    if __effect is Ellipsis:
        logger.warn("Invalid effect for opname %s", __opname)
    elif isinstance(__effect, tuple):
        FIXED_STACK_EFFECTS_DETAIL[__opname] = __effect
    elif callable(__effect):
        SIMPLE_VARIABLE_STACK_EFFECTS_DETAIL[__opname] = __effect
    else:
        logger.warn("Unhandled opname %s: %s", __opname, __effect)
    __prior_effect = __effect

del __prior_effect, __opname, __effect


def stack_effects_comprehensive(instruction: dis.Instruction) -> Tuple[
    int,  #                             Number of values popped
    Tuple[int, ...],  #                 Values pushed (unconditional)
    Tuple[bool, Tuple[int, ...]],  #    (branch w/ extra, values)
]:
    opname: str = instruction.opname
    oparg: Optional[int] = instruction.arg

    if (effect := FIXED_STACK_EFFECTS_DETAIL.get(opname)) is not None:
        pop, push = effect
        return pop, push, NoBranchDependent
    
    elif (effect_fn := SIMPLE_VARIABLE_STACK_EFFECTS_DETAIL.get(opname)) is not None:
        assert oparg is not None
        pop, push = effect_fn(oparg)
        return pop, push, NoBranchDependent

    elif opname == "FOR_ITER":
        return 1, (), (False, (TOS, 0))

    elif opname in ("JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP"):
        # NOTE: These instructions have been removed since they are extraneous special cases.
        #       https://github.com/faster-cpython/ideas/issues/567
        #       https://github.com/python/cpython/issues/102859
        return 1, (), (True, (TOS,))

    elif opname == "SETUP_WITH":
        return 1, (0, 1), (True, (2, 3, 4, 5, 6))

    elif opname == "SETUP_FINALLY":
        return 0, (), (True, (0, 1, 2, 3, 4, 5))

    elif opname == "SETUP_ASYNC_WITH":
        return 1, (0,), (True, (1, 2, 3, 4, 5))  # ??

    raise ValueError(f"Invalid opname {opname}")


def stack_effect_detail(instruction: dis.Instruction, *, jump: bool = False) -> Tuple[int, int]:
    pop, unconditional, (branch, conditional) = stack_effects_comprehensive(instruction)
    return pop, len(unconditional) + len(conditional if branch == jump else ())


jump_instructions = set(dis.hasjabs) | set(dis.hasjrel)


def make_jump_absolute(arg: int) -> dis.Instruction:
    return dis.Instruction(
        opname="JUMP_ABSOLUTE",
        opcode=dis.opmap.get("JUMP_ABSOLUTE", -1),
        arg=arg,
        argval=None,
        argrepr=f"{arg}",
        offset=-999,
        starts_line=None,
        is_jump_target=False,
    )

def make_return(is_jump_target: bool) -> dis.Instruction:
    return dis.Instruction(
        opname="RETURN_VALUE",
        opcode=dis.opmap["RETURN_VALUE"],
        arg=None,
        argval=None,
        argrepr="",
        offset=-999,
        starts_line=None,
        is_jump_target=is_jump_target,
    )


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


return_instructions = {dis.opmap[name] for name in ("RETURN_VALUE", "RAISE_VARARGS", "RERAISE") if name in dis.opmap}
unconditional_jump_names = {"JUMP_ABSOLUTE", "JUMP_FORWARD", "JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"}

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

def make_name_map(bytecode: Iterable[dis.Instruction], code: CodeType) -> Dict[ArgScope, "str"]:
    name_sources = {
        VariableScope.LOCAL: code.co_varnames,
        VariableScope.NONLOCAL: code.co_freevars,
        VariableScope.GLOBAL: code.co_names,
    }
    scope_map = dict_join(store_opcodes, load_opcodes, del_opcodes)
    keys = [ArgScope(i.arg, scope) for i in bytecode if (scope := scope_map.get(i.opname)) in name_sources]
    assert not any(arg is None for arg, _ in keys)

    return {key: name_sources[key.scope][key.arg] for key in sorted(keys)}
