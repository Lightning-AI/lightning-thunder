import dis
from typing import Tuple, Optional

# this is Python 3.10 specific for the time being.

#  *  0 -- when not jump
#  *  1 -- when jump
#  * -1 -- maximal

# input, output probably would be smart to highlight inplace mods and global side effects
# (e.g. setup_annotations, import_star), too
fixed_stack_effects_detail = {
    "NOP": (0, 0),
    "EXTENDED_ARG": (0, 0),
    # Stack manipulation
    "POP_TOP": (1, 0),
    "ROT_TWO": (2, 2),
    "ROT_THREE": (3, 3),
    "ROT_FOUR": (4, 4),
    "DUP_TOP": (1, 2),
    "DUP_TOP_TWO": (2, 4),
    # Unary operators
    "UNARY_POSITIVE": (1, 1),
    "UNARY_NEGATIVE": (1, 1),
    "UNARY_NOT": (1, 1),
    "UNARY_INVERT": (1, 1),
    "SET_ADD": (2, 1),  # these leave the container on the stack
    "LIST_APPEND": (2, 1),
    "MAP_ADD": (3, 1),
    # Binary operators
    "BINARY_POWER": (2, 1),
    "BINARY_MULTIPLY": (2, 1),
    "BINARY_MATRIX_MULTIPLY": (2, 1),
    "BINARY_MODULO": (2, 1),
    "BINARY_ADD": (2, 1),
    "BINARY_SUBTRACT": (2, 1),
    "BINARY_SUBSCR": (2, 1),
    "BINARY_FLOOR_DIVIDE": (2, 1),
    "BINARY_TRUE_DIVIDE": (2, 1),
    "INPLACE_FLOOR_DIVIDE": (2, 1),
    "INPLACE_TRUE_DIVIDE": (2, 1),
    "INPLACE_ADD": (2, 1),
    "INPLACE_SUBTRACT": (2, 1),
    "INPLACE_MULTIPLY": (2, 1),
    "INPLACE_MATRIX_MULTIPLY": (2, 1),
    "INPLACE_MODULO": (2, 1),
    "BINARY_LSHIFT": (2, 1),
    "BINARY_RSHIFT": (2, 1),
    "BINARY_AND": (2, 1),
    "BINARY_XOR": (2, 1),
    "BINARY_OR": (2, 1),
    "INPLACE_POWER": (2, 1),
    "INPLACE_LSHIFT": (2, 1),
    "INPLACE_RSHIFT": (2, 1),
    "INPLACE_AND": (2, 1),
    "INPLACE_XOR": (2, 1),
    "INPLACE_OR": (2, 1),
    "STORE_SUBSCR": (3, 0),
    "DELETE_SUBSCR": (2, 0),
    "GET_ITER": (1, 1),
    "PRINT_EXPR": (1, 0),
    "LOAD_BUILD_CLASS": (0, 1),
    "RETURN_VALUE": (1, 0),
    "IMPORT_STAR": (1, 0),
    "SETUP_ANNOTATIONS": (0, 0),
    "YIELD_VALUE": (1, 1),  # I think
    "YIELD_FROM": (2, 1),  # I am very unsure
    "POP_BLOCK": (0, 0),
    "POP_EXCEPT": (3, 0),
    "STORE_NAME": (1, 0),
    "DELETE_NAME": (0, 0),
    "STORE_ATTR": (2, 0),
    "DELETE_ATTR": (1, 0),
    "STORE_GLOBAL": (1, 0),
    "DELETE_GLOBAL": (0, 0),
    "LOAD_CONST": (0, 1),
    "LOAD_NAME": (0, 1),
    "LOAD_ATTR": (1, 1),
    "COMPARE_OP": (2, 1),
    "IS_OP": (2, 1),
    "CONTAINS_OP": (2, 1),
    "JUMP_IF_NOT_EXC_MATCH": (2, 0),
    "IMPORT_NAME": (2, 1),
    "IMPORT_FROM": (1, 2),
    # Jumps
    "JUMP_FORWARD": (0, 0),
    "JUMP_ABSOLUTE": (0, 0),
    "POP_JUMP_IF_FALSE": (1, 0),
    "POP_JUMP_IF_TRUE": (1, 0),
    "LOAD_GLOBAL": (0, 1),
    "RERAISE": (3, 0),
    "WITH_EXCEPT_START": (7, 8),  # ??!?
    "LOAD_FAST": (0, 1),
    "STORE_FAST": (1, 0),
    "DELETE_FAST": (0, 0),
    # Closures
    "LOAD_CLOSURE": (0, 1),
    "LOAD_DEREF": (0, 1),
    "LOAD_CLASSDEREF": (0, 1),
    "STORE_DEREF": (1, 0),
    "DELETE_DEREF": (0, 0),
    # Iterators and generators
    "GET_AWAITABLE": (1, 1),
    "BEFORE_ASYNC_WITH": (1, 2),
    "GET_AITER": (1, 1),
    "GET_ANEXT": (1, 2),
    "GET_YIELD_FROM_ITER": (1, 1),
    "END_ASYNC_FOR": (7, 0),
    "LOAD_METHOD": (1, 2),
    "LOAD_ASSERTION_ERROR": (0, 1),
    "LIST_TO_TUPLE": (1, 1),
    "GEN_START": (1, 0),
    "LIST_EXTEND": (2, 1),
    "SET_UPDATE": (2, 1),
    "DICT_MERGE": (2, 1),
    "DICT_UPDATE": (2, 1),
    "COPY_DICT_WITHOUT_KEYS": (2, 2),
    "MATCH_CLASS": (3, 2),
    "GET_LEN": (1, 2),
    "MATCH_MAPPING": (1, 2),
    "MATCH_SEQUENCE": (1, 2),
    "MATCH_KEYS": (2, 4),
}


def stack_effect_detail(opname: str, oparg: Optional[int], *, jump: bool = False) -> Tuple[int, int]:
    if opname in fixed_stack_effects_detail:
        return fixed_stack_effects_detail[opname]
    if opname == "ROT_N":
        assert oparg is not None
        return (oparg, oparg)
    if opname in {"BUILD_TUPLE", "BUILD_LIST", "BUILD_SET", "BUILD_STRING"}:
        assert oparg is not None
        return (oparg, 1)
    if opname == "BUILD_MAP":
        assert oparg is not None
        return (2 * oparg, 1)
    if opname == "BUILD_CONST_KEY_MAP":
        assert oparg is not None
        return (oparg + 1, 1)
    if opname in {"JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP"}:
        return (1, 1) if jump else (1, 0)
    if opname == "SETUP_FINALLY":
        return (0, 6) if jump else (0, 0)
    # Exception handling
    if opname == "RAISE_VARARGS":
        assert oparg is not None
        return (oparg, 0)
    # Functions and calls
    if opname == "CALL_FUNCTION":
        assert oparg is not None
        return (oparg + 1, 1)
    if opname == "CALL_METHOD":
        assert oparg is not None
        return (oparg + 2, 1)
    if opname == "CALL_FUNCTION_KW":
        assert oparg is not None
        return (oparg + 2, 1)
    if opname == "CALL_FUNCTION_EX":
        assert oparg is not None
        return (2 + ((oparg & 0x01) != 0), 1)
    if opname == "MAKE_FUNCTION":
        assert oparg is not None
        return (
            2 + ((oparg & 0x01) != 0) + ((oparg & 0x02) != 0) + ((oparg & 0x04) != 0) + ((oparg & 0x08) != 0),
            1,
        )
    if opname == "BUILD_SLICE":
        assert oparg is not None
        return (oparg, 1)
    if opname == "SETUP_ASYNC_WITH":
        return (1, 6) if jump else (1, 1)  # ??
    if opname == "SETUP_WITH":
        return (1, 7) if jump else (1, 2)
    if opname == "FORMAT_VALUE":
        assert oparg is not None
        return (2, 1) if ((oparg & 0x04) != 0) else (1, 1)
    if opname == "UNPACK_SEQUENCE":
        assert oparg is not None
        return (1, oparg)
    if opname == "UNPACK_EX":
        assert oparg is not None
        return (1, (oparg & 0xFF) + (oparg >> 8) + 1)
    if opname == "FOR_ITER":
        return (1, 0) if jump else (1, 2)

    raise ValueError(f"Invalid opname {opname}")


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
