"""Extension of the builtin `dis` module."""
from __future__ import annotations

import dis
from typing import Any

from typing_extensions import Self

from thunder.core.utils import _OrderedSet

__all__ = (
    "ThunderInstruction",
    "InstructionSet",
    "JUMP_ABSOLUTE",
    "RETURN_VALUE",
    "POP_TOP",
    "EXTENDED_ARG",
    "UNCONDITIONAL_BACKWARD",
    "UNCONDITIONAL_JUMP_INSTRUCTIONS",
    "ABSOLUTE_JUMP_INSTRUCTIONS",
    "RELATIVE_JUMP_INSTRUCTIONS",
    "JUMP_INSTRUCTIONS",
    "RAISE_RETURN_INSTRUCTIONS",
    "RETURN_INSTRUCTIONS",
    "UNSAFE_OPCODES",
)


class ThunderInstruction(dis.Instruction):
    """Thin wrapper on top of dis.Instruction to implement thunder specific logic."""

    line_no: int

    def __hash__(self) -> int:
        # We sometimes want to use an instruction as a key so we can map back to nodes.
        # `dis.Instruction` is a named tuple and therefore implements recursive constituent
        # hashing which can lead to unwanted collisions. We instead override this behavior
        # to instead use identity hashing.
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def oparg(self) -> int:
        assert self.arg is not None, self
        return self.arg

    def modify_copy(self, **kwargs: Any) -> ThunderInstruction:
        assert type(self) is ThunderInstruction, self
        if "opname" in kwargs:
            kwargs.setdefault("opcode", dis.opmap.get(kwargs["opname"], -1))
        result = ThunderInstruction(**{**self._asdict(), **kwargs})
        result.line_no = self.line_no
        return result

    @classmethod
    def make(cls, opname: str, arg: int | None, line_no: int, **kwargs: Any) -> Self:
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
        result = cls(**ctor_kwargs)  # type: ignore
        result.line_no = line_no
        return result

    @classmethod
    def make_jump_absolute(cls, arg: int, line_no: int = -1) -> ThunderInstruction:
        return cls.make(JUMP_ABSOLUTE, arg, argrepr=f"{arg}", line_no=line_no)

    @classmethod
    def make_return(cls, is_jump_target: bool, line_no: int = -1) -> ThunderInstruction:
        return cls.make(RETURN_VALUE, arg=None, argrepr="", is_jump_target=is_jump_target, line_no=line_no)


class InstructionSet(_OrderedSet[str, int | ThunderInstruction]):
    """Convenience class for checking opcode properties."""

    def canonicalize(self, i: str | int | ThunderInstruction) -> str:
        if isinstance(i, str):
            return i

        elif isinstance(i, int):
            return dis.opname[i]

        else:
            assert isinstance(i, ThunderInstruction)
            return i.opname


# Special opcodes
JUMP_ABSOLUTE = "JUMP_ABSOLUTE"
RETURN_VALUE = "RETURN_VALUE"
POP_TOP = "POP_TOP"
EXTENDED_ARG = "EXTENDED_ARG"


UNCONDITIONAL_BACKWARD = InstructionSet(("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"))
UNCONDITIONAL_JUMP_INSTRUCTIONS = InstructionSet((JUMP_ABSOLUTE, "JUMP_FORWARD", *UNCONDITIONAL_BACKWARD))

ABSOLUTE_JUMP_INSTRUCTIONS = InstructionSet(dis.hasjabs)
RELATIVE_JUMP_INSTRUCTIONS = InstructionSet(dis.hasjrel)
JUMP_INSTRUCTIONS = InstructionSet((*dis.hasjabs, *dis.hasjrel, *UNCONDITIONAL_JUMP_INSTRUCTIONS))

RAISE_RETURN_INSTRUCTIONS = InstructionSet(("RAISE_VARARGS", "RERAISE"))
RETURN_INSTRUCTIONS = InstructionSet((RETURN_VALUE, *RAISE_RETURN_INSTRUCTIONS))


# https://github.com/Lightning-AI/lightning-thunder/issues/1075
UNSAFE_OPCODES = InstructionSet(("SETUP_WITH", "SETUP_FINALLY"))
