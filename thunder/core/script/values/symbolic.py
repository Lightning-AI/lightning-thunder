"""Introduce references inside simple blocks."""
from __future__ import annotations

import dataclasses
import itertools
import sys
from typing import Any, NamedTuple, TypeAlias
from collections.abc import Callable
from collections.abc import Iterable, Iterator

from typing_extensions import Self

from thunder.core.script import parse
from thunder.core.script.values import base
from thunder.core.utils import FrozenDict, safe_zip

__all__ = ("OutputRef", "make_symbolic", "Symbolic", "NestedReference", "ConstRef")


# =============================================================================
# == Opcode-specific behavior =================================================
# =============================================================================
def rotate_N(oparg: int) -> tuple[int, ...]:
    return (-1,) + tuple(range(-oparg, -1))


_AliasMask = tuple[int | None, ...]
ALIAS_OPCODES = FrozenDict[str, _AliasMask | Callable[[int], _AliasMask]](
    parse.fill_ellipses(
        #
        # Stack manipulation
        ROT_N=rotate_N,  #                              A,B,...,Z   -> Z,A,B,...
        ROT_FOUR=rotate_N,
        ROT_THREE=rotate_N,
        ROT_TWO=rotate_N,
        DUP_TOP=(-1, -1),  #                            A           -> A,A
        DUP_TOP_TWO=(-2, -1) * 2,  #                    A,B         -> A,B,A,B
        #
        # Insertion leaves container on the stack       A,B         -> A
        SET_ADD=(-2,),
        SET_UPDATE=...,
        LIST_APPEND=...,
        LIST_EXTEND=...,
        DICT_MERGE=...,
        DICT_UPDATE=...,
        MAP_ADD=(-3,),
        COPY_DICT_WITHOUT_KEYS=(-2, None),  #           A,B         -> A,C  (I am unsure...)
        #
        # Misc.
        GET_LEN=(-1, None),
        MATCH_MAPPING=(-1, None),
        MATCH_SEQUENCE=...,
        MATCH_KEYS=(-1, -2, None) + () if sys.version_info >= (3, 11) else (None,),
        #
        # Jump dependent
        FOR_ITER=(-1, None),
        # NOTE: These instructions have been removed since they are extraneous special cases.
        #       https://github.com/faster-cpython/ideas/issues/567
        #       https://github.com/python/cpython/issues/102859
        JUMP_IF_TRUE_OR_POP=(-1,),
        JUMP_IF_FALSE_OR_POP=(-1,),
        #
        # This isn't actually correct. `LOAD_METHOD` will return either
        #   A -> B, A
        #   A -> B, NULL
        # However the `A | NULL` is only consumed by `CALL_METHOD`, so it's ok to use this alias.
        LOAD_METHOD=(None, -1),  #                      A           -> B,A
    )
)


# =============================================================================
# == Symbolic flow ============================================================
# =============================================================================
IndexT: TypeAlias = base.Reference | base.TraitName
NestedReference: TypeAlias = IndexT | tuple[IndexT, ...]


@dataclasses.dataclass(frozen=True, eq=True)
class OutputRef:
    """Identifies the producer of a value within a block."""

    instruction: parse.ThunderInstruction  # Acts as a key for the producer Flow.
    idx: NestedReference  #                  Indexes the producer's outputs.


class ConstRef(NamedTuple):
    """Convenience wrapper to access `ExternalRef(VariableKey(..., CONST))` as a reference.

    This saves us from having to plumb constants through named inputs, since:
        A)  `ExternalRef`s cannot appear in Symbolic outputs.
            (Since that implies a producer relationship which doesn't make sense.)
        B)  Symbolic reference outputs must reference an input, which would mean an entry of
            `{some_random_name: VariableKey(..., CONST)}` would have to be added to named inputs
            which is tedious.
    """

    identifier: Any


@dataclasses.dataclass(frozen=True, eq=False)
class Symbolic:
    """Represents abstract flow immediately after functionalization."""

    # VariableKey:          References the value of that variable at the start of the block
    # OutputRef:            Reference values created by an earlier instruction within the block
    # SingletonValue.Tag:   Reserved for special cases.
    Input = parse.VariableKey | OutputRef | base.NonPyObject.Tag
    inputs: base.HybridMap[Input]

    # NestedReference:      Aliases the input at this position.
    # AbstractValue:        New value created by this instruction
    Output = NestedReference | ConstRef | base.AbstractValue
    outputs: tuple[Output, ...]

    BeginState = FrozenDict[parse.VariableKey, base.AbstractValue]
    EndState = FrozenDict[parse.VariableKey, Input]
    Block = tuple[FrozenDict[parse.ThunderInstruction, "Symbolic"], BeginState, EndState]

    def __post_init__(self) -> None:
        # If an `AbstractValue` appears in `Symbolic.outputs` that implies that the symbolic
        # node in question is the value's producer. However it doesn't make sense for an external
        # value to be produced within the compiled function.
        assert not any(isinstance(o, base.ExternalRef) for o in self.outputs), self

    @property
    def uses(self) -> Iterable[parse.VariableKey]:
        """Block inputs used by this node.

        NOTE: This does not include values produced by an earlier node in the block.
        """
        for i in itertools.chain(self.inputs.ordered, self.inputs.named.values()):
            if isinstance(i, parse.VariableKey) and not i.is_const:
                yield i

    def substitute(self, replace_map: base.ReplaceMap) -> Self:
        outputs = tuple(base.substitute_value(o, replace_map) for o in self.outputs)
        return dataclasses.replace(self, outputs=outputs)


# =============================================================================
# == Conversion from parsed representation ====================================
# =============================================================================
def make_symbolic(blocks: tuple[parse.FunctionalizedBlock, ...]) -> Iterator[Symbolic.Block]:
    for block, begin_state, end_state in blocks:
        # `functionalize_blocks` produces unique values, so provenance is unambiguous.
        producers: dict[parse.PlaceholderValue | None, Symbolic.Input] = {v: k for k, v in begin_state.items()}
        producers[None] = base.NonPyObject.Tag.DELETED
        assert len(producers) == len(begin_state) + 1, (producers, end_state)

        symbolic_blocks: dict[parse.ThunderInstruction, Symbolic] = {}
        for instruction, raw_inputs, raw_outputs in block:
            for idx, o in enumerate(raw_outputs):
                assert o not in producers
                producers[o] = OutputRef(instruction, base.Reference(idx))

            outputs: tuple[Symbolic.Output, ...] = tuple(base.IntermediateValue() for _ in raw_outputs)
            if alias := ALIAS_OPCODES.get(instruction.opname):
                mask = alias(len(outputs)) if callable(alias) else alias
                mask = (base.Reference(i) if i is not None else i for i in mask)
                outputs = tuple(o if o_mask is None else o_mask for o, o_mask in safe_zip(outputs, mask))
            inputs = base.HybridMap(ordered=tuple(producers[i] for i in raw_inputs))
            symbolic_blocks[instruction] = Symbolic(inputs, outputs)

        begin = {k: base.AbstractRef(v) for k, v in begin_state.items() if not k.is_const}
        end = {k: producers[v] for k, v in end_state.items() if not k.is_const}
        yield (FrozenDict(symbolic_blocks), FrozenDict(begin), FrozenDict(end))
