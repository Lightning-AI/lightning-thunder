"""Convert a `CodeType` object into a series of simple blocks."""
from __future__ import annotations

import dis
import itertools
from types import CodeType
from typing import Any, NewType, TypeVar
from collections.abc import Iterable, Mapping

from thunder.core.script.parse import stack_effect
from thunder.core.script.parse.instructions import *  # There are a lot of constants, and it defines `__all__`


__all__ = ("parse_bytecode", "ParseDetailInstruction", "EpilogueFixup", "Jump", "Edges")

OrderedBlocks = tuple[tuple["ThunderInstruction", ...], ...]
BlockIdx = NewType("BlockIdx", int)
Jump = NewType("Jump", bool)
Edges = tuple[tuple[BlockIdx, BlockIdx, Jump], ...]


class ParseDetailInstruction(ThunderInstruction):
    """Allow us to distinguish instructions that are added during parsing."""

    pass


class EpilogueFixup(ParseDetailInstruction):
    pass


def compute_jump(instruction: ThunderInstruction, position: int) -> int | None:
    if instruction in ABSOLUTE_JUMP_INSTRUCTIONS:
        return instruction.oparg

    elif instruction in UNCONDITIONAL_BACKWARD:
        return position + 1 - instruction.oparg

    elif "BACKWARD" in instruction.opname:
        # TODO: POP_JUMP_BACKWARD_IF_... variants
        raise NotImplementedError(instruction.opname)

    elif instruction in RELATIVE_JUMP_INSTRUCTIONS:
        return position + 1 + instruction.oparg

    return None


IntT = TypeVar("IntT", bound=int, covariant=True)
StartIndex = NewType("StartIndex", int)
RawBlocks = dict[StartIndex, tuple[ThunderInstruction, ...]]


def get_free_key(x: Mapping[IntT, Any]) -> int:
    key = -len(x)
    while key in x:
        key -= 1
    return key


def partition(co: CodeType) -> tuple[RawBlocks, int]:
    bytecode = tuple(ThunderInstruction(*i) for i in dis.get_instructions(co, first_line=0))

    # Determine the boundaries for the simple blocks.
    split_after = JUMP_INSTRUCTIONS | RETURN_INSTRUCTIONS
    follows_jump = itertools.chain([0], (int(i in split_after) for i in bytecode))
    new_block = (int(i or j.is_jump_target) for i, j in zip(follows_jump, bytecode))

    # Split the bytecode (and instruction number) into groups
    group_indices = tuple(itertools.accumulate(new_block))
    groups = itertools.groupby(enumerate(bytecode), key=lambda args: group_indices[args[0]])

    # Drop the group index, copy from the groupby iter, and unzip `enumerate`.
    groups = (zip(*tuple(i)) for _, i in groups)
    blocks: dict[StartIndex, list[ThunderInstruction]] = {
        StartIndex(start): list(block) for (start, *_), block in groups
    }

    # If the last instruction is not a jump or return (which means we split
    # because the next instruction was a jump target) then we need to tell
    # the current block how to advance.
    for start, block in blocks.items():
        if block[-1] not in split_after:
            next_start = StartIndex(start + len(block))
            assert bytecode[next_start].is_jump_target
            block.append(ParseDetailInstruction.make_jump_absolute(next_start))

    line_no = 1
    for instruction in itertools.chain(*[block for block in blocks.values()]):
        instruction.line_no = line_no = instruction.starts_line or line_no

    return {k: tuple(v) for k, v in blocks.items()}, line_no


def consolidate_returns(blocks: RawBlocks) -> RawBlocks:
    def is_return(block: tuple[ThunderInstruction, ...]) -> bool:
        assert block and not any(i.opname == RETURN_VALUE for i in block[:-1])
        return block[-1].opname == RETURN_VALUE

    blocks = blocks.copy()
    return_blocks = {k: v for k, v in blocks.items() if is_return(v)}
    if len(return_blocks) > 1:
        new_return_start = StartIndex(get_free_key(blocks))
        for start, (*body, prior_return) in return_blocks.items():
            assert is_return((prior_return,)), prior_return
            blocks[start] = (*body, ParseDetailInstruction.make_jump_absolute(new_return_start))
        return_blocks = {new_return_start: (ParseDetailInstruction.make_return(is_jump_target=True),)}

    # Move return block to the end. This isn't always valid (since a block might
    # expect to fall through and reach it), but that will be resolved by the
    # sort in `ProtoGraph`'s ctor.
    blocks = {k: v for k, v in blocks.items() if k not in return_blocks}
    blocks.update(return_blocks)
    return blocks


def connect_blocks(blocks: RawBlocks) -> tuple[OrderedBlocks, Edges]:
    def iter_raw_edges(blocks: RawBlocks) -> Iterable[tuple[StartIndex, StartIndex, Jump, int, int]]:
        for start, block in tuple(blocks.items()):
            raw_block_len = sum(not isinstance(i, ParseDetailInstruction) for i in block)
            *_, last_i = block
            if last_i in JUMP_INSTRUCTIONS:
                end = start + raw_block_len - 1
                _, (push_nojump, push_jump) = stack_effect.stack_effect_detail(last_i)
                if last_i not in UNCONDITIONAL_JUMP_INSTRUCTIONS:
                    yield start, StartIndex(end + 1), Jump(False), max(push_jump - push_nojump, 0), last_i.line_no

                if (jump_offset := compute_jump(last_i, end)) is not None:
                    yield start, StartIndex(jump_offset), Jump(True), max(push_nojump - push_jump, 0), last_i.line_no

    blocks = blocks.copy()
    edges: list[tuple[StartIndex, StartIndex, Jump]] = []
    for source, destination, jump, pop_suffix, line_no in iter_raw_edges(blocks):
        if pop_suffix:
            blocks[epilogue := StartIndex(get_free_key(blocks))] = (
                *(EpilogueFixup.make(POP_TOP, None, line_no=line_no) for _ in range(pop_suffix)),
                EpilogueFixup.make_jump_absolute(destination, line_no=line_no),
            )
            edges.extend(((source, epilogue, jump), (epilogue, destination, jump)))
        else:
            edges.append((source, destination, jump))

    to_idx = {k: BlockIdx(idx) for idx, k in enumerate(blocks.keys())}
    return tuple(blocks.values()), tuple((to_idx[source], to_idx[sink], jump) for source, sink, jump in edges)


def parse_bytecode(co: CodeType) -> tuple[OrderedBlocks, Edges]:
    raw_blocks, last_line_no = partition(co)
    raw_blocks = consolidate_returns(raw_blocks)
    blocks, edges = connect_blocks(raw_blocks)
    for instruction in itertools.chain(*blocks):
        instruction.line_no = getattr(instruction, "line_no", last_line_no)
    return blocks, edges
