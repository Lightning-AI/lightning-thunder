"""Replay the CPython stack machine to determine data flow within a simple block."""
from __future__ import annotations

import dataclasses
import enum
import inspect
import itertools
import marshal
import textwrap
from types import CodeType
from typing import Any, NamedTuple, NewType

import networkx as nx

from thunder.core.script import algorithms
from thunder.core.script.parse import disassemble, instructions, stack_effect
from thunder.core.utils import safe_zip, FrozenDict, InferringDict

__all__ = ("VariableScope", "VariableKey", "ParsedFunctional", "FunctionalizedBlock", "PlaceholderValue")


class VariableScope(enum.Enum):
    CONST = enum.auto()
    LOCAL = enum.auto()
    NONLOCAL = enum.auto()
    GLOBAL = enum.auto()
    STACK = enum.auto()


class VariableKey(NamedTuple):
    """Denotes the location of a variable.
    For example, `x = 5` assigns the variable stored in `VariableKey(5, VariableScope.CONST)`
    to the location `VariableKey("x", VariableScope.LOCAL)`. (Provided `x` is a local variable.)
    The type of `identifier` varies based on `scope`:
        `marshal`able   VariableScope.CONST
        str             VariableScope.LOCAL / NONLOCAL / GLOBAL
        int             VariableScope.STACK
        Any             VariableScope.BOUNDARY
    """

    identifier: Any
    scope: VariableScope

    def __repr__(self) -> str:
        return f"VariableKey({self.identifier}, scope={self.scope.name})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, VariableKey)
            and self.scope == other.scope
            and type(self.identifier) is (_ := type(other.identifier))  # Conflict between `ruff` and `yesqa`
            and self.identifier == other.identifier
        )

    def __lt__(self, other: tuple[Any, ...]) -> bool:
        assert isinstance(other, VariableKey), (self, other)
        try:
            return (self.scope.value, self.identifier) < (other.scope.value, other.identifier)
        except TypeError:
            assert self.scope == other.scope, (self, other)
            if self.scope == VariableScope.CONST:
                # We prefer to use native ordering. However for unorderable types (e.g. CodeType)
                # `marshal` at least provides a consistent ordering.
                return marshal.dumps(self.identifier) < marshal.dumps(other.identifier)
            raise

    @property
    def is_const(self) -> bool:
        return self.scope == VariableScope.CONST


def _compute_stack_offsets(disassembled: disassemble.Disassembled) -> tuple[int, ...]:
    # If we convert the stack indices to a common basis then we can ignore stack effect and
    # treat VariableScope.STACK variables just like any other local.
    G = algorithms.TypedDiGraph[disassemble.BlockIdx]((i, j) for i, j, _ in disassembled.edges)
    G.add_nodes_from(range(len(disassembled.blocks)))
    offsets: dict[disassemble.BlockIdx, int] = {i: 0 for i in G.nodes if not G.in_degree(i)}  # type: ignore[misc]
    assert len(offsets) == 1, G

    for source, sink in nx.edge_dfs(G):
        net_stack_effect = 0
        for instruction in disassembled.blocks[source]:
            pop, push_by_branch = stack_effect.stack_effect_detail(instruction)
            net_stack_effect += max(push_by_branch) - pop
        expected = offsets[source] + net_stack_effect
        actual = offsets.setdefault(sink, expected)
        assert actual == expected, (actual, expected)

    assert all(v >= 0 for v in offsets.values()), offsets
    return tuple(offsets[disassemble.BlockIdx(i)] for i in range(len(disassembled.blocks)))


LOAD_OPNAMES = FrozenDict[str, VariableScope](
    LOAD_CONST=VariableScope.CONST,
    LOAD_FAST=VariableScope.LOCAL,
    LOAD_DEREF=VariableScope.NONLOCAL,
    LOAD_CLOSURE=VariableScope.NONLOCAL,
    LOAD_GLOBAL=VariableScope.GLOBAL,
)

STORE_OPNAMES = FrozenDict[str, VariableScope](
    STORE_FAST=VariableScope.LOCAL,
    STORE_DEREF=VariableScope.NONLOCAL,
    STORE_GLOBAL=VariableScope.GLOBAL,
)

DEL_OPNAMES = FrozenDict[str, VariableScope](
    DELETE_FAST=VariableScope.LOCAL,
    DELETE_DEREF=VariableScope.NONLOCAL,
    DELETE_GLOBAL=VariableScope.GLOBAL,
)

PlaceholderValue = NewType("PlaceholderValue", str)
Inputs = NewType("Inputs", tuple[PlaceholderValue, ...])
Outputs = NewType("Outputs", tuple[PlaceholderValue, ...])
BeginState = FrozenDict["VariableKey", PlaceholderValue]
EndState = FrozenDict["VariableKey", PlaceholderValue | None]
FunctionalNode = tuple[instructions.ThunderInstruction, Inputs, Outputs]
FunctionalizedBlock = NewType("FunctionalizedBlock", tuple[tuple[FunctionalNode, ...], BeginState, EndState])


@dataclasses.dataclass(frozen=True)
class ParsedFunctional:
    blocks: tuple[FunctionalizedBlock, ...]
    provenance: disassemble.Disassembled

    @staticmethod
    def make(co: CodeType) -> ParsedFunctional:
        disassembled = disassemble.Disassembled.make(co)
        return ParsedFunctional(_functionalize_blocks(disassembled), disassembled)

    @property
    def summary(self) -> str:
        return _summarize(self)


def _functionalize_blocks(disassembled: disassemble.Disassembled) -> tuple[FunctionalizedBlock, ...]:
    code = disassembled.code
    errors: list[str] = []
    if code.co_cellvars:
        errors.append(
            "Nonlocal variables are not supported but\n"
            f"  {code.co_name}() defined in {code.co_filename}:{code.co_firstlineno}\n"
            f"  defines nonlocal variable{'s' if len(code.co_cellvars) > 1 else ''}: {', '.join(code.co_cellvars)}"
        )

    def report_unsupported(msg: str, instruction: instructions.ThunderInstruction) -> None:
        source_lines, _ = inspect.getsourcelines(code)
        errors.append(
            f"{msg}{instruction} found\n"
            f"  {code.co_name}() defined in {code.co_filename}:{code.co_firstlineno}\n"
            f"  line {instruction.line_no + code.co_firstlineno}: {source_lines[instruction.line_no].rstrip()}"
        )

    name_arrays = FrozenDict[VariableScope, tuple[str, ...]](
        {
            VariableScope.CONST: code.co_consts,
            VariableScope.LOCAL: code.co_varnames,
            VariableScope.NONLOCAL: (*code.co_cellvars, *code.co_freevars),
            VariableScope.GLOBAL: code.co_names,
        }
    )

    def to_key(instruction: instructions.ThunderInstruction, scope: VariableScope) -> VariableKey:
        assert scope != VariableScope.STACK, "Indexing into the stack is not permitted."
        assert scope in name_arrays, f"Unknown variable scope: {scope}"
        if scope == VariableScope.NONLOCAL:
            report_unsupported("nonlocal variables are not supported but instruction = ", instruction)
        return VariableKey(name_arrays[scope][instruction.oparg], scope)

    def convert(block: tuple[instructions.ThunderInstruction, ...], stack_offset: int) -> FunctionalizedBlock:
        stack: list[PlaceholderValue] = [PlaceholderValue(f"Initial_stack_{i}") for i in range(stack_offset)]
        begin_variables = {VariableKey(idx, VariableScope.STACK): v for idx, v in enumerate(stack)}
        end_variables = InferringDict[VariableKey, PlaceholderValue | None](
            lambda key: begin_variables.setdefault(key, PlaceholderValue(f"Initial: ({key.identifier} {key.scope})"))
        )

        assert block
        functionalized: list[tuple[instructions.ThunderInstruction, Inputs, Outputs]] = []
        for idx, instruction in enumerate(block):
            # These are already reflected in the next opcode's argument
            if instruction.opname == instructions.EXTENDED_ARG:
                continue

            elif instruction in instructions.UNSAFE_OPCODES:
                # These are unsafe to run, but we should still be able to parse them.
                report_unsupported("Unsupported instruction = ", instruction)

            pop, push_by_branch = stack_effect.stack_effect_detail(instruction)
            push = max(push_by_branch)

            def assert_expected_stack_effects(pop_i: int, push_i: int) -> None:
                assert (pop, push) == (pop_i, push_i), f"{instruction=} {pop=} {push=}"

            # Peek at the stack to track variable mutations.
            if (store_scope := STORE_OPNAMES.get(instruction.opname)) is not None:
                assert_expected_stack_effects(1, 0)
                end_variables[to_key(instruction, store_scope)] = stack.pop()

            elif (del_scope := DEL_OPNAMES.get(instruction.opname)) is not None:
                assert_expected_stack_effects(1, 0)
                end_variables[to_key(instruction, del_scope)] = None

            elif (load_scope := LOAD_OPNAMES.get(instruction.opname)) is not None:
                assert_expected_stack_effects(0, 1)
                loaded = end_variables[load_key := to_key(instruction, load_scope)]
                assert loaded is not None, f"Access to deleted variable: {load_key}, {instruction}"
                stack.append(loaded)

            else:
                # We have already functionalized variable accesses, so we can prune loads and stores.
                inputs = tuple(stack.pop() for _ in range(pop))
                outputs = Outputs(tuple(PlaceholderValue(f"{idx}_{instruction.opname}__{idy}") for idy in range(push)))
                stack.extend(outputs)
                functionalized.append((instruction, Inputs(tuple(reversed(inputs))), outputs))

        end_stack = {VariableKey(idx, VariableScope.STACK): v for idx, v in enumerate(stack)}
        end_state: EndState = FrozenDict({**end_variables, **end_stack})
        return FunctionalizedBlock((tuple(functionalized), FrozenDict(begin_variables), end_state))

    stack_offsets = _compute_stack_offsets(disassembled)
    functionalized = tuple(convert(block, offset) for block, offset in safe_zip(disassembled.blocks, stack_offsets))
    if errors:
        raise RuntimeError("Preprocessing issues detected:\n" + textwrap.indent("\n\n".join(errors), " " * 4))

    return functionalized


# =============================================================================
# == Summary for debugging and testing ========================================
# =============================================================================
def _summarize(parsed: ParsedFunctional) -> str:
    # Clear identifiers for input stack values.
    to_symbol = FrozenDict[int, str](enumerate("⓵ ⓶ ⓷ ⓸ ⓹ ⓺ ⓻ ⓼ ⓽ ⓾ Ⓐ Ⓑ Ⓒ Ⓓ Ⓔ Ⓕ".split()))

    # Group output edges.
    grouped_edges: dict[int, str] = {
        source: ", ".join(f"{sink}{'(Jump)' if jump else ''}" for _, sink, jump in sinks)
        for source, sinks in itertools.groupby(parsed.provenance.edges, lambda e: e[0])
    }

    # Best effort to apply descriptive names.
    inputs_outputs = {}
    block_headers: list[str] = []
    for idx, (functionalized_block, begin, end) in enumerate(parsed.blocks):
        begin_stack = tuple(v for k, v in begin.items() if k.scope == VariableScope.STACK)
        stack_names: dict[str, str] = {v: to_symbol.get(idx, f"S{idx}") + "\u2009" for idx, v in enumerate(begin_stack)}
        names: dict[str, str] = {**{v: f"{k.identifier}" for k, v in begin.items()}, **stack_names}
        names.update({v: f"v{idx}" for idx, v in enumerate(itertools.chain(*[o for _, _, o in functionalized_block]))})
        for instruction, inputs, outputs in functionalized_block:
            inputs_outputs[instruction] = (tuple(names[i] for i in inputs), tuple(names[o] for o in outputs))

        end_stack = {k: v for k, v in end.items() if k.scope == VariableScope.STACK}
        assert tuple(k.identifier for k in end_stack) == tuple(range(len(end_stack))), end_stack
        end_stack_str = ", ".join(names[i or ""] for i in end_stack.values())
        block_headers.append(f"Block {idx}:  [{', '.join(stack_names.values())}] => [{end_stack_str}]")

    # Group loads and stores.
    prefix = {opname: opname.split("_")[0] for opname in itertools.chain(STORE_OPNAMES, LOAD_OPNAMES, DEL_OPNAMES)}
    condensed: list[list[tuple[str, instructions.ThunderInstruction | None]]] = []
    for raw_block in parsed.provenance.blocks:
        condensed.append([])
        for prefix_or_i, group in itertools.groupby(raw_block, lambda i: prefix.get(i.opname, i)):
            if isinstance(prefix_or_i, str):
                name = ", ".join(f"{i.argval}: {i.opname[len(prefix_or_i) + 1:]}" for i in group)
                condensed[-1].append((f"{prefix_or_i}[{name.replace(': FAST', '')}]", None))
            else:
                opname = f"{prefix_or_i.opname}{'' if type(prefix_or_i) is instructions.ThunderInstruction else '*'}"
                condensed[-1].append((opname, prefix_or_i))

    # Write lines.
    lines: list[str] = []
    width = max(len(name) for name, _ in itertools.chain(*condensed))
    width = max(width, max(len(i) for i in block_headers)) + 5
    for idx, (condensed_block, (_, _, end)) in enumerate(safe_zip(condensed, parsed.blocks)):
        lines.append(block_headers[idx])
        for name, maybe_i in condensed_block:
            inputs, outputs = inputs_outputs.get(maybe_i, ((), ()))  # type: ignore[assignment, arg-type]
            if inputs or outputs:
                name = f"{name} ".ljust(width, ".").replace("..", ". ")
                name = f"{name} ({', '.join(inputs)}) -> {', '.join(outputs)}"
            lines.append(f"  {name}")
        if idx in grouped_edges:
            lines.append(f"      -> {grouped_edges[idx]}")
        lines.append("")

    return "\n".join(lines)
