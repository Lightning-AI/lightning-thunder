from __future__ import annotations

import dataclasses
import itertools
from types import CodeType
from typing import cast, Callable, TypeVar
from collections.abc import Iterable, Iterator

from thunder.core.script import parse, values
from thunder.core.script.algorithms import TypedDiGraph, flatten_map, sort_adjacent
from thunder.core.script.instrumentation import InstrumentingBase
from thunder.core.utils import OrderedSet

T = TypeVar("T")


# =============================================================================
# == Inter-ProtoBlock abstract value flow =====================================
# =============================================================================
#
# ProtoBlocks are weakly coupled by design. The `VariableKey` slots allow edges
# to be deduced (e.g. `x` at the start of one block must be the same as `x` at
# the end of the prior block), but there's no strong requirement. (And indeed,
# the ProtoGraph immediately after parsing has all unconnected `AbstractRef`s
# for input values.) Similarly, ProtoGraph serves only to record organize the
# block topology, check invariants, and provide various helper methods.
#
# This weak coupling exists to facilitate graph rewrites and reduce the surface
# area for self-inconsistent representation. By readily discarding (deduced)
# information we don't need to carry invariants through complex passes; we can
# simply decouple the graph, perform whatever local modifications we like, and
# then reconnect everything. This representation is immutable (notwithstanding
# a few implementation details), so "decouple" means emitting a new erased
# graph. (Though simple value replacements can be done directly.)
JumpTarget = tuple["ProtoBlock", parse.Jump]


@dataclasses.dataclass(frozen=True, eq=False)
class ProtoBlock(InstrumentingBase):
    """Stores abstract data flow for a code block."""

    flow: values.IntraBlockFlow
    jump_targets: tuple[JumpTarget, ...] = dataclasses.field(default=(), init=False)
    uses: OrderedSet[parse.VariableKey] = dataclasses.field(default_factory=OrderedSet, init=False)

    def __repr__(self) -> str:
        return f"ProtoBlock: {hex(id(self))}"

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        self.uses.clear()
        self.uses.update(self._flow_uses)

    def add_jump_target(self, other: ProtoBlock, jump: bool) -> None:
        """We need to add jump targets after all ProtoBlocks are initialized."""

        # Override `frozen=True` for this one limited use case.
        object.__setattr__(self, "jump_targets", self.jump_targets + ((other, jump),))

    @property
    def node_flow(self) -> Iterable[tuple[parse.ThunderInstruction, values.Materialized]]:
        yield from self.flow.materialized.items()

    @property
    def begin_state(self) -> Iterator[tuple[parse.VariableKey, values.AbstractValue]]:
        yield from self.flow.begin_state

    @property
    def end_state(self) -> Iterator[tuple[parse.VariableKey, values.AbstractValue]]:
        yield from self.flow.end_state

    @property
    def _flow_uses(self) -> Iterable[parse.VariableKey]:
        flat_inputs = OrderedSet(itertools.chain(*(n.inputs for _, n in self.flow.symbolic)))
        yield from (i for i in flat_inputs if isinstance(i, parse.VariableKey))

    def transform(self, replace_map: values.ReplaceMap) -> ProtoBlock:
        """Create a copy of `self` but allow values to be substituted."""
        transformed = ProtoBlock(self.flow.substitute(replace_map))
        transformed.uses.update(self.uses)

        # NOTE: The caller will need to repopulate `transformed.jump_targets`
        return transformed


class ProtoGraph:
    protoblocks: tuple[ProtoBlock, ...]
    root: ProtoBlock
    parents: dict[ProtoBlock, tuple[ProtoBlock, ...]]

    def __init__(self, protoblocks: Iterable[ProtoBlock]) -> None:
        G = TypedDiGraph[ProtoBlock]()
        for protoblock in (protoblocks := tuple(protoblocks)):
            is_return = tuple(protoblock.flow.symbolic)[-1][0].opname == parse.RETURN_VALUE
            G.add_node(protoblock, is_return=is_return)

        for protoblock in protoblocks:
            for destination, jump in protoblock.jump_targets:
                G.add_edge(protoblock, destination, adjacent=not jump)

        self.protoblocks = tuple(sort_adjacent(G))
        assert len(G) == len(self.protoblocks) == len(protoblocks), (len(G), len(self.protoblocks), len(protoblocks))

        self.root = self.protoblocks[0]
        root_stack = [(k, v) for k, v in self.root.begin_state if k.scope == parse.VariableScope.STACK]
        assert not root_stack, f"Root block should not have stack inputs: {root_stack}"
        nodes = cast(Iterable[ProtoBlock], G.nodes)  # For some reason mypy needs this.
        self.parents = {protoblock: tuple(G.predecessors(protoblock)) for protoblock in nodes}

    @classmethod
    def from_code(cls, co: CodeType) -> ProtoGraph:
        """Given a method, disassemble it to a sequence of simple blocks."""
        raw_blocks, edges, _ = parse.functionalize_blocks(co)
        protoblocks = tuple(
            ProtoBlock(values.IntraBlockFlow(symbolic, begin, end))
            for symbolic, begin, end in values.make_symbolic(raw_blocks)
        )
        for source, sink, jump in edges:
            protoblocks[source].add_jump_target(protoblocks[sink], jump)

        return cls(protoblocks)

    @property
    def edges(self) -> Iterator[tuple[ProtoBlock, ProtoBlock]]:
        for protoblock in self.protoblocks:
            yield from ((protoblock, target) for target, _ in protoblock.jump_targets)

    @property
    def flat_flow(self) -> Iterable[tuple[parse.ThunderInstruction, tuple[values.Symbolic, values.Materialized]]]:
        for protoblock in self.protoblocks:
            for instruction, node in protoblock.flow.symbolic:
                yield instruction, (node, protoblock.flow.materialized[instruction])

    def substitute(self, transformed: dict[ProtoBlock, ProtoBlock]) -> ProtoGraph:
        """Copies the ProtoGraph with block level substitutions while retaining the same topology."""
        assert not (delta := OrderedSet(transformed.keys()) - OrderedSet(self)), delta

        # TODO(robieta): Right now block order is load bearing, so we have to preserve it.
        transformed = {k: transformed.get(k) or k.transform({}) for k in self}
        for old_protoblock, new_protoblock in transformed.items():
            for old_target, is_jump in old_protoblock.jump_targets:
                new_protoblock.add_jump_target(transformed[old_target], is_jump)

        return ProtoGraph(transformed.values())

    def transform(self, replace_map: values.ReplaceMap) -> ProtoGraph:
        """Copies the ProtoGraph with value replacements.

        NOTE: This is strictly a condensing transform, and this is only invertable
              (using another `.transform` call) in trivial cases.
        """
        assert not any(isinstance(k, values.NonPyObject) for k in replace_map), replace_map.keys()
        replace_map: values.ReplaceMap = dict(flatten_map(replace_map))
        return self.substitute({protoblock: protoblock.transform(replace_map) for protoblock in self})

    def unlink(self) -> ProtoGraph:
        """Copies the ProtoGraph but replaces all block inputs with references. (Useful for graph rewrites.)"""
        transformed: dict[ProtoBlock, ProtoBlock] = {}
        for protoblock in self:
            begin_vars = (v for _, v in protoblock.begin_state if not isinstance(v, values.AbstractRef))
            replace_map = {v: values.AbstractRef(f"Unlink: {v}") for v in begin_vars}
            transformed[protoblock] = new_protoblock = protoblock.transform(replace_map)
            new_protoblock.__post_init__()
        return self.substitute(transformed)

    def replace_symbolic(
        self,
        replace_fn: Callable[[parse.ThunderInstruction, values.Symbolic], values.Symbolic | None],
        retain_uses: bool = False,
    ) -> ProtoGraph:
        replacements: dict[parse.ThunderInstruction, tuple[values.Symbolic, values.Symbolic]] = {}
        for instruction, (old_symbolic, _) in self.flat_flow:
            new_symbolic = replace_fn(instruction, old_symbolic)
            if new_symbolic is None:
                continue

            assert len(old_symbolic.inputs.ordered) == len(new_symbolic.inputs.ordered), (old_symbolic, new_symbolic)
            assert len(old_symbolic.outputs) == len(new_symbolic.outputs), (old_symbolic, new_symbolic)
            replacements[instruction] = (old_symbolic, new_symbolic)

        transformed: dict[ProtoBlock, ProtoBlock] = {}
        for protoblock in (unlinked := self.unlink()):
            new_symbolic_flow = {k: replacements.get(k, (None, v))[1] for k, v in protoblock.flow.symbolic}
            new_flow = dataclasses.replace(protoblock.flow, _symbolic=new_symbolic_flow)
            transformed[protoblock] = new_protoblock = ProtoBlock(new_flow)
            new_protoblock.uses.update(protoblock.uses if retain_uses else ())
        return unlinked.substitute(transformed)

    def __iter__(self) -> Iterator[ProtoBlock]:
        yield from self.protoblocks

    def __getitem__(self, index: int) -> ProtoBlock:
        return self.protoblocks[index]

    def debug_print_protoflows(self) -> None:
        """
        Print out the node_flow for each protoblock in the
        protograph, in a way that's nice to read and debug with.
        """

        counter = 0
        idxes: dict[values.AbstractValue, int] = {}
        for pb in self:
            for _, node in pb.node_flow:
                for val in itertools.chain(node.inputs, node.outputs):
                    if val not in idxes.keys():
                        idxes[val] = counter
                        counter += 1

        def to_index_str(values: tuple[values.AbstractValue, ...]) -> str:
            indices = (str(idxes[v]) for v in values)
            return f"({', '.join(indices)})"

        for i, pb in enumerate(self):
            print(f"Protoblock {i}:")
            print(f"{'':>22}Inputs, Outputs")
            for instruction, node in pb.node_flow:
                print(f" {instruction.opname:>20}, {to_index_str(node.inputs)} -> {to_index_str(node.outputs)}")
            print("\n")
