from __future__ import annotations


import abc
import collections
import dataclasses
import functools
import inspect
import itertools
from types import CodeType
from typing import cast, overload, Any, Literal, NewType
from collections.abc import Iterable, Iterator, Mapping

from thunder.core.script import algorithms, instrumentation, parse, values
from thunder.core.utils import debug_asserts_enabled, FrozenDict, OrderedSet

__all__ = ("ProtoBlock", "ProtoGraph")

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
JumpTarget = NewType("JumpTarget", tuple["ProtoBlock", parse.Jump])
Uses = NewType("Uses", OrderedSet[parse.VariableKey])


@dataclasses.dataclass(frozen=True, eq=False)
class ProtoBlock(instrumentation.InstrumentingBase):  # type: ignore[misc,no-any-unimported]
    """Stores abstract data flow for a code block."""

    flow: values.IntraBlockFlow
    jump_targets: tuple[JumpTarget, ...] = dataclasses.field(default=(), init=False)
    uses: Uses = dataclasses.field(default_factory=lambda: Uses(OrderedSet()), init=False)

    def __repr__(self) -> str:
        ops = "\n".join(f"  {i.opname}" for i, _ in self.flow.symbolic)
        return f"ProtoBlock: {hex(id(self))}\n{ops}"

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        self.uses.update(self.flow.uses)

    def add_jump_target(self, other: ProtoBlock, jump: parse.Jump) -> None:
        """We need to add jump targets after all ProtoBlocks are initialized."""

        # Override `frozen=True` for this one limited use case.
        object.__setattr__(self, "jump_targets", self.jump_targets + ((other, jump),))


@dataclasses.dataclass(frozen=True, eq=False)
class ProtoGraph:
    protoblocks: tuple[ProtoBlock, ...]
    root: ProtoBlock
    parents: Mapping[ProtoBlock, tuple[ProtoBlock, ...]]

    Provenance = values.ParsedSymbolic | tuple[type["ProtoGraphTransform"], "ProtoGraph"]
    provenance: Provenance

    def __init__(self, protoblocks: Iterable[ProtoBlock], provenance: Provenance) -> None:
        G = algorithms.TypedDiGraph[ProtoBlock]()
        for protoblock in (protoblocks := tuple(protoblocks)):
            is_return = tuple(protoblock.flow.symbolic)[-1][0].opname == parse.RETURN_VALUE
            G.add_node(protoblock, is_return=is_return)

        for protoblock in protoblocks:
            for destination, jump in protoblock.jump_targets:
                G.add_edge(protoblock, destination, adjacent=not jump)

        assert protoblocks
        object.__setattr__(self, "protoblocks", tuple(algorithms.sort_adjacent(G)))
        assert len(G) == len(self.protoblocks) == len(protoblocks), (len(G), len(self.protoblocks), len(protoblocks))

        object.__setattr__(self, "root", self.protoblocks[0])
        root_stack = [(k, v) for k, v in self.root.flow.begin_state if k.scope == parse.VariableScope.STACK]
        assert not root_stack, f"Root block should not have stack inputs: {root_stack}"

        nodes = cast(Iterable[ProtoBlock], G.nodes)  # For some reason mypy needs this.
        parents = {protoblock: tuple(G.predecessors(protoblock)) for protoblock in nodes}
        object.__setattr__(self, "parents", FrozenDict(parents))
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_code(cls, co: CodeType) -> ProtoGraph:
        """Given a method, disassemble it to a sequence of simple blocks."""
        parsed = values.ParsedSymbolic.make(parse.ParsedFunctional.make(co))
        protoblocks = tuple(
            ProtoBlock(values.IntraBlockFlow(symbolic, begin, end)) for symbolic, begin, end in parsed.blocks
        )
        for source, sink, jump in parsed.provenance.provenance.edges:
            protoblocks[source].add_jump_target(protoblocks[sink], jump)

        return cls(protoblocks, parsed)

    def __iter__(self) -> Iterator[ProtoBlock]:
        yield from self.protoblocks

    def __getitem__(self, index: int) -> ProtoBlock:
        return self.protoblocks[index]

    def __len__(self) -> int:
        return len(self.protoblocks)

    def __repr__(self) -> str:
        return "\n\n".join(repr(protoblock) for protoblock in self)

    @property
    def flat_flow(self) -> Iterable[tuple[parse.ThunderInstruction, values.Symbolic, values.Materialized]]:
        for protoblock in self:
            for instruction, symbolic in protoblock.flow.symbolic:
                yield instruction, symbolic, protoblock.flow.materialized[instruction]

    @property
    def is_linked(self) -> bool:
        # NOTE: `is_linked` is vacuously True for a single block graph.
        flat_begin = itertools.chain(*(i.flow._begin.values() for i in self if i is not self.root))
        return len(self) == 1 or any(not isinstance(i, values.AbstractRef) for i in flat_begin)

    def unlink(self) -> ProtoGraph:
        return Unlink(self).apply(or_default=True)

    def link(self) -> ProtoGraph:
        if result := ProtoGraphTransform.chain(self, AddTransitive, MatchStacks, Connect):
            assert AddTransitive(result).apply(or_default=False) is None
        assert (result or self).is_linked
        return result or self

    def debug_print_protoflows(self) -> None:
        """
        Print out the node_flow for each protoblock in the
        protograph, in a way that's nice to read and debug with.
        """

        counter = 0
        idxes: dict[values.AbstractValue, int] = {}
        for pb in self:
            for node in pb.flow.materialized.values():
                for val in itertools.chain(node.inputs.ordered, node.outputs):
                    if val not in idxes.keys():
                        idxes[val] = counter
                        counter += 1

        def to_index_str(values: tuple[values.AbstractValue, ...]) -> str:
            indices = (str(idxes[v]) for v in values)
            return f"({', '.join(indices)})"

        for i, pb in enumerate(self):
            print(f"Protoblock {i}:")
            print(f"{'':>22}Inputs, Outputs")
            for instruction, node in pb.flow.materialized.items():
                print(f" {instruction.opname:>20}, {to_index_str(node.inputs.ordered)} -> {to_index_str(node.outputs)}")
            print("\n")


# =============================================================================
# == Graph transforms (Base classes) ==========================================
# =============================================================================
#   ProtoGraphTransform
#       ReplaceProtoBlocks
#           ReplaceValues
#           CondenseValues
#           ReplaceSymbolic


class ProtoGraphTransform(abc.ABC):
    """Handles mechanical portions of graph rewrites.
    The base case is unopinionated; it simply accepts whatever new ProtoGraph is
    emitted by `self._apply`. The primary feature it provides is checking.

    NOTE:
        The convention adopted is for the pass logic to produce `T | None`
        (e.g. `ProtoGraph | None`) where `None` signals that no change is
        applicable.

    Forbid Linked:
        A key invariant of ProtoGraph is that every AbstractValue has exactly
        **one** producer, which is set by the symbolic flow. (With the exception
        of `AbstractRef`s which are placeholders for an as-yet unspecified
        AbstractValue.) However, within a block there is a flat list of
        **concrete** values specifying the state at the start of the block.

        If one were to replace all instances of `X` in a ProtoGraph with `Y`,
        this invariant would be preserved. On the other hand, if one were to
        replace `X` with `Y` **only at the symbolic producer of `X`** then
        downstream blocks could still have `X` as a block input, despite the
        fact that `X` no longer has a producer. (Note that this is only a problem
        across blocks; within blocks the materialization pass respects the update
        and emits a consistent materialized state for the new ProtoBlock.)

        It is often convenient to simply rewrite the symbolic flow within a
        single ProtoBlock. In that case the correct procedure is to generate an
        unlinked ProtoGraph, perform the local rewrites, and then relink it.
        (Where the connection pass will handle reconciliation automatically.)

    Check idempotence:
        Nearly all passes are expected to be idempotent. This provides a good
        deal of free test coverage since it produces both a test case (the result
        of `self._apply`) and an expected result. (That `self._apply` returns
        `None`.) We perform this check many times in order to flush out
        non-deterministic passes. (Though the value is configurable if a pass is
        particularly expensive.)

        However, given the potential added start up latency and possibility of
        spurious failures this check is gated by `debug_asserts_enabled`, which
        defaults to `False`. (Except for unit tests.)
    """

    _forbid_linked: bool = False
    _kwargs: FrozenDict[str, Any]  # Used to replay transform for `idempotent` check.
    _idempotent_repeats: int = 10

    @abc.abstractmethod
    def _apply(self) -> ProtoGraph | None:
        """Override this method to emit an (optional) new ProtoGraph."""
        ...

    def __new__(cls, *args: Any, **kwargs: Any) -> ProtoGraphTransform:
        self = super().__new__(cls)
        bound = inspect.signature(self.__class__.__init__).bind(None, *args, **kwargs).arguments
        bound.pop("self")
        bound.pop("proto_graph")
        self._kwargs = FrozenDict(bound)
        return self

    def __init__(self, proto_graph: ProtoGraph) -> None:
        assert not (self._forbid_linked and len(proto_graph) > 1 and proto_graph.is_linked), self
        assert isinstance(proto_graph, ProtoGraph)
        self._protograph = proto_graph

    @property
    def protograph(self) -> ProtoGraph:
        return self._protograph

    @overload
    def apply(self, or_default: Literal[False]) -> ProtoGraph | None:
        ...

    @overload
    def apply(self, or_default: Literal[True]) -> ProtoGraph:
        ...

    def apply(self, or_default: bool = False) -> ProtoGraph | None:
        result = self._apply()
        if debug_asserts_enabled():
            result_to_check = result or self.protograph
            for i in range(self._idempotent_repeats):
                assert self.__class__(proto_graph=result_to_check, **self._kwargs)._apply() is None, (i, self)
        return result or (self.protograph if or_default else None)

    @staticmethod
    def chain(proto_graph: ProtoGraph, *transforms: type[ProtoGraphTransform]) -> ProtoGraph | None:
        initial = proto_graph
        for transform in transforms:
            proto_graph = transform(proto_graph).apply(or_default=True)
        return None if proto_graph is initial else proto_graph


class ReplaceProtoBlocks(ProtoGraphTransform):
    """Helper to replace individual ProtoBlocks while retaining the same ProtoGraph topology."""

    @abc.abstractmethod
    def apply_to_protoblock(self, protoblock: ProtoBlock) -> values.IntraBlockFlow | None:
        ...

    def post_apply(self, old: ProtoBlock, new: ProtoBlock) -> None:
        pass

    def _apply(self) -> ProtoGraph | None:
        # TODO(robieta): Right now block order is load bearing, so we have to preserve it.
        transformed = {i: self.apply_to_protoblock(i) for i in self.protograph}

        if any(transformed.values()):
            replacements = {i: ProtoBlock(flow or i.flow) for i, flow in transformed.items()}
            for old_protoblock, new_protoblock in replacements.items():
                self.post_apply(old_protoblock, new_protoblock)
                for old_target, is_jump in old_protoblock.jump_targets:
                    new_protoblock.add_jump_target(replacements[old_target], is_jump)
            return ProtoGraph(replacements.values(), provenance=(self.__class__, self.protograph))
        return None


class ReplaceValues(ReplaceProtoBlocks):
    """Copies the ProtoGraph with value replacements.

    NOTE: This is strictly a condensing transform, and this is only invertible
          (using another `ReplaceValues`) in trivial cases.
    """

    _retain_uses: bool = True

    @abc.abstractproperty
    def replace_map(self) -> values.ReplaceMap:
        ...

    @functools.cached_property
    def _replace_map(self) -> values.ReplaceMap:
        replace_map = self.replace_map
        assert not (invalid := [k for k in replace_map if isinstance(k, values.NonPyObject)]), invalid
        return FrozenDict(algorithms.flatten_map(replace_map))

    def apply_to_protoblock(self, protoblock: ProtoBlock) -> values.IntraBlockFlow | None:
        return protoblock.flow.substitute(self._replace_map)

    def post_apply(self, old: ProtoBlock, new: ProtoBlock) -> None:
        if self._retain_uses:
            new.uses.update(old.uses)


class CondenseValues(ReplaceValues):
    ValueEdges = Iterable[tuple[values.AbstractValue, values.AbstractValue]]

    @abc.abstractproperty
    def edges(self) -> ValueEdges:
        ...

    @property
    def replace_map(self) -> values.ReplaceMap:
        replace_map: dict[values.AbstractValue, values.AbstractValue] = {}
        edges = itertools.chain(self.edges, self._phivalue_constituent_edges)
        for v, condensed in algorithms.compute_condense_map(edges).items():
            # Check invariants.
            assert condensed
            if not isinstance(v, values.AbstractPhiValue):
                invariants = ({c.identity for c in condensed} == {v.identity}, not isinstance(v, values.AbstractRef))
                assert all(invariants) or not any(invariants), (invariants, v, condensed)

            # `AbstractPhiValue._unpack_apply` will determine if we need an AbstractPhiValue.
            if (replacement := values.substitute_value(values.AbstractPhiValue(tuple(condensed)), {})) != v:
                replace_map[v] = replacement

        return FrozenDict(replace_map)

    @property
    def _phivalue_constituent_edges(self) -> ValueEdges:
        # AbstractPhiValues are somewhat unusual in that mismatches between blocks
        # are expected (that's sort of the point...) so we need to decompose them
        # so the condense pass doesn't get tripped up.
        for _, initial_ref in self.protograph.root.flow.begin_state:
            if isinstance(initial_ref, values.AbstractPhiValue):
                yield from ((constituent, initial_ref) for constituent in initial_ref.constituents)


class ReplaceSymbolic(ReplaceProtoBlocks):
    _forbid_linked = True

    @abc.abstractmethod
    def apply_to_symbolic(
        self,
        instruction: parse.ThunderInstruction,
        symbolic: values.Symbolic,
        inputs: values.HybridMap[values.AbstractValue],
    ) -> values.Symbolic | None:
        ...

    def apply_to_protoblock(self, protoblock: ProtoBlock) -> values.IntraBlockFlow | None:
        flow_state = values.DigestFlow(protoblock.flow._begin)
        updated_symbolic: dict[parse.ThunderInstruction, values.Symbolic | None] = {}
        for i, symbolic in protoblock.flow.symbolic:
            updated_symbolic[i] = self.apply_to_symbolic(i, symbolic, symbolic.inputs.map(flow_state.get))
            _ = flow_state.next(i, updated_symbolic[i] or symbolic)

        if any(updated_symbolic.values()):
            new_symbolic = {k: v or protoblock.flow._symbolic[k] for k, v in updated_symbolic.items()}
            return dataclasses.replace(protoblock.flow, _symbolic=FrozenDict(new_symbolic))
        return None


# =============================================================================
# == Graph transforms (Applied) ===============================================
# =============================================================================
class Unlink(ReplaceProtoBlocks):
    def apply_to_protoblock(self, protoblock: ProtoBlock) -> values.IntraBlockFlow | None:
        if protoblock is not self.protograph.root:
            uses = (flow := protoblock.flow).uses.copy()
            end: values.Symbolic.EndState = FrozenDict({k: v for k, v in flow._end.items() if k != v})
            uses.update(v for v in end.values() if isinstance(v, parse.VariableKey) and not v.is_const)
            any_non_ref = any(not isinstance(i, values.AbstractRef) for i in flow._begin.values())
            if any_non_ref or len(end) < len(flow._end) or flow._begin.keys() ^ uses:  # symmetric_difference
                begin: FrozenDict[parse.VariableKey, values.AbstractValue]
                begin = FrozenDict({k: values.AbstractRef(f"Unlink: {k}") for k in uses})
                return dataclasses.replace(protoblock.flow, _begin=begin, _end=end)

        return None

    def _apply(self) -> ProtoGraph | None:
        result = super()._apply()
        assert len(result or self.protograph) == 1 or not (result or self.protograph).is_linked, result
        return result


class AddTransitive(ReplaceProtoBlocks):
    """Extend abstract value flows to include those needed by downstream blocks.
    This pass effectively functionalizes the abstract value flow by plumbing
    reads through parents as transitive dependencies. Note that we assume
    variables are only modified by `STORE_...` and `DELETE_...` instructions.
    This is not a sound assumption since opaque calls (`CALL_FUNCTION`,
    `CALL_METHOD`, etc.) could mutate global and nonlocal variables. This does
    not, however, pose an overall soundness problem because we can check for
    state mutations during inlining and rerun flow analysis.
    """

    def apply_to_protoblock(self, protoblock: ProtoBlock) -> values.IntraBlockFlow | None:
        if missing := self.expanded_uses[protoblock].difference(protoblock.uses):
            flow = protoblock.flow
            begin = {**{k: values.AbstractRef("Transitive") for k in missing}, **flow._begin}
            end = {**{use: use for use in self.target_uses(protoblock, self.expanded_uses)}, **flow._end}
            return dataclasses.replace(flow, _begin=FrozenDict(begin), _end=FrozenDict(end))
        return None

    def post_apply(self, old: ProtoBlock, new: ProtoBlock) -> None:
        new.uses.update(self.expanded_uses[old])

    @functools.cached_property
    def expanded_uses(self) -> Mapping[ProtoBlock, Uses]:
        """Identify new transitive value dependencies.
        The process is more involved than simply checking for mismatches because
        adding a transitive value to a block may necessitate adding a transitive
        value to the prior block and so on.
        """
        uses = {protoblock: protoblock.uses.copy() for protoblock in self.protograph}
        blocks_to_process = collections.deque(uses.keys())

        while blocks_to_process:
            protoblock = blocks_to_process.popleft()
            target_uses = self.target_uses(protoblock, uses)

            # The reason we can ignore ALL `_OutputRef`s (including those that would index into a composite)
            # is that the (potential) composite's dependencies are already handled by `ProtoBlock._flow_uses`.
            transitive_uses = OrderedSet(
                source
                for use in target_uses
                if isinstance(source := protoblock.flow._end.get(use, use), parse.VariableKey)
                and source.scope != parse.VariableScope.CONST
            )

            if transitive_uses - uses[protoblock]:
                uses[protoblock].update(transitive_uses)
                blocks_to_process.extend(self.protograph.parents[protoblock])

        return FrozenDict(uses)

    @staticmethod
    def target_uses(protoblock: ProtoBlock, uses: Mapping[ProtoBlock, Uses] = FrozenDict()) -> Uses:
        flat_uses = itertools.chain(*(uses.get(target, target.uses) for target, _ in protoblock.jump_targets))
        return Uses(OrderedSet(use for use in flat_uses if use.scope != parse.VariableScope.CONST))


class MatchStacks(ReplaceProtoBlocks):
    """Ensure stacks match across blocks.

    ProtoGraph doesn't rely on stack behavior (push, pop TOS, etc.), however it
    is still a good sanity check. (Which is why `Connect._inter_block_edges` asserts.)
    """

    def apply_to_protoblock(self, protoblock: ProtoBlock) -> values.IntraBlockFlow | None:
        upstream = OrderedSet[parse.VariableKey]()
        for parent in self.protograph.parents[protoblock]:
            upstream.update(k for k in parent.flow._end if k.scope == parse.VariableScope.STACK)

        if delta := upstream - protoblock.flow._begin:
            begin = {**protoblock.flow._begin, **{k: values.AbstractRef(f"Match stack: {k}") for k in delta}}
            return dataclasses.replace(protoblock.flow, _begin=FrozenDict(begin))
        return None

    def post_apply(self, old: ProtoBlock, new: ProtoBlock) -> None:
        new.uses.update(old.uses)


class Connect(CondenseValues):
    @property
    def edges(self) -> CondenseValues.ValueEdges:
        yield from self._inter_block_edges(self.protograph)
        yield from self._graph_input_edges(self.protograph)

    @staticmethod
    def _graph_input_edges(proto_graph: ProtoGraph) -> CondenseValues.ValueEdges:
        for key, initial_ref in proto_graph.root.flow.begin_state:
            if isinstance(initial_ref.identity, values.ExternalRef):
                continue

            assert isinstance(initial_ref, values.AbstractRef), initial_ref
            assert key.scope not in (
                parse.VariableScope.CONST,
                parse.VariableScope.STACK,
            ), (key, proto_graph.root.flow._begin)
            yield values.CompositeValue().add_identity(values.ExternalRef(key)), initial_ref

    @staticmethod
    def _inter_block_edges(proto_graph: ProtoGraph) -> CondenseValues.ValueEdges:
        for protoblock in proto_graph:
            for child, _ in protoblock.jump_targets:
                outputs = dict(protoblock.flow.end_state)
                child_inputs = dict(child.flow.begin_state)
                for key, child_input in child_inputs.items():
                    yield outputs.get(key, values.NonPyObject(values.NonPyObject.Tag.MISSING)), child_input

                # `AddTransitive` should ensure the stacks match.
                # (Except for return blocks which may discard the stack.)
                opname = tuple(child.flow.symbolic)[-1][0].opname
                if opname not in parse.RAISE_RETURN_INSTRUCTIONS:
                    s_out = tuple(sorted(i.identifier for i in outputs if i.scope == parse.VariableScope.STACK))
                    s_in = tuple(sorted(i.identifier for i in child_inputs if i.scope == parse.VariableScope.STACK))
                    assert s_out == s_in, f"{s_out=} != {s_in=}, {opname}"
