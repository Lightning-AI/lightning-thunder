from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
import sys
import textwrap
from types import CodeType
from typing import cast, final, Any, Callable, Deque, Generic, Literal, TypeVar
from collections.abc import Iterable, Iterator, Mapping

from thunder.core.script import parse
from thunder.core.script.algorithms import TypedDiGraph, flatten_map, sort_adjacent
from thunder.core.script.instrumentation import InstrumentingBase
from thunder.core.utils import safe_zip, FrozenDict, OrderedSet

T = TypeVar("T")


# =============================================================================
# == Variables (base classes) =================================================
# =============================================================================
ReplaceMap = Mapping["AbstractValue", "AbstractValue"]
AbstractValues = tuple["AbstractValue", ...]


class AbstractValue:
    """Represents a value during instruction parsing. (Prior to type binding.)"""

    def __copy__(self) -> AbstractValue:
        raise NotImplementedError

    @final
    def substitute(self, replace_map: ReplaceMap) -> AbstractValue:
        """Find the replacement for `self`, and recursively substitute. (If applicable.)

        Some abstract values reference other abstract values. When we make substitution during
        graph transformations it is necessary to also consider replacement of an abstract
        value's constituents. Any subclass which must be unpacked in this manner should
        override `_unpack_apply`.
        """
        new_self = replace_map.get(self, self)
        if new_self != (x := replace_map.get(new_self, new_self)):
            msg = f"""
                `replace_map` may not contain chains.
                    {self}
                    {new_self}
                    {x}
                See `flatten_map`."""
            raise ValueError(textwrap.dedent(msg))

        return new_self._unpack_apply(replace_map)

    def _unpack_apply(self, _: ReplaceMap) -> AbstractValue:
        """Recursively update any constituent references in the abstract value."""
        return self


@dataclasses.dataclass(frozen=True, eq=True)
class _TrivialComposite(AbstractValue, Generic[T]):
    """Models an AbstractValue that references other (possibly also AbstractValue) state.

    Note: `constituents` should not contain cycles.
    """

    constituents: tuple[T, ...]

    def __getitem__(self, index: int) -> T:
        return self.constituents[index]

    def _unpack_apply(self, replace_map: ReplaceMap) -> AbstractValue:
        return self.__class__(**self._unpack_kwargs(replace_map))

    def _unpack_kwargs(self, replace_map: ReplaceMap) -> dict[str, Any]:
        """Allow subclasses to participate in `_unpack_apply`."""
        constituents = (i.substitute(replace_map) if isinstance(i, AbstractValue) else i for i in self.constituents)
        return dict(constituents=tuple(constituents))


# =============================================================================
# == Intra-ProtoBlock abstract value flow =====================================
# =============================================================================
#
# `ProtoBlocks` employ a dual representation, where node inputs and outputs can
# be viewed as either a reference based DAG or a sequence of ops with concrete
# `AbstractValue` inputs and outputs.
#
# At the boundaries of a ProtoBlock values have named (VariableKey) slots;
# within the ProtoBlock there is no need for such slots (since there is no
# control flow within a block and those named slots tell you how to build the
# directed *cyclic* graph for the larger program) so they are stripped during
# parsing.
#
# The inputs of a protoblock are stored as a map of `VariableKey -> AbstractValue`
# and act as the intra-block DAG sources. The outputs are stored as references
# since every ProtoBlock output must have a unique producer. (Either an input
# or a node within the block.)
#
# The canonical representation for intra-block flow is "symbolic" (reference
# based). If an `AbstractValue` appear in a symbolic node's outputs that
# indicates that the node is that value's producer. Otherwise all inputs and
# outputs are references: inputs reference either the begin state or the
# outputs of a prior node while output references index into the node's inputs.
#
# When analyzing a graph we are generally interested in the concrete properties
# of values; provenance is generally only important when connecting blocks and
# performing rewrites. For these cases `IntraBlockFlow` generates a
# "materialized" flow which resolves all references to `AbstractValue`s. The
# symbolic representation is sufficient to emit the materialized representation,
# but the reverse is not true.
EdgeIndex = Literal[0, -1]
ValueIndex = int | tuple[int, ...]


@dataclasses.dataclass(frozen=True, eq=True)
class _OutputRef:
    """Identifies the producer of a value within a ProtoBlock."""

    instruction: parse.ThunderInstruction  #  Acts as a key for the producer Flow.
    idx: ValueIndex  #                        Indexes the node's outputs.


@dataclasses.dataclass(frozen=True, eq=False)
class _Symbolic:
    """Represents abstract value flow within a ProtoBlock."""

    # VariableKey:      References the value of that variable at the start of the ProtoBlock
    # OutputRef:        Reference values created by an earlier instruction within the block
    Input = parse.VariableKey | _OutputRef
    inputs: tuple[Input, ...]

    # ValueIndex:       Aliases the input at this position.
    # AbstractValue:   New value created by this instruction
    Output = ValueIndex | AbstractValue
    outputs: tuple[Output, ...]

    BeginState = FrozenDict[parse.VariableKey, AbstractValue]
    EndState = FrozenDict[parse.VariableKey, Input]
    Block = tuple[FrozenDict[parse.ThunderInstruction, "_Symbolic"], BeginState, EndState]

    def __post_init__(self) -> None:
        assert not any(isinstance(o, ExternalRef) for o in self.outputs), self


@dataclasses.dataclass(frozen=True, eq=False)
class _Materialized:
    """Flow element where all symbolic references have been resolved to concrete `AbstractValue`s."""

    inputs: AbstractValues
    outputs: AbstractValues


@dataclasses.dataclass(frozen=True, eq=False)
class IntraBlockFlow:
    _flow: FrozenDict[parse.ThunderInstruction, _Symbolic]

    BeginVariables = FrozenDict[parse.VariableKey, AbstractValue]
    _begin: BeginVariables

    EndVariable = _Symbolic.Input | None
    EndVariables = FrozenDict[parse.VariableKey, EndVariable]  # `None` indicates an explicit `del`
    _end: EndVariables

    def __post_init__(self) -> None:
        object.__setattr__(self, "_flow", FrozenDict({**self._flow}))
        object.__setattr__(self, "_begin", FrozenDict({**self._begin}))
        object.__setattr__(self, "_end", FrozenDict({**self._end}))

    @property
    def symbolic(self) -> Iterable[tuple[parse.ThunderInstruction, _Symbolic]]:
        yield from self._flow.items()

    @functools.cached_property
    def materialized(self) -> FrozenDict[parse.ThunderInstruction, _Materialized]:
        """Walk the flow resolving references as they we encounter them."""
        result: dict[parse.ThunderInstruction, _Materialized] = {}
        unused: IntraBlockFlow.EndVariables = FrozenDict({})  # `end` should never be needed.
        for instruction, node in self.symbolic:
            inputs = tuple(self._getitem_impl(0, i, (self._begin, unused), result) for i in node.inputs)
            outputs = [resolve_composites(inputs, o) for o in node.outputs]
            assert all(isinstance(o, AbstractValue) for o in outputs), outputs
            result[instruction] = _Materialized(tuple(inputs), tuple(outputs))

        return FrozenDict(result)

    def __getitem__(self, key: tuple[EdgeIndex, _Symbolic.Input | None]) -> AbstractValue:
        return self._getitem_impl(*key, (self._begin, self._end), self.materialized)  # __getitem__ packs args

    def substitute(self, replace_map: ReplaceMap) -> IntraBlockFlow:
        """Replace `AbstractValue`s within the flow. (Block inputs and producer nodes.)"""

        def replace(x: T) -> T:
            return x.substitute(replace_map) if isinstance(x, AbstractValue) else x  # type: ignore[return-value]

        return self.__class__(
            _flow={k: _Symbolic(v.inputs, tuple(replace(o) for o in v.outputs)) for k, v in self.symbolic},
            _begin={k: replace(v) for k, v in self._begin.items()},
            _end=self._end,
        )

    @classmethod
    def _getitem_impl(
        cls,
        index: EdgeIndex,
        key: _Symbolic.Input | None,
        variables: tuple[IntraBlockFlow.BeginVariables, IntraBlockFlow.EndVariables],
        materialized: Mapping[parse.ThunderInstruction, _Materialized],
    ) -> AbstractValue:
        # We need to index while materializing (before `self.materialized` is
        # available) so we factor the logic into a standlone method.
        if key is None:
            return NonPyObject(NonPyObject.Tag.MISSING)

        elif isinstance(key, _OutputRef):
            return resolve_composites(materialized[key.instruction].outputs, key.idx)

        assert isinstance(key, parse.VariableKey)
        if key.is_const:
            return ExternalRef(key)

        assert index in (0, -1)
        if index == 0:
            return variables[0][key]
        return cls._getitem_impl(0, variables[1][key], variables, materialized)

    @staticmethod
    def _boundary_state(
        variables: Mapping[parse.VariableKey, T],
        resolve: Callable[[T], AbstractValue],
    ) -> Iterator[tuple[parse.VariableKey, AbstractValue]]:
        for k, v in sorted(variables.items(), key=lambda kv: kv[0]):
            assert not k.is_const, k
            if not isinstance(v_resolved := resolve(v), NonPyObject):
                yield k, v_resolved


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
JumpTarget = tuple["ProtoBlock", bool]


@dataclasses.dataclass(frozen=True, eq=False)
class ProtoBlock(InstrumentingBase):
    """Stores abstract data flow for a code block."""

    flow: IntraBlockFlow
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
    def node_flow(self) -> Iterable[tuple[parse.ThunderInstruction, _Materialized]]:
        yield from self.flow.materialized.items()

    @property
    def begin_state(self) -> Iterator[tuple[parse.VariableKey, AbstractValue]]:
        yield from self.flow._boundary_state(self.flow._begin, lambda v: v)

    @property
    def end_state(self) -> Iterator[tuple[parse.VariableKey, AbstractValue]]:
        yield from (flow := self.flow)._boundary_state(flow._end, lambda v_ref: flow[0, v_ref])

    @property
    def _flow_uses(self) -> Iterable[parse.VariableKey]:
        flat_inputs = OrderedSet(itertools.chain(*(n.inputs for _, n in self.flow.symbolic)))
        yield from (i for i in flat_inputs if isinstance(i, parse.VariableKey))

    def transform(self, replace_map: ReplaceMap) -> ProtoBlock:
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
            ProtoBlock(IntraBlockFlow(symbolic, begin, end))
            for symbolic, begin, end in make_symbolic(raw_blocks)
        )
        for source, sink, jump in edges:
            protoblocks[source].add_jump_target(protoblocks[sink], jump)

        return cls(protoblocks)

    @property
    def edges(self) -> Iterator[tuple[ProtoBlock, ProtoBlock]]:
        for protoblock in self.protoblocks:
            yield from ((protoblock, target) for target, _ in protoblock.jump_targets)

    @property
    def flat_flow(self) -> Iterable[tuple[parse.ThunderInstruction, tuple[_Symbolic, _Materialized]]]:
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

    def transform(self, replace_map: ReplaceMap) -> ProtoGraph:
        """Copies the ProtoGraph with value replacements.

        NOTE: This is strictly a condensing transform, and this is only invertable
              (using another `.transform` call) in trivial cases.
        """
        assert not any(isinstance(k, NonPyObject) for k in replace_map), replace_map.keys()
        replace_map: ReplaceMap = dict(flatten_map(replace_map))
        return self.substitute({protoblock: protoblock.transform(replace_map) for protoblock in self})

    def unlink(self) -> ProtoGraph:
        """Copies the ProtoGraph but replaces all block inputs with references. (Useful for graph rewrites.)"""
        transformed: dict[ProtoBlock, ProtoBlock] = {}
        for protoblock in self:
            begin_vars = (v for _, v in protoblock.begin_state if not isinstance(v, AbstractRef))
            replace_map = {v: AbstractRef(f"Unlink: {v}") for v in begin_vars}
            transformed[protoblock] = new_protoblock = protoblock.transform(replace_map)
            new_protoblock.__post_init__()
        return self.substitute(transformed)

    def replace_symbolic(
        self,
        replace_fn: Callable[[parse.ThunderInstruction, _Symbolic], _Symbolic | None],
        retain_uses: bool = False,
    ) -> ProtoGraph:
        replacements: dict[parse.ThunderInstruction, tuple[_Symbolic, _Symbolic]] = {}
        for instruction, (old_symbolic, _) in self.flat_flow:
            new_symbolic = replace_fn(instruction, old_symbolic)
            if new_symbolic is None:
                continue

            assert len(old_symbolic.inputs) == len(new_symbolic.inputs), (old_symbolic, new_symbolic)
            assert len(old_symbolic.outputs) == len(new_symbolic.outputs), (old_symbolic, new_symbolic)
            replacements[instruction] = (old_symbolic, new_symbolic)

        transformed: dict[ProtoBlock, ProtoBlock] = {}
        for protoblock in (unlinked := self.unlink()):
            new_symbolic_flow = {k: replacements.get(k, (None, v))[1] for k, v in protoblock.flow.symbolic}
            new_flow = dataclasses.replace(protoblock.flow, _flow=new_symbolic_flow)
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
        idxes: dict[AbstractValue, int] = {}
        for pb in self:
            for _, node in pb.node_flow:
                for val in itertools.chain(node.inputs, node.outputs):
                    if val not in idxes.keys():
                        idxes[val] = counter
                        counter += 1

        def to_index_str(values: AbstractValues) -> str:
            indices = (str(idxes[v]) for v in values)
            return f"({', '.join(indices)})"

        for i, pb in enumerate(self):
            print(f"Protoblock {i}:")
            print(f"{'':>22}Inputs, Outputs")
            for instruction, node in pb.node_flow:
                print(f" {instruction.opname:>20}, {to_index_str(node.inputs)} -> {to_index_str(node.outputs)}")
            print("\n")


# =============================================================================
# == Normal values ============================================================
# =============================================================================
@dataclasses.dataclass(frozen=True, eq=False)
class AbstractRef(AbstractValue):
    """Placeholder value which will be resolved during parsing."""

    _debug_info: str = "N/A"


@dataclasses.dataclass(frozen=True, eq=True)
class NonPyObject(AbstractValue):
    """Singleton values used to signal some special interpreter state."""

    class Tag(enum.Enum):
        DELETED = enum.auto()
        MISSING = enum.auto()
        NULL = enum.auto()

    tag: Tag

    def __repr__(self) -> str:
        return self.tag.name


class IntermediateValue(AbstractValue):
    """A (potentially) new value produced by an instruction."""

    pass


@dataclasses.dataclass(frozen=True, eq=True)
class ExternalRef(AbstractValue):
    """Reference values outside of the parsed code. (Arguments, constants, globals, etc.)"""

    key: parse.VariableKey


# =============================================================================
# == Composite values =========================================================
# =============================================================================
@dataclasses.dataclass(frozen=True, eq=True)
class _Composite(_TrivialComposite[T]):
    v: AbstractValue

    def _unpack_kwargs(self, replace_map: ReplaceMap) -> dict[str, Any]:
        return {**super()._unpack_kwargs(replace_map), **dict(v=self.v.substitute(replace_map))}


class CompositeRef(_Composite[_Symbolic.Output]):
    pass


class CompositeValue(_Composite[AbstractValue]):
    pass


@dataclasses.dataclass(frozen=True, eq=True)
class AbstractPhiValue(_TrivialComposite[AbstractValue], AbstractValue):
    def __post_init__(self) -> None:
        # Flatten nested PhiValues. e.g.
        #   𝜙[𝜙[A, B], 𝜙[A, C]] -> 𝜙[A, B, C]
        constituents = itertools.chain(*[self.flatten(i) for i in self.constituents])

        # Ensure a consistent order.
        constituents = tuple(v for _, v in sorted({hash(v): v for v in constituents}.items()))
        object.__setattr__(self, "constituents", constituents)

    def __getitem__(self, _: int) -> AbstractValue:
        # The semantics of indexing into an `AbstractPhiValue`` are not well defined:
        #  - The order of `constituents` is arbitrary
        #  - It's unclear if the desire is to select one consitiuent or create a new `AbstractPhiValue`
        #    which indexes into each constituent.
        # If a concrete use case emerges we can tackle it; until then we refuse for safety.
        raise NotImplementedError

    def _unpack_apply(self, replace_map: ReplaceMap) -> AbstractValue:
        result = super()._unpack_apply(replace_map)
        assert isinstance(result, AbstractPhiValue)
        return result if len(result.constituents) > 1 else result.constituents[0]

    @classmethod
    def flatten(cls, v: AbstractValue) -> Iterable[AbstractValue]:
        constituents = [cls.flatten(i) for i in v.constituents] if isinstance(v, AbstractPhiValue) else [[v]]
        yield from itertools.chain(*constituents)


def resolve_composites(indexed: AbstractValues, output: _Symbolic.Output) -> AbstractValue:
    if isinstance(output, int):
        output = (output,)

    if isinstance(output, tuple):
        assert output and all(isinstance(idx_i, int) for idx_i in output), output
        result = indexed[output[0]]
        for idx_i in output[1:]:
            assert isinstance(result, _TrivialComposite), result  # We can only unpack a (possibly nested) composite.
            result = result[idx_i]
        return result

    elif isinstance(output, CompositeRef):
        return CompositeValue(tuple(resolve_composites(indexed, i) for i in output.constituents), output.v)

    return output


def is_detail(v: AbstractValue) -> bool:
    if isinstance(v, (AbstractRef, CompositeRef)) or type(v) in (AbstractValue, _TrivialComposite, _Composite):
        return True

    elif isinstance(v, CompositeValue):
        return is_detail(v.v) or any(is_detail(i) for i in v.constituents)

    return isinstance(v, AbstractPhiValue) and any(is_detail(i) for i in v.constituents)


# =============================================================================
# == Conversion from parsed representation ====================================
# =============================================================================
def rotate_N(oparg: int) -> tuple[int, ...]:
    return (-1,) + tuple(range(-oparg, -1))


_AliasMask = tuple[int | None, ...]
ALIAS_OPCODES = FrozenDict[str, Callable[[int], _AliasMask]](
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
        MAP_ADD=...,
        DICT_MERGE=...,
        DICT_UPDATE=...,
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


def make_symbolic(blocks: tuple[parse.FunctionalizedBlock, ...]) -> Iterator[_Symbolic.Block]:
    for block, begin_state, end_state in blocks:
        # `functionalize_blocks` produces unique values, so provenance is unambiguous.
        producers: dict[parse.PlaceholderValue | None, _Symbolic.Input] = {v: k for k, v in begin_state.items()}
        producers[None] = NonPyObject.Tag.DELETED
        assert len(producers) == len(begin_state) + 1, (producers, end_state)

        symbolic_blocks: dict[parse.ThunderInstruction, _Symbolic] = {}
        for instruction, raw_inputs, raw_outputs in block:
            for idx, o in enumerate(raw_outputs):
                assert o not in producers
                producers[o] = _OutputRef(instruction, idx)

            outputs: tuple[_Symbolic.Output, ...] = tuple(IntermediateValue() for _ in raw_outputs)
            if alias := ALIAS_OPCODES.get(instruction.opname):
                mask = alias(len(outputs)) if callable(alias) else alias
                mask = (i if i is not None else i for i in mask)
                outputs = tuple(o if o_mask is None else o_mask for o, o_mask in safe_zip(outputs, mask))
            symbolic_blocks[instruction] = _Symbolic(tuple(producers[i] for i in raw_inputs), outputs)

        begin = {k: AbstractRef(v) for k, v in begin_state.items() if not k.is_const}
        end = {k: producers[v] for k, v in end_state.items() if not k.is_const}
        yield (FrozenDict(symbolic_blocks), FrozenDict(begin), FrozenDict(end))
