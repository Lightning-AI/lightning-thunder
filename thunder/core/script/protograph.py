from __future__ import annotations

import collections
import dataclasses
import functools
import inspect
import itertools
import marshal
import textwrap
from types import CodeType, MappingProxyType
from typing import cast, final, Any, Deque, Literal, NamedTuple, TypeVar
from collections.abc import Callable, Iterable, Iterator, Mapping

from thunder.core.script.algorithms import TypedDiGraph, flatten_map, sort_adjacent
from thunder.core.script.instrumentation import InstrumentingBase
from thunder.core.script.python_ir_data import (
    stack_effect_adjusted,
    PushNew,
    ThunderInstruction,
    VariableScope,
    DEL_OPNAMES,
    LOAD_OPNAMES,
    RETURN_VALUE,
    STORE_OPNAMES,
    EXTENDED_ARG,
    UNSAFE_OPCODES,
)
from thunder.core.utils import InferringDict, OrderedSet

T = TypeVar("T")


class VariableKey(NamedTuple):
    """Denotes the location of a variable.
    For example, `x = 5` assigns the variable stored in `VariableKey(5, VariableScope.CONST)`
    to the location `VariableKey("x", VariableScope.LOCAL)`. (Provided `x` is a local variable.)
    The type of `identifier` varies based on `scope`:
        `marshal`able   VariableScope.CONST
        str             VariableScope.LOCAL / NONLOCAL / GLOBAL
        int             VariableScope.STACK
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
            # We prefer to use native ordering. However for unorderable types (e.g. CodeType)
            # `marshal` at least provides a consistent ordering.
            assert self.scope == VariableScope.CONST and other.scope == VariableScope.CONST
            return marshal.dumps(self.identifier) < marshal.dumps(other.identifier)


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
                See `flatten_map`.
            """
            raise ValueError(textwrap.dedent(msg))

        return new_self._unpack_apply(replace_map)

    def _unpack_apply(self, _: ReplaceMap) -> AbstractValue:
        """Recursively update any constituent references in the abstract value."""
        return self


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


@dataclasses.dataclass(frozen=True, eq=True)
class _OutputRef:
    """Identifies the producer of a value within a ProtoBlock."""

    instruction: ThunderInstruction  #  Acts as a key for the producer Flow.
    idx: int  #                         Indexes the node's outputs.


@dataclasses.dataclass(frozen=True, eq=False)
class _Symbolic:
    """Represents abstract value flow within a ProtoBlock."""

    # VariableKey:      References the value of that variable at the start of the ProtoBlock
    # OutputRef:        Reference values created by an earlier instruction within the block
    Input = VariableKey | _OutputRef
    inputs: tuple[Input, ...]

    # int:              Aliases the input at this position
    # AbstractValue:   New value created by this instruction
    Output = int | AbstractValue
    outputs: tuple[Output, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class _Materialized:
    """Flow element where all symbolic references have been resolved to concrete `AbstractValue`s."""

    inputs: AbstractValues
    outputs: AbstractValues


@dataclasses.dataclass(frozen=True, eq=False)
class IntraBlockFlow:
    _flow: Mapping[ThunderInstruction, _Symbolic]

    BeginVariables = Mapping[VariableKey, AbstractValue]
    _begin: BeginVariables

    EndVariable = _Symbolic.Input | None
    EndVariables = Mapping[VariableKey, EndVariable]  # `None` indicates an explicit `del`
    _end: EndVariables

    def __post_init__(self) -> None:
        object.__setattr__(self, "_flow", MappingProxyType({**self._flow}))
        object.__setattr__(self, "_begin", MappingProxyType({**self._begin}))
        object.__setattr__(self, "_end", MappingProxyType({**self._end}))

    @property
    def symbolic(self) -> Iterable[tuple[ThunderInstruction, _Symbolic]]:
        yield from self._flow.items()

    @functools.cached_property
    def materialized(self) -> MappingProxyType[ThunderInstruction, _Materialized]:
        """Walk the flow resolving references as they we encounter them."""

        def resolve(inputs: AbstractValues, o: _Symbolic.Output) -> AbstractValue:
            assert isinstance(o, (int, AbstractValue))
            return inputs[o] if isinstance(o, int) else o

        result: dict[ThunderInstruction, _Materialized] = {}
        for instruction, node in self.symbolic:
            inputs = tuple(self._getitem_impl(0, i, (self._begin, self._end), result) for i in node.inputs)
            outputs = [resolve(inputs, o) for o in node.outputs]
            assert all(isinstance(o, AbstractValue) for o in outputs), outputs
            result[instruction] = _Materialized(tuple(inputs), tuple(outputs))

        return MappingProxyType(result)

    def __getitem__(self, key: tuple[EdgeIndex, _Symbolic.Input | None]) -> AbstractValue:
        return self._getitem_impl(*key, (self._begin, self._end), self.materialized)  # __getitem__ packs args

    @classmethod
    def from_instructions(cls, instructions: tuple[ThunderInstruction, ...], code: CodeType) -> IntraBlockFlow:
        flow: dict[ThunderInstruction, _Symbolic] = {}
        begin_variables: dict[VariableKey, AbstractValue] = {}

        def end_missing(key: VariableKey) -> IntraBlockFlow.EndVariable:
            if key.scope != VariableScope.CONST:
                begin_variables.setdefault(key, AbstractRef(f"Variable initial value: ({key.identifier} {key.scope})"))
            return key

        end_variables = InferringDict[VariableKey, IntraBlockFlow.EndVariable](end_missing)
        block_inputs: Deque[AbstractValue] = collections.deque()
        stack: Deque[_Symbolic.Input] = collections.deque()

        def peek_stack(pop: bool = False) -> _Symbolic.Input:
            # If the stack is empty we can infer that we are trying to reference
            # a value was already on the stack at the start of the block.
            if not stack:
                index = -len(block_inputs) - 1
                block_inputs.appendleft(AbstractRef(f"Inferred stack input: {index}"))
                stack.append(VariableKey(index, VariableScope.STACK))
            return stack.pop() if pop else stack[-1]

        def make_unsupported(msg: str, instruction: ThunderInstruction):
            source_lines, _ = inspect.getsourcelines(code)
            msg = f"""{msg}{instruction} found
            {code.co_name} defined in {code.co_filename}:{code.co_firstlineno}
            line {instruction.line_no + code.co_firstlineno}: {source_lines[instruction.line_no].rstrip()}"""
            return RuntimeError(msg)

        def to_key(instr: ThunderInstruction, scope: VariableScope) -> VariableKey:
            arg = instr.arg
            assert arg is not None

            if scope == VariableScope.CONST:
                return VariableKey(code.co_consts[arg], scope)

            elif scope == VariableScope.LOCAL:
                return VariableKey(code.co_varnames[arg], scope)

            elif scope == VariableScope.NONLOCAL:
                # TODO: Support nonlocal variables.
                # Nonlocal variables load (LOAD_DEREF) from frame->localsplus.
                # We cannot model or access the content of stack frames here.
                # We will have to emit some nonlocal AbstractValue and resolve it later, once we can prove
                # See https://github.com/python/cpython/blob/0ba07b2108d4763273f3fb85544dde34c5acd40a/Include/internal/pycore_code.h#L119-L133
                # for more explanation of localsplus.
                raise make_unsupported("nonlocal variables are not supported but instruction = ", instr)

            elif scope == VariableScope.GLOBAL:
                return VariableKey(code.co_names[arg], scope)

            elif scope == VariableScope.STACK:
                raise RuntimeError("Indexing into the stack is not permitted. Use `peek_stack` instead.")

            else:
                raise NotImplementedError("Unknown variable scope: {scope}")

        assert instructions
        for instruction in instructions:
            if instruction.opname == EXTENDED_ARG:
                # these are already reflexted in the next opcode's argument
                continue

            elif instruction in UNSAFE_OPCODES:
                raise make_unsupported("Unsupported instruction = ", instruction)

            assert hasattr(instruction, "line_no"), instruction
            pop, push = stack_effect_adjusted(instruction)

            def assert_expected_stack_effects(pop_i: int, push_i: tuple[int, ...]) -> None:
                assert (pop, push) == (pop_i, push_i), f"{instruction=} {pop=} {push=}"

            # Peek at the stack to track variable mutations.
            if (store_scope := STORE_OPNAMES.get(instruction.opname)) is not None:
                assert_expected_stack_effects(1, ())
                end_variables[to_key(instruction, store_scope)] = peek_stack(pop=False)

            elif (del_scope := DEL_OPNAMES.get(instruction.opname)) is not None:
                assert_expected_stack_effects(1, ())
                end_variables[to_key(instruction, del_scope)] = None

            # Handle stack inputs and outputs.
            inputs = [peek_stack(pop=True) for _ in range(pop)]
            new_intermediates: list[IntermediateValue] = []

            def lookup(index: int) -> _Symbolic.Output:
                """Handle alias resolution and new outputs."""
                if index < 0:
                    # Negative values index into the inputs.
                    return index

                elif index == len(new_intermediates):
                    new_intermediates.append(IntermediateValue())

                return new_intermediates[index]

            if (load_scope := LOAD_OPNAMES.get(instruction.opname)) is not None:
                assert_expected_stack_effects(0, PushNew)
                loaded = end_variables[load_key := to_key(instruction, load_scope)]
                assert loaded is not None, f"Access to deleted variable: {load_key}, {instruction}"
                stack.append(loaded)

            elif not (store_scope or del_scope):
                # We have already functionalized variable accesses, so we can prune loads and stores.
                outputs = [lookup(index) for index in push]
                stack.extend(_OutputRef(instruction, idx) for idx in range(len(outputs)))
                flow[instruction] = _Symbolic(tuple(reversed(inputs)), tuple(outputs))

        for idx, v in enumerate(block_inputs, start=-len(block_inputs)):
            begin_variables[VariableKey(idx, VariableScope.STACK)] = v

        for idx, v_ref in enumerate(stack, start=-len(block_inputs)):
            end_variables[VariableKey(idx, VariableScope.STACK)] = v_ref

        return cls(flow, begin_variables, {k: v for k, v in end_variables.items() if k.scope != VariableScope.CONST})

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
        materialized: Mapping[ThunderInstruction, _Materialized],
    ) -> AbstractValue:
        # We need to index while materializing (before `self.materialized` is
        # available) so we factor the logic into a standlone method.
        if key is None:
            return ValueMissing()

        elif isinstance(key, _OutputRef):
            return materialized[key.instruction].outputs[key.idx]

        assert isinstance(key, VariableKey)
        if key.scope == VariableScope.CONST:
            return ExternalRef(key)

        assert index in (0, -1)
        if index == 0:
            return variables[0][key]
        return cls._getitem_impl(0, variables[1][key], variables, materialized)

    @staticmethod
    def _boundary_state(
        variables: Mapping[VariableKey, T],
        resolve: Callable[[T], AbstractValue],
    ) -> Iterator[tuple[VariableKey, AbstractValue]]:
        for k, v in sorted(variables.items(), key=lambda kv: kv[0]):
            assert k.scope != VariableScope.CONST
            if not isinstance(v_resolved := resolve(v), ValueMissing):
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

    raw_instructions: tuple[ThunderInstruction, ...]  # For debugging only.
    flow: IntraBlockFlow
    jump_targets: tuple[JumpTarget, ...] = dataclasses.field(default=(), init=False)
    uses: OrderedSet[VariableKey] = dataclasses.field(default_factory=OrderedSet, init=False)

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
    def node_flow(self) -> Iterable[tuple[ThunderInstruction, _Materialized]]:
        yield from self.flow.materialized.items()

    @property
    def begin_state(self) -> Iterator[tuple[VariableKey, AbstractValue]]:
        yield from self.flow._boundary_state(self.flow._begin, lambda v: v)

    @property
    def end_state(self) -> Iterator[tuple[VariableKey, AbstractValue]]:
        yield from (flow := self.flow)._boundary_state(flow._end, lambda v_ref: flow[0, v_ref])

    @property
    def _flow_uses(self) -> Iterable[VariableKey]:
        flat_inputs = OrderedSet(itertools.chain(*(n.inputs for _, n in self.flow.symbolic)))
        yield from (i for i in flat_inputs if isinstance(i, VariableKey))

    def transform(self, replace_map: ReplaceMap) -> ProtoBlock:
        """Create a copy of `self` but allow values to be substituted."""
        transformed = ProtoBlock(self.raw_instructions, self.flow.substitute(replace_map))
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
            is_return = tuple(protoblock.flow.symbolic)[-1][0].opname == RETURN_VALUE
            G.add_node(protoblock, is_return=is_return)

        for protoblock in protoblocks:
            for destination, jump in protoblock.jump_targets:
                G.add_edge(protoblock, destination, adjacent=not jump)

        self.protoblocks = tuple(sort_adjacent(G))
        assert len(G) == len(self.protoblocks) == len(protoblocks), (len(G), len(self.protoblocks), len(protoblocks))

        self.root = self.protoblocks[0]
        root_stack = [(k, v) for k, v in self.root.begin_state if k.scope == VariableScope.STACK]
        assert not root_stack, f"Root block should not have stack inputs: {root_stack}"
        nodes = cast(Iterable[ProtoBlock], G.nodes)  # For some reason mypy needs this.
        self.parents = {protoblock: tuple(G.predecessors(protoblock)) for protoblock in nodes}

    @property
    def edges(self) -> Iterator[tuple[ProtoBlock, ProtoBlock]]:
        for protoblock in self.protoblocks:
            yield from ((protoblock, target) for target, _ in protoblock.jump_targets)

    def substitute(self, transformed: dict[ProtoBlock, ProtoBlock]) -> ProtoGraph:
        """Copies the ProtoGraph with block level substitutions while retaining the same topology."""
        assert not (delta := OrderedSet(transformed.keys()) - OrderedSet(self)), delta

        # TODO(robieta): Right now block order is load bearing, so we have to preserve it.
        transformed = {k: transformed.get(k) or k.transform({}) for k in self}
        for old_protoblock, new_protoblock in transformed.items():
            for old_target, is_jump in old_protoblock.jump_targets:
                new_protoblock.add_jump_target(transformed[old_target], is_jump)

        return ProtoGraph(transformed.values())

    def transform(self, replace_map: dict[AbstractValue, AbstractValue]) -> ProtoGraph:
        """Copies the ProtoGraph with value replacements.

        NOTE: This is strictly a condensing transform, and this is only invertable
              (using another `.transform` call) in trivial cases.
        """
        assert (v := replace_map.get(ValueMissing(), missing := object())) is missing, v
        replace_map = dict(flatten_map(replace_map))
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
class ValueMissing(AbstractValue):
    """Models `del` and similar operations. (But NOT `= None`)"""

    pass


class IntermediateValue(AbstractValue):
    """A (potentially) new value produced by an instruction."""

    pass


@dataclasses.dataclass(frozen=True, eq=True)
class ExternalRef(AbstractValue):
    """Reference values outside of the parsed code. (Arguments, constants, globals, etc.)"""

    key: VariableKey


@dataclasses.dataclass(frozen=True, eq=True)
class AbstractPhiValue(AbstractValue):
    """A value which aliases one of several inputs."""

    constituents: tuple[AbstractValue, ...]

    def __post_init__(self) -> None:
        # Flatten nested PhiValues. e.g.
        #   𝜙[𝜙[A, B], 𝜙[A, C]] -> 𝜙[A, B, C]
        constituents = itertools.chain(*[self.flatten(i) for i in self.constituents])

        # Ensure a consistent order.
        constituents = tuple(v for _, v in sorted({hash(v): v for v in constituents}.items()))
        object.__setattr__(self, "constituents", constituents)

    def _unpack_apply(self, replace_map: ReplaceMap) -> AbstractValue:
        result = self.__class__(tuple(i.substitute(replace_map) for i in self.constituents))
        return result if len(result.constituents) > 1 else result.constituents[0]

    @classmethod
    def flatten(cls, v: AbstractValue) -> Iterable[AbstractValue]:
        if isinstance(v, AbstractPhiValue):
            for i in v.constituents:
                yield from cls.flatten(i)
        else:
            yield v


def is_detail(v: AbstractValue) -> bool:
    if isinstance(v, AbstractRef) or type(v) is AbstractValue:
        return True
    return isinstance(v, AbstractPhiValue) and any(is_detail(i) for i in v.constituents)
