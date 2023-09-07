import collections
import dataclasses
import functools
import inspect
import itertools
import marshal
import textwrap
from types import CodeType, MappingProxyType
from typing import final, Any, Deque, Literal, NamedTuple, TypeVar
from collections.abc import Iterable, Iterator, Mapping

import networkx as nx

from thunder.core.script.instrumentation import InstrumentingBase
from thunder.core.script.python_ir_data import (
    stack_effect_adjusted,
    PushNew,
    ThunderInstruction,
    VariableScope,
    DEL_OPNAMES,
    LOAD_OPNAMES,
    STORE_OPNAMES,
    RETURN_VALUE,
    EXTENDED_ARG,
)
from thunder.core.utils import OrderedSet

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
            and type(self.identifier) is type(other.identifier)
            and self.identifier == other.identifier
        )

    def __lt__(self, other: "VariableKey") -> bool:
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
ReplaceMap = dict["_AbstractValue", "_AbstractValue"]
_AbstractValues = tuple["_AbstractValue", ...]


class _AbstractValue:
    """Represents a value during instruction parsing. (Prior to type binding.)"""

    def __copy__(self) -> "_AbstractValue":
        raise NotImplementedError

    @property
    def is_detail(self) -> bool:
        return AbstractValue not in self.__class__.__mro__

    @final
    def substitute(self, replace_map: ReplaceMap) -> "_AbstractValue":
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
                See `_flatten_replace_map`.
            """
            raise ValueError(textwrap.dedent(msg))

        return new_self._unpack_apply(replace_map)

    def _unpack_apply(self, _: ReplaceMap) -> "_AbstractValue":
        """Recursively update any constituent references in the abstract value."""
        return self

    @staticmethod
    def _flatten_replace_map(replace_map: ReplaceMap) -> None:
        """Remove intermediate steps in the replace map.

        For example, if we `replace_map` is `{A: B, B: C}` we will update it to `{A: C, B: C}`.
        (Effectively converting A->B->C to A->C.)
        """
        G = nx.from_edgelist(replace_map.items(), nx.DiGraph).reverse()
        G.remove_edges_from(nx.selfloop_edges(G))
        assert nx.is_directed_acyclic_graph(G)
        for cluster in nx.connected_components(G.to_undirected()):
            (root,) = (i for i in cluster if not G.in_degree(i))
            replace_map.update({i: root for i in cluster if i is not root})


class AbstractValue(_AbstractValue):
    """Abstract value which is suitable for wider use. (Namely Value/PhiValue conversion.)"""

    pass


# =============================================================================
# == Intra-ProtoBlock abstract value flow =====================================
# =============================================================================
#
# `ProtoBlocks` employ a dual representation, where node inputs and outputs can
# be viewed as either a reference based DAG or a sequence of ops with concrete
# `_AbstractValue` inputs and outputs.
#
# At the boundaries of a ProtoBlock values have named (VariableKey) slots;
# within the ProtoBlock there is no need for such slots (since there is no
# control flow within a block and those named slots tell you how to build the
# directed *cyclic* graph for the larger program) so they are stripped during
# parsing.
#
# The inputs of a protoblock are stored as a map of `VariableKey -> _AbstractValue`
# and act as the intra-block DAG sources. The outputs are stored as references
# since every ProtoBlock output must have a unique producer. (Either an input
# or a node within the block.)
#
# The canonical representation for intra-block flow is "symbolic" (reference
# based). If an `_AbstractValue` appear in a symbolic node's outputs that
# indicates that the node is that value's producer. Otherwise all inputs and
# outputs are references: inputs reference either the begin state or the
# outputs of a prior node while output references index into the node's inputs.
#
# When analyzing a graph we are generally interested in the concrete properties
# of values; provenance is generally only important when connecting blocks and
# performing rewrites. For these cases `IntraBlockFlow` generates a
# "materialized" flow which resolves all references to `_AbstractValue`s. The
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
    # _AbstractValue:   New value created by this instruction
    Output = int | _AbstractValue
    outputs: tuple[Output, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class _Materialized:
    """Flow element where all symbolic references have been resolved to concrete `_AbstractValue`s."""

    inputs: _AbstractValues
    outputs: _AbstractValues


@dataclasses.dataclass(frozen=True, eq=False)
class IntraBlockFlow:
    _flow: Mapping[ThunderInstruction, _Symbolic]

    BeginVariables = Mapping[VariableKey, _AbstractValue]
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

        def resolve(inputs: _AbstractValues, o: _Symbolic.Output) -> _AbstractValue:
            assert isinstance(o, (int, _AbstractValue))
            return inputs[o] if isinstance(o, int) else o

        result: dict[ThunderInstruction, _Materialized] = {}
        for instruction, node in self.symbolic:
            inputs = [self._getitem_impl(0, i, (self._begin, self._end), result) for i in node.inputs]
            outputs = [resolve(inputs, o) for o in node.outputs]
            assert all(isinstance(o, _AbstractValue) for o in outputs), outputs
            result[instruction] = _Materialized(tuple(inputs), tuple(outputs))

        return MappingProxyType(result)

    def __getitem__(self, key: tuple[EdgeIndex, _Symbolic.Input]) -> _AbstractValue:
        return self._getitem_impl(*key, (self._begin, self._end), self.materialized)  # __getitem__ packs args

    def variable_state(self, *, index: EdgeIndex) -> Iterator[tuple[VariableKey, _AbstractValue]]:
        assert index in (0, -1), index
        for k in sorted(variables := (self._begin, self._end)[index]):
            assert k.scope != VariableScope.CONST
            v = variables[k] if index == 0 else self[0, variables[k]]
            if not isinstance(v, ValueMissing):
                yield k, v

    @classmethod
    def from_instructions(cls, instructions: tuple[ThunderInstruction, ...], code: CodeType) -> "IntraBlockFlow":
        flow: dict[ThunderInstruction, _Symbolic] = {}
        begin_variables: IntraBlockFlow.BeginVariables = {}
        end_variables: IntraBlockFlow.EndVariables = {}
        block_inputs: Deque[_AbstractValue] = collections.deque()
        stack: Deque[_Symbolic.Input] = collections.deque()

        def peek_stack(pop: bool = False) -> _Symbolic.Input:
            # If the stack is empty we can infer that we are trying to reference
            # a value was already on the stack at the start of the block.
            if not stack:
                index = -len(block_inputs) - 1
                block_inputs.appendleft(_AbstractRef(f"Inferred stack input: {index}"))
                stack.append(VariableKey(index, VariableScope.STACK))
            return stack.pop() if pop else stack[-1]

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
                # We will have to emit some nonlocal _AbstractValue and resolve it later, once we can prove
                # See https://github.com/python/cpython/blob/0ba07b2108d4763273f3fb85544dde34c5acd40a/Include/internal/pycore_code.h#L119-L133
                # for more explanation of localsplus.
                source_lines, first_line = inspect.getsourcelines(code)
                msg = f"""nonlocal variables are not supported but instruction = {instr} found
              {code.co_name} defined in {code.co_filename}:{code.co_firstlineno}
              line {instr.line_no + code.co_firstlineno}: {source_lines[instr.line_no].rstrip()}"""
                raise RuntimeError(msg)

            elif scope == VariableScope.GLOBAL:
                return VariableKey(code.co_names[arg], scope)

            elif scope == VariableScope.STACK:
                raise RuntimeError("Indexing into the stack is not permitted. Use `peek_stack` instead.")

            else:
                raise NotImplementedError("Unknown variable scope: {scope}")

        def peek_variable(instr: ThunderInstruction, scope: VariableScope) -> IntraBlockFlow.EndVariable:
            key = to_key(instr, scope)
            if scope == VariableScope.CONST:
                return key

            v = end_variables.get(key, missing := object())
            assert v is not None, f"Access to deleted variable: {key}"
            if v is missing:
                default = _AbstractRef(f"Variable initial value: ({key.identifier} {key.scope})")
                begin_variables.setdefault(key, default)
                v = end_variables[key] = key

            return v

        assert instructions
        for instruction in instructions:
            if instruction.opname == EXTENDED_ARG:
                # these are already reflexted in the next opcode's argument
                continue

            assert hasattr(instruction, "line_no"), instruction
            pop, push = stack_effect_adjusted(instruction)

            def assert_expected_stack_effects(*expected) -> None:
                assert (pop, push) == tuple(expected), f"{instruction=} {pop=} {push=}"

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
                stack.append(peek_variable(instruction, load_scope))

            elif not (store_scope or del_scope):
                # We have already functionalized variable accesses, so we can prune loads and stores.
                outputs = [lookup(index) for index in push]
                stack.extend(_OutputRef(instruction, idx) for idx in range(len(outputs)))
                flow[instruction] = _Symbolic(tuple(reversed(inputs)), tuple(outputs))

        for idx, v in enumerate(block_inputs, start=-len(block_inputs)):
            begin_variables[VariableKey(idx, VariableScope.STACK)] = v

        for idx, v_ref in enumerate(stack, start=-len(block_inputs)):
            end_variables[VariableKey(idx, VariableScope.STACK)] = v_ref

        return cls(flow, begin_variables, end_variables)

    def substitute(self, replace_map: ReplaceMap) -> "IntraBlockFlow":
        """Replace `_AbstractValue`s within the flow. (Block inputs and producer nodes.)"""

        def replace(x: T) -> T:
            return x.substitute(replace_map) if isinstance(x, _AbstractValue) else x

        return self.__class__(
            _flow={k: _Symbolic(v.inputs, tuple(replace(o) for o in v.outputs)) for k, v in self.symbolic},
            _begin={k: replace(v) for k, v in self._begin.items()},
            _end=self._end,
        )

    @classmethod
    def _getitem_impl(
        cls,
        index: EdgeIndex,
        key: _Symbolic.Input,
        variables: tuple["IntraBlockFlow.BeginVariables", "IntraBlockFlow.EndVariables"],
        materialized: Mapping[ThunderInstruction, _Materialized],
    ) -> _AbstractValue:
        # We need to index while materializing (before `self.materialized` is
        # available) so we factor the logic into a standlone method.
        if isinstance(key, _OutputRef):
            return materialized[key.instruction].outputs[key.idx]

        assert isinstance(key, VariableKey)
        if key.scope == VariableScope.CONST:
            return ExternalRef(key)

        assert index in (0, -1)
        v = variables[index][key]
        return v if index == 0 else cls._getitem_impl(0, v, variables, materialized)


# =============================================================================
# == Inter-ProtoBlock abstract value flow =====================================
# =============================================================================
#
# ProtoBlocks are weakly coupled by design. The `VariableKey` slots allow edges
# to be deduced (e.g. `x` at the start of one block must be the same as `x` at
# the end of the prior block), but there's no strong requirement. (And indeed,
# the ProtoGraph immediately after parsing has all unconnected `_AbstractRef`s
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

    def add_jump_target(self, other: "ProtoBlock", jump: bool) -> None:
        """We need to add jump targets after all ProtoBlocks are initialized."""

        # Override `frozen=True` for this one limited use case.
        object.__setattr__(self, "jump_targets", self.jump_targets + ((other, jump),))

    @property
    def node_flow(self) -> Iterable[tuple[ThunderInstruction, _Materialized]]:
        yield from self.flow.materialized.items()

    @property
    def begin_state(self):
        return self.flow.variable_state(index=0)

    @property
    def end_state(self):
        return self.flow.variable_state(index=-1)

    @property
    def _flow_uses(self) -> Iterable[VariableKey]:
        flat_inputs = OrderedSet(itertools.chain(*(n.inputs for _, n in self.flow.symbolic)))
        yield from (i for i in flat_inputs if isinstance(i, VariableKey))

    def transform(self, replace_map: ReplaceMap) -> "ProtoBlock":
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
        protoblocks = tuple(protoblocks)

        # Ensure the protoblocks form a single connected graph.
        G = nx.DiGraph()
        G.add_nodes_from(protoblocks)
        for protoblock in protoblocks:
            for destination, _ in protoblock.jump_targets:
                G.add_edge(protoblock, destination)
        assert len(G.nodes) == len(protoblocks), f"{len(G.nodes)=} {len(protoblocks)=}"
        connected_components = tuple(nx.connected_components(G.to_undirected()))
        assert len(connected_components) == 1, [len(i) for i in connected_components]

        # Compute some basic topological features.
        (self.root,) = (protoblock for protoblock in protoblocks if not G.in_degree(protoblock))
        root_stack = [(k, v) for k, v in self.root.begin_state if k.scope == VariableScope.STACK]
        assert not root_stack, f"Root block should not have stack inputs: {root_stack}"
        self.parents = {protoblock: tuple(G.predecessors(protoblock)) for protoblock in protoblocks}

        # Sort the nodes.
        #   Topological sort is not defined for cyclic graphs, however we don't
        #   need a true sort. Rather we need to respect `jump=False` connectivity.
        #   And because our graph represents a linear instruction stream we are
        #   guaranteed that a valid sort of just the non-jump graph exists.
        G = nx.DiGraph()
        for protoblock in protoblocks:
            G.add_node(protoblock)
            for destination, jump in protoblock.jump_targets:
                if not jump:
                    G.add_edge(protoblock, destination)

        protoblock_to_idx = {protoblock: idx for idx, protoblock in enumerate(protoblocks)}
        sort_map = {}
        for nodes in nx.connected_components(G.to_undirected()):
            assert all(G.in_degree(node) <= 1 and G.out_degree(node) <= 1 for node in nodes)
            nodes = tuple(nx.topological_sort(G.subgraph(nodes)))
            if nodes[0] is self.root:
                primary_key = -1
            elif nodes[-1].raw_instructions[-1].opname == RETURN_VALUE:
                primary_key = len(protoblocks) + 1
            else:
                primary_key = protoblock_to_idx[nodes[0]]
            sort_map.update({node: (primary_key, idx) for idx, node in enumerate(nodes)})

        self.protoblocks = tuple(sorted(protoblocks, key=lambda x: sort_map[x]))

    @property
    def edges(self) -> Iterator[tuple[ProtoBlock, ProtoBlock]]:
        for protoblock in self.protoblocks:
            yield from ((protoblock, target) for target, _ in protoblock.jump_targets)

    def substitute(self, transformed: dict[ProtoBlock, ProtoBlock]) -> "ProtoGraph":
        """Copies the ProtoGraph with block level substitutions while retaining the same topology."""
        assert not (delta := OrderedSet(transformed.keys()) - self), delta

        # TODO(robieta): Right now block order is load bearing, so we have to preserve it.
        transformed = {k: transformed.get(k) or k.transform({}) for k in self}
        for old_protoblock, new_protoblock in transformed.items():
            for old_target, is_jump in old_protoblock.jump_targets:
                new_protoblock.add_jump_target(transformed[old_target], is_jump)

        return ProtoGraph(transformed.values())

    def transform(self, replace_map: dict[_AbstractValue, _AbstractValue]) -> "ProtoGraph":
        """Copies the ProtoGraph with value replacements.

        NOTE: This is strictly a condensing transform, and this is only invertable
              (using another `.transform` call) in trivial cases.
        """
        assert (v := replace_map.get(ValueMissing(), missing := object())) is missing, v
        replace_map = replace_map.copy()
        _AbstractValue._flatten_replace_map(replace_map)
        return self.substitute({protoblock: protoblock.transform(replace_map) for protoblock in self})

    def unlink(self) -> "ProtoGraph":
        """Copies the ProtoGraph but replaces all block inputs with references. (Useful for graph rewrites.)"""
        transformed: dict[ProtoBlock, ProtoBlock] = {}
        for protoblock in self:
            begin_vars = (v for _, v in protoblock.variable_state(index=0) if not isinstance(v, _AbstractRef))
            replace_map = {v: _AbstractRef(f"Unlink: {v}") for v in begin_vars}
            transformed[protoblock] = new_protoblock = protoblock.transform(replace_map)
            new_protoblock.__post_init__()
        return self.substitute(transformed)

    def __iter__(self) -> Iterator[ProtoBlock]:
        yield from self.protoblocks

    def __getitem__(self, index) -> ProtoBlock:
        return self.protoblocks[index]

    def debug_print_protoflows(self):
        """
        Print out the node_flow for each protoblock in the
        protograph, in a way that's nice to read and debug with.
        """

        counter = 0
        idxes = {}
        for pb in self:
            for _, node in pb.node_flow:
                for val in itertools.chain(node.inputs, node.outputs):
                    if val not in idxes.keys():
                        idxes[val] = counter
                        counter += 1

        def to_index_str(values):
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
class _AbstractRef(_AbstractValue):
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

    constituents: tuple[AbstractValue]

    def __post_init__(self) -> None:
        # Flatten nested PhiValues. e.g.
        #   𝜙[𝜙[A, B], 𝜙[A, C]] -> 𝜙[A, B, C]
        constituents = itertools.chain(*[self.flatten(i) for i in self.constituents])

        # Ensure a consistent order.
        constituents = tuple(v for _, v in sorted({id(v): v for v in constituents}.items()))
        object.__setattr__(self, "constituents", constituents)
        assert not any(i.is_detail for i in self.constituents), self.constituents

    def _unpack_apply(self, replace_map: ReplaceMap) -> "AbstractValue":
        result = self.__class__(tuple(i.substitute(replace_map) for i in self.constituents))
        return result if len(result.constituents) > 1 else result.constituents[0]

    @classmethod
    def flatten(cls, v: AbstractValue):
        if isinstance(v, AbstractPhiValue):
            for i in v.constituents:
                yield from cls.flatten(i)
        else:
            yield v
