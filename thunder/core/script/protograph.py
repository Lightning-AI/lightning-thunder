import collections
import dataclasses
import dis
import itertools
import marshal
from types import CodeType
from typing import Any, Deque, Literal, Optional, NamedTuple
from collections.abc import Iterable, Iterator

import networkx as nx

from thunder.core.script.python_ir_data import (
    del_opcodes,
    load_opcodes,
    store_opcodes,
    stack_effects_comprehensive,
    NoBranchDependent,
    PushNew,
    VariableScope,
    RETURN_VALUE,
)
from thunder.core.utils import OrderedSet


# Aliases to make type annotations more expressive.
IsJump = bool
EdgeIndex = Literal[0, -1]
_AbstractValues = tuple["_AbstractValue", ...]
NodeFlow = tuple[dis.Instruction, _AbstractValues, _AbstractValues]


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

    def __lt__(self, other: "VariableKey") -> bool:
        try:
            return (self.scope.value, self.identifier) < (other.scope.value, other.identifier)
        except TypeError:
            # We prefer to use native ordering. However for unorderable types (e.g. CodeType)
            # `marshal` at least provides a consistent ordering.
            assert self.scope == VariableScope.CONST and other.scope == VariableScope.CONST
            return marshal.dumps(self.identifier) < marshal.dumps(other.identifier)


class _AbstractValue:
    """Represents a value during instruction parsing. (Prior to type binding.)"""

    def __hash__(self) -> int:
        return hash(id(self))


class AbstractValue(_AbstractValue):
    """Abstract value which is suitable for wider use. (Namely Value/PhiValue conversion.)"""

    pass


@dataclasses.dataclass(frozen=True, eq=False)
class ProtoBlock:
    raw_instructions: tuple[dis.Instruction]
    stacks: tuple[Deque[_AbstractValue], Deque[_AbstractValue]]
    stack_conditional: tuple[bool, _AbstractValues]
    variables: dict[VariableKey, _AbstractValues]
    node_flow: tuple[NodeFlow, ...]

    jump_targets: tuple[tuple["ProtoBlock", IsJump]] = ()
    uses: OrderedSet[VariableKey] = dataclasses.field(default_factory=OrderedSet)

    def __repr__(self) -> str:
        return f"ProtoBlock: {hex(id(self))}"

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        values_used = set(itertools.chain(*(inputs for _, inputs, _ in self.node_flow)))
        self.uses.update(k for k, v in self.state(index=0) if v in values_used)

    def add_jump_target(self, other: "ProtoBlock", jump: bool) -> None:
        """We need to add jump targets after all ProtoBlocks are initialized."""

        # Override `frozen=True` for this one limited use case.
        object.__setattr__(self, "jump_targets", self.jump_targets + ((other, jump),))

    @staticmethod
    def from_instructions(instructions: Iterable[dis.Instruction], code: CodeType) -> "ProtoBlock":
        raw_instructions = tuple(instructions)

        # Parse abstract flow. The inital pass is built using only instructions
        # in the ProtoBlock. Later passes will extend for block connectivity.
        stack_conditional = (False, ())
        block_inputs: Deque[_AbstractValue] = collections.deque()
        stack: Deque[_AbstractValue] = collections.deque()
        variables: dict[VariableKey, list[_AbstractValue]] = {}
        node_flow: list[NodeFlow] = []

        def peek_stack(pop: bool = False) -> _AbstractValue:
            # If the stack is empty we can infer that we are trying to reference
            # a value was already on the stack at the start of the block.
            if not stack:
                block_inputs.appendleft(
                    inferred := _AbstractRef(f"Inferred stack input: {len(block_inputs)}")
                )
                stack.append(inferred)
            return stack.pop() if pop else stack[-1]

        def peek_variable(arg: Optional[int], scope: VariableScope) -> list[_AbstractValue]:
            # The first value should always be an _AbstractRef; if a variable is
            # undefined a later pass will convert it to `ValueMissing`. However
            # local block parsing is too early to make that determination.
            assert arg is not None
            if scope == VariableScope.CONST:
                return [ExternalRef(VariableKey(code.co_consts[arg], scope))]

            identifier = {
                VariableScope.LOCAL: code.co_varnames,
                VariableScope.NONLOCAL: code.co_freevars,
                VariableScope.GLOBAL: code.co_names,
            }[scope][arg]

            # The first value should always be an _AbstractRef; if a variable is
            # undefined a later pass will convert it to `ValueMissing`. However
            # local block parsing is too early to make that determination.
            default = _AbstractRef(f"Variable initial value: ({identifier} {scope})")
            return variables.setdefault(VariableKey(identifier, scope), [default])

        assert raw_instructions
        for idx, instruction in enumerate(raw_instructions):
            inputs: list[_AbstractValue] = []
            outputs: list[_AbstractValue] = []
            new_intermediates: list[IntermediateValue] = []
            pop, push, (extra_branch, extra) = stack_effects_comprehensive(instruction)
            assert (idx + 1) == len(raw_instructions) or not extra, "Branch instruction mid-block"

            def assert_expected_stack_effects(*expected) -> None:
                assert (pop, push, (extra_branch, extra)) == tuple(
                    expected
                ), f"{instruction=} {pop=} {push=} {(extra_branch, extra)=}"

            # Peek at the stack to track variable mutations.
            if (store_scope := store_opcodes.get(instruction.opname)) is not None:
                assert_expected_stack_effects(1, (), NoBranchDependent)
                peek_variable(instruction.arg, store_scope).append(peek_stack(pop=False))

            elif (del_scope := del_opcodes.get(instruction.opname)) is not None:
                assert_expected_stack_effects(1, (), NoBranchDependent)
                peek_variable(instruction.arg, del_scope).append(ValueMissing())

            # Handle stack inputs.
            for _ in range(pop):
                inputs.append(peek_stack(pop=True))

            def lookup(index: int) -> _AbstractValue:
                """Handle alias resolution and new outputs."""
                if index < 0:
                    # Negative values index into the popped values.
                    return inputs[-1 - index]

                if index == len(new_intermediates):
                    new_intermediates.append(IntermediateValue())

                return new_intermediates[index]

            # Handle outputs.
            if (load_scope := load_opcodes.get(instruction.opname)) is not None:
                assert_expected_stack_effects(0, PushNew, NoBranchDependent)
                outputs.append(peek_variable(instruction.arg, load_scope)[-1])

            else:
                outputs.extend(lookup(index) for index in push)
                stack_conditional = (extra_branch, tuple(lookup(index) for index in extra))

                # Nodes on node flow:
                #   1) We have already functionalized variable accesses, so we can prune loads and stores.
                #   2) Inputs are consumed from the stack so we reverse them to get argument order.
                #   3) Conditional outputs are a block level concept. They are always added to node outputs.
                extra_outputs = tuple(lookup(index) for index in extra)
                stack_conditional = (extra_branch, extra_outputs)
                if not (store_scope or del_scope):
                    node_flow.append(
                        (instruction, tuple(reversed(inputs)), tuple((*outputs, *extra_outputs)))
                    )

            stack.extend(outputs)

        stacks = (block_inputs, stack)
        return ProtoBlock(raw_instructions, stacks, stack_conditional, variables, tuple(node_flow))

    @staticmethod
    def stack_args(args: Iterable[VariableKey]) -> tuple[int, ...]:
        stack_args = tuple(sorted([i.identifier for i in args if i.scope == VariableScope.STACK]))
        assert stack_args == tuple(range(-len(stack_args), 0)), stack_args
        return stack_args

    def stack_effect(self, is_jump: bool) -> tuple[int, int]:
        return len(tuple(self.stack(0, None))), len(tuple(self.stack(-1, is_jump)))

    def stack(self, index: EdgeIndex, is_jump: Optional[bool]) -> Iterable[_AbstractValue]:
        yield from self.stacks[index]
        extra_branch, extra_outputs = self.stack_conditional
        if index == -1 and is_jump in (extra_branch, None):
            yield from extra_outputs

    def state(
        self, index: EdgeIndex, is_jump: Optional[bool] = None
    ) -> Iterator[tuple[VariableKey, _AbstractValue]]:
        for k, values in sorted(self.variables.items(), key=lambda x: x[0]):
            v = values[index]
            assert k.scope not in (VariableScope.CONST, VariableScope.STACK) and isinstance(
                v, _AbstractValue
            ), (k, v)
            yield k, v

        stack = tuple(self.stack(index, is_jump))
        for idx, v in enumerate(stack):
            assert isinstance(v, _AbstractValue), v
            yield VariableKey(idx - len(stack), VariableScope.STACK), v

    def transform(self, replace_map: dict[_AbstractValue, _AbstractValue]) -> "ProtoBlock":
        """Create a copy of `self` but allow values to be substituted."""

        def make_replacement(values):
            cls = values.__class__
            assert cls in (list, tuple, collections.deque)
            assert all(isinstance(i, _AbstractValue) for i in values)
            return cls((replace_map.get(i, i) for i in values))

        extra_branch, extra_outputs = self.stack_conditional
        node_flow = [
            (instruction, make_replacement(inputs), make_replacement(outputs))
            for instruction, inputs, outputs in self.node_flow
        ]

        transformed = ProtoBlock(
            self.raw_instructions,
            tuple(make_replacement(stack) for stack in self.stacks),
            (extra_branch, make_replacement(extra_outputs)),
            {k: make_replacement(v) for k, v in self.variables.items()},
            tuple(node_flow),
        )
        transformed.uses.update(replace_map.get(i, i) for i in self.uses)

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
        (self.root,) = [protoblock for protoblock in protoblocks if not G.in_degree(protoblock)]
        assert not self.root.stacks[0], "Root block should not have stack inputs"

        # self.parents = {protoblock: tuple(G.predecessors(protoblock)) for protoblock in protoblocks}
        parents = collections.defaultdict(list)
        for protoblock in protoblocks:
            parents.setdefault(protoblock, [])
            for child, is_jump in protoblock.jump_targets:
                parents[child].append((protoblock, is_jump))
        self.parents = {block: tuple(block_parents) for block, block_parents in parents.items()}

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

    def transform(self, replace_map: dict[_AbstractValue, _AbstractValue]) -> "ProtoGraph":
        transformed = {protoblock: protoblock.transform(replace_map) for protoblock in self}
        for old_protoblock, new_protoblock in transformed.items():
            for old_target, is_jump in old_protoblock.jump_targets:
                new_protoblock.add_jump_target(transformed[old_target], is_jump)

        return ProtoGraph(transformed.values())

    def __iter__(self) -> Iterator[ProtoBlock]:
        yield from self.protoblocks


@dataclasses.dataclass(frozen=True, eq=False)
class _AbstractRef(_AbstractValue):
    """Placeholder value which will be resolved during parsing."""

    _debug_info: str = "N/A"


class ValueMissing(AbstractValue):
    """Models `del` and similar operations. (But NOT `= None`)"""

    pass


class IntermediateValue(AbstractValue):
    """A (potentially) new value produced by an instruction."""

    pass


@dataclasses.dataclass(frozen=True, eq=False)
class ExternalRef(AbstractValue):
    """Reference values outside of the parsed code. (Arguments, constants, globals, etc.)"""

    key: VariableKey


@dataclasses.dataclass(frozen=True, eq=True)
class AbstractPhiValue(AbstractValue):
    """A value which aliases one of several inputs."""

    constituents: tuple[AbstractValue]

    def __post_init__(self) -> None:
        assert all(isinstance(i, AbstractValue) for i in self.constituents), self.constituents

        # Ensure a consistent order.
        constituents = tuple(v for _, v in sorted({id(v): v for v in self.constituents}.items()))
        object.__setattr__(self, "constituents", constituents)
