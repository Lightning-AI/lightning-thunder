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
    stack_effect_adjusted,
    PushNew,
    VariableScope,
    RETURN_VALUE,
)
from thunder.core.utils import OrderedSet


# Aliases to make type annotations more expressive.
JumpTarget = tuple["ProtoBlock", bool]
_AbstractValues = tuple["_AbstractValue", ...]
NodeFlow = tuple[dis.Instruction, _AbstractValues, _AbstractValues]
EdgeIndex = Literal[0, -1]


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
    """Stores abstract data flow for a code block.

    Notes:
        `variables` should not contain `VariableScope.CONST` or `VariableScope.STACK`
        `uses` is indexed WRT block inputs. (This matters for `VariableScope.STACK`)
    """

    raw_instructions: tuple[dis.Instruction, ...]
    begin_stack: Deque[_AbstractValue]
    stack_effect: tuple[int, _AbstractValues]
    variables: dict[VariableKey, _AbstractValues]
    node_flow: tuple[NodeFlow, ...]

    jump_targets: tuple[JumpTarget, ...] = ()
    uses: OrderedSet[VariableKey] = dataclasses.field(default_factory=OrderedSet)

    def __repr__(self) -> str:
        return f"ProtoBlock: {hex(id(self))}"

    def __hash__(self) -> int:
        return id(self)

    def __post_init__(self) -> None:
        values_used = set(itertools.chain(*(inputs for _, inputs, _ in self.node_flow)))
        self.uses.update(k for k, v in self.variable_state(index=0) if v in values_used)

    def add_jump_target(self, other: "ProtoBlock", jump: bool) -> None:
        """We need to add jump targets after all ProtoBlocks are initialized."""

        # Override `frozen=True` for this one limited use case.
        object.__setattr__(self, "jump_targets", self.jump_targets + ((other, jump),))

    @staticmethod
    def from_instructions(instructions: Iterable[dis.Instruction], code: CodeType) -> "ProtoBlock":
        raw_instructions = tuple(instructions)

        # Parse abstract flow. The inital pass is built using only instructions
        # in the ProtoBlock. Later passes will extend for block connectivity.
        block_inputs: Deque[_AbstractValue] = collections.deque()
        stack: Deque[_AbstractValue] = collections.deque()
        variables: dict[VariableKey, list[_AbstractValue]] = {}
        node_flow: list[NodeFlow] = []

        def peek_stack(pop: bool = False) -> _AbstractValue:
            # If the stack is empty we can infer that we are trying to reference
            # a value was already on the stack at the start of the block.
            if not stack:
                inferred = _AbstractRef(f"Inferred stack input: {len(block_inputs)}")
                stack.append(inferred)
                block_inputs.appendleft(inferred)
            return stack.pop() if pop else stack[-1]

        def peek_variable(instr: dis.Instruction, scope: VariableScope) -> list[_AbstractValue]:
            """
            Get the _AbstractValue of the variable given by the instruction and variable scope.

            Returns the list of _AbstractValues associated with the variable.

            The first value of the list is created here and should always be an _AbstractRef;
            if a variable is undefined a later pass will convert it to `ValueMissing`.
            Local block parsing is too early to make that determination.

            The last value is the current (abstract) value of the variable.
            """
            arg = instr.arg
            assert arg is not None
            if scope == VariableScope.CONST:
                identifier = code.co_consts[arg]
                return [ExternalRef(VariableKey(identifier, scope))]
            elif scope == VariableScope.LOCAL:
                identifier = code.co_varnames[arg]
                default = _AbstractRef(f"Variable initial value: ({identifier} {scope})")
                return variables.setdefault(VariableKey(identifier, scope), [default])
            elif scope == VariableScope.NONLOCAL:
                # TODO: Support nonlocal variables.
                # Nonlocal variables load (LOAD_DEREF) from frame->localsplus.
                # We cannot model or access the content of stack frames here.
                # We will have to emit some nonlocal _AbstractValue and resolve it later, once we can prove
                # See https://github.com/python/cpython/blob/0ba07b2108d4763273f3fb85544dde34c5acd40a/Include/internal/pycore_code.h#L119-L133
                # for more explanation of localsplus.
                msg = f"nonlocal variables are not supported but instruction = {instr} found"
                raise RuntimeError(msg)
            elif scope == VariableScope.GLOBAL:
                identifier = code.co_names[arg]
                default = _AbstractRef(f"Variable initial value: ({identifier} {scope})")
                return variables.setdefault(VariableKey(identifier, scope), [default])
            elif scope == VariableScope.STACK:
                raise RuntimeError("Peeking stack variables is not supported.")
            else:
                raise NotImplementedError("Unknown variable scope: {scope}")


        assert raw_instructions
        for instruction in raw_instructions:
            assert hasattr(instruction, "line_no"), instruction
            pop, push = stack_effect_adjusted(instruction)

            def assert_expected_stack_effects(*expected) -> None:
                assert (pop, push) == tuple(expected), f"{instruction=} {pop=} {push=}"

            # Peek at the stack to track variable mutations.
            if (store_scope := store_opcodes.get(instruction.opname)) is not None:
                assert_expected_stack_effects(1, ())
                peek_variable(instruction, store_scope).append(peek_stack(pop=False))

            elif (del_scope := del_opcodes.get(instruction.opname)) is not None:
                assert_expected_stack_effects(1, ())
                peek_variable(instruction, del_scope).append(ValueMissing())

            # Handle stack inputs and outputs.
            inputs: list[_AbstractValue] = [peek_stack(pop=True) for _ in range(pop)]
            outputs: list[_AbstractValue]
            new_intermediates: list[IntermediateValue] = []

            def lookup(index: int) -> _AbstractValue:
                """Handle alias resolution and new outputs."""
                if index < 0:
                    # Negative values index into the popped values.
                    return inputs[-1 - index]

                if index == len(new_intermediates):
                    new_intermediates.append(IntermediateValue())

                return new_intermediates[index]

            if (load_scope := load_opcodes.get(instruction.opname)) is not None:
                assert_expected_stack_effects(0, PushNew)
                outputs = [peek_variable(instruction, load_scope)[-1]]

            else:
                outputs = [lookup(index) for index in push]

            # We have already functionalized variable accesses, so we can prune loads and stores.
            # Inputs are consumed from the stack so we reverse them to get argument order.
            if not (store_scope or del_scope or load_scope):
                node_flow.append((instruction, tuple(reversed(inputs)), tuple(outputs)))

            stack.extend(outputs)

        return ProtoBlock(
            raw_instructions=instructions,
            begin_stack=block_inputs,
            stack_effect=(len(block_inputs), tuple(stack)),
            variables={k: tuple(v) for k, v in variables.items()},
            node_flow=tuple(node_flow),
        )

    @staticmethod
    def stack_args(args: Iterable[VariableKey]) -> tuple[int, ...]:
        stack_args = tuple(sorted([i.identifier for i in args if i.scope == VariableScope.STACK]))
        assert stack_args == tuple(range(-len(stack_args), 0)), stack_args
        return stack_args

    @property
    def end_stack(self) -> Iterable[_AbstractValue]:
        pop, push = self.stack_effect
        yield from itertools.islice(self.begin_stack, 0, len(self.begin_stack) - pop)
        yield from push

    def variable_state(self, *, index: EdgeIndex) -> Iterator[tuple[VariableKey, _AbstractValue]]:
        for k, values in sorted(self.variables.items(), key=lambda x: x[0]):
            v = values[index]
            assert k.scope not in (VariableScope.CONST, VariableScope.STACK), k
            assert isinstance(v, _AbstractValue), v
            yield k, v

        stack = tuple(self.begin_stack if index == 0 else self.end_stack)
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

        pop, push = self.stack_effect
        node_flow = tuple(
            (instruction, make_replacement(inputs), make_replacement(outputs))
            for instruction, inputs, outputs in self.node_flow
        )

        transformed = ProtoBlock(
            raw_instructions=self.raw_instructions,
            begin_stack=make_replacement(self.begin_stack),
            stack_effect=(pop, make_replacement(push)),
            variables={k: make_replacement(v) for k, v in self.variables.items()},
            node_flow=node_flow,
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
        assert not self.root.begin_stack, "Root block should not have stack inputs"
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

    def transform(self, replace_map: dict[_AbstractValue, _AbstractValue]) -> "ProtoGraph":
        assert (v := replace_map.get(ValueMissing())) is None, v
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


@dataclasses.dataclass(frozen=True, eq=True)
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

        # Flatten nested PhiValues. e.g.
        #   𝜙[𝜙[A, B], 𝜙[A, C]] -> 𝜙[A, B, C]
        constituents = itertools.chain(*[self.flatten(i) for i in self.constituents])

        # Ensure a consistent order.
        constituents = tuple(v for _, v in sorted({id(v): v for v in constituents}.items()))
        object.__setattr__(self, "constituents", constituents)

    @classmethod
    def flatten(cls, v: AbstractValue):
        if isinstance(v, AbstractPhiValue):
            for i in v.constituents:
                yield from cls.flatten(i)
        else:
            yield v
