import collections
import dataclasses
import functools
import dis
import inspect
import itertools
import sys
from typing import Callable, Deque, Dict, List, Literal, Optional, Set, Tuple, Union
from collections.abc import Iterable, Iterator

import networkx as nx

from thunder.core.script.graph import (
    check_graph,
    replace_values,
    Block,
    Graph,
    MROAwareObjectRef,
    Node,
    NULL,
    PhiValue,
    Value,
)
from thunder.core.script.python_ir_data import (
    ArgScope,
    compute_jump,
    del_opcodes,
    get_instruction,
    jump_instructions,
    load_opcodes,
    make_jump_absolute,
    make_name_map,
    make_return,
    return_instructions,
    stack_effects_comprehensive,
    store_opcodes,
    unconditional_jump_names,
    NoBranchDependent,
    PushNew,
    VariableScope,
    SUPPORTS_PREPROCESSING,
)
from thunder.core.utils import dict_join, OrderedSet

# Aliases to make type annotations more expressive.
IsJump = bool
EdgeIndex = Union[Literal[0], Literal[-1]]
_AbstractValues = tuple["_AbstractValue", ...]
NodeFlow = tuple[dis.Instruction, _AbstractValues, _AbstractValues]


DEBUG_ASSERTS = False


def enable_debug_asserts():
    global DEBUG_ASSERTS
    DEBUG_ASSERTS = True


class Super:
    pass


def parse_bytecode(method: Callable) -> tuple["ProtoBlock", ...]:
    """Given a method, disassemble it to a sequence of simple blocks."""
    bytecode: tuple[dis.Instruction, ...] = tuple(dis.get_instructions(method))

    # Determine the boundaries for the simple blocks.
    split_after_opcodes = jump_instructions | return_instructions
    follows_jump = itertools.chain([0], (int(i.opcode in split_after_opcodes) for i in bytecode))
    new_block = (int(i or j.is_jump_target) for i, j in zip(follows_jump, bytecode))

    # Split the bytecode (and instruction number) into groups
    group_indices = tuple(itertools.accumulate(new_block))
    groups = itertools.groupby(enumerate(bytecode), key=lambda args: group_indices[args[0]])

    # Drop the group index, copy from the groupby iter, and unzip `enumerate`.
    groups = (zip(*tuple(i)) for _, i in groups)

    blocks: dict[int, tuple[int, list[dis.Instruction]]] = {
        start: (len(block), list(block)) for (start, *_), block in groups
    }

    # If the last instruction is not a jump or return (which means we split
    # because the next instruction was a jump target) then we need to tell
    # the current block how to advance.
    for start, (block_size, block) in blocks.items():
        assert block, "Block is empty"
        instruction = block[-1]
        if instruction.opcode not in split_after_opcodes:
            next_start = start + block_size
            assert bytecode[next_start].is_jump_target
            block.append(make_jump_absolute(next_start))

    # Consolidate `return` statements.
    def is_return(block: list[dis.Instruction]) -> bool:
        assert block and not any(i.opcode == dis.opmap["RETURN_VALUE"] for i in block[:-1])
        return block[-1].opcode == dis.opmap["RETURN_VALUE"]

    return_starts = tuple(start for start, (_, block) in blocks.items() if is_return(block))
    assert return_starts, "No return instruction found"
    if len(return_starts) > 1:
        new_return_start = len(bytecode) + 1
        for _, block in (blocks[start] for start in return_starts):
            prior_return = block.pop()
            assert prior_return.opcode == dis.opmap["RETURN_VALUE"], dis.opname[prior_return.opcode]
            block.append(make_jump_absolute(new_return_start))

        blocks[new_return_start] = 1, [make_return(is_jump_target=True)]
        assert is_return(blocks[new_return_start][1])

    # Assign line numbers once structure is finalized.
    line_no = 0
    for instruction in itertools.chain(*[block for _, block in blocks.values()]):
        if (starts_line := instruction.starts_line) is not None:
            line_no = starts_line - method.__code__.co_firstlineno
        instruction.line_no = line_no

    protoblocks = {start: ProtoBlock(instructions) for start, (_, instructions) in blocks.items()}

    # If the last instruction is a jump we need to compute the jump target(s).
    for start, (raw_block_len, (*_, last_instruction)) in blocks.items():
        if last_instruction.opcode in jump_instructions:
            end = start + raw_block_len - 1
            if last_instruction.opname not in unconditional_jump_names:  # Fallthrough
                protoblocks[start].jump_targets.append((protoblocks[end + 1], False))

            if (jump_offset := compute_jump(last_instruction, end)) is not None:  # Jump
                protoblocks[start].jump_targets.append((protoblocks[jump_offset], True))

    return tuple(protoblocks.values())


class _AbstractValue:
    """Represents a value during instruction parsing. (Prior to type binding.)"""

    def __hash__(self) -> int:
        return hash(id(self))


class AbstractValue(_AbstractValue):
    """Abstract value which is suitable for wider use. (Namely Value/PhiValue conversion.)"""

    pass


class ProtoBlock:
    raw_instructions: tuple[dis.Instruction]
    jump_targets: list[tuple["ProtoBlock", IsJump]]

    # (block_inputs, block_outputs)
    stacks: tuple[Deque[_AbstractValue], Deque[_AbstractValue]]
    stack_conditional: tuple[bool, _AbstractValues]

    # Should not contain `VariableScope.CONST` or `VariableScope.STACK`
    variables: dict[ArgScope, _AbstractValues]

    # Entries are indexed WRT block inputs. (This matters for `VariableScope.STACK`)
    uses: OrderedSet[ArgScope]

    # Instruction level value flow.
    node_flow: tuple[NodeFlow, ...]

    def __init__(self, instructions: Iterable[dis.Instruction]) -> None:
        self.raw_instructions = tuple(instructions)
        self.jump_targets = []

        # Parse abstract flow. The inital pass is built using only instructions
        # in the ProtoBlock. Later passes will extend for block connectivity.
        self.stack_conditional = (False, ())
        block_inputs: Deque[_AbstractValue] = collections.deque()
        stack: Deque[_AbstractValue] = collections.deque()
        variables: dict[ArgScope, list[_AbstractValue]] = {}
        node_flow: list[NodeFlow] = []

        def peek_stack(pop: bool = False) -> _AbstractValue:
            # If the stack is empty we can infer that we are trying to reference
            # a value was already on the stack at the start of the block.
            if not stack:
                block_inputs.appendleft(inferred := _AbstractRef(f"Inferred stack input: {len(block_inputs)}"))
                stack.append(inferred)
            return stack.pop() if pop else stack[-1]

        def peek_variable(arg: Optional[int], scope: VariableScope) -> list[_AbstractValue]:
            # The first value should always be an _AbstractRef; if a variable is
            # undefined a later pass will convert it to `ValueMissing`. However
            # local block parsing is too early to make that determination.
            assert arg is not None
            if scope == VariableScope.CONST:
                return [ExternalRef(arg, scope)]
            return variables.setdefault(
                ArgScope(arg, scope), [_AbstractRef(f"Variable initial value: ({arg} {scope})")]
            )

        assert self.raw_instructions
        for idx, instruction in enumerate(self.raw_instructions):
            inputs: list[_AbstractValue] = []
            outputs: list[_AbstractValue] = []
            new_intermediates: list[IntermediateValue] = []
            pop, push, (extra_branch, extra) = stack_effects_comprehensive(instruction)
            assert (idx + 1) == len(self.raw_instructions) or not extra, "Branch instruction mid-block"

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
                self.stack_conditional = (extra_branch, tuple(lookup(index) for index in extra))

                # Nodes on node flow:
                #   1) We have already functionalized variable accesses, so we can prune loads and stores.
                #   2) Inputs are consumed from the stack so we reverse them to get argument order.
                #   3) Conditional outputs are a block level concept. They are always added to node outputs.
                extra_outputs = tuple(lookup(index) for index in extra)
                self.stack_conditional = (extra_branch, extra_outputs)
                if not (store_scope or del_scope):
                    node_flow.append((instruction, tuple(reversed(inputs)), tuple((*outputs, *extra_outputs))))

            stack.extend(outputs)

        self.stacks = (block_inputs, stack)
        self.variables = {k: tuple(v) for k, v in variables.items()}
        values_used = set(itertools.chain(*(inputs for _, inputs, _ in node_flow)))
        self.uses = OrderedSet(k for k, v in self.state(0) if v in values_used)
        self.node_flow = tuple(node_flow)

    def __repr__(self) -> str:
        return f"ProtoBlock: {hex(id(self))}"

    def __hash__(self) -> int:
        return id(self)

    @staticmethod
    def topology(blocks: Iterable["ProtoBlock"]):
        parents = collections.defaultdict(list)
        for block in blocks:
            parents.setdefault(block, [])
            for child, is_jump in block.jump_targets:
                parents[child].append((block, is_jump))
        (root,) = (block for block, block_parents in parents.items() if not block_parents)
        assert not root.stacks[0], "Root block should not have stack inputs"
        return root, {block: tuple(block_parents) for block, block_parents in parents.items()}

    @staticmethod
    def stack_args(args: Iterable[ArgScope]) -> tuple[int, ...]:
        stack_args = tuple(sorted([i.arg for i in args if i.scope == VariableScope.STACK]))
        assert stack_args == tuple(range(-len(stack_args), 0)), stack_args
        return stack_args

    def stack_effect(self, is_jump: bool) -> tuple[int, int]:
        return len(tuple(self.stack(0, None))), len(tuple(self.stack(-1, is_jump)))

    def stack(self, index: EdgeIndex, is_jump: Optional[bool]) -> Iterable[_AbstractValue]:
        yield from self.stacks[index]
        extra_branch, extra_outputs = self.stack_conditional
        if index == -1 and is_jump in (extra_branch, None):
            yield from extra_outputs

    def state(self, index: EdgeIndex, is_jump: Optional[bool] = None) -> Iterator[tuple[ArgScope, _AbstractValue]]:
        for k, values in sorted(self.variables.items(), key=lambda x: x[0]):
            v = values[index]
            assert k.scope not in (VariableScope.CONST, VariableScope.STACK) and isinstance(v, _AbstractValue), (k, v)
            yield k, v

        stack = tuple(self.stack(index, is_jump))
        for idx, v in enumerate(stack):
            assert isinstance(v, _AbstractValue), v
            yield ArgScope(idx - len(stack), VariableScope.STACK), v

    def replace_values(self, replace_map: dict[_AbstractValue, _AbstractValue]) -> None:
        def make_replacement(values):
            cls = values.__class__
            assert cls in (list, tuple, collections.deque)
            assert all(isinstance(i, _AbstractValue) for i in values)
            return cls(replace_map.get(i, i) for i in values)

        self.stacks = tuple(make_replacement(stack) for stack in self.stacks)
        extra_branch, extra_outputs = self.stack_conditional
        self.stack_conditional = (extra_branch, make_replacement(extra_outputs))
        self.variables = {k: make_replacement(v) for k, v in self.variables.items()}
        self.node_flow = [
            (instruction, make_replacement(inputs), make_replacement(outputs))
            for instruction, inputs, outputs in self.node_flow
        ]


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

    arg: int
    scope: VariableScope


@dataclasses.dataclass(frozen=True, eq=False)
class AbstractPhiValue(AbstractValue):
    """A value which aliases one of several inputs."""

    constituents: tuple[AbstractValue]

    def __post_init__(self) -> None:
        assert all(isinstance(i, AbstractValue) for i in self.constituents), self.constituents


def _add_transitive(protoblocks: tuple[ProtoBlock, ...]) -> None:
    """Extend abstract value flows to include those needed by downstream blocks.

    This pass effectively functionalizes the abstract value flow. Note that
    we assume that variables are only modified by `STORE_...` and `DELETE_...`
    instructions. This is not a sound assumption since opaque calls
    (`CALL_FUNCTION`, `CALL_METHOD`, etc.) could mutate global and nonlocal
    variables. This does not, however, pose a soundness problem because we
    can check for state mutations during inlining and rerun flow analysis.

    The process is more involved than simply checking for mismatches because
    adding a transitive value to a block may necessitate adding a transitive
    value to the prior block and so on.
    """
    _, parents = ProtoBlock.topology(protoblocks)
    keys_used = {block: OrderedSet(block.uses) for block in protoblocks}
    blocks_to_process = collections.deque(protoblocks)
    while blocks_to_process:
        block = blocks_to_process.popleft()
        initial_state = dict(block.state(0))
        num_used = len(keys_used[block])
        for jump_target, is_jump in block.jump_targets:
            produced = dict(block.state(-1, is_jump))

            pop, push = block.stack_effect(is_jump)
            keys_used[block].update(
                # We need to correct the stack indices, because `produced` is
                # indexed based on the output stack but `uses` is indexed using
                # the input stack.
                ArgScope(k.arg - ((pop - push) if k.scope == VariableScope.STACK else 0), k.scope)
                for k in keys_used[jump_target]
                if k not in produced
            )

            edge_values = {produced[k] for k in keys_used[jump_target] if k in produced}
            keys_used[block].update(k for k, v in initial_state.items() if v in edge_values)

        if len(keys_used[block]) > num_used:
            blocks_to_process.extend(parent for parent, _ in parents[block])

    for block in protoblocks:
        block.uses.update(keys_used[block])
        for key in sorted(keys_used[block], reverse=True):
            if key.scope == VariableScope.STACK:
                while -key.arg > len(block.stacks[0]):
                    block.stacks[0].appendleft(ref := _AbstractRef(f"Transitive: {key}"))
                    block.stacks[1].appendleft(ref)
            else:
                block.variables.setdefault(key, (_AbstractRef(f"Transitive: {key}"),))


def _condense_values(protoblocks: tuple[ProtoBlock, ...]) -> None:
    """Bind references to more tangible values.

    We make liberal use of `_AbstractRef`s when constructing value flow because
    it allows us to process locally and defer global analysis. However we need
    to resolve these references before we can bind our proto-Graph to a fully
    fledged Graph.
    """
    # We only need the block connectivity to pair inputs and outputs. Once that
    # is done we can operate entirely on the value graph. (And headaches like
    # `is_jump` or variable scope simply disappear.)
    G = nx.DiGraph()
    for protoblock in protoblocks:
        for child, is_jump in protoblock.jump_targets:
            outputs = dict(protoblock.state(index=-1, is_jump=is_jump))
            child_inputs = dict(child.state(index=0))
            assert (
                # `_add_transitive` should ensure the stacks match.
                (s_out := ProtoBlock.stack_args(outputs)) == (s_in := ProtoBlock.stack_args(child_inputs))
                or
                # except for return blocks where we're going to discard the stack.
                child.raw_instructions[-1].opcode in return_instructions
            ), f"{s_out=} != {s_in=}, {child.raw_instructions[-1].opname}"
            for key, child_input in child_inputs.items():
                G.add_edge(outputs.get(key, ValueMissing()), child_input)

    # Our goal is to remove `_AbstractValue`s and bind them to `AbstractValue`s.
    # In most cases that means an earlier value in the graph; however for the
    # root node those references are truly external. Fortunately it's simple to
    # replace them: we can simply define an equality relation with a new
    # `ExternalRef` which will automatically take precidence over the
    # `_AbstractRef` during the chain folding pass.
    root, _ = ProtoBlock.topology(protoblocks)
    for (arg, scope), (initial_ref, *_) in root.variables.items():
        assert isinstance(initial_ref, _AbstractRef) and scope not in (VariableScope.CONST, VariableScope.STACK)
        G.add_edge(ExternalRef(arg, scope), initial_ref)

    # While not strictly necessary, it's much easier to debug the intermediate
    # graph logic if we first convert all values to indices.
    values = tuple(G.nodes)
    value_to_idx = {value: idx for idx, value in enumerate(values)}
    G = nx.from_edgelist(((value_to_idx[source], value_to_idx[sink]) for source, sink in G.edges), nx.DiGraph)
    index_alias_map: dict[int, set[int]] = {}

    # We can decompose the graph into disjoint use-def chains and analyze them independently.
    for nodes in nx.connected_components(G.to_undirected()):
        subgraph = G.subgraph(nodes)
        equality_edges: set[tuple[int, int]] = {(node, node) for node in subgraph.nodes}

        while True:
            # Condense pairs in `equality_edges`. For example, given the
            # following graph and `equality_edges`:
            #   0 → 1 → 2 → 3 → 4 → 5
            #               ↑┄──┘
            #
            #   equality_edges = {(0, 1), (3, 4)}
            #
            # After grouping we're left with:
            #   {0, 1} → 2 → {3, 4} → 5
            clusters: dict[int, int] = {}
            for cluster in nx.connected_components(nx.from_edgelist(equality_edges, nx.Graph)):
                # The choice of "canonical" index is arbitrary as long as it is consistent.
                clusters.update({i: min(cluster) for i in cluster})
            assert len(clusters) == len(subgraph)
            reduced_subgraph = nx.from_edgelist(((clusters.get(i, i), clusters.get(j, j)) for i, j in subgraph.edges), nx.DiGraph)
            reduced_subgraph.remove_edges_from(nx.selfloop_edges(reduced_subgraph))
            num_equality_edges = len(equality_edges)

            # Condense chains.
            equality_edges.update(
                (source, sink) for source, sink in reduced_subgraph.edges if reduced_subgraph.in_degree(sink)
            )

            # Condense loops.
            for cycle in nx.simple_cycles(reduced_subgraph):
                equality_edges.update(zip(cycle, itertools.chain(cycle[1:], cycle[:1])))

            if len(equality_edges) == num_equality_edges:
                # No progress has been made, exit loop.
                break

        # Once we isolate the irreducible values we can flood them through the
        # graph to resolve the `_AbstractRef`s.
        for cluster in nx.connected_components(nx.from_edgelist(equality_edges, nx.Graph)):
            cluster_subgraph = subgraph.subgraph(cluster)
            cluster_roots = {idx for idx in cluster if not isinstance(values[idx], _AbstractRef)}
            assert cluster_roots, [values[idx] for idx in cluster]
            for idx in cluster_roots:
                for reachable in itertools.chain([idx], *nx.dfs_successors(cluster_subgraph, idx).values()):
                    index_alias_map.setdefault(reachable, set()).add(idx)

            # By definition anything that isn't an `_AbstractRef` should not alias another value
            assert all(len(index_alias_map[idx]) == 1 for idx in cluster_roots), [
                [values[j] for j in index_alias_map[idx]] for idx in cluster_roots
            ]

    # And finally update the block value flows to reflect the changes.
    replace_map: dict[_AbstractValue, _AbstractValue] = {}
    for idx, source_indices in index_alias_map.items():
        if source_indices == {idx}:
            assert not isinstance(v := values[idx], _AbstractRef), f"Unhandled reference: {idx} {v}"
        else:
            new_values = tuple(values[idy] for idy in sorted(source_indices))
            constants: tuple[ExternalRef] = tuple(
                v for v in new_values if isinstance(v, ExternalRef) and v.scope == VariableScope.CONST
            )
            if len(constants) == len(new_values) and len({i.arg for i in constants}) == 1:
                replace_map[values[idx]], *_ = constants

            elif len(new_values) > 1:
                replace_map[values[idx]] = AbstractPhiValue(new_values)

            else:
                (replace_map[values[idx]],) = new_values

    for protoblock in protoblocks:
        protoblock.replace_values(replace_map)


def _bind_to_graph(
    protoblocks: tuple[ProtoBlock, ...],
    func: Callable,
    method_self: Optional[object] = None,
    mro_klass: Optional[type] = None,
) -> Graph:
    """Convert abstract value graph into a concrete Graph.

    The key nuance of this conversion is that the mapping from `AbstractValue`
    to `Value` is contextual. The first time we "see" an `AbstractValue` it
    maps to a `Value`. If we encounter it in any other block it maps to a
    PhiValue and we need to set the proper connectivity.

    This is perhaps clearer with an example. Suppose you have an argument `x`
    which is used by the root block and passed to the next block, and suppose
    you have another value `y` which is created in the root block and passed to
    the next block. In the abstract flow this is represented as:
           ________        ___________
    `x` -> | Root | -`x`-> | Block 1 | -> ...
           |  `y` | -`y`-> |         |
           --------        -----------

    On the other hand, `Graph` represents the same connectivity as:
                       ________                        ___________
    `x` ←┈┈→ `𝜙x_0` -> | Root | -`𝜙x_0` ←┈┈→ `𝜙x_1` -> | Block 1 | -> ...
                       |  `y` | -`y` ←┈┈┈┈┈→ `𝜙y_0` -> |         |
                       --------                        -----------

    (This diagram does not show the reason for PhiValues: to accept multiple inputs.)
    """
    # Peek at the signature and live objects to create Values. This is the
    # *only* region where this is permitted.
    # =========================================================================
    # TODO(robieta): Lazily generate specializations during runtime.
    signature = inspect.signature(func)
    names = make_name_map(itertools.chain(*(i.raw_instructions for i in protoblocks)), func.__code__)
    for i, _ in enumerate(signature.parameters.keys()):
        names[ArgScope(i, VariableScope.LOCAL)] = func.__code__.co_varnames[i]
    func_constants = func.__code__.co_consts
    func_globals = dict_join(func.__builtins__, func.__globals__, {"super": Super()})

    @functools.cache  # This is for correctness, not performance.
    def get_initial_value(key: ArgScope) -> Value:
        if key.scope == VariableScope.CONST:
            return Value(value=func_constants[key.arg], is_const=True)

        name = names[key]
        if key.scope == VariableScope.LOCAL:
            if key.arg == 0 and method_self is not None:
                return Value(value=method_self, name=name, is_function_arg=True)
            elif (p := signature.parameters.get(name)) is not None:
                return Value(typ=p.annotation, name=name, is_function_arg=True)
            return Value(value=NULL, name=name)

        if key.scope == VariableScope.NONLOCAL:
            raise RuntimeError(f"nonlocal variables are not supported but (key, name) = ({key}, {name}) found")

        if key.scope == VariableScope.GLOBAL:
            return Value(name=name, value=func_globals[name], is_global=True)

        raise ValueError(f"Unhandled key: {key=}, name: {name=}")

    del func
    # End live inspection region.
    # =========================================================================

    root, parents = ProtoBlock.topology(protoblocks)
    blocks = {protoblock: Block() for protoblock in protoblocks}
    blocks[root].jump_sources.append(None)
    self_value = get_initial_value(ArgScope(0, VariableScope.LOCAL)) if method_self is not None else None

    # Block inputs require special handling since we may need to create `PhiValue`s.
    input_conversions = {}
    for protoblock, block in blocks.items():
        uses = OrderedSet(protoblock.uses)
        for arg_scope, abstract_value in protoblock.state(index=0):
            if protoblock is root:
                value = get_initial_value(arg_scope)
                is_arg = arg_scope.scope == VariableScope.LOCAL and value.value is not NULL
                assert isinstance(abstract_value, ExternalRef) or not is_arg
                input_conversions[(abstract_value, protoblock)] = PhiValue([value], [None], block) if is_arg else value

            else:
                value = PhiValue([], [], block) if arg_scope in uses else Value(value=NULL)
                input_conversions[(abstract_value, protoblock)] = value

    @functools.cache  # Again, for correctness
    def convert(value: AbstractValue, protoblock: ProtoBlock) -> Value:
        assert isinstance(value, AbstractValue), value
        if (out := input_conversions.get((value, protoblock), missing := object())) is not missing:
            return out

        if isinstance(value, ValueMissing):
            return Value(value=NULL)

        elif isinstance(value, (IntermediateValue, AbstractPhiValue)):
            # For now we discard any information and just treat them as opaque.
            # TODO(robieta): refine
            return Value()

        elif isinstance(value, ExternalRef) and value.scope == VariableScope.CONST:
            return get_initial_value(ArgScope(value.arg, value.scope))

        raise ValueError(f"Cannot convert abstract value: {value}, {protoblock} {protoblock is root=}")

    def make_nodes(node_flow: Iterable[NodeFlow]) -> Iterable[Node]:
        for instruction, inputs, outputs in node_flow:
            node = Node(
                i=instruction,
                inputs=[convert(v, protoblock) for v in inputs],
                outputs=[convert(v, protoblock) for v in outputs],
                line_no=instruction.line_no,
            )

            if node.i.opname in ("LOAD_ATTR", "LOAD_METHOD"):
                # Once we set `parent` (so PhiValue can traverse through it)
                # we can prune these just like all other load instructions.
                node.outputs[0].parent = node.inputs[0]
                node.outputs[0].name = node.i.argrepr
                continue

            elif node.i.opname == "CALL_FUNCTION":
                # Note: `super` handling is not currently generic. Corner cases
                #       such as `super(**{})` or `super_alias = super; super_alias()`
                #       will not be correctly handled.
                # TODO(robieta): handle `super` without load bearing names.
                if instruction.arg == 0 and isinstance(node.inputs[0].value, Super):
                    assert self_value is not None, "super() called in free context"
                    node.outputs[0].value = MROAwareObjectRef(self_value, start_klass=mro_klass)

            elif node.i.opname == "FOR_ITER":
                node.outputs[1].node = node
                node.outputs[1].name = ".for_item_iter"

            yield node

    # First pass: populate nodes and jump targets.
    for protoblock, block in blocks.items():
        block.nodes = list(make_nodes(protoblock.node_flow))

        for target, is_jump in protoblock.jump_targets:
            jump_target = blocks[target]
            last_node = block.nodes[-1]
            jump_target.jump_sources.append(last_node)
            last_node.jump_targets.append((protoblock.stack_effect(is_jump), jump_target))

    # Second pass: link blocks.
    for protoblock, block in blocks.items():
        block_values = {
            k: v
            for k, abstract_v in protoblock.state(index=0)
            if isinstance(v := convert(abstract_v, protoblock), PhiValue)
        }
        block.block_inputs = list(OrderedSet(block_values.values()))

        for parent, is_jump in parents[protoblock]:
            parent_state = dict(parent.state(index=-1, is_jump=is_jump))
            for arg_scope, sink in block_values.items():
                source = convert(parent_state.get(arg_scope, ValueMissing()), parent)
                if source.value is not NULL and source not in sink.values:
                    sink.add_missing_value(v=source, jump_source=blocks[parent].nodes[-1])

    # Third pass: specify block outputs.
    for protoblock, block in blocks.items():
        block.block_outputs = OrderedSet(
            v
            for _, abstract_v in protoblock.state(index=-1, is_jump=None)
            if (v := convert(abstract_v, protoblock)).phi_values
        )

    gr = Graph(list(blocks.values()))

    needed_params = {
        v for k in sorted(root.uses) if k.scope == VariableScope.LOCAL and (v := get_initial_value(k)).value is not NULL
    }

    gr.local_variables_at_start = [
        get_initial_value(ArgScope(i, VariableScope.LOCAL)) for i in range(len(signature.parameters))
    ]
    missing_params = needed_params.difference(gr.local_variables_at_start)
    assert not missing_params, f"missing params {missing_params}"

    # bound_args = [module.forward.__self__]
    gr.self_value = self_value
    gr.ismethod = self_value is not None
    # deal with other flags?
    # NESTED, GENERATOR, NOFREE, COROUTINE, ITERABLE_COROUTINE, ASYNC_GENERATOR
    gr.co_flags = inspect.CO_OPTIMIZED | inspect.CO_NEWLOCALS
    gr.co_argcount = 0
    gr.co_posonlyargcount = 0
    gr.co_kwonlyargcount = 0
    gr.func_defaults = []
    gr.func_kwdefaults = {}
    for p in signature.parameters.values():
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            gr.co_argcount += 1
            gr.co_posonlyargcount += 1
        elif p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            gr.co_argcount += 1
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            gr.co_kwonlyargcount += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            gr.co_flags |= inspect.CO_VARARGS
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            gr.co_flags |= inspect.CO_VARKEYWORDS
        else:
            assert False, f"unknown parameter kind {p.kind}"

        if p.default is not inspect._empty:
            if p.kind == inspect.Parameter.KEYWORD_ONLY:
                gr.func_kwdefaults[p.name] = p.default
            else:
                gr.func_defaults.append(p.default)
    return gr


def acquire_partial(
    pfunc: functools.partial,
    module: Optional[object] = None,
    mro_klass: Optional[type] = None,
) -> Graph:
    # This is complicated due to the semantics of calling Python functions.
    # The partial wrapper does the following:
    # def pfunc.__call__(*args, **kwargs):
    #    kw = pfunc.keywords.copy()
    #    kw.update(kwargs)
    #    return pfunc.func(*pfunc.args, *args, **kw)

    # This means:
    # - positional partial_args are applied from the front and once
    #   they are bound, they are removed from the signature,
    # - keyword only args get new defautls,
    # - binding a positional arg as a keyword arg effectively (i.e. in how
    #   it can be set in calls) makes that arg and all args to the right
    #   keyword only.
    # - things that cannot be bound to parameters may show up in varargs
    #   or kwargs parameters of the function.

    gr = acquire_method(pfunc.func, module, mro_klass)

    # first we shuffle positional args to kw only if they are in the kwargs of the partial
    pos_param_names = [v.name for v in gr.local_variables_at_start[: gr.co_argcount]]
    pos_param_names_to_idx = {n: i for i, n in enumerate(pos_param_names)}
    kw_pos_param_idx = [pos_param_names_to_idx[k] for k in pfunc.keywords if k in pos_param_names_to_idx]
    if kw_pos_param_idx:
        # convert positional default args to kw ones
        kw_pos_param_min = min(kw_pos_param_idx)
        if kw_pos_param_min < gr.co_posonlyargcount:
            raise TypeError(
                f"cannot bin positional-only argument {pos_param_names[kw_pos_param_min]} as keyword in partial"
            )

        num_to_kw = gr.co_argcount - kw_pos_param_min
        if gr.func_defaults:
            to_kw = gr.func_defaults[-num_to_kw:]
            del gr.func_defaults[-num_to_kw:]
            to_kw_names = pos_param_names[-num_to_kw:]
            gr.func_kwdefaults.update(zip(to_kw_names, to_kw))
        # convert positional args to kw only
        gr.co_kwonlyargcount += num_to_kw
        gr.co_argcount -= num_to_kw

    # deal with positional args. some will be mapped to concrete positional args, some might be added to varargs (*args)
    if gr.ismethod:
        arg_start = 1
        arg_count = gr.co_argcount - 1
    else:
        arg_start = 0
        arg_count = gr.co_argcount

    args_to_bind = pfunc.args[:arg_count]
    args_for_varargs = pfunc.args[arg_count:]

    # do we need to drop positional default args?
    posarg_default_start = gr.co_argcount - len(gr.func_defaults)
    posarg_default_to_delete = len(args_to_bind) + arg_start - posarg_default_start
    if posarg_default_to_delete > 0:
        gr.func_defaults = gr.func_defaults[posarg_default_to_delete:]

    bound_values = gr.local_variables_at_start[arg_start : arg_start + len(args_to_bind)]
    del gr.local_variables_at_start[arg_start : arg_start + len(args_to_bind)]

    for bound_value, arg in zip(bound_values, args_to_bind):
        bound_value.is_function_arg = False
        bound_value.is_const = True
        # TODO: check type?
        bound_value.value = arg
        gr.co_argcount -= 1
        if gr.co_posonlyargcount > 0:
            gr.co_posonlyargcount -= 1

    # handle keyword arguments to concrete parameters, collect in kwargs those for kw-varargs (**kwargs)
    param_names_to_idx = {
        v.name: i for i, v in enumerate(gr.local_variables_at_start[: gr.co_argcount + gr.co_kwonlyargcount])
    }
    kwargs = {}
    for argname, argvalue in pfunc.keywords.items():
        idx = param_names_to_idx.get(argname, -1)
        if idx == -1:
            kwargs[argname] = argvalue
            continue
        gr.func_kwdefaults[argname] = argvalue

    # for varargs and kwargs fed from partial we need the following prelude:
    # TODO: (but maybe we should just have a prelude always for the consts, too...)
    # if it has *varargs:
    #    TMP1 = LOAD_CONST partial_args_for_varargs (needs to be a tuple)
    #    varargs = TMP1 + varargs
    # if it has **kwargs:
    #    TMP2 = LOAD_CONST partial_kwargs
    #    kwargs = partial_kwargs | kwargs

    if args_for_varargs or kwargs:
        prelude = Block()
        jump_node = Node(i=make_jump_absolute(None), inputs=[], outputs=[], line_no=0)
        prelude.nodes.append(jump_node)
        jump_target = gr.blocks[0]
        assert jump_target.jump_sources[0] is None
        jump_target.jump_sources[0] = jump_node
        jump_node.jump_targets.append((0, jump_target))
        prelude.jump_sources.append(None)
        for i in jump_target.block_inputs:
            assert i.jump_sources[0] is None
            i.jump_sources[0] = jump_node
    else:
        prelude = None

    # handle *args (varargs)
    if args_for_varargs:
        if kw_pos_param_idx:
            raise TypeError(
                f"partial tried to bind {len(pfunc.args)} positional arguments, but only {arg_count} are allowed after keyword binding"
            )
        if not (gr.co_flags & inspect.CO_VARARGS):
            raise TypeError(
                f"partial tried to bind {len(pfunc.args)} positional arguments, but only {arg_count} are allowed"
            )
        # the variable for varargs is at gr.co_argcount + gr.co_kwonlyargcount
        v_vararg_param = gr.local_variables_at_start[gr.co_argcount + gr.co_kwonlyargcount]
        v_partial_varargs = Value(name="partial_varargs", value=tuple(args_for_varargs), is_const=True)
        v_varargs_new = Value(name="varargs_with_partial")  # type is tuple
        pv = PhiValue([v_vararg_param], [None], block=prelude)
        new_n = Node(
            i=get_instruction(opname="BINARY_ADD", arg=None),
            inputs=[v_partial_varargs, pv],
            outputs=[v_varargs_new],
            line_no=0,
        )  # line number?
        prelude.nodes.insert(0, new_n)
        prelude.block_outputs.add(v_varargs_new)
        # replace v_vararg_param with v_varargs_new in remainder
        replace_values(gr, {v_vararg_param: v_varargs_new})
        prelude.block_inputs.append(pv)

    # handle **kwargs
    if kwargs:
        if not (gr.co_flags & inspect.CO_VARKEYWORDS):
            raise TypeError(
                f"function does not have **kwargs but partial tries to bind unknown keywords {tuple(kwargs)}."
            )

        # the variable for varargs is at gr.co_argcount + gr.co_kwonlyargcount
        v_kwvararg_param = gr.local_variables_at_start[
            gr.co_argcount + gr.co_kwonlyargcount + (1 if gr.co_flags & inspect.CO_VARARGS else 0)
        ]
        v_partial_kwvarargs = Value(name="partial_kwvarargs", value=kwargs, is_const=True)
        v_kwvarargs_new = Value(name="kwvarargs_with_partial")  # type is dict
        pv = PhiValue([v_kwvararg_param], [None], block=prelude)
        new_n = Node(
            i=get_instruction(opname="BINARY_OR", arg=None),
            inputs=[v_partial_kwvarargs, pv],
            outputs=[v_kwvarargs_new],
            line_no=0,
        )  # line number?
        prelude.nodes.insert(-1, new_n)
        prelude.block_outputs.add(v_kwvarargs_new)
        # replace v_vararg_param with v_varargs_new in remainder
        replace_values(gr, {v_kwvararg_param: v_kwvarargs_new})
        prelude.block_inputs.append(pv)

    if prelude:
        gr.blocks.insert(0, prelude)
    return gr


@functools.cache
def _construct_protoblocks(func):
    """Protoblocks are parse level constructs, so it is safe to reuse them."""
    protoblocks = parse_bytecode(func)
    _add_transitive(protoblocks)
    _condense_values(protoblocks)
    return protoblocks


def acquire_method(
    method: Callable,
    module: Optional[object] = None,
    mro_klass: Optional[type] = None,
) -> Graph:
    assert SUPPORTS_PREPROCESSING, sys.version_info
    if isinstance(method, functools.partial):
        return acquire_partial(method, module, mro_klass)
    if callable(method) and not inspect.ismethod(method) and not inspect.isfunction(method):
        method = method.__call__

    method_self, func = (method.__self__, method.__func__) if inspect.ismethod(method) else (None, method)
    assert not inspect.ismethod(func)

    module = module or method_self
    if mro_klass is None and module is not None:
        mro_klass = type(module)

    gr = _bind_to_graph(_construct_protoblocks(func), func, method_self, mro_klass)
    gr.source_start_line = 1
    try:
        gr.source_lines, _ = inspect.getsourcelines(method)
    except OSError:
        gr.source_lines = ["# Failed to extract source."]

    gr.method = method
    gr.module = module
    gr.mro_klass = mro_klass
    if DEBUG_ASSERTS:
        check_graph(gr)
    return gr


def remove_unused_values(gr: Graph) -> None:
    gr.ensure_links()

    def remove_value(v: Value) -> None:
        for pv in v.phi_values:
            bl = pv.block
            pv.remove_value(v)
            if not pv.values:
                remove_value(pv)
                bl.block_inputs.remove(pv)
                if pv in bl.block_outputs:
                    bl.block_outputs.remove(pv)

    for i in gr.blocks[0].block_inputs:
        if len(i.values) == 1 and i.values[0] is None:
            remove_value(i)

    gr.blocks[0].block_inputs = [i for i in gr.blocks[0].block_inputs if len(i.values) != 1 or i.values[0] is not None]

    values_used = set()

    INDEX_OPS = {"BINARY_SUBSCR"}

    def mark_used(v: Value) -> None:
        if v in values_used:
            return
        values_used.add(v)
        if v.node and v.node.i.opname in INDEX_OPS:
            for i in v.node.inputs:
                mark_used(i)
        if v.parent is not None:
            mark_used(v.parent)
        if isinstance(v, PhiValue):
            for w in v.values:
                mark_used(w)

    for bl in gr.blocks:
        for n in bl.nodes:
            if n.i.opname not in INDEX_OPS:
                for i in n.inputs:
                    mark_used(i)

    for bl in gr.blocks:
        for i in bl.block_inputs[:]:
            if i not in values_used:
                for v in i.values[:]:
                    if v is not None:
                        i.remove_value(v)
                bl.block_inputs.remove(i)
        bl.block_outputs = OrderedSet(o for o in bl.block_outputs if o in values_used)
        for n in bl.nodes[:]:
            if n.i.opname in INDEX_OPS and not any((o in values_used) for o in n.outputs):
                bl.nodes.remove(n)
    for i in gr.local_variables_at_start:
        if i is not None:
            i.phi_values = [pv for pv in i.phi_values if pv in values_used]

    for bl in gr.blocks:
        for n in bl.nodes:
            for o in n.outputs:
                o.phi_values = [pv for pv in o.phi_values if pv in values_used]

    # remove things only used in current block (and not in own phi) from outputs
    # TODO: think if this would obsolete the above
    outputs_used = set()
    for bl in gr.blocks:
        for i in bl.block_inputs:
            assert isinstance(i, PhiValue)
            for v in i.values:
                outputs_used.add(v)
    for bl in gr.blocks:
        bl.block_outputs = OrderedSet(o for o in bl.block_outputs if o in outputs_used)

    if DEBUG_ASSERTS:
        check_graph(gr)
