import collections
import functools
import dis
import inspect
import itertools
import sys
from typing import Callable, Optional
from collections.abc import Iterable

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
from thunder.core.script.protograph import (
    _AbstractRef,
    _AbstractValue,
    AbstractPhiValue,
    AbstractValue,
    ExternalRef,
    IntermediateValue,
    NodeFlow,
    ProtoBlock,
    ProtoGraph,
    ValueMissing,
    VariableKey,
)
from thunder.core.script.python_ir_data import (
    compute_jump,
    get_epilogue,
    get_instruction,
    make_jump_absolute,
    make_return,
    JumpEpilogue,
    NoJumpEpilogue,
    VariableScope,
    JUMP_INSTRUCTIONS,
    RAISE_RETURN_INSTRUCTIONS,
    RETURN_INSTRUCTIONS,
    RETURN_VALUE,
    SUPPORTS_PREPROCESSING,
    UNCONDITIONAL_JUMP_INSTRUCTIONS,
)
from thunder.core.utils import OrderedSet


DEBUG_ASSERTS = False


def enable_debug_asserts():
    global DEBUG_ASSERTS
    DEBUG_ASSERTS = True


class Super:
    pass


def parse_bytecode(method: Callable) -> ProtoGraph:
    """Given a method, disassemble it to a sequence of simple blocks."""
    bytecode: tuple[dis.Instruction, ...] = tuple(dis.get_instructions(method, first_line=0))
    make_protoblock = functools.partial(ProtoBlock.from_instructions, code=method.__code__)

    # Determine the boundaries for the simple blocks.
    split_after = JUMP_INSTRUCTIONS | RETURN_INSTRUCTIONS
    follows_jump = itertools.chain([0], (int(i in split_after) for i in bytecode))
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
        if block[-1] not in split_after:
            next_start = start + block_size
            assert bytecode[next_start].is_jump_target
            block.append(make_jump_absolute(next_start))

    # Consolidate `return` statements.
    def is_return(block: list[dis.Instruction]) -> bool:
        assert block and not any(i.opname == RETURN_VALUE for i in block[:-1])
        return block[-1].opname == RETURN_VALUE

    return_starts = tuple(start for start, (_, block) in blocks.items() if is_return(block))
    assert return_starts, "No return instruction found"
    if len(return_starts) > 1:
        new_return_start = len(bytecode) + 1
        for _, block in (blocks[start] for start in return_starts):
            assert is_return([prior_return := block.pop()]), prior_return
            block.append(make_jump_absolute(new_return_start))

        blocks[new_return_start] = 1, [make_return(is_jump_target=True)]
        assert is_return(blocks[new_return_start][1])

    # Assign line numbers once structure is (mostly) finalized.
    line_no = 1
    for instruction in itertools.chain(*[block for _, block in blocks.values()]):
        instruction.line_no = line_no = instruction.starts_line or line_no

    # Move return block to the end. This isn't always valid (since a block might
    # expect to fall through and reach it), but that will be resolved by the
    # sort in `ProtoGraph`'s ctor.
    blocks[return_starts[0]] = blocks.pop(return_starts[0])

    # Create and link protoblocks.
    protoblocks = {idx: make_protoblock(instructions) for idx, (_, instructions) in blocks.items()}

    def handle_jumps(jump, last_instruction, source, sink):
        if epilogue := get_epilogue(last_instruction, jump=jump):
            yield (epilogue_block := make_protoblock([epilogue]))
            source.add_jump_target(epilogue_block, jump)
            epilogue_block.add_jump_target(sink, jump)

        else:
            source.add_jump_target(sink, jump)

    def iter_protoblocks():
        for start, (raw_block_len, (*_, last_i)) in blocks.items():
            yield (source := protoblocks[start])

            if last_i in JUMP_INSTRUCTIONS:
                end = start + raw_block_len - 1
                if last_i not in UNCONDITIONAL_JUMP_INSTRUCTIONS:
                    yield from handle_jumps(False, last_i, source, protoblocks[end + 1])

                if (jump_offset := compute_jump(last_i, end)) is not None:
                    yield from handle_jumps(True, last_i, source, protoblocks[jump_offset])

    return ProtoGraph(iter_protoblocks())


def check_idempotent(f: Callable[[ProtoGraph], tuple[ProtoGraph, bool]]):

    @functools.wraps(f)
    def wrapped(protograph: ProtoGraph):
        protograph, had_effect = f(protograph)
        if DEBUG_ASSERTS:
            _, had_effect_on_rerun = f(protograph)
            assert not had_effect_on_rerun

        return protograph, had_effect
    return wrapped



def _get_missing_transitive(protograph: ProtoGraph) -> dict[ProtoBlock, OrderedSet[VariableKey]]:
    """Identify new transitive value dependencies.

    The process is more involved than simply checking for mismatches because
    adding a transitive value to a block may necessitate adding a transitive
    value to the prior block and so on.
    """
    uses = {protoblock: protoblock.uses.copy() for protoblock in protograph}
    blocks_to_process = collections.deque(protograph)
    while blocks_to_process:
        protoblock = blocks_to_process.popleft()
        initial_use_count = len(uses[protoblock])
        final_state = dict(protoblock.variable_state(index=-1))

        # We need to correct the stack indices, because `final_state` is indexed
        # to the output stack but `uses` is indexed to the input stack.
        pop, values_pushed = protoblock.stack_effect
        net_stack_effect = pop - len(values_pushed)

        # Any key that is is `child.uses` but not in `final_state` must be
        # a purely transitive dependency.
        child_uses = (uses[target] for target, _ in protoblock.jump_targets)
        child_keys_used = OrderedSet(itertools.chain(*child_uses))
        for key in child_keys_used - final_state:
            if key.scope == VariableScope.STACK:
                assert isinstance(key.identifier, int)
                key = VariableKey(key.identifier - net_stack_effect, key.scope)
            uses[protoblock].add(key)

        # Otherwise we must determine what the appropriate input key is.
        # (Recall that `uses` is indexed to `.variable_state(index=0)`)
        edge_values = {v for k, v in final_state.items() if k in child_keys_used}
        uses[protoblock].update(k for k, v in protoblock.variable_state(index=0) if v in edge_values)

        # If new uses are added then add parents to the queue.
        if len(uses[protoblock]) > initial_use_count:
            blocks_to_process.extend(protograph.parents[protoblock])

    new_uses = {i: uses_i.difference(i.uses) for i, uses_i in uses.items()}
    return {i: new_uses_i for i, new_uses_i in new_uses.items() if new_uses_i}


@check_idempotent
def _add_transitive(protograph: ProtoGraph) -> tuple[ProtoGraph, bool]:
    """Extend abstract value flows to include those needed by downstream blocks.

    This pass effectively functionalizes the abstract value flow by plumbing
    reads through parents as transitive dependencies. Note that we assume
    variables are only modified by `STORE_...` and `DELETE_...` instructions.
    This is not a sound assumption since opaque calls (`CALL_FUNCTION`,
    `CALL_METHOD`, etc.) could mutate global and nonlocal variables. This does
    not, however, pose an overall soundness problem because we can check for
    state mutations during inlining and rerun flow analysis.
    """
    protograph = protograph.transform({})
    missing_transitive = _get_missing_transitive(protograph)
    for protoblock, new_uses in missing_transitive.items():
        protoblock.uses.update(new_uses)
        for key in new_uses:
            if key.scope == VariableScope.STACK:
                assert isinstance(key.identifier, int) and key.identifier < 0
                while -key.identifier > len(protoblock.begin_stack):
                    protoblock.begin_stack.appendleft(_AbstractRef("Transitive"))
            else:
                protoblock.variables.setdefault(key, (_AbstractRef(f"Transitive"),))
        initial_keys = OrderedSet(k for k, _ in protoblock.variable_state(index=0))

        # Ensure the new transitive dependencies are reflected in the variable state.
        assert not (delta := new_uses.difference(initial_keys)), delta

    return protograph, bool(missing_transitive)


@check_idempotent
def _condense_values(proto_graph: ProtoGraph) -> tuple[ProtoGraph, bool]:
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
    for protoblock in proto_graph:
        for child, _ in protoblock.jump_targets:
            outputs = dict(protoblock.variable_state(index=-1))
            child_inputs = dict(child.variable_state(index=0))
            for key, child_input in child_inputs.items():
                G.add_edge(outputs.get(key, ValueMissing()), child_input)

            # `_add_transitive` should ensure the stacks match.
            # (Except for return blocks which may discard the stack.)
            if child.raw_instructions[-1] not in RAISE_RETURN_INSTRUCTIONS:
                s_out, s_in = [ProtoBlock.stack_args(i) for i in (outputs, child_inputs)]
                assert s_out == s_in, f"{s_out=} != {s_in=}, {child.raw_instructions[-1].opname}"

    # Our goal is to remove `_AbstractValue`s and bind them to `AbstractValue`s.
    # In most cases that means an earlier value in the graph; however for the
    # root node those references are truly external. Fortunately it's simple to
    # replace them: we can simply define an equality relation with a new
    # `ExternalRef` which will automatically take precidence over the
    # `_AbstractRef` during the chain folding pass.
    for key, (initial_ref, *_) in proto_graph.root.variables.items():
        if isinstance(initial_ref, ExternalRef):
            continue

        assert isinstance(initial_ref, _AbstractRef), initial_ref
        assert key.scope not in (VariableScope.CONST, VariableScope.STACK)
        G.add_edge(ExternalRef(key), initial_ref)
    G.remove_edges_from(nx.selfloop_edges(G))

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
            reduced_subgraph = nx.from_edgelist(((clusters[i], clusters[j]) for i, j in subgraph.edges), nx.DiGraph)
            reduced_subgraph.remove_edges_from(nx.selfloop_edges(reduced_subgraph))
            num_equality_edges = len(equality_edges)

            # Condense chains.
            equality_edges.update(reduced_subgraph.edges)

            # Condense loops.
            for cycle in nx.simple_cycles(reduced_subgraph):
                equality_edges.update(zip(cycle, itertools.chain(cycle[1:], cycle[:1])))

            if len(equality_edges) == num_equality_edges:
                # No progress has been made, exit loop.
                break

        roots = {idx for idx in cluster if not subgraph.in_degree(idx)}
        assert roots, [values[idx] for idx in nodes]
        for idx in roots:
            successors = nx.dfs_successors(subgraph, idx)
            for reachable in itertools.chain([idx], *successors.values()):
                index_alias_map.setdefault(reachable, set()).add(idx)

        # By definition anything that isn't an `_AbstractRef` should not alias another value.
        for idx in nodes:
            if isinstance(values[idx], AbstractPhiValue):
                known_constituents = {value_to_idx[v] for v in values[idx].constituents}
                index_alias_map[idx] = {i for i in index_alias_map[idx] if i not in known_constituents} | {idx}

            else:
                is_ref = isinstance(values[idx], _AbstractRef)
                invariants = (idx in roots, index_alias_map[idx] == {idx}, not is_ref)
                assert all(invariants) or not any(invariants), (idx, values[idx], invariants)

    # And finally update the block value flows to reflect the changes.
    replace_map: dict[_AbstractValue, _AbstractValue] = {}
    for idx, source_indices in index_alias_map.items():
        assert source_indices
        v = values[idx]
        if source_indices == {idx}:
            assert not isinstance(v, _AbstractRef), f"Unhandled reference: {idx} {v}"
            continue

        new_values = tuple(OrderedSet(values[idy] for idy in sorted(source_indices)))
        replace_map[v] = AbstractPhiValue(new_values) if len(new_values) > 1 else new_values[0]

    return proto_graph.transform(replace_map), any(v != {k} for k, v in index_alias_map.items())


def _is_epilogue(protoblock: ProtoBlock) -> bool:
    return any(isinstance(i, (NoJumpEpilogue, JumpEpilogue)) for i, _, _ in protoblock.node_flow)


def _bind_to_graph(
    proto_graph: ProtoGraph,
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
    func_globals = {**func.__builtins__, **func.__globals__, **{"super": Super()}}

    # NOTE:
    #   `inspect.signature` will expose parameters in intuitive order. However that
    #   is not necessarily how Python represents them internally. Specifically, varargs
    #   and varkwargs are moved to the end. This convention is load bearing (since it
    #   allows the interpreter index into a flat args array) so we must respect it
    #   here. (`func.__code__.co_varnames` is the canonical ordering.)
    arg_ordered_parameters = func.__code__.co_varnames[: len(signature.parameters)]
    if set(arg_ordered_parameters) != set(signature.parameters):
        assert hasattr(func, "__wrapped__")
        msg = f"({', '.join(arg_ordered_parameters)}) != ({', '.join(signature.parameters.keys())})"
        raise NotImplementedError(msg)

    self_key: Optional[VariableKey] = None
    self_value: Optional[Value] = None
    if method_self is not None:
        self_key = VariableKey(arg_ordered_parameters[0], VariableScope.LOCAL)
        self_value = Value(value=method_self, name=self_key.identifier, is_function_arg=True)

    @functools.cache  # This is for correctness, not performance.
    def get_initial_value(key: VariableKey) -> Value:
        if key.scope == VariableScope.CONST:
            return Value(value=key.identifier, is_const=True)

        elif key == self_key:
            return self_value

        name = key.identifier
        assert isinstance(name, str)
        if key.scope == VariableScope.LOCAL:
            if (p := signature.parameters.get(name)) is not None:
                return Value(typ=p.annotation, name=name, is_function_arg=True)
            return Value(value=NULL, name=name)

        if key.scope == VariableScope.NONLOCAL:
            msg = f"nonlocal variables are not supported but (key, name) = ({key}, {name}) found"
            raise RuntimeError(msg)

        if key.scope == VariableScope.GLOBAL:
            return Value(name=name, value=func_globals[name], is_global=True)

        raise ValueError(f"Unhandled key: {key=}, name: {name=}")

    del func
    # End live inspection region.
    # =========================================================================

    assert not _is_epilogue(proto_graph.root)
    assert not (missing_transitive := _get_missing_transitive(proto_graph)), missing_transitive

    def prologue(protoblock: ProtoBlock) -> ProtoBlock:
        if _is_epilogue(protoblock):
            (protoblock,) = proto_graph.parents[protoblock]
            assert not _is_epilogue(protoblock)
        return protoblock

    def fold_epilogue_stack(protoblock: ProtoBlock, epilogue: ProtoBlock):
        """Combine the stack effect of an instruction and its epilogue."""
        pop, push = protoblock.stack_effect
        pop_epilogue, push_epilogue = epilogue.stack_effect

        assert _is_epilogue(epilogue)
        assert epilogue in {jump_target for jump_target, _ in protoblock.jump_targets}
        assert len(epilogue.node_flow) == 1
        assert push_epilogue == epilogue.node_flow[0][2]
        return pop, (*push[:-pop_epilogue], *push_epilogue)

    blocks = {protoblock: Block() for protoblock in proto_graph if not _is_epilogue(protoblock)}
    blocks[proto_graph.root].jump_sources.append(None)

    # Block inputs require special handling since we may need to create `PhiValue`s.
    input_conversions = {}
    for protoblock, block in blocks.items():
        for key, abstract_value in protoblock.variable_state(index=0):
            if protoblock is proto_graph.root:
                value = get_initial_value(key)
                if key.scope == VariableScope.LOCAL and value.value is not NULL:
                    assert isinstance(abstract_value, ExternalRef), abstract_value
                    value = PhiValue([value], [None], block)

            elif key in protoblock.uses:
                value = PhiValue([], [], block)

            else:
                value = Value(value=NULL)

            input_conversions[(abstract_value, protoblock)] = value

    @functools.cache  # Again, for correctness
    def convert(value: AbstractValue, protoblock: ProtoBlock) -> Value:
        assert not _is_epilogue(protoblock)
        assert isinstance(value, AbstractValue), value
        if (out := input_conversions.get((value, protoblock), missing := object())) is not missing:
            return out

        if isinstance(value, ValueMissing):
            return Value(value=NULL)

        elif isinstance(value, (IntermediateValue, AbstractPhiValue)):
            # For now we discard any information and just treat them as opaque.
            # TODO(robieta): refine
            return Value()

        elif isinstance(value, ExternalRef) and value.key.scope == VariableScope.CONST:
            return get_initial_value(value.key)

        raise ValueError(f"Cannot convert abstract value: {value}, {protoblock} {protoblock is proto_graph.root=}")

    def iter_node_flow(protoblock: ProtoBlock) -> Iterable[Node]:
        yield from protoblock.node_flow[:-1]
        instruction, inputs, outputs = protoblock.node_flow[-1]
        if any(_is_epilogue(jump_target) for jump_target, _ in protoblock.jump_targets):
            assert all(_is_epilogue(target) for target, _ in protoblock.jump_targets)
            outputs_by_branch = tuple(
                fold_epilogue_stack(protoblock, jump_target)[1] for jump_target, _ in protoblock.jump_targets
            )

            merged_outputs = []
            for values in itertools.zip_longest(*outputs_by_branch, fillvalue=None):
                value_set = set(values).difference({None})
                assert len(value_set) == 1, value_set
                merged_outputs.append(value_set.pop())
            outputs = tuple(merged_outputs)

        yield instruction, inputs, outputs

    def make_nodes(protoblock: ProtoBlock) -> Iterable[Node]:
        for instruction, inputs, outputs in iter_node_flow(protoblock):
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
        block.nodes = list(make_nodes(protoblock))
        for target, _ in protoblock.jump_targets:
            pop, values_pushed = protoblock.stack_effect
            if _is_epilogue(target):
                pop, values_pushed = fold_epilogue_stack(protoblock, target)
                ((target, _),) = target.jump_targets

            jump_target = blocks[target]
            last_node = block.nodes[-1]
            jump_target.jump_sources.append(last_node)
            last_node.jump_targets.append(((pop, len(values_pushed)), jump_target))

    # Second pass: link blocks.
    for protoblock, block in blocks.items():
        block_values = {
            k: v
            for k, abstract_v in protoblock.variable_state(index=0)
            if isinstance(v := convert(abstract_v, protoblock), PhiValue)
        }

        block.block_inputs = list(OrderedSet(block_values.values()))
        for parent in proto_graph.parents[protoblock]:
            parent_key = prologue(parent)
            parent_state = dict(parent.variable_state(index=-1))
            for key, sink in block_values.items():
                source = convert(parent_state.get(key, ValueMissing()), parent_key)
                if source.value is not NULL and source not in sink.values:
                    sink.add_missing_value(v=source, jump_source=blocks[parent_key].nodes[-1])

    # Third pass: specify block outputs once we know which Values are passed to another Block.
    for protoblock, block in blocks.items():
        boundary_protoblocks = (protoblock,)
        if any(_is_epilogue(jump_target) for jump_target, _ in protoblock.jump_targets):
            boundary_protoblocks = tuple(target for target, _ in protoblock.jump_targets)
            assert all(_is_epilogue(target) for target in boundary_protoblocks)

        block.block_outputs = OrderedSet()
        for boundary_protoblock in boundary_protoblocks:
            for _, abstract_value in boundary_protoblock.variable_state(index=-1):
                # NOTE: the key for convert is `protoblock`, not `boundary_protoblock`
                if (v := convert(abstract_value, protoblock)).phi_values:
                    block.block_outputs.add(v)

    param_keys = tuple(VariableKey(p, VariableScope.LOCAL) for p in arg_ordered_parameters)
    missing = {
        k: v
        for k in proto_graph.root.uses.difference(param_keys)
        if k.scope == VariableScope.LOCAL and (v := get_initial_value(k)).value is not NULL
    }
    assert not missing, f"missing params {missing}"

    gr = Graph(list(blocks.values()))
    gr.local_variables_at_start = [get_initial_value(k) for k in param_keys]

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
def _construct_protograph(func):
    """Protoblocks are parse level constructs, so it is safe to reuse them."""
    proto_graph = parse_bytecode(func)
    proto_graph, _ = _add_transitive(proto_graph)
    proto_graph, _ = _condense_values(proto_graph)
    return proto_graph


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

    gr = _bind_to_graph(_construct_protograph(func), func, method_self, mro_klass)
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
