import collections
import dataclasses
import functools
import itertools
from typing import Callable

import networkx as nx

from thunder.core.script.protograph import (
    _AbstractRef,
    _AbstractValue,
    AbstractPhiValue,
    ExternalRef,
    ProtoGraph,
    ProtoBlock,
    VariableKey,
    ValueMissing,
)
from thunder.core.script.python_ir_data import VariableScope, RAISE_RETURN_INSTRUCTIONS
from thunder.core.utils import debug_asserts_enabled, OrderedSet


def check_idempotent(f: Callable[[ProtoGraph], tuple[ProtoGraph, bool]]):
    @functools.wraps(f)
    def wrapped(protograph: ProtoGraph):
        protograph, had_effect = f(protograph)
        if debug_asserts_enabled():
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

        # `AbstractPhiValue._unpack_apply` will determine if it is needed after deduplication.
        candidate_constituents = tuple(values[idy] for idy in sorted(source_indices))
        replace_map[v] = AbstractPhiValue(candidate_constituents).substitute({})

    return proto_graph.transform(replace_map), any(v != {k} for k, v in index_alias_map.items())


def _tuple_fold(protograph: ProtoGraph) -> tuple[ProtoGraph, bool]:
    """Replace tuple accesses (`BINARY_SUBSCR`, `UNPACK_SEQUENCE` instructions) with their members, if known.

    Note: The pass, as it is currently written, only folds one layer of tuples, from one known source, the
    `BUILD_TUPLE` instruction. This should be enough for our needs. However, it's easy enough to imagine a world
    where we would want to call this in a loop with other operations that fold values. In other words, tuple folding
    would be more powerful the more `_AbstractValue`s we can prove to be tuples with a known source.
    """

    @dataclasses.dataclass(slots=True, unsafe_hash=True, eq=True)
    class TupleKey:
        tup_value: _AbstractValue
        idx: int

        def __lt__(self, other):
            return self.__hash__() < other.__hash__()

    # {(tup, idx): val}
    tuple_sources: collections.defaultdict[TupleKey] = collections.defaultdict()
    queries: collections.defaultdict[TupleKey] = collections.defaultdict()

    for pb in protograph:
        for instr, ivals, ovals in pb.node_flow:
            if instr.opname == "BUILD_TUPLE":
                assert ivals is not None  # The elements of the tuple
                assert len(ovals) == 1  # The output tuple
                tuple_sources.update({TupleKey(ovals[0], _i): i for _i, i in enumerate(ivals)})
            elif instr.opname == "BINARY_SUBSCR":
                assert len(ivals) == 2  # The tuple, and the index
                assert len(ovals) == 1  # The result of tup[idx]
                ref = ivals[1]
                if isinstance(ref, ExternalRef):
                    if ref.key.scope == VariableScope.CONST and isinstance(ref.key.identifier, int):
                        queries.update({TupleKey(ivals[0], ref.key.identifier): ovals[0]})
            elif instr.opname == "UNPACK_SEQUENCE":
                assert len(ivals) == 1  # The tuple to unpack
                assert len(ovals)  # The elements of the tuple, unpacked
                queries.update({TupleKey(ivals[0], _i): o for _i, o in enumerate(ovals)})
            elif instr.opname == "UNPACK_EX":
                assert len(ivals) == 1  # The tuple to unpack
                assert len(ovals) >= 2  # [remaining, unpack...]
                # Note: The remaining elements are a list, not a tuple.
                # That makes this pass technically unsound, as even if an element is updated,
                # this pass will fold the old value. It's unclear if there's any reason in the language
                # why unpacking this way returns a list, but it's unlikely to come up in practice,
                # and editing this list can be made an error in the future.
                # Another note: It should be possible to fold the remaining elements as well,
                # but as it is actually a list, let's not track that for now.
                queries.update({TupleKey(ivals[0], _i): o for _i, o in enumerate(ovals[1:])})

    if not queries or not tuple_sources:
        return (protograph, False)

    transform_replacements: dict[_AbstractValue, _AbstractValue] = {}
    for query_key, query_value in queries.items():
        if source_value := tuple_sources.get(query_key):
            transform_replacements[query_value] = source_value

    return (protograph.transform(transform_replacements), True)


def apply_protograph_passes(protograph: "ProtoGraph") -> "ProtoGraph":
    protograph, _ = _add_transitive(protograph)
    protograph, _ = _condense_values(protograph)
    protograph, _ = _tuple_fold(protograph)
    return protograph
