import collections
import dataclasses
import functools
import itertools
from typing import Callable, Concatenate
from collections.abc import Iterable

from typing_extensions import ParamSpec

from thunder.core.script.algorithms import compute_condense_map
from thunder.core.script.protograph import (
    AbstractRef,
    AbstractValue,
    AbstractPhiValue,
    ExternalRef,
    IntraBlockFlow,
    ProtoGraph,
    ProtoBlock,
    VariableKey,
    ValueMissing,
)
from thunder.core.script.python_ir_data import VariableScope, RAISE_RETURN_INSTRUCTIONS
from thunder.core.utils import debug_asserts_enabled, OrderedSet

ValueEdges = Iterable[tuple[AbstractValue, AbstractValue]]
P = ParamSpec("P")
IDEMPOTENT_REPEATS = 10  # Check for nondeterministic behavior.


def check_idempotent(
    f: Callable[Concatenate[ProtoGraph, P], tuple[ProtoGraph, bool]]
) -> Callable[Concatenate[ProtoGraph, P], tuple[ProtoGraph, bool]]:
    @functools.wraps(f)
    def wrapped(protograph: ProtoGraph, /, *args: P.args, **kwargs: P.kwargs) -> tuple[ProtoGraph, bool]:
        protograph, had_effect = f(protograph, *args, **kwargs)
        if debug_asserts_enabled():
            for _ in range(IDEMPOTENT_REPEATS):
                _, had_effect_on_rerun = f(protograph, *args, **kwargs)
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
    end_uses = {protoblock: OrderedSet[VariableKey]() for protoblock in protograph}
    blocks_to_process = collections.deque(protograph)
    while blocks_to_process:
        protoblock = blocks_to_process.popleft()
        target_uses = tuple(itertools.chain(*(uses[target] for target, _ in protoblock.jump_targets)))
        end_uses[protoblock].update(k for k in target_uses if k.scope != VariableScope.CONST)
        transitive_uses = OrderedSet(
            source
            for use in target_uses
            if isinstance(source := protoblock.flow._end.get(use, use), VariableKey)
            and source.scope != VariableScope.CONST
        )
        if transitive_uses - uses[protoblock]:
            uses[protoblock].update(transitive_uses)
            blocks_to_process.extend(protograph.parents[protoblock])

    new_uses = {i: (uses_i.difference(i.uses), end_uses[i]) for i, uses_i in uses.items()}
    return {i: new_uses_i for i, new_uses_i in new_uses.items() if new_uses_i[0]}


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
    substitutions = {}
    for protoblock, (new_uses, end_uses) in _get_missing_transitive(protograph).items():
        new_flow = IntraBlockFlow(
            _flow=protoblock.flow._flow,
            _begin={**{k: AbstractRef("Transitive") for k in new_uses}, **protoblock.flow._begin},
            _end={**{k: k for k in end_uses}, **protoblock.flow._end},
        )
        substitutions[protoblock] = new_protoblock = ProtoBlock(protoblock.raw_instructions, new_flow)
        new_protoblock.uses.update(protoblock.uses | new_uses)

        # Ensure the new transitive dependencies are reflected in the variable state.
        initial_keys = OrderedSet(k for k, _ in new_protoblock.begin_state)
        assert not (delta := new_uses.difference(initial_keys)), delta

    return protograph.substitute(substitutions), bool(substitutions)


def _inter_block_edges(proto_graph: ProtoGraph) -> ValueEdges:
    for protoblock in proto_graph:
        for child, _ in protoblock.jump_targets:
            outputs = dict(protoblock.end_state)
            child_inputs = dict(child.begin_state)
            for key, child_input in child_inputs.items():
                yield outputs.get(key, ValueMissing()), child_input

            # `_add_transitive` should ensure the stacks match.
            # (Except for return blocks which may discard the stack.)
            if child.raw_instructions[-1] not in RAISE_RETURN_INSTRUCTIONS:
                s_out = tuple(sorted(i.identifier for i in outputs if i.scope == VariableScope.STACK))
                s_in = tuple(sorted(i.identifier for i in child_inputs if i.scope == VariableScope.STACK))
                assert s_out == s_in, f"{s_out=} != {s_in=}, {child.raw_instructions[-1].opname}"


def _graph_input_edges(proto_graph: ProtoGraph) -> ValueEdges:
    for key, initial_ref in proto_graph.root.begin_state:
        if isinstance(initial_ref, ExternalRef):
            continue

        assert isinstance(initial_ref, AbstractRef), initial_ref
        assert key.scope not in (VariableScope.CONST, VariableScope.STACK)
        yield ExternalRef(key), initial_ref


def _phivalue_constituent(proto_graph: ProtoGraph) -> ValueEdges:
    for _, initial_ref in proto_graph.root.begin_state:
        if isinstance(initial_ref, AbstractPhiValue):
            yield from ((constituent, initial_ref) for constituent in initial_ref.constituents)


@check_idempotent
def _condense_values(
    proto_graph: ProtoGraph,
    *edge_sources: Callable[[ProtoGraph], ValueEdges],
) -> tuple[ProtoGraph, bool]:
    edge_sources = (*edge_sources, _phivalue_constituent)
    edges = tuple(itertools.chain(*[fn(proto_graph) for fn in edge_sources]))
    replace_map: dict[AbstractValue, AbstractValue] = {}
    for v, condensed in compute_condense_map(edges).items():
        # Check invariants.
        assert condensed
        if not isinstance(v, AbstractPhiValue):
            invariants = (set(condensed) == {v}, not isinstance(v, AbstractRef))
            assert all(invariants) or not any(invariants), (invariants, v, condensed)

        # `AbstractPhiValue._unpack_apply` will determine if we need an AbstractPhiValue.
        if (replacement := AbstractPhiValue(condensed).substitute({})) != v:  # type: ignore
            replace_map[v] = replacement

    return proto_graph.transform(replace_map), bool(replace_map)


def _tuple_fold(protograph: ProtoGraph) -> tuple[ProtoGraph, bool]:
    """Replace tuple accesses (`BINARY_SUBSCR`, `UNPACK_SEQUENCE` instructions) with their members, if known.

    Note: The pass, as it is currently written, only folds one layer of tuples, from one known source, the
    `BUILD_TUPLE` instruction. This should be enough for our needs. However, it's easy enough to imagine a world
    where we would want to call this in a loop with other operations that fold values. In other words, tuple folding
    would be more powerful the more `AbstractValue`s we can prove to be tuples with a known source.
    """

    @dataclasses.dataclass(slots=True, unsafe_hash=True, eq=True)
    class TupleKey:
        tup_value: AbstractValue
        idx: int

        def __lt__(self, other: "TupleKey") -> bool:
            return self.__hash__() < other.__hash__()

    # {(tup, idx): val}
    tuple_sources: collections.defaultdict[TupleKey, AbstractValue] = collections.defaultdict()
    queries: collections.defaultdict[TupleKey, AbstractValue] = collections.defaultdict()

    for pb in protograph:
        for instruction, node in pb.node_flow:
            if (opname := instruction.opname) == "BUILD_TUPLE":
                assert node.inputs is not None  # The elements of the tuple
                assert len(node.outputs) == 1  # The output tuple
                tuple_sources.update({TupleKey(node.outputs[0], _i): i for _i, i in enumerate(reversed(node.inputs))})
            elif opname == "BINARY_SUBSCR":
                assert len(node.inputs) == 2  # The tuple, and the index
                assert len(node.outputs) == 1  # The result of tup[idx]
                if isinstance(ref := node.inputs[0], ExternalRef):
                    if ref.key.scope == VariableScope.CONST and isinstance(ref.key.identifier, int):
                        queries.update({TupleKey(node.inputs[1], ref.key.identifier): node.outputs[0]})
            elif opname == "UNPACK_SEQUENCE":
                assert len(node.inputs) == 1  # The tuple to unpack
                assert len(node.outputs)  # The elements of the tuple, unpacked
                queries.update({TupleKey(node.inputs[0], _i): o for _i, o in enumerate(node.outputs)})
            elif opname == "UNPACK_EX":
                continue
                assert len(node.inputs) == 1  # The tuple to unpack
                assert len(node.outputs) >= 2  # [remaining, unpack...]
                # Note: The remaining elements are a list, not a tuple.
                # That makes this pass technically unsound, as even if an element is updated,
                # this pass will fold the old value. It's unclear if there's any reason in the language
                # why unpacking this way returns a list, but it's unlikely to come up in practice,
                # and editing this list can be made an error in the future.
                # Another note: It should be possible to fold the remaining elements as well,
                # but as it is actually a list, let's not track that for now.
                queries.update({TupleKey(node.inputs[0], _i): o for _i, o in enumerate(node.outputs[1:])})

    if not queries or not tuple_sources:
        return (protograph, False)

    transform_replacements: dict[AbstractValue, AbstractValue] = {}
    for query_key, query_value in queries.items():
        if source_value := tuple_sources.get(query_key):
            transform_replacements[query_value] = source_value

    if not transform_replacements:
        return (protograph, False)

    return (protograph.transform(transform_replacements), True)


def apply_protograph_passes(protograph: ProtoGraph) -> ProtoGraph:
    protograph, _ = _add_transitive(protograph)
    protograph, _ = _condense_values(protograph, _inter_block_edges, _graph_input_edges)
    protograph, _ = _tuple_fold(protograph)
    return protograph
