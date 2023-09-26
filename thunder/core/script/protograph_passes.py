import collections
import dataclasses
import functools
import itertools
from typing import Callable, Concatenate
from collections.abc import Iterable

from typing_extensions import ParamSpec

from thunder.core.script.algorithms import compute_condense_map
from thunder.core.script.protograph import (
    is_detail,
    _Symbolic,
    AbstractPhiValue,
    AbstractRef,
    AbstractValue,
    CompositeRef,
    CompositeValue,
    ExternalRef,
    IntermediateValue,
    IntraBlockFlow,
    ProtoGraph,
    ProtoBlock,
    VariableKey,
    ValueMissing,
)
from thunder.core.script.python_ir_data import ThunderInstruction, VariableScope, RAISE_RETURN_INSTRUCTIONS
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

        # The reason we can ignore ALL `_OutputRef`s (including those that would index into a composite)
        # is that the (potential) composite's dependencies are already handled by `ProtoBlock._flow_uses`.
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


def _connect_protograph(proto_graph: "ProtoGraph") -> "ProtoGraph":
    proto_graph, _ = _add_transitive(proto_graph)
    proto_graph, _ = _condense_values(proto_graph, _inter_block_edges, _graph_input_edges)
    assert not (missing := _get_missing_transitive(proto_graph)), missing
    for protoblock in proto_graph:
        for k, v in protoblock.begin_state:
            assert not is_detail(v), (k, v)
    return proto_graph


def _tuple_fold(proto_graph: ProtoGraph) -> tuple[ProtoGraph, bool]:
    """Replace tuple accesses (`BINARY_SUBSCR`, `UNPACK_SEQUENCE` instructions) with their members, if known.

    Note: The pass, as it is currently written, only folds one layer of tuples, from one known source, the
    `BUILD_TUPLE` instruction. This should be enough for our needs. However, it's easy enough to imagine a world
    where we would want to call this in a loop with other operations that fold values. In other words, tuple folding
    would be more powerful the more `AbstractValue`s we can prove to be tuples with a known source.
    """
    original_graph = proto_graph
    replacements: dict[ThunderInstruction, tuple[_Symbolic.Output, ...]] = {}
    known_tuples: OrderedSet[IntermediateValue] = OrderedSet()

    def replace_fn(instruction: ThunderInstruction, old_symbolic: _Symbolic) -> _Symbolic | None:
        if (new_outputs := replacements.pop(instruction, None)) is not None:
            return dataclasses.replace(old_symbolic, outputs=new_outputs)

    for instruction, (_, node) in proto_graph.flat_flow:
        if instruction.opname == "BUILD_TUPLE" and isinstance(output := node.outputs[0], IntermediateValue):
            assert len(node.outputs) == 1, node.outputs
            constituents = tuple(range(-len(node.inputs), 0))
            replacements[instruction] = (CompositeRef(v=output, constituents=constituents),)
            known_tuples.add(output)

    while replacements:
        proto_graph = _connect_protograph(proto_graph.replace_symbolic(replace_fn))
        assert not replacements, replacements
        for instruction, (symbolic_node, materialized_node) in proto_graph.flat_flow:
            if (opname := instruction.opname) == "BINARY_SUBSCR":
                to_index, index = materialized_node.inputs
                is_tuple = isinstance(to_index, CompositeValue) and to_index.v in known_tuples
                is_const_idx = isinstance(index, ExternalRef) and index.key.scope == VariableScope.CONST
                if is_tuple and is_const_idx and isinstance(idx := index.key.identifier, int):
                    replacements[instruction] = ((-2, idx),)

            elif opname == "UNPACK_SEQUENCE":
                (to_unpack,) = materialized_node.inputs
                if isinstance(to_unpack, CompositeValue) and to_unpack.v in known_tuples:
                    indices = range(-1, -len(materialized_node.outputs) - 1, -1)
                    replacements[instruction] = tuple((-1, idx) for idx in indices)

            elif instruction.opname == "UNPACK_EX":
                pass  # TODO(apaz-cli): figure out indexing.

            # Remove no-ops so we know when to break out of the loop.
            if replacements.get(instruction) == symbolic_node.outputs:
                replacements.pop(instruction)

    return proto_graph, proto_graph is not original_graph


def apply_protograph_passes(protograph: ProtoGraph) -> ProtoGraph:
    protograph = _connect_protograph(protograph)
    protograph, _ = _tuple_fold(protograph)
    assert not (missing := _get_missing_transitive(protograph)), missing
    return protograph
