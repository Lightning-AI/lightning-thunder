import collections
import dataclasses
import functools
import itertools
from typing import Concatenate
from collections.abc import Callable
from collections.abc import Iterable

from typing_extensions import ParamSpec

from thunder.core.script import parse, values
from thunder.core.script.protograph import ProtoGraph, ProtoGraphTransform, AddTransitive, ReplaceSymbolic
from thunder.core.utils import debug_asserts_enabled

ValueEdges = Iterable[tuple[values.AbstractValue, values.AbstractValue]]
P = ParamSpec("P")
IDEMPOTENT_REPEATS = 10  # Check for nondeterministic behavior.
KNOWN_TUPLE = values.TraitName("__known_tuple")


# TODO(robieta): remove once `_prune_epilogues` is removed.
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


def _connect_protograph(proto_graph: "ProtoGraph") -> "ProtoGraph":
    proto_graph = proto_graph.link()
    assert AddTransitive(proto_graph).apply(or_default=False) is None
    for protoblock in proto_graph:
        for k, v in protoblock.flow.begin_state:
            assert not v.is_detail, (k, v)
    return proto_graph


class MarkTuples(ReplaceSymbolic):
    def apply_to_symbolic(
        self,
        instruction: parse.ThunderInstruction,
        symbolic: values.Symbolic,
        _: values.HybridMap[values.AbstractValue],
    ) -> values.Symbolic | None:
        if instruction.opname == "BUILD_TUPLE" and isinstance(output := symbolic.outputs[0], values.IntermediateValue):
            assert len(symbolic.outputs) == 1, symbolic.outputs
            ordered = tuple(values.Reference(i) for i in range(-len(symbolic.inputs.ordered), 0))
            new_output = values.CompositeRef(ordered=ordered).add_named(KNOWN_TUPLE, values.ConstRef(True))
            return dataclasses.replace(symbolic, outputs=(new_output.add_identity(output),))
        return None


class IndexTuples(ReplaceSymbolic):
    def apply_to_symbolic(
        self,
        instruction: parse.ThunderInstruction,
        symbolic: values.Symbolic,
        inputs: values.HybridMap[values.AbstractValue],
    ) -> values.Symbolic | None:
        replacement: values.Symbolic | None = None
        if instruction.opname == "BINARY_SUBSCR":
            to_index, index = inputs.ordered
            is_tuple = isinstance(to_index, values.CompositeValue) and to_index.get(KNOWN_TUPLE)
            index_key = index.key if isinstance(index, values.ExternalRef) and index.key.is_const else None
            if is_tuple and index_key and isinstance(idx := index_key.identifier, int):
                assert len(symbolic.outputs) == 1
                replacement = dataclasses.replace(symbolic, outputs=((values.Reference(0), values.Reference(idx)),))

        elif instruction.opname == "UNPACK_SEQUENCE":
            (to_unpack,) = inputs.ordered
            if isinstance(to_unpack, values.CompositeValue) and to_unpack.get(KNOWN_TUPLE):
                indices = (values.Reference(idx) for idx in range(-1, -len(symbolic.outputs) - 1, -1))
                outputs = tuple((values.Reference(0), idx) for idx in indices)
                replacement = dataclasses.replace(symbolic, outputs=outputs)

        elif instruction.opname == "UNPACK_EX":
            pass  # TODO(apaz-cli): figure out indexing.

        return replacement if (replacement and replacement.outputs != symbolic.outputs) else None


def _tuple_fold(proto_graph: ProtoGraph) -> ProtoGraph:
    """Replace tuple accesses (`BINARY_SUBSCR`, `UNPACK_SEQUENCE` instructions) with their members, if known."""
    return ProtoGraphTransform.chain(proto_graph, MarkTuples, IndexTuples) or proto_graph


def apply_protograph_passes(protograph: ProtoGraph) -> ProtoGraph:
    protograph = _tuple_fold(protograph.unlink())
    protograph = _connect_protograph(protograph)
    assert AddTransitive(protograph).apply(or_default=False) is None
    return protograph
