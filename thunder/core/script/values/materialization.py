from __future__ import annotations

import dataclasses
import functools
import itertools
from types import MappingProxyType
from typing import Literal, TypeVar
from collections.abc import Callable, Iterator

from thunder.core.script import parse
from thunder.core.script.values import base, composite, symbolic
from thunder.core.utils import FrozenDict, OrderedSet
from collections.abc import Iterable

__all__ = ("Materialized", "DigestFlow", "IntraBlockFlow")
T = TypeVar("T")


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
VarT = TypeVar("VarT", bound=base.AbstractValue, covariant=True)
ConcreteState = FrozenDict[parse.VariableKey, base.AbstractValue]
EndState = FrozenDict[parse.VariableKey, symbolic.Symbolic.Input]


@dataclasses.dataclass(frozen=True, eq=False)
class Materialized:
    """Flow element where all symbolic references have been resolved to concrete `AbstractValue`s."""

    inputs: base.HybridMap[base.AbstractValue]
    outputs: tuple[base.AbstractValue, ...]


class DigestFlow:
    """One-shot helper for materializing a block."""

    GetT = Callable[[symbolic.Symbolic.Input], base.AbstractValue]

    def __init__(self, begin: ConcreteState) -> None:
        self._begin = begin
        self._result: dict[parse.ThunderInstruction, Materialized] = {}

    def next(self, instruction: parse.ThunderInstruction, symbolic: symbolic.Symbolic) -> Materialized:
        """Lookup the materialized node corresponding to a symbolic node."""

        # NB: `inputs_after_op` will be needed after we introduce mutations.
        inputs = inputs_after_op = symbolic.inputs.map(self.get)
        outputs = tuple(composite.InternalRef.resolve(o, inputs=inputs_after_op) for o in symbolic.outputs)
        assert all(isinstance(o, base.AbstractValue) for o in outputs), outputs

        self._result[instruction] = result = Materialized(inputs, outputs)
        return result

    def get(self, key: symbolic.Symbolic.Input) -> base.AbstractValue:
        """Resolve a Symbolic input based on the block state at that node."""
        result: base.AbstractValue
        if isinstance(key, base.NonPyObject.Tag):
            result = base.NonPyObject(key)

        elif isinstance(key, symbolic.OutputRef):
            inputs = base.HybridMap(ordered=self._result[key.instruction].outputs)
            result = composite.InternalRef.resolve(key.idx, inputs=inputs)

        else:
            assert isinstance(key, parse.VariableKey), key
            result = base.ExternalRef(key) if key.is_const else self._begin[key]
        return result


@dataclasses.dataclass(frozen=True, eq=False)
class IntraBlockFlow:
    _symbolic: FrozenDict[parse.ThunderInstruction, symbolic.Symbolic]
    _begin: ConcreteState
    _end: EndState

    StateIterT = Iterator[tuple[parse.VariableKey, base.AbstractValue]]

    def __post_init__(self) -> None:
        assert not (forbidden := tuple(i for i in self._symbolic if i in parse.FORBIDDEN_INSTRUCTIONS)), forbidden
        object.__setattr__(self, "_symbolic", FrozenDict(self._symbolic))

        missing = {i: base.AbstractRef("Inferred") for i in self.uses if i not in self._begin}
        object.__setattr__(self, "_begin", FrozenDict({**missing, **self._begin}))
        assert not any(k.is_const for k in self._begin), self._begin
        assert not any(isinstance(v, composite.InternalRef) for v in self._begin.values()), self._begin

        object.__setattr__(self, "_end", FrozenDict(self._end))

    @functools.cache
    def __getitem__(self, key: tuple[symbolic.Symbolic.Input, Literal[0, 1]]) -> base.AbstractValue:
        assert key[1] in (0, 1)
        return self._computed[1:][key[1]](key[0])

    @property
    def symbolic(self) -> Iterable[tuple[parse.ThunderInstruction, symbolic.Symbolic]]:
        yield from self._symbolic.items()

    @property
    def materialized(self) -> FrozenDict[parse.ThunderInstruction, Materialized]:
        return self._computed[0]

    @property
    def begin_state(self) -> StateIterT:
        yield from self._sort_and_filter_state(iter(self._begin.items()))

    @property
    def end_state(self) -> StateIterT:
        yield from self._sort_and_filter_state((k, self[v, 1]) for k, v in self._end.items())

    @staticmethod
    def _sort_and_filter_state(kv: StateIterT) -> StateIterT:
        yield from ((k, v) for k, v in sorted(kv) if not isinstance(v, base.NonPyObject))

    @property
    def uses(self) -> OrderedSet[parse.VariableKey]:
        assignment = (v for k, v in self._end.items() if isinstance(v, parse.VariableKey) and not v.is_const and k != v)
        return OrderedSet(itertools.chain(*(s.uses for _, s in self.symbolic), assignment))

    _Computed = tuple[
        FrozenDict[parse.ThunderInstruction, Materialized],
        DigestFlow.GetT,  # Begin
        DigestFlow.GetT,  # End
    ]

    @functools.cached_property
    def _computed(self) -> _Computed:
        flow_state = DigestFlow(self._begin)
        materialized_flow: FrozenDict[parse.ThunderInstruction, Materialized]
        materialized_flow = FrozenDict({i: flow_state.next(i, s) for i, s in self.symbolic})  # Populates `flow_state`
        return materialized_flow, DigestFlow(self._begin).get, flow_state.get

    def substitute(self, replace_map: base.ReplaceMap) -> IntraBlockFlow | None:
        """Replace `AbstractValue`s within the flow. (Block inputs and producer nodes.)"""
        replace_map_view = MappingProxyType(replace_map)
        new_symbolic: FrozenDict[parse.ThunderInstruction, symbolic.Symbolic]
        new_symbolic = FrozenDict({k: (s.substitute(replace_map_view)) for k, s in self.symbolic})
        begin = ConcreteState({k: base.substitute_value(v, replace_map_view) for k, v in self._begin.items()})

        # TODO(robieta): Check if a value is only present in `materialized` and error.
        if self._symbolic != new_symbolic or self._begin != begin:
            return dataclasses.replace(self, _symbolic=new_symbolic, _begin=begin)
        return None
