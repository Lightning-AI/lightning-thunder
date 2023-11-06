import abc
import dataclasses
import itertools
from typing import Any, TypeVar
from collections.abc import Iterable

from typing_extensions import Self

from thunder.core.script import parse
from thunder.core.script.values import base, symbolic
from thunder.core.utils import FrozenDict

__all__ = ("InternalRef", "OrderedSlice", "CompositeValue", "CompositeRef", "AbstractPhiValue")

T = TypeVar("T")


# =============================================================================
# == References ===============================================================
# =============================================================================
class InternalRef(base.AbstractValue, abc.ABC):
    @abc.abstractmethod
    def _resolve(self, inputs: base.HybridMap[base.AbstractValue]) -> base.AbstractValue:
        """Defines how to concretize itself."""
        ...

    @property
    def is_detail(self) -> bool:
        # All ref types are unsuitable for Graph binding.
        return True

    @classmethod
    def resolve(
        cls, output: symbolic.Symbolic.Output, *, inputs: base.HybridMap[base.AbstractValue] | None = None
    ) -> base.AbstractValue:
        inputs = base.HybridMap() if inputs is None else inputs
        if isinstance(output, (int, str)):
            return inputs[output]

        elif isinstance(output, symbolic.ConstRef):
            return base.ExternalRef(parse.VariableKey(output.identifier, parse.VariableScope.CONST))

        if isinstance(output, tuple):
            cls.validate_reference(output)
            result = inputs[output[0]]
            for idx in output[1:]:
                # We can only unpack a (possibly nested) composite.
                assert isinstance(result, base.HybridMap), result
                result = result[idx]

            assert isinstance(result, base.AbstractValue)
            return result

        elif isinstance(output, InternalRef):
            return output._resolve(inputs)

        return output

    @staticmethod
    def validate_reference(x: symbolic.NestedReference) -> None:
        x = (x,) if isinstance(x, (int, str)) else x
        assert isinstance(x, tuple) and x and all(isinstance(xi, (int, str)) for xi in x), x


# =============================================================================
# == Nesting ==================================================================
# =============================================================================
@dataclasses.dataclass(frozen=True, eq=True)
class OrderedSlice:
    reference: symbolic.NestedReference
    slice: slice

    def __hash__(self) -> int:
        # `slice` isn't hashable until 3.12
        return hash((self.reference, self.slice.start, self.slice.stop, self.slice.step))


@dataclasses.dataclass(frozen=True, eq=True, repr=False)
class _Composite(base.AbstractValue, base.HybridMap[T], __is_detail=True):
    """Models an AbstractValue that references other (possibly also AbstractValue) state.

    Note: `ordered` and `named` should not contain cycles.
    """

    Identity = base.TraitName("__Thunder_Object_Identity")

    def _unpack_apply(self, replace_map: base.ReplaceMap) -> base.AbstractValue:
        new_self = self.map(lambda x: base.substitute_value(x, replace_map))
        assert isinstance(new_self, _Composite)  # For mypy since we can't hint `_Composite[T] -> _Composite[T1]`
        return new_self

    def add_identity(self, identity: T) -> Self:
        return self.add_named(self.Identity, identity)

    # NOTE: We don't override `identity`. (Instead retaining `return self` from AbstractValue.)
    #       This is because we generally won't know a good value, and passes should do their
    #       own type checking. (And that checking should almost always be done on the materialized
    #       value, not the symbolic reference.)


@dataclasses.dataclass(frozen=True, eq=True, repr=False)
class CompositeValue(_Composite[base.AbstractValue]):
    def __post_init__(self) -> None:
        assert all(isinstance(i, base.AbstractValue) for i in self.ordered)
        assert all(isinstance(i, base.AbstractValue) for i in self.named.values())

    @property
    def is_detail(self) -> bool:
        return any(i.is_detail for i in itertools.chain(self.ordered, self.named.values()))

    @property
    def identity(self) -> base.AbstractValue:
        return self.named.get(self.Identity, self)


@dataclasses.dataclass(frozen=True, eq=True)
class CompositeRef(InternalRef, _Composite[symbolic.Symbolic.Output | OrderedSlice]):
    def __post_init__(self) -> None:
        assert not any(isinstance(i, OrderedSlice) for i in self.named.values())

    def _resolve(self, inputs: base.HybridMap[base.AbstractValue]) -> CompositeValue:
        ordered: list[base.AbstractValue] = []
        for i in self.ordered:
            if isinstance(i, OrderedSlice):
                slice_target = self.resolve(i.reference, inputs=inputs) if i.reference else inputs
                assert isinstance(slice_target, base.HybridMap)
                ordered.extend(slice_target.ordered[i.slice])
            else:
                ordered.append(self.resolve(i, inputs=inputs))

        named: dict[base.TraitName, base.AbstractValue] = {}
        for k, v in self.named.items():
            assert not isinstance(v, OrderedSlice)
            named[k] = self.resolve(v, inputs=inputs)

        return CompositeValue(ordered=tuple(ordered), named=FrozenDict(named))


# =============================================================================
# == Unions ===================================================================
# =============================================================================
@dataclasses.dataclass(frozen=True, eq=True)
class AbstractPhiValue(base.AbstractValue):
    constituents: tuple[base.AbstractValue, ...]

    def __post_init__(self) -> None:
        # Flatten nested PhiValues. e.g.
        #   𝜙[𝜙[A, B], 𝜙[A, C]] -> 𝜙[A, B, C]
        constituents = itertools.chain(*[self.flatten(i) for i in self.constituents])

        # Ensure a consistent order.
        constituents = tuple(v for _, v in sorted({hash(v): v for v in constituents}.items()))
        assert not any(isinstance(i, InternalRef) for i in constituents)
        object.__setattr__(self, "constituents", constituents)

    def __getitem__(self, _: Any) -> base.AbstractValue:
        # The semantics of indexing into an `AbstractPhiValue`` are not well defined:
        #  - The order of `constituents` is arbitrary
        #  - It's unclear if the desire is to select one constituent or create a new `AbstractPhiValue`
        #    which indexes into each constituent.
        # If a concrete use case emerges we can tackle it; until then we refuse for safety.

        # TODO(robieta): Handle traits
        raise NotImplementedError

    def _unpack_apply(self, replace_map: base.ReplaceMap) -> base.AbstractValue:
        result = AbstractPhiValue(tuple(base.substitute_value(v, replace_map) for v in self.constituents))
        return result if len(result.constituents) > 1 else result.constituents[0]

    @classmethod
    def flatten(cls, v: base.AbstractValue) -> Iterable[base.AbstractValue]:
        constituents = [cls.flatten(i) for i in v.constituents] if isinstance(v, AbstractPhiValue) else [[v]]
        yield from itertools.chain(*constituents)

    @property
    def is_detail(self) -> bool:
        return any(i.is_detail for i in self.constituents)
