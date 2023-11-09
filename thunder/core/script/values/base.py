from __future__ import annotations

import dataclasses
import enum
import textwrap
from typing import overload, Any, Generic, NewType, TypeVar
from collections.abc import Callable
from collections.abc import Mapping

from typing_extensions import Self

from thunder.core.script import parse
from thunder.core.utils import FrozenDict

__all__ = (
    # HybridMap
    "Reference",
    "TraitName",
    "HybridMap",
    #
    # Values
    "AbstractValue",
    "AbstractRef",
    "NonPyObject",
    "IntermediateValue",
    "ExternalRef",
    #
    # Substitution
    "substitute_value",
    "ReplaceMap",
)


# =============================================================================
# == Generic (hybrid tuple/dict) container ====================================
# =============================================================================
T = TypeVar("T")
T1 = TypeVar("T1")
Reference = NewType("Reference", int)
TraitName = NewType("TraitName", str)


@dataclasses.dataclass(frozen=True, eq=True)
class HybridMap(Generic[T]):
    ordered: tuple[T, ...] = dataclasses.field(kw_only=True, default_factory=tuple)
    named: FrozenDict[TraitName, T] = dataclasses.field(kw_only=True, default_factory=FrozenDict)

    def __getitem__(self, key: Reference | TraitName) -> T:
        if isinstance(key, int):
            return self.ordered[key]
        elif isinstance(key, str):
            return self.named[key]
        raise TypeError(f"Invalid key: {key}")

    def __repr__(self) -> str:
        parts = [f"{self.__class__.__name__}("]
        if self.ordered:
            ordered = "\n".join(repr(i) for i in self.ordered)
            parts.append(f"  ordered:\n{textwrap.indent(ordered, ' ' * 4)}")

        if self.named:
            named = "\n".join(f"{k}: {v}" for k, v in self.named.items())
            parts.append(f"  named:\n{textwrap.indent(named, ' ' * 4)}")

        return "\n".join((*parts, ")"))

    @overload
    def map(self, f: Callable[[T], T]) -> Self:
        ...

    @overload
    def map(self, f: Callable[[T], T1]) -> HybridMap[T1]:
        ...

    def map(self, f: Any) -> Any:
        ordered = tuple(f(i) for i in self.ordered)
        named: FrozenDict[TraitName, T] = FrozenDict({k: f(v) for k, v in self.named.items()})
        return dataclasses.replace(self, ordered=ordered, named=named)

    def get(self, name: TraitName) -> T | None:
        return self.named.get(name)

    def add_named(self, name: TraitName, value: T) -> Self:
        named = dict(self.named)
        named.update({name: value})  # Preserve order.
        return dataclasses.replace(self, named=FrozenDict(named))


# =============================================================================
# == Simple value types =======================================================
# =============================================================================
class AbstractValue:
    """Represents a value during instruction parsing. (Prior to type binding.)"""

    __is_detail = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        cls.__is_detail = kwargs.pop("__is_detail", False)  # `dataclasses` forces this into kwargs for some reason.
        super().__init_subclass__(**kwargs)

    def __copy__(self) -> AbstractValue:
        raise NotImplementedError

    @property
    def is_detail(self) -> bool:
        return self.__is_detail

    @property
    def identity(self) -> AbstractValue:
        """Analogous to `id(obj)`. For composites there is a layer of state management above the value itself.

        This is not suitable for equality checks (for example, mutation does not change
        an object's identity), but it is often the appropriate target for `isinstance` checks.
        """
        return self

    def _unpack_apply(self, _: ReplaceMap) -> AbstractValue:
        """Recursively update any constituent references in the abstract value."""
        return self


@dataclasses.dataclass(frozen=True, eq=False)
class AbstractRef(AbstractValue, __is_detail=True):
    """Placeholder value which will be resolved during parsing."""

    _debug_info: str = "N/A"


@dataclasses.dataclass(frozen=True, eq=True)
class NonPyObject(AbstractValue):
    """Singleton values used to signal some special interpreter state."""

    class Tag(enum.Enum):
        DELETED = enum.auto()
        MISSING = enum.auto()
        NULL = enum.auto()

    tag: Tag

    def __repr__(self) -> str:
        return self.tag.name


class IntermediateValue(AbstractValue):
    """A (potentially) new value produced by an instruction."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(at {hex(id(self))})"


@dataclasses.dataclass(frozen=True, eq=True)
class ExternalRef(AbstractValue):
    """Reference values outside of the parsed code. (Arguments, constants, globals, etc.)"""

    key: parse.VariableKey

    def __repr__(self) -> str:
        if self.key.is_const:
            return f"Const({self.key.identifier})"
        return f"{self.__class__.__name__}({self.key.identifier}, {self.key.scope.name})"


# =============================================================================
# == Value substitution =======================================================
# =============================================================================
ReplaceMap = Mapping["AbstractValue", "AbstractValue"]


@overload
def substitute_value(v: AbstractValue, replace_map: ReplaceMap) -> AbstractValue:
    ...


@overload
def substitute_value(v: T, replace_map: ReplaceMap) -> T:
    ...


def substitute_value(v: Any, replace_map: ReplaceMap) -> Any:
    """Find the replacement for `v`, and recursively substitute. (If applicable.)

    Some abstract values reference other abstract values. When we make substitution during
    graph transformations it is necessary to also consider replacement of an abstract
    value's constituents. Any subclass which must be unpacked in this manner should
    override `_unpack_apply`.
    """
    if not isinstance(v, AbstractValue):
        return v

    new_v = replace_map.get(v, v)
    if new_v != (x := replace_map.get(new_v, new_v)):
        msg = f"""
            `replace_map` may not contain chains.
                {v}
                {new_v}
                {x}
            See `flatten_map`."""
        raise ValueError(textwrap.dedent(msg))

    return new_v._unpack_apply(replace_map)
