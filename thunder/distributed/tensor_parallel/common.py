from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from thunder.core.proxies import TensorProxy


__all__ = [
    "PrePostProcessInterface",
    "NoOp",
]


class PrePostProcessInterface(ABC):
    @abstractmethod
    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        return x, (None,)

    @abstractmethod
    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        return y


@dataclass(frozen=True)
class NoOp(PrePostProcessInterface):
    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        return super().preprocess(x)

    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        return super().postprocess(y)
