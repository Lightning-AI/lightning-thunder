from __future__ import annotations
from typing import TYPE_CHECKING

from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

if TYPE_CHECKING:
    from thunder.core.symbol import BoundSymbol, VariableInterface
    from thunder.core.trace import TraceCtx


__all__ = [
    "flatten_tensor_subclasses",
]


def flatten_tensor_subclasses(computation_trace: TraceCtx) -> TraceCtx:

    return computation_trace
