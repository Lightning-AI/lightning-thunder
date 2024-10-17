from __future__ import annotations
from typing import TYPE_CHECKING

from thunder.core.proxies import SubclassTensorProxy
from thunder.core import utils

if TYPE_CHECKING:
    from thunder.core.trace import TraceCtx
    from thunder.core.symbol import BoundSymbol


__all__ = [
    "flatten_tensor_subclasses",
]


def flatten_tensor_subclasses(computation_trace: TraceCtx) -> TraceCtx:
    bsym: BoundSymbol
    for bsym in computation_trace.bound_symbols:
        for a in bsym.flat_proxy_args + bsym.flat_proxy_outs:
            utils.check(
                not isinstance(a, SubclassTensorProxy),
                lambda: f"{bsym} has Tensor Subclasses of {a}",
                exception_type=NotImplementedError,
            )
    return computation_trace
