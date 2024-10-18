from collections.abc import Callable, Sequence
from enum import auto, Enum
from typing import Any

from thunder.core.prims import make_prim
from thunder.core.proxies import TensorProxy
from thunder.core.trace import TraceCtx


class IDs(Enum):
    TORCH_AUTOGRAD_FUNCTION = auto()


def torch_autograd_function_meta(
    *,
    backward: TraceCtx | Callable,
    return_none_instead_of_grads: bool,
    saved_tensors: Sequence[TensorProxy],
    saved_other: Sequence[Any],
    flat_args: Sequence[TensorProxy],
    flat_output: Sequence[TensorProxy],
):
    return tuple(TensorProxy(like=out) for out in flat_output)


connect_to_torch_autograd = make_prim(
    IDs.TORCH_AUTOGRAD_FUNCTION,
    "connect_to_torch_autograd",
    meta=torch_autograd_function_meta,
)
