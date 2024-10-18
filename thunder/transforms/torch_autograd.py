from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import auto, Enum
from typing import Any

from thunder.core.baseutils import check

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


def get_backward(trace: TraceCtx) -> TraceCtx | Callable:
    from thunder.executors.torchex import connect_to_autograd_impl

    connect_to_autograd_bsym = next(
        filter(
            lambda bsym: bsym.sym.id in (connect_to_autograd_impl.id, connect_to_torch_autograd.id),
            reversed(trace.bound_symbols),
        ),
        None,
    )
    check(connect_to_autograd_bsym is not None, lambda: "Could not find connect_to_autograd in the trace")
    return connect_to_autograd_bsym.kwargs["backward"]


def set_backward(trace: TraceCtx, backward: TraceCtx | Callable):
    from thunder.executors.torchex import connect_to_autograd_impl

    bsyms = list(trace.bound_symbols)
    connect_to_autograd_bsym = next(
        filter(
            lambda bsym: bsym.sym.id in (connect_to_autograd_impl.id, connect_to_torch_autograd.id),
            reversed(trace.bound_symbols),
        ),
        None,
    )
    check(connect_to_autograd_bsym is not None, lambda: "Could not find connect_to_autograd in the trace")
    connect_to_autograd_bsym_idx = bsyms.index(connect_to_autograd_bsym)
    bsyms[connect_to_autograd_bsym_idx] = replace(
        connect_to_autograd_bsym, kwargs=connect_to_autograd_bsym.kwargs | {"backward": backward}
    )
    trace.bound_symbols = bsyms
    return trace
