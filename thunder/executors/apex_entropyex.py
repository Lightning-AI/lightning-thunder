from functools import partial
from typing import Any
from collections.abc import Callable

import torch

from lightning_utilities.core.imports import package_available

import thunder.torch as ltorch
from thunder.core.proxies import TensorProxy
from thunder.core.symbol import Symbol
from thunder.core.utils import check, same_shape
from thunder.core.transforms import get_grad, put_grad, put_grads, mean_backward, restore_reduced_dims

from thunder.extend import OperatorExecutor, register_executor

__all__ = [
    "apex_ex",
]

APEX_CROSS_ENTROPY_AVAILABLE: bool = package_available("xentropy_cuda")

xentropy_cuda: None | Any = None
if APEX_CROSS_ENTROPY_AVAILABLE:
    # NOTE Even if the Apex package is available it can still fail to import properly
    try:
        import xentropy_cuda
    except Exception as ex:
        print(f"xentropy_cuda failed to import with exception {ex}")
        APEX_CROSS_ENTROPY_AVAILABLE = False


# TODO Does apex have a version this should use?
apex_ex = OperatorExecutor("apex", version="0.1")
register_executor(apex_ex)


def apex_available() -> bool:
    return APEX_CROSS_ENTROPY_AVAILABLE


#
# Registers the apex_cross_entropy operation
#


# TODO Consider performing the reduction as part of a traceable epilogue
#   See "Update the apex cross entropy executor to put its reduction in a
#        traceable epilogue"
# NOTE Apex's cross entropy doesn't accept ignore_index >= 0, or the weight, size_average, or reduce parameters
def _apex_cross_entropy_impl(
    a: torch.Tensor,
    /,
    target: torch.Tensor,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses: torch.Tensor
    max_log_sum_exp: torch.Tensor
    losses, max_log_sum_exp = xentropy_cuda.forward(a, target, label_smoothing, half_to_float := False)

    if reduction == "mean":
        losses = losses.mean().to(a.dtype)
    elif reduction == "sum":
        losses = losses.sum().to(a.dtype)
    elif reduction == "none":
        losses = losses.to(a.dtype)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    return losses, max_log_sum_exp


# TODO Consider definining a "like" function instead of a direct meta for the impl
def _apex_cross_entropy_meta(
    a: TensorProxy,
    /,
    target: TensorProxy,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> tuple[TensorProxy, TensorProxy]:
    losses: TensorProxy
    if reduction == "none":
        losses = TensorProxy(like=a)
    elif reduction == "mean":
        losses = TensorProxy(like=a, shape=())
    elif reduction == "sum":
        losses = TensorProxy(like=a, shape=())
    else:
        check(
            False,
            lambda: f"apex cross entropy expected reduction to be 'none', 'mean', or 'sum', but was given {reduction}",
        )

    max_log_sum_exp: TensorProxy = TensorProxy(like=target, dtype=a.dtype)
    return losses, max_log_sum_exp


apex_xentropy = apex_ex.register_operator(
    "apex_cross_entropy", meta=_apex_cross_entropy_meta, fn=_apex_cross_entropy_impl
)

#
# Registers the apex_cross_entropy_backward function
#


def _apex_cross_entropy_backward_impl(
    g: torch.Tensor,
    a: torch.Tensor,
    /,
    *,
    target: torch.Tensor,
    max_log_sum_exp: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    return xentropy_cuda.backward(g, a, max_log_sum_exp, target, label_smoothing)


def _apex_cross_entropy_backward_meta(
    g: TensorProxy,
    a: TensorProxy,
    /,
    *,
    target: TensorProxy,
    max_log_sum_exp: TensorProxy,
    label_smoothing: float,
) -> TensorProxy:
    return TensorProxy(like=a)


apex_xentropy_bwd = apex_ex.register_operator(
    "apex_cross_entropy_backward", meta=_apex_cross_entropy_backward_meta, fn=_apex_cross_entropy_backward_impl
)


#
# Registers apex cross entropy as an executor for torch.cross_entropy
#


def _cross_entropy_checker(
    a: TensorProxy,
    /,
    target: TensorProxy,
    weight: None | TensorProxy = None,
    size_average: None | Any = None,
    ignore_index: int = -100,
    reduce: None | Any = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> bool:
    probability_target: bool = same_shape(a.shape, target.shape)
    if probability_target or label_smoothing > 0.0:
        return False

    torch_dtype: torch.dtype = ltorch.to_torch_dtype(a.dtype)
    if torch_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False

    if ignore_index >= 0:
        return False

    if weight is not None:
        return False

    # NOTE These parameters are deprecated and not supported
    if size_average is not None or reduce is not None:
        return False

    if reduction not in ["sum", "mean", "none"]:
        return False

    # Checks from
    # https://github.com/NVIDIA/apex/blob/7b2e71b0d4013f8e2f9f1c8dd21980ff1d76f1b6/apex/contrib/csrc/xentropy/xentropy_kernel.cu#L587-L590
    if a.ndim != 2:
        return False

    if target.ndim != 1:
        return False

    if a.shape[0] != target.shape[0]:
        return False

    if a.numel == 0:
        return False

    # Xentropy kernel produces incorrect results if a.shape[1] is less
    # than 30 and not a multiple of 4
    if a.shape[1] < 30 and a.shape[1] % 4 != 0:
        return False

    return True


# Translate calls from torch.nn.functional.cross_entropy to apex_xentropy (when the checker above returns True)
def _cross_entropy_transform(
    a: TensorProxy,
    /,
    target: TensorProxy,
    weight: None | TensorProxy = None,
    size_average: None | Any = None,
    ignore_index: int = -100,
    reduce: None | Any = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> TensorProxy:
    result, _ = apex_xentropy(a, target, reduction, label_smoothing)
    return result


def _apex_cross_entropy_grad(
    a: TensorProxy,
    /,
    target: TensorProxy,
    weight: None | TensorProxy = None,
    size_average: None | Any = None,
    ignore_index: int = -100,
    reduce: None | Any = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> TensorProxy:
    fwd: TensorProxy
    max_log_sum_exp: TensorProxy
    fwd, max_log_sum_exp = apex_xentropy(a, target, reduction, label_smoothing)

    g: TensorProxy = get_grad(fwd)

    if reduction == "mean":
        g = mean_backward(max_log_sum_exp.ndim, max_log_sum_exp.shape, (0,), g)
    elif reduction == "sum":
        g = restore_reduced_dims(g, (0,), max_log_sum_exp.shape)

    # NOTE Apex's xentropy bwd requires the grad computation to be performed in fp32
    a_ = a.contiguous()
    a_grad: TensorProxy = apex_xentropy_bwd(
        g, a_, target=target, max_log_sum_exp=max_log_sum_exp, label_smoothing=label_smoothing
    )

    a_grad = a_grad.to(a.dtype)
    put_grad(a, a_grad)

    return fwd


# Registers the implementation for torch.nn.functional.cross_entropy
apex_ex.register_implementation(
    ltorch.cross_entropy,
    checker=_cross_entropy_checker,
    execution_transform=_cross_entropy_transform,
    grad_transform=_apex_cross_entropy_grad,
)
