from typing import Any

import torch

import thunder.torch as ltorch

from lightning_utilities.core.imports import package_available

APEX_CROSS_ENTROPY_AVAILABLE = package_available("xentropy_cuda")

xentropy_cuda: None | Any = None
if APEX_CROSS_ENTROPY_AVAILABLE:
    import xentropy_cuda


def cross_entropy_impl(
    a,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    if ignore_index >= 0:
        raise ValueError("Cannot use ignore_index with apex_xentropy.")

    if weight is not None or size_average is not None or reduce is not None:
        raise ValueError("Cannot use weight, size_average or reduce with apex_xentropy.")

    half_to_float = False
    losses, max_log_sum_exp = xentropy_cuda.forward(a, target, label_smoothing, half_to_float)

    if reduction == "mean":
        return losses.mean().to(a.dtype)
    elif reduction == "sum":
        return losses.sum().to(a.dtype)
    elif reduction == "none":
        return losses.to(a.dtype)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def cross_entropy_checker(
    a,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    if xentropy_cuda is None:
        return False

    torch_dtype = ltorch.to_torch_dtype(a.dtype)
    if torch_dtype not in (torch.float16, torch.float32):
        return False

    # We only want to use this function if ignore_index is not used
    if ignore_index >= 0:
        return False

    # We only want to use this function if weight is None
    if weight is not None:
        return False

    # These arguments are deprecated and not supported
    if size_average is not None or reduce is not None:
        return False

    # We only support reduction of "sum", "mean" or "none"
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


_op_to_xentropy = {
    "torch.nn.functional.cross_entropy": ("apex_cross_entropy", cross_entropy_checker, cross_entropy_impl),
}


def register_apex_entropyex(*, add_to_default_executors: bool = True) -> None:
    assert (
        APEX_CROSS_ENTROPY_AVAILABLE
    ), f"Trying to register the Apex cross entropy executor, but the xentropy_cuda package is not available"

    from thunder.executors import add_operator_executor

    return add_operator_executor("apex_xentropy", _op_to_xentropy, add_to_default_executors=add_to_default_executors)


def deregister_apex_entropyex() -> None:
    from thunder.executors import remove_operator_executor

    return remove_operator_executor("apex_xentropy")
