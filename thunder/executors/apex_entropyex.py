from functools import partial
from typing import Any, Callable

import torch

from lightning_utilities.core.imports import package_available

import thunder.torch as ltorch
from thunder.core.prims import prim_ctx
from thunder.core.proxies import TensorProxy
from thunder.core.symbol import Symbol
from thunder.core.transforms import (
    deregister_augmented_forward_and_backward,
    register_augmented_forward_with_checker,
    register_backward,
)

APEX_CROSS_ENTROPY_AVAILABLE = package_available("xentropy_cuda")

xentropy_cuda: None | Any = None
if APEX_CROSS_ENTROPY_AVAILABLE:
    import xentropy_cuda

# This function wraps the xentropy_cuda.forward function for concrete execution.
# It accepts and returns real PyTorch tensors.
def cross_entropy_impl(
    return_max_log_sum_exp: bool,
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
        losses = losses.mean().to(a.dtype)
    elif reduction == "sum":
        losses = losses.sum().to(a.dtype)
    elif reduction == "none":
        losses = losses.to(a.dtype)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    if return_max_log_sum_exp:
        return losses, max_log_sum_exp
    return losses


# This function wraps the xentropy_cuda.backward function for concrete execution.
# It accepts and returns real PyTorch tensors.
def apex_cross_entropy_backward_impl(
    grad,
    logits,
    labels,
    max_log_sum_exp,
    smoothing,
):
    return xentropy_cuda.backward(grad.contiguous(), logits, max_log_sum_exp, labels, smoothing)


# This function checks if the xentropy_cuda.forward function can be used for
# concrete execution. It accepts abstract TensorProxy objects and return a bool.
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

    probability_target = a.shape == target.shape
    if probability_target or label_smoothing > 0.0:
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


class apexsymbol:
    """Decorator for registering a custom Symbol for Apex cross entropy executor.
    """
    def __init__(self, id: str):
        self.id = id

    def __call__(self, fn: Callable) -> Symbol:
        sym = Symbol(name=fn.__name__, meta=prim_ctx(fn), id=self.id, is_prim=True)
        return sym


# This function is the abstract forward function for the Apex cross entropy
# executor. It accepts abstract TensorProxy objects and returns abstract
# TensorProxy objects. In a way, this function is the "specification" of the
# Apex cross entropy forward function that is used at trace time and can be
# replaced by a concrete implementation for execution. Now to our tracing
# engine, this function is just another function that can be traced. It is not
# special in any way. The only thing that makes it special is that we will
# register a concrete implementation for it in the next step.
@apexsymbol("apex_cross_entropy_forward")
def apex_cross_entropy_forward(
    a,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    max_log_sum_exp = TensorProxy(like=target)
    if reduction == "none":
        return TensorProxy(like=target), max_log_sum_exp
    elif reduction == "mean":
        return TensorProxy(like=target, shape=()), max_log_sum_exp
    elif reduction == "sum":
        return TensorProxy(like=target, shape=()), max_log_sum_exp
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


# This function is the abstract backward function for the Apex cross entropy
# executor. It accepts abstract TensorProxy objects and returns abstract
# TensorProxy objects. It is the "specification" of the Apex cross entropy
# backward function that is used at trace time and can be replaced by a concrete
# implementation for execution.
@apexsymbol("apex_cross_entropy_backward")
def apex_cross_entropy_backward(
    grad,
    logits,
    labels,
    max_log_sum_exp,
    smoothing,
):
    return TensorProxy(like=logits)


# Here we create a dictionary that maps from Symbol names to concrete
# implementations of the Apex cross entropy inference forward, trainign forward
# and backward functions.
_op_to_xentropy = {
    "torch.nn.functional.cross_entropy": (
        "apex_cross_entropy",
        cross_entropy_checker,
        partial(cross_entropy_impl, False),
    ),
    "apex_cross_entropy_forward": (
        "apex_cross_entropy_forward",
        cross_entropy_checker,
        partial(cross_entropy_impl, True),
    ),
    "apex_cross_entropy_backward": (
        "apex_cross_entropy_backward",
        lambda *args, **kwargs: True,
        apex_cross_entropy_backward_impl,
    ),
}

# Check out
# https://github.com/Lightning-AI/lightning-thunder/blob/main/dev_tutorials/thunder-add-vjp-rule.md
# for a tutorial on how to add a VJP rule for any Symbol. We use our new
# primitives to register a VJP rule for torch.nn.functional.cross_entropy. This
# function is registered as the augmented forward rule for
# torch.nn.functional.cross_entropy inside the register_apex_entropyex function
# below.
def apex_cross_entropy_forward_rule(
    a,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    loss, max_log_sum_exp = apex_cross_entropy_forward(
        a,
        target,
        weight,
        size_average,
        ignore_index,
        reduce,
        reduction,
        label_smoothing,
    )
    primal = loss
    saved_for_backward = (a, target, max_log_sum_exp, reduction, label_smoothing)
    return primal, saved_for_backward


# This function is the backward rule for torch.nn.functional.cross_entropy. It
# accepts the primal output and saved_for_backward from the forward pass and
# returns the backward output. The backward output is a tuple of the backward
# output for each input to the forward pass. In this case, the forward pass has
# 7 inputs, so the backward output is a tuple of 7 elements. The backward output
# for each input is None if the input is not differentiable. This function is
# registered as the backward rule for torch.nn.functional.cross_entropy inside
# the register_apex_entropyex function below.
def apex_cross_entropy_backward_rule(
    logits,
    labels,
    max_log_sum_exp,
    reduction,
    smoothing,
    grad,
):
    from thunder.core.transforms import mean_backward, sum_backward

    if reduction == "mean":
        grad = mean_backward(max_log_sum_exp.ndim, max_log_sum_exp.shape, (0,), grad)
    elif reduction == "sum":
        grad = sum_backward(max_log_sum_exp.shape, (0,), grad)
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    grad_logits = apex_cross_entropy_backward(
        grad,
        logits,
        labels,
        max_log_sum_exp,
        smoothing,
    )
    return grad_logits, *([None] * 7)


def register_apex_entropyex(*, add_to_default_executors: bool = True) -> None:
    assert (
        APEX_CROSS_ENTROPY_AVAILABLE
    ), f"Trying to register the Apex cross entropy executor, but the xentropy_cuda package is not available"

    from thunder.executors import add_operator_executor

    # Forward rule is conditionally used based on the arguments and the checker
    # function. The backward rule is always used when the forward rule is used.
    register_augmented_forward_with_checker(
        "torch.nn.functional.cross_entropy", cross_entropy_checker, apex_cross_entropy_forward_rule
    )
    register_backward("torch.nn.functional.cross_entropy")(apex_cross_entropy_backward_rule)

    return add_operator_executor("apex_xentropy", _op_to_xentropy, add_to_default_executors=add_to_default_executors)


def deregister_apex_entropyex() -> None:
    from thunder.executors import remove_operator_executor

    deregister_augmented_forward_and_backward("torch.nn.functional.cross_entropy")

    return remove_operator_executor("apex_xentropy")
