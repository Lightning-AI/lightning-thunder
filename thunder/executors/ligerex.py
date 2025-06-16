from typing import Tuple, Optional
from collections.abc import Sequence
import math
import functools

import torch

try:
    import triton

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    import liger_kernel.ops.rms_norm
    import liger_kernel.ops.layer_norm
    import liger_kernel.ops.cross_entropy
    import liger_kernel.ops.group_norm
    from liger_kernel.ops.utils import calculate_settings

    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

import thunder
import thunder.core.dtypes as dtypes
from thunder.core.proxies import TensorProxy, AnyProxy
from thunder.core.transforms import get_grad, put_grads
import thunder.core.utils as utils
import thunder.core.devices as devices

from thunder.extend import OperatorExecutor, register_executor


liger_ex: OperatorExecutor = OperatorExecutor("liger", version="0.5.8")
register_executor(liger_ex)


def liger_available() -> bool:
    return LIGER_AVAILABLE


def triton_available() -> bool:
    return TRITON_AVAILABLE


prod = lambda *args: functools.reduce(lambda x, y: x * y, args)


def rms_norm_fwd_meta(
    X: TensorProxy,
    W: TensorProxy,
    eps: float,
    offset: int,
    casting_mode: bool,
) -> tuple[TensorProxy, TensorProxy, TensorProxy, int, int, bool]:
    *n_rows, n_cols = X.shape
    n_rows = prod(*n_rows)
    rstd_dtype = (
        thunder.dtypes.float32
        if casting_mode
        in (liger_kernel.ops.rms_norm._CASTING_MODE_LLAMA.value, liger_kernel.ops.rms_norm._CASTING_MODE_GEMMA.value)
        else X.dtype
    )
    Y = TensorProxy(like=X)
    RSTD = TensorProxy(like=X, shape=(n_rows,), dtype=rstd_dtype)
    BLOCK_SIZE, num_warps = liger_kernel.ops.rms_norm.calculate_settings(n_cols)
    return Y, TensorProxy(like=X, shape=(n_rows, n_cols)), RSTD, BLOCK_SIZE, num_warps, casting_mode


if liger_available():
    liger_rms_norm_forward = liger_ex.register_operator(
        "liger_rms_norm_forward",
        meta=rms_norm_fwd_meta,
        fn=liger_kernel.ops.rms_norm.rms_norm_forward,
    )


def rms_norm_bwd_meta(
    dY: TensorProxy,
    X: TensorProxy,
    W: TensorProxy,
    RSTD: TensorProxy,
    offset: float,
    casting_mode: str,
    BLOCK_SIZE: int,
    num_warps: int,
    in_place: bool = False,
) -> tuple[TensorProxy, TensorProxy]:
    return TensorProxy(like=X), TensorProxy(like=W)


if liger_available():
    liger_rms_norm_backward = liger_ex.register_operator(
        "liger_rms_norm_backward",
        meta=rms_norm_bwd_meta,
        fn=liger_kernel.ops.rms_norm.rms_norm_backward,
    )


def rms_norm_meta(x: TensorProxy, shape, w: TensorProxy, eps: float):
    return TensorProxy(like=x)


if liger_available():
    rms_norm = liger_ex.register_operator(
        "rms_norm",
        meta=rms_norm_meta,
        fn=torch.nn.functional.rms_norm,
        replaces=torch.nn.functional.rms_norm,
    )


def rms_norm_grad_transform(x: TensorProxy, shape, weight: TensorProxy, eps: float):
    Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = liger_rms_norm_forward(
        x, weight, eps, offset=0.0, casting_mode="llama"
    )
    dY = get_grad(Y)
    dX, dW = liger_rms_norm_backward(
        dY,
        X,
        weight,
        RSTD,
        offset=0.0,
        casting_mode="llama",
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        in_place=False,
    )
    dX = dX.view(*x.shape)
    put_grads((x, weight), (dX, dW))
    return Y


def rms_norm_execution_transform(x: TensorProxy, weight: TensorProxy, eps: float):
    Y, *_ = liger_rms_norm_forward(x, weight, eps, offset=0.0, casting_mode="llama")
    return Y


if liger_available():
    liger_ex.register_implementation(
        rms_norm,
        execution_transform=rms_norm_execution_transform,
        grad_transform=rms_norm_grad_transform,
    )


def layer_norm_fwd_meta(
    X: TensorProxy,
    W: TensorProxy,
    B: TensorProxy,
    eps: float,
) -> tuple[TensorProxy, TensorProxy, TensorProxy, TensorProxy, int, int]:
    *n_rows, n_cols = X.shape
    n_rows = prod(*n_rows)
    BLOCK_SIZE, num_warps = liger_kernel.ops.layer_norm.calculate_settings(n_cols)
    Y = TensorProxy(like=X, shape=(n_rows, n_cols), dtype=X.dtype)
    Mean = TensorProxy(like=X, shape=(n_rows,), dtype=X.dtype)
    RSTD = TensorProxy(like=X, shape=(n_rows,), dtype=X.dtype)
    return Y, TensorProxy(like=X), Mean, RSTD, BLOCK_SIZE, num_warps


if liger_available():
    liger_layer_norm_forward = liger_ex.register_operator(
        "liger_layer_norm_forward",
        meta=layer_norm_fwd_meta,
        fn=liger_kernel.ops.layer_norm.layer_norm_forward,
    )


def layer_norm_bwd_meta(
    dY: TensorProxy,
    X: TensorProxy,
    W: TensorProxy,
    B: TensorProxy,
    Mean: TensorProxy,
    RSTD: TensorProxy,
) -> tuple[TensorProxy, TensorProxy, TensorProxy]:
    return TensorProxy(like=X), TensorProxy(like=W), TensorProxy(like=B)


if liger_available():
    liger_layer_norm_backward = liger_ex.register_operator(
        "liger_layer_norm_backward",
        meta=layer_norm_bwd_meta,
        fn=liger_kernel.ops.layer_norm.layer_norm_backward,
    )


def layer_norm_meta(x, shape, w, b, eps):
    return TensorProxy(like=x)


if liger_available():
    layer_norm = liger_ex.register_operator(
        "layer_norm",
        meta=layer_norm_meta,
        fn=torch.nn.functional.layer_norm,
        replaces=torch.nn.functional.layer_norm,
    )


def layer_norm_grad_transform(x, shape, weight, bias, eps):
    Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = liger_layer_norm_forward(x, weight, bias, eps)
    dY = get_grad(Y)
    dX, dW, dB = liger_layer_norm_backward(dY, X, weight, bias, Mean, RSTD)
    dX = dX.view(*x.shape)
    put_grads((x, weight), (dX, dW))
    return Y


def layer_norm_execution_transform(x, weight, bias, eps):
    Y, *_ = liger_layer_norm_forward(x, weight, bias, eps)
    return Y


if liger_available():
    liger_ex.register_implementation(
        layer_norm,
        execution_transform=layer_norm_execution_transform,
        grad_transform=layer_norm_grad_transform,
    )


def cross_entropy_fwd_meta(
    _input: TensorProxy,
    target: TensorProxy,
    weight: TensorProxy | None,
    ignore_index: int | None,
    lse_square_scale: float | None,
    label_smoothing: float | None,
    reduction: str | None,
    softcap: float | None = None,
    return_z_loss: bool | None = False,
) -> tuple[TensorProxy, TensorProxy, TensorProxy]:
    BT, V = _input.shape
    n_rows = BT
    loss = TensorProxy(like=_input, shape=(n_rows,), dtype=_input.dtype, device=_input.device)
    z_loss = TensorProxy(like=_input, shape=(n_rows,), dtype=_input.dtype, device=_input.device)
    return loss, z_loss, TensorProxy(like=_input)


if liger_available():
    liger_cross_entropy_forward = liger_ex.register_operator(
        "liger_cross_entropy_forward",
        meta=cross_entropy_fwd_meta,
        fn=liger_kernel.ops.cross_entropy.cross_entropy_forward,
    )


def cross_entropy_bwd_meta(
    _input: TensorProxy,
    grad_output: TensorProxy,
) -> TensorProxy:
    return TensorProxy(like=_input)


if liger_available():
    liger_cross_entropy_backward = liger_ex.register_operator(
        "liger_cross_entropy_backward",
        meta=cross_entropy_bwd_meta,
        fn=liger_kernel.ops.cross_entropy.cross_entropy_backward,
    )


def cross_entropy_meta(
    x: TensorProxy,
    target: TensorProxy,
    weight: TensorProxy | None = None,
    size_average: bool | None = None,
    ignore_index: int | None = -100,
    reduce: bool | None = None,
    reduction: str | None = "mean",
    label_smoothing: float | None = 0.0,
) -> TensorProxy:
    return TensorProxy(like=x)


if liger_available():
    cross_entropy = liger_ex.register_operator(
        "cross_entropy",
        meta=cross_entropy_meta,
        fn=torch.nn.functional.cross_entropy,
        replaces=torch.nn.functional.cross_entropy,
    )


def cross_entropy_grad_transform(
    x: TensorProxy,
    target: TensorProxy,
    weight: TensorProxy | None = None,
    size_average: bool | None = None,
    ignore_index: int | None = -100,
    reduce: bool | None = None,
    reduction: str | None = "mean",
    label_smoothing: float | None = 0.0,
):
    loss, _, _ = liger_cross_entropy_forward(x, target, weight, ignore_index, 0.0, label_smoothing, reduction)
    dloss = get_grad(loss)
    dX = liger_cross_entropy_backward(x, dloss)
    put_grads((x,), (dX,))
    return loss


def cross_entropy_execution_transform(
    x: TensorProxy,
    target: TensorProxy,
    weight: TensorProxy | None = None,
    size_average: bool | None = None,
    ignore_index: int | None = -100,
    reduce: bool | None = None,
    reduction: str | None = "mean",
    label_smoothing: float | None = 0.0,
):
    loss, _, _ = liger_cross_entropy_forward(x, target, weight, ignore_index, 0.0, label_smoothing, reduction)
    return loss


if liger_available():
    liger_ex.register_implementation(
        cross_entropy,
        execution_transform=cross_entropy_execution_transform,
        grad_transform=cross_entropy_grad_transform,
    )


def group_norm_fwd_meta(
    X: TensorProxy,
    num_channels: int,
    num_groups: int,
    W: TensorProxy,
    B: TensorProxy,
    eps: float,
) -> tuple[TensorProxy, TensorProxy, TensorProxy, TensorProxy, int]:
    shape = X.shape
    batch_size = shape[0]
    hidden_size = X.shape[-1]
    Y = TensorProxy(like=X, shape=X.shape, dtype=X.dtype, device=X.device)
    BLOCK_SIZE = min(liger_kernel.ops.group_norm.MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))
    Mean = TensorProxy(like=X, shape=(batch_size, num_groups), dtype=X.dtype, device=X.device)
    RSTD = TensorProxy(like=X, shape=(batch_size, num_groups), dtype=X.dtype, device=X.device)
    return Y, TensorProxy(like=X), Mean, RSTD, BLOCK_SIZE


if liger_available() and triton_available():
    liger_group_norm_forward = liger_ex.register_operator(
        "liger_group_norm_forward",
        meta=group_norm_fwd_meta,
        fn=liger_kernel.ops.group_norm.group_norm_forward,
    )


def group_norm_bwd_meta(
    dY: TensorProxy,
    X: TensorProxy,
    W: TensorProxy,
    B: TensorProxy,
    Mean: TensorProxy,
    RSTD: TensorProxy,
    num_channels: int,
    num_groups: int,
) -> tuple[TensorProxy, TensorProxy, TensorProxy]:
    return TensorProxy(like=X), TensorProxy(like=W), TensorProxy(like=B)


if liger_available():
    liger_group_norm_backward = liger_ex.register_operator(
        "liger_group_norm_backward",
        meta=group_norm_bwd_meta,
        fn=liger_kernel.ops.group_norm.group_norm_backward,
    )


def group_norm_meta(
    x: TensorProxy, num_groups: int, weight: TensorProxy | None, bias: TensorProxy | None, eps: float = 1e-5
):
    return TensorProxy(like=x)


if liger_available():
    group_norm = liger_ex.register_operator(
        "group_norm",
        meta=group_norm_meta,
        fn=torch.nn.functional.group_norm,
        replaces=torch.nn.functional.group_norm,
    )


def group_norm_grad_transform(
    x: TensorProxy, num_groups: int, weight: TensorProxy | None, bias: TensorProxy | None, eps: float = 1e-5
):
    num_channels = x.shape[1]
    Y, X, Mean, RSTD, BLOCK_SIZE = liger_group_norm_forward(x, num_channels, num_groups, weight, bias, eps)
    dY = get_grad(Y)
    dX, dW, dB = liger_group_norm_backward(dY, X, weight, bias, Mean, RSTD, num_channels, num_groups)
    put_grads((x, weight, bias), (dX, dW, dB))
    return Y


def group_norm_execution_transform(
    x: TensorProxy, num_groups: int, weight: TensorProxy | None, bias: TensorProxy | None, eps: float = 1e-5
):
    num_channels = x.shape[1]
    Y, *_ = liger_group_norm_forward(x, num_channels, num_groups, weight, bias, eps)
    return Y


if liger_available():
    liger_ex.register_implementation(
        group_norm,
        execution_transform=group_norm_execution_transform,
        grad_transform=group_norm_grad_transform,
    )
