from typing import Tuple, Optional
from collections.abc import Sequence
import math
import functools

import torch

import litgpt

import triton

import liger_kernel.ops.rms_norm
import liger_kernel.ops.layer_norm
import liger_kernel.ops.cross_entropy
import liger_kernel.ops.rope
import liger_kernel.ops.geglu
import liger_kernel.ops.swiglu
import liger_kernel.ops.kl_div
import liger_kernel.ops.fused_linear_cross_entropy
import liger_kernel.ops.group_norm
from liger_kernel.ops.utils import calculate_settings

import thunder
import thunder.core.dtypes as dtypes
from thunder.core.proxies import TensorProxy, AnyProxy
from thunder.core.transforms import get_grad, put_grads
import thunder.core.utils as utils
import thunder.core.devices as devices

from thunder.extend import OperatorExecutor, register_executor


liger_ex: OperatorExecutor = OperatorExecutor("liger", version="0.5.8")
register_executor(liger_ex)


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


liger_rms_norm_backward = liger_ex.register_operator(
    "liger_rms_norm_backward",
    meta=rms_norm_bwd_meta,
    fn=liger_kernel.ops.rms_norm.rms_norm_backward,
)


def rms_norm_meta(x: TensorProxy, shape, w: TensorProxy, eps: float):
    return TensorProxy(like=x)


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


liger_ex.register_implementation(
    rms_norm,
    execution_transform=rms_norm_execution_transform,
    grad_transform=rms_norm_grad_transform,
)


def geglu_fwd_meta(
    a: TensorProxy,
    b: TensorProxy,
) -> TensorProxy:
    return TensorProxy(like=a)


liger_geglu_forward = liger_ex.register_operator(
    "liger_geglu_forward",
    meta=geglu_fwd_meta,
    fn=liger_kernel.ops.geglu.geglu_forward,
)


def geglu_bwd_meta(
    a: TensorProxy,
    b: TensorProxy,
    dc: TensorProxy,
) -> tuple[TensorProxy, TensorProxy]:
    return TensorProxy(like=a), TensorProxy(like=b)


liger_geglu_backward = liger_ex.register_operator(
    "liger_geglu_backward",
    meta=geglu_bwd_meta,
    fn=liger_kernel.ops.geglu.geglu_backward,
)


def geglu_grad_transform(a: TensorProxy, b: TensorProxy) -> TensorProxy:
    c = liger_geglu_forward(a, b)
    dc = get_grad(c)
    da, db = liger_geglu_backward(a, b, dc)
    put_grads((a, b), (da, db))
    return c


def geglu_execution_transform(a: TensorProxy, b: TensorProxy) -> TensorProxy:
    c = liger_geglu_forward(a, b)
    return c


def geglu_impl(a: TensorProxy, b: TensorProxy) -> TensorProxy:
    c = liger_geglu_forward(a, b)
    return c


liger_geglu = liger_ex.register_operator("liger_geglu", fn=geglu_impl, like=geglu_impl)


liger_ex.register_implementation(
    liger_geglu,
    execution_transform=geglu_execution_transform,
    grad_transform=geglu_grad_transform,
)


def rope_fwd_meta(
    q: TensorProxy,
    k: TensorProxy,
    cos: TensorProxy,
    sin: TensorProxy,
) -> tuple[TensorProxy, TensorProxy, TensorProxy, TensorProxy]:
    return TensorProxy(like=q), TensorProxy(like=k), cos, sin


liger_rope_forward = liger_ex.register_operator(
    "liger_rope_forward",
    meta=rope_fwd_meta,
    fn=liger_kernel.ops.rope.rope_forward,
)


def rope_bwd_meta(
    dq: TensorProxy,
    dk: TensorProxy,
    cos: TensorProxy,
    sin: TensorProxy,
) -> tuple[TensorProxy, TensorProxy, TensorProxy, TensorProxy]:
    return TensorProxy(like=dq), TensorProxy(like=dk)


liger_rope_backward = liger_ex.register_operator(
    "liger_rope_backward",
    meta=rope_bwd_meta,
    fn=liger_kernel.ops.rope.rope_backward,
)


def rope_grad_transform(q, k, cos, sin):
    q_out, k_out, _, _ = liger_rope_forward(q, k, cos, sin)
    q_out_grad = get_grad(q_out)
    k_out_grad = get_grad(k_out)
    dq, dk = liger_rope_backward(q_out_grad, k_out_grad, cos, sin)
    put_grads((q, k), (dq, dk))
    return q_out, k_out


def rope_execution_transform(q, k, cos, sin):
    q_out, k_out, _, _ = liger_rope_forward(q, k, cos, sin)
    return q_out, k_out


def rope_impl(q, k, cos, sin):
    qr, kr, _, _ = liger_rope_forward(q, k, cos, sin)
    return qr, kr


liger_rope = liger_ex.register_operator("liger_rope", fn=rope_impl, like=rope_impl)


liger_ex.register_implementation(
    liger_rope,
    execution_transform=rope_execution_transform,
    grad_transform=rope_grad_transform,
)


def apply_rope_meta(x, cos, sin):
    return TensorProxy(like=x)


litgpt_apply_rope = liger_ex.register_operator(
    "litgpt_apply_rope", fn=litgpt.model.apply_rope, meta=apply_rope_meta, replaces=litgpt.model.apply_rope
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


liger_layer_norm_backward = liger_ex.register_operator(
    "liger_layer_norm_backward",
    meta=layer_norm_bwd_meta,
    fn=liger_kernel.ops.layer_norm.layer_norm_backward,
)


def layer_norm_meta(x, shape, w, b, eps):
    return TensorProxy(like=x)


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


liger_ex.register_implementation(
    cross_entropy,
    execution_transform=cross_entropy_execution_transform,
    grad_transform=cross_entropy_grad_transform,
)


def swiglu_fwd_meta(
    a: TensorProxy,
    b: TensorProxy,
) -> TensorProxy:
    return TensorProxy(like=a)


def swiglu_forward_impl(
    a: TensorProxy,
    b: TensorProxy,
) -> TensorProxy:
    _, _, res = liger_kernel.ops.swiglu.swiglu_forward(a, b)
    return res


liger_swiglu_forward = liger_ex.register_operator(
    "liger_swiglu_forward",
    meta=swiglu_fwd_meta,
    fn=swiglu_forward_impl,
)


def swiglu_bwd_meta(
    a: TensorProxy,
    b: TensorProxy,
    dc: TensorProxy,
) -> tuple[TensorProxy, TensorProxy]:
    return TensorProxy(like=a), TensorProxy(like=b)


liger_swiglu_backward = liger_ex.register_operator(
    "liger_swiglu_backward",
    meta=swiglu_bwd_meta,
    fn=liger_kernel.ops.swiglu.swiglu_backward,
)


def swiglu_grad_transform(a, b):
    res = liger_swiglu_forward(a, b)
    grad_res = get_grad(res)
    grad_a, grad_b = liger_swiglu_backward(a, b, grad_res)
    put_grads((a, b), (grad_a, grad_b))
    return res


liger_ex.register_implementation(
    liger_swiglu_forward,
    grad_transform=swiglu_grad_transform,
    execution_transform=liger_swiglu_forward,
)


def kl_div_fwd_meta(
    y_pred: TensorProxy,
    y_true: TensorProxy,
    log_target: bool | None = False,
    reduction: str | None = "none",
    eps: float | None = 1e-10,
) -> TensorProxy:
    BT, V = y_pred.shape
    out_size = (BT, V) if reduction == liger_kernel.ops.kl_div._REDUCTION_MODE_NONE.value else (BT,)

    return TensorProxy(like=y_pred, shape=out_size)


liger_kl_div_forward = liger_ex.register_operator(
    "liger_kl_div_forward",
    meta=kl_div_fwd_meta,
    fn=liger_kernel.ops.kl_div.kldiv_forward_triton,
)


def kl_div_bwd_meta(
    target: TensorProxy,
    grad_output: TensorProxy,
    new_grads: TensorProxy | None = None,
    log_target: bool = False,
) -> TensorProxy:
    return TensorProxy(like=new_grads)


liger_kl_div_backward = liger_ex.register_operator(
    "liger_kl_div_backward",
    meta=kl_div_bwd_meta,
    fn=liger_kernel.ops.kl_div.kldiv_backward_triton,
)


def kl_div_meta(
    _input: TensorProxy,
    target: TensorProxy,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "none",
    log_target: bool = False,
) -> TensorProxy:
    return TensorProxy(like=_input)


kl_div = liger_ex.register_operator(
    "kl_div",
    meta=kl_div_meta,
    fn=torch.nn.functional.kl_div,
    replaces=torch.nn.functional.kl_div,
)


def kl_div_grad_transform(
    _input: TensorProxy,
    target: TensorProxy,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "none",
    log_target: bool = False,
) -> TensorProxy:
    x = liger_kl_div_forward(_input, target, log_target=log_target, reduction=reduction)
    dx = get_grad(x)
    grad_input = TensorProxy(like=_input)
    grad = liger_kl_div_backward(target, dx, _input, log_target)
    put_grads((_input,), (grad,))
    return x


def kl_div_execution_transform(
    input: TensorProxy,
    target: TensorProxy,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "none",
    log_target: bool = False,
) -> TensorProxy:
    x = liger_kl_div_forward(input, target, reduction=reduction, log_target=log_target)
    return x


liger_ex.register_implementation(
    kl_div,
    execution_transform=kl_div_execution_transform,
    grad_transform=kl_div_grad_transform,
)


def fused_linear_cross_entropy_fwd_meta(
    _input: TensorProxy,
    weight: TensorProxy,
    target: TensorProxy,
    bias: TensorProxy | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> tuple[TensorProxy, TensorProxy, TensorProxy, TensorProxy | None]:
    BT, H = _input.shape
    grad_weight = TensorProxy(like=weight)
    grad_input = TensorProxy(like=_input)
    grad_bias = None if bias is None else TensorProxy(like=bias)
    loss = TensorProxy(like=_input, shape=(BT,))
    return loss, grad_input, grad_weight, grad_bias


liger_fused_linear_cross_entropy_forward = liger_ex.register_operator(
    "liger_fused_linear_cross_entropy_forward",
    meta=fused_linear_cross_entropy_fwd_meta,
    fn=liger_kernel.ops.fused_linear_cross_entropy.fused_linear_cross_entropy_forward,
)


def fused_linear_cross_entropy_bwd_meta(
    grad_output: TensorProxy,
    grad_input: TensorProxy,
    grad_weight: TensorProxy,
    grad_bias: TensorProxy | None = None,
) -> tuple[TensorProxy, TensorProxy, TensorProxy | None]:
    return (
        TensorProxy(like=grad_input),
        TensorProxy(like=grad_weight),
        None if grad_bias is None else TensorProxy(like=grad_bias),
    )


liger_fused_linear_cross_entropy_backward = liger_ex.register_operator(
    "liger_fused_linear_cross_entropy_backward",
    meta=fused_linear_cross_entropy_bwd_meta,
    fn=liger_kernel.ops.fused_linear_cross_entropy.fused_linear_cross_entropy_backward,
)


def fused_linear_cross_entropy_grad_transform(
    _input: TensorProxy,
    weight: TensorProxy,
    target: TensorProxy,
    bias: TensorProxy = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
):
    loss, grad_input_1, grad_weight_1, grad_bias_1 = liger_fused_linear_cross_entropy_forward(
        _input,
        weight,
        target,
        bias=bias,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    grad_loss = get_grad(loss)
    grad_input, grad_weight, grad_bias = liger_fused_linear_cross_entropy_backward(
        grad_loss, grad_input_1, grad_weight_1, grad_bias_1
    )
    put_grads((_input, weight, target), (grad_input, grad_weight, grad_bias))
    return loss


liger_ex.register_implementation(
    liger_fused_linear_cross_entropy_forward,
    grad_transform=fused_linear_cross_entropy_grad_transform,
    execution_transform=liger_fused_linear_cross_entropy_forward,
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


liger_group_norm_backward = liger_ex.register_operator(
    "liger_group_norm_backward",
    meta=group_norm_bwd_meta,
    fn=liger_kernel.ops.group_norm.group_norm_backward,
)


def group_norm_meta(
    x: TensorProxy, num_groups: int, weight: TensorProxy | None, bias: TensorProxy | None, eps: float = 1e-5
):
    return TensorProxy(like=x)


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


liger_ex.register_implementation(
    group_norm,
    execution_transform=group_norm_execution_transform,
    grad_transform=group_norm_grad_transform,
)
