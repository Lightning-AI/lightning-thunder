# WARNING: cudnn layernorm executor is experimental. Tests that use cudnn might fail.
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import numpy as np

from thunder.core.devices import to_torch_device
from thunder.core.dtypes import to_torch_dtype
from thunder.core.transforms import get_grad, put_grad
from thunder.core.proxies import TensorProxy
from thunder.executors.cudnnex import cudnn_available, torch_to_cudnn_dtype
from thunder.extend import OperatorExecutor

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number
    from thunder.core.proxies import NumberLike


__all__ = [
    "cudnn_layernorm_ex",
]
cudnn_layernorm_ex: None | OperatorExecutor = None


@dataclass(frozen=True)
class CudnnTensorAttributes:
    size: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device_index: int


def make_cacheable_cudnn_graph_inputs(func):
    def wrapper(*args, **kwargs):
        cudnn_input_args = [
            (
                CudnnTensorAttributes(arg.size(), arg.stride(), arg.dtype, args.device_index)
                if isinstance(arg, torch.Tensor)
                else arg
            )
            for arg in args
        ]
        return func(*cudnn_input_args, **kwargs)

    return wrapper


@make_cacheable_cudnn_graph_inputs
@lru_cache(maxsize=1024)
def _make_cudnn_layer_norm_graph(a_4d, weight_4d, bias_4d, *, is_inference: bool):
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    input = graph.tensor(name="input", dim=a_4d.size, stride=a_4d.stride, data_type=torch_to_cudnn_dtype(a_4d.dtype))
    scale = graph.tensor(
        name="scale", dim=weight_4d.size, stride=weight_4d.stride, data_type=torch_to_cudnn_dtype(weight_4d.dtype)
    )
    bias = graph.tensor(
        name="bias", dim=bias_4d.size, stride=bias_4d.stride, data_type=torch_to_cudnn_dtype(bias_4d.dtype)
    )

    epsilon = graph.tensor(
        name="epsilon", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT, is_pass_by_value=True
    )

    if is_inference:
        Y, _, _ = graph.layernorm(
            name="LN",
            norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
            input=input,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
        )
    else:
        Y, mean, inv_var = graph.layernorm(
            name="LN",
            norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
            input=input,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
        )
        mean.set_output(True).set_data_type(torch.float32)
        inv_var.set_output(True).set_data_type(torch.float32)

    Y.set_output(True).set_data_type(torch_to_cudnn_dtype(a_4d.dtype)).set_stride(a_4d.stride)

    graph.build([cudnn.heur_mode.A])

    if is_inference:
        return input, scale, bias, epsilon, Y, graph
    else:
        return input, scale, bias, epsilon, (Y, mean, inv_var), graph


# cudnn only supports following:
# input tensor shape: N, C, (D), H, W
# normalized shape:  (C, (D), H, W)
# convert all tensor shapes to above format
def _transform_layer_norm_inputs(a, normalized_shape, weight, bias):
    elements_to_normalize = np.prod(normalized_shape)
    batch_size = np.prod(a.shape[: -len(normalized_shape)], dtype=int)

    # Assume strides to be NCHW contiguous
    assumed_stride = (elements_to_normalize, 1, 1, 1)
    a_4d = CudnnTensorAttributes((batch_size, elements_to_normalize, 1, 1), assumed_stride, a.dtype, a.device.index)
    weight_4d = CudnnTensorAttributes(
        (1, elements_to_normalize, 1, 1), assumed_stride, weight.dtype, weight.device.index
    )
    bias_4d = CudnnTensorAttributes((1, elements_to_normalize, 1, 1), assumed_stride, bias.dtype, bias.device.index)

    return a_4d, weight_4d, bias_4d


def layer_norm_impl(
    a: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: Number = 1e-5,
) -> torch.Tensor:
    if weight is None:
        weight = torch.ones(normalized_shape, dtype=a.dtype, device=a.device)
    if bias is None:
        bias = torch.zeros(normalized_shape, dtype=a.dtype, device=a.device)
    a_4d, weight_4d, bias_4d = _transform_layer_norm_inputs(a, normalized_shape, weight, bias)
    input, scale, B, epsilon, Y, graph = _make_cudnn_layer_norm_graph(a_4d, weight_4d, bias_4d, is_inference=True)

    Y_actual = torch.zeros_like(a, device="cuda")

    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    cudnn_to_torch_tensor = {input: a, scale: weight, B: bias, epsilon: epsilon_cpu, Y: Y_actual}

    graph.execute(cudnn_to_torch_tensor, workspace)

    return Y_actual


def layer_norm_checker(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    if cudnn is None:
        return False

    t_device = to_torch_device(a.device)
    t_dtype = to_torch_dtype(a.dtype)
    if weight is None:
        weight = torch.ones(normalized_shape, dtype=t_dtype, device=t_device)
    if bias is None:
        bias = torch.zeros(normalized_shape, dtype=t_dtype, device=t_device)
    a_4d, weight_4d, bias_4d = _transform_layer_norm_inputs(a, normalized_shape, weight, bias)

    try:
        _make_cudnn_layer_norm_graph(a_4d, weight_4d, bias_4d, is_inference=True)
    except Exception:
        return False

    return True


def _layer_norm_aug_fwd_meta(
    a: TensorProxy,
    /,
    normalized_shape: Sequence[int],
    weight: TensorProxy | None = None,
    bias: TensorProxy | None = None,
    eps: NumberLike = 1e-5,
) -> tuple[TensorProxy, TensorProxy, TensorProxy]:
    return (
        TensorProxy(like=a),
        TensorProxy(like=a, shape=[a.shape[0], 1, 1, 1]),
        TensorProxy(like=a, shape=[a.shape[0], 1, 1, 1]),
    )


def layer_norm_aug_fwd_impl(
    a: torch.TensorProxy,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight is None:
        weight = torch.ones(normalized_shape, dtype=a.dtype, device=a.device)
    if bias is None:
        bias = torch.zeros(normalized_shape, dtype=a.dtype, device=a.device)
    a_4d, weight_4d, bias_4d = _transform_layer_norm_inputs(a, normalized_shape, weight, bias)
    input, scale, B, epsilon, (Y, mean, inv_var), graph = _make_cudnn_layer_norm_graph(
        a_4d, weight_4d, bias_4d, is_inference=False
    )

    Y_actual = torch.zeros_like(a, device="cuda")
    mean_actual = torch.empty((a.size(0), 1, 1, 1), device="cuda", dtype=torch.float32)
    inv_var_actual = torch.empty((a.size(0), 1, 1, 1), device="cuda", dtype=torch.float32)

    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    cudnn_to_torch_tensor = {
        input: a,
        scale: weight,
        B: bias,
        epsilon: epsilon_cpu,
        Y: Y_actual,
        mean: mean_actual,
        inv_var: inv_var_actual,
    }

    graph.execute(cudnn_to_torch_tensor, workspace)

    return Y_actual, mean_actual, inv_var_actual


@make_cacheable_cudnn_graph_inputs
@lru_cache(maxsize=1024)
def _make_cudnn_layer_norm_bwd_graph(
    grad_4d: CudnnTensorAttributes,
    a_4d: CudnnTensorAttributes,
    weight_4d: CudnnTensorAttributes,
    mean_4d: CudnnTensorAttributes,
    inv_var_4d: CudnnTensorAttributes,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cudnn.pygraph,
]:
    bwd_graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    DY = bwd_graph.tensor(
        name="DY", dim=grad_4d.size, stride=grad_4d.stride, data_type=torch_to_cudnn_dtype(grad_4d.dtype)
    )
    X_bwd = bwd_graph.tensor(name="X", dim=a_4d.size, stride=a_4d.stride, data_type=torch_to_cudnn_dtype(a_4d.dtype))
    weight_bwd = bwd_graph.tensor(
        name="weight", dim=weight_4d.size, stride=weight_4d.stride, data_type=torch_to_cudnn_dtype(weight_4d.dtype)
    )
    mean_bwd = bwd_graph.tensor(
        name="mean", dim=mean_4d.size, stride=mean_4d.stride, data_type=torch_to_cudnn_dtype(mean_4d.dtype)
    )
    inv_var_bwd = bwd_graph.tensor(
        name="inv_var", dim=inv_var_4d.size, stride=inv_var_4d.stride, data_type=torch_to_cudnn_dtype(inv_var_4d.dtype)
    )

    DX, DWeight, Dbias = bwd_graph.layernorm_backward(
        name="layernorm_bwd", grad=DY, input=X_bwd, scale=weight_bwd, mean=mean_bwd, inv_variance=inv_var_bwd
    )

    DX.set_output(True).set_data_type(torch_to_cudnn_dtype(a_4d.dtype))
    DWeight.set_output(True).set_data_type(torch_to_cudnn_dtype(weight_4d.dtype))
    Dbias.set_output(True).set_data_type(torch_to_cudnn_dtype(weight_4d.dtype))

    bwd_graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    return DY, X_bwd, weight_bwd, mean_bwd, inv_var_bwd, DX, DWeight, Dbias, bwd_graph


def _layer_norm_bwd_meta(
    grad: TensorProxy,
    a: TensorProxy,
    weight: TensorProxy | None,
    bias: TensorProxy | None,
    mean: TensorProxy,
    inv_var: TensorProxy,
    normalized_shape: Sequence[int],
) -> tuple[TensorProxy, TensorProxy | None, TensorProxy | None]:
    a_grad = TensorProxy(like=a)
    weight_grad = TensorProxy(like=a, shape=normalized_shape) if weight is not None else None
    bias_grad = TensorProxy(like=a, shape=normalized_shape) if grad is not None else None
    return a_grad, weight_grad, bias_grad


def layer_norm_bwd_impl(
    grad: torch.Tensor,
    a: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    mean: torch.Tensor,
    inv_var: torch.Tensor,
    normalized_shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if no_weight_grad := weight is None:
        weight = torch.ones(normalized_shape, dtype=a.dtype, device=a.device)
    if no_bias_grad := bias is None:
        bias = torch.zeros(normalized_shape, dtype=a.dtype, device=a.device)
    normalized_shape = weight.shape
    grad_4d, _, _ = _transform_layer_norm_inputs(grad, normalized_shape, weight, bias)
    a_4d, weight_4d, bias_4d = _transform_layer_norm_inputs(a, normalized_shape, weight, bias)

    mean_4d = CudnnTensorAttributes(mean.shape, mean.stride(), mean.dtype, mean.device.index)
    inv_var_4d = CudnnTensorAttributes(inv_var.shape, inv_var.stride(), inv_var.dtype, inv_var.device.index)

    DY, X_bwd, weight_bwd, mean_bwd, inv_var_bwd, DX, DWeight, Dbias, bwd_graph = _make_cudnn_layer_norm_bwd_graph(
        grad_4d,
        a_4d,
        weight_4d,
        mean_4d,
        inv_var_4d,
    )

    grad_a = torch.empty_like(a)
    grad_weight = torch.empty_like(weight) if weight is not None else None
    grad_bias = torch.empty_like(bias) if bias is not None else None

    workspace = torch.empty(bwd_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    tensor_map = {
        DY: grad,
        X_bwd: a,
        weight_bwd: weight,
        mean_bwd: mean,
        inv_var_bwd: inv_var,
        DX: grad_a,
        DWeight: grad_weight,
        Dbias: grad_bias,
    }

    bwd_graph.execute(tensor_map, workspace)

    if no_weight_grad:
        grad_weight = None
    if no_bias_grad:
        grad_bias = None
    return grad_a, grad_weight, grad_bias


def cudnn_layer_norm_grad_transform(
    a,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
):
    normalized, mean, inv_var = layer_norm_aug_fwd(a, normalized_shape, weight, bias, eps)
    grad = get_grad(normalized)
    d_a, d_weight, d_bias = layer_norm_bwd(grad, a, weight, bias, mean, inv_var, normalized_shape)
    put_grad(a, d_a)
    if weight is not None:
        put_grad(weight, d_weight)
    if bias is not None:
        put_grad(bias, d_bias)

    return normalized


if cudnn_available():
    from thunder.extend import register_executor
    import cudnn

    cudnn_layernorm_ex: OperatorExecutor = OperatorExecutor("cudnn_layernorm", version=cudnn.backend_version())
    register_executor(cudnn_layernorm_ex)

    import thunder.torch as ltorch

    layer_norm = cudnn_layernorm_ex.register_operator("cudnn_layernorm", like=ltorch.layer_norm, fn=layer_norm_impl)
    layer_norm_aug_fwd = cudnn_layernorm_ex.register_operator(
        "cudnn_layernorm_aug_fwd",
        meta=_layer_norm_aug_fwd_meta,
        fn=layer_norm_aug_fwd_impl,
    )
    layer_norm_bwd = cudnn_layernorm_ex.register_operator(
        "cudnn_layernorm_bwd",
        meta=_layer_norm_bwd_meta,
        fn=layer_norm_bwd_impl,
    )
    cudnn_layernorm_ex.register_implementation(
        ltorch.layer_norm,
        layer_norm,
        checker=layer_norm_checker,
        grad_transform=cudnn_layer_norm_grad_transform,
    )
