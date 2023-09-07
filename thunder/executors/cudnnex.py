from typing import Any

import torch
import numpy as np

from lightning_utilities.core.imports import package_available

CUDNN_AVAILABLE = package_available("cudnn")

cudnn: None | Any = None
if CUDNN_AVAILABLE:
    import cudnn

# WARNING: cudnn executor is experimental. Tests that use cudnn might fail.\n
# Issue for tracking support: https://github.com/Lightning-AI/lightning-thunder/issues/880~

from dataclasses import dataclass
from functools import lru_cache
from typing import Union, Dict

import thunder.core.dtypes as dtypes


@dataclass(frozen=True)
class CudnnTensorAttributes:
    size: tuple
    stride: tuple
    dtype: torch.dtype


@lru_cache(maxsize=1024)
def _make_cudnn_sdpa_graph(query, key, value, attn_mask, is_causal):
    b, h, s_q, _ = query.size
    _, _, _, d_v = value.size

    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Q = graph.tensor(name="Q", dim=query.size, stride=query.stride, data_type=torch_to_cudnn_dtype(query.dtype))
    K = graph.tensor(name="K", dim=key.size, stride=key.stride, data_type=torch_to_cudnn_dtype(key.dtype))
    V = graph.tensor(name="V", dim=value.size, stride=value.stride, data_type=torch_to_cudnn_dtype(value.dtype))

    if attn_mask is None:
        Bias = None
    else:
        Bias = graph.tensor(
            name="bias", dim=attn_mask.size, stride=attn_mask.stride, data_type=torch_to_cudnn_dtype(attn_mask.dtype)
        )

    Attn_scale = graph.tensor(
        name="Attn_scale", dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT, is_pass_by_value=True
    )

    O, _ = graph.scaled_dot_product_flash_attention(
        name="scaled_dot_product_flash_attention",
        q=Q,
        k=K,
        v=V,
        is_inference=True,
        bias=Bias,
        use_causal_mask=is_causal,
        attn_scale=Attn_scale,
    )

    O.set_output(True).set_data_type(torch_to_cudnn_dtype(value.dtype)).set_stride([d_v * s_q * h, d_v * s_q, d_v, 1])

    graph.check_support()

    graph.build()

    return Q, K, V, Attn_scale, Bias, O, graph


@lru_cache(maxsize=1024)
def _make_cudnn_layer_norm_graph(a_4d, weight_4d, bias_4d):
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

    Y, _, _ = graph.layernorm(
        name="LN",
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
        input=input,
        scale=scale,
        bias=bias,
        epsilon=epsilon,
    )

    Y.set_output(True).set_data_type(torch_to_cudnn_dtype(a_4d.dtype)).set_stride(a_4d.stride)

    graph.check_support()
    graph.build()

    return input, scale, bias, epsilon, Y, graph


def torch_to_cudnn_dtype(lc_dtype: dtypes.dtype):
    _torch_to_cudnn_dtype_map: dict[Union[None, torch.dtype], cudnn.data_type] = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        None: cudnn.data_type.NOT_SET,
    }
    return _torch_to_cudnn_dtype_map[lc_dtype]


def sdpa_impl(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    key = key.transpose(-2, -1)
    if attn_mask is not None:
        while attn_mask.ndim < query.ndim:
            attn_mask = attn_mask.unsqueeze(0)

    cudnn_input_args = [
        CudnnTensorAttributes(arg.size(), arg.stride(), arg.dtype) if isinstance(arg, torch.Tensor) else arg
        for arg in [query, key, value, attn_mask, is_causal]
    ]

    Q, K, V, Attn_scale, Bias, O, graph = _make_cudnn_sdpa_graph(*cudnn_input_args)

    b, h, s_q, d_q = query.size()
    _, _, _, d_v = value.size()
    O_actual = torch.zeros(b, h, s_q, d_v, dtype=value.dtype, device="cuda")

    # Default value of scale, if not provided, in all torch versions
    if scale is None:
        scale = 1 / d_q**0.5
    Attn_scale_cpu = torch.full((1, 1, 1, 1), scale, dtype=torch.float32, device="cpu")

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    cudnn_to_torch_tensor = {Q: query, K: key, V: value, Attn_scale: Attn_scale_cpu, O: O_actual}

    if Bias:
        cudnn_to_torch_tensor[Bias] = attn_mask

    graph.execute(cudnn_to_torch_tensor, workspace)

    return O_actual


def sdpa_checker(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if cudnn is None:
        return False

    # TODO: the checker needs to call cudnn's check_support API
    # Just returning the output of check_support is sufficient to cover all
    # cudnn support across versions.
    # Below checks only exist as a workaround until check_support is implemented.

    if query.ndim != 4:
        return False

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        return False

    _, h, _, _ = query.size()
    if h % 8 != 0:
        return False

    return True


def layer_norm_impl(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    # cudnn only supports following:
    # input tensor shape: N, C, (D), H, W
    # normalized shape:  (C, (D), H, W)
    # convert all tensor shapes to above format
    elements_to_normalize = np.prod(normalized_shape)
    batch_size = np.prod(a.shape[: -len(normalized_shape)], dtype=int)
    a_4d = a.view(batch_size, elements_to_normalize, 1, 1)
    weight_4d = weight.view(1, elements_to_normalize, 1, 1)
    bias_4d = bias.view(1, elements_to_normalize, 1, 1)

    cudnn_input_args = [
        CudnnTensorAttributes(arg.size(), arg.stride(), arg.dtype) if isinstance(arg, torch.Tensor) else arg
        for arg in [a_4d, weight_4d, bias_4d]
    ]

    input, scale, B, epsilon, Y, graph = _make_cudnn_layer_norm_graph(*cudnn_input_args)

    Y_actual = torch.zeros_like(a, device="cuda")

    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    cudnn_to_torch_tensor = {input: a, scale: weight, B: bias, epsilon: epsilon_cpu, Y: Y_actual}

    graph.execute(cudnn_to_torch_tensor, workspace)

    return Y_actual


def layer_norm_checker(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    if cudnn is None:
        return False

    # TODO: the checker needs to call cudnn's check_support API
    # Just returning the output of check_support is sufficient to cover all
    # cudnn support across versions.
    # Below checks only exist as a workaround until check_support is implemented.

    if weight is None or bias is None:
        return False

    return True


_op_to_cudnn = {
    "torch.nn.functional.scaled_dot_product_attention": ("cudnn_sdpa", sdpa_checker, sdpa_impl),
    "torch.layer_norm": ("cudnn_layer_norm", layer_norm_checker, layer_norm_impl),
}


def register_cudnnex(*, add_to_default_executors: bool = True) -> None:
    assert CUDNN_AVAILABLE, f"Trying to register the cudnn executor, but the cudnn package is not available"

    print(
        "WARNING: cudnn executor is experimental. Tests that use cudnn might fail.\n"
        "Issue for tracking support: https://github.com/Lightning-AI/lightning-thunder/issues/880"
    )

    from thunder.executors import add_operator_executor

    add_operator_executor("cudnn", _op_to_cudnn, add_to_default_executors=add_to_default_executors)


def deregister_cudnnex() -> None:
    from thunder.executors import remove_operator_executor

    remove_operator_executor("cudnn")
