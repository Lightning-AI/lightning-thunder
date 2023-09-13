from typing import Any

import torch
import numpy as np

from lightning_utilities.core.imports import package_available

CUDNN_AVAILABLE = package_available("cudnn")

cudnn: None | Any = None
if CUDNN_AVAILABLE:
    import cudnn

    cudnn_backend_version = cudnn.backend_version()

# WARNING: cudnn executor is experimental. Tests that use cudnn might fail.\n
# Issue for tracking support: https://github.com/Lightning-AI/lightning-thunder/issues/880~

from dataclasses import dataclass
from functools import lru_cache
from typing import Union, Dict

import thunder.core.dtypes as dtypes
from thunder.core.proxies import TensorProxy


@dataclass(frozen=True)
class CudnnTensorAttributes:
    size: tuple
    stride: tuple
    dtype: torch.dtype


def make_cacheable_cudnn_graph_inputs(func):
    def wrapper(*args, **kwargs):
        cudnn_input_args = [
            CudnnTensorAttributes(arg.size(), arg.stride(), arg.dtype) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        return func(*cudnn_input_args, **kwargs)

    return wrapper


@make_cacheable_cudnn_graph_inputs
@lru_cache(maxsize=1024)
def _make_cudnn_sdpa_graph(query, key, value, attn_mask, dropout_p, is_causal):
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    Q = graph.tensor(name="Q", dim=query.size, stride=query.stride, data_type=torch_to_cudnn_dtype(query.dtype))
    K = graph.tensor(name="K", dim=key.size, stride=key.stride, data_type=torch_to_cudnn_dtype(key.dtype))
    V = graph.tensor(name="V", dim=value.size, stride=value.stride, data_type=torch_to_cudnn_dtype(value.dtype))

    Bias = None
    if attn_mask is not None:
        Bias = graph.tensor(
            name="bias", dim=attn_mask.size, stride=attn_mask.stride, data_type=torch_to_cudnn_dtype(attn_mask.dtype)
        )

    scalar_dim_stride = tuple([1] * len(query.size))
    dropout_tuple = None
    Seed = None
    Offset = None
    if dropout_p != 0.0:
        Seed = graph.tensor(
            name="Seed", dim=scalar_dim_stride, stride=scalar_dim_stride, data_type=cudnn.data_type.INT64
        )
        Offset = graph.tensor(
            name="Offset", dim=scalar_dim_stride, stride=scalar_dim_stride, data_type=cudnn.data_type.INT64
        )
        dropout_tuple = (dropout_p, Seed, Offset)

    Attn_scale = graph.tensor(
        name="Attn_scale",
        dim=scalar_dim_stride,
        stride=scalar_dim_stride,
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
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
        dropout=dropout_tuple,
    )

    O.set_output(True).set_data_type(torch_to_cudnn_dtype(value.dtype))

    graph.check_support()
    graph.build()

    return Q, K, V, Attn_scale, Bias, Seed, Offset, O, graph


@make_cacheable_cudnn_graph_inputs
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
        dtypes.float16: cudnn.data_type.HALF,
        dtypes.bfloat16: cudnn.data_type.BFLOAT16,
        dtypes.float32: cudnn.data_type.FLOAT,
        None: cudnn.data_type.NOT_SET,
    }
    return _torch_to_cudnn_dtype_map[lc_dtype]


def _transform_sdpa_inputs(query, key, value, attn_mask):
    def compute_NHWC_strides(shape):
        strides = [1] * len(shape)
        stride = 1
        for i in reversed(range(len(shape))):
            strides[i] = stride
            stride *= shape[i]
        return tuple(strides)

    query_4d = CudnnTensorAttributes(query.shape, compute_NHWC_strides(query.shape), query.dtype)

    # Cudnn does not do a transpose operation for key
    strides_4d = compute_NHWC_strides(key.shape)
    shape_4d_T = (*key.shape[:-2], key.shape[-1], key.shape[-2])
    strides_4d_T = (*strides_4d[:-2], strides_4d[-1], strides_4d[-2])
    key_4d_T = CudnnTensorAttributes(shape_4d_T, strides_4d_T, key.dtype)

    value_4d = CudnnTensorAttributes(value.shape, compute_NHWC_strides(value.shape), value.dtype)

    attn_mask_4d = None
    if attn_mask is not None:
        attn_mask_shape = (1, 1, *attn_mask.shape)
        attn_mask_4d = CudnnTensorAttributes(attn_mask_shape, compute_NHWC_strides(attn_mask_shape), attn_mask.dtype)

    return query_4d, key_4d_T, value_4d, attn_mask_4d


def sdpa_impl(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    query_4d, key_4d_T, value_4d, attn_mask_4d = _transform_sdpa_inputs(query, key, value, attn_mask)

    Q, K, V, Attn_scale, Bias, Seed, Offset, O, graph = _make_cudnn_sdpa_graph(
        query_4d, key_4d_T, value_4d, attn_mask_4d, dropout_p, is_causal
    )

    b, h, s_q, d_q = query.size()
    _, _, _, d_v = value.size()
    O_actual = torch.zeros(b, h, s_q, d_v, dtype=value.dtype, device="cuda")

    # Default value of scale, if not provided, in all torch versions
    if scale is None:
        scale = 1 / d_q**0.5
    Attn_scale_cpu = torch.full((1, 1, 1, 1), scale, dtype=torch.float32, device="cpu")

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    cudnn_to_torch_tensor = {Q: query, K: key, V: value, Attn_scale: Attn_scale_cpu, O: O_actual}

    if Bias is not None:
        cudnn_to_torch_tensor[Bias] = attn_mask

    if Seed is not None:
        cudnn_to_torch_tensor[Seed] = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")

    if Offset is not None:
        cudnn_to_torch_tensor[Offset] = torch.full((1, 1, 1, 1), 1, dtype=torch.int64, device="cuda")

    graph.execute(cudnn_to_torch_tensor, workspace)

    return O_actual


def sdpa_checker(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if cudnn is None:
        return False

    query_4d, key_4d_T, value_4d, attn_mask_4d = _transform_sdpa_inputs(query, key, value, attn_mask)

    try:
        _make_cudnn_sdpa_graph(query_4d, key_4d_T, value_4d, attn_mask_4d, dropout_p, is_causal)
    except Exception as e:
        return False

    # Bug in cudnn 8.9.5 and earlier where embedding dim support is missing
    _, _, _, d_q = query.size()
    _, _, _, d_kv = value.size()
    for d in [d_q, d_kv]:
        if d % 8 != 0 or d > 128:
            return False

    return True


# cudnn only supports following:
# input tensor shape: N, C, (D), H, W
# normalized shape:  (C, (D), H, W)
# convert all tensor shapes to above format
def _transform_layer_norm_inputs(a, normalized_shape, weight, bias):
    elements_to_normalize = np.prod(normalized_shape)
    batch_size = np.prod(a.shape[: -len(normalized_shape)], dtype=int)

    # Assume strides to be NCHW contiguous
    assumed_stride = (elements_to_normalize, 1, 1, 1)
    a_4d = CudnnTensorAttributes((batch_size, elements_to_normalize, 1, 1), assumed_stride, a.dtype)
    weight_4d = CudnnTensorAttributes((1, elements_to_normalize, 1, 1), assumed_stride, weight.dtype)
    bias_4d = CudnnTensorAttributes((1, elements_to_normalize, 1, 1), assumed_stride, bias.dtype)

    return a_4d, weight_4d, bias_4d


def layer_norm_impl(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    a_4d, weight_4d, bias_4d = _transform_layer_norm_inputs(a, normalized_shape, weight, bias)
    input, scale, B, epsilon, Y, graph = _make_cudnn_layer_norm_graph(a_4d, weight_4d, bias_4d)

    Y_actual = torch.zeros_like(a, device="cuda")

    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    cudnn_to_torch_tensor = {input: a, scale: weight, B: bias, epsilon: epsilon_cpu, Y: Y_actual}

    graph.execute(cudnn_to_torch_tensor, workspace)

    return Y_actual


def layer_norm_checker(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    if cudnn is None:
        return False

    a_4d, weight_4d, bias_4d = _transform_layer_norm_inputs(a, normalized_shape, weight, bias)

    try:
        _make_cudnn_layer_norm_graph(a_4d, weight_4d, bias_4d)
    except:
        return False

    return True


_op_to_cudnn = {
    "torch.nn.functional.scaled_dot_product_attention": ("cudnn_sdpa", sdpa_checker, sdpa_impl),
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
