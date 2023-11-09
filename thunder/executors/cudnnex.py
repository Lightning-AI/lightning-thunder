from typing import Any

import torch
import numpy as np

from lightning_utilities.core.imports import package_available

CUDNN_AVAILABLE = package_available("cudnn")

cudnn: None | Any = None
cudnn_backend_version: None | Any = None
if CUDNN_AVAILABLE:
    import cudnn

    cudnn_backend_version = cudnn.backend_version()


def cudnn_available() -> bool:
    return CUDNN_AVAILABLE


# WARNING: cudnn executor is experimental. Tests that use cudnn might fail.\n
# Issue for tracking support: https://github.com/Lightning-AI/lightning-thunder/issues/880~

from dataclasses import dataclass
from functools import lru_cache
from typing import Union, Dict

import thunder.core.dtypes as dtypes
from thunder.core.proxies import TensorProxy

from thunder.extend import OperatorExecutor, register_executor

cudnn_ex: OperatorExecutor = OperatorExecutor("cudnn", version=cudnn_backend_version)
register_executor(cudnn_ex)


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

    # TODO: update to do tensor.stride_order when available from FE
    _, h, s_q, _ = query.size
    _, _, _, d_v = value.size
    stride_o = (h * s_q * d_v, s_q * d_v, d_v, 1)
    O.set_output(True).set_data_type(torch_to_cudnn_dtype(value.dtype)).set_stride(stride_o)

    graph.build([cudnn.heur_mode.A])

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    if Seed is not None:
        seed_device_tensor = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
    else:
        seed_device_tensor = None

    if Offset is not None:
        offset_device_tensor = torch.full((1, 1, 1, 1), 1, dtype=torch.int64, device="cuda")
    else:
        offset_device_tensor = None

    return Q, K, V, Attn_scale, Bias, Seed, seed_device_tensor, Offset, offset_device_tensor, O, workspace, graph


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

    key_4d = CudnnTensorAttributes(key.shape, compute_NHWC_strides(key.shape), key.dtype)

    value_4d = CudnnTensorAttributes(value.shape, compute_NHWC_strides(value.shape), value.dtype)

    attn_mask_4d = None
    if attn_mask is not None:
        attn_mask_shape = (1, 1, *attn_mask.shape)
        attn_mask_4d = CudnnTensorAttributes(attn_mask_shape, compute_NHWC_strides(attn_mask_shape), attn_mask.dtype)

    return query_4d, key_4d, value_4d, attn_mask_4d


def sdpa_impl(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    query_4d, key_4d, value_4d, attn_mask_4d = _transform_sdpa_inputs(query, key, value, attn_mask)

    Q, K, V, Attn_scale, Bias, Seed, seed_tensor, Offset, offset_tensor, O, workspace, graph = _make_cudnn_sdpa_graph(
        query_4d, key_4d, value_4d, attn_mask_4d, dropout_p, is_causal
    )

    b, h, s_q, d_q = query.size()
    _, _, _, d_v = value.size()
    O_actual = torch.empty(b, h, s_q, d_v, dtype=value.dtype, device="cuda")

    # Default value of scale, if not provided, in all torch versions
    if scale is None:
        scale = 1 / d_q**0.5
    Attn_scale_cpu = torch.full((1, 1, 1, 1), scale, dtype=torch.float32, device="cpu")

    cudnn_to_torch_tensor = {Q: query, K: key, V: value, Attn_scale: Attn_scale_cpu, O: O_actual}

    if Bias is not None:
        cudnn_to_torch_tensor[Bias] = attn_mask

    if Seed is not None:
        cudnn_to_torch_tensor[Seed] = seed_tensor

    if Offset is not None:
        cudnn_to_torch_tensor[Offset] = offset_tensor

    graph.execute(cudnn_to_torch_tensor, workspace)

    return O_actual


def sdpa_checker(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if cudnn is None:
        return False

    query_4d, key_4d, value_4d, attn_mask_4d = _transform_sdpa_inputs(query, key, value, attn_mask)

    try:
        _make_cudnn_sdpa_graph(query_4d, key_4d, value_4d, attn_mask_4d, dropout_p, is_causal)
    except Exception as e:
        return False

    # Bug in cudnn 8.9.5 and earlier where embedding dim support is missing
    _, _, _, d_q = query.size()
    _, _, _, d_kv = value.size()
    for d in [d_q, d_kv]:
        if d % 8 != 0 or d > 128:
            return False

    return True


import thunder.torch as ltorch

sdpa = cudnn_ex.register_operator("cudnn_sdpa", like=ltorch.scaled_dot_product_attention, fn=sdpa_impl)
cudnn_ex.register_implementation(ltorch.scaled_dot_product_attention, sdpa, checker=sdpa_checker)
