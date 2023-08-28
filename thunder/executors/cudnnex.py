from typing import Any

import torch

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
def _make_cudnn_graph(query, key, value, attn_mask, is_causal):
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

    Q, K, V, Attn_scale, Bias, O, graph = _make_cudnn_graph(*cudnn_input_args)

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

    if query.ndim != 4:
        return False

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        return False

    return True


_op_to_cudnn = {
    "torch.nn.functional.scaled_dot_product_attention": ("cudnn_sdpa", sdpa_checker, sdpa_impl),
}


def register_cudnnex(*, add_to_default_executors: bool = True) -> None:
    assert CUDNN_AVAILABLE, f"Trying to register the cudnn executor, but the cudnn package is not available"

    print("WARNING: cudnn executor is experimental. Tests that use cudnn might fail.\n"
          "Issue for tracking support: https://github.com/Lightning-AI/lightning-thunder/issues/880")

    from thunder.executors import add_operator_executor

    add_operator_executor("cudnn", _op_to_cudnn, add_to_default_executors=add_to_default_executors)


def deregister_cudnnex() -> None:
    from thunder.executors import remove_operator_executor

    remove_operator_executor("cudnn")
