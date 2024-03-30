from typing import Any

import torch
import numpy as np
import random

from lightning_utilities.core.imports import package_available


def cudnn_available() -> bool:
    return package_available("cudnn")


def cudnn_version() -> int:
    if cudnn_available():
        return cudnn.backend_version()
    return 0


cudnn: None | Any = None
cudnn_backend_version: None | Any = None
if cudnn_available():
    import cudnn

    cudnn_backend_version = cudnn.backend_version()
    # Mapping from device to cudnn handles
    device_to_cudnn_handle = {}


# This function creates a new handle for the device that cudnn should
# run its kernels on. As the suggested approach by cudnn is to make a few handles
# as possible, this function caches these per-device handles.
def _get_cudnn_handle(query_device):
    handle = device_to_cudnn_handle.get(query_device, None)
    if handle is None:
        with torch.cuda.device(query_device):
            handle = cudnn.create_handle()
            device_to_cudnn_handle[query_device] = handle

    # Make sure the user stream is set on the handle
    # Fetch the current user stream and pass the data pointer to set_stream API
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device=query_device).cuda_stream)

    return handle


# WARNING: cudnn executor is experimental. Tests that use cudnn might fail.\n
# Issue for tracking support: https://github.com/Lightning-AI/lightning-thunder/issues/880~

from dataclasses import dataclass
from functools import lru_cache
from typing import Union, Dict

from thunder.core.langctxs import langctx
import thunder.core.dtypes as dtypes
from thunder.torch import TensorLike
from thunder.core.compile_data import get_compile_option
from thunder.core.proxies import Proxy, TensorProxy


from thunder.core.transforms import (
    get_grad,
    put_grad,
    put_grads,
)
from thunder.extend import OperatorExecutor, register_executor
import thunder.torch as ltorch

cudnn_ex: OperatorExecutor = OperatorExecutor("cudnn", version=cudnn_backend_version)
register_executor(cudnn_ex)


@dataclass(frozen=True)
class CudnnTensorAttributes:
    size: tuple
    stride: tuple
    dtype: torch.dtype
    device_index: int


from collections import OrderedDict


# Cache already built cudnn graphs to save on runtime compilation overhead
class CudnnexLRUCache(OrderedDict):
    def __init__(self, maxlen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._maxlen = maxlen

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        elif len(self) == self._maxlen:
            oldest = next(iter(self))
            del self[oldest]
        super().__setitem__(key, value)


_cudnnex_cache = CudnnexLRUCache(maxlen=1024)


def _make_cudnn_sdpa_forward_graph(query, key, value, attn_mask, dropout_p, is_causal):
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(query.device_index),
    )

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
            name="Seed", dim=scalar_dim_stride, stride=scalar_dim_stride, data_type=cudnn.data_type.INT32
        )
        Offset = graph.tensor(
            name="Offset", dim=scalar_dim_stride, stride=scalar_dim_stride, data_type=cudnn.data_type.INT32
        )
        dropout_tuple = (dropout_p, Seed, Offset)

    Attn_scale = graph.tensor(
        name="Attn_scale",
        dim=scalar_dim_stride,
        stride=scalar_dim_stride,
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    O, softmax_stats = graph.scaled_dot_product_flash_attention(
        name="scaled_dot_product_flash_attention",
        q=Q,
        k=K,
        v=V,
        is_inference=False,
        bias=Bias,
        use_causal_mask=is_causal,
        attn_scale=Attn_scale,
        dropout=dropout_tuple,
    )

    # TODO: update to do tensor.stride_order when available from FE
    b, h, s_q, _ = query.size
    _, _, _, d_v = value.size

    dim_o = (b, h, s_q, d_v)
    stride_o = (h * s_q * d_v, s_q * d_v, d_v, 1)
    O.set_output(True).set_data_type(torch_to_cudnn_dtype(value.dtype)).set_dim(dim_o).set_stride(stride_o)

    softmax_stats.set_output(True).set_data_type(torch_to_cudnn_dtype(torch.float32))

    # Validate the graph before querying the cache key
    # Validation makes sure all missing properties are inferred and filled, as they affect cache key.
    graph.validate()
    cache_key = graph.key()

    # If a built graph does not exist in cache already, make one and place it in
    if cache_key not in _cudnnex_cache:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        _cudnnex_cache[cache_key] = (
            Q,
            K,
            V,
            Attn_scale,
            Bias,
            Seed,
            Offset,
            O,
            softmax_stats,
            graph,
        )
    return _cudnnex_cache[cache_key]


def torch_to_cudnn_dtype(lc_dtype: dtypes.dtype):
    _torch_to_cudnn_dtype_map: dict[None | torch.dtype, cudnn.data_type] = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
        dtypes.bool8: cudnn.data_type.BOOLEAN,
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

    query_4d = CudnnTensorAttributes(query.shape, compute_NHWC_strides(query.shape), query.dtype, query.device.index)

    key_4d = CudnnTensorAttributes(key.shape, compute_NHWC_strides(key.shape), key.dtype, key.device.index)

    value_4d = CudnnTensorAttributes(value.shape, compute_NHWC_strides(value.shape), value.dtype, value.device.index)

    attn_mask_4d = None
    if attn_mask is not None:
        # Make attn_mask to be of the same dimensionality as other input tensors
        attn_mask_shape = (1,) * (query.ndim - attn_mask.ndim) + attn_mask.shape

        # cudnn does not support boolean attn_mask, so make one with -inf
        attn_mask_dtype = query.dtype if attn_mask.dtype in [torch.bool, dtypes.bool8] else attn_mask.dtype
        attn_mask_4d = CudnnTensorAttributes(
            attn_mask_shape, compute_NHWC_strides(attn_mask_shape), attn_mask_dtype, attn_mask.device.index
        )

    return query_4d, key_4d, value_4d, attn_mask_4d


# sdpa requires that the embedding dim stride be one.
# And when registering for sdpa, cudnn assumes NHWC layout. (See _transform_sdpa_inputs())
def _sdpa_enforce_input_tensor_contiguity(a: torch.Tensor) -> torch.Tensor:
    if a.stride(-1) == 1:
        # TODO(vedaanta-nvidia): there's an inconsistency between
        # _transform_sdpa_inputs and this function, leading to a potential bug.
        # _transform_sdpa_inputs always creates contiguous strides, but code
        # here creates a partially contiguous stride when the last dimension is
        # contiguous but other dimensions are not.
        return a
    else:
        return a.contiguous()


def _cudnn_sdpa_forward_meta(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: TensorLike | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> tuple[TensorProxy, TensorProxy, TensorProxy, TensorProxy]:
    batch_size, num_heads, query_seq_len, E = query.shape
    key_seq_len = key.shape[-2]
    Ev = value.shape[-1]

    return (
        output := TensorProxy(like=query, shape=(batch_size, num_heads, query_seq_len, Ev)),
        softmax_stats := TensorProxy(
            shape=(batch_size, num_heads, query_seq_len, 1),
            dtype=dtypes.float32,
            device=query.device,
            requires_grad=False,
        ),
        seed := TensorProxy(shape=(1, 1, 1, 1), dtype=dtypes.int32, device=query.device, requires_grad=False),
        offset := TensorProxy(shape=(1, 1, 1, 1), dtype=dtypes.int32, device=query.device, requires_grad=False),
    )


def _cudnn_sdpa_fwd_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    query_4d, key_4d, value_4d, attn_mask_4d = _transform_sdpa_inputs(query, key, value, attn_mask)

    (
        Q,
        K,
        V,
        Attn_scale,
        Bias,
        Seed,
        Offset,
        O,
        softmax_stats,
        graph,
    ) = _make_cudnn_sdpa_forward_graph(query_4d, key_4d, value_4d, attn_mask_4d, dropout_p, is_causal)

    b, h, s_q, d_q = query.size()
    _, _, _, d_v = value.size()
    O_actual = torch.empty(b, h, s_q, d_v, dtype=value.dtype, device=query.device)
    softmax_stats_actual = torch.empty(b, h, s_q, 1, dtype=torch.float32, device=query.device)
    workspace = torch.empty(graph.get_workspace_size(), device=query.device, dtype=torch.uint8)

    seed_tensor = (
        torch.full((1, 1, 1, 1), random.randint(0, 123902390), dtype=torch.int32, device=query.device) if Seed else None
    )
    offset_tensor = (
        torch.full((1, 1, 1, 1), random.randint(0, 123902390), dtype=torch.int32, device=query.device)
        if Offset
        else None
    )

    # Default value of scale, if not provided, in all torch versions
    if scale is None:
        scale = 1 / d_q**0.5
    Attn_scale_cpu = torch.full((1, 1, 1, 1), scale, dtype=torch.float32, device="cpu")

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        attn_mask = attn_bias

    cudnn_to_torch_tensor = {
        Q: _sdpa_enforce_input_tensor_contiguity(query).detach(),
        K: _sdpa_enforce_input_tensor_contiguity(key).detach(),
        V: _sdpa_enforce_input_tensor_contiguity(value).detach(),
        Attn_scale: Attn_scale_cpu,
        Seed: seed_tensor,
        Offset: offset_tensor,
        O: O_actual,
        softmax_stats: softmax_stats_actual,
    }
    if attn_mask is not None:
        cudnn_to_torch_tensor[Bias] = attn_mask.detach()

    # Even though the handle is created on query.device, cudnn still requires to set current device to query.device.
    # This is most probably a bug and is being actively looked into.
    with torch.cuda.device(query.device):
        graph.execute(cudnn_to_torch_tensor, workspace, handle=_get_cudnn_handle(query.device))

    return O_actual, softmax_stats_actual, seed_tensor, offset_tensor


def _cudnn_sdpa_checker(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: TensorLike | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> bool:
    # TODO(#58): make the checker more conservative.
    if cudnn is None:
        return False

    if len(query.size()) != 4:
        return False
    _, _, _, d_q = query.size()

    if len(value.size()) != 4:
        return False
    _, _, _, d_kv = value.size()

    # Bug in cudnn 8.9.5 and earlier where embedding dim support is missing
    for d in [d_q, d_kv]:
        if d % 8 != 0 or d > 128:
            return False

    return True


cudnn_sdpa_fwd = cudnn_ex.register_operator(
    "cudnn_sdpa_fwd",
    meta=_cudnn_sdpa_forward_meta,
    fn=_cudnn_sdpa_fwd_impl,
)


def _make_cudnn_sdpa_backward_graph(
    query, key, value, attn_mask, dropout_p, is_causal, grad_query_stride, grad_key_stride, grad_value_stride
):
    b, h, s_q, _ = query.size
    _, _, _, d_v = value.size

    # cuDNN < 9.0.0 might produce nan gradients for sequence length < 64
    assert s_q >= 64, "CUDNN SDPA requires sequence length to be at least 64 for backward pass"

    graph = cudnn.pygraph(
        io_data_type=torch_to_cudnn_dtype(query.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(query.device_index),
    )

    Q = graph.tensor(name="Q", dim=query.size, stride=query.stride, data_type=torch_to_cudnn_dtype(query.dtype))
    K = graph.tensor(name="K", dim=key.size, stride=key.stride, data_type=torch_to_cudnn_dtype(key.dtype))
    V = graph.tensor(name="V", dim=value.size, stride=value.stride, data_type=torch_to_cudnn_dtype(value.dtype))

    dim_o = (b, h, s_q, d_v)
    stride_o = (h * s_q * d_v, s_q * d_v, d_v, 1)
    O = graph.tensor(name="O", dim=dim_o, stride=stride_o, data_type=torch_to_cudnn_dtype(query.dtype))
    dO = graph.tensor_like(O)

    dim_stats = (b, h, s_q, 1)
    stride_stats = (h * s_q, s_q, 1, 1)
    Stats = graph.tensor(name="Stats", dim=dim_stats, stride=stride_stats, data_type=cudnn.data_type.FLOAT)

    Bias = None
    dBias = None
    if attn_mask is not None:
        Bias = graph.tensor(
            name="bias", dim=attn_mask.size, stride=attn_mask.stride, data_type=torch_to_cudnn_dtype(attn_mask.dtype)
        )
        dBias = graph.tensor_like(Bias)

    scalar_dim_stride = tuple([1] * len(query.size))
    dropout_tuple = None
    Seed = None
    Offset = None
    if dropout_p != 0.0:
        Seed = graph.tensor(
            name="Seed", dim=scalar_dim_stride, stride=scalar_dim_stride, data_type=cudnn.data_type.INT32
        )
        Offset = graph.tensor(
            name="Offset", dim=scalar_dim_stride, stride=scalar_dim_stride, data_type=cudnn.data_type.INT32
        )
        dropout_tuple = (dropout_p, Seed, Offset)

    Attn_scale = graph.tensor(
        name="Attn_scale",
        dim=scalar_dim_stride,
        stride=scalar_dim_stride,
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    dQ, dK, dV = graph.scaled_dot_product_flash_attention_backward(
        q=Q,
        k=K,
        v=V,
        o=O,
        dO=dO,
        stats=Stats,
        attn_scale=Attn_scale,
        bias=Bias,
        dBias=dBias,
        use_causal_mask=is_causal,
        dropout=dropout_tuple,
    )

    dQ.set_output(True).set_dim(query.size).set_stride(grad_query_stride).set_data_type(
        torch_to_cudnn_dtype(query.dtype)
    )
    dK.set_output(True).set_dim(key.size).set_stride(grad_key_stride).set_data_type(torch_to_cudnn_dtype(key.dtype))
    dV.set_output(True).set_dim(value.size).set_stride(grad_value_stride).set_data_type(
        torch_to_cudnn_dtype(value.dtype)
    )

    # Validate the graph before querying the cache key
    # Validation makes sure all missing properties are inferred and filled, as they affect cache key.
    graph.validate()
    cache_key = graph.key()

    # If a built graph does not exist in cache already, make one and place it in
    if cache_key not in _cudnnex_cache:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

        _cudnnex_cache[cache_key] = (
            Q,
            K,
            V,
            O,
            dO,
            Stats,
            Seed,
            Offset,
            Attn_scale,
            Bias,
            dQ,
            dK,
            dV,
            dBias,
            graph,
        )
    return _cudnnex_cache[cache_key]


def _replace_dim_with(size: torch.Size, dim: int, dim_size: int) -> torch.Size:
    return torch.Size(size[:dim] + (dim_size,) + size[dim + 1 :])


def _cudnn_sdpa_bwd_meta(
    grad_out: TensorLike,
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: None | TensorProxy,
    dropout_p: float,
    is_causal: bool,
    out: TensorLike,
    softmax_stats: TensorLike,
    philox_seed: TensorLike,
    philox_offset: TensorLike,
    *,
    scale: None | float = None,
    cat_grad_qkv: bool,
) -> tuple[TensorProxy, ...]:
    if cat_grad_qkv:
        grad_qkv = TensorProxy(
            like=query, shape=_replace_dim_with(query.size(), 1, query.size(1) + key.size(1) + value.size(1))
        )
        grads = (grad_qkv,)
    else:
        grad_query = TensorProxy(like=query)
        grad_key = TensorProxy(like=key)
        grad_value = TensorProxy(like=value)
        grads = (grad_query, grad_key, grad_value)

    if attn_mask is not None:
        grad_attn_mask = TensorProxy(like=attn_mask)
        grads = grads + (grad_attn_mask,)

    return grads


def _same_size_except(*args, except_dim: int) -> bool:
    shapes = [_replace_dim_with(shape, except_dim, 0) for shape in args]
    return all(shape == shapes[0] for shape in shapes)


# Allocates an empty tensor that will hold dQ, dK, and dV, concatenated.
# `query`, `key` and `value` merely provide necessary metadata such as sizes
# and dtypes. They don't have to be passed in as `torch.Tensor`s.
def _allocate_catted_grad_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    assert _same_size_except(query.size(), key.size(), value.size(), except_dim=1)
    assert query.dtype == key.dtype == value.dtype
    assert query.device == key.device == value.device

    b, s, d = query.size(0), query.size(2), query.size(3)
    h_q, h_k, h_v = query.size(1), key.size(1), value.size(1)
    h_qkv = h_q + h_k + h_v

    # Create grad_qkv as a tensor of size [b,h_qkv,s,d] and allocation order
    # [0,2,1,3] from major to minor.
    return torch.empty(b, s, h_qkv, d, dtype=query.dtype, device=query.device).permute(0, 2, 1, 3)


def _cudnn_sdpa_bwd_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: None | torch.Tensor,
    dropout_p: float,
    is_causal: bool,
    out: torch.Tensor,
    softmax_stats: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: None | float = None,
    cat_grad_qkv: bool,
) -> tuple[torch.Tensor, ...]:
    query_4d, key_4d, value_4d, attn_mask_4d = _transform_sdpa_inputs(query, key, value, attn_mask)
    query = _sdpa_enforce_input_tensor_contiguity(query)
    key = _sdpa_enforce_input_tensor_contiguity(key)
    value = _sdpa_enforce_input_tensor_contiguity(value)

    # When cat_grad_qkv is on, allocate dQKV and make dQ, dK, and dV
    # slices of that. Otherwise, allocate them individually.
    grad_qkv: None | torch.Tensor = None
    if cat_grad_qkv:
        grad_qkv = _allocate_catted_grad_qkv(query, key, value)
        grad_query, grad_key, grad_value = grad_qkv.split([query.size(1), key.size(1), value.size(1)], dim=1)
    else:
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)

    (
        Q,
        K,
        V,
        O,
        dO,
        Stats,
        Seed,
        Offset,
        Attn_scale,
        Bias,
        dQ,
        dK,
        dV,
        dBias,
        graph,
    ) = _make_cudnn_sdpa_backward_graph(
        query_4d,
        key_4d,
        value_4d,
        attn_mask_4d,
        dropout_p,
        is_causal,
        grad_query.stride(),
        grad_key.stride(),
        grad_value.stride(),
    )

    # Default value of scale, if not provided, in all torch versions
    if scale is None:
        scale = query.shape[-1] ** -0.5
    Attn_scale_cpu = torch.full((1, 1, 1, 1), scale, dtype=torch.float32, device="cpu")

    cudnn_to_torch_tensor = {
        dO: grad_out.detach(),
        Q: query.detach(),
        K: key.detach(),
        V: value.detach(),
        Attn_scale: Attn_scale_cpu,
        O: out.detach(),
        Stats: softmax_stats,
        Seed: philox_seed,
        Offset: philox_offset,
        dQ: grad_query,
        dK: grad_key,
        dV: grad_value,
    }
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attn_mask = attn_bias

        grad_attn_mask = torch.empty_like(attn_mask) if attn_mask is not None else None

        cudnn_to_torch_tensor[Bias] = attn_mask.detach()
        cudnn_to_torch_tensor[dBias] = grad_attn_mask

    workspace = torch.empty(graph.get_workspace_size(), device=query.device, dtype=torch.uint8)

    # Even though the handle is created on query.device, cudnn still requires to set current device to query.device.
    # This is most probably a bug and is being actively looked into.
    with torch.cuda.device(query.device):
        graph.execute(cudnn_to_torch_tensor, workspace, handle=_get_cudnn_handle(query.device))

    if cat_grad_qkv:
        grads = (grad_qkv,)
    else:
        grads = (grad_query, grad_key, grad_value)

    if attn_mask is not None:
        grads = grads + (grad_attn_mask,)
    return grads


cudnn_sdpa_bwd = cudnn_ex.register_operator(
    "cudnn_sdpa_bwd",
    meta=_cudnn_sdpa_bwd_meta,
    fn=_cudnn_sdpa_bwd_impl,
)


@langctx("torch")
def _cudnn_sdpa_fwd_wrapper(
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask: None | TensorProxy = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> TensorProxy:
    output, _, _, _ = cudnn_sdpa_fwd(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)

    return output


@langctx("torch")
def _cudnn_sdpa_bwd_wrapper(
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask: None | TensorProxy,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
):
    primal, softmax_stats, seed, offset = cudnn_sdpa_fwd(
        query, key, value, attn_mask, dropout_p, is_causal, scale=scale
    )

    description = """\
This flag is for enabling nvFuser's zipping optimization that seeks to avoid
expensive concatenation. https://github.com/NVIDIA/Fuser/issues/1768 has more
details. When this flag is true, cudnn_sdpa_bwd may cat dQ, dK and dV as one
tensor and return them as slices of that tensor.
"""
    may_cat_grad_qkv: None | bool = get_compile_option("cudnn_sdpa_bwd_may_cat_grad_qkv", description)
    if may_cat_grad_qkv is None:
        may_cat_grad_qkv = False
    assert isinstance(may_cat_grad_qkv, bool)
    cat_grad_qkv = may_cat_grad_qkv and _same_size_except(query.size(), key.size(), value.size(), except_dim=1)

    grads = cudnn_sdpa_bwd(
        get_grad(primal),
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        primal,
        softmax_stats,
        seed,
        offset,
        scale=scale,
        cat_grad_qkv=cat_grad_qkv,
    )

    if attn_mask is not None:
        grad_attn_mask = grads[-1]
        grads = grads[:-1]
        put_grad(attn_mask, grad_attn_mask)

    if cat_grad_qkv:
        # The `split` is done outside `cudnn_sdpa_bwd` so it can be picked up
        # by nvfuserex.
        (grad_qkv,) = grads
        grad_query, grad_key, grad_value = grad_qkv.split([query.size(1), key.size(1), value.size(1)], dim=1)
    else:
        grad_query, grad_key, grad_value = grads
    put_grads((query, key, value), (grad_query, grad_key, grad_value))

    return primal


# Registers the implementation for torch.nn.functional.scaled_dot_product_attention
cudnn_ex.register_implementation(
    ltorch.scaled_dot_product_attention,
    checker=_cudnn_sdpa_checker,
    execution_transform=_cudnn_sdpa_fwd_wrapper,
    grad_transform=_cudnn_sdpa_bwd_wrapper,
)
