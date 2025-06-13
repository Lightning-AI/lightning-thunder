from collections import OrderedDict
from itertools import accumulate
import random
import torch

from thunder.torch import TensorLike
from thunder.core.proxies import TensorProxy
from thunder.core.compile_data import get_compile_option
from thunder.core.prims import OpTags
from thunder.core.transforms import get_grad, put_grad, put_grads
import thunder.core.dtypes as dtypes
import thunder.torch as ltorch
from thunder.core.proxies import pyval

from thunder.extend import OperatorExecutor, register_executor

import cudnn

cudnn_backend_version = cudnn.backend_version()

cudnn_ex: OperatorExecutor = OperatorExecutor("cudnn", version=cudnn_backend_version)
register_executor(cudnn_ex)


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
# Mapping from device to cudnn handles
_device_to_cudnn_handle = {}


# This function creates a new handle for the device that cudnn should
# run its kernels on. As the suggested approach by cudnn is to make a few handles
# as possible, this function caches these per-device handles.
def _get_cudnn_handle(query_device):
    handle = _device_to_cudnn_handle.get(query_device, None)
    if handle is None:
        with torch.cuda.device(query_device):
            handle = cudnn.create_handle()
            _device_to_cudnn_handle[query_device] = handle

    # Make sure the user stream is set on the handle
    # Fetch the current user stream and pass the data pointer to set_stream API
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device=query_device).cuda_stream)

    return handle

    handle = _device_to_cudnn_handle.get(query_device, None)
    if handle is None:
        with torch.cuda.device(query_device):
            handle = cudnn.create_handle()
            _device_to_cudnn_handle[query_device] = handle
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device=query_device).cuda_stream)
    return handle


def _make_cudnn_sdpa_forward_graph(
    query, key, value, attn_mask, dropout_p, is_causal, query_stride, key_stride, value_stride
):
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(query.device.index),
    )

    Q = graph.tensor(name="Q", dim=query.shape, stride=query_stride, data_type=torch_to_cudnn_dtype(query.dtype))
    K = graph.tensor(name="K", dim=key.shape, stride=key_stride, data_type=torch_to_cudnn_dtype(key.dtype))
    V = graph.tensor(name="V", dim=value.shape, stride=value_stride, data_type=torch_to_cudnn_dtype(value.dtype))
    Bias = None
    if attn_mask is not None:
        attn_mask_stride = _compute_row_major_strides(attn_mask.shape)
        Bias = graph.tensor(
            name="bias", dim=attn_mask.shape, stride=attn_mask_stride, data_type=torch_to_cudnn_dtype(attn_mask.dtype)
        )

    scalar_dim_stride = tuple([1] * len(query.shape))
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
    b, h, s_q, _ = query.shape
    _, _, _, d_v = value.shape

    dim_o = (b, h, s_q, d_v)
    stride_o = (h * s_q * d_v, s_q * d_v, d_v, 1)
    O.set_output(True).set_data_type(torch_to_cudnn_dtype(value.dtype)).set_dim(dim_o).set_stride(stride_o)

    softmax_stats.set_output(True).set_data_type(torch_to_cudnn_dtype(torch.float32))

    cache_key = graph.key()
    # If a built graph does not exist in cache already, make one and place it in
    if cache_key not in _cudnnex_cache:
        graph.build([cudnn.heur_mode.A])

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
    import cudnn

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


def _compute_row_major_strides(shape):
    """
    Compute contiguous strides for a row-major layout tensor of the given shape.

    Args:
        shape: The shape of the tensor.

    Returns:
        A tuple of strides for the tensor.
    """
    initial = 1 if shape else None
    return tuple(accumulate(reversed(shape[1:]), lambda x, y: x * y, initial=initial))[::-1]


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
    query = _sdpa_enforce_input_tensor_contiguity(query)
    key = _sdpa_enforce_input_tensor_contiguity(key)
    value = _sdpa_enforce_input_tensor_contiguity(value)

    if attn_mask is not None:
        if query.ndim > attn_mask.ndim:
            attn_mask = attn_mask.view(*((1,) * (query.ndim - attn_mask.ndim)), *attn_mask.shape)
        # As cudnn does not support boolean attn_mask, convert these to additive mask with -inf
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attn_mask = attn_bias

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
    ) = _make_cudnn_sdpa_forward_graph(
        query, key, value, attn_mask, dropout_p, is_causal, query.stride(), key.stride(), value.stride()
    )

    b, h_q, s_q, d_q = query.size()
    _, _, _, d_v = value.size()
    O_actual = torch.empty(b, h_q, s_q, d_v, dtype=value.dtype, device=query.device)
    softmax_stats_actual = torch.empty(b, h_q, s_q, 1, dtype=torch.float32, device=query.device)
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

    cudnn_to_torch_tensor = {
        Q: query.detach(),
        K: key.detach(),
        V: value.detach(),
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
    if globals().get("cudnn", None) is None:
        return False

    if query.device.type != "cuda" or key.device != query.device or value.device != query.device:
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

    dropout_p = pyval(dropout_p)
    is_causal = pyval(is_causal)
    if scale is not None:
        scale = pyval(scale)
    try:
        # TensorProxy do not contain stride information, but cudnn graph requires them.
        # Assume row major layout for now. If the strides during execution are different, a new graph will be built.
        query_stride = _compute_row_major_strides(query.size())
        key_stride = _compute_row_major_strides(key.size())
        value_stride = _compute_row_major_strides(value.size())

        if attn_mask is not None:
            # Make attn_mask to be of the same dimensionality as other input tensors
            attn_mask_shape = (1,) * (query.ndim - attn_mask.ndim) + attn_mask.shape
            # cudnn does not support boolean attn_mask, so make it additive mask instead.
            # During execution, similar change to attn_mask buffer will be made, where all values of False will be replaced with -inf
            attn_mask_dtype = query.dtype if attn_mask.dtype in [torch.bool, dtypes.bool8] else attn_mask.dtype
            attn_mask = TensorProxy(like=attn_mask, shape=attn_mask_shape, dtype=attn_mask_dtype)

        # Build both forward and backward graphs
        _make_cudnn_sdpa_forward_graph(
            query, key, value, attn_mask, dropout_p, is_causal, query_stride, key_stride, value_stride
        )
        _make_cudnn_sdpa_backward_graph(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            query_stride,
            key_stride,
            value_stride,
            query_stride,
            key_stride,
            value_stride,  # Use the same strides as inputs for their respective grads
        )

    # Please turn on cudnn API logging for helpful messages that mention why the graph is not supported.
    # For cudnn backend logging, refer https://docs.nvidia.com/deeplearning/cudnn/latest/reference/troubleshooting.html
    # For cudnn frontend logging, refer https://github.com/NVIDIA/cudnn-frontend?tab=readme-ov-file#debugging
    except cudnn.cudnnGraphNotSupportedError:
        return False
    # Otherwise just raise the error.
    # These errors can be due to internal cudnn bugs, or user error.
    except Exception:
        raise

    return True


cudnn_sdpa_fwd = cudnn_ex.register_operator(
    "cudnn_sdpa_fwd",
    meta=_cudnn_sdpa_forward_meta,
    fn=_cudnn_sdpa_fwd_impl,
    tags=(OpTags.DONT_AUTO_RECOMPUTE_IN_BACKWARD,),
)


def _make_cudnn_sdpa_backward_graph(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    query_stride,
    key_stride,
    value_stride,
    grad_query_stride,
    grad_key_stride,
    grad_value_stride,
):
    b, h, s_q, _ = query.shape
    _, _, _, d_v = value.shape

    graph = cudnn.pygraph(
        io_data_type=torch_to_cudnn_dtype(query.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(query.device.index),
    )

    Q = graph.tensor(name="Q", dim=query.shape, stride=query_stride, data_type=torch_to_cudnn_dtype(query.dtype))
    K = graph.tensor(name="K", dim=key.shape, stride=key_stride, data_type=torch_to_cudnn_dtype(key.dtype))
    V = graph.tensor(name="V", dim=value.shape, stride=value_stride, data_type=torch_to_cudnn_dtype(value.dtype))

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
        attn_mask_stride = _compute_row_major_strides(attn_mask.shape)
        Bias = graph.tensor(
            name="bias", dim=attn_mask.shape, stride=attn_mask_stride, data_type=torch_to_cudnn_dtype(attn_mask.dtype)
        )
        dBias = graph.tensor_like(Bias)

    scalar_dim_stride = tuple([1] * len(query.shape))
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

    dQ.set_output(True).set_dim(query.shape).set_stride(grad_query_stride).set_data_type(
        torch_to_cudnn_dtype(query.dtype)
    )
    dK.set_output(True).set_dim(key.shape).set_stride(grad_key_stride).set_data_type(torch_to_cudnn_dtype(key.dtype))
    dV.set_output(True).set_dim(value.shape).set_stride(grad_value_stride).set_data_type(
        torch_to_cudnn_dtype(value.dtype)
    )

    cache_key = graph.key()
    # If a built graph does not exist in cache already, make one and place it in
    if cache_key not in _cudnnex_cache:
        graph.build([cudnn.heur_mode.A])

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

    if attn_mask is not None:
        if query.ndim > attn_mask.ndim:
            attn_mask = attn_mask.view(*((1,) * (query.ndim - attn_mask.ndim)), *attn_mask.shape)
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attn_mask = attn_bias

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
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        query.stride(),
        key.stride(),
        value.stride(),
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
        grad_attn_mask = torch.empty_like(attn_mask)

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
