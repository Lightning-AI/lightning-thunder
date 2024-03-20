import math
from looseversion import LooseVersion

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import thunder.core.dtypes as dtypes
from thunder.core.proxies import Proxy, TensorProxy
import thunder.core.utils as utils
import thunder.core.devices as devices
from thunder.core.compile_data import get_compile_option

import thunder.torch as ltorch
from thunder.torch import TensorLike

from thunder.core.transforms import (
    get_grad,
    put_grad,
    put_grads,
)
from thunder.extend import OperatorExecutor, register_executor

from typing import Tuple
from enum import auto, Enum

sdpa_ex: OperatorExecutor = OperatorExecutor("sdpa", version="0.1")
register_executor(sdpa_ex)


class SpdaBackend(Enum):
    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    MEMORY_EFFICIENT = 2


# Both flash attention and memory efficient sdpa require that the last stride be one.
def _sdpa_enforce_input_tensor_contiguity(a: torch.Tensor) -> torch.Tensor:
    if a is None or a.stride(-1) == 1:
        return a
    else:
        return a.contiguous()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _sdpa_pad_head_dimension(a: torch.Tensor) -> torch.Tensor:
    head_size = a.shape[-1]
    # If the head is already a multiple of 8, then we don't need to pad. The
    # pad op can be quite expensive in some cases.
    if head_size % 8 == 0:
        return a
    padding_size = ceil_div(head_size, 8) * 8 - head_size
    return torch.nn.functional.pad(a, [0, padding_size], value=0.0)


def _sdpa_slice_head_dimension(a: torch.Tensor, head_size: int) -> torch.Tensor:
    # ditto pad_head_dimension: the slice can be expensive, so skip if possible.
    if head_size % 8 == 0:
        return a
    return a[:, :, :, 0:head_size]


def _sdpa_pad_scale(a: None | float, head_size: int) -> float:
    if a is not None:
        return a

    if head_size % 8 == 0:
        return None

    return 1.0 / math.sqrt(head_size)


# Configure attention mask argument for memory efficient sdpa kernel
def _attention_mask_memory_efficient_helper(attn_mask: None | torch.Tensor, query: torch.Tensor) -> None | torch.Tensor:
    if attn_mask is None:
        return None

    # When a boolean mask is used, it needs to be converted to an additive mask where zero'd elements are filled
    # with a very negative value that should become ~0 after softmax
    if attn_mask.dtype == torch.bool:
        attn_mask = torch.masked_fill(torch.zeros_like(attn_mask, dtype=query.dtype), attn_mask == False, -math.inf)

    # Expand the number of heads in attention mask to match query, key, and value tensors.
    num_heads = query.shape[1]
    head_dim = query.shape[-1]

    batch_size, _, query_seq_len, key_seq_len = attn_mask.shape
    expanded_attn_mask = attn_mask.expand(batch_size, num_heads, query_seq_len, key_seq_len)

    utils.check(
        head_dim > 0,
        lambda: f"Expected head dimension to be greater than 0.",
    )

    utils.check(
        key_seq_len > 0,
        lambda: f"Expected key-value sequence length to be greater than 0.",
    )

    # Pad and slice attention mask to ensure correct alignment.
    if head_dim != key_seq_len:
        ceil_power_of_eight = ceil_div(key_seq_len, 8) * 8
        padded_size = ceil_power_of_eight - key_seq_len
        padded_attn_mask = torch.nn.functional.pad(expanded_attn_mask, [0, padded_size], value=0.0)
        return padded_attn_mask[:, :, :, 0:key_seq_len]
    else:
        return expanded_attn_mask.contiguous()


# TODO These checks should be converted to compile-time checks using a checker function
# This helper function checks that the shape of input tensors are supported by fused sdpa implementation.
def _input_shape_check_fused_scaled_dot_product_attention(
    query: TensorLike, key: TensorLike, value: TensorLike, attn_mask: None | TensorLike
):
    # Restrict input tensors to 4 dimension
    utils.check(
        query.ndim == 4,
        lambda: f"grad_forward_sdpa: Expected query tensor to have 4 dimension, but it has {query.ndim}.",
    )
    utils.check(
        key.ndim == 4,
        lambda: f"grad_forward_sdpa: Expected key tensor to have 4 dimension, but it has {key.ndim}.",
    )
    utils.check(
        value.ndim == 4,
        lambda: f"grad_forward_sdpa: Expected value tensor to have 4 dimension, but it has {value.ndim}.",
    )
    utils.check(
        attn_mask is None or attn_mask.ndim == 4,
        lambda: f"grad_forward_sdpa: Expected attn_mask tensor to have 4 dimension, but it has {attn_mask.ndim}.",
    )

    # query (batch_size, num_heads, query_seq_len, E)
    # key (batch_size, num_heads, key_seq_len, E)
    # value (batch_size, num_heads, key_seq_len, Ev)
    # attn_mask (batch_size, num_heads, query_seq_len, key_seq_len)
    inputs = [query, key, value]
    if attn_mask is not None:
        inputs.append(attn_mask)

    # NOTE aten::scaled_dot_product_efficient_attention does not support broadcastable batch size.
    utils.check(
        all(a.shape[0] == inputs[0].shape[0] for a in inputs),
        lambda: "grad_forward_sdpa: Expected all inputs to have same batch_size.",
    )

    # Check for the same number of heads
    utils.check(
        all(a.shape[1] == 1 or a.shape[1] == inputs[0].shape[1] for a in inputs),
        lambda: "grad_forward_sdpa: Expected all inputs to have same number of attention heads or a broadcastable dimension.",
    )


# TODO These checks should be converted to compile-time checks using a checker function
# This helper function checks that the dtypes of input tensors are supported by fused sdpa implementation.
def _input_dtype_check_fused_scaled_dot_product_attention(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: None | TensorLike,
    supported_dtypes: tuple[dtypes.dtype],
):
    utils.check(
        query.dtype in supported_dtypes,
        lambda: f"grad_forward_sdpa: Only {supported_dtypes} dtypes are supported, but query has {query.dtype}.",
    )
    utils.check(
        key.dtype in supported_dtypes,
        lambda: f"grad_forward_sdpa: Only {supported_dtypes} dtypes are supported, but key has {key.dtype}.",
    )
    utils.check(
        value.dtype in supported_dtypes,
        lambda: f"grad_forward_sdpa: Only {supported_dtypes} dtypes are supported, but value has {value.dtype}.",
    )


# This helper function maps to aten::_scaled_dot_product_efficient_attention function.
def _grad_forward_scaled_dot_product_efficient_attention_meta(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: None | TensorLike,
    dropout_p: float = 0.0,
    is_causal=False,
    scale: None | float = None,
) -> tuple[TensorProxy, TensorProxy, TensorProxy, TensorProxy]:
    # Reference metadata:
    # https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py#L4863-L4899
    # * query (batch_size, num_heads, query_seq_len, E)
    # * key (batch_size, num_heads, key_seq_len, E)
    # * value (batch_size, num_heads, key_seq_len, Ev)
    # * attn_mask (batch_size, num_heads, query_seq_len, key_seq_len)
    # * output (batch_size, num_heads, query_seq_len, Ev)

    # FP64 is not supported by aten memory efficient implementation
    supported_dtypes = (dtypes.float32, dtypes.float16, dtypes.bfloat16)
    _input_dtype_check_fused_scaled_dot_product_attention(query, key, value, attn_mask, supported_dtypes)
    _input_shape_check_fused_scaled_dot_product_attention(query, key, value, attn_mask)

    batch_size, num_heads, query_seq_len, E = query.shape
    key_seq_len = key.shape[-2]
    Ev = value.shape[-1]
    logsumexp_dim = math.ceil(query_seq_len / 32) * 32

    return (
        output := TensorProxy(like=query, shape=(batch_size, num_heads, query_seq_len, Ev)),
        log_sumexp := TensorProxy(
            shape=(batch_size, num_heads, logsumexp_dim), dtype=dtypes.float32, device=query.device, requires_grad=False
        ),
        philox_seed := TensorProxy(shape=(), dtype=dtypes.int64, device=query.device, requires_grad=False),
        philox_offset := TensorProxy(shape=(), dtype=dtypes.int64, device=query.device, requires_grad=False),
    )


def _grad_forward_scaled_dot_product_efficient_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: None | torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: None | float = None,
) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    # Reference: https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L394-L415
    return torch.ops.aten._scaled_dot_product_efficient_attention(
        _sdpa_enforce_input_tensor_contiguity(query),
        _sdpa_enforce_input_tensor_contiguity(key),
        _sdpa_enforce_input_tensor_contiguity(value),
        _attention_mask_memory_efficient_helper(attn_mask, query),
        compute_logsumexp := True,
        dropout_p,
        is_causal,
        scale=scale,
    )


sdpea_gradfwd = sdpa_ex.register_operator(
    "sdpaex_grad_forward_scaled_dot_product_efficient_attention",
    meta=_grad_forward_scaled_dot_product_efficient_attention_meta,
    fn=_grad_forward_scaled_dot_product_efficient_attention_impl,
)


# This helper function maps to aten::_scaled_dot_product_flash_attention function.
def _grad_forward_scaled_dot_product_flash_attention_meta(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
) -> (TensorProxy, TensorProxy, TensorProxy, TensorProxy, int, int, TensorProxy, TensorProxy, TensorProxy):
    # Reference metadata:
    # https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py
    # * query (batch_size, num_heads, query_seq_len, E)
    # * key (batch_size, num_heads, key_seq_len, E)
    # * value (batch_size, num_heads, key_seq_len, Ev)
    # * output (batch_size, num_heads, query_seq_len, Ev)

    # FP64 is not supported by aten memory efficient implementation
    supported_dtypes = (dtypes.float16, dtypes.bfloat16)
    _input_dtype_check_fused_scaled_dot_product_attention(query, key, value, attn_mask := None, supported_dtypes)
    _input_shape_check_fused_scaled_dot_product_attention(query, key, value, attn_mask := None)

    batch_size, num_heads, query_seq_len, E = query.shape
    key_seq_len = key.shape[2]
    logsumexp_dim = math.ceil(query_seq_len / 16) * 16

    utils.check(
        E == key.shape[-1],
        lambda: f"scaled dot product flash attention expects query head dim {E} to equal key head dim {key.shape[-1]}",
    )

    return (
        output := TensorProxy(like=query, shape=(batch_size, num_heads, query_seq_len, E)),
        log_sumexp := TensorProxy(
            shape=(batch_size, num_heads, logsumexp_dim), dtype=dtypes.float32, device=query.device, requires_grad=False
        ),
        cum_seq_q := TensorProxy(shape=(batch_size + 1,), dtype=dtypes.int64, device=query.device, requires_grad=False),
        cum_seq_k := TensorProxy(shape=(batch_size + 1,), dtype=dtypes.int64, device=query.device, requires_grad=False),
        query_seq_len,
        key_seq_len,
        philox_seed := TensorProxy(shape=(), dtype=dtypes.int64, device=query.device, requires_grad=False),
        philox_offset := TensorProxy(shape=(), dtype=dtypes.int64, device=query.device, requires_grad=False),
        debug_attn_mask := TensorProxy(shape=(), dtype=dtypes.int64, device=query.device, requires_grad=False),
    )


def _grad_forward_scaled_dot_product_flash_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: None | float = None,
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor, torch.Tensor, torch.Tensor):
    primal, *remaining_args = torch.ops.aten._scaled_dot_product_flash_attention(
        _sdpa_pad_head_dimension(_sdpa_enforce_input_tensor_contiguity(query)),
        _sdpa_pad_head_dimension(_sdpa_enforce_input_tensor_contiguity(key)),
        _sdpa_pad_head_dimension(_sdpa_enforce_input_tensor_contiguity(value)),
        dropout_p,
        is_causal,
        return_debug_mask=False,
        scale=_sdpa_pad_scale(scale, value.shape[-1]),
    )
    return _sdpa_slice_head_dimension(primal, value.shape[-1]), *remaining_args


sdpfa_gradfwd = sdpa_ex.register_operator(
    "sdpafx_grad_forward_scaled_dot_product_efficient_attention",
    meta=_grad_forward_scaled_dot_product_flash_attention_meta,
    fn=_grad_forward_scaled_dot_product_flash_attention_impl,
)


# The backward decomposition of scaled_dot_product_attention cannot be efficiently fused, so we have this
# scaled_dot_product_efficient_attention_backward primitive. Executors can override the primitive using
# internal implementations.
def _scaled_dot_product_efficient_attention_backward_meta(
    grad_out: TensorLike,
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: None | TensorLike,
    out: TensorLike,
    logsumexp: TensorLike,
    philox_seed: TensorLike,
    philox_offset: TensorLike,
    dropout_p: float,
    is_causal: bool = False,
    *,
    scale: None | float = None,
) -> (TensorProxy, TensorProxy, TensorProxy, None | TensorProxy):
    # FP64 is not supported by aten memory efficient implementation
    supported_dtypes = (dtypes.float32, dtypes.float16, dtypes.bfloat16)
    _input_dtype_check_fused_scaled_dot_product_attention(query, key, value, attn_mask, supported_dtypes)
    _input_shape_check_fused_scaled_dot_product_attention(query, key, value, attn_mask)

    # Reference metadata:
    # https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py#L4907-L4956
    grad_query = TensorProxy(like=query, shape=query.shape)
    grad_key = TensorProxy(like=key, shape=key.shape)
    grad_value = TensorProxy(like=value, shape=value.shape)
    grad_attn_mask = None
    if attn_mask is not None:
        grad_attn_mask = TensorProxy(like=attn_mask, shape=attn_mask.shape)
    # Return gradients for query, key, value, and attn_mask tensor inputs
    return (grad_query, grad_key, grad_value, grad_attn_mask)


# TODO Move calls to masked_fill to a transform instead of hiding them in the impl
def _scaled_dot_product_efficient_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: None | torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
    scale: None | float,
) -> (torch.Tensor, torch.Tensor, torch.Tensor, None | torch.Tensor):
    grad_input_mask = [a.requires_grad for a in (query, key, value)]
    if attn_mask is None:
        grad_input_mask.append(False)
    else:
        grad_input_mask.append(attn_mask.requires_grad)

    # Reference: https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L394-L415
    return torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out,
        _sdpa_enforce_input_tensor_contiguity(query),
        _sdpa_enforce_input_tensor_contiguity(key),
        _sdpa_enforce_input_tensor_contiguity(value),
        _attention_mask_memory_efficient_helper(attn_mask, query),
        out,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask,
        is_causal,
        scale=scale,
    )


sdpea_bwd = sdpa_ex.register_operator(
    "sdpaex_scaled_dot_product_efficient_attention_backward",
    meta=_scaled_dot_product_efficient_attention_backward_meta,
    fn=_scaled_dot_product_efficient_attention_backward_impl,
)


# The backward decomposition of scaled_dot_product_attention cannot be efficiently fused, so we have this
# scaled_dot_product_flash_attention_backward primitive. Executors can override the primitive using
# internal implementations.
def _scaled_dot_product_flash_attention_backward_meta(
    grad_out: TensorLike,
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    out: TensorLike,
    logsumexp: TensorLike,
    cum_seq_q: TensorLike,
    cum_seq_k: TensorLike,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: TensorLike,
    philox_offset: TensorLike,
    *,
    scale: None | float = None,
) -> (TensorProxy, TensorProxy, TensorProxy):
    # FP64 is not supported by aten memory efficient implementation
    supported_dtypes = (dtypes.float16, dtypes.bfloat16)
    _input_dtype_check_fused_scaled_dot_product_attention(query, key, value, attn_mask := None, supported_dtypes)
    _input_shape_check_fused_scaled_dot_product_attention(query, key, value, attn_mask := None)

    batch_size, num_heads, query_seq_len, E = query.shape

    # Reference metadata:
    # https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py#L4907-L4956
    grad_query = TensorProxy(like=query, shape=(batch_size, num_heads, max_q, E))
    grad_key = TensorProxy(like=key, shape=(batch_size, num_heads, max_k, E))
    grad_value = TensorProxy(like=value, shape=(batch_size, num_heads, max_k, E))
    return (grad_query, grad_key, grad_value)


def _scaled_dot_product_flash_attention_backward_impl(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    scale: None | float,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    grads = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        _sdpa_pad_head_dimension(grad_out),
        _sdpa_pad_head_dimension(_sdpa_enforce_input_tensor_contiguity(query)),
        _sdpa_pad_head_dimension(_sdpa_enforce_input_tensor_contiguity(key)),
        _sdpa_pad_head_dimension(_sdpa_enforce_input_tensor_contiguity(value)),
        _sdpa_pad_head_dimension(out),
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=_sdpa_pad_scale(scale, value.shape[-1]),
    )
    return (_sdpa_slice_head_dimension(g, value.shape[-1]) for g in grads)


sdpfa_bwd = sdpa_ex.register_operator(
    "sdpafx_scaled_dot_product_efficient_attention_backward",
    meta=_scaled_dot_product_flash_attention_backward_meta,
    fn=_scaled_dot_product_flash_attention_backward_impl,
)


def _scaled_dot_product_attention_fused(
    query: Proxy,
    key: Proxy,
    value: Proxy,
    attn_mask: None | Proxy,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
):
    # Figure out which SDPA to use. There are performance cliffs to the various
    # implementations, and this makes the decision cognizant of those cliffs.
    backend = _fused_sdp_choice(query, key, value, attn_mask, dropout_p, is_causal, scale)

    utils.check(
        backend != SpdaBackend.ERROR,
        lambda: "Unable to find valid backend for scaled_dot_product_attention.",
    )
    utils.check(
        backend != SpdaBackend.MATH,
        lambda: "The fallback to sdpa thunder reference is not implemented.",
        exception_type=NotImplementedError,
    )

    tensor_args = (query, key, value)
    scalar_args = (dropout_p, is_causal)
    if backend == SpdaBackend.FLASH_ATTENTION:
        # Use flash attention kernel
        (primal, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, _) = sdpfa_gradfwd(
            *tensor_args, *scalar_args, scale=scale
        )
    elif backend == SpdaBackend.MEMORY_EFFICIENT:
        # Use memory efficient kernel, which supports fp32 and attention mask arguments
        (primal, logsumexp, philox_seed, philox_offset) = sdpea_gradfwd(
            *tensor_args, attn_mask, *scalar_args, scale=scale
        )
    return primal


def _scaled_dot_product_attention_grad(
    query: Proxy,
    key: Proxy,
    value: Proxy,
    attn_mask: None | Proxy,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
):
    # Figure out which SDPA to use. There are performance cliffs to the various
    # implementations, and this makes the decision cognizant of those cliffs.
    backend = _fused_sdp_choice(query, key, value, attn_mask, dropout_p, is_causal, scale)

    utils.check(
        backend != SpdaBackend.ERROR,
        lambda: "Unable to find valid backend for scaled_dot_product_attention.",
    )
    utils.check(
        backend != SpdaBackend.MATH,
        lambda: "The fallback to sdpa thunder reference is not implemented.",
        exception_type=NotImplementedError,
    )

    tensor_args = (query, key, value)
    scalar_args = (dropout_p, is_causal)
    if backend == SpdaBackend.FLASH_ATTENTION:
        # Use flash attention kernel
        (primal, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, _) = sdpfa_gradfwd(
            *tensor_args, *scalar_args, scale=scale
        )
        g = get_grad(primal)
        grad_query, grad_key, grad_val = sdpfa_bwd(
            g,
            *tensor_args,
            primal,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale=scale,
        )
        put_grads((query, key, value), (grad_query, grad_key, grad_val))
    elif backend == SpdaBackend.MEMORY_EFFICIENT:
        # Use memory efficient kernel, which supports fp32 and attention mask arguments
        (primal, logsumexp, philox_seed, philox_offset) = sdpea_gradfwd(
            *tensor_args, attn_mask, *scalar_args, scale=scale
        )
        g = get_grad(primal)
        grad_query, grad_key, grad_val, grad_attn_mask = sdpea_bwd(
            g,
            query,
            key,
            value,
            attn_mask,
            primal,
            logsumexp,
            philox_seed,
            philox_offset,
            dropout_p,
            is_causal,
            scale=scale,
        )
        put_grads((query, key, value), (grad_query, grad_key, grad_val))
        if attn_mask is not None:
            put_grad(attn_mask, grad_attn_mask)
    return primal


# This helper function converts Thunder Proxy to PyTorch Meta Tensor
def _convert_to_meta_tensor(a: None | TensorProxy) -> None | torch.Tensor:
    from thunder.torch import _thunder_to_torch_dtype_map

    if a is None:
        return None
    return torch.empty(
        a.shape,
        dtype=_thunder_to_torch_dtype_map[a.dtype],
        requires_grad=a.requires_grad,
        device="meta",
    )


# This helper function converts PyTorch meta tensor to FakeTensor, which
# models stride order for contiguity checks.
def _convert_to_fake_tensor(mode: FakeTensorMode, a: None | torch.Tensor) -> None | FakeTensor:
    if a is None:
        return None
    return FakeTensor(mode, a, device="cuda")


# Convert input tensors represented as Thunder Proxy to PyTorch FakeTensor.
# Determine which fused sdpa kernel.
def _fused_sdp_choice(
    query: Proxy,
    key: Proxy,
    value: Proxy,
    attn_mask: None | Proxy,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: None | float = None,
) -> int:
    input_tensors = (query, key, value, attn_mask)
    meta_input_tensors = list(map(_convert_to_meta_tensor, input_tensors))
    with FakeTensorMode() as mode:
        fake_query, fake_key, fake_value, fake_attn_mask = list(
            map(lambda a: _convert_to_fake_tensor(mode, a), meta_input_tensors)
        )

    import thunder

    if isinstance(is_causal, thunder.core.proxies.IntegerProxy):
        is_causal = is_causal.value

    if LooseVersion(torch.__version__) < LooseVersion("2.2.0"):
        # Figure out which SDPA to use. There are performance cliffs to the
        # various implementations, and this makes the decision cognizant of
        # those cliffs.
        backend = torch._fused_sdp_choice(
            fake_query,
            fake_key,
            fake_value,
            fake_attn_mask,
            dropout_p,
            is_causal,
            scale=scale,
        )
        return SpdaBackend(backend)
    else:
        from torch.backends.cuda import (
            SDPAParams,
            can_use_efficient_attention,
            can_use_flash_attention,
            flash_sdp_enabled,
            math_sdp_enabled,
            mem_efficient_sdp_enabled,
        )

        sdp_params = SDPAParams(fake_query, fake_key, fake_value, fake_attn_mask, dropout_p, is_causal)

        enable_debug: None | bool = get_compile_option(
            "sdpa_debug", "Enables sdpa backend warning messages when a specific kernel is unavailable."
        )
        # Set default value.
        if enable_debug is None:
            enable_debug = False
        assert isinstance(enable_debug, bool)

        if flash_sdp_enabled() and can_use_flash_attention(sdp_params, enable_debug):
            return SpdaBackend.FLASH_ATTENTION
        elif mem_efficient_sdp_enabled() and can_use_efficient_attention(sdp_params, enable_debug):
            return SpdaBackend.MEMORY_EFFICIENT
        elif math_sdp_enabled():
            return SpdaBackend.MATH
        else:
            return SpdaBackend.ERROR


def _scaled_dot_product_attention_checker(
    query: Proxy,
    key: Proxy,
    value: Proxy,
    attn_mask: None | Proxy,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: None | float,
) -> bool:
    input_tensors = (query, key, value, attn_mask)
    if any(map(lambda a: a is not None and a.device is devices.cpu, input_tensors)):
        return False

    # Register augmented fusion only for memory_efficient and flash attention sdpa
    backend = _fused_sdp_choice(query, key, value, attn_mask, dropout_p, is_causal, scale)
    return backend == SpdaBackend.FLASH_ATTENTION or backend == SpdaBackend.MEMORY_EFFICIENT


sdpa_ex.register_implementation(
    ltorch.scaled_dot_product_attention,
    checker=_scaled_dot_product_attention_checker,
    execution_transform=_scaled_dot_product_attention_fused,
    grad_transform=_scaled_dot_product_attention_grad,
)
