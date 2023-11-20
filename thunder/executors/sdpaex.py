import math
from looseversion import LooseVersion

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import thunder.core.dtypes as dtypes
from thunder.core.proxies import Proxy, TensorProxy
import thunder.core.utils as utils
import thunder.core.devices as devices

import thunder.torch as ltorch
from thunder.torch import TensorLike

from thunder.core.transforms import (
    get_grad,
    put_grad,
    put_grads,
    register_augmented_forward_with_checker,
    register_backward,
)
from thunder.extend import OperatorExecutor, register_executor

from typing import Tuple
from enum import auto, Enum

sdpa_ex: OperatorExecutor = OperatorExecutor("sdpa", version="0.1")
register_executor(sdpa_ex)


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
    padding_size = ceil_div(head_size, 8) * 8 - head_size
    return torch.nn.functional.pad(a, [0, padding_size], value=0.0)


def _sdpa_slice_head_dimension(a: torch.Tensor, head_size: int) -> torch.Tensor:
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

    if head_dim > key_seq_len:
        # Pad and slice attention mask to ensure correct alignment.
        padded_size = head_dim - key_seq_len
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


def _grad_forward_scaled_dot_product_efficient_attention_checker(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: None | TensorLike = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: None | float = None,
) -> bool:
    tensor_inputs = [query, key, value]
    if attn_mask is not None:
        tensor_inputs.append(attn_mask)

    # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention' with arguments from the 'CPU' backend.
    if any(map(lambda a: a.device is devices.cpu, tensor_inputs)):
        return False

    # TODO: Model PyTorch's choice of efficient kernels and fallbacks
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
    if scale is not None and LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
        return False
    return True


sdpa_ex.register_implementation(
    ltorch.grad_forward_scaled_dot_product_efficient_attention,
    sdpea_gradfwd,
    checker=_grad_forward_scaled_dot_product_efficient_attention_checker,
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


def _grad_forward_scaled_dot_product_flash_attention_checker(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: None | TensorLike = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: None | float = None,
) -> bool:
    tensor_inputs = [query, key, value]
    if attn_mask is not None:
        tensor_inputs.append(attn_mask)

    # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention' with arguments from the 'CPU' backend.
    if any(map(lambda a: a.device is devices.cpu, tensor_inputs)):
        return False

    # TODO: Model PyTorch's choice of efficient kernels and fallbacks
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
    if scale is not None and LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
        return False
    return True


sdpa_ex.register_implementation(
    ltorch.grad_forward_scaled_dot_product_flash_attention,
    sdpea_gradfwd,
    checker=_grad_forward_scaled_dot_product_flash_attention_checker,
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


def _scaled_dot_product_efficient_attention_backward_checker(
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
) -> bool:
    tensor_inputs = [query, key, value]
    if attn_mask is not None:
        tensor_inputs.append(attn_mask)

    # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention' with arguments from the 'CPU' backend.
    if any(map(lambda a: a.device is devices.cpu, tensor_inputs)):
        return False

    # TODO: Model PyTorch's choice of efficient kernels and fallbacks
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
    if scale is not None and LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
        return False
    return True


sdpa_ex.register_implementation(
    ltorch.scaled_dot_product_efficient_attention_backward,
    sdpea_bwd,
    checker=_scaled_dot_product_efficient_attention_backward_checker,
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


def _scaled_dot_product_flash_attention_backward_checker(
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
    *,
    scale: None | float,
) -> bool:
    tensor_inputs = [query, key, value]
    # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_flash_attention' with arguments from the 'CPU' backend.
    if any(map(lambda a: a.device is devices.cpu, tensor_inputs)):
        return False

    # TODO: Model PyTorch's choice of efficient kernels and fallbacks
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
    if scale is not None and LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
        return False
    return True


sdpa_ex.register_implementation(
    ltorch.scaled_dot_product_flash_attention_backward,
    sdpea_bwd,
    checker=_scaled_dot_product_flash_attention_backward_checker,
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
    # NOTE Select fused sdpa using PyTorch eager mode selection behavior
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
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
    # NOTE Select fused sdpa using PyTorch eager mode selection behavior
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
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


class SpdaBackend(Enum):
    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    MEMORY_EFFICIENT = 2


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

    # NOTE Select fused sdpa using PyTorch eager mode selection behavior
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
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


def scaled_dot_product_attention_aug_fw(
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask: TensorProxy | None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
):
    # NOTE Select fused sdpa using PyTorch eager mode selection behavior
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
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
    input_args = (*tensor_args, attn_mask, *scalar_args, scale)
    if backend == SpdaBackend.FLASH_ATTENTION:
        # Use flash attention kernel
        (primal, *remaining_results, debug_attn_mask) = sdpfa_gradfwd(*tensor_args, *scalar_args, scale=scale)
        # NOTE Remaining results contains [logsumexp, *flash_attn_only_residuals, *philox_residuals]
        residuals = (*input_args, primal, *remaining_results)
        return primal, residuals
    elif backend == SpdaBackend.MEMORY_EFFICIENT:
        # Use memory efficient kernel, which supports fp32 and attention mask arguments
        (primal, logsumexp, *philox_residuals) = sdpea_gradfwd(*tensor_args, attn_mask, *scalar_args, scale=scale)
        flash_attn_only_residuals = (None,) * 4
        residuals = (*input_args, primal, logsumexp, *flash_attn_only_residuals, *philox_residuals)
        return primal, residuals


def scaled_dot_product_efficient_attention_aug_fw_rule_check(
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask: None | TensorProxy,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: None | float,
) -> bool:
    if sdpa_ex.is_active:
        return _scaled_dot_product_attention_checker(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)
    return False


register_augmented_forward_with_checker(
    "torch.nn.functional.scaled_dot_product_attention",
    scaled_dot_product_efficient_attention_aug_fw_rule_check,
    scaled_dot_product_attention_aug_fw,
)


@register_backward("torch.nn.functional.scaled_dot_product_attention")
def scaled_dot_product_attention_backward(
    query: Proxy,
    key: Proxy,
    value: Proxy,
    attn_mask: None | Proxy,
    dropout_p: float,
    is_causal: bool,
    scale: None | float,
    out: Proxy,
    logsumexp: Proxy,
    cum_seq_q: None | Proxy,
    cum_seq_k: None | Proxy,
    max_q: None | int,
    max_k: None | int,
    philox_seed: Proxy,
    philox_offset: Proxy,
    grad_out: Proxy,
):
    tensor_args = (query, key, value)
    scalar_args = (dropout_p, is_causal)
    flash_attention_args = (cum_seq_q, cum_seq_k, max_q, max_k)
    philox_args = (philox_seed, philox_offset)
    use_flash_attn = all(map(lambda a: a is not None, (cum_seq_q, cum_seq_k, max_q, max_k)))
    if use_flash_attn:
        (
            grad_query,
            grad_key,
            grad_val,
        ) = sdpfa_bwd(
            grad_out,
            *tensor_args,
            out,
            logsumexp,
            *flash_attention_args,
            *scalar_args,
            *philox_args,
            scale=scale,
        )
        # grad_attn_mask is None since it is not supported by flash_attention kernel
        return grad_query, grad_key, grad_val
    else:
        (
            grad_query,
            grad_key,
            grad_val,
            grad_attn_mask,
        ) = sdpea_bwd(
            grad_out,
            *tensor_args,
            attn_mask,
            out,
            logsumexp,
            *philox_args,
            *scalar_args,
            scale=scale,
        )
        return grad_query, grad_key, grad_val, grad_attn_mask
