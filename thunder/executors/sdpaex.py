import math
from looseversion import LooseVersion

import torch

import thunder.core.dtypes as dtypes
from thunder.core.proxies import TensorProxy
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

sdpa_ex: OperatorExecutor = OperatorExecutor("sdpa", version="0.1")
register_executor(sdpa_ex)


# TODO These checks should be converted to compile-time checks using a checker function
def _input_check_scaled_dot_product_efficient_attention(
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

    # FP64 is not supported by aten implementation
    supported_dtypes = (dtypes.float32, dtypes.float16, dtypes.bfloat16)
    utils.check(
        query.dtype in supported_dtypes,
        lambda: f"grad_forward_sdpa: Only fp32, half & bf16 dtypes are supported, but query has {query.dtype}.",
    )
    utils.check(
        key.dtype in supported_dtypes,
        lambda: f"grad_forward_sdpa: Only fp32, half & bf16 dtypes are supported, but key has {key.dtype}.",
    )
    utils.check(
        value.dtype in supported_dtypes,
        lambda: f"grad_forward_sdpa: Only fp32, half & bf16 dtypes are supported, but value has {value.dtype}.",
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
        all(a.shape[1] == inputs[0].shape[1] for a in inputs),
        lambda: "grad_forward_sdpa: Expected all inputs to have same number of attention heads.",
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
    _input_check_scaled_dot_product_efficient_attention(query, key, value, attn_mask)

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
    # When a boolean mask is used, it needs to be converted to an additive mask where zero'd elements are filled
    # with a very negative value that should become ~0 after softmax
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = torch.masked_fill(torch.zeros_like(attn_mask, dtype=query.dtype), attn_mask == False, -math.inf)

    # Reference: https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L394-L415
    return torch.ops.aten._scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_mask,
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
    out, logsumexp, philox_seed, philox_offset = sdpea_gradfwd(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )
    return out, (query, key, value, attn_mask, out, logsumexp, philox_seed, philox_offset, dropout_p, is_causal, scale)


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
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask: TensorProxy | None,
    out: TensorProxy,
    logsumexp: TensorProxy,
    philox_seed: TensorProxy,
    philox_offset: TensorProxy,
    dropout_p,
    is_causal: bool,
    scale: float | None,
    grad_out: TensorProxy,
):
    (
        grad_query,
        grad_key,
        grad_val,
        grad_attn_mask,
    ) = sdpea_bwd(
        grad_out,
        query,
        key,
        value,
        attn_mask,
        out,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        is_causal,
        scale,
    )
    return grad_query, grad_key, grad_val, grad_attn_mask


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
    scale: None | float = None,
) -> (TensorProxy, TensorProxy, TensorProxy, None | TensorProxy):
    _input_check_scaled_dot_product_efficient_attention(query, key, value, attn_mask)

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
        # When a boolean mask is used, it needs to be converted to an additive mask where zero'd elements are filled
        # with a very negative value that should become ~0 after softmax
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.masked_fill(torch.zeros_like(attn_mask, dtype=query.dtype), attn_mask == False, -math.inf)

    # Reference: https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L394-L415
    return torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out,
        query,
        key,
        value,
        attn_mask,
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


def _scaled_dot_product_attention_grad(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None
):
    fwd, logsumexp, philox_seed, philox_offset = sdpea_gradfwd(
        query, key, value, attn_mask, dropout_p, is_causal, scale=scale
    )

    g = get_grad(fwd)
    grad_query, grad_key, grad_val, grad_attn_mask = sdpea_bwd(
        g, query, key, value, attn_mask, fwd, logsumexp, philox_seed, philox_offset, dropout_p, is_causal, scale=scale
    )
    put_grads((query, key, value), (grad_query, grad_key, grad_val))

    if attn_mask is not None:
        put_grad(attn_mask, grad_attn_mask)

    return fwd


def _scaled_dot_product_attention_checker(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None
) -> bool:
    # query (batch_size, num_heads, query_seq_len, E)
    # key (batch_size, num_heads, key_seq_len, E)
    # value (batch_size, num_heads, key_seq_len, Ev)
    # attn_mask (batch_size, num_heads, query_seq_len, key_seq_len)

    batch_size, num_heads, _, _ = query.shape

    tensor_inputs = [query, key, value]
    if attn_mask is not None:
        tensor_inputs.append(attn_mask)

    if any(map(lambda a: a.device is devices.cpu, tensor_inputs)):
        return False

    # NOTE Expected query, key, value, and attn_mask tensor to have 4 dimensions
    if any(map(lambda a: a.ndim != 4, tensor_inputs)):
        return False

    # NOTE FP64 is not supported by aten implementation
    supported_dtypes = (dtypes.float32, dtypes.float16, dtypes.bfloat16)
    if any(map(lambda a: a.dtype not in supported_dtypes, [query, key, value])):
        return False

    # NOTE aten::scaled_dot_product_efficient_attention does not support broadcastable batch size.
    if any(map(lambda a: a.shape[0] != batch_size, tensor_inputs)):
        return False

    # NOTE Expected all inputs to have same number of attention heads.
    if any(map(lambda a: a.shape[1] != num_heads, tensor_inputs)):
        return False

    return True


sdpa_ex.register_implementation(
    ltorch.scaled_dot_product_attention,
    checker=_scaled_dot_product_attention_checker,
    execution_transform=ltorch.scaled_dot_product_attention,
    grad_transform=_scaled_dot_product_attention_grad,
)
