from __future__ import annotations

from enum import Enum
from contextlib import contextmanager

import torch
from looseversion import LooseVersion

import thunder.core.utils as utils
from thunder.core.symbol import BoundSymbol
from thunder.core.proxies import variableify, Proxy, unvariableify
from thunder.core.prims import PrimIDs
from thunder.core.transform_common import order_proxies
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from thunder.core.compile_data import get_compile_option


try:
    torch.cuda.graphs.is_current_stream_capturing()

    def is_cudagraph_capturing():
        return torch.cuda.graphs.is_current_stream_capturing()

except RuntimeError:
    # no cudagraph support (CPU only, ROCm, ...)
    def is_cudagraph_capturing():
        return False


# TODO Make these tags
comment_symbols = {
    PrimIDs.COMMENT,
    PrimIDs.UNPACK_TRIVIAL,
    PrimIDs.UNPACK_EMPTY_DICT,
}


# TODO Document this better
# TODO Review non-proxy inputs as being consumed -- currently only proxies can be inputs and outputs of these regions
class Region:
    def __init__(self, producers, consumers, bound_symbols: list[BoundSymbol]):
        # Stores input data
        self.bound_symbols = bound_symbols

        # Identifies inputs and outputs
        # NOTE Inputs and outputs are "variableified" sets
        consumes = set()
        produces = set()

        for bsym in self.bound_symbols:
            flatouts = bsym.flat_outs

            produces.update(
                variableify(x) for x in flatouts if isinstance(x, Proxy) and producers[x] in self.bound_symbols
            )

            # Short-circuits if the symbol is a comment, because comments don't consume anything
            #   Note that comments may produce things
            if bsym.sym.id in comment_symbols:
                continue

            # Updates what this region consumes, skipping symbols that never consume anything
            consumes.update(variableify(x) for x in bsym.flat_args if isinstance(x, Proxy))

        inputs = set()
        outputs = set()

        # Inputs are things which this consumes which are produced before it
        for x in consumes:
            x = unvariableify(x)

            if producers[x] not in self.bound_symbols:
                inputs.add(variableify(x))

        # Outputs are things this produces that are consumed after it
        for x in produces:
            x = unvariableify(x)
            consumed_by = consumers.get(x, ())
            for bsym in consumed_by:
                if bsym not in self.bound_symbols:
                    outputs.add(variableify(x))
                    break

        proxy_order = order_proxies(self.bound_symbols)
        self.inputs = utils.OrderedSet(sorted(inputs, key=lambda p: proxy_order[p.proxy.name]))
        self.outputs = utils.OrderedSet(sorted(outputs, key=lambda p: proxy_order[p.proxy.name]))

    def __repr__(self) -> str:
        s = "[Region:"

        for bsym in self.bound_symbols:
            s += f"\n{str(bsym)}"

        s += "]"

        return s


# Helper to use torch.autograd.Function as an implementation for a symbol.
# See `transformer_engineex.py` for example.
class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def pop_saved_tensors(self):
        try:
            return self.saved_tensors
        finally:
            del self.saved_tensors


@contextmanager
def set_saved_tensors(ctx, saved_tensors):
    ctx.saved_tensors = saved_tensors
    try:
        yield
    finally:
        del ctx.saved_tensors


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
    supported_dtypes: tuple[dtypes.dtype, ...],
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


# This helper function converts Thunder Proxy to PyTorch Meta Tensor
def _convert_to_meta_tensor(a: None | TensorProxy) -> None | torch.Tensor:
    from thunder.core.dtypes import to_torch_dtype

    if a is None:
        return None
    return torch.empty(
        a.shape,
        dtype=to_torch_dtype(a.dtype),
        requires_grad=a.requires_grad,
        device="meta",
    )


# This helper function converts PyTorch meta tensor to FakeTensor, which
# models stride order for contiguity checks.
def _convert_to_fake_tensor(mode: FakeTensorMode, a: None | torch.Tensor) -> None | FakeTensor:
    if a is None:
        return None
    return FakeTensor(mode, a, device="cuda")


class SpdaBackend(Enum):
    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    MEMORY_EFFICIENT = 2


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

        args = []
        if hasattr(SDPAParams, "enable_gqa"):
            args.append(False)

        sdp_params = SDPAParams(fake_query, fake_key, fake_value, fake_attn_mask, dropout_p, is_causal, *args)

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
