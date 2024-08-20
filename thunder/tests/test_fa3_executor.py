import pytest
import torch

import thunder
import thunder.core
from thunder.executors.fa3ex import fa3_ex
from thunder.tests.framework import requiresCUDA

# flash attn 3
try:
    from flash_attn_interface import _flash_attn_forward, _flash_attn_backward, flash_attn_func

    HAS_FA3 = True
except:
    HAS_FA3 = False

import math
from einops import rearrange, repeat


# helper function for test_fa3_accuracy_vs_ref
def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores /= softcap
        scores = scores.tanh()
        scores *= softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


# verify that fa3 kernel is accurate against manual reference implementation
@requiresCUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bfloat16", "float16"))
def test_fa3_accuracy_vs_ref(dtype: torch.dtype):
    if not HAS_FA3:
        pytest.skip("fa3 not built")

    batch = 4
    seq_len = 128
    num_heads = 6
    dim_per_head = 64
    device = "cuda"

    q = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    cfn = thunder.jit(fn, executors=[fa3_ex])

    # Verifies the result is close to PyTorch
    out = cfn(q, k, v)
    out_ref = attention_ref(q, k, v)[0]
    out_pt = attention_ref(
        q,
        k,
        v,
        upcast=False,
        reorder_ops=True,
    )[0]

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)
    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + 3e-5
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + 3e-5
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + 3e-5


# verify that fa3 kernel is accurate against torch native sdpa kernel
@requiresCUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bfloat16", "float16"))
def test_fa3_accuracy_vs_torch_sdpa(dtype: torch.dtype):
    if not HAS_FA3:
        pytest.skip("fa3 not built")

    batch = 4
    seq_len = 128
    num_heads = 6
    dim_per_head = 64
    device = "cuda"

    q = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)

    out = flash_attn_func(q, k, v)[0]

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    # from https://github.com/Dao-AILab/flash-attention/issues/1128#issuecomment-2269962552:
    # FA uses (batch, seqlen, nheads, headdim). Torch sdpa expects (batch, nheads, seqlen, headdim).
    q, k, v = (torch.transpose(a, 1, 2) for a in (q, k, v))
    out_ref = fn(q, k, v)
    out_ref = torch.transpose(out_ref, 1, 2)

    # Verifies the result is close to PyTorch
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-2)


# verify that fa3 kernel is properly being called when used in a valid trace
@requiresCUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bfloat16", "float16"))
def test_fa3_used(dtype: torch.dtype):
    if not HAS_FA3:
        pytest.skip("fa3 not built")

    batch = 4
    seq_len = 128
    num_heads = 6
    dim_per_head = 64
    device = "cuda"

    query = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype, requires_grad=True)

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    cfn = thunder.jit(fn, executors=[fa3_ex])

    thunder_result = cfn(query, key, value)

    # Verifies fa3 fwd was called
    extrace = thunder.last_traces(cfn)[-1]
    assert any(bsym.sym.name == "fa3_fwd" for bsym in extrace.bound_symbols)

    loss = thunder_result.sum()
    loss.backward()

    # Verifies fa3 bwd was called
    bw_extrace = thunder.last_backward_traces(cfn)[-1]
    assert any(bsym.sym.name == "fa3_bwd" for bsym in bw_extrace.bound_symbols)


# verify that checker is correctly returning False on invalid fa3 use cases
@requiresCUDA
def test_checker():

    if not HAS_FA3:
        pytest.skip("fa3 not built")

    batch = 10
    seq_len = 128
    num_heads = 4
    dim_per_head = 32

    # default valid inputs
    device = "cuda"
    dtype = torch.float16
    attn_mask = None
    dropout_p = 0.0

    # helper function to verify that the fa3 executor is not getting getting given various combinations of invalid inputs
    def check(device, dtype, attn_mask, dropout_p):
        query = torch.randn([batch, seq_len, num_heads, dim_per_head], device=device, dtype=dtype)
        key = torch.randn([batch, seq_len, num_heads, dim_per_head], device=device, dtype=dtype)
        value = torch.randn([batch, seq_len, num_heads, dim_per_head], device=device, dtype=dtype)

        def fn(query, key, value, attn_mask=None, dropout_p=0.0):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0
            )

        cfn = thunder.jit(fn, executors=[fa3_ex])

        thunder_result = cfn(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p)

        # Verifies fa3 was not called
        extrace = thunder.last_traces(cfn)[-1]
        assert not any(bsym.sym.name == "fa3_fwd" for bsym in extrace.bound_symbols)

    # currently fa3 requires gpu inputs
    check("cpu", dtype, attn_mask, dropout_p)
    # currently fa3 is fp16 and bf16
    check(device, torch.float32, attn_mask, dropout_p)
    # currently fa3 doesn't support attn_mask != None
    check(
        device,
        dtype,
        torch.randn([batch, seq_len, num_heads, num_heads], device="cuda", dtype=torch.float16),
        dropout_p,
    )
    # currently fa3 doesn't support dropout != 0.0
    check(device, dtype, attn_mask, 0.5)
