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


@requiresCUDA
def test_fa3():
    if not HAS_FA3:
        pytest.skip("fa3 not built")

    batch = 4
    seq_len = 128
    num_heads = 6
    dim_per_head = 64
    device = 'cuda'
    dtype = torch.float16

    query = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype)
    key = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype)
    value = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", dtype=dtype)

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    cfn = thunder.jit(fn, executors=[fa3_ex])

    # Verifies the result is close to PyTorch
    thunder_result = cfn(query, key, value)
    torch_result = fn(query, key, value)

    torch.testing.assert_close(thunder_result, torch_result)

    # Verifies fa3 was called
    extrace = thunder.last_traces(cfn)[-1]
    assert any(bsym.sym.name == "fa3_fwd" for bsym in extrace.bound_symbols)


# verify that checker is correctly returning False on invalid fa3 use cases
def test_checker():

    if not HAS_FA3:
        pytest.skip("fa3 not built")

    batch = 10
    seq_len = 128
    num_heads = 4
    dim_per_head = 32

    device = "cuda"
    dtype = torch.float16
    attn_mask = None
    dropout_p = 0.0

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

    check("cpu", dtype, attn_mask, dropout_p)
    check(device, torch.bfloat16, attn_mask, dropout_p)
    check(
        device,
        dtype,
        torch.randn([batch, seq_len, num_heads, num_heads], device="cuda", dtype=torch.float16),
        dropout_p,
    )
    check(device, dtype, attn_mask, 0.5)
