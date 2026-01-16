import pytest
import torch

import thunder
from lightning_utilities.core.imports import package_available


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not package_available("tilegym"), reason="requires tilegym")
def test_tilegym_executor_sdpa_rewrites_and_runs():
    tilegym_ex = thunder.get_executor("tilegym")
    assert tilegym_ex is not None

    def fn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

    # Choose a shape that avoids other SDPA executors' restrictions interfering with this test:
    # - Head dim divisible by 8
    # - No explicit attn_mask, no dropout
    B, H, S, D = 2, 8, 256, 128
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    jfn = thunder.jit(fn, executors=(tilegym_ex, *thunder.get_default_executors()))
    out = jfn(q, k, v)
    ref = fn(q, k, v)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    trace = thunder.last_traces(jfn)[-1]
    assert any(bsym.sym.executor is tilegym_ex for bsym in trace.bound_symbols)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not package_available("tilegym"), reason="requires tilegym")
def test_tilegym_executor_rms_norm_rewrites_and_runs():
    tilegym_ex = thunder.get_executor("tilegym")
    assert tilegym_ex is not None

    def fn(x, w):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, 1e-6)

    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=False)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=False)

    jfn = thunder.jit(fn, executors=(tilegym_ex, *thunder.get_default_executors()))
    out = jfn(x, w)
    ref = fn(x, w)

    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    trace = thunder.last_traces(jfn)[-1]
    assert any(bsym.sym.executor is tilegym_ex for bsym in trace.bound_symbols)
