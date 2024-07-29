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


@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_sdpa(device: str, dtype: torch.dtype):

    if not HAS_FA3:
        pytest.skip("fa3 not built")

    batch = 10
    seq_len = 128
    num_heads = 4
    dim_per_head = 32

    query = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda")
    key = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda")
    value = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda")

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    cfn = thunder.jit(fn, executors=[fa3_ex])

    # Verifies the result is close to PyTorch
    thunder_result = cfn(query, key, value)
    torch_result = fn(query, key, value)

    torch.testing.assert_close(thunder_result, torch_result)

    # Verifies sdpa was called
    extrace = thunder.last_traces(cfn)[-1]
    assert any(bsym.sym.name == "fa3_fwd" for bsym in extrace.bound_symbols)
