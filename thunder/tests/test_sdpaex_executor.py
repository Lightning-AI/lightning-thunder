import pytest
import torch

import thunder
from thunder.tests.framework import requiresCUDA, run_snippet
from thunder.tests.opinfos import get_opinfo
from thunder.executors.sdpaex import sdpa_ex


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16], ids=("float32", "bfloat16", "float16")
)
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_sdpa(device: str, dtype: torch.dtype):
    batch = 10
    seq_len = 128
    num_heads = 4
    dim_per_head = 32

    query = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda")
    key = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda")
    value = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda")

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    cfn = thunder.compile(fn, executors_list=[sdpa_ex])

    # Verifies the result is close to PyTorch
    thunder_result = cfn(query, key, value)
    torch_result = fn(query, key, value)

    torch.testing.assert_close(thunder_result, torch_result)

    # Verifies sdpa was called
    extrace = thunder.last_traces(cfn)[-1]
    assert any(
        bsym.sym.name == "sdpaex_grad_forward_scaled_dot_product_efficient_attention" for bsym in extrace.bound_symbols
    )


def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)
    torch.testing.assert_close(thunder_result, torch_result, equal_nan=True, atol=1e-3, rtol=1e-4)

    last_trace = thunder.last_traces(op)[-1]
    fused_sdpa_kernels = [
        "sdpaex_grad_forward_scaled_dot_product_efficient_attention",
        "sdpafx_grad_forward_scaled_dot_product_efficient_attention",
    ]
    assert any(bsym.sym.name in fused_sdpa_kernels for bsym in last_trace.bound_symbols)


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_sdpa_torch_consistency(device: str, dtype: torch.dtype):
    from thunder.executors.sdpaex import _scaled_dot_product_attention_checker

    op = get_opinfo("grad_forward_scaled_dot_product_attention")

    def fn(*args, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(*args, **kwargs)

    cfn = thunder.compile(fn, executors_list=[sdpa_ex])
    for sample in op.sample_input_generator(op, device, dtype, requires_grad=False):
        result = run_snippet(
            snippet_torch_consistency,
            op,
            device,
            dtype,
            cfn,
            op.torch_reference,
            sample,
        )

        if result is not None:
            return result
