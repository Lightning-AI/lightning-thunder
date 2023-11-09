from typing import Any

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder import dtypes
from thunder.tests.framework import instantiate, requiresCUDA, ops, run_snippet, NOTHING, TorchExecutor
from thunder.tests.opinfos import OpInfo, get_opinfo
import thunder.core.devices as devices

cudnn = pytest.importorskip("cudnn")
from thunder.executors.cudnnex import cudnn_ex
from thunder.executors.cudnn_layernormex import cudnn_layernorm_ex


# WARNING: cudnn executor is experimental. Tests that use cudnn might fail.\n
# Issue for tracking support: https://github.com/Lightning-AI/lightning-thunder/issues/880
# NOTE This test modifies the global executor map, so it technically should not
# be run in parallel with other tests
@requiresCUDA
def test_cudnn_sdpa():
    # expect sdpa to fail for 8.9.2 and below
    if cudnn.backend_version() <= 8902:
        pytest.xfail("Only interleaved layout is supported pre 8.9.2.")

    for dtype in (thunder.float16, thunder.bfloat16):
        b, h, s_q, s_kv, d_q, d_v = 8, 8, 256, 256, 64, 64
        shape_Q = (b, h, s_q, d_q)
        shape_K = (b, h, s_kv, d_q)
        shape_V = (b, h, s_q, d_v)

        query = 1 * (torch.randn(shape_Q, dtype=thunder.torch.to_torch_dtype(dtype), device="cuda") - 0.5)
        key = 2 * (torch.randn(shape_K, dtype=thunder.torch.to_torch_dtype(dtype), device="cuda") - 0.5)
        value = 3 * (torch.randn(shape_V, dtype=thunder.torch.to_torch_dtype(dtype), device="cuda") - 0.5)
        is_causal = False
        attn_mask = torch.randn(
            s_q, s_kv, requires_grad=False, device="cuda", dtype=thunder.torch.to_torch_dtype(dtype)
        )

        expected = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal, attn_mask=attn_mask
        )

        def test(query, key, value, is_causal=False, attn_mask=None):
            return thunder.torch.scaled_dot_product_attention(
                query, key, value, is_causal=is_causal, attn_mask=attn_mask
            )

        ctest = thunder.compile(test, executors_list=[cudnn_ex])
        actual = ctest(query, key, value, is_causal=is_causal, attn_mask=attn_mask)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
        last_trace = thunder.last_traces(ctest)[-1]
        assert any(bsym.sym.name == "cudnn_sdpa" for bsym in last_trace.bound_symbols)


# NOTE These wrappers are necessary because preprocessing cannot compile the function directly
def layer_norm_wrapper(*args, **kwargs):
    return thunder.torch.layer_norm(*args, **kwargs)


def spda_wrapper(*args, **kwargs):
    return thunder.torch.scaled_dot_product_attention(*args, **kwargs)


op_name_to_fn = {
    "layer_norm": layer_norm_wrapper,
    "scaled_dot_product_attention": spda_wrapper,
}


def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)
    # TODO: Pass a custom_comparator which has higher tol for bf16 cases
    assert_close(thunder_result, torch_result, equal_nan=True, atol=0.0625, rtol=5e-2)


# WARNING: cudnn executor is experimental. Tests that use cudnn might fail.\n
# Issue for tracking support: https://github.com/Lightning-AI/lightning-thunder/issues/880
# TODO Make it easier for executors to write tests like this, including writing them out-of-tree
# TODO The executor passed below is just a "dummy" that actually gets ignored -- we should provide
#   a way to use decorators like @ops without a particular executor
@ops(
    (
        get_opinfo("layer_norm"),
        get_opinfo("scaled_dot_product_attention"),
    ),
    supported_devicetypes=(devices.DeviceType.CUDA,),
    supported_dtypes=(dtypes.float16, dtypes.bfloat16),
    supported_executors=(TorchExecutor,),
)
def test_cudnn_vs_torch_consistency(op, device, dtype, *_):
    # expect layer_norm to fail for 8.9.3 and below
    if op.name == "layer_norm":
        if cudnn.backend_version() <= 8903:
            pytest.xfail("Only fp32 weight/bias supported pre 8.9.3.")

    # expect sdpa to fail for 8.9.2 and below
    if op.name == "scaled_dot_product_attention":
        if cudnn.backend_version() <= 8902:
            pytest.xfail("Only interleaved layout is supported pre 8.9.2.")

    for sample in op.reference_inputs(device, dtype, requires_grad=False):
        cfn = thunder.compile(op_name_to_fn[op.name], executors_list=[cudnn_ex, cudnn_layernorm_ex])

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
