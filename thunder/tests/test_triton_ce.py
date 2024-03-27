from typing import Any
from numbers import Number

import pytest

import torch
from torch.testing import assert_close

import thunder


from thunder.tests.opinfos import get_opinfo
from thunder.tests.framework import instantiate, requiresCUDA, requiresTriton, run_snippet, IN_CI
from thunder.executors import triton_utils


from lightning_utilities.core.imports import package_available

TRITON_AVAILABLE = package_available("triton")

# Requires triton 2.1 or greater
triton: None | Any = None
min_triton_version = "2.1"
if triton_utils.is_triton_version_at_least(min_triton_version):
    from thunder.executors.triton_crossentropy import triton_ex


# NOTE This test modifies the global executor map, so it technically should not
# be run in parallel with other tests
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
    ids=("float16", "bfloat16", "float32", "float64"),
)
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
@requiresTriton
def test_triton_cross_entropy(device, dtype):
    if IN_CI:
        pytest.skip("Currently these tests are skipped in CI for speed")

    logits = torch.randn([2048, 50257], device=device, dtype=dtype)
    labels = torch.randint(0, 50257, [2048], device=device)
    reduction = "sum"
    ignore_index = labels[5].item()
    weight = torch.rand(50257, device=device, dtype=dtype, requires_grad=False)
    expected = torch.nn.functional.cross_entropy(
        logits, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
    )

    def test(logits, labels, weight, reduction, ignore_index):
        return thunder.torch.cross_entropy(
            logits, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

    ctest = thunder.jit(test, executors=[triton_ex])
    actual = ctest(logits, labels, weight, reduction, ignore_index)
    torch.testing.assert_close(actual, expected)
    last_trace = thunder.last_traces(ctest)[-1]
    assert any(bsym.sym.name == "triton_crossentropy" for bsym in last_trace.bound_symbols)


def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)
    torch.cuda.synchronize()

    # Sets atol and rtol to looser tolerances than assert_close's defaults
    atol: Number = 1e-1
    rtol: Number = 1.3e-6
    assert_close(thunder_result, torch_result, equal_nan=True, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
    ids=("float16", "bfloat16", "float32", "float64"),
)
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
@requiresTriton
def test_triton_cross_entropy_vs_torch_consistency(device, dtype):
    if IN_CI:
        pytest.skip("Currently these tests are skipped in CI for speed")
    if dtype == torch.float16 or dtype == torch.bfloat16:
        pytest.skip("Currently skipping float16 and bfloat16 due to numerical accuracy")

    opinfo = get_opinfo("cross_entropy")

    def foo(*args, **kwargs):
        return torch.nn.functional.cross_entropy(*args, **kwargs)

    ce = thunder.jit(foo, executors=[triton_ex])

    # NOTE reference inputs take a long time to run in CI, so this uses sample inputs in CI
    # opinfo.reference_inputs if not IN_CI else opinfo.sample_inputs
    # reference inputs for cross_entropy contains cases not implemented in Thunder
    input_generator = opinfo.reference_inputs

    for sample in input_generator(device=device, dtype=dtype, requires_grad=False):
        result = run_snippet(
            snippet_torch_consistency,
            opinfo,
            device,
            dtype,
            ce,
            opinfo.torch_reference,
            sample,
        )
        if result is not None:
            return result
