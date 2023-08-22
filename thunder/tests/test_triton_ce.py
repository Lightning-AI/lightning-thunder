from typing import Any

import torch
from torch.testing import assert_close

import thunder
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes
import thunder.executors as executors

import thunder.torch as ltorch

from thunder.tests.opinfos import OpInfo, get_opinfo
from thunder.tests.framework import instantiate, requiresCUDA, requiresTriton, run_snippet, Executor, ops

from lightning_utilities.core.imports import package_available

TRITON_AVAILABLE = package_available("triton")

triton: None | Any = None
if TRITON_AVAILABLE:
    import triton
    from thunder.executors.triton_crossentropy import deregister_triton_entropyex, register_triton_entropyex


# NOTE This test modifies the global executor map, so it technically should not
# be run in parallel with other tests
@instantiate(dtypes=(thunder.float32,), devicetypes=(thunder.devices.DeviceType.CUDA,))
@requiresCUDA
@requiresTriton
def test_triton_cross_entropy(executor, device, dtype):
    try:
        register_triton_entropyex()
        logits = torch.randn([2048, 50257], device=device, dtype=thunder.torch.to_torch_dtype(dtype))
        labels = torch.randint(0, 50257, [2048], device=device)
        reduction = "sum"
        ignore_index = labels[5].item()
        weight = torch.rand(50257, device=device, dtype=thunder.torch.to_torch_dtype(dtype), requires_grad=False)
        expected = torch.nn.functional.cross_entropy(
            logits, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

        def test(logits, labels, weight, reduction, ignore_index):
            return thunder.torch.cross_entropy(
                logits, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
            )

        ctest = thunder.compile(test, executors_list=["triton_crossentropy"] + executor.executors_list())
        actual = ctest(logits, labels, weight, reduction, ignore_index)
        torch.testing.assert_close(actual, expected)
        last_trace = thunder.last_traces(ctest)[-1]
        assert any(bsym.sym.name == "triton_cross_entropy" for bsym in last_trace.bound_symbols)
    finally:
        deregister_triton_entropyex()


def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)

    assert_close(thunder_result, torch_result, equal_nan=True, atol=1e-3, rtol=1e-4)


@requiresCUDA
@requiresTriton
def test_triton_cross_entropy_vs_torch_consistency():
    try:
        register_triton_entropyex()
        opinfo = get_opinfo("cross_entropy")

        def foo(*args, **kwargs):
            return torch.nn.functional.cross_entropy(*args, **kwargs)

        ce = thunder.compile(foo)

        device = "cuda"
        for dtype in (torch.float32, torch.float64):
            for sample in opinfo.reference_inputs(device=device, dtype=dtype, requires_grad=False):
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
    finally:
        deregister_triton_entropyex()
