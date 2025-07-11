from functools import partial

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.tests.framework import requiresCUDA, run_snippet, IN_CI, assert_closer
from thunder.tests.opinfos import get_opinfo
from thunder.core.transforms import grad

xentropy_cuda = pytest.importorskip("xentropy_cuda")
from thunder.executors.apexex import apex_ex


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_apex_cross_entropy(device: str, dtype: torch.dtype):
    logits = torch.randn([2048, 50257], device=device, dtype=thunder.torch.to_torch_dtype(dtype))
    labels = torch.randint(0, 50257, [2048], device=device)

    def fn(logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, reduction="mean", ignore_index=-1)

    cfn = thunder.jit(fn, executors=[apex_ex])

    # Verifies the result is close to PyTorch
    thunder_result = cfn(logits, labels)
    torch_result = fn(logits, labels)

    assert_close(thunder_result, torch_result)

    # Verifies apex_cross_entropy was called
    extrace = thunder.last_traces(cfn)[-1]
    assert any(bsym.sym.name == "apex_cross_entropy" for bsym in extrace.bound_symbols)


def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)
    assert_close(thunder_result, torch_result, equal_nan=True, atol=1e-3, rtol=1e-4)

    last_trace = thunder.last_traces(op)[-1]
    assert any(bsym.sym.name == "apex_cross_entropy" for bsym in last_trace.bound_symbols)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=("float16", "float32"))
@pytest.mark.parametrize("device,", ["cuda"])
def test_apex_torch_consistency(device: str, dtype: torch.dtype):
    from thunder.executors.apex_entropyex_impl import _cross_entropy_checker

    op = get_opinfo("cross_entropy")

    def fn(*args, **kwargs):
        return torch.nn.functional.cross_entropy(*args, **kwargs)

    cfn = thunder.jit(fn, executors=[apex_ex])
    at_least_one_supported_input = False

    # NOTE reference inputs take a long time to run in CI, so this uses sample inputs in CI
    input_generator = op.reference_inputs if not IN_CI else op.sample_inputs

    for sample in input_generator(device, dtype, requires_grad=False):
        if not _cross_entropy_checker(*sample.args, **sample.kwargs):
            continue

        at_least_one_supported_input = True
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

    if not at_least_one_supported_input:
        raise ValueError("No supported inputs were generated by the OpInfo")


@pytest.mark.skip(reason="too much memory (https://github.com/Lightning-AI/lightning-thunder/issues/392)")
@pytest.mark.xfail(reason="this was not tested yet, but it should be fixed")  # todo
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], ids=("float16", "bfloat16", "float32")
)
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_apex_cross_entropy_backward(device, dtype):
    from thunder.core.transforms import value_and_grad
    from thunder.common import CompileData
    from thunder.core.compile_data import compile_data_and_stats

    logits = torch.randn([2048, 50257], device=device, dtype=thunder.torch.to_torch_dtype(dtype), requires_grad=True)
    labels = torch.randint(0, 50257, [2048], device=device)

    # -1 is supported by apex cross entropy but 1 is not. The case of 1 is
    # used to test that the conditional rules are working correctly and that
    # the apex cross entropy is not used
    ignore_indices = (-1, 1)

    for ignore_index in ignore_indices:

        @value_and_grad
        def test(logits, labels):
            return thunder.torch.cross_entropy(logits, labels, reduction="mean", ignore_index=ignore_index)

        cd = CompileData(
            fn=test,
            executors_list=[apex_ex],
            disable_preprocessing=True,
        )
        with compile_data_and_stats(cd, None):
            initial_trace = thunder.trace()(test, logits, labels)

        from thunder.executors.apex_entropyex import apex_xentropy, apex_xentropy_bwd

        # This is a workaround for the issue with python_ctx replacing symbols
        # with their "call_ctx" values which are not traceable and accept only
        # regular torch tensors
        initial_trace.python_ctx = lambda: {
            "apex_cross_entropy": apex_xentropy,
            "apex_cross_entropy_backward": apex_xentropy_bwd,
        }

        ctest = thunder.jit(
            initial_trace.python_callable(),
            executors_list=[apex_ex],
            disable_torch_autograd=True,
        )
        actual = ctest(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits, labels, reduction="mean", ignore_index=ignore_index)
        expected_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[1][0], expected_grad)
        last_trace = thunder.last_traces(ctest)[-1]
        is_any_fw = any(bsym.sym.name == "apex_cross_entropy" for bsym in last_trace.bound_symbols)
        is_any_bw = any(bsym.sym.name == "apex_cross_entropy_backward" for bsym in last_trace.bound_symbols)

        if ignore_index == -1:
            assert is_any_fw
            assert is_any_bw
        else:
            assert not is_any_fw
            assert not is_any_bw


@pytest.mark.skip(reason="too much memory (https://github.com/Lightning-AI/lightning-thunder/issues/392)")
@pytest.mark.xfail(reason="this was not tested yet, but it should be fixed")  # todo
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], ids=("float16", "bfloat16", "float32")
)
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_apex_cross_entropy_phantom_grad(device, dtype):
    logits = torch.randn([2048, 50257], device=device, dtype=thunder.torch.to_torch_dtype(dtype), requires_grad=True)
    labels = torch.randint(0, 50257, [2048], device=device)

    def foo(logits, labels):
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="sum", ignore_index=-1)
        return ce

    cfoo = thunder.jit(foo, executors_list=[apex_ex])
    cfoo_grad = grad(cfoo)
    (thunder_grad,) = cfoo_grad(logits, labels)

    torch_result = foo(logits, labels)
    torch_result.backward()
    torch_grad = logits.grad
    logits.grad = None

    # Computes the reference in double precision on the CPU
    reference_logits = logits.cpu().double().detach().requires_grad_(True)
    reference_labels = labels.cpu()
    reference_result = foo(reference_logits, reference_labels)
    reference_result.backward()
    reference_grad = reference_logits.grad.cuda()
    reference_logits.grad = None

    # (mruberry) In bf16 I see the following failure:
    #   Mismatched elements: 13 / 102926336 (0.0%)
    #   Greatest absolute difference: 4.741927263568393e-05 at index (1852, 25836) (up to 1e-05 allowed)
    #   Greatest relative difference: 0.054034659671222666 at index (569, 3344) (up to 0.016 allowed)
    comp = assert_close
    if dtype in (torch.float16, torch.bfloat16):
        comp = partial(assert_close, atol=1e-4, rtol=1e-2)

    assert_closer(reference=reference_grad, candidate=thunder_grad, competitor=torch_grad, comparator=comp)

    # Verifies that apex cross entropy was used to compute the grad
    extrace = thunder.last_traces(cfoo_grad)[-1]
    is_any_fw = any(bsym.sym.name == "apex_cross_entropy" for bsym in extrace.bound_symbols)
    is_any_bw = any(bsym.sym.name == "apex_cross_entropy_backward" for bsym in extrace.bound_symbols)

    assert is_any_fw and is_any_bw

    # Tests with mean reduction
    def bar(logits, labels):
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="mean", ignore_index=-1)
        return ce

    cbar = thunder.jit(bar, executors=[apex_ex])
    cbar_grad = grad(cbar)
    (thunder_grad,) = cbar_grad(logits, labels)

    torch_result = bar(logits, labels)
    torch_result.backward()
    torch_grad = logits.grad
    logits.grad = None

    # Computes the reference in double precision on the CPU
    reference_logits = logits.cpu().double().detach().requires_grad_(True)
    reference_labels = labels.cpu()
    reference_result = bar(reference_logits, reference_labels)
    reference_result.backward()
    reference_grad = reference_logits.grad.cuda()
    reference_logits.grad = None

    assert_closer(reference=reference_grad, candidate=thunder_grad, competitor=torch_grad, comparator=comp)

    # Tests with none reduction
    def caz(logits, labels):
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none", ignore_index=-1)
        return ce.sum()

    ccaz = thunder.jit(caz, executors=[apex_ex])
    ccaz_grad = grad(ccaz)
    (thunder_grad,) = ccaz_grad(logits, labels)

    torch_result = caz(logits, labels)
    torch_result.backward()
    torch_grad = logits.grad
    logits.grad = None

    # Computes the reference in double precision on the CPU
    reference_logits = logits.cpu().double().detach().requires_grad_(True)
    reference_labels = labels.cpu()
    reference_result = caz(reference_logits, reference_labels)
    reference_result.backward()
    reference_grad = reference_logits.grad.cuda()
    reference_logits.grad = None

    assert_closer(reference=reference_grad, candidate=thunder_grad, competitor=torch_grad, comparator=comp)
