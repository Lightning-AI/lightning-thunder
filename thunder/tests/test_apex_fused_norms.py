import pytest
import torch
from torch.testing import assert_close

fused_layer_norm_cuda = pytest.importorskip("fused_layer_norm_cuda")

from torch.distributed import is_available
from thunder.executors.apexex import apex_ex
import thunder


# See https://github.com/NVIDIA/apex/issues/1853
@pytest.mark.skipif(not is_available(), reason="torch.distributed is not available")
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("memory_efficient", [True, False])
def test_apex_fused_rms_norm(requires_grad, memory_efficient):
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction

    def fn(x, weight, normalized_shape, eps):
        return FusedRMSNormAffineMixedDtypesFunction.apply(x, weight, normalized_shape, eps, memory_efficient)

    device = "cuda"
    normalized_shape = (3, 2)
    x = torch.randn(4, 5, *normalized_shape, device=device, requires_grad=requires_grad)
    weight = torch.randn(*normalized_shape, device=device, requires_grad=requires_grad)
    eps = 1e-5

    expected = fn(x, weight, normalized_shape, eps)
    jfn = thunder.jit(fn, executors=[apex_ex])
    actual = jfn(x, weight, normalized_shape, eps)

    assert_close(actual, expected)

    if requires_grad:
        grad_output = torch.rand_like(actual)
        actual_grad = torch.autograd.grad(actual, [x, weight], grad_output)
        expected_grad = torch.autograd.grad(expected, [x, weight], grad_output)

        assert_close(actual_grad, expected_grad)


# See https://github.com/NVIDIA/apex/issues/1853
@pytest.mark.skipif(not is_available(), reason="torch.distributed is not available")
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("memory_efficient", [True, False])
def test_apex_fused_rms_norm_autoregister(requires_grad, memory_efficient):
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction

    def fn(x, weight, normalized_shape, eps):
        return FusedRMSNormAffineMixedDtypesFunction.apply(x, weight, normalized_shape, eps, memory_efficient)

    device = "cuda"
    normalized_shape = (3, 2)
    x = torch.randn(4, 5, *normalized_shape, device=device, requires_grad=requires_grad)
    weight = torch.randn(*normalized_shape, device=device, requires_grad=requires_grad)
    eps = 1e-5

    expected = fn(x, weight, normalized_shape, eps)
    jfn = thunder.jit(fn, executors=())
    actual = jfn(x, weight, normalized_shape, eps)

    assert_close(actual, expected)

    if requires_grad:
        grad_output = torch.rand_like(actual)
        actual_grad = torch.autograd.grad(actual, [x, weight], grad_output)
        expected_grad = torch.autograd.grad(expected, [x, weight], grad_output)

        assert_close(actual_grad, expected_grad)
