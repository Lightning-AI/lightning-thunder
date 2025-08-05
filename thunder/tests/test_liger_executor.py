import pytest

import torch
from torch.testing import assert_close

import litgpt

import thunder
import thunder.core
from thunder.tests.litgpt_model import GPT
from thunder.tests.framework import requiresCUDA
from thunder.tests import litgpt_model

from thunder.executors.ligerex import liger_ex


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_rms_norm(device: str, dtype: torch.dtype):
    hidden_size = 64

    x = torch.randn(32, 10, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.ones(hidden_size, device=device, dtype=dtype, requires_grad=True)
    eps = 1e-5

    def fn(x, weight, eps):
        return torch.nn.functional.rms_norm(x, (hidden_size,), weight, eps)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, weight, eps)
    torch_result = fn(x, weight, eps)

    go = torch.randn_like(torch_result)
    grad_ref, grad_ref_weight = torch.autograd.grad(torch_result, (x, weight), go)
    grad_res, grad_res_weight = torch.autograd.grad(thunder_result, (x, weight), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)
    assert_close(grad_ref_weight, grad_res_weight)

    assert thunder.executors.ligerex.liger_rms_norm_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_rms_norm_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_layer_norm(device: str, dtype: torch.dtype):
    hidden_size = 64

    x = torch.randn(32, 10, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.ones(hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.zeros(hidden_size, device=device, dtype=dtype, requires_grad=True)
    eps = 1e-5

    def fn(x, weight, bias, eps):
        return torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, weight, bias, eps)
    torch_result = fn(x, weight, bias, eps)

    go = torch.randn_like(torch_result)
    grad_ref, grad_ref_weight = torch.autograd.grad(torch_result, (x, weight), go)
    grad_res, grad_res_weight = torch.autograd.grad(thunder_result, (x, weight), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)
    assert_close(grad_ref_weight, grad_res_weight)

    assert thunder.executors.ligerex.liger_layer_norm_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_layer_norm_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_cross_entropy(device: str, dtype: torch.dtype):
    hidden_size = 64
    batch_size = 32
    num_classes = 100

    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_classes, hidden_size, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, num_classes, (batch_size,), device=device)

    def fn(x, target, weight):
        logits = torch.matmul(x, weight.t())
        return torch.nn.functional.cross_entropy(logits, target)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, target, weight)
    torch_result = fn(x, target, weight)

    go = torch.randn_like(torch_result)
    grad_ref = torch.autograd.grad(torch_result, (x,), go)
    grad_res = torch.autograd.grad(thunder_result, (x,), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)

    assert thunder.executors.ligerex.liger_cross_entropy_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_cross_entropy_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_group_norm(device: str, dtype: torch.dtype):
    num_groups = 3
    num_channels = 6
    hidden_size = 64

    x = torch.randn(32, num_channels, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.ones(num_channels, device=device, dtype=dtype, requires_grad=True)
    bias = torch.zeros(num_channels, device=device, dtype=dtype, requires_grad=True)

    def fn(x, n, w, b):
        return torch.nn.functional.group_norm(x, n, w, b, eps=1e-5)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, num_groups, weight, bias)
    torch_result = fn(x, num_groups, weight, bias)

    go = torch.randn_like(torch_result)
    grad_ref, grad_ref_weight, grad_ref_bias = torch.autograd.grad(torch_result, (x, weight, bias), go)
    grad_res, grad_res_weight, grad_res_bias = torch.autograd.grad(thunder_result, (x, weight, bias), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)
    assert_close(grad_ref_weight, grad_res_weight)
    assert_close(grad_ref_bias, grad_res_bias, rtol=1e-5, atol=1e-5)

    assert thunder.executors.ligerex.liger_group_norm_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_group_norm_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }
