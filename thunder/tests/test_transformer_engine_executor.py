from typing import Any

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.tests.framework import requiresCUDA

te = pytest.importorskip("transformer_engine", reason="transformer_engine was not found, skipping the tests.")
from thunder.executors.transformer_engineex import transformer_engine_ex
from transformer_engine.common import recipe
import transformer_engine.pytorch as te

# FP8 is supported on compute arch 8.9 onwards.
# Skip the tests if current hardware is not supported.
is_supported, msg = te.fp8.check_fp8_support()
if not is_supported:
    pytest.skip(msg, allow_module_level=True)

# Create an FP8 recipe.
fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)


@requiresCUDA
def test_te_linear_forward_backward():
    # Test Description:
    # Verify that `torch.nn.functional.linear` is replaced with `te_linear_*`
    # and the output as well as the gradients match for thunder compiled code.
    dtype = torch.bfloat16
    device = "cuda"

    # TE inputs (3D input)
    x_te = torch.randn(3, 768, 4096, device=device, dtype=dtype, requires_grad=True)
    te_linear1 = te.Linear(4096, 4096, params_dtype=dtype)
    te_linear2 = te.Linear(4096, 4096, params_dtype=dtype)

    # thunder inputs
    x = x_te.detach().clone()
    x.requires_grad_(True)
    w1 = te_linear1.weight.detach().clone()
    w1.requires_grad_(True)
    w2 = te_linear2.weight.detach().clone()
    w2.requires_grad_(True)

    def fn(x, w1, w2):
        o = torch.nn.functional.linear(x, w1)
        return torch.nn.functional.linear(o + x, w2)

    cfn = thunder.jit(fn, executors=[transformer_engine_ex])

    # Enable autocasting for the forward pass
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        thunder_result = cfn(x, w1, w2)

    # Enable autocasting for the forward pass
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        inter_result = te_linear1(x_te)
        te_result = te_linear2(inter_result + x_te)

    # Verifies the result is close to TE
    assert_close(thunder_result, te_result)

    grad_output = torch.randn_like(te_result)
    te_result.backward(grad_output)
    thunder_result.backward(grad_output)

    assert_close(x.grad, x_te.grad)
    assert_close(w1.grad, te_linear1.weight.grad)
    assert_close(w2.grad, te_linear2.weight.grad)

    # Verifies te_linear was called
    forward_trace = thunder.last_traces(cfn)
    backward_trace = thunder.last_backward_traces(cfn)
    assert any(bsym.sym.name.startswith("te_linear") for bsym in forward_trace[-1].bound_symbols)
    assert any(bsym.sym.name.startswith("te_functional_linear_backward") for bsym in backward_trace[-1].bound_symbols)


@requiresCUDA
def test_te_linear_forward_backward_multiple_iteration():
    # Test Description:
    # In this test, we verify whether a model using TransformerEngine Linear
    # and transformer_engine executor converge to same state.
    # Since, the FP8 operations are stateful, we want to verify that
    # our output matches over multiple iterations (where state handling comes into picture)
    dtype = torch.bfloat16
    device = "cuda"
    iterations = 5

    # TE inputs
    input_shape = (768, 4096)
    te_linear1 = te.Linear(4096, 4096, params_dtype=dtype)
    te_linear2 = te.Linear(4096, 2048, params_dtype=dtype)

    def clone_params(*params):
        return tuple(param.detach().clone() for param in params)

    # Parameters for thunder to optimize
    w1, w2, b1, b2 = clone_params(te_linear1.weight, te_linear2.weight, te_linear1.bias, te_linear2.bias)

    target_value = torch.tensor(42, dtype=dtype, device=device)

    def train_model(model, optimizer):
        # Run for `iterations`.
        for iter_n in range(iterations):
            x = torch.ones(*input_shape, device=device, dtype=dtype) + iter_n
            # Enable autocasting for the forward pass
            with te.fp8_autocast(fp8_recipe=fp8_recipe):
                result = model(x)
                loss = torch.nn.functional.mse_loss(result.sum(), target_value)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def te_model(x):
        return te_linear2(te_linear1(x))

    te_sgd_optimizer = torch.optim.SGD(list(te_linear1.parameters()) + list(te_linear2.parameters()))

    train_model(te_model, te_sgd_optimizer)

    def fn(x, w1, w2, b1, b2):
        o = torch.nn.functional.linear(x, w1, b1)
        return torch.nn.functional.linear(o, w2, b2)

    cfn = thunder.jit(fn, executors=[transformer_engine_ex])

    # Enable grad on thunder params.
    list(map(lambda t: t.requires_grad_(True), (w1, w2, b1, b2)))
    thunder_sgd_optimizer = torch.optim.SGD([w1, w2, b1, b2])

    def thunder_model(x):
        return cfn(x, w1, w2, b1, b2)

    train_model(thunder_model, thunder_sgd_optimizer)

    # Verify that the weights and biases converge to same value after few iterations.
    assert_close(w1, te_linear1.weight)
    assert_close(w2, te_linear2.weight)
    assert_close(b1, te_linear1.bias)
    assert_close(b2, te_linear2.bias)


@requiresCUDA
def test_te_linear_invalid_inputs():
    def assert_not_transformed(x, w):
        def fn(x, w):
            return torch.nn.functional.linear(x, w)

        cfn = thunder.jit(fn, executors=[transformer_engine_ex])
        cfn(x, w)
        trace = thunder.last_traces(cfn)[-1]
        assert not any(bsym.sym.name.startswith("te_linear") for bsym in trace.bound_symbols)

    # CPU is not supported.
    device = "cpu"
    x = torch.randn(16, 16, device=device)
    w = torch.randn(16, 16, device=device)
    assert_not_transformed(x, w)

    # Input shapes are not supported by TE.
    device = "cuda"
    x = torch.randn(16, 4, device=device)
    w = torch.randn(16, 4, device=device)
    assert_not_transformed(x, w)


@requiresCUDA
def test_te_with_autocast():
    def foo(x, w):
        return thunder.torch.linear(x, w)

    device = "cuda"
    x = torch.randn(16, 16, device=device, requires_grad=True)
    w = torch.randn(16, 16, device=device, requires_grad=True)

    cfunc = thunder.compile(
        thunder.core.transforms.autocast(foo, dtype=thunder.dtypes.bfloat16),
        executors_list=[transformer_engine_ex],
        disable_preprocessing=True,
    )
    cfunc(x, w)

    fwd_traces = thunder.last_traces(cfunc)
    # Verify that we have replaced `prims.linear` with `te_linear`
    assert any(bsym.sym.name.startswith("te_linear") for bsym in fwd_traces[-1].bound_symbols)
