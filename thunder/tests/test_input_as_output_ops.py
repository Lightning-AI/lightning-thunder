from functools import partial

import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.torch as ttorch
from thunder.tests.framework import instantiate

import pytest


@instantiate(dtypes=(thunder.core.dtypes.inexact,))
def test_input_as_output_prim(executor, device, dtype):
    # Tests passing all arguments as function inputs
    def torch_foo(x, y):
        z = x * y
        z = z * z
        x.copy_(z)
        # y = y*y
        return y
    def foo(x, y):
        z = torch.mul(x,y)
        z = torch.mul(z,z)
        thunder.core.prims.input_as_output(z,x)
        # TODO error
        # thunder/core/transforms.py:3923: in backward_fn
        #    env = reconstruct_forward_env_for_backward(trace, saved_for_backward)
        # IndexError: tuple index out of range
        # y = y*y  
        return y
    
    traced_nvfuser_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    # inplace updated input can not require grad(RuntimeError: a leaf Variable that requires grad is being used in an in-place operation)
    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=False)
    b = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=True)
    a1=a.detach().clone()
    b1=b.detach().clone()
    b1.requires_grad_()

    thunder_result = traced_nvfuser_foo(a, b)
    torch_result = torch_foo(a1, b1)
    assert_close(thunder_result, torch_result)
    assert_close(a, a1)

    g = torch.ones_like(thunder_result)
    thunder_result.backward(g)

    g1 = torch.ones_like(torch_result)
    torch_result.backward(g)
    assert_close(g, g1)
    assert_close(a.grad, a1.grad)
    assert_close(b.grad, b1.grad)