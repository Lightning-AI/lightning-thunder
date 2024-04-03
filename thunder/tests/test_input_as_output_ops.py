from functools import partial

import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.dtypes as datatypes
import thunder.torch as ttorch
from thunder.tests.framework import instantiate


@instantiate()
def test_input_as_output_prim_fwd(executor, device, dtype):
    def torch_foo(x, y):
        z = x * y
        z = z + z
        z = x + z
        x.copy_(z)
        return y

    def foo(x, y):
        z = x * y
        z = z + z
        z = x + z
        thunder.core.prims.input_as_output(z, x)
        return y

    traced_nvfuser_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=False)
    b = make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=False)
    a1 = a.detach().clone()
    b1 = b.detach().clone()
    thunder_result = traced_nvfuser_foo(a, b)
    torch_result = torch_foo(a1, b1)

    custom_comparator = (
        partial(assert_close, atol=1e-2, rtol=1e-2)
        if dtype in (datatypes.bfloat16, datatypes.float16)
        else assert_close
    )
    assert_close(thunder_result, torch_result)
    custom_comparator(a, a1)


@instantiate(dtypes=(datatypes.floating,))
def test_input_as_output_prim_bwd(executor, device, dtype):
    def torch_foo(x, y):
        z = x * y
        z = z * x
        x.copy_(z)
        p = y * y
        return p

    def foo(x, y):
        z = x * y
        z = z * x
        thunder.core.prims.input_as_output(z, x)
        p = y * y
        return p

    traced_nvfuser_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    # inplace updated input can not require grad(RuntimeError: a leaf Variable that requires grad is being used in an in-place operation)
    a = make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=False)
    b = make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=True)
    a1 = a.detach().clone()
    b1 = b.detach().clone()
    b1.requires_grad_()

    thunder_result = traced_nvfuser_foo(a, b)
    torch_result = torch_foo(a1, b1)
    assert_close(thunder_result, torch_result)
    custom_comparator = (
        partial(assert_close, atol=1e-2, rtol=1e-2)
        if dtype in (datatypes.bfloat16, datatypes.float16)
        else assert_close
    )
    custom_comparator(a, a1)

    g = torch.ones_like(thunder_result)
    thunder_result.backward(g)

    g1 = torch.ones_like(torch_result)
    torch_result.backward(g1)
    assert_close(g, g1)
    assert_close(a.grad, a1.grad)
    assert_close(b.grad, b1.grad)
