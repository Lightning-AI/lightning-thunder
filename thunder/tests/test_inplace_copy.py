from functools import partial

import pytest
import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.dtypes as datatypes
import thunder.torch as ttorch
from thunder.tests.framework import instantiate, nvFuserExecutor


@instantiate()
def test_prim_inplace_copy_fwd(executor, device, dtype):
    def torch_foo(x, y):
        z = x * y
        z = z + z
        z = x + z
        o = x.copy_(z)
        return o

    def foo(x, y):
        z = x * y
        z = z + z
        z = x + z
        # NOTE: nvfuserex doesn't support `return z`, i.e. the copy_from argument
        o = thunder.core.prims.copy_(z, x)
        return o

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
    custom_comparator(thunder_result, torch_result)
    custom_comparator(a, a1)


@instantiate(dtypes=(datatypes.floating,))
def test_prim_inplace_copy_bwd(executor, device, dtype):
    def torch_foo(x, y):
        z = x * y
        z = z * x
        o = x.copy_(z)
        p = y * y
        return p

    def foo(x, y):
        z = x * y
        z = z * x
        o = thunder.core.prims.copy_(z, x)
        p = y * y
        return p

    traced_nvfuser_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
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
    assert_close(b.grad, b1.grad)


@instantiate(dtypes=(thunder.float32, thunder.float64))
def test_batch_norm_running_stats(executor, device, dtype):
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1_bn = nn.BatchNorm3d(2, track_running_stats=True)
            # To address the failure, use a workaround since `add_` is utilized in `nn.BatchNorm3d` when `num_batches_tracked` is not None.
            self.dense1_bn.num_batches_tracked = None

        def forward(self, x):
            x = self.dense1_bn(x)
            return x

    tdtype = ttorch.to_torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device, requires_grad=True)
    net = Net().train().to(device=device, dtype=tdtype)
    torch_net = Net().train().to(device=device, dtype=tdtype)
    thunder_net = executor.make_callable(net)
    x = make((3, 2, 3, 4, 12))
    x1 = x.detach().clone()
    x1.requires_grad_()
    thunder_out = thunder_net(x)
    thunder_out.sum().backward()
    torch_out = torch_net(x1)
    torch_out.sum().backward()

    assert_close(thunder_out, torch_out)
    assert_close(net.state_dict()["dense1_bn.running_mean"], torch_net.state_dict()["dense1_bn.running_mean"])
    assert_close(net.state_dict()["dense1_bn.running_var"], torch_net.state_dict()["dense1_bn.running_var"])
    assert_close(x.grad, x1.grad)


@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_inplace_copy_sanity_check(executor, device, dtype):
    def func1(x, y):
        z = x * y
        x = thunder.core.prims.copy_(z, x)
        return x + y

    def func2(x, y):
        z = x * y
        thunder.core.prims.copy_(z, x)
        thunder.core.prims.copy_(y, x)
        return x

    def func3(x, y):
        z = x * y
        thunder.core.prims.copy_(z, x)
        thunder.core.prims.copy_(x, y)
        return y

    def func4(x, y):
        z = x * y
        o = thunder.core.prims.copy_(z, x)
        thunder.core.prims.copy_(o, y)
        return y

    for foo in (func1, func2, func3, func4):
        traced_foo = executor.make_callable(foo)

        tdtype = ttorch.to_torch_dtype(dtype)
        a = make_tensor((4, 4), device=device, dtype=tdtype)
        b = make_tensor((4, 4), device=device, dtype=tdtype)
        with pytest.raises(
            NotImplementedError,
            match=r"If you are sure you don't want to use this check, it can be disabled by setting `disable_inplace_copy_check=True` in `thunder.jit`.$",
        ):
            traced_foo(a, b)
