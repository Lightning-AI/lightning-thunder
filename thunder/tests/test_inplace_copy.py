from functools import partial

import pytest
import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.dtypes as datatypes
import thunder.torch as ttorch
from thunder.tests.framework import instantiate, nvFuserExecutor


@instantiate(dtypes=datatypes.all_dtypes - datatypes.float_8bit_dtypes)
def test_prim_inplace_copy_fwd(executor, device, dtype):
    def torch_foo(x, y):
        z = x + y
        o = x.copy_(z)
        return o

    def foo(x, y):
        z = x + y
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

    assert_close(thunder_result, torch_result)
    assert_close(a, a1)


@instantiate(dtypes=datatypes.float_math_dtypes)
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
    assert_close(
        net.state_dict()["dense1_bn.num_batches_tracked"], torch_net.state_dict()["dense1_bn.num_batches_tracked"]
    )
    assert_close(x.grad, x1.grad)


@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_inplace_copy_sanity_check(executor, device, dtype):
    def func0(x, y):
        z = x * y
        x = thunder.core.prims.copy_(z, x)
        return x + y

    def func1(x, y):
        z = x * y
        o1 = thunder.core.prims.copy_(z, x)
        o2 = thunder.core.prims.copy_(y, x)
        return x, o1, o2

    def func2(x, y):
        z = x * y
        o1 = thunder.core.prims.copy_(z, x)
        o2 = thunder.core.prims.copy_(x, y)
        return y, o1, o2

    def func3(x, y):
        z = x * y
        o1 = thunder.core.prims.copy_(z, x)
        o2 = thunder.core.prims.copy_(o1, y)
        return y, o2

    for foo in (func0, func1, func2, func3):
        traced_foo = executor.make_callable(foo)

        tdtype = ttorch.to_torch_dtype(dtype)
        a = make_tensor((4, 4), device=device, dtype=tdtype)
        b = make_tensor((4, 4), device=device, dtype=tdtype)
        with pytest.raises(
            NotImplementedError,
            match=r"If you are sure you don't want to use this check, it can be disabled by setting `disable_inplace_copy_check=True` in `thunder.jit`.$",
        ):
            traced_foo(a, b)


@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_inplace_copy_dst_copy_returned_issue_1109(executor, device, dtype):
    def func(T0):
        T1 = torch.sin(T0)
        T0.copy_(T1)  # destination.copy_(source)
        T2 = torch.cos(T1)
        T0.copy_(T2)
        # T1 & T2 should be returned as separate buffer, instead of sharing
        # storage with T0
        return T1, T2

    tdtype = ttorch.to_torch_dtype(dtype)
    # This pattern is unsafe in general. Disabling sanity check to silence
    # exception for testing
    traced_foo = executor.make_callable(func, disable_inplace_copy_check=True)
    a = make_tensor((4, 4), device=device, dtype=tdtype)
    a_ref = a.clone()

    o_thunder = traced_foo(a)
    o_eager = func(a_ref)

    assert_close(a_ref, a)
    for o, o_ref in zip(o_thunder, o_eager):
        assert_close(o, o_ref)
