from functools import partial
import itertools

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

    def foo_copy_(x, y):
        z = x + y
        # NOTE: nvfuserex doesn't support `return z`, i.e. the copy_from argument
        o = thunder.core.prims.copy_(z, x)
        return o

    def foo_copy_to_out_(x, y):
        z = x + y
        o = thunder.core.prims.copy_to_out_(z, out=x)
        return o

    tdtype = ttorch.to_torch_dtype(dtype)

    for thunder_foo in [foo_copy_, foo_copy_to_out_]:
        traced_foo = executor.make_callable(thunder_foo)
        x = make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=False)
        x_ref = x.detach().clone()
        y = make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=False)
        y_ref = y.detach().clone()

        thunder_result = traced_foo(x, y)
        torch_result = torch_foo(x_ref, y_ref)
        assert_close(thunder_result, torch_result)
        assert_close([x, y], [x_ref, y_ref])


@instantiate(dtypes=datatypes.float_math_dtypes)
def test_prim_inplace_copy_bwd(executor, device, dtype):
    def torch_foo(x, y):
        z = x * y
        z = z * x
        o = x.copy_(z)
        p = y * y
        return p

    def foo_copy_(x, y):
        z = x * y
        z = z * x
        o = thunder.core.prims.copy_(z, x)
        p = y * y
        return p

    def foo_copy_to_out_(x, y):
        z = x * y
        z = z * x
        o = thunder.core.prims.copy_to_out_(z, out=x)
        p = y * y
        return p

    tdtype = ttorch.to_torch_dtype(dtype)
    for thunder_foo in [foo_copy_, foo_copy_to_out_]:
        traced_foo = executor.make_callable(thunder_foo)
        x = make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=False)
        x_ref = x.detach().clone()
        y = make_tensor((4, 4), device=device, dtype=tdtype, requires_grad=True)
        y_ref = y.detach().clone().requires_grad_()

        thunder_result = traced_foo(x, y)
        torch_result = torch_foo(x_ref, y_ref)
        assert_close(thunder_result, torch_result)
        assert_close([x, y], [x_ref, y_ref])

        custom_comparator = (
            partial(assert_close, atol=1e-2, rtol=1e-2)
            if dtype in (datatypes.bfloat16, datatypes.float16)
            else assert_close
        )
        custom_comparator(x, x_ref)

        g = torch.ones_like(thunder_result)
        thunder_result.backward(g)
        g_ref = torch.ones_like(torch_result)
        torch_result.backward(g_ref)
        assert_close(g, g_ref)
        assert_close(y.grad, y_ref.grad)


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
    copy_prims = [thunder.core.prims.copy_, lambda x, y: thunder.core.prims.copy_to_out_(x, out=y)]
    for copy_prim in copy_prims:

        def func0(x, y):
            z = x * y
            x = copy_prim(z, x)
            return x + y

        def func1(x, y):
            z = x * y
            o1 = copy_prim(z, x)
            o2 = copy_prim(y, x)
            return x, o1, o2

        def func2(x, y):
            z = x * y
            o1 = copy_prim(z, x)
            o2 = copy_prim(x, y)
            return y, o1, o2

        def func3(x, y):
            z = x * y
            o1 = copy_prim(z, x)
            o2 = copy_prim(o1, y)
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
def test_copy_to_out_sanity_check_on_computed(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)
    a = make_tensor((4, 4), device=device, dtype=tdtype)
    b = make_tensor((4, 4), device=device, dtype=tdtype)
    a_ref = a.detach().clone()
    b_ref = b.detach().clone()

    def torch_good(x, y):
        z = x * y
        o = x.copy_(z)
        return o

    def good(x, y):
        z = x * y
        o = thunder.core.prims.copy_to_out_(z, out=x)
        return o

    def bad1(x, y):
        z = x * y
        o = thunder.core.prims.copy_to_out_(z, out=x)
        return o, z

    def bad2(x, y):
        z = x * y
        o = thunder.core.prims.copy_to_out_(z, out=x)
        return o + z

    def bad3(x, y):
        o = thunder.core.prims.copy_to_out_(y, out=x)
        return o

    def bad4(x, y):
        z = x * y
        o = thunder.core.prims.copy_to_out_(z, out=x)
        return o, torch.concat((z, z))  # not fused

    def bad5(x, y):
        x2 = torch.concat((x, x))
        y2 = torch.concat((y, y))  # not fused
        o = thunder.core.prims.copy_to_out_(y2, out=x2)
        return o

    traced_good = executor.make_callable(good)
    out = traced_good(a, b)
    out_ref = torch_good(a_ref, b_ref)
    assert_close([a, b, out], [a_ref, b_ref, out_ref])

    for foo in [bad1, bad2, bad3, bad4, bad5]:
        print(foo.__name__)
        traced_foo = executor.make_callable(foo)
        with pytest.raises(
            NotImplementedError,
            match=r"If you are sure you don't want to use this check, it can be disabled by setting `disable_inplace_copy_check=True` in `thunder.jit`.$",
        ):
            traced_foo(a, b)


@instantiate(executors=(nvFuserExecutor,), dtypes=(thunder.float32,))
def test_nvfuser_add_output_alias(executor, device, dtype):
    tdtype = ttorch.to_torch_dtype(dtype)

    def check(func, test_close=True):
        traced_foo = executor.make_callable(func, disable_inplace_copy_check=True)
        a = make_tensor((4, 4), device=device, dtype=tdtype)
        a_ref = a.clone()

        outs = traced_foo(a)
        outs_ref = func(a_ref)

        if test_close:
            for o, o_ref in zip(outs, outs_ref):
                assert_close(o, o_ref)

        for (o1, o1_ref), (o2, o2_ref) in itertools.combinations(zip(outs, outs_ref), 2):
            assert (o1.data_ptr() == o2.data_ptr()) == (o1_ref.data_ptr() == o2_ref.data_ptr())

    def func1(T0):
        T1 = torch.sin(T0)
        T0.copy_(T1)
        T2 = torch.cos(T1)
        T0.copy_(T2)
        return T0, T1, T2

    check(func1)

    def func2(T0):
        T1 = torch.sin(T0)
        T0.copy_(T1)
        T0.add_(1)
        return T0, T1

    # T0.copy_(T1) does not necessarily take effect before T0.add_(T1),
    # because nvFuser does not establish dependency between fd.add_output(T1, T0) and fd.ops.add(T0, 1).
    # We would need to functionalize T0.copy_
    check(func2, test_close=False)

    def func3(T0):
        T0.add_(1)
        T1 = torch.sin(T0)
        T0.copy_(T1)
        return T0, T1

    # Functionalization replaces the source of copy_ with the intermediate result of the addition.
    # nvFuser raises an error for this
    # check(func3)
