from collections.abc import Callable

import pytest
import torch.testing

import thunder
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
import thunder.core.prims as prims
from thunder.core.symbol import Symbol
from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.tests.framework import instantiate, ops, NOTHING, TorchExecutor, TorchCompileExecutor, nvFuserExecutor
from thunder.tests.opinfos import opinfos
from thunder.tests.test_inplace_functionalization import _inplace_opinfos


@ops(_inplace_opinfos, supported_dtypes=(dtypes.float32,))
def test_update_aliases(op, device, dtype, executor, _):
    sample = next(op.sample_inputs(device, dtype))

    # polygamma expects an int as its first argument and a tensor as its second but
    # torch.Tensor.polygamma_ wants opposite; tensor first, int second.
    # ref:
    # - https://pytorch.org/docs/stable/special.html#torch.special.polygamma
    # - https://pytorch.org/docs/stable/generated/torch.Tensor.polygamma_.html
    args = list(sample.args)
    if op.name == "polygamma_":
        args[0], args[1] = args[1], args[0]

    j_op = executor.make_callable(
        op.torch_reference, skip_inplace_alias_updates=False, skip_inplace_functionalization=True
    )
    actual = j_op(*sample.args, **sample.kwargs)
    expected = op.torch_reference(*args, **sample.kwargs)
    torch.testing.assert_close(actual, expected, equal_nan=True)


@instantiate(
    devicetypes=(devices.DeviceType.CPU,),
    dtypes=NOTHING,
)
def test_inplace_on_view(executor, device, dtype):
    def f(x, _):
        y = x.view((3, 2))
        y[0][0] = 0  # Fails on CUDA
        return x

    def g(x, _):
        y = x.view((3, 2))
        y[0][0] = 0  # Fails on CUDA
        return y

    def h(x, y):
        c = torch.exp(x)
        d = torch.tanh(y)
        e = c.view(-1)

        d.div_(x)
        e += d.flatten()

        return c, d, e

    def i(x, y):
        c = torch.exp(x)
        d = torch.tanh(y)
        e = c.view(-1)

        e += d.flatten()
        d.div_(x)

        return c, d, e

    def j(x, _):
        a = x.view(-1)
        b = x.view(-1)
        x.add_(1)
        aa = a + 1
        bb = b + 1
        return aa, bb

    for fn in [f, g, h, i, j]:
        a = make_tensor((2, 3), dtype=torch.float32, device=device)
        b = make_tensor((2, 3), dtype=torch.float32, device=device)
        a_, b_ = a.clone().detach(), b.clone().detach()
        jfn = executor.make_callable(fn, skip_inplace_alias_updates=False, skip_inplace_functionalization=True)
        actual = jfn(a, b)
        expected = fn(a_, b_)
        torch.testing.assert_close(actual, expected)


@instantiate(
    devicetypes=(devices.DeviceType.CPU,),
    dtypes=NOTHING,
)
def test_inplace_on_chunk(executor, device, dtype):
    def f(x):
        y, z = x.chunk(2, dim=1)  # Fails on CUDA with stride issues
        y.relu_()
        return y, z

    def g(x):
        y, z = x.chunk(2, dim=0)
        y.relu_()
        yy = y + 1
        return yy, z

    def h(x):
        x.relu_()
        y = x.chunk(2, dim=0)
        y[0].add_(1)
        return y

    for fn in [f, g, h]:
        a = make_tensor((2, 3), dtype=torch.float32, device=device)
        a_ = a.clone().detach()
        jfn = executor.make_callable(fn, skip_inplace_alias_updates=False, skip_inplace_functionalization=True)
        actual = jfn(a)
        expected = fn(a_)
        torch.testing.assert_close(actual, expected)


@instantiate(
    dtypes=NOTHING,
)
def test_chained_inplace(executor, device, dtype):
    def f(x):
        x.add_(1).sin_().mul_(5)
        return x

    def g(x):
        x.add_(1).sin().mul_(5)
        return x

    def h(x):
        x.exp_()
        x.sin_()
        y = x.cos()
        return y

    for fn in [f, g, h]:
        a = make_tensor((2, 3), dtype=torch.float32, device=device)
        a_ = a.clone().detach()
        jfn = executor.make_callable(
            fn, skip_inplace_alias_updates=False, skip_inplace_functionalization=True, disable_inplace_copy_check=True
        )
        actual = jfn(a)
        expected = fn(a_)
        torch.testing.assert_close(actual, expected)


@instantiate(
    dtypes=NOTHING,
)
def test_nn_module_inplace(executor, device, dtype):
    def f(x):
        m = torch.nn.ReLU(inplace=True)
        y = m(x)
        return y

    a = make_tensor((2, 3), dtype=torch.float32, device=device)
    a_ = a.clone().detach()
    jf = executor.make_callable(f, skip_inplace_alias_updates=False, skip_inplace_functionalization=True)
    actual = jf(a)
    expected = f(a_)
    torch.testing.assert_close(actual, expected)


@instantiate(
    dtypes=NOTHING,
)
def test_inplace(executor, device, dtype):
    def f(a, b):
        c = a + b
        c.add_(1)
        return c

    def g(a, b):
        c = a + b
        c.add_(1)
        d = c + c
        d.add_(1)
        c.add_(d)
        return c, d

    for fn in [f, g]:
        a = make_tensor((2, 3), dtype=torch.float32, device=device)
        b = make_tensor((2, 3), dtype=torch.float32, device=device)
        a_, b_ = a.clone().detach(), b.clone().detach()
        jfn = executor.make_callable(fn, skip_inplace_alias_updates=False, skip_inplace_functionalization=True)
        actual = jfn(a, b)
        expected = fn(a_, b_)
        torch.testing.assert_close(actual, expected)


@instantiate(
    devicetypes=(devices.DeviceType.CPU,),
    dtypes=NOTHING,
)
def test_aliased_input(executor, device, dtype):
    def f(x, y, z):
        return y.exp_().add(x) + z.exp()  # Fails on CUDA because operations have been reordered.

    a = make_tensor((2, 1, 2), dtype=torch.float32, device=device)
    b = a.clone()
    c = a.view(1, 2, 2)
    a_ = a.clone().detach()
    b_ = b.clone().detach()
    c_ = c.clone().detach()
    jfn = executor.make_callable(
        f, skip_inplace_alias_updates=False, skip_inplace_functionalization=True, disable_inplace_copy_check=True
    )
    actual = jfn(a, b, c)
    expected = f(a_, b_, c_)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(a, a_)
    torch.testing.assert_close(b, b_)
    torch.testing.assert_close(c, c_)
