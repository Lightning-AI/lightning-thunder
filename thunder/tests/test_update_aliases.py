from collections.abc import Callable

import pytest
import torch.testing

import thunder.core.dtypes as dtypes
from thunder.tests.make_tensor import make_tensor
from thunder.tests.framework import instantiate, ops, NOTHING, TorchExecutor, TorchCompileExecutor, nvFuserExecutor
from thunder.tests.test_inplace_functionalization import _inplace_opinfos


@ops(_inplace_opinfos, supported_dtypes=(dtypes.float32,))
def test_update_aliases(op, device, dtype, executor, _):
    sample = next(op.sample_inputs(device, dtype))
    # The sample generator is the one for `polygamma`.
    # `polygamma` expects an int as its first argument and a tensor as its second but
    # `polygamma_` wants opposite; tensor first, int second.
    args = list(sample.args)
    if op.name == "polygamma_":
        args[0], args[1] = args[1], args[0]

    j_op = executor.make_callable(
        op.torch_reference, skip_inplace_alias_updates=False, skip_inplace_functionalization=True
    )
    actual = j_op(*args, **sample.kwargs)
    expected = op.torch_reference(*args, **sample.kwargs)
    torch.testing.assert_close(actual, expected, equal_nan=True)


# These tests fail with nvFuser because its fusion pass dce's the final copy_ bsym, which has no output.
@instantiate(
    dtypes=NOTHING,
    executors=(
        TorchExecutor,
        TorchCompileExecutor,
    ),
)
def test_setitem_on_view(executor, device, dtype):
    def f(x, _):
        y = x.view((3, 2))
        y[0][0] = 0
        return x

    def g(x, _):
        y = x.view((3, 2))
        y[0][0] = 0
        return y

    for fn in [f, g]:
        a = make_tensor((2, 3), dtype=torch.float32, device=device)
        b = make_tensor((2, 3), dtype=torch.float32, device=device)
        a_, b_ = a.clone().detach(), b.clone().detach()
        jfn = executor.make_callable(fn, skip_inplace_alias_updates=False, skip_inplace_functionalization=True)
        actual = jfn(a, b)
        expected = fn(a_, b_)
        torch.testing.assert_close(actual, expected)


@instantiate(
    dtypes=NOTHING,
)
def test_inplace_on_view(executor, device, dtype):
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

    for fn in [h, i, j]:
        a = make_tensor((2, 3), dtype=torch.float32, device=device)
        b = make_tensor((2, 3), dtype=torch.float32, device=device)
        a_, b_ = a.clone().detach(), b.clone().detach()
        jfn = executor.make_callable(fn, skip_inplace_alias_updates=False, skip_inplace_functionalization=True)
        actual = jfn(a, b)
        expected = fn(a_, b_)
        torch.testing.assert_close(actual, expected)


# These tests fail with nvFuser with stride issues.
# See https://github.com/NVIDIA/Fuser/issues/3957.
@instantiate(
    dtypes=NOTHING,
    executors=(
        TorchExecutor,
        TorchCompileExecutor,
    ),
)
def test_inplace_on_chunk_non_default_dim(executor, device, dtype):
    def f(x):
        y, z = x.chunk(2, dim=1)
        y.relu_()
        return y, z

    for fn in [f]:
        a = make_tensor((2, 3), dtype=torch.float32, device=device)
        a_ = a.clone().detach()
        jfn = executor.make_callable(fn, skip_inplace_alias_updates=False, skip_inplace_functionalization=True)
        actual = jfn(a)
        expected = fn(a_)
        torch.testing.assert_close(actual, expected)


@instantiate(
    dtypes=NOTHING,
)
def test_inplace_on_chunk(executor, device, dtype):
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

    for fn in [g, h]:
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
    dtypes=NOTHING,
)
def test_aliased_input(executor, device, dtype):
    def f(x, y, z):
        return y.exp_().add(x) + z.exp()

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


@instantiate(
    dtypes=NOTHING,
)
def test_write_to_intermediate_result(executor, device, dtype):
    if executor == nvFuserExecutor:
        pytest.xfail("nvFuser does not support writing to intermediate results")

    def fn(x):
        y = x.view(-1)
        y.add_(1)
        return y

    a = make_tensor((2, 3), dtype=torch.float32, device=device)
    jfn = executor.make_callable(fn, skip_inplace_alias_updates=True, skip_inplace_functionalization=True)
    actual = jfn(a)
    expected = fn(a)
    torch.testing.assert_close(actual, expected)
