import pytest
import torch.testing

import thunder
from thunder.examine import get_fusions
import thunder.core.dtypes as dtypes
from thunder.tests.make_tensor import make_tensor, make_tensor_like
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


@instantiate(
    dtypes=(dtypes.float32,),
)
def test_inplace_to_alias_func_args(executor, device, dtype):
    shape = (2, 2)
    torch_dtype = dtypes.to_torch_dtype(dtype)

    # copied from https://github.com/Lightning-AI/lightning-thunder/issues/738
    def f(a, b):
        return a.exp_() + b.tanh_(), a

    jitted_f = executor.make_callable(
        f, skip_inplace_alias_updates=False, skip_inplace_functionalization=True, disable_inplace_copy_check=True
    )

    a = make_tensor(shape, device=device, dtype=torch_dtype)
    b = make_tensor(shape, device=device, dtype=torch_dtype)
    a_ref, b_ref = a.clone().detach(), b.clone().detach()

    res_of_a, a_out = jitted_f(a, a)
    ref_res_of_a, ref_a_out = f(a_ref, a_ref)

    fds = get_fusions(thunder.last_traces(jitted_f)[-1])
    for _, fd in fds:
        print(fd.last_used.repro_script_for())

    assert (thunder.cache_hits(jitted_f), thunder.cache_misses(jitted_f)) == (0, 1)
    torch.testing.assert_close(res_of_a, ref_res_of_a)
    torch.testing.assert_close(a, a_ref)
    assert a_out.data_ptr() == a.data_ptr()

    a = make_tensor_like(a)
    a_ref = a.clone().detach()
    res_of_a_and_b, _ = jitted_f(a, b)
    ref_res_of_a_and_b, _ = f(a_ref, b_ref)
    assert (thunder.cache_hits(jitted_f), thunder.cache_misses(jitted_f)) == (0, 2)
    torch.testing.assert_close(res_of_a_and_b, ref_res_of_a_and_b)

    res_of_b, _ = jitted_f(b, b)
    ref_res_of_b, _ = f(b_ref, b_ref)
    assert (thunder.cache_hits(jitted_f), thunder.cache_misses(jitted_f)) == (1, 2)
    torch.testing.assert_close(res_of_b, ref_res_of_b)
    torch.testing.assert_close(b, b_ref)

    b = make_tensor_like(b)
    b_ref = b.clone().detach()
    res_of_b_and_a, _ = jitted_f(b, a)
    ref_res_of_b_and_a, _ = f(b_ref, a_ref)
    assert (thunder.cache_hits(jitted_f), thunder.cache_misses(jitted_f)) == (2, 2)
    torch.testing.assert_close(res_of_b_and_a, ref_res_of_b_and_a)

    def f(a, b):
        return a.exp() + b.tanh()

    jitted_f = executor.make_callable(
        f, skip_inplace_alias_updates=False, skip_inplace_functionalization=True, disable_inplace_copy_check=True
    )
    jitted_f(a, a)
    assert (thunder.cache_hits(jitted_f), thunder.cache_misses(jitted_f)) == (0, 1)
    jitted_f(a, b)
    assert (thunder.cache_hits(jitted_f), thunder.cache_misses(jitted_f)) == (0, 2)
    jitted_f(b, a)
    assert (thunder.cache_hits(jitted_f), thunder.cache_misses(jitted_f)) == (1, 2)
    jitted_f(b, b)
    assert (thunder.cache_hits(jitted_f), thunder.cache_misses(jitted_f)) == (2, 2)

    def f(a, b, c):
        d = a.exp_()
        e = b.tanh_()
        f = c.cosh_()
        return d + e + f

    a = make_tensor(shape, device=device, dtype=torch_dtype)
    a_expected = a.exp().tanh().cosh()

    jitted_f = executor.make_callable(
        f, skip_inplace_alias_updates=False, skip_inplace_functionalization=True, disable_inplace_copy_check=True
    )
    out = jitted_f(a, a, a)

    torch.testing.assert_close(a, a_expected)
    torch.testing.assert_close(out, 3 * a_expected)

    a, b = make_tensor_like(a), make_tensor_like(a)
    a_ref, b_ref = a.clone().detach(), b.clone().detach()
    out, out_ref = jitted_f(a, b, b), f(a_ref, b_ref, b_ref)
    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close((a, b), (a_ref, b_ref))

    a, b = make_tensor_like(a), make_tensor_like(a)
    a_ref, b_ref = a.clone().detach(), b.clone().detach()
    out, out_ref = jitted_f(a, b, a), f(a_ref, b_ref, a_ref)
    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close((a, b), (a_ref, b_ref))

    def f(a):
        return a.zero_()

    a = make_tensor(shape, device=device, dtype=torch_dtype)
    out_expected = torch.zeros_like(a)

    jitted_f = executor.make_callable(
        f, skip_inplace_alias_updates=False, skip_inplace_functionalization=True, disable_inplace_copy_check=True
    )
    out = jitted_f(a)

    torch.testing.assert_close(out, out_expected)
