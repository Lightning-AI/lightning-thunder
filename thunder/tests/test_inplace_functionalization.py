from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch.testing

import thunder
import thunder.core.devices as devices
from thunder.core import dtypes
from thunder.core.prims import PrimIDs
from thunder.tests.framework import (
    instantiate,
    ops,
    requiresCUDA,
    NOTHING,
    TorchExecutor,
    TorchCompileExecutor,
    nvFuserExecutor,
)
from thunder.tests.opinfos import opinfos, OpInfo, make_number, SampleInput
from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.torch import _torch_to_thunder_function_map, _inplace_to_out_of_place

if TYPE_CHECKING:
    from thunder.core.symbol import Symbol


# `SampleInput`s of ops with `inplace` argument do not seem to come with `inplace` arg, so give it to them.
def sample_generator_wrapper(sample_generator):

    def f(*args, **kwargs):
        for sample in sample_generator(*args, **kwargs):
            yield SampleInput(*(list(sample.args) + [True]), **sample.kwargs)

    return f


def inplace_masked_fill_sample_generator(op, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    number = partial(make_number, dtype=dtype)

    # pred_shape, a_shape, value
    cases = (((2, 2, 2), (2, 2, 2), number()),)

    for pred_shape, a_shape, value in cases:
        pred, a = make(pred_shape, dtype=torch.bool, requires_grad=False), make(a_shape)
        yield SampleInput(a, pred, value)


_torchsymbol_to_torch: dict[Symbol, Callable] = {v: k for k, v in _torch_to_thunder_function_map.items()}
_functional_to_inplace: dict[Callable, Callable] = {
    functional: inplace for inplace, (functional, index) in _inplace_to_out_of_place.items() if index == -1
}
_functional_to_functional_with_inplace_arg: dict[Callable, tuple[Callable, int]] = {
    functional: (inplace, index) for inplace, (functional, index) in _inplace_to_out_of_place.items() if index >= 0
}
_inplace_opinfos: list[OpInfo] = []
for op in opinfos:
    if not (op.op in _functional_to_inplace or op.op in _functional_to_functional_with_inplace_arg):
        continue
    # ops that have an argument of `inplace` such as `F.relu` and `F.gelu`
    if op.op in _functional_to_functional_with_inplace_arg:
        inplace_op, _ = _functional_to_functional_with_inplace_arg[op.op]
        assert op.name != "masked_fill"
        inplace_opinfo = OpInfo(
            inplace_op,
            sample_input_generator=sample_generator_wrapper(op.sample_input_generator),
            torch_reference=getattr(torch.nn.functional, op.name),
        )
        _inplace_opinfos.append(inplace_opinfo)
    # in-place ops whose name ends with `_`
    if op.op in _functional_to_inplace:
        inplace_op = _functional_to_inplace[op.op]
        inplace_opinfo = OpInfo(
            inplace_op,
            sample_input_generator=(
                op.sample_input_generator if op.name != "masked_fill" else inplace_masked_fill_sample_generator
            ),
            torch_reference=_torchsymbol_to_torch[inplace_op],
        )
        _inplace_opinfos.append(inplace_opinfo)


@dataclass(frozen=True)
class InplaceOpWrapper:
    torch_func: Callable
    is_polygamma: bool
    jitted: bool

    def __call__(self, *args, **kwargs):
        # polygamma expects an int as its first argument and a tensor as its second but
        # torch.Tensor.polygamma_ wants opposite; tensor first, int second.
        # ref:
        # - https://pytorch.org/docs/stable/special.html#torch.special.polygamma
        # - https://pytorch.org/docs/stable/generated/torch.Tensor.polygamma_.html
        args = list(args)
        idx = int(self.is_polygamma and self.jitted)
        t = args[idx] + 1.0
        args[idx] = t

        self.torch_func(*args, **kwargs)
        return t


@ops(_inplace_opinfos, supported_dtypes=(dtypes.float32,))
def test_functionalization(op: OpInfo, device: str, dtype: dtypes.dtype, executor, _):
    import thunder

    is_polygamma = op.name == "polygamma_"
    inplace_op = InplaceOpWrapper(op.torch_reference, is_polygamma, False)
    jitted_inplace_op = executor.make_callable(InplaceOpWrapper(op.torch_reference, is_polygamma, True))
    sample: SampleInput
    for idx, sample in enumerate(op.sample_inputs(device, dtype)):
        if idx > 0:
            break

        args = list(sample.args)
        if is_polygamma:
            tmp = args[0]
            args[0] = args[1]
            args[1] = tmp
        expected = inplace_op(*args, **sample.kwargs)
        actual = jitted_inplace_op(*sample.args, **sample.kwargs)
        torch.testing.assert_close(actual, expected, equal_nan=True)

    # make sure `prims.copy_` does not exist in the trace thanks to functionalization
    fw_extrace = thunder.last_traces(jitted_inplace_op)[-1]
    assert not list(
        filter(
            lambda bsym: bsym.sym.id == PrimIDs.COPY_,
            fw_extrace.bound_symbols,
        )
    )


@pytest.fixture
def turn_off_tf32_and_set_seed(monkeypatch):
    import torch

    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    torch.manual_seed(42)


@instantiate(
    dtypes=(thunder.float32, thunder.float64),
    devicetypes=(devices.DeviceType.CUDA,),
    decorators=(pytest.mark.parametrize("train", (False, True)),),
)
@requiresCUDA
def test_parse_resnet18(executor, device, dtype, turn_off_tf32_and_set_seed, train: bool):
    from contextlib import nullcontext

    import thunder

    torchvision = pytest.importorskip("torchvision")

    tdtype = thunder.torch.to_torch_dtype(dtype)
    model = torchvision.models.resnet18(weights=None).to(device=device, dtype=tdtype)
    ref_model = torchvision.models.resnet18(weights=None).to(device=device, dtype=tdtype)
    if not train:
        model = model.eval()
        ref_model = ref_model.eval()
        ctx = torch.no_grad
    else:
        model = model.train()
        ref_model = ref_model.train()
        ctx = nullcontext
    ref_model.load_state_dict(model.state_dict())

    jitted = executor.make_callable(model)
    x = make_tensor((1, 3, 224, 224), dtype=tdtype, device=device)

    with ctx():
        out1 = ref_model(x)
        out2 = jitted(x)
        torch.testing.assert_close(out1, out2)
        # Numerical accuracy error when TorchExecutor, `train=True` and dtype is fp32.
        # with RTX6000 Ada and CUDA 12.3, I see somewhat huge error:
        # E   AssertionError: Tensor-likes are not close!
        # E
        # E   Mismatched elements: 9401 / 9408 (99.9%)
        # E   Greatest absolute difference: 0.07035164535045624 at index (4, 1, 0, 3) (up to 1e-05 allowed)
        # E   Greatest relative difference: 343.7076110839844 at index (5, 0, 5, 4) (up to 1.3e-06 allowed)
        # E   The failure occurred for item [0]
        if train and dtype == thunder.float64:
            torch_grads = torch.autograd.grad(out1, ref_model.parameters(), torch.ones_like(out1))
            thunder_grads = torch.autograd.grad(out2, jitted.parameters(), torch.ones_like(out2))
            torch.testing.assert_close(torch_grads, thunder_grads)


@instantiate(
    dtypes=NOTHING,
)
def test_inplace_to_views(executor, device, _):
    import thunder

    def f(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = torch.exp(a)
        d = torch.tanh(b)

        e = c.view(-1)
        e += d.flatten()

        d.div_(a)
        return c, d, e

    a, b = (make_tensor((2, 2), device=device, dtype=torch.float32) for _ in range(2))
    a_, b_ = a.clone().detach(), b.clone().detach()

    jitted_f = executor.make_callable(f)

    c, d, e = jitted_f(a, b)
    c_, d_, e_ = f(a_, b_)

    torch.testing.assert_close((c, d, e), (c_, d_, e_))

    def g(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.exp(a)
        d = torch.tanh(b)

        e, _ = c.chunk(2)
        e *= 1.5

        d.div_(a)
        return d, e

    a, b = (make_tensor((2, 2), device=device, dtype=torch.float32) for _ in range(2))
    a_, b_ = a.clone().detach(), b.clone().detach()

    jitted_g = executor.make_callable(g)

    d, e = jitted_g(a, b)
    d_, e_ = g(a_, b_)

    torch.testing.assert_close((d, e), (d_, e_))

    def h(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.exp(a)
        d = torch.tanh(b)

        e = c.view(-1)
        e.add_(d.flatten())

        d.div_(a)
        return c, d, e / 2.0

    a, b = (make_tensor((2, 2), device=device, dtype=torch.float32) for _ in range(2))
    a_, b_ = a.clone().detach(), b.clone().detach()

    jitted_h = executor.make_callable(h)

    c, d, e = jitted_h(a, b)
    c_, d_, e_ = h(a_, b_)

    torch.testing.assert_close((c, d, e), (c_, d_, e_))


@instantiate(
    executors=(nvFuserExecutor,),
    dtypes=(dtypes.float32,),
)
@requiresCUDA
def test_inplace_to_args_with_nvfuser(executor, device, _):

    def func(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a += b
        c = torch.exp(a)
        d = torch.tanh(b)

        e = c.view(-1)
        e.add_(d.flatten())

        d.div_(a)
        return c, d, e / 2.0

    a, b = (make_tensor((2, 2), device=device, dtype=torch.float32) for _ in range(2))
    a_, b_ = a.clone().detach(), b.clone().detach()

    jitted = executor.make_callable(func)

    c, d, e = jitted(a, b)
    c_, d_, e_ = func(a_, b_)

    torch.testing.assert_close((c, d, e), (c_, d_, e_))


@instantiate(
    dtypes=NOTHING,
)
def test_error_of_inplace_to_views(executor, device, _):
    import thunder

    def f(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = torch.exp(a)
        d = torch.tanh(b)

        e = c.flatten()
        e += d.flatten()

        d.div_(a)
        return c, d, e

    a, b = (make_tensor((2, 2), device=device, dtype=torch.float32) for _ in range(2))
    jitted_f = executor.make_callable(f)

    with pytest.raises(NotImplementedError, match="in-place op of `torch.Tensor.add_` to `torch.flatten` output"):
        _ = jitted_f(a, b)

    def f(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = torch.exp(a)
        d = torch.tanh(b)

        e, _ = c.chunk(2)
        e *= 1.5

        d.div_(a)
        return c, d, e

    jitted_f = executor.make_callable(f)
    with pytest.raises(NotImplementedError, match="in-place op of `torch.Tensor.mul_`"):
        _ = jitted_f(a, b)


@instantiate(
    dtypes=NOTHING,
)
def test_multiple_inplace_to_args(executor, device, _):

    def f(a):
        a.exp_()
        a.sin_()
        return a.cos()

    x = make_tensor((2, 2), device=device, dtype=torch.float32)
    x_ref = x.clone().detach()
    expected = f(x_ref)

    jitted = executor.make_callable(f)
    actual = jitted(x)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, x_ref)

    def f_with_view(a):
        b = a.view(-1)
        return b.exp_().sin_().cos()

    x = make_tensor((2, 2), device=device, dtype=torch.float32)
    x_ref = x.clone().detach()
    expected = f_with_view(x_ref)

    jitted = executor.make_callable(f_with_view)
    actual = jitted(x)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, x_ref)

    def f(a):
        return a.exp_().sin_()

    x = make_tensor((2, 2), device=device, dtype=torch.float32)
    x_ref = x.clone().detach()
    expected = f(x_ref)
    jitted = executor.make_callable(f)
    actual = jitted(x)
    torch.testing.assert_close(actual, expected)
    assert x.data_ptr() == actual.data_ptr()


@instantiate(
    dtypes=NOTHING,
)
def test_multiple_views_before_inplace_to_base(executor, device, _):
    from thunder.tests.framework import nvFuserTestExecutor

    if type(executor) is nvFuserTestExecutor:
        pytest.skip(
            "nvFuser doesn't enforce the order between `z=x.view(-1)` and "
            "`x.add_(1)`, so the behavior is undefined due to this "
            "race condition. See https://github.com/NVIDIA/Fuser/issues/2839."
        )

    # ref: https://github.com/pytorch/pytorch/blob/29e2e2a/test/test_functionalization.py#L159-L169
    def f(x):
        y = x.view(-1)
        z = x.view(-1)
        x.add_(1)
        # y should have been updated.
        y2 = y + 1
        # z should have been updated too.
        z2 = z + 1
        return z2

    x = make_tensor((2, 2), device=device, dtype=torch.float32)
    x_ref = x.clone().detach()
    expected = f(x_ref)

    jitted = executor.make_callable(f)
    actual = jitted(x)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, x_ref)

    def f(x):
        x = x.add(1)
        y = x.view(-1)
        z = x.view(-1)
        x.add_(1)
        # y should have been updated.
        y2 = y + 1
        # z should have been updated too.
        z2 = z + 1
        return z2

    x = make_tensor((2, 2), device=device, dtype=torch.float32)
    x_ref = x.clone().detach()
    expected = f(x_ref)

    jitted = executor.make_callable(f)
    actual = jitted(x)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, x_ref)

    # ref: https://github.com/Lightning-AI/lightning-thunder/pull/869#issuecomment-2257738623
    def f(x):
        x = x.add(1)
        z = x.view(-1)[::2]  # Note that basic slicing is used here which also produces a view
        x.add_(1)
        # z should have been updated
        z2 = z + 1
        return z2

    x = make_tensor((2, 2), device=device, dtype=torch.float32)

    jitted = executor.make_callable(f)
    with pytest.raises(
        RuntimeError,
        match="Fail to propagate the in-place change of `t3` to `z` because of the different number of elements: 4 and 2",
    ):
        jitted(x)


@instantiate(
    dtypes=NOTHING,
)
def test_multiple_inplace_to_multiple_args(executor, device, _):

    def f(xs, ys, z):
        for i in range(len(xs)):
            ys[i].add_(xs[i].exp_().sin_())
            z.add_(ys[i])
        return z

    jitted = executor.make_callable(f)
    xs = [make_tensor((2, 2), device=device, dtype=torch.float32) for _ in range(2)]
    ys = [make_tensor((2, 2), device=device, dtype=torch.float32) for _ in range(2)]
    z = make_tensor((2, 2), device=device, dtype=torch.float32)
    xs_ref = [x.clone().detach() for x in xs]
    ys_ref = [x.clone().detach() for x in ys]
    z_ref = z.clone().detach()

    res = jitted(xs, ys, z)
    res_ref = f(xs_ref, ys_ref, z_ref)

    torch.testing.assert_close(actual=res, expected=res_ref)
    torch.testing.assert_close(actual=z, expected=z_ref)
    torch.testing.assert_close(actual=xs, expected=xs_ref)
    torch.testing.assert_close(actual=ys, expected=ys_ref)


@instantiate(
    dtypes=NOTHING,
)
def test_inplace_to_tensors_with_grad(executor, device, _):
    @torch.no_grad
    def add_y(x, y):
        x.add_(y, alpha=0.1)

    @torch.no_grad
    def add_grad(x, y):
        x.add_(x.grad, alpha=0.1)

    for f in (add_y, add_grad):
        jitted_f = executor.make_callable(f)
        x = make_tensor((2, 2), device=device, dtype=torch.float32, requires_grad=True)
        x.grad = make_tensor((2, 2), device=device, dtype=torch.float32)
        y = make_tensor((2, 2), device=device, dtype=torch.float32)

        x_ref = x.clone().detach().requires_grad_(True)
        x_ref.grad = x.grad.clone().detach()
        y_ref = y.clone().detach()

        res = jitted_f(x, y)
        res_ref = f(x_ref, y_ref)

        torch.testing.assert_close(x, x_ref)
        torch.testing.assert_close(x.grad, x_ref.grad)
        torch.testing.assert_close(y, y_ref)
        torch.testing.assert_close(res, res_ref)


@instantiate(
    dtypes=NOTHING,
    executors=(TorchExecutor, TorchCompileExecutor, nvFuserExecutor),
)
def test_single_tensor_adam_like(executor, device, _):

    def single_tensor_adam(
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        state_steps: list[torch.Tensor],
        *,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.9,
        eps: float = 1e-5,
    ) -> None:
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]

            exp_avg.mul_(beta1).add_(grad)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1)

            step_t.add_(1)
            bias_correction2 = 1 - beta2**step_t
            step_size = lr / (1 - beta1**step_t)
            step_size_neg = step_size.neg()
            denom = exp_avg_sq.sqrt() / (bias_correction2 * step_size_neg).add(eps / step_size_neg)
            param.addcdiv_(exp_avg, denom)

    shape = (4,)
    params = [make_tensor(shape, device=device, dtype=torch.float32, high=2, low=1) for _ in range(2)]
    tensors = [
        [make_tensor(shape, device=device, dtype=torch.float32, high=2, low=1) for _ in range(2)] for _ in range(3)
    ]
    tensors = [params] + tensors
    state_steps = [torch.tensor(1, device=device) for _ in range(2)]

    ref_tensors = [[t.clone().detach() for t in tensorlist] for tensorlist in tensors]
    ref_state_steps = [torch.tensor(1, device=device) for _ in range(2)]
    single_tensor_adam(*ref_tensors, state_steps=ref_state_steps)

    jitted = executor.make_callable(single_tensor_adam)
    params, grads, exp_avgs, exp_avg_sqs = tensors

    jitted(params, grads, exp_avgs, exp_avg_sqs, state_steps)
    torch.testing.assert_close(actual=tensors + [state_steps], expected=ref_tensors + [ref_state_steps])


@instantiate(
    dtypes=NOTHING,
)
def test_inplace_to_arg_return_value(executor, device, _):

    def f(a, b):
        c = a + b
        b.mul_(c)
        return b

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    a_, b_ = a.clone().detach(), b.clone().detach()

    b__out = f(a_, b_)

    jitted = executor.make_callable(f)
    b_out = jitted(a, b)
    torch.testing.assert_close(b_out, b__out)
    assert b.data_ptr() == b_out.data_ptr()


@instantiate(
    dtypes=NOTHING,
)
def test_no_self_repeat_in_subsymbols(executor, device, _):

    def f(a, b, c):
        a.add_(b, alpha=c)
        return a.add_(b, alpha=c)

    def functional_f(a, b, c):
        d = a.add(b, alpha=c)
        return d.add(b, alpha=c)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    c = make_tensor((1,), device=device, dtype=torch.float32)

    a_out_ref = executor.make_callable(functional_f)(a, b, c)

    jitted = executor.make_callable(f)
    a_out = jitted(a, b, c)
    torch.testing.assert_close(a_out, a_out_ref)

    traces = thunder.last_traces(jitted)
    for t in filter(lambda t: t._provenance is not None and "Functionalize in-place ops" in t._provenance.pss, traces):
        for bsym in filter(lambda b: b.subsymbols, t.bound_symbols):
            assert bsym.rhs != bsym.subsymbols[0].rhs, bsym


@instantiate(
    dtypes=NOTHING,
)
def test_inplace_copy_on_fusion_inputs_issue_791(executor, device, _):

    def f(x, y, idx, src):
        x.index_copy_(0, idx, src)
        z = x + 1
        y.index_copy_(0, idx, src)
        return z

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    a_, b_ = a.clone().detach(), b.clone().detach()
    idx = torch.arange(2).to(device=device)
    src = make_tensor((2, 2), device=device, dtype=torch.float32)

    o_ = f(a_, b_, idx, src)

    jitted = executor.make_callable(f)
    o = jitted(a, b, idx, src)

    torch.testing.assert_close(a, a_)
    torch.testing.assert_close(b, b_)
    torch.testing.assert_close(o, o_)


@instantiate(
    dtypes=(dtypes.float32,),
)
def test_inplace_to_alias_func_args(executor, device, dtype):
    shape = (2, 2)
    torch_dtype = dtypes.to_torch_dtype(dtype)

    # copied from https://github.com/Lightning-AI/lightning-thunder/issues/738
    def f(a, b):
        return a.exp_() + b.tanh_(), a

    jitted_f = executor.make_callable(f)

    a = make_tensor(shape, device=device, dtype=torch_dtype)
    b = make_tensor(shape, device=device, dtype=torch_dtype)
    a_ref, b_ref = a.clone().detach(), b.clone().detach()

    res_of_a, a_out = jitted_f(a, a)
    ref_res_of_a, ref_a_out = f(a_ref, a_ref)
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

    # TODO(crcrpar): The message should be from the check of in-place to aliases of different shapes.
    with pytest.raises(RuntimeError, match="Attempting to reshape a.shape"):
        jitted_f(a, a[0, :])

    def f(a, b):
        return a.exp() + b.tanh()

    jitted_f = executor.make_callable(f)
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

    jitted_f = executor.make_callable(f)
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


@instantiate(dtypes=NOTHING)
def test_reshape_flatten_error_out(executor, device, _):

    def f(x):
        y = x.reshape(6, 4)
        y.add_(1)
        return y

    def g(x):
        y = x.flatten()
        y.add_(1)
        return y

    def h(x):
        tmp = torch.randn((6, 4))
        y = x.reshape_as(tmp)
        y.add_(1)
        return y

    for fn in (f, g, h):
        x = make_tensor((3, 2, 4), device=device, dtype=torch.float32)
        jitted = executor.make_callable(fn)

        with pytest.raises(NotImplementedError, match="in-place op of"):
            jitted(x)

    def f_with_clone(a):
        y = x.reshape(6, 4)
        z = y.clone()
        z = z + 1
        return z

    x = make_tensor((3, 2, 4), device=device, dtype=torch.float32)
    jitted = executor.make_callable(f_with_clone)
    jitted(x)


@instantiate(dtypes=NOTHING)
def test_aliases_and_functionalizable_inplace(executor, device, _):

    def f(a, x, y):
        return a.exp().add_(x) + y.exp()

    jitted = executor.make_callable(f)

    x = make_tensor((2, 2), device=device, dtype=torch.float32)
    a = x.clone()
    y = x.view(1, 2, 2)

    expected = f(a, x, y)
    actual = jitted(a, x, y)
    torch.testing.assert_close(actual, expected)


# ref: https://github.com/Lightning-AI/lightning-thunder/issues/1236
@instantiate(dtypes=NOTHING)
def test_unused_view_input(executor, device, _):

    def f(a, x, unused):
        return a.exp().add_(x)

    x = make_tensor((2, 2), device=device, dtype=torch.float32)
    a = x.clone()
    unused = x[0]

    jitted = executor.make_callable(f)

    expected = f(a, x, unused)
    actual = jitted(a, x, unused)
    torch.testing.assert_close(actual, expected)


@instantiate(dtypes=NOTHING)
def test_inplace_on_to(executor, device, _):

    def f_self_result(a):
        return a.to().sin_()

    def f_copy(a):
        return a.to(torch.float64).sin_()

    for f in (f_self_result, f_copy):
        x = make_tensor((2, 2), device=device, dtype=torch.float32)
        x_ref = x.clone().detach()
        jitted = executor.make_callable(f)
        actual = jitted(x)
        expected = f(x_ref)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x, x_ref)
