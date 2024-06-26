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
from thunder.tests.framework import instantiate, ops, requiresCUDA, NOTHING, TorchExecutor
from thunder.tests.opinfos import opinfos, OpInfo, make_number, SampleInput
from thunder.tests.make_tensor import make_tensor
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
    jitted_inplace_op = thunder.jit(
        InplaceOpWrapper(op.torch_reference, is_polygamma, True),
        executors=executor.executors_list(),
    )
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
        # Backward fails with nvfuserExecutor, RuntimeError: Unsupported iterable object type for define_vector! Index:0
        # Numerical accuracy error when TorchExecutor, `train=True` and dtype is fp32.
        # with RTX6000 Ada and CUDA 12.3, I see somewhat huge error:
        # E   AssertionError: Tensor-likes are not close!
        # E
        # E   Mismatched elements: 9401 / 9408 (99.9%)
        # E   Greatest absolute difference: 0.07035164535045624 at index (4, 1, 0, 3) (up to 1e-05 allowed)
        # E   Greatest relative difference: 343.7076110839844 at index (5, 0, 5, 4) (up to 1.3e-06 allowed)
        # E   The failure occurred for item [0]
        if train and executor == TorchExecutor and dtype == thunder.float64:
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

    a, b = (make_tensor((2, 2), device=device.type, dtype=torch.float32) for _ in range(2))
    a_, b_ = a.clone().detach(), b.clone().detach()

    jittd_f = thunder.jit(f, executors=executor.executors_list())

    c, d, e = jittd_f(a, b)
    c_, d_, e_ = f(a_, b_)

    torch.testing.assert_close((c, d, e), (c_, d_, e_))

    def g(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.exp(a)
        d = torch.tanh(b)

        e, _ = c.chunk(2)
        e *= 1.5

        d.div_(a)
        return d, e

    a, b = (make_tensor((2, 2), device=device.type, dtype=torch.float32) for _ in range(2))
    a_, b_ = a.clone().detach(), b.clone().detach()

    jittd_g = thunder.jit(g, executors=executor.executors_list())

    d, e = jittd_g(a, b)
    d_, e_ = g(a_, b_)

    torch.testing.assert_close((d, e), (d_, e_))

    def h(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.exp(a)
        d = torch.tanh(b)

        e = c.view(-1)
        e.add_(d.flatten())

        d.div_(a)
        return c, d, e / 2.0

    a, b = (make_tensor((2, 2), device=device.type, dtype=torch.float32) for _ in range(2))
    a_, b_ = a.clone().detach(), b.clone().detach()

    jittd_h = thunder.jit(h, executors=executor.executors_list())

    c, d, e = jittd_h(a, b)
    c_, d_, e_ = h(a_, b_)

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

    a, b = (make_tensor((2, 2), device=device.type, dtype=torch.float32) for _ in range(2))
    jittd_f = thunder.jit(f, executors=executor.executors_list())

    with pytest.raises(NotImplementedError, match="in-place op of `torch.Tensor.add_` to `torch.flatten` output"):
        _ = jittd_f(a, b)

    def f(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = torch.exp(a)
        d = torch.tanh(b)

        e, _ = c.chunk(2)
        e *= 1.5

        d.div_(a)
        return c, d, e

    jittd_f = thunder.jit(f, executors=executor.executors_list())
    with pytest.raises(NotImplementedError, match="in-place op of `torch.Tensor.mul_`"):
        _ = jittd_f(a, b)
