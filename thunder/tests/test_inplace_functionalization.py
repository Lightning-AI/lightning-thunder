from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import pytest
import torch.testing

from thunder.core import dtypes
from thunder.core.prims import PrimIDs
from thunder.tests.framework import instantiate, ops, requiresCUDA, NOTHING
from thunder.tests.opinfos import opinfos, OpInfo, make_number, SampleInput
from thunder.tests.make_tensor import make_tensor
from thunder.torch import _torch_to_thunder_function_map, _inplace_to_out_of_place


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


_torchsymbol_to_torch: dict[Sybmol, Callable] = {v: k for k, v in _torch_to_thunder_function_map.items()}
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


def test_invalid_cases():
    import thunder

    a = torch.randn((2, 2))

    def f_with_reshape(a: torch.Tensor) -> torch.Tensor:
        b = torch.reshape(a, (-1,))
        b.exp_()
        return b

    jitted = thunder.jit(f_with_reshape)
    with pytest.raises(NotImplementedError) as excinfo:
        jitted(a)
    assert "in-place op of `torch.exp_` to `torch.reshape` output" in str(excinfo.value)

    def f_with_contiguous(a: torch.Tensor) -> torch.Tensor:
        b = a.contiguous()
        b.exp_()
        return b

    jitted = thunder.jit(f_with_contiguous)
    with pytest.raises(NotImplementedError) as excinfo:
        jitted(a)
    assert "in-place op of `torch.exp_` to `torch.Tensor.contiguous` output" in str(excinfo.value)


# TODO(crcrpar): Investigate the numerical accuracy when `train=True` and dtype is fp32.
# with RTX6000 Ada and CUDA 12.3, I see somewhat huge error:
# E   AssertionError: Tensor-likes are not close!
# E
# E   Mismatched elements: 913 / 1000 (91.3%)
# E   Greatest absolute difference: 0.000273287296295166 at index (0, 50) (up to 1e-05 allowed)
# E   Greatest relative difference: 0.4177769422531128 at index (0, 727) (up to 1.3e-06 allowed)
@requiresCUDA
@pytest.mark.parametrize("train", (False, True))
def test_parse_resnet18(train: bool):
    import thunder

    torchvision = pytest.importorskip("torchvision")

    device = torch.device("cuda")
    dtype = torch.float64 if train else torch.float32
    with device:
        model: nn.Module = torchvision.models.resnet18(weights=None).to(device=device, dtype=dtype)
        ref_model: nn.Module = torchvision.models.resnet18(weights=None).to(device=device, dtype=dtype)
    if not train:
        model = model.eval()
        ref_model = ref_model.eval()
    ref_model.load_state_dict(model.state_dict())

    jitted = thunder.jit(model)
    x = make_tensor((1, 3, 224, 224), dtype=dtype, device=device)
    torch.testing.assert_close(jitted(x), ref_model(x))


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

    jittd_f = thunder.jit(f, executors=executor.executors_list())

    c, d, e = jittd_f(a, b)
    c_, d_, e_ = f(a_, b_)

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
    a_, b_ = a.clone().detach(), b.clone().detach()

    jittd_f = thunder.jit(f, executors=executor.executors_list())

    with pytest.raises(NotImplementedError, match="in-place op of `torch.Tensor.add_` to `torch.flatten` output"):
        c, d, e = jittd_f(a, b)
