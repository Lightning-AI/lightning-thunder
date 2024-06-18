from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from collections.abc import Callable

import torch.testing

from thunder.core import dtypes
from thunder.core.prims import PrimIDs
from thunder.tests.framework import ops
from thunder.tests.opinfos import opinfos, OpInfo, make_number, SampleInput
from thunder.tests.make_tensor import make_tensor
from thunder.torch import _torch_to_thunder_function_map, _inplace_to_out_of_place


# `SampleInput`s of ops with `inplace` argument do not seem to come with `inplace` arg, so give it to them.
def sample_generator_wrapper(sample_generator, is_silu: bool = False):

    def f(*args, **kwargs):
        for sample in sample_generator(*args, **kwargs):
            if not is_silu:
                yield SampleInput(*(list(sample.args) + [True]), **sample.kwargs)
            else:
                # silu treats `inplace` as a kwarg
                # ref: https://github.com/Lightning-AI/lightning-thunder/commit/335d84c89
                new_kwargs = {"inplace": True}
                if sample.kwargs:
                    new_kwargs.update(sample.kwargs)
                yield SampleInput(*sample.args, **new_kwargs)

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
            sample_input_generator=sample_generator_wrapper(op.sample_input_generator, is_silu=op.name == "silu"),
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

        return self.torch_func(*args, **kwargs)


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
