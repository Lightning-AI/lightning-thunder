from contextlib import contextmanager
from functools import partial

import pytest
import thunder
import thunder.core.devices as devices
import torch
from thunder.core import dtypes

from thunder.tests.framework import instantiate, ops, TorchExecutor
from thunder.tests.make_tensor import make_tensor
from thunder.tests.opinfos import get_opinfo, OpInfo
from thunder.torch import _torch_to_thunder_function_map

tmp_torch_to_thunder = dict(_torch_to_thunder_function_map)


@contextmanager
def recover(disabled_op_list=None):
    if disabled_op_list is None:
        disabled_op_list = []
    global _torch_to_thunder_function_map
    list(_torch_to_thunder_function_map.pop(k.torch_reference, None) for k in disabled_op_list)
    try:
        yield
    finally:
        _torch_to_thunder_function_map.update(tmp_torch_to_thunder)


_unsupported_opinfos = (get_opinfo("zeta"), get_opinfo("nextafter"), get_opinfo("item"))


@ops(_unsupported_opinfos, supported_dtypes=(dtypes.float32,), supported_executors=(TorchExecutor,))
def test_fallback_exception(
    op: OpInfo,
    device: str,
    dtype: dtypes.dtype,
    executor,
    comp,
):
    with recover(_unsupported_opinfos):
        assert op.torch_reference not in _torch_to_thunder_function_map
        for sample in op.sample_inputs(device, dtype, requires_grad=True):
            comp = sample.comp if sample.comp is not None else comp
            tfn = thunder.jit(
                op.torch_reference,
                executors=executor.executors_list(),
            )
            with pytest.raises(NotImplementedError, match="^Exception encountered when doing automatic registration"):
                thunder_result = tfn(*sample.args, **sample.kwargs)


_fallback_opinfos = [
    get_opinfo("embedding"),
    get_opinfo("digamma"),
    get_opinfo("is_complex"),
    get_opinfo("conv2d"),
    get_opinfo("reshape"),
]


@ops(_fallback_opinfos, supported_dtypes=(dtypes.float32,), supported_executors=(TorchExecutor,))
def test_fallback_forward(op, device, dtype, executor, comp):
    with recover(_fallback_opinfos):
        for sample in op.sample_inputs(device, dtype, requires_grad=True):
            comp = sample.comp if sample.comp is not None else comp
            tfn = thunder.jit(
                op.torch_reference,
                executors=executor.executors_list(),
            )
            thunder_result = tfn(*sample.args, **sample.kwargs)

            torch_result = op.torch_reference(*sample.args, **sample.kwargs)
            comp(thunder_result, torch_result)


_fallback_opinfos = [get_opinfo("embedding"), get_opinfo("digamma"), get_opinfo("conv2d"), get_opinfo("unfold")]


@ops(_fallback_opinfos, supported_dtypes=(dtypes.float32,), supported_executors=(TorchExecutor,))
def test_fallback_backward(op, device, dtype, executor, comp):
    with recover(_fallback_opinfos):
        for sample in op.sample_inputs(device, dtype, requires_grad=True):
            comp = sample.comp if sample.comp is not None else comp
            tfn = thunder.jit(
                op.torch_reference,
                executors=executor.executors_list(),
            )
            thunder_result = tfn(*sample.args, **sample.kwargs)

            bwd_trc = thunder.last_backward_traces(tfn)[-1]
            vjp_op_name = f"{op.name}_vjp"
            assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trc.bound_symbols)


@instantiate(dtypes=(thunder.float32,), devicetypes=(devices.DeviceType.CUDA,), executors=(TorchExecutor,))
def test_nanogpt_block(executor, device, dtype):
    skip_ops = [
        get_opinfo("layer_norm"),
        get_opinfo("linear"),
        get_opinfo("gelu"),
        get_opinfo("scaled_dot_product_attention"),
    ]

    with recover(skip_ops):
        import thunder.tests.nanogpt_model as nanogpt_model

        tdtype = thunder.torch.to_torch_dtype(dtype)
        make = partial(make_tensor, dtype=tdtype, device=device)

        config = nanogpt_model.GPTConfig(dropout=0)
        model = nanogpt_model.Block(config).to(device=device, dtype=tdtype)
        jitted = executor.make_callable(model)

        x = make((2, config.block_size, config.n_embd))

        cache_entry, _, _ = thunder.compile_data(jitted).get_computation_and_inputs(x)
        bwd_trcs = cache_entry.backward_traces
        for op in skip_ops:
            vjp_op_name = f"{op.name}_vjp"
            assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)


@instantiate(dtypes=(thunder.float64,), devicetypes=(devices.DeviceType.CUDA,), executors=(TorchExecutor,))
def test_alexnet(executor, device, dtype):
    torchvision = pytest.importorskip("torchvision")
    skip_ops = [get_opinfo("conv2d"), get_opinfo("linear"), get_opinfo("adaptive_avg_pool2d"), get_opinfo("max_pool2d")]
    with recover(skip_ops):
        tdtype = thunder.torch.to_torch_dtype(dtype)
        model = torchvision.models.alexnet(weights=None).to(device=device, dtype=tdtype)
        model = model.train()

        jitted = executor.make_callable(model)
        x = make_tensor((1, 3, 224, 224), dtype=tdtype, device=device)

        cache_entry, _, _ = thunder.compile_data(jitted).get_computation_and_inputs(x)
        bwd_trcs = cache_entry.backward_traces
        for op in skip_ops:
            vjp_op_name = f"{op.name}_vjp"
            assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)
