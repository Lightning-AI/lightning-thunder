from functools import partial
from unittest.mock import patch

import pytest
import thunder
import thunder.core.devices as devices
import thunder.torch
import torch
from thunder.core import dtypes

from thunder.tests.framework import instantiate, ops, TorchExecutor
from thunder.tests.make_tensor import make_tensor
from thunder.tests.opinfos import get_opinfo, OpInfo

_unsupported_opinfos = [get_opinfo("zeta"), get_opinfo("nextafter"), get_opinfo("item")]
_fallback_opinfos1 = [
    get_opinfo("embedding"),
    get_opinfo("type_as"),
    get_opinfo("conv2d"),
    get_opinfo("reshape"),
]
_fallback_opinfos2 = [get_opinfo("embedding"), get_opinfo("conv2d"), get_opinfo("unfold"), get_opinfo("expand")]
skip_ops = [
    get_opinfo("layer_norm"),
    get_opinfo("linear"),
    get_opinfo("gelu"),
    get_opinfo("scaled_dot_product_attention"),
]
skip_ops1 = [get_opinfo("conv2d"), get_opinfo("linear"), get_opinfo("adaptive_avg_pool2d"), get_opinfo("max_pool2d")]
disable_opinfos = _unsupported_opinfos + _fallback_opinfos1 + _fallback_opinfos2 + skip_ops + skip_ops1
# tmp1 = dict(thunder.core.jit_ext._general_jit_lookaside_map)
# list(tmp1.pop(k.torch_reference, None) for k in disable_opinfos)
# tmp2 = dict(thunder.torch._torch_to_thunder_function_map)
# list(tmp2.pop(k.torch_reference, None) for k in disable_opinfos)
# tmp3 = dict(thunder.core.jit_ext._minimal_lookaside_map)
# list(tmp3.pop(k.torch_reference, None) for k in disable_opinfos)


# @ops(_unsupported_opinfos, supported_dtypes=(dtypes.float32,), supported_executors=(TorchExecutor,))
# def test_fallback_exception(
#     op: OpInfo,
#     device: str,
#     dtype: dtypes.dtype,
#     executor,
#     comp,
# ):
#     # with patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, tmp, clear=True):
#     with patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, tmp1, clear=True):
#         with patch.dict(thunder.torch._torch_to_thunder_function_map, tmp2, clear=True):
#             with patch.dict(thunder.core.jit_ext._minimal_lookaside_map, tmp3, clear=True):
#                 for sample in op.sample_inputs(device, dtype, requires_grad=True):
#                     comp = sample.comp if sample.comp is not None else comp
#                     tfn = thunder.jit(
#                         op.torch_reference,
#                         executors=executor.executors_list(),
#                     )
#                     with pytest.raises(
#                         NotImplementedError, match="^Exception encountered when doing automatic registration"
#                     ):
#                         thunder_result = tfn(*sample.args, **sample.kwargs)


# def test_torch_inplace_ops():
#     def func1(x):
#         return torch.relu_(x)

#     def func2(x):
#         return torch.nn.functional.selu(x, inplace=True)

#     with patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, {}, clear=True):
#         with patch.dict(thunder.torch._torch_to_thunder_function_map, {}, clear=True):
#             with patch.dict(thunder.core.jit_ext._minimal_lookaside_map, {}, clear=True):
#                 x = torch.rand((4, 4)).cuda()
#                 jfn = thunder.jit(func1, executors=[thunder.pytorch_executor])
#                 assert torch.relu_ not in thunder.torch._torch_to_thunder_function_map

#                 jfn = thunder.jit(func2, executors=[thunder.pytorch_executor])
#                 with pytest.raises(NotImplementedError, match="has inplace=True, please use manual registration$"):
#                     thunder_result = jfn(x)


# @ops(_fallback_opinfos1, supported_dtypes=(dtypes.float32,), supported_executors=(TorchExecutor,))
# def test_fallback_forward(op, device, dtype, executor, comp):
#     with patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, tmp1, clear=True):
#         with patch.dict(thunder.torch._torch_to_thunder_function_map, tmp2, clear=True):
#             with patch.dict(thunder.core.jit_ext._minimal_lookaside_map, tmp3, clear=True):
#                 for sample in op.sample_inputs(device, dtype):
#                     comp = sample.comp if sample.comp is not None else comp
#                     tfn = thunder.jit(
#                         op.torch_reference,
#                         executors=executor.executors_list(),
#                     )
#                     thunder_result = tfn(*sample.args, **sample.kwargs)
#                     torch_result = op.torch_reference(*sample.args, **sample.kwargs)
#                     comp(thunder_result, torch_result)
#                     fwd_trc = thunder.last_traces(tfn)[-1]
#                     assert any(
#                         bsym.sym.name.endswith(op.name) and not bsym.subsymbols for bsym in fwd_trc.bound_symbols
#                     )


# @ops(_fallback_opinfos2, supported_dtypes=(dtypes.float32,), supported_executors=(TorchExecutor,))
# def test_fallback_backward(op, device, dtype, executor, comp):
#     with patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, tmp1, clear=True):
#         with patch.dict(thunder.torch._torch_to_thunder_function_map, tmp2, clear=True):
#             with patch.dict(thunder.core.jit_ext._minimal_lookaside_map, tmp3, clear=True):
#                 for sample in op.sample_inputs(device, dtype, requires_grad=True):
#                     comp = sample.comp if sample.comp is not None else comp
#                     tfn = thunder.jit(
#                         op.torch_reference,
#                         executors=executor.executors_list(),
#                     )
#                     thunder_result = tfn(*sample.args, **sample.kwargs)

#                     bwd_trc = thunder.last_backward_traces(tfn)[-1]
#                     vjp_op_name = f"{op.name}_vjp"
#                     assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trc.bound_symbols)


# @instantiate(dtypes=(thunder.float32,), devicetypes=(devices.DeviceType.CUDA,), executors=(TorchExecutor,))
# def test_nanogpt_block(executor, device, dtype):
#     with patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, tmp1, clear=True):
#         with patch.dict(thunder.torch._torch_to_thunder_function_map, tmp2, clear=True):
#             with patch.dict(thunder.core.jit_ext._minimal_lookaside_map, tmp3, clear=True):
#                 import thunder.tests.nanogpt_model as nanogpt_model

#                 tdtype = thunder.torch.to_torch_dtype(dtype)
#                 make = partial(make_tensor, dtype=tdtype, device=device)

#                 config = nanogpt_model.GPTConfig(dropout=0)
#                 model = nanogpt_model.Block(config).to(device=device, dtype=tdtype)
#                 jitted = executor.make_callable(model)

#                 x = make((2, config.block_size, config.n_embd))

#                 cache_entry, _, _ = thunder.compile_data(jitted).get_computation_and_inputs(x)
#                 bwd_trcs = cache_entry.backward_traces
#                 for op in skip_ops:
#                     vjp_op_name = f"{op.name}_vjp"
#                     assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)


# @instantiate(dtypes=(thunder.float64,), devicetypes=(devices.DeviceType.CUDA,), executors=(TorchExecutor,))
# def test_alexnet(executor, device, dtype):
#     torchvision = pytest.importorskip("torchvision")

#     with patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, tmp1, clear=True):
#         with patch.dict(thunder.torch._torch_to_thunder_function_map, tmp2, clear=True):
#             with patch.dict(thunder.core.jit_ext._minimal_lookaside_map, tmp3, clear=True):
#                 tdtype = thunder.torch.to_torch_dtype(dtype)
#                 model = torchvision.models.alexnet(weights=None).to(device=device, dtype=tdtype)
#                 model = model.train()

#                 jitted = executor.make_callable(model)
#                 x = make_tensor((1, 3, 224, 224), dtype=tdtype, device=device)

#                 cache_entry, _, _ = thunder.compile_data(jitted).get_computation_and_inputs(x)
#                 bwd_trcs = cache_entry.backward_traces
#                 for op in skip_ops1:
#                     vjp_op_name = f"{op.name}_vjp"
#                     assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)
