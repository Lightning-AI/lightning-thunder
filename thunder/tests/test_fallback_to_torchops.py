from functools import partial
from unittest.mock import patch

import pytest
import thunder
import torch

from thunder.tests.framework import requiresCUDA, TorchExecutor
from thunder.tests.make_tensor import make_tensor
from thunder.tests.opinfos import get_opinfo, OpInfo

_unsupported_opinfos = [get_opinfo("zeta"), get_opinfo("nextafter"), get_opinfo("item")]
_fallback_opinfos1 = [get_opinfo("embedding"), get_opinfo("type_as"), get_opinfo("conv2d"), get_opinfo("reshape")]
_fallback_opinfos2 = [get_opinfo("conv2d"), get_opinfo("embedding"), get_opinfo("unfold"), get_opinfo("expand")]
skip_ops = [
    get_opinfo("layer_norm"),
    get_opinfo("linear"),
    get_opinfo("gelu"),
    get_opinfo("scaled_dot_product_attention"),
]
skip_ops1 = [get_opinfo("conv2d"), get_opinfo("linear"), get_opinfo("adaptive_avg_pool2d"), get_opinfo("max_pool2d")]
disable_opinfos = _unsupported_opinfos + _fallback_opinfos1 + _fallback_opinfos2 + skip_ops + skip_ops1
tmp1 = dict(thunder.core.jit_ext._general_jit_lookaside_map)
list(tmp1.pop(k.torch_reference, None) for k in disable_opinfos)
tmp2 = dict(thunder.torch._torch_to_thunder_function_map)
list(tmp2.pop(k.torch_reference, None) for k in disable_opinfos)
tmp3 = dict(thunder.core.jit_ext._minimal_lookaside_map)
list(tmp3.pop(k.torch_reference, None) for k in disable_opinfos)

# mock all the global variables that are modified during registration
@patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, tmp1, clear=True)
@patch.dict(thunder.torch._torch_to_thunder_function_map, tmp2, clear=True)
@patch.dict(thunder.core.jit_ext._minimal_lookaside_map, tmp3, clear=True)
@patch.dict(thunder.executors.torchex.ex._implmap, {})
@patch.dict(thunder.executors.torchex.ex._opmap, {})
@patch.dict(thunder.core.transforms.augmented_forward_impls, {})
@patch.dict(thunder.core.transforms.backward_impls, {})
class TestFallbackToTorch:
    def test_torch_inplace_ops(self):
        def func1(x):
            return torch.relu_(x)

        def func2(x):
            return torch.nn.functional.selu(x, inplace=True)

        with patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, {}, clear=True):
            with patch.dict(thunder.torch._torch_to_thunder_function_map, {}, clear=True):
                with patch.dict(thunder.core.jit_ext._minimal_lookaside_map, {}, clear=True):
                    x = torch.rand((4, 4))
                    jfn = thunder.jit(func1, executors=[thunder.pytorch_executor], enable_fallback_to_torch=True)
                    assert torch.relu_ not in thunder.torch._torch_to_thunder_function_map

                    jfn = thunder.jit(func2, executors=[thunder.pytorch_executor], enable_fallback_to_torch=True)
                    with pytest.raises(NotImplementedError, match="has inplace=True, please use manual registration$"):
                        thunder_result = jfn(x)

    @pytest.mark.parametrize("op", _fallback_opinfos2, ids=[op.name for op in _fallback_opinfos2])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fallback_backward(self, op, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        dtype = torch.float32
        for sample in op.sample_inputs(device, dtype, requires_grad=True):
            executor = TorchExecutor
            tfn = thunder.jit(op.torch_reference, executors=executor.executors_list(), enable_fallback_to_torch=True)
            cache_entry, _, _ = thunder.compile_data(tfn).get_computation_and_inputs(*sample.args, **sample.kwargs)
            bwd_trcs = cache_entry.backward_traces

            vjp_op_name = f"{op.name}_vjp"
            assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)

    @pytest.mark.parametrize("op", _fallback_opinfos1, ids=[op.name for op in _fallback_opinfos1])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fallback_forward(self, op, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        dtype = torch.float32
        executor = TorchExecutor
        for sample in op.sample_inputs(device, dtype):
            tfn = thunder.jit(op.torch_reference, executors=executor.executors_list(), enable_fallback_to_torch=True)
            cache_entry, _, _ = thunder.compile_data(tfn).get_computation_and_inputs(*sample.args, **sample.kwargs)
            trcs = cache_entry.computation_traces
            assert any(bsym.sym.name.endswith(op.name) and not bsym.subsymbols for bsym in trcs[-1].bound_symbols)

    @pytest.mark.parametrize("op", _unsupported_opinfos, ids=[op.name for op in _unsupported_opinfos])
    def test_fallback_exception(self, op: OpInfo):
        dtype = torch.float32
        executor = TorchExecutor
        for sample in op.sample_inputs("cpu", dtype, requires_grad=True):
            tfn = thunder.jit(op.torch_reference, executors=executor.executors_list(), enable_fallback_to_torch=True)
            with pytest.raises(NotImplementedError, match="^Exception encountered when doing automatic registration"):
                thunder_result = tfn(*sample.args, **sample.kwargs)

    @requiresCUDA
    def test_nanogpt_block(self):
        import thunder.tests.nanogpt_model as nanogpt_model

        tdtype = torch.float32  # thunder.torch.to_torch_dtype(dtype)
        device = torch.device("cuda")
        executor = TorchExecutor
        make = partial(make_tensor, dtype=tdtype, device=device)

        config = nanogpt_model.GPTConfig(dropout=0)
        model = nanogpt_model.Block(config).to(device=device, dtype=tdtype)
        jitted = executor.make_callable(model, enable_fallback_to_torch=True)

        x = make((2, config.block_size, config.n_embd))

        cache_entry, _, _ = thunder.compile_data(jitted).get_computation_and_inputs(x)
        bwd_trcs = cache_entry.backward_traces
        for op in skip_ops:
            vjp_op_name = f"{op.name}_vjp"
            assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)

    @requiresCUDA
    def test_alexnet(self):
        torchvision = pytest.importorskip("torchvision")

        tdtype = torch.float32
        device = torch.device("cuda")
        model = torchvision.models.alexnet(weights=None).to(device=device, dtype=tdtype)
        model = model.train()

        executor = TorchExecutor
        jitted = executor.make_callable(model, enable_fallback_to_torch=True)
        x = make_tensor((1, 3, 224, 224), dtype=tdtype, device=device)

        cache_entry, _, _ = thunder.compile_data(jitted).get_computation_and_inputs(x)
        bwd_trcs = cache_entry.backward_traces
        for op in skip_ops1:
            vjp_op_name = f"{op.name}_vjp"
            assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)
