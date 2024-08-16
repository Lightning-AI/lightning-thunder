from functools import partial
from unittest.mock import patch

import pytest
import thunder
import thunder.torch.default_torch_ops as ops
import torch

from thunder.tests.framework import requiresCUDA, TorchExecutor
from thunder.tests.make_tensor import make_tensor
from thunder.tests.opinfos import get_opinfo, OpInfo
from thunder.tests.test_einops import skipIfNoCUDA
from torch.testing._internal.common_device_type import skipCPUIfNoLapack
from torch.testing._internal.common_methods_invocations import op_db

_name2func = {}
[_name2func.setdefault(v.__name__, v) for v in ops.torch_auto_registered_ops[torch]]
[_name2func.setdefault(f"nn.functional.{v.__name__}", v) for v in ops.torch_auto_registered_ops[torch.nn.functional]]
# Use the sample input from torch.xx to test torch.tensor.xx
[_name2func.setdefault(f"Tensor.{v.__name__}", v) for v in ops.torch_auto_registered_ops[torch.Tensor]]
# NOTE some `opinfo.name`s don't have torch.xxx but are only used as torch.Tensor.xxx
torch_tensor_space_names = [v.__name__ for v in ops.torch_auto_registered_ops[torch.Tensor]]
_opinfos = [opinfo for opinfo in op_db if opinfo.name in _name2func or opinfo.name in torch_tensor_space_names]


# Note that successfully catching an exception in this test is also noted as passed
@skipIfNoCUDA
@pytest.mark.parametrize("op_info,", _opinfos, ids=list(map(lambda opinfo: opinfo.name, _opinfos)))
@pytest.mark.parametrize("requires_grad", [True, False], ids=("train", "inference"))
@pytest.mark.parametrize("device,", ["cuda", "cpu"])
def test_torch_ops_trace(device, requires_grad, op_info):
    from itertools import islice

    # for op_info in op_infos:
    if device == "cuda" and torch.float32 not in op_info.dtypesIfCUDA:
        return
    if device == "cpu" and not torch.float32 in op_info.dtypes:
        return
    # No cuda backend support
    if op_info.name in ("nonzero_static",) and device == "cuda":
        return
    if device == "cpu" and not torch._C.has_lapack and skipCPUIfNoLapack in op_info.decorators:
        return

    def get_method(op_info):
        # Check if we have registered this method.
        if _name2func.get(f"Tensor.{op_info.name}", None):
            # Call the method as `x.method(*args, **kwargs)`
            # We have a different path for `torch.Tensor.method` and `x.method`.
            return lambda x, *args, **kwargs: getattr(x, f"{op_info.name}")(*args, **kwargs)
        return None

    funcs = [_name2func.get(op_info.name, None), get_method(op_info)]
    for func in funcs:
        if func is None:
            continue
        # It takes too long, test only the first 5 sample inputs
        gen = islice(
            op_info.sample_inputs_func(
                op_info, device=torch.device(device), dtype=torch.float32, requires_grad=requires_grad
            ),
            5,
        )
        for sample in gen:
            try:
                jfun = thunder.jit(func)
                out = jfun(sample.input, *sample.args, **sample.kwargs)
            except Exception as e:
                assert isinstance(e, NotImplementedError)
                assert str(e).startswith(f"Exception encountered when doing automatic registration") or str(
                    e
                ).startswith(f"Unsupported type:")
                break
            else:
                if requires_grad:
                    trc = thunder.last_backward_traces(jfun)[-1]
                    fwd_trc = thunder.last_traces(jfun)[-1]
                    # skip if it is not differentiable
                    outs = fwd_trc.output[0]["output"]
                    outs = outs if isinstance(outs, tuple) else (outs,)
                    if all(not thunder.core.dtypes.is_inexact_dtype(o.dtype) for o in outs):
                        continue
                    vjp_op_name = f"{op_info.name.split('.')[-1]}_vjp"
                    if op_info.name == "mm":
                        assert any(bsym.sym.name.endswith(vjp_op_name) for bsym in trc.bound_symbols)
                    else:
                        assert any(bsym.sym.name == vjp_op_name for bsym in trc.bound_symbols)
                else:
                    fwd_trc = thunder.last_traces(jfun)[-1]
                    assert any(
                        bsym.sym.name.endswith(op_info.name.split(".")[-1]) and not bsym.subsymbols
                        for bsym in fwd_trc.bound_symbols
                    )


# Replace manual registration of some operations with automatic registration for network test cases
_skip_ops_nanogpt = [
    get_opinfo("layer_norm"),
    get_opinfo("linear"),
    get_opinfo("gelu"),
    get_opinfo("scaled_dot_product_attention"),
]
_skip_ops_alexnet = [
    get_opinfo("conv2d"),
    get_opinfo("linear"),
    get_opinfo("adaptive_avg_pool2d"),
    get_opinfo("max_pool2d"),
]
_disable_opinfos = _skip_ops_nanogpt + _skip_ops_alexnet
_tmp_general_jit_lookaside_map = dict(thunder.core.jit_ext._general_jit_lookaside_map)
list(_tmp_general_jit_lookaside_map.pop(k.torch_reference, None) for k in _disable_opinfos)
_tmp_torch_to_thunder_function_map = dict(thunder.torch._torch_to_thunder_function_map)
list(_tmp_torch_to_thunder_function_map.pop(k.torch_reference, None) for k in _disable_opinfos)
_tmp_minimal_lookaside_map = dict(thunder.core.jit_ext._minimal_lookaside_map)
list(_tmp_minimal_lookaside_map.pop(k.torch_reference, None) for k in _disable_opinfos)
from thunder.torch import register_default_torch_op


# mock all the global variables that are modified during registration
@patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, _tmp_general_jit_lookaside_map, clear=True)
@patch.dict(thunder.torch._torch_to_thunder_function_map, _tmp_torch_to_thunder_function_map, clear=True)
@patch.dict(thunder.core.jit_ext._minimal_lookaside_map, _tmp_minimal_lookaside_map, clear=True)
@patch.dict(thunder.executors.torchex.ex._implmap, {})
@patch.dict(thunder.executors.torchex.ex._opmap, {})
@patch.dict(thunder.core.transforms.augmented_forward_impls, {})
@patch.dict(thunder.core.transforms.backward_impls, {})
class TestFallbackToTorch:
    def _tmp_update_jit_lookup(self, torchfn):
        from thunder.core.interpreter import interpreter_needs_wrap
        from thunder.core.jit_ext import (
            _general_jit_lookaside_map,
            ensure_recursive_proxies,
            record_source_loc_in_symbol_header,
        )

        _general_jit_lookaside_map.update(
            {
                torchfn: ensure_recursive_proxies(
                    interpreter_needs_wrap(
                        record_source_loc_in_symbol_header(thunder.torch._torch_to_thunder_function_map[torchfn])
                    )
                )
            }
        )

    @requiresCUDA
    def test_nanogpt_block(self):
        import thunder.tests.nanogpt_model as nanogpt_model

        for op in _skip_ops_nanogpt:
            if op.name == "gelu":
                register_default_torch_op(op.torch_reference, torch)
            else:
                register_default_torch_op(op.torch_reference, torch.nn.functional)
            self._tmp_update_jit_lookup(op.torch_reference)
        tdtype = torch.float32
        device = torch.device("cuda")
        executor = TorchExecutor
        make = partial(make_tensor, dtype=tdtype, device=device)

        config = nanogpt_model.GPTConfig(dropout=0)
        model = nanogpt_model.Block(config).to(device=device, dtype=tdtype)
        jitted = executor.make_callable(model)

        x = make((2, config.block_size, config.n_embd))

        cache_entry, _, _ = thunder.compile_data(jitted).get_computation_and_inputs(x)
        bwd_trcs = cache_entry.backward_traces
        for op in _skip_ops_nanogpt:
            vjp_op_name = f"{op.name}_vjp"
            assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)

    @requiresCUDA
    def test_alexnet(self):
        torchvision = pytest.importorskip("torchvision")

        for op in _skip_ops_alexnet:
            register_default_torch_op(op.torch_reference, torch.nn.functional)
            self._tmp_update_jit_lookup(op.torch_reference)
        tdtype = torch.float32
        device = torch.device("cuda")
        model = torchvision.models.alexnet(weights=None).to(device=device, dtype=tdtype)
        model = model.train()

        executor = TorchExecutor
        jitted = executor.make_callable(model)
        x = make_tensor((1, 3, 224, 224), dtype=tdtype, device=device)

        cache_entry, _, _ = thunder.compile_data(jitted).get_computation_and_inputs(x)
        bwd_trcs = cache_entry.backward_traces
        for op in _skip_ops_alexnet:
            vjp_op_name = f"{op.name}_vjp"
            assert any(bsym.sym.name == vjp_op_name for bsym in bwd_trcs[-1].bound_symbols)
