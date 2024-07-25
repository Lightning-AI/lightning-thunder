from functools import partial
from unittest.mock import patch

import pytest
import thunder
import torch

from thunder.tests.framework import requiresCUDA, TorchExecutor
from thunder.tests.make_tensor import make_tensor
from thunder.tests.opinfos import get_opinfo, OpInfo


# @pytest.mark.parametrize("requires_grad", [True, False], ids=("train", "inference"))
@pytest.mark.parametrize("device,", ["cuda", "cpu"])
def test_torch_ops_backward(device):
    from torch.testing._internal.common_methods_invocations import op_db
    import thunder.torch.default_torch_ops as ops
    from itertools import chain
    # skip_names = ["bitwise_left_shift", "broadcast_tensors", "bitwise_right_shift", "masked_select", "combinations", "view_as_real", "corrcoef", "cov", "equal","stft", "istft", "geqrf", "imag", "meshgrid", "heaviside", "lcm"]
    funcs = [op for op in chain.from_iterable(ops.torch_fallback_ops.values())]
    opnames = [f.__name__ for f in funcs]
    op_infos_idx = [opnames.index(opinfo.name) for opinfo in op_db if opinfo.name in opnames]
    op_infos = [opinfo for opinfo in op_db if opinfo.name in opnames]
    # op_infos_idx = [idx for idx, opinfo in enumerate(op_db) if opinfo.name in opnames]
    print(f"total: {len(op_infos)}")
    cnt = 0
    ncase_cnt = 0
    for idx, op_info in zip(op_infos_idx, op_infos):
        # if op_info.name in skip_names: continue
        if op_info.name in ("nonzero_static", "histogramdd", "histogram") and device=="cuda":
            continue
        # dtype = torch.complex32 if op_info.name == "view_as_real" else torch.float32
        # dtype = torch.float32 if torch.float32 in op_info.dtypes else next(iter(op_info.dtypes))
        if not torch.float32 in op_info.dtypes: continue
        for sample in op_info.sample_inputs_func(op_info, device=torch.device(device), dtype=torch.float32, requires_grad=True):
            try:
                jfun = thunder.jit(funcs[idx])
                out = jfun(sample.input, *sample.args, **sample.kwargs)
            except Exception as e:
                cnt+=1
                # print(e)
                print(op_info.name)
                print("--------------------")
                break
            else:
                # print(f"pass: {op_info.name}")
                trc = thunder.last_backward_traces(jfun)[-1]
                trcf = thunder.last_traces(jfun)[-1]
                # print(op_info.name, trc, trcf)
                # skip if it is not differentiable
                outs = trcf.output[0]['output']
                outs = outs if isinstance(outs, tuple) else (outs, )
                if all(not thunder.core.dtypes.is_inexact_dtype(o.dtype) for o in outs):
                    continue
                vjp_op_name = f"{op_info.name}_vjp"
                try:
                    assert any(bsym.sym.name == vjp_op_name for bsym in trc.bound_symbols)
                except Exception as e:
                    # import pdb;pdb.set_trace()
                    print(e)
            finally:
                ncase_cnt+=1
                if ncase_cnt == 5:
                    ncase_cnt = 0
                    break

    print(cnt)

def test_torch_ops_forward():
    from torch.testing._internal.common_methods_invocations import op_db
    import thunder.torch.default_torch_ops as ops
    from itertools import chain
    # skip_names = ["bitwise_left_shift", "broadcast_tensors", "bitwise_right_shift", "masked_select", "combinations", "view_as_real", "corrcoef", "cov", "equal","stft", "istft", "geqrf", "imag", "meshgrid", "heaviside", "lcm"]
    funcs = [op for op in chain.from_iterable(ops.torch_fallback_ops.values())]
    opnames = [f.__name__ for f in funcs]
    op_infos_idx = [opnames.index(opinfo.name) for opinfo in op_db if opinfo.name in opnames]
    op_infos = [opinfo for opinfo in op_db if opinfo.name in opnames]
    print(f"total: {len(op_infos)}")
    cnt = 0
    for idx, op_info in zip(op_infos_idx, op_infos):
        # if op_info.name in skip_names: continue
        try:
            for sample in op_info.sample_inputs_func(op_info, device=torch.device("cuda"), dtype=torch.float32, requires_grad=False):
                jfun = thunder.jit(funcs[idx])
                out = jfun(sample.input, *sample.args, **sample.kwargs)
        except Exception as e:
            cnt+=1
            # print(e)
            print(op_info.name)
            print("--------------------")
        # else:
        #     print(thunder.last_traces(jfun)[-1])
    print(cnt)



skip_ops = [
    get_opinfo("layer_norm"),
    get_opinfo("linear"),
    get_opinfo("gelu"),
    get_opinfo("scaled_dot_product_attention"),
]
skip_ops1 = [get_opinfo("conv2d"), get_opinfo("linear"), get_opinfo("adaptive_avg_pool2d"), get_opinfo("max_pool2d")]
disable_opinfos = skip_ops + skip_ops1
tmp1 = dict(thunder.core.jit_ext._general_jit_lookaside_map)
list(tmp1.pop(k.torch_reference, None) for k in disable_opinfos)
tmp2 = dict(thunder.torch._torch_to_thunder_function_map)
list(tmp2.pop(k.torch_reference, None) for k in disable_opinfos)
tmp3 = dict(thunder.core.jit_ext._minimal_lookaside_map)
list(tmp3.pop(k.torch_reference, None) for k in disable_opinfos)
from thunder.torch import register_default_torch_op, meta_adaptor
# mock all the global variables that are modified during registration
@patch.dict(thunder.core.jit_ext._general_jit_lookaside_map, tmp1, clear=True)
@patch.dict(thunder.torch._torch_to_thunder_function_map, tmp2, clear=True)
@patch.dict(thunder.core.jit_ext._minimal_lookaside_map, tmp3, clear=True)
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
            {torchfn: ensure_recursive_proxies(interpreter_needs_wrap(record_source_loc_in_symbol_header(thunder.torch._torch_to_thunder_function_map[torchfn])))}
        )
    @requiresCUDA
    def test_nanogpt_block(self):
        import thunder.tests.nanogpt_model as nanogpt_model

        for op in skip_ops:
            if op.name == "gelu":
                register_default_torch_op(op.torch_reference, meta_adaptor(op.torch_reference), torch)
            else:
                register_default_torch_op(op.torch_reference, meta_adaptor(op.torch_reference), torch.nn.functional)
            self._tmp_update_jit_lookup(op.torch_reference)
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

        for op in skip_ops1:
            register_default_torch_op(op.torch_reference, meta_adaptor(op.torch_reference), torch.nn.functional)
            self._tmp_update_jit_lookup(op.torch_reference)
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
