import pytest

import torch
from torch.testing import assert_close

import litgpt

import thunder
import thunder.core
from thunder.tests.litgpt_model import GPT
from thunder.tests.framework import requiresCUDA
from thunder.tests import litgpt_model

from thunder.executors.ligerex import liger_ex


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_rms_norm(device: str, dtype: torch.dtype):
    hidden_size = 64

    x = torch.randn(32, 10, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.ones(hidden_size, device=device, dtype=dtype, requires_grad=True)
    eps = 1e-5

    def fn(x, weight, eps):
        return torch.nn.functional.rms_norm(x, (hidden_size,), weight, eps)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, weight, eps)
    torch_result = fn(x, weight, eps)

    go = torch.randn_like(torch_result)
    grad_ref, grad_ref_weight = torch.autograd.grad(torch_result, (x, weight), go)
    grad_res, grad_res_weight = torch.autograd.grad(thunder_result, (x, weight), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)
    assert_close(grad_ref_weight, grad_res_weight)

    assert thunder.executors.ligerex.liger_rms_norm_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_rms_norm_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }


class MergeGegluTransform(thunder.core.transform_common.Transform):
    def transform_traces_per_prologue(self, prologue_trace, compute_trace, epilogue_trace, **kwargs):
        new_compute_trace = thunder.core.trace.from_trace(compute_trace)
        bound_symbols = compute_trace.bound_symbols[:]
        while bound_symbols:
            bsym = bound_symbols.pop(0)
            if bsym.sym == thunder.executors.ligerex.liger_geglu:
                for i, bsym2 in enumerate(bound_symbols):
                    assert not any(o is bsym.output for o in bsym2.flat_outs)
                    if bsym2.sym == thunder.executors.ligerex.liger_geglu:
                        break
                bsym2 = bound_symbols.pop(i)
                assert bsym2.sym == thunder.executors.ligerex.liger_geglu

                output = (bsym.output, bsym2.output)
                args = (bsym.args[0], bsym2.args[0], *bsym.args[1:])

                new_compute_trace.bound_symbols.append(bsym.from_bsym(args=args, output=output, sym=liger_geglu))
            else:
                new_compute_trace.bound_symbols.append(bsym.from_bsym())
        new_compute_trace.set_provenance(thunder.core.trace.TraceProvenance(self.__class__))
        return prologue_trace, new_compute_trace, epilogue_trace


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_geglu(device: str, dtype: torch.dtype):
    hidden_size = 64

    x = torch.randn(32, 10, hidden_size, device=device, dtype=dtype, requires_grad=True)
    _b = torch.randn(32, 10, hidden_size, device=device, dtype=dtype, requires_grad=True)

    linear = torch.nn.Linear(hidden_size // 2, hidden_size * 2).to(device).to(dtype)

    def fn(x, y):
        return x * torch.nn.functional.gelu(y)

    jfn = thunder.jit(fn, executors=[liger_ex], transforms=(MergeGegluTransform(),))

    thunder_result = jfn(x, _b)
    torch_result = fn(x, _b)

    go = torch.randn_like(torch_result)
    grad_ref_a, grad_ref_b = torch.autograd.grad(torch_result, (x, _b), go)
    grad_res_a, grad_res_b = torch.autograd.grad(thunder_result, (x, _b), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref_a, grad_res_a)
    assert_close(grad_res_b, grad_res_b)


class MergeRopeTransform(thunder.core.transform_common.Transform):
    def transform_traces_pre_prologue(self, prologue_trace, compute_trace, epilogue_trace, **kwargs):
        new_compute_trace = thunder.core.trace.from_trace(compute_trace)
        bound_symbols = compute_trace.bound_symbols[:]
        while bound_symbols:
            bsym = bound_symbols.pop(0)
            if bsym.sym == thunder.executors.ligerex.litgpt_apply_rope:
                for i, bsym2 in enumerate(bound_symbols):
                    assert not any(o is bsym.output for o in bsym2.flat_outs)
                    if bsym2.sym == thunder.executors.ligerex.litgpt_apply_rope:
                        break
                bsym2 = bound_symbols.pop(i)
                assert bsym2.sym == thunder.executors.ligerex.litgpt_apply_rope

                output = (bsym.output, bsym2.output)
                args = (bsym.args[0], bsym2.args[0], *bsym.args[1:])

                new_compute_trace.bound_symbols.append(
                    bsym.from_bsym(args=args, output=output, sym=thunder.executors.ligerex.liger_rope)
                )
            else:
                new_compute_trace.bound_symbols.append(bsym.from_bsym())
        new_compute_trace.set_provenance(thunder.core.trace.TraceProvenance(self.__class__))
        return prologue_trace, new_compute_trace, epilogue_trace


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_rope(device: str, dtype: torch.dtype):
    hidden_size = 64

    cfg = litgpt_model.Config.from_name("llama2-like", n_layer=1)

    model = litgpt_model.GPT(cfg)
    model = model.to(device)
    model.max_seq_length = 1024
    model.set_kv_cache(1)

    model.mask_cache = model.mask_cache.to(device)

    for layer in model.transformer.h:
        if hasattr(layer.attn, "kv_cache"):
            layer.attn.kv_cache.k = layer.attn.kv_cache.k.to(device)
            layer.attn.kv_cache.v = layer.attn.kv_cache.v.to(device)

    inp = torch.arange(1, 6, dtype=torch.int64, device=device)[None]
    inp_pos = torch.arange(1, 6, dtype=torch.int64, device=device)

    jfn = thunder.jit(model, executors=[liger_ex], transforms=(MergeRopeTransform(),))

    thunder_result = jfn(inp, inp_pos)
    torch_result = model(inp, inp_pos)

    assert_close(thunder_result, torch_result)


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_layer_norm(device: str, dtype: torch.dtype):
    hidden_size = 64

    x = torch.randn(32, 10, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.ones(hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.zeros(hidden_size, device=device, dtype=dtype, requires_grad=True)
    eps = 1e-5

    def fn(x, weight, bias, eps):
        return torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, weight, bias, eps)
    torch_result = fn(x, weight, bias, eps)

    go = torch.randn_like(torch_result)
    grad_ref, grad_ref_weight = torch.autograd.grad(torch_result, (x, weight), go)
    grad_res, grad_res_weight = torch.autograd.grad(thunder_result, (x, weight), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)
    assert_close(grad_ref_weight, grad_res_weight)

    assert thunder.executors.ligerex.liger_layer_norm_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_layer_norm_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_cross_entropy(device: str, dtype: torch.dtype):
    hidden_size = 64
    batch_size = 32
    num_classes = 100

    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_classes, hidden_size, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, num_classes, (batch_size,), device=device)

    def fn(x, target, weight):
        logits = torch.matmul(x, weight.t())
        return torch.nn.functional.cross_entropy(logits, target)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, target, weight)
    torch_result = fn(x, target, weight)

    go = torch.randn_like(torch_result)
    grad_ref = torch.autograd.grad(torch_result, (x,), go)
    grad_res = torch.autograd.grad(thunder_result, (x,), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)

    assert thunder.executors.ligerex.liger_cross_entropy_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_cross_entropy_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }


class FuseSwigLUTransform(thunder.core.transform_common.Transform):
    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        _, consumers = thunder.core.utils.producers_and_consumers(computation_trace)
        new_computation_trace = thunder.core.trace.from_trace(computation_trace)
        bsyms_to_skip = set()
        for b in computation_trace.bound_symbols:
            if b in bsyms_to_skip:
                continue
            new_bsym = b
            if b.sym == thunder.torch.silu:
                c = consumers[b.output]
                if len(c) == 1 and c[0].sym == thunder.torch.mul:
                    (mul,) = c
                    mul_l, mul_r = mul.args
                    if mul_l is b.output:
                        other = mul_r
                    else:
                        other = mul_l
                    new_bsym = b.from_bsym(
                        sym=thunder.executors.ligerex.liger_swiglu_forward,
                        output=mul.output,
                        args=(b.args[0], other),
                        subsymbols=[],
                    )
                    bsyms_to_skip.add(mul)
            new_computation_trace.bound_symbols.append(new_bsym)
        new_computation_trace.set_provenance(thunder.core.trace.TraceProvenance("constructed by FuseSwigLU"))
        return prologue_trace, new_computation_trace, epilogue_trace


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_swiglu(device: str, dtype: torch.dtype):
    _input = torch.randn(2, 8, 8, device=device, dtype=dtype, requires_grad=True)
    _b = torch.randn(2, 8, 8, device=device, dtype=dtype, requires_grad=True)

    def fn(a, b):
        return torch.nn.functional.silu(a) * b

    jfn = thunder.jit(fn, executors=[liger_ex], transform=(FuseSwigLUTransform(),))

    thunder_result = jfn(_input, _b)
    torch_result = fn(_input, _b)

    go = torch.randn_like(torch_result)
    grad_ref, grad_ref_weight = torch.autograd.grad(torch_result, (_input, _b), go)
    grad_res, grad_res_weight = torch.autograd.grad(thunder_result, (_input, _b), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)
    assert_close(grad_ref_weight, grad_res_weight)


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_kernel_kl_div(device: str, dtype: torch.dtype):
    x = torch.randn(2, 3, device=device, dtype=dtype, requires_grad=True)
    x = torch.log_softmax(x, dim=1)
    target = torch.randn(2, 3, device=device, dtype=dtype, requires_grad=True)
    target = torch.softmax(target, dim=1)

    def fn(x, target):
        return torch.nn.functional.kl_div(x, target, reduction="none", log_target=False)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, target)
    torch_result = fn(x, target)

    go = torch.randn_like(torch_result)
    grad_ref = torch.autograd.grad(torch_result, x, go)
    grad_res = torch.autograd.grad(thunder_result, x, go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)

    assert thunder.executors.ligerex.liger_kl_div_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_kl_div_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }


class FuseLinearCrossEntropyTransform(thunder.core.transform_common.Transform):
    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        _, consumers = thunder.core.utils.producers_and_consumers(computation_trace)
        new_computation_trace = thunder.core.trace.from_trace(computation_trace)
        bsyms_to_skip = set()
        for b in computation_trace.bound_symbols:
            if b in bsyms_to_skip:
                continue
            new_bsym = b
            if b.sym == thunder.torch.linear:
                c = consumers[b.output]
                if len(c) == 1 and c[0].sym == thunder.torch.cross_entropy:
                    (ce,) = c
                    assert not ce.kwargs
                    assert not b.kwargs
                    assert ce.args[0] is b.output
                    inp, weight, bias = b.args
                    _, targets, ce_weight, size_average, ignore_index, reduce, reduction, label_smoothing = ce.args
                    assert ce_weight is None
                    assert size_average is None
                    assert reduce is None
                    new_bsym = b.from_bsym(
                        sym=thunder.executors.ligerex.liger_fused_linear_cross_entropy_forward,
                        output=ce.output,
                        args=(inp, weight, targets, bias, ignore_index, label_smoothing, reduction),
                        subsymbols=[],
                    )
                    bsyms_to_skip.add(ce)
            new_computation_trace.bound_symbols.append(new_bsym)
        new_computation_trace.set_provenance(
            thunder.core.trace.TraceProvenance("constructed by FuseLinearCrossEntropy")
        )
        return prologue_trace, new_computation_trace, epilogue_trace


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_fused_linear_cross_entropy(device: str, dtype: torch.dtype):
    batch_size = 32
    hidden_size = 64
    num_classes = 100

    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(num_classes, hidden_size, device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, num_classes, (batch_size,), device=device)
    bias = torch.zeros(num_classes, device=device, dtype=dtype, requires_grad=True)

    def fn(x, weight, target, bias):
        logits = torch.nn.functional.linear(x, weight, bias)
        loss = torch.nn.functional.cross_entropy(logits, target)
        return loss

    jfn = thunder.jit(fn, executors=[liger_ex], transforms=(FuseLinearCrossEntropyTransform(),))

    thunder_result = jfn(x, weight, target, bias)

    logits = torch.nn.functional.linear(x, weight, bias)
    torch_result = fn(x, weight, target, bias)

    go = torch.randn_like(torch_result)
    grad_ref, grad_ref_weight, grad_ref_bias = torch.autograd.grad(torch_result, (x, weight, bias), go)
    grad_res, grad_res_weight, grad_res_bias = torch.autograd.grad(thunder_result, (x, weight, bias), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)
    assert_close(grad_ref_weight, grad_res_weight)
    assert_close(grad_ref_bias, grad_ref_bias)


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device", ["cuda"])
@requiresCUDA
def test_liger_group_norm(device: str, dtype: torch.dtype):
    num_groups = 3
    num_channels = 6
    hidden_size = 64

    x = torch.randn(32, num_channels, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.ones(num_channels, device=device, dtype=dtype, requires_grad=True)
    bias = torch.zeros(num_channels, device=device, dtype=dtype, requires_grad=True)

    def fn(x, n, w, b):
        return torch.nn.functional.group_norm(x, n, w, b, eps=1e-5)

    jfn = thunder.jit(fn, executors=[liger_ex])

    thunder_result = jfn(x, num_groups, weight, bias)
    torch_result = fn(x, num_groups, weight, bias)

    go = torch.randn_like(torch_result)
    grad_ref, grad_ref_weight, grad_ref_bias = torch.autograd.grad(torch_result, (x, weight, bias), go)
    grad_res, grad_res_weight, grad_res_bias = torch.autograd.grad(thunder_result, (x, weight, bias), go)

    assert_close(thunder_result, torch_result)
    assert_close(grad_ref, grad_res)
    assert_close(grad_ref_weight, grad_res_weight)
    assert_close(grad_ref_bias, grad_res_bias, rtol=1e-5, atol=1e-5)

    assert thunder.executors.ligerex.liger_group_norm_forward in {
        bsym.sym for bsym in thunder.last_traces(jfn)[-1].bound_symbols
    }
    assert thunder.executors.ligerex.liger_group_norm_backward in {
        bsym.sym for bsym in thunder.last_backward_traces(jfn)[-1].bound_symbols
    }
