import platform
import pytest
import torch
from torch._dynamo import is_inductor_supported

import thunder
from thunder.executors.torch_compile import (
    supported_ops,
    torch_compile_ex,
    torch_compile_cat_ex,
    torch_compile_xentropy_ex,
)
from thunder.executors.torchex import ex as pytorch_ex
from thunder.executors.nvfuserex import nvfuserex
from thunder.tests.bf16 import device_supports_bf16
from thunder.tests.framework import requiresCUDA
from torch.testing import assert_close


def test_supported_ops_are_in_pytorch_executor():
    """If this fails, the list of supported ops should be updated (or something went wrong)."""
    assert supported_ops - pytorch_ex.implmap.keys() == set()


# Disabling on windows temporarily, until our windows runners source the
# appropriate visual studio config.
@pytest.mark.skipif(not is_inductor_supported() or platform.system() == "Windows", reason="inductor unsupported")
def test_torch_compile_litgpt():
    from litgpt.model import GPT

    model = GPT.from_name("llama1-like", n_layer=1)
    x = torch.randint(model.max_seq_length, (2, 5))
    cmodel = thunder.jit(model, executors=[torch_compile_ex])
    _ = cmodel(x)
    forward_trace = thunder.last_traces(cmodel)[-1].python()
    # a single torch.compile region. normally you would want to enable sdpa too
    assert "TorchCompile0" in forward_trace
    assert "TorchCompile1" not in forward_trace


# Testing the following issue:
# https://github.com/Lightning-AI/lightning-thunder/issues/292 The issue was
# that the CSE pass was not being run correctly on the TorchCompile region.
# Here we test that everything works as expected.
@pytest.mark.skip(reason="https://github.com/NVIDIA/Fuser/issues/3688")
@pytest.mark.skipif(not is_inductor_supported(), reason="inductor unsupported")
@requiresCUDA
@pytest.mark.skipif(not device_supports_bf16(torch.device("cuda")), reason="bf16 is not supported")
def test_torch_compile_cat_nvfuser_phi2_tanh():
    from thunder.tests.litgpt_model import Config
    from litgpt.model import GPT

    device = torch.device("cuda")
    config = Config.from_name("phi-2", n_layer=1, gelu_approximate="tanh")
    with device:
        model = GPT(config).to(torch.bfloat16)
    x = torch.randint(model.max_seq_length, (5, 5), device=device)
    cmodel = thunder.jit(model, executors=[torch_compile_cat_ex, nvfuserex])
    logits = cmodel(x)
    logits.sum().backward()


@requiresCUDA
@pytest.mark.skipif(not is_inductor_supported(), reason="inductor unsupported")
@pytest.mark.skipif(not device_supports_bf16(torch.device("cuda")), reason="bf16 is not supported")
def test_torch_compile_cat_rope_single_fusion():
    from thunder.benchmarks import LlamaQKVSplitRopeBenchmark
    from thunder.examine import get_fusions

    bench = LlamaQKVSplitRopeBenchmark(
        config="Llama-3-8B",
        batchdims=(1,),
        device="cuda",
        dtype=thunder.bfloat16,
        requires_grad=True,
    )

    jfn = thunder.jit(bench.fn(), executors=[torch_compile_cat_ex, nvfuserex])
    args, kwargs = bench.make_batch()
    jfn(*args, **kwargs)
    forward_execution_trace = thunder.last_traces(jfn)[-1]
    assert len(get_fusions(forward_execution_trace)) == 1
    assert len(forward_execution_trace.bound_symbols) == 5

    backward_execution_trace = thunder.last_backward_traces(jfn)[-1]
    assert len(get_fusions(backward_execution_trace)) == 1
    assert len(backward_execution_trace.bound_symbols) == 15


@pytest.mark.skipif(not is_inductor_supported() or platform.system() == "Windows", reason="inductor unsupported")
def test_transform_for_execution_for_callable():
    def fn(a):
        return a.type("torch.DoubleTensor")

    a = torch.randn(3)
    jfn = thunder.jit(fn, executors=(thunder.executors.torch_compile.torch_compile_ex,))
    assert_close(jfn(a), fn(a))


@pytest.mark.skipif(not is_inductor_supported(), reason="inductor unsupported")
@requiresCUDA
@pytest.mark.skipif(not device_supports_bf16(torch.device("cuda")), reason="bf16 is not supported")
def test_litgpt_fabric_for_callable():
    from typing import Any
    from collections.abc import Callable
    from litgpt.model import Config, GPT
    import torch.nn as nn

    def jit(fn: Callable, executors: list[str]) -> Any:
        assert executors is not None
        return thunder.jit(fn, executors=executors)

    def forward_and_loss(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        logits = model(input_ids)
        return logits

    forward_and_loss_jitted = jit(forward_and_loss, executors=("sdpa", "torchcompile", "nvfuser", "torch"))

    config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)

    with torch.device("cuda"):
        model = GPT(config)

    input_ids = torch.zeros(1, 2, dtype=torch.int64, device="cuda")
    out = forward_and_loss(model, input_ids)
    out_jitted = forward_and_loss_jitted(model, input_ids)

    assert_close(out, out_jitted)


@requiresCUDA
def test_torch_compile_xentropy_loss():
    from transformers.loss.loss_utils import ForCausalLMLoss

    logits = torch.randn(1, 2, 6, device="cuda", requires_grad=True)
    logits = logits[..., :-1, :].contiguous()
    labels = torch.randint(0, 6, (1, 2), device="cuda")
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = 6

    closs_fn = thunder.jit(ForCausalLMLoss, executors=[torch_compile_xentropy_ex])
    _ = closs_fn(logits, labels, vocab_size, ignore_index=-1, shift_labels=shift_labels)
    forward_trace = thunder.last_traces(closs_fn)[-1].python()

    # make a single torch.compile region
    assert "TorchCompile0" in forward_trace
    assert "TorchCompile1" not in forward_trace
