import platform
import pytest
import torch
from torch._dynamo import is_inductor_supported

import thunder
from thunder.executors.torch_compile import supported_ops, torch_compile_ex, torch_compile_cat_ex
from thunder.executors.torchex import ex as pytorch_ex
from thunder.executors.nvfuserex import nvfuserex
from thunder.tests.bf16 import device_supports_bf16
from thunder.tests.framework import requiresCUDA


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
@pytest.mark.skipif(not is_inductor_supported(), reason="inductor unsupported")
@requiresCUDA
@pytest.mark.skipif(not device_supports_bf16(torch.device("cuda")), reason="bf16 is not supported")
def test_torch_compile_cat_nvfuser_phi2_tanh():
    from litgpt.config import Config
    from litgpt.model import GPT

    device = torch.device("cuda")
    config = Config.from_name("phi-2", n_layer=1, gelu_approximate="tanh")
    with device:
        model = GPT(config).to(torch.bfloat16)
    x = torch.randint(model.max_seq_length, (5, 5), device=device)
    cmodel = thunder.jit(model, executors=[torch_compile_cat_ex, nvfuserex])
    logits = cmodel(x)
    logits.sum().backward()


@pytest.mark.skipif(not is_inductor_supported(), reason="inductor unsupported")
@requiresCUDA
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
    assert len(backward_execution_trace.bound_symbols) == 14
