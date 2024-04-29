import sys

import pytest
import torch

import thunder
from thunder.executors.torch_compile import supported_ops, torch_compile_ex
from thunder.executors.torchex import ex as pytorch_ex
from thunder.tests.litgpt_model import GPT


def test_supported_ops_are_in_pytorch_executor():
    """If this fails, the list of supported ops should be updated (or something went wrong)."""
    assert supported_ops - pytorch_ex.implmap.keys() == set()


@pytest.mark.xfail(sys.platform == "win32", reason="unicode error in torch.compile", raises=SyntaxError, strict=True)
def test_torch_compile_litgpt():
    model = GPT.from_name("llama1-like", n_layer=1)
    x = torch.randint(model.max_seq_length, (2, 5))
    cmodel = thunder.jit(model, executors=[torch_compile_ex])
    _ = cmodel(x)
    forward_trace = thunder.last_traces(cmodel)[-1].python()
    # a single torch.compile region. normally you would want to enable sdpa too
    assert "TorchCompile0" in forward_trace
    assert "TorchCompile1" not in forward_trace
