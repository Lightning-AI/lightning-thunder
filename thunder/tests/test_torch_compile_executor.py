import torch
from thunder.executors.torchex import ex as pytorch_ex
from thunder.executors.torch_compile import supported_ops, torch_compile_ex
from thunder.tests.litgpt_model import GPT
import thunder


def test_supported_ops_are_in_pytorch_executor():
    """If this fails, the list of supported ops should be updated (or something went wrong)."""
    assert supported_ops - pytorch_ex.implmap.keys() == set()


def test_torch_compile_litgpt():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with device:
        model = GPT.from_name("llama1-like", n_layer=1)
        x = torch.randint(model.max_seq_length, (2, 5))
    cmodel = thunder.jit(model, executors=[torch_compile_ex])
    y = cmodel(x)
    forward_trace = thunder.last_traces(cmodel)[-1].python()
    assert "TorchCompile0" in forward_trace
    assert "sdpaex_grad_forward_scaled_dot_product_efficient_attention" in forward_trace
    assert "TorchCompile1" in forward_trace
