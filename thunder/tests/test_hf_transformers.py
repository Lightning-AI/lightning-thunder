import pytest
import torch

from thunder import jit

from thunder.dynamo import thunderfx


def skip_if_no_cuda(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")


def skip_if_no_bf16(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("Requires CUDA BF16 support")


def xfail_not_working(device, compile, model_id):
    # https://github.com/Lightning-AI/lightning-thunder/issues/2038
    if compile == jit and model_id == "microsoft/Phi-3-mini-128k-instruct":
        pytest.xfail("Known issue with jit compilation for this model")

    # comparison check for grads fails on CUDA for this model
    # https://github.com/Lightning-AI/lightning-thunder/issues/2040
    if device == "cuda" and model_id == "Qwen/Qwen2.5-7B-Instruct":
        pytest.xfail("Known issue with Qwen model on CUDA")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("compile", [thunderfx, jit], ids=["thunderfx", "thunderjit"])
@pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3-mini-128k-instruct"])
def test_hf(device, compile, model_id) -> None:
    skip_if_no_cuda(device)
    skip_if_no_bf16(device)
    xfail_not_working(device, compile, model_id)

    from transformers import AutoConfig, AutoModelForCausalLM

    configuration = AutoConfig.from_pretrained(
        model_id,
        # Scaled down for testing
        vocab_size=16,
        pad_token_id=15,
        max_position_embeddings=32,
        num_hidden_layers=1,
        _attn_implementation="sdpa",
    )
    configuration.hidden_size = configuration.num_attention_heads
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(configuration).to(torch.float64)

    compiled_model = compile(model)

    input_ids = torch.arange(start=0, end=configuration.vocab_size, device=device).unsqueeze(0)
    ref_output = model(input_ids=input_ids, labels=input_ids)
    ref_loss = ref_output.loss

    compiled_output = compiled_model(input_ids=input_ids, labels=input_ids)
    compiled_loss = compiled_output.loss

    # Less strict tolerance probably due to different type promotion order for bfloat16
    # See the comments in the linked issue for more details
    # https://github.com/Lightning-AI/lightning-thunder/issues/1407
    torch.testing.assert_close(compiled_loss, ref_loss, rtol=1e-2, atol=1e-2)

    loss_grad = torch.randn_like(compiled_loss)
    grads_ref = torch.autograd.grad(ref_loss, model.parameters(), grad_outputs=loss_grad)
    grads_compiled = torch.autograd.grad(compiled_loss, model.parameters(), grad_outputs=loss_grad)
    torch.testing.assert_close(grads_ref, grads_compiled, rtol=1e-2, atol=1e-2)
