import torch
import os
import pytest


if os.getenv("ALLOW_COVERAGE_TRACE") != "1":
    pytest.skip("Skipping test_coverage_hf_diffusers.py in regular CI", allow_module_level=True)

hf_diffusers_unet2d_condition_model_ids = [
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "ionet-official/bc8-alpha",
    "stabilityai/sd-turbo",
    "runwayml/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
]

from thunder.tests.framework import requiresCUDA


@requiresCUDA
@pytest.mark.parametrize("model_id", hf_diffusers_unet2d_condition_model_ids)
def test_hf_diffusers(model_id):
    from thunder.dynamo import thunderfx
    from diffusers import UNet2DConditionModel

    unet_config = UNet2DConditionModel.load_config(model_id, subfolder="unet", torch_dtype=torch.bfloat16)
    unet = UNet2DConditionModel(unet_config)
    in_channels = unet.config.in_channels
    cross_attention_dim = unet.config.cross_attention_dim
    addition_embed_type = unet.config.addition_embed_type

    sample_size = 4
    batch_size = 1
    seq_length = 4

    if "xl" in model_id:
        time_ids_dim = 6
        text_embeds_dim = 4
        if "refiner" in model_id:
            time_ids_dim = 2
            text_embeds_dim = 4
    else:
        time_ids_dim = None
        text_embeds_dim = None

    input_shape = (batch_size, in_channels, sample_size, sample_size)
    hidden_states_shape = (batch_size, seq_length, cross_attention_dim)

    unet = unet.to("cuda", dtype=torch.bfloat16).requires_grad_(True)
    compiled_model = thunderfx(unet)

    def make_inputs(dtype=torch.bfloat16):
        added_cond_kwargs = {}
        with torch.device("cuda"):
            input = torch.randn(input_shape, dtype=dtype)
            hidden_states = torch.randn(hidden_states_shape, dtype=dtype)
            timestep = torch.ones(batch_size, dtype=torch.long)
            if addition_embed_type is not None:
                assert text_embeds_dim is not None and time_ids_dim is not None
                time_ids_shape = (batch_size, time_ids_dim)
                text_embeds_shape = (batch_size, text_embeds_dim)
                added_cond_kwargs["time_ids"] = torch.randn(time_ids_shape, device="cuda", dtype=dtype)
                added_cond_kwargs["text_embeds"] = torch.randn(text_embeds_shape, device="cuda", dtype=dtype)
        return (input, timestep, hidden_states), {"added_cond_kwargs": added_cond_kwargs}

    compiled_args, compiled_kwargs = make_inputs(torch.bfloat16)
    compiled_output = compiled_model(*compiled_args, **compiled_kwargs)

    ref_output = unet(*compiled_args, **compiled_kwargs)

    ref_output = ref_output.sample
    compiled_output = compiled_output.sample

    torch.testing.assert_close(compiled_output, ref_output, rtol=1e-2, atol=2e-1)

    # TODO: Currently fails, needs investigation https://github.com/Lightning-AI/lightning-thunder/issues/2153
    # loss_grad = torch.randn_like(compiled_output)
    # grads_ref = torch.autograd.grad(ref_output, unet.parameters(), grad_outputs=loss_grad)
    # grads_compiled = torch.autograd.grad(compiled_output, unet.parameters(), grad_outputs=loss_grad)
    # torch.testing.assert_close(grads_ref, grads_compiled, rtol=1e-1, atol=1e-1)
