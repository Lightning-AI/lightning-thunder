import torch
from thunder.core.interpreter import interpret

import math
import os
from typing import List

import numpy as np
from einops import rearrange

from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from sgm.modules.diffusionmodules.sampling import EulerEDMSampler

from sgm.util import instantiate_from_config


def load_model_from_config(config, ckpt):
    print("instantiate")
    model = instantiate_from_config(config.model)

    print(f"Loading model from {ckpt}")
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            pl_sd["global_step"]
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError

    m, u = model.load_state_dict(sd, strict=False)

    model.eval()
    model.model.half()
    model.cuda()
    return model


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    # Hardcoded demo settings; might undergo some changes in the future

    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = "A professional photograph of an astronaut riding a pig"
            if negative_prompt is None:
                negative_prompt = ""

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            orig_width = init_dict["orig_width"]
            orig_height = init_dict["orig_height"]

            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

    return value_dict


def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    # samples = embed_watemark(samples)
    for sample in samples:
        sample = 255.0 * sample.cpu().permute(1, 2, 0).numpy()
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1


def init_sampling():
    num_rows, num_cols = 1, 1

    steps = 40  # 40 ## 1...1000

    discretization_config = {
        "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
        "params": {
            "sigma_min": 0.03,  # 0.0292
            "sigma_max": 14.61,  # 14.6146,
            "rho": 3.0,
        },
    }

    guider_config = {
        "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
        "params": {
            "scale": 5.0,  # 0.0..100.0,
        },
    }

    sampler = EulerEDMSampler(
        num_steps=steps,
        discretization_config=discretization_config,
        guider_config=guider_config,
        s_churn=0.0,  # min_value=0.0
        s_tmin=0.0,  # min_value=0.0
        s_tmax=999.0,  # min_value=0.0
        s_noise=1.0,  # min_value=0.0
        verbose=True,
    )
    return sampler, num_rows, num_cols


def get_unconditional_conditioning(
    self, batch_c, batch_uc=None, force_uc_zero_embeddings=None
):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    ucg_rates = list()
    for embedder in self.embedders:
        ucg_rates.append(embedder.ucg_rate)
        embedder.ucg_rate = 0.0
    c = self(batch_c)
    uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

    for embedder, rate in zip(self.embedders, ucg_rates):
        embedder.ucg_rate = rate
    return c, uc


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: List = None,
    batch2model_input: List = None,
):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []

    with torch.no_grad():
        with torch.autocast("cuda"):
            with model.ema_scope():
                num_samples = [num_samples]
                # load_model(model.conditioner)

                batch = {
                    "original_size_as_tuple": torch.tensor(
                        [[1024, 1024]], device="cuda:0"
                    ),
                    "txt": ["Oil painting of a sunrise, detailed, 8k"],
                    "crop_coords_top_left": torch.tensor([[0, 0]], device="cuda:0"),
                    "target_size_as_tuple": torch.tensor(
                        [[1024, 1024]], device="cuda:0"
                    ),
                }

                batch_uc = {
                    "txt": [""],
                    "original_size_as_tuple": torch.tensor(
                        [[1024, 1024]], device="cuda:0"
                    ),
                    "crop_coords_top_left": torch.tensor([[0, 0]], device="cuda:0"),
                    "target_size_as_tuple": torch.tensor(
                        [[1024, 1024]], device="cuda:0"
                    ),
                }

                c, uc = get_unconditional_conditioning(
                    model.conditioner,
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to(
                                "cuda", torch.float16
                            ),
                            (c, uc),
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape, device="cuda", dtype=torch.float16)

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = sampler(denoiser, randn, cond=c, uc=uc)

                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                grid = torch.stack([samples])
                grid = rearrange(grid, "n b c h w -> (n h) (b w) c")

                return samples


def run_txt2img(
    prompt,
    model,
    version,
    version_dict,
    is_legacy=False,
):
    W, H = 1024, 1024
    C = version_dict["C"]
    F = version_dict["f"]

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        prompt=prompt,
    )
    sampler, num_rows, num_cols = init_sampling()
    num_samples = num_rows * num_cols

    # c_do_sample = interpret(do_sample)
    out = do_sample(
        model,
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
    )
    return out


if __name__ == "__main__":
    version = "SDXL-base-1.0"
    version_dict = {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    }

    add_pipeline = False

    torch.manual_seed(42)

    save_locally, save_path = (
        True,
        "output",
    )

    config = version_dict["config"]
    ckpt = version_dict["ckpt"]

    config = OmegaConf.load(config)

    model = load_model_from_config(config, ckpt)

    is_legacy = version_dict["is_legacy"]

    prompt = "Oil painting of a sunrise, detailed, 8k"

    finish_denoising = False

    samples = interpret(run_txt2img)(
        prompt,
        model,
        version,
        version_dict,
        is_legacy=is_legacy,
    )

    perform_save_locally(save_path, samples)
