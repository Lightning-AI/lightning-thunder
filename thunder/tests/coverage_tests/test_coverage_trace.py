import torch
import traceback
import thunder
import os
import pytest


if os.getenv("ALLOW_COVERAGE_TRACE") != "1":
    pytest.skip("Skipping test_coverage_trace.py in regular CI", allow_module_level=True)


from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForImageClassification,
)
from thunder.tests.test_core import run_prologue


MODEL_LIST = [
    "gpt2",
    "bert-base-uncased",
    "google/reformer-enwik8",
    "facebook/bart-large",
    "t5-base",
    "xlnet-base-cased",
    "facebook/dinov2-small",
    "albert-base-v2",
    "google/electra-base-discriminator",
    "facebook/opt-1.3b",
    "google/vit-base-patch16-224",
]

MODEL_TASKS = {"openai/clip-vit-large-patch14": "clip", "facebook/dinov2-small": "vision"}


def get_model_class(model_name, config):
    task = MODEL_TASKS.get(model_name, "text")
    if task == "vision":
        return AutoModelForImageClassification
    elif config.architectures and "CausalLM" in config.architectures[0]:
        return AutoModelForCausalLM
    elif config.architectures and "Seq2SeqLM" in config.architectures[0]:
        return AutoModelForSeq2SeqLM
    else:
        return AutoModel


# custom input depending on model task
def get_dummy_input(model_name, config):
    task = MODEL_TASKS.get(model_name, "text")

    if task == "vision":
        return {"pixel_values": torch.randn(1, 3, 224, 224, device="cpu", dtype=torch.float32)}
    else:
        return {"input_ids": torch.randint(0, 1000, (1, 16), device="cpu")}


@pytest.mark.skip(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2436")
@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_model_trace(model_name):
    print(f"\n=== Testing {model_name} ===")

    config = AutoConfig.from_pretrained(model_name)
    model_class = get_model_class(model_name, config)
    model = model_class.from_config(config).to("meta")
    input_sample = get_dummy_input(model_name, config)

    jmodel = thunder.jit(model)
    ce, pro_to_comp, pro_to_epi = run_prologue(jmodel, **input_sample)

    print(f"[SUCCESS] {model_name} Trace acquired!")
