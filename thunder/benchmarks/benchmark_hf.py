from pathlib import Path
from datetime import datetime
import torch
import transformers

import thunder
from thunder.dev_utils.benchmark import benchmark_n
from thunder.recipes.base import BaseRecipe

from torch.profiler import profile, record_function, ProfilerActivity


class DebugRecipe(BaseRecipe):
    def setup_config(self):
        config = super().setup_config()
        return config


device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
root = Path(f"./results/{timestamp}")

save_traces = True
save_profiles = False
plugins = None  # or "reduce-overhead"
plugins_str = "" if plugins is None else plugins if isinstance(plugins, str) else "-".join(plugins)

# Define recipes
nvfuser_recipe = DebugRecipe()
torchcompile_recipe = DebugRecipe(fuser="torch.compile")
recipes = [nvfuser_recipe, torchcompile_recipe]


def run_and_profile(tag: str, fn, model, inp, compiled_models: dict[str, torch.nn.Module], cache=None):
    print(f"[{tag}] running PyTorch eager")
    eager_time = benchmark_n(10, fn, model, inp)

    timings = [f"Eager: {eager_time:.2f}ms"]

    for name, compiled_model in compiled_models.items():
        print(f"[{tag}] running Thunder ({name})")
        thunder_time = benchmark_n(10, fn, compiled_model, inp, cache=cache)
        timings.append(f"Thunder ({name}): {thunder_time:.2f}ms")

        if save_traces:
            if fn.__name__ == "generate":
                compiled_model(**inp)
            else:
                fn(compiled_model, inp, cache=cache)

            trace = thunder.last_traces(compiled_model)[-1]
            trace_path = root / name / f"{tag}_trace_{plugins_str}.py"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_path, "w") as f:
                f.write(str(trace))

        if save_profiles:
            profile_path = root / name / f"{tag}_profile_{plugins_str}.txt"
            profile_path.parent.mkdir(parents=True, exist_ok=True)

            with profile(activities=[ProfilerActivity.CUDA]) as thunder_prof:
                for _ in range(10):
                    with record_function(fn.__name__):
                        fn(compiled_model, inp)

            with open(profile_path, "w") as f:
                f.write(thunder_prof.key_averages().table(sort_by="cpu_time_total"))

    with open(root / f"{tag}_timings_{plugins_str}.txt", "w") as f:
        f.write("\n".join(timings))


def inference_gen():
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    base_model.eval().requires_grad_(False).to(device)
    inp = tokenizer(["Hello world! Here's a long story"], return_tensors="pt").to(device)

    def generate(model, inp, cache=None):
        model.generate(
            **inp,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            cache_implementation=cache,
            max_new_tokens=100,
            top_p=1.0,
            temperature=1,
        )

    compiled_models = {
        recipe.fuser[0] if isinstance(recipe.fuser, list) else recipe.fuser: thunder.compile(
            base_model, plugins=plugins, recipe=recipe
        )
        for recipe in recipes
    }

    run_and_profile("inference_gen", generate, base_model, inp, compiled_models, cache="static")


def inference_fwd():
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    base_model.eval().requires_grad_(False).to(device)
    inp = tokenizer(["Hello world! Here's a long story"], return_tensors="pt", max_length=2048, truncation=True).to(
        device
    )

    def fwd(model, inp, cache=None):
        model(**inp)

    compiled_models = {
        recipe.fuser[0] if isinstance(recipe.fuser, list) else recipe.fuser: thunder.compile(
            base_model, plugins=plugins, recipe=recipe
        )
        for recipe in recipes
    }

    run_and_profile("inference_fwd", fwd, base_model, inp, compiled_models, cache="static")


def training_fwd():
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    inp = tokenizer(["Hello world! Here's a long story"], return_tensors="pt", max_length=2048, truncation=True)
    inp["labels"] = inp["input_ids"]
    inp.to(device)

    def fwd(model, inp, cache=None):
        model(**inp)

    compiled_models = {
        recipe.fuser[0] if isinstance(recipe.fuser, list) else recipe.fuser: thunder.compile(
            base_model, plugins=plugins, recipe=recipe
        )
        for recipe in recipes
    }

    run_and_profile("training_fwd", fwd, base_model, inp, compiled_models, cache="static")


def training_fwd_bwd():
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    inp = tokenizer(["Hello world! Here's a long story"], return_tensors="pt", max_length=2048, truncation=True)
    inp["labels"] = inp["input_ids"]
    inp.to(device)

    def fwd_bwd(model, inp, cache=None):
        model(**inp).loss.backward()

    compiled_models = {
        recipe.fuser[0] if isinstance(recipe.fuser, list) else recipe.fuser: thunder.compile(
            base_model, plugins=plugins, recipe=recipe
        )
        for recipe in recipes
    }

    run_and_profile("fwd_bwd", fwd_bwd, base_model, inp, compiled_models, cache="static")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    root.mkdir(parents=True, exist_ok=True)

    inference_gen()
    inference_fwd()
    training_fwd()
    training_fwd_bwd()
