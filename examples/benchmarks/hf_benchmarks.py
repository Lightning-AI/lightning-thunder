from pathlib import Path

import torch
import transformers

import thunder
from thunder.dev_utils.benchmark import benchmark_n
from thunder.recipes.base import BaseRecipe

from torch.profiler import profile, record_function, ProfilerActivity


class DebugRecipe(BaseRecipe):
    def setup_config(self):
        config = super().setup_config()
        config["skip_inplace_functionalization"] = True
        config["skip_alias_functionalization"] = False
        return config


device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

root = Path("./results")

save_traces = False
save_profiles = False
plugins = None #  "reduce-overhead"

plugins_str = "" if plugins is None else plugins if type(plugins) == str else "-".join(plugins)

recipe = DebugRecipe()


def inference_gen():
    print("Inference gen")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with torch.device(device):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        model.requires_grad_(False)
        model.eval()

        inp = tokenizer(["Hello world! Here's a long story"], return_tensors='pt')

    def generate(model, inp, cache=None):
        _ = model.generate(**inp, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            cache_implementation=cache,
            max_new_tokens=100,
            top_p=1.0
        )

    print("  running PyTorch eager")
    eager_time = benchmark_n(10, generate, model, inp)

    thunder_model = thunder.compile(model, plugins=plugins, recipe=recipe)

    if save_profiles:
        with profile(activities=[ProfilerActivity.CUDA]) as thunder_prof:
            for _ in range(10):
                with record_function("generate"):
                    _ = generate(thunder_model, inp, cache='static')

        with open(root / f"inference_gen_profile_{plugins_str}.txt", "w") as f:
            table = thunder_prof.key_averages().table(sort_by="cpu_time_total")
            f.write(table)

    if save_traces:
        trace = thunder.last_traces(thunder_model)[-1]
        with open(root / f"inference_gen_trace_{plugins_str}.py", "w") as f:
            f.write(str(trace))

    print("  running Thunder")
    thunder_time = benchmark_n(10, generate, thunder_model, inp, cache='static')

    with open(root / f"inference_gen_timings_{plugins_str}.txt", "w") as f:
        f.write(f"Eager: {eager_time:.2f}ms\n")
        f.write(f"Thunder: {thunder_time:.2f}ms")


def inference_fwd():
    print("Inference fwd")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with torch.device(device):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        model.requires_grad_(False)
        model.eval()

        inp = tokenizer(
            ["Hello world! Here's a long story"],
            return_tensors='pt', max_length=2048, truncation=True
        )

    def fwd(model, inp):
        _ = model(**inp)

    print("  running PyTorch eager")
    eager_time = benchmark_n(10, fwd, model, inp)

    thunder_model = thunder.compile(model.model, plugins=plugins, recipe=recipe)

    if save_traces:
        trace = thunder.last_traces(thunder_model)[-1]
        with open(root / f"inference_fwd_trace_{plugins_str}.py", "w") as f:
            f.write(str(trace))

    if save_profiles:
        with profile(activities=[ProfilerActivity.CUDA]) as thunder_prof:
            for _ in range(10):
                with record_function("generate"):
                    _ = fwd(thunder_model, inp)

        with open(root / f"inference_fwd_profile_{plugins_str}.txt", "w") as f:
            table = thunder_prof.key_averages().table(sort_by="cpu_time_total")
            f.write(table)

    print("  running Thunder")
    thunder_time = benchmark_n(10, fwd, thunder_model, inp)

    with open(root / f"inference_fwd_timings_{plugins_str}.txt", "w") as f:
        f.write(f"Eager: {eager_time:.2f}ms\n")
        f.write(f"Thunder: {thunder_time:.2f}ms")


def training_fwd():
    print("Training fwd")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with torch.device(device):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        inp = tokenizer(
            ["Hello world! Here's a long story"],
            return_tensors='pt', max_length=2048, truncation=True
        )
        inp["labels"] = inp["input_ids"]

    def fwd(model, inp):
        _ = model(**inp)

    print("  running PyTorch eager")
    eager_time = benchmark_n(10, fwd, model, inp)

    thunder_model = thunder.compile(model.model, plugins=plugins, recipe=recipe)

    if save_traces:
        trace = thunder.last_traces(thunder_model)[-1]
        with open(root / f"training_fwd_trace_{plugins_str}.py", "w") as f:
            f.write(str(trace))

    if save_profiles:
        with profile(activities=[ProfilerActivity.CUDA]) as thunder_prof:
            for _ in range(10):
                with record_function("fwd"):
                    _ = fwd(thunder_model, inp)

        with open(root / f"training_fwd_profile_{plugins_str}.txt", "w") as f:
            table = thunder_prof.key_averages().table(sort_by="cpu_time_total")
            f.write(table)

    print("  running Thunder")
    thunder_time = benchmark_n(10, fwd, thunder_model, inp)

    with open(root / f"training_fwd_timings_{plugins_str}.txt", "w") as f:
        f.write(f"Eager: {eager_time:.2f}ms\n")
        f.write(f"Thunder: {thunder_time:.2f}ms")


def training_fwd_bwd():
    print("Training fwd_bwd")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with torch.device(device):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        inp = tokenizer(
            ["Hello world! Here's a long story"],
            return_tensors='pt', max_length=2048, truncation=True
        )
        inp["labels"] = inp["input_ids"]

    print(inp.keys())

    def fwd_bwd(model, inp):
        loss = model(**inp).loss
        loss.backward()

    print("  running PyTorch eager")
    eager_time = benchmark_n(10, fwd_bwd, model, inp)

    thunder_model = thunder.compile(model.model, plugins=plugins, recipe=recipe)

    if save_traces:
        trace = thunder.last_traces(thunder_model)[-1]
        with open(root / f"training_fwd_bwd_trace_{plugins_str}.py", "w") as f:
            f.write(str(trace))

    if save_profiles:
        with profile(activities=[ProfilerActivity.CUDA]) as thunder_prof:
            for _ in range(10):
                with record_function("fwd_bwd"):
                    _ = fwd_bwd(thunder_model, inp)

        with open(root / f"training_fwd_bwd_profile_{plugins_str}.txt", "w") as f:
            table = thunder_prof.key_averages().table(sort_by="cpu_time_total")
            f.write(table)

    print("  running Thunder")
    thunder_time = benchmark_n(10, fwd_bwd, thunder_model, inp)

    with open(root / f"training_fwd_bwd_timings_{plugins_str}.txt", "w") as f:
        f.write(f"Eager: {eager_time:.2f}ms\n")
        f.write(f"Thunder: {thunder_time:.2f}ms")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    root.mkdir(parents=True, exist_ok=True)

    inference_gen()
    inference_fwd()
    training_fwd()
    training_fwd_bwd()
