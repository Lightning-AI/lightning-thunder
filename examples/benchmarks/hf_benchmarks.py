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
        config["skip_inplace_functionalization"] = False
        config["skip_alias_functionalization"] = True
        return config


device = "cuda"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

root = Path("./results")
save_traces = True
save_profiles = False
plugins = None  # or "reduce-overhead"
plugins_str = "" if plugins is None else plugins if isinstance(plugins, str) else "-".join(plugins)

recipe = DebugRecipe()

def run_and_profile(tag, fn, model, inp, trace_key, cache=None):
    print(f"  running PyTorch eager for {tag}")
    eager_time = benchmark_n(10, fn, model, inp)

    thunder_model = thunder.compile(model, plugins=plugins, recipe=recipe)
    print(f"  running Thunder for {tag}")
    thunder_time = benchmark_n(10, fn, thunder_model, inp, cache=cache)

    # Save timings
    with open(root / f"{tag}_timings_{plugins_str}.txt", "w") as f:
        f.write(f"Eager: {eager_time:.2f}ms\n")
        f.write(f"Thunder: {thunder_time:.2f}ms")

    # Save trace and run once 
    if save_traces:
        if fn.__name__ == "generate":
            # need to run the model once to get the trace
            thunder_model(**inp)
        else:

            _ = fn(thunder_model, inp, cache=cache)
        trace = thunder.last_traces(thunder_model)[-1]
        with open(root / f"{tag}_trace_{plugins_str}.py", "w") as f:
            f.write(str(trace))

    if save_profiles:
        with profile(activities=[ProfilerActivity.CUDA]) as thunder_prof:
            for _ in range(10):
                with record_function(fn.__name__):
                    _ = fn(thunder_model, inp)
            
            # thunder_prof.export_chrome_trace(str(root / f"{tag}_thunder_trace_{plugins_str}.json"))

        with open(root / f"training_fwd_profile_{plugins_str}.txt", "w") as f:
            table = thunder_prof.key_averages().table(sort_by="cpu_time_total")
            f.write(table)
            
def inference_gen():
    print("Inference gen")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to(device)
    inp = tokenizer(["Hello world! Here's a long story"], return_tensors='pt').to(device)

    def generate(model, inp, cache=None):
        model.generate(**inp, pad_token_id=tokenizer.eos_token_id, do_sample=False,
                       cache_implementation=cache, max_new_tokens=100, top_p=1.0)

    run_and_profile("inference_gen", generate, model, inp, "generate", cache='static')

def inference_fwd():
    print("Inference fwd")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to(device)
    inp = tokenizer(["Hello world! Here's a long story"], return_tensors='pt', max_length=2048, truncation=True).to(device)

    def fwd(model, inp, cache=None):
        model(**inp)

    run_and_profile("inference_fwd", fwd, model, inp, "fwd")

def training_fwd():
    print("Training fwd")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    inp = tokenizer(["Hello world! Here's a long story"], return_tensors='pt', max_length=2048, truncation=True)
    inp = {**inp, "labels": inp["input_ids"].clone()}
    inp = {k: v.to(device) for k, v in inp.items()}

    def fwd(model, inp, cache=None):
        model(**inp)

    run_and_profile("training_fwd", fwd, model, inp, "fwd")

def training_fwd_bwd():
    print("Training fwd_bwd")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    inp = tokenizer(["Hello world! Here's a long story"], return_tensors='pt', max_length=2048, truncation=True)
    inp = {**inp, "labels": inp["input_ids"].clone()}
    inp = {k: v.to(device) for k, v in inp.items()}

    def fwd_bwd(model, inp, cache=None):
        loss = model(**inp).loss
        loss.backward()

    run_and_profile("training_fwd_bwd", fwd_bwd, model, inp, "fwd_bwd")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    root.mkdir(parents=True, exist_ok=True)

    inference_gen()
    inference_fwd()
    training_fwd()
    training_fwd_bwd()
