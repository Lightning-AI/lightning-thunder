import os
from functools import partial
import multiprocessing
import json
import os
import torch

torch.jit.script = lambda x: x

import traceback

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForImageClassification,
)

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
        return {"pixel_values": torch.randn(1, 3, 224, 224, dtype=torch.float32)}
    else:
        return {"input_ids": torch.randint(0, 1000, (1, 16))}


def try_model(output_dir, model_name):
    import thunder
    from thunder.extend import TemporaryExecutor
    from thunder.recipes import HFTransformers

    @thunder._with_cache_info_ctx
    def run_prologue(jfn, *args, **kwargs):
        cd = thunder.compile_data(jfn)
        cs = thunder.compile_stats(jfn)

        ci = thunder._get_cache_info()
        cd.populate_cache_info(ci, *args, **kwargs)
        traces = cd.acquire_initial_trace(cd.fn, args, kwargs, cd, cs, cd.executors_list[0])
        cache_entry = cd.apply_transforms_and_build_cache_entry(cd, cs, ci, *traces)
        with thunder.compile_data_and_stats(cd, cs):
            pro_to_comp, pro_to_epi = cache_entry.prologue_fn(*args, **kwargs)
        return cache_entry, pro_to_comp, pro_to_epi

    print(f"\n=== Testing {model_name} ===")

    output_file_name = os.path.join(output_dir, f"{model_name.replace('/', '__')}.txt")

    try:
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if getattr(config, "rope_scaling", None) is not None:
                if getattr(config.rope_scaling, 'rope_type', None) == "dynamic":
                    config.rope_scaling['rope_type'] = "default"
                elif getattr(config.rope_scaling, 'type', None) == "dynamic":
                    config.rope_scaling['type'] = "default"
            model_class = get_model_class(model_name, config)
            model = model_class.from_config(config, trust_remote_code=True, attn_implementation="sdpa").eval()
            input_sample = get_dummy_input(model_name, config)

    except Exception as setup_err:
        print(f"[SKIPPED] {model_name} - model setup failed")
        with open(output_file_name, "w") as f:
            print(f"[SKIPPED] {model_name} - model setup failed", file=f)
            traceback.print_exc(file=f)
        return

    from thunder import pytorch_executor
    lookaside_executor = TemporaryExecutor()
    lookasides = HFTransformers().setup_lookasides()
    for lookaside in lookasides:
        lookaside_executor._lookasides[lookaside._fn] = lookaside._replace_with
    try:
        jmodel = thunder.jit(model, executors=[lookaside_executor, pytorch_executor])
        ce, pro_to_comp, pro_to_epi = run_prologue(jmodel, **input_sample)
        print(f"[SUCCESS] {model_name} Trace acquired!")
        with open(output_file_name, "w") as f:
            print(f"[SUCCESS] {model_name} Trace acquired!", file=f)

    except Exception as thunder_err:
        print(f"[FAILURE] {model_name} - Thunder trace acquisition failed")
        with open(output_file_name, "w") as f:
            print(f"[FAILURE] {model_name} - Thunder trace acquisition failed", file=f)
            traceback.print_exc(file=f)


def aggregate_results(output_dir, results_file):
    results = []
    for filename in [el for el in os.listdir(output_dir) if os.path.splitext(el)[1] == ".txt"]:
        model = filename[:-4]
        with open(os.path.join(output_dir, filename)) as f:
            result = f.read()
        status = result.split()[0]
        last = ""
        if status != "[SUCCESS]":
            last = [el for el in result.split("\n") if "Error" in el]
            last = last[-1] if last else [el for el in result.split("\n") if el][-1]
        results.append({"model": model, "status": status, "last": last})

    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--models-file', type=str, required=False, default="")
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--results-file', type=str, required=False, default="")
    args = parser.parse_args()

    if args.models_file:
        os.makedirs(args.output_dir, exist_ok=True)

        with open(args.models_file) as f:
            model_list = [el.strip() for el in f.readlines() if el.strip() and not el.strip().startswith("#")]

        with multiprocessing.Pool(16) as pool:
            pool.map(partial(try_model, args.output_dir), model_list)

    if args.results_file:
        aggregate_results(args.output_dir, args.results_file)
