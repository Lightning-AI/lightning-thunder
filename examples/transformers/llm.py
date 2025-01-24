import torch
import transformers

import thunder
from thunder.recipes import HFTransformers

from thunder.dev_utils.benchmark import benchmark_n


def main():
    # model_name = "meta-llama/Llama-3.1-8B"
    model_name = "meta-llama/Llama-3.2-1B"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "microsoft/Phi-3.5-mini-instruct"
    # model_name = "microsoft/phi-4"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with torch.device("cuda:0"):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        inp = tokenizer(["Hello world! Here's a long story"], return_tensors='pt')

    reduce_overhead = True
    fuser = "nvfuser"

    def generate(model, inp, cache=None):
        out = model.generate(**inp, do_sample=False, cache_implementation=cache, max_new_tokens=100)
        print(tokenizer.decode(out[0].tolist()))

    print(f"Eager: {benchmark_n(2, generate, model, inp):.2f}ms")

    thunder_model = thunder.compile(model, recipe=HFTransformers(reduce_overhead=reduce_overhead, fuser=fuser, show_progress=True))

    generate(thunder_model, inp, cache="static")
    print({bsym.sym.name for bsym in thunder.last_traces(thunder_model)[-1].bound_symbols})

    print(f"Thunder: {benchmark_n(2, generate, thunder_model, inp, cache='static'):.2f}ms")

    model.forward = torch.compile(model.forward, mode="reduce-overhead" if reduce_overhead else "default")
    print(f"Torch Compile: {benchmark_n(2, generate, model, inp, cache='static'):.2f}ms")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    main()
