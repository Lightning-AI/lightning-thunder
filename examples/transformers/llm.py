import torch
import transformers

import thunder
from thunder.recipes import HFTransformers

from utils import benchmark, benchmark_n


def main():
    # model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "meta-llama/Llama-3.2-1B"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "microsoft/Phi-3.5-mini-instruct"
    # model_name = "microsoft/phi-4"

    device = "cuda:0"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    with torch.device(device):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            # model_name, device_map={"": device}, torch_dtype=torch.bfloat16
            model_name, torch_dtype=torch.bfloat16
        )

    x = tokenizer(["Hello world!"], return_tensors='pt').to(device)

    reduce_overhead = True
    fuser = "nvfuser"

    def generate(inp, cache=None):
        model.generate(**inp, do_sample=False, past_key_values=cache, max_new_tokens=100)

    print(f"Eager: {benchmark_n(2, generate, x):.2f}ms")

    thunder_model = thunder.compile(model, recipe=HFTransformers(reduce_overhead=reduce_overhead, fuser=fuser))
    cache = thunder_model._get_cache("static", 1, 128, model.device, model.config.to_dict())

    def thunder_generate(inp, cache=None):
        thunder_model.generate(**inp, do_sample=False, past_key_values=cache, max_new_tokens=100)
 
    print(f"Thunder: {benchmark_n(2, thunder_generate, x, cache=cache):.2f}ms")

    # # torchcompile_model = torch.compile(model, mode="reduce-overhead" if reduce_overhead else "default")
    # model.forward = torch.compile(model.forward, mode="reduce-overhead")
    # print(f"Torch Compile: {benchmark_n(2, generate, x, cache='static'):.2f}ms")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    main()
