import torch
import transformers

import thunder

from thunder.dev_utils.benchmark import benchmark_n


def main():
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # model_name = "meta-llama/Llama-3.1-8B"
    model_name = "meta-llama/Llama-3.2-1B"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "microsoft/phi-4"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with torch.device(device):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model.requires_grad_(False)
        model.eval()
        # apparently, Transformers 4.51.3 does not instantiate models on the default device
        model.to(device)

        inp = tokenizer(["Hello world! Here's a long story"], return_tensors="pt")

    def generate(model, inp, cache=None):
        out = model.generate(**inp, do_sample=False, cache_implementation=cache, max_new_tokens=100)
        print(tokenizer.decode(out[0].tolist()))

    print("\nGenerating with PyTorch eager:")
    eager_time = benchmark_n(2, generate, model, inp)

    thunder_model = thunder.compile(
        model,
        recipe="hf-transformers",
    )

    print("\nGenerating with Thunder:")
    thunder_time = benchmark_n(2, generate, thunder_model, inp, cache="static")

    print(f"\nEager: {eager_time:.2f}ms")
    print(f"Thunder: {thunder_time:.2f}ms")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    main()
