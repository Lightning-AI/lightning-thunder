import torch
import transformers

import thunder
from thunder.recipes import HFTransformers

from thunder.dev_utils.benchmark import benchmark


def main():
    # model_name = "bert-base-uncased"
    model_name = "bert-large-uncased"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    with torch.device("cuda:0"):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        model.requires_grad_(False)
        model.eval()

        inp = tokenizer(["Hello world!"], return_tensors='pt')

    reduce_overhead = True
    fuser = "nvfuser"

    print(f"Eager: {benchmark(model, **inp):.2f}ms")

    thunder_model = thunder.compile(model, recipe=HFTransformers(fuser=fuser, show_progress=True), plugins="reduce-overhead")
    print(f"Thunder: {benchmark(thunder_model, **inp):.2f}ms")

    torchcompile_model = torch.compile(model, mode="reduce-overhead" if reduce_overhead else "default")
    print(f"Torch Compile: {benchmark(torchcompile_model, **inp):.2f}ms")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    main()
