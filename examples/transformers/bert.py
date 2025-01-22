import torch
import transformers

import thunder
from thunder.recipes import HFTransformers

from utils import benchmark


def main():
    # model = "bert-base-uncased"
    model = "bert-large-uncased"

    with torch.device("cuda:0"):
        config = transformers.AutoConfig.from_pretrained(model)
        model = transformers.AutoModel.from_config(config)
        model.requires_grad_(False)
        model.eval()

        inp = torch.randint(1, 20, (1, 512))

    reduce_overhead = True
    fuser = "nvfuser"
    use_cache = True

    print(f"Eager: {benchmark(model, inp, use_cache=use_cache):.2f}ms")

    thunder_model = thunder.compile(model, recipe=HFTransformers(reduce_overhead=reduce_overhead, fuser=fuser))
    print(f"Thunder: {benchmark(thunder_model, inp, use_cache=use_cache):.2f}ms")

    torchcompile_model = torch.compile(model, mode="reduce-overhead" if reduce_overhead else "default")
    print(f"Torch Compile: {benchmark(torchcompile_model, inp, use_cache=use_cache):.2f}ms")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    main()
