import torch
import transformers

import thunder

from thunder.dev_utils.benchmark import benchmark


def main():
    # model_name = "bert-large-uncased"
    model_name = "bert-base-uncased"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with torch.device(device):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        model.requires_grad_(False)
        model.eval()

        inp = tokenizer(["Hello world!"], return_tensors='pt')

    print(f"Eager: {benchmark(model, **inp):.2f}ms")

    thunder_model = thunder.compile(model)

    print(f"Thunder: {benchmark(thunder_model, **inp):.2f}ms")

    if torch.cuda.is_available():
        thunder_model = thunder.compile(model, plugins="reduce-overhead")

        print(f"Thunder with 'reduce-overhead': {benchmark(thunder_model, **inp):.2f}ms")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True

    main()
