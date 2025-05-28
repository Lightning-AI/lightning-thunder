import torch
from transformers import ViTForImageClassification

import thunder

from thunder.dev_utils.benchmark import benchmark


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with torch.device(device):
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float32)
        model.requires_grad_(False)
        model.eval()
        # apparently, Transformers 4.51.3 does not instantiate models on the default device
        model.to(device)

        inp = torch.randn(128, 3, 224, 224)

    out = model(inp)

    thunder_model = thunder.compile(model, plugins="reduce-overhead" if torch.cuda.is_available() else None)

    thunder_out = thunder_model(inp)

    torch.testing.assert_close(out, thunder_out, atol=1e-2, rtol=1e-2)

    print(f"Eager: {benchmark(model, inp):.2f}ms")
    print(f"Thunder: {benchmark(thunder_model, inp):.2f}ms")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    main()
