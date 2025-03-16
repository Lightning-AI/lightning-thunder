import torch
import torchvision.models as models

import thunder

from thunder.dev_utils.benchmark import benchmark


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with torch.device(device):
        model = models.vit_b_16()
        model.requires_grad_(False)
        model.eval()

        inp = torch.randn(128, 3, 224, 224)

    out = model(inp)

    thunder_model = thunder.compile(model, plugins="reduce-overhead" if torch.cuda.is_available() else None)

    thunder_out = thunder_model(inp)

    # print(thunder.last_traces(thunder_model)[-1])

    torch.testing.assert_close(out, thunder_out)

    print(f"Eager: {benchmark(model, inp):.2f}ms")
    print(f"Thunder: {benchmark(thunder_model, inp):.2f}ms")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    main()
