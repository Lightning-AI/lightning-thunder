import torch
import torchvision.models as models

import thunder


# TODO: needs a recipe


def main():
    with torch.device("cuda:0"):
        model = models.swin_v2_b()
        model.requires_grad_(False)
        model.eval()

        inp = torch.randn(128, 3, 224, 224)

    print(model)

    thunder_model = thunder.compile(model)

    out = thunder_model(inp)
    print(out)



if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    main()
