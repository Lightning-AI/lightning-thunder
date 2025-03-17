import torch
import torch.nn as nn

import thunder


def main():
    model = nn.Sequential(
        nn.Linear(2048, 4096),
        nn.ReLU(),
        nn.Linear(4096, 64)
    )

    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)

    print(thunder_model)

    print(thunder.last_traces(thunder_model)[-1])


if __name__ == "__main__":
    main()
