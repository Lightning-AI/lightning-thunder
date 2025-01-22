import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.ReLU(),
    nn.Linear(4096, 64)
)

import thunder

compiled_model = thunder.compile(model, recipe=None)
x = torch.randn(64, 2048)
y = compiled_model(x)

print(compiled_model)

# print(thunder.last_traces(compiled_model)[-1])