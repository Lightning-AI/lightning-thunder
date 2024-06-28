import torch
import thunder

dim = 4096


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


model = Model().to(device="cuda")
x = torch.randn(4, dim, dim, device="cuda")

from debug_transform import DebugTransform


def callback(bsym, *args, **kwargs):
    return f"Before: {bsym.sym.name} - Memory Allocated: {torch.cuda.memory_allocated()}"


mem_profile_transform = DebugTransform(callback=callback)

jmodel = thunder.jit(
    model,
    post_optimization_transforms=[
        mem_profile_transform,
    ],
)
o = jmodel(x)
o.sum().backward()

trace = thunder.last_traces(jmodel)[-1]
bwd_trace = thunder.last_backward_traces(jmodel)[-1]

print(trace)
# print(bwd_trace)
