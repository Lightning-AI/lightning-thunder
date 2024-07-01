import torch
import thunder
import itertools

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
    # String returned from the callback will be used in the header for debug_symbol.
    input_strides = []
    for arg in itertools.chain(args, kwargs.values()):
        if isinstance(arg, torch.Tensor):
            input_strides.append(arg.stride())

    output_str = (
        f"Input Strides: {[stride for stride in input_strides]}\n"
        f"{bsym.sym.name} - Memory Allocated: {torch.cuda.memory_allocated()}"
    )
    return output_str


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
