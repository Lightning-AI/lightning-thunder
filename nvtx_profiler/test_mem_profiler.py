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

# Transform for profiling with NVTX markers.
# This transform adds NVTX markers to all the computation symbols in the execution trace.
# It also marks the start and end of trace with cudaProfilerStart and cudaProfilerEnd to specify the capture range.
from memory_profile_transform import MemoryProfileTransform

mem_profile_transform = MemoryProfileTransform()

# import nvtx
# profile = nvtx.Profile()
# profile.enable()
# jmodel = profile_as_post_opt_tfms(thunder.jit(model))
# jmodel(x)
torch.cuda.cudart().cudaProfilerStart()
jmodel = thunder.jit(
    model,
    post_optimization_transforms=[
        mem_profile_transform,
    ],
)
o = jmodel(x)
o.sum().backward()
torch.cuda.cudart().cudaProfilerStop()
# profile.disable()
trace = thunder.last_traces(jmodel)[-1]
bwd_trace = thunder.last_backward_traces(jmodel)[-1]

print(bwd_trace)
