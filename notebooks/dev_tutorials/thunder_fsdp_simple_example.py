# imports
from thunder.tests.litgpt_model import GPT, Config
import torch
import torch.distributed
import thunder
import thunder.distributed
from thunder.distributed.transforms.fsdp_v2 import FSDPTransform
import os

# # # # # # # #
# Create Model
# # # # # # # #

# NOTE: We create the model on CPU.
device='cpu'
dim = 64
def create_model():
    layers = []
    layers.append(torch.nn.Linear(dim, dim))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(dim, dim))
    return torch.nn.Sequential(*layers).to(device)

# Model
model = create_model()

# Input
x = torch.randn(dim, dim, device=device)

# # # # # # # #
# Setup for distributed
# # # # # # # #
torch.distributed.init_process_group(backend='nccl')

rank = int(os.environ["LOCAL_RANK"])

device = f"cuda:{rank}"

# # # # # # # #
# Move model to correct device
# # # # # # # #
model.to(device)


# Move inputs to correct device
# # # # # # # #
x = x.to(device)

# # # # # # # #
# Wrap the model in thunder.distributed.fsdp
# # # # # # # #

# thunder.distributed.fsdp takes care of moving the parameter
# shard to the correct GPU for the current process.
cmodel = thunder.jit(model, transforms=[FSDPTransform()])

# Run the forward pass.
cmodel(x)

# # # # # # # #
# Check the traces
# # # # # # # #
fwd_traces = thunder.last_traces(cmodel)
bwd_traces = thunder.last_backward_traces(cmodel)

# # # # # # # #
# Print and check to see if they match ours
# # # # # # # #
if rank == 0:
    print(fwd_traces[-1])
    print("*******"* 8)
    print(bwd_traces[-1])
