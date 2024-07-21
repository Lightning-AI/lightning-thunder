import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import thunder

# torchrun --standalone --nproc_per_node=4 a.py
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(100, 100)

    def forward(self, x):
        return self.linear(x)

dist.init_process_group('nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.getenv('LOCAL_RANK', 0))
set_seed(local_rank)

model = SimpleLinearModel().to(local_rank)

ddp_model = thunder.distributed.ddp(thunder.jit(model))
ddp_model = DDP(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

batch_size = 32
input_size = 100

for step in range(10):
    inputs = torch.broadcast_to(torch.tensor(step * 0.01 + (1 * local_rank), dtype=torch.float32), (32, 100)).to(local_rank)
    targets = inputs * 2

    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if rank == 0:
        print(f"Step [{step+1}/10], Loss: {loss.item()}")
   
