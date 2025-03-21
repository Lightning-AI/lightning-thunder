# torchrun --nnodes 1 --nproc-per-node 2 test_dtensor.py
import torch.nn as nn
import torch
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
import os
from thunder.dynamo import thunderfx
import torch.distributed as dist

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
num_devices = 2
mesh = DeviceMesh("cuda", list(range(num_devices)))

hidden_size = 16


def model(x, w):
    # return torch.nn.functional.linear(x, w)
    # return torch.add(x, w)
    return torch.mul(x, w)


weight = distribute_tensor(torch.randn(hidden_size, hidden_size, requires_grad=False), mesh, [Shard(0)])
bias = distribute_tensor(torch.randn(hidden_size, requires_grad=False), mesh, [Shard(0)])

in_dtensor = distribute_tensor(torch.randn(hidden_size, hidden_size, requires_grad=True), mesh, [Shard(0)])

expected = torch.compile(model)(in_dtensor, weight)
tmodel = thunderfx(model)
actual = tmodel(in_dtensor, weight)

# def model(x):
#     return x + 1

# in_tensor = torch.randn(num_devices, 4)
# mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
# in_dtensor = dist.tensor.distribute_tensor(in_tensor, mesh, [Shard(0)])

# print(in_dtensor.shape)

# expected = torch.compile(model)(in_dtensor)
# actual = thunderfx(model, nv_enable_matmul=True, nv_enable_linear=True)(in_dtensor)

torch.testing.assert_close(actual.to_local(), expected.to_local())

g_o = distribute_tensor(torch.ones(hidden_size, hidden_size), mesh, [Shard(0)])
expected_g = torch.autograd.grad(
    expected,
    (in_dtensor,),
    g_o,
)
actual_g = torch.autograd.grad(actual, (in_dtensor,), g_o)

torch.testing.assert_close(actual_g, expected_g)

if LOCAL_RANK == 0:
    import thunder

    thunder_fn = tmodel._backend.subgraph_infos[0].thunder_compiled_fns[0]
    traces = thunder.last_traces(thunder_fn)
    traces[0].save_trace("generated_dtensor_trcs/dtensor_init_trc.py")
    traces[-1].save_trace("generated_dtensor_trcs/dtensor_exec_trc.py")

    pro_traces = thunder.last_prologue_traces(thunder_fn)
    pro_traces[0].save_trace("generated_dtensor_trcs/dtensor_pro_trc.py")

    bwd_traces = thunder.last_backward_traces(thunder_fn)
    bwd_traces[0].save_trace("generated_dtensor_trcs/dtensor_bwd_init_trc.py")
    bwd_traces[-1].save_trace("generated_dtensor_trcs/dtensor_bwd_exec_trc.py")
