from typing import Optional, Any

import torch
import torch.distributed as tdist

import thunder.core.utils as utils


__all__ = [
    "ddp",
]


# TODO Verify parameters are not partially initialized
# TODO Handle buffers
# TODO Improve initial broadcast logic
# Syncs a module's parameters across multiple processes
#   world, if specified, is a list of (rank, device) tuples
#   broadcast_from, if specified, is the rank to broadcast tensors from
#   At least one of world or broadcast_from must be specified so that we can
#       coordinate the broadcasting of parameters
def ddp(
    model: torch.nn.Module,
    rank: int,
    *,
    world: Any | None = None,
    broadcast_from: int | None = None,
    process_group: tdist.ProcessGroup | None = None,
    bucket_size_in_mb: float = 25.0,
) -> torch.nn.Module:
    """Thunder's Distributed Data Parallel.

    This function does two things. One is to broadcast the parameters hosted on the rank specified
    by ``broadcast_from`` to all the other ranks belonging to ``process_group``. The other is to
    updates the behavior of backward trace generation and optimization of it so that each gradient
    gets pre-averaged, i.e., divided by world size, and asynchronously allreduced.

    Args:
        model: A model before :func:`thunder.compile`d
        rank:

    Keyword Args:
        world:
        broadcast_from: The rank of the device hosting the parameters to broadcast. The lowest rank
            will be used if none specified.
        process_group: PyTorch's process group. Use the default process group if none specified.
        bucket_size_in_mb: Size of a gradient bucket.

    Return:
        :class:`torch.nn.Module` with the parameters synchronized among all the ranks involved.


    .. note::
        Currently this does not support gradient bucketing.

    .. code-block:: python
        :linenos:
        :caption: ddp_example.py

        # $ torchrun --nproc-per-node=<N_GPU> ddp_example.py
        import os
        import math

        import torch
        import torch.distributed as tdist
        import torch.nn as nn
        import torch.nn.functional as F

        import thunder
        import thunder.distributed as dist


        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        BATCH_SIZE = 8
        IN_FEATURES = 32
        OUT_FEATURES = 64
        N_CLASSES = 4


        def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.randn(BATCH_SIZE, IN_FEATURES, device=f"cuda:{LOCAL_RANK}", requires_grad=True)
            y = torch.randn(BATCH_SIZE, N_CLASSES, device=f"cuda:{LOCAL_RANK}").softmax(dim=1).requires_grad_()
            return x, y


        def new_gelu(a: torch.Tensor):
            return 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * torch.pow(a, 3.0))))


        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = nn .Linear(IN_FEATURES, OUT_FEATURES)
                self.l2 = nn.Linear(OUT_FEATURES, N_CLASSES)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = new_gelu(self.l1(x))
                return self.l2(h)


        def main():
            tdist.init_process_group(backend="nccl")

            model = MyModel().to(LOCAL_RANK)
            ddp_model = dist.ddp(model, LOCAL_RANK, broadcast_from=0)
            compiled = thunder.compile(ddp_model)
            optimizer = torch.optim.AdamW(compiled.parameters())
            losses = []
            loss_all_reduce_workers = []

            for _ in range(10):
                optimizer.zero_grad()
                x, y = get_batch()
                out = compiled(x)
                loss = F.cross_entropy(y, out)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    losses.append(loss.detach())
                    loss_all_reduce_workers.append(tdist.all_reduce(losses[-1], op=tdist.ReduceOp.AVG, async_op=True))

            if LOCAL_RANK == 0:
                for i, (loss, worker)  in enumerate(zip(losses, loss_all_reduce_workers)):
                    assert worker.wait()
                    print(f"# {i}-th loss: {loss.item()}")


        if __name__ == "__main__":
            main()

    """

    utils.check(
        tdist.is_available(),
        lambda: "ddp requires torch distributed to be available (but it's not)",
    )
    utils.check(
        world is not None or broadcast_from is not None,
        lambda: "At least one of world_size or broadcast_from must be specified",
    )

    pg = tdist.distributed_c10d._get_default_group() if process_group is None else process_group
    utils.check(pg is not None, lambda: "Both process group and default process group are None")
    model.use_ddp = True
    model.process_group_for_ddp = pg
    model.bucket_size_in_mb = bucket_size_in_mb

    # Infers device information from model
    # TODO Verify parameters are not partially initialized
    # TODO Handle buffers
    named_params = model.named_parameters()
    _, first_param = next(named_params)
    device = first_param.device
    devicetype = device.type
    deviceindex = device.index
    for name, param in named_params:
        utils.check(
            param.device.type == devicetype,
            lambda: (
                "Trying to ddp a model with parameters on devices with different device types, "
                f"including {devicetype} and {param.device.type}"
            ),
        )
        utils.check(
            deviceindex == param.device.index,
            lambda: (
                "Trying to ddp a model with parameters on multiple devices, including devices "
                f"{deviceindex} and {param.device.index}, but currently only models with all their "
                "parameters on one device are supported"
            ),
        )

    # Validates world information, if available
    lowest_device_index = deviceindex
    if world is not None:
        found_broadcast_process = False
        for rank_, dev in world:
            utils.check(dev.type == devicetype, lambda: "Found a world with multiple device types")
            if rank_ == rank:
                utils.check(
                    dev == device,
                    lambda: f"World entry ({rank_}, {dev}) disagrees with inferred device {device} of rank {rank}",
                )
            lowest_device_index = min(lowest_device_index, dev.index)
            if rank_ == broadcast_from:
                found_broadcast_process = True

        utils.check(
            not broadcast_from or found_broadcast_process,
            lambda: (
                f"Trying to broadcast from rank={broadcast_from}, " "but didn't find that rank in the world description"
            ),
        )

    # Identifies which process to broadcast from
    broadcast_from = broadcast_from if broadcast_from is not None else lowest_device_index

    # Starts broadcasts
    # TODO Make these broadcast asyncs
    # TODO Perform up to two broadcasts at a time
    # https://github.com/Lightning-AI/lightning-thunder/issues/727
    # TODO "Bucket" small tensors together before broadcasting
    with torch.no_grad():
        for name, param in model.named_parameters():
            tdist.broadcast(param, src=broadcast_from, group=pg, async_op=False)

    return model


def fsdp(
    model: torch.nn.Module,
    rank: int,
    *,
    world: Any | None = None,
    broadcast_from: int | None = None,
    process_group: tdist.ProcessGroup | None = None,
) -> torch.nn.Module:
    # We are going to use DDP to broadcast the parameters
    distributed_params_module = ddp(
        model, rank, world=world, broadcast_from=broadcast_from, process_group=process_group
    )
    distributed_params_module.use_ddp = False
    distributed_params_module.use_fsdp = True
    process_group = distributed_params_module.process_group_for_ddp

    current_rank = tdist.get_rank(group=process_group)
    world_size = tdist.get_world_size(group=process_group)

    # Now we need to shard the parameters
    # We will definitely change the sharding logic in the future
    for name, param in distributed_params_module.named_parameters():
        # Note [FSDP Sharding]
        # Here we shard the parameters on the first
        # dimension and all internal code will assume that the parameters are
        # sharded on the first dimension

        # narrow the param to the current rank on the first dimension
        utils.check(
            param.shape[0] % world_size == 0,
            lambda: (
                f"Current sharding requires the first dimension of the parameter to be divisible by the world size, "
                f"but got {param.shape[0]} and {world_size}"
            ),
        )
        chunk_size = param.shape[0] // world_size
        # NOTE This could be a ShardTensor to indicate other parts of the code
        # that it's sharded and should be treated differently
        param.data = param.data.narrow(0, chunk_size * current_rank, chunk_size).clone()
    return distributed_params_module
