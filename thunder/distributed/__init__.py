from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import auto, Enum
from typing import Any

import torch
import torch.distributed as tdist

import thunder.core.utils as utils


__all__ = [
    "ddp",
    "fsdp",
    "FSDPBucketingStrategy",
]


_skip_data_parallel_grad_sync = ContextVar("skip_data_parallel_grad_sync", default=False)


def set_skip_data_parallel_grad_sync(value: bool) -> Token:
    """Set whether to skip data parallel grad sync.

    Args:
        value: Whether to skip data parallel grad sync.

    Returns:
        A token that can be used to restore the previous value.
    """
    return _skip_data_parallel_grad_sync.set(value)


def reset_skip_data_parallel_grad_sync(token: Token) -> None:
    """Reset whether to skip data parallel grad sync.

    Args:
        token: The token returned by :func:`set_skip_data_parallel_grad_sync`.
    """
    _skip_data_parallel_grad_sync.reset(token)


def get_skip_data_parallel_grad_sync() -> bool:
    """Get whether to skip data parallel grad sync.

    Returns:
        Whether to skip data parallel grad sync.
    """
    return _skip_data_parallel_grad_sync.get()


@contextmanager
def skip_data_parallel_grad_sync() -> None:
    """A context manager to skip data parallel grad sync."""
    token = set_skip_data_parallel_grad_sync(True)
    try:
        yield
    finally:
        reset_skip_data_parallel_grad_sync(token)


# TODO Verify parameters are not partially initialized
# TODO Handle buffers
# TODO Improve initial broadcast logic
# Syncs a module's parameters across multiple processes
#   broadcast_from, if specified, is the rank to broadcast tensors from
def ddp(
    model: torch.nn.Module,
    *,
    broadcast_from: int | None = None,
    bucket_size_in_mb: float = 25.0,
) -> torch.nn.Module:
    """Thunder's Distributed Data Parallel.

    This function does two things. One is to broadcast the parameters hosted on the rank specified
    by ``broadcast_from`` to all the other ranks belonging to default process_group. The other is to
    updates the behavior of backward trace generation and optimization of it so that each gradient
    gets pre-averaged, i.e., divided by world size, and asynchronously allreduced.

    Args:
        model: A model before :func:`thunder.compile`d

    Keyword Args:
        broadcast_from: The rank of the device hosting the parameters to broadcast. The lowest rank
            will be used if none specified.
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

    pg = tdist.distributed_c10d._get_default_group()
    utils.check(pg is not None, lambda: "The default process group is None")
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

    # Identifies which process to broadcast from
    broadcast_from = broadcast_from if broadcast_from is not None else 0

    # Starts broadcasts
    # TODO Make these broadcast asyncs
    # TODO Perform up to two broadcasts at a time
    # https://github.com/Lightning-AI/lightning-thunder/issues/727
    # TODO "Bucket" small tensors together before broadcasting
    with torch.no_grad():
        for name, param in model.named_parameters():
            tdist.broadcast(param, src=broadcast_from, group=pg, async_op=False)

    return model


class FSDPType(Enum):
    """
    This specifies the sharding strategy to be used for FSDP in Thunder.

    Attributes:
        ZERO2: Similar to torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP
        ZERO3: Similar to torch.distributed.fsdp.ShardingStrategy.FULL_SHARD
    """

    ZERO2 = auto()
    ZERO3 = auto()


class FSDPBucketingStrategy(Enum):
    """Enum class to specify how we group parameters into a bucket for collective communication in fsdp."""

    NONE = auto()
    """Disables Bucketing."""
    LAYER = auto()
    """Creates buckets per layer such as :class:`torch.nn.Linear` and :class:`torch.nn.LayerNorm`."""
    BLOCK = auto()
    """Creates buckets per block such as Transformer block."""


def get_extract_bucket_name_from_tensor_proxy(granularity: FSDPBucketingStrategy):
    from thunder.core.proxies import TensorProxy

    # TODO(crcrpar): Consider having bucket_name include dtype (and device) in it
    # as it's possible (especially `FSDPBucketingStrategy.BLOCK`) that parameters of a block
    # have different dtypes such as BF16 and FP8.
    def f(tensor: TensorProxy) -> str:
        bucket_name: str = "fsdp_fwd_"
        match granularity:
            case FSDPBucketingStrategy.LAYER:
                bucket_name += "_".join(tensor.name.split("_")[:-1])
            case FSDPBucketingStrategy.BLOCK:
                t_name_split = tensor.name.split("_")
                i = (
                    ([x.isdigit() for x in t_name_split].index(True) + 1)
                    if any(x.isdigit() for x in t_name_split)
                    else len(t_name_split)
                )
                bucket_name += "_".join(t_name_split[:i])
            case _:
                utils.check(False, lambda: f"Invalid {granularity=} is passed.")
        return bucket_name

    return f


def fsdp(
    model: torch.nn.Module,
    *,
    broadcast_from: int | None = None,
    sharding_strategy: FSDPType = FSDPType.ZERO2,
    bucketing_strategy: FSDPBucketingStrategy = FSDPBucketingStrategy.NONE,
) -> torch.nn.Module:
    """Convert ``model`` into Fully Sharded Data Parallel.

    This splits ``model``'s parameters in their first dimension into ``world_size`` chunks
    then has rank-``i`` host ``i``-th chunks of them.
    This means the implementation is different from :class:`torch.distributed.fsdp.FullyShardedDataParallel`
    which creates what's called :class:`torch.distributed.fsdp._flat_param.FlatParameter` as of
    https://github.com/pytorch/pytorch/blob/647f14e70baffa383515c28c2ac219b7084c41c2. PyTorch however
    seems to be interested in per-parameter sharding as per https://github.com/pytorch/pytorch/issues/114299.

    To apply bucketing of collective communications, specify either
    :obj:`~thunder.distributed.FSDPBucketingStrategy.LAYER` or :obj:`BucketingStrategy.BLOCK` as
    ``bucketing_strategy``.
    The latter uses one collective communication, be it AllGather to unshard parameters or
    ReduceScatter to shard gradients, for one Transformer block. The former users one per layer such as
    :class:`torch.nn.Linear` and :class:`torch.nn.LayerNorm`.

     Args:
        model:

    Keyword Args:
        broadcast_from:
        sharding_strategy:
        bucketing_strategy:

     Returns:
        :class:`torch.nn.Module`
    """
    utils.check(isinstance(sharding_strategy, FSDPType), lambda: f"FSDPType.ZERO2 and FSDPType.ZERO3 are supported.")

    # We are going to use DDP to broadcast the parameters
    distributed_params_module = ddp(model, broadcast_from=broadcast_from)
    distributed_params_module.use_ddp = False
    distributed_params_module.use_fsdp = True
    distributed_params_module.sharding_strategy = sharding_strategy
    distributed_params_module.bucketing_strategy = bucketing_strategy
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
