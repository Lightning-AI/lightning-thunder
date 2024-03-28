import os
from itertools import chain
from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import auto, Enum
from typing import TYPE_CHECKING, Any
from collections.abc import Generator
from functools import partial


import torch
import torch.distributed as tdist

import thunder.core.utils as utils
from thunder.core.proxies import DDPType

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

__all__ = [
    "ddp",
    "fsdp",
    "FSDPBucketingStrategy",
    "FSDPType",
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
def skip_data_parallel_grad_sync() -> Generator[Any, Any, Any]:
    """A context manager to skip data parallel grad sync."""
    token = set_skip_data_parallel_grad_sync(True)
    try:
        yield
    finally:
        reset_skip_data_parallel_grad_sync(token)


def _sync_grads(module: torch.nn.Module) -> None:
    import thunder

    if hasattr(module, "process_group_for_ddp"):
        # This branch is required when a function that takes the model as an input is jitted instead
        # of the model itself. In that case, the user won't have a reference to a `ThunderModule` so this needs to use
        # the reference set by ddp and fsdp on the module directly
        process_group = module.process_group_for_ddp
    elif (cd := thunder.compile_data(module)) is not None:
        # The ordinary jitted module branch
        process_group = cd.process_group_for_ddp
    else:
        raise RuntimeError(
            f"Expected `{type(module).__name__}` to have been jitted or to contain a `process_group_for_ddp` attribute"
        )

    params_with_grad = [p for p in module.parameters() if p.grad is not None]
    if not params_with_grad:
        return
    grads = [p.grad for p in params_with_grad]
    torch._foreach_div_(grads, process_group.size())
    with tdist.distributed_c10d._coalescing_manager(group=process_group, async_ops=True) as cm:
        for g in grads:
            tdist.distributed_c10d.all_reduce(g)
    cm.wait()


# TODO Verify parameters are not partially initialized
# TODO Handle buffers
# TODO Improve initial broadcast logic
# Syncs a module's parameters across multiple processes
#   broadcast_from is the rank to broadcast tensors from
def ddp(
    model: torch.nn.Module,
    *,
    broadcast_from: int | None = 0,
    bucket_size_in_mb: float = 25.0,
) -> torch.nn.Module:
    """Thunder's Distributed Data Parallel.

    This function does two things. One is to broadcast the parameters hosted on the rank specified
    by ``broadcast_from`` to all the other ranks belonging to default process_group. The other is to
    update the behavior of backward trace generation and optimization of it so that each gradient
    gets pre-averaged, i.e., divided by world size, and asynchronously all-reduced.

    Args:
        model: A model before ``thunder.jit`` applied

    Keyword Args:
        broadcast_from: The rank of the device hosting the parameters to broadcast. If None is passed,
            broadcasting will be skipped. Skipping can be useful for models whose weights have been loaded
            from a checkpoint. Defaults to 0.
        bucket_size_in_mb: Size of a gradient bucket.


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
            ddp_model = dist.ddp(model)
            compiled = thunder.jit(ddp_model)
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
                f"including {devicetype} and {param.device.type} for {name!r}"
            ),
        )
        utils.check(
            deviceindex == param.device.index,
            lambda: (
                "Trying to ddp a model with parameters on multiple devices, including devices "
                f"{deviceindex} and {param.device.index} for {name!r}, but currently only models with all their "
                "parameters on one device are supported"
            ),
        )

    # Note [DistributedDataParallel and ddp_type]
    # If model was wrapped with thunder.distributed.ddp it would have a
    # .use_ddp attribute set to True and all parameters would be already
    # broadcasted to all other processes. So that our tracing is aware of
    # this we need to mark the ddp_type of model's parameters as
    # thunder.proxies.DDPType.REPLICATED
    for p in model.parameters():
        p.ddp_type = DDPType.REPLICATED

    if broadcast_from is None:
        return model

    # Starts broadcasts
    # TODO Make these broadcast asyncs
    # TODO Perform up to two broadcasts at a time
    # See issue "Update ddp to use async broadcasts"
    # TODO "Bucket" small tensors together before broadcasting
    with torch.no_grad():
        for param in model.parameters():
            tdist.broadcast(param, src=broadcast_from, group=pg, async_op=False)

    return model


class FSDPType(Enum):
    """
    Specifies the sharding strategy to be used for FSDP in Thunder.

    Attributes:
        ZERO2: Similar to :attr:`torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP`.
        ZERO3: Similar to :attr:`torch.distributed.fsdp.ShardingStrategy.FULL_SHARD`.
    """

    ZERO2 = auto()
    ZERO3 = auto()


class FSDPBucketingStrategy(Enum):
    """
    Specify how we group parameters into a bucket for collective communication in fsdp.


    Attributes:
        NONE: Disables bucketing.
        LAYER: Creates buckets per layer such as :class:`torch.nn.Linear` and :class:`torch.nn.LayerNorm`.
        BLOCK: Creates buckets per block such as Transformer block.
    """

    NONE = auto()
    LAYER = auto()
    BLOCK = auto()


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
    device: torch.device | None = None,
    broadcast_from: int | None = None,
    sharding_strategy: FSDPType = FSDPType.ZERO2,
    bucketing_strategy: FSDPBucketingStrategy = FSDPBucketingStrategy.NONE,
) -> torch.nn.Module:
    """Convert ``model`` into Fully Sharded Data Parallel.

    This splits ``model``'s parameters in their first dimension into ``world_size`` chunks
    then has rank-``i`` host ``i``-th chunks of them.
    This means the implementation is different from :class:`torch.distributed.fsdp.FullyShardedDataParallel`
    which creates what's called :class:`torch.distributed.fsdp._flat_param.FlatParameter` as of
    https://github.com/pytorch/pytorch/tree/647f14e7. PyTorch however
    seems to be interested in per-parameter sharding as per https://github.com/pytorch/pytorch/issues/114299.

    To apply bucketing of collective communications, specify either
    :obj:`~thunder.distributed.FSDPBucketingStrategy.LAYER` or :obj:`BucketingStrategy.BLOCK` as
    ``bucketing_strategy``.
    The latter uses one collective communication, be it AllGather to unshard parameters or
    ReduceScatter to shard gradients, for one Transformer block. The former users one per layer such as
    :class:`torch.nn.Linear` and :class:`torch.nn.LayerNorm`.

     Args:
        model: The model to convert.

    Keyword Args:
        device: The corresponding model shard will be moved to this device. We recommend setting this to ``torch.cuda.current_device()``.
        broadcast_from: The rank of the device hosting the parameters to broadcast. If None is passed,
            broadcasting will be skipped (default). Enabling can be useful for models whose weights have been loaded
            from a checkpoint in a single rank.
        sharding_strategy:
        bucketing_strategy:

     Returns:
        :class:`torch.nn.Module`

    """
    utils.check(isinstance(sharding_strategy, FSDPType), lambda: f"FSDPType.ZERO2 and FSDPType.ZERO3 are supported.")
    utils.check(
        tdist.is_available(),
        lambda: "fsdp requires torch distributed to be available (but it's not)",
    )

    process_group = tdist.distributed_c10d._get_default_group()
    utils.check(process_group is not None, lambda: "The default process group is None")
    model.use_fsdp = True
    model.process_group_for_ddp = process_group
    model.sharding_strategy = sharding_strategy
    model.bucketing_strategy = bucketing_strategy

    # Shard the parameters
    _shard_params(model, process_group, device, broadcast_from)

    # See Note [DistributedDataParallel and ddp_type]
    # If model was wrapped with thunder.distributed.fsdp it would have a
    # .use_fsdp attribute set to True and all parameters would be already
    # sharded across all other processes. So that our tracing is aware of
    # this we need to mark the ddp_type of model's parameters as
    # thunder.proxies.DDPType.FULLY_SHARDED
    for p in model.parameters():
        p.ddp_type = DDPType.FULLY_SHARDED

    return model


@torch.no_grad()
def _shard_params(
    module: torch.nn.Module, process_group: "ProcessGroup", device: torch.device | None, broadcast_from: int | None
) -> None:
    """Shards the parameters on the first dimension."""
    global_rank = tdist.get_rank(group=process_group)
    world_size = tdist.get_world_size(group=process_group)
    if device is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)

    # We will definitely change the sharding logic in the future
    for module_name, submodule in module.named_modules():
        # Materialize meta-parameters on-device if necessary.
        # This is done before sharding in case the materialization logic depends on the tensor shape.
        # The tradeoff is that all of a module's direct parameters need to fit in device.
        # Each module only initializes its own parameters and not those of its children (recurse=False)
        if any(t.is_meta for t in chain(submodule.parameters(recurse=False), submodule.buffers(recurse=False))):
            # TODO: we could also support calling a "param_init_fn" argument like PyTorch
            _materialize(submodule, device)
        else:
            # Move leftover params and buffers to device. This is at least required to broadcast.
            # Cannot `submodule.to(device)` because we don't want it to recurse
            submodule._apply(partial(torch.Tensor.to, device=device), recurse=False)

        # Broadcast parameters if requested
        if broadcast_from is not None:
            for tensor in chain(submodule.parameters(recurse=False), submodule.buffers(recurse=False)):
                tdist.broadcast(tensor, src=broadcast_from, group=process_group, async_op=False)

        # Note [FSDP Sharding]
        # All internal code will assume that the parameters are sharded on the first dimension
        for param_name, param in submodule.named_parameters(recurse=False, prefix=module_name):
            _shard_param(param, global_rank, world_size, param_name)


def _shard_param(param: torch.Tensor, rank: int, world_size: int, name: str) -> None:
    utils.check(
        param.shape[0] % world_size == 0,
        lambda: (
            f"Current sharding requires the first dimension of the parameter {name!r} ({param.shape[0]})"
            f" to be divisible by the world size ({world_size})"
        ),
    )
    chunk_size = param.shape[0] // world_size
    # NOTE This could be a ShardTensor to indicate other parts of the code
    # that it's sharded and should be treated differently
    shard = param.data.narrow(0, chunk_size * rank, chunk_size).clone()
    param.data = shard


@torch.no_grad()
def _unshard_params(module: torch.nn.Module, process_group: "ProcessGroup", cpu_offload: bool = False) -> None:
    """Unshard a module's parameters.

    This supports CPU offloading of parameters.
    """
    from thunder.executors.torchex import _all_gather_prim_impl

    cpu = torch.device("cpu")
    for param in module.parameters():
        out = _all_gather_prim_impl(param.data, group=process_group, do_async=0)
        if cpu_offload:
            out = out.to(device=cpu)
        param.data = out
    # TODO(@carmocca): this should cpu-offload buffers or persistent buffers


def _materialize(module: torch.nn.Module, device: torch.device) -> None:
    """Materialize a module's direct children parameters by calling ``module.reset_parameters()``."""
    module.to_empty(device=device, recurse=False)
    if not hasattr(module, "reset_parameters"):
        raise TypeError(
            f"Materialization requires that the `{type(module).__name__}.reset_parameters` method is implemented."
            " This method is used to initialize any children parameters or buffers in this module."
        )
    module.reset_parameters()
