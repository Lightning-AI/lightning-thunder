from __future__ import annotations
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
from torch.utils.weak import WeakTensorKeyDictionary

import thunder.core.utils as utils
from thunder.core.proxies import DistParallelType
from thunder.distributed.tensor_parallel import column_parallel
from thunder.distributed.tensor_parallel import row_parallel

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from thunder.core.module import ThunderModule


__all__ = [
    "ddp",
    "fsdp",
    "FSDPBucketingStrategy",
    "FSDPType",
    "column_parallel",
    "row_parallel",
]


_skip_data_parallel_grad_sync = ContextVar("skip_data_parallel_grad_sync", default=False)


def _avoid_torch_nccl_record_streams(func):
    """
    Avoids the allocator thrashing issue in PyTorch NCCL backend.
    """

    env_var = "TORCH_NCCL_AVOID_RECORD_STREAMS"
    value = os.environ.get(env_var, "0")

    def wrapper(*args, **kwargs):
        try:
            os.environ[env_var] = "1"
            return func(*args, **kwargs)
        finally:
            os.environ[env_var] = value

    return wrapper


@_avoid_torch_nccl_record_streams
def copy_default_process_group() -> ProcessGroup:
    """Create a new process group with the same ranks as the default process group.

    Returns:
        A new process group with the same ranks as the default process group.
    """
    default_pg = tdist.distributed_c10d._get_default_group()
    ranks = list(range(tdist.get_world_size(group=default_pg)))
    backend = tdist.distributed_c10d.get_backend(default_pg)
    # What's the better way to query this from the default process group? This
    # is the default value for `is_high_priority_stream` in PyTorch
    # default_pg.options returns ProcessGroup.Options object while
    # ProcessGroupNCCL.Options is required
    options = None
    if backend == "nccl":
        options = tdist.ProcessGroupNCCL.Options()
        options.is_high_priority_stream = False
    return tdist.new_group(ranks, backend=backend, pg_options=options)


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

    if getattr(module, "use_ddp", False):
        params_with_grad = [p for p in module.parameters() if p.grad is not None]
        if not params_with_grad:
            return
        grads = [p.grad for p in params_with_grad]
        torch._foreach_div_(grads, process_group.size())
        with tdist.distributed_c10d._coalescing_manager(group=process_group, async_ops=True) as cm:
            for g in grads:
                tdist.distributed_c10d.all_reduce(g, group=process_group)
        cm.wait()
    elif getattr(module, "use_fsdp", False):

        def prep_shard(
            g: torch.Tensor,
            rank: int,
            world_size: int,
        ) -> torch.Tensor:
            chunk_size = g.size(0) // world_size
            return g.narrow(0, rank * chunk_size, chunk_size)

        rank: int = tdist.distributed_c10d.get_rank(process_group)
        world_size: int = tdist.distributed_c10d.get_world_size(process_group)
        params_with_grad = tuple(filter(lambda p: hasattr(p, "_thunder_fsdp_unsharded_grad"), module.parameters()))
        if not params_with_grad:
            return
        unsharded_grads = [p._thunder_fsdp_unsharded_grad for p in params_with_grad]
        sharded_grads = [prep_shard(g, rank, world_size) for g in unsharded_grads]
        with tdist.distributed_c10d._coalescing_manager(group=process_group, async_ops=True) as cm:
            for u, s in zip(unsharded_grads, sharded_grads):
                tdist.distributed_c10d.reduce_scatter_tensor(
                    s, u, op=tdist.distributed_c10d.ReduceOp.AVG, group=process_group
                )
        cm.wait()
        for p, g in zip(params_with_grad, sharded_grads):
            p.grad = g
            del p._thunder_fsdp_unsharded_grad
    else:
        import warnings

        warnings.warn(
            "No op since neither `use_ddp` nor `use_fsdp` set. Have you applied either `thunder.distributed.ddp` or `thunder.distributed.fsdp`?"
        )


# When the user calls ddp(jitted_module), this function does the following
# - Marks the original function with appropiate attributes (use_ddp...)
# - Broadcasts parameters if necessary
# - It then registers a transform (callback that runs before prologue is executed) that transforms the
#   prologue and compute trace, that insert syncs (and grad syncs for the backward, handled by thunder automatically.)


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
    from thunder.core.module import ThunderModule

    if isinstance(model, ThunderModule):
        from thunder.distributed.transforms.ddp_v2 import DDPTransform
        from thunder.core.transforms import add_transform

        process_group = copy_default_process_group()
        utils.check(process_group is not None, lambda: "The default process group is None")
        # will insert syncs for parameters (and gradient syncs in the backward pass, this is handled by thunder)
        # usually, other transforms will remove the forward syncs inserted by this transform.
        transform_from_trace_to_ddp_trace = DDPTransform(
            process_group=process_group, bucket_size_in_mb=bucket_size_in_mb, broadcast_from=broadcast_from
        )
        model_new = add_transform(model, transform=transform_from_trace_to_ddp_trace)
        return model_new

    pg = copy_default_process_group()
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

    # Note [DistributedDataParallel and distparallel_type]
    # If model was wrapped with thunder.distributed.ddp it would have a
    # .use_ddp attribute set to True and all parameters would be already
    # broadcasted to all other processes. So that our tracing is aware of
    # this we need to mark the distparallel_type of model's parameters as
    # thunder.proxies.DistParallelType.REPLICATED
    for p in model.parameters():
        p.distparallel_type = DistParallelType.REPLICATED

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

    # TODO(crcrpar): Consider making `BLOCK`'s behavior more meticulous for models with simple structure
    # as in https://github.com/Lightning-AI/lightning-thunder/blob/b24e5b23/thunder/tests/distributed/test_ddp.py#L53-L60
    # For the linked model, `block` cannot put `t_net1_weight` and `t_net1_bias` into the same bucket.
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
    model: torch.nn.Module | ThunderModule,
    *,
    device: torch.device | None = None,
    broadcast_from: int | None = None,
    sharding_strategy: FSDPType = FSDPType.ZERO2,
    bucketing_strategy: FSDPBucketingStrategy = FSDPBucketingStrategy.NONE,
    move_state_dict_to_cpu: bool | None = None,
) -> torch.nn.Module | ThunderModule:
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

    See :doc:`/notebooks/dev_tutorials/fsdp_tutorial` to see how parameters are sharded across devices and how communications calls are inserted.

    Args:
        model: The model to convert.

    Keyword Args:
        device: The corresponding model shard will be moved to this device. We recommend setting this to ``torch.cuda.current_device()``.
        broadcast_from: The rank of the device hosting the parameters to broadcast. If None is passed,
            broadcasting will be skipped (default). Enabling can be useful for models whose weights have been loaded
            from a checkpoint in a single rank.
        sharding_strategy:
        bucketing_strategy:
        move_state_dict_to_cpu: Move all-gather'ed parameters of :func:`~thunder.core.module.ThunderModule.original_state_dict` to CPU
            as each all-gather is finished.

     Returns:
        :class:`torch.nn.Module`

    """
    from thunder.core.module import ThunderModule

    utils.check(isinstance(sharding_strategy, FSDPType), lambda: f"FSDPType.ZERO2 and FSDPType.ZERO3 are supported.")
    utils.check(
        tdist.is_available(),
        lambda: "fsdp requires torch distributed to be available (but it's not)",
    )

    if isinstance(model, ThunderModule):
        from thunder.core.transforms import add_transform
        from thunder.distributed.transforms.fsdp_v2 import FSDPTransform
        from thunder.transforms import MaterializationTransform

        if device is None:
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device("cuda", local_rank)
        return add_transform(
            model,
            transform=[
                FSDPTransform(
                    device=device,
                    broadcast_from=broadcast_from,
                    sharding_strategy=sharding_strategy,
                    bucketing_strategy=bucketing_strategy,
                    release_original_parameters=True,
                    move_state_dict_to_cpu=False if move_state_dict_to_cpu is None else move_state_dict_to_cpu,
                ),
                MaterializationTransform(device, init=MaterializationTransform.init_from_original_module_init()),
            ],
        )

    if move_state_dict_to_cpu is not None:
        import warnings

        warnings.warn(
            "`move_state_dict_to_cpu` is only effective when `model` is `ThunderModule`, i.e., `thunder.jit(model)`"
        )
    process_group = copy_default_process_group()
    utils.check(process_group is not None, lambda: "The default process group is None")
    model.use_fsdp = True
    model.process_group_for_ddp = process_group
    model.sharding_strategy = sharding_strategy
    model.bucketing_strategy = bucketing_strategy

    # Shard the parameters
    _shard_params(model, process_group, device, broadcast_from, allow_padding_for_fsdp=True)

    # See Note [DistributedDataParallel and distparallel_type]
    # If model was wrapped with thunder.distributed.fsdp it would have a
    # .use_fsdp attribute set to True and all parameters would be already
    # sharded across all other processes. So that our tracing is aware of
    # this we need to mark the distparallel_type of model's parameters as
    # thunder.proxies.DistParallelType.FULLY_SHARDED
    for p in model.parameters():
        p.distparallel_type = DistParallelType.FULLY_SHARDED

    return model


@torch.no_grad()
def _shard_params(
    module: torch.nn.Module,
    process_group: ProcessGroup,
    device: torch.device | None,
    broadcast_from: int | None,
    allow_padding_for_fsdp: bool = False,
) -> None:
    """Shards the parameters on the first dimension."""
    global_rank = tdist.get_rank(group=process_group)
    world_size = tdist.get_world_size(group=process_group)
    if device is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)

    # In case there is a weight sharing, we don't want to shard the same param
    # multiple times. We use `sharded_params` to keep track of already sharded param.
    sharded_params = WeakTensorKeyDictionary()
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
            if param in sharded_params:
                continue
            _shard_param(param, global_rank, world_size, param_name, allow_padding_for_fsdp=allow_padding_for_fsdp)
            # Mark the param as sharded so that we don't reshard it (in case model has shared parameters)
            sharded_params[param] = True


@torch.no_grad()
def _shard_tensor(
    param: torch.Tensor,
    rank: int,
    world_size: int,
    name: str,
    *,
    allow_padding_for_fsdp: bool = False,
    dim: int | None = None,
) -> tuple[torch.Tensor, int | None]:

    dim_to_shard = 0 if dim is None else dim
    if allow_padding_for_fsdp:
        utils.check(dim_to_shard == 0, lambda: f"Invalid {dim=} with {allow_padding_for_fsdp=}, Only 0 is supported")
        padded_param_shape = list(param.shape)
        orig_0dim_size = param.size(dim_to_shard)
        chunk_size = (padded_param_shape[0] + world_size - 1) // world_size
        padded_param_shape[0] = chunk_size * world_size
        _thunder_fsdp_padding_size = padded_param_shape[0] - param.size(0)
        if _thunder_fsdp_padding_size > 0:
            padded_param = torch.empty(padded_param_shape, device=param.device, dtype=param.dtype)
            padded_param[:orig_0dim_size].copy_(param)
            shard = padded_param.data.narrow(0, chunk_size * rank, chunk_size).clone()
            return shard, _thunder_fsdp_padding_size
        else:
            shard = param.data.narrow(0, chunk_size * rank, chunk_size).clone()
            return shard, None
    else:
        utils.check(
            param.shape[dim_to_shard] % world_size == 0,
            lambda: (
                f"Current sharding requires the sharded dimension of the parameter {name!r} ({param.shape[dim_to_shard]})"
                f" to be divisible by the world size ({world_size})"
            ),
        )
        chunk_size = param.shape[dim_to_shard] // world_size
        # NOTE This could be a ShardTensor to indicate other parts of the code
        # that it's sharded and should be treated differently
        shard = param.data.narrow(dim_to_shard, chunk_size * rank, chunk_size).clone()
        return shard, None


def _shard_param(
    param: torch.Tensor,
    rank: int,
    world_size: int,
    name: str,
    *,
    allow_padding_for_fsdp: bool = False,
    dim: int | None = None,
) -> None:
    shard, padding_size = _shard_tensor(
        param,
        rank,
        world_size,
        name,
        allow_padding_for_fsdp=allow_padding_for_fsdp,
        dim=dim,
    )
    param.data = shard
    if allow_padding_for_fsdp:
        param._thunder_fsdp_padding_size = padding_size


@torch.no_grad()
def _unshard_params(module: torch.nn.Module, process_group: ProcessGroup, cpu_offload: bool = False) -> None:
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
