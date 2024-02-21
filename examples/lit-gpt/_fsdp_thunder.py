"""Fabric Strategy to support Thunder FSDP: To be upstreamed into Fabric eventually."""
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, List, Literal, Optional, Union

import torch
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.strategy import TBroadcast, _Sharded
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

import thunder
from thunder.distributed import FSDPBucketingStrategy, FSDPType, fsdp

_FSDP_TYPE = Union[FSDPType, Literal["ZERO2", "ZERO3"]]
_BUCKETING_STRATEGY = Union[FSDPBucketingStrategy, Literal["NONE", "LAYER", "BLOCK"]]


class FSDPThunderStrategy(ParallelStrategy, _Sharded):
    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        sharding_strategy: "_FSDP_TYPE" = "FULL_SHARD",
        bucketing_strategy: "_BUCKETING_STRATEGY" = "NONE",
    ):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision=precision)
        self.parallel_devices = parallel_devices
        self.cluster_environment: Optional[ClusterEnvironment] = cluster_environment
        self.sharding_strategy = (
            FSDPType[sharding_strategy.upper()] if isinstance(sharding_strategy, str) else sharding_strategy
        )
        self.bucketing_strategy = (
            FSDPBucketingStrategy[bucketing_strategy.upper()]
            if isinstance(bucketing_strategy, str)
            else bucketing_strategy
        )

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        return {"num_replicas": self.num_nodes * self.num_processes, "rank": self.global_rank}

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()

    @override
    def setup_module(self, module: Module) -> Module:
        module = module.to(self.root_device)
        module = fsdp(
            module,
            broadcast_from=0,
            sharding_strategy=self.sharding_strategy,
            bucketing_strategy=self.bucketing_strategy,
        )

        # NOTE @IvanYaschuck says that `fsdp(compile(model))` could be supported in the future so that the user owns the `compile` call.
        # we would still `compile(fsdp(undo_compile(compile(model))))` internally
        from thunder.executors.sdpaex import sdpa_ex

        return thunder.compile(
            module,
            executors_list=[sdpa_ex, thunder.nvfuser_executor, thunder.pytorch_executor],
            # nv_enable_bookend because of https://github.com/Lightning-AI/lightning-thunder/issues/2025
            nv_enable_bookend=True,
        )

    @override
    def module_to_device(self, module: Module) -> None:
        pass

    @override
    def module_init_context(self, empty_init: Optional[bool] = None) -> ContextManager:
        if empty_init:
            raise NotImplementedError
        return self.precision.module_init_context()

    @override
    def module_sharded_context(self) -> ContextManager:
        return nullcontext()

    @override
    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[self.root_device.index])
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src)
        return obj[0]

    @override
    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        raise NotImplementedError

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        raise NotImplementedError

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _setup_distributed(self) -> None:
        reset_seed()
        self._set_world_ranks()
        process_group_backend = _get_default_process_group_backend_for_device(self.root_device)
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, process_group_backend)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank
