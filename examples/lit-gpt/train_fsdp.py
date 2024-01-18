import logging
import re
import time
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, Optional, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
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
from torch.utils.data import DataLoader, IterableDataset
from typing_extensions import override

import thunder
from thunder.tests.lit_gpt_model import GPT, Config

model_name = "open_llama_3b"
learning_rate = 6e-4
micro_batch_size = 2
max_iters = 50


class FSDPThunderStrategy(ParallelStrategy, _Sharded):
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
        from thunder.distributed import fsdp

        module = module.to(self.root_device)
        module = fsdp(module, rank=self.local_rank, broadcast_from=0)

        # NOTE @IvanYaschuck says that `fsdp(compile(model))` could be supported in the future so that the user owns the `compile` call.
        # we would still `compile(fsdp(undo_compile(compile(model))))` internally
        from thunder.executors.sdpaex import sdpa_ex

        return thunder.compile(module, executors_list=[sdpa_ex, thunder.nvfuser_executor, thunder.pytorch_executor])

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


def main(compile: str = "eager") -> None:
    strategy = (
        FSDPThunderStrategy()
        if compile == "thunder" else
        FSDPStrategy(auto_wrap_policy=always_wrap_policy, sharding_strategy="SHARD_GRAD_OP")
    )

    fabric = L.Fabric(devices="2", strategy=strategy, precision="bf16-true")
    fabric.launch()

    fabric.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP)

    config = Config.from_name(model_name)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module():
        og_model = model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    if compile == "inductor":
        # Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
        pattern = re.compile(".*Profiler function .* will be ignored")
        logging.getLogger("torch._dynamo.variables.torch").addFilter(
            lambda record: not pattern.search(record.getMessage())
        )

        model = torch.compile(model)
    elif compile == "thunder":
        pass  # fabric.setup does this
    elif compile != "eager":
        raise ValueError(compile)

    model = fabric.setup(model)
    if compile == "thunder":
        model.max_seq_length = og_model.max_seq_length
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-1, foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)

    train_data = DummyDataset(model.max_seq_length)
    train_dataloader = DataLoader(train_data, batch_size=micro_batch_size, num_workers=2)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    train(fabric, model, optimizer, train_dataloader)
    fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(
    fabric: L.Fabric, model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader
) -> None:
    train_iter = iter(train_dataloader)
    t0 = None
    assert max_iters > 5
    for i in range(max_iters):
        iter_t0 = time.perf_counter()
        if i == 5:  # warmup
            t0 = iter_t0
        input_ids, targets = next(train_iter)

        logits = model(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        loss_item = loss.item()  # synchronization
        t1 = time.perf_counter()
        fabric.print(
            f"iter {i}: loss {loss_item :.4f}, iter time: {(t1 - iter_t0) * 1000:.2f}ms, t: {input_ids.size(1)}"
        )
    fabric.print(f"Total time: {(t1 - t0):.2f}s")


class DummyDataset(IterableDataset):
    def __init__(self, max_seq_length: int):
        super().__init__()
        self.max_seq_length = max_seq_length

    def __iter__(self):
        t = self.max_seq_length
        while True:
            data = torch.randint(0, 100, (t + 1,), dtype=torch.int64)
            x = data[:t]
            y = data[1 : t + 1]
            yield x, y


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
