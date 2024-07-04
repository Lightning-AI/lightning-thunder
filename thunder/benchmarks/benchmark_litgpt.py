import os
import time
import warnings
from typing import Any
import contextlib
from contextlib import nullcontext, AbstractContextManager

import torch
import functools
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as torch_dist
from torch.distributed.device_mesh import init_device_mesh
import warnings

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointWrapper,
)

import thunder
from thunder.tests.litgpt_model import Config, GPT, Block
import transformer_engine.pytorch as te

from lightning.fabric.plugins.precision.transformer_engine import TransformerEnginePrecision
from lightning.fabric.utilities.throughput import measure_flops
from lightning.fabric.utilities import Throughput


world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
if world_size > 1:
    # Avoids the allocator thrashing issue in PyTorch NCCL backend.
    # See https://github.com/Lightning-AI/lightning-thunder/issues/420
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    torch_dist.init_process_group(backend="nccl")
    pg = torch_dist.distributed_c10d._get_default_group()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)


def is_transformer_engine(low_precision_mode: str) -> bool:
    return low_precision_mode == "fp8-delayed-te"


def get_context_manager(low_precision_mode: str, compile_mode: str) -> AbstractContextManager:
    if is_transformer_engine(low_precision_mode) and "thunder" not in compile_mode:
        @contextlib.contextmanager
        def fp8_autocast_context():
            with te.fp8_autocast(enabled=True):
                yield
        return fp8_autocast_context
    else:
        return nullcontext


def get_converted_model(low_precision_mode: str, compile_mode: str, model: Any) -> Any:
    if is_transformer_engine(low_precision_mode) and "thunder" not in compile_mode:
        te_precision = TransformerEnginePrecision(weights_dtype=torch.bfloat16, replace_layers=True)
        return te_precision.convert_module(model)
    else:
        return model


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    import inspect

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, fused=use_fused
    )
    return optimizer


# NOTE(crcrpar): Calling this method seems to bloat the memory consumption to some extent.
# e.g. ref: https://github.com/Lightning-AI/lightning-thunder/issues/439
def _run_fwd_bwd_one_microbatch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    gradient_accumulation_steps: int,
    device: torch.device,
    ctx: AbstractContextManager,
) -> torch.Tensor:
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    with ctx():
        logits = model(input_ids)
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1) / gradient_accumulation_steps
    loss.backward()
    return loss


class Benchmark_litGPT:
    def __init__(
        self,
        compile: str = "eager",
        dynamic: bool = False,
        nsys_enabled: bool = False,
        distributed_mode: str = "none",
        micro_batch_size: int = 1,
        global_batch_size: int | None = None,
        model_name: str = "Llama-2-7b-hf",
        shard_mode: str = "zero2",
        bucketing_mode: str = "none",
        sharding_size: int | None = None,
        ddp_bucket_size: float = 256.0,
        fsdp_bucket_params: float | None = None,
        checkpoint_activations: bool = False,
        n_layers: int | None = None,
        profiler_start: int = 15,
        profiler_stop: int = 15,
        skip_data_sync: bool = False,
        low_precision_mode: str = "none",
    ):
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        learning_rate = 5e-4  # max learning rate
        weight_decay = 1e-1
        beta1 = 0.9
        beta2 = 0.95

        self.max_iters = 45
        self.warmup_iter = 25
        assert self.max_iters > self.warmup_iter

        self.device = device
        self.model_name = model_name
        self.config = Config.from_name(self.model_name)
        self.compile = compile
        self.dynamic = dynamic
        self.distributed_mode = distributed_mode
        self.shard_mode = shard_mode
        self.bucketing_mode = bucketing_mode
        self.sharding_size = sharding_size
        self.ddp_bucket_size = ddp_bucket_size
        self.fsdp_bucket_params = fsdp_bucket_params
        self.checkpoint_activations = checkpoint_activations
        self.micro_batch_size = micro_batch_size
        self.low_precision_mode = low_precision_mode
        self.context_manager = get_context_manager(low_precision_mode, compile)

        # Clarify benchmark assumptions
        if self.sharding_size is not None:
            assert (
                "thunder" not in self.compile
            ), "Hybrid Sharding (FSDP/DP) using --sharding_size is not yet supported for Thunder. Coming soon."

            assert self.shard_mode in [
                "hybrid_zero2",
                "hybrid_zero3",
            ], "Sharding Size is only used with Hybrid FSDP/DP style parallelism."

            assert (
                world_size % self.sharding_size == 0
            ), f"World size {world_size} is not divisible by the sharding size {self.sharding_size}"

        if self.bucketing_mode != "none" and self.distributed_mode != "fsdp":
            print(
                f"[WARNING] --bucketing_mode {self.bucketing_mode} will be ignored as \
             it is only used for FSDP style parallelism but running {self.distributed_mode}"
            )

        assert not (
            "thunder" in self.compile and self.bucketing_mode == "size"
        ), "'size' bucketing mode is not supported for Thunder. Please use 'none' or 'block'."

        if self.fsdp_bucket_params is not None:
            if self.distributed_mode != "fsdp":
                print(
                    f"[WARNING] Found --fsdp_bucket_params but Distributed mode is {self.distributed_mode}. Will be ignored"
                )

            if self.bucketing_mode != "size":
                print(
                    f"[WARNING] Bucketing mode is set to {self.bucketing_mode}. --fsdp_bucket_params will be ignored."
                )

        if "thunder" in self.compile and is_transformer_engine(self.low_precision_mode):
            self.compile += "_transformerengine"

        if global_batch_size is not None:
            self.global_batch_size = global_batch_size
        else:
            self.global_batch_size = (
                self.micro_batch_size * world_size if world_size is not None else self.micro_batch_size
            )
        assert (
            self.global_batch_size % self.micro_batch_size == 0
        ), f"Global Batch Size {self.global_batch_size} should be a multiple of Micro Batch Size {self.micro_batch_size}."
        self.gradient_accumulation_steps = int(self.global_batch_size / self.micro_batch_size)
        if world_size:
            self.gradient_accumulation_steps = int(self.gradient_accumulation_steps / world_size)
            assert (
                self.global_batch_size % self.micro_batch_size * world_size == 0
            ), f"Global Batch Size {self.global_batch_size} should be a multiple Micro Batch Size {self.micro_batch_size} * World Size {world_size}."

        if self.checkpoint_activations:
            warnings.warn(
                "Activations checkpointing is configured, but Thunder does not support checkpointing. Checkpointing will be ignored."
            )
        self.skip_data_sync = skip_data_sync

        # Profiling Args
        self.nsys_enabled = nsys_enabled
        self.profiler_start = profiler_start
        self.profiler_stop = profiler_stop

        if n_layers is not None:
            self.config.n_layer = n_layers

        # Initialize the model
        t0 = time.perf_counter()
        print(f"Loading model with {self.config.__dict__}")
        self.model = self.init_model()
        print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

        self.model = get_converted_model(low_precision_mode, self.compile, self.model)

        # Setup the distributed algorithm choices
        self.model = self.setup_distributed(self.model)

        # Setup activations checkpointing
        if self.checkpoint_activations:
            self.setup_activation_checkpointing()

        # Initialize the optimizer after the model is sharded if using FSDP
        self.optimizer = configure_optimizers(
            self.model, weight_decay, learning_rate, (beta1, beta2), device_type="cuda"
        )

        # Compile the model
        self.model = self.setup_compile(self.model)

        # Setup the Dummy dataloader for training
        self.train_dataloader = self.setup_dummy_dataloader()
        self.train_data_iter = iter(self.train_dataloader)

        # Setup empty metrics dict
        if global_rank in [0, None]:
            self.perf_metrics = {
                "average_iter_time": None,
                "model_flops": None,
                "model_flop_per_sec": None,
                "tokens_per_sec": None,
            }

    def init_model(self):
        init_device = torch.device("meta") if self.distributed_mode == "fsdp" else self.device
        with init_device:
            model = GPT(self.config)
        model.to(dtype=torch.bfloat16)
        return model

    def setup_distributed(self, model):
        if self.distributed_mode == "none":
            return model

        # Distributed Setup
        # TODO: Change compiler call names
        if "thunder" in self.compile:
            if self.distributed_mode == "ddp":
                from thunder.distributed import ddp

                model = ddp(
                    model,
                    broadcast_from=0,
                    bucket_size_in_mb=self.ddp_bucket_size,
                )
            elif self.distributed_mode == "fsdp":
                from thunder.distributed import fsdp, FSDPType, FSDPBucketingStrategy

                sharding_strategy = {"zero2": FSDPType.ZERO2, "zero3": FSDPType.ZERO3}[self.shard_mode]
                bucketing_strategy = {
                    "none": FSDPBucketingStrategy.NONE,
                    "block": FSDPBucketingStrategy.BLOCK,
                    "layer": FSDPBucketingStrategy.LAYER,
                }[self.bucketing_mode]
                model = fsdp(
                    model,
                    broadcast_from=None,
                    sharding_strategy=sharding_strategy,
                    bucketing_strategy=bucketing_strategy,
                )
        else:
            if self.distributed_mode == "ddp":
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    bucket_cap_mb=self.ddp_bucket_size,
                )
            elif self.distributed_mode == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

                mesh = None
                if self.sharding_size is not None:
                    mesh = init_device_mesh("cuda", (int(world_size / self.sharding_size), self.sharding_size))

                litgpt_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
                size_auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=self.fsdp_bucket_params
                )
                zero_bucket_wrap_policy = lambda module, recurse, nonwrapped_numel: nonwrapped_numel >= 0

                custom_wrap_policy = {
                    "block": litgpt_auto_wrap_policy,
                    "size": size_auto_wrap_policy,
                    "none": zero_bucket_wrap_policy,
                }[self.bucketing_mode]

                sharding_strategy: ShardingStrategy = {
                    "zero2": ShardingStrategy.SHARD_GRAD_OP,
                    "zero3": ShardingStrategy.FULL_SHARD,
                    "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
                    "hybrid_zero3": ShardingStrategy.HYBRID_SHARD,
                }[self.shard_mode]

                # AssertionError: Dynamo only supports FSDP with use_orig_params=True
                torch.cuda.set_device(local_rank)
                model = FSDP(
                    model,
                    sharding_strategy=sharding_strategy,
                    auto_wrap_policy=custom_wrap_policy,
                    device_id=local_rank,
                    use_orig_params=True,
                    device_mesh=mesh,
                )
        return model

    def setup_activation_checkpointing(self):
        if any(isinstance(mod, CheckpointWrapper) for mod in self.model.modules()):
            warnings.warn(
                "FSDP checkpointing is configured, but the model already contains checkpointed layers."
                " Checkpointing will be ignored."
            )
            return

        check_fn = lambda submodule: isinstance(submodule, Block)
        apply_activation_checkpointing(self.model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

    def setup_compile(self, model):
        if self.compile == "inductor":
            print("Resetting cache size for torch.compile")
            import torch._dynamo.config as dynamo_config

            dynamo_config.cache_size_limit = 64
            model = torch.compile(model)
        elif "thunder" in self.compile:
            executors = [thunder.nvfuser_executor, thunder.pytorch_executor]
            if "inductor_cat" in self.compile:
                from thunder.executors.torch_compile import torch_compile_cat_ex as torch_compile_ex

                executors.insert(0, torch_compile_ex)
            elif "inductor" in self.compile:
                from thunder.executors.torch_compile import torch_compile_ex

                executors.insert(0, torch_compile_ex)
            if "cudnn" in self.compile:
                from thunder.executors.cudnnex import cudnn_ex

                executors.insert(0, cudnn_ex)
            else:
                from thunder.executors.sdpaex import sdpa_ex

                executors.insert(0, sdpa_ex)

            if "transformerengine" in self.compile:
                from thunder.executors.transformer_engineex import transformer_engine_ex

                executors.insert(0, transformer_engine_ex)

            model = thunder.jit(model, executors=executors)

        elif self.compile != "eager":
            raise ValueError(f"Invalid compile option: {self.compile}")

        return model

    def setup_dummy_dataloader(self):
        def pad_collate(batch):
            x, y = zip(*batch)
            x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
            y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)
            return x_padded, y_padded

        train_data = DummyDataset(self.config.block_size, self.dynamic)
        train_dataloader = DataLoader(
            train_data, batch_size=self.micro_batch_size, num_workers=2, collate_fn=pad_collate
        )

        return train_dataloader

    def calculate_model_flops(self):
        meta = torch.device("meta")
        device = self.device
        self.device = meta

        # calculate flops on a meta-device model because we only care about the shapes and
        # because the flops calculator installs hooks on the model
        meta_model = self.init_model()

        x = torch.randint(0, 1, (self.micro_batch_size, meta_model.config.block_size), device=meta)
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: torch.nn.functional.cross_entropy(
            y.reshape(-1, y.size(-1)), x.reshape(-1), ignore_index=-1
        )
        self.perf_metrics["model_flops"] = measure_flops(meta_model, model_fwd, model_loss)

        self.device = device

    def train(self):
        t0 = None
        if global_rank in [0, None]:
            # Calculate the model FLOPs
            self.calculate_model_flops()
            # Setup throughput Collection
            self.throughput = Throughput(window_size=self.max_iters - self.warmup_iter, world_size=world_size)

        if self.skip_data_sync:
            data_sync_ctx = self.model.no_sync
        else:
            data_sync_ctx = nullcontext

        for i in range(self.max_iters):
            iter_t0 = time.perf_counter()
            if i == self.warmup_iter:  # warmup
                t0 = iter_t0

            if self.nsys_enabled and i == self.profiler_start and global_rank in [0, None]:
                print("=====Start NSYS Profiling======")
                torch.cuda.cudart().cudaProfilerStart()

            with data_sync_ctx():
                for step_idx in range(self.gradient_accumulation_steps - 1):
                    input_ids, targets = next(self.train_data_iter)
                    input_ids = input_ids.to(self.device)
                    targets = targets.to(self.device)
                    with self.context_manager():
                        logits = self.model(input_ids)
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = targets.reshape(-1)
                    loss = (
                        torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
                        / self.gradient_accumulation_steps
                    )
                    loss.backward()

            input_ids, targets = next(self.train_data_iter)
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            with self.context_manager():
                logits = self.model(input_ids)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = (
                torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1) / self.gradient_accumulation_steps
            )
            loss.backward()

            # Simple Gradient Accumulation Implementation
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.nsys_enabled and i == self.profiler_stop and global_rank in [0, None]:
                print("=====Stop NSYS Profiling======")
                torch.cuda.cudart().cudaProfilerStop()

            loss_item = loss.item()  # synchronization
            t1 = time.perf_counter()
            if global_rank in [0, None]:
                print(
                    f"iter {i}: loss {loss_item:.4f}, iter time: {(t1 - iter_t0) * 1000:.2f}ms, t: {input_ids.size(1)}"
                )
                if i >= self.warmup_iter:
                    self.throughput.update(
                        time=(t1 - t0),
                        flops=self.perf_metrics["model_flops"],
                        batches=i,
                        samples=(i * self.micro_batch_size * self.gradient_accumulation_steps),
                        lengths=(i * self.micro_batch_size * self.gradient_accumulation_steps * self.config.block_size),
                    )

        if global_rank in [0, None]:
            # print(f"Total time: {(t1 - t0):.2f}s")
            self.perf_metrics["average_iter_time"] = ((t1 - t0) * 1000) / (self.max_iters - self.warmup_iter)

    def add_perf_metrics(self):
        metrics = self.throughput.compute()
        self.perf_metrics["tokens_per_sec"] = metrics.get("items_per_sec", metrics["device/items_per_sec"])
        self.perf_metrics["model_flop_per_sec"] = metrics.get("flops_per_sec", metrics["device/flops_per_sec"])
        self.perf_metrics["memory_used_GB"] = torch.cuda.max_memory_allocated() / 1e9

    def add_model_info_to_metrics(self):
        if global_rank in [0, None]:
            self.perf_metrics["model_name"] = self.model_name
            self.perf_metrics["Num GPUS"] = world_size
            self.perf_metrics["Seq Len"] = self.config.block_size
            self.perf_metrics["Micro BS"] = self.micro_batch_size
            self.perf_metrics["Global BS"] = self.global_batch_size
            self.perf_metrics["GA"] = self.gradient_accumulation_steps
            if self.distributed_mode in ["fsdp"]:
                self.perf_metrics["Distributed Mode"] = (
                    str(self.distributed_mode)
                    + "_"
                    + str(self.shard_mode)
                    + "_"
                    + str(self.bucketing_mode)
                    + "_bucketing"
                )
                self.perf_metrics["Sharding Size"] = self.sharding_size
            else:
                self.perf_metrics["Distributed Mode"] = self.distributed_mode
                self.perf_metrics["Sharding Size"] = None
            self.perf_metrics["compiler"] = self.compile


class DummyDataset(IterableDataset):
    def __init__(self, max_seq_length: int, dynamic: bool):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.dynamic = dynamic

    def __iter__(self):
        while True:
            if self.dynamic:
                t = torch.randint(10, self.max_seq_length + 1, (1,))
            else:
                t = self.max_seq_length
            data = torch.randint(0, 100, (t + 1,), dtype=torch.int64)
            x = data[:t]
            y = data[1 : t + 1]
            yield x, y


def benchmark_main(return_metrics_as_json=False, json_path="", **kwargs) -> None:
    """
    Runs a training benchmark for lit-GPT models and
    prints out the performance metrics.
    """

    benchmark = Benchmark_litGPT(**kwargs)
    try:
        benchmark.train()

        if global_rank in [0, None]:
            benchmark.add_perf_metrics()

            print(
                f"Model name: {benchmark.model_name}\nSeq Length: {benchmark.config.block_size}\nMicro BS: {benchmark.micro_batch_size}\nGlobal BS: {benchmark.global_batch_size}"
            )
            print(
                f"Number of Layers: {benchmark.config.n_layer}\nNumber of parameters: {sum(p.numel() for p in benchmark.model.parameters() if p.requires_grad) / 1e9:.02f}B"
            )
            print(f"Distributed Mode: {benchmark.distributed_mode}")
            if benchmark.distributed_mode == "fsdp":
                print(f"Sharding Mode: {benchmark.shard_mode}\nBucketing: {benchmark.bucketing_mode}")
                if benchmark.sharding_size is not None:
                    print(
                        f"Sharding Size: {benchmark.sharding_size}\nReplicate DP Groups: {int(world_size/benchmark.sharding_size)}"
                    )
                if benchmark.bucketing_mode == "size":
                    print(f"Bucketing Number Params: {benchmark.fsdp_bucket_params}")
            elif benchmark.distributed_mode == "ddp":
                print(f"DDP Bucketing Size: {benchmark.ddp_bucket_size} MB")
            print(f"Compiler: {benchmark.compile}")
            print(f"Low Precision Mode: {benchmark.low_precision_mode}")
            print(f"Average iter time: {benchmark.perf_metrics['average_iter_time']:.2f} ms")
            print(f"Memory used: {benchmark.perf_metrics['memory_used_GB']:.02f} GB")
            print(f"Tokens/s: {benchmark.perf_metrics['tokens_per_sec']:.02f}")
            print(f"Tokens/s/GPU: {(benchmark.perf_metrics['tokens_per_sec']/world_size):.02f}")
            print(f"TFLOP/s: {benchmark.perf_metrics['model_flop_per_sec'] / 1e12:.02f}")

    except Exception as error:
        # Helps catch OutOfMemory Errors and post processing of errors
        if global_rank in [0, None]:
            print("An error occurred:", type(error).__name__, "–", error)

    if global_rank in [0, None]:
        if return_metrics_as_json:
            benchmark.add_model_info_to_metrics()
            json_path = str(json_path)
            import json

            with open(json_path, "w") as file:
                json.dump(benchmark.perf_metrics, file, indent=4)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(benchmark_main)

    # ref: https://github.com/pytorch/pytorch/blob/3af12447/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1110-L1116
    if world_size > 1:
        torch_dist.destroy_process_group()
