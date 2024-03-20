import os
import time

import torch
import functools
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as torch_dist

import thunder
from thunder.tests.lit_gpt_model import Config, GPT, Block

try:
    from lightning.fabric.utilities.throughput import measure_flops

    # from lightning.fabric.utilities import Throughput
    LIGHTNING_AVAILABLE = True
except:
    LIGHTNING_AVAILABLE = False

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
if world_size > 1:
    torch_dist.init_process_group(backend="nccl")
    pg = torch_dist.distributed_c10d._get_default_group()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    import inspect

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, fused=use_fused
    )
    return optimizer


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
        n_layers: int | None = None,
        profiler_start: int = 15,
        profiler_stop: int = 15,
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
        self.micro_batch_size = micro_batch_size
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
            # TODO: Remove when gradient accumulation is ready for benchmarking.
            if self.gradient_accumulation_steps > 1:
                print(
                    f"[WARNING] Gradient Accumulation is not fully supported yet. Benchmarking results may not be accurate. Gradient Accumulation Steps = {self.gradient_accumulation_steps}"
                )

        # Profiling Args
        self.nsys_enabled = nsys_enabled
        self.profiler_start = profiler_start
        self.profiler_stop = profiler_stop

        if n_layers is not None:
            self.config.n_layer = n_layers

        # Initialize the model
        self.model = self.init_model()

        # Setup the distributed algorithm choices
        if self.distributed_mode != "none":
            self.model = self.setup_distributed()

        # Initialize the optimizer after the model is sharded if using FSDP
        self.optimizer = configure_optimizers(
            self.model, weight_decay, learning_rate, (beta1, beta2), device_type="cuda"
        )

        # Compile the model
        if self.compile not in ["eager", None]:
            self.model = self.setup_compile()

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
        print(f"Loading model with {self.config.__dict__}")
        init_device = torch.device("meta") if self.distributed_mode == "fsdp" else self.device
        t0 = time.perf_counter()
        with init_device:
            model = GPT(self.config)
            model.to(dtype=torch.bfloat16)
        print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

        return model

    def setup_distributed(self):
        # Distributed Setup
        # TODO: Change compiler call names
        if "thunder" in self.compile:
            if self.distributed_mode == "ddp":
                from thunder.distributed import ddp

                model = ddp(
                    self.model,
                    broadcast_from=0,
                    bucket_size_in_mb=256.0,
                )
            elif self.distributed_mode == "fsdp":
                from thunder.distributed import fsdp, FSDPType, FSDPBucketingStrategy

                sharding_strategy = {"zero2": FSDPType.ZERO2, "zero3": FSDPType.ZERO3}[self.shard_mode]
                bucketing_strategy = {"none": FSDPBucketingStrategy.NONE, "block": FSDPBucketingStrategy.BLOCK}[
                    self.bucketing_mode
                ]
                model = fsdp(
                    self.model,
                    broadcast_from=None,
                    sharding_strategy=sharding_strategy,
                    bucketing_strategy=bucketing_strategy,
                )
        else:
            if self.distributed_mode == "ddp":
                model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[local_rank],
                    bucket_cap_mb=256.0,
                )
            elif self.distributed_mode == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

                litgpt_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
                zero_bucket_wrap_policy = lambda module, recurse, nonwrapped_numel: nonwrapped_numel >= 0
                sharding_strategy: ShardingStrategy = {
                    "zero2": ShardingStrategy.SHARD_GRAD_OP,
                    "zero3": ShardingStrategy.FULL_SHARD,
                }[self.shard_mode]
                # AssertionError: Dynamo only supports FSDP with use_orig_params=True
                torch.cuda.set_device(local_rank)
                model = FSDP(
                    self.model,
                    sharding_strategy=sharding_strategy,
                    auto_wrap_policy=litgpt_auto_wrap_policy,
                    device_id=local_rank,
                    use_orig_params=True,
                )
        return model

    def setup_compile(self):
        if self.compile == "inductor":
            # model = torch.compile(self.model, fullgraph=True, mode="reduce-overhead")
            print("Resetting cache size for torch.compile")
            import torch._dynamo.config as dynamo_config

            dynamo_config.cache_size_limit = 64
            model = torch.compile(self.model)
        elif "thunder" in self.compile:
            executors_list = [thunder.nvfuser_executor, thunder.pytorch_executor]
            if "inductor" in self.compile:
                from thunder.executors.torch_compile import torch_compile_executor as torch_compile_ex

                executors_list.insert(0, torch_compile_ex)
            if "cudnn" in self.compile:
                from thunder.executors.cudnnex import cudnn_ex

                executors_list.insert(0, cudnn_ex)
            else:
                from thunder.executors.sdpaex import sdpa_ex

                executors_list.insert(0, sdpa_ex)

            if "transformerengine" in self.compile:
                from thunder.executors.transformer_engineex import transformer_engine_ex

                executors_list.insert(0, transformer_engine_ex)

            model = thunder.jit(self.model, executors=executors_list)

        elif self.compile != "eager":
            raise ValueError(compile)

        return model

    def setup_dummy_dataloader(self):
        def pad_collate(batch):
            x, y = zip(*batch)
            x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
            y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)
            return x_padded, y_padded

        train_data = DummyDataset(self.model.max_seq_length, self.dynamic)
        train_dataloader = DataLoader(
            train_data, batch_size=self.micro_batch_size, num_workers=2, collate_fn=pad_collate
        )

        return train_dataloader

    def calculate_model_flops(self):
        input_ids, targets = next(self.train_data_iter)
        input_ids = input_ids.to(device=self.device)
        targets = targets.to(device=self.device)

        model_fwd = lambda: self.model(input_ids)
        model_loss = lambda y: torch.nn.functional.cross_entropy(
            y.reshape(-1, y.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        if LIGHTNING_AVAILABLE:
            self.perf_metrics["model_flops"] = measure_flops(self.model, model_fwd, model_loss) / 1e12

    def train(self):
        t0 = None
        # if global_rank in [0, None]:
        #     #Calculate the model FLOPs
        #     self.calculate_model_flops()
        # Setup Perf Collection
        # self.throughput = Throughput(window_size=10, world_size=world_size)

        if "transformerengine" in self.compile:
            import transformer_engine.pytorch as te

            te_ctx = te.fp8_autocast
        else:
            from contextlib import nullcontext

            te_ctx = nullcontext

        for i in range(self.max_iters):
            iter_t0 = time.perf_counter()
            if i == self.warmup_iter:  # warmup
                t0 = iter_t0

            for step_idx in range(self.gradient_accumulation_steps):
                input_ids, targets = next(self.train_data_iter)
                input_ids = input_ids.to(device=self.device)
                targets = targets.to(device=self.device)

                if self.nsys_enabled and i == self.profiler_start and global_rank in [0, None] and step_idx == 0:
                    print("=====Start NSYS Profiling======")
                    torch.cuda.cudart().cudaProfilerStart()

                with te_ctx():
                    logits = self.model(input_ids)

                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)
                loss = (
                    torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
                    / self.gradient_accumulation_steps
                )

                loss.backward()

                # Simple Gradient Accumulation Implementation
                if (step_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # torch.cuda.synchronize()
                if (
                    self.nsys_enabled
                    and i == self.profiler_stop
                    and global_rank in [0, None]
                    and ((step_idx + 1) % self.gradient_accumulation_steps == 0)
                ):
                    print("=====Stop NSYS Profiling======")
                    torch.cuda.cudart().cudaProfilerStop()

            loss_item = loss.item()  # synchronization
            t1 = time.perf_counter()
            if global_rank in [0, None]:
                print(
                    f"iter {i}: loss {loss_item:.4f}, iter time: {(t1 - iter_t0) * 1000:.2f}ms, t: {input_ids.size(1)}"
                )

            # if global_rank in [0, None] and i >=warmup_iter:
            #     self.throughput.update(
            #         time=(t1-t0),
            #         flops=self.model_flops,
            #         batches=i,
            #         samples=(i * self.micro_batch_size * self.gradient_accumulation_steps),
            #         lengths=(i * self.micro_batch_size * self.gradient_accumulation_steps * self.model.max_seq_length),
            #     )

            #     metrics = self.throughput.compute()
            #     if i % 10 == 0:
            #         print(metrics)

        if global_rank in [0, None]:
            # print(f"Total time: {(t1 - t0):.2f}s")
            # print(f"Average time per iter: {((t1 - t0)*1000)/(max_iters-warmup_iter):.2f}ms")
            self.perf_metrics["average_iter_time"] = ((t1 - t0) * 1000) / (self.max_iters - self.warmup_iter)

    def add_perf_metrics(self):
        # tokens_per_sec = total number of benchmarked iterations x global BS x block_size / total elapsed time (s)
        #               = global BS x block_size / (total elapsed time (s)/total number of benchmarked iterations)
        #               = global BS x block_size / average iter time (s)
        self.perf_metrics["tokens_per_sec"] = (
            self.global_batch_size * self.model.max_seq_length * 1000 / self.perf_metrics["average_iter_time"]
        )  # tokens/s
        if self.perf_metrics["model_flops"] is not None:
            self.perf_metrics["model_flop_per_sec"] = (
                self.perf_metrics["model_flops"] * 1000 / self.perf_metrics["average_iter_time"]
            )
            if world_size is not None:
                self.perf_metrics["model_flop_per_sec"] *= world_size
        self.perf_metrics["memory_used_GB"] = torch.cuda.max_memory_allocated() / 1e9

    def add_model_info_to_metrics(self):
        if global_rank in [0, None]:
            self.perf_metrics["model_name"] = self.model_name
            self.perf_metrics["Num GPUS"] = world_size
            self.perf_metrics["Seq Len"] = self.model.max_seq_length
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
                f"Model name: {benchmark.model_name}\nSeq Length: {benchmark.model.max_seq_length}\nMicro BS: {benchmark.micro_batch_size}\nGlobal BS: {benchmark.global_batch_size}"
            )
            print(
                f"Number of Layers: {benchmark.config.n_layer}\nNumber of parameters: {sum(p.numel() for p in benchmark.model.parameters() if p.requires_grad) / 1e9:.02f}B"
            )
            print(f"Distributed Mode: {benchmark.distributed_mode}")
            if benchmark.distributed_mode == "fsdp":
                print(
                    f"Sharding Mode: {benchmark.shard_mode}\nSharding Size: {benchmark.sharding_size}\nBucketing: {benchmark.bucketing_mode}"
                )
            print(f"Compiler: {benchmark.compile}")
            print(f"Average iter time: {benchmark.perf_metrics['average_iter_time']:.2f} ms")
            print(f"Memory used: {benchmark.perf_metrics['memory_used_GB']:.02f} GB")
            print(f"Throughput (Tokens/s): {benchmark.perf_metrics['tokens_per_sec']:.02f} tokens/s")
            print(
                f"Normalized Throughput (Tokens/s/GPU): {(benchmark.perf_metrics['tokens_per_sec']/world_size):.02f} tokens/s/gpu"
            )
            if benchmark.perf_metrics["model_flop_per_sec"] is not None:
                print(f"Model TFLOP/s: {benchmark.perf_metrics['model_flop_per_sec']:.02f} TFLOP/s")

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
