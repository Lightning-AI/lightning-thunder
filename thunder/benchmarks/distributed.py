import argparse
from dataclasses import dataclass
import json
import os

import torch

from thunder.benchmarks import (
    run_multiprocess_benchmark,
    BenchmarkRunStatistics,
    _nanogpt_configs,
    NanoGPTConfig,
    NanoGPTBenchmark,
    LitGPTBenchmark,
    LitGPTConfig,
)
from thunder.tests.litgpt_model import name_to_config
from thunder.distributed import FSDPBucketingStrategy
from thunder.distributed import FSDPType


_nanogpt_model_names: tuple[str, ...] = tuple(_nanogpt_configs.keys())
_llama_model_names: tuple[str, ...] = tuple(name_to_config.keys())
_llama2_configs: tuple[str, ...] = ("open_llama_7b", "Llama-2-7b-hf")
fsdp_bucketing_strategies = {str(e).split(".")[1].lower(): e for e in FSDPBucketingStrategy}
fsdp_sharding_strategies = {str(e).split(".")[1].lower(): e for e in FSDPType}


@dataclass
class ResultFormatter:
    model_name: str
    base_name: str
    suffix: str
    dtype: str
    world_size: int
    total_callable_construction_time: int
    warmup_stats: list[BenchmarkRunStatistics]
    benchmark_stats: list[BenchmarkRunStatistics]
    typehint: bool
    rank_mem_info: dict[int, tuple[int, int]]

    def __post_init__(self) -> None:
        avg_tcct = self.total_callable_construction_time // self.world_size

        total_warmup_time_ns = sum(stat.total_time for stat in self.warmup_stats)
        avg_total_warmup_time = total_warmup_time_ns / len(self.warmup_stats)

        median_bench: BenchmarkRunStatistics
        median_bench_time: int
        sorted_bench_stats = sorted(self.benchmark_stats, key=lambda stat: stat.total_time)
        if len(sorted_bench_stats) % 2 == 1:
            median_bench = sorted_bench_stats[len(sorted_bench_stats) // 2]
            median_bench_time = median_bench.total_time
        else:
            right_middle = len(sorted_bench_stats) // 2
            left_middle = right_middle - 1
            left_stat = sorted_bench_stats[left_middle]
            right_stat = sorted_bench_stats[right_middle]
            median_bench_time = (left_stat.total_time + right_stat.total_time) // 2
            median_bench = left_stat
        self.median_bench = median_bench
        self.median_bench_time = median_bench_time

        total_backward_calls: int = sum(stat.called_backward for stat in self.benchmark_stats) // self.world_size
        total_bench_time = sum(stat.total_time for stat in self.benchmark_stats)
        self.avg_bench_time = total_bench_time / len(self.benchmark_stats)
        initialization_estimate_ns: float = (avg_total_warmup_time - self.avg_bench_time) * len(self.warmup_stats)

        self.total_initialization_time_ns: float = avg_tcct + initialization_estimate_ns
        self.callable_construction_percentage: float = avg_tcct / self.total_initialization_time_ns * 100
        self.initialization_percentage: float = initialization_estimate_ns / self.total_initialization_time_ns * 100

        total_host_time_ns: int = median_bench.host_stop_time - median_bench.start_time
        self.host_time_percentage: float = total_host_time_ns / median_bench.total_time * 100

        self.avg_tcct = avg_tcct
        self.initialization_estimate_ns = initialization_estimate_ns

    def _convert_rank_mem_info(self) -> dict[str, float]:
        a = {}
        r = {}
        for rank in sorted(self.rank_mem_info):
            allocated, reserved = self.rank_mem_info[rank]
            if self.typehint:
                a[f"d_peak_mem_allocated_rank{rank}"] = allocated / 1024 / 1024
                r[f"d_peak_mem_reserved_rank{rank}"] = reserved / 1024 / 1024
            else:
                a[f"peak_mem_allocated_rank{rank}"] = allocated / 1024 / 1024
                r[f"peak_mem_reserved_rank{rank}"] = reserved / 1024 / 1024
        a.update(r)
        return a

    def to_json(self) -> dict[str, float | str | int]:
        d = {}
        if self.typehint:
            d = {
                "s_torch_version": str(torch.__version__),
                "d_num_runs": len(self.benchmark_stats) // self.world_size,
                "s_model_name": self.model_name,
                "s_dtype": self.dtype,
                "s_base_name": self.base_name,
                "s_full_name": f"{self.base_name}-{self.suffix}",
                "s_suffix": self.suffix,
                "l_world_size": self.world_size,
                "d_perf_median": self.median_bench_time / 1000.0,
                "d_perf_average": self.avg_bench_time / 1000.0,
                "d_callable_construction_time": self.avg_tcct / 1000.0,
                "d_initialization_time": self.initialization_estimate_ns / 1000.0,
                "d_warmup_average": self.initialization_estimate_ns / 1000.0,
                "d_host_time_percentage": self.host_time_percentage,
            }
        else:
            d = {
                "torch_version": str(torch.__version__),
                "num_runs": len(self.benchmark_stats) // self.world_size,
                "model_name": self.model_name,
                "dtype": self.dtype,
                "base_name": self.base_name,
                "full_name": f"{self.base_name}-{self.suffix}",
                "suffix": self.suffix,
                "world_size": self.world_size,
                "perf_median": self.median_bench_time / 1000.0,
                "perf_average": self.avg_bench_time / 1000.0,
                "callable_construction_time": self.avg_tcct / 1000.0,
                "initialization_time": self.initialization_estimate_ns / 1000.0,
                "warmup_average": self.initialization_estimate_ns / 1000.0,
                "host_time_percentage": self.host_time_percentage,
            }
        d.update(self._convert_rank_mem_info())
        return d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed Benchmark for nanogpt or llama2 models with Thunder's ddp or fsdp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-xl",
        choices=_nanogpt_model_names + _llama_model_names,
        help=f"parameter to specify model config. Available options are {_nanogpt_model_names + _llama_model_names}.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help=f"sequence length for nanogpt benchmark. 128 or 512 are commonly used. This option is only used when one of {_nanogpt_model_names}",
    )
    parser.add_argument("--skip-torch", action="store_true", help="set this to skip PyTorch things")
    parser.add_argument("--skip-torch-compile", action="store_true", help="set this to skip torch compile")
    parser.add_argument(
        "--dataparallel-strategy",
        "-D",
        type=str,
        default="ddp",
        choices=("ddp", "fsdp"),
        help="Specify data parallel implementation",
    )
    parser.add_argument(
        "--bucket-sizes",
        nargs="+",
        type=float,
        default=[0, 25],
        help="parameters of `bucket_size_in_mb` for thunder.distributed.ddp. This is a hyperparameter of distributed data parallel.",
    )
    parser.add_argument(
        "--bucketing-strategies",
        nargs="+",
        type=str,
        default=("none", "block"),
        choices=tuple(fsdp_bucketing_strategies.keys()),
        help="Bucketing strategies to run. This is a hyperparameter of (fully) sharded data parallel.",
    )
    parser.add_argument(
        "--sharding-strategies",
        nargs="+",
        type=str,
        default=("zero2",),
        choices=tuple(fsdp_sharding_strategies.keys()),
        help="Sharding strategies to run",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="world size. This script will compare the number against available device count",
    )
    parser.add_argument(
        "--batchdims",
        type=int,
        default=None,
        help="Batch size. For nanogpt, default is 16, for llama models, 1",
    )
    parser.add_argument(
        "--typehint",
        action="store_true",
        help="add prefix to suggest type of values in JSON. s for string, d for float, l for int",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"), help="dtype")
    parser.add_argument("--out", type=str, default=".", help="Directory to dump benchmark result in json")
    args = parser.parse_args()
    if args.skip_torch and args.skip_torch_compile:
        raise ValueError("Specifying both `--skip-torch` and `--skip-torch-compile` doesn't make sense")
    if args.model in _llama2_configs and args.dataparallel_strategy == "ddp":
        raise ValueError(
            "Llama2 models are memory hungry so that DDP would not capable of running them even on 80GB devices"
        )
    return args


# TODO Port these benchmarks to pytest (and targets.py)
# See issue "Create distributed pytest benchmarks"
if __name__ == "__main__":
    args = parse_args()

    torch_dtype: torch.dtype = getattr(torch, args.dtype)
    config, b = None, None
    if args.model in _nanogpt_configs:
        config = {"seq_len": args.seq_len}
        config.update(**_nanogpt_configs[args.model])
        config = NanoGPTConfig(**config)
        kwargs = {}
        if args.batchdims is not None:
            kwargs["batchdims"] = (args.batchdims,)
        b = NanoGPTBenchmark(config, dtype=torch_dtype, **kwargs)
    else:
        kwargs = {}
        if args.batchdims is not None:
            kwargs["batchdims"] = (args.batchdims,)
        else:
            kwargs["batchdims"] = (1,)
        config = LitGPTConfig.from_name(args.model)
        b = LitGPTBenchmark(config, dtype=torch_dtype, **kwargs)

    results: list[dict[str, int | float | str]] = []

    if torch.distributed.is_available() and torch.cuda.device_count() > 1:
        # you can set the num of devices via `CUDA_VISIBLE_DEVICES` env var.
        world_size = args.world_size
        if world_size > torch.cuda.device_count():
            msg = f"{args.world_size=} must be <= {torch.cuda.device_count()=}"
            raise ValueError(msg)

        if not args.skip_torch:
            if args.dataparallel_strategy == "ddp":
                from thunder.benchmarks import default_torch_ddp_executor

                print("torch - DistributedDataParallel")
                total_cct, all_warmup_stats, all_benchmark_stats, rank_mem_info = run_multiprocess_benchmark(
                    b,
                    default_torch_ddp_executor,
                    world_size=world_size,
                    extended_printout=False,
                )
                results.append(
                    ResultFormatter(
                        model_name=args.model,
                        base_name="torch_ddp",
                        suffix="bucket_size_25MB",
                        dtype=args.dtype,
                        world_size=world_size,
                        total_callable_construction_time=total_cct,
                        warmup_stats=all_warmup_stats,
                        benchmark_stats=all_benchmark_stats,
                        typehint=args.typehint,
                        rank_mem_info=rank_mem_info,
                    ).to_json()
                )

                if not args.skip_torch_compile:
                    from thunder.benchmarks import default_torch_compile_ddp_executor

                    print("torch.compile - DistributedDataParallel")
                    total_cct, all_warmup_stats, all_benchmark_stats, rank_mem_info = run_multiprocess_benchmark(
                        b,
                        default_torch_compile_ddp_executor,
                        world_size=world_size,
                        extended_printout=False,
                    )
                    results.append(
                        ResultFormatter(
                            model_name=args.model,
                            base_name="torch_compile_ddp",
                            suffix="bucket_size_25MB",
                            dtype=args.dtype,
                            world_size=world_size,
                            total_callable_construction_time=total_cct,
                            warmup_stats=all_warmup_stats,
                            benchmark_stats=all_benchmark_stats,
                            typehint=args.typehint,
                            rank_mem_info=rank_mem_info,
                        ).to_json()
                    )
            else:
                import functools
                from torch.distributed.fsdp import ShardingStrategy
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                from thunder.benchmarks import get_default_torch_fsdp_executor
                from thunder.tests.nanogpt_model import Block as NanoGPTBlock
                from thunder.tests.litgpt_model import Block as GPTBlock

                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
                auto_wrap_policies = (
                    None,
                    functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={NanoGPTBlock, GPTBlock}),
                )
                for auto_wrap_policy in auto_wrap_policies:
                    print(f"torch - FullyShardedDataParallel - {sharding_strategy=} - {auto_wrap_policy=}")
                    total_cct, all_warmup_stats, all_benchmark_stats, rank_mem_info = run_multiprocess_benchmark(
                        b,
                        get_default_torch_fsdp_executor(
                            sharding_strategy=sharding_strategy,
                            auto_wrap_policy=auto_wrap_policy,
                            apply_torch_compile=False,
                        ),
                        world_size=world_size,
                        extended_printout=False,
                    )
                    results.append(
                        ResultFormatter(
                            model_name=args.model,
                            base_name="torch_fsdp",
                            suffix=(
                                str(sharding_strategy).lower() + "-bucketing_" + "block"
                                if auto_wrap_policy is not None
                                else "none"
                            ),
                            dtype=args.dtype,
                            world_size=world_size,
                            total_callable_construction_time=total_cct,
                            warmup_stats=all_warmup_stats,
                            benchmark_stats=all_benchmark_stats,
                            typehint=args.typehint,
                            rank_mem_info=rank_mem_info,
                        ).to_json()
                    )

                if not args.skip_torch_compile:
                    for auto_wrap_policy in auto_wrap_policies:
                        print(f"torch.compile - FullyShardedDataParallel - {sharding_strategy=} - {auto_wrap_policy=}")
                        total_cct, all_warmup_stats, all_benchmark_stats, rank_mem_info = run_multiprocess_benchmark(
                            b,
                            get_default_torch_fsdp_executor(
                                sharding_strategy=sharding_strategy,
                                auto_wrap_policy=auto_wrap_policy,
                                apply_torch_compile=True,
                            ),
                            world_size=world_size,
                            extended_printout=False,
                        )
                        results.append(
                            ResultFormatter(
                                model_name=args.model,
                                base_name="torch_compile_fsdp",
                                suffix=(
                                    str(sharding_strategy).lower() + "-bucketing_" + "block"
                                    if auto_wrap_policy is not None
                                    else "none"
                                ),
                                dtype=args.dtype,
                                world_size=world_size,
                                total_callable_construction_time=total_cct,
                                warmup_stats=all_warmup_stats,
                                benchmark_stats=all_benchmark_stats,
                                typehint=args.typehint,
                                rank_mem_info=rank_mem_info,
                            ).to_json()
                        )

        if args.dataparallel_strategy == "ddp":
            from thunder.benchmarks import get_default_thunder_ddp_dynamic_strides_executor

            print(f"# bucket sizes: {args.bucket_sizes=}")
            for bucket_size_in_mb in args.bucket_sizes:
                print(f"thunder - ddp - {bucket_size_in_mb=}")
                total_cct, all_warmup_stats, all_benchmark_stats, rank_mem_info = run_multiprocess_benchmark(
                    b,
                    get_default_thunder_ddp_dynamic_strides_executor(bucket_size_in_mb),
                    world_size=world_size,
                    extended_printout=False,
                )
                results.append(
                    ResultFormatter(
                        model_name=args.model,
                        base_name="thunder-ddp",
                        suffix=f"bucket_size_{bucket_size_in_mb}MB",
                        dtype=args.dtype,
                        world_size=world_size,
                        total_callable_construction_time=total_cct,
                        warmup_stats=all_warmup_stats,
                        benchmark_stats=all_benchmark_stats,
                        typehint=args.typehint,
                        rank_mem_info=rank_mem_info,
                    ).to_json()
                )
        else:
            from itertools import product
            from thunder.benchmarks import get_default_thunder_fsdp_dynamic_strides_executor

            bucketing_strategies = [fsdp_bucketing_strategies[s] for s in args.bucketing_strategies]
            sharding_strategies = [fsdp_sharding_strategies[s] for s in args.sharding_strategies]
            for bucketing_strategy, sharding_strategy in product(bucketing_strategies, sharding_strategies):
                print(f"thunder - fsdp - {bucketing_strategy=} - {sharding_strategy=}")
                total_cct, all_warmup_stats, all_benchmark_stats, rank_mem_info = run_multiprocess_benchmark(
                    b,
                    get_default_thunder_fsdp_dynamic_strides_executor(bucketing_strategy, sharding_strategy),
                    world_size=world_size,
                    extended_printout=False,
                )
                results.append(
                    ResultFormatter(
                        model_name=args.model,
                        base_name="thunder-fsdp",
                        suffix=f"bucketing_strategy_{str(bucketing_strategy).lower().split('.')[-1]}_{(str(sharding_strategy).lower().split('.')[-1])}",
                        dtype=args.dtype,
                        world_size=world_size,
                        total_callable_construction_time=total_cct,
                        warmup_stats=all_warmup_stats,
                        benchmark_stats=all_benchmark_stats,
                        typehint=args.typehint,
                        rank_mem_info=rank_mem_info,
                    ).to_json()
                )

        out = args.out
        if not os.path.exists(out):
            os.makedirs(out)
        json_path = os.path.join(
            out,
            f"{args.model}-{args.dataparallel_strategy}-seq_len_{args.seq_len}-dtype_{args.dtype}-world_size_{args.world_size}.json",
        )
        with open(
            json_path,
            "w",
        ) as f:
            json.dump(results, f, indent=2)
        print(f"\n\tSaved json to {json_path}")
