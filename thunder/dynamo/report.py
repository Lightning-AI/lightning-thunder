from __future__ import annotations
from typing import TYPE_CHECKING
import subprocess
import sys
from pathlib import Path
import json
import argparse

import torch
from thunder.dynamo.compiler import thunderfx
from thunder.benchmarks.targets import ComputeType, backward_only
from thunder.dynamo.utils import run_backward

if TYPE_CHECKING:
    from thunder.dynamo.utils import SubgraphInfo
    from os import PathLike
    from collections.abc import Callable


def run_repro(ex_dict, ex_name, model, compute_type, *inputs):  # CLI options
    if ex_name == "eager":
        compiled_fn = model
    elif ex_name == "torch_inductor":
        compiled_fn = ex_dict[ex_name](model, inputs)
    else:
        compiled_fn = ex_dict[ex_name](model)

    results = {}
    match compute_type:
        case "forward":
            try:
                result = compiled_fn(*inputs)
            except Exception as e:
                raise e
            results["forward"] = result
        case "forward+backward":
            try:
                forward_result, grads = run_backward(compiled_fn, *inputs)
            except Exception as e:
                raise e
            results["forward"] = forward_result
            results["backward"] = grads
    return results


def get_thunder_graph_names(subgraph_infos: list[SubgraphInfo]):
    thunder_graph_names = []
    for graph_idx, subgraph_info in enumerate(subgraph_infos):
        for node in subgraph_info.split_graph_module.graph.nodes:
            target = node.target
            if isinstance(target, str) and target.startswith("thunder_"):
                thunder_graph_names.append(f"graph{graph_idx}_{target}")
    return thunder_graph_names


def thunderfx_save_report(
    fn: Callable,
    *args,
    compile_kwargs: dict = None,
    folder_path: str | PathLike = "/tmp/thunderfx_report",
    check_consistency: bool = True,
    check_benchmark: bool = True,
    save_benchmark_inputs: bool = True,
    **kwargs,
):
    try:
        compiled = thunderfx(fn, **compile_kwargs) if compile_kwargs is not None else thunderfx(fn)
        compiled(*args, **kwargs)
    except Exception as e:
        print(f"Failed to run the function using ThunderFX with exception: {e}")
        try:
            compiled._backend.save_reproducer_to_folder(folder_path)
        except Exception as repro_e:
            print(f"Failed to save reproducer due to {repro_e}")
            return
        print(f"The reproducer file is saved in {folder_path}")
        return
    print("The input callable can be successfully executed by ThunderFX.")
    if not check_benchmark and not check_consistency:
        return

    thunder_graph_names = get_thunder_graph_names(compiled._backend.subgraph_infos)
    EXECUTOR_NAMES = ("eager", "thunder", "torch_inductor")

    report_result: dict[str, list] = {}
    for g_name in thunder_graph_names:
        for ex in EXECUTOR_NAMES:
            # Sets the consistency field to None for eager
            report_result[f"{g_name}[{ex}]"] = [None] if ex == "eager" else []

    folder = Path(folder_path)
    # NOTE If the input folder path contains subfolders named 'benchmark' or 'consistency', they will be overwritten.
    folder.mkdir(exist_ok=True)
    # Checks consistency with Torch eager
    if check_consistency:
        print("Verifying consistency between Thunder and Torch eager ...")
        consistency_folder = folder / "consistency"
        consistency_folder.mkdir(exist_ok=True)
        compiled._backend.save_reproducer_to_folder(consistency_folder)
        for file in consistency_folder.glob("*.py"):
            g_name = file.name.rstrip(".py")
            cmd = [sys.executable, folder / file, "--check_consistency=True", "--compute_type=forward+backward"]
            consistency_result = subprocess.run(cmd, capture_output=True, text=True)
            if consistency_result.returncode:
                error = consistency_result.stderr
                print(f"[{g_name}] Consistency check failed: {error}")
            else:
                print(f"[{g_name}] Consistency check succeeded")

    # Benchmark
    if check_benchmark:
        print("Analyzing performance through benchmarking, this might take a moment...")
        benchmark_folder = folder / "benchmark"
        benchmark_folder.mkdir(exist_ok=True)
        compiled._backend.save_reproducer_to_folder(benchmark_folder, save_input_tensor=True, use_pytest_benchmark=True)

        benchmark_json_files = []
        for file in benchmark_folder.glob("*.py"):
            benchmark_json_files.append(str(benchmark_folder / f"{file.name.replace('.py', '.json')}"))
            benchmark_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    benchmark_folder / file,
                    "--benchmark-timer=torch.utils.benchmark.utils.timer.timer",
                    "--benchmark-warmup=on",
                    "--benchmark-group-by=param:compute_type",
                    f"--benchmark-json={benchmark_json_files[-1]}",
                    "--disable-warnings",
                    "-q",
                ],
                capture_output=True,
                text=True,
            )
        print(benchmark_result.stdout)
        print("Max allocated memory usage:")
        for tmp_json in benchmark_json_files:
            with open(tmp_json) as file:
                data = json.load(file)
                benchs = data["benchmarks"]
                forward_benchs = [bench for bench in benchs if "forward" in bench["param"]]
                backward_benchs = [bench for bench in benchs if "backward" in bench["param"]]

                forward_benchs_sorted = sorted(
                    forward_benchs, key=lambda x: x["extra_info"]["max_allocated_memory_MB"], reverse=True
                )
                backward_benchs_sorted = sorted(
                    backward_benchs, key=lambda x: x["extra_info"]["max_allocated_memory_MB"], reverse=True
                )

                for bk in forward_benchs_sorted:
                    print(f"{bk['name'].lstrip('test_')}: {bk['extra_info']['max_allocated_memory_MB']/1000} GB")
                print("\n")
                for bk in backward_benchs_sorted:
                    print(f"{bk['name'].lstrip('test_')}: {bk['extra_info']['max_allocated_memory_MB']/1000} GB")
                print("\n")
