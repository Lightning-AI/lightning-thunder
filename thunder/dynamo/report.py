from __future__ import annotations
from typing import TYPE_CHECKING
import subprocess
import sys
from pathlib import Path
import json
import torch

from thunder.dynamo.compiler import thunderfx
from thunder.core.utils import sequencify
from thunder.core.pytree import tree_flatten


if TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Callable


def run_backward(fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    result = sequencify(result)

    forward_inputs = tree_flatten((args, kwargs))[0]
    forward_inputs = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, forward_inputs))
    differentiable_tensor_result = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, result))

    output_grads = []
    for diff_result in differentiable_tensor_result:
        output_grads.append(torch.ones_like(diff_result))

    for i in forward_inputs:
        i.grad = None

    torch.autograd.backward(result, output_grads, inputs=forward_inputs)
    return result, [t.grad for t in forward_inputs]


def run_repro(executor_dict, executor_name, model, compute_type, *inputs) -> dict[str, float]:
    """Helper function to execute the forward or backward pass based on the `compute_type` using the executor specified by `executor_name` in `executor_dict`.
    If the execution fails, an error is raised. On success, the function returns a dictionary containing the forward results and gradient results.
    """
    if executor_name == "eager":
        compiled_fn = model
    elif executor_name == "torch_inductor":
        compiled_fn = executor_dict[executor_name](model, inputs)
    else:
        compiled_fn = executor_dict[executor_name](model)

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
        case _:
            raise ValueError(
                f"Invalid compute type: '{compute_type}'. Only 'forward' or 'forward+backward' are allowed."
            )
    return results


def thunderfx_report(
    fn: Callable,
    *args,
    compile_kwargs: dict = None,
    folder_path: str | PathLike = "/tmp/thunderfx_report",
    check_consistency: bool = True,
    check_benchmark: bool = True,
    save_consistency_inputs: bool = False,
    save_benchmark_inputs: bool = False,
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

    folder = Path(folder_path)
    # NOTE If the input folder path contains subfolders named 'benchmark' or 'consistency', they will be overwritten.
    folder.mkdir(exist_ok=True)
    # Checks consistency with Torch eager
    if check_consistency:
        print("\nVerifying consistency between Thunder and Torch eager ...")
        consistency_folder = folder / "consistency"
        consistency_folder.mkdir(exist_ok=True)
        compiled._backend.save_reproducer_to_folder(consistency_folder, save_input_tensor=save_consistency_inputs)
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
        print("\nAnalyzing performance through benchmarking, this might take a moment...")
        benchmark_folder = folder / "benchmark"
        benchmark_folder.mkdir(exist_ok=True)
        compiled._backend.save_reproducer_to_folder(
            benchmark_folder, save_input_tensor=save_benchmark_inputs, use_pytest_benchmark=True
        )

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
                    "-q",  # reduce the output
                ],
                capture_output=True,
                text=True,
            )
            g_name = file.name.rstrip(".py")
            if benchmark_result.returncode:
                print(
                    f"Failed to run the benchmarking script({benchmark_folder / file}), exception: {benchmark_result.stderr}"
                )
            else:
                print(f"{g_name}:\n{benchmark_result.stdout}\n")

        print("Max allocated memory usage:")
        for tmp_json in benchmark_json_files:
            with open(tmp_json) as file:
                data = json.load(file)
                bench = data["benchmarks"]

            def print_sorted_memory_info(compute_t: str):
                filtered_bench = [b for b in bench if compute_t in b["param"]]
                filtered_bench_sorted = sorted(filtered_bench, key=lambda x: x["extra_info"]["max_allocated_memory_MB"])
                for bk in filtered_bench_sorted:
                    print(f"{bk['name'].lstrip('test_')}: {bk['extra_info']['max_allocated_memory_MB']/1000} GB")
                print("\n")

            print_sorted_memory_info("forward")
            print_sorted_memory_info("backward")
