from __future__ import annotations

import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from abc import abstractmethod

import torch

from torch.nn.modules.module import _addindent

from thunder.core.pytree import tree_flatten
from thunder.core.utils import check, sequencify

from thunder.dynamo.compiler import thunderfx
from thunder.dynamo.utils import _get_example_inputs_from_placeholder, _readable, arg_like, CompilerType, get_env


if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from typing import List, Type
    from collections.abc import Sequence

    from thunder.dynamo.utils import ExampleInputMetaData


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


def run_repro1(compiled_fn, compute_type, *inputs) -> dict[str, float]:
    """Helper function to execute the forward or backward pass based on the `compute_type` using the executor specified by `executor_name` in `executor_dict`.
    If the execution fails, an error is raised. On success, the function returns a dictionary containing the forward results and gradient results.
    """
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


class FXGraphReportBase:
    def __init__(self, graph: torch.fx.GraphModule, graph_name: str):
        self.graph = graph
        self.graph_name = graph_name

    def get_input_metadata(self):
        placeholders = list(n for n in self.graph.graph.nodes if n.op == "placeholder")
        example_input_metadata = map(partial(_get_example_inputs_from_placeholder, only_metadata=True), placeholders)
        return list(example_input_metadata)

    def write_eager_repro(self, folder):
        self.write_repro(folder, "eager", custom_executor_str="compiled_model=model")

    def write_thunder_repro(self, folder):
        thunder_compile_str = "compiled_model=thunder.jit(model)"
        thunder_import_str = ["import thunder"]
        self.write_repro(folder, "thunder", thunder_import_str, thunder_compile_str)

    def write_inductor_repro(self, folder):
        inductor_compile_str = "compiled_model=torch.compile(model)"
        inductor_import_str = ["import torch"]
        self.write_repro(folder, "torchcompile", inductor_import_str, inductor_compile_str)

    @abstractmethod
    def write_repro(
        self,
        folder: str | PathLike,
        custom_executor_name: str,
        custom_import_str: str,
        custom_executor_str: str,
        save_input_tensor: bool,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData],
    ): ...


class FXGraphBenchmarkReport(FXGraphReportBase):
    def write_repro(
        self,
        folder: str | PathLike,
        custom_executor_name: str = "",
        custom_import_str: str = "",
        custom_executor_str: str = "",
        save_input_tensor: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
    ):
        if inputs == None:
            inputs = self.get_input_metadata()
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        torch_env, thunder_pkgs = get_env()
        # Ideally we'd use print_readable, but we want verbose=False and there's no
        # way to set that with print_readable.
        readable = _readable(self.graph, "DynamoModule", print_output=False)
        has_cuda_args = any(hasattr(arg, "device") and arg.device.type == "cuda" for arg in inputs)

        comment_str = f'''"""
Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:
{torch_env}

Versions of Thunder related libraries:
{thunder_pkgs}"""
'''
        del torch_env, thunder_pkgs

        code_str = ""
        code_str += f"""# NOTE: This script requires `pytest-benchmark==4.0.0` to be installed.
# To execute the script, run `pytest {self.graph_name}.py --benchmark-timer=torch.utils.benchmark.utils.timer.timer --benchmark-warmup=on --benchmark-group-by=param:compute_type`
# To check the peak allocated CUDA memory, use --benchmark-json=json_file_name and look at the "max_allocated_memory_MB" field in the json file
# To run tests for a specific compute_type, use the pytest `-k` option.
# For example:
#   - `-k "forward"` will run only the forward pass.
#
# Available options:
#   - compute_type: "forward", "backward"

import pytest
from thunder.benchmarks.targets import parametrize_compute_type_only_training, benchmark_for_compute_type
"""
        # The packages that are likely to be used by the code generated from the Torch GraphModule
        code_str += "\n".join([v.import_str for v in torch.fx.graph._custom_builtins.values()]) + "\n"
        code_str += "\n".join(custom_import_str) + "\n"

        code_str += f"""

@parametrize_compute_type_only_training"""
        func_str = f"def test_{self.graph_name}(benchmark, compute_type):\n{readable}\n"

        if any(arg is None for arg in inputs):
            func_str += f"# Warning: The inputs that cannot be inferred are set to None, requiring the user to manually give inputs according to the code\n"
        if save_input_tensor:
            example_inputs = eval(f"[\n{chr(10).join(arg_like(a) for a in inputs)}]")
            input_file_name = folder / f"{self.graph_name}_inputs.pt"
            torch.save(example_inputs, input_file_name)
            func_str += f"inputs = torch.load('{input_file_name}')\n"
        else:
            input_str = f"""inputs = [\n{chr(10).join(arg_like(a) for a in inputs)}\n"""
            func_str += f"{_addindent(input_str, 4)}\n]\n"
            del input_str

        func_str = f"""{func_str}
model = DynamoModule()
{custom_executor_str}
"""
        if not has_cuda_args:
            func_str += f"""benchmark(compiled, *inputs)"""
        else:
            func_str += f"""benchmark_for_compute_type(compute_type, benchmark, compiled_model, inputs, {{}})"""

        with open(folder / f"{self.graph_name}_{custom_executor_name}.py", "w") as f:
            print(comment_str, file=f)
            print(code_str, file=f)
            print(_addindent(func_str, 4), file=f)


class FXGraphReport(FXGraphReportBase):
    def write_repro(
        self,
        folder: str | PathLike,
        custom_executor_name: str = "",
        custom_import_str: str = "",
        custom_executor_str: str = "",
        save_input_tensor: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
    ):
        check(
            "compiled_model" in custom_executor_str and "model" in custom_executor_str,
            lambda: "custom_executor_str needs to be in the form `compiled_model=your_executor_code(model, your_other_args)`, the inputs of the `model` is named as `inputs`",
            ValueError,
        )
        if inputs == None:
            inputs = self.get_input_metadata()
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        torch_env, thunder_pkgs = get_env()
        # Ideally we'd use print_readable, but we want verbose=False and there's no
        # way to set that with print_readable.
        readable = _readable(self.graph, "DynamoModule", print_output=False)

        COMMAND_LINE_ARGS = """
import argparse

parser = argparse.ArgumentParser(description="Script for executing an FX graph with specified configurations.")

parser.add_argument(
    "--check_consistency",
    type=bool,
    default=False,
    help="Whether to check consistency (default: False)"
)
parser.add_argument(
    "--compute_type",
    type=str,
    choices=["forward", "forward+backward"],
    default="forward",
    help="Type of computation to perform (forward, forward+backward)"
)

args = parser.parse_args()
compute_type = args.compute_type
check_acc = args.check_consistency
"""

        comment_str = f'''"""
Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:
{torch_env}

Versions of Thunder related libraries:
{thunder_pkgs}"""
'''
        del torch_env, thunder_pkgs

        code_str = ""
        # The packages that are likely to be used by the code generated from the Torch GraphModule
        code_str += "\n".join([v.import_str for v in torch.fx.graph._custom_builtins.values()]) + "\n"
        code_str += "\n".join(custom_import_str) + "\n"
        code_str += f"\n{COMMAND_LINE_ARGS}"
        func_str = f"def test_{self.graph_name}():\n{readable}\n"

        if any(arg is None for arg in inputs):
            func_str += f"# Warning: The inputs that cannot be inferred are set to None, requiring the user to manually give inputs according to the code\n"
        if save_input_tensor:
            example_inputs = eval(f"[\n{chr(10).join(arg_like(a) for a in inputs)}]")
            input_file_name = folder / f"{self.graph_name}_inputs.pt"
            torch.save(example_inputs, input_file_name)
            func_str += f"inputs = torch.load('{input_file_name}')\n"
        else:
            input_str = f"""inputs = [\n{chr(10).join(arg_like(a) for a in inputs)}\n"""
            func_str += f"{_addindent(input_str, 4)}\n]\n"
            del input_str

        func_str += f"""
model = DynamoModule()
from thunder.dynamo.report import run_repro1

{custom_executor_str}
result = run_repro1(compiled_model, compute_type, *inputs)
if check_acc:
    eager_result = run_repro1(model, compute_type, *inputs)
    for (compute_t, eager_v), (_, cur_v) in zip(eager_result.items(), result.items()):
        torch.testing.assert_close(eager_v, cur_v, msg=lambda e : f'{{compute_t}}: {{e}}')
"""

        with open(folder / f"{self.graph_name}_{custom_executor_name}.py", "w") as f:
            print(comment_str, file=f)
            print(code_str, file=f)
            print(_addindent(func_str, 4), file=f)
            print(f"\ntest_{self.graph_name}()", file=f)


class FXReport:
    def __init__(self, graph_report_cls: type[FXGraphReportBase], graphs: list[torch.fx.GraphModule]):
        self.fx_graph_reports: list[FXGraphReportBase] = [
            graph_report_cls(g, f"graph{idx}") for idx, g in enumerate(graphs)
        ]

    def make_full_reports(self):
        pass


def fx_report(fn: Callable, *args, use_benchmark: bool = False, **kwargs) -> FXReport:
    graphs = []

    def helper_backend(gm, example_inputs):
        """Helper function to collect FX graphs."""
        graphs.append(gm)
        return gm.forward

    try:
        compiled = torch.compile(fn, backend=helper_backend)
        compiled(*args, **kwargs)
    except Exception as e:
        print(f"Failed to run the function using torch.compile with exception: {e}")
    if use_benchmark:
        return FXReport(FXGraphBenchmarkReport, graphs)
    return FXReport(FXGraphReport, graphs)
