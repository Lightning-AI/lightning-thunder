from __future__ import annotations

import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
import textwrap

import torch
from thunder.core.pytree import tree_flatten
from thunder.core.utils import sequencify
from thunder.dynamo.compiler import thunderfx
from thunder.dynamo.utils import _get_example_inputs_from_placeholder, _readable, arg_like, get_env
from thunder.dynamo.repro_script_template import benchmark_multi_exe_code_template, repro_code_template


if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
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


def run_repro(compiled_fn, compute_type, *inputs) -> dict[str, float]:
    """Helper function to execute the forward or backward pass based on the `compute_type` using the executor specified by `executor_name` in `executor_dict`.
    If the execution fails, an error is raised. On success, the function returns a dictionary containing the forward results and gradient results.
    """
    results = {}
    match compute_type:
        case "forward":
            result = compiled_fn(*inputs)
            results["forward"] = result
        case "forward+backward":
            forward_result, grads = run_backward(compiled_fn, *inputs)
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
    serialize_consistency_inputs: bool = False,
    serialize_benchmark_inputs: bool = False,
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
        compiled._backend.save_reproducer_to_folder(consistency_folder, serialize_inputs=serialize_consistency_inputs)
        for file in consistency_folder.glob("*.py"):
            g_name = file.name.rstrip(".py")
            cmd = [sys.executable, file, "--check_consistency=True", "--compute_type=forward+backward"]
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
            benchmark_folder, serialize_inputs=serialize_benchmark_inputs, use_pytest_benchmark=True
        )

        benchmark_json_files = []
        for file in benchmark_folder.glob("*.py"):
            benchmark_json_files.append(str(benchmark_folder / f"{file.name.replace('.py', '.json')}"))
            benchmark_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    file,
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
                print(f"Failed to run the benchmarking script({file}), exception: {benchmark_result.stderr}")
            else:
                print(f"{g_name}:\n{benchmark_result.stdout}\n")

        print("Max allocated CUDA memory usage:")
        for tmp_json in benchmark_json_files:
            with open(tmp_json) as file:
                data = json.load(file)
                bench = data["benchmarks"]

            def print_sorted_memory_info(compute_t: str):
                filtered_bench = [b for b in bench if compute_t in b["param"]]
                filtered_bench_sorted = sorted(
                    filtered_bench, key=lambda x: x["extra_info"].get("max_allocated_memory_MB", 0)
                )
                for bk in filtered_bench_sorted:
                    print(f"{bk['name'].lstrip('test_')}: {bk['extra_info'].get('max_allocated_memory_MB', 0)/1000} GB")
                print("\n")

            print_sorted_memory_info("forward")
            print_sorted_memory_info("backward")


class FXGraphReport:
    """
    Encapsulates an FX Graph module with metadata to aid in generating
    reproduction and benchmarking scripts for various executors.
    """

    def __init__(self, graph: torch.fx.GraphModule, graph_name: str):
        self.graph = graph
        self.graph_name = graph_name

    def get_input_metadata(self):
        placeholders = list(n for n in self.graph.graph.nodes if n.op == "placeholder")
        example_input_metadata = map(partial(_get_example_inputs_from_placeholder, only_metadata=True), placeholders)
        return list(example_input_metadata)

    def write_eager_repro(self, folder, use_benchmark: bool = False, serialize_inputs: bool = False):
        if use_benchmark:
            self.write_benchmark_repro(
                folder,
                f"{self.graph_name}_benchmark_eager.py",
                ["eager"],
                executor_str=["None"],
                serialize_inputs=serialize_inputs,
            )
        else:
            self.write_repro(
                folder,
                f"{self.graph_name}_repro_eager.py",
                executor_str="None",
                serialize_inputs=serialize_inputs,
            )

    def write_thunder_repro(self, folder, use_benchmark: bool = False, serialize_inputs: bool = False):
        thunder_compile_str = "thunder.jit"
        thunder_import_str = ["import thunder"]
        if use_benchmark:
            self.write_benchmark_repro(
                folder,
                f"{self.graph_name}_benchmark_thunder.py",
                ["thunder"],
                [thunder_compile_str],
                thunder_import_str,
            )
        else:
            self.write_repro(
                folder,
                f"{self.graph_name}_repro_thunder.py",
                thunder_compile_str,
                thunder_import_str,
                serialize_inputs,
            )

    def write_inductor_repro(self, folder, use_benchmark: bool = False, serialize_inputs: bool = False):
        inductor_compile_str = "torch.compile"
        inductor_import_str = ["import torch"]
        if use_benchmark:
            self.write_benchmark_repro(
                folder,
                f"{self.graph_name}_benchmark_torchcompile.py",
                ["torchcompile"],
                [inductor_compile_str],
                inductor_import_str,
                serialize_inputs,
            )
        else:
            self.write_repro(
                folder,
                f"{self.graph_name}_repro_torchcompile.py",
                inductor_compile_str,
                inductor_import_str,
                serialize_inputs,
            )

    def _get_input_str(self, folder, inputs, serialize_inputs):
        input_str = ""
        if any(arg is None for arg in inputs):
            input_str += f"# Warning: The inputs that cannot be inferred are set to None, requiring the user to manually give inputs according to the code\n"
        if serialize_inputs:
            example_inputs = eval(f"[\n{chr(10).join(arg_like(a) for a in inputs)}]")
            input_file_name = folder / f"{self.graph_name}_inputs.pt"
            torch.save(example_inputs, input_file_name)
            input_str += f"inputs = torch.load('{input_file_name}')\n"
        else:
            input_str = "inputs = [\n"
            input_str += textwrap.indent("\n".join(arg_like(a) for a in inputs), "    ")
            input_str += "\n]"
        return input_str

    def write_benchmark_repro(
        self,
        folder: str | PathLike,
        file_name: str,
        executor_name_str: list[str],
        executor_str: list[str],
        import_str: list[str] = None,
        serialize_inputs: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
        **kwargs,
    ) -> None:
        """
        Generates a benchmark reproduction script for a given FX graph module using various executors and writes it to the specified file.

        Args:
            folder (str | PathLike): The target directory where the script will be saved.
            file_name (str): The name of the output script file.
            executor_name_str (list[str]): A list of executor names to be used in the script.
            executor_str (list[str]): A list of compilation commands to run the benchmark.
                Each command should be applicable to the original model, e.g.,
                ``compile_command(original_model)``.
                Example values: ``"thunder.jit"`` or ``"partial(thunder.jit, executors=[nvfuser_executor])"``.
            import_str (list[str], optional): A list of necessary import statements for the script.
                For example, if using the ``nvfuser_executor``, include:
                ``import_str=["from thunder import nvfuser_executor"]``.
            serialize_inputs (bool, optional): Whether to serialize the inputs for reproducibility. Defaults to False.
                If enabled, all inputs will be serialized into a single file: "{graph_name}_input.pt".
            inputs (Sequence[torch.Tensor | ExampleInputMetaData], optional):
                The input tensors or metadata for the FX graph module. Defaults to None.
                If not provided, inputs will be inferred from the placeholders in the FX graph.
            **kwargs: Additional arguments for customization.

        Example:
            See the example in :func:`fx_report`.
        """

        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        if inputs == None:
            inputs = self.get_input_metadata()
        has_cuda_args = any(hasattr(arg, "device") and arg.device.type == "cuda" for arg in inputs)
        has_requires_grad_args = any(hasattr(arg, "requires_grad") and arg.requires_grad for arg in inputs)
        torch_env, thunder_pkgs = get_env()
        readable = textwrap.indent(_readable(self.graph, "DynamoModule", print_output=False), "    ")
        # The packages that are likely to be used by the code generated from the Torch GraphModule
        torch_import_str = "\n".join([v.import_str for v in torch.fx.graph._custom_builtins.values()])
        import_str = "" if import_str == None else "\n".join(import_str)
        input_str = textwrap.indent(self._get_input_str(folder, inputs, serialize_inputs), "    ")
        call_bench_str = f"benchmark_for_compute_type(compute_type, benchmark, compiled_model, inputs, {{}}, has_cuda={True if has_cuda_args else False})"
        compute_type_decorator = (
            "@parametrize_compute_type_only_training"
            if has_requires_grad_args
            else """@pytest.mark.parametrize("compute_type", (ComputeType.INFERENCE,), ids=("inference",))"""
        )

        executor_names_str = f"executor_names={executor_name_str}"
        executors_str = "executors=[\n    " + ",\n    ".join(executor_str) + "\n]"
        extra_comment_str = kwargs.get("extra_comment_str") if "extra_comment_str" in kwargs else ""
        code_str = benchmark_multi_exe_code_template.format(
            torch_env=torch_env,
            thunder_pkgs=thunder_pkgs,
            torch_import_str=torch_import_str,
            import_str=import_str,
            dynamo_module=readable,
            inputs=input_str,
            executors=executors_str,
            graph_name=self.graph_name,
            call_benchmark=call_bench_str,
            executor_names=executor_names_str,
            compute_type_decorator=compute_type_decorator,
            extra_comment_str=extra_comment_str,
        )
        with open(folder / file_name, "w") as f:
            print(code_str, file=f)

    def write_repro(
        self,
        folder: str | PathLike,
        file_name: str,
        executor_str: str,
        import_str: list[str] = None,
        serialize_inputs: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
        **kwargs,
    ) -> None:
        """
        Generates a reproduction script for a given FX graph module and writes it to a file.

        Args:
            folder (str | PathLike): The target directory where the script will be saved.
            file_name (str): The name of the output script file.
            executor_str (str): compilation commands to compile the graph. e.g.: ``"partial(thunder.jit, executors=[nvfuser_executor])"``
            import_str (list[str], optional): A list of necessary import statements for the script.
                For example, if using the ``nvfuser_executor``, include:
                ``import_str=["from thunder import nvfuser_executor"]``.
            serialize_inputs (bool, optional): Whether to serialize the inputs for reproducibility.
                Defaults to False. If enabled, all inputs will be saved to a file named
                "{graph_name}_input.pt".
            inputs (Sequence[torch.Tensor | ExampleInputMetaData], optional):
                The input tensors or metadata for the FX graph module. Defaults to None, in
                which case the inputs will be inferred from the placeholders in the FX graph.
            **kwargs: Additional arguments for customization.

        Example:
            See the example in :func:`fx_report`.
        """
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        if inputs == None:
            inputs = self.get_input_metadata()
        torch_env, thunder_pkgs = get_env()
        readable = textwrap.indent(_readable(self.graph, "DynamoModule", print_output=False), "    ")
        # The packages that are likely to be used by the code generated from the Torch GraphModule
        torch_import_str = "\n".join([v.import_str for v in torch.fx.graph._custom_builtins.values()])
        import_str = "" if import_str == None else "\n".join(import_str)
        input_str = textwrap.indent(self._get_input_str(folder, inputs, serialize_inputs), "    ")
        extra_comment_str = kwargs.get("extra_comment_str") if "extra_comment_str" in kwargs else ""

        code_str = repro_code_template.format(
            torch_env=torch_env,
            thunder_pkgs=thunder_pkgs,
            torch_import_str=torch_import_str,
            import_str=import_str,
            dynamo_module=readable,
            inputs=input_str,
            executor_str=executor_str,
            graph_name=self.graph_name,
            extra_comment_str=extra_comment_str,
        )

        with open(folder / file_name, "w") as f:
            print(code_str, file=f)


class FXReport:
    """
    This class stores a list of FXGraphReport instances, each of which wraps an FX graph
    module and provides methods to generate reproduction and benchmark scripts.
    """

    def __init__(self, graphs: list[torch.fx.GraphModule], graph_names: list[str] = None):
        if graph_names is None:
            graph_names = [f"graph{idx}" for idx in range(len(graphs))]
        self.fx_graph_reports: list[FXGraphReport] = [FXGraphReport(g, name) for name, g in zip(graph_names, graphs)]

    def __str__(self):
        return f"<FXReport with {len(self.fx_graph_reports)} FXGraphReports accessible via .fx_graph_reports>"


def fx_report(fn: Callable, *args, compile_options: dict = None, **kwargs) -> FXReport:
    """
    This function compiles a given function using :func:`torch.compile` with specified ``compile_options``,
    and applies a custom backend that intercepts and collects FX graph modules during execution.
    The function then returns an :class:`FXReport`, which contains utilities to generate
    reproduction and benchmark scripts for each collected FX graph module.

    Args:
        fn (Callable): The function to be compiled and analyzed.
        *args: Positional arguments of ``fn``.
        compile_options(dict): the options that are passed to :func:`torch.compile`
        **kwargs: Keyword arguments of ``fn``.

    Example:
        .. code-block:: python

        import tempfile

        import torch
        from thunder.dynamo.report import fx_report

        def model(x):
            return x * 2

        report = fx_report(model, torch.randn((2, 2), requires_grad=True, device="cuda"), compile_options={"dynamic": False})
        print(len(report.fx_graph_reports))
        with tempfile.TemporaryDirectory() as tmpdir:
            for graph_report in report.fx_graph_reports:
                graph_report.write_eager_repro(tmpdir)
                graph_report.write_thunder_repro(tmpdir, use_benchmark=True)
                graph_report.write_inductor_repro(tmpdir)

                # Executes using the user-defined executor.
                my_executor = "partial(thunder.jit, transforms=[NvtxProfileTransform()], executors=[nvfuser_executor])"
                my_imports = [
                    "import thunder",
                    "from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform",
                    "from thunder import nvfuser_executor",
                    "from functools import partial",
                ]
                graph_report.write_repro(
                    tmpdir, f"{graph_report.graph_name}_mythunder_repro.py", executor_str=my_executor, import_str=my_imports
                )
                graph_report.write_benchmark_repro(
                    tmpdir,
                    f"{graph_report.graph_name}_mythunder_benchmark.py",
                    executor_name_str=["mythunder"],
                    executor_str=[my_executor],
                    import_str=my_imports,
                )
    """
    graphs = []

    def helper_backend(gm, example_inputs):
        """Helper function to collect FX graphs."""
        graphs.append(gm)
        return gm.forward

    if compile_options is None:
        compile_options = {}
    compiled = torch.compile(fn, **compile_options, backend=helper_backend)
    compiled(*args, **kwargs)

    return FXReport(graphs)
