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
from thunder.core.utils import sequencify, create_python_callable_from_bsym
from thunder.dynamo.compiler import thunderfx
from thunder.dynamo.utils import (
    _get_example_inputs_from_placeholder,
    _readable,
    arg_like,
    get_env,
    get_split_reasons_string,
    CompilerType,
)
from thunder.dynamo.repro_script_template import (
    benchmark_multi_exe_code_template,
    repro_code_template,
    bsym_torch_compile_repro_template,
)
from thunder import last_traces, last_backward_traces


if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from collections.abc import Sequence

    from thunder.dynamo.utils import ExampleInputMetaData
    from thunder.dev_utils.utils import BenchmarkComparisonData
    from thunder.core.trace import TraceCtx
    from thunder.core.symbol import BoundSymbol


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

    def __repr__(self):
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


class ThunderSplitGraphReport(FXGraphReport):
    """
    A report class representing a Thunder-split FX subgraph, extending :class:FXGraphReport.

    This class encapsulates details about a subgraph that was split from the original
    Dynamo FX graph due to Thunder-specific transformations.

    Attributes:
        graph: The Thunder-split FX graph.
        graph_name: The name of the Thunder-split FX graph.
        compiled_fn: The Thunder-compiled function corresponding to :attr:`graph`.
        example_input: An example input used for execution.
        thunder_options: Configuration options specific to :func:`thunder.jit`.
        split_reason: The reason why :func:`thunder.dynamo.splitter._splitter` split the original graph.
        fusion_reports (list[ThunderFusionReport]): A list of fusion reports for the
            nvFusion regions generated when using the nvFuser executor.
            See :class:`ThunderFusionReport` for more details.
        fwd_trc: The forward trace, available only after calling :meth:`_create_thunder_traces`.
        bwd_trc: The backward trace, available only after calling :meth:`_create_thunder_traces`.

    For an example, see the documentation of :func:`analyze_thunder_splits`.
    """

    def __init__(
        self,
        graph: torch.fx.GraphModule,
        graph_name: str,
        compiled_fn: Callable,
        example_input,
        thunder_options: dict,
        split_reason: str,
    ):
        super().__init__(graph, graph_name)
        self.compiled_fn = compiled_fn
        self.example_input = example_input
        self.thunder_options = thunder_options
        self.split_reason = split_reason

        self.fusion_reports: list[ThunderFusionReport] = []
        self.fwd_trc: TraceCtx = None
        self.bwd_trc: TraceCtx = None

    def __repr__(self):
        return f"<ThunderSplitGraphReport with {len(self.fusion_reports)} ThunderFusionReport accessible via .fusion_reports>"

    def _create_thunder_traces(self):
        example_inputs = eval(f"[\n{chr(10).join(arg_like(a) for a in self.example_input)}]")
        # Executes to get the trace
        run_backward(self.compiled_fn, *example_inputs)
        self.fwd_trc = last_traces(self.compiled_fn)[-1]
        self.bwd_trc = last_backward_traces(self.compiled_fn)[-1]

    def create_fusion_reports(self):
        """
        Runs the Thunder-compiled function to obtain the nvFusion definition
        and generate the :class:`ThunderFusionReport` instance based on it.
        """
        self._create_thunder_traces()
        for trace, prefix in [(self.fwd_trc, "forward"), (self.bwd_trc, "backward")]:
            for bsym in trace.bound_symbols:
                if bsym.sym.is_fusion and "nvFusion" in bsym.sym.name:
                    self.fusion_reports.append(ThunderFusionReport(bsym, f"{self.graph_name}_{bsym.sym.name}_{prefix}"))

    def write_thunder_repro(self, folder, use_benchmark=False, serialize_inputs=False):
        thunder_ex_str = (
            f"partial(thunder.jit, {self.thunder_options})" if self.thunder_options is None else "thunder.jit"
        )
        has_cuda_args = any(hasattr(arg, "device") and arg.device.type == "cuda" for arg in self.example_input)
        import_str = ["import thunder", "from functools import partial"]
        if has_cuda_args:
            # Since Thunder compile options don't clearly indicate required imports,
            # we include commonly used transforms by default.
            import_str.extend(
                [
                    "from thunder.transforms.cudagraph import CUDAGraphTransform",
                    "from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform",
                ]
            )
        if not use_benchmark:
            super().write_repro(
                folder,
                f"{self.graph_name}_repro_thunder.py",
                executor_str=thunder_ex_str,
                import_str=import_str,
                serialize_inputs=serialize_inputs,
                inputs=self.example_input,
                extra_comment_str=self.split_reason,
            )
            return

        executor_names_list = ["thunder"]
        executors = [thunder_ex_str]

        super().write_benchmark_repro(
            folder,
            f"{self.graph_name}_benchmark_thunder.py",
            executor_names_list,
            executor_str=executors,
            import_str=import_str,
            serialize_inputs=serialize_inputs,
            inputs=self.example_input,
            extra_comment_str=self.split_reason,
        )


class ThunderFusionReport:
    """
    A report class representing a Thunder nvFusion region.

    This class encapsulates information about a nvFusion region created during
    Thunder execution, including the symbolic representation and its name.

    Attributes:
        nvfusion_bsym (BoundSymbol): The symbolic representation of the nvFusion region.
        name (str): The name of the fusion region.

    For an example, see the documentation of :func:`analyze_thunder_splits`.
    """

    def __init__(self, bsym: BoundSymbol, name: str):
        self.nvfusion_bsym = bsym
        self.name = name

    def __repr__(self):
        return f"<ThunderFusionReport of bound symbol\n{self.nvfusion_bsym}>"

    def write_nvfuser_repro(self, folder, file_name=None):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        nvfuser_callable = self.nvfusion_bsym._call_ctx[self.nvfusion_bsym.sym.name]
        fd = nvfuser_callable.last_used

        # The API for nvFuser version >=2.14
        get_repro = getattr(fd, "repro_script_for", None)
        # The legacy nvFuser API
        if get_repro is None:
            get_repro = getattr(fd, "getReproString", None)
        if get_repro is None:
            raise RuntimeError("The installed version of nvFuser does not support repro generation unless on crash.")

        inputs = self.get_inputs()
        repro_code = get_repro(inputs)
        comment_str = f'"""\n{self.nvfusion_bsym}\n"""'

        if file_name == None:
            file_name = f"{self.name}_repro_nvfuser.py"
        with open(folder / file_name, "w") as f:
            print(comment_str, file=f)
            print(repro_code, file=f)

    def get_inputs(self):
        return self.nvfusion_bsym._call_ctx[self.nvfusion_bsym.sym.name].last_inputs

    def write_inductor_repro(self, folder: PathLike, file_name=None):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        python_func = create_python_callable_from_bsym(self.nvfusion_bsym)
        nvfusion_name = self.nvfusion_bsym.sym.name

        inputs = self.get_inputs()
        inputs = "[" + "".join(arg_like(inp) for inp in inputs) + "]"
        program = bsym_torch_compile_repro_template.format(
            python_func=python_func, func_name=nvfusion_name, inputs=inputs
        )
        if file_name == None:
            file_name = f"{self.name}_repro_inductor.py"
        with open(folder / file_name, "w") as f:
            f.write(program)

    def run_benchmark(self) -> BenchmarkComparisonData:
        from thunder.dev_utils.utils import _benchmark_fusion_region_with_nvfuser_and_torch_compile

        return _benchmark_fusion_region_with_nvfuser_and_torch_compile(self.nvfusion_bsym)


class ThunderFXGraphReport(FXGraphReport):
    """
    A Thunder-specific report class for a Dynamo FX Graph, extending FXGraphReport,
    providing the ability to save reproduction/benchmark scripts for the original FX graph.
    Additionally, it includes information about Thunder-split subgraphs in `subgraph_reports`.

    Attributes:
        graph (torch.fx.GraphModule): The original Dynamo FX graph before being split by Thunder.
        graph_name (str): The name of the original Dynamo FX graph.
        split_reason (str): Reasons explaining why the subgraph was split.
        subgraph_reports (list[ThunderSplitGraphReport]): A list of reports for each
            Thunder-split FX graph. For more details, see :class:`ThunderSplitGraphReport`.

    For an example, see the documentation of :func:`analyze_thunder_splits`.
    """

    def __init__(
        self, gm: torch.fx.GraphModule, gm_name: str, split_reason: str, subgraph_reports: list[ThunderSplitGraphReport]
    ):
        super().__init__(gm, gm_name)
        self.split_reason = split_reason
        self.subgraph_reports: list[ThunderSplitGraphReport] = subgraph_reports

    def __repr__(self):
        return f"<ThunderFXGraphReport with {len(self.subgraph_reports)} ThunderSplitGraphReport accessible via .subgraph_reports>"


def analyze_thunder_splits(
    report: FXGraphReport,
    **thunder_options,
) -> ThunderFXGraphReport:
    """
    Generates a :class:`ThunderFXGraphReport` based on an :class:`FXGraphReport`.

    The :class:`ThunderFXGraphReport` provides additional details about Thunder-specific splits
    and nvFusion regions. For more details, see :class:`ThunderFXGraphReport`.

    Example:
        .. code-block:: python

        import tempfile
        from pathlib import Path
        import torch
        from thunder.dynamo.report import (
            fx_report, FXReport, ThunderFXGraphReport, FXGraphReport,
            ThunderSplitGraphReport, ThunderFusionReport, analyze_thunder_splits
        )
        from pathlib import Path

        x = torch.ones(2, 2, device="cuda", requires_grad=True)

        # Dynamo segments `foo` into two graphs. Each graph contains one Thunder-split graph,
        # and each Thunder-split graph has one nvFusion region.
        def foo(x):
            x = x.exp()
            torch._dynamo.graph_break()
            y = torch.sinc(x) + torch.cos(x)
            return y + 1

        # If using `torch.compile` alone, you can stop here and query the reports in `FXReport`.
        # For more details, see the example in :func:`fx_report`.
        results: FXReport = fx_report(foo, x)

        fx_graph_report: FXGraphReport
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for idx, fx_graph_report in enumerate(results.fx_graph_reports):
                # `ThunderFXGraphReport` extends `FXGraphReport`, providing the ability to save
                # reproduction/benchmark scripts for the original FX graph. Additionally, it
                # includes information about Thunder-split subgraphs in `subgraph_reports`.
                thunder_fx_graph_report: ThunderFXGraphReport = analyze_thunder_splits(fx_graph_report)
                # Saves a reproduction script for the original FX graph
                thunder_fx_graph_report.write_thunder_repro(tmp_path)

                thunder_split_report: ThunderSplitGraphReport
                for thunder_split_report in thunder_fx_graph_report.subgraph_reports:
                    split_folder = tmp_path / str(idx)
                    thunder_split_report.write_eager_repro(split_folder)
                    thunder_split_report.write_thunder_repro(split_folder)
                    thunder_split_report.write_inductor_repro(split_folder)

                    # If you are only interested in the Thunder-split FX graph, you can stop here.
                    # If you want to inspect Thunder traces and nvFusion regions further, explicitly call
                    # `ThunderSplitGraphReport.create_fusion_reports()` and analyze as shown below.
                    thunder_split_report.create_fusion_reports()
                    print(f"fwd_trace:\n{thunder_split_report.fwd_trc}\n")
                    print(f"bwd_trace:\n{thunder_split_report.bwd_trc}\n")
                    nvfusion_report: ThunderFusionReport
                    for nvfusion_report in thunder_split_report.fusion_reports:
                        nvfusion_report.write_nvfuser_repro(split_folder / "nvfusion")
                        nvfusion_report.write_inductor_repro(split_folder / "nvfusion")
                        bench_data = nvfusion_report.run_benchmark()
                        print(bench_data)
                        print("---"*10)

    """
    from thunder.dynamo.utils import remove_empty_autocast, recompile_graph, get_thunder_module_names
    from thunder.dynamo.splitter import _splitter
    from thunder import jit

    # Splits the FX graph module using Thunder splitter
    gm = remove_empty_autocast(report.graph)
    # Dynamo uses lazy generation of the underlying Python code, so we need to
    # force recompilation of the GraphModule before passing it to Thunder.
    recompile_graph(gm)
    _thunder_jit = partial(jit, **thunder_options, nv_store_fusion_inputs=True)
    _, subgraph_info = _splitter(gm, _thunder_jit, torch.compile, _unused_sample_args=None)

    thunder_module_names = [f"{report.graph_name}_{name}" for name in get_thunder_module_names(subgraph_info)]
    original_modules_to_thunder_modules = (
        [m, compiled_m]
        for m, compiled_m in subgraph_info.submodule_to_compiled_functions.items()
        if compiled_m.compiler == CompilerType.THUNDER
    )
    example_inputs = subgraph_info.thunder_compiled_fns_example_inputs
    split_reason = get_split_reasons_string(subgraph_info)

    subgraph_reports = []
    for name, sub_gm_pair, example_input in zip(
        thunder_module_names, original_modules_to_thunder_modules, example_inputs
    ):
        subgraph_reports.append(
            ThunderSplitGraphReport(
                sub_gm_pair[0],
                name,
                sub_gm_pair[1].compiled_fn,
                example_input,
                thunder_options,
                split_reason,
            )
        )
    report = ThunderFXGraphReport(report.graph, report.graph_name, split_reason, subgraph_reports)
    return report
