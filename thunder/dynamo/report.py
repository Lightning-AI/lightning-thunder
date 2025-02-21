from __future__ import annotations

import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
import textwrap
import copy


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
    example_input_meta_to_input,
)

from thunder.dynamo.repro_script_template import (
    pytest_benchmark_multi_exe_code_template,
    bsym_torch_compile_repro_template,
    FXGRAPH_CLASS_NAME,
    INPUTS_NAME,
    CALLABLE_NAME,
    COMPILED_CALLABLE_NAME,
)
from thunder import last_traces, last_backward_traces
from thunder.benchmarks.targets import backward_only
from thunder.dynamo.benchmark_utils import (
    TorchCompileSpecification,
    ThunderCompileSpecification,
    TorchEagerSpecification,
    WallTime,
    KernelTime,
    check_timing,
    check_timing_bsym,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from collections.abc import Sequence

    from thunder.dynamo.utils import ExampleInputMetaData
    from thunder.core.trace import TraceCtx
    from thunder.core.symbol import BoundSymbol
    from thunder.dynamo.benchmark_utils import CompileSpecificationInterface, TimerInterface


def run_forward_backward(fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    result = sequencify(result)

    differentiable_tensor_result = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, result))

    if not differentiable_tensor_result:
        return result, None

    forward_inputs = tree_flatten((args, kwargs))[0]
    forward_inputs = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, forward_inputs))

    output_grads = []
    for diff_result in differentiable_tensor_result:
        output_grads.append(torch.ones_like(diff_result))

    for i in forward_inputs:
        i.grad = None

    torch.autograd.backward(result, output_grads, inputs=forward_inputs)
    return result, [t.grad for t in forward_inputs]


def thunderfx_pytest_benchmark_report(
    fn: Callable,
    *args,
    compile_kwargs: dict = None,
    folder_path: str | PathLike = "/tmp/thunderfx_report",
    check_consistency: bool = False,
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

    def __init__(self, graph: torch.fx.GraphModule, graph_name: str, example_input_meta: list[ExampleInputMetaData]):
        self.graph = graph
        self.graph_name = graph_name
        self.example_input_meta = example_input_meta

    def make_example_inputs(self):
        return [example_input_meta_to_input(meta) for meta in self.example_input_meta]

    def write_eager_repro(self, folder, use_benchmark: bool = False, serialize_inputs: bool = False):
        if use_benchmark:
            self.write_pytest_benchmark(
                folder,
                f"{self.graph_name}_benchmark_eager.py",
                ["eager"],
                executor_str=["None"],
                serialize_inputs=serialize_inputs,
            )
        else:
            torcheager = TorchEagerSpecification()
            self.write_repro(folder, torcheager, f"{self.graph_name}_repro_eager.py", serialize_inputs=serialize_inputs)

    def write_thunder_repro(self, folder, use_benchmark: bool = False, serialize_inputs: bool = False):
        thunder_compile_str = "thunder.jit"
        thunder_import_str = ["import thunder"]
        default_thunderjit = ThunderCompileSpecification()
        if use_benchmark:
            self.write_pytest_benchmark(
                folder,
                f"{self.graph_name}_benchmark_thunder.py",
                ["thunder"],
                [thunder_compile_str],
                thunder_import_str,
            )
        else:
            self.write_repro(
                folder, default_thunderjit, f"{self.graph_name}_repro_thunder.py", serialize_inputs=serialize_inputs
            )

    def write_inductor_repro(self, folder, use_benchmark: bool = False, serialize_inputs: bool = False):
        default_torchcompile = TorchCompileSpecification()
        inductor_compile_str = "torch.compile"
        inductor_import_str = ["import torch"]
        if use_benchmark:
            self.write_pytest_benchmark(
                folder,
                f"{self.graph_name}_benchmark_torchcompile.py",
                ["torchcompile"],
                [inductor_compile_str],
                inductor_import_str,
                serialize_inputs,
            )
        else:
            default_torchcompile = TorchCompileSpecification()
            self.write_repro(
                folder,
                default_torchcompile,
                f"{self.graph_name}_repro_torchcompile.py",
                serialize_inputs=serialize_inputs,
            )

    def _get_input_str(self, folder, inputs, serialize_inputs):
        input_str = ""
        if any(arg is None for arg in inputs):
            input_str += f"# Warning: The inputs that cannot be inferred are set to None, requiring the user to manually give inputs according to the code\n"
        if serialize_inputs:
            example_inputs = self.make_example_inputs()
            input_file_name = folder / f"{self.graph_name}_inputs.pt"
            torch.save(example_inputs, input_file_name)
            input_str += f"{INPUTS_NAME} = torch.load('{input_file_name}')\n"
        else:
            input_str = f"{INPUTS_NAME} = [\n"
            input_str += textwrap.indent("\n".join(arg_like(a) for a in inputs), "    ")
            input_str += "\n]"
        return input_str

    def write_pytest_benchmark(
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
            .. code-block:: python

            import tempfile

            import torch
            from thunder.dynamo.report import fx_report

            def model(x):
                return x * 2

            report = fx_report(model, torch.randn((2, 2), requires_grad=True, device="cuda"), compile_options={"dynamic": False})
            print(len(report.fx_graph_reports))
            graph_report = report.fx_graph_reports[0]
            my_executor = "partial(thunder.jit, transforms=[NvtxProfileTransform()], executors=[nvfuser_executor])"
            my_imports = [
                "import thunder",
                "from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform",
                "from thunder import nvfuser_executor",
                "from functools import partial",
            ]
            with tempfile.TemporaryDirectory() as tmpdir:
                graph_report.write_pytest_benchmark(
                    tmpdir,
                    f"{graph_report.graph_name}_mythunder_benchmark.py",
                    executor_name_str=["mythunder"],
                    executor_str=[my_executor],
                    import_str=my_imports,
                )
        """

        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        if inputs == None:
            inputs = self.example_input_meta
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
        code_str = pytest_benchmark_multi_exe_code_template.format(
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

    def _get_import_code(self, compile_fn: CompileSpecificationInterface = None, time_fn: TimerInterface = None):
        import_strs = [
            "\n".join(v.import_str for v in torch.fx.graph._custom_builtins.values()),
            "\n".join(compile_fn.import_str() or []) if compile_fn else "",
            "\n".join(time_fn.import_str() or []) if time_fn else "",
        ]
        return "\n".join(filter(None, import_strs))

    def _get_fx_graph_class_str(self, class_name: str = FXGRAPH_CLASS_NAME):
        return _readable(self.graph, class_name, print_output=False)

    def _get_repro_code(
        self,
        folder,
        compile_fn: CompileSpecificationInterface,
        time_fn: TimerInterface,
        serialize_inputs: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
        **kwargs,
    ):
        from thunder.dynamo.repro_script_template import repro_bench_code_template

        folder = Path(folder)
        torch_env, thunder_pkgs = get_env()
        class_str = textwrap.indent(self._get_fx_graph_class_str(), "    ")
        input_str = textwrap.indent(self._get_input_str(folder, inputs, serialize_inputs), "    ")
        extra_comment_str = kwargs.get("extra_comment_str") if "extra_comment_str" in kwargs else ""
        import_str = self._get_import_code(compile_fn, time_fn)
        code_str = repro_bench_code_template.format(
            torch_env=torch_env,
            thunder_pkgs=thunder_pkgs,
            import_str=import_str,
            dynamo_module=class_str,
            inputs=input_str,
            graph_name=self.graph_name,
            extra_comment_str=extra_comment_str,
        )
        return code_str

    def run_repro(
        self,
        compile_fn: CompileSpecificationInterface,
        check_consistency=False,
    ):
        compiled_model = compile_fn.compile(self.graph)
        example_inputs = self.make_example_inputs()
        result = run_forward_backward(compiled_model, *example_inputs)

        if check_consistency:
            eager_result = run_forward_backward(compiled_model, *example_inputs)
            torch.testing.assert_close(result, eager_result)
        return result

    def write_repro(
        self,
        folder: str | PathLike,
        compile_fn: CompileSpecificationInterface,
        file_name: str = None,
        check_consistency: bool = False,
        serialize_inputs: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
        **kwargs,
    ) -> None:
        """
        Generates a reproduction script for the FX graph module with the given compile specification and writes it to a file.

        Args:
            folder (str | PathLike): The target directory where the script will be saved.
            compile_fn (CompileSpecificationInterface): Specifies how the FX graph module should be compiled.
                See :class:`CompileSpecificationInterface` for details.
            file_name (str): The name of the output script file. Default is the :attr:`graph_name`
            check_consistency (bool, optional): Whether to verify the correctness of the
            compiled module by comparing its output with Torch eager mode. Defaults to False.
            serialize_inputs (bool, optional): Whether to serialize the inputs for reproducibility.
                Defaults to False. If enabled, all inputs will be saved to a file named
                "{graph_name}_input.pt".
            inputs (Sequence[torch.Tensor | ExampleInputMetaData], optional):
                The input tensors or metadata for the FX graph module. Defaults to None, in
                which case the inputs will be inferred from the placeholders in the FX graph.
            **kwargs: Additional arguments for customization.
        """
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        if inputs == None:
            inputs = self.example_input_meta
        code_str = self._get_repro_code(folder, compile_fn, None, serialize_inputs, inputs, **kwargs)
        compile_str = compile_fn.to_source(CALLABLE_NAME)
        code_str += textwrap.indent(f"{COMPILED_CALLABLE_NAME} = {compile_str}\n", "    ")

        run_str = f"from thunder.dynamo.report import run_forward_backward\nfwd_result, grads = run_forward_backward({COMPILED_CALLABLE_NAME}, *{INPUTS_NAME})\n"
        code_str += textwrap.indent(run_str, "    ")

        if check_consistency:
            check_str = f"eager_fwd_result, eager_grads = run_forward_backward({CALLABLE_NAME}, *{INPUTS_NAME})\ntorch.testing.assert_close(fwd_result, eager_fwd_result)\ntorch.testing.assert_close(grads, eager_grads)\n"

            code_str += textwrap.indent(check_str, "    ")

        code_str = f"{code_str}\ntest_{self.graph_name}()"

        if file_name is None:
            file_name = f"{self.graph_name}.py"
        with open(folder / file_name, "w") as f:
            print(code_str, file=f)

    def run_benchmark(self, compile_fn: CompileSpecificationInterface, time_fn: TimerInterface):
        compiled_fn = compile_fn.compile(self.graph)
        example_inputs = self.make_example_inputs()
        forward_only = not any(hasattr(arg, "requires_grad") and arg.requires_grad for arg in example_inputs)
        fwd_measurement = time_fn.time(
            "compiled_fn(*example_inputs)", globals={"compiled_fn": compiled_fn, "example_inputs": example_inputs}
        )
        bwd_measurement = None
        if not forward_only:
            backward_fn, backward_setup = backward_only(
                compiled_fn, *example_inputs, setup_graph_on_each_invocation=True
            )
            bwd_measurement = time_fn.time(
                "backward_fn(*backward_args)",
                setup="backward_args=backward_setup()",
                globals={"backward_fn": backward_fn, "backward_setup": backward_setup},
            )
        return fwd_measurement, bwd_measurement

    def write_benchmark(
        self,
        folder: str | PathLike,
        compile_fn: CompileSpecificationInterface,
        time_fn: TimerInterface,
        file_name: str = None,
        serialize_inputs: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
        **kwargs,
    ):
        """
        Generates a benchmark reproduction script for the given compilation and timing specification and writes it to the specified file.

        Args:
            folder (str | PathLike): The target directory where the script will be saved.
            compile_fn (CompileSpecificationInterface): Specifies how the FX graph module should be compiled.
                See :class:`CompileSpecificationInterface` for details.
            time_fn(TimerInterface): Specifies how the compiled callable is timed. See :class:`TimerInterface` for details.
            file_name (str): The name of the output script file. Default is the :attr:`graph_name`
            serialize_inputs (bool, optional): Whether to serialize the inputs for reproducibility. Defaults to False.
                If enabled, all inputs will be serialized into a single file: "{graph_name}_input.pt".
            inputs (Sequence[torch.Tensor | ExampleInputMetaData], optional):
                The input tensors or metadata for the FX graph module. Defaults to None.
                If not provided, inputs will be inferred from the placeholders in the FX graph.
            **kwargs: Additional arguments for customization.
        """
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        if inputs == None:
            inputs = self.example_input_meta
        forward_only = not any(hasattr(arg, "requires_grad") and arg.requires_grad for arg in inputs)
        code_str = self._get_repro_code(folder, compile_fn, time_fn, serialize_inputs, inputs, **kwargs)
        compile_str = compile_fn.to_source(CALLABLE_NAME)
        fwd_timing_str = time_fn.to_source(COMPILED_CALLABLE_NAME, INPUTS_NAME)
        bwd_timing_str = time_fn.to_source("backward_fn", "backward_args")
        code_str = f"""{code_str}
    {COMPILED_CALLABLE_NAME} = {compile_str}
    # forward
    fwd_measurement = {fwd_timing_str}
    print("fwd_measurement=", fwd_measurement)
"""
        if not forward_only:
            code_str = f"""{code_str}
    # backward
    from thunder.benchmarks.targets import backward_only
    backward_fn, backward_setup = backward_only({COMPILED_CALLABLE_NAME}, *{INPUTS_NAME})
    backward_args = backward_setup()
    bwd_measurement = {bwd_timing_str}
    print("bwd_measurement=", bwd_measurement)
"""

        code_str += f"test_{self.graph_name}()"
        if file_name is None:
            file_name = f"{self.graph_name}.py"
        with open(folder / file_name, "w") as f:
            print(code_str, file=f)


class FXReport:
    """
    This class stores a list of FXGraphReport instances, each of which wraps an FX graph
    module and provides methods to generate reproduction and benchmark scripts.
    """

    def __init__(self, graphs: list[torch.fx.GraphModule], graph_names: list[str] = None):
        self.fx_graph_reports = []
        if graph_names is None:
            graph_names = [f"graph{idx}" for idx in range(len(graphs))]

        for g_name, g in zip(graph_names, graphs):
            placeholders = list(n for n in g.graph.nodes if n.op == "placeholder")
            example_input_metadata = list(
                map(partial(_get_example_inputs_from_placeholder, only_metadata=True), placeholders)
            )
            self.fx_graph_reports.append(FXGraphReport(g, g_name, example_input_metadata))

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
        from thunder import nvfuser_executor, sdpa_executor
        from thunder.dynamo.benchmark_utils import ThunderCompileSpecification, WallTime
        from thunder.dynamo.report import fx_report

        def model(x):
            return x * 2

        report = fx_report(model, torch.randn((2, 2), requires_grad=True, device="cuda"), compile_options={"dynamic": False})
        print(len(report.fx_graph_reports))
        with tempfile.TemporaryDirectory() as tmpdir:
            for graph_report in report.fx_graph_reports:
                graph_report.write_eager_repro(tmpdir)
                # Uses default `thunder.jit`
                graph_report.write_thunder_repro(tmpdir, use_benchmark=True)
                # Uses default `torch.compile`
                graph_report.write_inductor_repro(tmpdir)

                my_thunderjit = ThunderCompileSpecification(executors=[sdpa_executor, nvfuser_executor])
                graph_name = graph_report.graph_name
                graph_report.write_repro(
                    tmpdir, my_thunderjit, check_consistency=True, file_name=f"{graph_name}_mythunder_repro.py"
                )
                graph_report.write_benchmark(tmpdir, my_thunderjit, WallTime, f"{graph_name}_mythunder_benchmark.py")
    """
    graphs = []

    def helper_backend(gm, example_inputs):
        """Helper function to collect FX graphs."""
        graphs.append(copy.deepcopy(gm))
        from torch._inductor import compile

        return compile(gm, example_inputs)

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
        example_input_metas,
        compiled_fn: Callable,
        thunder_options: dict,
        split_reason: str,
    ):
        super().__init__(graph, graph_name, example_input_metas)
        self.compiled_fn = compiled_fn
        self.thunder_options = thunder_options
        self.split_reason = split_reason

        self.fusion_reports: list[ThunderFusionReport] = []
        self.fwd_trc: TraceCtx = None
        self.bwd_trc: TraceCtx = None

    def __repr__(self):
        return f"<ThunderSplitGraphReport with {len(self.fusion_reports)} ThunderFusionReport accessible via .fusion_reports>"

    def _create_thunder_traces(self):
        example_inputs = self.make_example_inputs()
        # Executes to get the trace
        run_forward_backward(self.compiled_fn, *example_inputs)
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
        has_cuda_args = any(hasattr(arg, "device") and arg.device.type == "cuda" for arg in self.example_input_meta)
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
            default_thunderjit = ThunderCompileSpecification()
            self.write_repro(
                folder,
                default_thunderjit,
                f"{self.graph_name}_repro_thunder.py",
                serialize_inputs=serialize_inputs,
                extra_comment_str=self.split_reason,
            )
            return

        executor_names_list = ["thunder"]
        executors = [thunder_ex_str]

        self.write_pytest_benchmark(
            folder,
            f"{self.graph_name}_benchmark_thunder.py",
            executor_names_list,
            executor_str=executors,
            import_str=import_str,
            serialize_inputs=serialize_inputs,
            extra_comment_str=self.split_reason,
        )


# TODO: `ThunderFusionReport` is expected to inherit from `FXGraphReport` for consistency.
# However, we currently cannot convert bound symbols to an FX graph.
# In the future, we might add support for this conversion, making `ThunderFusionReport`
# consistent with `FXGraphReport`.
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

    def run_benchmark(self, compile_fn: CompileSpecificationInterface, timer_fn: TimerInterface):
        compiled_fn = compile_fn.compile(self.nvfusion_bsym)
        inputs = self.make_example_inputs()
        return timer_fn.time("compiled_fn(*inputs)", globals={"compiled_fn": compiled_fn, "inputs": inputs})

    def run_repro(
        self,
        compile_fn: CompileSpecificationInterface,
    ):
        compiled_fn = compile_fn.compile(self.nvfusion_bsym)
        inputs = self.make_example_inputs()
        return compiled_fn(*inputs)

    def _get_nvfuser_code(self):
        nvfuser_callable = self.nvfusion_bsym._call_ctx[self.nvfusion_bsym.sym.name]
        fd = nvfuser_callable.last_used

        # The API for nvFuser version >=2.14
        get_repro = getattr(fd, "repro_script_for", None)
        # The legacy nvFuser API
        if get_repro is None:
            get_repro = getattr(fd, "getReproString", None)
        if get_repro is None:
            raise RuntimeError("The installed version of nvFuser does not support repro generation unless on crash.")

        inputs = self.make_example_inputs()
        nvfuser_repro_code = get_repro(inputs)
        return nvfuser_repro_code

    def write_nvfuser_benchmark(self, folder, time_fn: TimerInterface, file_name=None, **kwargs):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        extra_comment_str = kwargs.get("extra_comment_str") if "extra_comment_str" in kwargs else ""
        repro_code_str = self._get_nvfuser_code()
        timing_import_str = "\n".join(time_fn.import_str() or [])
        timing_str = time_fn.to_source("nvfuser_fn", "inputs")
        timing_str = timing_str.replace("*inputs", "inputs")
        repro_code_str = repro_code_str.replace("fd.execute(inputs)\n", "")
        comment_str = f'"""\n{self.nvfusion_bsym}\n\n{extra_comment_str}"""'
        code_str = f"""{comment_str}
{timing_import_str}
{repro_code_str}
nvfuser_fn = fd.execute
measurement = {timing_str}
print(measurement)
"""
        if file_name == None:
            file_name = f"{self.name}_benchmark_nvfuser.py"
        with open(folder / file_name, "w") as f:
            print(code_str, file=f)

    def write_nvfuser_repro(self, folder, file_name=None):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        repro_code_str = self._get_nvfuser_code()
        comment_str = f'"""\n{self.nvfusion_bsym}\n"""'

        if file_name == None:
            file_name = f"{self.name}_repro_nvfuser.py"
        with open(folder / file_name, "w") as f:
            print(comment_str, file=f)
            print(repro_code_str, file=f)

    def make_example_inputs(self):
        return [example_input_meta_to_input(meta) for meta in self.get_inputs_meta()]

    def get_inputs_meta(self):
        return self.nvfusion_bsym._call_ctx[self.nvfusion_bsym.sym.name].last_inputs_meta

    def _get_inductor_code(self, **kwargs):
        python_func = create_python_callable_from_bsym(self.nvfusion_bsym)
        nvfusion_name = self.nvfusion_bsym.sym.name
        extra_comment_str = kwargs.get("extra_comment_str") if "extra_comment_str" in kwargs else ""

        inputs = self.get_inputs_meta()
        inputs = "[" + "".join(arg_like(inp) for inp in inputs) + "]"
        inductor_code_str = bsym_torch_compile_repro_template.format(
            python_func=python_func, func_name=nvfusion_name, inputs=inputs, extra_comment_str=extra_comment_str
        )
        return inductor_code_str

    def write_inductor_repro(self, folder: PathLike, file_name=None):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        code_str = self._get_inductor_code()
        code_str = f"""{code_str}
out = torch_compiled_callable(*inputs)
"""
        if file_name == None:
            file_name = f"{self.name}_repro_inductor.py"
        with open(folder / file_name, "w") as f:
            f.write(code_str)

    def write_inductor_benchmark(self, folder: PathLike, time_fn: TimerInterface, file_name=None, **kwargs):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        code_str = self._get_inductor_code(**kwargs)
        timing_import_str = "\n".join(time_fn.import_str() or [])
        code_str = f"""{code_str}
{timing_import_str}
measurement = {time_fn.to_source("torch_compiled_callable", "inputs")}
print(measurement)
"""
        if file_name == None:
            file_name = f"{self.name}_benchmark_inductor.py"
        with open(folder / file_name, "w") as f:
            f.write(code_str)


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
        self,
        gm: torch.fx.GraphModule,
        gm_name: str,
        example_input_metas,
        split_reason: str,
        subgraph_reports: list[ThunderSplitGraphReport],
    ):
        super().__init__(gm, gm_name, example_input_metas)
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

    """
    from thunder.dynamo.utils import remove_empty_autocast, recompile_graph, get_thunder_module_names
    from thunder.dynamo.splitter import _splitter
    from thunder import jit

    # Splits the FX graph module using Thunder splitter
    gm = remove_empty_autocast(report.graph)
    # Dynamo uses lazy generation of the underlying Python code, so we need to
    # force recompilation of the GraphModule before passing it to Thunder.
    recompile_graph(gm)
    thunder_jit = partial(jit, **thunder_options, nv_store_fusion_inputs_meta=True)
    _, subgraph_info = _splitter(gm, thunder_jit, torch.compile, _unused_sample_args=None)

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
                example_input,
                sub_gm_pair[1].compiled_fn,
                thunder_options,
                split_reason,
            )
        )
    report = ThunderFXGraphReport(
        report.graph, report.graph_name, report.example_input_meta, split_reason, subgraph_reports
    )
    return report


def get_thunder_split_reports(fn: Callable, *args, thunder_compile_kwargs: dict = None, **kwargs):
    reports = fx_report(fn, *args, **kwargs)
    if thunder_compile_kwargs is None:
        thunder_compile_kwargs = {}
    split_reports = []
    for fx_graph_report in reports.fx_graph_reports:
        thunder_fx_graph_report = analyze_thunder_splits(fx_graph_report, **thunder_compile_kwargs)
        print(f"\n{thunder_fx_graph_report.graph_name}: {thunder_fx_graph_report.split_reason}\n")
        for split_report in thunder_fx_graph_report.subgraph_reports:
            split_reports.append(split_report)
    return split_reports


def get_nvfusion_reports(split_reports: list[ThunderSplitGraphReport]):
    nvfusion_reports = []
    for split_report in split_reports:
        split_report.create_fusion_reports()
        for nvfusion_report in split_report.fusion_reports:
            nvfusion_reports.append(nvfusion_report)
    return nvfusion_reports


def thunderfx_benchmark_report(
    fn: Callable,
    *args,
    folder_path: str | PathLike,
    thunder_compile_kwargs: dict = None,
    compare_fusion: bool = False,
    rtol=0.5,
    atol=0.0,
    **kwargs,
):
    """
    A utility function that analyzes the runnability and performance benchmarks of each FX graph.

    1. Checks if the callable can be executed with `torch.compile`.
    - If it fails, attempts to run it eagerly.
    - If eager execution also fails, an error is printed, and the function returns.
    - If eager execution succeeds, the analysis continues.

    2. Collects all ThunderFX subgraphs and verifies whether each subgraph can be successfully executed by Thunder.

    3. For each subgraph:
    - Compares wall time and kernel time between `torch.compile` and Thunder.
    - Reports performance metrics and saves the benchmark script in `folder_path` if the difference exceeds the tolerance (`rtol`, `atol` in seconds).
    - Uses `math.isclose` for tolerance checks.

    4. If `compare_fusion` is `True`:
    - Also compares the wall time and kernel time of each nvFusion region.
    - Saves the benchmark script when necessary, following the same criteria as above.

    Note:
    - This function may run out of memory (OOM) as it allocates random tensors when executing
    the graph module in each Report. To prevent OOM issues, users must manually free the
    input model and arguments to free up memory for `get_nvfusion_reports`, `check_timing`,
    and `check_timing_bsym`.

    Here is an example:

    ```python
    split_reports = get_thunder_split_reports(model, x)

    # Frees the parameters and inputs to make room for the reports
    del model
    del x

    # Running the model generates the NVFusion symbol, which requires additional memory for the input.
    # To free up space before generating the NVFusion reports, both the model and input are deleted.
    nvfusion_reports = get_nvfusion_reports(split_reports)

    check_timing(folder_path, split_reports[0], torchcompile, thunderjit_specification, WallTime, "walltime", rtol, atol)
    check_timing_bsym(folder_path, nvfusion_reports[0], KernelTime, "kerneltime", rtol, atol)
    ```
    """
    try:
        torch_compiled = torch.compile(fn)
        torch_compiled(*args, **kwargs)
    except Exception as e:
        print(f"Failed to run the function using torch.compile with exception: {e}")
        print(f"Trying with Torch eager...")
        try:
            run_forward_backward(fn, *args, **kwargs)
        except Exception as e:
            print(f"Failed to run the function with exception: {e}")
            return
        print("The input callable can be successfully executed.")
    else:
        print("The input callable can be successfully executed by torch.compile.")

    if thunder_compile_kwargs is None:
        thunder_compile_kwargs = {}
    split_reports = get_thunder_split_reports(fn, *args, **kwargs, thunder_compile_kwargs=thunder_compile_kwargs)
    thunderjit_specification = ThunderCompileSpecification(**thunder_compile_kwargs)
    torchcompile = TorchCompileSpecification()
    for split_report in split_reports:
        try:
            split_report.run_repro(thunderjit_specification)
        except Exception as e:
            print(f"Failed to run {split_report.graph_name} using Thunder with exception: {e}\n")
            continue
        print(f"{split_report.graph_name} can be successfully executed by Thunder\n")
        check_timing(
            folder_path, split_report, torchcompile, thunderjit_specification, WallTime, "walltime", rtol, atol
        )
        check_timing(
            folder_path, split_report, torchcompile, thunderjit_specification, KernelTime, "kerneltime", rtol, atol
        )

    if not compare_fusion:
        return
    nvfusion_reports = get_nvfusion_reports(split_reports)

    for nvfusion_report in nvfusion_reports:
        check_timing_bsym(folder_path, nvfusion_report, WallTime, "walltime", rtol, atol)
        check_timing_bsym(folder_path, nvfusion_report, KernelTime, "kerneltime", rtol, atol)
