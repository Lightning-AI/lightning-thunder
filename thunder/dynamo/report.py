from __future__ import annotations

import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
import textwrap
import copy
from itertools import chain
from looseversion import LooseVersion
import shutil

import torch
from thunder.core.pytree import tree_flatten
from thunder.core.utils import sequencify, create_python_callable_from_bsym
from thunder.dynamo.compiler import thunderfx, ThunderCompiler
from thunder.dynamo.utils import (
    _get_example_inputs_from_placeholder,
    _readable,
    arg_like,
    get_env,
    get_split_reasons_string,
    CompilerType,
    recompile_graph,
    has_higher_order_operator,
    input_to_example_input_meta,
    example_input_meta_to_input,
    format_python_file,
)

from thunder.dynamo.repro_script_template import (
    pytest_benchmark_multi_exe_code_template,
    bsym_torch_compile_repro_template,
    main_code,
    comment_str_template,
    FXGRAPH_CLASS_NAME,
    INPUTS_NAME,
    CALLABLE_NAME,
    COMPILED_CALLABLE_NAME,
)
from thunder import last_traces, last_backward_traces, compile_stats
from thunder.benchmarks.utils import backward_only
from thunder.dynamo.benchmark_utils import (
    TorchCompileSpecification,
    ThunderCompileSpecification,
    TorchEagerSpecification,
    TorchInductorSpecification,
    WallTime,
    KernelTime,
    WallTimeWithMemoryUsage,
    check_metrics,
    check_nvfusion_timing,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike
    from collections.abc import Sequence
    from typing import TextIO

    from torch._dynamo.output_graph import GraphCompileReason
    from thunder.dynamo.utils import ExampleInputMetaData
    from thunder.core.trace import TraceCtx
    from thunder.core.symbol import BoundSymbol
    from thunder.dynamo.benchmark_utils import CompileSpecificationInterface, TimerInterface


def run_forward_backward(fn, *args, benchmark=False, **kwargs):
    result = fn(*args, **kwargs)
    result = sequencify(result)

    differentiable_tensor_result = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, result))

    if not differentiable_tensor_result:
        if benchmark:
            return
        return result, None

    forward_inputs = tree_flatten((args, kwargs))[0]
    inputs_requires_grad = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, forward_inputs))
    if isinstance(fn, torch.nn.Module):
        params_requires_grad = list(
            f for f in chain(fn.parameters(), fn.buffers()) if isinstance(f, torch.Tensor) and f.requires_grad
        )
        inputs_requires_grad = inputs_requires_grad + params_requires_grad

    output_grads = []
    for diff_result in differentiable_tensor_result:
        output_grads.append(torch.ones_like(diff_result))

    for i in inputs_requires_grad:
        i.grad = None

    torch.autograd.backward(differentiable_tensor_result, output_grads, inputs=inputs_requires_grad)
    if benchmark:
        for i in inputs_requires_grad:
            i.grad = None
        return
    return result, [t.grad for t in inputs_requires_grad]


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
                    print(
                        f"{bk['name'].lstrip('test_')}: {bk['extra_info'].get('max_allocated_memory_MB', 0) / 1000} GB"
                    )
                print("\n")

            print_sorted_memory_info("forward")
            print_sorted_memory_info("backward")


class FXGraphReport:
    """
    Encapsulates an FX Graph module with metadata to aid in generating
    reproduction and benchmarking scripts for various executors.
    """

    def __init__(self, graph: torch.fx.GraphModule, graph_name: str, example_input_meta: list[ExampleInputMetaData]):
        if LooseVersion(torch.__version__) < LooseVersion("2.6.0"):
            # NOTE: PyTorch 2.6 changes the structure of GraphModule for higher order ops.
            # In newer torch version the higher order ops are nested as submodules within the module that uses them,
            # but in the older version they were separate sibling modules.
            if has_higher_order_operator(graph):
                raise RuntimeError(
                    "The Reporting Tool for Torch higher-order operators is supported only in PyTorch version 2.6 or later."
                )
        self.graph = graph
        self.graph_name = graph_name
        self.example_input_meta = example_input_meta
        self.ops = {node.target for node in self.graph.graph.nodes if node.op in ("call_function", "call_method")}

    def __str__(self):
        output = f"Graph Name: {self.graph_name}\n"
        output += "  Operators used in the graph:\n"
        for op in self.ops:
            output += f"    {op}\n"
        return output

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
            self.write_repro(
                folder, torcheager, file_name=f"{self.graph_name}_repro_eager.py", serialize_inputs=serialize_inputs
            )

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
                folder,
                default_thunderjit,
                file_name=f"{self.graph_name}_repro_thunder.py",
                serialize_inputs=serialize_inputs,
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
                file_name=f"{self.graph_name}_repro_torchcompile.py",
                serialize_inputs=serialize_inputs,
            )

    def _get_input_str(self, folder, inputs, serialize_inputs):
        input_str = ""
        if any(arg is None for arg in inputs):
            input_str += "# Warning: The inputs that cannot be inferred are set to None, requiring the user to manually give inputs according to the code\n"
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

            report = fx_report(model, dynamic=False)(torch.randn((2, 2), requires_grad=True, device="cuda"))
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
        readable = _readable(self.graph, "DynamoModule", print_output=False)
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
        return "\n".join(filter(None, import_strs)).lstrip("\n")

    def _get_fx_graph_class_str(self, class_name: str = FXGRAPH_CLASS_NAME):
        return _readable(self.graph, class_name, print_output=False)

    def _get_comment_str(self, extra_comment_str):
        torch_env, thunder_pkgs = get_env()

        comment_str = comment_str_template.format(
            torch_env=torch_env,
            thunder_pkgs=thunder_pkgs,
            extra_comment_str=extra_comment_str,
        )
        return comment_str

    def _get_repro_code(
        self,
        folder,
        compile_fn: CompileSpecificationInterface,
        time_fn: TimerInterface,
        serialize_inputs: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
    ):
        from thunder.dynamo.repro_script_template import repro_bench_code_template

        folder = Path(folder)
        class_str = self._get_fx_graph_class_str()
        input_str = textwrap.indent(self._get_input_str(folder, inputs, serialize_inputs), "    ")
        import_str = self._get_import_code(compile_fn, time_fn)
        code_str = repro_bench_code_template.format(
            import_str=import_str,
            dynamo_module=class_str,
            inputs=input_str,
            graph_name=self.graph_name,
        )
        return code_str

    def run_repro(
        self,
        compile_fn: CompileSpecificationInterface,
        check_consistency=False,
    ):
        torch._dynamo.reset()
        example_inputs = self.make_example_inputs()
        # ref: https://github.com/pytorch/pytorch/blob/0ef5ba43a6e7fe806ea9f27929bf4328ffd1ebf4/torch/_inductor/compile_fx.py#L1921-L1922
        # The compile_fn may mutate the GraphModule, so we need to deepcopy it
        graph = copy.deepcopy(self.graph)
        # To avoid the AssertionError: attribute nodes of Graph object out of sync
        recompile_graph(graph)
        compiled_model = compile_fn.compile(graph, inputs=example_inputs)
        result = run_forward_backward(compiled_model, *example_inputs)

        if check_consistency:
            eager_result = run_forward_backward(graph, *example_inputs)
            torch.testing.assert_close(result, eager_result)
        return result

    def write_repro(
        self,
        folder: str | PathLike,
        compile_fn: CompileSpecificationInterface,
        *,
        file_name: str = None,
        check_consistency: bool = False,
        serialize_inputs: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
        extra_comment_str: str = "",
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
        code_str = self._get_repro_code(folder, compile_fn, None, serialize_inputs, inputs)
        comment_str = self._get_comment_str(extra_comment_str)
        compile_str = compile_fn.to_source(CALLABLE_NAME)
        code_str += textwrap.indent(f"{COMPILED_CALLABLE_NAME} = {compile_str}\n", "    ")

        run_str = f"from thunder.dynamo.report import run_forward_backward\nfwd_result, grads = run_forward_backward({COMPILED_CALLABLE_NAME}, *{INPUTS_NAME})\n"
        code_str += textwrap.indent(run_str, "    ")

        if check_consistency:
            check_str = f"eager_fwd_result, eager_grads = run_forward_backward({CALLABLE_NAME}, *{INPUTS_NAME})\ntorch.testing.assert_close(fwd_result, eager_fwd_result)\ntorch.testing.assert_close(grads, eager_grads)\n"

            code_str += textwrap.indent(check_str, "    ")

        code_str = f"{code_str}\n{main_code.format(graph_name=self.graph_name)}\n{comment_str}"

        if file_name is None:
            file_name = f"{self.graph_name}.py"
        with open(folder / file_name, "w") as f:
            print(code_str, file=f)
        format_python_file(folder / file_name)

    def run_benchmark(
        self,
        compile_fn: CompileSpecificationInterface,
        time_fn: TimerInterface,
        *,
        reset_torch_dynamo=True,
        example_inputs=None,
        measure_fwd_bwd_together=False,
    ):
        # From torch.compile docs - https://pytorch.org/docs/stable/generated/torch.compile.html
        # > Multiple compiled results can be associated with a frame up to torch._dynamo.config.cache_size_limit, which defaults to 8; at which point we will fall back to eager.
        # Ref: https://github.com/pytorch/pytorch/blob/34d726011f482b716d879bf665aef100a7c08a8d/torch/_dynamo/__init__.py#L97
        # > reset function clears all compile caches and restore initial state.
        # Sets the `reset_torch_dynamo` to True when you need to reset Dynamo's state *as if* you had started a fresh process invocation.
        if reset_torch_dynamo:
            torch._dynamo.reset()
        if example_inputs is None:
            example_inputs = self.make_example_inputs()
        # ref: https://github.com/pytorch/pytorch/blob/0ef5ba43a6e7fe806ea9f27929bf4328ffd1ebf4/torch/_inductor/compile_fx.py#L1921-L1922
        # The compile_fn may mutate the GraphModule, so we need to deepcopy it
        graph = copy.deepcopy(self.graph)
        # To avoid the AssertionError: attribute nodes of Graph object out of sync
        recompile_graph(graph)
        compiled_fn = compile_fn.compile(graph, inputs=example_inputs)

        if measure_fwd_bwd_together:
            fwd_bwd_measurement = time_fn.time(
                "run_forward_backward(compiled_fn, *example_inputs, benchmark=True)",
                globals={
                    "run_forward_backward": run_forward_backward,
                    "compiled_fn": compiled_fn,
                    "example_inputs": example_inputs,
                },
            )
            return compiled_fn, fwd_bwd_measurement, None
        forward_only = not any(hasattr(arg, "requires_grad") and arg.requires_grad for arg in example_inputs)
        fwd_measurement = time_fn.time(
            "compiled_fn(*example_inputs)", globals={"compiled_fn": compiled_fn, "example_inputs": example_inputs}
        )
        bwd_measurement = None
        if not forward_only:
            backward_fn, backward_setup = backward_only(compiled_fn, *example_inputs)
            backward_args = backward_setup()
            bwd_measurement = time_fn.time(
                "backward_fn(*backward_args)", globals={"backward_fn": backward_fn, "backward_args": backward_args}
            )
        return compiled_fn, fwd_measurement, bwd_measurement

    def write_benchmark(
        self,
        folder: str | PathLike,
        compile_fn: CompileSpecificationInterface,
        time_fn: TimerInterface,
        *,
        file_name: str = None,
        serialize_inputs: bool = False,
        inputs: Sequence[torch.Tensor | ExampleInputMetaData] = None,
        extra_comment_str: str = "",
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
        code_str = self._get_repro_code(folder, compile_fn, time_fn, serialize_inputs, inputs)
        comment_str = self._get_comment_str(extra_comment_str)
        compile_str = compile_fn.to_source(CALLABLE_NAME)
        fwd_timing_str = time_fn.to_source(COMPILED_CALLABLE_NAME, INPUTS_NAME)
        bwd_timing_str = time_fn.to_source("backward_fn", "backward_args")
        code_str = f"""{code_str}
    {COMPILED_CALLABLE_NAME} = {compile_str}
    # forward
    fwd_measurement = {fwd_timing_str}
    print("fwd_measurement=", fwd_measurement)
    if hasattr(fwd_measurement, "max_allocated_memory"):
        from thunder.dynamo.benchmark_utils import get_pretty_memory_str
        print(f"fwd_measurement.max_allocated_memory={{get_pretty_memory_str(fwd_measurement.max_allocated_memory)}}")
"""
        if not forward_only:
            code_str = f"""{code_str}
    # backward
    from thunder.benchmarks.utils import backward_only
    backward_fn, backward_setup = backward_only({COMPILED_CALLABLE_NAME}, *{INPUTS_NAME})
    backward_args = backward_setup()
    bwd_measurement = {bwd_timing_str}
    print("bwd_measurement=", bwd_measurement)
    if hasattr(bwd_measurement, "max_allocated_memory"):
        print(f"bwd_measurement.max_allocated_memory={{get_pretty_memory_str(bwd_measurement.max_allocated_memory)}}")
"""

        code_str = f"{code_str}\n{main_code.format(graph_name=self.graph_name)}\n{comment_str}"
        if file_name is None:
            file_name = f"{self.graph_name}.py"
        with open(folder / file_name, "w") as f:
            print(code_str, file=f)
        format_python_file(folder / file_name)


class FXReport:
    """
    This class stores a list of FXGraphReport instances, each of which wraps an FX graph
    module and provides methods to generate reproduction and benchmark scripts.
    """

    def __init__(
        self,
        graphs: list[torch.fx.GraphModule],
        graph_names: list[str] | None = None,
        dynamo_break_reasons: list[GraphCompileReason] | None = None,
    ):
        self.fx_graph_reports = []
        self.dynamo_break_reasons = dynamo_break_reasons
        if graph_names is None:
            graph_names = [f"graph{idx}" for idx in range(len(graphs))]

        for g_name, g in zip(graph_names, graphs):
            placeholders = list(n for n in g.graph.nodes if n.op == "placeholder")
            example_input_metadata = list(
                map(partial(_get_example_inputs_from_placeholder, only_metadata=True), placeholders)
            )
            self.fx_graph_reports.append(FXGraphReport(g, g_name, example_input_metadata))

    def __str__(self):
        output = f"Dynamo Graph Count: {len(self.fx_graph_reports)}\n"
        if self.dynamo_break_reasons:
            output += "Dynamo Break Reasons:\n"
            for idx, reason in enumerate(self.dynamo_break_reasons):
                output += f"  Break Reason {idx + 1}:\n"
                output += f"    Reason: {reason.reason}\n"
                output += "    User Stack:\n"
                for frame_summary in reason.user_stack:
                    output += f"      {frame_summary}\n"
        output += "Graph information:\n"
        for idx, graph_report in enumerate(self.fx_graph_reports):
            output += textwrap.indent(f"{graph_report}\n", "  ")
        return output


def fx_report(fn: Callable, **torch_compile_kwargs) -> Callable[..., FXReport]:
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

        report = fx_report(model, dynamic=False)(torch.randn((2, 2), requires_grad=True, device="cuda"))
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
                graph_report.write_benchmark(tmpdir, my_thunderjit, WallTime, file_name=f"{graph_name}_mythunder_benchmark.py")
    """
    if compile_stats(fn) is not None:
        raise ValueError(
            "fx_report requires the original (uncompiled) callable and cannot be used on the Thunder-compiled function."
        )
    graphs = []
    break_reasons = []

    def helper_backend(gm, example_inputs):
        """Helper function to collect FX graphs."""
        graphs.append(copy.deepcopy(gm))
        if gm.compile_subgraph_reason.graph_break:
            break_reasons.append(gm.compile_subgraph_reason)

        from torch._inductor import compile

        return compile(gm, example_inputs)

    torch._dynamo.reset()
    compiled = torch.compile(fn, **torch_compile_kwargs, backend=helper_backend)

    def inner_fn(*args, **kwargs):
        compiled(*args, **kwargs)
        return FXReport(graphs, dynamo_break_reasons=break_reasons)

    return inner_fn


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
        self.fwd_trc: TraceCtx | None = None
        self.bwd_trc: TraceCtx | None = None

    def _create_thunder_traces(self):
        example_inputs = self.make_example_inputs()
        # Executes to get the trace
        _, grads = run_forward_backward(self.compiled_fn, *example_inputs)
        self.fwd_trc = last_traces(self.compiled_fn)[-1]
        if grads and (backward_traces := last_backward_traces(self.compiled_fn)):
            self.bwd_trc = backward_traces[-1]

    def create_fusion_reports(self):
        """
        Runs the Thunder-compiled function to obtain the nvFusion definition
        and generate the :class:`ThunderFusionReport` instance based on it.
        """
        self._create_thunder_traces()

        for trace, prefix in [(self.fwd_trc, "forward"), (self.bwd_trc, "backward")]:
            # `self.bwd_trc` can be None.
            if trace is None:
                continue
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
                file_name=f"{self.graph_name}_repro_thunder.py",
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

    def __str__(self):
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

    def write_nvfuser_benchmark(self, folder, time_fn: TimerInterface, file_name=None, extra_comment_str=""):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        repro_code_str = self._get_nvfuser_code()
        timing_import_str = "\n".join(time_fn.import_str() or [])
        timing_str = time_fn.to_source("nvfuser_fn", "inputs")
        timing_str = timing_str.replace("*inputs", "inputs")
        repro_code_str = repro_code_str.replace("fd.execute(inputs)\n", "")
        comment_str = f'"""\n{self.nvfusion_bsym}\n\n{extra_comment_str}"""'
        code_str = f"""{timing_import_str}
{repro_code_str}
nvfuser_fn = fd.execute
measurement = {timing_str}
print(measurement)
{comment_str}
"""
        if file_name == None:
            file_name = f"{self.name}_benchmark_nvfuser.py"
        with open(folder / file_name, "w") as f:
            print(code_str, file=f)
        format_python_file(folder / file_name)

    def write_nvfuser_repro(self, folder, file_name=None):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        repro_code_str = self._get_nvfuser_code()
        comment_str = f'"""\n{self.nvfusion_bsym}\n"""'
        repro_code_str = f"{repro_code_str}\n{comment_str}"

        if file_name == None:
            file_name = f"{self.name}_repro_nvfuser.py"
        with open(folder / file_name, "w") as f:
            print(repro_code_str, file=f)
        format_python_file(folder / file_name)

    def make_example_inputs(self):
        return example_input_meta_to_input(input_to_example_input_meta(self.get_fake_inputs()))

    def get_fake_inputs(self):
        return self.nvfusion_bsym._call_ctx[self.nvfusion_bsym.sym.name].last_used.fake_inputs

    def _get_inductor_code(self, extra_comment_str=""):
        python_func = create_python_callable_from_bsym(self.nvfusion_bsym)
        nvfusion_name = self.nvfusion_bsym.sym.name

        inputs = self.get_fake_inputs()
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
        format_python_file(folder / file_name)

    def write_inductor_benchmark(self, folder: PathLike, time_fn: TimerInterface, file_name=None, extra_comment_str=""):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        code_str = self._get_inductor_code(extra_comment_str)
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
        format_python_file(folder / file_name)


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

    def __str__(self):
        output = f"Thunder-specific Information of {self.graph_name}:\n"
        output += textwrap.indent(self.split_reason, "  ")
        for report in self.subgraph_reports:
            output += textwrap.indent(report.__str__(), "  ")
        return output


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
        results: FXReport = fx_report(foo)(x)

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
    from thunder.dynamo.utils import remove_empty_autocast, get_thunder_module_names
    from thunder.dynamo.splitter import _splitter
    from thunder import jit

    # Splits the FX graph module using Thunder splitter
    gm = remove_empty_autocast(report.graph)
    # Dynamo uses lazy generation of the underlying Python code, so we need to
    # force recompilation of the GraphModule before passing it to Thunder.
    recompile_graph(gm)

    # Get the default options from thunderfx if not specified.
    for k, v in ThunderCompiler().thunder_options.items():
        if k not in thunder_options:
            thunder_options[k] = v

    thunder_jit = partial(jit, **thunder_options, nv_save_fake_inputs=True)
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


def check_torch_compile_runnability(fn: Callable, stream: TextIO = sys.stdout, **torch_compile_kwargs):
    """
    Checks if the input callable can be successfully executed by torch.compile.
    If not, it will try to run it in eager mode.
    """
    # WAR for triton error https://github.com/pytorch/pytorch/issues/124565
    if torch.cuda.is_available():
        torch.empty(1, device="cuda", requires_grad=True).backward()
    torch._dynamo.reset()
    torch_compiled = torch.compile(fn, **torch_compile_kwargs)

    def inner_fn(*args, **kwargs):
        try:
            run_forward_backward(torch_compiled, *args, **kwargs)
        except Exception as e:
            stream.write(f"Failed to run the function using torch.compile with exception: {e}")
            stream.write("Trying with Torch eager...")
            try:
                run_forward_backward(fn, *args, **kwargs)
            except Exception as e:
                stream.write(f"Failed to run the function in eager mode with exception: {e}")
                return
            stream.write("The input callable can be successfully executed in eager mode.")
        else:
            stream.write("The input callable can be successfully executed by torch.compile.")

    return inner_fn


def get_thunder_fxgraph_reports(fn: Callable, stream: TextIO = sys.stdout, **compile_kwargs):
    """
    Generates a list of :class:`ThunderFXGraphReport` objects for a given callable.

    This function performs the following steps:
    1. Checks if the callable can be executed with `torch.compile` or eager execution when `check_runnablility` is `True`.
    2. Generates the dynamo segmented FX graphs and further analyzes the Thunder-split subgraphs of the FX graph.

    Parameters:
        fn: The callable to analyze.
        stream: Stream to write output to.
        **compile_kwargs: Keyword arguments for Thunder and torch.compile.

    Returns:
        A function that takes *args, **kwargs and returns a list of ThunderFXGraphReport objects.
    """
    from thunder.dynamo.utils import get_torch_compile_kwargs

    torch_compile_kwargs = get_torch_compile_kwargs(**compile_kwargs)
    thunder_jit_kwargs = {k: v for k, v in compile_kwargs.items() if k not in torch_compile_kwargs}

    def inner_fn(*args, **kwargs):
        reports = fx_report(fn, **torch_compile_kwargs)(*args, **kwargs)
        stream.write(str(reports))
        thunder_fxgraph_reports = []
        for fxgraph_report in reports.fx_graph_reports:
            thunder_fxgraph_report = analyze_thunder_splits(fxgraph_report, **thunder_jit_kwargs)
            stream.write(str(thunder_fxgraph_report))
            thunder_fxgraph_reports.append(thunder_fxgraph_report)
        return thunder_fxgraph_reports

    return inner_fn


def make_nvfusion_reports(split_reports: list[ThunderSplitGraphReport]):
    """
    Creates nvFusion reports for each Thunder-split report.
    """
    for split_report in split_reports:
        split_report.create_fusion_reports()


# NOTE: Here we use TorchInductorSpecification (TorchInductor without TorchDynamo) to compare with Thunder. Reason: https://github.com/Lightning-AI/lightning-thunder/issues/1521
def thunderfx_benchmark_report_from_splits(
    thunder_fxgraph_reports: list[ThunderFXGraphReport],
    folder_path: str | PathLike,
    compare_fusion: bool = False,
    time_rtol=0.5,
    time_atol=0.0,
    memory_usage_rtol=0.5,
    memory_usage_atol=0.0,
    stream: TextIO = sys.stdout,
    **thunder_jit_kwargs,
):
    """
    A utility function that analyzes the runnability and performance benchmarks of each Thunder-split FX graph and nvFusion regions(optional).
    For each graph, it:
    - Saves the scripts in `folder_path/graph_name/failed` if execution fails.
    - Prints out the performance metrics and saves the benchmark script in `folder_path` if the difference exceeds the tolerance (`time_rtol`, `time_atol` in seconds; `memory_usage_rtol`, `memory_usage_atol` in Bytes).
    - In general, the function will create the following folder structure:
    folder_path
    └── graph0
        ├── failed
        │   └── failed_graph0_thunder_1_thunder_WallTime.py
        └── memory_issue
            ├── graph0_thunder_0_inductor_backend_WallTimeWithMemoryUsage.py
            └── graph0_thunder_0_thunder_WallTimeWithMemoryUsage.py
        ├── graph0_thunder_0_thunder_walltime.py
        └── nvfusion_reports
            └── graph0_thunder_0_nvfusion0_forward_nvfuser_kerneltime.py
    It separates subfolders for each `graph[index]` and `nvfusion_report`.
    The script for each FX graph is named as `graph[index]_[thunder_split_name]_[executor_name]_[timer_name].py`.
    The script for each nvfusion region is named as `graph[index]_[thunder_split_name]_[nvfusion_name]_[forward/backward]_[executor_name]_[timer_name].py`.
    """
    folder_path = Path(folder_path)
    thunderjit = ThunderCompileSpecification(**thunder_jit_kwargs)
    torchinductor = TorchInductorSpecification()

    runnable_split_reports: list[ThunderSplitGraphReport] = []
    for thunder_fxgraph_report in thunder_fxgraph_reports:
        graph_folder = folder_path / thunder_fxgraph_report.graph_name
        for split_report in thunder_fxgraph_report.subgraph_reports:
            _, measure_thunder = check_metrics(
                graph_folder,
                split_report,
                torchinductor,
                thunderjit,
                WallTimeWithMemoryUsage,
                time_rtol,
                time_atol,
                memory_usage_rtol,
                memory_usage_atol,
                stream,
            )
            check_metrics(
                graph_folder,
                split_report,
                torchinductor,
                thunderjit,
                KernelTime,
                time_rtol,
                time_atol,
                memory_usage_rtol,
                memory_usage_atol,
                stream,
            )
            if measure_thunder is not None:
                runnable_split_reports.append(split_report)
    if not compare_fusion:
        return
    make_nvfusion_reports(runnable_split_reports)

    for graph_report in thunder_fxgraph_reports:
        graph_nvfusion_folder = folder_path / graph_report.graph_name / "nvfusion_reports"
        for split_report in graph_report.subgraph_reports:
            for nvfusion_report in split_report.fusion_reports:
                check_nvfusion_timing(graph_nvfusion_folder, nvfusion_report, WallTime, time_rtol, time_atol, stream)
                check_nvfusion_timing(graph_nvfusion_folder, nvfusion_report, KernelTime, time_rtol, time_atol, stream)


def thunderfx_benchmark_report(
    fn: Callable,
    folder_path: str | PathLike,
    check_torch_runnablility: bool = True,
    time_rtol=0.5,
    time_atol=0.0,
    memory_usage_rtol=0.5,
    memory_usage_atol=0.0,
    compare_fusion: bool = False,
    stream: TextIO = sys.stdout,
    **compile_kwargs,
):
    """
    A utility function that analyzes the runnability and performance benchmarks of each FX graph.

    1. Checks if the callable can be executed with `torch.compile` when `check_torch_runnablility` is `True`.
    - If it fails, attempts to run it eagerly.
    - If eager execution also fails, an error is printed, and the function returns.
    - If eager execution succeeds, the analysis continues.

    2. Collects all ThunderFX subgraphs and verifies whether each subgraph can be successfully executed by Thunder.

    3. For each subgraph:
    - Compares wall time and kernel time between `torch.compile` and Thunder.
    - Saves the scripts in `folder_path/graph_name/failed` if execution fails.
    - Reports performance metrics and saves the benchmark script in `folder_path/graph_name/` if the difference exceeds the tolerance (`time_rtol`, `time_atol` in seconds)
    - Reports memory usage and saves the benchmark script in `folder_path/graph_name/memory_issue` if the difference exceeds the tolerance (`memory_usage_rtol`, `memory_usage_atol` in Bytes).
    - Uses `math.isclose` for tolerance checks.

    4. If `compare_fusion` is `True`:
    - Also compares the wall time and kernel time of each nvFusion region.
    - Saves the benchmark script when necessary in `folder_path/graph_name/nvfusion_reports`, following the same criteria as above.

    Note:
    - This function may run out of memory (OOM) as it allocates random tensors when executing
    the graph module in each Report. To prevent OOM issues, users must manually free the
    input model and arguments to free up memory for `make_nvfusion_reports`, `check_timing`,
    and `check_nvfusion_timing`.
    - See `thunderfx_benchmark_report_from_splits` for details on the generated folders and scripts.

    Returns:
        A wrapped function that performs the analysis when called with inputs

    Here is an example:

    ```python
    thunder_fxgraph_reports = get_thunder_fxgraph_reports(model)(x)

    # Frees the parameters and inputs to make room for the reports
    del model
    del x

    thunderfx_benchmark_report_from_splits(thunder_fxgraph_reports, folder_path, compare_fusion=True)
    ```
    """
    from thunder.dynamo.utils import get_torch_compile_kwargs

    folder_path = Path(folder_path)
    folder_path.mkdir(exist_ok=True, parents=True)
    torch_compile_kwargs = get_torch_compile_kwargs(**compile_kwargs)
    thunder_jit_kwargs = {k: v for k, v in compile_kwargs.items() if k not in torch_compile_kwargs}

    def inner_fn(*args, **kwargs):
        if check_torch_runnablility:
            check_torch_compile_runnability(fn, stream, **torch_compile_kwargs)(*args, **kwargs)
        thunder_fxgraph_reports = get_thunder_fxgraph_reports(fn, stream=stream, **compile_kwargs)(*args, **kwargs)
        thunderfx_benchmark_report_from_splits(
            thunder_fxgraph_reports,
            folder_path,
            compare_fusion,
            time_rtol,
            time_atol,
            memory_usage_rtol,
            memory_usage_atol,
            stream,
            **thunder_jit_kwargs,
        )

    return inner_fn


def save_failing_repros(
    reports: list[FXGraphReport],
    compile_fn: CompileSpecificationInterface,
    repros_folder: str | PathLike,
    *,
    check_consistency: bool = False,
):
    """
    Saves the repros for the failing reports. The failing reason is saved as comment in the repro file.
    example usage:
    ```python
    # Gets the Dynamo FX Graph reports
    report = fx_report(model)(x)
    # Saves the repros for the failing reports using TorchCompile
    save_failing_repros(report.fx_graph_reports, TorchCompileSpecification(), "repros")
    ```
    """
    repros_folder = Path(repros_folder)
    repros_folder.mkdir(exist_ok=True, parents=True)
    for report in reports:
        try:
            report.run_repro(compile_fn, check_consistency)
        except Exception as e:
            comment = f"Failed to run the function using {compile_fn.name} with exception: {e}"
            report.write_repro(
                repros_folder, compile_fn, extra_comment_str=comment, check_consistency=check_consistency
            )


def create_folder(folder_path: str | PathLike, force_overwrite: bool = False):
    folder_path = Path(folder_path)

    if folder_path.exists():
        if not folder_path.is_dir():
            raise RuntimeError(f"{folder_path} exists and is not a directory.")

        if force_overwrite:
            shutil.rmtree(folder_path)
        else:
            raise RuntimeError(f"Folder {folder_path} already exists. Use force_overwrite=True to overwrite.")

    folder_path.mkdir(parents=True, exist_ok=False)


def save_thunderfx_repros(
    fn: Callable,
    folder_path: str | PathLike,
    *,
    use_benchmark: bool = False,
    check_runnability: bool = False,
    save_fusion: bool = False,
    save_trace: bool = False,
    stream: TextIO = sys.stdout,
    force_overwrite: bool = False,
    **compile_kwargs,
):
    """
    Saves reproduction scripts for ThunderFX subgraphs.

    This function:
    1. Creates a folder structure to organize the repros
    .
    └── graph0
        ├── fusion_reports
        │   ├── graph0_thunder_0_nvFusion0_forward_repro_nvfuser.py
        │   ├── graph0_thunder_0_nvFusion1_forward_repro_nvfuser.py
        │   ├── graph0_thunder_0_nvFusion2_backward_repro_nvfuser.py
        ├── graph0_thunder_0_bwd_trace.py
        ├── graph0_thunder_0_fwd_trace.py
        └── graph0_thunder_0.py

    2. For each Thunder FX graph and its subgraphs:
        - Checks runnability if requested
        - Saves benchmark or repro scripts
        - Saves trace information if requested
        - Saves nvFusion repros if requested

    Args:
        fn: The callable to analyze
        folder_path: Path to save repros to
        use_benchmark: If True, saves benchmark scripts instead of repros
        check_runnability: If True, checks if graphs can run with Thunder
        save_fusion: If True, saves nvFusion repros
        save_trace: If True, saves trace information
        stream: Stream to write output log informationto
        force_overwrite: If True, overwrites existing folder at folder_path
        **compile_kwargs: Keyword arguments for Thunder and torch.compile

    Returns:
        A wrapped function that saves repros when called with inputs
    """
    from thunder.dynamo.utils import get_torch_compile_kwargs

    folder_path = Path(folder_path)
    create_folder(folder_path, force_overwrite)
    torch_compile_kwargs = get_torch_compile_kwargs(**compile_kwargs)
    thunder_jit_kwargs = {k: v for k, v in compile_kwargs.items() if k not in torch_compile_kwargs}
    thunderjit = ThunderCompileSpecification(**thunder_jit_kwargs)

    def inner_fn(*args, **kwargs):
        thunder_fxgraph_reports = get_thunder_fxgraph_reports(fn, stream=stream, **compile_kwargs)(*args, **kwargs)
        for thunder_fxgraph_report in thunder_fxgraph_reports:
            graph_folder = folder_path / thunder_fxgraph_report.graph_name
            graph_folder.mkdir(exist_ok=True, parents=True)
            for split_report in thunder_fxgraph_report.subgraph_reports:
                if check_runnability or save_trace or save_fusion:
                    try:
                        split_report.create_fusion_reports()
                    except Exception as e:
                        stream.write(f"Failed to run the {split_report.graph_name} using Thunder with exception: {e}\n")
                        split_report.write_repro(
                            graph_folder, thunderjit, file_name=f"failed_{split_report.graph_name}.py"
                        )
                        continue
                    else:
                        stream.write(f"Successfully ran the {split_report.graph_name} using Thunder\n")
                if use_benchmark:
                    split_report.write_benchmark(graph_folder, thunderjit, WallTime)
                else:
                    split_report.write_repro(graph_folder, thunderjit)
                if save_trace:
                    with open(graph_folder / f"{split_report.graph_name}_fwd_trace.py", "w") as f:
                        f.write(str(split_report.fwd_trc))
                    with open(graph_folder / f"{split_report.graph_name}_bwd_trace.py", "w") as f:
                        f.write(str(split_report.bwd_trc))
                if save_fusion:
                    fusion_folder = graph_folder / "fusion_reports"
                    fusion_folder.mkdir(exist_ok=True, parents=True)
                    for fusion_report in split_report.fusion_reports:
                        if use_benchmark:
                            fusion_report.write_nvfuser_benchmark(fusion_folder, WallTime)
                        else:
                            fusion_report.write_nvfuser_repro(fusion_folder)

    return inner_fn
