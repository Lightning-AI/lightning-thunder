from __future__ import annotations
from typing import TYPE_CHECKING
import sys
import inspect
from pathlib import Path
import torch
from torch.utils.benchmark import Timer as TorchBenchmarkTimer
from torch.profiler import profile, ProfilerActivity
from thunder.dynamo.utils import thunder_options_to_str
from thunder.core.utils import check
from torch.utils.benchmark.utils.common import select_unit as select_time_unit

if TYPE_CHECKING:
    from typing import TextIO
    from collections.abc import Callable
    from torch.utils.benchmark.utils.common import Measurement


class CompileSpecificationInterface:
    """
    Defines an interface for specifying how to compile a callable and generate its corresponding source code.

    To create a custom compile specification, subclass this interface and implement the methods as needed.

    Example implementations:
        - :class:`TorchCompileSpecification`
        - :class:`TorchEagerSpecification`
        - :class:`ThunderCompileSpecification`
    """

    def compile(self, fn: Callable, **kwargs) -> Callable:
        """Compiles the given callable and returns the compiled version."""
        raise NotImplementedError("Subclasses should implement the 'compile' method if needed.")

    def to_source(self, fn_name: str) -> str:
        """Converts the compile function to its source code representation."""
        raise NotImplementedError("Subclasses should implement the 'to_source' method if needed.")

    def import_str(self) -> list[str] | None:
        """Returns the necessary imports."""
        return None


class ThunderCompileSpecification(CompileSpecificationInterface):
    """
    A compile specification for :func:`thunder.jit`.

    Attributes:
        thunder_options (dict): Compilation options to be passed to :func:`thunder.jit`.

    Example usage:
        from thunder import nvfuser_executor
        spec = ThunderCompileSpecification(executors=[nvfuser_executor])
        compiled_fn = spec.compile(my_function)
    """

    def __init__(self, specification_name="thunder", **kwargs):
        self.name = specification_name
        self.thunder_options: dict = kwargs

    def compile(self, fn, **kwargs):
        from thunder import jit

        return jit(fn, **self.thunder_options)

    def to_source(self, fn_name):
        thunder_options_str = thunder_options_to_str(self.thunder_options)
        return (
            f"thunder.jit({fn_name})" if not thunder_options_str else f"thunder.jit({fn_name}, {thunder_options_str})"
        )

    def import_str(self):
        return ["import thunder"]


class ThunderCompilerOnGraphModuleSpecification(CompileSpecificationInterface):
    def __init__(self, specification_name="thunderfx", **kwargs):
        self.name = specification_name
        self.thunder_options: dict = kwargs

    def compile(self, gm, **kwargs):
        from thunder.dynamo import ThunderCompiler

        thunder_compiler = ThunderCompiler(**kwargs)
        split_gm = thunder_compiler(gm, sample_args=None)
        return split_gm


class TorchCompileSpecification(CompileSpecificationInterface):
    """
    A compile specification for :func:`torch.compile`.
    """

    def __init__(self, specification_name="torchcompile", **kwargs):
        self.name = specification_name
        self.torch_compile_options: dict = kwargs

    def compile(self, fn, **kwargs):
        return torch.compile(fn, **self.torch_compile_options)

    def to_source(self, fn_name):
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.torch_compile_options.items())
        return f"torch.compile({fn_name}, {kwargs_str})"

    def import_str(self):
        # WAR for triton error https://github.com/pytorch/pytorch/issues/124565
        comment_str = '# Workaround for "RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered"\n# https://github.com/pytorch/pytorch/issues/124565'
        code = 'if torch.cuda.is_available():\n    torch.empty(1, device="cuda", requires_grad=True).backward()'
        return ["import torch", comment_str, code]


class TorchEagerSpecification(CompileSpecificationInterface):
    """
    A compile specification for Torch eager mode, which returns the callable unchanged.
    """

    def __init__(self, specification_name="torcheager"):
        self.name = specification_name

    def compile(self, fn, **kwargs):
        return fn

    def to_source(self, fn_name):
        return fn_name


class TorchInductorSpecification(CompileSpecificationInterface):
    """
    A compilation specification for using TorchInductor without TorchDynamo.
    For details on why compilation without TorchDynamo is needed, see:
    https://github.com/Lightning-AI/lightning-thunder/issues/1521
    """

    def __init__(self, specification_name="inductor_backend", *, skip_symbolic_trace=True):
        self.name: str = specification_name
        # self.skip_symbolic_trace decides whether to skip symbolic trace for self.compile
        self.skip_symbolic_trace = skip_symbolic_trace

    @staticmethod
    def torch_inductor(fn, inputs, *, skip_symbolic_trace=False):
        from torch._inductor import compile as inductor_compile
        from torch.fx import symbolic_trace

        if not skip_symbolic_trace:
            fn = symbolic_trace(fn)
        return inductor_compile(fn, inputs)

    def compile(self, fn, *, inputs, **kwargs):
        return self.torch_inductor(fn, inputs, skip_symbolic_trace=self.skip_symbolic_trace)

    # to_source will always use symbolic trace
    def to_source(self, fn_name):
        return f"TorchInductorSpecification.torch_inductor({fn_name}, inputs)"

    def import_str(self):
        return ["import torch", "from thunder.dynamo.benchmark_utils import TorchInductorSpecification"]


class BoundSymbolNvfuserSpecification(CompileSpecificationInterface):
    def __init__(self, specification_name="nvfuser"):
        self.name: str = specification_name

    # Returns the nvFuser callable from the nvFuser bound symbol.
    # See the TODO in :class:`thunder.dynamo.report.ThunderFusionReport` for more details.
    def compile(self, nvfusion_bsym):
        fd = nvfusion_bsym._call_ctx[nvfusion_bsym.sym.name].last_used
        return lambda *args: fd.execute(args)


class BoundSymbolTorchCompileSpecification(CompileSpecificationInterface):
    def __init__(self, specification_name="torchcompile"):
        self.name: str = specification_name

    # Returns the torch compile callable from the nvFuser bound symbol.
    # See the TODO in :class:`thunder.dynamo.report.ThunderFusionReport` for more details.
    def compile(self, bsym):
        from thunder.executors.torch_compile import make_compiled

        return make_compiled(bsym.subsymbols, bsym.flat_args, bsym.flat_outs)


# NOTE: This class is modified from https://github.com/NVIDIA/Fuser/blob/212ac38e08c47251356e0f0ee8f48e21a12b2293/benchmarks/python/core.py#L141
# TODO: Collaborate with the nvFuser team to unify this implementation,
# ensuring a single version exists (either in Thunder or nvFuser) so that improvements benefit both projects.
class TorchProfileTimer:
    def __init__(self):
        self.prof = profile(activities=[ProfilerActivity.CUDA])
        self.current_time = 0.0

    def _get_kernel_time(self, prof_averages: torch.autograd.profiler_util.EventList) -> float:
        """
        Arguments:
            prof_averages: Output of self.prof.key_averages()
        Returns:
            time_value: Elapsed CUDA time in seconds.
        """
        from torch.autograd import DeviceType

        elapsed_cuda_time = 0
        has_cuda_event = False
        for event in prof_averages:
            if event.device_type != DeviceType.CUDA:
                continue
            has_cuda_event = True
            # Re: torch profiler API changes in https://github.com/pytorch/pytorch/pull/123247
            elapsed_cuda_time = (
                elapsed_cuda_time + event.self_device_time_total
                if hasattr(event, "self_device_time_total")
                else event.self_cuda_time_total
            )

        return elapsed_cuda_time / 1e6

    def _increment_global_time(self, elapsed_time: float) -> None:
        self.current_time += elapsed_time

    def __call__(self):
        """
        Custom torchprofiler-based timer used by pytest-benchmark.
        At every timer call, the profiler is stopped to compute the elapsed CUDA time
        and the global clock is incremented. The profiler is restarted before returning to continue tracing.

        Returns:
            self.current_time: Global monotonic clock variable
        """
        try:
            self.prof.stop()
        except AssertionError:
            self.prof.start()
            return self.current_time

        prof_averages = self.prof.key_averages()
        elapsed_cuda_time = self._get_kernel_time(prof_averages)
        self._increment_global_time(elapsed_cuda_time)
        # Clear the internal profiler object to avoid accumulating function events and then restart the profiler
        # See PR: https://github.com/pytorch/pytorch/pull/125510
        self.prof.profiler = None

        return self.current_time

    def cleanup(self):
        """
        Stops a running torchprofiler instance if found.
        """
        self.current_time = 0.0
        try:
            self.prof.stop()
        except AssertionError:
            pass


# TODO: We want to refine the extensibility to support customizing the memory timer.
# e.g. Create a base class with an abstract method get_max_allocated(),
# Support the peak reserved memory timer by adding a e.g. MaxReservedMemoryTimer base class.
class TimerWithCUDAMemoryUsage:
    """
    A timer wrapper that tracks CUDA memory usage alongside timing measurements.

    NOTE: `torch.cuda.max_memory_allocated()` is used to record the peak allocatedmemory usage.
    and the memory stats is reset by `torch.cuda.reset_peak_memory_stats()` after each timer call.
    See https://pytorch.org/docs/stable/notes/cuda.html#memory-management for more details.

    Example usage:
        t = TimerWithCUDAMemoryUsage(TimerInterface.time)
        t0 = t()  # Records initial memory and time
        # ... code to measure ...
        t1 = t()  # Records final memory and time
        duration = t1 - t0  # Get elapsed time
        memory_mb = t.max_allocated_memory  # Get peak memory usage in B

    Note:
        The memory tracking adds some overhead to the timing measurements.
        Memory usage is recorded in bytes (B).
    """

    def __init__(self, timer=torch.utils.benchmark.utils.timer.timer):
        self.max_allocated_memory = 0.0
        self.timer = timer

    def __call__(self):
        self.max_allocated_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return self.timer()


class TimerInterface:
    """
    Defines an interface for specifying how to timing a callable and generate a statistic object.
    To create a custom timing way, subclass this interface and implement the methods as needed.

    Example implementations:
        - :class:`WallTime`
        - :class:`KernelTime`
    """

    def time(self, *args, **kwargs):
        """
        Measures the execution time of a given callable.
        Subclasses must implement this method to define a specific timing mechanism.

        Returns:
            Any: The timing result, typically a statistics object.
        """
        raise NotImplementedError("Subclasses should implement the 'time' method if needed.")

    def to_source(self, *args, **kwargs) -> str:
        """
        Converts the timing function into its source code representation.
        Subclasses can implement this method to define how the timing logic is represented as code.
        """
        raise NotImplementedError("Subclasses should implement the 'to_source' method if needed.")

    def import_str(self, *args, **kwargs):
        """
        Returns the necessary import statements for the timing implementation.
        """
        return None


class TorchBenchmarkTimerSpecification(TimerInterface):
    """
    A timing utility that measures execution time using :class:`torch.utils.benchmark.utils.timer.Timer`.
    the inner timer is used as the custom timer in timeit.Timer to return the current time, default is :func:`torch.utils.benchmark.utils.timer.timer`.
    See: :class:`torch.utils.benchmark.utils.timer.Timer` for more details.
    """

    def __init__(
        self,
        name: str = "TorchBenchmarkTimerSpecification",
        inner_timer: Callable = torch.utils.benchmark.utils.timer.timer,
        *,
        threshold: float | None = None,
        min_run_time: float | None = None,
        max_run_time: float | None = None,
    ):
        self.inner_timer = inner_timer
        self.name = name

        default_params = inspect.signature(TorchBenchmarkTimer.adaptive_autorange).parameters

        self.threshold = threshold if threshold is not None else default_params["threshold"].default
        self.min_run_time = min_run_time if min_run_time is not None else default_params["min_run_time"].default
        self.max_run_time = max_run_time if max_run_time is not None else default_params["max_run_time"].default

    def time(self, stmt="pass", setup="pass", globals=None) -> Measurement:
        """
        Measures execution time using PyTorch's :func:`torch.utils.benchmark.Timer.adaptive_autorange()`.

        Args:
            stmt (str, optional): Code snippet to be run in a loop and timed.
            setup (str, optional): Optional setup code. Used to define variables used in `stmt`
            globals (dict, optional): A dictionary of global variables for the executed code. Defaults to `None`.

        Returns:
            Measurement: A benchmarking result containing execution time statistics, see :class:`torch.utils.benchmark.utils.common.Measurement`.
        """
        t = TorchBenchmarkTimer(stmt=stmt, setup=setup, globals=globals, timer=self.inner_timer)
        measurement = t.adaptive_autorange(
            threshold=self.threshold, min_run_time=self.min_run_time, max_run_time=self.max_run_time
        )
        if hasattr(self.inner_timer, "max_allocated_memory"):
            measurement.max_allocated_memory = self.inner_timer.max_allocated_memory
        return measurement

    def import_str(self):
        return [f"from thunder.dynamo.benchmark_utils import {self.name}"]

    def __repr__(self):
        return f"{self.name}(threshold={self.threshold}, min_run_time={self.min_run_time}, max_run_time={self.max_run_time})"

    def to_source(self, fn_name, inputs_name):
        return f'{self.__repr__()}.time("{fn_name}(*{inputs_name})", globals={{"{fn_name}":{fn_name}, "{inputs_name}": {inputs_name}}})'

    def __call__(
        self,
        *,
        threshold: float | None = None,
        min_run_time: float | None = None,
        max_run_time: float | None = None,
    ):
        return self.__class__(
            name=self.name,
            inner_timer=self.inner_timer,
            threshold=threshold if threshold is not None else self.threshold,
            min_run_time=min_run_time if min_run_time is not None else self.min_run_time,
            max_run_time=max_run_time if max_run_time is not None else self.max_run_time,
        )


WallTime = TorchBenchmarkTimerSpecification("WallTime")

walltime_with_memory_usage = TimerWithCUDAMemoryUsage()
WallTimeWithMemoryUsage = TorchBenchmarkTimerSpecification("WallTimeWithMemoryUsage", walltime_with_memory_usage)

torch_profile_timer = TorchProfileTimer()
KernelTime = TorchBenchmarkTimerSpecification("KernelTime", torch_profile_timer)


def check_threshold(a, b, rtol, atol):
    import math

    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def get_pretty_time_str(a):
    time_unit, time_scale = select_time_unit(a)
    return f"{a / time_scale:.3f} {time_unit}"


def get_pretty_memory_str(value):
    """
    Converts memory size in bytes to human readable string with appropriate unit suffix.

    Args:
        value (float): Memory size in bytes

    Returns:
        str: Memory size with appropriate unit (B, KB, MB, GB, TB, PB)
    """
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    suffix_idx = 0

    while suffix_idx + 1 < len(suffixes) and value >= 1000.0:
        value /= 1000.0
        suffix_idx += 1

    return f"{value:.3f} {suffixes[suffix_idx]}"


def check_timing(a: float, b: float, a_name: str, b_name: str, test_name: str, timer_name: str, rtol, atol):
    """
    Compare two timing measurements against a defined threshold and generate a log message.
    If not exceed the threshold, abs(a-b) <= max(rtol * max(abs(a), abs(b)), atol).

    Parameters:
        a, b (float): Timing measurement for the compilation methods. Typically obtained from `Measurement.median`.
        a_name, b_name (str): Identifier for the two compilation methods to be compared (e.g., thunder, torch.compile).
        test_name (str): Name of the function whose performance is being measured.
        timer_name (str): Identifier for the timing mechanism used.
        rtol (float): Relative tolerance for the comparison.
        atol (float): Absolute tolerance for the comparison (in seconds).

    Returns:
        Tuple[bool, str]: A tuple where:
            - The first element is a boolean indicating whether the performance difference exceeds the threshold.
            - The second element is a log string detailing the comparison results.
    """
    a_time_str = get_pretty_time_str(a)
    b_time_str = get_pretty_time_str(b)
    if not check_threshold(a, b, rtol=rtol, atol=atol):
        log_str = f"Benchmark {timer_name} for **{test_name}** requires investigation: {b_name}({b_time_str}) and {a_name}({a_time_str}) is not close (rtol={rtol}, atol={atol})\n"
        return False, log_str
    log_str = (
        f"Benchmark {timer_name} ran successfully on **{test_name}**: {b_name}({b_time_str}) , {a_name}({a_time_str})\n"
    )
    return True, log_str


def is_report_using_cuda(report):
    from thunder.dynamo.utils import ExampleInputMetaData
    from thunder.core.pytree import tree_map

    def _check_tensor(x):
        return isinstance(x, ExampleInputMetaData) and x.device.type == "cuda"

    return any(tree_map(_check_tensor, report.example_input_meta))


def check_memory_usage(m1, m2, m1_name, m2_name, test_name, rtol, atol):
    m1_memory_str = get_pretty_memory_str(m1)
    m2_memory_str = get_pretty_memory_str(m2)
    if not check_threshold(m1, m2, rtol=rtol, atol=atol):
        log_str = f"Max_allocated_memory for **{test_name}** requires investigation: {m1_name}({m1_memory_str}) and {m2_name}({m2_memory_str}) is not close (rtol={rtol}, atol={atol})\n"
        return False, log_str
    log_str = f"Max_allocated_memory: [{test_name}] {m1_name}({m1_memory_str}); {m2_name}({m2_memory_str})\n"
    return True, log_str


def check_metrics(
    folder_path,
    report,
    compile_fn1: Callable,
    compile_fn2: Callable,
    timer_fn: Callable,
    time_rtol=0.5,
    time_atol=0.0,
    memory_usage_rtol=0.5,
    memory_usage_atol=0.0,
    stream: TextIO = sys.stdout,
):
    """
    Check the timing and memory usage (if using WallTimeWithMemoryUsage) of graph report using two different compilation specifications with the provided timer configuration
    and generate a benchmark script if the difference exceeds the threshold.
    """
    folder_path = Path(folder_path)
    graph_name = report.graph_name
    if not is_report_using_cuda(report):
        stream.write(f"{graph_name} doesn't use CUDA, skip benchmark {timer_fn.name}")
        return None, None
    filename1 = f"{graph_name}_{compile_fn1.name}_{timer_fn.name}.py"
    filename2 = f"{graph_name}_{compile_fn2.name}_{timer_fn.name}.py"

    def try_and_log_benchmark(compile_fn, filename):
        try:
            return report.run_benchmark(compile_fn, timer_fn)
        except Exception as e:
            msg = f"Benchmark {timer_fn.name} on {graph_name} using {compile_fn.name} failed with exception {e}, benchmark script failed_{filename} is saved"
            stream.write(msg)
            failed_folder = folder_path / "failed"
            report.write_benchmark(
                failed_folder, compile_fn, timer_fn, file_name=f"failed_{filename}", extra_comment_str=msg
            )
            return None

    _, *measure1 = try_and_log_benchmark(compile_fn1, filename1)
    _, *measure2 = try_and_log_benchmark(compile_fn2, filename2)

    if measure1 is None or measure2 is None:
        return measure1, measure2

    perf_record = False
    memory_record = False
    log_strs = ""
    for m1, m2, name in zip(measure1, measure2, ("forward", "backward")):
        check(
            (m1 is None) == (m2 is None),
            f"{name} measurement for the two compilation methods should either both be None or both not None, but got {m1} and {m2}",
        )
        if m1 is None:
            continue
        if timer_fn.name == "WallTimeWithMemoryUsage":
            memory_ret = check_memory_usage(
                m1.max_allocated_memory,
                m2.max_allocated_memory,
                compile_fn1.name,
                compile_fn2.name,
                f"{graph_name} {name}",
                memory_usage_rtol,
                memory_usage_atol,
            )
            if not memory_ret[0]:
                memory_record = True
            log_strs += memory_ret[1]

        ret = check_timing(
            m1.median,
            m2.median,
            compile_fn1.name,
            compile_fn2.name,
            f"{graph_name} {name}",
            timer_fn.name,
            time_rtol,
            time_atol,
        )
        if not ret[0]:
            perf_record = True
        log_strs += ret[1]

    def save_script(folder_path, compile_fn, filename):
        report.write_benchmark(folder_path, compile_fn, timer_fn, file_name=filename, extra_comment_str=log_strs)

    stream.write(log_strs)
    if perf_record:
        save_script(folder_path, compile_fn1, filename1)
        save_script(folder_path, compile_fn2, filename2)
        stream.write(f"The scripts are saved: {filename1}, {filename2}\n")
    if memory_record:
        memory_issue_folder = folder_path / "memory_issue"
        save_script(memory_issue_folder, compile_fn1, filename1)
        save_script(memory_issue_folder, compile_fn2, filename2)
    stream.write("\n")
    return measure1, measure2


def check_nvfusion_timing(folder_path, report, timer_fn, rtol=0.5, atol=0.0, stream: TextIO = sys.stdout):
    """
    Check the timing of the nvfusion region report using two different compilation specifications with the provided timer configuration
    and generate a benchmark script if the difference exceeds the threshold.
    """
    graph_name = report.name
    timer_name = timer_fn.name
    bsym_nvfuser = BoundSymbolNvfuserSpecification()
    bsym_torchcompile = BoundSymbolTorchCompileSpecification()
    measure1 = report.run_benchmark(bsym_nvfuser, timer_fn)
    measure2 = report.run_benchmark(bsym_torchcompile, timer_fn)

    ret = check_timing(
        measure1.median, measure2.median, bsym_nvfuser.name, bsym_torchcompile.name, graph_name, timer_name, rtol, atol
    )
    stream.write(ret[1])
    if not ret[0]:
        extra_comment = f"Benchmark results:\n{ret[1]}\n"
        filename1 = f"{graph_name}_{bsym_nvfuser.name}_{timer_name}.py"
        filename2 = f"{graph_name}_{bsym_torchcompile.name}_{timer_name}.py"
        report.write_nvfuser_benchmark(folder_path, timer_fn, file_name=filename1, extra_comment_str=extra_comment)
        report.write_inductor_benchmark(folder_path, timer_fn, file_name=filename2, extra_comment_str=extra_comment)
        stream.write(f"The scripts are saved: {filename1}, {filename2}\n")
    stream.write("\n")
