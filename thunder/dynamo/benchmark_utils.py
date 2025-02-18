from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch.utils.benchmark import Timer as TorchBenchmarkTimer
from torch.profiler import profile, ProfilerActivity
from thunder.dynamo.utils import thunder_options_to_str

if TYPE_CHECKING:
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

    def compile(self, fn: Callable) -> Callable:
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

    def compile(self, fn):
        from thunder import jit

        return jit(fn, **self.thunder_options)

    def to_source(self, fn_name):
        thunder_options_str = thunder_options_to_str(self.thunder_options)
        return f"thunder.jit({fn_name}, {thunder_options_str})"

    def import_str(self):
        return ["import thunder"]


class TorchCompileSpecification(CompileSpecificationInterface):
    """
    A compile specification for :func:`torch.compile`.
    """

    def __init__(self, specification_name="torchcompile", **kwargs):
        self.name = specification_name
        self.torch_compile_options: dict = kwargs

    def compile(self, fn):
        return torch.compile(fn, **self.torch_compile_options)

    def to_source(self, fn_name):
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.torch_compile_options.items())
        return f"torch.compile({fn_name}, {kwargs_str})"

    def import_str(self):
        # WAR for triton error https://github.com/pytorch/pytorch/issues/124565
        comment_str = '# Workaround for "RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered"\n# https://github.com/pytorch/pytorch/issues/124565'
        code = f'torch.cuda.is_available():\n    torch.empty(1, device="cuda", requires_grad=True).backward()'
        return ["import torch", comment_str, code]


class TorchEagerSpecification(CompileSpecificationInterface):
    """
    A compile specification for Torch eager mode, which returns the callable unchanged.
    """

    def __init__(self, specification_name="torcheager"):
        self.name = specification_name

    def compile(self, fn):
        return fn

    def to_source(self, fn_name):
        return fn_name


class TorchInductorSpecification(CompileSpecificationInterface):
    """
    A compilation specification for using TorchInductor without TorchDynamo.
    For details on why compilation without TorchDynamo is needed, see:
    https://github.com/Lightning-AI/lightning-thunder/issues/1521
    """

    def __init__(self, inputs, specification_name="inductor_backend"):
        self.name: str = specification_name
        self.inputs: list = inputs

    @staticmethod
    def torch_inductor(fn, inputs):
        from torch._inductor import compile as inductor_compile
        from torch.fx import symbolic_trace

        fx_graph = symbolic_trace(fn)
        return inductor_compile(fx_graph, inputs)

    def compile(self, fn):
        return self.torch_inductor(fn, self.inputs)

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
        return nvfusion_bsym._call_ctx[nvfusion_bsym.sym.name]


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


class TimerInterface:
    """
    Defines an interface for specifying how to timing a callable and generate a statistic object.
    To create a custom timing way, subclass this interface and implement the methods as needed.

    Example implementations:
        - :class:`WallTime`
        - :class:`KernelTime`
    """

    @staticmethod
    def time(fn, *args, **kwargs):
        """
        Measures the execution time of a given callable.
        Subclasses must implement this method to define a specific timing mechanism.

        Returns:
            Any: The timing result, typically a statistics object.
        """
        raise NotImplementedError("Subclasses should implement the 'time' method if needed.")

    @staticmethod
    def to_source(*args, **kwargs) -> str:
        """
        Converts the timing function into its source code representation.
        Subclasses can implement this method to define how the timing logic is represented as code.
        """
        raise NotImplementedError("Subclasses should implement the 'to_source' method if needed.")

    @staticmethod
    def import_str(*args, **kwargs):
        """
        Returns the necessary import statements for the timing implementation.
        """
        return None


class WallTime(TimerInterface):
    """
    A timing utility that measures execution time using `torch.utils.benchmark.Timer`.
    This class implements `TimerInterface` and provides a method to measure wall-clock time
    using PyTorch's benchmarking utilities.
    """

    @staticmethod
    def time(stmt="pass", setup="pass", globals=None, min_run_time: float = 0.2) -> Measurement:
        """
        Measures execution time using PyTorch's :func:`torch.utils.benchmark.Timer.blocked_autorange()`.

        Args:
            stmt (str, optional): Code snippet to be run in a loop and timed.
            setup (str, optional): Optional setup code. Used to define variables used in `stmt`
            globals (dict, optional): A dictionary of global variables for the executed code. Defaults to `None`.
            min_run_time (float, optional): The minimum execution time (in seconds) to determine the number of runs. Defaults to `0.2`.

        Returns:
            Measurement: A benchmarking result containing execution time statistics, see :class:`torch.utils.benchmark.utils.common.Measurement`.
        """
        t = TorchBenchmarkTimer(stmt=stmt, setup=setup, globals=globals)
        return t.blocked_autorange(min_run_time=min_run_time)

    @staticmethod
    def import_str():
        return ["from thunder.dynamo.benchmark_utils import WallTime"]

    @staticmethod
    def to_source(fn_name, inputs_name):
        return f'WallTime.time("{fn_name}(*{inputs_name})", globals={{"{fn_name}":{fn_name}, "{inputs_name}": {inputs_name}}})'


class KernelTime(TimerInterface):
    """
    A timing utility that measures CUDA kernel time using PyTorch's :class:`torch.utils.benchmark.Timer` with a custom time function :class:`TorchProfileTimer`.
    """

    @staticmethod
    def time(stmt="pass", setup="pass", globals=None, min_run_time: float = 0.2) -> Measurement:
        """
        Measures kernel time using PyTorch's `Timer.blocked_autorange()` with the kernel timing using :func:`torch.profiler.profile`.
        More details see :class:`TorchProfileTimer`
        """
        inner_timer = TorchProfileTimer()
        t = TorchBenchmarkTimer(stmt=stmt, setup=setup, timer=inner_timer, globals=globals)
        return t.blocked_autorange(min_run_time=min_run_time)

    @staticmethod
    def import_str():
        return ["from thunder.dynamo.benchmark_utils import KernelTime"]

    @staticmethod
    def to_source(fn_name, inputs_name):
        return f'KernelTime.time("{fn_name}(*{inputs_name})", globals={{"{fn_name}":{fn_name}, "{inputs_name}": {inputs_name}}})'
