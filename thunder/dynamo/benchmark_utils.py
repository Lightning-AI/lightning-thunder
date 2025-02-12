from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch.utils.benchmark import Timer as TorchBenchmarkTimer
from thunder.dynamo.utils import thunder_options_to_str
from thunder.dynamo.timer import TorchProfileTimer

if TYPE_CHECKING:
    from collections.abc import Callable


class CompileSpecificationInterface:
    def compile(self, fn) -> Callable:
        """Compile a given function with provided arguments."""
        raise NotImplementedError("Subclasses should implement the 'compile' method if needed.")

    def to_source(self, fn_name) -> str:
        """Convert the function to its source representation."""
        raise NotImplementedError("Subclasses should implement the 'to_source' method if needed.")

    def import_str(self) -> list[str] | None:
        """Return the necessary imports."""
        return None  # Default implementation returns None


class ThunderCompileSpecification(CompileSpecificationInterface):
    def __init__(self, **kwargs):
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
    def __init__(self, **kwargs):
        self.torch_compile_options: dict = kwargs

    def compile(self, fn):
        return torch.compile(fn, **self.torch_compile_options)

    def to_source(self, fn_name):
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.torch_compile_options.items())
        return f"torch.compile({fn_name}, {kwargs_str})"

    def import_str(self):
        return ["import torch"]


class TorchEagerSpecification(CompileSpecificationInterface):
    def compile(self, fn):
        return fn

    def to_source(self, fn_name):
        return fn_name


class TimerInterface:
    @staticmethod
    def time(fn, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement the 'compile' method if needed.")

    @staticmethod
    def to_source(*args, **kwargs):
        raise NotImplementedError("Subclasses should implement the 'compile' method if needed.")

    @staticmethod
    def import_str(*args, **kwargs):
        return None


class WallTime(TimerInterface):
    @staticmethod
    def time(stmt="pass", setup="pass", globals=None, min_run_time: float = 0.2) -> Measurement:
        t = TorchBenchmarkTimer(stmt=stmt, setup=setup, globals=globals)
        return t.blocked_autorange(min_run_time=min_run_time)

    @staticmethod
    def import_str():
        return ["from thunder.dynamo.benchmark_utils import WallTime"]

    @staticmethod
    def to_source(fn_name="compiled_model", inputs_name="inputs"):
        return f'WallTime.time("{fn_name}(*{inputs_name})", globals={{"{fn_name}":{fn_name}, "{inputs_name}": {inputs_name}}})'


class KernelTime(TimerInterface):
    @staticmethod
    def time(stmt="pass", setup="pass", globals=None, min_run_time: float = 0.2) -> Measurement:
        inner_timer = TorchProfileTimer()
        t = TorchBenchmarkTimer(stmt=stmt, setup=setup, timer=inner_timer, globals=globals)
        return t.blocked_autorange(min_run_time=min_run_time)

    @staticmethod
    def import_str():
        return ["from thunder.dynamo.benchmark_utils import KernelTime"]

    @staticmethod
    def to_source(fn_name="compiled_model", inputs_name="inputs"):
        return f'KernelTime.time("{fn_name}(*{inputs_name})", globals={{"{fn_name}":{fn_name}, "{inputs_name}": {inputs_name}}})'


class BoundSymbolNvfuserSpecification(CompileSpecificationInterface):
    def compile(self, nvfusion_bsym):
        return nvfusion_bsym._call_ctx[nvfusion_bsym.sym.name]


class BoundSymbolTorchCompileSpecification(CompileSpecificationInterface):
    def compile(self, bsym):
        from thunder.executors.torch_compile import make_compiled

        return make_compiled(bsym.subsymbols, bsym.flat_args, bsym.flat_outs)
