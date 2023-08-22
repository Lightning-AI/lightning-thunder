import dataclasses
import functools
from typing import Any, Callable, Dict, List, Tuple, Sequence
from enum import auto, Enum
import time
from functools import partial
import textwrap
from numbers import Number
import operator
import math
import sys
import argparse
from dataclasses import dataclass

import torch
from torch.testing import make_tensor

import thunder
import thunder.torch as ltorch
from thunder.cudagraphs import CUDAGraphExecutor
import thunder.core.dtypes as dtypes

# List of all benchmarks
benchmarks: list = []


# Prints the benchmarks in alphabetical order (either by classname or the benchmark's "name" attribute)
def list_benchmarks(use_classname: bool = True) -> None:
    print("Available benchmarks:")

    name_fn = lambda x: x[0]
    if not use_classname:
        name_fn = lambda x: x[1].name

    for x in sorted(benchmarks, key=name_fn):
        name = name_fn(x)
        cls_name, b = x
        print(f"\t{name}. {b.description}")


@dataclasses.dataclass
class BenchmarkArg:
    """
    Describes a benchmark argument.
    """

    name: str
    description: str


# Simple metaclass that automatically adds defined benchmarks to the list of benchmarks above
class UserFacingBenchmarkMeta(type):
    def __new__(metacls, clsname, bases, namespace):
        return super(UserFacingBenchmarkMeta, metacls).__new__(metacls, clsname, bases, namespace)

    def __init__(cls: type, name: str, bases, namespace: dict) -> None:
        benchmarks.append((name, cls))

        return None


# Base class for benchmarks
class Benchmark:
    """
    Encapsulates a benchmark.

    To add a benchmark:
        1) Create a new class for the benchmark that is a subclass of this class.

        2) If the benchmark should be user-facing (for example, it is not a subclass of Benchmark
            designed to be inherited by other benchmarks), then set the class's metaclass to
            UserFacingBenchmarkMeta:

            class MyBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):

        2) Define the name and description class methods:

            @classmethod
            @property
            def name(cls) -> str:
                return "my_name"

            @classmethod
            @property
            def description(cls) -> str:
                return "My description"

            name() should return a short, distinct, and a valid filename (e.g. "nanogpt, llamba-block").
            description() should return a short sentence describing the benchmark (e.g. "NanoGPT's LayerNorm module forward").

        2) Define a sequence of BenchmarkArgs the benchmark accepts.

            _args = (
                BenchmarkArg(...),
                BenchmarkArg(...),
            )

        2) Implement an args() method that returns the sequence of _args.

            @classmethod
            @property
            def args(cls) -> tuple[BenchmarkArg, ...]:
                return cls._args

        3) Implement the __init__() method.
            It should accept a positional parameter for each benchmark arg, in the order
            they are returned from args()

            def __init__(self, arg0=default0, arg1=default1, ...):
                super().__init__(self)

                self.devices: list[str] = [dev0, ...]

            __init__() should call super().
            __init__() should set the "devices" attribute to be a list of all devices (as strings) used
            __init__() can accept additional optional parameters, like parameters with default values
            or kwargs, but these parameters must be after the benchmark arg parameters.

        4) Implement the make_batch() method.

            def make_batch(self) -> tuple[list, dict]:

            make_batch() should produce a valid input for the benchmark, possibly modified by
            the initialization arguments, as a list of args and a dict of kwargs.

        5) Implement the fn() method.

            def fn(self) -> Callable:

            fn() should return a Callable to be benchmarked.
            The returned callable must accept the output of make_batch(), and it may be modified depending on
            the initialization arguments.

    """

    def __init__(self):
        self.devices: list[str] = []

    @classmethod
    @property
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @property
    def description(cls) -> str:
        raise NotImplementedError

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        raise NotImplementedError

    def make_batch(self) -> tuple[list, dict]:  # args, kwargs
        raise NotImplementedError

    def fn(self, *args, **kwargs) -> Callable:
        raise NotImplementedError


def describe_benchmark(benchmark: type) -> None:
    print(f"{benchmark.name}. {benchmark.description}")
    print("Arguments:")
    for arg in benchmark.args:
        print(f"\t{arg.name}. {arg.description}")


# Base class for benchmark executors
class BenchmarkExecutor:
    """
    To add a benchmark executor ...

    1) Define a make_callable() method:

        (self, fn: Callable) -> Callable

        That accepts a function fn and returns another function that executes it using a
        particular executor (possibly with settings determined by __init__())

    2) Add one or more labels to BENCHMARK_EXECUTORS and update _benchmark_executor_map to
        map the label or labels to instantiations of the executor

    """

    def __init__(self):
        pass

    def make_callable(self, fn: Callable) -> Callable:
        raise NotImplementedError


# NOTE All times are in nanoseconds from epoch
@dataclass
class BenchmarkRunStatistics:
    total_time: int
    has_extended_stats: bool = False
    last_trace_start: int = -1
    last_trace_stop: int = -1
    last_trace_cache_start: int = -1
    last_trace_cache_stop: int = -1
    last_trace_tracing_start: int = -1
    last_trace_tracing_stop: int = -1
    last_trace_execution_start: int = -1
    last_trace_execution_stop: int = -1


# A timing helper
def _benchmark(
    fn: Callable, generator: Callable, wait_for_computation: Callable, repetitions: int
) -> list[BenchmarkRunStatistics]:
    stats = []
    for _ in range(repetitions):
        args, kwargs = generator()
        wait_for_computation()
        start: int = time.time_ns()
        fn(*args, **kwargs)
        wait_for_computation()
        stop: int = time.time_ns()

        cd = thunder.compile_data(fn)
        stat = BenchmarkRunStatistics(stop - start)
        if cd is not None:
            stat.has_extended_stats = True
            stat.last_trace_start = cd.last_trace_start
            stat.last_trace_stop = cd.last_trace_stop
            stat.last_trace_cache_start = cd.last_trace_cache_start
            stat.last_trace_cache_stop = cd.last_trace_cache_stop
            stat.last_trace_tracing_start = cd.last_trace_tracing_start
            stat.last_trace_tracing_stop = cd.last_trace_tracing_stop
            stat.last_trace_execution_start = cd.last_trace_execution_start
            stat.last_trace_execution_stop = cd.last_trace_execution_stop

        stats.append(stat)

    return stats


# TODO Consider filling a large tensor with zeros in an attempt to prevent caching
def wait_for_cuda_computation() -> None:
    torch.cuda.synchronize()


# Prints nanoseconds as microseconds, rounded
def ns_to_us(ns: Number) -> str:
    us = "\u03BCs"
    return f"{round(ns / 1000)}{us}"


def _prettyprint_stats(
    benchmark_name: str,
    *,
    callable_construction_time: int,
    warmup_stats: list[BenchmarkRunStatistics],
    benchmark_stats: list[BenchmarkRunStatistics],
) -> None:
    assert len(warmup_stats) > 0, "Expected at least one warmup run"
    assert len(benchmark_stats) > 0, "Expected at least one benchmark run"

    # Converts callable construction time, in nanoseconds, to a string (in rounded microseconds)
    callable_construction_time_us: str = ns_to_us(callable_construction_time)

    # Computes total warmup time, in nanoseconds, and converts it to a string (in rounded microsecnods)
    total_warmup_time_ns: int = 0
    for stat in warmup_stats:
        total_warmup_time_ns += stat.total_time
    total_warmup_time_us: str = ns_to_us(total_warmup_time_ns)

    # Computes average warmup time
    avg_warmup_time_ns: int = total_warmup_time_ns / len(warmup_stats)
    avg_warmup_time_us: str = ns_to_us(avg_warmup_time_ns)

    # Identifies the median benchmark run
    sorted_benchmark_stats = sorted(benchmark_stats, key=lambda x: x.total_time)
    median_benchmark_time_ns: int = -1
    median_benchmark_stat: BenchmarkRunStatistics

    # Handles the case where there are an odd number of benchmark stats (median is the "middle" run)
    if len(sorted_benchmark_stats) % 2 == 1:
        median_benchmark_stat = sorted_benchmark_stats[len(sorted_benchmark_stats) // 2]
        median_benchmark_time_ns = median_benchmark_stat.total_time
    else:
        # Handles the case where there is an even number of benchmark stats (median is the average of the two "middle" runs)
        # NOTE In this case, while we compute the median as expected, we pick the "left middle" as the "representative"
        #   median run for extended statistic analysis
        right_middle: int = len(sorted_benchmark_stats) // 2
        left_middle: int = right_middle - 1
        left_stat: BenchmarkRunStatistics = sorted_benchmark_stats[left_middle]
        right_stat: BenchmarkRunStatistics = sorted_benchmark_stats[right_middle]
        median_benchmark_time_ns = (left_stat.total_time + right_stat.total_time) / 2
        median_benchmark_stat = left_stat

    median_benchmark_time_us: str = ns_to_us(median_benchmark_time_ns)

    # Computes the average benchmark run time and estimates initialization time
    # NOTE This pre-averages the values because I (mruberry) was concerned (almost certainly unnecessarily) about
    #   how large they could become (if there were a lot of benchmark runs, for example)
    avg_benchmark_time_ns: int = 0
    for stat in benchmark_stats:
        avg_benchmark_time_ns += stat.total_time / len(benchmark_stats)

    initialization_estimate_ns: int = (avg_warmup_time_ns - avg_benchmark_time_ns) * len(warmup_stats)
    initialization_estimate_us: str = ns_to_us(initialization_estimate_ns)

    us = "\u03BCs"

    # TODO Refactor the common preamble and then extend with extended statistics when available
    if not median_benchmark_stat.has_extended_stats:
        print(
            textwrap.dedent(
                f"""\
            {benchmark_name} benchmark results:
                The median time of {len(benchmark_stats)} benchmark iterations was {median_benchmark_time_us}.
                Constructing the callable took {callable_construction_time_us}.
                The total time taken by {len(warmup_stats)} warmup iterations was {total_warmup_time_us} (an average of {avg_warmup_time_us} per iteration).
                The estimated initialization time is {initialization_estimate_us}.
        """
            )
        )
        return

    # NOTE At this point in the program extended statistics are available
    trace_time_ns = median_benchmark_stat.last_trace_stop - median_benchmark_stat.last_trace_start
    cache_time_ns = median_benchmark_stat.last_trace_cache_stop - median_benchmark_stat.last_trace_cache_start
    tracing_time_ns = median_benchmark_stat.last_trace_tracing_stop - median_benchmark_stat.last_trace_tracing_start
    trace_execution_time_ns = (
        median_benchmark_stat.last_trace_execution_stop - median_benchmark_stat.last_trace_execution_start
    )

    trace_time_us: str = ns_to_us(trace_time_ns)
    cache_time_us: str = ns_to_us(cache_time_ns)
    tracing_time_us: str = ns_to_us(tracing_time_ns)
    trace_execution_time_us: str = ns_to_us(trace_execution_time_ns)

    cache_time_percentage: str = f"{round(cache_time_ns / trace_time_ns * 100)}%"
    tracing_time_percentage: str = f"{round(tracing_time_ns / trace_time_ns * 100)}%"
    trace_execution_time_percentage: str = f"{round(trace_execution_time_ns / trace_time_ns * 100)}%"

    print(
        textwrap.dedent(
            f"""\
        {benchmark_name} benchmark results:
            The median time of {len(benchmark_stats)} benchmark iterations was {median_benchmark_time_us}.
            Constructing the callable took {callable_construction_time_us}.
            The total time taken by {len(warmup_stats)} warmup iterations was {total_warmup_time_us} (an average of {avg_warmup_time_us} per iteration).
            The estimated initialization time is {initialization_estimate_us}.
            The median benchmark run's total time in tracing logic was {trace_time_us}.
            The median benchmark run's cache lookup time was {cache_time_us}, {cache_time_percentage} of the total time in tracing logic.
            The median benchmark run's time spent tracing was {tracing_time_us}, {tracing_time_percentage} of the total time in tracing logic.
            The median benchmark run's time to execute the traced program was {trace_execution_time_us}, {trace_execution_time_percentage} of the total time in tracing logic.
    """
        )
    )


def run_benchmark(benchmark: Benchmark, constructor: Callable, *, warmup_iters: int = 10, benchmark_iters: int = 20):
    devices: list[str] = benchmark.devices
    if len(devices) == 0:
        raise RuntimeError("Found a benchmark with no specified devices")

    # Determines the "wait for computation function," to be run after calls to make_batch() and the benchmark
    #   function to ensure that computation has finished
    wait_for_computation_fn = lambda: None
    for device in devices:
        device = thunder.core.devices.device_from_string(device)
        if device.devicetype is thunder.core.devices.DeviceType.CUDA:
            wait_for_computation_fn = wait_for_cuda_computation
            break

    # Measures the construction of the callable
    start_time: int = time.time_ns()
    benchmark_callable = constructor(benchmark.fn())
    stop_time: int = time.time_ns()
    callable_construction_time: int = stop_time - start_time

    # Creates a batch to initialize the device being used for the benchmark
    benchmark.make_batch()
    wait_for_computation_fn()

    _run_benchmark = partial(_benchmark, benchmark_callable, benchmark.make_batch, wait_for_computation_fn)

    # Performs warmup iters
    warmup_stats: list[BenchmarkRunStatistics] = _run_benchmark(warmup_iters)

    # Benchmarks
    benchmark_stats: list[BenchmarkRunStatistics] = _run_benchmark(benchmark_iters)

    # Prints statistics
    _prettyprint_stats(
        benchmark.name,
        callable_construction_time=callable_construction_time,
        warmup_stats=warmup_stats,
        benchmark_stats=benchmark_stats,
    )


# This Benchmark currently fails when run with the thunder executor, see https://github.com/Lightning-AI/lightning-thunder/issues/818
class StackedAddBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="depth",
            description="The number of additions to perform. Default is 100.",
        ),
        BenchmarkArg(
            name="shape",
            description="The shape of the both tensors. Default is (16, 16).",
        ),
        BenchmarkArg(
            name="device",
            description="A string representing the device to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype of the tensors. Default is thunder.float32.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "stacked-additions"

    @classmethod
    @property
    def description(cls) -> str:
        return "Adds two tensors [depth] times."

    @classmethod
    @property
    def args(cls):
        return cls._args

    def __init__(
        self,
        depth: int = 100,
        shape: Sequence[int] = (16, 16),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
    ) -> None:
        super().__init__()

        self.depth: int = depth
        self.shape: Sequence[int] = shape
        self.device: str = device
        self.dtype: dtypes.dtype = dtype
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(dtype)

        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        a = make_tensor(self.shape, device=self.device, dtype=self.tdtype, requires_grad=False)
        b = make_tensor(self.shape, device=self.device, dtype=self.tdtype, requires_grad=False)
        return (a, b), {"depth": self.depth}

    def fn(self) -> Callable:
        def foo(a, b, *, depth):
            for _ in range(depth):
                a = a + b

            return a

        return foo


class NanoGPTGeLUBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="shape",
            description="The shape of the input. Default is (1024, 1024).",
        ),
        BenchmarkArg(
            name="device",
            description="A string representing the device to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype of the tensors. Default is thunder.float32.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-gelu"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT's 'new GeLU' elementwise unary operation."

    @classmethod
    @property
    def args(cls):
        return cls._args

    def __init__(
        self, shape: Sequence[int] = (1024, 1024), device: str = "cuda", dtype: dtypes.dtype = thunder.float32
    ) -> None:
        super().__init__()

        self.shape: Sequence[int] = shape
        self.device: str = device
        self.dtype: dtypes.dtype = dtype
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(dtype)

        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        return (make_tensor(self.shape, device=self.device, dtype=self.tdtype, requires_grad=False),), {}

    def fn(self) -> Callable:
        def foo(a):
            return 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * torch.pow(a, 3.0))))

        return foo


def default_torch_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return fn


def default_torch_compile_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.reset()
    return torch.compile(fn)


def default_thunder_always_trace_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.compile(fn)


def default_thunder_static_caching_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.compile(fn, use_static_caching=True)


def default_thunder_last_used_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.compile(fn, use_last_executed=True)


# TODO Add descriptions to the executors when listed, and list them alphabetically
# TODO Allow querying benchmark for details
# TODO Allow specifying benchmark arguments
# TODO Move command-line processing to benchmarks.py so users call `python benchmarks.py ...`
# TODO Validate benchmark extensibility point, so benchmarks from multiple files appear as expected,
#   and/or let other benchmark files easily have a "main" function that replicates this one
# TODO Port other benchmarks
# TODO Port additional executors
# TODO Add parsing for common benchmark arguments, like shape, dtype, device, and string and integer values
# WIP -- Running benchmarks from the command line
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--listbenchmarks", action="store_true")

#     args = parser.parse_args()

#     if args.listbenchmarks:
#         list_benchmarks(use_classname=False)
#         sys.exit(0)
