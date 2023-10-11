import dataclasses
import functools
from typing import Any, Callable, Dict, List, Tuple, Optional
from collections.abc import Sequence
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
import tempfile


from lightning_utilities.core.imports import package_available

import torch
from torch.testing import make_tensor
import torch.multiprocessing as mp

import thunder
import thunder.torch as ltorch
from thunder.cudagraphs import CUDAGraphExecutor
import thunder.core.dtypes as dtypes
import thunder.core.devices as Devices
import thunder.executors as executors
from thunder.tests import nanogpt_model, hf_bart_self_attn, lit_llama_model

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


# TODO Update the metaclass to review the class and point out missing properties/mistakes (like not defining a name)
# Simple metaclass that automatically adds defined benchmarks to the list of benchmarks above
class UserFacingBenchmarkMeta(type):
    def __new__(metacls, clsname, bases, namespace):
        return super().__new__(metacls, clsname, bases, namespace)

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
            This function typically prints the benchmark's parameters.

        6) Optionally implement the postprocess_for_backward() method.

            def postprocess_for_backward(self, out: Any) -> Any:

            This will be given the output of fn(), and if it returns a torch.Tensor t that requires grad then
            the benchmark will call t.backward(torch.randn_like(t)).

            By default, postprocess_for_backward() returns the output of fn(), or the first element of
            the output of fn() if fn() returns a Sequence.

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

    def postprocess_for_backward(self, out: Any) -> Any:
        if isinstance(out, Sequence):
            return out[0]

        return out


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
    start_time: int
    stop_time: int
    host_stop_time: int
    called_backward: bool
    has_extended_stats: bool = False
    last_trace_host_start: int = -1
    last_trace_host_stop: int = -1
    last_trace_cache_start: int = -1
    last_trace_cache_stop: int = -1
    last_trace_tracing_start: int = -1
    last_trace_tracing_stop: int = -1
    last_trace_host_execution_start: int = -1
    last_trace_host_execution_stop: int = -1


# A timing helper
def _benchmark(
    benchmark: Benchmark, fn: Callable, wait_for_computation: Callable, repetitions: int
) -> list[BenchmarkRunStatistics]:
    stats = []
    for _ in range(repetitions):
        args, kwargs = benchmark.make_batch()
        wait_for_computation()
        called_backward: bool = False
        start: int = time.time_ns()
        result = fn(*args, **kwargs)

        # Calls backward, if the output requires grad
        # TODO Consider zeroing the grad when generating a batch
        grad_tensor = benchmark.postprocess_for_backward(result)
        if grad_tensor is not None and isinstance(grad_tensor, torch.Tensor) and grad_tensor.requires_grad:
            grad_tensor.backward(torch.randn_like(grad_tensor))
            called_backward = True
        host_stop: int = time.time_ns()
        wait_for_computation()
        stop: int = time.time_ns()

        cs = thunder.compile_stats(fn)
        stat = BenchmarkRunStatistics(
            total_time=(stop - start),
            start_time=start,
            stop_time=stop,
            host_stop_time=host_stop,
            called_backward=called_backward,
        )

        # TODO Ensure the compile data statistics are always populated
        if cs is not None and cs.last_trace_host_start > 0:
            stat.has_extended_stats = True
            stat.last_trace_host_start = cs.last_trace_host_start
            stat.last_trace_host_stop = cs.last_trace_host_stop
            stat.last_trace_cache_start = cs.last_trace_cache_start
            stat.last_trace_cache_stop = cs.last_trace_cache_stop
            stat.last_trace_tracing_start = cs.last_trace_tracing_start
            stat.last_trace_tracing_stop = cs.last_trace_tracing_stop
            stat.last_trace_host_execution_start = cs.last_trace_host_execution_start
            stat.last_trace_host_execution_stop = cs.last_trace_host_execution_stop

        stats.append(stat)

    return stats


# TODO Consider filling a large tensor with zeros in an attempt to prevent caching
def wait_for_cuda_computation() -> None:
    torch.cuda.synchronize()


# Prints nanoseconds as microseconds, rounded
def ns_to_us(ns: Number) -> str:
    us = "\u03BCs"
    return f"{round(ns / 1000):.2e}{us}"


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
    total_warmup_time_ns: int = sum(stat.total_time for stat in warmup_stats)
    total_warmup_time_us: str = ns_to_us(total_warmup_time_ns)

    # Computes average warmup time
    avg_warmup_time_ns: float = total_warmup_time_ns / len(warmup_stats)
    avg_warmup_time_us: str = ns_to_us(avg_warmup_time_ns)

    # Identifies the median benchmark run
    sorted_benchmark_stats = sorted(benchmark_stats, key=lambda x: x.total_time)
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
        median_benchmark_time_ns: int = (left_stat.total_time + right_stat.total_time) // 2
        median_benchmark_stat = left_stat

    median_benchmark_time_us: str = ns_to_us(median_benchmark_time_ns)

    # Computes the average benchmark run time and estimates initialization time
    total_backward_calls: int = sum(stat.called_backward for stat in benchmark_stats)
    total_benchmark_time_ns: int = sum(stat.total_time for stat in benchmark_stats)

    avg_benchmark_time_ns = total_benchmark_time_ns / len(benchmark_stats)

    initialization_estimate_ns: float = (avg_warmup_time_ns - avg_benchmark_time_ns) * len(warmup_stats)
    initialization_estimate_us: str = ns_to_us(initialization_estimate_ns)

    total_initialization_time_ns: float = callable_construction_time + initialization_estimate_ns
    total_initialization_time_us: str = ns_to_us(total_initialization_time_ns)
    callable_construction_percentage: str = f"{round(callable_construction_time / total_initialization_time_ns * 100)}%"
    initialization_percentage: str = f"{round(initialization_estimate_ns / total_initialization_time_ns * 100)}%"

    total_time_ns: int = total_warmup_time_ns + total_benchmark_time_ns
    total_time_us: str = ns_to_us(total_time_ns)

    total_host_time_ns: int = median_benchmark_stat.host_stop_time - median_benchmark_stat.start_time
    total_host_time_us: str = ns_to_us(total_host_time_ns)
    host_time_percentage: str = f"{round(total_host_time_ns / median_benchmark_stat.total_time * 100)}%"

    preamble = f"""\
    {benchmark_name} benchmark results:
        The median time of {len(benchmark_stats)} benchmark iterations is {median_benchmark_time_us}.
        The estimated callable construction and initialization time is {total_initialization_time_us}.
        The median benchmark run's host time is {total_host_time_us}, {host_time_percentage} of the total time.
        Constructing the callable took {callable_construction_time_us}, {callable_construction_percentage} of the total construction and initialization time.
        The estimated initialization time is {initialization_estimate_us}, {initialization_percentage} of the total construction and initialization time.
        The total time taken by {len(warmup_stats)} warmup iterations is {total_warmup_time_us} (an average of {avg_warmup_time_us} per iteration).
        The total time to run all the iterations (warmup and benchmark) was is {total_time_us}.
        The benchmark called backward() {total_backward_calls} times.
    """
    if median_benchmark_stat.has_extended_stats:
        # NOTE At this point in the program extended statistics are available
        trace_time_ns = median_benchmark_stat.last_trace_host_stop - median_benchmark_stat.last_trace_host_start
        cache_time_ns = median_benchmark_stat.last_trace_cache_stop - median_benchmark_stat.last_trace_cache_start
        tracing_time_ns = median_benchmark_stat.last_trace_tracing_stop - median_benchmark_stat.last_trace_tracing_start
        trace_execution_time_ns = (
            median_benchmark_stat.last_trace_host_execution_stop - median_benchmark_stat.last_trace_host_execution_start
        )

        trace_time_us: str = ns_to_us(trace_time_ns)
        cache_time_us: str = ns_to_us(cache_time_ns)
        tracing_time_us: str = ns_to_us(tracing_time_ns)
        trace_execution_time_us: str = ns_to_us(trace_execution_time_ns)

        trace_time_percentage: str = f"{round(trace_time_ns / median_benchmark_stat.total_time * 100)}%"
        cache_time_percentage: str = f"{round(cache_time_ns / median_benchmark_stat.total_time * 100)}%"
        tracing_time_percentage: str = f"{round(tracing_time_ns / median_benchmark_stat.total_time * 100)}%"
        trace_execution_time_percentage: str = (
            f"{round(trace_execution_time_ns / median_benchmark_stat.total_time * 100)}%"
        )

        before_trace_time_ns = median_benchmark_stat.last_trace_host_start - median_benchmark_stat.start_time
        accelerator_wait_time_ns = median_benchmark_stat.stop_time - median_benchmark_stat.last_trace_host_stop

        before_trace_time_us: str = ns_to_us(before_trace_time_ns)
        accelerator_wait_time_us: str = ns_to_us(accelerator_wait_time_ns)

        before_trace_time_percentage: str = f"{round(before_trace_time_ns / median_benchmark_stat.total_time * 100)}%"
        accelerator_wait_time_percentage: str = f"{round(accelerator_wait_time_ns / median_benchmark_stat.total_time * 100)}%"

        extension = f"""\
            The median benchmark took {before_trace_time_us} to get into the tracing logic, {before_trace_time_percentage} of the total time.
            The median benchmark took {accelerator_wait_time_us} waiting for the accelerator's computation to finish, {accelerator_wait_time_percentage} of the total time.
            The median benchmark run's total time in tracing logic is {trace_time_us}, {trace_time_percentage} of the total time.
            The median benchmark run's cache lookup time is {cache_time_us}, {cache_time_percentage} of the total time.
            The median benchmark run's time spent tracing is {tracing_time_us}, {tracing_time_percentage} of the total time.
            The median benchmark run's time to request the traced program be executed is {trace_execution_time_us}, {trace_execution_time_percentage} of the total time.
        """
    else:
        extension = ""

    output = textwrap.dedent(preamble) + textwrap.indent(textwrap.dedent(extension), " " * 4)
    print(output)


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message)
    else:
        print(message)


# TODO Consider isolating each benchmark run in a subprocess to avoid cache reuse across benchmarks
#   (which has been observed, to get around this just run one benchmark at a time)
def _run_benchmark(
    benchmark: Benchmark, constructor: Callable, *, warmup_iters: int = 10, benchmark_iters: int = 20
) -> tuple[int, list, list]:
    # Determines the "wait for computation function," to be run after calls to make_batch() and the benchmark
    #   function to ensure that computation has finished
    devices: list[str] = benchmark.devices
    wait_for_computation_fn = lambda: None
    for device in devices:
        device: thunder.core.devices.Device = thunder.core.devices.device_from_string(device)
        if device.devicetype is thunder.core.devices.DeviceType.CUDA:
            wait_for_computation_fn = wait_for_cuda_computation
            break

    # Creates a batch to initialize the device being used for the benchmark
    benchmark.make_batch()
    wait_for_computation_fn()

    # Measures the construction of the callable
    # NOTE Callable construction probably doesn't use an accelerator, but this waits for the accelerator
    #   to finish its work just incase
    benchmark_fn = benchmark.fn()
    wait_for_computation_fn()
    start_time: int = time.time_ns()
    benchmark_callable = constructor(benchmark_fn)
    wait_for_computation_fn()
    stop_time: int = time.time_ns()
    callable_construction_time: int = stop_time - start_time

    my_benchmark = partial(_benchmark, benchmark, benchmark_callable, wait_for_computation_fn)

    # Performs warmup iters
    warmup_stats: list[BenchmarkRunStatistics] = my_benchmark(warmup_iters)

    # Benchmarks
    benchmark_stats: list[BenchmarkRunStatistics] = my_benchmark(benchmark_iters)

    return callable_construction_time, warmup_stats, benchmark_stats


def run_benchmark(
    benchmark: Benchmark, constructor: Callable, *, warmup_iters: int = 10, benchmark_iters: int = 20
) -> None:
    print(f"Running benchmark {benchmark.name}")
    _print_benchmark_arguments(benchmark)

    devices: list[str] = benchmark.devices
    if len(devices) == 0:
        raise RuntimeError("Found a benchmark with no specified devices")

    callable_construction_time, warmup_stats, benchmark_stats = _run_benchmark(
        benchmark, constructor, warmup_iters=warmup_iters, benchmark_iters=benchmark_iters
    )

    _prettyprint_stats(
        benchmark.name,
        callable_construction_time=callable_construction_time,
        warmup_stats=warmup_stats,
        benchmark_stats=benchmark_stats,
    )


# TODO Extend this to work with CPU devices, too
def ddp_runner(args):
    init_method, world_size, rank, benchmark, ddp_constructor, warmup_iters, benchmark_iters = args

    torch.distributed.init_process_group(
        init_method=init_method,
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    benchmark.device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    stats = _run_benchmark(benchmark, ddp_constructor(rank), warmup_iters=warmup_iters, benchmark_iters=benchmark_iters)
    return rank, stats


# TODO Consider muting processes other than rank 0 by redirecting their stdout to devnull
def run_multiprocess_benchmark(
    benchmark: Benchmark,
    ddp_constructor: Callable,
    *,
    world_size: int = 2,
    warmup_iters: int = 10,
    benchmark_iters: int = 20,
):
    print(f"Running distributed benchmark {benchmark.name} with {world_size=}")
    _print_benchmark_arguments(benchmark)

    assert (
        torch.distributed.is_available()
    ), f"Trying to run a distributed benchmark, but torch.distributed is not available"

    # Ensures the benchmark is running on a single CUDA device (which is overridden later)
    assert (
        len(benchmark.devices) == 1
        and Devices.device_from_string(benchmark.devices[0]).devicetype == Devices.DeviceType.CUDA
    ), f"Distributed benchmarking currently only supports benchmarks that run on a single CUDA device"

    # Ensures the benchmark returns a module (because ddp is only supported on modules)
    benchmark_fn = benchmark.fn()
    assert isinstance(
        benchmark_fn, torch.nn.Module
    ), f"Distributed benchmarking currently only supports module benchmarks"

    # Validates world size
    assert (
        world_size <= torch.cuda.device_count()
    ), f"Requested world size of {world_size} is greater than the number of available cuda devices {torch.cuda.device_count()}"

    FILE_SCHEMA: str = "file://"
    if sys.platform == "win32":
        FILE_SCHEMA = "file:///"
    file_name = tempfile.NamedTemporaryFile(delete=False).name
    init_method = f"{FILE_SCHEMA}{file_name}"

    input_data = [
        (init_method, world_size, rank, benchmark, ddp_constructor, warmup_iters, benchmark_iters)
        for rank in range(world_size)
    ]

    from concurrent.futures import ProcessPoolExecutor as Pool

    # NOTE This uses the ProcessPoolExecutor because that allows spawning processes within worker threads
    #   which dynamo relies on
    # TODO Consider adding a timeout (possibly configurable when calling run_multiprocess_benchmark())
    # TODO In Python 3.11+ ProcessPoolExecutor has the max_tasks_per_child parameter
    #   which this should set to 1
    # TODO Consider defining our own multiprocessing pool that uses max_tasks_per_child and supports dynamo
    try:
        pool = Pool(mp_context=mp.get_context("spawn"))
        results = pool.map(ddp_runner, input_data)

        # Aggregates statistics
        total_cct: int = 0
        all_warmup_stats = []
        all_benchmark_stats = []
        for rank, (callable_construction_time, warmup_stats, benchmark_stats) in results:
            total_cct += callable_construction_time
            all_warmup_stats.extend(warmup_stats)
            all_benchmark_stats.extend(benchmark_stats)

        avg_cct: int = total_cct // world_size

        _prettyprint_stats(
            benchmark_name=f"{benchmark.name}-ddp",
            callable_construction_time=avg_cct,
            warmup_stats=all_warmup_stats,
            benchmark_stats=all_benchmark_stats,
        )
    finally:
        pool.shutdown()


#
# Common executors (defined here for convenience)
#


def default_torch_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return fn


def default_torch_ddp_executor(_) -> Callable:
    def func(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.nn.parallel.DistributedDataParallel(fn)

    return func


def default_torch_compile_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.reset()
    return torch.compile(fn)


def default_torch_compile_ddp_executor(_) -> Callable:
    def func(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch._dynamo.reset()
        return torch.compile(torch.nn.parallel.DistributedDataParallel(fn))

    return func


def default_thunder_always_trace_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.compile(fn, cache_mode="always trace")


def default_thunder_dynamic_strides_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.compile(fn, cache_mode="dynamic strides")


def default_thunder_ddp_dynamic_strides_executor(rank) -> Callable:
    from thunder.distributed import ddp

    def func(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return thunder.compile(
            ddp(fn, rank, broadcast_from=0),
            cache_mode="dynamic strides",
        )

    return func


def default_thunder_dynamic_strides_executor_no_grad(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.compile(fn, disable_torch_autograd_support=True)


def default_thunder_fixed_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.compile(fn, cache_mode="fixed")


# TODO Add grad support
def default_thunder_triton_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True

    TRITON_AVAILABLE = package_available("triton")
    assert TRITON_AVAILABLE, "Trying to benchmark with the thunder+triton executor, but triton is not available"

    from thunder.executors.triton_crossentropy import register_triton_entropyex

    register_triton_entropyex(add_to_default_executors=False)

    executors_list = ("triton_crossentropy", executors.NVFUSER, executors.TORCH)

    return thunder.compile(fn, executors_list=executors_list, disable_torch_autograd_support=True)


# TODO Add grad support
def default_thunder_apex_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True

    APEX_CROSS_ENTROPY_AVAILABLE = package_available("xentropy_cuda")
    assert APEX_CROSS_ENTROPY_AVAILABLE, "Trying to benchmark with the thunder+apex executor, but apex is not available"

    from thunder.executors.apex_entropyex import register_apex_entropyex

    register_apex_entropyex(add_to_default_executors=False)

    executors_list = ("apex_xentropy", executors.NVFUSER, executors.TORCH)
    return thunder.compile(fn, executors_list=executors_list, disable_torch_autograd_support=True)


# TODO Add grad support
def default_thunder_cudnn_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True

    CUDNN_AVAILABLE = package_available("cudnn")
    assert CUDNN_AVAILABLE, "Trying to benchmark with the thunder+cudnn executor, but cudnn is not available"

    from thunder.executors.cudnnex import register_cudnnex

    register_cudnnex(add_to_default_executors=False)

    executors_list = ("cudnn", executors.NVFUSER, executors.TORCH)
    return thunder.compile(fn, executors_list=executors_list, disable_torch_autograd_support=True)


# TODO Add grad support
def default_thunder_cudagraphs_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True

    executors_list = []

    # Adds the Apex executor, if available
    APEX_CROSS_ENTROPY_AVAILABLE = package_available("xentropy_cuda")
    if APEX_CROSS_ENTROPY_AVAILABLE:
        from thunder.executors.apex_entropyex import register_apex_entropyex

        register_apex_entropyex(add_to_default_executors=False)
        executors_list.append("apex_xentropy")

    executors_list.extend((executors.NVFUSER, executors.TORCH))
    return thunder.compile(
        fn,
        executors_list=executors_list,
        use_cudagraphs=True,
        disable_torch_autograd_support=True,
    )


#
# Benchmarks
#
# TODO Document a pattern to define benchmarks in another file


def _print_benchmark_arguments(bmark: Benchmark) -> None:
    print(f"{bmark.name} benchmark parameters:")
    for arg in bmark.args:
        print(f"\t{arg.name}={getattr(bmark, arg.name)}")


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
        BenchmarkArg(
            name="requires_grad",
            description="Whether the input tensors require grad. Default is False.",
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
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        depth: int = 100,
        shape: Sequence[int] = (16, 16),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.depth: int = depth
        self.shape: Sequence[int] = shape
        self.device: str = device
        self.dtype: dtypes.dtype = dtype
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(dtype)
        self.requires_grad: bool = requires_grad

        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        a = make_tensor(self.shape, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
        b = make_tensor(self.shape, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
        return (a, b), {
            "depth": self.depth,
        }

    def fn(self) -> Callable:
        def foo(a, b, *, depth):
            for _ in range(depth):
                a = a + b

            return a

        return foo


class ReshapeViewBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="depth",
            description="The number of reshape sequences (int) to perform. Default is 100.",
        ),
        BenchmarkArg(
            name="device",
            description="The device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (torch.dtype, thunder.dtypes.dtype) of the the input tensor. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the input tensors require grad. Default is False.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "reshape-view"

    @classmethod
    @property
    def description(cls) -> str:
        return "Performs many reshape (view) operations on the same tensor."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        depth: int = 100,
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.depth: int = depth
        self.device: str = device
        self.dtype: dtypes.dtype = dtype
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(dtype)
        self.requires_grad: bool = requires_grad

        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        shape = (16, 16, 16, 16, 16)
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
        return (make(shape),), {"depth": self.depth}

    def fn(self) -> Callable:
        def foo(a: torch.Tensor, *, depth: int):
            for _ in range(depth):
                a = a.reshape(32, 32, 8, 4, 2, 16)
                a = a.reshape(1024, 2, 32, 2, 8)
                a = a.reshape(4, 4, 4, 4, 4, 4, 4, 4, 4, 4)
                a = a.reshape(2, 2, 16, 8, 4, 2, 1, 32, 2, 2, 2)
                a = a.reshape(16, 65536)
            return a

        return foo


#
# NanoGPT benchmarks
#


# Taken from nanogpt_model.py
@dataclasses.dataclass
class NanoGPTConfig:
    block_size: int = 1024
    seq_len: int = 128
    # NOTE The original GPT-2 vocab_size was 50257, but recent nanoGPT uses 50304 for GPU performance
    # https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L111
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    def update(self, **kwargs) -> None:
        for field in dataclasses.fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])


_nanogpt_configs = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}


# Prints a NanoGPT config (useful for understanding what the NanoGPT benchmarks are doing)
def _print_nanogpt_config(cfg: NanoGPTConfig) -> None:
    print("NanoGPT Config:")
    for field in dataclasses.fields(cfg):
        print(f"\t{field.name}={getattr(cfg, field.name)}")


def _extract_nanogpt_config(config: str | NanoGPTConfig):
    if isinstance(config, NanoGPTConfig):
        return config

    assert isinstance(config, str), "Expected the configuration to be a NanoGPTConfig object or a string"
    assert config in _nanogpt_configs, f"Expected {config=} to be in {_nanogpt_configs.keys()}"

    result: NanoGPTConfig = NanoGPTConfig()
    result.update(**_nanogpt_configs[config])
    return result


class NanoGPTGeLUBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len,). Default is (16,).",
        ),
        BenchmarkArg(
            name="device",
            description="A string representing the device to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype of the tensors. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the input tensors require grad. Default is False.",
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
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        gpt_config = _extract_nanogpt_config(config)
        self.shape: Sequence[int] = batchdims + (gpt_config.seq_len, 4 * gpt_config.n_embd)
        self.device: str = device
        self.dtype: dtypes.dtype = dtype
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(dtype)
        self.requires_grad: bool = (requires_grad,)

        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        return (make_tensor(self.shape, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad),), {}

    def fn(self) -> Callable:
        def foo(a):
            return torch.nn.functional.gelu(a, approximate="tanh")

        return foo


class NanoGPTBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len,). Default is (16,).",
        ),
        BenchmarkArg(
            name="indices_dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and targets. Default is thunder.int64.",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT with targets."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        indices_dtype: dtypes.dtype = thunder.int64,
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.indices_dtype = indices_dtype
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.indices_tdtype: torch.dtype = ltorch.to_torch_dtype(self.indices_dtype)
        self.model_tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, low=0, high=255, device=self.device, dtype=self.indices_tdtype, requires_grad=False)
        shape = self.batchdims + (self.config.seq_len,)

        x = make(shape)
        targets = make(shape)
        return (x, targets), {}

    def fn(self) -> Callable:
        gpt = (
            nanogpt_model.GPT(self.config)
            .to(device=self.device, dtype=self.model_tdtype)
            .requires_grad_(self.requires_grad)
        )
        return gpt

    def postprocess_for_backward(self, output: tuple[torch.Tensor, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.requires_grad:
            return None
        logits, loss = output
        return loss


class NanoGPTCrossEntropyBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len, config.vocab_size). Default is (16,).",
        ),
        BenchmarkArg(
            name="indices_dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and targets. Default is thunder.int64.",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the logits. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the logits require grad. Default is False.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-cross-entropy"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT Cross Entropy."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        indices_dtype: dtypes.dtype = thunder.int64,
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.indices_dtype = indices_dtype
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.indices_tdtype: torch.dtype = ltorch.to_torch_dtype(self.indices_dtype)
        self.logits_dtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, low=0, high=255, device=self.device, requires_grad=self.requires_grad)

        logits_shape = self.batchdims + (self.config.seq_len, self.config.vocab_size)
        logits = make(logits_shape, dtype=self.logits_dtype)

        targets_shape = self.batchdims + (self.config.seq_len,)
        targets = make(targets_shape, dtype=self.indices_tdtype, requires_grad=False)

        return (logits, targets), {}

    def fn(self) -> Callable:
        def foo(logits, targets):
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return foo


class NanoGPTCSABenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len, config.n_embd). Default is (16,).",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-csa"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT's Causal Selft Attention (CSA) module."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=False)
        shape = self.batchdims + (self.config.seq_len, self.config.n_embd)

        return (make(shape),), {}

    def fn(self) -> Callable:
        gpt_csa = (
            nanogpt_model.CausalSelfAttention(self.config)
            .to(device=self.device, dtype=self.tdtype)
            .requires_grad_(self.requires_grad)
        )
        return gpt_csa


class NanoGPTBlockBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len, config.n_embd). Default is (16,).",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-block"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT's Block module."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=False)
        shape = self.batchdims + (self.config.seq_len, self.config.n_embd)

        return (make(shape),), {}

    def fn(self) -> Callable:
        gpt_block = (
            nanogpt_model.Block(self.config)
            .to(device=self.device, dtype=self.tdtype)
            .requires_grad_(self.requires_grad)
        )
        return gpt_block


class NanoGPTBlockLoopBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len, config.n_embd). Default is (16,).",
        ),
        BenchmarkArg(
            name="depth", description="The number (int) of block modules to run in sequence. Default is config.n_layer."
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-block-loop"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT's Block module run sequentially."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        depth: None | int = None,
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.depth: int = depth if depth is not None else self.config.n_layer
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=False)
        shape = self.batchdims + (self.config.seq_len, self.config.n_embd)

        return (make(shape),), {}

    def fn(self) -> Callable:
        class nanoGPTBlockLoop(torch.nn.Module):
            def __init__(slf):
                super().__init__()
                slf.transformer = torch.nn.ModuleDict(
                    dict(
                        drop=torch.nn.Dropout(self.config.dropout),
                        h=torch.nn.ModuleList([nanogpt_model.Block(self.config) for _ in range(self.depth)]),
                        ln_f=torch.nn.LayerNorm(self.config.n_embd),
                    ),
                )

            def forward(self, x):
                x = self.transformer.drop(x)
                for block in self.transformer.h:
                    x = block(x)
                x = self.transformer.ln_f(x)
                return x

        module = nanoGPTBlockLoop().to(device=self.device, dtype=self.tdtype).requires_grad_(self.requires_grad)
        return module


class NanoGPTMLPBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len, config.n_embd). Default is (16,).",
        ),
        BenchmarkArg(
            name="device",
            description="The device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-mlp"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT's MLP module."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=False)
        shape = self.batchdims + (self.config.seq_len, self.config.n_embd)

        return (make(shape),), {}

    def fn(self) -> Callable:
        gpt_mlp = (
            nanogpt_model.MLP(self.config).to(device=self.device, dtype=self.tdtype).requires_grad_(self.requires_grad)
        )
        return gpt_mlp


class NanoGPTLayerNormBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len, config.n_embd). Default is (16,).",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-layernorm"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT's layer norm operation."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=False)
        shape = self.batchdims + (self.config.seq_len, self.config.n_embd)

        return (make(shape),), {}

    def fn(self) -> Callable:
        class nanoGPTLayerNorm(torch.nn.Module):
            def __init__(slf):
                super().__init__()
                slf.ln = torch.nn.LayerNorm(self.config.n_embd)

            def forward(slf, x):
                return slf.ln(x)

        layernorm_module = (
            nanoGPTLayerNorm().to(device=self.device, dtype=self.tdtype).requires_grad_(self.requires_grad)
        )
        return layernorm_module


class NanoGPTEmbeddingBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len,). Default is (16,).",
        ),
        BenchmarkArg(
            name="indices_dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input. Default is thunder.int64.",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-embedding"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT's embedding operations (followed by dropout and layernorm)."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        indices_dtype: dtypes.dtype = thunder.int64,
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.indices_dtype = indices_dtype
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tindices_dtype: torch.dtype = ltorch.to_torch_dtype(self.indices_dtype)
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        shape = self.batchdims + (self.config.seq_len,)
        idx = make_tensor(shape, low=0, high=255, device=self.device, dtype=self.tindices_dtype)

        return (idx,), {}

    def fn(self) -> Callable:
        class nanoGPTEmbedding(torch.nn.Module):
            def __init__(slf):
                super().__init__()
                slf.wte = torch.nn.Embedding(self.config.vocab_size, self.config.n_embd)
                slf.wpe = torch.nn.Embedding(self.config.block_size, self.config.n_embd)
                slf.drop = torch.nn.Dropout(self.config.dropout)
                slf.ln = torch.nn.LayerNorm(self.config.n_embd)

            def forward(slf, idx):
                # This is a part of the GPT's forward pass before the transformer block
                device = idx.device
                b, t = idx.shape
                pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
                tok_emb = slf.wte(idx)  # token embeddings of shape (b, t, n_embd)
                pos_emb = slf.wpe(pos)  # position embeddings of shape (1, t, n_embd)
                x = slf.drop(tok_emb + pos_emb)
                # LayerNorm is the first operation in nanoGPT's Block before the attention
                return slf.ln(x)

        module = nanoGPTEmbedding().to(device=self.device, dtype=self.tdtype).requires_grad_(self.requires_grad)
        return module


class NanoGPTSDPABenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The nanoGPT config (str, NanoGPTConfig) to use. String options are 'gpt2', 'gpt2-medium', 'gpt2-large', and 'gpt2-xl'. Default is 'gpt2-medium'. See the NanoGPT model for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The inputs will have innermost dimensions of (config.n_head, config.seq_len, config.n_embd). Default is (16,).",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the input requires grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "nanogpt-sdpa"

    @classmethod
    @property
    def description(cls) -> str:
        return "NanoGPT's Scaled Dot Product Attention call."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | NanoGPTConfig = "gpt2-medium",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
        shape = self.batchdims + (self.config.n_head, self.config.seq_len, self.config.n_embd // self.config.n_head)

        q = make(shape)
        k = make(shape)
        v = make(shape)

        return (q, k, v), {"dropout": self.config.dropout}

    def fn(self) -> Callable:
        class nanoGPTScaledDotProductAttention(torch.nn.Module):
            def __init__(slf):
                super().__init__()

            def forward(slf, q, k, v, *, dropout):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True
                )

        return nanoGPTScaledDotProductAttention()


# Taken from HuggingFace Bart-Large model config:
# https://huggingface.co/facebook/bart-large/blob/main/config.json
class HuggingFaceSelfAttnBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="embed_dim",
            description="The number (int) of embedded dimensions. Default is 1024.",
        ),
        BenchmarkArg(
            name="num_heads",
            description="The number (int) of heads. Default is 16.",
        ),
        BenchmarkArg(
            name="sequences",
            description="The number (int) of sequences to execute. Default is 8.",
        ),
        BenchmarkArg(
            name="seq_length",
            description="The length (int) of each sequence. The input will have dimensions (sequences, seq_length, embed_dim). Default is 1024.",
        ),
        BenchmarkArg(
            name="dropout",
            description="The dropout likelihood (float). Default is .1.",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "hf-bart"

    @classmethod
    @property
    def description(cls) -> str:
        return "Huggingface's Bart large config."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        sequences: int = 8,
        seq_length: int = 1024,
        dropout: float = 0.1,
        device: str = "cuda",
        dtype: thunder.dtypes.dtype | torch.dtype | str = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sequences = sequences
        self.seq_length = seq_length
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=False)

        a = make((self.sequences, self.seq_length, self.embed_dim))
        b = make((self.sequences, 1, 1, self.seq_length))

        return (a, b), {}

    def fn(self) -> Callable:
        bart_model = (
            hf_bart_self_attn.BartAttention(
                self.embed_dim,
                self.num_heads,
                dropout=self.dropout,
            )
            .to(device=self.device, dtype=self.tdtype)
            .requires_grad_(self.requires_grad)
        )

        return bart_model


class LLaMABlockBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _config_map = {
        "7B": lit_llama_model.LLaMAConfig.from_name("7B"),
    }

    _args = (
        BenchmarkArg(
            name="config",
            description="The configuration (str) to use. Options are '7B'. Default is '7B'.",
        ),
        BenchmarkArg(
            name="sequences",
            description="The number (int) of sequences to execute. Default is 8.",
        ),
        BenchmarkArg(
            name="seq_length",
            description="The length (int) of each sequence. The input will have dimensions (sequences, seq_length, embed_dim). Default is 1024.",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "llama-block"

    @classmethod
    @property
    def description(cls) -> str:
        return "Lit-llama's block module"

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str = "7B",
        sequences: int = 8,
        seq_length: int = 1024,
        device: str = "cuda",
        dtype: thunder.dtypes.dtype | torch.dtype | str = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = self._config_map[config]
        self.sequences = sequences
        self.seq_length = seq_length
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=False)

        a = make((self.sequences, self.seq_length, self.config.n_embd))

        return (a,), {}

    def fn(self) -> Callable:
        model = (
            lit_llama_model.Block(self.config)
            .to(device=self.device, dtype=self.tdtype)
            .requires_grad_(self.requires_grad)
        )
        return model


# TODO Add descriptions to the executors when listed, and list them alphabetically
# TODO Allow querying benchmark for details
# TODO Allow specifying benchmark arguments
# TODO Move command-line processing to benchmarks.py so users call `python benchmarks.py ...`
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
