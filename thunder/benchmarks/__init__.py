import dataclasses
from typing import Any
from collections.abc import Callable
from collections.abc import Sequence
import time
from functools import partial
import textwrap
from numbers import Number
import sys
from dataclasses import dataclass
import tempfile


from lightning_utilities.core.imports import package_available

import torch
import torch.nn as nn
from torch.testing import make_tensor
import torch.multiprocessing as mp

import thunder
import thunder.torch as ltorch
import thunder.core.dtypes as dtypes
import thunder.core.devices as Devices
from thunder.core.transforms import grad, clear_grads, populate_grads
import thunder.executors as executors
from thunder.tests import nanogpt_model, hf_bart_self_attn, litgpt_model
from thunder.tests.make_tensor import make_tensor, make_tensor_like
from thunder.tests.litgpt_model import Config as LitGPTConfig

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
    benchmark: Benchmark,
    fn: Callable,
    wait_for_computation: Callable,
    repetitions: int,
    *,
    use_grad_transform: bool = False,
    compile_backward: bool = False,
) -> list[BenchmarkRunStatistics]:
    stats = []
    for _ in range(repetitions):
        # TODO - set grads to none
        args, kwargs = benchmark.make_batch()
        wait_for_computation()
        called_backward: bool = False
        start: int = time.time_ns()
        result = fn(*args, **kwargs)
        if compile_backward:
            # NOTE In this case backward has been compiled, so nothing more to be done
            pass
        elif use_grad_transform:
            # This populates the grads on the module (even though they're cleared in a moment)
            #   because calling backward() in PyTorch populates the grads, so for a far comparison
            #   the benchmark needs to account for that happening
            if isinstance(fn, torch.nn.Module):
                populate_grads(result, fn)
            else:
                populate_grads(result, args=args, kwargs=kwargs)
        else:
            # Calls backward, if the output requires grad
            grad_tensor = benchmark.postprocess_for_backward(result)
            if grad_tensor is not None and isinstance(grad_tensor, torch.Tensor) and grad_tensor.requires_grad:
                grad_tensor.backward(torch.ones_like(grad_tensor))
                called_backward = True

        host_stop: int = time.time_ns()
        wait_for_computation()
        stop: int = time.time_ns()
        if isinstance(fn, torch.nn.Module):
            clear_grads(fn)

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
    extended_printout: bool = True,
    rank_mem_info: dict[int, tuple[int, int]] = {},
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

    if not extended_printout:
        short_printout = f"""\
        {benchmark_name} benchmark results:
            The median time of {len(benchmark_stats)} benchmark iterations is {median_benchmark_time_us}.
        """
        if rank_mem_info:
            short_printout += "\n    " + "*" * 20 + " Memory Usage " + "*" * 20
            for rank, (memory_allocated, memory_reserved) in rank_mem_info.items():
                short_printout += f"\n    rank-{rank} - peak allocated memory {memory_allocated/1024/1024:.2f}MB, peak reserved: {memory_reserved/1024/1024:.2f}MB"
            short_printout += "\n"

        print(short_printout)
        return

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
    if rank_mem_info:
        short_printout += "\n    " + "*" * 20 + " Memory Usage " + "*" * 20
        for rank, (memory_allocated, memory_reserved) in rank_mem_info.items():
            short_printout += f"\n    rank-{rank} - peak allocated memory {memory_allocated/1024/1024:.2f}MB, peak reserved: {memory_reserved/1024/1024:.2f}MB"
        short_printout += "\n"
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
        accelerator_wait_time_percentage: str = (
            f"{round(accelerator_wait_time_ns / median_benchmark_stat.total_time * 100)}%"
        )

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
    benchmark: Benchmark,
    constructor: Callable,
    *,
    warmup_iters: int = 10,
    benchmark_iters: int = 20,
    use_grad_transform: bool = False,
    compile_backward: bool = False,
) -> tuple[int, list, list, int, int]:
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

    assert not use_grad_transform or not compile_backward, "Can't set both use_grad_transform and compile_backward!"
    if use_grad_transform:
        from thunder.core.transforms import _grad_specifier_default

        def grad_specifier(outs) -> None:
            grad_tensor = benchmark.postprocess_for_backward(outs)
            _grad_specifier_default(grad_tensor)

        benchmark_callable = constructor(benchmark_fn)
        benchmark_callable = grad(benchmark_callable, grad_specifier=grad_specifier)
    elif compile_backward:

        def _fn(*args, **kwargs):
            result = benchmark_fn(*args, **kwargs)
            grad_tensor = benchmark.postprocess_for_backward(result)
            grad_tensor.backward(torch.ones_like(grad_tensor))

        benchmark_callable = constructor(_fn)
    else:
        benchmark_callable = constructor(benchmark_fn)

    wait_for_computation_fn()
    stop_time: int = time.time_ns()
    callable_construction_time: int = stop_time - start_time
    my_benchmark = partial(
        _benchmark,
        benchmark,
        benchmark_callable,
        wait_for_computation_fn,
        use_grad_transform=use_grad_transform,
        compile_backward=compile_backward,
    )

    # Performs warmup iters
    warmup_stats: list[BenchmarkRunStatistics] = my_benchmark(warmup_iters)

    # Benchmarks
    cur_dev = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(cur_dev)
    benchmark_stats: list[BenchmarkRunStatistics] = my_benchmark(benchmark_iters)
    memory_stats = torch.cuda.memory_stats(cur_dev)
    memory_allocated = memory_stats["allocated_bytes.all.peak"]
    memory_reserved = memory_stats["reserved_bytes.all.peak"]

    return callable_construction_time, warmup_stats, benchmark_stats, memory_allocated, memory_reserved


# TODO Support for grad transforming the benchmarks is currently a prototype
#   Reconcile the grad transform and calling torch.autograd.grad so they can be directly compared
def run_benchmark(
    benchmark: Benchmark,
    constructor: Callable,
    *,
    warmup_iters: int = 10,
    benchmark_iters: int = 20,
    use_grad_transform: bool = False,
    extended_printout: bool = True,
    compile_backward: bool = False,
) -> None:
    print(f"Running benchmark {benchmark.name}")
    _print_benchmark_arguments(benchmark)

    devices: list[str] = benchmark.devices
    if len(devices) == 0:
        raise RuntimeError("Found a benchmark with no specified devices")

    callable_construction_time, warmup_stats, benchmark_stats, _ = _run_benchmark(
        benchmark,
        constructor,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        use_grad_transform=use_grad_transform,
        compile_backward=compile_backward,
    )

    _prettyprint_stats(
        benchmark.name,
        callable_construction_time=callable_construction_time,
        warmup_stats=warmup_stats,
        benchmark_stats=benchmark_stats,
        extended_printout=extended_printout,
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

    import os

    os.environ["LOCAL_RANK"] = str(rank)

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
    extended_printout: bool = True,
) -> tuple[int, Sequence[BenchmarkRunStatistics], Sequence[BenchmarkRunStatistics], dict[int, tuple[int, int]]]:
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
        rank_mem_info = {}
        for rank, (
            callable_construction_time,
            warmup_stats,
            benchmark_stats,
            memory_allocated,
            memory_reserved,
        ) in results:
            total_cct += callable_construction_time
            all_warmup_stats.extend(warmup_stats)
            all_benchmark_stats.extend(benchmark_stats)
            rank_mem_info[rank] = (memory_allocated, memory_reserved)

        avg_cct: int = total_cct // world_size

        _prettyprint_stats(
            benchmark_name=f"{benchmark.name}",
            callable_construction_time=avg_cct,
            warmup_stats=all_warmup_stats,
            benchmark_stats=all_benchmark_stats,
            extended_printout=extended_printout,
            rank_mem_info=rank_mem_info,
        )
    finally:
        pool.shutdown()

    return total_cct, all_warmup_stats, all_benchmark_stats, rank_mem_info


#
# Common executors (defined here for convenience)
#


def torch_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return fn


def torch_compile_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.reset()
    return torch.compile(fn)


def thunder_torch_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn, executors=[thunder.pytorch_executor])


def thunder_torch_compile_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn, exec[thunder.pytorch_executor], use_torch_compile=True)


from thunder.executors.apex_entropyex import apex_ex, apex_available

thunder_apex_executor: None | Callable = None
thunder_apex_nvfuser_executor: None | Callable = None
if apex_available():

    def thunder_apex_executor(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return thunder.jit(fn, executors=[apex_ex])

    def thunder_apex_nvfuser_executor(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return thunder.jit(fn, executors=[apex_ex, thunder.nvfuser_executor])


from thunder.executors.cudnnex import cudnn_ex, cudnn_available
from thunder.executors.cudnn_layernormex import cudnn_layernorm_ex

thunder_cudnn_executor: None | Callable = None
thunder_cudnn_nvfuser_executor: None | Callable = None
thunder_cudnn_layer_norm_executor: None | Callable = None
thunder_cudnn_layer_norm_nvfuser_executor: None | Callable = None
if cudnn_available():

    def thunder_cudnn_executor(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return thunder.jit(fn, executors=[cudnn_ex])

    def thunder_cudnn_nvfuser_executor(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return thunder.jit(fn, executors=[cudnn_ex, thunder.nvfuser_executor])

    def thunder_cudnn_layer_norm_executor(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return thunder.jit(fn, executors=[cudnn_layernorm_ex])

    def thunder_cudnn_layer_norm_nvfuser_executor(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return thunder.jit(fn, executors=[cudnn_layernorm_ex, thunder.nvfuser_executor])


from thunder.executors.sdpaex import sdpa_ex


def thunder_sdpa_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn, executors=[sdpa_ex])


from thunder.executors.torch_compile import torch_compile_executor as torch_compile_ex


def thunder_sdpa_torch_compile_nvfuser_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn, executors=[sdpa_ex, torch_compile_ex, thunder.nvfuser_executor])


def default_torch_ddp_executor(_) -> Callable:
    def func(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.nn.parallel.DistributedDataParallel(fn)

    return func


@dataclass(frozen=True)
class get_default_torch_fsdp_executor:
    from torch.distributed.fsdp import ShardingStrategy

    sharding_strategy: ShardingStrategy
    apply_torch_compile: bool
    auto_wrap_policy: Any | None

    def __call__(self, _) -> Callable:
        def func(fn: Callable) -> Callable:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            torch.backends.cuda.matmul.allow_tf32 = True
            if not self.apply_torch_compile:
                return FSDP(fn, sharding_strategy=self.sharding_strategy, auto_wrap_policy=self.auto_wrap_policy)
            else:
                return torch.compile(
                    FSDP(
                        fn,
                        sharding_strategy=self.sharding_strategy,
                        use_orig_params=True,
                        auto_wrap_policy=self.auto_wrap_policy,
                    )
                )

        return func


def default_torch_compile_ddp_executor(_) -> Callable:
    def func(fn: Callable) -> Callable:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch._dynamo.reset()
        return torch.compile(torch.nn.parallel.DistributedDataParallel(fn))

    return func


def default_thunder_torch_executor(fn: Callable) -> Callable:
    from thunder.executors import TORCH

    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn, executors=[TORCH])


def default_thunder_always_trace_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn, cache="always trace")


def default_thunder_dynamic_strides_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn)


thunder_executor = default_thunder_dynamic_strides_executor


@dataclass(frozen=True)
class get_default_thunder_ddp_dynamic_strides_executor:
    bucket_size_in_mb: float = 25

    def __call__(self, _) -> Callable:
        from thunder.distributed import ddp
        from thunder.executors.torch_compile import torch_compile_executor
        from thunder.executors.sdpaex import sdpa_ex

        def func(fn: Callable) -> Callable:
            torch.backends.cuda.matmul.allow_tf32 = True
            return thunder.jit(
                ddp(fn, bucket_size_in_mb=self.bucket_size_in_mb),
                executors=[
                    sdpa_ex,
                    torch_compile_executor,
                    thunder.nvfuser_executor,
                ],
            )

        return func


@dataclass(frozen=True)
class get_default_thunder_fsdp_dynamic_strides_executor:
    from thunder.distributed import FSDPBucketingStrategy
    from thunder.distributed import FSDPType

    bucketing_strategy: FSDPBucketingStrategy
    sharding_strategy: FSDPType

    def __call__(self, _) -> Callable:
        from thunder.distributed import fsdp
        from thunder.executors.torch_compile import torch_compile_executor
        from thunder.executors.sdpaex import sdpa_ex

        def func(fn: Callable) -> Callable:
            torch.backends.cuda.matmul.allow_tf32 = True
            return thunder.jit(
                fsdp(
                    fn,
                    bucketing_strategy=self.bucketing_strategy,
                    sharding_strategy=self.sharding_strategy,
                ),
                executors=[
                    sdpa_ex,
                    torch_compile_executor,
                    thunder.nvfuser_executor,
                ],
            )

        return func


def default_thunder_dynamic_strides_executor_no_grad(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn)


def default_thunder_fixed_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True
    return thunder.jit(fn, cache="same input")


# TODO Add grad support
def default_thunder_triton_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True

    TRITON_AVAILABLE = package_available("triton")
    assert TRITON_AVAILABLE, "Trying to benchmark with the thunder+triton executor, but triton is not available"

    from thunder.executors.triton_crossentropy import register_triton_entropyex

    register_triton_entropyex(add_to_default_executors=False)

    executors_list = ("triton_crossentropy", executors.NVFUSER, executors.TORCH)

    return thunder.jit(fn, executors=executors_list, disable_torch_autograd=True)


# TODO Add grad support
def default_thunder_apex_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True

    APEX_CROSS_ENTROPY_AVAILABLE = package_available("xentropy_cuda")
    assert APEX_CROSS_ENTROPY_AVAILABLE, "Trying to benchmark with the thunder+apex executor, but apex is not available"

    from thunder.executors.apex_entropyex import register_apex_entropyex

    register_apex_entropyex(add_to_default_executors=False)

    executors_list = ("apex_xentropy", executors.NVFUSER, executors.TORCH)
    return thunder.jit(fn, executors=executors_list, disable_torch_autograd=True)


# TODO Add grad support
def default_thunder_cudnn_executor(fn: Callable) -> Callable:
    torch.backends.cuda.matmul.allow_tf32 = True

    assert package_available("cudnn"), "Trying to benchmark with the thunder+cudnn executor, but cudnn is not available"

    from thunder.executors.cudnnex import register_cudnnex

    register_cudnnex(add_to_default_executors=False)

    # executors_list = ("cudnn", executors.NVFUSER, executors.TORCH)
    return thunder.jit(fn, executors=executors, disable_torch_autograd=True)


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
    return thunder.jit(fn, executors=executors_list, use_cudagraphs=True, disable_torch_autograd=True)


#
# Benchmarks
#
# TODO Document a pattern to define benchmarks in another file


def _print_benchmark_arguments(bmark: Benchmark) -> None:
    print(f"{bmark.name} benchmark parameters:")
    for arg in bmark.args:
        print(f"\t{arg.name}={getattr(bmark, arg.name)}")


class BwdModule(torch.nn.Module):
    def __init__(self, postprocess_for_backward: Callable):
        super().__init__()

        self.postprocess_for_backward = postprocess_for_backward

    def forward(self, *args, **kwargs):
        bwd_tensor: torch.Tensor = self.postprocess_for_backward(*args, **kwargs)
        return bwd_tensor


# This class can be chained with another module, using sequential, to produce
#   an output suitable for calling .backward() on, simplifying its integration into other benchmarks
class SumModule(torch.nn.Module):
    def __init__(self, postprocess_for_backward: Callable):
        super().__init__()

        self.postprocess_for_backward = postprocess_for_backward

    def forward(self, *args, **kwargs):
        bwd_tensor: torch.Tensor = self.postprocess_for_backward(*args, **kwargs)
        return bwd_tensor.sum()


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
    "test": dict(
        n_layers=1, n_head=1, n_embd=64, seq_len=2, dropout=0, block_size=6, vocab_size=1024
    ),  # for test purposes
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

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.shape: Sequence[int] = batchdims + (self.config.seq_len, 4 * self.config.n_embd)
        self.device: str = device
        self.dtype: dtypes.dtype = dtype
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(dtype)
        self.requires_grad: bool = requires_grad

        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        return (make_tensor(self.shape, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad),), {}

    def fn(self) -> Callable:
        def foo(a):
            return torch.nn.functional.gelu(a, approximate="tanh").sum()

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
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the model. Default is thunder.bfloat16.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
        BenchmarkArg(
            name="only_return_loss",
            description="Whether the model only returns the loss or not. Default is False.",
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
        dtype: dtypes.dtype = thunder.bfloat16,
        requires_grad: bool = True,
        only_return_loss: bool = False,
    ) -> None:
        super().__init__()

        self.config = _extract_nanogpt_config(config)
        self.batchdims = batchdims
        self.indices_dtype = indices_dtype
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad
        self.only_return_loss: bool = only_return_loss

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

        if not self.only_return_loss:
            return gpt

        # NOTE This module filters NanoGPT's (logits, loss) output to the tensor to call ".backward()" on
        class FilterForBwd(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, tup: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
                logits: torch.Tensor
                loss: torch.Tensor
                logits, loss = tup
                return loss

        ffb = FilterForBwd()
        module: torch.nn.Module = torch.nn.Sequential(gpt, ffb)

        return module

    def postprocess_for_backward(self, output: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor | None:
        if not self.requires_grad:
            return None
        logits, loss = output
        return loss


class EinsumBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="shapes",
            description="A sequence of input tensors' shapes. Default is ((16, 16), (16, 16))",
        ),
        BenchmarkArg(
            name="equation",
            description="A string representing an einsum equation. Default is 'ij,jk->ik'",
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
        return "einsum"

    @classmethod
    @property
    def description(cls) -> str:
        return "Einsum Benchmark"

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        shapes: Sequence[Sequence[int]] = ((16, 16), (16, 16)),
        equation: str = "ij,jk->ik",
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.shapes: Sequence[Sequence[int]] = shapes
        self.equation = equation
        self.device: str = device
        self.dtype: torch.dtype = ltorch.to_torch_dtype(dtype)
        self.requires_grad: bool = requires_grad

        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, low=-1, high=+1, device=self.device, requires_grad=self.requires_grad)

        operands = tuple(make(shape, dtype=self.dtype, requires_grad=self.requires_grad) for shape in self.shapes)
        return (self.equation, operands), {}

    def fn(self) -> Callable:
        def einsum(eq, operands):
            return torch.einsum(eq, *operands)

        return einsum


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

        return (logits.view(-1, logits.size(-1)), targets.view(-1)), {}

    def fn(self) -> Callable:
        def foo(logits, targets):
            return torch.nn.functional.cross_entropy(
                logits,
                targets,
                ignore_index=-1,
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
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
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
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
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
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
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
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
        shape = self.batchdims + (self.config.seq_len, self.config.n_embd)

        return (make(shape),), {}

    def fn(self) -> Callable:
        gpt_mlp = (
            nanogpt_model.MLP(self.config).to(device=self.device, dtype=self.tdtype).requires_grad_(self.requires_grad)
        )
        return gpt_mlp


class LlamaMLPBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The Lit-GPT config to use. Default is 'Llama-2-7b-hf'. See the litgpt_model.py for details.",
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
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.bfloat16.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "litgpt-llamamlp"

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | LitGPTConfig = "Llama-2-7b-hf",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.bfloat16,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = LitGPTConfig.from_name(config) if not isinstance(config, LitGPTConfig) else config
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
        shape = self.batchdims + (self.config.block_size, self.config.n_embd)

        return (make(shape),), {}

    def fn(self) -> Callable:
        module = (
            litgpt_model.LLaMAMLP(self.config)
            .to(device=self.device, dtype=self.tdtype)
            .requires_grad_(self.requires_grad)
        )
        return module


class LitGPTCausalSelfAttentionBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The Lit-GPT config to use. Default is 'Llama-2-7b-hf'. See the litgpt_model.py for details.",
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
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.bfloat16.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "litgpt-csa"

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | LitGPTConfig = "Llama-2-7b-hf",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.bfloat16,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = LitGPTConfig.from_name(config) if not isinstance(config, LitGPTConfig) else config
        self.batchdims = batchdims
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
        x = make(self.batchdims + (self.config.block_size, self.config.n_embd))
        cos = make((self.config.block_size, self.config.rope_n_elem), requires_grad=False)
        sin = make((self.config.block_size, self.config.rope_n_elem), requires_grad=False)
        mask = None
        input_pos = None
        return (x, cos, sin, mask, input_pos), {}

    def fn(self) -> Callable:
        module = (
            litgpt_model.CausalSelfAttention(self.config)
            .to(device=self.device, dtype=self.tdtype)
            .requires_grad_(self.requires_grad)
        )
        return module


class LlamaRMSNormBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="n_embd",
            description="Size of the input for the layer.",
        ),
        BenchmarkArg(
            name="dim",
            description="Dimension over which apply the norm.",
        ),
        BenchmarkArg(
            name="eps",
            description="Epsilon value.",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.bfloat16.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "llama-rmsnorm"

    @classmethod
    @property
    def description(cls) -> str:
        return "Llama's RMS norm operation."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        n_embd: int,
        dim: int = -1,
        eps: float = 1e-5,
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.bfloat16,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.size = n_embd
        self.dim = dim
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
        shape = (self.size,)
        return (make(shape),), {}

    def fn(self) -> Callable:
        module = (
            litgpt_model.RMSNorm(self.size, self.dim, self.eps)
            .to(device=self.device, dtype=self.tdtype)
            .requires_grad_(self.requires_grad)
        )
        return module


class LitGPTBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.block_size,). Default is (16,).",
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
        return "litgpt"

    @classmethod
    @property
    def description(cls) -> str:
        return "LitGPT."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: LitGPTConfig,
        batchdims: Sequence[int] = (8,),
        indices_dtype: dtypes.dtype = thunder.int64,
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = config
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

        self.name = f"litgpt ({self.config.name})"

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, low=0, high=255, device=self.device, dtype=self.indices_tdtype, requires_grad=False)
        shape = self.batchdims + (self.config.block_size,)

        x = make(shape)
        return (x,), {}

    def fn(self) -> Callable:
        gpt = (
            litgpt_model.GPT(self.config)
            .to(device=self.device, dtype=self.model_tdtype)
            .requires_grad_(self.requires_grad)
        )
        return gpt

    def postprocess_for_backward(self, output: torch.Tensor) -> torch.Tensor | None:
        if not self.requires_grad:
            return
        logits = output
        targets = make_tensor_like(logits)  # fake targets
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss


# This block of code is after the "attn" projection in the forward
# method of the CausalSelfAttention class and before the
# "scaled_dot_product_attention" call.
class QKVSplitRope(nn.Module):
    def __init__(self, config, use_apex) -> None:
        self.fused_apply_rotary_pos_emb_cached = None
        if use_apex:
            try:
                from apex.transformer.functional import fused_apply_rotary_pos_emb_cached

                self.fused_apply_rotary_pos_emb_cached = fused_apply_rotary_pos_emb_cached
            except ImportError:
                pass

        super().__init__()
        self.config = config
        self.apply_rope = litgpt_model.apply_rope
        self.use_apex = use_apex

    def forward(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = qkv.shape  # batch size, sequence length

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and self.config.n_query_groups != 1:
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        if self.use_apex:
            # Apex kernel expect q, k to be (T, B, nh, hs)
            # And cos and sin to be (T, 1, 1, rope_n_elem)
            cos = cos.unsqueeze(-2).unsqueeze(-2)
            sin = sin.unsqueeze(-2).unsqueeze(-2)
            q = q.permute(2, 0, 1, 3)
            k = k.permute(2, 0, 1, 3)
            q = self.fused_apply_rotary_pos_emb_cached(q, cos, sin)
            k = self.fused_apply_rotary_pos_emb_cached(k, cos, sin)
            q = q.permute(1, 2, 0, 3)
            k = k.permute(1, 2, 0, 3)
            return q, k, v

        q_roped = self.apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = self.apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
        return q, k, v


class LlamaQKVSplitRopeBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The Lit-GPT config to use. Default is 'Llama-2-7b-hf'. See the litgpt_model.py for details.",
        ),
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.block_size, config.n_embd). Default is (16,).",
        ),
        BenchmarkArg(
            name="device",
            description="The device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.bfloat16.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
        BenchmarkArg(
            name="use_apex",
            description="Whether to use apex's fused_apply_rotary_pos_emb_cached function. Default is False.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "litgpt-csa-qkv-split-rope"

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str | LitGPTConfig = "Llama-2-7b-hf",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.bfloat16,
        requires_grad: bool = True,
        use_apex: bool = False,
    ) -> None:
        super().__init__()

        self.config = LitGPTConfig.from_name(config) if not isinstance(config, LitGPTConfig) else config
        self.batchdims = batchdims
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)
        self.devices: list[str] = [device]
        self.use_apex = use_apex

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
        qkv = make(
            self.batchdims
            + (self.config.block_size, (self.config.n_head + 2 * self.config.n_query_groups) * self.config.head_size)
        )
        cos = make((self.config.block_size, self.config.rope_n_elem), requires_grad=False)
        sin = make((self.config.block_size, self.config.rope_n_elem), requires_grad=False)
        return (qkv, cos, sin), {}

    def fn(self) -> Callable:
        module = (
            QKVSplitRope(self.config, self.use_apex)
            .to(device=self.device, dtype=self.tdtype)
            .requires_grad_(self.requires_grad)
        )
        return module


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
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)
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


class LitGPTSDPABenchmark(NanoGPTSDPABenchmark):
    @classmethod
    @property
    def name(cls) -> str:
        return "llama2-sdpa"

    @classmethod
    @property
    def description(cls) -> str:
        return "Lit-GPT's Scaled Dot Product Attention call."

    def __init__(
        self,
        config: str = "Llama-2-7b-hf",
        batchdims: Sequence[int] = (16,),
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.bfloat16,
        requires_grad: bool = True,
    ) -> None:
        from thunder.tests.litgpt_model import Config

        litgptconfig = Config.from_name(config) if not isinstance(config, Config) else config
        nanogptconfig = NanoGPTConfig(
            n_head=litgptconfig.n_head,
            seq_len=litgptconfig.block_size,
            n_embd=litgptconfig.n_embd,
        )
        super().__init__(nanogptconfig, batchdims, device, dtype, requires_grad)


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


# TODO this benchmark doesn't seem to be called by any target and is almost the same
# as benchmarks/nvfuser_benchmarks.py::GPTBlockBenchmark
class GPTBlockBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="config",
            description="The configuration (str) to use. Default is 'open_llama_7b'.",
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
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and model. Default is thunder.bfloat16.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "gpt-block"

    @classmethod
    @property
    def description(cls) -> str:
        return "Lit-GPT's block module"

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: str = "open_llama_7b",
        sequences: int = 8,
        seq_length: int = 1024,
        device: str = "cuda",
        dtype: thunder.dtypes.dtype | torch.dtype | str = thunder.bfloat16,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = LitGPTConfig.from_name(config)
        self.sequences = sequences
        self.seq_length = seq_length
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

        self.cos, self.sin = litgpt_model.build_rope_cache(
            seq_len=seq_length, n_elem=self.config.rope_n_elem, device=self.device
        )

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, device=self.device, dtype=self.tdtype, requires_grad=self.requires_grad)

        a = make((self.sequences, self.seq_length, self.config.n_embd))

        return (a, self.cos, self.sin), {}

    def fn(self) -> Callable:
        model = (
            litgpt_model.Block(self.config).to(device=self.device, dtype=self.tdtype).requires_grad_(self.requires_grad)
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
