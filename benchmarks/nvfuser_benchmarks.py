import argparse
import dataclasses
import datetime
import functools
from itertools import chain
import math
import operator
import os
import pathlib
import textwrap
import time
from typing import Any, Callable, Dict, List, Tuple
import types

# Note: This needs to run before `import torch`.
_DEFAULT_DEVICE = {"luca": "5", "mike": "2", "taylor": "7"}.get(os.getenv("USER", None), None)
if _DEFAULT_DEVICE is not None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _DEFAULT_DEVICE)

import torch
from torch.testing import make_tensor

import thunder
import thunder.torch as ltorch
from thunder.tests import nanogpt_model
import thunder.core.proxies as proxies
from thunder.cudagraphs import CUDAGraphExecutor
from thunder.tests import nanogpt_model, lit_llama_model, hf_bart_self_attn

from lightning_utilities.core.imports import package_available

TRITON_AVAILABLE = package_available("triton")
if TRITON_AVAILABLE:
    from thunder.executors.triton_crossentropy import register_triton_entropyex

from lightning_utilities.core.imports import package_available

TRITON_AVAILABLE = package_available("triton")
if TRITON_AVAILABLE:
    from thunder.executors.triton_crossentropy import register_triton_entropyex

APEX_CROSS_ENTROPY_AVAILABLE = package_available("xentropy_cuda")
if APEX_CROSS_ENTROPY_AVAILABLE:
    from thunder.executors.apex_entropyex import register_apex_entropyex

# This file contains custom nvFuser-related benchmarks.

# To use this file, try running it with -v, which will list all available benchmarks, and -h, which will describe
#   options for controlling how the benchmarks are run. Once a benchmark, like nanogpt, is selected, information
#   on how to further modify the benchmark can be obtained by running this script with "nanogpt -h", which will
#   describe all possible kwargs that can be specified to control the benchmark. For example, to set the dtype,
#   run "nanogpt --dtype float16".


def get_grad(module_or_function, args):
    if isinstance(module_or_function, torch.nn.Module):
        return tuple(x.grad for x in chain.from_iterable((args, module_or_function.parameters())))
    else:
        return tuple(x.grad for x in args)


def zero_grad(module_or_function, args):
    if isinstance(module_or_function, torch.nn.Module):
        module_or_function.zero_grad(set_to_none=True)
    else:
        for x in args:
            x.grad = None


def make_forward_and_backward_fn(postprocess_for_backward, forward_fn, get_grad=False):
    """Returns a function that runs forward_fn and then backward_fn, and returns the output of forward_fn.

    Args:
        postprocess_for_backward: Function to run on the output of forward_fn before running .backward().
        forward_fn: Function to run forward.
        get_grad: If True, returns the output of forward_fn and the gradients of forward_fn's inputs.
    """

    def fn(*args, **kwargs):
        out = forward_fn(*args, **kwargs)
        out = postprocess_for_backward(out)
        grad_out = torch.ones(out.shape, dtype=out.dtype, device=out.device)
        zero_grad(forward_fn, args)
        out.backward(grad_out)
        # This option has high memory overhead, so it's disabled by default for now
        if get_grad:
            return out, get_grad(forward_fn, args)
        return out

    return fn


@dataclasses.dataclass
class BenchmarkArg:
    """
    Describes a benchmark argument.
    """

    name: str
    description: str
    default: Any


# TODO Restore CUDA graphs
# TODO Restore lightning.compile profile info
# TODO Add lightning.compile with the Torch executor as the principal executor (as lc+torch?)
# TODO Improve error messages when CUDA graphs is requested but an executor that doesn't support CUDA graphs has also been requested
class Benchmark:
    """Encapsulates a benchmark.

    To add a benchmark:
        1) Subclass `Benchmark`
        2) Implement `args` so the CLI knows how to parse arguments.
        3) Implement `__init__`. Any args specified in `cls.args` will be passed as kwargs to the ctor.
        4) Implement `make_batch` so the benchmark runner can invoke `fn`
        5) Add your class to `benchmarks` dict.
    """

    def __init__(self, name: str, fn: Callable, shortname=None):
        self.name = name
        self._fn = fn

        # Used for profile filenames
        self.shortname = shortname if shortname is not None else name

    @functools.cache
    def compiled_fn(self, executor: str) -> tuple[str, Callable]:
        # Prepares and validates the executor base and extensions
        executors = executor.split("+")
        base_executor = executors[0]
        assert base_executor in (
            "thunder",
            "torch",
            "torch.compile",
        ), f"Unexpected executor base {base_executor}, accepted executor bases are thunder, torch and torch.compile"

        use_cudagraphs: bool = False
        executor_extensions = executors[1:]
        for extension in executor_extensions:
            assert extension in (
                "nvfuser",
                "triton",
                "apex",
                "cudagraphs",
            ), f"Unexpected executor extension {extension}, accepted extensions are nvfuser, triton, apex and cudagraphs"

            if extension in ("nvfuser", "triton", "apex"):
                assert (
                    base_executor == "thunder"
                ), f"The executor extension {extension} is only available with the thunder executor base"

            if extension == "triton":
                assert (
                    TRITON_AVAILABLE
                ), "Trying to run a benchmark with a Triton executor extension, but Triton is not available"
                register_triton_entropyex(add_to_default_executors=False)

            if extension == "apex":
                assert (
                    APEX_CROSS_ENTROPY_AVAILABLE
                ), "Trying to run a benchmark with the Apex executor extension, but the xentropy_cuda package is not available"
                register_apex_entropyex()

            if extension == "cudagraphs":
                use_cudagraphs = True

        if base_executor == "torch":
            if use_cudagraphs and not self.backward:
                return executor, CUDAGraphExecutor(self._fn)

            if self.backward:
                compiled_fn = make_forward_and_backward_fn(self.postprocess_for_backward, self._fn)
                compiled_fn = CUDAGraphExecutor(compiled_fn) if use_cudagraphs else compiled_fn
                return (
                    f"fw+bw: {executor}",
                    compiled_fn,
                )
            return executor, self._fn
        elif base_executor == "torch.compile":
            options = {"triton.cudagraphs": True} if use_cudagraphs else None
            forward_fn = torch.compile(self._fn, options=options)
            if self.backward:
                return (
                    f"fw+bw: {executor}",
                    make_forward_and_backward_fn(self.postprocess_for_backward, forward_fn),
                )
            return executor, forward_fn

        assert base_executor == "thunder", f"Unknown executor base {base_executor}"

        # Constructs the executors list
        _executor_extension_map: dict[str, list] = {
            "nvfuser": [thunder.executors.NVFUSER],
            "apex": ["apex_xentropy"],
            "triton": ["triton_crossentropy"],
            "cudagraphs": [],
        }

        def _map_to_executor(ext: str) -> list:
            return _executor_extension_map[ext]

        executors_list = []
        for ext in executor_extensions:
            executors_list.extend(_map_to_executor(ext))
        executors_list.append(thunder.executors.TORCH)

        if not self.backward:
            tom = thunder.compile(
                self._fn,
                use_static_caching=True,
                use_cudagraphs=use_cudagraphs,
                executors_list=executors_list,
            )

            return executor, tom

        # NOTE self.backward is True
        if use_cudagraphs:
            raise NotImplementedError("thunder backward + CUDA graphs is currently disabled")

        # Benchmark forward+backward embedded into PyTorch's Autograd
        forward_fn = thunder.compile(
            self._fn,
            use_static_caching=True,
            use_generated_backward=True,
            use_rematerialization=True,
            executors_list=executors_list,
        )

        return (
            f"fw+bw: {executor}",
            make_forward_and_backward_fn(self.postprocess_for_backward, forward_fn),
        )

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        raise NotImplementedError

    def make_batch(self) -> tuple[list, dict]:  # args, kwargs
        raise NotImplementedError

    @property
    def postprocess_for_backward(self):
        """Function to run on the output of forward_fn before running .backward()."""
        return lambda x: x


def median(l):
    if len(l) == 0:
        raise ValueError("Empty lists have no median value!")

    s = sorted(l)

    if len(s) % 2 == 1:
        return s[len(s) // 2]

    return (s[len(s) // 2] + s[len(s) // 2 - 1]) / 2


# TODO: make order of gen and fn consistent
def time_ns(fn, gen, *, warmup_iters=5, iters=20):
    elapsed = []

    def _helper():
        args, kwargs = gen()
        torch.cuda.synchronize()
        start = time.time_ns()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time_ns()
        return end - start, result

    # First run
    initial_time, initial_result = _helper()
    # TODO: saving the initial_result has high memory cost
    initial_result = None

    for _ in range(warmup_iters):
        _helper()

    for _ in range(iters):
        # Sets result to None to allow the Python GC to collect the previous result
        result = None
        t, result = _helper()
        elapsed.append(t)

    # Computes statistics
    avg = functools.reduce(operator.add, elapsed, 0) / iters
    m = median(elapsed)

    stats = {
        "warmup_iters": warmup_iters,
        "iters": iters,
        "initial": initial_time,
        "initial_result": initial_result,
        "average": avg,
        "median": m,
        "final": elapsed[-1],
        "final_result": result,
    }

    return stats


# TODO: consider refactoring this function with the above so they share more code
# def time_thunder_ns(fn, gen, *, warmup_iters=5, iters=20):
#     fn = thunder_compile(fn,)

#     def _helper(fn_):
#         args, kwargs = gen()
#         torch.cuda.synchronize()
#         start = time.time_ns()
#         result = fn_(*args, **kwargs)
#         torch.cuda.synchronize()
#         end = time.time_ns()
#         return end - start, result

#     # NOTE: Initial run
#     initial_time, initial_result = _helper(fn)
#     fusion = initial_result["fusion"]

#     # NOTE: Warmups (a common benchmarking technique)
#     for _ in range(warmup_iters):
#         _helper(fn)

#     # Acquires lazy Thunder stats
#     elapsed = []
#     for _ in range(iters):
#         t, result = _helper(fn)
#         elapsed.append(t)

#     # Computes lazy statistics
#     avg = functools.reduce(operator.add, elapsed, 0) / iters
#     m = median(elapsed)

#     stats = {
#         "warmup_iters": warmup_iters,
#         "iters": iters,
#         "initial": initial_time,
#         "initial_result": initial_result,
#         "lazy_average": avg,
#         "lazy_median": m,
#         "lazy_final": elapsed[-1],
#         "lazy_final_result": result,
#     }

#     # Computes fusion results
#     elapsed = []
#     for _ in range(iters):
#         t, result = _helper(fusion)
#         elapsed.append(t)

#     # Computes fusion statistics
#     avg = functools.reduce(operator.add, elapsed, 0) / iters
#     m = median(elapsed)

#     stats.update(
#         {
#             "fusion_average": avg,
#             "fusion_median": m,
#             "fusion_final": elapsed[-1],
#             "fusion_final_result": result,
#         }
#     )

#     return stats


def percent(numerator, denominator):
    return f"{round(numerator / denominator * 100, 2)}%"


def ns_to_us(ns):
    return f"{ns / 1000:.2f}"


# TODO Restore this utility as a separate tool
# def prettyprint_program(profile, with_timings=False):
#     print(f"Prettyprinting profile of {len(profile)} fusions")

#     for region in profile:
#         if region.is_supported:
#             print(f"Fused region of {len(region.symbols)} operators")
#         else:
#             print(f"Unfused region of {len(region.symbols)} operators")

#         thunder_fusion = region.fusion

#         # NOTE: this can occur if the region has no outputs
#         if thunder_fusion is None:
#             print("region has no outputs, so was not profiled")
#             continue

#         def gen():
#             return construct_inputs_for_region(region), {}

#         fusion, code = get_torch_code_for_region(region, contiguous=False)

#         print("Torch code for the region:")
#         print(code)

#         if with_timings:
#             pt2 = torch.compile(fusion)

#             thunder_stats = time_ns(thunder_fusion, gen)
#             pt2_stats = time_ns(pt2, gen)

#             print(f"thunder+nvFuser median time: {ns_to_us(thunder_stats['median'])}")
#             print(f"pt2 median time: {ns_to_us(pt2_stats['median'])}")


# TODO Restore these utilities as a separate tool
# def summarize_profile(profile):
#     print(f"Summarizing profile of {len(profile)} fusions")

#     unfused_regions = 0
#     unfused_prims = set()
#     for region in profile:
#         if not region.is_supported:
#             unfused_regions += 1
#             for sym in region.symbols:
#                 unfused_prims.add(sym.name)

#     print(f"{unfused_regions} fusions were executed by PyTorch")
#     print(f"Unfused prims: {unfused_prims}")


# def construct_inputs_for_region(region):
#     inps = []
#     for inp in region.inputs:
#         # Unwraps variables
#         if isinstance(inp, Variable):
#             inp = inp.proxy

#         if isinstance(inp, proxies.TensorProxy):
#             tdtype = ltorch.torch_dtype(inp.dtype)
#             a = make_tensor(inp.shape, device="cuda", dtype=tdtype)
#             inps.append(a)
#         elif isinstance(inp, proxies.NumberProxy):
#             inps.append(inp.value)
#         else:
#             inps.append(inp)

#     return inps


# def get_torch_code_for_region(region, * contiguous):
# fusion, code = fuse_torch_region(
#     region.inputs, region.outputs, region.symbols, _return_code=True, _contiguous=contiguous
# )
# return fusion, code


# def _prettyprint_thunder_nvfuser_profile_info(profile):
#     prettyprint_program(profile)


# def _prettyprint_thunder_nvfuser_stats(stats):
#     us = "\u03BCs"

#     def _print_breakdown(s):
#         ns = None
#         meta = None
#         if s == "initial":
#             ns = stats["initial"]
#             meta = stats["initial_result"]["meta"]
#         elif s == "lazy":
#             ns = stats["lazy_final"]
#             meta = stats["lazy_final_result"]["meta"]

#         iter_desc = None
#         if s == "initial":
#             iter_desc = "first"
#         elif s == "lazy":
#             iter_desc = "final lazy"

#         print(f"The {iter_desc} iteration took {ns_to_us(ns)}{us}, and can be broken down as...")
#         a_time = meta["acquisition_time"]
#         t_time = meta["translation_time"]
#         e_time = meta["invocation_time"]
#         print(f"{ns_to_us(a_time)}{us}, {percent(a_time, ns)} of the time, was spent in program acquisition")
#         print(
#             f"{ns_to_us(t_time)}{us}, {percent(t_time, ns)} of the time, was spent translating the program to a fusion definition"
#         )
#         print(f"{ns_to_us(e_time)}{us}, {percent(e_time, ns)} of the time, was spent invoking nvFuser.execute()")

#         accounted_time = a_time + t_time + e_time
#         unaccounted_time = ns - accounted_time
#         print(
#             f"{ns_to_us(unaccounted_time)}{us}, {percent(unaccounted_time, ns)} of the time, is unaccounted for, but is probably how long the kernels took to execute."
#         )

#     print("Thunder+nvFuser results:")
#     print(f"The median time of {stats['iters']} lazy post-warmup iterations was {ns_to_us(stats['lazy_median'])}{us}")
#     print(
#         f"The median time of {stats['iters']} fused post-warmup iterations was {ns_to_us(stats['fusion_median'])}{us}"
#     )
#     _print_breakdown("initial")
#     _print_breakdown("lazy")


def _prettyprint_stats(name, stats):
    us = "\u03BCs"
    print(
        textwrap.dedent(
            f"""\
        {name} results:
            The static init time is: {ns_to_us(stats.get('static_init', 'N/A'))}{us}
            The median time of {stats['iters']} post-warmup iterations was {ns_to_us(stats['median'])}{us}
            The initial iteration took {ns_to_us(stats['initial'])}{us}
            The final iteration took {ns_to_us(stats['final'])}{us}
    """
        )
    )


# def _compare_stats(thunder_name, thunder_stats, name_b, stats_b):
#     thunder_initial = thunder_stats["initial"]
#     other_initial = stats_b["initial"]

#     print(f"Results of comparing Thunder and {name_b}:")
#     if thunder_initial < other_initial:
#         print(
#             f"{thunder_name} was initially faster than {name_b}, taking only {percent(thunder_initial, other_initial)} of the time"
#         )
#     else:
#         print(
#             f"{name_b} was initially faster than {thunder_name}, taking only {percent(other_initial, thunder_initial)} of the time"
#         )

#     thunder_lazy_median = thunder_stats["lazy_median"]
#     other_median = stats_b["median"]
#     name = f"Lazy {thunder_name}"

#     if thunder_lazy_median < other_median:
#         print(
#             f"{name} was faster post-warmup than {name_b}, taking only {percent(thunder_lazy_median, other_median)} of the time"
#         )
#     else:
#         print(
#             f"{name_b} was faster post-warmup than {name}, taking only {percent(other_median, thunder_lazy_median)} of the time"
#         )

#     thunder_fused_median = thunder_stats["fusion_median"]
#     name = f"Fused {thunder_name}"

#     if thunder_fused_median < other_median:
#         print(
#             f"{name} was faster post-warmup than {name_b}, taking only {percent(thunder_fused_median, other_median)} of the time"
#         )
#     else:
#         print(
#             f"{name_b} was faster post-warmup than {name}, taking only {percent(other_median, thunder_fused_median)} of the time"
#         )


# def _benchmark(name, *, gen, iters, thunder_fn, other_name, other_fn):
#     print(f"Benchmark: {name}")

#     thunder_stats = time_thunder_ns(thunder_fn, gen, iters=iters)
#     profile_info = thunder_stats["lazy_final_result"]["profile_info"]

#     _prettyprint_thunder_nvfuser_profile_info(profile_info)
#     _prettyprint_thunder_nvfuser_stats(thunder_stats)

#     other_stats = time_ns(other_fn, gen, iters=iters)
#     _prettyprint_stats(other_name, other_stats)

#     _compare_stats("Thunder + nvFuser", thunder_stats, other_name, other_stats)


#
# Elementwise binary benchmarks
#


# def _add_nvfuser_vs_pt2_factory(shape, *, iters, make_arg):
#     def gen():
#         a = make_arg(shape)
#         b = make_arg(shape)
#         return (a, b), {}

#     # Constructs pt2 function
#     def _add(a, b):
#         return a + b

#     pt2_fn = torch.compile(_add)

#     shape_str = "x".join(str(l) for l in shape)
#     _benchmark(f"add_{shape_str}", gen=gen, iters=iters, thunder_fn=tlang.add, other_name="pt2", other_fn=pt2_fn)


# def add_64x64(iters, make_arg):
#     _add_nvfuser_vs_pt2_factory((64, 64), iters=iters, make_arg=make_arg)


# def add_kwargs_64x64(iters, make_arg):
#     shape = (64, 64)

#     def gen():
#         a = make_arg(shape)
#         b = make_arg(shape)
#         return (), {"a": a, "b": b}

#     # Constructs pt2 function
#     def _add(a, b):
#         return a + b

#     pt2_fn = torch.compile(_add)

#     shape_str = "x".join(str(l) for l in shape)
#     _benchmark(f"add_kwargs_{shape_str}", gen=gen, iters=iters, thunder_fn=tlang.add, other_name="pt2", other_fn=pt2_fn)


# def add_1024x1024(iters, make_arg):
#     _add_nvfuser_vs_pt2_factory((1024, 1024), iters=iters, make_arg=make_arg)


# def add_4096x4(iters, make_arg):
#     _add_nvfuser_vs_pt2_factory((4096, 4), iters=iters, make_arg=make_arg)


# def add_4x4096(iters, make_arg):
#     _add_nvfuser_vs_pt2_factory((4, 4096), iters=iters, make_arg=make_arg)


# def add_dozen_64x64(iters, make_arg):
#     shape = (64, 64)

#     def gen():
#         args = []
#         for _ in range(12):
#             args.append(make_arg(shape))
#         return tuple(args), {}

#     def _add_dozen(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11):
#         return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11

#     pt2_fn = torch.compile(_add_dozen)

#     shape_str = "x".join(str(l) for l in shape)
#     _benchmark(f"add_dozen_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_dozen, other_name="pt2", other_fn=pt2_fn)


# def add_hundred_64x64(iters, make_arg):
#     shape = (64, 64)

#     def gen():
#         args = []
#         for _ in range(100):
#             args.append(make_arg(shape))
#         return tuple(args), {}

#     # fmt: off
#     def _add_hundred(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99):
#         return a0 + a1 + a2 + a3 + a4 + a5+ a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31 + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39 + a40 + a41 + a42 + a43 + a44 + a45 + a46 + a47 + a48 + a49 + a50 + a51 + a52 + a53 + a54 + a55 + a56 + a57 + a58 + a59 + a60 + a61 + a62 + a63 + a64 + a65 + a66 + a67 + a68 + a69 + a70 + a71 + a72 + a73 + a74 + a75 + a76 + a77 + a78 + a79 + a80 + a81 + a82 + a83 + a84 + a85 + a86 + a87 + a88 + a89 + a90 + a91 + a92 + a93 + a94 + a95 + a96 + a97 + a98 + a99
#     # fmt: on

#     pt2_fn = torch.compile(_add_hundred)

#     shape_str = "x".join(str(l) for l in shape)
#     _benchmark(
#         f"add_hundred_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_hundred, other_name="pt2", other_fn=pt2_fn
#     )


# def add_dozen_64x64(iters, make_arg):
#     shape = (64, 64)

#     def gen():
#         args = []
#         for _ in range(12):
#             args.append(make_arg(shape))
#         return tuple(args), {}

#     def _add_dozen(*args):
#         cur = args[0]
#         for a in args[1:]:
#             cur = cur + a
#         return cur

#     pt2_fn = torch.compile(_add_dozen)

#     shape_str = "x".join(str(l) for l in shape)
#     _benchmark(f"add_dozen_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_dozen, other_name="pt2", other_fn=pt2_fn)


# def _add_contiguous_transposed_nvfuser_vs_pt2_factory(shape, *, iters, make_arg):
#     def gen():
#         a = make_arg(shape)
#         b = make_arg(shape).T
#         return (a, b), {}

#     # Makes PyTorch2 function
#     def _add(a, b):
#         return a + b

#     pt2_fn = torch.compile(_add)

#     shape_str = "x".join(str(l) for l in shape)
#     _benchmark(
#         f"add_{shape_str}_contiguous_transposed",
#         gen=gen,
#         iters=iters,
#         thunder_fn=tlang.add,
#         other_name="pt2",
#         other_fn=pt2_fn,
#     )


# def add_1024x1024_contiguous_transposed(iters, make_arg):
#     _add_contiguous_transposed_nvfuser_vs_pt2_factory((1024, 1024), iters=iters, make_arg=make_arg)


# #
# # Elementwise unary benchmarks
# #


# def _elementwise_unary_nvfuser_vs_pt2_factory(shape, *, thunder_op, torch_op, iters, make_arg):
#     def gen():
#         return (make_arg(shape),), {}

#     # Makes pt2 function
#     def _foo(a):
#         return torch_op(a)

#     pt2_fn = torch.compile(_foo)

#     shape_str = "x".join(str(l) for l in shape)
#     name = f"{torch_op.__name__}{shape_str}"
#     _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_op, other_name="pt2", other_fn=pt2_fn)


# def abs_64x64(iters, make_arg):
#     _elementwise_unary_nvfuser_vs_pt2_factory(
#         (64, 64), thunder_op=tlang.abs, torch_op=torch.abs, iters=iters, make_arg=make_arg
#     )


#
# Reduction benchmarks
#

# FIXME: _preprocess doesn't work with *args, **kwargs
# def var_constructor(shape=(64, 64), device='cuda', dtype=thunder.float32, **kwargs):
#     tdtype = ltorch.torch_dtype(dtype)
#     make = functools.partial(make_tensor, device=device, dtype=tdtype)

#     def gen():
#         return (make(shape),), kwargs

#     # NOTE: this wrapper is necessary because inspect can't acquire the signature of functions like
#     #   torch.var
#     def foo(*args, **kwargs):
#         return torch.var(*args, **kwargs)

#     thunder_fn = thunder_compile(foo)
#     args, kwargs = gen()
#     thunder_info = thunder_fn(*args, **kwargs)

#     return "var", gen, foo, thunder_info['fusion'], thunder_info['profile_info']


# def simple_number_conditional_constructor(shape=(64, 64), device="cuda", dtype=thunder.float32):
#     tdtype = ltorch.to_torch_dtype(dtype)
#     make = functools.partial(make_tensor, device=device, dtype=tdtype)

#     def gen():
#         return (make(shape), make(shape), 2), {}

#     def foo(alpha, beta, n):
#         if n < 0:
#             result = alpha - beta
#         else:
#             result = alpha + beta

#         return alpha, result

#     thunder_fn = thunder_compile(foo)

#     args, kwargs = gen()
#     thunder_info = thunder_fn(*args, **kwargs)

#     return "simple-number-conditional", gen, foo, thunder_info["fusion"], thunder_info["profile_info"]


# def simple_kwarg_conditional_constructor(shape=(64, 64), device="cuda", dtype=thunder.float32):
#     tdtype = ltorch.to_torch_dtype(dtype)
#     make = functools.partial(make_tensor, device=device, dtype=tdtype)

#     def gen():
#         return (make(shape), make(shape)), {"n": 2}

#     def foo(alpha, beta, n):
#         if n < 0:
#             result = alpha - beta
#         else:
#             result = alpha + beta

#         return alpha, result

#     thunder_fn = thunder_compile(foo)

#     args, kwargs = gen()
#     thunder_info = thunder_fn(*args, **kwargs)

#     return "simple-kwarg-conditional", gen, foo, thunder_info["fusion"], thunder_info["profile_info"]


def _stacked_add_benchmark(a, b, *, depth=100):
    cur = a
    for _ in range(depth):
        cur = cur + b

    return cur


# FIXME: currently not working with lightning.compile + nvFuser CUDA graphs
class StackedAddBenchmark(Benchmark):
    _universal_args = (
        BenchmarkArg(
            name="depth",
            description="The number of additions to perform.",
            default=100,
        ),
        BenchmarkArg(
            name="shape",
            description="The shape of the input and intermediates.",
            default=(16, 16),
        ),
        BenchmarkArg(
            name="device",
            description="The device to run on. Default is 'cuda'.",
            default="cuda",
        ),
        BenchmarkArg(
            name="dtype",
            description="The device of the model. Default is 'thunder.float32'.",
            default=thunder.float32,
        ),
    )

    @classmethod
    @property
    def args(cls):
        return cls._universal_args

    def __init__(self, depth, shape, device, dtype, **kwargs) -> None:
        self.depth = depth
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.tdtype = ltorch.to_torch_dtype(dtype)

        super().__init__(
            name=f"StackedAddBenchmark",
            # NOTE: partial here won't work with torch.compile, which is why depth is passed as a kwarg
            #   (See make_batch)
            # fn=partial(_stacked_add_benchmark, depth=self.depth)
            fn=_stacked_add_benchmark,
        )

    def make_batch(self) -> tuple[list, dict]:
        a = make_tensor(self.shape, device=self.device, dtype=self.tdtype)
        b = make_tensor(self.shape, device=self.device, dtype=self.tdtype)
        return (a, b), {"depth": self.depth}


#
# nanoGPT benchmarks
#
# TODO: maybe put these in their own file?


# Taken from nanogpt_model.py
@dataclasses.dataclass
class GPTConfig:
    name: str = ""
    block_size: int = 1024
    seq_len: int = 128
    # GPT-2 vocab_size is 50257 but recent nanoGPT uses 50304 for GPU performance
    # https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L111
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

    def update(self, **kwargs) -> None:
        for field in dataclasses.fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])


class GPTBenchMarkBase(Benchmark):
    _configs = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }

    # Callable or class which will produce the function to be benchmarked.
    benchmark_factory = None

    # String name for the benchmark. Can include variables from `kwargs`. (e.g. "foo_{dtype}")
    name_template = None

    # Any benchmark specific arguments. They will be added to `_universal_args`.
    extra_args = None

    _universal_args = (
        BenchmarkArg(
            name="config",
            description="The nanogpt configuration to use. Options are gpt2, gpt2-medium, gpt2-large, and gpt2-xl. Default is gpt2-medium. See the nanogpt model for details.",
            default="gpt2-medium",
        ),
        BenchmarkArg(
            name="device",
            description="The device to run on. Default is 'cuda'.",
            default="cuda",
        ),
        BenchmarkArg(
            name="dtype",
            description="The device of the model. Default is 'thunder.float32'.",
            default=thunder.float32,
        ),
    )

    def __init__(self, config, device, dtype, backward=False, **kwargs) -> None:
        cls = self.__class__
        gpt_config = GPTConfig()
        gpt_config.update(**cls._configs[config])
        gpt_config.update(**kwargs)
        self.backward = backward

        assert cls.benchmark_factory is not None
        tdtype = ltorch.to_torch_dtype(dtype)
        model = cls.benchmark_factory(gpt_config)
        if not isinstance(cls.benchmark_factory, types.FunctionType):
            model = model.to(device=device, dtype=tdtype)

            # Ensures no parameters require grad (suitable for forward testing)
            if not backward:
                for p in model.parameters():
                    p.requires_grad = False

        kwargs.update({"config": config, "gpt_config": gpt_config, "device": device, "dtype": dtype, "tdtype": tdtype})
        self.ctor_kwargs = kwargs
        super().__init__(
            name=cls.name_template.format(**kwargs),
            shortname=cls.name_template.replace(r"{gpt_config}", r"{config}").format(**kwargs),
            fn=model,
        )

    @classmethod
    @property
    def args(cls):
        return cls._universal_args + cls.extra_args

    def make_batch(self) -> tuple[list, dict]:
        return self._make_batch(**self.ctor_kwargs)

    def _make_batch(self) -> tuple[list, dict]:
        raise NotImplementedError


class NanoGPTBenchmark(GPTBenchMarkBase):
    benchmark_factory = nanogpt_model.GPT
    name_template = "nanogpt-{gpt_config}"
    extra_args = (
        BenchmarkArg(
            name="dropout",
            description="The dropout probability to use. Default is .1.",
            default=0.1,
        ),
        BenchmarkArg(
            name="batch_dims",
            description="The batch dimensions to use for the input. (The input will have innermost dimensions of config.block_size. Default is (16,).",
            default=(16,),
        ),
        BenchmarkArg(
            name="indices_dtype",
            description="The dtype of the input. Default is int64.",
            default=thunder.int64,
        ),
    )

    def _make_batch(self, batch_dims, gpt_config, device, indices_dtype, **_) -> tuple[list, dict]:
        shape = batch_dims + (gpt_config.seq_len,)
        x = make_tensor(shape, low=0, high=255, device=device, dtype=ltorch.to_torch_dtype(indices_dtype))
        targets = make_tensor(shape, low=0, high=255, device=device, dtype=ltorch.to_torch_dtype(indices_dtype))
        return (x, targets), {}

    @property
    def postprocess_for_backward(self):
        # nanogpt model returns a tuple of (logits, loss) for the forward pass
        # and we only want to return the loss for the backward pass
        return lambda nanogpt_model_output: nanogpt_model_output[1]


class NanoGPTBlockBenchmark(GPTBenchMarkBase):
    benchmark_factory = nanogpt_model.Block
    name_template = "nanogpt-{gpt_config}-block"
    extra_args = (
        BenchmarkArg(
            name="dropout",
            description="The dropout probability to use. Default is .1.",
            default=0.1,
        ),
        BenchmarkArg(
            name="batch_dims",
            description="The batch dimensions to use for the input. (The input will have innermost dimensions of (config.seq_len, config.n_embd). Default is (16,).",
            default=(16,),
        ),
    )

    def _make_batch(self, batch_dims, gpt_config, device, tdtype, **_) -> tuple[list, dict]:
        return (
            make_tensor(
                batch_dims
                + (
                    gpt_config.seq_len,
                    gpt_config.n_embd,
                ),
                device=device,
                dtype=tdtype,
                requires_grad=self.backward,
            ),
        ), {}


class NanoGPTCSABenchmark(GPTBenchMarkBase):
    benchmark_factory = nanogpt_model.CausalSelfAttention
    name_template = "nanogpt-{gpt_config}-csa"
    extra_args = (
        BenchmarkArg(
            name="dropout",
            description="The dropout probability to use. Default is .1.",
            default=0.1,
        ),
        BenchmarkArg(
            name="batch_dims",
            description="The batch dimensions to use for the input. (The input will have innermost dimensions of (config.seq_len, config.n_embd). Default is (8,).",
            default=(8,),
        ),
    )

    def _make_batch(self, batch_dims, gpt_config, device, tdtype, **_) -> tuple[list, dict]:
        return (
            make_tensor(
                batch_dims
                + (
                    gpt_config.seq_len,
                    gpt_config.n_embd,
                ),
                device=device,
                dtype=tdtype,
                requires_grad=self.backward,
            ),
        ), {}


class NanoGPTMLPBenchmark(GPTBenchMarkBase):
    benchmark_factory = nanogpt_model.MLP
    name_template = "nanogpt-{gpt_config}-mlp"
    extra_args = (
        BenchmarkArg(
            name="dropout",
            description="The dropout probability to use. Default is .1.",
            default=0.1,
        ),
        BenchmarkArg(
            name="batch_dims",
            description="The batch dimensions to use for the input. (The input will have an innermost dimension of length (config.seq_len, config.n_embd). Default is (16,).",
            default=(16,),
        ),
    )

    def _make_batch(self, batch_dims, gpt_config, device, tdtype, **_) -> tuple[list, dict]:
        x = make_tensor(
            batch_dims + (gpt_config.seq_len, gpt_config.n_embd),
            device=device,
            dtype=tdtype,
            requires_grad=self.backward,
        )
        return (x,), {}


# Current version of benchmarking doesn't correctly support benchmark_factory to
# be a callable that returns a nn.Module
class nanoGPTLayerNorm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = torch.nn.LayerNorm(config.n_embd)

    def forward(self, x):
        return self.ln(x)


class NanoGPTLayerNormBenchmark(GPTBenchMarkBase):
    benchmark_factory = nanoGPTLayerNorm
    name_template = "nanogpt-{gpt_config}-layer-norm"
    extra_args = (
        BenchmarkArg(
            name="batch_dims",
            description="The batch dimensions to use for the input. (The input will have an innermost dimension of length (config.seq_len, config.n_embd). Default is (16,).",
            default=(16,),
        ),
    )

    def _make_batch(self, batch_dims, gpt_config, device, tdtype, **_) -> tuple[list, dict]:
        x = make_tensor(
            batch_dims + (gpt_config.seq_len, gpt_config.n_embd),
            device=device,
            dtype=tdtype,
            requires_grad=self.backward,
        )
        return (x,), {}


class nanoGPTCrossEntropy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, logits, targets):
        return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


class NanoGPTCrossEntropyBenchmark(GPTBenchMarkBase):
    benchmark_factory = nanoGPTCrossEntropy
    name_template = "nanogpt-{gpt_config}-cross-entropy"
    extra_args = (
        BenchmarkArg(
            name="batch_dims",
            description="The batch dimensions to use for the input. Default is (16,).",
            default=(16,),
        ),
        BenchmarkArg(
            name="indices_dtype",
            description="The dtype of the input. Default is int64.",
            default=thunder.int64,
        ),
    )

    def _make_batch(self, batch_dims, gpt_config, device, indices_dtype, tdtype, **_) -> tuple[list, dict]:
        logits = make_tensor(
            batch_dims + (gpt_config.seq_len, gpt_config.vocab_size),
            low=0,
            high=255,
            device=device,
            dtype=tdtype,
            requires_grad=self.backward,
        )
        targets = make_tensor(
            batch_dims + (gpt_config.seq_len,),
            low=0,
            high=255,
            device=device,
            dtype=ltorch.to_torch_dtype(indices_dtype),
        )
        return (logits, targets), {}


class nanoGPTEmbedding(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = torch.nn.Embedding(config.block_size, config.n_embd)
        self.drop = torch.nn.Dropout(config.dropout)
        self.ln = torch.nn.LayerNorm(config.n_embd)

    def forward(self, idx):
        # This is a part of the GPT's forward pass before the transformer block
        device = idx.device
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        # LayerNorm is the first operation in nanoGPT's Block before the attention
        return self.ln(x)


class NanoGPTEmbeddingBenchmark(GPTBenchMarkBase):
    benchmark_factory = nanoGPTEmbedding
    name_template = "nanogpt-{gpt_config}-embedding"
    extra_args = (
        BenchmarkArg(
            name="batch_dims",
            description="The batch dimensions to use for the input. Default is (16,).",
            default=(16,),
        ),
        BenchmarkArg(
            name="indices_dtype",
            description="The dtype of the input. Default is int64.",
            default=thunder.int64,
        ),
    )

    def _make_batch(self, batch_dims, gpt_config, device, indices_dtype, tdtype, **_) -> tuple[list, dict]:
        idx = make_tensor(
            batch_dims + (gpt_config.seq_len,),
            low=0,
            high=255,
            device=device,
            dtype=ltorch.to_torch_dtype(indices_dtype),
        )
        return (idx,), {}


class nanoGPTBlockLoop(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = torch.nn.ModuleDict(
            dict(
                drop=torch.nn.Dropout(config.dropout),
                h=torch.nn.ModuleList([nanogpt_model.Block(config) for _ in range(config.n_layer)]),
                ln_f=torch.nn.LayerNorm(config.n_embd),
            )
        )

    def forward(self, x):
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x


class NanoGPTBlockLoopBenchmark(GPTBenchMarkBase):
    benchmark_factory = nanoGPTBlockLoop
    name_template = "nanogpt-{gpt_config}-block-loop"
    extra_args = (
        BenchmarkArg(
            name="batch_dims",
            description="The batch dimensions to use for the input. Default is (16,).",
            default=(16,),
        ),
    )

    def _make_batch(self, batch_dims, gpt_config, device, tdtype, **_) -> tuple[list, dict]:
        x = make_tensor(
            batch_dims + (gpt_config.seq_len, gpt_config.n_embd),
            low=0,
            high=255,
            device=device,
            dtype=tdtype,
        )
        return (x,), {}


c = math.sqrt(2.0 / math.pi)


def new_gelu(a):
    return 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * torch.pow(a, 3.0))))


class NanoGPTGeLUBenchmark(GPTBenchMarkBase):
    benchmark_factory = lambda *_, **__: new_gelu
    name_template = "nanogpt-{gpt_config}-gelu"
    extra_args = (
        BenchmarkArg(
            name="shape",
            description="The shape of the input. Default is (1024, 1024).",
            default=(1024, 1024),
        ),
    )

    def _make_batch(self, shape, device, tdtype, **_) -> tuple[list, dict]:
        return (make_tensor(shape, device=device, dtype=tdtype, requires_grad=self.backward),), {}


class HuggingFaceSelfAttnBenchmark(Benchmark):
    # Taken from HuggingFace Bart-Large model config:
    # https://huggingface.co/facebook/bart-large/blob/main/config.json
    _configs = {
        "bart-large": dict(embed_dim=1024, num_heads=16),
    }

    def __init__(self, *, config, device, dtype, dropout, sequences, seq_length, **_) -> None:
        cls = self.__class__
        self.device = device
        self.dtype = dtype
        self.sequences = sequences
        self.seq_length = seq_length
        self.hf_config = cls._configs[config]

        self.tdtype = ltorch.to_torch_dtype(dtype)
        bart_model = hf_bart_self_attn.BartAttention(
            self.hf_config["embed_dim"],
            self.hf_config["num_heads"],
            dropout=dropout,
        ).to(device=self.device, dtype=self.tdtype)

        super().__init__(
            name=f"hf-{self.hf_config}-self-attn",
            shortname=f"hf-{config}-self-attn",
            fn=bart_model,
        )

    @classmethod
    @property
    def args(cls):
        return (
            BenchmarkArg(
                name="config",
                description="The hf-Bart configuration to use. Options are bart-large. Default is bart-larg.",
                default="bart-large",
            ),
            BenchmarkArg(
                name="dropout",
                description="The dropout probability to use. Default is .1.",
                default=0.1,
            ),
            BenchmarkArg(
                name="sequences",
                description="The number of sequences executed by the model.",
                default=8,
            ),
            BenchmarkArg(
                name="seq_length",
                description="The sequence length of each sequence.",
                default=1024,
            ),
            BenchmarkArg(
                name="device",
                description="The device of the model. Default is 'cuda'.",
                default="cuda",
            ),
            BenchmarkArg(
                name="dtype",
                description="The dtype of the model. Default is float32.",
                default=thunder.float32,
            ),
        )

    def make_batch(self) -> tuple[list, dict]:
        return (
            make_tensor(
                (
                    self.sequences,
                    self.seq_length,
                    self.hf_config["embed_dim"],
                ),
                device=self.device,
                dtype=self.tdtype,
            ),
            make_tensor(
                (
                    self.sequences,
                    1,
                    1,
                    self.seq_length,
                ),
                device=self.device,
                dtype=self.tdtype,
            ),
        ), {}


class LLaMABlockBenchmark(Benchmark):
    _configs = {"7B": "7B"}

    def __init__(self, *, config, device, dtype, sequences, seq_length, **_) -> None:
        cls = self.__class__
        self.device = device
        self.dtype = dtype
        self.sequences = sequences
        self.seq_length = seq_length
        self.config = lit_llama_model.LLaMAConfig.from_name(config)

        self.tdtype = ltorch.to_torch_dtype(dtype)
        model = lit_llama_model.Block(self.config).to(device=self.device, dtype=self.tdtype)

        super().__init__(
            name=f"llama-{config}-block",
            shortname=f"llama-{config}-block",
            fn=model,
        )

    @classmethod
    @property
    def args(cls):
        return (
            BenchmarkArg(
                name="config",
                description="The llama configuration to use. Default is 7B.",
                default="7B",
            ),
            BenchmarkArg(
                name="sequences",
                description="The number of sequences executed by the model.",
                default=8,
            ),
            BenchmarkArg(
                name="seq_length",
                description="The sequence length of each sequence.",
                default=1024,
            ),
            BenchmarkArg(
                name="device",
                description="The device of the model. Default is 'cuda'.",
                default="cuda",
            ),
            BenchmarkArg(
                name="dtype",
                description="The dtype of the model. Default is float32.",
                default=thunder.float32,
            ),
        )

    def make_batch(self) -> tuple[list, dict]:
        return (
            make_tensor(
                (
                    self.sequences,
                    self.seq_length,
                    self.config.n_embd,
                ),
                device=self.device,
                dtype=self.tdtype,
            ),
        ), {}


# NOTE: new benchmark style, in development
def benchmark(
    benchmark: Benchmark,
    *,
    # See `== Universal Arguments ==` section for descriptions.
    executors,
    use_tf32,
    warmup_iters=5,
    iters=20,
    print_program=False,
    profile=False,
    nsight=False,
):
    # TODO Restore option to print the program
    # if print_program:
    #     _, _, thunder_profile = benchmark.compiled_fn("thunder+nvfuser")
    #     prettyprint_program(thunder_profile)

    # Workaround an initialization issue with Kineto and CUDA graphs
    #   https://github.com/pytorch/pytorch/issues/75504#issuecomment-1467065935
    if profile:
        with torch.profiler.profile():
            pass

    torch.backends.cuda.matmul.allow_tf32 = use_tf32

    print(f"Benchmarking {benchmark.name}")
    for executor in executors:
        # With some versions of PyTorch, this will complain with
        #   "Must call `torch._dynamo.reset()` before changing backends."
        # Other versions of PyTorch will complain that torch has no attribute _dynamo
        try:
            torch._dynamo.reset()
        except:
            pass

        t0 = time.time()
        x = torch.mm(x := torch.ones((1024, 1024), device="cuda"), x)
        del x
        cuda_init_time = time.time() - t0

        t0 = time.time()
        name, fn = benchmark.compiled_fn(executor)
        static_init_time = time.time() - t0

        stats = time_ns(fn, benchmark.make_batch, warmup_iters=warmup_iters, iters=iters)
        stats["static_init"] = static_init_time * 1e9
        _prettyprint_stats(name, stats)

        if profile:
            assert name is not None
            inputs = [benchmark.make_batch() for _ in range(5)]
            with torch.profiler.profile(with_stack=True) as p:
                for args, kwargs in inputs:
                    _ = fn(*args, **kwargs)
                    torch.cuda.synchronize()
                    p.step()

            trace_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "traces"))
            pathlib.Path(trace_dir).mkdir(exist_ok=True)

            # E.g. benchmarks/traces/nanogpt-gpt2-medium_torch.compile_cuda_graphs_2023_03_13:13.42.35.pt.trace.json.gz
            now = datetime.datetime.now().strftime("%Y_%m_%d:%H.%M.%S.pt.trace.json.gz")
            p.export_chrome_trace(os.path.join(trace_dir, f"{benchmark.shortname}_{name}_{now}"))

        if nsight:
            # Based on https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223
            torch.cuda.profiler.start()
            for _ in range(3):
                args, kwargs = benchmark.make_batch()
                _ = fn(*args, **kwargs)
                torch.cuda.synchronize()
            torch.cuda.profiler.stop()

    if print_program:
        _, _, thunder_profile = benchmark.compiled_fn("thunder+nvfuser", with_profile_info=True)
        prettyprint_program(thunder_profile)


# TODO: port to benchmarks dict below
# benchmarks = {
# # Elementwise Binary benchmarks
# "add_64x64": add_64x64,
# "add_kwargs_64x64": add_kwargs_64x64,
# "add_dozen_64x64": add_dozen_64x64,
# "add_hundred_64x64": add_hundred_64x64,
# "add_stack100_64x64": add_stack100_64x64,
# "add_stack1000_64x64": add_stack1000_64x64,
# "add_dozen_64x64": add_dozen_64x64,  # Requires supporting *args
# "add_1024x1024": add_1024x1024,
# "add_4096x4": add_4096x4,
# "add_4x4096": add_4x4096,
# "add_1024x1024_contiguous_transposed": add_1024x1024_contiguous_transposed,
# # Elementwise Unary benchmarks
# "abs_64x64": abs_64x64,
# }

# TODO: replace old benchmarks with this style
# value format: Benchmark, string description
benchmarks = {
    # Reduction benchmarks
    # FIXME: _preprocess needs to support *args, **kwargs
    # "var": var_constructor,
    # Control flow benchmarks
    # "simple-number-conditional": simple_number_conditional_constructor,
    # "simple-kwarg-conditional": simple_kwarg_conditional_constructor,
    # Generic benchmarks
    "stacked-additions": (StackedAddBenchmark, "adding in a loop"),
    # nanogpt benchmarks
    "nanogpt": (NanoGPTBenchmark, "nanogpt forward"),
    "nanogpt-block": (NanoGPTBlockBenchmark, "nanogpt block module forward"),
    "nanogpt-block-loop": (NanoGPTBlockLoopBenchmark, "nanogpt block module in a loop"),
    "nanogpt-csa": (NanoGPTCSABenchmark, "nanogpt CausalSelfAttention (CSA) module forward"),
    "nanogpt-mlp": (NanoGPTMLPBenchmark, "nanogpt MLP module forward"),
    "nanogpt-layer-norm": (NanoGPTLayerNormBenchmark, "nanogpt LayerNorm module"),
    "nanogpt-cross-entropy": (NanoGPTCrossEntropyBenchmark, "nanogpt cross entropy loss"),
    "nanogpt-embedding": (NanoGPTEmbeddingBenchmark, "nanogpt embedding+dropout+layer norm"),
    "nanogpt-gelu": (NanoGPTGeLUBenchmark, "nanogpt gelu function forward"),
    "hf-bart-self-attn": (HuggingFaceSelfAttnBenchmark, "hf bart self-attn module forward"),
    "llama-block": (LLaMABlockBenchmark, "lit llama block forward"),
}


# TODO Provide a programmatic way to run these benchmarks and acquire the results
# TODO Allow a file to specify which benchmarks to run and how
if __name__ == "__main__":
    executors = (
        "thunder",
        "torch",
        "torch.compile",
    )

    # Argparse doesn't properly share parents, so subparsers have to make a parent without defaults.
    # Otherwise the subparser will override already parsed values with the defaults.
    def make_shared_parser(argument_default=None):
        parser = argparse.ArgumentParser(add_help=False, argument_default=argument_default)
        listed_executors = textwrap.indent("\n".join(executors), " " * 4)

        def add_argument(*args, **kwargs):
            if argument_default is argparse.SUPPRESS:
                kwargs.pop("default", None)
            return parser.add_argument(*args, **kwargs)

        # == Universal Arguments ==
        add_argument(
            "--executors",
            "-x",
            default=",".join(executors),
            type=str,
            help=f"Specifies the executors to collect statistics for. (Default: all)\n{listed_executors}",
        )
        add_argument(
            "--no-tf-32",
            "--no-tf32",
            dest="disable_tf32",
            action="store_true",
            help="Disable the use of TensorFloat-32 for single precision matrix multiplications. (Default: on)",
        )
        add_argument(
            "--iters",
            default=20,
            type=int,
            help="Specifies the number of post-warmup iterations to use to collect statistics for each executor. (Default: 20)",
        )
        add_argument(
            "--warmup_iters",
            default=5,
            type=int,
            help="The number of warmup iterations to run for each executor before collecting statistics. (Default: 5)",
        )
        add_argument(
            "--print_program",
            "-pp",
            action="store_true",
            help="Displays information about lightning+nvFuser fusions",
        )
        add_argument(
            "--profile",
            action="store_true",
            help="Collect a trace using torch.profiler.profile()",
        )
        add_argument(
            "--nsight",
            action="store_true",
            help="Run a profiled region for nsight systems. (Requires `nsys python nvfuser_benchmarks.py ...`)",
        )
        add_argument(
            "--backward",
            action="store_true",
            help="Run forward+backward pass for benchmarks that support it.",
        )
        return parser

    parser = argparse.ArgumentParser(
        add_help=True,
        parents=[make_shared_parser()],
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Compare backends.
              $ python nvfuser_benchmarks.py -x torch-eager,torch.compile,thunder+nvfuser -pp nanogpt --iters 50

              # Detailed profiling of a region.
              $ python nvfuser_benchmarks.py -x thunder+nvfuser nanogpt-mlp --warmup_iters 500 --iters 1000 --profile
        """
        ),
    )
    subparsers = parser.add_subparsers(title="benchmarks", dest="benchmark", required=True)
    for benchmark_name, (benchmark_constructor, _) in benchmarks.items():
        subparser = subparsers.add_parser(benchmark_name, parents=[make_shared_parser(argparse.SUPPRESS)])
        for benchmark_arg in benchmark_constructor.args:
            subparser.add_argument(
                f"--{benchmark_arg.name}",
                default=benchmark_arg.default,
                help=benchmark_arg.description,
                # TODO Make this more robust
                type=(
                    type(benchmark_arg.default)
                    if isinstance(benchmark_arg.default, (str, int, float))
                    else lambda x: getattr(thunder, x)
                ),
            )

    parsed = parser.parse_args()
    assert parsed.benchmark in benchmarks  # Argparse should guard against this.

    benchmark_constructor, benchmark_desc = benchmarks[parsed.benchmark]
    kwargs = {arg.name: getattr(parsed, arg.name) for arg in benchmark_constructor.args}

    # NOTE: in the future, if running multiple benchmarks from one process, the benchmarks
    #   should probably be run in a subprocess to minimize executor caching
    benchmark(
        benchmark_constructor(backward=parsed.backward, **kwargs),
        executors=parsed.executors.split(","),
        use_tf32=not parsed.disable_tf32,
        iters=parsed.iters,
        warmup_iters=parsed.warmup_iters,
        print_program=parsed.print_program,
        profile=parsed.profile,
    )
