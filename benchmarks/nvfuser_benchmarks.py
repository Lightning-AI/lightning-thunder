import argparse
import math
import multiprocessing
import operator
import os
import time
import warnings
from functools import partial, reduce
from multiprocessing import Process
from dataclasses import dataclass
from enum import auto, Enum
import sys
from typing import Any

import torch
from torch.testing import make_tensor, assert_close
import torch.nn as nn
import torch.nn.functional as F

import thunder
import thunder.core.lang as tlang
import thunder.langs.torch as ttorch
import thunder.core.dtypes as dtypes
import thunder.core.proxies as proxies
from thunder.executors.torch import _fuse_region as fuse_torch_region
from thunder.tests import nanogpt_model
from thunder.tests import hf_bart_self_attn
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten

# This file contains custom nvFuser-related benchmarks.

# To use this file, try running it with -v, which will list all available benchmarks, and -h, which will describe
#   options for controlling how the benchmarks are run. Once a benchmark, like nanogpt, is selected, information
#   on how to further modify the benchmark can be obtained by running this script with "nanogpt -i", which will
#   describe all possible kwargs that can be specified to control the benchmark. For example, to set the dtype,
#   run "nanogpt dtype=float16".


@dataclass
class BenchmarkArg:
    """
    Describes a benchmark argument.
    """

    name: str
    description: str
    default: Any


# Helper to easily modify thunder construction
def thunder_compile(fn):
    return thunder.make_traced(
        fn, executor="nvfuser", _preprocess=True, _info=True, _return_fusion=True, _profile_info=True
    )


# TODO: throw error on unrecongized executor
def get_executors(torch_fn, thunder_fn, args):
    def _helper(x):
        if x == "torch-eager":
            return (x, torch_fn)
        if x == "torch.compile":
            return (x, torch.compile(torch_fn))
        if x == "torch.compile_nvfuser_prims":
            return (x, torch.compile(torch_fn, backend="nvprims_nvfuser"))
        if x == "thunder+nvfuser" or x == "thunder" or x == "nvfuser":
            return ("thunder+nvfuser", thunder_fn)

        raise ValueError(f"Unknown executor {x} requested")

    executors = tuple(_helper(x) for x in args)
    return executors


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

    for _ in range(warmup_iters):
        _helper()

    for _ in range(iters):
        t, result = _helper()
        elapsed.append(t)

    # Computes statistics
    avg = reduce(operator.add, elapsed, 0) / iters
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
#     avg = reduce(operator.add, elapsed, 0) / iters
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
#     avg = reduce(operator.add, elapsed, 0) / iters
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
    return round(ns / 1000, 2)


def prettyprint_profile(profile):
    print(f"Prettyprinting profile of {len(profile)} fusions")

    for region in profile:
        if region.is_supported:
            print(f"Fused region of {len(region.symbols)} operators")
        else:
            print(f"Unfused region of {len(region.symbols)} operators")

        thunder_fusion = region.fusion

        # NOTE: this can occur if the region has no outputs
        if thunder_fusion is None:
            print("region has no outputs, so was not profiled")
            continue

        def gen():
            return construct_inputs_for_region(region), {}

        fusion, code = get_torch_code_for_region(region, contiguous=False)
        pt2 = torch.compile(fusion)

        print("Torch code for the region:")
        print(code)

        thunder_stats = time_ns(thunder_fusion, gen)
        pt2_stats = time_ns(pt2, gen)

        print(f"thunder+nvFuser median time: {ns_to_us(thunder_stats['median'])}")
        print(f"pt2 median time: {ns_to_us(pt2_stats['median'])}")


def summarize_profile(profile):
    print(f"Summarizing profile of {len(profile)} fusions")

    unfused_regions = 0
    unfused_prims = set()
    for region in profile:
        if not region.is_supported:
            unfused_regions += 1
            for sym in region.symbols:
                unfused_prims.add(sym.name)

    print(f"{unfused_regions} fusions were executed by PyTorch")
    print(f"Unfused prims: {unfused_prims}")


def construct_inputs_for_region(region):
    inps = []
    for inp in region.inputs:
        if isinstance(inp, proxies.TensorProxy):
            tdtype = ttorch.torch_dtype(inp.dtype)
            a = make_tensor(inp.shape, device="cuda", dtype=tdtype)
            inps.append(a)
        elif isinstance(inp, proxies.NumberProxy):
            inps.append(inp.value)
        else:
            inps.append(inp)

    return inps


def get_torch_code_for_region(region, *, contiguous):
    fusion, code = fuse_torch_region(
        region.inputs, region.outputs, region.symbols, _return_code=True, _contiguous=contiguous
    )
    return fusion, code


# def _prettyprint_thunder_nvfuser_profile_info(profile):
#     prettyprint_profile(profile)


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

    print(f"{name} results:")
    print(f"The median time of {stats['iters']} post-warmup iterations was {ns_to_us(stats['median'])}{us}")
    print(f"The initial iteration took {ns_to_us(stats['initial'])}{us}")
    print(f"The final iteration took {ns_to_us(stats['final'])}{us}")


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


# def add_stack100_64x64(iters, make_arg):
#     shape = (64, 64)

#     def gen():
#         a = make_arg(shape)
#         b = make_arg(shape)
#         return (a, b), {}

#     def _add_stack100(a, b):
#         cur = a
#         for _ in range(100):
#             cur = cur + b

#         return cur

#     pt2_fn = torch.compile(_add_stack100)

#     shape_str = "x".join(str(l) for l in shape)
#     _benchmark(
#         f"add_stack100_{shape_str}",
#         gen=gen,
#         iters=iters,
#         thunder_fn=_add_stack100,
#         other_name="pt2",
#         other_fn=pt2_fn,
#     )


# def add_stack1000_64x64(iters, make_arg):
#     shape = (64, 64)

#     def gen():
#         a = make_arg(shape)
#         b = make_arg(shape)
#         return (a, b), {}

#     def _add_stack1000(a, b):
#         cur = a
#         for _ in range(1000):
#             cur = cur + b

#         return cur

#     pt2_fn = torch.compile(_add_stack1000)

#     shape_str = "x".join(str(l) for l in shape)
#     _benchmark(
#         f"add_stack1000_{shape_str}",
#         gen=gen,
#         iters=iters,
#         thunder_fn=_add_stack1000,
#         other_name="pt2",
#         other_fn=pt2_fn,
#     )


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
#     tdtype = ttorch.torch_dtype(dtype)
#     make = partial(make_tensor, device=device, dtype=tdtype)

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


def simple_number_conditional_constructor(shape=(64, 64), device="cuda", dtype=thunder.float32):
    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, device=device, dtype=tdtype)

    def gen():
        return (make(shape), make(shape), 2), {}

    def foo(alpha, beta, n):
        if n < 0:
            result = alpha - beta
        else:
            result = alpha + beta

        return alpha, result

    thunder_fn = thunder_compile(foo)

    args, kwargs = gen()
    thunder_info = thunder_fn(*args, **kwargs)

    return "simple-number-conditional", gen, foo, thunder_info["fusion"], thunder_info["profile_info"]


def simple_kwarg_conditional_constructor(shape=(64, 64), device="cuda", dtype=thunder.float32):
    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, device=device, dtype=tdtype)

    def gen():
        return (make(shape), make(shape)), {"n": 2}

    def foo(alpha, beta, n):
        if n < 0:
            result = alpha - beta
        else:
            result = alpha + beta

        return alpha, result

    thunder_fn = thunder_compile(foo)

    args, kwargs = gen()
    thunder_info = thunder_fn(*args, **kwargs)

    return "simple-kwarg-conditional", gen, foo, thunder_info["fusion"], thunder_info["profile_info"]


#
# nanoGPT benchmarks
#
# TODO: maybe put these in their own file?


# Taken from nanogpt_model.py
@dataclass
class GPTConfig:
    name: str = ""
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


nanogpt_info = (
    BenchmarkArg(
        name="config",
        description="The nanogpt configuration to use. Options are gpt2, gpt2-medium, gpt2-large, and gpt2-xl. Default is gpt2-medium. See the nanogpt model for details.",
        default="gpt2-medium",
    ),
    BenchmarkArg(
        name="dropout",
        description="The dropout probability to use. Default is .1.",
        default=0.1,
    ),
    BenchmarkArg(
        name="inshape",
        description="The shape of the input. Default is (8, 64).",
        default=(8, 64),
    ),
    BenchmarkArg(
        name="device",
        description="The device to run on. Default is 'cuda'.",
        default="cuda",
    ),
    BenchmarkArg(
        name="dtype",
        description="The device of the model. Default is 'cuda'.",
        default=thunder.float32,
    ),
    BenchmarkArg(
        name="indices_dtype",
        description="The dtype of the input. Default is int64.",
        default=thunder.int64,
    ),
)


def nanogpt_constructor(
    *,
    config,
    dropout,
    inshape,
    device,
    dtype,
    indices_dtype,
):
    # Taken from nanogpt_model.py
    configs = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }

    tdtype = ttorch.torch_dtype(dtype)
    config = GPTConfig(name=config, dropout=dropout, **configs[config])
    m = nanogpt_model.GPT(config).to(device=device, dtype=tdtype)
    tom = thunder.make_traced(
        m, executor="nvfuser", _preprocess=True, _info=True, _return_fusion=True, _profile_info=True
    )

    indices_tdtype = ttorch.torch_dtype(indices_dtype)

    def gen():
        return (make_tensor(inshape, low=0, high=255, device=device, dtype=indices_tdtype),), {}

    inp, _ = gen()

    # TODO: if not benchmarking thunder, don't run thunder+nvFuser here
    # TODO: in the future acquire thunder's fusion object here, too
    thunder_info = tom(inp[0], None)
    thunder_profile = thunder_info["profile_info"]

    # TODO: wrap the fusion
    def _thunder_wrapper(inp):
        thunder_info = tom(inp, None)
        return thunder_info["result"]

    return f"nanogpt-{config}", gen, m, _thunder_wrapper, thunder_profile


nanogpt_block_info = (
    BenchmarkArg(
        name="config",
        description="The nanogpt configuration to use. Options are gpt2, gpt2-medium, gpt2-large, and gpt2-xl. Default is gpt2-medium. See the nanogpt model for details.",
        default="gpt2-medium",
    ),
    BenchmarkArg(
        name="dropout",
        description="The dropout probability to use. Default is .1.",
        default=0.1,
    ),
    BenchmarkArg(
        name="batch_dims",
        description="The batch dimensions to use for the input. (The input will have innermost dimensions of (config.block_size, config.n_embd). Default is (8,).",
        default=(8,),
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


# TODO: @mike -- is block_size what we want for seq len here?
# TODO: refactor these nanogpt component benchmarks to use a common format
def nanogpt_block_constructor(
    config,
    dropout,
    batch_dims,
    device,
    dtype,
):
    # Taken from nanogpt_model.py
    configs = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }

    tdtype = ttorch.torch_dtype(dtype)
    config = GPTConfig(name=config, dropout=dropout, **configs[config])
    m = nanogpt_model.Block(config).to(device=device, dtype=tdtype)
    tom = thunder.make_traced(
        m, executor="nvfuser", _preprocess=True, _info=True, _return_fusion=True, _profile_info=True
    )

    def gen():
        return (
            make_tensor(
                batch_dims
                + (
                    config.block_size,
                    config.n_embd,
                ),
                device=device,
                dtype=tdtype,
            ),
        ), {}

    inp, _ = gen()
    # TODO: in the future acquire thunder's fusion object here, too
    thunder_info = tom(inp[0])
    thunder_profile = thunder_info["profile_info"]

    # TODO: wrap the fusion
    def _thunder_wrapper(inp):
        thunder_info = tom(inp)
        return thunder_info["result"]

    return f"nanogpt-{config}-block", gen, m, _thunder_wrapper, thunder_profile


nanogpt_csa_info = (
    BenchmarkArg(
        name="config",
        description="The nanogpt configuration to use. Options are gpt2, gpt2-medium, gpt2-large, and gpt2-xl. Default is gpt2-medium. See the nanogpt model for details.",
        default="gpt2-medium",
    ),
    BenchmarkArg(
        name="dropout",
        description="The dropout probability to use. Default is .1.",
        default=0.1,
    ),
    BenchmarkArg(
        name="batch_dims",
        description="The batch dimensions to use for the input. (The input will have innermost dimensions of (config.block_size, config.n_embd). Default is (8,).",
        default=(8,),
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


# TODO: @mike -- is block_size what we want for seq len here?
def nanogpt_csa_constructor(
    config,
    dropout,
    batch_dims,
    device,
    dtype,
):
    # Taken from nanogpt_model.py
    configs = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }

    tdtype = ttorch.torch_dtype(dtype)
    config = GPTConfig(name=config, dropout=dropout, **configs[config])
    m = nanogpt_model.CausalSelfAttention(config).to(device=device, dtype=tdtype)
    tom = thunder.make_traced(
        m, executor="nvfuser", _preprocess=True, _info=True, _return_fusion=True, _profile_info=True
    )

    def gen():
        return (
            make_tensor(
                batch_dims
                + (
                    config.block_size,
                    config.n_embd,
                ),
                device=device,
                dtype=tdtype,
            ),
        ), {}

    inp, _ = gen()

    # TODO: in the future acquire thunder's fusion object here, too
    thunder_info = tom(inp[0])
    thunder_profile = thunder_info["profile_info"]

    # TODO: wrap the fusion
    def _thunder_wrapper(inp):
        thunder_info = tom(inp)
        return thunder_info["result"]

    return f"nanogpt-{config}-csa", gen, m, _thunder_wrapper, thunder_profile


nanogpt_mlp_info = (
    BenchmarkArg(
        name="config",
        description="The nanogpt configuration to use. Options are gpt2, gpt2-medium, gpt2-large, and gpt2-xl. Default is gpt2-medium. See the nanogpt model for details.",
        default="gpt2-medium",
    ),
    BenchmarkArg(
        name="dropout",
        description="The dropout probability to use. Default is .1.",
        default=0.1,
    ),
    BenchmarkArg(
        name="batch_dims",
        description="The batch dimensions to use for the input. (The input will have an innermost dimension of length config.n_embd). Default is (8,).",
        default=(8,),
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


# TODO: @mike -- shape? (i.e. include seq len)
def nanogpt_mlp_constructor(
    config,
    dropout,
    batch_dims,
    device,
    dtype,
):
    # Taken from nanogpt_model.py
    configs = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }

    tdtype = ttorch.torch_dtype(dtype)
    config = GPTConfig(name=config, dropout=dropout, **configs[config])
    m = nanogpt_model.MLP(config).to(device=device, dtype=tdtype)
    tom = thunder.make_traced(
        m, executor="nvfuser", _preprocess=True, _info=True, _return_fusion=True, _profile_info=True
    )

    def gen():
        return (make_tensor(batch_dims + (config.n_embd,), device=device, dtype=tdtype),), {}

    inp, _ = gen()

    # TODO: if thunder+nvFuser isn't an executor, don't run thunder here
    # TODO: in the future acquire thunder's fusion object here, too
    thunder_info = tom(inp[0])
    thunder_profile = thunder_info["profile_info"]

    # TODO: wrap the fusion
    def _thunder_wrapper(inp):
        thunder_info = tom(inp)
        return thunder_info["result"]

    return f"nanogpt-{config}-mlp", gen, m, _thunder_wrapper, thunder_profile


nanogpt_gelu_info = (
    BenchmarkArg(
        name="shape",
        description="The shape of the input. Default is (1024, 1024).",
        default=(1024, 1024),
    ),
    BenchmarkArg(
        name="device",
        description="The device of the input. Default is 'cuda'.",
        default="cuda",
    ),
    BenchmarkArg(
        name="dtype",
        description="The dtype of the input. Default is float32.",
        default=thunder.float32,
    ),
)


def nanogpt_gelu_constructor(shape, device, dtype):
    def new_gelu(a):
        return 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * torch.pow(a, 3.0))))

    tdtype = ttorch.torch_dtype(dtype)
    thunder_fn = thunder.make_traced(
        new_gelu, executor="nvfuser", _preprocess=True, _info=True, _return_fusion=True, _profile_info=True
    )

    def gen():
        return (make_tensor(shape, device=device, dtype=tdtype),), {}

    inp, _ = gen()

    # TODO: if thunder+nvFuser isn't an executor, don't run thunder here
    # TODO: in the future acquire thunder's fusion object here, too
    thunder_info = thunder_fn(inp[0])
    thunder_profile = thunder_info["profile_info"]

    return f"nanogpt-gelu", gen, new_gelu, thunder_info["fusion"], thunder_profile


hf_bart_self_attn_info = (
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


def hf_bart_self_attn_constructor(
    config,
    dropout,
    sequences,
    seq_length,
    device,
    dtype,
):
    # Taken from HuggingFace Bart-Large model config:
    # https://huggingface.co/facebook/bart-large/blob/main/config.json
    configs = {
        "bart-large": dict(embed_dim=1024, num_heads=16),
    }

    cfg = configs[config]

    tdtype = ttorch.torch_dtype(dtype)
    bart_model = hf_bart_self_attn.BartAttention(cfg["embed_dim"], cfg["num_heads"], dropout=dropout)
    m = bart_model.to(device=device, dtype=tdtype)
    tom = thunder.make_traced(
        m, executor="nvfuser", _preprocess=True, _info=True, _return_fusion=True, _profile_info=True
    )

    def gen():
        return (
            make_tensor(
                (
                    sequences,
                    seq_length,
                    cfg["embed_dim"],
                ),
                device=device,
                dtype=tdtype,
            ),
            make_tensor(
                (
                    sequences,
                    1,
                    1,
                    seq_length,
                ),
                device=device,
                dtype=tdtype,
            ),
        ), {}

    inp, _ = gen()

    # TODO: if thunder+nvFuser isn't an executor, don't run thunder here
    # TODO: in the future acquire thunder's fusion object here, too
    thunder_info = tom(*inp)
    thunder_profile = thunder_info["profile_info"]

    # TODO: wrap the fusion
    def _thunder_wrapper(inp, mask):
        thunder_info = tom(inp, mask)
        return thunder_info["result"]

    return f"hf-{config}-self-attn", gen, m, _thunder_wrapper, thunder_profile


# NOTE: new benchmark style, in development
def benchmark(name, gen, torch_fn, thunder_fn, thunder_profile, *, executors=None, warmup_iters=5, iters=20):
    executors = get_executors(torch_fn, thunder_fn, executors)

    print(f"Benchmarking {name}")

    for name, ex in executors:
        # With some versions of PyTorch, this will complain with
        #   "Must call `torch._dynamo.reset()` before changing backends."
        # Other versions of PyTorch will complain that torch has no attribute _dynamo
        try:
            torch._dynamo.reset()
        except:
            pass

        stats = time_ns(ex, gen)
        _prettyprint_stats(name, stats)


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
# value format: constructor fn, sequence of BenchmarkArgs, string description
benchmarks = {
    # Reduction benchmarks
    # FIXME: _preprocess needs to support *args, **kwargs
    # "var": var_constructor,
    # Control flow benchmarks
    # "simple-number-conditional": simple_number_conditional_constructor,
    # "simple-kwarg-conditional": simple_kwarg_conditional_constructor,
    # nanogpt benchmarks
    "nanogpt": (nanogpt_constructor, nanogpt_info, "nanogpt forward"),
    "nanogpt-block": (nanogpt_block_constructor, nanogpt_block_info, "nanogpt block module forward"),
    "nanogpt-csa": (nanogpt_csa_constructor, nanogpt_csa_info, "nanogpt CausalSelfAttention (CSA) module forward"),
    "nanogpt-mlp": (nanogpt_mlp_constructor, nanogpt_mlp_info, "nanogpt MLP module forward"),
    "nanogpt-gelu": (nanogpt_gelu_constructor, nanogpt_gelu_info, "nanogpt gelu function forward"),
    "hf-bart-self-attn": (hf_bart_self_attn_constructor, hf_bart_self_attn_info, "hf bart self-attn module forward"),
}


# Tries to extract a Python object from a string
def _extract_python_obj(s):
    try:
        a = None
        cstr = f"a={s}"
        _locals = {}
        exec(cstr, None, _locals)
        return _locals["a"]
    except Exception as e:
        return s


def _print_benchmarks():
    print(f"There are {len(benchmarks)} available benchmarks:\n")

    for idx, (k, (_, _, desc)) in enumerate(benchmarks.items()):
        print(f"{idx}. {k}. {desc}")

    print("\nFor more information on how to call a particular benchmark, use <benchmark_name> -info")


def _print_info(args):
    print(f"This benchmark accepts up to {len(args)} kwargs")
    for arg in args:
        print(f"{arg.name}. {arg.description}")


# TODO: provide a programmatic way to run these benchmarks and acquire the results
# TODO: allow a file to specify which benchmarks to run and how
if __name__ == "__main__":
    benchmark_name = sys.argv[1]

    # Handles special -view or -v command which lists all benchmarks
    # Short-circuits if view is requested
    if benchmark_name == "-view" or benchmark_name == "-v":
        _print_benchmarks()
        sys.exit(0)

    benchmark_constructor, benchmark_info, benchmark_desc = benchmarks[benchmark_name]

    # Handles kwargs, which control the benchmark itself
    #   These are specified like dtype=float16
    additional_args = sys.argv[2:]
    kwargs = tuple(x for x in additional_args if "=" in x)
    directives = tuple(x for x in additional_args if "=" not in x)

    kwargs = {k: _extract_python_obj(v) for k, v in (x.split("=") for x in kwargs)}

    # Handles directives, which describe how the benchmark is to be run
    #   These are specified like -x "('thunder', 'torch.compile')"
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-info",
        "-i",
        default=None,
        action="store_true",
        help="Displays information on how to call the specified benchmark.",
    )
    parser.add_argument("-view", "-v", default=None, action="store_true", help="Lists all available benchmarks")
    parser.add_argument(
        "-executors",
        "-x",
        default="('torch-eager', 'torch.compile', 'torch.compile_nvfuser_prims', 'thunder+nvfuser')",
        help="Specifies the executors to collect statistics for. Default is all executors, -x \"('torch-eager', 'torch.compile', 'torch.compile_nvfuser_prims', 'thunder+nvfuser')\"",
    )
    parser.add_argument(
        "-iters",
        default=20,
        help="Specifies the number of post-warmup iterations to use to collect statistics for each executor. Default is -iters 20",
    )
    parser.add_argument(
        "-warmup_iters",
        default=5,
        help="The number of warmup iterations to run for each executor before collecting statistics. Default is -warmup_iters 5",
    )

    parsed_directives = parser.parse_args(directives)

    view_requested = parsed_directives.view is not None
    info_requested = parsed_directives.info is not None
    executors = _extract_python_obj(parsed_directives.executors)
    iters = parsed_directives.iters
    warmup_iters = parsed_directives.warmup_iters

    # Short-circuits if view is requested
    if view_requested:
        _print_benchmarks()
        sys.exit(0)

    # Short-circuits if info is requested
    if info_requested:
        _print_info(benchmark_info)
        sys.exit(0)

    # Resolves dtypes
    def _resolve_dtype(x):
        try:
            # TODO: this should test if the dtype string is valid instead of just trying to acquire
            #   a thunder attribute
            dtype = getattr(thunder, x)
        except Exception:
            return x

        return dtype

    # Constructs kwargs from benchmark info and user-supplied kwargs
    user_kwargs = tree_map(_resolve_dtype, kwargs)
    actual_kwargs = {x.name: x.default for x in benchmark_info}
    actual_kwargs.update(user_kwargs)

    # NOTE: in the future, if running multiple benchmarks from one process, the benchmarks
    #   should probably be run in a subprocess to minimize executor caching
    name, gen, torch_fn, thunder_fn, thunder_profile = benchmark_constructor(**actual_kwargs)
    benchmark(
        name, gen, torch_fn, thunder_fn, thunder_profile, executors=executors, iters=iters, warmup_iters=warmup_iters
    )
