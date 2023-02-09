import argparse
import math
import multiprocessing
import operator
import os
import time
import warnings
from functools import partial, reduce
from multiprocessing import Process

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

# This file contains custom nvFuser-related benchmarks.


def median(l):
    if len(l) == 0:
        raise ValueError("Empty lists have no median value!")

    s = sorted(l)

    if len(s) % 2 == 1:
        return s[len(s) // 2]

    return (s[len(s) // 2] + s[len(s) // 2 - 1]) / 2


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
def time_thunder_ns(fn, gen, *, warmup_iters=5, iters=20):
    fn = thunder.make_traced(fn, executor="nvfuser", _info=True, _return_fusion=True, _profile_info=True)

    def _helper(fn_):
        args, kwargs = gen()
        torch.cuda.synchronize()
        start = time.time_ns()
        result = fn_(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time_ns()
        return end - start, result

    # NOTE: Initial run
    initial_time, initial_result = _helper(fn)
    fusion = initial_result["fusion"]

    # NOTE: Warmups (a common benchmarking technique)
    for _ in range(warmup_iters):
        _helper(fn)

    # Acquires lazy Thunder stats
    elapsed = []
    for _ in range(iters):
        t, result = _helper(fn)
        elapsed.append(t)

    # Computes lazy statistics
    avg = reduce(operator.add, elapsed, 0) / iters
    m = median(elapsed)

    stats = {
        "warmup_iters": warmup_iters,
        "iters": iters,
        "initial": initial_time,
        "initial_result": initial_result,
        "lazy_average": avg,
        "lazy_median": m,
        "lazy_final": elapsed[-1],
        "lazy_final_result": result,
    }

    # Computes fusion results
    elapsed = []
    for _ in range(iters):
        t, result = _helper(fusion)
        elapsed.append(t)

    # Computes fusion statistics
    avg = reduce(operator.add, elapsed, 0) / iters
    m = median(elapsed)

    stats.update(
        {
            "fusion_average": avg,
            "fusion_median": m,
            "fusion_final": elapsed[-1],
            "fusion_final_result": result,
        }
    )

    return stats


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


def _prettyprint_thunder_nvfuser_profile_info(profile):
    prettyprint_profile(profile)


def _prettyprint_thunder_nvfuser_stats(stats):
    us = "\u03BCs"

    def _print_breakdown(s):
        ns = None
        meta = None
        if s == "initial":
            ns = stats["initial"]
            meta = stats["initial_result"]["meta"]
        elif s == "lazy":
            ns = stats["lazy_final"]
            meta = stats["lazy_final_result"]["meta"]

        iter_desc = None
        if s == "initial":
            iter_desc = "first"
        elif s == "lazy":
            iter_desc = "final lazy"

        print(f"The {iter_desc} iteration took {ns_to_us(ns)}{us}, and can be broken down as...")
        a_time = meta["acquisition_time"]
        t_time = meta["translation_time"]
        e_time = meta["invocation_time"]
        print(f"{ns_to_us(a_time)}{us}, {percent(a_time, ns)} of the time, was spent in program acquisition")
        print(
            f"{ns_to_us(t_time)}{us}, {percent(t_time, ns)} of the time, was spent translating the program to a fusion definition"
        )
        print(f"{ns_to_us(e_time)}{us}, {percent(e_time, ns)} of the time, was spent invoking nvFuser.execute()")

        accounted_time = a_time + t_time + e_time
        unaccounted_time = ns - accounted_time
        print(
            f"{ns_to_us(unaccounted_time)}{us}, {percent(unaccounted_time, ns)} of the time, is unaccounted for, but is probably how long the kernels took to execute."
        )

    print("Thunder+nvFuser results:")
    print(f"The median time of {stats['iters']} lazy post-warmup iterations was {ns_to_us(stats['lazy_median'])}{us}")
    print(
        f"The median time of {stats['iters']} fused post-warmup iterations was {ns_to_us(stats['fusion_median'])}{us}"
    )
    _print_breakdown("initial")
    _print_breakdown("lazy")


def _prettyprint_stats(name, stats):
    us = "\u03BCs"

    print(f"{name} results:")
    print(f"The median time of {stats['iters']} post-warmup iterations was {ns_to_us(stats['median'])}{us}")
    print(f"The initial iteration took {ns_to_us(stats['initial'])}{us}")
    print(f"The final iteration took {ns_to_us(stats['final'])}{us}")


def _compare_stats(thunder_name, thunder_stats, name_b, stats_b):
    thunder_initial = thunder_stats["initial"]
    other_initial = stats_b["initial"]

    print(f"Results of comparing Thunder and {name_b}:")
    if thunder_initial < other_initial:
        print(
            f"{thunder_name} was initially faster than {name_b}, taking only {percent(thunder_initial, other_initial)} of the time"
        )
    else:
        print(
            f"{name_b} was initially faster than {thunder_name}, taking only {percent(other_initial, thunder_initial)} of the time"
        )

    thunder_lazy_median = thunder_stats["lazy_median"]
    other_median = stats_b["median"]
    name = f"Lazy {thunder_name}"

    if thunder_lazy_median < other_median:
        print(
            f"{name} was faster post-warmup than {name_b}, taking only {percent(thunder_lazy_median, other_median)} of the time"
        )
    else:
        print(
            f"{name_b} was faster post-warmup than {name}, taking only {percent(other_median, thunder_lazy_median)} of the time"
        )

    thunder_fused_median = thunder_stats["fusion_median"]
    name = f"Fused {thunder_name}"

    if thunder_fused_median < other_median:
        print(
            f"{name} was faster post-warmup than {name_b}, taking only {percent(thunder_fused_median, other_median)} of the time"
        )
    else:
        print(
            f"{name_b} was faster post-warmup than {name}, taking only {percent(other_median, thunder_fused_median)} of the time"
        )


def _benchmark(name, *, gen, iters, thunder_fn, other_name, other_fn):
    print(f"Benchmark: {name}")

    thunder_stats = time_thunder_ns(thunder_fn, gen, iters=iters)
    profile_info = thunder_stats["lazy_final_result"]["profile_info"]

    _prettyprint_thunder_nvfuser_profile_info(profile_info)
    _prettyprint_thunder_nvfuser_stats(thunder_stats)

    other_stats = time_ns(other_fn, gen, iters=iters)
    _prettyprint_stats(other_name, other_stats)

    _compare_stats("Thunder + nvFuser", thunder_stats, other_name, other_stats)


#
# Elementwise binary benchmarks
#


def _add_nvfuser_vs_pt2_factory(shape, *, iters, make_arg):
    def gen():
        a = make_arg(shape)
        b = make_arg(shape)
        return (a, b), {}

    # Constructs pt2 function
    def _add(a, b):
        return a + b

    pt2_fn = torch.compile(_add)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(f"add_{shape_str}", gen=gen, iters=iters, thunder_fn=tlang.add, other_name="pt2", other_fn=pt2_fn)


def add_64x64(iters, make_arg):
    _add_nvfuser_vs_pt2_factory((64, 64), iters=iters, make_arg=make_arg)


def add_kwargs_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        a = make_arg(shape)
        b = make_arg(shape)
        return (), {"a": a, "b": b}

    # Constructs pt2 function
    def _add(a, b):
        return a + b

    pt2_fn = torch.compile(_add)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(f"add_kwargs_{shape_str}", gen=gen, iters=iters, thunder_fn=tlang.add, other_name="pt2", other_fn=pt2_fn)


def add_1024x1024(iters, make_arg):
    _add_nvfuser_vs_pt2_factory((1024, 1024), iters=iters, make_arg=make_arg)


def add_4096x4(iters, make_arg):
    _add_nvfuser_vs_pt2_factory((4096, 4), iters=iters, make_arg=make_arg)


def add_4x4096(iters, make_arg):
    _add_nvfuser_vs_pt2_factory((4, 4096), iters=iters, make_arg=make_arg)


def add_dozen_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        args = []
        for _ in range(12):
            args.append(make_arg(shape))
        return tuple(args), {}

    def _add_dozen(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11):
        return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11

    pt2_fn = torch.compile(_add_dozen)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(f"add_dozen_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_dozen, other_name="pt2", other_fn=pt2_fn)


def add_hundred_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        args = []
        for _ in range(100):
            args.append(make_arg(shape))
        return tuple(args), {}

    # fmt: off
    def _add_hundred(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99):
        return a0 + a1 + a2 + a3 + a4 + a5+ a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31 + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39 + a40 + a41 + a42 + a43 + a44 + a45 + a46 + a47 + a48 + a49 + a50 + a51 + a52 + a53 + a54 + a55 + a56 + a57 + a58 + a59 + a60 + a61 + a62 + a63 + a64 + a65 + a66 + a67 + a68 + a69 + a70 + a71 + a72 + a73 + a74 + a75 + a76 + a77 + a78 + a79 + a80 + a81 + a82 + a83 + a84 + a85 + a86 + a87 + a88 + a89 + a90 + a91 + a92 + a93 + a94 + a95 + a96 + a97 + a98 + a99
    # fmt: on

    pt2_fn = torch.compile(_add_hundred)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(
        f"add_hundred_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_hundred, other_name="pt2", other_fn=pt2_fn
    )


def add_dozen_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        args = []
        for _ in range(12):
            args.append(make_arg(shape))
        return tuple(args), {}

    def _add_dozen(*args):
        cur = args[0]
        for a in args[1:]:
            cur = cur + a
        return cur

    pt2_fn = torch.compile(_add_dozen)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(f"add_dozen_{shape_str}", gen=gen, iters=iters, thunder_fn=_add_dozen, other_name="pt2", other_fn=pt2_fn)


def add_stack100_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        a = make_arg(shape)
        b = make_arg(shape)
        return (a, b), {}

    def _add_stack100(a, b):
        cur = a
        for _ in range(100):
            cur = cur + b

        return cur

    pt2_fn = torch.compile(_add_stack100)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(
        f"add_stack100_{shape_str}",
        gen=gen,
        iters=iters,
        thunder_fn=_add_stack100,
        other_name="pt2",
        other_fn=pt2_fn,
    )


def add_stack1000_64x64(iters, make_arg):
    shape = (64, 64)

    def gen():
        a = make_arg(shape)
        b = make_arg(shape)
        return (a, b), {}

    def _add_stack1000(a, b):
        cur = a
        for _ in range(1000):
            cur = cur + b

        return cur

    pt2_fn = torch.compile(_add_stack1000)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(
        f"add_stack1000_{shape_str}",
        gen=gen,
        iters=iters,
        thunder_fn=_add_stack1000,
        other_name="pt2",
        other_fn=pt2_fn,
    )


def _add_contiguous_transposed_nvfuser_vs_pt2_factory(shape, *, iters, make_arg):
    def gen():
        a = make_arg(shape)
        b = make_arg(shape).T
        return (a, b), {}

    # Makes PyTorch2 function
    def _add(a, b):
        return a + b

    pt2_fn = torch.compile(_add)

    shape_str = "x".join(str(l) for l in shape)
    _benchmark(
        f"add_{shape_str}_contiguous_transposed",
        gen=gen,
        iters=iters,
        thunder_fn=tlang.add,
        other_name="pt2",
        other_fn=pt2_fn,
    )


def add_1024x1024_contiguous_transposed(iters, make_arg):
    _add_contiguous_transposed_nvfuser_vs_pt2_factory((1024, 1024), iters=iters, make_arg=make_arg)


#
# Elementwise unary benchmarks
#


def _elementwise_unary_nvfuser_vs_pt2_factory(shape, *, thunder_op, torch_op, iters, make_arg):
    def gen():
        return (make_arg(shape),), {}

    # Makes pt2 function
    def _foo(a):
        return torch_op(a)

    pt2_fn = torch.compile(_foo)

    shape_str = "x".join(str(l) for l in shape)
    name = f"{torch_op.__name__}{shape_str}"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_op, other_name="pt2", other_fn=pt2_fn)


def abs_64x64(iters, make_arg):
    _elementwise_unary_nvfuser_vs_pt2_factory(
        (64, 64), thunder_op=tlang.abs, torch_op=torch.abs, iters=iters, make_arg=make_arg
    )


#
# Reduction benchmarks
#


def _all_reduce_nvfuser_vs_pt2_factory(shape, *, thunder_op, torch_op, iters, make_arg):
    def gen():
        return (make_arg(shape),), {}

    # Makes pt2 function
    def _foo(a):
        return torch_op(a)

    pt2_fn = torch.compile(_foo)

    shape_str = "x".join(str(l) for l in shape)
    name = f"{torch_op.__name__}{shape_str}_all_reduce"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_op, other_name="pt2", other_fn=pt2_fn)


def var_1024x1024_all_reduce(iters, make_arg):
    _all_reduce_nvfuser_vs_pt2_factory(
        (1024, 1024), thunder_op=ttorch.var, torch_op=torch.var, iters=iters, make_arg=make_arg
    )


def simple_number_conditional(iters, make_arg):
    shape = (64, 64)

    def gen():
        return (make_arg(shape), make_arg(shape), 2), {}

    def foo(alpha, beta, n):
        if n < 0:
            result = alpha - beta
        else:
            result = alpha + beta

        return alpha, result

    pt2_fn = torch.compile(foo)

    name = f"simple_number_conditional"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=foo, other_name="pt2", other_fn=pt2_fn)


def simple_kwarg_conditional(iters, make_arg):
    shape = (64, 64)

    def gen():
        return (make_arg(shape), make_arg(shape)), {"n": 2}

    def foo(alpha, beta, n):
        if n < 0:
            result = alpha - beta
        else:
            result = alpha + beta

        return alpha, result

    pt2_fn = torch.compile(foo)

    name = f"simple_kwarg_conditional"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=foo, other_name="pt2", other_fn=pt2_fn)


#
# nanoGPT benchmarks
#
# TODO: maybe put these in their own file?


def new_gelu(a):
    return 0.5 * a * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * torch.pow(a, 3.0))))


def thunder_new_gelu(a):
    return 0.5 * a * (1.0 + ttorch.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * ttorch.pow(a, 3.0))))


def _nanogpt_new_gelu_vs_pt2_factory(shape, *, iters, make_arg):
    def gen():
        a = make_arg(shape)
        return (a,), {}

    pt2_fn = torch.compile(new_gelu)

    shape_str = "x".join(str(l) for l in shape)
    name = f"nanogpt_gelu_{shape_str}"
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_new_gelu, other_name="pt2", other_fn=pt2_fn)


def nanogpt_new_gelu_64x64(iters, make_arg):
    _nanogpt_new_gelu_vs_pt2_factory((64, 64), iters=iters, make_arg=make_arg)


def nanogpt_new_gelu_512x512(iters, make_arg):
    _nanogpt_new_gelu_vs_pt2_factory((512, 512), iters=iters, make_arg=make_arg)


def nanogpt_new_gelu_1024x1024(iters, make_arg):
    _nanogpt_new_gelu_vs_pt2_factory((1024, 1024), iters=iters, make_arg=make_arg)


class NanoGPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout()

    def forward(self, a):
        b = self.c_fc(a)
        c = new_gelu(b)
        d = self.c_proj(c)
        e = self.dropout(d)
        return e


def NanoGPTMLP_forward_functional(a, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
    b = torch.nn.functional.linear(a, c_fc_weight, c_fc_bias)
    c = new_gelu(b)
    d = torch.nn.functional.linear(c, c_proj_weight, c_proj_bias)
    e = torch.nn.functional.dropout(d)
    return e


def thunder_NanoGPTMLP_forward_functional(a, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
    b = ttorch.linear(a, c_fc_weight, c_fc_bias)
    c = thunder_new_gelu(b)
    d = ttorch.linear(c, c_proj_weight, c_proj_bias)
    e = ttorch.dropout(d)
    return e


# TODO: update factory to accept different model sizes
def _nanogpt_mlp_factory(n, *, dtype, iters, make_arg):
    class Config:
        pass

    # These numbers from the "gpt" config
    config = Config()
    config.n_embd = 768
    config.n_head = 12
    config.block_size = 1024

    tdtype = ttorch.torch_dtype(dtype)
    mlp = NanoGPTMLP(config).to("cuda", dtype=tdtype)

    def gen():
        a = make_arg((n, config.n_embd))
        return (a, mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight, mlp.c_proj.bias), {}

    thunder_fn = thunder_NanoGPTMLP_forward_functional
    pt_fn = NanoGPTMLP_forward_functional
    pt2_fn = torch.compile(NanoGPTMLP_forward_functional)

    shape_str = f"{n}x768"
    name = f"nanogpt_mlp_{shape_str}_{dtype}"

    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_fn, other_name="PyTorch", other_fn=pt_fn)
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_fn, other_name="pt2", other_fn=pt2_fn)


def nanogpt_mlp_8x768_float32(iters, make_arg):
    _nanogpt_mlp_factory(8, dtype=dtypes.float32, iters=iters, make_arg=make_arg)


class NanoGPTCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # TODO: re-enable me
        # regularization
        # self.attn_dropout = nn.Dropout(config.dropout)
        # self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


def NanoGPTCausalSelfAttention_forward_functional(
    x, c_attn_weight, c_attn_bias, n_embd, n_head, bias, c_proj_weight, c_proj_bias
):
    B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = torch.nn.functional.linear(x, c_attn_weight, c_attn_bias).split(n_embd, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))

    att = torch.softmax(att, dim=-1)
    att = torch.nn.functional.dropout(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    y = torch.nn.functional.linear(y, c_proj_weight, c_proj_bias)
    y = torch.nn.functional.dropout(y)

    return y


def thunder_NanoGPTCausalSelfAttention_forward_functional(
    x, c_attn_weight, c_attn_bias, n_embd, n_head, bias, c_proj_weight, c_proj_bias
):
    B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = ttorch.linear(x, c_attn_weight, c_attn_bias).split(n_embd, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))

    att = ttorch.softmax(att, dim=-1)
    att = ttorch.dropout(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    y = ttorch.linear(y, c_proj_weight, c_proj_bias)
    y = ttorch.dropout(y)

    return y


# TODO: allow other gpt sizes to be specified
def _nanogpt_csa_factory(n, *, dtype, iters, make_arg):
    class Config:
        pass

    config = Config()
    config.n_embd = 768
    config.n_head = 12
    config.block_size = 1024
    config.dropout = True

    tdtype = ttorch.torch_dtype(dtype)
    csa = NanoGPTCausalSelfAttention(config).to("cuda", dtype=tdtype)

    def gen():
        a = make_arg((2, n, config.n_embd))
        return (
            a,
            csa.c_attn.weight,
            csa.c_attn.bias,
            config.n_embd,
            config.n_head,
            csa.bias,
            csa.c_proj.weight,
            csa.c_proj.bias,
        ), {}

    thunder_fn = thunder_NanoGPTCausalSelfAttention_forward_functional
    pt_fn = NanoGPTCausalSelfAttention_forward_functional
    pt2_fn = torch.compile(NanoGPTCausalSelfAttention_forward_functional)

    shape_str = f"2x{n}x768"
    name = f"nanogpt_csa_{shape_str}_{dtype}"

    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_fn, other_name="PyTorch", other_fn=pt_fn)
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_fn, other_name="pt2", other_fn=pt2_fn)


# TODO: revise with real shapes
def nanogpt_csa_2x8x768_float32(iters, make_arg):
    _nanogpt_csa_factory(8, dtype=dtypes.float32, iters=iters, make_arg=make_arg)


class NanoGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NanoGPTCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = NanoGPTMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def NanoGPTBlock_forward_functional(
    a,
    ln_1_normalized_shape,
    ln_1_weight,
    ln_1_bias,
    ln_1_eps,
    csa_c_attn_weight,
    csa_c_attn_bias,
    n_embd,
    n_head,
    csa_bias,
    csa_c_proj_weight,
    csa_c_proj_bias,
    ln_2_normalized_shape,
    ln_2_weight,
    ln_2_bias,
    ln_2_eps,
    mlp_c_fc_weight,
    mlp_c_fc_bias,
    mlp_c_proj_weight,
    mlp_c_proj_bias,
):
    b = torch.nn.functional.layer_norm(a, ln_1_normalized_shape, ln_1_weight, ln_1_bias, ln_1_eps)
    a = a + NanoGPTCausalSelfAttention_forward_functional(
        b,
        csa_c_attn_weight,
        csa_c_attn_bias,
        n_embd,
        n_head,
        csa_bias,
        csa_c_proj_weight,
        csa_c_proj_bias,
    )

    c = torch.nn.functional.layer_norm(a, ln_2_normalized_shape, ln_2_weight, ln_2_bias, ln_2_eps)
    a = a + NanoGPTMLP_forward_functional(c, mlp_c_fc_weight, mlp_c_fc_bias, mlp_c_proj_weight, mlp_c_proj_bias)

    return a


def thunder_NanoGPTBlock_forward_functional(
    a,
    ln_1_normalized_shape,
    ln_1_weight,
    ln_1_bias,
    ln_1_eps,
    csa_c_attn_weight,
    csa_c_attn_bias,
    n_embd,
    n_head,
    csa_bias,
    csa_c_proj_weight,
    csa_c_proj_bias,
    ln_2_normalized_shape,
    ln_2_weight,
    ln_2_bias,
    ln_2_eps,
    mlp_c_fc_weight,
    mlp_c_fc_bias,
    mlp_c_proj_weight,
    mlp_c_proj_bias,
):

    b = ttorch.layer_norm(a, ln_1_normalized_shape, ln_1_weight, ln_1_bias, ln_1_eps)
    a = a + thunder_NanoGPTCausalSelfAttention_forward_functional(
        b,
        csa_c_attn_weight,
        csa_c_attn_bias,
        n_embd,
        n_head,
        csa_bias,
        csa_c_proj_weight,
        csa_c_proj_bias,
    )

    c = ttorch.layer_norm(a, ln_2_normalized_shape, ln_2_weight, ln_2_bias, ln_2_eps)
    a = a + thunder_NanoGPTMLP_forward_functional(c, mlp_c_fc_weight, mlp_c_fc_bias, mlp_c_proj_weight, mlp_c_proj_bias)

    return a


# TODO: allow other gpt sizes to be specified
def _nanogpt_block_factory(n, *, dtype, iters, make_arg):
    class Config:
        pass

    config = Config()
    config.n_embd = 768
    config.n_head = 12
    config.block_size = 1024
    config.dropout = True

    tdtype = ttorch.torch_dtype(dtype)
    block = NanoGPTBlock(config).to("cuda", dtype=tdtype)

    def gen():
        a = make_arg((2, n, config.n_embd))
        return (
            a,
            block.ln_1.normalized_shape,
            block.ln_1.weight,
            block.ln_1.bias,
            block.ln_1.eps,
            block.attn.c_attn.weight,
            block.attn.c_attn.bias,
            config.n_embd,
            config.n_head,
            block.attn.bias,
            block.attn.c_proj.weight,
            block.attn.c_proj.bias,
            block.ln_2.normalized_shape,
            block.ln_2.weight,
            block.ln_2.bias,
            block.ln_2.eps,
            block.mlp.c_fc.weight,
            block.mlp.c_fc.bias,
            block.mlp.c_proj.weight,
            block.mlp.c_proj.bias,
        ), {}

    thunder_fn = thunder_NanoGPTBlock_forward_functional
    pt_fn = NanoGPTBlock_forward_functional
    pt2_fn = torch.compile(NanoGPTBlock_forward_functional)

    shape_str = f"2x{n}x768"
    name = f"nanogpt_block{shape_str}_{dtype}"

    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_fn, other_name="PyTorch", other_fn=pt_fn)
    _benchmark(name, gen=gen, iters=iters, thunder_fn=thunder_fn, other_name="pt2", other_fn=pt2_fn)


# TODO: revise with real shapes
def nanogpt_block_2x8x768_float32(iters, make_arg):
    _nanogpt_block_factory(8, dtype=dtypes.float32, iters=iters, make_arg=make_arg)


benchmarks = {
    # Elementwise Binary benchmarks
    "add_64x64": add_64x64,
    "add_kwargs_64x64": add_kwargs_64x64,
    "add_dozen_64x64": add_dozen_64x64,
    "add_hundred_64x64": add_hundred_64x64,
    "add_stack100_64x64": add_stack100_64x64,
    "add_stack1000_64x64": add_stack1000_64x64,
    "add_dozen_64x64": add_dozen_64x64,  # Requires supporting *args
    "add_1024x1024": add_1024x1024,
    "add_4096x4": add_4096x4,
    "add_4x4096": add_4x4096,
    "add_1024x1024_contiguous_transposed": add_1024x1024_contiguous_transposed,
    # Elementwise Unary benchmarks
    "abs_64x64": abs_64x64,
    # Reduction benchmarks
    "var_1024x1024_all_reduce": var_1024x1024_all_reduce,  # Requires supporting sequence proxies
    # Control flow benchmarks
    "simple_number_conditional": simple_number_conditional,
    "simple_kwarg_conditional": simple_kwarg_conditional,
    # Network snippet benchmarks
    "nanogpt_new_gelu_64x64": nanogpt_new_gelu_64x64,
    "nanogpt_new_gelu_512x512": nanogpt_new_gelu_512x512,
    "nanogpt_new_gelu_1024x1024": nanogpt_new_gelu_1024x1024,
    "nanogpt_mlp_8x768_float32": nanogpt_mlp_8x768_float32,
    "nanogpt_csa_2x8x768_float32": nanogpt_csa_2x8x768_float32,
    "nanogpt_block_2x8x768_float32": nanogpt_block_2x8x768_float32,
}


def _run_benchmark(benchmark_fn, *args):
    p = Process(target=benchmark_fn, args=args)
    p.start()
    p.join()


# TODO: allow specifying iters, dtype, benchmarks
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", action="store", help="float32, int64, ...")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Acquires dtype (default is float32)
    dtype = thunder.float32
    if args.dtype is not None:
        dtype = getattr(thunder, args.dtype)
        if not thunder.dtypes.is_dtype(dtype):
            raise ValueError("Unknown dtype {args.dtype} specified!")

    # TODO: allow specifying a particular CUDA device
    device = "cuda"

    iters = 20

    make_arg = partial(make_tensor, device=device, dtype=ttorch.torch_dtype(dtype))

    multiprocessing.set_start_method("spawn")

    # Ignores warnings during benchmarks
    # NOTE: setting this environment variable effective sets
    #   warnings.simplewarningsfilter('ignore') in each (sub)process
    # NOTE: pt2 will throw extraneous warnings
    os.environ["PYTHONWARNINGS"] = "ignore"
    for k, v in benchmarks.items():
        _run_benchmark(v, iters, make_arg)
