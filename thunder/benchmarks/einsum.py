from collections.abc import Callable
from functools import partial, wraps
from collections.abc import Sequence

import pytest
import torch
import thunder

from thunder.benchmarks import EinsumBenchmark
from thunder.benchmarks.targets import (
    make_setup,
    fwd_executors,
    fwd_executor_ids,
    grad_executors,
    grad_executors_ids,
    thunder_gradv1,
    thunder_torchcompile_gradv1,
    wrap_for_benchmark,
)


def _instantiate_benchmark_env(
    shapes: Sequence[Sequence[int]],
    equation: str,
    executor: Callable,
    device: str = "cuda:0",
    dtype: thunder.dtypes.dtype = thunder.bfloat16,
    requires_grad: bool = False,
) -> Callable:
    bench: Benchmark = EinsumBenchmark(
        shapes=shapes, equation=equation, device=device, dtype=dtype, requires_grad=requires_grad
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    return setup, fn


_test_size_map = {
    "small": 16,
    "medium": 128,
    "large": 512,
}


def single_dim_contraction_cases(size: str = "small", left_broadcasts: None | bool = None):
    n = _test_size_map[size]

    lhs_shape = [2, n, n]
    rhs_shape = [n, n]

    if left_broadcasts is not None:
        if left_broadcasts:
            lhs_shape[-1] = 1
        else:
            rhs_shape[0] = 1

    return (lhs_shape, rhs_shape), "bij,jk"


def multidim_contraction_cases(size: str = "small", left_broadcasts: None | bool = None):
    n = _test_size_map[size]

    lhs_shape = [2, 8, n, n]
    rhs_shape = [2, n, 8, n]

    if left_broadcasts is not None:
        if left_broadcasts:
            lhs_shape[-1] = 1
        else:
            rhs_shape[-1] = 1

    return (lhs_shape, rhs_shape), "bijk,bklj->bli"


class TestEinsumBenchmarks:
    @pytest.mark.parametrize(
        "executor,",
        fwd_executors,
        ids=fwd_executor_ids,
    )
    @pytest.mark.parametrize(
        "size,",
        _test_size_map.keys(),
    )
    @pytest.mark.parametrize(
        "sample_gen,", (single_dim_contraction_cases, multidim_contraction_cases), ids=("singledim", "multidim")
    )
    @pytest.mark.parametrize("left_broadcasts,", (None, True, False), ids=("", "left_broadcasts", "right_broadcasts"))
    def test_einsum_fwd(
        self, benchmark, executor: None | Callable, size: str, sample_gen: Callable, left_broadcasts: None | bool
    ):
        setup, fn = _instantiate_benchmark_env(
            *sample_gen(size, left_broadcasts), executor=executor, requires_grad=False
        )
        benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)

    @pytest.mark.parametrize(
        "executor,",
        [ge for ge in grad_executors if ge not in (thunder_gradv1, thunder_torchcompile_gradv1)],
        ids=[gei for gei in grad_executors_ids if gei not in ("thunder-gradv1", "thunder+torchcompile-gradv1")],
    )
    @pytest.mark.parametrize(
        "size,",
        _test_size_map.keys(),
    )
    @pytest.mark.parametrize(
        "sample_gen,", (single_dim_contraction_cases, multidim_contraction_cases), ids=("singledim", "multidim")
    )
    @pytest.mark.parametrize(
        "left_broadcasts,",
        # False/right_broadcasts is disabled because of
        # https://github.com/NVIDIA/Fuser/issues/1590.
        # TODO: update once the issue is fixed.
        (None, True),
        ids=("", "left_broadcasts"),
    )
    def test_einsum_grad(
        self, benchmark, executor: None | Callable, size: str, sample_gen: Callable, left_broadcasts: None | bool
    ):
        setup, fn = _instantiate_benchmark_env(
            *sample_gen(size, left_broadcasts), executor=executor, requires_grad=True
        )
        benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)
