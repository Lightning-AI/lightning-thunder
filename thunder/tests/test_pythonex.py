import pytest
import math

import thunder


def _run_cache_symbolic_values(fn, ref_fn, *args):
    jit_fn = thunder.jit(fn, cache="symbolic values")
    out = jit_fn(*args)

    out_ref = ref_fn(*args)
    assert out == out_ref


def test_fmod():
    def foo(a, b):
        return a % b

    _run_cache_symbolic_values(foo, foo, 2.0, 1.3)


def test_bitwise_or():
    def foo(a, b):
        return a | b

    _run_cache_symbolic_values(foo, foo, 3, 5)


def test_bitwise_and():
    def foo(a, b):
        return a & b

    _run_cache_symbolic_values(foo, foo, 3, 5)


def test_bitwise_xor():
    def foo(a, b):
        return a ^ b

    _run_cache_symbolic_values(foo, foo, 3, 5)


def test_math_atan2():
    def foo(a, b):
        # TODO: calling through math.atan2 bakes in constant, this needs to be investigated.
        return thunder.clang.atan2(a, b)

    # NOTE: we have thunder.clang in foo, which cannot be run with non-proxy
    _run_cache_symbolic_values(foo, math.atan2, 2.0, 1.3)


def test_math_fmod():
    def foo(a, b):
        return thunder.clang.fmod(a, b)

    _run_cache_symbolic_values(foo, math.fmod, 2.0, 1.3)
