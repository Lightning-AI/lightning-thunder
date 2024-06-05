import pytest

import thunder
import thunder.core
from thunder.executors.sdpaex import sdpa_ex
from thunder.tests.framework import requiresCUDA, run_snippet
from thunder.tests.opinfos import get_opinfo
from collections import namedtuple

CudaVersion = namedtuple("CudaVersion", "major minor")

def _run_cache_symbolic_values(fn, ref_fn, *args):
    jit_fn = thunder.jit(fn, cache="symbolic values")
    out = jit_fn(*args)

    out_ref = ref_fn(*args)
    assert out == out_ref

def test_fmod():
    def foo(a, b):
        return a % b
        # note to myself. this one is not getting traced.
        #return math.atan2(a, b)
        #return thunder.clang.atan2(a, b)

    _run_cache_symbolic_values(foo, foo, 2.0, 1.3)
    #out_ref = math.atan2(a, b)

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

    # TODO: we cannot run foo(2.0, 1.3), inputs is converted to NumberProxy. I think this needs to be fixed.
    _run_cache_symbolic_values(foo, math.atan2, 2.0, 1.3)
