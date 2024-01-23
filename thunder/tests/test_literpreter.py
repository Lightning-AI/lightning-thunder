from collections.abc import Iterable, Iterator, Sequence
from functools import partial, wraps
from itertools import product

import sys
import dis
from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.core.jit import is_jitting, jit, JITError

from thunder.core.jit_ext import litjit
import thunder.clang as clang
import thunder.torch as ltorch
import thunder.core.prims as prims

#
# Test suite for the litjit extension of the Python interpreter
#


def skipif_python_3_11_plus(f):
    if sys.version_info >= (3, 11):
        return pytest.mark.skip(f, reason=f"not yet implemented for Python 3.11+, got {sys.version_info=}")
    return f


# This test is just here to prevent pytest from complaining that it couldn't find any tests on Python 3.11
#   This can be removed as soon as some tests are passing on 3.11
def test_dummy():
    pass


@skipif_python_3_11_plus
def test_binary_add_tensors():
    def foo(a, b):
        return a + b

    jfoo = litjit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@skipif_python_3_11_plus
def test_torch_add_tensors():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = litjit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@skipif_python_3_11_plus
def test_torch_add_tensors_closure():
    def foo(a, b):
        c = a + b

        def bar():
            return torch.add(c, 1)

        return bar()

    jfoo = litjit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@skipif_python_3_11_plus
def test_intermediate_torch_operations():
    def foo(a, b):
        c = a + b
        d = torch.sub(c, b)
        e = torch.mul(d, a)
        f = torch.matmul(e, c)
        g = [e, f]
        return torch.cat(g)

    jfoo = litjit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@skipif_python_3_11_plus
@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1933", raises=BaseException)
def test_add_numbers():
    def foo(a, b):
        return a + b

    jfoo = litjit(foo)

    a = 2
    b = 3

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


_test_add_global_global = 2


@skipif_python_3_11_plus
@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1935", raises=BaseException)
def test_global_fails():
    def foo():
        return _test_add_global_global

    jfoo = litjit(foo)

    with pytest.raises(NotImplementedError):
        jfoo()


@skipif_python_3_11_plus
@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1936", raises=BaseException)
def test_nonlocal_outside_interpreter_fails():
    def foo():
        x = 3

        def bar():
            nonlocal x
            x = 4

        jbar = litjit(bar)

        jbar()

        return x

    with pytest.raises(NotImplementedError):
        foo()
