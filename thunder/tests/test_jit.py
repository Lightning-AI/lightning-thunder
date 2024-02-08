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
from thunder.core.jit import JITError
from thunder.core.jit_ext import minimal_thunder_jit


#
# Test suite for the thunder.jit entrypoint
#


# TODO Refactor this parameterization so it's easy to apply
@pytest.mark.parametrize(
    "jit", (thunder.jit, minimal_thunder_jit), ids=("thunder.jit", "thunder.jit-translate_functions")
)
def test_simple_addition(jit):
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_torch_addition(jit):
    def foo(a, b):
        return torch.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_sharp_edges_pass():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = thunder.jit(foo, sharp_edges="error")

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


_test_load_global_sharp_edge_global = 3


def test_load_global_sharp_edge():
    def foo(a):
        return torch.add(a, _test_load_global_sharp_edge_global)

    jfoo = thunder.jit(foo, sharp_edges="error")

    a = torch.randn((2, 2))

    with pytest.raises(JITError):
        jfoo(a)
