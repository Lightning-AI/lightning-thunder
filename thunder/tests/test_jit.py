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
import thunder.clang as clang
import thunder.core.prims as prims

#
# Test suite for the thunder.jit entrypoint
#

#
# Basic functionality tests
#


# TODO Refactor this parameterization so it's easy to apply
@pytest.mark.parametrize(
    "jit", (thunder.jit, minimal_thunder_jit), ids=("thunder.jit", "thunder.jit-translate_functions")
)
def test_binary_add_tensors(jit):
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_binary_ops_compare_numbers(jit):
    cmp_ops = ["!=", "==", "<=", "<", ">", ">="]

    inps = (
        (5, 9),
        (2, 8),
        (8, 2),
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (5.5, 2.2),
        (-5.5, 2.2),
        (5, 2.2),
    )

    for op in cmp_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)

        for a, b in inps:
            assert jfoo(a, b) == foo(a, b)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_binary_ops_int_numbers(jit):
    # Issue https://github.com/Lightning-AI/lightning-thunder/issues/594 for more ops
    # "<<", ">>",
    int_ops = ["+", "&", "//", "*", "%", "|", "**", "-", "/", "^"]

    int_inps = (
        (5, 9),
        (2, 8),
        (8, 2),
    )

    for op in int_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.jit(bar)

        for a, b in int_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_binary_ops_bool_numbers(jit):
    bool_ops = ["+", "&", "//", "*", "%", "|", "**", "-", "/", "^"]

    bool_inps = (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    )

    for op in bool_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.jit(bar)
        for a, b in bool_inps:
            if op not in {"//", "/", "%"} or b:  # could check exceptions for these
                assert jfoo(a, b) == foo(a, b)
                assert jbar(a, b) == bar(a, b)
            else:
                for fn in foo, jfoo, bar, jbar:
                    with pytest.raises(ZeroDivisionError):
                        fn(a, b)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_binary_ops_float_numbers(jit):
    float_ops = ["+", "//", "*", "%", "**", "-", "/"]

    float_inps = (
        (5.5, 2.2),
        (-5.5, 2.2),
        (5, 2.2),
    )

    for op in float_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.jit(bar)
        for a, b in float_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_binary_ops_complex_numbers(jit):
    float_ops = ["+", "*", "**", "-", "/"]

    float_inps = (
        (5.5j, 2.2j),
        (-5.5j, 2.2j),
        (5j, 2.2j),
    )

    for op in float_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.jit(bar)
        for a, b in float_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_binary_add_tensor_number(jit):
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = 3

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_hasattr_on_proxies(jit):
    def foo(a, b):
        if hasattr(a, "__why_would_it__"):
            raise Exception("Nobody expects the Spanish Inquisition")
        if hasattr(b, "__why_would_it__"):
            raise Exception("Oh well, never happens")
        return a + b

    a = torch.randn((2, 2))
    b = 3

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@pytest.mark.parametrize(
    "jit", (thunder.jit, minimal_thunder_jit), ids=("thunder.jit", "thunder.jit-translate_functions")
)
def test_clang_add_tensors(jit):
    def foo(a, b):
        return clang.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = a + b

    assert_close(actual, expected)


@pytest.mark.parametrize(
    "jit", (thunder.jit, minimal_thunder_jit), ids=("thunder.jit", "thunder.jit-translate_functions")
)
def test_prim_add_tensors(jit):
    def foo(a, b):
        return prims.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = a + b

    assert_close(actual, expected)


@pytest.mark.parametrize(
    "jit", (thunder.jit, minimal_thunder_jit), ids=("thunder.jit", "thunder.jit-translate_functions")
)
def test_python_fn_binary_add_tensors(jit):
    def bar(a, b):
        return a + b

    def foo(a, b):
        return bar(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_torch_add_tensors(jit):
    def foo(a, b):
        return torch.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


#
# String tests
#


@pytest.mark.parametrize("jit", (minimal_thunder_jit,), ids=("thunder.jit-translate_functions",))
def test_string_return(jit):
    def foo(s):
        return s

    jfoo = thunder.jit(foo)
    actual = jfoo("hi")
    expected = "hi"

    assert actual == expected


#
# Sharp edges tests
#


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
