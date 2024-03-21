from collections.abc import Iterable, Iterator, Sequence
from functools import partial, wraps
from itertools import product
import random
import math

import sys
import dis
from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.core.jit_ext import minimal_thunder_jit, ThunderSharpEdgeError
import thunder.clang as clang
import thunder.core.prims as prims

#
# Test suite for the thunder.jit entrypoint
#

#
# Args, kwargs, varargs, varkwargs tests
#


def test_basic_kwargs():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a=a, b=b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_tuple_kwargs():
    def foo(a, b):
        return a + b[0]

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a=a, b=(a, b))
    expected = foo(a, (a, b))

    assert_close(actual, expected)


def test_kwargs_inorder():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(b=b, a=a)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_kwonly_args():
    def foo(*, a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a=a, b=b)
    expected = foo(a=a, b=b)

    assert_close(actual, expected)


def test_posonly_and_kwonly_args():
    def foo(a, /, b, *, c):
        return a + b + c

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))
    c = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b=b, c=c)
    expected = foo(a, b, c=c)

    assert_close(actual, expected)


def test_varargs():
    def foo(*args):
        a, b = args
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_positional_args_and_varargs():
    def foo(a, b, *args):
        c = a + b
        for x in args:
            c = c + x
        return x

    args = []
    for _ in range(5):
        args.append(torch.randn((2, 2)))

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b, *args)
    expected = foo(a, b, *args)

    assert_close(actual, expected)


def test_positional_args_varargs_and_kwargs():
    def foo(a, b, *args, z):
        c = a + b + z
        for x in args:
            c = c + x
        return x

    args = []
    for _ in range(5):
        args.append(torch.randn((2, 2)))

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))
    z = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b, *args, z=z)
    expected = foo(a, b, *args, z=z)

    assert_close(actual, expected)


def test_varkwargs():
    def foo(**kwargs):
        return kwargs["a"] + kwargs["b"][0]

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    actual = jfoo(a=a, b=[b])
    expected = foo(a=a, b=[b])

    assert_close(actual, expected)


def test_empty_varkwargs():
    def foo(**kwargs):
        return len(kwargs)

    jfoo = thunder.functional.jit(foo)

    actual = jfoo()
    expected = foo()

    assert_close(actual, expected)


def test_args_varargs_kwargs_and_varkwargs():
    def foo(a, b, /, c, *args, d, e, **kwargs):
        accum = 0
        accum = accum + a + b + c
        for arg in args:
            accum = accum + arg
        accum = accum + d + e
        for k, v in kwargs.items():
            accum = accum + v

        return accum

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))
    c = torch.randn((2, 2))
    args = [torch.randn((2, 1)), torch.randn((1, 2))]
    d = torch.randn((1, 1))
    e = torch.randn((1,))
    kwargs = {"x": torch.randn((2, 2)), "y": torch.randn((2, 2))}

    actual = jfoo(a, b, c, *args, d=d, e=e, **kwargs)
    expected = foo(a, b, c, *args, d=d, e=e, **kwargs)

    assert_close(actual, expected)


def test_default_parameters():
    def foo(a, b=3):
        return a + b

    jfoo = thunder.functional.jit(foo)
    a = torch.randn((2, 2))

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)

    actual = jfoo(a, 4)
    expected = foo(a, 4)

    assert_close(actual, expected)


def test_default_parameters_tensor():
    def foo(a, b=torch.randn((2, 2))):
        return a + b

    jfoo = thunder.functional.jit(foo)
    a = torch.randn((2, 2))

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)

    actual = jfoo(a, 4)
    expected = foo(a, 4)

    assert_close(actual, expected)


#
# Binary operation tests
#


def test_binary_add_tensors():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_binary_ops_compare_numbers():
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
        jfoo = thunder.functional.jit(foo)

        for a, b in inps:
            assert jfoo(a, b) == foo(a, b)


def test_binary_ops_int_numbers():
    # TODO: see issue "Implement logical and arithmetic left and right shifts"
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
        jfoo = thunder.functional.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.functional.jit(bar)

        for a, b in int_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


def test_binary_ops_bool_numbers():
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
        jfoo = thunder.functional.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.functional.jit(bar)
        for a, b in bool_inps:
            if op not in {"//", "/", "%"} or b:  # could check exceptions for these
                assert jfoo(a, b) == foo(a, b)
                assert jbar(a, b) == bar(a, b)
            else:
                for fn in foo, jfoo, bar, jbar:
                    with pytest.raises(ZeroDivisionError):
                        fn(a, b)


def test_binary_ops_float_numbers():
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
        jfoo = thunder.functional.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.functional.jit(bar)
        for a, b in float_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


def test_binary_ops_complex_numbers():
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
        jfoo = thunder.functional.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.functional.jit(bar)
        for a, b in float_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


def test_binary_add_tensor_number():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = 3

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_hasattr_on_proxies():
    def foo(a, b):
        if hasattr(a, "__why_would_it__"):
            raise Exception("Nobody expects the Spanish Inquisition")
        if hasattr(b, "__why_would_it__"):
            raise Exception("Oh well, never happens")
        return a + b

    a = torch.randn((2, 2))
    b = 3

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_clang_add_tensors():
    def foo(a, b):
        return clang.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = a + b

    assert_close(actual, expected)


def test_prim_add_tensors():
    def foo(a, b):
        return prims.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = a + b

    assert_close(actual, expected)


def test_python_fn_binary_add_tensors():
    def bar(a, b):
        return a + b

    def foo(a, b):
        return bar(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors():
    def foo(a, b):
        return torch.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


#
# String tests
#


def test_string_return():
    def foo(s):
        return s

    jfoo = thunder.functional.jit(foo)
    actual = jfoo("hi")
    expected = "hi"

    assert actual == expected


def test_binary_add_strings():
    def foo(a, b):
        return a + b

    jfoo = thunder.functional.jit(foo)
    actual = jfoo("he", "llo")
    expected = "hello"

    assert actual == expected


#
# None tests
#


def test_none_return():
    def foo(n):
        return n

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(None)
    expected = foo(None)

    assert actual == expected


def test_none_condition():
    def foo(a, b):
        if b is None:
            return a
        return a + b

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    actual = jfoo(a, b)
    expected = foo(a, b)
    assert_close(actual, expected)

    actual = jfoo(a, None)
    expected = foo(a, None)
    assert_close(actual, expected)


def test_filtering_nones():
    def foo(seq):
        accum = 0
        for x in (x for x in seq if x is not None):
            accum = accum + x
        return accum

    jfoo = thunder.functional.jit(foo)

    seq = (0, 1, None, None, 2, 3, None, 4, 5, None)

    actual = jfoo(seq)
    expected = foo(seq)
    assert_close(actual, expected)


#
# slice inputs
#


def test_slice_input():
    def foo(lst, slc):
        return lst[slc], slc

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]
    slc = slice(0, 2, 1)

    actual = jfoo(lst, slc)
    expected = foo(lst, slc)

    assert actual == expected


#
# ellipsis tests
#


def test_ellipsis_input():
    def foo(a, ell):
        return a[ell], ell

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))

    actual = jfoo(a, ...)
    expected = foo(a, ...)

    actual_t, actual_ell = actual
    expected_t, expected_ell = expected

    assert_close(actual_t, expected_t)
    assert actual_ell is expected_ell


#
# torch.dtype inputs
#


def test_torch_dtypes():
    def foo(a, dtyp):
        return a.to(dtyp)

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    dtyp = torch.float64

    actual = jfoo(a, dtyp)
    expected = foo(a, dtyp)
    assert_close(actual, expected, check_dtype=True)


#
# torch.device inputs
#


def test_torch_device_input():
    def foo(a, dev):
        return a.to(dev)

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    dev = torch.device("cpu")

    actual = jfoo(a, dev)
    expected = foo(a, dev)

    assert_close(actual, expected)


#
# Tuple tests
#


def test_tuple_len():
    def foo(tup):
        return len(tup) + 5

    jfoo = thunder.functional.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_binary_subscr():
    def foo(tup):
        return tup[0], tup[1:3]

    jfoo = thunder.functional.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_assignment():
    def foo(tup):
        tup[0] = 5

    jfoo = thunder.functional.jit(foo)

    tup = (0, 1, 2, 3, 4)

    with pytest.raises(TypeError):
        jfoo(tup)


def test_tuple_unpack_repack():
    def foo(tup):
        a, b, *_ = tup
        return a, (b, a)

    jfoo = thunder.functional.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_list_conversion():
    def foo(tup):
        return list(tup)

    jfoo = thunder.functional.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_enumerate():
    def foo(tup):
        accum = 0
        for x in tup:
            accum += x
        return accum

    jfoo = thunder.functional.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_addition():
    def foo(tup0, tup1):
        return tup0 + tup1

    jfoo = thunder.functional.jit(foo)

    tup0 = (0, 1, 2, 3, 4)
    tup1 = (5, 6, 7, 8, 9, 10)

    actual = jfoo(tup0, tup1)
    expected = foo(tup0, tup1)

    assert actual == expected


def test_tuple_list_addition():
    def foo(tup):
        return tup + [5, 6, 7]

    jfoo = thunder.functional.jit(foo)

    tup = (0, 1, 2, 3, 4)

    with pytest.raises(TypeError):
        jfoo(tup)


def test_nested_tuples():
    def foo(tup):
        tup0, tup1 = tup
        tup2, tup3 = tup0
        return tup1, tup3

    jfoo = thunder.functional.jit(foo)

    tup3 = (3, 4)
    tup2 = (1, 2)
    tup1 = (-1, 0)
    tup0 = (tup2, tup3)
    tup = (tup0, tup1)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_nested_tuples_with_tensors():
    def foo(tup):
        tup0, tup1 = tup
        tup2, tup3 = tup0
        return tup1, tup3, tup2[0] + tup3[1]

    jfoo = thunder.functional.jit(foo)

    a = torch.randn(2, 2)
    b = torch.randn(2, 2)

    tup3 = (3, 4)
    tup2 = (a, b)
    tup1 = (-1, 0)
    tup0 = (tup2, tup3)
    tup = (tup0, tup1)

    actual = jfoo(tup)
    expected = foo(tup)

    assert_close(actual, expected)


def test_tuple_equality():
    def foo(tup0, tup1):
        return tup0 == tup1

    jfoo = thunder.functional.jit(foo)

    tup0 = (1, 3, 5)
    tup1 = (1, 3, 5)

    actual = jfoo(tup0, tup1)
    expected = foo(tup0, tup1)

    assert_close(actual, expected)

    tup2 = (4, 6)
    actual = jfoo(tup0, tup2)
    expected = foo(tup0, tup2)

    assert_close(actual, expected)


def test_tuple_bool():
    def foo(tup):
        return bool(tup)

    jfoo = thunder.functional.jit(foo)

    tup0 = (1, 3, 5)

    actual = jfoo(tup0)
    expected = foo(tup0)

    assert_close(actual, expected)

    tup1 = []

    actual = jfoo(tup1)
    expected = foo(tup1)

    assert_close(actual, expected)


#
# torch.Size tests
#
# NOTE torch.Size is treated as a tuple


def test_torchsize():
    def foo(a, shape):
        return a.reshape(shape)

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((4, 4))
    s = torch.Size((16, 1))

    actual = jfoo(a, s)
    expected = foo(a, s)

    assert_close(actual, expected)


#
# List tests
#


def test_list_len():
    def foo(lst):
        return len(lst) + 2

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    actual = jfoo(lst)
    expected = foo(lst)

    assert_close(actual, expected)


def test_list_binary_subscr():
    def foo(lst):
        return lst[0], lst[1:3]

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    actual = jfoo(lst)
    expected = foo(lst)

    assert_close(actual, expected)


def test_list_unpack_repack():
    def foo(lst):
        a, b, *_ = lst
        return [a, b]

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    actual = jfoo(lst)
    expected = foo(lst)

    assert_close(actual, expected)


def test_list_tuple_conversion():
    def foo(lst):
        return tuple(lst)

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    actual = jfoo(lst)
    expected = foo(lst)

    assert actual == expected


def test_list_enumerate():
    def foo(lst):
        accum = 0
        for x in lst:
            accum += x
        return accum

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    actual = jfoo(lst)
    expected = foo(lst)

    assert actual == expected


def test_list_addition():
    def foo(lst0, lst1):
        return lst0 + lst1

    jfoo = thunder.functional.jit(foo)

    lst0 = [0, 1, 2, 3, 4]
    lst1 = [5, 6, 7, 8, 9, 10]

    actual = jfoo(lst0, lst1)
    expected = foo(lst0, lst1)

    assert actual == expected


def test_tuple_list_addition():
    def foo(lst, tup):
        return tup + lst

    jfoo = thunder.functional.jit(foo)

    lst = [-1, -2, -3, -4, -5]
    tup = (0, 1, 2, 3, 4)

    with pytest.raises(TypeError):
        jfoo(lst, tup)


def test_list_basic_binary_subscr_assignment():
    def foo(lst):
        lst[0] = 5

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    with pytest.raises(NotImplementedError):
        jfoo(lst)


def test_list_reverse():
    def foo(lst):
        return lst.reverse()

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    actual = jfoo(lst)
    expected = foo(lst)

    assert_close(actual, expected)


def test_list_contains():
    def foo(lst):
        return 3 in lst

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    actual = jfoo(lst)
    expected = foo(lst)

    assert_close(actual, expected)


def test_list_slice_assignment():
    def foo(lst):
        lst[0:2] = [5, 1, 3]

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    with pytest.raises(NotImplementedError):
        jfoo(lst)


def test_list_append():
    def foo(lst):
        lst.append(3)

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    with pytest.raises(NotImplementedError):
        jfoo(lst)


def test_list_extend():
    def foo(lst):
        lst.extend([1, 2])

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    with pytest.raises(NotImplementedError):
        jfoo(lst)


def test_list_clear():
    def foo(lst):
        lst.clear()

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    with pytest.raises(NotImplementedError):
        jfoo(lst)


def test_list_insert():
    def foo(lst):
        lst.insert(2, 6)

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    with pytest.raises(NotImplementedError):
        jfoo(lst)


def test_list_pop():
    def foo(lst):
        return lst.pop()

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    with pytest.raises(NotImplementedError):
        jfoo(lst)


def test_list_remove():
    def foo(lst):
        lst.remove(2)

    jfoo = thunder.functional.jit(foo)

    lst = [0, 1, 2, 3, 4]

    with pytest.raises(NotImplementedError):
        jfoo(lst)


def test_list_equality():
    def foo(lst0, lst1):
        return lst0 == lst1

    jfoo = thunder.functional.jit(foo)

    lst0 = [1, 3, 5]
    lst1 = [1, 3, 5]

    actual = jfoo(lst0, lst1)
    expected = foo(lst0, lst1)

    assert_close(actual, expected)

    lst2 = [4, 6]
    actual = jfoo(lst0, lst2)
    expected = foo(lst0, lst2)

    assert_close(actual, expected)


def test_list_bool():
    def foo(lst):
        return bool(lst)

    jfoo = thunder.functional.jit(foo)

    lst0 = [1, 3, 5]

    actual = jfoo(lst0)
    expected = foo(lst0)

    assert_close(actual, expected)

    lst1 = []

    actual = jfoo(lst1)
    expected = foo(lst1)

    assert_close(actual, expected)


#
# Dict tests
#
def test_dict_getitem():
    def foo(d):
        return d["a"] + d[0]

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {"a": a, 0: b}

    actual = jfoo(d)
    expected = foo(d)

    assert_close(actual, expected)

    d = {"a": a, 0: 4.0}

    actual = jfoo(d)
    expected = foo(d)

    assert_close(actual, expected)


def test_dict_contains():
    def foo(d):
        return 3 in d

    jfoo = thunder.functional.jit(foo)

    d = {3: 4, 5: 6}

    actual = jfoo(d)
    expected = foo(d)

    assert_close(actual, expected)

    d = {0: 1}

    actual = jfoo(d)
    expected = foo(d)

    assert_close(actual, expected)


def test_dict_del():
    def foo(d):
        del d["a"]

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {"a": a, 0: b}

    with pytest.raises(NotImplementedError):
        jfoo(d)


def test_dict_eq():
    def foo(d0, d1):
        return d0 == d1

    jfoo = thunder.functional.jit(foo)

    d0 = {0: 0, 1: 1}
    d1 = {0: 0, 1: 1}

    actual = jfoo(d0, d1)
    expected = foo(d0, d1)

    assert_close(actual, expected)

    d2 = {0: 4}

    actual = jfoo(d0, d2)
    expected = foo(d0, d2)

    assert_close(actual, expected)


def test_dict_bitwise_or():
    def foo(d0, d1):
        return d0 | d1

    jfoo = thunder.functional.jit(foo)

    d0 = {0: 0, 1: 1}
    d1 = {0: 0, 4: 5}

    actual = jfoo(d0, d1)
    expected = foo(d0, d1)

    assert_close(actual, expected)


def test_dict_len():
    def foo(d):
        return len(d)

    jfoo = thunder.functional.jit(foo)

    d = {0: 0, 1: 1}

    actual = jfoo(d)
    expected = foo(d)

    assert_close(actual, expected)


def test_dict_iter():
    def foo(d):
        l = []
        for x in d:
            l.append(x)
        return l

    jfoo = thunder.functional.jit(foo)

    d = {0: 0, 1: 1, 5: 7, "x": "y", "hello": "goodbye", 9: [1, 2]}

    actual = jfoo(d)
    expected = foo(d)

    assert actual == expected


def test_dict_reverse():
    def foo(d):
        l = []
        for x in reversed(d):
            l.append(x)
        return l

    jfoo = thunder.functional.jit(foo)

    d = {0: 0, 1: 1, 5: 7, "x": "y", "hello": "goodbye", 9: [1, 2]}

    actual = jfoo(d)
    expected = foo(d)

    assert actual == expected


def test_dict_setitem():
    def foo(d):
        d[5] = 9

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {"a": a, 0: b}

    with pytest.raises(NotImplementedError):
        jfoo(d)


def test_dict_clear():
    def foo(d):
        d.clear()

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {"a": a, 0: b}

    with pytest.raises(NotImplementedError):
        jfoo(d)


def test_dict_get():
    def foo(d):
        return d.get(0, 5), d.get(10, 9)

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {"a": a, 0: b}

    actual = jfoo(d)
    expected = foo(d)

    assert actual == expected


def test_dict_items():
    def foo(d):
        accum = 0
        for k, v in d.items():
            accum = accum + k + v
        return accum

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {0: 1, 2: 3, 4: 5}

    actual = jfoo(d)
    expected = foo(d)

    assert actual == expected


def test_dict_keys():
    def foo(d):
        accum = 0
        for k in d.keys():
            accum = accum + k
        return accum

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {0: 1, 2: 3, 4: 5}

    actual = jfoo(d)
    expected = foo(d)

    assert actual == expected


def test_dict_values():
    def foo(d):
        accum = 0
        for v in d.values():
            accum = accum + v
        return accum

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {0: 1, 2: 3, 4: 5}

    actual = jfoo(d)
    expected = foo(d)

    assert actual == expected


def test_dict_pop():
    def foo(d):
        d.popitem()

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {"a": a, 0: b}

    with pytest.raises(NotImplementedError):
        jfoo(d)


def test_dict_setdefault():
    def foo(d):
        d.setdefault(5, 7)

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {"a": a, 0: b}

    with pytest.raises(NotImplementedError):
        jfoo(d)


def test_dict_update():
    def foo(d):
        d.update({1: 1})

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    d = {"a": a, 0: b}

    with pytest.raises(NotImplementedError):
        jfoo(d)

    def foo(d):
        d |= {1: 1}

    jfoo = thunder.functional.jit(foo)

    with pytest.raises(NotImplementedError):
        jfoo(d)


def test_dict_return():
    def foo(d):
        return d

    jfoo = thunder.functional.jit(foo)

    d = {0: 1, 2: 3, 4: 5}

    actual = jfoo(d)
    expected = foo(d)

    assert actual == expected


#
# General collection tests
#
def test_nested_collections():
    def foo(d):
        d0 = d[0]
        d1 = d[1]
        t1 = d[2]
        l0 = d[3]
        t0, l1 = t1
        a = d1[0][1] + t0[0]
        b = l0[0] + l1[1]

        return {
            "a": a,
            "b": b,
            "c": [d1, d0],
            0: (t0, l1),
            1: {0: t0, 1: a},
        }

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    l0 = [a, 1, {"x": 0, "y": 1}]
    l1 = [b, 0, 2]
    t0 = (a, a, -1, b)
    t1 = (t0, l1)
    l2 = [l1, t0, 5]

    d0 = {0: 0, 1: 1, "a": a}
    d1 = {0: d0, 1: l2}
    d2 = {0: d0, 1: d1, 2: t1, 3: l0}

    actual = jfoo(d2)
    expected = foo(d2)

    assert_close(actual, expected)


#
# Return value tests
#
# Return values must be proxies or (simple) printable objects (or both)


def test_return_number():
    def foo():
        return 5

    jfoo = thunder.functional.jit(foo)

    expected = foo()
    actual = jfoo()

    assert_close(expected, actual)


def test_return_object():
    def foo():
        return object()

    jfoo = thunder.functional.jit(foo)

    with pytest.raises(RuntimeError):
        jfoo()


def test_return_object_in_collection():
    def foo():
        return [object()]

    jfoo = thunder.functional.jit(foo)

    with pytest.raises(RuntimeError):
        jfoo()


def test_return_tuple():
    def foo(a, b):
        return (a, b)

    jfoo = thunder.functional.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


def test_return_list():
    def foo(a, b):
        return [a, b]

    jfoo = thunder.functional.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


def test_return_list_with_intermediates():
    def foo(a, b):
        l = [a, b]
        l.append(3)
        return l

    jfoo = thunder.functional.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


@pytest.mark.xfail(reason='issue: "jit-eager: allow sets as a return value"')
def test_return_set():
    def foo(a, b):
        return {a, b}

    jfoo = thunder.functional.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


def test_return_dict():
    def foo(a, b):
        return {1: a, 2: b}

    jfoo = thunder.functional.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


def test_return_varargs():
    def foo(*args):
        return args

    jfoo = thunder.functional.jit(foo)

    expected = foo(5, 3, 9, 9)
    actual = jfoo(5, 3, 9, 9)
    assert_close(expected, actual)


#
# No caching tests
#
# TODO RC1 Simple test that the option works and programs run as expected

#
# Same input caching tests
#
# TODO RC1 Verify works and fails as expected


#
# Constant values caching tests
#
def test_constant_values_caching():
    def foo(a, b):
        return a + b

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(a, b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    c = torch.rand((2, 1))

    expected = foo(a, c)
    actual = jfoo(a, c)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    actual = jfoo(a, c)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    expected = foo(b, c)
    actual = jfoo(b, c)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 3

    expected = foo(a, 1)
    actual = jfoo(a, 1)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 3

    actual = jfoo(a, 1)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 4

    expected = foo(a, 2)
    actual = jfoo(a, 2)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 4

    actual = jfoo(a, 2)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 5

    expected = foo("he", "llo")
    actual = jfoo("he", "llo")
    assert expected == actual

    assert thunder.cache_misses(jfoo) == 5
    assert thunder.cache_hits(jfoo) == 5

    actual = jfoo("he", "llo")
    assert expected == actual

    assert thunder.cache_misses(jfoo) == 5
    assert thunder.cache_hits(jfoo) == 6


def test_constant_values_caching_args_vs_kwargs():
    def foo(a, b):
        return a + b

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(b=b, a=a)

    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1


def test_constant_values_cache_is_not_shared():
    def foo(a, b):
        return a + b

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    jfoo = thunder.functional.jit(foo)

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0


def test_constant_values_caching_with_tuples():
    def foo(tup0, tup1):
        return tup0[0] + tup1[1]

    jfoo = thunder.functional.jit(foo)

    tup0 = (0, 1)
    tup1 = (2, 3)

    actual = jfoo(tup0, tup1)
    expected = foo(tup0, tup1)

    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    jfoo(tup0, tup1)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    tup2 = (0, 1)

    jfoo(tup2, tup1)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 2

    a = torch.randn((2, 2))
    tup3 = (a, 1)

    actual = jfoo(tup3, tup1)
    expected = foo(tup3, tup1)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    b = torch.randn((2, 2))
    tup4 = (b, 1)

    actual = jfoo(tup4, tup1)
    expected = foo(tup4, tup1)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 3

    c = torch.randn((2, 1))
    tup5 = (c, 1)

    actual = jfoo(tup5, tup1)
    expected = foo(tup5, tup1)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 3


def test_constant_values_caching_with_lists():
    def foo(lst0, lst1):
        return lst0[0] + lst1[1]

    jfoo = thunder.functional.jit(foo)

    lst0 = (0, 1)
    lst1 = (2, 3)

    actual = jfoo(lst0, lst1)
    expected = foo(lst0, lst1)

    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    jfoo(lst0, lst1)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    lst2 = (0, 1)

    jfoo(lst2, lst1)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 2

    a = torch.randn((2, 2))
    lst3 = (a, 1)

    actual = jfoo(lst3, lst1)
    expected = foo(lst3, lst1)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    b = torch.randn((2, 2))
    lst4 = (b, 1)

    actual = jfoo(lst4, lst1)
    expected = foo(lst4, lst1)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 3

    c = torch.randn((2, 1))
    tup5 = (c, 1)

    actual = jfoo(tup5, lst1)
    expected = foo(tup5, lst1)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 3


def test_constant_values_caching_with_kwargs():
    def foo(a, b):
        return a + b

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    expected = foo(a, b)
    actual = jfoo(a, b=b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(a, b=b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    actual = jfoo(b=b, a=a)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 2


def test_constant_values_caching_with_none():
    def foo(a, b):
        if b is None:
            return a * 2
        return a * b

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    expected = foo(a, b)
    actual = jfoo(a, b=b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(a, b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    actual = jfoo(a, None)
    expected = foo(a, None)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    actual = jfoo(a, None)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2


def test_constant_values_caching_with_torch_dtypes():
    def foo(a, dtyp):
        return a.to(dtyp)

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    dtyp0 = torch.bfloat16

    expected = foo(a, dtyp0)
    actual = jfoo(a, dtyp0)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(a, dtyp0)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    dtyp1 = torch.float64

    expected = foo(a, dtyp1)
    actual = jfoo(a, dtyp1)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1


# NOTE This intentionally tests empty tuples and lists together to ensure
#   () and [] generate distinct programs
def test_constant_values_empty_tuples_and_lists_caching():
    def foo(seq):
        return seq

    jfoo = thunder.functional.jit(foo)

    empty_tup = ()

    expected = foo(empty_tup)
    actual = jfoo(empty_tup)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(())
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    empty_list = []

    expected = foo(empty_list)
    actual = jfoo(empty_list)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    actual = jfoo(empty_list)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2


def test_constant_values_empty_dict_caching():
    def foo(d):
        return len(d)

    jfoo = thunder.functional.jit(foo)

    empty_dict = {}

    expected = foo(empty_dict)
    actual = jfoo(empty_dict)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo({})
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    d = {1: 1}
    actual = jfoo(d)
    expected = foo(d)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    d = {1: 1}
    actual = jfoo(d)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2


def test_constant_values_dict_caching():
    def foo(d):
        return d["a"] + d["b"]

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))
    d0 = {"a": a, "b": b}

    expected = foo(d0)
    actual = jfoo(d0)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(d0)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    c = torch.randn((2, 2))
    d1 = {"a": a, "b": c}

    expected = foo(d1)
    actual = jfoo(d1)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 2

    d = torch.randn((2, 1))
    d2 = {"a": a, "b": d}

    expected = foo(d2)
    actual = jfoo(d2)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    actual = jfoo(d2)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 3


def constant_values_varkwargs_caching():
    def foo(**kwargs):
        return kwargs["a"] + kwargs["b"]

    jfoo = thunder.functional.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    expected = foo(a=a, b=b)
    actual = jfoo(a=a, b=b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(a=a, b=b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    c = torch.randn((2, 2))

    expected = foo(a=a, b=c)
    actual = jfoo(a=a, b=c)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 2

    d = torch.randn((2, 1))

    expected = foo(a=a, b=d)
    actual = jfoo(a=a, b=d)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    actual = jfoo(a=a, b=d)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 3

    # Adding an unused key causes a cache miss
    expected = foo(a=a, b=d, c=3)
    actual = jfoo(a=a, b=d, c=3)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 2

    expected = foo(a=a, b=d, c=3)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 3


# NOTE Values like complex(math.inf, 0) and math.inf compare as equal
#   This test verifies that they are understood as distinct values
#   (since we understand numbers as type x value)
def test_constant_values_caching_float_complex_equality():
    def foo(a):
        return a

    jfoo = thunder.functional.jit(foo)

    a = complex(math.inf, 0)

    expected = foo(a)
    actual = jfoo(a)
    assert_close(actual, expected, check_dtype=True)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(a)
    assert_close(actual, expected, check_dtype=True)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    b = math.inf

    expected = foo(b)
    actual = jfoo(b)
    assert_close(actual, expected, check_dtype=True)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    actual = jfoo(b)
    assert_close(actual, expected, check_dtype=True)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    c = complex(math.inf, -math.inf)

    expected = foo(c)
    actual = jfoo(c)
    assert_close(actual, expected, check_dtype=True)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 2

    d = complex(math.nan, 0)

    expected = foo(d)
    actual = jfoo(d)
    assert_close(actual, expected, check_dtype=True, equal_nan=True)

    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 2

    actual = jfoo(d)
    assert_close(actual, expected, check_dtype=True, equal_nan=True)

    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 3

    e = math.nan

    expected = foo(e)
    actual = jfoo(e)
    assert_close(actual, expected, check_dtype=True, equal_nan=True)

    assert thunder.cache_misses(jfoo) == 5
    assert thunder.cache_hits(jfoo) == 3

    actual = jfoo(e)
    assert_close(actual, expected, check_dtype=True, equal_nan=True)

    assert thunder.cache_misses(jfoo) == 5
    assert thunder.cache_hits(jfoo) == 4


#
# Symbolic values caching tests
#


def test_symbolic_value_warning():
    def foo():
        return

    with pytest.warns(UserWarning):
        thunder.functional.jit(foo, cache="symbolic values")


#
# Graceful unsupported input handling tests
#


def test_callable_class_failure():
    m = torch.nn.Linear(10, 10)
    jfoo = thunder.functional.jit(m)
    a = torch.randn((10, 10))

    with pytest.raises(NotImplementedError):
        jfoo(a)


#
# Partial tests
#


def test_partial_simple():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    pfoo = partial(foo, b=b)
    jfoo = thunder.functional.jit(pfoo)

    actual = jfoo(a)
    expected = pfoo(a)

    assert_close(actual, expected)


def test_partial_partial_arg():
    def foo(a, b, c):
        return a + b + c

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))
    c = torch.randn((2, 2))

    pfoo = partial(partial(foo, c=c), b=b)
    jfoo = thunder.functional.jit(pfoo)

    actual = jfoo(a)
    expected = pfoo(a)

    assert_close(actual, expected)


@pytest.mark.xfail(reason="Support for partials with positional arguments is not yet implemented")
def test_partial_positional_arg():
    def foo(a, b, c):
        return a + b + c

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))
    c = torch.randn((2, 2))

    pfoo = partial(foo, a, c=c)
    jfoo = thunder.functional.jit(pfoo)

    actual = jfoo(b)
    expected = pfoo(b)

    assert_close(actual, expected)


#
# jit(jit...) tests
#


def test_jit_jit():
    def foo(a, b):
        return a + b

    jfoo = thunder.functional.jit(foo)

    def bar(a, b):
        return jfoo(a, b)

    jbar = thunder.functional.jit(bar)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    actual: torch.Tensor
    with pytest.warns():
        actual = jbar(a, b)

    expected = a + b
    assert_close(actual, expected)


#
# Sharp edges tests
#
# See thunder/core/options.py for a longer description of sharp edges and the option.


def test_sharp_edges_pass():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.functional.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


#
# Sharp edges -- Unexpected inputs
#

_test_load_global_sharp_edge_global = 3


def test_load_global_sharp_edge():
    def foo(a):
        return torch.add(a, _test_load_global_sharp_edge_global)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    a = torch.randn((2, 2))

    with pytest.raises(ThunderSharpEdgeError):
        jfoo(a)


_test_store_global_sharp_edge_global = 4


def test_store_global_sharp_edge():
    def foo():
        global _test_store_global_sharp_edge_global
        _test_store_global_sharp_edge_global = 5

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_globals_sharp_edge():
    def foo():
        g = globals()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_vars_sharp_edge():
    def foo():
        g = vars()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_locals_sharp_edge():
    def foo():
        l = locals()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_accessing_globals_through_function_sharp_edge():
    def foo():
        x = foo.__globals__

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_input_sharp_edge():
    def foo():
        inp = input()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


@pytest.mark.xfail(reason='issue: "sharp edges: loading closures"')
def test_input_closure_sharp_edge():
    x = 5

    def foo():
        return x

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


# NOTE This is NOT a sharp edge because we assume that loading modules and functions (globally or as closures) are
#   not sharp edges
def test_fn_closure_no_sharp_edge():
    def bar(x):
        return x

    def foo():
        return bar(5)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    actual = jfoo()
    expected = foo()
    assert_close(expected, actual)


def _test_fn_global_no_sharp_edge_fn():
    return 7


@pytest.mark.xfail(reason='issue: "sharp edge: allow function and module loads"')
def test_fn_global_no_sharp_edge():
    def foo(x):
        return x + _test_fn_global_no_sharp_edge_fn()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    actual = jfoo(2)
    expected = foo(2)
    assert_close(expected, actual)


def test_nonlocal_write_sharp_edge():
    x = 5

    def foo():
        nonlocal x
        x = 7

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_nonlocal_no_mutation_not_sharp_edge():
    def foo():
        # In this case, we do have a STORE_DEREF for x
        # as it is a co_cellvar (captured by nested_fn)
        x = 0

        def nested_fn():
            return x

        y = nested_fn()

        # This STORE_DEREF is safe as `x` is local here.
        x = 1
        return x, y

    jfoo = thunder.jit(foo, sharp_edges="error")

    assert_close(jfoo(), foo())


# NOTE This is NOT a sharp edge -- this test is to ensure this works as expected
def test_intermediate_closure_not_sharp_edge():
    def foo(x):
        def bar():
            return x

        return bar()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    expected = foo(5)
    actual = jfoo(5)

    assert_close(expected, actual)


def test_intermediate_nonlocal_not_sharp_edge():
    def foo(x):
        def bar():
            nonlocal x
            x = 9

        return x

        return bar()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    expected = foo(5)
    actual = jfoo(5)

    assert_close(expected, actual)


def test_intermediate_default_param_sharp_edge():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    def foo(a, b=b):
        return a + b

    def bar(a):
        return foo(a)

    jbar = thunder.functional.jit(bar, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jbar(a)


def test_intermediate_from_partial_sharp_edge():
    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    def foo(a, b):
        return a + b

    pfoo = partial(foo, b=b)

    def bar(a):
        return pfoo(a)

    jbar = thunder.functional.jit(bar, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jbar(a)


#
# Sharp edges --- modifying inputs
#


@pytest.mark.xfail(reason="Fails with an AssertionError, not the correct sharp edges warning")
def test_modifying_input_list_sharp_edge():
    def foo(lst):
        list.append(lst, 5)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")
    lst = [1, 2, 3]

    with pytest.raises(ThunderSharpEdgeError):
        jfoo(lst)


#
# Sharp edges -- Side effects
#


def test_calling_open_sharp_edge():
    def foo():
        open("nonexistent file")

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_print_sharp_edge():
    def foo(a):
        print(a)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    a = torch.randn((2, 2))

    with pytest.raises(ThunderSharpEdgeError):
        jfoo(a)


#
# Sharp edges -- Random module (surprising inputs and side effects)
#


def test_calling_random_seed_sharp_edge():
    def foo():
        random.seed(1234)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    state = random.getstate()
    try:
        with pytest.raises(ThunderSharpEdgeError):
            jfoo()
    finally:
        random.setstate(state)


def test_calling_random_setstate_sharp_edge():
    def foo():
        return random.getstate()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_random_setstate_sharp_edge():
    def foo(state):
        random.setstate(state)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    state = random.getstate()
    with pytest.raises(ThunderSharpEdgeError):
        jfoo(state)


def test_calling_random_randbytes_sharp_edge():
    def foo():
        return random.randbytes(20)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_random_randrange_sharp_edge():
    def foo():
        return random.randrange(10)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_random_randint_sharp_edge():
    def foo():
        return random.randint(0, 10)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_random_getrandbits_sharp_edge():
    def foo():
        return random.getrandbits(10)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_calling_random_choice_sharp_edge():
    def foo():
        l = [1, 3, 5]
        return random.choice(l)

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    with pytest.raises(ThunderSharpEdgeError):
        jfoo()


def test_accessing_random_function():
    # Just accessing shouldn't raise an error.
    def foo():
        a = random.choice if 0 > 1 else 1
        return a

    jfoo = thunder.functional.jit(foo, sharp_edges="error")
    jfoo()


def test_random_functions():
    # Additional random functions
    fn_arg_tuple = (
        (random.choices, ([1, 2])),
        (random.shuffle, ([1, 2])),
        (random.sample, ([1, 2], 1)),
        (random.random, ()),
        (random.uniform, ()),
        (random.triangular, ()),
        (random.betavariate, ()),
        (random.expovariate, ()),
        (random.gammavariate, ()),
        (random.gauss, ()),
        (random.lognormvariate, ()),
        (random.normalvariate, ()),
        (random.vonmisesvariate, ()),
        (random.paretovariate, ()),
        (random.weibullvariate, ()),
        # only in python 3.12
        # (random.binomialvariate, (10, 0.7)),
    )

    for fn, arg in fn_arg_tuple:

        def foo():
            _ = fn(*arg)
            return 1

        jfoo = thunder.functional.jit(foo, sharp_edges="error")
        with pytest.raises(ThunderSharpEdgeError):
            jfoo()


# NOTE This use of randomness is OK (and we could add a Proxy for random.Random)
def test_random_Random_class():
    def foo():
        return random.Random(1234).random()

    jfoo = thunder.functional.jit(foo, sharp_edges="error")

    expected = foo()
    actual = jfoo()

    assert_close(expected, actual)
