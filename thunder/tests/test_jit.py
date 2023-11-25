from functools import partial

import pytest
import torch
from torch.testing import assert_close

from thunder.core.jit import jit, pjit, JITError


def test_no_return():
    def foo():
        pass

    jfoo = pjit(foo)
    assert jfoo() == foo()


def test_constant_return():
    def foo():
        return 5

    jfoo = pjit(foo)
    assert jfoo() == foo()


def test_constant_addition():
    def foo():
        return 3 + 5

    jfoo = pjit(foo)
    assert jfoo() == foo()


def test_input_number_addition():
    def foo(a, b):
        return a + 2 + b

    jfoo = pjit(foo)

    args = (5, 2)

    assert jfoo(*args) == foo(*args)


def test_input_tensor_addition():
    def foo(a, b):
        return a + 2 + b

    jfoo = pjit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_constant_if():
    def foo(a, b):
        if 3 < 5:
            return a + b
        else:
            assert False

    jfoo = pjit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_function_call():
    def bar(a, b):
        return a + b

    def foo(a, b):
        return bar(a + 1, b)

    jfoo = pjit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_inner_function_definition():
    def foo(a, b):
        def bar(a, b):
            return a + b

        return bar(a + 1, b)

    jfoo = pjit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_inner_closure():
    def foo(a, b):
        def bar(a):
            return a + b

        return bar(a + 1)

    jfoo = pjit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_unpack_sequence():
    def foo(tup):
        a, b = tup
        return a + b

    jfoo = pjit(foo)

    args = (4, 3)

    thunder_result = jfoo(args)
    python_result = foo(args)

    assert_close(thunder_result, python_result)


def test_exception_traceback():
    def bar(a):
        raise ValueError(f"I don't like {a}")

    def foo(b):
        return bar(b + 1)

    jfoo = pjit(foo)

    args = (4,)

    # TODO: change to ValueError once that is supported!
    with pytest.raises(JITError) as excinfo:
        thunder_result = jfoo(*args)
    assert "in foo in file" in str(excinfo.value)
    assert "in bar in file" in str(excinfo.value)


# test kwargs
# test compile + exec
# test random.randint
# test random.seed, random.getstate, random.setstate
# test isinstance
