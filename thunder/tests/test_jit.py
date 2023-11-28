from functools import partial

import pytest
import torch
from torch.testing import assert_close

from thunder.core.jit import jit, JITError


def test_no_return():
    def foo():
        pass

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_constant_return():
    def foo():
        return 5

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_constant_addition():
    def foo():
        return 3 + 5

    jfoo = jit(foo)
    assert jfoo() == foo()


def test_input_number_addition():
    def foo(a, b):
        return a + 2 + b

    jfoo = jit(foo)

    args = (5, 2)

    assert jfoo(*args) == foo(*args)


def test_input_tensor_addition():
    def foo(a, b):
        return a + 2 + b

    jfoo = jit(foo)

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

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_function_call():
    def bar(a, b):
        return a + b

    def foo(a, b):
        return bar(a + 1, b)

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_inner_function_definition():
    def foo(a, b):
        def bar(a, b):
            return a + b

        return bar(a + 1, b)

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_inner_closure():
    def foo(a, b):
        def bar(a):
            return a + b

        return bar(a + 1)

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


def test_unpack_sequence():
    def foo(tup):
        a, b = tup
        return a + b

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(args)
    python_result = foo(args)

    assert_close(thunder_result, python_result)


def test_exception_traceback():
    def bar(a):
        raise ValueError(f"I don't like {a}")

    def foo(b):
        return bar(b + 1)

    jfoo = jit(foo)

    args = (4,)

    # TODO: change to ValueError once that is supported!
    with pytest.raises(JITError) as excinfo:
        thunder_result = jfoo(*args)
    assert "in foo in file" in str(excinfo.value)
    assert "in bar in file" in str(excinfo.value)


def test_walrus_operator():
    def foo(a, b):
        c = (a := b)
        return c

    jfoo = jit(foo)

    assert jfoo(3, 8) == foo(3, 8)


def test_build_map():
    def foo(a, b):
        return {0: a, 1: b, 2: 3, "a": 4, a: 5}

    jfoo = jit(foo)

    # a, b
    cases = (
        (-3, 9),
        (1, 1),
        (0, 1),
        (2, 5),
    )

    for a, b in cases:
        assert jfoo(a, b) == foo(a, b)


# TODO https://github.com/Lightning-AI/lightning-thunder/issues/1543
def test_kwargs():
    def foo(a, b, *, c=2):
        return a + b + c

    jfoo = jit(foo)

    assert jfoo(2, 3) == foo(2, 3)
    assert jfoo(a=2, b=7, c=3) == foo(a=2, b=7, c=3)

    # Same case as above except c can be specified positionally
    def foo(a, b, c=2):
        return a + b + c

    jfoo = jit(foo)

    assert jfoo(2, 3) == foo(2, 3)
    assert jfoo(a=2, b=7, c=3) == foo(a=2, b=7, c=3)


def test_args_kwargs():
    def bar(a, b):
        return a + b

    def foo(a, **kwargs):
        return bar(a, **kwargs)

    jfoo = jit(foo)
    assert jfoo(2, b=3) == foo(2, b=3)
    assert jfoo(a=2, b=3) == foo(a=2, b=3)


def test_partials():
    def foo(a, b, c):
        return a - b * c

    pfoo = partial(foo, 2, c=3)
    jpfoo = jit(pfoo)

    assert jpfoo(4) == pfoo(4)
    assert jpfoo(-9) == pfoo(-9)

    ppfoo = partial(pfoo, -5)
    jppfoo = jit(ppfoo)

    assert jppfoo() == ppfoo()

    # Tests that keywords "stack" as expected (later partials take precedence)
    pfoo = partial(foo, c=2)
    ppfoo = partial(pfoo, c=-2)
    jppfoo = jit(ppfoo)

    assert jppfoo(7, 9) == ppfoo(7, 9)
    assert jppfoo(7, 9, c=4) == ppfoo(7, 9, c=4)

    # Tests that args "stack" as expected
    pfoo = partial(foo, 7)
    ppfoo = partial(pfoo, 9)
    jppfoo = jit(ppfoo)

    assert jppfoo(5) == ppfoo(5)
    assert jppfoo(-3) == ppfoo(-3)


def test_using_imported_modules():
    import operator

    def foo(a, b):
        return operator.add(a, b)

    jfoo = jit(foo)

    assert jfoo(3, 5) == foo(3, 5)


def test_reduce():
    import functools
    import operator

    def foo(a, b):
        return functools.reduce(operator.add, (a, b))

    jfoo = jit(foo)

    assert jfoo(3, 5) == foo(3, 5)
    assert jfoo(-2, 0) == foo(-2, 0)


def test_calling_methods():
    class mycls:
        def __init__(self, v: int):
            self.v = v

        def my_add(self, b):
            return self.v + b

    x = mycls(5)

    def foo(x, a):
        return x.my_add(a)

    jfoo = jit(foo)

    assert jfoo(x, 7) == foo(x, 7)


def test_callable_classes():
    class mycls:
        def __init__(self, v: int):
            self.v = v

        def __call__(self, b):
            return self.v + b

    x = mycls(5)

    def foo(x, a):
        return x(a)

    jfoo = jit(foo)

    assert jfoo(x, 7) == foo(x, 7)


def test_build_slice():
    def foo(a, b):
        l = [0, 1, 2, 3, 4, 5, 6]
        return l[a:b], l[a:], l[:b], l[1:2:2], l[0:a:b]

    jfoo = jit(foo)

    assert jfoo(1, 4) == foo(1, 4)
    assert jfoo(0, -1) == foo(0, -1)


@pytest.mark.xfail
def test_nanogpt_mlp():
    from thunder.benchmarks import NanoGPTMLPBenchmark

    bench = NanoGPTMLPBenchmark(config="gpt2", device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))
