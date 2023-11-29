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


def test_if():
    def foo(a, b):
        if a < b:
            return a
        elif b > a:
            return b
        else:
            return 0

    jfoo = jit(foo)

    cases = (
        (5, 3),
        (9, 12),
        (2, 2),
    )

    for case in cases:
        assert jfoo(*case) == foo(*case)


def test_dunder_bool():
    class mycls:
        def __init__(self, value):
            self.value = value

        # True if self.value is even
        def __bool__(self):
            return (self.value % 2) == 0

    def foo(a):
        if a:
            return 1
        return -1

    jfoo = jit(foo)

    cases = (
        (mycls(4),),
        (mycls(5),),
    )

    for case in cases:
        assert jfoo(*case) == foo(*case)


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
    # NOTE The addition of closing over value also tests
    #   the STORE_DEREF opcode
    def foo(a, b):
        value = 5

        def bar(a):
            return a + b + value

        return bar(a + 1)

    jfoo = jit(foo)

    args = (4, 3)

    thunder_result = jfoo(*args)
    python_result = foo(*args)

    assert_close(thunder_result, python_result)


@pytest.mark.xfail(reason="Waits for do_raise to work")
def test_delete_deref():
    def foo(a, b):
        value = 5

        def bar(a):
            nonlocal value
            del value
            return a + b + value

        return bar(a + 1)

    jfoo = jit(foo)

    args = (4, 3)

    with pytest.raises(NameError, match="'value'"):
        python_result = foo(*args)
    with pytest.raises(NameError, match="'value'"):
        thunder_result = jfoo(*args)

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

        @classmethod
        def my_add_class(cls, b):
            o = cls(2)
            return o.v + b

        @staticmethod
        def my_add_static(b):
            return 3 + b

    x = mycls(5)

    # these use LOAD_METHOD / CALL_METHOD
    def foo(x, a):
        return x.my_add(a)

    def foo_class(x, a):
        return x.my_add_class(a)

    def foo_static(x, a):
        return x.my_add_static(a)

    # these use LOAD_ATTR / CALL_FUNCTION
    def bar(x, a):
        meth = x.my_add(a)
        return meth

    def bar_class(x, a):
        meth = x.my_add_class(a)
        return meth

    def bar_static(x, a):
        meth = x.my_add_static(a)
        return meth

    jfoo = jit(foo)
    jfoo_class = jit(foo_class)
    jfoo_static = jit(foo_static)
    jbar = jit(bar)
    jbar_class = jit(bar_class)
    jbar_static = jit(bar_static)

    assert jfoo(x, 7) == foo(x, 7)
    assert jfoo_class(x, 7) == foo_class(x, 7)
    assert jfoo_static(x, 7) == foo_static(x, 7)
    assert jbar(x, 7) == bar(x, 7)
    assert jbar_class(x, 7) == bar_class(x, 7)
    assert jbar_static(x, 7) == bar_static(x, 7)


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


def test_format_value():
    # Tests FVS_HAVE_SPEC and FVC_NONE
    def foo(a, b):
        return f"{a:3.2f}, {b:2.1f}"

    jfoo = jit(foo)

    assert jfoo(2.34, 123234.79289) == foo(2.34, 123234.79289)

    class mycls:
        def __repr__(self):
            return "repr"

        def __str__(self):
            return "str"

    # Tests FVC_NONE
    def foo(a, b):
        return f"{a}, {b}"

    jfoo = jit(foo)

    x = mycls()
    assert jfoo(x, "goodbye") == foo(x, "goodbye")

    # Tests FVC_STR
    def foo(a):
        return f"{a!s}"

    jfoo = jit(foo)

    assert jfoo(x) == foo(x)

    # Tests FVC_REPR
    def foo(a):
        return f"{a!r}"

    jfoo = jit(foo)

    assert jfoo(x) == foo(x)

    # Tests FVC_ASCII
    def foo(a):
        return f"{a!a}"

    jfoo = jit(foo)

    assert jfoo(x) == foo(x)


def test_import():
    def foo(a, b):
        import operator

        return operator.add(a, b)

    jfoo = jit(foo)

    assert jfoo(-1, 3) == foo(-1, 3)

    def foo(a, b):
        from operator import add

        return add(a, b)

    jfoo = jit(foo)

    assert jfoo(2, 7) == foo(2, 7)

    def foo(a):
        import torch.nn as nn
        from torch.nn.functional import relu

        return relu(a)

    jfoo = jit(foo)

    a = torch.randn((2, 2))

    assert_close(jfoo(a), foo(a))


def test_nanogpt_mlp():
    from thunder.benchmarks import NanoGPTMLPBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTMLPBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


@pytest.mark.xfail
def test_nanogpt_csa():
    from thunder.benchmarks import NanoGPTCSABenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTCSABenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


@pytest.mark.xfail
def test_nanogpt_block():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))


@pytest.mark.xfail
def test_nanogpt():
    from thunder.benchmarks import NanoGPTBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))
