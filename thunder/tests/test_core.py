import operator
import os
import tempfile
from functools import partial, reduce
import dataclasses
import re
import weakref

import pytest
import torch
from torch.testing import assert_close, make_tensor
from lightning_utilities import compare_version

import thunder
from thunder import cache_option, cache_hits, cache_misses
import thunder.core.proxies
import thunder.examine as examine
import thunder.clang as clang
import thunder.core.profile
import thunder.tests.bf16
import thunder.torch as ltorch

import thunder.core.codeutils as codeutils
from thunder.tests.framework import (
    instantiate,
    NOTHING,
    TorchExecutor,
    nvFuserExecutor,
    requiresCUDA,
    TestExecutor,
    set_default_dtype_ctx,
)
import thunder.core.dtypes as dtypes
import thunder.core.prims as prims
from thunder.core.trace import TraceCtx, set_tracectx, reset_tracectx, tracectx
from thunder.core.symbol import BoundSymbol


#
# Tests related to the test framework itself
#
# TODO Move these into a new test_testing.py?

# Using instantiate with parametrize

_parametrize_decorator = pytest.mark.parametrize("num, s", ((2, "hi"), (-5, "bye")))


@thunder._with_cache_info_ctx
def run_prologue(jfn, *args, **kwargs):
    cd = thunder.compile_data(jfn)
    cs = thunder.compile_stats(jfn)
    ci = thunder._get_cache_info()
    cd.populate_cache_info(ci, *args, **kwargs)
    traces = cd.acquire_initial_trace(cd.fn, args, kwargs, cd, cs, cd.executors_list[0])
    cache_entry = cd.apply_transforms_and_build_cache_entry(cd, cs, ci, *traces)
    with thunder.compile_data_and_stats(cd, cs):
        pro_to_comp, pro_to_epi = cache_entry.prologue_fn(*args, **kwargs)
    return cache_entry, pro_to_comp, pro_to_epi


@instantiate(dtypes=NOTHING, decorators=(_parametrize_decorator,))
def test_instantiate_and_pytest_parametrize(executor, device: str, dtype: dtypes.dtype, num: int, s: str):
    assert isinstance(num, int)
    assert isinstance(s, str)

    assert num == 2 or num == -5
    assert s == "hi" or s == "bye"


#
# Tests related to running valid Python programs
#


@instantiate(dtypes=NOTHING)
def test_make_callable_from_trace(executor, device: str, dtype: dtypes.dtype):
    def foo(a, b):
        return a + b

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    traced_foo = thunder.trace(inline_trace=False)(foo, a, b)
    assert len(traced_foo.bound_symbols) == 4
    assert traced_foo.bound_symbols[-2].sym.name == "add"

    # Ensure that the trace can be fused and executed
    fusion = executor.make_callable(traced_foo)
    actual = fusion(a, b)
    torch.testing.assert_close(actual, a + b)


# Tests that traces don't generate duplicate names
#   (at least not within the first 10k names tested below)
def test_name_generation():
    # NOTE This function is just because trace's currently require a function to
    #   construct them
    def foo():
        pass

    trace = TraceCtx(foo)

    names = set()
    for ctr in range(10000):
        name = trace._gen_name(ctr)
        assert name not in names, f"Found duplicate name {name}"

        names.add(name)


@instantiate(dtypes=(thunder.float32,))
def test_integer_isinstance_mimicry(executor, device: str, dtype: dtypes.dtype):
    # isinstance() works as expected
    def foo(a, b, c):
        if isinstance(a, int):
            return clang.add(a, b)

        return clang.add(b, c)

    traced_foo = executor.make_callable(foo)

    tdtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 1), device=device, dtype=tdtype)
    b = make_tensor((2, 2), device=device, dtype=tdtype)
    c = make_tensor((1, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b, c)
    torch_result = b + c
    assert_close(thunder_result, torch_result)

    thunder_result = traced_foo(2, b, c)
    torch_result = 2 + b
    assert_close(thunder_result, torch_result)

    # type() doesn't work (it returns the actual type)
    def bar(a, b, c):
        if type(a) is int:
            return clang.add(a, b)

        return clang.add(b, c)

    traced_bar = executor.make_callable(bar)

    try:
        thunder_result = traced_bar(a, b, c)
        torch_result = b + c
        assert_close(thunder_result, torch_result)
        pytest.fail()
    except BaseException:
        pass

    try:
        thunder_result = traced_bar(2, b, c)
        torch_result = 2 + b
        assert_close(thunder_result, torch_result)
        pytest.fail()
    except BaseException:
        pass


# TODO Subsume this by test_elementwise when sample inputs are expanded to include more numbers
@instantiate(dtypes=NOTHING)
def test_integer_return(executor, device, _):
    if executor == nvFuserExecutor:
        pytest.xfail("nvFuser does not support only scalar outputs")

    def foo(a, b):
        return clang.add(a, b)

    traced_foo = executor.make_callable(foo)

    thunder_result = traced_foo(3, 4)
    python_result = 3 + 4
    assert_close(thunder_result, python_result)


@instantiate(dtypes=(thunder.float32,))
def test_crazy_collections_in_and_out(executor, device, dtype):
    def foo(a, b, c, *, ka, kb, kc):
        d = {
            5: 2,
            7: 9,
            "a": [a, b],
            "b": {"a": a, "b": b, "c": [b, (a, c)]},
            "x": (a, [a, a, a], (b, (a, a, c, b))),
        }

        e = a["a"]["a"] + b[0]
        f = c[1]["c"] + b[1]
        g = e + f
        h = f + ka + kb
        # NOTE The following line is intentionally not returned
        i = ka + ka
        j = kc[0] + kc[1]

        d["j"] = j

        return (
            a,
            (g,),
            (((j,),),),
            g,
            g,
            b,
            e,
            d["j"],
            (f, d, c, (d,), c, {"a": a, 5: f, "b": h}),
            (5,),
            (),
            (a,),
            [5, a, (b,), (), {}],
            {},
        )

    traced_foo = executor.make_callable(foo)
    tdtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)
    c = make_tensor((2, 2), device=device, dtype=tdtype)

    args = ({"a": {"a": a}}, (b, c), (3, {"c": c}))
    kwargs = {"ka": b, "kb": 3.0, "kc": (a, 2)}
    thunder_result = traced_foo(*args, **kwargs)
    torch_result = foo(*args, **kwargs)

    assert_close(thunder_result, torch_result)


@instantiate(dtypes=(thunder.float32,))
def test_nested_empty_tuple_unpack(executor, device, dtype):
    def foo(a):
        pass

    cfoo = executor.make_callable(foo)
    torch_dtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2, 2), device=device, dtype=torch_dtype)

    inp = {
        0: (
            (
                a,
                a,
            ),
            [a, (a, a), {}],
            {},
            (),
        )
    }

    cfoo(inp)


@instantiate(dtypes=(thunder.float32,))
def test_grad_unpack(executor, device, dtype):
    tdtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=tdtype, requires_grad=True)
    a.grad = make_tensor((2, 2), device=device, dtype=tdtype)

    def get_grad_times_two(a):
        return a.grad * 2

    jitted_get_grad_times_two = executor.make_callable(get_grad_times_two)
    actual = jitted_get_grad_times_two(a)
    expected = a.grad * 2
    assert_close(actual, expected)

    # a.grad is unpacked before a
    def grad_first(a):
        return a.grad + a

    # a.grad is unpacked after a
    def grad_second(a):
        return a + a.grad

    jitted_grad_first = executor.make_callable(grad_first)
    jitted_grad_second = executor.make_callable(grad_second)
    actual_first = jitted_grad_first(a)
    actual_second = jitted_grad_second(a)
    expected = a + a.grad
    assert_close(actual_first, expected)
    assert_close(actual_second, expected)


@instantiate(dtypes=(thunder.float32,))
def test_grad_no_recompile(executor, device, dtype):
    # Checks that having .grad or not does not cause recompile

    def foo(a):
        return a * 2

    cfoo = executor.make_callable(foo)

    tdtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=tdtype, requires_grad=True)
    a.grad = make_tensor((2, 2), device=device, dtype=tdtype)
    cfoo(a)
    assert thunder.cache_misses(cfoo) == 1

    a.grad = None
    cfoo(a)
    assert thunder.cache_misses(cfoo) == 1

    b = make_tensor((3, 3), device=device, dtype=tdtype, requires_grad=True)
    cfoo(b)
    assert thunder.cache_misses(cfoo) == 2

    b.grad = make_tensor((3, 3), device=device, dtype=tdtype)
    cfoo(b)
    assert thunder.cache_misses(cfoo) == 2


@instantiate(dtypes=(thunder.float32,))
def test_grad_recompile(executor, device, dtype):
    # Checks that change in the metadata of a.grad causes recompile

    def foo(a):
        return a.grad * 2

    cfoo = executor.make_callable(foo)

    tdtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=tdtype, requires_grad=True)
    a.grad = make_tensor((2, 2), device=device, dtype=tdtype)
    cfoo(a)
    assert thunder.cache_misses(cfoo) == 1

    b = make_tensor((3, 3), device=device, dtype=tdtype, requires_grad=True)
    b.grad = make_tensor((3, 3), device=device, dtype=tdtype)
    cfoo(b)
    assert thunder.cache_misses(cfoo) == 2


@instantiate(dtypes=(thunder.float32,))
def test_optimizer_unpack(executor, device, dtype):
    class Optimizer(torch.optim.Optimizer):
        def __init__(self, params):
            self.param_groups = [{"params": params}]

        def _init_group(self, group, params, grads):
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)

        @torch.no_grad
        def step(self):
            for group in self.param_groups:
                params = []
                grads = []
                self._init_group(group, params, grads)
                for param, grad in zip(params, grads):
                    param -= 0.1 * grad

    tdtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2, 2), device=device, dtype=tdtype, requires_grad=True)
    ref_a = a.detach().clone()
    a.grad = make_tensor((2, 2), device=device, dtype=tdtype)

    b = make_tensor((2, 2), device=device, dtype=tdtype)
    ref_b = b.detach().clone()

    optimizer = Optimizer([a, b])
    cstep = executor.make_callable(optimizer.step)
    cstep()

    expected_a = ref_a - 0.1 * a.grad
    assert_close(a, expected_a)
    assert_close(b, ref_b)


@instantiate(dtypes=(thunder.float32,))
def test_varargs(executor, device, dtype):
    def foo(*args):
        return reduce(operator.add, args)

    traced_foo = executor.make_callable(foo)
    tdtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    packed = (a, a, a, a, a)

    thunder_result = traced_foo(*packed)
    torch_result = foo(*packed)

    assert_close(thunder_result, torch_result)


@instantiate(dtypes=(thunder.float32,))
def test_kwargs(executor, device, dtype):
    def foo(**kwargs):
        return kwargs["a"] + kwargs["b"]

    traced_foo = executor.make_callable(foo)
    tdtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2,), device=device, dtype=tdtype)

    thunder_result = traced_foo(a=a, b=b)
    torch_result = foo(a=a, b=b)

    assert_close(thunder_result, torch_result)


@instantiate(dtypes=(thunder.float32,))
def test_varargs_and_kwargs(executor, device, dtype):
    def foo(a, b, *posargs, e, **kwargs):
        accum = a
        for x in posargs:
            accum = a + x

        d = b + e + kwargs["f"]

        return accum, d, kwargs["g"]

    traced_foo = executor.make_callable(foo)
    tdtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)
    c = make_tensor((2, 2), device=device, dtype=tdtype)
    d = make_tensor((2,), device=device, dtype=tdtype)
    e = make_tensor((2,), device=device, dtype=tdtype)
    f = make_tensor((2,), device=device, dtype=tdtype)
    g = make_tensor((2,), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b, c, d, e=e, f=f, g=g)
    torch_result = foo(a, b, c, d, e=e, f=f, g=g)

    assert_close(thunder_result, torch_result)


@instantiate(dtypes=(thunder.float32,))
def test_no_return(executor, device, dtype):
    def foo(a, b):
        c = a + b
        pass

    traced_foo = executor.make_callable(foo)
    tdtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b=b)
    torch_result = foo(a, b)

    assert_close(thunder_result, torch_result)


@instantiate(dtypes=NOTHING)
def test_no_input(executor, device, dtype):
    def foo():
        return 3, ()

    traced_foo = executor.make_callable(foo)

    thunder_result = traced_foo()
    torch_result = foo()

    assert_close(thunder_result, torch_result)


@instantiate(dtypes=(thunder.float32,))
def test_no_compute(executor, device, dtype):
    def foo(a, b):
        return a, 3.0

    traced_foo = executor.make_callable(foo)
    tdtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b=b)
    torch_result = foo(a, b)

    assert_close(thunder_result, torch_result)


@instantiate(dtypes=(thunder.float32,))
def test_strings_in_and_out(executor, device, dtype):
    def foo(a, b, c="ok"):
        return a, b, "hello"

    cfoo = executor.make_callable(foo)

    lc_result = cfoo("a", b="b")
    assert lc_result == ("a", "b", "hello")


@instantiate(dtypes=(thunder.float32,))
def test_objects_in_and_out(executor, device, dtype):
    a = object()
    b = object()
    c = object()

    def foo(a, b, c=c):
        return a, b, object()

    cfoo = executor.make_callable(foo)

    lc_result = cfoo(a, b=b)
    a, b, c = lc_result

    assert type(a) is object
    assert type(b) is object
    assert type(c) is object


@instantiate(dtypes=(thunder.float32,))
def test_devices_in_and_out(executor, device, dtype):
    dev = thunder.devices.Device(device)

    def foo(a, dev=dev):
        return a, dev

    cfoo = executor.make_callable(foo)

    lc_result = cfoo(1, dev)

    x, y = lc_result

    assert x == 1
    assert y == thunder.devices.to_torch_device(dev)


@instantiate(dtypes=(thunder.float32,))
def test_partial(executor, device, dtype):
    def foo(a, *, b, c=2):
        return a, b, c

    pfoo = partial(foo, b=3, c=4)
    cpfoo = executor.make_callable(pfoo)

    lc_result = cpfoo(1)
    py_result = pfoo(1)

    assert_close(lc_result, py_result)

    # Tests that later partials override earlier partials correctly
    ppfoo = partial(pfoo, b=2, c=8)
    cppfoo = executor.make_callable(ppfoo)

    lc_result = cppfoo(1)
    py_result = ppfoo(1)

    assert_close(lc_result, py_result)


# Tests that partials that specify default args are not supported (yet)
@instantiate(dtypes=(thunder.float32,))
def test_partial_args(executor, device, dtype):
    def foo(a, b):
        return a + b

    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    pfoo = partial(foo, a)
    jpfoo = executor.make_callable(pfoo)

    res = jpfoo(b)
    expected = pfoo(b)

    assert_close(res, expected)


@instantiate(dtypes=(thunder.float32,))
def test_constant_creation(executor, device, dtype):
    def py_foo(a):
        return a + 1.0

    def foo(a):
        x = prims.convert_element_type(1, float)
        return a + x

    cfoo = thunder.jit(foo, executors=executor.executors_list())

    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)

    lc_result = cfoo(a)
    python_result = py_foo(a)

    assert_close(lc_result, python_result)


#
# Tests related to printing signatures and bound symbols
#


def test_siginfo_printing():
    def foo(a=object(), b=torch.float32, *, c=(object(), object())):
        return a, b, c[0], a, c

    siginfo = codeutils.get_siginfo(foo, (), {})

    s0 = siginfo.prettyprint()
    s1 = siginfo.prettyprint()

    assert s0 == s1

    trace = TraceCtx(foo)
    with tracectx(trace):
        s0 = trace.python()
        s1 = trace.python()

        assert s0 == s1


def test_consistent_trace_and_boundsymbol_printing():
    def foo(a=object(), b=(torch.float32, object())):
        return a, b, b[1]

    cfoo = thunder.jit(foo)
    _ = cfoo()
    traces = thunder.last_traces(cfoo)

    # Tests consistent printing of traces
    s0 = str(traces[0])
    s1 = str(traces[0])

    assert s0 == s1

    # Tests consistent printing of bound symbols outside the trace context
    for bsym in traces[0].bound_symbols:
        s0 = str(bsym)
        s1 = str(bsym)
        assert s0 == s1


def test_consistent_boundsymbol_collection_printing():
    def foo(tup, d):
        (a, b), c = tup
        e = c + d["dict"]["val"]
        return a + b, e

    cfoo = thunder.jit(foo)
    _ = cfoo(((2, 3), 4), {"dict": {"val": 2}})
    traces = thunder.last_traces(cfoo)

    # Tests consistent printing of bound symbols outside the trace context
    for bsym in traces[0].bound_symbols:
        s0 = str(bsym)
        s1 = str(bsym)
        assert s0 == s1


def test_consistent_boundsymbol_collection_hard_printing():
    def foo(tup):
        (a, b), c = tup
        d = b["dict"]["val"]
        return a + d, c

    cfoo = thunder.jit(foo)
    _ = cfoo(((2, {"dict": {"val": 2}}), 4))
    traces = thunder.last_traces(cfoo)

    # Tests consistent printing of bound symbols outside the trace context
    for bsym in traces[0].bound_symbols:
        s0 = str(bsym)
        s1 = str(bsym)
        assert s0 == s1


def test_to_printable_not_collection():
    import numpy as np

    inps = ("abc", torch.Size([2, 2]), torch.Tensor(1, 2), np.ndarray((2, 2)))
    for inp in inps:
        out = codeutils.to_printable(None, inp)
        assert inp is out


def test_to_printable_collection():
    from collections import namedtuple

    MyTuple = namedtuple("MyTuple", ["x", "y"])

    inps = (MyTuple("abc", "def"),)
    for inp in inps:
        out = codeutils.to_printable(None, inp)
        assert inp == out


#
# Type promotion tests
#
# TODO Maybe move to test_type_promotion.py?


# TODO This test just spot-checks type promotion -- it could probably be better
@instantiate(dtypes=NOTHING)
def test_type_promotion_tensors(executor, device, _):
    if executor == TorchExecutor:
        pytest.xfail('see issue "vmap of sum doesn\'t work when dims are passed as a keyword argument"')

    def foo(a, b):
        return a + b

    traced_foo = executor.make_callable(foo)

    b1 = make_tensor((2, 2), device=device, dtype=torch.bool)
    i64 = make_tensor((2, 2), device=device, dtype=torch.int64)
    bf16 = make_tensor((2, 2), device=device, dtype=torch.bfloat16)
    f16 = make_tensor((2, 2), device=device, dtype=torch.float16)
    f32 = make_tensor((2, 2), device=device, dtype=torch.float32)

    # float16 x float16 type promotion -- float16 result dtype
    result = traced_foo(f16, f16)
    assert result.dtype is torch.float16

    # float16 x float32 type promotion -- float32 result dtype
    result = traced_foo(f16, f32)
    assert result.dtype is torch.float32

    # float16 x bfloat16 type promotion -- float32 result dtype
    if thunder.tests.bf16.device_supports_bf16(device):
        result = traced_foo(f16, bf16)
        assert result.dtype is torch.float32

    # int64 x float16 type promotion -- float16 result dtype
    result = traced_foo(f16, i64)
    assert result.dtype is torch.float16

    # bool x int64 type promotion -- int64 result dtype
    result = traced_foo(b1, i64)
    assert result.dtype is torch.int64

    # f x int64 type promotion -- float result dtype
    result = traced_foo(2.0, i64)
    assert result.dtype is torch.float32

    # b1 x int64 type promotion -- int64 result dtype
    result = traced_foo(b1, i64)
    assert result.dtype is torch.int64

    def bar(a, b, c):
        return a - b + c

    traced_bar = executor.make_callable(bar)

    # float x int64 x float16 type promotion -- float16 result dtype
    result = traced_bar(2.0, i64, f16)
    assert result.dtype is torch.float16

    # float x int x int64 -- float32 result dtype
    result = traced_bar(2.1, -1, i64)
    assert result.dtype is torch.float32


@instantiate(dtypes=NOTHING)
def test_type_promotion_numbers_and_tensors(executor, device, _):
    if executor == TorchExecutor:
        pytest.xfail('See issue "Type promotion with the torchexecutor and elementwise operations is incorrect"')

    def foo(a, b, c):
        return a + b + c

    cfoo = executor.make_callable(foo)

    f16 = make_tensor((2, 2), device=device, dtype=torch.float16)
    f32 = make_tensor((2, 2), device=device, dtype=torch.float32)
    i64 = make_tensor((2, 2), device=device, dtype=torch.int64)

    result = cfoo(5, f32, 2)
    assert result.dtype is torch.float32

    result = cfoo(f32, 1, f32)
    assert result.dtype is torch.float32

    result = cfoo(i64, 3.0, f16)
    assert result.dtype is torch.float16

    result = cfoo(i64, 3.0, i64)
    assert result.dtype is torch.float32


@instantiate(dtypes=NOTHING)
def test_int_to_float_type_promotion(executor, device, _):
    def foo(a, b):
        return a / b

    cfoo = executor.make_callable(foo)

    i64 = make_tensor((2, 2), device=device, dtype=torch.int64)
    f16 = make_tensor((2, 2), device=device, dtype=torch.float16)

    # int64 x int64 -- float32 result dtype
    result = cfoo(i64, i64)
    assert result.dtype is torch.float32

    # int x int64 -- float32 result dtype
    result = cfoo(2, i64)
    assert result.dtype is torch.float32

    # int64 x bool -- float32 result dtype
    result = cfoo(i64, True)
    assert result.dtype is torch.float32

    # int64 x float16 -- float16 result dtype
    result = cfoo(i64, f16)
    assert result.dtype is torch.float16


#
# Caching tests
#


@instantiate(dtypes=(thunder.float32,))
def test_static_caching(executor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)
    c = make_tensor((2, 2), device=device, dtype=torch_dtype)
    d = make_tensor((2, 1), device=device, dtype=torch_dtype)
    e = make_tensor((2, 2), device=device, dtype=torch.bool)

    def foo(a, b):
        return a + b

    cfoo = thunder.jit(foo, cache_mode="constant values")

    assert cache_option(cfoo) == thunder.CACHE_OPTIONS.CONSTANT_VALUES

    # Tensor x tensor
    result = cfoo(a, b)
    assert cache_misses(cfoo) == 1
    assert cache_hits(cfoo) == 0
    assert_close(result, a + b)

    # Same tensors -- cache hit
    result = cfoo(a, b)
    assert cache_misses(cfoo) == 1
    assert cache_hits(cfoo) == 1
    assert_close(result, a + b)

    # Different tensor, same metadata -- cache hit
    result = cfoo(a, c)
    assert cache_misses(cfoo) == 1
    assert cache_hits(cfoo) == 2
    assert_close(result, a + c)

    # Different tensor, different shape -- cache miss
    result = cfoo(a, d)
    assert cache_misses(cfoo) == 2
    assert cache_hits(cfoo) == 2
    assert_close(result, a + d)

    # Different tensor, different dtype -- cache miss
    result = cfoo(a, e)
    assert cache_misses(cfoo) == 3
    assert cache_hits(cfoo) == 2
    assert_close(result, a + e)

    # Tensor x float number -- cache miss
    result = cfoo(a, 1.0)
    assert cache_misses(cfoo) == 4
    assert cache_hits(cfoo) == 2
    assert_close(result, a + 1.0)

    # Tensor x float number, different tensor data -- cache hit
    result = cfoo(b, 1.0)
    assert cache_misses(cfoo) == 4
    assert cache_hits(cfoo) == 3
    assert_close(result, b + 1.0)

    # Tensor x float number, different number value -- cache miss
    result = cfoo(b, 3.0)
    assert cache_misses(cfoo) == 5
    assert cache_hits(cfoo) == 3
    assert_close(result, b + 3.0)

    # Tensor x int number, different number type -- cache miss
    result = cfoo(b, 3)
    assert cache_misses(cfoo) == 6
    assert cache_hits(cfoo) == 3
    assert_close(result, b + 3)

    # Tensor x int number -- cache hit
    result = cfoo(b, 3)
    assert cache_misses(cfoo) == 6
    assert cache_hits(cfoo) == 4
    assert_close(result, b + 3)

    def bar(a, b):
        return a, b

    cbar = thunder.jit(bar, cache_mode="constant values")

    astr = "a"
    bstr = "b"

    # String x string -- cache miss
    cbar(astr, bstr)
    assert cache_misses(cbar) == 1
    assert cache_hits(cbar) == 0

    # Same strings -- cache hit
    cbar(astr, bstr)
    assert cache_misses(cbar) == 1
    assert cache_hits(cbar) == 1

    # Same string values -- different strings
    bother_str = "b"
    cbar(astr, bother_str)
    assert cache_misses(cbar) == 1
    assert cache_hits(cbar) == 2

    # Object x string -- cache miss
    cbar(object(), bother_str)
    assert cache_misses(cbar) == 2
    assert cache_hits(cbar) == 2

    # TODO: test objects in prologues
    # object() != object() -- cache miss
    # cbar(object(), bother_str)
    # assert cache_misses(cbar) == 3
    # assert cache_hits(cbar) == 2

    # Module tests
    m = torch.nn.Linear(5, 5, device=device, dtype=torch_dtype)
    cm = thunder.jit(m, cache_mode="constant values")

    inp = make_tensor((5, 5), device=device, dtype=torch_dtype)

    result = cm(inp)
    torch_result = m(inp)

    assert_close(result, torch_result)

    assert cache_misses(cm) == 1
    assert cache_hits(cm) == 0

    # Same input -- cache hit

    result = cm(inp)
    torch_result = m(inp)

    assert_close(result, torch_result)

    assert cache_misses(cm) == 1
    assert cache_hits(cm) == 1

    # Different input, same metadata -- cache hit
    inp = make_tensor((5, 5), device=device, dtype=torch_dtype)
    result = cm(inp)
    torch_result = m(inp)

    assert_close(result, torch_result)

    assert cache_misses(cm) == 1
    assert cache_hits(cm) == 2

    # Different input, different metadata -- cache miss
    inp = make_tensor((6, 5), device=device, dtype=torch_dtype)
    result = cm(inp)
    torch_result = m(inp)

    assert_close(result, torch_result)

    assert cache_misses(cm) == 2
    assert cache_hits(cm) == 2

    #
    # Sequence tests
    #
    def caz(tup):
        accum = 0
        for x in tup:
            accum += x
        return accum

    ccaz = thunder.jit(caz, cache_mode="constant values")

    inp0 = [5, 3, 7]
    thunder_result = ccaz(inp0)
    torch_result = caz(inp0)

    assert_close(thunder_result, torch_result)

    assert cache_misses(ccaz) == 1
    assert cache_hits(ccaz) == 0

    # List with different values -- cache miss
    inp1 = [6, 3, 7]
    thunder_result = ccaz(inp1)
    torch_result = caz(inp1)

    assert_close(thunder_result, torch_result)

    assert cache_misses(ccaz) == 2
    assert cache_hits(ccaz) == 0

    # List with same values -- cache hit
    inp2 = [5, 3, 7]
    thunder_result = ccaz(inp2)
    torch_result = caz(inp2)

    assert_close(thunder_result, torch_result)

    assert cache_misses(ccaz) == 2
    assert cache_hits(ccaz) == 1

    # List with same values but different types -- cache miss
    inp3 = [5.0, 3, 7]
    thunder_result = ccaz(inp3)
    torch_result = caz(inp3)

    assert_close(thunder_result, torch_result)

    assert cache_misses(ccaz) == 3
    assert cache_hits(ccaz) == 1

    #
    # Kwarg tests
    #
    def daz(*, a, b):
        return a + b

    cdaz = thunder.jit(daz, cache_mode="constant values")

    inp0 = {"a": a, "b": b}
    thunder_result = cdaz(**inp0)
    torch_result = daz(**inp0)

    assert_close(thunder_result, torch_result)

    assert cache_misses(cdaz) == 1
    assert cache_hits(cdaz) == 0

    # Same keys and tensor metadata the same -- cache hit
    inp1 = {"a": b, "b": a}
    thunder_result = cdaz(**inp1)
    torch_result = daz(**inp1)

    assert_close(thunder_result, torch_result)

    assert cache_misses(cdaz) == 1
    assert cache_hits(cdaz) == 1

    # Same keys but different tensor metadata -- cache hit
    inp2 = {"a": b, "b": e}
    thunder_result = cdaz(**inp2)
    torch_result = daz(**inp2)

    assert_close(thunder_result, torch_result)

    assert cache_misses(cdaz) == 2
    assert cache_hits(cdaz) == 1


#
# Tests related to trace manipulation and transformation
#
# TODO Maybe move to test_transforms.py?


@instantiate(dtypes=(thunder.float32,))
def test_bsym_toposort(executor: TestExecutor, device: str, dtype: dtypes.dtype):
    tdtype: torch.dtype = ltorch.to_torch_dtype(dtype)
    make = partial(make_tensor, device=device, dtype=tdtype, requires_grad=False)

    a = make((2, 2))
    b = make((2, 2))

    def foo(a, b):
        return a + b, a - b

    cfoo = executor.make_callable(foo)
    _, _ = cfoo(a, b)
    traces = thunder.last_traces(cfoo)
    trc = traces[0]

    from thunder.core.transforms import bsym_list_to_dag, TOPOSORT_ORDER, toposort_bsym_dag, Node

    roots, leaves = bsym_list_to_dag(trc.bound_symbols)
    top_down_bsyms = toposort_bsym_dag(roots, TOPOSORT_ORDER.TOP_DOWN)
    bottom_up_bsyms = toposort_bsym_dag(leaves, TOPOSORT_ORDER.BOTTOM_UP)

    top_down_add_bsym = top_down_bsyms[2]
    bottom_up_sub_bsym = bottom_up_bsyms[2]

    assert top_down_add_bsym.sym.id == "torch.add"
    assert bottom_up_sub_bsym.sym.id == "torch.sub"

    def prefer_sub_selector(eligible_nodes: list[Node]) -> int:
        for idx, node in enumerate(eligible_nodes):
            if node.bsym.sym.id == "torch.sub":
                return idx

        return 0

    sub_preferring_bsyms = toposort_bsym_dag(roots, TOPOSORT_ORDER.TOP_DOWN, prefer_sub_selector)
    sub_preferring_sub_bsym = sub_preferring_bsyms[2]

    assert sub_preferring_sub_bsym.sym.id == "torch.sub"

    # Tests collection and reshape with -1 input
    def bar(a, shape):
        b = a + 3
        c = a + 2.0
        return ltorch.reshape(b, shape) + 2, c

    a = make((4, 3, 2, 3))

    cbar = executor.make_callable(bar)
    expected = cbar(a, (12, -1))
    traces = thunder.last_traces(cbar)
    trc = traces[0]

    roots, leaves = bsym_list_to_dag(trc.bound_symbols)
    top_down_bsyms = toposort_bsym_dag(roots, TOPOSORT_ORDER.TOP_DOWN)
    bottom_up_bsyms = toposort_bsym_dag(leaves, TOPOSORT_ORDER.BOTTOM_UP)

    top_down_reshape_bsym = top_down_bsyms[3]
    bottom_up_reshape_bsym = bottom_up_bsyms[2]

    assert top_down_reshape_bsym.sym.id == "torch.reshape"
    assert bottom_up_reshape_bsym.sym.id == "torch.reshape"

    # Tests when the symbol list doesn't contain unpack operator
    bsyms_without_unpack = trc.bound_symbols[1:]
    roots, leaves = bsym_list_to_dag(bsyms_without_unpack)
    assert len(leaves) == 1 and leaves[0].bsym.sym.id == prims.PrimIDs.RETURN


# Verifies that using only some of the results of a function works as expected
#   (the other results are dce'd)
@instantiate(dtypes=(thunder.float32,))
def test_partial_results(executor: TestExecutor, device: str, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)

    def foo(a):
        a, b = torch.var_mean(a)
        return a

    cfoo = executor.make_callable(foo, disable_preprocessing=False)

    result = cfoo(a)
    torch_result = foo(a)

    assert_close(result, torch_result)


def test_visitor_transform():
    device = "cpu"
    dtype = torch.float32
    a = make_tensor((2, 2), device=device, dtype=dtype, requires_grad=True)
    b = make_tensor((2, 2), device=device, dtype=dtype, requires_grad=False)

    def foo(a, b):
        return a + b

    trc = thunder.trace()(foo, a, b)

    from thunder.core.transforms import visitor_transform, VISIT_TYPE

    # Adds a comment before each bound symbols
    def add_comments(bsym) -> VISIT_TYPE:
        prims.comment(f"{bsym.sym.id}")
        return VISIT_TYPE.INSERT_BEFORE

    transformed_trc = visitor_transform(trc, add_comments, provenance="Add comments")

    first_comment = transformed_trc.bound_symbols[0]
    second_comment = transformed_trc.bound_symbols[2]
    third_comment = transformed_trc.bound_symbols[4]
    fourth_comment = transformed_trc.bound_symbols[6]

    assert first_comment.sym.id is prims.PrimIDs.COMMENT
    assert second_comment.sym.id is prims.PrimIDs.COMMENT
    assert third_comment.sym.id is prims.PrimIDs.COMMENT
    assert fourth_comment.sym.id is prims.PrimIDs.COMMENT

    assert first_comment.args[0] == "PrimIDs.UNPACK_TRIVIAL"
    assert second_comment.args[0] == "PrimIDs.UNPACK_TRIVIAL"
    assert third_comment.args[0] == "torch.add"
    assert fourth_comment.args[0] == "PrimIDs.RETURN"

    # Comments result of additions
    def comment_add_results(bsym) -> VISIT_TYPE:
        if bsym.sym.id == "torch.add":
            add_result = bsym.output
            prims.comment(f"add result ndims is {add_result.ndim}")
            return VISIT_TYPE.INSERT_AFTER

        # NOTE In this case either of INSERT_BEFORE or INSERT_AFTER is fine (both just preserve the bsym)
        return VISIT_TYPE.INSERT_BEFORE

    transformed_trc = visitor_transform(trc, comment_add_results, provenance="Comment add results")

    comment = transformed_trc.bound_symbols[3]
    assert comment.sym.id is prims.PrimIDs.COMMENT
    assert comment.args[0] == "add result ndims is 2"


def test_insert_inplace():
    device = "cpu"
    dtype = torch.float32
    a = make_tensor((2, 2), device=device, dtype=dtype, requires_grad=True)
    b = make_tensor((2, 2), device=device, dtype=dtype, requires_grad=False)

    def foo(a, b):
        return a + b

    trc = thunder.trace()(foo, a, b)

    from thunder.core.transforms import insert_inplace

    def add_comments() -> None:
        prims.comment("Unpacking is done")
        prims.comment("About to add some tensors!")

    insert_inplace(trc, 2, add_comments)

    first_comment = trc.bound_symbols[2]
    second_comment = trc.bound_symbols[3]

    assert first_comment.sym.id is prims.PrimIDs.COMMENT
    assert second_comment.sym.id is prims.PrimIDs.COMMENT

    assert first_comment.args[0] == "Unpacking is done"
    assert second_comment.args[0] == "About to add some tensors!"


def test_replace_inplace():
    device = "cpu"
    dtype = torch.float32
    a = make_tensor((2, 2), device=device, dtype=dtype, requires_grad=True)
    b = make_tensor((2, 2), device=device, dtype=dtype, requires_grad=False)

    def foo(a, b):
        return a + b

    trc = thunder.trace()(foo, a, b)

    from thunder.core.transforms import insert_inplace, replace_inplace

    def add_comments() -> None:
        prims.comment("Unpacking is done")
        prims.comment("About to add some tensors!")

    insert_inplace(trc, 2, add_comments)

    def capitalize(bsym: BoundSymbol) -> None:
        assert bsym.sym.id is prims.PrimIDs.COMMENT
        prims.comment("The following comment is uppercase:")
        prims.comment(bsym.args[0].upper())

    replace_inplace(trc, 2, capitalize)

    first_comment = trc.bound_symbols[2]
    uppercase_comment = trc.bound_symbols[3]
    original_comment = trc.bound_symbols[4]

    assert first_comment.sym.id is prims.PrimIDs.COMMENT
    assert uppercase_comment.sym.id is prims.PrimIDs.COMMENT
    assert original_comment.sym.id is prims.PrimIDs.COMMENT

    assert first_comment.args[0] == "The following comment is uppercase:"
    assert uppercase_comment.args[0] == "UNPACKING IS DONE"
    assert original_comment.args[0] == "About to add some tensors!"


@instantiate(dtypes=NOTHING)
def test_detached_trace(executor, device: str, _):
    # This test ensures that the detached_trace context manager works as expected.
    #   It should be possible to enter a detached trace, and then exit it, and
    #   the trace should be restored to its original state.
    from thunder.core.trace import get_tracectx, TraceCtx, detached_trace

    try:
        new_trace = TraceCtx(None)
        trace_token = set_tracectx(new_trace)
        outer_trace = get_tracectx()
        assert outer_trace is not None
        assert outer_trace is trace_token.var.get()
        with detached_trace():
            assert get_tracectx() is not None
            assert get_tracectx() is not outer_trace
    finally:
        reset_tracectx(trace_token)

    # Detached trace should work even if there is no outer trace
    assert get_tracectx() is None
    with detached_trace():
        pass


@instantiate(dtypes=(thunder.float32,))
def test_normalized_args_prims_sum(executor, device: str, dtype: dtypes.dtype):
    # This test verifies that the recorded trace for a call to prims.sum
    # has its positional and keyword arguments normalized to the same form.
    # See issue "vmap of sum doesn't work when dims are passed as a keyword
    # argument"
    a = make_tensor((2, 2), device=device, dtype=ltorch.to_torch_dtype(dtype))

    def func_dim_posarg(x):
        return prims.sum(x, (0, 1))

    def func_dim_kwarg(x):
        return prims.sum(x, dims=(0, 1))

    c1 = executor.make_callable(func_dim_posarg)
    c2 = executor.make_callable(func_dim_kwarg)
    out1 = c1(a)
    out2 = c2(a)
    torch.testing.assert_close(out1, out2)

    trace1 = thunder.last_traces(c1)[0]
    trace2 = thunder.last_traces(c2)[0]
    sum1 = next(s for s in trace1.bound_symbols if s.sym.name == "sum")
    sum2 = next(s for s in trace2.bound_symbols if s.sym.name == "sum")

    assert len(sum1.args) == 2
    assert len(sum2.args) == 2
    assert len(sum1.kwargs) == 0
    assert len(sum2.kwargs) == 0


@instantiate(dtypes=(thunder.float32,))
def test_symbol_all_constant_args(executor, device: str, dtype: dtypes.dtype):
    def foo():
        return clang.add(1, 2)

    trace = thunder.trace()(foo)

    assert len(trace.bound_symbols) == 1
    assert trace.bound_symbols[0].are_all_args_constant

    def bar(a, b):
        return clang.add(a, b)

    trace = thunder.trace()(bar, 1, 2)
    # Trace consists of two trivial unpack and addition
    assert len(trace.bound_symbols) == 4
    symbol = trace.bound_symbols[-2]
    assert symbol.sym.name == "add"
    assert not symbol.are_all_args_constant


@instantiate(dtypes=(thunder.float32,), executors=(TorchExecutor,))
def test_bound_symbol_header(executor, device: str, dtype: dtypes.dtype):
    def foo(x):
        return clang.sin(x)

    a = make_tensor((2, 2), device=device, dtype=ltorch.to_torch_dtype(dtype))
    trace = thunder.trace()(foo, a)

    assert len(trace.bound_symbols) == 3
    sin_symbol = trace.bound_symbols[1]
    assert sin_symbol.sym.name == "sin"

    # Test setting header with a string
    sin_symbol.header = "Testing\nThis symbol's\nHeader"
    assert "# Testing\n# This symbol's\n# Header\nt0 = prims.sin(x)" in str(sin_symbol)
    assert "\n  # Testing\n  # This symbol's\n  # Header\n  t0 = prims.sin(x)" in str(trace)

    # Test setting header with a list of strings
    sin_symbol.header = "Testing\nThis symbol's\nHeader".splitlines()
    assert "# Testing\n# This symbol's\n# Header\nt0 = prims.sin(x)" in str(sin_symbol)
    assert "\n  # Testing\n  # This symbol's\n  # Header\n  t0 = prims.sin(x)" in str(trace)


@instantiate(dtypes=(thunder.float32,), executors=(TorchExecutor,))
def test_bound_symbol_header_context(executor, device: str, dtype: dtypes.dtype):
    from thunder.core.symbol import bsym_header

    def foo(x):
        return clang.sin(x)

    a = make_tensor((2, 2), device=device, dtype=ltorch.to_torch_dtype(dtype))

    header = "Testing\nThis symbol's\nHeader"
    with bsym_header(header):
        trace = thunder.trace()(foo, a)

    assert len(trace.bound_symbols) == 3
    sin_symbol = trace.bound_symbols[1]
    assert sin_symbol.sym.name == "sin"
    assert "# Testing\n# This symbol's\n# Header\nt0 = prims.sin(x)" in str(sin_symbol)
    assert "\n  # Testing\n  # This symbol's\n  # Header\n  t0 = prims.sin(x)" in str(trace)
    # the unbind, the sin and the return all have the header
    assert str(trace).count("Testing") == 3


# Check to verify the issue in "KeyError thrown in thunder.executor.utils.Region
# when None is passed in as input".
@instantiate(dtypes=(thunder.float32,))
def test_argument_of_none(executor, device, dtype):
    from thunder.executors.utils import Region

    def foo(x, y, z):
        return x + y

    tdtype = ltorch.to_torch_dtype(dtype)
    a, b = (make_tensor((1,), device=device, dtype=tdtype) for _ in range(2))
    c = None
    trace = thunder.trace()(foo, a, b, c)

    producers = thunder.core.utils.producers(trace)
    consumers = thunder.core.utils.consumers(trace)
    region_bsyms = trace.bound_symbols[:3]
    region = Region(producers, consumers, region_bsyms)
    assert len(region.inputs) == 0 and sorted(str(v) for v in region.outputs) == [
        '<TensorProxy(name="t0", dtype=thunder.dtypes.float32, shape=(1,))>'
    ]


# This test ensures that calls to torch functions are recorded in the trace
@instantiate(executors=(TorchExecutor,), dtypes=NOTHING)
def test_torch_call_recording(executor, device: str, _):
    def func(a):
        return ltorch.dropout(a)

    a = make_tensor((2, 3), device=device, dtype=torch.float32)

    torch_trace = thunder.trace()(func, a)
    assert len(torch_trace.bound_symbols) == 3
    assert torch_trace.bound_symbols[-2].sym.name == "dropout"
    assert torch_trace.bound_symbols[-2].sym.id == "torch.nn.functional.dropout"

    # Ensure that the trace can be fused and executed
    fusion = executor.make_callable(torch_trace)
    actual = fusion(a)
    assert actual.shape == (2, 3)


# Asserts that all the elements of a collection are equal to each other.
def all_eq(l):
    for e1 in l:
        for e2 in l:
            assert e1 == e2


# Asserts that all the elements of a collection are not equal to each other,
# and that elements are equal to themselves.
def all_neq(l):
    el = enumerate(l)
    for i, e1 in el:
        for j, e2 in el:
            assert e1 == e2 if i == j else e1 != e2


# TODO This test needs to be updated because it no longer compares kwargs vs. positional args
@instantiate(dtypes=(thunder.float32,))
def test_boundsymbol_hash_eq_examples(executor, device, dtype: dtypes.dtype):
    torch_dtype = ltorch.to_torch_dtype(dtype)

    a = make_tensor((2, 2), device=device, dtype=torch_dtype)
    b = make_tensor((2, 2), device=device, dtype=torch_dtype)

    # Returns the bound symbols for a function and args.
    def compile_bsyms(fn, args):
        fn = executor.make_callable(fn)
        _ = fn(*args)
        traces = thunder.last_traces(fn)
        return traces[0].bound_symbols

    # Extracts the bound symbols for the function with
    # the given symbol names.
    def extract_bsyms(fn, args, ops):
        return [b for b in compile_bsyms(fn, args) if b.sym.name in ops]

    # We want .rhs for a * b and torch.mul() to hash and compare
    # the same for writing the CSE pass.
    def mul_rhs(a, b):
        c = a + b
        d = a + b
        e = ltorch.mul(a, b)
        return c, d, e

    bsyms = extract_bsyms(mul_rhs, (a, b), ("mul",))
    all_eq([hash(b.rhs) for b in bsyms])
    all_eq([b.rhs for b in bsyms])

    # The current way BoundSymbols are compared treats args and kwargs the same,
    # so the same semantic call can be considered 'equal' if the arguments are
    # passed differently.
    def mul_rhs_kwargs(a, b):
        c = a * b
        d = ltorch.mul(a, b)
        return c, d

    bsyms = extract_bsyms(mul_rhs_kwargs, (a, b), ("mul",))
    all_eq([hash(b.rhs) for b in bsyms])
    all_eq([b.rhs for b in bsyms])

    # Also make sure the symbols are the same.
    all_eq([b.sym for b in bsyms])
    all_eq([hash(b.sym) for b in bsyms])

    # Assert that rhs of identical operators with same kwargs are equal.
    def same_kwargs(device, dtype):
        a = ltorch.full((2, 2), 5, device=device, dtype=dtype)
        b = ltorch.full((2, 2), 5, device=device, dtype=dtype)
        return a + b

    bsyms = extract_bsyms(same_kwargs, (device, dtype), ("full",))
    all_eq([hash(b.rhs) for b in bsyms])
    all_eq([b.rhs for b in bsyms])

    # The symbols should be the same.
    all_eq([b.sym for b in bsyms])
    all_eq([hash(b.sym) for b in bsyms])

    # Assert that the kwargs are different and hash differently.
    def diff_kwargs(device, dtype):
        a = ltorch.full((1, 2), 2, device=device, dtype=dtype)
        b = ltorch.full((2, 3), 5, device=device, dtype=dtype)
        c = ltorch.full((2, 3), 5, device=device)
        return a, b, c

    bsyms = extract_bsyms(diff_kwargs, (device, dtype), ("full",))
    all_neq([hash(b.rhs) for b in bsyms])
    all_neq([b.rhs for b in bsyms])

    # Assert that boundsymbols for different ops hash/compare differently.
    def different_ops(a, b):
        c = a + b
        d = a - b
        return c, d

    c, d = extract_bsyms(different_ops, (a, b), ("add", "sub"))
    assert hash(c.sym) != hash(d.sym)
    assert hash(c) != hash(d)
    assert hash(c.rhs) != hash(d.rhs)
    assert c.sym != d.sym
    assert c != d
    assert c.rhs != d.rhs


# @instantiate(dtypes=NOTHING)
# @requiresCUDA
# def test_torch_call_lowering_for_nvfuser(executor, device, _):
#     pytest.xfail(reason="lower_for_nvfuser is removed and replaced with 'flattening'")
#     # This test ensures that calls to torch functions are lowered to the
#     # nvFuser supported primitives
#     from thunder import _get_executor
#     from thunder.executors.nvfuser import lower_for_nvfuser

#     def func(a):
#         cos = tlang.cos(a)
#         return ttorch.softmax(cos, 1) + a

#     a = make_tensor((2, 3), device=device, dtype=torch.float32)

#     trace = thunder.make_trace(func, executor=executor)(a)
#     assert len(trace.symbols) == 3
#     assert trace.symbols[0].name == "cos"
#     assert trace.symbols[1].name == "torch.nn.functional.softmax"
#     assert trace.symbols[2].name == "add"

#     nvfuser_trace = thunder.make_trace(lower_for_nvfuser(func), executor=executor)(a)
#     assert len(nvfuser_trace.symbols) == 11
#     assert not any(s.name == "torch.nn.functional.softmax" for s in nvfuser_trace.symbols)

#     # Ensure that the trace can be fused and executed
#     fusion = executor.make_callable(nvfuser_trace)
#     actual = fusion(a)
#     expected = thunder.make_traced(func, executor=executor)(a)
#     assert_close(actual, expected)


@instantiate(dtypes=NOTHING)
def test_nested_trace(executor, device, _):
    # This test ensures that trace() can be called from within a traced
    # function without leaking the trace context.
    # from thunder import _get_executor

    def foo(a, b):
        return clang.add(a, b)

    def bar(a, b):
        foo_trace = thunder.trace(inline_trace=False)(foo, a, b)
        assert len(foo_trace.bound_symbols) == 4
        assert foo_trace.bound_symbols[-2].sym.name == "add"
        return clang.mul(a, b)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    bar_trace = thunder.trace()(bar, a, b)
    assert len(bar_trace.bound_symbols) == 4
    assert bar_trace.bound_symbols[-2].sym.name == "mul"

    fusion = executor.make_callable(bar_trace)
    actual = fusion(a, b)
    expected = a * b
    assert_close(actual, expected)


@instantiate(dtypes=NOTHING)
def test_nested_trace_no_name_collision(executor, device, _):
    def foo(a, b):
        return clang.add(a, b)

    def bar(__a, __b):
        a, b = __a, __b
        foo_trace = thunder.trace(inline_trace=False)(foo, a, b)
        # The name of the output of the add symbol should not be the same as
        # the name of the first argument to the bar function.
        assert foo_trace.bound_symbols[-2].sym.name == "add"
        assert foo_trace.bound_symbols[-2].output.name != foo_trace.args[0].name
        return foo(a, b)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    thunder.trace()(bar, a, b)


@instantiate(dtypes=NOTHING)
def test_trace_args_no_name_collision(executor, device, _):
    from thunder.core.trace import detached_trace
    from thunder.core.proxies import TensorProxy

    with detached_trace():
        a = TensorProxy(
            name="__a",
            shape=(2, 2),
            device=thunder.core.devices.cpu,
            dtype=thunder.core.dtypes.float32,
            requires_grad=False,
        )

    def func(*args):
        return args[0] + args[1]

    trace = thunder.trace()(func, a, a)
    # trace.args must have non-duplicate names
    # because Python disallows duplicate names in function definitions
    assert trace.args[0].name != trace.args[1].name


@instantiate(dtypes=NOTHING)
def test_eval_trace(executor, device, _):
    # This test ensures that eval_trace() can be called from within a trace
    #   and that all the symbols in the trace are properly evaluated.

    from thunder.core.transforms import eval_trace
    from thunder.core.trace import TraceCtx
    from thunder.core.proxies import TensorProxy

    def foo(a, b, *, c=5):
        return clang.mul(clang.add(a, b), c)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    c = 4.0

    # Test eval_trace() with eager proxy execution
    foo_trace = thunder.trace()(foo, a, b, c=c)
    try:
        trace = TraceCtx(None)
        trace_token = set_tracectx(trace)
        new_args = [arg for arg in foo_trace.args]
        new_kwargs = {k: v for k, v in foo_trace.kwargs.items()}
        # TODO: trace object doesn't respect the original tuple/non-tuple spec
        # for output, it's always a tuple
        actual = eval_trace(foo_trace, *new_args, **new_kwargs)
        # actual = result[0]
        assert isinstance(actual, TensorProxy)
        # assert actual.shape == foo_trace.output[0].shape
        # assert actual.dtype == foo_trace.output[0].dtype
        # assert actual.device == foo_trace.output[0].device
        assert actual.shape == foo_trace.output.shape
        assert actual.dtype == foo_trace.output.dtype
        assert actual.device == foo_trace.output.device
    finally:
        reset_tracectx(trace_token)

    # Test eval_trace() with retracing + fusion + execution
    def eval_trace_as_function(trace):
        def func(*args, **kwargs):
            return eval_trace(trace, *args, **kwargs)

        return func

    foo_traced = executor.make_callable(eval_trace_as_function(foo_trace))
    actual = foo_traced(a, b, c=c)
    expected = (a + b) * c
    assert_close(actual, expected)

    # Test eval_trace() with retracing
    foo_trace2 = thunder.trace()(eval_trace_as_function(foo_trace), a, b, c=c)
    # How to test that two traces are equal?
    # Two operators and others are do-nothing annotations
    assert len(foo_trace2.bound_symbols) == 7
    assert foo_trace2.bound_symbols[-3].sym.name == "add"
    assert foo_trace2.bound_symbols[-2].sym.name == "mul"


@instantiate(
    dtypes=NOTHING,
    decorators=(pytest.mark.xfail(reason='issue "flaky test: test_transforms_vjp_{2_1, 1_2}_nvfuser_cuda_None"'),),
)
def test_transforms_vjp_1_2(executor, device, _):
    from thunder.core.transforms import vjp

    # 1 input, 2 outputs
    def func_1_2(x):
        a = clang.sin(x)
        b = clang.add(0.2, a)
        c = clang.asin(b)
        return b, c

    a = make_tensor((2, 3), device=device, dtype=torch.float32)

    g1 = make_tensor((2, 3), device=device, dtype=torch.float32)
    g2 = make_tensor((2, 3), device=device, dtype=torch.float32)

    primals = (a,)
    cotangents = (g1, g2)
    initial_trace = thunder.trace()(vjp(func_1_2), primals, cotangents)
    vjp_eager = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)
    out_p, grads = vjp_eager(primals, cotangents)
    expected_out_p = executor.make_callable(func_1_2)(a)
    assert_close(out_p, expected_out_p, equal_nan=True, atol=1e-3, rtol=1e-5)

    # Now check the gradients
    # TODO: We will have this automatically tested with OpInfo tests
    aa = a.clone().requires_grad_(True)

    def pt_func_1_2(x):
        a = torch.sin(x)
        b = torch.add(0.2, a)
        c = torch.asin(b)
        return b, c

    out = pt_func_1_2(aa)
    expected_grads = torch.autograd.grad(out, aa, grad_outputs=(g1, g2), retain_graph=True)
    assert_close(expected_grads, grads, equal_nan=True, atol=1e-3, rtol=1e-5)


@instantiate(
    dtypes=NOTHING,
)
def test_transforms_vjp_2_2_kwarg(executor, device, _):
    # This test ensures that combination of positional and keyword arguments
    # is differentiable.
    from thunder.core.transforms import vjp

    # 2 inputs, 1 kwarg, 2 outputs
    def func_2_2(x, y, *, z):
        def func(x):
            a = clang.sin(x)
            b = clang.add(0.2, a)
            c = clang.asin(b)
            return c

        a, b = func(x), func(y)
        c = clang.add(a, b)
        d = clang.add(c, func(z))
        return c, d

    x = make_tensor((2, 3), device=device, dtype=torch.float64)
    y = make_tensor((2, 3), device=device, dtype=torch.float64)
    z = make_tensor((2, 3), device=device, dtype=torch.float64)

    g1 = make_tensor((2, 3), device=device, dtype=torch.float64)
    g2 = make_tensor((2, 3), device=device, dtype=torch.float64)

    primals = (x, y)
    primal_kwargs = {"z": z}
    cotangents = (g1, g2)
    initial_trace = thunder.trace()(vjp(func_2_2), primals, cotangents, **primal_kwargs)
    vjp_eager = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)
    out_p, grads = vjp_eager(primals, cotangents, **primal_kwargs)
    expected_out_p = executor.make_callable(func_2_2)(*primals, **primal_kwargs)
    assert_close(out_p, expected_out_p, equal_nan=True)

    # Now check the gradients
    # TODO: We will have this automatically tested with OpInfo tests
    xx = x.clone().requires_grad_(True)
    yy = y.clone().requires_grad_(True)
    zz = z.clone().requires_grad_(True)

    def pt_func_2_2(x, y, *, z):
        def func(x):
            a = torch.sin(x)
            b = torch.add(0.2, a)
            c = torch.asin(b)
            return c

        a, b = func(x), func(y)
        c = torch.add(a, b)
        d = torch.add(c, func(z))
        return c, d

    out = pt_func_2_2(xx, yy, z=zz)
    expected_grads = torch.autograd.grad(out, [xx, yy, zz], grad_outputs=(g1, g2), retain_graph=True)
    # vjp returns a tuple of (primals, cotangents) where cotangents is a tuple of
    # derivatives with respect to the positional arguments and a dict of derivatives
    # with respect to the keyword arguments.
    *gprimals, gkwargs = grads
    assert_close(expected_grads[:2], gprimals, equal_nan=True)
    assert_close(expected_grads[2], gkwargs["z"], equal_nan=True)


@instantiate(
    dtypes=NOTHING,
    decorators=(pytest.mark.xfail(reason='issue "flaky test: test_transforms_vjp_{2_1, 1_2}_nvfuser_cuda_None"'),),
)
def test_transforms_vjp_2_1(executor, device, _):
    from thunder.core.transforms import vjp

    def pt_func_2_1(x, y):
        a = torch.sin(x + y)
        b = torch.add(0.2, a)
        c = torch.asin(b)
        return c

    def func_2_1(x, y):
        a = clang.sin(x + y)
        b = clang.add(0.2, a)
        c = clang.asin(b)
        return c

    a = make_tensor((2, 3), device=device, dtype=torch.float32)
    b = make_tensor((2, 3), device=device, dtype=torch.float32)
    g1 = make_tensor((2, 3), device=device, dtype=torch.float32)
    primals = (a, b)
    cotangents = (g1,)
    initial_trace = thunder.trace()(vjp(func_2_1), primals, cotangents)
    vjp_eager = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)
    out_p, grads = vjp_eager(primals, cotangents)
    expected_out_p = executor.make_callable(func_2_1)(*primals)
    assert_close(out_p, expected_out_p, equal_nan=True)

    aa = a.clone().requires_grad_(True)
    bb = b.clone().requires_grad_(True)
    out = pt_func_2_1(aa, bb)
    expected_grads = torch.autograd.grad(out, [aa, bb], grad_outputs=(g1,), retain_graph=True)
    assert_close(expected_grads, grads, equal_nan=True)


def test_traceback():
    def f(a):
        return -(a > 0)  # negating a bool tensor raises

    compiled_f = thunder.jit(f)
    a = torch.ones((), dtype=torch.float32)
    with pytest.raises(RuntimeError) as excinfo:
        compiled_f(a)
    assert "on a bool tensor" in str(excinfo.value)
    # this should actually be in excinfo.traceback[-1] but see
    # https://github.com/Lightning-AI/lightning-thunder/issues/844
    assert any(("torch.neg" in str(tb.statement)) for tb in excinfo.traceback)
    assert any(("thunder.computation" in str(tb.path)) for tb in excinfo.traceback)


@instantiate(
    dtypes=NOTHING,
    executors=(TorchExecutor,),
)
def test_torch_tensor_to_memory_format(executor: TestExecutor, device: str, _):
    inp = torch.randn(2, 4, 5, 3, device=device, dtype=torch.float32)

    def torch_to(a, memory_format):
        return a.to(memory_format=memory_format)

    cfn = executor.make_callable(torch_to, disable_preprocessing=False)

    for m_format in [torch.contiguous_format, torch.channels_last, torch.preserve_format]:
        thunder_result = cfn(inp, torch.contiguous_format)
        torch_result = torch_to(inp, torch.contiguous_format)
        assert_close(torch_result, thunder_result, check_stride=True)


# TODO See issue "Add contiguous and clang.stride_order OpInfos that check stride
# consistency with PyTorch"
@instantiate(
    dtypes=NOTHING,
    executors=(TorchExecutor,),
)
def test_contiguous_and_stride_order(executor: TestExecutor, device: str, _):
    inp = torch.randn(2, 4, 5, 3, device=device, dtype=torch.float32).permute(0, 3, 1, 2)

    def foo(a, order):
        return clang.stride_order(a, order)

    # stride order, expected strides (from shape 2, 4, 5, 3)
    test_cases = (
        ((3, 2, 1, 0), (60, 20, 5, 1)),
        ((3, 0, 2, 1), (60, 1, 15, 3)),
    )

    for stride_order, expected_strides in test_cases:
        cfoo = executor.make_callable(foo)
        o = cfoo(inp, stride_order)
        assert o.shape == inp.shape
        assert o.stride() == expected_strides
        assert_close(inp, o)

    # order=none case
    thunder_result = cfoo(inp, None)
    assert thunder_result.stride() == (60, 20, 5, 1)
    assert_close(thunder_result, inp)

    # Memory format tests
    def channels_last_2d(a, memory_format):
        return a.contiguous(memory_format=memory_format)

    cfn = executor.make_callable(channels_last_2d, disable_preprocessing=False)

    # Contiguous cases
    a = torch.randn((4, 3, 2), device=device, dtype=torch.float32)
    thunder_result = cfn(a, torch.contiguous_format)
    torch_result = channels_last_2d(a, torch.contiguous_format)
    assert_close(torch_result, thunder_result, check_stride=True)

    b = a.permute(0, 2, 1)
    thunder_result = cfn(a, torch.contiguous_format)
    torch_result = channels_last_2d(a, torch.contiguous_format)
    assert_close(torch_result, thunder_result, check_stride=True)

    # Channels last 2D cases
    a = torch.randn((5, 4, 3, 2), device=device, dtype=torch.float32)
    thunder_result = cfn(a, torch.channels_last)
    torch_result = channels_last_2d(a, torch.channels_last)
    assert_close(torch_result, thunder_result, check_stride=True)

    b = a.permute(3, 1, 0, 2)
    thunder_result = cfn(b, torch.channels_last)
    torch_result = channels_last_2d(b, torch.channels_last)
    assert_close(torch_result, thunder_result, check_stride=True)

    # Channels last 3D cases
    a = torch.randn((5, 4, 3, 7, 2), device=device, dtype=torch.float32)
    thunder_result = cfn(a, torch.channels_last_3d)
    torch_result = channels_last_2d(a, torch.channels_last_3d)
    assert_close(torch_result, thunder_result, check_stride=True)

    b = a.permute(0, 4, 2, 1, 3)
    thunder_result = cfn(a, torch.channels_last_3d)
    torch_result = channels_last_2d(a, torch.channels_last_3d)
    assert_close(torch_result, thunder_result, check_stride=True)


@instantiate(dtypes=NOTHING)
def test_inplace(executor, device, _):
    # Usually in this scenario we would make a big list of
    # the names of methods to test, then use getattr() to call
    # them in the trace. However, this would not also test that
    # the syntax wouldn't get broken by preprocessing.

    def test_add(s, o):
        s += o
        return s

    def test_and(s, o):
        s &= o
        return s

    def test_concat(s, o):
        s.__iconcat__(o)
        return s

    def test_floordiv(s, o):
        s //= o
        return s

    def test_lshift(s, o):
        s <<= o
        return s

    def test_matmul(s, o):
        s @= o
        return s

    def test_mod(s, o):
        s %= o
        return s

    def test_mul(s, o):
        s *= o
        return s

    def test_or(s, o):
        s |= o
        return s

    def test_pow(s, o):
        s **= o
        return s

    def test_rshift(s, o):
        s >>= o
        return s

    def test_sub(s, o):
        s -= o
        return s

    def test_truediv(s, o):
        s /= o
        return s

    def test_xor(s, o):
        s ^= o
        return s

    t1 = make_tensor((2, 3), device=device, dtype=torch.float32)
    t2 = make_tensor((1, 2), device=device, dtype=torch.float32)

    tests = (
        test_add,
        test_and,
        test_concat,
        test_floordiv,
        test_lshift,
        test_matmul,
        test_mod,
        test_mul,
        test_or,
        test_pow,
        test_rshift,
        test_sub,
        test_truediv,
        test_xor,
    )

    for t in tests:
        cfn = thunder.jit(t)
        # Some ops of `tests` already have in-place supported, leading to broadcast error
        with pytest.raises(RuntimeError, match="not supported|Attempting"):
            cfn(t1, t2)
        # Note: Python maps inplace operations on (immutuables) to
        #       out of place operations, NumberProxy does this, too.

        if t not in {
            test_concat,
            test_lshift,
            test_matmul,
            test_rshift,
        }:
            assert cfn(5, 6) == t(5, 6)

        if t not in {test_and, test_concat, test_lshift, test_matmul, test_or, test_rshift, test_xor}:
            assert cfn(1.2, 2.4) == t(1.2, 2.4)
            if t not in {test_floordiv, test_mod}:
                assert cfn(1.2j, 2.4j) == t(1.2j, 2.4j)


@instantiate(dtypes=NOTHING)
def test_thunder_autocast_transform(executor, device, _):
    from thunder.transforms.autocast import autocast

    def f(a, b, c):
        return a @ (b + c)

    # The following functions needs to be updated as autocast_impls grows.
    def g(a, b, c):
        return a + b - c

    def h(a, b, c):
        return (a @ b) + c

    for func, should_autocast in ((f, True), (g, False), (h, False)):
        dtype = thunder.bfloat16 if device == "cpu" else thunder.float16
        torch_dtype = ltorch.to_torch_dtype(dtype)
        x, y, z = (torch.randn((2, 2), device=device, dtype=torch.float32) for _ in range(3))
        initial_trace = thunder.trace()(autocast(func, dtype=dtype), x, y, z)
        compiled = thunder.jit(initial_trace.python_callable(), executors=executor.executors_list())
        out = compiled(x, y, z)
        traces = thunder.last_traces(compiled)
        assert out.dtype == (torch_dtype if should_autocast else torch.float32), traces[-1]

        # note(crcrpar): This test could be broken in the future as thunder autocast develops.
        devicetype = torch.device(device).type
        with torch.autocast(device_type=devicetype, dtype=torch_dtype):
            torch_output = func(x, y, z)
        assert out.dtype == torch_output.dtype


@instantiate(dtypes=NOTHING)
def test_torch_scaled_dot_product_attention_non_decomposed(executor, device, _):
    n_embd = 32
    B = 2
    qkv = make_tensor(B, n_embd, 3 * n_embd, device=device, dtype=torch.float32)

    def func(qkv):
        # Preprocessing doesn't support nonlocal variables yet, so
        # we need to define the constants here.
        n_embd = 32
        n_head = 16
        B = 2
        T = 32
        C = n_embd
        q, k, v = qkv.split(n_embd, dim=2)  # Results in 3 non-contiguous but "viewable" tensors
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        return y

    compiled = thunder.jit(func, executors=executor.executors_list())
    out = compiled(qkv)
    traces = thunder.last_traces(compiled)
    torch.testing.assert_close(out, func(qkv))
    assert "scaled_dot_product_attention" in tuple(bsym.sym.id for bsym in traces[-1].bound_symbols)


@instantiate(
    dtypes=NOTHING,
    # https://github.com/Lightning-AI/lightning-thunder/issues/946
    decorators=(pytest.mark.xfail(reason="Thunder JIT may rename variables differently, causing the test to fail."),),
)
def test_cse(executor, device, _):
    from thunder.core.pytree import tree_flatten

    def func(x, y, device):
        a = x * y
        b = y / x
        c = x * y  # Expected to be removed in favor of `a`.
        d = y / x  # Expected to be removed in favor of `b`.
        z = a * b  # Expected to be intact.
        w = c * d  # Expected to be converted to `w = a * b` and then removed in favor of `z`.
        m = w * 1  # Expected to be updated to `m = z * 1`.
        a = clang.uniform(w.shape, device=device, dtype=thunder.float16)
        b = clang.uniform(w.shape, device=device, dtype=thunder.float16)
        c = clang.uniform(z.shape, device=device, dtype=thunder.float16)
        d = clang.uniform(z.shape, device=device, dtype=thunder.float16)
        return z, w, m, (a, b, c, d)

    x, y = (make_tensor((2, 2), device=device, dtype=torch.float32) for _ in range(2))
    initial_trace = thunder.trace()(func, x, y, device)
    compiled_func = thunder.jit(
        initial_trace.python_callable(),
        executors=executor.executors_list(),
    )
    compiled_func(x, y, device)
    traces = thunder.last_traces(compiled_func)
    flatten_dce_trace = [
        t for t in traces if t._provenance is not None and t._provenance.pss.startswith("Dead Code Elimination")
    ][1]

    from thunder.core.transform_common import cse

    flatten_cse_trace = cse(flatten_dce_trace)

    # # CSE is supposed to remove `c`, `d`, and `w`.
    assert len(flatten_cse_trace.bound_symbols) == len(flatten_dce_trace.bound_symbols) - 3
    assert len([bsym for bsym in flatten_cse_trace.bound_symbols if bsym.sym.id == prims.PrimIDs.UNIFORM]) == 4

    assert [t.name for t in tree_flatten(flatten_cse_trace.output)[0]] == ["t4", "t4", "t6", "t14", "t15", "t16", "t17"]


def test_symbol_flat_args():
    from thunder.core.symbol import Symbol, BoundSymbol

    def func(x, y, *, z):
        return x * y + z

    sym = Symbol(meta=func, name="func", id=0)
    bsym = BoundSymbol(sym, args=(1, 2), kwargs={"z": 3}, output=None)
    assert bsym.flat_args == [1, 2, 3]


@instantiate(dtypes=NOTHING)
def test_preserve_weight_names(executor, device: str, dtype: dtypes.dtype):
    import inspect

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(3, 4)
            self.fc2 = torch.nn.Linear(4, 5)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = MLP().to(device=device, dtype=ltorch.to_torch_dtype(dtype))
    x = torch.randn(2, 3, device=device, dtype=ltorch.to_torch_dtype(dtype))

    compiled = thunder.jit(model, executors=executor.executors_list())
    compiled(x)
    traces = thunder.last_traces(compiled)
    sig = inspect.signature(traces[-1].python_callable())
    assert "t_fc1_bias" in sig.parameters
    assert "t_fc1_weight" in sig.parameters
    assert "t_fc2_bias" in sig.parameters
    assert "t_fc2_weight" in sig.parameters


@requiresCUDA
def test_clone():
    def foo(a):
        return a.clone()

    jfoo = thunder.jit(foo)
    for shp in ((3, 5), [7], (8, 6, 4)):
        for dev in (torch.device("cpu"), torch.device("cuda:0")):
            for dt in (torch.float32, torch.float16, torch.bfloat16):
                # there are issues with layouts other than strided; see
                # test_clone_sparse_coo.
                lout = torch.strided
                b = jfoo(torch.randn(shp, device=dev, layout=lout, dtype=dt))
                assert b.dtype == dt
                assert b.layout == lout
                assert b.device == dev
                assert b.shape == torch.Size(shp)


# Separate out the sparse test because creating a sparse tensor is tricky.
def test_clone_sparse_coo():
    def foo(a):
        return a.clone()

    jfoo = thunder.jit(foo)
    shp = (3, 5)
    dev = torch.device("cpu")
    dt = torch.float32
    # randn(layout=torch.sparse_coo, ...) will throw an exception deep in
    # PyTorch, so we use to_sparse() from a dense tensor to get a sparse one.
    b = jfoo(torch.randn(shp, device=dev, dtype=dt).to_sparse())
    assert b.dtype == dt
    assert b.layout == torch.sparse_coo
    assert b.device == dev
    assert b.shape == torch.Size(shp)


@pytest.mark.xfail(reason="we improperly use an alias")
def test_clone_alias():
    def foo(a):
        b = a.clone()
        b[0] = 42

    jfoo = thunder.jit(foo)
    arg = torch.tensor([7, 19])
    jfoo(arg)
    assert arg[0] == 7


@instantiate(dtypes=(thunder.float32,))
def test_default_method(executor, device: str, dtype: dtypes.dtype):
    # This test ensures that when no language context is given, it will fallback to the default implementation.
    from thunder.core.trace import detached_trace
    from thunder.core.proxies import TensorProxy

    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)

    with detached_trace():
        b = TensorProxy(
            name="__b",
            shape=(2, 2),
            device=thunder.core.devices.cpu,
            dtype=thunder.core.dtypes.float32,
            requires_grad=False,
        )

    # torch.numel(a) and a.numel() will run on PyTorch contenxt
    # b.numel will fall back to the default implementation
    assert torch.numel(a) == a.numel() == b.numel


# @instantiate(
#     dtypes=NOTHING,
#     executors=[
#         TorchEx(),
#     ],
# )
# def test_get_executor(executor, device, _):
#     from thunder import _get_executor
#     from thunder.executors.torch import torchCtx

#     with pytest.raises(ValueError, match="No executor specified!"):
#         _get_executor(None)

#     ex = _get_executor(executor)
#     if executor.name == "TorchEx":
#         assert isinstance(ex, torchCtx)


# TODO Move to test_tensor_creation.py
# @instantiate(
#     dtypes=(thunder.float32, thunder.float16),
# )
# def test_uniform(executor, device, dtype):
#     if isinstance(executor, nvFuser) and LooseVersion(executor.version()) < "0.0.3":
#         pytest.skip("'uniform' not implemented before nvfuser 0.0.3")

#     thunder_uniform = executor.make_callable(tlang.uniform)
#     uniform = partial(thunder_uniform, dtype=dtype, device=device)

#     # lo, hi, shape
#     cases = (
#         (-12.0, 128, (8, 12, 7)),
#         (-0.3, 0.5, (2, 3, 4, 1)),
#         (0.0, 128.0, (2, 4)),
#         (-12.0, 0.0, (8, 3)),
#         (-1e-3, 1e-3, (8, 3)),
#         (0.0, 7.0, (0, 3)),
#         (0.0, 1.0, ()),
#     )

#     for lo, hi, shape in cases:
#         result = uniform(shape, lo, hi)
#         assert result.shape == shape
#         # note: numpy.random.uniform take value from [lo, hi)
#         #       But that doesn't seem to be the case for all backends. I'm relaxing this
#         if result.numel() != 0:
#             assert result.min() >= lo
#             assert result.max() <= hi

#     def foo():
#         return tlang.uniform([2, 3, 4], 0.5, 1.0, dtype=dtype, device=device)

#     thunder_static_uniform = executor.make_callable(foo)
#     result = thunder_static_uniform()
#     result.shape == (2, 3, 4)
#     result.min() >= 0.5
#     result.max() <= 1.0


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_torch_gen_remove_last_used_variables(executor, device, _):
#     # This test is to make sure that the last used variables are removed
#     # from the generated code. This is important for freeing up memory.
#     from thunder.executors.torch import _fuse_region

#     def foo(a):
#         b = a + 1.0
#         c = b + 1.0
#         d = c + 1.0
#         e = d + 1.0
#         return e

#     a = make_tensor((2, 2), device=device, dtype=torch.float32)
#     trace = thunder.make_trace(foo, executor=executor)(a)
#     code_str, _ = _fuse_region((), [trace.outputs], trace.symbols)

#     # Check that there are for del commands
#     assert code_str.count("del") == 4

#     def foo(a):
#         b = a + 1.0
#         c = b + 1.0
#         d = c + 1.0
#         e = d + 1.0
#         return e, d

#     a = make_tensor((2, 2), device=device, dtype=torch.float32)
#     trace = thunder.make_trace(foo, executor=executor)(a)
#     code_str, _ = _fuse_region(_, [trace.outputs], trace.symbols, global_outputs=trace.outputs)
#     # Same as above, but now the last del should be removed since the variable
#     # is used in the output
#     assert code_str.count("del") == 3


@instantiate(dtypes=(thunder.float32,), executors=(TorchExecutor,))
def test_bound_symbol_source_location_context(executor, device: str, dtype: dtypes.dtype):
    def foo(x):
        return clang.sin(x)

    a = make_tensor((2, 2), device=device, dtype=ltorch.to_torch_dtype(dtype))

    lineno = foo.__code__.co_firstlineno + 1
    jfn = thunder.jit(foo)
    jfn(a)

    trace = thunder.last_traces(jfn)[0]

    assert len(trace.bound_symbols) == 3
    sin_symbol = trace.bound_symbols[1]
    assert str(trace).count("return clang.sin(x)") == 1
    assert str(trace).count(f"# {__file__}:{lineno}") == 1


@instantiate(dtypes=(thunder.float32,), executors=(TorchExecutor,))
def test_refine_source_location(executor, device: str, dtype: dtypes.dtype):
    def foo_thunder(x):
        return thunder.torch.softmax(x, 0)

    def foo_torch(x):
        return torch.softmax(x, 0)

    a = make_tensor((2, 2), device=device, dtype=ltorch.to_torch_dtype(dtype))

    jfn_thunder = thunder.jit(foo_thunder)
    jfn_thunder(a)
    jfn_torch = thunder.jit(foo_torch)
    jfn_torch(a)

    trace_thunder = thunder.last_traces(jfn_thunder)[0]
    trace_torch = thunder.last_traces(jfn_torch)[0]

    # make sure we are not showing the internals of Thunder
    assert str(trace_thunder).count("return _softmax(a, dim=dim, dtype=dtype)") == 0
    assert str(trace_thunder).count("return thunder.torch.softmax(x, 0)") == 1
    # torch.softmax should be traced as usual
    assert str(trace_torch).count("return torch.softmax(x, 0)") == 1


def test_torch_device():
    # Test `thunder.jit` support for `torch.device()`.
    if not torch.cuda.is_available():
        # thunder.core.devices.Device __init__ calls `torch.cuda.device_count()` when DeviceType is CUDA.
        # https://github.com/Lightning-AI/lightning-thunder/blob/067f15aae47ad71229732ca6c35a5d190135e48c/thunder/core/devices.py#L96-L101
        pytest.skip("CUDA not available")

    # Check the output against the PyTorch eager output.
    def _test(foo, inputs):
        for input in inputs:
            actual = thunder.jit(foo)(input)
            expected = foo(input)
            assert actual.device == expected.device

    # Test with str input
    device_strs = ("cpu", "cuda", "cuda:0", "meta")

    def foo1(dev):
        # If we return the device here, thunder.jit version will return `thunder.device`
        # while eager will return `torch.device`
        # https://github.com/Lightning-AI/lightning-thunder/issues/573
        return torch.ones(3, 3, device=torch.device(dev))

    _test(foo1, device_strs)

    # Test with str and index input
    device_strs_and_idxs = (("cpu", 0), ("cpu", 1), ("cuda", 0), ("meta", 0), ("meta", 1))

    def foo2(dev_and_idx):
        return torch.ones(3, 3, device=torch.device(*dev_and_idx))

    _test(foo2, device_strs_and_idxs)

    def foo2_1(dev_and_idx):
        dev_type, idx = dev_and_idx
        return torch.ones(3, 3, device=torch.device(type=dev_type, index=idx))

    _test(foo2_1, device_strs_and_idxs)

    # Test with `torch.device` as input
    torch_devices = (torch.device("cpu"), torch.device("cuda"), torch.device("meta"))

    def foo3(device):
        return torch.ones(3, 3, device=torch.device(device))

    _test(foo3, torch_devices)

    # Test with `thunder.device` as input
    tensor_proxy_devices = (
        torch.ones(1, device=torch.device("cpu")),
        torch.ones(1, device=torch.device("cuda")),
        torch.ones(1, device=torch.device("meta")),
    )

    # Here `torch.device()` will see a `thunder.device` as input.
    def foo4(ref_t):
        return torch.ones(3, 3, device=torch.device(ref_t.device))

    _test(foo4, tensor_proxy_devices)

    # Error inputs
    error_inputs = (
        ((torch.device("cpu"), 0), RuntimeError),
        (("cuda:0", 0), RuntimeError),
        (("cpu:",), ValueError),
        (("cuda:",), ValueError),
    )

    def foo_error(args):
        return torch.device(*args)

    for inp, err in error_inputs:
        with pytest.raises(err):
            thunder.jit(foo_error)(inp)


def test_grad_ctx():
    # NOTE - This test would start failing if tags on Proxies are dropped
    # as the computation under `no_grad` won't be treated as constant
    # and grad won't match with PyTorch eager.

    # Test `enable_grad` on a function works correctly
    @torch.enable_grad()
    def foo1(x):
        return x + 1

    x = torch.randn(3, 3, requires_grad=True)
    thunder.jit(foo1)(x).sum().backward()
    assert x.grad is not None

    # Test `no_grad` on a function works correctly
    @torch.no_grad()
    def foo2(x):
        return x + 1

    x = torch.randn(3, 3, requires_grad=True)
    res = thunder.jit(foo2)(x)
    assert not res.requires_grad

    # Test `no_grad` ctx correctly disable gradient computation
    def foo3(x):
        with torch.no_grad():
            y = x * 3
        return x * 2 + y

    x = torch.randn(3, 3, requires_grad=True)
    with torch.no_grad():
        x_ref = x.clone()
        x_ref.requires_grad_(True)

    foo3(x_ref).sum().backward()
    thunder.jit(foo3)(x).sum().backward()
    # Verify the gradients match
    torch.testing.assert_close(x.grad, x_ref.grad)

    # Test nested `no_grad` and `enable_grad`
    def foo4(x):
        with torch.enable_grad():
            with torch.no_grad():
                y = x * 3
            z = x * 4
        return x * 2 + y + z

    x = torch.randn(3, 3, requires_grad=True)
    with torch.no_grad():
        x_ref = x.clone()
        x_ref.requires_grad_(True)

    foo4(x_ref).sum().backward()
    thunder.jit(foo4)(x).sum().backward()
    # Verify the gradients match
    torch.testing.assert_close(x.grad, x_ref.grad)

    def foo5(x):
        return x * 2

    x = torch.randn(3, 3, requires_grad=True)
    with torch.no_grad():
        x_ref = x.clone()
        x_ref.requires_grad_(True)

    jfoo = thunder.jit(foo5)
    with torch.no_grad():
        o = jfoo(x)
        assert o.grad_fn is None
        assert thunder.cache_misses(jfoo) == 1  # First compilation

    # Running it out of `torch.no_grad`, should lead to recompile.
    foo5(x_ref).sum().backward()
    jfoo(x).sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    assert thunder.cache_misses(jfoo) == 2


def test_serialize_trace():
    import dill as pickle

    def fn(a, b, l):
        res = a + b
        for t in l:
            res = res + t
        return res

    tm = thunder.jit(fn)
    a, b = torch.randn(2, 5, device=("cuda" if torch.cuda.is_available() else "cpu"))
    tm(a, b, [a, b])
    trace = thunder.last_traces(tm)[0]

    assert str(pickle.loads(pickle.dumps(trace))) == str(trace)

    prologue_trace = thunder.last_prologue_traces(tm)[0]

    assert str(pickle.loads(pickle.dumps(prologue_trace))) == str(prologue_trace)

    # check that these are looked up rather than duplicated
    device = thunder.devices.Device("cpu")
    assert pickle.loads(pickle.dumps(device)) is device
    fp32 = thunder.dtypes.float32
    assert pickle.loads(pickle.dumps(fp32)) is fp32


@pytest.mark.parametrize("requires_grad", (True, False))
def test_dataclass_output(requires_grad):
    # Test both `requires_grad={True, False}` as both have
    # different code path.
    @dataclasses.dataclass
    class TestDataclass:
        t: torch.Tensor
        s: torch.Tensor
        i: int
        f: float
        g: tuple

    def foo(x):
        # TestDataClass as the output and part of the nested output.
        return TestDataclass(x, x + 2, x.numel(), x.numel() / 2.0, (x,)), (
            TestDataclass(x, x + 2, x.numel(), x.numel() / 2.0, (x)),
            {"x": x, "y": x + 3},
        )

    jfoo = thunder.jit(foo)

    x = torch.randn(3, 3, requires_grad=requires_grad)
    x_jit = x.detach().clone()
    x_jit.requires_grad_(requires_grad)

    actual_container, actual_tuple = jfoo(x_jit)
    expected_container, expected_tuple = foo(x)

    def _test_container(actual_container, expected_container):
        assert dataclasses.is_dataclass(actual_container)
        assert isinstance(actual_container, TestDataclass)
        torch.testing.assert_close(actual_container.t, expected_container.t)
        torch.testing.assert_close(actual_container.s, expected_container.s)
        torch.testing.assert_close(actual_container.i, expected_container.i)
        torch.testing.assert_close(actual_container.f, expected_container.f)
        torch.testing.assert_close(actual_container.g[0], expected_container.g[0])

    _test_container(actual_container, expected_container)
    _test_container(actual_tuple[0], expected_tuple[0])
    torch.testing.assert_close(actual_tuple[1], expected_tuple[1])

    if requires_grad:
        # Test computing grad
        cotangent = torch.randn_like(expected_container.t)
        (actual_container.t + actual_tuple[0].s).backward(cotangent)
        (expected_container.t + expected_tuple[0].s).backward(cotangent)
        torch.testing.assert_close(x.grad, x_jit.grad)


@pytest.mark.parametrize("requires_grad", (True, False))
def test_dataclass_input(requires_grad):
    @dataclasses.dataclass
    class TestDataclass:
        t: torch.Tensor
        s: torch.Tensor

    def foo(x):
        return x.t + x.s

    jfoo = thunder.jit(foo)

    t = torch.randn(3, 3, requires_grad=requires_grad)
    s = torch.randn(3, 3, requires_grad=requires_grad)
    actual = jfoo(TestDataclass(t, s))
    expected = foo(TestDataclass(t, s))

    torch.testing.assert_close(actual, expected)


def test_proxy_repr():
    # Verify that we can call `__repr__` on different proxy subclasses.
    t = thunder.core.trace.TraceCtx()
    with thunder.core.trace.tracectx(t):
        p = thunder.core.proxies.NumberProxy("number", 1, python_type=int)
        c = thunder.core.proxies.CollectionProxy((1, 2), name="collection")
        t = thunder.core.proxies.TensorProxy(
            "tensor",
            shape=(1,),
            dtype=thunder.core.dtypes.float16,
            device=thunder.core.devices.Device("cpu"),
            requires_grad=True,
        )
        assert p.__repr__() == '<NumberProxy(name="number")>'
        assert t.__repr__() == '<TensorProxy(name="tensor", dtype=thunder.dtypes.float16, shape=(1,))>'
        assert c.__repr__() == '<CollectionProxy(name="collection")>'


def test_type_string():
    def fn(x):
        result = 2 * x
        return result

    jfn = thunder.jit(fn)

    a = torch.randn(2, 2)

    jfn(a)

    tr = thunder.last_traces(jfn)[0]

    assert tr.bound_symbols[1].sym == ltorch.mul
    (pystr,) = tr.bound_symbols[1].python(0)

    assert pystr == 'result = ltorch.mul(2, x)  # result: "cpu f32[2, 2]"'


def test_dtype_in_trace():
    def fn(x):
        return x.to(torch.float16)

    jfn = thunder.jit(fn)

    x = torch.randn(
        3,
    )

    jfn(x)

    tr = thunder.last_traces(jfn)[0]
    assert tr.bound_symbols[1].sym == ltorch.to
    (pystr,) = tr.bound_symbols[1].subsymbols[0].python(0)

    assert "convert_element_type(x, dtypes.float16)" in pystr


def test_factory_functions_default_dtype():
    def fn(x):
        o = torch.ones(x.shape)
        return o.dtype

    x = torch.randn(3, 3)
    jfn = thunder.jit(fn)
    actual_dtype = jfn(x)

    assert fn(x) == jfn(x)
    assert actual_dtype == torch.float32

    # Check with a different default dtype.
    with set_default_dtype_ctx(torch.float16):
        actual_dtype = jfn(x)
        assert actual_dtype == torch.float16

    assert thunder.cache_misses(jfn) == 2


def test_change_default_dtype_in_jitted_fn():
    default_dtype = torch.get_default_dtype()
    try:

        def fn(x):
            torch.set_default_dtype(torch.float16)
            o = torch.ones(x.shape)
            return o.dtype

        jfn = thunder.jit(fn)
        with pytest.raises(RuntimeError, match="Default dtype is changed during the execution of jitted function"):
            jfn(torch.randn(3, 3))
    finally:
        torch.set_default_dtype(default_dtype)


@requiresCUDA
def test_factory_functions_default_device():
    def fn(x):
        o = torch.ones(x.shape)
        return o.device

    x = torch.randn(3, 3)
    jfn = thunder.jit(fn)
    actual_device = jfn(x)

    assert fn(x) == jfn(x)
    assert actual_device == torch.device("cpu")

    # Check with a different default device.
    org_device = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        actual_device = jfn(x)
        assert actual_device == fn(x)
    finally:
        torch.set_default_device(org_device)
        # hard clean for https://github.com/Lightning-AI/lightning-thunder/issues/844
        try:
            torch._GLOBAL_DEVICE_CONTEXT.device_context.__exit__(None, None, None)
            del torch._GLOBAL_DEVICE_CONTEXT.device_context
        except Exception:
            pass

    assert thunder.cache_misses(jfn) == 2


@requiresCUDA
def test_change_default_device_in_jitted_fn():
    default_device = torch.get_default_device()
    try:

        def fn(x):
            torch.set_default_device("cuda")
            o = torch.ones(x.shape)
            return o.device

        jfn = thunder.jit(fn)
        with pytest.raises(RuntimeError, match="Default device is changed during the execution of jitted function"):
            jfn(torch.randn(3, 3))
    finally:
        torch.set_default_device(default_device)
        # hard clean for https://github.com/Lightning-AI/lightning-thunder/issues/844
        try:
            torch._GLOBAL_DEVICE_CONTEXT.device_context.__exit__(None, None, None)
            del torch._GLOBAL_DEVICE_CONTEXT.device_context
        except Exception:
            pass


@requiresCUDA
@pytest.mark.xfail(
    compare_version("torch", operator.le, "2.7.1", use_base_version=True),
    reason="When using device as context in PyTorch, it doesn't reflect in torch.get_default_device - see https://github.com/pytorch/pytorch/issues/131328",
    strict=True,
)
def test_change_default_device_with_ctx():
    def fn(x):
        o = torch.ones(x.shape)
        return o.device

    x = torch.randn(3)

    with torch.device("cuda"):
        jfn = thunder.jit(fn)
        actual_device = jfn(x)
        assert actual_device == fn(x)


def test_arange_default_dtype():
    # If any of start, end, or stop are floating-point, the dtype is inferred to be the default dtype, see get_default_dtype().
    # Otherwise, the dtype is inferred to be torch.int64.
    def fn():
        return torch.arange(start=1, end=2, step=0.5).dtype

    jfn = thunder.jit(fn)
    assert fn() == jfn()
    assert jfn() == torch.float32

    def fn():
        return torch.arange(start=1, end=3, step=1).dtype

    jfn = thunder.jit(fn)
    assert fn() == jfn()
    assert jfn() == torch.int64


def test_randint_default_dtype():
    def fn():
        return torch.randint(0, 5, (2, 3))

    jfn = thunder.jit(fn)
    assert jfn().dtype == fn().dtype == torch.int64


def test_cat_mixed_dtypes():
    # We add a special test here instead of a sample in OpInfo.
    # When we add a mixed input sample in OpInfo, it will also be picked up for the test which
    # computes numerical Jacobian vector product and compares it with analytical. The test will produce failures
    # when run in precision lower than double (and we can't disable a sample based on tests).
    # See comment - https://github.com/Lightning-AI/lightning-thunder/pull/819#issuecomment-2244761476
    def fn(tensors):
        return torch.cat(tensors, dim=0)

    tensors = (torch.randn(3, requires_grad=True), torch.randn(3, dtype=torch.float16, requires_grad=True))
    with torch.no_grad():
        tensors_jit = tuple(t.detach().clone() for t in tensors)
        for t in tensors_jit:
            t.requires_grad_(True)

    # Compare forward
    jfn = thunder.jit(fn)
    expected = fn(tensors)
    actual = jfn(tensors_jit)
    torch.testing.assert_close(actual, expected)

    # Compare backward
    cotangent = torch.randn_like(expected)
    expected.backward(cotangent)
    actual.backward(cotangent)

    torch.testing.assert_close(tuple(t.grad for t in tensors), tuple(t.grad for t in tensors_jit))


@pytest.mark.parametrize("requires_grad", [True, False])
def test_reshape_noop_prims(requires_grad):
    # NOTE - We test for requires_grad with True and False,
    #        as the trace before execution may have `ltorch.reshape` (for requires_grad=False) or
    #        `prims.reshape` (for requires_grad=True) as the `grad` rule is only defined for `prims.reshape`.
    def fn(x: torch.Tensor, y: torch.Tensor):
        x_view = x.reshape(-1, 5)
        y_view = y.reshape(-1)
        return x_view + 3, y_view + 2

    t = torch.randn(8, 5, requires_grad=requires_grad)
    labels = torch.tensor([2, 4, 2, 3, 1, 0, 4, 4])

    jfn = thunder.jit(fn)
    actual = jfn(t, labels)
    expected = fn(t, labels)

    torch.testing.assert_close(actual, expected)


@requiresCUDA
@thunder.tests.framework.requiresNVFuser
def test_bound_symbol_sort_stability():
    class LlamaMLPLike(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc_1 = torch.nn.Linear(32, 32)
            self.fc_2 = torch.nn.Linear(32, 32)
            self.proj = torch.nn.Linear(32, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_fc_1 = self.fc_1(x)
            x_fc_2 = self.fc_2(x)
            x = torch.nn.functional.silu(x_fc_1) * x_fc_2
            return self.proj(x)

    with torch.device("cuda"):
        mlp = torch.nn.Sequential(*[LlamaMLPLike() for _ in range(16)]).requires_grad_(False)
    j = thunder.jit(mlp)
    j(torch.randn(32, 32, device="cuda"))
    lt = thunder.last_traces(j)[-1]
    assert all(
        (i % 2 + 1 == i_2)
        for i, i_2 in enumerate(
            [
                int(s.args[1].name.split("_")[-2])
                for s in lt.bound_symbols
                if s.sym.name == "linear" and "fc" in s.args[1].name
            ]
        )
    )

    fusions = examine.get_fusion_symbols(lt)

    no_number = partial(re.sub, r"nvFusion\d+", "nvFusion")
    fusions = [no_number(str(thunder.core.transform_common.canonicalize_proxies([f])[0])) for f in fusions]

    f0 = fusions[0]
    for f in fusions[1:]:
        assert f0 == f


def test_state_dict():
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.GELU(),
            torch.nn.Linear(4, 3),
        )

    m1 = make_model()
    m2 = make_model()

    jm1 = thunder.jit(m1)
    jm2 = thunder.jit(m2)

    inp = torch.randn(2, 3)

    jm2.load_state_dict(jm1.state_dict())

    torch.testing.assert_close(jm1(inp), jm2(inp))


def test_thunder_optimized_module_is_freed():
    mod = torch.nn.ReLU()
    opt_mod = thunder.jit(mod)
    ref_opt_mod = weakref.ref(opt_mod)
    x = torch.randn(10, 10)
    opt_mod(x)
    del x
    del mod
    del opt_mod
    assert ref_opt_mod() is None


@pytest.mark.xfail(strict=True)
def test_user_module_is_freed():
    mod = torch.nn.ReLU()
    opt_mod = thunder.jit(mod)
    ref_mod = weakref.ref(mod)
    x = torch.randn(10, 10)
    opt_mod(x)
    del x
    del mod
    del opt_mod
    assert ref_mod() is None


@pytest.mark.parametrize("requires_grad", [True, False])
def test_return_bsym_has_none_output(requires_grad):
    def fn(x):
        return x + 1

    x = torch.tensor([3.0], requires_grad=requires_grad)
    jfn = thunder.jit(fn)
    jfn(x)

    for trace in thunder.last_traces(jfn):
        return_bsym = trace.bound_symbols[-1]
        assert return_bsym.sym.id == thunder.prims.PrimIDs.RETURN
        assert return_bsym.output is None

    if requires_grad:
        for trace in thunder.last_backward_traces(jfn):
            return_bsym = trace.bound_symbols[-1]
            assert return_bsym.sym.id == thunder.prims.PrimIDs.RETURN
            assert return_bsym.output is None


def test_indexing_with_hashable_object():
    class HashableClass:
        def __hash__(self):
            return id(self)

    h = HashableClass()
    d = {h: 1, 1: 0}

    def fn():
        return d[h]

    jfn = thunder.jit(fn)
    assert jfn() == 1
    assert thunder.cache_misses(jfn) == 1  # Due to first compilation.

    # Call jfn with no changes
    # this should be cache hit.
    assert jfn() == 1
    assert thunder.cache_hits(jfn) == 1
    assert thunder.cache_misses(jfn) == 1

    # Change the value of the captured dict.
    # This should be a cache miss, verify that.
    d[h] = 2
    assert jfn() == 2  # Verify that jfn now returns 2
    assert thunder.cache_hits(jfn) == 1
    assert thunder.cache_misses(jfn) == 2


def test_profiling_decorator():
    @thunder.core.profile.annotate_for_profile("compile_and_run")
    def foo():
        def bar(a: torch.Tensor):
            t0 = torch.add(a, 42)
            t1 = torch.mul(t0, 0.25)
            return t1

        baz = thunder.jit(bar)
        baz(torch.randn(19))

    foo()


def test_saved_view_of_output_of_autograd_function_does_not_leak():
    # Verify that we have side-stepped the bug in torch.autograd.Function
    # where saving a view of the output for backward leads to leak.
    # See NOTE [Saved view of output of torch.autograd.Function leaks]
    def fn(idx, weight):
        tok_emb = torch.nn.functional.embedding(idx, weight)
        emb = torch.reshape(tok_emb, (2, 32))
        matmul = emb @ emb.T
        return tok_emb, matmul

    weight = make_tensor((16, 32), dtype=torch.float, device="cpu", requires_grad=True)
    x = make_tensor((1, 2), dtype=torch.int64, low=0, high=10, device="cpu")

    jfn = thunder.jit(fn)

    # Computation Trace for jfn
    # We save view of the output `tok_emb` for backward.
    # @torch.no_grad()
    # @no_autocast
    # def computation(idx, t_wte_weight):
    #   # idx: "cuda:0 i64[1, 2]"
    #   # t_wte_weight: "cuda:0 f32[16, 32]"
    #   tok_emb = torch.nn.functional.embedding(idx, t_wte_weight, None, None, 2.0, False, False)  # tok_emb: "cuda:0 f32[1, 2, 32]"
    #   [emb, t4] = nvFusion0(tok_emb)
    #       # emb = prims.reshape(tok_emb, (2, 32))  # emb: "cuda:0 f32[2, 32]"
    #       # t4 = prims.transpose(emb, (1, 0))  # t4: "cuda:0 f32[32, 2]"
    #   matmul = torch.matmul(emb, t4)  # matmul: "cuda:0 f32[2, 2]"
    #   return {'output': (tok_emb, matmul), 'flat_args': [idx, t_wte_weight], 'flat_output': (tok_emb, matmul)}, ((emb, idx, t4), ())

    prev_iter_refs = []
    for iter_n in range(4):
        tok_emb, _ = jfn(x, weight)
        if iter_n < 3:
            prev_iter_refs.append(weakref.ref(tok_emb))

    for ref in prev_iter_refs:
        assert ref() is None


def test_debug_options():
    from thunder import DebugOptions
    import dill

    initial_state = dill.dumps(dict(DebugOptions.__dict__))
    DebugOptions.register_option("test_option", bool, False, "Test Option")

    assert "Test Option" in DebugOptions.__doc__

    do = DebugOptions()
    assert do.test_option is False
    do = DebugOptions(test_option=True)
    assert do.test_option is True

    with pytest.raises(TypeError, match="test_option"):
        do = DebugOptions(test_option=5)

    del DebugOptions._docs["test_option"]
    del DebugOptions._defaults["test_option"]
    del DebugOptions.__annotations__["test_option"]
    del DebugOptions.test_option

    DebugOptions._set_docstring()
    assert dill.dumps(dict(DebugOptions.__dict__)) == initial_state


def test_default_tensor_proxy():
    from thunder.core.proxies import TensorProxy
    from thunder.core.trace import detached_trace
    from thunder.core.dtypes import float32
    from thunder.core.devices import cpu

    # It should be possible to create a TensorProxy with default values for all
    # optional arguments
    with detached_trace():
        t = TensorProxy(shape=(1,), device=cpu, dtype=float32)
    assert not t.requires_grad
    assert t.device == cpu
    assert t.dtype == float32


def test_proxy_same_name():
    from thunder.core.proxies import TensorProxy
    from thunder.core.trace import detached_trace
    from thunder.core.dtypes import float32
    from thunder.core.devices import cpu

    with detached_trace():
        t = TensorProxy(name="test", shape=(1,), device=cpu, dtype=float32)
        with pytest.raises(RuntimeError, match="already used"):
            t2 = TensorProxy(name="test", shape=(1,), device=cpu, dtype=float32)


def test_save_trace():
    def fn(x):
        return x + 1

    jfn = thunder.jit(fn)
    jfn(
        torch.rand(
            3,
        )
    )

    fwd_trace = thunder.last_traces(jfn)[-1]

    with tempfile.TemporaryDirectory() as tmp_dir:
        trace_name = os.path.join(tmp_dir, "tmp_trace.py")
        fwd_trace.save_trace(trace_name)

        with open(trace_name) as f:
            trace_contents = f.readlines()

        # Verify we find a few expected things in the
        # saved trace.
        trace_contents = "".join(trace_contents)
        assert ".add" in trace_contents
        assert "@torch.no_grad" in trace_contents


def test_unpack_sequence_element_info():
    def fn(x):
        return x.sin().cos()

    jfn = thunder.jit(fn)
    jfn(torch.randn(3, requires_grad=True))

    backward_trc = thunder.last_backward_traces(jfn)[-1]
    for bsym in backward_trc.bound_symbols:
        if bsym.sym.id == thunder.prims.PrimIDs.UNPACK_SEQUENCE and any(
            isinstance(out, thunder.core.proxies.TensorProxy) for out in bsym.flat_outs
        ):  # prims is unpack_sequence and any output is TensorProxy
            # Verify that we print information about the unpacked TensorProxy.
            assert "cpu f32[3]" in str(bsym)


@pytest.mark.parametrize("thunderfx_disable_split_autograd", (True, False))
def test_apply_autograd_memory(thunderfx_disable_split_autograd):
    from thunder.executors.torch_autograd import connect_to_autograd

    def foo():
        def backward(*args):
            return None

        x = torch.randn(2, 2, requires_grad=True)
        o = x.sum()

        connect_to_autograd(
            backward_fn=backward,
            flat_args=(x,),
            flat_output=(o,),
            saved_tensors=(o,),
            saved_other=(),
            return_none_instead_of_grads=True,
            disable_split_autograd=thunderfx_disable_split_autograd,
            is_differentiable_outputs=None,
        )
        return [weakref.ref(x), weakref.ref(o)]

    assert not any(wr() for wr in foo())


def test_thunder_jit_parts():
    m = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
    )

    inp = torch.randn((32, 64))

    jm = thunder.jit(m)

    ce, pro_to_comp, pro_to_epi = run_prologue(jm, inp)
    ce2, pro_to_comp2, pro_to_epi2 = thunder.compile_data(jm).get_computation_and_inputs(inp)

    def clean(tr):
        new_tr = thunder.core.trace.from_trace(tr)
        new_tr.bound_symbols = thunder.core.transform_common.canonicalize_proxies(tr.bound_symbols)
        res = str(new_tr)
        # some traces report timings, we don't want those to influcence
        res = re.sub(r"took \d+ ", "", res)
        return res

    assert clean(ce.prologue_traces[-1]) == clean(ce2.prologue_traces[-1])
    assert clean(ce.computation_traces[-1]) == clean(ce2.computation_traces[-1])
    assert clean(ce.epilogue_traces[-1]) == clean(ce2.epilogue_traces[-1])

    assert_close(pro_to_comp, pro_to_comp2)
    assert_close(pro_to_epi, pro_to_epi2)


def test_prims_pack_list():
    def foo():
        pass

    trace = TraceCtx(foo)

    a = torch.randn(2, 2)
    b = torch.randn(2, 2)

    with tracectx(trace):
        x = prims.unpack_trivial(a, name="x")
        y = prims.unpack_trivial(b, name="y")
        l = prims.pack_list(x, y)
        prims.python_return(l)

    func = trace.python_callable()
    actual = func()
    expected = [a, b]

    assert isinstance(actual, list) and actual == expected


def test_enum_printing():
    from enum import Enum

    def fn():
        pass

    class A(Enum):
        VALUE = 1

    trc = thunder.TraceCtx(fn)
    with thunder.core.trace.tracectx(trc):
        thunder.core.prims.python_return(A.VALUE)

    # the important bit here is that A_VALUE is there, so we can see
    # the enum constant's value
    assert "return _A_VALUE" in str(trc)
