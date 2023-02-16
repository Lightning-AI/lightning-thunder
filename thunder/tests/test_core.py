import operator
from functools import partial, reduce
from itertools import product

import pytest
import torch
from looseversion import LooseVersion
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.dtypes as datatypes
import thunder.core.lang as tlang
import thunder.core.proxies as proxies
import thunder.langs.torch as ttorch
from thunder.tests.framework import Executor, executors, NOTHING, nvFuser, requiresCUDA, TorchEx


@executors(dtypes=NOTHING)
def test_detached_trace(executor, device, _):
    # This test ensures that the detached_trace context manager works as expected.
    #   It should be possible to enter a detached trace, and then exit it, and
    #   the trace should be restored to its original state.
    from thunder.core.trace import get_trace, new_trace, reset_trace, detached_trace

    try:
        trace_token = new_trace()
        outer_trace = get_trace()
        assert outer_trace is not None
        assert outer_trace is trace_token.var.get()
        with detached_trace():
            assert get_trace() is not None
            assert get_trace() is not outer_trace
    finally:
        reset_trace(trace_token)


@executors(dtypes=(thunder.float32,))
def test_symbol_all_constant_args(executor, device, dtype):
    def foo():
        return tlang.maybe_convert_to_dtype(1, dtype)

    trace = thunder.make_trace(foo, executor=executor)()

    assert len(trace.symbols) == 1
    symbol = trace.symbols[0]
    assert symbol.name == "convert_element_type"
    assert symbol.are_all_args_constant

    def bar(a, b):
        return tlang.add(a, b)

    trace = thunder.make_trace(bar, executor=executor)(1, 2)
    assert len(trace.symbols) == 1
    symbol = trace.symbols[0]
    assert symbol.name == "add"
    assert not symbol.are_all_args_constant


@executors(dtypes=(thunder.float32,))
def test_integer_isinstance_mimicry(executor, device, dtype):
    # isinstance() works as expected
    def foo(a, b, c):
        if isinstance(a, int):
            return tlang.add(a, b)

        return tlang.add(b, c)

    traced_foo = thunder.make_traced(foo, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)
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
            return tlang.add(a, b)

        return tlang.add(b, c)

    traced_bar = thunder.make_traced(bar, executor=executor)

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


@executors(dtypes=NOTHING)
def test_nested_make_trace(executor, device, _):
    # This test ensures that make_trace() can be called from within a traced
    # function without leaking the trace context.
    from thunder import _get_executor

    def foo(a, b):
        return tlang.add(a, b)

    def bar(a, b):
        foo_trace = thunder.make_trace(foo, executor=executor)(a, b)
        assert len(foo_trace.symbols) == 1
        assert foo_trace.symbols[0].name == "add"
        return tlang.mul(a, b)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    bar_trace = thunder.make_trace(bar, executor=executor)(a, b)
    assert len(bar_trace.symbols) == 1
    assert bar_trace.symbols[0].name == "mul"

    ex = _get_executor(executor)
    fusion = ex.fuse(bar_trace)
    actual = fusion(a, b)
    expected = a * b
    assert_close(actual, expected)


@executors(dtypes=NOTHING)
def test_nested_make_trace_no_name_collision(executor, device, _):
    def foo(a, b):
        return tlang.add(a, b)

    def bar(*args):
        foo_trace = thunder.make_trace(foo, executor=executor)(*args)
        # The name of the output of the add symbol should not be the same as
        # the name of the first argument to the bar function.
        assert foo_trace.symbols[0].outputs[0].name != foo_trace.args[0].name
        return foo(*args)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    thunder.make_trace(bar, executor=executor)(a, b)


@executors(dtypes=NOTHING)
def test_eval_trace(executor, device, _):
    # This test ensures that eval_trace() can be called from within a traced
    # region and all the symbols in the trace are properly evaluated.
    from thunder.core.transforms import eval_trace
    from thunder.core.trace import new_trace, reset_trace
    from thunder.core.proxies import TensorProxy

    def foo(a, b, *, c=5):
        return tlang.mul(tlang.add(a, b), c)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    c = 4.0

    # Test eval_trace() with eager proxy execution
    foo_trace = thunder.make_trace(foo, executor=executor)(a, b, c=c)
    try:
        trace_token = new_trace()
        new_args = [arg.proxy for arg in foo_trace.args]
        new_kwargs = {k: v.proxy for k, v in foo_trace.kwargs.items()}
        actual = eval_trace(foo_trace, *new_args, **new_kwargs)
        assert isinstance(actual, TensorProxy)
        assert actual.shape == foo_trace.outputs.proxy.shape
        assert actual.dtype == foo_trace.outputs.proxy.dtype
        assert actual.device == foo_trace.outputs.proxy.device
        assert ord(actual.name[-1]) - ord(foo_trace.outputs.proxy.name[-1]) == len(foo_trace.names)
    finally:
        reset_trace(trace_token)

    # Test eval_trace() with retracing + fusion + execution
    def eval_trace_as_function(trace):
        def func(*args, **kwargs):
            return eval_trace(trace, *args, **kwargs)

        return func

    foo_traced = thunder.make_traced(eval_trace_as_function(foo_trace), executor=executor)
    actual = foo_traced(a, b, c=c)
    expected = (a + b) * c
    assert_close(actual, expected)

    # Test eval_trace() with retracing
    foo_trace2 = thunder.make_trace(eval_trace_as_function(foo_trace), executor=executor)(a, b, c=c)
    # How to test that two traces are equal?
    assert len(foo_trace2.symbols) == 2
    assert foo_trace2.symbols[0].name == "add"
    assert foo_trace2.symbols[1].name == "mul"


@executors(
    dtypes=NOTHING,
    executors=[
        TorchEx(),
    ],
)
def test_transforms_identity(executor, device, _):
    # This test ensures that identity() can be called from within a traced
    # function without leaking the trace context.
    # Also tests that identity() can be nested.
    # Also tests that identity() can be used with "torch" executor.
    from thunder.core.transforms import identity, Transforms
    from thunder import _get_executor

    def func(a, b, *, c=5):
        return tlang.mul(tlang.mul(tlang.add(a, b), 1), c)

    nested_id_func = identity(identity(identity(func)))

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    c = 4.0

    nested_id_trace = thunder.make_trace(nested_id_func, executor=executor)(a, b, c=c)
    assert len(nested_id_trace.symbols) == 1
    assert nested_id_trace.symbols[0].op == Transforms.IdentityOp

    trace = nested_id_trace.symbols[0].kwargs.get("trace", None)
    for _ in range(2):
        assert len(trace.symbols) == 1
        assert trace.symbols[0].op == Transforms.IdentityOp
        trace = trace.symbols[0].kwargs.get("trace", None)
    assert len(trace.symbols) == 4
    assert trace.symbols[0].name == "add"
    assert trace.symbols[1].name == "convert_element_type"
    assert trace.symbols[2].name == "mul"
    assert trace.symbols[3].name == "mul"

    ex = _get_executor(executor)
    fusion = ex.fuse(nested_id_trace)
    actual = fusion(a, b, c=c)
    expected = thunder.make_traced(func, executor=executor)(a, b, c=c)
    torch.testing.assert_close(actual, expected)


@executors(
    dtypes=NOTHING,
    executors=[
        TorchEx(),
    ],
)
def test_transforms_inline(executor, device, _):
    # This test ensures that inline() can be called from within a traced
    # function removing (inlining) all identity() transforms.
    # Also tests that inline() can be nested.
    # Also tests that inline() can be used with "torch" executor.
    from thunder.core.transforms import identity, inline, Transforms

    def func(a, b):
        return tlang.mul(tlang.add(a, b), 1)

    nested_id_func = identity(identity(identity(func)))

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    inlined_nested_id_trace = thunder.make_trace(inline(nested_id_func), executor=executor)(a, b)
    assert len(inlined_nested_id_trace.symbols) == 3
    assert not any(symbol.op == Transforms.IdentityOp for symbol in inlined_nested_id_trace.symbols)
    assert inlined_nested_id_trace.symbols[0].name == "add"
    assert inlined_nested_id_trace.symbols[1].name == "convert_element_type"
    assert inlined_nested_id_trace.symbols[2].name == "mul"

    transforms = (inline, identity, inline, inline, identity, identity, inline)
    for transform in transforms:
        transformed_func = transform(func)

    # Since the outer-most transform is inline, the trace should not contain
    # any identity transforms.
    transformed_trace = thunder.make_trace(transformed_func, executor=executor)(a, b)
    assert len(transformed_trace.symbols) == 3
    assert not any(symbol.op == Transforms.IdentityOp for symbol in transformed_trace.symbols)


@executors(
    dtypes=NOTHING,
)
def test_transforms_vmap_identity(executor, device, _):
    pytest.skip("Skipped temporarily until we have a fix")
    from thunder.core.transforms import identity, vmap

    def func(a):
        return tlang.sin(a)

    a = torch.randn(2, 2)

    thunder.make_trace(vmap(identity(func)), executor="torch")(a)


@executors(
    dtypes=NOTHING,
)
def test_transforms_jvp_eager(executor, device, _):
    from thunder.core.transforms import jvp_eager

    def func(a, b):
        c = tlang.sin(a)
        return tlang.mul(tlang.add(c, b), 1)

    a = torch.ones(2, 3, device=device, dtype=torch.float32)
    b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

    primals = (a, b)
    tangents = (a, b)
    out_p, out_t = jvp_eager(func, primals, tangents, executor=executor)
    expected_out_p = torch.sin(a) + b
    expected_out_t = torch.cos(a) + b
    assert_close(out_p, expected_out_p)
    assert_close(out_t, expected_out_t)


@executors(
    dtypes=NOTHING,
)
def test_transforms_vmap_x(executor, device, _):
    from thunder.core.transforms import vmap_eager

    def func(a, b):
        assert isinstance(a, proxies.TensorProxy)
        assert isinstance(b, proxies.TensorProxy)
        assert a.ndim == 1
        assert a.shape == b.shape
        c = tlang.sin(a)
        return tlang.mul(tlang.add(c, b), 1)

    a = torch.ones(2, 3, device=device, dtype=torch.float32)
    b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

    args = (a, b)
    out = vmap_eager(func, args, executor=executor)
    expected_out_p = torch.sin(a) + b
    assert_close(out, expected_out_p)


@executors(
    dtypes=NOTHING,
)
def test_transforms_jvp_vmap(executor, device, _):
    from thunder.core.transforms import vmap, jvp, inline

    def func(a, b):
        assert isinstance(a, proxies.TensorProxy)
        assert isinstance(b, proxies.TensorProxy)
        assert a.ndim == 1
        assert a.shape == b.shape
        c = tlang.sin(a)
        return tlang.mul(tlang.add(c, b), 1)

    a = torch.ones(2, 3, device=device, dtype=torch.float32)
    b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

    args = (a, b)
    out_p, out_t = thunder.make_traced(inline(jvp(inline(vmap(func)))), executor="torch")(args, args)
    expected_out_p = torch.sin(a) + b
    assert_close(out_p, expected_out_p)
    expected_out_t = torch.cos(a) + b
    assert_close(out_t, expected_out_t)


@executors(
    dtypes=NOTHING,
)
def test_transforms_vmap_inline_jvp(executor, device, _):
    from thunder.core.transforms import vmap, jvp, inline

    def func(a, b):
        assert isinstance(a, proxies.TensorProxy)
        assert isinstance(b, proxies.TensorProxy)
        assert a.ndim == 1
        assert a.shape == b.shape
        c = tlang.sin(a)
        return tlang.mul(tlang.add(c, b), 1)

    a = torch.ones(2, 3, device=device, dtype=torch.float32)
    b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

    args = (a, b)
    # vmap(inline(jvp)) works! But vmap(jvp) does not.
    out_p, out_t = thunder.make_traced(inline(vmap(inline(jvp(func)), out_dims=(0, 0))), executor="torch")(args, args)
    expected_out_p = torch.sin(a) + b
    assert_close(out_p, expected_out_p)
    expected_out_t = torch.cos(a) + b
    assert_close(out_t, expected_out_t)


@executors(
    dtypes=NOTHING,
)
def test_transforms_jvp(executor, device, _):
    from thunder.core.transforms import jvp, inline, identity

    def func(a, b):
        c = tlang.sin(a)
        return tlang.mul(tlang.add(c, b), 1)

    a = torch.ones(2, 3, device=device, dtype=torch.float32)
    b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

    primals = (a, b)
    tangents = (a, b)
    out_p, out_t = thunder.make_traced(inline(identity(jvp(identity(func)))), executor=executor)(primals, tangents)
    expected_out_p = torch.sin(a) + b
    expected_out_t = torch.cos(a) + b
    assert_close(out_p, expected_out_p)
    assert_close(out_t, expected_out_t)


@executors(
    dtypes=NOTHING,
)
def test_transforms_jvp_python_number(executor, device, _):
    from thunder.core.transforms import jvp, inline

    scalars = (
        2,
        2.0,
        True,
    )
    for scalar in scalars:

        def func(a):
            return tlang.mul(a, scalar)

        a = make_tensor((2, 3), device=device, dtype=torch.float32)

        primals = (a,)
        tangents = (a,)
        out_p, out_t = thunder.make_traced(inline(jvp(func)), executor=executor)(primals, tangents)

        expected_out_p = a * scalar
        expected_out_t = a * scalar
        assert_close(out_p, expected_out_p)
        assert_close(out_t, expected_out_t)


@executors(
    dtypes=NOTHING,
    executors=[
        TorchEx(),
    ],
)
def test_get_executor(executor, device, _):
    from thunder import _get_executor
    from thunder.executors.torch import torchCtx

    with pytest.raises(ValueError, match="No executor specified!"):
        _get_executor(None)

    ex = _get_executor(executor)
    if executor.name == "TorchEx":
        assert isinstance(ex, torchCtx)


# TODO: subsume this by test_elementwise when sample inputs are expanded to include more numbers
@executors(dtypes=NOTHING)
def test_integer_return(executor, device, _):
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = thunder.make_traced(foo, executor=executor)

    thunder_result = traced_foo(3, 4)
    python_result = 3 + 4
    assert_close(thunder_result, python_result)


# TODO: this test just spot-checks type promotion -- it could probably be better
@executors(dtypes=NOTHING)
def test_type_promotion_tensors(executor, device, _):
    def foo(a, b):
        return a + b

    traced_foo = thunder.make_traced(foo, executor=executor)

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

    traced_bar = thunder.make_traced(bar, executor=executor)

    # float x int64 x float16 type promotion -- float16 result dtype
    result = traced_bar(2.0, i64, f16)
    assert result.dtype is torch.float16

    # float x int x int64 -- float32 result dtype
    result = traced_bar(2.1, -1, i64)
    assert result.dtype is torch.float32


@executors(dtypes=NOTHING)
def test_type_promotion_numbers_and_tensors(executor, device, _):
    def foo(a, b, c):
        return a + b + c

    traced_foo = thunder.make_traced(foo, executor=executor)

    f16 = make_tensor((2, 2), device=device, dtype=torch.float16)
    f32 = make_tensor((2, 2), device=device, dtype=torch.float32)
    i64 = make_tensor((2, 2), device=device, dtype=torch.int64)

    result = traced_foo(5, f32, 2)
    assert result.dtype is torch.float32

    result = traced_foo(f32, 1, f32)
    assert result.dtype is torch.float32

    result = traced_foo(i64, 3.0, f16)
    assert result.dtype is torch.float16

    result = traced_foo(i64, 3.0, i64)
    assert result.dtype is torch.float32


@executors(dtypes=NOTHING)
def test_int_to_float_type_promotion(executor, device, _):
    def foo(a, b):
        return a / b

    traced_foo = thunder.make_traced(foo, executor=executor)

    i64 = make_tensor((2, 2), device=device, dtype=torch.int64)
    f16 = make_tensor((2, 2), device=device, dtype=torch.float16)

    # int64 x int64 -- float32 result dtype
    result = traced_foo(i64, i64)
    assert result.dtype is torch.float32

    # int x int64 -- float32 result dtype
    result = traced_foo(2, i64)
    assert result.dtype is torch.float32

    # int64 x bool -- float32 result dtype
    result = traced_foo(i64, True)
    assert result.dtype is torch.float32

    # int64 x float16 -- float16 result dtype
    result = traced_foo(i64, f16)
    assert result.dtype is torch.float16


# TODO: put this in test_tensor_creation.py
# TODO: specify multiple specific devices (today the test suite just passes a devicetype)
# TODO: add test for full (), which will cause a segfault
@executors(dtypes=(thunder.float32,))
def test_full(executor, device, dtype):
    traced_full = thunder.make_traced(tlang.full, executor=executor)

    tdtype = ttorch.torch_dtype(dtype)

    thunder_result = traced_full((1, 2, 3), 1.0, device=device, dtype=tdtype)
    torch_result = torch.full((1, 2, 3), 1.0, device=device, dtype=tdtype)

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
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
        i = ka + ka  # NOTE: not returned (ignored computation)
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

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)
    c = make_tensor((2, 2), device=device, dtype=tdtype)

    args = ({"a": {"a": a}}, (b, c), (3, {"c": c}))
    kwargs = {"ka": b, "kb": 3.0, "kc": (a, 2)}
    thunder_result = traced_foo(*args, **kwargs)
    torch_result = foo(*args, **kwargs)

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_varargs_and_kwargs(executor, device, dtype):
    def foo(a, b, *posargs, e, **kwargs):
        accum = a
        for x in posargs:
            accum = a + x

        d = b + e + kwargs["f"]

        return accum, d, kwargs["g"]

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

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


# TODO: write these tests
@executors(dtypes=(thunder.float32,))
def test_varargs(executor, device, dtype):
    def foo(*args):
        return reduce(operator.add, args)

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    packed = (a, a, a, a, a)

    thunder_result = traced_foo(*packed)
    torch_result = foo(*packed)

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_kwargs(executor, device, dtype):
    def foo(**kwargs):
        return kwargs["a"] + kwargs["b"]

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2,), device=device, dtype=tdtype)

    thunder_result = traced_foo(a=a, b=b)
    torch_result = foo(a=a, b=b)

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_no_return(executor, device, dtype):
    def foo(a, b):
        c = a + b
        pass

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b=b)
    torch_result = foo(a, b)

    assert_close(thunder_result, torch_result)


@executors(dtypes=NOTHING)
def test_no_input(executor, device, dtype):
    def foo():
        return 3, ()

    traced_foo = thunder.make_traced(foo, executor=executor)

    thunder_result = traced_foo()
    torch_result = foo()

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_no_compute(executor, device, dtype):
    def foo(a, b):
        return a, 3.0

    traced_foo = thunder.make_traced(foo, executor=executor)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b=b)
    torch_result = foo(a, b)

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_fusion_reuse(executor, device, dtype):
    def foo(a, b, *, flag=False):
        if flag:
            return a + b
        return a - b

    traced_foo = thunder.make_traced(foo, executor=executor, _return_fusion=True)
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    args = (a,)
    kwargs = {"b": b, "flag": True}

    thunder_result = traced_foo(*args, **kwargs)

    torch_result = foo(*args, **kwargs)
    assert_close(thunder_result["result"], torch_result)

    fusion_result = thunder_result["fusion"](*args, **kwargs)
    assert_close(fusion_result, torch_result)

    # Calls the fusion with new tensor data (but preserves the flag arg)
    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    args = (a,)
    kwargs = {"b": b, "flag": True}

    fusion_result = thunder_result["fusion"](*args, **kwargs)
    torch_result = foo(*args, **kwargs)
    assert_close(fusion_result, torch_result)

    # Calls the fusion with new tensor data, and verifies the flag arg is ignored
    a = make_tensor((2,), device=device, dtype=tdtype)
    b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

    args = (a,)
    kwargs = {"b": b, "flag": False}

    fusion_result = thunder_result["fusion"](*args, **kwargs)
    torch_result = foo(*args, b=b, flag=True)
    assert_close(fusion_result, torch_result)

    # Tests with PyTorch fallback
    def bar(a, b):
        c = a @ b
        return c + c

    traced_bar = thunder.make_traced(bar, executor=executor, _return_fusion=True)

    a = make_tensor((4, 16), device=device, dtype=tdtype)
    b = make_tensor((16, 8), device=device, dtype=tdtype)

    thunder_result = traced_bar(a, b)
    torch_result = bar(a, b)
    assert_close(torch_result, thunder_result["result"])

    fusion_result = thunder_result["fusion"](a, b)
    assert_close(torch_result, fusion_result)


# TODO: probably only want to run this on nvFuser
# TODO: maybe move to special test_nvfuser?
@executors(
    dtypes=(thunder.float32,),
)
@requiresCUDA
def test_hybrid_execution(executor, device, dtype):
    def foo(a, b, bias=None):
        c = a + b
        d = c + a
        e = d + 2
        f = ttorch.linear(a, c, bias)
        g = c + f
        return (c, e, 2), 5, f, g

    def bar(a, b, bias=None):
        c = a + b
        d = c + a
        e = d + 2
        f = torch.nn.functional.linear(a, c, bias)
        g = c + f
        return (c, e, 2), 5, f, g

    traced_foo = thunder.make_traced(foo, executor="nvfuser")
    tdtype = ttorch.torch_dtype(dtype)

    a = make_tensor((2, 2), device="cuda", dtype=torch.float32)
    b = make_tensor((2, 2), device="cuda", dtype=torch.float32)
    bias = None

    result = traced_foo(a, b, bias)
    torch_result = bar(a, b, bias)

    assert_close(result, torch_result)


@executors(dtypes=NOTHING)
def test_dtype_conversion(executor: Executor, device, dtype):
    if isinstance(executor, nvFuser) and LooseVersion(executor.version()) < "0":
        pytest.xfail("https://github.com/csarofeen/pytorch/issues/2370")

    # FIXME
    if isinstance(executor, nvFuser) and device == "cuda" and dtype is None:
        pytest.skip("RuntimeError: Illegal Cast value from  DataType: double to DataType: __bfloat")

    make = partial(make_tensor, (2, 2), device=device)

    def foo(a, dtype):
        return tlang.maybe_convert_to_dtype(a, dtype)

    thunder_fn = thunder.make_traced(foo, executor=executor)

    strong_dtypes = set(datatypes.strong_dtypes)
    supported_dtypes = set(datatypes.resolve_dtypes(executor.supported_dtypes))
    dtypes = strong_dtypes.intersection(supported_dtypes)
    for a, b in product(dtypes, dtypes):
        a = ttorch.torch_dtype(a)
        b = ttorch.torch_dtype(b)
        t = make(dtype=a)
        thunder_result = thunder_fn(t, b)
        torch_result = t.to(b)
        assert_close(thunder_result, torch_result)
