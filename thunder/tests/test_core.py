import operator
from functools import partial, reduce
from itertools import product

import pytest
import torch
from looseversion import LooseVersion
from torch.testing import assert_close, make_tensor

import thunder
import thunder.clang as clang
import thunder.core.proxies as proxies
import thunder.torch as ltorch
import thunder.core.codeutils as codeutils
from thunder.core.pytree import tree_flatten_only, tree_unflatten
from thunder.tests.framework import instantiate, NOTHING, TorchExecutor, nvFuserExecutor
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
import thunder.core.prims as prims

#
# Tests related to running valid Python programs
#


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
    assert y is dev


@instantiate(dtypes=(thunder.float32,))
def test_partial(executor, device, dtype):
    def foo(a, *, b, c=2):
        return a, b, c

    pfoo = partial(foo, b=3, c=4)
    cpfoo = executor.make_callable(pfoo)

    lc_result = cpfoo(1)
    py_result = pfoo(1)

    assert_close(lc_result, py_result)

    ppfoo = partial(pfoo, b=2, c=8)
    cppfoo = executor.make_callable(ppfoo)

    lc_result = cppfoo(1)
    py_result = ppfoo(1)

    assert_close(lc_result, py_result)


@instantiate(dtypes=(thunder.float32,))
def test_constant_creation(executor, device, dtype):
    def foo(a):
        x = prims.convert_element_type(1, float)
        return a + x

    cfoo = thunder.compile_with_info(foo, executors_list=executor.executors_list())

    torch_dtype = ltorch.to_torch_dtype(dtype)
    a = make_tensor((2, 2), device=device, dtype=torch_dtype)

    lc_result, traces = cfoo(a)
    python_result = foo(a)

    assert_close(lc_result, python_result)

    for trace in traces:
        fn = trace.python_callable()
        lc_result = fn(a)
        assert_close(lc_result, python_result)


@instantiate(dtypes=NOTHING)
def test_detached_trace(executor, device: str, _):
    # This test ensures that the detached_trace context manager works as expected.
    #   It should be possible to enter a detached trace, and then exit it, and
    #   the trace should be restored to its original state.
    from thunder.core.trace import set_tracectx, get_tracectx, TraceCtx, reset_tracectx, detached_trace

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


@instantiate(dtypes=(thunder.float32,))
def test_symbol_all_constant_args(executor, device: str, dtype: dtypes.dtype):
    def foo():
        return clang.maybe_convert_to_dtype(1, dtype)

    trace = thunder._make_trace(foo)()

    assert len(trace.bound_symbols) == 1
    symbol = trace.bound_symbols[0]
    assert symbol.sym.name == "convert_element_type"
    assert symbol.are_all_args_constant

    def bar(a, b):
        return clang.add(a, b)

    trace = thunder._make_trace(bar)(1, 2)
    # Trace consists of two trivial unpack and addition
    assert len(trace.bound_symbols) == 3
    symbol = trace.bound_symbols[-1]
    assert symbol.sym.name == "add"
    assert not symbol.are_all_args_constant


# This test ensures that calls to torch functions are recorded in the trace
@instantiate(executors=(TorchExecutor,), dtypes=NOTHING)
def test_torch_call_recording(executor, device: str, _):
    def func(a):
        return ltorch.dropout(a)

    a = make_tensor((2, 3), device=device, dtype=torch.float32)

    torch_trace = thunder._make_trace(func)(a)
    assert len(torch_trace.bound_symbols) == 2
    assert torch_trace.bound_symbols[-1].sym.name == "dropout"
    assert torch_trace.bound_symbols[-1].sym.id == "torch.nn.functional.dropout"

    # Ensure that the trace can be fused and executed
    # TODO: Restore this
    # ex = _get_executor(executor)
    # fusion = ex.fuse(torch_trace)
    # actual = fusion(a)
    # assert actual.shape == (2, 3)


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
#     ex = _get_executor(executor)
#     fusion = ex.fuse(nvfuser_trace)
#     actual = fusion(a)
#     expected = thunder.make_traced(func, executor=executor)(a)
#     assert_close(actual, expected)


@instantiate(dtypes=NOTHING)
def test_nested_make_trace(executor, device, _):
    # This test ensures that make_trace() can be called from within a traced
    # function without leaking the trace context.
    # from thunder import _get_executor

    def foo(a, b):
        return clang.add(a, b)

    def bar(a, b):
        foo_trace = thunder._make_trace(foo)(a, b)
        assert len(foo_trace.bound_symbols) == 3
        assert foo_trace.bound_symbols[-1].sym.name == "add"
        return clang.mul(a, b)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    bar_trace = thunder._make_trace(bar)(a, b)
    assert len(bar_trace.bound_symbols) == 3
    assert bar_trace.bound_symbols[-1].sym.name == "mul"

    # TODO: Restore this once there's an equivalent
    # ex = _get_executor(executor)
    # fusion = ex.fuse(bar_trace)
    # actual = fusion(a, b)
    # expected = a * b
    # assert_close(actual, expected)


@instantiate(dtypes=NOTHING)
def test_nested_make_trace_no_name_collision(executor, device, _):
    def foo(a, b):
        return clang.add(a, b)

    def bar(a, b):
        foo_trace = thunder._make_trace(foo)(a, b)
        # The name of the output of the add symbol should not be the same as
        # the name of the first argument to the bar function.
        assert foo_trace.bound_symbols[0].output[0].name != foo_trace.args[0].name
        return foo(a, b)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    thunder._make_trace(bar)(a, b)


@instantiate(dtypes=NOTHING)
def test_eval_trace(executor, device, _):
    # This test ensures that eval_trace() can be called from within a trace
    #   and that all the symbols in the trace are properly evaluated.

    from thunder.core.transforms import eval_trace
    from thunder.core.trace import TraceCtx, reset_tracectx, set_tracectx, maybe_start_trace
    from thunder.core.proxies import TensorProxy

    def foo(a, b, *, c=5):
        return clang.mul(clang.add(a, b), c)

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    c = 4.0

    # Test eval_trace() with eager proxy execution
    foo_trace = thunder._make_trace(foo)(a, b, c=c)
    try:
        trace = TraceCtx(None)
        trace_token = set_tracectx(trace)
        new_args = [arg for arg in foo_trace.args]
        new_kwargs = {k: v for k, v in foo_trace.kwargs.items()}
        # TODO: trace object doesn't respect the original tuple/non-tuple spec
        # for output, it's always a tuple
        actual = eval_trace(foo_trace, *new_args, **new_kwargs)[0]
        assert isinstance(actual, TensorProxy)
        assert actual.shape == foo_trace.output[0].shape
        assert actual.dtype == foo_trace.output[0].dtype
        assert actual.device == foo_trace.output[0].device
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
    foo_trace2 = thunder._make_trace(eval_trace_as_function(foo_trace))(a, b, c=c)
    # How to test that two traces are equal?
    # Two operators and others are do-nothing annotations
    assert len(foo_trace2.bound_symbols) == 7
    assert foo_trace2.bound_symbols[-2].sym.name == "add"
    assert foo_trace2.bound_symbols[-1].sym.name == "mul"


@instantiate(
    dtypes=NOTHING,
    executors=[
        TorchExecutor,
        # TODO: nvFuser executor doesn't support duplicate outputs
        # TODO: nvFuser executor doesn't support clashing input and output names
    ],
)
def test_eval_trace_duplicate_output(executor, device, _):
    pytest.xfail(
        "NotImplementedError: An existing proxy __a is being passed as an input, but its name is not the same name (a) as the unpack is requesting"
    )

    # This test ensures that eval_trace() can evaluate a trace with duplicate
    # outputs.
    from thunder.core.transforms import eval_trace, identity

    def foo1(a):
        return a, a

    a = torch.ones((2, 2), device=device, dtype=torch.float32)

    foo_trace = thunder._make_trace(foo1)(a)
    assert len(foo_trace.bound_symbols) == 1
    assert foo_trace.bound_symbols[0].sym.name == "unpack_trivial"
    assert len(foo_trace.output) == 2
    assert foo_trace.output[0].name == foo_trace.output[1].name

    def func(a):
        return eval_trace(foo_trace, a)

    actual = executor.make_callable(func)(a)
    assert_close(actual, (a, a))

    # Identity is needed to ensure that the duplicated outputs of a symbol
    # don't cause problems.
    def foo2(a):
        a = 1.0 * a
        return a, a

    for foo in [foo1, foo2]:
        foo_trace = thunder._make_trace(identity(foo))(a)
        assert len(foo_trace.bound_symbols) == 2
        assert len(foo_trace.output) == 2
        assert foo_trace.output[0].name == foo_trace.output[1].name

    # TODO: enable this once executors can work with identity call
    #     actual = executor.make_callable(partial(eval_trace, foo_trace))(a)
    #     assert_close(actual, (a, a))


@instantiate(
    dtypes=NOTHING,
    executors=[
        TorchExecutor,
    ],
)
def test_transforms_identity(executor, device, _):
    pytest.xfail(
        "NotImplementedError: An existing proxy __a is being passed as an input, but its name is not the same name (a) as the unpack is requesting"
    )

    # This test ensures that identity() can be called from within a traced
    # function without leaking the trace context.
    # Also tests that identity() can be nested.
    # Also tests that identity() can be used with "torch" executor.
    from thunder.core.transforms import identity, Transforms

    # from thunder import _get_executor

    def func(a, b, *, c=5):
        return clang.mul(clang.mul(clang.add(a, b), 1), c)

    nested_id_func = identity(identity(identity(func)))

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)
    c = 4.0

    nested_id_trace = thunder._make_trace(nested_id_func)(a, b, c=c)
    # one annotating symbol per input and one actual symbol
    assert len(nested_id_trace.bound_symbols) == 3
    assert nested_id_trace.bound_symbols[-1].sym.id == Transforms.IdentityOp

    trace = nested_id_trace.bound_symbols[-1].kwargs.get("trace", None)
    for _ in range(2):
        assert len(trace.bound_symbols) == 3
        assert trace.bound_symbols[-1].sym.id == Transforms.IdentityOp
        trace = trace.bound_symbols[-1].kwargs.get("trace", None)
    assert len(trace.bound_symbols) == 7
    assert trace.bound_symbols[-4].sym.name == "add"
    assert trace.bound_symbols[-3].sym.name == "convert_element_type"
    assert trace.bound_symbols[-2].sym.name == "mul"
    assert trace.bound_symbols[-1].sym.name == "mul"

    # TODO: Restore this once there's an equivalent
    # ex = _get_executor(executor)
    # fusion = ex.fuse(nested_id_trace)
    # actual = fusion(a, b, c=c)
    # expected = executor.make_callable(func)(a, b, c=c)
    # torch.testing.assert_close(actual, expected)


@instantiate(
    dtypes=NOTHING,
    executors=[
        TorchExecutor,
    ],
)
def test_transforms_inline(executor, device, _):
    pytest.xfail(
        "NotImplementedError: An existing proxy __a is being passed as an input, but its name is not the same name (a) as the unpack is requesting"
    )
    # This test ensures that inline() can be called from within a traced
    # function removing (inlining) all identity() transforms.
    # Also tests that inline() can be nested.
    # Also tests that inline() can be used with "torch" executor.
    from thunder.core.transforms import identity, inline, Transforms

    def func(a, b):
        return clang.mul(clang.add(a, b), 1)

    nested_id_func = identity(identity(identity(func)))

    a = make_tensor((2, 2), device=device, dtype=torch.float32)
    b = make_tensor((2, 2), device=device, dtype=torch.float32)

    inlined_nested_id_trace = thunder._make_trace(inline(nested_id_func))(a, b)
    # TODO: Something fishy is going on here, "unpacking" is repeated several times
    assert len(inlined_nested_id_trace.bound_symbols) == 9
    assert not any(symbol.sym.id == Transforms.IdentityOp for symbol in inlined_nested_id_trace.bound_symbols)
    assert inlined_nested_id_trace.bound_symbols[-3].sym.name == "add"
    assert inlined_nested_id_trace.bound_symbols[-2].sym.name == "convert_element_type"
    assert inlined_nested_id_trace.bound_symbols[-1].sym.name == "mul"

    transforms = (inline, identity, inline, inline, identity, identity, inline)
    for transform in transforms:
        transformed_func = transform(func)

    # Since the outer-most transform is inline, the trace should not contain
    # any identity transforms.
    transformed_trace = thunder._make_trace(transformed_func)(a, b)
    assert len(transformed_trace.bound_symbols) == 6
    assert not any(symbol.sym.id == Transforms.IdentityOp for symbol in transformed_trace.bound_symbols)


@instantiate(
    dtypes=NOTHING,
    executors=(
        TorchExecutor,
        # TODO: nvFuser executor does not support full(shape=()) yet
    ),
)
def test_transforms_vmap_axis_size(executor, device, _):
    from thunder.core.transforms import inline, vmap

    actual = executor.make_callable(inline(vmap(lambda: 2, axis_size=4)))()
    expected = torch.full((4,), 2, device="cpu")
    assert_close(actual, expected)

    actual = executor.make_callable(inline(vmap(lambda x: x, axis_size=4)))(2)
    assert_close(actual, expected)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_vmap_identity(executor, device, _):
#     pytest.skip("Skipped temporarily until we have a fix")
#     from thunder.core.transforms import identity, vmap

#     def func(a):
#         return tlang.sin(a)

#     a = torch.randn(2, 2)

#     thunder.make_trace(vmap(identity(func)), executor=executor)(a)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_jvp_eager(executor, device, _):
#     from thunder.core.transforms import jvp_eager

#     def func(a, b):
#         c = tlang.sin(a)
#         return tlang.mul(tlang.add(c, b), 1)

#     a = torch.ones(2, 3, device=device, dtype=torch.float32)
#     b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

#     primals = (a, b)
#     tangents = (a, b)
#     out_p, out_t = jvp_eager(func, primals, tangents, executor=executor)
#     expected_out_p = torch.sin(a) + b
#     expected_out_t = torch.cos(a) + b
#     assert_close(out_p, expected_out_p)
#     assert_close(out_t, expected_out_t)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_vjp_1_2(executor, device, _):
#     from thunder.core.transforms import inline, vjp

#     # 1 input, 2 outputs
#     def func_1_2(x):
#         a = tlang.sin(x)
#         b = tlang.add(0.2, a)
#         c = tlang.asin(b)
#         return b, c

#     a = make_tensor((2, 3), device=device, dtype=torch.float32)

#     g1 = make_tensor((2, 3), device=device, dtype=torch.float32)
#     g2 = make_tensor((2, 3), device=device, dtype=torch.float32)

#     vjp_eager = executor.make_callable(inline(vjp(func_1_2)))

#     primals = (a,)
#     cotangents = (g1, g2)
#     out_p, grads = vjp_eager(primals, cotangents)
#     expected_out_p = executor.make_callable(func_1_2)(a)
#     assert_close(out_p, expected_out_p, equal_nan=True)

#     # Now check the gradients
#     # TODO: We will have this automatically tested with OpInfo tests
#     aa = a.clone().requires_grad_(True)

#     def pt_func_1_2(x):
#         a = torch.sin(x)
#         b = torch.add(0.2, a)
#         c = torch.asin(b)
#         return b, c

#     out = pt_func_1_2(aa)
#     expected_grads = torch.autograd.grad(out, aa, grad_outputs=(g1, g2), retain_graph=True)
#     assert_close(expected_grads, grads, equal_nan=True)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_vjp_2_2_kwarg(executor, device, _):
#     # This test ensures that combination of positional and keyword arguments
#     # is differentiable.
#     from thunder.core.transforms import inline, vjp

#     # 2 inputs, 1 kwarg, 2 outputs
#     def func_2_2(x, y, *, z):
#         def func(x):
#             a = tlang.sin(x)
#             b = tlang.add(0.2, a)
#             c = tlang.asin(b)
#             return c

#         a, b = func(x), func(y)
#         c = tlang.add(a, b)
#         d = tlang.add(c, func(z))
#         return c, d

#     x = make_tensor((2, 3), device=device, dtype=torch.float64)
#     y = make_tensor((2, 3), device=device, dtype=torch.float64)
#     z = make_tensor((2, 3), device=device, dtype=torch.float64)

#     g1 = make_tensor((2, 3), device=device, dtype=torch.float64)
#     g2 = make_tensor((2, 3), device=device, dtype=torch.float64)

#     vjp_eager = executor.make_callable(inline(vjp(func_2_2)))

#     primals = (x, y)
#     primal_kwargs = {"z": z}
#     cotangents = (g1, g2)
#     out_p, grads = vjp_eager(primals, cotangents, **primal_kwargs)
#     expected_out_p = executor.make_callable(func_2_2)(*primals, **primal_kwargs)
#     assert_close(out_p, expected_out_p, equal_nan=True)

#     # Now check the gradients
#     # TODO: We will have this automatically tested with OpInfo tests
#     xx = x.clone().requires_grad_(True)
#     yy = y.clone().requires_grad_(True)
#     zz = z.clone().requires_grad_(True)

#     def pt_func_2_2(x, y, *, z):
#         def func(x):
#             a = torch.sin(x)
#             b = torch.add(0.2, a)
#             c = torch.asin(b)
#             return c

#         a, b = func(x), func(y)
#         c = torch.add(a, b)
#         d = torch.add(c, func(z))
#         return c, d

#     out = pt_func_2_2(xx, yy, z=zz)
#     expected_grads = torch.autograd.grad(out, [xx, yy, zz], grad_outputs=(g1, g2), retain_graph=True)
#     # vjp returns a tuple of (primals, cotangents) where cotangents is a tuple of
#     # derivatives with respect to the positional arguments and a dict of derivatives
#     # with respect to the keyword arguments.
#     *gprimals, gkwargs = grads
#     assert_close(expected_grads[:2], gprimals, equal_nan=True)
#     assert_close(expected_grads[2], gkwargs["z"], equal_nan=True)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_vjp_2_1(executor, device, _):
#     from thunder.core.transforms import inline, vjp

#     def pt_func_2_1(x, y):
#         a = torch.sin(x + y)
#         b = torch.add(0.2, a)
#         c = torch.asin(b)
#         return c

#     def func_2_1(x, y):
#         a = tlang.sin(x + y)
#         b = tlang.add(0.2, a)
#         c = tlang.asin(b)
#         return c

#     vjp_eager = executor.make_callable(inline(vjp(func_2_1)))
#     a = make_tensor((2, 3), device=device, dtype=torch.float32)
#     b = make_tensor((2, 3), device=device, dtype=torch.float32)
#     g1 = make_tensor((2, 3), device=device, dtype=torch.float32)
#     primals = (a, b)
#     cotangents = (g1,)
#     out_p, grads = vjp_eager(primals, cotangents)
#     expected_out_p = executor.make_callable(func_2_1)(*primals)
#     assert_close(out_p, expected_out_p, equal_nan=True)

#     aa = a.clone().requires_grad_(True)
#     bb = b.clone().requires_grad_(True)
#     out = pt_func_2_1(aa, bb)
#     expected_grads = torch.autograd.grad(out, [aa, bb], grad_outputs=(g1,), retain_graph=True)
#     assert_close(expected_grads, grads, equal_nan=True)


# @instantiate(
#     dtypes=NOTHING,
#     executors=(
#         TorchEx(),
#         # TODO: enable nvFuser executor
#         # thunder/executors/nvfuser.py:240: AssertionError
#         # assert len(shape) > 0 in _full_preprocessor
#     ),
# )
# def test_transforms_vmap_inline_value_and_grad(executor, device, _):
#     # This test checks whether it's possible to vmap a function that is
#     # traced with inline and value_and_grad.
#     # For applications see
#     # https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#per-example-gradients
#     # https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html
#     from thunder.core.transforms import inline, value_and_grad, vmap
#     from thunder.core import prims

#     def func(x):
#         a = prims.sin(x)
#         a = prims.sum(a, ())
#         return prims.sum(a, tuple(range(a.ndim)))

#     vjp_func = executor.make_callable(value_and_grad(func))
#     a = make_tensor((2, 3), device=device, dtype=torch.float32)
#     single_out, (single_grad,) = vjp_func(a)

#     aaa = torch.stack([a, a, a])
#     vmap_inline_vjp = executor.make_callable(vmap(inline(value_and_grad(func))))
#     batched_out, (batched_grad,) = vmap_inline_vjp(aaa)
#     for i in range(3):
#         assert_close(single_out, batched_out[i])
#         assert_close(single_grad, batched_grad[i])


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_vmap_x(executor, device, _):
#     from thunder.core.transforms import vmap_eager

#     def func(a, b):
#         assert isinstance(a, proxies.TensorProxy)
#         assert isinstance(b, proxies.TensorProxy)
#         assert a.ndim == 1
#         assert a.shape == b.shape
#         c = tlang.sin(a)
#         return tlang.mul(tlang.add(c, b), 1)

#     a = torch.ones(2, 3, device=device, dtype=torch.float32)
#     b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

#     args = (a, b)
#     out = vmap_eager(func, args, executor=executor)
#     expected_out_p = torch.sin(a) + b
#     assert_close(out, expected_out_p)


@instantiate(
    dtypes=NOTHING,
)
def test_transforms_inline_jvp_inline_vmap(executor, device, _):
    pytest.xfail("AttributeError: 'NoneType' object has no attribute 'mul'")
    from thunder.core.transforms import vmap, jvp, inline

    if executor == nvFuserExecutor:
        # Couldn't find metadata for 1.0 of type <class 'float'>
        pytest.xfail("Something is failing with the nvFuser executor")

    def func(a, b):
        assert isinstance(a, proxies.TensorProxy)
        assert isinstance(b, proxies.TensorProxy)
        assert a.ndim == 1
        assert a.shape == b.shape
        c = clang.sin(a)
        return clang.mul(clang.add(c, b), 1)

    a = torch.ones(2, 3, device=device, dtype=torch.float32)
    b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

    args = (a, b)
    out_p, out_t = executor.make_callable(inline(jvp(inline(vmap(func)))))(args, args)
    expected_out_p = torch.sin(a) + b
    assert_close(out_p, expected_out_p)
    expected_out_t = torch.cos(a) + b
    assert_close(out_t, expected_out_t)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_inline_vmap_inline_jvp(executor, device, _):
#     from thunder.core.transforms import vmap, jvp, inline

#     def func(a, b):
#         assert isinstance(a, proxies.TensorProxy)
#         assert isinstance(b, proxies.TensorProxy)
#         assert a.ndim == 1
#         assert a.shape == b.shape
#         c = tlang.sin(a)
#         return tlang.mul(tlang.add(c, b), 1)

#     a = torch.ones(2, 3, device=device, dtype=torch.float32)
#     b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

#     args = (a, b)
#     out_p, out_t = thunder.make_traced(inline(vmap(inline(jvp(func)), out_dims=(0, 0))), executor=executor)(args, args)
#     expected_out_p = torch.sin(a) + b
#     assert_close(out_p, expected_out_p)
#     expected_out_t = torch.cos(a) + b
#     assert_close(out_t, expected_out_t)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_vmap_jvp(executor, device, _):
#     from thunder.core.transforms import vmap, jvp

#     def func(a, b):
#         assert isinstance(a, proxies.TensorProxy)
#         assert isinstance(b, proxies.TensorProxy)
#         assert a.ndim == 1
#         assert a.shape == b.shape
#         c = tlang.sin(a)
#         return tlang.mul(tlang.add(c, b), 1)

#     a = torch.ones(2, 3, device=device, dtype=torch.float32)
#     b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

#     args = (a, b)
#     out_p, out_t = thunder.make_traced(vmap(jvp(func), out_dims=(0, 0)), executor=executor)(args, args)
#     expected_out_p = torch.sin(a) + b
#     assert_close(out_p, expected_out_p)
#     expected_out_t = torch.cos(a) + b
#     assert_close(out_t, expected_out_t)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_jvp_vmap(executor, device, _):
#     from thunder.core.transforms import vmap, jvp

#     def func(a, b):
#         assert isinstance(a, proxies.TensorProxy)
#         assert isinstance(b, proxies.TensorProxy)
#         assert a.ndim == 1
#         assert a.shape == b.shape
#         c = tlang.sin(a)
#         return tlang.mul(tlang.add(c, b), 1)

#     a = torch.ones(2, 3, device=device, dtype=torch.float32)
#     b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

#     args = (a, b)
#     out_p, out_t = thunder.make_traced(jvp(vmap(func, out_dims=(0, 0))), executor=executor)(args, args)
#     expected_out_p = torch.sin(a) + b
#     assert_close(out_p, expected_out_p)
#     expected_out_t = torch.cos(a) + b
#     assert_close(out_t, expected_out_t)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_jvp(executor, device, _):
#     from thunder.core.transforms import jvp, inline, identity

#     def func(a, b):
#         c = tlang.sin(a)
#         return tlang.mul(tlang.add(c, b), 1)

#     a = torch.ones(2, 3, device=device, dtype=torch.float32)
#     b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

#     primals = (a, b)
#     tangents = (a, b)
#     out_p, out_t = thunder.make_traced(inline(identity(jvp(identity(func)))), executor=executor)(primals, tangents)
#     expected_out_p = torch.sin(a) + b
#     expected_out_t = torch.cos(a) + b
#     assert_close(out_p, expected_out_p)
#     assert_close(out_t, expected_out_t)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_jvp_no_inline(executor, device, _):
#     from thunder.core.transforms import jvp, inline, identity

#     def func(a, b):
#         c = tlang.sin(a)
#         return tlang.mul(tlang.add(c, b), 1)

#     a = torch.ones(2, 3, device=device, dtype=torch.float32)
#     b = torch.ones(2, 3, device=device, dtype=torch.float32) * 2

#     primals = (a, b)
#     tangents = (a, b)
#     out_p, out_t = thunder.make_traced(jvp(func), executor=executor)(primals, tangents)
#     expected_out_p = torch.sin(a) + b
#     expected_out_t = torch.cos(a) + b
#     assert_close(out_p, expected_out_p)
#     assert_close(out_t, expected_out_t)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_vmap_sum(executor, device, _):
#     from thunder.core.transforms import vmap

#     def func(a):
#         assert isinstance(a, proxies.TensorProxy)
#         assert a.ndim == 1
#         return ttorch.sum(a)

#     a = torch.ones(2, 3, device=device, dtype=torch.float32)

#     out = thunder.make_traced(vmap(func, out_dims=0), executor="torch")(a)
#     expected_out = torch.sum(a, dim=1)
#     assert_close(out, expected_out)


# @instantiate(
#     dtypes=NOTHING,
# )
# def test_transforms_jvp_python_number(executor, device, _):
#     from thunder.core.transforms import jvp, inline

#     scalars = (
#         2,
#         2.0,
#         True,
#     )
#     for scalar in scalars:

#         def func(a):
#             return tlang.mul(a, scalar)

#         a = make_tensor((2, 3), device=device, dtype=torch.float32)

#         primals = (a,)
#         tangents = (a,)
#         out_p, out_t = thunder.make_traced(inline(jvp(func)), executor=executor)(primals, tangents)

#         expected_out_p = a * scalar
#         expected_out_t = a * scalar
#         assert_close(out_p, expected_out_p)
#         assert_close(out_t, expected_out_t)


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


# TODO: this test just spot-checks type promotion -- it could probably be better
@instantiate(dtypes=NOTHING)
def test_type_promotion_tensors(executor, device, _):
    if executor == TorchExecutor:
        pytest.xfail("TorchExecutor currently fails at float x int64 x float16 type promotion")

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


# @instantiate(dtypes=NOTHING)
# def test_type_promotion_numbers_and_tensors(executor, device, _):
#     def foo(a, b, c):
#         return a + b + c

#     traced_foo = executor.make_callable(foo)

#     f16 = make_tensor((2, 2), device=device, dtype=torch.float16)
#     f32 = make_tensor((2, 2), device=device, dtype=torch.float32)
#     i64 = make_tensor((2, 2), device=device, dtype=torch.int64)

#     result = traced_foo(5, f32, 2)
#     assert result.dtype is torch.float32

#     result = traced_foo(f32, 1, f32)
#     assert result.dtype is torch.float32

#     result = traced_foo(i64, 3.0, f16)
#     assert result.dtype is torch.float16

#     result = traced_foo(i64, 3.0, i64)
#     assert result.dtype is torch.float32


# @instantiate(dtypes=NOTHING)
# def test_int_to_float_type_promotion(executor, device, _):
#     def foo(a, b):
#         return a / b

#    traced_foo = executor.make_callable(foo)

#     i64 = make_tensor((2, 2), device=device, dtype=torch.int64)
#     f16 = make_tensor((2, 2), device=device, dtype=torch.float16)

#     # int64 x int64 -- float32 result dtype
#     result = traced_foo(i64, i64)
#     assert result.dtype is torch.float32

#     # int x int64 -- float32 result dtype
#     result = traced_foo(2, i64)
#     assert result.dtype is torch.float32

#     # int64 x bool -- float32 result dtype
#     result = traced_foo(i64, True)
#     assert result.dtype is torch.float32

#     # int64 x float16 -- float16 result dtype
#     result = traced_foo(i64, f16)
#     assert result.dtype is torch.float16


# TODO: put this in test_tensor_creation.py
# TODO: specify multiple specific devices (today the test suite just passes a devicetype)
# TODO: add test for full (), which will cause a segfault
# @instantiate(dtypes=(thunder.float32,))
# def test_full(executor, device, dtype):
#     traced_full = executor.make_callable(tlang.full)

#     tdtype = ttorch.torch_dtype(dtype)

#     thunder_result = traced_full((1, 2, 3), 1.0, device=device, dtype=tdtype)
#     torch_result = torch.full((1, 2, 3), 1.0, device=device, dtype=tdtype)

#     assert_close(thunder_result, torch_result)


# @instantiate(dtypes=(thunder.float32,))
# def test_fusion_reuse(executor, device, dtype):
#     def foo(a, b, *, flag=False):
#         if flag:
#             return a + b
#         return a - b

#     traced_foo = executor.make_callable(foo, _return_fusion=True)
#     tdtype = ttorch.torch_dtype(dtype)

#     a = make_tensor((2,), device=device, dtype=tdtype)
#     b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

#     args = (a,)
#     kwargs = {"b": b, "flag": True}

#     thunder_result = traced_foo(*args, **kwargs)

#     torch_result = foo(*args, **kwargs)
#     assert_close(thunder_result["result"], torch_result)

#     fusion_result = thunder_result["fusion"](*args, **kwargs)
#     assert_close(fusion_result, torch_result)

#     # Calls the fusion with new tensor data (but preserves the flag arg)
#     a = make_tensor((2,), device=device, dtype=tdtype)
#     b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

#     args = (a,)
#     kwargs = {"b": b, "flag": True}

#     fusion_result = thunder_result["fusion"](*args, **kwargs)
#     torch_result = foo(*args, **kwargs)
#     assert_close(fusion_result, torch_result)

#     # Calls the fusion with new tensor data, and verifies the flag arg is ignored
#     a = make_tensor((2,), device=device, dtype=tdtype)
#     b = make_tensor((2, 2, 2), device=device, dtype=tdtype)

#     args = (a,)
#     kwargs = {"b": b, "flag": False}

#     fusion_result = thunder_result["fusion"](*args, **kwargs)
#     torch_result = foo(*args, b=b, flag=True)
#     assert_close(fusion_result, torch_result)

#     # Tests with PyTorch fallback
#     def bar(a, b):
#         c = a @ b
#         return c + c

#     traced_bar = executor.make_callable(bar, _return_fusion=True)

#     a = make_tensor((4, 16), device=device, dtype=tdtype)
#     b = make_tensor((16, 8), device=device, dtype=tdtype)

#     thunder_result = traced_bar(a, b)
#     torch_result = bar(a, b)
#     assert_close(torch_result, thunder_result["result"])

#     fusion_result = thunder_result["fusion"](a, b)
#     assert_close(torch_result, fusion_result)


# # TODO: probably only want to run this on nvFuser
# # TODO: maybe move to special test_nvfuser?
# @instantiate(
#     dtypes=(thunder.float32,),
#     executors=[
#         nvFuser(),
#     ],
# )
# @requiresCUDA
# def test_hybrid_execution(executor, device, dtype):
#     def foo(a, b, bias=None):
#         c = a + b
#         d = c + a
#         e = d + 2
#         f = ttorch.linear(a, c, bias)
#         g = c + f
#         return (c, e, 2), 5, f, g

#     def bar(a, b, bias=None):
#         c = a + b
#         d = c + a
#         e = d + 2
#         f = torch.nn.functional.linear(a, c, bias)
#         g = c + f
#         return (c, e, 2), 5, f, g

#     traced_foo = executor.make_callable(foo)
#     tdtype = ttorch.torch_dtype(dtype)

#     a = make_tensor((2, 2), device="cuda", dtype=torch.float32)
#     b = make_tensor((2, 2), device="cuda", dtype=torch.float32)
#     bias = None

#     result = traced_foo(a, b, bias)
#     torch_result = bar(a, b, bias)

#     assert_close(result, torch_result)


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


# def test_codeutils_unpack_pack():
#     o = object()
#     cases = (
#         [1, 2, 3],
#         [1, "a", object()],
#         (1, 2, 3),
#         (1, "a", object()),
#         {"a": 1, 2: 3, "x": object()},
#         [1, 2, [3, 4, o, [], 6, [[], [], (o, 5)], ((), ()), {}]],
#         {"x": (1, 2), (3, 4): ["a", o, ("c", {"d": o, 5: 3, 7: [1, 2, ((),)]})]},
#     )

#     for c in cases:
#         leaves, keys, packinfo = codeutils.unpack(c)
#         packed = codeutils.pack(leaves, packinfo)
#         assert c == packed


# def test_codeutils_siginfo():
#     def foo(a, b, *argles, c, d=2, e="hello", **vargles):
#         pass

#     o = object()
#     siginfo = codeutils.get_siginfo(foo, (3, 4, 7, "a", (1, 2)), {"c": 2, "e": 9, "f": 1, "g": o})

#     assert siginfo.args == (("a", 3), ("b", 4))
#     assert siginfo.varargs == ("argles", (7, "a", (1, 2)))
#     assert siginfo.kwargs == {"c": 2, "d": 2, "e": 9}
#     assert siginfo.varkwargs == ("vargles", {"f": 1, "g": o})


# def test_tree_flatten_only():
#     tree = [1, "a"]
#     flat, spec = tree_flatten_only(tree, lambda x: isinstance(x, str))
#     tree_only = tree_unflatten(flat, spec)

#     assert tree_only == ["a"]

#     flat[0] = "b"
#     tree_only = tree_unflatten(flat, spec)

#     assert tree_only == ["b"]

#     tree = [1, 2]
#     flat, spec = tree_flatten_only(tree, lambda x: isinstance(x, str))
#     tree_only = tree_unflatten(flat, spec)

#     assert tree_only == []

#     tree = [1, {"a": 1, "b": "two", "c": {"d": 5}}]
#     flat, spec = tree_flatten_only(tree, lambda x: isinstance(x, str))
#     tree_only = tree_unflatten(flat, spec)

#     assert tree_only == [{"b": "two"}]

#     tree = [1, {"a": 1, "b": 2, "c": {"d": "five"}}]
#     flat, spec = tree_flatten_only(tree, lambda x: isinstance(x, str))
#     tree_only = tree_unflatten(flat, spec)

#     assert tree_only == [{"c": {"d": "five"}}]


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
