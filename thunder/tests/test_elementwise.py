from functools import partial

import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.clang as tlang
import thunder.torch as ttorch
from thunder.tests.framework import executors, NOTHING


# Tests for elementwise binary operators

# TODO: test that the operator variant works properly
# TODO: generate the following tests using opinfos (more number sample inputs needed)


@executors(dtypes=(thunder.float32,))
def test_abs_integer(executor, device, dtype):
    def foo(a, b):
        a_abs = tlang.abs(a)
        return tlang.add(a_abs, b)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = -3
    b = make_tensor((1, 8), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b)
    torch_result = 3 + b
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_abs_float(executor, device, dtype):
    def foo(a, b):
        a_abs = tlang.abs(a)
        return tlang.add(a_abs, b)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = -2.7
    b = make_tensor((1, 8), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b)
    torch_result = abs(a) + b
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_core_tensor_methods(executor, device, dtype):
    def foo(a, b, c, d):
        return a + b - c + (d - a)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = make_tensor((4, 4), device=device, dtype=tdtype)
    b = make_tensor((2, 1, 4), device=device, dtype=tdtype)
    c = make_tensor((4, 1), device=device, dtype=tdtype)
    d = make_tensor((1, 1, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b, c, d)
    torch_result = a + b - c + (d - a)
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_integer_constant(executor, device, dtype):
    def foo(a, b):
        c = tlang.add(a, 2)
        return tlang.add(c, b)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = make_tensor((2, 1), device=device, dtype=tdtype)
    b = make_tensor((1, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, b)
    torch_result = (a + 2) + b

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_integer_input(executor, device, dtype):
    def foo(a, b):
        return tlang.add(a, b)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = make_tensor((2, 1), device=device, dtype=tdtype)

    thunder_result = traced_foo(a, 3)
    torch_result = a + 3

    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_integer_inputs(executor, device, dtype):
    def foo(a, b, c):
        d = tlang.add(a, b)
        return tlang.add(c, d)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = make_tensor((3, 2), device=device, dtype=tdtype)

    thunder_result = traced_foo(3, 4, a)
    torch_result = 3 + 4 + a
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_integer_constants(executor, device, dtype):
    def foo(a):
        b = tlang.add(2, 3)
        return tlang.add(a, b)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = make_tensor((2, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a)
    torch_result = 5 + a
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_add_floats(executor, device, dtype):
    def foo(a, b):
        c = tlang.add(2.0, a)
        return tlang.add(b, c)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = make_tensor((2, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(0.7, a)
    torch_result = 2.0 + 0.7 + a
    assert_close(thunder_result, torch_result)


# TODO: this test can be replaced with OpInfo generated tests (dtype None tests and error inputs tests)
@executors(dtypes=NOTHING)
def test_where(executor, device, dtype):
    # Tests where type promotion and number support

    thunder_fn = executor.make_callable(tlang.where)
    torch_fn = torch.where

    shape = (5, 5)
    make = partial(make_tensor, device=device)
    pred = make(shape, dtype=torch.bool)

    # int64 x float32
    i64 = make(shape, dtype=torch.int64)
    f32 = make(shape, dtype=torch.float32)

    thunder_result = thunder_fn(pred, i64, f32)
    torch_result = torch_fn(pred, i64, f32)
    assert_close(thunder_result, torch_result)

    # int x float32
    thunder_result = thunder_fn(pred, 5, f32)
    torch_result = torch_fn(pred, 5, f32)
    assert_close(thunder_result, torch_result)

    # int64 x int
    thunder_result = thunder_fn(pred, i64, 5)
    torch_result = torch_fn(pred, i64, 5)
    assert_close(thunder_result, torch_result)

    # int x int
    thunder_result = thunder_fn(pred, -1, 5)
    torch_result = torch_fn(pred, -1, 5)
    assert_close(thunder_result, torch_result)

    # int64 x float
    thunder_result = thunder_fn(pred, i64, -2.3)
    torch_result = torch_fn(pred, i64, -2.3)
    assert_close(thunder_result, torch_result)

    # FIXME: https://github.com/csarofeen/pytorch/issues/2380
    # float x int
    # thunder_result = thunder_fn(pred, 3., 5)
    # torch_result = torch_fn(pred, 3., 5)
    # assert_close(thunder_result, torch_result)

    # TODO:
    # float x float
    # int x complex
    # int64 x complex
    # int32 x complex64
    # predicate as number (True/False)
