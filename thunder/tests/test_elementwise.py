from functools import partial
import builtins
import math
import operator

import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.clang as tlang
import thunder.torch as ttorch
import thunder.core.devices as devices
from thunder.tests.framework import instantiate, NOTHING, ops, run_snippet
from thunder.tests.opinfos import elementwise_binary_ops


# TODO Enable the remaining elementwise unary operations (following the pattern of abs)
# TODO Expand testing to elementwise binary operations (following a similar pattern)
@instantiate(dtypes=NOTHING, devicetypes=(devices.DeviceType.CPU,))
def test_elementwise_dunder_operations_on_numbers(executor, device, dtype):
    # op, allowed types
    elementwise_unary_ops = (
        (builtins.abs, (bool, int, float, complex)),
        # (math.ceil, (bool, int, float)),
        # (math.floor, (bool, int, float)),
        # (operator.inv, (bool, int)),
        # (operator.neg, (bool, int, float, complex)),
        # # TODO see issue "Implement positive operations"
        # # operator.pos,
        # (builtins.round, (bool, int, float)),
        # (math.trunc, (bool, int, float)),
    )

    bool_inps = [False, True]
    int_inps = [-1, 0, 2]
    float_inps = [-0.7, 0, 0.3, 1.1]
    complex_inps = [complex(1, 0.3), complex(-4.1, 0.9)]

    _type_to_input_map = {
        bool: bool_inps,
        int: int_inps,
        float: float_inps,
        complex: complex_inps,
    }

    def gather_inputs(allowed_types):
        inps = []

        for typ in allowed_types:
            inps.extend(_type_to_input_map[typ])

        return inps

    for op, allowed_types in elementwise_unary_ops:

        def foo(a):
            return op(a)

        cfoo = executor.make_callable(foo)

        for a in gather_inputs(allowed_types):
            actual = cfoo(a)
            expected = foo(a)

            assert_close(actual, expected)


# TODO: see issue "Test operator and method variants of operations using
# OpInfos"
@instantiate(dtypes=(thunder.float32,))
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


# NOTE This cannot be an OpInfo test (as OpInfos work today, anyway)
#   because it tests multiple dtypes for correct promotion behavior
@instantiate(dtypes=NOTHING)
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

    # TODO fix issue "Currently nvFuser tensor x float operations result in
    # float64 results"
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
