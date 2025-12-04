from functools import partial
import builtins
import itertools
import math
import operator

import pytest
import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.clang as tlang
import thunder.torch as ttorch
import thunder.core.devices as devices
from thunder.tests.framework import instantiate, NOTHING


# allowed a-types, allowed b-types, special handling
_elementwise_binary_op_to_test_info = {
    operator.add: ((bool, int, float), (bool, int, float), None),
    operator.sub: ((bool, int, float), (bool, int, float), None),
    operator.mul: ((bool, int, float), (bool, int, float), None),
    operator.truediv: ((bool, int, float), (bool, int, float), "nonzero_only"),
    operator.floordiv: ((bool, int, float), (bool, int, float), "nonzero_only"),
    operator.mod: ((bool, int, float), (bool, int, float), "nonzero_only"),
    operator.pow: ((bool, int, float, complex), (int,), "pow_exponent"),
    operator.and_: ((bool, int), (bool, int), None),
    operator.or_: ((bool, int), (bool, int), None),
    operator.xor: ((bool, int), (bool, int), None),
    operator.lshift: ((bool, int), (int,), "shift_count"),
    operator.rshift: ((bool, int), (int,), "shift_count"),
}


@instantiate(
    dtypes=NOTHING,
    devicetypes=(devices.DeviceType.CPU,),
    decorators=(
        pytest.mark.parametrize("op", _elementwise_binary_op_to_test_info.keys(), ids=lambda op: op.__name__),
        pytest.mark.parametrize("cache_option", ("constant values", "symbolic values")),
    ),
)
def test_elementwise_binary_operations_on_numbers(executor, device, dtype, op, cache_option):
    a_types, b_types, special = _elementwise_binary_op_to_test_info[op]

    if cache_option == "symbolic values":
        a_types = tuple({int, float}.intersection(a_types))
        b_types = tuple({int, float}.intersection(b_types))

    bool_inps = [False, True]
    int_inps = [-1, 0, 2]
    float_inps = [-0.7, 0.0, 0.3, 1.1]
    complex_inps = [complex(1, 0.3), complex(-4.1, 0.9)]

    exponent_inps = [0, 1, 2]
    shift_inps = [0, 1, 2]

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

    def filter_b_values(vals, special):
        if special == "nonzero_only":
            return [v for v in vals if v != 0]
        if special == "pow_exponent":
            return exponent_inps
        if special == "shift_count":
            return shift_inps
        return vals

    def foo(a, b):
        return op(a, b)

    cfoo = executor.make_callable(foo, cache=cache_option)

    a_vals = gather_inputs(a_types)
    b_vals = filter_b_values(gather_inputs(b_types), special)

    for a in a_vals:
        for b in b_vals:
            actual = cfoo(a, b)
            expected = foo(a, b)
            assert_close(actual, expected)

    if cache_option == "symbolic values":
        assert thunder.cache_misses(cfoo) == len(a_types) * len(b_types)


_elementwise_unary_op_to_test_info = {
    builtins.abs: (bool, int, float, complex),
    math.ceil: (bool, int, float),
    math.floor: (bool, int, float),
    operator.inv: (bool, int),
    operator.neg: (bool, int, float, complex),
    operator.pos: (bool, int, float, complex),
    builtins.round: (bool, int, float),
    math.trunc: (bool, int, float),
}


@instantiate(
    dtypes=NOTHING,
    devicetypes=(devices.DeviceType.CPU,),
    decorators=(
        pytest.mark.parametrize("op", _elementwise_unary_op_to_test_info.keys(), ids=lambda op: op.__name__),
        pytest.mark.parametrize("cache_option", ("constant values", "symbolic values")),
    ),
)
def test_elementwise_dunder_operations_on_numbers(executor, device, dtype, op, cache_option):
    allowed_types = _elementwise_unary_op_to_test_info[op]

    if cache_option == "symbolic values":
        allowed_types = tuple({int, float}.intersection(allowed_types))

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

    def foo(a):
        return op(a)

    cfoo = executor.make_callable(foo, cache=cache_option)

    for a in gather_inputs(allowed_types):
        actual = cfoo(a)
        expected = foo(a)

        assert_close(actual, expected)

    if cache_option == "symbolic values":
        assert thunder.cache_misses(cfoo) == len(allowed_types)


@instantiate(
    dtypes=NOTHING,
    devicetypes=(devices.DeviceType.CPU,),
    decorators=(pytest.mark.parametrize("cache_option", ("constant values", "symbolic values")),),
)
def test_where_on_numbers(executor, device, dtype, cache_option):
    bool_inps = [False, True]
    int_inps = [-1, 2]
    float_inps = [-0.7, 1.1]
    complex_inps = [complex(1, 0.3), complex(-4.1, 0.9)]

    def foo(pred, a, b):
        return thunder.core.prims.where(pred, a, b)

    def foo_python(pred, a, b):
        return a if pred else b

    cfoo = executor.make_callable(foo, cache=cache_option)

    for inps in [bool_inps, int_inps, float_inps, complex_inps]:
        for pred, a, b in itertools.product(bool_inps, inps, inps):
            actual = cfoo(pred, a, b)
            expected = foo_python(pred, a, b)
            assert_close(actual, expected)

    if cache_option == "symbolic values":
        assert thunder.cache_misses(cfoo) == 4


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
