from functools import partial

import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.torch as ttorch
from thunder.tests.framework import executors

import pytest


# TODO: convert these tests to OpInfo generated tests


@executors(dtypes=(thunder.float32,))
def test_torch_var(executor, device, dtype):
    # Tests passing all arguments as function inputs
    def foo(a, dim, *, keepdim=False, correction=1):
        return ttorch.var(a, dim, keepdim=keepdim, correction=correction)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

    # Full reduction
    thunder_result = traced_foo(a, [0, 1])
    torch_result = torch.var(a, [0, 1])
    assert_close(thunder_result, torch_result)

    # Reduce along dim 1
    thunder_result = traced_foo(a, [1])
    torch_result = torch.var(a, [1])
    assert_close(thunder_result, torch_result)

    # Specifying the correction
    thunder_result = traced_foo(a, [1], correction=2)
    torch_result = torch.var(a, [1], correction=2)
    assert_close(thunder_result, torch_result)

    # Specifying keepdim
    thunder_result = traced_foo(a, [1], keepdim=True, correction=2)
    torch_result = torch.var(a, [1], keepdim=True, correction=2)
    assert_close(thunder_result, torch_result)

    # Tests passing arguments as constants
    def foo(a):
        return ttorch.var(a, [0, 1], keepdim=True, correction=2)

    traced_foo = executor.make_callable(foo)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a)
    torch_result = torch.var(a, [0, 1], keepdim=True, correction=2)
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_torch_mean(executor, device, dtype):
    def foo(a, dim=None, keepdim=False, *, dtype=None):
        return ttorch.mean(a, dim, keepdim, dtype=dtype)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

    # Full reduction
    thunder_result = traced_foo(a, [0, 1])
    torch_result = torch.mean(a, [0, 1])
    assert_close(thunder_result, torch_result)

    # Reduce along dim 1
    thunder_result = traced_foo(a, [1])
    torch_result = torch.mean(a, [1])
    assert_close(thunder_result, torch_result)

    # Reduce with () dims
    thunder_result = traced_foo(a, ())
    torch_result = torch.mean(a, ())
    assert_close(thunder_result, torch_result)


@executors(dtypes=(thunder.float32,))
def test_var_mean(executor, device, dtype):
    def foo(a, dim=None, unbiased=None, keepdim=False, *, correction=None):
        return ttorch.var_mean(a, dim, unbiased, keepdim=keepdim, correction=correction)

    traced_foo = executor.make_callable(foo)

    tdtype = ttorch.to_torch_dtype(dtype)
    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

    # Full reduction
    thunder_result = traced_foo(a, [0, 1])
    torch_result = torch.var_mean(a, [0, 1])
    assert_close(thunder_result, torch_result)

    # Reduce along dim 1
    thunder_result = traced_foo(a, [1])
    torch_result = torch.var_mean(a, [1])
    assert_close(thunder_result, torch_result)

    # Tests passing arguments as constants
    def foo(a):
        return ttorch.var_mean(a, [0, 1], keepdim=True, correction=2)

    traced_foo = executor.make_callable(foo)

    a = torch.testing.make_tensor((4, 4), device=device, dtype=tdtype)

    thunder_result = traced_foo(a)
    torch_result = torch.var_mean(a, [0, 1], keepdim=True, correction=2)
    assert_close(thunder_result, torch_result)


# TODO: autogenerate consistency tests using opinfos
@executors(dtypes=(thunder.float32,))
def test_layer_norm(executor, device, dtype):
    thunder_fn = executor.make_callable(ttorch.layer_norm)
    torch_fn = torch.nn.functional.layer_norm
    tdtype = ttorch.to_torch_dtype(dtype)

    # TODO: improve these
    # input_shape, normalized_shape, kwargs
    cases = (
        ((1, 2, 3), (1, 2, 3), {"eps": 0.5}),
        ((2, 2, 3), (2, 3), {"eps": -0.5}),
        ((1,), (1,), {}),
        ((1, 2), (2,), {}),
        # ((0, 1), (1,), {}),  # nvFuser doesn't handle tensors with zero elements
    )

    make_arg = partial(make_tensor, device=device, dtype=tdtype)

    for input_shape, normalized_shape, kwargs in cases:
        # Shape of weight and bias should be the same as normalized_shape
        a = make_arg(input_shape)
        weight = make_arg(normalized_shape)
        bias = make_arg(normalized_shape)

        thunder_result = thunder_fn(a, normalized_shape, weight, bias, **kwargs)
        torch_result = torch_fn(a, normalized_shape, weight, bias, **kwargs)
        assert_close(thunder_result, torch_result, atol=1e-3, rtol=0.0)
