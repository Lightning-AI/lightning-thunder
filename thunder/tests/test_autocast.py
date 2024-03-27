import itertools

import pytest
import torch

import thunder
import thunder.torch as ltorch
from thunder.core import devices, dtypes
from thunder.tests.framework import instantiate, TorchExecutor
import thunder.tests.bf16


# TODO This test currently ignores the "should_autocast" argument enumerated in it
@instantiate(
    dtypes=dtypes.float_dtypes - {float},
)
def test_thunder_autocast_transform(executor, device, dtype):
    from thunder.core.transforms import autocast

    # TODO: Consider adding support for device specific dtypes in the test
    # instantiator.
    torch_device = torch.device(device)
    if torch_device.type == "cpu" and dtype == dtypes.float16:
        pytest.skip("float16 matmul is not supported on CPU.")
    if torch_device.type == "cuda" and dtype == dtypes.bfloat16 and not thunder.tests.bf16.device_supports_bf16(device):
        pytest.skip(f"bfloat16 is not supported on {torch.cuda.get_device_name()}")

    def f(a, b, c):
        return a @ (b + c)

    # The following functions needs to be updated as autocast_impls grows.
    def g(a, b, c):
        return a + b - c

    def h(a, b, c):
        return (a @ b) + c

    torch_dtype = ltorch.to_torch_dtype(dtype)
    if torch_device.type == "cpu":
        autocast_dtypes = (thunder.bfloat16,)
    elif torch_device.type == "cuda":
        autocast_dtypes = (
            (thunder.bfloat16, thunder.float16)
            if thunder.tests.bf16.device_supports_bf16(device)
            else (thunder.float16,)
        )
    else:
        pytest.fail(f"Invalid combination of parameters: {executor=}, {device=}, {dtype=}")
    for (func, should_autocast), autocast_dtype in itertools.product(
        ((f, True), (g, False), (h, True)), autocast_dtypes
    ):
        autocast_torch_dtype = ltorch.to_torch_dtype(autocast_dtype)
        x, y, z = (torch.randn((2, 2), device=device, dtype=torch_dtype) for _ in range(3))
        compiled = executor.make_callable_legacy(autocast(func, dtype=autocast_dtype))
        out = compiled(x, y, z)

        devicetype = torch.device(device).type
        # note(crcrpar): This test could be broken in the future as thunder autocast develops.
        with torch.autocast(device_type=devicetype, dtype=autocast_torch_dtype):
            torch_output = func(x, y, z)
        assert out.dtype == torch_output.dtype


@instantiate(
    executors=[TorchExecutor],
    dtypes=dtypes.float_dtypes - {float},
)
def test_no_autocast(executor, device, dtype):
    from thunder.core.symbol import Symbol
    from thunder.core.proxies import NumberProxy

    del executor

    is_autocast_enabled = Symbol("is_autocast_enabled", meta=lambda: NumberProxy(python_type=bool), _module=torch)
    is_autocast_cpu_enabled = Symbol(
        "is_autocast_cpu_enabled", meta=lambda: NumberProxy(python_type=bool), _module=torch
    )

    def func():
        return is_autocast_enabled(), is_autocast_cpu_enabled()

    trace = thunder.trace()(func)
    no_autocast_cfunc = trace.python_callable()
    cfunc = no_autocast_cfunc.__wrapped__.__wrapped__  # This function can be affected by the autocast context.
    b1, b2 = no_autocast_cfunc()
    assert b1 is False
    assert b2 is False

    torch_dtype = ltorch.to_torch_dtype(dtype)
    torch_device = torch.device(device)
    if torch_device.type == "cpu" and dtype == dtypes.float16:
        pytest.skip("float16 matmul is not supported on CPU.")
    if torch_device.type == "cuda" and dtype == dtypes.bfloat16 and not thunder.tests.bf16.device_supports_bf16(device):
        pytest.skip(f"bfloat16 is not supported on {torch.cuda.get_device_name()}")

    devicetype = torch.device(device).type
    test_dtype = torch.float16
    if torch_device.type == "cpu":
        test_dtype = torch.bfloat16
    with torch.autocast(device_type=devicetype, dtype=test_dtype):
        b1, b2 = no_autocast_cfunc()
        b3, b4 = cfunc()
    assert b1 is False
    assert b2 is False
    assert b3 is (True if torch_device.type == "cuda" else False)
    assert b4 is (True if torch_device.type == "cpu" else False)


@instantiate(
    dtypes=dtypes.float_dtypes - {float},
)
def test_compile_autocast(executor, device, dtype):
    del executor

    def func(a, b):
        return a @ b

    torch_dtype = ltorch.to_torch_dtype(dtype)
    torch_device = torch.device(device)
    if dtype == dtypes.float64:
        pytest.skip("float64 autocast is not supported.")
    if torch_device.type == "cpu" and dtype == dtypes.float16:
        pytest.skip("float16 matmul is not supported on CPU.")
    if torch_device.type == "cuda" and dtype == dtypes.bfloat16 and not thunder.tests.bf16.device_supports_bf16(device):
        pytest.skip(f"bfloat16 is not supported on {torch.cuda.get_device_name()}")
    a = torch.randn(2, 2, device=device, dtype=torch_dtype)
    b = torch.randn(2, 2, device=device, dtype=torch_dtype)
    cfunc = thunder.jit(func)
    devicetype = torch.device(device).type
    test_dtype = torch.float16 if torch_device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=devicetype, dtype=test_dtype):
        output = cfunc(a, b)
    assert output.dtype == (torch.float16 if torch_device.type == "cuda" else torch.bfloat16)
