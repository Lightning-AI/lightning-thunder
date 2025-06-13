import itertools

import pytest
import torch
from torch._dynamo.eval_frame import is_inductor_supported

import thunder
import thunder.tests.bf16
import thunder.torch as ltorch
from thunder.core import dtypes
from thunder.tests.framework import instantiate, TorchExecutor, requiresCUDA


# TODO This test currently ignores the "should_autocast" argument enumerated in it
@instantiate(
    dtypes=dtypes.float_math_dtypes,
)
def test_thunder_autocast_transform(executor, device, dtype):
    from thunder.transforms.autocast import autocast

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
        initial_trace = thunder.trace()(autocast(func, dtype=autocast_dtype), x, y, z)
        compiled = executor.make_callable(initial_trace.python_callable(), disable_torch_autograd=True)
        out = compiled(x, y, z)

        devicetype = torch.device(device).type
        # note(crcrpar): This test could be broken in the future as thunder autocast develops.
        with torch.autocast(device_type=devicetype, dtype=autocast_torch_dtype):
            torch_output = func(x, y, z)
        assert out.dtype == torch_output.dtype


@instantiate(
    executors=[TorchExecutor],
    dtypes=dtypes.float_math_dtypes,
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
    python_callable = trace.python_callable()
    # 2 unwraps for:
    # @no_grad()
    # @no_autocast
    cfunc = python_callable.__wrapped__.__wrapped__
    b1, b2 = python_callable()
    assert b1 is False
    assert b2 is False

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
        b1, b2 = python_callable()
        b3, b4 = cfunc()
    assert not b1
    assert not b2
    assert not b3
    assert not b4


@instantiate(
    dtypes=dtypes.float_math_dtypes,
    decorators=(pytest.mark.skipif(not is_inductor_supported(), reason="inductor unsupported"),),
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


def test_autocast_mixed_dtype_inputs():
    def foo(x, w):
        return torch.nn.functional.linear(x, w)

    # Mixed input types.
    x, w = torch.randn(16, 16, dtype=torch.bfloat16), torch.randn(16, 16)

    jfoo = thunder.jit(foo)

    with torch.autocast("cpu", torch.bfloat16):
        eager_out = foo(x, w)
        jit_out = jfoo(x, w)

    torch.testing.assert_close(eager_out, jit_out)


def test_autocast_mixed_dtype_inputs_on_prims():
    # Verify that the autocast rules are applied when
    # directly using the prims.
    # See - https://github.com/Lightning-AI/lightning-thunder/issues/725
    def foo(x, w):
        return thunder.prims.linear(x, w, None)

    # Mixed input types.
    x, w = torch.randn(16, 16, dtype=torch.bfloat16), torch.randn(16, 16)

    jfoo = thunder.jit(foo)

    with torch.autocast("cpu", torch.bfloat16):
        jit_out = jfoo(x, w)

    assert jit_out.dtype == torch.bfloat16
    exec_trace = thunder.last_traces(jfoo)[0]
    assert any(bsym.sym.id == thunder.prims.PrimIDs.CONVERT_ELEMENT_TYPE for bsym in exec_trace.bound_symbols)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_autocast_convolution(dim, requires_grad):
    conv_fn = getattr(torch.nn.functional, f"conv{dim}d")

    def foo(x, w, b=None):
        return conv_fn(x, w, b)

    x = torch.rand(1, 2, *(dim * (8,)), requires_grad=requires_grad)
    w = torch.rand(3, 2, *(dim * (4,)), requires_grad=requires_grad)
    b = torch.rand(3, requires_grad=requires_grad)
    go = torch.rand(1, 3, *(dim * (5,)))

    jfoo = thunder.jit(foo)

    with torch.autocast("cpu", torch.float16):
        eager_out = foo(x, w, b)
        jit_out = jfoo(x, w, b)

    torch.testing.assert_close(eager_out, jit_out)

    if requires_grad:
        eager_grads = torch.autograd.grad(eager_out, [x, w, b], go)
        jit_grads = torch.autograd.grad(jit_out, [x, w, b], go)

        for eg, jg in zip(eager_grads, jit_grads):
            torch.testing.assert_close(eg, jg, rtol=5e-2, atol=5e-2)

    with torch.autocast("cpu", torch.float16):
        eager_out = foo(x, w)
        jit_out = jfoo(x, w)

    torch.testing.assert_close(eager_out, jit_out)

    if requires_grad:
        go = torch.randn_like(eager_out)
        eager_grads = torch.autograd.grad(eager_out, [x, w], go)
        jit_grads = torch.autograd.grad(jit_out, [x, w], go)

        for eg, jg in zip(eager_grads, jit_grads):
            torch.testing.assert_close(eg, jg, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("b_dtype", (torch.float, torch.bfloat16))
def test_autocast_torch_matmul(requires_grad, device, b_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skipping - CUDA is not available")

    def foo(a, b):
        output = torch.matmul(a, b)
        return output

    a = torch.rand([3, 1, 2], dtype=torch.float, device=device, requires_grad=requires_grad)
    b = torch.rand([2, 3], dtype=b_dtype, device=device, requires_grad=requires_grad)

    with torch.autocast(device, torch.bfloat16):
        expected = foo(a, b)
        actual = thunder.jit(foo)(a, b)

    torch.testing.assert_close(actual, expected)

    if requires_grad:
        go = torch.ones_like(expected) / expected.numel()
        eager_grads = torch.autograd.grad(expected, [a, b], go)
        jit_grads = torch.autograd.grad(actual, [a, b], go)

        for eg, jg in zip(eager_grads, jit_grads):
            torch.testing.assert_close(eg, jg, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("b_dtype", (torch.float, torch.bfloat16))
def test_autocast_trace(requires_grad, device, b_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skipping - CUDA is not available")

    def foo(a, b):
        with torch.autocast(device, torch.bfloat16):
            autocast_output = torch.matmul(a, b)
            with torch.autocast(device, torch.bfloat16, enabled=False):
                if a.dtype != b.dtype:
                    b = b.to(a.dtype)
                non_autocast_output = torch.matmul(a, b)
        return autocast_output, non_autocast_output

    a = torch.rand([3, 1, 2], dtype=torch.float, device=device, requires_grad=requires_grad)
    b = torch.rand([2, 3], dtype=b_dtype, device=device, requires_grad=requires_grad)

    autocast_expected, non_autocast_expected = foo(a, b)
    autocast_actual, non_autocast_actual = thunder.jit(foo)(a, b)

    torch.testing.assert_close(autocast_actual, autocast_expected)
    torch.testing.assert_close(non_autocast_actual, non_autocast_expected)

    if requires_grad:
        go = torch.ones_like(autocast_actual) / autocast_actual.numel()
        eager_grads = torch.autograd.grad(autocast_expected, [a, b], go)
        jit_grads = torch.autograd.grad(autocast_actual, [a, b], go)

        for eg, jg in zip(eager_grads, jit_grads):
            torch.testing.assert_close(eg, jg, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("b_dtype", (torch.float, torch.bfloat16))
@requiresCUDA
def test_autocast_cpu_and_cuda(requires_grad, b_dtype):
    def foo(a, b, c, d):
        with torch.autocast("cpu", torch.bfloat16):
            cpu_output = torch.matmul(a, b)
            with torch.autocast("cuda", torch.bfloat16):
                cuda_output = torch.matmul(c, d)
        return cpu_output, cuda_output

    a = torch.rand([3, 1, 2], dtype=torch.float, device="cpu", requires_grad=requires_grad)
    b = torch.rand([2, 3], dtype=b_dtype, device="cpu", requires_grad=requires_grad)
    c = torch.rand([3, 1, 2], dtype=torch.float, device="cuda", requires_grad=requires_grad)
    d = torch.rand([2, 3], dtype=b_dtype, device="cuda", requires_grad=requires_grad)

    cpu_expected, cuda_expected = foo(a, b, c, d)
    cpu_actual, cuda_actual = thunder.jit(foo)(a, b, c, d)

    torch.testing.assert_close(cpu_actual, cpu_expected)
    torch.testing.assert_close(cuda_actual, cuda_expected)

    if requires_grad:
        go = torch.ones_like(cpu_actual) / cpu_actual.numel()
        eager_grads = torch.autograd.grad(cpu_expected, [a, b], go)
        jit_grads = torch.autograd.grad(cpu_actual, [a, b], go)

        for eg, jg in zip(eager_grads, jit_grads):
            torch.testing.assert_close(eg, jg, rtol=5e-3, atol=5e-3)
