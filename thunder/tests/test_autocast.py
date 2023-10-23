import itertools

import pytest
import torch

import thunder
import thunder.torch as ltorch
from thunder.core import devices, dtypes
from thunder.tests.framework import instantiate


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
    if torch_device.type == "cuda" and dtype == dtypes.bfloat16 and not torch.cuda.is_bf16_supported():
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
        autocast_dtypes = (thunder.bfloat16, thunder.float16) if torch.cuda.is_bf16_supported() else (thunder.float16,)
    else:
        pytest.fail(f"Invalid combination of parameters: {executor=}, {device=}, {dtype=}")
    for (func, should_autocast), autocast_dtype in itertools.product(
        ((f, True), (g, False), (h, True)), autocast_dtypes
    ):
        autocast_torch_dtype = ltorch.to_torch_dtype(autocast_dtype)
        x, y, z = (torch.randn((2, 2), device=device, dtype=torch_dtype) for _ in range(3))
        compiled = executor.make_callable(autocast(func, dtype=autocast_dtype))
        out = compiled(x, y, z)

        devicetype = torch.device(device).type
        # note(crcrpar): This test could be broken in the future as thunder autocast develops.
        with torch.autocast(device_type=devicetype, dtype=autocast_torch_dtype):
            torch_output = func(x, y, z)
        assert out.dtype == torch_output.dtype
