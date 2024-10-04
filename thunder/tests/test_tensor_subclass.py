import pytest

import torch
import torch.nn as nn

from thunder.core import devices
from thunder.core import dtypes
from thunder.tests.framework import DynamoThunderExecutor, instantiate, nvFuserExecutor
from thunder.tests.make_tensor import make_tensor


@instantiate(
    executors=(
        DynamoThunderExecutor,
        nvFuserExecutor,
    ),
    devicetypes=(devices.DeviceType.CUDA,),
    dtypes=(dtypes.bfloat16,),
)
def test_torchao_float8_linear(executor, device, dtype):
    pytest.importorskip("torchao")
    from torchao.float8 import convert_to_float8_training

    torch_device = devices.to_torch_device(device)
    torch_dtype = dtypes.to_torch_dtype(dtype)

    if (capability := torch.cuda.get_device_capability(torch_device)) < (8, 9):
        pytest.skip(f"Requires capability>=(8, 9) but  {capability=}")

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.GELU(approximate="tanh"),
        nn.Linear(32, 16),
    ).to(device=torch_device, dtype=torch_dtype)
    model = convert_to_float8_training(model)
    jitted = executor.make_callable(model)
    optimizer = torch.optim.AdamW(jitted.parameters())

    if executor == DynamoThunderExecutor:
        for _ in range(2):
            optimizer.zero_grad()
            x = make_tensor((16, 64), device=torch_device, dtype=torch_dtype, requires_grad=True)
            out = jitted(x)
            out.backward(torch.rand_like(out))

            optimizer.step()
    else:
        x = make_tensor((16, 64), device=torch_device, dtype=torch_dtype, requires_grad=True)
        with pytest.raises(AttributeError, match="Unknown attribute stride"):
            _ = jitted(x)
