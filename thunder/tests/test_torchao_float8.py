import pytest

pytest.importorskip("torchao")

import torch
from torchao.float8 import convert_to_float8_training

import thunder


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="Requires cuda of 8.9 or higher",
)
def test_float8_linear():
    model: torch.nn.Module = (
        torch.nn.Sequential(
            torch.nn.Linear(2048, 4096),
            torch.nn.Linear(4096, 128),
        )
        .bfloat16()
        .cuda()
    )
    convert_to_float8_training(model)
    x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)

    jitted = thunder.jit(model)
    _ = jitted(x)
