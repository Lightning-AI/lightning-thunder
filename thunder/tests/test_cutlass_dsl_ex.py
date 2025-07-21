from __future__ import annotations
from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import thunder
from thunder.executors.cutlass_dsl_ex import cutlass_dsl_ex, is_device_quack_compat
from thunder.tests.framework import requiresCUDA

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable


_quack_available = find_spec("quack") is not None
quack_available = pytest.mark.skipif(
    not is_device_quack_compat() or not _quack_available,
    reason="quack requires SM9.0/10.0",
)
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DTYPE_IDS = tuple(str(a) for a in _DTYPES)


@pytest.fixture(autouse=True, scope="module")
def set_cuda_as_default_device():
    original_default_device: torch.device | None = None
    if torch.cuda.is_available():
        original_default_device = torch.get_default_device()
        torch.set_default_device("cuda")
    yield

    # Teardown
    if original_default_device is not None:
        torch.set_default_device(original_default_device)


def jit_with_cutlass_dsl_ex(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return thunder.jit(fn, executors=[cutlass_dsl_ex])


@requiresCUDA
@quack_available
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_quack_cross_entropy(dtype: torch.dtype):
    x = torch.randn((128, 1024), dtype=dtype, requires_grad=True)
    ref_x = x.clone().detach()
    targets = torch.randint(0, 128, (128,), dtype=torch.int64)

    jitted = jit_with_cutlass_dsl_ex(F.cross_entropy)

    expected = F.cross_entropy(ref_x, targets, reduction="none")
    actual = jitted(x, targets, reduction="none")
    torch.testing.assert_close(expected, actual)

    # expected_grad = torch.autograd.grad((expected,), (ref_x, targets), )
    # actual_grad = torch.autograd.grad((actual,), (x, targets))
    # torch.testing.assert_close(expected_grad, actual_grad)


@requiresCUDA
@quack_available
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_quack_softmax(dtype: torch.dtype):
    x = torch.randn((128, 1024), dtype=dtype, requires_grad=True)
    ref_x = x.clone().detach()

    jitted = jit_with_cutlass_dsl_ex(F.softmax)

    expected = F.softmax(ref_x, dim=-1, reduction="none")
    actual = jitted(x, dim=-1, reduction="none")
    torch.testing.assert_close(expected, actual)

    # expected_grad = torch.autograd.grad((expected,), (ref_x,))
    # actual_grad = torch.autograd.grad((actual,), (x,))
    # torch.testing.assert_close(expected_grad, actual_grad)


@requiresCUDA
@quack_available
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_quack_layernorm(dtype: torch.dtype):
    x = torch.randn((128, 1024), dtype=dtype, requires_grad=True)
    ref_x = x.clone().detach()

    module = nn.LayerNorm(1024).cuda()
    jitted = jit_with_cutlass_dsl_ex(module)

    expected = module(ref_x)
    actual = jitted(x)
    torch.testing.assert_close(expected, actual)


@requiresCUDA
@quack_available
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_quack_rmsrnorm(dtype: torch.dtype):
    x = torch.randn((128, 1024), dtype=dtype, requires_grad=True)
    ref_x = x.clone().detach()

    module = nn.RMSNorm(1024).cuda()
    jitted = jit_with_cutlass_dsl_ex(module)

    expected = module(ref_x)
    actual = jitted(x)
    torch.testing.assert_close(expected, actual)

    # expected_grad = torch.autograd.grad((expected,), (ref_x,))
    # actual_grad = torch.autograd.grad((actual,), (x,))
    # torch.testing.assert_close(expected_grad, actual_grad)
