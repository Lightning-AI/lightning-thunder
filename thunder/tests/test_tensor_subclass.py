from __future__ import annotations
import pytest
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from torch.utils import _pytree as pytree

from thunder.core import devices
from thunder.core import dtypes
from thunder.tests.framework import DynamoThunderExecutor
from thunder.tests.framework import instantiate
from thunder.tests.framework import nvFuserExecutor
from thunder.tests.framework import TorchExecutor
from thunder.tests.framework import TorchCompileExecutor
from thunder.tests.framework import TorchCompileCatExecutor
from thunder.tests.make_tensor import make_tensor

if TYPE_CHECKING:
    from typing import Any


class TensorSubclassForTest(torch.Tensor):
    _x: torch.Tensor
    _scale: torch.Tensor
    __slots__ = ["_x", "_scale"]

    def __new__(cls, x: torch.Tensor, scale: torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(f"Invalid `scale`: {scale}")
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            x.size(),
            strides=x.stride(),
            storage_offset=x.storage_offset(),
            dtype=x.dtype,
            layout=x.layout,
            requires_grad=x.requires_grad,
            device=x.device,
        )
        self._x = x
        self._scale = scale

        return self

    # ref: https://github.com/albanD/subclass_zoo/blob/ec47458/base_tensor.py#L22
    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):
        return f"TensorSubclassForTest(dtype={self._x.dtype}, scale={self._scale})"

    def __tensor_flatten__(self):
        return ["_x", "_scale"], {}

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, torch.Tensor], metadata: dict[str, Any], outer_size, outer_stride
    ):
        return TensorSubclassForTest(inner_tensors["_x"], inner_tensors["_scale"])

    @staticmethod
    def from_tensor(x: torch.Tensor):
        scale = x.abs().max()
        return TensorSubclassForTest(x, scale)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def allowed_subclass(typ):
            return (
                issubclass(cls, typ)
                or issubclass(torch._subclasses.FakeTensor, typ)
                or issubclass(torch._subclasses.functional_tensor.FunctionalTensor, typ)
            )

        def maybe_unwrap(t: TensorSubclassForTest | Any):
            if isinstance(t, TensorSubclassForTest):
                if t.is_floating_point():
                    return t._x * t._scale
                else:
                    return t._x
            return t

        if not all(allowed_subclass(t) for t in types):
            return NotImplementedError(f"Unsupported types are included: {types}")

        scales = tuple(t._scale for t in pytree.tree_flatten((args, kwargs))[0] if isinstance(t, TensorSubclassForTest))

        unwrapped_args, unwrapped_kwargs = pytree.tree_map(maybe_unwrap, (args, kwargs))
        out = func(*unwrapped_args, **unwrapped_kwargs)
        if not isinstance(out, torch.Tensor):
            return out
        else:
            out = cast(torch.Tensor, out)
            return TensorSubclassForTest(out, scales[0])


@instantiate(
    executors=(nvFuserExecutor, TorchExecutor, TorchCompileCatExecutor, TorchCompileExecutor),
    dtypes=(dtypes.float32,),
    devicetypes=(devices.DeviceType.CUDA,),
)
def test_subclass(executor, device, _):

    def f(a, b):
        return a + b

    x = TensorSubclassForTest.from_tensor(make_tensor((2, 2), device=device, dtype=torch.float32))
    y = TensorSubclassForTest.from_tensor(make_tensor((2, 2), device=device, dtype=torch.float32))

    out = f(x, y)

    torch_compiled = torch.compile(f)
    expected = torch_compiled(x, y)

    jitted = executor.make_callable(f)
    if executor == nvFuserExecutor:
        with pytest.raises(RuntimeError, match="Traceable tensor subclasses are not supported"):
            jitted(x, y)
            torch.cuda.synchronize()
    else:
        actual = jitted(x, y)
        torch.testing.assert_close(actual, expected)


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
