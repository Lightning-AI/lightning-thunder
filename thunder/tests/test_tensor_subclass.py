from __future__ import annotations
import pytest
from typing import TYPE_CHECKING, cast

import torch
from torch.utils import _pytree as pytree

from thunder.core import devices
from thunder.core import dtypes
from thunder.tests.framework import instantiate
from thunder.tests.framework import DynamoThunderExecutor
from thunder.tests.framework import nvFuserExecutor
from thunder.tests.framework import TorchExecutor
from thunder.tests.framework import TorchCompileExecutor
from thunder.tests.framework import TorchCompileCatExecutor
from thunder.tests.make_tensor import make_tensor

if TYPE_CHECKING:
    from typing import Any


@torch._dynamo.allow_in_graph
class Converter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return ScaleTensorSubclass.from_tensor(x)

    @staticmethod
    def backward(ctx, g):
        return g


def to_scale_tensor_subclass(t: torch.Tensor) -> ScaleTensorSubclass:
    return Converter.apply(t)


class ScaleTensorSubclass(torch.Tensor):
    _x: torch.Tensor
    _scale: torch.Tensor
    __slots__ = ["_x", "_scale"]

    def __new__(cls, x: torch.Tensor, scale: torch.Tensor):
        assert scale.numel() == 1, f"Invalid `scale`: {scale}"
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
        return f"ScaleTensorSubclass(dtype={self._x.dtype}, device={self._x.device}, x={self._x}, scale={self._scale})"

    def __tensor_flatten__(self) -> tuple[list[str], dict[str, Any]]:
        return ["_x", "_scale"], {}

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        outer_size,
        outer_stride,
    ) -> ScaleTensorSubclass:
        return ScaleTensorSubclass(inner_tensors["_x"], inner_tensors["_scale"])

    @staticmethod
    def from_tensor(x: torch.Tensor) -> ScaleTensorSubclass:
        scale = x.abs().max()
        return ScaleTensorSubclass(x, scale)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def allowed_subclass(typ):
            return (
                issubclass(cls, typ)
                or issubclass(torch._subclasses.FakeTensor, typ)
                or issubclass(torch._subclasses.functional_tensor.FunctionalTensor, typ)
            )

        def maybe_unwrap_and_scale(t: ScaleTensorSubclass | Any):
            if isinstance(t, ScaleTensorSubclass):
                if t.is_floating_point():
                    return t._x * t._scale
                else:
                    return t._x
            return t

        if not all(allowed_subclass(t) for t in types):
            return NotImplementedError(f"Unsupported types are included: {types}")

        scales = tuple(t._scale for t in pytree.tree_flatten((args, kwargs))[0] if isinstance(t, ScaleTensorSubclass))
        unwrapped_args, unwrapped_kwargs = pytree.tree_map(maybe_unwrap_and_scale, (args, kwargs))
        out = func(*unwrapped_args, **unwrapped_kwargs)
        if not isinstance(out, torch.Tensor):
            return out
        else:
            out = cast(torch.Tensor, out)
            return ScaleTensorSubclass(out, scales[0])


@instantiate(
    executors=(nvFuserExecutor, TorchExecutor, TorchCompileCatExecutor, TorchCompileExecutor, DynamoThunderExecutor),
    dtypes=(dtypes.float32,),
    devicetypes=(devices.DeviceType.CUDA,),
)
def test_subclass_inputs(executor, device, _):

    def f(a, b):
        return a + b

    x = ScaleTensorSubclass.from_tensor(make_tensor((2, 2), device=device, dtype=torch.float32))
    y = ScaleTensorSubclass.from_tensor(make_tensor((2, 2), device=device, dtype=torch.float32))
    expected = f(x, y)

    jitted = executor.make_callable(f)
    if executor != DynamoThunderExecutor:
        with pytest.raises(NotImplementedError, match="has Tensor Subclasses of"):
            jitted(x, y)


@instantiate(
    executors=(nvFuserExecutor, TorchExecutor, TorchCompileCatExecutor, TorchCompileExecutor, DynamoThunderExecutor),
    dtypes=(dtypes.float32,),
    devicetypes=(devices.DeviceType.CUDA,),
)
def test_conversion_to_subclass(executor, device, _):

    def f(a, b, c):
        d = a + b
        e = to_scale_tensor_subclass(d)
        return e - c

    x = make_tensor((2, 2), device=device, dtype=torch.float32)
    y = make_tensor((2, 2), device=device, dtype=torch.float32)
    z = ScaleTensorSubclass.from_tensor(make_tensor((2, 2), device=device, dtype=torch.float32))

    jitted = executor.make_callable(f)
    if executor != DynamoThunderExecutor:
        with pytest.raises(NotImplementedError, match="tensor subclasses are found"):
            jitted(x, y, z)
