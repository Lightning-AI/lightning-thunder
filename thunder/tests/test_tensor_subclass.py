from __future__ import annotations
from typing import TYPE_CHECKING

from lightning_utilities.core.imports import package_available
import pytest
import torch
import torch.nn as nn
from torch.utils import _pytree as pytree

import thunder
from thunder.dynamo.compiler import ThunderCompiler
from thunder.tests.framework import (
    DynamoThunderExecutor,
    TorchExecutor,
    instantiate,
    nvFuserExecutor,
)
from thunder.tests.make_tensor import make_tensor

if TYPE_CHECKING:
    from typing import Any


TORCHAO_AVAILABLE = package_available("torchao")


@torch._dynamo.allow_in_graph
class EncapsulateXandScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor):
        return ScaleTensorSubclass(x, scale)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def encapsulate_x_and_scale(x, scale) -> ScaleTensorSubclass:
    return EncapsulateXandScale.apply(x, scale)


@torch._dynamo.allow_in_graph
class ToScaleTensorSubclass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return ScaleTensorSubclass.from_tensor(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def to_scale_tensor_subclass(x: torch.Tensor) -> ScaleTensorSubclass:
    return ToScaleTensorSubclass.apply(x)


class ScaleTensorSubclass(torch.Tensor):
    _x: torch.Tensor
    _scale: torch.Tensor
    __slots__ = ["_x", "_scale"]

    def __new__(cls, x: torch.Tensor, scale: torch.Tensor):
        assert scale.numel() == 1, f"Invalid `scale`: {scale}"
        dtype = x.dtype
        device = x.device
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            x.size(),
            dtype=dtype,
            device=device,
            # strides=x.stride(),
            # storage_offset=x.storage_offset(),
            # layout=x.layout,
            requires_grad=x.requires_grad,
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
    def __torch_dispatch__(cls, aten_ir_op: torch._ops.OpOverload, types, args=(), kwargs=None):

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
        out = aten_ir_op(*unwrapped_args, **unwrapped_kwargs)
        return out


@instantiate(
    dtypes=(thunder.core.dtypes.float32,),
)
def test_func_of_subclass_ctor_wrapper(executor, device, _):

    def f(x: torch.Tensor, scale: torch.Tensor) -> ScaleTensorSubclass:
        y = ScaleTensorSubclass(x, scale)
        return y

    jitted = executor.make_callable(f)

    dtype = torch.float32
    shape = (2, 2)
    x = make_tensor(shape, device=device, dtype=dtype)
    scale = make_tensor((), device=device, dtype=dtype)

    expected = f(x, scale)
    actual = jitted(x, scale)
    torch.testing.assert_close((expected._x, expected._scale), (actual._x, actual._scale))

    def f(x: torch.Tensor, scale: torch.Tensor):
        y = ScaleTensorSubclass(x, scale)
        z = ScaleTensorSubclass(y._x, y._scale)
        return z

    jitted = executor.make_callable(f)

    expected = f(x, scale)
    actual = jitted(x, scale)
    torch.testing.assert_close((expected._x, expected._scale), (actual._x, actual._scale))

    print(thunder.last_traces(jitted)[0])


@instantiate(
    dtypes=(thunder.core.dtypes.float32,),
)
def test_func_calling_converter(executor, device, _):

    def f(x: torch.Tensor, scale: torch.Tensor) -> ScaleTensorSubclass:
        y = encapsulate_x_and_scale(x, scale)
        return y

    jitted = executor.make_callable(f)

    dtype = torch.float32
    shape = (2, 2)

    x = make_tensor(shape, device=device, dtype=dtype)
    scale = make_tensor((), device=device, dtype=dtype)

    expected = f(x, scale)
    actual = jitted(x, scale)
    torch.testing.assert_close((expected._x, expected._scale), (actual._x, actual._scale))

    def g(x: torch.Tensor) -> ScaleTensorSubclass:
        y = to_scale_tensor_subclass(x)
        return y

    jitted = thunder.jit(g)
    x = make_tensor(shape, device=device, dtype=dtype)

    expected = g(x)
    actual = jitted(x)
    torch.testing.assert_close((expected._x, expected._scale), (actual._x, actual._scale))


@instantiate(
    dtypes=(thunder.core.dtypes.float32,),
    decorators=(pytest.mark.parametrize("requires_grad", (False, True), ids=("fwd_only", "with_bwd")),),
)
def test_func_of_subclass_simple_math(executor, device, _, requires_grad):

    def f(x: ScaleTensorSubclass, y: ScaleTensorSubclass) -> torch.Tensor:
        out = x + y
        return out

    jitted = executor.make_callable(f)

    dtype = torch.float32
    shape = (2, 2)
    x = ScaleTensorSubclass(
        make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad),
        make_tensor((), device=device, dtype=dtype),
    )
    y = ScaleTensorSubclass(
        make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad),
        make_tensor((), device=device, dtype=dtype),
    )

    expected = f(x, y)
    actual = jitted(x, y)
    assert type(expected) is type(actual)
    torch.testing.assert_close(expected, actual)
    if requires_grad:
        actual.mean().backward()

    def g(x: ScaleTensorSubclass, data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        y = EncapsulateXandScale.apply(data, scale)
        out = x + y
        return out

    jitted = executor.make_callable(g)

    x = ScaleTensorSubclass(
        make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad),
        make_tensor((), device=device, dtype=dtype),
    )
    data = make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    scale = make_tensor((), device=device, dtype=dtype)

    expected = g(x, data, scale)
    actual = jitted(x, data, scale)
    assert type(expected) is type(actual)
    torch.testing.assert_close(expected, actual)
    if requires_grad:
        actual.mean().backward()


@instantiate(
    dtypes=(thunder.core.dtypes.float32, thunder.core.dtypes.bfloat16),
    devicetypes=(thunder.core.devices.DeviceType.CUDA,),
    executors=(TorchExecutor, nvFuserExecutor, DynamoThunderExecutor),
    decorators=(
        pytest.mark.skipif(
            not (TORCHAO_AVAILABLE and torch.cuda.get_device_capability() >= (8, 9)),
            reason="Requires capability >= 8.9 and torchao",
        ),
        pytest.mark.parametrize("bias", (True, False)),
    ),
)
def test_torchao_float8_linear(executor, device, dtype, bias):
    from torchao.float8 import convert_to_float8_training

    batch_size, in_features, out_features = 16, 32, 64
    device = torch.device("cuda")
    torch_dtype = thunder.core.dtypes.to_torch_dtype(dtype)

    model = nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.GELU(approximate="tanh"),
        nn.Linear(out_features, out_features, bias=bias),
    ).to(device=device, dtype=torch_dtype)
    fp8_model = convert_to_float8_training(model)
    x = make_tensor((batch_size, in_features), device=device, dtype=torch_dtype)

    expected: torch.Tensor
    jitted: nn.Module
    backend: ThunderCompiler | None = None

    if is_thunderfx := executor == DynamoThunderExecutor:
        torch._dynamo.reset()
        expected = torch.compile(fp8_model)(x)
        backend = ThunderCompiler()
        jitted = torch.compile(fp8_model, backend=backend)
    else:
        expected = fp8_model(x)
        jitted = executor.make_callable(fp8_model)

    if bias and dtype == thunder.core.dtypes.bfloat16 and executor == nvFuserExecutor:
        with pytest.raises(
            RuntimeError, match="Failed to compute the min-cut on the graph due to a path with infinite capacity"
        ):
            jitted(x)
        return
    actual = jitted(x)
    if bias and dtype == thunder.core.dtypes.bfloat16 and executor == DynamoThunderExecutor:
        with pytest.raises(AssertionError, match="Tensor-likes are not close"):
            torch.testing.assert_close(actual, expected)
        return

    if (dtype == thunder.core.dtypes.bfloat16 and executor != DynamoThunderExecutor) or (
        not bias and dtype == thunder.core.dtypes.bfloat16 and executor == DynamoThunderExecutor
    ):
        pytest.xfail("numerical error")
    torch.testing.assert_close(actual, expected)

    # TODO(crcrpar): Think of how to push tensor subclasses to `thunder.jit`.
    # Currently no subgraphs go to thunder.jit.
    if is_thunderfx:
        for subgraph in backend.subgraph_infos:
            if not bias and dtype == thunder.core.dtypes.bfloat16:
                assert not subgraph.thunder_compiled_fns
            else:
                assert subgraph.thunder_compiled_fns
