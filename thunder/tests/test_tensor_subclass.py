from __future__ import annotations
from typing import TYPE_CHECKING

import torch

import thunder
from thunder.tests.make_tensor import make_tensor

if TYPE_CHECKING:
    from typing import Any


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
            # requires_grad=x.requires_grad,
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
        if not isinstance(out, torch.Tensor):
            return out
        else:
            return ScaleTensorSubclass(out, scales[0])


def test_func_of_subclass_ctor_wrapper():

    def f(x: torch.Tensor, scale: torch.Tensor) -> ScaleTensorSubclass:
        return ScaleTensorSubclass(x, scale)

    device = torch.device("cuda")
    dtype = torch.float32
    shape = (2, 2)
    x = make_tensor(shape, device=device, dtype=dtype)
    scale = make_tensor((), device=device, dtype=dtype)

    jitted = thunder.jit(f)

    expected = f(x, scale)
    actual = jitted(x, scale)
    assert type(expected) is type(actual)
    torch.testing.assert_close((expected._x, expected._scale), (actual._x, actual._scale))
