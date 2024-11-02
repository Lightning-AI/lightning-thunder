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


# Error message:
#         unpack_fn = d.get(inst)
#         if unpack_fn is None:
# >           raise NotImplementedError(f"Unpacking from {inst} {provenance}")
# E           NotImplementedError: Unpacking from LOOKASIDE ProvenanceRecord(
# E             i1 = INPUT_FN()
# E             i2 = LOAD_ATTR(i1, '__globals__')
# E             i3 = BINARY_SUBSCR(i2, 'ScaleTensorSubclass')
# E             i4 = LOOKASIDE(i3)
# E           )
#
# thunder/core/jit_ext.py:1503: NotImplementedError
#
# The above exception was the direct cause of the following exception:
#
#     def test_subclass_ctor():
#
#         def f(x: torch.Tensor, scale: torch.Tensor) -> ScaleTensorSubclass:
#             return ScaleTensorSubclass(x, scale)
#
#         device = torch.device("cuda")
#         dtype = torch.float32
#         shape = (2, 2)
#         x = make_tensor(shape, device=device, dtype=dtype)
#         scale = make_tensor((), device=device, dtype=dtype)
#
#         jitted = thunder.jit(f)
#
#         expected = f(x, scale)
# >       actual = jitted(x, scale)
#
# thunder/tests/test_tensor_subclass.py:104:
# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
# thunder/__init__.py:768: in wrapped
#     return fn(*args, **kwargs)
# thunder/__init__.py:818: in fn_
#     cache_entry, inps, pro_to_epi = get_computation_and_inputs(*args, **kwargs)
# thunder/__init__.py:750: in wrapped
#     cache_entry, inps, pro_to_epi = get_computation_and_inputs_fn(*args, **kwargs)
# thunder/core/langctxs.py:136: in _fn
#     result = fn(*args, **kwargs)
# thunder/__init__.py:234: in cache_info_wrapper
#     res = fn(*args, **kwargs)
# thunder/__init__.py:522: in get_computation_and_inputs
#     jit_results: TraceResults = thunder_general_jit(
# thunder/core/jit_ext.py:1788: in thunder_general_jit
#     pro_to_comp_proxies, pro_to_epi_proxies = unpack_inputs(ctx, prologue_trace, pro_to_comp, pro_to_epi, args, kwargs)
# thunder/core/jit_ext.py:1576: in unpack_inputs
#     pro_to_comp = tuple(sorted((unpack(v) for v in pro_to_comp_inps), key=lambda x: param_ordering[id(x)][1]))
# thunder/core/jit_ext.py:1576: in <genexpr>
#     pro_to_comp = tuple(sorted((unpack(v) for v in pro_to_comp_inps), key=lambda x: param_ordering[id(x)][1]))
def test_subclass_ctor():

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
