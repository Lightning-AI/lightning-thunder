from __future__ import annotations
import operator
import importlib
from functools import partial, wraps
from numbers import Number
from typing import TYPE_CHECKING
from collections.abc import Callable
from collections.abc import Hashable, Sequence
from types import ModuleType

import torch

import thunder.core.dtypes as dtypes
from thunder.core.dtypes import to_torch_dtype, to_dtype
import thunder.core.devices as devices
from thunder.core.devices import to_torch_device, to_device
import thunder.core.prims as prims
from thunder.core.proxies import NumberProxy, TensorProxy, pytype
from thunder.core.symbol import Symbol
from thunder.distributed.prims import DistributedReduceOps
import thunder.distributed.prims as dist_prims
import thunder.core.utils as utils

import thunder.torch as ltorch

from thunder.extend import OperatorExecutor, register_executor, add_always_executor

from thunder.core.transforms import (
    get_grad,
    put_grad,
)

if TYPE_CHECKING:
    from thunder.common import CompileData

ex = OperatorExecutor("torch", version=torch.__version__)
register_executor(ex)
add_always_executor(ex)

# Common annotations
TensorLike = TensorProxy
DeviceLike = str | devices.Device | torch.device
dtypeLike = dtypes.dtype | torch.dtype

#
# Helper functions
#


def _always_executable(*args, **kwargs) -> bool:
    return True


def _register_torch_operation(name: str, *, like: None | Symbol = None, module: type | ModuleType = torch) -> Symbol:
    like: Symbol = like if like is not None else getattr(ltorch, name)
    return ex.register_operator(name, like=like, module=module)


def _register_implementation(
    id_or_symbol: Hashable | Symbol,
    op: None | Symbol = None,
    *,
    checker: Callable,
    execution_transform: Callable = None,
):
    ex.register_implementation(id_or_symbol, op, checker=checker, execution_transform=execution_transform)


#
# Data movement operations
#

to = _register_torch_operation("to", module=torch.Tensor)


def _convert_element_type_prim_checker(a: Number | TensorProxy, dtype: dtypes.dtype) -> bool:
    return isinstance(a, TensorProxy)


# NOTE The convert element type primitive is (currently) modeled as always creating a copy
def _convert_element_type_transform(
    a: TensorLike,
    /,
    dtype: dtypes.dtype,
) -> TensorLike:
    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return to(a, torch_dtype, copy=True)


def _to_transform(
    a: TensorLike,
    tensor_dtype_or_device: None | TensorLike | dtypeLike | DeviceLike = None,
    optional_positional_dtype: None | dtypeLike = None,
    /,
    *,
    device: None | DeviceLike = None,
    dtype: None | dtypeLike = None,
    copy: bool = False,
    memory_format: None | torch.memory_format = None,
) -> TensorLike:
    device: None | devices.Device
    dtype: None | dtypes.dtype
    device, dtype = ltorch._parse_to_device_and_dtype(
        tensor_dtype_or_device, optional_positional_dtype, device=device, dtype=dtype
    )

    torch_device: None | torch.device = to_torch_device(device)
    torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

    kwargs = {"copy": copy}
    if torch_device is not None:
        kwargs["device"] = torch_device
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    if memory_format is not None:
        kwargs["memory_format"] = memory_format
    return to(a, **kwargs)


def _device_put_transform(a: TensorProxy, device: devices.Device) -> TensorProxy:
    torch_device: str = device.device_str()
    return to(a, torch_device)


_register_implementation(prims.device_put, checker=_always_executable, execution_transform=_device_put_transform)
_register_implementation(
    prims.convert_element_type,
    checker=_convert_element_type_prim_checker,
    execution_transform=_convert_element_type_transform,
)
_register_implementation(ltorch.to, checker=_always_executable, execution_transform=_to_transform)

#
# Disable torch.autocast operations
#


def no_autocast(fn):
    """
    A decorator that disables torch.autocast for the duration of the decorated
    function.

    In Thunder this is useful when you want to ensure that the generated
    function is not run with PyTorch's autocast enabled to execute exactly as
    generated.

    Args:
        fn: The function to decorate.

    Returns:
        The decorated function.
    """
    # This decorator intentionally does not use the torch.autocast decorator
    # because it is much slower than the implementation here. This is because
    # the torch.autocast decorator has a lot more overhead to support various
    # features that are not needed in Thunder.
    from torch import set_autocast_enabled

    prev_cpu = torch.is_autocast_cpu_enabled()
    prev = torch.is_autocast_enabled()

    @wraps(fn)
    def no_autocast_fn(*args, **kwargs):
        try:
            set_autocast_enabled("cpu", False)
            set_autocast_enabled("cuda", False)
            return fn(*args, **kwargs)
        finally:
            set_autocast_enabled("cpu", prev_cpu)
            set_autocast_enabled("cuda", prev)

    return no_autocast_fn


#
# Tensor creation operations
#
arange = _register_torch_operation("arange")
full = _register_torch_operation("full")
full_like = _register_torch_operation("full_like")
ones = _register_torch_operation("ones")
ones_like = _register_torch_operation("ones_like")
tensor_from_sequence = _register_torch_operation("tensor")
zeros = _register_torch_operation("zeros")
zeros_like = _register_torch_operation("zeros_like")
rand = _register_torch_operation("rand")
randint = _register_torch_operation("randint")
randn = _register_torch_operation("randn")
empty = _register_torch_operation("empty")
einsum = _register_torch_operation("einsum")
clone = _register_torch_operation("clone")


def _uniform_philox_like(
    shape: Sequence[int],
    *,
    stride: None = None,
    device: DeviceLike,
    dtype: dtypeLike,
    seed: TensorProxy,
    offset: TensorProxy,
) -> tuple[TensorLike, TensorLike]:
    random_values = ltorch.uniform_philox(shape, 0.0, 1.0, device=device, dtype=dtype, seed=seed, offset=offset)
    offset: TensorProxy = TensorProxy(shape=(), device=devices.cpu, dtype=dtypes.int64)
    return random_values, offset


uniform_philox = _register_torch_operation("ops.rngprims.philox_rand", like=_uniform_philox_like)


# NOTE We define a custom PyTorch uniform operation, because PyTorch has no out-of-place uniform
def _uniform(
    shape: Sequence[int], minval: Number, maxval: Number, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    t: torch.Tensor = torch.empty(shape, device=device, dtype=dtype)
    t.uniform_(minval, maxval)
    return t


def _uniform_meta(
    shape: Sequence[int], minval: Number, maxval: Number, *, device: torch.device, dtype: torch.dtype
) -> TensorProxy:
    thunder_device = to_device(device)
    thunder_dtype = to_dtype(dtype)

    return TensorProxy(shape=shape, device=thunder_device, dtype=thunder_dtype, requires_grad=False)


uniform = ex.register_operator("uniform", meta=_uniform_meta, fn=_uniform)


def _arange_transform(
    start: Number,
    end: None | Number = None,
    step: Number = 1,
    *,
    device: None | DeviceLike = None,
    dtype: None | dtypeLike = None,
):
    torch_device: None | torch.device = to_torch_device(device)
    torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

    if end is None:
        end = start
        start = 0

    return arange(start=start, step=step, end=end, device=torch_device, dtype=torch_dtype)


# TODO Remove or restore exogenous_like
# def _exogenous_like_helper(likes: Sequence[torch.Tensor], /) -> tuple[torch.Tensor, ...]:
#     return tuple([torch.zeros_like(x) for x in likes])


# def exogenous_like(bsym: BoundSymbol, likes: Sequence[TensorProxy], /) -> BoundSymbol:
#     sym = Symbol(name="exogenous_like", meta=None)
#     ctx: dict[str, Any] = {"exogenous_like": _exogenous_like_helper}

#     return sym.bind(likes, output=bsym.output, _call_ctx=ctx)


def _full_transform(
    shape: Sequence[int], fill_value: Number, *, device: None | devices.Device, dtype: None | dtypes.dtype
) -> TensorProxy:
    torch_device: None | torch.device = to_torch_device(device)
    torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

    return full(shape, fill_value, device=torch_device, dtype=torch_dtype)


def _full_like_transform(
    a: TensorLike, /, fill_value: Number, *, device: None | DeviceLike = None, dtype: None | dtypeLike = None
) -> TensorLike:
    torch_device: None | torch.device = to_torch_device(device)
    torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

    return full_like(a, fill_value=fill_value, device=torch_device, dtype=torch_dtype)


def _ones_transform(*shape: int, device: None | DeviceLike = None, dtype: None | dtypeLike = None) -> TensorLike:
    torch_device: None | torch.device = to_torch_device(device)
    torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

    return ones(*shape, device=torch_device, dtype=torch_dtype)


def _ones_like_transform(
    a: TensorLike, /, *, device: None | DeviceLike = None, dtype: None | dtypeLike = None
) -> TensorLike:
    torch_device: None | torch.device = to_torch_device(device)
    torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

    return ones_like(a, device=torch_device, dtype=torch_dtype)


def _iota_transform(
    length: Number, *, start: Number, step: Number, device: devices.Device, dtype: dtypes.dtype
) -> TensorLike:
    torch_device: torch.device = to_torch_device(device)
    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    end: Number = start + length * step

    return arange(start=start, step=step, end=end, device=torch_device, dtype=torch_dtype)


def _uniform_transform(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypeLike,
) -> TensorLike:
    torch_device: torch.device = to_torch_device(device)
    torch_dtype: torch.dtype = to_torch_dtype(dtype)

    return uniform(shape, minval, maxval, device=torch_device, dtype=torch_dtype)


# NOTE minval == 0. and maxval == 1. due to the checker
def _uniform_philox_prim_transform(
    shape: Sequence[int],
    minval: float,
    maxval: float,
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
) -> TensorLike:
    torch_device = to_torch_device(device)
    torch_dtype = to_torch_dtype(dtype)

    seed_tensor: TensorLike = ltorch.tensor(seed) if isinstance(seed, int) else seed
    offset_tensor: TensorLike = ltorch.tensor(offset) if isinstance(offset, int) else offset

    random_values, offset = uniform_philox(
        shape, stride=None, seed=seed_tensor, offset=offset_tensor, device=torch_device, dtype=torch_dtype
    )

    return random_values


# TODO Consider restricting to seed and offset being tensors, too?
def _uniform_philox_prim_checker(
    shape: Sequence[int],
    minval: float,
    maxval: float,
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
) -> bool:
    if minval != 0 or maxval != 1:
        return False

    if offset % 4 != 0:
        return False

    if device.devicetype != devices.DeviceType.CUDA:
        return False

    return True


# NOTE minval == 0. and maxval == 1. due to the checker
def _uniform_philox_transform(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypeLike,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
) -> TensorLike:
    torch_device = to_torch_device(device)
    torch_dtype = to_torch_dtype(dtype)

    seed_tensor: TensorLike = ltorch.tensor(seed) if isinstance(seed, int) else seed
    offset_tensor: TensorLike = ltorch.tensor(offset) if isinstance(offset, int) else offset

    random_values, offset = uniform_philox(
        shape, stride=None, seed=seed_tensor, offset=offset_tensor, device=torch_device, dtype=torch_dtype
    )

    return random_values


# TODO -- How can we validate that the tensor has the appropriate offset?
def _uniform_philox_checker(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypeLike,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
) -> bool:
    if minval != 0 or maxval != 1:
        return False

    if isinstance(offset, TensorProxy) or offset % 4 != 0:
        return False

    if to_device(device).devicetype != devices.DeviceType.CUDA:
        return False

    return True


def _zeros_transform(*shape: int, device: None | DeviceLike = None, dtype: None | dtypeLike = None) -> TensorLike:
    torch_device: None | torch.device = to_torch_device(device)
    torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

    return zeros(*shape, device=torch_device, dtype=torch_dtype)


def _zeros_like_transform(
    a: TensorLike, /, *, device: None | DeviceLike = None, dtype: None | dtypeLike = None
) -> TensorLike:
    torch_device: None | torch.device = to_torch_device(device)
    torch_dtype: None | torch.dtype = to_torch_dtype(dtype)

    return zeros_like(a, device=torch_device, dtype=torch_dtype)


def _randint_prims_transform(
    low: int,
    high: int,
    shape: tuple[int, ...],
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
) -> TensorLike:
    torch_device: torch.device = to_torch_device(device)
    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return randint(low, high, shape, device=torch_device, dtype=torch_dtype)


def _randn_prims_transform(
    shape: tuple[int, ...],
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
) -> TensorLike:
    torch_device: torch.device = to_torch_device(device)
    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return randn(shape, device=torch_device, dtype=torch_dtype)


def _empty_prims_transform(
    shape: tuple[int, ...],
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
) -> TensorLike:
    torch_device: torch.device = to_torch_device(device)
    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return empty(shape, device=torch_device, dtype=torch_dtype)


def _clone_prims_transform(a: TensorLike, **kwargs) -> TensorLike:
    return clone(a)


def _tensor_from_sequence_prims_transform(
    seq_or_number, *, device: devices.Device, dtype: None | dtypes.dtype
) -> TensorLike:
    torch_device: torch.device = to_torch_device(device)
    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return tensor_from_sequence(seq_or_number, device=torch_device, dtype=torch_dtype)


def _get_and_update_rng_state_impl(seed, offset, device):
    state = torch.cuda.get_rng_state(device)
    seed, offset = torch.chunk(state, 2)
    # We follow the nvFuser way here. The offset used by nvfuser = pytorch_offset // 4
    # See Note [Divide offset by 4] https://github.com/NVIDIA/Fuser/blob/729f36c/csrc/rng.cpp#L54
    seed = seed.view(torch.int64).item()
    offset = offset.view(torch.int64).item() // 4
    # We follow the nvFuser way here. pytorch_new_offset = (nvfuser_offset + 1) * 4
    # See Note [Divide offset by 4] https://github.com/NVIDIA/Fuser/blob/729f36c/csrc/rng.cpp#L54
    new_offset = (offset + 1) * 4
    seed_portion = torch.tensor([seed], device="cpu").view(torch.uint8)
    offset_portion = torch.tensor([new_offset], device="cpu").view(torch.uint8)
    new_state = torch.cat([seed_portion, offset_portion])
    torch.cuda.set_rng_state(new_state, device)
    return seed, offset


get_and_update_rng_state_impl = ex.register_operator(
    "get_and_update_rng_state_impl",
    meta=prims.get_and_update_rng_state.meta,
    fn=_get_and_update_rng_state_impl,
)


_register_implementation(prims.full, checker=_always_executable, execution_transform=_full_transform)
_register_implementation(prims.iota, checker=_always_executable, execution_transform=_iota_transform)
_register_implementation(prims.uniform, checker=_always_executable, execution_transform=_uniform_transform)
_register_implementation(
    prims.uniform_philox, checker=_uniform_philox_prim_checker, execution_transform=_uniform_philox_prim_transform
)
_register_implementation(prims.get_and_update_rng_state, get_and_update_rng_state_impl, checker=_always_executable)
_register_implementation(prims.randint, checker=_always_executable, execution_transform=_randint_prims_transform)
_register_implementation(prims.randn, checker=_always_executable, execution_transform=_randn_prims_transform)
_register_implementation(prims.empty, checker=_always_executable, execution_transform=_empty_prims_transform)
_register_implementation(prims.clone, checker=_always_executable, execution_transform=_clone_prims_transform)
_register_implementation(
    prims.tensor_from_sequence, checker=_always_executable, execution_transform=_tensor_from_sequence_prims_transform
)

_register_implementation(ltorch.arange, checker=_always_executable, execution_transform=_arange_transform)
_register_implementation(ltorch.full, checker=_always_executable, execution_transform=_full_transform)
_register_implementation(ltorch.full_like, checker=_always_executable, execution_transform=_full_like_transform)
_register_implementation(ltorch.ones, checker=_always_executable, execution_transform=_ones_transform)
_register_implementation(ltorch.ones_like, checker=_always_executable, execution_transform=_ones_like_transform)
_register_implementation(ltorch.uniform, checker=_always_executable, execution_transform=_uniform_transform)
_register_implementation(
    ltorch.uniform_philox, checker=_uniform_philox_checker, execution_transform=_uniform_philox_transform
)
_register_implementation(ltorch.zeros, checker=_always_executable, execution_transform=_zeros_transform)
_register_implementation(ltorch.zeros_like, checker=_always_executable, execution_transform=_zeros_like_transform)

#
# Reshaping and permuting operations
#

cat = _register_torch_operation("cat")
chunk = _register_torch_operation("chunk")
diagonal = _register_torch_operation("diagonal")
expand = _register_torch_operation("expand", module=torch.Tensor)
flatten = _register_torch_operation("flatten")
flip = _register_torch_operation("flip")
getitem = _register_torch_operation("__getitem__", like=ltorch.getitem, module=torch.Tensor)
movedim = _register_torch_operation("movedim")
permute = _register_torch_operation("permute")
repeat = _register_torch_operation("repeat", module=torch.Tensor)
reshape = _register_torch_operation("reshape")
select = _register_torch_operation("select")
split = _register_torch_operation("split")
stack = _register_torch_operation("stack")
squeeze = _register_torch_operation("squeeze")
tensor_split = _register_torch_operation("tensor_split")
transpose = _register_torch_operation("transpose")
unbind = _register_torch_operation("unbind")
unfold = _register_torch_operation("unfold", module=torch.Tensor)
unsqueeze = _register_torch_operation("unsqueeze")
view = _register_torch_operation("view", module=torch.Tensor)
view_as = _register_torch_operation("view_as", module=torch.Tensor)
all_tensor = _register_torch_operation("all", like=ltorch.all_tensor)
any_tensor = _register_torch_operation("any", like=ltorch.any_tensor)


def _broadcast_in_dim_prim_transform(
    a: TensorProxy, /, shape: Sequence[int], broadcast_dimensions: Sequence[int]
) -> TensorProxy:
    s = list(shape)

    for broadcast_dim in broadcast_dimensions:
        s[broadcast_dim] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = unsqueeze(v, idx)

    return expand(v, shape)


def _flip_transform(a: TensorLike, /, *dims: int) -> TensorLike:
    dims = utils.extract_shape_from_varargs(dims)
    return flip(a, dims)


def _permute_transform(a: TensorLike, /, *dims: int) -> TensorLike:
    dims = utils.extract_shape_from_varargs(dims)
    return permute(a, dims)


# NOTE The transpose prim is analogous to PyTorch's permute operation, and the argument names do not match
def _transpose_prim_transform(a: TensorProxy, /, permutation: Sequence[int]) -> TensorLike:
    return permute(a, permutation)


def _reshape_transform(a: TensorLike, /, *dims: int) -> TensorLike:
    dims = utils.extract_shape_from_varargs(dims)
    return reshape(a, dims)


# TODO When getitem is fully supported this can be changed to be an execution transform instead of a direct impl
def _slice_prim_impl(
    a: torch.Tensor, start_indices: Sequence[int], end_indices: Sequence[int], strides: None | Sequence[int] = None
) -> torch.Tensor:
    _strides = strides if strides is not None else [1] * len(start_indices)

    slices: list = []
    for start, stop, step in zip(start_indices, end_indices, _strides):
        slices.append(slice(start, stop, step))

    return operator.getitem(a, slices)


# NOTE PyTorch has a bug where it doesn't interpret calls like squeeze(a, None) correctly
def _squeeze_transform(a: TensorLike, /, dim: None | int | Sequence[int] = None) -> TensorLike:
    if dim is None:
        return squeeze(a)
    return squeeze(a, dim)


_register_implementation(
    prims.broadcast_in_dim, checker=_always_executable, execution_transform=_broadcast_in_dim_prim_transform
)
_register_implementation(prims.cat, cat, checker=_always_executable)
_register_implementation(prims.flip, flip, checker=_always_executable)
# NOTE - `ltorch.reshape` short circuits when new shape is same as original shape and returns the input proxy as output.
#        `prims.reshape` doesn't do that and returns a new proxy. So we add `torch_prims_reshape_impl` which is consistent
#        with `prims.reshape` semantics otherwise this can lead incorrectness.
torch_prims_reshape_impl = ex.register_operator("torch_prims_reshape_impl", meta=prims.reshape.meta, fn=torch.reshape)
_register_implementation(prims.reshape, torch_prims_reshape_impl, checker=_always_executable)
slice_prim_impl = ex.register_operator("torch_slice_prim_impl", meta=prims.slice_prim.meta, fn=_slice_prim_impl)
_register_implementation(prims.slice_prim, slice_prim_impl, checker=_always_executable)
_register_implementation(prims.squeeze, checker=_always_executable, execution_transform=_squeeze_transform)
_register_implementation(prims.transpose, checker=_always_executable, execution_transform=_transpose_prim_transform)
_register_implementation(prims.unfold, unfold, checker=_always_executable)
_register_implementation(prims.view, view, checker=_always_executable)

_register_implementation(ltorch.cat, cat, checker=_always_executable)
_register_implementation(ltorch.chunk, chunk, checker=_always_executable)
_register_implementation(ltorch.diagonal, diagonal, checker=_always_executable)
_register_implementation(ltorch.expand, expand, checker=_always_executable)
_register_implementation(ltorch.flatten, flatten, checker=_always_executable)
_register_implementation(ltorch.flip, checker=_always_executable, execution_transform=_flip_transform)
_register_implementation(ltorch.getitem, getitem, checker=_always_executable)
_register_implementation(ltorch.movedim, movedim, checker=_always_executable)
_register_implementation(ltorch.permute, checker=_always_executable, execution_transform=_permute_transform)
_register_implementation(ltorch.repeat, repeat, checker=_always_executable)
_register_implementation(ltorch.reshape, checker=_always_executable, execution_transform=_reshape_transform)
_register_implementation(ltorch.select, select, checker=_always_executable)
_register_implementation(ltorch.split, split, checker=_always_executable)
_register_implementation(ltorch.stack, stack, checker=_always_executable)
_register_implementation(ltorch.squeeze, checker=_always_executable, execution_transform=_squeeze_transform)
_register_implementation(ltorch.tensor_split, tensor_split, checker=_always_executable)
_register_implementation(ltorch.transpose, transpose, checker=_always_executable)
_register_implementation(ltorch.unbind, unbind, checker=_always_executable)
_register_implementation(ltorch.unfold, unfold, checker=_always_executable)
_register_implementation(ltorch.unsqueeze, unsqueeze, checker=_always_executable)
_register_implementation(ltorch.view, view, checker=_always_executable)
_register_implementation(ltorch.view_as, view_as, checker=_always_executable)
_register_implementation(ltorch.all_tensor, all_tensor, checker=_always_executable)
_register_implementation(ltorch.any_tensor, any_tensor, checker=_always_executable)

#
# Memory format operations
#
contiguous = _register_torch_operation("contiguous", module=torch.Tensor)


# TODO Detect if the tensor is already contiguous as requested, and if so just return it
# TODO Review how strides are set if the tensor contains no elements
def _stride_order_prim_impl(a: torch.Tensor, order: Sequence[int]) -> torch.Tensor:
    # Canonicalizes permutation as a tuple so it can be compared to the channels_last special cases below
    order = tuple(order)

    # Special cases channels_last and channels_last_3d cases
    if order == (3, 0, 2, 1):
        return a.contiguous(memory_format=torch.channels_last)
    elif order == (4, 0, 3, 2, 1):
        return a.contiguous(memory_format=torch.channels_last_3d)

    # Creates a tensor with the appropriate shape and strides, then copies the input
    #   tensor into it
    ordered_dims = sorted(zip(a.shape, order), key=lambda x: x[1])
    ordered_strides = [1]
    accum = ordered_dims[0][0]
    for dim_length, _ in ordered_dims[1:]:
        ordered_strides.append(accum)
        accum *= dim_length

    strides = tuple(ordered_strides[x] for x in order)
    return torch.empty_strided(a.shape, strides, device=a.device, dtype=a.dtype).copy_(a)


stride_order_prim_impl = ex.register_operator(
    "torch_stride_order_prim_impl", meta=prims.stride_order.meta, fn=_stride_order_prim_impl
)
_register_implementation(prims.stride_order, stride_order_prim_impl, checker=_always_executable)
_register_implementation(ltorch.contiguous, contiguous, checker=_always_executable)


#
# Elementwise unary operations
#

# NOTE torch_abs to avoid a conflict with Python's builtin abs()
torch_abs = _register_torch_operation("abs")
acos = _register_torch_operation("acos")
acosh = _register_torch_operation("acosh")
asin = _register_torch_operation("asin")
asinh = _register_torch_operation("asinh")
atan = _register_torch_operation("atan")
atanh = _register_torch_operation("atanh")
bitwise_not = _register_torch_operation("bitwise_not")
ceil = _register_torch_operation("ceil")
cos = _register_torch_operation("cos")
cosh = _register_torch_operation("cosh")
digamma = _register_torch_operation("digamma")
erf = _register_torch_operation("erf")
erfc = _register_torch_operation("erfc")
erfinv = _register_torch_operation("erfinv")
exp = _register_torch_operation("exp")
exp2 = _register_torch_operation("exp2")
expm1 = _register_torch_operation("expm1")
floor = _register_torch_operation("floor")
frexp = _register_torch_operation("frexp")
isfinite = _register_torch_operation("isfinite")
isinf = _register_torch_operation("isinf")
isnan = _register_torch_operation("isnan")
lgamma = _register_torch_operation("lgamma")
log = _register_torch_operation("log")
log10 = _register_torch_operation("log10")
log1p = _register_torch_operation("log1p")
log2 = _register_torch_operation("log2")
# TODO Update ndtri to be like thudner.torch...ndtri when it's available
ndtri = _register_torch_operation("ndtri", like=prims.ndtri, module=torch.special)
neg = _register_torch_operation("neg")
reciprocal = _register_torch_operation("reciprocal")
# # NOTE torch_round to avoid a name conflict with the builtin round
torch_round = _register_torch_operation("round")
rsqrt = _register_torch_operation("rsqrt")
# # NOTE That PyTorch's "sgn" corresponds with the "sign" primitive
sgn = _register_torch_operation("sgn", like=ltorch.sign)
# # NOTE torch.sign isn't bound here because thunder always uses sgn
# sign =  _register_torch_operation("sign")
signbit = _register_torch_operation("signbit")
sin = _register_torch_operation("sin")
sinh = _register_torch_operation("sinh")
sqrt = _register_torch_operation("sqrt")
tan = _register_torch_operation("tan")
tanh = _register_torch_operation("tanh")
trunc = _register_torch_operation("trunc")
real = _register_torch_operation("real")
imag = _register_torch_operation("imag")


def _elementwise_unary_checker(a: Number | TensorLike) -> bool:
    return isinstance(a, TensorLike)


# NOTE PyTorch doesn't have an erfcinv implementation
def _erfcinv_impl(a: torch.Tensor) -> torch.Tensor:
    return torch.erfinv(1 - a)


_register_elementwise_unary_implementation = partial(_register_implementation, checker=_elementwise_unary_checker)

_register_elementwise_unary_implementation(prims.abs, torch_abs)
_register_elementwise_unary_implementation(prims.acos, acos)
_register_elementwise_unary_implementation(prims.acosh, acosh)
_register_elementwise_unary_implementation(prims.asin, asin)
_register_elementwise_unary_implementation(prims.asinh, asinh)
_register_elementwise_unary_implementation(prims.atan, atan)
_register_elementwise_unary_implementation(prims.atanh, atanh)
_register_elementwise_unary_implementation(prims.bitwise_not, bitwise_not)
_register_elementwise_unary_implementation(prims.ceil, ceil)
_register_elementwise_unary_implementation(prims.cos, cos)
_register_elementwise_unary_implementation(prims.cosh, cosh)
_register_elementwise_unary_implementation(prims.digamma, digamma)
_register_elementwise_unary_implementation(prims.erf, erf)
_register_elementwise_unary_implementation(prims.erfc, erfc)
erfcinv = ex.register_operator("torch_erfcinv_impl", meta=prims.erfcinv, fn=_erfcinv_impl)
_register_elementwise_unary_implementation(prims.erfcinv, erfcinv)
_register_elementwise_unary_implementation(prims.erfinv, erfinv)
_register_elementwise_unary_implementation(prims.exp, exp)
_register_elementwise_unary_implementation(prims.exp2, exp2)
_register_elementwise_unary_implementation(prims.expm1, expm1)
_register_elementwise_unary_implementation(prims.floor, floor)
_register_elementwise_unary_implementation(prims.isfinite, isfinite)
_register_elementwise_unary_implementation(prims.lgamma, lgamma)
_register_elementwise_unary_implementation(prims.log, log)
_register_elementwise_unary_implementation(prims.log10, log10)
_register_elementwise_unary_implementation(prims.log1p, log1p)
_register_elementwise_unary_implementation(prims.log2, log2)
_register_elementwise_unary_implementation(prims.ndtri, ndtri)
_register_elementwise_unary_implementation(prims.neg, neg)
_register_elementwise_unary_implementation(prims.reciprocal, reciprocal)
_register_elementwise_unary_implementation(prims.round, torch_round)
_register_elementwise_unary_implementation(prims.rsqrt, rsqrt)
_register_elementwise_unary_implementation(prims.sign, sgn)
_register_elementwise_unary_implementation(prims.signbit, signbit)
_register_elementwise_unary_implementation(prims.sin, sin)
_register_elementwise_unary_implementation(prims.sinh, sinh)
_register_elementwise_unary_implementation(prims.sqrt, sqrt)
_register_elementwise_unary_implementation(prims.tan, tan)
_register_elementwise_unary_implementation(prims.tanh, tanh)
_register_elementwise_unary_implementation(prims.trunc, trunc)
_register_elementwise_unary_implementation(prims.real, real)
_register_elementwise_unary_implementation(prims.imag, imag)

_register_elementwise_unary_implementation(ltorch.abs, torch_abs)
_register_elementwise_unary_implementation(ltorch.acos, acos)
_register_elementwise_unary_implementation(ltorch.acosh, acosh)
_register_elementwise_unary_implementation(ltorch.asin, asin)
_register_elementwise_unary_implementation(ltorch.asinh, asinh)
_register_elementwise_unary_implementation(ltorch.atan, atan)
_register_elementwise_unary_implementation(ltorch.atanh, atanh)
_register_elementwise_unary_implementation(ltorch.bitwise_not, bitwise_not)
_register_elementwise_unary_implementation(ltorch.ceil, ceil)
_register_elementwise_unary_implementation(ltorch.cos, cos)
_register_elementwise_unary_implementation(ltorch.cosh, cosh)
_register_elementwise_unary_implementation(ltorch.digamma, digamma)
_register_elementwise_unary_implementation(ltorch.erf, erf)
_register_elementwise_unary_implementation(ltorch.erfc, erfc)
_register_elementwise_unary_implementation(ltorch.erfinv, erfinv)
_register_elementwise_unary_implementation(ltorch.exp, exp)
_register_elementwise_unary_implementation(ltorch.exp2, exp2)
_register_elementwise_unary_implementation(ltorch.expm1, expm1)
_register_elementwise_unary_implementation(ltorch.floor, floor)
_register_elementwise_unary_implementation(ltorch.isfinite, isfinite)
_register_elementwise_unary_implementation(ltorch.isinf, isinf)
_register_elementwise_unary_implementation(ltorch.isnan, isnan)
_register_elementwise_unary_implementation(ltorch.lgamma, lgamma)
_register_elementwise_unary_implementation(ltorch.log, log)
_register_elementwise_unary_implementation(ltorch.log10, log10)
_register_elementwise_unary_implementation(ltorch.log1p, log1p)
_register_elementwise_unary_implementation(ltorch.log2, log2)
# TODO Update ndtri when it's added back to thunder.torch...
# _register_elementwise_unary_implementation(ltorch.ndtri, ndtri)
_register_elementwise_unary_implementation(ltorch.neg, neg)
_register_elementwise_unary_implementation(ltorch.reciprocal, reciprocal)
_register_elementwise_unary_implementation(ltorch.round, torch_round)
_register_elementwise_unary_implementation(ltorch.rsqrt, rsqrt)
_register_elementwise_unary_implementation(ltorch.sign, sgn)
_register_elementwise_unary_implementation(ltorch.signbit, signbit)
_register_elementwise_unary_implementation(ltorch.sin, sin)
_register_elementwise_unary_implementation(ltorch.sin, sin)
_register_elementwise_unary_implementation(ltorch.sqrt, sqrt)
_register_elementwise_unary_implementation(ltorch.tan, tan)
_register_elementwise_unary_implementation(ltorch.tanh, tanh)
_register_elementwise_unary_implementation(ltorch.trunc, trunc)
_register_elementwise_unary_implementation(ltorch.real, real)
_register_elementwise_unary_implementation(ltorch.imag, imag)


def _frexp_transform(a: TensorProxy):
    return frexp(a)


_register_implementation(prims.frexp, checker=_always_executable, execution_transform=_frexp_transform)

_register_implementation(ltorch.frexp, checker=_always_executable, execution_transform=_frexp_transform)


# nn.functional elementwise unary
celu = _register_torch_operation("celu", module=torch.nn.functional)
elu = _register_torch_operation("elu", module=torch.nn.functional)
gelu = _register_torch_operation("gelu", module=torch.nn.functional)
hardsigmoid = _register_torch_operation("hardsigmoid", module=torch.nn.functional)
hardshrink = _register_torch_operation("hardshrink", module=torch.nn.functional)
hardswish = _register_torch_operation("hardswish", module=torch.nn.functional)
hardtanh = _register_torch_operation("hardtanh", module=torch.nn.functional)
leaky_relu = _register_torch_operation("leaky_relu", module=torch.nn.functional)
logsigmoid = _register_torch_operation("logsigmoid", module=torch.nn.functional)
mish = _register_torch_operation("mish", module=torch.nn.functional)
prelu = _register_torch_operation("prelu", module=torch.nn.functional)
relu = _register_torch_operation("relu", module=torch.nn.functional)
relu6 = _register_torch_operation("relu6", module=torch.nn.functional)
rrelu = _register_torch_operation("rrelu", module=torch.nn.functional)
selu = _register_torch_operation("selu", module=torch.nn.functional)
silu = _register_torch_operation("silu", module=torch.nn.functional)
softplus = _register_torch_operation("softplus", module=torch.nn.functional)
softshrink = _register_torch_operation("softshrink", module=torch.nn.functional)
softsign = _register_torch_operation("softsign", module=torch.nn.functional)
tanhshrink = _register_torch_operation("tanhshrink", module=torch.nn.functional)
threshold = _register_torch_operation("threshold", module=torch.nn.functional)


def _elementwise_unary_with_inplace_checker(a: TensorProxy, /, inplace: bool = False) -> bool:
    return isinstance(a, TensorProxy) and not inplace


_register_elementwise_unary_implementation(ltorch.elu, elu, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.celu, celu, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.gelu, gelu, checker=_always_executable)
_register_elementwise_unary_implementation(
    ltorch.hardsigmoid, hardsigmoid, checker=_elementwise_unary_with_inplace_checker
)
_register_elementwise_unary_implementation(ltorch.hardshrink, hardshrink, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.hardswish, hardswish, checker=_elementwise_unary_with_inplace_checker)
_register_elementwise_unary_implementation(ltorch.hardtanh, hardtanh, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.leaky_relu, leaky_relu, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.mish, mish, checker=_elementwise_unary_with_inplace_checker)
_register_elementwise_unary_implementation(ltorch.prelu, prelu, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.relu, relu, checker=_elementwise_unary_with_inplace_checker)
_register_elementwise_unary_implementation(ltorch.relu6, relu6, checker=_elementwise_unary_with_inplace_checker)
_register_elementwise_unary_implementation(ltorch.rrelu, rrelu, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.selu, selu, checker=_elementwise_unary_with_inplace_checker)
_register_elementwise_unary_implementation(ltorch.silu, silu, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.softplus, softplus, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.softshrink, softshrink, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.softsign, softsign, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.tanhshrink, tanhshrink, checker=_always_executable)
_register_elementwise_unary_implementation(ltorch.threshold, threshold, checker=_always_executable)


#
# Elementwise binary operations
#
# TODO Review type promotion differences

add = _register_torch_operation("add")
atan2 = _register_torch_operation("atan2")
bitwise_and = _register_torch_operation("bitwise_and")
bitwise_or = _register_torch_operation("bitwise_or")
bitwise_xor = _register_torch_operation("bitwise_xor")
copysign = _register_torch_operation("copysign")
eq = _register_torch_operation("eq")
floor_divide = _register_torch_operation("floor_divide")
fmod = _register_torch_operation("fmod")
ge = _register_torch_operation("ge")
gt = _register_torch_operation("gt")
logical_and = _register_torch_operation("logical_and")
logical_or = _register_torch_operation("logical_or")
logical_xor = _register_torch_operation("logical_xor")
ldexp = _register_torch_operation("ldexp")
le = _register_torch_operation("le")
lt = _register_torch_operation("lt")
maximum = _register_torch_operation("maximum")
minimum = _register_torch_operation("minimum")
mul = _register_torch_operation("mul")
ne = _register_torch_operation("ne")
nextafter = _register_torch_operation("nextafter")
polygamma = _register_torch_operation("polygamma", module=torch.special)
pow = _register_torch_operation("pow")
remainder = _register_torch_operation("remainder")
sub = _register_torch_operation("sub")
true_divide = _register_torch_operation("true_divide")
zeta = _register_torch_operation("zeta", module=torch.special)
div = _register_torch_operation("div")
bitwise_left_shift = _register_torch_operation("bitwise_left_shift")
bitwise_right_shift = _register_torch_operation("bitwise_right_shift")


# NOTE PyTorch elementwise operations require at least one input to be a tensor
def _elementwise_binary_checker(a: Number | TensorProxy, b: Number | TensorProxy) -> bool:
    return isinstance(a, TensorLike) or isinstance(b, TensorLike)


def _div_checker(
    a: Number | TensorProxy,
    b: Number | TensorProxy,
    *,
    rounding_mode: None | str = None,
    out: None | TensorProxy = None,
) -> TensorProxy:
    return _elementwise_binary_checker(a, b) and (rounding_mode is None or isinstance(rounding_mode, str))


# NOTE add and sub have special check and factory functions to support alpha
def _add_sub_checker(
    a: Number | TensorProxy, b: Number | TensorProxy, *, alpha: None | Number | TensorProxy = None
) -> bool:
    return _elementwise_binary_checker(a, b) and (alpha is None or isinstance(alpha, Number))


# NOTE add and sub have a custom execution transform because the torch operations don't support alpha=None
def _add_transform(
    a: Number | TensorProxy, b: Number | TensorProxy, *, alpha: None | Number | TensorProxy = None
) -> TensorProxy:
    if alpha is None:
        return add(a, b)

    return add(a, b, alpha=alpha)


# Maps exact inputs to truncation division
def _div_prim_impl(a: Number | torch.Tensor, b: Number | torch.Tensor) -> torch.Tensor:
    def is_exact_number_or_exact_tensor(x):
        if isinstance(x, (int, bool)):
            return True
        # We use PyTorch's dtype attribute (instead of `dtypes.is_exact_dtype`) so that torch.compile on prims.div works.
        elif isinstance(x, torch.Tensor) and (not x.dtype.is_complex and not x.dtype.is_floating_point):
            return True
        return False

    if is_exact_number_or_exact_tensor(a) and is_exact_number_or_exact_tensor(b):
        return torch.div(a, b, rounding_mode="trunc")

    return torch.true_divide(a, b)


# NOTE add and sub have a custom execution transform because the torch operations don't support alpha=None
def _sub_transform(a: Number | TensorProxy, b: Number | TensorProxy, *, alpha: None | Number = None) -> TensorProxy:
    if alpha is None:
        return sub(a, b)

    return sub(a, b, alpha=alpha)


def _div_transform(
    a: Number | TensorProxy,
    b: Number | TensorProxy,
    /,
    *,
    rounding_mode: None | str = None,
    out: None | TensorProxy = None,
) -> TensorProxy:
    if rounding_mode is None:
        return div(a, b)

    return div(a, b, rounding_mode=rounding_mode)


_register_elementwise_binary_implementation = partial(_register_implementation, checker=_elementwise_binary_checker)

_register_elementwise_binary_implementation(prims.add, add)
_register_elementwise_binary_implementation(prims.atan2, atan2)
_register_elementwise_binary_implementation(prims.bitwise_and, bitwise_and)
_register_elementwise_binary_implementation(prims.bitwise_or, bitwise_or)
_register_elementwise_binary_implementation(prims.bitwise_xor, bitwise_xor)
div_prim_impl = ex.register_operator("torch_div_prim_impl", meta=prims.div.meta, fn=_div_prim_impl)
_register_elementwise_binary_implementation(prims.div, div_prim_impl)
_register_elementwise_binary_implementation(prims.eq, eq)
_register_elementwise_binary_implementation(prims.fmod, fmod)
_register_elementwise_binary_implementation(prims.ge, ge)
_register_elementwise_binary_implementation(prims.gt, gt)
_register_elementwise_binary_implementation(prims.le, le)
_register_elementwise_binary_implementation(prims.lt, lt)
_register_elementwise_binary_implementation(prims.maximum, maximum)
_register_elementwise_binary_implementation(prims.minimum, minimum)
_register_elementwise_binary_implementation(prims.mul, mul)
_register_elementwise_binary_implementation(prims.ne, ne)
_register_elementwise_binary_implementation(prims.nextafter, nextafter)
_register_elementwise_binary_implementation(prims.pow, pow)
_register_elementwise_binary_implementation(prims.remainder, remainder)
_register_elementwise_binary_implementation(prims.sub, sub)
_register_elementwise_binary_implementation(prims.zeta, zeta)
_register_elementwise_binary_implementation(prims.bitwise_left_shift, bitwise_left_shift)
_register_elementwise_binary_implementation(prims.bitwise_right_shift, bitwise_right_shift)

_register_elementwise_binary_implementation(ltorch.add, checker=_add_sub_checker, execution_transform=_add_transform)
_register_elementwise_binary_implementation(ltorch.atan2, atan2)
_register_elementwise_binary_implementation(ltorch.bitwise_and, bitwise_and)
_register_elementwise_binary_implementation(ltorch.bitwise_or, bitwise_or)
_register_elementwise_binary_implementation(ltorch.bitwise_xor, bitwise_xor)
_register_elementwise_binary_implementation(ltorch.copysign, copysign)
_register_elementwise_binary_implementation(ltorch.eq, eq)
_register_elementwise_binary_implementation(ltorch.floor_divide, floor_divide)
_register_elementwise_binary_implementation(ltorch.fmod, fmod)
_register_elementwise_binary_implementation(ltorch.ge, ge)
_register_elementwise_binary_implementation(ltorch.gt, gt)
_register_elementwise_binary_implementation(ltorch.logical_and, logical_and)
_register_elementwise_binary_implementation(ltorch.logical_or, logical_or)
_register_elementwise_binary_implementation(ltorch.logical_xor, logical_xor)
_register_elementwise_binary_implementation(ltorch.ldexp, ldexp)
_register_elementwise_binary_implementation(ltorch.le, le)
_register_elementwise_binary_implementation(ltorch.lt, lt)
_register_elementwise_binary_implementation(ltorch.maximum, maximum)
_register_elementwise_binary_implementation(ltorch.minimum, minimum)
_register_elementwise_binary_implementation(ltorch.mul, mul)
_register_elementwise_binary_implementation(ltorch.ne, ne)
_register_elementwise_binary_implementation(ltorch.nextafter, nextafter)
_register_elementwise_binary_implementation(ltorch.polygamma, polygamma)
_register_elementwise_binary_implementation(ltorch.pow, pow)
_register_elementwise_binary_implementation(ltorch.remainder, remainder)
_register_elementwise_binary_implementation(ltorch.sub, checker=_add_sub_checker, execution_transform=_sub_transform)
_register_elementwise_binary_implementation(ltorch.true_divide, true_divide)
_register_elementwise_binary_implementation(ltorch.zeta, zeta)
_register_elementwise_binary_implementation(ltorch.div, checker=_div_checker, execution_transform=_div_transform)
_register_elementwise_binary_implementation(ltorch.bitwise_left_shift, bitwise_left_shift)
_register_elementwise_binary_implementation(ltorch.bitwise_right_shift, bitwise_right_shift)

#
# Elementwise ternary operations
#

addcdiv = _register_torch_operation("addcdiv")
addcmul = _register_torch_operation("addcmul")
lerp = _register_torch_operation("lerp")


def _addcdiv_checker(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> bool:
    # PyTorch doesn't support Integer division with torch.addcdiv
    if dtypes.is_exact_dtype(dtypes.to_dtype(b)) and dtypes.is_exact_dtype(dtypes.to_dtype(c)):
        return False

    # PyTorch doesn't support complex32, float16, or bfloat16 addcdiv
    common_dtype, _ = utils.elementwise_type_promotion(
        a, b, c, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    common_dtype = dtypes.to_strong_dtype(common_dtype)
    not_supported_types = (dtypes.complex32, dtypes.float16, dtypes.bfloat16)
    if common_dtype in not_supported_types:
        return False

    # PyTorch requires that value can be safely cast to the common dtype for its type promotion to agree with thunder's
    if value is not None:
        if not utils.can_safe_cast_to(cast_from=utils.to_dtype(value), cast_to=common_dtype):
            return False

    return True


# NOTE PyTorch addcdiv doesn't accept value = None
def _addcdiv_transform(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> TensorLike:
    if value is None:
        return addcdiv(a, b, c)
    return addcdiv(a, b, c, value=value)


def _addcmul_checker(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> bool:
    common_dtype, _ = utils.elementwise_type_promotion(
        a, b, c, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    common_dtype: dtypes.dtype = dtypes.to_strong_dtype(common_dtype)

    # PyTorch doesn't support bool or complex32 inputs for torch.addcmul
    if dtypes.is_boolean_dtype(common_dtype) or common_dtype is dtypes.complex32:
        return False

    # PyTorch requires that value can be safely cast to the common dtype for its type promotion to agree with thunder's
    if value is not None:
        if not utils.can_safe_cast_to(cast_from=utils.to_dtype(value), cast_to=common_dtype):
            return False

    return True


# NOTE PyTorch's addcmul doesn't accept value = None
def _addcmul_transform(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> TensorLike:
    if value is None:
        return addcmul(a, b, c)
    return addcmul(a, b, c, value=value)


def _lerp_checker(start: TensorLike, end: TensorLike, weight: Number | TensorLike) -> TensorLike:
    return (
        isinstance(start, TensorLike)
        and isinstance(end, TensorLike)
        and isinstance(weight, (Number, NumberProxy, TensorLike))
    )


_register_implementation(ltorch.addcdiv, checker=_addcdiv_checker, execution_transform=_addcdiv_transform)
_register_implementation(ltorch.addcmul, checker=_addcmul_checker, execution_transform=_addcmul_transform)
_register_implementation(ltorch.lerp, lerp, checker=_lerp_checker)

#
# Conditional operations
#

clamp = _register_torch_operation("clamp")
where = _register_torch_operation("where")
masked_fill = _register_torch_operation("masked_fill", module=torch.Tensor)
tril = _register_torch_operation("tril")
triu = _register_torch_operation("triu")


def _where_prim_checker(pred: Number | TensorProxy, a: Number | TensorProxy, b: Number | TensorProxy) -> bool:
    return isinstance(pred, TensorProxy)


# NOTE masked_fill's mask but be a boolean tensor, and value must be a number or a number tensor that can be safely cast to
#   the dtype of a, because the result will have the dtype of a
# NOTE PyTorch's checking of whether a number can be safely cast to a different type is based on the number's value, not
#   the type of the number as is done here
def _masked_fill_checker(a: TensorLike, /, mask: TensorLike, value: Number | TensorLike) -> bool:
    if not dtypes.is_boolean_dtype(mask.dtype):
        return False

    value_dtype: type | dtypes.dtype
    if isinstance(value, (Number, NumberProxy)):
        value_dtype = pytype(value)
    else:
        if len(value.shape) != 0:
            return False
        value_dtype = value.dtype

    return utils.can_safe_cast_to(cast_from=value_dtype, cast_to=a.dtype)


# NOTE PyTorch's tril does not have a fill_value parameter
def _tril_checker(a: TensorLike, /, diagonal: int = 0, *, fill_value: None | Number = None) -> bool:
    return fill_value is None


def _tril_transform(a: TensorLike, /, diagonal: int = 0, *, fill_value: None | Number = None) -> TensorLike:
    return tril(a, diagonal)


# NOTE PyTorch's triu like tril does not have a fill_value parameter
def _triu_checker(a: TensorLike, /, diagonal: int = 0, *, fill_value: None | Number = None) -> bool:
    return fill_value is None


def _triu_transform(a: TensorLike, /, diagonal: int = 0, *, fill_value: None | Number = None) -> TensorLike:
    return triu(a, diagonal)


_register_implementation(prims.where, where, checker=_where_prim_checker)

_register_implementation(ltorch.clamp, clamp, checker=_always_executable)
_register_implementation(ltorch.masked_fill, masked_fill, checker=_masked_fill_checker)
_register_implementation(ltorch.tril, checker=_tril_checker, execution_transform=_tril_transform)
_register_implementation(ltorch.triu, checker=_triu_checker, execution_transform=_triu_transform)
_register_implementation(ltorch.where, where, checker=_always_executable)


#
# Reduction operations
#


amax = _register_torch_operation("amax")
amin = _register_torch_operation("amin")
mean = _register_torch_operation("mean")
# NOTE prod is from torch._refs because torch.prod has bugs parsing its input properly
prod = _register_torch_operation("prod", module=torch._refs)
sum = _register_torch_operation("sum")
cumsum = _register_torch_operation("cumsum")
var = _register_torch_operation("var")
var_mean = _register_torch_operation("var_mean")
std = _register_torch_operation("std")
argmax = _register_torch_operation("argmax")
argmin = _register_torch_operation("argmin")
topk = _register_torch_operation("topk")
atleast_1d = _register_torch_operation("atleast_1d")
atleast_2d = _register_torch_operation("atleast_2d")
atleast_3d = _register_torch_operation("atleast_3d")


#
# Sort and dim permutations operations
#


sort = _register_torch_operation("sort")
argsort = _register_torch_operation("argsort")


# NOTE The following transforms are necessary because thunder uses the parameter name 'dims' while PyTorch
#   uses 'dim'
def _amax_prim_transform(a: TensorProxy, /, dims: Sequence[int]) -> TensorProxy:
    return amax(a, dims)


def _amin_prim_transform(a: TensorProxy, /, dims: Sequence[int]) -> TensorProxy:
    return amin(a, dims)


def _prod_prim_transform(a: TensorProxy, /, dims: Sequence[int]) -> TensorProxy:
    return prod(a, dims)


def _sum_prim_transform(a: TensorProxy, /, dims: Sequence[int]) -> TensorProxy:
    return sum(a, dims)


def _var_prim_transform(a: TensorProxy, /, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    return var(a, dims, correction=correction)


def _var_mean_prim_transform(a: TensorProxy, /, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    return var_mean(a, dims, correction=correction)


def _std_prim_transform(a: TensorProxy, /, dims: Sequence[int], *, correction: Number) -> TensorProxy:
    return std(a, dims, correction=correction)


def _cumsum_transform(a: TensorProxy, dim: int, *, dtype: None | dtypeLike = None) -> TensorProxy:
    if dtype is None:
        return cumsum(a, dim)

    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return cumsum(a, dim, dtype=torch_dtype)


def _argmax_transform(a: TensorProxy, /, dim: int):
    return argmax(a, dim)


def _argmin_transform(a: TensorProxy, /, dim: int):
    return argmin(a, dim)


# NOTE This transform translates number proxies to boolean values
# and handles dim = None
def _topk_transform(a: TensorProxy, /, k: int, dim: int | None = None, largest: Number = 1, sorted: Number = 1):
    if dim is None:
        dim = a.ndim - 1 if a.ndim > 0 else 0

    return topk(a, k, dim, bool(largest), bool(sorted))


_register_implementation(prims.amax, checker=_always_executable, execution_transform=_amax_prim_transform)
_register_implementation(prims.amin, checker=_always_executable, execution_transform=_amin_prim_transform)
_register_implementation(prims.prod, checker=_always_executable, execution_transform=_prod_prim_transform)
_register_implementation(prims.sum, checker=_always_executable, execution_transform=_sum_prim_transform)
_register_implementation(prims.var, checker=_always_executable, execution_transform=_var_prim_transform)
_register_implementation(prims.var_mean, checker=_always_executable, execution_transform=_var_mean_prim_transform)
_register_implementation(prims.std, checker=_always_executable, execution_transform=_std_prim_transform)
_register_implementation(prims.argmax, checker=_always_executable, execution_transform=_argmax_transform)
_register_implementation(prims.argmin, checker=_always_executable, execution_transform=_argmin_transform)
_register_implementation(prims.topk, checker=_always_executable, execution_transform=_topk_transform)

_register_implementation(ltorch.amax, amax, checker=_always_executable)
_register_implementation(ltorch.amin, amin, checker=_always_executable)
_register_implementation(ltorch.mean, mean, checker=_always_executable)
_register_implementation(ltorch.prod, prod, checker=_always_executable)
_register_implementation(ltorch.sum, sum, checker=_always_executable)
_register_implementation(ltorch.cumsum, checker=_always_executable, execution_transform=_cumsum_transform)
_register_implementation(ltorch.var, var, checker=_always_executable)
_register_implementation(ltorch.var_mean, var_mean, checker=_always_executable)
_register_implementation(ltorch.std, std, checker=_always_executable)
_register_implementation(ltorch.argmax, argmax, checker=_always_executable)
_register_implementation(ltorch.argmin, argmin, checker=_always_executable)
_register_implementation(ltorch.topk, topk, checker=_always_executable, execution_transform=_topk_transform)
_register_implementation(ltorch.atleast_1d, atleast_1d, checker=_always_executable)
_register_implementation(ltorch.atleast_2d, atleast_2d, checker=_always_executable)
_register_implementation(ltorch.atleast_3d, atleast_3d, checker=_always_executable)


#
# Sort and dim permutations operations
#


# NOTE this transform translates number proxies to boolean values
# and handles dim = None
def _sort_transform(a: TensorProxy, /, dim: int | None = None, descending: bool = False, stable: bool = False):
    if dim is None:
        dim = a.ndim - 1 if a.ndim > 0 else 0

    # NOTE: args past `a` are passed as kwargs to avoid issues with multiple `torch.sort` overloadings
    return sort(a, dim=dim, descending=bool(descending), stable=bool(stable))


_register_implementation(prims.sort, checker=_always_executable, execution_transform=_sort_transform)

_register_implementation(ltorch.sort, checker=_always_executable, execution_transform=_sort_transform)


def _argsort_transform(a: TensorProxy, /, dim: int | None = None, descending: bool = False, stable: bool = False):
    """Transforms argsort operation for execution in torch executor.

    Args:
        a: Input tensor
        dim: Dimension to sort along (defaults to last dim if None)
        descending: Sort in descending order if True
        stable: Use stable sorting algorithm if True
    """
    # NOTE: args past `a` are passed as kwargs to avoid issues with multiple `torch.argsort` overloadings
    return argsort(a, dim=dim, descending=descending, stable=stable)


# Register the implementation
_register_implementation(prims.argsort, checker=_always_executable, execution_transform=_argsort_transform)

_register_implementation(ltorch.argsort, checker=_always_executable, execution_transform=_argsort_transform)

#
# Scatter and gather operations
#

gather = _register_torch_operation("gather")
index_add = _register_torch_operation("index_add")
index_copy = _register_torch_operation("index_copy")
index_put = _register_torch_operation("index_put")
scatter = _register_torch_operation("scatter")
scatter_add = _register_torch_operation("scatter_add")
index_select = _register_torch_operation("index_select")
take_along_dim = _register_torch_operation("take_along_dim")


# NOTE PyTorch has a different order for and names of the parameters
def _index_add_prim_transform(a: TensorProxy, /, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    return index_add(a, dim, index, value)


# NOTE PyTorch has a different order for and names of the parameters
def _index_copy_prim_transform(a: TensorProxy, /, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    return index_copy(a, dim, index, value)


def _index_put_prim_transform(
    a: TensorLike, /, indices: Sequence[TensorLike], values: TensorLike, accumulate: bool = False
) -> TensorLike:
    return index_put(a, indices, values, accumulate)


def _gather_prim_transform(a: TensorProxy, /, index: TensorProxy, dim: int) -> TensorProxy:
    return gather(a, dim, index)


def _gather_transform(a: TensorLike, /, dim: int, index: TensorLike) -> TensorLike:
    return gather(a, dim, index)


def _scatter_prim_transform(a: TensorProxy, /, index: TensorProxy, src: TensorProxy, dim: int) -> TensorProxy:
    return scatter(a, dim, index, src)


def _scatter_transform(
    a: TensorProxy,
    /,
    dim: int,
    index: TensorProxy,
    src: TensorProxy | None = None,
    *,
    value: Number | None = None,
    reduce: None | str = None,
) -> TensorProxy:
    utils.check(
        reduce is None, lambda: "scatter: `reduce` argument other than None is not supported", NotImplementedError
    )

    utils.check(
        (src is not None) ^ (value is not None),
        lambda: "scatter: only one of the arguments ('src', 'value') can be non-None",
    )

    if src is not None:
        return scatter(a, dim, index, src)
    else:
        return scatter(a, dim, index, value)


# NOTE torch.compile has a compilation issue with scatter add in bfloat16,
#      hence the special case here.
def _scatter_add_prim_transform(a: TensorProxy, /, index: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    if a.dtype == dtypes.bfloat16:
        a = a.to(torch.float32)
        value = value.to(torch.float32)
        result = scatter_add(a, dim, index, value)
        return result.to(torch.bfloat16)
    return scatter_add(a, dim, index, value)


# NOTE torch.compile has a compilation issue with scatter add in bfloat16,
#      hence the special case here.
def _scatter_add_transform(a: TensorLike, /, dim: int, index: TensorLike, src: TensorLike) -> TensorLike:
    # NOTE scatter_add does not participate in type promotion, so if a has the bfloat16 dtype, then so does src
    if a.dtype == dtypes.bfloat16:
        a = a.to(torch.float32)
        src = src.to(torch.float32)
        result = scatter_add(a, dim, index, src)
        return result.to(torch.bfloat16)
    return scatter_add(a, dim, index, src)


def _take_prim_transform(a: TensorProxy, /, index: TensorProxy, dim: int) -> TensorProxy:
    return index_select(a, dim, index)


def _take_along_axis_prim_transform(a: TensorProxy, /, index: TensorProxy, dim: int) -> TensorProxy:
    return take_along_dim(a, index, dim)


_register_implementation(prims.gather, checker=_always_executable, execution_transform=_gather_prim_transform)
_register_implementation(prims.index_add, checker=_always_executable, execution_transform=_index_add_prim_transform)
_register_implementation(prims.index_copy, checker=_always_executable, execution_transform=_index_copy_prim_transform)
_register_implementation(prims.index_put, checker=_always_executable, execution_transform=_index_put_prim_transform)
_register_implementation(prims.scatter, checker=_always_executable, execution_transform=_scatter_prim_transform)
_register_implementation(prims.scatter_add, checker=_always_executable, execution_transform=_scatter_add_prim_transform)
_register_implementation(prims.take, checker=_always_executable, execution_transform=_take_prim_transform)
_register_implementation(
    prims.take_along_axis, checker=_always_executable, execution_transform=_take_along_axis_prim_transform
)

_register_implementation(ltorch.gather, checker=_always_executable, execution_transform=_gather_transform)
_register_implementation(ltorch.index_add, index_add, checker=_always_executable)
_register_implementation(ltorch.index_copy, index_copy, checker=_always_executable)
_register_implementation(ltorch.index_put, index_put, checker=_always_executable)
_register_implementation(ltorch.index_select, index_select, checker=_always_executable)
_register_implementation(ltorch.scatter, checker=_always_executable, execution_transform=_scatter_transform)
_register_implementation(ltorch.scatter_add, checker=_always_executable, execution_transform=_scatter_add_transform)
_register_implementation(ltorch.take_along_dim, take_along_dim, checker=_always_executable)

# out of place setitem helper


def _copy_with_setitem_impl(a, key, value):
    c = a.clone()
    c[key] = value
    return c


copy_with_setitem_impl = ex.register_operator(
    "copy_with_setitem_impl", meta=prims.copy_with_setitem_meta, fn=_copy_with_setitem_impl
)
_register_implementation(prims.copy_with_setitem, copy_with_setitem_impl, checker=_always_executable)

#
# Linear algebra operations
#

matmul = _register_torch_operation("matmul")
outer = _register_torch_operation("outer")

_register_implementation(prims.matmul, matmul, checker=_always_executable)

_register_implementation(ltorch.matmul, matmul, checker=_always_executable)
_register_implementation(ltorch.outer, outer, checker=_always_executable)

#
# Normalization operations
#

batch_norm = _register_torch_operation("batch_norm", module=torch.nn.functional)
instance_norm = _register_torch_operation("instance_norm", module=torch.nn.functional)

layer_norm = _register_torch_operation("layer_norm", module=torch.nn.functional)
local_response_norm = _register_torch_operation("local_response_norm", module=torch.nn.functional)

_register_implementation(ltorch.batch_norm, batch_norm, checker=_always_executable)
_register_implementation(ltorch.instance_norm, instance_norm, checker=_always_executable)
_register_implementation(ltorch.layer_norm, layer_norm, checker=_always_executable)
_register_implementation(ltorch.local_response_norm, local_response_norm, checker=_always_executable)

#
# NN operations
#

bmm = _register_torch_operation("bmm")
baddbmm = _register_torch_operation("baddbmm")
convolution = _register_torch_operation("convolution")
conv1d = _register_torch_operation("conv1d", module=torch.nn.functional)
conv2d = _register_torch_operation("conv2d", module=torch.nn.functional)
conv3d = _register_torch_operation("conv3d", module=torch.nn.functional)
mse_loss = _register_torch_operation("mse_loss", module=torch.nn.functional)
dropout = _register_torch_operation("dropout", module=torch.nn.functional)
embedding = _register_torch_operation("embedding", module=torch.nn.functional)
embedding_backward = _register_torch_operation("torch.ops.aten.embedding_backward", like=ltorch.embedding_backward)
one_hot = _register_torch_operation("one_hot", module=torch.nn.functional)
group_norm = _register_torch_operation("group_norm", module=torch.nn.functional)
interpolate = _register_torch_operation("interpolate", module=torch.nn.functional)
linear = _register_torch_operation("linear", module=torch.nn.functional)
logsumexp = _register_torch_operation("logsumexp")
log_softmax = _register_torch_operation("log_softmax", module=torch.nn.functional)
log_softmax_backward = _register_torch_operation(
    "torch.ops.aten._log_softmax_backward_data", like=ltorch.log_softmax_backward
)
max_pool1d = _register_torch_operation("max_pool1d", module=torch.nn.functional)
max_pool2d = _register_torch_operation("max_pool2d", module=torch.nn.functional)
max_pool3d = _register_torch_operation("max_pool3d", module=torch.nn.functional)
adaptive_avg_pool2d = _register_torch_operation("adaptive_avg_pool2d", module=torch.nn.functional)
adaptive_avg_pool2d_backward = _register_torch_operation(
    "torch.ops.aten._adaptive_avg_pool2d_backward", like=ltorch.adaptive_avg_pool2d_backward
)
multi_dot = _register_torch_operation("torch.linalg.multi_dot", like=ltorch.multi_dot)


def _max_pool_with_indices_helper(
    ndim: int,
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    ceil_mode: bool = False,
) -> [TensorProxy, TensorProxy]:
    def div_rtn(x, y):
        q = x // y
        r = x % y
        if r != 0 and (r < 0) != (y < 0):
            q -= 1
        return q

    def pooling_output_shape(in_, kernel_, pad_, stride_, dilation_, ceil_mode_: bool):
        out_size = (
            div_rtn(in_ + 2 * pad_ - dilation_ * (kernel_ - 1) - 1 + (stride_ - 1 if ceil_mode else 0), stride_) + 1
        )
        if ceil_mode and (out_size - 1) * stride_ >= in_ + pad_:
            out_size -= 1
        return out_size

    def get_maybe_ith_entry(arg_name: str, seq: int | Sequence[int], i: int, default: int | None = None):
        if seq is None:
            return default

        if not isinstance(seq, Sequence):
            return seq

        if len(seq) == 1:
            return seq[0]
        else:
            utils.check(
                i < len(seq),
                lambda: f"invalid pooling argument: {arg_name} needs to be None / a scalar / size-{ndim} Sequence, but received {seq}",
            )
            return seq[i]

    out_sizes = list(a.shape[:-ndim])
    for i in range(ndim):
        in_ = a.shape[i - ndim]  # i - ndim is the i-th spatial dimension
        kernel_ = get_maybe_ith_entry("kernel_size", kernel_size, i)
        stride_ = get_maybe_ith_entry("stride", stride, i, kernel_)
        pad_ = get_maybe_ith_entry("padding", padding, i)
        dilation_ = get_maybe_ith_entry("dilation", dilation, i)
        utils.check(
            kernel_ is not None and stride_ is not None and pad_ is not None and dilation_ is not None,
            lambda: "max_pool argument extraction failed.",
        )
        out_sizes.append(pooling_output_shape(in_, kernel_, pad_, stride_, dilation_, ceil_mode))

    return TensorProxy(like=a, shape=out_sizes), TensorProxy(like=a, shape=out_sizes)


def max_pool_with_indices_backward_meta(
    grad: TensorProxy,
    a: TensorProxy,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None,
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    ceil_mode: bool,
    result1: TensorProxy,
) -> TensorProxy:
    return TensorProxy(like=a)


max_pool2d_with_indices_meta = partial(_max_pool_with_indices_helper, 2)

max_pool2d_with_indices = ex.register_operator(
    "max_pool2d_with_indices", meta=max_pool2d_with_indices_meta, fn=torch.ops.aten.max_pool2d_with_indices
)
max_pool2d_with_indices_backward = ex.register_operator(
    "max_pool2d_with_indices_backward",
    meta=max_pool_with_indices_backward_meta,
    fn=torch.ops.aten.max_pool2d_with_indices_backward,
)

max_pool3d_with_indices_meta = partial(_max_pool_with_indices_helper, 3)

max_pool3d_with_indices = ex.register_operator(
    "max_pool3d_with_indices", meta=max_pool3d_with_indices_meta, fn=torch.ops.aten.max_pool3d_with_indices
)
max_pool3d_with_indices_backward = ex.register_operator(
    "max_pool3d_with_indices_backward",
    meta=max_pool_with_indices_backward_meta,
    fn=torch.ops.aten.max_pool3d_with_indices_backward,
)

nll_loss = _register_torch_operation("nll_loss", module=torch.nn.functional)
pad = _register_torch_operation("pad", module=torch.nn.functional)
scaled_dot_product_attention = _register_torch_operation("scaled_dot_product_attention", module=torch.nn.functional)
softmax = _register_torch_operation("softmax", like=ltorch._softmax)


# NOTE This transform translates number proxies to boolean values
def _convolution_transform(
    a: TensorProxy,
    weight: TensorProxy,
    bias: None | TensorProxy,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: Number,
    output_padding: Sequence[int],
    groups: int,
) -> TensorProxy:
    return convolution(a, weight, bias, stride, padding, dilation, bool(transposed), output_padding, groups)


# NOTE PyTorch's nn.functional.interpolate only supports 3D, 4D, and 5D interpolation
def _interpolate_checker(
    a: TensorLike,
    /,
    size: int | Sequence[int] | None = None,
    scale_factor: float | Sequence[float] | None = None,
    mode: str = "nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
) -> TensorLike:
    return 3 <= a.ndim and a.ndim <= 5


def _log_softmax_transform(a: TensorLike, /, dim: int, *, dtype: None | dtypeLike = None) -> TensorLike:
    if dtype is None:
        return log_softmax(a, dim)

    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return log_softmax(a, dim, dtype=torch_dtype)


def _log_softmax_backward_transform(g: TensorProxy, /, output: TensorProxy, dim: int, dtype: dtypeLike) -> TensorLike:
    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return log_softmax_backward(g, output, dim, torch_dtype)


def _softmax_transform(a: TensorLike, /, dim: int, *, dtype: None | dtypeLike = None) -> TensorLike:
    if dtype is None:
        return softmax(a, dim)

    torch_dtype: torch.dtype = to_torch_dtype(dtype)
    return softmax(a, dim, dtype=torch_dtype)


# NOTE This transform is necessary because PyTorch's nll_loss operation has some additional deprecated parameters
def _nll_loss_transform(
    a: TensorProxy,
    /,
    target: TensorProxy,
    weight: None | TensorProxy = None,
    ignore_index: int = None,
    reduction: str = "mean",
) -> TensorProxy:
    return nll_loss(a, target=target, weight=weight, ignore_index=ignore_index, reduction=reduction)


_reduction_str_to_num_map: dict[str, int] = {
    "none": 0,
    "mean": 1,
    "sum": 2,
}


# TODO Review writing this in a traceable way, registering ops.aten.nll_loss_backward and ops.aten.nll_loss2d_backward as operations
def _nll_loss_backward_impl(
    g: torch.Tensor,
    a: torch.Tensor,
    /,
    target: torch.Tensor,
    weight: None | torch.Tensor,
    reduction: str,
    ignore_index: int,
    total_weight: None | torch.Tensor,
) -> torch.Tensor:
    reduction: int = _reduction_str_to_num_map[reduction]

    if total_weight is None:
        # NOTE aten.nll_loss_backward expects total_weight to be a float64 tensor
        total_weight = torch.tensor(0.0, dtype=torch.float64, device=a.device)
    else:
        total_weight = total_weight.to(a.dtype)

    if a.ndim <= 2:
        return torch.ops.aten.nll_loss_backward(g, a, target, weight, reduction, ignore_index, total_weight)

    if a.ndim == 4:
        return torch.ops.aten.nll_loss2d_backward(g, a, target, weight, reduction, ignore_index, total_weight)

    # NOTE a.ndim == 3 or a.ndim > 4
    # Extracts shape information
    N = a.shape[0]
    C = a.shape[1]
    no_C_shape = list(a.shape)
    C_dim = 1 if a.ndim >= 2 else 0
    no_C_shape.pop(C_dim)

    # support empty batches, see #15870
    if a.numel() > 0:
        a_ = a.reshape([N, C, 1, -1])
    else:
        a_ = a.reshape([N, C, 0, 0])

    if target.numel() > 0:
        target_ = target.reshape([N, 1, -1])
    else:
        target_ = target.reshape([N, 0, 0])

    if reduction != "none":
        return torch.ops.aten.nll_loss2d_backward(g, a_, target_, weight, reduction, ignore_index, total_weight)

    # g must have same dimension as target.
    if g.numel() > 0:
        g_ = g.reshape([N, 1, -1])
    else:
        g_ = g.reshape([N, 0, 0])

    result = torch.ops.aten.nll_loss2d_backward(g_, a_, target_, weight, reduction, ignore_index, total_weight)
    return result.reshape(no_C_shape)


# NOTE PyTorch doesn't have a padding operation exactly like XLA's
#   When dilations are all zero, torch.nn.functional.pad can substitute for XLA's
#   Otherwise, this first dilates the original tensor by copying it into a slice of
#   a larger tensor, then pads the dilated tensor
def _pad_prim_impl(
    a: torch.Tensor, /, padding_value: Number, padding_config: Sequence[tuple[int, int, int]]
) -> torch.Tensor:
    intermediate_shape = []
    intermediate_slices = []
    pad_config = []
    just_pad = True
    for l, (low, high, dilation) in zip(a.shape, padding_config):
        assert dilation >= 0

        if dilation > 0:
            just_pad = False

        intermediate_length = l + max(0, l - 1) * dilation
        intermediate_shape.append(intermediate_length)
        intermediate_slices.append(slice(None, None, dilation + 1))

        pad_config.append((low, high))

    pad_config = [x for y in reversed(pad_config) for x in y]

    if just_pad:
        return torch.nn.functional.pad(a, pad_config, value=padding_value)

    result = torch.full(intermediate_shape, padding_value, device=a.device, dtype=a.dtype)
    result[intermediate_slices] = a
    result = torch.nn.functional.pad(result, pad_config, value=padding_value)
    return result


_register_implementation(prims.convolution, checker=_always_executable, execution_transform=_convolution_transform)
_register_implementation(prims.embedding, embedding, checker=_always_executable)
_register_implementation(prims.embedding_backward, embedding_backward, checker=_always_executable)
_register_implementation(prims.linear, linear, checker=_always_executable)

_register_implementation(ltorch.baddbmm, baddbmm, checker=_always_executable)
_register_implementation(ltorch.bmm, bmm, checker=_always_executable)
_register_implementation(ltorch.convolution, checker=_always_executable, execution_transform=_convolution_transform)
_register_implementation(ltorch.conv1d, conv1d, checker=_always_executable)
_register_implementation(ltorch.conv2d, conv2d, checker=_always_executable)
_register_implementation(ltorch.conv3d, conv3d, checker=_always_executable)
_register_implementation(ltorch.mse_loss, mse_loss, checker=_always_executable)
_register_implementation(ltorch.dropout, dropout, checker=_always_executable)
_register_implementation(ltorch.embedding, embedding, checker=_always_executable)
_register_implementation(ltorch.embedding_backward, embedding_backward, checker=_always_executable)
_register_implementation(ltorch.one_hot, one_hot, checker=_always_executable)
_register_implementation(ltorch.group_norm, group_norm, checker=_always_executable)
_register_implementation(ltorch.interpolate, interpolate, checker=_interpolate_checker)
_register_implementation(ltorch.linear, linear, checker=_always_executable)
_register_implementation(ltorch.logsumexp, logsumexp, checker=_always_executable)
_register_implementation(ltorch.log_softmax, checker=_always_executable, execution_transform=_log_softmax_transform)
_register_implementation(
    ltorch.log_softmax_backward, checker=_always_executable, execution_transform=_log_softmax_backward_transform
)
_register_implementation(ltorch.max_pool1d, max_pool1d, checker=_always_executable)


def max_pool2d_bwd_wrapper(
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> tuple[TensorProxy, TensorProxy] | TensorProxy:
    if stride is None:
        stride = kernel_size
    primals = max_pool2d_with_indices(a, kernel_size, stride, padding, dilation, ceil_mode)

    grad = get_grad(primals[0])
    grad_a = max_pool2d_with_indices_backward(grad, a, kernel_size, stride, padding, dilation, ceil_mode, primals[1])
    put_grad(a, grad_a)

    if return_indices:
        return primals
    else:
        return primals[0]


def max_pool3d_bwd_wrapper(
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> tuple[TensorProxy, TensorProxy] | TensorProxy:
    primals = max_pool3d_with_indices(a, kernel_size, stride, padding, dilation, ceil_mode)

    grad = get_grad(primals[0])
    grad_a = max_pool3d_with_indices_backward(grad, a, kernel_size, stride, padding, dilation, ceil_mode, primals[1])
    put_grad(a, grad_a)

    if return_indices:
        return primals
    else:
        return primals[0]


# ltorch.max_pool2d/3d decomposition uses convolution, which has performance issue running through torchex. We added grad_transform that keep both forward and backward max_pool as a torch composite operator, which avoids the performance issue. For details: https://github.com/Lightning-AI/lightning-thunder/issues/164. Aten doesn't have explicit functions for max_pool1d fwd/bwd. So the specialization is only done for 2d/3d case.
ex.register_implementation(
    ltorch.max_pool2d, max_pool2d, checker=_always_executable, grad_transform=max_pool2d_bwd_wrapper
)
ex.register_implementation(
    ltorch.max_pool3d, max_pool3d, checker=_always_executable, grad_transform=max_pool3d_bwd_wrapper
)


def log_sigmoid_backward_meta(
    g: TensorProxy,
    a: TensorProxy,
    buffer: TensorProxy,
) -> TensorProxy:
    return TensorProxy(like=a)


log_sigmoid_backward = ex.register_operator(
    "torch.ops.aten.log_sigmoid_backward",
    meta=log_sigmoid_backward_meta,
    fn=torch.ops.aten.log_sigmoid_backward,
)


def log_sigmoid_backward_wrapper(
    a: TensorProxy,
) -> TensorLike:
    from thunder.torch import abs, exp, logsigmoid

    fwd = logsigmoid(a)
    g = get_grad(fwd)
    if a.device.type == "cpu":
        # buffer is used by PyTorch in cpu-based calculations.  See
        # https://github.com/pytorch/pytorch/blob/7667235a23e2ffca4d32e6e16aa60a683418e159/torch/_decomp/decompositions.py#L332
        buffer = exp(-abs(a))
        a_grad = log_sigmoid_backward(g, a, buffer)
    else:
        placeholder_buffer = empty((0,), device=a.device, dtype=a.dtype)
        a_grad = log_sigmoid_backward(g, a, placeholder_buffer)
    put_grad(a, a_grad)
    return fwd


ex.register_implementation(
    ltorch.logsigmoid,
    logsigmoid,
    checker=_always_executable,
    grad_transform=log_sigmoid_backward_wrapper,
)


def adaptive_avg_pool2d_bwd_wrapper(
    a: TensorProxy,
    /,
    output_size: int | Sequence[int],
) -> TensorProxy:
    primals = adaptive_avg_pool2d(a, output_size)

    grad = get_grad(primals)
    grad_a = adaptive_avg_pool2d_backward(grad, a)
    put_grad(a, grad_a)

    return primals


ex.register_implementation(
    ltorch.adaptive_avg_pool2d,
    adaptive_avg_pool2d,
    checker=_always_executable,
    grad_transform=adaptive_avg_pool2d_bwd_wrapper,
)
_register_implementation(ltorch.adaptive_avg_pool2d_backward, adaptive_avg_pool2d_backward, checker=_always_executable)
_register_implementation(ltorch.nll_loss, checker=_always_executable, execution_transform=_nll_loss_transform)
nll_loss_backward = ex.register_operator(
    "torch_nll_loss_backward_impl", meta=ltorch.nll_loss_backward, fn=_nll_loss_backward_impl
)
_register_implementation(ltorch.nll_loss_backward, nll_loss_backward, checker=_always_executable)
_register_implementation(ltorch.pad, pad, checker=_always_executable)
pad_prim_impl = ex.register_operator("torch_pad_prim_impl", meta=prims.pad.meta, fn=_pad_prim_impl)
_register_implementation(prims.pad, pad_prim_impl, checker=_always_executable)
_register_implementation(ltorch._softmax, checker=_always_executable, execution_transform=_softmax_transform)
_register_implementation(ltorch.scaled_dot_product_attention, scaled_dot_product_attention, checker=_always_executable)


# Probability Distribution ops
def _multinomial_transform(
    input: TensorLike,
    num_samples: int,
    replacement: bool,
    seed: int | None = None,
):
    if seed is not None:
        input_torch_device = to_torch_device(input.device)
        generator = torch.Generator(input_torch_device).manual_seed(seed)
    else:
        generator = None
    return multinomial_helper(input, num_samples, replacement, generator=generator)


def _multinomial_helper_impl(
    a: torch.Tensor,
    num_samples: int,
    replacement: bool = False,
    seed: int | None = None,
):
    if seed is None:
        generator = None
    else:
        generator = torch.Generator(a.device).manual_seed(seed)
    return torch.multinomial(a, num_samples, replacement, generator=generator)


multinomial_helper = ex.register_operator("multinomial_helper", like=prims.multinomial, fn=_multinomial_helper_impl)
_register_implementation(prims.multinomial, checker=_always_executable, execution_transform=multinomial_helper)


#
# Distributed operations
#
# NOTE DISTRIBUTED AVAILABILITY
# PyTorch is often built without distributed support, which can be queried for using
#   torch.distributed.is_available(). When PyTorch is built without distributed then we
#   want to avoid accessing any parts of the torch.distributed module except
#   the is_available() function.

if torch.distributed.is_available():

    def _all_gather_prim_impl(
        a: torch.Tensor,
        /,
        group: torch.distributed.ProcessGroup,
        do_async: Number,
        dim: int | None = None,
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        result_shape = list(a.shape)
        if dim is not None:
            utils.check_type(dim, int)
            utils.check(dim >= 0 and dim < a.dim(), lambda: f"dim must satisfy 0 <= {dim=} < {a.dim()=}")
            result_shape[dim] *= group.size()
        else:
            result_shape[0] *= group.size()
        out = torch.empty(result_shape, dtype=a.dtype, device=a.device)
        do_async: bool = bool(do_async)

        handle: None | torch.distributed.distributed_c10d.Work = torch.distributed.all_gather_into_tensor(
            out, a, group, do_async
        )

        if do_async:
            return handle, out

        return out

    def _all_reduce_prim_impl(
        a: torch.Tensor,
        /,
        op: DistributedReduceOps,
        group: torch.distributed.ProcessGroup,
        do_async: Number,
        skip_clone: bool = False,
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        c: torch.Tensor = a.clone() if not skip_clone else a
        op: torch.distributed.ReduceOp = ltorch.to_torch_distributed_reduce_op(op)
        do_async: bool = bool(do_async)

        handle: None | torch.distributed.distributed_c10d.Work = torch.distributed.all_reduce(c, op, group, do_async)

        # NOTE This returns (handle, tensor reference), which is how FutureTensorProxies
        #   are currently modeled by executors
        if do_async:
            return handle, c

        # NOTE do_async is False
        return c

    def _broadcast_prim_impl(
        a: torch.Tensor, /, root: int, group: torch.distributed.ProcessGroup, do_async: Number
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        out = a.clone()
        do_async: bool = bool(do_async)

        handle: None | torch.distributed.distributed_c10d.Work = torch.distributed.broadcast(out, root, group, do_async)

        if do_async:
            return handle, out

        return out

    def _reduce_scatter_prim_impl(
        a: torch.Tensor,
        /,
        op: DistributedReduceOps,
        group: torch.distributed.ProcessGroup,
        do_async: Number,
        dim: int | None,
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        result_shape = list(a.shape)
        if dim is not None:
            utils.check_type(dim, int)
            utils.check(dim >= 0 and dim < a.dim(), lambda: f"dim must satisfry 0 <= {dim=} < {a.dim()=}")
            result_shape[dim] //= group.size()
        else:
            result_shape[0] //= group.size()
        out = torch.empty(result_shape, dtype=a.dtype, device=a.device)
        op: torch.distributed.ReduceOp = ltorch.to_torch_distributed_reduce_op(op)
        do_async: bool = bool(do_async)

        handle: None | torch.distributed.distributed_c10d.Work = torch.distributed.reduce_scatter_tensor(
            out, a, op, group, do_async
        )

        if do_async:
            return handle, out
        return out

    # NOTE This is a very particular implementation of wait that may need to be
    #   generalized in the future
    # NOTE The implementation of wait actually models the FutureTensorProxy as a tuple
    #   of (handle, tensor reference), there's no conflict with the trace representation
    #   with doing this, as the trace representation treates FutureTensorProxy as opaque
    # NOTE This way of representing FutureTensorProxies may need to change in the future
    def _wait_prim_impl(a: tuple[torch.distributed.distributed_c10d.Work, torch.Tensor], /) -> torch.Tensor:
        handle: torch.distributed.distributed_c10d.Work
        t: torch.Tensor
        handle, t = a
        handle.wait()
        return t

    _key_to_bucket_and_views: dict[str, tuple[torch.Tensor, list[torch.Tensor]]] = {}

    def _pack_prim_impl(
        tensors: list[torch.Tensor],
        bucket_key: str,
    ) -> torch.Tensor:
        if bucket_key not in _key_to_bucket_and_views:
            buffer = torch._utils._flatten_dense_tensors(tensors)
            offset = 0
            views = []
            for t in tensors:
                n = t.numel()
                v = buffer[offset : offset + n].view_as(t)
                views.append(v)
                offset += n
            _key_to_bucket_and_views[bucket_key] = (buffer, views)
        buffer, _ = _key_to_bucket_and_views[bucket_key]
        return buffer

    def _unpack_prim_impl(
        buffer: torch.Tensor,
        tensors: list[torch.Tensor],
        bucket_key: str,
    ) -> list[torch.Tensor]:
        _, views = _key_to_bucket_and_views[bucket_key]
        torch._foreach_copy_(tensors, views, non_blocking=True)
        return tensors

    # TODO(crcrpar): Make this compatible with the torch.compile executor as it's doing really well for cat and reshape.
    # NOTE(crcrpar): why no caching/resue of buffer?
    # This prim is only used by fsdp backward for now.
    # Bucketing of reduce-scatter, i.e., creating a buffer for
    # multiple unsharded gradients is a bit tricky because it requires
    # indices for copies from unsharded ones to their bucket.
    # To be specific, let's say our environment has 2 cuda devices and we have two unsharded gradeints whose shapes are (32, 4) and (4,).
    # Obviously a bucket for these two would have the shape of (32 * 4 + 4 = 160,),
    # and its shape after reduce_scatter will be (160 // 2 = 80,).
    # Sharded gradients on rank-0 should be tensors of (32 // 2, 4) and (4 // 2,).
    # If we do the same packing as bucketing for ddp, the first 32 * 4 elements will be
    # that (32, 4) tensor and the rest, the other (4,) tensor.
    # In this case, sharded bucket on rank 0, will be the first 80 elements out of the bucket, i.e., 80 elements of (32, 4) tensor, which is wrong.
    # So what we need to do here is interleave the tensors of one bucket.
    # In this example, the first 80 elements of this bucket and the rest need to have
    # the same number of values from the two unsharded gradients, i.e.,
    # the first chunk of a bucket needs to consists of the first chunks of the two unsharded gradients.
    # To support individual copies from gradient to its bucket requires a mask or an arrayy of indices to achieve correct behavior.
    # In PyTorch, the op for this is [`Tensor.index_copy_`](https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html) where even the index tensor needs to be on the same device as ``self`` and ``tensor``.
    # So caching of the bucketing for fsdp backward would bloat up the memory consumption, which is the main reason this doesn't do any caching.
    #
    # example of two unsharded gradients of [4, 2] and [4], world size of 4:
    # --------  ------
    # | 0, 1 |  | 8  |
    # | 2, 3 |  | 9  |
    # | 4, 5 |  | 10 |
    # | 6, 7 |  | 11 |
    # --------  ------
    #
    # If we naively pack these two into a bucket, it will look like:
    # ----------------------------------------
    # | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 |
    # ----------------------------------------
    #
    # Each rank would receive 3 elements from ReduceScatter of this bucket:
    #    rank0        rank1        rank2        rank3
    # -----------  -----------  -----------  -------------
    # | 0, 1, 2 |, | 3, 4, 5 |, | 6, 7, 8 |, | 9, 10, 11 |
    # -----------  -----------  -----------  -------------
    # then each rank splits their chunk into two tensors as sharded gradients:
    # So rank0 will consider [[0, 1]] as sharded grad for first parameter and [2], the other
    # but they should be [[0, 1]] and [8]. To make this happen the bucket needs to be
    # ----------------------------------------
    # | 0, 1, 8, 2, 3, 9, 4, 5, 10, 6, 7, 11 |
    # ----------------------------------------
    def _pack_for_fsdp_prim_impl(
        tensors: list[torch.Tensor],
        world_size: int,
        mode: str,
    ) -> torch.Tensor:
        match mode:
            case "scatter":
                from itertools import chain

                return torch.cat(
                    list(chain.from_iterable(zip(*[torch.chunk(t.view(-1), world_size) for t in tensors])))
                )
            case "gather":
                return torch._utils._flatten_dense_tensors(tensors)
            case _:
                utils.check(False, lambda: f"Invalid {mode=}. Supported are (gather, scatter)")

    def _unpack_for_fsdp_prim_impl(
        buffer: torch.Tensor,
        tensors: list[torch.Tensor],
        world_size: int,
        mode: str,
    ) -> list[torch.Tensor]:
        if mode == "gather":
            buffer_2d = buffer.view(world_size, -1)
            offset = 0
            result = []
            for t in tensors:
                n = t.numel()
                chunk = buffer_2d[:, offset : offset + n]
                shape = list(t.shape)
                shape[0] *= world_size
                result.append(chunk.reshape(shape))
                offset += n
            return result
        else:
            import math

            offset = 0
            result = []
            for t in tensors:
                shape = list(t.shape)
                shape[0] //= world_size
                n = math.prod(shape)
                result.append(buffer[offset : offset + n].view(shape))
                offset += n
            return result

    def _update_bucket_view_prim_impl(
        tensor: torch.Tensor,
        index_of_dst_view: int,
        bucket_key: str,
    ) -> torch.Tensor:
        if bucket_key not in _key_to_bucket_and_views:
            return tensor
        _, views = _key_to_bucket_and_views[bucket_key]
        views[index_of_dst_view].copy_(tensor)
        return tensor

    # TODO(crcrpar): Update to return None instead
    def _stash_grad_for_fsdp_prim_impl(
        grad: torch.Tensor,
        param_fqn: str,
        compile_data: CompileData,
    ) -> None:
        grad_name = "_thunder_fsdp_unsharded_grad"
        param = compile_data.fn.get_parameter(param_fqn)
        if torch.is_tensor(unsharded_grad := getattr(param, grad_name, None)):
            unsharded_grad += grad
        else:
            setattr(param, grad_name, grad)

        return grad

    all_gather_prim_impl = ex.register_operator(
        "torch_all_gather_prim_impl", meta=dist_prims.all_gather.meta, fn=_all_gather_prim_impl
    )
    _register_implementation(dist_prims.all_gather, all_gather_prim_impl, checker=_always_executable)
    all_reduce_prim_impl = ex.register_operator(
        "torch_all_reduce_prim_impl", meta=dist_prims.all_reduce.meta, fn=_all_reduce_prim_impl
    )
    _register_implementation(dist_prims.all_reduce, all_reduce_prim_impl, checker=_always_executable)
    broadcast_prim_impl = ex.register_operator(
        "torch_broadcast_prim_impl", meta=dist_prims.broadcast.meta, fn=_broadcast_prim_impl
    )
    _register_implementation(dist_prims.broadcast, broadcast_prim_impl, checker=_always_executable)
    reduce_scatter_prim_impl = ex.register_operator(
        "torch_reduce_scatter_prim_impl", meta=dist_prims.reduce_scatter.meta, fn=_reduce_scatter_prim_impl
    )
    _register_implementation(dist_prims.reduce_scatter, reduce_scatter_prim_impl, checker=_always_executable)
    wait_prim_impl = ex.register_operator("torch_wait_prim_impl", meta=dist_prims.wait.meta, fn=_wait_prim_impl)
    _register_implementation(dist_prims.wait, wait_prim_impl, checker=_always_executable)
    pack_prim_impl = ex.register_operator("torch_pack_prim_impl", meta=dist_prims.pack.meta, fn=_pack_prim_impl)
    _register_implementation(dist_prims.pack, pack_prim_impl, checker=_always_executable)
    unpack_prim_impl = ex.register_operator("torch_unpack_prim_impl", meta=dist_prims.unpack.meta, fn=_unpack_prim_impl)
    _register_implementation(dist_prims.unpack, unpack_prim_impl, checker=_always_executable)
    unpack_for_fsdp_prim_impl = ex.register_operator(
        "torch_unpack_for_fsdp_prim_impl", meta=dist_prims.unpack_for_fsdp.meta, fn=_unpack_for_fsdp_prim_impl
    )
    _register_implementation(dist_prims.unpack_for_fsdp, unpack_for_fsdp_prim_impl, checker=_always_executable)
    update_bucket_view_prim_impl = ex.register_operator(
        "torch_update_bucket_view_prim_impl",
        meta=dist_prims.update_bucket_view.meta,
        fn=_update_bucket_view_prim_impl,
    )
    _register_implementation(dist_prims.update_bucket_view, update_bucket_view_prim_impl, checker=_always_executable)

    pack_for_fsdp_prim_impl = ex.register_operator(
        "torch_pack_for_fsdp_prim_impl",
        meta=dist_prims.pack_for_fsdp.meta,
        fn=_pack_for_fsdp_prim_impl,
    )
    _register_implementation(
        dist_prims.pack_for_fsdp,
        pack_for_fsdp_prim_impl,
        checker=_always_executable,
    )

    stash_grad_for_fsdp_prim_impl = ex.register_operator(
        "torch_stash_grad_for_fsdp_prim_impl",
        meta=dist_prims.stash_grad_for_fsdp.meta,
        fn=_stash_grad_for_fsdp_prim_impl,
    )
    _register_implementation(
        dist_prims.stash_grad_for_fsdp,
        stash_grad_for_fsdp_prim_impl,
        checker=_always_executable,
    )

# Memory access operations
item = _register_torch_operation("item", module=torch.Tensor)
_register_implementation(prims.item, item, checker=_always_executable)


has_einops = importlib.util.find_spec("einops") is not None
if has_einops:
    has_einops = importlib.util.find_spec("einops._backends") is not None
if has_einops:
    import einops
    from einops._backends import TorchBackend

    class EinopsThunderBackend(TorchBackend):
        framework_name = "thunder"

        def __init__(self):
            super().__init__()

            self.torch = ltorch

        def is_appropriate_type(self, input):
            return isinstance(input, TensorLike)

        def is_float_type(self, input):
            return dtypes.is_float_dtype(input.dtype)

    # We force the registration of the backend here to not use
    # the torch backend when diverting isinstance
    einops._backends._type2backend[TensorProxy] = EinopsThunderBackend()


def _copy__impl(copy_from, copy_to, grad_enabled):
    if grad_enabled and copy_to.is_leaf and copy_to.requires_grad:
        raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
    copy_to.copy_(copy_from)
    return copy_to


copy_ = ex.register_operator(
    "copy_", meta=prims.copy_, tags=(prims.OpTags.DONT_DCE,), fn=_copy__impl, module=torch.Tensor
)
_register_implementation(prims.copy_, copy_, checker=_always_executable)


def _shape_impl(t):
    return t.shape


shape = ex.register_operator("shape", meta=prims.shape_meta, fn=_shape_impl)
_register_implementation(prims.shape, shape, checker=_always_executable)


shallow_copy = ex.register_operator("shallow_copy", meta=prims.shallow_copy, fn=lambda x: x)
_register_implementation(prims.shallow_copy, shallow_copy, checker=_always_executable)


update_aliases = ex.register_operator("update_aliases", meta=prims.update_aliases, fn=lambda x: x)
_register_implementation(prims.update_aliases, update_aliases, checker=_always_executable)
