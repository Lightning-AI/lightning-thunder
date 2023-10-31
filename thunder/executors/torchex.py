import operator
from dataclasses import replace
from functools import wraps, partial
from inspect import signature
from itertools import groupby
from numbers import Number
from typing import Union, Callable, Any, Tuple, Optional
from collections.abc import Sequence

import torch
import math
from looseversion import LooseVersion

import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
from thunder.core.prims import PrimIDs
from thunder.core.trace import TraceCtx, set_tracectx, reset_tracectx, from_trace
from thunder.core.proxies import TensorProxy, FutureTensorProxy, variableify
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.symbol import Symbol, BoundSymbol
from thunder.distributed.prims import DistributedReduceOps
import thunder.distributed.prims as dist_prims
import thunder.core.devices as devices
import thunder.core.utils as utils

from thunder.executors.utils import Region, Executor
from thunder.executors.passes import replace_redundant_inputs

import thunder.torch as ltorch
from thunder.torch import DeviceLike, dtypeLike, TensorLike


# NOTE This is part of the executor interface
def name() -> Executor:
    return Executor.TORCH


torch_ctx = {
    "torch": torch,
}

# NOTE _ops_map is declared here and defined after the callables have been defined
#   below
_ops_map: dict[Any, tuple[Callable, Callable]] = {}


# Helper to signal that an operation is always executable
def _always_executable(*args, **kwargs) -> bool:
    return True


def _is_autocast_enabled() -> bool:
    return torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()


#
# Data movement operations
#


def convert_element_type(
    bsym: BoundSymbol,
    a: Union[TensorProxy, Number],
    dtype: dtypes.dtype,
) -> BoundSymbol:
    # Handles converting a tensor to a numbertype, which Thunder allows but
    # Torch does not
    if isinstance(a, torch.Tensor) and dtypes.is_numbertype(dtype):
        torch_dtype = ltorch.to_torch_dtype(dtypes.numbertype_to_dtype(dtype))

    # Handles number conversions
    if isinstance(a, Number):
        if not dtypes.is_numbertype(dtype):
            dtype = dtypes.dtype_to_numbertype(dtype)

        sym = Symbol(name=f"{dtype.__name__}", meta=None, python_impl=lambda a, dtype: dtype(a))
        tbsym = BoundSymbol(sym, args=(a,), kwargs={}, output=bsym.output)
        return tbsym

    sym = Symbol(name="to", meta=None, _module=torch.Tensor)

    torch_dtype = ltorch.to_torch_dtype(dtype)
    tbsym = BoundSymbol(sym, args=(a,), kwargs={"dtype": torch_dtype}, output=bsym.output)

    return tbsym


# TODO Currently this prints like "Tensor.to(...)" but it should print as "torch.Tensor.to(...)"
# NOTE to is only a method in PyTorch, so this calls torch.Tensor.to
# NOTE This translates the lightning.compile device to a string representing a PyTorch device
def device_put(bsym: BoundSymbol, a: TensorProxy, device: devices.Device) -> BoundSymbol:
    sym = Symbol(name="to", meta=None, _module=torch.Tensor)

    torch_device = str(device)
    tbsym = BoundSymbol(sym, args=(a, torch_device), kwargs={}, output=bsym.output)

    return tbsym


# NOTE This is a direct lowering of torch.Tensor.to
def to(
    bsym: BoundSymbol,
    a,
    tensor_dtype_or_device: Optional = None,
    optional_positional_dtype: Optional = None,
    /,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
) -> BoundSymbol:
    sym = Symbol(name="to", meta=None, _module=torch.Tensor)

    device, dtype = ltorch._parse_to_device_and_dtype(
        tensor_dtype_or_device, optional_positional_dtype, device=device, dtype=dtype
    )
    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)

    return sym.bind(a, device=device, dtype=dtype, output=bsym.output)


#
# Tensor creation operations
#


def arange(
    bsym: BoundSymbol,
    start: Number,
    end: Optional[Number] = None,
    step: Number = 1,
    *,
    device: Union[str, devices.Device, torch.device] = "cpu",
    dtype: Optional[Union[dtypes.dtype, torch.dtype]] = None,
):
    sym = Symbol(name="arange", meta=None, _module=torch)

    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)

    if end is None:
        end = start
        start = 0

    tbsym = sym.bind(start, end, step, device=device, dtype=dtype, output=bsym.output)
    return tbsym


def _exogenous_like_helper(likes: Sequence[torch.Tensor], /) -> tuple[torch.Tensor, ...]:
    return tuple([torch.zeros_like(x) for x in likes])


def exogenous_like(bsym: BoundSymbol, likes: Sequence[TensorProxy], /) -> BoundSymbol:
    sym = Symbol(name="exogenous_like", meta=None)
    ctx: dict[str, Any] = {"exogenous_like": _exogenous_like_helper}

    return sym.bind(likes, output=bsym.output, _call_ctx=ctx)


def full(
    bsym: BoundSymbol, shape: Sequence[int], fill_value: Number, *, device: devices.Device, dtype: dtypes.dtype
) -> BoundSymbol:
    sym = Symbol(name="full", meta=None, _module=torch)

    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)

    tbsym = sym.bind(shape, fill_value, device=device, dtype=dtype, output=bsym.output)
    return tbsym


def full_like(
    bsym: BoundSymbol,
    a: TensorLike,
    fill_value: Number,
    *,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
) -> BoundSymbol:
    sym = Symbol(name="full_like", meta=None, _module=torch)

    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)

    return sym.bind(a, fill_value, device=device, dtype=dtype, output=bsym.output)


# TODO Review whether this helper is required (and if it is, document why better)
def _iota_helper(length, *, start, step, device, dtype) -> torch.Tensor:
    end = start + length * step
    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)
    return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype)


def iota(bsym: BoundSymbol, length, *, start, step, device, dtype) -> BoundSymbol:
    sym = Symbol(name="iota_helper", meta=None)
    ctx: dict[str, Any] = {"iota_helper": _iota_helper}

    kwargs = {
        "start": start,
        "step": step,
        "device": device,
        "dtype": dtype,
    }

    tbsym = BoundSymbol(
        sym,
        args=(length,),
        kwargs=kwargs,
        output=bsym.output,
        _call_ctx=ctx,
    )
    return tbsym


# NOTE This helper is necessary because PyTorch doesn't define a uniform operation
def uniform_helper(shape: Sequence[int], minval: Number, maxval: Number, *, device: torch.device, dtype: torch.dtype):
    t = torch.empty(shape, device=device, dtype=dtype)
    t.uniform_(minval, maxval)
    return t


def uniform_prim(
    bsym: BoundSymbol,
    shape: Sequence[int],
    minval: Number,
    maxval: Number,
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
) -> BoundSymbol:
    sym = Symbol(name="uniform_helper", meta=None)
    ctx: dict[str, Any] = {"uniform_helper": uniform_helper}

    torch_device = ltorch.to_torch_device(device)
    torch_dtype = ltorch.to_torch_dtype(dtype)

    return sym.bind(shape, minval, maxval, device=torch_device, dtype=torch_dtype, output=bsym.output, _call_ctx=ctx)


def _uniform_philox_check(
    shape: Sequence[int],
    minval: float,
    maxval: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rng_seed: torch.Tensor,
    rng_offset: torch.Tensor,
) -> bool:
    utils.check(
        minval == 0.0 and maxval == 1.0,
        lambda: f"not supported combination of minval and maxval of {(minval, maxval)}",
    )

    # ref: https://github.com/pytorch/pytorch/blob/b60273b88a25e282728a036dd08d39035a369d1f/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L230
    utils.check(rng_offset % 4 == 0, lambda: f"`rng_offset` must be a multiple of 4 but {rng_offset % 4 = }")

    utils.check(
        ltorch.to_thunder_device(device).devicetype == devices.DeviceType.CUDA,
        lambda: "`uniform_philox` does not support CPU",
    )

    return True


# ref: https://github.com/pytorch/pytorch/blob/b60273b88a25e282728a036dd08d39035a369d1f/test/test_prims.py#L277
def uniform_philox_helper(
    shape: Sequence[int],
    minval: Number,
    maxval: Number,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rng_seed: torch.Tensor,
    rng_offset: torch.Tensor,
) -> torch.Tensor:
    result, _ = torch.ops.rngprims.philox_rand(
        shape,
        seed=rng_seed,
        offset=rng_offset,
        stride=None,
        device=device,
        dtype=dtype,
    )
    return result


def uniform_philox_prim(
    bsym: BoundSymbol,
    shape: Sequence[int],
    minval: Number,
    maxval: Number,
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
    rng_seed: int,
    rng_offset: int,
) -> BoundSymbol:
    sym = Symbol(name="uniform_philox_helper", meta=None)
    ctx: dict[str, Any] = {"uniform_philox_helper": uniform_philox_helper}

    torch_device = ltorch.to_torch_device(device)
    torch_dtype = ltorch.to_torch_dtype(dtype)

    return sym.bind(
        shape,
        minval,
        maxval,
        device=torch_device,
        dtype=torch_dtype,
        rng_seed=torch.tensor(rng_seed),
        rng_offset=torch.tensor(rng_offset),
        output=bsym.output,
        _call_ctx=ctx,
    )


def zeros_like(
    bsym: BoundSymbol,
    a: TensorLike,
    /,
    *,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
) -> BoundSymbol:
    sym = Symbol(name="zeros_like", meta=None, _module=torch)

    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)

    return sym.bind(a, device=device, dtype=dtype, output=bsym.output)


#
# Shape operations
#


def _broadcast_in_dim_helper(a, shape, broadcast_dims):
    s = list(shape)
    for broadcast_dim in broadcast_dims:
        s[broadcast_dim] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = v.unsqueeze(idx)

    return v.expand(shape)


def broadcast_in_dim(bsym: BoundSymbol, a, shape, broadcast_dims) -> BoundSymbol:
    sym = Symbol(name="broadcast_in_dim", meta=None)
    ctx: Dict[str, Any] = {"broadcast_in_dim": _broadcast_in_dim_helper}

    tbsym = BoundSymbol(
        sym,
        args=(a, shape, broadcast_dims),
        kwargs={},
        output=bsym.output,
        _call_ctx=ctx,
    )

    return tbsym


def cat(bsym: BoundSymbol, tensors, dim=0) -> BoundSymbol:
    sym = Symbol(name="cat", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(tensors, dim), kwargs={}, output=bsym.output)

    return tbsym


def chunk(bsym: BoundSymbol, a: TensorProxy, chunks: int, dim: int = 0) -> BoundSymbol:
    sym = Symbol(name="chunk", meta=None, _module=torch)
    return sym.bind(a, chunks, dim, output=bsym.output)


def diagonal(bsym: BoundSymbol, a: TensorLike, offset: int = 0, dim1: int = 0, dim2: int = 1) -> TensorLike:
    sym = Symbol(name="diagonal", meta=None, _module=torch)
    return sym.bind(a, offset, dim1, dim2, output=bsym.output)


def expand(bsym: BoundSymbol, tensor, *shape):
    sym = Symbol(name="expand", meta=None, _module=torch.Tensor)
    return sym.bind(tensor, *shape, output=bsym.output)


def flatten(bsym: BoundSymbol, a: TensorLike, start_dim: int = 0, end_dim: int = -1) -> TensorLike:
    sym = Symbol(name="flatten", meta=None, _module=torch)
    return sym.bind(a, start_dim, end_dim, output=bsym.output)


def unbind(bsym: BoundSymbol, a: TensorLike, dim: int = 0) -> TensorLike:
    sym = Symbol(name="unbind", meta=None, _module=torch)
    return sym.bind(a, dim, output=bsym.output)


def flip(bsym: BoundSymbol, a: TensorLike, dims: Sequence[int]) -> TensorLike:
    sym = Symbol(name="flip", meta=None, _module=torch)
    return sym.bind(a, dims, output=bsym.output)


def getitem(bsym: BoundSymbol, tensor, key) -> BoundSymbol:
    sym = Symbol(name="__getitem__", meta=None, _module=torch.Tensor)
    tbsym = BoundSymbol(sym, args=(tensor, key), kwargs={}, output=bsym.output)

    return tbsym


def movedim(
    bsym: BoundSymbol, a: TensorLike, /, source: int | Sequence[int], destination: int | Sequence[int]
) -> TensorLike:
    sym = Symbol(name="movedim", meta=None, _module=torch)
    return sym.bind(a, source, destination, output=bsym.output)


# NOTE PyTorch doesn't have a padding operation exactly like XLA's
#   When dilations are all zero, torch.nn.functional.pad can substitute for XLA's
#   Otherwise, this first dilates the original tensor by copying it into a slice of
#   a larger tensor, then pads the dilated tensor
def _pad_helper(a, padding_value, padding_config):
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


def pad(bsym: BoundSymbol, a, padding_value, padding_config):
    sym = Symbol(name="pad", meta=None)
    ctx: Dict[str, Any] = {"pad": _pad_helper}

    tbsym = BoundSymbol(
        sym,
        args=(a, padding_value, padding_config),
        kwargs={},
        output=bsym.output,
        _call_ctx=ctx,
    )

    return tbsym


def reshape(bsym: BoundSymbol, a, *shape):
    shape = utils.extract_shape_from_varargs(shape)
    sym = Symbol(name="reshape", meta=None, _module=torch)
    return sym.bind(a, shape, output=bsym.output)


def repeat(bsym: BoundSymbol, a, *repeats):
    repeats = utils.extract_shape_from_varargs(repeats)
    utils.check_valid_shape(repeats)
    sym = Symbol(name="repeat", meta=None, _module=torch.Tensor)
    return sym.bind(a, repeats, output=bsym.output)


# TODO Review if nvFuser can handle striding
def _slice_helper(a, start_indices, end_indices, strides=None):
    _strides = strides if strides is not None else [1] * len(start_indices)

    slices = []
    for start, stop, step in zip(start_indices, end_indices, _strides):
        slices.append(slice(start, stop, step))

    return operator.getitem(a, slices)


def slice_prim(bsym: BoundSymbol, a, start_indices, end_indices, strides=None):
    sym = Symbol(name="slice_prim", meta=None)
    ctx: Dict[str, Any] = {"slice_prim": _slice_helper}

    kwargs = {"strides": strides} if strides is not None else {}

    tbsym = BoundSymbol(
        sym,
        args=(a, start_indices, end_indices),
        kwargs=kwargs,
        output=bsym.output,
        _call_ctx=ctx,
    )

    return tbsym


def split(bsym: BoundSymbol, a, size_or_sections, dim=0):
    sym = Symbol(name="split", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, size_or_sections, dim), kwargs={}, output=bsym.output)

    return tbsym


# TODO We can probably depend on PyTorch 2.0
# NOTE dim as a sequence is only supported on PyTorch 2.0 and greater
def _squeeze_helper(a: torch.Tensor, dim: Optional[Union[int, Sequence[int]]]):
    if dim is None:
        return a.squeeze()

    if isinstance(dim, int):
        return a.squeeze(dim)

    for d in sorted(dim, reverse=True):
        a = torch.Tensor.squeeze(a, d)

    return a


# TODO Review naming to better avoid collisions
# NOTE It's important the name isn't "squeeze" because that can name collide with functions
#   named "squeeze" (like the kind we construct in the tests) and cause the function to be
#   called recursively forever
def squeeze(bsym: BoundSymbol, a, dim=None):
    sym = Symbol(name="squeeze_helper", meta=None)
    ctx: Dict[str, Any] = {"squeeze_helper": _squeeze_helper}

    tbsym = BoundSymbol(
        sym,
        args=(a, dim),
        kwargs={},
        output=bsym.output,
        _call_ctx=ctx,
    )

    return tbsym


# NOTE Order of index and dim changes
def take(bsym: BoundSymbol, a, index, dim):
    sym = Symbol(name="index_select", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim, index), kwargs={}, output=bsym.output)

    return tbsym


def take_along_axis(bsym: BoundSymbol, a, index, dim):
    sym = Symbol(name="take_along_dim", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, index, dim), kwargs={}, output=bsym.output)

    return tbsym


# NOTE Order of index, value and dim changes
def index_add(bsym: BoundSymbol, a, index, value, dim):
    sym = Symbol(name="index_add", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim, index, value), kwargs={}, output=bsym.output)

    return tbsym


# NOTE Order of index, value and dim changes
def scatter_add(bsym: BoundSymbol, a, index, value, dim):
    sym = Symbol(name="scatter_add", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim, index, value), kwargs={}, output=bsym.output)

    return tbsym


def tensor_split(bsym: BoundSymbol, a, size_or_sections, dim=0):
    sym = Symbol(name="tensor_split", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, size_or_sections, dim), kwargs={}, output=bsym.output)

    return tbsym


def transpose(bsym: BoundSymbol, a, dim0, dim1):
    sym = Symbol(name="transpose", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim0, dim1), kwargs={}, output=bsym.output)

    return tbsym


# NOTE Alphabetized as "transpose"
def prim_transpose(bsym: BoundSymbol, a, permutation):
    sym = Symbol(name="permute", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, permutation), kwargs={}, output=bsym.output)

    return tbsym


def permute(bsym: BoundSymbol, a, *dims: int):
    # NOTE This is necessary because torch.permute() requires the permutation be
    #   specified as a tuple, not varargs
    dims = utils.extract_shape_from_varargs(dims)
    sym = Symbol(name="permute", meta=None, _module=torch)
    return sym.bind(a, dims, output=bsym.output)


def unsqueeze(bsym: BoundSymbol, a, dim: int):
    sym = Symbol(name="unsqueeze", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim), kwargs={}, output=bsym.output)

    return tbsym


def view(bsym: BoundSymbol, a, *shape):
    sym = Symbol(name="view", meta=None, _module=torch.Tensor)
    tbsym = BoundSymbol(sym, args=(a, shape), kwargs={}, output=bsym.output)

    return tbsym


#
# Memory format operations
#


def contiguous(
    bsym: BoundSymbol, a: TensorLike, /, *, memory_format: torch.memory_format = torch.contiguous_format
) -> BoundSymbol:
    sym = Symbol(name="contiguous", meta=None, _module=torch.Tensor)
    return sym.bind(a, memory_format=memory_format, output=bsym.output)


# TODO Detect if the tensor is already contiguous as requested, and if so just return it
# TODO Review how strides are set if the tensor contains no elements
def _stride_order_helper(a: torch.Tensor, order: Sequence[int]) -> torch.Tensor:
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


def stride_order(bsym: BoundSymbol, a: TensorProxy, order: Sequence[int]) -> BoundSymbol:
    sym = Symbol(name="stride_order", meta=None)
    ctx: dict[str, Any] = {"stride_order": _stride_order_helper}

    return sym.bind(a, order, output=bsym.output, _call_ctx=ctx)


#
# Elementwise unary operations
#


def _elementwise_unary_check(a: Union[TensorProxy, Number]) -> bool:
    return isinstance(a, TensorProxy)


def _elementwise_unary_factory(name: str, *, module=torch) -> Callable:
    def fn(bsym: BoundSymbol, a: TensorProxy) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=module)
        tbsym = BoundSymbol(sym, args=(a,), kwargs={}, output=bsym.output)

        return tbsym

    return fn


# TODO Review which of these operators are in special

# NOTE torch_abs to avoid a name conflict with the builtin abs
# TODO torch.abs is actually not implemented for unsigned dtypes
torch_abs = _elementwise_unary_factory("abs")
acos = _elementwise_unary_factory("acos")
acosh = _elementwise_unary_factory("acosh")
asin = _elementwise_unary_factory("asin")
asinh = _elementwise_unary_factory("asinh")
atan = _elementwise_unary_factory("atan")
atanh = _elementwise_unary_factory("atanh")
bitwise_not = _elementwise_unary_factory("bitwise_not")
ceil = _elementwise_unary_factory("ceil")
cos = _elementwise_unary_factory("cos")
cosh = _elementwise_unary_factory("cosh")
digamma = _elementwise_unary_factory("digamma")
erf = _elementwise_unary_factory("erf")
erfc = _elementwise_unary_factory("erfc")
# NOTE PyTorch doesn't have erfcinv, although it can be implemented using erfinv
# erfcinv
erfinv = _elementwise_unary_factory("erfinv")
exp = _elementwise_unary_factory("exp")
exp2 = _elementwise_unary_factory("exp2")
expm1 = _elementwise_unary_factory("expm1")
floor = _elementwise_unary_factory("floor")
isfinite = _elementwise_unary_factory("isfinite")
lgamma = _elementwise_unary_factory("lgamma")
log = _elementwise_unary_factory("log")
log10 = _elementwise_unary_factory("log10")
log1p = _elementwise_unary_factory("log1p")
log2 = _elementwise_unary_factory("log2")
ndtri = _elementwise_unary_factory("ndtri", module=torch.special)
neg = _elementwise_unary_factory("neg")
reciprocal = _elementwise_unary_factory("reciprocal")
# NOTE torch_round to avoid a name conflict with the builtin round
torch_round = _elementwise_unary_factory("round")
rsqrt = _elementwise_unary_factory("rsqrt")
# NOTE That PyTorch's "sgn" corresponds with the "sign" primitive
sgn = _elementwise_unary_factory("sgn")
# NOTE torch.sign isn't bound here because lightning.compile always uses sgn
# sign =  _elementwise_unary_factory("sign")
signbit = _elementwise_unary_factory("signbit")
sin = _elementwise_unary_factory("sin")
sinh = _elementwise_unary_factory("sinh")
sqrt = _elementwise_unary_factory("sqrt")
tan = _elementwise_unary_factory("tan")
tanh = _elementwise_unary_factory("tanh")
trunc = _elementwise_unary_factory("trunc")
real = _elementwise_unary_factory("real")

#
# Elementwise binary operations
#
# TODO Review type promotion differences
# TODO Review restricting torch implemenations of prims to not have additional functionality


# NOTE Most PyTorch elementwise binary operations do not support number x number inputs, and the few that do
#   (like add, mul, sub, and div) return tensors instead of numbers
def _elementwise_binary_check(a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> bool:
    return not (isinstance(a, Number) and isinstance(b, Number))


def _elementwise_binary_factory(name: str, *, module=torch) -> Callable:
    def fn(bsym: BoundSymbol, a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=module)
        tbsym = BoundSymbol(sym, args=(a, b), kwargs={}, output=bsym.output)

        return tbsym

    return fn


# Maps exact inputs to truncation division
def div_prim(bsym: BoundSymbol, a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> BoundSymbol:
    sym = Symbol(name="div", meta=None, _module=torch)
    kwargs = {}
    if dtypes.is_exact_dtype(dtypes.to_dtype(a)) and dtypes.is_exact_dtype(dtypes.to_dtype(b)):
        kwargs = {"rounding_mode": "trunc"}

    tbsym = BoundSymbol(sym, args=(a, b), kwargs=kwargs, output=bsym.output)
    return tbsym


# NOTE add and sub have special check and factory functions to support alpha
def _add_sub_check(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, alpha: Optional[Number] = None
) -> bool:
    return not (isinstance(a, Number) and isinstance(b, Number))


def _add_sub_factory(name: str) -> Callable:
    def fn(
        bsym: BoundSymbol,
        a: Union[TensorProxy, Number],
        b: Union[TensorProxy, Number],
        *,
        alpha: Optional[Number] = None,
    ) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=torch)
        # NOTE The alpha kwarg can ONLY be specified to PyTorch if it's not None
        if alpha is not None:
            tbsym = sym.bind(a, b, alpha=alpha, output=bsym.output)
        else:
            tbsym = sym.bind(a, b, output=bsym.output)
        return tbsym

    return fn


def polygamma(bsym: BoundSymbol, n: int, a: TensorLike) -> BoundSymbol:
    sym = Symbol(name="polygamma", meta=None, _module=torch)
    return sym.bind(n, a, output=bsym.output)


add = _add_sub_factory("add")
atan2 = _elementwise_binary_factory("atan2")
bitwise_and = _elementwise_binary_factory("bitwise_and")
bitwise_or = _elementwise_binary_factory("bitwise_or")
bitwise_xor = _elementwise_binary_factory("bitwise_xor")
copysign = _elementwise_binary_factory("copysign")
div = _elementwise_binary_factory("div")
floor_divide = _elementwise_binary_factory("floor_divide")
eq = _elementwise_binary_factory("eq")
fmod = _elementwise_binary_factory("fmod")
ge = _elementwise_binary_factory("ge")
gt = _elementwise_binary_factory("gt")
logical_and = _elementwise_binary_factory("logical_and")
le = _elementwise_binary_factory("le")
lt = _elementwise_binary_factory("lt")
mul = _elementwise_binary_factory("mul")
ne = _elementwise_binary_factory("ne")
nextafter = _elementwise_binary_factory("nextafter")
pow = _elementwise_binary_factory("pow")
# TODO Remainder bool isn't implement, so mark it as non-fusible
remainder = _elementwise_binary_factory("remainder")
sub = _add_sub_factory("sub")
zeta = _elementwise_binary_factory("zeta", module=torch.special)


def _addcmul_check(a: TensorLike, b: TensorLike, c: TensorLike, *, value: Optional[Number] = None) -> bool:
    # PyTorch doesn't support non-tensor inputs for torch.addcmul
    if not (all(isinstance(arg, TensorLike) for arg in [a, b, c]) and isinstance(value, (Number, type(None)))):
        return False

    common_dtype, _ = utils.elementwise_type_promotion(
        a, b, c, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    common_dtype = dtypes.to_strong_dtype(common_dtype)
    # PyTorch doesn't support bool or complex32 inputs for torch.addcmul
    if dtypes.is_boolean_dtype(common_dtype) or common_dtype is dtypes.complex32:
        return False

    # here the promotion is different from torch.addcmul,
    # the type promotion also considers the type of value
    if value is not None:
        if not utils.can_safe_cast_to(cast_from=utils.to_dtype(value), cast_to=common_dtype):
            return False
    return True


def _addcdiv_check(a: TensorLike, b: TensorLike, c: TensorLike, *, value: Optional[Number] = None) -> bool:
    # PyTorch doesn't support non-tensor inputs for torch.addcdiv
    if not (all(isinstance(arg, TensorLike) for arg in [a, b, c]) and isinstance(value, (Number, type(None)))):
        return False

    # PyTorch doesn't support Integer division with torch.addcdiv
    if dtypes.is_exact_dtype(dtypes.to_dtype(b)) and dtypes.is_exact_dtype(dtypes.to_dtype(c)):
        return False

    common_dtype, _ = utils.elementwise_type_promotion(
        a, b, c, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    common_dtype = dtypes.to_strong_dtype(common_dtype)
    not_supported_types = (dtypes.complex32, dtypes.float16, dtypes.bfloat16)
    if common_dtype in not_supported_types:
        return False

    # here the promotion is different from torch.addcdiv,
    # the type promotion also considers the type of value
    if value is not None:
        if not utils.can_safe_cast_to(cast_from=utils.to_dtype(value), cast_to=common_dtype):
            return False
    return True


def _addcmul_addcdiv_factory(name: str) -> Callable:
    def fn(
        bsym: BoundSymbol, a: TensorLike, b: TensorLike, c: TensorLike, *, value: Optional[Number] = None
    ) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=torch)

        if value is not None:
            tbsym = sym.bind(a, b, c, value=value, output=bsym.output)
        else:
            tbsym = sym.bind(a, b, c, output=bsym.output)
        return tbsym

    return fn


addcmul = _addcmul_addcdiv_factory("addcmul")
addcdiv = _addcmul_addcdiv_factory("addcdiv")


#
# Conditional and masking operations
#
# TODO Review type promotion differences
# TODO Review restricting torch implemenations of prims to not have additional functionality


def _elementwise_ternary_check(
    a: Union[TensorProxy, Number],
    b: Union[TensorProxy, Number],
    c: Union[TensorProxy, Number],
) -> bool:
    return not (isinstance(a, Number) and isinstance(b, Number) and isinstance(c, Number))


def _elementwise_ternary_factory(name: str) -> Callable:
    def fn(
        bsym: BoundSymbol,
        a: Union[TensorProxy, Number],
        b: Union[TensorProxy, Number],
        c: Union[TensorProxy, Number],
    ) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=torch)
        tbsym = BoundSymbol(sym, args=(a, b, c), kwargs={}, output=bsym.output)

        return tbsym

    return fn


where = _elementwise_ternary_factory("where")


def _clamp_check(
    a: TensorLike,
    min: None | Number | TensorLike = None,
    max: None | Number | TensorLike = None,
) -> bool:
    # torch supports combination of arguments:
    # (Tensor input, Tensor min, Tensor max)
    # (Tensor input, Number min, Number max)
    if isinstance(min, TensorLike) and isinstance(max, Number):
        return False
    if isinstance(min, Number) and isinstance(max, TensorLike):
        return False
    return True


def clamp(
    bsym: BoundSymbol,
    a: TensorLike,
    min: None | Number | TensorLike = None,
    max: None | Number | TensorLike = None,
) -> BoundSymbol:
    sym = Symbol(name="clamp", meta=None, _module=torch)

    tbsym = sym.bind(a, min, max, output=bsym.output)
    return tbsym


def masked_fill(bsym: BoundSymbol, a, mask, value):
    sym = Symbol(name="masked_fill", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, mask, value), kwargs={}, output=bsym.output)

    return tbsym


def _tril_check(a: TensorProxy, diagonal: int = 0, fill_value: None | Number = None):
    return fill_value is None


def tril(bsym: BoundSymbol, a, diagonal: int = 0, *, fill_value: None | Number = None):
    sym = Symbol(name="tril", meta=None, _module=torch)
    return sym.bind(a, diagonal, output=bsym.output)


#
# Reduction operations
#
# TODO Capture torch reductions for amax, amin, prod, and sum


def _prim_reduction_factory(name_or_symbol: Union[str, Symbol]) -> Callable:
    def fn(bsym: BoundSymbol, a: TensorProxy, dims, *, output_dtype: Optional[dtypes.dtype] = None) -> BoundSymbol:
        sym = name_or_symbol
        if isinstance(name_or_symbol, str):
            sym = Symbol(name=name_or_symbol, meta=None, _module=torch)

        output_dtype = ltorch.to_torch_dtype(output_dtype)

        kwargs: dict
        if output_dtype is not None:
            kwargs = {
                "dtype": output_dtype,
            }
        else:
            kwargs = {}

        tbsym = BoundSymbol(sym, args=(a, dims), kwargs=kwargs, output=bsym.output)

        return tbsym

    return fn


amax_prim = _prim_reduction_factory("amax")
amin_prim = _prim_reduction_factory("amin")
prod_prim = _prim_reduction_factory(Symbol(name="prod", meta=None, _module=torch._refs))
sum_prim = _prim_reduction_factory("sum")


# TODO Add type annotations
# TODO Review if this needs more of a wrapper around torch.var to implement the prim properly
# TODO Implement output dtype properly
def var_prim(bsym: BoundSymbol, a: TensorProxy, dims, *, correction: int) -> BoundSymbol:
    sym = Symbol(name="var", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dims), kwargs={"correction": correction}, output=bsym.output)

    return tbsym


def var_mean_prim(bsym: BoundSymbol, a: TensorProxy, dims, *, correction: int) -> BoundSymbol:
    sym = Symbol(name="var_mean", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dims), kwargs={"correction": correction}, output=bsym.output)

    return tbsym


# NOTE var does not allow keepdim to be specified if dim is not specified
# NOTE This is the direct translation of torch.var
def _var_factory(name: str) -> Callable:
    def fn(
        bsym: BoundSymbol,
        a,
        dim=None,
        unbiased: Optional[bool] = None,
        keepdim: bool = False,
        *,
        correction: Optional[int] = None,
    ) -> BoundSymbol:
        utils.check(
            correction is None or unbiased is None,
            lambda: f"Cannot specify both {unbiased=} and {correction=} when calling var",
        )

        sym = Symbol(name=name, meta=None, _module=torch)

        # Explicitly specifies dimensions (if unspecified) if keepdim or correction must be specified
        if dim is None and (keepdim is not None or correction is not None):
            dim = tuple(range(a.ndim))

        args = tuple(x for x in (a, dim) if x is not None)
        kwargs = {
            "correction": correction,
        }

        # Adds keepdim
        if keepdim:
            kwargs["keepdim"] = keepdim

        # NOTE PyTorch does not allow both correction and unbiased to be specified
        if unbiased is not None:
            kwargs["unbiased"] = unbiased
            del kwargs["correction"]

        tbsym = BoundSymbol(sym, args=args, kwargs=kwargs, output=bsym.output)

        return tbsym

    return fn


var = _var_factory("var")
var_mean = _var_factory("var_mean")

#
# Matmul operations
#


def linear(bsym: BoundSymbol, a: TensorProxy, w: TensorProxy, bias: TensorProxy) -> BoundSymbol:
    sym = Symbol(name="linear", meta=None, _module=torch.nn.functional)
    tbsym = BoundSymbol(sym, args=(a, w, bias), kwargs={}, output=bsym.output)

    return tbsym


def matmul(bsym: BoundSymbol, a: TensorProxy, b: TensorProxy) -> BoundSymbol:
    sym = Symbol(name="matmul", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, b), kwargs={}, output=bsym.output)

    return tbsym


#
# NN operations
#
def convolution(
    bsym: BoundSymbol,
    a: TensorLike,
    weight: TensorLike,
    bias: Optional[TensorLike],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: Number,
    output_padding: Sequence[int],
    groups: int,
) -> BoundSymbol:
    sym = Symbol(name="convolution", meta=None, _module=torch)
    return sym.bind(
        a, weight, bias, stride, padding, dilation, bool(transposed), output_padding, groups, output=bsym.output
    )


def conv1d(
    bsym: BoundSymbol,
    a: TensorLike,
    weight: TensorLike,
    bias: Optional[TensorLike] = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> BoundSymbol:
    sym = Symbol(name="conv1d", meta=None, _module=torch.nn.functional)
    return sym.bind(a, weight, bias, stride, padding, dilation, groups, output=bsym.output)


def conv2d(
    bsym: BoundSymbol,
    a: TensorLike,
    weight: TensorLike,
    bias: Optional[TensorLike] = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> BoundSymbol:
    sym = Symbol(name="conv2d", meta=None, _module=torch.nn.functional)
    return sym.bind(a, weight, bias, stride, padding, dilation, groups, output=bsym.output)


def conv3d(
    bsym: BoundSymbol,
    a: TensorLike,
    weight: TensorLike,
    bias: Optional[TensorLike] = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> BoundSymbol:
    sym = Symbol(name="conv3d", meta=None, _module=torch.nn.functional)
    return sym.bind(a, weight, bias, stride, padding, dilation, groups, output=bsym.output)


def cross_entropy(
    bsym: BoundSymbol,
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    sym = Symbol(name="cross_entropy", _module=torch.nn.functional)
    return sym.bind(
        input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing, output=bsym.output
    )


def _cross_entropy_backward_helper(
    g: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    reduction: str,
    ignore_index: int,
    label_smoothing: int,
) -> torch.Tensor:
    # forward - given input and target
    # a = log_softmax(input, dim)
    # return cross_entropy(a, target, weight, reduction, ignore_index, label_smoothing)

    # backward - given grad_cross_entropy and saved_tensors
    # grad_a = torch.ops.aten.nll_loss_backward(grad, a, target, weight, reduction, ignore_index, total_weight)
    # return torch.ops.aten._log_softmax_backward_data(grad_a, a, dim, a.scalar_type())

    if reduction == "none":
        reduction_idx = 0
    elif reduction == "mean":
        reduction_idx = 1
    elif reduction == "sum":
        reduction_idx = 2
    else:
        reduction_idx = -1

    utils.check(
        reduction_idx > -1 and reduction_idx < 3,
        lambda: f"{reduction} is not a valid value for reduction parameter.",
    )

    # TODO Add support nll_loss_nd, weight tensor, and label_smoothing options.
    # See https://github.com/Lightning-AI/lightning-thunder/issues/704
    utils.check(input.ndim <= 2 and target.ndim <= 1, lambda: f"multi-dimension cross-entropy is not supported.")

    utils.check(weight is None, lambda: f"weight tensor argument is not supported.")

    utils.check(label_smoothing == 0.0, lambda: f"label smoothing values not equal to zero are not supported.")

    dim = 0 if input.dim() == 1 else 1
    a = torch.log_softmax(input, dim, input.dtype)

    if weight is not None:
        total_weight = torch.sum(weight)
    elif reduction == "none":
        total_weight = torch.tensor(0.0, dtype=input.dtype, device=input.device)
    elif reduction == "sum" or reduction == "mean":
        total_weight = torch.sum(torch.ne(target, ignore_index)).to(dtype=input.dtype, device=input.device)

    g_a = torch.ops.aten.nll_loss_backward(g, a, target, weight, reduction_idx, ignore_index, total_weight)
    return torch.ops.aten._log_softmax_backward_data(g_a, a, dim, input.dtype)


def cross_entropy_backward(
    bsym: BoundSymbol,
    grad: TensorProxy,
    input: TensorProxy,
    target: TensorProxy,
    weight: TensorProxy = None,
    reduction: str = "mean",
    ignore_index: Number = -100,
    label_smoothing: Number = 0.0,
) -> BoundSymbol:
    sym = Symbol(name="cross_entropy_backward", meta=None)
    ctx: Dict[str, Any] = {"cross_entropy_backward": _cross_entropy_backward_helper}
    return sym.bind(
        grad, input, target, weight, reduction, ignore_index, label_smoothing, output=bsym.output, _call_ctx=ctx
    )


def dropout(bsym: BoundSymbol, a: TensorProxy, p: Number = 0.5, training=True, inplace=False) -> BoundSymbol:
    sym = Symbol(name="dropout", meta=None, _module=torch.nn.functional)

    kwargs = {
        "training": training,
        "inplace": inplace,
    }

    tbsym = BoundSymbol(sym, args=(a, p), kwargs=kwargs, output=bsym.output)
    return tbsym


def embedding(
    bsym: BoundSymbol,
    a,
    weight,
    padding_idx=-1,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
) -> BoundSymbol:
    sym = Symbol(name="embedding", meta=None, _module=torch.nn.functional)

    kwargs = {
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
    }

    tbsym = BoundSymbol(sym, args=(a, weight), kwargs=kwargs, output=bsym.output)
    return tbsym


def embedding_backward(
    bsym: BoundSymbol,
    grad,
    indices,
    num_weights,
    padding_idx,
    scale_grad_by_freq,
    sparse,
) -> BoundSymbol:
    sym = Symbol(name="embedding_backward", meta=None, _module=torch.ops.aten)
    tbsym = BoundSymbol(
        sym,
        args=(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse),
        kwargs={},
        output=bsym.output,
    )
    return tbsym


def gelu(bsym: BoundSymbol, a: TensorProxy, *, approximate: str = "none") -> BoundSymbol:
    sym = Symbol(name="gelu", meta=None, _module=torch.nn.functional)
    return sym.bind(a, approximate=approximate, output=bsym.output)


def group_norm(
    bsym: BoundSymbol,
    a: TensorProxy,
    num_groups: int,
    weight: Optional[TensorProxy] = None,
    bias: Optional[TensorProxy] = None,
    eps: float = 1e-5,
) -> BoundSymbol:
    sym = Symbol(name="group_norm", meta=None, _module=torch.nn.functional)
    return sym.bind(a, num_groups, weight, bias, eps, output=bsym.output)


# TODO: add support for other modes and params
def interpolate(
    bsym: BoundSymbol,
    a: TensorProxy,
    size: int | Sequence[int] | None = None,
    scale_factor: float | Sequence[float] | None = None,
    mode: str = "nearest",
) -> BoundSymbol:
    sym = Symbol(name="interpolate", meta=None, _module=torch.nn.functional)
    return sym.bind(a, size, scale_factor, mode, output=bsym.output)


def _interpolate_check(
    a: TensorProxy,
    size: int | Sequence[int] | None = None,
    scale_factor: float | Sequence[float] | None = None,
    mode: str = "nearest",
) -> bool:
    # PyTorch only supports 3D to 5D inputs only.
    # Our decomposition does not have such limitations
    return 3 <= a.ndim <= 55


def layer_norm(bsym: BoundSymbol, a, normalized_shape, weight=None, bias=None, eps: Number = 1e-5):
    sym = Symbol(name="layer_norm", meta=None, _module=torch.nn.functional)
    tbsym = BoundSymbol(sym, args=(a, normalized_shape, weight, bias, eps), kwargs={}, output=bsym.output)

    return tbsym


def logsumexp(bsym: BoundSymbol, a: TensorProxy, dim: Number, keepdim: bool = False) -> BoundSymbol:
    sym = Symbol(name="logsumexp", meta=None, _module=torch)
    return sym.bind(a, dim, keepdim, output=bsym.output)


def log_softmax(bsym: BoundSymbol, a: TensorProxy, dim: Number, dtype=None) -> BoundSymbol:
    torch_dtype = None
    if dtype is not None:
        torch_dtype = ltorch.to_torch_dtype(dtype)
    sym = Symbol(name="log_softmax", meta=None, _module=torch)
    return sym.bind(a, dim, dtype=torch_dtype, output=bsym.output)


def _log_softmax_backward_helper(g: torch.Tensor, output: torch.Tensor, dim: Number, dtype) -> torch.Tensor:
    return torch.ops.aten._log_softmax_backward_data(g, output, dim, dtype)


def log_softmax_backward(bsym: BoundSymbol, grad: TensorProxy, output: TensorProxy, dim: Number, dtype) -> BoundSymbol:
    torch_dtype = None
    if dtype is not None:
        torch_dtype = ltorch.to_torch_dtype(dtype)

    sym = Symbol(name="log_softmax_backward", meta=None)
    ctx: Dict[str, Any] = {"log_softmax_backward": _log_softmax_backward_helper}
    return sym.bind(grad, output, dim, torch_dtype, output=bsym.output, _call_ctx=ctx)


def _nll_loss_check(
    a: TensorProxy,
    target: TensorProxy,
    weight: Optional[TensorProxy] = None,
    size_average: Optional[bool] = None,
    ignore_index: Number = -1,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> bool:
    # NOTE size_average parameter is deprecated in PyTorch
    if size_average != None:
        return False

    # NOTE reduce parameter is deprecated in PyTorch.
    if reduce != None:
        return False

    return True


def nll_loss(
    bsym: BoundSymbol,
    a: TensorProxy,
    target: TensorProxy,
    weight: Optional[TensorProxy] = None,
    size_average: Optional[bool] = None,
    ignore_index: Number = -1,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> BoundSymbol:
    sym = Symbol(name="nll_loss", _module=torch.nn.functional)
    return sym.bind(a, target, weight, size_average, ignore_index, reduce, reduction, output=bsym.output)


def _nll_loss_backward_helper(
    g: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    reduction: str,
    ignore_index: int,
    total_weight: torch.Tensor,
) -> torch.Tensor:
    if reduction == "none":
        reduction_idx = 0
    elif reduction == "mean":
        reduction_idx = 1
    elif reduction == "sum":
        reduction_idx = 2
    else:
        reduction_idx = -1

    utils.check(
        reduction_idx > -1 and reduction_idx < 3,
        lambda: f"{reduction} is not a valid value for reduction parameter.",
    )

    if total_weight is None:
        total_weight = torch.tensor(0.0, dtype=torch.float64, device=input.device)
    else:
        # ATen expect total weight to have double dtype
        total_weight = total_weight.to(dtype=input.dtype)

    if input.ndim <= 2:
        return torch.ops.aten.nll_loss_backward(g, input, target, weight, reduction_idx, ignore_index, total_weight)
    elif input.ndim == 4:
        return torch.ops.aten.nll_loss2d_backward(g, input, target, weight, reduction_idx, ignore_index, total_weight)
    else:
        # input.ndim == 3 or input.ndim > 4
        # extract shape information
        N = input.shape[0]
        C = input.shape[1]
        no_C_shape = list(input.shape)
        C_dim = 1 if input.ndim >= 2 else 0
        no_C_shape.pop(C_dim)

        # support empty batches, see #15870
        if input.numel() > 0:
            input_ = input.reshape([N, C, 1, -1])
        else:
            input_ = input.reshape([N, C, 0, 0])

        if target.numel() > 0:
            target_ = target.reshape([N, 1, -1])
        else:
            target_ = target.reshape([N, 0, 0])

        if reduction != "none":
            return torch.ops.aten.nll_loss2d_backward(
                g, input_, target_, weight, reduction_idx, ignore_index, total_weight
            )
        else:
            # g must have same dimension as target.
            if g.numel() > 0:
                g_ = g.reshape([N, 1, -1])
            else:
                g_ = g.reshape([N, 0, 0])

            result = torch.ops.aten.nll_loss2d_backward(
                g_, input_, target_, weight, reduction_idx, ignore_index, total_weight
            )
            return result.reshape(no_C_shape)


def nll_loss_backward(
    bsym: BoundSymbol,
    grad: TensorProxy,
    input: TensorProxy,
    target: TensorProxy,
    weight: TensorProxy,
    reduction: str,
    ignore_index: int,
    total_weight: TensorProxy,
) -> BoundSymbol:
    sym = Symbol(name="nll_loss_backward", meta=None)
    ctx: Dict[str, Any] = {"nll_loss_backward": _nll_loss_backward_helper}
    return sym.bind(
        grad, input, target, weight, reduction, ignore_index, total_weight, output=bsym.output, _call_ctx=ctx
    )


def relu(bsym: BoundSymbol, a: TensorProxy, inplace=False) -> BoundSymbol:
    sym = Symbol(name="relu", meta=None, _module=torch.nn.functional)
    # NOTE: inplace is ignored since only
    # inplace=False is supported and it has a default value.
    return sym.bind(a, output=bsym.output)


def relu6(bsym: BoundSymbol, a: TensorProxy, inplace=False) -> BoundSymbol:
    sym = Symbol(name="relu6", meta=None, _module=torch.nn.functional)
    # NOTE: inplace is ignored since only
    # inplace=False is supported and it has a default value.
    return sym.bind(a, output=bsym.output)


def selu(bsym: BoundSymbol, a: TensorProxy, inplace=False) -> BoundSymbol:
    sym = Symbol(name="selu", meta=None, _module=torch.nn.functional)
    # NOTE: inplace is ignored since only
    # inplace=False is supported and it has a default value.
    return sym.bind(a, output=bsym.output)


def softmax(bsym: BoundSymbol, a: TensorProxy, dim: Number, dtype=None) -> BoundSymbol:
    torch_dtype = None
    if dtype is not None:
        torch_dtype = ltorch.to_torch_dtype(dtype)
    sym = Symbol(name="softmax", meta=None, _module=torch)
    return sym.bind(a, dim, dtype=torch_dtype, output=bsym.output)


def _scaled_dot_product_attention_check(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
) -> bool:
    # TODO: Model PyTorch's choice of efficient kernels and fallbacks
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
    if scale is not None and LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
        return False
    return True


def scaled_dot_product_attention(
    bsym: BoundSymbol,
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
) -> BoundSymbol:
    sym = Symbol(name="scaled_dot_product_attention", meta=None, _module=torch.nn.functional, id=bsym.sym.id)

    kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": bool(is_causal),
    }
    if LooseVersion(torch.__version__) >= LooseVersion("2.1.0"):
        kwargs["scale"] = scale
    else:
        utils.check(
            scale is None,
            lambda: f"scaled_dot_product_attention with scale argument requires PyTorch >= 2.1.0",
        )

    tbsym = BoundSymbol(sym, args=(query, key, value), kwargs=kwargs, output=bsym.output)
    return tbsym


def _grad_forward_scaled_dot_product_attention_check(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: Optional[TensorLike] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> bool:
    tensor_inputs = [query, key, value]
    if attn_mask is not None:
        tensor_inputs.append(attn_mask)

    # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention' with arguments from the 'CPU' backend.
    if any(map(lambda a: a.device is devices.cpu, tensor_inputs)):
        return False

    # TODO: Model PyTorch's choice of efficient kernels and fallbacks
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
    if scale is not None and LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
        return False
    return True


def grad_forward_scaled_dot_product_efficient_attention_helper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    # When a boolean mask is used, it needs to be converted to an additive mask where zero'd elements are filled
    # with a very negative value that should become ~0 after softmax
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = torch.masked_fill(torch.zeros_like(attn_mask, dtype=query.dtype), attn_mask == False, -math.inf)

    # Reference: https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L394-L415
    return torch.ops.aten._scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_mask,
        compute_logsumexp := True,
        dropout_p,
        is_causal,
        scale=scale,
    )


def grad_forward_scaled_dot_product_efficient_attention(
    bsym: BoundSymbol,
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask: Optional[TensorProxy] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> BoundSymbol:
    sym = Symbol(name="grad_forward_scaled_dot_product_efficient_attention", meta=None)
    ctx: Dict[str, Any] = {
        "grad_forward_scaled_dot_product_efficient_attention": grad_forward_scaled_dot_product_efficient_attention_helper
    }
    return sym.bind(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        output=bsym.output,
        _call_ctx=ctx,
    )


def _scaled_dot_product_efficient_attention_backward_helper(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    grad_input_mask = [a.requires_grad for a in (query, key, value)]
    if attn_mask is None:
        grad_input_mask.append(False)
    else:
        grad_input_mask.append(attn_mask.requires_grad)
        # When a boolean mask is used, it needs to be converted to an additive mask where zero'd elements are filled
        # with a very negative value that should become ~0 after softmax
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.masked_fill(torch.zeros_like(attn_mask, dtype=query.dtype), attn_mask == False, -math.inf)

    # Reference: https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L394-L415
    return torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out,
        query,
        key,
        value,
        attn_mask,
        out,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask,
        is_causal,
        scale=scale,
    )


def _scaled_dot_product_efficient_attention_backward_check(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> bool:
    tensor_inputs = [query, key, value]
    if attn_mask is not None:
        tensor_inputs.append(attn_mask)

    # NOTE: NotImplementedError: Could not run 'aten::_scaled_dot_product_efficient_attention' with arguments from the 'CPU' backend.
    if any(map(lambda a: a.device is devices.cpu, tensor_inputs)):
        return False

    # TODO: Model PyTorch's choice of efficient kernels and fallbacks
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
    if scale is not None and LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
        return False
    return True


def scaled_dot_product_efficient_attention_backward(
    bsym: BoundSymbol,
    grad_out: TensorProxy,
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask: Optional[TensorProxy],
    out: TensorProxy,
    logsumexp: TensorProxy,
    philox_seed: TensorProxy,
    philox_offset: TensorProxy,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: Optional[float],
) -> BoundSymbol:
    sym = Symbol(name="scaled_dot_product_efficient_attention_backward", meta=None)
    ctx: Dict[str, Any] = {
        "scaled_dot_product_efficient_attention_backward": _scaled_dot_product_efficient_attention_backward_helper
    }
    return sym.bind(
        grad_out,
        query,
        key,
        value,
        attn_mask,
        out,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        is_causal,
        scale,
        output=bsym.output,
        _call_ctx=ctx,
    )


#
# Distributed Operations
#
# NOTE DISTRIBUTED AVAILABILITY
# PyTorch is often built without distributed support, which can be queried for using
#   torch.distributed.is_available(). When PyTorch is built without distributed then we
#   want to avoid accessing any parts of the torch.distributed module except
#   the is_available() function.

if torch.distributed.is_available():

    def all_reduce_prim_helper(
        a: torch.Tensor,
        op: torch.distributed.ReduceOp,
        group: torch.distributed.ProcessGroup,
        do_async: bool,
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        c = a.clone()

        handle = torch.distributed.all_reduce(c, op, group, do_async)

        # NOTE This returns (handle, tensor reference), which is how FutureTensorProxies
        #   are currently modeled by executors
        if do_async:
            return handle, c

        # NOTE do_async is False
        return c

    def all_reduce_prim(
        bsym: BoundSymbol,
        a: TensorProxy,
        op: DistributedReduceOps,
        group: torch.distributed.ProcessGroup,
        do_async: Number,
    ) -> BoundSymbol:
        sym = Symbol(name="all_reduce_prim_helper", meta=None)
        ctx: dict[str, Any] = {"all_reduce_prim_helper": all_reduce_prim_helper}
        do_async = bool(do_async)

        op = ltorch.to_torch_distributed_reduce_op(op)

        return sym.bind(a, op, group, do_async, output=bsym.output, _call_ctx=ctx)

    def all_gather_prim_helper(
        a: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        async_op: bool,
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        out = torch.empty((group.size() * a.shape[0],) + a.shape[1:], dtype=a.dtype, device=a.device)
        handle = torch.distributed.all_gather_into_tensor(out, a, group, async_op)
        if async_op:
            return handle, out
        return out

    def all_gather_prim(
        bsym: BoundSymbol,
        a: TensorProxy,
        group: torch.distributed.ProcessGroup,
        async_op: Number,
    ) -> BoundSymbol:
        sym = Symbol(name="all_gather_prim_helper", meta=None)
        ctx: dict[str, Any] = {"all_gather_prim_helper": all_gather_prim_helper}
        async_op = bool(async_op)

        return sym.bind(a, group, async_op, output=bsym.output, _call_ctx=ctx)

    def broadcast_prim_helper(
        a: torch.Tensor,
        src: int,
        group: torch.distributed.ProcessGroup,
        async_op: bool,
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        out = a.clone()
        handle = torch.distributed.broadcast(out, src, group, async_op)
        if async_op:
            return handle, out
        return out

    def broadcast_prim(
        bsym: BoundSymbol,
        a: TensorProxy,
        src: int,
        group: torch.distributed.ProcessGroup,
        async_op: Number,
    ) -> BoundSymbol:
        sym = Symbol(name="broadcast_prim_helper", meta=None)
        ctx: dict[str, Any] = {"broadcast_prim_helper": broadcast_prim_helper}
        async_op = bool(async_op)

        return sym.bind(a, src, group, async_op, output=bsym.output, _call_ctx=ctx)

    def reduce_scatter_prim_helper(
        a: torch.Tensor,
        op: torch.distributed.ReduceOp,
        group: torch.distributed.ProcessGroup,
        async_op: bool,
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        out = torch.empty((a.shape[0] // group.size(),) + a.shape[1:], dtype=a.dtype, device=a.device)
        handle = torch.distributed.reduce_scatter_tensor(out, a, op, group, async_op)
        if async_op:
            return handle, out
        return out

    def reduce_scatter_prim(
        bsym: BoundSymbol,
        a: TensorProxy,
        op: DistributedReduceOps,
        group: torch.distributed.ProcessGroup,
        async_op: Number,
    ) -> BoundSymbol:
        sym = Symbol(name="reduce_scatter_prim_helper", meta=None)
        ctx: dict[str, Any] = {"reduce_scatter_prim_helper": reduce_scatter_prim_helper}
        async_op = bool(async_op)

        op = ltorch.to_torch_distributed_reduce_op(op)

        return sym.bind(a, op, group, async_op, output=bsym.output, _call_ctx=ctx)

    # NOTE This is a very particular implementation of wait that may need to be
    #   generalized in the future
    # NOTE The implementation of wait actually models the FutureTensorProxy as a tuple
    #   of (handle, tensor reference), there's no conflict with the trace representation
    #   with doing this, as the trace representation treates FutureTensorProxy as opaque
    # NOTE This way of representing FutureTensorProxies may need to change in the future
    def wait_helper(a: tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]) -> torch.Tensor:
        handle, tensor_ref = a
        handle.wait()
        return tensor_ref

    def wait_prim(bsym: BoundSymbol, a: FutureTensorProxy) -> BoundSymbol:
        sym = Symbol(name="wait_helper", meta=None)
        ctx: dict[str, Any] = {"wait_helper": wait_helper}

        return sym.bind(a, output=bsym.output, _call_ctx=ctx)

else:

    def all_gather_prim(bsym: BoundSymbol, a: TensorProxy, group: Any, async_op: bool) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def all_reduce_prim(bsym: BoundSymbol, a: TensorProxy, op: Any, group: Any, do_async: bool) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def broadcast_prim(bsym: BoundSymbol, a: TensorProxy, src: int, group: Any, async_op: bool) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def reduce_scatter_prim(bsym: BoundSymbol, a: TensorProxy, op: Any, group: Any, async_op: bool) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def wait_prim(bsym: BoundSymbol, a: FutureTensorProxy) -> BoundSymbol:
        utils.check(False, lambda: f"torch.distributed is not available")


# TODO Refine prim ops to have less functionality to better debug errors
# Maps from symbol ids to a tuple of (is_fusible, translate) callables
_ops_map.update(
    {
        # Data movement operations
        PrimIDs.CONVERT_ELEMENT_TYPE: (_always_executable, convert_element_type),
        PrimIDs.DEVICE_PUT: (_always_executable, device_put),
        "torch.Tensor.to": (_always_executable, to),
        # Tensor creation operations
        "torch.arange": (_always_executable, arange),
        PrimIDs.EXOGENOUS_LIKE: (_always_executable, exogenous_like),
        PrimIDs.FULL: (_always_executable, full),
        "torch.full_like": (_always_executable, full_like),
        PrimIDs.IOTA: (_always_executable, iota),
        PrimIDs.UNIFORM: (_always_executable, uniform_prim),
        PrimIDs.UNIFORM_PHILOX: (_uniform_philox_check, uniform_philox_prim),
        "torch.zeros_like": (_always_executable, zeros_like),
        # Shape operations
        PrimIDs.BROADCAST_IN_DIM: (_always_executable, broadcast_in_dim),
        "torch.cat": (_always_executable, cat),
        PrimIDs.CAT: (_always_executable, cat),
        "torch.Tensor.contiguous": (_always_executable, contiguous),
        PrimIDs.STRIDE_ORDER: (_always_executable, stride_order),
        "torch.chunk": (_always_executable, chunk),
        PrimIDs.FLIP: (_always_executable, flip),
        "torch.flip": (_always_executable, flip),
        "torch.diagonal": (_always_executable, diagonal),
        "torch.Tensor.expand": (_always_executable, expand),
        "torch.flatten": (_always_executable, flatten),
        "torch.unbind": (_always_executable, unbind),
        "torch.Tensor.__getitem__": (_always_executable, getitem),
        "torch.movedim": (_always_executable, movedim),
        PrimIDs.PAD: (_always_executable, pad),
        PrimIDs.RESHAPE: (_always_executable, reshape),
        "torch.reshape": (_always_executable, reshape),
        "torch.Tensor.repeat": (_always_executable, repeat),
        PrimIDs.SLICE: (_always_executable, slice_prim),
        "torch.split": (_always_executable, split),
        "torch.squeeze": (_always_executable, squeeze),
        PrimIDs.SQUEEZE: (_always_executable, squeeze),
        PrimIDs.TAKE: (_always_executable, take),
        PrimIDs.INDEX_ADD: (_always_executable, index_add),
        PrimIDs.TAKE_ALONG_AXIS: (_always_executable, take_along_axis),
        PrimIDs.SCATTER_ADD: (_always_executable, scatter_add),
        "torch.tensor_split": (_always_executable, tensor_split),
        "torch.transpose": (_always_executable, transpose),
        PrimIDs.TRANSPOSE: (_always_executable, prim_transpose),
        "torch.permute": (_always_executable, permute),
        "torch.unsqueeze": (_always_executable, unsqueeze),
        PrimIDs.VIEW: (_always_executable, view),
        # NOTE torch.Tensor.view is intentionally not lowered because
        #   whether a view is possible depends on a tensor's strides, which
        #   we do not enforce
        # "torch.Tensor.view": (_always_executable, view),
        # Elementwise unary operations
        "torch.abs": (_elementwise_unary_check, torch_abs),
        PrimIDs.ABS: (_elementwise_unary_check, torch_abs),
        "torch.acos": (_elementwise_unary_check, acos),
        PrimIDs.ACOS: (_elementwise_unary_check, acos),
        "torch.acosh": (_elementwise_unary_check, acosh),
        PrimIDs.ACOSH: (_elementwise_unary_check, acosh),
        "torch.asin": (_elementwise_unary_check, asin),
        PrimIDs.ASIN: (_elementwise_unary_check, asin),
        "torch.asinh": (_elementwise_unary_check, asinh),
        PrimIDs.ASINH: (_elementwise_unary_check, asinh),
        "torch.atan": (_elementwise_unary_check, atan),
        PrimIDs.ATAN: (_elementwise_unary_check, atan),
        "torch.atanh": (_elementwise_unary_check, atanh),
        PrimIDs.ATANH: (_elementwise_unary_check, atanh),
        "torch.bitwise_not": (_elementwise_unary_check, bitwise_not),
        PrimIDs.BITWISE_NOT: (_elementwise_unary_check, bitwise_not),
        "torch.ceil": (_elementwise_unary_check, ceil),
        PrimIDs.CEIL: (_elementwise_unary_check, ceil),
        "torch.cos": (_elementwise_unary_check, cos),
        PrimIDs.COS: (_elementwise_unary_check, cos),
        "torch.cosh": (_elementwise_unary_check, cosh),
        PrimIDs.COSH: (_elementwise_unary_check, cosh),
        "torch.digamma": (_elementwise_unary_check, digamma),
        PrimIDs.DIGAMMA: (_elementwise_unary_check, digamma),
        "torch.erf": (_elementwise_unary_check, erf),
        PrimIDs.ERF: (_elementwise_unary_check, erf),
        "torch.erfc": (_elementwise_unary_check, erfc),
        PrimIDs.ERFC: (_elementwise_unary_check, erfc),
        "torch.erfinv": (_elementwise_unary_check, erfinv),
        PrimIDs.ERFINV: (_elementwise_unary_check, erfinv),
        "torch.exp": (_elementwise_unary_check, exp),
        PrimIDs.EXP: (_elementwise_unary_check, exp),
        "torch.exp2": (_elementwise_unary_check, exp2),
        PrimIDs.EXP2: (_elementwise_unary_check, exp2),
        "torch.expm1": (_elementwise_unary_check, expm1),
        PrimIDs.EXPM1: (_elementwise_unary_check, expm1),
        "torch.floor": (_elementwise_unary_check, floor),
        PrimIDs.FLOOR: (_elementwise_unary_check, floor),
        "torch.isfinite": (_elementwise_unary_check, isfinite),
        PrimIDs.ISFINITE: (_elementwise_unary_check, isfinite),
        "torch.lgamma": (_elementwise_unary_check, lgamma),
        PrimIDs.LGAMMA: (_elementwise_unary_check, lgamma),
        "torch.log": (_elementwise_unary_check, log),
        PrimIDs.LOG: (_elementwise_unary_check, log),
        "torch.log10": (_elementwise_unary_check, log10),
        PrimIDs.LOG10: (_elementwise_unary_check, log10),
        "torch.log1p": (_elementwise_unary_check, log1p),
        PrimIDs.LOG1P: (_elementwise_unary_check, log1p),
        "torch.log2": (_elementwise_unary_check, log2),
        PrimIDs.LOG2: (_elementwise_unary_check, log2),
        "torch.ndtri": (_elementwise_unary_check, ndtri),
        PrimIDs.NDTRI: (_elementwise_unary_check, ndtri),
        "torch.neg": (_elementwise_unary_check, neg),
        PrimIDs.NEG: (_elementwise_unary_check, neg),
        "torch.reciprocal": (_elementwise_unary_check, reciprocal),
        PrimIDs.RECIPROCAL: (_elementwise_unary_check, reciprocal),
        "torch.relu": (_always_executable, relu),
        "torch.relu6": (_always_executable, relu6),
        "torch.selu": (_always_executable, selu),
        "torch.round": (_elementwise_unary_check, torch_round),
        PrimIDs.ROUND: (_elementwise_unary_check, torch_round),
        "torch.rsqrt": (_elementwise_unary_check, rsqrt),
        PrimIDs.RSQRT: (_elementwise_unary_check, rsqrt),
        "torch.sign": (_elementwise_unary_check, sgn),
        PrimIDs.SIGN: (_elementwise_unary_check, sgn),
        "torch.signbit": (_elementwise_unary_check, signbit),
        PrimIDs.SIGNBIT: (_elementwise_unary_check, signbit),
        "torch.sin": (_elementwise_unary_check, sin),
        PrimIDs.SIN: (_elementwise_unary_check, sin),
        "torch.sinh": (_elementwise_unary_check, sinh),
        PrimIDs.SINH: (_elementwise_unary_check, sinh),
        "torch.sqrt": (_elementwise_unary_check, sqrt),
        PrimIDs.SQRT: (_elementwise_unary_check, sqrt),
        "torch.tan": (_elementwise_unary_check, tan),
        PrimIDs.TAN: (_elementwise_unary_check, tan),
        "torch.tanh": (_elementwise_unary_check, tanh),
        PrimIDs.TANH: (_elementwise_unary_check, tanh),
        "torch.trunc": (_elementwise_unary_check, trunc),
        PrimIDs.TRUNC: (_elementwise_unary_check, trunc),
        "torch.real": (_elementwise_unary_check, real),
        PrimIDs.REAL: (_elementwise_unary_check, real),
        # Elementwise binary operations
        "torch.add": (_add_sub_check, add),
        PrimIDs.ADD: (_elementwise_binary_check, add),
        "torch.atan2": (_elementwise_binary_check, atan2),
        PrimIDs.ATAN2: (_elementwise_binary_check, atan2),
        "torch.bitwise_and": (_elementwise_binary_check, bitwise_and),
        PrimIDs.BITWISE_AND: (_elementwise_binary_check, bitwise_and),
        "torch.bitwise_or": (_elementwise_binary_check, bitwise_or),
        PrimIDs.BITWISE_OR: (_elementwise_binary_check, bitwise_or),
        "torch.bitwise_xor": (_elementwise_binary_check, bitwise_xor),
        PrimIDs.BITWISE_XOR: (_elementwise_binary_check, bitwise_xor),
        "torch.copysign": (_elementwise_binary_check, copysign),
        "torch.div": (_elementwise_binary_check, div),
        PrimIDs.DIV: (_elementwise_binary_check, div_prim),
        "torch.eq": (_elementwise_binary_check, eq),
        PrimIDs.EQ: (_elementwise_binary_check, eq),
        "torch.floor_divide": (_elementwise_binary_check, floor_divide),
        "torch.fmod": (_elementwise_binary_check, fmod),
        PrimIDs.FMOD: (_elementwise_binary_check, fmod),
        "torch.ge": (_elementwise_binary_check, ge),
        PrimIDs.GE: (_elementwise_binary_check, ge),
        "torch.gt": (_elementwise_binary_check, gt),
        PrimIDs.GT: (_elementwise_binary_check, gt),
        "torch.logical_and": (_elementwise_binary_check, logical_and),
        "torch.le": (_elementwise_binary_check, le),
        PrimIDs.LE: (_elementwise_binary_check, le),
        "torch.lt": (_elementwise_binary_check, lt),
        PrimIDs.LT: (_elementwise_binary_check, lt),
        "torch.mul": (_elementwise_binary_check, mul),
        PrimIDs.MUL: (_elementwise_binary_check, mul),
        "torch.ne": (_elementwise_binary_check, ne),
        PrimIDs.NE: (_elementwise_binary_check, ne),
        "torch.nextafter": (_elementwise_binary_check, nextafter),
        PrimIDs.NEXTAFTER: (_elementwise_binary_check, nextafter),
        "torch.polygamma": (_always_executable, polygamma),
        "torch.pow": (_elementwise_binary_check, pow),
        PrimIDs.POW: (_elementwise_binary_check, pow),
        "torch.remainder": (_elementwise_binary_check, remainder),
        PrimIDs.REMAINDER: (_elementwise_binary_check, remainder),
        "torch.sub": (_add_sub_check, sub),
        PrimIDs.SUB: (_elementwise_binary_check, sub),
        "torch.special.zeta": (_elementwise_binary_check, zeta),
        PrimIDs.ZETA: (_elementwise_binary_check, zeta),
        "torch.addcmul": (_addcmul_check, addcmul),
        "torch.addcdiv": (_addcdiv_check, addcdiv),
        # Conditional and masking operations
        "torch.masked_fill": (_always_executable, masked_fill),
        "torch.tril": (_tril_check, tril),
        PrimIDs.WHERE: (_elementwise_ternary_check, where),
        "torch.where": (_elementwise_ternary_check, where),
        "torch.clamp": (_clamp_check, clamp),
        # Reduction operators
        PrimIDs.AMAX: (_always_executable, amax_prim),
        PrimIDs.AMIN: (_always_executable, amin_prim),
        PrimIDs.PROD: (_always_executable, prod_prim),
        PrimIDs.SUM: (_always_executable, sum_prim),
        "torch.var": (_always_executable, var),
        PrimIDs.VAR: (_always_executable, var_prim),
        PrimIDs.VAR_MEAN: (_always_executable, var_mean_prim),
        # Matmul operations
        PrimIDs.LINEAR: (_always_executable, linear),
        PrimIDs.MATMUL: (_always_executable, matmul),
        # NN operations
        "torch.nn.functional.conv1d": (_always_executable, conv1d),
        "torch.nn.functional.conv2d": (_always_executable, conv2d),
        "torch.nn.functional.conv3d": (_always_executable, conv3d),
        "torch.nn.functional.cross_entropy": (_always_executable, cross_entropy),
        "cross_entropy_backward": (_always_executable, cross_entropy_backward),
        "torch.nn.functional.dropout": (_always_executable, dropout),
        PrimIDs.CONVOLUTION: (_always_executable, convolution),
        "torch.convolution": (_always_executable, convolution),
        PrimIDs.EMBEDDING: (_always_executable, embedding),
        PrimIDs.EMBEDDING_BACKWARD: (_always_executable, embedding_backward),
        "torch.nn.functional.embedding": (_always_executable, embedding),
        "torch.nn.functional.gelu": (_always_executable, gelu),
        "torch.nn.functional.group_norm": (_always_executable, group_norm),
        "torch.nn.functional.interpolate": (_interpolate_check, interpolate),
        "torch.layer_norm": (_always_executable, layer_norm),
        "torch.logsumexp": (_always_executable, logsumexp),
        "torch.log_softmax": (_always_executable, log_softmax),
        "log_softmax_backward": (_always_executable, log_softmax_backward),
        "torch.nn.functional.nll_loss": (_nll_loss_check, nll_loss),
        "nll_loss_backward": (_always_executable, nll_loss_backward),
        "torch.softmax": (_always_executable, softmax),
        "torch.nn.functional.scaled_dot_product_attention": (
            _scaled_dot_product_attention_check,
            scaled_dot_product_attention,
        ),
        "scaled_dot_product_efficient_attention_backward": (
            _scaled_dot_product_efficient_attention_backward_check,
            scaled_dot_product_efficient_attention_backward,
        ),
        "grad_forward_scaled_dot_product_efficient_attention": (
            _grad_forward_scaled_dot_product_attention_check,
            grad_forward_scaled_dot_product_efficient_attention,
        ),
        # Distributed operations
        dist_prims.PrimIDs.ALL_GATHER: (_always_executable, all_gather_prim),
        dist_prims.PrimIDs.ALL_REDUCE: (_always_executable, all_reduce_prim),
        dist_prims.PrimIDs.BROADCAST: (_always_executable, broadcast_prim),
        dist_prims.PrimIDs.REDUCE_SCATTER: (_always_executable, reduce_scatter_prim),
        dist_prims.PrimIDs.WAIT: (_always_executable, wait_prim),
    }
)

#
# Executor interface functions
#


# NOTE This is part of the executor interface
def is_supported(bsym: BoundSymbol, *, prims_only: bool = False) -> bool:
    sym = bsym.sym

    if prims_only and not sym.is_prim:
        return False

    fusible_check, _ = _ops_map.get(sym.id, (None, None))
    if fusible_check is None:
        return False

    is_fusible = fusible_check(*bsym.args, **bsym.kwargs)
    return is_fusible


# TODO This is identical to the same function in the nvFuser executor -- is it
#   interesting to refactor them?
# NOTE This is part of the executor interface
# Returns True if the operation is executable and False if it's not
# An operation is executable if it's directly executable (see is_supported)
#   OR it's a composite operation and all the operations it calls are
#   executable
def can_execute(bsym: BoundSymbol, *, prims_only: bool = False) -> bool:
    if is_supported(bsym, prims_only=prims_only):
        return True

    if len(bsym.subsymbols) == 0:
        return False

    # Checks if all the operations this calls are executable
    can_execute_ = True
    for ssym in bsym.subsymbols:
        if not can_execute(ssym, prims_only=prims_only):
            can_execute_ = False
            break

    return can_execute_


def get_translator(bsym: BoundSymbol) -> Callable:
    return _ops_map[bsym.sym.id][1]


# NOTE This is part of the executor interface
def fuse(region: Region) -> list[BoundSymbol]:
    bsyms: List[BoundSymbol] = []

    for bsym in region.bound_symbols:
        translator = get_translator(bsym)
        tbsym = translator(bsym, *bsym.args, **bsym.kwargs)
        bsyms.append(tbsym)

    return bsyms


#
# Code related to PyTorch's grad transform
#


class ThunderFunction(torch.autograd.Function):
    @staticmethod
    def get_forward_backward_splitter(func, compile_config, compile_data, compile_stats):
        from thunder import trace
        from thunder.executors import transform_for_execution
        from thunder.executors.passes import del_last_used
        from thunder.core.rematerialization import rematerialize_forward_and_backward
        from thunder.core.transforms import forward_and_backward_from_trace
        from thunder.cudagraphs import CUDAGraphExecutor
        from thunder.distributed.utils import sort_waits, sort_data_parallel_syncs

        def make_trace(func):
            return partial(trace(compile_data=compile_data, inline_trace=False, insert_ddp_syncs=True), func)

        def split_forward_backward(*args, **kwargs):
            # NOTE: This function is rather slow, so it's intended to be used
            # behind a cache.
            ba = signature(func).bind(*args, **kwargs)
            ba.apply_defaults()
            args, kwargs = ba.args, ba.kwargs
            flat_args, _ = tree_flatten((args, kwargs))
            tensor_cls = (torch.Tensor, TensorProxy)
            requires_grad_mask = tuple(isinstance(arg, tensor_cls) and arg.requires_grad for arg in flat_args)
            # If none of the inputs require gradients, raise an error
            if not any(requires_grad_mask):
                raise RuntimeError(
                    "PyTorch's Autograd interface requires at least one tensor input with requires_grad=True"
                )

            primal_trace = make_trace(func)(*args, **kwargs)
            primal_trace = sort_data_parallel_syncs(primal_trace)

            # torch.autograd.Function doesn't support non-flat outputs, the
            # grads wouldn't be propagated and backward receives None for each
            # non-flat non-tensor output. The output must also be a flat tuple,
            # not any other container type. So we need to flatten the outputs of
            # the forward trace and inputs of the backward trace.
            fw_trace, bw_trace = forward_and_backward_from_trace(primal_trace, torch_autograd=True)

            # Update the backward trace to only compute gradients for the
            # inputs that require gradients
            assert bw_trace.bound_symbols[-1].sym.id == PrimIDs.RETURN
            filtered_grads = tuple(
                (arg_grad if requires_grad else None)
                for arg_grad, requires_grad in utils.safe_zip(bw_trace.bound_symbols[-1].args[0], requires_grad_mask)
            )

            # autograd.Function.backward expects a flat tuple of gradients
            bw_trace.bound_symbols[-1] = replace(bw_trace.bound_symbols[-1], args=(filtered_grads,))
            bw_trace.output = (filtered_grads,)

            # Now we can run the optimization passes on the forward trace
            fw_extrace, fw_extraces = transform_for_execution(
                fw_trace,
                executors_list=compile_config.get("executors_list", None),
                only_execute_prims=compile_config.get("only_execute_prims", False),
                use_rematerialization=False,
                use_del_last_used=False,
            )

            # Some of the optimization passes change proxies in the trace and
            # any change in the forward trace must be reflected in the backward
            # trace.
            original_bw_saved_tensors_for_backward = bw_trace.args[0][0]
            new_fw_saved_tensors_for_backward = fw_extraces[-1].output[1][0]
            swap_map = {
                variableify(x): y
                for x, y in zip(original_bw_saved_tensors_for_backward, new_fw_saved_tensors_for_backward)
            }
            new_bsyms = replace_redundant_inputs(swap_map, bw_trace.bound_symbols)
            # replace_redundant_inputs doesn't replace the output of
            # UNPACK_SEQUENCE so we do it manually. Here we have certain
            # assumptions about the structure of the backward trace.
            assert bw_trace.bound_symbols[0].sym.id == PrimIDs.UNPACK_TRIVIAL
            assert bw_trace.bound_symbols[0].kwargs["name"] == "saved_for_backward"
            assert bw_trace.bound_symbols[4].sym.id == PrimIDs.UNPACK_SEQUENCE
            assert bw_trace.bound_symbols[4].args[0].name == "C0"
            new_bsyms[4] = new_bsyms[4].from_bsym_swap_proxies(
                swap_map,
                skip_inputs=False,
                skip_output=False,
                skip_subsymbols=False,
            )
            bw_trace.bound_symbols = new_bsyms

            # Now we can run the optimization passes on the backward trace
            bw_extrace, bw_extraces = transform_for_execution(
                bw_trace,
                executors_list=compile_config.get("executors_list", None),
                only_execute_prims=compile_config.get("only_execute_prims", False),
                use_rematerialization=False,
                use_del_last_used=False,
            )

            fw_extrace, bw_extrace = fw_extraces[-1], bw_extraces[-1]
            fw_extrace, bw_extrace = rematerialize_forward_and_backward(fw_extrace, bw_extrace)

            # We need to sort the waits in the backward trace to overlap
            # computation with communication
            bw_extrace = sort_waits(bw_extrace)

            fw_extrace, _ = del_last_used(fw_extrace)
            fw_extraces.append(fw_extrace)

            bw_extrace, _ = del_last_used(bw_extrace)
            bw_extraces.append(bw_extrace)

            if compile_stats is not None:
                compile_stats.primal_trace = primal_trace
                compile_stats.forward_last_traces = fw_extraces
                compile_stats.backward_last_traces = bw_extraces

                if compile_data.use_cudagraphs or compile_config.get("use_cudagraphs", False):
                    fw = CUDAGraphExecutor(
                        fw_extrace.python_callable(), num_constant_args=compile_data.num_constant_args
                    )
                    bw = CUDAGraphExecutor(bw_extrace.python_callable(), num_constant_args=len(bw_extrace.args[0][0]))
                    return fw, bw

            return fw_extrace.python_callable(), bw_extrace.python_callable()

        return split_forward_backward

    @staticmethod
    def forward(ctx, compiled_backward, saved_tensors, saved_other, flat_output, *flat_args):
        # Here we just propagate the tensors through the autograd graph
        ctx.saved_other = saved_other
        ctx.compiled_backward = compiled_backward

        # We must save tensors using ctx.save_for_backward
        ctx.save_for_backward(*saved_tensors)
        return flat_output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *args):
        grads = ctx.compiled_backward((ctx.saved_tensors, ctx.saved_other), args)
        return (None, None, None, None, *grads)


def thunder_backward(*, compile_data=None, compile_stats=None, **compile_config):
    """Decorator to wrap a Thunder function for use with PyTorch autograd.

    Args:
        thunder_func: A Thunder function.

    Returns:
        A wrapped function that can be used with PyTorch autograd.

    Example:
    >>> import torch
    >>> import thunder.clang as clang
    >>> from thunder.executors.torchex import thunder_backward
    >>> @thunder_backward()
    ... def func(a, b):
    ...     c = a + b
    ...     d = c * b
    ...     e = clang.sin(d) + clang.cos(c)
    ...     return e
    >>> a = torch.randn(3, device="cuda", requires_grad=True)
    >>> b = torch.randn(3, device="cuda", requires_grad=True)
    >>> c = func(a, b)
    >>> print(c)
    >>> sum(c).sum().backward()
    >>> print(f"a.grad: {a.grad}")
    >>> print(f"b.grad: {b.grad}")
    """

    compile_config = compile_config | {"disable_preprocessing": True} | {"disable_torch_autograd_support": True}

    def decorator(thunder_func):
        from thunder import compile

        # Compile's caching only works for many calls to the same compiled function
        # It does not work if the same function is compiled many times, so we must
        # decorate the augmented forward pass once with compile once and reuse it
        split_fw_bw = ThunderFunction.get_forward_backward_splitter(
            thunder_func, compile_config, compile_data, compile_stats
        )
        compiled_split_fw_bw = compile(
            split_fw_bw,
            **compile_config,
        )
        sig = signature(thunder_func)

        @wraps(thunder_func)
        def wrapper(*args, **kwargs):
            # Fetch the compiled forward and backward functions using the
            # compiled function cache
            compiled_forward, compiled_backward = compiled_split_fw_bw(*args, **kwargs)

            # Compiled forward function currently doesn't support positional
            # arguments passed as kwargs, so we must bind them here
            ba = sig.bind(*args, **kwargs)
            args, kwargs = ba.args, ba.kwargs

            # Run the compiled forward function
            data_for_autograd, (saved_tensors, saved_other) = compiled_forward(*args, **kwargs)

            # Connect produced tensors with PyTorch's autograd graph
            ThunderFunction.apply(
                compiled_backward,
                saved_tensors,
                saved_other,
                data_for_autograd["flat_output"],
                *data_for_autograd["flat_args"],
            )
            return data_for_autograd["output"]

        return wrapper

    return decorator


if torch.distributed.is_available():
    from torch.distributed.distributed_c10d import ProcessGroup

    def insert_bsym_to_allreduce_grads(
        backward_trace: TraceCtx,
        process_group: Optional[ProcessGroup],
    ) -> TraceCtx:
        """Insert :class:`BoundSymbol`s of pre-averaging, async all_reduce, and wait.

        Args:
            joint_trace: A trace representing backward.
            process_group:
        """
        from torch.distributed.distributed_c10d import _get_default_group
        from thunder.core import prims
        from thunder.core.transforms import visitor_transform, VISIT_TYPE

        # NOTE(crcrpar): To do "pre-averaging" to mitigate grad overflow,
        # we need to know the world size of ddp.
        pg: ProcessGroup = _get_default_group() if process_group is None else process_group
        world_size = float(pg.size())
        gradients, orig_grads_spec = tree_flatten(backward_trace.output)
        grad_to_future = utils.ProxyDict()
        for grad in gradients:
            if not isinstance(grad, TensorProxy):
                continue
            grad_to_future[grad] = True

        class AllReduceGradVisitor:
            def __init__(self):
                self.future_tensor_proxies: list[FutureTensorProxy] = []

            def __call__(self, bsym: BoundSymbol) -> None:
                sym: Symbol = bsym.sym
                if sym.id == PrimIDs.RETURN:
                    prims.python_return(
                        *[
                            dist_prims.wait(grad_to_future[grad]) if isinstance(grad, TensorProxy) else None
                            for grad in gradients
                        ]
                    )
                    return VISIT_TYPE.REPLACE
                grads_of_bsym = tuple(t for t in bsym._flat_outs if isinstance(t, TensorProxy) and t in grad_to_future)
                if len(grads_of_bsym) == 0:
                    # NOTE(crcrpar): Wouldn't `VISIT_TYPE.NOOP` be more lucid?
                    return VISIT_TYPE.INSERT_AFTER
                for grad in grads_of_bsym:
                    preaveraged = ltorch.true_divide(grad, world_size)
                    future = ltorch.all_reduce(preaveraged, group=pg, async_op=True)
                    grad_to_future[grad] = future

                return VISIT_TYPE.INSERT_AFTER

        backward_trace_with_grads_allreduced = visitor_transform(
            trace_from=backward_trace,
            visit=AllReduceGradVisitor(),
            provenance="All-reduce gradients tranform",
        )
        return backward_trace_with_grads_allreduced
