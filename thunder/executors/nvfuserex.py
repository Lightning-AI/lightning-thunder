from dataclasses import dataclass
from functools import partial, lru_cache
from numbers import Number
from typing import Union, List, Any, Optional, Dict, Callable, Set, Tuple, Type
from collections.abc import Sequence
import collections.abc

import torch
from looseversion import LooseVersion

import thunder.core.dtypes as dtypes

import thunder.torch as ltorch
from thunder.core import prims, utils
from thunder.core.prims import PrimIDs
from thunder.core.proxies import NumberProxy, Proxy, TensorProxy, unvariableify, Variable, pyval
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.utils import OrderedSet
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.devices import Device, DeviceType
from thunder.executors.utils import *
import thunder.executors.torchex as torchex
import thunder.core.codeutils as codeutils
from thunder.core.codeutils import Printable


def name() -> Executor:
    return Executor.NVFUSER


# Imports nvFuser
# NOTE The nvFuser API changed after PyTorch 1.13

assert (
    nvfuser_available()
), f"Attempting to import nvFuser but it's either not available or is an old version that's unsupported, the minimum supported version is {required_nvfuser_version()}"
nv_version = nvfuser_version()

import nvfuser
from nvfuser import DataType, FusionDefinition
from nvfuser import compute_contiguity as nv_compute_contiguity

nvTensor = nvfuser._C.Tensor
nvNumber = nvfuser._C.Scalar


#
# Helper functions
#

_lcdtype_to_nvdtype_map: dict[Union[None, dtypes.dtype, type], DataType] = {
    dtypes.complex128: DataType.ComplexDouble,
    dtypes.complex64: DataType.ComplexFloat,
    dtypes.float64: DataType.Double,
    dtypes.float32: DataType.Float,
    dtypes.float16: DataType.Half,
    dtypes.bfloat16: DataType.BFloat16,
    dtypes.int64: DataType.Int,
    dtypes.int32: DataType.Int32,
    dtypes.bool8: DataType.Bool,
    dtypes.complex128_: DataType.ComplexDouble,
    dtypes.complex64_: DataType.ComplexFloat,
    dtypes.float64_: DataType.Double,
    dtypes.float32_: DataType.Float,
    dtypes.float16_: DataType.Half,
    dtypes.bfloat16_: DataType.BFloat16,
    dtypes.int64_: DataType.Int,
    dtypes.int32_: DataType.Int32,
    dtypes.bool8_: DataType.Bool,
    # Number types
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
    # Null types
    None: DataType.Null,
}


def lcdtype_to_nvdtype(lcdtype: Union[dtypes.dtype, type]) -> DataType:
    return _lcdtype_to_nvdtype_map[lcdtype]


# TODO What kind of constants can nvFuser support?
# TODO Is there a better type annotation for an nvConstant?
# TODO Handle devices!
# Helper to map objects to nvFuser fusion definitions
def _define_constant(fd: FusionDefinition, constant: Any) -> Any:
    if isinstance(constant, Number):
        val = pyval(constant)
        nvdtype = lcdtype_to_nvdtype(type(val))
        if nv_version >= LooseVersion("0.0.14"):
            return fd.define_scalar(constant, nvdtype)
        else:
            return fd.define_constant(constant, nvdtype)
    if isinstance(constant, (dtypes.dtype, type)):
        return lcdtype_to_nvdtype(constant)
    if isinstance(constant, Device):
        return None

    utils.check(False, lambda: f"Cannot translate {constant} of type {type(constant)} into an nvFuser constant")


def getnv(x: Any, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    if isinstance(x, Proxy):
        return lc_to_nv_map[x]
    if isinstance(x, (Number, dtypes.dtype, type, Device)):
        return _define_constant(fd, x)

    utils.check(False, lambda: f"Cannot translate {x} of type {type(x)} to an nvFuser object")


# NOTE _ops_map is declared here and defined after the callables have been defined
#   below
_ops_map: dict[Any, tuple[Callable, Callable]] = {}


# TODO Check the CUDA arch?
def is_supported_device(device: Device) -> bool:
    utils.check_type(device, Device)
    return device.devicetype is DeviceType.CUDA


def is_supported_devicetype(devicetype: DeviceType) -> bool:
    utils.check_type(devicetype, DeviceType)
    return devicetype is DeviceType.CUDA


_low_precision_floats = (dtypes.float16, dtypes.float16_, dtypes.bfloat16, dtypes.bfloat16_)


def is_supported_dtype(dtype: type | dtypes.dtype, *, allow_low_precision_floats: bool = True) -> bool:
    utils.check_type(dtype, (type, dtypes.dtype))

    if not allow_low_precision_floats:
        if dtype in _low_precision_floats:
            return False

    return dtype in _lcdtype_to_nvdtype_map


def is_supported_tensor(a: TensorProxy, *, allow_low_precision_floats: bool = True) -> bool:
    utils.check_type(a, TensorProxy)
    devicetype_supported = a.device.devicetype is DeviceType.CUDA
    dtype_supported = is_supported_dtype(a.dtype)

    if not allow_low_precision_floats:
        if a.dtype in _low_precision_floats:
            return False

    rank_supported = a.ndim <= 8
    return devicetype_supported and dtype_supported and rank_supported


def is_supported_tensor_or_number(a: Union[TensorProxy, Number]) -> bool:
    if isinstance(a, Number):
        return True

    return is_supported_tensor(a)


# Returns True when all arguments given are supported tensors
#   Throws an error if any arguments are not tensors
# TODO Add a check for the tensor have > 0 elements?
def are_supported_tensors(*args) -> bool:
    for a in args:
        if not is_supported_tensor(a):
            return False

    return True


# Returns True when all arguments given are supported tensors or numbers
#   Throws an error if any arguments are not numbers or tensors
def are_supported_tensors_or_numbers(*args) -> bool:
    for a in args:
        if not is_supported_tensor_or_number(a):
            return False

    return True


#
# Data movement operations
#


def _convert_element_type_check(a: Union[TensorProxy, Number], dtype: Union[type, dtypes.dtype]) -> bool:
    return is_supported_tensor_or_number(a) and is_supported_dtype(dtype)


# TODO Review conversion of numbers vs. tensors
def convert_element_type(
    a: Union[TensorProxy, Number], dtype: Union[type, dtypes.dtype], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdtype = lcdtype_to_nvdtype(dtype)

    return fd.ops.cast(nva, nvdtype)


#
# Tensor creation operations
#


def _full_check(shape: Sequence[int], fill_value: Number, *, device: Device, dtype: dtypes.dtype) -> bool:
    return is_supported_device(device) and is_supported_dtype(dtype)


# TODO Improve device handling
# TODO Materialize shape (if necessary)
# NOTE nvFuser's full prim requires shape to be a sequence of Python numbers
# NOTE nvFuser's full prim requires fill_value be an nvScalar (or nvConstant?)
# NOTE nvFuser's full prim accepts no device argument
def full(
    shape: Sequence[int],
    fill_value: Number,
    *,
    device: Device,
    dtype: dtypes.dtype,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nv_fill_value = getnv(fill_value, fd, lc_to_nv_map)
    nvdtype = lcdtype_to_nvdtype(dtype)

    return fd.ops.full(shape, nv_fill_value, nvdtype)


def _iota_check(length: Number, *, start: Number, step: Number, device: Device, dtype: dtypes.dtype) -> bool:
    return is_supported_device(device) and is_supported_dtype(dtype)


# TODO Improve device handling
def iota(
    length: Number,
    *,
    start: Number,
    step: Number,
    device: Device,
    dtype: dtypes.dtype,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nvlength = getnv(length, fd, lc_to_nv_map)
    nvstart = getnv(start, fd, lc_to_nv_map)
    nvstep = getnv(step, fd, lc_to_nv_map)
    nvdtype = lcdtype_to_nvdtype(dtype)

    return fd.ops.iota(nvlength, nvstart, nvstep, nvdtype)


def _uniform_check(
    shape: Sequence[int], minval: Number, maxval: Number, *, device: Device, dtype: dtypes.dtype
) -> bool:
    if nv_version < LooseVersion("0.0.3"):
        return False

    return is_supported_device(device) and is_supported_dtype(dtype)


# TODO Add type annotations
# TODO Fix device handling
# NOTE Shape must be a list of nvScalars or nvConstants
def uniform(
    shape, minval, maxval, *, device: Device, dtype: dtypes.dtype, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nvdtype = lcdtype_to_nvdtype(dtype)

    nv_minval = getnv(minval, fd, lc_to_nv_map)
    nv_maxval = getnv(maxval, fd, lc_to_nv_map)

    nvshape = list(getnv(x, fd, lc_to_nv_map) for x in shape)

    return fd.ops.uniform(nv_minval, nv_maxval, nvshape, dtype=nvdtype)


def _uniform_philox_check(
    shape: Sequence[int],
    minval: Number,
    maxval: Number,
    *,
    device: Device,
    dtype: dtypes.dtype,
    rng_seed: int,
    rng_offset: int,
) -> bool:
    if nv_version < LooseVersion("0.0.3"):
        return False

    return is_supported_device(device) and is_supported_dtype(dtype)


def uniform_philox(
    shape: Sequence[int],
    minval: Number,
    maxval: Number,
    *,
    device: Device,
    dtype: dtypes.dtype,
    rng_seed: int,
    rng_offset: int,
    fd: FusionDefinition,
    lc_to_nv_map: dict[Any, Any],
) -> Any:
    nvdtype = lcdtype_to_nvdtype(dtype)

    nv_minval = getnv(minval, fd, lc_to_nv_map)
    nv_maxval = getnv(maxval, fd, lc_to_nv_map)

    nvshape = list(getnv(x, fd, lc_to_nv_map) for x in shape)

    nv_rng_seed = getnv(rng_seed, fd, lc_to_nv_map)
    nv_rng_offset = getnv(rng_offset, fd, lc_to_nv_map)

    return fd.ops.uniform(
        nv_minval,
        nv_maxval,
        nvshape,
        dtype=nvdtype,
        rng_seed=nv_rng_seed,
        rng_offset=nv_rng_offset,
    )


#
# Functions related to testing if a bound symbol can be fused
#
# TODO Maybe refuse to fuse tensors with no elements?

#
# Shape operations
#


# TODO Check that the tensor dtype is supported by nvFuser -- extract to tensor_supported()?
def _broadcast_in_dim_check(a: TensorProxy, shape: list[int], broadcast_dimensions: list[int]) -> bool:
    return is_supported_tensor(a)


# TODO Carefully consider how shape and broadcast dimensions being constant here relates to
#   the caching of fusions on stride and contiguity information -- do those things being constant
#   imply these values are constant, too?
# TODO Review translating proxy numbers to actual numbers
def broadcast_in_dim(
    a: TensorProxy, shape: list[int], broadcast_dimensions: list[int], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.broadcast_in_dim(nva, shape, broadcast_dimensions)


def _cat_check(tensors: list[TensorProxy], dim: int) -> bool:
    # nvFuser cat fusion is currently disabled due to
    #   https://github.com/Lightning-AI/lightning-thunder/issues/1071
    return False

    # Validates tensors and concatenated dimension lengths
    for t in tensors:
        if not is_supported_tensor(t):
            return False

        # See https://github.com/NVIDIA/Fuser/issues/21
        #   nvFuser cannot concatenate dimensions of length 1
        if t.shape[dim] == 1:
            return False

    return True


def _stride_order_check(a: TensorProxy, order: Sequence[int]) -> bool:
    if nv_version < LooseVersion("0.0.20"):
        return False

    return is_supported_tensor(a)


def stride_order(a: TensorProxy, order: Sequence[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.stride_order(nva, order)


# NOTEnvFuser's cat prim accepts dim as a Python Number, not a constant
def cat(tensors: list[TensorProxy], dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nvtensors = list(getnv(t, fd, lc_to_nv_map) for t in tensors)

    return fd.ops.cat(nvtensors, dim)


# NOTE nvFuser does not support dilation > 0
def _pad_check(a: TensorProxy, padding_value: Number, padding_config: tuple[int, int, int]) -> bool:
    if a.numel == 0:
        return False

    if nv_version < LooseVersion("0.0.6"):
        return False

    if not is_supported_tensor(a):
        return False

    for lo, hi, dilation in padding_config:
        if dilation > 0:
            return False

    return True


# NOTE Translating to nvFuser's pad operation
#   nvFuser's pad op requires pad_widths to be a sequence of Python numbers
#   (lo_n, hi_n, lo_{n-1}, hi_{n-1}, ...) where dimensions are counted in reverse
#   as shown, and dilation is not supported.
#   This is in constrant to lightning.compile's pad primitive, which specifies padding
#   and dilation as an  ndim-length list of (lo, hi, dilation) triples.
# NOTE padding_value must be an nvConstant (or nvScalar?)
def pad(
    a: TensorProxy,
    padding_value: Number,
    padding_config: tuple[int, int, int],
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nv_padding_value = getnv(padding_value, fd, lc_to_nv_map)

    pad_widths = []

    for lo, hi, dilation in reversed(padding_config):
        pad_widths.extend([lo, hi])

    return fd.ops.pad(nva, pad_widths, nv_padding_value)


def _reshape_check(a: TensorProxy, shape: list[int]) -> bool:
    return is_supported_tensor(a)


def reshape(a: TensorProxy, shape: list[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)

    return fd.ops.reshape(nv_a, a.shape, shape)


# NOTE nvFuser's slice operation only supports all strides == 1
def _slice_check(
    a: TensorProxy, start_indices: Sequence[int], end_indices: Sequence[int], strides: Optional[Sequence[int]] = None
) -> bool:
    if nv_version < LooseVersion("0.0.6"):
        return False

    if not is_supported_tensor(a):
        return False

    # Checks that strides are not specified or all are explicitly set to 1
    if strides is not None:
        for stride in strides:
            if stride != 1:
                return False

    return True


def nv_slice(
    a: TensorProxy,
    start_indices: Sequence[int],
    end_indices: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.slice(nva, start_indices, end_indices, strides)


def _squeeze_check(a: TensorProxy, dims: Sequence[int]) -> bool:
    return is_supported_tensor(a)


# NOTE nvFuser's squeeze operation requires the shape of the tensor be specified
def squeeze(a: TensorProxy, dims: Sequence[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.squeeze(nva, a.shape, dims)


def _take_check(a: TensorProxy, index: TensorProxy, dim: int) -> bool:
    return are_supported_tensors(a, index)


def take(a: TensorProxy, index: TensorProxy, dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)
    nv_index = getnv(index, fd, lc_to_nv_map)

    return fd.ops.index_select(nv_a, nv_index, dim)


# TODO Check that the nvFuser version is >= 0.0.10 when this operator was added
def take_along_axis(a: TensorProxy, index: TensorProxy, dim: int, *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nv_a = getnv(a, fd, lc_to_nv_map)
    nv_index = getnv(index, fd, lc_to_nv_map)

    return fd.ops.take_along_axis(nv_a, nv_index, dim)


def _tranpose_check(a: TensorProxy, permutation: Sequence[int]) -> bool:
    return is_supported_tensor(a)


def transpose(a: TensorProxy, permutation: Sequence[int], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.permute(nva, permutation)


#
# Elementwise unary operations
#


# TODO Check that the tensor dtype is supported by nvFuser -- extract to tensor_supported()?
def _elementwise_unary_check(
    a: Union[TensorProxy, Number], *, version_required: LooseVersion = LooseVersion("0.0.0")
) -> bool:
    return is_supported_tensor_or_number(a) and nv_version > version_required


# NOTE nv_abs to avoid a name conflict with the builin abs
def nv_abs(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.abs(nva)


def acos(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = lc_to_nv_map[a]

    return fd.ops.acos(nva)


def acosh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.acosh(nva)


def asin(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.asin(nva)


def asinh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.asinh(nva)


def atan(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.atan(nva)


def atanh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.atanh(nva)


def bitwise_not(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.bitwise_not(nva)


def ceil(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.ceil(nva)


def cos(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.cos(nva)


def cosh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.cosh(nva)


def erf(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erf(nva)


def erfc(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfc(nva)


def erfcinv(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfcinv(nva)


def erfinv(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfinv(nva)


def exp(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.exp(nva)


def exp2(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.exp2(nva)


def expm1(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.expm1(nva)


def floor(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.floor(nva)


def isfinite(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.isfinite(nva)


def lgamma(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.lgamma(nva)


def log(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log(nva)


def log10(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log10(nva)


def log1p(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log1p(nva)


def log2(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log2(nva)


def ndtri(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.ndtri(nva)


def neg(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.neg(nva)


def reciprocal(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.reciprocal(nva)


# NOTE nv_round to avoid a name conflict with the builtin round
def nv_round(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.round(nva)


def rsqrt(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.rsqrt(nva)


def sign(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sign(nva)


def signbit(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.signbit(nva)


def sin(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sin(nva)


def sinh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sinh(nva)


def sqrt(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sqrt(nva)


def tan(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.tan(nva)


def tanh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.tanh(nva)


def trunc(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.trunc(nva)


def real(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.real(nva)


#
# Elementwise binary operations
#


# TODO Review support for all elementwise binary operators, like nextafter
def _elementwise_binary_check(a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> bool:
    return are_supported_tensors_or_numbers(a, b)


# TODO Generalize to use an elementwise binary helper or factory?
# TODO Convert Python numbers to constants?
def add(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.add(nva, nvb)


def atan2(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.atan2(nva, nvb)


def bitwise_and(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_and(nva, nvb)


def bitwise_or(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_or(nva, nvb)


def bitwise_xor(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_xor(nva, nvb)


# TODO nvFuser's div operation is not equivalent to the div primitive
#   (mruberry) I need to investigate if nvFuser exposes a truncation division operation
def div(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    # TODO nvFuser sometimes generates an innacurate result when dividing by a number
    #   Remove this workaround once the issue is fixed
    #   See: https://github.com/NVIDIA/Fuser/issues/160
    if isinstance(b, Number):
        return fd.ops.mul(nva, fd.ops.reciprocal(nvb))

    # NOTE It's currently significantly faster for nvFuser to multiply the reciprocal than divide
    # return fd.ops.div(nva, nvb)
    return fd.ops.mul(nva, fd.ops.reciprocal(nvb))


def eq(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.eq(nva, nvb)


def fmod(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.fmod(nva, nvb)


def ge(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.ge(nva, nvb)


def gt(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.gt(nva, nvb)


def le(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.le(nva, nvb)


def lt(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.lt(nva, nvb)


def mul(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.mul(nva, nvb)


def ne(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.ne(nva, nvb)


def nextafter(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.nextafter(nva, nvb)


def pow(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.pow(nva, nvb)


def remainder(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.remainder(nva, nvb)


def sub(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.sub(nva, nvb)


#
# Conditional operations
#


# TODO Check supported dtypes
# TODO Properly implement this check
def _where_check(pred, a, b) -> bool:
    return are_supported_tensors_or_numbers(pred, a, b)


def where(
    pred: Union[TensorProxy, Number],
    a: Union[TensorProxy, Number],
    b: Union[TensorProxy, Number],
    *,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nvpred = getnv(pred, fd, lc_to_nv_map)
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.where(nvpred, nva, nvb)


#
# Reduction operations
#


# TODO Checks that the dtype is supported by nvFuser
def _reduction_check(a: TensorProxy, dims: Sequence[int], *, output_dtype: Optional[dtypes.dtype] = None) -> bool:
    dtype_supported = output_dtype is None or is_supported_dtype(output_dtype, allow_low_precision_floats=False)
    return is_supported_tensor(a, allow_low_precision_floats=False) and dtype_supported


# TODO Review if this accepts empty dim sequences
def amax(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    output_dtype: Optional[dtypes.dtype] = None,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = dims
    nvdtype = lcdtype_to_nvdtype(output_dtype)

    return fd.ops.max(nva, nvdims, dtype=nvdtype)


# TODO Review if this accepts empty dim sequences
def amin(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    output_dtype: Optional[dtypes.dtype] = None,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = dims
    nvdtype = lcdtype_to_nvdtype(output_dtype)

    return fd.ops.min(nva, nvdims, dtype=nvdtype)


# TODO Review if this accepts empty dim sequences
def prod(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    output_dtype: Optional[dtypes.dtype] = None,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = dims
    nvdtype = lcdtype_to_nvdtype(output_dtype)

    return fd.ops.prod(nva, nvdims, dtype=nvdtype)


def sum(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    output_dtype: Optional[dtypes.dtype] = None,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = dims
    nvdtype = lcdtype_to_nvdtype(output_dtype)

    # NOTE nvFuser's sum primitive does not accept empty dims sequences
    if len(dims) == 0:
        if nvdtype != DataType.Null:
            return fd.ops.cast(nva, nvdtype)
        return nva

    return fd.ops.sum(nva, nvdims, dtype=nvdtype)


# NOTE https://github.com/NVIDIA/Fuser/pull/121
#   nvFuser's var operation does not support 0-dim inputs
def _var_check(a: TensorProxy, dims: Sequence[int], *, correction: Number) -> bool:
    return is_supported_tensor(a, allow_low_precision_floats=False) and len(a.shape) > 0


# TODO Add type annotations
# TODO Review translation of dims and correction
def var(a: TensorProxy, dims: Sequence[int], *, correction: Number, fd: FusionDefinition, lc_to_nv_map: dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = list(dims)
    nvcorrection = correction

    return fd.ops.var(nva, nvdims, nvcorrection)


# NOTE https://github.com/NVIDIA/Fuser/pull/121
#   nvFuser's var_mean operation does not support 0-dim inputs
# TODO Support complex tensors
#   var(complex) = var(real) + var(imag)
def _var_mean_check(
    a: TensorProxy,
    dim=None,
    *,
    correction: Optional[int] = None,
) -> bool:
    if nv_version < LooseVersion("0.0.7"):
        return False

    if not is_supported_tensor(a, allow_low_precision_floats=False):
        return False

    if len(a.shape) == 0:
        return False

    if dtypes.is_complex_dtype(dtypes.to_dtype(a)):
        return False

    return True


# NOTE nvFuser's var_mean op has the signature (tensor, dims, correction, keepdim)
def var_mean(
    a: TensorProxy,
    dim,
    *,
    correction: int,
    fd: FusionDefinition,
    lc_to_nv_map: dict,
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = list(dim)
    return fd.ops.var_mean(nva, nvdims, correction)


_ops_map.update(
    {
        # Data movement operations
        PrimIDs.CONVERT_ELEMENT_TYPE: (_convert_element_type_check, convert_element_type),
        # Tensor creation operations
        PrimIDs.FULL: (_full_check, full),
        PrimIDs.IOTA: (_iota_check, iota),
        PrimIDs.UNIFORM: (_uniform_check, uniform),
        PrimIDs.UNIFORM_PHILOX: (_uniform_philox_check, uniform_philox),
        # Shape operations
        PrimIDs.BROADCAST_IN_DIM: (_broadcast_in_dim_check, broadcast_in_dim),
        PrimIDs.CAT: (_cat_check, cat),
        PrimIDs.STRIDE_ORDER: (_stride_order_check, stride_order),
        PrimIDs.PAD: (_pad_check, pad),
        PrimIDs.RESHAPE: (_reshape_check, reshape),
        PrimIDs.SLICE: (_slice_check, nv_slice),
        PrimIDs.SQUEEZE: (_squeeze_check, squeeze),
        PrimIDs.TAKE: (_take_check, take),
        # TAKE_ALONG_AXIS is currently disabled
        #   See https://github.com/NVIDIA/Fuser/issues/458
        # PrimIDs.TAKE_ALONG_AXIS: (_take_check, take_along_axis),
        PrimIDs.TRANSPOSE: (_tranpose_check, transpose),
        # Elementwise unary operations
        PrimIDs.ABS: (_elementwise_unary_check, nv_abs),
        PrimIDs.ACOS: (_elementwise_unary_check, acos),
        PrimIDs.ACOSH: (_elementwise_unary_check, acosh),
        PrimIDs.ASIN: (_elementwise_unary_check, asin),
        PrimIDs.ASINH: (_elementwise_unary_check, asinh),
        PrimIDs.ATAN: (_elementwise_unary_check, atan),
        PrimIDs.ATANH: (_elementwise_unary_check, atanh),
        PrimIDs.BITWISE_NOT: (_elementwise_unary_check, bitwise_not),
        PrimIDs.CEIL: (_elementwise_unary_check, ceil),
        PrimIDs.COS: (_elementwise_unary_check, cos),
        PrimIDs.COSH: (_elementwise_unary_check, cosh),
        PrimIDs.ERF: (_elementwise_unary_check, erf),
        PrimIDs.ERFC: (_elementwise_unary_check, erfc),
        PrimIDs.ERFCINV: (_elementwise_unary_check, erfcinv),
        PrimIDs.ERFINV: (_elementwise_unary_check, erfinv),
        PrimIDs.EXP: (_elementwise_unary_check, exp),
        PrimIDs.EXP2: (_elementwise_unary_check, exp2),
        PrimIDs.EXPM1: (_elementwise_unary_check, expm1),
        PrimIDs.FLOOR: (_elementwise_unary_check, floor),
        PrimIDs.ISFINITE: (_elementwise_unary_check, isfinite),
        PrimIDs.LGAMMA: (_elementwise_unary_check, lgamma),
        PrimIDs.LOG: (_elementwise_unary_check, log),
        PrimIDs.LOG10: (_elementwise_unary_check, log10),
        PrimIDs.LOG1P: (_elementwise_unary_check, log1p),
        PrimIDs.LOG2: (_elementwise_unary_check, log2),
        # NOTE nvFuser does not have an ndtri operation
        # PrimIDs.NDTRI: (_elementwise_unary_check, ndtri),
        PrimIDs.NEG: (_elementwise_unary_check, neg),
        PrimIDs.RECIPROCAL: (_elementwise_unary_check, reciprocal),
        PrimIDs.ROUND: (_elementwise_unary_check, nv_round),
        PrimIDs.RSQRT: (_elementwise_unary_check, rsqrt),
        PrimIDs.SIGN: (_elementwise_unary_check, sign),
        PrimIDs.SIGNBIT: (partial(_elementwise_unary_check, version_required=LooseVersion("0.0.11")), signbit),
        PrimIDs.SIN: (_elementwise_unary_check, sin),
        PrimIDs.SINH: (_elementwise_unary_check, sinh),
        PrimIDs.SQRT: (_elementwise_unary_check, sqrt),
        PrimIDs.TAN: (_elementwise_unary_check, tan),
        PrimIDs.TANH: (_elementwise_unary_check, tanh),
        PrimIDs.TRUNC: (_elementwise_unary_check, trunc),
        PrimIDs.REAL: (_elementwise_unary_check, real),
        # Elementwise binary operations
        PrimIDs.ADD: (_elementwise_binary_check, add),
        PrimIDs.ATAN2: (_elementwise_binary_check, atan2),
        PrimIDs.BITWISE_AND: (_elementwise_binary_check, bitwise_and),
        PrimIDs.BITWISE_OR: (_elementwise_binary_check, bitwise_or),
        PrimIDs.BITWISE_XOR: (_elementwise_binary_check, bitwise_xor),
        PrimIDs.DIV: (_elementwise_binary_check, div),
        PrimIDs.EQ: (_elementwise_binary_check, eq),
        PrimIDs.FMOD: (_elementwise_binary_check, fmod),
        PrimIDs.GE: (_elementwise_binary_check, ge),
        PrimIDs.GT: (_elementwise_binary_check, gt),
        PrimIDs.LE: (_elementwise_binary_check, le),
        PrimIDs.LT: (_elementwise_binary_check, lt),
        PrimIDs.MUL: (_elementwise_binary_check, mul),
        PrimIDs.NE: (_elementwise_binary_check, ne),
        PrimIDs.NEXTAFTER: (_elementwise_binary_check, nextafter),
        PrimIDs.POW: (_elementwise_binary_check, pow),
        PrimIDs.REMAINDER: (_elementwise_binary_check, remainder),
        PrimIDs.SUB: (_elementwise_binary_check, sub),
        # Conditional prims
        PrimIDs.WHERE: (_where_check, where),
        # Reduction operations
        PrimIDs.AMAX: (_reduction_check, amax),
        PrimIDs.AMIN: (_reduction_check, amin),
        PrimIDs.PROD: (_reduction_check, prod),
        PrimIDs.SUM: (_reduction_check, sum),
        PrimIDs.VAR: (_var_check, var),
        PrimIDs.VAR_MEAN: (_var_mean_check, var_mean),
    }
)

#
# Executor interface functions
#


# TODO Remove the limitation that nvFuser only supports primitives
# TODO Document the executor interface (and maybe enshrine it in code)
# NOTE This is part of the executor interface
# Returns whether the particular bound symbol can be directly executed by
#   the executor (as opposed to can_execute, below)
def is_supported(bsym: BoundSymbol, *, prims_only: bool = False) -> bool:
    sym = bsym.sym

    if prims_only and not sym.is_prim:
        return False

    check, _ = _ops_map.get(sym.id, (None, None))
    if check is None:
        return False
    return check(*bsym.args, **bsym.kwargs)


# TODO Make this non-recursive?
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
    can_execute_: bool
    for ssym in bsym.subsymbols:
        can_execute_ = can_execute(ssym, prims_only=prims_only)
        if not can_execute_:
            return False

    return can_execute_


def get_translator(bsym: BoundSymbol) -> Callable:
    _, translator = _ops_map[bsym.sym.id]
    return translator


def compute_symbolic_shape(shape: Union[torch.Size, Sequence[int]]) -> tuple[int, ...]:
    """
    Computes the symbolic shape of a tensor using nvFuser's notion of a symbolic
    shape, it's represented by 1s and -1s. 1s represent dimensions that are
    known to be 1, and -1s represent dimensions that are not known to be 1.

    For example, the symbolic shape of a tensor with shape (1, 2, 3) is (1, -1, -1).

    Args:
        shape (Union[torch.Size, Sequence[int]]): The shape of the tensor.

    Returns:
        Tuple[int, ...]: The symbolic shape of the tensor.
    """
    return tuple(1 if l == 1 else -1 for l in shape)


def compute_contiguity(shape: Union[torch.Size, Sequence[int]], stride: Sequence[int]) -> tuple[bool, ...]:
    """
    Computes the contiguity of a tensor using nvFuser's notion of contiguity, it's
    represented by True, False and None. True represents dimensions that are contiguous,
    and False represents dimensions that are not contiguous, and None represents
    stride-0 or size-1 dimensions.

    For example, the contiguity of a tensor with shape (1, 2, 3) and stride (6, 3, 1)
    is (None, True, True).

    Args:
        shape (Union[torch.Size, Sequence[int]]): The shape of the tensor.
        stride (Sequence[int]): The stride of the tensor.

    Returns:
        Tuple[bool, ...]: The contiguity of the tensor.
    """
    return tuple(nv_compute_contiguity(shape, stride))


@lru_cache(maxsize=2048)
def compute_symbolic_shape_and_contiguity(
    shape: Union[torch.Size, Sequence[int]], stride: Sequence[int]
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """
    Computes the symbolic shape and contiguity of a tensor using nvFuser's notion of
    symbolic shape and contiguity. See compute_symbolic_shape and compute_contiguity
    for more details.

    This function is caching the results of compute_symbolic_shape and compute_contiguity
    to speed up the computation.

    Args:
        shape (Union[torch.Size, Sequence[int]]): The shape of the tensor.
        stride (Sequence[int]): The stride of the tensor.

    Returns:
        Tuple[Tuple[int, ...], Tuple[bool, ...]]: The symbolic shape and contiguity of the tensor.
    """
    return compute_symbolic_shape(shape), compute_contiguity(shape, stride)


def get_symbolic_shape_and_contiguity(t: torch.Tensor) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    return compute_symbolic_shape_and_contiguity(t.shape, t.stride())


# NOTE Assumes inputs and outputs are unique and sorted
def create_fd(
    trace, region, input_descriptors: Sequence[Union[type, tuple[tuple[int, ...], tuple[bool, ...]]]]
) -> FusionDefinition:
    lc_to_nv_map = utils.ProxyDict()

    def keyfn(x):
        x = unvariableify(x)
        return utils.get_name(trace, x)

    inputs = list(unvariableify(x) for x in sorted(region.inputs, key=keyfn))
    outputs = list(unvariableify(x) for x in sorted(region.outputs, key=keyfn))

    # constants = list(unvariableify(x) for x in sorted(region.constants, key=keyfn))

    # NOTE nvFuser's default max length is 1024 operations at the time of this writing
    #   This arbitrarily increases it to 9999
    # TODO Review splititng very large fusions or removing the max length restriction completely
    #   See https://github.com/Lightning-AI/lightning-thunder/issues/901
    fd = FusionDefinition(max_length=9999)
    with fd:
        # 0) Adds constants

        # for c in constants:
        #     nv = _define_constant(fd, c)
        #     lc_to_nv_map[c] = nv

        # 1) Inputs are added and mapped to nvFuser objects

        # NOTE x is the trace's annotation of the input, y is the actual concrete input descriptor at call time
        def add_input(x: Any, y: Any) -> Any:
            nv: Any
            if isinstance(x, NumberProxy):
                utils.check_type(y, type)
                python_type = y
                nvdtype = lcdtype_to_nvdtype(python_type)
                nv = fd.define_scalar(nvdtype)
            elif isinstance(x, TensorProxy):
                utils.check_type(y, tuple)
                symbolic_shape, contiguity, dtype = y
                nvdtype = lcdtype_to_nvdtype(ltorch.to_thunder_dtype(dtype))
                if nv_version >= LooseVersion("0.0.17"):
                    nv = fd.define_tensor(shape=symbolic_shape, contiguity=contiguity, dtype=nvdtype)
                elif nv_version >= LooseVersion("0.0.9"):
                    nv = fd.define_tensor(symbolic_sizes=symbolic_shape, contiguity=contiguity, dtype=nvdtype)
                else:
                    nv = fd.define_tensor(symbolic_sizes=symbolic_shape, contiguous=contiguity, dtype=nvdtype)
            elif isinstance(x, Proxy):
                utils.check(False, lambda: f"Unsupported proxy type {x} in fusion", exception_type=AssertionError)
            else:
                nv = x

            lc_to_nv_map[x] = nv
            return nv

        for pinp, inp in zip(inputs, input_descriptors):
            add_input(pinp, inp)

        # 2) Translates bound symbols

        def translate_bound_symbol(bsym: BoundSymbol) -> Any:
            translator = get_translator(bsym)
            nvresults = translator(*bsym.args, **bsym.kwargs, fd=fd, lc_to_nv_map=lc_to_nv_map)

            # Updates map
            for out, nvout in zip(utils.sequencify(bsym.output), utils.sequencify(nvresults)):
                # NOTE out can be None if an operation returned multiple results but only some are used,
                #   in which case DCE will replace the unused results with None
                if out is not None:
                    lc_to_nv_map[out] = nvout

        for bsym in region.bound_symbols:
            translate_bound_symbol(bsym)

        # 3) Adds outputs
        # TODO Translate numbers to tensors (and provide the information to translate them back to numbers!)
        for out in outputs:
            nvout = lc_to_nv_map[out]
            fd.add_output(nvout)

    return fd


# TODO Review using out_printables or adding a check that everything in bsym.output is printable
# TODO Maybe instead of using a custom printer to always unpack the sequence we should model
#   the unpack explicitly
# NOTE Custom nvFuser fusion printer which automatically unpacks the list returned by nvFuser,
#   even when there is only one output
def nvfusion_printer(
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: dict[str, Printable]
) -> str:
    utils.check(
        len(kwarg_printables) == 0,
        lambda: f"Expected no kwargs for nvFusion but got {kwarg_printables}",
        exception_type=AssertionError,
    )

    arg_str = (
        ""
        if (arg_printables is None or len(arg_printables) == 0)
        else ", ".join(codeutils.prettyprint(x) for x in arg_printables)
    )

    # NOTE nvFuser fusions always return a sequence
    result_str = f"{codeutils.prettyprint(bsym.output)} = "

    s = f"{result_str}{bsym.name_with_module()}({arg_str})"
    return s


# NOTE Currently assumes that only numbers and tensors are passed in
# TODO Add check that only numbers and tensors are passed in
# TODO Inline the get_symbolic_shape_and_contiguity call
def to_descriptors(args):
    return tuple(
        type(arg) if isinstance(arg, Number) else (*get_symbolic_shape_and_contiguity(arg), arg.dtype) for arg in args
    )


@dataclass
class FusionDefinitionWrapper:
    """
    A callable object wrapping a nvFuser fusion definition.
    """

    counter: int
    get_fd: Callable[[tuple[Union[type, tuple[tuple[int, ...], tuple[bool, ...]]], ...]], FusionDefinition]
    cache_info: Optional[Callable] = None
    cache_clear: Optional[Callable] = None
    last_used: Optional[FusionDefinition] = None

    def __call__(self, *args):
        fd = self.get_fd(to_descriptors(args))
        self.last_used = fd
        return fd.execute(args)

    def __repr__(self):
        return f"FusionDefinitionWrapper{self.counter}"


# NOTE This is part of the executor interface
# Translates a region to nvFuser, then creates a Python string invoking the call
def fuse(region: Region) -> list[BoundSymbol]:
    utils.check(
        len(region.bound_symbols) > 0,
        lambda: f"Trying to fuse an empty sequence of bound symbols",
        exception_type=AssertionError,
    )

    # NOTE Region Inputs and Outputs
    # The inputs and outputs to a region are represented as sets, which are sorted by name
    #   for determinism. Because they're sets, the inputs and outputs to each region are
    #   unique.
    # It's OK to reorder inputs to regions and outputs from regions, become the dataflow of those
    #   objects is captured by names in the trace.
    # These properties are distinct from the inputs and outputs to the trace itself, which
    #   may contain duplicates and whose order must be preserved.

    def keyfn(x):
        x = unvariableify(x)
        return utils.get_name(region.trace, x)

    # keyfn = partial(utils.get_name, trace)
    sorted_inputs = list(unvariableify(x) for x in sorted(region.inputs, key=keyfn))
    sorted_outputs = list(unvariableify(x) for x in sorted(region.outputs, key=keyfn))
    # sorted_constants = list(unvariableify(x) for x in sorted(region.constants, key=keyfn))

    # Constructs the fusion definition
    # NOTE Fusion definition construction has to both recreate the trace and map
    #   traced objects to nvFuser objects.
    # The construction has the following steps:
    #   1) Inputs are added and mapped to nvFuser objects
    #   2) The bound symbols are translated, and their inputs and outputs mapped to nvFuser objects
    #   3) The outputs are added and mapped to nvFuser objects
    #       3a) The outputs are processed so that numbers are returned as tensors
    #   4) A BoundSymbol invoking the fusion is created
    #       4a) Its subsymbols are the operations in this region, so that when printed it has a
    #               a comment describing the operations it performs
    #   5) Additional BoundSymbols are added to create a post-fusion epilogue that translates numbers
    #       returned as tensors back to numbers
    #       5a) NOTE The translated numbers must be given the correct name
    #   6) The list of BoundSymbols is returned
    #

    tensor_indices = []
    for idx, x in enumerate(sorted_inputs):
        if isinstance(x, TensorProxy):
            tensor_indices.append(idx)

    # create_fd is an expensive function so we cache using the descriptors of
    # inputs
    @lru_cache(maxsize=2048)
    def get_fd(input_descriptors) -> FusionDefinition:
        # A closure over local trace and region
        return create_fd(region.trace, region, input_descriptors)

    # TODO Re-enable a static fusion option
    # NOTE This fusion definition uses only region's proxy inputs to construct the fusion
    # Inputs are assumed to be contiguous
    # static_fd = get_fd(to_descriptors(sorted_inputs))

    #
    # NOTE Currently this code is a mess because it splices caching code with debug statements
    #   in between comments relating to the previous code path
    #
    # TODO Clean this up

    # TODO Do we need to cache on symbolic shapes? This might always be inferrable
    #   from the trace without loss of generality -- the contiguity is the real concern (at least today)
    # TODO Consider how to best generate this wrapper for maintenance and speed
    #   (mruberry) I'm seeing about 3.5us of extra time per tensor input, vs. just
    #   calling the underlying fusion directly
    # TODO We should think how to express "static fusion" mode and if we want to
    # throw an error if the static constraint is violated or just require users
    # promise the input is unchanging
    fn_ = FusionDefinitionWrapper(region.counter, get_fd, get_fd.cache_info, get_fd.cache_clear)

    # 4) Creates a BoundSymbol invoking the fusion

    fn_name = f"nvFusion{region.counter}"
    ctx: dict[str, Any] = {fn_name: fn_}
    sym = Symbol(name=fn_name, meta=None, python_printer=nvfusion_printer, is_fusion=True)

    # Adds a comment explaining what the fusion (conceptually) does
    bsym = BoundSymbol(
        sym,
        args=tuple(sorted_inputs),
        kwargs={},
        output=tuple(sorted_outputs),
        subsymbols=tuple(region.bound_symbols),
        _call_ctx=ctx,
    )

    # 5) Adds post-fusion epilogue
    # TODO Add tensor -> number mapping

    # 6) Returns the list of bound symbols
    return [bsym]
