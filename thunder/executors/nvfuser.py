from functools import partial, lru_cache
from numbers import Number
from typing import Union, List, Any, Optional, Dict, Callable, Set, Tuple, Type, Sequence
import collections.abc

import torch
from looseversion import LooseVersion

import thunder.core.dtypes as dtypes

import thunder.torch as ltorch
from thunder.core import prims, utils
from thunder.core.prims import PrimIDs
from thunder.core.proxies import NumberProxy, Proxy, TensorProxy, unvariableify, Variable
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.utils import OrderedSet
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.devices import Device, DeviceType
from thunder.executors.utils import *
import thunder.executors.torch as torchex
import thunder.core.codeutils as codeutils
from thunder.core.codeutils import Printable

# from thunder.core.transforms import register_augmented_forward, register_backward, restore_reduced_dims

import thunder.executors.passes as passes

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

_lcdtype_to_nvdtype_map: Dict[Union[None, dtypes.dtype, Type], DataType] = {
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


def lcdtype_to_nvdtype(lcdtype: Union[dtypes.dtype, Type]) -> DataType:
    return _lcdtype_to_nvdtype_map[lcdtype]


# TODO What kind of constants can nvFuser support?
# TODO Is there a better type annotation for an nvConstant?
# TODO Handle devices!
# Helper to map objects to nvFuser fusion definitions
def _define_constant(fd: FusionDefinition, constant: Any) -> Any:
    if isinstance(constant, Number):
        return fd.define_constant(constant)
    if isinstance(constant, (dtypes.dtype, type)):
        return lcdtype_to_nvdtype(constant)
    if isinstance(constant, Device):
        return None

    utils.check(False, lambda: f"Cannot translate {constant} of type {type(constant)} into an nvFuser constant")


def getnv(x: Any, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    if isinstance(x, Proxy):
        return lc_to_nv_map[x]
    if isinstance(x, (Number, dtypes.dtype, type, Device)):
        return _define_constant(fd, x)

    utils.check(False, lambda: f"Cannot translate {x} of type {type(x)} to an nvFuser object")


# NOTE _ops_map is declared here and defined after the callables have been defined
#   below
_ops_map: Dict[Any, Tuple[Callable, Callable]] = {}


# Note that nvFuser pad() does not perform padding between elements as is done
# in XLA. It instead requires a flat list of static widths (int instead of
# nvNumber), and that argument must come before the fill value.
def _pad_wrapper(fd):
    def _fn(x: nvTensor, value: nvNumber, pad_widths: List[int], dilations: List[int]):
        if all(dil == 0 for dil in dilations):
            return fd.ops.pad(x, pad_widths, value)
        else:
            raise NotImplementedError("Padding between elements is not supported by the nvFuser executor")

    return _fn


# NOTE: nvFuser's pad op requires pad_widths to be a sequence of Python numbers
# (lo_n, hi_n, lo_{n-1}, hi_{n-1}, ...) where dimensions are counted in reverse
# as shown, and dilation is not supported. This translates from
# lightning.compile's pad primitive, which specifies padding and dilation as an
# ndim-length list of (lo, hi, dilation) triples to nvFuser's pad operation. As
# nvFuser currently does not support padding between elements, if a dilation !=
# 0 is encountered then a NotImplementedError will be thrown by _pad_wrapper
# above.
def _pad_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
    x, fill_value, padding_config = nv_args

    def _realize_number(x):
        if isinstance(x, nvNumber):
            for p, nv in variable_to_nvfuser_map.items():
                if nv is x:
                    return p.proxy.value
            raise AssertionError("Failed to find the value of nvNumber when preprocessing pad()!")
        return x

    pad_widths = []
    dilations = []
    # Note that nvfuser also requires pad widths in reverse order so that fewer
    # than ndim pairs may be passed.
    for start, end, dilation in reversed(padding_config):
        pad_widths += [_realize_number(start), _realize_number(end)]
        dilations.append(_realize_number(dilation))

    def _add_constant_number(x):
        if isinstance(x, Number) and not isinstance(x, NumberProxy):
            nv = fd.define_constant(x)
            return nv
        return x

    return (x, _add_constant_number(fill_value), pad_widths, dilations), {}


#
# Functions related to testing if a bound symbol can be fused
#
# TODO Maybe refuse to fuse tensors with no elements?

#
# Shape operations
#


# TODO Check that the tensor dtype is supported by nvFuser -- extract to tensor_supported()?
def _broadcast_in_dim_check(a: TensorProxy, shape: List[int], broadcast_dimensions: List[int]) -> bool:
    return a.device.devicetype is DeviceType.CUDA


def _div_wrapper(fd):
    def _fn(a, b):
        # TODO: nvfuser sometimes generates an accuracy mismatched result
        # when the divisor is a scalar
        # Remove this workaround once the issue is fixed
        # See: https://github.com/NVIDIA/Fuser/issues/160
        if isinstance(b, nvNumber):
            return fd.ops.mul(a, fd.ops.reciprocal(b))
        return fd.ops.div(a, b)


# TODO Carefully consider how shape and broadcast dimensions being constant here relates to
#   the caching of fusions on stride and contiguity information -- do those things being constant
#   imply these values are constant, too?
# TODO Review translating proxy numbers to actual numbers
def broadcast_in_dim(
    a: TensorProxy, shape: List[int], broadcast_dimensions: List[int], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.broadcast_in_dim(nva, shape, broadcast_dimensions)


#
# Elementwise unary operations
#


# TODO Check that the tensor dtype is supported by nvFuser -- extract to tensor_supported()?
def _elementwise_unary_check(a: Union[TensorProxy, Number]) -> bool:
    return isinstance(a, Number) or a.device.devicetype is DeviceType.CUDA


# NOTE nv_abs to avoid a name conflict with the builin abs
def nv_abs(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.abs(nva)


def acos(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = lc_to_nv_map[a]

    return fd.ops.acos(nva)


def acosh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.acosh(nva)


def asin(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.asin(nva)


def asinh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.asinh(nva)


def atan(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.atan(nva)


def atanh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.atanh(nva)


def bitwise_not(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.bitwise_not(nva)


def ceil(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.ceil(nva)


def cos(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.cos(nva)


def cosh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.cosh(nva)


def erf(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erf(nva)


def erfc(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfc(nva)


def erfcinv(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfcinv(nva)


def erfinv(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.erfinv(nva)


def exp(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.exp(nva)


def exp2(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.exp2(nva)


def expm1(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.expm1(nva)


def floor(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.floor(nva)


def isfinite(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.isfinite(nva)


def lgamma(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.lgamma(nva)


def log(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log(nva)


def log10(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log10(nva)


def log1p(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log1p(nva)


def log2(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.log2(nva)


def ndtri(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.ndtri(nva)


def neg(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.neg(nva)


def reciprocal(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.reciprocal(nva)


# NOTE nv_round to avoid a name conflict with the builtin round
def nv_round(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.round(nva)


def rsqrt(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.rsqrt(nva)


def sign(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sign(nva)


def sin(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sin(nva)


def sinh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sinh(nva)


def sqrt(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.sqrt(nva)


def tan(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.tan(nva)


def tanh(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.tanh(nva)


def trunc(a: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)

    return fd.ops.trunc(nva)


#
# Elementwise binary operations
#


# TODO Review support for all elementwise binary operators, like nextafter
def _elementwise_binary_check(a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> bool:
    return (isinstance(a, Number) or a.device.devicetype is DeviceType.CUDA) and (
        isinstance(b, Number) or b.device.devicetype is DeviceType.CUDA
    )


# TODO Generalize to use an elementwise binary helper or factory?
# TODO Convert Python numbers to constants?
def add(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.add(nva, nvb)


def atan2(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.atan2(nva, nvb)


# @register_augmented_forward(nvOps.VAR_MEAN)
def _var_mean_aug_fwd(a, dim, *, correction):
    v, m = var_mean(a, dim, correction=correction)

    return (v, m), (a, dim, correction, m)


# @register_backward(nvOps.VAR_MEAN)
def _var_mean_bwd(a, dim, correction, mean, grad_v, grad_m):
    def n_elem_reduced(a, dims):
        dims = utils.canonicalize_dims(a.ndim, dims)
        reduction_size = 1
        for idx, size in enumerate(a.size()):
            if idx in dims:
                reduction_size *= size
        return reduction_size

    def mean_backward(a, dims, grad):
        mean_local_grad = 1.0 / n_elem_reduced(a, dims)
        return restore_reduced_dims(grad, dims, a.shape) * mean_local_grad

    def var_backward(a, dims, correction, mean, grad):
        # Doing first the multiplication to avoid Python's float division
        var_local_grad = (2.0 * restore_reduced_dims(grad, dims, a.shape)) / (n_elem_reduced(a, dims) - correction)
        return var_local_grad * (a - restore_reduced_dims(mean, dims, a.shape))

    return (
        var_backward(a, dim, correction, mean, grad_v) + mean_backward(a, dim, grad_m),
        None,
    )


def bitwise_and(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.bitwise_and(nva, nvb)


def div(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    # TODO nvFuser sometimes generates an innacurate result when dividing by a number
    #   Remove this workaround once the issue is fixed
    #   See: https://github.com/NVIDIA/Fuser/issues/160
    if isinstance(b, Number):
        return fd.ops.mul(nva, fd.ops.reciprocal(nvb))

    return fd.ops.div(nva, nvb)


def eq(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.eq(nva, nvb)


def fmod(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.fmod(nva, nvb)


def ge(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.ge(nva, nvb)


def lt(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.lt(nva, nvb)


def mul(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.mul(nva, nvb)


def nextafter(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.nextafter(nva, nvb)


def pow(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.pow(nva, nvb)


def remainder(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.remainder(nva, nvb)


def sub(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, fd: FusionDefinition, lc_to_nv_map: Dict
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
    return isinstance(pred, Number) or pred.device.devicetype is DeviceType.CUDA


def where(
    pred: Union[TensorProxy, Number],
    a: Union[TensorProxy, Number],
    b: Union[TensorProxy, Number],
    *,
    fd: FusionDefinition,
    lc_to_nv_map: Dict,
) -> Any:
    nvpred = getnv(pred, fd, lc_to_nv_map)
    nva = getnv(a, fd, lc_to_nv_map)
    nvb = getnv(b, fd, lc_to_nv_map)

    return fd.ops.where(nvpred, nva, nvb)


#
# Tensor creation operations
#


# TODO Check that the dtype is a supported dtype
def _uniform_check(
    shape: Sequence[int], minval: Number, maxval: Number, *, device: Device, dtype: dtypes.dtype
) -> bool:
    return device.devicetype is DeviceType.CUDA


# TODO Add type annotations
# TODO Fix device handling
# TODO Review tuple translation
def uniform(
    shape, minval, maxval, *, device: Device, dtype: dtypes.dtype, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nvdtype = lcdtype_to_nvdtype(dtype)

    nv_minval = getnv(minval, fd, lc_to_nv_map)
    nv_maxval = getnv(maxval, fd, lc_to_nv_map)

    nvshape = list(lc_to_nv_map[x] for x in shape)

    return fd.ops.uniform(nv_minval, nv_maxval, nvshape, dtype=nvdtype)


#
# Data movement operations
#
# TODO Reorder these classes to be consistent


# TODO Fix this check
# TODO Check that the tensor dtype is supported by nvFuser -- extract to tensor_supported()?
def _convert_element_type_check(a: Union[TensorProxy, Number], dtype: Union[Type, dtypes.dtype]) -> bool:
    return True


def convert_element_type(
    a: Union[TensorProxy, Number], dtype: Union[Type, dtypes.dtype], *, fd: FusionDefinition, lc_to_nv_map: Dict
) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdtype = lcdtype_to_nvdtype(dtype)
    return fd.ops.cast(nva, nvdtype)


#
# Reduction operations
#
# TODO Restore direct var_mean translation


# TODO Checks that the dtype is supported by nvFuser
def _reduction_check(a: TensorProxy, dim: Sequence[int], *, output_dtype: Optional[dtypes.dtype] = None) -> bool:
    return a.device.devicetype is DeviceType.CUDA


# TODO Review if this accepts empty dim sequences
def amax(
    a: TensorProxy,
    dims: Sequence[int],
    *,
    output_dtype: Optional[dtypes.dtype] = None,
    fd: FusionDefinition,
    lc_to_nv_map: Dict,
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
    lc_to_nv_map: Dict,
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
    lc_to_nv_map: Dict,
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
    lc_to_nv_map: Dict,
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


def _var_check(a: TensorProxy, dims, *, correction: int) -> bool:
    return a.device.devicetype is DeviceType.CUDA


# TODO Add type annotations
# TODO Review translation of dims and correction
def var(a: TensorProxy, dims, *, correction: int, fd: FusionDefinition, lc_to_nv_map: Dict) -> Any:
    nva = getnv(a, fd, lc_to_nv_map)
    nvdims = list(dims)
    nvcorrection = correction

    return fd.ops.var(nva, nvdims, nvcorrection)


_ops_map.update(
    {
        # Shape Operations
        PrimIDs.BROADCAST_IN_DIM: (_broadcast_in_dim_check, broadcast_in_dim),
        # Data movement operations
        PrimIDs.CONVERT_ELEMENT_TYPE: (_convert_element_type_check, convert_element_type),
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
        PrimIDs.SIN: (_elementwise_unary_check, sin),
        PrimIDs.SINH: (_elementwise_unary_check, sinh),
        PrimIDs.SQRT: (_elementwise_unary_check, sqrt),
        PrimIDs.TAN: (_elementwise_unary_check, tan),
        PrimIDs.TANH: (_elementwise_unary_check, tanh),
        PrimIDs.TRUNC: (_elementwise_unary_check, trunc),
        # Elementwise binary operations
        PrimIDs.ADD: (_elementwise_binary_check, add),
        PrimIDs.ATAN2: (_elementwise_binary_check, atan2),
        PrimIDs.BITWISE_AND: (_elementwise_binary_check, bitwise_and),
        PrimIDs.DIV: (_elementwise_binary_check, div),
        PrimIDs.EQ: (_elementwise_binary_check, eq),
        PrimIDs.FMOD: (_elementwise_binary_check, fmod),
        PrimIDs.GE: (_elementwise_binary_check, ge),
        PrimIDs.LT: (_elementwise_binary_check, lt),
        PrimIDs.MUL: (_elementwise_binary_check, mul),
        PrimIDs.NEXTAFTER: (_elementwise_binary_check, nextafter),
        PrimIDs.POW: (_elementwise_binary_check, pow),
        PrimIDs.REMAINDER: (_elementwise_binary_check, remainder),
        PrimIDs.SUB: (_elementwise_binary_check, sub),
        # Conditional prims
        PrimIDs.WHERE: (_where_check, where),
        # Tensor creation operations
        PrimIDs.UNIFORM: (_uniform_check, uniform),
        # Reduction operations
        PrimIDs.AMAX: (_reduction_check, amax),
        PrimIDs.AMIN: (_reduction_check, amin),
        PrimIDs.PROD: (_reduction_check, prod),
        PrimIDs.SUM: (_reduction_check, sum),
        PrimIDs.VAR: (_var_check, var),
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
    if sym.is_prim:
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
            break

    return can_execute_


def get_translator(bsym: BoundSymbol) -> Callable:
    _, translator = _ops_map[bsym.sym.id]
    return translator


def compute_symbolic_shape(shape: Union[torch.Size, Sequence[int]]) -> Tuple[int, ...]:
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


def compute_contiguity(shape: Union[torch.Size, Sequence[int]], stride: Sequence[int]) -> Tuple[bool, ...]:
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
) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
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


def get_symbolic_shape_and_contiguity(t: torch.Tensor) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
    return compute_symbolic_shape_and_contiguity(t.shape, t.stride())


# NOTE Assumes inputs and outputs are unique and sorted
def create_fd(
    trace, region, input_descriptors: Sequence[Union[type, Tuple[Tuple[int, ...], Tuple[bool, ...]]]]
) -> FusionDefinition:
    lc_to_nv_map = utils.ProxyDict(trace)

    def keyfn(x):
        x = unvariableify(x)
        return utils.get_name(trace, x)

    inputs = list(unvariableify(x) for x in sorted(region.inputs, key=keyfn))
    outputs = list(unvariableify(x) for x in sorted(region.outputs, key=keyfn))

    # constants = list(unvariableify(x) for x in sorted(region.constants, key=keyfn))

    fd = FusionDefinition()
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
                if nv_version >= LooseVersion("0.0.9"):
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
    bsym: BoundSymbol, out_printables: Any, arg_printables: Sequence[Printable], kwarg_printables: Dict[str, Printable]
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
    outputs = codeutils.sequencify(bsym.output)
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


# NOTE This is part of the executor interface
# Translates a region to nvFuser, then creates a Python string invoking the call
def fuse(
    trace: TraceCtx, producers, consumers, bound_symbols: Sequence[BoundSymbol], counter: int
) -> List[BoundSymbol]:
    utils.check(
        len(bound_symbols) > 0,
        lambda: f"Trying to fuse an empty sequence of bound symbols",
        exception_type=AssertionError,
    )

    region = Region(trace, producers, consumers, bound_symbols)

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
        return utils.get_name(trace, x)

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
        return create_fd(trace, region, input_descriptors)

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
    def fn_(*args, use_static_fusion=False):
        """Wrapper for nvFuser fusion.

        Args:
            use_static_fusion (bool, optional):
                Whether to consider input's metadata consistency with the fusion. Defaults to False.

        Returns:
            List[Tensor, ...]: Result of a fusion execution.
        """

        if use_static_fusion:
            return static_fd.execute(args)

        # Constructs key
        input_descriptors = to_descriptors(args)
        fd = get_fd(input_descriptors)

        fn_.last_used = fd

        return fd.execute(args)

    # 4) Creates a BoundSymbol invoking the fusion
    # ctx: Dict[str, Any] = {"nvFuser": fd.execute}
    fn_name = f"nvFusion{counter}"
    ctx: Dict[str, Any] = {fn_name: fn_}
    sym = Symbol(name=fn_name, meta=None, python_printer=nvfusion_printer)

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


#
# OLD CODE BELOW HERE
#


# NOTE: "reshape" used to be called "view"
# use_reshape = hasattr(FusionDefinition.Operators, "reshape")


# _torch_dtype_to_nvfuser_dtype_map = {
#     torch.cdouble: DataType.ComplexDouble,
#     torch.cfloat: DataType.ComplexFloat,
#     torch.double: DataType.Double,
#     torch.float: DataType.Float,
#     torch.half: DataType.Half,
#     torch.bfloat16: DataType.BFloat16,
#     torch.long: DataType.Int,
#     torch.int: DataType.Int32,
#     torch.bool: DataType.Bool,
#     # Python scalars
#     complex: DataType.ComplexDouble,
#     float: DataType.Double,
#     int: DataType.Int,
#     bool: DataType.Bool,
# }

# _thunder_dtype_to_nvfuser_dtype_scalar_map = {
#     complex: DataType.ComplexDouble,
#     float: DataType.Double,
#     int: DataType.Int,
#     bool: DataType.Bool,
# }


# # Wrapper for prims.convert_element_type
# # NOTE: Necessary to ...
# #   1) convert numbertypes to the appropriate datatype,
# #       the conversion depends on whether the input is a scalar or tensor
# #   2) handle constants, which nvFuser will refuse to convert
# def _convert_element_type_translation(fd):
#     def _fn(a, dtype):
#         nvfuser_dtype = dtype

#         if dtypes.is_numbertype(dtype):
#             if isinstance(a, nvTensor):
#                 tensor_dtype = torch.bool

#                 if dtype is int:
#                     tensor_dtype = dtypes.int64
#                 if dtype is float:
#                     tensor_dtype = dtypes.float32
#                 if dtype is complex:
#                     tensor_dtype = dtypes.complex64

#                 nvfuser_dtype = _thunder_dtype_to_nvfuser_dtype_map[tensor_dtype]
#             elif isinstance(a, nvNumber):
#                 # a is a number
#                 number_dtype = bool
#                 if dtype is int:
#                     number_dtype = dtypes.int64
#                 if dtype is float:
#                     number_dtype = dtypes.float64
#                 if dtype is complex:
#                     number_dtype = dtypes.complex128

#                 nvfuser_dtype = _thunder_dtype_to_nvfuser_dtype_map[number_dtype]
#             elif isinstance(a, Number):
#                 return dtype(a)
#             else:
#                 raise ValueError(f"Trying to cast unknown object {a}!")

#         return fd.ops.cast(a, nvfuser_dtype)

#     return _fn


# # A composite implementation of c++ std::remainder and Python math.remainder.
# #
# # It is distinct from nvfuser's internal remainder definition that uses floor
# # rounding mode to match PyTorch, Jax, and Numpy remainder.
# def _remainder_wrapper(fd):
#     def _fn(a, b):
#         c = fd.ops.div(a, b)
#         d = fd.ops.round(c)
#         e = fd.ops.mul(d, b)
#         return fd.ops.sub(a, e)

#     return _fn


# # NOTE: this function is needed because currently nvfuser has a different signature from torch op
# #       more context: https://github.com/csarofeen/pytorch/pull/2449#issuecomment-1427491532
# def _index_select_wrapper(fd):
#     def _fn(a, dim, index):
#         return fd.ops.index_select(a, index, dim)

#     return _fn


# # TODO: consider refactoring the preprocessors with a common pattern to bind or flatten/unflatten?


# # NOTE: nvFuser's reshape takes args (tensor, original_shape, new_shape)
# def _reshape_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
#     # TODO: FIXME
#     assert len(nv_kwargs) == 0

#     nv_t, nv_shape = nv_args
#     t, _ = sym_args

#     original_shape = t.proxy.shape

#     def _realize_numbers(x):
#         if isinstance(x, nvNumber):
#             for p, nv in variable_to_nvfuser_map.items():
#                 if nv is x:
#                     return p.proxy.value
#             raise AssertionError("Failed to find the value of nvNumber when preprocessing broadcast_in_dim()!")
#         return x

#     realized_shape = tuple(_realize_numbers(x) for x in nv_shape)

#     return (nv_t, original_shape, realized_shape), {}


# def _squeeze_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
#     # TODO: FIXME
#     assert len(nv_kwargs) == 0

#     nv_t, nv_dims = nv_args
#     t, _ = sym_args
#     original_shape = t.proxy.shape

#     def _realize_numbers(x):
#         if isinstance(x, nvNumber):
#             for p, nv in variable_to_nvfuser_map.items():
#                 if nv is x:
#                     return p.proxy.value
#             raise AssertionError("Failed to find the value of nvNumber when preprocessing broadcast_in_dim()!")
#         return x

#     realized_dims = tuple(_realize_numbers(x) for x in nv_dims)

#     return (nv_t, original_shape, realized_dims), {}


# # TODO: combine constants
# # NOTE: nvFuser's elementwise operations do not accept Python numbers as arguments, so
# #   this converts Python numbers to nvConstants
# def _elementwise_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
#     # Adds scalars as constants
#     flat_args, arg_structure = tree_flatten(nv_args)
#     flat_kwargs, kwarg_structure = tree_flatten(nv_kwargs)

#     def _add_constant_number(x):
#         if isinstance(x, Number) and not isinstance(x, NumberProxy):
#             nv = fd.define_constant(x)
#             return nv
#         return x

#     flat_args = tuple(_add_constant_number(x) for x in flat_args)
#     flat_kwargs = tuple(_add_constant_number(x) for x in flat_kwargs)

#     return tree_unflatten(flat_args, arg_structure), tree_unflatten(flat_kwargs, kwarg_structure)


# # NOTE: nvFuser's broadcast_in_dim primitive does not accept nvScalars as arguments,
# #   so this converts nvScalars to Python numbers
# # TODO: rewrite this to exploit sym_args and sym_kwargs?
# def _nvScalars_to_Numbers_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
#     # Converts scalars to actual values
#     flat_args, arg_structure = tree_flatten(nv_args)
#     flat_kwargs, kwarg_structure = tree_flatten(nv_kwargs)

#     def _realize_numbers(x):
#         if isinstance(x, nvNumber):
#             for p, nv in variable_to_nvfuser_map.items():
#                 if nv is x:
#                     return p.proxy.value
#             raise AssertionError("Failed to find the value of nvNumber when preprocessing broadcast_in_dim()!")
#         return x

#     flat_args = tuple(_realize_numbers(x) for x in flat_args)
#     flat_kwargs = tuple(_realize_numbers(x) for x in flat_kwargs)

#     return tree_unflatten(flat_args, arg_structure), tree_unflatten(flat_kwargs, kwarg_structure)


# # NOTE: nvFuser's full prim requires shape to be a sequence of Python numbers, the fill value must
# #   be a nvScalar (or nvConstant?), and it accepts no device argument
# # NOTE: the full prim has a bug where it will segfault when shape is an empty sequence
# # TODO: add an assertion on device
# # TODO: revise to use sym_args?
# def _full_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
#     (
#         shape,
#         fill_value,
#     ) = nv_args
#     dtype = nv_kwargs["dtype"]

#     # FIXME: https://github.com/csarofeen/pytorch/issues/2358
#     assert len(shape) > 0

#     def _realize_number(x):
#         if isinstance(x, nvNumber):
#             for p, nv in variable_to_nvfuser_map.items():
#                 if nv is x:
#                     return p.proxy.value
#             raise AssertionError("Failed to find the value of nvNumber when preprocessing broadcast_in_dim()!")
#         return x

#     def _number_to_constant(x):
#         if isinstance(x, Number) and not isinstance(x, NumberProxy):
#             nv = fd.define_constant(x)
#             return nv
#         return x

#     shape = tuple(_realize_number(s) for s in shape)

#     return (shape, _number_to_constant(fill_value), dtype), {}


# if nvfuser_version >= LooseVersion("0.0.6"):
#     # nvFuser has two slice implementation holes:
#     # 1. It does not support strides > 1, this condition raises an error.
#     # 2. Slices beyond the bounds of a tensor do not return a zero-element tensor.
#     #    Silently, an empty tensor of the slice size is returned.  There is an issue filed.
#     #    https://github.com/NVIDIA/Fuser/issues/52
#     ops_to_nvfuser_ops_map[prims.Ops.SLICE] = "slice"
#     ops_to_nvfuser_preprocessors_map[prims.Ops.SLICE] = _nvScalars_to_Numbers_preprocessor

# if nvfuser_version >= LooseVersion("0.0.3"):
#     # Note: uniform is added in nvfuser version 0.0.3
#     # NOTE: this will need to be updated when python refactor is done.
#     def _uniform_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
#         (
#             shape,
#             minval,
#             maxval,
#         ) = nv_args
#         dtype = nv_kwargs["dtype"]
#         # TODO: dropping device here. add assert
#         device = nv_kwargs["device"]
#         # NOTE: nvfuser only accepts cuda device and can't take device index yet
#         assert device == "cuda"

#         def _add_constant_number(x):
#             if isinstance(x, Number) and not isinstance(x, NumberProxy):
#                 nv = fd.define_constant(x)
#                 return nv
#             return x

#         shape = tuple(_add_constant_number(s) for s in shape)
#         minval = _add_constant_number(minval)
#         maxval = _add_constant_number(maxval)

#         return (shape, minval, maxval), {"dtype": dtype}

#     def _uniform_helper(fd):
#         # TODO: should accept device hint.
#         def _fn(shape, minval, maxval, *, dtype):
#             return fd.ops.uniform(minval, maxval, shape, dtype=dtype)

#         return _fn

#     ops_to_nvfuser_ops_map[prims.Ops.UNIFORM] = _uniform_helper
#     ops_to_nvfuser_preprocessors_map[prims.Ops.UNIFORM] = _uniform_preprocessor


# def _var_mean_prim_meta(a, dim, *, correction, **kwargs):
#     output_dtype = a.dtype
#     if utils.is_complex_dtype(output_dtype):
#         output_dtype = utils.corresponding_real_dtype(output_dtype)

#     var = prims.reduction_meta(a, dim, output_dtype=output_dtype)
#     mean = prims.reduction_meta(a, dim, output_dtype=a.dtype)

#     return (var, mean)


# var_mean_prim = prims.make_prim(nvOps.VAR_MEAN, "var_mean", _var_mean_prim_meta)


# def var_mean(a, dim=None, unbiased=None, keepdim=False, *, correction=None):
#     correction = ttorch._set_correction(unbiased, correction)

#     # reduces over all dimensions if dim=() is passed
#     if dim == () or dim == []:
#         dim = None
#     dim = ttorch._reduction_dims(a.shape, dim)

#     # For complex tensors eager computes the variance as the sum of variances of
#     # the real and imaginary parts
#     # TODO: Creating a complex tensor from real and imaginary parts is not supported
#     utils.check(
#         not utils.is_complex_dtype(a.dtype),
#         lambda: "Complex tensors are not supported!",
#     )

#     v, m = var_mean_prim(a, dim, correction=correction)

#     if keepdim:
#         output_shape = [a.shape[i] if i not in dim else 1 for i in range(a.ndim)]
#         broadcast_dims = [i for i in range(a.ndim) if i not in dim]
#         v = prims.broadcast_in_dim(v, output_shape, broadcast_dims)
#         m = prims.broadcast_in_dim(m, output_shape, broadcast_dims)

#     return v, m


# def _get_nvfuser_op(fd, op):
#     nv_op = ops_to_nvfuser_ops_map[op]

#     # TODO: always directly look up the appropriate callable
#     if isinstance(nv_op, str):
#         return getattr(fd.ops, ops_to_nvfuser_ops_map[op])

#     # nv_op is a callable
#     return nv_op(fd)


# def _make_contiguous_strides_for(shape):
#     """Returns the strides of a contiguous tensor if row_major."""
#     if len(shape) == 0:
#         return ()

#     multiplier = 1
#     strides = []
#     for l in reversed(shape):
#         strides.append(multiplier)
#         if l != 0:
#             multiplier *= l

#     result = tuple(reversed(strides))

#     return result


# # Creates an nvFuser input for the corresponding proxy
# def _add_input(fd, variable, variable_to_nvfuser_map):
#     nv = None
#     x = variable.proxy
#     if isinstance(x, NumberProxy):
#         python_type = x.python_type
#         nv_dtype = _thunder_dtype_to_nvfuser_dtype_scalar_map[python_type]
#         nv = fd.define_scalar(nv_dtype)
#     elif isinstance(x, TensorProxy):
#         nv_dtype = _thunder_dtype_to_nvfuser_dtype_map[x.dtype]
#         # TODO: carefully review define tensor args
#         # TODO: fix striding assumption -- currently intermediates produces from
#         #   PyTorch are made contiguous so it's true
#         # NOTE: there is a bug when defining a tensor with ndims!
#         # nv = fd.define_tensor(ndims=len(x.shape), dtype=nv_dtype)
#         strides = x.strides if x.strides is not None else _make_contiguous_strides_for(x.shape)
#         nv = fd.define_tensor(sizes=x.shape, strides=strides, dtype=nv_dtype)
#     else:
#         raise ValueError(f"Trying to add an unknown proxy {x} as an input!")

#     variable_to_nvfuser_map[variable] = nv
#     return nv


# # Finds or creates the nvFuser object associated with x,
# #   possibly updating datastructures for proxies.
# def _get_nv(x, *, fd, variable_to_nvfuser_map):
#     # TODO: revise this
#     #   This is here because nvFuser accepts some numbers, particularly numbers
#     #   in collections, but some operations require a defined nvNumber and not
#     #   a constant number. Because we're treemapping when calling this function,
#     #   it can't disambiguate numbers in collections vs. a number argument.
#     #   So this explicitly doesn't convert numbers to nvNumber, and that
#     #   is left to preprocessing functions.
#     # if isinstance(x, Number) and not isinstance(x, Proxy):
#     #     return x
#     if dtypes.is_dtype(x) and not dtypes.is_numbertype(x):
#         return _thunder_dtype_to_nvfuser_dtype_map[x]

#     if not isinstance(x, Variable):
#         return x

#     if x not in variable_to_nvfuser_map:
#         return _add_input(fd, x, variable_to_nvfuser_map)

#     return variable_to_nvfuser_map[x]


# # Acquires a variable name or passes a constant by value
# def _extract_name(x):
#     if isinstance(x, Variable):
#         return x.name

#     return str(x)


# def _fuse_region(inputs, outputs, symbols):
#     # TODO: ensure this is true in the _fuse call
#     assert len(outputs) > 0
#     assert len(symbols) > 0

#     variable_to_nvfuser_map = {}

#     if nvfuser_version >= LooseVersion("0.0.1"):
#         fd = FusionDefinition()
#         fs = fd
#     else:
#         fs = Fusion()
#         fd = FusionDefinition(fs)

#     with fd:
#         # Adds inputs
#         for inp in inputs:
#             _add_input(fd, inp, variable_to_nvfuser_map)

#         # Adds symbols
#         __get_nv = partial(_get_nv, fd=fd, variable_to_nvfuser_map=variable_to_nvfuser_map)
#         for sym in symbols:
#             nv_args = tree_map(__get_nv, sym.args)
#             nv_kwargs = tree_map(__get_nv, sym.kwargs)
#             nv_pre = ops_to_nvfuser_preprocessors_map.get(sym.op, None)
#             if nv_pre is not None:
#                 # TODO: should preprocessing functions be called with the symbol's args and kwargs
#                 #   or the nv args and kwargs or both?
#                 nv_args, nv_kwargs = nv_pre(fd, variable_to_nvfuser_map, sym.args, sym.kwargs, nv_args, nv_kwargs)
#             nv_op = _get_nvfuser_op(fd, sym.op)
#             nv_result = nv_op(*nv_args, **nv_kwargs)

#             # Associates variables to the nvFuser results
#             # NOTE: it's assumed that NV operations produce results with proxies as leaves
#             variables, _ = tree_flatten(sym.outputs)
#             nvs, _ = tree_flatten(nv_result)
#             for v, nv in zip(variables, nvs):
#                 if v in variable_to_nvfuser_map:
#                     raise AssertionError(f"An output {v} was already in the variable map {variable_to_nvfuser_map}!")
#                 assert isinstance(v, Variable)
#                 variable_to_nvfuser_map[v] = nv

#         # Adds outputs

#         # TODO: refactor this class and the following dict
#         # TODO: probably don't need position in nvOutput
#         class nvOutput:
#             def __init__(self, position, *, is_number=False):
#                 self.position = position
#                 self.is_number = is_number

#         variable_to_nvOutput_map = {}
#         nvfuser_output_ctr = 0
#         for idx, o in enumerate(outputs):
#             # Asserts that all outputs are proxies, that they are unique, and that they
#             #   were produced by the above fusion
#             assert isinstance(o, Variable)
#             assert o in variable_to_nvfuser_map
#             assert o not in variable_to_nvOutput_map
#             # Validates that each output from the fusino appears only once

#             # Ensures that the output is only added as a fusion output once
#             # NOTE: nvFuser doesn't support scalar outputs, so this
#             #   wraps them in tensors (they are unwrapped later)
#             is_number = False
#             if isinstance(o.proxy, NumberProxy):
#                 is_number = True
#                 dtype = _thunder_dtype_to_nvfuser_dtype_scalar_map[o.proxy.python_type]
#                 tensor_out = fd.ops.full((1,), variable_to_nvfuser_map[o], dtype)
#                 fd.add_output(tensor_out)
#             else:
#                 fd.add_output(variable_to_nvfuser_map[o])

#             nvOut = nvOutput(nvfuser_output_ctr, is_number=is_number)
#             variable_to_nvOutput_map[o] = nvOut
#             nvfuser_output_ctr += 1

#     #
#     # Builds the callable
#     #
#     # NOTE: the only reason the callable is built today is to handle unwrapping numbers
#     #   from tensors

#     # Defines utilities
#     tab = "  "

#     # Creates signature
#     arg_str = ", ".join(tuple(_extract_name(inp) for inp in inputs))
#     cstr = f"def fusion({arg_str}):"

#     # Creates call to fusion
#     result_str = ", ".join(tuple(_extract_name(out) for out in outputs))

#     # Handles no inputs
#     if len(inputs) == 0:
#         cstr += f"\n{tab}{result_str}, = _fusion(())"
#     else:
#         cstr += f"\n{tab}{result_str}, = _fusion(({arg_str},))"

#     # Converts tensors to numbers, where appropriate
#     out_strs = []
#     for o in outputs:
#         if isinstance(o.proxy, NumberProxy):
#             out_strs.append(f"{o.name}.cpu().item()")
#         else:
#             out_strs.append(f"{o.name}")
#     out_str = ", ".join(out_strs)
#     cstr += f"\n{tab}return {out_str}"

#     # Creates context
#     ctx = {
#         "_fusion": fs.execute,
#     }

#     code = compile(cstr, "nvfuser.gen", mode="exec")
#     exec(code, ctx)
#     fusion = ctx["fusion"]

#     return fusion


# def lower_for_nvfuser_mapper(symbol: prims.Symbol):
#     """For a given symbol, returns the nvFuser-compatible function that
#     implements the symbol if possible. Otherwise, returns the original
#     function.

#     Args:
#         symbol (prims.Symbol): The symbol to lower.

#     Returns:
#         Callable: The nvFuser-compatible function that implements the symbol
#     """

#     # If the symbol is a core primitive, then we don't need to do anything
#     prim_func = getattr(prims, symbol.name, None)
#     if prim_func is not None:
#         return prim_func

#     # If the symbol is a nvFuser primitive, then we don't need to do anything
#     if symbol.op == nvOps.VAR_MEAN:
#         return var_mean

#     # SLICE primitive doesn't use `symbol.name`
#     if symbol.op == prims.Ops.SLICE:
#         return prims.slice_prim

#     # All other symbols are treated as composite functions
#     # We decompose them into primitives if the decomposition is fully supported
#     # by nvFuser. Otherwise, we keep them as is.
#     decomposed_fn = symbol.decomposed_fn
#     proxy_args = tree_map(lambda x: x.proxy if isinstance(x, Variable) else x, symbol.args)
#     proxy_kwargs = tree_map(lambda x: x.proxy if isinstance(x, Variable) else x, symbol.kwargs)
#     trace = make_trace(lower_for_nvfuser(decomposed_fn))(*proxy_args, **proxy_kwargs)
#     all_supported = all(s.op in ops_to_nvfuser_ops_map for s in trace.symbols)
#     if all_supported:
#         return lower_for_nvfuser(decomposed_fn)

#     # When the decomposition is not supported, we use the original trace recording
#     # function that was used to create the given symbol on the trace.
#     return symbol.non_decomposed_fn


# def lower_for_nvfuser(func):
#     """Converts PyTorch functions to core Thunder primitives if they are supported by nvFuser.

#     Args:
#         func (Callable): A Thunder function to be transformed.
#     """

#     def wrapper(*args, **kwargs):
#         trace = make_trace(func)(*args, **kwargs)
#         return eval_trace(trace, *args, **kwargs, symbol_mapper=lower_for_nvfuser_mapper)

#     return wrapper


# # TODO: support NumPy arrays
# # TODO: possibly support caching on the object that fusion returns
# # fuse returns a function that, when called with actual PyTorch tensors and Python numbers
# #   in place of the corresponding TensorProxies and NumberProxies, computes the given
# #   trace.
# # NOTE: the function can be reused, but it should be called with tensors that have the
# #   same metadata, numbers of the same type, all conditionals on the number evaluated
# #   the same as previous number inputs, and all other values constant.
# def _fuse(
#     trace,
#     *,
#     profile_info=False,
#     mode=None,
#     args=None,
#     kwargs=None,
#     static_inputs=[],
# ):
#     # TODO: improve name canonicalization
#     def _canonicalize_variable_name(x):
#         return x.replace(".", "_").replace("[", "").replace("]", "")

#     # Separates the trace into parts to execute with nvFuser, and parts to execute with PyTorch
#     # TODO: consider where this pass should live in the future
#     # TODO: consider reordering operations cleverly
#     # TODO: there are more elegant ways to express this logic; consider refactoring it

#     #
#     # TODO: maybe generalize is_supported to an executor
#     class Region:
#         def __init__(self, is_supported):
#             self.symbols = []
#             self.is_supported = is_supported
#             self.inputs = []
#             self.outputs = []
#             self.fusion = None

#     regions = []

#     # Variables <-> producers
#     variables_to_producers_map = {}
#     symbols_to_produced_map = {}

#     # Variables <-> consumers
#     variables_to_consumers_map = {}
#     symbols_to_consumed_map = {}

#     symbols_to_region_map = {}

#     cur_region = None

#     # NOTE: this takes advantage of the fact that both symbols and the trace itself stores inputs
#     #   as args and kwargs
#     def _extract_input_variables(sym):
#         flat_args, _ = tree_flatten(sym.args)
#         flat_kwargs, _ = tree_flatten(sym.kwargs)

#         return tuple(x for x in (flat_args + flat_kwargs) if isinstance(x, Variable))

#     def _update_producers(variable, sym):
#         # Updates variable -> producer mapping (one to one)
#         assert variable not in variables_to_producers_map
#         variables_to_producers_map[variable] = sym

#         # Updates symbol -> producer mapping (one to many)
#         if sym in symbols_to_produced_map:
#             symbols_to_produced_map[sym].append(variable)
#         else:
#             symbols_to_produced_map[sym] = [variable]

#     def _update_consumers(variable, sym):
#         # Updates variable -> consumers mapping (one to many)
#         if variable in variables_to_consumers_map:
#             variables_to_consumers_map[variable].append(sym)
#         else:
#             variables_to_consumers_map[variable] = [sym]

#         # Updates symbol -> consumed mapping (one to many)
#         if sym in symbols_to_consumed_map:
#             symbols_to_consumed_map[sym].append(variable)
#         else:
#             symbols_to_consumed_map[sym] = [variable]

#     def _update_region(sym, cur_region):
#         # NOTE: Semantically, is_supported(sym)
#         region = None

#         op_supported = sym.op in ops_to_nvfuser_ops_map
#         if cur_region is None or op_supported != cur_region.is_supported:
#             region = Region(op_supported)
#             regions.append(region)
#         else:
#             region = cur_region

#         region.symbols.append(sym)
#         symbols_to_region_map[sym] = region
#         return region

#     # Retrace for nvFuser if possible
#     proxy_args = tree_map(lambda x: x.proxy if isinstance(x, Variable) else x, trace.args)
#     proxy_kwargs = tree_map(lambda x: x.proxy if isinstance(x, Variable) else x, trace.kwargs)
#     func = partial(eval_trace, trace)
#     # TODO: need to be more careful about preserving names here
#     original_trace = trace
#     trace = make_trace(lower_for_nvfuser(func), executor="nvfuser")(*proxy_args, **proxy_kwargs)

#     # Processes input proxies
#     # TODO: is input its own region?
#     variables = _extract_input_variables(trace)
#     for v in variables:
#         _update_producers(v, "input")

#     # Identifies regions, producers and consumers
#     for sym in trace.symbols:
#         cur_region = _update_region(sym, cur_region)

#         variables = _extract_input_variables(sym)
#         for v in variables:
#             _update_consumers(v, sym)

#         flat_outputs, _ = tree_flatten(sym.outputs)
#         for v in (o for o in flat_outputs if isinstance(o, Variable)):
#             _update_producers(v, sym)

#     # Takes view operations at the tail of nvFuser regions and puts them in a Torch executor region
#     # TODO: this is just a hack for reshape and transpose at the moment, and reshape isn't always a view op!
#     def _is_view_op(sym):
#         return sym.op in (prims.Ops.TRANSPOSE, prims.Ops.RESHAPE)

#     new_regions = []
#     for region_idx, region in enumerate(regions):
#         if region.is_supported:
#             tail_idx = 0
#             for idx, sym in enumerate(reversed(region.symbols)):
#                 if _is_view_op(sym):
#                     tail_idx += 1
#                     continue
#                 break
#             if tail_idx > 0:
#                 tail_view_ops = region.symbols[-tail_idx:]
#                 if len(tail_view_ops) == len(region.symbols):
#                     region.is_supported = False
#                 else:
#                     new_regions.append((1 + region_idx + len(new_regions), tail_view_ops))
#                     region.symbols = region.symbols[:-tail_idx]

#     for region_idx, symbols in new_regions:
#         region = Region(False)
#         region.symbols = symbols

#         # Remap symbols
#         for sym in symbols:
#             symbols_to_region_map[sym] = region

#         regions.insert(region_idx, region)

#     # Merges regions that are next to each other and both unsupported
#     previous_region = None
#     for region in regions:
#         if previous_region is not None:
#             if not previous_region.is_supported and not region.is_supported:
#                 previous_region.symbols.extend(region.symbols)
#                 for sym in region.symbols:
#                     symbols_to_region_map[sym] = previous_region
#                 region.symbols = []
#                 # NOTE: preserves previous_region
#                 continue

#         previous_region = region

#     # Processes outputs
#     # TODO: is output its own region?
#     flat_outputs, output_structure = tree_flatten(trace.outputs)
#     for v in (o for o in flat_outputs if isinstance(o, Variable)):
#         _update_consumers(v, "output")

#     # Identifies inputs and outputs for each region
#     _has_torch_region = False
#     ctx = {}
#     for region in regions:
#         consumed = []
#         produced = []
#         for sym in region.symbols:
#             # NOTE: it's possible that a symbol doesn't consume a proxy
#             if sym in symbols_to_consumed_map:
#                 consumed.extend(symbols_to_consumed_map[sym])
#             if sym in symbols_to_produced_map:
#                 produced.extend(symbols_to_produced_map[sym])
#         consumed = OrderedSet(consumed)
#         produced = OrderedSet(produced)

#         # A proxy that's consumed but not produced in the region is an input
#         # TODO: consider ordering inputs in some sensible way
#         region.inputs = consumed - produced

#         # A proxy that's produced in the region and consumed in another region is an output
#         outputs = []
#         for p in produced:
#             consumers = variables_to_consumers_map.get(p, ())
#             for c in consumers:
#                 if c == "output" or symbols_to_region_map[c] is not region:
#                     region.outputs.append(p)
#                     break

#         # Short-circuits if the region outputs nothing
#         # NOTE: because regions are functional, this means the region does nothing
#         if len(region.outputs) == 0:
#             region.fusion = None
#         elif region.is_supported:
#             region.fusion = _fuse_region(region.inputs, region.outputs, region.symbols)
#         else:
#             # CASE: not region.is_supported (currently using PyTorch to run)
#             _has_torch_region = True
#             region.fusion, ctx_update = _fuse_torch_region(region.inputs, region.outputs, region.symbols)
#             ctx.update(ctx_update)

#     #
#     # Creates the callable connecting the fusions
#     #

#     # Common utils
#     tab = "  "

#     # Creates the signature
#     cstr = ""
#     if _has_torch_region:
#         cstr += "@torch.no_grad()\n"

#     cstr += f"def fusion(*args, **kwargs):"

#     # Acquires inputs
#     flat_positional_inputs, _ = tree_flatten(trace.args)
#     flat_kwarg_inputs, _ = tree_flatten(trace.kwargs)

#     cstr += f"\n{tab}# Extracts inputs"
#     cstr += f"\n{tab}flat_args, _ = tree_flatten(args)"
#     cstr += f"\n{tab}flat_kwargs, _ = tree_flatten(kwargs)"

#     inner_signature_arg_names = []
#     bound_numbers = {}
#     for idx, pinp in enumerate(flat_positional_inputs):
#         if isinstance(pinp, Variable):
#             cstr += f"\n{tab}{pinp.name} = flat_args[{idx}]"
#             if isinstance(pinp.proxy, TensorProxy):
#                 inner_signature_arg_names.append(pinp.name)
#             if isinstance(pinp.proxy, NumberProxy):
#                 if mode == "cudagraphs":
#                     bound_numbers[pinp.name] = pinp.proxy.value
#                 else:
#                     inner_signature_arg_names.append(pinp.name)
#     for idx, kwinp in enumerate(flat_kwarg_inputs):
#         if isinstance(kwinp, Variable):
#             cstr += f"\n{tab}{kwinp.name} = flat_kwargs[{idx}]"
#             if isinstance(kwinp.proxy, TensorProxy):
#                 inner_signature_arg_names.append(kwinp.name)
#             if isinstance(kwinp.proxy, NumberProxy):
#                 if mode == "cudagraphs":
#                     bound_numbers[kwinp.name] = kwinp.proxy.value
#                 else:
#                     inner_signature_arg_names.append(kwinp.name)

#     # Constructs inner fusion
#     if_arg_str = ", ".join(inner_signature_arg_names)
#     ifstr = f"def _inner_fusion({if_arg_str}):"

#     # Binds numbers (when using CUDA graphs)
#     for k, v in bound_numbers.items():
#         ifstr += f"\n{tab}{k} = {v}"

#     # Calls fusion(s)
#     ifstr += f"\n{tab}# Invokes fusion(s)"

#     for idx, region in enumerate(regions):
#         # Skips regions that do nothing
#         if region.fusion is None:
#             continue

#         if isinstance(region.fusion, str):
#             ifstr += region.fusion
#         else:
#             arg_str = ", ".join(tuple(_extract_name(inp) for inp in region.inputs))
#             result_str = ", ".join(tuple(_extract_name(out) for out in region.outputs))
#             ifstr += f"\n{tab}{result_str} = _fusion{idx}({arg_str})"

#     # Returns region outputs which are also outputs of the entire fusion
#     if_outputs = []
#     for out in flat_outputs:
#         for region in regions:
#             if out in region.outputs:
#                 if_outputs.append(out)
#     if_output_str = ", ".join(tuple(_extract_name(out) for out in if_outputs))
#     ifstr += f"\n{tab}return {if_output_str}"

#     if len(if_outputs) > 0:
#         cstr += f"\n{tab}{if_output_str} = _inner_fusion({if_arg_str})"

#     # Constructs return statement
#     output_str = ", ".join(tuple(_extract_name(out) for out in flat_outputs))
#     cstr += f"\n{tab}# Assembles output"
#     cstr += f"\n{tab}return tree_unflatten(({output_str},), output_structure)"

#     if len(if_outputs) > 0:
#         cstr = f"{ifstr}\n{cstr}"

#     if mode == "cudagraphs":
#         flat_args, _ = tree_flatten(args)
#         flat_kwargs, _ = tree_flatten(kwargs)
#         flat_original_positional_inputs, _ = tree_flatten(original_trace.args)
#         flat_original_kwarg_inputs, _ = tree_flatten(original_trace.kwargs)
#         if_args = []
#         for arg, parg in zip(flat_args, flat_original_positional_inputs):
#             if isinstance(parg, Variable) and isinstance(parg.proxy, TensorProxy):
#                 if_args.append((arg, parg))
#         for kwarg, pkwarg in zip(flat_kwargs, flat_original_kwarg_inputs):
#             if isinstance(pkwarg, Variable) and isinstance(pkwarg.proxy, TensorProxy):
#                 if_args.append((kwarg, pkwarg))

#         if_code = compile(ifstr, "nvfuser.gen", mode="exec")
#         if_ctx = ctx

#         for idx, region in enumerate(regions):
#             if_ctx[f"_fusion{idx}"] = region.fusion

#         exec(if_code, if_ctx)
#         if_callable = if_ctx["_inner_fusion"]
#         actual_args = [x[0] for x in if_args]

#         # Warmup
#         # TODO: stream syncs probably not necessary
#         torch.cuda.synchronize()
#         stream = torch.cuda.Stream()
#         stream.wait_stream(torch.cuda.current_stream())
#         with torch.cuda.stream(stream):
#             if_callable(*actual_args)
#         stream.synchronize()
#         torch.cuda.current_stream().wait_stream(stream)
#         torch.cuda.synchronize()

#         # Records the graph
#         static_outputs = None
#         # TODO: improve code for handling params we assume to be static (from modules)
#         static_inputs_raw = set(static_inputs) if static_inputs is not None else set()
#         static_inputs = []
#         static_input_names = set()
#         for arg, proxy in if_args:
#             # if proxy.name in static_input_names:
#             if arg in static_inputs_raw:
#                 static_inputs.append(arg)
#                 static_input_names.add(proxy.name)
#             else:
#                 static_inputs.append(torch.empty_like(arg))
#         graph = torch.cuda.CUDAGraph()
#         with torch.cuda.graph(graph, stream=stream):
#             static_outputs = if_callable(*static_inputs)

#         # TODO: refactor cstr construction to avoid this redundancy
#         cstr = f"def fusion(*args, **kwargs):"
#         cstr += f"\n{tab}# Extracts inputs"
#         cstr += f"\n{tab}flat_args, _ = tree_flatten(args)"
#         cstr += f"\n{tab}flat_kwargs, _ = tree_flatten(kwargs)"

#         # TODO: this should be smarter about extracting variables -- currently that's done to support
#         #   the packing later
#         static_counter = 0
#         for idx, (pinp, poriginal) in enumerate(zip(flat_positional_inputs, flat_original_positional_inputs)):
#             if isinstance(pinp, Variable):
#                 cstr += f"\n{tab}{pinp.name} = flat_args[{idx}]"
#                 if isinstance(pinp.proxy, TensorProxy):
#                     if poriginal.name not in static_input_names:
#                         cstr += f"\n{tab}static_inputs[{static_counter}].copy_(flat_args[{idx}])"
#                     static_counter += 1
#         for idx, (kwinp, kworiginal) in enumerate(zip(flat_kwarg_inputs, flat_original_kwarg_inputs)):
#             if isinstance(kwinp, Variable):
#                 cstr += f"\n{tab}{kwinp.name} = flat_kwargs[{idx}]"
#                 if isinstance(kwinp.proxy, TensorProxy):
#                     if kworiginal.name not in static_input_names:
#                         cstr += f"\n{tab}static_inputs[{static_counter}].copy_(flat_kwargs[{idx}])"
#                     static_counter += 1

#         cstr += f"\n{tab}graph.replay()"

#         cstr += f"\n{tab}# Assembles output"
#         cstr += f"\n{tab}{if_output_str} = static_outputs"
#         cstr += f"\n{tab}return tree_unflatten(({output_str},), output_structure)"

#         ctx = {
#             "tree_flatten": tree_flatten,
#             "tree_unflatten": tree_unflatten,
#             "output_structure": output_structure,
#             "graph": graph,
#             "static_inputs": static_inputs,
#             "static_outputs": static_outputs,
#         }

#         # Compiles the function
#         code = compile(cstr, "nvfuser.gen", mode="exec")
#         exec(code, ctx)
#         fusion = ctx["fusion"]

#         if profile_info:
#             return fusion, regions

#         return fusion

#     # Creates context
#     addtl_ctx = {
#         "tree_flatten": tree_flatten,
#         "tree_unflatten": tree_unflatten,
#         "output_structure": output_structure,
#     }
#     ctx.update(addtl_ctx)

#     for idx, region in enumerate(regions):
#         ctx[f"_fusion{idx}"] = region.fusion

#     # Compiles the function
#     code = compile(cstr, "nvfuser.gen", mode="exec")
#     exec(code, ctx)
#     fusion = ctx["fusion"]

#     if profile_info:
#         return fusion, regions

#     return fusion


# class nvFuserCtx:
#     def __init__(self):
#         pass

#     def intercept(self, op):
#         """"""

#         # TODO: update match to not be on strings
#         if op == "torch.var_mean":
#             return var_mean

#         return None

#     def fuse(
#         self,
#         trace,
#         *,
#         profile_info=False,
#         mode=None,
#         args=None,
#         kwargs=None,
#         static_inputs=None,
#     ):
#         return _fuse(trace, profile_info=profile_info, mode=mode, args=args, kwargs=kwargs, static_inputs=static_inputs)
