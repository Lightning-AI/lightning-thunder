from numbers import Number
from typing import Any, Callable, Union, Optional, Tuple, Type
from collections.abc import Sequence
from functools import partial, reduce
import math
import operator
from enum import Enum, auto

from thunder.core.proxies import TensorProxy
import thunder.core.prims as prims
import thunder.core.utils as utils
import thunder.core.dtypes as dtypes
from thunder.core.langctx import langctx
from thunder.core.symbol import Symbol
from thunder.core.pytree import tree_map
import thunder.clang as clang
import thunder.core.devices as devices

__all__ = [
    "is_available",
]

# NOTE torch is a requirement
import torch

# Type annotation helpers
TensorLike = TensorProxy
DeviceLike = Union[str, devices.Device, torch.device]
dtypeLike = Union[dtypes.dtype, torch.dtype]


#
# default langctx interface: available devices
#
def available_devices() -> Sequence[devices.Device]:
    available_devices = [devices.cpu]

    # Short-circuits if there are no CUDA devices
    if not torch.cuda.is_available:
        return available_devices

    # NOTE torch.cuda.is_available, extends with CUDA devices
    cuda_devices = tuple(devices.Device(devices.DeviceType.CUDA, idx) for idx in range(torch.cuda.device_count()))
    available_devices.extend(cuda_devices)

    return available_devices


#
# langctx interface: is_available, to_dtype, tensor_cls, tensorproxy, lookup_method
#


# Trivial implementation of the langctx interface since torch is a requiremented
def is_available():
    return True


# TODO language contexts like Torch could be
#   expanded to allow datatypes that the original language didn't support
_thunder_to_torch_dtype_map = {
    bool: torch.bool,
    int: torch.int32,
    float: torch.float32,
    complex: torch.complex64,
    dtypes.bool8_: torch.bool,
    dtypes.bool8: torch.bool,
    dtypes.uint8_: torch.uint8,
    dtypes.uint8: torch.uint8,
    dtypes.int8_: torch.int8,
    dtypes.int8: torch.int8,
    dtypes.int16_: torch.int16,
    dtypes.int16: torch.int16,
    dtypes.int32_: torch.int32,
    dtypes.int32: torch.int32,
    dtypes.int64_: torch.int64,
    dtypes.int64: torch.int64,
    dtypes.bfloat16_: torch.bfloat16,
    dtypes.bfloat16: torch.bfloat16,
    dtypes.float16_: torch.float16,
    dtypes.float16: torch.float16,
    dtypes.float32_: torch.float32,
    dtypes.float32: torch.float32,
    dtypes.float64_: torch.float64,
    dtypes.float64: torch.float64,
    dtypes.complex32_: torch.complex32,
    dtypes.complex32: torch.complex32,
    dtypes.complex64_: torch.complex64,
    dtypes.complex64: torch.complex64,
    dtypes.complex128_: torch.complex128,
    dtypes.complex128: torch.complex128,
}

_torch_to_thunder_dtype_map = {
    v: k
    for k, v in _thunder_to_torch_dtype_map.items()
    if not utils.is_weak_dtype(k) and not (type(k) is type and issubclass(k, Number))
}

# NOTE This is defined here and populated as functions are defined below
# It maps torch functions, like torch.foo, to their corresponding functions here
_torch_to_thunder_function_map = {}


def to_dtype(x: Any) -> dtypes.dtype:
    if isinstance(x, torch.Tensor):
        return to_thunder_dtype(x.dtype)
    if isinstance(x, torch.dtype):
        return to_thunder_dtype(x)

    return dtypes.to_dtype(x)


def to_thunder_dtype(dtype: Optional[Union[torch.dtype, dtypes.dtype]]) -> Optional[dtypes.dtype]:
    if isinstance(dtype, dtypes.dtype) or dtype is None:
        return dtype
    return _torch_to_thunder_dtype_map[dtype]


def to_torch_dtype(dtype: Optional[Union[torch.dtype, dtypes.dtype]]) -> Optional[torch.dtype]:
    if isinstance(dtype, torch.dtype) or dtype is None:
        return dtype
    return _thunder_to_torch_dtype_map[dtype]


tensor_cls = torch.Tensor


def tensorproxy(name: Optional[str], t: torch.Tensor) -> TensorProxy:
    device = devices.device_from_string(str(t.device))
    dtype = to_thunder_dtype(t.dtype)

    # NOTE Without tuple(t.shape) then the shape would be a torch.Size object
    return TensorProxy(name, shape=tuple(t.shape), device=device, dtype=dtype)


# Convers from a torch device, or a string representing such a device, to a Thunder device
def to_thunder_device(device: Optional[Union[str, torch.device, devices.Device]]) -> Optional[devices.Device]:
    if isinstance(device, devices.Device) or device is None:
        return device

    devicestr = device if isinstance(device, str) else str(device)
    return devices.device_from_string(devicestr)


def to_torch_device(device: Optional[Union[str, torch.device, devices.Device]]) -> Optional[torch.device]:
    if isinstance(device, torch.device) or device is None:
        return device

    return str(device)


# Helpers for defining torch methods
_torch_methods = {}


def method_lookup(name: str) -> Optional[Symbol]:
    return _torch_methods.get(name, None)


#
# torch operation definitions
#


# A wrapper that executes the operations within the torch language context
# NOTE because this module defines the torch language context, a reference to itself
#   is aquired by inspecting the __module__ attribute of the is_available function defined
#   above
# NOTE Functions that set is_method=True must be able to accept a tensor as their first positional input
class torchsymbol:
    def __init__(self, *torchfns, is_method: bool = False, id: Optional[str] = None):
        self.torchfns = torchfns
        self.is_method = is_method
        self.id = id

    def __call__(self, fn: Callable) -> Symbol:
        module_name = torchsymbol.__module__
        module = utils.get_module(module_name)
        _fn = langctx(module)(fn)

        id: str
        if self.id is None:
            name = fn.__name__
            if hasattr(torch, name):
                id = f"torch.{name}"
            elif hasattr(torch.nn.functional, name):
                id = f"torch.nn.functional.{name}"
            elif hasattr(torch.Tensor, name):
                id = f"torch.Tensor.{name}"
            elif hasattr(torch.ops.aten, name):
                id = f"torch.ops.aten.{name}"
            else:
                utils.check(
                    False,
                    lambda: f"The torchsymbol decorator failed to infer an id for {name}, specify one explicitly (with id=<your id>)",
                    exception_type=AssertionError,
                )
        else:
            id = self.id

        sym = Symbol(name=fn.__name__, meta=_fn, id=id)

        if self.is_method:
            _torch_methods[sym.name] = sym
            torch_method = getattr(torch.Tensor, fn.__name__)
            _torch_to_thunder_function_map[torch_method] = sym

        if self.torchfns is not None:
            for torchfn in self.torchfns:
                _torch_to_thunder_function_map[torchfn] = sym

        return sym


#
# Tensor properties
#


def size(a):
    def fn_(idx: Optional[int] = None):
        if idx is None:
            return a.shape
        return a.shape[idx]

    return fn_


#
# Data movement and transformation operations
#


# NOTE This handles a.float()
#   It avoids using the name "float" to not collide with the builtin
#   "float"
def to_float(a):
    return clang.maybe_convert_to_dtype(a, dtypes.float32)


# NOTE to's parsing is a little whacky
#   to supports five first positional arguments
#   1) a tensor, in which case device and dtype cannot be specified (although we allow them to be)
#   2) a dtype, in which case device cannot be specified (although we allow this)
#   3) a device, in which case dtype can be specified,
#   4) None, in which case device and dtype come from kwargs (which may also be None, a.to() is valid and just returns)
#       a itself
#   5) device and dtype
def _parse_to_device_and_dtype(
    tensor_dtype_or_device: Optional = None,
    optional_positional_dtype: Optional = None,
    /,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
):
    # Case 3 and 5 -- device first
    if isinstance(tensor_dtype_or_device, (torch.device, devices.Device, str)):
        utils.check(device is None, lambda: f"to received both a positional and keyword device argument")
        device = to_thunder_device(tensor_dtype_or_device)

        if optional_positional_dtype is not None:
            utils.check(dtype is None, lambda: f"to received both a positional and keyword dtype argument")
            dtype = to_thunder_dtype(optional_positional_dtype)
        else:
            dtype = to_thunder_dtype(dtype)
    # Case 2 -- dtype first
    elif isinstance(tensor_dtype_or_device, (torch.dtype, dtypes.dtype)):
        utils.check(dtype is None, lambda: f"to received both a positional and keyword dtype argument")
        device = to_thunder_device(device)
        dtype = to_thunder_dtype(tensor_dtype_or_device)
    # Case 4 -- None first
    elif tensor_dtype_or_device is None:
        device = to_thunder_device(device)
        dtype = to_thunder_dtype(dtype)
    # Case 1 -- tensor first
    else:
        # See https://github.com/Lightning-AI/lightning-thunder/issues/317
        #   It'd be nice to write torch.Tensor here instead of TensorProxy
        utils.check_type(tensor_dtype_or_device, TensorProxy)
        device_ = tensor_dtype_or_device.device if device is None else to_thunder_device(device)
        dtype_ = tensor_dtype_or_device.true_dtype if dtype is None else to_thunder_dtype(dtype)
        device, dtype = device_, dtype_

    return device, dtype


# TODO Model non_blocking, copy, and memory_format (as kwargs)
@torchsymbol(torch.Tensor.to, is_method=True)
def to(
    a,
    tensor_dtype_or_device: Optional = None,
    optional_positional_dtype: Optional = None,
    /,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
):
    device, dtype = _parse_to_device_and_dtype(
        tensor_dtype_or_device, optional_positional_dtype, device=device, dtype=dtype
    )

    # NOTE to() returns the tensor unmodified if the device and dtype requested are the same
    #   (and copy=False)
    # NOTE clang.device_put does nothing when device is None or a.device == device
    a = clang.device_put(a, device)

    if dtype is not None:
        return clang.maybe_convert_to_dtype(a, dtype)

    return a


@torchsymbol(torch.Tensor.type_as, is_method=True)
def type_as(a: TensorProxy, b: TensorProxy) -> TensorProxy:
    # NOTE This type check is intentional since we're accessing the true_dtype
    #   attribute of the TensorProxy
    # TODO Create a generic Tensor annotation, and support both PyTorch
    #   tensors and TensorProxies being passed to this operation
    utils.check_type(b, TensorProxy)

    return clang.maybe_convert_to_dtype(a, b.true_dtype)


#
# Elementwise unary operaitons
#
# TODO Add type annotations


@torchsymbol(torch.abs, is_method=True)
def abs(a):
    return clang.abs(a)


@torchsymbol(torch.acos, is_method=True)
def acos(a):
    return clang.acos(a)


@torchsymbol(torch.acosh, is_method=True)
def acosh(a):
    return clang.acosh(a)


@torchsymbol(torch.asin, is_method=True)
def asin(a):
    return clang.asin(a)


@torchsymbol(torch.asinh, is_method=True)
def asinh(a):
    return clang.asinh(a)


@torchsymbol(torch.atan, is_method=True)
def atan(a):
    return clang.atan(a)


@torchsymbol(torch.atanh, is_method=True)
def atanh(a):
    return clang.atanh(a)


@torchsymbol(torch.bitwise_not, is_method=True)
def bitwise_not(a):
    return clang.bitwise_not(a)


@torchsymbol(torch.ceil, is_method=True)
def ceil(a):
    return clang.ceil(a)


@torchsymbol(torch.cos, is_method=True)
def cos(a):
    return clang.cos(a)


@torchsymbol(torch.cosh, is_method=True)
def cosh(a):
    return clang.cosh(a)


@torchsymbol(torch.erf, is_method=True)
def erf(a):
    return clang.erf(a)


@torchsymbol(torch.erfc, is_method=True)
def erfc(a):
    return clang.erfc(a)


@torchsymbol(torch.erfinv, is_method=True)
def erfinv(a):
    return clang.erfinv(a)


@torchsymbol(torch.exp, is_method=True)
def exp(a):
    return clang.exp(a)


@torchsymbol(torch.exp2, is_method=True)
def exp2(a):
    return clang.exp2(a)


@torchsymbol(torch.expm1, is_method=True)
def expm1(a):
    return clang.expm1(a)


@torchsymbol(torch.floor, is_method=True)
def floor(a):
    return clang.floor(a)


@torchsymbol(torch.isfinite, is_method=True)
def isfinite(a):
    return clang.isfinite(a)


@torchsymbol(torch.lgamma, is_method=True)
def lgamma(a):
    return clang.lgamma(a)


@torchsymbol(torch.log, is_method=True)
def log(a):
    return clang.log(a)


@torchsymbol(torch.log10, is_method=True)
def log10(a):
    return clang.log10(a)


@torchsymbol(torch.log1p, is_method=True)
def log1p(a):
    return clang.log1p(a)


@torchsymbol(torch.log2, is_method=True)
def log2(a):
    return clang.log2(a)


# TODO Move to special
# @torchsymbol(torch.ndtri, is_method=True)
# def ndtri(a):
#     return clang.ndtri(a)


@torchsymbol(torch.neg, is_method=True)
def neg(a):
    return clang.neg(a)


@torchsymbol(torch.reciprocal, is_method=True)
def reciprocal(a):
    return clang.reciprocal(a)


@torchsymbol(torch.round, is_method=True)
def round(a):
    return clang.round(a)


@torchsymbol(torch.rsqrt, is_method=True)
def rsqrt(a):
    return clang.rsqrt(a)


# TODO Complain about complex numbers like PyTorch does?
# TODO Add sgn
@torchsymbol(torch.sign, is_method=True)
def sign(a):
    return clang.sign(a)


@torchsymbol(torch.signbit, is_method=True)
def signbit(a):
    return clang.signbit(a)


# TODO Move this to torch.nn.functional
@torchsymbol(torch.nn.functional.silu)
def silu(a):
    return clang.silu(a)


@torchsymbol(torch.sin, is_method=True)
def sin(a):
    return clang.sin(a)


@torchsymbol(torch.sinh, is_method=True)
def sinh(a):
    return clang.sinh(a)


@torchsymbol(torch.sqrt, is_method=True)
def sqrt(a):
    return clang.sqrt(a)


@torchsymbol(torch.tan, is_method=True)
def tan(a):
    return clang.tan(a)


@torchsymbol(torch.tanh, is_method=True)
def tanh(a):
    return clang.tanh(a)


@torchsymbol(torch.trunc, is_method=True)
def trunc(a):
    return clang.trunc(a)


#
# Elementwise binary operations
#


@torchsymbol(torch.add, is_method=True)
def add(a, b, *, alpha=None):
    if alpha is not None:
        b = b * alpha

    return clang.add(a, b)


@torchsymbol(torch.atan2, is_method=True)
def atan2(a, b):
    return clang.atan2(a, b)


@torchsymbol(torch.bitwise_and, is_method=True)
def bitwise_and(a, b):
    return clang.bitwise_and(a, b)


@torchsymbol(torch.bitwise_or, is_method=True)
def bitwise_or(a, b):
    return clang.bitwise_or(a, b)


@torchsymbol(torch.bitwise_xor, is_method=True)
def bitwise_xor(a, b):
    return clang.bitwise_xor(a, b)


@torchsymbol(torch.copysign, is_method=True)
def copysign(a, b):
    return clang.copysign(a, b)


@torchsymbol(torch.eq, is_method=True)
def eq(a, b):
    return clang.eq(a, b)


@torchsymbol(torch.floor_divide, is_method=True)
def floor_divide(a, b):
    return clang.floor_divide(a, b)


@torchsymbol(torch.fmod, is_method=True)
def fmod(a, b):
    return clang.fmod(a, b)


@torchsymbol(torch.ge, is_method=True)
def ge(a, b):
    return clang.ge(a, b)


@torchsymbol(torch.gt, is_method=True)
def gt(a, b):
    return clang.gt(a, b)


@torchsymbol(torch.logical_and, is_method=True)
def logical_and(a, b):
    return clang.logical_and(a, b)


@torchsymbol(torch.le, is_method=True)
def le(a, b):
    return clang.le(a, b)


@torchsymbol(torch.lt, is_method=True)
def lt(a, b):
    return clang.lt(a, b)


# NOTE This is just an alias for proxies to find operation defined for the modulus
#   operator
# TODO Review this alias
def mod(a, b):
    return clang.mod(a, b)


@torchsymbol(torch.mul, is_method=True)
def mul(a, b):
    return clang.mul(a, b)


@torchsymbol(torch.ne, is_method=True)
def ne(a, b):
    return clang.ne(a, b)


@torchsymbol(torch.nextafter, is_method=True)
def nextafter(a, b):
    return clang.nextafter(a, b)


@torchsymbol(torch.pow, is_method=True)
def pow(a, b):
    return clang.pow(a, b)


@torchsymbol(torch.remainder, is_method=True)
def remainder(a, b):
    return clang.remainder(a, b)


@torchsymbol(torch.sub, is_method=True)
def sub(a, b, *, alpha=None):
    if alpha is not None:
        b = b * alpha

    return clang.sub(a, b)


@torchsymbol(torch.true_divide, is_method=True)
def true_divide(a: Number | TensorLike, b: Number | TensorLike) -> Number | TensorLike:
    return clang.true_divide(a, b)


#
# Conditional operations and masking operations
#
# TODO Can this be a method?
@torchsymbol(torch.where, is_method=True)
def where(pred: TensorLike, /, a: Number | TensorLike, b: Number | TensorLike) -> TensorLike:
    return clang.where(pred, a, b)


def _mask_tensor(a, mask):
    utils.check(
        dtypes.is_boolean_dtype(mask.dtype), lambda: f"_mask_tensor: mask ({mask.dtype=}) must have a boolean dtype"
    )

    if dtypes.is_boolean_dtype(a.dtype):
        return a & mask

    return where(mask, a, 0)


# NOTE: masked_fill is a strange wrapper around where, it probably exists only because of PyTorch's inplace pattern
# NOTE: PyTorch's masked fill requires value be a number or number tensor
# NOTE: PyTorch's masked fill is only defined as a tensor method that implicitly takes a as the first argument
# NOTE: PyTorch's masked_fill_ requires the dtype of a not change, so it checks that
#   value can be safely cast to a
# TODO: PyTorch's masked_fill always returns a contiguous tensor
# TODO: add number tensor support
@torchsymbol(torch.masked_fill, is_method=True)
def masked_fill(a: TensorLike, /, mask: TensorLike, value: Number | TensorLike) -> TensorLike:
    result = where(mask, value, a)
    return result


# NOTE The key to understanding tril is that it generates a mask
#   which (by default) masks elements of a matrix (or batch of matrices)
#   s.t. elements whose row number is greater than or equal to its column number
#   are preserved (and other numbers are set to zero).
#   When diagonal is specified, the mask computation changes so that
#   elements with rownum + diagonal >= colnum are preserved.
@torchsymbol(torch.tril, is_method=True)
def tril(a: TensorLike, /, diagonal: int = 0) -> TensorLike:
    utils.check(a.ndim >= 2, lambda: f"tril: a ({a.ndim=}) must have at least two dimensions")

    nrows, ncols = a.shape[-2:]
    row_numbers = arange(nrows, device=a.device).unsqueeze(-1)
    col_numbers = arange(ncols, device=a.device).unsqueeze(-2)

    mask = (row_numbers + diagonal) >= col_numbers

    return _mask_tensor(a, mask)


#
# Tensor creation operations
#


@torchsymbol(torch.arange)
def arange(
    start: Number,
    end: Optional[Number] = None,
    step: Number = 1,
    *,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
) -> TensorLike:
    if device is None:
        device = "cpu"

    device = to_thunder_device(device)
    dtype = to_thunder_dtype(dtype)

    if end is None:
        end = start
        start = 0
    return clang.arange(start=start, step=step, stop=end, device=device, dtype=dtype)


@torchsymbol(torch.full)
def full(
    shape: Sequence[int], fill_value: Number, *, device: Optional[DeviceLike] = None, dtype: Optional[dtypeLike] = None
) -> TensorLike:
    if device is None:
        device = "cpu"

    device = to_thunder_device(device)
    dtype = to_thunder_dtype(dtype)

    return clang.full(shape, fill_value, device=device, dtype=dtype)


@torchsymbol(torch.full_like)
def full_like(
    a: TensorLike, /, fill_value: Number, *, device: Optional[DeviceLike] = None, dtype: Optional[dtypeLike] = None
) -> TensorLike:
    device = to_thunder_device(device)
    dtype = to_thunder_dtype(dtype)
    return clang.full_like(a, fill_value, device=device, dtype=dtype)


@torchsymbol(torch.ones)
def ones(
    shape: Sequence[int], /, *, device: Optional[DeviceLike] = None, dtype: Optional[dtypeLike] = None
) -> TensorLike:
    return full(shape, 1, device=device, dtype=dtype)


@torchsymbol(torch.ones_like)
def ones_like(
    a: TensorLike, /, *, device: Optional[DeviceLike] = None, dtype: Optional[dtypeLike] = None
) -> TensorLike:
    return full_like(a, 1, device=device, dtype=dtype)


# TODO: based on uniform_, check if Torch now has a functional uniform
# NOTE: the uniform_ documentation suggests the interval is specified using "from" and "to",
#   but from is a reserved keyword in Python
@torchsymbol(is_method=False, id="torch.uniform")
def uniform(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: Union[DeviceLike],
    dtype: Union[dtypeLike],
) -> TensorLike:
    device = to_thunder_device(device)
    dtype = to_thunder_dtype(dtype)

    return clang.uniform(shape, minval, maxval, device=device, dtype=dtype)


@torchsymbol(is_method=False, id="torch.uniform_like")
def uniform_like(
    a: TensorLike,
    /,
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
) -> TensorLike:
    device = to_thunder_device(device)
    dtype = to_thunder_dtype(dtype)

    return clang.uniform_like(a, minval, maxval, device=device, dtype=dtype)


@torchsymbol(torch.zeros)
def zeros(shape: Sequence[int], *, device: Optional[DeviceLike] = None, dtype: Optional[dtypeLike] = None):
    return full(shape, 0, device=device, dtype=dtype)


@torchsymbol(torch.zeros_like)
def zeros_like(a, /, *, device: Optional[DeviceLike] = None, dtype: Optional[dtypeLike] = None):
    return full_like(a, 0, device=device, dtype=dtype)


#
# Shape operations
#


# TODO Create proper contiguous with memory format support
@torchsymbol(torch.Tensor.contiguous, is_method=True)
def contiguous(a: TensorLike, /) -> TensorLike:
    return a


# TODO: should this allow negative steps?
# TODO: we should probably be consistent about start/stop/step vs. start/end/stride language
# TODO Add type annotations
@torchsymbol(torch.Tensor.__getitem__, id="torch.Tensor.__getitem__")
def basic_indexing(a: TensorLike, /, key):
    start_indices = []
    end_indices = []
    strides = []

    # Resolves ellipses and unsqueezes
    unsqueeze_dims_pre_ellipsis = []
    unsqueeze_dims_post_ellipsis = []
    specified_slices = 0
    ellipsis_idx = None

    if isinstance(key, (Number, slice)):
        key = (key,)

    for idx, x in enumerate(key):
        if x is Ellipsis:
            utils.check(ellipsis_idx is None, lambda: f"Found two (or more) ellipses in key={key}")
            ellipsis_idx = idx
        elif isinstance(x, (Number, slice)):
            specified_slices += 1
        elif x is None:
            if ellipsis_idx is None:
                unsqueeze_dims_pre_ellipsis.append(idx)
            else:
                unsqueeze_dims_post_ellipsis.append(idx)
        else:
            raise ValueError(f"Found unexpected value {x} in key={key}")

    utils.check(
        specified_slices <= len(a.shape),
        lambda: f"Too many slices ({specified_slices}) specified for a.shape={a.shape}",
    )

    ellipsis_dims = len(a.shape) - specified_slices
    # NOTE: both these checks are required
    #   ellipsis_dims > 0 handles the implicit ellipsis matching 1+ dimensions
    #   ellipsis_idx not being None handles an explicit ellipsis which matches no dimensions
    if ellipsis_idx is not None or ellipsis_dims > 0:
        ellipsis_slices = [slice(None, None, None)] * ellipsis_dims
        if ellipsis_idx is not None:
            key = list(key)[:ellipsis_idx] + ellipsis_slices + list(key)[ellipsis_idx + 1 :]
        else:
            # NOTE: without an explicit ellipsis, there is an implicit ellipsis at the end of the key
            key = list(key) + ellipsis_slices

    # Unsqueezes
    unsqueeze_dims_post_ellipsis = [x + ellipsis_dims - 1 for x in unsqueeze_dims_post_ellipsis]
    unsqueeze_dims = unsqueeze_dims_pre_ellipsis + unsqueeze_dims_post_ellipsis
    if len(unsqueeze_dims) > 0:
        a = clang.unsqueeze(a, unsqueeze_dims)

    def _convert_none(x):
        if x is None:
            return slice(None, None, None)

        return x

    key = tuple(_convert_none(x) for x in key)

    # Handles numbers and slices
    squeeze_dims = []
    for idx, (l, x) in enumerate(zip(a.shape, key)):
        if isinstance(x, slice):
            start = x.start if x.start is not None else 0
            stop = x.stop if x.stop is not None else l
            step = x.step if x.step is not None else 1

            # Tests for negative step (PyTorch doesn't allow step < 1)
            utils.check(step >= 1, lambda: f"Expected step={step} to be weakly greater than 1")

            # Canonicalizes start and stop (allowing for values like -1)
            # NOTE: canonicalization is custom because start and stop beyond the length are allowed
            if start < 0:
                start = start + l
            utils.check(start >= 0, lambda: f"start={x.start} is not a valid index for length {l}")
            if stop < 0:
                stop = stop + l
            utils.check(stop >= 0, lambda: f"end={x.stop} is not a valid index for length {l}")

            # Handles start > stop, which occurs with slices like 3:1:1
            # NOTE: because step is always strictly positive, it's sufficient to check start
            #   and stop only here
            if start > stop:
                start = 0
                stop = 0

            # Handles overflow
            # NOTE: This is a little odd, but we just want the slice to be zero
            if start >= l:
                start = 0
                stop = 0

            if stop >= l:
                stop = l

            start_indices.append(start)
            end_indices.append(stop)
            strides.append(step)
        elif isinstance(x, Number):
            # NOTE: numbers must be valid indices after canonicalization, unlike start and stop
            x = utils.canonicalize_dim(l, x)
            start_indices.append(x)
            end_indices.append(x + 1)
            strides.append(1)
            squeeze_dims.append(idx)
        else:
            # NOTE: this is redundant with the ValueError exception above
            raise ValueError(f"Found unexpected value {x} in key={key}")

    result = prims.slice_prim(a, start_indices, end_indices, strides)

    if len(squeeze_dims) > 0:
        result = prims.squeeze(result, squeeze_dims)

    return result


@torchsymbol(torch.Tensor.expand, is_method=True)
def expand(a: TensorLike, /, *shape) -> TensorLike:
    return clang.expand(a, *shape)


# TODO Add type annotations
def get_item(a: TensorLike, /, key):
    return basic_indexing(a, key)


@torchsymbol(torch.reshape, is_method=True)
def reshape(a: TensorLike, /, *shape) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)

    return clang.reshape(a, shape)


# TODO consider revising this to just call _split_indices
# Splits a tensor along a split dimension dim into n tensors
# If input is divisible by n then every tensor will have the same length along the split dimension
# If input is not divisible by n, then the first int(input.size(dim) % n) tensors will have length
#   int(input.size(dim) / n) + 1 along the split dimension, and the remaining tensors will have
#   length int(input.size(dim) / n) along the split dimension
def _split_n(a: TensorLike, n: int, dim: int = 0) -> Tuple[TensorLike, ...]:
    dim = utils.canonicalize_dim(a.ndim, dim)

    splits = []
    dim_length = a.shape[dim]
    min_split_size = dim_length // n
    num_splits_one_extra = dim_length % n
    start_idx = 0
    for split_idx in range(n):
        split_size = min_split_size + 1 if (split_idx < num_splits_one_extra) else min_split_size
        s = clang.slice_in_dim(a, start_idx, start_idx + split_size, dim=dim)
        splits.append(s)
        start_idx = start_idx + split_size

    return tuple(splits)


# TODO could this (and other things) be revised to combine the slice_in_dim calls?
# Splits a tensor along a split dimension dim at the indices in indices
def _split_indices(a: TensorLike, indices: int, dim: int = 0) -> Tuple[TensorLike, ...]:
    dim = utils.canonicalize_dim(a.ndim, dim)

    splits = []
    start_idx = 0
    for idx in indices:
        splits.append(clang.slice_in_dim(a, start_idx, idx, dim=dim))
        start_idx = idx

    splits.append(clang.slice_in_dim(a, start_idx, a.shape[dim], dim=dim))
    return tuple(splits)


# TODO Type annoations
# See https://pytorch.org/docs/master/generated/torch.split.html
# NOTE: split is not tensor_split
#   Like tensor_split, split can work with a number or a sequence
#   If given a number, it creates tensors of equal length along the
#   split dimension, and if this is not possible then only the
#   last tensor will have a shorter length along the split
#   dimension.
#   If given a sequence, then the values in the sequence
#   define the lengths of the split dimension, not the indices
#   at which to split, and the values must sum to the length of the dimension.
@torchsymbol(torch.split, is_method=True)
def split(a, size_or_sections, dim=0):
    # TODO: see note in tensor_split
    if isinstance(size_or_sections, TensorProxy):
        raise NotImplemented

    dim = utils.canonicalize_dim(a.ndim, dim)

    utils.check(
        size_or_sections,
        (Number, Sequence),
        lambda: f"size_or_sections={size_or_sections} should be a Number or a Sequence!",
    )

    # TODO: consider revising this to just call _split_indices
    if isinstance(size_or_sections, Number):
        target_length = size_or_sections

        # Short-circuits special-case of zero
        if target_length == 0:
            utils.check(
                a.shape[dim] == 0,
                lambda: f"When size_or_sections={size_or_sections} is zero then the length of the split dimension ({a.shape[dim]}) must also be zero",
            )
            return full_like(a)

        last_length = a.shape[dim] % target_length
        num_splits = a.shape[dim] // target_length
        cur_idx = 0
        splits = []

        for _ in range(num_splits):
            splits.append(clang.slice_in_dim(a, cur_idx, cur_idx + target_length, dim=dim))
            cur_idx = cur_idx + target_length

        # Handles tail
        if last_length > 0:
            splits.append(clang.slice_in_dim(a, cur_idx, a.shape[dim], dim=dim))

        return splits

    # NOTE: isinstance(size_or_sections, Sequence)
    # Converts lengths to indices

    s = reduce(operator.add, size_or_sections, 0)
    utils.check(
        s == a.shape[dim],
        lambda: f"size_or_sections={size_or_sections} must sum to the length of the split dimension ({len(a.shape[dim])})",
    )

    # NOTE: because split requires overspecifying the lengths, the final split is ignored
    cur = 0
    indices = []
    for l in size_or_sections[: len(size_or_sections) - 1]:
        cur += l
        indices.append(cur)

    return _split_indices(a, indices, dim)


# TODO Add type annotations
# See https://pytorch.org/docs/master/generated/torch.squeeze.html
@torchsymbol(torch.squeeze, is_method=True)
def squeeze(a: TensorLike, /, dim: Optional[Tuple[int, ...] | int] = None) -> TensorLike:
    # Converts dim to a tuple of numbers
    dims = dim
    if dim is None:
        dims = []
        for idx, l in enumerate(a.shape):
            if l == 1:
                dims.append(idx)
    elif isinstance(dim, Number):
        dims = (dim,)

    return clang.squeeze(a, dims)


# TODO Add type annotations
# See https://pytorch.org/docs/master/generated/torch.tensor_split.html
@torchsymbol(torch.tensor_split, is_method=True)
def tensor_split(a: TensorLike, /, indices_or_sections, dim=0):
    # TODO: consider if we even should support this, it could introduce data-dependent control flow
    # NOTE: this will also catch number tensors
    if isinstance(indices_or_sections, TensorProxy):
        raise NotImplemented

    utils.check(
        indices_or_sections,
        (Number, Sequence),
        lambda: f"indices_or_sections={indices_or_sections} should be a Number or a Sequence!",
    )

    # TODO: maybe revise _split_n to a call to _split_indices
    if isinstance(indices_or_sections, Number):
        return _split_n(a, indices_or_sections, dim)

    # NOTE: isinstance(indices_or_sections, Sequence)
    return _split_indices(a, indices_or_sections, dim)


@torchsymbol(torch.transpose, is_method=True)
def transpose(a: TensorLike, /, dim0: int, dim1: int) -> TensorLike:
    dim0, dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))

    permutation = list(range(0, a.ndim))
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    return clang.transpose(a, permutation)


def matrix_transpose(a: TensorLike, /) -> TensorLike:
    """Transposes the last two dimensions of a tensor.

    This function is used to implement the `.mT` attribute.

    Args:
        a (TensorProxy): The tensor to transpose.

    Returns:
        TensorProxy: The transposed tensor.

    Examples:
        >>> a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> def func(x): return x.mT
        >>> traced_func = thunder.compile(func)
        >>> traced_func(a)
        tensor([[1, 4],
                [2, 5],
                [3, 6]])
    """
    return clang.matrix_transpose(a)


@torchsymbol(torch.permute, is_method=True)
def permute(a: TensorLike, /, *dims: tuple[int, ...]) -> TensorLike:
    dims = utils.extract_shape_from_varargs(dims)
    return clang.transpose(a, dims)


@torchsymbol(torch.unsqueeze, is_method=True)
def unsqueeze(a: TensorLike, /, dim: int) -> TensorLike:
    return clang.unsqueeze(a, (dim,))


@torchsymbol(torch.cat)
def cat(tensors: Sequence[TensorLike], dim: int = 0) -> TensorLike:
    return clang.cat(tensors, dim)


# TODO Add type annotations
@torchsymbol(torch.stack)
def stack(tensors: Sequence[TensorLike], dim: int = 0) -> TensorLike:
    return clang.stack(tensors, dim)


@torchsymbol(torch.index_select)
def index_select(a: TensorLike, /, dim: int, index: TensorLike) -> TensorLike:
    return clang.take(a, index, dim)


@torchsymbol(torch.index_add)
def index_add(a: TensorLike, /, dim: int, index: TensorLike, source: TensorLike) -> TensorLike:
    return clang.index_add(a, index, source, dim)


@torchsymbol(torch.take_along_dim)
def take_along_dim(input: TensorLike, indices: TensorLike, dim: int) -> TensorLike:
    return clang.take_along_axis(input, indices, dim)


@torchsymbol(torch.scatter_add)
def scatter_add(a: TensorLike, dim: int, index: TensorLike, source: TensorLike) -> TensorLike:
    return clang.scatter_add(a, index, source, dim)


# TODO Add type annotations
@torchsymbol(torch.flatten, is_method=True)
def flatten(a: TensorLike, /, start_dim=0, end_dim=-1) -> TensorLike:
    s = a.shape
    if end_dim < 0:
        # end_dim is inclusive in flatten(!)
        end_dim = len(s) + end_dim + 1
    if start_dim + 1 >= end_dim:
        return a
    s_new = tuple(s[:start_dim]) + (-1,) + tuple(s[end_dim:])
    return reshape(a, s_new)


# TODO Review view functionalization
# TODO Add type annotations
@torchsymbol(torch.Tensor.view, is_method=True)
def view(a: TensorLike, /, *shape) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)
    return reshape(a, shape)


#
# Reduction operations
#


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)
    COMPLEX_TO_FLOAT = (1,)
    # Keeps the output in the computation type (used for mean)
    KEEP_PROMOTED_TYPE = (2,)
    ALWAYS_BOOL = (3,)


def _reduction_dtypes(
    arg,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
    dtype=None,
):
    # even though some reductions, like amin or amax, don't strictly require type promotion,
    # all the math ops (including comparisons) are still defined only for a computation type,
    # so promotion will still happen. We are doing it explicitly here
    inp_dtype = dtype if dtype is not None else arg.dtype
    computation_dtype = utils.get_computation_dtype(inp_dtype)
    if (
        output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME
        or output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    ):
        result_dtype = dtype if dtype else arg.dtype
        if output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT and utils.is_complex_dtype(result_dtype):
            result_dtype = utils.corresponding_real_dtype(result_dtype)
    elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE:
        result_dtype = None
    else:  # ALWAYS_BOOL
        result_dtype = torch.bool
    return computation_dtype, result_dtype


def _reduction_dims(shape, dims: Optional[Sequence]) -> tuple[int, ...]:
    if isinstance(dims, int):
        dims = (dims,)
    if dims is None or len(dims) == 0:
        return tuple(range(len(shape)))

    dims = tuple(utils.canonicalize_dim(len(shape), idx) for idx in dims)
    utils.check_no_duplicates(dims)

    return dims


# TODO Restore out support?
def _reduction(
    a: TensorProxy,
    prim: Callable,
    *,
    has_identity: bool = True,
    accepts_dim_tuple: bool = True,  # to handle min/argmin that accept single dim only
    dims=None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,  # should be specified for ops that support it
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
) -> TensorProxy:
    # TODO: check that a is the correct type?

    # reduces over all dimensions if dim=() is passed
    if dims == () or dims == []:
        dims = None
    if isinstance(dims, int):
        dims = (dims,)

    utils.check(
        a.ndim <= 64,
        lambda: f"Received a tensor with {a.ndim} dimensions, but only tensors with up to 64 dims are supported!",
    )

    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, int)

    if isinstance(dims, int):
        dims = (dims,)

    dims = _reduction_dims(a.shape, dims)

    if not has_identity:
        valid_shape = (a.ndim == 0) or all(a.shape[i] for i in dims)
        utils.check(
            valid_shape,
            lambda: "Can't reduce over a zero-size dimension when computing a reduction without an identity value.",
        )

    computation_dtype, result_dtype = _reduction_dtypes(a, output_dtype_kind, dtype)

    a = clang.maybe_convert_to_dtype(a, computation_dtype)
    result = prim(a, dims)

    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = tree_map(lambda x: prims.broadcast_in_dim(x, output_shape, broadcast_dims), result)

    if result_dtype is not None:
        result = tree_map(lambda x: clang.maybe_convert_to_dtype(x, result_dtype), result)

    return result


# Helper to handle the unbiased->correction deprecation on ops like var
def _set_correction(
    unbiased: Optional[bool] = None,
    correction: Optional[int] = None,
):
    utils.check(
        correction is None or unbiased is None,
        lambda: f"Both correction and unbiased cannot be specified",
        exception_type=AssertionError,
    )

    if correction is None and unbiased is None:
        correction = 1
    elif correction is None and unbiased is not None:
        correction = 0 if not unbiased else 1

    utils.check_type(correction, int)
    utils.check(correction >= 0, lambda: f"{correction=} must be non-negative")

    return correction


def _dim_var_dispatch(dim=None, unbiased=None):
    # NOTE There's the following overload of torch.var:
    # var(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
    # Requiring explicitly converting bool dims to unbiased arg
    if unbiased is None and isinstance(dim, bool):
        unbiased = dim
        dim = None
    return dim, unbiased


@torchsymbol(torch.amax, is_method=True)
def amax(a, dim=None, keepdim: bool = False):
    return _reduction(
        a,
        prims.amax,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@torchsymbol(torch.amin, is_method=True)
def amin(a, dim=None, keepdim: bool = False):
    return _reduction(
        a,
        prims.amin,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@torchsymbol(torch.mean, is_method=True)
def mean(a: TensorProxy, dim=None, keepdim: bool = False, *, dtype=None):
    dtype = dtype if dtype is not None else a.dtype
    utils.check(
        not utils.is_integer_dtype(dtype) and not utils.is_boolean_dtype(dtype),
        lambda: f"dtype={dtype} is not a floating point or complex dtype",
    )

    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE,
    )

    dims = _reduction_dims(a.shape, dim)  # type: ignore[arg-type]
    nelem = 1 if a.ndim == 0 else reduce(operator.mul, (a.shape[i] for i in dims), 1)
    result = result / nelem
    result_dtype = a.dtype if dtype is None else dtype
    result = clang.maybe_convert_to_dtype(result, result_dtype)
    return result


@torchsymbol(torch.prod, is_method=True)
def prod(a: TensorProxy, dim=None, keepdim=False, *, dtype=None):
    # Promotes all exact dtypes to int64
    if dtype is None:
        if utils.is_exact_dtype(a.dtype):
            dtype = dtypes.int64
        else:
            dtype = a.dtype

    result = _reduction(
        a,
        prims.prod,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )

    return result


@torchsymbol(torch.sum, is_method=True)
def sum(a: TensorProxy, dim=None, keepdim=False, *, dtype=None):
    # Promotes all exact dtypes to int64
    if dtype is None:
        if utils.is_exact_dtype(a.dtype):
            dtype = dtypes.int64
        else:
            dtype = a.dtype

    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )

    return result


@torchsymbol(torch.var, is_method=True)
def var(
    a: TensorProxy,
    dim=None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
) -> TensorProxy:
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = _set_correction(unbiased, correction)

    result = _reduction(
        a,
        partial(prims.var, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    return result


# TODO: consider being more aggressive about kwarg-only
@torchsymbol(torch.var_mean)
def var_mean(
    a: TensorProxy,
    dim=None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
) -> tuple[TensorProxy, TensorProxy]:
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = _set_correction(unbiased, correction)

    result = _reduction(
        a,
        partial(prims.var_mean, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    return result


#
# norm operations
#


def _normalize(a: TensorProxy, norm_dims, eps: Number):
    """Computes mean and 1/std of a tensor along norm_dims. Used as a helper function for normalization layers.

    Args:
        a (Tensor): input tensor
        norm_dims (DimsType): dimensions to normalize over
        eps (float): epsilon for numerical stability

    Returns:
        out (Tensor): normalized tensor.
        mean (Tensor): mean of the tensor along norm_dims.
        rstd (Tensor): 1/std of the tensor along norm_dims.
    """
    norm_dims = utils.canonicalize_dims(a.ndim, norm_dims)
    computation_dtype = utils.get_computation_dtype(a.dtype)
    a_acc = clang.maybe_convert_to_dtype(a, computation_dtype)
    biased_var, mean = var_mean(a_acc, dim=norm_dims, unbiased=False, keepdim=True)
    rstd = rsqrt(biased_var + eps)
    out = (a - mean) * rstd
    return out, mean, rstd


# TODO: likely want to refactor these normalizations
def _native_layer_norm(a: TensorProxy, normalized_shape, weight, bias, eps: Number):
    # Validates inputs
    normalized_ndim = len(normalized_shape)
    utils.check(normalized_ndim >= 1, lambda: f"Expected normalized_shape={normalized_shape} to have length >= 1!")

    # NOTE Containers are canonicalized in the following checks since
    #   (1, 2, 3) != [1, 2, 3]
    utils.check(
        weight is None or weight.shape == tuple(normalized_shape),
        lambda: f"Expected weight.shape={weight.shape} to be the same as normalized_shape={normalized_shape}!",
    )
    utils.check(
        bias is None or bias.shape == tuple(normalized_shape),
        lambda: f"Expected bias.shape={bias.shape} to be the same as normalized_shape={normalized_shape}!",
    )
    utils.check(
        a.ndim >= normalized_ndim,
        lambda: f"Expected a.ndim={a.ndim} to be greater than or equal to len(normalized_shape)={normalized_ndim}",
    )
    utils.check(
        a.shape[(a.ndim - normalized_ndim) :] == tuple(normalized_shape),
        lambda: f"Expected the last {len(normalized_shape)} dimensions of a (a.shape={a.shape}) to be the same as {normalized_shape}",
    )

    axis = a.ndim - normalized_ndim
    reduction_dims = list(range(axis, a.ndim))
    out, mean, rstd = _normalize(a, reduction_dims, eps)

    # Handles weight and bias
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias

    out = clang.maybe_convert_to_dtype(out, a.dtype)
    # TODO Is the following conversion or conversions CPU only?
    # if input.device.type == "cpu":
    mean = clang.maybe_convert_to_dtype(mean, a.dtype)
    rstd = clang.maybe_convert_to_dtype(rstd, a.dtype)

    return out, mean, rstd


# TODO Add type annotations
# TODO Move this to nn.functional
@torchsymbol(torch.nn.functional.layer_norm)
def layer_norm(a, normalized_shape, weight=None, bias=None, eps: Number = 1e-5):
    return _native_layer_norm(a, normalized_shape, weight, bias, eps)[0]


#
# nn operations
#


# TODO bmm is more restrictive than matmul
@torchsymbol(torch.bmm, is_method=True)
def bmm(a, b):
    return matmul(a, b)


def _dropout_helper(a, p):
    """Helper function for all dropout-type operators. During training, some of the elements of the input tensor are
    randomly masked.

    Returns the masked tensor of the boolean values.
    """

    r = uniform_like(a, 0.0, 1.0)
    result = r < p

    return result


# TODO Is this a method?
# TODO Move this to nn.functional
# NOTE The id must be explicitly specified so as not to resolve to torch.dropout
#   (Using torch.nn.functional.dropout is just for readability as it's the documented operator)
@torchsymbol(torch.nn.functional.dropout, id="torch.nn.functional.dropout")
def dropout(a: TensorProxy, p: Number = 0.5, training: bool = True, inplace: bool = False):
    if not training or inplace:
        raise NotImplementedError("Only training=True, inplace=False is currently supported in dropout")

    utils.check(
        p <= 1 and p >= 0,
        lambda: f"Dropout probability has to be between 0 and 1, but got, {p}",
    )

    if p == 1:
        return zeros_like(a)

    if p == 0:
        return a

    scale = 1 / (1 - p)
    dropout_mask = _dropout_helper(a, 1 - p)

    return a * dropout_mask * scale


# TODO Move this to nn.functional
@torchsymbol(torch.nn.functional.linear)
def linear(a, w, bias=None):
    return prims.linear(a, w, bias)


# NOTE: this wrapper for prim matmul just broadcasts batch dimensions
@torchsymbol(torch.matmul, is_method=True)
def matmul(a, b):
    if a.ndim == 1 or b.ndim == 1:
        return prims.matmul(a, b)

    a_batch_dims = a.shape[:-2]
    b_batch_dims = b.shape[:-2]

    batch_dims_broadcast = list(clang.compute_broadcast_shape(a_batch_dims, b_batch_dims))

    a_broadcast_shape = batch_dims_broadcast + list(a.shape[-2:])
    if not utils.same_shape(a_broadcast_shape, a.shape):
        a = clang.expand(a, a_broadcast_shape)

    b_broadcast_shape = batch_dims_broadcast + list(b.shape[-2:])
    if not utils.same_shape(b_broadcast_shape, b.shape):
        b = clang.expand(b, b_broadcast_shape)

    return prims.matmul(a, b)


@torchsymbol(torch.nn.functional.embedding, id="torch.nn.functional.embedding")
def embedding(a, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    # TODO: add embedding_renorm_ so we can remove embedding prim
    # NOTE: padding_idx has impact on backward and is not supported by take
    if max_norm is not None or padding_idx is not None:
        padding_idx = padding_idx if padding_idx is not None else -1
        return prims.embedding(
            a,
            weight,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

    # padding_idx / sparse not used by forward

    if a.ndim == 1:
        return clang.take(weight, a, 0)

    output_shape = list(a.shape) + list(weight.shape[1:])
    flatten_indices = clang.reshape(a, [a.numel])
    flatten_output = clang.take(weight, flatten_indices, 0)
    return clang.reshape(flatten_output, output_shape)


@torchsymbol(torch.ops.aten.embedding_backward)
def embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    result = prims.embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse)
    return result


# CompositeImplicitAutograd - don't register decomp
@torchsymbol(torch.softmax, torch.nn.functional.softmax, is_method=True)
def softmax(a, dim, dtype=None):
    result_dtype = dtype or a.dtype
    computation_dtype = utils.get_computation_dtype(result_dtype)
    a_ = clang.maybe_convert_to_dtype(a, computation_dtype)

    if a.numel == 0:
        a_exp = exp(a_)
    else:
        a_max = amax(a_, dim, keepdim=True)
        a_exp = exp(a_ - a_max)

    result = true_divide(a_exp, sum(a_exp, dim, keepdim=True))
    converted = clang.maybe_convert_to_dtype(result, result_dtype)
    return converted


@torchsymbol(torch.outer)
def outer(a, b):
    utils.check(a.ndim == 1, lambda: f"Expected {a.ndim=} to be one")
    utils.check(b.ndim == 1, lambda: f"Expected {b.ndim=} to be one")

    return clang.mul(a[:, None], b[None, :])


@torchsymbol(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    logits = query @ key.transpose(-2, -1) / (query.size(-1) ** 0.5)
    if attn_mask is None and is_causal:
        L, S = logits.shape[-2:]
        attn_mask = (arange(L, device=query.device)[:, None] >= arange(S, device=query.device)[None, :]).to(
            dtype=logits.dtype
        )
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -math.inf)
    if attn_mask is not None:
        logits = logits + attn_mask
    attn_weight = softmax(logits, dim=-1)
    attn_weight = dropout(attn_weight, dropout_p)
    return attn_weight @ value


# TODO Add type annotations, change the name "input" to "a", require "a" be specified positionally
@torchsymbol(torch.nn.functional.cross_entropy)
def cross_entropy(
    input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0
):
    utils.check(
        size_average is None and reduce is None,
        lambda: f"Deprecated size_average={size_average} and reduce={reduce} is not supported!",
    )
    utils.check(
        input.ndim >= 1,
        lambda: f"cross_entropy gets input.ndim: {input.ndim} < 1",
    )
    # NOTE: label_smoothing < 0 will just be ignored.
    utils.check(
        label_smoothing <= 1.0,
        lambda: f"label_smoothing must be less than 1.0. Got: {label_smoothing}",
    )
    # extract shape information
    C_dim = 1 if input.ndim >= 2 else 0
    N = input.shape[0] if input.ndim >= 2 else 1
    C = input.shape[C_dim]
    feature_size = int(input.numel / N / C)

    # short-cut to output empty tensor
    if input.numel == 0:
        if reduction == "none":
            output_shape = list(input.shape)
            output_shape.pop(C_dim)
            return clang.full(output_shape, 0.0, device=input.device, dtype=input.dtype)
        elif reduction == "mean":
            # TODO: I can't use `float("nan")` here
            fill_value = math.nan
        elif reduction == "sum":
            fill_value = 0.0
        else:
            raise ValueError(f"reduction argument: {reduction} to cross_entropy is not supported")

        return clang.full([], fill_value, device=input.device, dtype=input.dtype)

    if weight is not None:
        utils.check(
            weight.ndim == 1 and weight.numel == C,
            lambda: f"inconsisten input: {input.shape} / weight: {weight.shape} to cross_entropy!",
        )
        bcast_weight = clang.reshape(weight, [C] + [1 for i in range(2, input.ndim)])

    # log_softmax
    # softmax_input = _softmax_decomp(input, C_dim)
    # log_softmax_input = clang.log(softmax_input)
    # implementation suggested by Jacob to avoid division in _softmax_decomp
    input_max = amax(input, C_dim, keepdim=True)
    input_prime = clang.sub(input, input_max)
    sumexp = sum(clang.exp(input_prime), C_dim, keepdim=True)
    log_softmax_input = clang.sub(input_prime, clang.log(sumexp))

    out = clang.neg(log_softmax_input)

    if input.shape == target.shape:
        utils.check(
            utils.is_float_dtype(target.dtype),
            lambda: f"expect float dtype for probability target, but got: {target.dtype}!",
        )
        utils.check(
            ignore_index < 0,
            lambda: f"ignore_index is not supported for probability target, set ignore_index < 0!",
        )

        if label_smoothing > 0.0:
            target = clang.add(clang.mul(target, 1 - label_smoothing), label_smoothing / C)

        out = clang.mul(out, target)

        if weight is not None:
            out = clang.mul(out, bcast_weight)

        if target.ndim == 1:
            out = _reduction(
                out,
                prims.sum,
                output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
            )
        else:
            out = _reduction(
                out,
                prims.sum,
                dims=C_dim,
                output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
            )

        if reduction == "none":
            return out
        # TODO: duplicate this in probability target!
        elif reduction == "sum":
            # NOTE: do we need to promote dtype?!
            return _reduction(
                out,
                prims.sum,
                output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
            )
        elif reduction == "mean":
            reduced_sum = _reduction(
                out,
                prims.sum,
                output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
            )
            # NOTE: does it work with dynamic size?!
            return clang.true_divide(reduced_sum, N * feature_size)
        else:
            raise ValueError(f"reduction argument: {reduction} to cross_entropy is not supported")
    else:
        utils.check(
            utils.is_integer_dtype(target.dtype),
            lambda: f"expect integer dtype for class indices target, but got: {target.dtype}!",
        )
        no_C_shape = list(input.shape)
        no_C_shape.pop(C_dim)
        utils.check(
            input.ndim == target.ndim + 1 and no_C_shape == list(target.shape),
            lambda: f"inconsisten shape input: {input.shape} / target: {target.shape} to cross_entropy!",
        )

        # nll_loss
        if weight is not None:
            out = clang.mul(out, bcast_weight)

        smooth_loss_no_sum = out
        # TODO: swap reshape with unsqueeze when nvfuser support is added
        # bcast_target = clang.unsqueeze(target, [C_dim])
        bcast_target_shape = list(input.shape)
        bcast_target_shape[C_dim] = 1
        bcast_target = clang.reshape(target, bcast_target_shape)

        out = clang.take_along_axis(out, bcast_target, C_dim)

        if label_smoothing > 0:
            # smooth_loss shape [N, SPATIAL...]
            smooth_loss = _reduction(
                smooth_loss_no_sum,
                prims.sum,
                dims=[C_dim],
                output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
            )
        # NOTE: [handling of 'ignore_index']
        #       Semantically, I think we are doing the right thing here where we mask out the ignore_index entries on output from clang.take_along_axis. Because targets is expected to be within [0, C)
        #       However, in Torch/ATen implementation, 'ignore_index' can be outside of the range, so is targets. So it could even prevent an out-of-bound error from NLLLoss. Which diverges from the behavior here.
        #       Note that we can mimic that behavior by mask targets before take_along_axis, but that's going to add more operations here, which means more overhead. Let's not do that until we see real examples exploiting the behavior.
        #       Alternatively, we can revisit the choice of numpy.take_along_axis.
        #       jax.numpy.take_along_axis gives a 'mode' arg custom out-of-bound behavior. But that might be slightly tricky to handle for codegen.
        if ignore_index >= 0:
            # mask shape [N, 1, SPATIAL...]
            mask = clang.eq(bcast_target, ignore_index)
            out = clang.where(mask, 0, out)
            if label_smoothing > 0:
                # TODO: switch to squeeze
                smooth_loss = clang.where(clang.reshape(mask, list(smooth_loss.shape)), 0, smooth_loss)

        if reduction == "none":
            # TODO: swap reshape with squeeze when nvfuser support is added
            # return clang.squeeze(out, [C_dim])
            out = clang.reshape(out, target.shape)
            if label_smoothing > 0:
                ret = smooth_loss
        # TODO: duplicate this in probability target!
        elif reduction == "sum":
            # NOTE: do we need to promote dtype?!
            out = _reduction(
                out,
                prims.sum,
                output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
            )
            if label_smoothing > 0:
                ret = _reduction(
                    smooth_loss,
                    prims.sum,
                    output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
                )
        elif reduction == "mean":
            # NOTE: do we need to promote dtype?!
            reduced_sum = _reduction(
                out,
                prims.sum,
                output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
            )
            if label_smoothing > 0:
                ret = _reduction(
                    smooth_loss,
                    prims.sum,
                    output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
                )
            if weight is not None:
                # NOTE: this seems unreasonably complicated. Am I missing something obvious?!
                input_shape = list(input.shape)
                expanded_weight = clang.expand(bcast_weight, input_shape)
                # DEBUG!!! this gives segfaults
                selected_weight = clang.take_along_axis(expanded_weight, bcast_target, C_dim)

                if ignore_index >= 0:
                    selected_weight = clang.where(mask, 0, selected_weight)

                bcast_weight_sum = _reduction(
                    selected_weight,
                    prims.sum,
                    output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
                )
                out = clang.true_divide(reduced_sum, bcast_weight_sum)
                if label_smoothing > 0:
                    ret = clang.true_divide(ret, bcast_weight_sum)
            elif ignore_index >= 0:
                mask_sum = _reduction(
                    mask,
                    prims.sum,
                    dtype=to_thunder_dtype(torch.float),
                    output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
                )
                # NOTE: does the call numel here work with dynamic shape?!
                out = clang.true_divide(reduced_sum, clang.sub(target.numel, mask_sum))
                if label_smoothing > 0:
                    ret = clang.true_divide(ret, clang.sub(target.numel, mask_sum))
                elif target.ndim == 0:
                    # NOTE: this is pytorch implementation details.
                    # overwrite output to 0 when target hits ignore_index AND label_smoothing is missing.
                    # https://github.com/pytorch/pytorch/pull/64572
                    out = clang.where(clang.eq(target, ignore_index), 0, out)
            else:
                out = clang.true_divide(reduced_sum, target.numel)
                if label_smoothing > 0:
                    ret = clang.true_divide(ret, target.numel)
        else:
            raise ValueError(f"reduction argument: {reduction} to cross_entropy is not supported")

        if label_smoothing > 0:
            return clang.add(clang.mul(out, 1 - label_smoothing), clang.mul(ret, (label_smoothing / C)))
        else:
            return out


#
# torch -> thunder object mapping
#


_torch_to_thunder_complete_map = {
    **_torch_to_thunder_dtype_map,
    **_torch_to_thunder_function_map,
}

#
# Prim implementations
#
# NOTE These operations are called when a primitive is invoked eagerly.
#   They handle number, torch.Tensor, and np.ndarray inputs.
# TODO Reconcile these definitions with the primitive operator mappings in the PyTorch executor
# TODO Consider having NumPy arrays handled by the NumPy language definition -- but that would require a more
#   complicated dispatch mechanism
from thunder.core.prims import PrimIDs as pids
import numpy as np
import thunder.numpy as lnp

_primid_to_impl_map = {}


class eager_for:
    def __init__(self, id):
        self.id = id

    def __call__(self, fn):
        _primid_to_impl_map[self.id] = fn
        return fn


def get_eager_implementation_for(id: prims.PrimIDs) -> Optional[Callable]:
    return _primid_to_impl_map.get(id, None)


@eager_for(pids.CONVERT_ELEMENT_TYPE)
def _convert_element_type_eager(
    a: Union[torch.Tensor, np.ndarray, Number], dtype: Union[dtypes.dtype, type]
) -> Union[torch.Tensor, Number]:
    utils.check_type(a, (torch.Tensor, np.ndarray, Number))
    utils.check_type(dtype, (dtypes.dtype, type))

    if isinstance(a, Number):
        utils.check(
            dtype in dtypes.all_numbertypes, lambda: f"Expected {dtype} to be a numbertype in {dtypes.all_numbertypes}"
        )
        return dtype(a)

    if isinstance(a, torch.Tensor):
        torch_dtype = to_torch_dtype(dtype)
        return a.to(dtype)

    if isinstance(a, np.ndarray):
        np_dtype = lnp.to_numpy_dtype(dtype)
        return a.astype(dtype)

    utils.check(False, lambda: f"Unexpected case!", exception_type=AssertionError)


def _elementwise_binary_eager(
    a: Union[torch.Tensor, np.ndarray, Number],
    b: Union[torch.Tensor, np.ndarray, Number],
    *,
    name: str,
    number_fn: Optional[Callable] = None,
    torch_fn: Optional[Callable] = None,
):
    utils.check_type(name, str)
    utils.check_type(a, (torch.Tensor, np.ndarray, Number))
    utils.check_type(b, (torch.Tensor, np.ndarray, Number))

    if isinstance(a, Number) and isinstance(b, Number):
        utils.check(number_fn is not None, lambda: f"", exception_type=NotImplementedError)
        return number_fn(a, b)

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        utils.check(False, lambda: f"{name} does not yet support NumPy arrays", exception_type=NotImplementedError)

    # NOTE isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    utils.check(
        torch_fn is not None, lambda: f"{name} does not yet support Torch tensors", exception_type=NotImplementedError
    )
    return torch_fn(a, b)


add_eager = partial(_elementwise_binary_eager, name="add_eager", number_fn=operator.add, torch_fn=torch.add)
eager_for(pids.ADD)(add_eager)
