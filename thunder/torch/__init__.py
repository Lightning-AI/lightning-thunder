from __future__ import annotations
import builtins
import itertools
import math
import operator
import collections
import re
import sys
from collections.abc import Callable
from collections.abc import Sequence
from enum import Enum
from functools import partial, reduce, wraps
from numbers import Number
from types import NoneType, ModuleType
from typing import Any, overload
import builtins
import collections
import itertools
import math
import operator
import re

import opt_einsum

# Initializes the language context
from thunder.torch.langctx import register_method, register_property
from thunder.core.baseutils import run_once
import thunder.clang as clang
import thunder.core.devices as devices
from thunder.core.devices import to_device
import thunder.core.dtypes as dtypes
from thunder.core.dtypes import to_torch_dtype, to_dtype, _thunder_to_torch_dtype_map, _torch_to_thunder_dtype_map
import thunder.core.prims as prims
import thunder.core.utils as utils
import thunder.distributed.prims as dist_prims
from thunder.core.langctxs import langctx, Languages, get_langctx
from thunder.core.compile_data import get_compile_data
from thunder.core.proxies import (
    FloatProxy,
    IntegerProxy,
    NumberProxy,
    NumberLike,
    TensorProxy,
    FutureTensorProxy,
    pyval,
    TupleProxy,
    ListProxy,
    DictProxy,
    numberproxy,
    ProxyTag,
)
from thunder.core.pytree import tree_map, tree_flatten, tree_unflatten
from thunder.core.symbol import Symbol
from thunder.core.transforms import register_grad, register_augmented_forward, register_backward
from thunder.core.prims import get_grad, put_grad
import thunder
from thunder.torch.default_torch_ops import _auto_registered_operators_returning_views


__all__ = [
    "is_available",
]

# NOTE torch is a requirement
import torch
import torch.utils.checkpoint
import torch._higher_order_ops.wrap

import warnings

# Type annotation helpers
TensorLike = TensorProxy
FutureTensorLike = FutureTensorProxy
DeviceLike = str | devices.Device | torch.device
dtypeLike = dtypes.dtype | torch.dtype


# TODO RC1 Remove this map
_torch_noinline_functions = {
    torch.nn.modules.utils._single,
    torch.nn.modules.utils._pair,
    torch.nn.modules.utils._triple,
    torch.nn.modules.utils._quadruple,
}

# Maps torch functions, like torch.foo, to their corresponding thunder.torch functions
# NOTE This is defined here and populated as functions are defined below
_torch_to_thunder_function_map: dict[Callable, Callable] = {}

#
# torch operation definitions
#

# in-place sym -> out-of-place (= functional) sym with index of `inplace: bool` argument
# If an in-place op doesn't have `inplace: bool` argument, set -1.
_inplace_to_out_of_place: dict[Callable, tuple[Callable, int]] = {}


# Helpers for factory functions to get default dtype and device.
def get_default_dtype():
    # `thunder.jit` will create cache info and stash the default dtype
    # observed at the beginning of jitting.
    cache_info = thunder._get_cache_info()

    # Currently, changing dtype during the jitted function is unsupported.
    utils.check(
        cache_info["default_dtype"] == torch.get_default_dtype(),
        lambda: "Default dtype is changed during the execution of jitted function. This is currently unsupported.",
    )
    return torch.get_default_dtype()


def maybe_get_default_dtype(dtype):
    return dtype or get_default_dtype()


def get_default_device():
    # `thunder.jit` will create cache info and stash the default device
    # observed at the beginning of jitting.
    cache_info = thunder._get_cache_info()

    # Currently, changing device during the jitted function is unsupported.
    utils.check(
        cache_info["default_device"] == torch.get_default_device(),
        lambda: "Default device is changed during the execution of jitted function. This is currently unsupported.",
    )
    return torch.get_default_device()


def maybe_get_default_device(device):
    return device or get_default_device()


# A wrapper that executes the operations within the torch language context
# NOTE because this module defines the torch language context, a reference to itself
#   is acquired by inspecting the __module__ attribute of the is_available function defined
#   above
# NOTE Functions that set is_method=True must be able to accept a tensor as their first positional input
class torchsymbol:
    def __init__(
        self,
        *torchfns,
        is_method: bool = False,
        method_name: None | str = None,
        is_property: bool = False,
        id: str | None = None,
        is_prim: bool = False,
        tags: None | list[Any] = None,
    ):
        self.torchfns = torchfns
        self.is_method = is_method or (method_name is not None)
        self.method_name: None | str = method_name
        self.is_property = is_property
        self.id = id
        # When is_prim is True, the function is treated as a primitive, so that
        # executors must execute it directly without decomposition.
        self.is_prim = is_prim
        self.tags = tags

    def __call__(self, fn: Callable) -> Symbol:
        _fn = langctx(Languages.TORCH)(fn)

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
            elif hasattr(torch.special, name):
                id = f"torch.special.{name}"
            else:
                utils.check(
                    False,
                    lambda: f"The torchsymbol decorator failed to infer an id for {name}, specify one explicitly (with id=<your id>)",
                    exception_type=AssertionError,
                )
        else:
            id = self.id

        if self.is_prim:
            sym = Symbol(
                name=fn.__name__, meta=langctx(Languages.PRIMS)(_fn), id=id, is_prim=self.is_prim, tags=self.tags
            )
        else:
            sym = Symbol(name=fn.__name__, meta=_fn, id=id, is_prim=self.is_prim, tags=self.tags)

        if self.is_method:
            method_name: str = self.method_name if self.method_name is not None else fn.__name__
            register_method(method_name, sym)
            torch_method: None | Callable = getattr(torch.Tensor, method_name, None)
            if torch_method is not None:
                _torch_to_thunder_function_map[torch_method] = sym
        elif self.is_property:
            method_name: str = self.method_name if self.method_name is not None else fn.__name__
            register_property(method_name, sym)
            torch_property = getattr(torch.Tensor, method_name, None)
            if torch_property is not None:
                _torch_to_thunder_function_map[torch_property] = sym

        if self.torchfns is not None:
            for torchfn in self.torchfns:
                _torch_to_thunder_function_map[torchfn] = sym

        if self.tags and prims.OpTags.IN_PLACE in self.tags:
            if self.id is not None:
                name = self.id
            _inplace_to_out_of_place[sym] = globals()[name[:-1]], -1

        return sym


# This is function maps an implementation for `torch` operation without creating a Symbol.
# So, the registered implementation will not show up in trace as a Symbol (but will get inlined).
# This is helpful if we want to support a torch operation and bake it's output directly into the trace.
# See `clone` and `torch.device` for example.
def register_function(torchfn, thunderfn_impl):
    _torch_to_thunder_function_map[torchfn] = thunderfn_impl


#
# Tensor properties
#


@torchsymbol(torch.Tensor.dim, is_method=True)
def dim(a: TensorLike, /) -> int:
    return a.ndim


# NOTE: Named `compute_len` so that it doesn't
#       conflict with built-in `len`
def compute_len(a: TensorLike, /) -> int:
    if a.ndim == 0:
        raise TypeError("len() of a 0-d tensor")
    return a.shape[0]


register_method("len", compute_len)


@torchsymbol(torch.is_floating_point, is_method=True)
def is_floating_point(a: TensorLike, /) -> bool:
    return dtypes.is_float_dtype(a.dtype)


# Handles the size method
def size(a: TensorLike, /, dim: None | int = None) -> int | Sequence[int]:
    if dim is not None:
        return prims.shape(a)[dim]
    return prims.shape(a)


register_method("size", size)


@torchsymbol(torch.numel, torch.Tensor.numel, is_method=True)
def numel(a: TensorLike, /) -> int:
    return a._numel


register_method("numel", numel)


@torchsymbol(torch.Tensor.is_complex, is_property=True, id="torch.is_complex")
def is_complex(a: TensorLike, /) -> bool:
    return dtypes.is_complex_dtype(a.dtype)


_torch_to_thunder_function_map[torch.is_complex] = is_complex
_torch_to_thunder_function_map[torch.Tensor.is_complex] = is_complex
register_method("is_complex", is_complex)


@torchsymbol(torch.Tensor.is_cuda, is_property=True, id="torch.is_cuda")
def is_cuda(a: TensorLike, /) -> bool:
    return a.device.devicetype is devices.DeviceType.CUDA


# is nested always returns False for now:
# https://github.com/Lightning-AI/lightning-thunder/issues/93#issuecomment-2030416883
@torchsymbol(torch.Tensor.is_nested, is_property=True, id="torch.is_nested")
def is_nested(a: TensorLike, /) -> bool:
    return False


_torch_dtype_to_old_torch_typestring_map = {
    torch.float32: "FloatTensor",
    torch.float64: "DoubleTensor",
    torch.float16: "HalfTensor",
    torch.bfloat16: "BFloat16Tensor",
    torch.uint8: "ByteTensor",
    torch.int8: "CharTensor",
    torch.int16: "ShortTensor",
    torch.int32: "IntTensor",
    torch.long: "LongTensor",
    torch.bool: "BoolTensor",
}

_old_torch_typestring_to_torch_dtype_map = {v: k for k, v in _torch_dtype_to_old_torch_typestring_map.items()}


def _device_and_dtype_to_old_torch_typestring(device: DeviceLike, dtype: dtypeLike) -> str:
    torch_dtype = to_torch_dtype(dtype)
    dtype_str = _torch_dtype_to_old_torch_typestring_map.get(torch_dtype)
    devicetype_str: str = ""
    if device.devicetype is not devices.DeviceType.CPU:
        devicetype_str = f"{devices.devicetype_string(device.devicetype)}."
    return f"torch.{devicetype_str}{dtype_str}"


def _old_torch_typestring_to_devicetype_and_dtype(typestring: str) -> tuple[DeviceLike, dtypeLike]:

    # Two cases:
    #    - torch.DtypeTensor
    #    - torch.device.DtypeTensor

    _, *dev_and_dtype = typestring.split(".")
    devicetype_str = "cpu"
    dtype_str = ""

    if len(dev_and_dtype) == 1:
        # when devicetype_str is omitted, device type is CPU
        (dtype_str,) = dev_and_dtype
        dtype_str = _old_torch_typestring_to_torch_dtype_map[dtype_str]

    if len(dev_and_dtype) == 2:
        devicetype_str, dtype_str = dev_and_dtype
        dtype_str = _old_torch_typestring_to_torch_dtype_map[dtype_str]

    # Value error
    # expected the string to split into one or two elements
    # and devicetype_str should be either "cpu" or "cuda"
    utils.check(
        devicetype_str in ("cpu", "cuda") and 1 <= len(dev_and_dtype) <= 2,
        lambda: f"type(): unrecognized torch typestring {typestring}",
        exception_type=ValueError,
    )

    return devicetype_str, dtype_str


@torchsymbol(torch.Tensor.type, is_method=True)
def type(a: TensorLike, /, dtype: None | str | dtypeLike = None, non_blocking: bool = False) -> str | TensorLike:
    utils.check(
        not non_blocking,
        lambda: f"type(): `non_blocking==True` is currently not supported.",
        exception_type=NotImplementedError,
    )

    if dtype is None:
        # returns the type of the input tensor in string
        return _device_and_dtype_to_old_torch_typestring(a.device, a.dtype)

    if isinstance(dtype, str):
        devtype, dtype = _old_torch_typestring_to_devicetype_and_dtype(dtype)

        if devtype == a.device.type:
            # This handles two cases:
            # 1. When a tensor is already on a CUDA device, and the device type string is CUDA. In this case the tensor remains on its current device.
            # 2. When a tensor is on a CPU device and the device type string is omitted, the tensor remains on the CPU device.
            dev = a.device
        else:
            dev = to_device(devtype)
    else:
        # dtype is assumed to be torch.dtype (e.g. torch.int32)
        dev = a.device

    return to(a, dev, dtype)


register_method("type", type)

#
# Data movement and transformation operations
#


# NOTE This handles a.float()
#   It avoids using the name "float" to not collide with the builtin
#   "float"
def to_float(a: NumberLike | TensorLike) -> Number | TensorLike:
    return clang.maybe_convert_to_dtype(a, dtypes.float32)


register_method("float", to_float)


# NOTE to's parsing is a little whacky
#   to supports five first positional arguments
#   1) a tensor, in which case device and dtype cannot be specified (although we allow them to be)
#   2) a dtype, in which case device cannot be specified (although we allow this)
#   3) a device, in which case dtype can be specified,
#   4) None, in which case device and dtype come from kwargs (which may also be None, a.to() is valid and just returns)
#       a itself
#   5) device and dtype
def _parse_to_device_and_dtype(
    tensor_dtype_or_device: None | TensorLike | dtypeLike | DeviceLike = None,
    optional_positional_dtype: None | dtypeLike = None,
    /,
    device: None | DeviceLike = None,
    dtype: None | dtypeLike = None,
) -> tuple[devices.Device, dtypes.dtype]:
    # Case 3 and 5 -- device first
    if isinstance(tensor_dtype_or_device, (torch.device, devices.Device, str)):
        utils.check(device is None, lambda: f"to received both a positional and keyword device argument")
        device = to_device(tensor_dtype_or_device)

        if optional_positional_dtype is not None:
            utils.check(dtype is None, lambda: f"to received both a positional and keyword dtype argument")
            dtype = to_dtype(optional_positional_dtype)
        else:
            dtype = to_dtype(dtype)
    # Case 2 -- dtype first
    elif isinstance(tensor_dtype_or_device, (torch.dtype, dtypes.dtype)):
        utils.check(dtype is None, lambda: f"to received both a positional and keyword dtype argument")
        device = to_device(device) if device is not None else None
        dtype = to_dtype(tensor_dtype_or_device)
    # Case 4 -- None first
    elif tensor_dtype_or_device is None:
        device = to_device(device) if device is not None else None
        dtype = to_dtype(dtype)
    # Case 1 -- tensor first
    else:
        # It'd be nice to write torch.Tensor here instead of TensorProxy.
        # See issue "Translate isinstance(a, torch.Tensor) calls so that
        # TensorProxies can pass as torch.Tensors"
        utils.check_type(tensor_dtype_or_device, TensorProxy)
        device_ = tensor_dtype_or_device.device if device is None else to_device(device)
        dtype_ = tensor_dtype_or_device.true_dtype if dtype is None else to_dtype(dtype)
        device, dtype = device_, dtype_

    return device, dtype


def _will_to_return_self(input_device, input_dtype, device, dtype, memory_format, copy):
    return not (
        copy
        or (device is not None and device != input_device)
        or (dtype is not None and dtype != input_dtype)
        or (memory_format in (torch.channels_last, torch.channels_last_3d))
    )


# TODO Model non_blocking (as kwargs)
@torchsymbol(torch.Tensor.to, is_method=True)
def to(
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
    device, dtype = _parse_to_device_and_dtype(
        tensor_dtype_or_device, optional_positional_dtype, device=device, dtype=dtype
    )

    if copy:
        if device is not None:
            device = to_device(device)
            a = prims.device_put(a, device)
        if dtype is not None:
            dtype = to_dtype(dtype)
            a = prims.convert_element_type(a, dtype)
        if memory_format is not None:
            # NOTE not sure if we need to handle torch.preserve_format explicitly
            if memory_format == torch.channels_last:
                a = prims.stride_order(a, (3, 0, 2, 1))
            elif memory_format == torch.channels_last_3d:
                a = prims.stride_order(a, (4, 0, 3, 2, 1))
        return a

    # NOTE copy == False
    # NOTE to() returns the tensor unmodified if the device and dtype requested are the same
    #   (and copy=False)
    # NOTE clang.device_put does nothing when device is None or a.device == device
    a = clang.device_put(a, device)

    if dtype is not None:
        return clang.maybe_convert_to_dtype(a, dtype)

    if memory_format is not None:
        # NOTE not sure if we need to handle torch.preserve_format explicitly
        if memory_format == torch.channels_last:
            a = prims.stride_order(a, (3, 0, 2, 1))
        elif memory_format == torch.channels_last_3d:
            a = prims.stride_order(a, (4, 0, 3, 2, 1))

    return a


@torchsymbol(torch.Tensor.cuda, is_method=True)
def cuda(
    a: TensorLike,
    /,
    device: None | DeviceLike = None,
    non_blocking: bool = False,
    memory_format: None | torch.memory_format = None,
) -> TensorLike:
    # Modeled similar to PyTorch:
    # https://github.com/pytorch/pytorch/blob/e3ac61587aa368c613ef01df1f328a396b64cd5d/tools/autograd/templates/python_variable_methods.cpp#L496-L501
    # If `device` is None, this function defaults `device` to current CUDA device
    # and delegates actual data-movement and layout ordering to `Tensor.to`.

    # NOTE: `Tensor.to` doesn't model `non_blocking` currently.
    utils.check(not non_blocking, lambda: "cuda(): `non_blocking==True` is currently not supported.")

    if device is None:
        # Move tensor to `current` GPU device.
        cuda_idx = torch.cuda.current_device()
        device = devices.Device(devices.DeviceType.CUDA, cuda_idx)
    else:
        device = to_device(device)
        utils.check(
            device.devicetype == devices.DeviceType.CUDA,
            lambda: f"cuda(): Invalid device {device.device_str()}, must be cuda device",
        )

    return to(a, device=device, memory_format=memory_format)


@torchsymbol(torch.Tensor.type_as, is_method=True)
def type_as(a: TensorProxy, b: TensorProxy, /) -> TensorProxy:
    # NOTE This type check is intentional since we're accessing the true_dtype
    #   attribute of the TensorProxy
    # TODO Create a generic Tensor annotation, and support both PyTorch
    #   tensors and TensorProxies being passed to this operation
    utils.check_type(b, TensorProxy)

    return to(a, b.true_dtype, device=b.device)


@torchsymbol(torch.Tensor.long, is_method=True)
def long(a: TensorLike, /, memory_format: torch.memory_format = torch.preserve_format) -> TensorLike:
    return to(a, dtype=dtypes.int64, memory_format=memory_format)


#
# Tensor creation operations
#


@torchsymbol(torch.arange)
def arange(
    start: NumberLike,
    end: None | Number = None,
    step: NumberLike = 1,
    *,
    device: None | DeviceLike = None,
    dtype: None | dtypeLike = None,
) -> TensorLike:
    device = maybe_get_default_device(device)
    device = to_device(device)
    # From torch docs - https://pytorch.org/docs/stable/generated/torch.arange.html
    # If any of start, end, or stop are floating-point, the dtype is inferred to be the default dtype, see get_default_dtype().
    # Otherwise, the dtype is inferred to be torch.int64.
    if dtype is None:  # infer the dtype
        if any(map(lambda x: isinstance(x, float), (start, end, step))):
            dtype = maybe_get_default_dtype(dtype)
        else:
            dtype = torch.int64

    dtype = to_dtype(dtype)

    if end is None:
        end = start
        start = 0
    return clang.arange(start=start, step=step, stop=end, device=device, dtype=dtype)


# Infers dtype from the fill_value and dtype
def _infer_full_dtype(fill_value: NumberLike, dtype: None | dtypeLike) -> dtypeLike:

    # Short-circuits if dtype is explicitly specified
    if dtype is not None:
        return to_dtype(dtype)

    # NOTE dtype is None
    fill_value_dtype = dtypes.numbertype_to_dtype(dtypes.to_dtype(fill_value))

    if dtypes.is_boolean_dtype(fill_value_dtype):
        return dtypes.bool8

    if dtypes.is_nonboolean_integer_dtype(fill_value_dtype):
        return dtypes.int64

    current_default_dtype = get_default_dtype()

    # NOTE When the `fill_value' is a complex dtype, Thunder infers a slightly different dtype than Torch.
    # Torch (2.5.0a0+git8927fc2):
    #     float64 -> complex128
    #     float32, float16, bfloat16 -> complex64
    # (Ref: the torch function: https://github.com/pytorch/pytorch/blob/cd307fb0b1a833f9297d2233653b514ed4aa3163/aten/src/ATen/native/TensorFactories.cpp#L584-L604)
    # Thunder uses `dtypes.corresponding_complex_dtype` (see its implementation for details)
    # The only difference is that when `fill_value_dtype` is float16, Thunder returns complex32 but Torch returns complex64
    if dtypes.is_complex_dtype(fill_value_dtype):
        return dtypes.corresponding_complex_dtype(current_default_dtype)

    # NOTE fill_value_dtype is a non-complex floating-point type
    return to_dtype(current_default_dtype)


@torchsymbol(torch.full)
def full(
    shape: Sequence[int], fill_value: NumberLike, *, device: None | DeviceLike = None, dtype: None | dtypeLike = None
) -> TensorLike:
    device = to_device(maybe_get_default_device(device))
    dtype = _infer_full_dtype(fill_value, dtype)
    return clang.full(shape, fill_value, device=device, dtype=dtype)


@torchsymbol(torch.full_like)
def full_like(
    a: TensorLike, /, fill_value: NumberLike, *, device: None | DeviceLike = None, dtype: None | dtypeLike = None
) -> TensorLike:
    device = to_device(device)
    dtype = to_dtype(dtype)
    return clang.full_like(a, fill_value, device=device, dtype=dtype)


# NOTE ones, unlike full, can accept an integer shape
@torchsymbol(torch.ones)
def ones(*shape: int, device: None | DeviceLike = None, dtype: None | dtypeLike = None) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)
    return full(shape, 1, device=device, dtype=maybe_get_default_dtype(dtype))


@torchsymbol(torch.ones_like)
def ones_like(a: TensorLike, /, *, device: None | DeviceLike = None, dtype: None | dtypeLike = None) -> TensorLike:
    return full_like(a, 1, device=device, dtype=dtype)


@torchsymbol(torch.tensor, is_method=False, id="torch.tensor")
def tensor(
    seq_or_number: Sequence | Number,
    *,
    device: None | DeviceLike = None,
    dtype: None | dtypeLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> TensorLike:
    # TODO: Support torch.Tensor/np.ndarray as input similar to `torch.tensor`
    utils.check(
        isinstance(seq_or_number, (Number, Sequence)),
        lambda: f"Currently only directly constructing tensors with a single number or a Sequence of numbers is supported, but received {n}",
        exception_type=NotImplementedError,
    )
    utils.check(
        not requires_grad, lambda: "requires_grad=True is not yet supported within thunder.jit", NotImplementedError
    )
    utils.check(not pin_memory, lambda: "pin_memory=True is not supported within thunder.jit", NotImplementedError)

    if isinstance(seq_or_number, (Number, NumberProxy)):
        # Infer dtype from value (as `full` will use default dtype if dtype=None).
        if dtype is None:
            dtype = dtypes.numbertype_to_dtype(dtypes.to_dtype(seq_or_number))
        return full((), seq_or_number, dtype=dtype, device=device)

    return clang.tensor_from_sequence(seq_or_number, dtype=dtype, device=device)


# TODO based on uniform_, check if Torch now has a functional uniform
# NOTE the uniform_ documentation suggests the interval is specified using "from" and "to",
#   but from is a reserved keyword in Python
@torchsymbol(is_method=False, id="torch.uniform")
def uniform(
    shape: Sequence[int],
    minval: NumberLike = 0.0,
    maxval: NumberLike = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypeLike,
) -> TensorLike:
    device = to_device(maybe_get_default_device(device))
    dtype = to_dtype(maybe_get_default_dtype(dtype))

    return clang.uniform(shape, minval, maxval, device=device, dtype=dtype)


@torchsymbol(is_method=False, id="torch.uniform_like")
def uniform_like(
    a: TensorLike,
    /,
    minval: NumberLike = 0.0,
    maxval: NumberLike = 1.0,
    *,
    device: None | DeviceLike = None,
    dtype: None | dtypeLike = None,
) -> TensorLike:
    device = to_device(device)
    dtype = to_dtype(dtype)

    return clang.uniform_like(a, minval, maxval, device=device, dtype=dtype)


@torchsymbol(torch.multinomial, is_method=True, id="torch.multinomial")
def multinomial(
    a: TensorLike,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: torch.Generator | None = None,
    out: TensorLike | None = None,
) -> TensorLike:
    utils.check(out is None, lambda: "Non-None out is not supported", NotImplementedError)

    # See issue "randomness: enable PyTorch generators for operations like
    # multinomial"
    utils.check(
        generator is None, lambda: f"multinomial does not yet support specifying a generator", NotImplementedError
    )

    seed = None
    samples = prims.multinomial(a, num_samples, replacement, seed)
    return samples


# TODO Maybe update this to return an offset of how far to advance the seed to acquire new values
# See issue "Maybe return offset from thunder.torch.uniform_philox"
@torchsymbol(is_method=False, id="torch.uniform_philox")
def uniform_philox(
    shape: Sequence[int],
    minval: NumberLike = 0.0,
    maxval: NumberLike = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypeLike,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
) -> TensorLike:
    device = to_device(maybe_get_default_device(device))
    dtype = to_dtype(maybe_get_default_dtype(dtype))

    return clang.uniform_philox(shape, minval, maxval, device=device, dtype=dtype, seed=seed, offset=offset)


@torchsymbol(torch.randn)
def randn(
    *shape,
    generator: None | torch.Generator = None,
    dtype: None | dtypeLike = None,
    device: None | DeviceLike = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    pin_memory: bool = False,
    out: TensorLike = None,
):
    utils.check(
        not requires_grad, lambda: "requires_grad=True is not yet supported within thunder.jit", NotImplementedError
    )
    utils.check(layout == torch.strided, lambda: "Only torch.strided layout is supported", NotImplementedError)
    utils.check(not pin_memory, lambda: "pin_memory=True is not supported within thunder.jit", NotImplementedError)
    # NOTE: Currently, we don't model randomness
    utils.check(generator is None, lambda: "generator is not None which is currently unsupported", NotImplementedError)
    utils.check(out is None, lambda: "out is not None which is currently unsupported", NotImplementedError)

    device = to_device(maybe_get_default_device(device))
    dtype = to_dtype(maybe_get_default_dtype(dtype))
    shape = tuple(utils.extract_shape_from_varargs(shape))
    return prims.randn(shape, device=device, dtype=dtype)


@torchsymbol(torch.randn_like)
def randn_like(
    a,
    /,
    *,
    dtype: None | dtypeLike = None,
    device: None | DeviceLike = None,
    layout: None | torch.layout = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
):
    utils.check(
        not requires_grad, lambda: "requires_grad=True is not supported within thunder.jit", NotImplementedError
    )
    utils.check(
        layout is None or layout == torch.strided, lambda: "Only torch.strided layout is supported", NotImplementedError
    )
    utils.check(
        memory_format == torch.preserve_format,
        lambda: "preserve_format!=torch.preserve_format is not supported within thunder.jit",
        NotImplementedError,
    )

    if dtype is None:
        dtype = a.dtype

    if device is None:
        device = a.device
    return randn(a.shape, dtype=dtype, device=device)


@torchsymbol(torch.bernoulli, is_method=True)
def bernoulli(a: TensorLike, *, generator=None, out=None):
    # NOTE: Currently, we don't model randomness
    utils.check(
        generator is None,
        lambda: "bernoulli: generator is not None which is currently unsupported",
        NotImplementedError,
    )
    utils.check(out is None, lambda: "bernoulli: out is not None which is currently unsupported", NotImplementedError)
    utils.check(dtypes.is_float_dtype(a.dtype), lambda: f"bernoulli only supports floating point dtypes, got {a.dtype}")
    return (uniform_like(a) < a).to(a.dtype)


# NOTE zeros, like ones, and unlike full, can accept an integer shape
@torchsymbol(torch.zeros)
def zeros(*shape: int, device: None | DeviceLike = None, dtype: None | dtypeLike = None) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)
    return full(shape, 0, device=device, dtype=maybe_get_default_dtype(dtype))


@torchsymbol(torch.zeros_like)
def zeros_like(a: TensorLike, /, *, device: DeviceLike | None = None, dtype: dtypeLike | None = None) -> TensorLike:
    return full_like(a, 0, device=device, dtype=dtype)


@torchsymbol(torch.empty)
def empty(
    *size: int,
    device: None | DeviceLike = None,
    dtype: None | dtypeLike = None,
    out: None | TensorLike = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    pin_memory: bool = False,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> TensorLike:
    size = utils.extract_shape_from_varargs(size)

    utils.check(out is None, lambda: "empty(): out is not None which is currently unsupported", NotImplementedError)
    utils.check(layout == torch.strided, lambda: "Only torch.strided layout is supported", NotImplementedError)
    utils.check(
        not requires_grad, lambda: "requires_grad=True is not yet supported within thunder.jit", NotImplementedError
    )
    utils.check(not pin_memory, lambda: "pin_memory=True is not supported within thunder.jit", NotImplementedError)
    utils.check(
        memory_format == torch.contiguous_format,
        lambda: "Only torch.contiguous_format is supported",
        NotImplementedError,
    )

    dtype = to_dtype(maybe_get_default_dtype(dtype))
    device = to_device(maybe_get_default_device(device))

    return clang.empty(size, device=device, dtype=dtype)


#
# Shape operations
#


# TODO Update this to take a *args series of tensors or a sequence of tensors
@torchsymbol(torch.cat)
def cat(tensors: Sequence[TensorLike], dim: int = 0) -> TensorLike:
    return clang.cat(tensors, dim)


@torchsymbol(torch.chunk, is_method=True)
def chunk(a: TensorLike, chunks: int, dim: int = 0) -> Sequence[TensorLike]:
    utils.check(a.ndim > 0, lambda: f"chunk: a ({a.ndim=}) must be at least 1-dimensional")
    utils.check(chunks > 0, lambda: f"chunk: chunks ({chunks=}) must be greater than 0")

    dim = utils.canonicalize_dim(a.ndim, dim)
    a_dim_len = a.shape[dim]

    # a_dim_len == 0?
    # Easy case, return `chunk` number of copies of `a` as slices slice(0, 1) at dim=dim.
    if a_dim_len == 0:
        return tuple(clang.slice_in_dim(a, 0, 1, dim=dim) for _ in range(chunks))

    # chunks == 1?
    # Easy case, return a copy of `a` as a slice(0, a_dim_len) at dim=dim.
    if chunks == 1:
        return (clang.slice_in_dim(a, 0, a_dim_len, dim=dim),)

    # NOTE: in the code below a_dim_len > 0 and chunks > 1.
    # In the output, the first len - 1 tensors
    # will always have shape[dim] = ceil(a.shape[dim] / chunks).
    chunk_len = (a_dim_len + chunks - 1) // chunks
    # Based on `chunk_len` above, the len of the result is either
    # `chunk` or less, and is defined as ceil(a.shape[dim] / chunk_len).
    # So we update `chunks` to this new value below.
    chunks = (a_dim_len + chunk_len - 1) // chunk_len
    chunk_len_last = a_dim_len - (chunks - 1) * chunk_len

    # A generator that defines start and stop for each chunk.
    chunk_start_end_gen = itertools.chain(
        ((chunk_start, chunk_start + chunk_len) for chunk_start in range(0, a_dim_len - chunk_len_last, chunk_len)),
        # Last chunk
        ((a_dim_len - chunk_len_last, a_dim_len),),
    )

    return tuple(clang.slice_in_dim(a, *chunk_data, dim=dim) for chunk_data in chunk_start_end_gen)


@torchsymbol(torch.Tensor.contiguous, is_method=True)
def contiguous(a: TensorLike, /, *, memory_format: torch.memory_format = torch.contiguous_format) -> TensorLike:
    # NOTE PyTorch supports the following memory formats:
    #   - torch.preserve_format
    #   - torch.contiguous_format
    #   - torch.channels_last
    #   - torch.channels_last_3d
    #
    #   torch.channels_last is also known as channels_last_2d, and only applies to 4D tensors (NCHW dims with NHWC strides)
    #   torch.channels_last_3d only applies to 5D tensors (NCDHW dims with NDHWC strides)

    if memory_format is torch.preserve_format:
        # TODO Should this case raise a NotImplementedError? We don't know the format of a
        #   to preserve it
        return a
    elif memory_format is torch.contiguous_format:
        return clang.stride_order(a)
    elif memory_format is torch.channels_last:
        utils.check(a.ndim == 4, lambda: f"Expected a 4D tensor for the channels last memory format")
        return clang.stride_order(a, (3, 0, 2, 1))
    elif memory_format is torch.channels_last_3d:
        utils.check(a.ndim == 5, lambda: f"Expected a 5D tensor for the channels last 3D memory format")
        return clang.stride_order(a, (4, 0, 3, 2, 1))

    utils.check(False, lambda: f"Found unexpected memory_format={memory_format}", exception_type=ValueError)


@torchsymbol(torch.diagonal, is_method=True)
def diagonal(a: TensorLike, /, offset: int = 0, dim1: int = 0, dim2: int = 1) -> TensorLike:
    return clang.diagonal(a, offset, dim1, dim2)


@torchsymbol(torch.Tensor.expand, is_method=True)
def expand(a: TensorLike, /, *shape: int) -> TensorLike:
    return clang.expand(a, *shape)


@torchsymbol(torch.Tensor.expand_as, is_method=True)
def expand_as(a: TensorLike, b: TensorLike, /) -> TensorLike:
    return expand(a, b.size())


@torchsymbol(torch.flatten, is_method=True)
def flatten(a: TensorLike, /, start_dim: int = 0, end_dim: int = -1) -> TensorLike:
    return clang.flatten(a, start_dim, end_dim)


@torchsymbol(torch.flip, is_method=True)
def flip(a: TensorLike, /, *dims: int) -> TensorLike:
    dims = utils.extract_shape_from_varargs(dims)

    # PyTorch supports 0-dim inputs with len(dims) <= 1
    if a.ndim == 0 and isinstance(dims, Sequence) and len(dims) > 0:
        utils.check(
            len(dims) == 1
            and (
                (isinstance(dims[0], (int, IntegerProxy)) and dims[0] in (0, -1))
                or (isinstance(dims[0], NumberProxy) and pyval(dims[0]) in (0, -1))
            ),
            lambda: f"Expected {dims=} to be a sequence of integers in range [-1, 0], and of length 1",
        )
        return clang.flip(a, ())

    return clang.flip(a, dims)


# fake out of place variant
@torchsymbol(id="setitem")
def setitem(inp, idx, val):
    return clang.copy_with_setitem(inp, idx, val)


@torchsymbol(torch.Tensor.__setitem__, id="setitem_", is_method=True, tags=(prims.OpTags.IN_PLACE,))
def setitem_(inp, idx, val):
    prims.copy_(setitem(inp, idx, val), inp)


@torchsymbol(torch.Tensor.__getitem__, id="torch.Tensor.__getitem__", method_name="getitem")
def getitem(a: TensorLike, /, key) -> TensorLike:
    return clang.getitem(a, key)


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


register_method("mT", matrix_transpose)


@torchsymbol(torch.movedim, is_method=True)
def movedim(a: TensorLike, /, source: int | Sequence[int], destination: int | Sequence[int]) -> TensorLike:
    return clang.movedim(a, source, destination)


@torchsymbol(torch.nn.functional.pad)
def pad(a: TensorProxy, /, pad: tuple[int, ...], mode: str | None = "constant", value: NumberLike | None = None):
    utils.check(mode == "constant", lambda: f"Mode arguments other than constant are not supported")
    utils.check(len(pad) % 2 == 0, lambda: f"Padding length must be divisible by 2")
    utils.check(
        len(pad) <= a.ndim * 2,
        lambda: f"Padding length should be less than or equal to two times the input dimension.",
    )

    pad_config = []
    for dim in range(a.ndim * 2 - 1, 0, -2):
        if dim >= len(pad):
            pad_config.append((0, 0, 0))
        else:
            pad_config.append((pad[dim - 1], pad[dim], 0))

    if value is None:
        value = 0
    a_typ = to_dtype(a, true_dtype=True)
    # Note that this can be unsafe. This can happen, for example, if `a` is an
    # integer tensor and `value` is a float. It can also be more subtle, where
    # `a` is a lower-precision float than `value`.
    if a_typ is not to_dtype(value, true_dtype=True):
        warnings.warn("`value` and Tensor input are of different types. This " "may create numeric issues.")
    v2 = clang.maybe_convert_to_dtype(value, a_typ)

    return clang.pad(a, v2, pad_config)


@torchsymbol(torch.permute, is_method=True)
def permute(a: TensorLike, /, *dims: int) -> TensorLike:
    dims = utils.extract_shape_from_varargs(dims)
    return clang.transpose(a, dims)


@torchsymbol(torch.Tensor.repeat, is_method=True)
def repeat(a: TensorLike, /, *repeats: int) -> TensorLike:
    repeats = utils.extract_shape_from_varargs(repeats)
    utils.check_valid_shape(repeats)
    utils.check(a.ndim <= len(repeats), f"Expected {a.ndim=} <= {len(repeats)=}")

    repeats = tuple(repeats)
    new_dims = len(repeats) - a.ndim
    out_shape = repeats[:new_dims] + tuple(repeats[i] * a.shape[i] for i in range(-a.ndim, 0))
    if 0 in out_shape:
        return zeros(*out_shape, device=a.device, dtype=a.dtype)

    a_orig_shape = a.shape
    a = prims.broadcast_in_dim(
        a,
        repeats[:new_dims] + tuple(s for pair in zip(repeats[new_dims:], a_orig_shape) for s in pair),
        tuple(new_dims + offset for offset in range(1, 2 * a.ndim, 2)),
    )
    return reshape(a, out_shape)


@torchsymbol(torch.reshape, is_method=True)
def reshape(a: TensorLike, /, *shape: int) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)

    return clang.reshape(a, shape)


@torchsymbol(torch.unflatten, is_method=True)
def unflatten(a: TensorLike, /, dim: int, sizes=Sequence[int]) -> TensorLike:
    utils.check(
        len(sizes) > 0,
        lambda: f"unflatten() sizes must be non-empty",
        RuntimeError,
    )
    dim = utils.canonicalize_dim(a.ndim, dim)
    return a.view(a.shape[:dim] + tuple(sizes) + a.shape[dim + 1 :])


@torchsymbol(torch.select, is_method=True)
def select(a: TensorLike, /, dim: int, index: int):
    # dim check
    utils.check(
        a.ndim != 0,
        lambda: f"select() cannot be applied to a 0-dim tensor.",
    )
    dim = utils.canonicalize_dim(a.ndim, dim)

    # index check
    dim_length = a.shape[dim]

    wrapped_index = index + dim_length if index < 0 else index
    utils.check(
        (wrapped_index < dim_length and wrapped_index >= 0),
        lambda: f"select(): index {index} out of range for tensor of size {a.shape} at dimension {dim}",
    )

    # `torch.select` returns view with given dimension removed
    # while `slice_in_dim` preserves the sliced dim, hence the `squeeze`
    a_sliced = clang.slice_in_dim(a, wrapped_index, wrapped_index + 1, dim=dim)
    return squeeze(a_sliced, dim)


# TODO consider revising this to just call _split_indices
# Splits a tensor along a split dimension dim into n tensors
# If input is divisible by n then every tensor will have the same length along the split dimension
# If input is not divisible by n, then the first int(input.size(dim) % n) tensors will have length
#   int(input.size(dim) / n) + 1 along the split dimension, and the remaining tensors will have
#   length int(input.size(dim) / n) along the split dimension
def _split_n(a: TensorLike, n: int, dim: int = 0) -> tuple[TensorLike, ...]:
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
def _split_indices(a: TensorLike, indices: int, dim: int = 0) -> tuple[TensorLike, ...]:
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
def split(a: TensorProxy, size_or_sections: int | Sequence[int], /, dim=0) -> TensorProxy | list[TensorProxy]:
    # TODO See note in tensor_split
    if isinstance(size_or_sections, TensorProxy):
        raise NotImplementedError

    dim = utils.canonicalize_dim(a.ndim, dim)

    utils.check_type(
        size_or_sections,
        (int, IntegerProxy, Sequence),
    )

    # TODO: consider revising this to just call _split_indices
    if isinstance(size_or_sections, (int, IntegerProxy)):
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
@torchsymbol(torch.stack)
def stack(tensors: Sequence[TensorLike], /, dim: int = 0) -> TensorLike:
    return clang.stack(tensors, dim)


# See https://pytorch.org/docs/master/generated/torch.squeeze.html
@torchsymbol(torch.squeeze, is_method=True)
def squeeze(a: TensorLike, /, dim: None | int | Sequence[int] = None) -> TensorLike:
    # Converts dim to a tuple of numbers
    dims = dim
    if dim is None:
        dims = []
        for idx, l in enumerate(a.shape):
            if l == 1:
                dims.append(idx)
    elif isinstance(dim, (int, NumberProxy)):
        dims = (dim,)

    # a.shape is being indexed below.
    # We want to make sure that dims is valid.
    dims = utils.canonicalize_dims(a.ndim, dims)

    # Make sure that squeezing a non-1 dim is a no-op
    # and it does not error as {prim/clang}.squeeze would.
    dims = tuple(d for d in dims if a.shape[d] == 1)

    return clang.squeeze(a, dims)


@torchsymbol(torch.t, is_method=True)
def t(a: TensorLike, /) -> TensorLike:
    utils.check(
        a.ndim <= 2,
        lambda: f"t() expects a tensor with <= 2 dimensions, but self is {a.ndim}D",
        RuntimeError,
    )
    return prims.transpose(a, (1, 0)) if a.ndim == 2 else a


@run_once
def warn_ndim_not_2():
    warnings.warn(
        "The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and will throw an error in a future release."
        "Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor."
    )


def reverse_dims_T(a: TensorLike, /) -> TensorLike:
    if a.ndim != 2:
        warn_ndim_not_2()
    return a if a.ndim < 2 else prims.transpose(a, tuple(reversed(range(a.ndim))))


register_method("T", reverse_dims_T)


# TODO Add type annotations
# See https://pytorch.org/docs/master/generated/torch.tensor_split.html
@torchsymbol(torch.tensor_split, is_method=True)
def tensor_split(a: TensorLike, /, indices_or_sections, dim=0):
    # TODO Consider if we even should support this, it could introduce data-dependent control flow
    # NOTE This will also catch number tensors
    if isinstance(indices_or_sections, TensorProxy):
        raise NotImplementedError

    utils.check(
        indices_or_sections,
        (Number, Sequence),
        lambda: f"indices_or_sections={indices_or_sections} should be a Number or a Sequence!",
    )

    # TODO: maybe revise _split_n to a call to _split_indices
    if isinstance(indices_or_sections, (Number, NumberProxy)):
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


@torchsymbol(torch.unbind, is_method=True)
def unbind(a: TensorLike, /, dim: int = 0) -> tuple[TensorLike, ...]:
    utils.check(
        a.ndim > 0,
        lambda: f"Dimension specified as={dim} but tensor has no dimensions.",
    )
    return tuple(s.squeeze(dim) for s in tensor_split(a, a.shape[dim], dim))


@torchsymbol(torch.Tensor.unfold, is_method=True)
def unfold(a: TensorLike, /, dim: int, size: int, step: int) -> TensorLike:
    return clang.unfold(a, dim, size, step)


@torchsymbol(torch.unsqueeze, is_method=True)
def unsqueeze(a: TensorLike, /, dim: int) -> TensorLike:
    return clang.unsqueeze(a, dim)


# TODO Review view functionalization
# TODO Add type annotations
@torchsymbol(torch.Tensor.view, is_method=True)
def view(a: TensorLike, /, *shape) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)
    return reshape(a, shape)


@torchsymbol(torch.Tensor.view_as, is_method=True)
def view_as(a: TensorLike, b: TensorLike, /) -> TensorLike:
    return view(a, b.size())


#
# Elementwise unary operations
#
# TODO Add type annotations


@torchsymbol(torch.abs, is_method=True)
def abs(a: NumberLike | TensorLike, /) -> Number | TensorLike:
    return clang.abs(a)


@torchsymbol(torch.abs_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def abs_(a: NumberLike | TensorLike, /) -> Number | TensorLike:
    return prims.copy_(abs(a), a)


@torchsymbol(torch.acos, is_method=True)
def acos(a: NumberLike | TensorLike, /) -> Number | TensorLike:
    return clang.acos(a)


@torchsymbol(torch.acos_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def acos_(a: TensorLike, /) -> TensorLike:
    return prims.copy_(acos(a), a)


@torchsymbol(torch.acosh, is_method=True)
def acosh(a):
    return clang.acosh(a)


@torchsymbol(torch.acosh_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def acosh_(a):
    return prims.copy_(acosh(a), a)


@torchsymbol(torch.asin, is_method=True)
def asin(a):
    return clang.asin(a)


@torchsymbol(torch.asin_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def asin_(a):
    return prims.copy_(asin(a), a)


@torchsymbol(torch.asinh, is_method=True)
def asinh(a):
    return clang.asinh(a)


@torchsymbol(torch.asinh_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def asinh_(a):
    return prims.copy_(asinh(a), a)


@torchsymbol(torch.atan, is_method=True)
def atan(a):
    return clang.atan(a)


@torchsymbol(torch.atan_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def atan_(a):
    return prims.copy_(atan(a), a)


@torchsymbol(torch.atanh, is_method=True)
def atanh(a):
    return clang.atanh(a)


@torchsymbol(torch.atanh_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def atanh_(a):
    return prims.copy_(atanh(a), a)


@torchsymbol(torch.bitwise_not, is_method=True)
def bitwise_not(a):
    return clang.bitwise_not(a)


@torchsymbol(torch.Tensor.bitwise_not_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def bitwise_not_(a):
    return prims.copy_(bitwise_not(a), a)


@torchsymbol(torch.ceil, is_method=True)
def ceil(a):
    return clang.ceil(a)


@torchsymbol(torch.ceil_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def ceil_(a):
    return prims.copy_(ceil(a), a)


@torchsymbol(torch.cos, is_method=True)
def cos(a):
    return clang.cos(a)


@torchsymbol(torch.cos_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def cos_(a):
    return prims.copy_(cos(a), a)


@torchsymbol(torch.cosh, is_method=True)
def cosh(a):
    return clang.cosh(a)


@torchsymbol(torch.cosh_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def cosh_(a):
    return prims.copy_(cosh(a), a)


@torchsymbol(torch.digamma, torch.special.digamma, is_method=True)
def digamma(a):
    return clang.digamma(a)


@torchsymbol(torch.Tensor.digamma_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def digamma_(a):
    return prims.copy_(digamma(a), a)


@torchsymbol(torch.erf, is_method=True)
def erf(a):
    return clang.erf(a)


@torchsymbol(torch.erf_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def erf_(a):
    return prims.copy_(erf(a), a)


@torchsymbol(torch.erfc, is_method=True)
def erfc(a):
    return clang.erfc(a)


@torchsymbol(torch.erfc_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def erfc_(a):
    return prims.copy_(erfc(a), a)


@torchsymbol(torch.erfinv, is_method=True)
def erfinv(a):
    return clang.erfinv(a)


@torchsymbol(torch.Tensor.erfinv_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def erfinv_(a):
    return prims.copy_(erfinv(a), a)


@torchsymbol(torch.exp, is_method=True)
def exp(a):
    return clang.exp(a)


@torchsymbol(torch.exp_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def exp_(a):
    return prims.copy_(exp(a), a)


@torchsymbol(torch.exp2, is_method=True)
def exp2(a):
    return clang.exp2(a)


@torchsymbol(torch.exp2_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def exp2_(a):
    return prims.copy_(exp2(a), a)


# fake out of place variant
@torchsymbol(id="exponential")
def exponential(a: Tensor, rate: float = 1, *, generator: None | torch.Generator = None) -> Tensor:
    utils.check(
        generator is None,
        lambda: "exponential: generator is not None which is currently unsupported",
    )
    utils.check(
        thunder.dtypes.is_float_dtype(a.dtype),
        lambda: f"Exponential distribution is a continuous probability distribution. \
        dtype must be a floating point but you specified {a.dtype}",
    )
    utils.check(
        rate > 0.0,
        lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
    )
    uniform_val = uniform_like(a)

    # copying numerics of transformation::exponential see comment:
    # curand_uniform has (0,1] bounds. log(1) is 0 and exponential excludes 0.
    # we need log to be not 0, and not underflow when converted to half
    # fast __logf approximation can underflow, so set log to -epsilon/2 for 1 or close to 1 args
    epsilon = torch.finfo(thunder.dtypes.to_torch_dtype(a.dtype)).eps / 2
    condition = uniform_val >= (1.0 - epsilon)
    log_uniform = where(condition, -epsilon, log(uniform_val))

    return (-1 / rate) * log_uniform


@torchsymbol(torch.Tensor.exponential_, id="exponential_", is_method=True, tags=(prims.OpTags.IN_PLACE,))
def exponential_(a: Tensor, rate: float = 1, *, generator: None | torch.Generator = None) -> Tensor:
    return prims.copy_(exponential(a, rate=rate, generator=generator), a)


@torchsymbol(torch.expm1, is_method=True)
def expm1(a):
    return clang.expm1(a)


@torchsymbol(torch.expm1_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def expm1_(a):
    return prims.copy_(expm1(a), a)


@torchsymbol(torch.floor, is_method=True)
def floor(a):
    return clang.floor(a)


@torchsymbol(torch.floor_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def floor_(a):
    return prims.copy_(floor(a), a)


@torchsymbol(torch.isfinite, is_method=True)
def isfinite(a):
    return clang.isfinite(a)


@torchsymbol(torch.lgamma, is_method=True)
def lgamma(a):
    return clang.lgamma(a)


@torchsymbol(torch.Tensor.lgamma_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def lgamma_(a):
    return prims.copy_(lgamma(a), a)


@torchsymbol(torch.log, is_method=True)
def log(a):
    return clang.log(a)


@torchsymbol(torch.log_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def log_(a):
    return prims.copy_(log(a), a)


@torchsymbol(torch.log10, is_method=True)
def log10(a):
    return clang.log10(a)


@torchsymbol(torch.log10_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def log10_(a):
    return prims.copy_(log10(a), a)


@torchsymbol(torch.log1p, is_method=True)
def log1p(a):
    return clang.log1p(a)


@torchsymbol(torch.log1p_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def log1p_(a):
    return prims.copy_(log1p(a), a)


@torchsymbol(torch.log2, is_method=True)
def log2(a):
    return clang.log2(a)


@torchsymbol(torch.log2_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def log2_(a):
    return prims.copy_(log2(a), a)


# TODO Move to special
# @torchsymbol(torch.ndtri, is_method=True)
# def ndtri(a):
#     return clang.ndtri(a)


@torchsymbol(torch.neg, is_method=True)
def neg(a):
    return clang.neg(a)


@torchsymbol(torch.neg_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def neg_(a):
    return prims.copy_(neg(a), a)


@torchsymbol(torch.reciprocal, is_method=True)
def reciprocal(a):
    return clang.reciprocal(a)


@torchsymbol(torch.reciprocal_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def reciprocal_(a):
    return prims.copy_(reciprocal(a), a)


@torchsymbol(torch.round, is_method=True)
def round(a):
    return clang.round(a)


@torchsymbol(torch.round_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def round_(a):
    return prims.copy_(round(a), a)


@torchsymbol(torch.rsqrt, is_method=True)
def rsqrt(a):
    return clang.rsqrt(a)


@torchsymbol(torch.rsqrt_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def rsqrt_(a):
    return prims.copy_(rsqrt(a), a)


# TODO Complain about complex numbers like PyTorch does?
# TODO Add sgn
@torchsymbol(torch.sign, is_method=True)
def sign(a):
    return clang.sign(a)


@torchsymbol(torch.Tensor.sign_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def sign_(a):
    return prims.copy_(sign(a), a)


@torchsymbol(torch.signbit, is_method=True)
def signbit(a):
    return clang.signbit(a)


@torchsymbol(torch.sin, is_method=True)
def sin(a):
    return clang.sin(a)


@torchsymbol(torch.sin_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def sin_(a):
    return prims.copy_(sin(a), a)


@torchsymbol(torch.sinh, is_method=True)
def sinh(a):
    return clang.sinh(a)


@torchsymbol(torch.sinh_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def sinh_(a):
    return prims.copy_(sinh(a), a)


@torchsymbol(torch.sqrt, is_method=True)
def sqrt(a):
    return clang.sqrt(a)


@torchsymbol(torch.sqrt_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def sqrt_(a):
    return prims.copy_(sqrt(a), a)


@torchsymbol(torch.tan, is_method=True)
def tan(a):
    return clang.tan(a)


@torchsymbol(torch.tan_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def tan_(a):
    return prims.copy_(tan(a), a)


@torchsymbol(torch.tanh, is_method=True)
def tanh(a):
    return clang.tanh(a)


@torchsymbol(torch.tanh_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def tanh_(a):
    return prims.copy_(tanh(a), a)


@torchsymbol(torch.trunc, is_method=True)
def trunc(a):
    return clang.trunc(a)


@torchsymbol(torch.trunc_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def trunc_(a):
    return prims.copy_(trunc(a), a)


@torchsymbol(torch.real, is_method=False)
def real(a):
    return clang.real(a)


#
# nn.functional elementwise unary
#
# TODO Move these to torch.nn.functional


@torchsymbol(torch.celu, torch.nn.functional.celu, id="torch.celu", is_method=True)
def celu(a: TensorLike, /, alpha: float = 1.0, inplace: bool = False) -> TensorLike:
    negative_domain_value = alpha * expm1(a / alpha)
    out = where(a > 0, a, negative_domain_value)
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[celu] = celu, 2


@torchsymbol(torch.nn.functional.elu, is_method=False)
def elu(a: TensorProxy, /, alpha: float = 1.0, inplace: bool = False) -> TensorLike:
    negative_domain_value = alpha * expm1(a)
    out = where(a > 0, a, negative_domain_value)
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[elu] = elu, 2


@torchsymbol(torch.nn.functional.gelu, is_method=False)
def gelu(a: TensorProxy, /, *, approximate: str = "none") -> TensorLike:
    if approximate == "none":
        # gelu(a) = a * Phi(a), where Phi is the cdf for the Normal Gaussian.
        # We use the error function to compute Phi.
        phi_a = 0.5 + 0.5 * erf(a / (math.sqrt(2)))
        return a * phi_a
    elif approximate == "tanh":
        a_pow_3 = a * a * a
        return 0.5 * a * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * a_pow_3)))
    else:
        raise ValueError(f"gelu does not support the approximate={approximate} argument")


@torchsymbol(torch.nn.functional.leaky_relu, is_method=False)
def leaky_relu(a: TensorProxy, /, negative_slope: float = 0.01, inplace: bool = False) -> TensorLike:
    out = where(a > 0, a, a * negative_slope)
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[leaky_relu] = leaky_relu, 2


@torchsymbol(torch.nn.functional.logsigmoid, is_method=False)
def logsigmoid(a: TensorProxy, /) -> TensorLike:
    return where(a > 0, -log1p(exp(-a)), a - log1p(exp(a)))


@torchsymbol("log_sigmoid_backward", id="log_sigmoid_backward")
def log_sigmoid_backward(g: TensorProxy, a: TensorProxy, buffer: TensorProxy) -> TensorLike:
    # buffer is used by PyTorch in cpu-based calculations.  See
    # https://github.com/pytorch/pytorch/blob/7667235a23e2ffca4d32e6e16aa60a683418e159/torch/_decomp/decompositions.py#L332
    # This is addressed in the custom grad fn thunder.core.transforms._log_sigmoid_grad.
    return g * where(a > 0, exp(-a) / (1 + exp(-a)), 1 - exp(a) / (1 + exp(a)))


# TODO Should this use clamp? -- Would that propagate NaNs properly?
@torchsymbol(torch.relu, torch.nn.functional.relu, id="torch.relu", is_method=True)
def relu(a: TensorLike, /, inplace: bool = False) -> TensorLike:

    out = where(a > 0, a, 0)
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[relu] = relu, 1


@torchsymbol(torch.relu_, torch.nn.functional.relu_, id="torch.relu_", is_method=True)
def relu_(
    a: TensorLike,
    /,
) -> TensorLike:
    return prims.copy_(relu(a, False), a)


# The default value of `inplace` is False, so no need to tweak args/kwargs
_inplace_to_out_of_place[relu_] = relu, -1


# id=torch.relu because we ignore inplace argument in torch.nn.functional.relu
@torchsymbol(torch.nn.functional.relu6, id="torch.relu6", is_method=False)
def relu6(a: TensorProxy, /, inplace: bool = False) -> TensorLike:
    out = clamp(a, 0, 6)
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[relu6] = relu6, 1


@torchsymbol(torch.nn.functional.hardshrink, is_method=False)
def hardshrink(a: TensorProxy, /, lambd: float = 0.5) -> TensorLike:
    utils.check(
        not dtypes.is_complex_dtype(a.dtype),
        lambda: f"hardshrink not implemented for '{a.dtype}'",
    )
    return where(abs(a) <= lambd, 0, a)


@torchsymbol(torch.nn.functional.hardswish, id="torch.hardswish", is_method=False)
def hardswish(a: TensorProxy, /, inplace: bool = False) -> TensorLike:
    utils.check(
        dtypes.is_float_dtype(a.dtype),
        lambda: f"hardswish only supports floating point dtypes, got {a.dtype}",
        exception_type=ValueError,
    )
    out = a * relu6(a + 3) / 6
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[hardswish] = hardswish, 1


# id=torch.selu because we ignore inplace argument in torch.nn.functional.selu
@torchsymbol(torch.selu, torch.nn.functional.selu, id="torch.selu", is_method=False)
def selu(a: TensorProxy, /, inplace: bool = False) -> TensorLike:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    rhs = alpha * expm1(a)

    out = scale * where(a > 0, a, rhs)
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[selu] = selu, 1


@torchsymbol(torch.nn.functional.silu)
def silu(a: TensorLike, /, inplace: bool = False) -> TensorLike:
    out = clang.silu(a)
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[silu] = silu, 1


@torchsymbol(torch.nn.functional.tanhshrink)
def tanhshrink(a: TensorLike, /) -> TensorLike:
    return a - tanh(a)


_inplace_to_out_of_place[tanhshrink] = tanhshrink, -1

#
# Elementwise binary operations
#


@torchsymbol(torch.add, is_method=True)
def add(
    a: NumberLike | TensorLike, b: NumberLike | TensorLike, /, *, alpha: Number | TensorLike = 1
) -> Number | TensorLike:
    if isinstance(alpha, TensorProxy) or alpha != 1:
        b = b * alpha

    return clang.add(a, b)


@torchsymbol(torch.Tensor.add_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def add_(
    a: TensorLike,
    b: NumberLike | TensorLike,
    /,
    *,
    alpha: Number | TensorLike = 1,
) -> TensorLike:
    return prims.copy_(add(a, b, alpha=alpha), a)


@torchsymbol(torch.atan2, is_method=True)
def atan2(a, b, /):
    return clang.atan2(a, b)


@torchsymbol(torch.Tensor.atan2_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def atan2_(a, b, /):
    return prims.copy_(atan2(a, b), a)


@torchsymbol(torch.bitwise_and, is_method=True)
def bitwise_and(a, b, /):
    return clang.bitwise_and(a, b)


@torchsymbol(torch.Tensor.bitwise_and_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def bitwise_and_(a, b, /):
    return prims.copy_(bitwise_and(a, b), a)


@torchsymbol(torch.bitwise_or, is_method=True)
def bitwise_or(a, b, /):
    return clang.bitwise_or(a, b)


@torchsymbol(torch.Tensor.bitwise_or_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def bitwise_or_(a, b, /):
    return prims.copy_(bitwise_or(a, b), a)


@torchsymbol(torch.bitwise_xor, is_method=True)
def bitwise_xor(a, b, /):
    return clang.bitwise_xor(a, b)


@torchsymbol(torch.Tensor.bitwise_xor_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def bitwise_xor_(a, b, /):
    return prims.copy_(bitwise_xor(a, b), a)


@torchsymbol(torch.copysign, is_method=True)
def copysign(a, b, /):
    return clang.copysign(a, b)


@torchsymbol(torch.Tensor.copysign_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def copysign_(a, b, /):
    return prims.copy_(copysign(a, b), a)


@torchsymbol(torch.Tensor.copy_, is_method=True)  # , tags=(prims.OpTags.IN_PLACE,))
def copy_(a, b, /):
    return prims.copy_(b, a)


# TODO Implement div
@torchsymbol(torch.div, is_method=True)
def div(
    a: Number | TensorLike,
    b: Number | TensorLike,
    /,
    *,
    rounding_mode: None | str = None,
    out: None | TensorLike = None,
) -> Number | TensorLike:
    utils.check(out is None, lambda: "out is not None which is currently unsupported", NotImplementedError)

    if rounding_mode is None:
        return true_divide(a, b)
    elif rounding_mode == "trunc":
        return clang.trunc_divide(a, b)
    elif rounding_mode == "floor":
        return floor_divide(a, b)
    else:
        raise ValueError(f"div does not support the rounding_mode={rounding_mode} argument")


@torchsymbol(torch.Tensor.div_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def div_(
    a: TensorLike,
    b: Number | TensorLike,
    /,
    *,
    rounding_mode: None | str = None,
) -> TensorLike:
    return prims.copy_(div(a, b), a)


@torchsymbol(torch.eq, is_method=True)
def eq(a, b, /):
    return clang.eq(a, b)


@torchsymbol(torch.Tensor.eq_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def eq_(a, b, /):
    return prims.copy_(eq(a, b), a)


@torchsymbol(torch.floor_divide, is_method=True)
def floor_divide(a, b, /):
    return clang.floor_divide(a, b)


@torchsymbol(torch.Tensor.floor_divide_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def floor_divide_(a, b, /):
    return prims.copy_(floor_divide(a, b), a)


@torchsymbol(torch.fmod, is_method=True)
def fmod(a, b, /):
    return clang.fmod(a, b)


@torchsymbol(torch.Tensor.fmod_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def fmod_(a, b, /):
    return prims.copy_(fmod(a, b), a)


@torchsymbol(torch.ge, is_method=True)
def ge(a, b, /):
    return clang.ge(a, b)


@torchsymbol(torch.Tensor.ge_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def ge_(a, b, /):
    return prims.copy_(ge(a, b), a)


@torchsymbol(torch.gt, is_method=True)
def gt(a, b, /):
    return clang.gt(a, b)


@torchsymbol(torch.Tensor.gt_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def gt_(a, b, /):
    return prims.copy_(gt(a, b), a)


@torchsymbol(torch.logical_and, is_method=True)
def logical_and(a, b, /):
    return clang.logical_and(a, b)


@torchsymbol(torch.Tensor.logical_and_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def logical_and_(a, b, /):
    return prims.copy_(logical_and(a, b), a)


@torchsymbol(torch.logical_not, is_method=True)
def logical_not(a: TensorLike, /) -> TensorLike:
    return clang.logical_not(a)


@torchsymbol(torch.Tensor.logical_not_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def logical_not_(a: TensorLike, /) -> TensorLike:
    return prims.copy_(logical_not(a), a)


@torchsymbol(torch.le, is_method=True)
def le(a, b, /):
    return clang.le(a, b)


@torchsymbol(torch.Tensor.le_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def le_(a, b, /):
    return prims.copy_(le(a, b), a)


@torchsymbol(torch.lt, is_method=True)
def lt(a, b, /):
    return clang.lt(a, b)


@torchsymbol(torch.Tensor.lt_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def lt_(a, b, /):
    return prims.copy_(lt(a, b), a)


@torchsymbol(torch.maximum, is_method=True)
def maximum(a: TensorProxy, b: TensorProxy) -> TensorProxy:
    return clang.maximum(a, b)


@torchsymbol(torch.minimum, is_method=True)
def minimum(a: TensorProxy, b: TensorProxy) -> TensorProxy:
    return clang.minimum(a, b)


# NOTE This is just an alias for proxies to find operation defined for the modulus
#   operator
# TODO Review this alias
def mod(a, b):
    return clang.mod(a, b)


def mod_(a, b):
    return prims.copy_(mod(a, b), a)


@torchsymbol(torch.mul, is_method=True)
def mul(a, b, /):
    return clang.mul(a, b)


@torchsymbol(torch.Tensor.mul_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def mul_(a, b, /):
    return prims.copy_(mul(a, b), a)


@torchsymbol(torch.ne, is_method=True)
def ne(a, b, /):
    return clang.ne(a, b)


@torchsymbol(torch.Tensor.ne_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def ne_(a, b, /):
    return prims.copy_(ne(a, b), a)


@torchsymbol(torch.nextafter, is_method=True)
def nextafter(a, b, /):
    return clang.nextafter(a, b)


@torchsymbol(torch.Tensor.nextafter_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def nextafter_(a, b, /):
    return prims.copy_(nextafter(a, b), a)


# TODO Extend to tensor x tensor
@torchsymbol(torch.polygamma, torch.special.polygamma, is_method=True)
def polygamma(n: int, a: TensorLike, /) -> TensorLike:
    utils.check(
        isinstance(n, (int, NumberProxy)), lambda: f"polygamma(n, a) expects the first argument to be an integer."
    )
    utils.check(n >= 0, lambda: f"polygamma(n, a) does not support negative {n=}.")

    # NOTE Use digamma for n == 0 case; otherwise zeta(1, a) returns math.inf
    if n == 0:
        return digamma(a)

    sign = 1 if (n % 2) == 1 else -1
    # Compute in log-space for numerical stability
    factorial_mul_zeta = exp(lgamma(n + 1.0) + log(zeta(n + 1.0, a)))
    return sign * factorial_mul_zeta


@torchsymbol(torch.Tensor.polygamma_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def polygamma_(n: int, a: TensorLike, /) -> TensorLike:
    return prims.copy_(polygamma(n, a), a)


@torchsymbol(torch.pow, is_method=True)
def pow(a, b, /):
    return clang.pow(a, b)


@torchsymbol(torch.Tensor.pow_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def pow_(a, b, /):
    return prims.copy_(pow(a, b), a)


@torchsymbol(torch.remainder, is_method=True)
def remainder(a, b, /):
    return clang.remainder(a, b)


@torchsymbol(torch.Tensor.remainder_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def remainder_(a, b, /):
    return prims.copy_(remainder(a, b), a)


@torchsymbol(torch.sub, is_method=True)
def sub(a, b, /, *, alpha: NumberLike | TensorLike = 1):
    if isinstance(alpha, TensorProxy) or alpha != 1:
        b = b * alpha

    return clang.sub(a, b)


@torchsymbol(torch.Tensor.sub_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def sub_(a, b, /, *, alpha: NumberLike | TensorLike = 1):
    return prims.copy_(sub(a, b, alpha=alpha), a)


@torchsymbol(torch.true_divide, is_method=True)
def true_divide(a: NumberLike | TensorLike, b: NumberLike | TensorLike, /) -> Number | TensorLike:
    return clang.true_divide(a, b)


@torchsymbol(torch.Tensor.true_divide_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def true_divide_(a: TensorLike, b: NumberLike | TensorLike, /) -> TensorLike:
    return prims.copy_(true_divide(a, b))


@torchsymbol(torch.special.zeta)
def zeta(a, b, /):
    return clang.zeta(a, b)


#
# Elementwise Ternary operations
#


# For calculate op1(a, op2(value, op2(b, c))) by promoting all input tensors at once
# NOTE use this explicit type promotion because a direct combination of add/mul will have a redundant cast,
# which may lead to accuracy problems.
# TODO remove after issue "Redundant cast removal could be performed through metadata-only
# operations, like broadcasting" is resolved
def addcmul_addcdiv_helper(
    a, b, c, op1, op2, *, value=None, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
):
    inputs = [a, b, c]
    computation_dtype, result_dtype = utils.elementwise_type_promotion(*inputs, type_promotion_kind=type_promotion_kind)
    a, b, c = map(partial(to, dtype=computation_dtype), inputs)

    d = op2(b, c)
    if value is not None:
        d = value * d
    result = op1(a, d)
    return to(result, result_dtype)


@torchsymbol(torch.addcmul, is_method=True)
def addcmul(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> TensorLike:
    return addcmul_addcdiv_helper(a, b, c, add, mul, value=value)


@torchsymbol(torch.Tensor.addcmul_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def addcmul_(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> TensorLike:
    return prims.copy_(addcmul(a, b, c, value=value), a)


@torchsymbol(torch.addcdiv, is_method=True)
def addcdiv(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> TensorLike:
    return addcmul_addcdiv_helper(a, b, c, add, true_divide, value=value)


@torchsymbol(torch.Tensor.addcdiv_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def addcdiv_(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> TensorLike:
    return prims.copy_(addcdiv(a, b, c, value=value), a)


@torchsymbol(torch.lerp, is_method=True)
def lerp(start: TensorLike, end: TensorLike, weight: Number | TensorLike) -> TensorLike:
    return clang.lerp(start, end, weight)


@torchsymbol(torch.Tensor.lerp_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def lerp_(start: TensorLike, end: TensorLike, weight: Number | TensorLike) -> TensorLike:
    return prims.copy_(lerp(start, end, weight), start)


#
# Conditional operations and masking operations
#


@torchsymbol(torch.clamp, is_method=True)
def clamp(
    a: TensorLike,
    /,
    min: None | Number | TensorLike = None,
    max: None | Number | TensorLike = None,
) -> TensorLike:
    utils.check(
        min is not None or max is not None,
        lambda: f"clamp: At least one of 'min' or 'max' must not be None",
        ValueError,
    )
    input_types = [to_dtype(x) for x in [a, min, max] if x is not None]
    # Bool and complex are not supported
    utils.check(
        not all(dtypes.is_boolean_dtype(input_type) for input_type in input_types),
        lambda: f"clamp is not supported for boolean type",
    )
    utils.check(
        not any(utils.is_complex_dtype(input_type) for input_type in input_types),
        lambda: f"clamp is not supported for complex types",
    )

    # torch.clamp outputs nan when one of a, min, max is nan
    # when min is greater than max, outputs max
    if min is not None:
        # nan in min is handled by keeping min's nan when not a>min
        a = where(a != a, a, where(a > min, a, min))

    if max is not None:
        a = where(a != a, a, where(a < max, a, max))

    return a


@torchsymbol(torch.clamp_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def clamp_(
    a: TensorLike, /, min: None | Number | TensorLike = None, max: None | Number | TensorLike = None
) -> TensorLike:
    return prims.copy_(clamp(a, min, max), a)


def _mask_tensor(a, mask, fill_value):
    utils.check(
        dtypes.is_boolean_dtype(mask.dtype), lambda: f"_mask_tensor: mask ({mask.dtype=}) must have a boolean dtype"
    )

    if dtypes.is_boolean_dtype(a.dtype):
        return a & mask

    return where(mask, a, fill_value)


# NOTE masked_fill is a strange wrapper around where, it probably exists only because of PyTorch's inplace pattern
# NOTE PyTorch's masked fill requires value be a number or number tensor
# NOTE PyTorch's masked fill is only defined as a tensor method that implicitly takes a as the first argument
# NOTE PyTorch's masked_fill_ requires the dtype of a not change, so it checks that
#   value can be safely cast to a (for numbers, it checks that the actual number value can safely be cast)
# NOTE We have chosen not to emulate PyTorch's odd type promotion behavior for this operation
@torchsymbol(torch.masked_fill, is_method=True)
def masked_fill(a: TensorLike, /, mask: TensorLike, value: NumberLike | TensorLike) -> TensorLike:
    a_dtype = a.dtype
    value_dtype = to_dtype(value, true_dtype=True)
    if utils._elementwise_type_promotion(a_dtype, value_dtype) == value_dtype:
        from thunder.core.dtypes import dtype_to_numbertype

        value = dtype_to_numbertype(a_dtype)(value)
    return where(mask, value, a)


@torchsymbol(torch.Tensor.masked_fill_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def masked_fill_(a: TensorLike, /, mask: TensorLike, value: NumberLike | TensorLike) -> TensorLike:
    return prims.copy_(masked_fill(a, mask, value), a)


# NOTE The key to understanding tril is that it generates a mask
#   which (by default) masks elements of a matrix (or batch of matrices)
#   s.t. elements whose row number is greater than or equal to its column number
#   are preserved (and other numbers are set to zero).
#   When diagonal is specified, the mask computation changes so that
#   elements with rownum + diagonal >= colnum are preserved.
@torchsymbol(torch.tril, is_method=True)
def tril(a: TensorLike, /, diagonal: int = 0, *, fill_value: None | Number = None) -> TensorLike:
    utils.check(a.ndim >= 2, lambda: f"tril: a ({a.ndim=}) must have at least two dimensions")

    nrows, ncols = a.shape[-2:]
    row_numbers = arange(nrows, device=a.device).unsqueeze(-1)
    col_numbers = arange(ncols, device=a.device).unsqueeze(-2)

    mask = (row_numbers + diagonal) >= col_numbers

    if fill_value is None:
        fill_value = 0

    return _mask_tensor(a, mask, fill_value)


@torchsymbol(torch.Tensor.tril_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def tril_(a: TensorLike, /, diagonal: int = 0, *, fill_value: None | Number = None) -> TensorLike:
    return prims.copy_(tril(a, diagonal, fill_value=fill_value), a)


@torchsymbol(torch.where, is_method=True)
def where(
    pred: TensorLike,
    a: None | Number | TensorLike = None,
    b: None | Number | TensorLike = None,
    /,
) -> TensorLike:
    utils.check(
        isinstance(a, (Number, NumberProxy, TensorProxy)) and isinstance(b, (Number, NumberProxy, TensorProxy)),
        lambda: f"torch.where() does not support only specifying a condition",
        exception_type=NotImplementedError,
    )
    return clang.where(pred, a, b)


@torchsymbol(torch.nan_to_num, is_method=True)
def nan_to_num(
    a: TensorLike,
    nan: None | Number = 0.0,
    posinf: None | Number = None,
    neginf: None | Number = None,
    /,
    out: None | TensorLike = None,
) -> TensorLike:
    """Replaces NaN, positive infinity, and negative infinity values in input tensor with values specified by nan, posinf, and neginf.
        When nan, posinf, and neginf values are greater than the a.dtype's max value, it is replaced with float("inf").
        If they are smaller than the a.dtype's min value, it is replaced with -float("inf").
        Otherwise, they are replaced with the exact values specified by nan, posinf, and neginf.

    Args:
        a (Tensor): input tensor
        nan (Number): value to replace NaNs with. Default is zero
        posinf (Number): value to replace positive infinity. If None, positive infinity values are replaced with the greatest finite value represented by a's type
        neginf (Number): value to replace negative infinity. If None, negative infinity values are replaced with the lowest finite value represented by a's type
        out (Tensor): output tensor which is not supported yet

    Returns:
        result (Tensor): tensor with replaced values

    Examples:
        >>> a = torch.tensor((float("nan"), float("inf"), -float("inf"))) # a.dtype is torch.float32
        >>> nan = torch.finfo(torch.float64).max
        >>> result = torch.nan_to_num(a, nan=nan, posinf=1, neginf=0)
        >>> result
        tensor([inf, 1., 0.])
    """

    utils.check(out is None, lambda: "out is not None which is currently unsupported", NotImplementedError)

    if dtypes.is_boolean_dtype(a.dtype):
        # NOTE PyTorch returns a.clone()
        return a | a

    if dtypes.is_integer_dtype(a.dtype):
        # NOTE PyTorch returns a.clone()
        return a - 0

    a_dtype_max = torch.finfo(to_torch_dtype(a.dtype)).max
    a_dtype_min = torch.finfo(to_torch_dtype(a.dtype)).min
    inf = float("inf")

    def convert(x, if_none):
        if x is None:
            return if_none
        if x > a_dtype_max:
            return inf
        if x < a_dtype_min:
            return -inf
        return x

    nan = convert(nan, 0)
    posinf = convert(posinf, a_dtype_max)
    neginf = convert(neginf, a_dtype_min)

    result = where(a != a, nan, a)
    result = where(a == -inf, neginf, result)
    result = where(a == inf, posinf, result)
    return result


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


def _reduction_dims(shape, dims: Sequence | None) -> tuple[int, ...]:
    if isinstance(dims, (int, NumberProxy)):
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
    dtype: None | torch.dtype = None,  # should be specified for ops that support it
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
) -> TensorProxy:
    # TODO: check that a is the correct type?

    # reduces over all dimensions if dim=() is passed
    if dims == () or dims == []:
        dims = None
    if isinstance(dims, (int, IntegerProxy)):
        dims = (dims,)

    utils.check(
        a.ndim <= 64,
        lambda: f"Received a tensor with {a.ndim} dimensions, but only tensors with up to 64 dims are supported!",
    )

    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, (int, IntegerProxy))

    if isinstance(dims, (int, IntegerProxy)):
        dims = (dims,)

    dims = _reduction_dims(a.shape, dims)

    if not has_identity:
        valid_shape = (a.ndim == 0) or all(a.shape[i] for i in dims)
        utils.check(
            valid_shape,
            lambda: "Can't reduce over a zero-size dimension when computing a reduction without an identity value.",
        )

    computation_dtype, result_dtype = _reduction_dtypes(a, output_dtype_kind, dtype)

    a = to(a, computation_dtype)
    result = prim(a, dims)

    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = tree_map(lambda x: prims.broadcast_in_dim(x, output_shape, broadcast_dims), result)

    if result_dtype is not None:
        result = tree_map(lambda x: to(x, result_dtype), result)

    return result


@torchsymbol(torch.all, is_method=True, method_name="all", id="torch.all")
def all_tensor(
    a: TensorLike, /, dim: None | int | Sequence[int] = None, keepdim: bool = False, *, out: None | TensorLike = None
) -> TensorLike:
    # named as all_tensor to avoid confusion with python's built-in all function
    utils.check(out is None, lambda: "out is not None which is currently unsupported", NotImplementedError)
    result = logical_not(any_tensor(logical_not(a), dim=dim, keepdim=keepdim))

    # Pytorch's torch.all matches the behavior of NumPy in returning output of dtype bool for all supported dtypes except uint8.
    # For uint8 the dtype of output is uint8 iteself (https://pytorch.org/docs/stable/generated/torch.all.html)
    if a.dtype is dtypes.uint8:
        result = to(result, dtype=dtypes.uint8)
    return result


@torchsymbol(torch.any, is_method=True, method_name="any", id="torch.any")
def any_tensor(a: TensorLike, /, dim: None | int | Sequence[int] = None, keepdim: bool = False) -> TensorLike:
    # named as any_tensor to avoid confusion with python's built-in any function
    a_ = clang.maybe_convert_to_dtype(a, dtypes.bool8)
    if isinstance(dim, Sequence) and len(dim) == 0:
        # PyTorch returns a_.clone()
        result = a_ | a_
    else:
        result = ne(sum(a_, dim=dim, keepdim=keepdim), False)

    # Pytorch's torch.any matches the behavior of NumPy in returning output of dtype bool for all supported dtypes except uint8.
    # For uint8 the dtype of output is uint8 iteself (https://pytorch.org/docs/stable/generated/torch.any.html)
    if a.dtype is dtypes.uint8:
        return prims.convert_element_type(result, dtypes.uint8)
    return result


@torchsymbol(torch.amax, is_method=True)
def amax(a, /, dim=None, keepdim: bool = False):
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
def amin(a, /, dim=None, keepdim: bool = False):
    return _reduction(
        a,
        prims.amin,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


# NOTE: Using name `torch_max` to avoid conflict with Python's `max`
@overload
def torch_max(a: TensorLike, /) -> TensorLike: ...


@overload
def torch_max(a: TensorLike, /, dim: NumberLike, keepdim: bool = False) -> tuple[TensorLike, TensorLike]: ...


@overload
def torch_max(a: TensorLike, b: TensorLike, /) -> TensorLike: ...


@torchsymbol(torch.max, is_method=True, method_name="max", id="torch.max")
def torch_max(
    a, /, dim: NumberLike | TensorLike | None = None, keepdim: bool = False
) -> TensorLike | tuple[TensorLike, TensorLike]:
    utils.check_type(dim, (NumberLike, TensorLike, NoneType))
    utils.check_type(
        keepdim, (bool, IntegerProxy)
    )  # `keepdim` can be a [IntegerProxy (bool type) name=keepdim, value=False]
    if isinstance(dim, TensorLike):
        # overload - torch_max(a: TensorLike, b: TensorLike, /) -> TensorLike
        # This overload corresponds to taking the elementwise max between tensors `a` and `b`.
        utils.check(not keepdim, lambda: "keepdim=True is invalid for torch.max(a, b) overload.")
        b = dim
        return maximum(a, b)

    if dim is None:
        # overload - torch_max(a: TensorLike, /) -> TensorLike
        # This overload corresponds to taking the max over the flattened tensor.
        utils.check(not keepdim, lambda: "keepdim=True is invalid for torch.max(a) overload.")
        dim = list(range(a.ndim))
        return amax(a, dim, keepdim)

    # overload - torch_max(a: TensorLike, /, dim: int | tuple[int], keepdim: bool = False) -> TensorLike, TensorLike
    # This overload corresponds to taking the max along the specified dimension `dim`.
    # NOTE: It returns first occurence of the maximum value along the dimension and it's corresponding index.
    utils.check_type(dim, NumberLike)
    max_vals = amax(a, dim, keepdim)
    argmax_vals = argmax(a, dim, keepdim)
    return max_vals, argmax_vals


@torchsymbol(torch.clone, is_method=True)
def clone(a: TensorProxy, *, memory_format=torch.preserve_format) -> TensorProxy:
    """Produce a copy of a tensor as a distinct new tensor."""
    # Our implementation currently does not introduce a copy, and so nothing
    # except preserve_format is feasible to support.
    # If you're hitting this you could try commenting this check out; if your
    # model does not actually rely on specified memory formats then it should
    # be fine.
    if memory_format is not torch.preserve_format:
        raise NotImplementedError("only preserve_format is currently supported")
    return prims.clone(a)


# Because we do not use @torchsymbol, we need to manually register the
# implementation.
register_function(torch.clone, clone)
register_function(torch.Tensor.clone, clone)
register_method("clone", clone)


@torchsymbol(torch.nn.functional.glu, is_method=False)
def glu(a: TensorProxy, /, dim: int = -1) -> TensorProxy:
    dim = utils.canonicalize_dim(len(a.shape), dim)
    utils.check(
        a.shape[dim] % 2 == 0,
        lambda: f"Halving dimension must be even, but dimension {dim} is size {a.shape[dim]}",
    )
    chunk_size = a.shape[dim] // 2
    left, right = split(a, (chunk_size, chunk_size), dim=dim)
    out = left * sigmoid(right)
    return out


@torchsymbol(torch.mean, is_method=True)
def mean(a: TensorProxy, /, dim=None, keepdim: bool = False, *, dtype=None) -> TensorProxy:
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
    result = to(result, result_dtype)
    return result


@torchsymbol(torch.prod, is_method=True)
def prod(
    a: TensorProxy, /, dim: None | Sequence[int] = None, keepdim: bool = False, *, dtype: None | dtypeLike = None
) -> TensorProxy:
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
def sum(
    a: TensorLike, /, dim: None | int | Sequence[int] = None, keepdim: bool = False, *, dtype: None | dtypeLike = None
) -> TensorLike:
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


def _sum_grad(
    a: TensorLike, /, dim: None | int | Sequence[int] = None, keepdim: bool = False, *, dtype: None | dtypeLike = None
) -> TensorLike:
    fwd = sum(a, dim=dim, keepdim=keepdim, dtype=dtype)

    g = get_grad(fwd)

    if dim is None or keepdim:
        g = expand(g, a.shape)
    else:
        dim = utils.canonicalize_dims(a.ndim, utils.sequencify(dim))
        broadcast_dimensions = [d for d in range(a.ndim) if d not in dim]
        g = prims.broadcast_in_dim(g, a.shape, broadcast_dimensions)

    a_grad = g.to(a.dtype)
    put_grad(a, a_grad)

    return fwd


register_grad(sum, _sum_grad)


# NOTE This decomposition can not be efficiently fused, so make it primitive
@torchsymbol(torch.cumsum, is_method=True, is_prim=True)
def cumsum(a: TensorLike, dim: int, *, dtype: None | dtypeLike = None) -> TensorLike:
    # check the input dimension
    utils.canonicalize_dim(a.ndim, dim)
    if dtype is None:
        return TensorProxy(like=a)
    else:
        return TensorProxy(like=a, dtype=to_dtype(dtype))


@torchsymbol(torch.Tensor.cumsum_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def cumsum_(a: TensorLike, dim: int, *, dtype: None | dtypeLike = None) -> TensorLike:
    return prims.copy_(cumsum(a, dim, dtype=dtype), a)


@torchsymbol(torch.var, is_method=True)
def var(
    a: TensorProxy,
    /,
    dim=None,
    *,
    keepdim: bool = False,
    correction: NumberLike = 1,
) -> TensorProxy:
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


@torchsymbol(torch.var_mean, tags=(prims.OpTags.REDUCTION_OP,))
def var_mean(
    a: TensorProxy,
    /,
    dim=None,
    *,
    keepdim: bool = False,
    correction: NumberLike = 1,
) -> tuple[TensorProxy, TensorProxy]:
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


@torchsymbol(torch.argmax, is_method=True)
def argmax(a: TensorLike, /, dim: int | None = None, keepdim: bool | None = False):
    return clang.argmax(a, dim, keepdim)


@torchsymbol(torch.argmin, is_method=True)
def argmin(a: TensorLike, /, dim: int | None = None, keepdim: bool | None = False):
    return clang.argmin(a, dim, keepdim)


@torchsymbol(torch.topk, is_method=True)
def topk(
    a: TensorLike, /, k: int, dim: None | int = None, largest: bool = True, sorted: bool = True, *, out=None
) -> (TensorLike, TensorLike):
    return clang.topk(a, k, dim, largest, sorted, out=out)


@torchsymbol(torch.sort, is_method=True)
def sort(
    a: TensorLike, /, dim: None | int = None, descending: bool = False, stable: bool = False, *, out=None
) -> (TensorLike, TensorLike):
    return clang.sort(a, dim, descending, stable, out=out)


#
# Scatter and gather-related operations
#


# NOTE PyTorch also has an alpha parameter
@torchsymbol(torch.index_add, is_method=True)
def index_add(a: TensorLike, /, dim: int, index: TensorLike, source: TensorLike) -> TensorLike:
    return clang.index_add(a, index, source, dim)


@torchsymbol(torch.Tensor.index_add_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def index_add_(a: TensorLike, /, dim: int, index: TensorLike, source: TensorLike) -> TensorLike:
    return prims.copy_(index_add(a, dim, index, source), a)


@torchsymbol(torch.index_copy, is_method=True)
def index_copy(a: TensorLike, /, dim: int, index: TensorLike, source: TensorLike) -> TensorLike:
    return clang.index_copy(a, index, source, dim)


@torchsymbol(torch.Tensor.index_copy_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def index_copy_(a: TensorLike, /, dim: int, index: TensorLike, source: TensorLike) -> TensorLike:
    return prims.copy_(index_copy(a, dim, index, source), a)


@torchsymbol(torch.index_select, is_method=True)
def index_select(a: TensorLike, /, dim: int, index: TensorLike) -> TensorLike:
    return clang.take(a, index, dim)


@torchsymbol(torch.gather, is_method=True)
def gather(a: TensorLike, /, dim: int, index: TensorLike) -> TensorLike:
    return clang.gather(a, indices=index, dim=dim)


# NOTE: PyTorch uses `src` for torch.Tensor arguments and `value` for scalars
# when referencing the source of the values
@torchsymbol(torch.scatter, is_method=True)
def scatter(
    a: TensorLike,
    /,
    dim: int,
    index: TensorLike,
    src: TensorLike | None = None,
    *,
    value: None | Number = None,
    reduce: None | str = None,
) -> TensorLike:
    utils.check(
        reduce is None, lambda: "scatter: `reduce` argument other than None is not supported", NotImplementedError
    )

    utils.check(
        (src is not None) ^ (value is not None),
        lambda: f"scatter: only one of the arguments (`src`, `value`) can be non-None",
    )

    if src is not None:
        return clang.scatter(a, index, src, dim)
    else:
        return clang.scatter(a, index, value, dim)


@torchsymbol(torch.Tensor.scatter_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def scatter_(
    a: TensorLike,
    /,
    dim: int,
    index: TensorLike,
    src: TensorLike | None = None,
    *,
    value: None | Number = None,
    reduce: None | str = None,
) -> TensorLike:
    utils.check(
        reduce is None, lambda: "scatter_: `reduce` argument other than None is not supported", NotImplementedError
    )

    utils.check(
        (src is not None) ^ (value is not None),
        lambda: f"scatter_: only one of the arguments (`src`, `value`) can be non-None",
    )

    if src is None:
        src = value

    return prims.copy_(clang.scatter(a, index, src, dim), a)


# NOTE PyTorch's scatter_add has a parameter named 'src', not 'source'
@torchsymbol(torch.scatter_add, is_method=True)
def scatter_add(a: TensorLike, /, dim: int, index: TensorLike, src: TensorLike) -> TensorLike:
    return clang.scatter_add(a, indices=index, value=src, dim=dim)


@torchsymbol(torch.Tensor.scatter_add_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def scatter_add_(a: TensorLike, /, dim: int, index: TensorLike, src: TensorLike) -> TensorLike:
    return prims.copy_(scatter_add(a, dim, index, src), a)


@torchsymbol(torch.take_along_dim, is_method=True)
def take_along_dim(a: TensorLike, /, indices: TensorLike, dim: int) -> TensorLike:
    return clang.take_along_axis(a, indices, dim)


@torchsymbol(torch.index_put, is_method=True)
def index_put(
    a: TensorLike, /, indices: Sequence[TensorLike], values: TensorLike, accumulate: bool = False
) -> TensorLike:
    return clang.index_put(a, indices, values, accumulate)


@torchsymbol(torch.index_put_, is_method=True, tags=(prims.OpTags.IN_PLACE,))
def index_put_(
    a: TensorLike,
    /,
    indices: Sequence[TensorLike],
    values: TensorLike,
    accumulate: bool = False,
) -> TensorLike:
    return prims.copy_(index_put(a, indices, values, accumulate), a)


#
# Linear Algebra operations
#


# Constructs a generalized diagonal tensor of the given rank
# of shape (dim_len,) * rank and boolean dtype.
def generalized_diagonal_tensor(dim_len, rank, device):
    assert rank >= 2

    def construct_eye(dim_len):
        iota = arange(dim_len, device=device)
        eye = unsqueeze(iota, -1) == unsqueeze(iota, 0)
        return eye

    eye = construct_eye(dim_len)
    diag = eye
    for _ in range(2, rank):
        diag = unsqueeze(diag, -1) * eye
    return diag


@torchsymbol(torch.einsum, is_method=False)
def einsum(equation: str, *operands: TensorLike | Sequence[TensorLike]) -> TensorLike:
    utils.check(
        isinstance(equation, str),
        lambda: f"Sublist inputs are not currently supported. "
        "Rewrite the sublist input to use a string equation, "
        "and/or file an issue requesting sublist einsum support here: "
        "https://github.com/Lightning-AI/lightning-thunder/issues/new/choose",
        exception_type=NotImplementedError,
    )

    operands = list(utils.sequencify(operands))
    utils.check_types(operands, TensorProxy)

    orig_eq = equation
    # Removes spaces and replaces ... with . to facilitate parsing later
    equation = re.sub(r"\s", "", equation)
    equation = re.sub(r"\.\.\.", ".", equation)

    # Splits the equation into input/output subscripts
    input_output_subscripts = equation.split("->")
    input_subscript: str
    output_subscript: None | str = None
    if len(input_output_subscripts) == 1:
        (input_subscript,) = input_output_subscripts
    else:
        utils.check(
            len(input_output_subscripts) == 2, lambda: f"Found multiple arrows (->) in einsum equation", ValueError
        )
        input_subscript, output_subscript = input_output_subscripts

    subscripts = input_subscript.split(",")
    utils.check(
        len(subscripts) == len(operands),
        lambda: f"Found {len(subscripts)} operand(s) in the equation, but {len(operands)} tensor(s) were provided",
        ValueError,
    )

    # Iterator class that maps each subscript label to a corresponding dimension.
    # These dimensions are iterated left-to-right with positive indices before
    # hitting the ellipis signaling symbol ('.'), and from right-to-left with
    # negative indices afterwards.
    class LabelDimIter:
        def __init__(self, pos, eq):
            self.pos = pos
            self.eq = eq

        def __iter__(self):
            self.eq_iter = zip(self.eq, range(len(self.eq)))
            self.seen_ellipsis = False
            return self

        def __next__(self):
            l, d = next(self.eq_iter)
            if l == ".":
                utils.check(
                    not self.seen_ellipsis,
                    lambda: f"Incorrect subscript for operand #{self.pos}: " "it contains two or more ellipses",
                    ValueError,
                )
                self.seen_ellipsis = True
                self.ellipsis_start = d
                self.ellipsis_end = -1

                rest_eq = self.eq[d + 1 :]
                self.eq_iter = zip(reversed(rest_eq), range(-1, -len(rest_eq) - 1, -1))
            # This variable has meaning only if ellipsis is present.
            self.ellipsis_end = d - 1

            return l, d

    # Returns a label -> list of dims positions map.
    # If Ellipsis is in the map,
    # __getitem__(Ellipsis) will be a 2-list [ellipsis_start_dim, ellipsis_end_dim]
    # (spanning subdims includes ellipsis_start_dim and ellipsis_end_dim)
    # with ellipsis_start_dim >= 0 and ellipsis_end_dim < 0,
    def get_subscript_spec(pos, subscript, operand=None):
        n_subscripted_dims = 0
        label_dims_map = {}
        label_dim_iter = LabelDimIter(pos, subscript)
        for l, d in label_dim_iter:
            # Skip ellipsis
            if l == ".":
                continue

            n_subscripted_dims = n_subscripted_dims + 1
            dims = label_dims_map.setdefault(l, [])
            if dims:
                utils.check(
                    operand is None or operand.shape[d] == operand.shape[dims[-1]],
                    lambda: f"Incorrect subscript for operand #{pos}: "
                    f"repeated label '{l}' requires dimensions "
                    f"{d} and {dims[-1]} to have the same lenght, "
                    f"but got {operand.shape[d]} != {operand.shape[dims[-1]]}",
                    ValueError,
                )
            dims.append(d)

        # Cannot subscript more dims than there are in the operand.
        utils.check(
            operand is None or n_subscripted_dims <= operand.ndim,
            lambda: f"Incorrect subscript for operand #{pos}: "
            f"it subscripts more dimenions ({n_subscripted_dims}) "
            f"then there are in the operand ({operand.ndim})",
            ValueError,
        )

        # Check no present ellipsis implies all dims are subscripted.
        utils.check(
            operand is None or label_dim_iter.seen_ellipsis or n_subscripted_dims == operand.ndim,
            lambda: f"Incorrect subscript for operand #{pos}: "
            "in the absence of ellipsis the number of subscripted dims "
            f"{n_subscripted_dims} has to match the operand's dimensionality "
            f"{operand.ndim}",
            ValueError,
        )

        if label_dim_iter.seen_ellipsis:
            label_dims_map.setdefault(Ellipsis, []).extend((label_dim_iter.ellipsis_start, label_dim_iter.ellipsis_end))
        return label_dims_map

    # Do some basic subscript checking.
    # TODO: consolidate Numpy/PyTorch ways of handling ellipses.
    # opt_einsum does it the Numpy way.
    # This is fine for PyTorch as long as opt_einsum is in the path,
    # and as long as it's computation path is not it's default.
    operand_subscript_specs = [
        get_subscript_spec(pos, op_spec, op) for pos, (op_spec, op) in enumerate(zip(subscripts, operands))
    ]
    out_spec = get_subscript_spec(len(operands), "" if output_subscript is None else output_subscript)

    # NOTE: the checks below are must since opt_einsum is not doing them. {
    operand_union_subscript_spec = collections.ChainMap(*operand_subscript_specs)
    for l, dims in out_spec.items():
        # Skip ellipsis.
        if l is Ellipsis:
            continue

        # check uniqueness.
        utils.check(
            len(dims) == 1,
            lambda: f"Output subscript string '{output_subscript}' includes multiple '{l}' labels",
            ValueError,
        )

        # check output labels are coming from operand's subscripts.
        utils.check(
            l in operand_union_subscript_spec,
            lambda: f"Output subscript string '{output_subscript}' includes a '{l}' label "
            "which does not apper in neither of the operand's subsripts",
            ValueError,
        )
    # }

    # Check Ellipsis consistency - ellipsis-covered subshapes need to broadcast. {
    op_ellipsis_shapes = []
    for op_spec, op, s in zip(operand_subscript_specs, operands, subscripts):
        ellipsis_start, ellipsis_end = op_spec.get(Ellipsis, (0, -op.ndim - 1))
        op_ellipsis_shapes.append(op.shape[ellipsis_start : ellipsis_end + op.ndim + 1])

    try:
        clang.compute_broadcast_shape(*op_ellipsis_shapes)
    except Exception as e:
        raise ValueError("Implied by ellipsis operands' dimensions do not jointy broadcast") from e
    # }

    # A helper function that removes characters from a string.
    def removechars(s, chars):
        return s.translate(str.maketrans(dict.fromkeys(chars)))

    # Given an operand and it's labels, contract along dimensions specified
    # in the index form in contr_dims and in the label form in contr_labels.
    def contract_operand(operand, operand_labels, contr_dims, contr_labels):
        if contr_dims:
            operand = sum(operand, contr_dims)
            operand_labels = removechars(operand_labels, contr_labels)
        return operand, operand_labels

    # Constructs a generalized diagonal mask that broadcasts
    # over t, with diagonals across dimensions from the
    # diagonal_dims argument.
    # The result of this function is used for masking diagonals
    # when contracting over repeated labels in subscripts.
    def construct_broadcastable_diagonal(t, diagonal_dims):
        # TODO: revisit and use index_put instead.
        dim_len = t.shape[diagonal_dims[0]]
        gen_diagonal = generalized_diagonal_tensor(dim_len, len(diagonal_dims), t.device)
        return prims.broadcast_in_dim(gen_diagonal, t.shape, diagonal_dims)

    # Labels unique to each operand are trivially contractable with sum.
    def find_unique_labels(operand, labels, unique_labels):
        if unique_labels:
            dims = [labels.index(name) for name in unique_labels]
            return dims, unique_labels
        else:
            return [], []

    # Repeated labels imply contractions over masked diagonals.
    def find_repated_labels(operand, labels, counts, keep_labels):
        orig_labels = labels
        # Dims to contract over, these correspond to repeated labels.
        dims = []
        # Groups diagonal dimensions of the same length.
        # This allows to construct a broadcastable diagonal mask
        # in a single call (per unique dim length)
        # hence reducing the number of kernel calls.
        diag_groups: Dict[int, list[int]] = {}

        for label, count in counts.items():
            # Process only repeated labels.
            if count > 1:
                label_dims = [d for d, l in enumerate(orig_labels) if l == label]
                label_dim_len = operand.shape[label_dims[0]]
                diag_groups.setdefault(label_dim_len, []).extend(label_dims)
                if label not in keep_labels:
                    # Fully contract the labels which need no preservation
                    # (for example, when they are not in the output).
                    labels = labels.replace(label, "")
                else:
                    # Otherwise contract over first (count - 1) occurrencies.
                    labels = labels[::-1].replace(label, "", count - 1)[::-1]
                    label_dims = label_dims[1:]
                dims.extend(label_dims)

        if dims:
            # "Diagonalize" over each dim group.
            for diag_dims in diag_groups.values():
                diag = construct_broadcastable_diagonal(operand, diag_dims)
                operand = where(diag, operand, 0)

        return operand, labels, dims, []

    def find_broadcast_labels(a, a_labels, b, b_labels):
        common_contraction_set = frozenset(a_labels) & frozenset(b_labels) & contraction_set
        a_contr_dims = [a_labels.index(l) for l in common_contraction_set]
        b_contr_dims = [b_labels.index(l) for l in common_contraction_set]

        a_broadcast_dims = []
        a_broadcast_labels = []

        b_broadcast_dims = []
        b_broadcast_labels = []

        for a_contr_dim, b_contr_dim in zip(a_contr_dims, b_contr_dims):
            if a.shape[a_contr_dim] == 1 or b.shape[b_contr_dim] == 1:
                a_broadcast_dims.append(a_contr_dim)
                a_broadcast_labels.append(a_labels[a_contr_dim])

                b_broadcast_dims.append(b_contr_dim)
                b_broadcast_labels.append(b_labels[b_contr_dim])

        return a_broadcast_dims, a_broadcast_labels, b_broadcast_dims, b_broadcast_labels

    # Process contraction path.
    _, contractions = opt_einsum.contract_path(orig_eq, *operands, einsum_call=False)
    for operand_indices, contraction_set, eineq, _, contr_op_type in contractions.contraction_list:
        input_eq, output_eq = eineq.split("->")
        input_labels = input_eq.split(",")

        if len(operand_indices) == 1:
            operand = operands.pop(operand_indices[0])
            (labels,) = input_labels
            counts = collections.Counter(labels)

            # Find unique contraction indices.
            unique_labels = [l for l in contraction_set if counts[l] == 1]
            unique_contr_dims, unique_contr_labels = find_unique_labels(operand, labels, unique_labels)

            # Find repeated indices over "diagonalized" operand.
            operand, labels, repeated_contr_dims, repeated_contr_labels = find_repated_labels(
                operand, labels, counts, output_eq
            )

            # Contract over unique and repeated dims/labels
            operand, labels = contract_operand(
                operand,
                labels,
                unique_contr_dims + repeated_contr_dims,
                unique_contr_labels + repeated_contr_labels,
            )

        elif len(operand_indices) == 2:
            a, b = map(operands.pop, operand_indices)
            a_labels, b_labels = input_labels

            a_counts = collections.Counter(a_labels)
            b_counts = collections.Counter(b_labels)

            a_unique_labels = [l for l in contraction_set if a_counts[l] == 1 and b_counts[l] == 0]
            b_unique_labels = [l for l in contraction_set if b_counts[l] == 1 and a_counts[l] == 0]

            # Find unique to each operand dims/labels
            a_unique_contr_dims, a_unique_contr_labels = find_unique_labels(a, a_labels, a_unique_labels)
            b_unique_contr_dims, b_unique_contr_labels = find_unique_labels(b, b_labels, b_unique_labels)

            # Find repeated labels which are not in the output and not in the other operand.
            a, a_labels, a_repeated_contr_dims, a_repeated_contr_labels = find_repated_labels(
                a, a_labels, a_counts, output_eq + b_labels
            )
            b, b_labels, b_repeated_contr_dims, b_repeated_contr_labels = find_repated_labels(
                b, b_labels, b_counts, output_eq + a_labels
            )

            # Find broadcast dims that we can also sum out to reduce op domain.
            (
                a_broadcast_contr_dims,
                a_broadcast_contr_labels,
                b_broadcast_contr_dims,
                b_broadcast_contr_labels,
            ) = find_broadcast_labels(a, a_labels, b, b_labels)

            # Contract dims/labels from the previous steps
            a, a_labels = contract_operand(
                a,
                a_labels,
                a_unique_contr_dims + a_repeated_contr_dims + a_broadcast_contr_dims,
                a_unique_contr_labels + a_repeated_contr_labels + a_broadcast_contr_labels,
            )
            b, b_labels = contract_operand(
                b,
                b_labels,
                b_unique_contr_dims + b_repeated_contr_dims + b_broadcast_contr_dims,
                b_unique_contr_labels + b_repeated_contr_labels + b_broadcast_contr_labels,
            )

            # Process remaining contractions below.
            a_labels_set = frozenset(a_labels)
            b_labels_set = frozenset(b_labels)

            # Filter out labels contracted over the previous steps.
            contraction_set = contraction_set & a_labels_set & b_labels_set
            # Non-contractable labels present in the output and in the operands
            # form the basis of "batch" dimensions.
            batch_labels_set = frozenset(output_eq) & a_labels_set & b_labels_set

            # Partition operand's dimensions into
            # batch dimensions, contraction dimensions, and the rest dimensions.
            # Additionally, the batch dimensions are sorted lexicographically
            # to facilitate operands' mutual alignment across batch dimensions.
            def partition_align_dims(operand, labels, labels_set):
                # Extract sublabels' corresponding dimension indices.
                def get_dim_idxs(sublabels, force_lex_order=False):
                    if force_lex_order:
                        sublabels = sorted(sublabels)
                    return [labels.index(c) for c in sublabels]

                # Batch dimensions are shared among several operands
                # (see the definition of batch_labels_set variable).
                # As such, their order is fixed to enforce a matching alignment.
                batch_idx = get_dim_idxs(batch_labels_set, force_lex_order=True)
                contraction_idx = get_dim_idxs(contraction_set)
                rest_dims_idx = get_dim_idxs(labels_set - batch_labels_set - contraction_set)
                return batch_idx, contraction_idx, rest_dims_idx

            def apply_perm(seq, perm):
                return [seq[i] for i in perm]

            def get_shape_numel(shape):
                return reduce(operator.mul, shape, 1)

            # a and b can be contracted with bmm if
            # - contraction set is non-empty,
            # - a's and b's contraction dimension have the same lenght,
            #   NOTE: if needed, we can relax this requirement
            #   to just being broadcastable.
            #   TODO: investigate this.
            # - both a_rest and b_rest dimensions are non-empty.
            def is_bmm_contractable(a, a_contr, a_rest, b, b_contr, b_rest):
                if (a_contr or b_contr) and (a_rest and b_rest):
                    a_contr_shape = [a.shape[d] for d in a_contr]
                    b_contr_shape = [b.shape[d] for d in b_contr]
                    return get_shape_numel(a_contr_shape) == get_shape_numel(b_contr_shape)
                return False

            a_batch, a_contr, a_rest = partition_align_dims(a, a_labels, a_labels_set)
            b_batch, b_contr, b_rest = partition_align_dims(b, b_labels, b_labels_set)

            if contr_op_type and contr_op_type.startswith("OUTER"):
                # Outer product path. It could also be identified by the contraction set being empty.
                a = reshape(a, a.shape + (1,) * b.ndim)
                operand = a * b
                labels = a_labels + b_labels

            elif (
                dtypes.is_float_dtype(a.dtype)
                and dtypes.is_float_dtype(b.dtype)
                and is_bmm_contractable(a, a_contr, a_rest, b, b_contr, b_rest)
            ):
                # The path to contractions with a single matmul call.
                # The a and b operands' dimensions are permuted into
                # (a_batch, a_rest, a_contr) and (b_batch, b_contr, b_rest) shapes,
                # then the "rest" and the "contraction" dimensions get squashed so that
                # a and b represent batched matrices suitable for contracting
                # with a single matmul call.

                a_perm = a_batch + a_rest + a_contr
                b_perm = b_batch + b_contr + b_rest

                a = permute(a, a_perm)
                b = permute(b, b_perm)

                # broadcast shape of a_batch and b_batch dims.
                res_batch_shape = clang.compute_broadcast_shape(a.shape[: len(a_batch)], b.shape[: len(b_batch)])
                # shape of a_rest dims.
                res_a_shape = a.shape[len(a_batch) : -len(contraction_set)]
                # shape of b_rest dims.
                res_b_shape = b.shape[len(b_batch) + len(contraction_set) :]
                # The shape the result of matmul will be reshaped into.
                res_shape = res_batch_shape + res_a_shape + res_b_shape

                # reshape (a_batch, a_rest, a_contr) -> (a_batch, numel(a_rest), numel(a_contr)).
                a_rest_shape = get_shape_numel(a.shape[len(a_batch) : len(a_batch) + len(a_rest)])
                a_contr_shape = get_shape_numel(a.shape[-len(a_contr) :])
                a = reshape(a, a.shape[: len(a_batch)] + (a_rest_shape, a_contr_shape))

                # reshape (b_batch, b_contr, b_rest) -> (b_batch, numel(b_contr), numel(b_rest)).
                b_rest_shape = get_shape_numel(b.shape[-len(b_rest) :])
                b_contr_shape = get_shape_numel(b.shape[len(b_batch) : len(b_batch) + len(b_contr)])
                b = reshape(b, b.shape[: len(b_batch)] + (b_contr_shape, b_rest_shape))

                # Perform a gemm/bmm contraction and restore the full-dim shape.
                operand = matmul(a, b)
                operand = reshape(operand, res_shape)

                # Update the operand's labels.
                # These will correspond to the string "{batch_dims}{a_rest}{b_rest}".
                a_labels = "".join(apply_perm(list(a_labels), a_perm))
                b_labels = "".join(apply_perm(list(b_labels), b_perm))
                labels = a_labels[: -len(a_contr)] + b_labels[-len(b_rest) :]

            else:
                # The path to contractions with sum, aka TDOT, if contraction set is non-empty.
                # If it is empty, it is similar to the OUTER path above but with special treatment
                # of "batch" dimensions that requires alignment and handling of broadcasting.
                # When the contraction set is non-emtpy,
                # the idea is very similar to what is done in the gemm/bmm path above.
                # Note: contracting over last dimensions for best efficiency
                # when reducing over a * b.
                a_perm = a_batch + a_rest + a_contr
                b_perm = b_batch + b_rest + b_contr

                a = permute(a, a_perm)
                b = permute(b, b_perm)

                # a = (a_batch, a_rest, a_contr),
                # b = (b_batch, b_rest, b_contr), so
                # a and b are aligned to
                # a = (a_batch, a_rest, 1(b_rest), a_contr),
                # b = (b_batch, 1(a_rest), b_rest, b_contr).
                a = reshape(
                    a,
                    a.shape[: len(a_batch) + len(a_rest)] + (1,) * len(b_rest) + a.shape[len(a_batch) + len(a_rest) :],
                )
                b = reshape(b, b.shape[: len(b_batch)] + (1,) * len(a_rest) + b.shape[len(b_batch) :])

                # Contraction set could be empty.
                # This case is distinguished since sum over empty dims
                # sums all the elements, and we want to avoid that.
                if contraction_set:
                    dims = list(range(-len(contraction_set), 0))
                    operand = sum(a * b, dims)
                else:
                    # Nothing to contract and a and b are already aligned and broadcastable,
                    # so the result is a simple mul.
                    operand = a * b

                # Update the operand's labels.
                # These will correspond to the string "{batch_dims}{a_rest}{b_rest}".
                a_labels = "".join(apply_perm(list(a_labels), a_perm))
                b_labels = "".join(apply_perm(list(b_labels), b_perm))
                labels = a_labels[: len(a_batch) + len(a_rest)] + b_labels[len(b_batch) : len(b_batch) + len(b_rest)]

        else:
            raise NotImplementedError

        # Check that contraction labels are contracted and the output's labels are preserved.
        assert frozenset(labels) == frozenset(output_eq)
        # Permute to the output's dim order.
        if labels != output_eq:
            perm = tuple(labels.index(label) for label in output_eq)
            operand = permute(operand, perm)

        operands.append(operand)

    return operands[0]


# NOTE: this wrapper for prim matmul just broadcasts batch dimensions
@torchsymbol(torch.matmul, is_method=True)
def matmul(a: TensorLike, b: TensorLike, /) -> TensorLike:
    if a.ndim == 1 or b.ndim == 1:
        return prims.matmul(a, b)

    # Case nd @ 2d --> reduce to a 2d gemm
    if a.ndim > 2 and b.ndim == 2:
        a_batch_dims = a.shape[:-2]

        # a -> a_2d by flattening batch dims with the row space
        a_2d = a.reshape(-1, a.shape[-1])

        # 2d gemm
        res_2d = prims.matmul(a_2d, b)

        # reshape `res` from 2d to a proper nd shape
        res = res_2d.reshape(*a_batch_dims, -1, b.shape[-1])
        return res

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


@torchsymbol(torch.outer, is_method=True)
def outer(a: TensorLike, b: TensorLike, /) -> TensorLike:
    utils.check_types((a, b), TensorProxy)

    utils.check(
        a.ndim == 1,
        lambda: f"Expected both inputs to be 1-dimensional vectors, but the first input had {a.ndim} dimensions",
    )
    utils.check(
        b.ndim == 1,
        lambda: f"Expected both inputs to be 1-dimensional vectors, but the second input had {b.ndim} dimensions",
    )

    return a[:, None] * b[None, :]


#
# Normalization operations
#


def _normalize(a: TensorProxy, /, norm_dims, eps: Number) -> tuple[TensorLike, TensorLike, TensorLike]:
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
    a_acc = to(a, computation_dtype)
    biased_var, mean = var_mean(a_acc, dim=norm_dims, correction=0, keepdim=True)
    rstd = rsqrt(biased_var + eps)
    out = (a - mean) * rstd
    return out, mean, rstd


@torchsymbol(torch.nn.functional.normalize, is_method=True)
def normalize(
    a: TensorProxy, /, p: float = 2.0, dim: int | Sequence[int] = 1, eps: float = 1e-12, out: None | TensorProxy = None
) -> TensorProxy:
    utils.check(
        dtypes.is_float_dtype(a.dtype) or dtypes.is_complex_dtype(a.dtype),
        lambda: f"normalize: Expected a floating point or complext tensor as input. Got {a.dtype}",
        TypeError,
    )
    utils.check(out is None, lambda: "normalize: out is not None which is currently unsupported", NotImplementedError)
    computation_dtype, result_dtype = _reduction_dtypes(a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT, a.dtype)
    if p == 0.0:
        denom = sum(a != 0.0, dim=dim, keepdim=True)
    elif p == float("inf"):
        denom = amax(abs(a), dim=dim, keepdim=True)
    elif p == -float("inf"):
        denom = amin(abs(a), dim=dim, keepdim=True)
    else:
        dim = utils.canonicalize_dim(a.ndim, dim)
        a_ = clang.maybe_convert_to_dtype(a, computation_dtype)
        is_p_even = p % 2.0 == 0
        if dtypes.is_complex_dtype(a.dtype) or not is_p_even:
            a_ = abs(a_)
        denom = a_**p
        denom = sum(denom, dim=dim, keepdim=True)
        denom = denom ** (1.0 / p)
    denom = clamp(denom, min=eps)
    denom = expand_as(denom, a)
    denom = clang.maybe_convert_to_dtype(denom, result_dtype)
    out = a / denom
    return out


def _check_normalized_shape_and_get_reduction_dims(a, normalized_shape, weight=None, bias=None):
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
        a.shape[-normalized_ndim:] == tuple(normalized_shape),
        lambda: f"Expected the last {len(normalized_shape)} dimensions of a (a.shape={a.shape}) to be the same as {normalized_shape}",
    )

    axis = a.ndim - normalized_ndim
    reduction_dims = list(range(axis, a.ndim))
    return reduction_dims


# TODO: likely want to refactor these normalizations
def _native_layer_norm(
    a: TensorProxy, /, normalized_shape, weight, bias, eps: Number
) -> tuple[TensorLike, TensorLike, TensorLike]:
    reduction_dims = _check_normalized_shape_and_get_reduction_dims(a, normalized_shape, weight, bias)
    out, mean, rstd = _normalize(a, reduction_dims, eps)

    # Handles weight and bias
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias

    out = to(out, a.dtype)
    # TODO Is the following conversion or conversions CPU only?
    # if input.device.type == "cpu":
    mean = to(mean, a.dtype)
    rstd = to(rstd, a.dtype)

    return out, mean, rstd


# TODO Add type annotations
# TODO Move this to nn.functional
@torchsymbol(torch.nn.functional.layer_norm)
def layer_norm(
    a: TensorLike,
    /,
    normalized_shape: Sequence[int],
    weight: None | TensorLike = None,
    bias: None | TensorLike = None,
    eps: NumberLike = 1e-5,
) -> TensorLike:
    # Note [LayerNorm with parameter sharding]
    # Sharding messes up the normalized_shape argument, so we need to get the
    # unsharded normalized shape from the weight
    if weight is not None:
        normalized_ndim = len(weight.shape)
        normalized_shape = a.shape[-normalized_ndim:]
    if bias is not None:
        normalized_ndim = len(bias.shape)
        normalized_shape = a.shape[-normalized_ndim:]
    return _native_layer_norm(a, normalized_shape, weight, bias, eps)[0]


def rms_norm(
    a: TensorLike,
    /,
    normalized_shape: Sequence[int],
    weight: None | TensorLike = None,
    eps: None | float = None,
):
    if eps is None:
        eps = torch.finfo(to_torch_dtype(a.dtype)).eps
    reduction_dims = _check_normalized_shape_and_get_reduction_dims(a, normalized_shape, weight)
    norm_a = mean(a * a, dim=reduction_dims, keepdim=True)
    a_normed = a * rsqrt(norm_a + eps)
    if weight is not None:
        a_normed = a_normed * weight
    return a_normed


if hasattr(torch.nn.functional, "rms_norm"):
    rms_norm = torchsymbol(torch.nn.functional.rms_norm)(rms_norm)


def _native_batch_norm(
    a: TensorLike,
    /,
    weight: None | TensorLike,
    bias: None | TensorLike,
    running_mean: None | TensorLike,
    running_var: None | TensorLike,
    training: bool,
    momentum: Number,
    eps: Number,
) -> TensorLike:
    params_shape = (1, -1) + (1,) * (a.ndim - 2)
    computation_dtype = utils.get_computation_dtype(a.dtype)
    a_acc = to(a, computation_dtype)
    if training:
        reduction_dims = (0,) + tuple(range(2, a.ndim))
        # this should be keepdim=False  because of https://github.com/NVIDIA/Fuser/issues/1964
        biased_var, mean = var_mean(a_acc, dim=reduction_dims, correction=0, keepdim=False)
        rstd = rsqrt(biased_var + eps)
        bcast_rstd = reshape(rstd, params_shape)
        bcast_mean = reshape(mean, params_shape)
        out = (a - bcast_mean) * bcast_rstd

        if running_mean is not None:
            new_running_mean = (1 - momentum) * running_mean + momentum * mean
            if not utils.are_same_dtypes(new_running_mean, running_mean):
                new_running_mean = to(new_running_mean, running_mean.dtype)
            prims.copy_(new_running_mean, running_mean)
        if running_var is not None:
            n = a.numel() / a.shape[1]
            unbiased_var = biased_var * (n / (n - 1))
            new_running_var = (1 - momentum) * running_var + momentum * unbiased_var
            if not utils.are_same_dtypes(new_running_var, running_var):
                new_running_var = to(new_running_var, running_var.dtype)
            prims.copy_(new_running_var, running_var)
    else:
        running_var_acc = to(running_var, computation_dtype)
        rstd = rsqrt(running_var_acc + eps)
        mean = reshape(running_mean, params_shape)
        rstd = reshape(rstd, params_shape)
        out = (a_acc - mean) * rstd

    # Handles weight and bias
    if weight is not None:
        # Inserting a conversion to the computation_dtype for weight and bias to
        # disable nvFuser executors's bookend optimization (nv_enable_bookend),
        # preventing the executor to push out the shape operations out of the
        # fusion region.
        weight = to(weight, computation_dtype)
        weight = reshape(weight, params_shape)
        out = out * weight
    if bias is not None:
        bias = to(bias, computation_dtype)
        bias = reshape(bias, params_shape)
        out = out + bias

    out = to(out, a.dtype)
    return out


@torchsymbol(torch.nn.functional.batch_norm)
def batch_norm(
    a: TensorLike,
    running_mean: None | TensorLike = None,
    running_var: None | TensorLike = None,
    weight: None | TensorLike = None,
    bias: None | TensorLike = None,
    training: bool = False,
    momentum: NumberLike = 0.1,
    eps: NumberLike = 1e-5,
) -> TensorLike:
    # Validates inputs
    input_shape = tuple(a.shape)
    utils.check(len(input_shape) >= 2, lambda: f"Expected input_shape={input_shape} to have length >= 2!")

    # NOTE Containers are canonicalized in the following checks since
    #   (1, 2, 3) != [1, 2, 3]
    utils.check(
        weight is None or weight.shape == (input_shape[1],),
        lambda: f"Expected weight.shape={weight.shape} to be {(input_shape[1],)}!",
    )
    utils.check(
        bias is None or bias.shape == (input_shape[1],),
        lambda: f"Expected bias.shape={bias.shape} to be {(input_shape[1],)}!",
    )
    utils.check(
        running_mean is None or running_mean.shape == (input_shape[1],),
        lambda: f"Expected running_mean.shape={running_mean.shape} to be {(input_shape[1],)}!",
    )
    utils.check(
        running_var is None or running_var.shape == (input_shape[1],),
        lambda: f"Expected running_var.shape={running_var.shape} to be {(input_shape[1],)}!",
    )
    if training:
        size_prods = input_shape[0]
        for i in range(len(input_shape) - 2):
            size_prods *= input_shape[i + 2]
        utils.check(
            size_prods != 1, lambda: f"Expected more than 1 value per channel when training, got input size {size}"
        )
    else:
        utils.check(
            running_mean is not None and running_var is not None,
            lambda: f"running_mean and running_var must be defined in evaluation mode",
        )
    computation_dtype = utils.get_computation_dtype(a.dtype)
    # Check mixed input types
    params = [x for x in (weight, bias, running_mean, running_var) if x is not None]
    if params:
        if utils.is_low_precision_dtype(a.dtype):
            utils.check(
                utils.are_same_dtypes(params[0], a) or utils.are_same_dtypes(params[0], computation_dtype),
                lambda: f"Expected to have type {computation_dtype} or {a.dtype} but got {params[0].dtype}",
            )
        else:
            utils.check_same_dtype(params[0], a)
        utils.check_same_dtype(*params)

    result = _native_batch_norm(a, weight, bias, running_mean, running_var, training, momentum, eps)
    return result


#
# NN Operations
#


@torchsymbol(torch.baddbmm, is_method=True)
def baddbmm(
    a: TensorLike, b1: TensorLike, b2: TensorLike, *, beta: float = 1.0, alpha: float = 1.0, out: TensorLike = None
) -> TensorLike:
    utils.check(out is None, lambda: "Non-None out is not supported", NotImplementedError)

    utils.check_same_dtype(a, b1, b2)
    utils.check_same_device(a, b1, b2)
    utils.check(b1.ndim == 3, lambda: f"batch1 must be a 3D tensor, found {b1.ndim} instead.")
    utils.check(b2.ndim == 3, lambda: f"batch2 must be a 3D tensor, found {b2.ndim} instead.")

    if a.dtype not in dtypes.inexact_dtypes:
        utils.check_type(beta, int)
        utils.check_type(alpha, int)

    t0 = matmul(b1, b2)
    t1 = alpha * t0
    return t1 + (beta * a)


# TODO bmm is more restrictive than matmul
@torchsymbol(torch.bmm, is_method=True)
def bmm(a: TensorLike, b: TensorLike, /) -> TensorLike:
    return matmul(a, b)


@torchsymbol(torch.convolution, is_method=False)
def convolution(
    a: TensorLike,
    weight: TensorLike,
    bias: None | TensorLike,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> TensorLike:
    # The checks below is a PyTorch limitation of supporting only 1D, 2D and 3D convolutions.
    utils.check(a.ndim <= 5, lambda: f"Expected {a.ndim=} to be <= 5 as only up to 3D convolutions are supported")
    utils.check(
        weight.ndim <= 5, lambda: f"Expected {weight.ndim=} to be <= 5 as only up to 3D convolutions are supported"
    )

    return clang.convolution(
        a,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


# Helper functions that are useful for "window"-based ops
# like convolution, pooling and similar. {


# A decorator function for conv/pool-like functions that handles
# batch dim insertion if needed.
# PyTorch frontend allows inputs without batch dim, but our
# prim backend does not, hence this helper.
def handle_nn_op_batch_dim(f):
    @wraps(f)
    def batch_handler(dim, a, *args, **kwargs):
        # Insert batch dim into a if not present.
        # This is needed for most nn ops.
        batch_dim_inserted: bool = False
        if a.ndim == dim + 1:
            a = unsqueeze(a, 0)
            batch_dim_inserted = True

        res = f(dim, a, *args, **kwargs)

        # Undo batch dim insertion if needed.
        if batch_dim_inserted:
            res = squeeze(res, 0)

        return res

    return batch_handler


# A helper function to converts an interger to 1-len tuple.
# It is used to handle arguments like stride/dilation/padding and similar.
def int_to_seq(param):
    if isinstance(param, (int, NumberProxy)):
        return (param,)
    else:
        return param


# Transforms (x,) -> (x,) * rank.
# It is used to map arguments like stride/dilation/padding to a rank-len
# tuples for easier subsequent processing.
def maybe_to_rank_len_sequence(param, rank):
    param = int_to_seq(param)
    if len(param) == 1:
        return (param[0],) * rank
    else:
        return tuple(param)


# }


# Pad input with `pad_value`. Pool-like padding has some restrictions,
# see the checks below.
def apply_padding_for_pool_ops(dim, a, padding, kernel_size, pad_value):
    padding = maybe_to_rank_len_sequence(padding, dim)
    kernel_size = maybe_to_rank_len_sequence(kernel_size, dim)
    utils.check(
        len(padding) == dim
        and all(isinstance(p, (int, IntegerProxy)) and 0 <= p <= k // 2 for p, k in zip(padding, kernel_size)),
        lambda: f"Implied {padding=} (with dimensionality {dim}) should contain integers "
        f"between 0 and `kernel_size / 2` (with the implied {kernel_size=})",
    )

    # No need to pad batch and channels dims, only spatial dims.
    new_padding = [(0, 0, 0), (0, 0, 0)]
    for p in padding:
        new_padding.append((p, p, 0))

    a = prims.pad(a, clang.maybe_convert_to_dtype(pad_value, a.dtype, enforce_safe_casting=True), new_padding)
    return a


# TODO: add support for transposed and layout.
@handle_nn_op_batch_dim
def _conv_helper(
    dim: int,
    a: TensorProxy,
    weight: TensorProxy,
    bias: TensorProxy | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    *,
    conv_function=clang.convolution,
) -> TensorProxy:
    # a, weight rank check
    utils.check(dim + 1 <= a.ndim <= dim + 2, lambda: f"{a.ndim=} should be either {dim + 1} or {dim + 2}")
    utils.check(weight.ndim == dim + 2, lambda: f"{weight.ndim=} should be equal to {dim + 2}")

    # Handle stride, padding, dilation {
    def process_padding_str(padding, stride: Sequence[int], dilation: Sequence[int], a: TensorProxy):
        if isinstance(padding, str):
            # Means no padding
            if padding == "valid":
                return (0,), a
            elif padding == "same":
                # padding == "same" only works with strides equal to 1.
                # NOTE: stride has to be a Sequence, see the annotation!
                utils.check(
                    all(s == 1 for s in stride), lambda: f"{padding=} requires all `strides` to be 1, but got {stride=}"
                )
                utils.check(
                    len(dilation) == 1 or len(dilation) == dim, lambda: f"{len(dilation)=} has to be either 1 or {dim}"
                )
                utils.check(
                    all(isinstance(d, (int, IntegerProxy)) and d >= 1 for d in dilation),
                    lambda: f"{dilation=} has to be a Sequences of integers >= 1",
                )

                # Need to pad a because "low" padding might not be equal to "high" padding,
                # and clang.convolution assumes this equality.
                # Expand to len == dim for easier processing of the pad arguments.
                if len(dilation) == 1:
                    dilation = (dilation[0],) * dim

                def pad_lo_hi_dilation_seq():
                    # No need to pad batch and channels dim
                    res = [(0, 0, 0), (0, 0, 0)]
                    _, _, *kernel_size = weight.shape
                    for d, k in zip(dilation, kernel_size):
                        total_p = d * (k - 1)
                        lo = total_p // 2
                        hi = total_p - lo
                        res.append((lo, hi, 0))
                    return res

                a = prims.pad(
                    a, clang.maybe_convert_to_dtype(0, a.dtype, enforce_safe_casting=True), pad_lo_hi_dilation_seq()
                )
                return (0,), a
            else:
                utils.check(
                    False,
                    lambda: f"padding string values other than ('valid', 'same') " "are not supported, got {padding=}",
                )
        else:
            return padding, a

    stride = int_to_seq(stride)
    dilation = int_to_seq(dilation)
    padding, a = process_padding_str(int_to_seq(padding), stride, dilation, a)
    # }

    res = conv_function(
        a,
        weight,
        bias,
        stride,
        padding,
        dilation,
        False,  # transposed
        (0,) * dim,  # output_padding
        groups,
    )
    return res


@handle_nn_op_batch_dim
def _max_pool_helper(
    dim: int,
    a: TensorProxy,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> TensorProxy:
    utils.check(
        not return_indices,
        lambda: "{return_indices=} is not supported",
        NotImplementedError,
    )

    utils.check(
        not ceil_mode,
        lambda: "{ceil_mode=} is not supported",
        NotImplementedError,
    )

    if stride is None:
        stride = kernel_size

    kernel_size = maybe_to_rank_len_sequence(kernel_size, dim)
    utils.check(
        len(kernel_size) == dim and all(isinstance(k, (int, IntegerProxy)) and k > 0 for k in kernel_size),
        lambda: f"Implied {kernel_size=} (with dimensionality {dim}) should either be a non-negative integer "
        f"or a sequence of non-negative integers of length {dim}",
    )

    # Check channels > 0 {
    n_channels = a.shape[1]
    utils.check(n_channels > 0, lambda: f"in_channels={n_channels} should be greater than zero")
    # }

    # Apply padding {
    a = apply_padding_for_pool_ops(dim, a, padding, kernel_size, -float("inf"))
    # }

    # Dimensionality of the kernel.
    kernel_numel = reduce(operator.mul, kernel_size, 1)

    # Construct the kernel basis which is represented as eye(kernel_numel).
    kernel_basis = generalized_diagonal_tensor(kernel_numel, rank=2, device=a.device)
    kernel_basis = to(kernel_basis, a.dtype)

    # Next steps - reshape kernel_basis to be used as weights in the convolution op.
    # We will use the trick of groups=n_channels to make sure that convolution does not occur
    # across several channel dimensions, this implies n_channels / groups == 1 and is reflected
    # in the 1-len dimension inserted to the left from the spatial dimensions.
    # The very first new 1-len dimension is expanded to n_channels so that per-channel
    # max pool is retained.
    kernel_basis = reshape(kernel_basis, (1, kernel_numel, 1, *kernel_size))
    kernel_basis = expand(kernel_basis, (n_channels, kernel_numel, 1, *kernel_size))
    # Flatten channels and the kernel basis dimensions.
    kernel_basis = flatten(kernel_basis, 0, 1)

    # Decompose (project) input in the kernel basis.
    res = _conv_helper(dim, a, kernel_basis, None, stride, padding=0, dilation=dilation, groups=n_channels)

    # Reshape projection by splitting the out_channels dimension
    # into channels and the kernel basis dimensions.
    res = reshape(res, (res.shape[0], n_channels, kernel_numel) + res.shape[-dim:])
    # Find a basis vector that has the largest projection scalar.
    # This is the max_pool result.
    res = amax(res, 2)
    return res


@handle_nn_op_batch_dim
def _avg_pool_helper(
    dim: int,
    a: TensorProxy,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: NumberLike | None = None,
) -> TensorProxy:
    utils.check(
        not ceil_mode,
        lambda: "{ceil_mode=} is not supported",
        NotImplementedError,
    )

    utils.check(
        count_include_pad,
        lambda: "{count_include_pad=} is not supported",
        NotImplementedError,
    )

    if stride is None:
        stride = kernel_size

    kernel_size = maybe_to_rank_len_sequence(kernel_size, dim)
    utils.check(
        len(kernel_size) == dim and all(isinstance(k, (int, IntegerProxy)) and k > 0 for k in kernel_size),
        lambda: f"Implied {kernel_size=} (with dimensionality {dim}) should either be a non-negative integer "
        f"or a sequence of non-negative integers of length {dim}",
    )

    # Check channels > 0 {
    n_channels = a.shape[1]
    utils.check(n_channels > 0, lambda: f"in_channels={n_channels} should be greater than zero")
    # }

    # Apply padding {
    a = apply_padding_for_pool_ops(dim, a, padding, kernel_size, 0)
    # }

    # Dimensionality of the kernel.
    kernel_numel = reduce(operator.mul, kernel_size, 1)

    # nn.functional.avg_pool does not have `divisor_override`.
    # TODO: look into PyTorch side; is this behavior deliberate? Could be that
    # 1D case is niche.
    # If needed, handle it with checks and transforms. For now unconditionally
    # override value with kernel_numel.
    if divisor_override is None or dim == 1:
        divisor_override = kernel_numel

    utils.check(
        isinstance(divisor_override, (Number, NumberProxy)) and divisor_override > 0,
        lambda: f"{divisor_override=} should be a greater than 0 scalar",
    )

    kernel = ones(*kernel_size, device=a.device, dtype=a.dtype) / divisor_override
    kernel = reshape(kernel, (1, 1, *kernel_size))
    kernel = expand(kernel, (n_channels, 1, *kernel_size))

    # groups set to n_channels as pool ops operate over spatial domains and never over channels.
    res = _conv_helper(dim, a, kernel, None, stride, padding=0, dilation=1, groups=n_channels)
    return res


@torchsymbol(torch.avg_pool1d, torch.nn.functional.avg_pool1d, id="torch.nn.functional.avg_pool1d", is_method=False)
def avg_pool1d(
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: NumberLike | None = None,
) -> TensorProxy:
    return _avg_pool_helper(1, a, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


@torchsymbol(torch.nn.functional.avg_pool2d, id="torch.nn.functional.avg_pool2d", is_method=False)
def avg_pool2d(
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: NumberLike | None = None,
) -> TensorProxy:
    return _avg_pool_helper(2, a, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


@torchsymbol(torch.nn.functional.avg_pool3d, id="torch.nn.functional.avg_pool3d", is_method=False)
def avg_pool3d(
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: NumberLike | None = None,
) -> TensorProxy:
    return _avg_pool_helper(3, a, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


@torchsymbol(
    torch.nn.functional.adaptive_avg_pool2d, id="torch.nn.functional.adaptive_avg_pool2d", is_method=False, is_prim=True
)
def adaptive_avg_pool2d(
    a: TensorProxy,
    /,
    output_size: int | Sequence[int],
) -> TensorProxy:
    from thunder.core.baseutils import check_valid_shape, check_valid_length

    utils.check_type(output_size, (int, IntegerProxy, Sequence))
    if isinstance(output_size, Sequence):
        utils.check(len(output_size) == 2, lambda: f"adaptive_avg_pool2d: output_size must be 2")
        utils.check_types(output_size, (int, IntegerProxy))
        check_valid_shape(output_size)
    else:
        check_valid_length(output_size)
        output_size = (output_size, output_size)

    utils.check_type(a, TensorProxy)
    a_ndim = a.ndim
    utils.check(
        a_ndim == 3 or a_ndim == 4,
        lambda: f"adaptive_avg_pool2d: Expected 3D or 4D tensor, but got {a.shape}",
    )
    for i in (-2, -1):
        utils.check(
            a.shape[i] > 0,
            lambda: f"adaptive_avg_pool2d: Expected input to have non-zero size for non-batch dimensions, but input has sizes {a.shape} with dimension {i + a_ndim} being empty",
        )
    output_shape_ = a.shape[:-2] + tuple(output_size)
    return TensorProxy(like=a, shape=output_shape_)


@torchsymbol(
    torch.ops.aten._adaptive_avg_pool2d_backward,
    "adaptive_avg_pool2d_backward",
    id="adaptive_avg_pool2d_backward",
    is_prim=True,
)
def adaptive_avg_pool2d_backward(g: TensorProxy, a: TensorProxy, /) -> TensorProxy:
    # Followed the cuda implementation in Pytorch for adaptive_avg_pool2d_backward here
    # short cut for empty tensor
    if 0 in a.shape:
        return TensorProxy(like=a)
    utils.check_type(g, TensorProxy)
    utils.check_type(a, TensorProxy)
    utils.check_same_device(g, a)
    utils.check_same_dtype(g, a)
    grad_ndim = g.ndim
    utils.check(
        grad_ndim == 3 or grad_ndim == 4,
        lambda: f"adaptive_avg_pool2d_backward: Expected 3D or 4D tensor, but got {g.shape}",
    )
    for i in range(1, grad_ndim):
        utils.check(
            g.shape[i] > 0,
            lambda: f"adaptive_avg_pool2d_backward: Expected grad to have non-zero size for non-batch dimensions, but grad has sizes {g.shape} with dimension {i} being empty",
        )
    return TensorProxy(like=a)


@torchsymbol(torch.max_pool1d, torch.nn.functional.max_pool1d, id="torch.nn.functional.max_pool1d", is_method=False)
def max_pool1d(
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> TensorProxy:
    return _max_pool_helper(1, a, kernel_size, stride, padding, dilation, return_indices, ceil_mode)


@torchsymbol(torch.max_pool2d, torch.nn.functional.max_pool2d, id="torch.nn.functional.max_pool2d", is_method=False)
def max_pool2d(
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> TensorProxy:
    return _max_pool_helper(2, a, kernel_size, stride, padding, dilation, return_indices, ceil_mode)


@torchsymbol(torch.max_pool3d, torch.nn.functional.max_pool3d, id="torch.nn.functional.max_pool3d", is_method=False)
def max_pool3d(
    a: TensorProxy,
    /,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> TensorProxy:
    return _max_pool_helper(3, a, kernel_size, stride, padding, dilation, return_indices, ceil_mode)


@torchsymbol(torch.conv1d, torch.nn.functional.conv1d, id="torch.nn.functional.conv1d", is_method=False)
def conv1d(
    a: TensorProxy,
    /,
    weight: TensorProxy,
    bias: TensorProxy | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int = 1,
    groups: int = 1,
) -> TensorProxy:
    return _conv_helper(1, a, weight, bias, stride, padding, dilation, groups)  # means 1D convolution


@torchsymbol(torch.conv2d, torch.nn.functional.conv2d, id="torch.nn.functional.conv2d", is_method=False)
def conv2d(
    a: TensorProxy,
    /,
    weight: TensorProxy,
    bias: TensorProxy | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int = 1,
    groups: int = 1,
) -> TensorProxy:
    return _conv_helper(2, a, weight, bias, stride, padding, dilation, groups)  # means 2D convolution


@torchsymbol(torch.conv3d, torch.nn.functional.conv3d, id="torch.nn.functional.conv3d", is_method=False)
def conv3d(
    a: TensorProxy,
    /,
    weight: TensorProxy,
    bias: TensorProxy | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | str = 0,
    dilation: int = 1,
    groups: int = 1,
) -> TensorProxy:
    return _conv_helper(3, a, weight, bias, stride, padding, dilation, groups)  # means 3D convolution


def _dropout_helper(a, p):
    """Helper function for all dropout-type operators. During training, some of the elements of the input tensor are
    randomly masked.

    Returns the masked tensor of the boolean values.
    """

    r = uniform_like(a, 0.0, 1.0)
    result = r < p

    return result


@torchsymbol(torch.nn.functional.cross_entropy)
def cross_entropy(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike = None,
    size_average: None | Any = None,
    ignore_index: int = -100,
    reduce: None | Any = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> TensorLike:
    utils.check(
        size_average is None and reduce is None,
        lambda: f"Deprecated size_average={size_average} and reduce={reduce} is not supported!",
    )

    _cross_entropy_input_checks(a, target, weight, ignore_index, reduction, label_smoothing)

    # class dimension is either the first one if no batch dim present (i.e. a.shape[0]),
    # or right next to it (i.e. a.shape[1]).
    class_dim = 1 if a.ndim >= 2 else 0

    # NOTE This short-circuit is subject to change and is placed ahead of other input checks to match PyTorch behavior.
    # The expected behavior when the target and input have zero elements:
    #   reduction = 'none' --- tensor([], shape) or tensor(0.)
    #   reduction = 'sum'  --- tensor(0.)
    #   reduction = 'mean' --- tensor(nan)
    # Mean reduction on empty tensors produces NaN.
    if a.numel() == 0:
        if reduction == "none":
            output_shape = list(a.shape)
            output_shape.pop(class_dim)
            return full(output_shape, 0.0, device=a.device, dtype=a.dtype)
        elif reduction == "sum":
            return full(result_shape := [], fill_value := 0.0, device=a.device, dtype=a.dtype)
        elif reduction == "mean":
            return full(result_shape := [], fill_value := float("nan"), device=a.device, dtype=a.dtype)

    if a.shape == target.shape:
        return _cross_entropy_loss_probability_target(a, target, weight, ignore_index, reduction, label_smoothing)
    elif label_smoothing != 0.0:
        return _cross_entropy_loss_label_smoothing(a, target, weight, ignore_index, reduction, label_smoothing)
    else:
        log_softmax_input = log_softmax(a, dim=class_dim)
        return nll_loss(log_softmax_input, target, weight, ignore_index, reduction)


def _cross_entropy_input_checks(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike,
    ignore_index: int,
    reduction: str,
    label_smoothing: float,
):
    utils.check(
        reduction in ("none", "sum", "mean"),
        lambda: f'Expected reduction string to be "none", "sum", or "mean", but it is {reduction}.',
        exception_type=ValueError,
    )

    utils.check(
        a.ndim >= 1,
        lambda: f"Expected the input tensor to have more than 1 dimension, but it has {a.ndim} dimensions.",
    )

    utils.check(
        label_smoothing >= 0.0 and label_smoothing <= 1.0,
        lambda: f"Expected label_smoothing to be in [0, 1] range but got {label_smoothing}.",
    )

    # class dimension is either the first one if no batch dim present (i.e. a.shape[0]),
    # or right next to it (i.e. a.shape[1]).
    class_dim = 1 if a.ndim >= 2 else 0
    num_class = a.shape[class_dim]

    utils.check(
        weight is None or (weight.ndim == 1 and weight.shape[0] == num_class),
        lambda: f"Expected a 1D tensor with {num_class} elements for weight argument, \
            but found a tensor with {weight.ndim} dimensions and {weight.shape[0]} elements.",
    )

    if a.shape != target.shape:
        utils.check(
            utils.is_integer_dtype(target.dtype),
            lambda: f"Expected target to be a tensor with an integer dtype, but it has dtype {target.dtype}.",
        )

        utils.check(
            a.ndim == target.ndim + 1,
            lambda: f"Expected the input tensor to have {(target.ndim + 1)=} dimensions, but it has {a.ndim} dimensions.",
        )

        # target should match input in dims which do not correspond to the class dim, i.e.
        # (input.shape[:class_dim] + input.shape[class_dim + 1:]) == target.shape <=> True
        expected_target_shape = a.shape[:class_dim] + a.shape[class_dim + 1 :]

        utils.check(
            expected_target_shape == target.shape,
            lambda: f"Expected the target tensor to have the same shape as the input tensor except for the class dimension \
                {expected_target_shape}, but it has shape {target.shape}.",
        )
    else:
        # target represents class probabilities and is the range [0.0, 1.0]
        utils.check(
            utils.is_float_dtype(target.dtype),
            lambda: f"Expected the target to have float dtype when target contains class probabilities \
                but it is {target.dtype}.",
        )
        utils.check(
            ignore_index < 0,
            lambda: "ignore_index argument is not supported when target contains class probabilities.",
        )


def _cross_entropy_loss_probability_target(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike,
    ignore_index: int,
    reduction: str,
    label_smoothing: float,
) -> TensorLike:
    # class dimension is either the first one if no batch dim present (i.e. a.shape[0]),
    # or right next to it (i.e. a.shape[1]).
    class_dim = 1 if a.ndim >= 2 else 0
    num_class = a.shape[class_dim]

    if label_smoothing > 0.0:
        target = (target * (1 - label_smoothing)) + (label_smoothing / num_class)

    out = log_softmax(a, dim=class_dim) * target

    if weight is not None:
        bcast_weight = reshape(weight, [num_class] + [1 for _ in range(2, a.ndim)])
        out = out * bcast_weight

    out = -out

    if reduction == "none":
        return sum(out, dim=class_dim)
    elif reduction == "sum":
        return sum(out)
    elif reduction == "mean":
        return sum(out) / (a.numel() // num_class)


def _cross_entropy_loss_label_smoothing(
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike,
    ignore_index: int,
    reduction: str,
    label_smoothing: int,
) -> TensorLike:
    # class dimension is either the first one if no batch dim present (i.e. a.shape[0]),
    # or right next to it (i.e. a.shape[1]).
    class_dim = 1 if a.ndim >= 2 else 0
    num_class = a.shape[class_dim]

    log_softmax_value = log_softmax(a, dim=class_dim)

    if weight is not None:
        bcast_weight = reshape(weight, [num_class] + [1 for _ in range(2, len(a.shape))])
        out = -(log_softmax_value * bcast_weight)
    else:
        out = -log_softmax_value

    smooth_loss = sum(out, dim=class_dim)

    # Make target broadcastable with output, which has same shape as input tensor.
    selected_target_mask = target != ignore_index
    smooth_loss = where(selected_target_mask, smooth_loss, 0)

    if reduction == "none":
        ret = smooth_loss
    elif reduction == "sum":
        ret = sum(smooth_loss)
    elif reduction == "mean":
        reduced_sum = sum(out)
        if weight is not None:
            # Gather the weights for each target class.
            # Mask the ignored target classes.
            # Sum together all target weights.
            # Make target broadcastable with output, which has same shape as input tensor.
            expanded_weight = expand(bcast_weight, a.shape)
            bcast_target = unsqueeze(target, class_dim)
            selected_weight = take_along_dim(expanded_weight, bcast_target, class_dim)
            selected_weight = where(selected_target_mask, squeeze(selected_weight), 0)
            ret = reduced_sum / sum(selected_weight)
        else:
            # The weight tensor is none, so the total weight is the number of valid target elements not equal to
            # ignore_index argument
            ret = reduced_sum / sum(selected_target_mask)

    nll_loss_value = nll_loss(log_softmax_value, target, weight, ignore_index, reduction)

    return (nll_loss_value * (1.0 - label_smoothing)) + (ret * (label_smoothing / num_class))


# TODO Is this a method?
# TODO Move this to nn.functional
# NOTE The id must be explicitly specified so as not to resolve to torch.dropout
#   (Using torch.nn.functional.dropout is just for readability as it's the documented operator)
@torchsymbol(torch.nn.functional.dropout, id="torch.nn.functional.dropout")
def dropout(a: TensorProxy, /, p: NumberLike = 0.5, training: bool = True, inplace: bool = False) -> TensorProxy:
    if inplace:
        raise NotImplementedError("Only inplace=False is currently supported in dropout")

    if not training:
        return a

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

    out = a * dropout_mask * scale
    if inplace:
        return prims.copy_(out, a)
    return out


_inplace_to_out_of_place[dropout] = dropout, 3


@torchsymbol(torch.nn.functional.embedding, id="torch.nn.functional.embedding")
def embedding(
    a: TensorLike, /, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
) -> TensorLike:
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
    flatten_indices = reshape(a, [a.numel()])
    flatten_output = clang.take(weight, flatten_indices, 0)
    return reshape(flatten_output, output_shape)


@torchsymbol(torch.ops.aten.embedding_backward)
def embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    result = prims.embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse)
    return result


@torchsymbol(torch.nn.functional.one_hot, id="torch.nn.functional.one_hot", is_method=False)
def one_hot(a: TensorLike, /, num_classes: int) -> TensorLike:
    # TODO: refactor when we're ready to support auto-inference for `num_classes = -1` using `.item()`
    utils.check(
        num_classes >= 1,
        lambda: f"Currently supports only positive input for num_classes, got num_classes={num_classes}",
        exception_type=NotImplementedError,
    )
    # TODO: would we want to implement this check in the future?
    #  utils.check(a.any() >= 0, lambda f"input tensor should have non-negative values", exception_type=ValueError)

    canvas = zeros(*a.shape, num_classes, device=a.device, dtype=dtypes.int64)
    index = a.unsqueeze(-1)
    src = ones_like(index, device=a.device, dtype=dtypes.int64)

    return scatter_add(canvas, dim=-1, index=index, src=src)


@torchsymbol(torch.group_norm, torch.nn.functional.group_norm, id="torch.nn.functional.group_norm", is_method=False)
def group_norm(
    a: TensorProxy,
    /,
    num_groups: int,
    weight: None | TensorProxy = None,
    bias: None | TensorProxy = None,
    eps: float = 1e-5,
) -> TensorProxy:
    utils.check(a.ndim >= 2, lambda: f"group_norm: {a.ndim=} should be at least 2")

    batch_size, num_channels, *inner_dims = a.shape

    # To avoid division by zero in the check below.
    utils.check(num_groups > 0, lambda: f"group_norm: {num_groups=} should be greater than 0")
    utils.check(
        num_channels % num_groups == 0, lambda: f"group_norm: {num_channels=} should be divisible by {num_groups=}"
    )
    utils.check(
        weight is None or (weight.ndim == 1 and weight.numel() == num_channels),
        lambda: f"group_norm: {weight.ndim=} should be equal to 1 and {weight.numel=} to {num_channels=}",
    )
    utils.check(
        bias is None or (bias.ndim == 1 and bias.numel() == num_channels),
        lambda: f"group_norm: {bias.ndim=} should be equal to 1 and {bias.numel=} to {num_channels=}",
    )

    # Empty `a` implies empty result.
    if any(d == 0 for d in a.shape):
        return zeros_like(a)

    # Split channels (num_channels,) -> (num_groups, num_channels // num_groups).
    a_groupped = view(a, [batch_size, num_groups, num_channels // num_groups] + inner_dims)

    # Perform Normalization (yes, subtract mean, divide by sd) over all the dims
    # but the batch and the group dim.
    res, *_ = _normalize(a_groupped, norm_dims=range(2, a_groupped.ndim), eps=eps)
    # Restore the channel dimension
    res = view(res, a.shape)

    # Reshape weight/bias (they are usually learnable parameters)
    # to be broadcastable with the current `res`.
    params_shape = [1, num_channels] + [1 for i in range(2, a.ndim)]
    weight = view(weight, params_shape) if weight is not None else None
    bias = view(bias, params_shape) if bias is not None else None

    if weight is not None:
        res = res * weight
    if bias is not None:
        res = res + bias

    res = to(res, a.dtype)
    return res


def _interpolate_scale_factor_helper(
    a: TensorLike,
    scale_factor: Sequence[float] | float,
    mode: str = "nearest",
) -> TensorLike:
    assert mode == "nearest"

    # a is assumed to be at least 3D.
    batch, channels, *spatial_dims = a.shape
    dim = len(spatial_dims)

    if isinstance(scale_factor, (float, FloatProxy)):
        utils.check(scale_factor > 0, lambda: f"{scale_factor=} is expected to be strictly positive")
        scale_factor = (scale_factor,) * dim
    else:
        utils.check(
            (
                isinstance(scale_factor, Sequence)
                and len(scale_factor) == dim
                and all(isinstance(s, (float, FloatProxy)) and s > 0 for s in scale_factor)
            ),
            lambda: f"{scale_factor=} is expected to be a strictly positive floating point number or "
            f"a sequence of strictly positive floating point numbers of length {dim}",
        )

    # perform nearest up/down-sampling
    def nearest_sampler(t, input_dim, output_dim, *, scale, dim):
        # It is expected that output_dim = int(input_dim * scale).
        # Indices [0, ..., output_dim - 1] are mapped to [0, ..., input_dim - 1]
        # with the rule i -> int(i * scale)
        # Values at these indices is the result.
        selected_idx = arange(0, output_dim, device=a.device)
        selected_idx = to(selected_idx * scale, selected_idx.dtype)
        return clang.take(t, selected_idx, dim=dim)

    def dim_expander(t, dim, n_repeats):
        t = unsqueeze(t, dim + 1)
        t = expand(t, t.shape[: dim + 1] + (n_repeats,) + t.shape[dim + 2 :])
        return t

    res_output_spatial_dims = []

    for k, (scale, input_dim) in enumerate(zip(reversed(scale_factor), reversed(spatial_dims))):
        output_dim = int(scale * input_dim)
        utils.check(
            output_dim > 0,
            lambda: f"provided scale_factor value {scale} results " f"in a zero length output at dimension {k + 2}",
        )
        res_output_spatial_dims.append(output_dim)

        # k iterates from the end, and we skip the first 2
        # dimenions corresponding to batches and channels.
        curr_dim = 2 + (len(spatial_dims) - k - 1)

        if output_dim <= input_dim:
            if output_dim <= input_dim // 2:
                # scale_factor <= 1 (i.e. output_dim <= input_dim) implies simple slice
                # when output_dim <= input_dim // 2.
                stride = input_dim // output_dim
                end = input_dim - (input_dim % output_dim)
                a = clang.slice_in_dim(a, 0, end, stride=stride, dim=curr_dim)
            else:
                # In this case slice will not do and explicit downsample is needed.
                a = nearest_sampler(a, input_dim, output_dim, scale=1.0 / scale, dim=curr_dim)
        else:
            if output_dim % input_dim == 0:
                # In this case we can just expand dim.
                n_repeats = output_dim // input_dim
                a = dim_expander(a, curr_dim, n_repeats)
            else:
                # In this case expand will not cut it and explicit upsampling is needed.
                a = nearest_sampler(a, input_dim, output_dim, scale=1.0 / scale, dim=curr_dim)

    output_shape = [batch, channels] + res_output_spatial_dims[::-1]
    return reshape(a, output_shape)


def _interpolate_size_helper(
    a: TensorLike,
    size: Sequence[int] | int,
    mode: str = "nearest",
) -> TensorLike:
    batch, channels, *spatial_dims = a.shape
    dim = len(spatial_dims)

    if isinstance(size, (int, IntegerProxy)):
        utils.check(size > 0, lambda: f"{size=} is expected to be greater than zero")
        size = (size,) * dim
    else:
        utils.check(
            (
                isinstance(size, Sequence)
                and len(size) == dim
                and all(isinstance(s, (int, IntegerProxy)) and s > 0 for s in size)
            ),
            lambda: f"{size=} is expected to be a greater than zero integer "
            f"or a sequence of strictly positive integers of length {dim}",
        )

    scale_factor = tuple(output_size / input_size for output_size, input_size in zip(size, spatial_dims))

    return _interpolate_scale_factor_helper(a, scale_factor)


# TODO Implement additional modes and parameters
@torchsymbol(torch.nn.functional.interpolate, is_method=False)
def interpolate(
    a: TensorLike,
    /,
    size: None | int | Sequence[int] = None,
    scale_factor: float | Sequence[float] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
    antialias: bool = False,
) -> TensorLike:
    utils.check(
        mode == "nearest",
        lambda: f"only mode='nearest' is supported at the moment, but got {mode=}",
        exception_type=NotImplementedError,
    )

    utils.check(a.ndim >= 3, lambda: f"Expected {a.ndim=} >= 3")
    utils.check(a.numel() > 0, lambda: f"Expected {a.numel=} to be greater than 0")
    utils.check(
        align_corners == None,
        lambda: f"Thunder does not yet support 'align_corners'.",
        exception_type=NotImplementedError,
    )
    utils.check(
        recompute_scale_factor is None or recompute_scale_factor == False,
        lambda: f"Thunder does not yet support 'recompute_scale_factor=True'.",
        exception_type=NotImplementedError,
    )
    utils.check(
        antialias == False,
        lambda: f"Thunder does not yet support 'antialias=True'.",
        exception_type=NotImplementedError,
    )

    utils.check(
        (size is not None) ^ (scale_factor is not None),
        lambda: "Only one of `size` or `scale_factor` has to be specified, but " f"got {size=} and {scale_factor=}",
    )

    if size is not None:
        return _interpolate_size_helper(a, size, mode)
    else:
        return _interpolate_scale_factor_helper(a, scale_factor, mode)


@torchsymbol(torch.Tensor.item, is_method=True)
def item(a: TensorLike) -> Number:
    return prims.item(a)


# PyTorch does not support backward for torch.item
register_grad(item.id, item)


# TODO Move this to nn.functional
@torchsymbol(torch.nn.functional.linear)
def linear(a: TensorLike, w: TensorLike, /, bias: None | TensorLike = None) -> TensorLike:
    return prims.linear(a, w, bias)


@torchsymbol(torch.logsumexp, is_method=True)
def logsumexp(a: TensorLike, /, dim: int | Sequence[int], keepdim: bool = False) -> TensorLike:
    input_max = amax(a, dim, keepdim=True)
    input_max_sans_inf = where(abs(input_max) == float("inf"), 0, input_max)
    result = log(sum(exp(a - input_max_sans_inf), dim, keepdim))
    squeeze_max = input_max_sans_inf if keepdim else squeeze(input_max_sans_inf, dim)
    return result + squeeze_max


# The dim parameter in torch.nn.functional.log_softmax is optional.
# Inferring dim parameter is deprecated, so we made dim a required parameter in our log_softmax definition.
# See the PyTorch documentation:
# https://pytorch.org/docs/master/generated/torch.nn.functional.log_softmax.html
# https://pytorch.org/docs/master/special.html?#torch.special.log_softmax
@torchsymbol(torch.log_softmax, torch.special.log_softmax, torch.nn.functional.log_softmax, is_method=True)
def log_softmax(a: TensorLike, /, dim: int, *, dtype: None | dtypeLike = None) -> TensorLike:
    result_dtype: dtypeLike = dtype or a.dtype
    result_dtype: dtypes.dtype = to_dtype(result_dtype)

    # If dtype parameter is specified, the input tensor is cast to dtype before the operation is performed.
    # We cast the input to the corresponding computation dtype and the output to the desired dtype.
    computation_dtype = utils.get_computation_dtype(result_dtype)
    a_ = a.to(computation_dtype)

    result = a_ - logsumexp(a_, dim, keepdim=True)

    converted = result.to(result_dtype)
    return converted


# TODO Update annotations and consider moving to torchex
# We improve the efficiency of cross_entropy backward decomposition by adding the log_softmax_backward
# and nll_loss_backward primitives. Executors can override the primitives using internal implementations.
# See issue "Cross_entropy is decomposed for backward but the decomposition is
# not fusible currently"
@torchsymbol("log_softmax_backward", id="log_softmax_backward")
def log_softmax_backward(g: TensorProxy, /, output: TensorProxy, dim: int, dtype: dtypeLike) -> TensorLike:
    dtype: dtypes.dtype = to_dtype(dtype)
    g_input = g - exp(output) * sum(g, dim=dim, keepdim=True)
    return to(g_input, dtype)


# This helper function implements the aten nll_loss_forward, which returns the primal and total_weight tensors.
# The total_weight tensor is used in the backward pass.
def _nll_loss_helper(
    a: TensorProxy, target: TensorProxy, weight: None | TensorProxy, ignore_index: int, reduction: str
) -> tuple[TensorProxy, None | TensorLike]:
    utils.check(
        reduction in ("none", "sum", "mean"),
        lambda: f'Expected reduction string to be "none", "sum", or "mean", but it is {reduction}.',
        exception_type=ValueError,
    )

    # NOTE This short-circuit is subject to change and is placed ahead of other input checks to match PyTorch behavior.
    # The expected behavior when the target and input have zero elements:
    #   reduction = 'none' --- tensor([], shape) or tensor(0.)
    #   reduction = 'sum'  --- tensor(0.)
    #   reduction = 'mean' --- tensor(nan)
    # Mean reduction on empty tensors produces NaN.
    if a.numel() == 0 and target.numel() == 0:
        if reduction == "none":
            # Keep target shape if it is non-trivial
            result_shape = target.shape if target.shape != (0,) else []
            return full(result_shape, fill_value := 0.0, device=a.device, dtype=a.dtype), None
        elif reduction == "sum":
            return full(result_shape := [], fill_value := 0.0, device=a.device, dtype=a.dtype), None
        elif reduction == "mean":
            return full(result_shape := [], fill_value := float("nan"), device=a.device, dtype=a.dtype), None

    utils.check(
        utils.is_integer_dtype(target.dtype),
        lambda: f"Expected target to be a tensor with an integer dtype, but it has dtype {target.dtype}.",
    )

    utils.check(
        a.ndim >= 1,
        lambda: f"Expected the input tensor to have more than 1 dimension, but it has {a.ndim} dimensions.",
    )

    utils.check(
        a.ndim == target.ndim + 1,
        lambda: f"Expected the input tensor to have {(target.ndim + 1)=} dimensions, but it has {a.ndim} dimensions.",
    )

    # class dimension is either the first one if no batch dim present (i.e. a.shape[0]),
    # or right next to it (i.e. a.shape[1]).
    class_dim = 1 if a.ndim >= 2 else 0
    num_class = a.shape[class_dim]
    # target should match input in dims which do not correspond to the class dim, i.e.
    # (input.shape[:class_dim] + input.shape[class_dim + 1:]) == target.shape <=> True
    expected_target_shape = a.shape[:class_dim] + a.shape[class_dim + 1 :]

    utils.check(
        expected_target_shape == target.shape,
        lambda: f"Expected the target tensor to have the same shape as the input tensor except for the class dimension \
            {expected_target_shape}, but it has shape {target.shape}.",
    )

    utils.check(
        weight is None or (weight.ndim == 1 and weight.shape[0] == num_class),
        lambda: f"Expected a 1D tensor with {num_class} elements for weight argument, \
            but found a tensor with {weight.ndim} dimensions and {weight.shape[0]} elements.",
    )

    # NOTE: [Handling of 'ignore_index' parameter]
    # What does it mean to ignore an index?
    #   The 'ignore_index' parameter specifies a target value that does not contribute to input gradient.
    # 'ignore_index' can be outside of the [0, num_class) range, which can cause out-of-bounds errors when gathering
    # values from input tensor.
    #
    # What does ATen do?
    #   ATen prevents nll_loss from having these errors by skipping target values that match ignore_index first before
    # indexing the input tensor.
    #
    # What do we do?
    #   We mask the ignore_index entries on the output tensor from take_along_axis because we expect the targets to be
    # within [0, num_class) range.
    #
    # Why do we like our approach better?
    #   Mimicking Aten behavior requires masking the target tensor before calling take_along_axis, which would add more
    # operations to the fusion. We should follow this approach until we see real examples where ignore_index is
    # out-of-bounds of [0, num_class) range.
    #
    # What are the alternative options?
    #   We can add a `mode` parameter to take_along_axis that controls how to handle out-of-bounds indices.
    # The jax.numpy.take_along_axis has this feature.

    out = -a

    if weight is not None:
        bcast_weight = reshape(weight, [num_class] + [1 for _ in range(2, a.ndim)])
        out = out * bcast_weight

    # Make target broadcastable with output, which has same shape as input tensor.
    bcast_target = unsqueeze(target, class_dim)

    out = take_along_dim(out, bcast_target, class_dim)
    selected_target_mask = bcast_target != ignore_index
    out = where(selected_target_mask, out, 0)

    # This section handles applying the reduction parameter to the output.
    # We return None for the total_weight when reduction is "none" or "sum" since it is unused in the backwards pass.
    if reduction == "none":
        return squeeze(out, class_dim), None
    elif reduction == "sum":
        return sum(out), None
    elif reduction == "mean":
        reduced_sum = sum(out)
        if weight is not None:
            # Gather the weights for each target class.
            # Mask the ignored target classes.
            # Sum together all target weights.
            expanded_weight = expand(bcast_weight, a.shape)
            selected_weight = take_along_dim(expanded_weight, bcast_target, class_dim)
            selected_weight = where(selected_target_mask, selected_weight, 0)
            bcast_weight_sum = sum(selected_weight)
            return (reduced_sum / bcast_weight_sum), bcast_weight_sum
        else:
            # The weight tensor is none, so the total weight is the number of valid target elements not equal to
            # ignore_index argument
            total_weight = sum(selected_target_mask)
            out = reduced_sum / total_weight
            return out, total_weight


# Aten nll_loss_forward returns primal and total_weight tensors. The total_weight tensor is used in the backwards pass.
# PyTorch nll_loss only returns primal, so a helper function is used in the augmented forward function.
@torchsymbol(torch.nn.functional.nll_loss)
def nll_loss(
    a: TensorProxy,
    /,
    target: TensorProxy,
    weight: None | TensorProxy = None,
    ignore_index: int = None,
    reduction: str = "mean",
) -> TensorProxy:
    # Resolve ignore_index if it is not specified by user.
    if ignore_index is None:
        ignore_index = -1
    result, _ = _nll_loss_helper(a, target, weight, ignore_index, reduction)
    return result


# TODO Make not a prim
# The decomposition of `nll_loss_backward` requires a scatter operation
@torchsymbol("nll_loss_backward", id="nll_loss_backward", is_prim=True)
def nll_loss_backward(
    g: TensorLike,
    a: TensorLike,
    /,
    target: TensorLike,
    weight: None | TensorLike,
    reduction: str,
    ignore_index: int,
    total_weight: TensorLike,
) -> TensorLike:
    return TensorProxy(like=g, shape=a.shape)


@torchsymbol(torch.nn.functional.mse_loss)
def mse_loss(
    a: TensorLike,
    /,
    target: TensorLike,
    size_average: None | Any = None,
    reduce: None | Any = None,
    reduction: str = "mean",
) -> TensorLike:
    utils.check(
        size_average is None and reduce is None,
        lambda: f"Deprecated size_average={size_average} and reduce={reduce} is not supported!",
    )
    utils.check(
        reduction in ("none", "sum", "mean"),
        lambda: f'Expected reduction string to be "none", "sum", or "mean", but it is {reduction}.',
        exception_type=ValueError,
    )

    # warn broadcasting
    if a.size() != target.size():
        warnings.warn(
            f"Using a target size {target.size()} that is different to the input size {a.size()}"
            "This will likely lead to incorrect results due to broadcasting."
            "Please ensure they have the same size."
        )
    out = (a - target) ** 2

    # maybe add _apply_loss_reduction
    # (like https://github.com/pytorch/pytorch/blob/df5829d0babaefc6e271897d6fffd40073d8b723/torch/_refs/nn/functional/__init__.py#L490)
    # not sure if this would be useful
    if reduction == "none":
        return out
    elif reduction == "sum":
        return sum(out)
    elif reduction == "mean":
        return mean(out)
    else:
        raise ValueError(f"Reduction argument {reduction} to mse_loss is not supported")


# TODO Add annotations
# NOTE The scale parameter is kwarg-only in PyTorch
@torchsymbol(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None):
    for arg_name, arg in zip(("query", "key", "value"), (query, key, value)):
        utils.check(
            dtypes.is_float_dtype(arg.dtype),
            lambda: f"{arg_name}.dtype={arg.dtype} is expected to be a floating type",
            ValueError,
        )

    # Reference implementation:
    # https://github.com/pytorch/pytorch/blob/d62a80a/aten/src/ATen/native/transformers/attention.cpp#L639-L697
    if scale is None:
        scale = 1 / query.size(-1) ** 0.5
    # This implementation doesn't match your usual attention textbook formula, but it's meant to be more stable
    # https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118
    scale = scale**0.5
    logits = (query * scale) @ (key.transpose(-2, -1) * scale)
    if is_causal:
        utils.check(
            attn_mask is None,
            lambda: "scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True",
            ValueError,
        )
        logits = logits.tril(fill_value=-math.inf)

    if attn_mask is not None:
        if dtypes.is_boolean_dtype(attn_mask.dtype):
            # Boolean mask value False implies logit value set to -inf
            # to have no effect in the subsequent softmax
            logits = where(attn_mask, logits, -math.inf)
        elif dtypes.is_float_dtype(attn_mask.dtype):
            # Otherwise, attn_mask represents an additive attention tensor
            logits = logits + attn_mask
        else:
            utils.check(
                False, lambda: f"{attn_mask.dtype=} is expected to be of the boolean or a floating type", ValueError
            )

    attn_weight = softmax(logits, dim=-1)
    attn_weight = dropout(attn_weight, dropout_p)
    return attn_weight @ value


@torchsymbol(torch.sigmoid, torch.nn.functional.sigmoid, torch.special.expit, is_method=True)
def sigmoid(a: TensorLike, /) -> TensorLike:
    return clang.sigmoid(a)


# CompositeImplicitAutograd - don't register decomp
@torchsymbol(torch.softmax, torch.nn.functional.softmax, is_method=True, id="torch.softmax")
def _softmax(
    a: TensorLike,
    /,
    dim: int,
    *,
    dtype: None | dtypeLike = None,
) -> TensorLike:
    result_dtype: dtypeLike = dtype or a.dtype
    result_dtype: dtypes.dtype = to_dtype(result_dtype)
    computation_dtype = utils.get_computation_dtype(result_dtype)
    a_ = a.to(computation_dtype)

    if a.numel == 0:
        a_exp = exp(a_)
    else:
        a_max = amax(a_, dim, keepdim=True)
        a_exp = exp(a_ - a_max)

    result = a_exp / sum(a_exp, dim, keepdim=True)
    converted = result.to(result_dtype)
    return converted


register_method("softmax", _softmax)


# A wrapper to support `torch.nn.Softmax` whose `forward` passes the kwarg of `_stacklevel=5` to `torch.nn.functional.softmax`.
# ref: https://github.com/pytorch/pytorch/blob/8d12ba9acfa20ed7df438a8892c9bf8e6bef5775/torch/nn/modules/activation.py#L1545
def softmax(a: TensorLike, dim: int, dtype: None | dtypeLike = None, _stacklevel: int = 3) -> TensorLike:
    return _softmax(a, dim=dim, dtype=dtype)


def torch_device(type: DeviceLike, index: int | None = None) -> devices.Device:
    if isinstance(type, (devices.Device, torch.device)):
        # PyTorch behavior:
        # >>> torch.device(torch.device("cuda"), 0)
        # TypeError: device(): argument 'type' (position 1) must be str, not torch.device
        utils.check(index is None, lambda: f"device(): `index` is only allowed when `device` is a `str`.")
        return to_device(type)

    # NOTE: device_or_str is `str`
    if index is not None:
        # PyTorch behavior:
        # >>> torch.device("cuda:0", 0)
        # RuntimeError: type (string) must not include an index because index was passed explicitly: cuda:0
        has_device_idx = len(type.split(":")) > 1
        utils.check(
            not has_device_idx,
            lambda: f"device string must not include an index because index was passed explicitly: {type}",
        )
        if isinstance(index, NumberProxy):
            index.make_static_constrained()
            prims.sink(index)
            index = index.value

    return devices.Device(type, index)


# We don't use @torchsymbol as we don't want `torch.device()` to appear in trace as a symbol.
# Because of this, we need to manually register the implementation.
register_function(torch.device, torch_device)


# Tag to use on Proxies created in `no_grad` regions.
# VJP transform will treat BoundSymbol's whose output has these tags
# as constant.
ProxyTag.register_tag("DETACHED_AUTOGRAD_GRAPH")


# This is just a marker Symbol. `tag_no_grad_symbols_pass` pass uses these symbols
# to find the `no_grad` regions and mark the BoundSymbols within them as constant
# for VJP using the `DETACHED_AUTOGRAD_GRAPH` tag.
@torchsymbol(torch._C._set_grad_enabled, id="set_grad_enabled", tags=(prims.OpTags.CTX_MANAGER_ENTER_EXIT_OP,))
def _set_grad_enabled_with_warning(enabled: bool) -> None:
    cd = get_compile_data()
    if cd is None:
        warnings.warn(
            "torch.enable_grad/torch.no_grad/torch._C._set_grad_enabled have no effect, use thunder.jit for correct behaviour"
        )
        return
    get_compile_data().is_grad_enabled = enabled


def _unwrap_if_dead(tensor):
    return tensor


register_function(torch._C._functorch.unwrap_if_dead, _unwrap_if_dead)


@torchsymbol(
    torch.utils.checkpoint.checkpoint,
    torch.ops.higher_order.tag_activation_checkpoint,
    id="activation_checkpoint",
)
def checkpoint(
    function: Callable[..., TensorLike],
    *args: TensorLike,
    context_fn: None | Callable[..., Any] = None,
    debug: None | bool = None,
    determinism_check: None | str = None,
    preserve_rng_state: None | bool = None,
    use_reentrant: bool = False,
    **kwargs: Any,
) -> TensorLike:
    utils.check(
        not use_reentrant,
        lambda: "torch.checkpoint: use_reentrant=True is not supported in Thunder",
    )
    # NOTE: Thunder currently ignores the context_fn, debug, determinism_check, preserve_rng_state arguments
    # Let's raise a warning if any of these arguments are passed
    if context_fn is not None:
        warnings.warn("torch.checkpoint: context_fn is not supported in Thunder and will be ignored")
    if debug is not None:
        warnings.warn("torch.checkpoint: debug is not supported in Thunder and will be ignored")
    if determinism_check is not None:
        warnings.warn("torch.checkpoint: determinism_check is not supported in Thunder and will be ignored")
    if preserve_rng_state is not None:
        warnings.warn("torch.checkpoint: preserve_rng_state is not supported in Thunder and will be ignored")
    return function(*args, **kwargs)


@register_augmented_forward(
    "activation_checkpoint",
)
def _augmented_forward_checkpoint(
    function: Callable[..., TensorLike],
    *args: TensorLike,
    context_fn: None | Callable[..., Any] = None,
    debug: None | bool = None,
    determinism_check: None | str = None,
    preserve_rng_state: None | bool = None,
    use_reentrant: bool = False,
    **kwargs: Any,
) -> TensorLike:
    result = function(*args, **kwargs)
    saved_for_backward = (function, args, kwargs)
    return result, saved_for_backward


@register_backward(
    "activation_checkpoint",
)
def _backward_checkpoint(
    function,
    args,
    kwargs,
    *grad_outputs,
) -> tuple[None | TensorLike, ...]:
    from thunder.core.transforms import vjp

    _, grads = vjp(function)(args, grad_outputs, **kwargs)
    return grads


#
# Distributed operations
#
# NOTE DISTRIBUTED AVAILABILITY
# PyTorch is often built without distributed support, which can be queried for using
#   torch.distributed.is_available(). When PyTorch is built without distributed then we
#   want to avoid accessing any parts of the torch.distributed module except
#   the is_available() function.

if torch.distributed.is_available():
    DistributedReduceOpLike = str | torch.distributed.ReduceOp | dist_prims.DistributedReduceOps

    # string name, PyTorch enum value, thunder.jit enum value
    _reduceop_triples = (
        ("sum", torch.distributed.ReduceOp.SUM, dist_prims.DistributedReduceOps.SUM),
        ("max", torch.distributed.ReduceOp.MAX, dist_prims.DistributedReduceOps.MAX),
    )

    def to_thunder_distributed_reduce_op(op: DistributedReduceOpLike | None):
        if isinstance(op, str):
            for s, top, pop in _reduceop_triples:
                if op == s:
                    return pop

            utils.check(False, lambda: f"Unknown distributed reduce op string {op}")

        if isinstance(op, torch.distributed.ReduceOp):
            for s, top, pop in _reduceop_triples:
                if op is top:
                    return pop

            utils.check(False, lambda: f"Unknown distributed reduce op {op}")

        if op is None:
            return dist_prims.DistributedReduceOps.SUM

        return op

    def to_torch_distributed_reduce_op(op: None | DistributedReduceOpLike) -> None | torch.distributed.ReduceOp:
        if isinstance(op, dist_prims.DistributedReduceOps):
            for s, top, pop in _reduceop_triples:
                if op is pop:
                    return top

            utils.check(False, lambda: f"Couldn't map the distributed reduce op {op} to a PyTorch reduce op")

        return op

    @torchsymbol(
        is_method=False,
        id="functional_all_gather",
    )
    def all_gather(
        a: TensorLike,
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
        dim: int | None = None,
    ) -> TensorLike | FutureTensorLike:
        group = group if group is not None else torch.distributed.new_group()

        return dist_prims.all_gather(a, group, async_op, dim=dim)

    @torchsymbol(
        torch.distributed.all_gather_into_tensor,
        is_method=False,
        id="all_gather_",
        tags=(prims.OpTags.IN_PLACE, prims.OpTags.DONT_DCE),
    )
    def all_gather_(
        output_tensor: TensorLike,
        input_tensor: TensorLike,
        /,
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
    ) -> TensorLike:
        result_numel = input_tensor._numel * group.size()
        utils.check(result_numel == output_tensor._numel, lambda: f"{output_tensor._numel=} should be {result_numel=}")
        group = group if group is not None else torch.distributed.new_group()
        out_or_work = dist_prims.all_gather(input_tensor, group, async_op, dim=None)
        if async_op:
            out = dist_prims.wait(out_or_work)
        else:
            out = out_or_work
        return prims.copy_(out.view(output_tensor.shape), output_tensor)

    # NOTE torch.distributed.all_reduce is an inplace operation (although the underlying NCCL
    #   call does not need to be inplace). This, however, is modeled as an out-of-place functional
    #   operation, hence the id "functional_all_reduce", and why we do not translate PyTorch
    #   calls directly to this.
    # PyTorch uses torch.ops.c10d_functional.all_reduce as an ID for a similar operation.
    # This operation is based on torch.distributed.all_reduce, see:
    #   https://pytorch.org/docs/master/distributed.html#torch.distributed.all_reduce
    @torchsymbol(
        torch.ops._c10d_functional.all_reduce,
        is_method=False,
        id="functional_all_reduce",
    )
    def all_reduce(
        a: TensorLike,
        /,
        op: DistributedReduceOpLike = torch.distributed.ReduceOp.SUM,
        group: None | torch.distributed.ProcessGroup | str = None,
        async_op: bool = False,
        **kwargs,
    ) -> TensorLike | FutureTensorLike:
        # note: torch.ops._c10d_functional takes name of group
        if isinstance(group, str):
            from torch._C._distributed_c10d import _resolve_process_group

            group = _resolve_process_group(group_name=group)
        op = to_thunder_distributed_reduce_op(op)
        group = group if group is not None else torch.distributed.new_group()

        return dist_prims.all_reduce(a, op, group, async_op)

    @torchsymbol(
        torch.distributed.all_reduce,
        is_method=False,
        id="all_reduce_",
        tags=(prims.OpTags.IN_PLACE,),
    )
    def all_reduce_(
        a: TensorLike,
        /,
        op: DistributedReduceOpLike = torch.distributed.ReduceOp.SUM,
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
    ) -> TensorLike:
        utils.check(
            not async_op,
            lambda: f"`torch.distributed.all_reduce` with {async_op=} is not supported",
            NotImplementedError,
        )
        op = to_thunder_distributed_reduce_op(op)
        group = group if group is not None else torch.distributed.new_group()

        out = dist_prims.all_reduce(a, op, group, async_op, skip_clone=True)
        return prims.copy_(out, a)

    @torchsymbol(
        is_method=False,
        id="functional_broadcast",
    )
    def broadcast(
        a: TensorLike,
        src: int,
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
    ) -> TensorLike | FutureTensorLike:
        group = group if group is not None else torch.distributed.new_group()

        return dist_prims.broadcast(a, src, group, async_op)

    @torchsymbol(
        is_method=False,
        id="functional_reduce_scatter",
    )
    def reduce_scatter(
        a: TensorLike,
        op: DistributedReduceOpLike | None = None,
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
        dim: int | None = None,
    ) -> TensorLike | FutureTensorLike:
        op = to_thunder_distributed_reduce_op(op)
        group = group if group is not None else torch.distributed.new_group()

        return dist_prims.reduce_scatter(a, op, group, async_op, dim=dim)

    @torchsymbol(
        torch.distributed.reduce_scatter_tensor,
        is_method=False,
        id="reduce_scatter_",
        tags=(prims.OpTags.IN_PLACE, prims.OpTags.DONT_DCE),
    )
    def reduce_scatter_(
        output: TensorLike,
        input: TensorLike,
        op: DistributedReduceOpLike | None = None,
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
    ) -> TensorLike:
        op = to_thunder_distributed_reduce_op(op)
        group = group if group is not None else torch.distributed.new_group()
        result_numel = input._numel // group.size()
        utils.check(result_numel == output._numel, lambda: f"{output._numel=} should be {result_numel=}")
        out_or_work = dist_prims.reduce_scatter(input, op, group, async_op, dim=None)
        if async_op:
            out = dist_prims.wait(out_or_work)
        else:
            out = out_or_work
        return prims.copy_(out.view(output.shape), output)

    @torchsymbol(
        torch.ops._c10d_functional.wait_tensor,
        is_method=True,
        id="torch.Tensor.wait",
    )
    def wait(slf: TensorLike) -> TensorLike:
        return slf

else:

    def all_gather(
        a: TensorLike,
        group: Any | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def all_gather_(
        output_tensor: TensorLike,
        input_tensor: TensorLike,
        /,
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: "torch.distributed is not available")

    # NOTE torch.distributed is not available
    def all_reduce(
        a: TensorLike,
        op: Any,
        group: Any | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def all_reduce_(
        a: TensorLike,
        /,
        op: DistributedReduceOpLike = "torch.distributed.ReduceOp.SUM",
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: "torch.distributed is not available")

    def broadcast(
        a: TensorLike,
        src: int,
        group: Any | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def reduce_scatter(
        a: TensorLike,
        op: Any,
        group: Any | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def reduce_scatter_(
        output: TensorLike,
        input: TensorLike,
        op: DistributedReduceOpLike | None = None,
        group: torch.distributed.ProcessGroup | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: "torch.distributed is not available")

    def wait(slf) -> None:
        utils.check(False, lambda: "torch.distributed is not available")


# ref: https://github.com/pytorch/pytorch/blob/b99ef1a/torch/_functorch/autograd_function.py#L715-L752
@torchsymbol(
    torch.ops.higher_order.autograd_function_apply,
    id="torch.ops.higher_order.autograd_function_apply",
    is_method=False,
)
def autograd_function_apply(
    fwd: Callable[list[TensorProxy], TensorProxy | tuple[TensorProxy, ...]],
    bwd: Callable[list[TensorProxy], TensorProxy | tuple[TensorProxy, ...]],
    *args: Any,
    args_tensor_mask: Sequence[bool] | None,
    non_differentiable_idx: Sequence[int] | None = None,
) -> TensorProxy | tuple[TensorProxy, ...]:
    result, saved_for_backward = fwd(None, *args)
    return result


@register_augmented_forward("torch.ops.higher_order.autograd_function_apply")
def augmented_forward_autograd_function_apply(
    fwd: Callable[list[Any | TensorProxy], TensorProxy | tuple[TensorProxy, ...]],
    bwd: Callable[list[Any | TensorProxy], tuple[TensorProxy, ...]],
    *args: Any,
    args_tensor_mask: Sequence[bool],
    non_differentiable_idx: Sequence[int] | None = None,
) -> tuple[TensorProxy | tuple[TensorProxy, ...], tuple[Any, ...]]:
    result, saved_for_backward = fwd(None, *args)
    return result, (saved_for_backward, bwd, args_tensor_mask, non_differentiable_idx)


@register_backward("torch.ops.higher_order.autograd_function_apply")
def backward_autograd_function_apply(
    saved_for_backward: tuple[Any, ...],
    bwd: Callable[list[Any | TensorProxy], tuple[TensorProxy, ...]],
    args_tensor_mask: Sequence[bool],
    non_differentiable_idx: Sequence[int] | None = None,
    *grad_output: Sequence[TensorProxy],
) -> tuple[Any, ...]:
    return bwd(None, *grad_output, *saved_for_backward)


@torchsymbol(
    torch.amp.autocast_mode._enter_autocast,
    id="torch.amp.autocast_mode._enter_autocast",
    tags=(prims.OpTags.DONT_DCE, prims.OpTags.CTX_MANAGER_ENTER_EXIT_OP),
)
def autocast_enter(device_type, dtype=None, enabled=True, _unused_cache_enabled=True):
    # We may receive device_type=cuda:0
    # PyTorch applies autocast irrespective of device index.
    # So, here we grab the device_type from the string.
    device_type, unused_deviceno = devices._device_from_string_helper(device_type)
    device_type = devices.devicetype_string(device_type)
    if dtype is None:
        dtype = torch.get_autocast_dtype(device_type)
    get_compile_data().autocast_stack.push(device_type, dtype, enabled)


@torchsymbol(
    torch.amp.autocast_mode._exit_autocast,
    id="torch.amp.autocast_mode._exit_autocast",
    tags=(prims.OpTags.DONT_DCE, prims.OpTags.CTX_MANAGER_ENTER_EXIT_OP),
)
def autocast_exit(*args):
    if get_compile_data().autocast_stack.is_empty():
        return
    get_compile_data().autocast_stack.pop()


#
# The automatically registered torch operators
#


def _is_differentiable(arg: Any) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(arg, (torch.Tensor, FakeTensor, TensorProxy)):
        return dtypes.is_inexact_dtype(to_dtype(arg.dtype))
    return False


def _make_differentiable_wrapper(func: Callable, *args, **kwargs):
    flat_args, spec = tree_flatten((args, kwargs))
    differentiable_args_idx = tuple(i for i, a in enumerate(flat_args) if _is_differentiable(a))
    differentiable_args = tuple(arg for i, arg in enumerate(flat_args) if i in differentiable_args_idx)

    def wrapper(*diff_args):
        diff_args_iter = iter(diff_args)
        full_args = [next(diff_args_iter) if i in differentiable_args_idx else arg for i, arg in enumerate(flat_args)]
        full_args, full_kwargs = tree_unflatten(full_args, spec)
        return func(*full_args, **full_kwargs)

    return wrapper, differentiable_args


def register_default_torch_ops():
    from thunder.torch import default_torch_ops

    for m, fns in default_torch_ops.torch_auto_registered_ops.items():
        for fn in fns:
            # Ensure no inplace op in the list
            utils.check(
                not fn.__name__.endswith("_"),
                lambda: f"Automatic registration does not support in-place op of {m.__name__}.{fn.__name__}, please manually register it",
            )
            register_default_torch_op(fn, m)


def _get_torch_function_name(torch_module: ModuleType, torchfn: Callable):
    # Handle special cases where torchfn.__name__ differs from the name used to call it in Python,
    # e.g., `torch.nn.functional.logsigmoid.__name__` is 'log_sigmoid'.
    special_cases = {torch.nn.functional.logsigmoid: "logsigmoid"}
    # Operators in the following namespace have an extra prefix in their __name__ attribute compared to their Python call name.
    # e.g., `torch.special.xlogy.__name__` is 'special_xlogy' instead of just 'xlogy'.
    name_prefix_map = {torch.special: "special_", torch.linalg: "linalg_", torch.fft: "fft_"}
    if function_name := special_cases.get(torchfn, None):
        return function_name
    if not (function_name := getattr(torchfn, "__name__", None)):
        raise RuntimeError(
            f"The function {torchfn} from the module {torch_module} does not have a __name__ attribute. Please ensure that you are passing a valid PyTorch function."
        )
    prefix = name_prefix_map.get(torch_module, None)
    if prefix and function_name.startswith(prefix):
        function_name = function_name[len(prefix) :]
    utils.check(
        getattr(torch_module, function_name, None),
        lambda: f"Incorrect function name {function_name} inferred for PyTorch function {torchfn} from module {torch_module}.",
    )
    return function_name


def register_default_torch_op(torchfn: Callable, torch_module):
    from thunder.core.transforms import augmented_forward_impls, backward_impls
    from thunder.executors.torchex import _always_executable, ex

    fn_meta = meta_adaptor(torchfn)
    _fn = langctx(Languages.TORCH)(fn_meta)
    _fn.__torchfn = torchfn
    torchfn_name = _get_torch_function_name(torch_module, torchfn)
    sym = Symbol(
        name=torchfn_name,
        meta=_fn,
        id=f"{torch_module.__name__}.{torchfn_name}",
        is_prim=True,
        tags=(prims.OpTags.AUTO_REGISTERED,),
    )

    # NOTE: We follow the manual registration approach, where functions with the same name in both torch and torch.Tensor share the same symbol.
    # Therefore, we reuse the existing symbol instead of creating a new one.
    sym_exist = False
    if torch_module in (torch, torch.Tensor):
        other_module = torch.Tensor if torch_module is torch else torch
        if (torch_fn := getattr(other_module, torchfn_name, None)) and torch_fn in _torch_to_thunder_function_map:
            sym = _torch_to_thunder_function_map[torch_fn]
            sym_exist = True

    _torch_to_thunder_function_map[torchfn] = sym

    # TODO: convert to an assert after #1140 is fixed
    if torchfn_name not in __builtins__ and not hasattr(sys.modules["thunder.torch"], torchfn_name):
        setattr(sys.modules["thunder.torch"], torchfn_name, sym)

    # We need to invoke `register_method` on methods
    # so that `x.method` is registered to the TensorProxy.
    if torch_module is torch.Tensor:
        register_method(torchfn.__name__, sym)

    # If the function exists either in the torch or torch.Tensor namespace and is already registered,
    # return early to prevent overwriting the existing mapping.
    if sym_exist:
        return

    op = ex.register_operator(torchfn_name, module=torch_module, meta=fn_meta)
    ex.register_implementation(sym, op, checker=_always_executable)
    augmented_forward_impls[sym.id] = augmented_forward_adaptor(op)

    _vjp_impl_wrapper = partial(_vjp_impl, torchfn)

    bwd_op = ex.register_operator(torchfn_name + "_vjp", meta=backward_adaptor(torchfn), fn=_vjp_impl_wrapper)
    ex.register_implementation(bwd_op.id, bwd_op, checker=_always_executable)
    backward_impls[sym.id] = bwd_op


# Note this function should be used inside the fake mode context manager
def _get_fake_arg(inp: Any):
    if inp is None:
        return inp
    if isinstance(inp, NumberProxy):
        if inp.value is None:
            raise NotImplementedError("Unsupported for NumberProxy.value=None")
        else:
            return inp.value
    elif isinstance(inp, TensorProxy):
        return torch.empty(
            inp.shape,
            dtype=_thunder_to_torch_dtype_map[inp.dtype],
            requires_grad=inp.requires_grad,
            device=devices.to_torch_device(inp.device),
        )
    elif isinstance(inp, (TupleProxy, ListProxy, DictProxy)):
        return inp._value
    elif isinstance(inp, (torch.dtype, torch.device, torch.Size, torch.memory_format, str, bool, int, float, complex)):
        return inp
    elif isinstance(inp, devices.Device):
        return devices.to_torch_device(inp)
    elif isinstance(inp, dtypes.dtype):
        return to_torch_dtype(inp)
    else:
        raise NotImplementedError(f"Unsupported type: {builtins.type(inp)}")


def _fake_type_to_thunder(inp: Any):
    from thunder.core.proxies import _cls_to_number_proxy_map
    from torch._subclasses.fake_tensor import FakeTensor

    if inp is None:
        return inp
    if isinstance(inp, tuple(_cls_to_number_proxy_map.keys())):
        return numberproxy(builtins.type(inp), inp)
    elif isinstance(inp, FakeTensor):
        return TensorProxy(
            shape=inp.shape,
            device=to_device(inp.device),
            dtype=_torch_to_thunder_dtype_map[inp.dtype],
            requires_grad=inp.requires_grad,
        )
    elif isinstance(inp, torch.Size):
        return tuple(inp)
    elif isinstance(inp, torch.device):
        return to_device(inp)
    elif isinstance(inp, torch.dtype):
        return _torch_to_thunder_dtype_map[inp]
    elif isinstance(inp, (str, bool, int, float, complex)):
        return inp
    else:
        raise NotImplementedError(f"Unsupported type: {builtins.type(inp)}")


def augmented_forward_adaptor(sym_op: Callable):
    def augmented_forward(*args, **kwargs):
        from thunder.core.transforms import VJPDual

        out = sym_op(*args, **kwargs)
        primal = out if isinstance(out, tuple) else (out,)
        saved_for_backward = ((args, kwargs),)
        return VJPDual(primal, saved_for_backward)

    return augmented_forward


def backward_adaptor(torch_func: Callable):
    def backward(saved_for_backward, *grad_output):
        inp_args, inp_kwargs = saved_for_backward
        flat_args, spec = tree_flatten(inp_args)
        if not any(map(_is_differentiable, grad_output)):
            return tree_unflatten(len(flat_args) * [None], spec)
        if any(map(_is_differentiable, tree_flatten(inp_kwargs)[0])):
            raise NotImplementedError(
                f"Exception encountered when doing automatic registration for {torch_func} because there is keyword argument that requires gradient, please use manual registration."
            )
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_inp_args, fake_inp_kwargs = tree_map(_get_fake_arg, (inp_args, inp_kwargs))
            wrapped_torch_func, diff_args = _make_differentiable_wrapper(torch_func, *fake_inp_args, **fake_inp_kwargs)
            try:
                _, fake_vjp = torch.autograd.functional.vjp(
                    wrapped_torch_func, diff_args, v=tree_map(_get_fake_arg, grad_output)
                )
            except Exception as e:
                msg = f"Exception encountered when doing automatic registration for {torch_func}, please use manual registration: {repr(e)}"
                raise NotImplementedError(msg) from e

        out_vjp = tree_map(_fake_type_to_thunder, fake_vjp)
        iter_out_vjp = iter(out_vjp if isinstance(out_vjp, Sequence) else (out_vjp,))
        fake_vjp = tuple(next(iter_out_vjp) if _is_differentiable(arg) else None for arg in flat_args)
        return tree_unflatten(fake_vjp, spec)

    return backward


def _vjp_impl(torchfn, saved_for_backward, *gs):
    inp_args, inp_kwargs = saved_for_backward

    wrapped_func, diff_args = _make_differentiable_wrapper(torchfn, *inp_args, **inp_kwargs)
    _, vjp_outs = torch.autograd.functional.vjp(wrapped_func, diff_args, v=gs)
    flat_inp_args, inp_spec = tree_flatten(inp_args)
    iter_vjp_out = iter(vjp_outs if isinstance(vjp_outs, Sequence) else (vjp_outs,))
    vjp_outs = tuple(next(iter_vjp_out) if _is_differentiable(arg) else None for arg in flat_inp_args)
    return tree_unflatten(vjp_outs, inp_spec)


def meta_adaptor(torch_func: Callable):
    def meta_func(*args, **kwargs):
        from torch._subclasses.fake_tensor import FakeTensorMode

        if kwargs.get("inplace", False):
            raise NotImplementedError(f"{torch_func} has inplace=True, please use manual registration")
        if kwargs.get("out", None) is not None:
            raise NotImplementedError(f"{torch_func} specifies 'out' argument, please use manual registration")

        with FakeTensorMode():
            fake_args, fake_kwargs = tree_map(_get_fake_arg, (args, kwargs))
            try:
                fake_outs = torch_func(*fake_args, **fake_kwargs)
            except Exception as e:
                msg = f"Exception encountered when doing automatic registration for {torch_func.__name__}, please use manual registration: {repr(e)}"
                raise NotImplementedError(msg) from e

        flat_fake_outs, spec_outs = tree_flatten(fake_outs)
        outs = tuple(map(_fake_type_to_thunder, flat_fake_outs))
        if hasattr(spec_outs.type, "__module__") and spec_outs.type.__module__.startswith("torch.return_types"):
            return outs
        return tree_unflatten(outs, spec_outs)

    return meta_func


@langctx(Languages.TORCH)
def check_overlap_ops():
    from thunder.torch import default_torch_ops

    torch_lang_ctx = get_langctx()
    for m, fns in default_torch_ops.torch_auto_registered_ops.items():
        for fn in fns:
            if fn in _torch_to_thunder_function_map:
                raise RuntimeError(
                    f"{m.__name__}.{fn.__name__} is already registered in _torch_to_thunder_function_map, please remove it from default_torch_ops.py"
                )
            # NOTE - Some tensor methods like `float`, `size` are just registered as methods without `torchsymbol` so they don't show up in `_torch_to_thunder_function_map`.
            if m is torch.Tensor and torch_lang_ctx.has_method(fn.__name__):
                raise RuntimeError(
                    f"{m.__name__}.{fn.__name__} is already registered as method for TensorProxy under torch_lang_ctx, please remove it from default_torch_ops.py"
                )


# Verify that there is no overlap between automatically registered operations and manually registered operations.
check_overlap_ops()
register_default_torch_ops()


#
# torch -> thunder object mapping
#


_torch_to_thunder_complete_map = {
    **_torch_to_thunder_dtype_map,
    **_torch_to_thunder_function_map,
    **{fn: fn for fn in _torch_noinline_functions},
}

# records the torch symbols that may return tensor views
# ref: https://pytorch.org/docs/stable/tensor_view.html
# NOTE Symbols that return tensor views can interfere with in-place operators
# See :func:`thunder.core.functionalization.check_inplace_to_views` for the details.
_syms_that_may_return_views: set[Symbol] = {
    reshape,
    contiguous,
    to,
    flatten,
    _torch_to_thunder_function_map[torch.Tensor.reshape_as],
}

_syms_returning_views: set[Symbol] = {
    diagonal,
    expand,
    expand_as,
    movedim,
    permute,
    select,
    squeeze,
    transpose,
    t,
    real,
    unflatten,
    unfold,
    unsqueeze,
    view,
    view_as,
    unbind,
    split,
    tensor_split,
    chunk,
    getitem,
    prims.shallow_copy,
}

# Add all auto-registered torch operators symbol that return tensor views to _syms_returning_views
_syms_returning_views.update({_torch_to_thunder_function_map[x] for x in _auto_registered_operators_returning_views})
