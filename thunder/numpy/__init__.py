from numbers import Number
from typing import Any

from thunder.core.proxies import TensorProxy
import thunder.core.prims as prims
import thunder.core.utils as utils
import thunder.core.dtypes as dtypes
from thunder.core.langctx import langctx
from thunder.core.symbol import Symbol
import thunder.clang as clang
from thunder.core.trace import get_tracectx
import thunder.core.devices as devices


__all__ = [
    "is_available",
]

#
# The NumPy language
#
# TODO Make required?
# TODO Let preprocessing remap these operations, just like torch operations

np = None

try:
    import numpy

    np = numpy
except:
    pass

#
# langctx interface: is_available, to_dtype, tensor_cls, tensorproxy, ops
#


def is_available():
    return np is not None


# NOTE NumPy does not support bfloat16 or complexhalf datatypes
# TODO consider how to model "missing" dtypes better
_thunder_to_numpy_dtype_map = {
    bool: np.bool_,
    int: np.int_,
    float: np.float_,
    complex: np.cfloat,
    dtypes.bool8_: np.bool_,
    dtypes.bool8: np.bool_,
    dtypes.uint8_: np.uint8,
    dtypes.uint8: np.uint8,
    dtypes.int8_: np.int8,
    dtypes.int8: np.int8,
    dtypes.int16_: np.int16,
    dtypes.int16: np.int16,
    dtypes.int32_: np.int32,
    dtypes.int32: np.int32,
    dtypes.int64_: np.int64,
    dtypes.int64: np.int64,
    dtypes.float16_: np.float16,
    dtypes.float16: np.float16,
    dtypes.float32_: np.float32,
    dtypes.float32: np.float32,
    dtypes.float64_: np.float64,
    dtypes.float64: np.float64,
    dtypes.complex64_: np.complex64,
    dtypes.complex64: np.complex64,
    dtypes.complex128_: np.complex128,
    dtypes.complex128: np.complex128,
}

_numpy_to_thunder_dtype_map = {
    v: k
    for k, v in _thunder_to_numpy_dtype_map.items()
    if not utils.is_weak_dtype(k) and not (type(k) is type and issubclass(k, Number))
}


def to_dtype(x: Any) -> dtypes.dtype:
    if isinstance(x, np.ndarray):
        return _numpy_to_thunder_dtype_map(x.dtype)

    return dtypes.to_dtype(x)


# NOTE: dtype.type is required when working with NumPy dtyes, which have an odd
#   (at least from our perspective) relationship between the type classes and the
#   dtypes on tensors
def to_thunder_dtype(numpy_dtype: np.dtype) -> dtypes.dtype:
    return _numpy_to_thunder_dtype_map[numpy_dtype.type]


def to_numpy_dtype(thunder_dtype: dtypes.dtype) -> np.dtype:
    return _thunder_to_numpy_dtype_map[thunder_dtype]


tensor_cls = np.ndarray


# NOTE This both extracts the metadata from an ndarray to create a TensorProxy
#   AND then schedules a conversion of that ndarray to a tensor
def tensorproxy(name: str, a: np.ndarray) -> TensorProxy:
    device = devices.cpu
    dtype = numpy_to_thunder_dtype(a.dtype)

    p = TensorProxy(name, shape=a.shape, device=device, dtype=dtype)

    tracectx = get_tracectx()
    tracectx.post_unpack(lambda: prims.numpy_array_to_torch_tensor(p))

    return p


#
# NumPy operation definitions
#


# A wrapper that executes the operations within the NumPy language context
# NOTE because this module defines the NumPy language context, a reference to itself
#   is aquired by inspecting the __module__ attribute of the is_available function defined
#   above
def numpy_symbol(fn):
    module_name = numpy_symbol.__module__
    module = utils.get_module(module_name)
    _fn = langctx(module)(fn)
    sym = Symbol(name=fn.__name__, meta=_fn)
    return sym


#
# Tensor properties
#


def size(a):
    return a.numel


#
# Elementwise binary operators
#


# TODO Create a factory that adds ufunc support to elementwise operations
@numpy_symbol
def add(a, b, *, where=None):
    result = clang.add(a, b)
    if where is not None:
        return clang.where(where, result, a)
    return result


@numpy_symbol
def lt(a, b):
    return clang.lt(a, b)


#
# Conditional operators
#


@numpy_symbol
def where(pred, a, b):
    return clang.where(pred, a, b)
