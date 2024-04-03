import itertools
import math
import operator
import collections
import re
from collections.abc import Sequence
from enum import Enum
from functools import partial, reduce, wraps
from numbers import Number
from typing import Any, Union, Optional, Tuple
from collections.abc import Callable

import opt_einsum

# Initializes the language context
from thunder.torch.langctx import register_method

import thunder.clang as clang
import thunder.core.devices as devices
from thunder.core.devices import to_device
import thunder.core.dtypes as dtypes
from thunder.core.dtypes import to_torch_dtype, to_dtype, _thunder_to_torch_dtype_map, _torch_to_thunder_dtype_map
import thunder.core.prims as prims
import thunder.core.utils as utils
import thunder.distributed.prims as dist_prims
from thunder.core.langctxs import langctx, Languages
from thunder.core.proxies import TensorProxy, FutureTensorProxy
from thunder.core.pytree import tree_map
from thunder.core.symbol import Symbol
from thunder.core.transforms import register_grad, put_grads
from thunder.core.prims import get_grad, put_grad
from thunder.core.baseutils import run_once

__all__ = [
    "is_available",
]

# NOTE torch is a requirement
import torch
import torch.distributed as tdist

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
        id: str | None = None,
        is_prim: bool = False,
        tags: None | list[Any] = None,
    ):
        self.torchfns = torchfns
        self.is_method = is_method or (method_name is not None)
        self.method_name: None | str = method_name
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

        if self.torchfns is not None:
            for torchfn in self.torchfns:
                _torch_to_thunder_function_map[torchfn] = sym

        return sym


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
def size(a):
    def fn_(idx: int | None = None):
        if idx is None:
            return a.shape
        return a.shape[idx]

    return fn_


register_method("size", size)


#
# Data movement and transformation operations
#


# NOTE This handles a.float()
#   It avoids using the name "float" to not collide with the builtin
#   "float"
def to_float(a: Number | TensorLike) -> Number | TensorLike:
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


# TODO Model non_blocking, copy, and memory_format (as kwargs)
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
        return a

    # NOTE copy == False
    # NOTE to() returns the tensor unmodified if the device and dtype requested are the same
    #   (and copy=False)
    # NOTE clang.device_put does nothing when device is None or a.device == device
    a = clang.device_put(a, device)

    if dtype is not None:
        return clang.maybe_convert_to_dtype(a, dtype)

    return a


@torchsymbol(torch.Tensor.type_as, is_method=True)
def type_as(a: TensorProxy, b: TensorProxy, /) -> TensorProxy:
    # NOTE This type check is intentional since we're accessing the true_dtype
    #   attribute of the TensorProxy
    # TODO Create a generic Tensor annotation, and support both PyTorch
    #   tensors and TensorProxies being passed to this operation
    utils.check_type(b, TensorProxy)

    return to(a, b.true_dtype)


#
# Tensor creation operations
#


@torchsymbol(torch.arange)
def arange(
    start: Number,
    end: None | Number = None,
    step: Number = 1,
    *,
    device: None | DeviceLike = None,
    dtype: None | dtypeLike = None,
) -> TensorLike:
    if device is None:
        device = "cpu"

    device = to_device(device)
    dtype = to_dtype(dtype)

    if end is None:
        end = start
        start = 0
    return clang.arange(start=start, step=step, stop=end, device=device, dtype=dtype)


@torchsymbol(torch.full)
def full(
    shape: Sequence[int], fill_value: Number, *, device: None | DeviceLike = None, dtype: None | dtypeLike = None
) -> TensorLike:
    if device is None:
        device = "cpu"

    device = to_device(device)
    dtype = to_dtype(dtype)

    return clang.full(shape, fill_value, device=device, dtype=dtype)


@torchsymbol(torch.full_like)
def full_like(
    a: TensorLike, /, fill_value: Number, *, device: None | DeviceLike = None, dtype: None | dtypeLike = None
) -> TensorLike:
    device = to_device(device)
    dtype = to_dtype(dtype)
    return clang.full_like(a, fill_value, device=device, dtype=dtype)


# NOTE ones, unlike full, can accept an integer shape
@torchsymbol(torch.ones)
def ones(*shape: int, device: None | DeviceLike = None, dtype: None | dtypeLike = None) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)
    return full(shape, 1, device=device, dtype=dtype)


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
        not requires_grad, lambda: "requires_grad=True is not yet supported within thunder.compile", NotImplementedError
    )
    utils.check(not pin_memory, lambda: "pin_memory=True is not supported within thunder.compile", NotImplementedError)

    if isinstance(seq_or_number, Number):
        return full((), seq_or_number, dtype=dtype, device=device)

    return clang.tensor_from_sequence(seq_or_number, dtype=dtype, device=device)


# TODO based on uniform_, check if Torch now has a functional uniform
# NOTE the uniform_ documentation suggests the interval is specified using "from" and "to",
#   but from is a reserved keyword in Python
@torchsymbol(is_method=False, id="torch.uniform")
def uniform(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypeLike,
) -> TensorLike:
    device = to_device(device)
    dtype = to_dtype(dtype)

    return clang.uniform(shape, minval, maxval, device=device, dtype=dtype)


@torchsymbol(is_method=False, id="torch.uniform_like")
def uniform_like(
    a: TensorLike,
    /,
    minval: Number = 0.0,
    maxval: Number = 1.0,
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

    utils.check(
        generator is None,
        lambda: "Non-None generator is not supported",
        NotImplementedError,
    )

    seed = None
    samples = prims.multinomial(a, num_samples, replacement, seed)
    return samples


# TODO Maybe update this to return an offset of how far to advance the seed to acquire new values
# See issue "Maybe return offset from thunder.torch.uniform_philox"
@torchsymbol(is_method=False, id="torch.uniform_philox")
def uniform_philox(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypeLike,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
) -> TensorLike:
    device = to_device(device)
    dtype = to_dtype(dtype)

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
        not requires_grad, lambda: "requires_grad=True is not yet supported within thunder.compile", NotImplementedError
    )
    utils.check(layout == torch.strided, lambda: "Only torch.strided layout is supported", NotImplementedError)
    utils.check(not pin_memory, lambda: "pin_memory=True is not supported within thunder.compile", NotImplementedError)
    # NOTE: Currently, we don't model randomness
    utils.check(generator is None, lambda: "generator is not None which is currently unsupported", NotImplementedError)
    utils.check(out is None, lambda: "out is not None which is currently unsupported", NotImplementedError)
    if device is None:
        device = "cpu"
    device = to_device(device)

    # For now we default to `float32`,
    # however, we should add a default dtype or
    # rely on `torch.get_default_dtype`.
    if dtype is None:
        dtype = torch.float
    dtype = to_dtype(dtype)
    shape = utils.extract_shape_from_varargs(shape)
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
        not requires_grad, lambda: "requires_grad=True is not supported within thunder.compile", NotImplementedError
    )
    utils.check(
        layout is None or layout == torch.strided, lambda: "Only torch.strided layout is supported", NotImplementedError
    )
    utils.check(
        memory_format == torch.preserve_format,
        lambda: "preserve_format!=torch.preserve_format is not supported within thunder.compile",
        NotImplementedError,
    )

    if dtype is None:
        dtype = a.dtype

    if device is None:
        device = a.device
    return randn(a.shape, dtype=dtype, device=device)


# NOTE zeros, like ones, and unlike full, can accept an integer shape
@torchsymbol(torch.zeros)
def zeros(*shape: int, device: None | DeviceLike = None, dtype: None | dtypeLike = None) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)
    return full(shape, 0, device=device, dtype=dtype)


@torchsymbol(torch.zeros_like)
def zeros_like(a: TensorLike, /, *, device: DeviceLike | None = None, dtype: dtypeLike | None = None) -> TensorLike:
    return full_like(a, 0, device=device, dtype=dtype)


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


@torchsymbol(torch.flatten, is_method=True)
def flatten(a: TensorLike, /, start_dim: int = 0, end_dim: int = -1) -> TensorLike:
    return clang.flatten(a, start_dim, end_dim)


@torchsymbol(torch.flip, is_method=True)
def flip(a: TensorLike, /, *dims: int) -> TensorLike:
    dims = utils.extract_shape_from_varargs(dims)

    # PyTorch supports 0-dim inputs with len(dims) <= 1
    if a.ndim == 0 and isinstance(dims, Sequence) and len(dims) > 0:
        utils.check(
            len(dims) == 1 and isinstance(dims[0], int) and dims[0] in (0, -1),
            lambda: f"Expected {dims=} to be a sequence of integers in range [-1, 0], and of length 1",
        )
        return clang.flip(a, ())

    return clang.flip(a, dims)


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
def pad(a: TensorProxy, /, pad: tuple[int, ...], mode: str | None = "constant", value: Number | None = None):
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

    return clang.pad(a, value, pad_config)


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
        (int, Sequence),
    )

    # TODO: consider revising this to just call _split_indices
    if isinstance(size_or_sections, int):
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
    elif isinstance(dim, int):
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


@torchsymbol(torch.unbind, is_method=True)
def unbind(a: TensorLike, /, dim: int = 0) -> tuple[TensorLike, ...]:
    utils.check(
        len(a.size()) > 0,
        lambda: f"Dimension specified as={dim} but tensor has no dimensions.",
    )
    return tuple(s.squeeze(dim) for s in tensor_split(a, a.shape[dim], dim))


@torchsymbol(torch.unsqueeze, is_method=True)
def unsqueeze(a: TensorLike, /, dim: int) -> TensorLike:
    return clang.unsqueeze(a, dim)


# TODO Review view functionalization
# TODO Add type annotations
@torchsymbol(torch.Tensor.view, is_method=True)
def view(a: TensorLike, /, *shape) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)
    return reshape(a, shape)


#
# Elementwise unary operaitons
#
# TODO Add type annotations


@torchsymbol(torch.abs, is_method=True)
def abs(a: Number | TensorLike, /) -> Number | TensorLike:
    return clang.abs(a)


@torchsymbol(torch.acos, is_method=True)
def acos(a: Number | TensorLike, /) -> Number | TensorLike:
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


@torchsymbol(torch.digamma, torch.special.digamma, is_method=True)
def digamma(a):
    return clang.digamma(a)


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


@torchsymbol(torch.real, is_method=False)
def real(a):
    return clang.real(a)


#
# nn.functional elementwise unary
#
# TODO Move these to torch.nn.functional


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


# TODO Should this use clamp? -- Would that propagate NaNs properly?
@torchsymbol(torch.relu, torch.nn.functional.relu, id="torch.relu", is_method=True)
def relu(a: TensorLike, /, inplace: bool = False) -> TensorLike:
    utils.check(not inplace, lambda: f"relu only supports inplace=False", exception_type=NotImplementedError)

    return where(a > 0, a, 0)


# id=torch.relu because we ignore inplace argument in torch.nn.functional.relu
@torchsymbol(torch.nn.functional.relu6, id="torch.relu6", is_method=False)
def relu6(a: TensorProxy, /, inplace: bool = False) -> TensorLike:
    utils.check(not inplace, lambda: f"relu6 only supports inplace=False", exception_type=NotImplementedError)
    return clamp(a, 0, 6)


@torchsymbol(torch.nn.functional.hardswish, id="torch.hardswish", is_method=False)
def hardswish(a: TensorProxy, /, inplace: bool = False) -> TensorLike:
    utils.check(not inplace, lambda: f"hardswish only supports inplace=False", exception_type=NotImplementedError)
    utils.check(
        dtypes.is_float_dtype(a.dtype),
        lambda: f"hardswish only supports floating point dtypes, got {a.dtype}",
        exception_type=ValueError,
    )
    return a * relu6(a + 3) / 6


# id=torch.selu because we ignore inplace argument in torch.nn.functional.selu
@torchsymbol(torch.selu, torch.nn.functional.selu, id="torch.selu", is_method=False)
def selu(a: TensorProxy, /, inplace: bool = False) -> TensorLike:
    utils.check(not inplace, lambda: f"selu only supports inplace=False", exception_type=NotImplementedError)

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    rhs = alpha * expm1(a)

    return scale * where(a > 0, a, rhs)


@torchsymbol(torch.nn.functional.silu)
def silu(a, /):
    return clang.silu(a)


#
# Elementwise binary operations
#


@torchsymbol(torch.add, is_method=True)
def add(
    a: Number | TensorLike, b: Number | TensorLike, /, *, alpha: None | Number | TensorLike = None
) -> Number | TensorLike:
    if alpha is not None:
        b = b * alpha

    return clang.add(a, b)


@torchsymbol(torch.atan2, is_method=True)
def atan2(a, b, /):
    return clang.atan2(a, b)


@torchsymbol(torch.bitwise_and, is_method=True)
def bitwise_and(a, b, /):
    return clang.bitwise_and(a, b)


@torchsymbol(torch.bitwise_or, is_method=True)
def bitwise_or(a, b, /):
    return clang.bitwise_or(a, b)


@torchsymbol(torch.bitwise_xor, is_method=True)
def bitwise_xor(a, b, /):
    return clang.bitwise_xor(a, b)


@torchsymbol(torch.copysign, is_method=True)
def copysign(a, b, /):
    return clang.copysign(a, b)


# TODO Implement div


@torchsymbol(torch.eq, is_method=True)
def eq(a, b, /):
    return clang.eq(a, b)


@torchsymbol(torch.floor_divide, is_method=True)
def floor_divide(a, b, /):
    return clang.floor_divide(a, b)


@torchsymbol(torch.fmod, is_method=True)
def fmod(a, b, /):
    return clang.fmod(a, b)


@torchsymbol(torch.ge, is_method=True)
def ge(a, b, /):
    return clang.ge(a, b)


@torchsymbol(torch.gt, is_method=True)
def gt(a, b, /):
    return clang.gt(a, b)


@torchsymbol(torch.logical_and, is_method=True)
def logical_and(a, b, /):
    return clang.logical_and(a, b)


@torchsymbol(torch.le, is_method=True)
def le(a, b, /):
    return clang.le(a, b)


@torchsymbol(torch.lt, is_method=True)
def lt(a, b, /):
    return clang.lt(a, b)


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


@torchsymbol(torch.mul, is_method=True)
def mul(a, b, /):
    return clang.mul(a, b)


@torchsymbol(torch.ne, is_method=True)
def ne(a, b, /):
    return clang.ne(a, b)


@torchsymbol(torch.nextafter, is_method=True)
def nextafter(a, b, /):
    return clang.nextafter(a, b)


# TODO Extend to tensor x tensor
@torchsymbol(torch.polygamma, torch.special.polygamma, is_method=True)
def polygamma(n: int, a: TensorLike, /) -> TensorLike:
    utils.check(isinstance(n, int), lambda: f"polygamma(n, a) expects the first argument to be an integer.")
    utils.check(n >= 0, lambda: f"polygamma(n, a) does not support negative {n=}.")

    # NOTE Use digamma for n == 0 case; otherwise zeta(1, a) returns math.inf
    if n == 0:
        return digamma(a)

    sign = 1 if (n % 2) == 1 else -1
    # Compute in log-space for numerical stability
    factorial_mul_zeta = exp(lgamma(n + 1.0) + log(zeta(n + 1.0, a)))
    return sign * factorial_mul_zeta


@torchsymbol(torch.pow, is_method=True)
def pow(a, b, /):
    return clang.pow(a, b)


@torchsymbol(torch.remainder, is_method=True)
def remainder(a, b, /):
    return clang.remainder(a, b)


@torchsymbol(torch.sub, is_method=True)
def sub(a, b, /, *, alpha=None):
    if alpha is not None:
        b = b * alpha

    return clang.sub(a, b)


@torchsymbol(torch.true_divide, is_method=True)
def true_divide(a: Number | TensorLike, b: Number | TensorLike, /) -> Number | TensorLike:
    return clang.true_divide(a, b)


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


@torchsymbol(torch.addcdiv, is_method=True)
def addcdiv(a: TensorLike, b: TensorLike, c: TensorLike, /, *, value: None | Number = None) -> TensorLike:
    return addcmul_addcdiv_helper(a, b, c, add, true_divide, value=value)


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
def tril(a: TensorLike, /, diagonal: int = 0, *, fill_value: None | Number = None) -> TensorLike:
    utils.check(a.ndim >= 2, lambda: f"tril: a ({a.ndim=}) must have at least two dimensions")

    nrows, ncols = a.shape[-2:]
    row_numbers = arange(nrows, device=a.device).unsqueeze(-1)
    col_numbers = arange(ncols, device=a.device).unsqueeze(-2)

    mask = (row_numbers + diagonal) >= col_numbers

    if fill_value is None:
        fill_value = 0

    return _mask_tensor(a, mask, fill_value)


@torchsymbol(torch.where, is_method=True)
def where(pred: TensorLike, a: Number | TensorLike, b: Number | TensorLike, /) -> TensorLike:
    return clang.where(pred, a, b)


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
    dtype: None | torch.dtype = None,  # should be specified for ops that support it
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

    a = to(a, computation_dtype)
    result = prim(a, dims)

    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = tree_map(lambda x: prims.broadcast_in_dim(x, output_shape, broadcast_dims), result)

    if result_dtype is not None:
        result = tree_map(lambda x: to(x, result_dtype), result)

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


@torchsymbol(torch.mean, is_method=True)
def mean(a: TensorProxy, /, dim=None, keepdim: bool = False, *, dtype=None):
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


@torchsymbol(torch.var, is_method=True)
def var(
    a: TensorProxy,
    /,
    dim=None,
    *,
    keepdim: bool = False,
    correction: Number = 1,
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
    correction: Number = 1,
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


#
# Scatter and gather-related operations
#


# NOTE PyTorch also has an alpha parameter
@torchsymbol(torch.index_add)
def index_add(a: TensorLike, /, dim: int, index: TensorLike, source: TensorLike) -> TensorLike:
    return clang.index_add(a, index, source, dim)


@torchsymbol(torch.index_select, is_method=True)
def index_select(a: TensorLike, /, dim: int, index: TensorLike) -> TensorLike:
    return clang.take(a, index, dim)


# NOTE PyTorch's scatter_add has a parameter named 'src', not 'source'
@torchsymbol(torch.scatter_add)
def scatter_add(a: TensorLike, /, dim: int, index: TensorLike, src: TensorLike) -> TensorLike:
    return clang.scatter_add(a, indices=index, value=src, dim=dim)


@torchsymbol(torch.take_along_dim)
def take_along_dim(a: TensorLike, /, indices: TensorLike, dim: int) -> TensorLike:
    return clang.take_along_axis(a, indices, dim)


@torchsymbol(torch.index_put, is_method=True)
def index_put(
    a: TensorLike, /, indices: Sequence[TensorLike], values: TensorLike, accumulate: bool = False
) -> TensorLike:
    return clang.index_put(a, indices, values, accumulate)


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


@torchsymbol(torch.outer)
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


# TODO: likely want to refactor these normalizations
def _native_layer_norm(
    a: TensorProxy, /, normalized_shape, weight, bias, eps: Number
) -> tuple[TensorLike, TensorLike, TensorLike]:
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
    eps: Number = 1e-5,
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


@torchsymbol(torch.nn.functional.batch_norm)
def batch_norm(
    a: TensorLike,
    running_mean: None | TensorLike = None,
    running_var: None | TensorLike = None,
    weight: None | TensorLike = None,
    bias: None | TensorLike = None,
    training: bool = False,
    momentum: Number = 0.1,
    eps: Number = 1e-5,
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
    a_dtype = a.dtype
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

    # inplace operation is not supported, so running mean and running var can not be casted,
    # they are passed to prim operator directly
    (
        a,
        weight,
        bias,
    ) = (
        to(x, computation_dtype) if x is not None else x
        for x in (
            a,
            weight,
            bias,
        )
    )
    result = prims.batch_norm(a, weight, bias, running_mean, running_var, training, momentum, eps)
    output = to(result[0], a_dtype)
    return output


@torchsymbol("batch_norm_backward", id="batch_norm_backward")
def batch_norm_backward(
    g: TensorLike,
    a: TensorLike,
    weight: None | TensorLike,
    running_mean: None | TensorLike,
    running_var: None | TensorLike,
    save_mean: None | TensorLike,
    save_invstd: None | TensorLike,
    train: bool,
    eps: Number,
    output_mask: list[bool],
) -> tuple[TensorLike, None | TensorLike, None | TensorLike]:
    utils.check(a.ndim >= 2, lambda: f"Input tensor must have at least batch and channel dimensions!")
    if train:
        utils.check(
            save_mean is not None and save_invstd is not None,
            lambda: f"when train=True, save_mean and save_invstd are required",
        )
    else:
        utils.check(
            running_mean is not None and running_var is not None,
            lambda: f"when train=False, running_mean and running_var are required",
        )
    a_dtype = a.dtype
    if weight is not None:
        weight_dtype = weight.dtype
    else:
        weight_dtype = a_dtype
    computation_dtype = utils.get_computation_dtype(a.dtype)
    (
        grad_out_cast,
        a_cast,
        weight_cast,
        running_mean_cast,
        running_var_cast,
        save_mean_cast,
        save_invstd_cast,
    ) = (
        to(x, computation_dtype) if x is not None else x
        for x in (
            g,
            a,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
        )
    )
    input_shape = a.shape
    input_rank = a.dim()
    axis = 1
    input_size_prod = 1
    for x in input_shape:
        input_size_prod *= x

    num_features = input_size_prod / input_shape[axis]
    mean = save_mean_cast
    invstd = save_invstd_cast
    if not train:
        mean = running_mean_cast
        invstd = rsqrt(running_var_cast + eps)
    broadcast_mask = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]

    reduction_axes = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)

    mean = reshape(mean, broadcast_mask)
    norm = 1.0 / num_features
    grad_output_sum = sum(grad_out_cast, reduction_axes)
    dot_p = sum(grad_out_cast * (a_cast - mean), reduction_axes)

    grad_mean = reshape(grad_output_sum * norm, broadcast_mask)
    proj_scale = reshape(mul(dot_p * norm, invstd * invstd), broadcast_mask)

    if weight_cast is None:
        grad_scale = reshape(invstd, broadcast_mask) * 1.0
    else:
        grad_scale = reshape(invstd * weight_cast, broadcast_mask)

    if train:
        proj = (a_cast - mean) * proj_scale
        grad_input = ((grad_out_cast - proj) - grad_mean) * grad_scale
    else:
        grad_input = grad_out_cast * grad_scale

    if output_mask[1]:
        grad_weight = dot_p * invstd
    else:
        grad_weight = None

    if output_mask[2]:
        grad_bias = grad_output_sum
    else:
        grad_bias = None

    return (
        grad_input.to(a_dtype),
        None if grad_weight is None else clang.maybe_convert_to_dtype(grad_weight, weight_dtype),
        None if grad_bias is None else clang.maybe_convert_to_dtype(grad_bias, weight_dtype),
    )


#
# NN Operations
#


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
    if isinstance(param, int):
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
        len(padding) == dim and all(isinstance(p, int) and 0 <= p <= k // 2 for p, k in zip(padding, kernel_size)),
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
                    all(isinstance(d, int) and d >= 1 for d in dilation),
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

    res = clang.convolution(
        a, weight, bias, stride, padding, dilation, False, (0,) * dim, groups  # transposed  # output_padding
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
        len(kernel_size) == dim and all(isinstance(k, int) and k > 0 for k in kernel_size),
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
    divisor_override: Number | None = None,
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
        len(kernel_size) == dim and all(isinstance(k, int) and k > 0 for k in kernel_size),
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
        isinstance(divisor_override, Number) and divisor_override > 0,
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
    divisor_override: Number | None = None,
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
    divisor_override: Number | None = None,
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
    divisor_override: Number | None = None,
) -> TensorProxy:
    return _avg_pool_helper(3, a, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


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


# TODO Add annotations, make not a prim
# The backward decomposition of cross_entropy cannot be efficiently fused, so we have this cross_entropy_backward
# primitive. Executors can override the primitive using internal implementations.
# See issue "Cross_entropy is decomposed for backward but the decomposition is
# not fusible currently"
@torchsymbol("cross_entropy_backward", id="cross_entropy_backward", is_prim=True)
def cross_entropy_backward(g, a, /, target, weight, reduction, ignore_index, label_smoothing):
    return TensorProxy(like=g, shape=a.shape)


# TODO (mruberry) -- I think this implementation gets the dtype of the output incorrect
# TODO Revise this to consistently call other torch operations where possible
# TODO -- Maybe cut this up into _cross_entropy_mean, _cross_entropy_sum, ...
# TODO Add type annotations
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

    utils.check(
        a.ndim != 0,
        lambda: f"Cross entropy expects its input to have one or more dimensions, but it had zero dimensions",
    )

    # NOTE label_smoothing < 0 will just be ignored.
    utils.check(
        label_smoothing <= 1.0,
        lambda: f"Cross entropy's {label_smoothing=} must be less than or equal to 1.0",
    )

    # extract shape information
    C_dim = 1 if a.ndim >= 2 else 0
    N = a.shape[0] if a.ndim >= 2 else 1
    C = a.shape[C_dim]
    feature_size = int(a.numel / N / C)

    # Short-circuits if a is empty
    if a.numel == 0:
        if reduction == "none":
            output_shape = list(a.shape)
            output_shape.pop(C_dim)
            return zeros(output_shape, device=a.device, dtype=a.dtype)
        elif reduction == "mean":
            fill_value = float("nan")
        elif reduction == "sum":
            fill_value = 0.0
        else:
            raise ValueError(f"Reduction argument {reduction} to cross_entropy is not supported")

        return full([], fill_value, device=a.device, dtype=a.dtype)

    if weight is not None:
        utils.check(
            weight.ndim == 1 and weight.numel == C,
            lambda: f"Expected {weight.shape=} to have one dimension and {C} elements",
        )
        bcast_weight = reshape(weight, [C] + [1 for i in range(2, a.ndim)])

    log_softmax_a = log_softmax(a, C_dim)
    out = -log_softmax_a

    if a.shape == target.shape:
        utils.check(
            utils.is_float_dtype(target.dtype),
            lambda: f"expect float dtype for probability target, but got: {target.dtype}!",
        )
        utils.check(
            ignore_index < 0,
            lambda: f"ignore_index is not supported for probability target, set ignore_index < 0!",
        )

        if label_smoothing > 0.0:
            target = target * (1 - label_smoothing) + label_smoothing / C

        out = out * target

        if weight is not None:
            out = out * bcast_weight

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
            return reduced_sum / N * feature_size
        else:
            raise ValueError(f"reduction argument: {reduction} to cross_entropy is not supported")
    else:
        utils.check(
            utils.is_integer_dtype(target.dtype),
            lambda: f"expect integer dtype for class indices target, but got: {target.dtype}!",
        )
        no_C_shape = list(a.shape)
        no_C_shape.pop(C_dim)
        utils.check(
            a.ndim == target.ndim + 1 and no_C_shape == list(target.shape),
            lambda: f"Inconsistent shape input: {a.shape} / target: {a.shape} to cross_entropy!",
        )

        # nll_loss
        if weight is not None:
            out = out * bcast_weight

        smooth_loss_no_sum = out
        # TODO: swap reshape with unsqueeze when nvfuser support is added
        # bcast_target = clang.unsqueeze(target, [C_dim])
        bcast_target_shape = list(a.shape)
        bcast_target_shape[C_dim] = 1
        bcast_target = reshape(target, bcast_target_shape)

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
            mask = bcast_target == ignore_index
            out = where(mask, 0, out)
            if label_smoothing > 0:
                # TODO: switch to squeeze
                smooth_loss = where(reshape(mask, list(smooth_loss.shape)), 0, smooth_loss)

        if reduction == "none":
            # TODO: swap reshape with squeeze when nvfuser support is added
            # return clang.squeeze(out, [C_dim])
            out = reshape(out, target.shape)
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
                input_shape = list(a.shape)
                expanded_weight = clang.expand(bcast_weight, input_shape)
                # DEBUG!!! this gives segfaults
                selected_weight = clang.take_along_axis(expanded_weight, bcast_target, C_dim)

                if ignore_index >= 0:
                    selected_weight = where(mask, 0, selected_weight)

                bcast_weight_sum = _reduction(
                    selected_weight,
                    prims.sum,
                    output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
                )
                out = reduced_sum / bcast_weight_sum
                if label_smoothing > 0:
                    ret = ret / bcast_weight_sum
            elif ignore_index >= 0:
                mask_sum = _reduction(
                    mask,
                    prims.sum,
                    dtype=to_dtype(torch.float),
                    output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
                )
                # NOTE: does the call numel here work with dynamic shape?!
                out = reduced_sum / (target.numel - mask_sum)
                if label_smoothing > 0:
                    ret = ret / (target.numel - mask_sum)
                elif target.ndim == 0:
                    # NOTE: this is pytorch implementation details.
                    # overwrite output to 0 when target hits ignore_index AND label_smoothing is missing.
                    # https://github.com/pytorch/pytorch/pull/64572
                    out = where((target == ignore_index), 0, out)
            else:
                out = reduced_sum / target.numel
                if label_smoothing > 0:
                    ret = ret / target.numel
        else:
            raise ValueError(f"Reduction argument: {reduction} to cross_entropy is not supported")

        # TODO FIXME This is probably incorrect -- but somewhere above the dtype of out can disagree with PyTorch
        out = out.to(a.dtype)

        if label_smoothing > 0:
            return out * (1 - label_smoothing) + (ret * (label_smoothing / C))
        else:
            return out


# TODO The function cross_entropy_backward shouldn't be registered as a primitive operation (above), but as
#   a composite operation
def _cross_entropy_grad(
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
    fwd: TensorLike = cross_entropy(a, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    g: TensorLike = get_grad(fwd)
    a_grad: TensorLike = cross_entropy_backward(g, a, target, weight, reduction, ignore_index, label_smoothing)
    put_grad(a, a_grad)

    return fwd


register_grad(cross_entropy, _cross_entropy_grad)


# TODO Is this a method?
# TODO Move this to nn.functional
# NOTE The id must be explicitly specified so as not to resolve to torch.dropout
#   (Using torch.nn.functional.dropout is just for readability as it's the documented operator)
@torchsymbol(torch.nn.functional.dropout, id="torch.nn.functional.dropout")
def dropout(a: TensorProxy, /, p: Number = 0.5, training: bool = True, inplace: bool = False) -> TensorProxy:
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

    return a * dropout_mask * scale


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
    flatten_indices = reshape(a, [a.numel])
    flatten_output = clang.take(weight, flatten_indices, 0)
    return reshape(flatten_output, output_shape)


@torchsymbol(torch.ops.aten.embedding_backward)
def embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse):
    result = prims.embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse)
    return result


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
        weight is None or (weight.ndim == 1 and weight.numel == num_channels),
        lambda: f"group_norm: {weight.ndim=} should be equal to 1 and {weight.numel=} to {num_channels=}",
    )
    utils.check(
        bias is None or (bias.ndim == 1 and bias.numel == num_channels),
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

    if isinstance(scale_factor, float):
        utils.check(scale_factor > 0, lambda: f"{scale_factor=} is expected to be strictly positive")
        scale_factor = (scale_factor,) * dim
    else:
        utils.check(
            (
                isinstance(scale_factor, Sequence)
                and len(scale_factor) == dim
                and all(isinstance(s, float) and s > 0 for s in scale_factor)
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

    if isinstance(size, int):
        utils.check(size > 0, lambda: f"{size=} is expected to be greater than zero")
        size = (size,) * dim
    else:
        utils.check(
            (isinstance(size, Sequence) and len(size) == dim and all(isinstance(s, int) and s > 0 for s in size)),
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
) -> TensorLike:
    utils.check(
        mode == "nearest",
        lambda: f"only mode='nearest' is supported at the moment, but got {mode=}",
        exception_type=NotImplementedError,
    )

    utils.check(a.ndim >= 3, lambda: f"Expected {a.ndim=} >= 3")
    utils.check(a.numel > 0, lambda: f"Expected {a.numel=} to be greater than 0")

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
    if a.numel == 0 and target.numel == 0:
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

    # channels dimension is either the first one if no batch dim present (i.e. a.shape[0]),
    # or right next to it (i.e. a.shape[1]).
    channels_dim = 1 if a.ndim >= 2 else 0
    num_channels = a.shape[channels_dim]
    # target should match input in dims which do not correspond to the channels dim, i.e.
    # (input.shape[:channels_dim] + input.shape[channels_dim + 1:]) == target.shape <=> True
    expected_target_shape = a.shape[:channels_dim] + a.shape[channels_dim + 1 :]

    utils.check(
        expected_target_shape == target.shape,
        lambda: f"Expected the target tensor to have the same shape as the input tensor except for the channels dimension \
            {expected_target_shape}, but it has shape {target.shape}.",
    )

    utils.check(
        weight is None or (weight.ndim == 1 and weight.shape[0] == num_channels),
        lambda: f"Expected a 1D tensor with {num_channels} elements for weight argument, \
            but found a tensor with {weight.ndim} dimensions and {weight.shape[0]} elements.",
    )

    # NOTE: [Handling of 'ignore_index' parameter]
    # What does it mean to ignore an index?
    #   The 'ignore_index' parameter specifies a target value that does not contribute to input gradient.
    # 'ignore_index' can be outside of the [0, num_channels) range, which can cause out-of-bounds errors when gathering
    # values from input tensor.
    #
    # What does ATen do?
    #   ATen prevents nll_loss from having these errors by skipping target values that match ignore_index first before
    # indexing the input tensor.
    #
    # What do we do?
    #   We mask the ignore_index entries on the output tensor from take_along_axis because we expect the targets to be
    # within [0, num_channels) range.
    #
    # Why do we like our approach better?
    #   Mimicking Aten behavior requires masking the target tensor before calling take_along_axis, which would add more
    # operations to the fusion. We should follow this approach until we see real examples where ignore_index is
    # out-of-bounds of [0, num_channels) range.
    #
    # What are the alternative options?
    #   We can add a `mode` parameter to take_along_axis that controls how to handle out-of-bounds indices.
    # The jax.numpy.take_along_axis has this feature.

    out = -a

    if weight is not None:
        bcast_weight = reshape(weight, [num_channels] + [1 for _ in range(2, a.ndim)])
        out = out * bcast_weight

    # Make target broadcastable with output, which has same shape as input tensor.
    bcast_target = unsqueeze(target, channels_dim)

    out = take_along_dim(out, bcast_target, channels_dim)
    selected_target_mask = bcast_target != ignore_index
    out = where(selected_target_mask, out, 0)

    # This section handles applying the reduction parameter to the output.
    # We return None for the total_weight when reduction is "none" or "sum" since it is unused in the backwards pass.
    if reduction == "none":
        return squeeze(out, channels_dim), None
    elif reduction == "sum":
        return sum(out), None
    elif reduction == "mean":
        reduced_sum = sum(out)
        if weight is not None:
            # Gather the weights for each target class.
            # Mask the ignored target classes.
            # Sum together all target weights.
            expanded_weight = expand(bcast_weight, a.shape)
            selected_weight = take_along_dim(expanded_weight, bcast_target, channels_dim)
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
@torchsymbol(torch.softmax, torch.nn.functional.softmax, is_method=True)
def softmax(a: TensorLike, /, dim: int, *, dtype: None | dtypeLike = None) -> TensorLike:
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
    _reduceop_triples = (("sum", torch.distributed.ReduceOp.SUM, dist_prims.DistributedReduceOps.SUM),)

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
    ) -> TensorLike | FutureTensorLike:
        group = group if group is not None else torch.distributed.new_group()

        return dist_prims.all_gather(a, group, async_op)

    # NOTE torch.distributed.all_reduce is an inplace operation (although the underlying NCCL
    #   call does not need to be inplace). This, however, is modeled as an out-of-place functional
    #   operation, hence the id "functional_all_reduce", and why we do not translate PyTorch
    #   calls directly to this.
    # PyTorch uses torch.ops.c10d_functional.all_reduce as an ID for a similar operation.
    # This operation is based on torch.distributed.all_reduce, see:
    #   https://pytorch.org/docs/master/distributed.html#torch.distributed.all_reduce
    @torchsymbol(
        is_method=False,
        id="functional_all_reduce",
    )
    def all_reduce(
        a: TensorLike,
        /,
        op: DistributedReduceOpLike = torch.distributed.ReduceOp.SUM,
        group: None | torch.distributed.ProcessGroup = None,
        async_op: bool = False,
    ) -> TensorLike | FutureTensorLike:
        op = to_thunder_distributed_reduce_op(op)
        group = group if group is not None else torch.distributed.new_group()

        return dist_prims.all_reduce(a, op, group, async_op)

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
    ) -> TensorLike | FutureTensorLike:
        op = to_thunder_distributed_reduce_op(op)
        group = group if group is not None else torch.distributed.new_group()

        return dist_prims.reduce_scatter(a, op, group, async_op)

else:

    def all_gather(
        a: TensorLike,
        group: Any | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    # NOTE torch.distributed is not available
    def all_reduce(
        a: TensorLike,
        op: Any,
        group: Any | None = None,
        async_op: bool = False,
    ) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

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


#
# torch -> thunder object mapping
#


_torch_to_thunder_complete_map = {
    **_torch_to_thunder_dtype_map,
    **_torch_to_thunder_function_map,
    **{fn: fn for fn in _torch_noinline_functions},
}
