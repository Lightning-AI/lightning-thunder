import operator
from enum import Enum
from functools import partial, reduce, wraps
from numbers import Number
from typing import Callable, Optional, Sequence, Tuple

import torch

import thunder.core.dtypes as dtypes
import thunder.core.lang as tlang
import thunder.core.prims as prims
import thunder.core.proxies as proxies
import thunder.core.trace as trace
import thunder.core.utils as utils
from thunder.core.proxies import TensorProxy
from thunder.core.utils import langctx

__all__ = [
    # Language context
    "ctx",
    "TorchLangCtx",
    # Tensor creation operations
    "arange",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "uniform",
    "zeros_like",
    # Shape ops
    "contiguous",
    "reshape",
    "split",
    "squeeze",
    "tensor_split",
    "transpose",
    "unsqueeze",
    "index_select",
    "view",
    # Elementwise Unary Ops
    "abs",
    "acos",
    "acosh",
    "asin",
    "atan",
    "atanh",
    "bitwise_not",
    "bmm",
    "cos",
    "exp",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "tanh",
    "log",
    "log10",
    "log1p",
    "log2",
    "trunc",
    # Elementwise Binary Ops
    "add",
    "eq",
    "fmod",
    "ge",
    "lt",
    "mul",
    "nextafter",
    "pow",
    "remainder",
    "sub",
    "true_divide",
    # Elementwise Ternary Ops
    "masked_fill",
    "where",
    # Reduction Ops
    "_set_correction",
    "_reduction_dims",
    "amax",
    "amin",
    "mean",
    "prod",
    "sum",
    "var",
    "var_mean",
    # NN Ops
    # TODO: move to torch.nn.functional
    "dropout",
    "embedding",
    "softmax",
    # Norm Ops
    # Matmul Ops
    "linear",
    "matmul",
]

# The Torch language

#
# Language context
#

# TODO: language contexts like Torch could be
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


def to_dtype(x):
    if isinstance(x, torch.Tensor):
        return thunder_dtype(x.dtype)

    return dtypes.to_dtype(x)


def thunder_dtype(torch_dtype):
    return _torch_to_thunder_dtype_map[torch_dtype]


def torch_dtype(thunder_dtype):
    return _thunder_to_torch_dtype_map[thunder_dtype]


_torch_to_thunder_function_map = {}


def torch_function(torch_fn):
    def _torch_function_map_set(thunder_impl):
        # `make_prim` is a helper function that creates a function that is
        # recorded as a single symbol in the trace. Also it adds the
        # `thunder_impl` function to the `ops_to_meta_functions_map` registry
        name = torch_fn.__module__ + "." + torch_fn.__name__
        trace_recording_fn = prims.make_prim(torch_fn, name, thunder_impl)

        @wraps(thunder_impl)
        def wrapper(*args, **kwargs):
            return trace_recording_fn(*args, **kwargs)

        _torch_to_thunder_function_map[torch_fn] = wrapper

        return wrapper

    return _torch_function_map_set


def ctx():
    return TorchLangCtx()


class TorchLangCtx:
    # NOTE: language context is a singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        self.dtype_cls = torch.dtype
        self.tensor_cls = torch.Tensor
        self.module_cls = torch.nn.Module

    def __repr__(self):
        return f"[TorchLangCtx]"

    def proxy(self, x, *, name):
        if isinstance(x, torch.Tensor):
            dtype = thunder_dtype(x.dtype)
            return TensorProxy(name=name, shape=x.shape, device=str(x.device.type), dtype=dtype, strides=x.stride())
        else:
            return proxies.proxy(x, name=name)

    def thunder_dtype(self, torch_dtype):
        return _torch_to_thunder_dtype_map[torch_dtype]

    def torch_dtype(self, thunder_dtype):
        return _thunder_to_torch_dtype_map[thunder_dtype]

    #
    # Tensor methods
    #

    # Attribute accesses
    def size(self, a, dim=None):
        if dim is None:
            return a.shape
        return a.shape[dim]

    def stride(self, a):
        return a.strides

    #
    # Shape Methods
    #

    def contiguous(self, a):
        return _contiguous_disambiguator(a)

    # TODO: assumes basic indexing, add advanced indexing support
    def get_item(self, a, key):
        return basic_indexing(a, key)

    def matrix_transpose(self, a):
        return tlang.matrix_transpose(a)

    # TODO: refactor so disambiguator's aren't needed
    def split(self, a, sizes_or_sections, dim=0):
        return _split_disambiguator(a, sizes_or_sections, dim)

    def transpose(self, a, dim0, dim1):
        return _transpose_disambiguator(a, dim0, dim1)

    def unsqueeze(self, a, dim):
        return _unsqueeze_disambiguator(a, dim)

    def view(self, a, *shape):
        shape = utils.extract_shape_from_varargs(shape)
        return _view_disambiguator(a, shape)

    def reshape(self, a, *shape):
        shape = utils.extract_shape_from_varargs(shape)
        # view is in fact implemented as reshape (?)
        return tlang.reshape(a, shape)

    #
    # Elementwise Unary Methods
    #
    def abs(self, a):
        return tlang.abs(a)

    def acos(self, a):
        return tlang.acos(a)

    def acosh(self, a):
        return tlang.acosh(a)

    def asin(self, a):
        return tlang.asin(a)

    def atan(self, a):
        return tlang.atan(a)

    def atanh(self, a):
        return tlang.atanh(a)

    def bitwise_not(self, a):
        return tlang.bitwise_not(a)

    def cos(self, a):
        return tlang.cos(a)

    def exp(self, a):
        return tlang.exp(a)

    #
    # Elementwise Binary Methods
    #

    # +
    def add(self, a, b):
        return tlang.add(a, b)

    # ==
    def eq(self, a, b):
        return tlang.eq(a, b)

    # fmod
    def fmod(self, a, b):
        return tlang.fmod(a, b)

    # >=
    def ge(self, a, b):
        return tlang.ge(a, b)

    # <
    def lt(self, a, b):
        return tlang.lt(a, b)

    # *
    def mul(self, a, b):
        return tlang.mul(a, b)

    # negation
    def neg(self, a):
        return tlang.neg(a)

    # next floating point value from a in the direction of b
    def nextafter(self, a, b):
        return tlang.nextafter(a, b)

    # **
    def pow(self, a, b):
        return tlang.pow(a, b)

    # -
    def sub(self, a, b):
        return tlang.sub(a, b)

    # /
    def true_divide(self, a, b):
        return tlang.true_divide(a, b)

    #
    # Elementwise ternary methods
    #

    def masked_fill(self, a, mask, value):
        return masked_fill_disambiguator(a, mask, value)

    #
    # Reduction Methods
    #

    def var(self, *args, **kwargs):
        return var(*args, **kwargs)

    #
    # Matmul methods
    #

    def linear(self, *args, **kwargs):
        return linear_disambiguator(*args, **kwargs)

    def matmul(self, *args, **kwargs):
        return matmul_disambiguator(*args, **kwargs)


#
# Tensor Creation Ops
#
@torch_function(torch.arange)
def arange(start, end, step=1, *, device="cpu", dtype=None):
    return tlang.arange(start=start, step=step, stop=end, device=device, dtype=dtype)


# TODO: switch to zeros for readability (after zeros is implemented)
def empty(shape, *, device, dtype=None):
    return full(shape, 0.0, device=device, dtype=dtype)


def empty_like(tensor, *, device=None, dtype=None):
    return zeros_like(tensor, device=device, dtype=dtype)


# TODO: check these signatures
def full(shape, fill_value, *, device, dtype=None):
    return tlang.full(shape, fill_value, device=device, dtype=dtype)


def full_like(tensor, fill_value, *, device=None, dtype=None):
    return tlang.full_like(tensor, fill_value, device=device, dtype=dtype)


# TODO: based on uniform_, check if Torch now has a functional uniform
# NOTE: the uniform_ documentation suggests the interval is specified using "from" and "to",
#   but from is a reserved keyword in Python
def uniform(shape, minval=0.0, maxval=1.0, *, device, dtype):
    return tlang.uniform(shape, minval, maxval, device=device, dtype=dtype)


# TODO: maybe just make this a passthrough?
def zeros_like(tensor, *, device=None, dtype=None):
    return full_like(tensor, 0.0, device=device, dtype=dtype)


#
# Shape Ops
#


def _contiguous_disambiguator(a):
    return contiguous(a)


# TODO: create proper contiguous with stride modeling and memory format support
def contiguous(a):
    return a


# TODO: should this allow negative steps?
# TODO: we should probably be consistent about start/stop/step vs. start/end/stride language
def basic_indexing(a, key):
    start_indices = []
    end_indices = []
    strides = []

    # Resolves ellipses and unsqueezes
    unsqueeze_dims_pre_ellipsis = []
    unsqueeze_dims_post_ellipsis = []
    specified_slices = 0
    ellipsis_idx = None
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
        a = tlang.unsqueeze(a, unsqueeze_dims)

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


def reshape(a, shape):
    return tlang.reshape(a, shape)


# TODO: consider revising this to just call _split_indices
# Splits a tensor along a split dimension dim into n tensors
# If input is divisible by n then every tensor will have the same length along the split dimension
# If input is not divisible by n, then the first int(input.size(dim) % n) tensors will have length
#   int(input.size(dim) / n) + 1 along the split dimension, and the remaining tensors will have
#   length int(input.size(dim) / n) along the split dimension
def _split_n(a, n, dim=0):
    dim = utils.canonicalize_dim(a.ndim, dim)

    splits = []
    dim_length = a.shape[dim]
    min_split_size = dim_length // n
    num_splits_one_extra = dim_length % n
    start_idx = 0
    for split_idx in range(n):
        split_size = min_split_size + 1 if (split_idx < num_splits_one_extra) else min_split_size
        s = tlang.slice_in_dim(a, start_idx, start_idx + split_size, dim=dim)
        splits.append(s)
        start_idx = start_idx + split_size

    return tuple(splits)


# TODO: could this (and other things) be revised to combine the slice_in_dim calls?
# Splits a tensor along a split dimension dim at the indices in indices
def _split_indices(a, indices, dim=0):
    dim = utils.canonicalize_dim(a.ndim, dim)

    splits = []
    start_idx = 0
    for idx in indices:
        splits.append(tlang.slice_in_dim(a, start_idx, idx, dim=dim))
        start_idx = idx

    splits.append(tlang.slice_in_dim(a, start_idx, a.shape[dim], dim=dim))
    return tuple(splits)


def _split_disambiguator(*args, **kwargs):
    return split(*args, **kwargs)


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
            splits.append(tlang.slice_in_dim(a, cur_idx, cur_idx + target_length, dim=dim))
            cur_idx = cur_idx + target_length

        # Handles tail
        if last_length > 0:
            splits.append(tlang.slice_in_dim(a, cur_idx, a.shape[dim], dim=dim))

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


# See https://pytorch.org/docs/master/generated/torch.squeeze.html
def squeeze(a, dim=None):
    dims = dim
    if dim is None:
        dims = []
        for idx, l in enumerate(a.shape):
            if l == 1:
                dims.append(idx)
    if isinstance(dim, Number):
        dims = (dim,)

    return tlang.squeeze(a, dims)


# See https://pytorch.org/docs/master/generated/torch.tensor_split.html
def tensor_split(a, indices_or_sections, dim=0):
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


def _transpose_disambiguator(*args, **kwargs):
    return transpose(*args, **kwargs)


def transpose(a, dim0, dim1):
    dim0, dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))

    permutation = list(range(0, a.ndim))
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    return tlang.transpose(a, permutation)


def _unsqueeze_disambiguator(*args, **kwargs):
    return unsqueeze(*args, **kwargs)


def unsqueeze(a, dim):
    return tlang.unsqueeze(a, (dim,))


@torch_function(torch.index_select)
def index_select(a, dim, index):
    return tlang.index_select(a, dim, index)


def _view_disambiguator(*args, **kwargs):
    return view(*args, **kwargs)


# TODO: review view functionalization
def view(a, shape):
    return reshape(a, shape)


#
# Elementwise Unary Ops
#
@torch_function(torch.abs)
def abs(a):
    return tlang.abs(a)


@torch_function(torch.acos)
def acos(a):
    return tlang.acos(a)


@torch_function(torch.cos)
def cos(a):
    return tlang.cos(a)


@torch_function(torch.exp)
def exp(a):
    return tlang.exp(a)


@torch_function(torch.rsqrt)
def rsqrt(a):
    return tlang.rsqrt(a)


@torch_function(torch.sigmoid)
def sigmoid(a):
    return tlang.sigmoid(a)


@torch_function(torch.sign)
def sign(a):
    return tlang.sign(a)


@torch_function(torch.sin)
def sin(a):
    return tlang.sin(a)


@torch_function(torch.sinh)
def sinh(a):
    return tlang.sinh(a)


@torch_function(torch.sqrt)
def sqrt(a):
    return tlang.sqrt(a)


@torch_function(torch.tanh)
def tanh(a):
    return tlang.tanh(a)


@torch_function(torch.log)
def log(a):
    return tlang.log(a)


@torch_function(torch.log10)
def log10(a):
    return tlang.log10(a)


@torch_function(torch.log1p)
def log1p(a):
    return tlang.log1p(a)


@torch_function(torch.log2)
def log2(a):
    return tlang.log2(a)


@torch_function(torch.trunc)
def trunc(a):
    return tlang.trunc(a)


#
# Elementwise Binary Ops
#


@torch_function(torch.add)
def add(a, b, *, alpha=None):
    if alpha is not None:
        b = b * alpha

    return a + b


@torch_function(torch.eq)
def eq(a, b):
    return tlang.eq(a, b)


@torch_function(torch.fmod)
def fmod(a, b):
    return tlang.fmod(a, b)


@torch_function(torch.ge)
def ge(a, b):
    return tlang.ge(a, b)


@torch_function(torch.lt)
def lt(a, b):
    return tlang.lt(a, b)


@torch_function(torch.mul)
def mul(a, b):
    return tlang.mul(a, b)


@torch_function(torch.neg)
def neg(a):
    return tlang.neg(a)


@torch_function(torch.nextafter)
def nextafter(a, b):
    return tlang.nextafter(a, b)


@torch_function(torch.pow)
def pow(a, b):
    return tlang.pow(a, b)


# A composite operation that matches PyTorch, Jax, and Numpy remainder
# torch.remainder(a, b) == a - a.div(b, rounding_mode="floor") * b
@torch_function(torch.remainder)
def remainder(a, b):
    mod = tlang.fmod(a, b)
    lhs = tlang.bitwise_not(mod == 0)
    rhs = tlang.bitwise_not((b < 0) == (mod < 0))
    mask = tlang.bitwise_and(lhs, rhs)
    return tlang.where(mask, add(mod, b), mod)


@torch_function(torch.sub)
def sub(a, b):
    return tlang.sub(a, b)


@torch_function(torch.true_divide)
def true_divide(a, b):
    return tlang.true_divide(a, b)


#
# Elementwise ternary prims
#


def masked_fill_disambiguator(a, mask, value):
    return masked_fill(a, mask, value)


# NOTE: masked_fill is a strange wrapper around where, it probably exists only because of PyTorch's inplace pattern
# NOTE: PyTorch's masked fill requires value be a number or number tensor
# NOTE: PyTorch's masked fill is only defined as a tensor method that implicitly takes a as the first argument
# NOTE: PyTorch's masked_fill_ requires the dtype of a not change, so it checks that
#   value can be safely cast to a
# TODO: PyTorch's masked_fill always returns a contiguous tensor
# TODO: add number tensor support
def masked_fill(a, mask, value):
    if isinstance(value, TensorProxy):
        raise NotImplementedError

    result = where(mask, value, a)
    return result


def where(pred, a, b):
    return tlang.where(pred, a, b)


#
# Reduction Ops
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


def _reduction_dims(shape, dims: Optional[Sequence]) -> Tuple[int, ...]:
    if dims is None or len(dims) == 0:
        return tuple(range(len(shape)))

    dims = tuple(utils.canonicalize_dim(len(shape), idx) for idx in dims)
    utils.check_no_duplicates(dims)

    return dims


# TODO: restore out support?
def _reduction(
    a,
    prim: Callable,
    *,
    has_identity: bool = True,
    accepts_dim_tuple: bool = True,  # to handle min/argmin that accept single dim only
    dims=None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,  # should be specified for ops that support it
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
):
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

    a = tlang.maybe_convert_to_dtype(a, computation_dtype)
    result = prim(a, dims)

    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)

    if result_dtype is not None:
        result = tlang.maybe_convert_to_dtype(result, result_dtype)

    return result


# Helper to handle the unbiased->correction deprecation on ops like var
def _set_correction(
    unbiased: Optional[bool] = None,
    correction: Optional[int] = None,
):
    if correction is not None and unbiased is not None:
        raise RuntimeError("cannot specify both correction and unbiased arguments")
    elif correction is None and unbiased is None:
        correction = 1
    elif correction is None and unbiased is not None:
        correction = 0 if unbiased is False else 1
    if not isinstance(correction, int):
        raise ValueError("correction argument should be integer")
    if correction < 0:
        raise ValueError("correction argument should be non-negative")
    return correction


def _dim_var_dispatch(dim=None, unbiased=None):
    # There's the following overload of torch.var:
    # var(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
    # We need to explicitly convert bool dims to unbiased arg
    if unbiased is None and isinstance(dim, bool):
        unbiased = dim
        dim = None
    return dim, unbiased


def amax(a, dim, keepdim):
    return _reduction(
        a,
        prims.amax,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def amin(a, dim, keepdim):
    return _reduction(
        a,
        prims.amin,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def mean(a, dim=None, keepdim: bool = False, *, dtype=None):
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
    result = tlang.maybe_convert_to_dtype(result, result_dtype)
    return result


def prod(a, dim=None, keepdim=False, *, dtype=None):
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


def sum(a, dim=None, keepdim=False, *, dtype=None):
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


def var(
    a,
    dim=None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
):
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
# TODO: use of @langctx here is just for testing and could be removed
#  (the method call to var below would need to be replaced with a function call)
@langctx(ctx())
def var_mean(
    a,
    dim=None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[int] = None,
):
    # TODO: programmatically add this redirection to all operations
    # TODO: avoid string construction
    intercepted = trace.get_executor_context().intercept("torch.var_mean")
    if intercepted is not None:
        return intercepted(a, dim, unbiased, keepdim, correction=correction)

    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    v = var(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return v, m


#
# NN Ops
#
# def _uniform_helper(shape, low=0., high=1., *, dtype, device):
#     utils.validate_shape(shape)

#     assert isinstance(low, Number)
#     assert isinstance(high, Number)

#     return prims._uniform_helper(shape, low=low, high=high, dtype=dtype, device=device)


def _dropout_helper(self, val):
    """Helper function for all dropout-type operators. During training, some of the elements of the input tensor are
    randomly masked.

    Returns the masked tensor of the boolean values.
    """

    r = uniform(self.shape, 0.0, 1.0, dtype=dtypes.float32, device=self.device)
    result = r < val

    return result


# full torch signature is: a, p, training, inplace
@torch_function(torch.nn.functional.dropout)
def dropout(a, p=0.5, training=True, inplace=False):
    if not training or inplace:
        raise NotImplementedError("only training=True, inplace=False is supported in dropout")
    utils.check(
        p <= 1 and p >= 0,
        lambda: f"dropout probability has to be between 0 and 1, but got, {p}",
    )

    if p == 1:
        return zeros_like(a)

    if p == 0:
        return a

    scale = 1 / (1 - p)
    dropout_mask = _dropout_helper(a, 1 - p)

    return a * dropout_mask * scale


@torch_function(torch.nn.functional.embedding)
def embedding(a, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
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


# CompositeImplicitAutograd - don't register decomp
@torch_function(torch.nn.functional.softmax)
def softmax(a, dim, dtype=None):
    result_dtype = dtype or a.dtype
    computation_dtype = utils.get_computation_dtype(result_dtype)
    a_ = tlang.maybe_convert_to_dtype(a, computation_dtype)

    if a.numel() == 0:
        a_exp = exp(a_)
    else:
        a_max = amax(a_, dim, keepdim=True)
        a_exp = exp(a_ - a_max)

    result = true_divide(a_exp, sum(a_exp, dim, keepdim=True))
    converted = tlang.maybe_convert_to_dtype(result, result_dtype)
    return converted


#
# Norm Ops
#


def _normalize(a, norm_dims, eps):
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
    a_acc = tlang.maybe_convert_to_dtype(a, computation_dtype)
    biased_var, mean = var_mean(a_acc, dim=norm_dims, unbiased=False, keepdim=True)
    rstd = rsqrt(biased_var + eps)
    out = (a - mean) * rstd
    return out, mean, rstd


# TODO: likely want to refactor these normalizations
def native_layer_norm(a, normalized_shape, weight, bias, eps):
    # Validates inputs
    normalized_ndim = len(normalized_shape)
    utils.check(normalized_ndim >= 1, lambda: f"Expected normalized_shape={normalized_shape} to have length >= 1!")
    # NOTE: canonicalizes the container for comparison to a tuple since
    # (1, 2, 3) != [1, 2, 3]
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

    out = tlang.maybe_convert_to_dtype(out, a.dtype)
    # TODO: review why this conversion cpu only?
    # if input.device.type == "cpu":
    mean = tlang.maybe_convert_to_dtype(mean, a.dtype)
    rstd = tlang.maybe_convert_to_dtype(rstd, a.dtype)

    return out, mean, rstd


@torch_function(torch.nn.functional.layer_norm)
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return native_layer_norm(input, normalized_shape, weight, bias, eps)[0]


#
# Matmul Ops
#
def linear_disambiguator(*args, **kwargs):
    return linear(*args, **kwargs)


@torch_function(torch.nn.functional.linear)
def linear(a, w, bias=None):
    return prims.linear(a, w, bias)


def matmul_disambiguator(*args, **kwargs):
    return matmul(*args, **kwargs)


# NOTE: this wrapper for prim matmul just broadcasts batch dimensions
def matmul(a, b):
    if a.ndim == 1 or b.ndim == 1:
        return prims.matmul(a, b)

    a_batch_dims = a.shape[:-2]
    b_batch_dims = b.shape[:-2]

    batch_dims_broadcast = list(tlang.compute_broadcast_shape(a_batch_dims, b_batch_dims))

    a_broadcast_shape = batch_dims_broadcast + list(a.shape[-2:])
    if not utils.same_shape(a_broadcast_shape, a.shape):
        a = tlang.expand(a, a_broadcast_shape)

    b_broadcast_shape = batch_dims_broadcast + list(b.shape[-2:])
    if not utils.same_shape(b_broadcast_shape, b.shape):
        b = tlang.expand(b, b_broadcast_shape)

    return prims.matmul(a, b)


# TODO: make a proper bmm (which is much more restricted than matmul)
@torch_function(torch.bmm)
def bmm(a, b):
    return prims.matmul(a, b)


#
# torch -> thunder object mapping
#

_torch_to_thunder_complete_map = {
    **_torch_to_thunder_dtype_map,
    **_torch_to_thunder_function_map,
}
