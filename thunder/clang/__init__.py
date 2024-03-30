import math
from functools import reduce
from numbers import Number
from typing import Union, List, Optional, Any
from collections.abc import Callable
from collections.abc import Sequence
from collections import namedtuple
import operator
from types import EllipsisType, NoneType
import copy
import time
import warnings

from thunder.core.baseutils import run_once
from thunder.core.compile_data import using_symbolic_values
from thunder.clang.langctx import register_method
from thunder.core.langctxs import langctx, Languages

import thunder.core.dtypes as dtypes
from thunder.core import utils
import thunder.core.prims as prims
from thunder.core.proxies import TensorProxy, pyval, pytype, proxy, AnyProxy, Proxy
import thunder.core.devices as devices

# This file defines the operations in thunder.jit's "core" language.
#
# These operators are intended to be used when defining user-facing languages, like the torch or NumPy
# languages.

__all__ = []

TensorLike = TensorProxy
DeviceLike = Union[str, devices.Device]

_clang_fn_set: set = set()


# Decorator that sets the core language context and registers the function
class clangop:
    def __init__(self, *, method_name: None | str = None):
        self.method_name: None | str = method_name

    def __call__(self, fn: Callable) -> Callable:
        _fn = langctx(Languages.CLANG)(fn)
        _clang_fn_set.add(_fn)

        if self.method_name is not None:
            register_method(self.method_name, _fn)

        return _fn


#
# Unpacking operations
#


# Checks a tensor's shape and metadata (for use with "constant value" caching)
@clangop()
def check_tensor_shape_and_metadata(t: TensorProxy, /) -> None:
    return prims.check_tensor_shape_and_metadata(
        t, tuple(t.shape), str(t.device), dtypes.to_torch_dtype(t.dtype), t.requires_grad
    )


@clangop()
def check_none(p: AnyProxy, /) -> None:
    return prims.check_none(p)


@clangop()
def check_literal_like(p: AnyProxy, v: Any, /) -> None:
    return prims.check_literal_like(p, v)


@clangop()
def check_type(x: Any, typ: type, /) -> None:
    return prims.check_type(x, typ)


@clangop()
def check_instance(x: Any, types: tuple[type], /) -> None:
    return prims.check_instance(x, types)


# Checks a number's value
@clangop()
def check_number_type_and_value(n: Number, value: Number, /) -> None:
    return prims.check_number_type_and_value(n, value)


@clangop()
def check_string_value(s: str, value: str, /) -> None:
    return prims.check_string_value(s, value)


@clangop()
def unpack_tuple(tup: tuple, /) -> tuple:
    return prims.unpack_tuple(tup)


@clangop()
def unpack_list(lst: list, /) -> list:
    return prims.unpack_list(lst)


@clangop()
def unpack_dict_key(d: dict, key: int | str, /) -> Proxy:
    return prims.unpack_dict_key(d, key)


@clangop()
def check_empty(seq: tuple | list, /) -> None:
    return prims.check_empty(seq)


@clangop()
def construct_tuple(tup: tuple, /) -> tuple:
    return prims.construct_tuple(tup)


#
# Data movement and transformation operations
#


# TODO Review revising enforce_safe_casting to be more like NumPy's
@clangop()
def maybe_convert_to_dtype(a, dtype, *, enforce_safe_casting=False):
    """If a has the same dtype as the given dtype, returns a unmodified.

    Otherwise returns a converted to the given dtype.
    """

    utils.check(utils.is_dtype(dtype), lambda: f"Unknown dtype {dtype}!")

    if isinstance(a, Sequence):
        return tuple(maybe_convert_to_dtype(x, dtype) for x in a)
    if isinstance(a, TensorProxy):
        # Translates numbertypes to dtypes
        if dtypes.is_numbertype(dtype):
            dtype = dtypes.numbertype_to_dtype(dtype)
    elif isinstance(a, Number):
        # NOTE This allows conversions like (5, float32) -> 5., which is a little odd
        dtype = utils.dtype_to_numbertype(dtype)
    else:
        raise ValueError(
            f"Trying to convert the type of the data of an unknown object {a} of {type(a)} that is neither a tensor, number, or sequence!"
        )

    if not utils.are_same_dtypes(a, dtype):
        if enforce_safe_casting:
            utils.check(
                utils.can_safe_cast_to(cast_from=utils.to_dtype(a), cast_to=dtype),
                lambda: f"Can't safe case from a={a} with dtype {utils.to_dtype(a)} to {dtype}!",
            )

        return prims.convert_element_type(a, dtype)

    return a


# TODO Consider maybe_device_put analogous to maybe_convert_to_dtype above
@clangop()
def device_put(a, device):
    if device is not None and a.device != device:
        return prims.device_put(a, device)
    return a


#
# Tensor creation operations
#
# TODO Is there a good helper/wrapper for _like functions?


# TODO Add type annotations
@clangop()
def arange(*, start: Number, step: Number, stop: Number, device: DeviceLike, dtype: dtypes.dtype | None = None):
    # Validates inputs
    # Checks that start, step, and stop are finite
    # TODO Semantically an infinite step seems fine?
    utils.check(math.isfinite(start), lambda: f"start={start} was non-finite")
    utils.check(math.isfinite(step), lambda: f"step={step} was non-finite")
    utils.check(math.isfinite(stop), lambda: f"stop={stop} was non-finite")

    # Checks that start, step, and stop are not complex
    utils.check(not isinstance(start, complex), lambda: f"start={start} was complex")
    utils.check(not isinstance(step, complex), lambda: f"step={step} was complex")
    utils.check(not isinstance(stop, complex), lambda: f"stop={stop} was complex")

    # Checks that step makes progress
    utils.check(
        (start == stop) or (step < 0 and stop < start) or (step > 0 and stop > start),
        lambda: f"step={step} must make progress from start={start} to stop={stop}",
    )

    # Canonicalizes device
    if isinstance(device, str):
        device = devices.Device(device)

    # (Optionally) infers dtype
    # TODO Replace with default datatypes for integer and float
    if dtype is None:
        if all(tuple(isinstance(x, int) for x in (start, step, stop))):
            dtype = dtypes.int64
        else:
            dtype = dtypes.float32

    length = math.ceil((stop - start) / step)

    if utils.is_exact_dtype(dtype):
        return prims.iota(
            length,
            start=start,
            step=step,
            device=device,
            dtype=dtype,
        )

    index = prims.iota(
        length,
        start=0,
        step=1,
        dtype=dtypes.int64,
        device=device,
    )

    result = start + index * step
    result = maybe_convert_to_dtype(result, dtype)
    return result


@clangop()
def convolution(
    a: TensorLike,
    weight: TensorLike,
    bias: TensorLike | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> TensorLike:
    return prims.convolution(a, weight, bias, stride, padding, dilation, bool(transposed), output_padding, groups)


@clangop()
def full(
    shape: Sequence[int], fill_value: Number, *, device: DeviceLike, dtype: None | dtypes.dtype = None
) -> TensorLike:
    # Infers dtype from the fill_value when not explicitly provided
    if dtype is None:
        dtype = dtypes.numbertype_to_dtype(dtypes.to_dtype(fill_value))
    device = devices.to_device(device)

    return prims.full(shape, fill_value, device=device, dtype=dtype)


@clangop()
def full_like(
    a: TensorLike | Number,
    fill_value: Number,
    *,
    device: DeviceLike | None = None,
    dtype: dtypes.dtype | None = None,
) -> TensorLike:
    if isinstance(a, Number):
        dtype = pytype(fill_value) if dtype is None else dtypes.dtype_to_numbertype(dtype)
        utils.check(
            device is None or devices.to_device(device).devicetype is devices.DeviceType.CPU,
            lambda: f"Numbers can only be created on the CPU, but found a request for device={device}",
        )
        return maybe_convert_to_dtype(fill_value, dtype)

    device = devices.to_device(device) if device is not None else a.device
    dtype = dtype if dtype is not None else a.true_dtype

    return full(a.shape, fill_value, device=device, dtype=dtype)


@clangop()
def uniform(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypes.dtype,
) -> TensorProxy:
    device = devices.to_device(device)

    return prims.uniform(shape, minval, maxval, device=device, dtype=dtype)


# TODO Handle a being a number
@clangop()
def uniform_like(
    a: TensorProxy,
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: str | devices.Device | None = None,
    dtype: dtypes.dtype | None = None,
):
    device = devices.to_device(device) if device is not None else a.device
    dtype = dtype if dtype is not None else a.true_dtype

    return prims.uniform(a.shape, minval, maxval, device=device, dtype=dtype)


@clangop()
def uniform_philox(
    shape: Sequence[int],
    minval: Number = 0.0,
    maxval: Number = 1.0,
    *,
    device: DeviceLike,
    dtype: dtypes.dtype,
    seed: int | TensorProxy,
    offset: int | TensorProxy,
) -> TensorProxy:
    device = devices.to_device(device)

    return prims.uniform_philox(
        shape,
        minval,
        maxval,
        device=device,
        dtype=dtype,
        seed=seed,
        offset=offset,
    )


@clangop()
def tensor_from_sequence(
    sequence: Sequence[Number | Sequence], *, dtype: None | dtypes.dtype = None, device: None | DeviceLike = None
) -> TensorLike:
    # NOTE: default the device to `cpu`
    if device is None:
        device = "cpu"
    device: DeviceLike = devices.to_device(device)

    # NOTE: dtype is None means that the prim should infer the dtype.
    dtype: None | dtypes.dtype = dtypes.to_dtype(dtype)

    return prims.tensor_from_sequence(sequence, dtype=dtype, device=device)


#
# Shape operations
#


@clangop()
def diagonal(a: TensorLike, offset: int = 0, dim1: int = 0, dim2: int = 1) -> TensorLike:
    utils.check(
        a.ndim >= 2,
        lambda: f"diagonal() expected a tensor with at least two dimensions, but got a tensor with {a.ndims} dimensions",
    )

    diag_length = max(0, min(a.shape[dim1] + min(offset, 0), a.shape[dim2] - max(offset, 0)))

    a = movedim(a, (dim1, dim2), (-2, -1))

    i = arange(start=0, step=1, stop=diag_length, device=a.device)
    j = arange(start=abs(offset), step=1, stop=(abs(offset) + diag_length), device=a.device)
    if offset >= 0:
        return a[..., i, j]
    return a[..., j, i]


# Expands a to the specified shape, possibly adding new dimensions and expanding
#   dimensions of length 1 to any length
@clangop()
def expand(a: TensorLike, *shape: int) -> TensorLike:
    shape = utils.extract_shape_from_varargs(shape)

    # TODO: improve this error message with error context
    utils.check(
        len(shape) >= len(a.shape),
        lambda: "expand: the requested shape has too few dimensions!",
    )

    offset = len(shape) - len(a.shape)
    shape_ = list(shape)
    for idx, x in enumerate(a.shape):
        offset_idx = idx + offset
        requested_length = shape[offset_idx]
        utils.check(
            requested_length == x or x == 1 or requested_length == -1,
            lambda: f"expand: attempting to expand a dimension of length {x}!",
        )

        shape_[offset_idx] = requested_length if requested_length != -1 else x

    # At this point shape must be valid
    # utils.check_valid_shape(shape_)

    # NOTE: Converting shape_ to tuple makes it possible to apply CSE
    return prims.broadcast_in_dim(a, tuple(shape_), tuple(range(offset, len(a.shape) + offset)))


# TODO Resolve the start & end vs. start & stop inconsistencies with our operators (this one is start & end)
# This is modeled after PyTorch's flatten,
#   see https://pytorch.org/docs/master/generated/torch.flatten.html
# NOTE Flatten is inclusive of both its start and end dims.
@clangop()
def flatten(a: TensorLike, start_dim: int = 0, end_dim: int = -1) -> TensorLike:
    start, end = utils.canonicalize_dims(a.ndim, (start_dim, end_dim))

    num_flattened_dims = end - start
    utils.check(
        num_flattened_dims >= 0, lambda: f"Expected {end_dim=} to specify a more inner dimension than {start_dim=}"
    )

    # NOTE Flattening a number tensor returns a tensor with one dimension in both PyTorch and NumPy
    if a.ndim == 0:
        return unsqueeze(a, 0)

    # Short-circuits if no dimensions are flattened
    if num_flattened_dims == 0:
        return a

    # NOTE end + 1 since end_dim is INCLUSIVE in flatten and exclusive when indexing into a Python sequence
    shape = tuple(a.shape[:start]) + (reduce(operator.mul, a.shape[start : end + 1]),) + tuple(a.shape[end + 1 :])

    # NOTE Instead of computing the length of the flattened dimension this could just insert a -1 and let
    #   reshape's logic figure it out
    return reshape(a, shape)


@clangop()
def flip(a: TensorLike, dims: Sequence[int] | int | None = None) -> TensorLike:
    if dims is not None:
        if isinstance(dims, int):
            dims = (dims,)
        elif isinstance(dims, Sequence) and len(dims) == 0:
            # Short circuit when dims is an empty Sequence
            return a

        dims = utils.canonicalize_dims(a.ndim, dims)
        return prims.flip(a, dims)
    else:
        # Flip over all of the dims
        return prims.flip(a, tuple(range(a.ndim)))


# IndexingSignature stores a meta-data for an indexing key.
# It is a named tuple with the following fields:
#
# unsqueeze - a sequence of indices in the key that have value None.
# These are dimensions that are supposed to be inserted into the
# indexing result.
#
# basic - a sequence of pairs (a_dim, key_idx) that indicate that the a_dim'th
# dimension of the input expects an application of basic indexing key[key_idx].
# (a_dim, key_idx) == (None, None) if key is an instance of Number or slice.
# See _get_indexing_signature for more on what constitutes a basic indexing.
#
# advanced - a sequence of pairs (a_dim, key_idx) that indicate that the a_dim'th
# dimension of the input expects an application of advanced indexing key[key_idx].
# (a_dim, key_idx) == (None, None) if key is an instance of TensorLike or
# a sequence which is not a tuple.
# See _get_indexing_signature for more on what constitutes a basic indexing.
IndexingSignature = namedtuple("IndexingSignature", ["basic", "advanced", "unsqueeze"])


# Given an indexing key, partition it into regions of basic/advanced indexing
# and a region that corresponds to newly inserted dimensions.
# Returns IndexingSignature.
def _get_indexing_signature(key: Any) -> IndexingSignature:
    sig = IndexingSignature([], [], [])

    # key is None or Ellipsis -> indexing is a no-op,
    # and we just return an empty signature.
    if isinstance(key, (type(None), EllipsisType)):
        return sig

    # Numbers and slices are examples of basic indexing.
    if isinstance(key, (Number, slice)):
        sig.basic.append((None, None))
        return sig

    # TensorLike triggers advanced indexing.
    if isinstance(key, TensorLike):
        sig.advanced.append((None, None))
        return sig

    utils.check_type(key, Sequence)

    # Sequences which are not tuples trigger advanced indexing.
    if not isinstance(key, tuple):
        sig.advanced.append((None, None))
        return sig

    # We use this iterator over key for convenient signature population.
    # It returns pairs (i, key[i]) where
    # i = 0, ..., (position of Ellipsis) (left-to-right indexing of key), and
    # i = -1, ..., (negative position of Ellipsis + 1) (right-to-left indexing of key
    # with negative indices).
    class IndexingKeyIter:
        def __init__(self, k):
            self.k = k

        def __iter__(self):
            self.k_iter = zip(range(len(self.k)), self.k)
            return self

        def __next__(self):
            i, v = next(self.k_iter)
            if v is Ellipsis:
                rest_k = self.k[i + 1 :]
                self.k_iter = zip(range(-1, -len(rest_k) - 1, -1), reversed(rest_k))
            return i, v

    has_ellipses = False

    # See how IndexingKeyIter iterates over key.
    # a_dim corresponds to the input's dimension a basic/advanced
    # is expected to be applied to.
    # We add this information to the signature.
    a_dim = 0
    # advance is applied to a_dim.
    # It is an increment before Ellipsis is seen, and a decrement
    # afterwards.
    advance = lambda dim: dim + 1 if dim >= 0 else dim - 1

    for i, k in IndexingKeyIter(key):
        if k is Ellipsis:
            utils.check(not has_ellipses, lambda: f"Found two (or more) ellipses in {key=}")
            has_ellipses = True
            # Ellipsis is spotted -> iteration direction is changed
            # to iterate from left-most position to the position before Ellipsis.
            # We use negative indices for simplicity.
            a_dim = -1
        elif k is None:
            sig.unsqueeze.append(i)
        else:
            if isinstance(k, (Number, slice)):
                sig.basic.append((a_dim, i))
            elif isinstance(k, (TensorLike, Sequence)):
                sig.advanced.append((a_dim, i))
            else:
                raise ValueError(f"{key[i]=} has unexpected {type(key[i])=}")

            a_dim = advance(a_dim)

    return sig


# TODO: should this allow negative steps?
# TODO: we should probably be consistent about start/stop/step vs. start/end/stride language
# TODO Add type annotations
@clangop()
def _basic_indexing(a: TensorLike, /, key) -> TensorLike:
    start_indices = []
    end_indices = []
    strides = []

    # Resolves ellipses and unsqueezes
    unsqueeze_dims_pre_ellipsis = []
    unsqueeze_dims_post_ellipsis = []
    specified_slices = 0
    ellipsis_idx = None

    if isinstance(key, (Number, slice, EllipsisType)):
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
    # NOTE Both these checks are required
    #   ellipsis_dims > 0 handles the implicit ellipsis matching 1+ dimensions
    #   ellipsis_idx not being None handles an explicit ellipsis which matches no dimensions
    if ellipsis_idx is not None or ellipsis_dims > 0:
        ellipsis_slices = [slice(None, None, None)] * ellipsis_dims
        if ellipsis_idx is not None:
            key = list(key)[:ellipsis_idx] + ellipsis_slices + list(key)[ellipsis_idx + 1 :]
        else:
            # NOTE Without an explicit ellipsis, there is an implicit ellipsis at the end of the key
            key = list(key) + ellipsis_slices

    # Unsqueezes
    unsqueeze_dims_post_ellipsis = [x + ellipsis_dims - 1 for x in unsqueeze_dims_post_ellipsis]
    unsqueeze_dims = unsqueeze_dims_pre_ellipsis + unsqueeze_dims_post_ellipsis
    if len(unsqueeze_dims) > 0:
        a = unsqueeze(a, unsqueeze_dims)

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
            # NOTE Because step is always strictly positive, it's sufficient to check start
            #   and stop only here
            if start > stop:
                start = 0
                stop = 0

            # Handles overflow
            # NOTE This is a little odd, but we just want the slice to be zero
            if start >= l:
                start = 0
                stop = 0

            if stop >= l:
                stop = l

            start_indices.append(start)
            end_indices.append(stop)
            strides.append(step)
        elif isinstance(x, Number):
            # NOTE Numbers must be valid indices after canonicalization, unlike start and stop
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
        result = prims.squeeze(result, tuple(squeeze_dims))

    return result


# TODO Add more advanced indexing support
# TODO Review the modeling of advanced indexing in terms of flatten and take
# NOTE Advanced indexing with boolean tensors has data-dependent metadata (it is akin to indexing with nonzero)
@clangop()
def _advanced_indexing(a: TensorLike, /, key) -> TensorLike:
    # Advanced indexing currently supports the following cases:
    #   - a 0D or 1D integer tensor
    #   - a series of one or more 0D or 1D integer tensors containing at most one ellipsis as the first sequence element and at least one sequence element

    utils.check(
        isinstance(key, (TensorLike, Sequence)),
        lambda: f"Advanced indexing currently only supports keys that are ellipses, integer tensors or sequences, but got {key=}",
    )

    if isinstance(key, (TensorLike)):
        key = utils.sequencify(key)

    # Validates (currently supported) inputs
    utils.check(len(key) > 0, lambda: f"Advanced indexing expected a non-empty sequence for a key, but got {key=}")

    num_ellipses: int = 0
    for x in key:
        if x is Ellipsis:
            num_ellipses += 1
        utils.check(
            isinstance(x, (EllipsisType, TensorLike)),
            lambda: f"Advanced indexing currently only supports tensors as sequence elements (possibly with a starting ellipsis), but found an object of type {type(x)}",
        )
        if isinstance(x, TensorLike):
            utils.check(
                dtypes.is_nonboolean_integer_dtype(x.dtype) and (x.ndim == 1 or x.ndim == 0),
                lambda: f"Advanced indexing currently only supports zero or one-dimensional integer tensors, but found a tensor with dtype {x.dtype} and {x.ndim} dimensions",
            )

    utils.check(num_ellipses <= 1, lambda: f"Found two or more ellipses in an advanced indexing key")

    # NOTE When the key has an ellipsis it can be longer than the number of dimensions in a
    #   (in this case the ellipsis matches no dimensions)
    seq_len: int = len(key)
    utils.check(
        a.ndim >= (seq_len - num_ellipses),
        lambda: f"Trying to (advanced) index into a tensor with {a.ndim} dimensions but with {seq_len - num_ellipses} index tensors",
    )

    # TODO Think about relaxing this
    has_ellipsis: bool = num_ellipses > 0
    utils.check(
        not has_ellipsis or key[0] is Ellipsis,
        lambda: f"Advanced indexing currently only supports ellipses as the first sequence element",
    )

    # The following models two advanced indexing cases:
    #
    #   1) (Ellipsis, 1D tensor, 1D tensor, ...)
    #   2) (1D tensor, 1D tensor, ...)
    #
    # In both cases the tensor is partially or completely flattened and a flattened index is constructed for it,
    #   such that take(flattened_tensor, flattened_index, dim) will compute the indexing operation.
    #
    # In the first case, the dimensions of the tensor corresponding to the tensor sequence elements is flattened,
    #   and the "flattened take" happens in this new innermost dimension. For example, a tensor (4, 5, 2) being indexed into
    #   like (..., a, b) will flatten its two dimensions corresponding to a and b to become (4, 10). Then the take
    #   happens in that innermost dimension.
    #
    # In the second case, the dimensions corresponding to the tensor sequence elements are still flattened, and the
    #   "flattened take" again happens in this new outermost dimension. For example, a tensor (4, 5, 2) being indexed
    #   into like (a, b) will flatten its two dimensions corresponding to a and b and become (20, 2). Then the take
    #   happens in that outermost dimension.
    #
    # Conceptually, this technique relies on the 1D tensor sequence elements being contiguous in the key, which is why it
    #   doesn't support an ellipsis in the middle of the key. There are probably some reasonable generalizations that
    #   would support those use cases. One idea would be to "materialize" the ellipses as a series of 1D tensors,
    #   but actually generating those tensors seems expensive.
    flattened: TensorLike
    subtensor_shape: list[int]
    dim: int
    modified_key = list(key)

    if has_ellipsis:
        del modified_key[0]
        subtensor_shape = a.shape[a.ndim - (seq_len - 1) :]
        dim = -1
        flattened = flatten(a, a.ndim - (seq_len - 1), -1)
    else:
        # NOTE No ellipsis case
        subtensor_shape = a.shape[: len(key)]
        dim = 0
        flattened = flatten(a, 0, seq_len - 1)

    # Canonicalizes tensor indices, wrapping negative indices like -5
    # NOTE This does not check if the indices are valid. In PyTorch invalid indices
    #   will trigger a device-side assertion.
    def wrap_tensor(t: TensorLike, dim_length: int) -> TensorLike:
        return t + where(t < 0, dim_length, 0)

    # NOTE The following code might be a little too complicated. Conceptually it aligns
    #   the key and a subset of the tensor dimensions, like in the following examples:
    #
    #   tensor dims:  a  b  c  d
    #           key: k0 k1 k2
    #
    #   tensor dims: a  b  c  d
    #           key:  ... k0 k1
    #
    # And then it iterates through both in reverse to pair c and k2, b and k1, etc.
    #   to compute the correct flattened index.
    #
    # A detail is that the first pairing is treated specially to initialize the loop variables.
    l = subtensor_shape[-1]
    flattened_idx = wrap_tensor(key[-1], l)
    accum: int = l
    for k, l in zip(reversed(key[:-1]), reversed(subtensor_shape[:-1])):
        wrapped = wrap_tensor(k, l)
        flattened_idx = flattened_idx + wrapped * accum
        accum *= l

    res = take(flattened, flattened_idx, dim=dim)

    # take always keeps the indexed dim.
    # If all keys are 0-dim, this dim has to be squeezed.
    if all(k.ndim == 0 for k in modified_key if isinstance(k, TensorLike)):
        res = squeeze(res, (dim,))
    return res


# NOTE Advanced indexing is triggered whenever:
#   - key is a sequence but not a tuple
#   - key is an tensor
#   - key is a tuple that contains a sequence or tensor
# NOTE: currently supported indexing:
# - all basic indexing.
# - advanced indexing:
#       * 0D or 1D TensorLike indices.
#       * basic indexing + a single index which is a 1-length Sequence.
@clangop(method_name="getitem")
def getitem(a: TensorLike, /, key) -> TensorLike:
    sig = _get_indexing_signature(key)
    utils.check(
        (a.ndim == 0 and (len(sig.basic) + len(sig.advanced)) <= 1) or (a.ndim >= len(sig.basic) + len(sig.advanced)),
        lambda: f"{key=} tries to index more dimensions than {a.ndim=}",
    )

    # We do not support mixing basic and advanced indexing together yet,
    # but a very special case when there is a single advanced index which
    # is a sequence of length 1.
    if len(sig.advanced) == 1 and not isinstance(key, TensorLike):
        (_, key_idx), *_ = sig.advanced
        if key_idx is not None:
            key_idx = key_idx if key_idx >= 0 else len(key) + key_idx
            index = key[key_idx]
            if isinstance(index, Sequence) and len(index) == 1 and isinstance(index[0], Number):
                start = index[0]
                # Hande -1 to avoid empty slices
                if start == -1:
                    end = None
                else:
                    end = start + 1
                # 1-len Sequence -> a slice
                key = tuple(key[:key_idx]) + (slice(start, end),) + tuple(key[key_idx + 1 :])
                return _basic_indexing(a, key)

    utils.check(
        not (len(sig.basic) > 0 and len(sig.advanced) > 0),
        lambda: f"{key=} mixes basic and advanced indexing that is not currently supported",
        NotImplementedError,
    )

    if isinstance(key, TensorLike) or (isinstance(key, Sequence) and not isinstance(key, tuple)):
        return _advanced_indexing(a, key)

    if isinstance(key, tuple):
        for x in key:
            if isinstance(x, TensorLike) or isinstance(x, Sequence):
                return _advanced_indexing(a, key)

    return _basic_indexing(a, key)


# Based on NumPy's https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html
@clangop()
def movedim(a: TensorLike, /, source: int | Sequence[int], destination: int | Sequence[int]) -> TensorLike:
    src, dst = utils.sequencify(source), utils.sequencify(destination)

    utils.check(
        len(src) == len(dst), lambda: f"Received a source {source} and destination {destination} of different lengths"
    )

    src, dst = utils.canonicalize_dims(a.ndim, src), utils.canonicalize_dims(a.ndim, dst)

    # Verifies that dims are uniquely specified
    # NOTE This must be done after canonicalization, since canonicalization resolves different ways of specifying the same dim
    src_set = set(src)
    utils.check(len(src_set) == len(src), lambda: f"Found at least one source dimension specified multiple times")
    utils.check(len(set(dst)) == len(dst), lambda: f"Found at least one destination dimension specified multiple times")

    # Constructs a permutation that moves the dimensions as requested
    # NOTE Essentially move_dim specifies a partial permutation, where dimensions not explicitly specified as moving
    #   are ordered as they are in the original tensor and "fill in" around the explicit permutation

    explicit_permutation_map = dict(zip(dst, src))

    # Creates the "fill in" values
    #   For example, if we have a tensor of rank 5, we can label its dims 0 1 2 3 4
    #   If the explicit permutation specifies how dimensions 2 and 4 move, then the implicit dimeensions
    #   which are "filled in" around the explicitly permuted dimensions are 0 1 and 3, which is what this
    #   iterator returns
    implicit_permutation_generator = iter(sorted(set(range(a.ndim)) - src_set))

    perm = []
    for idx in range(a.ndim):
        explicit_perm: None | int = explicit_permutation_map.get(idx, None)
        if explicit_perm is not None:
            perm.append(explicit_perm)
        else:
            perm.append(next(implicit_permutation_generator))

    return transpose(a, perm)


@clangop()
def pad(input: TensorProxy, padding_value: TensorProxy, padding_config: Sequence[tuple[int, int, int]]) -> TensorProxy:
    return prims.pad(input, padding_value, padding_config)


# NOTE shape may have a single -1 value, which is a marker that the length of that dimension
#   should be inferred
@clangop()
def reshape(a: TensorLike, shape: Sequence[int]) -> TensorLike:
    # Short-circuit on a no-op reshape.
    # Useful to produce simpler traces for complex decompositions
    # like einsum, for example.
    if a.shape == tuple(shape):
        return a

    # Checks for -1 marker value
    numel = 1
    neg_one_idx = None
    for idx, l in enumerate(shape):
        if l >= 0:
            numel *= l
        else:
            utils.check(l == -1, "Found a negative dimension length {l} in shape={shape}!")
            utils.check(neg_one_idx is None, "Found two -1 markers in shape={shape}!")
            neg_one_idx = idx

    # Short-circuits if no shape inference is needed
    if neg_one_idx is None:
        return prims.reshape(a, tuple(shape))

    # Constructs the inferred shape, replacing -1 with the necessary length
    utils.check(a.numel % numel == 0, lambda: f"Trying to reshape, but can't infer how to reshape {a.shape} to {shape}")
    remaining = a.numel // numel
    shape = list(shape)
    shape[neg_one_idx] = remaining
    shape = tuple(shape)
    # NOTE alternatively a new tuple could be constructed as follows:
    # shape = shape[:neg_one_idx] + (remaining,) + shape[neg_one_idx + 1:]
    return prims.reshape(a, shape)


# https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.slice_in_dim.html
# NOTE: this implementation derived from
#   https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/slicing.html#slice_in_dim
@clangop()
def slice_in_dim(a, start_index, limit_index, stride=1, dim=0):
    len_dim = a.shape[dim]
    start_index = utils.canonicalize_dim_idx(len_dim, start_index)
    limit_index = utils.canonicalize_dim_idx(len_dim, limit_index)

    # Handles the start idx being greater than the dimension length by returning
    #   a tensor with no elements
    if start_index >= len_dim:
        shape = list(a.shape)
        shape[dim] = 0
        return full(shape, 0, device=a.device, dtype=a.dtype)

    # Handles the limit idx being greater than the dimension length by clamping it
    if limit_index >= len_dim:
        limit_index = len_dim

    # Constructs args for the slice prim
    start_indices = [0] * a.ndim
    limit_indices = list(a.shape)
    strides = [1] * a.ndim

    start_indices[dim] = start_index
    limit_indices[dim] = limit_index
    strides[dim] = stride

    return prims.slice_prim(a, start_indices, limit_indices, strides)


@clangop()
def squeeze(a, dims):
    dims = utils.canonicalize_dims(a.ndim, dims)
    result = prims.squeeze(a, tuple(dims))
    return result


# NOTE This function is named after NumPy's "transpose", which actually performs a permutation
@clangop()
def transpose(a, permutation):
    permutation = utils.canonicalize_dims(a.ndim, permutation)
    # Short-circuit when transpose is a no-op.
    # Useful for producing simpler traces.
    if tuple(permutation) == tuple(range(a.ndim)):
        return a
    return prims.transpose(a, tuple(permutation))


# TODO Consider moving this out of the core language and into something like NumPy's
#   lib.stride_tricks
@clangop()
def stride_order(a: TensorLike, order: None | Sequence[int] = None) -> TensorLike:
    """Creates a dense, non-overlapping and strided tensor with the same data and metadata as a.

    A dense, non-overlapping and strided tensor has three properties:
        - Dense. Its data is stored contiguously in memory in an array
        - Non-overlapping. Each array element has a unique index in the tensor
        - Strided. The array position of an element x with index index_x is <index_x, strides>, the dot product of its index and the tensor's strides

    Intuitively, dense and non-overlapping tensors are distinguished from arbitrarily strided tensors because they are neither
    "overlapped", where multiple indices refer to the same elements, nor do they have "gaps" where memory addresses between the
    first and last array elements do not point to the tensor's data.

    If a permutation is provided the strides will be ordered (least to greatest) like the permutation.
    If no permutation is provided the strides will be ordered from outermost to innermost (..., 2, 1, 0).
    For example, if a is a 4D tensor with dimensions labeled NCHW, then strided(a, (3, 0, 2, 1)) produces a
    dense, non-overlapping and strided tensor where the C dimension has a corresponding stride of one.

    .. note::

        No other thunder.jit operations specify how their outputs are represented in memory, and thunder.jit
        does not model strides. This operation is an explicit directive to construct a dense, non-overlapping and
        strided tensor, but operations on that tensor do not have to preserve those properties.
    """
    if order is None:
        order = tuple(range(a.ndim - 1, -1, -1))

    return prims.stride_order(a, order)


# expanding `a` to the shape of `ref` except dimension `exclude_dim`
# This is basically broadcasting `a` to `ref`, while preserving the shape at dimension `exclude_dim`
def _maybe_expand_exclude_dim(a: TensorProxy, ref: TensorProxy, exclude_dim: int) -> TensorProxy:
    utils.check(
        a.ndim == ref.ndim, lambda: f"Expected a (rank={a.ndim}) to have the same rank as ref (rank={ref.ndim})"
    )
    target_shape = list(ref.shape)
    target_shape[exclude_dim] = a.shape[exclude_dim]
    if not utils.same_shape(a.shape, target_shape):
        return expand(a, target_shape)

    return a


@clangop()
def index_add(a: TensorProxy, indices: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.index_add(a, indices, value, dim)


@clangop()
def take(a: TensorProxy, indices: TensorProxy, dim: int) -> TensorProxy:
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.take(a, indices, dim)


@clangop()
def take_along_axis(a: TensorProxy, /, indices: TensorProxy, dim: int) -> TensorProxy:
    dim = utils.canonicalize_dim(a.ndim, dim)
    indices = _maybe_expand_exclude_dim(indices, a, dim)
    return prims.take_along_axis(a, indices, dim)


@clangop()
def scatter_add(a: TensorProxy, /, indices: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.scatter_add(a, indices, value, dim)


# NOTE revisit the decomposition when fuser supports index_put
@clangop()
def index_put(
    a: TensorLike, /, indices: Sequence[TensorLike], values: TensorLike, accumulate: bool = False
) -> TensorLike:
    utils.check(
        len(indices) <= a.ndim, lambda: f"Too many indices for tensor of dimension {a.ndim} (got {len(indices)} )"
    )

    # broadcast all index tensors together
    broadcast_indices = maybe_broadcast(*indices)

    # expand values
    # the expand rule is: Left-align the input shape and the index shape,
    # and join the index shape and the remaining input shape to form the expanded_shape. e.g.:
    # a shape:      m n p q
    # index shape   x y
    # expand_shape: x y p q
    if broadcast_indices:
        dims_indexed = len(broadcast_indices)
        expanded_shape = broadcast_indices[0].shape + a.shape[dims_indexed:]
        values = expand(values, expanded_shape)

    return prims.index_put(a, broadcast_indices, values, accumulate)


# Unsqueezes a, adding zero or more dimensions of length 1
# Added dimensions are specified by their position in the final tensor
# Based on https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.expand_dims.html
# NOTE: the dimensions do not have to be specified in any order
@clangop()
def unsqueeze(a, /, dims: int | Sequence[int]) -> TensorProxy:
    if isinstance(dims, Number):
        dims = (dims,)

    # Short-circuits if dims is empty
    if len(dims) == 0:
        return a

    final_rank = a.ndim + len(dims)
    # Canonicalizes and sorts dimensions
    dims = sorted(utils.canonicalize_dims(final_rank, dims))

    # Validates that (canonicalized) dimensions are unique
    utils.check_no_duplicates(dims)

    # Constructs expanded (unsqueezed) shape and determines final position of original dims
    shape = []
    broadcast_dims = []
    dims_idx = 0
    a_idx = 0
    for idx in range(final_rank):
        if dims_idx < len(dims) and dims[dims_idx] == idx:
            shape.append(1)
            dims_idx += 1
        else:
            shape.append(a.shape[a_idx])
            broadcast_dims.append(a_idx + dims_idx)
            a_idx += 1

    return prims.broadcast_in_dim(a, shape, broadcast_dims)


@clangop()
def cat(tensors: list[TensorProxy], dim: int):
    """Concatenates the given sequence of tensors in the given dimension."""
    return prims.cat(tensors, dim)


@clangop()
def stack(tensors: list[TensorProxy], dim: int):
    """Concatenates the given sequence of tensors in a new (the given) dimension."""
    shapes = tuple(t.shape for t in tensors)
    utils.check(shapes, lambda: f"list of tensors cannot be empty")
    for i, s in enumerate(shapes[1:], start=1):
        utils.check(
            s == shapes[0], lambda: f"tensors must be of the same shape, tensor at {i} is {s} instead of {shapes[0]}"
        )
    tensors_ = [unsqueeze(t, dim) for t in tensors]
    return prims.cat(tensors_, dim)


@clangop()
def compute_broadcast_shape(*_shapes):
    """Computes the common shape with the fewest dimensions that all input shapes can be broadcast to."""
    shapes = tuple(x for x in filter(lambda x: x is not None, _shapes))

    # Short-circuits if there are no inputs shapes
    #   This might happen in calls like add(2, 3)
    if len(shapes) == 0:
        return None

    common_shape = [
        1,
    ] * reduce(max, (len(shape) for shape in shapes))

    for shape in shapes:
        for idx in range(-1, -1 - len(shape), -1):
            if common_shape[idx] == 1:
                common_shape[idx] = shape[idx]

            utils.check(
                (shape[idx] == 1) or (common_shape[idx] == shape[idx]),
                lambda: f"Attempting to broadcast a dimension of length {shape[idx]}!",
            )

    return tuple(common_shape)


@run_once
def mT_scalar_warning():
    warnings.warn(
        "Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases.",
        UserWarning,
    )


@clangop(method_name="mT")
def matrix_transpose(a: TensorProxy) -> TensorProxy:
    """Transposes the last two dimensions of a tensor.

    This function is used to implement the `.mT` attribute.

    Args:
        a (TensorProxy): The tensor to transpose.

    Returns:
        TensorProxy: The transposed tensor.

    Examples:
        >>> a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> def func(x): return x.mT
        >>> traced_func = thunder.make_traced(func, executor="torch")
        >>> traced_func(a)
        tensor([[1, 4],
                [2, 5],
                [3, 6]])
    """

    if a.ndim == 0:
        mT_scalar_warning()
        return a
    elif a.ndim == 1:
        raise RuntimeError(f"tensor.mT is only supported on matrices or batches of matrices. Got 1-D tensor.")

    dim0, dim1 = -2, -1
    dim0, dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))
    permutation = list(range(a.ndim))
    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
    return transpose(a, permutation)


# TODO: add scalar support
# TODO: review hasattr pattern
@clangop()
def maybe_broadcast(*args):
    """Returns tensors with the same shape, possibly broadcasting inputs to the result shape."""

    # Computes common shape
    common_shape = compute_broadcast_shape(*map(lambda t: t.shape if hasattr(t, "shape") else None, args))

    def _maybe_broadcast(x, shape):
        if hasattr(x, "shape"):
            if not utils.same_shape(x.shape, common_shape):
                return expand(x, common_shape)

        return x

    return tuple(_maybe_broadcast(x, common_shape) for x in args)


#
# Elementwise unary operations
#
# TODO Consider annotating these operators with kind and type promotion information


# TODO Add supported dtypes
def _elementwise_unary_wrapper(
    a,
    *,
    prim,
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, type_promotion_kind=type_promotion_kind)

    a = maybe_convert_to_dtype(a, computation_dtype)
    result = prim(a)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


# TODO Return self for bool and uint datatypes?
@clangop(method_name="abs")
def abs(a: TensorProxy | Number):
    # Short-circuits for unsigned types like bool and int8
    if dtypes.is_unsigned_dtype(dtypes.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a,
        prim=prims.abs,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    )


@clangop()
def acos(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.acos, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def acosh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.acosh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def asin(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.asin, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def asinh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.asinh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def atan(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.atan, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def atanh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.atanh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop(method_name="bitwise_not")
def bitwise_not(a: TensorLike | Number) -> TensorLike | Number:
    return _elementwise_unary_wrapper(
        a,
        prim=prims.bitwise_not,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE,
    )


@clangop(method_name="ceil")
def ceil(a: TensorLike | Number) -> TensorLike | Number:
    # Short-circuits on unsigned inputs
    if dtypes.is_exact_dtype(dtypes.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a,
        prim=prims.ceil,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


@clangop()
def cos(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.cos, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def cosh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.cosh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def digamma(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.digamma, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def erf(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.erf, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def erfc(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.erfc, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def erfcinv(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.erfcinv, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def erfinv(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.erfinv, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def exp(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.exp, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def exp2(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.exp2, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def expm1(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.expm1, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop(method_name="floor")
def floor(a: TensorLike | Number) -> TensorLike | Number:
    # Short-circuits on unsigned inputs
    if dtypes.is_exact_dtype(dtypes.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a,
        prim=prims.floor,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


@clangop()
def isfinite(a):
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return full_like(a, True, dtype=dtypes.bool8)

    return _elementwise_unary_wrapper(
        a,
        prim=prims.isfinite,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    )


@clangop()
def lgamma(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.lgamma, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def log(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.log, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def log10(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.log10, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def log1p(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.log1p, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def log2(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.log2, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def ndtri(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.ndtri, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


# TODO Should this have PRESERVE as its type promotion kind?
@clangop(method_name="neg")
def neg(a):
    return _elementwise_unary_wrapper(
        a,
        prim=prims.neg,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


@clangop()
def reciprocal(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.reciprocal, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop(method_name="round")
def round(a: TensorLike | Number) -> TensorLike | Number:
    # Short-circuits on inputs with exact dtypes
    if utils.is_exact_dtype(utils.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a,
        prim=prims.round,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


@clangop()
def rsqrt(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.rsqrt, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def sigmoid(a):
    # We manually promote a, then determine type of the constant 1.0 value
    computation_dtype, result_dtype = utils.elementwise_type_promotion(
        a, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )
    a = maybe_convert_to_dtype(a, computation_dtype)
    one = 1 + 0j if isinstance(computation_dtype, dtypes.complexfloating) else 1.0
    result = reciprocal(add(one, exp(-a)))
    result = maybe_convert_to_dtype(result, result_dtype)
    return result


# TODO Review type promotionkind for sign
def sign(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sign, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
    )


# TODO Add supported dtypes to exclude complex
def signbit(a):
    if dtypes.is_unsigned_dtype(dtypes.to_dtype(a)):
        return full_like(a, False, dtype=dtypes.bool8)

    return _elementwise_unary_wrapper(
        a, prim=prims.signbit, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clangop()
def silu(a):
    return a * sigmoid(a)


@clangop()
def sin(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sin, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def sinh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sinh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def sqrt(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sqrt, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def tan(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.tan, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


def tanh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.tanh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


def trunc(a: TensorLike | Number) -> TensorLike | Number:
    # Short-circuits on unsigned inputs (which are already trivially truncated)
    if dtypes.is_exact_dtype(dtypes.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a,
        prim=prims.trunc,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


@clangop(method_name="real")
def real(a: TensorProxy | Number):
    # Short-circuits for non-complex types
    if not dtypes.is_complex_dtype(dtypes.to_dtype(a)):
        return a

    return _elementwise_unary_wrapper(
        a,
        prim=prims.real,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    )


#
# Elementwise binary operations
#
# TODO Consider annotating these operators with kind and type promotion information


# TODO Add supported dtypes
def _elementwise_binary_wrapper(a, b, *, prim, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, b, type_promotion_kind=type_promotion_kind)

    a, b = maybe_broadcast(a, b)
    a, b = maybe_convert_to_dtype(a, computation_dtype), maybe_convert_to_dtype(b, computation_dtype)

    result = prim(a, b)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


@clangop(method_name="add")
def add(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.add)


@clangop()
def atan2(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.atan2, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop(method_name="bitwise_and")
def bitwise_and(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.bitwise_and, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@clangop(method_name="bitwise_or")
def bitwise_or(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.bitwise_or, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


@clangop(method_name="bitwise_xor")
def bitwise_xor(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.bitwise_xor, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


# NOTE Python's math.copysign has int to float type promotion, and PyTorch's copysign does, too
# NOTE copysign is not defined for complex types, and this must be explicitly checked for since the
#   following definition would be valid for complex numbers, too
@clangop()
def copysign(a, b):
    utils.check(
        not dtypes.is_complex_dtype(dtypes.to_dtype(a)) and not dtypes.is_complex_dtype(dtypes.to_dtype(b)),
        lambda: f"copysign is not defined for complex dtypes",
    )

    computation_dtype, result_dtype = utils.elementwise_type_promotion(
        a, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )

    compute_a = maybe_convert_to_dtype(a, computation_dtype)

    result = where(signbit(b), -abs(compute_a), abs(compute_a))
    return maybe_convert_to_dtype(result, result_dtype)


@clangop(method_name="eq")
def eq(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.eq, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


# NOTE Floor division in Python is defined to complement its modulus operator, s.t.

#   let a and b be numbers
#   q = a // b
#   r = a % b
#   b * q + r = a

#   This is NOT equivalent to floor(a/b). For example:

#   import math
#   a = .25
#   b = .001

#   # Compares flooring true division vs. floor division
#   math.floor(a / b)  # 250
#   a // b  # 249.

#   # Tests the invariant
#   q = a // b
#   r = a % b
#   b * q + r   # .25 == a

# See CPython's implementation here:
# https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636


# NOTE This is distinct from true_divide, which also wraps prims.div, because it doesn't promote
#   integers to floating point values
def _c_div(a: TensorProxy | Number, b: TensorProxy | Number) -> TensorProxy | Number:
    return _elementwise_binary_wrapper(
        a, b, prim=prims.div, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


def _floor_divide_integer(
    a: TensorProxy | Number, b: TensorProxy | Number, *, computation_dtype
) -> TensorProxy | Number:
    # Converts truncation division to floor division

    offset = logical_and(signbit(a) != signbit(b), fmod(a, b) != 0)
    return _c_div(a, b) - offset


def _floor_divide_float(a: TensorProxy | Number, b: TensorProxy | Number) -> TensorProxy | Number:
    mod = fmod(a, b)
    div = (a - mod) / b

    # Ensures that the remainder has the same sign as denominator
    different_signed_inputs = (a < 0) ^ (b < 0)
    non_zero_remainder = mod != 0
    mask = non_zero_remainder & different_signed_inputs
    div = where(mask, div - 1, div)

    # Maps quotient to nearest integer value
    floor_div = floor(div)
    mask = (div - floor_div) > 0.5
    floor_div = where(mask, floor_div + 1, floor_div)

    true_div = a / b

    # Copies signbit where floor division is zero
    floor_div = where(div != 0, floor_div, copysign(0, true_div))

    # Follows true divide behavior when the denominator is zero
    return where(b == 0, true_div, floor_div)


# Dispatches floor division to integer or floating point specializations
@clangop(method_name="floor_divide")
def floor_divide(a: TensorProxy | Number, b: TensorProxy | Number) -> TensorProxy | Number:
    computation_dtype, _ = utils.elementwise_type_promotion(
        a, b, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )

    utils.check(not dtypes.is_complex_dtype(computation_dtype), lambda: f"Complex floor division is not supported")

    if dtypes.is_float_dtype(computation_dtype):
        return _floor_divide_float(a, b)

    # NOTE At this point the datatypes are neither complex nor floating point, so they are exact types
    return _floor_divide_integer(a, b, computation_dtype=computation_dtype)


@clangop()
def fmod(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.fmod)


@clangop(method_name="mod")
def mod(a, b):
    return remainder(a, b)


@clangop(method_name="ge")
def ge(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.ge, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clangop(method_name="gt")
def gt(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.gt, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clangop()
def logical_and(a, b):
    if not utils.is_boolean_dtype(dtypes.to_dtype(a)):
        a = a != 0
    if not utils.is_boolean_dtype(dtypes.to_dtype(b)):
        b = b != 0

    return a & b


@clangop(method_name="le")
def le(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.le, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clangop(method_name="lt")
def lt(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.lt, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clangop()
def maximum(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.maximum)


@clangop()
def minimum(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.minimum)


@clangop(method_name="mul")
def mul(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.mul)


@clangop(method_name="ne")
def ne(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.ne, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    )


@clangop()
def nextafter(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.nextafter, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
    )


@clangop(method_name="pow")
def pow(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.pow, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    )


@clangop()
def remainder(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.remainder)


@clangop(method_name="sub")
def sub(a, b):
    return _elementwise_binary_wrapper(a, b, prim=prims.sub)


@clangop(method_name="true_divide")
def true_divide(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.div, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
def zeta(a, b):
    return _elementwise_binary_wrapper(
        a, b, prim=prims.zeta, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


#
# Conditional operators
#


@clangop()
def where(pred, a, b):
    # Performs type promotion
    promotiontype, _ = utils.elementwise_type_promotion(
        a, b, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
    )

    a, b = maybe_convert_to_dtype(a, promotiontype), maybe_convert_to_dtype(b, promotiontype)

    # Broadcasts
    pred, a, b = maybe_broadcast(pred, a, b)

    # Short circuits in the case that the predicate is a number
    if isinstance(pred, Number) and pytype(pred) is bool:
        if pyval(pred):
            return a
        return b

    return prims.where(pred, a, b)


# Helper function to support `keepdim` argument
def _argmaxmin_helper(prim, a: TensorProxy, dim: int | None, keepdim: bool | None):
    assert prim in (prims.argmax, prims.argmin)
    if dim is not None:
        dim = utils.canonicalize_dim(len(a.shape), dim)

    result = prim(a, dim)

    # For keepdim, unsqueeze only if a.ndim > 0
    if keepdim and a.ndim > 0:
        # restore all dims as `1` for keepdim=True and dim=None
        if dim is None:
            dim = range(a.ndim)
        result = unsqueeze(result, dim)

    return result


@clangop()
def argmax(a: TensorProxy, /, dim: int | None = None, keepdim: bool | None = False):
    return _argmaxmin_helper(prims.argmax, a, dim, keepdim)


@clangop()
def argmin(a: TensorProxy, /, dim: int | None = None, keepdim: bool | None = False):
    return _argmaxmin_helper(prims.argmin, a, dim, keepdim)


@clangop()
def topk(
    a: TensorLike, /, k: int, dim: int | None = None, largest: bool = True, sorted: bool = True, *, out=None
) -> (TensorProxy, TensorProxy):
    if dim is None:
        dim = a.ndim - 1 if a.ndim > 0 else 0
    dim = utils.canonicalize_dim(a.ndim, dim)

    return prims.topk(a, k, dim, bool(largest), bool(sorted), out=out)
