from __future__ import annotations
from collections import namedtuple
from collections.abc import Callable, Sequence
from functools import partial, reduce
from numbers import Number
from types import EllipsisType, NoneType
from typing import Any, Union
import math
import operator
import warnings

from thunder.clang.langctx import register_method
from thunder.core import utils
from thunder.core.baseutils import run_once
from thunder.core.langctxs import langctx, Languages
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes
import thunder.core.prims as prims
from thunder.core.proxies import (
    AnyProxy,
    IntegerProxy,
    NumberLike,
    NumberProxy,
    Proxy,
    TensorProxy,
    pytype,
    pyval,
)

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


# Checks a tensor's shape and metadata (for use with cache check)
@clangop()
def check_tensor_shape_and_metadata(t: TensorProxy, /) -> None:
    return prims.check_tensor_shape_and_metadata(
        t,
        # replace Proxy entries with `-1`s as wild card, as we any value is
        # allowed for proxy entries
        tuple(t.shape),
        t.device.device_str(),
        dtypes.to_torch_dtype(t.dtype),
        t.requires_grad,
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
def check_number_type_and_value(n: NumberLike, value: Number, /) -> None:
    return prims.check_number_type_and_value(n, value)


@clangop()
def check_string_value(s: str, value: str, /) -> None:
    return prims.check_string_value(s, value)


@clangop()
def check_slice_value(s: slice, value: slice, /) -> None:
    return prims.check_slice_value(s, value)


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
    elif isinstance(a, (Number, NumberProxy)):
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
def arange(
    *, start: NumberLike, step: NumberLike, stop: NumberLike, device: DeviceLike, dtype: dtypes.dtype | None = None
):
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
        # TODO: maybe something like a isIntegerType?
        if all(tuple(isinstance(x, (int, IntegerProxy)) for x in (start, step, stop))):
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
    shape: Sequence[int], fill_value: NumberLike, *, device: DeviceLike, dtype: None | dtypes.dtype = None
) -> TensorLike:
    # Infers dtype from the fill_value when not explicitly provided
    if dtype is None:
        dtype = dtypes.numbertype_to_dtype(dtypes.to_dtype(fill_value))
    device = devices.to_device(device)

    return prims.full(tuple(shape), fill_value, device=device, dtype=dtype)


@clangop()
def full_like(
    a: TensorLike | Number,
    fill_value: NumberLike,
    *,
    device: DeviceLike | None = None,
    dtype: dtypes.dtype | None = None,
) -> TensorLike:
    if isinstance(a, (Number, NumberProxy)):
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
def empty(shape: Sequence[int], *, device: DeviceLike, dtype: dtypes.dtype) -> TensorLike:
    device = devices.to_device(device)

    return prims.empty(tuple(shape), device=device, dtype=dtype)


@clangop()
def uniform(
    shape: Sequence[int],
    minval: NumberLike = 0.0,
    maxval: NumberLike = 1.0,
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
    minval: NumberLike = 0.0,
    maxval: NumberLike = 1.0,
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
    minval: NumberLike = 0.0,
    maxval: NumberLike = 1.0,
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

    assert isinstance(key, list)

    # eliminate None by unsqeezing and replacing with 'slice(None)' aka ':'
    _key = []
    for idx, x in enumerate(key):
        if x is None:
            a = unsqueeze(a, idx)
            _key.append(slice(None))
        else:
            assert isinstance(x, (NumberProxy, Number, slice))
            _key.append(x)

    for _ in range(idx + 1, len(a.shape)):
        _key.append(slice(None))

    key = _key

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
        elif isinstance(x, (Number, NumberProxy)):
            # NOTE Numbers must be valid indices after canonicalization, unlike start and stop
            x = utils.canonicalize_dim(l, x)
            start_indices.append(x)
            end_indices.append(x + 1)
            strides.append(1)
            squeeze_dims.append(idx)
        else:
            # NOTE: this is redundant with the ValueError exception above
            raise ValueError(f"Found unexpected value {x} in key={key}")

    # performance optimization; check if we need slicing
    if (
        all([x == 0 for x in start_indices])
        and all([x == l for x, l in zip(end_indices, a.shape)])
        and all([x == 1 for x in strides])
        and len(squeeze_dims) == 0
    ):
        result = a
    else:
        result = prims.slice_prim(a, start_indices, end_indices, strides)

    if len(squeeze_dims) > 0:
        result = prims.squeeze(result, tuple(squeeze_dims))

    return result


# TODO Add more advanced indexing support
# TODO Review the modeling of advanced indexing in terms of flatten and take
# NOTE Advanced indexing with boolean tensors has data-dependent metadata (it is akin to indexing with nonzero)
@clangop()
def _advanced_indexing(a: TensorLike, /, key) -> TensorLike:
    """
    Implements the advanced part of indexing for Thunder tensors.

    key is expected to be a sequence of per-dimension indices.
    basic indexing other than keeping dimensions as is is expected to be done separately, see getitem.

    Supports:
      - n-dimensional integer tensor indices
      - Broadcasting of multiple index tensors to a common shape
      - Keeping dimensions (by indexing with 'slice(None)' aka  ':')
      - Sequences of integer tensor indices

    Not yet supported:
      - Boolean tensor indices (these require data-dependent metadata, like nonzero)

    Modeling:
      - All index tensors are broadcast to a common shape
      - Indices are flattened and used with `take` to gather elements
      - The result is reshaped to match the broadcast index shape and any remaining dimensions
      - Permutations are applied to match PyTorch/NumPy semantics
      - Negative indices are wrapped as in PyTorch/NumPy
    """
    assert isinstance(key, Sequence), (
        "advanced indexing needs a sequence of keys (that are either slice(None) or Sequence or Tensor)"
    )

    def _to_tensorproxies(x: Sequence, device: devices.DeviceType):
        if not isinstance(x, list):
            x = list(x)
        # Convert list of numbers to tensor if possible
        for idx, val in enumerate(x):
            if isinstance(val, (NumberProxy)):
                x[idx] = utils.get_numberlike_value(val)
        if all(isinstance(i, int) for i in x):
            x = tensor_from_sequence(x, dtype=dtypes.int64, device=device)
        return x

    basic_keys = []  # (key index, key)
    advanced_keys = []  # (key index, key)

    input_shape = prims.shape(a)

    # Canonicalize tensor indices, wrapping negative indices like -5
    def wrap_tensor(t: TensorLike, dim_length: int) -> TensorLike:
        return t + where(t < 0, dim_length, 0)

    # advanced indexing has the following positioning of the output dimension(s)
    # - if there is only one or only advanced indices in consecutive dimensions, the new dimension(s) are placed there,
    # - if there are advanced index dimensions "with gaps", the advanced index dimension(s) is placed in the front
    # this is computed as target_dim here
    # also, two normalizations are done here:
    # - sequences are converted to tensors,
    # - indices are wrapped (by adding the appropriate dimension's size to negative indices)
    last_i = None
    target_dim = None
    idx_dims = []
    idx_input_shapes = []
    non_idx_dims = []
    non_idx_shapes = []
    idx_numel = 1
    idxes = []
    for i, k in enumerate(key):
        if isinstance(k, (TensorLike, Sequence)):
            if last_i is not None and last_i + 1 != i:
                target_dim = 0
            elif target_dim is None:
                target_dim = i
            last_i = i
            idx_dims.append(i)
            if not isinstance(k, TensorLike):
                k = _to_tensorproxies(k, a.device)
            assert utils.is_exact_dtype(k.dtype)
            if k.dtype == dtypes.bool8:
                raise NotImplementedError(
                    "boolean advanced indexing is not implemented (the result would be of unknown shape)"
                )
            k = wrap_tensor(k, input_shape[i])
            idxes.append(k)
            idx_input_shapes.append(input_shape[i])
            idx_numel *= input_shape[i]
        else:
            assert k == slice(None), (
                "advanced part can only have skipped dims ('slice(None)' aka ':') and sequcnes / tensors"
            )
            non_idx_dims.append(i)
            non_idx_shapes.append(input_shape[i])

    # treat non-indexed dimensions as ":"
    for i in range(i + 1, len(input_shape)):
        non_idx_dims.append(i)
        non_idx_shapes.append(input_shape[i])

    assert target_dim is not None

    # for multi-dimensional indexing, we join the dimensions and compute a single index into them
    # this also takes care of the broadcasting between index tensors
    if len(idx_dims) > 1:
        a = transpose(a, [*non_idx_dims[:target_dim], *idx_dims, *non_idx_dims[target_dim:]])
        # TODO: the reshape is inefficient as it might create a copy, we might look at having a prim
        #       for this instead...
        a = reshape(a, [*non_idx_shapes[:target_dim], idx_numel, *non_idx_shapes[target_dim:]])

        # this also does the broadcasting
        flattened_idx = idxes[0]
        for d, i in enumerate(idxes[1:], start=1):
            flattened_idx = flattened_idx * idx_input_shapes[d] + i

        idx = flattened_idx
    else:
        (idx,) = idxes

    # handle multi-dimensional indices (and 0-d!) by making them one-dimensional first and then reshaping the output after the take
    index_shape = idx.shape
    if len(index_shape) != 1:
        idx = reshape(idx, (-1,))

    # this actually does the indexing
    output = take(a, idx, dim=target_dim)

    # for multi-dimensional indices, reshape the output
    if len(index_shape) != 1:
        output_shape = [*non_idx_shapes[:target_dim], *index_shape, *non_idx_shapes[target_dim:]]
        output = reshape(output, output_shape)

    return output


@clangop()
def copy_with_setitem(a: TensorLike, key, value: TensorLike) -> TensorLike:
    # TODO: do more checking here. We used to have a check
    #     lambda: f"{key=} tries to index more dimensions than {a.ndim=}",
    return prims.copy_with_setitem(a, key, value)


# NOTE: currently supported indexing:
# - all basic indexing.
# - advanced indexing with integer tensors
@clangop(method_name="getitem")
def getitem(a: TensorLike, /, key) -> TensorLike:
    """
    Implements indexing (mostly torch-style):
    - Basic indexing:
      - Integers index into a dimension and removes it
      - Slices (":" etc.)
      - None adds singleton dimensions
      - Ellipsis aka "..." (implicit ":" for as many dimensions to cover all)
    - Advanced indexing:
      - Lists
      - Tensors

    Mixing is allowed, the getitem splits this into a basic indexing operation
    and an advanced indexing operation (but can skip either).
    """

    # FIXME: This is a quick WAR to avoid accessing shape attribute of a without
    # definition. This needs to be done properly somewhere else. See issue
    # github.com/Lightning-AI/lightning-thunder/issues/1253
    old_shape = prims.shape(a)

    # unify key to list of dimension indices (outermost tuple is indexing along dimensions)
    if not isinstance(key, tuple):
        key = [key]
    else:
        key = [*key]

    # eliminate Ellipsis aka '...' by filling up dimensions with slice(None) aka ':'
    if any(isinstance(k, EllipsisType) for k in key):
        # expand the ellipsis type
        indexing_dims = sum(1 for k in key if not isinstance(k, (NoneType, EllipsisType)))
        ellipsis_length = len(old_shape) - indexing_dims
        new_key = []
        for k in key:
            if isinstance(k, EllipsisType):
                new_key.extend(slice(None) for _ in range(ellipsis_length))
                ellipsis_length = 0  # if there are multiple, we only use the first
            else:
                new_key.append(k)
        key = new_key

    # decompose into basic and advanced indexing
    # - None, slices and integer indices are handled by basic indexing
    # - tensors and sequences are handled by advanced indexing
    # For the things not handled, we insert slice(None) aka ':' at the right dimension
    input_dim = 0
    basic_indices = []
    advanced_indices = []
    have_advanced = False
    have_basic = False

    for k in key:
        if isinstance(k, (Number, NumberProxy)):  # removes an input dimension form basic output
            basic_indices.append(k)
            have_basic = True
            input_dim += 1
        elif isinstance(k, slice):  # this keeps a dimension form input
            if k != slice(None):
                have_basic = True
            basic_indices.append(k)
            advanced_indices.append(slice(None))
            input_dim += 1
        elif k is None:  # adds an basic output dim, does not take an input dim
            have_basic = True
            basic_indices.append(None)
            advanced_indices.append(slice(None))
        elif isinstance(k, (TensorLike, Sequence)):  # keeps a dimension and puts this into advance indexing
            input_dim += 1
            basic_indices.append(slice(None))
            advanced_indices.append(k)
            have_advanced = True
        else:
            raise IndexError(f"invalid index of type {type(k).__name__}")

    if input_dim > len(old_shape):
        raise IndexError(f"too many indices for tensor of dimension {len(old_shape)}")

    # now dispatch into the two parts, basic first, then advanced
    if have_basic:
        a = _basic_indexing(a, basic_indices)

    if have_advanced:
        a = _advanced_indexing(a, advanced_indices)

    return a


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
    utils.check(len(src_set) == len(src), lambda: "Found at least one source dimension specified multiple times")
    utils.check(len(set(dst)) == len(dst), lambda: "Found at least one destination dimension specified multiple times")

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
def index_copy(a: TensorProxy, indices: TensorProxy, value: TensorProxy, dim: int) -> TensorProxy:
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.index_copy(a, indices, value, dim)


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
def gather(a: TensorProxy, /, indices: TensorProxy, dim: int) -> TensorProxy:
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.gather(a, indices, dim)


@clangop()
def scatter(a: TensorProxy, /, index: TensorProxy, src: TensorProxy | Number, dim: int) -> TensorProxy:
    dim = utils.canonicalize_dim(a.ndim, dim)
    return prims.scatter(a, index, src, dim)


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
    broadcast_indices = maybe_broadcast(*indices, treat_cpu_scalar_tensors_as_numbers=False)

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
    if isinstance(dims, (Number, NumberProxy)):
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
def unfold(a: TensorProxy, /, dim: int, size: int, step: int) -> TensorProxy:
    return prims.unfold(a, dim, size, step)


@clangop()
def cat(tensors: list[TensorProxy], dim: int):
    """Concatenates the given sequence of tensors in the given dimension."""
    # Upcast tensors only if we have more than 1 tensor.
    # NumPy and PyTorch support upcasting with mixed dtypes.
    if len(tensors) > 1:
        _, output_dtype = utils.elementwise_type_promotion(
            *tensors, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
        )
        promoted_tensors = []
        for t in tensors:
            if t.dtype != output_dtype:
                t = prims.convert_element_type(t, output_dtype)
            promoted_tensors.append(t)
    else:
        promoted_tensors = tensors
    return prims.cat(promoted_tensors, dim)


@clangop()
def stack(tensors: list[TensorProxy], dim: int):
    """Concatenates the given sequence of tensors in a new (the given) dimension."""
    shapes = tuple(t.shape for t in tensors)
    utils.check(shapes, lambda: "list of tensors cannot be empty")
    for i, s in enumerate(shapes[1:], start=1):
        utils.check(
            s == shapes[0], lambda: f"tensors must be of the same shape, tensor at {i} is {s} instead of {shapes[0]}"
        )
    tensors_ = [unsqueeze(t, dim) for t in tensors]
    return cat(tensors_, dim)


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
        raise RuntimeError("tensor.mT is only supported on matrices or batches of matrices. Got 1-D tensor.")

    dim0, dim1 = -2, -1
    dim0, dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))
    permutation = list(range(a.ndim))
    permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
    return transpose(a, permutation)


# TODO: add scalar support
# TODO: review hasattr pattern
# NOTE: the tensor is not broadcasted if it is a CPU scalar tensor and treat_cpu_scalar_tensors_as_numbers=True
@clangop()
def maybe_broadcast(*args, treat_cpu_scalar_tensors_as_numbers=True):
    """Returns tensors with the same shape, possibly broadcasting inputs to the result shape."""

    # Computes common shape
    common_shape = compute_broadcast_shape(*map(lambda t: t.shape if hasattr(t, "shape") else None, args))

    def _maybe_broadcast(x, shape):
        if treat_cpu_scalar_tensors_as_numbers and utils.is_cpu_scalar_tensor(x):
            return x
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
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.NUMBER_TO_INT,
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
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.NUMBER_TO_INT,
    )


@clangop()
def frexp(a: TensorLike, /) -> tuple[TensorLike, TensorLike]:
    return prims.frexp(a)


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
def isnan(a: TensorLike) -> TensorLike:
    return prims.ne(a, a)


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
@clangop()
def sign(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.sign, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE
    )


# TODO Add supported dtypes to exclude complex
@clangop()
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


@clangop()
def tanh(a):
    return _elementwise_unary_wrapper(
        a, prim=prims.tanh, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )


@clangop()
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


@clangop(method_name="imag")
def imag(a: TensorProxy | Number, /) -> TensorLike:
    utils.check(
        dtypes.is_complex_dtype(dtypes.to_dtype(a)),
        lambda: "imag is not implemented for tensors with non-complex dtypes",
    )

    return _elementwise_unary_wrapper(
        a,
        prim=prims.imag,
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
        lambda: "copysign is not defined for complex dtypes",
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

    utils.check(not dtypes.is_complex_dtype(computation_dtype), lambda: "Complex floor division is not supported")

    if dtypes.is_float_dtype(computation_dtype):
        return _floor_divide_float(a, b)

    # NOTE At this point the datatypes are neither complex nor floating point, so they are exact types
    return _floor_divide_integer(a, b, computation_dtype=computation_dtype)


@clangop(method_name="trunc_divide")
def trunc_divide(a: TensorProxy | Number, b: TensorProxy | Number, /) -> TensorProxy | Number:
    return trunc(_c_div(a, b))


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


@clangop()
def logical_not(a: TensorLike, /) -> TensorLike:
    if not utils.is_boolean_dtype(dtypes.to_dtype(a)):
        return a == 0
    return ~a


@clangop()
def logical_or(a: TensorLike, b: TensorLike) -> TensorLike:
    if not utils.is_boolean_dtype(dtypes.to_dtype(a)):
        a = a != 0
    if not utils.is_boolean_dtype(dtypes.to_dtype(b)):
        b = b != 0
    return bitwise_or(a, b)


@clangop()
def logical_xor(a: TensorLike, b: TensorLike, /) -> TensorLike:
    if not utils.is_boolean_dtype(dtypes.to_dtype(a)):
        a = a != 0
    if not utils.is_boolean_dtype(dtypes.to_dtype(b)):
        b = b != 0
    return a ^ b


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


@clangop()
def bitwise_left_shift(a, b):
    return _elementwise_binary_wrapper(
        a,
        b,
        prim=prims.bitwise_left_shift,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE,
    )


@clangop()
def bitwise_right_shift(a, b):
    return _elementwise_binary_wrapper(
        a,
        b,
        prim=prims.bitwise_right_shift,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.PRESERVE,
    )


#
# Elementwise ternary operations
#


@clangop()
def lerp(start: TensorLike, end: TensorLike, weight: Number | TensorLike) -> TensorLike:
    inputs = (start, end, weight)
    # torch.lerp does not promote types and only accepts floating-point inputs
    computation_dtype, result_dtype = utils.elementwise_type_promotion(
        *inputs, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )

    inputs = maybe_broadcast(*inputs)
    inputs = map(partial(maybe_convert_to_dtype, dtype=computation_dtype), inputs)

    result = prims.lerp(*inputs)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


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
    a: TensorLike, /, k: int, dim: int | None = None, largest: bool = True, sorted: bool = True
) -> tuple[TensorProxy, TensorProxy]:
    if dim is None:
        dim = a.ndim - 1 if a.ndim > 0 else 0
    dim = utils.canonicalize_dim(a.ndim, dim)

    return prims.topk(a, k, dim, bool(largest), bool(sorted))


@clangop()
def sort(
    a: TensorLike, /, dim: None | int = None, descending: bool = False, stable: bool = False
) -> (TensorProxy, TensorProxy):
    if dim is None:
        dim = a.ndim - 1 if a.ndim > 0 else 0
    dim = utils.canonicalize_dim(a.ndim, dim)

    return prims.sort(a, dim, descending, stable)
