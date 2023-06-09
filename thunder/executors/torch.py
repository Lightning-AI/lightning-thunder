import operator
from functools import wraps
from numbers import Number
from typing import Union, Callable, Any, Tuple, Sequence, Optional

import torch

import thunder.core.dtypes as dtypes
from thunder.core.prims import PrimIDs
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.proxies import Proxy, TensorProxy
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.symbol import Symbol, BoundSymbol
import thunder.core.devices as devices
import thunder.core.utils as utils

from thunder.executors.utils import Region, Executor

import thunder.torch as ltorch
from thunder.torch import DeviceLike, dtypeLike


def name() -> Executor:
    return Executor.TORCH


torch_ctx = {
    "torch": torch,
}

# NOTE _ops_map is declared here and defined after the callables have been defined
#   below
_ops_map: dict[Any, Tuple[Callable, Callable]] = {}


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


def full(
    bsym: BoundSymbol, shape: Sequence[int], fill_value: Number, *, device: devices.Device, dtype: dtypes.dtype
) -> BoundSymbol:
    sym = Symbol(name="full", meta=None, _module=torch)

    device = str(device)
    dtype = ltorch.to_torch_dtype(dtype)

    tbsym = sym.bind(shape, fill_value, device=device, dtype=dtype, output=bsym.output)
    return tbsym


def _iota_helper(length, *, start, step, device, dtype):
    end = start + length * step
    torch_device = str(device)
    torch_dtype = ltorch.to_torch_dtype(dtype)
    return torch.arange(start=start, step=step, end=end, device=torch_device, dtype=torch_dtype)


def iota(bsym: BoundSymbol, length, *, start, step, device, dtype) -> BoundSymbol:
    sym = Symbol(name="iota_helper", meta=None)
    ctx: Dict[str, Any] = {"iota_helper": _iota_helper}

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


def contiguous(bsym: BoundSymbol, a) -> BoundSymbol:
    sym = Symbol(name="contiguous", meta=None, _module=torch.Tensor)
    tbsym = BoundSymbol(sym, args=(a,), kwargs={}, output=bsym.output)

    return tbsym


def getitem(bsym: BoundSymbol, tensor, key) -> BoundSymbol:
    sym = Symbol(name="__getitem__", meta=None, _module=torch.Tensor)
    tbsym = BoundSymbol(sym, args=(tensor, key), kwargs={}, output=bsym.output)

    return tbsym


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


def reshape(bsym: BoundSymbol, a, shape):
    sym = Symbol(name="reshape", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, shape), kwargs={}, output=bsym.output)

    return tbsym


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


# NOTE Order of index and dim changes
def take_along_axis(bsym: BoundSymbol, a, index, dim):
    sym = Symbol(name="take_along_dim", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, index, dim), kwargs={}, output=bsym.output)

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


def unsqueeze(bsym: BoundSymbol, a, dim: int):
    sym = Symbol(name="unsqueeze", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim), kwargs={}, output=bsym.output)

    return tbsym


def view(bsym: BoundSymbol, a, *shape):
    sym = Symbol(name="view", meta=None, _module=torch.Tensor)
    tbsym = BoundSymbol(sym, args=(a, shape), kwargs={}, output=bsym.output)

    return tbsym


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

#
# Elementwise binary operations
#
# TODO Review type promotion differences
# TODO Review restricting torch implemenations of prims to not have additional functionality


def _elementwise_binary_check(a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> bool:
    return not (isinstance(a, Number) and isinstance(b, Number))


def _elementwise_binary_factory(name: str) -> Callable:
    def fn(bsym: BoundSymbol, a: Union[TensorProxy, Number], b: Union[TensorProxy, Number]) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=torch)
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


add = _elementwise_binary_factory("add")
atan2 = _elementwise_binary_factory("atan2")
bitwise_and = _elementwise_binary_factory("bitwise_and")
bitwise_xor = _elementwise_binary_factory("bitwise_xor")
copysign = _elementwise_binary_factory("copysign")
div = _elementwise_binary_factory("div")
floor_divide = _elementwise_binary_factory("floor_divide")
eq = _elementwise_binary_factory("eq")
fmod = _elementwise_binary_factory("fmod")
ge = _elementwise_binary_factory("ge")
gt = _elementwise_binary_factory("gt")
logical_and = _elementwise_binary_factory("logical_and")
lt = _elementwise_binary_factory("lt")
mul = _elementwise_binary_factory("mul")
ne = _elementwise_binary_factory("ne")
nextafter = _elementwise_binary_factory("nextafter")
pow = _elementwise_binary_factory("pow")
# TODO Remainder bool isn't implement, so mark it as non-fusible
remainder = _elementwise_binary_factory("remainder")
sub = _elementwise_binary_factory("sub")

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


def masked_fill(bsym: BoundSymbol, a, mask, value):
    sym = Symbol(name="masked_fill", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, mask, value), kwargs={}, output=bsym.output)

    return tbsym


def tril(bsym: BoundSymbol, a, diagonal: int = 0):
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


# NOTE var does not allow keepdim to be specified if dim is not specified
# NOTE This is the direct translation of torch.var
def var(
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

    sym = Symbol(name="var", meta=None, _module=torch)

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


def layer_norm(bsym: BoundSymbol, a, normalized_shape, weight=None, bias=None, eps: Number = 1e-5):
    sym = Symbol(name="layer_norm", meta=None, _module=torch.nn.functional)
    tbsym = BoundSymbol(sym, args=(a, normalized_shape, weight, bias, eps), kwargs={}, output=bsym.output)

    return tbsym


def softmax(bsym: BoundSymbol, a, dim, dtype=None) -> BoundSymbol:
    torch_dtype = None
    if dtype is not None:
        torch_dtype = ltorch.to_torch_dtype(dtypes.numbertype_to_dtype(dtype))

    sym = Symbol(name="softmax", meta=None, _module=torch)

    kwargs = {
        "dtype": torch_dtype,
    }

    tbsym = BoundSymbol(sym, args=(a, dim), kwargs=kwargs, output=bsym.output)
    return tbsym


# TODO Refine prim ops to have less functionality to better debug errors
# Maps from symbol ids to a tuple of (is_fusable, translate) callables
_ops_map.update(
    {
        # Data movement operations
        PrimIDs.CONVERT_ELEMENT_TYPE: (_always_executable, convert_element_type),
        PrimIDs.DEVICE_PUT: (_always_executable, device_put),
        "torch.Tensor.to": (_always_executable, to),
        # Tensor creation operations
        "torch.arange": (_always_executable, arange),
        PrimIDs.FULL: (_always_executable, full),
        PrimIDs.IOTA: (_always_executable, iota),
        # Shape operations
        PrimIDs.BROADCAST_IN_DIM: (_always_executable, broadcast_in_dim),
        PrimIDs.CAT: (_always_executable, cat),
        "torch.Tensor.contiguous": (_always_executable, contiguous),
        "torch.Tensor.__getitem__": (_always_executable, getitem),
        PrimIDs.PAD: (_always_executable, pad),
        PrimIDs.RESHAPE: (_always_executable, reshape),
        PrimIDs.SLICE: (_always_executable, slice_prim),
        "torch.split": (_always_executable, split),
        "torch.squeeze": (_always_executable, squeeze),
        PrimIDs.SQUEEZE: (_always_executable, squeeze),
        PrimIDs.TAKE: (_always_executable, take),
        PrimIDs.TAKE_ALONG_AXIS: (_always_executable, take_along_axis),
        "torch.tensor_split": (_always_executable, tensor_split),
        "torch.transpose": (_always_executable, transpose),
        PrimIDs.TRANSPOSE: (_always_executable, prim_transpose),
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
        # Elementwise binary operations
        "torch.add": (_elementwise_binary_check, add),
        PrimIDs.ADD: (_elementwise_binary_check, add),
        "torch.atan2": (_elementwise_binary_check, atan2),
        PrimIDs.ATAN2: (_elementwise_binary_check, atan2),
        "torch.bitwise_and": (_elementwise_binary_check, bitwise_and),
        PrimIDs.BITWISE_AND: (_elementwise_binary_check, bitwise_and),
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
        "torch.lt": (_elementwise_binary_check, lt),
        PrimIDs.LT: (_elementwise_binary_check, lt),
        "torch.mul": (_elementwise_binary_check, mul),
        PrimIDs.MUL: (_elementwise_binary_check, mul),
        "torch.ne": (_elementwise_binary_check, ne),
        PrimIDs.NE: (_elementwise_binary_check, ne),
        "torch.nextafter": (_elementwise_binary_check, nextafter),
        PrimIDs.NEXTAFTER: (_elementwise_binary_check, nextafter),
        "torch.pow": (_elementwise_binary_check, pow),
        PrimIDs.POW: (_elementwise_binary_check, pow),
        "torch.remainder": (_elementwise_binary_check, remainder),
        PrimIDs.REMAINDER: (_elementwise_binary_check, remainder),
        "torch.sub": (_elementwise_binary_check, sub),
        PrimIDs.SUB: (_elementwise_binary_check, sub),
        # Conditional and masking operations
        "torch.masked_fill": (_always_executable, masked_fill),
        "torch.tril": (_always_executable, tril),
        PrimIDs.WHERE: (_elementwise_ternary_check, where),
        "torch.where": (_elementwise_ternary_check, where),
        # Reduction operators
        PrimIDs.AMAX: (_always_executable, amax_prim),
        PrimIDs.AMIN: (_always_executable, amin_prim),
        PrimIDs.PROD: (_always_executable, prod_prim),
        PrimIDs.SUM: (_always_executable, sum_prim),
        "torch.var": (_always_executable, var),
        PrimIDs.VAR: (_always_executable, var_prim),
        # Matmul operations
        PrimIDs.LINEAR: (_always_executable, linear),
        PrimIDs.MATMUL: (_always_executable, matmul),
        # NN operations
        "torch.nn.functional.dropout": (_always_executable, dropout),
        PrimIDs.EMBEDDING: (_always_executable, embedding),
        PrimIDs.EMBEDDING_BACKWARD: (_always_executable, embedding_backward),
        "torch.nn.functional.embedding": (_always_executable, embedding),
        "torch.layer_norm": (_always_executable, layer_norm),
        "torch.softmax": (_always_executable, softmax),
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
# OLD CODE BELOW HERE
#

# def convert_element_type(a, dtype):
#     # Handles converting a tensor to a numbertype, which Thunder allows but
#     #   Torch does not
#     if isinstance(a, torch.Tensor) and dtypes.is_numbertype(dtype):
#         dtype = ttorch.torch_dtype(dtypes.numbertype_to_dtype(dtype))

#     # Handles number conversions
#     if isinstance(a, Number):
#         if not dtypes.is_numbertype(dtype):
#             dtype = dtypes.dtype_to_numbertype(ttorch.thunder_dtype(dtype))
#         return dtype(a)

#     return a.to(dtype)


# def broadcast_in_dim(a, shape, broadcast_dims):
#     s = list(shape)
#     for broadcast_dim in broadcast_dims:
#         s[broadcast_dim] = -1

#     v = a
#     for idx, x in enumerate(s):
#         if x != -1:
#             v = v.unsqueeze(idx)

#     return v.expand(shape)


# # NOTE: PyTorch doesn't have a padding operation exactly like XLA's
# #   When dilations are all zero, torch.nn.functional.pad can substitute for XLA's
# #   Otherwise, this first dilates the original tensor by copying it into a slice of
# #   a larger tensor, then pads the dilated tensor
# def pad(a, padding_value, padding_config):
#     intermediate_shape = []
#     intermediate_slices = []
#     pad_config = []
#     just_pad = True
#     for l, (low, high, dilation) in zip(a.shape, padding_config):
#         assert dilation >= 0

#         if dilation > 0:
#             just_pad = False

#         intermediate_length = l + max(0, l - 1) * dilation
#         intermediate_shape.append(intermediate_length)
#         intermediate_slices.append(slice(None, None, dilation + 1))

#         pad_config.append((low, high))

#     pad_config = [x for y in reversed(pad_config) for x in y]

#     if just_pad:
#         return torch.nn.functional.pad(a, pad_config, value=padding_value)

#     result = torch.full(intermediate_shape, padding_value, device=a.device, dtype=a.dtype)
#     result[intermediate_slices] = a
#     result = torch.nn.functional.pad(result, pad_config, value=padding_value)
#     return result


# def erfcinv_helper(a):
#     erfinv = _elementwise_unary_torch(torch.erfinv)
#     return erfinv(1 - a)

# # A composite implementation of c++ std::remainder and Python math.remainder.
# def remainder_helper(a, b):
#     return a - torch.round(a.div(b)) * b


# def slice_helper(a, start_indices, end_indices, strides=None):
#     _strides = strides if strides is not None else [1] * len(start_indices)

#     slices = []
#     for start, stop, step in zip(start_indices, end_indices, _strides):
#         slices.append(slice(start, stop, step))

#     return operator.getitem(a, slices)


# # TODO: dim as a sequence is only supported on PyTorch 2.0 and greater
# def squeeze_helper(a, dim):
#     for d in sorted(dim, reverse=True):
#         a = a.squeeze(d)

#     return a


# def view_helper(a, shape):
#     return a.view(shape)


# def is_tensor(a):
#     return isinstance(a, torch.Tensor)


# def uniform_helper(shape, minval=0.0, maxval=1.0, *, device, dtype):
#     t = torch.empty(shape, device=device, dtype=dtype)
#     t.uniform_(minval, maxval)
#     return t


# # NOTE: many PyTorch operations don't accept numbers as inputs,
# #   so this helper wraps and unwraps numbers
# def _elementwise_unary_torch(op):
#     @wraps(op)
#     def _fn(x):
#         if isinstance(x, torch.Tensor):
#             return op(x)

#         return op(torch.tensor(x)).item()

#     return _fn


# def sum_helper(a, dims, output_dtype=None, **kwargs):
#     output_dtype_ = _get_torch(output_dtype)
#     # NOTE: PyTorch's sum reduces all dimensions if empty list is passed
#     #   but Thunder follows NumPy's behavior of returning the original
#     #   tensor if an empty list is passed.
#     if len(dims) == 0:
#         return a.to(output_dtype_)
#     return torch.sum(a, dim=dims, dtype=output_dtype_)


# # Handles adding two Python numbers, which PyTorch allows but returns
# #   as a tensor, while Thunder expects a Python number
# def add_helper(a, b, alpha=1):
#     if any(map(is_tensor, (a, b, alpha))):
#         return torch.add(a, b, alpha=alpha)

#     return a + b * alpha


# # NOTE: PyTorch's torch.eq expects tensor x tensor or tensor x number
# #   but the == operator allows number x tensor
# def eq_helper(a, b):
#     return a == b


class ThunderFunction(torch.autograd.Function):
    @staticmethod
    def augmented_forward_pass_wrapper(trace, *args):
        from thunder.core.transforms import augmented_forward_pass

        result, env = augmented_forward_pass(*args, trace=trace)
        saved_for_backward = {key: tuple(value) for key, value in env.items()}
        return result, saved_for_backward

    @staticmethod
    def backward_pass_wrapper(trace, saved_for_backward, cotangents):
        from thunder.core.transforms import backward_pass, VJPDual

        env = {key: VJPDual(*value) for key, value in saved_for_backward.items()}
        out = backward_pass(env, trace, cotangents)
        return out

    @staticmethod
    def forward(ctx, thunder_func, executors_list, *flat_args):
        from thunder import _make_trace as make_trace
        from thunder import compile

        trace = make_trace(thunder_func)(*flat_args)
        augmented_trace_fn = lambda *args: ThunderFunction.augmented_forward_pass_wrapper(trace, *args)
        augmented_trace_fn.__name__ = "augmented_trace_fn"  # compile doesn't like lambdas
        out, saved_info = compile(
            augmented_trace_fn,
            executors_list=executors_list,
            disable_preprocessing=True,
        )(*flat_args)
        ctx.executors_list = executors_list
        ctx.trace = trace

        # We must save tensors using ctx.save_for_backward
        flat_env, ctx.env_spec = tree_flatten(saved_info)
        is_tensor = tuple(isinstance(x, torch.Tensor) for x in flat_env)
        ctx.save_for_backward(*(x for x, is_tensor in zip(flat_env, is_tensor) if is_tensor))
        ctx.saved_non_tensors = tuple(x for x, is_tensor in zip(flat_env, is_tensor) if not is_tensor)
        ctx.is_tensor = is_tensor
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *args):
        from thunder import compile

        # Restore saved_info from ctx.saved_non_tensors and ctx.saved_tensors
        saved_tensors = iter(ctx.saved_tensors)
        saved_non_tensors = iter(ctx.saved_non_tensors)
        flat_saved_info = tuple(
            next(saved_tensors) if is_tensor else next(saved_non_tensors) for is_tensor in ctx.is_tensor
        )
        saved_info = tree_unflatten(flat_saved_info, ctx.env_spec)

        backward = lambda saved_info, *args: ThunderFunction.backward_pass_wrapper(ctx.trace, saved_info, args)
        backward.__name__ = "backward"  # compile doesn't like lambdas
        grads = compile(
            backward,
            executors_list=ctx.executors_list,
            disable_preprocessing=True,
        )(saved_info, *args)
        return (None, None, *grads)


def thunder_backward(executors_list=(Executor.NVFUSER,)):
    """Decorator to wrap a Thunder function for use with PyTorch autograd.

    Args:
        thunder_func: A Thunder function.

    Returns:
        A wrapped function that can be used with PyTorch autograd.

    Example:
    >>> import torch
    >>> import thunder.core.lang as tlang
    >>> from thunder.executors.torch import thunder_backward
    >>> @thunder_backward()
    ... def func(a, b):
    ...     c = a + b
    ...     d = c * b
    ...     e = tlang.sin(d) + tlang.cos(c)
    ...     return e
    >>> a = torch.randn(3, device="cuda", requires_grad=True)
    >>> b = torch.randn(3, device="cuda", requires_grad=True)
    >>> c = func(a, b)
    >>> print(c)
    >>> sum(c).sum().backward()
    >>> print(f"a.grad: {a.grad}")
    >>> print(f"b.grad: {b.grad}")
    """

    def flat_wrapper(flat_func, *flat_args):
        return ThunderFunction.apply(flat_func, executors_list, *flat_args)

    def decorator(thunder_func):
        @wraps(thunder_func)
        def wrapper(*args, **kwargs):
            flat_func, flat_args, _ = utils.flatten_func(thunder_func, args, kwargs)
            return flat_wrapper(flat_func, *flat_args)

        return wrapper

    return decorator
