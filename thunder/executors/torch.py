import operator
from functools import wraps
from numbers import Number
from typing import Union, Callable, Any, Tuple

import torch

import thunder.core.dtypes as dtypes
from thunder.core.prims import PrimIDs
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.proxies import Proxy, TensorProxy
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.symbol import Symbol, BoundSymbol
import thunder.core.devices as devices

from thunder.executors.utils import *

import thunder.torch as ltorch

torch_ctx = {
    "torch": torch,
}

# NOTE _ops_map is declared here and defined after the callables have been defined
#   below
_ops_map: Dict[Any, Tuple[Callable, Callable]] = {}


# Helper to signal that an operation is always executable
def _always_executable(*args, **kwargs) -> bool:
    return True


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


#
# Tensor creation operations
#


def full(
    bsym: BoundSymbol, shape: Sequence[int], fill_value: Number, *, device: devices.Device, dtype: dtypes.dtype
) -> BoundSymbol:
    sym = Symbol(name="full", meta=None, _module=torch)

    torch_device = str(device)
    torch_dtype = ltorch.to_torch_dtype(dtype)

    kwargs = {"device": torch_device, "dtype": torch_dtype}

    tbsym = BoundSymbol(sym, args=(shape, fill_value), kwargs=kwargs, output=bsym.output)
    return tbsym


def iota_helper(length, *, start, step, device, dtype):
    end = start + length * step
    return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype)


def iota(bsym: BoundSymbol, length, *, start, step, device, dtype) -> BoundSymbol:
    sym = Symbol(name="arange", meta=None, _module=torch)

    end = start + length * step

    torch_device = str(device)
    torch_dtype = ltorch.to_torch_dtype(dtype)

    kwargs = {
        "start": start,
        "step": step,
        "end": end,
        "device": torch_device,
        "dtype": torch_dtype,
    }

    tbsym = BoundSymbol(sym, args=(), kwargs=kwargs, output=bsym.output)

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


def cat(bsym: BoundSymbol, tensors, dim=0):
    sym = Symbol(name="cat", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(tensors, dim), kwargs={}, output=bsym.output)

    return tbsym


def getitem(bsym: BoundSymbol, tensor, key):
    sym = Symbol(name="__getitem__", meta=None, _module=torch.Tensor)
    tbsym = BoundSymbol(sym, args=(tensor, key), kwargs={}, output=bsym.output)

    return tbsym


def split(bsym: BoundSymbol, a, size_or_sections, dim=0):
    sym = Symbol(name="split", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, size_or_sections, dim), kwargs={}, output=bsym.output)

    return tbsym


def tensor_split(bsym: BoundSymbol, a, size_or_sections, dim=0):
    sym = Symbol(name="tensor_split", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, size_or_sections, dim), kwargs={}, output=bsym.output)

    return tbsym


def transpose(bsym: BoundSymbol, a, dim0, dim1):
    sym = Symbol(name="transpose", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim0, dim1), kwargs={}, output=bsym.output)

    return tbsym


# NOTE: PyTorch doesn't have a padding operation exactly like XLA's
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


def prim_transpose(bsym: BoundSymbol, a, permutation):
    sym = Symbol(name="permute", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, permutation), kwargs={}, output=bsym.output)

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


def view(bsym: BoundSymbol, a, shape):
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
remainder = _elementwise_binary_factory("remainder")
sub = _elementwise_binary_factory("sub")

#
# Elementwise ternary operations
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

#
# Reduction operations
#
# TODO Capture torch reductions for amax, amin, prod, and sum


def _prim_reduction_factory(name: str) -> Callable:
    def fn(bsym: BoundSymbol, a: TensorProxy, dims, *, output_dtype: Optional[dtypes.dtype] = None) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=torch)

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
prod_prim = _prim_reduction_factory("prod")
sum_prim = _prim_reduction_factory("sum")


# TODO Add type annotations
# TODO Review if this needs more of a wrapper around torch.var to implement the prim properly
# TODO Implement output dtype properly
def var_prim(bsym: BoundSymbol, a: TensorProxy, dims, *, correction: int) -> BoundSymbol:
    sym = Symbol(name="var", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dims), kwargs={"correction": correction}, output=bsym.output)

    return tbsym


# TODO Add type annotations
# TODO Check for unbiased and correction both being specified and error
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
    sym = Symbol(name="var", meta=None, _module=torch)

    args = tuple(x for x in (a, dim) if x is not None)
    kwargs = {
        "keepdim": keepdim,
        "correction": correction,
    }

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
    *,
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


# TODO Refine prim ops to have less functionality to better debug errors
# Maps from symbol ids to a tuple of (is_fusable, translate) callables
_ops_map.update(
    {
        # Data movement operations
        PrimIDs.CONVERT_ELEMENT_TYPE: (_always_executable, convert_element_type),
        PrimIDs.DEVICE_PUT: (_always_executable, device_put),
        # Tensor creation operations
        PrimIDs.FULL: (_always_executable, full),
        PrimIDs.IOTA: (_always_executable, iota),
        # Shape operations
        PrimIDs.BROADCAST_IN_DIM: (_always_executable, broadcast_in_dim),
        PrimIDs.CAT: (_always_executable, cat),
        PrimIDs.PAD: (_always_executable, pad),
        PrimIDs.RESHAPE: (_always_executable, reshape),
        PrimIDs.SLICE: (_always_executable, slice_prim),
        "torch.squeeze": (_always_executable, squeeze),
        PrimIDs.SQUEEZE: (_always_executable, squeeze),
        "torch.transpose": (_always_executable, transpose),
        PrimIDs.TRANSPOSE: (_always_executable, prim_transpose),
        PrimIDs.TAKE: (_always_executable, take),
        PrimIDs.TAKE_ALONG_AXIS: (_always_executable, take_along_axis),
        PrimIDs.VIEW: (_always_executable, view),
        "torch.Tensor.__getitem__": (_always_executable, getitem),
        "torch.split": (_always_executable, split),
        "torch.tensor_split": (_always_executable, tensor_split),
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
        # Elementwise ternary operations
        PrimIDs.WHERE: (_elementwise_ternary_check, where),
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
def fuse(
    trace: TraceCtx, producers, consumers, bound_symbols: Sequence[BoundSymbol], counter: int
) -> List[BoundSymbol]:
    bsyms: List[BoundSymbol] = []

    for bsym in bound_symbols:
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


# # Maps the Thunder primitives to their corresponding torch operation names
# # TODO: handle more scalar arguments (like add does above)
# ops_to_torch_ops_map = {
#     # Data movement and transformation prims
#     prims.Ops.CONVERT_ELEMENT_TYPE: convert_element_type,
#     # Tensor creation prims
#     prims.Ops.FULL: "torch.full",
#     prims.Ops.IOTA: iota_helper,
#     prims.Ops.UNIFORM: uniform_helper,
#     # Shape prims
#     prims.Ops.BROADCAST_IN_DIM: broadcast_in_dim,
#     prims.Ops.PAD: pad,
#     prims.Ops.RESHAPE: "torch.reshape",
#     prims.Ops.SLICE: slice_helper,
#     prims.Ops.SQUEEZE: squeeze_helper,
#     # NOTE: PyTorch's transpose is not equivalent to the transpose prim
#     prims.Ops.TRANSPOSE: "torch.permute",
#     prims.Ops.INDEX_SELECT: "torch.index_select",
#     prims.Ops.VIEW: view_helper,
#     # Elementwise unary prims
#     prims.Ops.ABS: _elementwise_unary_torch(torch.abs),
#     prims.Ops.ACOS: _elementwise_unary_torch(torch.acos),
#     prims.Ops.ACOSH: _elementwise_unary_torch(torch.acosh),
#     prims.Ops.ASIN: _elementwise_unary_torch(torch.asin),
#     prims.Ops.ASINH: _elementwise_unary_torch(torch.asinh),
#     prims.Ops.ATAN: _elementwise_unary_torch(torch.atan),
#     prims.Ops.ATANH: _elementwise_unary_torch(torch.atanh),
#     prims.Ops.BITWISE_NOT: _elementwise_unary_torch(torch.bitwise_not),
#     prims.Ops.CEIL: _elementwise_unary_torch(torch.ceil),
#     prims.Ops.COS: _elementwise_unary_torch(torch.cos),
#     prims.Ops.COSH: _elementwise_unary_torch(torch.cosh),
#     prims.Ops.ERF: _elementwise_unary_torch(torch.erf),
#     prims.Ops.ERFC: _elementwise_unary_torch(torch.erfc),
#     prims.Ops.ERFCINV: erfcinv_helper,
#     prims.Ops.ERFINV: _elementwise_unary_torch(torch.erfinv),
#     prims.Ops.EXP: _elementwise_unary_torch(torch.exp),
#     prims.Ops.EXP2: _elementwise_unary_torch(torch.exp2),
#     prims.Ops.EXPM1: _elementwise_unary_torch(torch.expm1),
#     prims.Ops.FLOOR: _elementwise_unary_torch(torch.floor),
#     prims.Ops.ISFINITE: _elementwise_unary_torch(torch.isfinite),
#     prims.Ops.RSQRT: _elementwise_unary_torch(torch.rsqrt),
#     prims.Ops.SIGN: _elementwise_unary_torch(torch.sgn),
#     prims.Ops.SIN: _elementwise_unary_torch(torch.sin),
#     prims.Ops.SINH: _elementwise_unary_torch(torch.sinh),
#     prims.Ops.SQRT: _elementwise_unary_torch(torch.sqrt),
#     prims.Ops.TAN: _elementwise_unary_torch(torch.tan),
#     prims.Ops.TANH: _elementwise_unary_torch(torch.tanh),
#     prims.Ops.LGAMMA: _elementwise_unary_torch(torch.lgamma),
#     prims.Ops.LOG: _elementwise_unary_torch(torch.log),
#     prims.Ops.LOG10: _elementwise_unary_torch(torch.log10),
#     prims.Ops.LOG1P: _elementwise_unary_torch(torch.log1p),
#     prims.Ops.LOG2: _elementwise_unary_torch(torch.log2),
#     prims.Ops.NEG: _elementwise_unary_torch(torch.neg),
#     prims.Ops.NDTRI: _elementwise_unary_torch(torch.special.ndtri),
#     prims.Ops.RECIPROCAL: _elementwise_unary_torch(torch.reciprocal),
#     prims.Ops.ROUND: _elementwise_unary_torch(torch.round),
#     prims.Ops.TRUNC: _elementwise_unary_torch(torch.trunc),
#     # Elementwise binary prims
#     prims.Ops.ADD: add_helper,
#     prims.Ops.ATAN2: "torch.atan2",
#     prims.Ops.BITWISE_AND: "torch.bitwise_and",
#     prims.Ops.DIV: "torch.div",
#     prims.Ops.FMOD: "torch.fmod",
#     prims.Ops.EQ: eq_helper,
#     prims.Ops.GE: "torch.ge",
#     prims.Ops.LT: "torch.lt",
#     prims.Ops.MUL: "torch.mul",
#     prims.Ops.NEXTAFTER: "torch.nextafter",
#     prims.Ops.POW: "torch.pow",
#     prims.Ops.REMAINDER: remainder_helper,
#     prims.Ops.SUB: "torch.sub",
#     # Elementwise ternary prims
#     prims.Ops.WHERE: "torch.where",
#     # Reduction prims
#     prims.Ops.AMAX: "torch.amax",
#     prims.Ops.AMIN: "torch.amin",
#     prims.Ops.PROD: torch._refs.prod,
#     prims.Ops.SUM: sum_helper,
#     prims.Ops.VAR: "torch.var",
#     # NOTE: VAR_MEAN is here to execute nvFuser traces with PyTorch
#     nvOps.VAR_MEAN: "torch.var_mean",
#     # Matmul prims
#     prims.Ops.LINEAR: "torch.nn.functional.linear",
#     prims.Ops.MATMUL: "torch.matmul",
#     # NN prims
#     prims.Ops.EMBEDDING: "torch.nn.functional.embedding",
#     prims.Ops.EMBEDDING_BACKWARD: "torch.ops.aten.embedding_backward",
# }


# # NOTE: this class is here to help with proper printing
# class ProxyName:
#     def __init__(self, name):
#         self.name = name

#     def __repr__(self):
#         return self.name


# def _get_torch(x):
#     if dtypes.is_dtype(x) and not dtypes.is_numbertype(x):
#         return _thunder_to_torch_dtype_map[x]

#     if isinstance(x, str):
#         return f"'{x}'"

#     if isinstance(x, type):
#         return x.__name__

#     if isinstance(x, Variable):
#         return ProxyName(x.name)

#     return x


# def _get_torch_op(op):
#     if op in ttorch._torch_to_thunder_function_map:
#         return op
#     return ops_to_torch_ops_map[op]


# # Acquires a proxies name or passes a constant by value
# # TODO: put this into executors utils
# def _extract_name(x):
#     if isinstance(x, (Variable, Proxy)):
#         return x.name

#     return str(x)


# # TODO: refactor _fuse_region to be called by a common executor utility to generate fusions
# def _fuse_region(inputs, outputs, symbols, *, _return_code=False, _contiguous=True):
#     # Defines utilities
#     tab = "  "

#     # Initializes context
#     ctx = {
#         "torch": torch,
#         "inf": float("inf"),  # NOTE: not always necessary
#     }

#     # Creates signature
#     # NOTE: PyTorch fusions are run in a no grad context
#     # arg_str = ", ".join(tuple(_extract_name(inp) for inp in inputs))
#     # cstr = f"@torch.no_grad()\ndef fusion({arg_str}):"

#     # Calls PyTorch and Python operations
#     # cstr += f"\n{tab}# Executes the trace"
#     cstr = ""
#     for sym in symbols:
#         torch_args = tree_map(_get_torch, sym.args)
#         torch_kwargs = tree_map(_get_torch, sym.kwargs)
#         torch_op = _get_torch_op(sym.op)
#         result = sym.outputs[0]

#         # TODO: relax requirement that prim outputs are proxies?
#         for out in sym.outputs:
#             if not isinstance(out, Variable):
#                 raise NotImplementedError

#         op_str = None
#         if isinstance(torch_op, str):
#             op_str = torch_op
#         else:
#             op_str = torch_op.__name__
#             ctx[op_str] = torch_op

#         # # NOTE: currently assumes that the trace is stored in the "trace" kwarg
#         # if "trace" in torch_kwargs and any(isinstance(v, Trace) for v in torch_kwargs.values()):
#         #     key = result.name + "_" + op_str + "_trace"
#         #     ctx[key] = torch_kwargs["trace"]
#         #     torch_kwargs["trace"] = key

#         result_str = ", ".join(out.name for out in sym.outputs)
#         arg_str = ", ".join(f"{a}" for a in torch_args)
#         kwarg_str = ", ".join(f"{k}={v}" for k, v in torch_kwargs.items())
#         segue_str = ", " if (len(arg_str) > 0 and len(kwarg_str) > 0) else ""

#         cstr += f"\n{tab}{result_str} = {op_str}({arg_str}{segue_str}{kwarg_str})"

#     # Constructs outputs
#     output_names = []
#     output_strs = []
#     for out in outputs:
#         if isinstance(out, Variable) and isinstance(out.proxy, TensorProxy):
#             out = out.proxy
#             output_names.append(_extract_name(out))
#             if _contiguous:
#                 # TODO: FIXME: currently makes all outputs contiguous to simplify stride analysis
#                 output_strs.append(f"{_extract_name(out)}.contiguous()")
#             else:
#                 output_strs.append(f"{_extract_name(out)}")
#         else:
#             output_strs.append(_extract_name(out))
#     out_names_str = ", ".join(output_names)
#     out_str = ", ".join(output_strs)
#     cstr += f"\n{tab}{out_names_str} = {out_str}"
#     # cstr += f"\n{tab}return {out_str}"

#     # code = compile(cstr, "torch.gen", mode="exec")
#     # exec(code, ctx)
#     # fusion = ctx["fusion"]

#     # if _return_code:
#     #     return fusion, cstr

#     # return fusion

#     return cstr, ctx


# # TODO: intercept PyTorch operations and handle functions whose results are
# #   bound to multiple values (i.e. a, b = foo(x, y))
# # Creates a Python callable that executes the trace using PyTorch and Python
# # NOTE: does this by compiling a function from a string
# def _fuse(trace):
#     flat_outputs, output_structure = tree_flatten(trace.outputs)

#     # Short-circuits if the fusion has no outputs
#     if len(flat_outputs) == 0:

#         def _fusion(*args, **kwargs):
#             return None

#         return _fusion

#     #
#     # Constructs the program
#     #

#     # Writes the signatures
#     # NOTE: PyTorch fusions are run in a no grad context
#     tab = "  "
#     cstr = f"@torch.no_grad()\ndef fusion(*args, **kwargs):"
#     # TODO: maybe consider the possibility of name conflicts?
#     ctx = {
#         "torch": torch,
#         "tree_flatten": tree_flatten,
#         "tree_unflatten": tree_unflatten,
#         "output_structure": output_structure,
#         "inf": float("inf"),  # NOTE: not always necessary
#     }

#     # Acquires inputs
#     flat_positional_inputs, _ = tree_flatten(trace.args)
#     flat_kwarg_inputs, _ = tree_flatten(trace.kwargs)

#     cstr += f"\n{tab}# Extracts inputs"
#     cstr += f"\n{tab}flat_args, _ = tree_flatten(args)"
#     cstr += f"\n{tab}flat_kwargs, _ = tree_flatten(kwargs)"

#     for idx, pinp in enumerate(flat_positional_inputs):
#         if isinstance(pinp, Variable):
#             cstr += f"\n{tab}{pinp.name} = flat_args[{idx}]"
#     for idx, kwinp in enumerate(flat_kwarg_inputs):
#         if isinstance(kwinp, Variable):
#             cstr += f"\n{tab}{kwinp.name} = flat_kwargs[{idx}]"

#     # Calls PyTorch and Python operations
#     cstr += f"\n{tab}# Executes the trace"
#     for sym in trace.symbols:
#         torch_args = tree_map(_get_torch, sym.args)
#         torch_kwargs = tree_map(_get_torch, sym.kwargs)
#         torch_op = _get_torch_op(sym.op)
#         result = sym.outputs[0]

#         if not isinstance(result, Variable):
#             raise NotImplementedError

#         # NOTE: currently assumes result is always a proxy
#         op_str = None
#         if isinstance(torch_op, str):
#             op_str = torch_op
#         else:
#             op_str = torch_op.__name__
#             ctx[op_str] = torch_op

#         # NOTE: currently assumes that the trace is stored in the "trace" kwarg
#         if "trace" in torch_kwargs and any(isinstance(v, Trace) for v in torch_kwargs.values()):
#             key = result.name + "_" + op_str + "_trace"
#             ctx[key] = torch_kwargs["trace"]
#             torch_kwargs["trace"] = key

#         result_str = ", ".join(out.name for out in sym.outputs)
#         arg_str = ", ".join(f"{a}" for a in torch_args)
#         kwarg_str = ", ".join(f"{k}={v}" for k, v in torch_kwargs.items())
#         segue_str = ", " if (len(arg_str) > 0 and len(kwarg_str) > 0) else ""

#         cstr += f"\n{tab}{result_str} = {op_str}({arg_str}{segue_str}{kwarg_str})"

#     # Constructs output
#     # NOTE: len(flat_outputs) > 0
#     torch_outputs = tree_map(_get_torch, flat_outputs)
#     output_str = ", ".join(_extract_name(x) for x in torch_outputs)
#     cstr += f"\n{tab}return tree_unflatten(({output_str},), output_structure)"

#     # Compiles the function
#     code = compile(cstr, "torch.gen", mode="exec")
#     exec(code, ctx)
#     fusion = ctx["fusion"]

#     return fusion


# class torchCtx:
#     def __init__(self):
#         pass

#     def intercept(self, op):
#         return None

#     # TODO: maybe return some actual profiling information
#     # TODO: don't ignore additional kwargs, like mode
#     def fuse(self, trace, *, profile_info=False, **kwargs):
#         if profile_info:
#             return _fuse(trace), None
#         return _fuse(trace)


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
    def forward(ctx, thunder_func, executor, *flat_args):
        from thunder import make_trace, make_traced

        trace = make_trace(thunder_func, executor=executor)(*flat_args)
        augmented_trace_fn = lambda *args: ThunderFunction.augmented_forward_pass_wrapper(trace, *args)
        out, saved_info = make_traced(augmented_trace_fn, executor=executor)(*flat_args)
        ctx.executor = executor
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
        from thunder import make_traced

        # Restore saved_info from ctx.saved_non_tensors and ctx.saved_tensors
        saved_tensors = iter(ctx.saved_tensors)
        saved_non_tensors = iter(ctx.saved_non_tensors)
        flat_saved_info = tuple(
            next(saved_tensors) if is_tensor else next(saved_non_tensors) for is_tensor in ctx.is_tensor
        )
        saved_info = tree_unflatten(flat_saved_info, ctx.env_spec)

        backward = lambda saved_info, *args: ThunderFunction.backward_pass_wrapper(ctx.trace, saved_info, args)
        grads = make_traced(backward, executor=ctx.executor)(saved_info, *args)
        return (None, None, *grads)


def thunder_backward(executor="nvfuser"):
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
        return ThunderFunction.apply(flat_func, executor, *flat_args)

    def decorator(thunder_func):
        @wraps(thunder_func)
        def wrapper(*args, **kwargs):
            flat_func, flat_args, _ = flatten_func(thunder_func, args, kwargs)
            return flat_wrapper(flat_func, *flat_args)

        return wrapper

    return decorator
