import operator
from functools import wraps, partial
from numbers import Number
from typing import Union, Callable, Any, Tuple, Sequence, Optional

import torch
from looseversion import LooseVersion

import thunder.core.dtypes as dtypes
from thunder.core.prims import PrimIDs, DistributedReduceOps
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.proxies import Proxy, TensorProxy, FutureTensorProxy
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.symbol import Symbol, BoundSymbol
import thunder.core.devices as devices
import thunder.core.utils as utils

from thunder.executors.utils import Region, Executor

import thunder.torch as ltorch
from thunder.torch import DeviceLike, dtypeLike, TensorLike


# NOTE This is part of the executor interface
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


def _exogenous_like_helper(likes: Sequence[torch.Tensor], /) -> tuple[torch.Tensor, ...]:
    return tuple([torch.zeros_like(x) for x in likes])


def exogenous_like(bsym: BoundSymbol, likes: Sequence[TensorProxy], /) -> BoundSymbol:
    sym = Symbol(name="exogenous_like", meta=None)
    ctx: dict[str, Any] = {"exogenous_like": _exogenous_like_helper}

    return sym.bind(likes, output=bsym.output, _call_ctx=ctx)


def full(
    bsym: BoundSymbol, shape: Sequence[int], fill_value: Number, *, device: devices.Device, dtype: dtypes.dtype
) -> BoundSymbol:
    sym = Symbol(name="full", meta=None, _module=torch)

    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)

    tbsym = sym.bind(shape, fill_value, device=device, dtype=dtype, output=bsym.output)
    return tbsym


def full_like(
    bsym: BoundSymbol,
    a: TensorLike,
    fill_value: Number,
    *,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
) -> BoundSymbol:
    sym = Symbol(name="full_like", meta=None, _module=torch)

    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)

    return sym.bind(a, fill_value, device=device, dtype=dtype, output=bsym.output)


# TODO Review whether this helper is required (and if it is, document why better)
def _iota_helper(length, *, start, step, device, dtype) -> torch.Tensor:
    end = start + length * step
    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)
    return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype)


def iota(bsym: BoundSymbol, length, *, start, step, device, dtype) -> BoundSymbol:
    sym = Symbol(name="iota_helper", meta=None)
    ctx: dict[str, Any] = {"iota_helper": _iota_helper}

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


# NOTE This helper is necessary because PyTorch doesn't define a uniform operation
def uniform_helper(shape: Sequence[int], minval: Number, maxval: Number, *, device: torch.device, dtype: torch.dtype):
    t = torch.empty(shape, device=device, dtype=dtype)
    t.uniform_(minval, maxval)
    return t


def uniform_prim(
    bsym: BoundSymbol,
    shape: Sequence[int],
    minval: Number,
    maxval: Number,
    *,
    device: devices.Device,
    dtype: dtypes.dtype,
) -> BoundSymbol:
    sym = Symbol(name="uniform_helper", meta=None)
    ctx: dict[str, Any] = {"uniform_helper": uniform_helper}

    torch_device = ltorch.to_torch_device(device)
    torch_dtype = ltorch.to_torch_dtype(dtype)

    return sym.bind(shape, minval, maxval, device=torch_device, dtype=torch_dtype, output=bsym.output, _call_ctx=ctx)


def zeros_like(
    bsym: BoundSymbol,
    a: TensorLike,
    /,
    *,
    device: Optional[DeviceLike] = None,
    dtype: Optional[dtypeLike] = None,
) -> BoundSymbol:
    sym = Symbol(name="zeros_like", meta=None, _module=torch)

    device = ltorch.to_torch_device(device)
    dtype = ltorch.to_torch_dtype(dtype)

    return sym.bind(a, device=device, dtype=dtype, output=bsym.output)


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


def expand(bsym: BoundSymbol, tensor, *shape):
    sym = Symbol(name="expand", meta=None, _module=torch.Tensor)
    return sym.bind(tensor, *shape, output=bsym.output)


def flatten(bsym: BoundSymbol, a: TensorLike, start_dim: int = 0, end_dim: int = -1) -> TensorLike:
    sym = Symbol(name="flatten", meta=None, _module=torch)
    return sym.bind(a, start_dim, end_dim, output=bsym.output)


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


def movedim(
    bsym: BoundSymbol, a: TensorLike, /, source: int | Sequence[int], destination: int | Sequence[int]
) -> TensorLike:
    sym = Symbol(name="movedim", meta=None, _module=torch)
    return sym.bind(a, source, destination, output=bsym.output)


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


def reshape(bsym: BoundSymbol, a, *shape):
    shape = utils.extract_shape_from_varargs(shape)
    sym = Symbol(name="reshape", meta=None, _module=torch)
    return sym.bind(a, shape, output=bsym.output)


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


# NOTE Order of index, value and dim changes
def index_add(bsym: BoundSymbol, a, index, value, dim):
    sym = Symbol(name="index_add", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim, index, value), kwargs={}, output=bsym.output)

    return tbsym


def take_along_axis(bsym: BoundSymbol, a, index, dim):
    sym = Symbol(name="take_along_dim", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, index, dim), kwargs={}, output=bsym.output)

    return tbsym


# NOTE Order of index, value and dim changes
def scatter_add(bsym: BoundSymbol, a, index, value, dim):
    sym = Symbol(name="scatter_add", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dim, index, value), kwargs={}, output=bsym.output)

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


def permute(bsym: BoundSymbol, a, *dims: int):
    # NOTE This is necessary because torch.permute() requires the permutation be
    #   specified as a tuple, not varargs
    dims = utils.extract_shape_from_varargs(dims)
    sym = Symbol(name="permute", meta=None, _module=torch)
    return sym.bind(a, dims, output=bsym.output)


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


# NOTE Most PyTorch elementwise binary operations do not support number x number inputs, and the few that do
#   (like add, mul, sub, and div) return tensors instead of numbers
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


# NOTE add and sub have special check and factory functions to support alpha
def _add_sub_check(
    a: Union[TensorProxy, Number], b: Union[TensorProxy, Number], *, alpha: Optional[Number] = None
) -> bool:
    return not (isinstance(a, Number) and isinstance(b, Number))


def _add_sub_factory(name: str) -> Callable:
    def fn(
        bsym: BoundSymbol,
        a: Union[TensorProxy, Number],
        b: Union[TensorProxy, Number],
        *,
        alpha: Optional[Number] = None,
    ) -> BoundSymbol:
        sym = Symbol(name=name, meta=None, _module=torch)
        # NOTE The alpha kwarg can ONLY be specified to PyTorch if it's not None
        if alpha is not None:
            tbsym = sym.bind(a, b, alpha=alpha, output=bsym.output)
        else:
            tbsym = sym.bind(a, b, output=bsym.output)
        return tbsym

    return fn


add = _add_sub_factory("add")
atan2 = _elementwise_binary_factory("atan2")
bitwise_and = _elementwise_binary_factory("bitwise_and")
bitwise_or = _elementwise_binary_factory("bitwise_or")
bitwise_xor = _elementwise_binary_factory("bitwise_xor")
copysign = _elementwise_binary_factory("copysign")
div = _elementwise_binary_factory("div")
floor_divide = _elementwise_binary_factory("floor_divide")
eq = _elementwise_binary_factory("eq")
fmod = _elementwise_binary_factory("fmod")
ge = _elementwise_binary_factory("ge")
gt = _elementwise_binary_factory("gt")
logical_and = _elementwise_binary_factory("logical_and")
le = _elementwise_binary_factory("le")
lt = _elementwise_binary_factory("lt")
mul = _elementwise_binary_factory("mul")
ne = _elementwise_binary_factory("ne")
nextafter = _elementwise_binary_factory("nextafter")
pow = _elementwise_binary_factory("pow")
# TODO Remainder bool isn't implement, so mark it as non-fusible
remainder = _elementwise_binary_factory("remainder")
sub = _add_sub_factory("sub")

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


def var_mean_prim(bsym: BoundSymbol, a: TensorProxy, dims, *, correction: int) -> BoundSymbol:
    sym = Symbol(name="var_mean", meta=None, _module=torch)
    tbsym = BoundSymbol(sym, args=(a, dims), kwargs={"correction": correction}, output=bsym.output)

    return tbsym


# NOTE var does not allow keepdim to be specified if dim is not specified
# NOTE This is the direct translation of torch.var
def _var_factory(name: str) -> Callable:
    def fn(
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

        sym = Symbol(name=name, meta=None, _module=torch)

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

    return fn


var = _var_factory("var")
var_mean = _var_factory("var_mean")

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
def cross_entropy(
    bsym: BoundSymbol,
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    sym = Symbol(name="cross_entropy", _module=torch.nn.functional)
    return sym.bind(
        input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing, output=bsym.output
    )


def _cross_entropy_backward_helper(
    g: torch.Tensor,
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    reduction: str,
    ignore_index: int,
    label_smoothing: int,
) -> torch.Tensor:
    # forward - given input and target
    # a = log_softmax(input, dim)
    # return cross_entropy(a, target, weight, reduction, ignore_index, label_smoothing)

    # backward - given grad_cross_entropy and saved_tensors
    # grad_a = torch.ops.aten.nll_loss_backward(grad, a, target, weight, reduction, ignore_index, total_weight)
    # return torch.ops.aten._log_softmax_backward_data(grad_a, a, dim, a.scalar_type())

    if reduction == "none":
        reduction_idx = 0
    elif reduction == "mean":
        reduction_idx = 1
    elif reduction == "sum":
        reduction_idx = 2
    else:
        reduction_idx = -1

    utils.check(
        reduction_idx > -1 and reduction_idx < 3,
        lambda: f"{reduction} is not a valid value for reduction parameter.",
    )

    # TODO Add support nll_loss_nd, weight tensor, and label_smoothing options.
    # See https://github.com/Lightning-AI/lightning-thunder/issues/704
    utils.check(input.ndim <= 2 and target.ndim <= 1, lambda: f"multi-dimension cross-entropy is not supported.")

    utils.check(weight is None, lambda: f"weight tensor argument is not supported.")

    utils.check(label_smoothing == 0.0, lambda: f"label smoothing values not equal to zero are not supported.")

    batch_size = 0 if input.dim() <= 1 else input.shape[1]
    dim = 0 if input.dim() == 1 else 1
    a = torch.log_softmax(input, dim, input.dtype)

    if weight is not None:
        total_weight = torch.sum(weight)
    elif reduction == "none":
        total_weight = torch.tensor(0.0, dtype=input.dtype, device=input.device)
    elif reduction == "sum" or reduction == "mean":
        total_weight = torch.sum(torch.ne(target, ignore_index)).to(dtype=input.dtype, device=input.device)

    g_a = torch.ops.aten.nll_loss_backward(g, a, target, weight, reduction_idx, ignore_index, total_weight)
    return torch.ops.aten._log_softmax_backward_data(g_a, a, dim, input.dtype)


def cross_entropy_backward(
    bsym: BoundSymbol,
    grad: TensorProxy,
    input: TensorProxy,
    target: TensorProxy,
    weight: TensorProxy = None,
    reduction: str = "mean",
    ignore_index: Number = -100,
    label_smoothing: Number = 0.0,
) -> BoundSymbol:
    sym = Symbol(name="cross_entropy_backward", meta=None)
    ctx: Dict[str, Any] = {"cross_entropy_backward": _cross_entropy_backward_helper}
    return sym.bind(
        grad, input, target, weight, reduction, ignore_index, label_smoothing, output=bsym.output, _call_ctx=ctx
    )


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

def gelu(bsym: BoundSymbol, a: TensorProxy, *, approximate: str = "none") -> BoundSymbol:
    sym = Symbol(name="gelu", meta=None, _module=torch.nn.functional)
    return sym.bind(a, approximate=approximate, output=bsym.output)

def layer_norm(bsym: BoundSymbol, a, normalized_shape, weight=None, bias=None, eps: Number = 1e-5):
    sym = Symbol(name="layer_norm", meta=None, _module=torch.nn.functional)
    tbsym = BoundSymbol(sym, args=(a, normalized_shape, weight, bias, eps), kwargs={}, output=bsym.output)

    return tbsym


def relu(bsym: BoundSymbol, a: TensorProxy, inplace=False) -> BoundSymbol:
    sym = Symbol(name="relu", meta=None, _module=torch.nn.functional)
    # NOTE: inplace is ignored since only
    # inplace=False is supported and it has a default value.
    return sym.bind(a, output=bsym.output)


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


def _scaled_dot_product_attention_check(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
) -> bool:
    # TODO: Model PyTorch's choice of efficient kernels and fallbacks
    # See https://github.com/Lightning-AI/lightning-thunder/issues/622
    if scale is not None and LooseVersion(torch.__version__) < LooseVersion("2.1.0"):
        return False
    return True


def scaled_dot_product_attention(
    bsym: BoundSymbol,
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
) -> BoundSymbol:
    sym = Symbol(name="scaled_dot_product_attention", meta=None, _module=torch.nn.functional, id=bsym.sym.id)

    kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": bool(is_causal),
    }
    if LooseVersion(torch.__version__) >= LooseVersion("2.1.0"):
        kwargs["scale"] = scale
    else:
        utils.check(
            scale is None,
            lambda: f"scaled_dot_product_attention with scale argument requires PyTorch >= 2.1.0",
        )

    tbsym = BoundSymbol(sym, args=(query, key, value), kwargs=kwargs, output=bsym.output)
    return tbsym


#
# Distributed Operations
#
# NOTE DISTRIBUTED AVAILABILITY
# PyTorch is often built without distributed support, which can be queried for using
#   torch.distributed.is_available(). When PyTorch is built without distributed then we
#   want to avoid accessing any parts of the torch.distributed module except
#   the is_available() function.

if torch.distributed.is_available():

    def all_reduce_prim_helper(
        a: torch.Tensor,
        op: torch.distributed.ReduceOp,
        group: torch.distributed.ProcessGroup,
        do_async: bool,
    ) -> torch.Tensor | tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]:
        c = a.clone()

        handle = torch.distributed.all_reduce(c, op, group, do_async)

        # NOTE This returns (handle, tensor reference), which is how FutureTensorProxies
        #   are currently modeled by executors
        if do_async:
            return handle, c

        # NOTE do_async is False
        return c

    def all_reduce_prim(
        bsym: BoundSymbol,
        a: TensorProxy,
        op: DistributedReduceOps,
        group: torch.distributed.ProcessGroup,
        do_async: Number,
    ) -> BoundSymbol:
        sym = Symbol(name="all_reduce_prim_helper", meta=None)
        ctx: dict[str, Any] = {"all_reduce_prim_helper": all_reduce_prim_helper}
        do_async = bool(do_async)

        op = ltorch.to_torch_distributed_reduce_op(op)

        return sym.bind(a, op, group, do_async, output=bsym.output, _call_ctx=ctx)

    # NOTE This is a very particular implementation of wait that may need to be
    #   generalized in the future
    # NOTE The implementation of wait actually models the FutureTensorProxy as a tuple
    #   of (handle, tensor reference), there's no conflict with the trace representation
    #   with doing this, as the trace representation treates FutureTensorProxy as opaque
    # NOTE This way of representing FutureTensorProxies may need to change in the future
    def wait_helper(a: tuple[torch.distributed.distributed_c10d.Work, torch.Tensor]) -> torch.Tensor:
        handle, tensor_ref = a
        handle.wait()
        return tensor_ref

    def wait_prim(bsym: BoundSymbol, a: FutureTensorProxy) -> BoundSymbol:
        sym = Symbol(name="wait_helper", meta=None)
        ctx: dict[str, Any] = {"wait_helper": wait_helper}

        return sym.bind(a, output=bsym.output, _call_ctx=ctx)

else:

    def all_reduce_prim(bsym: BoundSymbol, a: TensorProxy, op: Any, group: Any, do_async: bool) -> None:
        utils.check(False, lambda: f"torch.distributed is not available")

    def wait_prim(bsym: BoundSymbol, a: FutureTensorProxy) -> BoundSymbol:
        utils.check(False, lambda: f"torch.distributed is not available")


# TODO Refine prim ops to have less functionality to better debug errors
# Maps from symbol ids to a tuple of (is_fusible, translate) callables
_ops_map.update(
    {
        # Data movement operations
        PrimIDs.CONVERT_ELEMENT_TYPE: (_always_executable, convert_element_type),
        PrimIDs.DEVICE_PUT: (_always_executable, device_put),
        "torch.Tensor.to": (_always_executable, to),
        # Tensor creation operations
        "torch.arange": (_always_executable, arange),
        PrimIDs.EXOGENOUS_LIKE: (_always_executable, exogenous_like),
        PrimIDs.FULL: (_always_executable, full),
        "torch.full_like": (_always_executable, full_like),
        PrimIDs.IOTA: (_always_executable, iota),
        PrimIDs.UNIFORM: (_always_executable, uniform_prim),
        "torch.zeros_like": (_always_executable, zeros_like),
        # Shape operations
        PrimIDs.BROADCAST_IN_DIM: (_always_executable, broadcast_in_dim),
        "torch.cat": (_always_executable, cat),
        PrimIDs.CAT: (_always_executable, cat),
        "torch.Tensor.contiguous": (_always_executable, contiguous),
        "torch.Tensor.expand": (_always_executable, expand),
        "torch.flatten": (_always_executable, flatten),
        "torch.Tensor.__getitem__": (_always_executable, getitem),
        "torch.movedim": (_always_executable, movedim),
        PrimIDs.PAD: (_always_executable, pad),
        PrimIDs.RESHAPE: (_always_executable, reshape),
        "torch.reshape": (_always_executable, reshape),
        PrimIDs.SLICE: (_always_executable, slice_prim),
        "torch.split": (_always_executable, split),
        "torch.squeeze": (_always_executable, squeeze),
        PrimIDs.SQUEEZE: (_always_executable, squeeze),
        PrimIDs.TAKE: (_always_executable, take),
        PrimIDs.INDEX_ADD: (_always_executable, index_add),
        PrimIDs.TAKE_ALONG_AXIS: (_always_executable, take_along_axis),
        PrimIDs.SCATTER_ADD: (_always_executable, scatter_add),
        "torch.tensor_split": (_always_executable, tensor_split),
        "torch.transpose": (_always_executable, transpose),
        PrimIDs.TRANSPOSE: (_always_executable, prim_transpose),
        "torch.permute": (_always_executable, permute),
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
        "torch.relu": (_always_executable, relu),
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
        "torch.add": (_add_sub_check, add),
        PrimIDs.ADD: (_elementwise_binary_check, add),
        "torch.atan2": (_elementwise_binary_check, atan2),
        PrimIDs.ATAN2: (_elementwise_binary_check, atan2),
        "torch.bitwise_and": (_elementwise_binary_check, bitwise_and),
        PrimIDs.BITWISE_AND: (_elementwise_binary_check, bitwise_and),
        "torch.bitwise_or": (_elementwise_binary_check, bitwise_or),
        PrimIDs.BITWISE_OR: (_elementwise_binary_check, bitwise_or),
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
        "torch.le": (_elementwise_binary_check, le),
        PrimIDs.LE: (_elementwise_binary_check, le),
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
        "torch.sub": (_add_sub_check, sub),
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
        PrimIDs.VAR_MEAN: (_always_executable, var_mean_prim),
        # Matmul operations
        PrimIDs.LINEAR: (_always_executable, linear),
        PrimIDs.MATMUL: (_always_executable, matmul),
        # NN operations
        "torch.nn.functional.cross_entropy": (_always_executable, cross_entropy),
        "cross_entropy_backward": (_always_executable, cross_entropy_backward),
        "torch.nn.functional.dropout": (_always_executable, dropout),
        PrimIDs.EMBEDDING: (_always_executable, embedding),
        PrimIDs.EMBEDDING_BACKWARD: (_always_executable, embedding_backward),
        "torch.nn.functional.embedding": (_always_executable, embedding),
        "torch.nn.functional.gelu": (_always_executable, gelu),
        "torch.layer_norm": (_always_executable, layer_norm),
        "torch.softmax": (_always_executable, softmax),
        "torch.nn.functional.scaled_dot_product_attention": (
            _scaled_dot_product_attention_check,
            scaled_dot_product_attention,
        ),
        # Distributed operations
        PrimIDs.ALL_REDUCE: (_always_executable, all_reduce_prim),
        PrimIDs.WAIT: (_always_executable, wait_prim),
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
# Code related to PyTorch's grad transform
#


class ThunderFunction(torch.autograd.Function):
    @staticmethod
    def get_forward_backward_splitter(func, compile_config):
        from thunder import trace
        from thunder.executors import transform_for_execution
        from thunder.executors.passes import del_last_used
        from thunder.core.rematerialization import split_vjp_trace_into_forward_and_backward
        from thunder.core.transforms import inline, vjp

        def make_trace(func):
            return partial(trace, func, inline_trace=False)

        def split_forward_backward(*args, **kwargs):
            # NOTE: This function is rather slow, so it's intended to be used
            # behind a cache.

            # We need to run the function once to get the outputs that can be
            # used to construct the backward trace
            # Since autograd.Function accepts flattened inputs in forward,
            # and expects flattened outputs in backward, we need to flatten
            # the given function and its inputs when we generate the "joint trace"
            flat_func, flat_args, spec = utils.flatten_func(func, args, kwargs)
            out = flat_func(*flat_args)
            joint_trace = make_trace(inline(vjp(flat_func)))(flat_args, utils.sequencify(out))
            extrace, _ = transform_for_execution(
                joint_trace,
                executors_list=compile_config.get("executors_list", None),
                only_execute_prims=compile_config.get("only_execute_prims", False),
                use_rematerialization=compile_config.get("use_rematerialization", True),
            )
            forward_trace, backward_trace = split_vjp_trace_into_forward_and_backward(extrace)
            # Del calls were ignored in the splitting process, so we need to call it here
            forward_trace, _ = del_last_used(forward_trace)
            backward_trace, _ = del_last_used(backward_trace)
            backward_trace_fn = backward_trace.python_callable()

            def backward_fn(saved_info, *args):
                return backward_trace_fn(*saved_info, *args)

            return forward_trace.python_callable(), backward_fn

        return split_forward_backward

    @staticmethod
    def forward(ctx, split_fw_bw, spec, *flat_args):
        # We can't send unflatten args with a spec to the split_fw_bw
        # function, because spec is not hashable. So we need to unflatten
        # the args here.
        args, kwargs = tree_unflatten(flat_args, spec)
        compiled_forward, compiled_backward = split_fw_bw(*args, **kwargs)
        out, saved_info = compiled_forward(*flat_args)
        ctx.compiled_backward = compiled_backward

        # We must save tensors using ctx.save_for_backward
        is_tensor = tuple(isinstance(x, torch.Tensor) for x in saved_info)
        ctx.is_all_tensor = all(is_tensor)
        if ctx.is_all_tensor:
            ctx.save_for_backward(*saved_info)
            return out

        ctx.save_for_backward(*(x for x, is_tensor in zip(saved_info, is_tensor) if is_tensor))
        ctx.saved_non_tensors = tuple(x for x, is_tensor in zip(saved_info, is_tensor) if not is_tensor)
        ctx.is_tensor = is_tensor
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *args):
        # Restore saved_info from ctx.saved_non_tensors and ctx.saved_tensors
        if ctx.is_all_tensor:
            flat_saved_info = ctx.saved_tensors
        else:
            saved_tensors = iter(ctx.saved_tensors)
            saved_non_tensors = iter(ctx.saved_non_tensors)
            flat_saved_info = tuple(
                next(saved_tensors) if is_tensor else next(saved_non_tensors) for is_tensor in ctx.is_tensor
            )
        grads = ctx.compiled_backward(flat_saved_info, *args)
        return (None, None, *grads)


def thunder_backward(**compile_config):
    """Decorator to wrap a Thunder function for use with PyTorch autograd.

    Args:
        thunder_func: A Thunder function.

    Returns:
        A wrapped function that can be used with PyTorch autograd.

    Example:
    >>> import torch
    >>> import thunder.clang as clang
    >>> from thunder.executors.torchex import thunder_backward
    >>> @thunder_backward()
    ... def func(a, b):
    ...     c = a + b
    ...     d = c * b
    ...     e = clang.sin(d) + clang.cos(c)
    ...     return e
    >>> a = torch.randn(3, device="cuda", requires_grad=True)
    >>> b = torch.randn(3, device="cuda", requires_grad=True)
    >>> c = func(a, b)
    >>> print(c)
    >>> sum(c).sum().backward()
    >>> print(f"a.grad: {a.grad}")
    >>> print(f"b.grad: {b.grad}")
    """

    compile_config = compile_config | {"disable_preprocessing": True}

    def decorator(thunder_func):
        from thunder import compile

        # Compile's caching only works for many calls to the same compiled function
        # It does not work if the same function is compiled many times, so we must
        # decorate the augmented forward pass once with compile once and reuse it
        split_fw_bw = ThunderFunction.get_forward_backward_splitter(thunder_func, compile_config)
        compiled_split_fw_bw = compile(
            split_fw_bw,
            **compile_config,
        )

        @wraps(thunder_func)
        def wrapper(*args, **kwargs):
            # We must save the spec of the args to be able to unflatten them later
            # torch.autograd.Functions support only flat arguments
            flat_args, spec = tree_flatten((args, kwargs))
            return ThunderFunction.apply(compiled_split_fw_bw, spec, *flat_args)

        return wrapper

    return decorator
