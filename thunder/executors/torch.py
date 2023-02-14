import operator
from numbers import Number
from functools import wraps

import torch

import thunder.core.dtypes as dtypes
import thunder.langs.torch as ttorch
from thunder.core import prims
from thunder.core.proxies import Proxy, TensorProxy
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.trace import Trace, Variable
from thunder.executors.executor_prims import nvOps

__all__ = [
    "torchCtx",
    "_fuse_region",
]

# TODO: can probably remove weak dtypes from this map
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


def convert_element_type(a, dtype):
    # Handles converting a tensor to a numbertype, which Thunder allows but
    #   Torch does not
    if isinstance(a, torch.Tensor) and dtypes.is_numbertype(dtype):
        dtype = ttorch.torch_dtype(dtypes.numbertype_to_dtype(dtype))

    # Handles number conversions
    if isinstance(a, Number):
        if not dtypes.is_numbertype(dtype):
            dtype = dtypes.dtype_to_numbertype(ttorch.thunder_dtype(dtype))
        return dtype(a)

    return a.to(dtype)


def broadcast_in_dim(a, shape, broadcast_dims):
    s = list(shape)
    for broadcast_dim in broadcast_dims:
        s[broadcast_dim] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = v.unsqueeze(idx)

    return v.expand(shape)


def slice_helper(a, start_indices, end_indices, strides=None):
    _strides = strides if strides is not None else [1] * len(start_indices)

    slices = []
    for start, stop, step in zip(start_indices, end_indices, _strides):
        slices.append(slice(start, stop, step))

    return operator.getitem(a, slices)


# TODO: dim as a sequence is only supported on PyTorch 2.0 and greater
def squeeze_helper(a, dim):
    for d in sorted(dim, reverse=True):
        a = a.squeeze(d)

    return a


def view_helper(a, shape):
    return a.view(shape)


def is_tensor(a):
    return isinstance(a, torch.Tensor)


def iota_helper(length, *, start, step, device, dtype):
    end = start + length * step
    return torch.arange(start=start, step=step, end=end, device=device, dtype=dtype)


def uniform_helper(shape, minval=0.0, maxval=1.0, *, device, dtype):
    t = torch.empty(shape, device=device, dtype=dtype)
    t.uniform_(minval, maxval)
    return t


# NOTE: many PyTorch operations don't accept numbers as inputs,
#   so this helper wraps and unwraps numbers
def _elementwise_unary_torch(op):
    @wraps(op)
    def _fn(x):
        if isinstance(x, torch.Tensor):
            return op(x)

        return op(torch.tensor(x)).item()

    return _fn


# Handles adding two Python numbers, which PyTorch allows but returns
#   as a tensor, while Thunder expects a Python number
def add_helper(a, b, alpha=1):
    if any(map(is_tensor, (a, b, alpha))):
        return torch.add(a, b, alpha=alpha)

    return a + b * alpha


# NOTE: PyTorch's torch.eq expects tensor x tensor or tensor x number
#   but the == operator allows number x tensor
def eq_helper(a, b):
    return a == b


# Maps the Thunder primitives to their corresponding torch operation names
# TODO: handle more scalar arguments (like add does above)
ops_to_torch_ops_map = {
    # Data movement and transformation prims
    prims.Ops.CONVERT_ELEMENT_TYPE: convert_element_type,
    # Tensor creation prims
    prims.Ops.FULL: "torch.full",
    prims.Ops.IOTA: iota_helper,
    prims.Ops.UNIFORM: uniform_helper,
    # Shape prims
    prims.Ops.BROADCAST_IN_DIM: broadcast_in_dim,
    prims.Ops.RESHAPE: "torch.reshape",
    prims.Ops.SLICE: slice_helper,
    prims.Ops.SQUEEZE: squeeze_helper,
    # NOTE: PyTorch's transpose is not equivalent to the transpose prim
    prims.Ops.TRANSPOSE: "torch.permute",
    prims.Ops.VIEW: view_helper,
    # Elementwise unary prims
    prims.Ops.ABS: _elementwise_unary_torch(torch.abs),
    prims.Ops.ACOS: _elementwise_unary_torch(torch.acos),
    prims.Ops.ACOSH: _elementwise_unary_torch(torch.acosh),
    prims.Ops.ASIN: _elementwise_unary_torch(torch.asin),
    prims.Ops.ATAN: _elementwise_unary_torch(torch.atan),
    prims.Ops.ATANH: _elementwise_unary_torch(torch.atanh),
    prims.Ops.BITWISE_NOT: _elementwise_unary_torch(torch.bitwise_not),
    prims.Ops.CEIL: _elementwise_unary_torch(torch.ceil),
    prims.Ops.COS: _elementwise_unary_torch(torch.cos),
    prims.Ops.COSH: _elementwise_unary_torch(torch.cosh),
    prims.Ops.ERF: _elementwise_unary_torch(torch.erf),
    prims.Ops.ERFC: _elementwise_unary_torch(torch.erfc),
    prims.Ops.EXP: _elementwise_unary_torch(torch.exp),
    prims.Ops.EXPM1: _elementwise_unary_torch(torch.expm1),
    prims.Ops.FLOOR: _elementwise_unary_torch(torch.floor),
    prims.Ops.ISFINITE: _elementwise_unary_torch(torch.isfinite),
    prims.Ops.RSQRT: _elementwise_unary_torch(torch.rsqrt),
    prims.Ops.SIN: _elementwise_unary_torch(torch.sin),
    prims.Ops.SQRT: _elementwise_unary_torch(torch.sqrt),
    prims.Ops.TANH: _elementwise_unary_torch(torch.tanh),
    prims.Ops.LOG: _elementwise_unary_torch(torch.log),
    prims.Ops.LOG10: _elementwise_unary_torch(torch.log10),
    prims.Ops.LOG1P: _elementwise_unary_torch(torch.log1p),
    prims.Ops.LOG2: _elementwise_unary_torch(torch.log2),
    # Elementwise binary prims
    prims.Ops.ADD: add_helper,
    prims.Ops.ATAN2: "torch.atan2",
    prims.Ops.BITWISE_AND: "torch.bitwise_and",
    prims.Ops.DIV: "torch.div",
    prims.Ops.EQ: eq_helper,
    prims.Ops.LT: "torch.lt",
    prims.Ops.MUL: "torch.mul",
    prims.Ops.POW: "torch.pow",
    prims.Ops.SUB: "torch.sub",
    # Elementwise ternary prims
    prims.Ops.WHERE: "torch.where",
    # Reduction prims
    prims.Ops.AMAX: "torch.amax",
    prims.Ops.SUM: "torch.sum",
    prims.Ops.VAR: "torch.var",
    # NOTE: VAR_MEAN is here to execute nvFuser traces with PyTorch
    nvOps.VAR_MEAN: "torch.var_mean",
    # Matmul prims
    prims.Ops.LINEAR: "torch.nn.functional.linear",
    prims.Ops.MATMUL: "torch.matmul",
    # NN prims
    prims.Ops.EMBEDDING: "torch.nn.functional.embedding",
}


# NOTE: this class is here to help with proper printing
class ProxyName:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def _get_torch(x):
    if dtypes.is_dtype(x) and not dtypes.is_numbertype(x):
        return _thunder_to_torch_dtype_map[x]

    if isinstance(x, str):
        return f"'{x}'"

    if isinstance(x, type):
        return x.__name__

    if isinstance(x, Variable):
        return ProxyName(x.name)

    return x


def _get_torch_op(op):
    return ops_to_torch_ops_map[op]


# Acquires a proxies name or passes a constant by value
# TODO: put this into executors utils
def _extract_name(x):
    if isinstance(x, (Variable, Proxy)):
        return x.name

    return str(x)


# TODO: refactor _fuse_region to be called by a common executor utility to generate fusions
def _fuse_region(inputs, outputs, symbols, *, _return_code=False, _contiguous=True):
    # Defines utilities
    tab = "  "

    # Initializes context
    ctx = {
        "torch": torch,
        "inf": float("inf"),  # NOTE: not always necessary
    }

    # Creates signature
    # NOTE: PyTorch fusions are run in a no grad context
    arg_str = ", ".join(tuple(_extract_name(inp) for inp in inputs))
    cstr = f"@torch.no_grad()\ndef fusion({arg_str}):"

    # Calls PyTorch and Python operations
    cstr += f"\n{tab}# Executes the trace"
    for sym in symbols:
        torch_args = tree_map(_get_torch, sym.args)
        torch_kwargs = tree_map(_get_torch, sym.kwargs)
        torch_op = _get_torch_op(sym.op)
        result = sym.outputs[0]

        # TODO: relax requirement that prim outputs are proxies?
        for out in sym.outputs:
            if not isinstance(out, Variable):
                raise NotImplementedError

        op_str = None
        if isinstance(torch_op, str):
            op_str = torch_op
        else:
            op_str = torch_op.__name__
            ctx[op_str] = torch_op

        result_str = ", ".join(out.name for out in sym.outputs)
        arg_str = ", ".join(f"{a}" for a in torch_args)
        kwarg_str = ", ".join(f"{k}={v}" for k, v in torch_kwargs.items())
        segue_str = ", " if (len(arg_str) > 0 and len(kwarg_str) > 0) else ""

        cstr += f"\n{tab}{result_str} = {op_str}({arg_str}{segue_str}{kwarg_str})"

    # Constructs outputs
    output_strs = []
    for out in outputs:
        if isinstance(out, TensorProxy):
            if _contiguous:
                # TODO: FIXME: currently makes all outputs contiguous to simplify stride analysis
                output_strs.append(f"{_extract_name(out)}.contiguous()")
            else:
                output_strs.append(f"{_extract_name(out)}")
        else:
            output_strs.append(_extract_name(out))
    out_str = ", ".join(output_strs)
    cstr += f"\n{tab}return {out_str}"

    code = compile(cstr, "torch.gen", mode="exec")
    exec(code, ctx)
    fusion = ctx["fusion"]

    if _return_code:
        return fusion, cstr

    return fusion


# TODO: intercept PyTorch operations and handle functions whose results are
#   bound to multiple values (i.e. a, b = foo(x, y))
# Creates a Python callable that executes the trace using PyTorch and Python
# NOTE: does this by compiling a function from a string
def _fuse(trace):
    flat_outputs, output_structure = tree_flatten(trace.outputs)

    # Short-circuits if the fusion has no outputs
    if len(flat_outputs) == 0:

        def _fusion(*args, **kwargs):
            return None

        return _fusion

    #
    # Constructs the program
    #

    # Writes the signatures
    # NOTE: PyTorch fusions are run in a no grad context
    tab = "  "
    cstr = f"@torch.no_grad()\ndef fusion(*args, **kwargs):"
    # TODO: maybe consider the possibility of name conflicts?
    ctx = {
        "torch": torch,
        "tree_flatten": tree_flatten,
        "tree_unflatten": tree_unflatten,
        "output_structure": output_structure,
        "inf": float("inf"),  # NOTE: not always necessary
    }

    # Acquires inputs
    flat_positional_inputs, _ = tree_flatten(trace.args)
    flat_kwarg_inputs, _ = tree_flatten(trace.kwargs)

    cstr += f"\n{tab}# Extracts inputs"
    cstr += f"\n{tab}flat_args, _ = tree_flatten(args)"
    cstr += f"\n{tab}flat_kwargs, _ = tree_flatten(kwargs)"

    for idx, pinp in enumerate(flat_positional_inputs):
        if isinstance(pinp, Variable):
            cstr += f"\n{tab}{pinp.name} = flat_args[{idx}]"
    for idx, kwinp in enumerate(flat_kwarg_inputs):
        if isinstance(kwinp, Variable):
            cstr += f"\n{tab}{kwinp.name} = flat_kwargs[{idx}]"

    # Calls PyTorch and Python operations
    cstr += f"\n{tab}# Executes the trace"
    for sym in trace.symbols:
        torch_args = tree_map(_get_torch, sym.args)
        torch_kwargs = tree_map(_get_torch, sym.kwargs)
        torch_op = _get_torch_op(sym.op)
        result = sym.outputs[0]

        if not isinstance(result, Variable):
            raise NotImplementedError

        # NOTE: currently assumes result is always a proxy
        op_str = None
        if isinstance(torch_op, str):
            op_str = torch_op
        else:
            op_str = torch_op.__name__
            ctx[op_str] = torch_op

        # NOTE: currently assumes that the trace is stored in the "trace" kwarg
        if "trace" in torch_kwargs and any(isinstance(v, Trace) for v in torch_kwargs.values()):
            key = result.name + "_" + op_str + "_trace"
            ctx[key] = torch_kwargs["trace"]
            torch_kwargs["trace"] = key

        arg_str = ", ".join(f"{a}" for a in torch_args)
        kwarg_str = ", ".join(f"{k}={v}" for k, v in torch_kwargs.items())
        segue_str = ", " if (len(arg_str) > 0 and len(kwarg_str) > 0) else ""

        cstr += f"\n{tab}{result.name} = {op_str}({arg_str}{segue_str}{kwarg_str})"

    # Constructs output
    # NOTE: len(flat_outputs) > 0
    torch_outputs = tree_map(_get_torch, flat_outputs)
    output_str = ", ".join(_extract_name(x) for x in torch_outputs)
    cstr += f"\n{tab}return tree_unflatten(({output_str},), output_structure)"

    # Compiles the function
    code = compile(cstr, "torch.gen", mode="exec")
    exec(code, ctx)
    fusion = ctx["fusion"]

    return fusion


class torchCtx:
    def __init__(self):
        pass

    def intercept(self, op):
        return None

    # TODO: maybe return some actual profiling information
    def fuse(self, trace, *, profile_info=False):
        if profile_info:
            return _fuse(trace), None
        return _fuse(trace)
