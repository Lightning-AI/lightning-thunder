from functools import partial
from numbers import Number

import torch
from looseversion import LooseVersion

import thunder.core.dtypes as dtypes

# TODO: review language and executor dependencies
import thunder.langs.torch as ttorch
from thunder import make_trace
from thunder.core import prims, utils
from thunder.core.proxies import NumberProxy, Proxy, TensorProxy
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.trace import Variable
from thunder.core.transforms import eval_trace
from thunder.core.utils import OrderedSet

# TODO: consider further refactoring this
from thunder.executors.executor_prims import nvOps
from thunder.executors.torch import _fuse_region as _fuse_torch_region

# Imports nvFuser
# NOTE: nvFuser API changed after PyTorch 1.13
nvfuser_version = LooseVersion("0.0.0")
try:
    import nvfuser

    if hasattr(nvfuser, "version"):
        nvfuser_version = LooseVersion(nvfuser.version())
        from nvfuser import DataType, FusionDefinition
    else:
        from nvfuser._C import DataType, Fusion, FusionDefinition

    nvTensor = nvfuser._C.Tensor
    nvNumber = nvfuser._C.Scalar
except ImportError:
    import torch._C._nvfuser as nvfuser
    from torch._C._nvfuser import DataType, Fusion, FusionDefinition

    nvTensor = torch._C._nvfuser.Tensor
    nvNumber = torch._C._nvfuser.Scalar

# NOTE: "reshape" used to be called "view"
use_reshape = hasattr(FusionDefinition.Operators, "reshape")

__all__ = [
    "nvFuserCtx",
]


_torch_dtype_to_nvfuser_dtype_map = {
    torch.cdouble: DataType.ComplexDouble,
    torch.cfloat: DataType.ComplexFloat,
    torch.double: DataType.Double,
    torch.float: DataType.Float,
    torch.half: DataType.Half,
    torch.bfloat16: DataType.BFloat16,
    torch.long: DataType.Int,
    torch.int: DataType.Int32,
    torch.bool: DataType.Bool,
    # Python scalars
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
}

_thunder_dtype_to_nvfuser_dtype_map = {
    dtypes.complex128: DataType.ComplexDouble,
    dtypes.complex64: DataType.ComplexFloat,
    dtypes.float64: DataType.Double,
    dtypes.float32: DataType.Float,
    dtypes.float16: DataType.Half,
    dtypes.bfloat16: DataType.BFloat16,
    dtypes.int64: DataType.Int,
    dtypes.int32: DataType.Int32,
    dtypes.bool8: DataType.Bool,
    dtypes.complex128_: DataType.ComplexDouble,
    dtypes.complex64_: DataType.ComplexFloat,
    dtypes.float64_: DataType.Double,
    dtypes.float32_: DataType.Float,
    dtypes.float16_: DataType.Half,
    dtypes.bfloat16_: DataType.BFloat16,
    dtypes.int64_: DataType.Int,
    dtypes.int32_: DataType.Int32,
    dtypes.bool8_: DataType.Bool,
}

_thunder_dtype_to_nvfuser_dtype_scalar_map = {
    complex: DataType.ComplexDouble,
    float: DataType.Double,
    int: DataType.Int,
    bool: DataType.Bool,
}


# Wrapper for prims.convert_element_type
# NOTE: Necessary to ...
#   1) convert numbertypes to the appropriate datatype,
#       the conversion depends on whether the input is a scalar or tensor
#   2) handle constants, which nvFuser will refuse to convert
def _convert_element_type_translation(fd):
    def _fn(a, dtype):
        nvfuser_dtype = dtype

        if dtypes.is_numbertype(dtype):
            if isinstance(a, nvTensor):
                tensor_dtype = torch.bool

                if dtype is int:
                    tensor_dtype = dtypes.int64
                if dtype is float:
                    tensor_dtype = dtypes.float32
                if dtype is complex:
                    tensor_dtype = dtypes.complex64

                nvfuser_dtype = _thunder_dtype_to_nvfuser_dtype_map[tensor_dtype]
            elif isinstance(a, nvNumber):
                # a is a number
                number_dtype = bool
                if dtype is int:
                    number_dtype = dtypes.int64
                if dtype is float:
                    number_dtype = dtypes.float64
                if dtype is complex:
                    number_dtype = dtypes.complex128

                nvfuser_dtype = _thunder_dtype_to_nvfuser_dtype_map[number_dtype]
            elif isinstance(a, Number):
                return dtype(a)
            else:
                raise ValueError(f"Trying to cast unknown object {a}!")

        return fd.ops.cast(a, nvfuser_dtype)

    return _fn


# NOTE: this function is needed because currently nvfuser has a different signature from torch op
#       more context: https://github.com/csarofeen/pytorch/pull/2449#issuecomment-1427491532
def _index_select_wrapper(fd):
    def _fn(a, dim, index):
        return fd.ops.index_select(a, index, dim)

    return _fn


# TODO: consider refactoring the preprocessors with a common pattern to bind or flatten/unflatten?


# NOTE: nvFuser's reshape takes args (tensor, original_shape, new_shape)
def _reshape_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
    # TODO: FIXME
    assert len(nv_kwargs) == 0

    nv_t, nv_shape = nv_args
    t, _ = sym_args

    original_shape = t.proxy.shape

    def _realize_numbers(x):
        if isinstance(x, nvNumber):
            for p, nv in variable_to_nvfuser_map.items():
                if nv is x:
                    return p.proxy.value
            raise AssertionError("Failed to find the value of nvNumber when preprocessing broadcast_in_dim()!")
        return x

    realized_shape = tuple(_realize_numbers(x) for x in nv_shape)

    return (nv_t, original_shape, realized_shape), {}


def _squeeze_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
    # TODO: FIXME
    assert len(nv_kwargs) == 0

    nv_t, nv_dims = nv_args
    t, _ = sym_args
    original_shape = t.proxy.shape

    def _realize_numbers(x):
        if isinstance(x, nvNumber):
            for p, nv in variable_to_nvfuser_map.items():
                if nv is x:
                    return p.proxy.value
            raise AssertionError("Failed to find the value of nvNumber when preprocessing broadcast_in_dim()!")
        return x

    realized_dims = tuple(_realize_numbers(x) for x in nv_dims)

    return (nv_t, original_shape, realized_dims), {}


# TODO: combine constants
# NOTE: nvFuser's elementwise operations do not accept Python numbers as arguments, so
#   this converts Python numbers to nvConstants
def _elementwise_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
    # Adds scalars as constants
    flat_args, arg_structure = tree_flatten(nv_args)
    flat_kwargs, kwarg_structure = tree_flatten(nv_kwargs)

    def _add_constant_number(x):
        if isinstance(x, Number) and not isinstance(x, NumberProxy):
            nv = fd.define_constant(x)
            return nv
        return x

    flat_args = tuple(_add_constant_number(x) for x in flat_args)
    flat_kwargs = tuple(_add_constant_number(x) for x in flat_kwargs)

    return tree_unflatten(flat_args, arg_structure), tree_unflatten(flat_kwargs, kwarg_structure)


# NOTE: nvFuser's broadcast_in_dim primitive does not accept nvScalars as arguments,
#   so this converts nvScalars to Python numbers
# TODO: rewrite this to exploit sym_args and sym_kwargs?
def _nvScalars_to_Numbers_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
    # Converts scalars to actual values
    flat_args, arg_structure = tree_flatten(nv_args)
    flat_kwargs, kwarg_structure = tree_flatten(nv_kwargs)

    def _realize_numbers(x):
        if isinstance(x, nvNumber):
            for p, nv in variable_to_nvfuser_map.items():
                if nv is x:
                    return p.proxy.value
            raise AssertionError("Failed to find the value of nvNumber when preprocessing broadcast_in_dim()!")
        return x

    flat_args = tuple(_realize_numbers(x) for x in flat_args)
    flat_kwargs = tuple(_realize_numbers(x) for x in flat_kwargs)

    return tree_unflatten(flat_args, arg_structure), tree_unflatten(flat_kwargs, kwarg_structure)


# NOTE: nvFuser's full prim requires shape to be a sequence of Python numbers, the fill value must
#   be a nvScalar (or nvConstant?), and it accepts no device argument
# NOTE: the full prim has a bug where it will segfault when shape is an empty sequence
# TODO: add an assertion on device
# TODO: revise to use sym_args?
def _full_preprocessor(fd, variable_to_nvfuser_map, sym_args, sym_kwargs, nv_args, nv_kwargs):
    (
        shape,
        fill_value,
    ) = nv_args
    dtype = nv_kwargs["dtype"]

    # FIXME: https://github.com/csarofeen/pytorch/issues/2358
    assert len(shape) > 0

    def _realize_number(x):
        if isinstance(x, nvNumber):
            for p, nv in variable_to_nvfuser_map.items():
                if nv is x:
                    return p.proxy.value
            raise AssertionError("Failed to find the value of nvNumber when preprocessing broadcast_in_dim()!")
        return x

    def _number_to_constant(x):
        if isinstance(x, Number) and not isinstance(x, NumberProxy):
            nv = fd.define_constant(x)
            return nv
        return x

    shape = tuple(_realize_number(s) for s in shape)

    return (shape, _number_to_constant(fill_value), dtype), {}


# Maps the Thunder primitives to their corresponding nvfuser operation names
# TODO: map directly to the nvfuser operations, not their names
# TODO: review the cast operation on tensors vs scalars
ops_to_nvfuser_ops_map = {
    # Data movement and transformation prims
    prims.Ops.CONVERT_ELEMENT_TYPE: _convert_element_type_translation,
    # Tensor creation prims
    prims.Ops.FULL: "full",
    # Shape prims
    prims.Ops.BROADCAST_IN_DIM: "broadcast_in_dim",
    # NOTE: "reshape" was called "view" in earlier versions of nvFuser
    prims.Ops.RESHAPE: "reshape" if use_reshape else "view",
    # TODO: can re-enable squeeze by allowing one prim to become multiple nvFuser prims
    # prims.Ops.SQUEEZE: "squeeze",
    # See https://github.com/csarofeen/pytorch/issues/2396 for slice request
    # prims.Ops.SLICE
    # NOTE: nvFuser exposes the "transpose" prim as "permute"
    prims.Ops.TRANSPOSE: "permute",
    prims.Ops.INDEX_SELECT: _index_select_wrapper,
    # Elementwise unary prims
    prims.Ops.ABS: "abs",
    prims.Ops.ACOS: "acos",
    # prims.Ops.ACOSH: "acosh",
    prims.Ops.ASIN: "asin",
    prims.Ops.ATAN: "atan",
    prims.Ops.ATANH: "atanh",
    prims.Ops.BITWISE_NOT: "bitwise_not",
    prims.Ops.CEIL: "ceil",
    prims.Ops.COS: "cos",
    prims.Ops.COSH: "cosh",
    prims.Ops.ERF: "erf",
    prims.Ops.ERFC: "erfc",
    prims.Ops.EXP: "exp",
    prims.Ops.EXPM1: "expm1",
    prims.Ops.FLOOR: "floor",
    # The isfinite translation is incorrect, see https://github.com/csarofeen/pytorch/issues/2230
    # nvFuser's isfinite returns its output in the same datatype as the input,
    #   but prims.isfinite always expects a boolean return (consistent with
    #   Python, NumPy, JAX, and PyTorch)
    prims.Ops.ISFINITE: "isfinite",
    prims.Ops.RSQRT: "rsqrt",
    prims.Ops.SIGN: "sign",
    prims.Ops.SIN: "sin",
    prims.Ops.SINH: "sinh",
    prims.Ops.SQRT: "sqrt",
    prims.Ops.TAN: "tan",
    prims.Ops.TANH: "tanh",
    prims.Ops.LOG: "log",
    prims.Ops.LOG10: "log10",
    prims.Ops.LOG1P: "log1p",
    prims.Ops.LOG2: "log2",
    prims.Ops.TRUNC: "trunc",
    # Elementwise binary prims
    prims.Ops.ADD: "add",
    prims.Ops.ATAN2: "atan2",
    prims.Ops.BITWISE_AND: "bitwise_and",
    prims.Ops.DIV: "div",
    prims.Ops.EQ: "eq",
    prims.Ops.LT: "lt",
    prims.Ops.MUL: "mul",
    prims.Ops.POW: "pow",
    prims.Ops.SUB: "sub",
    # Elementwise ternary prims
    prims.Ops.WHERE: "where",
    # Reduction prims
    prims.Ops.AMAX: "max",
    prims.Ops.SUM: "sum",
    prims.Ops.VAR: "var",
    nvOps.VAR_MEAN: "var_mean",
}

ops_to_nvfuser_preprocessors_map = {
    # Shape prims
    prims.Ops.RESHAPE: _reshape_preprocessor,
    # prims.Ops.SQUEEZE: _squeeze_preprocessor,
    prims.Ops.TRANSPOSE: _nvScalars_to_Numbers_preprocessor,
    prims.Ops.INDEX_SELECT: _nvScalars_to_Numbers_preprocessor,
    # Elementwise unary prims
    prims.Ops.ABS: _elementwise_preprocessor,
    prims.Ops.ACOS: _elementwise_preprocessor,
    prims.Ops.RSQRT: _elementwise_preprocessor,
    prims.Ops.SIGN: _elementwise_preprocessor,
    prims.Ops.SIN: _elementwise_preprocessor,
    prims.Ops.SINH: _elementwise_preprocessor,
    prims.Ops.SQRT: _elementwise_preprocessor,
    prims.Ops.TAN: _elementwise_preprocessor,
    prims.Ops.TANH: _elementwise_preprocessor,
    # prims.Ops.ACOSH:_elementwise_preprocessor,
    prims.Ops.ASIN: _elementwise_preprocessor,
    prims.Ops.ATAN: _elementwise_preprocessor,
    prims.Ops.ATANH: _elementwise_preprocessor,
    prims.Ops.BITWISE_NOT: _elementwise_preprocessor,
    prims.Ops.CEIL: _elementwise_preprocessor,
    prims.Ops.COS: _elementwise_preprocessor,
    prims.Ops.COSH: _elementwise_preprocessor,
    prims.Ops.ERF: _elementwise_preprocessor,
    prims.Ops.ERFC: _elementwise_preprocessor,
    prims.Ops.EXP: _elementwise_preprocessor,
    prims.Ops.EXPM1: _elementwise_preprocessor,
    prims.Ops.FLOOR: _elementwise_preprocessor,
    prims.Ops.TRUNC: _elementwise_preprocessor,
    # Elementwise binary prims
    prims.Ops.ADD: _elementwise_preprocessor,
    prims.Ops.ATAN2: _elementwise_preprocessor,
    prims.Ops.BITWISE_AND: _elementwise_preprocessor,
    prims.Ops.DIV: _elementwise_preprocessor,
    prims.Ops.EQ: _elementwise_preprocessor,
    prims.Ops.LT: _elementwise_preprocessor,
    prims.Ops.MUL: _elementwise_preprocessor,
    prims.Ops.POW: _elementwise_preprocessor,
    prims.Ops.SUB: _elementwise_preprocessor,
    # Elementwise ternary prims
    prims.Ops.WHERE: _elementwise_preprocessor,
    # Shape prims
    prims.Ops.BROADCAST_IN_DIM: _nvScalars_to_Numbers_preprocessor,
    # Reduction prims
    prims.Ops.AMAX: _nvScalars_to_Numbers_preprocessor,
    prims.Ops.SUM: _nvScalars_to_Numbers_preprocessor,
    prims.Ops.VAR: _nvScalars_to_Numbers_preprocessor,
    nvOps.VAR_MEAN: _nvScalars_to_Numbers_preprocessor,
    # Tensor creation prims
    prims.Ops.FULL: _full_preprocessor,
}


def _var_mean_prim_meta(a, dim, *, correction, **kwargs):
    output_dtype = a.dtype
    if utils.is_complex_dtype(output_dtype):
        output_dtype = utils.corresponding_real_dtype(output_dtype)

    var = prims.reduction_meta(a, dim, output_dtype=output_dtype)
    mean = prims.reduction_meta(a, dim, output_dtype=a.dtype)

    return (var, mean)


var_mean_prim = prims.make_prim(nvOps.VAR_MEAN, "var_mean", _var_mean_prim_meta)


def var_mean(a, dim=None, unbiased=None, keepdim=False, *, correction=None):
    correction = ttorch._set_correction(unbiased, correction)

    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    dim = ttorch._reduction_dims(a.shape, dim)

    # For complex tensors eager computes the variance as the sum of variances of
    # the real and imaginary parts
    # TODO: Creating a complex tensor from real and imaginary parts is not supported
    utils.check(
        not utils.is_complex_dtype(a.dtype),
        lambda: "Complex tensors are not supported!",
    )

    v, m = var_mean_prim(a, dim, correction=correction)

    if keepdim:
        output_shape = [a.shape[i] if i not in dim else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dim]
        v = prims.broadcast_in_dim(v, output_shape, broadcast_dims)
        m = prims.broadcast_in_dim(m, output_shape, broadcast_dims)

    return v, m


def _get_nvfuser_op(fd, op):
    nv_op = ops_to_nvfuser_ops_map[op]

    # TODO: always directly look up the appropriate callable
    if isinstance(nv_op, str):
        return getattr(fd.ops, ops_to_nvfuser_ops_map[op])

    # nv_op is a callable
    return nv_op(fd)


def _make_contiguous_strides_for(shape):
    """Returns the strides of a contiguous tensor if row_major."""
    if len(shape) == 0:
        return ()

    multiplier = 1
    strides = []
    for l in reversed(shape):
        strides.append(multiplier)
        if l != 0:
            multiplier *= l

    result = tuple(reversed(strides))

    return result


# Creates an nvFuser input for the corresponding proxy
def _add_input(fd, variable, variable_to_nvfuser_map):
    nv = None
    x = variable.proxy
    if isinstance(x, NumberProxy):
        python_type = x.python_type
        nv_dtype = _thunder_dtype_to_nvfuser_dtype_scalar_map[python_type]
        nv = fd.define_scalar(nv_dtype)
    elif isinstance(x, TensorProxy):
        nv_dtype = _thunder_dtype_to_nvfuser_dtype_map[x.dtype]
        # TODO: carefully review define tensor args
        # TODO: fix striding assumption -- currently intermediates produces from
        #   PyTorch are made contiguous so it's true
        # NOTE: there is a bug when defining a tensor with ndims!
        # nv = fd.define_tensor(ndims=len(x.shape), dtype=nv_dtype)
        strides = x.strides if x.strides is not None else _make_contiguous_strides_for(x.shape)
        nv = fd.define_tensor(sizes=x.shape, strides=strides, dtype=nv_dtype)
    else:
        raise ValueError(f"Trying to add an unknown proxy {x} as an input!")

    variable_to_nvfuser_map[variable] = nv
    return nv


# Finds or creates the nvFuser object associated with x,
#   possibly updating datastructures for proxies.
def _get_nv(x, *, fd, variable_to_nvfuser_map):
    # TODO: revise this
    #   This is here because nvFuser accepts some numbers, particularly numbers
    #   in collections, but some operations require a defined nvNumber and not
    #   a constant number. Because we're treemapping when calling this function,
    #   it can't disambiguate numbers in collections vs. a number argument.
    #   So this explicitly doesn't convert numbers to nvNumber, and that
    #   is left to preprocessing functions.
    # if isinstance(x, Number) and not isinstance(x, Proxy):
    #     return x
    if dtypes.is_dtype(x) and not dtypes.is_numbertype(x):
        return _thunder_dtype_to_nvfuser_dtype_map[x]

    if not isinstance(x, Variable):
        return x

    if x not in variable_to_nvfuser_map:
        return _add_input(fd, x, variable_to_nvfuser_map)

    return variable_to_nvfuser_map[x]


# Acquires a variable name or passes a constant by value
def _extract_name(x):
    if isinstance(x, Variable):
        return x.name

    return str(x)


def _fuse_region(inputs, outputs, symbols):
    # TODO: ensure this is true in the _fuse call
    assert len(outputs) > 0
    assert len(symbols) > 0

    variable_to_nvfuser_map = {}

    if nvfuser_version >= LooseVersion("0.0.1"):
        fd = FusionDefinition()
        fs = fd
    else:
        fs = Fusion()
        fd = FusionDefinition(fs)

    with fd:
        # Adds inputs
        for inp in inputs:
            _add_input(fd, inp, variable_to_nvfuser_map)

        # Adds symbols
        __get_nv = partial(_get_nv, fd=fd, variable_to_nvfuser_map=variable_to_nvfuser_map)
        for sym in symbols:
            nv_args = tree_map(__get_nv, sym.args)
            nv_kwargs = tree_map(__get_nv, sym.kwargs)
            nv_pre = ops_to_nvfuser_preprocessors_map.get(sym.op, None)
            if nv_pre is not None:
                # TODO: should preprocessing functions be called with the symbol's args and kwargs
                #   or the nv args and kwargs or both?
                nv_args, nv_kwargs = nv_pre(fd, variable_to_nvfuser_map, sym.args, sym.kwargs, nv_args, nv_kwargs)
            nv_op = _get_nvfuser_op(fd, sym.op)
            nv_result = nv_op(*nv_args, **nv_kwargs)

            # Associates variables to the nvFuser results
            # NOTE: it's assumed that NV operations produce results with proxies as leaves
            variables, _ = tree_flatten(sym.outputs)
            nvs, _ = tree_flatten(nv_result)
            for v, nv in zip(variables, nvs):
                if v in variable_to_nvfuser_map:
                    raise AssertionError(f"An output {v} was already in the variable map {variable_to_nvfuser_map}!")
                assert isinstance(v, Variable)
                variable_to_nvfuser_map[v] = nv

        # Adds outputs

        # TODO: refactor this class and the following dict
        # TODO: probably don't need position in nvOutput
        class nvOutput:
            def __init__(self, position, *, is_number=False):
                self.position = position
                self.is_number = is_number

        variable_to_nvOutput_map = {}
        nvfuser_output_ctr = 0
        for idx, o in enumerate(outputs):
            # Asserts that all outputs are proxies, that they are unique, and that they
            #   were produced by the above fusion
            assert isinstance(o, Variable)
            assert o in variable_to_nvfuser_map
            assert o not in variable_to_nvOutput_map
            # Validates that each output from the fusino appears only once

            # Ensures that the output is only added as a fusion output once
            # NOTE: nvFuser doesn't support scalar outputs, so this
            #   wraps them in tensors (they are unwrapped later)
            is_number = False
            if isinstance(o.proxy, NumberProxy):
                is_number = True
                dtype = _thunder_dtype_to_nvfuser_dtype_scalar_map[o.proxy.python_type]
                tensor_out = fd.ops.full((1,), variable_to_nvfuser_map[o], dtype)
                fd.add_output(tensor_out)
            else:
                fd.add_output(variable_to_nvfuser_map[o])

            nvOut = nvOutput(nvfuser_output_ctr, is_number=is_number)
            variable_to_nvOutput_map[o] = nvOut
            nvfuser_output_ctr += 1

    #
    # Builds the callable
    #
    # NOTE: the only reason the callable is built today is to handle unwrapping numbers
    #   from tensors

    # Defines utilities
    tab = "  "

    # Creates signature
    arg_str = ", ".join(tuple(_extract_name(inp) for inp in inputs))
    cstr = f"def fusion({arg_str}):"

    # Creates call to fusion
    result_str = ", ".join(tuple(_extract_name(out) for out in outputs))

    # Handles no inputs
    if len(inputs) == 0:
        cstr += f"\n{tab}{result_str}, = _fusion(())"
    else:
        cstr += f"\n{tab}{result_str}, = _fusion(({arg_str},))"

    # Converts tensors to numbers, where appropriate
    out_strs = []
    for o in outputs:
        if isinstance(o.proxy, NumberProxy):
            out_strs.append(f"{o.name}.cpu().item()")
        else:
            out_strs.append(f"{o.name}")
    out_str = ", ".join(out_strs)
    cstr += f"\n{tab}return {out_str}"

    # Creates context
    ctx = {
        "_fusion": fs.execute,
    }

    code = compile(cstr, "nvfuser.gen", mode="exec")
    exec(code, ctx)
    fusion = ctx["fusion"]

    return fusion


def lower_for_nvfuser_mapper(symbol: prims.Symbol):
    """For a given symbol, returns the nvFuser-compatible function that
    implements the symbol if possible. Otherwise, returns the original
    function.

    Args:
        symbol (prims.Symbol): The symbol to lower.

    Returns:
        Callable: The nvFuser-compatible function that implements the symbol
    """

    # If the symbol is a core primitive, then we don't need to do anything
    prim_func = getattr(prims, symbol.name, None)
    if prim_func is not None:
        return prim_func

    # If the symbol is a nvFuser primitive, then we don't need to do anything
    if symbol.op == nvOps.VAR_MEAN:
        return var_mean

    # SLICE primitive doesn't use `symbol.name`
    if symbol.op == prims.Ops.SLICE:
        return prims.slice_prim

    assert symbol.op in ttorch._torch_to_thunder_function_map, f"Unknown op {symbol.name}, {symbol.op}!"

    # All other symbols are treated as composite functions
    # We decompose them into primitives if the decomposition is fully supported
    # by nvFuser. Otherwise, we keep them as is.
    # TODO: shall we store the meta function in the symbol?
    decomposed_fn = prims.ops_to_meta_functions_map[symbol.op]
    proxy_args = tree_map(lambda x: x.proxy if isinstance(x, Variable) else x, symbol.args)
    proxy_kwargs = tree_map(lambda x: x.proxy if isinstance(x, Variable) else x, symbol.kwargs)
    trace = make_trace(lower_for_nvfuser(decomposed_fn))(*proxy_args, **proxy_kwargs)
    all_supported = all(s.op in ops_to_nvfuser_ops_map for s in trace.symbols)
    if all_supported:
        return lower_for_nvfuser(decomposed_fn)

    # When the decomposition is not supported, we use the original function
    # TODO: shall we store the original function in the symbol?
    original_fn = ttorch._torch_to_thunder_function_map[symbol.op]
    return original_fn


def lower_for_nvfuser(func):
    """Converts PyTorch functions to core Thunder primitives if they are supported by nvFuser.

    Args:
        func (Callable): A Thunder function to be transformed.
    """

    def wrapper(*args, **kwargs):
        trace = make_trace(func)(*args, **kwargs)
        return eval_trace(trace, *args, **kwargs, symbol_mapper=lower_for_nvfuser_mapper)

    return wrapper


# TODO: support NumPy arrays
# TODO: possibly support caching on the object that fusion returns
# fuse returns a function that, when called with actual PyTorch tensors and Python numbers
#   in place of the corresponding TensorProxies and NumberProxies, computes the given
#   trace.
# NOTE: the function can be reused, but it should be called with tensors that have the
#   same metadata, numbers of the same type, all conditionals on the number evaluated
#   the same as previous number inputs, and all other values constant.
def _fuse(
    trace,
    *,
    profile_info=False,
):
    # Separates the trace into parts to execute with nvFuser, and parts to execute with PyTorch
    # TODO: consider where this pass should live in the future
    # TODO: consider reordering operations cleverly
    # TODO: there are more elegant ways to express this logic; consider refactoring it

    #
    # TODO: maybe generalize is_supported to an executor
    class Region:
        def __init__(self, is_supported):
            self.symbols = []
            self.is_supported = is_supported
            self.inputs = []
            self.outputs = []
            self.fusion = None

    regions = []

    # Variables <-> producers
    variables_to_producers_map = {}
    symbols_to_produced_map = {}

    # Variables <-> consumers
    variables_to_consumers_map = {}
    symbols_to_consumed_map = {}

    symbols_to_region_map = {}

    cur_region = None

    # NOTE: this takes advantage of both symbols and the trace itself stores inputs
    #   as args and kwargs
    def _extract_input_variables(sym):
        flat_args, _ = tree_flatten(sym.args)
        flat_kwargs, _ = tree_flatten(sym.kwargs)

        return tuple(x for x in (flat_args + flat_kwargs) if isinstance(x, Variable))

    def _update_producers(variable, sym):
        # Updates variable -> producer mapping (one to one)
        assert variable not in variables_to_producers_map
        variables_to_producers_map[variable] = sym

        # Updates symbol -> producer mapping (one to many)
        if sym in symbols_to_produced_map:
            symbols_to_produced_map[sym].append(variable)
        else:
            symbols_to_produced_map[sym] = [variable]

    def _update_consumers(variable, sym):
        # Updates variable -> consumers mapping (one to many)
        if variable in variables_to_consumers_map:
            variables_to_consumers_map[variable].append(sym)
        else:
            variables_to_consumers_map[variable] = [sym]

        # Updates symbol -> consumed mapping (one to many)
        if sym in symbols_to_consumed_map:
            symbols_to_consumed_map[sym].append(variable)
        else:
            symbols_to_consumed_map[sym] = [variable]

    def _update_region(sym, cur_region):
        # NOTE: Semantically, is_supported(sym)
        region = None

        op_supported = sym.op in ops_to_nvfuser_ops_map
        if cur_region is None or op_supported != cur_region.is_supported:
            region = Region(op_supported)
            regions.append(region)
        else:
            region = cur_region

        region.symbols.append(sym)
        symbols_to_region_map[sym] = region
        return region

    # Retrace for nvFuser if possible
    proxy_args = tree_map(lambda x: x.proxy if isinstance(x, Variable) else x, trace.args)
    proxy_kwargs = tree_map(lambda x: x.proxy if isinstance(x, Variable) else x, trace.kwargs)
    func = partial(eval_trace, trace)
    trace = make_trace(lower_for_nvfuser(func), executor="nvfuser")(*proxy_args, **proxy_kwargs)

    # Processes input proxies
    # TODO: is input its own region?
    variables = _extract_input_variables(trace)
    for v in variables:
        _update_producers(v, "input")

    # Identifies regions and where proxies are produced and consumed
    for sym in trace.symbols:
        cur_region = _update_region(sym, cur_region)

        variables = _extract_input_variables(sym)
        for v in variables:
            _update_consumers(v, sym)

        flat_outputs, _ = tree_flatten(sym.outputs)
        for v in (o for o in flat_outputs if isinstance(o, Variable)):
            _update_producers(v, sym)

    # Processes outputs
    # TODO: is output its own region?
    flat_outputs, output_structure = tree_flatten(trace.outputs)
    for v in (o for o in flat_outputs if isinstance(o, Variable)):
        _update_consumers(v, "output")

    # Identifies inputs and outputs for each region, creates fusion for each region
    for region in regions:
        consumed = []
        produced = []
        for sym in region.symbols:
            # NOTE: it's possible that a symbol doesn't consume a proxy
            if sym in symbols_to_consumed_map:
                consumed.extend(symbols_to_consumed_map[sym])
            if sym in symbols_to_produced_map:
                produced.extend(symbols_to_produced_map[sym])
        consumed = OrderedSet(consumed)
        produced = OrderedSet(produced)

        # A proxy that's consumed but not produced in the region is an input
        # TODO: consider ordering inputs in some sensible way
        region.inputs = consumed - produced

        # A proxy that's produced in the region and consumed in another region is an output
        outputs = []
        for p in produced:
            consumers = variables_to_consumers_map.get(p, ())
            for c in consumers:
                if c == "output" or symbols_to_region_map[c] is not region:
                    region.outputs.append(p)
                    break

        # Short-circuits if the region outputs nothing
        # NOTE: because regions are functional, this means the region does nothing
        if len(region.outputs) == 0:
            region.fusion = None
        elif region.is_supported:
            region.fusion = _fuse_region(region.inputs, region.outputs, region.symbols)
        else:
            # CASE: not region.is_supported (currently using PyTorch to run)
            region.fusion = _fuse_torch_region(region.inputs, region.outputs, region.symbols)

    #
    # Creates the callable connecting the fusions
    #

    # Common utils
    tab = "  "

    # Creates the signature
    cstr = f"def fusion(*args, **kwargs):"

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

    # Calls fusion(s)
    cstr += f"\n{tab}# Invokes fusion(s)"

    for idx, region in enumerate(regions):
        # Skips regions that do nothing
        if region.fusion is None:
            continue

        arg_str = ", ".join(tuple(_extract_name(inp) for inp in region.inputs))
        result_str = ", ".join(tuple(_extract_name(out) for out in region.outputs))
        cstr += f"\n{tab}{result_str} = _fusion{idx}({arg_str})"

    # Constructs return statement
    output_str = ", ".join(tuple(_extract_name(out) for out in flat_outputs))
    cstr += f"\n{tab}# Assembles output"
    cstr += f"\n{tab}return tree_unflatten(({output_str},), output_structure)"

    # Creates context
    ctx = {
        "tree_flatten": tree_flatten,
        "tree_unflatten": tree_unflatten,
        "output_structure": output_structure,
    }

    for idx, region in enumerate(regions):
        ctx[f"_fusion{idx}"] = region.fusion

    # Compiles the function
    code = compile(cstr, "nvfuser.gen", mode="exec")
    exec(code, ctx)
    fusion = ctx["fusion"]

    if profile_info:
        return fusion, regions

    return fusion


class nvFuserCtx:
    def __init__(self):
        pass

    def intercept(self, op):
        """"""

        # TODO: update match to not be on strings
        if op == "torch.var_mean":
            return var_mean

        return None

    def fuse(
        self,
        trace,
        *,
        profile_info=False,
    ):
        return _fuse(trace, profile_info=profile_info)
