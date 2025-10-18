from functools import partial
from collections.abc import Callable
from enum import auto, Enum
from collections.abc import Sequence
from looseversion import LooseVersion

from thunder.torch import torchsymbol, TensorLike, register_function
import thunder.torch as ltorch
from thunder.core.pytree import tree_flatten
from thunder import clang
from thunder.clang.utils import (
    create_maybe_convert_to_dtype_with_prim,
    _elementwise_unary_wrapper,
    maybe_broadcast_impl,
    expand_impl,
)
from thunder.torch.experimental.dtensor_utils import run_with_fake_tensor
from thunder.torch.experimental.dtensor_proxy import DTensorProxy, create_dtensor_proxy_from_proxies
from thunder.torch.langctx import register_method
import thunder.core.dtypes as dtypes

from thunder.core.proxies import TensorProxy, AnyProxy
from thunder.core.transforms import (
    register_grad,
    put_grad,
    put_grads,
    get_grad,
)
from thunder.executors.torchex import ex as pytorchex
from thunder.executors.pythonex import ex as pythonex
from thunder.core.prims import make_prim, OpTags
from thunder.core import prims
from thunder.core import baseutils
from thunder.core import utils

import torch


class DTensorPrimIDs(Enum):
    # DTensor-specific primitives
    CHECK_DTENSOR_SPEC_REPR = auto()
    ADD = auto()
    MUL = auto()
    RESHAPE = auto()
    TRANSPOSE = auto()
    CONVERT_ELEMENT_TYPE = auto()
    BROADCAST_IN_DIM = auto()
    _GROUPED_MM = auto()
    EXP = auto()
    LINEAR = auto()
    NEG = auto()
    RECIPROCAL = auto()


dtensor_torchsymbol = partial(torchsymbol, allow_tensor_subclass_proxy=True)


def dispatch_to_impl(single_device_symbol, dtensor_symbol):
    def wrapper(*args, **kwargs):
        filter_tensor_proxies = list(filter(lambda t: isinstance(t, TensorProxy), tree_flatten((args, kwargs))[0]))
        # number only variant of the operator.
        if not filter_tensor_proxies:
            return single_device_symbol(*args, **kwargs)

        dtensor_tensor_proxies = map(lambda t: isinstance(t, DTensorProxy), filter_tensor_proxies)
        if all(dtensor_tensor_proxies):
            return dtensor_symbol(*args, **kwargs)
        else:
            return single_device_symbol(*args, **kwargs)

    return wrapper


def register_function_for_dtensor(torch_fn, single_device_symbol, dtensor_symbol, is_method=False):
    register_function(torch_fn, dispatch_to_impl(single_device_symbol, dtensor_symbol))

    if is_method:
        method_name: str = torch_fn.__name__
        torch_method: None | Callable = getattr(torch.Tensor, method_name, None)
        register_method_for_dtensor(torch_method, single_device_symbol, dtensor_symbol)


def register_method_for_dtensor(torch_fn, single_device_symbol, dtensor_symbol):
    method_wrapper = dispatch_to_impl(single_device_symbol, dtensor_symbol)
    register_function(torch_fn, method_wrapper)
    register_method(torch_fn.__name__, method_wrapper)


def _check_dtensor_spec_repr_meta(s: AnyProxy, value: str) -> None:
    baseutils.check_type(s, AnyProxy)


check_dtensor_spec_repr = make_prim(
    DTensorPrimIDs.CHECK_DTENSOR_SPEC_REPR,
    "check_dtensor_spec_repr",
    meta=_check_dtensor_spec_repr_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_dtensor_spec_repr(s: object, value: str) -> None:
    utils.check(repr(s) == value, lambda: f"Expected '{s} to be equal to '{value}")


_check_python_repr_impl = pythonex.register_operator(
    "check_python_repr", like=check_dtensor_spec_repr, fn=_check_dtensor_spec_repr
)

pythonex.register_implementation(check_dtensor_spec_repr, _check_python_repr_impl)


def handle_check_dtensor_spec_in_prologue(prim, prologue_trace, args) -> bool:
    if prim == check_dtensor_spec_repr:
        # How does torch.compile guard for this?
        a = args[0]
        o = AnyProxy(None, prefix="dtensor_spec")
        bsym = prims.unpack_attr.bind(a, "_spec", output=o)
        prologue_trace.bound_symbols.append(bsym)
        check_dtensor_spec_repr(o, repr(args[1]))

        # Also adds metadata check for _local_tensor
        t = TensorProxy(like=a.local_tensor, requires_grad=a.local_tensor.requires_grad)
        bsym = prims.unpack_attr.bind(a, "_local_tensor", output=t)
        prologue_trace.bound_symbols.append(bsym)
        clang.check_tensor_shape_and_metadata(t)
        return True

    return False


def dtensor_reshape_meta(a, shape):
    output = run_with_fake_tensor(torch.reshape, a, shape)
    local_tensor_proxy = TensorProxy(
        like=a.local_tensor, shape=output._local_tensor.shape, dtype=dtypes.to_dtype(output._local_tensor.dtype)
    )
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_reshape_prim = make_prim(DTensorPrimIDs.RESHAPE, "dtensor_reshape_prim", meta=dtensor_reshape_meta)

dtensor_reshape_prim_impl = pytorchex.register_operator(
    "dtensor_reshape_prim", like=dtensor_reshape_prim, fn=torch.reshape
)

pytorchex.register_implementation(dtensor_reshape_prim, dtensor_reshape_prim_impl)


def _dtensor_reshape_prim_grad(a: TensorLike, shape: tuple[int, ...]) -> TensorLike:
    fwd = dtensor_reshape_prim(a, shape)

    g = get_grad(fwd)
    a_grad = dtensor_reshape_prim(g, a.shape)
    put_grads((a,), (a_grad,))

    return fwd


register_grad(dtensor_reshape_prim, _dtensor_reshape_prim_grad)


@dtensor_torchsymbol(torch.reshape, id="dtensor.torch.reshape")
def dtensor_reshape(a: TensorLike, shape: tuple[int, ...]) -> TensorLike:
    return dtensor_reshape_prim(a, shape)


def dtensor_transpose_meta(a, permutation):
    output = run_with_fake_tensor(torch.permute, a, permutation)
    local_tensor_proxy = TensorProxy(
        like=a.local_tensor, shape=output._local_tensor.shape, dtype=dtypes.to_dtype(output._local_tensor.dtype)
    )
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_transpose_prim = make_prim(DTensorPrimIDs.TRANSPOSE, "dtensor_transpose_prim", meta=dtensor_transpose_meta)

dtensor_transpose_prim_impl = pytorchex.register_operator(
    "dtensor_transpose_prim", like=dtensor_transpose_prim, fn=torch.permute
)

pytorchex.register_implementation(dtensor_transpose_prim, dtensor_transpose_prim_impl)


def _dtensor_transpose_prim_grad(a: TensorLike, permutation) -> TensorLike:
    fwd = dtensor_transpose_prim(a, permutation)

    g = get_grad(fwd)
    undo = sorted(range(len(permutation)), key=permutation.__getitem__)
    a_grad = dtensor_transpose_prim(g, undo)
    put_grads((a,), (a_grad,))

    return fwd


register_grad(dtensor_transpose_prim, _dtensor_transpose_prim_grad)


@dtensor_torchsymbol(torch.transpose, id="dtensor.torch.transpose")
def dtensor_transpose(a: TensorLike, dim0: int, dim1: int) -> TensorLike:
    dim0, dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))

    permutation = list(range(a.ndim))
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    return dtensor_transpose_prim(a, permutation)


def dtensor_convert_element_type_meta(a, dtype):
    tdtype = ltorch.to_torch_dtype(dtype)
    output = run_with_fake_tensor(lambda x, dt: x.to(dt), a, tdtype)
    local_tensor_proxy = TensorProxy(like=a.local_tensor, shape=output._local_tensor.shape, dtype=dtype)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_convert_element_type_prim = make_prim(
    DTensorPrimIDs.CONVERT_ELEMENT_TYPE,
    "dtensor_convert_element_type_prim",
    meta=dtensor_convert_element_type_meta,
)

dtensor_convert_element_type_prim_impl = pytorchex.register_operator(
    "dtensor_convert_element_type_prim",
    like=dtensor_convert_element_type_prim,
    fn=lambda x, dt: x.to(ltorch.to_torch_dtype(dt)),
)

pytorchex.register_implementation(dtensor_convert_element_type_prim, dtensor_convert_element_type_prim_impl)


def _dtensor_convert_element_type_prim_grad(a: TensorLike, dtype) -> TensorLike:
    fwd = dtensor_convert_element_type_prim(a, dtype)

    g = get_grad(fwd)
    g_converted = dtensor_convert_element_type_prim(g, a.dtype)
    put_grad(a, g_converted)

    return fwd


register_grad(dtensor_convert_element_type_prim, _dtensor_convert_element_type_prim_grad)


def dtensor_broadcast_in_dim_meta(a, shape, broadcast_dimensions):
    output = run_with_fake_tensor(lambda x, s, bd: x.broadcast_to(s), a, shape, broadcast_dimensions)
    local_tensor_proxy = TensorProxy(like=a.local_tensor, shape=output._local_tensor.shape)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, a.requires_grad)


# TODO: Add gradient for `dtensor_broadcast_in_dim_prim` which requires `sum`.
dtensor_broadcast_in_dim_prim = make_prim(
    DTensorPrimIDs.BROADCAST_IN_DIM, "dtensor_broadcast_in_dim_prim", meta=dtensor_broadcast_in_dim_meta
)

dtensor_broadcast_in_dim_prim_impl = pytorchex.register_operator(
    "dtensor_broadcast_in_dim_prim", like=dtensor_broadcast_in_dim_prim, fn=lambda x, s, bd: x.broadcast_to(s)
)

pytorchex.register_implementation(dtensor_broadcast_in_dim_prim, dtensor_broadcast_in_dim_prim_impl)


maybe_convert_to_dtype = create_maybe_convert_to_dtype_with_prim(dtensor_convert_element_type_prim)
_elementwise_unary_wrapper = partial(_elementwise_unary_wrapper, dtype_conversion_fn=maybe_convert_to_dtype)


def dtensor_linear_meta(a, w, bias):
    output = run_with_fake_tensor(torch.nn.functional.linear, a, w, bias)
    local_tensor_proxy = TensorProxy(like=a.local_tensor)
    local_tensor_proxy = TensorProxy(
        like=a.local_tensor, shape=output._local_tensor.shape, dtype=dtypes.to_dtype(output._local_tensor.dtype)
    )
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


# TODO: Add grad rule once the prims used for linear grad-rule are available.
dtensor_linear_prim = make_prim(DTensorPrimIDs.LINEAR, "dtensor_linear_prim", meta=dtensor_linear_meta)

dtensor_linear_prim_impl = pytorchex.register_operator(
    "dtensor_linear_prim", like=dtensor_linear_prim, fn=torch.nn.functional.linear
)

pytorchex.register_implementation(dtensor_linear_prim, dtensor_linear_prim_impl)


@dtensor_torchsymbol(torch.nn.functional.linear, id="dtensor.torch.nn.functional.linear")
def dtensor_linear(a: TensorLike, w: TensorLike, bias: None | TensorLike = None) -> TensorLike:
    return dtensor_linear_prim(a, w, bias)


def dtensor_exp_meta(a):
    output = run_with_fake_tensor(torch.exp, a)
    local_tensor_proxy = TensorProxy(like=a.local_tensor)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_exp_prim = make_prim(DTensorPrimIDs.EXP, "dtensor_exp_prim", meta=dtensor_exp_meta)

dtensor_exp_prim_impl = pytorchex.register_operator("dtensor_exp_prim", like=dtensor_exp_prim, fn=torch.exp)

pytorchex.register_implementation(dtensor_exp_prim, dtensor_exp_prim_impl)


def _dtensor_exp_prim_grad(a: TensorLike) -> TensorLike:
    fwd = dtensor_exp_prim(a)

    g = get_grad(fwd)
    a_grad = g * fwd
    put_grad(a, a_grad)

    return fwd


register_grad(dtensor_exp_prim, _dtensor_exp_prim_grad)


@dtensor_torchsymbol(torch.exp, id="dtensor.torch.exp")
def dtensor_exp(a: TensorLike) -> TensorLike:
    return _elementwise_unary_wrapper(
        a,
        prim=dtensor_exp_prim,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )


def dtensor_neg_meta(a):
    output = run_with_fake_tensor(torch.neg, a)
    local_tensor_proxy = TensorProxy(like=a.local_tensor)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_neg_prim = make_prim(DTensorPrimIDs.NEG, "dtensor_neg_prim", meta=dtensor_neg_meta)

dtensor_neg_prim_impl = pytorchex.register_operator("dtensor_neg_prim", like=dtensor_neg_prim, fn=torch.neg)

pytorchex.register_implementation(dtensor_neg_prim, dtensor_neg_prim_impl)


@dtensor_torchsymbol(torch.neg, id="dtensor.torch.neg")
def dtensor_neg(a: TensorLike) -> TensorLike:
    return _elementwise_unary_wrapper(
        a,
        prim=dtensor_neg_prim,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


def dtensor_reciprocal_meta(a):
    output = run_with_fake_tensor(torch.reciprocal, a)
    local_tensor_proxy = TensorProxy(like=a.local_tensor)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_reciprocal_prim = make_prim(DTensorPrimIDs.RECIPROCAL, "dtensor_reciprocal_prim", meta=dtensor_reciprocal_meta)

dtensor_reciprocal_prim_impl = pytorchex.register_operator(
    "dtensor_reciprocal_prim", like=dtensor_reciprocal_prim, fn=torch.reciprocal
)

pytorchex.register_implementation(dtensor_reciprocal_prim, dtensor_reciprocal_prim_impl)


@dtensor_torchsymbol(torch.reciprocal, id="dtensor.torch.reciprocal")
def dtensor_reciprocal(a: TensorLike) -> TensorLike:
    return _elementwise_unary_wrapper(
        a,
        prim=dtensor_reciprocal_prim,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )


if torch.distributed.is_available():
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Placement, DeviceMesh

    def dtensor_from_local_meta(
        x,
        mesh,
        placements,
        *,
        run_check: bool = False,
        shape: torch.Size | None = None,
        stride: tuple[int, ...] | None = None,
    ):
        res = run_with_fake_tensor(
            DTensor.from_local, x, mesh, placements, run_check=run_check, shape=shape, stride=stride
        )
        from thunder.torch.experimental.dtensor_proxy import proxify_dtensor

        res = proxify_dtensor(res)
        return res

    dtensor_from_local_prim = make_prim("dtensor_from_local", "dtensor_from_local", meta=dtensor_from_local_meta)

    dtensor_from_local_prim_impl = pytorchex.register_operator(
        "dtensor_from_local", like=dtensor_from_local_prim, fn=DTensor.from_local
    )

    pytorchex.register_implementation(dtensor_from_local_prim, dtensor_from_local_prim_impl)

    @dtensor_torchsymbol(DTensor.from_local, id="dtensor.torch.from_local")
    def dtensor_from_local(
        x,
        mesh,
        placements,
        *,
        run_check: bool = False,
        shape: torch.Size | None = None,
        stride: tuple[int, ...] | None = None,
    ) -> DTensorProxy | None:
        return dtensor_from_local_prim(x, mesh, placements, run_check=run_check, shape=shape, stride=stride)

    def dtensor_redistribute_meta(
        dtensor,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
    ) -> DTensorProxy | None:
        res = run_with_fake_tensor(DTensor.redistribute, dtensor, device_mesh, placements, async_op=async_op)
        from thunder.torch.experimental.dtensor_proxy import proxify_dtensor

        res = proxify_dtensor(res)
        return res

    dtensor_redistribute_prim = make_prim(
        "dtensor_redistribute", "dtensor_redistribute", meta=dtensor_redistribute_meta
    )

    dtensor_redistribute_prim_impl = pytorchex.register_operator(
        "dtensor_redistribute", like=dtensor_redistribute_prim, fn=DTensor.redistribute
    )

    @dtensor_torchsymbol(DTensor.redistribute, id="dtensor.torch.redistribute")
    def dtensor_redistribute(
        dtensor,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
    ) -> DTensorProxy | None:
        return dtensor_redistribute_prim(dtensor, device_mesh, placements, async_op=async_op)

    pytorchex.register_implementation(dtensor_redistribute_prim, dtensor_redistribute_prim_impl)

    def dtensor_to_local_meta(dtensor, *, grad_placements: Sequence[Placement] | None = None):
        res = run_with_fake_tensor(DTensor.to_local, dtensor, grad_placements=grad_placements)
        from thunder.core.proxies import proxy

        res = proxy(res)
        return res

    dtensor_to_local_prim = make_prim("dtensor_to_local", "dtensor_to_local", meta=dtensor_to_local_meta)

    dtensor_to_local_prim_impl = pytorchex.register_operator(
        "dtensor_to_local", like=dtensor_to_local_prim, fn=DTensor.to_local
    )

    pytorchex.register_implementation(dtensor_to_local_prim, dtensor_to_local_prim_impl)

    @dtensor_torchsymbol(DTensor.to_local, id="dtensor.torch.to_local")
    def dtensor_to_local(dtensor, *, grad_placements: Sequence[Placement] | None = None) -> DTensorProxy | None:
        return dtensor_to_local_prim(dtensor, grad_placements=grad_placements)


expand = partial(expand_impl, broadcast_prim=dtensor_broadcast_in_dim_prim)
maybe_broadcast = partial(maybe_broadcast_impl, expand_fn=expand)


def _elementwise_binary_wrapper(a, b, *, prim, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(a, b, type_promotion_kind=type_promotion_kind)

    a, b = maybe_broadcast(a, b)
    a, b = maybe_convert_to_dtype(a, computation_dtype), maybe_convert_to_dtype(b, computation_dtype)

    result = prim(a, b)
    result = maybe_convert_to_dtype(result, result_dtype)

    return result


def dtensor_add_meta(a, b):
    output = run_with_fake_tensor(torch.add, a, b)
    local_tensor_proxy = TensorProxy(like=a.local_tensor)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_add_prim = make_prim(DTensorPrimIDs.ADD, "dtensor_add_prim", meta=dtensor_add_meta)

dtensor_add_prim_impl = pytorchex.register_operator("dtensor_add_prim", like=dtensor_add_prim, fn=torch.add)

pytorchex.register_implementation(dtensor_add_prim, dtensor_add_prim_impl)


@dtensor_torchsymbol(torch.add, id="dtensor.torch.add")
def dtensor_add(a: TensorLike, b: TensorLike) -> TensorLike:
    return _elementwise_binary_wrapper(
        a,
        b,
        prim=dtensor_add_prim,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


def dtensor_mul_meta(a, b):
    output = run_with_fake_tensor(torch.mul, a, b)
    local_tensor_proxy = TensorProxy(like=a.local_tensor)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_mul_prim = make_prim(DTensorPrimIDs.MUL, "dtensor_mul_prim", meta=dtensor_mul_meta)

dtensor_mul_prim_impl = pytorchex.register_operator("dtensor_mul_prim", like=dtensor_mul_prim, fn=torch.mul)

pytorchex.register_implementation(dtensor_mul_prim, dtensor_mul_prim_impl)


def _dtensor_mul_prim_grad(a: TensorLike, b: TensorLike) -> TensorLike:
    fwd = dtensor_mul_prim(a, b)

    g = get_grad(fwd)
    a_grad = dtensor_mul_prim(b, g)
    b_grad = dtensor_mul_prim(a, g)
    put_grads((a, b), (a_grad, b_grad))

    return fwd


register_grad(dtensor_mul_prim, _dtensor_mul_prim_grad)


@dtensor_torchsymbol(torch.mul, id="dtensor.torch.mul")
def dtensor_mul(a: TensorLike, b: TensorLike) -> TensorLike:
    return _elementwise_binary_wrapper(
        a,
        b,
        prim=dtensor_mul_prim,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )


if LooseVersion(torch.__version__) >= "2.8":

    def dtensor_grouped_mm_meta(a, b, offsets):
        output = run_with_fake_tensor(torch._grouped_mm, a, b, offsets)
        local_tensor_proxy = TensorProxy(
            like=a.local_tensor, dtype=dtypes.to_dtype(output._local_tensor.dtype), shape=output._local_tensor.shape
        )
        spec = output._spec
        spec_proxy = AnyProxy(spec, history=a.history)
        return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)

    dtensor_grouped_mm_prim = make_prim(
        DTensorPrimIDs._GROUPED_MM, "dtensor_grouped_mm_prim", meta=dtensor_grouped_mm_meta
    )

    dtensor_grouped_mm_prim_impl = pytorchex.register_operator(
        "dtensor_grouped_mm_prim", like=dtensor_grouped_mm_prim, fn=torch._grouped_mm
    )

    pytorchex.register_implementation(dtensor_grouped_mm_prim, dtensor_grouped_mm_prim_impl)

    @dtensor_torchsymbol(torch._grouped_mm, id="dtensor.torch._grouped_mm")
    def dtensor_grouped_mm(a: TensorLike, b: TensorLike, offsets: TensorLike, *, bias=None, dtype=None) -> TensorLike:
        assert bias is None, "bias is not supported"
        assert dtype is None, "dtype is not supported"
        return dtensor_grouped_mm_prim(a, b, offsets)


# NOTE: Currently only as a helper.
# TODO: Add this as a torch symbol.
def _dtensor_sigmoid(x):
    computation_dtype, result_dtype = utils.elementwise_type_promotion(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )
    x = dtensor_convert_element_type_prim(x, computation_dtype)
    result = dtensor_reciprocal(dtensor_add(dtensor_exp(-x), 1.0))
    return dtensor_convert_element_type_prim(result, result_dtype)


@dtensor_torchsymbol(torch.nn.functional.silu, id="dtensor.torch.nn.functional.silu")
def dtensor_silu(a: TensorLike, inplace: bool = False) -> TensorLike:
    assert not inplace, "inplace is not supported"
    return a * _dtensor_sigmoid(a)


def register_dtensor_torch_and_prims():
    register_function_for_dtensor(torch.add, ltorch.add, dtensor_add, is_method=True)
    register_function_for_dtensor(torch.mul, ltorch.mul, dtensor_mul, is_method=True)
    register_function_for_dtensor(torch.reshape, ltorch.reshape, dtensor_reshape, is_method=True)
    register_function_for_dtensor(torch.transpose, ltorch.transpose, dtensor_transpose, is_method=True)
    register_function_for_dtensor(torch.nn.functional.linear, ltorch.linear, dtensor_linear, is_method=False)
    register_function_for_dtensor(torch.exp, ltorch.exp, dtensor_exp, is_method=True)
    register_function_for_dtensor(torch.neg, ltorch.neg, dtensor_neg, is_method=True)
    register_function_for_dtensor(torch.reciprocal, ltorch.reciprocal, dtensor_reciprocal, is_method=True)
    register_function_for_dtensor(torch.nn.functional.silu, ltorch.silu, dtensor_silu, is_method=False)
    if LooseVersion(torch.__version__) >= "2.8":
        register_function_for_dtensor(torch._grouped_mm, ltorch._grouped_mm, dtensor_grouped_mm, is_method=False)
