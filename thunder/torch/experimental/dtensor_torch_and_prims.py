from functools import partial
from collections.abc import Callable

from thunder.torch import torchsymbol, TensorLike, register_function
import thunder.torch as ltorch
from thunder.core.pytree import tree_flatten
from thunder import clang
from thunder.torch.experimental.dtensor_utils import run_with_fake_tensor
from thunder.torch.experimental.dtensor_proxy import DTensorProxy, create_dtensor_proxy_from_proxies
from thunder.torch.langctx import register_method
import thunder.core.dtypes as dtypes

from thunder.core.proxies import TensorProxy, AnyProxy
from thunder.core.transforms import (
    register_grad,
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
    "check_dtensor_spec_repr",
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


def dtensor_mul_meta(a, b):
    output = run_with_fake_tensor(torch.mul, a, b)
    local_tensor_proxy = TensorProxy(like=a.local_tensor)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_mul_prim = make_prim("dtensor_mul_prim", "dtensor_mul_prim", meta=dtensor_mul_meta)

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
    return dtensor_mul_prim(a, b)


def dtensor_reshape_meta(a, shape):
    output = run_with_fake_tensor(torch.reshape, a, shape)
    local_tensor_proxy = TensorProxy(
        like=a.local_tensor, shape=output._local_tensor.shape, dtype=dtypes.to_dtype(output._local_tensor.dtype)
    )
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_reshape_prim = make_prim("dtensor_reshape_prim", "dtensor_reshape_prim", meta=dtensor_reshape_meta)

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


def register_dtensor_torch_and_prims():
    register_function_for_dtensor(torch.mul, ltorch.mul, dtensor_mul, is_method=True)
    register_function_for_dtensor(torch.reshape, ltorch.reshape, dtensor_reshape, is_method=True)
