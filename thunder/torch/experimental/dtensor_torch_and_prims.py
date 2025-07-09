from functools import partial
from collections.abc import Callable

from thunder.torch import torchsymbol, TensorLike, register_function
import thunder.torch as ltorch
from thunder.core.pytree import tree_flatten
from thunder import clang
from thunder.torch.experimental.dtensor_utils import run_with_fake_tensor
from thunder.torch.experimental.dtensor_proxy import DTensorProxy, create_dtensor_proxy_from_proxies
from thunder.torch.langctx import register_method

from thunder.core.proxies import TensorProxy, AnyProxy
from thunder.core.transforms import (
    register_grad,
    put_grads,
    get_grad,
    put_grad
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


def dtensor_linear_meta(a, w, bias):
    output = run_with_fake_tensor(torch.nn.functional.linear, a, w, bias)
    local_tensor_proxy = TensorProxy(like=a.local_tensor)
    spec = output._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return create_dtensor_proxy_from_proxies(local_tensor_proxy, spec_proxy, False)


dtensor_linear_prim = make_prim("dtensor_linear_prim", "dtensor_linear_prim", meta=dtensor_linear_meta)

dtensor_linear_prim_impl = pytorchex.register_operator("dtensor_linear_prim", like=dtensor_linear_prim, fn=torch.nn.functional.linear)

pytorchex.register_implementation(dtensor_linear_prim, dtensor_linear_prim_impl)


# def _dtensor_linear_prim_grad(a: TensorLike, w: TensorLike, bias: None | TensorLike) -> TensorLike:
#     fwd = dtensor_linear_prim(a, w, bias)

#     g = get_grad(fwd)

#     first_dim = -2
#     grad_a = ltorch.matmul(g.reshape(-1, g.shape[-1]), w).reshape(a.shape)

#     grad_w: TensorLike
#     if a.ndim == 1:
#         grad_w = ltorch.matmul(g.unsqueeze(first_dim).mT, a.unsqueeze(first_dim))
#     else:
#         grad_w = ltorch.matmul(g.reshape(-1, g.shape[-1]).mT, a.reshape(-1, a.shape[-1]))

#     put_grads((a, w), (grad_a, grad_w))

#     if bias is not None:
#         if g.ndim > 1:
#             grad_bias = ltorch.sum(g, tuple(range(g.ndim - 1)))
#         else:
#             grad_bias = g
#         put_grad(bias, grad_bias)

#     return fwd


# register_grad(dtensor_linear_prim, _dtensor_linear_prim_grad)


@dtensor_torchsymbol(torch.nn.functional.linear, id="dtensor.torch.nn.functional.linear")
def dtensor_linear(a: TensorLike, w: TensorLike, bias: None | TensorLike = None) -> TensorLike:
    return dtensor_linear_prim(a, w, bias)

from typing import Optional
from torch.distributed.tensor import DTensor

def dtensor_from_local_meta(x, mesh, placements, *, run_check: bool = False, shape: Optional[torch.Size] = None, stride: Optional[tuple[int, ...]] = None):
    res = run_with_fake_tensor(DTensor.from_local, x, mesh, placements, run_check=run_check, shape=shape, stride=stride)
    from thunder.torch.experimental.dtensor_proxy import proxify_dtensor
    res = proxify_dtensor(res)
    return res

# def dtensor_from_local_fn(x, mesh, placements, *, run_check: bool = False, shape: Optional[torch.Size] = None, stride: Optional[tuple[int, ...]] = None):
#     return DTensor.from_local(x, mesh, placements, run_check=run_check, shape=shape, stride=stride)

dtensor_from_local_prim = make_prim("dtensor_from_local", "dtensor_from_local", meta=dtensor_from_local_meta)

dtensor_from_local_prim_impl = pytorchex.register_operator("dtensor_from_local", like=dtensor_from_local_prim, fn=DTensor.from_local)

pytorchex.register_implementation(dtensor_from_local_prim, dtensor_from_local_prim_impl)

def dtensor_redistribute_meta(dtensor, device_mesh: "Optional[DeviceMesh]" = None, placements: "Optional[Sequence[Placement]]" = None, *, async_op: bool = False) -> "DTensor":
    res = run_with_fake_tensor(DTensor.redistribute, dtensor, device_mesh, placements, async_op=async_op)
    from thunder.torch.experimental.dtensor_proxy import proxify_dtensor
    res = proxify_dtensor(res)
    return res

dtensor_redistribute_prim = make_prim("dtensor_redistribute", "dtensor_redistribute", meta=dtensor_redistribute_meta)

dtensor_redistribute_prim_impl = pytorchex.register_operator("dtensor_redistribute", like=dtensor_redistribute_prim, fn=DTensor.redistribute)

pytorchex.register_implementation(dtensor_redistribute_prim, dtensor_redistribute_prim_impl)

def dtensor_to_local_meta(dtensor, *, grad_placements: "Optional[Sequence[Placement]]" = None):
    res = run_with_fake_tensor(DTensor.to_local, dtensor, grad_placements=grad_placements)
    from thunder.core.proxies import proxy
    res = proxy(res)
    return res

dtensor_to_local_prim = make_prim("dtensor_to_local", "dtensor_to_local", meta=dtensor_to_local_meta)

dtensor_to_local_prim_impl = pytorchex.register_operator("dtensor_to_local", like=dtensor_to_local_prim, fn=DTensor.to_local)

pytorchex.register_implementation(dtensor_to_local_prim, dtensor_to_local_prim_impl)


def register_dtensor_torch_and_prims():
    register_function_for_dtensor(torch.mul, ltorch.mul, dtensor_mul, is_method=True)
    register_function_for_dtensor(torch.nn.functional.linear, ltorch.linear, dtensor_linear, is_method=False)
