from functools import partial
from collections.abc import Callable

from thunder.torch import torchsymbol, TensorLike, register_function, TensorProxy
import thunder.torch as ltorch
from thunder.core.pytree import tree_map, tree_flatten, tree_unflatten
from thunder import clang
from thunder.torch.experimental.dtensor_aten_tracing_utils import trace_torch_op_to_aten_ops
from thunder.torch.experimental.dtensor_proxy import DTensorProxy
from thunder.torch.langctx import register_method

import torch

dtensor_torchsymbol = partial(torchsymbol, allow_only_tensorproxy=False)


def dispatch_to_impl(single_device_symbol, dtensor_symbol):

    def wrapper(*args, **kwargs):
        filter_tensor_proxies = list(filter(lambda t: isinstance(t, TensorProxy), tree_flatten((args, kwargs))[0]))
        # number only variant of the operator.
        if filter_tensor_proxies == []:
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


@dtensor_torchsymbol(torch.ops.aten.mul.Tensor, id="aten.mul.Tensor")
def aten_mul(a, b):
    return clang.mul(a, b)


@dtensor_torchsymbol(torch.mul, id="dtensor.torch.mul")
def dtensor_mul(a: TensorLike, b: TensorLike) -> TensorLike:
    return trace_torch_op_to_aten_ops(ltorch.mul, a, b)


@dtensor_torchsymbol(torch.ops.aten.add.Tensor, id="aten.add.Tensor")
def aten_add(a, b, alpha=1):
    if isinstance(alpha, TensorProxy) or alpha != 1:
        b = b * alpha

    return clang.add(a, b)


@dtensor_torchsymbol(torch.add, id="dtensor.torch.add")
def dtensor_add(a: TensorLike, b: TensorLike, alpha=1) -> TensorLike:
    return trace_torch_op_to_aten_ops(ltorch.add, a, b, alpha=alpha)


def register_dtensor_and_aten_function():
    register_function_for_dtensor(torch.add, ltorch.add, dtensor_add, is_method=True)
    register_function_for_dtensor(torch.mul, ltorch.mul, dtensor_mul, is_method=True)
