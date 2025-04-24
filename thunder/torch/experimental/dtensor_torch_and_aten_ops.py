from functools import partial
from collections.abc import Callable

from thunder.torch import torchsymbol, TensorLike, register_function, TensorProxy
import thunder.torch as ltorch
from thunder.core.pytree import tree_map, tree_flatten, tree_unflatten
from thunder import clang
from thunder.torch.experimental.dtensor_aten_tracing_utils import trace_torch_op_to_aten_ops, get_fx_graph_and_output
from thunder.torch.experimental.dtensor_proxy import DTensorProxy
from thunder.torch.langctx import register_method
import thunder.core.utils as utils
from thunder.core.prims import make_prim, _make_elementwise_binary_prim

from thunder.core.proxies import DistParallelType, FutureTensorProxy, pytype, TensorProxy, AnyProxy
from thunder.core.transforms import register_augmented_forward, register_backward
from thunder.distributed import get_skip_data_parallel_grad_sync
from thunder.executors.torchex import ex as pytorchex
from thunder.executors.pythonex import ex as pythonex

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._guards import TracingContext, tracing

dtensor_torchsymbol = partial(torchsymbol, allow_tensor_subclass_proxy=True)


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


def dtensor_mul_meta(a, b):
    with tracing(TracingContext(FakeTensorMode())):
        _, output = get_fx_graph_and_output(torch.mul, a, b)
    local_tensor_proxy = a._local_tensor
    spec = output[0]._spec
    spec_proxy = AnyProxy(spec, history=a.history)
    return DTensorProxy(
        local_tensor_proxy=local_tensor_proxy,
        spec=spec_proxy,
        shape=tuple(spec.shape),
        device=local_tensor_proxy.device,
        dtype=local_tensor_proxy.dtype,
        requires_grad=local_tensor_proxy.requires_grad,
        grad=None,
        distparallel_type=None,
        thunder_fsdp_padding_size=None,
    )


dtensor_mul_prim = make_prim("dtensor_mul_prim", "dtensor_mul_prim", meta=dtensor_mul_meta)

dtensor_mul_prim_impl = pytorchex.register_operator("dtensor_mul_prim", like=dtensor_mul_prim, fn=torch.mul)

pytorchex.register_implementation(dtensor_mul_prim, dtensor_mul_prim_impl)


@dtensor_torchsymbol(torch.mul, id="dtensor.torch.mul")
def dtensor_mul(a: TensorLike, b: TensorLike) -> TensorLike:
    return dtensor_mul_prim(a, b)


@dtensor_torchsymbol(torch.ops.aten.add.Tensor, id="aten.add.Tensor")
def aten_add(a, b, alpha=1):
    if isinstance(alpha, TensorProxy) or alpha != 1:
        b = b * alpha

    return clang.add(a, b)


@dtensor_torchsymbol(torch.add, id="dtensor.torch.add")
def dtensor_add(a: TensorLike, b: TensorLike, alpha=1) -> TensorLike:
    return trace_torch_op_to_aten_ops(torch.add, a, b, alpha=alpha)


def register_dtensor_and_aten_function():
    register_function_for_dtensor(torch.add, ltorch.add, dtensor_add, is_method=True)
    register_function_for_dtensor(torch.mul, ltorch.mul, dtensor_mul, is_method=True)
