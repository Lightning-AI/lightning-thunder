from collections.abc import Sequence
from functools import wraps
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._guards import TracingContext, tracing
from functorch.compile import aot_function
from torch.distributed.tensor import DTensor

import thunder
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.codeutils import SigInfo
from thunder.core import prims
from thunder.core import utils
from thunder.core.proxies import ProxyInterface, Proxy
from thunder.core.pytree import tree_map
from thunder.core.trace import TraceCtx, tracectx, get_tracectx, detached_trace
from thunder.core.proxies import TensorProxy, NumberProxy
from thunder.core.devices import to_torch_device
from thunder.core.dtypes import to_torch_dtype
from thunder.core.pytree import tree_map, tree_flatten, tree_unflatten
from thunder.executors.passes import transform_for_execution, dce
from thunder.executors.torchex import ex as pytorchex
from thunder.executors.pythonex import ex as pythonex
from thunder.dynamo.utils import _checkpoint_function_converter

from thunder.torch.experimental.dtensor_proxy import DTensorProxy


def get_fx_graph_and_output(torch_op, *args, **kwargs) -> tuple[torch.fx.GraphModule, Sequence[Any]]:
    """
    Generate a Torch FX graph and the corresponding output by tracing a TraceCtx object.

    Args:
        trc (TraceCtx): Trace which will be used to generate the FX Graph
        args_for_fx (Sequence[Any]): A sequence of input arguments to be passed during the FX tracing.

    Returns:
        tuple[torch.fx.GraphModule, Sequence[Any]]:
            - A `torch.fx.GraphModule` object representing the traced computation graph in FX.
            - Output(s) from evaluating the graph with the given inputs.
    """

    def f(*args, **kwargs):
        return torch_op(*args, **kwargs)

    tracing_ctx = TracingContext.try_get()
    fake_mode = tracing_ctx.fake_mode

    with fake_mode:

        def materialize_fake_tensors(t):
            # `aot_function` can't handle these proxy types.
            if isinstance(t, NumberProxy):
                return t.value

            if not isinstance(t, TensorProxy):
                return t

            if isinstance(t, DTensorProxy):
                i_t = torch.randn(
                    t._local_tensor.shape,
                    device=to_torch_device(t._local_tensor.device),
                    dtype=to_torch_dtype(t._local_tensor.dtype),
                )
                return DTensor(i_t, t._spec._o, requires_grad=False)

            return torch.randn(t.shape, device=to_torch_device(t.device), dtype=to_torch_dtype(t.dtype))

        args, kwargs = tree_map(materialize_fake_tensors, (args, kwargs))

    def aot_wrapped_fn(*args, **kwargs):
        return f(*args, **kwargs)

    fwd_graph = None

    def get_graph(name):
        def f(fx_g: torch.fx.GraphModule, inps):
            nonlocal fwd_graph
            assert name != "backward", "aot shouldn't have reached backward as it will be handled by thunder"
            if name == "forward":
                fwd_graph = fx_g
            return fx_g

        return f

    aot_output = aot_function(aot_wrapped_fn, fw_compiler=get_graph("forward"), bw_compiler=get_graph("backward"))(
        *args, **kwargs
    )
    # Example value of `fwd_graph`
    # def forward(self, arg0_1, arg1_1):
    #   mul = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    #   return (mul,)

    # Example value of `aot_output`
    # DTensor(local_tensor=FakeTensor(..., device='cuda:0', size=(8, 16)), device_mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),))

    return fwd_graph, aot_output
