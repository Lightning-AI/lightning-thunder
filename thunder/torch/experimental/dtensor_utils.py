from collections.abc import Sequence
from typing import Any

import torch
from torch._guards import TracingContext
from functorch.compile import aot_function
from torch.distributed.tensor import DTensor

from thunder.core.pytree import tree_map
from thunder.core.proxies import TensorProxy, NumberProxy
from thunder.core.devices import to_torch_device
from thunder.core.dtypes import to_torch_dtype
from thunder.core.pytree import tree_map
from thunder.core.trace import TraceCtx
from thunder.torch.experimental.dtensor_proxy import DTensorProxy
from thunder.core.prims import PrimIDs
from thunder.core.symbol import Symbol
from thunder.core.trace import from_trace

from torch._subclasses.fake_tensor import FakeTensorMode
from torch._guards import TracingContext, tracing


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
                    t.local_tensor.shape,
                    device=to_torch_device(t.local_tensor.device),
                    dtype=to_torch_dtype(t.local_tensor.dtype),
                )
                return DTensor(i_t, t.spec._o, requires_grad=False)

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


def run_with_fake_tensor(torch_op, *args, **kwargs):
    with tracing(TracingContext(FakeTensorMode())):
        fx_graph, output = get_fx_graph_and_output(torch_op, *args, **kwargs)
    return fx_graph, output


def check_dtensor_cotangent_metadata(dtensor, metadata):
    if not dtensor._spec == metadata:
        raise RuntimeError(
            "Metadata (placement and mesh) has changed for cotangent between tracing and runtime"
            f"during tracing it was {metadata} but at runtime it is {dtensor._spec}."
        )


def check_dtensor_cotangent_metadata_in_backward(bw_trace: TraceCtx):
    # NOTE: The metadata (placement and mesh) of the cotangent DTensor
    #       can be different at runtime than the one we assumed during tracing.
    #       Because of this, we currently add a check in backward to verify the same.
    #       However, in future, we should add a symbol which will take care of mapping
    #       the cotangent metadata at runtime to the cotangent metadata during tracing.
    #       Also refer: https://github.com/pytorch/pytorch/pull/118670

    # Quick implementation of a symbol to verify
    # that the metadata of the cotangent at runtime as that as during tracing.
    check_dtensor_cotangent_metadata_symbol = Symbol(
        name="check_dtensor_cotangent_metadata",
        meta=lambda dtensor, metadata: None,
        python_impl=check_dtensor_cotangent_metadata,
    )
    new_bw_trace = from_trace(bw_trace)
    new_bsyms = []
    for bsym in bw_trace.bound_symbols:
        # Find the `unpack_sequence` for the cotangents.
        if bsym.sym.id == PrimIDs.UNPACK_SEQUENCE and bsym.args[0].name == "cotangents":
            new_bsyms.append(bsym)
            args = bsym.args[0].collection()
            for arg in args:
                # For every DTensor cotangent,
                # add symbol to verify that the metadata is the same as during tracing.
                if isinstance(arg, DTensorProxy):
                    bsym = check_dtensor_cotangent_metadata_symbol.bind(arg, arg.spec._o, output=None)
                    new_bsyms.append(bsym)
        else:
            new_bsyms.append(bsym)

    new_bw_trace.bound_symbols = new_bsyms

    return new_bw_trace
