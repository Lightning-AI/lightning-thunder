from collections.abc import Sequence
from functools import wraps
from itertools import chain
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._guards import TracingContext, tracing
from functorch.compile import aot_function
from torch.distributed.tensor._dtensor_spec import DTensorSpec, DeviceMesh, TensorMeta
from torch.distributed.tensor import DeviceMesh, DTensor, Partial, Placement, Replicate, Shard  # noqa: F401

from thunder.core.symbol import BoundSymbol
from thunder.core.codeutils import SigInfo
from thunder.core import prims
from thunder.core import utils
from thunder.core.proxies import ProxyInterface
from thunder.core.pytree import tree_map
from thunder.core.trace import TraceCtx, tracectx, get_tracectx, detached_trace
from thunder.core.proxies import TensorProxy, NumberProxy
from thunder.core.devices import to_torch_device
from thunder.core.dtypes import to_torch_dtype
from thunder.core.pytree import tree_map, tree_flatten, tree_unflatten
from thunder import trace
from thunder.core.transforms import eval_trace
from thunder.executors.torch_compile import to_torch_translator

from thunder.torch.experimental.dtensor_proxy import DTensorProxy
from thunder.torch.experimental import dtensor_prims_and_impl


def trace_from_bsym_or_bsyms(bsym_or_bsyms: BoundSymbol | Sequence[BoundSymbol]) -> TraceCtx:
    """
    Create a `TraceCtx` object from a single `BoundSymbol` or a sequence of `BoundSymbol` objects.

    Args:
        bsym_or_bsyms (BoundSymbol | Sequence[BoundSymbol]): A single `BoundSymbol` or a sequence of `BoundSymbol`
            objects to create the trace from.

    Returns:
        TraceCtx: A `TraceCtx` object created from the `bsym_or_bsyms` argument.
    """
    bsyms = list(utils.sequencify(bsym_or_bsyms))
    trace_args = bsyms[0].flat_args
    trace_name = bsyms[0].sym.name

    unpack_bsyms = [
        prims.unpack_trivial.bind(a, name=a.name, output=a)
        for a in filter(lambda a: isinstance(a, ProxyInterface), trace_args)
    ]

    trace = TraceCtx()
    trace.bound_symbols.extend(unpack_bsyms + bsyms)
    trace.args = trace_args
    with tracectx(trace):
        prims.python_return(bsyms[-1].output)
    with tracectx(trace):
        # note(crcrpar): Give prefix `tmp` to avoid infinite recursion due to the same name
        trace._siginfo = SigInfo.from_name_and_args(f"tmp_{trace_name}", trace.args)

    return trace


def make_trace_executable(trace_to_convert: TraceCtx, *args_for_eval, **kwargs_for_eval) -> TraceCtx:
    """
    Converts a TraceCtx object into a PyTorch executable trace.

    Args:
        trace_to_convert (TraceCtx): The trace context object to be converted into an executable form.
        *args_for_eval: Positional arguments that may be used during the evaluation of the trace.
        **kwargs_for_eval: Keyword arguments that may be used during the evaluation of the trace.

    Returns:
        TraceCtx: A PyTorch executable `TraceCtx`.
    """

    @wraps(trace_to_convert.python_callable())
    def torch_interpreted_func(*args, **kwargs):
        return eval_trace(trace_to_convert, *args, **kwargs, symbol_mapper=to_torch_translator)

    torch_trace = trace(inline_trace=False)(torch_interpreted_func, *args_for_eval, **kwargs_for_eval)
    return torch_trace


def get_fx_graph(trc: TraceCtx, args_for_fx: Sequence[Any], args_for_trace: Sequence[Any]):
    f = trc.python_callable(include_decorators=False)

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

        fake_t = tree_map(materialize_fake_tensors, args_for_fx)

    def aot_wrapped_fn(args_for_fx):
        return f(*args_for_fx)

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
        fake_t
    )

    aten_graph = fwd_graph
    # `aten_graph`
    # class <lambda>(torch.nn.Module):
    # def forward(self, arg0_1: "f32[8, 16]", arg1_1: "f32[8, 16]"):
    #     # No stacktrace found for following nodes
    #     mul: "f32[8, 16]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    #     return (mul,)

    from thunder.dynamo.utils import _checkpoint_function_converter
    import thunder

    # Replaces `aten` ops with thunder equivalent
    # and this makes the `g` traceable with thunder.
    _checkpoint_function_converter(aten_graph)
    aten_graph.recompile()
    # `aten_graph`
    # class <lambda>(torch.nn.Module):
    # def forward(self, arg0_1: "f32[8, 16]", arg1_1: "f32[8, 16]"):
    #     # No stacktrace found for following nodes
    #     aten_mul = thunder_core_symbol_aten_mul(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    #     return (aten_mul,)

    # Otherwise `thunder.trace` get's confused.
    del aten_graph.meta

    @wraps(aten_graph)
    def wrap(*args):
        return aten_graph(*args)

    def flatten_tensor_proxy(t):
        # aten_graph takes local_tensors as input.
        return t._local_tensor

    flattened_args = tuple(
        flatten_tensor_proxy(arg) if isinstance(arg, DTensorProxy) else arg for arg in args_for_trace
    )

    trc = thunder.trace(rename_proxies=False, inline_trace=False)(wrap, *flattened_args)

    aten_syms = []
    # This seems hacky (is there a better way?)
    for bsym in trc.bound_symbols:
        if "aten" in bsym.sym.name:
            aten_syms.append(bsym)

    # Get args of the `return` BoundSymbol
    assert trc.bound_symbols[-1].sym == prims.python_return
    proxy_out = trc.bound_symbols[-1].flat_args

    with tracectx(trc):
        flat_output, _ = pytree.tree_flatten(aot_output)

    return aten_syms, flat_output, proxy_out


def decompose_into_aten_subsymbols(bsym_executable_trace, comp_trace, *args, **kwargs):
    flat_args, _ = pytree.tree_flatten((args, kwargs))
    filter_tensor_proxies = list(filter(lambda t: isinstance(t, DTensorProxy), flat_args))

    with tracectx(comp_trace):
        new_args = []
        for arg in flat_args:
            if isinstance(arg, DTensorProxy):
                # Add call to get `local_tensor` for input DTensors to current `comp_trace` scope
                tp = arg
                if not comp_trace.has_name(tp.name):
                    comp_trace.add_name(tp.name)
                if not comp_trace.has_name(tp._local_tensor.name):
                    comp_trace.add_name(tp._local_tensor.name)
                new_args.append(dtensor_prims_and_impl.get_dtensor_inner_tensor(tp))
            else:
                new_args.append(arg)

    # NOTE: Setting the TracingContext is important else `aot_function` used in `get_fx_graph`
    #       may find different FakeTensorMode.
    with tracing(TracingContext(FakeTensorMode())):
        aten_bsyms, flat_fake_output, flat_proxy_output = get_fx_graph(bsym_executable_trace, flat_args, new_args)

    # `aten_bsyms`
    # [t0 = thunder.torch.experimental.dtensor_torch_and_aten_ops.aten_mul(t3, t6)  # t0: "cuda:0 f32[8, 16]"
    #   # t0 = prims.mul(t3, t6)  # t0: "cuda:0 f32[8, 16]"]
    # `flat_fake_output`
    # [DTensor(local_tensor=FakeTensor(..., device='cuda:0', size=(8, 16)), device_mesh=DeviceMesh('cuda', [0, 1]), placements=(Shard(dim=0),))]
    # `proxy_output`
    # [<TensorProxy(name="t0", dtype=thunder.dtypes.float32, shape=(8, 16))>]

    with tracectx(comp_trace):
        # Add the aten operations bsyms (on local_tensor) to the current `comp_trace` scope
        comp_trace.peek_scope().extend(aten_bsyms)

        # Add relevant name to the Trace.
        for bsym in aten_bsyms:
            for arg in bsym.flat_proxy_args:
                if not comp_trace.has_name(arg.name):
                    comp_trace.add_name(arg.name)
            for out in bsym.flat_proxy_outs:
                if not comp_trace.has_name(out.name):
                    comp_trace.add_name(out.name)

        outs = []

        # Add `construct_dtensor` which will return the final output for the symbol to the current `comp_trace` scope.
        for _, flat_fake_o, proxy_o in zip(bsym.flat_outs, flat_fake_output, flat_proxy_output):
            o = dtensor_prims_and_impl.construct_dtensor(proxy_o, flat_fake_o._spec)
            outs.append(o)

    return outs


def _get_inner_tensor_proxy(*args, **kwargs):
    def get_local_tensor_from_dtensor(t):
        if isinstance(t, DTensorProxy):
            local_tensor = t._local_tensor
            current_trace = get_tracectx()
            # Maybe add name to current trace.
            if not current_trace.has_name(local_tensor.name):
                current_trace.names.add(local_tensor.name)

            return local_tensor

        return t

    return tree_map(get_local_tensor_from_dtensor, (args, kwargs))


def trace_torch_op_to_aten_ops(op, *args, **kwargs):
    flat_args, _ = pytree.tree_flatten((args, kwargs))
    tensor_proxies = list(filter(lambda t: isinstance(t, TensorProxy), flat_args))
    filter_tensor_proxies = list(map(lambda t: isinstance(t, DTensorProxy), tensor_proxies))
    assert len(filter_tensor_proxies) == len(tensor_proxies), f"Expected all tensors to be DTensor but found a mix"

    with detached_trace():
        tensor_proxy_args, tensor_proxy_kwargs = _get_inner_tensor_proxy(*args, **kwargs)
        out = op(*tensor_proxy_args, **tensor_proxy_kwargs)
        _, out_spec = tree_flatten(out)
        current_trace = get_tracectx()
        op_bsym = current_trace.bound_symbols[0]

    # `op_bsym`
    # t0 = ltorch.mul(t1, t3)  # t0: "cuda:0 f32[8, 16]"
    #   # t0 = prims.mul(t1, t3)  # t0: "cuda:0 f32[8, 16]"

    trc = trace_from_bsym_or_bsyms(op_bsym)
    # `trc`
    # @torch.no_grad()
    # @no_autocast
    # def tmp_mul(t1, t3):
    # # t1: "cuda:0 f32[8, 16]"
    # # t3: "cuda:0 f32[8, 16]"
    # t0 = ltorch.mul(t1, t3)  # t0: "cuda:0 f32[8, 16]"
    #     # t0 = prims.mul(t1, t3)  # t0: "cuda:0 f32[8, 16]"
    # return t0

    executable_trc = make_trace_executable(trc, *op_bsym.flat_args)
    # `executable_trc`
    # @torch.no_grad()
    # @no_autocast
    # def tmp_mul(t1, t3):
    # # t1: "cuda:0 f32[8, 16]"
    # # t3: "cuda:0 f32[8, 16]"
    # t0 = torch.mul(t1, t3)  # t0: "cuda:0 f32[8, 16]"
    #     # t0 = ltorch.mul(t1, t3)  # t0: "cuda:0 f32[8, 16]"
    #     # t0 = prims.mul(t1, t3)  # t0: "cuda:0 f32[8, 16]"
    # return t0

    current_computation_trc = get_tracectx()
    aten_outs = decompose_into_aten_subsymbols(executable_trc, current_computation_trc, *args, **kwargs)
    return tree_unflatten(aten_outs, out_spec)
