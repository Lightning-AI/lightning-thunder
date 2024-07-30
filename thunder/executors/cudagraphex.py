import time
from collections.abc import Hashable, Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch

from thunder.extend import FusionExecutor, register_executor
from thunder.core import utils, prims
from thunder.core.proxies import Proxy, unvariableify
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance, get_tracectx, set_tracectx, reset_tracectx
from thunder.executors.utils import Region
from thunder.executors.data_dependent_partition import fuse_bound_symbols, Node


@dataclass(**utils.default_dataclass_params)
class ArgsDescriptor:
    dtypes: tuple
    sizes: tuple
    strides: tuple
    non_tensor_args: tuple
    args: list = field(hash=False, repr=False, compare=False)


def to_arg_descriptor(*args):
    def extract_descriptor(arg):
        if isinstance(arg, torch.Tensor):
            return arg.dtype, arg.size(), arg.stride(), None
        else:
            return type(arg), None, None, arg

    dtypes, sizes, strides, non_tensor_args = zip(*map(extract_descriptor, args))
    return ArgsDescriptor(dtypes, sizes, strides, non_tensor_args, args)


@lru_cache
def build_cuda_graph(
    fn: Callable, args_descriptor: ArgsDescriptor, static_args_mask: tuple[bool, ...]
) -> tuple[torch.cuda.CUDAGraph, Sequence[torch.Tensor | Any], Sequence[torch.Tensor | Any]]:

    def get_static_buffer(x):
        if isinstance(x, torch.Tensor):
            return torch.empty_like(x).copy_(x)
        return x

    args = args_descriptor.args

    # Warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        static_inputs = tuple(
            get_static_buffer(arg) if not is_static else arg for arg, is_static in zip(args, static_args_mask)
        )
        for _ in range(3):
            fn(*static_inputs)

    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # Record
    # NOTE: We are using default private pool here, but it is possibly better to
    # use a custom pool for better memory management. See CUDA Graphs Tree in
    # PyTorch's Inductor: torch/_inductor/cudagraph_trees.py
    # Design doc: https://docs.google.com/document/d/1ZrxLGWz7T45MSX6gPsL6Ln4t0eZCSfWewtJ_qLd_D0E/view
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = fn(*static_inputs)

    return graph, static_inputs, static_outputs


class CUDAGraphCallable:
    def __init__(self, fn: Callable, num_static_inputs: None | int = None):
        self.fn = fn
        self.num_static_inputs = num_static_inputs

    def __call__(self, *args):
        if self.num_static_inputs is not None:
            static_inputs_mask = (True,) * self.num_static_inputs + (False,) * (len(args) - self.num_static_inputs)
        else:
            static_inputs_mask = tuple(isinstance(arg, torch.nn.Parameter) for arg in args)

        args_descriptor = to_arg_descriptor(*args)

        graph, static_inputs, static_outputs = build_cuda_graph(self.fn, args_descriptor, static_inputs_mask)

        for static_input, arg in utils.safe_zip(static_inputs, args):
            if id(static_input) != id(arg) and isinstance(static_input, torch.Tensor) and isinstance(arg, torch.Tensor):
                static_input.copy_(arg)

        graph.replay()

        return static_outputs


def make_callable(
    fn_name: str,
    bsyms: list[BoundSymbol],
    inputs: list[Proxy],
    outputs: list[Proxy],
) -> Callable:
    from inspect import Parameter, Signature

    region_fn_params = (
        Parameter(getattr(param, "name", f"arg{i}"), Parameter.POSITIONAL_ONLY) for i, param in enumerate(inputs)
    )

    region_fn_signature = Signature(region_fn_params)

    def region_fn():
        pass

    region_fn.__signature__ = region_fn_signature
    region_fn.__name__ = fn_name

    region_trace = TraceCtx(region_fn)
    region_trace.bound_symbols = bsyms
    region_trace.args = inputs
    region_trace.kwargs = {}
    region_trace.bound_symbols.append(prims.python_return.bind(outputs, output=()))

    return region_trace.python_callable()


class CUDAGraphExecutor(FusionExecutor):
    def __init__(self, name: Hashable):
        super().__init__(name, version=torch.version.cuda)

    def fuse(self, region: Region, fusion_counter: int, num_static_inputs: None | int = None) -> BoundSymbol:
        inputs = [unvariableify(inp) for inp in sorted(region.inputs, key=lambda var: var.proxy.name)]
        outputs = [unvariableify(out) for out in sorted(region.outputs, key=lambda var: var.proxy.name)]

        fusion_name = f"CUDAGraph{fusion_counter}"
        fusion_callable: Callable = make_callable(f"{fusion_name}_fn", region.bound_symbols, inputs, outputs)
        fusion_callable = CUDAGraphCallable(fusion_callable, num_static_inputs)

        fusion_sym = Symbol(fusion_name, meta=None, is_fusion=True, executor=self)
        fusion_bsym = BoundSymbol(
            fusion_sym,
            inputs,
            {},
            outputs,
            region.bound_symbols,
            _call_ctx={fusion_name: fusion_callable},
        )

        return fusion_bsym

    def can_fuse(self, bsym: BoundSymbol):
        curr_tracectx = get_tracectx()
        assert hasattr(curr_tracectx, "clear_collection_names")

        # Related to backward traces.
        # No CollectionProxies in fusions! {
        # Arguments of `clear_mutable_collection` should not be fused
        if bsym.sym.id == "clear_mutable_collection":
            curr_tracectx.clear_collection_names.add(bsym.args[0].name)
            return False

        # We let DEL to get fused, unless the deleted proxy is a CollectionProxy
        # consumed by the `clear_mutable_collection` symbol
        if bsym.sym.id == prims.PrimIDs.DEL and bsym.args[0].name in curr_tracectx.clear_collection_names:
            return False
        # }

        do_not_fuse_sym_set = {
            # Skip the very beginning and the very end of the trace
            prims.PrimIDs.UNPACK_TRIVIAL,
            prims.PrimIDs.UNPACK_KEY,
            prims.PrimIDs.UNPACK_EMPTY_DICT,
            prims.PrimIDs.UNPACK_SEQUENCE,
            prims.PrimIDs.RETURN,
            # Data-dependent ops
            prims.PrimIDs.ITEM,
        }

        if bsym.sym.id in do_not_fuse_sym_set:
            return False

        return True

    def fusion_pass(self, trace: TraceCtx, num_static_inputs: None | int = None) -> TraceCtx:
        start_time_ns: int = time.perf_counter_ns()

        def _should_fuse(a: Node, b: Node):
            # TODO: modify the logic to be able to potentially better handle
            # islands around data-dependent ops once these are supported by Thunder.

            def _can_fuse_node(n: Node):
                if len(n.group_bsyms) > 1:
                    return True

                bsym: BoundSymbol = n.group_bsyms[0]
                return self.can_fuse(bsym)

            return _can_fuse_node(a) and _can_fuse_node(b)

        fused_trace: TraceCtx = from_trace(trace)
        # Tracking CollectionProxies that are being consumed
        # by the `clear_collection_names`.
        # We want to avoid them in the fusion regions!
        fused_trace.clear_collection_names = set()
        fused_trace_tok = set_tracectx(fused_trace)

        bound_symbols_groups = fuse_bound_symbols(trace, _should_fuse)

        producers, consumers = utils.producers_and_consumers(trace)

        fusion_counter: int = 0
        fused_bsyms = []
        for bsyms in bound_symbols_groups:
            # Trivial symbols that we do not want to fuse
            if len(bsyms) == 1:
                fused_bsyms.append(bsyms[0])
                continue

            if not self.can_fuse(bsyms[0]):
                fused_bsyms.extend(bsyms)
            else:
                region = Region(producers, consumers, bsyms)
                fusion_bsym: BoundSymbol = self.fuse(region, fusion_counter, num_static_inputs)
                fusion_counter += 1
                fused_bsyms.append(fusion_bsym)

        fused_trace.bound_symbols = fused_bsyms
        delattr(fused_trace, "clear_collection_names")
        reset_tracectx(fused_trace_tok)

        end_time_ns = time.perf_counter_ns()
        elapsed_time_ns = end_time_ns - start_time_ns
        elapsed_time_ms = elapsed_time_ns // 1000000
        fused_trace.set_provenance(TraceProvenance(f"CUDAGraph fusion (took {elapsed_time_ms} milliseconds)"))

        return fused_trace


cudagraphex = CUDAGraphExecutor(name="cudagraphex")
register_executor(cudagraphex)
