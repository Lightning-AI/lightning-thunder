import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from thunder.core.transform_common import Transform
from thunder.core import utils, prims, vjp_utils
from thunder.core.proxies import Proxy, ProxyTag, unvariableify
from thunder.core.symbol import BoundSymbol, Symbol
from thunder.core.trace import (
    TraceCtx,
    from_trace,
    TraceProvenance,
    get_tracectx,
    set_tracectx,
    reset_tracectx,
    TraceTag,
)
from thunder.executors.utils import Region
from thunder.executors.data_dependent_partition import fuse_bound_symbols, Node


@dataclass(**utils.default_dataclass_params)
class ArgsDescriptor:
    dtypes: tuple
    sizes: tuple
    strides: tuple
    non_tensor_args: tuple


def to_arg_descriptor(*args):
    def extract_descriptor(arg):
        if isinstance(arg, torch.Tensor):
            return arg.dtype, arg.size(), arg.stride(), None
        else:
            return type(arg), None, None, arg

    if args:
        dtypes, sizes, strides, non_tensor_args = zip(*map(extract_descriptor, args))
    else:
        dtypes = sizes = strides = non_tensor_args = None
    return ArgsDescriptor(dtypes, sizes, strides, non_tensor_args)


class CUDAGraphRunner:
    """A class to facilitate creating and running cudagraphs for a CUDAGraphTransform.

    Key methods:
      .make_cuda_graph_callable_from_symbols
             the entry point from the CUDAGraphTransform that returns a callable
             (mapping to .call_cuda_graph) for a given series of bound symbols and
             inputs and outputs.
      .call_cuda_graph
             runs (and builds on cache miss) a cuda graph. This is (via the callable
             from .make_cuda_graph_callable_from_symbols) the entry point during
             execution

    There are two cache/information layers, one mapping to the callables via the name
    of the fusion in the trace (.python_callables acts as a cache, .trace_symbols is just for
    inspection). There is a separate cuda_graph_cache as there could be reaons to
    generate multiple graphs for inputs (e.g. changes in strides for inputs), this
    is .cuda_graph_cache.

    Note that these are good for inspection but are considered internals and might
    change.
    """

    def __init__(self, *, share_mem_pool: bool = False):
        self.cuda_graph_cache = {}  # cahce_key (.make_cache_key) -> (graph, static_inputs, static_outputs)
        self.python_callables = {}  # fn_name -> (callable. static_input_mask (or None))
        self.trace_symbols = {}  # fn_name -> (bsyms, inputs, outputs)
        self.name_counter = 1

        # NOTE: Every invocation of CUDAGraphTransform has a single CUDAGraphRunner associated with it.
        # This should allow the runner to share a single memory pool across all graphs it constructs on a given device.
        if share_mem_pool:
            self.mem_pool = [torch.cuda.graph_pool_handle()] * torch.cuda.device_count()
            self.stream = [torch.cuda.Stream()] * torch.cuda.device_count()
        else:
            self.mem_pool = None
            self.stream = None

    def get_static_buffer(self, x):
        if isinstance(x, torch.Tensor):
            return torch.empty_like(x).copy_(x)
        return x

    def build_cuda_graph(
        self, fn: Callable, args: list[any], static_args_mask: tuple[bool, ...]
    ) -> tuple[torch.cuda.CUDAGraph, Sequence[torch.Tensor | Any], Sequence[torch.Tensor | Any]]:
        static_inputs = tuple(
            self.get_static_buffer(arg) if not is_static else arg for arg, is_static in zip(args, static_args_mask)
        )

        torch.cuda.synchronize()
        if self.stream:
            # In the case of multiple devices and shared memory pooling, we want to use one stream/pool per device
            cur_device_index = None
            for arg in static_inputs:
                if isinstance(arg, torch.Tensor):
                    cur_device_index = arg.device.index
                    break
                assert cur_device_index is not None, (
                    "No tensor found in static inputs, cannot infer which stream to use for graph capture"
                )
            stream = self.stream[cur_device_index]
            pool = self.mem_pool[cur_device_index]
        else:
            stream = torch.cuda.Stream()
            pool = self.mem_pool
        stream.wait_stream(torch.cuda.current_stream())

        # Warmup
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(*static_inputs)

        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # Record
        # NOTE: we are (optionally) using a global memory pool and stream here
        # This may have unintended consequences if graph capture and replay occur in different orders
        # so disabled by default.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=pool):
            static_outputs = fn(*static_inputs)

        return graph, static_inputs, static_outputs

    def make_static_inputs_mask(self, fn_name, *args):
        _, static_inputs_mask = self.python_callables[fn_name]
        if static_inputs_mask is None:
            static_inputs_mask = tuple(isinstance(arg, torch.nn.Parameter) for arg in args)
        return static_inputs_mask

    def make_cache_key(self, fn_name, *args):
        # if args_descriptor included torch.nn.Parameter-ness, we would
        # could use the static_inputs_mask or None as the key.
        return (fn_name, self.make_static_inputs_mask(fn_name, *args), to_arg_descriptor(*args))

    def call_cuda_graph(self, fn_name, *args):
        fn, _ = self.python_callables[fn_name]

        cache_key = self.make_cache_key(fn_name, *args)

        cache_entry = self.cuda_graph_cache.get(cache_key)
        if cache_entry is None:
            static_inputs_mask = self.make_static_inputs_mask(fn_name, *args)
            cache_entry = self.build_cuda_graph(fn, args, static_inputs_mask)
            self.cuda_graph_cache[cache_key] = cache_entry

        graph, static_inputs, static_outputs = cache_entry

        for static_input, arg in utils.safe_zip(static_inputs, args):
            if id(static_input) != id(arg) and isinstance(static_input, torch.Tensor) and isinstance(arg, torch.Tensor):
                static_input.copy_(arg)

        graph.replay()
        return static_outputs

    def make_python_callable_from_symbols(
        self,
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
        region_trace.bound_symbols.append(prims.python_return.bind(outputs, output=None))
        return region_trace.python_callable()

    def make_cuda_graph_callable_from_symbols(
        self,
        bsyms: list[BoundSymbol],
        inputs: list[Proxy],
        outputs: list[Proxy],
        static_inputs_mask: Sequence[bool] | None = None,
    ) -> Callable:
        # previously, one could pass the number of static inputs to get an automatic static_inputs_mask,
        # but chances are that the transform could have more detailed information, so we take a mask
        #  static_inputs_mask = (True,) * num_static_inputs + (False,) * (len(inputs) - num_static_inputs)

        if static_inputs_mask is not None:
            static_inputs_mask = tuple(static_inputs_mask)  # ensure hashability
        else:
            static_inputs_mask = tuple(
                isinstance(i, Proxy) and ProxyTag.STATIC_MEMORY_LOCATION in i.tags for i in inputs
            )

        fn_name = f"CUDAGraph{self.name_counter}"
        self.name_counter += 1

        self.python_callables[fn_name] = (
            self.make_python_callable_from_symbols(fn_name, bsyms, inputs, outputs),
            static_inputs_mask,
        )
        self.trace_symbols[fn_name] = (bsyms, inputs, outputs)

        def callable(*args):
            return self.call_cuda_graph(fn_name, *args)

        callable.__name__ = f"{fn_name}_fn"
        callable.__qualname__ = f"{fn_name}_fn"

        return callable, fn_name


class CUDAGraphTransform(Transform):
    """
    Transform to fuse operations into CUDA graphs post optimization.

    This class provides the basic infrastructure, but it is expected that you might subclass this transform
    in order to override ``can_fuse```or other methods.
    """

    def __init__(self, *, share_mem_pool: bool = False):
        super().__init__()
        self.cuda_graph_runner = CUDAGraphRunner(share_mem_pool=share_mem_pool)
        self.outputs_from_forward = None

    def fuse(self, region: Region, fusion_counter: int) -> BoundSymbol:
        inputs = [unvariableify(inp) for inp in region.inputs]
        outputs = [unvariableify(out) for out in region.outputs]

        from thunder.executors.passes import _del_last_used

        region.bound_symbols = _del_last_used(region.bound_symbols, outputs)

        fusion_callable, fusion_name = self.cuda_graph_runner.make_cuda_graph_callable_from_symbols(
            region.bound_symbols,
            inputs,
            outputs,
        )

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

        # We let DEL to get fused (but should not see them much), unless the deleted proxy is a CollectionProxy
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

    def generate_fused_trace(self, trace: TraceCtx, **kwargs) -> TraceCtx:
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
                fusion_bsym: BoundSymbol = self.fuse(region, fusion_counter)
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

    def transform_trace_post_optimization(self, trace, **kwargs):
        if trace.siginfo().name == "backward_fn":
            # TODO: Backward TraceTag
            assert self.outputs_from_forward is not None, "called on backward without forward before"
            assert len(trace.bound_symbols[2].args) == 2 and trace.bound_symbols[2].args[0].name == "saved_for_backward"
            assert (
                trace.bound_symbols[9].sym.name == "unpack_sequence"
                and trace.bound_symbols[9].args[0] is trace.bound_symbols[2].output[0]
            )

            saved_for_backwards_unpacked = trace.bound_symbols[9].output
            assert len(saved_for_backwards_unpacked) == len(self.outputs_from_forward)
            for is_static, p_bw in zip(self.outputs_from_forward, saved_for_backwards_unpacked):
                if is_static:
                    p_bw.tags.add(ProxyTag.STATIC_MEMORY_LOCATION)
            self.outputs_from_forward = None

        new_trace = self.generate_fused_trace(trace, **kwargs)

        if TraceTag.AUGMENTED_FORWARD in new_trace.tags:
            assert self.outputs_from_forward is None, "called on augmented forward twice without backward in between"
            cudagraph_output_names = set()
            for bsym in new_trace.bound_symbols:
                if bsym.sym.name.startswith("CUDAGraph"):
                    for o in bsym.flat_proxy_outs:
                        cudagraph_output_names.add(o.name)
            saved_for_backward = vjp_utils.get_saved_for_backward_tensors(new_trace)
            self.outputs_from_forward = [
                o.name in cudagraph_output_names or ProxyTag.STATIC_MEMORY_LOCATION in o.tags
                for o in saved_for_backward
            ]

        return new_trace
