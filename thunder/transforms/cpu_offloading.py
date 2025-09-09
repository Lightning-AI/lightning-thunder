from collections.abc import Callable, Mapping


import thunder
import torch
from thunder.core import prims
from thunder.core.proxies import TensorProxy, Variable, variableify
from thunder.core.pytree import tree_map
from thunder.core.trace import from_trace, tracectx, TraceTag, TraceCtx
from thunder.core.transform_common import Transform
from thunder.core.symbol import BoundSymbol
from thunder.core.transforms import bsym_list_to_dag, Node, toposort_bsym_dag, TOPOSORT_ORDER
from thunder.core.vjp_utils import get_saved_for_backward_tensors
from thunder.extend import OperatorExecutor


stateful_offload_ex = OperatorExecutor("stateful_offload_ex")


class StreamStateManager:
    def __init__(self):
        self.offload_stream = torch.cuda.Stream()
        self.offload_tensor_event_pairs = {}
        self.reload_tensor_event_pairs = {}

    def offload_impl(self, t):
        # Due to https://github.com/Lightning-AI/lightning-thunder/issues/950
        # it may receive tensor on CPU.
        if not hasattr(t, "device") or t.device == torch.device("cpu"):
            return t
        with torch.cuda.stream(self.offload_stream):
            try:
                packed = torch.empty(
                    t.size(),
                    dtype=t.dtype,
                    layout=t.layout,
                    pin_memory=True,
                )
            except Exception:
                try:
                    packed = torch.empty(
                        t.size(),
                        dtype=t.dtype,
                        layout=t.layout,
                        pin_memory=False,
                    )
                except Exception as e:
                    raise e
            packed.copy_(t, non_blocking=True)
        offload_event = self.offload_stream.record_event()
        self.offload_tensor_event_pairs[packed] = offload_event
        return packed

    @staticmethod
    def offload_meta(t):
        return TensorProxy("offloaded_" + t.name, like=t, device=thunder.core.devices.Device("cpu"))

    def reload_impl(self, t, device):
        if not hasattr(t, "device") or t not in self.offload_tensor_event_pairs:
            return t
        # The reloaded tensor is allocated on the default stream because the PyTorch memory
        # allocator cannot move tensors from one stream to another without the expensive
        # calls to cudaFree and cudaMalloc.
        reloaded_tensor = torch.empty_like(t, device=device)
        self.offload_stream.wait_stream(torch.cuda.current_stream())
        self.offload_stream.wait_event(self.offload_tensor_event_pairs[t])
        with torch.cuda.stream(self.offload_stream):
            reloaded_tensor.copy_(t, non_blocking=True)
        reload_event = self.offload_stream.record_event()
        if isinstance(t, torch.Tensor):
            self.reload_tensor_event_pairs[t] = reload_event
        return reloaded_tensor

    @staticmethod
    def reload_meta(t, device):
        return TensorProxy(like=t, device=thunder.core.devices.Device(device))

    def wait_reload(self, t):
        self.offload_stream.wait_event(self.reload_tensor_event_pairs[t])
        return t

    @staticmethod
    def wait_reload_meta(t):
        return TensorProxy(like=t)


stream_manager = StreamStateManager()

offload = stateful_offload_ex.register_operator(
    "offload",
    fn=stream_manager.offload_impl,
    meta=stream_manager.offload_meta,
)


reload = stateful_offload_ex.register_operator(
    "reload",
    fn=stream_manager.reload_impl,
    meta=stream_manager.reload_meta,
)


wait_reload = stateful_offload_ex.register_operator(
    "wait_reload",
    fn=stream_manager.wait_reload,
    meta=stream_manager.wait_reload_meta,
)


# # Create a new executor to register the offload operators.
# offload_ex = OperatorExecutor("offload_ex")

# # NOTE: We create the offloaded CPU tensor in pinned memory and load the tensor back onto GPU with `to(non_blocking=True)`.
# #       These allow for better memory transfer speeds.
# #       Read the following tutorial for detailed explanation - https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html

# # Offload the GPU tensor to a pinned CPU tensor.
# def offload_impl(t):
#     # Due to https://github.com/Lightning-AI/lightning-thunder/issues/950
#     # it may receive tensor on CPU.
#     if not hasattr(t, "device") or t.device == torch.device("cpu"):
#         return t

#     packed = torch.empty(
#         t.size(),
#         dtype=t.dtype,
#         layout=t.layout,
#         pin_memory=True,
#     )
#     packed.copy_(t)
#     return packed

# offload = offload_ex.register_operator(
#     "offload",
#     meta=lambda t: TensorProxy("offloaded_" + t.name, like=t, device=thunder.core.devices.Device("cpu")),
#     fn=offload_impl,
# )

# # !!! Would it make sense to have a large global GPU buffer that all tensors can be loaded into?
# # Not clear that we don't need two saved tensors at the same time, also the del statements
# # release the memory.
# def reload_impl(t, device):
#     if t is None:
#         return t
#     return t.to(device, non_blocking=True)


# reload = offload_ex.register_operator(
#     "reload",
#     meta=lambda t, device: TensorProxy(like=t, device=thunder.core.devices.Device(device)),
#     fn=reload_impl,
# )


def get_symbol_to_idx(symbols):
    """
    This function returns a map from symbol to its position in the sequence.
    """
    return {sym: idx for idx, sym in enumerate(symbols)}


def move_large_ops_closer_to_end(execution_trace: TraceCtx) -> TraceCtx:
    bound_symbols = execution_trace.bound_symbols[:-1]
    return_bsym = execution_trace.bound_symbols[-1]

    def prefer_large_output_ops_closer_to_end(eligible_nodes: list[Node]) -> int:
        def key(node: Node) -> int:
            return sum(
                out._numel if isinstance(out, TensorProxy) else 0 for out in node.bsym.flat_proxy_outs
            )  # should be numel_ times size(dtype) # CollectionProxies?

        return max(range(len(eligible_nodes)), key=lambda i: key(eligible_nodes[i]))

    # This moves all del or clear collection at the bottom (as they don't return anything)
    bound_symbols = toposort_bsym_dag(
        bsym_list_to_dag(bound_symbols)[1],
        TOPOSORT_ORDER.BOTTOM_UP,
        selector=prefer_large_output_ops_closer_to_end,
    )

    # for idx, bsym in enumerate(bound_symbols):
    #     if bsym.sym.id == prims.PrimIDs.DEL:
    #         break

    bound_symbols = [
        bsym for bsym in bound_symbols if bsym.sym.id != prims.PrimIDs.DEL and bsym.sym.id != "clear_mutable_collection"
    ]
    bound_symbols.append(return_bsym)

    new_execution_trace = from_trace(execution_trace)
    new_execution_trace.bound_symbols = bound_symbols

    new_execution_trace = thunder.executors.passes.del_last_used(new_execution_trace, clear_mutable_collections=True)
    return new_execution_trace


def get_symbols_to_first_or_last_used_variables(symbols, first_used=False):
    """
    This function processes a sequence of symbols and determines which variables
    are first/last used by each symbol determined based on argument `first_used`.
    It returns a mapping from variables to the symbols where they were first/last used.

    Args:
        symbols (iterable): An iterable of symbols
        first_used (bool): Whether to return the map of first used variable to symbol mapping if True otherwise return the map for last used.
                           Defaults to False.

    Returns:
        variable_to_symbol (dict): A dictionary mapping each variable to the  symbol where it is first/last used based on `first_used` argument.
    """
    variable_to_symbol = {}

    def _mark_first_or_last_use(symbol, variable):
        if variable not in variable_to_symbol:
            variable_to_symbol[variable] = symbol

    iter_symbols = symbols if first_used else reversed(symbols)
    for symbol in iter_symbols:
        # If this function is used in the combined nvfuser+torch executor, there are no symbols but regions.
        # Regions do not have args, kwargs
        if hasattr(symbol, "inputs"):
            variables = tuple(symbol.inputs) + tuple(symbol.outputs)
        else:
            variables = (symbol.flat_variableified_proxy_args) + tuple(symbol.flat_variableified_proxy_outs)
        tree_map(lambda x: _mark_first_or_last_use(symbol, x), variables)

    return variable_to_symbol


class CPUOffloading(Transform):
    """
    Transform to implement CPU Offloading.

    Args:
        save_tensor_policy: Users can pass a callback with signature fn(offloaded_tensors, forward_trace) to filter
                            the offloaded_tensors based on their preference eg. biggest 20% intermediate tensors or
                            intermediates of certain operations
    """

    def __init__(
        self, save_tensor_policy: Callable[[tuple[TensorProxy, ...], TraceCtx], tuple[TensorProxy, ...]] | None = None
    ):
        self.forward_pass = None
        self.backward_pass = None
        self._offloaded_tensors: Mapping[Variable, TensorProxy] = {}
        self._offloaded_tensors_dev: Mapping[Variable, str] = {}
        self.save_tensor_policy = None
        if save_tensor_policy is not None:
            assert callable(save_tensor_policy)
            self.save_tensor_policy = save_tensor_policy

    def _get_tensors_to_offload(self, forward_trace):
        """
        Based on the `forward_trace`, we find the symbols that we want to offload to CPU.
        This function finds the intermediate tensors that are saved for backward i.e. ones that are not input or output of this trace.
        """
        return_bsym = forward_trace.bound_symbols[-1]
        trace_args = return_bsym.args[0]["flat_args"]
        saved_tensors = get_saved_for_backward_tensors(forward_trace)

        tensor_args_name = tuple(arg.name for arg in trace_args if isinstance(arg, TensorProxy))

        def is_in_tensor_args(t):
            return t.name in tensor_args_name

        def is_cuda_tensor(t):
            return t.device.type == "cuda"

        # Tensors which are intermediate and not argument to the computation trace are
        # the ones we are interested in offloading.
        tensors_to_offload = tuple(t for t in saved_tensors if ((not is_in_tensor_args(t)) and is_cuda_tensor(t)))

        # Apply users policy if present.
        if self.save_tensor_policy is not None:
            tensors_to_offload = self.save_tensor_policy(tensors_to_offload, forward_trace)
        self.tensors_to_offload = tensors_to_offload
        return self.tensors_to_offload

    def _replace_saved_tensors(self, forward_trace, new_output_map):
        return_bsym = forward_trace.bound_symbols.pop(-1)
        new_return_bsym = return_bsym.from_bsym_swap_proxies(new_output_map)

        # Replace the old return with our new return.
        forward_trace.bound_symbols.append(new_return_bsym)

    def _offload_tensors_from_forward(self, computation_trace):
        """
        This function takes the forward computation trace and performs following step
        1. Find the tensors to be offloaded using `_get_tensors_to_offload` (this also calls users `save_tensor_policy` if present).
        2. Insert calls to the `offload_to_cpu` symbol with the tensor to offload. These calls are placed after the last computational
           use of the tensors to be offloaded so that we free the memory as soon as possible.
        3. Finally, we update the last symbol i.e. `return` symbol to return the offloaded tensors instead of the original tensors.
        """
        # Step 1
        # Find the tensors to offload.
        # We offload saved tensors which are not arguments to the computation trace and are saved for backwards.
        tensors_to_offload = self._get_tensors_to_offload(computation_trace)

        # Step 2
        # Insert the offloading calls after the last use of the saved tensor (which we want to offload).
        # NOTE - We pass `computation_trace.bound_symbols[:-1]` as we don't want to pass the `return` symbol (which will otherwise be the last use of the saved tensors).
        variable_to_last_symbol = get_symbols_to_first_or_last_used_variables(
            computation_trace.bound_symbols[:-1], first_used=False
        )
        symbol_to_idx = get_symbol_to_idx(computation_trace.bound_symbols)

        # Bookkeeping for backward pass update.
        new_output_map: Mapping[Variable, TensorProxy] = {}
        new_output_dev_map: Mapping[Variable, str] = {}

        # Since we are inserting in the list (we need to obey increasing order) - else the insertions will be incorrect.
        sorted_tensors_to_offload = sorted(
            tensors_to_offload, key=lambda t: symbol_to_idx[variable_to_last_symbol[variableify(t)]]
        )
        for idx, t in enumerate(sorted_tensors_to_offload):
            last_used_symbol = variable_to_last_symbol[variableify(t)]
            last_used_symbol_idx = symbol_to_idx[last_used_symbol]
            computation_trace.push_scope([])
            with tracectx(computation_trace):
                o = offload(t)
                prims.python_del(t)
            scoped_comp = computation_trace.pop_scope()
            # scoped_comp[0].header = "Created by CPU Offloading Transform"
            offload_to_cpu_symbol = scoped_comp[0]
            del_symbol = scoped_comp[1]

            # This will insert `del` first and then push it down when we insert `offload_to_cpu`.
            computation_trace.bound_symbols.insert(last_used_symbol_idx + 1 + (idx * 2), del_symbol)
            computation_trace.bound_symbols.insert(last_used_symbol_idx + 1 + (idx * 2), offload_to_cpu_symbol)

            # Update bookkeeping.
            new_output_map[variableify(t)] = o
            new_output_dev_map[variableify(t)] = t.device.device_str()

        # Step 3
        # Update the return symbol to return our offloaded tensors in saved for backward.
        self._replace_saved_tensors(computation_trace, new_output_map)

        # Book keeping for backward pass update.
        self._offloaded_tensors = new_output_map
        self._offloaded_tensors_dev = new_output_dev_map
        return computation_trace

    def _load_tensors_for_backward(self, computation_trace):
        """
        This function takes the backward computation trace and performs following step
        1. Finds the unpack collection symbol which unpacks the saved tensors passed to the backward trace.
        2. Updates the unpack collection to unpack the offloaded tensors instead of the original ones.
        3. Before the first use of the offloaded tensor in computation, we insert the `load_to_gpu` to load the tensor back on GPU.
        """
        self.backward_pass = computation_trace
        offloaded_tensors = self._offloaded_tensors
        offloaded_tensors_dev_map = self._offloaded_tensors_dev

        compute_producers, compute_consumers = thunder.core.utils.producers_and_consumers(computation_trace)

        # We want to insert `loads` before the first use of offloaded_tensors.
        variable_to_first_symbol = get_symbols_to_first_or_last_used_variables(
            computation_trace.bound_symbols, first_used=True
        )

        symbol_to_idx = get_symbol_to_idx(computation_trace.bound_symbols)

        # Step 1 and 2
        # Update unpack collection so that it
        # outputs the offloaded tensor proxies (not the original ones).
        unpack_sym = compute_producers[list(offloaded_tensors.keys())[0].proxy]
        unpack_idx = symbol_to_idx[unpack_sym]
        unpack_sym_out = unpack_sym.output
        new_out = []
        for out in unpack_sym_out:
            if (vout := variableify(out)) in offloaded_tensors:
                new_out.append(offloaded_tensors[vout])
            else:
                new_out.append(out)
        new_unpack_bsym = BoundSymbol.from_bsym(unpack_sym, output=tuple(new_out))
        computation_trace.bound_symbols[unpack_idx] = new_unpack_bsym

        # Now we again find the first usages of offloaded tensor
        # This will actually point us to the first consumer of the offloaded tensor.
        offset = unpack_idx + 1
        variable_to_first_symbol = get_symbols_to_first_or_last_used_variables(
            computation_trace.bound_symbols[offset:], first_used=True
        )

        # Step 3
        # Load the offloaded tensors to GPU before usage.
        # Should iterate in correct order (else insertion positions will be incorrect).
        for idx, (vt, offloaded_t) in enumerate(
            sorted(offloaded_tensors.items(), key=lambda kv: symbol_to_idx[variable_to_first_symbol[kv[0]]])
        ):
            first_used_symbol = variable_to_first_symbol[vt]
            first_used_symbol_idx = symbol_to_idx[first_used_symbol]
            t = vt.proxy
            device = offloaded_tensors_dev_map[vt]

            with tracectx(computation_trace):
                new_sym = reload.bind(offloaded_t, device, output=t)

            # new_sym.header = "Created by CPU Offloading Transform"
            # The minus 3 is to allow overlap of data transfer and computation, but it's very unsophisticated.
            # The min of 3 is to avoid inserting before unpacking.
            insert_index = max(first_used_symbol_idx + idx - 3, 3)
            computation_trace.bound_symbols.insert(insert_index, new_sym)

        computation_trace = thunder.executors.passes.del_last_used(computation_trace, clear_mutable_collections=True)

        return computation_trace

    def transform_trace_post_optimization(self, computation_trace: thunder.TraceCtx, **kwargs):
        if self.forward_pass is None:
            self.forward_pass = computation_trace
            # Processing for the forward pass (only if we are going to compute backward).
            if TraceTag.AUGMENTED_FORWARD in computation_trace.tags:
                # Create a new copy of computation trace using `from_trace`.
                new_computation_trace = from_trace(computation_trace)
                # `from_trace` creates a shallow copy where `bound_symbols` and `provenance` are not copied.
                new_computation_trace.bound_symbols = computation_trace.bound_symbols

                new_computation_trace = self._offload_tensors_from_forward(new_computation_trace)
        else:
            # Skip if no tensor was offloaded.
            if len(self._offloaded_tensors) == 0:
                return computation_trace

            # Create a new copy of computation trace using `from_trace`.
            new_computation_trace = from_trace(computation_trace)
            # `from_trace` creates a shallow copy where `bound_symbols` and `provenance` are not copied.
            new_computation_trace.bound_symbols = computation_trace.bound_symbols

            # Move the ops that produce large outputs closer to the end of the trace.
            new_computation_trace = move_large_ops_closer_to_end(new_computation_trace)

            # Transform the backward trace to load offloaded tensors back to the device.
            new_computation_trace = self._load_tensors_for_backward(new_computation_trace)

        return new_computation_trace
