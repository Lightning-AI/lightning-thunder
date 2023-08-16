from dataclasses import dataclass, replace
from functools import partial
from itertools import chain, product
from typing import Callable, Optional, Sequence, Tuple, Union

from igraph import Graph

from thunder.core import prims, utils
from thunder.core.baseutils import BoundSymbolInterface, ProxyInterface
from thunder.core.prims import PrimIDs
from thunder.core.proxies import TensorProxy
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.trace import from_trace, TraceCtx, TraceProvenance


def find_external_producer_outputs(
    trace: TraceCtx,
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> Tuple[ProxyInterface, ...]:
    """Find producer's outputs that must be included in the output of the
    producer because they are used by other consumers.

    Args:
        trace (TraceCtx): Trace object.
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.

    Returns:
        Tuple[ProxyInterface, ...]: Producer's outputs that must be included in
        the output of the producer.
    """
    proxy_to_consumers = utils.consumers(trace)

    def filter_func(out: ProxyInterface):
        consumers = proxy_to_consumers.get(out, tuple())
        consumers = tuple(filter(lambda x: x.sym.name != "del", consumers))
        return len(consumers) == 1 and out.name in (x.name for x in consumer.args)

    rematerializable_producer_outputs = tuple(filter(filter_func, producer.output))

    return tuple(x for x in producer.output if x.name not in (y.name for y in rematerializable_producer_outputs))


def find_external_consumer_inputs(
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> Tuple[ProxyInterface, ...]:
    """Find consumer's inputs that must be included in the input of the
    consumer because they are produced by other producers.

    Args:
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.

    Returns:
        Tuple[ProxyInterface, ...]: Consumer's inputs that must be included in
        the input of the consumer.
    """
    external_consumer_inputs_names = tuple(
        sorted(
            set(x.name for x in consumer.args)
            - set(x.name for x in producer.output)
            - set(x.name for x in producer.args)
        )
    )
    return tuple(x for x in consumer.args if x.name in external_consumer_inputs_names)


def apply_rematerialization_for_producer(
    trace: TraceCtx,
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
    cut: Sequence[Union[ProxyInterface, str]],
) -> BoundSymbolInterface:
    """Update the producer node with the cut information.

    Args:
        producer (BoundSymbolInterface): Producer node.
        cut (Sequence[Union[ProxyInterface, str]]): Cut information.

    Returns:
        BoundSymbolInterface: Updated producer node.
    """
    # It's simple to update the producer node, all we need to do is to update
    # the producer's output with the cut information and the external outputs.
    cut_names = tuple(map(lambda x: x.name, cut)) if isinstance(cut[0], ProxyInterface) else tuple(cut)
    external_producer_outputs = find_external_producer_outputs(trace, producer, consumer)
    new_producer_output_names = tuple(x.name for x in external_producer_outputs) + cut_names
    # Remove the producer's inputs from the new producer's output.
    new_producer_output_names = tuple(
        x for x in new_producer_output_names if x not in (y.name for y in producer._flat_args)
    )
    all_produced_vars = tuple(chain.from_iterable((y for y in x._flat_outs) for x in producer.subsymbols))
    # Choose the new producer's output from all the produced variables.
    new_producer_output = tuple(x for x in all_produced_vars if x.name in new_producer_output_names)
    new_producer_output = tuple(sorted(new_producer_output, key=lambda x: x.name))
    new_producer = replace(producer, output=new_producer_output)
    return new_producer


def apply_rematerialization_for_consumer(
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
    cut: Sequence[Union[ProxyInterface, str]],
) -> BoundSymbolInterface:
    """Update the consumer node with the cut information.

    Args:
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.
        cut (Sequence[Union[ProxyInterface, str]]): Cut information.

    Returns:
        BoundSymbolInterface: Updated consumer node.
    """
    # It's a bit more complicated to update the consumer node, we need to
    # update the consumer's input with the cut information.
    # We need to keep consumer's inputs that are not in the cut and are not
    # produced by the producer. We call these inputs "external inputs".
    external_inputs = find_external_consumer_inputs(producer, consumer)
    all_produced_vars = tuple(chain.from_iterable((y for y in x._flat_outs) for x in producer.subsymbols))
    cut_names = tuple(map(lambda x: x.name, cut)) if isinstance(cut[0], ProxyInterface) else tuple(cut)
    cut_inputs = tuple(filter(lambda x: x.name in cut_names, (*all_produced_vars, *producer.args)))
    new_consumer_args = cut_inputs + external_inputs

    # We need to rematerialize the consumer's inputs that are not in the new consumer's inputs.
    rematerialized_inputs = tuple(
        filter(lambda x: x.name not in map(lambda x: x.name, new_consumer_args), consumer.args)
    )

    # Construct a temporary Trace object with subsymbols from both the producer and the consumer.
    trace = TraceCtx(None)
    trace.bound_symbols = (*producer.subsymbols, *consumer.subsymbols)

    recomputing_symbols = utils.find_producer_symbols(trace, rematerialized_inputs, cut_inputs)
    new_subsymbols = recomputing_symbols + tuple(consumer.subsymbols)

    # Some recomputing_symbols might require producer's inputs, so we need to
    # add them to the consumer's inputs.
    # Probably find_min_cut should have returned this information.
    all_args = tuple(
        chain.from_iterable(
            (x.name for x in bsym._flat_args if isinstance(x, ProxyInterface)) for bsym in new_subsymbols
        )
    )
    new_consumer_args += tuple(
        x for x in producer.args if x.name in all_args and x.name not in (x.name for x in new_consumer_args)
    )
    new_consumer_args = tuple(sorted(new_consumer_args, key=lambda x: x.name))
    new_consumer = replace(consumer, args=new_consumer_args, subsymbols=new_subsymbols)
    return new_consumer


def find_filtered_producer_consumer_pairs(
    trace: TraceCtx,
    filter_func: Optional[Callable] = None,
) -> Tuple[Tuple[BoundSymbolInterface, BoundSymbolInterface], ...]:
    """Find producer-consumer pairs among the filtered symbols.

    Args:
        trace (TraceCtx): Trace object.
        filter_func (Optional[Callable], optional): Filter function. Defaults to None.

    Returns:
        Tuple[Tuple[BoundSymbolInterface, BoundSymbolInterface], ...]: Producer-consumer bound symbol pairs.
    """
    filter_func = filter_func or (lambda x: True)
    proxy_to_consumers = utils.consumers(trace)
    producer_consumer_pairs = set()

    # We are looking for special producer-consumer pairs among the filtered symbols
    for producer in filter(filter_func, trace.bound_symbols):
        for out in producer._flat_outs:
            consumers = proxy_to_consumers.get(out, tuple())
            # We only care about producer's outputs with a single fusion consumer
            # prims.del is a special case that we don't care about
            consumers = tuple(filter(lambda x: x.sym.name != "del", consumers))
            if len(consumers) == 1 and filter_func(consumer := consumers[0]):
                producer_consumer_pairs.add((producer, consumer))
    return tuple(sorted(producer_consumer_pairs, key=lambda x: x[0].sym.name))


find_nvfuser_producer_consumer_pairs = partial(
    find_filtered_producer_consumer_pairs, filter_func=lambda x: x.sym.name.startswith("nvFusion")
)


# TODO: Consider moving this to the prims attribute
REMATERIALIZATION_BLOCK_LIST = {
    # We don't want to rematerialize the following primitives
    # Random primitives
    prims.PrimIDs.UNIFORM,
    # Reduction primitives
    prims.PrimIDs.SUM,
    prims.PrimIDs.VAR,
    prims.PrimIDs.VAR_MEAN,
    prims.PrimIDs.AMAX,
    prims.PrimIDs.AMIN,
    prims.PrimIDs.PROD,
    "torch.var_mean",
}


def find_cut(
    trace: TraceCtx,
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> Sequence[Union[ProxyInterface, str]]:
    """Find the minimal cut between the producer and the consumer.

    Args:
        trace (TraceCtx): Trace object.
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.

    Returns:
        Sequence[Union[ProxyInterface, str]]: Cut information.
    """
    # We are going to use the igraph library to find the minimal cut between the
    # producer and the consumer. Minimum cut is a set of edges that, if removed,
    # would disconnect the producer and the consumer. But we are not interested
    # in the edges, we are interested in the nodes that are connected by the
    # edges. These nodes are the cut nodes. So we need to reformulate our node
    # graph into an edge graph.

    # All the nodes from the producer that we connect to a "source" node will
    # not be in the cut. Similarly, all the nodes from the consumer that we
    # connect to a "sink" node will not be in the cut. So we need to add a
    # "source" node and a "sink" node to our graph. We also disallow the cut to
    # be in the consumer's part of the graph to avoid balancing the graph into
    # the producer from the consumer.

    # Determine which producer's outputs can be rematerialized
    external_producer_outputs = find_external_producer_outputs(trace, producer, consumer)

    # Required producer variables. These are the variables that are required to
    # be connected to the "source" node.
    required_producer_vars = tuple(x for x in producer.args)
    required_producer_vars += tuple(x for x in external_producer_outputs)

    # This is needed to avoid rematerializing random or reduction primitives.
    required_producer_vars += tuple(
        chain.from_iterable(
            (y for y in x._flat_outs) for x in producer.subsymbols if x.sym.id in REMATERIALIZATION_BLOCK_LIST
        )
    )

    # Required consumer variables. These are the variables that are required to
    # be connected to the "sink" node.
    required_consumer_vars = tuple(x.name for x in consumer.output)
    external_consumer_inputs = find_external_consumer_inputs(producer, consumer)
    required_consumer_vars += tuple(x.name for x in external_consumer_inputs)

    # To the required consumer variables we also need to add the path from the
    # consumer's output to the external consumer's inputs. This is needed to
    # avoid balancing the graph into the producer from the consumer.
    consumer_trace = TraceCtx(None)
    consumer_trace.bound_symbols = consumer.subsymbols
    required_consumer_symbols = tuple(
        utils.find_producer_symbols(consumer_trace, consumer.output, external_consumer_inputs)
    )
    required_consumer_vars += tuple(
        chain.from_iterable((y.name for y in x._flat_outs) for x in required_consumer_symbols)
    )

    # TODO: Use TensorProxy properties to compute the weights
    WEIGHT = 1.0

    # Create a graph
    edges = []
    name_to_id = {}
    capacities = []

    def add_edge(src, dst, capacity):
        if src not in name_to_id:
            name_to_id[src] = len(name_to_id)
        if dst not in name_to_id:
            name_to_id[dst] = len(name_to_id)
        src, dst = name_to_id[src], name_to_id[dst]
        edges.append((src, dst))
        capacities.append(capacity)

    utils.check(
        len(required_consumer_vars) > 0,
        "The consumer has no outputs. This is not supported by the cut finding algorithm.",
    )
    for var_name in required_consumer_vars:
        add_edge(var_name + "_in", "sink", capacity=float("inf"))

    sym_skip_list = (
        prims.PrimIDs.UNPACK_SEQUENCE,
        prims.PrimIDs.UNPACK_TRIVIAL,
        prims.PrimIDs.UNPACK_KEY,
        prims.PrimIDs.RETURN,
    )

    combined_trace = TraceCtx(None)
    combined_trace.bound_symbols = (*producer.subsymbols, *consumer.subsymbols)
    combined_consumers = utils.consumers(combined_trace)

    def get_weight(var):
        if isinstance(var, TensorProxy):
            return WEIGHT * var.dtype.bytes
        return WEIGHT

    def add_edges(var):
        var_name = var.name
        weight = get_weight(var)
        weight = weight / 2.0 if var_name in (x.name for x in producer.args) else weight
        add_edge(var_name + "_in", var_name + "_out", capacity=weight)
        for user in combined_consumers._dict.get(var_name, tuple()):
            if user.sym.id in sym_skip_list:
                continue
            for out in user._flat_outs:
                user_name = out.name
                add_edge(var_name + "_out", user_name + "_in", capacity=float("inf"))

    if not required_producer_vars:
        # If there are no required producer variables, we need to make sure that
        # the source node is added to the graph.
        add_edge("source", "source", capacity=float("inf"))

    for var in required_producer_vars:
        add_edge("source", var.name + "_in", capacity=float("inf"))
        add_edges(var)

    for symbol in chain(producer.subsymbols, consumer.subsymbols):
        for var in symbol._flat_outs:
            add_edges(var)

    g = Graph(
        n=len(name_to_id),
        edges=edges,
        directed=True,
        edge_attrs={"capacity": capacities},
    )
    source = name_to_id["source"]
    sink = name_to_id["sink"]
    partition = g.mincut(source, sink, "capacity").partition

    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, g.neighbors(n, mode="out")) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    id_to_name = dict(map(reversed, name_to_id.items()))
    cutset = ((id_to_name[u], id_to_name[v]) for u, v in cutset)
    cut_nodes = set()
    for node_in, node_out in cutset:
        if node_out == "sink":
            continue
        assert node_in.endswith("_in")
        assert node_out.endswith("_out")
        assert node_in[:-3] == node_out[:-4]
        node_name = node_in[:-3]
        cut_nodes.add(node_name)
    return tuple(sorted(cut_nodes))


# TODO: The following code is a temporary solution to update the call_ctx
# information of the nvFusion BoundSymbol object. This is needed because the
# nvFusion BoundSymbol object is created before the call_ctx information is
# updated. See more details in
# https://github.com/Lightning-AI/lightning-thunder/issues/515
def _update_nvfusion_call_ctx(trace: TraceCtx, bsym: BoundSymbolInterface) -> BoundSymbolInterface:
    """Update the call_ctx information of the nvFusion BoundSymbol object.

    Args:
        trace: The trace context.
        bsym: The nvFusion BoundSymbol object.

    Returns:
        The updated nvFusion BoundSymbol object.
    """
    from thunder.executors.nvfuserex import fuse

    @dataclass
    class BoundSymbolRegion:
        trace: TraceCtx
        inputs: tuple
        outputs: tuple
        bound_symbols: tuple
        counter: int

    def nvfusion_bsym_to_region(trace: TraceCtx, bsym: BoundSymbolInterface):
        counter = int(tuple(bsym._call_ctx.keys())[0].split("nvFusion")[-1])
        return BoundSymbolRegion(
            trace=trace,
            inputs=bsym.args,
            outputs=bsym.output,
            bound_symbols=bsym.subsymbols,
            counter=counter,
        )

    # fuse returns a new BoundSymbol object with correct updated call_ctx
    # information
    return fuse(nvfusion_bsym_to_region(trace, bsym))[0]


def rematerialize(trace: TraceCtx) -> tuple[TraceCtx, list[TraceCtx]]:
    """Rematerialize the trace.

    Args:
        trace (TraceCtx): Trace object.

    Returns:
        tuple[TraceCtx, list[TraceCtx]]: Rematerialized trace and the list of
            rematerialized traces.
    """
    # Find all the producers and consumers
    pairs = find_nvfuser_producer_consumer_pairs(trace)

    # Find the minimal cut between the producer and the consumer
    cuts = tuple(find_cut(trace, producer, consumer) for producer, consumer in pairs)

    # Pairs of producer and consumer are not unique. Each update to the producer
    # or consumer may affect the other. We need to update the producer and
    # consumer sequentially.
    producers = {producer for producer, _ in pairs}
    consumers = {consumer for _, consumer in pairs}
    new_bsyms = {bsym: bsym for bsym in producers | consumers}
    for (producer, consumer), cut in zip(pairs, cuts):
        current_producer = new_bsyms.get(producer, None) or producer
        current_consumer = new_bsyms.get(consumer, None) or consumer
        if cut:
            updated_producer = apply_rematerialization_for_producer(trace, current_producer, current_consumer, cut)
            updated_consumer = apply_rematerialization_for_consumer(current_producer, current_consumer, cut)
            new_bsyms[producer] = updated_producer
            new_bsyms[consumer] = updated_consumer
        else:
            new_bsyms[producer] = None
            new_bsyms[consumer] = None

    rematerialized_trace = from_trace(trace)
    rematerialized_trace.set_provenance(TraceProvenance("Rematerialization"))

    def replace_bound_symbol(bsym):
        new_bsym = new_bsyms.get(bsym, None)
        if new_bsym is not None:
            return _update_nvfusion_call_ctx(trace, new_bsym)
        return bsym

    # TODO: New bound symbols are still incorrect. Its _ctx_call dict points
    # to the old nvFuser fusion. We need to update it to use the new definition.

    new_bound_symbols = tuple(replace_bound_symbol(bsym) for bsym in trace.bound_symbols)
    rematerialized_trace.bound_symbols = new_bound_symbols
    return rematerialized_trace, [rematerialized_trace]


def rematerialize_forward_and_backward(fw_trace: TraceCtx, bw_trace: TraceCtx) -> tuple[TraceCtx, TraceCtx]:
    """Apply rematerialization optimization to the forward and backward traces.

    Args:
        fw_trace (TraceCtx): Forward trace.
        bw_trace (TraceCtx): Backward trace.

    Returns:
        tuple[TraceCtx, TraceCtx]: Rematerialized forward and backward traces.
    """
    # Circular dependency
    from thunder.core.transforms import (
        _update_backward_with_new_saved_for_backward,
        _update_forward_with_new_saved_for_backward,
    )

    def joint_fn(args, kwargs, cotangents):
        pass

    joint_extrace = TraceCtx(joint_fn)
    joint_extrace.args = (fw_trace.args, fw_trace.kwargs, bw_trace.args[1])
    assert fw_trace.bound_symbols[-1].sym.id == PrimIDs.RETURN
    assert bw_trace.bound_symbols[-1].sym.id == PrimIDs.RETURN
    # Omit the last RETURN symbol
    joint_extrace.bound_symbols = fw_trace.bound_symbols[:-1] + bw_trace.bound_symbols[:-1]
    # Add a new RETURN symbol
    joint_extrace.bound_symbols.append(
        replace(fw_trace.bound_symbols[-1], args=(fw_trace.bound_symbols[-1].args[0], bw_trace.bound_symbols[-1].args))
    )
    joint_extrace, _ = rematerialize(joint_extrace)

    # We need to update "save_for_backward" sequence
    new_bw_bsyms = joint_extrace.bound_symbols[len(fw_trace.bound_symbols) :]
    new_bw_bsyms = list(
        bsym
        for bsym in new_bw_bsyms
        if bsym.sym.id
        not in (
            PrimIDs.UNPACK_TRIVIAL,
            PrimIDs.UNPACK_SEQUENCE,
            PrimIDs.UNPACK_EMPTY_DICT,
            PrimIDs.UNPACK_KEY,
            PrimIDs.DEL,
            PrimIDs.RETURN,
        )
    )
    all_args = tuple(
        chain.from_iterable((x for x in bsym._flat_args if isinstance(x, ProxyInterface)) for bsym in new_bw_bsyms)
    )
    producers = utils.producers(new_bw_bsyms)
    new_required_for_backward = tuple(
        a for a in all_args if producers.get(a, None) is None and a.name not in (y.name for y in bw_trace.args[1])
    )
    new_required_for_backward = tuple(
        sorted({x.name: x for x in new_required_for_backward}.values(), key=lambda a: a.name)
    )  # Removes duplicates and sorts by name

    # Now construct the updated backward and forward traces
    new_bw_trace = from_trace(bw_trace)
    new_bw_trace.set_provenance(TraceProvenance("Rematerialization"))
    new_bw_trace.bound_symbols = new_bw_bsyms
    new_bw_trace.bound_symbols.append(replace(bw_trace.bound_symbols[-1], args=bw_trace.bound_symbols[-1].args))
    _update_backward_with_new_saved_for_backward(new_bw_trace, new_required_for_backward)

    new_fw_trace = from_trace(fw_trace)
    new_fw_trace.set_provenance(TraceProvenance("Rematerialization"))
    new_fw_trace.bound_symbols = list(
        bsym for bsym in joint_extrace.bound_symbols[: len(fw_trace.bound_symbols) - 1] if bsym.sym.id != PrimIDs.DEL
    )
    new_fw_trace.bound_symbols.append(replace(fw_trace.bound_symbols[-1], args=fw_trace.bound_symbols[-1].args))
    _update_forward_with_new_saved_for_backward(new_fw_trace, new_required_for_backward)
    return new_fw_trace, new_bw_trace


def split_vjp_trace_into_forward_and_backward(
    trace: TraceCtx,
) -> tuple[TraceCtx, TraceCtx]:
    """Split VJP transform applied trace into forward and backward.

    The input trace is supposed to follow the signature of ``func(inputs0, inputs1) -> (outputs0, outputs1)``,
    where
        - ``inputs0``: the inputs for the forward computation (a.k.a. primals)
        - ``inputs1``: the inputs for the backward computation (a.k.a. cotangents)
        - ``outputs0``: the results of the forward
        - ``outputs1``: the results of the backward, i.e. gradients

    This function splits the trace into the forward and the backward. This is feasible only when
    the extracted forward is independent from any of ``inputs1``. This function returns the following
    two functions:
        1. ``forward(*inputs0) -> (outputs0, saved_tensors_for_backward)``
        2. ``backward(*saved_tensors_for_backward, *inputs1) -> outputs1``

    Note that the two traces do not include :class:``BoundSymbols`` of ``prims.PrimIDs.del``.

    Args:
        trace: Trace of a function with VJP transform applied

    Returns:
        Two :class:``TraceCtx``s, one representing forward and the other backward.
        The forward trace has the signature of `forward_trace` with primals spelled out.
        The backward trace's arguments consist of ``*intermediate_values_of_forward`` and ``*cotangents``.
    """
    # There can be a case where the forward trace has a nvFusion merged with the backward trace.
    # In this case, we need to split the nvFusion into two nvFusions, one for the forward and the other
    # for the backward.
    nvfusion = find_common_nvfusion_consumer(trace)
    replace_nvfusion = None
    if nvfusion is not None:
        fw_nvfusion, bw_nvfusion = split_nvfusion_into_forward_and_backward(trace, nvfusion)
        replace_nvfusion = {
            "original": nvfusion,
            "forward": fw_nvfusion,
            "backward": bw_nvfusion,
        }
    bwd_trace, tensors_to_save_for_backward = _extract_backward_from_vjp_trace(trace, replace_nvfusion=replace_nvfusion)
    fwd_trace = _extract_forward_from_vjp_trace(trace, tensors_to_save_for_backward, replace_nvfusion=replace_nvfusion)
    return fwd_trace, bwd_trace


def find_common_nvfusion_consumer(
    trace: TraceCtx,
):
    """Find a common nvFusion of forward and backward traces.

    Args:
        trace: Trace of a function with VJP transform applied

    Returns:
        A :class:``BoundSymbol`` representing the common nvFusion.
    """
    # We are searching for a common nvFusion that is supposed to be only in the
    # forward trace, but also depends on the inputs of the backward trace.
    fw_input, _ = tree_flatten(trace.args[0])
    fw_output, _ = tree_flatten(trace.output[0])
    forward_bsyms = tuple(utils.find_producer_symbols(trace, fw_output, fw_input))
    bw_input, _ = tree_flatten(trace.args[1])
    filter_fn = lambda bsym: bsym.sym.name.startswith("nvFusion") and any(
        x.name == y.name for x, y in product(bsym._flat_args, bw_input)
    )
    common_consumers = tuple(filter(filter_fn, forward_bsyms))
    if common_consumers:
        utils.check(
            len(common_consumers) == 1,
            lambda: "Only the case of one common nvFusion between forward and backward is supported.",
        )
        return common_consumers[0]
    return None


def _subtrace_from_bsym(
    bsym: BoundSymbolInterface,
) -> TraceCtx:
    """Create a trace from a :class:``BoundSymbol``.

    Args:
        trace: Trace of a function with VJP transform applied
        bsym: A :class:``BoundSymbol``

    Returns:
        A :class:``TraceCtx`` representing the trace of the function represented by ``bsym``.
    """
    trace = TraceCtx(bsym.sym.__call__)
    trace.args = bsym.args
    trace.kwargs = bsym.kwargs
    trace.output = bsym.output
    # Current version of splitting function requires use of the python_return primitive at the end of the trace.
    trace.bound_symbols = tuple(bsym.subsymbols) + (prims.python_return.bind(*bsym.output, output=()),)
    return trace


def split_nvfusion_into_forward_and_backward(
    trace: TraceCtx,
    nvfusion: BoundSymbolInterface,
) -> tuple[BoundSymbolInterface, BoundSymbolInterface]:
    """Split a common nvFusion into forward and backward nvFusions.

    Args:
        trace: Trace of a function with VJP transform applied
        common_nvfusion: A :class:``BoundSymbol`` representing the common nvFusion.

    Returns:
        A tuple of two :class:``BoundSymbol``s representing the forward and backward nvFusions.
    """
    input1, _ = tree_flatten(trace.args[1])
    output0, _ = tree_flatten(trace.output[0])

    # We want to remove input1 from the given nvFusion.
    # First we find the specific subset of input1 that is used in the nvFusion.
    # Then we remove the subset from the nvFusion.
    nvfusion_input1 = tuple(x for x in input1 if x.name in (a.name for a in nvfusion._flat_args))
    nvfusion_input0 = tuple(x for x in nvfusion._flat_args if x.name not in (y.name for y in nvfusion_input1))
    nvfusion_output0 = tuple(x for x in nvfusion.output if x.name in (y.name for y in output0))
    nvfusion_output1 = tuple(x for x in nvfusion.output if x.name not in (y.name for y in nvfusion_output0))

    nvfusion_trace = _subtrace_from_bsym(nvfusion)
    nvfusion_trace.args = (nvfusion_input0, nvfusion_input1)
    nvfusion_trace.output = (nvfusion_output0, nvfusion_output1)
    fw_nvfusion_trace, bw_nvfusion_trace = split_vjp_trace_into_forward_and_backward(nvfusion_trace)

    fw_fusion_input = tree_flatten(fw_nvfusion_trace.args)[0]
    fw_fusion_output = tuple(
        x for x in tree_flatten(fw_nvfusion_trace.output)[0] if x.name not in (y.name for y in fw_fusion_input)
    )
    fw_fusion_output = tuple(sorted({x.name: x for x in fw_fusion_output}.values(), key=lambda x: x.name))
    fw_nvfusion = replace(
        nvfusion,
        args=fw_fusion_input,
        output=fw_fusion_output,
        subsymbols=tuple(x for x in fw_nvfusion_trace.bound_symbols if x.sym.id != prims.PrimIDs.RETURN),
    )

    bw_nvfusion = replace(
        nvfusion,
        args=tree_flatten(bw_nvfusion_trace.args)[0],
        output=tree_flatten(bw_nvfusion_trace.output)[0],
        subsymbols=tuple(x for x in bw_nvfusion_trace.bound_symbols if x.sym.id != prims.PrimIDs.RETURN),
    )

    # These nvFusions were constructed with naive splitting, so we might need to
    # update them with rematerialization.
    fw_bw_trace = from_trace(nvfusion_trace)
    fw_bw_trace.bound_symbols = [fw_nvfusion, bw_nvfusion, prims.python_return.bind(*nvfusion_trace.output, output=())]
    cut = find_cut(fw_bw_trace, fw_nvfusion, bw_nvfusion)
    if cut:
        original_fw_nvfusion = fw_nvfusion
        fw_nvfusion = apply_rematerialization_for_producer(fw_bw_trace, original_fw_nvfusion, bw_nvfusion, cut)
        bw_nvfusion = apply_rematerialization_for_consumer(original_fw_nvfusion, bw_nvfusion, cut)

    fw_nvfusion = _update_nvfusion_call_ctx(trace, fw_nvfusion)
    bw_nvfusion = _update_nvfusion_call_ctx(trace, bw_nvfusion)
    return fw_nvfusion, bw_nvfusion


def _extract_forward_from_vjp_trace(
    trace: TraceCtx,
    tensors_to_save_for_backward: list[TensorProxy],
    replace_nvfusion=None,
) -> TraceCtx:
    """Extract bound symbols defining forward computation from joint trace.

    The generated trace takes ``primals`` as its positional arguments and kwargs, if exists.
    This raises an exception if the trace is not splittable without the forward depending on any of cotangents.
    """
    fwd_trace = from_trace(trace)

    cotangent_args = (a.name for a in trace.args[1])

    fwd_outputs, _ = tree_flatten(fwd_trace.output[0])
    fwd_trace.output = fwd_outputs
    # note(crcrpar): Cotangents are required for unsplittable joint trace. Otherwise, infinite loop.
    # https://github.com/Lightning-AI/lightning-thunder/commit/20505d9772786ac43b1a47811a6846c166f87022
    # seems to fix the infinite loop.
    forward_bsyms = list(
        {
            v: v
            for v in utils.find_producer_symbols(
                trace, fwd_outputs, tree_flatten((fwd_trace.args, fwd_trace.kwargs))[0]
            )
        }.keys()
    )

    if replace_nvfusion is not None:
        utils.check(
            replace_nvfusion["original"] in forward_bsyms,
            lambda: "Something went wrong. The common nvFusion is not included in the forward trace.",
        )
        forward_bsyms = [
            replace_nvfusion["forward"] if bsym == replace_nvfusion["original"] else bsym for bsym in forward_bsyms
        ]

    # now that we have collected bound symbols of forward computation, make sure they don't depend on any cotangents
    all_tensor_arg_names = {t.name for bsym in forward_bsyms for t in bsym._flat_args if isinstance(t, TensorProxy)}
    used_cotangent_args = tuple(a for a in cotangent_args if a in all_tensor_arg_names)
    if used_cotangent_args:
        raise RuntimeError(
            f"Impossible to split the trace due to the dependency on cotangents of {used_cotangent_args}"
        )

    fwd_trace.args = tree_flatten((fwd_trace.args[0], fwd_trace.kwargs))[0]
    fwd_trace.kwargs = {}
    fwd_trace.bound_symbols = forward_bsyms + [prims.python_return.bind(*fwd_outputs, output=())]

    fwd_trace._siginfo = None

    def forward_fn(*args):
        pass

    fwd_trace.fn = forward_fn
    fwd_siginfo = fwd_trace.siginfo()
    fwd_siginfo.args = [(a.name, a) for a in fwd_trace.args]
    fwd_siginfo.varargs = None
    fwd_siginfo.kwargs = {}

    fwd_outputs, fwd_outputs_spec = tree_flatten(trace.output[0])
    fwd_trace.output = fwd_trace.output, tensors_to_save_for_backward
    fwd_trace.bound_symbols[-1].args = (
        tree_unflatten(fwd_trace.bound_symbols[-1].args, fwd_outputs_spec),
        tensors_to_save_for_backward,
    )
    return fwd_trace


def _extract_backward_from_vjp_trace(
    trace: TraceCtx,
    replace_nvfusion=None,
) -> tuple[TraceCtx, list[TensorProxy]]:
    """Extract backward definition from joint trace.

    The generated trace, the first of two return values, defines the backward computation.
    Its arguments follow the structure of ``*saved_for_backward, *cotangents`` for the sake of
    easy embedding into :class:``torch.autograd.Function``.
    The second return value, a list of :class:``TensorProxy``s, is a list of intermediate and output
    ``Tensor``s that will be used in the backward.
    """
    bwd_trace = from_trace(trace)
    bwd_trace.bound_symbols = trace.bound_symbols
    result, grads = bwd_trace.output
    result, _ = tree_flatten(result)

    forward_args, cotangents = bwd_trace.args
    flat_args, _ = tree_flatten((bwd_trace.args, bwd_trace.kwargs))

    forward_bsyms = list({v: v for v in utils.find_producer_symbols(bwd_trace, result, flat_args)}.keys())

    bound_symbols_for_backward = []
    # It's assumed that the common nvFusion is not included in the constructed
    # backward trace.
    if replace_nvfusion is not None:
        utils.check(
            replace_nvfusion["original"] in forward_bsyms,
            lambda: "Something went wrong. The common nvFusion is not included in the forward trace.",
        )
        forward_bsyms = [
            replace_nvfusion["forward"] if bsym == replace_nvfusion["original"] else bsym for bsym in forward_bsyms
        ]
        bound_symbols_for_backward += [replace_nvfusion["backward"]]

    forward_intermediate_map = {out.name: out for bsym in forward_bsyms for out in bsym._flat_outs}
    for r in result:
        if r.name not in forward_intermediate_map:
            forward_intermediate_map[r.name] = r
    for a in tree_flatten((bwd_trace.args[1]))[0]:
        if a.name in forward_intermediate_map:
            del forward_intermediate_map[a.name]
    forward_intermediates = list(forward_intermediate_map.values())

    return_bsym = bwd_trace.bound_symbols[-1]
    return_bsym.args = grads
    bound_symbols_for_backward += [
        bsym for bsym in utils.find_producer_symbols(trace, grads, stop_proxies=flat_args) if bsym not in forward_bsyms
    ] + [return_bsym]

    # Remove all unpack args primitives from the backward trace
    bound_symbols_for_backward = [
        bsym
        for bsym in bound_symbols_for_backward
        if bsym.sym.id
        not in (
            prims.PrimIDs.UNPACK_EMPTY_DICT,
            prims.PrimIDs.UNPACK_KEY,
            prims.PrimIDs.UNPACK_SEQUENCE,
            prims.PrimIDs.UNPACK_TRIVIAL,
        )
    ]

    if replace_nvfusion is not None:
        bound_symbols_for_backward.remove(replace_nvfusion["original"])

    backward_trace = from_trace(trace)
    backward_trace._siginfo = None
    backward_trace.bound_symbols = bound_symbols_for_backward
    consumed_vars = set(utils.consumers(backward_trace)._dict.keys())
    used_forward_intermediate_results = tuple(
        a
        for a in (forward_intermediates + tree_flatten((bwd_trace.args[0], bwd_trace.kwargs))[0])
        if a.name in consumed_vars
    )
    # Remove duplicates
    used_forward_intermediate_results = tuple(
        sorted({x.name: x for x in used_forward_intermediate_results}.values(), key=lambda a: a.name)
    )
    backward_trace.args = tuple(used_forward_intermediate_results + bwd_trace.args[1])
    backward_trace.kwargs = None
    backward_trace.output = grads

    # N.B.(crcrpar): Override `siginfo` so that the trace would spell out arguments.
    # Otherwise, it'd end up `backward_trace(*args)`.
    def backward_fn(*args):
        pass

    backward_trace.fn = backward_fn
    bwd_siginfo = backward_trace.siginfo()
    bwd_siginfo.args = [(a.name, a) for a in bwd_siginfo.varargs[1]]
    bwd_siginfo.varargs = None
    backward_trace._siginfo = bwd_siginfo
    return backward_trace, used_forward_intermediate_results
