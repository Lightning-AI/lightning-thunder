from dataclasses import dataclass, replace
from functools import partial
from itertools import chain, product
from typing import Callable, Optional, Tuple, Union
from collections.abc import Sequence
import time

from igraph import Graph

from thunder.core import prims, utils
from thunder.core.baseutils import BoundSymbolInterface, ProxyInterface
from thunder.core.prims import PrimIDs
from thunder.core.proxies import TensorProxy
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.symbol import has_tags
from thunder.core.trace import from_trace, TraceCtx, TraceProvenance


def find_external_producer_outputs(
    trace: TraceCtx,
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> tuple[ProxyInterface, ...]:
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
        if len(consumers) == 0:
            return True
        consumers = tuple(filter(lambda x: x.sym.name != "del", consumers))
        return len(consumers) == 1 and out.name in (x.name for x in consumer.args)

    rematerializable_producer_outputs = tuple(filter(filter_func, producer.output))

    return tuple(x for x in producer.output if x.name not in (y.name for y in rematerializable_producer_outputs))


def find_external_consumer_inputs(
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> tuple[ProxyInterface, ...]:
    """Find consumer's inputs that must be included in the input of the
    consumer because they are produced by other producers.

    Args:
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.

    Returns:
        Tuple[ProxyInterface, ...]: Consumer's inputs that must be included in
        the input of the consumer.
    """
    all_produced_vars = tuple(chain.from_iterable((y for y in x._flat_outs) for x in producer.subsymbols))
    external_consumer_inputs_names = tuple(
        sorted(
            {x.name for x in consumer.args}
            - {x.name for x in producer.output}
            - {x.name for x in producer.args}
            - {x.name for x in all_produced_vars}
        )
    )
    return tuple(x for x in consumer.args if x.name in external_consumer_inputs_names)


def apply_rematerialization_for_producer(
    external_producer_outputs,
    producer: BoundSymbolInterface,
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
) -> tuple[tuple[BoundSymbolInterface, BoundSymbolInterface], ...]:
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
    order_in_trace = {
        bsym: i for i, bsym in enumerate(filter(filter_func, trace.bound_symbols))
    }

    # We are looking for special producer-consumer pairs among the filtered symbols
    for producer in filter(filter_func, trace.bound_symbols):
        for out in producer._flat_outs:
            consumers = proxy_to_consumers.get(out, tuple())
            consumers = filter(filter_func, consumers)
            for consumer in consumers:
                producer_consumer_pairs.add((producer, consumer))
    return tuple(
        sorted(
            producer_consumer_pairs,
            key=lambda pair: (order_in_trace[pair[0]], order_in_trace[pair[1]]),
        )
    )


find_nvfuser_producer_consumer_pairs = partial(
    find_filtered_producer_consumer_pairs, filter_func=lambda x: x.sym.name.startswith("nvFusion")
)


def find_cut(
    external_producer_outputs: Sequence[ProxyInterface],
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

    # Required producer variables. These are the variables that are required to
    # be connected to the "source" node.
    required_producer_vars = tuple(x for x in producer.args)
    required_producer_vars += tuple(x for x in external_producer_outputs)

    # This is needed to avoid rematerializing random or reduction primitives.
    tags = {prims.OpTags.REDUCTION_OP, prims.OpTags.RANDOM_OP}
    required_producer_vars += tuple(
        chain.from_iterable(
            (y for y in x._flat_outs) for x in producer.subsymbols if has_tags(x, tags)
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
        edges.append((
            name_to_id.setdefault(src, len(name_to_id)),
            name_to_id.setdefault(dst, len(name_to_id))
        ))
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

    id_to_name = dict(map(reversed, name_to_id.items()))

    g_edges = g.get_edgelist()
    cut = g.mincut(source, sink, "capacity").cut
    cut_nodes = set()
    for cut_edge_id in cut:
        u, v = g_edges[cut_edge_id]
        node_in, node_out = id_to_name[u], id_to_name[v]
        if node_out == "sink":
            continue
        assert node_in.endswith("_in")
        assert node_out.endswith("_out")
        assert node_in[:-3] == node_out[:-4]
        var_name = node_in[:-3]
        cut_nodes.add(var_name)
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
    import copy
    start_time_ns = time.time_ns()

    rematerialized_trace = from_trace(trace)
    rematerialized_trace.bound_symbols = copy.copy(trace.bound_symbols)

    def replace_bsym_and_update_call_ctx(bsym, new_bsyms):
        new_bsym = new_bsyms.get(bsym, None)
        if new_bsym is not None:
            return _update_nvfusion_call_ctx(rematerialized_trace, new_bsym)
        return bsym

    # Find all the producers and consumers
    pairs = find_nvfuser_producer_consumer_pairs(rematerialized_trace)

    # Pairs of producer and consumer are not unique. Each update to the producer
    # or consumer may affect the other. We need to update the producer and
    # consumer sequentially.
    producers = {producer for producer, _ in pairs}
    consumers = {consumer for _, consumer in pairs}
    new_bsyms = {bsym: bsym for bsym in producers | consumers}
    for producer, consumer in pairs:
        current_producer = new_bsyms.get(producer, None) or producer
        current_consumer = new_bsyms.get(consumer, None) or consumer
        # Determine which producer's outputs cannot be rematerialized
        external_producer_outputs = find_external_producer_outputs(rematerialized_trace, current_producer, current_consumer)
        # Find the minimal cut between the producer and the consumer
        cut = find_cut(external_producer_outputs, current_producer, current_consumer)
        if cut:
            updated_producer = apply_rematerialization_for_producer(
                external_producer_outputs, current_producer, cut
            )
            updated_consumer = apply_rematerialization_for_consumer(current_producer, current_consumer, cut)
            # As we replace bound symbols of the input trace with updated ones every iteration,
            # we should keep track of the map of `current` to `updated` as well as `producer`/`consumer`
            # to `updated` ones.
            # ref: https://github.com/Lightning-AI/lightning-thunder/pull/868#discussion_r1305640813
            new_bsyms[producer] = new_bsyms[current_producer] = updated_producer
            new_bsyms[consumer] = new_bsyms[current_consumer] = updated_consumer
        else:
            new_bsyms[producer] = None
            new_bsyms[consumer] = None

        # We need to update the trace with the new producer and consumer so that
        # the next iteration can use the updated bound symbols for
        # `find_external_producer_outputs` function.
        rematerialized_trace.bound_symbols = tuple(
            new_bsyms.get(bsym, bsym) for bsym in rematerialized_trace.bound_symbols
        )

    # New bound symbols are still incorrect. Its _ctx_call dict points to the
    # old nvFuser fusion. We need to update it to use the new definition.
    rematerialized_trace.bound_symbols = tuple(
        replace_bsym_and_update_call_ctx(bsym, new_bsyms) for bsym in trace.bound_symbols
    )

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000

    rematerialized_trace.set_provenance(TraceProvenance(f"Rematerialization (took {elapsed_time_millis} milliseconds)"))
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
        a
        for a in all_args
        if producers.get(a, None) is None
        and a.name not in (y.name for y in tree_flatten(bw_trace.args[1])[0] if isinstance(y, ProxyInterface))
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
