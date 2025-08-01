from dataclasses import replace
from functools import partial
from itertools import chain, takewhile
from collections.abc import Callable
from collections.abc import Sequence
from collections import defaultdict
import time

import networkx as nx

from thunder.core import prims, utils
from thunder.core.baseutils import BoundSymbolInterface, ProxyInterface
from thunder.core.proxies import TensorProxy, variableify, NumberProxy, Proxy
from thunder.core.symbol import has_tags
from thunder.core.trace import from_trace, TraceCtx, TraceProvenance
from thunder.core.transforms import bsym_list_to_dag, toposort_bsym_dag, TOPOSORT_ORDER
from thunder.core.transform_common import order_proxies


def find_external_producer_outputs(
    proxy_to_consumers: dict[ProxyInterface, tuple[BoundSymbolInterface, ...]],
    next_consumers: Sequence[BoundSymbolInterface],
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> tuple[ProxyInterface, ...]:
    """Find producer's outputs that must be included in the output of the
    producer because they are used by other consumers.

    Args:
        proxy_to_consumers (dict[ProxyInterface, tuple[BoundSymbolInterface, ...]]): A dictionary that maps a producer's
            output to the consumers that use it.
        next_consumers (Sequence[BoundSymbolInterface]): Other consumers that
            use the producer's output.
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.

    Returns:
        Tuple[ProxyInterface, ...]: Producer's outputs that must be included in
        the output of the producer.
    """
    local_consumer_info = utils.consumers(list(chain((producer, consumer), next_consumers)))

    def is_rematerializable(out: ProxyInterface):
        # First check local information to see if the output is used by other
        # consumers.
        local_consumers = local_consumer_info.get(out, tuple())
        if len(local_consumers) > 1:
            return False

        # If the output is not used by fusion consumers, check global information
        # to see if the output is used by other consumers.
        global_consumers = proxy_to_consumers.get(out, tuple())
        global_consumers = tuple(
            x for x in global_consumers if x.sym is not prims.python_del and x not in chain((consumer,), next_consumers)
        )

        # If the output is used by other global consumers, it's not rematerializable.
        if len(global_consumers) > 0:
            return False

        if len(local_consumers) == 0:
            return True

        # If the output is used by a single local consumer, it's rematerializable
        return len(local_consumers) == 1 and out.name in (x.name for x in consumer.args)

    rematerializable_producer_outputs = tuple(filter(is_rematerializable, producer.output))

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
    all_produced_vars = tuple(chain.from_iterable((y for y in x.flat_proxy_outs) for x in producer.subsymbols))
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
    cut: Sequence[ProxyInterface | str],
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
    new_producer_output_names = (
        tuple(x.name if isinstance(x, ProxyInterface) else x for x in external_producer_outputs) + cut_names
    )
    # Remove the producer's inputs from the new producer's output.
    new_producer_output_names = tuple(
        x for x in new_producer_output_names if x not in (y.name for y in producer.flat_args)
    )
    all_produced_vars = tuple(chain.from_iterable((y for y in x.flat_proxy_outs) for x in producer.subsymbols))
    # Choose the new producer's output from all the produced variables.
    new_producer_output = tuple(x for x in all_produced_vars if x.name in new_producer_output_names)
    proxy_order = order_proxies(producer.subsymbols)
    new_producer_output = tuple(sorted(new_producer_output, key=lambda p: proxy_order[p.name]))
    new_producer = replace(producer, output=new_producer_output)
    return new_producer


def apply_rematerialization_for_consumer(
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
    cut: Sequence[ProxyInterface | str],
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
    all_produced_vars = tuple(chain.from_iterable((y for y in x.flat_proxy_outs) for x in producer.subsymbols))
    cut_names = tuple(map(lambda x: x.name, cut)) if isinstance(cut[0], ProxyInterface) else tuple(cut)
    cut_inputs = tuple(filter(lambda x: x.name in cut_names, (*all_produced_vars, *producer.args)))
    new_consumer_args = cut_inputs + external_inputs

    # We need to rematerialize the consumer's inputs that are not in the new consumer's inputs.
    rematerialized_inputs = tuple(
        filter(lambda x: x.name not in map(lambda x: x.name, new_consumer_args), consumer.args)
    )

    # In the case where there are no tensors to rematerialize it is
    # possible to terminate early and return the consumer as it was.
    if not rematerialized_inputs:
        return consumer

    # Construct a temporary Trace object with subsymbols from the producer.
    trace = TraceCtx(None)
    trace.bound_symbols = producer.subsymbols

    recomputing_symbols = utils.find_producer_symbols(trace, rematerialized_inputs, cut_inputs)
    new_subsymbols = recomputing_symbols + tuple(consumer.subsymbols)

    # Some recomputing_symbols might require producer's inputs, so we need to
    # add them to the consumer's inputs.
    # Probably find_min_cut should have returned this information.
    all_args = set(chain.from_iterable((x.name for x in bsym.flat_proxy_args) for bsym in new_subsymbols))
    all_outs = set(chain.from_iterable((x.name for x in bsym.flat_proxy_outs) for bsym in new_subsymbols))
    new_consumer_args += tuple(
        x
        for x in producer.args
        if x.name in all_args and x.name not in (x.name for x in new_consumer_args) and x.name not in all_outs
    )

    # The recomputing_symbols may originate from multiple producers.
    # Directly adding these symbols at the beginning of the consumer can disrupt the topological order of subsymbols. To ensure
    # correct execution order, we reorder the new_subsymbols here.
    _, leaves = bsym_list_to_dag(list(new_subsymbols))
    new_subsymbols = toposort_bsym_dag(leaves, TOPOSORT_ORDER.BOTTOM_UP)
    proxy_order = order_proxies(new_subsymbols)
    new_consumer_args = tuple(sorted(new_consumer_args, key=lambda x: proxy_order[x.name]))
    new_consumer = replace(consumer, args=new_consumer_args, subsymbols=new_subsymbols)
    return new_consumer


def find_filtered_producer_consumer_pairs(
    trace: TraceCtx,
    filter_func: Callable | None = None,
    *,
    proxy_to_consumers=None,
) -> tuple[tuple[BoundSymbolInterface, BoundSymbolInterface], ...]:
    """Find producer-consumer pairs among the filtered symbols.

    Args:
        trace (TraceCtx): Trace object.
        filter_func (Optional[Callable], optional): Filter function. Defaults to None.

    Returns:
        Tuple[Tuple[BoundSymbolInterface, BoundSymbolInterface], ...]: Producer-consumer bound symbol pairs.
    """
    filter_func = filter_func or (lambda x: True)
    proxy_to_consumers = utils.consumers(trace) if proxy_to_consumers is None else proxy_to_consumers
    producer_consumer_pairs = set()
    order_in_trace = {bsym: i for i, bsym in enumerate(filter(filter_func, trace.bound_symbols))}

    # We are looking for special producer-consumer pairs among the filtered symbols
    for producer in filter(filter_func, trace.bound_symbols):
        for out in producer.flat_outs:
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
find_fusion_producer_consumer_pairs = partial(
    find_filtered_producer_consumer_pairs, filter_func=lambda x: x.sym.is_fusion
)


def find_cut(
    external_producer_outputs: Sequence[ProxyInterface],
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> Sequence[ProxyInterface | str]:
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

    # This is needed to avoid rematerializing random or expensive primitives.
    tags = {prims.OpTags.REDUCTION_OP, prims.OpTags.RANDOM_OP, prims.OpTags.MATMUL_OP}
    required_producer_vars += tuple(
        chain.from_iterable((y for y in x.flat_outs) for x in producer.subsymbols if has_tags(x, tags))
    )

    # We can apply rematerialization for any pair of symbols with is_fusion=True
    # property. Currently this could be an nvFuser or a TorchCompile fusion.
    # These executors might have a different coverage of supported operators. So
    # we need to mark unsupported by consumer operators variables as required
    # producer variables. So that we don't move them to the consumer.
    if producer.sym.executor != consumer.sym.executor:
        required_producer_vars += tuple(
            chain.from_iterable(
                (y for y in x.flat_outs)
                for x in producer.subsymbols
                if not has_tags(x, tags) and not consumer.sym.executor.can_fuse(x)
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
    # Filtering out the Nones coming from update_aliases + DCE
    required_consumer_vars += tuple(
        chain.from_iterable((y.name for y in x.flat_outs if y is not None) for x in required_consumer_symbols)
    )

    # If there is overlap between required consumer and producer variables,
    # it's impossible to find a valid mincut, so return empty tuple
    if any(x.name in required_consumer_vars for x in required_producer_vars):
        return tuple()

    # TODO: Use TensorProxy properties to compute the weights
    WEIGHT = 1.0

    # Create a graph
    edges = []

    def add_edge(src, dst, capacity):
        edges.append((src, dst, {"capacity": capacity}))

    utils.check(
        len(required_consumer_vars) > 0,
        lambda: "The consumer has no outputs. This is not supported by the cut finding algorithm.",
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
        elif isinstance(var, NumberProxy):
            return 0.0
        return WEIGHT

    def add_edges(var):
        var_name = var.name
        weight = get_weight(var)
        weight = weight / 2.0 if var_name in (x.name for x in producer.args) else weight
        add_edge(var_name + "_in", var_name + "_out", capacity=weight)
        for user in combined_consumers._dict.get(var_name, tuple()):
            if user.sym.id in sym_skip_list:
                continue
            for out in user.flat_proxy_outs:
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
        for var in symbol.flat_proxy_outs:
            add_edges(var)

    g = nx.DiGraph()
    g.add_edges_from(edges)

    try:
        _, (reachable, non_reachable) = nx.minimum_cut(g, "source", "sink")
    except Exception:
        raise RuntimeError(
            "Failed to compute the min-cut on the graph due to a path with infinite capacity."
            "Please report this error along with your function and relevant details at: https://github.com/Lightning-AI/lightning-thunder/issues/new"
        )

    cut_edges = set()
    for u, nbrs in ((n, g[n]) for n in reachable):
        cut_edges.update((u, v) for v in nbrs if v in non_reachable)

    cut_nodes = set()
    for node_in, node_out in cut_edges:
        if node_out == "sink":
            continue
        assert node_in.endswith("_in"), node_in
        assert node_out.endswith("_out"), node_out
        assert node_in[:-3] == node_out[:-4]
        var_name = node_in[:-3]
        cut_nodes.add(var_name)
    return tuple(sorted(cut_nodes))


def rematerialize(trace: TraceCtx) -> TraceCtx:
    """Rematerialize the trace.

    Args:
        trace (TraceCtx): Trace object.

    Returns:
        TraceCtx: Rematerialized trace and the list of
            rematerialized traces.
    """
    start_time_ns = time.perf_counter_ns()

    static_consumer_info = utils.consumers(trace)

    # Find all the producers and consumers
    pairs = find_fusion_producer_consumer_pairs(trace, proxy_to_consumers=static_consumer_info)

    # Pairs of producer and consumer are not unique. Each update to the producer
    # or consumer may affect the other. We need to update the producer and
    # consumer sequentially.
    producers = {producer for producer, _ in pairs}
    consumers = {consumer for _, consumer in pairs}
    new_bsyms = {bsym: bsym for bsym in producers | consumers}
    computed_cuts_for_producers = defaultdict(tuple)
    for i, (producer, consumer) in enumerate(pairs):
        current_producer = new_bsyms.get(producer, None) or producer
        current_consumer = new_bsyms.get(consumer, None) or consumer
        # Determine which producer's outputs cannot be rematerialized
        next_consumers = takewhile(lambda x: x[0] == producer, pairs[i + 1 :])
        next_consumers = tuple(consumer for _, consumer in next_consumers)
        next_consumers = tuple(new_bsyms.get(bsym, bsym) for bsym in next_consumers)
        external_producer_outputs = find_external_producer_outputs(
            static_consumer_info, next_consumers, current_producer, current_consumer
        )
        # Find the minimal cut between the producer and the consumer
        cut = find_cut(external_producer_outputs, current_producer, current_consumer)
        if cut:
            # If we have already computed the cut for the producer, we need to
            # update the external producer outputs with the previous cut
            # information.
            external_producer_outputs += computed_cuts_for_producers.get(producer, tuple())

            updated_producer = apply_rematerialization_for_producer(external_producer_outputs, current_producer, cut)
            updated_consumer = apply_rematerialization_for_consumer(current_producer, current_consumer, cut)
            # As we replace bound symbols of the input trace with updated ones every iteration,
            # we should keep track of the map of `current` to `updated` as well as `producer`/`consumer`
            # to `updated` ones.
            # ref: https://github.com/Lightning-AI/lightning-thunder/pull/868#discussion_r1305640813
            new_bsyms[producer] = new_bsyms[current_producer] = updated_producer
            new_bsyms[consumer] = new_bsyms[current_consumer] = updated_consumer

            computed_cuts_for_producers[producer] += cut

    rematerialized_trace = from_trace(trace)
    rematerialized_trace.bound_symbols = tuple(new_bsyms.get(bsym, bsym) for bsym in trace.bound_symbols)

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000

    rematerialized_trace.set_provenance(TraceProvenance(f"Rematerialization (took {elapsed_time_millis} milliseconds)"))
    return rematerialized_trace


def replace_uniform(trace: TraceCtx) -> TraceCtx:
    """For better rematerialization, replace the uniform operator with the stateless uniform_philox operator and manually update the RNG state."""
    from thunder.core.compile_data import get_compile_option

    disable_replace_uniform: None | bool = get_compile_option(
        "disable_replace_uniform", "Disables the replace_uniform transform to avoid dropout rematerialization"
    )
    if disable_replace_uniform:
        return trace

    start_time_ns = time.perf_counter_ns()
    from thunder.core.trace import VariableInterface
    from thunder.core.devices import Device
    from thunder.core.transforms import VISIT_TYPE, visitor_transform

    swapmap: dict[VariableInterface, Proxy] = {}
    prev_state: dict[Device, Proxy] = {}

    def visit_(bsym: BoundSymbolInterface) -> VISIT_TYPE:
        if bsym.sym.id == prims.PrimIDs.UNIFORM:
            dev = bsym.kwargs["device"]
            if dev not in prev_state:
                seed, offset = prims.get_and_update_rng_state(None, None, dev)
            else:
                seed, offset = prims.get_and_update_rng_state(*prev_state[dev], dev)
            out = prims.uniform_philox(*bsym.args, **bsym.kwargs, seed=seed, offset=offset)
            new_vo = variableify(out)
            swapmap[new_vo] = bsym.output
            prev_state[dev] = [seed, offset]
            return VISIT_TYPE.REPLACE
        return VISIT_TYPE.NO_OP

    new_trace = visitor_transform(trace, visit_)

    bound_symbols: list[BoundSymbolInterface] = []
    for bsym in new_trace.bound_symbols:
        nbsym: BoundSymbolInterface = bsym.from_bsym_swap_proxies(swapmap)
        bound_symbols.append(nbsym)

    new_trace.bound_symbols = bound_symbols

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    new_trace.set_provenance(
        TraceProvenance(f"Transform for replace uniform (took {elapsed_time_millis} milliseconds)")
    )
    return new_trace
