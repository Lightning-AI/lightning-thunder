from thunder.core.trace import from_trace
from thunder.core.transforms import bsym_list_to_dag, Node, toposort_bsym_dag, TOPOSORT_ORDER
from thunder.core.utils import check
from thunder.distributed.prims import PrimIDs


def sort_data_parallel_syncs(primal_trace):
    """
    Sorts the data parallel syncs in the primal trace to be as close to the
    consumer as possible. This is done to help data locality both for the
    forward pass and the backward pass.

    Args:
        primal_trace (TraceCtx): The primal trace to sort.

    Returns:
        TraceCtx: The sorted primal trace.
    """

    order_in_trace = {bsym: i for i, bsym in enumerate(primal_trace.bound_symbols)}

    def prefer_everything_over_syncs(eligible_nodes: list[Node]) -> int:
        # Prefer other nodes over "synchronize"
        def key(node: Node) -> int:
            match node.bsym.sym.id:
                case PrimIDs.SYNCHRONIZE:
                    return len(order_in_trace) + 1
                case _:
                    # Prefer nodes that are earlier in the trace
                    return order_in_trace[node.bsym]

        return min(range(len(eligible_nodes)), key=lambda i: key(eligible_nodes[i]))

    new_primal_trace = from_trace(primal_trace)

    if any(bsym.sym.id == PrimIDs.SYNCHRONIZE for bsym in primal_trace.bound_symbols):
        new_primal_trace.bound_symbols = toposort_bsym_dag(
            bsym_list_to_dag(primal_trace.bound_symbols)[0],
            TOPOSORT_ORDER.TOP_DOWN,
            selector=prefer_everything_over_syncs,
        )
        return new_primal_trace
    else:
        return primal_trace


# TODO: Currently prefer the most memory-efficient way for ZeRO3,
# https://github.com/Lightning-AI/lightning-thunder/issues/1925
# Need a strategy to balance the efficiency
# and memory usage in the future
def sort_waits_for_zero3(execution_trace):
    """
    Sorts the wait_prim_impl nodes in the execution trace to be as far from the
    communication ops as possible, except for the all_gather_prim_impl nodes, the all_gather_prim_impl nodes
    are sorted to be next to wait_prim_impl node to reduce the peak allocated memory

    Args:
        execution_trace (TraceCtx): The execution trace to sort.

    Returns:
        TraceCtx: The sorted execution trace.
    """
    from thunder.executors.torchex import (
        wait_prim_impl,
        reduce_scatter_prim_impl,
        all_reduce_prim_impl,
        all_gather_prim_impl,
    )

    if not any(bsym.sym.id == wait_prim_impl.id for bsym in execution_trace.bound_symbols):
        return execution_trace

    order_in_trace = {bsym: i for i, bsym in enumerate(execution_trace.bound_symbols)}

    def prefer_comm_over_other_over_wait_over_allgather(eligible_nodes: list[Node]) -> int:
        # Prefer communication ops other than "all_gather_prim_impl" over other nodes and prefer other
        # nodes over "wait_prim_impl", pick "all_gather_prim_impl" last.
        def key(node: Node) -> int:
            match node.bsym.sym.id:
                case (wait_prim_impl.id):
                    return len(order_in_trace)
                case (reduce_scatter_prim_impl.id | all_reduce_prim_impl.id):
                    # Prefer larger communication ops over smaller ones
                    return -node.bsym.args[0].numel
                case (all_gather_prim_impl.id):
                    return len(order_in_trace) + order_in_trace[node.bsym]
                case _:
                    # Prefer nodes that are earlier in the trace
                    return order_in_trace[node.bsym]

        return min(range(len(eligible_nodes)), key=lambda i: key(eligible_nodes[i]))

    new_execution_trace = from_trace(execution_trace)

    # TODO: This pass doesn't behave correctly if del nodes are present in the trace
    check(
        not any(bsym.sym.name == "del" for bsym in execution_trace.bound_symbols),
        lambda: "Cannot sort execution trace with del nodes",
    )
    new_execution_trace.bound_symbols = toposort_bsym_dag(
        bsym_list_to_dag(execution_trace.bound_symbols)[0],
        TOPOSORT_ORDER.TOP_DOWN,
        selector=prefer_comm_over_other_over_wait_over_allgather,
    )
    return new_execution_trace


def sort_waits(execution_trace):
    """
    Sorts the wait_prim_impl nodes in the execution trace to be as far from the
    communication ops as possible. This is done to overlap communication with
    computation as much as possible.

    Args:
        execution_trace (TraceCtx): The execution trace to sort.

    Returns:
        TraceCtx: The sorted execution trace.
    """
    from thunder.executors.torchex import (
        wait_prim_impl,
        reduce_scatter_prim_impl,
        all_reduce_prim_impl,
        all_gather_prim_impl,
    )

    if not any(bsym.sym.id == wait_prim_impl.id for bsym in execution_trace.bound_symbols):
        return execution_trace

    order_in_trace = {bsym: i for i, bsym in enumerate(execution_trace.bound_symbols)}

    def prefer_comm_over_other_over_wait(eligible_nodes: list[Node]) -> int:
        # Prefer communication ops over other nodes and prefer other
        # nodes over "wait_prim_impl"
        def key(node: Node) -> int:
            match node.bsym.sym.id:
                case (wait_prim_impl.id):
                    return len(order_in_trace)
                case (reduce_scatter_prim_impl.id | all_reduce_prim_impl.id | all_gather_prim_impl.id):
                    # Prefer larger communication ops over smaller ones
                    return -node.bsym.args[0].numel
                case _:
                    # Prefer nodes that are earlier in the trace
                    return order_in_trace[node.bsym]

        return min(range(len(eligible_nodes)), key=lambda i: key(eligible_nodes[i]))

    new_execution_trace = from_trace(execution_trace)

    # TODO: This pass doesn't behave correctly if del nodes are present in the trace
    check(
        not any(bsym.sym.name == "del" for bsym in execution_trace.bound_symbols),
        lambda: "Cannot sort execution trace with del nodes",
    )
    new_execution_trace.bound_symbols = toposort_bsym_dag(
        bsym_list_to_dag(execution_trace.bound_symbols)[0],
        TOPOSORT_ORDER.TOP_DOWN,
        selector=prefer_comm_over_other_over_wait,
    )
    return new_execution_trace
