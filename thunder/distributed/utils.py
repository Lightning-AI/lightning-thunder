from __future__ import annotations
from typing import TYPE_CHECKING

from thunder.core.trace import from_trace
from thunder.core.transforms import bsym_list_to_dag, Node, toposort_bsym_dag, TOPOSORT_ORDER
from thunder.core.utils import check
from thunder.distributed.prims import PrimIDs

if TYPE_CHECKING:
    from thunder.core.symbol import BoundSymbol
    from thunder.core.trace import TraceCtx


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
        unpack_for_fsdp_prim_impl,
    )

    if not any(bsym.sym.id == wait_prim_impl.id for bsym in execution_trace.bound_symbols):
        return execution_trace

    order_in_trace = {bsym: i for i, bsym in enumerate(execution_trace.bound_symbols)}

    def prefer_comm_over_other_over_wait_over_allgather(eligible_nodes: list[Node]) -> int:
        # Prefer communication ops other than "all_gather_prim_impl" over other nodes and prefer other
        # nodes over "wait_prim_impl", pick "all_gather_prim_impl" last.
        def key(node: Node) -> int:
            match node.bsym.sym.id:
                case wait_prim_impl.id | unpack_for_fsdp_prim_impl.id:
                    return len(order_in_trace)
                case reduce_scatter_prim_impl.id | all_reduce_prim_impl.id:
                    # Prefer larger communication ops over smaller ones
                    return -node.bsym.args[0].numel
                case all_gather_prim_impl.id:
                    return len(order_in_trace) + order_in_trace[node.bsym]
                case _:
                    # Prefer nodes that are earlier in the trace
                    return order_in_trace[node.bsym]

        return max(range(len(eligible_nodes)), key=lambda i: key(eligible_nodes[i]))

    new_execution_trace = from_trace(execution_trace)

    # TODO: This pass doesn't behave correctly if del nodes are present in the trace
    check(
        not any(bsym.sym.name == "del" for bsym in execution_trace.bound_symbols),
        lambda: "Cannot sort execution trace with del nodes",
    )
    new_execution_trace.bound_symbols = toposort_bsym_dag(
        bsym_list_to_dag(execution_trace.bound_symbols)[1],
        TOPOSORT_ORDER.BOTTOM_UP,
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
                case wait_prim_impl.id:
                    return len(order_in_trace)
                case reduce_scatter_prim_impl.id | all_reduce_prim_impl.id | all_gather_prim_impl.id:
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


def limit_in_flight_allgathers(
    execution_trace: TraceCtx,
    max_in_flight_comms: int,
    bucketing_enabled: bool,
) -> TraceCtx:
    from collections import deque

    from thunder.core import utils
    from thunder.executors.torchex import (
        all_gather_prim_impl,
        pack_for_fsdp_prim_impl,
        unpack_for_fsdp_prim_impl,
        wait_prim_impl,
    )

    new_execution_trace = from_trace(execution_trace)
    orig_bound_symbols: list[BoundSymbol] = execution_trace.bound_symbols
    pack_bsyms = []
    allgather_bsyms = []
    wait_bsyms = []
    unpack_bsyms = []
    # record all the bound symbols except packs+allgathers in order
    bound_symbols: list[BoundSymbol] = []
    producers, consumers = utils.producers_and_consumers(execution_trace)
    for i, bsym in enumerate(orig_bound_symbols):
        match bsym.sym.id:
            case pack_for_fsdp_prim_impl.id:
                pack_consumer = consumers.get(bsym.flat_proxy_outs[0], None)
                check(
                    pack_consumer is not None and len(pack_consumer) == 1,
                    lambda: f"Pack operator should have one consumer",
                )
                # skip the pack operator corresponds to allgather
                if pack_consumer[0].sym.id != all_gather_prim_impl.id:
                    bound_symbols.append(bsym)
            case all_gather_prim_impl.id:
                allgather_bsyms.append(bsym)
                if bucketing_enabled:
                    expected_pack = producers[bsym.flat_proxy_args[0]]
                    check(
                        expected_pack.sym.id == pack_for_fsdp_prim_impl.id,
                        lambda: f"The producer of allgather should be pack operator (but got {expected_pack})",
                    )
                    pack_bsyms.append(expected_pack)
            case wait_prim_impl.id:
                if producers[bsym.flat_proxy_args[0]].sym.id == all_gather_prim_impl.id:
                    wait_bsyms.append(bsym)
                    if bucketing_enabled:
                        wait_consumer = consumers.get(bsym.flat_proxy_outs[0], None)
                        check(
                            wait_consumer is not None
                            and len(wait_consumer) == 1
                            and wait_consumer[0].sym.id == unpack_for_fsdp_prim_impl.id,
                            lambda: f"The consumer of wait operator should be unpack operator",
                        )
                        unpack_bsyms.append(wait_consumer[0])
                bound_symbols.append(bsym)
            case _:
                bound_symbols.append(bsym)

    # if no allgather+wait exists
    if len(allgather_bsyms) == 0:
        return execution_trace
    if not bucketing_enabled:
        pack_bsyms = [None for _ in allgather_bsyms]
        unpack_bsyms = [None for _ in allgather_bsyms]
    comms = list(utils.safe_zip(pack_bsyms, allgather_bsyms))
    waits = list(utils.safe_zip(wait_bsyms, unpack_bsyms))
    check(
        len(comms) == len(waits),
        lambda: f"The number of allgathers (={len(comms)}) should be equal to the number of waits (={len(waits)})",
    )
    del pack_bsyms, allgather_bsyms, wait_bsyms, unpack_bsyms, producers, consumers

    new_bsyms = deque()
    n_running_comms = 0
    idx_comm = len(waits) - 1
    idx_wait = len(waits) - 1
    for i in range(len(bound_symbols) - 1, -1, -1):
        new_bsyms.appendleft(bound_symbols[i])
        if bound_symbols[i] == waits[idx_wait][0]:
            n_running_comms += 1
            idx_wait -= 1
        if n_running_comms == max_in_flight_comms:
            new_bsyms.appendleft(comms[idx_comm][1])
            if bucketing_enabled:
                new_bsyms.appendleft(comms[idx_comm][0])
            idx_comm -= 1
            n_running_comms -= 1
        if bound_symbols[i] == waits[0][0]:
            for idx in range(idx_comm, -1, -1):
                new_bsyms.appendleft(comms[idx][1])
                if bucketing_enabled:
                    new_bsyms.appendleft(comms[idx][0])
            for idx in range(i - 1, -1, -1):
                new_bsyms.appendleft(bound_symbols[idx])
            break

    check(len(new_bsyms) == len(orig_bound_symbols), lambda: f"{len(orig_bound_symbols) = } but {len(new_bsyms) = }")

    new_execution_trace.bound_symbols = new_bsyms
    return new_execution_trace
