from typing import Dict, Any, List, Callable, Sequence, Tuple, Optional
from collections import deque
from dataclasses import replace
from itertools import chain

from thunder.core.trace import TraceCtx, from_trace, TraceProvenance, VariableInterface
import thunder.core.utils as cutils
from thunder.core.utils import ProxyDict
from thunder.core.symbol import BoundSymbol
from thunder.core.pytree import tree_flatten, tree_unflatten
import thunder.core.prims as prims
from thunder.executors.utils import Region, Node, graph_from_regions, toposort, Executor
from thunder.core.proxies import Proxy, variableify, unvariableify
from thunder.executors import torchex as TorchEx


# NOTE Runs a Dead Code Elimination (DCE) pass
#   Technically this could be a "transform", because it is semantic-preserving.
# TODO We could look at reconciling the ideas of what a trace produces and the return prim
def dce(trace: TraceCtx) -> Tuple[TraceCtx, List[TraceCtx]]:
    producers: ProxyDict = cutils.producers(trace)

    flat, _ = tree_flatten(trace.output)
    needed_proxies = set(tuple(variableify(x) for x in flat if isinstance(x, Proxy)))
    dced = []

    # NOTE These primitives are marked to not be collected because they dictate the function's output
    #   (return) or have side-effects (comment, print)
    dont_collect = {prims.PrimIDs.RETURN, prims.PrimIDs.COMMENT, prims.PrimIDs.PRINT, prims.PrimIDs.UNPACK_TRIVIAL}

    for bsym in reversed(trace.bound_symbols):
        # Preserves symbols that should never be collected
        if bsym.sym.id in dont_collect:
            needed = True
        else:
            needed = False

        some_unused = False
        nouts = []
        outs = bsym._flat_outs
        for out in outs:
            if isinstance(out, Proxy):
                if variableify(out) in needed_proxies and producers[out] == bsym:
                    needed = True
                    nouts.append(out)
                else:
                    some_unused = True
                    nouts.append(None)
            else:
                nouts.append(out)

        # Replaces unused Proxy outputs with None
        # TODO Refactor this into an output update function
        if some_unused:
            _, out_spec = tree_flatten(bsym.output)
            bsym.output = tree_unflatten(nouts, out_spec)
            # Deletes caches that rely on output
            if hasattr(bsym, "_flat_outs"):
                del bsym._flat_outs
            if hasattr(bsym, "_var_output"):
                del bsym._var_output
            if hasattr(bsym, "_hash"):
                del bsym._hash

        if needed:
            dced.append(bsym)
            for x in chain(bsym._flat_args, bsym._flat_kwargs):
                if isinstance(x, Proxy):
                    needed_proxies.add(variableify(x))

    dcetrace = from_trace(trace)
    dcetrace.bound_symbols = list(reversed(dced))
    dcetrace.set_provenance(TraceProvenance("Dead Code Elimination"))
    return dcetrace, [dcetrace]


comment_symbols = {prims.PrimIDs.COMMENT, prims.PrimIDs.UNPACK_TRIVIAL}


# TODO Review deleting non-proxies
def del_last_used(trace: TraceCtx) -> tuple[TraceCtx, list[TraceCtx]]:
    """Mark last used intermediates to be deleted. This is necessary to avoid memory leaks.

    Args:
        trace: trace to be transformed
    Returns:
        list: transformed trace
    """
    del_trace = from_trace(trace)
    bsyms = deque()

    outs = cutils.sequencify(trace.output)
    flat_outs, _ = tree_flatten(outs)

    # TODO Replace with ProxySet (which does not exist at the time of this writing)
    handled = ProxyDict(trace)
    for out in flat_outs:
        if isinstance(out, Proxy):
            handled[out] = None

    for bsym in reversed(trace.bound_symbols):
        if bsym.sym.id in comment_symbols:
            bsyms.appendleft(bsym)
            continue

        flat_outs, _ = tree_flatten(bsym.output)
        flat_args, _ = tree_flatten(bsym.args)
        flat_kwargs, _ = tree_flatten(bsym.kwargs)

        to_del = []
        for x in chain(flat_outs, flat_args, flat_kwargs):
            if not isinstance(x, Proxy) or x in handled:
                continue

            handled[x] = None
            to_del.append(x)

        if len(to_del) > 0:
            del_sym = prims.python_del.bind(to_del, output=None)
            bsyms.appendleft(del_sym)

        bsyms.appendleft(bsym)

    del_trace.bound_symbols = list(bsyms)
    del_trace.set_provenance(TraceProvenance("Delete Last Used"))

    return del_trace, [del_trace]


# TODO Improve executor annotation
# TODO What if an operation is SOMETIMES fusible? This should originate
#   constraints properly
# TODO (mruberry) I think it would be helpful if the claim phase
#   added comments describing the claimed regions, but these comments
#   would need to be removed later because they're interesting
#   while claiming but not so much later
#   I think we should extend the idea of provenance so it can
#   have a "friendly" print version of the trace vs. the actual
#   output, so comments can appear temporarily and not actually be
#   part of the input to subsequence traces
#   Until this is done -- let's just model this as doing nothing, because
#   it won't print any debugging information
# TODO Consider more advanced claiming strategies, like claiming
#   portions of operations
# Identifies which executor will execute each operation
#   Executors are queried in the order provided
#   An error is thrown if none of the given executors can execute the operation
# TODO Improve this to explain what couldn't be executed
# TODO This could probably be done more efficiently
def claim(trace: TraceCtx, executors_list: Sequence, *, prims_only: bool = False) -> tuple[TraceCtx, list[TraceCtx]]:
    def _set_executor(bsym: BoundSymbol, ex) -> None:
        bsym._executor = ex

        for sbsym in bsym.subsymbols:
            _set_executor(sbsym, ex)

    def _find_executor(bsym: BoundSymbol) -> tuple[BoundSymbol, bool]:
        # Attempts to find an executor for the symbol
        for ex in executors_list:
            if ex.can_execute(bsym, prims_only=prims_only):
                _set_executor(bsym, ex)
                return True

        # If no executors can execute the symbol directly, find
        #   executors for the sub-symbols
        if len(bsym.subsymbols) == 0:
            return False

        for sbsym in bsym.subsymbols:
            found = _find_executor(sbsym)
            if not found:
                return False

        return True

    for bsym in trace.bound_symbols:
        found = _find_executor(bsym)
        cutils.check(found, lambda: f"Could not find executor for bound symbol {bsym}")

    return trace, []


def flatten(trace: TraceCtx, *, prims_only: bool = False) -> tuple[TraceCtx, list[TraceCtx]]:
    flattenedtrace = from_trace(trace)
    flattened: list[BoundSymbol] = []

    # TODO Maybe make this nonrecursive
    def _flatten(bsym: BoundSymbol):
        nonlocal flattened
        ex = bsym._executor

        if ex is not None and ex.is_supported(bsym, prims_only=prims_only):
            # Propagates executor to subsymbols
            flattened.append(bsym)
        else:
            cutils.check(
                len(bsym.subsymbols) > 0,
                lambda: f"Trying to flatten {bsym} for execution but it's not supported and has no subsymbols",
                exception_type=AssertionError,
            )
            for ssym in bsym.subsymbols:
                _flatten(ssym)

    for bsym in trace.bound_symbols:
        _flatten(bsym)

    flattenedtrace.bound_symbols = flattened
    flattenedtrace.set_provenance(TraceProvenance("Flatten"))

    return flattenedtrace, [flattenedtrace]


def fuse(trace: TraceCtx) -> tuple[TraceCtx, list[TraceCtx]]:
    fusedtrace = from_trace(trace)
    fused_bsyms = []

    producers = cutils.producers(trace)
    consumers = cutils.consumers(trace)

    batch = []
    batch_ex = trace.bound_symbols[0]._executor if len(trace.bound_symbols) > 0 else None

    regions = []

    # Constructs regions of contiguous operations that all share
    #   an executor
    for bsym in trace.bound_symbols:
        if batch_ex != bsym._executor:
            # Constructs a region representing what to fuse (currently unused)
            region = Region(trace, producers, consumers, batch, batch_ex, -1)
            regions.append(region)

            # Updates region collection metadata
            batch = [bsym]
            batch_ex = bsym._executor
        else:
            batch.append(bsym)

    # Processes last batch
    if len(batch) > 0:
        region = Region(trace, producers, consumers, batch, batch_ex, -1)
        regions.append(region)

    g = graph_from_regions(regions)

    # TODO Maybe implement a more advanced selection criteria?
    def _selector(last_added: Optional[Node], nodes: list[Node]) -> Node:
        if last_added is None or last_added.region.executor.name() != Executor.NVFUSER:
            # NOTE In this case the last added is None or the executor
            #   was not nvFuser
            # Attempts to find a non-nvFuser region to go next
            for node in nodes:
                if node.region.executor.name() != Executor.NVFUSER:
                    return node

            # Defaults to returning the first eligible node
            return nodes[0]

        # NOTE In this case the last added region's executor is nvFuser
        # Attempts to find another nvFuser region
        for node in nodes:
            if node.region.executor.name() == Executor.NVFUSER:
                return node

        # Defaults to returning the first eligible node
        return nodes[0]

    toposorted = toposort(g, _selector)

    # Merges adjacent regions that share an executor
    node = toposorted.reset()
    while node is not None:
        # Tries to merge one or more regions into the current node
        while True:
            peek = toposorted.peek()

            if peek is None:
                break

            ar = node.node.region
            br = peek.node.region

            if ar.executor is br.executor:
                toposorted.merge(node, peek)
            else:
                break

        node = toposorted.next()

    # Translates the (possibly) merged linearization to a trace
    # Counts how many fusions (per executor) have been constructed
    #   (Used to name fusions like nvFusion0, nvFusion1, ...)
    executor_ctrs = {}
    node = toposorted.reset()
    while node is not None:
        region = node.node.region
        ex = region.executor

        # Regions that would go to nvFuser but are entirely composed of shape operations
        #   are sent to the torch executor instead
        # TODO Think about being more clever about when this occurs
        #   Today fusion happens after flattening, but we should toposort and do this
        #   analysis before flattening occurs
        if ex.name() == Executor.NVFUSER and region.only_shape_operations():
            ex = TorchEx

        if ex not in executor_ctrs:
            executor_ctrs[ex] = 0
        counter = executor_ctrs[ex]
        executor_ctrs[ex] += 1

        region.counter = counter
        fused_bsyms.extend(ex.fuse(region))
        node = toposorted.next()

    # Constructs the new trace
    fusedtrace.bound_symbols = fused_bsyms
    fusedtrace.set_provenance(TraceProvenance("Fusion"))

    return fusedtrace, [fusedtrace]
