from typing import Dict, Any, List, Callable, Sequence, Tuple
from collections import deque
from itertools import chain

from thunder.core.trace import TraceCtx, from_trace, TraceProvenance, VariableInterface
import thunder.core.utils as utils
from thunder.core.utils import ProxyDict
from thunder.core.symbol import BoundSymbol
from thunder.core.pytree import tree_flatten
import thunder.core.prims as prims
from thunder.executors.utils import *
from thunder.core.proxies import Proxy


# TODO Review preserving return statements -- here they are arbitrarily preserved, which seems hacky
# NOTE Runs a Dead Code Elimination (DCE) pass
#   Technically this could be a "transform", because it is semantic-preserving.
# TODO Remove unused constants
# TODO Review DCE for unpacking operations -- a, _ = tup would be nice when the second
#   element of the tuple is unused
# TODO Consider tracking only proxy computations
# TODO We could probably remove RETURN and the UNPACK prims from dont collect with better producer-consumer modeling
def dce(trace: TraceCtx) -> Tuple[TraceCtx, List[TraceCtx]]:
    producers: ProxyDict = utils.producers(trace)

    # NOTE Not an ordered set because we're just checking for membership
    needed_nodes = set()

    seen = set()
    q = deque()

    flat_outputs, _ = tree_flatten(trace.output)
    q.extend(flat_outputs)

    dont_collect = set(
        (
            prims.PrimIDs.RETURN,
            prims.PrimIDs.COMMENT,
            prims.PrimIDs.UNPACK_DICT,
            prims.PrimIDs.UNPACK_SEQUENCE,
            prims.PrimIDs.UNPACK_TRIVIAL,
            prims.PrimIDs.PRINT,
        )
    )

    while True:
        try:
            x = q.popleft()

            # Skips constants, which aren't produced by other nodes
            if not isinstance(x, Proxy):
                continue

            # TODO Consider modeling these objects as produced by a signature primitive
            # Skips objects which have no producer
            if x not in producers:
                continue

            producer: BoundSymbol = producers[x]

            # Skips producers that have already been visited
            if producer in needed_nodes:
                continue

            needed_nodes.add(producer)

            flat_args = producer._flat_args
            flat_kwargs = producer._flat_kwargs

            new_proxies = [arg for arg in chain(flat_args, flat_kwargs)]

            q.extend(new_proxies)
        except IndexError:
            break

    dcetrace = from_trace(trace)
    dcetrace.bound_symbols = [
        bsym for bsym in trace.bound_symbols if (bsym in needed_nodes or bsym.sym.id in dont_collect)
    ]
    dcetrace.set_provenance(TraceProvenance("Dead Code Elimination"))

    return dcetrace, [dcetrace]


# TODO Review deleting non-proxies
def del_last_used(trace: TraceCtx) -> Tuple[TraceCtx, List[TraceCtx]]:
    """Mark last used intermediates to be deleted. This is necessary to avoid memory leaks.

    Args:
        trace: trace to be transformed
    Returns:
        list: transformed trace
    """

    del_trace = from_trace(trace)
    bsyms = deque()

    outs = utils.sequencify(trace.output)
    flat_outs, _ = tree_flatten(outs)

    # TODO Replace with ProxySet (which does not exist at the time of this writing)
    handled = ProxyDict(trace)
    for out in flat_outs:
        if isinstance(out, Proxy):
            handled[out] = None

    for bsym in reversed(trace.bound_symbols):
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
            del_sym = prims.python_del.bind(to_del)
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
def claim(trace: TraceCtx, executors_list: Sequence, *, prims_only: bool = False) -> Tuple[TraceCtx, List[TraceCtx]]:
    def _set_executor(bsym: BoundSymbol, ex):
        bsym._executor = ex

        for sbsym in bsym.subsymbols:
            _set_executor(sbsym, ex)

    def _find_executor(bsym: BoundSymbol):
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
            if not _find_executor(sbsym):
                return False

        return True

    for bsym in trace.bound_symbols:
        utils.check(_find_executor(bsym), lambda: f"Could not find executor for bound symbol {bsym}")

    return trace, []


def flatten(trace: TraceCtx, *, prims_only: bool = False) -> Tuple[TraceCtx, List[TraceCtx]]:
    flattenedtrace = from_trace(trace)
    flattened: List[BoundSymbol] = []

    # TODO Maybe make this nonrecursive
    def _flatten(bsym: BoundSymbol):
        nonlocal flattened
        ex = bsym._executor

        if ex is not None and ex.is_supported(bsym, prims_only=prims_only):
            # Propagates executor to subsymbols
            bsym._executor = ex
            flattened.append(bsym)
        else:
            utils.check(
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


def fuse(trace: TraceCtx) -> Tuple[TraceCtx, List[TraceCtx]]:
    fusedtrace = from_trace(trace)
    fused_bsyms = []

    producers = utils.producers(trace)
    consumers = utils.consumers(trace)

    batch = []
    batch_ex = trace.bound_symbols[0]._executor if len(trace.bound_symbols) > 0 else None
    _executor_ctrs = {batch_ex: 0}
    for bsym in trace.bound_symbols:
        if batch_ex != bsym._executor:
            fused_bsyms.extend(batch_ex.fuse(trace, producers, consumers, batch, _executor_ctrs[batch_ex]))
            _executor_ctrs[batch_ex] += 1
            batch = [bsym]
            batch_ex = bsym._executor

            if batch_ex not in _executor_ctrs:
                _executor_ctrs[batch_ex] = 0
        else:
            batch.append(bsym)

    # Processes last batch
    if len(batch) > 0:
        fused_bsyms.extend(batch_ex.fuse(trace, producers, consumers, batch, _executor_ctrs[batch_ex]))

    fusedtrace.bound_symbols = fused_bsyms
    fusedtrace.set_provenance(TraceProvenance("Fusion"))

    return fusedtrace, [fusedtrace]
