from itertools import chain
import time

import thunder.core.prims as prims
from thunder.core.proxies import Proxy, variableify
from thunder.core.pytree import tree_flatten, tree_map
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import from_trace, TraceProvenance, TraceCtx as Trace
from thunder.core.utils import ProxyDict, producers


#
# Common optimization and transform passes
#
# NOTE This avoids transforms depending on passes, since passes depend on transforms

# NOTE Runs a Dead Code Elimination (DCE) pass
#   Technically this could be a "transform", because it is semantic-preserving.
# TODO We could look at reconciling the ideas of what a trace produces and the return prim
# TODO This calls variableify(), but we could directly construct Variable objects instead
def dce(trace: Trace) -> tuple[Trace, list[Trace]]:
    start_time_ns = time.time_ns()

    producer_map: ProxyDict = producers(trace)

    flat_trace_outputs, _ = tree_flatten(trace.output)
    needed_proxies = set(tuple(variableify(x) for x in flat_trace_outputs if isinstance(x, Proxy)))
    dced = []

    # NOTE These primitives are marked to not be collected because they dictate the function's output
    #   (RETURN), have side effects (PRINT), or are comments (COMMENT, UNPACK_TRIVIAL)
    dont_collect = {
        prims.PrimIDs.RETURN,
        prims.PrimIDs.COMMENT,
        prims.PrimIDs.PRINT,
        prims.PrimIDs.UNPACK_TRIVIAL,
    }

    bsym: BoundSymbol
    for bsym in reversed(trace.bound_symbols):
        # Preserves symbols that should never be collected
        if bsym.sym.id in dont_collect:
            needed = True
        else:
            needed = False

        # NOTE This block is run even if we know we're preserving the operation, because it
        #   may mark some of the operation's outputs as unused
        some_unused = False
        for out in bsym.flat_proxy_outs:
            if variableify(out) in needed_proxies and producer_map[out] == bsym:
                needed = True
            else:
                some_unused = True

        if needed:
            nbsym: BoundSymbol = bsym

            # Replaces unused Proxy outputs with None
            if some_unused:

                def _helper(x):
                    if isinstance(x, Proxy) and (variableify(x) not in needed_proxies or producer_map[x] != bsym):
                        return None
                    return x

                nbsym_output = tree_map(_helper, bsym.output)
                nbsym = bsym.from_bsym(output=nbsym_output)

            dced.append(nbsym)
            for x in chain(nbsym.flat_proxy_args, nbsym.flat_proxy_kwargs):
                needed_proxies.add(variableify(x))

    dcetrace = from_trace(trace)
    dcetrace.bound_symbols = list(reversed(dced))
    
    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    dcetrace.set_provenance(TraceProvenance(f"Dead Code Elimination (took {elapsed_time_millis} milliseconds)"))

    return dcetrace, [dcetrace]