from typing import Dict, Any, List, Callable, Sequence, Tuple, Optional
from collections import deque
from dataclasses import replace
from itertools import chain

from thunder.core.trace import TraceCtx, from_trace, TraceProvenance, VariableInterface
import thunder.core.dtypes as dtypes
import thunder.core.utils as cutils
from thunder.core.utils import ProxyDict
from thunder.core.symbol import BoundSymbol
from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
import thunder.core.prims as prims
from thunder.executors.utils import Region, Node, graph_from_regions, toposort, Executor, group_bookend_meta_ops
from thunder.core.proxies import Proxy, variableify, unvariableify
from thunder.executors import torchex as TorchEx


# NOTE Runs a Dead Code Elimination (DCE) pass
#   Technically this could be a "transform", because it is semantic-preserving.
# TODO We could look at reconciling the ideas of what a trace produces and the return prim
def dce(trace: TraceCtx) -> Tuple[TraceCtx, List[TraceCtx]]:
    producers: ProxyDict = cutils.producers(trace)

    flat_trace_outputs, _ = tree_flatten(trace.output)
    needed_proxies = set(tuple(variableify(x) for x in flat_trace_outputs if isinstance(x, Proxy)))
    dced = []

    # NOTE These primitives are marked to not be collected because they dictate the function's output
    #   (RETURN), have side effects (PRINT), or are comments (COMMENT, UNPACK_TRIVIAL)
    dont_collect = {prims.PrimIDs.RETURN, prims.PrimIDs.COMMENT, prims.PrimIDs.PRINT, prims.PrimIDs.UNPACK_TRIVIAL}

    for bsym in reversed(trace.bound_symbols):
        # Preserves symbols that should never be collected
        if bsym.sym.id in dont_collect:
            needed = True
        else:
            needed = False

        # NOTE This block is run even if we know we're preserving the operation, because it
        #   may mark some of the operation's outputs as unused
        some_unused = False
        for out in bsym._flat_outs:
            if isinstance(out, Proxy):
                if variableify(out) in needed_proxies and producers[out] == bsym:
                    needed = True
                else:
                    some_unused = True

        if needed:
            nbsym = bsym

            # Replaces unused Proxy outputs with None
            if some_unused:

                def _helper(x):
                    if isinstance(x, Proxy) and (variableify(x) not in needed_proxies or producers[x] != bsym):
                        return None
                    return x

                nbsym_output = tree_map(_helper, bsym.output)
                nbsym = bsym.from_bsym(output=nbsym_output)

            dced.append(nbsym)
            for x in chain(nbsym._flat_args, nbsym._flat_kwargs):
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

        def _fuse_region_with_executor(executor, region_fuse):
            if executor not in executor_ctrs:
                executor_ctrs[executor] = 0
            counter = executor_ctrs[executor]
            executor_ctrs[executor] += 1

            region_fuse.counter = counter
            fused_bsyms.extend(executor.fuse(region_fuse))

        # Regions that would go to nvFuser but are entirely composed of shape operations
        #   are sent to the torch executor instead
        # TODO Think about being more clever about when this occurs
        #   Today fusion happens after flattening, but we should toposort and do this
        #   analysis before flattening occurs
        if ex.name() == Executor.NVFUSER:
            if region.only_shape_operations():
                _fuse_region_with_executor(TorchEx, region)
            else:
                list_region = group_bookend_meta_ops(region, producers, consumers)
                for sub_region in list_region:
                    _fuse_region_with_executor(sub_region.executor, sub_region)
        else:
            _fuse_region_with_executor(ex, region)

        node = toposorted.next()

    # Constructs the new trace
    fusedtrace.bound_symbols = fused_bsyms
    fusedtrace.set_provenance(TraceProvenance("Fusion"))

    return fusedtrace, [fusedtrace]


# Removes excessive float casts, like those that occur when autocasting
# NOTE This passes actually changes a program's semantics, because it will take a sequence like
#   fp32 -> fp16 -> fp32 and remove all the operations, but casting fp32 values to fp16 can
#   changes the values (because most fp32 values are not representable in fp16)
# NOTE This only handles conversions performed by CONVERT_ELEMENT_TYPE, and not conversions caused
#   by other Symbols, like torch.to, which may be unflattened
# TODO This could be extended to non-float conversions, like complex -> complex conversions
def remove_redundant_casts(trace: TraceCtx) -> tuple[TraceCtx, list[TraceCtx]]:
    rrctrace = from_trace(trace)

    # Returns a tuple (is proxy float->float conversion?, object to convert, dtype to convert to)
    def is_eligible_cast(bsym: BoundSymbol) -> tuple[bool, Any, Any]:
        # Ignores operations other than CONVERT_ELEMENT_TYPE
        if bsym.sym.id is not prims.PrimIDs.CONVERT_ELEMENT_TYPE:
            return False, None, None

        # Parses arguments
        # TODO We should consider canonicalizing how BoundSymbols express their arguments
        a: Any
        dtyp: dtypes.dtype

        if len(bsym.args) == 2:
            a, dtyp = bsym.args
        elif len(bsym.args) == 1:
            cutils.check(len(bsym.kwargs) == 1, lambda: f"Expected two arguments for convert element type")
            (a,) = bsym.args
            dtyp = bsym.kwargs["dtype"]
        else:
            a = bsym.kwargs["a"]
            dtyp = bsym.kwargs["dtype"]

        if not isinstance(a, Proxy):
            return False, None, None

        adtyp = dtypes.to_dtype(a)
        is_float_to_float_conversion = dtypes.is_float_dtype(dtypes.to_dtype(a)) and dtypes.is_float_dtype(dtyp)

        return is_float_to_float_conversion, a, dtyp

    # Updates intermediate conversions, identifies no-ops, and updates no-op consumers
    # NOTE These are separate maps. A no-op in this context is a cast from the
    #   input's dtype to itself, like the following:
    #
    #   b = prims.convert_element_type(a, float32)  # a: f32
    #
    #   For these operations, everywhere b is consumed can be replaced with a.
    #
    #   When there is an intermediate conversion, however, we don't want to replace all uses
    #   of its output with its input. For example, the dtype modified output could
    #   actually be consumed by non-cast operations.

    # TODO This is intentionally commented out. See TODO below on consumer analysis.
    # consumers = cutils.consumers(trace)
    replacement_map = {}
    intermediate_map = {}
    nbsyms = []
    for bsym in trace.bound_symbols:
        is_proxy_f2f_conversion, a, dtyp = is_eligible_cast(bsym)

        # Replaces inputs due to no-op casts for all operations
        if not is_proxy_f2f_conversion:
            nbsym = bsym
            if bsym.has_input(replacement_map):
                nbsym = bsym.from_bsym_swap_inputs(replacement_map)
            nbsyms.append(nbsym)
            continue

        # NOTE is_proxy_f2f_conversion is True
        va = variableify(a)
        vo = variableify(bsym.output)

        # Identifies updated input
        orig = intermediate_map.get(va, a)
        orig_dtype = dtypes.to_dtype(orig)

        # Elides no-ops, marking their outputs for replacement
        if orig_dtype == dtyp:
            replacement_map[vo] = orig
            intermediate_map[vo] = orig
            continue

        # NOTE In this case there is a more original input

        # Only marks this output for replacement with the more original input if it's
        #   not consumed by a non-cast operation
        has_non_cast_consumer = False
        # TODO (mruberry) I'm not sure whether the following is worthwhile, although
        #   I'm leaving it as a comment because we may want to revive it in the future.
        #   Essentially, this would be a heuristic that says: "if x is being consumed,
        #   don't bother finding the precursor of x to cast, just cast x itself."
        #   That may improve data locality, but it could also lead to excessive
        #   casts.
        # for consumer in consumers.get(bsym.output, ()):
        #     if consumer.sym.id is not prims.PrimIDs.CONVERT_ELEMENT_TYPE:
        #         has_non_cast_consumer = True
        #         break

        # When this operation has non-cast consumers, later conversion operations
        #   might as well consume its output to try and improve data locality and
        #   not have to preserve the original tensor for so long
        if has_non_cast_consumer:
            intermediate_map[vo] = bsym.output
        else:
            intermediate_map[vo] = orig

        # Possibly creates a new BoundSymbol consuming the original instead of the current input
        if orig is a:
            nbsyms.append(bsym)
        else:
            # NOTE This is faster than using from_bsym_swap_inputs, and relies on us only working
            #   with prims.convert_element_type
            nbsym = bsym.from_bsym(args=(orig, dtyp), kwargs={})
            nbsyms.append(nbsym)
            cutils.check(
                nbsym.subsymbols is None or len(nbsym.subsymbols) == 0,
                lambda: f"Expected no subsymbols when creating a new BoundSymbol in the remove redundant casts pass",
                exception_type=AssertionError,
            )

    rrctrace.bound_symbols = nbsyms

    # Updates the trace's output
    def map_redundant(x: Any) -> Any:
        if isinstance(x, Proxy):
            return replacement_map.get(variableify(x), x)
        return x

    new_trace_output = tree_map(map_redundant, trace.output)
    rrctrace.output = new_trace_output

    rrctrace.set_provenance(TraceProvenance("Remove redundant casts"))
    return rrctrace, [rrctrace]
