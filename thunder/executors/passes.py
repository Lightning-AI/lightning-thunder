from typing import Dict, Any, List, Tuple, Optional
from collections.abc import Callable
from collections.abc import Sequence
from collections import deque
from dataclasses import dataclass, replace
from itertools import chain
from functools import partial
import time

from thunder.core.trace import TraceCtx, from_trace, TraceProvenance, VariableInterface
import thunder.core.dtypes as dtypes
import thunder.core.utils as cutils
from thunder.core.utils import ProxyDict, check, safe_map_flat
from thunder.core.symbol import BoundSymbol
from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
import thunder.core.prims as prims
from thunder.core.proxies import Proxy, variableify, unvariableify, Variable, CollectionProxy
import thunder.core.transforms as transforms
from thunder.core.transform_common import dce
from thunder.core.trace import get_tracectx
from thunder.executors.pythonex import clear_collection

from thunder.extend import Executor, get_always_executors, OperatorExecutor, FusionExecutor

comment_symbols = {prims.PrimIDs.COMMENT, prims.PrimIDs.UNPACK_TRIVIAL}


# Transforms a trace by determining which execution transforms to call given the list of executors in priority order
def _transform_for_operator_executor_execution(trace: TraceCtx, executors_list: Sequence[Executor]) -> TraceCtx:
    start_time_ns = time.time_ns()

    swapmap: dict[Variable, Proxy] = {}

    def update_swapmap(o: Any, no: Any) -> None:
        if isinstance(o, Proxy):
            check(
                isinstance(no, Proxy),
                lambda: f"Expected an execution transform to produce outputs with the same type, but found {type(o)} and {type(no)}",
            )

            vo = variableify(o)
            vno = variableify(no)
            if vo == vno:
                return
            swapmap[vno] = o

    def preserve_bsym(bsym: BoundSymbol) -> Any:
        trace = get_tracectx()
        trace.scopes[-1].append(bsym)
        for p in chain(bsym.flat_proxy_outs, bsym.flat_proxy_args):
            trace.names.add(p.name)
        return bsym.output

    # TODO Consider using an enum for this function's return values
    # Tries to find an executor for the BoundSymbol
    #   If the BoundSymbol already has an executor then None is returned
    #   If the executor has an execution transform, it's called and True is returned
    #   If no executor can execute the BoundSymbol, False is returned
    def visit_helper_(bsym: BoundSymbol) -> None | bool:
        if bsym.sym.python_impl is not None:
            return None

        ex: Executor
        for ex in executors_list:
            # TODO Consider allowing operator executors to claim portions of operations
            # TODO Should FusionExecutors be allowed to claim bsym with bsym.sym.executor?
            if (isinstance(ex, OperatorExecutor) and ex.can_execute(bsym)) or (
                isinstance(ex, FusionExecutor) and ex.can_fuse(bsym)
            ):
                execution_transform: None | Callable = ex.get_execution_transform(bsym.sym)
                out: Any
                if execution_transform is not None:
                    out = execution_transform(*bsym.args, **bsym.kwargs)
                elif isinstance(ex, OperatorExecutor):
                    # NOTE execution_transform is None and the executor is an operator executor
                    # Calls the operator executor's operation
                    # TODO Instead of directly acquiring the symbol from the implmap, we probably
                    #   want to hide this behind a function
                    op = ex.implmap[bsym.sym.id].symbol
                    out = op(*bsym.args, **bsym.kwargs)
                elif isinstance(ex, FusionExecutor):
                    # NOTE execution_transform is None and the executor is a fusion executor
                    # Preserves the symbol as is (it will be handled in the fusion pass)
                    out = preserve_bsym(bsym)
                else:
                    raise AssertionError("Unknown executor")

                safe_map_flat(update_swapmap, bsym.output, out)
                return True

        if bsym.sym.executor is not None:
            return None

        return False

    def visit_(bsym: BoundSymbol) -> transforms.VISIT_TYPE:
        result: None | bool = visit_helper_(bsym)

        if result is None:
            return transforms.VISIT_TYPE.NO_OP

        if result is True:
            return transforms.VISIT_TYPE.REPLACE

        # NOTE result is False (which means no executor was found for the symbol)
        cutils.check(not bsym.sym.is_prim, lambda: f"Failed to find an executor for bound symbol {bsym=}")
        for sbsym in bsym.subsymbols:
            visit_(sbsym)

        return transforms.VISIT_TYPE.REPLACE

    extrace = transforms.visitor_transform(trace, visit_)

    # Restores original variables
    bound_symbols: list[BoundSymbol] = []
    for bsym in extrace.bound_symbols:
        nbsym: BoundSymbol = bsym.from_bsym_swap_proxies(swapmap)
        bound_symbols.append(nbsym)

    extrace.bound_symbols = bound_symbols

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    extrace.set_provenance(
        TraceProvenance(f"Transform for operator executor execution (took {elapsed_time_millis} milliseconds)")
    )
    return extrace


def transform_for_execution(trace: TraceCtx, executors_list: Sequence[Executor]) -> TraceCtx:
    start_time_ns = time.time_ns()

    trace = dce(trace)

    #
    # Step 1 Performs execution transforms
    #
    extrace = _transform_for_operator_executor_execution(trace, executors_list)
    extrace = dce(extrace)

    #
    # Step 2 Fusion executors can transform the trace
    #
    for ex in executors_list:
        if isinstance(ex, FusionExecutor):
            extrace = ex.fusion_pass(extrace)

    #
    # Step 3 "Always" executors are given the opportunity to execute unclaimed symbols
    #
    # NOTE "Always" executors cannot produce symbols that other executors are expected to execute
    #   (The only exception is that they can produce symbols that the Python executor is expected to execute)
    # NOTE This occurs if a fusion executor declines to execute a symbol after running its fusion pass
    extrace = _transform_for_operator_executor_execution(extrace, get_always_executors())

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000

    extrace.set_provenance(TraceProvenance(f"Transform for execution (took {elapsed_time_millis} milliseconds)"))
    return extrace


# This is needed to ensure that subsymbol changes are reflected in the Python
# code generator.
def _update_fusion_call_ctx(bsym: BoundSymbol) -> BoundSymbol:
    """Update the call_ctx information of the fusion BoundSymbol object.

    Args:
        bsym: The fusion BoundSymbol object.

    Returns:
        The updated fusion BoundSymbol object.
    """

    @dataclass
    class BoundSymbolRegion:
        inputs: tuple
        outputs: tuple
        bound_symbols: tuple

    def find_counter(s):
        head = s.rstrip("0123456789")
        counter = s[len(head) :]
        return int(counter)

    counter = find_counter(bsym.sym.name)

    def fusion_bsym_to_region(bsym: BoundSymbol):
        return BoundSymbolRegion(
            inputs=tuple(filter(None, map(variableify, bsym.args))),
            outputs=tuple(filter(None, map(variableify, bsym.output))),
            bound_symbols=bsym.subsymbols,
        )

    # fuse returns a new BoundSymbol object with correct updated call_ctx
    # information
    return bsym.sym.executor.fuse(fusion_bsym_to_region(bsym), counter)


def update_fusion_call_ctx(trace: TraceCtx) -> TraceCtx:
    """Updates the call context of the trace to be the current call context.

    Some of the fusion bound symbols may have been created with a different call
    context and modified after the fact. This pass ensures that the call context
    is correct.

    Args:
        trace (TraceCtx): trace to be transformed
    Returns:
        (TraceCtx): transformed trace
    """
    start_time_ns = time.time_ns()

    new_trace = from_trace(trace)
    new_trace.bound_symbols = []
    for bsym in trace.bound_symbols:
        if bsym.sym.is_fusion:
            new_trace.bound_symbols.append(_update_fusion_call_ctx(bsym))
        else:
            new_trace.bound_symbols.append(bsym)

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    new_trace.set_provenance(TraceProvenance(f"Update Call Context (took {elapsed_time_millis} milliseconds)"))
    return new_trace


# TODO Review deleting non-proxies
def del_last_used(trace: TraceCtx, *, clear_collections=False) -> TraceCtx:
    """Mark last used intermediates to be deleted. This lets the Python garbage collector free
        unused tensor memory.

    Args:
        trace: trace to be transformed
        clear_collections: whether to clear collections
    Returns:
        list: transformed trace
    """
    start_time_ns = time.time_ns()

    del_trace = from_trace(trace)
    bsyms = deque()

    outs = cutils.sequencify(trace.output)
    flat_outs, _ = tree_flatten(outs)

    # TODO Replace with ProxySet (which does not exist at the time of this writing)
    handled = ProxyDict()
    for out in flat_outs:
        if isinstance(out, Proxy):
            handled[out] = None

    bsym: BoundSymbol
    for bsym in reversed(trace.bound_symbols):
        if bsym.sym.id in comment_symbols:
            bsyms.appendleft(bsym)
            continue

        to_del = []
        for x in chain(bsym.flat_proxy_outs, bsym.flat_proxy_args):
            if x in handled:
                continue

            handled[x] = None
            to_del.append(x)

        to_clear_collections = []
        if clear_collections:
            for x in to_del:
                if isinstance(x, CollectionProxy):
                    to_clear_collections.append(x)

        # NOTE The check for return avoids putting dels after the return statement
        if to_del and bsym.sym.id is not prims.PrimIDs.RETURN:
            del_sym: BoundSymbol = prims.python_del.bind(*to_del, output=None)
            bsyms.appendleft(del_sym)

            for x in to_clear_collections:
                bsyms.appendleft(clear_collection.bind(x, output=None))

        bsyms.appendleft(bsym)

    del_trace.bound_symbols = list(bsyms)

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000

    del_trace.set_provenance(TraceProvenance(f"Delete Last Used (took {elapsed_time_millis} milliseconds)"))

    return del_trace
