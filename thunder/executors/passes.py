from typing import Dict, Any, List, Tuple, Optional
from collections.abc import Callable
from collections.abc import Sequence
from collections import deque
from dataclasses import replace
from itertools import chain
from functools import partial
import time

from thunder.core.trace import TraceCtx, from_trace, TraceProvenance, VariableInterface
import thunder.core.dtypes as dtypes
import thunder.core.utils as cutils
from thunder.core.utils import ProxyDict, check
from thunder.core.symbol import BoundSymbol
from thunder.core.pytree import tree_flatten, tree_unflatten, tree_map
import thunder.core.prims as prims
from thunder.core.proxies import Proxy, variableify, unvariableify, Variable
import thunder.core.transforms as transforms
from thunder.core.transform_common import dce

from thunder.extend import Executor, get_always_executors, OperatorExecutor, FusionExecutor

comment_symbols = {prims.PrimIDs.COMMENT, prims.PrimIDs.UNPACK_TRIVIAL}


# Transforms a trace by determining which execution transforms to call given the list of executors in priority order
def _transform_for_operator_executor_execution(trace: TraceCtx, executors_list: Sequence[Executor]) -> TraceCtx:
    swapmap: dict[Variable, Proxy] = {}

    def update_swapmap_(original: BoundSymbol, new_output: Any) -> None:
        new_flat_outs, _ = tree_flatten(new_output)

        for o, no in zip(original.flat_outs, new_flat_outs):
            if isinstance(o, Proxy):
                check(
                    isinstance(no, Proxy),
                    lambda: f"Expected an execution transform to produce outputs with the same type, but found {type(o)} and {type(no)}",
                )

                vo = variableify(o)
                vno = variableify(no)
                if vo == vno:
                    continue
                swapmap[vno] = o

    # TODO Consider using an enum for this function's return values
    # Tries to find an executor for the BoundSymbol
    #   If the BoundSymbol already has an executor then None is returned
    #   If the executor has an execution transform, it's called and True is returned
    #   If no executor can execute the BoundSymbol, False is returned
    def visit_helper_(bsym: BoundSymbol) -> None | bool:
        if bsym.sym.executor is not None or bsym.sym.python_impl is not None:
            return None

        ex: Executor
        for ex in executors_list:
            # TODO Consider allowing operator executors to claim portions of operations
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
                    # NOTE It'd be nice to just preserve the original BoundSymbol here, but it's not really
                    #   clear how best to do that -- maybe we should support acquiring and editing the scope
                    #   directly in the visitor transform (updating the signature of visit to be (bsym, scope))
                    out = bsym.sym(*bsym.args, **bsym.kwargs)
                else:
                    raise AssertionError("Unknown executor")

                update_swapmap_(bsym, out)
                return True

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


# TODO Review deleting non-proxies
def del_last_used(trace: TraceCtx) -> TraceCtx:
    """Mark last used intermediates to be deleted. This lets the Python garbage collector free
        unused tensor memory.

    Args:
        trace: trace to be transformed
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

        # NOTE The check for return avoids putting dels after the return statement
        if len(to_del) > 0 and bsym.sym.id is not prims.PrimIDs.RETURN:
            # NOTE The following logic just helps the deletions print prettier
            del_sym: BoundSymbol
            if len(to_del) > 1:
                del_sym = prims.python_del.bind(tuple(to_del), output=None)
            else:
                del_sym = prims.python_del.bind(*to_del, output=None)
            bsyms.appendleft(del_sym)

        bsyms.appendleft(bsym)

    del_trace.bound_symbols = list(bsyms)

    end_time_ns = time.time_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000

    del_trace.set_provenance(TraceProvenance(f"Delete Last Used (took {elapsed_time_millis} milliseconds)"))

    return del_trace
