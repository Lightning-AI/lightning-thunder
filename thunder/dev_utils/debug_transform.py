from thunder.core.trace import TraceCtx as Trace, from_trace, TraceProvenance
from thunder.core.prims import PrimIDs
import thunder
from functools import partial
from thunder.core.symbol import Symbol
from thunder.dev_utils.utils import NON_COMPUTATION_PRIMS


def debug_impl(*args, computation_bsym, debug_bsym, callback, **kwargs):
    debug_bsym.header = callback(computation_bsym, *args, **kwargs)


class DebugTransform(thunder.core.transforms.PostOptimizationTransform):
    def __init__(self, callback):
        self.callback = callback

    def transform_trace(self, trace: Trace, **kwargs) -> Trace:
        debug_trace = from_trace(trace)
        cnt = 1
        for bound_symbol in trace.bound_symbols:
            # Synchronize and stop profiling at return.
            if PrimIDs.RETURN == bound_symbol.sym.id:

                debug_trace.bound_symbols.append(bound_symbol)
                break

            if bound_symbol.sym.id in NON_COMPUTATION_PRIMS:
                # Just append the symbol.
                debug_trace.bound_symbols.append(bound_symbol)
                continue

            debug_sym_name = f"debug_{cnt}"

            def bind_postprocess(bsym) -> None:
                # This dict is then used by trace.python_ctx() to resolve the
                # BoundSymbol to the actual function.
                bsym._call_ctx = {
                    debug_sym_name: partial(
                        debug_impl, computation_bsym=bound_symbol, debug_bsym=bsym, callback=self.callback
                    )
                }

            debug_sym = Symbol(
                debug_sym_name, lambda *args, **kwargs: None, is_prim=True, _bind_postprocess=bind_postprocess
            )
            debug_bsym = debug_sym.bind(*bound_symbol.args, output=None, **bound_symbol.kwargs)

            debug_trace.bound_symbols.append(debug_bsym)
            debug_trace.bound_symbols.append(bound_symbol)
            cnt += 1

        debug_trace.set_provenance(TraceProvenance("Debug Transform"))
        return debug_trace
