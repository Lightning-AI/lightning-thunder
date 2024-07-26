from thunder.core.trace import TraceCtx as Trace, from_trace, TraceProvenance
import thunder
from functools import partial
from thunder.core.symbol import Symbol
from thunder.dev_utils.utils import NON_COMPUTATION_PRIMS


# NOTE: `computation_bsym, debug_bsym, callback` are mandatory keyword-only arguments.
def debug_impl(*args, computation_bsym, debug_bsym, callback, **kwargs):
    output = callback(computation_bsym, *args, **kwargs)
    thunder.core.utils.check_type(output, str)
    debug_bsym.header = output


class DebugTransform(thunder.core.transforms.Transform):
    def __init__(self, callback):
        self.callback = callback

    def transform_trace_post_optimization(self, trace: Trace, **kwargs) -> Trace:
        debug_trace = from_trace(trace)
        cnt = 1
        for bound_symbol in trace.bound_symbols:
            if bound_symbol.sym.id in NON_COMPUTATION_PRIMS:
                # Just append the symbol.
                debug_trace.bound_symbols.append(bound_symbol)
                continue

            # we need unique name for each symbol.
            debug_sym_name = f"debug_{bound_symbol.sym.name}_{cnt}"

            def bind_postprocess(bsym) -> None:
                # This dict is then used by trace.python_ctx() to resolve the
                # BoundSymbol to the actual function.
                bsym._call_ctx = {
                    debug_sym_name: partial(
                        debug_impl, computation_bsym=bound_symbol, debug_bsym=bsym, callback=self.callback
                    )
                }

            # Create a new debug symbol for this bsym.
            debug_sym = Symbol(
                debug_sym_name, lambda *args, **kwargs: None, is_prim=True, _bind_postprocess=bind_postprocess
            )
            debug_bsym = debug_sym.bind(*bound_symbol.args, output=None, **bound_symbol.kwargs)

            debug_trace.bound_symbols.append(debug_bsym)
            debug_trace.bound_symbols.append(bound_symbol)
            cnt += 1

        debug_trace.set_provenance(TraceProvenance("Debug Transform"))
        return debug_trace
