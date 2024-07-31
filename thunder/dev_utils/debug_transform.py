from collections.abc import Callable
from functools import partial
import time

from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
import thunder
from thunder.core.symbol import Symbol, BoundSymbol
from thunder.dev_utils.utils import NON_COMPUTATION_PRIMS


def create_debug_boundsymbol(name: str, bsym: BoundSymbol, call_ctx: Callable):
    debug_sym = Symbol(name, lambda *_: None, is_prim=True)
    debug_bsym = debug_sym.bind(*bsym.args, output=None, _call_ctx={name: partial(call_ctx, bsym)}, **bsym.kwargs)
    return debug_bsym


class DebugTransform(thunder.core.transforms.Transform):
    def __init__(
        self,
        *,
        pre_callback: Callable[[tuple[BoundSymbol, ...]], str] | None = None,
        post_callback: Callable[[tuple[BoundSymbol, ...]], str] | None = None,
    ):
        self.pre_callback = pre_callback
        self.post_callback = post_callback

    def transform_trace_post_optimization(self, trace: TraceCtx, **kwargs) -> TraceCtx:
        start_time_ns = time.perf_counter_ns()
        debug_trace = from_trace(trace)
        debug_counter = 1

        new_bsyms: list[BoundSymbol] = []
        for bsym in trace.bound_symbols:
            sym_name = bsym.sym.name

            if bsym.sym.id in NON_COMPUTATION_PRIMS:
                new_bsyms.append(bsym)
                continue

            if self.pre_callback is not None:
                def _pre_call_ctx(bsym, *args, **kwargs):
                    out = self.pre_callback(bsym, *args, **kwargs)
                    thunder.core.utils.check_type(out, str)
                    pre_debug_bsym.header = out

                pre_debug_name = f"debug_pre_{sym_name}{debug_counter}"
                pre_debug_bsym = create_debug_boundsymbol(pre_debug_name, bsym, _pre_call_ctx)
                new_bsyms.append(pre_debug_bsym)

            new_bsyms.append(bsym)

            if self.post_callback is not None:
                def _post_call_ctx(bsym, *args, **kwargs):
                    out = self.post_callback(bsym, *args, **kwargs)
                    thunder.core.utils.check_type(out, str)
                    post_debug_bsym.header = out

                post_debug_name = f"debug_post_{sym_name}{debug_counter}"
                post_debug_bsym = create_debug_boundsymbol(post_debug_name, bsym, _post_call_ctx)
                new_bsyms.append(post_debug_bsym)

        debug_trace.bound_symbols = new_bsyms
        elapsed_time_ns = time.perf_counter_ns() - start_time_ns

        debug_trace.set_provenance(TraceProvenance(f"Debug trace (took {elapsed_time_ns * 1e-6:.2f} milliseconds)"))

        debug_counter += 1
        return debug_trace
