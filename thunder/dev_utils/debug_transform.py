from collections.abc import Callable
from functools import partial
import time

from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
import thunder
from thunder.core.symbol import Symbol, BoundSymbol
from thunder.dev_utils.utils import NON_COMPUTATION_PRIMS


def create_debug_boundsymbol(name: str, bsym: BoundSymbol, call_ctx: Callable):
    def bind_postprocess(debug_bsym):
        debug_bsym._call_ctx = {name: partial(call_ctx, debug_bsym, bsym)}

    debug_sym = Symbol(name, lambda *_, **__: None, is_prim=True, _bind_postprocess=bind_postprocess)
    debug_bsym = debug_sym.bind(*bsym.args, output=None, **bsym.kwargs)
    return debug_bsym


class _DebugTransform(thunder.core.transforms.Transform):
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

                def _pre_call_ctx(pre_debug_bsym, bsym, *args, **kwargs):
                    out = self.pre_callback(bsym, *args, **kwargs)
                    thunder.core.utils.check_type(out, str)
                    pre_debug_bsym.header = out

                pre_debug_name = f"debug_pre_{sym_name}{debug_counter}"
                pre_debug_bsym = create_debug_boundsymbol(pre_debug_name, bsym, _pre_call_ctx)
                new_bsyms.append(pre_debug_bsym)

            new_bsyms.append(bsym)

            if self.post_callback is not None:

                def _post_call_ctx(post_debug_bsym, bsym, *args, **kwargs):
                    out = self.post_callback(bsym, *args, **kwargs)
                    thunder.core.utils.check_type(out, str)
                    post_debug_bsym.header = out

                post_debug_name = f"debug_post_{sym_name}{debug_counter}"
                post_debug_bsym = create_debug_boundsymbol(post_debug_name, bsym, _post_call_ctx)
                new_bsyms.append(post_debug_bsym)

            debug_counter += 1

        debug_trace.bound_symbols = new_bsyms
        elapsed_time_ns = time.perf_counter_ns() - start_time_ns

        debug_trace.set_provenance(TraceProvenance(f"Debug trace (took {elapsed_time_ns * 1e-6:.2f} milliseconds)"))

        return debug_trace


def debug_execution_trace(cfn, pre_callback: Callable | None = None, post_callback: Callable | None = None):
    """
    Adds a debugging transform to the trace allowing pre and post execution callbacks.

    The function inserts debug symbols in the computation traces to call the callbacks before and/or after each symbol
    in the trace. These callbacks can be used to inspect or log information about the execution of the computation.

    Args:
        cfn: :func:`thunder.jit` function to debug.
        pre_callback: An optional callable that is executed before each bound symbol is processed.
            It should have the signature ``(BoundSymbol, *args, **kwargs)`` and return a string. If :obj:`None`, no
            pre-execution callback is used.
        post_callback: An optional callable that is executed after each bound symbol is processed.
            It should have the signature ``(BoundSymbol, *args, **kwargs)`` and return a string. If :obj:`None`, no
            post-execution callback is used.
    """
    if pre_callback is None and post_callback is None:
        raise RuntimeError(
            "debug_execution_trace: Both `pre_callback` and `post_callback` were None, expected atleast one of them to not be None."
        )
    _debug_transform = _DebugTransform(pre_callback=pre_callback, post_callback=post_callback)
    return thunder.core.transforms.add_transform(cfn, transform=_debug_transform)
