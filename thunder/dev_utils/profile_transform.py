import thunder
import torch
import contextlib
import re

from thunder.core import prims
from thunder.core.symbol import Symbol
from thunder.core.trace import TraceTag


def create_boundsymbol(name: str, bsym, fn):
    def bind_postprocess(debug_bsym):
        debug_bsym._call_ctx = {name: fn}

    debug_sym = Symbol(name, lambda *_, **__: None, is_prim=True, _bind_postprocess=bind_postprocess)
    if bsym is not None:
        debug_bsym = debug_sym.bind(*bsym.args, output=bsym.output, **bsym.kwargs)
    else:
        debug_bsym = debug_sym.bind(output=None)
    return debug_bsym


class ProfileTransform(thunder.core.transform_common.Transform):
    def __init__(self, *, warmup_runs=3, number_runs=1, start_idx=0, end_idx=None, input_match=None, backward=False):
        self.input_match = input_match
        if input_match is None:
            self.start_idx = start_idx
            self.end_idx = end_idx if end_idx is not None else -1
        else:
            self.match_start_idx = start_idx
            self.match_end_idx = end_idx if end_idx is not None else 1

        self.enabled = True
        self.run_counter = 0
        self.warmup_runs = warmup_runs
        self.number_runs = number_runs
        self.prof = None
        self.backward = backward
        self.computed_enabled = False

    def start_profile(self):
        self.run_counter += 1
        self.computed_enabled = (
            self.enabled
            and (self.run_counter > self.warmup_runs)
            and (self.run_counter <= (self.warmup_runs + self.number_runs))
        )
        if self.computed_enabled:
            self.prof = torch.profiler.profile(with_stack=True)
            self.prof.__enter__()

    def end_profile(self):
        if self.computed_enabled:
            self.prof.__exit__(None, None, None)

    def get_profile(self):
        return self.prof

    @contextlib.contextmanager
    def record_function(self, name):
        if self.computed_enabled:
            with torch.profiler.record_function(name):
                yield
        else:
            yield

    def _make_impl(self, trace, bsym, *, name):
        arg_list = [
            thunder.core.codeutils.prettyprint(a) if isinstance(a, thunder.Proxy) else f"_{i}"
            for i, a in enumerate(bsym.args)
        ]
        with thunder.core.trace.tracectx(thunder.core.trace.TraceCtx()):
            kwarg_dict = {
                k: thunder.core.proxies.AnyProxy(None, name=k) for i, (k, v) in enumerate(bsym.kwargs.items())
            }
        arg_str = ", ".join(arg_list)
        if kwarg_dict:
            if arg_str:
                arg_str += ", *, "
            else:
                arg_str += "*, "
            arg_str += ", ".join(kwarg_dict.keys())
        oname = name
        (name,) = bsym.python(indent=1, print_depth=1)
        name = name.replace('"', "'")

        declstr = f"def fn({arg_str}):"
        record_function_str = f" with self.record_function({repr(name)}):"
        return_str = "  return " + thunder.core.codeutils.prettyprint(bsym.output)

        bsym = bsym.from_bsym(kwargs=kwarg_dict)
        function_code = "\n".join([declstr, record_function_str, *bsym.python(indent=1), return_str])
        ctx = trace.python_ctx().copy()
        ctx["self"] = self
        fn = thunder.core.baseutils.build_callable("fn", function_code, oname, ctx)
        fn.__name__ = name
        fn.__qualname__ = name
        fn.__code__ = fn.__code__.replace(co_name=name)
        return fn

    def transform_trace_post_optimization(self, computation_trace, **kwargs):
        if self.backward ^ (TraceTag.BACKWARD in computation_trace.tags):
            return computation_trace

        if self.input_match is not None:
            self.match_list = []
            for i, bsym in enumerate(computation_trace.bound_symbols):
                for a in bsym.args:
                    if isinstance(a, thunder.TensorProxy) and re.match(self.input_match, a.name):
                        self.match_list.append((i, a.name))
            start_idx = self.match_list[self.match_start_idx][0]
            end_idx = self.match_list[self.match_end_idx][0]
        else:
            start_idx = self.start_idx
            end_idx = self.end_idx

        new_bound_symbols = []

        new_trace = thunder.core.trace.from_trace(computation_trace)

        need_end = False

        for i, bsym in enumerate(computation_trace.bound_symbols[:]):
            if i == end_idx or (bsym.sym == prims.python_return and need_end):
                need_end = False
                new_bound_symbols.append(create_boundsymbol("end_profiling", None, self.end_profile))
            if bsym.sym in {
                prims.unpack_trivial,
                prims.unpack_sequence,
                prims.python_return,
                prims.python_del,
                prims.update_aliases,
                thunder.executors.torchex.update_aliases,
            }:
                new_bound_symbols.append(bsym)
                if i == start_idx:
                    start_idx += 1
                continue
            if i == start_idx:
                need_end = True
                new_bound_symbols.append(create_boundsymbol("start_profiling", None, self.start_profile))
            new_bound_symbols.append(
                create_boundsymbol(
                    f"{bsym.sym.name}_{i}", bsym, self._make_impl(computation_trace, bsym, name=f"{bsym.sym.name}_{i}")
                )
            )

        new_trace.bound_symbols = new_bound_symbols
        return new_trace
