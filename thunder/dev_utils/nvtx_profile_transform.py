from thunder.core.trace import TraceCtx as Trace, from_trace, TraceProvenance
from thunder.core.prims import PrimIDs
from thunder.extend import OperatorExecutor
import time
import torch
import thunder


class Timer:
    def __init__(self):
        self.start_time_ns = None
        self.end_time_ns = None

    def __enter__(self):
        self.start_time_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.end_time_ns = time.perf_counter_ns()

    def get_elapsed_time_in_ms(self):
        elapsed_time_ns = self.end_time_ns - self.start_time_ns
        return elapsed_time_ns // int(1e6)


nvtx_profiler_ex = OperatorExecutor("nvtx_profiler_ex")


def nvtx_push_impl(msg):
    torch.cuda.nvtx.range_push(msg)


def nvtx_pop_impl():
    torch.cuda.nvtx.range_pop()


# Symbols for profiling.
nvtx_push = nvtx_profiler_ex.register_operator("nvtx_range_push", meta=lambda msg: None, fn=nvtx_push_impl)
nvtx_pop = nvtx_profiler_ex.register_operator("nvtx_range_pop", meta=lambda: None, fn=nvtx_pop_impl)

NON_COMPUTATION_PRIMS = (
    PrimIDs.ASSERT_TENSOR_METADATA,
    PrimIDs.CHECK_TENSOR_SHAPE_AND_METADATA,
    PrimIDs.CHECK_NONE,
    PrimIDs.CHECK_EMPTY,
    PrimIDs.CHECK_LITERAL_LIKE,
    PrimIDs.CHECK_TYPE,
    PrimIDs.CHECK_INSTANCE,
    PrimIDs.CHECK_NUMBER_TYPE_AND_VALUE,
    PrimIDs.CHECK_BOOL_CONVERSION,
    PrimIDs.CHECK_STRING_VALUE,
    PrimIDs.CHECK_LEN,
    PrimIDs.ASSERT_COMPARE,
    PrimIDs.PYTHON_VARS,
    PrimIDs.UNPACK_FUNCTION_OBJ,
    PrimIDs.UNPACK_CACHE_INFO,
    PrimIDs.UNPACK_ATTR,
    PrimIDs.UNPACK_GETITEM,
    PrimIDs.UNPACK_EMPTY_DICT,
    PrimIDs.UNPACK_ITER,
    PrimIDs.UNPACK_NEXT,
    PrimIDs.UNPACK_KEY,
    PrimIDs.UNPACK_SEQUENCE,
    PrimIDs.UNPACK_TRIVIAL,
    PrimIDs.UNPACK_TUPLE,
    PrimIDs.UNPACK_LIST,
    PrimIDs.UNPACK_DICT_KEY,
    PrimIDs.CONSTRUCT_TUPLE,
    PrimIDs.PACK_SETITEM,
    # TODO: UNPACK_SET
    # Utility prims
    PrimIDs.COMMENT,
    PrimIDs.DEL,
    PrimIDs.PRINT,
    PrimIDs.RETURN,
)


class NvtxProfileTransform(thunder.core.transforms.PostOptimizationTransform):
    def transform_trace(self, trace: Trace, **kwargs) -> Trace:
        with Timer() as timer:
            profile_trace = from_trace(trace)

            for bound_symbol in trace.bound_symbols:
                if bound_symbol.sym.id in NON_COMPUTATION_PRIMS:
                    profile_trace.bound_symbols.append(bound_symbol)
                    continue

                # Add nvtx range for the symbol.
                profile_trace.bound_symbols.append(
                    nvtx_push.bind(f"{''.join(bound_symbol.python(indent=0))}", output=None)
                )
                profile_trace.bound_symbols.append(bound_symbol)
                profile_trace.bound_symbols.append(nvtx_pop.bind(output=None))

        profile_trace.set_provenance(
            TraceProvenance(f"NVTX Profile Transform (took {timer.get_elapsed_time_in_ms()} milliseconds)")
        )
        return profile_trace
