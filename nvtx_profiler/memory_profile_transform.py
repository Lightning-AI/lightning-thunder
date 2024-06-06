from thunder.core.trace import TraceCtx as Trace, from_trace, TraceProvenance
import thunder.core.prims as prims
from thunder.core.prims import PrimIDs
from thunder.extend import OperatorExecutor
import time
import torch
import thunder
from functools import partial
from thunder.core.symbol import Symbol


memory_profiler_ex = OperatorExecutor("memory_profiler_ex")

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
)


def memory_checker_impl(bsym):
    bsym.header = f"Memory Allocated: {torch.cuda.memory_allocated()}"


class MemoryProfileTransform(thunder.core.transforms.PostOptimizationTransform):
    def __call__(self, trace: Trace, **kwargs) -> Trace:
        profile_trace = from_trace(trace)
        cnt = 1
        for bound_symbol in trace.bound_symbols:
            # Synchronize and stop profiling at return.
            if PrimIDs.RETURN == bound_symbol.sym.id:

                profile_trace.bound_symbols.append(bound_symbol)
                break

            if bound_symbol.sym.id in NON_COMPUTATION_PRIMS:
                # Just append the symbol.
                profile_trace.bound_symbols.append(bound_symbol)
                continue

            def bind_postprocess(bsym) -> None:
                # This dict is then used by trace.python_ctx() to resolve the
                # BoundSymbol to the actual function.
                bsym._call_ctx = {f"memory_checker_{cnt}": partial(memory_checker_impl, bsym=bsym)}

            mem_sym = Symbol(f"memory_checker_{cnt}", lambda: None, is_prim=True, _bind_postprocess=bind_postprocess)
            mem_bsym = mem_sym.bind(output=None)

            profile_trace.bound_symbols.append(mem_bsym)
            profile_trace.bound_symbols.append(bound_symbol)
            cnt += 1

        return profile_trace
