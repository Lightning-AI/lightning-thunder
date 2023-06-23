from typing import Optional, Any, List, Tuple
from collections.abc import Sequence

from thunder.core.trace import TraceCtx
from thunder.core.rematerialization import rematerialize
import thunder.executors.torchex as torchex
import thunder.executors.pythonex as pythonex
from thunder.executors.utils import Executor, nvfuser_available, nvfuser_version
import thunder.executors.passes as passes


# NOTE This import must be guarded because nvFuser is not always available
# TODO Should we throw a warning if nvFuser isn't available, possibly with instructions for how to install it?
#   It is important we support CPU-only builds for development and debugging

# TODO Properly expose functions using __all__
__all__ = [
    "transform_for_execution",
]

NVFUSER = Executor.NVFUSER
TORCH = Executor.TORCH
PYTHON = Executor.PYTHON


_executor_map = {
    Executor.TORCH: torchex,
    Executor.PYTHON: pythonex,
}

if nvfuser_available():
    import thunder.executors.nvfuserex as nvfuserex

    _executor_map[Executor.NVFUSER] = nvfuserex


# TODO Improve on the "Any" annotation here
def get_executor(ex: Executor) -> Any:
    return _executor_map[ex]


# TODO Constraint generation based off executor requirements
# TODO Consider making this faster by reusing more data
# TODO Create a general mechanism for running traces that produces reproducible provenance and the
#   appropriate error checks
def transform_for_execution(
    trace: TraceCtx,
    executors_list: Optional[Sequence[Executor]] = None,
    *,
    only_execute_prims=False,
    use_rematerialization=False,
) -> tuple[TraceCtx, list[TraceCtx]]:
    # Acquires the executors
    if executors_list is None:
        executors_list: list[Executor]
        if nvfuser_available():
            executors_list = [Executor.NVFUSER, Executor.TORCH, Executor.PYTHON]
        else:
            executors_list = [Executor.TORCH, Executor.PYTHON]

    # NOTE The Python executor is required to execute any nontrivial program
    if PYTHON not in executors_list:
        executors_list.append(PYTHON)

    if torchex._is_autocast_enabled():
        raise RuntimeError(
            "A callable optimized by thunder will not respect `torch.autocast`. "
            "If your use case needs to use `torch.autocast`, file a feature request issue."
        )

    # Translates executor names to actual executors
    executors_list = tuple(_executor_map[ex] for ex in executors_list)

    traces: list[TraceCtx] = []

    try:
        dce_trace, dce_traces = passes.dce(trace)
        traces.extend(dce_traces)
    except Exception as e:
        print(f"The dead code elimination pass failed when invoked on:\n{trace}")
        raise e

    claimed_trace, claimed_traces = passes.claim(dce_trace, executors_list, prims_only=only_execute_prims)
    traces.extend(claimed_traces)

    flattened_trace, flattened_traces = passes.flatten(claimed_trace, prims_only=only_execute_prims)
    traces.extend(flattened_traces)

    flattened_dce_trace, flattened_dce_traces = passes.dce(flattened_trace)
    traces.extend(flattened_dce_traces)

    fused_trace, fused_traces = passes.fuse(flattened_dce_trace)
    traces.extend(fused_traces)

    if use_rematerialization:
        remat_trace, remat_traces = rematerialize(fused_trace)
        traces.extend(remat_traces)
        fused_trace = remat_trace

    lifetime_trace, lifetime_traces = passes.del_last_used(fused_trace)
    traces.extend(lifetime_traces)

    return lifetime_trace, traces
