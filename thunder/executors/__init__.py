from typing import Optional, Any, List, Tuple
from collections.abc import Sequence
from contextvars import ContextVar

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

_default_executors = [Executor.TORCH, Executor.PYTHON]

if nvfuser_available():
    import thunder.executors.nvfuserex as nvfuserex

    _executor_map[Executor.NVFUSER] = nvfuserex
    _default_executors = [Executor.NVFUSER] + _default_executors


# Creates ContextVar
_executorsctx = ContextVar("executors", default=_executor_map)
_default_executorsctx = ContextVar("default_executors", default=_default_executors)


def set_executorsctx(ctx):
    """Sets the executors mapping"""

    return _executorsctx.set(ctx)


# NOTE The executorsctx is expected to always have a value, because it has a
#   default value (set above)
def get_executorsctx():
    return _executorsctx.get()


def set_default_executorsctx(ctx):
    """Sets the executors mapping"""

    return _default_executorsctx.set(ctx)


# NOTE The default_executorsctx is expected to always have a value, because it has a
#   default value (set above)
def get_default_executorsctx():
    return _default_executorsctx.get()


def add_executor(id, executor, *, add_to_default_executors: bool = True):
    executor_map = get_executorsctx()

    executor_map[id] = executor
    set_executorsctx(executor_map)

    if add_to_default_executors:
        defaults = list_default_executors()
        defaults = (id,) + tuple(defaults)
        set_default_executors(defaults)


def list_executors() -> tuple:
    executor_map = get_executorsctx()
    return tuple(executor_map.keys())


def list_default_executors() -> tuple:
    return tuple(get_default_executorsctx())


def set_default_executors(new_defaults: Sequence):
    executor_map = get_executorsctx()

    # Validates each entry in the sequence is an executor
    for x in new_defaults:
        if x not in executor_map:
            raise ValueError(f"Trying to set an unknown executor {x} as a default executor!")

    set_default_executorsctx(new_defaults)


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
        executors_list = list_default_executors()

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

    redundant_removed, redundant_removed_traces = passes.remove_redundant_casts(flattened_trace)
    traces.extend(redundant_removed_traces)

    postflatten_dce_trace, postflatten_dce_traces = passes.dce(redundant_removed)
    traces.extend(postflatten_dce_traces)

    fused_trace, fused_traces = passes.fuse(postflatten_dce_trace)
    traces.extend(fused_traces)

    if use_rematerialization:
        remat_trace, remat_traces = rematerialize(fused_trace)
        traces.extend(remat_traces)
        fused_trace = remat_trace

    lifetime_trace, lifetime_traces = passes.del_last_used(fused_trace)
    traces.extend(lifetime_traces)

    return lifetime_trace, traces


from thunder.core.symbol import Symbol, BoundSymbol
from thunder.executors.utils import Region


# Defines an executor that can take over individual operations
class OpExecutor:
    def __init__(self, name, op_map):
        self._name = name
        self.op_map = op_map

    def name(self) -> Any:
        return self._name

    def is_supported(self, bsym: BoundSymbol, *, prims_only: bool = False) -> bool:
        sym = bsym.sym

        if prims_only and not sym.is_prim:
            return False

        _, check, _ = self.op_map.get(sym.id, (None, None, None))
        if check is None:
            return False
        return check(*bsym.args, **bsym.kwargs)

    def can_execute(self, bsym: BoundSymbol, *, prims_only: bool = False) -> bool:
        sym = bsym.sym

        if self.is_supported(bsym, prims_only=prims_only):
            return True

        if len(bsym.subsymbols) == 0:
            return False

        # Checks if all the operations this calls are executable
        can_execute_ = True
        for ssym in bsym.subsymbols:
            if not self.can_execute(ssym, prims_only=prims_only):
                can_execute_ = False
                break

        return can_execute_

    def fuse(self, region: Region) -> list[BoundSymbol]:
        bsyms: List[BoundSymbol] = []

        for bsym in region.bound_symbols:
            name, _, impl = self.op_map[bsym.sym.id]
            ctx = {name: impl}
            sym = Symbol(name=name, meta=None)
            bsym = BoundSymbol(
                sym,
                args=bsym.args,
                kwargs=bsym.kwargs,
                output=bsym.output,
                subsymbols=(),
                _call_ctx=ctx,
            )
            bsyms.append(bsym)

        return bsyms


def add_operator_executor(name, op_map, *, add_to_default_executors: bool = True):
    opex = OpExecutor(name, op_map)
    add_executor(name, opex, add_to_default_executors=add_to_default_executors)
