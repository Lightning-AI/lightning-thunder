from enum import Enum, auto
from typing import List, Set, Dict, Callable, Optional, Sequence

import torch
from looseversion import LooseVersion

from thunder.core.symbol import BoundSymbol
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
import thunder.core.utils as utils
from thunder.core.proxies import Variable, variableify, Proxy, unvariableify

import thunder.executors.passes as passes

# TODO Consider renaming this file to common.py?


class Executor(Enum):
    NVFUSER = auto()
    TORCH = auto()
    PYTHON = auto()


# NOTE This is here because we can only import the nvFuser executor conditional on its being available
def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def nvfuser_version() -> Optional[LooseVersion]:
    # Short-circuits if CUDA isn't available
    if not is_cuda_available():
        return None

    try:
        import nvfuser

        if hasattr(nvfuser, "version"):
            return LooseVersion(nvfuser.version())

        # NOTE: This import of nvFuser may or may not have version info
        return LooseVersion("0.0.0")
    except ImportError:
        pass

    try:
        # NOTE This import of nvFuser is so old it didn't have version info
        import torch._C._nvfuser as nvfuser

        return LooseVersion("0.0.0")
    except ImportError:
        pass

    # NOTE This occurs when both attempts at importing nvFuser failed
    return None


def required_nvfuser_version() -> LooseVersion:
    return LooseVersion("0.0.1")


# NOTE We require nvFuser version 0.0.1 or greater
def nvfuser_available() -> bool:
    v = nvfuser_version()
    return v is not None and v >= required_nvfuser_version()


# TODO Document this better
# TODO Review non-proxy inputs as being consumed -- currently only proxies can be inputs and outputs of these regions
class Region:
    # Identifies the inputs and outputs for the given sequence of bound symbols
    def __init__(self, trace: TraceCtx, producers, consumers, bound_symbols: Sequence[BoundSymbol]):
        self.bound_symbols = bound_symbols

        consumes = set()
        produces = set()

        for bsym in self.bound_symbols:
            flatouts = bsym._flat_outs

            produces.update(
                variableify(x) for x in flatouts
                if isinstance(x, Proxy) and producers[x] in self.bound_symbols
            )

            consumes.update(variableify(x) for x in bsym._flat_args if isinstance(x, Proxy))
            consumes.update(variableify(x) for x in bsym._flat_kwargs if isinstance(x, Proxy))

            # TODO Revise constant modeling
            # self.constants.update(variableify(trace.get_object_meta(x)) for x in flatargs if trace.is_constant(x))
            # self.constants.update(variableify(trace.get_object_meta(x)) for x in flatkwargs if trace.is_constant(x))

        self.inputs = set()
        self.outputs = set()

        # Inputs are things which this consumes which are produced before it
        for x in consumes:
            x = unvariableify(x)
            if producers[x] not in self.bound_symbols:
                self.inputs.add(variableify(x))

        # Outputs are things this produces that are consumed after it
        for x in produces:
            x = unvariableify(x)
            consumed_by = consumers[x]
            for bsym in consumed_by:
                if bsym not in self.bound_symbols:
                    self.outputs.add(variableify(x))
                    break
