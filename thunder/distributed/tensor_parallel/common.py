from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from enum import Enum
from enum import auto
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable
    from torch.distributed import ProcessGroup
    from thunder.common import CompileData
    from thunder.core.proxies import ProxyInterface
    from thunder.core.proxies import TensorProxy
    from thunder.core.symbol import BoundSymbol
    from thunder.core.trace import TraceCtx
    from thunder.core.trace import TraceProvenance
    from thunder.core.trace import VariableInterface
    from thunder.core.transforms import VISIT_TYPE


__all__ = [
    "ComputationTraceTransformVisitorForTensorParallel",
    "TensorParallelLayerType",
    "NoOp",
    "PrePostProcessInterface",
    "TransformForTensorParallel",
]


class TensorParallelLayerType(Enum):
    COLUMN_PARALLEL_LINEAR = auto()
    ROW_PARALLEL_LINEAR = auto()

    COLUMN_PARALLEL_EMBED = auto()
    ROW_PARALLEL_EMBED = auto()


class PrePostProcessInterface(ABC):
    """Defining interfaces of pre/post-process of tensor parallelized ops."""

    @abstractmethod
    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        """Apply preprocessing to tensro parallel op's inputs.

        The second return value could be consumed by :func:`PrePostProcessInterface.postprocess`.
        """
        return x, (None,)

    @abstractmethod
    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        """Apply postprocessing to tensor parallel op's outputs."""
        return y


@dataclass(frozen=True)
class NoOp(PrePostProcessInterface):
    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        return super().preprocess(x)

    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        return super().postprocess(y)


@dataclass
class ComputationTraceTransformVisitorForTensorParallel:
    """Wrap tensor parallel ops with necessary preprocessing and postprocessing.

    With the reference of ``bsyms_before_allgather``, this takes care of inputs and ouptuts of
    tensor parallel ops by applying defined processings. Each pair of them is suppoed to be defined
    based on :clss:`PrePostProcessInterface`.
    """

    bsym2prepostprocess: dict[BoundSymbol, PrePostProcessInterface]

    def __post_init__(self):
        self.swap_map: dict[VariableInterface, ProxyInterface] = {}

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        from thunder.core.transforms import VISIT_TYPE
        from thunder.core.trace import get_tracectx
        from thunder.core.proxies import variableify

        pre_post_process: PrePostProcessInterface | None = None
        if bsym in self.bsym2prepostprocess:
            pre_post_process = self.bsym2prepostprocess[bsym]
            orig_arg = bsym.flat_proxy_args[0]
            new_arg, preprocess_artifacts = pre_post_process.preprocess(orig_arg)
            if new_arg.name != orig_arg.name:
                self.swap_map[variableify(orig_arg)] = new_arg

        new_bsym = bsym.from_bsym_swap_proxies(self.swap_map, skip_output=True)
        trace = get_tracectx()
        trace.scopes[-1].append(new_bsym)

        if bsym in self.bsym2prepostprocess:
            y = bsym.flat_proxy_outs[0]
            processed_y = pre_post_process.postprocess(y, preprocess_artifacts)
            self.swap_map[variableify(y)] = processed_y

        return VISIT_TYPE.REPLACE


@dataclass
class TransformForTensorParallel:
    rank: int
    world_size: int
    compile_data: CompileData
    chunked_param_name2layer_type: set[str]
    process_group: ProcessGroup

    def __post_init__(self):
        from thunder.common import CompileData
        from thunder.core import utils

        utils.check_type(self.compile_data, CompileData)
        if getattr(self.compile_data, "use_fsdp", False) or getattr(self.compile_data.fn, "use_fsdp", False):
            raise NotImplementedError("Currently thunder does not support the combination of fsdp and tensor parallel")

    def get_visitor_of_computation_trc_and_provenance(
        self,
        prologue_trace: TraceCtx,
        computation_trace: TraceCtx,
    ) -> tuple[Callable[[BoundSymbol], VISIT_TYPE], TraceProvenance | str]:
        raise NotImplementedError("Inherit this class and implement `get_visitor_of_computation_trc`")

    def __call__(
        self,
        prologue_trace: TraceCtx,
        computation_trace: TraceCtx,
        epilogue_trace: TraceCtx,
        **kwargs,
    ) -> tuple[TraceCtx, TraceCtx, TraceCtx]:
        from thunder.core import prims
        from thunder.core import utils
        from thunder.core.pytree import tree_flatten
        from thunder.core.transforms import visitor_transform

        modules_and_thunder_modules = [
            (bsym.args[0], bsym.output)
            for bsym in prologue_trace.bound_symbols
            if bsym.sym is prims.unpack_thunder_module
        ]

        if len(modules_and_thunder_modules) != 1:
            raise NotImplementedError("cannot deal with modules other than the compiled module")

        visit, provenance = self.get_visitor_of_computation_trc_and_provenance(
            prologue_trace=prologue_trace,
            computation_trace=computation_trace,
        )
        new_computation_trace = visitor_transform(
            computation_trace,
            visit=visit,
            provenance=provenance,
        )
        return prologue_trace, new_computation_trace, epilogue_trace
