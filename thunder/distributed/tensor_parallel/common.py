from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from abc import abstractproperty
from enum import Enum
from enum import auto
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable
    from torch.distributed import ProcessGroup
    from thunder.common import CompileData
    from thunder.core.proxies import ProxyInterface
    from thunder.core.proxies import TensorProxy
    from thunder.core.proxies import DistParallelType
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
        """Apply preprocessing to tensor parallel op's inputs.

        The second return value could be consumed by :func:`PrePostProcessInterface.postprocess`.
        """
        return x, (None,)

    @abstractmethod
    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        """Apply postprocessing to tensor parallel op's outputs."""
        return y

    def maybe_modify_args_and_kwargs(self, bsym: BoundSymbol) -> BoundSymbol:
        """No-op. Mainly for row-wise parallel linear."""
        return bsym


@dataclass(frozen=True)
class NoOp(PrePostProcessInterface):
    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        return super().preprocess(x)

    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        return super().postprocess(y)


@dataclass
class ComputationTraceTransformVisitorForTensorParallel:
    """Wrap tensor parallel ops with necessary preprocessing and postprocessing.

    With the reference of ``bsyms_before_allgather``, this takes care of inputs and outputs of
    tensor parallel ops by applying defined processings. Each pair of them is supposed to be defined
    based on :clss:`PrePostProcessInterface`.
    """

    bsym_to_prepostprocess: dict[BoundSymbol, PrePostProcessInterface]

    def __post_init__(self):
        self.swap_map: dict[VariableInterface, ProxyInterface] = {}

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        from thunder.core.transforms import VISIT_TYPE
        from thunder.core.trace import get_tracectx
        from thunder.core.proxies import variableify

        pre_post_process: PrePostProcessInterface | None = None
        if bsym in self.bsym_to_prepostprocess:
            pre_post_process = self.bsym_to_prepostprocess[bsym]
            orig_arg = bsym.flat_proxy_args[0]
            new_arg, preprocess_artifacts = pre_post_process.preprocess(orig_arg)
            if new_arg.name != orig_arg.name:
                self.swap_map[variableify(orig_arg)] = new_arg

        new_bsym = bsym.from_bsym_swap_proxies(self.swap_map, skip_output=True)
        if pre_post_process is not None:
            new_bsym = pre_post_process.maybe_modify_args_and_kwargs(new_bsym)
        trace = get_tracectx()
        trace.scopes[-1].append(new_bsym)

        if pre_post_process is not None:
            y = bsym.flat_proxy_outs[0]
            processed_y = pre_post_process.postprocess(y, preprocess_artifacts)
            self.swap_map[variableify(y)] = processed_y

        return VISIT_TYPE.REPLACE


@dataclass
class TransformForTensorParallel:
    rank: int
    world_size: int
    compile_data: CompileData
    chunked_param_name_to_layer_type: dict[str, Any]
    process_group: ProcessGroup

    def __post_init__(self):
        from thunder.common import CompileData
        from thunder.core import utils

        utils.check_type(self.compile_data, CompileData)
        if getattr(self.compile_data, "use_fsdp", False) or getattr(self.compile_data.fn, "use_fsdp", False):
            raise NotImplementedError("Currently thunder does not support the combination of fsdp and tensor parallel")

    @abstractmethod
    def get_visitor_of_computation_trace_and_provenance(
        self,
        computation_trace: TraceCtx,
    ) -> tuple[Callable[[BoundSymbol], VISIT_TYPE], TraceProvenance | str]: ...

    @abstractmethod
    def _calc_new_shape(self, orig_shape) -> tuple[int, ...]: ...

    @abstractproperty
    def distparallel_type(self) -> DistParallelType: ...

    def __call__(
        self,
        prologue_trace: TraceCtx,
        computation_trace: TraceCtx,
        epilogue_trace: TraceCtx,
        **kwargs,
    ) -> tuple[TraceCtx, TraceCtx, TraceCtx]:
        from thunder.core import prims
        from thunder.core import utils
        from thunder.core.transforms import visitor_transform

        modules_and_thunder_modules = [
            (bsym.args[0], bsym.output)
            for bsym in prologue_trace.bound_symbols
            if bsym.sym is prims.unpack_thunder_module
        ]
        ((_, thunder_module_proxy),) = modules_and_thunder_modules

        prologue_producers, prologue_consumers = utils.producers_and_consumers(prologue_trace)
        pro_out_p: TensorProxy
        comp_inp_p: TensorProxy
        for pro_out_p, comp_inp_p in zip(prologue_trace.output, computation_trace.args):
            if pro_out_p.name not in self.chunked_param_name_to_layer_type:
                continue
            bsym = prologue_producers[pro_out_p]
            if bsym.sym.id == prims.PrimIDs.UNPACK_PARAMETER:
                param_thunder_module, param_name = bsym.args
                assert param_thunder_module is thunder_module_proxy

                if (
                    proxy_like_param_name := f"""t_{param_name.replace(".", "_")}"""
                ) in self.chunked_param_name_to_layer_type:

                    orig_shape = list(pro_out_p._shape)
                    new_shape = self._calc_new_shape(orig_shape)
                    pro_out_p._shape = new_shape
                    utils.check(
                        comp_inp_p.distparallel_type in (self.distparallel_type, DistParallelType.NONE),
                        lambda: f"{comp_inp_p.distparallel_type = } is not compatible with {self.distparallel_type=}",
                    )
                    pro_out_p._distparallel_type = self.distparallel_type
                    if comp_inp_p is not pro_out_p:
                        comp_inp_p._shape = new_shape
                        comp_inp_p._distparallel_type = self.distparallel_type

                    for c in prologue_consumers[pro_out_p]:
                        if c.sym is prims.check_tensor_shape_and_metadata:
                            # TODO have a more principled way to update this?
                            a0, _, _, *a2pp = c.args
                            c.args = (a0, tuple(new_shape), str(a0.device), *a2pp)

        for bsym in prologue_trace.bound_symbols:
            if bsym.sym is prims.check_tensor_shape_and_metadata and prologue_producers[bsym.args[0]].sym in (
                prims.unpack_parameter,
                prims.unpack_buffer,
            ):
                param_thunder_module, name = prologue_producers[bsym.args[0]].args
                assert param_thunder_module is thunder_module_proxy
                if name not in self.chunked_param_name_to_layer_type:
                    a0, shape, _, *a2pp = bsym.args
                    bsym.args = (a0, shape, str(a0.device), *a2pp)

        if len(modules_and_thunder_modules) != 1:
            raise NotImplementedError("cannot deal with modules other than the compiled module")

        visit, provenance = self.get_visitor_of_computation_trace_and_provenance(
            computation_trace=computation_trace,
        )
        new_computation_trace = visitor_transform(
            computation_trace,
            visit=visit,
            provenance=provenance,
        )
        return prologue_trace, new_computation_trace, epilogue_trace
