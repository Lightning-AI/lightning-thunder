from __future__ import annotations
from abc import ABC
from abc import abstractmethod
import copy
from enum import Enum
from enum import auto
from dataclasses import dataclass
from dataclasses import field
from itertools import chain
from typing import TYPE_CHECKING

import torch

from thunder.core.module import ThunderModule
from thunder.core.proxies import DistParallelType
from thunder.core.proxies import TensorProxy
from thunder.core.proxies import variableify
from thunder.core.transform_common import Transform

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from torch.distributed import ProcessGroup
    from thunder.common import CompileData
    from thunder.core.module import ThunderModule
    from thunder.core.proxies import ProxyInterface
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


_TENSOR_PARALLLE_ENUM_VALS: set[DistParallelType] = {
    DistParallelType.COLUMN_WISE,
    DistParallelType.ROW_WISE,
}


@dataclass
class ComputationTraceTransformVisitorForTensorParallel:
    """Wrap tensor parallel ops with necessary preprocessing and postprocessing.

    With the reference of ``bsyms_before_allgather``, this takes care of inputs and outputs of
    tensor parallel ops by applying defined processings. Each pair of them is supposed to be defined
    based on :clss:`PrePostProcessInterface`.

    Args:
        bsym_to_prepostprocess:

    Attributes:
        swap_map: A map from the original output of a tensor-parallel opt to the post-processed output.
    """

    bsym_to_prepostprocess: dict[BoundSymbol, PrePostProcessInterface]
    distparallel_type: DistParallelType

    swap_map: dict[VariableInterface, ProxyInterface] = field(init=False, default_factory=dict)
    _has_other_tensor_parallel: bool = field(init=False, default=False)

    def _maybe_other_tensor_parallel(self, t: ProxyInterface) -> bool:
        if isinstance(t, TensorProxy) and not self._has_other_tensor_parallel:
            self._has_other_tensor_parallel = (
                t.distparallel_type in _TENSOR_PARALLLE_ENUM_VALS and t.distparallel_type != self.distparallel_type
            )

    @property
    def eligible_for_comm_optimization(self) -> bool:
        return self._has_other_tensor_parallel

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        from thunder.core.prims import PrimIDs
        from thunder.core.transforms import VISIT_TYPE
        from thunder.core.proxies import variableify

        if bsym.sym.id in {
            PrimIDs.UNPACK_TRIVIAL,
            PrimIDs.UNPACK_SEQUENCE,
            PrimIDs.UNPACK_KEY,
            PrimIDs.UNPACK_EMPTY_DICT,
        }:
            return VISIT_TYPE.NO_OP

        pre_post_process: PrePostProcessInterface | None = self.bsym_to_prepostprocess.get(bsym, None)
        new_bsym = bsym.from_bsym_swap_proxies(self.swap_map)
        for t in new_bsym.flat_proxy_args:
            self._maybe_other_tensor_parallel(t)

        if pre_post_process is not None:
            orig_arg = new_bsym.flat_proxy_args[0]
            new_arg, preprocess_artifacts = pre_post_process.preprocess(orig_arg)
            if new_arg.name != orig_arg.name:
                new_bsym = new_bsym.from_bsym_swap_proxies({variableify(orig_arg): new_arg})
            new_bsym = pre_post_process.maybe_modify_args_and_kwargs(new_bsym)
            # note(crcrpar): This header seems to be lost in the extrace.
            new_bsym.header = f"{pre_post_process.__class__.layer_type}"

        new_out = new_bsym.sym(*new_bsym.args, **new_bsym.kwargs)

        if pre_post_process is not None:
            from thunder.core import utils

            # This is because current support coverage are only `Linear` and `Embedding` that return one tensor.
            utils.check(
                len(new_bsym.flat_proxy_outs) == 1,
                lambda: f"{len(new_bsym.flat_proxy_outs)=} expected to be 1",
            )
            var_original_bsym_output = variableify(new_bsym.flat_proxy_outs[0])
            processed_y = pre_post_process.postprocess(new_out, preprocess_artifacts)
            self.swap_map[var_original_bsym_output] = processed_y
        else:
            from thunder.core.pytree import tree_flatten

            for orig_o, new_o in zip(
                new_bsym.flat_outs,
                tree_flatten(new_out)[0],
            ):
                if isinstance(orig_o, TensorProxy) and isinstance(new_o, TensorProxy) and orig_o.name != new_o.name:
                    self.swap_map[variableify(orig_o)] = new_o

        return VISIT_TYPE.REPLACE


@dataclass
class TransformForTensorParallel(Transform):
    rank: int
    world_size: int
    compile_data: CompileData
    process_group: ProcessGroup
    target_modules: Sequence[str]
    chunked_param_name_to_layer_type: dict[str, Any] = field(init=False, default_factory=dict)
    params_to_shard: dict[str, Any] = field(init=False, default_factory=dict)
    dim_to_shard: int = field(init=False, default=-1)

    def __post_init__(self):
        from thunder.common import CompileData
        from thunder.core import utils

        utils.check_type(self.compile_data, CompileData)
        if getattr(self.compile_data, "use_fsdp", False) or getattr(self.compile_data.fn, "use_fsdp", False):
            raise NotImplementedError("Currently thunder does not support the combination of fsdp and tensor parallel")
        self.device = torch.device("cuda", self.rank)

        if self.dim_to_shard == -1:
            raise ValueError(f"Set valid {self.dim_to_shard=} in inheritance")

    @abstractmethod
    def get_visitor_of_computation_trace_and_provenance(
        self,
        computation_trace: TraceCtx,
    ) -> tuple[ComputationTraceTransformVisitorForTensorParallel, TraceProvenance | str]: ...

    @abstractmethod
    def _calc_new_shape(self, orig_shape: list[int]) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def distparallel_type(self) -> DistParallelType: ...

    def transform_traces_pre_prologue(
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
        pro_out_p_dict: dict[VariableInterface, TensorProxy] = {}
        for pro_out_p, comp_inp_p in zip(prologue_trace.output[0], computation_trace.args):
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
                    pro_out_p_dict[variableify(pro_out_p)] = pro_out_p
                    if comp_inp_p is not pro_out_p:
                        comp_inp_p._shape = new_shape
                        comp_inp_p._distparallel_type = self.distparallel_type

        for bsym in prologue_trace.bound_symbols:
            if bsym.sym is prims.check_tensor_shape_and_metadata and prologue_producers[bsym.args[0]].sym in (
                prims.unpack_parameter,
                prims.unpack_buffer,
            ):
                param_thunder_module, _ = prologue_producers[bsym.args[0]].args
                assert param_thunder_module is thunder_module_proxy
                a0, shape, _, *a2pp = bsym.args
                if variableify(a0) in pro_out_p_dict:
                    new_shape = self._calc_new_shape(orig_shape=list(shape))
                    bsym.args = (a0, new_shape, a0.device.device_str(), *a2pp)

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
        if not visit.eligible_for_comm_optimization:
            return prologue_trace, new_computation_trace, epilogue_trace
        else:
            from thunder.distributed.tensor_parallel.optimize_comm import remove_redundant_comms

            return prologue_trace, remove_redundant_comms(new_computation_trace), epilogue_trace

    def transform_module(self, model: ThunderModule) -> None:
        import torch.nn as nn
        from thunder.core import utils
        from thunder.distributed import _shard_tensor

        for target_mod_name in self.target_modules:
            mod = model.get_submodule(target_mod_name)
            utils.check_type(
                mod,
                (
                    nn.Linear,
                    nn.Embedding,
                ),
            )
            for name, p in mod.named_parameters(prefix=target_mod_name, recurse=False):
                if p.ndim <= self.dim_to_shard:
                    continue
                self.chunked_param_name_to_layer_type["t_" + name.replace(".", "_")] = type(mod)
                self.params_to_shard[name] = None

        # Modify module
        for name, param in model.named_parameters():

            orig_param: torch.Tensor | None
            try:
                orig_param = model._model.get_parameter(name)
            except AttributeError:
                orig_param = None

            if param.is_meta:
                param._thunder_device = self.device
                if orig_param is not None:
                    orig_param._thunder_device = self.device
            elif param.device != self.device:
                with torch.no_grad():
                    new_param = torch.nn.Parameter(param.to(device=self.device), requires_grad=param.requires_grad)
                model._overrides_parameters[name] = new_param

            if name in self.params_to_shard:
                sharded_param, _ = _shard_tensor(
                    param,
                    self.rank,
                    self.world_size,
                    name,
                    allow_padding_for_fsdp=False,
                    dim=self.dim_to_shard,
                )
                sharded_param = torch.nn.Parameter(sharded_param.clone(), requires_grad=param.requires_grad)
                model._overrides_parameters[name] = sharded_param

    def transform_state_dict_for_submodule(
        self,
        model: ThunderModule,
        submodule_name: str,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        from thunder.distributed import _shard_tensor

        prefix = ""
        if submodule_name:
            prefix = f"{submodule_name}."
        new_state_dict = {}
        for k, v in state_dict.items():
            full_k = prefix + k
            if full_k in self.params_to_shard:
                v, _ = _shard_tensor(
                    v,
                    self.rank,
                    self.world_size,
                    full_k,
                    allow_padding_for_fsdp=False,
                    dim=self.dim_to_shard,
                )
            new_state_dict[k] = v
        return new_state_dict

    def reverse_transform_state_dict_for_submodule(
        self,
        model: ThunderModule,
        submodule_name: str,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        from thunder.executors.torchex import _all_gather_prim_impl

        new_state_dict = {}
        for name, tensor in state_dict.items():
            fqn: str
            if submodule_name:
                fqn = f"{submodule_name}.{name}"
            else:
                fqn = name

            if fqn not in self.params_to_shard:
                new_state_dict[name] = tensor
            else:
                all_gathered_param = _all_gather_prim_impl(
                    tensor,
                    group=self.process_group,
                    do_async=False,
                    dim=self.dim_to_shard,
                )
                new_state_dict[name] = all_gathered_param
        return new_state_dict
