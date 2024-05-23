from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import ClassVar

import torch.nn as nn
from torch.distributed import distributed_c10d

from thunder.core import utils
from thunder.distributed.tensor_parallel.common import PrePostProcessInterface
from thunder.distributed.tensor_parallel.common import TransformForTensorParallel
from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence
    from torch.distributed import ProcessGroup
    from thunder.core.module import ThunderModule
    from thunder.core.proxies import TensorProxy
    from thunder.core.trace import TraceCtx
    from thunder.core.trace import TraceProvenance


__all__ = [
    "convert_module_to_rowwise_parallel",
]


@dataclass(frozen=True)
class RowParallelLinearPrePostProcess(PrePostProcessInterface):
    process_group: ProcessGroup

    layer_type: ClassVar[TensorParallelLayerType] = TensorParallelLayerType.ROW_PARALLEL_LINEAR

    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        # split `x` in the last dim.
        import thunder.torch as ltorch

        chunked = ltorch.chunk(x, chunks=self.process_group.size(), dim=x.ndim - 1)
        local_chunk = chunked[distributed_c10d.get_rank(self.process_group)]
        return local_chunk, None

    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        # gather `y` along the last dimension
        from thunder.distributed import prims as dist_prims

        return dist_prims.synchronize_tensor_parallel_output(
            y,
            self.process_group,
            RowParallelLinearPrePostProcess.layer_type,
        )


@dataclass
class RowParallelEmbeddingPreProcess(PrePostProcessInterface):
    process_group: ProcessGroup

    layer_type: ClassVar[TensorParallelLayerType] = TensorParallelLayerType.ROW_PARALLEL_EMBED

    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        return super().preprocess(x)

    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        from thunder.distributed import prims as dist_prims

        return dist_prims.synchronize_tensor_parallel_output(
            y,
            self.process_group,
            RowParallelEmbeddingPreProcess.layer_type,
        )


@dataclass(frozen=True)
class TransformForRowWiseParallel(TransformForTensorParallel):

    def get_visitor_of_computation_trc_and_provenance(
        self,
        prologue_trace: TraceCtx,
        computation_trace: TraceCtx,
    ) -> tuple[Callable[[BoundSymbol], VISIT_TYPE], TraceProvenance | str]:
        from thunder.core import prims
        from thunder.core.pytree import tree_flatten

        modules_and_thunder_modules = [
            (bsym.args[0], bsym.output)
            for bsym in prologue_trace.bound_symbols
            if bsym.sym is prims.unpack_thunder_module
        ]

        if len(modules_and_thunder_modules) != 1:
            raise NotImplementedError("cannot deal with modules other than the compiled module")

        consumers = utils.consumers(computation_trace)
        flat_args, _ = tree_flatten((computation_trace.args, computation_trace.kwargs))
        bsym2prepostprocess: dict[BoundSymbol, PrePostProcessInterface] = {}
        for proxy in filter(lambda p: isinstance(p, TensorProxy), flat_args):
            for p_name in self.chunked_param_name2layer_type:
                if p_name == proxy.name:
                    consumer_bsym = consumers[proxy][0]
                    if consumer_bsym not in bsym2prepostprocess:
                        match self.chunked_param_name2layer_type[p_name]:
                            case nn.Linear:
                                bsym2prepostprocess[consumer_bsym] = ColumnParallelLinearPrePostProcess(
                                    process_group=self.process_group
                                )
                            case _:
                                utils.check(
                                    False,
                                    lambda: f"{self.chunked_param_name2layer_type[p_name]=} is not supported",
                                )
        utils.check(bsym2prepostprocess, lambda: f"{bsym2prepostprocess} must not be empty")

        visit = ComputationTraceTransformVisitorForTensorParallel(bsym2prepostprocess)
        return visit, "transform into column-wise tensor parallel"


def convert_module_to_rowwise_parallel(
    thunder_module: ThunderModule,
    target_modules: Sequence[str],
    process_group: ProcessGroup | None = None,
) -> ThunderModule:
    from thunder import compile_data as get_compile_data
    from thunder.distributed import _shard_param
    from thunder.core.transforms import add_transform
    from thunder.core.module import ThunderModule

    utils.check_type(thunder_module, ThunderModule)

    if process_group is None:
        process_group = distributed_c10d._get_default_group()
    rank = distributed_c10d.get_rank(process_group)
    world_size = distributed_c10d.get_world_size(process_group)

    chunked_param_name2layer_type: dict[str, Any] = {}
    for target_mod_name in target_modules:
        mod = thunder_module.get_submodule(target_mod_name)
        utils.check_type(
            mod,
            (nn.Linear, nn.Embedding),
        )
        for name, p in mod.named_parameters(recurse=False):
            chunked_param_name2layer_type["t_" + f"{target_mod_name}.{name}".replace(".", "_")] = type(mod)

    rowwise_thunder_module = add_transform(
        thunder_module,
        early_transform=TransformForRowWiseParallel(
            rank=rank,
            world_size=world_size,
            compile_data=get_compile_data(thunder_module),
            chunked_param_name2layer_type=chunked_param_name2layer_type,
            process_group=process_group,
        ),
    )

    for target_mod_name in target_modules:
        mod = rowwise_thunder_module.get_submodule(target_mod_name)
        for name, p in mod.named_parameters(recurse=False):
            _shard_param(p, rank, world_size, name, dim=1, allow_padding_for_fsdp=False)

    return rowwise_thunder_module
