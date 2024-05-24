from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import ClassVar

import torch.nn as nn
from torch.distributed import distributed_c10d

from thunder.core import utils
from thunder.core.proxies import TensorProxy
from thunder.distributed.tensor_parallel.common import PrePostProcessInterface
from thunder.distributed.tensor_parallel.common import ComputationTraceTransformVisitorForTensorParallel
from thunder.distributed.tensor_parallel.common import TransformForTensorParallel
from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable
    from collections.abc import Sequence
    from torch.distributed import ProcessGroup
    from thunder.core.trace import TraceCtx
    from thunder.core.trace import TraceProvenance
    from thunder.core.transforms import VISIT_TYPE
    from thunder.core.symbol import BoundSymbol
    from thunder.core.module import ThunderModule


__all__ = [
    "convert_module_to_columnwise_parallel",
]


@dataclass(frozen=True)
class ColumnParallelLinearPrePostProcess(PrePostProcessInterface):
    process_group: ProcessGroup

    layer_type: ClassVar[TensorParallelLayerType] = TensorParallelLayerType.COLUMN_PARALLEL_LINEAR

    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        return super().preprocess(x)

    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        from thunder.distributed import prims as dist_prims

        return dist_prims.synchronize_tensor_parallel_output(
            y,
            self.process_group,
            ColumnParallelLinearPrePostProcess.layer_type,
        )


@dataclass
class ColumnParallelEmbeddingPrePostProcess(PrePostProcessInterface):
    num_local_embeddings: int
    process_group: ProcessGroup

    layer_type: ClassVar[TensorParallelLayerType] = TensorParallelLayerType.COLUMN_PARALLEL_EMBED

    def __post_init__(self) -> None:
        from torch.distributed import distributed_c10d

        rank = distributed_c10d.get_rank(self.process_group)

        self.vocab_start_index: int = rank * self.num_local_embeddings
        self.vocab_end_index: int = (rank + 1) * self.num_local_embeddings - 1

    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[TensorProxy, TensorProxy]]:
        import thunder.torch as ltorch

        x = ltorch.sub(x, self.vocab_start_index)
        mask1 = ltorch.ge(x, self.num_local_embeddings)
        masked1 = ltorch.masked_fill(x, mask1, 0)
        mask2 = ltorch.le(x, -1)
        masked2 = ltorch.masked_fill(masked1, mask2, 0)
        return masked2, (mask1, mask2)

    def postprocess(self, y: TensorProxy, masks: Any) -> TensorProxy:
        from thunder.distributed import prims as dist_prims
        import thunder.torch as ltorch

        utils.check(len(masks) == 2, lambda: f"Expected 2 masks but {len(masks)=}")
        for mask in masks:
            utils.check(
                mask.shape == y.shape[: mask.ndim],
                lambda: f"{mask.shape = }, {y.shape = }",
            )
            mask = ltorch.unsqueeze(mask, mask.ndim)
            unflattened_mask = ltorch.repeat(mask, (1, 1, y.shape[-1]))
            utils.check(
                unflattened_mask.shape == y.shape,
                lambda: f"{unflattened_mask.shape = }, {y.shape = }",
            )
            y = ltorch.masked_fill(y, unflattened_mask, 0.0)
        return dist_prims.synchronize_tensor_parallel_output(
            y,
            self.process_group,
            ColumnParallelEmbeddingPrePostProcess.layer_type,
        )


@dataclass
class TransformForColumnWiseParallel(TransformForTensorParallel):

    def _calc_new_shape(self, orig_shape: list[int]) -> tuple[int, ...]:
        new_shape = orig_shape[:]
        new_shape[0] //= self.process_group.size()
        return tuple(new_shape)

    def get_visitor_of_computation_trc_and_provenance(
        self,
        computation_trace: TraceCtx,
    ) -> tuple[Callable[[BoundSymbol], VISIT_TYPE], TraceProvenance | str]:
        from thunder.core.pytree import tree_flatten

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
                            case nn.Embedding:
                                bsym2prepostprocess[consumer_bsym] = ColumnParallelEmbeddingPrePostProcess(
                                    num_local_embeddings=proxy.shape[0], process_group=self.process_group
                                )
                            case _:
                                utils.check(
                                    False,
                                    lambda: f"{self.chunked_param_name2layer_type[p_name]=} is not supported",
                                )
        utils.check(bsym2prepostprocess, lambda: f"{bsym2prepostprocess} must not be empty")

        visit = ComputationTraceTransformVisitorForTensorParallel(bsym2prepostprocess)
        return visit, "transform into column-wise tensor parallel"


# TODO(crcrpar): Add an option to turn off output all-gather.
def convert_module_to_columnwise_parallel(
    thunder_module: ThunderModule,
    target_modules: Sequence[str],
    process_group: ProcessGroup | None = None,
) -> ThunderModule:
    """Convert specified modules into column-wise parallel ones.

    This method has two effects:
        1. Chunks target modules' parameters in 0-th dimension.
        2. Insert preprocess and postprocess around modified module ops.

    Args:
        thunder_module:

    Keyword Args:
        target_modules: Names of modules to convert into column-wise.
        process_group:


    Example:
        .. code-block:: python

            import os

            import torch
            import torch.nn
            import torch.nn.functional as F
            from torch.distributed import distributed_c10d

            import thunder
            from thunder.distributed import convert_module_to_columnwise_parallel


            class Model(nn.Module):
                def __init__(
                    self,
                    num_embeddings: int,
                    embedding_dim: int,
                    n_hidden: int,
                    n_out: int,
                ) -> None:
                    super().__init__()
                    self.embed = nn.Embedding(num_embeddings, embedding_dim)
                    self.l1 = nn.Linear(embedding_dim, n_hidden)
                    self.l2 = nn.Linear(n_hidden, n_out)

                def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                    feature = self.embed(tokens)
                    h = F.gelu(self.l1(feature), approximate='tanh')
                    return self.l2(h)

            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            distributed_c10d.init_process_group()
            model = Model().to(device)
            jitted_model = thunder.jit(model)
            # `l2`'s output size (= n_out) needs to be divisible by `world_size`
            tp_model = convert_module_to_columnwise_parallel(
                jitted_model,
                target_modules=("embed", "l2",),
            )

            x = torch.randn(4, n_in, device=device)
            out = tp_model(x)  # shape: [4, n_out]
    """
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
            (
                nn.Linear,
                nn.Embedding,
            ),
        )
        for name, p in mod.named_parameters(recurse=False):
            chunked_param_name2layer_type["t_" + f"{target_mod_name}.{name}".replace(".", "_")] = type(mod)

    import copy

    # Modify module
    for module_name, _ in thunder_module._model.named_modules():
        if module_name not in target_modules:
            continue
        submodule = thunder_module.get_submodule(module_name)

        # we use a copy to let the user's module alone (TODO: does this fully work?)
        module_copy = copy.copy(submodule)
        for pn, p in submodule.named_parameters(recurse=False, prefix=module_name):
            # if we don't have an override or it is just the original, do create a copy
            if thunder_module._overrides.get(pn, p) is p:
                thunder_module._overrides[pn] = copy.copy(p)
            _shard_param(thunder_module._overrides[pn], rank, world_size, pn, allow_padding_for_fsdp=True)

    colwise_thunder_module = add_transform(
        thunder_module,
        early_transform=TransformForColumnWiseParallel(
            rank=rank,
            world_size=world_size,
            compile_data=get_compile_data(thunder_module),
            chunked_param_name2layer_type=chunked_param_name2layer_type,
            process_group=process_group,
        ),
    )

    return colwise_thunder_module
