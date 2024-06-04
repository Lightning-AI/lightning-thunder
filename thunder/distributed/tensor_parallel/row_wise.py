from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import ClassVar

import torch.nn as nn
from torch.distributed import distributed_c10d

from thunder.core import utils
from thunder.core.proxies import DistParallelType
from thunder.core.proxies import TensorProxy
from thunder.core.proxies import variableify
from thunder.distributed.tensor_parallel.common import PrePostProcessInterface
from thunder.distributed.tensor_parallel.common import ComputationTraceTransformVisitorForTensorParallel
from thunder.distributed.tensor_parallel.common import TransformForTensorParallel
from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence
    from collections.abc import Callable
    from torch.distributed import ProcessGroup
    from thunder.core.module import ThunderModule
    from thunder.core.symbol import BoundSymbol
    from thunder.core.trace import TraceCtx
    from thunder.core.trace import TraceProvenance
    from thunder.core.transforms import VISIT_TYPE


__all__ = [
    "row_parallel",
]


@dataclass(frozen=True)
class RowParallelLinearPrePostProcess(PrePostProcessInterface):
    process_group: ProcessGroup
    bias_or_none: TensorProxy | None

    layer_type: ClassVar[TensorParallelLayerType] = TensorParallelLayerType.ROW_PARALLEL_LINEAR

    def preprocess(self, x: TensorProxy) -> tuple[TensorProxy, tuple[Any, ...]]:
        from thunder.distributed import prims as dist_prims

        # split `x` in the last dim.
        return (
            dist_prims.synchronize_tensor_parallel_input(
                x, self.process_group, RowParallelLinearPrePostProcess.layer_type
            ),
            None,
        )

    def postprocess(self, y: TensorProxy, _: Any) -> TensorProxy:
        # gather `y` along the last dimension
        import thunder.torch as ltorch
        from thunder.distributed import prims as dist_prims

        all_reduced = dist_prims.synchronize_tensor_parallel_output(
            y,
            self.process_group,
            RowParallelLinearPrePostProcess.layer_type,
        )
        if (bias := self.bias_or_none) is not None:
            return ltorch.add(all_reduced, bias)
        else:
            return all_reduced

    def maybe_modify_args_and_kwargs(self, bsym: BoundSymbol) -> BoundSymbol:
        """Replace `bias` of `bsym` with `None` if it's Tensor to avoid redundant accumulation.

        The removed `bias` is added by `postprocess` after all-reduce.
        The local row-wise parallel linear operation with bias could be
            y_<rank> = linear(x_<rank>, weight_<rank>, bias)
        which leads to a wrong result after all_reduce as `bias` is added <world_size> times.
            y = all_reduce(y_<rank>)
        """
        if self.bias_or_none is not None:
            return bsym.from_bsym_swap_proxies({variableify(self.bias_or_none): None}, skip_output=True)
        else:
            return super().maybe_modify_args_and_kwargs(bsym)


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


@dataclass
class TransformForRowWiseParallel(TransformForTensorParallel):

    @property
    def distparallel_type(self) -> DistParallelType:
        return DistParallelType.ROW_WISE

    def _calc_new_shape(self, orig_shape: list[int]) -> tuple[int, ...]:
        new_shape = orig_shape[:]
        new_shape[1] //= self.process_group.size()
        return tuple(new_shape)

    def get_visitor_of_computation_trace_and_provenance(
        self,
        computation_trace: TraceCtx,
    ) -> tuple[Callable[[BoundSymbol], VISIT_TYPE], TraceProvenance | str]:
        from thunder.core.pytree import tree_flatten

        consumers = utils.consumers(computation_trace)
        flat_args, _ = tree_flatten((computation_trace.args, computation_trace.kwargs))
        bsym_to_prepostprocess: dict[BoundSymbol, PrePostProcessInterface] = {}
        for proxy in filter(lambda p: isinstance(p, TensorProxy), flat_args):
            if (layer_type := self.chunked_param_name_to_layer_type.get(proxy.name, None)) is not None:
                consumer_bsym = consumers[proxy][0]
                if consumer_bsym not in bsym_to_prepostprocess:
                    match layer_type:
                        case nn.Linear:
                            orig_args = consumer_bsym.args
                            utils.check(
                                len(orig_args) == 3,
                                lambda: f"{consumer_bsym.sym.id} expected to have 3 args but {orig_args}",
                            )
                            bias_or_none = orig_args[2]
                            utils.check(
                                isinstance(bias_or_none, TensorProxy) or bias_or_none is None,
                                lambda: f"{orig_args[-1]} expected to be either `None` or `TensorProxy`",
                            )
                            bsym_to_prepostprocess[consumer_bsym] = RowParallelLinearPrePostProcess(
                                process_group=self.process_group,
                                bias_or_none=bias_or_none,
                            )
                        case nn.Embedding:
                            bsym_to_prepostprocess[consumer_bsym] = RowParallelEmbeddingPreProcess(
                                process_group=self.process_group
                            )
                        case _:
                            utils.check(
                                False,
                                lambda: f"{self.chunked_param_name_to_layer_type[proxy.name]=} is not supported",
                            )
        utils.check(bsym_to_prepostprocess, lambda: f"{bsym_to_prepostprocess} must not be empty")

        visit = ComputationTraceTransformVisitorForTensorParallel(bsym_to_prepostprocess)
        return visit, "transform into row-wise tensor parallel"


def row_parallel(
    thunder_module: ThunderModule,
    target_modules: Sequence[str],
    process_group: ProcessGroup | None = None,
) -> ThunderModule:
    """Convert specified modules into row-wise parallel ones.

    This method has two effects:
        1. Chunks target modules' parameters in 1st dimension.
        2. Insert preprocess and postprocess around modified module ops.

    Args:
        thunder_module:
        target_modules: Names of modules to convert into row-wise.
        process_group:

    Example:
        .. code-block:: python

            import os

            import torch
            import torch.nn
            import torch.nn.functional as F
            from torch.distributed import distributed_c10d

            import thunder
            from thunder.distributed import row_parallel


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
            # ``embedding_dim`` and `l2`'s input size (= n_hidden) need to be divisible by `world_size`
            tp_model = column_parallel(
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

    chunked_param_name_to_layer_type: dict[str, Any] = {}
    for target_mod_name in target_modules:
        mod = thunder_module.get_submodule(target_mod_name)
        utils.check_type(
            mod,
            (nn.Linear, nn.Embedding),
        )
        for name, p in mod.named_parameters(recurse=False):
            if p.ndim < 2:
                continue
            chunked_param_name_to_layer_type["t_" + f"{target_mod_name}.{name}".replace(".", "_")] = type(mod)

    import copy

    # Modify module
    for module_name, _ in thunder_module._model.named_modules():
        if module_name not in target_modules:
            continue
        submodule = thunder_module.get_submodule(module_name)

        for pn, p in submodule.named_parameters(recurse=False, prefix=module_name):
            # if we don't have an override or it is just the original, do create a copy
            if thunder_module._overrides_parameters.get(pn, p) is p:
                thunder_module._overrides_parameters[pn] = copy.copy(p)
            if p.ndim < 2:
                continue
            _shard_param(
                thunder_module._overrides_parameters[pn], rank, world_size, pn, dim=1, allow_padding_for_fsdp=False
            )

    rowwise_thunder_module = add_transform(
        thunder_module,
        early_transform=TransformForRowWiseParallel(
            rank=rank,
            world_size=world_size,
            compile_data=get_compile_data(thunder_module),
            chunked_param_name_to_layer_type=chunked_param_name_to_layer_type,
            process_group=process_group,
        ),
    )

    return rowwise_thunder_module
