from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed import distributed_c10d

from thunder.core import utils
from thunder.core.proxies import variableify
from thunder.core.proxies import TensorProxy

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence
    from torch.distributed import ProcessGroup
    from thunder.core.trace import TraceCtx
    from thunder.common import CompileData
    from thunder.core.symbol import VariableInterface
    from thunder.core.proxies import ProxyInterface
    from thunder.core.symbol import BoundSymbol
    from thunder.core.module import ThunderModule
    from thunder.core.transforms import VISIT_TYPE


__all__ = [
    "convert_module_to_columnwise_parallel",
]


class PrePostProcessInterface(ABC):
    @abstractmethod
    def preprocess(self, x: TensorProxy) -> TensorProxy:
        return x

    @abstractmethod
    def postprocess(self, y: TensorProxy) -> TensorProxy:
        return y


@dataclass(frozen=True)
class NoOp(PrePostProcessInterface):
    def preprocess(self, x: TensorProxy) -> TensorProxy:
        return super().preprocess(x)

    def postprocess(self, y: TensorProxy) -> TensorProxy:
        return super().postprocess(y)


@dataclass(frozen=True)
class LinearPrePostProcess(PrePostProcessInterface):
    process_group: ProcessGroup
    layer_type: str = field(default="linear", kw_only=True)

    def preprocess(self, x: TensorProxy) -> TensorProxy:
        return super().preprocess(x)

    def postprocess(self, y: TensorProxy) -> TensorProxy:
        from thunder.distributed import prims as dist_prims

        return dist_prims.synchronize_output_for_column_wise_tensor_parallel(
            y,
            self.process_group,
            self.layer_type,
        )


@dataclass
class EmbeddingPrePostProcess(PrePostProcessInterface):
    num_local_embeddings: int
    process_group: ProcessGroup

    layer_type: str = field(default="embedding", kw_only=True)

    def __post_init__(self) -> None:
        from torch.distributed import distributed_c10d

        rank = distributed_c10d.get_rank(self.process_group)

        self.vocab_start_index: int = rank * self.num_local_embeddings
        self.vocab_end_index: int = (rank + 1) * self.num_local_embeddings - 1

    def preprocess(self, x: TensorProxy) -> TensorProxy:
        import thunder.torch as ltorch

        masked_x = ltorch.where(x, x < self.vocab_start_index or x > self.vocab_end_index, -1)
        return masked_x

    def postprocess(self, y: TensorProxy) -> TensorProxy:
        from thunder.distributed import prims as dist_prims

        return dist_prims.synchronize_output_for_column_wise_tensor_parallel(
            y,
            self.process_group,
            self.layer_type,
        )


@dataclass
class TransformVisitor:
    process_group: ProcessGroup
    bsyms_before_allgather: dict[BoundSymbol, PrePostProcessInterface]

    def __post_init__(self):
        self.swap_map: dict[VariableInterface, ProxyInterface] = {}

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        from thunder.core.transforms import VISIT_TYPE
        from thunder.core.trace import get_tracectx
        from thunder.distributed import prims as dist_prims

        pre_post_process: PrePostProcessInterface | None = None
        if bsym in self.bsyms_before_allgather:
            pre_post_process = self.bsyms_before_allgather[bsym]
            orig_arg = bsym.flat_proxy_args[0]
            if (new_arg := pre_post_process.preprocess(orig_arg)).name != orig_arg.name:
                self.swap_map[variableify(orig_arg)] = new_arg

        new_bsym = bsym.from_bsym_swap_proxies(self.swap_map, skip_output=True)
        trace = get_tracectx()
        trace.scopes[-1].append(new_bsym)

        if bsym in self.bsyms_before_allgather:
            y = bsym.flat_proxy_outs[0]
            gathered_output = pre_post_process.postprocess(y)
            self.swap_map[variableify(y)] = gathered_output

        return VISIT_TYPE.REPLACE


@dataclass(frozen=True)
class TransformForColumnWiseParallel:
    rank: int
    world_size: int
    compile_data: CompileData
    chunked_param_name2layer_type: set[str]
    process_group: ProcessGroup

    def __post_init__(self):
        from thunder.common import CompileData

        utils.check_type(self.compile_data, CompileData)
        if getattr(self.compile_data, "use_fsdp", False) or getattr(self.compile_data.fn, "use_fsdp", False):
            raise NotImplementedError("Currently thunder does not support the combination of fsdp and tensor parallel")

    def __call__(
        self,
        prologue_trace: TraceCtx,
        computation_trace: TraceCtx,
        epilogue_trace: TraceCtx,
        **kwargs,
    ) -> tuple[TraceCtx, TraceCtx, TraceCtx]:
        from thunder.core import prims
        from thunder.core.pytree import tree_flatten
        from thunder.core.transforms import visitor_transform

        modules_and_thunder_modules = [
            (bsym.args[0], bsym.output)
            for bsym in prologue_trace.bound_symbols
            if bsym.sym is prims.unpack_thunder_module
        ]

        if len(modules_and_thunder_modules) != 1:
            raise NotImplementedError("cannot deal with modules other than the compiled module")

        consumers = utils.consumers(computation_trace)
        flat_args, _ = tree_flatten((computation_trace.args, computation_trace.kwargs))
        bsyms_before_allgather: dict[BoundSymbol, PrePostProcessInterface] = {}
        for proxy in filter(lambda p: isinstance(p, TensorProxy), flat_args):
            for p_name in self.chunked_param_name2layer_type:
                if p_name == proxy.name:
                    consumer_bsym = consumers[proxy][0]
                    if consumer_bsym not in bsyms_before_allgather:
                        match self.chunked_param_name2layer_type[p_name]:
                            case nn.Linear:
                                bsyms_before_allgather[consumer_bsym] = LinearPrePostProcess(
                                    process_group=self.process_group
                                )
                            case nn.Embedding:
                                bsyms_before_allgather[consumer_bsym] = EmbeddingPrePostProcess(
                                    num_local_embeddings=proxy.shape[0], process_group=self.process_group
                                )
                            case _:
                                utils.check(
                                    False,
                                    lambda: f"{self.chunked_param_name2layer_type[p_name]=} is not supported",
                                )
        utils.check(bsyms_before_allgather, lambda: f"{bsyms_before_allgather} must not be empty")

        visit = TransformVisitor(self.process_group, bsyms_before_allgather)
        new_computation_trace = visitor_transform(
            computation_trace,
            visit=visit,
            provenance="gather ouptut of column-wise tensor parallel layer",
        )
        if distributed_c10d.get_rank() == 0:
            print(visit.swap_map)

        return prologue_trace, new_computation_trace, epilogue_trace


# TODO(crcrpar): Support `nn.Embedding`
# TODO(crcrpar): Add an option to turn off output all-gather.
def convert_module_to_columnwise_parallel(
    thunder_module: ThunderModule,
    target_modules: Sequence[str],
    process_group: ProcessGroup | None = None,
) -> ThunderModule:
    """Convert specified modules into column-wise parallel ones.

    This method has two effects:
        1. Chunks target modules' parameters in 0-th dimension.
        2. Inserts all-gather in the last dimension after the target modules' computation.

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

    for target_mod_name in target_modules:
        mod = colwise_thunder_module.get_submodule(target_mod_name)
        for name, p in mod.named_parameters(recurse=False):
            _shard_param(p, rank, world_size, name, allow_padding_for_fsdp=False)

    return colwise_thunder_module
