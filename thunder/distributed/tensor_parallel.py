from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed import distributed_c10d

from thunder.core import utils
from thunder.core.proxies import variableify
from thunder.core.proxies import TensorProxy

if TYPE_CHECKING:
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


@dataclass
class TransformVisitor:
    process_group: ProcessGroup
    bsyms_before_allgather: set[BoundSymbol]

    def __post_init__(self):
        self.swap_map: dict[VariableInterface, ProxyInterface] = {}

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        from thunder.core.transforms import VISIT_TYPE
        from thunder.core.trace import get_tracectx
        from thunder.distributed import prims as dist_prims

        new_bsym = bsym.from_bsym_swap_proxies(self.swap_map, skip_output=True)
        trace = get_tracectx()
        trace.scopes[-1].append(new_bsym)

        if bsym in self.bsyms_before_allgather:
            output_to_gather = bsym.flat_proxy_outs[0]
            gathered_output = dist_prims.synchronize_output_for_column_wise_tensor_parallel(
                output_to_gather,
                self.process_group,
            )
            self.swap_map[variableify(output_to_gather)] = gathered_output

        return VISIT_TYPE.REPLACE


@dataclass
class TransformForColumnWiseParallel:
    rank: int
    world_size: int
    compile_data: CompileData
    chunked_param_names: set[str]
    process_group: ProcessGroup

    def __post_init__(self):
        from thunder.common import CompileData

        utils.check_type(self.compile_data, CompileData)
        if getattr(self.compile_data, "use_fsdp", False) or getattr(self.compile_data.fn, "use_fsdp", False):
            raise NotImplementedError("Currently thunder does not support the combination of fsdp and tensor parallel")

        self.swap_map: dict[VariableInterface, ProxyInterface] = {}

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
        bsyms_before_allgather: set[BoundSymbol] = set()
        for proxy in filter(lambda p: isinstance(p, TensorProxy), flat_args):
            for p_name in self.chunked_param_names:
                if p_name == proxy.name:
                    consumer_bsym = consumers[proxy][0]
                    bsyms_before_allgather.add(consumer_bsym)
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
    *,
    target_modules: Sequence[str],
    process_group: ProcessGroup,
) -> ThunderModule:
    """Convert specified modules into column-wise parallel ones.

    This method has two effects:
    1. Chunks target modules' parameters in 0-th dimension.
    2. Inserts communications before and after of the target modules' computation.

    Args:
        thunder_module:

    Keyword Args:
        target_modules: Names of modules to convert into column-wise.
        process_group:
    """
    from thunder import compile_data as get_compile_data
    from thunder.distributed import _shard_param
    from thunder.core.transforms import add_transform
    from thunder.core.module import ThunderModule

    utils.check_type(thunder_module, ThunderModule)

    rank = distributed_c10d.get_rank(process_group)
    world_size = distributed_c10d.get_world_size(process_group)

    chunked_param_names: list[str] = []
    for target_mod_name in target_modules:
        mod = thunder_module.get_submodule(target_mod_name)
        utils.check_type(mod, (nn.Linear,))
        for name, p in mod.named_parameters(recurse=False):
            chunked_param_names.append("t_" + f"{target_mod_name}.{name}".replace(".", "_"))

    param_proxy_name_set = sorted({name for name in chunked_param_names})
    colwise_thunder_module = add_transform(
        thunder_module,
        early_transform=TransformForColumnWiseParallel(
            rank=rank,
            world_size=world_size,
            compile_data=get_compile_data(thunder_module),
            chunked_param_names=param_proxy_name_set,
            process_group=process_group,
        ),
    )

    for target_mod_name in target_modules:
        mod = colwise_thunder_module.get_submodule(target_mod_name)
        for name, p in mod.named_parameters(recurse=False):
            _shard_param(p, rank, world_size, name, dim=0)

    return colwise_thunder_module
