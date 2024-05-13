from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed import distributed_c10d

from thunder.core import utils
from thunder.core.proxies import variableify

if TYPE_CHECKING:
    from collections.abc import Sequence
    from torch.distributed import ProcessGroup
    from thunder.core.trace import TraceCtx
    from thunder.core.proxies import TensorProxy
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
class VisitorTransformForColumnWiseOutput:
    chunked_parma_set: set[VariableInterface]
    process_group: ProcessGroup

    def __post_init__(self):
        self.swap_map: dict[VariableInterface, ProxyInterface] = {}

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        from thunder.core.transforms import VISIT_TYPE
        from thunder.distributed import prims as dist_prims

        new_bsym = bsym.from_bsym_swap_proxies(self.swap_map)
        if new_bsym == bsym and (
            requires_gather := len(self.chunked_param_set & {variableify(p) for p in bsym.flat_args}) == 0
        ):
            return VISIT_TYPE.NO_OP
        result = new_bsym()
        flat_result, _ = filter(lambda p: isinstance(p, ProxyInterface), result)
        utils.check(
            len(flat_result) == len(bsym.flat_proxy_outs),
            lambda: f"{len(flat_result)=} != {len(bsym.flat_proxy_outs)=}",
        )

        if requires_gather:
            # we need to all-gather output.
            for old_p, new_p in zip(bsym.flat_proxy_outs, flat_result):
                self.swap_map[variableify(old_p)] = new_p
            utils.check(
                len(flat_result) == 1,
                lambda: f"tensor parallel does not support bsym with multiple outputs. {len(bsym.flat_proxy_outs)=}",
            )
            out_to_gather = flat_result[0]
            utils.check_type(out_to_gather, TensorProxy)
            gathered = dist_prims.synchronize_output_for_column_wise_tensor_parallel(
                out_to_gather,
                group=self.process_group,
            )
            self.swap_map[variableify(out_to_gather)] = gathered
            return VISIT_TYPE.INSERT_AFTER
        else:
            for old_p, new_p in zip(bsym.flat_proxy_outs, flat_result):
                self.swap_map[variableify(old_p)] = new_p
            return VISIT_TYPE.REPLACE


@dataclass
class TransformForColumnWiseParallel:
    rank: int
    world_size: int
    compile_data: CompileData
    chunked_layers: Sequence[str]
    process_group: ProcessGroup

    def __post_init__(self):
        from thunder.common import CompileData

        utils.check_type(self.compile_data, CompileData)
        if getattr(self.compile_data, "use_fsdp", False) or getattr(self.compile_data.fn, "use_fsdp", False):
            raise NotImplementedError("Currently thunder does not support the combination of fsdp and tensor parallel")

        self.canonicalized_layer_names = {l_name.replace(".", "_") for l_name in self.chunked_layers}
        self.swap_map: dict[VariableInterface, ProxyInterface] = {}
        print(f"#### {self.canonicalized_layer_names=}")

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

        chunked_parma_proxy_set = utils.ProxyDict()
        chunked_parma_proxies: list[TensorProxy] = []
        flat_args, _ = tree_flatten((computation_trace.args, computation_trace.kwargs))
        for proxy in filter(lambda p: isinstance(p, TensorProxy), flat_args):
            for layer_name in self.canonicalized_layer_names:
                if layer_name in proxy.name:
                    chunked_parma_proxy_set[proxy] = None
                    chunked_parma_proxies.append(proxy)
        chunked_param_set: set[VariableInterface] = {variableify(p) for p in chunked_parma_proxies}

        new_computation_trace = visitor_transform(
            computation_trace,
            visit=VisitorTransformForColumnWiseOutput(
                chunked_parma_set=chunked_param_set,
                process_group=self.process_group,
            ),
            provenance="Insert comm for column wise tensor parallel",
        )

        if distributed_c10d.get_rank() == 0:
            print(f"#####\n{computation_trace}")
            print(f"#####\n{new_computation_trace}")

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

    .. note::

        This is alpha. This does not support any other distributed parallelisms such as
        :func:`~thunder.distributed.ddp` and :func:`~thunder.distributed.fsdp`.

    Args:
        thunder_module:

    Keyword Args:
        target_modules:
        process_group:
    """
    from thunder import compile_data as get_compile_data
    from thunder.distributed import _shard_param
    from thunder.core.transforms import add_transform
    from thunder.core.module import ThunderModule

    utils.check_type(thunder_module, ThunderModule)

    rank = distributed_c10d.get_rank(process_group)
    world_size = distributed_c10d.get_world_size(process_group)

    add_transform(
        thunder_module,
        early_transform=TransformForColumnWiseParallel(
            rank=rank,
            world_size=world_size,
            compile_data=get_compile_data(thunder_module),
            chunked_layers=target_modules,
            process_group=process_group,
        ),
    )

    for target_mod_name in target_modules:
        mod = thunder_module.get_submodule(target_mod_name)
        utils.check_type(mod, (nn.Linear,))
        for name, p in mod.named_parameters(recurse=False):
            _shard_param(p, rank, world_size, name, dim=0)

    return thunder_module
