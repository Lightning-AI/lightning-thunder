from typing import List, Dict, Optional, Tuple, Set
from collections.abc import Callable
import pprint
from functools import partial

import torch
from torch.fx.passes.split_module import split_module
import warnings
from collections.abc import Mapping

from thunder.core.baseutils import run_once

from thunder.dynamo.utils import (
    SubgraphInfo,
    CompiledFunction,
    CompilerType,
    SplitReason,
    SplitReasonType,
    is_node_supported,
    get_nodes_in_unsupported_ctx_regions,
    update_node_and_submodule,
)


@run_once
def _warn_thunder_compiler():
    warnings.warn(
        "The ThunderCompiler is in active development and may not work as expected."
        + " Please report any issues you encounter to the Lightning Thunder team."
    )


class ThunderCompiler:
    def __init__(self, *, thunder_options: dict | None = None, torch_inductor_options: dict | None = None):
        """
        A class that compiles a `fx.GraphModule` to a `thunder.ThunderModule`.
        This class is meant to be used as a backend for the `torch.compile`
        function.

        Keyword arguments:
            thunder_options: a dictionary of options to pass to `thunder.jit`.
            torch_inductor_options: a dictionary of options to pass to `torch.compile`.

        Example:
            >>> import torch
            >>> from thunder.dynamo import ThunderCompiler
            >>> backend = ThunderCompiler()
            >>> x = torch.ones(2, requires_grad=True)
            >>> @torch.compile(backend=backend)
            ... def func(x):
            ...     x = torch.sin(x)
            ...     if x.sum() > 0:
            ...         return x + 1
            ...     else:
            ...         return x - 1
            >>> out = func(x)
        """
        from thunder import ThunderModule, jit

        _warn_thunder_compiler()

        # Thunder-compiled functions should be readily available for inspection
        # and testing, so we will store them in a list[SubgraphInfo]. The order of the
        # functions in the list will be the same as the order in which they were
        # compiled.
        # Ref to the documentation of `SubgraphInfo` to know more about the information it contains.
        self.subgraph_infos: list[SubgraphInfo] = []

        if thunder_options is None:
            thunder_options = {}

        if torch_inductor_options is None:
            torch_inductor_options = {}

        self.thunder_options = thunder_options
        self._thunder_jit = partial(jit, **thunder_options)
        self._torch_compile = partial(torch.compile, **torch_inductor_options)

    def _splitter(
        self, gm: torch.fx.GraphModule, _unused_sample_args: list[torch.SymInt, torch.Tensor]
    ) -> torch.fx.GraphModule:
        """
        This method will split graph into multiple graph modules based on thunder supported operations.
        This function will try to split the graph in contiguous partitions.

        Example:
            # All operations are supported by thunder
            class GraphModule(torch.nn.Module):
                def forward(self, L_x_: "f32[2]"):
                    l_x_ = L_x_

                    y: "f32[2]" = torch.sin(l_x_)
                    matmul: "f32[]" = torch.matmul(l_x_, y);  l_x_ = y = None
                    return (matmul,)

            # Split Graph: All operations are supported by thunder, we will see only one partition.
            class GraphModule(torch.nn.Module):
                def forward(self, l_x_: "f32[2]"):
                    thunder_1 = self.thunder_1(l_x_);  l_x_ = None
                    return (thunder_1,)

                class thunder_1(torch.nn.Module):
                    def forward(self, l_x_: "f32[2]"):
                        y: "f32[2]" = torch.sin(l_x_)
                        matmul: "f32[]" = torch.matmul(l_x_, y);  l_x_ = y = None
                        return matmul

        Example:
            # With unsupported operation `sinc`
            class GraphModule(torch.nn.Module):
                def forward(self, L_x_: "f32[2]"):
                    l_x_ = L_x_

                    y: "f32[2]" = torch.sinc(l_x_)

                    matmul: "f32[]" = torch.matmul(l_x_, y);  l_x_ = y = None
                    return (matmul,)

            # Split Graph: Since `sinc` is unsupported, we will see two partitions, one for thunder and one for inductor.
            class GraphModule(torch.nn.Module):
                def forward(self, l_x_: "f32[2]"):
                    inductor_1 = self.inductor_1(l_x_)
                    thunder_2 = self.thunder_2(l_x_, inductor_1);  l_x_ = inductor_1 = None
                    return (thunder_2,)

                class inductor_1(torch.nn.Module):  # Partition for inductor
                    def forward(self, l_x_: "f32[2]"):
                        y: "f32[2]" = torch.sinc(l_x_);  l_x_ = None
                        return y

                class thunder_2(torch.nn.Module):  # Partition for thunder
                    def forward(self, l_x_: "f32[2]", y: "f32[2]"):
                        matmul: "f32[]" = torch.matmul(l_x_, y);  l_x_ = y = None
                        return matmul
        """
        # The callback below is called for every node in the graph.
        # It returns an `int` denoting the parition where the node should be placed.
        # We want to partition the graph into contiguous regions (with one or more operations)
        # into thunder supported or unsupported region.
        # `prev_value` is used to determine if we are still in same region (i.e. supported region or unsupported region).
        # `partition_cnt` is bumped everytime we change the region i.e. flip from supported to unsupported or from unsupported to supported.
        # `supported_partitions` is used to track the thunder supported partitions.
        prev_value = None
        partition_cnt = 0
        supported_partitions: set[int] = set()
        split_reasons: list[SplitReason] = []

        nodes_in_unsupported_ctx_regions = get_nodes_in_unsupported_ctx_regions(gm)

        def callback(node) -> int:
            nonlocal prev_value, partition_cnt, split_reasons, supported_partitions

            assert node.op not in (
                "placeholder",
                "get_attr",
                "output",
            ), f"fx.split_module should have only passed node.op=call_* but received {node.op}"

            if node in nodes_in_unsupported_ctx_regions:
                is_thunder_supported = False
                split_reason = SplitReason(
                    SplitReasonType.UNSUPPORTED_NODE,
                    info=f"node with name: {node.name} and target: {node.target} is not supported probably because it is in unsupported context.",
                )
                split_reasons.append(split_reason)
            else:
                is_thunder_supported, split_reason = is_node_supported(node)
                if split_reason is not None:
                    split_reasons.append(split_reason)

            if prev_value == is_thunder_supported:  # We are in the same region.
                return partition_cnt

            # There is a flip. Either from supported to unsupported or unsupported to supported.
            prev_value = is_thunder_supported
            partition_cnt += 1  # Bump the region cnt.

            if is_thunder_supported:
                supported_partitions.add(partition_cnt)
            return partition_cnt

        # `split_module` iterates over nodes and determines the partition to place them based on the callback.
        split_gm: torch.fx.GraphModule = split_module(
            gm, root_m=None, split_callback=callback, keep_original_order=True, keep_original_node_name=True
        )

        def is_thunder_supported_partition(node: torch.fx.Node) -> bool:
            return node.name.startswith("submod") and int(node.name.replace("submod_", "")) in supported_partitions

        # Call compile on the split region/s.
        thunder_compiled_fns = []
        submodule_to_compiled_fns = {}
        is_split = False
        for node in split_gm.graph.nodes:
            if is_thunder_supported_partition(node):
                # there is erase method on GraphModule
                graph_module = getattr(split_gm, node.name)
                jit_fn = self._thunder_jit(graph_module)
                update_node_and_submodule(split_gm, node, node.name.replace("submod", "thunder"), jit_fn)
                thunder_compiled_fns.append(jit_fn)
                submodule_to_compiled_fns[graph_module] = CompiledFunction(jit_fn, CompilerType.THUNDER)
            elif node.name.startswith("submod"):  # For inductor
                graph_module = getattr(split_gm, node.name)
                jit_fn = self._torch_compile(graph_module)
                update_node_and_submodule(split_gm, node, node.name.replace("submod", "inductor"), jit_fn)
                submodule_to_compiled_fns[graph_module] = CompiledFunction(jit_fn, CompilerType.TORCH_INDUCTOR)
                is_split = True
            else:
                # Everything else is a glue code to call and pass outputs between the other partitions.
                pass

        # We update the GraphModule in `update_node_and_submodule`, so we need to recompile.
        split_gm.recompile()

        # gm.print_readable()
        # Append the details regarding this graph/subgraph.
        self.subgraph_infos.append(
            SubgraphInfo(
                gm,
                split_gm,
                thunder_compiled_fns,
                submodule_to_compiled_fns,
                split_reasons,
            )
        )
        # split_gm.print_readable()
        return split_gm

    def __call__(self, gm: torch.fx.GraphModule, sample_args: list[torch.SymInt, torch.Tensor]):
        from thunder import jit

        # Dynamo uses lazy generation of the underlying Python code, so we need to
        # force recompilation of the GraphModule before passing it to Thunder.
        gm.real_recompile()

        # The whole graph may not be supported by `thunder`, so we split it in `thunder` supported sections
        # and unsupported sections which are passed to `torch.compile(backend='inductor')`
        split_module = self._splitter(gm, sample_args)
        return split_module
