from __future__ import annotations
from typing import TYPE_CHECKING
import copy
from functools import partial

import torch
from torch.fx.passes.split_module import split_module

from thunder.dynamo.utils import (
    SubgraphInfo,
    CompiledFunction,
    CompilerType,
    SplitReason,
    SplitReasonType,
    is_node_supported_by_thunder,
    get_nodes_in_unsupported_ctx_regions,
    update_node_and_submodule,
    recompile_graph,
    checkpoint_converter,
    _get_example_inputs_from_placeholder,
    _ThunderSplitGraphModule,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _splitter(
    gm: torch.fx.GraphModule,
    thunder_jit: Callable,
    torch_inductor: Callable,
    _unused_sample_args: list[torch.SymInt, torch.Tensor],
) -> tuple[torch.fx.GraphModule, SubgraphInfo]:
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
            # If node was in unsupported ctx region like `autocast`,
            # even though the operation maybe supported, we pass it to `torch.compile`
            # as `thunder` doesn't correctly work with these.
            is_thunder_supported = False
            split_reason = SplitReason(
                SplitReasonType.UNSUPPORTED_NODE,
                info=f"node with name: {node.name} and target: {node.target} is not supported probably because it is in unsupported context.",
            )
            split_reasons.append(split_reason)
        else:
            is_thunder_supported, split_reason = is_node_supported_by_thunder(node)
            if split_reason is not None:
                split_reasons.append(split_reason)

        if prev_value == is_thunder_supported:  # We are in the same region.
            return partition_cnt

        # There is a flip. Either from supported to unsupported or unsupported to supported.
        if prev_value is not None:
            partition_cnt += 1  # Bump the region cnt.
        prev_value = is_thunder_supported

        if is_thunder_supported:
            supported_partitions.add(partition_cnt)
        return partition_cnt

    # Removes the unused torch.autograd.function.FunctionCtx
    functionctx_nodes_to_del = (
        n for n in gm.graph.find_nodes(op="call_function", target=torch.autograd.function.FunctionCtx) if not n.users
    )
    for n in functionctx_nodes_to_del:
        gm.graph.erase_node(n)
    gm.recompile()

    # `split_module` iterates over nodes and determines the partition to place them based on the callback.
    original_split_gm: torch.fx.GraphModule = split_module(
        gm, root_m=None, split_callback=callback, keep_original_order=True, keep_original_node_name=True
    )

    # Workaround for the Torch bug https://github.com/pytorch/pytorch/pull/139275
    for submodule in original_split_gm.children():
        if not submodule.graph.find_nodes(op="output"):
            submodule.graph.output(())
    if not original_split_gm.graph.find_nodes(op="output"):
        original_split_gm.graph.output(())
    split_gm = copy.deepcopy(original_split_gm)

    def is_thunder_supported_partition(node: torch.fx.Node) -> bool:
        return node.name.startswith("submod") and int(node.name.replace("submod_", "")) in supported_partitions

    # Call compile on the split region/s.
    thunder_compiled_fns = []
    example_input_metadatas = []
    submodule_to_compiled_fns = {}
    for node in split_gm.graph.nodes:
        node_name = node.name
        if is_thunder_supported_partition(node):
            graph_module = getattr(split_gm, node.name)

            is_differentiable_outputs = []
            for n in graph_module.graph.nodes:
                if n.op == "output":
                    for n in n.all_input_nodes:
                        if "example_value" not in n.meta or n.meta["example_value"].grad_fn is None:
                            is_differentiable_outputs.append(False)
                        else:
                            is_differentiable_outputs.append(True)

            # Record the input tensor metadata of the current module based on the faketensor 'example_value' of the placeholder node
            placeholders = list(n for n in graph_module.graph.nodes if n.op == "placeholder")
            example_input_metadata = map(
                partial(_get_example_inputs_from_placeholder, only_metadata=True), placeholders
            )
            example_input_metadatas.append(list(example_input_metadata))
            # Replace PyTorch operators within the checkpointed function with the corresponding Thunder operators
            checkpoint_converter(split_gm, graph_module)

            jit_fn = thunder_jit(graph_module, is_differentiable_outputs=is_differentiable_outputs)
            # Update the node name from "submod_*" to "thunder_*" for more user-friendly names
            update_node_and_submodule(split_gm, node, node.name.replace("submod", "thunder"), jit_fn)
            thunder_compiled_fns.append(jit_fn)
            submodule_to_compiled_fns[getattr(original_split_gm, node_name)] = CompiledFunction(
                jit_fn, CompilerType.THUNDER
            )
        elif node.name.startswith("submod"):  # For inductor
            graph_module = getattr(split_gm, node.name)
            jit_fn = torch_inductor(graph_module)
            # Update the node name from "submod_*" to "inductor_*" for more user-friendly names
            update_node_and_submodule(split_gm, node, node.name.replace("submod", "inductor"), jit_fn)
            submodule_to_compiled_fns[getattr(original_split_gm, node_name)] = CompiledFunction(
                jit_fn, CompilerType.TORCH_INDUCTOR
            )
        else:
            # Everything else is a glue code to call and pass outputs between the other partitions.
            pass

    # We update the GraphModule in `update_node_and_submodule`, so we need to recompile.
    recompile_graph(split_gm)

    return split_gm, SubgraphInfo(
        gm,
        _ThunderSplitGraphModule(original_split_gm, supported_partitions),
        split_gm,
        thunder_compiled_fns,
        example_input_metadatas,
        submodule_to_compiled_fns,
        split_reasons,
    )
