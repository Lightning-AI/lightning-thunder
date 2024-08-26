from enum import Enum, auto
import dataclasses
from typing import List, Dict, Optional, Tuple
from collections.abc import Callable
import pprint
import itertools
import copy
from functools import partial
import operator
import inspect

import torch
from torch.fx.passes.split_module import split_module as fx_split_module
import warnings
from collections.abc import Mapping
from torch.fx.passes import operator_support

from thunder.core.baseutils import run_once
from thunder.torch.default_torch_ops import torch_auto_registered_ops
from thunder.torch import _torch_to_thunder_function_map

auto_register_ops = set(itertools.chain(*torch_auto_registered_ops.values()))


@run_once
def _warn_thunder_compiler():
    warnings.warn(
        "The ThunderCompiler is in active development and may not work as expected."
        + " Please report any issues you encounter to the Lightning Thunder team."
    )


class CompilerType(Enum):
    """
    An enumeration representing different types of compilers.
    """

    THUNDER = auto()
    TORCH_INDUCTOR = auto()


@dataclasses.dataclass
class CompiledFunction:
    """
    A dataclass representing a compiled function along with its original graph module and compiler type.

    Attributes:
        original_graph_module (torch.fx.GraphModule): The original graph module from which the function is compiled.
        compiled_fn (Callable): The compiled function.
        compiler (CompilerType): The type of compiler used to compile the function.
    """

    original_graph_module: torch.fx.GraphModule
    compiled_fn: Callable
    compiler: CompilerType


class SplitReasonType(Enum):
    """
    An enumeration representing different reasons for split in the graph.
    """

    UNSUPPORTED_NODE = auto()
    MISSING_OP_SUPPORT = auto()
    EXCEPTION_PROXY_THUNDER_OP = auto()
    EXCEPTION_META_THUNDER_OP = auto()


@dataclasses.dataclass
class SplitReason:
    """
    A dataclass containing information about a split.

    Attributes:
        type (SplitReasonType): Reason for the split.
        info (str): String with details of what caused the split.
        exception (Exception | None): Exception if there was any.
    """

    type: SplitReasonType
    info: str | None
    exception: Exception | None = None


@dataclasses.dataclass
class SubgraphInfo:
    """
    A dataclass containing information about a subgraph.

    Attributes:
        original_graph_module (torch.fx.GraphModule): The original graph module.
        compiled_functions (list[CompiledFunction]): A list of compiled functions derived from the subgraph. This will be a list with one function in case the graph was not split.
        is_split (bool): Indicates whether the subgraph has been split. This happens if there was a thunder unsupported functionality.
        split_reasons (list[SplitReason] | None): Optional list of reasons explaining why the subgraph was split. Present only if `is_split` is True.
        split_graph_module (torch.fx.GraphModule | None): Optional. The graph module for the split subgraph. Present only if `is_split` is True.
    """

    original_graph_module: torch.fx.GraphModule
    compiled_functions: list[CompiledFunction]
    is_split: bool
    split_reasons: list | None = None
    split_graph_module: torch.fx.GraphModule | None = None


def try_execute_symbol(thunder_symbol: "Symbol", node: torch.fx.Node) -> tuple[bool, SplitReason | None]:
    """
    Attempts to execute a given Thunder symbol within a tracing context, using proxies for the node's arguments.

    This function operates within a Thunder tracing context to generate proxies for the provided node's arguments.
    It then attempts to execute the Thunder symbol with these proxies. If any exceptions occur during proxy creation
    or execution, it returns a tuple indicating failure and provides a `SplitReason` detailing the exception.

    Args:
        thunder_symbol (Symbol): The Thunder symbol to be executed. This is expected to be a callable that can
            operate on proxy arguments.
        node (torch.fx.Node): The Torch FX node whose arguments are to be proxied and passed to the Thunder symbol.

    Returns:
        tuple[bool, SplitReason | None]: A tuple where the first element is a boolean whether the execution passed or failed.
            The second element is a `SplitReason` object if an error occurred, or `None` if the execution was successful.
    """
    import thunder
    from thunder.core.trace import TraceCtx
    from thunder.core.proxies import proxy

    trc = TraceCtx()
    # We need to be under trace context to generate proxies.
    with thunder.core.trace.tracectx(trc):
        try:

            def make_tensor_proxy(arg_node):
                # This is a Node in the graph representing a Tensor.
                if isinstance(arg_node, torch.fx.Node):
                    example_value = arg_node.meta["example_value"]

                    # This fails if the shape of the FakeTensor contains SymInts.
                    return proxy(example_value)

                # This is int, float, etc.
                # TODO(kshitij12345) - verify the above line for more cases.
                return arg_node

            proxy_args = tuple(map(make_tensor_proxy, node.args))
            proxy_kwargs = {k: make_tensor_proxy(v) for k, v in node.kwargs.items()}
        except Exception as e:
            return False, SplitReason(
                SplitReasonType.EXCEPTION_PROXY_THUNDER_OP,
                f"Failed while creating proxy for node with name: {node.name} and target: {node.target}, see exception field",
                exception=e,
            )

        try:
            thunder_symbol(*proxy_args, **proxy_kwargs)
        except Exception as e:
            return False, SplitReason(
                SplitReasonType.EXCEPTION_META_THUNDER_OP,
                f"Failed while running meta for node with name: {node.name} and target: {node.target}, see exception field",
                exception=e,
            )

    # Execution with proxies was successful.
    return True, None


class ThunderOperatorSupport:
    def __init__(self, gm):
        self.gm = gm
        self.unsupported_nodes = set()
        self.find_unsupported_ctx_regions(gm)
        self.split_reasons: list[SplitReason] = []

    def find_unsupported_ctx_regions(self, gm):
        """
        Finds the node within `autocast` or other supported context and marks them as unsupported.
        Even though, thunder may support the operation within the reason, it doesn't correctly apply the change
        triggered from the context.
        """
        # NOTE - Currently only detects the autocast regions.

        ctx_cnt = 0  # Count of `enters_autocast` we have seen till now

        # We want to mark nodes with `_enter_autocast` and `_exit_autocast`
        # as unsupported as `thunder` doesn't correctly deal with these stateful functions.
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in (torch.amp.autocast_mode._enter_autocast,):
                ctx_cnt += 1
            elif node.op == "call_function" and node.target in (torch.amp.autocast_mode._exit_autocast,):
                ctx_cnt -= 1
            else:
                if ctx_cnt > 0:
                    self.unsupported_nodes.add(node)

    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node):
        """
        Determine whether thunder can execute the operation described by this node.
        """
        # These are the nodes which are in unsupported context regions
        if node in self.unsupported_nodes:
            self.split_reasons.append(
                SplitReason(
                    SplitReasonType.UNSUPPORTED_NODE,
                    info=f"node with name: {node.name} and target: {node.target} is not supported probably because it is in unsupported context.",
                )
            )
            return False

        # Docs from the torch.fx.Node - https://pytorch.org/docs/stable/fx.html#torch.fx.Node
        # Each Node has a function specified by its op property
        # Below are the details for the ones this function is interested in -
        # `call_function` applies a free function to some values.
        #       name is similarly the name of the value to assign to.
        #       target is the function to be applied. args and kwargs represent
        #       the arguments to the function, following the Python calling convention
        # `call_method` calls a method on a value.
        #       name is as similar.
        #       target is the string name of the method to apply to the self argument.
        #       args and kwargs represent the arguments to invoke the module on, including the self argument
        #
        # NOTE: `call_module` should be inlined in dynamo graphs since https://github.com/pytorch/pytorch/pull/131275
        # But there is flag to disable inlining `call_module`. Determining `call_module` support would actually require calling `thunder.jit` on it.
        #
        # `call_module` applies a module in the module hierarchy’s forward() method to given arguments.
        #       name is as previous. target is the fully-qualified name of the module in the module hierarchy to call.
        #       args and kwargs represent the arguments to invoke the module on, excluding the self argument

        target = node.target  # Target is the function to call.
        if node.op == "call_method":
            self_arg = node.args[0]
            target = getattr(torch.Tensor, node.target, None)
            assert target is not None, f"Failed to find method {node.target}"

        # If the operation has automatic registration, we mark it as unsupported as `inductor` might be
        # able to deal with it better.
        if target in auto_register_ops:
            self.split_reasons.append(
                SplitReason(
                    SplitReasonType.MISSING_OP_SUPPORT,
                    info=f"node with name: {node.name} and target: {node.target} only has an automatic torch fallback in thunder.",
                )
            )
            return False

        # If thunder has a mapping for this operation, try executing the meta function and see.
        # We have a symbol for `torch.where`, but we don't support one overload of it.
        # So, we try and execute the meta to get a real signal.
        #
        # Regarding `inspect.isbuiltin`, dynamo graph uses `+`, `>` which are builtin `add`, `gt`.
        # We try to proxify the arguments and call these operations on them to see if they are supported.
        if target in _torch_to_thunder_function_map or inspect.isbuiltin(target):
            if target in [torch.ones]:  # Factory functions. (removing this will lead to split)
                # NOTE - Factory functions don't work as the expect `_cache_info` to be populated
                #        with default dtype but `_cache_info` is only created and populated in `thunder.jit` path.
                # my_trc = TraceCtx()
                # with tracectx(my_trc):
                #     thunder.torch.ones(3, 3)
                return True

            thunder_symbol_or_builtin = _torch_to_thunder_function_map.get(target, target)
            did_run, opt_split_reason = try_execute_symbol(thunder_symbol_or_builtin, node)
            if opt_split_reason is not None:
                self.split_reasons.append(opt_split_reason)
            return did_run

        # We found no automatic fallback registration and no mapping to thunder symbol.
        self.split_reasons.append(
            SplitReason(
                SplitReasonType.MISSING_OP_SUPPORT,
                info=f"node with name: {node.name} and target: {node.target} didn't have any mapping in thunder.",
            )
        )
        return False


def _all_graph_supported_by_thunder(gm: torch.fx.GraphModule, sample_input: list[torch.SymInt, torch.Tensor]) -> bool:
    """
    Determine whether there is any thunder unsupported operation.
    """
    # NOTE - Unused for now.
    op_support = ThunderOperatorSupport(gm)
    supported = True
    for node in gm.graph.nodes:
        if node.op in ["call_method", "call_function"]:
            supported = op_support.is_node_supported(gm, node)
            if not supported:
                break
    return supported


class ThunderCompiler:
    def __init__(self, **thunder_options):
        """
        A class that compiles a `fx.GraphModule` to a `thunder.ThunderModule`.
        This class is meant to be used as a backend for the `torch.compile`
        function.

        Keyword arguments:
            thunder_options: a dictionary of options to pass to `thunder.jit`.

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

        self.thunder_options = thunder_options
        self._thunder_jit = partial(jit, **thunder_options)

    def splitter(
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
                    submod_1 = self.submod_1(l_x_);  l_x_ = None
                    return (submod_1,)

                class submod_1(torch.nn.Module):
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
                    submod_1 = self.submod_1(l_x_)
                    submod_2 = self.submod_2(l_x_, submod_1);  l_x_ = submod_1 = None
                    return (submod_2,)

                class submod_1(torch.nn.Module):  # Partition for inductor
                    def forward(self, l_x_: "f32[2]"):
                        y: "f32[2]" = torch.sinc(l_x_);  l_x_ = None
                        return y

                class submod_2(torch.nn.Module):  # Partition for thunder
                    def forward(self, l_x_: "f32[2]", y: "f32[2]"):
                        matmul: "f32[]" = torch.matmul(l_x_, y);  l_x_ = y = None
                        return matmul
        """
        # Create an `ThunderOperatorSupport` instance which will be used in the callback.
        # This will determine whether the operation represented by the node is supported by thunder.
        operator_support = ThunderOperatorSupport(gm)

        # The callback below is called for every node in the graph.
        # It returns an `int` denoting the parition where the node should be placed.
        # We want to partition the graph into contiguous regions (with one or more operations)
        # into thunder supported or unsupported region.
        # `prev_value` is used to determine if we are still in same region (i.e. supported region or unsupported region).
        # `partition_cnt` is bumped everytime we change the region i.e. flip from supported to unsupported or from unsupported to supported.
        # `supported_partitions` is used to track the thunder supported partitions.
        prev_value = None
        partition_cnt = 0
        supported_partitions = set()

        def callback(node) -> int:
            assert node.op not in (
                "placeholder",
                "get_attr",
                "output",
            ), f"fx.split_module should have only passed node.op=call_* but received {node.op}"
            nonlocal prev_value, partition_cnt
            is_thunder_supported = operator_support.is_node_supported(gm, node)
            if prev_value == is_thunder_supported:  # We are in the same region.
                return partition_cnt

            # There is a flip. Either from supported to unsupported or unsupported to supported.
            prev_value = is_thunder_supported
            partition_cnt += 1  # Bump the region cnt.

            if is_thunder_supported:
                supported_partitions.add(partition_cnt)
            return partition_cnt

        # `fx_split_module` iterates over nodes and determines the partition to place them based on the callback.
        split_module: torch.fx.GraphModule = fx_split_module(
            gm, root_m=None, split_callback=callback, keep_original_order=True, keep_original_node_name=True
        )

        def is_thunder_supported_partition(node: torch.fx.Node) -> bool:
            return node.name.startswith("submod") and int(node.name.replace("submod_", "")) in supported_partitions

        # Call compile on the split region/s.
        comipled_fn = []
        for node in split_module.graph.nodes:
            if is_thunder_supported_partition(node):
                graph_module = getattr(split_module, node.name)
                jit_fn = self._thunder_jit(graph_module)
                setattr(split_module, node.name, jit_fn)
                comipled_fn.append(CompiledFunction(graph_module, jit_fn, CompilerType.THUNDER))
            elif node.name.startswith("submod"):  # For inductor
                graph_module = getattr(split_module, node.name)
                jit_fn = torch.compile(graph_module, backend="inductor")
                setattr(split_module, node.name, jit_fn)
                comipled_fn.append(CompiledFunction(graph_module, jit_fn, CompilerType.TORCH_INDUCTOR))
            else:
                # Everything else is a glue code to call and pass outputs between the other partitions.
                pass

        gm.print_readable()
        # Append the details regarding this graph/subgraph.
        self.subgraph_infos.append(SubgraphInfo(gm, comipled_fn, True, operator_support.split_reasons, split_module))
        split_module.print_readable()
        return split_module

    def __call__(self, gm: torch.fx.GraphModule, sample_args: list[torch.SymInt, torch.Tensor]):
        from thunder import jit

        # Dynamo uses lazy generation of the underlying Python code, so we need to
        # force recompilation of the GraphModule before passing it to Thunder.
        gm.real_recompile()

        # Check if the complete graph `gm` is supported by thunder
        # If yes, pass the whole `gm` to `thunder.jit` and return the compiled function.
        # if is_graph_supported_by_thunder(gm, sample_args):
        #     jitted_gm = self.thunder_jit(gm)
        #     self.thunder_fns.append(jitted_gm)
        #     self.thunder_to_gm[jitted_gm] = gm
        #     compiled_fn = CompiledFunction(gm, jitted_gm, CompilerType.THUNDER)
        #     self.subgraph_infos.append(SubgraphInfo(gm, [compiled_fn], False))
        #     return jitted_gm

        # The whole graph may not be supported by `thunder`, so we split it in `thunder` supported sections
        # and unsupported sections which are passed to `torch.compile(backend='inductor')`
        split_module = self.splitter(gm, sample_args)
        return split_module
