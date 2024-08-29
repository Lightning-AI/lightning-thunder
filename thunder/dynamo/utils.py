from enum import Enum, auto
import dataclasses
from typing import List, Dict, Optional, Tuple, Set
from collections.abc import Callable
import pprint
import itertools
import copy
import inspect

import torch
from torch.fx.passes.split_module import split_module
import warnings
from collections.abc import Mapping

from thunder.torch.default_torch_ops import torch_auto_registered_ops
from thunder.torch import _torch_to_thunder_function_map


auto_register_ops = set(itertools.chain(*torch_auto_registered_ops.values()))


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
        compiled_fn (Callable): The compiled function.
        compiler (CompilerType): The type of compiler used to compile the function.
    """

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
        split_graph_module (torch.fx.GraphModule): Optional. The graph module for the split subgraph.
        thunder_compiled_fns (list[Callable]): List of thunder optimized callables. This could be None if there the graph module was not supported by thunder. Look at the `split_reasons` for further information.
        compiled_functions (list[CompiledFunction]): A list of compiled functions derived from the subgraph. This will be a list with one function in case the graph was not split.
        split_reasons (list[SplitReason] | None): Optional list of reasons explaining why the subgraph was split. Present only if there are was a split.
    """

    original_graph_module: torch.fx.GraphModule
    split_graph_module: torch.fx.GraphModule
    thunder_compiled_fns: list[Callable]
    submodule_to_compiled_functions: Mapping[torch.fx.GraphModule, CompiledFunction]
    split_reasons: list | None = None


def _concrete_shape(x):
    """
    Get the concrete shape for a FakeTensor if it has `torch.SymInt` in its shape.
    """

    def get_backed_value(s):
        if isinstance(s, torch.SymInt):
            return s.node.hint
        # Value is already concrete.
        return s

    return tuple(map(get_backed_value, x.shape))


def try_execute_thunder_symbol(thunder_symbol: "Symbol", node: torch.fx.Node) -> tuple[bool, SplitReason | None]:
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

    @thunder._with_cache_info_ctx
    def _run_with_cache_info():

        # We need cache info here as the default dtype and device support
        # for factory functions like ones, zeros, etc expects these details to be present.
        # TODO: Move this to CompileData as well?
        # This details are in cache_info because `jit_ext.py`
        # adds checks in prologue for the details which are present in here.
        cache_info = thunder._get_cache_info()
        cache_info["default_dtype"] = torch.get_default_dtype()
        cache_info["default_device"] = torch.get_default_device()

        trc = TraceCtx()
        # We need to be under trace context to generate proxies.
        with thunder.core.trace.tracectx(trc):
            try:

                def make_tensor_proxy(arg_node):
                    # This is a Node in the graph representing a Tensor or tuple of Tensors.
                    if isinstance(arg_node, torch.fx.Node):
                        example_value = arg_node.meta["example_value"]

                        if isinstance(example_value, torch.Tensor):
                            # If `dynamic` shapes are enabled, we may see a FakeTensor
                            # where shape has SymInt. In that case, we check if we can
                            # get the concrete value from SymInt.
                            # Here, we only want to verify that thunder can run an operation.
                            # So, it is ok to verify with concrete value.
                            example_value = example_value.new_ones(
                                _concrete_shape(example_value), device=example_value.device, dtype=example_value.dtype
                            )
                        elif isinstance(example_value, tuple):
                            example_value = tuple(
                                e_v.new_ones(_concrete_shape(e_v), device=e_v.device, dtype=e_v.dtype)
                                for e_v in example_value
                            )
                        else:
                            # NOTE - This will be caught will be caught and be part of the SplitReason.
                            raise TypeError(
                                f"Received `make_tensor_proxy` received example_value which wasn't Tensor or Tuple"
                            )
                        return proxy(example_value)

                    # This is int, float, etc.
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

    return _run_with_cache_info()


def get_nodes_in_unsupported_ctx_regions(gm) -> set[torch.fx.Node]:
    """
    Finds the node within `autocast` or other supported context and marks them as unsupported.
    Even though, thunder may support the operation within the reason, it doesn't correctly apply the change
    triggered from the context.
    """
    # NOTE - Currently only detects the autocast regions.

    nodes_in_unsupported_ctx_regions: set[torch.fx.Node] = set()
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
                nodes_in_unsupported_ctx_regions.add(node)

    return nodes_in_unsupported_ctx_regions


def is_node_supported_by_thunder(node: torch.fx.Node) -> tuple[bool, SplitReason | None]:
    """
    Determine whether thunder can execute the operation described by this node.
    """
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
        split_reason = SplitReason(
            SplitReasonType.MISSING_OP_SUPPORT,
            info=f"node with name: {node.name} and target: {node.target} only has an automatic torch fallback in thunder.",
        )
        return False, split_reason

    # If thunder has a mapping for this operation, try executing the meta function and see.
    # We have a symbol for `torch.where`, but we don't support one overload of it.
    # So, we try and execute the meta to get a real signal.
    #
    # Regarding `inspect.isbuiltin`, dynamo graph uses `+`, `>` which are builtin `add`, `gt`.
    # We try to proxify the arguments and call these operations on them to see if they are supported.
    if target in _torch_to_thunder_function_map or inspect.isbuiltin(target):
        thunder_symbol_or_builtin = _torch_to_thunder_function_map.get(target, target)
        did_run, opt_split_reason = try_execute_thunder_symbol(thunder_symbol_or_builtin, node)
        return did_run, opt_split_reason

    # We found no automatic fallback registration and no mapping to thunder symbol.
    split_reason = SplitReason(
        SplitReasonType.MISSING_OP_SUPPORT,
        info=f"node with name: {node.name} and target: {node.target} didn't have any mapping in thunder.",
    )
    return False, split_reason


def update_node_and_submodule(
    graph_module: torch.fx.GraphModule, node: torch.fx.Node, new_name: str, new_callable: Callable
):
    """
    Updates the graph module and the node in place with a new name and a new callable as the target.

    This function removes the existing submodule associated with the node's current name in graph_module and replaces
    it with a new submodule using the specified new name and callable. The node's name and target are updated accordingly.

    Args:
        graph_module (torch.fx.GraphModule): The graph module containing the node and submodules.
        node (torch.fx.Node): The node to be updated within the graph module.
        new_name (str): The new name to assign to the node and the submodule.
        new_callable (Callable): The new callable to be used as the target for the submodule.
    """
    assert graph_module.delete_submodule(
        node.name
    ), f"Didn't find a submodule named {node.name} in graph_module {graph_module}"
    node.name = new_name
    node.target = new_name
    assert graph_module.add_submodule(
        node.name, new_callable
    ), f"Adding submodule with name {node.name} in graph_module {graph_module} failed"
