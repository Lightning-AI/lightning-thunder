from __future__ import annotations
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING
import dataclasses
import inspect
import itertools
import copy

import torch

from thunder.torch.default_torch_ops import torch_auto_registered_ops
from thunder.torch import _torch_to_thunder_function_map
from thunder.torch.langctx import torchctx
from thunder.core.utils import check
from thunder.core.pytree import tree_map

if TYPE_CHECKING:
    from thunder.core.symbol import Symbol

auto_register_ops = set(itertools.chain(*torch_auto_registered_ops.values()))


# Currently, thunder as mapping torch these function but they
# just throw warning.
UNSUPPORTED_THUNDER_FUNCTION = (torch._C._set_grad_enabled,)


class CompilerType(Enum):
    """
    An enumeration representing different types of compilers.
    """

    THUNDER = auto()
    TORCH_INDUCTOR = auto()


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class SplitReason:
    """A dataclass containing information about a split.

    Attributes:
        reason_type: Reason for the split.
        info: String with details of what caused the split.
        exception: Exception if there was any.
    """

    reason_type: SplitReasonType
    info: str | None
    exception: Exception | None = None


@dataclasses.dataclass(frozen=True)
class SubgraphInfo:
    """A dataclass containing information about a subgraph.

    Attributes:
        original_graph_module: The original graph module.
        split_graph_module: The graph module for the split subgraph.
        thunder_compiled_fns: List of thunder optimized callables.
            This could be :obj:`None` if there the graph module was not supported by thunder.
            Look at the :attr:`split_reasons` for further information.
        submodule_to_compiled_functions: Dict from subgraph to compiled function.
            This will be a dict with one pair in case the graph was not split.
        split_reasons: List of reasons explaining why the subgraph was split.
            Present only if there are was a split.
    """

    original_graph_module: torch.fx.GraphModule
    split_graph_module: torch.fx.GraphModule | None
    thunder_compiled_fns: list[Callable] | None
    submodule_to_compiled_functions: dict[torch.fx.GraphModule, CompiledFunction]
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


def get_proxy_inputs_from_node(node: torch.fx.Node) -> tuple[tuple, dict]:
    """Creates proxy inputs from a torch.fx.Node for use with Thunder.

    This function generates proxy inputs for a given torch.fx.Node

    Args:
        node (torch.fx.Node): The FX graph node to create proxy inputs for.
    """
    import thunder
    from thunder.core.trace import TraceCtx
    from thunder.core.proxies import proxy

    # We need to be under trace context to generate proxies.
    with thunder.core.trace.tracectx(TraceCtx()):

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
                        e_v.new_ones(_concrete_shape(e_v), device=e_v.device, dtype=e_v.dtype) for e_v in example_value
                    )
                else:
                    # NOTE - This will be caught will be caught and be part of the SplitReason.
                    raise TypeError(f"Received `make_tensor_proxy` received example_value which wasn't Tensor or Tuple")
                return proxy(example_value)

            # This is int, float, etc.
            return arg_node

        proxy_args = torch.fx.map_arg(node.args, make_tensor_proxy)
        proxy_kwargs = {k: torch.fx.map_arg(v, make_tensor_proxy) for k, v in node.kwargs.items()}
        return proxy_args, proxy_kwargs


def try_execute_thunder_symbol(thunder_symbol: Symbol, node: torch.fx.Node) -> tuple[bool, SplitReason | None]:
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

        try:
            proxy_args, proxy_kwargs = get_proxy_inputs_from_node(node)
        except Exception as e:
            return False, SplitReason(
                SplitReasonType.EXCEPTION_PROXY_THUNDER_OP,
                f"Failed while creating proxy for node with name: {node.name} and target: {node.target}, see exception field",
                exception=e,
            )

        # We need to be under trace context to generate proxies.
        with thunder.core.trace.tracectx(TraceCtx()):
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


def get_nodes_in_unsupported_ctx_regions(gm: torch.fx.GraphModule) -> set[torch.fx.Node]:
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

    def is_no_grad_ctx_enter(node):
        if node.target == torch._C._set_grad_enabled:
            arg: bool = node.args[0]
            assert isinstance(arg, bool)
            return not arg  # arg is False (i.e. grad was disabled)
        return False

    def is_no_grad_ctx_exit(node):
        if node.target == torch._C._set_grad_enabled:
            arg: bool = node.args[0]
            assert isinstance(arg, bool)
            return arg  # arg is True (i.e. grad was enabled)
        return False

    for node in gm.graph.nodes:
        if node.op == "call_function" and (
            node.target in (torch.amp.autocast_mode._enter_autocast,) or is_no_grad_ctx_enter(node)
        ):
            ctx_cnt += 1
        elif node.op == "call_function" and (
            node.target in (torch.amp.autocast_mode._exit_autocast,) or is_no_grad_ctx_exit(node)
        ):
            ctx_cnt -= 1
        else:
            if ctx_cnt > 0:
                nodes_in_unsupported_ctx_regions.add(node)

    return nodes_in_unsupported_ctx_regions


def is_graphmodule_supported_by_thunder(gm):
    nodes_in_unsupported_ctx_regions = get_nodes_in_unsupported_ctx_regions(gm)
    for node in gm.graph.nodes:
        if node.op in (
            "placeholder",
            "get_attr",
            "output",
        ):
            continue
        if node in nodes_in_unsupported_ctx_regions:
            split_reason = SplitReason(
                SplitReasonType.UNSUPPORTED_NODE,
                info=f"node with name: {node.name} and target: {node.target} is not supported probably because it is in unsupported context.",
            )
            return False, split_reason
        is_thunder_supported, split_reason = is_node_supported_by_thunder(node)
        if not is_thunder_supported:
            return False, split_reason
    return True, None


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

    # These functions are present in `_torch_to_thunder_function_map` but don't mimic exact behavior.
    # Eg. torch._C._set_grad_enabled's thunder implementation just throws warning that this is unsupported.
    if target in UNSUPPORTED_THUNDER_FUNCTION:
        split_reason = SplitReason(
            SplitReasonType.UNSUPPORTED_NODE,
            info=f"node with name: {node.name} and target: {node.target} has been manually disabled.",
        )
        return False, split_reason

    # If the operation is higher order function for checkpointing, check whether the submodule is supported by Thunder
    if target is torch.ops.higher_order.tag_activation_checkpoint:
        m = node.graph.owning_module
        assert hasattr(m, node.args[0].name)
        higher_order_module = getattr(m, node.args[0].name)
        is_module_supported, split_reason = is_graphmodule_supported_by_thunder(higher_order_module)
        return is_module_supported, split_reason

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

    # There are few operations which are registered only as method in `torchctx` and hence they don't exist
    # in `_torch_to_thunder_function_map` (eg. `float`)
    # For these method, we try to look them up with `torchctx` language context.
    # NOTE: We pass `node.target` which is a `str` (and not `target` from above which is actually function object).
    if torchctx.has_method(node.target):
        # `torchctx.get_method` requires args and kwargs to resolve which overload of the method is picked.
        try:
            args, kwargs = get_proxy_inputs_from_node(node)
        except Exception as e:
            return False, SplitReason(
                SplitReasonType.EXCEPTION_PROXY_THUNDER_OP,
                f"Failed while creating proxy for node with name: {node.name} and target: {node.target}, see exception field",
                exception=e,
            )
        # NOTE: `get_method` may throw if relevant method is not found, so we have guarded it with `has_method`.
        method = torchctx.get_method(node.target, args, kwargs)
        did_run, opt_split_reason = try_execute_thunder_symbol(method, node)
        return did_run, opt_split_reason

    # We found no automatic fallback registration and no mapping to thunder symbol.
    split_reason = SplitReason(
        SplitReasonType.MISSING_OP_SUPPORT,
        info=f"node with name: {node.name} and target: {node.target} didn't have any mapping in thunder.",
    )
    return False, split_reason


def update_node_and_submodule(
    graph_module: torch.fx.GraphModule,
    node: torch.fx.Node,
    new_name: str,
    new_callable: Callable,
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


def recompile_graph(gm: torch.fx.GraphModule):
    # NOTE - `gm` could also be the `_LazyGraphModule`, in which case calling `recompile` is not enough as it marks the `GraphModule`
    # and actual recompilation happens when `real_recompile` is called or either `forward` or `code` (or when user tries to observe the GraphModule).
    # See for more details - https://github.com/pytorch/pytorch/blob/39935e0fdef02c67ba808175dcc800d0695bfe1b/torch/fx/_lazy_graph_module.py#L65-L89
    if isinstance(gm, torch.fx._lazy_graph_module._LazyGraphModule):
        return gm.real_recompile()
    return gm.recompile()


def _get_example_inputs_from_placeholder(node) -> tuple[torch.Tensor]:
    from thunder.tests.make_tensor import make_tensor

    check(node.op == "placeholder", lambda: f"The node must be placeholder type", ValueError)
    # Prefers to use actual example value in GraphArg if available
    if "grapharg" in node.meta:
        example_value = node.meta["grapharg"].example
        if isinstance(example_value, torch.Tensor):
            return (example_value.detach().clone().requires_grad_(example_value.requires_grad),)

    check("example_value" in node.meta, lambda: "example_value does not exist in the meta of {node}", ValueError)
    example_value = node.meta["example_value"]

    if isinstance(example_value, torch.Tensor):
        sz = _concrete_shape(example_value)
        return (
            make_tensor(
                sz,
                dtype=example_value.dtype,
                device=example_value.device,
                requires_grad=example_value.requires_grad,
            ).as_strided(sz, example_value.stride()),
        )
    elif isinstance(example_value, tuple):
        return tuple(
            make_tensor(
                _concrete_shape(e_v),
                dtype=e_v.dtype,
                device=e_v.device,
                requires_grad=e_v.requires_grad,
            ).as_strided(_concrete_shape(e_v), e_v.stride())
            for e_v in example_value
        )
    else:
        raise TypeError(
            "The 'example_value' in the placeholder node is expected to be either a Tensor or a Tuple of Tensors."
        )
    

def _checkpoint_function_converter(gm: torch.fx.GraphModule):
    """
    Replace the Torch operators in the GraphModule called by activation checkpoint operator with the corresponding Thunder symbols in place
    Args:
        gm: The GraphModule of the checkpointed function, which is modified in place
    """
    new_graph = copy.deepcopy(gm.graph)
    for n in new_graph.nodes:
        # replace the torch operator in "call_function" node
        if n.op == "call_function":
            assert isinstance(n.target, Callable)
            if n.target.__module__ in ("_operator", "builtins"):
                continue
            check(
                n.target in _torch_to_thunder_function_map, lambda: f"Unexpected {n.target}, not registered in Thunder"
            )
            with new_graph.inserting_before(n):
                thunder_node = new_graph.call_function(
                    _torch_to_thunder_function_map[n.target], args=n.args, kwargs=n.kwargs
                )
            n.replace_all_uses_with(thunder_node)
            new_graph.erase_node(n)
        else:
            if n.op == "call_module":
                raise RuntimeError(
                    "Unexpected call_module detected inside a checkpoint. This should have been inlined in dynamo graphs"
                )
    new_graph.lint()
    gm.graph = new_graph
    recompile_graph(gm)


def checkpoint_converter(gm: torch.fx.GraphModule, sub_gm: torch.fx.GraphModule):
    """
    Utility function to convert the GraphModule that uses activation checkpointing into a Thunder-traceable GraphModule.

    Args:
        gm: The parent GraphModule containing the submodule(sub_gm), as well as the GraphModule of the checkpointed function.
        sub_gm: the GraphModule containing the checkpoint operator

    Note:
        The GraphModule of the checkpointed function is updated inplace
    """
    for n in sub_gm.graph.nodes:
        if n.op == "call_function":
            if n.target in (torch.ops.higher_order.tag_activation_checkpoint,):
                name = n.args[0].name
                assert hasattr(gm, name)
                function_module = getattr(gm, name)
                _checkpoint_function_converter(function_module)
