from __future__ import annotations
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING
import dataclasses
import inspect
import itertools
import copy
import warnings
from pathlib import Path

import torch
from torch.nn.modules.module import _addindent
from torch._subclasses.fake_tensor import FakeTensor

from thunder.torch.default_torch_ops import torch_auto_registered_ops
from thunder.torch import _torch_to_thunder_function_map
from thunder.torch.langctx import torchctx
from thunder.core.utils import check

if TYPE_CHECKING:
    from thunder.core.symbol import Symbol
    import os
    from typing import Any, TextIO

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


@dataclasses.dataclass
class ExampleInputMetaData:
    """
    Describes the metadata of a tensor, used to generate a random tensor with matching properties
    """

    requires_grad: bool
    layout: torch.layout
    device: str | torch.device
    dtype: torch.dtype
    shape: list[int]
    storage_shape: list[int]
    strides: list[int]
    min_val: int | None = None
    max_val: int | None = None

    def stride(self) -> list[int]:
        return self.strides


@dataclasses.dataclass(frozen=True)
class SubgraphInfo:
    """A dataclass containing information about a subgraph.

    Attributes:
        original_graph_module: The original graph module.
        original_split_graph_module: The original split graph module before any transformations are applied.
            Specifically, before the :func:`checkpoint_converter` replaces the Torch operators with Thunder symbols,
            and before any submodules are compiled by Thunder.
        split_graph_module: The graph module for the split subgraph. It contains the compiled thunder/inductor modules.
        thunder_compiled_fns: List of thunder optimized callables.
            This could be :obj:`None` if there the graph module was not supported by thunder.
            Look at the :attr:`split_reasons` for further information.
        thunder_compiled_fns_example_inputs: List containing metadata of sample inputs for `thunder_compiled_fns`.
            These inputs are used to generate random test inputs in the reproducer script.
        submodule_to_compiled_functions: Dict from subgraph in :attr:`original_split_graph_module` to compiled function.
            This will be a dict with one pair in case the graph was not split.
        split_reasons: List of reasons explaining why the subgraph was split.
            Present only if there are was a split.
    """

    original_graph_module: torch.fx.GraphModule
    original_split_graph_module: torch.fx.GraphModule | None
    split_graph_module: torch.fx.GraphModule | None
    thunder_compiled_fns: list[Callable] | None
    thunder_compiled_fns_example_inputs: list[list[ExampleInputMetaData]] | None
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
            # This is a Node in the graph representing a Tensor or tuple of Tensors or
            # a PyTorch object like one representing torch.autocast.
            if isinstance(arg_node, torch.fx.Node):
                if "example_value" not in arg_node.meta:
                    # This is a non tensor object like `torch.autocast` ctx manager object.
                    return arg_node

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
    from thunder.core.compile_data import compile_data_and_stats
    from thunder.common import CompileData, CompileStats

    # This is required for verifying `_enter_autocast`
    # which pushes state onto `CompileData.autocast_stack`.
    cd = CompileData(fn=lambda x: x, disable_preprocessing=True)
    cs = CompileStats()

    @compile_data_and_stats(cd, cs)
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

    # We want to mark nodes disabling `autograd` as unsupported
    # because `thunder` doesn't correctly deal with these stateful functions.

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
        if node.op == "call_function" and is_no_grad_ctx_enter(node):
            ctx_cnt += 1
        elif node.op == "call_function" and is_no_grad_ctx_exit(node):
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

    # The checkpointed function must be fully supported by Thunder
    if target is torch.ops.higher_order.tag_activation_checkpoint:
        m = node.graph.owning_module
        get_attr_node = node.args[0]
        assert get_attr_node.op == "get_attr"
        checkpointed_fn = getattr(m, get_attr_node.target)
        is_module_supported, split_reason = is_graphmodule_supported_by_thunder(checkpointed_fn)
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


def _get_storage_shape(t: torch.Tensor):
    shape = _concrete_shape(t)
    if t.is_contiguous():
        return shape
    strides = t.stride()
    storage_size = sum(strides[i] * (shape[i] - 1) for i in range(len(shape))) + 1
    return (storage_size,)


def _get_example_input_tensor_metadata(t: torch.Tensor) -> ExampleInputMetaData:
    meta_ev = ExampleInputMetaData(
        t.requires_grad, t.layout, t.device, t.dtype, _concrete_shape(t), _get_storage_shape(t), t.stride()
    )
    if not isinstance(t, FakeTensor):
        minmax: tuple[torch.Tensor, torch.Tensor] = torch.aminmax(t)
        meta_ev.min_val = minmax[0].cpu().item()
        meta_ev.max_val = minmax[1].cpu().item()
    return meta_ev


def _create_random_tensor_from_tensor_metadata(t: ExampleInputMetaData) -> torch.Tensor:
    from thunder.tests.make_tensor import make_tensor

    return make_tensor(t.storage_shape, dtype=t.dtype, device=t.device, requires_grad=t.requires_grad).as_strided(
        t.shape, t.stride()
    )


def _get_example_inputs_from_placeholder(
    node: torch.fx.Node, only_metadata=False
) -> tuple[torch.Tensor | ExampleInputMetaData] | torch.Tensor | ExampleInputMetaData:
    """Retrieves example input data for a given placeholder `torch.fx.Node`.
    - When `only_metadata` is `False`: Generates and returns a random example tensor based on the node's expected shape and data type, etc.
    - When `only_metadata` is `True`: Returns only the tensor's metadata (e.g., shape, data type) without generating an actual tensor.
    """
    check(node.op == "placeholder", lambda: f"The node must be placeholder type", ValueError)
    # Prefers to use actual example value in GraphArg if available
    if "grapharg" in node.meta:
        ev = node.meta["grapharg"].example
        if isinstance(ev, torch.Tensor):
            if only_metadata:
                return _get_example_input_tensor_metadata(ev)
            return ev.detach().clone().requires_grad_(ev.requires_grad)

    if "example_value" not in node.meta:
        return None
    example_value = node.meta["example_value"]

    if isinstance(example_value, torch.Tensor):
        ev_metadata = _get_example_input_tensor_metadata(example_value)
        if only_metadata:
            return ev_metadata
        return _create_random_tensor_from_tensor_metadata(ev_metadata)
    elif isinstance(example_value, tuple):
        ev_metadatas = tuple(_get_example_input_tensor_metadata(e_v) for e_v in example_value)
        if only_metadata:
            return ev_metadatas
        return tuple(_create_random_tensor_from_tensor_metadata(ev_metadata) for ev_metadata in ev_metadatas)
    elif isinstance(example_value, torch.types.py_sym_types):
        return example_value.node.hint
    else:
        raise TypeError(f"Unsupported example_value type: {type(example_value)}")


def _checkpoint_function_converter(gm: torch.fx.GraphModule):
    """
    Replace PyTorch operators in ``gm`` representing a checkpointed function with corresponding Thunder operators. The input ``gm`` is modified inplace.

    Args:
        gm (torch.fx.GraphModule): The GraphModule of the checkpointed function, which is modified inplace.
    """
    for n in gm.graph.nodes:
        # replace the torch operator in "call_function" node
        if n.op == "call_function":
            assert isinstance(n.target, Callable)
            if n.target.__module__ in ("_operator", "builtins"):
                continue
            check(
                n.target in _torch_to_thunder_function_map, lambda: f"Unexpected {n.target}, not registered in Thunder"
            )
            with gm.graph.inserting_before(n):
                thunder_node = gm.graph.call_function(
                    _torch_to_thunder_function_map[n.target], args=n.args, kwargs=n.kwargs
                )
            n.replace_all_uses_with(thunder_node)
            gm.graph.erase_node(n)
        else:
            if n.op == "call_module":
                raise RuntimeError(
                    "Unexpected call_module detected inside a checkpoint. This should have been inlined in dynamo graphs"
                )
    gm.graph.lint()
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
                checkpoint_target_node = n.args[0]
                if checkpoint_target_node.op == "get_attr":
                    function_module = getattr(checkpoint_target_node.graph.owning_module, checkpoint_target_node.target)
                else:
                    function_module = getattr(gm, n.args[0].name)
                _checkpoint_function_converter(function_module)


def remove_empty_autocast(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Function to remove empty autocast regions from GraphModule.

    Dynamo can provide empty autocast regions in which case, it is more performant to remove them
    from the graph than to compile them and pay the cost of calling a wrapped optimized function
    which does nothing.

    Args:
        graph_module: Graph module to which this pass is applied.

    """

    empty_autocast_removed_graph_module = copy.deepcopy(graph_module)

    # Dummy init node.
    prev_node = torch.fx.node.Node(graph_module.graph, "start_node", "call_function", lambda: None, None, None)
    nodes_to_erase = []
    for node in empty_autocast_removed_graph_module.graph.nodes:
        # As _enter_autocast and _exit_autocast functions map the regions created by context manager,
        # previous `_enter_autocast` will always correspond with current `_exit_autocast`.
        if (
            prev_node.target == torch.amp.autocast_mode._enter_autocast
            and node.target == torch.amp.autocast_mode._exit_autocast
        ):
            # NOTE: Order of node being appended matters.
            # The node to be erased has to have zero users.
            # So, we remove `_exit_autocast` first (which consumes output from `_enter_autocast`)
            # and then we can remove the corresponding `_enter_autocast`.
            nodes_to_erase.append(node)
            nodes_to_erase.append(prev_node)

        prev_node = node

    # Erase the marked nodes.
    for node in nodes_to_erase:
        empty_autocast_removed_graph_module.graph.erase_node(node)

    return empty_autocast_removed_graph_module


def arg_like_tensor(arg: torch.Tensor | ExampleInputMetaData, f: TextIO):
    """Creates a new argument like the given tensor or tensor metadata"""
    if isinstance(arg, torch.Tensor):
        min_val, max_val = torch.aminmax(arg)
        min_val = min_val.cpu().item()
        max_val = max_val.cpu().item()
    else:
        min_val, max_val = arg.min_val, arg.max_val
    storage_shape = _get_storage_shape(arg) if isinstance(arg, torch.Tensor) else arg.storage_shape
    if min_val is not None and min_val == max_val:
        meta = f"{storage_shape}, {min_val}, dtype={arg.dtype}, device='{arg.device}', requires_grad={arg.requires_grad}, layout={arg.layout}"
        print(f"  torch.full({meta}).as_strided({arg.shape}, {arg.stride()}),", file=f)
        return
    meta = f"{storage_shape}, dtype={arg.dtype},  device='{arg.device}', requires_grad={arg.requires_grad},"
    meta = f"{meta} low={min_val}, high={max_val},"
    print(f"  torch.testing.make_tensor({meta}).as_strided({arg.shape}, {arg.stride()}),", file=f)


def arg_like(arg: Any, f: TextIO):
    """Creates a new argument that is similar to the given arg."""
    if isinstance(arg, (torch.Tensor, ExampleInputMetaData)):
        arg_like_tensor(arg, f)
    else:
        # Assume it's a literal that we can just print directly.
        print(f"  {arg},", file=f)


def _readable(
    module: torch.fx.GraphModule,
    module_name: str,
    print_output: bool = False,
    include_stride: bool = True,
    include_device: bool = True,
    colored: bool = False,
):
    """Modified from `torch.fx.graph_module._print_readable` (https://github.com/pytorch/pytorch/blob/3192bdeea428f2bf3a95274ee59ea41c4f8e31e9/torch/fx/graph_module.py#L297).
    This is basically print_readable but it sets verbose=False (torch hardcodes it to True)."""
    graph = module.graph
    assert graph is not None and isinstance(
        graph, torch.fx.Graph
    ), "print_readable must be used on a module with a graph"

    verbose_python_code = graph.python_code(
        root_module="self",
        verbose=False,
        include_stride=include_stride,
        include_device=include_device,
        colored=colored,
    )
    module_code = verbose_python_code.src
    module_code = module_code.lstrip("\n")
    module_code = f"class {module_name}(torch.nn.Module):\n" + module_code
    module_code = _addindent(module_code, 2)

    submodule_code_list = [""]
    for submodule_name, submodule in module.named_children():
        if hasattr(submodule, "graph"):
            submodule_code_list.append(
                _readable(
                    submodule,
                    submodule_name,
                    print_output=False,
                    include_stride=include_stride,
                    include_device=include_device,
                    colored=colored,
                )
            )
    submodule_code = "\n".join(submodule_code_list)
    submodule_code = _addindent(submodule_code, 2)

    output = module_code + submodule_code
    if print_output:
        print(module_code + submodule_code)
    return output


def get_env() -> tuple[str, str]:
    """Retrieve detailed environment information using `torch.utils.collect_env.get_pretty_env_info()`.
    Additionally, include the installed versions of Thunder and NvFuser (if available via pip).
    """

    from torch.utils.collect_env import run, get_pretty_env_info, get_pip_packages

    torch_env = get_pretty_env_info()
    _, thunder_packages = get_pip_packages(run, {"lightning-thunder", "nvfuser"})
    return torch_env, thunder_packages


def thunder_options_to_str(thunder_options: dict) -> str:
    from thunder import resolve_executors

    option_str = ""
    for key, value in thunder_options.items():
        if key == "executors":
            executors = resolve_executors(value)
            option_str += f"{key}=[" + ",".join(f"thunder.extend.get_executor('{ex.name}')" for ex in executors) + "]"
        else:
            option_str += f"{key}={repr(value)}"
        option_str += ","
    return option_str


def reproducer(
    gm: torch.fx.GraphModule,
    thunder_options: dict,
    args: tuple[torch.Tensor | ExampleInputMetaData],
    folder: str | os.PathLike,
    graph_idx: int,
):
    # Ideally we'd use print_readable, but we want verbose=False and there's no
    # way to set that with print_readable.
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    torch_env, thunder_pkgs = get_env()
    readable = _readable(gm, "DynamoModule", print_output=False)
    has_cuda_args = any(hasattr(arg, "device") and arg.device.type == "cuda" for arg in args)
    thunder_options_str = thunder_options_to_str(thunder_options)
    with open(folder / f"g{graph_idx}.py", "w") as f:
        print('"""', file=f)
        print("Environment information get from `torch.utils.collect_env.get_pretty_env_info()`:\n", file=f)
        print(torch_env, file=f)
        print("\nVersions of Thunder related libraries:", file=f)
        print(thunder_pkgs, file=f)
        print("\nThe torch.fx.Graph:", file=f)
        print(gm.graph, file=f)
        print('"""', file=f)
        print("import os\n", file=f)
        print("import torch", file=f)
        print("import thunder", file=f)
        if has_cuda_args:
            print("import thunder.transforms.cudagraph", file=f)
            print("from thunder.dev_utils.nvtx_profile_transform ", end="", file=f)
            print("import NvtxProfileTransform\n", file=f)
            print("_execs = [", file=f)
            print('  thunder.extend.get_executor("nvfuser"),', file=f)
            print('  thunder.extend.get_executor("sdpa"),', file=f)
            print('  thunder.extend.get_executor("cudnn"),', file=f)
            print("]\n", file=f)
        print(f"def test_g{graph_idx}():", file=f)
        print(" ", _addindent(readable, 2), file=f)
        if any(arg is None for arg in args):
            print(
                "  # Warning: The inputs that cannot be inferred are set to None, requiring the user to manually give inputs according to the code"
            )
        print("  inputs = [", file=f)
        for a in args:
            print("  ", end="", file=f)
            arg_like(a, f)
        print("  ]", file=f)
        print(
            """  # NOTE the `BACKEND` environment variable is intended to provide some common ways to debug/benchmark thunder.jit
  # with different backend and compilation options. By default, it uses the original Thunder options that are executed""",
            file=f,
        )
        print('  backend = os.getenv("BACKEND")', file=f)
        # thunder_options_str = ", ".join(f"{key}={repr(value)}" for key, value in thunder_options.items())
        print('  if backend == None or backend == "thunder":', file=f)
        print(f"    fqn = thunder.jit(DynamoModule(), {thunder_options_str})", file=f)
        print('  elif backend == "torch.compile":', file=f)
        print("    fqn = torch.compile(DynamoModule())", file=f)
        print('  elif backend == "dynamo-eager":', file=f)
        print('    fqn = torch.compile(DynamoModule(), backend="eager")', file=f)
        if has_cuda_args:
            print('  elif backend == "thunder-nvtxprofile":', file=f)
            print("    fqn = thunder.jit(DynamoModule(), transforms=[NvtxProfileTransform()])", file=f)
            print('  elif backend == "thunder-no-torch.compile":', file=f)
            print("    fqn = thunder.jit(DynamoModule(), executors=_execs)", file=f)
            print('  elif backend == "thunder-cudagraph":', file=f)
            print("    xform = thunder.transforms.cudagraph.CUDAGraphTransform()", file=f)
            print("    fqn = thunder.jit(DynamoModule(), transform=[xform])", file=f)
            print(f'  post_graph = os.getenv("POST_GRAPH", "0")', file=f)
            print(f"  if int(post_graph) > 0:", file=f)
            print(f"    fqn = torch.cuda.make_graphed_callables(", file=f)
            print(f"      fqn, inputs,", file=f)
            print(f"      num_warmup_iters=1, allow_unused_input=True", file=f)
            print(f"    )", file=f)
            print(f'  torch.cuda.nvtx.range_push("g{graph_idx} warmups")', file=f)
        print(f"  for i in range(3): # warmup runs", file=f)
        print(f"    fqn(*inputs)", file=f)
        if has_cuda_args:
            print(f"  torch.cuda.synchronize()", file=f)
            print(f"  torch.cuda.nvtx.range_pop()", file=f)
            print(f'  torch.cuda.nvtx.range_push("g{graph_idx}")', file=f)
        print(f"  fqn(*inputs)", file=f)
        if has_cuda_args:
            print(f"  torch.cuda.synchronize()", file=f)
            print(f"  torch.cuda.nvtx.range_pop()", file=f)
        print(f"\ntest_g{graph_idx}()", file=f)
