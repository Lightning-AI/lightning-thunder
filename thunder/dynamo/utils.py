from __future__ import annotations
from collections.abc import Callable, Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING
import dataclasses
import inspect
import itertools
import copy
from types import NoneType
from collections import defaultdict
from collections import namedtuple

import torch
from torch.nn.modules.module import _addindent
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils.weak import TensorWeakRef

if torch.distributed.is_available():
    from torch.distributed.tensor import DTensor
else:
    DTensor = NoneType

from thunder.torch.default_torch_ops import torch_auto_registered_ops
from thunder.torch import _torch_to_thunder_function_map
from thunder.torch.langctx import torchctx
from thunder.core.utils import check
from thunder.core.pytree import tree_flatten

if TYPE_CHECKING:
    from numbers import Number
    from thunder.core.symbol import Symbol
    from typing import Any
    from collections.abc import Sequence

auto_register_ops = set(itertools.chain(*torch_auto_registered_ops.values()))


# Currently, thunder has mapping for these torch function but they
# just raise a warning (or don't support the exact behaviour)
# Previously used for `torch._C._set_grad_enabled` when it just raised a warning.
UNSUPPORTED_THUNDER_FUNCTION = ()


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
        exception: String with details of exception if there was any.
    """

    reason_type: SplitReasonType
    info: str | None
    exception: str | None = None


@dataclasses.dataclass(frozen=True)
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
    is_contiguous: bool
    _storage_offset: int
    min_val: int | None = dataclasses.field(default=None, compare=False, hash=False)
    max_val: int | None = dataclasses.field(default=None, compare=False, hash=False)

    def stride(self) -> list[int]:
        return self.strides

    def storage_offset(self) -> int:
        return self._storage_offset


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


class _ThunderSplitGraphModule:
    def __init__(self, split_graph_module, supported_partitions):
        self.split_graph_module = split_graph_module
        self.supported_indexes: set[int] = supported_partitions

    def is_thunder_supported_partition(self, node: torch.fx.Node) -> bool:
        return node.name.startswith("submod") and int(node.name.replace("submod_", "")) in self.supported_indexes


@dataclasses.dataclass()
class ProfileStats:
    """
    A dataclass that stores profiling statistics for a GraphModule.

    Attributes:
        gm: The GraphModule being profiled.
        input_meta_to_called_times: A dictionary mapping input metadata to the number of times the input has been called.
    """

    gm: torch.fx.GraphModule
    input_meta_to_called_times: dict[tuple[ExampleInputMetaData, Number], int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )


def _concrete_value(vals: torch.Size | Sequence):
    """
    Get the concrete value from the input `vals` if it contains `torch.SymInt`.
    """

    def get_backed_value(s):
        if isinstance(s, torch.SymInt):
            return s.node.hint
        # Value is already concrete.
        return s

    return tuple(map(get_backed_value, vals))


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

        def make_input_proxy(arg_node):
            # This is a Node in the graph representing a Tensor or tuple of Tensors or
            # a PyTorch object like one representing torch.autocast.
            if isinstance(arg_node, torch.fx.Node):
                # Higher-order operator nodes take get_attr nodes as input to get the called module
                if arg_node.op == "get_attr":
                    attr = getattr(arg_node.graph.owning_module, arg_node.target)
                    if isinstance(attr, torch.nn.Module):
                        return attr
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
                        _concrete_value(example_value.shape), device=example_value.device, dtype=example_value.dtype
                    )
                elif isinstance(example_value, tuple):
                    example_value = tuple(
                        e_v.new_ones(_concrete_value(e_v.shape), device=e_v.device, dtype=e_v.dtype)
                        for e_v in example_value
                    )
                elif isinstance(example_value, torch.types.py_sym_types) and example_value.node.has_hint():
                    return proxy(example_value.node.hint)
                else:
                    # NOTE - This will be caught and be part of the SplitReason.
                    raise TypeError(
                        f"`make_input_proxy` received unsupported example_value type: {type(example_value)}"
                    )
                return proxy(example_value)

            # This is int, float, etc.
            return arg_node

        proxy_args = torch.fx.map_arg(node.args, make_input_proxy)
        proxy_kwargs = {k: torch.fx.map_arg(v, make_input_proxy) for k, v in node.kwargs.items()}
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
    from thunder.core.transforms import value_and_grad

    # This is required for verifying `_enter_autocast`
    # which pushes state onto `CompileData.autocast_stack`.
    cd = CompileData(fn=lambda x: x, disable_preprocessing=True)
    cs = CompileStats()

    def get_requires_grad(arg_node):
        if not isinstance(arg_node, torch.fx.Node):
            return False

        if "example_value" not in arg_node.meta:
            return False

        example_value = arg_node.meta["example_value"]
        flattened_example_value, _ = tree_flatten(example_value)
        for x in flattened_example_value:
            if isinstance(x, torch.Tensor) and x.requires_grad:
                return True
        return False

    args, _ = tree_flatten((node.args, node.kwargs))
    requires_grad = any(map(get_requires_grad, args))

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
                exception=str(e),
            )

        function_to_run = value_and_grad(thunder_symbol) if requires_grad else thunder_symbol
        # We need to be under trace context to generate proxies.
        with thunder.core.trace.tracectx(TraceCtx()):
            try:
                function_to_run(*proxy_args, **proxy_kwargs)
            except Exception as e:
                return False, SplitReason(
                    SplitReasonType.EXCEPTION_META_THUNDER_OP,
                    f"Failed while running meta for node with name: {node.name} and target: {node.target}, see exception field",
                    exception=str(e),
                )

        # Execution with proxies was successful.
        return True, None

    return _run_with_cache_info()


def get_nodes_in_unsupported_ctx_regions(gm: torch.fx.GraphModule) -> set[torch.fx.Node]:
    """
    Finds the node within unsupported context and marks them as unsupported.
    Even though, thunder may support the operation within the reason, it doesn't correctly apply the change
    triggered from the context.
    """
    # NOTE - Currently doesn't ban any ctx (previously used for `no_grad` and `autocast`).

    nodes_in_unsupported_ctx_regions: set[torch.fx.Node] = set()
    ctx_cnt = 0  # Count of  we have seen till now

    UNSUPPORTED_THUNDER_CTX = ()
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in UNSUPPORTED_THUNDER_CTX:
            ctx_cnt += 1
        elif node.op == "call_function" and node.target in UNSUPPORTED_THUNDER_CTX:
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
    # Eg. previously torch._C._set_grad_enabled's thunder implementation just threw warning that this is unsupported.
    if target in UNSUPPORTED_THUNDER_FUNCTION:
        split_reason = SplitReason(
            SplitReasonType.UNSUPPORTED_NODE,
            info=f"node with name: {node.name} and target: {node.target} has been manually disabled.",
        )
        return False, split_reason

    # The higher order function must be fully supported by Thunder
    if target in (torch.ops.higher_order.tag_activation_checkpoint, torch.ops.higher_order.autograd_function_apply):
        m = node.graph.owning_module
        for arg_node in node.args:
            if arg_node.op == "get_attr":
                called_module = getattr(m, arg_node.target)
                is_module_supported, split_reason = is_graphmodule_supported_by_thunder(called_module)
                if not is_module_supported:
                    return is_module_supported, split_reason
        return True, None

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
                exception=str(e),
            )
        # NOTE: `get_method` may throw if relevant method is not found, so we have guarded it with `has_method`.
        method = torchctx.get_method(node.target, args, kwargs)
        did_run, opt_split_reason = try_execute_thunder_symbol(method, node)
        return did_run, opt_split_reason

    # checks einops operators
    if hasattr(target, "__module__") and target.__module__ == "einops.einops":
        from thunder.executors.torchex import has_einops

        if has_einops:
            import einops

            # According to https://github.com/Lightning-AI/lightning-thunder/blob/4f92190d/thunder/tests/test_einops.py
            einops_ops = (einops.reduce, einops.rearrange, einops.repeat, einops.einsum)
            if target in einops_ops:
                return True, None

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
    assert graph_module.delete_submodule(node.name), (
        f"Didn't find a submodule named {node.name} in graph_module {graph_module}"
    )
    node.name = new_name
    node.target = new_name
    assert graph_module.add_submodule(node.name, new_callable), (
        f"Adding submodule with name {node.name} in graph_module {graph_module} failed"
    )


def recompile_graph(gm: torch.fx.GraphModule):
    # NOTE - `gm` could also be the `_LazyGraphModule`, in which case calling `recompile` is not enough as it marks the `GraphModule`
    # and actual recompilation happens when `real_recompile` is called or either `forward` or `code` (or when user tries to observe the GraphModule).
    # See for more details - https://github.com/pytorch/pytorch/blob/39935e0fdef02c67ba808175dcc800d0695bfe1b/torch/fx/_lazy_graph_module.py#L65-L89
    if isinstance(gm, torch.fx._lazy_graph_module._LazyGraphModule):
        return gm.real_recompile()
    return gm.recompile()


# Gets the minimum storage shape according to the shape, stride and storage offset
def _get_storage_shape(t: torch.Tensor):
    shape = _concrete_value(t.shape)
    strides = _concrete_value(t.stride())
    storage_offset = t.storage_offset()
    storage_size = storage_offset + sum(strides[i] * (shape[i] - 1) for i in range(len(shape))) + 1
    return (storage_size,)


def _get_min_and_val(t: torch.Tensor) -> tuple[Number | None, Number | None]:
    # We assume that for TensorSubclass, `aminmax` is not supported which is true for FakeTensor and DTensor.
    if (
        (isinstance(t, torch.Tensor) and type(t) is not torch.Tensor)
        or t.device.type == "meta"
        or t.numel() == 0
        or t.dtype.is_complex
    ):
        return None, None
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz):
        t = t.to(torch.float32)
    minmax: tuple[torch.Tensor, torch.Tensor] = torch.aminmax(t)
    min_val = minmax[0].detach().cpu().item()
    max_val = minmax[1].detach().cpu().item()
    return min_val, max_val


def _get_example_input_tensor_metadata(t: torch.Tensor) -> ExampleInputMetaData:
    min_val, max_val = _get_min_and_val(t)
    meta_ev = ExampleInputMetaData(
        t.requires_grad,
        t.layout,
        t.device,
        t.dtype,
        _concrete_value(t.shape),
        _get_storage_shape(t),
        _concrete_value(t.stride()),
        t.is_contiguous(),
        t.storage_offset(),
        min_val,
        max_val,
    )
    return meta_ev


def _create_random_tensor_from_tensor_metadata(arg: ExampleInputMetaData) -> torch.Tensor:
    min_val, max_val = arg.min_val, arg.max_val
    if min_val is not None and min_val == max_val:
        tensor = torch.full(arg.storage_shape, min_val, dtype=arg.dtype, device=arg.device, layout=arg.layout)
    else:
        tensor = torch.testing.make_tensor(
            arg.storage_shape, dtype=arg.dtype, device=arg.device, low=min_val, high=max_val
        )
    return tensor.set_(tensor, size=arg.shape, storage_offset=arg.storage_offset(), stride=arg.stride()).requires_grad_(
        arg.requires_grad
    )


def example_input_meta_to_input(meta):
    if isinstance(meta, ExampleInputMetaData):
        return _create_random_tensor_from_tensor_metadata(meta)
    elif isinstance(meta, (int, bool, float)):
        return meta
    elif isinstance(meta, Sequence):
        return tuple(example_input_meta_to_input(i) for i in meta)
    else:
        raise TypeError(f"Unsupported input type: {type(meta)}")


def input_to_example_input_meta(input):
    if isinstance(input, torch.Tensor):
        return _get_example_input_tensor_metadata(input)
    elif isinstance(input, (int, bool, float)):
        return input
    elif isinstance(input, torch.types.py_sym_types):
        return input.node.hint
    elif isinstance(input, Sequence):
        return tuple(input_to_example_input_meta(i) for i in input)
    else:
        raise TypeError(f"Unsupported input type: {type(input)}")


def _get_example_inputs_from_placeholder(
    node: torch.fx.Node, only_metadata=False
) -> tuple[torch.Tensor | ExampleInputMetaData] | torch.Tensor | ExampleInputMetaData:
    """Retrieves example input data for a given placeholder `torch.fx.Node`.
    - When `only_metadata` is `False`: Generates and returns a random example tensor based on the node's expected shape and data type, etc.
    - When `only_metadata` is `True`: Returns only the tensor's metadata (e.g., shape, data type) without generating an actual tensor.
    """
    check(node.op == "placeholder", lambda: "The node must be placeholder type", ValueError)
    # Prefers to use actual example value in GraphArg if available
    if "grapharg" in node.meta:
        try:
            ev = node.meta["grapharg"].example
        except AssertionError:
            # TensorWeakRef is None
            pass
        else:
            if isinstance(ev, torch.Tensor):
                ev_metadata = _get_example_input_tensor_metadata(ev)
                if only_metadata:
                    return ev_metadata
                return _create_random_tensor_from_tensor_metadata(ev_metadata)

    if "example_value" not in node.meta:
        return None
    example_value = node.meta["example_value"]

    example_value = input_to_example_input_meta(example_value)
    if only_metadata:
        return example_value
    return example_input_meta_to_input(example_value)


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


def arg_like_tensor(arg: torch.Tensor | ExampleInputMetaData):
    """Creates a new argument like the given tensor or tensor metadata"""
    if isinstance(arg, torch.Tensor):
        arg = _get_example_input_tensor_metadata(arg)
    min_val, max_val = arg.min_val, arg.max_val
    shape = arg.shape if arg.is_contiguous and arg.storage_offset() == 0 else arg.storage_shape
    if min_val is not None and min_val == max_val:
        meta = f"{shape}, {min_val}, dtype={arg.dtype}, device='{arg.device}', requires_grad={arg.requires_grad}, layout={arg.layout}"
        tensor_str = f"torch.full({meta})"
    else:
        meta = f"{shape}, dtype={arg.dtype},  device='{arg.device}', requires_grad={arg.requires_grad},"
        meta = f"{meta} low={min_val}, high={max_val},"
        tensor_str = f"torch.testing.make_tensor({meta})"
    if arg.is_contiguous and arg.storage_offset() == 0:
        return f"{tensor_str},"
    return f"{tensor_str}.as_strided({arg.shape}, {arg.stride()}, {arg.storage_offset()}),"


def arg_like(arg: Any):
    """Creates a new argument that is similar to the given arg."""
    if isinstance(arg, (torch.Tensor, ExampleInputMetaData)):
        return arg_like_tensor(arg)
    elif isinstance(arg, Sequence):
        return "[" + "".join(arg_like(a) for a in arg) + "],"
    elif isinstance(arg, (int, bool, float)):
        return f"{arg},"
    else:
        raise TypeError(f"Unsupported input type: {type(arg)}")


def _readable(
    module: torch.fx.GraphModule,
    module_name: str,
    print_output: bool = False,
    verbose: bool = True,
    include_stride: bool = True,
    include_device: bool = True,
    colored: bool = False,
):
    """Modified from `torch.fx.graph_module._print_readable` (https://github.com/pytorch/pytorch/blob/3192bdeea428f2bf3a95274ee59ea41c4f8e31e9/torch/fx/graph_module.py#L297).
    Note: the include_stride and include_device take effects only when verbose is True."""
    graph = module.graph
    assert graph is not None and isinstance(graph, torch.fx.Graph), (
        "print_readable must be used on a module with a graph"
    )

    verbose_python_code = graph.python_code(
        root_module="self",
        verbose=verbose,
        include_stride=include_stride,
        include_device=include_device,
    )
    module_code = verbose_python_code.src
    submodule_names = [name for name, m in module.named_children() if hasattr(m, "graph")]
    # For higher-order functions, the callable is a submodule, and the code string initializes the object using for example`wrap_body_0 = self.wrap_body_0`.
    # Since `wrap_body_0` represents the class name of the submodule, it needs to be replaced with `wrap_body_0 = self.wrap_body_0()` to instantiate the object.
    for submodule_name in submodule_names:
        module_code = module_code.replace(f"self.{submodule_name}", f"self.{submodule_name}()")
    module_code = module_code.lstrip("\n")
    module_code = f"class {module_name}(torch.nn.Module):\n" + module_code
    module_code = _addindent(module_code, 4)

    submodule_code_list = [""]
    for submodule_name, submodule in module.named_children():
        if hasattr(submodule, "graph"):
            submodule_code_list.append(
                _readable(
                    submodule,
                    submodule_name,
                    print_output=False,
                    verbose=verbose,
                    include_stride=include_stride,
                    include_device=include_device,
                    colored=colored,
                )
            )
    submodule_code = "\n".join(submodule_code_list)
    submodule_code = _addindent(submodule_code, 4)

    output = module_code + submodule_code
    if print_output:
        print(module_code + submodule_code)
    return output


def get_env() -> tuple[str, str]:
    """Retrieve detailed environment information using `torch.utils.collect_env.get_pip_packages()`.
    Additionally, include the installed versions of Thunder and NvFuser (if available via pip).
    """

    from torch.utils.collect_env import run, get_pip_packages

    torch_env = "CUDA devices:\n"
    for i in range(torch.cuda.device_count()):
        torch_env += f"  {i}: {torch.cuda.get_device_name(i)}\n"
    torch_env += f"CUDA version: {torch.version.cuda}\n"
    _, packages = get_pip_packages(run)
    if packages is not None:
        torch_env += packages
    _, thunder_packages = get_pip_packages(run, {"lightning-thunder", "nvfuser"})
    return (
        torch_env,
        (
            thunder_packages
            if thunder_packages is not None
            else "pip list failed. Might be related to https://github.com/pytorch/pytorch/issues/144615"
        ),
    )


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


def get_split_reasons_string(subgraph_info: SubgraphInfo) -> str:
    split_reason_str = "Split Information:\n"
    if subgraph_info.split_reasons:
        num_submodules = len(subgraph_info.submodule_to_compiled_functions)
        num_thunder_submodules = len(subgraph_info.thunder_compiled_fns)
        split_reason_str += f"The original graph is split into {num_submodules} subgraphs, {num_thunder_submodules} of which are run by Thunder.\n"
        split_reason_str += f"The structure of the split module:\n{subgraph_info.split_graph_module}\n"
        split_reason_str += "Split Reasons:\n"
        for id, split_reason in enumerate(subgraph_info.split_reasons):
            split_reason_str += f"  Split Reason {id}:\n    {split_reason.info}\n"
    else:
        split_reason_str += "The original graph is not split, and is entirely run by Thunder.\n"
    return split_reason_str


def get_thunder_module_names(subgraph_info: SubgraphInfo) -> list[str]:
    thunder_module_names = []
    for node in subgraph_info.split_graph_module.graph.nodes:
        target = node.target
        if isinstance(target, str) and target.startswith("thunder_"):
            thunder_module_names.append(target)
    return thunder_module_names


def has_higher_order_operator(gm: torch.fx.GraphModule):
    for n in gm.graph.nodes:
        if isinstance(n.target, torch._ops.HigherOrderOperator):
            return True
    return False


def format_python_file(file_path: str) -> str:
    from lightning_utilities.core.imports import package_available

    if package_available("ruff"):
        import subprocess
        import sys

        # Ruff often prints warnings, progress messages, and other information that we don't need in this context.
        # Redirecting stdout and stderr to /dev/null to suppress unnecessary output.
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def get_thunder_jit_kwargs(**kwargs) -> dict:
    """
    Extracts and returns the kwargs for :func:`thunder.jit` from the given keyword arguments.

    """
    from thunder import jit

    thunder_jit_kwarg_names = inspect.getfullargspec(jit).kwonlyargs
    return {k: v for k, v in kwargs.items() if k in thunder_jit_kwarg_names}


def get_torch_compile_kwargs(**kwargs) -> dict:
    """
    Extracts and returns the kwargs for :func:`torch.compile` from the given keyword arguments.
    """
    # lightning has torch.compile wrapped in `lightning/fabric/wrappers.py`
    torch.compile = inspect.unwrap(torch.compile)
    torch_compile_kwarg_names = inspect.getfullargspec(torch.compile).kwonlyargs
    return {k: v for k, v in kwargs.items() if k in torch_compile_kwarg_names}


class ThunderAoTOptimizer:
    """
    Helper class that keeps track of profiling data used by the Ahead-of-Time (AoT) optimization process.

    This class maintains mappings between graph module IDs and their corresponding:
    - dispatch functions (dispatch_map)
    - original graph modules (id_to_gm_map)
    - profiling statistics (id_to_profile_stats)

    It also tracks whether profiling is currently active via the is_profiling flag.
    """

    def __init__(self):
        self.is_profiling = True
        self.dispatch_map: dict = {}
        self.id_to_gm_map: dict = {}
        self.id_to_profile_stats = {}


def has_symbolic_input(gm: torch.fx.GraphModule) -> bool:
    from torch._inductor.utils import is_symbolic

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for placeholder in placeholders:
        example_value = placeholder.meta.get("example_value", None)
        if example_value is not None and is_symbolic(example_value):
            return True
    return False


def default_filter(fn: Callable, cutoff: int = 2) -> set[int]:
    """
    Default filter function that selects which FX graphs to optimize based on profiling data.

    This function examines the FX graphs collected during profiling and selects only those
    that have been called at least 'cutoff' times for optimization.

    Args:
        fn: The profiling callable containing collected statistics
        cutoff: Minimum number of times a graph must be called to be selected for optimization (default: 2)

    Returns:
        A set of graph IDs that should be optimized
    """
    choosen = set()
    id_to_profile_stats = fn._tao.id_to_profile_stats
    for idx, stats in id_to_profile_stats.items():
        total_calls = sum(stats.input_meta_to_called_times.values())
        if total_calls >= cutoff:
            choosen.add(idx)

    return choosen


def get_or_create_example_inputs_from_placeholders(placeholders: list[torch.fx.Node]) -> list[torch.Tensor]:
    """
    Gets the weakref of the inputs if possible, otherwise create inputs for benchmarking
    """
    outs = []
    for p in placeholders:
        try:
            # Ref: https://github.com/pytorch/pytorch/blob/8f3d7972ad3e41ce4dcb1e9ff7bd1a3b0a671977/torch/_dynamo/variables/builder.py#L311
            input: TensorWeakRef | torch.SymInt = p.meta["grapharg"].example
            if isinstance(input, torch.SymInt):
                input = input.node.hint
        except (KeyError, AssertionError):
            # needs to create a new example input
            outs.append(_get_example_inputs_from_placeholder(p, only_metadata=False))
        else:
            outs.append(input)
    return outs


def default_optimizer(gm: torch.fx.GraphModule, stats: ProfileStats) -> Callable:
    """
    Default optimizer function that optimizes a GraphModule based on profiling statistics.

    This function:
    1. Checks if the GraphModule has symbolic inputs and raises NotImplementedError if it does
    2. Benchmarks the GraphModule with inductor, thunderfx, and eager
    3. Returns the GraphModule compiled with the fastest backend

    Args:
        gm: The GraphModule to optimize
        stats: ProfileStats object containing profiling information for the GraphModule

    Returns:
        The optimized GraphModule
    """
    from thunder.dynamo.report import FXGraphReport
    from thunder.dynamo.benchmark_utils import (
        TorchInductorSpecification,
        TorchEagerSpecification,
        ThunderCompilerOnGraphModuleSpecification,
        WallTime,
    )

    if has_symbolic_input(stats.gm):
        raise NotImplementedError("Optimizing graph module with symbolic inputs is not supported yet.")

    placeholders = [n for n in stats.gm.graph.nodes if n.op == "placeholder"]
    example_inputs_meta = [_get_example_inputs_from_placeholder(p, only_metadata=True) for p in placeholders]
    example_inputs = get_or_create_example_inputs_from_placeholders(placeholders)

    report = FXGraphReport(gm, "gm", example_inputs_meta)
    torcheager = TorchEagerSpecification()
    torchinductor = TorchInductorSpecification()
    thunder_compiler_on_gm = ThunderCompilerOnGraphModuleSpecification(nv_skip_cache=True)

    def get_compiled_fn_and_timing(report, compile_fn, timer_fn):
        try:
            compiled_fn, *measurement = report.run_benchmark(
                compile_fn,
                timer_fn,
                reset_torch_dynamo=False,
                example_inputs=example_inputs,
                measure_fwd_bwd_together=True,
            )
        except Exception as e:
            return str(e), float("inf")
        return compiled_fn, sum(m.median for m in measurement if m is not None)

    CompilerMeasurement = namedtuple("CompilerMeasurement", ["name", "compiled_fn", "time"])
    compiled_gm_to_measurement = []
    compiled_gm_to_measurement.append(
        CompilerMeasurement("thunderfx", *get_compiled_fn_and_timing(report, thunder_compiler_on_gm, WallTime))
    )
    compiled_gm_to_measurement.append(
        CompilerMeasurement("torchinductor", *get_compiled_fn_and_timing(report, torchinductor, WallTime))
    )
    compiled_gm_to_measurement.append(
        CompilerMeasurement("torcheager", *get_compiled_fn_and_timing(report, torcheager, WallTime))
    )

    sorted_compiled_gm_to_measurement = sorted(compiled_gm_to_measurement, key=lambda x: x.time)
    if sorted_compiled_gm_to_measurement[0].time == float("inf"):
        err_msg = ", ".join([f"{x.name} raised exception: {x.compiled_fn}" for x in sorted_compiled_gm_to_measurement])
        raise RuntimeError(f"No compiler was able to compile the graph module, {err_msg}")
    return sorted_compiled_gm_to_measurement[0].compiled_fn
