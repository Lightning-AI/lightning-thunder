from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import ast
import inspect

from torch import TensorType

from thunder.core import baseutils
from thunder.core.symbol import Symbol
from thunder.core.prims import OpTags
from thunder.core.transforms import VJPDual
from thunder.core.transforms import register_augmented_forward
from thunder.core.transforms import register_backward
from thunder.core.pytree import tree_flatten

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any
    from numbers import Number
    from torch import Argument, FunctionSchema, Tensor
    from torch.autograd.function import FunctionCtx
    from torch._library.custom_ops import CustomOpDef
    from torch._ops import OpOverload
    from torch._ops import OpOverloadPacket
    from thunder.core.proxies import TensorProxy


__all__ = [
    "_register_custom_op",
]


_BACKWARD_FN: str = "_backward_fn"
_SETUP_CONTEXT_FN: str = "_setup_context_fn"


@dataclass(frozen=True)
class SavedIndices:
    """Information about what indices need to be saved for backward pass"""

    # Which input indices to save: [0, 2] means save inputs[0], inputs[2]
    input_indices: list[int]
    # Which output indices to save: [1] means save outputs[1]
    output_indices: list[int]
    # Non-tensor data saved on ctx: {'multiplier': 2.0}
    scalar_data: dict[str, Any]

    def __str__(self) -> str:
        parts = []
        if self.input_indices:
            parts.append(f"inputs{self.input_indices}")
        if self.output_indices:
            parts.append(f"outputs{self.output_indices}")
        if self.scalar_data:
            parts.append(f"scalars={self.scalar_data}")
        return "SavedIndices(" + ", ".join(parts) + ")"

    @property
    def num_tensors_saved_for_backward(self) -> int:
        return len(self.input_indices) + len(self.output_indices)


class SetupContextAnalyzer(ast.NodeVisitor):
    """Lightweight AST analyzer focused on extracting save indices"""

    def __init__(self):
        # var_name -> ('input'|'output', index)
        self.var_to_source: dict[str, int] = {}
        # List of input indices that get saved
        self.saved_inputs: list[int] = []
        # List of output indices that get saved
        self.saved_outputs = []
        # Scalar data saved on ctx
        self.scalar_data: dict[str, Number] = {}
        self.in_function = False

    def visit_FunctionDef(self, node):
        """Track when we're inside the setup_context function"""
        if any(name in node.name for name in ["setup_context", "_setup_context"]):
            self.in_function = True

            # Map parameter names to their meanings
            params = [arg.arg for arg in node.args.args]
            if len(params) >= 3:
                inputs_param = params[1]  # Usually 'inputs'
                output_param = params[-1]  # Usually 'output' (last param)
                self.var_to_source[inputs_param] = ("inputs", None)
                self.var_to_source[output_param] = ("outputs", None)

            self.generic_visit(node)
            self.in_function = False
        else:
            self.generic_visit(node)

    def visit_Assign(self, node):
        """Track variable assignments and ctx attribute assignments"""
        if not self.in_function:
            return

        # Handle tuple unpacking: a, b, c = inputs
        if (
            isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.var_to_source
        ):
            source_type, _ = self.var_to_source[node.value.id]
            for i, target in enumerate(node.targets[0].elts):
                if isinstance(target, ast.Name):
                    # 'inputs' -> 'input'
                    self.var_to_source[target.id] = (source_type.rstrip("s"), i)

        # Handle single indexing: x = inputs[0]
        elif (
            isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.value.slice, ast.Constant)
        ):
            var_name = node.targets[0].id
            source_name = node.value.value.id
            index = node.value.slice.value

            if source_name in self.var_to_source:
                source_type, _ = self.var_to_source[source_name]
                self.var_to_source[var_name] = (source_type.rstrip("s"), index)

        # Handle ctx.attr = value
        elif (
            isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "ctx"
        ):
            attr_name = node.targets[0].attr
            if attr_name != "save_for_backward":
                value = self._extract_value(node.value)
                self.scalar_data[attr_name] = value

        self.generic_visit(node)

    def visit_Call(self, node):
        """Track ctx.save_for_backward calls"""
        if not self.in_function:
            return

        # Look for ctx.save_for_backward(...)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ctx"
            and node.func.attr == "save_for_backward"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in self.var_to_source:
                    source_type, index = self.var_to_source[arg.id]
                    if source_type == "input":
                        self.saved_inputs.append(index)
                    elif source_type == "output":
                        self.saved_outputs.append(index)

                elif isinstance(arg, ast.Subscript):
                    # Direct indexing in save call: ctx.save_for_backward(inputs[0])
                    source_info = self._analyze_subscript(arg)
                    if source_info:
                        source_type, index = source_info
                        if source_type == "input":
                            self.saved_inputs.append(index)
                        elif source_type == "output":
                            self.saved_outputs.append(index)

        self.generic_visit(node)

    def _analyze_subscript(self, node):
        """Analyze subscript like inputs[0] or output[1]"""
        if (
            isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and node.value.id in self.var_to_source
        ):
            source_type, _ = self.var_to_source[node.value.id]
            index = node.slice.value
            # 'inputs' -> 'input'
            return (source_type.rstrip("s"), index)
        return None

    def _extract_value(self, node):
        """Extract simple values from AST nodes"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return f"<var:{node.id}>"
        else:
            return f"<expr:{ast.unparse(node)}>"


def extract_saved_indices(
    setup_context_fn: Callable[[FunctionCtx, tuple[Tensor, ...], tuple[Tensor, ...]], None],
) -> SavedIndices:
    """
    Extract which input/output indices need to be saved for the augmented forward function.

    Args:
        setup_context_fn: setup context function that might call `ctx.save_for_backward`.

    Returns:
        SavedIndices object containing:
        - input_indices: List of input indices to save (e.g., [0, 2])
        - output_indices: List of output indices to save (e.g., [1])
        - scalar_data: Dict of scalar data to save (e.g., {'multiplier': 2.0})

    Example:
        >>> @torch.library.custom_op("mylib::mul", mutates_args=())
        >>> def mul(a, b):
        ...     return a * b
        >>> def setup_context(ctx, inputs, output):
        >>>     a, b = inputs
        >>>     ctx.save_for_backward(a, b)
        >>>
        >>> torch.library.register_autograd(
        ...     "mylib::mul", backward, setup_context=setup_context
        ... )
        >>> indices = extract_saved_indices(mul)
        >>> print(indices)  # SavedIndices(inputs=[0, 1])
    """

    # Get source code and parse AST
    try:
        source = inspect.getsource(setup_context_fn)
        tree = ast.parse(source)
    except (OSError, TypeError) as e:
        raise RuntimeError(f"Could not extract source code from setup_context function: {e}")

    # Analyze the AST
    analyzer = SetupContextAnalyzer()
    analyzer.visit(tree)

    return SavedIndices(
        input_indices=analyzer.saved_inputs,
        output_indices=analyzer.saved_outputs,
        scalar_data=analyzer.scalar_data,
    )


def _convert_to_meta_function(func):
    from thunder.core.langctxs import langctx, Languages
    from thunder.torch import meta_adaptor

    return langctx(Languages.TORCH)(meta_adaptor(func))


def _get_meta_function_from(custom_op: CustomOpDef) -> Callable[[Any], TensorProxy | tuple[TensorProxy, ...]]:
    return _convert_to_meta_function(custom_op._abstract_fn)


def _has_autograd_def(custom_op: CustomOpDef) -> bool:
    return (
        getattr(custom_op, _BACKWARD_FN, None) is not None and getattr(custom_op, _SETUP_CONTEXT_FN, None) is not None
    )


def create_augmented_forward_for_custom_op(
    symbol: Symbol,
    saved_indices: SavedIndices,
) -> Callable:
    """Creates a Thunder-compatible augmented forward function for a custom op."""
    input_indices: list[int] = saved_indices.input_indices
    output_indices: list[int] = saved_indices.output_indices

    def augmented_forward(*args, **kwargs):
        fwd_result = symbol(*args, **kwargs)

        flat_inputs, _ = tree_flatten((args, kwargs))
        flat_outputs, _ = tree_flatten(fwd_result)

        saved_for_backward: list[TensorProxy] = []
        for idx in input_indices:
            saved_for_backward.append(flat_inputs[idx])
        for idx in output_indices:
            saved_for_backward.append(flat_outputs[idx])
        return VJPDual(fwd_result, tuple(saved_for_backward))

    return augmented_forward


def define_backward_for(
    custom_op: CustomOpDef,
    num_saved_tensors: int,
    tensor_indices: tuple[int, ...],
) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    import torch

    backward_fn = getattr(custom_op, _BACKWARD_FN)

    def wrapped_backward(*args):
        # NOTE: It'd be possible to use lighter objects instead of `FunctionCtx`.
        ctx = torch.autograd.function.FunctionCtx()
        setattr(ctx, "saved_tensors", tuple(args[:num_saved_tensors]))
        backward_results = baseutils.sequencify(backward_fn(ctx, *args[num_saved_tensors:]))
        grad_results = [backward_results[i] for i in tensor_indices]
        return grad_results

    return _convert_to_meta_function(wrapped_backward), wrapped_backward


def _register_custom_op(custom_op: CustomOpDef) -> Symbol:
    """Register :func:`~torch.library.custom_op`'ed function to Thunder.

    :func:`torch.library.custom_op` operators always have schema as per https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.aotf6sastysc saying
    "Use torch.library.custom_op to decorate a function to turn it into a custom operator.
    The function must be decorated with type annotations, and you must correctly annotate inputs that are being mutated."
    An example schema is ``my_lib::foo(Tensor a, Tensor b, float? c=None, str d="") -> Tensor``.

    This function does three things:
    1. Register ``custom_op`` as a new :class:`~thunder.core.symbol.Symbol` whose ``fn`` is ``custom_op._opoverload`` and ``meta`` is based on ``custom_op._abstract_fn``
    2. When ``_setup_context_fn`` and ``_backward_fn`` are defined, register augmented forward and backward through :func:`thunder.core.transforms.register_augmented_forward` and :func:`thunder.core.transforms.register_backward`.

    Args:
        custom_op: :func:`torch.library.custom_op`'ed function. This is not ``torch.ops.{namespace}.{name}``.

    Returns:
        :class:`~thunder.core.symbol.Symbol`: A symbol representing the input ``custom_op``.
    """
    from thunder.executors.torchex import _always_executable
    from thunder.executors.custom_op_ex import custom_op_ex
    from thunder.torch import register_function

    # `custom_op` is `custom_op(name)(my_func)`,
    # `torch.ops.namespace.name` is `OpOverloadPacket.`
    # e.g. `torch.ops.my_lib.foo` is OpOverloadPacket and `torch.ops.my_lib.foo.default` is OpOverload.
    torch_opoverload: OpOverload = custom_op._opoverload
    torch_opoverload_packet: OpOverloadPacket = torch_opoverload._overloadpacket

    schema: FunctionSchema = torch_opoverload._schema
    schema_arguments: list[Argument] = schema.arguments
    tensor_indices: tuple[int] = tuple(i for i, arg in enumerate(schema_arguments) if isinstance(arg.type, TensorType))
    tensor_arity: int = len(tensor_indices)
    baseutils.check(tensor_arity > 0, lambda: f"arity of {custom_op._qualname} should be greater than 0: {schema}")
    schema_returns: list[Argument] = schema.returns
    return_arity: int = len(schema_returns)
    tensor_return_arity: int = len(list(filter(lambda a: isinstance(a.type, TensorType), schema_returns)))
    baseutils.check(return_arity == tensor_return_arity, lambda: f"Return values include non-Tensor values: {schema}")

    has_autograd_def = _has_autograd_def(custom_op)
    fn_name = custom_op._qualname.replace("::", "_")
    op_id = f"torch::ops::{custom_op._qualname}".replace("::", ".")
    meta_fn = _get_meta_function_from(custom_op)
    # NOTE: Especially when this `custom_op` doesn't have backward, and the caller program
    # involves parameter lifting, somehow the bsym of this custom_op seems to be removed
    # by `thunder/transforms/autodiff.py`'s `AugmentedForwardProcessor` of `grad_transform_on_trace`
    # So this tag marks the bsyms so that the processor does't see them as "constant" for VJP.
    tags: tuple[OpTags, ...]
    if not has_autograd_def:
        tags = (OpTags.TORCH_COMPILE_COMPLIANT_CUSTOM_OP,)
    else:
        tags = ()
    symbol = Symbol(
        name=fn_name,
        meta=meta_fn,
        id=op_id,
        is_prim=False,
        tags=tags,
    )
    # Register both `torch.ops.my_lib.foo` and `torch.ops.my_lib.foo.default`.
    register_function(torch_opoverload_packet, symbol)
    register_function(torch_opoverload, symbol)

    op: Symbol = custom_op_ex.register_operator(fn_name, meta=meta_fn, fn=torch_opoverload)
    custom_op_ex.register_implementation(symbol, op, checker=_always_executable)
    # TODO: Think about using `grad_transform` instead.
    if has_autograd_def:
        setup_context_fn = getattr(custom_op, _SETUP_CONTEXT_FN)
        saved_indices = extract_saved_indices(setup_context_fn)
        register_augmented_forward(symbol.id)(create_augmented_forward_for_custom_op(symbol, saved_indices))
        num_saved_tensors: int = saved_indices.num_tensors_saved_for_backward
        bwd_fn_name = f"{fn_name}_backward"
        backward_meta, backward_impl = define_backward_for(custom_op, num_saved_tensors, tensor_indices)
        backward_op = custom_op_ex.register_operator(bwd_fn_name, meta=backward_meta, fn=backward_impl)
        register_backward(symbol.id)(backward_op)

    return symbol
