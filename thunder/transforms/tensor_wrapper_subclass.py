from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from numbers import Number
from typing import TYPE_CHECKING, NamedTuple
import time
import warnings

import torch
from torch.fx import Node
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from thunder.core.baseutils import run_once
from thunder.core.codeutils import SigInfo
from thunder.core import devices
from thunder.core import dtypes
from thunder.core import prims
from thunder.core import utils
from thunder.core.proxies import ProxyInterface
from thunder.core.proxies import SubclassTensorProxy
from thunder.core.proxies import TensorProxy
from thunder.core.proxies import Variable
from thunder.core.proxies import variableify
from thunder.core.pytree import tree_flatten
from thunder.core.pytree import tree_map
from thunder.core.pytree import tree_unflatten
from thunder.core.trace import TraceCtx
from thunder.core.trace import TraceProvenance
from thunder.core.trace import from_trace
from thunder.core.trace import tracectx

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from optree import PyTreeSpec
    from torch.fx import GraphModule
    from torch._ops import OpOverload
    from thunder.core.symbol import BoundSymbol


__all__ = [
    "unroll_tensor_subclasses",
]


PLACEHOLDER: str = "placeholder"
CALL_FUNCTION: str = "call_function"
OUTPUT: str = "output"


@run_once
def warn_tensor_subclass_support() -> None:
    warnings.warn("Tensor Subclasses with `__torch_dispatch__` defined support is experimental")


class OutputWrapperForFxTracing(NamedTuple):
    inner_tensors: dict[str, torch.Tensor] | torch.Tensor
    metadata: dict[str, Any] | None


def _materialize_tensor_proxy(t: TensorProxy, fake_tensor_mode: FakeTensorMode | None) -> torch.Tensor:
    shape = t.shape
    device = devices.to_torch_device(t.device)
    dtype = dtypes.to_torch_dtype(t.dtype)
    requires_grad = t.requires_grad

    with torch.device("meta"):
        t = torch.empty(shape, dtype=dtype, requires_grad=requires_grad)
    if fake_tensor_mode is None:
        return t
    fakified_empty_tensor = fake_tensor_mode.fake_tensor_converter.from_meta_and_device(
        fake_mode=fake_tensor_mode, t=t, device=device
    )
    return fakified_empty_tensor


def _make_fake_subclass_tensor_from_subclass_tensor_proxy(
    tensor_proxy: SubclassTensorProxy,
    fake_tensor_mode: FakeTensorMode,
) -> torch.Tensor:
    utils.check(
        (subclass_type := getattr(tensor_proxy, SubclassTensorProxy.SUBCLASS_TYPE_ATTR, None)) is not None,
        lambda: f"{tensor_proxy} does not have `{SubclassTensorProxy.SUBCLASS_TYPE_ATTR}`",
    )
    utils.check(
        tensor_proxy._tensors,
        lambda: f"{tensor_proxy} has an empty `{tensor_proxy._tensors=}`",
    )
    tensor_attr_names = tensor_proxy._tensor_attr_names
    non_tensor_attr_names = tensor_proxy._non_tensor_attr_names
    inner_tensors = dict(
        zip(
            tensor_attr_names,
            [_materialize_tensor_proxy(t, fake_tensor_mode=fake_tensor_mode) for t in tensor_proxy._tensors],
        )
    )
    new_non_tensors = []
    for a in tensor_proxy._non_tensors:
        if isinstance(a, dtypes.dtype):
            new_non_tensors.append(dtypes.to_torch_dtype(a))
        elif isinstance(a, devices.Device):
            new_non_tensors.append(devices.to_torch_device(a))
        else:
            new_non_tensors.append(a)
    metadata = dict(zip(non_tensor_attr_names, new_non_tensors))
    subclass_tensor = subclass_type.__tensor_unflatten__(
        inner_tensors,
        metadata,
        outer_size=-1,
        outer_stride=-1,
    )
    fakified = fake_tensor_mode.from_tensor(subclass_tensor, static_shapes=True)
    return fakified


def materialize_tensor_proxy(
    t: TensorProxy | SubclassTensorProxy,
    fake_tensor_mode: FakeTensorMode,
) -> torch.Tensor:
    if isinstance(t, SubclassTensorProxy):
        return _make_fake_subclass_tensor_from_subclass_tensor_proxy(t, fake_tensor_mode)
    return _materialize_tensor_proxy(t, fake_tensor_mode)


def maybe_materialize_tensor(
    t: ProxyInterface,
    fake_tensor_mode: FakeTensorMode,
) -> ProxyInterface | torch.Tensor:
    if isinstance(t, (TensorProxy, SubclassTensorProxy)):
        return materialize_tensor_proxy(t, fake_tensor_mode)
    if isinstance(t, (Number, str)):
        return t
    return t.value


def proxy_fake_tensor(t: torch.Tensor | FakeTensor) -> ProxyInterface:
    if isinstance(t, FakeTensor) or (isinstance(t, torch.Tensor) and not issubclass(type(t), torch.Tensor)):
        return TensorProxy(
            None,
            shape=list(t.shape),
            dtype=dtypes.to_dtype(t.dtype),
            device=devices.to_device(t.device),
            requires_grad=t.requires_grad,
        )
    if torch.utils._python_dispatch.is_traceable_wrapper_subclass(t):
        tensor_attr_names, metadata = t.__tensor_flatten__()
        tensor_proxies = [proxy_fake_tensor(getattr(t, name)) for name in tensor_attr_names]
        non_tensor_attr_names = list(metadata.keys())
        non_tensors = list(metadata.values())
        p = SubclassTensorProxy(
            None,
            shape=list(t.shape),
            dtype=dtypes.to_dtype(t.dtype),
            device=devices.to_device(t.device),
            requires_grad=t.requires_grad,
            tensors=tensor_proxies,
            non_tensors=non_tensors,
            subclass_type=type(t),
        )
        p._tensor_attr_names = tensor_attr_names
        p._non_tensor_attr_names = non_tensor_attr_names
        for name, value in zip(tensor_attr_names + non_tensor_attr_names, tensor_proxies + non_tensors):
            setattr(p, name, value)
        return p
    return t


def trace_from_bsym_or_bsyms(bsym_or_bsyms: BoundSymbol | Sequence[BoundSymbol]) -> TraceCtx:
    from thunder.core.compile_data import get_compile_data

    cd = get_compile_data()
    temporary_executor = None
    if cd is not None:
        from thunder.extend import TemporaryExecutor

        executors_list = list(filter(lambda t: isinstance(t, TemporaryExecutor), cd.executors_list))
        if executors_list:
            temporary_executor = executors_list[0]

    bsyms = list(utils.sequencify(bsym_or_bsyms))
    trace_args = bsyms[0].flat_proxy_args
    trace_name = bsyms[0].sym.name

    if temporary_executor is not None and temporary_executor._implmap:
        tmp_bsyms = []
        for bsym in bsyms:
            if temporary_executor.can_execute(bsym) and bsym.subsymbols:
                tmp_bsyms.extend(bsym.subsymbols)
            else:
                tmp_bsyms.append(bsym)
        bsyms = tmp_bsyms
    unpack_bsyms = [
        prims.unpack_trivial.bind(a, name=a.name, output=a)
        for a in filter(lambda a: isinstance(a, ProxyInterface), trace_args)
    ]

    trace = TraceCtx()
    trace.bound_symbols.extend(unpack_bsyms + bsyms)
    trace.args = trace_args
    with tracectx(trace):
        prims.python_return(bsyms[-1].output)
    with tracectx(trace):
        # note(crcrpar): Give prefix `tmp` to avoid infinite recursion due to the same name
        trace._siginfo = SigInfo.from_name_and_args(f"tmp_{trace_name}", trace.args)
    return trace


def make_trace_executable(trace_to_convert: TraceCtx, *args_for_eval, **kwargs_for_eval):
    from functools import wraps
    from thunder import trace
    from thunder.core.transforms import eval_trace
    from thunder.executors.torch_compile import to_torch_translator

    @wraps(trace_to_convert.python_callable())
    def torch_interpreted_func(*args, **kwargs):
        return eval_trace(trace_to_convert, *args, **kwargs, symbol_mapper=to_torch_translator)

    torch_trace = trace(inline_trace=False)(torch_interpreted_func, *args_for_eval, **kwargs_for_eval)
    return torch_trace


@dataclass
class DesugarTensorSubclass:
    """Transforms tensor subclass operations into their underlying tensor operations.

    This class handles the desugaring of tensor subclass operations by:
    1. Identifying tensor subclass proxies that need to be flattened
    2. Converting bound symbols involving tensor subclasses to FX graphs
    3. Translating FX graphs back to Thunder bound symbols
    4. Managing the mapping between original and desugared operations

    Attributes:
        computation_trace: The trace context being processed
        swap_map: Maps variables to their corresponding proxy interfaces
        fake_tensor_mode: Mode for creating fake tensors during tracing
        flat_trace_args: Flattened arguments of the trace
        subclass_proxy_to_flatten: Set of tensor subclass proxies to be flattened
        bsym_to_new_outputs: Maps bound symbols to their new tensor proxy outputs
        fx_graphs:
            List of pairs of :class:`~thunder.core.symbol.BoundSymbol` and generated FX graph module
        proxy_to_strides:
            Maps :class:`~thunder.core.proxies.TensorProxy` and :class:`~thunder.core.proxies.SubclassTensorProxy`
            to strides of corresponding :class:`~torch._subclasses.fake_tensor.FakeTensor`s
            that are acquired through :mod:`torch.fx`.
        queried_proxies_of_strides: Track proxies whose strides are utilized in their materialization.
        saved_tensors_for_backward:
            Saved :class:`~thunder.core.proxies.TensorProxy`s and :class:`~thunder.core.proxies.SubclassTensorProxy`s
            that are gotten from updated :class:`~thunder.core.symbol.BoundSymbol` of return.
    """

    computation_trace: TraceCtx
    swap_map: dict[Variable, ProxyInterface] = field(init=False, default_factory=dict)
    fake_tensor_mode: FakeTensorMode = field(init=False, default_factory=FakeTensorMode)
    flat_trace_args: Sequence[ProxyInterface] = field(init=False, default=None)
    subclass_proxy_to_flatten: set[Variable] = field(init=False, default_factory=set)
    bsym_to_new_outputs: dict[BoundSymbol, list[TensorProxy]] = field(init=False, default_factory=dict)
    fx_graphs: list[tuple[BoundSymbol, GraphModule]] = field(init=False, default_factory=list)
    proxy_to_strides: dict[Variable, tuple[int, ...] | list[tuple[int, ...]]] = field(init=False, default_factory=dict)
    queried_proxies_of_strides: list[str] = field(init=False, default_factory=list)
    saved_tensors_for_backward: list[TensorProxy | SubclassTensorProxy | Any] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.flat_trace_args = self._get_flat_trace_args()
        self._identify_subclass_proxies_to_flatten()

    def _get_flat_trace_args(self) -> Sequence[ProxyInterface]:
        """Determine and flatten the trace arguments based on whether this is a backward trace."""
        if self._is_backward_trace():
            maybe_unpack_C0_bsym = self.computation_trace.bound_symbols[4]
            maybe_unpack_C1_bsym = self.computation_trace.bound_symbols[5]
            flat_args, _ = tree_flatten((maybe_unpack_C0_bsym.output, maybe_unpack_C1_bsym.output))
            return flat_args

        flat_args, _ = tree_flatten((self.computation_trace.args, self.computation_trace.kwargs))
        return flat_args

    def _is_backward_trace(self) -> bool:
        """Check if the current trace is a backward trace by examining bound symbols."""
        if len(self.computation_trace.bound_symbols) <= 6:
            return False

        maybe_unpack_C0_bsym = self.computation_trace.bound_symbols[4]
        maybe_unpack_C1_bsym = self.computation_trace.bound_symbols[5]

        return (
            maybe_unpack_C0_bsym.args
            and maybe_unpack_C1_bsym.args
            and (
                maybe_unpack_C0_bsym.sym.id,
                maybe_unpack_C1_bsym.sym.id,
                getattr(maybe_unpack_C0_bsym.args[0], "name", ""),
                getattr(maybe_unpack_C1_bsym.args[0], "name", ""),
            )
            == (
                prims.PrimIDs.UNPACK_SEQUENCE,
                prims.PrimIDs.UNPACK_SEQUENCE,
                "C0",
                "C1",
            )
        )

    def _is_augmented_forward_trace(self) -> bool:
        if self._is_backward_trace():
            return False
        fwd_return_bsyms = [
            bsym for bsym in self.computation_trace.bound_symbols if bsym.sym.id == prims.PrimIDs.RETURN
        ]
        utils.check(
            len(fwd_return_bsyms) == 1, lambda: f"{len(fwd_return_bsyms)} bsyms of {prims.PrimIDs.RETURN} found"
        )
        fwd_return_bsym: BoundSymbol = fwd_return_bsyms[0]
        args = fwd_return_bsym.args
        return isinstance(args, tuple) and len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], tuple)

    def _get_flat_saved_tensors_for_backward(
        self, bsym: BoundSymbol | None = None
    ) -> Sequence[TensorProxy | SubclassTensorProxy | Any]:
        if bsym is None:
            fwd_return_bsyms = [
                bsym for bsym in self.computation_trace.bound_symbols if bsym.sym.id == prims.PrimIDs.RETURN
            ]
            fwd_return_bsym: BoundSymbol = fwd_return_bsyms[0]
        else:
            fwd_return_bsym = bsym
        args = fwd_return_bsym.args
        return tree_flatten(args[1])[0]

    def _identify_subclass_proxies_to_flatten(self) -> None:
        """Identify SubclassTensorProxy instances that need to be flattened."""
        for arg in self.flat_trace_args:
            if isinstance(arg, SubclassTensorProxy):
                self.subclass_proxy_to_flatten.add(variableify(arg))

    def _get_tensor_attr_names(self, p: SubclassTensorProxy) -> list[str]:
        """Get the names of tensor attributes from a SubclassTensorProxy."""
        return p._tensor_attr_names

    def _get_non_tensor_attr_names(self, p: SubclassTensorProxy) -> list[str]:
        """Get the names of non-tensor attributes from a SubclassTensorProxy."""
        return p._non_tensor_attr_names

    def translate_fx_graph_into_bsym(
        self,
        bsym: BoundSymbol,
        fx_graph: GraphModule,
    ) -> list[BoundSymbol]:
        """Translate an FX graph into Thunder bound symbols.

        Args:
            bsym: The original bound symbol
            fx_graph: The FX graph to translate

        Returns:
            The translated bound symbol(s)

        This method converts operations in the FX graph to Thunder bound symbols,
        handling tensor subclass flattening and unflattening as needed.
        """
        from thunder.torch import _torch_to_thunder_function_map

        self.fx_graphs.append((bsym, fx_graph))

        unwrapped_bsym_args: dict[int, ProxyInterface] = {}
        list_of_flattening_bsyms: list[BoundSymbol] = []
        for a in bsym.flat_args:
            if isinstance(a, SubclassTensorProxy):
                if variableify(a) in self.subclass_proxy_to_flatten:
                    self.computation_trace.push_scope([])
                    with tracectx(self.computation_trace):
                        prims.flatten_tensor_subclass(a)
                    flattening_bsym = self.computation_trace.pop_scope()[0]
                    list_of_flattening_bsyms.append(flattening_bsym)
                tensor_attr_names = self._get_tensor_attr_names(a)
                tensors = a._tensors

                non_tensor_attr_names = self._get_non_tensor_attr_names(a)
                non_tensors = a._non_tensors
                metadata = dict(zip(non_tensor_attr_names, non_tensors))
                for name, t in zip(tensor_attr_names, tensors):
                    utils.check(
                        isinstance(t, TensorProxy),
                        lambda: f"{a=}, {tensor_attr_names = }, {tensors=}",
                    )
                    unwrapped_bsym_args[len(unwrapped_bsym_args)] = t
                # TODO(crcrpar): Think about how to verify the correctness of this flattening
                flat_metadata, _ = tree_flatten(metadata)
                for v in flat_metadata:
                    unwrapped_bsym_args[len(unwrapped_bsym_args)] = v
            else:
                if not isinstance(a, ProxyInterface):
                    from thunder.core.proxies import proxy

                    with tracectx(self.computation_trace):
                        a = proxy(a)
                unwrapped_bsym_args[len(unwrapped_bsym_args)] = a

        node: Node
        list_of_placeholder_node: list[Node] = []
        list_of_function_call_node: list[Node] = []
        node_of_output: Node
        for node in fx_graph.graph.nodes:
            if node.op == PLACEHOLDER:
                list_of_placeholder_node.append(node)
            if node.op == CALL_FUNCTION:
                list_of_function_call_node.append(node)
            if node.op == OUTPUT:
                node_of_output = node
        args = [n.target for n in list_of_placeholder_node]
        arg_name_to_index = {a: i for i, a in enumerate(args)}
        ltorch_ops_for_node_of_ops = []
        for node in list_of_function_call_node:
            op: OpOverload = node.target
            if op not in _torch_to_thunder_function_map:
                msg = (
                    f"`thunder.torch` does not have corresponding op for {op}. "
                    "Think about adding it to thunder/torch/default_torch_ops.py"
                    f"\nThe op is found while flattening the following BoundSymbol:\n{bsym}"
                    f"\ntorch.fx graph:\n{fx_graph.print_readable(print_output=False)}"
                )
                raise RuntimeError(msg)
            ltorch_ops_for_node_of_ops.append(_torch_to_thunder_function_map[op])

        bsyms: list[BoundSymbol] = []
        if list_of_flattening_bsyms:
            bsyms.extend(list_of_flattening_bsyms)

        # Define arg_mapper outside the loop
        def arg_mapper(arg, current_node):
            if isinstance(arg, Node):  # Fixed: check for Node type, not node variable
                if isinstance(arg.target, str):
                    return unwrapped_bsym_args[arg_name_to_index[arg.target]]
                else:
                    return fxnode_output_name_to_tensor_proxy[str(arg)]
            elif isinstance(arg, immutable_dict):
                return dict(arg)
            elif isinstance(arg, immutable_list):
                return list(arg)
            else:
                return arg

        fxnode_output_name_to_tensor_proxy: dict[str, OpOverload] = {}
        for node, ltorch_op in zip(list_of_function_call_node, ltorch_ops_for_node_of_ops):
            args: list[Node] = node.args

            # Use a lambda to bind the current node to arg_mapper
            current_mapper = lambda arg: arg_mapper(arg, node)

            arg_proxies = tree_map(current_mapper, args)
            kwargs_to_ltorch = tree_map(current_mapper, node.kwargs)
            self.computation_trace.push_scope([])

            try:
                with tracectx(self.computation_trace):
                    out = ltorch_op(*arg_proxies, **kwargs_to_ltorch)
            except Exception as e:
                msg = (
                    f"Failing to map `torch.{node}` to `thunder.torch` op of "
                    f"{ltorch_op} with args of {arg_proxies} and kwargs of {kwargs_to_ltorch}\n"
                    f"BoundSymbol in question is\n```python\n{bsym}\n```\n"
                    f"Corresponding torch.fx Graph is\n```python\n{fx_graph.print_readable(print_output=False)}\n```\n"
                    f"Original error is {e}"
                )
                raise type(e)(msg)
            else:
                fxnode_output_name_to_tensor_proxy[str(node)] = out
                bsyms.extend(self.computation_trace.pop_scope())

        if len(bsyms) == 0:
            return [bsym]

        orig_output = bsym.flat_outs[0]
        if is_subclass_ctor_bsym := bsym.sym.id == prims.PrimIDs.TENSOR_SUBCLASS_CTOR:
            utils.check_type(orig_output, SubclassTensorProxy)
        if isinstance(orig_output, SubclassTensorProxy):
            # note(crcrpar): args[0] would be list of tensors, and args[1] could be list of non-tensors.
            args: list[Node] = node_of_output.args[0]
            new_tensor_proxies = []
            new_non_tensor_values = []
            for a in args:
                value = a
                if isinstance(a, Node):
                    if isinstance(a.target, str):
                        value = unwrapped_bsym_args[arg_name_to_index[a.target]]
                    else:
                        value = fxnode_output_name_to_tensor_proxy[str(a)]
                if isinstance(value, TensorProxy):
                    new_tensor_proxies.append(value)
                elif isinstance(value, (immutable_dict, immutable_list)):
                    if isinstance(value, immutable_dict):
                        new_non_tensor_values.append(dict(value))
                    else:
                        new_non_tensor_values.append(list(value))
                else:
                    new_non_tensor_values.append(value)
            utils.check(
                len(orig_output._tensors) == len(new_tensor_proxies),
                lambda: (
                    f"The number of new tensor proxies for {orig_output=} does not match: "
                    f"{len(new_tensor_proxies)=} != {len(orig_output._tensors)=}"
                ),
            )
            with tracectx(self.computation_trace):
                new_subclass = orig_output.replace()
            new_subclass._tensors = new_tensor_proxies
            for name, value in zip(new_subclass._tensor_attr_names, new_tensor_proxies):
                setattr(new_subclass, name, value)
            bsyms.append(
                prims.unflatten_tensor_subclass.bind(
                    new_subclass._subclass_type,
                    dict(zip(new_subclass._tensor_attr_names, new_tensor_proxies)),
                    dict(zip(new_subclass._non_tensor_attr_names, new_subclass._non_tensors)),
                    output=new_subclass,
                )
            )

            self.swap_map[variableify(orig_output)] = new_subclass
            self.subclass_proxy_to_flatten.add(variableify(new_subclass))

        else:
            non_none_args = [n for n in node_of_output.args[0] if n is not None]
            utils.check(len(non_none_args) == 1, lambda: f"{node_of_output.args = }")
            new_out_node = non_none_args[0]
            self.swap_map[variableify(orig_output)] = fxnode_output_name_to_tensor_proxy[str(new_out_node)]

        args = ", ".join([t.name if isinstance(t, ProxyInterface) else f"{t}" for t in bsym.flat_args])
        header = f"{bsym.sym.id}({args})"
        for i, sbsym in enumerate(bsyms, 1):
            sbsym.header = f"[{i}/{len(bsyms)}] unrolled `__torch_dispatch__` of `{header}`"
        return bsyms

    def _materialize_proxy_for_fx(self, proxy: ProxyInterface | Number | str):
        if isinstance(proxy, SubclassTensorProxy):
            fake_tensor_subclass = _make_fake_subclass_tensor_from_subclass_tensor_proxy(proxy, self.fake_tensor_mode)
            if (var_proxy := variableify(proxy)) in self.proxy_to_strides:
                tensor_attr_names, metadata = fake_tensor_subclass.__tensor_flatten__()
                strides = self.proxy_to_strides[var_proxy]
                utils.check(len(tensor_attr_names) == len(strides), lambda: f"{tensor_attr_names = }, {strides = }")
                inner_tensors: dict[str, FakeTensor] = {}
                for name, stride in zip(tensor_attr_names, strides):
                    t = getattr(fake_tensor_subclass, name)
                    inner_tensors[name] = torch.as_strided(t, t.size(), stride)
                self.queried_proxies_of_strides.append(var_proxy.proxy.name)
                return fake_tensor_subclass.__class__.__tensor_unflatten__(
                    inner_tensors, metadata, outer_size=-1, outer_stride=-1
                )
            return fake_tensor_subclass
        elif isinstance(proxy, TensorProxy):
            fake_tensor = _materialize_tensor_proxy(proxy, self.fake_tensor_mode)
            if (var_proxy := variableify(proxy)) in self.proxy_to_strides:
                stride = self.proxy_to_strides[var_proxy]
                self.queried_proxies_of_strides.append(var_proxy.proxy.name)
                return torch.as_strided(fake_tensor, fake_tensor.size(), stride)
            return fake_tensor
        elif isinstance(proxy, (Number, str)):
            return proxy
        else:
            return proxy.value

    def _materialize_trace_args_for_fx(self, trace: TraceCtx):
        return tree_map(
            self._materialize_proxy_for_fx,
            trace.args,
        )

    def convert_trace_to_fx_graph_and_get_fake_result(
        self,
        trace: TraceCtx,
    ) -> tuple[GraphModule, tuple[OutputWrapperForFxTracing, ...], tuple[torch.Tensor, ...], PyTreeSpec]:
        """Convert a Thunder trace to an FX graph and execute it with fake tensors.

        Args:
            trace: The Thunder trace to convert

        Returns:
            A tuple containing:
            - The FX GraphModule
            - The wrapped outputs from executing the FX graph
            - The original tensor outputs
            - The PyTree specification for the output structure
        """

        def create_ctor(unflatten_method, tensor_names):
            def ctor(tensors, metadata):
                inner_tensors = dict(zip(tensor_names, tensors))
                return unflatten_method(inner_tensors, metadata, -1, -1)

            return ctor

        args = self._materialize_trace_args_for_fx(trace)
        desugared_args = []
        arg_idx_to_sugar: dict[int, tuple[int, Any]] = {}
        for a in args:
            if is_traceable_wrapper_subclass(a):
                start_idx = len(desugared_args)
                attrs, metadta = a.__tensor_flatten__()
                desugared_args.extend([getattr(a, name) for name in attrs])
                desugared_args.append(metadta)
                end_idx = len(desugared_args)
                arg_idx_to_sugar[start_idx] = end_idx, create_ctor(type(a).__tensor_unflatten__, attrs)
            else:
                desugared_args.append(a)

        out_specs: list[Any] = []
        orig_output: list[torch.Tensor] = []

        def transform_out(out: torch.Tensor) -> OutputWrapperForFxTracing:
            orig_output.append(out)
            if is_traceable_wrapper_subclass(out):
                from enum import Enum

                attrs, metadata = out.__tensor_flatten__()
                tensors = [getattr(out, name) for name in attrs]
                for key in metadata:
                    v = metadata[key]
                    if issubclass(type(v), Enum) and not isinstance(v, (torch.dtype, torch.device)):
                        metadata[key] = str(metadata[key])
                output = OutputWrapperForFxTracing(dict(zip(attrs, tensors)), metadata)
            else:
                output = OutputWrapperForFxTracing(out, None)
            return output

        desugared_proxy_args = []
        for a in trace.args:
            if isinstance(a, SubclassTensorProxy):
                names, metadata = a.__tensor_flatten__()
                desugared_proxy_args.extend([getattr(a, name) for name in names])
                desugared_proxy_args.append(metadata)
            else:
                desugared_proxy_args.append(a)

        extrace = make_trace_executable(trace, *trace.args, **trace.kwargs)
        utils.check(
            (len(extrace.bound_symbols) == len(trace.bound_symbols))
            or (
                len(extrace.bound_symbols) == len(trace.bound_symbols) - 1
                and any(bsym.sym.id == prims.PrimIDs.SHALLOW_COPY for bsym in trace.bound_symbols)
            ),
            lambda: (
                f"Input trace is\n{trace}\nExecution trace is\n{extrace}\n"
                f"Input has {len(trace.bound_symbols)} syms but execution trace has {len(extrace.bound_symbols)}"
            ),
        )
        f = extrace.python_callable(include_decorators=False)

        def f_with_wrap_and_unwrap(*desugared_args) -> tuple[OutputWrapperForFxTracing, ...]:
            args = []
            cur_idx = 0
            while cur_idx < len(desugared_args):
                if cur_idx in arg_idx_to_sugar:
                    end_idx, construct_subclass = arg_idx_to_sugar[cur_idx]
                    args_of_subclass = desugared_args[cur_idx:end_idx]
                    tensors = args_of_subclass[:-1]
                    metadata = args_of_subclass[-1]
                    subclass = construct_subclass(tensors, metadata)
                    args.append(subclass)

                    cur_idx = end_idx
                else:
                    args.append(desugared_args[cur_idx])
                    cur_idx += 1

            out = f(*args)
            # Specialcasing the output of initial computation trace
            if isinstance(out, dict) and len(out) == 2 and ("output", "flat_args") == tuple(out.keys()):
                sequencified_out = out
            else:
                sequencified_out = utils.sequencify(out)
            flat_out, out_spec = tree_flatten(sequencified_out)
            out_specs.append(out_spec)
            flat_cosmeticized_out = tree_map(transform_out, flat_out)
            return tree_unflatten(flat_cosmeticized_out, out_spec)

        with (
            enable_python_dispatcher(),
            FunctionalTensorMode(
                pre_dispatch=False,
                export=False,
                _allow_token_discovery=True,
            ),
        ):
            fx: GraphModule = make_fx(f_with_wrap_and_unwrap)(*desugared_args)

        arity_of_fx_forward = fx.forward.__code__.co_argcount - 1
        utils.check(
            arity_of_fx_forward == len(desugared_args),
            lambda: f"{arity_of_fx_forward=}, {len(desugared_args)=}, {desugared_args=}",
        )

        return fx, fx(*desugared_args), tuple(orig_output), out_specs[0]

    def __call__(self, bsym: BoundSymbol) -> list[BoundSymbol]:
        """Process a bound symbol, handling tensor subclasses appropriately.

        Args:
            bsym: The bound symbol to process

        Returns:
            A list of processed bound symbols
        """
        if self.proxy_to_strides:
            diff_of_proxy_to_strides = {
                variableify(v): self.proxy_to_strides[k] for k, v in self.swap_map.items() if k in self.proxy_to_strides
            }
            self.proxy_to_strides.update(diff_of_proxy_to_strides)

        updated_bsym = bsym.from_bsym_swap_proxies(self.swap_map)

        # Handle return operation
        if bsym.sym.id == prims.PrimIDs.RETURN:
            return self._handle_return_operation(updated_bsym)

        # Determine the type of bound symbol
        is_subclass_ctor, no_subclass_args = self._classify_bound_symbol(updated_bsym)

        # Fast path: if not a subclass constructor and has no subclass args, return as is
        if not is_subclass_ctor and no_subclass_args:
            return [updated_bsym]

        # Verify we can handle this bound symbol
        utils.check(
            len(updated_bsym.flat_outs) < 2,
            lambda: f"bsym has {len(updated_bsym.flat_outs)} outputs",
            exception_type=NotImplementedError,
        )

        # Convert the bound symbol to an FX graph and process it
        return self._process_bound_symbol_with_fx(updated_bsym, is_subclass_ctor)

    # TODO(crcrpar): Remove this method.
    def _handle_return_operation(self, bsym: BoundSymbol) -> list[BoundSymbol]:
        """Handle return operation bound symbols.

        Args:
            bsym: The return operation bound symbol

        Returns:
            A list containing the processed bound symbol
        """
        # Filter out SubclassTensorProxy entries from swap_map
        new_swap_map = {k: v for k, v in self.swap_map.items() if not isinstance(v, SubclassTensorProxy)}

        if self._is_augmented_forward_trace():
            updated_saved_for_backward = self._get_flat_saved_tensors_for_backward(bsym)
            self.saved_tensors_for_backward = list(updated_saved_for_backward)
        return [bsym]

    def _classify_bound_symbol(self, bsym: BoundSymbol) -> tuple[bool, bool]:
        """Determine if the bound symbol is a subclass constructor and if it has subclass args.

        Args:
            bsym: The bound symbol to classify

        Returns:
            A tuple of (is_subclass_ctor, no_subclass_args)
        """
        is_bsym_of_subclass_ctor = bsym.sym.id == prims.PrimIDs.TENSOR_SUBCLASS_CTOR
        returns_subclass = any(isinstance(a, SubclassTensorProxy) for a in bsym.flat_proxy_outs)
        no_subclass_args = all(not isinstance(a, SubclassTensorProxy) for a in bsym.flat_proxy_args)
        is_unpack = bsym.sym.id in {prims.PrimIDs.UNPACK_TRIVIAL, prims.PrimIDs.UNPACK_SEQUENCE}

        is_subclass_ctor = is_bsym_of_subclass_ctor or (no_subclass_args and returns_subclass and not is_unpack)
        return is_subclass_ctor, no_subclass_args

    def _process_bound_symbol_with_fx(self, bsym: BoundSymbol, is_subclass_ctor: bool) -> list[BoundSymbol]:
        """Process a bound symbol by converting it to an FX graph and handling the results.

        Args:
            bsym: The bound symbol to process
            is_subclass_ctor: Whether the bound symbol is a subclass constructor

        Returns:
            A list of processed bound symbols
        """
        # Convert the bound symbol to an FX graph
        trace = trace_from_bsym_or_bsyms(bsym)
        fx: GraphModule
        sequencified_cosmeticized_out: tuple[OutputWrapperForFxTracing, ...]
        orig_output: tuple[torch.Tensor, ...]
        fx, sequencified_cosmeticized_out, orig_output, _ = self.convert_trace_to_fx_graph_and_get_fake_result(trace)

        utils.check(
            len(sequencified_cosmeticized_out) == len(orig_output),
            lambda: f"{len(sequencified_cosmeticized_out)=}, {len(orig_output)=}",
        )

        # Handle subclass constructor case
        if is_subclass_ctor:
            return self._handle_subclass_constructor(bsym, sequencified_cosmeticized_out, orig_output)

        # Handle regular operation case
        return self._handle_regular_operation(bsym, fx, sequencified_cosmeticized_out, orig_output)

    def _handle_subclass_constructor(
        self,
        bsym: BoundSymbol,
        sequencified_cosmeticized_out: tuple[OutputWrapperForFxTracing, ...],
        orig_output: tuple[torch.Tensor, ...],
    ) -> list[BoundSymbol]:
        """Handle a bound symbol that constructs a tensor subclass.

        Args:
            bsym: The bound symbol to process
            sequencified_cosmeticized_out: The wrapped outputs from the FX graph
            orig_output: The original tensor outputs

        Returns:
            A list containing the processed bound symbol
        """
        utils.check(len(sequencified_cosmeticized_out) == 1 and len(orig_output) == 1, lambda: "")
        subclass_proxy = bsym.flat_outs[0]
        utils.check_type(subclass_proxy, SubclassTensorProxy)
        self.subclass_proxy_to_flatten.add(variableify(subclass_proxy))

        # Extract tensor attributes and metadata
        fake_tensor_subclass = orig_output[0]
        tensor_attr_names, metadata = fake_tensor_subclass.__tensor_flatten__()
        fake_tensors = [getattr(fake_tensor_subclass, name) for name in tensor_attr_names]

        # with tracectx(self.computation_trace):
        #     proxy_of_fake_tensors = [proxy_fake_tensor(t) for t in fake_tensors]

        # Set attributes on the proxy
        subclass_proxy._tensor_attr_names = tensor_attr_names
        subclass_proxy._non_tensor_attr_names = list(metadata.keys())
        # subclass_proxy._tensors.extend(proxy_of_fake_tensors)
        for name, proxy in zip(tensor_attr_names, subclass_proxy._tensors):
            setattr(subclass_proxy, name, proxy)
        for key, val in metadata.items():
            setattr(subclass_proxy, key, val)

        strides: list[tuple[int, ...]] = [t.stride() for t in fake_tensors]

        # TODO(crcrpar): Track only if strides indicate that the tensor is not contiguous.
        self.proxy_to_strides[variableify(subclass_proxy)] = strides
        for p, s in zip(subclass_proxy._tensors, strides):
            self.proxy_to_strides[variableify(p)] = s
        self.proxy_to_strides[variableify(subclass_proxy)] = strides
        bsym.header += f" Tensor Subclass Transform: {strides=} from {subclass_proxy.name=}"

        return [bsym]

    def _handle_regular_operation(
        self,
        bsym: BoundSymbol,
        fx: GraphModule,
        sequencified_cosmeticized_out: tuple[OutputWrapperForFxTracing, ...],
        orig_output: tuple[torch.Tensor, ...],
    ) -> list[BoundSymbol]:
        """Handle a bound symbol that is a regular operation (not a constructor).

        Args:
            bsym: The bound symbol to process
            fx: The FX graph module
            sequencified_cosmeticized_out: The wrapped outputs from the FX graph
            orig_output: The original tensor outputs

        Returns:
            A list of processed bound symbols
        """
        # Process outputs
        out = self._process_outputs(sequencified_cosmeticized_out, orig_output)

        # Create proxies for the outputs
        with tracectx(self.computation_trace):
            out_proxy = tree_map(proxy_fake_tensor, out)

        # Verify output counts match
        utils.check(
            len(bsym.flat_outs) == len(out_proxy),
            lambda: f"{len(bsym.flat_outs)=}, {len(out_proxy)=}, {out_proxy=}, {bsym.flat_outs=}",
        )

        # Update swap map with new proxies
        sequence_out = [variableify(a) for a in bsym.flat_outs]
        self.swap_map.update(dict(zip(sequence_out, utils.sequencify(out_proxy))))

        # Create bound symbol with updated outputs
        bsym_with_modified_output = bsym.from_bsym_swap_proxies(self.swap_map)
        self.bsym_to_new_outputs[bsym_with_modified_output] = bsym_with_modified_output

        # Translate the FX graph into bound symbols
        bsyms = self.translate_fx_graph_into_bsym(bsym_with_modified_output, fx)

        utils.check(
            len(bsym.flat_proxy_outs) == len(orig_output),
            lambda: f"{len(bsym.flat_proxy_outs)=}, {len(orig_output)=}",
        )

        strides_in_header: str | None = None
        for proxy, fake_tensor in zip(out_proxy, orig_output):
            # TODO(crcrpar): Track only proxies whose strides are not contiguous.
            if is_traceable_wrapper_subclass(fake_tensor):
                tensor_attr_names, _ = fake_tensor.__tensor_flatten__()
                strides = [getattr(fake_tensor, name).stride() for name in tensor_attr_names]
                var_p = variableify(proxy)
                self.proxy_to_strides[var_p] = strides

                strides_in_header = f"{proxy.name=}, {strides=}"
                # TODO(crcrpar): Might better track tensot components as well
                # for p, s in zip(proxy._tensors, strides):
                #     self.proxy_to_strides[variableify(p)] = s
            elif isinstance(fake_tensor, FakeTensor):
                stride = fake_tensor.stride()
                var_p = variableify(proxy)
                self.proxy_to_strides[var_p] = stride
                strides_in_header = f"{proxy.name=}, {stride=}"
            else:
                utils.check(False, lambda: f"{proxy=}, {fake_tensor=}")
        if strides_in_header is not None:
            bsyms[-1].header += f" Tensor Subclass Transform: {strides_in_header}"

        return bsyms

    def _process_outputs(
        self,
        sequencified_cosmeticized_out: tuple[OutputWrapperForFxTracing, ...],
        orig_output: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Process and validate the outputs from the FX graph execution.

        Args:
            sequencified_cosmeticized_out: The wrapped outputs from the FX graph
            orig_output: The original tensor outputs

        Returns:
            A list of processed outputs
        """
        out = []
        for cosmeticized_out, orig_out in zip(sequencified_cosmeticized_out, orig_output):
            if isinstance(cosmeticized_out.inner_tensors, dict):
                utils.check(
                    is_traceable_wrapper_subclass(orig_out), lambda: f"{cosmeticized_out=} don't match {orig_out=}"
                )
            out.append(orig_out)
        return out

    def get_proxy_to_strides_for_saved_for_backward(self) -> dict[Variable, tuple[int, ...] | list[tuple[int, ...]]]:
        if self._is_backward_trace():
            return {}
        if not self._is_augmented_forward_trace():
            return {}
        utils.check(
            (
                (self._is_augmented_forward_trace() and self.saved_tensors_for_backward)
                or (not self.saved_tensors_for_backward)
            ),
            lambda: f"{self._is_augmented_forward_trace() = }, {self.saved_tensors_for_backward = }",
        )
        saved_tensors_for_backward: list[TensorProxy | SubclassTensorProxy | Any]
        if self._is_augmented_forward_trace():
            saved_tensors_for_backward = self.saved_tensors_for_backward
        else:
            saved_tensors_for_backward = self._get_flat_saved_tensors_for_backward()

        d = {
            k: self.proxy_to_strides[k]
            for k in [variableify(v) for v in saved_tensors_for_backward]
            if k in self.proxy_to_strides
        }
        return d


def tensor_subclass_dce(trace: TraceCtx, is_bwd_trace: bool) -> TraceCtx:
    """Remove ``tensor.__tensor_flatten__``s as possible.

    This function tries to remove flattening of tensor subclass
    by replacing their outputs with tensor args of ``tensor``\'s constructor,
    either '`TensorSubclass(...)` or `TensorSubclass.__tensor_unflatten__(...)`.

    This function does not remove ``TensorSubclass(...)`` nor ``TensorSubclass.__tensor_unflatten__(...)``
    as they could be a saved tensor for backward.
    """
    start_time_ns = time.perf_counter_ns()
    swap_map: dict[Variable, TensorProxy] = {}
    producer_map = utils.producers(trace)
    bsym_to_exclude: set[BoundSymbol] = set()

    # Handle adhoc executor subsymbols for backward trace
    if is_bwd_trace:
        from thunder.core.compile_data import get_compile_data
        from thunder.extend import TemporaryExecutor

        cd = get_compile_data()
        temporary_executor: TemporaryExecutor | None = None
        if cd is not None:
            executors_list = list(filter(lambda executor: isinstance(executor, T), cd.executors_list))
            if executors_list:
                temporary_executor = executors_list[0]

        if temporary_executor is not None and temporary_executor._implmap:
            new_bsyms = []
            for bsym in trace.bound_symbols:
                if temporary_executor.can_execute(bsym) and bsym.subsymbols:
                    new_bsyms.extend(bsym.subsymbols)
                else:
                    new_bsyms.append(bsym)
            trace = from_trace(trace)
            trace.bound_symbols = new_bsyms
            # Rebuild producer map with updated bound symbols
            producer_map = utils.producers(trace)

    subclass_flatten_bsym: BoundSymbol
    for subclass_flatten_bsym in filter(
        lambda bsym: bsym.sym.id == prims.PrimIDs.FLATTEN_TENSOR_SUBCLASS,
        trace.bound_symbols,
    ):
        subclass_tensor_proxy: SubclassTensorProxy = subclass_flatten_bsym.flat_args[0]
        flatten_tensors: tuple[TensorProxy, ...] = subclass_flatten_bsym.output
        ctor_bsym: BoundSymbol = producer_map[subclass_tensor_proxy]
        match ctor_bsym.sym.id:
            case prims.PrimIDs.TENSOR_SUBCLASS_CTOR:
                ctor_tensors: list[TensorProxy] = ctor_bsym.args[6]
            case prims.PrimIDs.UNFLATTEN_TENSOR_SUBCLASS:
                ctor_tensors: list[TensorProxy] = list(ctor_bsym.args[1].values())
            case _:
                continue
        utils.check(
            len(flatten_tensors) == len(ctor_tensors),
            lambda: f"{flatten_tensors} and {ctor_tensors} have different number of tensors",
        )

        for k, v in zip(flatten_tensors, ctor_tensors):
            if k.name == v.name:
                continue
            swap_map[variableify(k)] = v
        bsym_to_exclude.add(subclass_flatten_bsym)

    if not swap_map:
        return trace

    new_bsyms: list[BoundSymbol] = []
    bsym: BoundSymbol
    for bsym in trace.bound_symbols:
        if bsym in bsym_to_exclude:
            continue
        new_bsyms.append(bsym.from_bsym_swap_proxies(swap_map, skip_output=True))

    new_trace = from_trace(trace)
    new_trace.bound_symbols = new_bsyms
    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    new_trace.set_provenance(
        TraceProvenance(f"DCE of Tensor Subclass Flattening/Unflattening (took {elapsed_time_millis} milliseconds)")
    )

    return new_trace


def unroll_tensor_subclasses(
    trace: TraceCtx,
    *,
    is_bwd_trace: bool = False,
    proxy_to_strides: dict[Variable, tuple[int, ...] | list[tuple[int, ...]]] | None = None,
) -> tuple[TraceCtx, dict[Variable, tuple[int, ...] | list[tuple[int, ...]]]]:
    """Unroll tensor subclasses in ``computation_trace``.

    Two things are happening inside of this function:
        * Reevaluate every single bsym of ``computation_trace.bound_symbols``.
        * Flatten tensor subclasses

    Each :class:`thunder.core.symbol.BoundSymbol` is reevaluated with torch.fx tracing and
    ``FakeTensorMode``. This is necessary because Thunder's initial trace cannot correctly infer the output
    type of an op with tensor subclasses. By translating each bsym into a callable and tracing it with
    ``torch.fx`` and ``FakeTensorMode``, we can tell the output type and the exact behavior of the bsym
    which is extended by subclass's ``__torch_dispatch__`` (note that the sequence of observed operations
    are free from tensor subclasses, everything is flattened).
    The output type information is then reflected to the output :class:`thunder.core.proxies.Proxy`.

    With this function applied, the :class:`thunder.core.trace.TraceCtx` is free from tensor subclasses.
    Exceptions are prologue (meaning the first few lines of the trace, before any math) and epilogue (meaning
    the last few lines of the trace, right before return statement).

    Args:
        trace:

    Returns:
        TraceCtx: transformed trace that is free from tensor subclasses, every ``__torch_dispatch__``
            behavior is spelled out.
    """
    start_time_ns = time.perf_counter_ns()

    desugar_tensor_subclass = DesugarTensorSubclass(computation_trace=trace)
    if proxy_to_strides is not None and proxy_to_strides:
        desugar_tensor_subclass.proxy_to_strides.update(proxy_to_strides)
    updated_bsyms: list[BoundSymbol] = []
    bsym: BoundSymbol
    for bsym in trace.bound_symbols:
        maybe_desugared_bsyms = desugar_tensor_subclass(bsym)
        updated_bsyms.extend(maybe_desugared_bsyms)

    if not desugar_tensor_subclass.subclass_proxy_to_flatten:
        return trace, desugar_tensor_subclass.proxy_to_strides

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000

    computation_trace_with_subclass_tensor_unrolled = from_trace(trace)
    computation_trace_with_subclass_tensor_unrolled.bound_symbols.extend(updated_bsyms)
    computation_trace_with_subclass_tensor_unrolled.set_provenance(
        TraceProvenance(f"tensor subclasses unrolled (took {elapsed_time_millis} milliseconds)")
    )
    dced_computation_trace = tensor_subclass_dce(computation_trace_with_subclass_tensor_unrolled, is_bwd_trace)
    warn_tensor_subclass_support()
    return (
        dced_computation_trace,
        desugar_tensor_subclass.get_proxy_to_strides_for_saved_for_backward() if not is_bwd_trace else {},
    )
