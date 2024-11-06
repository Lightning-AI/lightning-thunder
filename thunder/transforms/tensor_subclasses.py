from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from numbers import Number
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses import fake_tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

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
from thunder.executors.passes import transform_for_execution
from thunder.extend import get_executor

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from optree import PyTreeSpec
    from torch.fx import GraphModule
    from torch.fx import Node
    from torch._ops import OpOverload
    from thunder.core.symbol import Symbol, BoundSymbol
    from torch._C import _TensorMeta


__all__ = [
    "flatten_tensor_subclasses",
]


PLACEHOLDER: str = "placeholder"
CALL_FUNCTION: str = "call_function"
OUTPUT: str = "output"


class OutputWrapperForFxTracing(NamedTuple):
    inner_tensors: dict[str, torch.Tensor] | torch.Tensor
    metadata: dict[str, Any] | None


def _materialize_tensor_proxy(t: TensorProxy, fake_tensor_mode: fake_tensor.FakeTensorMode | None) -> torch.Tensor:
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
    fake_tensor_mode: fake_tensor.FakeTensorMode,
    subclass_to_attr_names: dict[_TensorMeta, tuple[list[str], list[str]]],
) -> torch.Tensor:
    utils.check(
        (subclass_type := getattr(tensor_proxy, SubclassTensorProxy.SUBCLASS_TYPE_ATTR, None)) is not None,
        lambda: f"{tensor_proxy} does not have `{SubclassTensorProxy.SUBCLASS_TYPE_ATTR}`",
    )
    utils.check(
        subclass_type in subclass_to_attr_names,
        lambda: f"{tensor_proxy}'s `{subclass_type=}` has never been observed",
    )
    utils.check(
        tensor_proxy._tensors,
        lambda: f"{tensor_proxy} has an empty `{tensor_proxy._tensors=}`",
    )
    tensor_attr_names, non_tensor_attr_names = subclass_to_attr_names[subclass_type]
    inner_tensors = dict(
        zip(
            tensor_attr_names,
            [_materialize_tensor_proxy(t, fake_tensor_mode=fake_tensor_mode) for t in tensor_proxy._tensors],
        )
    )
    metadata = dict(zip(non_tensor_attr_names, tensor_proxy._non_tensors))
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
    subclass_to_attr_names: dict[_TensorMeta, tuple[list[str], list[str]]],
) -> torch.Tensor:
    if isinstance(t, SubclassTensorProxy):
        return _make_fake_subclass_tensor_from_subclass_tensor_proxy(t, fake_tensor_mode, subclass_to_attr_names)
    return _materialize_tensor_proxy(t, fake_tensor_mode)


def maybe_materialize_tensor(
    t: ProxyInterface,
    fake_tensor_mode: FakeTensorMode,
    subclass_to_attr_names: dict[_TensorMeta, tuple[list[str], list[str]]],
) -> ProxyInterface | torch.Tensor:
    if isinstance(t, (TensorProxy, SubclassTensorProxy)):
        return materialize_tensor_proxy(t, fake_tensor_mode, subclass_to_attr_names)
    if isinstance(t, (Number, str)):
        return t
    return t.value


def proxy_fake_tensor(t: torch.Tensor | fake_tensor.FakeTensor) -> ProxyInterface:
    if isinstance(t, fake_tensor.FakeTensor) or (isinstance(t, torch.Tensor) and not issubclass(type(t), torch.Tensor)):
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
    bsyms = utils.sequencify(bsym_or_bsyms)

    trace = TraceCtx()
    trace.bound_symbols.extend(bsyms)
    trace.args = bsyms[0].flat_proxy_args
    with tracectx(trace):
        prims.python_return(bsyms[-1].output)
    with tracectx(trace):
        trace._siginfo = SigInfo.from_name_and_args(bsyms[0].sym.name, trace.args)
    return trace


def aten_core_ir_op_to_ltorch_op(aten_op: OpOverload) -> Symbol:
    import thunder.torch as ltorch

    op_name_without_overload = aten_op._opname
    utils.check(
        hasattr(ltorch, op_name_without_overload),
        lambda: f"{aten_op=} cannot find an appropriate ltorch op. Query: {op_name_without_overload}",
    )
    return getattr(ltorch, op_name_without_overload)


@dataclass
class DesugarTensorSubclass:
    computation_trace: TraceCtx
    swap_map: dict[Variable, ProxyInterface] = field(init=False, default_factory=dict)
    fake_tensor_mode: fake_tensor.FakeTensorMode = field(init=False, default_factory=fake_tensor.FakeTensorMode)
    fx_computation_trace: GraphModule = field(init=False, default=None)
    computation_trace_output: tuple[OutputWrapperForFxTracing, ...] = field(init=False, default=None)
    fx_computation_trace_result: tuple[torch.Tensor, ...] = field(init=False, default=None)
    spec_of_fx_computation_trace_result: PyTreeSpec = field(init=False, default=None)
    flat_trace_args: Sequence[ProxyInterface] = field(init=False, default=None)
    flat_trace_args_spec: Any = field(init=False, default=None)
    requires_desugarring: bool = field(init=False, default=False)
    subclass_type_to_attr_names: dict[_TensorMeta, tuple[list[str], list[str]]] = field(
        init=False, default_factory=dict
    )
    subclass_proxy_to_flatten: set[Variable] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self.flat_trace_args, self.flat_trace_args_spec = tree_flatten(
            (self.computation_trace.args, self.computation_trace.kwargs)
        )
        # TODO(crcrpar): From my perspective, this check is rather for the sake of faster compilation.
        # There could be a computation graph where none of the inputs are subclass while
        # that graph call subclass creation inside of it.
        self.requires_desugarring = any(isinstance(t, SubclassTensorProxy) for t in self.flat_trace_args)
        if not self.requires_desugarring:
            return

        for arg in self.flat_trace_args:
            self.maybe_update_subclass_type_dict(arg)

        (
            self.fx_computation_trace,
            self.computation_trace_output,
            self.fx_computation_trace_result,
            self.spec_of_fx_computation_trace_result,
        ) = self.convert_trace_to_fx_graph_and_get_fake_result(
            self.computation_trace,
        )
        self.subclass_proxy_to_flatten: set[Variable] = {
            variableify(a) for a in filter(lambda t: isinstance(t, SubclassTensorProxy), self.flat_trace_args)
        }

    def maybe_update_subclass_type_dict(self, proxy_arg: ProxyInterface) -> None:
        if not isinstance(proxy_arg, SubclassTensorProxy):
            return
        subclass_type = getattr(proxy_arg, SubclassTensorProxy.SUBCLASS_TYPE_ATTR)
        if subclass_type in self.subclass_type_to_attr_names and not hasattr(subclass_type, "_tensor_attr_names"):
            tensor_attr_names, non_tensor_attr_names = self.subclass_type_to_attr_names[subclass_type]
            for name, value in zip(tensor_attr_names, subclass_type._tensors):
                setattr(proxy_arg, name, value)
            for name, value in zip(non_tensor_attr_names, subclass_type._non_tensors):
                setattr(proxy_arg, name, value)
        elif subclass_type not in self.subclass_type_to_attr_names:
            tensor_attr_names = proxy_arg._tensor_attr_names
            non_tensor_attr_names = proxy_arg._non_tensor_attr_names
            self.subclass_type_to_attr_names[subclass_type] = tensor_attr_names, non_tensor_attr_names
        else:
            utils.check(False, lambda: f"{proxy_arg} hasn't gotten attribute names -- {subclass_type}")

    def _get_tensor_attr_names(self, p: SubclassTensorProxy) -> list[str]:
        subclass_type = p._subclass_type
        return self.subclass_type_to_attr_names[subclass_type][0]

    def _get_non_tensor_attr_names(self, p: SubclassTensorProxy) -> list[str]:
        subclass_type = p._subclass_type
        return self.subclass_type_to_attr_names[subclass_type][1]

    def translate_fx_graph_into_bsym(
        self,
        bsym: BoundSymbol,
        fx: GraphModule,
    ) -> BoundSymbol | tuple[BoundSymbol, ...]:
        import thunder.torch as ltorch

        unwrapped_bsym_args: dict[int, ProxyInterface] = {}
        list_of_unflatten_bsym: list[BoundSymbol] = []
        for a in bsym.flat_args:
            if isinstance(a, SubclassTensorProxy):
                if variableify(a) in self.subclass_proxy_to_flatten:
                    self.computation_trace.push_scope([])
                    with tracectx(self.computation_trace):
                        prims.flatten_tensor_subclass(a)
                    unflatten_bsym = self.computation_trace.pop_scope()[0]
                    list_of_unflatten_bsym.append(unflatten_bsym)
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
                utils.check(
                    not metadata,
                    lambda: f"Tensor Subclasses with nonempty metadata are not supported.",
                    exception_type=NotImplementedError,
                )
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
        for node in fx.graph.nodes:
            if node.op == PLACEHOLDER:
                list_of_placeholder_node.append(node)
            if node.op == CALL_FUNCTION:
                list_of_function_call_node.append(node)
            if node.op == OUTPUT:
                node_of_output = node
        args = [n.target for n in list_of_placeholder_node]
        arg_name_to_index = {a: i for i, a in enumerate(args)}
        ltorch_ops_for_node_of_ops = [getattr(ltorch, node.target._opname) for node in list_of_function_call_node]

        bsyms: list[BoundSymbol] = []
        if list_of_unflatten_bsym:
            bsyms.extend(list_of_unflatten_bsym)
        fxnode_output_name_to_tensor_proxy: dict[str, OpOverload] = {}
        for node, ltorch_op in zip(list_of_function_call_node, ltorch_ops_for_node_of_ops):
            args: list[Node] = node.args

            arg_proxies: list[ProxyInterface] = []
            for a in args:
                if isinstance(a.target, str):
                    arg_proxies.append(unwrapped_bsym_args[arg_name_to_index[a.target]])
                else:
                    arg_proxies.append(fxnode_output_name_to_tensor_proxy[str(a)])

            self.computation_trace.push_scope([])

            with tracectx(self.computation_trace):
                out = ltorch_op(*arg_proxies)
            fxnode_output_name_to_tensor_proxy[str(node)] = out
            bsyms.extend(self.computation_trace.pop_scope())
        if len(bsyms) == 0:
            return [bsym]

        orig_output = bsym.flat_outs[0]
        if is_subclass_ctor_bsym := bsym.sym.id == prims.PrimIDs.TENSOR_SUBCLASS_CTOR:
            utils.check_type(orig_output, SubclassTensorProxy)
        if isinstance(orig_output, SubclassTensorProxy):
            args: list[Node] = node_of_output.args[0]
            new_tensor_proxies = []
            for a in args:
                if isinstance(a.target, str):
                    new_tensor_proxies.append(unwrapped_bsym_args[arg_name_to_index[a.target]])
                else:
                    new_tensor_proxies.append(fxnode_output_name_to_tensor_proxy[str(a)])
            utils.check(
                len(orig_output._tensors) == len(new_tensor_proxies),
                lambda: (
                    f"The number of new tensor proxies for {orig_output=} does not match: "
                    f"{len(new_tensor_proxies)=} != {len(orig_output._tensors)=}"
                ),
            )
            with tracectx(self.computation_trace):
                new_subclass = orig_output.replace()
            for name, value in zip(new_subclass._tensor_attr_names, new_tensor_proxies):
                setattr(new_subclass, name, value)
            self.swap_map[variableify(orig_output)] = new_subclass
        return bsyms

    def convert_trace_to_fx_graph_and_get_fake_result(
        self,
        trace: TraceCtx,
    ) -> tuple[GraphModule, tuple[OutputWrapperForFxTracing, ...], tuple[torch.Tensor, ...], PyTreeSpec]:

        def create_ctor(unflatten_method, tensor_names):

            def ctor(tensors, metadata):
                inner_tensors = dict(zip(tensor_names, tensors))
                return unflatten_method(inner_tensors, metadata, -1, -1)

            return ctor

        args = tree_map(
            lambda t: maybe_materialize_tensor(
                t,
                self.fake_tensor_mode,
                self.subclass_type_to_attr_names,
            ),
            trace.args,
        )
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
                attrs, metadata = out.__tensor_flatten__()
                tensors = [getattr(out, name) for name in attrs]
                output = OutputWrapperForFxTracing(dict(zip(attrs, tensors)), metadata)
            else:
                output = OutputWrapperForFxTracing(out, None)
            return output

        extrace = transform_for_execution(trace, [get_executor("torch")])
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

        return fx, fx(*desugared_args), tuple(orig_output), out_specs[0]

    def __call__(self, bsym: BoundSymbol) -> list[BoundSymbol]:
        updated_bsym: BoundSymbol = bsym.from_bsym_swap_proxies(self.swap_map)
        if not any(isinstance(a, SubclassTensorProxy) for a in updated_bsym.flat_proxy_args):
            if bsym.sym.id == prims.PrimIDs.TENSOR_SUBCLASS_CTOR:
                subclass_proxy = updated_bsym.flat_proxy_outs[0]
                subclass_type = getattr(subclass_proxy, SubclassTensorProxy.SUBCLASS_TYPE_ATTR)
                utils.check(
                    (
                        has_attr_names := (
                            hasattr(subclass_proxy, "_tensor_attr_names")
                            and hasattr(subclass_proxy, "_non_tensor_attr_names")
                        )
                    )
                    or subclass_type in self.subclass_type_to_attr_names,
                    lambda: f"{subclass_type=} has not been observed",
                )
                if not has_attr_names:
                    tensor_attr_names, non_tensor_attr_names = self.subclass_type_to_attr_names[subclass_type]
                    subclass_proxy._tensor_attr_names = tensor_attr_names
                    subclass_proxy._non_tensor_attr_names = non_tensor_attr_names

                    for name, value in zip(
                        tensor_attr_names + non_tensor_attr_names,
                        subclass_proxy._tensors + subclass_proxy._non_tensors,
                    ):
                        setattr(subclass_proxy, name, value)
                self.subclass_proxy_to_flatten.add(variableify(subclass_proxy))
            return [updated_bsym]

        utils.check(
            len(updated_bsym.flat_outs) < 2,
            lambda: f"bsym has {len(updated_bsym.flat_outs)} outputs",
            exception_type=NotImplementedError,
        )

        if updated_bsym.sym.id == prims.PrimIDs.RETURN:
            unflatten_fake_tensor_result = tree_unflatten(
                self.fx_computation_trace_result,
                self.spec_of_fx_computation_trace_result,
            )
            outputs: dict[str, Any] = updated_bsym.args[0]  # {"output": ..., "flat_args": ...}
            utils.check_type(outputs, dict)
            utils.check(
                isinstance(outputs, dict) and len(outputs) == 2 and ("output", "flat_args") == tuple(outputs.keys()),
                lambda: fr"{outputs=} does not conform to the format of \{'output': ..., 'flat_args': [...]\}",
            )
            seq_outs = utils.sequencify(outputs["output"])
            seq_fake_ret = utils.sequencify(unflatten_fake_tensor_result["output"])
            utils.check(
                len(seq_outs) == len(seq_fake_ret),
                lambda: f"{outputs['output']=}, {unflatten_fake_tensor_result['output']=}",
            )

            bsyms: list[BoundSymbol] = []
            for proxy_output, fx_output in zip(seq_outs, seq_fake_ret):
                if not isinstance(proxy_output, SubclassTensorProxy):
                    continue
                tensor_attrs, metadata = proxy_output.__tensor_flatten__()
                tensors = [getattr(proxy_output, name) for name in tensor_attrs]
                bsyms.append(
                    prims.unflatten_tensor_subclass.bind(
                        type(fx_output), dict(zip(tensor_attrs, tensors)), metadata, output=proxy_output
                    )
                )
            return [*bsyms, updated_bsym]

        trace = trace_from_bsym_or_bsyms(updated_bsym)
        fx, sequencified_cosmeticized_out, orig_output, _ = self.convert_trace_to_fx_graph_and_get_fake_result(trace)
        utils.check(
            len(sequencified_cosmeticized_out) == len(orig_output),
            lambda: f"{len(sequencified_cosmeticized_out)=}, {len(orig_output)=}",
        )
        out = []
        for i, (cosmeticized_out, orig_out) in enumerate(zip(sequencified_cosmeticized_out, orig_output)):
            if isinstance(cosmeticized_out.inner_tensors, dict):
                utils.check(
                    is_traceable_wrapper_subclass(orig_out), lambda: f"{cosmeticized_out=} don't match {orig_out=}"
                )
                out.append(orig_out)
            else:
                out.append(orig_out.tensors)

        with tracectx(self.computation_trace):
            out_proxy = tree_map(proxy_fake_tensor, out)

        utils.check(
            len(updated_bsym.flat_outs) == len(out_proxy),
            lambda: f"{len(bsym.flat_outs)=}, {len(out_proxy)=}, {out_proxy=}, {bsym.flat_outs=}",
        )
        sequence_out = [variableify(a) for a in updated_bsym.flat_outs]
        self.swap_map.update(dict(zip(sequence_out, utils.sequencify(out_proxy))))

        bsym_with_modified_output = updated_bsym.from_bsym_swap_proxies(self.swap_map)
        return self.translate_fx_graph_into_bsym(bsym_with_modified_output, fx)


def flatten_tensor_subclasses(computation_trace: TraceCtx) -> TraceCtx:
    """Flatten tensor subclasses in ``computation_trace``.

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
        computation_trace:

    Returns:
        TraceCtx: transformed trace that is free from tensor subclasses, every ``__torch_dispatch__``
            behavior is spelled out.
    """
    desugar_tensor_subclass = DesugarTensorSubclass(computation_trace=computation_trace)
    if not desugar_tensor_subclass.requires_desugarring:
        return computation_trace
    updated_bsyms: list[BoundSymbol] = []
    bsym: BoundSymbol
    for bsym in computation_trace.bound_symbols:
        maybe_desugared_bsyms = desugar_tensor_subclass(bsym)
        updated_bsyms.extend(maybe_desugared_bsyms)

    computation_trace_with_subclass_tensor_proxy_output = from_trace(computation_trace)
    computation_trace_with_subclass_tensor_proxy_output.bound_symbols.extend(updated_bsyms)
    computation_trace_with_subclass_tensor_proxy_output.set_provenance(TraceProvenance("tensor subclasses desugared"))
    return computation_trace_with_subclass_tensor_proxy_output
