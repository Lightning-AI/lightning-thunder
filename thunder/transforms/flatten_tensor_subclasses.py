from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from numbers import Number
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses import fake_tensor
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
) -> torch.Tensor:
    utils.check(
        (subclass_type := getattr(tensor_proxy, "_tensor_subclass_type", None)) is not None,
        lambda: f"{tensor_proxy} does not have `_tensor_subclass_type`",
    )
    inner_tensor_proxies = tensor_proxy._tensors
    inner_tensors = dict(
        zip(
            tensor_proxy._tensor_attr_names,
            [_materialize_tensor_proxy(t, fake_tensor_mode=fake_tensor_mode) for t in inner_tensor_proxies],
        )
    )
    subclass_tensor = subclass_type.__tensor_unflatten__(
        inner_tensors, tensor_proxy._metadata, outer_size=-1, outer_stride=-1
    )
    fakified = fake_tensor_mode.from_tensor(subclass_tensor, static_shapes=True)
    return fakified


def materialize_tensor_proxy(
    t: TensorProxy | SubclassTensorProxy, fake_tensor_mode: fake_tensor_mode.FakeTensorMode
) -> torch.Tensor:
    if isinstance(t, SubclassTensorProxy):
        return _make_fake_subclass_tensor_from_subclass_tensor_proxy(t, fake_tensor_mode)
    return _materialize_tensor_proxy(t, fake_tensor_mode)


def maybe_materialize_tensor(
    t: ProxyInterface, fake_tensor_mode: fake_tensor.FakeTensorMode
) -> ProxyInterface | torch.Tensor:
    if isinstance(t, (TensorProxy, SubclassTensorProxy)):
        return materialize_tensor_proxy(t, fake_tensor_mode)
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
        tensor_attrs, metadata = t.__tensor_flatten__()
        tensor_proxies = [proxy_fake_tensor(getattr(t, name)) for name in tensor_attrs]
        p = SubclassTensorProxy(
            None,
            shape=list(t.shape),
            dtype=dtypes.to_dtype(t.dtype),
            device=devices.to_device(t.device),
            requires_grad=t.requires_grad,
        )
        p._set_tensor_attrs(tensor_attrs, tensor_proxies)
        p._set_non_tensor_attrs(metadata)
        p._torch_dispatch_impl = t.__torch_dispatch__
        p._tensor_subclass_type = type(t)
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
    trace_args_set: Any = field(init=False, default=None)
    requires_desugarring: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.requires_desugarring = any(
            isinstance(t, SubclassTensorProxy)
            for t in tree_flatten((self.computation_trace.args, self.computation_trace.kwargs))[0]
        )
        if not self.requires_desugarring:
            return
        (
            self.fx_computation_trace,
            self.computation_trace_output,
            self.fx_computation_trace_result,
            self.spec_of_fx_computation_trace_result,
        ) = self.convert_trace_to_fx_graph_and_get_fake_result(
            self.computation_trace,
        )
        self.flat_trace_args, self.flat_trace_args_spec = tree_flatten(
            (self.computation_trace.args, self.computation_trace.kwargs)
        )
        self.trace_args_set = {variableify(a) for a in self.flat_trace_args}

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
                if variableify(a) in self.trace_args_set:
                    self.computation_trace.push_scope([])
                    with tracectx(self.computation_trace):
                        prims.flatten_tensor_subclass(a)
                    unflatten_bsym = self.computation_trace.pop_scope()[0]
                    list_of_unflatten_bsym.append(unflatten_bsym)
                tensor_names = a._tensor_attr_names
                tensors = a._tensors
                metdata = a._metadata
                for name, t in zip(tensor_names, tensors):
                    unwrapped_bsym_args[len(unwrapped_bsym_args)] = t
                utils.check(
                    not metdata,
                    lambda: f"Tensor Subclasses with nonempty metdata are not supported.",
                    exception_type=NotImplementedError,
                )
            else:
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
            new_subclass._set_tensor_attrs(new_subclass._tensor_attr_names, new_tensor_proxies)
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

        args = tree_map(lambda t: maybe_materialize_tensor(t, self.fake_tensor_mode), trace.args)
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

    .. note::

        Currently any tensor subclass factories are not allowed.


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
