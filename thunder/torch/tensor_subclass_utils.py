from thunder.dynamo import thunderfx
from thunder.core.transforms import Transform
from thunder.core.symbol import BoundSymbol
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from numbers import Number
from typing import TYPE_CHECKING, NamedTuple
import time
import warnings
from functools import wraps

import torch
import torch.utils._pytree as pytree
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


def trace_from_bsym_or_bsyms(bsym_or_bsyms: BoundSymbol | Sequence[BoundSymbol]) -> TraceCtx:
    from thunder.core.compile_data import get_compile_data

    bsyms = list(utils.sequencify(bsym_or_bsyms))
    trace_args = bsyms[0].flat_args
    trace_name = bsyms[0].sym.name

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
    from thunder.executors.torchex import ex as pytorch_ex
    from thunder.executors.passes import transform_for_execution

    # torch_trace = transform_for_execution(trace_to_convert, executors_list=(pytorch_ex,))
    # trace_callable = torchex_trace.python_callable(include_decorators=False)

    @wraps(trace_to_convert.python_callable())
    def torch_interpreted_func(*args, **kwargs):
        return eval_trace(trace_to_convert, *args, **kwargs, symbol_mapper=to_torch_translator)

    torch_trace = trace(inline_trace=False)(torch_interpreted_func, *args_for_eval, **kwargs_for_eval)
    return torch_trace


def get_fx_graph(trc, args_for_fx, tensor_subclass):
    from thunder.core.proxies import TensorProxy, ScaleTensorProxy
    from thunder.core.devices import to_torch_device
    from thunder.core.dtypes import to_torch_dtype

    f = trc.python_callable(include_decorators=False)

    with FakeTensorMode():

        def materialize(t):
            if not isinstance(t, TensorProxy):
                return t

            if isinstance(t, ScaleTensorProxy):
                i_t = torch.randn(t.a.shape, device=to_torch_device(t.a.device), dtype=to_torch_dtype(t.a.dtype))
                return tensor_subclass(i_t, t.b)

            return torch.randn(t.shape, device=to_torch_device(t.device), dtype=to_torch_dtype(t.dtype))

        fake_t = tree_map(materialize, args_for_fx)

    args_to_unflatten = tuple(isinstance(t, tensor_subclass) for t in fake_t)

    def flatten_tensor(t):
        attrs, metadata = t.__tensor_flatten__()
        attrs = {name: getattr(t, name) for name in attrs}
        result = attrs, metadata  # , t.shape, t.stride()
        return result

    out_spec = []

    def wrapped_fn(args_for_fx):
        new_args = ()
        for unflatten, arg in zip(args_to_unflatten, args_for_fx):
            if unflatten:
                new_args += (tensor_subclass.__tensor_unflatten__(*arg),)
            else:
                new_args += (arg,)

        o = f(*new_args)
        o = pytree.tree_map_only(tensor_subclass, flatten_tensor, o)
        flat_o, _ = pytree.tree_flatten(o)
        out_spec.append(_)
        return flat_o  # flatten_tensor(o)

    flattened_args = tuple(flatten_tensor(arg) if isinstance(arg, tensor_subclass) else arg for arg in fake_t)

    with (
        enable_python_dispatcher(),
        FunctionalTensorMode(pre_dispatch=False, export=False, _allow_token_discovery=True),
    ):
        g = make_fx(wrapped_fn, tracing_mode="fake")(flattened_args)

    # print(out_spec)
    from thunder.dynamo.utils import _checkpoint_function_converter
    import thunder

    # Replaces `aten` ops with thunder equivalent
    # and this makes the `g` traceable with thunder.
    _checkpoint_function_converter(g)
    g.recompile()

    # Otherwise `trace` get's confused.
    del g.meta

    @wraps(g)
    def wrap(*args):
        return g(*args)

    def flatten_tensor_proxy(t):
        return {"a": t.a}, {"b": t.b}

    flattened_args = tuple(
        flatten_tensor_proxy(arg) if isinstance(arg, ScaleTensorProxy) else arg for arg in args_for_fx
    )

    trc = thunder.trace(rename_proxies=False)(wrap, flattened_args)

    aten_syms = []
    for bsym in trc.bound_symbols:
        if "aten" in bsym.sym.name:
            aten_syms.append(bsym)
    # print(trc.bound_symbols[-1].flat_args)
    # print()
    flat_output = pytree.tree_unflatten(trc.bound_symbols[-1].flat_args, out_spec[0])
    if not isinstance(flat_output, list):
        flat_output = [
            flat_output,
        ]

    return aten_syms, flat_output


def decompose_into_aten_subsymbols(bsym, comp_trace, tensor_subclass):
    from thunder.core.proxies import TensorProxy, ScaleTensorProxy, variableify

    filter_tensor_proxies = list(filter(lambda t: isinstance(t, ScaleTensorProxy), bsym.flat_args))
    tensor_proxies = list(filter(lambda t: isinstance(t, TensorProxy), bsym.flat_args))
    if len(filter_tensor_proxies) == len(tensor_proxies):
        trc = trace_from_bsym_or_bsyms(bsym)
        executable_trc = make_trace_executable(trc, *bsym.flat_args)
        aten_bsyms, output = get_fx_graph(executable_trc, bsym.flat_args, tensor_subclass)

        comp_trace.push_scope([])
        with tracectx(comp_trace):
            proxys = []
            for tp in filter_tensor_proxies:
                proxys.append(prims.get_subclass_inner_tensor(tp))
        syms = comp_trace.pop_scope()

        return_bsyms = []
        for o_proxy, o in zip(bsym.flat_outs, output):
            cons_bsym = prims.construct_subclass.bind(*o, output=o_proxy)
            return_bsyms.append(cons_bsym)

        bsym.subsymbols = syms + aten_bsyms + return_bsyms

    return bsym
