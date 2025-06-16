from numbers import Number
from typing import Any
from collections.abc import Callable, Sequence

import torch

import thunder
from thunder.core.trace import from_trace
from thunder.core.proxies import variableify, Variable, TensorProxy, NumberProxy
from thunder.core.symbol import BoundSymbol
from thunder.core.dtypes import to_dtype
from thunder.core.devices import to_device
from thunder.torch import _torch_to_thunder_function_map
from thunder.core.utils import get_symbols_to_last_used_variables
from thunder.core.trace import TraceCtx, tracectx
from thunder.core.codeutils import SigInfo
from thunder.core import utils
from thunder.core.proxies import ProxyInterface
from thunder.core import prims
from thunder.executors.passes import transform_for_execution
from thunder.executors.torchex import ex as pytorch_ex
from thunder.executors.pythonex import ex as pythonex_ex
from thunder.core.pytree import tree_flatten


__all__ = [
    "ConstantFolding",
]


_thunder_to_torch_function_map = {v: k for k, v in _torch_to_thunder_function_map.items()}

# Factory functions whose value we know.
TENSOR_FACTORY = (
    thunder.torch.tensor.id,
    thunder.torch.ones.id,
    thunder.torch.zeros.id,
)


def trace_from_bsym_or_bsyms(bsym_or_bsyms: BoundSymbol | Sequence[BoundSymbol]) -> TraceCtx:
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
        # NOTE: Give prefix `tmp` to avoid infinite recursion due to the same name
        trace._siginfo = SigInfo.from_name_and_args(f"tmp_{trace_name}", trace.args)

    def add_proxy_name_to_trace(bsym):
        for p in bsym.flat_proxy_args:
            trace.names.add(p.name)

        for p in bsym.flat_proxy_outs:
            trace.names.add(p.name)

    for bsym in bsyms:
        add_proxy_name_to_trace(bsym)

    return trace


def make_trace_executable(trace_to_convert: TraceCtx):
    torchex_trace = transform_for_execution(trace_to_convert, executors_list=(pytorch_ex, pythonex_ex))
    trace_callable = torchex_trace.python_callable(include_decorators=False)
    return trace_callable


def compute_with_constant_tensors(bsym, const_values) -> None | Any:
    """
    This function is used to compute the concrete output of the computation
    represented by BoundSymbol if it's inputs are known to be constant.

    To run the computation, it will use PyTorch eager functions
    from _torch_to_thunder_function_map or registered operator from
    `pythonex` executor.
    """

    def materialize_args(a):
        if isinstance(a, (TensorProxy, NumberProxy)):
            return const_values[variableify(a)]
        elif isinstance(a, NumberProxy):
            return a.value
        return a

    new_args = tuple(map(materialize_args, bsym.args))
    new_kwargs = {k: materialize_args(v) for k, v in bsym.kwargs.items()}

    trace = trace_from_bsym_or_bsyms(bsym)
    callable_from_trace = make_trace_executable(trace)
    flat_args, _ = tree_flatten((new_args, new_kwargs))
    return callable_from_trace(*flat_args)


class ConstantFolding(thunder.Transform):
    """Apply Constant Folding to computation trace.

    With this transform applied to a computation trace, successive passes
    (meaning trace transformations) can transform the simplified compute.


    .. code-block:: python
        :name: example-constant_folding

        from thunder.transforms import ConstantFolding

        model = ...
        transforms = [ConstantFolding()]
        jitted = thunder.jit(model, transforms=transforms)
        # If you prefer `ThunderCompiler`...
        from thunder.dynamo import ThunderCompiler
        backend = ThunderCompiler(transforms=transforms)
        jitted = torch.compile(model, backend=backend)


    To see the effect of this transform, let's use the following function:

    .. code-block:: python

        def forward(x):
            scale_t = torch.tensor([2.])
            scale_t = (scale_t * 10) / 5
            return x * scale_t

    The initial computation trace is as follows:

    .. code-block:: python

        def computation(x):
          # x: "cpu f32[3]"

          scale_t = ltorch.tensor([2.0], device=None, dtype=None, requires_grad=False, pin_memory=False)  # scale_t: "cpu f32[1]"
            # scale_t = prims.tensor_from_sequence([2.0], dtype=None, device=devices.Device("cpu"))  # scale_t: "cpu f32[1]"

          t1 = ltorch.mul(scale_t, 10)  # t1: "cpu f32[1]"
            # _ = prims.convert_element_type(10, float)
            # t1 = prims.mul(scale_t, 10.0)  # t1: "cpu f32[1]"
          t2 = ltorch.true_divide(t1, 5)  # t2: "cpu f32[1]"
            # _ = prims.convert_element_type(5, float)
            # t2 = prims.div(t1, 5.0)  # t2: "cpu f32[1]"

          t4 = ltorch.mul(x, t2)  # t4: "cpu f32[3]"
            # t3 = prims.broadcast_in_dim(t2, (3,), (0,))  # t3: "cpu f32[3]"
            # t4 = prims.mul(x, t3)  # t4: "cpu f32[3]"
          return t4

    This transform simplifies this trace into

    .. code-block:: python

        def computation(x):
          # x: "cpu f32[3]"
          t2 = prims.tensor_from_sequence([4.0], dtype=dtypes.float32, device=devices.Device("cpu"))  # t2: "cpu f32[1]"

          t4 = ltorch.mul(x, t2)  # t4: "cpu f32[3]"
            # t3 = prims.broadcast_in_dim(t2, (3,), (0,))  # t3: "cpu f32[3]"
            # t4 = prims.mul(x, t3)  # t4: "cpu f32[3]"
          return {'output': t4, 'flat_args': [x]}

    """

    def transform_traces_pre_prologue(self, prologue_trc, computation_trc, epilogue_trc, **kwargs):
        # Create a new trace
        const_folded_trace = from_trace(computation_trc)
        const_folded_trace.bound_symbols = computation_trc.bound_symbols

        const_values: dict[Variable, torch.Tensor | Number] = {}

        # Tag output from factory functions as constant value.
        for bsym in const_folded_trace.bound_symbols:
            if bsym.sym.id in TENSOR_FACTORY:
                torch_fn = _thunder_to_torch_function_map[bsym.sym]
                t = torch_fn(*bsym.args, **bsym.kwargs)
                const_values[variableify(bsym.output)] = t

        new_bsyms = []
        symbol_to_last_used_variables = get_symbols_to_last_used_variables(const_folded_trace.bound_symbols, ignore=())

        def is_constant(proxy):
            if isinstance(proxy, TensorProxy) and variableify(proxy) in const_values:
                return True
            elif isinstance(proxy, NumberProxy) and variableify(proxy) in const_values:
                return True
            elif isinstance(proxy, NumberProxy) and proxy.is_static_constrained():
                return True
            return False

        const_number_swapmap = {}
        for bsym in const_folded_trace.bound_symbols:
            # If bsym has constant inputs, try to compute the output.
            if all(map(is_constant, bsym.flat_proxy_args)) and bsym.sym.id not in TENSOR_FACTORY:
                if bsym.sym.id in (prims.unpack_trivial.id,):
                    new_bsyms.append(bsym)
                    continue
                new_concrete_output = compute_with_constant_tensors(bsym, const_values)
                if bsym.sym.id == prims.python_return.id:
                    new_concrete_output = None
                if (
                    new_concrete_output is not None
                ):  # Might happen for `python_return` as it won't have mapping in `_thunder_to_torch_map`.
                    # Create a new symbol with same output proxy but which will now represent the computed constant value.
                    # eg.
                    # known_tensor = torch.tensor(2)
                    # t = known_tensor + 1 --> t = torch.tensor(3)

                    # For `ndim==0`, we need to use full as `tensor_from_sequence` expects
                    # a sequence (and not plain numbers).
                    if isinstance(new_concrete_output, Number):
                        const_number_swapmap[variableify(bsym.output)] = new_concrete_output
                        new_bsym = bsym
                    elif new_concrete_output.ndim == 0:
                        assert isinstance(new_concrete_output, torch.Tensor)
                        new_bsym = BoundSymbol(
                            thunder.prims.full,
                            args=(
                                new_concrete_output.shape,
                                new_concrete_output.tolist(),
                            ),
                            kwargs={
                                "dtype": to_dtype(new_concrete_output.dtype),
                                "device": to_device(new_concrete_output.device),
                            },
                            output=bsym.output,
                        )
                    else:
                        assert isinstance(new_concrete_output, torch.Tensor)
                        new_bsym = BoundSymbol(
                            thunder.prims.tensor_from_sequence,
                            args=(new_concrete_output.tolist(),),
                            kwargs={
                                "dtype": to_dtype(new_concrete_output.dtype),
                                "device": to_device(new_concrete_output.device),
                            },
                            output=bsym.output,
                        )
                    new_bsyms.append(new_bsym)

                    # Update const_tensors (so that usage of the output of this symbol will also be used for further computation.)
                    const_values[variableify(bsym.output)] = new_concrete_output

                    # Clear tensors which won't be used further.
                    for proxy_v in symbol_to_last_used_variables[bsym]:
                        const_values.pop(proxy_v, None)

                    continue

            # BoundSymbol with non-constant inputs, keep it as-is
            new_bsyms.append(bsym)

        # Update all input NumberProxies by constant numbers if possible.
        const_folded_trace.bound_symbols = [
            bsym.from_bsym_swap_proxies(const_number_swapmap, skip_output=True) for bsym in new_bsyms
        ]

        const_folded_trace.set_provenance("Constant Folding pass")
        return prologue_trc, const_folded_trace, epilogue_trc
