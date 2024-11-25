from numbers import Number
from typing import Any
from collections.abc import Callable

import torch

import thunder
from thunder.core.trace import from_trace
from thunder.core.proxies import variableify, Variable, TensorProxy, NumberProxy
from thunder.core.symbol import BoundSymbol
from thunder.core.dtypes import to_dtype
from thunder.core.devices import to_device
from thunder.torch import _torch_to_thunder_function_map
from thunder.core.utils import get_symbols_to_last_used_variables


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


def get_python_operator(bsym) -> None | Callable:
    from thunder.executors.pythonex import ex as pythonex

    if pythonex.can_execute(bsym):
        # TODO - Is there a better way to do the same?
        # This seems brittle and tailored towards
        # current implementation of pythonex.
        impl = pythonex.implmap[bsym.sym.id]
        module = impl.symbol.module
        op = getattr(module, impl.symbol.id)
        return op
    return None


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

    # Try to see if the symbol is torch function
    torch_fn = _thunder_to_torch_function_map.get(bsym.sym, None)
    if torch_fn is not None:
        return torch_fn(*new_args, **new_kwargs)

    # Try to see if the symbol is a Python function
    python_fn = get_python_operator(bsym)
    if python_fn is not None:
        return python_fn(*new_args, **new_kwargs)
    return None


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
            getitem = scale_t[0]
            getitem_2 = torch.tensor([2., 3.])[0]
            return x + getitem + getitem_2

    The initial computation trace is as follows:

    .. code-block:: python

        def computation(x):
          # x: "cpu f32[3, 3]"
          scale_t = torch.tensor([2.0], device=torch.device("cpu"), dtype=None)  # scale_t: "cpu f32[1]"
            # scale_t = ltorch.tensor([2.0], device=torch.device("cpu"), dtype=None, requires_grad=False, pin_memory=False)  # scale_t: "cpu f32[1]"
              # scale_t = prims.tensor_from_sequence([2.0], dtype=None, device=devices.Device("cpu"))  # scale_t: "cpu f32[1]"
          t3 = torch.tensor([2.0, 3.0], device=torch.device("cpu"), dtype=None)  # t3: "cpu f32[2]"
            # t3 = ltorch.tensor([2.0, 3.0], device=torch.device("cpu"), dtype=None, requires_grad=False, pin_memory=False)  # t3: "cpu f32[2]"
              # t3 = prims.tensor_from_sequence([2.0, 3.0], dtype=None, device=devices.Device("cpu"))  # t3: "cpu f32[2]"
          getitem = Tensor.__getitem__(scale_t, 0)  # getitem: "cpu f32[]"
            # getitem = ltorch.getitem(scale_t, 0)  # getitem: "cpu f32[]"
              # (_,) = prims.shape(scale_t)
              # t17 = prims.slice_prim(scale_t, [0], [1], [1])  # t17: "cpu f32[1]"
              # getitem = prims.squeeze(t17, (0,))  # getitem: "cpu f32[]"
          del scale_t
          getitem_2 = Tensor.__getitem__(t3, 0)  # getitem_2: "cpu f32[]"
            # getitem_2 = ltorch.getitem(t3, 0)  # getitem_2: "cpu f32[]"
              # (_,) = prims.shape(t3)
              # t20 = prims.slice_prim(t3, [0], [1], [1])  # t20: "cpu f32[1]"
              # getitem_2 = prims.squeeze(t20, (0,))  # getitem_2: "cpu f32[]"
          del t3
          t6 = torch.add(x, getitem, alpha=1)  # t6: "cpu f32[3, 3]"
            # t6 = ltorch.add(x, getitem, alpha=1)  # t6: "cpu f32[3, 3]"
              # t6 = prims.add(x, getitem)  # t6: "cpu f32[3, 3]"
          del getitem
          t7 = torch.add(t6, getitem_2, alpha=1)  # t7: "cpu f32[3, 3]"
            # t7 = ltorch.add(t6, getitem_2, alpha=1)  # t7: "cpu f32[3, 3]"
              # t7 = prims.add(t6, getitem_2)  # t7: "cpu f32[3, 3]"
          del t6, getitem_2
          return t7

    This transform simplifies this trace into

    .. code-block:: python

        def computation(x):
          # x: "cpu f32[3, 3]"
          getitem = torch.full((), 2.0, device=torch.device("cpu"), dtype=torch.float32)  # getitem: "cpu f32[]"
            # getitem = ltorch.full((), 2.0, device=torch.device("cpu"), dtype=torch.float32)  # getitem: "cpu f32[]"
              # getitem = prims.full((), 2.0, device=devices.Device("cpu"), dtype=dtypes.float32)  # getitem: "cpu f32[]"
          t6 = torch.add(x, getitem, alpha=1)  # t6: "cpu f32[3, 3]"
            # t6 = ltorch.add(x, getitem, alpha=1)  # t6: "cpu f32[3, 3]"
              # t6 = prims.add(x, getitem)  # t6: "cpu f32[3, 3]"
          t7 = torch.add(t6, getitem, alpha=1)  # t7: "cpu f32[3, 3]"
            # t7 = ltorch.add(t6, getitem, alpha=1)  # t7: "cpu f32[3, 3]"
              # t7 = prims.add(t6, getitem)  # t7: "cpu f32[3, 3]"
          del t6, getitem
          return t7
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
                if bsym.flat_args == []:  # eg, unpack_trivial
                    continue
                new_concrete_output = compute_with_constant_tensors(bsym, const_values)
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
                        isinstance(new_concrete_output, torch.Tensor)
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
