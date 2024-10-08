import thunder
from thunder.core.trace import from_trace
from thunder.core.proxies import variableify, Variable, TensorProxy, NumberProxy
from thunder.core.symbol import BoundSymbol
from thunder.core.dtypes import to_dtype
from thunder.core.devices import to_device
from thunder.torch import _torch_to_thunder_function_map
from thunder.core.utils import get_symbols_to_last_used_variables

_thunder_to_torch_function_map = {v: k for k, v in _torch_to_thunder_function_map.items()}

# Factory functions whose value we know.
TENSOR_FACTORY = (
    thunder.torch.tensor.id,
    thunder.torch.ones.id,
    thunder.torch.zeros.id,
)


def compute_with_concrete_tensor_and_pytorch_eager(bsym, const_values):

    def materialize_args(a):
        if isinstance(a, TensorProxy):
            return const_values[variableify(a)]
        elif isinstance(a, NumberProxy):
            return a.value
        return a

    new_args = tuple(map(materialize_args, bsym.args))
    new_kwargs = {k: materialize_args(v) for k, v in bsym.kwargs.items()}
    torch_fn = _thunder_to_torch_function_map.get(bsym.sym, None)
    if torch_fn is None:
        return
    return torch_fn(*new_args, **new_kwargs)


class ConstantFolding(thunder.Transform):
    def transform_traces_pre_prologue(self, prologue_trc, computation_trc, epilogue_trc, **kwargs):
        # Create a new trace
        const_folded_trace = from_trace(computation_trc)
        const_folded_trace.bound_symbols = computation_trc.bound_symbols

        const_tensors: dict[Variable, BoundSymbol] = {}

        # Tag output from factory functions as constant value.
        for bsym in const_folded_trace.bound_symbols:
            if bsym.sym.id in TENSOR_FACTORY:
                torch_fn = _thunder_to_torch_function_map[bsym.sym]
                t = torch_fn(*bsym.args, **bsym.kwargs)
                const_tensors[variableify(bsym.output)] = t

        new_bsyms = []
        symbol_to_last_used_variables = get_symbols_to_last_used_variables(const_folded_trace.bound_symbols, ignore=())

        def is_constant(proxy):
            if isinstance(proxy, TensorProxy) and variableify(proxy) in const_tensors:
                return True
            elif isinstance(proxy, NumberProxy) and proxy.is_static_constrained():
                return True
            return False

        for bsym in const_folded_trace.bound_symbols:
            # If bsym has constant inputs, try to compute the output.
            if all(map(is_constant, bsym.flat_proxy_args)) and bsym.sym.id not in TENSOR_FACTORY:
                if bsym.flat_args == []:  # eg, unpack_trivial
                    continue
                new_concrete_output = compute_with_concrete_tensor_and_pytorch_eager(bsym, const_tensors)
                if (
                    new_concrete_output is not None
                ):  # Might happen for `python_return` as it won't have mapping in `_thunder_to_torch_map`.

                    # Create a new symbol with same output proxy but which will now represent the computed constant value.
                    # eg.
                    # known_tensor = torch.tensor(2)
                    # t = known_tensor + 1 --> t = torch.tensor(3)

                    # For `ndim==0`, we need to use full as `tensor_from_sequence` expects
                    # a sequence (and not plain numbers).
                    if new_concrete_output.ndim == 0:
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
                    const_tensors[variableify(bsym.output)] = new_concrete_output

                    # Clear tensors which won't be used further.
                    for proxy_v in symbol_to_last_used_variables[bsym]:
                        const_tensors.pop(proxy_v, None)

                    continue

            # BoundSymbol with non-constant inputs, keep it as-is
            new_bsyms.append(bsym)

        del const_tensors

        const_folded_trace.bound_symbols = new_bsyms

        const_folded_trace.set_provenance("Constant Folding pass")
        return prologue_trc, const_folded_trace, epilogue_trc
