import torch

from thunder.core.proxies import TensorProxy, NumberProxy
from thunder.core.devices import to_torch_device
from thunder.core.dtypes import to_torch_dtype
from thunder.core.pytree import tree_map
from thunder.core.trace import TraceCtx
from thunder.core.prims import PrimIDs
from thunder.core.symbol import Symbol
from thunder.core.trace import from_trace

from torch._subclasses.fake_tensor import FakeTensorMode

if torch.distributed.is_available():
    from torch.distributed.tensor import DTensor


def run_only_if_distributed_is_available(fn):
    def wrapper(*args, **kwargs):
        if torch.distributed.is_available():
            return fn(*args, **kwargs)

    return wrapper


def lazy_import_dtensor_proxy():
    from thunder.torch.experimental.dtensor_proxy import DTensorProxy

    return DTensorProxy


def run_with_fake_tensor(torch_op, *args, **kwargs):
    """
    Run a torch operation with fake tensors and return the output.

    Args:
        torch_op: The torch operation to execute
        *args: Arguments to pass to the torch operation
        **kwargs: Keyword arguments to pass to the torch operation

    Returns:
        The output of the torch operation executed with fake tensors
    """
    # To avoid cyclical dependency
    DTensorProxy = lazy_import_dtensor_proxy()

    def f(*args, **kwargs):
        return torch_op(*args, **kwargs)

    with FakeTensorMode():

        def materialize_fake_tensors(t):
            # Convert proxy types to fake tensors.
            if isinstance(t, NumberProxy):
                return t.value

            if not isinstance(t, TensorProxy):
                return t

            if isinstance(t, DTensorProxy):
                i_t = torch.randn(
                    t.local_tensor.shape,
                    device=to_torch_device(t.local_tensor.device),
                    dtype=to_torch_dtype(t.local_tensor.dtype),
                )
                return DTensor.from_local(i_t, t.spec._o.device_mesh, t.spec._o.placements)

            return torch.randn(t.shape, device=to_torch_device(t.device), dtype=to_torch_dtype(t.dtype))

        args, kwargs = tree_map(materialize_fake_tensors, (args, kwargs))

    return f(*args, **kwargs)


def check_dtensor_cotangent_metadata(dtensor, metadata):
    if not dtensor._spec == metadata:
        raise RuntimeError(
            "Metadata (placement and mesh) has changed for cotangent between tracing and runtime"
            f"during tracing it was {metadata} but at runtime it is {dtensor._spec}."
        )


def check_dtensor_cotangent_metadata_in_backward(bw_trace: TraceCtx):
    # NOTE: The metadata (placement and mesh) of the cotangent DTensor
    #       can be different at runtime than the one we assumed during tracing.
    #       Because of this, we currently add a check in backward to verify the same.
    #       However, in future, we should add a symbol which will take care of mapping
    #       the cotangent metadata at runtime to the cotangent metadata during tracing.
    #       Also refer: https://github.com/pytorch/pytorch/pull/118670

    # Quick implementation of a symbol to verify
    # that the metadata of the cotangent at runtime as that as during tracing.

    # To avoid cyclical dependency
    DTensorProxy = lazy_import_dtensor_proxy()

    check_dtensor_cotangent_metadata_symbol = Symbol(
        name="check_dtensor_cotangent_metadata",
        meta=lambda dtensor, metadata: None,
        python_impl=check_dtensor_cotangent_metadata,
    )
    new_bw_trace = from_trace(bw_trace)
    new_bsyms = []
    for bsym in bw_trace.bound_symbols:
        # Find the `unpack_sequence` for the cotangents.
        if bsym.sym.id == PrimIDs.UNPACK_SEQUENCE and bsym.args[0].name == "cotangents":
            new_bsyms.append(bsym)
            args = bsym.args[0].collection()
            for arg in args:
                # For every DTensor cotangent,
                # add symbol to verify that the metadata is the same as during tracing.
                if isinstance(arg, DTensorProxy):
                    bsym = check_dtensor_cotangent_metadata_symbol.bind(arg, arg.spec._o, output=None)
                    new_bsyms.append(bsym)
        else:
            new_bsyms.append(bsym)

    new_bw_trace.bound_symbols = new_bsyms

    return new_bw_trace
