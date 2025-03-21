from thunder.core.prims import make_prim, OpTags
from thunder.core import prims
from thunder.core.proxies import AnyProxy, TensorProxy
from thunder.executors.torchex import ex as pytorchex
from thunder.executors.pythonex import ex as pythonex
from thunder.core import baseutils
from thunder.core import utils
from thunder import clang
from thunder.core.transforms import register_grad, get_grad, put_grad


def _dtensor_inner_tensor(t):
    # Returning the `t._local_tensor` directly might cause problem with
    # (IIRC) DCE.
    return TensorProxy(like=t._local_tensor)


get_dtensor_inner_tensor = make_prim("get_dtensor_inner_tensor", "get_dtensor_inner_tensor", meta=_dtensor_inner_tensor)


def _construct_dtensor(local_tensor_proxy, spec):
    from thunder.torch.experimental.dtensor_proxy import DTensorProxy  # , proxy

    spec_proxy = AnyProxy(spec)
    # spec_proxy = spec
    # This should call unflatten
    return DTensorProxy(
        local_tensor_proxy=local_tensor_proxy,
        spec=spec_proxy,
        shape=tuple(spec.shape),
        device=local_tensor_proxy.device,
        dtype=local_tensor_proxy.dtype,
        requires_grad=local_tensor_proxy.requires_grad,
        grad=None,
        distparallel_type=None,
        thunder_fsdp_padding_size=None,
    )


construct_dtensor = make_prim("construct_dtensor", "construct_dtensor", meta=_construct_dtensor)


def _get_dtensor_inner_tensor(t):
    return t.to_local()


_get_dtensor_inner_tensor_impl = pytorchex.register_operator(
    "get_dtensor_inner_tensor", like=get_dtensor_inner_tensor, fn=_get_dtensor_inner_tensor
)
pytorchex.register_implementation(get_dtensor_inner_tensor, _get_dtensor_inner_tensor_impl)


def _construct_dtensor(t, metadata):
    from torch.distributed._tensor import DTensor

    return DTensor.from_local(t, metadata.mesh, metadata.placements)


_construct_dtensor_impl = pytorchex.register_operator(
    "construct_dtensor", like=construct_dtensor, fn=_construct_dtensor
)
pytorchex.register_implementation(construct_dtensor, _construct_dtensor_impl)


def _check_dtensor_spec_repr_meta(s: AnyProxy, value: str) -> None:
    baseutils.check_type(s, AnyProxy)


check_dtensor_spec_repr = make_prim(
    "check_dtensor_spec_repr",
    "check_dtensor_spec_repr",
    meta=_check_dtensor_spec_repr_meta,
    tags=(OpTags.DONT_DCE,),
)


def _check_dtensor_spec_repr(s: object, value: str) -> None:
    utils.check(repr(s) == value, lambda: f"Expected '{s} to be equal to '{value}")


_check_python_repr_impl = pythonex.register_operator(
    "check_python_repr", like=check_dtensor_spec_repr, fn=_check_dtensor_spec_repr
)

pythonex.register_implementation(check_dtensor_spec_repr, _check_python_repr_impl)


def handle_check_dtensor_spec_in_prologue(prim, prologue_trace, args) -> bool:
    if prim == check_dtensor_spec_repr:
        # How does torch.compile guard for this?
        a = args[0]
        o = AnyProxy(None, prefix="dtensor_spec")
        bsym = prims.unpack_attr.bind(a, "_spec", output=o)
        prologue_trace.bound_symbols.append(bsym)
        check_dtensor_spec_repr(o, repr(args[1]))

        # Also adds metadata check for _local_tensor
        t = TensorProxy(like=a._local_tensor, requires_grad=a._local_tensor.requires_grad)
        bsym = prims.unpack_attr.bind(a, "_local_tensor", output=t)
        prologue_trace.bound_symbols.append(bsym)
        clang.check_tensor_shape_and_metadata(t)
        return True

    return False


# Grad Rules
def get_dtensor_inner_tensor_grad_fn(dtensor):
    local_t = get_dtensor_inner_tensor(dtensor)
    g_local = get_grad(local_t)
    put_grad(dtensor, construct_dtensor(g_local, dtensor._spec._o))
    return local_t


register_grad(get_dtensor_inner_tensor, get_dtensor_inner_tensor_grad_fn)


def construct_dtensor_gradfn(local_tensor, spec):
    dtensor = construct_dtensor(local_tensor, spec)
    g_dtensor = get_grad(dtensor)
    put_grad(local_tensor, get_dtensor_inner_tensor(g_dtensor))
    return dtensor


register_grad(construct_dtensor, construct_dtensor_gradfn)
