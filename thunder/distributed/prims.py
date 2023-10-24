from enum import auto, Enum
from numbers import Number

import torch.distributed

import thunder.core.utils as utils
from thunder.core.prims import make_prim

from thunder.core.proxies import DDPType, FutureTensorProxy, pytype, TensorProxy
from thunder.core.transforms import register_augmented_forward, register_backward


class PrimIDs(Enum):
    # Distributed prims (Experimental!)
    ALL_GATHER = auto()
    ALL_REDUCE = auto()
    BROADCAST = auto()
    SYNCHRONIZE = auto()
    REDUCE_SCATTER = auto()
    WAIT = auto()


# This enum describes what all_reduce (below) will actually do
#   These operations are performed elementwise on all the "versions" of
#   the tensor across processes.
class DistributedReduceOps(Enum):
    SUM = auto()
    # AVG = auto()
    # PRODUCT = auto()
    # MIN = auto()
    # MAX = auto()
    # BAND = auto()
    # BOR = auto()
    # BXOR = auto()
    # PREMUL_SUM = auto()


def check_if_distributed_available() -> None:
    utils.check(
        torch.distributed.is_available(),
        lambda: f"PyTorch distributed is not available, {torch.distributed.is_available()=}",
    )


def all_gather_meta(a: TensorProxy, group: torch.distributed.ProcessGroup, do_async: Number) -> TensorProxy:
    check_if_distributed_available()
    utils.check_type(a, TensorProxy)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")

    # PyTorch's all_gather_into_tensor supports also other modes of gathering
    # but we only do concatenation on the first dimension for now
    result_shape = a.shape[0] * group.size(), *a.shape[1:]

    if do_async:
        return FutureTensorProxy(shape=result_shape, like=a)

    return TensorProxy(shape=result_shape, like=a)


# NOTE This is essentially a wrapper around
#   https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce
#   that models the operation as a functional one.
# TODO Support additional reduction operations
# TODO Consider our own distributed calls that don't just wrap PyTorch's
def all_reduce_meta(
    a: TensorProxy, op: DistributedReduceOps, group: torch.distributed.ProcessGroup, do_async: Number
) -> TensorProxy | FutureTensorProxy:
    check_if_distributed_available()
    utils.check_type(a, TensorProxy)
    utils.check_type(op, DistributedReduceOps)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")

    if do_async:
        return FutureTensorProxy(like=a)

    return TensorProxy(like=a)


def broadcast_meta(a: TensorProxy, root: int, group: torch.distributed.ProcessGroup, do_async: Number) -> TensorProxy:
    check_if_distributed_available()
    utils.check_type(a, TensorProxy)
    utils.check_type(root, int)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")

    if do_async:
        return FutureTensorProxy(like=a)

    return TensorProxy(like=a)


def reduce_scatter(
    a: TensorProxy, op: DistributedReduceOps, group: torch.distributed.ProcessGroup, do_async: Number
) -> TensorProxy:
    check_if_distributed_available()
    utils.check_type(a, TensorProxy)
    utils.check_type(op, DistributedReduceOps)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")

    utils.check(a.shape[0] % group.size() == 0, lambda: f"Expected {a.shape[0]=} to be divisible by {group.size()=}")

    # PyTorch's reduce_scatter_tensor supports also other modes of scattering
    # but we only do splitting on the first dimension for now
    result_shape = a.shape[0] // group.size(), *a.shape[1:]

    if do_async:
        return FutureTensorProxy(shape=result_shape, like=a)

    return TensorProxy(shape=result_shape, like=a)


# NOTE This is a very particular implementation of wait that may need to be
#   generalized in the future
def wait_meta(a: FutureTensorProxy) -> TensorProxy:
    check_if_distributed_available()
    utils.check_type(a, FutureTensorProxy)

    return TensorProxy(like=a)


def synchronize_meta(a: TensorProxy, group: torch.distributed.ProcessGroup) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(group, torch.distributed.ProcessGroup)
    return TensorProxy(like=a)


all_gather = make_prim(PrimIDs.ALL_GATHER, "all_gather", meta=all_gather_meta)
all_reduce = make_prim(PrimIDs.ALL_REDUCE, "all_reduce", meta=all_reduce_meta)
broadcast = make_prim(PrimIDs.BROADCAST, "broadcast", meta=broadcast_meta)
reduce_scatter = make_prim(PrimIDs.REDUCE_SCATTER, "reduce_scatter", meta=reduce_scatter)
synchronize = make_prim(PrimIDs.SYNCHRONIZE, "synchronize", meta=synchronize_meta)
wait = make_prim(PrimIDs.WAIT, "wait", meta=wait_meta)


@register_augmented_forward(PrimIDs.SYNCHRONIZE)
def synchronize_augmented_forward_rule(a: TensorProxy, group: torch.distributed.ProcessGroup) -> TensorProxy:
    utils.check(
        a.ddp_type == DDPType.REPLICATED,
        lambda: f"Expected {a.ddp_type=} to be {DDPType.REPLICATED=}",
    )
    # Assuming that the input is a replicated tensor, so no need to do anything
    # in the forward pass
    return a, (
        a.ddp_type,
        group,
    )


@register_backward(PrimIDs.SYNCHRONIZE)
def synchronize_backward_rule(
    ddp_type: DDPType, group: torch.distributed.ProcessGroup, grad: TensorProxy
) -> tuple[TensorProxy, None]:
    utils.check(
        ddp_type == DDPType.REPLICATED,
        lambda: f"Expected {ddp_type=} to be {DDPType.REPLICATED=}",
    )
    preaverage_grad = grad / group.size()
    all_reduced = all_reduce(preaverage_grad, DistributedReduceOps.SUM, group, do_async=True)
    return all_reduced, None
