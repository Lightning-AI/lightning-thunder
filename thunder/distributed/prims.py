from __future__ import annotations
from enum import auto, Enum
from numbers import Number
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import torch.distributed

import thunder.core.utils as utils
from thunder.core.prims import make_prim

from thunder.core.proxies import DistParallelType, FutureTensorProxy, pytype, TensorProxy
from thunder.core.transforms import register_augmented_forward, register_backward
from thunder.distributed import get_skip_data_parallel_grad_sync

if TYPE_CHECKING:
    from thunder.common import CompileData
    from thunder.distributed.tensor_parallel.common import TensorParallelLayerType


class PrimIDs(Enum):
    # Distributed prims (Experimental!)
    ALL_GATHER = auto()
    ALL_REDUCE = auto()
    BROADCAST = auto()
    REDUCE_SCATTER = auto()
    SYNCHRONIZE = auto()
    WAIT = auto()
    PACK = auto()
    UNPACK = auto()
    UNPACK_FOR_FSDP = auto()
    UPDATE_BUCKET_VIEW = auto()
    PACK_FOR_FSDP = auto()
    STASH_GRAD_FOR_FSDP = auto()

    SYNCHRONIZE_TENSOR_PARALLEL_OUTPUT = auto()
    SYNCHRONIZE_TENSOR_PARALLEL_INPUT = auto()


# This enum describes what all_reduce (below) will actually do
#   These operations are performed elementwise on all the "versions" of
#   the tensor across processes.
class DistributedReduceOps(Enum):
    SUM = auto()
    AVG = auto()
    PRODUCT = auto()
    MIN = auto()
    MAX = auto()
    BAND = auto()
    BOR = auto()
    BXOR = auto()
    # PREMUL_SUM = auto()


def check_if_distributed_available() -> None:
    utils.check(
        torch.distributed.is_available(),
        lambda: f"PyTorch distributed is not available, {torch.distributed.is_available()=}",
    )


def all_gather_meta(
    a: TensorProxy,
    /,
    group: torch.distributed.ProcessGroup,
    do_async: Number,
    dim: int | None = None,
) -> TensorProxy:
    check_if_distributed_available()
    utils.check_type(a, TensorProxy)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")

    if dim is not None:
        utils.check_type(dim, int)
        utils.check(dim >= 0 and dim < a.ndim, lambda: f"dim must satisfy 0 <= {dim=} < {a.ndim=}")
        result_shape = list(a.shape)
        result_shape[dim] *= group.size()
    else:
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
    a: TensorProxy,
    /,
    op: DistributedReduceOps,
    group: torch.distributed.ProcessGroup,
    do_async: Number,
    skip_clone: bool = False,
) -> TensorProxy | FutureTensorProxy:
    check_if_distributed_available()
    utils.check_type(a, TensorProxy)
    utils.check_type(op, DistributedReduceOps)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")
    utils.check_type(skip_clone, bool)

    if do_async:
        return FutureTensorProxy(like=a)

    return TensorProxy(like=a)


def broadcast_meta(
    a: TensorProxy, /, root: int, group: torch.distributed.ProcessGroup, do_async: Number
) -> TensorProxy:
    check_if_distributed_available()
    utils.check_type(a, TensorProxy)
    utils.check_type(root, int)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")

    if do_async:
        return FutureTensorProxy(like=a)

    return TensorProxy(like=a)


def reduce_scatter(
    a: TensorProxy,
    /,
    op: DistributedReduceOps,
    group: torch.distributed.ProcessGroup,
    do_async: Number,
    dim: int | None = None,
) -> TensorProxy:
    check_if_distributed_available()
    utils.check_type(a, TensorProxy)
    utils.check_type(op, DistributedReduceOps)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check(pytype(do_async) is bool, lambda: f"Expected {do_async=} to be a boolean value")

    result_shape = list(a.shape)
    if dim is not None:
        utils.check_type(dim, int)
        utils.check(dim >= 0 and dim < a.ndim, lambda: f"dim must satisfy 0 <= {dim=} < {a.ndim=}")
        utils.check(
            a.shape[dim] % group.size() == 0, lambda: f"Expected {a.shape[dim]=} to be divisible by {group.size()=}"
        )
        result_shape[dim] //= group.size()
    else:
        result_shape[0] //= group.size()
        utils.check(
            a.shape[0] % group.size() == 0, lambda: f"Expected {a.shape[0]=} to be divisible by {group.size()=}"
        )

    if do_async:
        return FutureTensorProxy(shape=result_shape, like=a)

    return TensorProxy(shape=result_shape, like=a)


# NOTE This is a very particular implementation of wait that may need to be
#   generalized in the future
def wait_meta(a: FutureTensorProxy, /) -> TensorProxy:
    check_if_distributed_available()
    utils.check_type(a, FutureTensorProxy)

    return TensorProxy(like=a)


def synchronize_meta(
    a: TensorProxy,
    /,
    group: torch.distributed.ProcessGroup,
) -> TensorProxy:
    utils.check_type(a, TensorProxy)
    utils.check_type(group, torch.distributed.ProcessGroup)

    match a.distparallel_type:
        case DistParallelType.REPLICATED:
            return TensorProxy(like=a)
        case DistParallelType.FULLY_SHARDED:
            # Assuming that the sharding is done on the first dimension
            # See [FSDP Sharding] in distributed/__init__.py
            unsharded_shape = a.shape[0] * group.size(), *a.shape[1:]
            return TensorProxy(shape=unsharded_shape, like=a, distparallel_type=DistParallelType.REPLICATED)
        case _:
            utils.check(False, lambda: f"Proxy {a} has unexpected {a.distparallel_type=}")


def pack_meta(
    tensors: list[TensorProxy],
    bucket_key: str,
) -> TensorProxy:
    utils.check(len(tensors) > 0, lambda: "Empty list is not expected")
    utils.check(
        all(isinstance(t, TensorProxy) for t in tensors),
        lambda: f"Every element of `tensors` must be TensorProxy but {[type(a) for a in tensors]}",
    )
    utils.check_same_dtype(*tensors)
    utils.check_same_device(*tensors)
    return TensorProxy(
        shape=(sum(t.numel for t in tensors),),
        device=tensors[0].device,
        dtype=tensors[0].dtype,
        requires_grad=False,
    )


def unpack_meta(buffer: TensorProxy, tensors: list[TensorProxy], bucket_key: str) -> list[TensorProxy]:
    utils.check(len(tensors) > 0, lambda: "Empty list is not expected")
    utils.check(
        all(isinstance(t, TensorProxy) for t in tensors),
        lambda: f"Every element of `tensors` must be TensorProxy but {[type(a) for a in tensors]}",
    )
    utils.check_same_dtype(buffer, *tensors)
    utils.check_same_device(buffer, *tensors)
    return [TensorProxy(like=t) for t in tensors]


def pack_for_fsdp_meta(
    tensors: list[TensorProxy],
    world_size: int,
    mode: str,
) -> TensorProxy:
    _supported = ("gather", "scatter")
    utils.check(mode in _supported, lambda: f"{mode=} is not supported: {_supported}")
    if world_size is not None:
        utils.check(
            isinstance(world_size, int) and world_size > 0,
            lambda: f"{world_size=} is supposed to be either {None=} or int",
        )
    return pack_meta(tensors, "")


def unpack_for_fsdp_meta(
    buffer: TensorProxy,
    tensors: list[TensorProxy],
    world_size: int,
    mode: str,
) -> list[TensorProxy]:
    utils.check(len(tensors) > 0, lambda: "Empty list is not expected")
    utils.check(
        all(isinstance(t, TensorProxy) for t in tensors),
        lambda: f"Every element of `tensors` must be TensorProxy but {[type(a) for a in tensors]}",
    )
    utils.check_same_dtype(buffer, *tensors)
    utils.check_same_device(buffer, *tensors)

    # TODO(crcrpar): Make `mode` Enum
    match mode:
        case "gather":
            utils.check(
                buffer.numel == sum(t.numel for t in tensors) * world_size,
                lambda: f"{buffer.numel=}, but {sum(t.numel for t in tensors) * world_size = }",
            )
        case "scatter":
            utils.check(
                buffer.numel == sum(t.numel for t in tensors) // world_size,
                lambda: f"{buffer.numel=}, but {sum(t.numel for t in tensors) // world_size = }",
            )
        case _:
            utils.check(False, lambda: f"Invalid {mode=}, `gather` and `scatter` are supported")

    result = []
    for sharded_param in tensors:
        shape = list(sharded_param.shape)
        shape[0] = int(shape[0] * world_size if mode == "gather" else shape[0] / world_size)
        result.append(TensorProxy(like=sharded_param, shape=tuple(shape)))
    return result


def update_bucket_view_meta(tensor: TensorProxy, index_of_dst_view: int, bucket_key: str) -> TensorProxy:
    return TensorProxy(like=tensor)


# [NOTE - shape of output]
# `ThunderFunction.backward` replaces outputs of this function with None, so the shape wouldn't matter a lot.
# TODO(crcrpar): Update this to return `None`
def stash_grad_for_fsdp_meta(
    grad: TensorProxy,
    param_fqn: str,
    compile_data: CompileData,
) -> TensorProxy:
    from thunder.common import CompileData

    utils.check_type(grad, TensorProxy)
    utils.check_type(param_fqn, str)
    utils.check_type(compile_data, CompileData)
    return TensorProxy(like=grad)


# see [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)'s Code 1.
def synchronize_tensor_parallel_output_meta(
    t: TensorProxy,
    group: torch.distributed.ProcessGroup,
    layer_type: TensorParallelLayerType,
) -> TensorProxy:
    from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

    utils.check_type(t, TensorProxy)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check_type(layer_type, TensorParallelLayerType)

    supported_ops = (
        TensorParallelLayerType.COLUMN_PARALLEL_EMBED,
        TensorParallelLayerType.COLUMN_PARALLEL_LINEAR,
        TensorParallelLayerType.ROW_PARALLEL_LINEAR,
        TensorParallelLayerType.ROW_PARALLEL_EMBED,
    )
    utils.check(
        layer_type in supported_ops,
        lambda: f"Unsupported {layer_type=}, supported ones are {supported_ops=}",
    )
    result_shape = list(t.shape)
    if layer_type in (TensorParallelLayerType.COLUMN_PARALLEL_LINEAR, TensorParallelLayerType.ROW_PARALLEL_EMBED):
        result_shape[-1] *= group.size()
    return TensorProxy(like=t, shape=result_shape)


def synchronize_tensor_parallel_input_meta(
    t: TensorProxy,
    group: torch.distributed.ProcessGroup,
    layer_type: TensorParallelLayerType,
) -> TensorProxy:
    from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

    utils.check_type(t, TensorProxy)
    utils.check_type(group, torch.distributed.ProcessGroup)
    utils.check_type(layer_type, TensorParallelLayerType)

    supported_ops = (
        TensorParallelLayerType.COLUMN_PARALLEL_LINEAR,
        TensorParallelLayerType.ROW_PARALLEL_LINEAR,
    )
    utils.check(
        layer_type in supported_ops,
        lambda: f"Unsupported {layer_type=}, supported ones are {supported_ops=}",
    )
    result_shape = list(t.shape)
    if layer_type == TensorParallelLayerType.ROW_PARALLEL_LINEAR:
        result_shape[-1] //= group.size()
    return TensorProxy(like=t, shape=result_shape)


all_gather = make_prim(PrimIDs.ALL_GATHER, "all_gather", meta=all_gather_meta)
all_reduce = make_prim(PrimIDs.ALL_REDUCE, "all_reduce", meta=all_reduce_meta)
broadcast = make_prim(PrimIDs.BROADCAST, "broadcast", meta=broadcast_meta)
reduce_scatter = make_prim(PrimIDs.REDUCE_SCATTER, "reduce_scatter", meta=reduce_scatter)
synchronize = make_prim(PrimIDs.SYNCHRONIZE, "synchronize", meta=synchronize_meta)
wait = make_prim(PrimIDs.WAIT, "wait", meta=wait_meta)
pack = make_prim(PrimIDs.PACK, "pack", meta=pack_meta)
pack_for_fsdp = make_prim(PrimIDs.PACK_FOR_FSDP, "pack_for_fsdp", meta=pack_for_fsdp_meta)
unpack = make_prim(PrimIDs.UNPACK, "unpack", meta=unpack_meta)
unpack_for_fsdp = make_prim(PrimIDs.UNPACK_FOR_FSDP, "unpack_for_fsdp", meta=unpack_for_fsdp_meta)
update_bucket_view = make_prim(PrimIDs.UPDATE_BUCKET_VIEW, "update_bucket_view", meta=update_bucket_view_meta)
stash_grad_for_fsdp = make_prim(
    PrimIDs.STASH_GRAD_FOR_FSDP,
    "stash_grad_for_fsdp",
    meta=stash_grad_for_fsdp_meta,
)
synchronize_tensor_parallel_output = make_prim(
    PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_OUTPUT,
    "synchronize_tensor_parallel_output",
    meta=synchronize_tensor_parallel_output_meta,
)
synchronize_tensor_parallel_input = make_prim(
    PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_INPUT,
    "synchronize_tensor_parallel_input",
    meta=synchronize_tensor_parallel_input_meta,
)


@register_augmented_forward(PrimIDs.SYNCHRONIZE)
def synchronize_augmented_forward_rule(
    a: TensorProxy,
    group: torch.distributed.ProcessGroup,
) -> tuple[TensorProxy, tuple]:
    match a.distparallel_type:
        case DistParallelType.REPLICATED:
            # Assuming that the input is a replicated tensor, so no need to do anything
            # in the forward pass
            return a, (
                a.distparallel_type,
                group,
            )
        case DistParallelType.FULLY_SHARDED:
            # Assuming that the sharding is done on the first dimension.
            # We do the communication on the side CUDA stream and wait is
            # immediately called on the result with the hope that the execution
            # passes would reorder the wait operation to be closer to the actual
            # usage of the tensor.
            return all_gather(a, group, True).wait(), (
                a.distparallel_type,
                group,
            )
        case _:
            utils.check(False, lambda: f"Proxy {a} has unexpected {a.distparallel_type=}")


@register_backward(PrimIDs.SYNCHRONIZE)
def synchronize_backward_rule(
    distparallel_type: DistParallelType,
    group: torch.distributed.ProcessGroup,
    grad: TensorProxy,
) -> tuple[TensorProxy, None]:
    if get_skip_data_parallel_grad_sync() and distparallel_type == DistParallelType.REPLICATED:
        return grad, None
    preaverage_grad = grad / group.size()
    match distparallel_type:
        case DistParallelType.REPLICATED:
            synced_grad = all_reduce(preaverage_grad, DistributedReduceOps.SUM, group, do_async=True).wait()
        case DistParallelType.FULLY_SHARDED:
            synced_grad = reduce_scatter(preaverage_grad, DistributedReduceOps.SUM, group, do_async=True).wait()
        case _:
            utils.check(False, lambda: f"synchronize with unexpected {distparallel_type=}")
    return synced_grad, None


class _TensorParallelOutputPostProcessFwdBwdInterface(ABC):

    @staticmethod
    @abstractmethod
    def forward(t: TensorProxy, group: torch.distributed.ProcessGroup) -> TensorProxy: ...

    @staticmethod
    @abstractmethod
    def backward(grad: TensorProxy, group: torch.distributed.ProcessGroup) -> TensorProxy: ...


class FwdGatherBwdSplitAlongLastDim(_TensorParallelOutputPostProcessFwdBwdInterface):

    @staticmethod
    def forward(t: TensorProxy, group: torch.distributed.ProcessGroup) -> TensorProxy:
        """Gather along last dim"""
        import thunder.torch as ltorch

        all_gathered = all_gather(t, group, True, 0).wait()
        chunked = ltorch.chunk(all_gathered, group.size(), 0)
        gathered = ltorch.cat(chunked, dim=-1)
        return gathered

    @staticmethod
    def backward(grad: TensorProxy, group: torch.distributed.ProcessGroup) -> TensorProxy:
        """Split along last dim"""
        from torch.distributed import distributed_c10d as c10d
        import thunder.torch as ltorch

        local_grad = ltorch.chunk(grad, c10d.get_world_size(group), dim=grad.ndim - 1)[c10d.get_rank(group)]
        return local_grad


class FwdAllReduceBwdIdentity(_TensorParallelOutputPostProcessFwdBwdInterface):

    @staticmethod
    def forward(t: TensorProxy, group: torch.distributed.ProcessGroup) -> TensorProxy:
        return all_reduce(t, DistributedReduceOps.SUM, group, do_async=True, skip_clone=True).wait()

    @staticmethod
    def backward(grad: TensorProxy, _: torch.distributed.ProcessGroup) -> TensorProxy:
        return grad


@register_augmented_forward(PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_OUTPUT)
def synchronize_tensor_parallel_output_forward_rule(
    t: TensorProxy,
    group: torch.distributed.ProcessGroup,
    layer_type: TensorParallelLayerType,
) -> tuple[TensorProxy, tuple[torch.distributed.ProcessGroup, TensorParallelLayerType]]:
    from thunder.distributed.tensor_parallel.common import TensorParallelLayerType
    import thunder.torch as ltorch

    match layer_type:
        case TensorParallelLayerType.COLUMN_PARALLEL_LINEAR:
            return FwdGatherBwdSplitAlongLastDim.forward(t, group), (group, layer_type)
        case TensorParallelLayerType.ROW_PARALLEL_LINEAR:
            return FwdAllReduceBwdIdentity.forward(t, group), (group, layer_type)
        case TensorParallelLayerType.COLUMN_PARALLEL_EMBED:
            return FwdAllReduceBwdIdentity.forward(t, group), (group, layer_type)
        case TensorParallelLayerType.ROW_PARALLEL_EMBED:
            return FwdGatherBwdSplitAlongLastDim.forward(t, group), (group, layer_type)
        case _:
            utils.check(False, lambda: f"Invalid {layer_type=}")


@register_backward(PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_OUTPUT)
def synchronize_tensor_parallel_output_backward_rule(
    group: torch.distributed.ProcessGroup,
    layer_type: TensorParallelLayerType,
    grad: TensorProxy,
) -> tuple[TensorProxy, None, None]:
    from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

    match layer_type:
        case TensorParallelLayerType.COLUMN_PARALLEL_LINEAR:
            return FwdGatherBwdSplitAlongLastDim.backward(grad, group), None, None
        case TensorParallelLayerType.ROW_PARALLEL_LINEAR:
            return FwdAllReduceBwdIdentity.backward(grad, group), None, None
        case TensorParallelLayerType.COLUMN_PARALLEL_EMBED:
            return FwdAllReduceBwdIdentity.backward(grad, group), None, None
        case TensorParallelLayerType.ROW_PARALLEL_EMBED:
            return FwdGatherBwdSplitAlongLastDim.backward(grad, group), None, None
        case _:
            utils.check(False, lambda: f"Invalid {layer_type=}")


@register_augmented_forward(PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_INPUT)
def synchronize_tensor_parallel_input_forward_rule(
    t: TensorProxy,
    group: torch.distributed.ProcessGroup,
    layer_type: TensorParallelLayerType,
) -> tuple[TensorProxy, tuple[torch.distributed.ProcessGroup, TensorParallelLayerType]]:
    from thunder.distributed.tensor_parallel.common import TensorParallelLayerType
    import thunder.torch as ltorch

    match layer_type:
        case TensorParallelLayerType.COLUMN_PARALLEL_LINEAR:
            return t, (group, layer_type)
        case TensorParallelLayerType.ROW_PARALLEL_LINEAR:
            from torch.distributed import distributed_c10d as c10d
            from thunder import clang

            chunk_size = t.shape[t.ndim - 1] // group.size()
            start_idx = chunk_size * c10d.get_rank(group)
            return clang.slice_in_dim(t, start_idx, start_idx + chunk_size, dim=t.ndim - 1), (group, layer_type)
        case _:
            utils.check(False, lambda: f"Invalid {layer_type=}")


@register_backward(PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_INPUT)
def synchronize_tensor_parallel_input_backward_rule(
    group: torch.distributed.ProcessGroup,
    layer_type: TensorParallelLayerType,
    grad: TensorProxy,
) -> tuple[TensorProxy, None, None]:
    from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

    match layer_type:
        case TensorParallelLayerType.COLUMN_PARALLEL_LINEAR:
            return (
                all_reduce(grad, DistributedReduceOps.SUM, group, do_async=True, skip_clone=True).wait(),
                None,
                None,
            )
        case TensorParallelLayerType.ROW_PARALLEL_LINEAR:
            import thunder.torch as ltorch

            all_gathered = all_gather(grad, group, True, 0).wait()
            chunked = ltorch.chunk(all_gathered, group.size(), 0)
            gathered_grad = ltorch.cat(chunked, dim=-1)
            return gathered_grad, None, None
        case _:
            utils.check(False, lambda: f"Invalid {layer_type=}")
