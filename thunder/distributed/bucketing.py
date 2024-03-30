from __future__ import annotations
from typing import TYPE_CHECKING
from thunder.core import devices
from thunder.core import dtypes
from thunder.core import utils
from thunder.core.proxies import TensorProxy, FutureTensorProxy
from thunder.distributed import prims as dist_prims

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


__all__ = [
    "Bucket",
    "FSDPBackwardBucket",
    "GradBuckets",
]


def calc_bytes(tensor: TensorProxy) -> int:
    return tensor.numel * tensor.dtype.bytes


def name_bucket(tensors: list[TensorProxy]) -> str:
    return "-".join(f"{t.name}_{t.dtype}_{t.device}_{t.shape}" for t in tensors)


class Bucket:
    def __init__(
        self,
        index: int,
        tensors: list[TensorProxy],
        tensor_indices: list[int],
        bucket_name: str | None = None,
    ) -> None:
        self.index = index
        self.orig_tensors = [t for t in tensors]
        self.tensor_names = [t.name for t in tensors]
        self.name_passed = bucket_name is not None
        self.bucket_name = name_bucket(tensors) if bucket_name is None else bucket_name
        self.tensors = utils.ProxyDict()
        for i, t in enumerate(tensors):
            self.tensors[t] = i
        self.tensor_indices = tensor_indices
        self.tensor_before_after_bucket_update = utils.ProxyDict()
        self.tensor_to_index = utils.ProxyDict()
        for t, i in zip(tensors, tensor_indices):
            self.tensor_to_index[t] = i
        self.preaveraged_tensors = [None for _ in range(len(self.orig_tensors))]
        self._future: FutureTensorProxy | None = None
        self._future_set: bool = False
        self._param_produced_future: TensorProxy | None = None
        self._storage: TensorProxy | None = None

    @property
    def name(self) -> str:
        return self.bucket_name

    def tell(self, grad):
        utils.check(grad in self.tensors, lambda: f"Wrong TensorProxy of {grad} passed")
        index = self.tensor_names.index(grad.name)
        self.preaveraged_tensors[index] = dist_prims.update_bucket_view(grad, index, self.name)

    def recieved_all_tensors(self):
        return all(t is not None for t in self.preaveraged_tensors)

    def pack(self) -> TensorProxy:
        self.storage = dist_prims.pack(self.preaveraged_tensors, self.name)
        return self.storage

    @property
    def storage(self) -> TensorProxy | None:
        return self._storage

    @storage.setter
    def storage(self, tensor: TensorProxy) -> None:
        utils.check(self._storage is None, lambda: "pack is allready called")
        self._storage = tensor

    def __hash__(self):
        return hash(self.name + str(self.index))

    @property
    def future(self) -> FutureTensorProxy | None:
        return self._future

    def set_future(self, new_future: FutureTensorProxy, caller: TensorProxy) -> None:
        if self._future_set:
            msg = f"future is already set to {self.future=}"
            raise RuntimeError(msg)
        self._future = new_future
        self._future_set = True
        self._param_produced_future = caller

    def has_future(self) -> bool:
        return self._future is not None

    def __str__(self) -> str:
        return f"Bucket({self.index}, {self.orig_tensors}, {self.tensor_indices}, {self.name})"


class _FSDPBucket(Bucket):
    def __init__(
        self, index: int, tensors: list[TensorProxy], tensor_indices: list[int], bucket_name: str | None = None
    ) -> None:
        super().__init__(index, tensors, tensor_indices, bucket_name)

    def pack(self, world_size: int) -> TensorProxy:
        self.storage = dist_prims.pack_for_fsdp(self.orig_tensors, world_size, self.mode)
        return self.storage

    def tell(self, grad: TensorProxy):
        utils.check(grad in self.tensors, lambda: f"Wrong TensorProxy of {grad} passed")
        index = self.tensor_names.index(grad.name)
        self.preaveraged_tensors[index] = grad


class FSDPForwardBucket(_FSDPBucket):
    mode = "gather"


class FSDPBackwardBucket(_FSDPBucket):
    mode = "scatter"


class GradBuckets:
    def __init__(
        self,
        grad_to_bucket: utils.ProxyDict,
        bucket_list: list[Bucket] = [],
    ) -> None:
        self.grad_to_bucket = grad_to_bucket
        self.bucket_list = bucket_list
        self.bucket_to_future: dict[Bucket, FutureTensorProxy] = {}

    def _maybe_allreduce(self, bucket: Bucket, group: ProcessGroup) -> None:
        if bucket.recieved_all_tensors():
            self.bucket_to_future[bucket] = dist_prims.all_reduce(
                bucket.pack(),
                dist_prims.DistributedReduceOps.SUM,
                group=group,
                do_async=True,
                skip_clone=True,
            )

    def tell(self, grad: TensorProxy, group: ProcessGroup) -> None:
        bucket = self.grad_to_bucket[grad]
        bucket.tell(grad)

        self._maybe_allreduce(bucket, group)

    def retrieve_allreduced_grads(self, group: ProcessGroup):
        for bucket in filter(lambda b: b not in self.bucket_to_future, self.bucket_list):
            self._maybe_allreduce(bucket, group)
        allreduced_grads = {}
        for bucket, future in self.bucket_to_future.items():
            allreduced_bucket = dist_prims.wait(future)
            grads = bucket.orig_tensors
            for grad, allreduced in zip(grads, dist_prims.unpack(allreduced_bucket, grads, bucket.bucket_name)):
                allreduced_grads[bucket.tensor_to_index[grad]] = allreduced
        return allreduced_grads

    @staticmethod
    def build(
        *,
        gradients_of_same_dtype_and_device: dict[tuple[dtypes.dtype, devices.Device], list[TensorProxy]],
        gradient_to_index: utils.ProxyDict,
        bucket_cap_in_mb: float,
    ) -> GradBuckets:
        bucket_cap_bytes = bucket_cap_in_mb * 1024 * 1024

        n_buckets: int = 0
        grad_to_bucket = utils.ProxyDict()
        bucket_list: list[Bucket] = []
        for grads in gradients_of_same_dtype_and_device.values():
            nbytes = 0
            cur_bucket_grads: list[TensorProxy] = []
            for grad in grads:
                nbytes += calc_bytes(grad)
                cur_bucket_grads.append(grad)
                if nbytes >= bucket_cap_bytes:
                    bucket = Bucket(n_buckets, cur_bucket_grads, [gradient_to_index[g] for g in cur_bucket_grads])
                    for g in cur_bucket_grads:
                        grad_to_bucket[g] = bucket
                    nbytes = 0
                    cur_bucket_grads.clear()
                    n_buckets += 1
                    bucket_list.append(bucket)
            if cur_bucket_grads:
                bucket = Bucket(n_buckets, cur_bucket_grads, [gradient_to_index[g] for g in cur_bucket_grads])
                for g in cur_bucket_grads:
                    grad_to_bucket[g] = bucket
                nbytes = 0
                cur_bucket_grads.clear()
                n_buckets += 1
                bucket_list.append(bucket)
        return GradBuckets(grad_to_bucket=grad_to_bucket, bucket_list=bucket_list)
