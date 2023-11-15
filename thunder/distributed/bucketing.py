from typing import TYPE_CHECKING
from thunder.core import devices
from thunder.core import dtypes
from thunder.core import utils
from thunder.core.proxies import TensorProxy, FutureTensorProxy
from thunder.distributed import prims as dist_prims

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def calc_bytes(tensor: TensorProxy) -> int:
    return tensor.numel * tensor.dtype.bytes


def calc_bucket_key(tensors: list[TensorProxy]) -> str:
    return "-".join(f"{t.name}_{t.dtype}_{t.device}_{t.shape}" for t in tensors)


class GradBucket:
    def __init__(
        self,
        index: int,
        grads: list[TensorProxy],
        grad_indices: list[int],
    ) -> None:
        self.index = index
        self.orig_grads = [g for g in grads]
        self.grad_names = [g.name for g in grads]
        self.bucket_key = calc_bucket_key(grads)
        self.grads = utils.ProxyDict()
        for i, g in enumerate(grads):
            self.grads[g] = i
        self.grad_indices = grad_indices
        self.grad_to_preaveraged = utils.ProxyDict()
        self.grad_to_index = utils.ProxyDict()
        for g, i in zip(grads, grad_indices):
            self.grad_to_index[g] = i
        self.preaveraged_tensors = []

    def tell(self, grad):
        utils.check(grad in self.grads, lambda: f"Wrong TensorProxy of {grad} passed")
        self.preaveraged_tensors.append(
            dist_prims.update_bucket_view(grad, self.grad_names, self.bucket_key),
        )

    def ready_for_allreduce(self):
        return len(self.preaveraged_tensors) == len(self.grads._dict)

    def pack(self):
        return dist_prims.pack(self.preaveraged_tensors, self.bucket_key)

    def __hash__(self):
        return self.index


class GradBuckets:
    def __init__(
        self,
        grad_to_bucket: utils.ProxyDict,
        bucket_list: list[GradBucket] = [],
        delay_allreduce: bool = False,
    ) -> None:
        self.grad_to_bucket = grad_to_bucket
        self.bucket_list = bucket_list
        self.delay_allreduce = delay_allreduce
        self.bucket_to_future: dict[GradBucket, FutureTensorProxy] = {}

    def _maybe_allreduce(self, bucket: GradBucket, group: "ProcessGroup") -> None:
        if bucket.ready_for_allreduce():
            self.bucket_to_future[bucket] = dist_prims.all_reduce(
                bucket.pack(),
                dist_prims.DistributedReduceOps.SUM,
                group=group,
                do_async=True,
            )

    def tell(self, grad: TensorProxy, group: "ProcessGroup") -> None:
        bucket = self.grad_to_bucket[grad]
        bucket.tell(grad)

        if not self.delay_allreduce:
            self._maybe_allreduce(bucket, group)

    def retrieve_allreduced_grads(self, group: "ProcessGroup"):
        if self.delay_allreduce:
            for bucket in self.bucket_list:
                self._maybe_allreduce(bucket, group)
        for bucket in filter(lambda b: b not in self.bucket_to_future, self.bucket_list):
            self._maybe_allreduce(bucket, group)
        allreduced_grads = {}
        for bucket, future in self.bucket_to_future.items():
            allreduced_bucket = dist_prims.wait(future)
            grads = bucket.orig_grads
            for grad, allreduced in zip(grads, dist_prims.unpack(allreduced_bucket, grads, bucket.bucket_key)):
                allreduced_grads[bucket.grad_to_index[grad]] = allreduced
        return allreduced_grads

    @staticmethod
    def build(
        *,
        gradients_of_same_dtype_and_device: dict[tuple[dtypes.dtype, devices.Device], list[TensorProxy]],
        gradient_to_index: utils.ProxyDict,
        bucket_cap_in_mb: float,
        delay_allreduce: bool,
    ) -> "GradBuckets":
        bucket_cap_bytes = bucket_cap_in_mb * 1024 * 1024

        n_buckets: int = 0
        grad_to_bucket = utils.ProxyDict()
        bucket_list: list[GradBucket] = []
        for grads in gradients_of_same_dtype_and_device.values():
            nbytes = 0
            cur_bucket_grads: list[TensorProxy] = []
            for grad in grads:
                nbytes += calc_bytes(grad)
                cur_bucket_grads.append(grad)
                if nbytes >= bucket_cap_bytes:
                    bucket = GradBucket(n_buckets, cur_bucket_grads, [gradient_to_index[g] for g in cur_bucket_grads])
                    for g in cur_bucket_grads:
                        grad_to_bucket[g] = bucket
                    nbytes = 0
                    cur_bucket_grads.clear()
                    n_buckets += 1
                    bucket_list.append(bucket)
            if cur_bucket_grads:
                bucket = GradBucket(n_buckets, cur_bucket_grads, [gradient_to_index[g] for g in cur_bucket_grads])
                for g in cur_bucket_grads:
                    grad_to_bucket[g] = bucket
                nbytes = 0
                cur_bucket_grads.clear()
                n_buckets += 1
                bucket_list.append(bucket)
        return GradBuckets(grad_to_bucket=grad_to_bucket, bucket_list=bucket_list, delay_allreduce=delay_allreduce)
