import math
import os
import sys

import torch
import torch.nn as nn

try:
    import expecttest
    import hypothesis
except ImportError:
    raise ImportError(
        "Required packages of `expecttest` and/or `hypothesis` are missing. "
        "Install them with `pip install expecttest hypothesis`"
    )
from torch.testing._internal import common_distributed, common_utils


__all__ = [
    "new_gelu",
    "ToyModel",
    "DataParallelTestCase",
]


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ToyModel(nn.Module):
    """Linear(12, 12) -> gelu -> Linear(12, 8)."""

    def __init__(self, bias: bool = True):
        super().__init__()
        self.net1 = nn.Linear(12, 12, bias=bias)
        self.net2 = nn.Linear(12, 8, bias=bias)

    def forward(self, x):
        return self.net2(new_gelu(self.net1(x)))


# note(crcrpar): How to write a test with `DDP`
# Just add a method to :class:`CompileDDPTest`. The class is responsible for
#     - calling `torch.distributed.init_process_group` with NCCL backend
#     - setting rank to each process group / device
# so what you'd need to do is to prepare a model and tensors, wrap the model with DDP, and
# `thunder.jit` the original model or the DDP'd model, and do some computation and/or
# examine the traces of the `thunder.jit`d.
# If you force a test to be run with >2 GPUs for a test, you might want to inherit `CompileDDPTest`
# and modify `world_size` to e.g. `max(torch.cuda.device_count(), 2)`.
# note(crcrpar): Why inheriting `common_distributed.MultiProcessTestCase`?
# When we're quite sure that we would only use `pytest` instead of `unittest`,
# IIUC it's possible to run a test that is dependent on `DistributedDataParallel` and/or
# `torch.distributed` by running the test file with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html),
# but I don't think (a) it's quite intuitive to require `torchrun` explicitly to run a test and
# (b) it's quite friendly to our CI as it's currently simply runs `pytest thunder/tests`.
# I would say it's feasible to write a test with `torch.distributed` by using `torch.multiprocessing`,
# but it would require us to make the function which defines the test logic picklable and would
# lead to boilerplate test functions.
# Ref: https://github.com/NVIDIA/apex/blob/7b2e71b0d4013f8e2f9f1c8dd21980ff1d76f1b6/apex/transformer/testing/distributed_test_base.py#L22
class DataParallelTestCase(common_distributed.MultiProcessTestCase):
    DISTRIBUTED_BACKEND = "nccl"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        torch.cuda.empty_cache()
        super().tearDown()

    # note(crcrpar): This means the world_size is up to two.
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @property
    def init_method(self):
        return f"{common_utils.FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        torch.distributed.init_process_group(
            init_method=self.init_method,
            backend=self.DISTRIBUTED_BACKEND,
            world_size=self.world_size,
            rank=self.rank,
        )

        local_rank = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)

        torch.distributed.barrier()
        self.run_test(test_name, pipe)
        torch.distributed.barrier()

        torch.distributed.destroy_process_group()
        sys.exit(0)
