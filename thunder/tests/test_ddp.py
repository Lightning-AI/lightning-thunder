import math
import sys
import unittest

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import thunder
try:
    import expecttest  # noqa: F401
    import hypothesis  # noqa: F401
except ImportError:
    raise ImportError(
        "Required packages of `expecttest` and/or `hypothesis` are missing. "
        "Install them with `pip install expecttest hypothesis`"
    )
from torch.testing._internal import common_distributed, common_utils


# Compile - DDP tests
def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(new_gelu(self.net1(x)))


# note(crcrpar): How to write a test with `DDP`
# Just add a method to :class:`CompileDDPTest`. The class is responsible for
#     - calling `torch.distributed.init_process_group` with NCCL backend
#     - setting rank to each process group / device
# so what you'd need to do is to prepare a model and tensors, wrap the model with DDP, and
# `thunder.compile` the original model or the DDP'd model, and do some computation and/or
# examine the traces of the `thunder.compile`d.
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
@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "DDP test requires CUDA and NCCL `torch.distributed` backend",
)
class CompileDDPTest(common_distributed.MultiProcessTestCase):

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
            world_size=int(self.world_size),
            rank=self.rank,
        )

        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        torch.distributed.barrier()
        self.run_test(test_name, pipe)
        torch.distributed.barrier()

        torch.distributed.destroy_process_group()
        sys.exit(0)

    # Ref: https://github.com/Lightning-AI/lightning-thunder/issues/646
    def test_ddp_compile_module(self):
        model = ToyModel().to(self.rank)
        ddp_model = DDP(thunder.compile(model, use_generated_backward=True), device_ids=[self.rank])

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

        x, labels = torch.randn(20, 10).to(self.rank), torch.randn(20, 5).to(self.rank)

        init_loss, last_loss = None, None
        for i in range(3):
            optimizer.zero_grad()
            outputs = ddp_model(x)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if i == 0:
                init_loss = loss.detach().item()
            if i == 2:
                last_loss = loss.detach().item()
        assert init_loss > last_loss

    # Ref: https://github.com/Lightning-AI/lightning-thunder/issues/599
    def test_compile_ddp_module(self):
        model = ToyModel().to(self.rank)
        with self.assertRaisesRegex(
            RecursionError,
            "maximum recursion depth exceeded while calling a Python object",
        ):
            thunder.compile(DDP(model, device_ids=[self.rank]), use_generated_backward=True)


if __name__ == "__main__":
    common_utils.run_tests()
