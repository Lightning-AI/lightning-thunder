from __future__ import annotations
from functools import partial
from functools import wraps
from typing import ClassVar, TYPE_CHECKING
import math
import os
import sys

import torch
import torch.nn as nn

from thunder.core import devices
from thunder.tests.framework import TorchExecutor, nvFuserExecutor
from thunder.tests.make_tensor import make_tensor

try:
    import expecttest
    import hypothesis
except ImportError:
    raise ImportError(
        "Required packages of `expecttest` and/or `hypothesis` are missing. "
        "Install them with `pip install expecttest hypothesis`"
    )


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


__all__ = [
    "create_per_process_dataloader",
    "distributed_wrapper",
    "executors_map",
    "new_gelu",
    "run_test_no_sync_grad_accumulation",
    "SmallModel",
    "ToyModel",
    "DataParallelTestCase",
]


executors_map = {
    TorchExecutor.name: TorchExecutor,
}
if nvFuserExecutor is not None:
    executors_map[nvFuserExecutor.name] = nvFuserExecutor


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ToyModel(nn.Module):
    """Linear(12, 12) -> gelu -> Linear(12, 8)."""

    N_IN: ClassVar[int] = 12
    N_HIDDEN: ClassVar[int] = 16
    N_OUT: ClassVar[int] = 8
    LAYER_NAMES: ClassVar[tuple[str, ...]] = ("net2", "net1")

    def __init__(self, bias: bool = True):
        super().__init__()
        self.net1 = nn.Linear(ToyModel.N_IN, ToyModel.N_HIDDEN, bias=bias)
        self.net2 = nn.Linear(ToyModel.N_HIDDEN, ToyModel.N_OUT, bias=bias)

    def forward(self, x):
        return self.net2(new_gelu(self.net1(x)))


# A simple map-style dataset design to support multiprocess testing
#   Creates a series of tensors on the CPU, and accepts a seed to ensure
#   tensor generation is consistent across processes
# See PyTorch's definition of a Dataset here:
#   https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataset.py#L41
# See the documentation for constructing a Dataset here:
#   https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset
# TODO Maybe in the future consider creating tensors on a device other than the CPU,
#   like the CUDA device associated with the process
# TODO Maybe a better name for this would something like SimpleSeededDataset?
class PerProcessDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples: int,
        tensor_shape: Sequence[int],
        tensor_dtype: torch.dtype,
        sample_seed: int | None = None,
    ):
        self._tensors = []

        device = torch.device("cpu")
        make = partial(make_tensor, tensor_shape, device=device, dtype=tensor_dtype, requires_grad=False)

        if sample_seed is not None:
            torch.manual_seed(sample_seed)

        for _ in range(num_samples):
            self._tensors.append(make())

        # Restores a random seed so that further random communications aren't synchronized
        #   across processes
        if sample_seed is not None:
            torch.seed()

    # __getitem__() and __len__() are the two methods required of PyTorch's
    #   Dataset interface
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._tensors[idx]

    def __len__(self) -> int:
        return len(self._tensors)


# TODO Update this to accept input shape and output shape parameters
class SmallModel(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.net1 = nn.Linear(2, 2, device=device, dtype=dtype)
        self.net2 = nn.Linear(2, 2, device=device, dtype=dtype)

    def forward(self, x):
        return self.net2(new_gelu(self.net1(x)))


if torch.distributed.is_available():
    from torch.testing._internal import common_distributed, common_utils

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
    class DistributedParallelTestCase(common_distributed.MultiProcessTestCase):
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

        @property
        def destroy_pg_upon_exit(self) -> bool:
            # Overriding base test class: do not auto destroy PG upon exit.
            return False

        @classmethod
        def _run(cls, rank, test_name, file_name, pipe, *, fake_pg=False):
            assert not fake_pg, "Not yet supported here..."

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
            try:
                self.run_test(test_name, pipe)
            except Exception:
                raise
            finally:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            sys.exit(0)

    # Configures PyTorch's default process group, must be called at the start of each
    #   distributed process
    def init_per_process_distributed(
        init_method: str, devicetype: devices.DeviceType, world_size: int, rank: int
    ) -> torch.distributed.ProcessGroup:
        backend: str
        if devicetype is devices.DeviceType.CUDA:
            backend = "nccl"
        elif devicetype is devices.DeviceType.CPU:
            backend = "gloo"
        else:
            raise ValueError(f"Unknown devicetype {devicetype}")

        torch.distributed.init_process_group(init_method=init_method, backend=backend, world_size=world_size, rank=rank)

        # NOTE _get_default_group is not a public PyTorch function, but there is no
        #   public mechanism to acquire the default process group, which is specified
        #   in operations by setting process_group=None.
        #   Actually acquiring the default ProcessGroup is not typically necessary, but
        #   thunder doesn't like to model primitives with implicit defaults,
        #   so we want to pass the ProcessGroup explicitly
        return torch.distributed.distributed_c10d._get_default_group()

    # Wraps a function so that it becomes one process of several executing the test
    #   See test_native_ddp and its helper _test_native_ddp_helper below for an example
    #   of how to use this wrapper.
    # NOTE This actually requires wrapping a stub, because the test framework manipulates
    #   functions in a way that does not allow them to be pickled.
    #   The actual logic must be implemented in a helper that can be pickled.
    # NOTE Tests wrapped with ddp_wrapper can be invoked directly, but you must invoke them
    #   like:
    #   if __name__ == '__main__':
    #       test_ddp.test_native_ddp_TorchEx_cpu_float32()
    class distributed_wrapper:
        def __init__(self, name, fn):
            self.fn = fn
            self.__name__ = name

        def __call__(self, test_stub):
            import multiprocessing as mp
            import tempfile
            import pytest

            if not torch.distributed.is_available():
                pytest.skip("This test requires torch.distributed be available")

            # Creates a temporary file for process group discovery
            FILE_SCHEMA: str = "file://"
            if sys.platform == "win32":
                FILE_SCHEMA = "file:///"
            file_name = tempfile.NamedTemporaryFile(delete=False).name
            init_method = f"{FILE_SCHEMA}{file_name}"

            @wraps(test_stub)
            def test_fn(executor, devices, dtype, **kwargs):
                world_size = len(devices)
                input_data = []

                for rank in range(world_size):
                    process_data = (init_method, world_size, rank, executor, devices[rank], dtype, kwargs)
                    input_data.append(process_data)

                ctx = mp.get_context("spawn")
                pool = ctx.Pool(world_size)

                def callback(result):
                    pass

                def error_callback(ex):
                    # NOTE: Don't raise the exception here, because it will be
                    # raised in the main process. Raising it here will cause a
                    # deadlock.
                    pass

                # The seconds to wait before the pool tasks complete
                TIMEOUT: int = 30
                try:
                    results_future = pool.map_async(self.fn, input_data, 1, callback, error_callback)
                    results = results_future.get(TIMEOUT)
                finally:
                    pool.close()
                    pool.join()

                # Raises the first assertion if any occurred
                root_results = results[0]
                if len(root_results) > 0:
                    raise (root_results[0])

            return test_fn

    # Creates a dataloader for a process
    #   If sample_seed is specified then the dataloader will load tensors with the same values
    #   on each process.
    #   If devicetype is specified and is CUDA, then the loaded tensors are placed on
    #   the CUDA device corresponding to the process's "rank"
    def create_per_process_dataloader(
        rank: int,
        num_samples: int,
        tensor_shape: Sequence[int],
        tensor_dtype: torch.dtype,
        sample_seed: int | None = None,
        *,
        devicetype: devices.DeviceType,
    ) -> torch.utils.data.DataLoader:
        dataset = PerProcessDataset(num_samples, tensor_shape, tensor_dtype, sample_seed=sample_seed)
        sampler = torch.utils.data.SequentialSampler(dataset)

        collate_fn = None
        if devicetype is not devices.DeviceType.CPU:
            assert devicetype is devices.DeviceType.CUDA, f"Unknown devicetype {devicetype}"
            device = torch.device("cuda", rank)

            def to_device(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
                return list([t.to(device) for t in tensors])

            collate_fn = to_device

        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, collate_fn=collate_fn)

        return dataloader

    def run_test_no_sync_grad_accumulation(
        test_case: DistributedParallelTestCase,
        get_model_and_optimizer: Callable[[torch.device], tuple[torch.nn.Module, torch.optim.Optimizer]],
        is_comm: Callable[[str], bool],
        dataset_size,
    ):
        from collections import defaultdict
        from contextlib import nullcontext
        from thunder.distributed import get_skip_data_parallel_grad_sync

        device = torch.device("cuda", test_case.rank)
        batch_size = 128
        num_micro_batch = 4
        micro_batch_size = batch_size // num_micro_batch
        with torch.no_grad():
            dataloader = [
                (torch.randn(batch_size, 12, device=device), torch.randn(batch_size, 8, device=device))
                for _ in range(dataset_size)
            ]

        # TODO(crcrpar): Use `last_traces` to check if allreduce was called, instead of `torch.profiler.profile`
        # See: https://github.com/Lightning-AI/lightning-thunder/pull/1881#issuecomment-1910455732
        def run_fwd_bwd(iter_count, model, x, y, num_grad_accum_steps: int | None = None):
            with torch.profiler.profile() as prof:
                pred = model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
                if num_grad_accum_steps is not None:
                    loss /= num_grad_accum_steps
                loss.backward()

            keys = tuple([e.key for e in prof.key_averages()])
            has_comms = any(is_comm(k) for k in keys)
            msg = f"{keys=}"
            if get_skip_data_parallel_grad_sync():
                test_case.assertFalse(has_comms, msg=msg)
            else:
                test_case.assertTrue(has_comms, msg=msg)

            return loss

        def get_ground_truth_loss_grads(device, dataloader):
            compiled_ddp_m, optimizer = get_model_and_optimizer(device)
            initial_state_dict = compiled_ddp_m.state_dict()

            losses, grads = [], []

            for iter_count, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                losses.append(run_fwd_bwd(iter_count, compiled_ddp_m, x, y, num_grad_accum_steps=None))
                grads.append([p.grad for p in compiled_ddp_m.parameters() if p.grad is not None])
                optimizer.step()

            return initial_state_dict, losses, grads

        device = torch.device("cuda", test_case.rank)
        initial_state_dict, ground_truth_losses, ground_truth_grads = get_ground_truth_loss_grads(device, dataloader)

        gradients = defaultdict(list)
        for use_no_sync in (True, False):
            jitted_model, optimizer = get_model_and_optimizer(device)
            jitted_model.load_state_dict(initial_state_dict)

            for iter_count, (x, y) in enumerate(dataloader):
                loss = torch.zeros((), device=device)
                with jitted_model.no_sync() if use_no_sync else nullcontext():
                    for i in range(num_micro_batch - 1):
                        cur_loss = run_fwd_bwd(
                            iter_count,
                            jitted_model,
                            x[i * micro_batch_size : (i + 1) * micro_batch_size, :],
                            y[i * micro_batch_size : (i + 1) * micro_batch_size, :],
                            num_micro_batch,
                        )
                        with torch.no_grad():
                            loss += cur_loss
                        if use_no_sync and i == 0 and iter_count == 0:
                            import thunder

                            # make sure the backward trace under `no_sync` has actual math computations.
                            no_sync_bwd_trc = thunder.last_backward_traces(jitted_model)[-1]
                            test_case.assertGreater(len(no_sync_bwd_trc.bound_symbols), 1)
                cur_loss = run_fwd_bwd(
                    iter_count, jitted_model, x[-micro_batch_size:, :], y[-micro_batch_size:, :], num_micro_batch
                )
                with torch.no_grad():
                    loss += cur_loss
                optimizer.step()
                gradients[use_no_sync].append([p.grad for p in jitted_model.parameters() if p.grad is not None])
                optimizer.zero_grad(set_to_none=True)

                num_expected_caches: int
                if use_no_sync:
                    num_expected_caches = 2
                else:
                    num_expected_caches = 1
                test_case.assertEqual(len(jitted_model._lc_cs.interpreter_cache), num_expected_caches)

                torch.testing.assert_close(loss, ground_truth_losses[iter_count], atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(
                    actual=gradients[use_no_sync][iter_count],
                    expected=ground_truth_grads[iter_count],
                    atol=5e-5,
                    rtol=5e-3,
                )
                if not use_no_sync:
                    torch.testing.assert_close(
                        actual=gradients[True][iter_count],
                        expected=gradients[False][iter_count],
                    )
