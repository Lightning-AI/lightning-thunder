import math
import os
import sys
import unittest
from typing import Optional
from itertools import product

import pytest

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import distributed_c10d as c10d
from torch.testing import assert_close, make_tensor

import thunder
from thunder.tests.framework import TorchExecutor, nvFuserExecutor
import thunder.torch as ltorch
from thunder.distributed import prims

try:
    import expecttest  # noqa: F401
    import hypothesis  # noqa: F401
except ImportError:
    raise ImportError(
        "Required packages of `expecttest` and/or `hypothesis` are missing. "
        "Install them with `pip install expecttest hypothesis`"
    )
from torch.testing._internal import common_distributed, common_utils


executors_map = {
    TorchExecutor.name: TorchExecutor,
}
if nvFuserExecutor is not None:
    executors_map[nvFuserExecutor.name] = nvFuserExecutor


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
        ddp_model = DDP(thunder.compile(model), device_ids=[self.rank])

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
            RuntimeError,
            r"Unsupported instruction = ThunderInstruction\(opname='SETUP_WITH'",
        ):
            thunder.compile(DDP(model, device_ids=[self.rank]))

    # TODO(crcrpar): Mandate multiple GPUs so that the timing of collectives matters especially for
    # nvfuser executor.
    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_all_reduce(self, executor):
        _executor = executors_map[executor]

        # NOTE torch.distributed.all_reduce is an inplace operation
        def foo(
            a,
            b,
            op: Optional[torch.distributed.ReduceOp],
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
        ):
            c = a + b

            handle = None
            if op is not None:
                handle = torch.distributed.all_reduce(c, op, group=process_group, async_op=async_op)
            else:
                handle = torch.distributed.all_reduce(c, group=process_group, async_op=async_op)

            if async_op:
                handle.wait()

            e = c + 1
            return a, e

        # NOTE lightning.compiles all_reduce is a functional operation
        def lc_foo(
            a,
            b,
            op: Optional[torch.distributed.ReduceOp],
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
        ):
            c = a + b

            d = None
            if op is not None:
                d = ltorch.all_reduce(c, op, group=process_group, async_op=async_op)
            else:
                d = ltorch.all_reduce(c, group=process_group, async_op=async_op)

            if async_op:
                d = prims.wait(d)

            e = d + 1
            return a, e

        device = f"cuda:{self.rank}"
        a = make_tensor((2, 2), device=device, dtype=torch.float32)
        b = make_tensor((2, 2), device=device, dtype=torch.float32)
        process_group = c10d.new_group()

        # NOTE Preprocessing is disabled because we call thunder.torch operations directly
        cfoo = thunder.compile(lc_foo, executors_list=_executor.executors_list(), disable_preprocessing=True)

        for op, async_op in product((None, torch.distributed.ReduceOp.SUM), (False, True)):
            expected = foo(a, b, op, process_group, async_op)
            actual = cfoo(a, b, op, process_group, async_op)

            self.assertEqual(actual, expected)

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_all_gather(self, executor):
        _executor = executors_map[executor]

        # NOTE torch.distributed.all_gather is an inplace operation
        def foo(
            a,
            b,
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
        ):
            c = a + b
            d = torch.empty((c.shape[0] * process_group.size(), *c.shape[1:]), device=c.device, dtype=c.dtype)
            handle = torch.distributed.all_gather_into_tensor(d, c, group=process_group, async_op=async_op)

            if async_op:
                handle.wait()

            e = d + 1
            return a, e

        # NOTE thunder.torch.all_gather is a functional operation
        def lc_foo(
            a,
            b,
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
        ):
            c = a + b

            d = ltorch.all_gather(c, group=process_group, async_op=async_op)

            if async_op:
                d = prims.wait(d)

            e = d + 1
            return a, e

        device = f"cuda:{self.rank}"
        a = make_tensor((2, 2), device=device, dtype=torch.float32)
        b = make_tensor((2, 2), device=device, dtype=torch.float32)
        process_group = c10d.new_group()

        # NOTE Preprocessing is disabled because we call thunder.torch operations directly
        cfoo = thunder.compile(lc_foo, executors_list=_executor.executors_list(), disable_preprocessing=True)

        for async_op in (True, False):
            expected = foo(a, b, process_group, async_op)
            actual = cfoo(a, b, process_group, async_op)

            self.assertEqual(actual, expected)

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_broadcast(self, executor):
        _executor = executors_map[executor]

        # NOTE torch.distributed.broadcast is an inplace operation
        def foo(
            a,
            b,
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
        ):
            if process_group.rank() == 0:
                c = a + b
            else:
                c = a * b - 888.0

            handle = torch.distributed.broadcast(c, 0, group=process_group, async_op=async_op)

            if async_op:
                handle.wait()

            e = c + 1
            return a, e

        # NOTE thunder.torch.all_gather is a functional operation
        def lc_foo(
            a,
            b,
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
        ):
            if process_group.rank() == 0:
                c = a + b
            else:
                c = a * b + 888.0

            d = ltorch.broadcast(c, 0, group=process_group, async_op=async_op)

            if async_op:
                d = prims.wait(d)

            e = d + 1
            return a, e

        device = f"cuda:{self.rank}"
        a = make_tensor((2, 2), device=device, dtype=torch.float32)
        b = make_tensor((2, 2), device=device, dtype=torch.float32)
        process_group = c10d.new_group()

        # NOTE Preprocessing is disabled because we call thunder.torch operations directly
        cfoo = thunder.compile(lc_foo, executors_list=_executor.executors_list(), disable_preprocessing=True)

        for async_op in (True, False):
            expected = foo(a, b, process_group, async_op)
            actual = cfoo(a, b, process_group, async_op)

            self.assertEqual(actual, expected)

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_reduce_scatter(self, executor):
        _executor = executors_map[executor]

        # NOTE torch.distributed.all_gather is an inplace operation
        def foo(
            a,
            b,
            op,
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
        ):
            c = a + b
            d = torch.empty((c.shape[0] // process_group.size(), *c.shape[1:]), device=c.device, dtype=c.dtype)
            if op is not None:
                handle = torch.distributed.reduce_scatter_tensor(d, c, op, group=process_group, async_op=async_op)
            else:
                handle = torch.distributed.reduce_scatter_tensor(d, c, group=process_group, async_op=async_op)

            if async_op:
                handle.wait()

            e = d + 1
            return a, e

        # NOTE thunder.torch.all_gather is a functional operation
        def lc_foo(
            a,
            b,
            op,
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
        ):
            c = a + b

            d = ltorch.reduce_scatter(c, op, group=process_group, async_op=async_op)

            if async_op:
                d = prims.wait(d)

            e = d + 1
            return a, e

        device = f"cuda:{self.rank}"
        a = make_tensor((4, 2), device=device, dtype=torch.float32)
        b = make_tensor((4, 2), device=device, dtype=torch.float32)
        process_group = c10d.new_group()

        # NOTE Preprocessing is disabled because we call thunder.torch operations directly
        cfoo = thunder.compile(lc_foo, executors_list=_executor.executors_list(), disable_preprocessing=True)

        for op, async_op in product((None, torch.distributed.ReduceOp.SUM), (False, True)):
            expected = foo(a, b, op, process_group, async_op)
            actual = cfoo(a, b, op, process_group, async_op)

            self.assertEqual(actual, expected)


common_utils.instantiate_parametrized_tests(CompileDDPTest)

from thunder.tests.framework import instantiate
from thunder.core import dtypes
from thunder.core import devices
from thunder.distributed import ddp

import torch.distributed as tdist
import torch.utils.data as tudata

import os
import copy
from collections.abc import Sequence
from functools import partial, wraps
import tempfile
import multiprocessing as mp


# Configures PyTorch's default process group, must be called at the start of each
#   distributed process
def init_per_process_distributed(
    init_method: str, devicetype: devices.DeviceType, world_size: int, rank: int
) -> tdist.ProcessGroup:
    backend: str
    if devicetype is devices.DeviceType.CUDA:
        backend = "nccl"
    elif devicetype is devices.DeviceType.CPU:
        backend = "gloo"
    else:
        raise ValueError(f"Unknown devicetype {devicetype}")

    tdist.init_process_group(init_method=init_method, backend=backend, world_size=world_size, rank=rank)

    # NOTE _get_default_group is not a public PyTorch function, but there is no
    #   public mechanism to acquire the default process group, which is specified
    #   in operations by setting process_group=None.
    #   Actually acquiring the default ProcessGroup is not typically necessary, but
    #   lightning.compile doesn't like to model primitives with implicit defaults,
    #   so we want to pass the ProcessGroup explicitly
    return tdist.distributed_c10d._get_default_group()


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
        sample_seed: Optional[int] = None,
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
    sample_seed: Optional[int] = None,
    *,
    devicetype: devices.DeviceType,
) -> tudata.DataLoader:
    dataset = PerProcessDataset(num_samples, tensor_shape, tensor_dtype, sample_seed=sample_seed)
    sampler = tudata.SequentialSampler(dataset)

    collate_fn = None

    if devicetype is not devices.DeviceType.CPU:
        assert devicetype is devices.DeviceType.CUDA, f"Unknown devicetype {devicetype}"
        device = torch.device("cuda", rank)

        def to_device(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
            return list([t.to(device) for t in tensors])

        collate_fn = to_device

    dataloader = tudata.DataLoader(dataset, sampler=sampler, collate_fn=collate_fn)

    return dataloader


# TODO Update this to accept input shape and output shape parameters
class SmallModel(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.net1 = nn.Linear(2, 2, device=device, dtype=dtype)
        self.net2 = nn.Linear(2, 2, device=device, dtype=dtype)

    def forward(self, x):
        return self.net2(new_gelu(self.net1(x)))


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
class ddp_wrapper:
    def __init__(self, name, fn):
        self.fn = fn
        self.__name__ = name

    def __call__(self, test_stub):
        if not tdist.is_available():
            pytest.skip("This test requires torch.distributed be available")

        # Creates a temporary file for process group discovery
        FILE_SCHEMA: str = "file://"
        if sys.platform == "win32":
            FILE_SCHEMA = "file:///"
        file_name = tempfile.NamedTemporaryFile(delete=False).name
        init_method = f"{FILE_SCHEMA}{file_name}"

        @wraps(test_stub)
        def test_fn(executor, devices, dtype):
            world_size = len(devices)
            input_data = []

            for rank in range(world_size):
                process_data = (init_method, world_size, rank, executor, devices[rank], dtype)
                input_data.append(process_data)

            ctx = mp.get_context("spawn")
            pool = ctx.Pool(world_size)

            results: list = []

            def callback(result):
                pass

            def error_callback(ex):
                raise ex

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


# NOTE This assumes that one process will have rank=0 -- could generalize that to root
# TODO Test training, this test just currently tests forward
def _test_native_ddp_helper(input_data):
    init_method, world_size, rank, executor, device, dtype = input_data

    num_samples = 2
    tensor_shape = (2, 2)
    sample_seed = 3456
    num_epochs = 1
    devicetype = devices.device_from_string(device).devicetype
    torch_dtype = ltorch.to_torch_dtype(dtype)

    pg = init_per_process_distributed(init_method, devicetype, world_size, rank)
    tdist.barrier(pg)

    dataloader = create_per_process_dataloader(
        rank,
        num_samples=num_samples,
        tensor_shape=tensor_shape,
        tensor_dtype=torch_dtype,
        sample_seed=sample_seed,
        devicetype=devicetype,
    )

    # Creates, compiles, and DDPs the model
    model = SmallModel(device, torch_dtype)
    ddp_model = ddp(
        model,
        rank=rank,
        broadcast_from=0,
        process_group=pg,
    )
    cmodel = thunder.compile(
        ddp_model,
        executors_list=executor.executors_list(),
        use_static_caching=True,
    )

    comparison_exceptions = []
    for epoch in range(num_epochs):
        for step, data in enumerate(dataloader):
            (inp,) = data
            pred = cmodel(inp)

            # Validates that each process got the same result by gathering all the tensors
            #   on rank 0 and comparing them
            # NOTE Exceptions thrown during the comparison process are recorded and returned
            #   to the spawning process for analysis
            gather_list = None
            if rank == 0:
                gather_list = []
                for _ in range(world_size):
                    gather_list.append(torch.empty_like(pred))

            tdist.gather(pred, gather_list, dst=0, group=pg, async_op=False)

            if rank == 0:
                for other in gather_list:
                    try:
                        assert_close(pred, other)
                    except Exception as e:
                        comparison_exceptions.append(e)

            pred.mean().backward()

            grad_gather_list = None
            for param_with_grad in filter(lambda p: p.grad is not None, cmodel.parameters()):
                if rank == 0:
                    grad_gather_list = []
                    for _ in range(world_size):
                        grad_gather_list.append(torch.empty_like(param_with_grad))

                grad = param_with_grad.grad

                tdist.gather(grad, grad_gather_list, dst=0, group=pg, async_op=False)

                if rank == 0:
                    for other in grad_gather_list:
                        try:
                            assert_close(grad, other)
                        except Exception as e:
                            comparison_exceptions.append(e)

    # NOTE This function is undocumented; its definition is here:
    #   https://github.com/pytorch/pytorch/blob/416bf4e/torch/distributed/distributed_c10d.py#L1359
    tdist.barrier(pg)
    tdist.destroy_process_group(pg)

    if rank == 0:
        return comparison_exceptions

    return None


# NOTE This is just a stub, see the NOTE for ddp_wrapper
@instantiate(dtypes=(thunder.float32,), num_devices=2)
@ddp_wrapper("test_native_ddp", _test_native_ddp_helper)
def test_native_ddp(executor, devices, dtype):
    pass


if __name__ == "__main__":
    common_utils.run_tests()
