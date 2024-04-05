import math
import multiprocessing as mp
import os
import sys
import tempfile
import unittest
import weakref
from collections.abc import Sequence
from functools import partial, wraps
from itertools import product

import pytest
import torch
import torch.distributed as tdist
import torch.nn as nn
import torch.utils.data as tudata
from torch.distributed import distributed_c10d as c10d
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close, make_tensor

import thunder
import thunder.torch as ltorch
from thunder.core import devices
from thunder.distributed import FSDPBucketingStrategy, FSDPType
from thunder.distributed import ddp, fsdp
from thunder.distributed import prims
from thunder.tests.framework import TorchExecutor, nvFuserExecutor
from thunder.tests.framework import instantiate

from thunder.executors.transformer_engineex import transformer_engine_ex, TE_AVAILABLE

is_fp8_supported: bool = False
# This will be correctly updated below when TE Engine is installed
# and if the current environment doesn't support FP8.
fp8_support_reason: str = ""
if TE_AVAILABLE:
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch.fp8 import check_fp8_support

    is_fp8_supported, fp8_support_reason = check_fp8_support()

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
        self.net1 = nn.Linear(12, 12)
        self.net2 = nn.Linear(12, 8)

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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "DDP test requires CUDA and NCCL `torch.distributed` backend",
)
class CompileDDPTest(DataParallelTestCase):
    # Reference issue "Add an example of DDP(compile(model)) to tests"
    def test_ddp_compile_module(self):
        model = ToyModel().to(self.rank)
        ddp_model = DDP(thunder.jit(model, device_ids=[self.rank]))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

        x, labels = torch.randn(20, 12).to(self.rank), torch.randn(20, 8).to(self.rank)

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

    # Reference issue "[tracker] Support DistributedDataParallel"
    def test_compile_ddp_module(self):
        model = ToyModel().to(self.rank)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"DistributedDataParallel.*not supported",
        ):
            cm = thunder.jit(DDP(model, device_ids=[self.rank]))
            x = torch.randn(20, 12).to(self.rank)
            outputs = cm(x)

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_sort_waits(self, executor):
        from thunder.distributed.utils import sort_waits

        _executor = executors_map[executor]

        def func(
            a,
            b,
            process_group: torch.distributed.ProcessGroup,
        ):
            d = ltorch.all_reduce(a, group=process_group, async_op=True).wait()
            c = a + b
            e = c @ b + a
            return e, d

        cfunc = thunder.jit(func, executors=_executor.executors_list())
        device = f"cuda:{self.rank}"
        a = make_tensor((2, 2), device=device, dtype=torch.float32)
        b = make_tensor((2, 2), device=device, dtype=torch.float32)
        process_group = c10d.new_group()
        _ = cfunc(a, b, process_group)
        execution_trace = thunder.last_traces(cfunc)[-2]
        sorted_execution_trace = sort_waits(execution_trace)
        # assert that there is at least one node between the all_reduce and wait
        all_reduce_idx = sorted_execution_trace.bound_symbols.index(
            next(filter(lambda n: n.sym.name == "torch_all_reduce_prim_impl", execution_trace.bound_symbols))
        )
        wait_idx = sorted_execution_trace.bound_symbols.index(
            next(filter(lambda n: n.sym.name == "torch_wait_prim_impl", execution_trace.bound_symbols))
        )
        self.assertGreater(wait_idx - all_reduce_idx, 1)
        self.assertEqual(wait_idx, len(sorted_execution_trace.bound_symbols) - 2)

    # TODO(crcrpar): Mandate multiple GPUs so that the timing of collectives matters especially for
    # nvfuser executor.
    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_all_reduce(self, executor):
        _executor = executors_map[executor]

        # NOTE torch.distributed.all_reduce is an inplace operation
        def foo(
            a,
            b,
            op: torch.distributed.ReduceOp | None,
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

        # NOTE thunders all_reduce is a functional operation
        def lc_foo(
            a,
            b,
            op: torch.distributed.ReduceOp | None,
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
        cfoo = thunder.jit(lc_foo, executors=_executor.executors_list())

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
        cfoo = thunder.jit(lc_foo, executors=_executor.executors_list())

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
        cfoo = thunder.jit(lc_foo, executors=_executor.executors_list())

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
        cfoo = thunder.jit(lc_foo, executors=_executor.executors_list())

        for op, async_op in product((None, torch.distributed.ReduceOp.SUM), (False, True)):
            expected = foo(a, b, op, process_group, async_op)
            actual = cfoo(a, b, op, process_group, async_op)

            self.assertEqual(actual, expected)

    @common_utils.parametrize("executor,bucket_size_in_mb", product(tuple(executors_map.keys()), (0, 1000)))
    def test_ddp_grad_bucketing(self, executor, bucket_size_in_mb: int):
        from thunder.distributed import ddp
        from thunder.executors.torchex import (
            pack_prim_impl,
            unpack_prim_impl,
            update_bucket_view_prim_impl,
            all_reduce_prim_impl,
        )

        device = torch.device("cuda", self.rank)
        m = ToyModel().to(device)
        cm = thunder.jit(
            ddp(m, bucket_size_in_mb=bucket_size_in_mb),
            executors=executors_map[executor].executors_list(),
        )
        x = torch.ones((2, 12)).to(device)
        cm(x).mean().backward()

        bwd_extrace = thunder.last_backward_traces(cm)[-1]
        bsym_sym_id_list = [bsym.sym.id for bsym in bwd_extrace.bound_symbols]
        pack_syms = tuple(filter(lambda a: a == pack_prim_impl.id, bsym_sym_id_list))
        unpack_syms = tuple(filter(lambda a: a == unpack_prim_impl.id, bsym_sym_id_list))
        update_bucket_view_syms = tuple(filter(lambda a: a == update_bucket_view_prim_impl.id, bsym_sym_id_list))
        if bucket_size_in_mb == 0:
            self.assertEqual(len(pack_syms), 0)
            self.assertEqual(len(unpack_syms), 0)
            self.assertEqual(len(update_bucket_view_syms), 0)
            for bsym in bwd_extrace.bound_symbols:
                if bsym.sym.id == all_reduce_prim_impl.id:
                    # oh, everything is put into `bsym.args`?
                    msg = f"{bsym.args=}, {bsym.kwargs=}"
                    self.assertTrue(bsym.args[-1], msg=msg)
        else:
            self.assertEqual(len(pack_syms), 1, msg=f"{pack_syms}")
            self.assertEqual(len(unpack_syms), 1, msg=f"{unpack_syms}")
            self.assertEqual(len(update_bucket_view_syms), 4, msg=f"{update_bucket_view_prim_impl}")

    def test_rematerialize_all_gather(self):
        device = torch.device("cuda", self.rank)
        m = ToyModel().to(device)
        cm = thunder.jit(
            fsdp(m, device=device, broadcast_from=0),
        )
        x = torch.ones((2, 12), device=device)
        cm(x).mean().backward()

        fwd_trc = [
            t for t in thunder.last_traces(cm) if getattr(t.get_provenance(), "pss", "") == "Augmented forward pass"
        ][0]
        bwd_trc = thunder.last_backward_traces(cm)[0]
        from thunder.core.rematerialization import rematerialize_all_gather

        result_fwd_trc, result_bwd_trc = rematerialize_all_gather(fwd_trc, bwd_trc)

        # check the return statement in forward trace is updated
        # TODO: this is not stable w.r.t. details of the processing, the sharded correspond to ("t_net1_weight", "t_net2_weight")
        #       in the original trace and are inputs to all_gather, the unshard are the outputs fo the corresponding wait
        #       If you fix this to be dynamically discerned, you'll be my hero.
        sharded_param_names = ("t3", "t4")
        unshard_param_names = ("t10", "t21")
        result_saved_for_bwd = [x.name for x in fwd_trc.bound_symbols[-1].args[1][0]]
        self.assertTrue(all(t not in sharded_param_names for t in result_saved_for_bwd))
        # todo/fixme: Investigate why the following assertion is failing
        # self.assertTrue(all(t in result_saved_for_bwd for t in unshard_param_names))

        result_saved_for_bwd = [x.name for x in result_fwd_trc.bound_symbols[-1].args[1][0]]
        # todo/fixme: Investigate why the following assertion is failing
        # self.assertTrue(all(t in result_saved_for_bwd for t in sharded_param_names))
        self.assertTrue(all(t not in unshard_param_names for t in result_saved_for_bwd))

        # check allgather is inserted in backward trace
        from thunder.distributed.prims import PrimIDs

        self.assertTrue(all(bsym.sym.id != PrimIDs.ALL_GATHER for bsym in bwd_trc.bound_symbols))
        self.assertTrue(any(bsym.sym.id == PrimIDs.ALL_GATHER for bsym in result_bwd_trc.bound_symbols))

    @unittest.mock.patch.dict(os.environ, {"KINETO_LOG_LEVEL": "5"})  # silence torch.profiler logs
    @common_utils.parametrize(
        "executor,bucket_size_in_mb,dataset_size",
        product(tuple(executors_map.keys()), (0, 25), (1, 2)),
    )
    def test_ddp_with_no_sync_grad_accumulation(self, executor: str, bucket_size_in_mb: float, dataset_size: int):
        # This case tries to guarantee the parity between `thunder.distributed.ddp` with and without `no_sync`
        # from the perspectives of trace and numeric.
        # At trace level, in `no_sync`, the backward trace should NOT have AllReduce while outside of `no_sync`,
        # the trace should have.
        # For numerical parity, we compare the accumulated gradients with and without `no_sync` and even against gradients without accumulation.
        # If they are different, it'd be impossible to keep replicas identical.
        from collections import defaultdict
        from contextlib import nullcontext
        from thunder.common import CACHE_OPTIONS
        from thunder.distributed import ddp
        from thunder.distributed import get_skip_data_parallel_grad_sync

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
            has_allreduce = any(("allreduce_" in k or "all_reduce" in k) for k in keys)
            msg = f"{keys=}"
            if get_skip_data_parallel_grad_sync():
                self.assertFalse(has_allreduce, msg=msg)
            else:
                self.assertTrue(has_allreduce, msg=msg)

            return loss

        def get_model_and_optimizer(device):
            m = ToyModel().to(device)
            ddp_m = ddp(m, bucket_size_in_mb=bucket_size_in_mb)
            compiled_ddp_m = thunder.jit(
                ddp_m,
                cache_mode=CACHE_OPTIONS.CONSTANT_VALUES,
                executors=executors_map[executor].executors_list(),
            )
            optimizer = torch.optim.SGD(compiled_ddp_m.parameters(), lr=1e-3)
            return compiled_ddp_m, optimizer

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

        device = torch.device("cuda", self.rank)

        batch_size = 128
        num_micro_batch = 4
        micro_batch_size = batch_size // num_micro_batch
        with torch.no_grad():
            dataloader = [
                (torch.randn(batch_size, 12, device=device), torch.randn(batch_size, 8, device=device))
                for _ in range(dataset_size)
            ]

        initial_state_dict, ground_truth_losses, ground_truth_grads = get_ground_truth_loss_grads(device, dataloader)

        gradients = defaultdict(list)
        for use_no_sync in (True, False):
            compiled_ddp_m, optimizer = get_model_and_optimizer(device)
            compiled_ddp_m.load_state_dict(initial_state_dict)

            for iter_count, (x, y) in enumerate(dataloader):
                loss = torch.zeros((), device=device)
                with compiled_ddp_m.no_sync() if use_no_sync else nullcontext():
                    for i in range(num_micro_batch - 1):
                        cur_loss = run_fwd_bwd(
                            iter_count,
                            compiled_ddp_m,
                            x[i * micro_batch_size : (i + 1) * micro_batch_size, :],
                            y[i * micro_batch_size : (i + 1) * micro_batch_size, :],
                            num_micro_batch,
                        )
                        with torch.no_grad():
                            loss += cur_loss
                cur_loss = run_fwd_bwd(
                    iter_count, compiled_ddp_m, x[-micro_batch_size:, :], y[-micro_batch_size:, :], num_micro_batch
                )
                with torch.no_grad():
                    loss += cur_loss
                optimizer.step()
                gradients[use_no_sync].append([p.grad for p in compiled_ddp_m.parameters() if p.grad is not None])
                optimizer.zero_grad(set_to_none=True)

                num_expected_caches: int
                if use_no_sync:
                    num_expected_caches = 2
                else:
                    num_expected_caches = 1
                self.assertEqual(len(compiled_ddp_m._lc_cs.interpreter_cache), num_expected_caches)

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

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_ddp_grad_parity_with_without_bucketing(self, executor):
        from thunder.distributed import ddp

        device = torch.device("cuda", self.rank)
        initial_model_state = ToyModel().to(device).state_dict()

        for bucket_size_in_mb in (0, 100):
            m = ToyModel().to(device)
            m.load_state_dict(initial_model_state)
            cm = thunder.jit(
                ddp(m, bucket_size_in_mb=bucket_size_in_mb),
                executors=executors_map[executor].executors_list(),
            )
            x = torch.ones((2, 12)).to(device)
            cm(x).mean().backward()

            if bucket_size_in_mb == 0:
                gradients = tuple(p.grad for p in cm.parameters() if p.grad is not None)
            else:
                self.assertEqual(tuple(p.grad for p in cm.parameters() if p.grad is not None), gradients)

    # TODO(crcrpar): Add torch compile to executors_list
    @common_utils.parametrize(
        "executor,bucketing_strategy,fsdptype",
        product(
            tuple(executors_map.keys()),
            (
                FSDPBucketingStrategy.LAYER,
                # todo/fixme: Investigate why BLOCK is failing with DDP
                # FSDPBucketingStrategy.BLOCK,
            ),
            (FSDPType.ZERO2, FSDPType.ZERO3),
        ),
        name_fn=lambda executor, bucketing_strategy, fsdptype: (
            f"executor_{executor}_bucketing_{str(bucketing_strategy).split('.')[1].lower()}_{(str(fsdptype).lower().split('.')[1])}"
        ),
    )
    def test_fsdp_grad_parity_with_without_bucketing(
        self,
        executor,
        bucketing_strategy: FSDPBucketingStrategy,
        fsdptype: FSDPType,
    ):
        from thunder.distributed import fsdp

        device = torch.device("cuda", self.rank)
        initial_model_state = ToyModel().state_dict()

        for strategy in (FSDPBucketingStrategy.NONE, bucketing_strategy):
            m = ToyModel()
            m.load_state_dict(initial_model_state)
            cm = thunder.jit(
                fsdp(m, device=device, bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype),
                executors=executors_map[executor].executors_list(),
            )
            x = torch.ones((2, 12), device=device)
            loss = cm(x).mean()
            loss.backward()

            if strategy == FSDPBucketingStrategy.NONE:
                gradients = tuple(p.grad for p in cm.parameters() if p.grad is not None)
                orig_loss = loss.detach()
            else:
                self.assertEqual(loss, orig_loss)
                self.assertEqual(tuple(p.grad for p in cm.parameters() if p.grad is not None), gradients)

                # Make sure that at least one of "pack" takes multiple tensors.
                from thunder.executors.torchex import pack_for_fsdp_prim_impl
                from thunder.distributed.prims import PrimIDs as DistPrimIDs

                for ex_trace in (thunder.last_traces(cm)[-1], thunder.last_backward_traces(cm)[-1]):
                    pack_bsyms = list(
                        filter(
                            lambda bsym: bsym.sym.id in {DistPrimIDs.PACK_FOR_FSDP, pack_for_fsdp_prim_impl.id},
                            ex_trace.bound_symbols,
                        )
                    )
                    has_pack_multiple_tensors = False
                    for bsym in pack_bsyms:
                        first_arg = bsym.args[0]
                        self.assertIsInstance(first_arg, list)
                        has_pack_multiple_tensors |= len(first_arg) > 1
                    self.assertTrue(has_pack_multiple_tensors, msg=f"{[bsym.args[0] for bsym in pack_bsyms]=}")

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_fsdp_shard_unshard(self):
        from thunder.distributed import _shard_params, _unshard_params

        device = torch.device("cuda", self.rank)
        pg = c10d.new_group()

        model = torch.nn.Linear(3, 5, bias=False, device="meta")
        with pytest.raises(RuntimeError, match=r"parameter 'weight' \(5\) to be divisible by the world size \(2\)"):
            _shard_params(model, pg, device, None)

        model = torch.nn.Linear(3, 4, bias=False, device="meta")
        weight = torch.arange(3 * 4, device="cpu", dtype=torch.float).view(4, 3)
        model.load_state_dict({"weight": weight}, assign=True)

        # each shard got its corresponding piece of the weight
        _shard_params(model, pg, device, None)
        expected = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]] if self.rank == 0 else [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        # the weight was moved to device
        assert torch.equal(model.weight, torch.tensor(expected, device=device))

        # unsharding reconstructs the original weight (and cpu offloads)
        _unshard_params(model, pg, cpu_offload=True)
        assert torch.equal(model.weight, weight)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_fsdp_broadcast_from(self):
        from thunder.distributed import _shard_params

        device = torch.device("cuda", self.rank)
        pg = c10d.new_group()

        model = torch.nn.Linear(3, 4, bias=False, device="meta")
        model.register_buffer("foo", torch.tensor([123.0]), persistent=False)
        weight = torch.arange(3 * 4, device="cpu", dtype=torch.float).view(4, 3)
        if self.rank == 0:
            weight *= -1.0
            model.foo *= -1.0
        model.load_state_dict({"weight": weight}, assign=True)

        _shard_params(model, pg, device, 0)
        # since rank 0's params are negative and rank 1's are positive, we know broadcasting worked if all params are negative
        expected = (
            [[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]] if self.rank == 0 else [[-6.0, -7.0, -8.0], [-9.0, -10.0, -11.0]]
        )
        # the weight was moved to device
        assert torch.equal(model.weight, torch.tensor(expected, device=device))
        # same check for the buffer
        assert torch.equal(model.foo, torch.tensor([-123.0], device=device))

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 devices")
    def test_materialize_meta_tensors(self):
        from thunder.distributed import _shard_params

        class Submodule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(4, 8)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.tensor(0))
                self.l = torch.nn.Linear(2, 4)
                self.inner = Submodule()

        device = torch.device("cuda", self.rank)
        pg = c10d.new_group()

        with torch.device("meta"):
            model = MyModel()
        with pytest.raises(TypeError, match="MyModel.reset_parameters` method is implemented"):
            _shard_params(model, pg, device, None)

        class MyModel2(MyModel):
            def reset_parameters(self):
                self.buf = torch.empty_like(self.buf)

        with torch.device("meta"):
            model = MyModel2()

        _shard_params(model, pg, device, None)
        # all parameters were moved
        assert len(list(model.parameters())) == 4
        assert all(p.device.type == "cuda" for p in model.parameters())
        # buffers were moved too
        assert model.buf.device.type == "cuda"

    @common_utils.parametrize(
        "executor,bucketing_strategy,fsdptype",
        product(
            tuple(executors_map.keys()),
            (FSDPBucketingStrategy.NONE, FSDPBucketingStrategy.LAYER, FSDPBucketingStrategy.BLOCK),
            (FSDPType.ZERO3,),
        ),
        name_fn=lambda executor, bucketing_strategy, fsdptype: (
            f"executor_{executor}_bucketing_{str(bucketing_strategy).split('.')[1].lower()}_{(str(fsdptype).lower().split('.')[1])}"
        ),
    )
    def test_limit_in_flight_allgathers(
        self,
        executor,
        bucketing_strategy: FSDPBucketingStrategy,
        fsdptype: FSDPType,
    ):
        from thunder.distributed import fsdp
        from thunder.tests.nanogpt_model import Block, GPTConfig

        def check_inflight_allgather_number(trc, n: int, is_bucket: bool):
            from thunder.core.utils import producers
            from thunder.executors.torchex import all_gather_prim_impl, pack_for_fsdp_prim_impl, wait_prim_impl

            producers = producers(trc)
            cnt = 0
            for idx, bsym in enumerate(trc.bound_symbols):
                if bsym.sym.id == all_gather_prim_impl.id:
                    cnt += 1
                    if is_bucket:
                        self.assertEqual(trc.bound_symbols[idx - 1].sym.id, pack_for_fsdp_prim_impl.id)
                self.assertLessEqual(cnt, n)
                if bsym.sym.id == wait_prim_impl.id:
                    if producers[bsym.flat_proxy_args[0]].sym.id == all_gather_prim_impl.id:
                        cnt -= 1

        device = torch.device("cuda", self.rank)
        config = GPTConfig(dropout=0)
        m = Block(config).to(device=device)
        cm = thunder.jit(
            fsdp(m, device=device, broadcast_from=0, bucketing_strategy=bucketing_strategy, sharding_strategy=fsdptype),
            executors=executors_map[executor].executors_list(),
        )
        x = torch.ones((2, config.block_size, config.n_embd), device=device)
        loss = cm(x).mean()
        loss.backward()

        # get the trace before sorting
        fwd_trc = thunder.last_traces(cm)[-2]
        bwd_trc = thunder.last_backward_traces(cm)[-2]

        from thunder.distributed.utils import limit_in_flight_allgathers

        is_bucketing = bucketing_strategy != FSDPBucketingStrategy.NONE
        for i in range(1, 12):
            aft_trc = limit_in_flight_allgathers(fwd_trc, i, is_bucketing)
            check_inflight_allgather_number(aft_trc, i, is_bucketing)
        for i in range(1, 6):
            aft_trc = limit_in_flight_allgathers(bwd_trc, i, is_bucketing)
            check_inflight_allgather_number(aft_trc, i, is_bucketing)

    def test_ddp_model_as_argument(self):
        # Sanity test to make sure passing model as argument to
        # thunder.jit with `ddp` compiles.
        device = torch.device("cuda", self.rank)
        model = torch.nn.Linear(5, 10, bias=False, device=device)
        x = torch.randn(2, 5, device=device)

        def fwd_loss(m, x):
            return m(x).sum()

        model = thunder.distributed.ddp(model)
        fwd_loss = thunder.jit(fwd_loss)
        fwd_loss(model, x)

        # notice how we cannot do `model.no_sync()` because it's not a ThunderModule
        with thunder.ThunderModule.no_sync(model):
            fwd_loss(model, x)


common_utils.instantiate_parametrized_tests(CompileDDPTest)


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
    #   thunder doesn't like to model primitives with implicit defaults,
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
        def test_fn(executor, devices, dtype, bucket_size_in_mb=0):
            world_size = len(devices)
            input_data = []

            for rank in range(world_size):
                process_data = (init_method, world_size, rank, executor, devices[rank], dtype, bucket_size_in_mb)
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


# NOTE This assumes that one process will have rank=0 -- could generalize that to root
# TODO Test training, this test just currently tests forward
def _test_native_ddp_helper(input_data):
    init_method, world_size, rank, executor, device, dtype, bucket_size_in_mb = input_data

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
    ddp_model = ddp(model)
    cmodel = thunder.jit(
        ddp_model,
        executors=executor.executors_list(),
    )

    comparison_exceptions = []
    for _ in range(num_epochs):
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
        bwd_extrace_sym_ids = [bsym.sym.id for bsym in thunder.last_backward_traces(cmodel)[-1].bound_symbols]
        pack_unpack_update_bucket_view_found = (
            "torch_pack_prim_impl" in bwd_extrace_sym_ids
            and "torch_unpack_prim_impl" in bwd_extrace_sym_ids
            and "torch_update_bucket_view_prim_impl" in bwd_extrace_sym_ids
        )
        return comparison_exceptions and (pack_unpack_update_bucket_view_found or bucket_size_in_mb == 0)

    return None


def _test_native_fsdp_helper(input_data):
    init_method, world_size, rank, executor, device, dtype, bucketing_strategy = input_data

    num_samples = 2
    tensor_shape = (2, 2)
    sample_seed = 3456
    num_epochs = 1
    devicetype = devices.device_from_string(device).devicetype
    torch_dtype = ltorch.to_torch_dtype(dtype)

    pg = init_per_process_distributed(init_method, devicetype, world_size, rank)
    tdist.barrier(pg)

    def finalize_pg(pg):
        # NOTE This function is undocumented; its definition is here:
        # https://github.com/pytorch/pytorch/blob/416bf4e/torch/distributed/distributed_c10d.py#L1359
        tdist.barrier(pg)
        tdist.destroy_process_group(pg)

    weakref.finalize(pg, finalize_pg, pg)

    dataloader = create_per_process_dataloader(
        rank,
        num_samples=num_samples,
        tensor_shape=tensor_shape,
        tensor_dtype=torch_dtype,
        sample_seed=sample_seed,
        devicetype=devicetype,
    )

    # Creates, compiles, and FSDPs the model
    model = SmallModel(device, torch_dtype)

    original_weight_net1_shape = model.net1.weight.shape

    fsdp_model = fsdp(model, bucketing_strategy=bucketing_strategy, device=device)

    # Check that the model is sharded
    sharded_weight_net1 = fsdp_model.net1.weight
    assert sharded_weight_net1.shape != original_weight_net1_shape
    assert sharded_weight_net1.shape == (1, 2)

    cmodel = thunder.jit(
        fsdp_model,
        executors=executor.executors_list(),
    )

    comparison_exceptions = []
    for _ in range(num_epochs):
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

            for param_with_grad in filter(lambda p: p.grad is not None, cmodel.parameters()):
                sharded_grad = param_with_grad.grad
                assert sharded_grad.shape == param_with_grad.shape

    if rank == 0:
        return comparison_exceptions

    return None


def _test_ddp_transformer_engine(input_data):
    # Test Description: We run a dummy training loop for a simple `Linear(Relu(Linear(x)))`
    # model with thunder (using TE executor) and with PyTorch eager + TE
    # and verify that the weights have converged to same value and
    # fp8 meta state is same after `n_iter`.
    init_method, world_size, rank, executor, device, dtype, _unused_bucketing_strategy = input_data
    devicetype = devices.device_from_string(device).devicetype
    _unused_dtype = ltorch.to_torch_dtype(dtype)
    init_per_process_distributed(init_method, devicetype, world_size, rank)

    torch.cuda.set_device(rank)

    dim = 256
    n_iter = 10

    class ThunderModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(dim, dim, bias=False)
            self.fc2 = torch.nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.fc2(torch.nn.functional.relu(self.fc1(x)))

    # Weights
    fc1_weight = torch.randn(dim, dim, requires_grad=True).cuda()
    fc2_weight = torch.randn(dim, dim, requires_grad=True).cuda()

    # Inputs (different input on different rank).
    if rank == 0:
        x = torch.arange(dim * dim, dtype=torch.float).view(dim, dim).cuda()
    if rank == 1:
        x = torch.randn(dim, dim).cuda() * 100

    thunder_model = ThunderModel().cuda()
    thunder_model.fc1.weight.data = fc1_weight.clone()
    thunder_model.fc2.weight.data = fc2_weight.clone()

    jit_model = thunder.jit(
        thunder.distributed.ddp(thunder_model),
        executors=[
            transformer_engine_ex,
        ]
        + executor.executors_list(),
    )

    optim = torch.optim.SGD(thunder_model.parameters())

    for _ in range(n_iter):
        with fp8_autocast():
            o = jit_model(x).sum()
        o.backward()
        optim.step()
        optim.zero_grad()

    class TEModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = TELinear(dim, dim, bias=False)
            self.fc2 = TELinear(dim, dim, bias=False)

        def forward(self, x):
            return self.fc2(torch.nn.functional.relu(self.fc1(x)))

    te_model = TEModel().cuda()
    te_model.fc1.weight.data = fc1_weight.clone()
    te_model.fc2.weight.data = fc2_weight.clone()

    ddp_model = DDP(te_model)

    optim = torch.optim.SGD(te_model.parameters())

    for _ in range(n_iter):
        with fp8_autocast():
            o = ddp_model(x).sum()

        o.backward()
        optim.step()
        optim.zero_grad()

    thunder_to_te_layer_map = {"te_linear_0": te_model.fc1, "te_linear_1": te_model.fc2}

    fwd_traces = thunder.last_traces(jit_model)

    def is_same_across_ranks(t):
        t_clone = t.clone()
        torch.distributed.all_reduce(t_clone, op=torch.distributed.ReduceOp.AVG)
        assert_close(t, t_clone)

    # Compare the state of the two models.
    comparison_exceptions = []
    for bound_symbol in fwd_traces[-1].bound_symbols:
        if "te_linear" in bound_symbol.sym.name:
            thunder_fp8_meta = bound_symbol._call_ctx[bound_symbol.sym.name].func.fp8_meta
            te_fp8_meta = thunder_to_te_layer_map[bound_symbol.sym.name].fp8_meta
            try:
                # fwd tensor history
                assert_close(thunder_fp8_meta["scaling_fwd"].scale, te_fp8_meta["scaling_fwd"].scale)
                assert_close(thunder_fp8_meta["scaling_fwd"].scale_inv, te_fp8_meta["scaling_fwd"].scale_inv)
                assert_close(thunder_fp8_meta["scaling_fwd"].amax_history, te_fp8_meta["scaling_fwd"].amax_history)
                # bwd tensor history
                assert_close(thunder_fp8_meta["scaling_bwd"].scale, te_fp8_meta["scaling_bwd"].scale)
                assert_close(thunder_fp8_meta["scaling_bwd"].scale_inv, te_fp8_meta["scaling_bwd"].scale_inv)
                assert_close(thunder_fp8_meta["scaling_bwd"].amax_history, te_fp8_meta["scaling_bwd"].amax_history)

                # This has to be on all ranks so that the computation is not blocked
                is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale)
                is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].scale_inv)
                is_same_across_ranks(thunder_fp8_meta["scaling_fwd"].amax_history)
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale)
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].scale_inv)
                is_same_across_ranks(thunder_fp8_meta["scaling_bwd"].amax_history)
            except Exception as e:
                # Return exceptions only for rank==0
                if rank == 0:
                    comparison_exceptions.append(e)

        # Compare weights after `n_iters`
        try:
            assert_close(thunder_model.fc1.weight, te_model.fc1.weight)
            assert_close(thunder_model.fc2.weight, te_model.fc2.weight)
        except Exception as e:
            # Return exceptions only for rank==0
            if rank == 0:
                comparison_exceptions.append(e)

        return comparison_exceptions


def _test_ddp_transformer_engine_llama_sanity(input_data):
    # Test Description: We run a dummy training loop for a Transformer Model
    # We run a few iterations to see that TransformerEngine doesn't throw internal assertion
    # due to reordering of forward and backward operators.
    # (This test will fail without `_rearrange_transformer_engine_linear` in `torch_autograd.py`)
    # For more details, see docstring for `_rearrange_transformer_engine_linear` in transformer_engine_ex.py.
    from thunder.tests.llama2_model import Transformer, ModelArgs

    init_method, world_size, rank, executor, device, dtype, _unused_bucketing_strategy = input_data
    devicetype = devices.device_from_string(device).devicetype
    _unused_dtype = ltorch.to_torch_dtype(dtype)
    init_per_process_distributed(init_method, devicetype, world_size, rank)

    torch.cuda.set_device(rank)
    # data
    batch_size = 2
    max_seq_len = 32
    vocab_size = 32

    model_args = dict(
        dim=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=32,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    model.to(device)
    x = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
    y = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.int64, device=device)
    jit_model = thunder.jit(
        thunder.distributed.ddp(model), executors=(transformer_engine_ex,) + thunder.get_default_executors()
    )

    sanity_exceptions = []
    try:
        for _ in range(5):
            with fp8_autocast():
                out = jit_model(x, y).sum()
            out.backward()

        fwd_exec_trace = thunder.last_traces(jit_model)[-1]
        bwd_exec_trace = thunder.last_backward_traces(jit_model)[-1]

        # Verify that the first te_linear in fwd_exec_trace is the
        # last one in bwd_exec_tarce.
        # We verify that by managing the `ctx` (CollectionProxy) output by `te_linear` which is
        # passed to backward.
        # As CollectionProxy don't implement __eq__, we verify them by name.
        first_ctx_name = None
        for bsym in fwd_exec_trace.bound_symbols:
            if bsym.sym.name.startswith("te_linear"):
                first_ctx_name = bsym.output[1].name
                break

        for bsym in reversed(bwd_exec_trace.bound_symbols):
            if bsym.sym.name.startswith("te_functional"):
                assert first_ctx_name == bsym.args[-1].name, (first_ctx_name, bsym.args[-1].name)
                break
    except Exception as e:
        sanity_exceptions.append(e)

    if rank == 0:
        return sanity_exceptions
    return None


# NOTE This is just a stub, see the NOTE for ddp_wrapper
@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    decorators=(pytest.mark.parametrize("bucket_size_in_mb", (0, 25)),),
)
@ddp_wrapper("test_native_ddp", _test_native_ddp_helper)
def test_native_ddp(executor, devices, dtype, bucket_size_in_mb):
    pass


# NOTE CPU is skipped because of
# RuntimeError: no support for _allgather_base in Gloo process group
@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    decorators=(
        pytest.mark.parametrize(
            "bucket_size_in_mb",
            (
                FSDPBucketingStrategy.NONE,
                FSDPBucketingStrategy.LAYER,
                FSDPBucketingStrategy.BLOCK,
            ),
        ),
    ),
)
@ddp_wrapper("test_native_fsdp", _test_native_fsdp_helper)
def test_native_fsdp(executor, devices, dtype, bucket_size_in_mb):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # NOTE: Setting `NVTE_TORCH_COMPILE`
        # It is important to set this flag so that TE doesn't use
        # `torch.compile` to fuse a few operations. This is because
        # `torch.compile` creates a new process and that leads to
        # the error : daemonic processes are not allowed to have children
        # when running the tests.
        # With the setting below, we use `torch.jit` for this test suite
        # See: https://github.com/NVIDIA/TransformerEngine/blob/a38b291b0d1b04847e8ab1df8550df642a03a27d/transformer_engine/pytorch/jit.py#L11-L19
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}, clear=True),
    ),
)
@ddp_wrapper("test_ddp_transformer_engine", _test_ddp_transformer_engine)
def test_ddp_transformer_engine(executor, devices, dtype):
    pass


@instantiate(
    dtypes=(thunder.float32,),
    num_devices=2,
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
    decorators=(
        pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is not installed."),
        pytest.mark.skipif(not is_fp8_supported, reason=fp8_support_reason),
        # See NOTE: Setting `NVTE_TORCH_COMPILE`
        unittest.mock.patch.dict(os.environ, {"NVTE_TORCH_COMPILE": "0"}, clear=True),
    ),
)
@ddp_wrapper("test_ddp_transformer_engine_llama_sanity", _test_ddp_transformer_engine_llama_sanity)
def test_ddp_transformer_engine_llama_sanity(executor, devices, dtype):
    pass


if __name__ == "__main__":
    common_utils.run_tests()
