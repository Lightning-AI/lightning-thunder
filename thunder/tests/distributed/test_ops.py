from itertools import product
import re
import unittest

import pytest
import torch

if not torch.distributed.is_available():
    pytest.skip(allow_module_level=True)
from torch.testing import make_tensor
from torch.distributed import distributed_c10d as c10d

import thunder
from thunder.distributed import prims
import thunder.torch as ltorch

from thunder.tests.distributed.helper import DistributedParallelTestCase, executors_map
from torch.testing._internal import common_utils


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "DDP test requires CUDA and NCCL `torch.distributed` backend",
)
class DistributedCollectiveOpTest(DistributedParallelTestCase):
    # TODO(crcrpar): Mandate multiple GPUs so that the timing of collectives matters especially for
    # nvfuser executor.
    @common_utils.parametrize("executor,inplace", product(tuple(executors_map.keys()), (False, True)))
    def test_all_reduce(self, executor, inplace: bool):
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
        func_to_jit = foo if inplace else lc_foo
        cfoo = thunder.jit(func_to_jit, executors=_executor.executors_list())

        for op, async_op in product((None, torch.distributed.ReduceOp.SUM), (False, True)):
            expected = foo(a, b, op, process_group, async_op)
            if inplace and async_op:
                with self.assertRaisesRegex(
                    NotImplementedError,
                    re.escape("`torch.distributed.all_reduce` with async_op=True is not supported"),
                ):
                    cfoo(a, b, op, process_group, async_op)
            else:
                actual = cfoo(a, b, op, process_group, async_op)
                self.assertEqual(actual, expected)

    @common_utils.parametrize("executor,dim,inplace", product(tuple(executors_map.keys()), (None, 0, 1), (False, True)))
    def test_all_gather(self, executor, dim: int | None, inplace: bool):
        _executor = executors_map[executor]

        # NOTE torch.distributed.all_gather is an inplace operation
        def foo(
            a,
            b,
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
            dim: int | None,
        ):
            c = a + b

            result_shape = list(c.shape)
            if dim is not None:
                result_shape[dim] *= process_group.size()
            else:
                result_shape[0] *= process_group.size()
            d = torch.empty(result_shape, device=c.device, dtype=c.dtype)
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
            dim: int | None,
        ):
            c = a + b

            d = ltorch.all_gather(c, group=process_group, async_op=async_op, dim=dim)

            if async_op:
                d = prims.wait(d)

            e = d + 1
            return a, e

        device = f"cuda:{self.rank}"
        a = make_tensor((2, 2), device=device, dtype=torch.float32)
        b = make_tensor((2, 2), device=device, dtype=torch.float32)
        process_group = c10d.new_group()

        # NOTE Preprocessing is disabled because we call thunder.torch operations directly
        func_to_jit = foo if inplace else lc_foo
        cfoo = thunder.jit(func_to_jit, executors=_executor.executors_list())

        for async_op in (True, False):
            expected = foo(a, b, process_group, async_op, dim)
            actual = cfoo(a, b, process_group, async_op, dim)
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

    @common_utils.parametrize("executor,dim,inplace", product(tuple(executors_map.keys()), (None, 0, 1), (False, True)))
    def test_reduce_scatter(self, executor, dim, inplace):
        _executor = executors_map[executor]

        # NOTE torch.distributed.all_gather is an inplace operation
        def foo(
            a,
            b,
            op,
            process_group: torch.distributed.ProcessGroup,
            async_op: bool,
            dim: int | None,
        ):
            c = a + b
            result_shape = list(a.shape)
            if dim is None:
                result_shape[0] //= process_group.size()
            else:
                result_shape[dim] //= process_group.size()
            d = torch.empty(result_shape, device=c.device, dtype=c.dtype)
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
            dim: int | None,
        ):
            c = a + b

            d = ltorch.reduce_scatter(c, op, group=process_group, async_op=async_op, dim=dim)

            if async_op:
                d = prims.wait(d)

            e = d + 1
            return a, e

        device = f"cuda:{self.rank}"
        a = make_tensor((4, 2), device=device, dtype=torch.float32)
        b = make_tensor((4, 2), device=device, dtype=torch.float32)
        process_group = c10d.new_group()

        # NOTE Preprocessing is disabled because we call thunder.torch operations directly
        func_to_jit = foo if inplace else lc_foo
        cfoo = thunder.jit(func_to_jit, executors=_executor.executors_list())

        for op, async_op in product((None, torch.distributed.ReduceOp.SUM), (False, True)):
            expected = foo(a, b, op, process_group, async_op, dim=dim)
            actual = cfoo(a, b, op, process_group, async_op, dim=dim)
            self.assertEqual(actual, expected)

    @common_utils.parametrize(
        "executor,op",
        product(tuple(executors_map.keys()), ("all_gather_into_tensor", "reduce_scatter_tensor")),
    )
    def test_native_collective_comms(self, executor, op):
        from thunder.executors.torchex import all_gather_prim_impl, reduce_scatter_prim_impl, wait_prim_impl

        device = f"cuda:{self.rank}"
        shape = (4, 2)
        group = torch.distributed.distributed_c10d._get_default_group()
        if op.startswith("all_gather"):
            output_shape = (shape[0] * group.size(), 2)
        else:
            output_shape = (shape[0] // group.size(), 2)

        comm = getattr(torch.distributed, op)
        _executor = executors_map[executor]

        def foo(
            a: torch.Tensor,
            b: torch.Tensor,
            output: torch.Tensor,
            group: torch.distributed.ProcessGroup,
        ):
            c = a + b
            handle = comm(output, c, group=group, async_op=True)
            e = c + 1
            handle.wait()
            f = e * b
            output *= 2
            return f

        jitted = _executor.make_callable(foo)
        a = make_tensor(shape, device=device, dtype=torch.float32)
        b = make_tensor(shape, device=device, dtype=torch.float32)
        output = torch.empty(output_shape, device=device, dtype=torch.float32)
        a_, b_, output_ = a.clone().detach(), b.clone().detach(), output.clone().detach()

        f = jitted(a, b, output, group)
        f_ = foo(a_, b_, output_, group)
        torch.testing.assert_close(f, f_)
        torch.testing.assert_close(output, output_)

        traces = thunder.last_traces(jitted)
        trace_with_waits_sorted = None
        for t in traces:
            if t._provenance is not None and t._provenance.pss == "Sort Waits":
                trace_with_waits_sorted = t
                break

        comm_idx = len(t.bound_symbols)
        for idx, bsym in enumerate(trace_with_waits_sorted.bound_symbols):
            if bsym.sym.id in {all_gather_prim_impl.id, reduce_scatter_prim_impl.id}:
                comm_idx = idx
            if bsym.sym.id == wait_prim_impl.id:
                self.assertGreater(idx, comm_idx + 2)


common_utils.instantiate_parametrized_tests(DistributedCollectiveOpTest)


if __name__ == "__main__":
    common_utils.run_tests()
