from itertools import product
import re
import unittest

import torch
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


common_utils.instantiate_parametrized_tests(DistributedCollectiveOpTest)


if __name__ == "__main__":
    common_utils.run_tests()
