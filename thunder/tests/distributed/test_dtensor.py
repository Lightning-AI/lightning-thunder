import unittest

import pytest
import torch

if not torch.distributed.is_available():
    pytest.skip(allow_module_level=True)

from thunder.dynamo import thunderfx
import thunder

from thunder.tests.distributed.helper import DistributedParallelTestCase
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "DTensor test requires CUDA and NCCL `torch.distributed` backend",
)
class DTensorTest(DistributedParallelTestCase):
    def test_dtensor_basic_op(self):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        def _helper(fn, in_dtensor, w_dtensor):
            expected = torch.compile(fn)(in_dtensor, w_dtensor)
            tmodel = thunder.jit(fn)
            actual = tmodel(in_dtensor, w_dtensor)

            torch.testing.assert_close(actual, expected)

            g_o = distribute_tensor(torch.ones(dim_size, dim_size), mesh, [Shard(0)])
            expected_g = torch.autograd.grad(
                expected,
                (in_dtensor, w_dtensor),
                g_o,
            )
            actual_g = torch.autograd.grad(actual, (in_dtensor, w_dtensor), g_o)

            torch.testing.assert_close(actual_g, expected_g)

        w_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])
        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        # Verify torch API works
        _helper(lambda x, w: torch.mul(x, w), in_dtensor, w_dtensor)

        # Verify calling method works
        _helper(lambda x, w: torch.Tensor.mul(x, w), in_dtensor, w_dtensor)

        # Verify calling special method works
        _helper(lambda x, w: x * w, in_dtensor, w_dtensor)

    def test_dtensor_unsupported(self):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        w_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        def fn(x, w):
            return torch.div(x, w)

        tmodel = thunder.jit(fn)
        with pytest.raises(AssertionError):
            tmodel(in_dtensor, w_dtensor)

        def fn(x, w):
            return x / w

        tmodel = thunder.jit(fn)
        with pytest.raises(AssertionError):
            tmodel(in_dtensor, w_dtensor)

    def test_dtensor_unsupported_mixed_input(self):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        def fn(x, w):
            return torch.div(x, w)

        w = torch.randn(dim_size, dim_size, requires_grad=True)

        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        tmodel = thunder.jit(fn)
        with pytest.raises(AssertionError):
            tmodel(in_dtensor, w)
