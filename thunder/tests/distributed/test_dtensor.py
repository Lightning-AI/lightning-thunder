import unittest

import pytest
import torch

if not torch.distributed.is_available():
    pytest.skip(allow_module_level=True)

from thunder.dynamo import thunderfx

from thunder.tests.distributed.helper import DistributedParallelTestCase
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_available() and torch.distributed.is_nccl_available(),
    "DTensor test requires CUDA and NCCL `torch.distributed` backend",
)
class DTensorTest(DistributedParallelTestCase):
    def test_dtensor(self):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        def fn(x, w):
            return torch.mul(x, w)

        weight = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        expected = torch.compile(fn)(in_dtensor, weight)
        tmodel = thunderfx(fn)
        actual = tmodel(in_dtensor, weight)

        torch.testing.assert_close(actual, expected)

        g_o = distribute_tensor(torch.ones(dim_size, dim_size), mesh, [Shard(0)])
        expected_g = torch.autograd.grad(
            expected,
            (in_dtensor,),
            g_o,
        )
        actual_g = torch.autograd.grad(actual, (in_dtensor,), g_o)

        torch.testing.assert_close(actual_g, expected_g)
