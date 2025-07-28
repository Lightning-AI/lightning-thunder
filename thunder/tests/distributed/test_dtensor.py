import unittest
from itertools import product

import pytest
import torch

if not torch.distributed.is_available():
    pytest.skip(allow_module_level=True)

from thunder.dynamo import thunderfx
import thunder

from thunder.tests.distributed.helper import DistributedParallelTestCase
from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate

from torch.testing._internal import common_utils

from thunder.tests.distributed.helper import executors_map


# NOTE: We run all these similar functions seperately
#       as we want to avoid nvfuser issue (https://github.com/NVIDIA/Fuser/issues/4507)
#       where trying to create FusionDefinition with same math operation can fail.
functions_to_test = {
    "torch.mul": lambda x, w: torch.mul(x, w),
    "x.mul(w)": lambda x, w: torch.Tensor.mul(x, w),
    "x * w": lambda x, w: x * w,
}


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_nccl_available(),
    "DTensor test requires CUDA and NCCL `torch.distributed` backend",
)
class DTensorTest(DistributedParallelTestCase):
    @pytest.mark.skip(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2355")
    @common_utils.parametrize("executor, fn_key", product(tuple(executors_map.keys()), functions_to_test.keys()))
    def test_dtensor_basic_op(self, executor, fn_key):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        def _helper(fn, in_dtensor, w_dtensor):
            expected = torch.compile(fn)(in_dtensor, w_dtensor)
            tmodel = thunder.jit(fn, executors=executors_map[executor].executors_list())
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
        _helper(functions_to_test[fn_key], in_dtensor, w_dtensor)

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

    @pytest.mark.skip(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2355")
    def test_dtensor_incorrect_cotangent(self):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        w_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])
        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        def fn(x, w):
            return torch.mul(x, w)

        tmodel = thunder.jit(fn)
        actual = tmodel(in_dtensor, w_dtensor)
        g_o = distribute_tensor(torch.ones(dim_size, dim_size), mesh, [Shard(1)])

        with pytest.raises(RuntimeError, match="has changed for cotangent between tracing and runtime"):
            torch.autograd.grad(actual, (in_dtensor, w_dtensor), g_o)


common_utils.instantiate_parametrized_tests(DTensorTest)

if __name__ == "__main__":
    common_utils.run_tests()
