import unittest
from itertools import product
from collections.abc import Sequence

import pytest
import torch

if not torch.distributed.is_available():
    pytest.skip(allow_module_level=True)

from thunder.dynamo import thunderfx
import thunder

from thunder.tests.distributed.helper import DistributedParallelTestCase
from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorConverter

from torch.testing._internal import common_utils

from thunder.tests.distributed.helper import executors_map
from thunder.tests.opinfos import OpInfo, get_opinfo
from thunder.tests.utils import is_output_differentiable, filter_differentiable_outputs
import thunder.core.dtypes as dtypes
from thunder.core.pytree import tree_flatten


# NOTE: We run all these similar functions seperately
#       as we want to avoid nvfuser issue (https://github.com/NVIDIA/Fuser/issues/4507)
#       where trying to create FusionDefinition with same math operation can fail.
functions_to_test = {
    "torch.mul": lambda x, w: torch.mul(x, w),
    "x.mul(w)": lambda x, w: torch.Tensor.mul(x, w),
    "x * w": lambda x, w: x * w,
}


# DTensor supported ops
dtensor_supported_ops = ("reshape",)

dtensor_supported_opinfos = [get_opinfo(op) for op in dtensor_supported_ops]


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_nccl_available(),
    "DTensor test requires CUDA and NCCL `torch.distributed` backend",
)
class DTensorTest(DistributedParallelTestCase):
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

        tmodel = thunder.jit(fn, executors=thunder.get_always_executors())
        with pytest.raises(AssertionError):
            tmodel(in_dtensor, w)

    def test_dtensor_incorrect_cotangent(self):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        w_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])
        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        def fn(x, w):
            return torch.mul(x, w)

        tmodel = thunder.jit(fn, executors=thunder.get_always_executors())
        actual = tmodel(in_dtensor, w_dtensor)
        g_o = distribute_tensor(torch.ones(dim_size, dim_size), mesh, [Shard(1)])

        with pytest.raises(RuntimeError, match="has changed for cotangent between tracing and runtime"):
            torch.autograd.grad(actual, (in_dtensor, w_dtensor), g_o)

    @common_utils.parametrize(
        "op, executor",
        product(dtensor_supported_opinfos, tuple(executors_map.keys())),
        lambda op, executor: op.name + "_" + executor,
    )
    def test_dtensor_opinfo(self, op: OpInfo, executor):
        # NOTE: This test only tests for dtype=torch.float32 and requires_grad=True
        #       not for all dtype which are supported by the operation.
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        thunder_op = thunder.jit(op.op, executors=executors_map[executor].executors_list())

        tested_sample_count = 0
        for sample in op.sample_inputs("cpu", dtypes.float32, requires_grad=True):
            # DTensorConverter converts inputs tensors to DTensor and creates DTensor
            # with possible placements based on the input shapes.
            # See - https://github.com/pytorch/pytorch/blob/eaa5d9d3d3dc642832b269b184f0c3ab8c990274/torch/testing/_internal/distributed/_tensor/common_dtensor.py#L521
            dtensor_converter = DTensorConverter(mesh, sample.args, sample.kwargs)
            for dtensor_args, dtensor_kwargs in dtensor_converter:
                if not dtensor_converter.successful():
                    continue

                torch_op = op.torch_reference

                # Computes PyTorch result
                try:
                    torch_result = torch_op(*dtensor_args, **dtensor_kwargs)
                except Exception:
                    # Unsupported input passed to `torch_op`, we expect an exception from `thunder_op` as well.
                    with pytest.raises(Exception):
                        thunder_op(*dtensor_args, **dtensor_kwargs)
                    continue

                thunder_result = thunder_op(*dtensor_args, **dtensor_kwargs)
                torch.testing.assert_close(thunder_result, torch_result)

                torch_flats, _ = tree_flatten((dtensor_args, dtensor_kwargs))
                torch_result = filter_differentiable_outputs(torch_result)
                if torch_result == []:
                    raise RuntimeError("test_dtensor_opinfo: Expected atleast 1 differentiable output.")

                grads = []
                assert isinstance(torch_result, torch.Tensor) or isinstance(torch_result, Sequence), (
                    "test_dtensor_opinfo:Expected a single torch tensor or a sequence of torch tensors"
                )
                if isinstance(torch_result, Sequence):
                    for x in torch_result:
                        assert isinstance(x, torch.Tensor), (
                            "test_dtensor_opinfo: Expected a single torch tensor or a sequence of torch tensors"
                        )
                        if is_output_differentiable(x):
                            grads.append(torch.ones_like(x))
                else:
                    if is_output_differentiable(torch_result):
                        grads = [torch.ones_like(torch_result)]

                torch_tensors_requiring_grad = tuple(
                    f for f in torch_flats if isinstance(f, torch.Tensor) and f.requires_grad
                )
                torch_grad_result = torch.autograd.grad(torch_result, torch_tensors_requiring_grad, grads)

                thunder_result = filter_differentiable_outputs(thunder_result)
                thunder_grad_result = torch.autograd.grad(thunder_result, torch_tensors_requiring_grad, grads)
                torch.testing.assert_close(thunder_grad_result, torch_grad_result)

                # Increment tested sample count
                tested_sample_count += 1

        assert tested_sample_count > 0, f"test_dtensor_opinfo:No samples tested for {op.name} with {executor} executor"


common_utils.instantiate_parametrized_tests(DTensorTest)

if __name__ == "__main__":
    common_utils.run_tests()
