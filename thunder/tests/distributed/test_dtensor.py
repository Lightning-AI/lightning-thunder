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

from torch.testing._internal import common_utils

from thunder.tests.distributed.helper import executors_map
from thunder.tests.opinfos import OpInfo, SampleInput, opinfos, reshape_opinfo
import thunder.core.dtypes as dtypes
from thunder.core.utils import tree_map, tree_flatten
from thunder.core.transforms import grad


# NOTE: We run all these similar functions seperately
#       as we want to avoid nvfuser issue (https://github.com/NVIDIA/Fuser/issues/4507)
#       where trying to create FusionDefinition with same math operation can fail.
functions_to_test = {
    "torch.mul": lambda x, w: torch.mul(x, w),
    "x.mul(w)": lambda x, w: torch.Tensor.mul(x, w),
    "x * w": lambda x, w: x * w,
}

dtensor_supported_ops = (reshape_opinfo,)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.distributed.is_nccl_available(),
    "DTensor test requires CUDA and NCCL `torch.distributed` backend",
)
class DTensorTest(DistributedParallelTestCase):
    @common_utils.parametrize("executor, fn_key", product(tuple(executors_map.keys()), functions_to_test.keys()))
    def test_dtensor_basic_op(self, executor, fn_key):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        if executor == "nvfuser":
            raise unittest.SkipTest("See PR: https://github.com/Lightning-AI/lightning-thunder/pull/2423")

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

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_dtensor_reshape(self, executor):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        def fn_reshape(x, shape):
            return torch.reshape(x, shape)

        def fn_reshape_method(x, shape):
            return x.reshape(shape)

        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        # Test different reshape shapes
        test_shapes = [
            (dim_size * dim_size,),  # Flatten to 1D
            (dim_size, dim_size),  # Keep original shape
            (4, 4, dim_size),  # Reshape to 3D
        ]

        def _test(fn, *args):
            expected = fn(*args)
            tmodel = thunder.jit(fn, executors=executors_map[executor].executors_list())
            actual = tmodel(*args)
            torch.testing.assert_close(actual, expected)

            # Test gradient
            g_o = distribute_tensor(torch.ones(shape), mesh, [Shard(0)])
            expected_g = torch.autograd.grad(expected, (in_dtensor,), g_o)
            actual_g = torch.autograd.grad(actual, (in_dtensor,), g_o)
            torch.testing.assert_close(actual_g, expected_g)

        for shape, fn in product(test_shapes, (fn_reshape, fn_reshape_method)):
            _test(fn, in_dtensor, shape)

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
        product(dtensor_supported_ops, tuple(executors_map.keys())),
        lambda op, executor: op.name + "_" + executor,
    )
    def test_dtensor_opinfo(self, op: OpInfo, executor):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        def to_dtensor(x):
            if isinstance(x, torch.Tensor):
                return distribute_tensor(x, mesh, [Replicate()])
            return x

        for sample in op.sample_inputs("cpu", dtypes.float32, requires_grad=True):
            args = tree_map(to_dtensor, sample.args)
            kwargs = tree_map(to_dtensor, sample.kwargs)
            torch_op = op.torch_reference
            torch_result = torch_op(*args, **kwargs)
            thunder_op = thunder.jit(op.op, executors=executors_map[executor].executors_list())
            thunder_result = thunder_op(*args, **kwargs)
            torch.testing.assert_close(thunder_result, torch_result)

            def is_output_differentiable(x):
                # grad_fn is set only if one of the input `requires_grad=True`
                # and the op is differentiable.
                # Example:
                # >>> x = torch.ones(3, requires_grad=True)
                # >>> y = torch.ones(3, requires_grad=False)
                # >>> (x + x).grad_fn  # <AddBackward0 object at 0x7f0502edcf40>
                # >>> (y + y).grad_fn  # None
                # >>> (y + x).grad_fn  # <AddBackward0 object at 0x7f0502e21060>
                # >>> (x < 1).grad_fn  # None (non-differentiable op)
                # Op with differentiable and non-differentiable outputs.
                # >>> torch.topk(x, k=2)
                # torch.return_types.topk(
                # values=tensor([1., 1.], grad_fn=<TopkBackward0>),
                # indices=tensor([0, 1]))
                # >>> torch.topk(torch.ones(3, requires_grad=False), k=2)
                # torch.return_types.topk(
                # values=tensor([1., 1.]),
                # indices=tensor([0, 1]))
                return x.grad_fn is not None or is_returning_self(x)

            def is_returning_self(x):
                if x.is_leaf and x.requires_grad:
                    return True
                return False

            def filter_differentiable_outputs(outputs):
                if isinstance(outputs, torch.Tensor):
                    # Otherwise `filter` below will
                    # iterate over the Tensor data.
                    outputs = [outputs]

                return list(filter(is_output_differentiable, outputs))

            # Computes PyTorch (competition) result
            torch_flats, spec = tree_flatten((args, kwargs))
            torch_result = filter_differentiable_outputs(torch_result)
            if torch_result == []:
                raise RuntimeError(
                    f"phantom_grad: Expected atleast 1 differentiable output. If {op.name} is non-differentiable, set op.supports_grad=False."
                )

            grads = []
            assert isinstance(torch_result, torch.Tensor) or isinstance(torch_result, Sequence), (
                "Expected a single torch tensor or a sequence of torch tensors when testing phantom grad torch consistency"
            )
            if isinstance(torch_result, Sequence):
                for x in torch_result:
                    assert isinstance(x, torch.Tensor), (
                        "Expected a single torch tensor or a sequence of torch tensors when testing phantom grad torch consistency"
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

            thunder_result = thunder_op(*args, **kwargs)
            thunder_result = filter_differentiable_outputs(thunder_result)
            thunder_grad_result = torch.autograd.grad(thunder_result, torch_tensors_requiring_grad, grads)
            torch.testing.assert_close(thunder_grad_result, torch_grad_result)


common_utils.instantiate_parametrized_tests(DTensorTest)

if __name__ == "__main__":
    common_utils.run_tests()
