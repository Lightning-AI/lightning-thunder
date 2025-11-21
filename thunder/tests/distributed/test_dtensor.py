import unittest
from itertools import product
from collections.abc import Sequence
from looseversion import LooseVersion
import os

import pytest
import torch

if not torch.distributed.is_available():
    pytest.skip(allow_module_level=True)

import thunder

from thunder.tests.distributed.helper import DistributedParallelTestCase
from torch.distributed.tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Shard, Replicate
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorConverter
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
)

from torch.testing._internal import common_utils

from thunder.tests.distributed.helper import executors_map
from thunder.tests.opinfos import OpInfo, get_opinfo
from thunder.tests.utils import is_output_differentiable, filter_differentiable_outputs
import thunder.core.dtypes as dtypes
from thunder.core.pytree import tree_flatten
from thunder.dynamo import thunderfx


# NOTE: We run all these similar functions seperately
#       as we want to avoid nvfuser issue (https://github.com/NVIDIA/Fuser/issues/4507)
#       where trying to create FusionDefinition with same math operation can fail.
functions_to_test = {
    "torch.mul": lambda x, w: torch.mul(x, w),
    "x.mul(w)": lambda x, w: torch.Tensor.mul(x, w),
    "x * w": lambda x, w: x * w,
}


# NOTE: OpInfo may use `clang` or `ltorch` ops to be jitted with thunder.jit.
#       However, for the current DTensor implementation, we add a dispatch in the `torch` operation lookaside
#       to choose between DTensor supported symbol (from `dtensor_torch_and_prims.py`) or the usual `ltorch` symbol.
#       This is why we need to make sure that the OpInfo uses PyTorch native op as `op` which is passed to thunder.jit.
class DTensorOpInfo:
    def __init__(
        self,
        *,
        name,
        op,
        torch_reference,
        supports_grad,
        sample_inputs,
        skip_noncontiguous_for_executor=(),
        skip_for_executor=(),
    ):
        self.name = name
        assert "torch" in op.__module__, "OpInfo must use PyTorch native op as `op` which is passed to thunder.jit"
        self.op = op
        self.torch_reference = torch_reference
        # NOTE: Not all DTensor ops support grad initially, use this to disable grad tests for them
        self.supports_grad = supports_grad
        # NOTE: This should generally reuse the sample_inputs from the OpInfo
        self.sample_inputs = sample_inputs

        # In some cases, non-contiguous inputs are not supported by the executor.
        assert isinstance(skip_noncontiguous_for_executor, tuple), "skip_noncontiguous_for_executor must be a tuple"
        self.skip_noncontiguous_for_executor = skip_noncontiguous_for_executor

        assert isinstance(skip_for_executor, tuple), "skip_for_executor must be a tuple"
        self.skip_for_executor = skip_for_executor


# DTensor supported ops
dtensor_supported_opinfos = (
    DTensorOpInfo(
        name="reshape",
        op=torch.reshape,
        torch_reference=torch.reshape,
        supports_grad=True,
        sample_inputs=get_opinfo("reshape").sample_inputs,
    ),
    DTensorOpInfo(
        name="linear",
        op=torch.nn.functional.linear,
        torch_reference=torch.nn.functional.linear,
        supports_grad=False,
        sample_inputs=get_opinfo("linear").sample_inputs,
    ),
    DTensorOpInfo(
        name="exp",
        op=torch.exp,
        torch_reference=torch.exp,
        supports_grad=True,
        sample_inputs=get_opinfo("exp").sample_inputs,
        # Ref: https://github.com/Lightning-AI/lightning-thunder/issues/2670
        skip_for_executor=("nvfuser",),
    ),
    DTensorOpInfo(
        name="neg",
        op=torch.neg,
        torch_reference=torch.neg,
        supports_grad=False,
        sample_inputs=get_opinfo("neg").sample_inputs,
        # Ref:https://github.com/NVIDIA/Fuser/pull/5124
        skip_noncontiguous_for_executor=("nvfuser",),
    ),
    DTensorOpInfo(
        name="reciprocal",
        op=torch.reciprocal,
        torch_reference=torch.reciprocal,
        supports_grad=False,
        sample_inputs=get_opinfo("reciprocal").sample_inputs,
        # Ref:https://github.com/NVIDIA/Fuser/pull/5124
        skip_noncontiguous_for_executor=("nvfuser",),
    ),
    DTensorOpInfo(
        name="add",
        op=torch.add,
        torch_reference=torch.add,
        supports_grad=False,
        sample_inputs=get_opinfo("add").sample_inputs,
        # Ref:https://github.com/NVIDIA/Fuser/issues/5314
        skip_for_executor=("nvfuser",),
    ),
    DTensorOpInfo(
        name="silu",
        op=torch.nn.functional.silu,
        torch_reference=torch.nn.functional.silu,
        supports_grad=False,
        sample_inputs=get_opinfo("silu").sample_inputs,
        # Ref:https://github.com/NVIDIA/Fuser/pull/5124
        skip_noncontiguous_for_executor=("nvfuser",),
    ),
)

skip_opinfos = (
    # RuntimeError: Metadata (placement and mesh) has changed for cotangent between tracing and runtimeduring tracing
    # it was Spec(S(1) on (1, 2, 1)) but at runtime it is Spec(S(1) on (1, 2, 1)).
    "reshape",
)


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

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_dtensor_convert_element_type(self, executor):
        from thunder.torch.experimental.dtensor_torch_and_prims import dtensor_convert_element_type_prim

        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 16

        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

        def fn(x):
            return dtensor_convert_element_type_prim(x, dtypes.bfloat16)

        tmodel = thunder.jit(fn, executors=executors_map[executor].executors_list())
        actual = tmodel(in_dtensor)
        expected = in_dtensor.to(torch.bfloat16)

        torch.testing.assert_close(actual, expected)

        g_o = distribute_tensor(torch.ones(dim_size, dim_size), mesh, [Shard(0)])
        expected_g = torch.autograd.grad(
            expected,
            (in_dtensor,),
            g_o,
        )
        actual_g = torch.autograd.grad(actual, (in_dtensor,), g_o)

        torch.testing.assert_close(actual_g, expected_g)

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    def test_dtensor_broadcast_in_dim(self, executor):
        from thunder.torch.experimental.dtensor_torch_and_prims import dtensor_broadcast_in_dim_prim

        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))
        dim_size = 16
        in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=False), mesh, [Shard(0)])

        def fn(x):
            return dtensor_broadcast_in_dim_prim(x, (dim_size, dim_size), (0, 1))

        tmodel = thunder.jit(fn, executors=executors_map[executor].executors_list())
        actual = tmodel(in_dtensor)
        expected = in_dtensor.broadcast_to((dim_size, dim_size))

        torch.testing.assert_close(actual, expected)

    @common_utils.parametrize("jit_fn", (thunder.jit, thunderfx), name_fn=lambda jit_fn: jit_fn.__name__)
    def test_dtensor_columnwise_parallel(self, jit_fn):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))
        dim_size = 16
        in_dtensor = torch.randn(dim_size, dim_size, requires_grad=False)
        m = torch.nn.Linear(dim_size, dim_size)
        m.requires_grad_(False)

        parallelized_model = parallelize_module(m, mesh, ColwiseParallel())

        # `parallelize_module` sets `requires_grad` to True, set it to False again.
        parallelized_model.requires_grad_(False)

        actual = parallelized_model(in_dtensor)
        expected = m(in_dtensor)
        torch.testing.assert_close(actual, expected)

        tmodel = jit_fn(parallelized_model, nv_enable_linear=True)

        if jit_fn == thunder.jit:
            # Original error caught by the interpreter:
            # File "/opt/pytorch/lightning-thunder/thunder/core/jit_ext.py", line 444, in _general_jit_getattr_lookaside
            # obj.original_value.__dict__,
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # AttributeError: 'object' object has no attribute '__dict__'. Did you mean: '__dir__'?
            with self.assertRaises(thunder.core.interpreter.InterpreterError):
                actual = tmodel(in_dtensor)
        else:
            actual = tmodel(in_dtensor)
            torch.testing.assert_close(actual, expected)

        if jit_fn == thunderfx:
            assert len(tmodel._backend.subgraph_infos) == 1
            assert len(tmodel._backend.subgraph_infos[0].thunder_compiled_fns) == 1
            assert len(tmodel._backend.subgraph_infos[0].split_reasons) == 0

    @common_utils.parametrize("executor", tuple(executors_map.keys()))
    @common_utils.parametrize(
        "input_shardings",
        [
            (
                [
                    Shard(
                        -1,
                    )
                ],
                [
                    Shard(1),
                ],
                [Replicate()],
            ),
        ],
    )
    def test_dtensor_grouped_mm(self, executor, input_shardings):
        if executor == "nvfuser" and "multidevice" in os.environ.get("NVFUSER_DISABLE", ""):
            raise unittest.SkipTest("test_dtensor_grouped_mm: nvfuser multidevice is disabled")

        if LooseVersion(torch.__version__) < "2.8":
            raise unittest.SkipTest("test_dtensor_grouped_mm: torch._grouped_mm is not available in torch < 2.8")

        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        if (torch.cuda.get_device_capability() < (9, 0)) and executor == "torch":
            raise unittest.SkipTest(
                "test_dtensor_grouped_mm: torch._grouped_mm doesn't support device capability < 9.0"
            )

        M = 16
        N = 64
        K = 32
        G = 2

        inp_shard, w_shard, offsets_shard = input_shardings
        in_dtensor = distribute_tensor(torch.randn(M, K, requires_grad=False, dtype=torch.bfloat16), mesh, inp_shard)
        w_dtensor = distribute_tensor(torch.randn(G, K, N, requires_grad=False, dtype=torch.bfloat16), mesh, w_shard)
        offsets_dtensor = distribute_tensor(torch.tensor([0, 16], dtype=torch.int32), mesh, offsets_shard)

        tfn = thunder.jit(torch._grouped_mm, executors=executors_map[executor].executors_list())

        tfn(in_dtensor, w_dtensor, offsets_dtensor)

        trcs = thunder.last_traces(tfn)
        init_trc = trcs[0]

        from thunder.torch.experimental.dtensor_torch_and_prims import dtensor_grouped_mm

        assert any(bsym.sym == dtensor_grouped_mm for bsym in init_trc.bound_symbols)

    @common_utils.parametrize(
        "op, executor",
        product(dtensor_supported_opinfos, tuple(executors_map.keys())),
        lambda op, executor: op.name + "_" + executor,
    )
    def test_dtensor_opinfo(self, op: OpInfo, executor):
        if op.name in skip_opinfos:
            raise unittest.SkipTest(f"test_dtensor_opinfo: Skipping {op.name} as it is in skip_opinfos")

        if executor in op.skip_for_executor:
            raise unittest.SkipTest(f"test_dtensor_opinfo: Skipping {op.name} as it is in skip_for_executor")

        # NOTE: This test only tests for dtype=torch.float32 and requires_grad=True
        #       not for all dtype which are supported by the operation.
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        thunder_op = thunder.jit(op.op, executors=executors_map[executor].executors_list(), nv_enable_linear=True)
        torch_op = op.torch_reference

        tested_sample_count = 0

        for sample in op.sample_inputs("cpu", dtypes.float32, requires_grad=op.supports_grad):
            # Skip if non-contiguous inputs are not supported by the executor.
            if executor in op.skip_noncontiguous_for_executor and not sample.args[0].is_contiguous():
                continue

            # DTensorConverter converts inputs tensors to DTensor and creates DTensor
            # with possible placements based on the input shapes.
            # See - https://github.com/pytorch/pytorch/blob/eaa5d9d3d3dc642832b269b184f0c3ab8c990274/torch/testing/_internal/distributed/_tensor/common_dtensor.py#L521
            dtensor_converter = DTensorConverter(mesh, sample.args, sample.kwargs)
            for dtensor_args, dtensor_kwargs in dtensor_converter:
                if not dtensor_converter.successful():
                    continue

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

                trace = thunder.last_traces(thunder_op)[0]
                assert any("dtensor" in bsym.sym.name for bsym in trace.bound_symbols)

                if op.supports_grad:
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

    def test_dtensor_from_local_symbolic_values(self):
        num_devices = self.world_size
        mesh = DeviceMesh("cuda", list(range(num_devices)))

        dim_size = 8
        local_tensor = torch.randn(dim_size, dim_size, device="cuda")

        def fn(x):
            return DTensor.from_local(x, mesh, [Shard(0)])

        tjit = thunder.jit(fn, cache="symbolic values")

        actual = tjit(local_tensor)
        expected = DTensor.from_local(local_tensor, mesh, [Shard(0)])

        torch.testing.assert_close(actual, expected)
        assert thunder.cache_misses(tjit) == 1
        assert thunder.cache_hits(tjit) == 0

        dim_size = 16
        local_tensor = torch.randn(dim_size, dim_size, device="cuda")
        actual = tjit(local_tensor)
        expected = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        torch.testing.assert_close(actual, expected)
        assert thunder.cache_misses(tjit) == 1
        assert thunder.cache_hits(tjit) == 1


common_utils.instantiate_parametrized_tests(DTensorTest)

if __name__ == "__main__":
    common_utils.run_tests()
