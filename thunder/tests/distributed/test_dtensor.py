from collections.abc import Sequence
import os

import pytest
import torch

if not torch.distributed.is_available() or not (torch.cuda.is_available() and torch.distributed.is_nccl_available()):
    pytest.skip(allow_module_level=True)

import thunder

from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorConverter


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


# NOTE: OpInfo may use `clang` or `ltorch` ops to be jitted with thunder.jit.
#       However, for the current DTensor implementation, we add a dispatch in the `torch` operation lookaside
#       to choose between DTensor supported symbol (from `dtensor_torch_and_prims.py`) or the usual `ltorch` symbol.
#       This is why we need to make sure that the OpInfo uses PyTorch native op as `op` which is passed to thunder.jit.
class DTensorOpInfo:
    def __init__(self, *, name, op, torch_reference, supports_grad, sample_inputs):
        self.name = name
        assert "torch" in op.__module__, "OpInfo must use PyTorch native op as `op` which is passed to thunder.jit"
        self.op = op
        self.torch_reference = torch_reference
        # NOTE: Not all DTensor ops support grad initially, use this to disable grad tests for them
        self.supports_grad = supports_grad
        # NOTE: This should generally reuse the sample_inputs from the OpInfo
        self.sample_inputs = sample_inputs


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
)

skip_opinfos = (
    # RuntimeError: Metadata (placement and mesh) has changed for cotangent between tracing and runtimeduring tracing
    # it was Spec(S(1) on (1, 2, 1)) but at runtime it is Spec(S(1) on (1, 2, 1)).
    "reshape",
)


@pytest.fixture(scope="module", autouse=True)
def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(f"cuda:{local_rank}")


def get_num_devices():
    world_size = int(os.environ["WORLD_SIZE"])
    return world_size


@pytest.mark.parametrize("executor", tuple(executors_map.keys()))
@pytest.mark.parametrize("fn_key", functions_to_test.keys())
def test_dtensor_basic_op(executor, fn_key):
    num_devices = get_num_devices()
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


def test_dtensor_unsupported():
    num_devices = get_num_devices()
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


def test_dtensor_unsupported_mixed_input():
    num_devices = get_num_devices()
    mesh = DeviceMesh("cuda", list(range(num_devices)))

    dim_size = 16

    def fn(x, w):
        return torch.div(x, w)

    w = torch.randn(dim_size, dim_size, requires_grad=True)

    in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=True), mesh, [Shard(0)])

    tmodel = thunder.jit(fn, executors=thunder.get_always_executors())
    with pytest.raises(AssertionError):
        tmodel(in_dtensor, w)


def test_dtensor_incorrect_cotangent():
    num_devices = get_num_devices()
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


@pytest.mark.parametrize("executor", tuple(executors_map.keys()))
def test_dtensor_convert_element_type(executor):
    from thunder.torch.experimental.dtensor_torch_and_prims import dtensor_convert_element_type_prim

    num_devices = get_num_devices()
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


@pytest.mark.parametrize("executor", tuple(executors_map.keys()))
def test_dtensor_broadcast_in_dim(executor):
    from thunder.torch.experimental.dtensor_torch_and_prims import dtensor_broadcast_in_dim_prim

    num_devices = get_num_devices()
    mesh = DeviceMesh("cuda", list(range(num_devices)))
    dim_size = 16
    in_dtensor = distribute_tensor(torch.randn(dim_size, dim_size, requires_grad=False), mesh, [Shard(0)])

    def fn(x):
        return dtensor_broadcast_in_dim_prim(x, (dim_size, dim_size), (0, 1))

    tmodel = thunder.jit(fn, executors=executors_map[executor].executors_list())
    actual = tmodel(in_dtensor)
    expected = in_dtensor.broadcast_to((dim_size, dim_size))

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("op", dtensor_supported_opinfos)
@pytest.mark.parametrize("executor", tuple(executors_map.keys()))
def test_dtensor_opinfo(op: OpInfo, executor):
    if op.name in skip_opinfos:
        raise pytest.skip(f"test_dtensor_opinfo: Skipping {op.name} as it is in skip_opinfos")

    # NOTE: This test only tests for dtype=torch.float32 and requires_grad=True
    #       not for all dtype which are supported by the operation.
    num_devices = get_num_devices()
    mesh = DeviceMesh("cuda", list(range(num_devices)))

    thunder_op = thunder.jit(op.op, executors=executors_map[executor].executors_list(), nv_enable_linear=True)
    torch_op = op.torch_reference

    tested_sample_count = 0

    for sample in op.sample_inputs("cpu", dtypes.float32, requires_grad=op.supports_grad):
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
