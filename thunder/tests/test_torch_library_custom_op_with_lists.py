from typing import TYPE_CHECKING

from lightning_utilities.core.imports import package_available
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._library.custom_ops import CustomOpDef

import thunder
from thunder.core import dtypes
from thunder.core import devices
from thunder.torch.custom_op import _deregister_custom_op
from thunder.torch.custom_op import _register_custom_op
from thunder.executors.custom_op_ex import custom_op_ex
from thunder.tests.framework import TorchExecutor
from thunder.tests.framework import instantiate

if TYPE_CHECKING:
    from thunder.core.symbol import BoundSymbol


@pytest.fixture(autouse=True)
def deregister_custom_op():
    yield
    _deregister_custom_op(list_mul)
    if has_triton_op:
        _deregister_custom_op(list_mul_triton)


@torch.library.custom_op("my_custom_op::list_mul", mutates_args=())
def list_mul(tensors: list[torch.Tensor], c: float | None = None, d: str = "") -> list[torch.Tensor]:
    if len(tensors) != 2:
        raise ValueError("The list of tensors must contain exactly two elements for this operation.")
    return [tensors[0] * tensors[1]]


@torch.library.register_kernel("my_custom_op::list_mul", "cpu")
def _(tensors: list[torch.Tensor], c: float | None = None, d: str = "") -> list[torch.Tensor]:
    return [
        torch.from_numpy(
            np.multiply(
                tensors[0].numpy(force=True),
                tensors[1].numpy(force=True),
            )
        )
    ]


@torch.library.register_kernel("my_custom_op::list_mul", "cuda")
def _(tensors: list[torch.Tensor], c: float | None = None, d: str = "") -> list[torch.Tensor]:
    return [tensors[0] * tensors[1]]


@torch.library.register_fake("my_custom_op::list_mul")
def _(tensors: list[torch.Tensor], c: float | None = None, d: str = "") -> list[torch.Tensor]:
    return [torch.empty_like(tensors[0])]


def setup_context_for_my_custom_op_list_mul(ctx, inputs, output) -> None:
    tensors_list, *_ = inputs
    ctx.save_for_backward(tensors_list[0], tensors_list[1])


def backward_of_my_custom_op_list_mul(ctx, grad) -> tuple[list[torch.Tensor], None, None]:
    a, b = ctx.saved_tensors
    return [torch.ops.my_custom_op.list_mul([grad, b]), torch.ops.my_custom_op.list_mul([grad, a])], None, None


torch.library.register_autograd(
    "my_custom_op::list_mul",
    backward_of_my_custom_op_list_mul,
    setup_context=setup_context_for_my_custom_op_list_mul,
)


has_triton_op = torch.cuda.is_available() and package_available("triton")
if has_triton_op:
    import triton
    import triton.language as tl

    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    @triton.jit
    def list_mul_triton_kernel(
        x_ptr,  # *Pointer* to first input vector.
        y_ptr,  # *Pointer* to second input vector.
        output_ptr,  # *Pointer* to output vector.
        n_elements,  # Size of the vector.
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
        # NOTE: `constexpr` so it can be used as a shape value.
    ):
        # There are multiple 'programs' processing different data. We identify which program
        # we are here:
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        # This program will process inputs that are offset from the initial data.
        # For instance, if you had a vector of length 256 and block_size of 64, the programs
        # would each access the elements [0:64, 64:128, 128:192, 192:256].
        # Note that offsets is a list of pointers:
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = offsets < n_elements
        # Load x and y from DRAM, masking out any extra elements in case the input is not a
        # multiple of the block size.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x * y
        # Write x + y back to DRAM.
        tl.store(output_ptr + offsets, output, mask=mask)

    @torch.library.triton_op("my_triton_op::list_mul", mutates_args=())
    def list_mul_triton(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(tensors) != 2:
            raise ValueError("The list of tensors must contain exactly two elements for this operation.")
        x = tensors[0]
        y = tensors[1]
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        torch.library.wrap_triton(list_mul_triton_kernel)[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return [output]

    torch.library.register_autograd(
        "my_triton_op::list_mul",
        backward_of_my_custom_op_list_mul,
        setup_context=setup_context_for_my_custom_op_list_mul,
    )


def _run_test(module_cls, custom_op: CustomOpDef, device: torch.device, dtype: torch.dtype):
    SHAPE = (8, 2)
    _symbol = _register_custom_op(custom_op)

    module = module_cls().to(device=device, dtype=dtype)
    jitted = thunder.jit(module, executors=[custom_op_ex])
    ref = module_cls().to(device=device, dtype=dtype)
    ref.load_state_dict(module.state_dict())

    x = torch.testing.make_tensor(SHAPE, device=device, dtype=dtype)
    y = torch.testing.make_tensor(SHAPE, device=device, dtype=dtype)
    inputs_list = [x, y]
    inputs_list_ref = [x.clone().detach() for x in inputs_list]

    ref_out = ref(inputs_list_ref)
    out = jitted(inputs_list)
    torch.testing.assert_close(ref_out, out)
    out.mean().backward()

    fwd_extrace = thunder.last_traces(jitted)[-1]
    bsym: BoundSymbol
    custom_ex_bsym_found: bool = False
    for bsym in fwd_extrace.bound_symbols:
        if bsym.sym.name == _symbol.name and bsym.sym.executor is custom_op_ex:
            custom_ex_bsym_found = True
    assert custom_ex_bsym_found


@instantiate(
    executors=(TorchExecutor,),
    devicetypes=(devices.DeviceType.CPU, devices.DeviceType.CUDA),
    dtypes=(dtypes.float32,),
)
def test_torch_library_custom_op(_, device: str, dtype: dtypes.dtype):
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

        def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
            h = torch.ops.my_custom_op.list_mul(tensors)
            activation = torch.relu(h[0])
            out = self.linear(activation)
            return out

    _run_test(MyModule, list_mul, devices.to_torch_device(device), dtypes.to_torch_dtype(dtype))


@pytest.mark.skipif(not has_triton_op, reason="triton is not available")
@instantiate(
    executors=(TorchExecutor,),
    devicetypes=(devices.DeviceType.CUDA,),
    dtypes=(dtypes.float32,),
)
def test_torch_library_triton_op(_, device: str, dtype: dtypes.dtype):
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

        def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
            h = torch.ops.my_triton_op.list_mul(tensors)
            activation = torch.relu(h[0])
            out = self.linear(activation)
            return out

    _run_test(MyModule, list_mul_triton, devices.to_torch_device(device), dtypes.to_torch_dtype(dtype))
