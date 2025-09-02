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
from thunder.torch.custom_op import _register_custom_op
from thunder.executors.custom_op_ex import custom_op_ex
from thunder.tests.framework import TorchExecutor
from thunder.tests.framework import instantiate

if TYPE_CHECKING:
    from thunder.core.symbol import BoundSymbol


@torch.library.custom_op("my_custom_op::mul", mutates_args=())
def mul(a: torch.Tensor, b: torch.Tensor, c: float | None = None, d: str = "") -> torch.Tensor:
    return a * b


@torch.library.register_kernel("my_custom_op::mul", "cpu")
def _(a: torch.Tensor, b: torch.Tensor, c: float | None = None, d: str = "") -> torch.Tensor:
    return torch.from_numpy(
        np.multiply(
            a.numpy(force=True),
            b.numpy(force=True),
        )
    )


@torch.library.register_kernel("my_custom_op::mul", "cuda")
def _(a: torch.Tensor, b: torch.Tensor, c: float | None = None, d: str = "") -> torch.Tensor:
    return a * b


@torch.library.register_fake("my_custom_op::mul")
def _(a: torch.Tensor, b: torch.Tensor, c: float | None = None, d: str = "") -> torch.Tensor:
    return torch.empty_like(a)


def setup_context_for_my_custom_op_mul(ctx, inputs, output) -> None:
    a, b, *_ = inputs
    ctx.save_for_backward(a, b)


def backward_of_my_custom_op_mul(ctx, grad) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    a, b = ctx.saved_tensors
    return torch.ops.my_custom_op.mul(grad, b), torch.ops.my_custom_op.mul(grad, a), None, None


torch.library.register_autograd(
    "my_custom_op::mul",
    backward_of_my_custom_op_mul,
    setup_context=setup_context_for_my_custom_op_mul,
)


has_triton_op = torch.cuda.is_available() and package_available("triton")
if has_triton_op:
    import triton
    import triton.language as tl

    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    @triton.jit
    def mul_triton_kernel(
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

    @torch.library.triton_op("my_triton_op::mul", mutates_args=())
    def mul_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        torch.library.wrap_triton(mul_triton_kernel)[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

    torch.library.register_autograd(
        "my_triton_op::mul",
        backward_of_my_custom_op_mul,
        setup_context=setup_context_for_my_custom_op_mul,
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
    x_ref = x.clone().detach()
    y_ref = y.clone().detach()

    ref_out = ref(x_ref, y_ref)
    out = jitted(x, y)
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

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            h = torch.ops.my_custom_op.mul(x, y)
            activation = torch.relu(h)
            out = self.linear(activation)
            return out

    _run_test(MyModule, mul, devices.to_torch_device(device), dtypes.to_torch_dtype(dtype))


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

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            h = torch.ops.my_triton_op.mul(x, y)
            activation = torch.relu(h)
            out = self.linear(activation)
            return out

    _run_test(MyModule, mul_triton, devices.to_torch_device(device), dtypes.to_torch_dtype(dtype))
