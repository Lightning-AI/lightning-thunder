import pytest
import torch

import thunder
import thunder.core
from thunder.executors.sdpaex import sdpa_ex
from thunder.tests.framework import requiresCUDA, run_snippet
from thunder.tests.opinfos import get_opinfo
from collections import namedtuple

CudaVersion = namedtuple("CudaVersion", "major minor")


def device_version_support(
    device: None | str | torch.device | thunder.core.devices.Device,
    /,
    cuda_min_version: CudaVersion,
    cuda_max_version: CudaVersion,
) -> bool:
    """Check if the cuda capability of a given device is supported."""
    if not torch.cuda.is_available():
        return False

    dev: torch.device = thunder.core.devices.to_torch_device(device)
    cuda_major: int
    cuda_minor: int
    cuda_major, cuda_minor = torch.cuda.get_device_capability(dev)

    lower_bound = cuda_major >= cuda_min_version.major and cuda_minor >= cuda_min_version.minor
    upper_bound = cuda_major <= cuda_max_version.major and cuda_minor >= cuda_max_version.minor
    return lower_bound and upper_bound


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16], ids=("float32", "bfloat16", "float16")
)
@pytest.mark.parametrize("device,", ["cuda"])
@pytest.mark.parametrize("requires_grad", [True, False])
@requiresCUDA
def test_sdpa(device: str, dtype: torch.dtype, requires_grad: bool):
    batch = 10
    seq_len = 128
    num_heads = 4
    dim_per_head = 32

    query = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", requires_grad=requires_grad)
    key = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", requires_grad=requires_grad)
    value = torch.randn([batch, seq_len, num_heads, dim_per_head], device="cuda", requires_grad=requires_grad)

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    cfn = thunder.jit(fn, executors=[sdpa_ex])

    # Verifies the result is close to PyTorch
    thunder_result = cfn(query, key, value)
    torch_result = fn(query, key, value)

    torch.testing.assert_close(thunder_result, torch_result)

    # Verifies sdpa was called
    extrace = thunder.last_traces(cfn)[-1]
    assert any(
        bsym.sym.name == "sdpaex_grad_forward_scaled_dot_product_efficient_attention" for bsym in extrace.bound_symbols
    )

    if requires_grad:
        grad_output = torch.rand_like(thunder_result)
        actual_grads = torch.autograd.grad(thunder_result, (query, key, value), grad_outputs=grad_output)
        expected_grads = torch.autograd.grad(torch_result, (query, key, value), grad_outputs=grad_output)
        torch.testing.assert_close(actual_grads, expected_grads)


@requiresCUDA
def test_sdpa_autocast_flash():
    # Flash Attention sdpa only support Ampere and Hopper devices.
    # Skip this test on Volta and prior devices.
    torch_device = torch.device("cuda")
    if not device_version_support(torch_device, CudaVersion(8, 0), CudaVersion(9, 0)):
        pytest.skip(f"sdpa flash attention is not supported on {torch.cuda.get_device_name()}")

    batch = 1
    seq_len = 2
    num_heads = 14
    dim_per_head = 8

    q = torch.randn((batch, seq_len, num_heads, dim_per_head), device="cuda", dtype=torch.float32)
    k = torch.randn((batch, seq_len, num_heads, dim_per_head), device="cuda", dtype=torch.float32)
    v = torch.randn((batch, seq_len, num_heads, dim_per_head), device="cuda", dtype=torch.float32)

    def fn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    cfn = thunder.jit(fn, executors=[sdpa_ex])

    # Verifies the result is close to PyTorch
    for autocast_dtype in (torch.bfloat16, torch.float16):
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            # NOTE The new context manager torch.nn.attention.sdpa_kernel takes
            # an opt-in approach. Any backends not included in the arguments
            # list are disabled within the context and restored when exiting context.
            # For example, in this case, we would use:
            # torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)

            # Only use flash attention in this test
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)

            thunder_result = cfn(q, k, v)
            torch_result = fn(q, k, v)
            assert thunder_result.dtype == torch_result.dtype
            torch.testing.assert_close(thunder_result, torch_result)

            # Verifies sdpa was called
            extrace = thunder.last_traces(cfn)[-1]
            assert any(
                bsym.sym.name == "sdpafx_grad_forward_scaled_dot_product_efficient_attention"
                for bsym in extrace.bound_symbols
            )

            # Enable memory efficient and math backends
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)


def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)
    torch.testing.assert_close(thunder_result, torch_result, equal_nan=True, atol=1e-3, rtol=1e-4)

    head_size = sample.args[0].shape[-1]
    if head_size % 8 == 0:
        last_trace = thunder.last_traces(op)[-1]
        fused_sdpa_kernels = [
            "sdpaex_grad_forward_scaled_dot_product_efficient_attention",
            "sdpafx_grad_forward_scaled_dot_product_efficient_attention",
        ]
        assert any(bsym.sym.name in fused_sdpa_kernels for bsym in last_trace.bound_symbols)


@pytest.mark.parametrize("dtype", [torch.float32], ids=("float32",))
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_sdpa_torch_consistency(device: str, dtype: torch.dtype):
    # Enable math and memory-efficient sdpa options for Volta and prior devices
    torch_device = torch.device(device)
    if not device_version_support(torch_device, CudaVersion(8, 0), CudaVersion(9, 0)):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    op = get_opinfo("grad_forward_scaled_dot_product_attention")

    def fn(*args, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(*args, **kwargs)

    cfn = thunder.jit(fn, executors=[sdpa_ex])
    for sample in op.sample_input_generator(op, device, dtype, requires_grad=False):
        result = run_snippet(
            snippet_torch_consistency,
            op,
            device,
            dtype,
            cfn,
            op.torch_reference,
            sample,
        )

        if result is not None:
            return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=("f32", "f16", "bf16"))
@pytest.mark.parametrize("device,", ["cuda"])
@pytest.mark.parametrize("attn_mask_requires_grad", [True, False])
@requiresCUDA
def test_sdpa_attn_mask(attn_mask_requires_grad, device: str, dtype: torch.dtype):
    # Enable math and memory-efficient sdpa options for Volta and prior devices
    torch_device = torch.device(device)
    if not device_version_support(torch_device, CudaVersion(8, 0), CudaVersion(9, 0)):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    def func(q, k, v, atten_mask):
        tmp = atten_mask * atten_mask
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, tmp)

    query = torch.randn(1, 28, 128, 128, dtype=dtype, device=device, requires_grad=True)
    key = torch.randn(1, 28, 128, 128, dtype=dtype, device=device, requires_grad=True)
    value = torch.randn(1, 28, 128, 128, dtype=dtype, device=device, requires_grad=True)
    attn_mask = torch.randn(1, 1, 128, 128, dtype=dtype, device=device, requires_grad=attn_mask_requires_grad)

    query1 = query.detach().clone().requires_grad_()
    key1 = key.detach().clone().requires_grad_()
    value1 = value.detach().clone().requires_grad_()
    attn_mask1 = attn_mask.detach().clone().requires_grad_(attn_mask_requires_grad)

    expected = func(query, key, value, attn_mask)
    output = expected.mean()
    output.backward()

    jfun = thunder.jit(func, executors=[sdpa_ex])
    actual = jfun(query1, key1, value1, attn_mask1)
    output = actual.mean()
    output.backward()

    torch.testing.assert_close(actual, expected, atol=7e-3, rtol=7e-3)
    torch.testing.assert_close(attn_mask1.grad, attn_mask.grad)
    torch.testing.assert_close(query.grad, query1.grad)
    torch.testing.assert_close(key.grad, key1.grad)
    torch.testing.assert_close(value.grad, value1.grad)
