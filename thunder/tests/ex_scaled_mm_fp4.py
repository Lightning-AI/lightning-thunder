import torch
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked, ceil_div, to_blocked

E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
EPS: float = 1e-12
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2
F8E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max

BLOCK_SIZE = 16
FP4_MAX_VAL = 6.0

def data_to_nvfp4_scale(x, block_size):
    orig_shape = x.shape
    x = x.reshape(-1, block_size)
    max_abs = torch.amax(torch.abs(x), 1) + 1e-12

    # x_orig_max / scale = x_in_fp4_domain_max
    # x_orig_max / x_in_fp4_domain_max = scale
    scale = max_abs / FP4_MAX_VAL

    # for the purposes of this function, just clamp to representable range of
    # `torch.float8_e4m3fn`. In real code, we would expect the modeling code to
    # handle this before the input data hits this function.
    scale = scale.clamp(max=F8E4M3_MAX_VAL)

    # cast to target dtype
    scale = scale.to(torch.float8_e4m3fn)
    scale = scale.reshape(orig_shape[0], -1)
    return scale

FP4_EBITS, FP4_MBITS = 2, 1

def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)

def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))

def _bfloat16_to_float4_e2m1fn_x2(x):
    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x



M, K, N = 128, 256, 512
A_ref = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 1000
B_ref = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 1000  # Ensure b is column-major for cuBLASLt

A_scale = data_to_nvfp4_scale(A_ref, BLOCK_SIZE)
B_scale = data_to_nvfp4_scale(B_ref, BLOCK_SIZE)

max_val = FP4_MAX_VAL
min_val = -1 * max_val

A = (A_ref.reshape(-1, BLOCK_SIZE) / A_scale.reshape(M * ceil_div(K, BLOCK_SIZE), 1).bfloat16()).reshape(M, K)
A = A.clamp(min=min_val, max=max_val)
A = _bfloat16_to_float4_e2m1fn_x2(A)
B = (B_ref.reshape(-1, BLOCK_SIZE) / B_scale.reshape(N * ceil_div(K, BLOCK_SIZE), 1).bfloat16()).reshape(N, K)
B = B.clamp(min=min_val, max=max_val)
B = _bfloat16_to_float4_e2m1fn_x2(B)


A_scale = to_blocked(A_scale)
B_scale = to_blocked(B_scale)

C = torch._scaled_mm(
    A,
    B.t(),
    A_scale,
    B_scale,
    out_dtype=torch.bfloat16,
    use_fast_accum=False,
)