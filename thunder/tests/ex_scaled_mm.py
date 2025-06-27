import torch

E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
EPS: float = 1e-12
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2

import thunder


def amax_to_scale(amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype):
    """Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: the float8 dtype.
        orig_dtype: The original dtype of the tensor.
    """
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == e4m3_type:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == e5m2_type:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # Ensure the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=torch.finfo(torch.float16).max)

    scale.copy_(res)
    return scale


def to_fp8_saturated(x: torch.Tensor, fp8_dtype: torch.dtype):
    if fp8_dtype == e4m3_type:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif fp8_dtype == e5m2_type:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    else:
        raise ValueError(f"to_fp8_saturated(): Unsupported fp8_dtype: {fp8_dtype}")

    return x.to(fp8_dtype)


def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype, dim=None):
    if dim is None:
        amax = torch.max(torch.abs(x))
    else:
        amax = torch.max(torch.abs(x), dim=dim, keepdim=True).values

    return amax_to_scale(amax, float8_dtype, x.dtype)


# Example usage
M, K, N = 128, 256, 512
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
y = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).t()  # Ensure b is column-major for cuBLASLt

x_scales = tensor_to_scale(x, e4m3_type, dim=1).reciprocal()
y_scales = tensor_to_scale(y, e4m3_type, dim=0).reciprocal()
print(x)
x_fp8 = to_fp8_saturated(x / x_scales, e4m3_type)
print(x_fp8)
y_fp8 = to_fp8_saturated(y / y_scales, e4m3_type)

x = torch._scaled_mm(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)
print("Result shape:", x.shape)

def test_scaled_mm(a, b, a_scale=None, b_scale=None):
    return torch._scaled_mm(a, b, scale_a=a_scale, scale_b=b_scale,  out_dtype=torch.bfloat16)


fn = thunder.jit(test_scaled_mm)
result = fn(x_fp8, y_fp8, x_scales, y_scales)
print("Result shape from thunder.jit:", result.shape)