# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# NOTE: `down_size`, and `pack_uint4` are copied from PyTorch's test code.
#
# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# NOTE: `pytorch_nvfp4_quantize` and `linear_to_swizzled_128_4` are copied from NVIDIA's Fuser's test code.
from __future__ import annotations
from typing import TYPE_CHECKING
import math

from looseversion import LooseVersion
import torch
import torch.nn as nn
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate

if TYPE_CHECKING:
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe


__all__ = [
    "GroupedLinear",
    "GroupedSwiGLU",
    "Llama4MoE",
    "NVFP4InferenceGroupedLinear",
    "NVFP4InferenceGroupedSwiGLU",
    "nvfuser_f16a_nvfp4weight_scaled_grouped_mm",
]


# Ref: https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L972-L974
def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


# Ref: https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L977-L982
def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


# Ref: Based on `_bfloat16_to_float4_e2m1fn_x2` of https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L985-L990
def to_fp4(x: torch.Tensor) -> torch.Tensor:
    x = _f32_to_floatx_unpacked(x.float(), ebits=2, mbits=1)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L8-L10
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L125-L148
def pytorch_nvfp4_quantize(a, a_global_scale):
    BLOCK_SIZE = 16
    assert a.size(-1) % BLOCK_SIZE == 0, (
        "The inner-most dim must be divisible by block_size; Padding is not implemented."
    )
    assert a.is_contiguous(), "Only contiguous tensors are supported."

    original_shape = a.shape
    a_fp32 = a.float().reshape(original_shape[0], -1, BLOCK_SIZE)

    # Find absolute maximum along blockwise dimension
    max_abs = torch.amax(torch.abs(a_fp32), dim=-1)
    block_scale_fp32 = (max_abs / FLOAT4_E2M1_MAX).float()

    scaled_block_scale_fp32 = block_scale_fp32 * a_global_scale
    scaled_block_scale_fp8 = torch.clamp(
        scaled_block_scale_fp32,
        min=FLOAT8_E4M3_EPS,
        max=FLOAT8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    scaled_block_scale_fp8_fp32 = scaled_block_scale_fp8.to(torch.float)
    total_scale = scaled_block_scale_fp8_fp32 / a_global_scale
    a_scaled = a_fp32 / total_scale.unsqueeze(-1)
    a_scaled = torch.clamp(a_scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    a_scaled = a_scaled.view(original_shape)
    return to_fp4(a_scaled), scaled_block_scale_fp8


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L63-L82
# apply swizzled on block scaling factor:
# 1. apply padding to [mn_t * 128 , k_t * 4]
# 2. apply swizzle
def linear_to_swizzled_128_4(a_sf_linear: torch.Tensor):
    mn, sf_k = a_sf_linear.shape
    m_tiles = (mn + 128 - 1) // 128
    mn_padded = m_tiles * 128
    k_tiles = (sf_k + 4 - 1) // 4
    k_padded = k_tiles * 4
    if mn_padded != mn or k_padded != sf_k:
        a_sf_padded = torch.empty(mn_padded, k_padded, dtype=a_sf_linear.dtype, device=a_sf_linear.device)
        a_sf_padded[0:mn, 0:sf_k] = a_sf_linear
    else:
        a_sf_padded = a_sf_linear
    # details about layout requirement on block-wise scaling factor
    # https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
    tmp = torch.reshape(a_sf_padded, (m_tiles, 4, 32, k_tiles, 4))
    return tmp.transpose(1, 3).reshape(mn_padded, k_padded)[:mn, :sf_k]


@torch.inference_mode()
def quantize_linear_weight_to_nvfp4(
    weight: torch.Tensor | nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize weight to nvfp4, returning (packed) e2m1 weight, e4m3 scale factor, fp32 global scale."""
    global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / weight.float().abs().amax()).to(torch.float32)
    fp4_weight, weight_scaling_factor = pytorch_nvfp4_quantize(weight, global_scale)
    weight_scale_interleaved = linear_to_swizzled_128_4(weight_scaling_factor)
    return fp4_weight, weight_scale_interleaved, global_scale


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L151-L152
def round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L13-L22
kE2M1ToFloatArray = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L25-L32
# Convert FP4 into FP32
def e2m1_to_fp32(int4_value):
    signBit = int4_value & 0x8
    int4_absValue = int4_value & 0x7
    float_result = kE2M1ToFloatArray[int4_absValue]
    if signBit:
        float_result = -float_result
    return float_result


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L35-L49
# Unpack float4_e2m1fn_x2 into two separate fp32 values
def unpack_fp4_bytes(a, dtype):
    assert a.dtype == torch.float4_e2m1fn_x2
    m, n = a.shape
    a = a.view(torch.uint8).flatten()
    upper_half_byte = (a & 0xF0) >> 4
    lower_half_byte = a & 0x0F
    upper_half_float = torch.tensor([e2m1_to_fp32(x) for x in upper_half_byte]).to(a.device)
    lower_half_float = torch.tensor([e2m1_to_fp32(x) for x in lower_half_byte]).to(a.device)
    out = torch.stack((lower_half_float, upper_half_float), dim=-1).reshape(m, n * 2)
    return out


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L55-L60
# restore swizzled on block scaling factor:
# 1. restore swizzle
# 2. removes padding via slicing to [:mn, :k]
def swizzled_to_linear_128_4(a_sf_swizzled: torch.Tensor, mn, k):
    mn_padded, sf_k_padded = a_sf_swizzled.shape
    m_tiles = mn_padded // 128
    k_tiles = sf_k_padded // 4
    tmp = torch.reshape(a_sf_swizzled, (m_tiles, k_tiles, 32, 4, 4))
    return tmp.transpose(1, 3).reshape(mn_padded, sf_k_padded)[:mn, :k]


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L85-L101
def dequantize_to_dtype(tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.float4_e2m1fn_x2
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = unpack_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = swizzled_to_linear_128_4(tensor_sf, m, k)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out


# NOTE: This custom op is registered with nvfuser translator in benchmark_inference.py
# using _register_nvfuser_translator. See benchmark_inference._register_nvfp4_ops().
@torch.library.custom_op("nvf_cutlass::f16a_nvfp4weight_scaled_grouped_mm", mutates_args=())
def nvfuser_f16a_nvfp4weight_scaled_grouped_mm(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> torch.Tensor:
    hp_weight = torch.empty(
        (fp4_weight.size(0), fp4_weight.size(1) * 2, fp4_weight.size(2)),
        device=activation.device,
        dtype=activation.dtype,
    )
    for i in range(fp4_weight.size(0)):
        # NOTE: dequantize here doesn't look right, since we have (g, k, n)
        hp_weight[i] = dequantize_to_dtype(
            fp4_weight[i], weight_scaling_factor[i], weight_global_scale[i], activation.dtype, fp4_weight.device, 16
        )
    return grouped_mm(activation, hp_weight, offsets)


@torch.library.register_fake("nvf_cutlass::f16a_nvfp4weight_scaled_grouped_mm")
def _(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> torch.Tensor:
    # fp4_weight shape: (groups, in_features // 2, out_features)
    # Validate that activation has at least 1 dimension
    if activation.ndim == 0:
        raise ValueError(f"Expected activation to have at least 1 dimension, got {activation.ndim}")

    if (
        len(
            {
                t.device
                for t in [
                    activation,
                    fp4_weight,
                    weight_scaling_factor,
                    weight_global_scale,
                    offsets,
                    blockscale_offsets,
                    problem_sizes,
                ]
            }
        )
        != 1
    ):
        raise ValueError("Expected all inputs to be on the same device.")

    # After unpacking: (groups, in_features, out_features)
    # Output shape should match activation.shape[:-1] + (out_features,)
    # This handles both 2D (tokens, hidden) and 3D (batch, seq_len, hidden) inputs
    out_features = fp4_weight.size(2)
    output_shape = activation.shape[:-1] + (out_features,)
    return torch.empty(output_shape, device=activation.device, dtype=torch.bfloat16)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _group_sizes_from_offsets(offsets: torch.Tensor) -> list[int]:
    group_sizes = []
    prev = 0
    if isinstance(offsets, DTensor):
        assert offsets.placements == (Replicate(),)
        offsets = offsets.to_local()

    for offset in offsets:
        group_sizes.append(offset - prev)
        prev = offset
    return group_sizes


if LooseVersion(torch.__version__) >= LooseVersion("2.8.0"):
    # Required otherwise, there is a graph-break.
    _grouped_mm = torch.compiler.allow_in_graph(torch._grouped_mm)


# This function should be replaced with torch._grouped_mm.  However,
# torch._grouped_mm is yet to be usable because it requires offsets being
# multiples of 16.
def grouped_mm(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        # NOTE: This path also works for `thunder.jit` as it has a lookaside for `torch.compiler.is_compiling`.
        return _grouped_mm(a, b, offsets)

    group_sizes = _group_sizes_from_offsets(offsets)
    group_outs = []
    for idx, group_a in enumerate(a.split(group_sizes)):
        group_outs.append(group_a @ b[idx])
    return torch.cat(group_outs)


class GroupedLinear(nn.Module):
    def __init__(self, groups: int, in_features: int, out_features: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, out_features, in_features, dtype=dtype, device=device))
        # Initialize the weight in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return grouped_mm(hidden_states, self.weight.transpose(-1, -2), offsets)


@torch.inference_mode()
def quantize_grouped_linear_weight_to_nvfp4(
    weight: torch.Tensor | nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize grouped linear's weight to nvfp4

    Args:
        weight: Parameter of `GroupedLinear` of [g, n, k]

    Returns:
        fp4_weight: [g, k // 2, n]
        scale_factors: [g, n, k // 16]
        global_scales: [g]

    Note:
        The reason we choose different layout of weight is to avoid performance
        regression for bf16. See
        https://github.com/Lightning-AI/lightning-thunder/pull/2659
    """
    assert weight.ndim == 3, "Weight must be a 3D tensor"

    device: torch.device = weight.device
    g, n, k = weight.size()

    with device:
        fp4_weight = torch.empty((g, n, k // 2), dtype=torch.float4_e2m1fn_x2)
        global_scales = torch.empty((g,), dtype=torch.float32)
        scale_factors = torch.empty((g, n, k // 16), dtype=torch.float8_e4m3fn)

    weight = weight.contiguous()
    for i in range(g):
        cur_weight = weight[i]
        global_scales[i] = cur_weight.abs().amax()
        cur_fp4_weight, cur_scale_factors = pytorch_nvfp4_quantize(cur_weight, global_scales[i])
        fp4_weight[i] = cur_fp4_weight
        scale_factors[i] = linear_to_swizzled_128_4(cur_scale_factors)

    return fp4_weight.transpose(-1, -2), scale_factors, global_scales


class NVFP4InferenceGroupedLinear(nn.Module):
    def __init__(
        self,
        fp4_weight: torch.Tensor,
        weight_scaling_factor: torch.Tensor,
        weight_global_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("fp4_weight", fp4_weight)
        self.register_buffer("weight_scaling_factor", weight_scaling_factor)
        self.register_buffer("weight_global_scale", weight_global_scale)

    @property
    def out_features(self) -> int:
        return self.fp4_weight.size(2)

    @property
    def in_features(self) -> int:
        return self.fp4_weight.size(1) * 2

    @staticmethod
    def compute_auxiliary_tensors(
        hidden_states: torch.Tensor,
        offsets: torch.Tensor,
        out_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute blockscale_offsets and problem_sizes for grouped mm.

        These can be computed once and reused across multiple forward calls with the same offsets.
        """
        tokens_per_group = offsets[1:] - offsets[:-1]
        problem_sizes = torch.stack(
            [
                tokens_per_group,
                torch.full_like(tokens_per_group, out_features),
                torch.full_like(tokens_per_group, hidden_states.size(1)),
            ],
            dim=1,
        )
        # Calculate block-scale offsets: round up to 128, then cumsum with initial 0
        rounded_tokens = ((tokens_per_group + 127) // 128) * 128
        blockscale_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=tokens_per_group.device),
                torch.cumsum(rounded_tokens, 0, dtype=torch.int32),
            ]
        )[0:-1]
        return blockscale_offsets, problem_sizes

    # TODO: Update this accordingly to the progress of nvfp4 kernel implementation.
    def forward(
        self,
        hidden_states: torch.Tensor,
        offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor | None = None,
        problem_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if blockscale_offsets is None or problem_sizes is None:
            # Compute them if not provided (backward compatibility)
            out_features = self.out_features
            blockscale_offsets, problem_sizes = self.compute_auxiliary_tensors(hidden_states, offsets, out_features)
        return torch.ops.nvf_cutlass.f16a_nvfp4weight_scaled_grouped_mm(
            hidden_states,
            self.fp4_weight,
            self.weight_scaling_factor,
            self.weight_global_scale,
            offsets[:-1],
            blockscale_offsets,
            problem_sizes,
        )

    @staticmethod
    def from_grouped_linear(grouped_linear: GroupedLinear, fqn: str | None = None) -> NVFP4InferenceGroupedLinear:
        """Create an NVFP4InferenceGroupedLinear from a GroupedLinear.

        Args:
            grouped_linear (GroupedLinear): The source GroupedLinear.
            fqn (str or None): Fully qualified name. Currently unused; reserved for future use or compatibility.
        """
        weight = grouped_linear.weight
        fp4_weight, weight_scaling_factor, weight_global_scale = quantize_grouped_linear_weight_to_nvfp4(weight)
        return NVFP4InferenceGroupedLinear(
            fp4_weight,
            weight_scaling_factor,
            weight_global_scale,
        )


class GroupedSwiGLU(nn.Module):
    def __init__(self, groups: int, hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.gate_proj = GroupedLinear(groups, hidden_size, intermediate_size, dtype, device)
        self.up_proj = GroupedLinear(groups, hidden_size, intermediate_size, dtype, device)
        self.down_proj = GroupedLinear(groups, intermediate_size, hidden_size, dtype, device)

    def forward(self, hidden_states: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states, offsets)) * self.up_proj(hidden_states, offsets),
            offsets,
        )


class NVFP4InferenceGroupedSwiGLU(nn.Module):
    """NVFP4 GroupedSwiGLU that efficiently reuses auxiliary tensor computations."""

    def __init__(
        self,
        gate_proj: NVFP4InferenceGroupedLinear,
        up_proj: NVFP4InferenceGroupedLinear,
        down_proj: NVFP4InferenceGroupedLinear,
    ):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, hidden_states: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # Compute auxiliary tensors once for all three operations
        intermediate_features = self.gate_proj.out_features
        blockscale_offsets_gate, problem_sizes_gate = NVFP4InferenceGroupedLinear.compute_auxiliary_tensors(
            hidden_states, offsets, intermediate_features
        )

        gate_out = self.gate_proj(hidden_states, offsets, blockscale_offsets_gate, problem_sizes_gate)
        up_out = self.up_proj(hidden_states, offsets, blockscale_offsets_gate, problem_sizes_gate)

        intermediate = torch.nn.functional.silu(gate_out) * up_out

        # For down_proj, we need different problem_sizes (different output features)
        hidden_features = self.down_proj.out_features
        blockscale_offsets_down, problem_sizes_down = NVFP4InferenceGroupedLinear.compute_auxiliary_tensors(
            intermediate, offsets, hidden_features
        )

        return self.down_proj(intermediate, offsets, blockscale_offsets_down, problem_sizes_down)

    @staticmethod
    def from_grouped_swiglu(grouped_swiglu: GroupedSwiGLU, fqn: str | None = None) -> NVFP4InferenceGroupedSwiGLU:
        """Create an NVFP4InferenceGroupedSwiGLU from a GroupedSwiGLU.

        Args:
            grouped_swiglu (GroupedSwiGLU): The source GroupedSwiGLU.
            fqn (str or None): Fully qualified name. Currently unused; reserved for future use or compatibility.
        """
        gate_proj = NVFP4InferenceGroupedLinear.from_grouped_linear(grouped_swiglu.gate_proj)
        up_proj = NVFP4InferenceGroupedLinear.from_grouped_linear(grouped_swiglu.up_proj)
        down_proj = NVFP4InferenceGroupedLinear.from_grouped_linear(grouped_swiglu.down_proj)
        return NVFP4InferenceGroupedSwiGLU(gate_proj, up_proj, down_proj)


# Slightly modified version of `thunder.tests.test_networks.Llama4MoE`
# to have the same singature as transformers' Llama4TextMoe -- in this file
# return values include `router_logits`.
# Ref: https://github.com/huggingface/transformers/blob/ff8b88a9/src/transformers/models/llama4/modeling_llama4.py#L147-L165
class Llama4MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.hidden_size,
            config.num_routed_experts,
            bias=False,
            dtype=config.dtype,
            device=config.device,
        )
        self.shared_experts = SwiGLU(
            config.hidden_size,
            config.intermediate_size * config.num_shared_experts,
            config.dtype,
            config.device,
        )
        self.routed_experts = GroupedSwiGLU(
            config.num_routed_experts,
            config.hidden_size,
            config.intermediate_size,
            config.dtype,
            config.device,
        )

    @staticmethod
    def from_transformers_llama4textmoe(moe: Llama4TextMoe) -> Llama4MoE:
        """[CAUTION] A converter written by Gemini 2.5."""
        from thunder.tests.llama4_moe import Config

        # 1. Create a config for the Llama4MoE model from the transformers config
        config = Config(
            "Llama4MoE",
            hidden_size=moe.hidden_dim,
            intermediate_size=moe.experts.intermediate_size,
            num_routed_experts=moe.num_experts,
            num_shared_experts=1,  # Based on HF implementation having one shared_expert
            dtype=moe.router.weight.dtype,
            device=moe.router.weight.device,
        )

        # 2. Create an instance of our Llama4MoE
        new_moe = Llama4MoE(config)

        # 3. Copy the router weights (called 'gate' in our implementation)
        new_moe.gate.weight.data.copy_(moe.router.weight.data)

        # 4. Copy the shared expert weights
        new_moe.shared_experts.gate_proj.weight.data.copy_(moe.shared_expert.gate_proj.weight.data)
        new_moe.shared_experts.up_proj.weight.data.copy_(moe.shared_expert.up_proj.weight.data)
        new_moe.shared_experts.down_proj.weight.data.copy_(moe.shared_expert.down_proj.weight.data)

        # 5. For the routed experts, we need to handle the combined gate_up_proj
        # to match GroupedLinear
        # https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/llama4/modeling_llama4.py#L55-L57
        # HF format: (groups, hidden_size, 2 * intermediate_size)
        # Our format: (groups, intermediate_size, hidden_size)

        # Split into gate and up projections
        gate_proj_w, up_proj_w = moe.experts.gate_up_proj.chunk(2, dim=2)

        new_moe.routed_experts.gate_proj.weight.data.copy_(gate_proj_w.transpose(-1, -2))
        new_moe.routed_experts.up_proj.weight.data.copy_(up_proj_w.transpose(-1, -2))

        # Handle down_proj
        # HF format: (groups, intermediate_size, hidden_size)
        # Our format: (groups, hidden, intermediate_size)
        new_moe.routed_experts.down_proj.weight.data.copy_(moe.experts.down_proj.transpose(-1, -2))

        return new_moe

    def run_routed_experts(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))  # [s, h]

        router_logits = self.gate(hidden_states)  # [s, n]
        topk_weight, topk_ids = router_logits.topk(1)  # [s, 1]
        router_scores = topk_weight.sigmoid()  # [s, 1]
        hidden_states = hidden_states * router_scores  # [s, h]

        counts = torch.zeros(
            topk_ids.size(0),
            self.config.num_routed_experts,
            device=topk_ids.device,
            dtype=torch.int32,
        )  # [s, n]
        counts = counts.scatter(1, topk_ids, 1)  # [s, n]
        tokens_per_expert = counts.sum(0)  # [n]

        token_ids_sorted_by_expert_id = topk_ids.view(-1).argsort()  # [s]
        tokens_sorted_by_expert_id = hidden_states[token_ids_sorted_by_expert_id]  # [s, h]

        # Without `torch.int32`, we see `RuntimeError: Offsets tensor must be integer (int32) tensor, but got torch.int64.`
        # from PyTorch when calling _grouped_mm.
        # Prepend 0 to offsets for correct grouping
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=tokens_per_expert.device),
                torch.cumsum(tokens_per_expert, 0, dtype=torch.int32),
            ]
        )[:-1]  # [n]
        outs_sorted_by_expert_id = self.routed_experts(tokens_sorted_by_expert_id, offsets)  # [s, h]

        token_ids_sorted_by_expert_inverse_id = torch.argsort(token_ids_sorted_by_expert_id)
        outs_sorted_by_token_id = outs_sorted_by_expert_id[token_ids_sorted_by_expert_inverse_id]

        return outs_sorted_by_token_id.view(batch_size, seq_len, -1), router_logits.view(batch_size, seq_len, -1)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outs_sorted_by_token_id, router_logits = self.run_routed_experts(hidden_states)
        return self.shared_experts(hidden_states) + outs_sorted_by_token_id, router_logits
