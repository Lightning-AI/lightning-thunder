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

if TYPE_CHECKING:
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe


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
def to_fp4(a: torch.Tensor) -> torch.Tensor:
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


@torch.library.custom_op("nvf_cutlass::nvfp4_scaled_mm", mutates_args=())
def nvfp4_scaled_mm(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # N.B. This would not work but it's fine as it's going to be overriden by Thunder.
    hp_weight = fp4_weight * weight_scaling_factor * weight_global_scale
    return activation @ hp_weight + bias


@torch.library.register_fake("nvf_cutlass::nvfp4_scaled_mm")
def _(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty((activation.size(0), fp4_weight.size(0)), device=activation.device, dtype=activation.dtype)


@torch.library.custom_op("nvf_cutlass::nvfp4_scaled_grouped_mm", mutates_args=())
def nvfp4_scaled_grouped_mm(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    ab_strides: torch.Tensor,
    c_strides: torch.Tensor,
    offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> torch.Tensor:
    hp_weight = fp4_weight * weight_scaling_factor * weight_global_scale
    return grouped_mm(activation, hp_weight, offsets)


@torch.library.register_fake("nvf_cutlass::nvfp4_scaled_grouped_mm")
def _(
    activation: torch.Tensor,
    fp4_weight: torch.Tensor,
    weight_scaling_factor: torch.Tensor,
    weight_global_scale: torch.Tensor,
    ab_strides: torch.Tensor,
    c_strides: torch.Tensor,
    offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((activation.size(0), fp4_weight.size(2)), device=activation.device, dtype=activation.dtype)


class NVFP4InferenceLinear(nn.Module):
    """NVFP4 Linear layer for Inference.

    Weight, its scaling factor, its global scale, and bias are registered as a buffer, not a parameter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        fp4_weight: torch.Tensor | nn.Parameter,
        weight_scaling_factor: torch.Tensor | nn.Parameter,
        weight_global_scale: torch.Tensor | nn.Parameter | None,
        bias: torch.Tensor | nn.Parameter | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("fp4_weight", fp4_weight)
        self.register_buffer("weight_scaling_factor", weight_scaling_factor)
        self.register_buffer("weight_global_scale", weight_global_scale)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return mm_a16_nvfp4weight(x, self.fp4_weight, self.weight_scaling_factor, self.weight_global_scale, self.bias)
        raise NotImplementedError()

    @classmethod
    def from_linear(linear: nn.Linear) -> NVFP4InferenceLinear:
        weight = linear.weight
        bias = linear.bias
        out_features, in_features = weight.size()
        fp4_weight, weight_scaling_factor, weight_global_scale = quantize_linear_weight_to_nvfp4(weight)
        return NVFP4InferenceLinear(
            in_features,
            out_features,
            fp4_weight=fp4_weight,
            weight_scaling_factor=weight_scaling_factor,
            weight_global_scale=weight_global_scale,
            bias=bias,
        )


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
    for group_a, group_b in zip(a.split(group_sizes), b.unbind()):
        group_outs.append(group_a @ group_b)
    return torch.cat(group_outs)


class GroupedLinear(nn.Module):
    def __init__(self, groups: int, in_features: int, out_features: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, in_features, out_features, dtype=dtype, device=device))
        # Initialize the weight in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return grouped_mm(hidden_states, self.weight, offsets)


@torch.inference_mode()
def quantize_grouped_linear_weight_to_nvfp4(
    weight: torch.Tensor | nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize grouped linear's weight to nvfp4

    Args:
        weight: Parameter of `GroupedLinear` of [g, n, k]
        m: hidden_states.size(0)
        tokens_per_expert_neg_one:

    Returns:
        fp4_weight: [g, n, k // 2]
        scale_factors: [g, n, k // 16]
        global_scales: [g]
        ab_strides: [g]
        c_strides: [g]
    """
    assert weight.ndim == 3, "Weight must be a 3D tensor"

    device: torch.device = weight.device
    g, n, k = weight.size()

    with device:
        ab_strides = torch.full((g,), k, dtype=torch.int32)
        c_strides = torch.full((g,), n, dtype=torch.int32)

        fp4_weight = torch.empty((g, n, k // 2), dtype=torch.float4_e2m1fn_x2)
        global_scales = torch.empty((g,), dtype=torch.float32)
        scale_factors = torch.empty((g, n, k // 16), dtype=torch.float8_e4m3fn)

    for i in range(g):
        cur_weight = weight[i]
        global_scales[i] = cur_weight.abs().amax()
        cur_fp4_weight, cur_scale_factors = pytorch_nvfp4_quantize(cur_weight, global_scales[i])
        fp4_weight[i] = cur_fp4_weight
        scale_factors[i] = linear_to_swizzled_128_4(cur_scale_factors)

    return fp4_weight, scale_factors, global_scales, ab_strides, c_strides


class NVFP4InferenceGroupedLinear(nn.Module):
    def __init__(
        self,
        fp4_weight: torch.Tensor,
        weight_scaling_factor: torch.Tensor,
        weight_global_scale: torch.Tensor,
        ab_strides: torch.Tensor,
        c_strides: torch.Tensor,
    ) -> None:
        self.register_buffer("fp4_weight", fp4_weight)
        self.register_buffer("weight_scaling_factor", weight_scaling_factor)
        self.register_buffer("weight_global_scale", weight_global_scale)
        self.register_buffer("ab_strides", ab_strides)
        self.register_buffer("c_strides", c_strides)

    # TODO
    def forward(self, hidden_states: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # blockscale_offsets, problem_sizes = compute_blockscale_offsets_and_problem_sizes(offsets, self.ab_strides, self.c_strides)
        # return grouped_mm_a16_nvfp4weight(hidden_states, self.fp4_weight, self.weight_scaling_factor, self.weight_global_scale, self.bias, self.ab_strides, self.c_strides, blockscale_offsets, problem_sizes)
        raise NotImplementedError()

    @classmethod
    def from_grouped_linear(grouped_linear: GroupedLinear) -> NVFP4InferenceGroupedLinear:
        weight = grouped_linear.weight
        (
            fp4_weight,
            weight_scaling_factor,
            weight_global_scale,
            ab_strides,
            c_strides,
        ) = quantize_grouped_linear_weight_to_nvfp4(weight)
        return NVFP4InferenceGroupedLinear(
            fp4_weight,
            weight_scaling_factor,
            weight_global_scale,
            ab_strides=ab_strides,
            c_strides=c_strides,
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
        # This is defined in `thunder.tests.test_networks`
        from thunder.tests.test_networks import Config

        # 1. Create a config for the Llama4MoE model from the transformers config
        config = Config(
            hidden_size=moe.config.hidden_size,
            intermediate_size=moe.config.intermediate_size,
            num_routed_experts=moe.config.num_local_experts,
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
        # and permute the weight dimensions to match GroupedLinear
        # HF format: (groups, in_features, out_features)
        # Our format: (groups, out_features, in_features)

        # Permute from (num_experts, hidden_size, 2 * intermediate_size) to
        # (num_experts, 2 * intermediate_size, hidden_size)
        gate_up_proj_permuted = moe.experts.gate_up_proj.permute(0, 2, 1)

        # Split into gate and up projections
        gate_proj_w, up_proj_w = gate_up_proj_permuted.chunk(2, dim=1)

        new_moe.routed_experts.gate_proj.weight.data.copy_(gate_proj_w)
        new_moe.routed_experts.up_proj.weight.data.copy_(up_proj_w)

        # Permute down_proj from (num_experts, intermediate_size, hidden_size) to
        # (num_experts, hidden_size, intermediate_size)
        down_proj_permuted = moe.experts.down_proj.permute(0, 2, 1)
        new_moe.routed_experts.down_proj.weight.data.copy_(down_proj_permuted)

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
        offsets = torch.cumsum(tokens_per_expert, 0, dtype=torch.int32)  # [n]
        outs_sorted_by_expert_id = self.routed_experts(tokens_sorted_by_expert_id, offsets)  # [s, h]

        token_ids_sorted_by_expert_inverse_id = torch.argsort(token_ids_sorted_by_expert_id)
        outs_sorted_by_token_id = outs_sorted_by_expert_id[token_ids_sorted_by_expert_inverse_id]

        return outs_sorted_by_token_id, router_logits

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outs_sorted_by_token_id, router_logits = self.run_routed_experts(hidden_states)
        return self.shared_experts(hidden_states) + outs_sorted_by_token_id, router_logits
