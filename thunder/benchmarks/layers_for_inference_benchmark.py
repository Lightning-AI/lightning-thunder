from __future__ import annotations
from typing import TYPE_CHECKING

from looseversion import LooseVersion
import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


@torch.inference_mode()
def quantize_weight_to_nvfp4(
    weight: torch.Tensor | nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize weight to nvfp4, returning (packed) e2m1 weight, e4m3 scale factor, fp32 global scale."""
    # global_scale = weight.abs().amax()
    # ...
    # return weight, weight, global_scale
    raise NotImplementedError()


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
        pass

    @classmethod
    def from_linear(linear: nn.Linear) -> NVFP4InferenceLinear:
        weight = linear.weight
        bias = linear.bias
        out_features, in_features = weight.size()
        fp4_weight, weight_scaling_factor, weight_global_scale = quantize_weight_to_nvfp4(weight)
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
