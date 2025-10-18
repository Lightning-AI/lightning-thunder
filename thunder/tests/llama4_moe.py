import math
from dataclasses import dataclass
from looseversion import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    name: str
    hidden_size: int
    intermediate_size: int
    num_routed_experts: int
    num_shared_experts: int
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


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
def grouped_mm(a: torch.Tensor, b: torch.Tensor, tokens_per_expert_or_offsets: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        offsets = tokens_per_expert_or_offsets  # [n]
        return _grouped_mm(a, b, offsets)

    group_outs = []
    tokens_per_expert = tokens_per_expert_or_offsets
    for idx, group_a in enumerate(a.split(tokens_per_expert)):
        group_outs.append(group_a @ b[idx])
    return torch.cat(group_outs)


class GroupedLinear(nn.Module):
    def __init__(self, groups: int, in_features: int, out_features: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, out_features, in_features, dtype=dtype, device=device))
        # Initialize the weight in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor, tokens_per_expert_or_offsets: torch.Tensor) -> torch.Tensor:
        return grouped_mm(hidden_states, self.weight.transpose(-1, -2), tokens_per_expert_or_offsets)


class GroupedSwiGLU(nn.Module):
    def __init__(self, groups: int, hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: str):
        super().__init__()
        self.gate_proj = GroupedLinear(groups, hidden_size, intermediate_size, dtype, device)
        self.up_proj = GroupedLinear(groups, hidden_size, intermediate_size, dtype, device)
        self.down_proj = GroupedLinear(groups, intermediate_size, hidden_size, dtype, device)

    def forward(self, hidden_states: torch.Tensor, tokens_per_expert_or_offsets: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states, tokens_per_expert_or_offsets))
            * self.up_proj(hidden_states, tokens_per_expert_or_offsets),
            tokens_per_expert_or_offsets,
        )


class Llama4MoE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.hidden_size, config.num_routed_experts, bias=False, dtype=config.dtype, device=config.device
        )
        self.shared_experts = SwiGLU(
            config.hidden_size, config.intermediate_size * config.num_shared_experts, config.dtype, config.device
        )
        self.routed_experts = GroupedSwiGLU(
            config.num_routed_experts, config.hidden_size, config.intermediate_size, config.dtype, config.device
        )

    def run_routed_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

        if not torch.compiler.is_compiling():
            tokens_per_expert_or_offsets = tokens_per_expert.tolist()
        else:
            tokens_per_expert_or_offsets = torch.cumsum(tokens_per_expert, 0, dtype=torch.int32)  # [n]

        outs_sorted_by_expert_id = self.routed_experts(
            tokens_sorted_by_expert_id, tokens_per_expert_or_offsets
        )  # [s, h]

        token_ids_sorted_by_expert_inverse_id = torch.argsort(token_ids_sorted_by_expert_id)
        outs_sorted_by_token_id = outs_sorted_by_expert_id[token_ids_sorted_by_expert_inverse_id]

        return outs_sorted_by_token_id

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states) + self.run_routed_experts(hidden_states)
