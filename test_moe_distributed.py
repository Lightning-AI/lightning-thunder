import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    ParallelStyle,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)

from thunder.dynamo import thunderfx

# Initialize process group
init_process_group(backend="nccl")

# Get local rank and world size
local_rank = int(torch.distributed.get_rank())
world_size = torch.distributed.get_world_size()

# Set device
device = f"cuda:{local_rank}"
torch.cuda.set_device(device)


# Sizes used in Llama 4 Maverick
@dataclass
class Config:
    hidden_size: int = 5120
    intermediate_size: int = 8192
    num_routed_experts: int = 128
    num_shared_experts: int = 1


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _group_sizes_from_offsets(offsets: torch.Tensor) -> list[int]:
    group_sizes = []
    prev = 0
    for offset in offsets:
        group_sizes.append(offset - prev)
        prev = offset
    return group_sizes


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
    def __init__(self, groups: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, in_features, out_features))
        # Initialize the weight in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor, tokens_per_expert_or_offsets: torch.Tensor) -> torch.Tensor:
        return grouped_mm(hidden_states, self.weight, tokens_per_expert_or_offsets)


class GroupedSwiGLU(nn.Module):
    def __init__(self, groups: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = GroupedLinear(groups, hidden_size, intermediate_size)
        self.up_proj = GroupedLinear(groups, hidden_size, intermediate_size)
        self.down_proj = GroupedLinear(groups, intermediate_size, hidden_size)

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
        self.gate = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        self.shared_experts = SwiGLU(config.hidden_size, config.intermediate_size * config.num_shared_experts)
        self.routed_experts = GroupedSwiGLU(config.num_routed_experts, config.hidden_size, config.intermediate_size)

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


@contextmanager
def default_tensor_type(dtype=torch.float32, device="cpu"):
    # Save
    prev_dtype = torch.get_default_dtype()
    prev_device = torch.get_default_device()

    # Set
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    yield

    # Restore
    torch.set_default_dtype(prev_dtype)
    torch.set_default_device(prev_device)


# Referred from torchtitan: https://github.com/pytorch/torchtitan/blob/827255bb484d0f0a97fe5bec22b70f8b4750f685/torchtitan/experiments/llama4/infra/expert_parallel.py#L25
class GroupedLinearColwiseParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: tuple[Placement | None] | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts or (Replicate(), Replicate())
        self.output_layout = output_layouts or Shard(-1)
        self.desired_input_layouts = (Replicate(), Replicate())
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        prepared_inputs = []
        # annotate module input placements/sharding with input_layouts
        for inp, input_layout, desired_input_layout in zip(inputs, input_layouts, desired_input_layouts):
            if isinstance(inp, torch.Tensor):
                if not isinstance(inp, DTensor):
                    inp = DTensor.from_local(inp, device_mesh, (input_layout,), run_check=False)
                if input_layout != desired_input_layout:
                    inp = inp.redistribute(placements=(desired_input_layout,), async_op=True)
            prepared_inputs.append(inp)
        return tuple(prepared_inputs)

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "weight", nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(2)]))
        )  # Column-wise sharding

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


class GroupedLinearRowwiseParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: tuple[Placement | None] | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts or (Shard(-1), Replicate())
        self.output_layout = output_layouts or Replicate()
        self.desired_input_layouts = (Shard(-1), Replicate())
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        prepared_inputs = []
        # annotate module input placements/sharding with input_layouts
        for inp, input_layout, desired_input_layout in zip(inputs, input_layouts, desired_input_layouts):
            if isinstance(inp, torch.Tensor):
                if not isinstance(inp, DTensor):
                    inp = DTensor.from_local(inp, device_mesh, (input_layout,), run_check=False)
                if input_layout != desired_input_layout:
                    inp = inp.redistribute(placements=(desired_input_layout,), async_op=True)
            prepared_inputs.append(inp)
        return tuple(prepared_inputs)

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter("weight", nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(1)])))

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


def parallelize_moe_model(model: Llama4MoE, device_mesh: torch.distributed.DeviceMesh):
    """Apply TensorParallel to the MoE model"""

    # Define the parallelization plan as a dictionary
    parallelize_plan = {
        # "gate": ColwiseParallel(use_local_output=True),
        # Shared experts - SwiGLU components
        "shared_experts.gate_proj": ColwiseParallel(use_local_output=False, output_layouts=Shard(2)),
        "shared_experts.up_proj": ColwiseParallel(use_local_output=False, output_layouts=Shard(2)),
        "shared_experts.down_proj": RowwiseParallel(),
        # Routed experts
        "routed_experts.gate_proj": GroupedLinearColwiseParallel(use_local_output=False, output_layouts=Shard(1)),
        "routed_experts.up_proj": GroupedLinearColwiseParallel(use_local_output=False, output_layouts=Shard(1)),
        "routed_experts.down_proj": GroupedLinearRowwiseParallel(),
    }

    # Parallelize the model
    parallelized_model = parallelize_module(
        model,
        device_mesh,
        parallelize_plan,
    )
    return parallelized_model


def test_llama4_moe_distributed():
    """Test the distributed MoE model with TensorParallel"""
    # Initialize device mesh for TensorParallel
    device_mesh = init_device_mesh("cuda", (world_size,))

    config = Config()

    # Create model with distributed tensors
    with default_tensor_type(dtype=torch.bfloat16, device=device):
        model = Llama4MoE(config)

    # Apply TensorParallel
    parallelized_model = parallelize_moe_model(model, device_mesh)

    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.max_memory_allocated())
    # Without this, `thunderfx` falls back to `inductor` for `_grouped_mm`
    # as it doesn't have a grad-rule for the same.
    parallelized_model.requires_grad_(False)

    batch_size, seq_len = 1, 2048
    inp = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16, device=device)

    # Run forward pass
    parallelized_model(inp)

    print(torch.cuda.max_memory_allocated())

    print(f"Rank {local_rank}: Distributed MoE test passed!")


if __name__ == "__main__":
    # This test should be run with torchrun or similar distributed launcher
    # Example: torchrun --local-ranks-filter=0,1 --nproc_per_node=2 test_moe_distributed.py
    test_llama4_moe_distributed()

    # Cleanup
    destroy_process_group()
