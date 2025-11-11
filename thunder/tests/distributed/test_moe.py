from functools import partial
import copy
import os
import unittest

import torch
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
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
)
from torch.distributed.device_mesh import init_device_mesh
import torch.nn as nn

import thunder.tests.llama4_moe as llama4_moe
from thunder.tests.distributed.helper import DistributedParallelTestCase
from thunder.dynamo import thunderfx


# Referred from torchtitan: https://github.com/pytorch/torchtitan/blob/827255bb/torchtitan/experiments/llama4/infra/expert_parallel.py#L25
class GroupedLinearColwiseParallel(ParallelStyle):
    def __init__(
        self,
        *,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh):
        prepared_inputs = []
        INPUT_LAYOUTS = (Replicate(), Replicate())
        assert len(INPUT_LAYOUTS) == len(inputs), "input_layouts and inputs have different lengths"
        # annotate module input placements/sharding with input_layouts
        for inp, input_layout in zip(inputs, INPUT_LAYOUTS):
            assert isinstance(inp, (torch.Tensor, list)), f"inp is not a torch.Tensor or list: {type(inp)}"
            if isinstance(inp, torch.Tensor):
                assert not isinstance(inp, DTensor), "inp is already a DTensor"
                inp = DTensor.from_local(inp, device_mesh, (input_layout,), run_check=False)
            prepared_inputs.append(inp)
        return tuple(prepared_inputs)

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "weight", nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(1)]))
        )  # Column-wise sharding

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        OUTPUT_LAYOUT = Shard(1)
        if outputs.placements != (OUTPUT_LAYOUT,):
            outputs = outputs.redistribute(placements=(OUTPUT_LAYOUT,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            self._prepare_input_fn,
            partial(self._prepare_output_fn, self.use_local_output),
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
        module.register_parameter("weight", nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(-1)])))

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


def parallelize_moe_model(model: llama4_moe.Llama4MoE, device_mesh: torch.distributed.DeviceMesh):
    """Apply TensorParallel to the MoE model"""

    # Define the parallelization plan as a dictionary
    parallelize_plan = {
        # Shared experts - SwiGLU components
        "shared_experts.gate_proj": ColwiseParallel(use_local_output=False, output_layouts=Shard(2)),
        "shared_experts.up_proj": ColwiseParallel(use_local_output=False, output_layouts=Shard(2)),
        "shared_experts.down_proj": RowwiseParallel(),
        # Routed experts
        "routed_experts.gate_proj": GroupedLinearColwiseParallel(use_local_output=False),
        "routed_experts.up_proj": GroupedLinearColwiseParallel(use_local_output=False),
        "routed_experts.down_proj": GroupedLinearRowwiseParallel(),
    }

    # Parallelize the model
    parallelized_model = parallelize_module(
        model,
        device_mesh,
        parallelize_plan,
    )
    return parallelized_model


class TestLlama4MoEDistributed(DistributedParallelTestCase):
    def test_llama4_moe_distributed(self):
        if "multidevice" in os.environ.get("NVFUSER_DISABLE", ""):
            raise unittest.SkipTest("test_llama4_moe_distributed: nvfuser multidevice is disabled")

        # Get world size
        world_size = self.world_size
        device = f"cuda:{self.rank}"

        # Initialize device mesh for TensorParallel
        device_mesh = init_device_mesh("cuda", (world_size,))

        config = llama4_moe.Config(
            name="small", hidden_size=256, intermediate_size=512, num_routed_experts=8, num_shared_experts=1
        )

        # Create model with distributed tensors
        model = llama4_moe.Llama4MoE(config)

        # Create a copy of the original model as
        # `parallelize_module` will modify the model in place.
        org_model = copy.deepcopy(model)

        # Apply TensorParallel
        parallelized_model = parallelize_moe_model(model, device_mesh)

        # Without this, `thunderfx` falls back to `inductor` for `_grouped_mm`
        # as it doesn't have a grad-rule for the same.
        parallelized_model.requires_grad_(False)

        batch_size, seq_len = 1, 2048
        inp = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16, device=device)

        # Run forward pass
        actual = parallelized_model(inp)
        expected = org_model(inp)
        assert any(isinstance(p, DTensor) for p in parallelized_model.parameters())
        assert all(not isinstance(p, DTensor) for p in org_model.parameters())

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

        tmodel = thunderfx(parallelized_model, nv_enable_linear=True, nv_enable_scatter=True)
        actual = tmodel(inp)

        # Verify that there was one FXGraph.
        assert len(tmodel._backend.subgraph_infos) == 1
        # Verify that the graph was not split.
        assert len(tmodel._backend.subgraph_infos[0].split_reasons) == 0

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
