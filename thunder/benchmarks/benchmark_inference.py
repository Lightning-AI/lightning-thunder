"""
Thunder Inference Benchmark following SemiAnalysis Methodology

This benchmark implements the methodology from the SemiAnalysis article:
"AMD vs NVIDIA Inference Benchmark: Who Wins? - Performance & Cost Per Million Tokens"
https://semianalysis.com/2025/05/23/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens

Models:
- Llama 3.1 8B - Lower memory footprint for local experimentation
- Llama 3.1 70B
- Llama 3.1 405B
- DeepSeekV3 670B
- Llama 4 Scout 17B
- Llama 4 Maverick

Key metrics:
- Throughput (tokens/second)
- Latency (ms/token)
- Time to First Token (TTFT)
- Time Between Output Tokens (TBOT)
- Cost per Million Tokens
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np

import torch
import torch.distributed

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel, ColwiseParallel
from torch.distributed.tensor import DTensor

# Import model configurations
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import HybridChunkedCache
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe, Llama4TextExperts

from lightning_utilities.core.imports import package_available


Llama4TextMoe_forward_original = Llama4TextMoe.forward
Llama4TextExperts_forward_original = Llama4TextExperts.forward


def topk1(values, topk, dim):
    assert topk == 1, "Only topk=1 is supported"
    return torch.max(values, dim=dim, keepdim=True)


def static_bincount(x, length):
    return torch.index_add(torch.zeros(length, device=x.device, dtype=x.dtype), 0, x, torch.ones_like(x))


def inverse_permutation(x):
    return torch.scatter(torch.zeros_like(x), 0, x, torch.arange(x.numel(), device=x.device))


def compute_offsets_and_permutations(router_indices, num_experts):
    flat_router_indices = router_indices.flatten()
    expert_permutation = torch.argsort(flat_router_indices)
    group_size_per_expert = static_bincount(flat_router_indices, num_experts)
    zero = torch.zeros(1, device=group_size_per_expert.device, dtype=group_size_per_expert.dtype)
    # torch.compile requires int32 for offsets
    # https://github.com/pytorch/pytorch/blob/ed6ae20cf0e31d49d54177251293267205e24021/torch/_meta_registrations.py#L7657
    expert_offsets = torch.cat([zero, group_size_per_expert.cumsum(0)]).to(torch.int32)
    inv_expert_permutation = inverse_permutation(expert_permutation)
    return expert_offsets, expert_permutation, inv_expert_permutation


# Grouped MM in PyTorch is supported only for compute capability == 9.0, 10.0
def grouped_mm(a, b, offsets):
    try:
        return torch._grouped_mm(a, b, offsets)
    except RuntimeError:
        # try:
        #     return (torch.nested.nested_tensor_from_jagged(a, offsets) @ b).values()
        # # RuntimeError: numel needs to be smaller than int32_t max; otherwise, please use packed_accessor64
        # except RuntimeError:
        c_list = []
        offsets_cpu = offsets.cpu()
        for i in range(offsets_cpu.size(0) - 1):
            a_i = a[offsets_cpu[i] : offsets_cpu[i + 1]]
            b_i = b[i]
            c_i = a_i @ b_i
            c_list.append(c_i)
        return torch.cat(c_list, dim=0)


# Ref:
# https://github.com/vllm-project/vllm/blob/3ee56e26be4cfddc17f7d2e5f38f15ab74ede1c2/vllm/model_executor/models/llama4.py#L48
def custom_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert renormalize is False, "Renormalization is not supported"
    router_scores, router_indices = topk1(gating_output, topk, dim=-1)
    router_scores = torch.sigmoid(router_scores.to(torch.float32)).to(hidden_states.dtype)
    return (router_scores, router_indices.to(torch.int32))


def experts(
    hidden_states_2d: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    hidden_states_2d = hidden_states_2d * topk_weights.reshape(-1, 1)
    expert_offsets, expert_permutation, inv_expert_permutation = compute_offsets_and_permutations(topk_ids, num_experts)
    sorted_per_expert_hidden_states = hidden_states_2d[expert_permutation]
    gate_up = grouped_mm(sorted_per_expert_hidden_states, w1, expert_offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    next_states = grouped_mm(up * torch.nn.functional.silu(gate), w2, expert_offsets)
    return next_states[inv_expert_permutation]


def Llama4TextMoe_forward(self: Llama4TextMoe, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
    shared_out = self.shared_expert(hidden_states)
    router_logits = self.router(hidden_states)

    router_logits_2d = router_logits.view(-1, router_logits.size(-1))
    hidden_states_2d = hidden_states.view(-1, self.hidden_dim)

    topk_weights, topk_ids = custom_routing_function(hidden_states_2d, router_logits_2d, self.top_k, renormalize=False)
    w1 = self.experts.gate_up_proj
    w2 = self.experts.down_proj
    routed_out = experts(
        hidden_states_2d,
        w1,
        w2,
        topk_weights,
        topk_ids,
        num_experts=self.num_experts,
    )
    routed_out = routed_out.view(hidden_states.shape)
    out = routed_out + shared_out
    return out, None


Llama4TextMoe.forward = Llama4TextMoe_forward


# When `vllm` and its fused_experts is available, use fused_experts as llama4 moe forward.
# Note that due to the expected shape difference between transformers and vllm,
# `fused_experts` requires `Tensor.contiguous`.
# NOTE(mkozuki) test env I used:
#     - dlcluster node: gb-nvl-081-compute04 (GB200)
#     - container: `gitlab-master.nvidia.com:5005/dl/dgx/vllm:25.06-py3-devel-arm64`
TEXT_MOE_REPLACED = False
if package_available("vllm"):
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    except ImportError:
        pass
    else:
        # ref:
        #   - https://github.com/vllm-project/vllm/blob/65397e40f58ff5657d9e8bbd860ed9d3fdf734a0/tests/kernels/moe/test_moe.py#L125
        #   - https://gitlab-master.nvidia.com/dl/vllm/vllm/-/blob/556acedfbda60ba40bca19bd333e6dbff0f90680/tests/kernels/moe/test_moe.py#L50
        def Llama4TextMoe_forward(self: Llama4TextMoe, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
            router_logits = self.router(hidden_states)

            router_logits_2d = router_logits.view(-1, router_logits.size(-1))
            hidden_states_2d = hidden_states.view(-1, self.hidden_dim)

            topk_weights, topk_ids = custom_routing_function(
                hidden_states_2d, router_logits_2d, self.top_k, renormalize=False
            )
            w1 = self.experts.gate_up_proj
            w2 = self.experts.down_proj
            return (
                fused_experts(
                    hidden_states_2d,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    inplace=True,
                    apply_router_weight_on_input=True,  # https://github.com/vllm-project/vllm/blob/3ee56e26be4cfddc17f7d2e5f38f15ab74ede1c2/vllm/model_executor/models/llama4.py#L80
                ),
                None,
            )

        Llama4TextMoe.forward = Llama4TextMoe_forward
        TEXT_MOE_REPLACED = True


import thunder

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")
os.environ["RANK"] = str(RANK)
os.environ["LOCAL_RANK"] = str(LOCAL_RANK)
os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
os.environ["MASTER_ADDR"] = MASTER_ADDR
os.environ["MASTER_PORT"] = MASTER_PORT
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

mesh = init_device_mesh("cuda", (WORLD_SIZE,), mesh_dim_names=("tp",))
device = torch.device("cuda", LOCAL_RANK)
torch.cuda.set_device(device)


@contextmanager
def timer():
    torch.cuda.synchronize()
    t1 = t2 = perf_counter()
    yield lambda: (t2 - t1) * 1000  # Convert to ms
    torch.cuda.synchronize()
    t2 = perf_counter()


# Standard benchmark scenarios following the three-scenario methodology
BENCHMARK_SCENARIOS = {
    "summarization": {
        "name": "Summarization (Prefill-Heavy)",
        "input_length": 4000,
        "output_length": 1000,
        "description": "4,000 input → 1,000 output tokens (80% prefill, 20% decode)",
        "workload_balance": "80% prefill, 20% decode computational cost",
        "hardware_focus": "Compute optimization provides maximum impact",
    },
    "chat": {
        "name": "Chat (Balanced)",
        "input_length": 1000,
        "output_length": 1000,
        "description": "1,000 input → 1,000 output tokens (50% prefill, 50% decode)",
        "workload_balance": "50% prefill, 50% decode computational cost",
        "hardware_focus": "Mixed optimization requirements",
    },
    "reasoning": {
        "name": "Reasoning (Decode-Heavy)",
        "input_length": 1000,
        "output_length": 4000,
        "description": "1,000 input → 4,000 output tokens (20% prefill, 80% decode)",
        "workload_balance": "20% prefill, 80% decode computational cost",
        "hardware_focus": "Memory bandwidth optimization dominates",
    },
}


@dataclass
class InferenceBenchmarkConfig:
    """Configuration for inference benchmarking following SemiAnalysis methodology"""

    model_name: str
    # Expected GPU memory requirements (FP16):
    # - 8B: ~16GB (suitable for local experimentation on consumer GPUs)
    # - 70B: ~140GB (requires multi-GPU setup or high-end datacenter GPUs)
    # - 405B: ~810GB (requires large multi-GPU clusters)
    # - 670B: ~1340GB (requires very large multi-GPU clusters)
    batch_size: int = 1
    input_length: int = 1024
    output_length: int = 1024
    num_layers: int | None = None
    num_iterations: int = 10
    warmup_iterations: int = 2
    device: str = "cuda"
    mode: str = "thunder"  # "thunder", "eager", "inductor"
    measure_ttft: bool = True
    measure_tbot: bool = True
    scenario: str | None = None  # Standard scenario name if using predefined configurations
    dtensor_single_gpu: bool = False
    load_nvfp4: bool = False  # Enable NVFP4 quantization

    # Cost calculation parameters (per GPU hour) # optional
    h100_cost_per_hour: float = 1.58
    h200_cost_per_hour: float = 1.63
    b200_cost_per_hour: float = 2.23

    # Memory bandwidth and compute specs
    # TODO check correctness of numbers (generated by AI)
    gpu_specs: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "H100": {"memory_bandwidth_gb": 3350, "fp16_tflops": 1979, "fp8_tflops": 3958},
            "H200": {"memory_bandwidth_gb": 4800, "fp16_tflops": 1979, "fp8_tflops": 3958},
            "B200": {"memory_bandwidth_gb": 8000, "fp16_tflops": 2529, "fp8_tflops": 5058},
        }
    )


@dataclass
class InferenceMetrics:
    """Metrics collected during inference benchmarking"""

    throughput_tokens_per_sec: float = 0.0
    latency_ms_per_token: float = 0.0
    time_to_first_token_ms: float = 0.0
    time_between_output_tokens_ms: float = 0.0
    total_time_ms: float = 0.0
    memory_used_gb: float = 0.0
    peak_memory_gb: float = 0.0
    cost_per_million_tokens: float = 0.0

    # Separate prefill and decode metrics
    prefill_throughput_tokens_per_sec: float = 0.0
    decode_throughput_tokens_per_sec: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0

    # Per-iteration metrics for variance analysis
    iteration_times: list[float] = field(default_factory=list)
    ttft_times: list[float] = field(default_factory=list)
    prefill_times: list[float] = field(default_factory=list)
    decode_times: list[float] = field(default_factory=list)


def ceil_div(a, b):
    return (a + b - 1) // b


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


def _bfloat16_to_float4_e2m1fn_x2(x):
    from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked

    FP4_EBITS, FP4_MBITS = 2, 1
    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x.to(torch.float32), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


class SemiAnalysisInferenceBenchmark:
    """Main benchmark class following SemiAnalysis methodology"""

    def __init__(self, config: InferenceBenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = InferenceMetrics()

        # Load model
        self.model = self._load_model()

        tp_plan = {
            "*.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=True),
            "*.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=True),
            "*.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=True),
            "*.layers.*.self_attn.o_proj": RowwiseParallel(use_local_output=True),
            "*.layers.*.feed_forward.gate_proj": ColwiseParallel(use_local_output=False),
            "*.layers.*.feed_forward.up_proj": ColwiseParallel(use_local_output=False),
            "*.layers.*.feed_forward.down_proj": RowwiseParallel(use_local_output=True),
            # FIXME: Getting AttributeError: 'AsyncCollectiveTensor' object has no attribute 'placements'
            # [rank0]: File "torch/distributed/tensor/parallel/style.py", line 276, in _prepare_output_fn
            # [rank0]:     if outputs.placements != output_layouts:
            # "*.layers.*.feed_forward.shared_expert.gate_proj": ColwiseParallel(use_local_output=False),
            # "*.layers.*.feed_forward.shared_expert.up_proj": ColwiseParallel(use_local_output=False),
            # "*.layers.*.feed_forward.shared_expert.down_proj": RowwiseParallel(use_local_output=True),
        }

        if self.config.dtensor_single_gpu or WORLD_SIZE > 1:
            self.model = parallelize_module(self.model, mesh, tp_plan)
            # assert isinstance(self.model.model.layers[0].self_attn.o_proj.weight, DTensor)
            # assert isinstance(self.model.model.layers[0].feed_forward.down_proj.weight, DTensor)

            # Required as that doesn't understand inference mode
            for p in self.model.parameters():
                p.requires_grad_(False)

        # Compile model
        self.model = self._compile_model(self.model)

    def _compile_model(self, model):
        match self.config.mode:
            case "eager":
                return model
            case "inductor":
                return torch.compile(model, mode="reduce-overhead")
            case "thunder":
                from thunder.dynamo import thunderfx

                # Set `nv_enable_linear` to True
                # once workaround is added for https://github.com/NVIDIA/Fuser/issues/4507
                return thunderfx(model, nv_enable_linear=False)
            case "thunderjit":
                # Set `nv_enable_linear` to True
                # once workaround is added for https://github.com/NVIDIA/Fuser/issues/4507
                return thunder.jit(model, nv_enable_linear=False)
            case _:
                raise ValueError(f"Unknown mode: {self.config.mode}")

    def _load_model(self) -> torch.nn.Module:
        """Load the model based on configuration"""
        model_id = self.config.model_name

        # Load model configuration (without quantization first)
        config = AutoConfig.from_pretrained(model_id)

        if hasattr(config, "text_config"):
            config = config.text_config

        # Set the number of layers
        if self.config.num_layers:
            config.num_hidden_layers = self.config.num_layers

        self.hf_config = config

        # Apply FP4 quantization to weights if requested
        if self.config.load_nvfp4:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
            self._quantize_to_nvfp4_and_materialize(model)
        else:
            # Create model on device
            with torch.device(self.device):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

        # NOTE(mkozuki): Transpose `Llama4TextExperts.gate_up_proj` for the better compatibility with vLLM's `fused_moe`.
        # In huggingface/transformers, w1 (= gate_up_proj) has the shape of (e, k, 2n) and w2 (= down_proj) has the shape of (e, n, k).
        # vLLM however requires w1 to have the shape of (e, 2n, k), and w2, (e, n, k).
        # Shapes I get from the command of `python thunder/benchmarks/benchmark_inference.py --model-name meta-llama/Llama-4-Maverick-17B-128E --mode thunder --input-length 32 --output-length 32 --batch-size 1 --num-iterations 20 --num-layers 4` are as follows:
        # The shapes are w/o transpose on gate_up_proj.
        # router_logits_2d.shape = torch.Size([32, 128]), hidden_states_2d.shape = torch.Size([32, 5120]), w1.shape = torch.Size([128, 5120, 16384]), w2.shape = torch.Size([128, 8192, 5120])
        if TEXT_MOE_REPLACED:
            with torch.inference_mode():
                for module in filter(lambda module: isinstance(module, Llama4TextExperts), model.modules()):
                    module.gate_up_proj = torch.nn.Parameter(module.gate_up_proj.transpose(1, 2).contiguous())

        return model

    def _quantize_to_nvfp4_and_materialize(self, model: torch.nn.Module, device: torch.device | str = "cuda"):
        modules_to_quantize = {
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
            "qkv_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
            "router",
        }

        def materialize(module: torch.nn.Module):
            module.to_empty(device=device, recurse=False)
            if len(tuple(module.parameters())) > 1:
                if not hasattr(module, "reset_parameters"):
                    raise TypeError(
                        f"Materialization requires that the `{type(module).__name__}.reset_parameters` method is implemented."
                        " This method is used to initialize any children parameters or buffers in this module."
                    )
                module.reset_parameters()

        def quantize_to_nvfp4(model: torch.nn.Module):
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and any(name in module_name for name in modules_to_quantize):
                    # Initialize weight to FP4
                    with torch.no_grad():
                        materialize(module)
                        # Convert to FP4 format
                        weight_fp4 = _bfloat16_to_float4_e2m1fn_x2(module.weight)

                        del module.weight
                        # Create a new parameter with FP4 data
                        # Note: PyTorch Linear layers expect float weights, so this is experimental
                        module.weight = torch.nn.Parameter(weight_fp4, requires_grad=False)

                        # remove bf16 from cache
                        torch.cuda.empty_cache()

                elif len(list(module.children())) < 1:
                    materialize(module)

        quantize_to_nvfp4(model)

    def generate_batch(self) -> tuple[torch.Tensor, HybridChunkedCache]:
        """Generate a batch of input tokens"""
        batch_size = self.config.batch_size
        input_length = self.config.input_length

        # Determine vocabulary size based on model
        if hasattr(self.model, "vocab_size"):
            vocab_size = self.model.vocab_size
        elif hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
            vocab_size = self.model.config.vocab_size
        else:
            # Default vocabulary size for older models
            vocab_size = 32000

        # Random input tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, input_length), device=self.device)
        past_key_values = HybridChunkedCache(
            self.hf_config, input_ids.shape[0], input_ids.shape[1] + self.config.output_length
        )
        for layer_idx in range(self.hf_config.num_hidden_layers):
            # key_states.shape[1] is used to retrieve the number of key value heads, all other dimensions can be 1 and ignored
            # https://github.com/huggingface/transformers/blob/9300728665aaeb0ebf4db99f9d9fbce916b4a183/src/transformers/cache_utils.py#L1822
            past_key_values.initialise_cache_layer(
                layer_idx, torch.empty(1, self.hf_config.num_key_value_heads // WORLD_SIZE, 1, 1, device=self.device)
            )

        return input_ids, past_key_values

    def get_next_token(self, input_ids: torch.Tensor, past_key_values: HybridChunkedCache) -> torch.Tensor:
        outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits  # [B, seq_len, vocab_size]
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        return next_token

    def prefill(self, input_ids: torch.Tensor, past_key_values: HybridChunkedCache) -> torch.Tensor:
        """
        Prefill phase: Process the entire input prompt at once.
        Returns the next token.

        Similar to: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L68-L82
        """
        return self.get_next_token(input_ids, past_key_values)

    def decode_one_token(self, input_ids: torch.Tensor, past_key_values: HybridChunkedCache) -> torch.Tensor:
        """
        Decode phase: Generate a single token given the current sequence.
        Returns the next token.
        """
        # input_pos: [B, 1] One token at the time
        assert input_ids.shape[-1] == 1, f"Expected shape (B, 1), but found {input_ids.shape}"
        return self.get_next_token(input_ids, past_key_values)

    @torch.inference_mode()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, past_key_values: HybridChunkedCache
    ) -> dict[str, Any]:
        """
        Generate tokens using separate prefill and decode phases.
        Returns detailed metrics for both phases.
        """
        # Prefill phase - process the entire prompt
        with timer() as prefill_timer:
            first_token = self.prefill(input_ids, past_key_values)
        prefill_time = prefill_timer()
        generated_tokens = [first_token]

        # Decode phase - generate remaining tokens one by one
        next_token = first_token
        with timer() as decode_timer:
            for _ in range(max_new_tokens - 1):
                next_token = self.decode_one_token(next_token, past_key_values)
                generated_tokens.append(next_token)

        total_decode_time = decode_timer()

        return {
            "prefill_time_ms": prefill_time,
            "decode_time_ms": total_decode_time,
            "generated_tokens": generated_tokens,
            "total_tokens": max_new_tokens,
        }

    def measure_inference_step(
        self, input_ids: torch.Tensor, past_key_values: HybridChunkedCache, max_new_tokens: int
    ) -> dict[str, float]:
        """Measure a single inference step with detailed timing using separate prefill/decode"""
        with timer() as total_timer:
            # Generate tokens with separate prefill/decode tracking
            generation_result = self.generate(input_ids, max_new_tokens, past_key_values)
        total_time = total_timer()

        # Extract metrics
        ttft = generation_result["prefill_time_ms"]  # Time to first token is the prefill time
        total_decode_time = generation_result["decode_time_ms"]
        avg_tbot = total_decode_time / (max_new_tokens - 1) if max_new_tokens > 1 else 0

        # Calculate throughput
        total_tokens = self.config.output_length * self.config.batch_size
        throughput = (total_tokens / total_time) * 1000  # tokens/second

        # Calculate separate prefill and decode throughput
        prefill_tokens = self.config.input_length * self.config.batch_size
        prefill_throughput = (prefill_tokens / generation_result["prefill_time_ms"]) * 1000

        decode_tokens = (self.config.output_length - 1) * self.config.batch_size
        decode_throughput = (decode_tokens / total_decode_time) * 1000 if total_decode_time > 0 else 0

        return {
            "ttft": ttft,
            "avg_tbot": avg_tbot,
            "total_time": total_time,
            "throughput": throughput,
            "prefill_throughput": prefill_throughput,
            "decode_throughput": decode_throughput,
            "prefill_time": generation_result["prefill_time_ms"],
            "total_decode_time": total_decode_time,
        }

    def run_benchmark(self) -> InferenceMetrics:
        """Run the full benchmark and collect metrics"""
        from tqdm import tqdm

        print(f"Running inference benchmark for {self.config.model_name}")

        print(f"Batch size: {self.config.batch_size}")
        print(f"Input length: {self.config.input_length}")
        print(f"Output length: {self.config.output_length}")
        print(f"Device: {self.device}")
        print(f"Mode: {self.config.mode}")

        # Warmup iterations
        print(f"\nWarming up with {self.config.warmup_iterations} iterations...")
        input_ids, past_key_values = self.generate_batch()

        for _ in tqdm(range(self.config.warmup_iterations)):
            past_key_values.reset()
            _ = self.measure_inference_step(input_ids, past_key_values, max_new_tokens=1)

        # Benchmark iterations
        print(f"\nRunning {self.config.num_iterations} benchmark iterations...")
        all_metrics = []

        for _ in tqdm(range(self.config.num_iterations)):
            past_key_values.reset()
            iter_metrics = self.measure_inference_step(input_ids, past_key_values, self.config.output_length)
            all_metrics.append(iter_metrics)

            # Track metrics
            self.metrics.iteration_times.append(iter_metrics["total_time"])
            self.metrics.ttft_times.append(iter_metrics["ttft"])
            self.metrics.prefill_times.append(iter_metrics["prefill_time"])
            self.metrics.decode_times.append(iter_metrics["total_decode_time"])

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics(all_metrics)

        # Calculate cost per million tokens
        self._calculate_cost_metrics()

        # Memory metrics
        if torch.cuda.is_available():
            self.metrics.memory_used_gb = torch.cuda.memory_allocated() / 1e9
            self.metrics.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

        return self.metrics

    def _calculate_aggregate_metrics(self, all_metrics: list[dict[str, Any]]):
        """Calculate aggregate metrics from individual iterations"""
        # Average throughput
        throughputs = [m["throughput"] for m in all_metrics]
        self.metrics.throughput_tokens_per_sec = np.mean(throughputs)

        # Average latency
        total_times = [m["total_time"] for m in all_metrics]
        total_tokens = self.config.output_length * self.config.batch_size
        self.metrics.latency_ms_per_token = np.mean(total_times) / total_tokens

        # TTFT
        ttfts = [m["ttft"] for m in all_metrics]
        self.metrics.time_to_first_token_ms = np.mean(ttfts)

        # TBOT
        self.metrics.time_between_output_tokens_ms = np.mean([m["avg_tbot"] for m in all_metrics])

        # Total time
        self.metrics.total_time_ms = np.mean(total_times)

        # Prefill metrics
        prefill_throughputs = [m["prefill_throughput"] for m in all_metrics]
        self.metrics.prefill_throughput_tokens_per_sec = np.mean(prefill_throughputs)
        prefill_times = [m["prefill_time"] for m in all_metrics]
        self.metrics.prefill_time_ms = np.mean(prefill_times)

        # Decode metrics
        decode_throughputs = [m["decode_throughput"] for m in all_metrics]
        self.metrics.decode_throughput_tokens_per_sec = np.mean(decode_throughputs)
        decode_times = [m["total_decode_time"] for m in all_metrics]
        self.metrics.decode_time_ms = np.mean(decode_times)

    def _calculate_cost_metrics(self):
        """Calculate cost per million tokens based on GPU type and usage"""
        # Detect GPU type (simplified - in real scenario would use actual detection)
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "Unknown"

        # Map to cost (simplified mapping)
        if "H100" in gpu_name:
            cost_per_hour = self.config.h100_cost_per_hour
        elif "H200" in gpu_name:
            cost_per_hour = self.config.h200_cost_per_hour
        elif "B200" in gpu_name:
            cost_per_hour = self.config.b200_cost_per_hour
        else:
            cost_per_hour = self.config.h100_cost_per_hour  # Default

        # Calculate cost per million tokens
        tokens_per_hour = self.metrics.throughput_tokens_per_sec * 3600
        if tokens_per_hour > 0:
            self.metrics.cost_per_million_tokens = (cost_per_hour / tokens_per_hour) * 1_000_000

    def print_results(self):
        """Print benchmark results in a formatted way"""
        print("\n" + "=" * 60)
        print(f"BENCHMARK RESULTS - {self.config.model_name}")
        if self.config.scenario:
            scenario_config = BENCHMARK_SCENARIOS[self.config.scenario]
            print(f"SCENARIO: {scenario_config['name']}")
        print("=" * 60)

        print("\nThroughput Metrics:")
        print(f"  Overall Throughput: {self.metrics.throughput_tokens_per_sec:.2f} tokens/sec")
        print(f"  Prefill Throughput: {self.metrics.prefill_throughput_tokens_per_sec:.2f} tokens/sec")
        print(f"  Decode Throughput: {self.metrics.decode_throughput_tokens_per_sec:.2f} tokens/sec")
        print(f"  Latency: {self.metrics.latency_ms_per_token:.2f} ms/token")

        print("\nLatency Breakdown:")
        print(f"  Time to First Token (TTFT): {self.metrics.time_to_first_token_ms:.2f} ms")
        print(f"  Time Between Output Tokens (TBOT): {self.metrics.time_between_output_tokens_ms:.2f} ms")
        print(f"  Prefill Time: {self.metrics.prefill_time_ms:.2f} ms")
        print(f"  Decode Time: {self.metrics.decode_time_ms:.2f} ms")
        print(f"  Total Generation Time: {self.metrics.total_time_ms:.2f} ms")

        print("\nMemory Usage:")
        print(f"  Current Memory: {self.metrics.memory_used_gb:.2f} GB")
        print(f"  Peak Memory: {self.metrics.peak_memory_gb:.2f} GB")

        print("\nCost Analysis:")
        print(f"  Cost per Million Tokens: ${self.metrics.cost_per_million_tokens:.4f}")

        # Variance analysis
        if self.metrics.iteration_times:
            print("\nVariance Analysis:")
            print(f"  Throughput Std Dev: {np.std([t for t in self.metrics.iteration_times]):.2f} ms")
            print(f"  TTFT Std Dev: {np.std(self.metrics.ttft_times):.2f} ms")

    def save_results(self, filename: str):
        """Save results to JSON file"""
        results = {
            "config": self.config.__dict__,
            "metrics": {
                "throughput_tokens_per_sec": self.metrics.throughput_tokens_per_sec,
                "prefill_throughput_tokens_per_sec": self.metrics.prefill_throughput_tokens_per_sec,
                "decode_throughput_tokens_per_sec": self.metrics.decode_throughput_tokens_per_sec,
                "latency_ms_per_token": self.metrics.latency_ms_per_token,
                "time_to_first_token_ms": self.metrics.time_to_first_token_ms,
                "time_between_output_tokens_ms": self.metrics.time_between_output_tokens_ms,
                "prefill_time_ms": self.metrics.prefill_time_ms,
                "decode_time_ms": self.metrics.decode_time_ms,
                "total_time_ms": self.metrics.total_time_ms,
                "memory_used_gb": self.metrics.memory_used_gb,
                "peak_memory_gb": self.metrics.peak_memory_gb,
                "cost_per_million_tokens": self.metrics.cost_per_million_tokens,
            },
            "detailed_metrics": {
                "iteration_times": self.metrics.iteration_times,
                "ttft_times": self.metrics.ttft_times,
                "prefill_times": self.metrics.prefill_times,
                "decode_times": self.metrics.decode_times,
            },
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {filename}")


def run_semianalysis_benchmark(
    model_name: str = "llama3.1-8b",
    batch_size: int = 1,
    input_length: int = 1024,  # default 1k -> 1k
    output_length: int = 1024,  # default 1k -> 1k
    num_iterations: int = 100,
    num_layers: int | None = None,
    mode: str = "thunder",
    save_results: bool = True,
    scenario: str | None = None,
    dtensor_single_gpu: bool = False,
    load_nvfp4: bool = False,
):
    """Main function to run the benchmark"""

    # Apply scenario configuration if specified
    if scenario is not None:
        if scenario not in BENCHMARK_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {list(BENCHMARK_SCENARIOS.keys())}")

        scenario_config = BENCHMARK_SCENARIOS[scenario]
        input_length = scenario_config["input_length"]
        output_length = scenario_config["output_length"]

        print(f"\nUsing standardized scenario: {scenario_config['name']}")
        print(f"Configuration: {scenario_config['description']}")
        print(f"Workload balance: {scenario_config['workload_balance']}")
        print(f"Hardware focus: {scenario_config['hardware_focus']}")

    config = InferenceBenchmarkConfig(
        model_name=model_name,
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
        num_iterations=num_iterations,
        mode=mode,
        num_layers=num_layers,
        scenario=scenario,
        dtensor_single_gpu=dtensor_single_gpu,
        load_nvfp4=load_nvfp4,
    )

    benchmark = SemiAnalysisInferenceBenchmark(config)

    metrics = benchmark.run_benchmark()
    benchmark.print_results()

    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scenario_suffix = f"_{scenario}" if scenario else ""
        filename = f"thunder_semianalysis_{model_name}_{scenario_suffix}_{timestamp}.json"
        benchmark.save_results(filename)

    return metrics


def list_scenarios():
    """Print available benchmark scenarios"""
    print("\nAvailable Standard Benchmark Scenarios:")
    print("=" * 50)
    for key, config in BENCHMARK_SCENARIOS.items():
        print(f"\n{key.upper()}:")
        print(f"  Name: {config['name']}")
        print(f"  Configuration: {config['description']}")
        print(f"  Workload Balance: {config['workload_balance']}")
        print(f"  Hardware Focus: {config['hardware_focus']}")
    print("\n" + "=" * 50)
    print("Use --scenario <scenario_name> to select a standard scenario")
    print("Or use --input-length and --output-length for custom configurations")


class CustomFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def main():
    """Command line interface for the benchmark"""
    parser = argparse.ArgumentParser(
        description="Thunder Inference Benchmark following SemiAnalysis Methodology",
        formatter_class=CustomFormatter,
        epilog="""
Standard Benchmark Scenarios:
  summarization  - Prefill-Heavy: 4,000 input → 1,000 output tokens
  chat          - Balanced: 1,000 input → 1,000 output tokens
  reasoning     - Decode-Heavy: 1,000 input → 4,000 output tokens

Use --list-scenarios for detailed scenario descriptions.

Examples:
  python inference_bmk.py --scenario chat --model-name llama3.1-8b
  python inference_bmk.py --input-length 2048 --output-length 512 --model-name llama3.1-8b --mode eager
  python inference_bmk.py --scenario chat --model-name llama3.1-8b --load-nvfp4
        """,
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="llama3.1-8b",  # Small model so it's easier to iterate locally.
        help="Model to benchmark",
    )

    # Scenario configuration (standardized scenarios vs custom)
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(BENCHMARK_SCENARIOS.keys()),
        help="Use standardized benchmark scenario. Available: "
        + ", ".join([f"{k} ({v['description'].replace('%', '%%')})" for k, v in BENCHMARK_SCENARIOS.items()])
        + ". If specified, overrides --input-length and --output-length.",
    )

    # Benchmark configuration (for custom experimentation when not using scenarios)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument(
        "--input-length", type=int, default=2048, help="Input sequence length (ignored if --scenario is used)"
    )
    parser.add_argument(
        "--output-length", type=int, default=128, help="Output sequence length (ignored if --scenario is used)"
    )
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup-iterations", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num-layers", type=int, help="Number of layers of the moddel")

    # Execution configuration
    parser.add_argument(
        "--mode",
        type=str,
        default="eager",
        help="Compilation mode: thunder, eager (default), or inductor",
    )

    parser.add_argument(
        "--dtensor-single-gpu",
        action="store_true",
        help="Use DTensor for single GPU",
    )
    parser.add_argument("--load-nvfp4", action="store_true", help="Enable NVFP4 quantization for linear layers")

    # Output configuration
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument(
        "--list-scenarios", action="store_true", help="List available standard benchmark scenarios and exit"
    )

    args = parser.parse_args()

    # Handle list scenarios
    if args.list_scenarios:
        list_scenarios()
        return None

    # Create output directory if needed
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmark
    metrics = run_semianalysis_benchmark(
        model_name=args.model_name,
        batch_size=args.batch_size,
        input_length=args.input_length,
        output_length=args.output_length,
        num_iterations=args.num_iterations,
        num_layers=args.num_layers,
        mode=args.mode,
        save_results=args.save_results,
        scenario=args.scenario,
        dtensor_single_gpu=args.dtensor_single_gpu,
        load_nvfp4=args.load_nvfp4,
    )

    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
    finally:
        from torch.distributed.distributed_c10d import destroy_process_group

        for process_group in mesh.get_all_groups():
            destroy_process_group(process_group)
