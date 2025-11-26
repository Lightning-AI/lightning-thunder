"""Inference benchmark focusing on throughput and latency metrics of prefill and decode phases.

AutoModelForCausalLM from Hugging Face transformers is used for model implementation.

Key metrics:
- Throughput (tokens/second)
- Latency (ms/token)
- Time to First Token (TTFT)
- Time Between Output Tokens (TBOT)
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import argparse
import json
import os
import statistics
import time
import warnings
from typing import Any
from collections.abc import Callable
from looseversion import LooseVersion

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel, ColwiseParallel
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import HybridChunkedCache, StaticCache
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor import DTensor

import thunder
from thunder.dynamo.compiler import thunderfx
from thunder.benchmarks.layers_for_inference_benchmark import (
    GroupedSwiGLU,
    Llama4MoE,
    NVFP4InferenceGroupedSwiGLU,
    nvfuser_f16a_nvfp4weight_scaled_grouped_mm,
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_EPS,
    FLOAT8_E4M3_MAX,
)
from thunder.tests.distributed.test_moe import GroupedLinearColwiseParallel, GroupedLinearRowwiseParallel
from thunder.transforms.cudagraph import CUDAGraphTransform
from thunder.torch.custom_op import _register_custom_op, _register_nvfuser_translator

if TYPE_CHECKING:
    from typing import Any


RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

DEVICE = torch.device("cuda", LOCAL_RANK)
torch.cuda.set_device(DEVICE)

if dist.is_torchelastic_launched():
    mesh = init_device_mesh("cuda", (WORLD_SIZE,), mesh_dim_names=("tp",))
else:
    mesh = None

LLAMA4_MAVERICK_MODEL_ID: str = "meta-llama/Llama-4-Maverick-17B-128E"


# TODO: Add mm quantization once nvfuser implements nvfp4 gemm
# Register nvfp4 custom ops with Thunder and nvFuser
def _register_nvfp4_ops():
    """Register nvfp4 custom operations with Thunder."""
    # Register f16a_nvfp4weight_scaled_grouped_mm with nvfuser translator
    _nvfp4_grouped_mm_symbol = _register_custom_op(nvfuser_f16a_nvfp4weight_scaled_grouped_mm)

    def nvfp4_grouped_mm_translator(
        activation,
        fp4_weight,
        weight_scaling_factor,
        global_scale,
        offsets,
        blockscale_offsets,
        problem_sizes,
        *,
        fd,
        lc_to_nv_map,
    ):
        from nvfuser_direct import DataType
        from thunder.executors.nvfuserex_impl import getnv

        nv_act = getnv(activation, fd, lc_to_nv_map)
        nv_fp4_w = getnv(fp4_weight, fd, lc_to_nv_map)
        nv_sf_w = getnv(weight_scaling_factor, fd, lc_to_nv_map)
        nv_alpha = getnv(global_scale, fd, lc_to_nv_map)
        nv_offsets = getnv(offsets, fd, lc_to_nv_map)
        nv_blocksf_offsets = getnv(blockscale_offsets, fd, lc_to_nv_map)
        nv_problem_sizes = getnv(problem_sizes, fd, lc_to_nv_map)
        # dynamic shape support has some concretization issue
        m_size = activation.shape[0]
        k_size = activation.shape[1]
        k_tile_size = k_size // 16

        reshaped_mat1 = fd.ops.reshape(nv_act, [m_size, k_tile_size, 16])
        scale1 = fd.ops.abs(reshaped_mat1)
        scale1 = fd.ops.max(scale1, 2)
        scale1 = fd.ops.div(scale1, FLOAT4_E2M1_MAX)
        scale1 = fd.ops.clamp(scale1, FLOAT8_E4M3_EPS, FLOAT8_E4M3_MAX)

        broadcast_scale1 = fd.ops.broadcast(scale1, [False, False, True])
        reshaped_scaled_mat1 = fd.ops.div(reshaped_mat1, broadcast_scale1)
        reshaped_scaled_mat1 = fd.ops.clamp(reshaped_scaled_mat1, -FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)

        scaled_mat1 = fd.ops.reshape(reshaped_scaled_mat1, [m_size, k_size])
        fp4_mat1 = fd.ops.cast(scaled_mat1, DataType.Float4_e2m1fn)
        fp8_scale1 = fd.ops.cast(scale1, DataType.Float8_e4m3fn)
        layout_fp8_scale1 = fd.ops.preprocess_grouped_matmul_input_sf(fp8_scale1, nv_offsets, nv_blocksf_offsets)
        out = fd.ops.cutlass_nvfp4_grouped_mm(
            fp4_mat1,
            nv_fp4_w,
            layout_fp8_scale1,
            nv_sf_w,
            nv_alpha,
            # NOTE: we might need to call contiguous on problem_sizes
            nv_problem_sizes,
            nv_offsets,
            nv_blocksf_offsets,
            DataType.BFloat16,
        )
        return out

    _register_nvfuser_translator(_nvfp4_grouped_mm_symbol, nvfp4_grouped_mm_translator)


# The logic is based on https://github.com/pytorch/ao/blob/b34c1037/torchao/quantization/quant_api.py#L230
def _replace_with_custom_fn_if_matches_filter_with_name(
    model,
    replacement_fn: Callable[[torch.nn.Module, str], torch.nn.Module],
    filter_fn: Callable[[torch.nn.Module, str], bool],
    cur_fqn="",
) -> None:
    """
    Recursively replaces each child module in `model` with the result of `replacement_fn(child)`

    replacement_fn (Callable[[torch.nn.Module, str], torch.nn.Module]): The function to replace matching modules.
    filter_fn (Callable[[torch.nn.Module, str], bool]): The function to filter matching modules.
    cur_fqn (str): The current fully qualified name of the module.

    Returns:
        None
    """
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model, cur_fqn[:-1])
        return model
    else:
        named_children_list = list(model.named_children())
        for name, child in named_children_list:
            new_child = _replace_with_custom_fn_if_matches_filter_with_name(
                child,
                replacement_fn,
                filter_fn,
                f"{cur_fqn}{name}.",
            )
            if new_child is not child:
                setattr(model, name, new_child)
        return model


def _replace_llama4_moe(model: nn.Module) -> None:
    """Replace Llama4TextMoe with Llama4MoE to use grouped gemm."""
    _replace_with_custom_fn_if_matches_filter_with_name(
        model,
        lambda model, cur_fqn: Llama4MoE.from_transformers_llama4textmoe(model),
        lambda model, cur_fqn: isinstance(model, Llama4TextMoe),
    )


def _quantize_llama4(model: nn.Module) -> None:
    """Replace linear and/or MoE with nvfp4 inference version.

    Args:
        model: The model to quantize

    Note: GroupedSwiGLU is always quantized when this function is called.
    """
    # Always quantize GroupedSwiGLU when this function is called
    _replace_with_custom_fn_if_matches_filter_with_name(
        model,
        NVFP4InferenceGroupedSwiGLU.from_grouped_swiglu,
        lambda model, cur_fqn: isinstance(model, GroupedSwiGLU),
    )


@contextmanager
def timer():
    torch.cuda.synchronize()
    t1 = t2 = time.perf_counter()
    yield lambda: (t2 - t1) * 1000  # Convert to ms
    torch.cuda.synchronize()
    t2 = time.perf_counter()


@dataclass
class InferenceBenchmarkConfig:
    """Configuration for inference benchmarking"""

    model_name: str
    batch_size: int
    input_length: int
    output_length: int
    num_layers: int | None
    num_iterations: int
    warmup_iterations: int
    enable_nvfp4: bool  # Enable NVFP4 registration and quantize GroupedSwiGLU in MoE
    fx_report_folder: str | None
    enable_nv_linear: bool
    mode: str
    disable_moe_replacement: bool
    attn_implementation: str | None
    profile: bool
    thunder_cache: str | None
    enable_thunder_cudagraph: bool


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


class InferenceBenchmark:
    """Main benchmark class"""

    def __init__(self, config: InferenceBenchmarkConfig):
        self.config = config
        self.metrics = InferenceMetrics()

        # NOTE: Model resides on meta device
        model = self._load_model()
        assert all(p.device == torch.device("meta") for p in model.parameters())

        # NOTE: Replacement happens before model is materialized
        #       otherwise, the memory usage will be increased due to
        #       additional parameters materialized from the replacement module
        if not self.config.disable_moe_replacement:
            _replace_llama4_moe(model)
        assert all(p.device == torch.device("meta") for p in model.parameters())

        tp_plan = {
            "*.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=True),
            "*.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=True),
            "*.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=True),
            "*.layers.*.self_attn.o_proj": RowwiseParallel(use_local_output=True),
            "*.layers.*.feed_forward.gate_proj": ColwiseParallel(use_local_output=False),
            "*.layers.*.feed_forward.up_proj": ColwiseParallel(use_local_output=False),
            "*.layers.*.feed_forward.down_proj": RowwiseParallel(use_local_output=True),
        }

        if not self.config.disable_moe_replacement:
            tp_plan.update(
                {
                    # Custom MoE
                    "*.layers.*.feed_forward.shared_experts.gate_proj": ColwiseParallel(
                        use_local_output=False, output_layouts=Shard(2)
                    ),
                    "*.layers.*.feed_forward.shared_experts.up_proj": ColwiseParallel(
                        use_local_output=False, output_layouts=Shard(2)
                    ),
                    "*.layers.*.feed_forward.shared_experts.down_proj": RowwiseParallel(),
                    "*.layers.*.feed_forward.routed_experts.gate_proj": GroupedLinearColwiseParallel(
                        use_local_output=False
                    ),
                    "*.layers.*.feed_forward.routed_experts.up_proj": GroupedLinearColwiseParallel(
                        use_local_output=False
                    ),
                    "*.layers.*.feed_forward.routed_experts.down_proj": GroupedLinearRowwiseParallel(),
                }
            )

        else:
            tp_plan.update(
                {
                    # HF MoE
                    "*.layers.*.feed_forward.shared_expert.gate_proj": ColwiseParallel(use_local_output=False),
                    "*.layers.*.feed_forward.shared_expert.up_proj": ColwiseParallel(use_local_output=False),
                    "*.layers.*.feed_forward.shared_expert.down_proj": RowwiseParallel(use_local_output=True),
                    # TODO:Need to write ParallelStyle for HF's grouped_mm implementation.
                }
            )

        if mesh:
            model = parallelize_module(model, mesh, tp_plan)

            # Sanity check
            if not self.config.disable_moe_replacement:
                assert type(model.model.layers[1].feed_forward.shared_experts.gate_proj.weight) == DTensor
                assert type(model.model.layers[1].feed_forward.shared_experts.up_proj.weight) == DTensor
                assert type(model.model.layers[1].feed_forward.shared_experts.down_proj.weight) == DTensor
                assert type(model.model.layers[1].feed_forward.routed_experts.gate_proj.weight) == DTensor
                assert type(model.model.layers[1].feed_forward.routed_experts.up_proj.weight) == DTensor
                assert type(model.model.layers[1].feed_forward.routed_experts.down_proj.weight) == DTensor
            else:
                assert type(model.model.layers[1].feed_forward.shared_expert.gate_proj.weight) == DTensor
                assert type(model.model.layers[1].feed_forward.shared_expert.up_proj.weight) == DTensor
                assert type(model.model.layers[1].feed_forward.shared_expert.down_proj.weight) == DTensor

        # Materialize the model on the device (after Llama4MoE replacement and sharding)
        model.to_empty(device=DEVICE)
        assert all(p.device == DEVICE for p in model.parameters())

        # Required as thunder doesn't understand inference mode
        # And some prims like `prims._grouped_mm` don't have grad rule defined yet.
        for p in model.parameters():
            p.requires_grad_(False)

        assert all(not p.requires_grad for p in model.parameters())

        # `thunderfx` seems to hide the access to vocab_size somewhere so
        # store it here before any compiler is applied.
        self.vocab_size = model.vocab_size

        if self.config.enable_nvfp4:
            _quantize_llama4(model)
        self.model = self._compile_model(model)

    @property
    def _thunder_jit_options(self) -> dict[str, Any]:
        # `nv_enable_linear=True` might fail with distributed run
        # ref: https://github.com/NVIDIA/Fuser/issues/4507
        res = {"transforms": []}
        if self.config.enable_nv_linear:
            res["nv_enable_linear"] = True
            res["nv_enable_matmul"] = True
        if self.config.mode == "thunderjit":
            from thunder.recipes.hf_transformers import SDPAMaskTransform

            if not hasattr(self, "_mask_transform"):
                self._mask_transform = SDPAMaskTransform()
            res["transforms"].append(self._mask_transform)
            res["executors"] = [self._mask_transform.get_executor(), *thunder.get_default_executors()]
        if self.config.enable_thunder_cudagraph:
            res["transforms"].append(CUDAGraphTransform())
        if self.config.thunder_cache is not None:
            res["cache"] = self.config.thunder_cache

        return res

    def _compile_model(self, model):
        match self.config.mode:
            case "eager":
                return model
            case "inductor":
                return torch.compile(model, mode="reduce-overhead")
            case "thunder":
                return thunderfx(model, **self._thunder_jit_options)
            case "thunderjit":
                return thunder.jit(model, **self._thunder_jit_options)
            case _:
                raise ValueError(f"Unknown mode: {self.config.mode}")

    def _load_model(self) -> torch.nn.Module:
        """Load the model based on configuration"""
        model_id = self.config.model_name
        config = AutoConfig.from_pretrained(model_id)

        if hasattr(config, "text_config"):
            config = config.text_config
        if self.config.num_layers:
            config.num_hidden_layers = self.config.num_layers

        self.hf_config = config

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch.bfloat16, attn_implementation=self.config.attn_implementation
            )

        return model

    def generate_batch(self) -> tuple[torch.Tensor, HybridChunkedCache]:
        """Generate a batch of input tokens"""
        batch_size = self.config.batch_size
        input_length = self.config.input_length

        input_ids = torch.randint(0, self.vocab_size, (batch_size, input_length), device=DEVICE)
        if LooseVersion(transformers.__version__) >= LooseVersion("4.55"):
            # Transformers deprecated HybridChunkedCache in favour of static in 4.55.x
            past_key_values = StaticCache(
                config=self.hf_config,
                max_batch_size=input_ids.shape[0],
                max_cache_len=input_ids.shape[1] + self.config.output_length,
                device=DEVICE,
                dtype=torch.bfloat16,
            )
        else:
            past_key_values = HybridChunkedCache(
                self.hf_config, input_ids.shape[0], input_ids.shape[1] + self.config.output_length
            )
            for layer_idx in range(self.hf_config.num_hidden_layers):
                # key_states.shape[1] is used to retrieve the number of key value heads, all other dimensions can be 1 and ignored
                # https://github.com/huggingface/transformers/blob/9300728665aaeb0ebf4db99f9d9fbce916b4a183/src/transformers/cache_utils.py#L1822
                dummy_key_states = torch.empty(1, self.hf_config.num_key_value_heads // WORLD_SIZE, 1, 1, device=DEVICE)
                past_key_values.initialise_cache_layer(layer_idx, dummy_key_states)

        return input_ids, past_key_values

    def get_next_token(
        self, input_ids: torch.Tensor, past_key_values: HybridChunkedCache | StaticCache
    ) -> torch.Tensor:
        with torch.no_grad():
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

    # TODO: Running `torchrun --nproc-per-node 2 thunder/benchmarks/benchmark_inference.py --input-length 32 --output-length 32 --mode eager --num-iterations 10`
    # with inference mode results in
    # [rank1]:   File "/opt/pytorch/lightning-thunder/thunder/benchmarks/layers_for_inference_benchmark.py", line 358, in grouped_mm
    # [rank1]:     group_outs.append(group_a @ b[idx])
    # [rank1]:                                 ~^^^^^
    # [rank1]: RuntimeError: Cannot set version_counter for inference tensor
    # @torch.inference_mode()
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
        # Generate tokens with separate prefill/decode tracking
        generation_result = self.generate(input_ids, max_new_tokens, past_key_values)
        total_time = generation_result["prefill_time_ms"] + generation_result["decode_time_ms"]

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
        print(f"Running inference benchmark for {self.config.model_name}")

        print(f"Batch size: {self.config.batch_size}")
        print(f"Input length: {self.config.input_length}")
        print(f"Output length: {self.config.output_length}")
        print(f"Mode: {self.config.mode}")

        print(f"\nWarming up with {self.config.warmup_iterations} iterations...")
        input_ids, past_key_values = self.generate_batch()

        for _ in tqdm(range(self.config.warmup_iterations), disable=LOCAL_RANK != 0):
            past_key_values.reset()
            # Use output_length to warm up sufficiently. Otherwise, Thunder's
            # first-run latency is terribly slow due to lack of dynamic shape
            # support.
            _ = self.measure_inference_step(input_ids, past_key_values, self.config.output_length)

        print(f"\nRunning {self.config.num_iterations} benchmark iterations...")
        all_metrics = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in tqdm(range(self.config.num_iterations), disable=LOCAL_RANK != 0):
            past_key_values.reset()

            if self.config.profile:
                torch.cuda.cudart().cudaProfilerStart()
            iter_metrics = self.measure_inference_step(input_ids, past_key_values, self.config.output_length)
            if self.config.profile:
                torch.cuda.cudart().cudaProfilerStop()

            all_metrics.append(iter_metrics)

            # Track metrics
            self.metrics.iteration_times.append(iter_metrics["total_time"])
            self.metrics.ttft_times.append(iter_metrics["ttft"])
            self.metrics.prefill_times.append(iter_metrics["prefill_time"])
            self.metrics.decode_times.append(iter_metrics["total_decode_time"])

        self._calculate_aggregate_metrics(all_metrics)

        if torch.cuda.is_available():
            self.metrics.memory_used_gb = torch.cuda.memory_allocated() / 1e9
            self.metrics.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

        if self.config.fx_report_folder is not None and self.config.mode == "thunder":
            self.model._backend.save_reproducer_to_folder(self.config.fx_report_folder)
            return

        return self.metrics

    def _calculate_aggregate_metrics(self, all_metrics: list[dict[str, Any]]):
        """Calculate aggregate metrics from individual iterations"""
        # Average throughput
        throughputs = [m["throughput"] for m in all_metrics]
        self.metrics.throughput_tokens_per_sec = statistics.mean(throughputs)

        # Average latency
        total_times = [m["total_time"] for m in all_metrics]
        total_tokens = self.config.output_length * self.config.batch_size
        self.metrics.latency_ms_per_token = statistics.mean(total_times) / total_tokens

        # TTFT
        ttfts = [m["ttft"] for m in all_metrics]
        self.metrics.time_to_first_token_ms = statistics.mean(ttfts)

        # TBOT
        self.metrics.time_between_output_tokens_ms = statistics.mean([m["avg_tbot"] for m in all_metrics])

        # Total time
        self.metrics.total_time_ms = statistics.mean(total_times)

        # Prefill metrics
        prefill_throughputs = [m["prefill_throughput"] for m in all_metrics]
        self.metrics.prefill_throughput_tokens_per_sec = statistics.mean(prefill_throughputs)
        prefill_times = [m["prefill_time"] for m in all_metrics]
        self.metrics.prefill_time_ms = statistics.mean(prefill_times)

        # Decode metrics
        decode_throughputs = [m["decode_throughput"] for m in all_metrics]
        self.metrics.decode_throughput_tokens_per_sec = statistics.mean(decode_throughputs)
        decode_times = [m["total_decode_time"] for m in all_metrics]
        self.metrics.decode_time_ms = statistics.mean(decode_times)

    def print_results(self):
        """Print benchmark results in a formatted way"""
        print("\n" + "=" * 60)
        print(f"BENCHMARK RESULTS - {self.config.model_name} {self.config.mode}")
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

        if len(self.metrics.iteration_times) > 1:
            print("\nVariance Analysis:")
            print(f"  Throughput Std Dev: {statistics.stdev(self.metrics.iteration_times):.2f} ms")
            print(f"  TTFT Std Dev: {statistics.stdev(self.metrics.ttft_times):.2f} ms")

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


class CustomFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def parse_args() -> argparse.Namespace:
    """Command line interface for the benchmark"""
    parser = argparse.ArgumentParser(
        description="Thunder Inference Benchmark",
        formatter_class=CustomFormatter,
        epilog="""
Examples:
  python benchmark_inference.py --input-length 2048 --output-length 512 --model-name meta-llama/Llama-4-Maverick-17B-128E --mode eager
        """,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=LLAMA4_MAVERICK_MODEL_ID,
        help="Model to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument(
        "--input-length",
        type=int,
        default=2048,
        help="Input sequence length",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=128,
        help="Output sequence length",
    )
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup-iterations", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument(
        "--num-layers",
        default=2,
        type=int,
        help="Number of layers of the moddel. Llama4 Maverick has 48 hidden layers, which could be too memory hungry",
    )
    parser.add_argument(
        "--disable-moe-replacement",
        action="store_true",
        help="Disallow replacement of Llama4TextMoe with our custom Llama4MoE which uses grouped gemm.",
    )

    # Execution configuration
    parser.add_argument(
        "--mode",
        type=str,
        default="eager",
        choices=("thunder", "eager", "inductor", "thunderjit"),
        help="Compilation mode: thunder, eager (default), or inductor. thunder runs thunderfx.",
    )
    parser.add_argument(
        "--fx-report-folder",
        default=None,
        type=str,
        help="Specify the folder for thunderfx_benchmark_report.",
    )

    parser.add_argument(
        "--enable-nvfp4",
        action="store_true",
        help="Enable NVFP4 quantization for MoE GroupedSwiGLU layers (has nvfuser grouped_mm support)",
    )
    parser.add_argument(
        "--enable-nv-linear",
        action="store_true",
        help="let nvfuser take care of linear and matmul, note that this might fail with distributed run. See: https://github.com/NVIDIA/Fuser/issues/4507",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap each non-warmup iteration with cudaProfilerStart() and cudaProfilerStop(). This allows us to run `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:<N> ... --profile` to record only the non-warmup iterations.",
    )

    parser.add_argument(
        "--thunder-trace",
        action="store_true",
        help="Enable debug dump of thunder trace",
    )
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument(
        "--thunder-cache",
        type=str,
        default=None,
        help="Cache option: no caching, same input, constant values, symbolic values. See `cache` argument of `thunder.jit` for more details.",
    )
    parser.add_argument("--enable-thunder-cudagraph", action="store_true", help="Pass CUDAGraphTransform to Thunder")
    parser.add_argument("--attn-implementation", type=str, default=None, help="Attention implementation")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    # Register NVFP4 custom ops with nvfuser translators when enabled
    if args.enable_nvfp4:
        try:
            _register_nvfp4_ops()
        except Exception as e:
            # If registration fails (e.g., nvfuser not available), warn and continue
            warnings.warn(f"Failed to register nvfp4 custom ops: {e}")

    config = InferenceBenchmarkConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        input_length=args.input_length,
        output_length=args.output_length,
        num_layers=args.num_layers,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
        mode=args.mode,
        enable_nvfp4=args.enable_nvfp4,
        fx_report_folder=args.fx_report_folder,
        enable_nv_linear=args.enable_nv_linear,
        disable_moe_replacement=args.disable_moe_replacement,
        attn_implementation=args.attn_implementation,
        profile=args.profile,
        thunder_cache=args.thunder_cache,
        enable_thunder_cudagraph=args.enable_thunder_cudagraph,
    )
    benchmark = InferenceBenchmark(config)

    benchmark.run_benchmark()
    benchmark.print_results()

    if args.thunder_trace and args.mode == "thunder":
        backend = benchmark.model._backend
        for subgraph_info in backend.subgraph_infos:
            assert isinstance(subgraph_info.original_graph_module, torch.fx.GraphModule)
            assert len(subgraph_info.thunder_compiled_fns)
            for thunder_fn in subgraph_info.thunder_compiled_fns:
                print(thunder.last_traces(thunder_fn)[-1])

    if args.save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"thunder_inference_{args.model_name.replace('/', '_')}_{timestamp}.json"
        path = os.path.join(args.output_dir, filename)
        benchmark.save_results(path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
    finally:
        if mesh:
            for process_group in mesh.get_all_groups():
                dist.destroy_process_group(process_group)
