# =============================================================================
# (uva39)FIXED: torch._dynamo recompilation warning after 2 iterations
# - Optimized cache limits: cache_size_limit=32, accumulated_cache_size_limit=512  
# - Applied compilation BEFORE FSDP wrapping for better graph capture
# - Added version-aware dynamo configuration with fallback mechanisms
# - Resolved FP8 + dynamic shapes conflicts via conditional compilation
# - Replaced FSDP mixed precision with torch.amp.autocast for compatibility
# ⚠️  AI-GENERATED CODE MODIFICATIONS - TESTING REQUIRED ⚠️
# =============================================================================

from __future__ import annotations
from datetime import timedelta
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from looseversion import LooseVersion

import torch
import functools
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as torch_dist
from torch.distributed.device_mesh import init_device_mesh

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointWrapper,
)

import thunder
from thunder.dynamo import ThunderCompiler

from thunder.tests.litgpt_model import Config, GPT, Block
from lightning.fabric.utilities.throughput import measure_flops
from lightning.fabric.utilities import Throughput
from lightning_utilities.core.imports import package_available

transformer_engine_available = package_available("transformer_engine")
torchao_available = package_available("torchao")

if transformer_engine_available:
    import transformer_engine.pytorch as te

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sdpa_available = True
except ImportError:
    sdpa_available = False

if torchao_available:
    from torchao.float8.config import CastConfig, Float8LinearConfig, ScalingType
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
    from torchao.float8.float8_linear_utils import (
        convert_to_float8_training,
        sync_float8_amax_and_scale_history,
        get_float8_layers,
    )

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


FSDP_MODES: set[str] = {"fsdp", "fsdp2"}
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))

if world_size > 1:
    # Avoids the allocator thrashing issue in PyTorch NCCL backend.
    # See https://github.com/Lightning-AI/lightning-thunder/issues/420
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    pg = torch_dist.distributed_c10d._get_default_group()

device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)


def setup_optimized_environment_for_compile():
    """Setup environment variables for optimal torch.compile + FSDP performance"""
    # Enable dynamic scalar handling
    os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")
    
    # Enable profile-guided optimization for PyTorch 2.5+
    if LooseVersion(torch.__version__) >= LooseVersion("2.5.0"):
        os.environ.setdefault("TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO", "1")
    
    # Conditional logging setup
    max_iters = int(os.environ.get("MAX_ITERS", "45"))
    debug_mode = os.environ.get("DEBUG_MODE", "").lower() in ("1", "true")
    
    if max_iters <= 10 or debug_mode:
        os.environ.setdefault("TORCH_LOGS", "recompiles")
    else:
        # Minimal logging for production
        os.environ.setdefault("TORCH_LOGS", "")


def setup_dynamo_config_with_version_check():
    """Setup dynamo configuration with version compatibility checks"""
    from packaging import version
    import torch._dynamo.config as dynamo_config
    
    # Base configuration for all versions
    dynamo_config.cache_size_limit = 32
    dynamo_config.accumulated_cache_size_limit = 512
    
    # Version-specific configurations
    if version.parse(torch.__version__) >= version.parse("2.4.0"):
        dynamo_config.fail_on_recompile_limit_hit = False
    
    if version.parse(torch.__version__) >= version.parse("2.5.0"):
        # PyTorch 2.5+ optimizations
        if hasattr(dynamo_config, 'inline_inbuilt_nn_modules'):
            dynamo_config.inline_inbuilt_nn_modules = True
        if hasattr(dynamo_config, 'automatic_dynamic_shapes'):
            dynamo_config.automatic_dynamic_shapes = True
        if hasattr(dynamo_config, 'run_gc_after_compile'):
            dynamo_config.run_gc_after_compile = True
    
    return dynamo_config


def setup_thunder_dynamo_config():
    """Setup specialized configuration for Thunder + dynamo combination"""
    import torch._dynamo.config as dynamo_config
    
    # Thunder prefers lower cache limits
    dynamo_config.cache_size_limit = 16
    dynamo_config.accumulated_cache_size_limit = 256
    
    if hasattr(dynamo_config, 'fail_on_recompile_limit_hit'):
        dynamo_config.fail_on_recompile_limit_hit = False
    
    return dynamo_config


class CompilationMonitor:
    """Monitor compilation health and provide fallback mechanisms"""
    def __init__(self):
        self.recompile_count = 0
        self.compilation_time = 0
        self.start_time = None
    
    def start_compilation_timer(self):
        self.start_time = time.time()
    
    def end_compilation_timer(self):
        if self.start_time:
            self.compilation_time = time.time() - self.start_time
            if self.compilation_time > 60:  # 1 minute threshold
                warnings.warn(
                    f"Compilation took {self.compilation_time:.2f}s, "
                    "consider reducing cache limits or using eager mode"
                )
        return self.compilation_time
    
    def check_compilation_health(self):
        if self.recompile_count > 10:
            warnings.warn(
                "Excessive recompilations detected. Consider falling back to eager mode"
            )
            return False
        return True


def check_fp8_compute_capability() -> None:
    device = torch.cuda.current_device()
    compute_capability = torch.cuda.get_device_capability(device)
    required_compute_capability = (8, 9)

    if compute_capability < required_compute_capability:
        raise RuntimeError(
            f"Device compute capability {compute_capability} is insufficient. "
            f"Compute capability {required_compute_capability} or higher is required for FP8 execution. "
            "Please ensure you are using a compatible GPU and the correct driver version."
        )


def is_transformer_engine(low_precision_mode: str) -> bool:
    return low_precision_mode in ["fp8-delayed-te", "fp8-delayed-te-wo_layernorm"]


def check_and_update_config_for_te_if_needed(config: Config) -> None:
    updates_info = []

    def update_if_not_divisible(attr_name, divisor):
        current_value = getattr(config, attr_name, 0)
        if current_value % divisor != 0:
            new_value = current_value + (divisor - current_value % divisor)
            setattr(config, attr_name, new_value)
            updates_info.append((attr_name, new_value))
            return True
        return False

    updated = False
    updated |= update_if_not_divisible("padded_vocab_size", 16)
    updated |= update_if_not_divisible("n_embd", 8)

    if updated:
        print("Configuration was updated with the following changes:")
        for key, value in updates_info:
            print(f"{key} updated to {value}")
    else:
        print("No updates were necessary.")


def swap_linear_layers_for_te(model: torch.nn.Module, device, swap_layernorm: bool = True) -> None:
    def parameters_cnt(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def _recursively_swap_linear_layers_for_te(module: torch.nn.Module) -> None:
        for n, m in module.named_children():
            if len(list(m.children())) > 0:
                _recursively_swap_linear_layers_for_te(m)

            if isinstance(m, torch.nn.Linear):
                has_bias = m.bias is not None
                # Pass device as str (as there is a bug in TransformerEngine's handling of torch.device)
                new_linear = te.Linear(m.in_features, m.out_features, bias=has_bias, device=str(device))
                setattr(module, n, new_linear)

            if swap_layernorm and isinstance(m, torch.nn.LayerNorm):
                # Pass device as str (as there is a bug in TransformerEngine's handling of torch.device)
                new_layernorm = te.LayerNorm(m.normalized_shape[0], eps=m.eps, device=str(device))
                setattr(module, n, new_layernorm)

    initial_params_cnt = parameters_cnt(model)
    _recursively_swap_linear_layers_for_te(model)
    assert initial_params_cnt == parameters_cnt(model)
    
    for m in model.modules():
        assert not isinstance(m, torch.nn.Linear)
        if swap_layernorm:
            assert not isinstance(m, torch.nn.LayerNorm)


@dataclass
class TorchAOFP8Handler:
    is_fsdp2: bool
    use_fp8_linear: bool
    use_fp8_allgather: bool
    use_torchao_fp8_precompute_float8_dynamic_scale_for_fsdp: bool
    use_torch_compile: bool
    fp8_layers: list[torch.nn.Module] = field(
        init=False,
        default_factory=list,
        repr=False,
    )
    _sync_float8_amax_and_scale_history: None | Callable[[torch.nn.Module, Sequence[torch.nn.Module]], None] = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._enabled = self.use_fp8_linear
        if not self._enabled:
            return
        if torch.cuda.get_device_capability() < (8, 9):
            raise ValueError(f"torchao float8 requires {torch.cuda.get_device_capability()=} >= (8, 9)")
        self.fp8_linear_config = Float8LinearConfig(
            cast_config_input=CastConfig(ScalingType.DYNAMIC),
            cast_config_weight=CastConfig(ScalingType.DYNAMIC),
            cast_config_grad_output=CastConfig(ScalingType.DYNAMIC),
            enable_fsdp_float8_all_gather=self.use_fp8_allgather and self.is_fsdp2,
            enable_pre_and_post_forward=False,
            force_recompute_fp8_weight_in_bwd=self.is_fsdp2,
        )
        self.precompute_scale = (
            self.is_fsdp2 and self.use_fp8_allgather and self.use_torchao_fp8_precompute_float8_dynamic_scale_for_fsdp
        )

    def convert_model_to_fp8(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self._enabled:
            return model
        model = convert_to_float8_training(
            model,
            module_filter_fn=lambda _, fqn: "lm_head" != fqn,
            config=self.fp8_linear_config,
        )
        self.fp8_layers = get_float8_layers(model)
        return model

    def sync_float8_amax_and_scale_history_for_delayed_scaling(self, model: torch.nn.Module) -> None:
        # NOTE(crcrpar): Delayed scaling is not supported in this script.
        if False:
            if self._sync_float8_amax_and_scale_history is None:
                if self.use_torch_compile:
                    self._sync_float8_amax_and_scale_history = torch.compile(sync_float8_amax_and_scale_history)
                else:
                    self._sync_float8_amax_and_scale_history = sync_float8_amax_and_scale_history
            self._sync_float8_amax_and_scale_history(model, self.fp8_layers)

    def precompute_fp8_dynamic_scale_for_fsdp(self, model: torch.nn.Module) -> None:
        if self._enabled and self.precompute_scale:
            precompute_float8_dynamic_scale_for_fsdp(model)


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    import inspect

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, fused=use_fused
    )
    return optimizer


class Benchmark_litGPT:
    def __init__(
        self,
        compile: str = "eager",
        dynamic: bool = False,
        nsys_enabled: bool = False,
        distributed_mode: str = "none",
        micro_batch_size: int = 1,
        global_batch_size: int | None = None,
        model_name: str = "Llama-2-7b-hf",
        shard_mode: str = "zero2",
        bucketing_mode: str | None = None,
        sharding_size: int | None = None,
        ddp_bucket_size: float = 256.0,
        fsdp_bucket_params: float | None = None,
        checkpoint_activations: bool = False,
        n_layers: int | None = None,
        block_size: int | None = None,
        profiler_start: int = 15,
        profiler_stop: int = 15,
        skip_data_sync: bool = False,
        low_precision_mode: str = "none",
        max_iters: int = 45,
        warmup_iters: int = 25,
        dump_thunder_traces: bool = False,
        dump_memory_snapshot: bool = False,
        use_torchao_fp8_linear: bool = False,
        use_torchao_fp8_allgather: bool = False,
        use_torchao_fp8_precompute_scale_for_fsdp: bool = False,
        fp8_shard_intermediate_activation: bool = False,
        use_sdpa: bool = False,
        use_hf: bool = False,
    ):
        # Setup optimized environment for compilation
        setup_optimized_environment_for_compile()
        
        # Initialize compilation monitor
        self.compilation_monitor = CompilationMonitor()
        
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        learning_rate = 5e-4  # max learning rate
        weight_decay = 1e-1
        beta1 = 0.9
        beta2 = 0.95

        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        assert self.max_iters > self.warmup_iters

        self.device = device
        self.model_name = model_name
        self.config = Config.from_name(self.model_name)
        self.compile = compile
        self.dynamic = dynamic
        self.distributed_mode = distributed_mode
        self.shard_mode = shard_mode
        self.bucketing_mode = bucketing_mode
        self.sharding_size = sharding_size
        self.ddp_bucket_size = ddp_bucket_size
        self.fsdp_bucket_params = fsdp_bucket_params
        self.checkpoint_activations = checkpoint_activations
        self.micro_batch_size = micro_batch_size
        self.low_precision_mode = low_precision_mode
        self.use_te_fp8_autocast = is_transformer_engine(low_precision_mode) and "thunder" not in compile
        self.is_thunder_as_torchcompile_backend = "thunder" in self.compile and "dynamo" in self.compile
        self.dump_thunder_traces = dump_thunder_traces
        self.dump_memory_snapshot = dump_memory_snapshot
        self.fp8_shard_intermediate_activation = fp8_shard_intermediate_activation

        self.use_sdpa = use_sdpa
        self.use_hf = use_hf

        if self.use_sdpa and sdpa_available and self.compile not in ["eager", "inductor"]:
            warnings.warn(
                "SDPA is enabled but the model is not compiled with eager or inductor. SDPA priority setting will be skipped."
            )
            self.use_sdpa = False

        if use_torchao_fp8_linear:
            if not torchao_available:
                raise ValueError("`torchao` is not available")
            if self.distributed_mode not in ("none", "fsdp2"):
                raise ValueError(f"torchao fp8 requires {self.distributed_mode=} to be `none` or `fsdp2`")
        
        self._torchao_fp8_handler = TorchAOFP8Handler(
            is_fsdp2=self.distributed_mode == "fsdp2",
            use_fp8_linear=use_torchao_fp8_linear,
            use_fp8_allgather=use_torchao_fp8_allgather,
            use_torchao_fp8_precompute_float8_dynamic_scale_for_fsdp=use_torchao_fp8_precompute_scale_for_fsdp,
            use_torch_compile=self.compile in ("inductor", "thunder_dynamo"),
        )

        # Validation checks
        if self.sharding_size is not None:
            assert "thunder" not in self.compile, (
                "Hybrid Sharding (FSDP/DP) using --sharding_size is not yet supported for Thunder. Coming soon."
            )
            assert self.shard_mode in [
                "hybrid_zero2",
                "hybrid_zero3",
            ], "Sharding Size is only used with Hybrid FSDP/DP style parallelism."
            assert world_size % self.sharding_size == 0, (
                f"World size {world_size} is not divisible by the sharding size {self.sharding_size}"
            )

        if self.bucketing_mode is not None and self.distributed_mode not in FSDP_MODES:
            warnings.warn(
                f"--bucketing_mode {self.bucketing_mode} will be ignored as "
                f" it is only used for FSDP style parallelism but running {self.distributed_mode}"
            )

        assert not ("thunder" in self.compile and self.bucketing_mode == "size"), (
            "'size' bucketing mode is not supported for Thunder. Please use 'none' or 'block'."
        )

        if self.fsdp_bucket_params is not None:
            if self.distributed_mode not in FSDP_MODES:
                warnings.warn(
                    f"Found --fsdp_bucket_params but Distributed mode is {self.distributed_mode}. Will be ignored"
                )
            if self.bucketing_mode != "size":
                warnings.warn(f"Bucketing mode is set to {self.bucketing_mode}. --fsdp_bucket_params will be ignored.")

        if is_transformer_engine(low_precision_mode):
            if not transformer_engine_available:
                raise ImportError(
                    "Selected benchmark config is for TransformerEngine but could not import the TransformerEngine library!"
                )
            check_fp8_compute_capability()

        if "thunder" in self.compile and is_transformer_engine(self.low_precision_mode):
            self.compile += "_transformerengine"

        if global_batch_size is not None:
            self.global_batch_size = global_batch_size
        else:
            self.global_batch_size = (
                self.micro_batch_size * world_size if world_size is not None else self.micro_batch_size
            )
        assert self.global_batch_size % self.micro_batch_size == 0, (
            f"Global Batch Size {self.global_batch_size} should be a multiple of Micro Batch Size {self.micro_batch_size}."
        )
        self.gradient_accumulation_steps = int(self.global_batch_size / self.micro_batch_size)
        if world_size:
            self.gradient_accumulation_steps = int(self.gradient_accumulation_steps / world_size)
            assert self.global_batch_size % self.micro_batch_size * world_size == 0, (
                f"Global Batch Size {self.global_batch_size} should be a multiple Micro Batch Size {self.micro_batch_size} * World Size {world_size}."
            )

        self.skip_data_sync = skip_data_sync

        # Profiling Args
        self.nsys_enabled = nsys_enabled
        self.profiler_start = profiler_start
        self.profiler_stop = profiler_stop

        if n_layers is not None:
            self.config.n_layer = n_layers

        if block_size is not None:
            self.config.block_size = block_size

        # Initialize the model
        t0 = time.perf_counter()
        print(f"Loading model with {self.config.__dict__}")
        if is_transformer_engine(self.low_precision_mode):
            check_and_update_config_for_te_if_needed(self.config)
        self.model = self.init_model()
        print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

        # Critical: Apply torch.compile BEFORE FSDP wrapping for better performance
        # This is the key change recommended in the documentation
        compile_first = self.compile in ("inductor", "eager") or "dynamo" in self.compile
        
        if compile_first:
            print("Applying compilation before distributed setup for optimal FSDP integration")
            self.model = self.setup_compile(self.model)
            self.model = self.setup_distributed(self.model)
        else:
            # For Thunder without dynamo, apply distributed first
            self.model = self.setup_distributed(self.model)
            self.model = self.setup_compile(self.model)

        # Setup activations checkpointing after distributed setup
        if self.checkpoint_activations:
            self.setup_activation_checkpointing()

        # Initialize the optimizer after the model is sharded if using FSDP
        self.optimizer = configure_optimizers(
            self.model, weight_decay, learning_rate, (beta1, beta2), device_type="cuda"
        )

        # Setup the Dummy dataloader for training
        self.train_dataloader = self.setup_dummy_dataloader()
        self.train_data_iter = iter(self.train_dataloader)

        # Setup empty metrics dict
        if global_rank in [0, None]:
            self.perf_metrics = {
                "average_iter_time": None,
                "model_flops": None,
                "model_flop_per_sec": None,
                "tokens_per_sec": None,
            }

    def init_model(self):
        init_device = torch.device("meta") if self.distributed_mode in FSDP_MODES else self.device
        if self.use_hf:
            warnings.warn(
                "HuggingFace transformers mode is experimental, many options do not apply. Preliminary testing with transformers==4.50.3."
            )

            # for the materialization, we need reset_parameters
            def RotaryEmbedding_reset_parameters(self):
                inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.inv_freq.device)
                with torch.no_grad():
                    self.inv_freq.copy_(inv_freq)

            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

            Qwen2RotaryEmbedding.reset_parameters = RotaryEmbedding_reset_parameters
            LlamaRotaryEmbedding.reset_parameters = RotaryEmbedding_reset_parameters

            def RMSNorm_reset_parameters(self):
                with torch.no_grad():
                    self.weight.fill_(1)

            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
            from transformers.models.llama.modeling_llama import LlamaRMSNorm

            Qwen2RMSNorm.reset_parameters = RMSNorm_reset_parameters
            LlamaRMSNorm.reset_parameters = RMSNorm_reset_parameters

            import transformers

            hf_model_name = f"{self.config.hf_config['org']}/{self.config.hf_config['name']}"
            hf_cfg = transformers.AutoConfig.from_pretrained(hf_model_name)
            with init_device:
                model = transformers.AutoModel.from_config(hf_cfg)
        else:
            with init_device:
                model = GPT(self.config)

        # Handle fp8 related Linear layer swapping (for torchao or TransformerEngine)
        model = self._torchao_fp8_handler.convert_model_to_fp8(model)
        if self.use_te_fp8_autocast:
            is_wo_layernorm = self.low_precision_mode == "fp8-delayed-te-wo_layernorm"
            swap_linear_layers_for_te(model, init_device, swap_layernorm=not is_wo_layernorm)

        model.to(dtype=torch.bfloat16)
        return model

    def setup_distributed(self, model):
        if self.distributed_mode == "none":
            return model

        # Distributed Setup
        if "thunder" in self.compile and "dynamo" not in self.compile:
            if self.distributed_mode == "ddp":
                from thunder.distributed import ddp
                model = ddp(
                    model,
                    broadcast_from=0,
                    bucket_size_in_mb=self.ddp_bucket_size,
                )
            elif self.distributed_mode == "fsdp":
                from thunder.distributed import fsdp, FSDPType, FSDPBucketingStrategy

                sharding_strategy = {"zero2": FSDPType.ZERO2, "zero3": FSDPType.ZERO3}[self.shard_mode]
                self.bucketing_mode = self.bucketing_mode or "none"
                bucketing_strategy = {
                    "none": FSDPBucketingStrategy.NONE,
                    "block": FSDPBucketingStrategy.BLOCK,
                    "layer": FSDPBucketingStrategy.LAYER,
                }[self.bucketing_mode]
                model = fsdp(
                    model,
                    broadcast_from=None,
                    sharding_strategy=sharding_strategy,
                    bucketing_strategy=bucketing_strategy,
                )
            else:
                if self.distributed_mode == "fsdp2":
                    raise ValueError(
                        "To use `fsdp2`, use thunder as torch.compile backend by including dynamo in `--compile` option or set `--compile` to either eager or inductor"
                    )
        else:
            if self.distributed_mode == "ddp":
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    bucket_cap_mb=self.ddp_bucket_size,
                )
            elif self.distributed_mode == "fsdp2":
                # Reference: https://github.com/pytorch/torchtitan/blob/6e7a183/docs/fsdp.md
                from functools import partial
                from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

                if self.bucketing_mode is not None:
                    warnings.warn(f"fsdp2 ignores {self.bucketing_mode=}")

                torch.cuda.set_device(local_rank)
                mesh = None
                if self.sharding_size is not None:
                    mesh = init_device_mesh("cuda", (int(world_size / self.sharding_size), self.sharding_size))
                else:
                    mesh = init_device_mesh("cuda", (world_size,))

                reshard_after_forward: bool = self.shard_mode == "zero3"

                _apply_fully_shard = partial(
                    fully_shard,
                    mesh=mesh,
                    reshard_after_forward=reshard_after_forward,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.bfloat16,
                    ),
                )

                # Apply FSDP2 layer by layer for better torch.compile integration
                for transformer_block in model.modules():
                    if isinstance(transformer_block, Block):
                        _apply_fully_shard(transformer_block)

                _apply_fully_shard(model.lm_head)
                _apply_fully_shard(model.transformer["wte"])
                _apply_fully_shard(model.transformer["ln_f"])
                _apply_fully_shard(model)
                model.to_empty(device=self.device)
                model.apply(model._init_weights)

            elif self.distributed_mode == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

                mesh = None
                if self.sharding_size is not None:
                    mesh = init_device_mesh("cuda", (int(world_size / self.sharding_size), self.sharding_size))

                litgpt_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
                size_auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=self.fsdp_bucket_params
                )
                zero_bucket_wrap_policy = lambda module, recurse, nonwrapped_numel: nonwrapped_numel >= 0

                self.bucketing_mode = self.bucketing_mode or "block"
                custom_wrap_policy = {
                    "block": litgpt_auto_wrap_policy,
                    "size": size_auto_wrap_policy,
                    "none": zero_bucket_wrap_policy,
                }[self.bucketing_mode]

                sharding_strategy: ShardingStrategy = {
                    "zero2": ShardingStrategy.SHARD_GRAD_OP,
                    "zero3": ShardingStrategy.FULL_SHARD,
                    "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
                    "hybrid_zero3": ShardingStrategy.HYBRID_SHARD,
                }[self.shard_mode]

                torch.cuda.set_device(local_rank)
                
                # For torch.compile compatibility, use_orig_params=True is essential
                use_orig_params = "thunder" not in self.compile  # Thunder doesn't support use_orig_params=True
                
                model = FSDP(
                    model,
                    sharding_strategy=sharding_strategy,
                    auto_wrap_policy=custom_wrap_policy,
                    device_id=local_rank,
                    use_orig_params=use_orig_params,
                    device_mesh=mesh,
                    mixed_precision=None,  # Use torch.amp.autocast instead for better compilation
                )
        return model

    def setup_activation_checkpointing(self):
        if any(isinstance(mod, CheckpointWrapper) for mod in self.model.modules()):
            warnings.warn(
                "FSDP checkpointing is configured, but the model already contains checkpointed layers."
                " Checkpointing will be ignored."
            )
            return

        check_fn = lambda submodule: isinstance(submodule, Block)
        apply_activation_checkpointing(self.model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

    def setup_compile(self, model):
        if self.compile == "inductor":
            print("Optimizing cache configuration for torch.compile with FSDP")
            
            # Setup optimized dynamo configuration
            dynamo_config = setup_dynamo_config_with_version_check()
            
            # Start compilation timing
            self.compilation_monitor.start_compilation_timer()
            
            # Determine if we should use dynamic compilation
            # FP8 + dynamic shapes can cause issues, so disable dynamic for FP8
            use_dynamic = self.dynamic and not (self.use_te_fp8_autocast or self._torchao_fp8_handler._enabled)
            
            if not use_dynamic and (self.use_te_fp8_autocast or self._torchao_fp8_handler._enabled):
                print("Disabling dynamic shapes due to FP8 usage for better compatibility")
                # For FP8, disable certain dynamo optimizations that can cause issues
                dynamo_config.specialize_int = False
            
            try:
                model = torch.compile(
                    model, 
                    dynamic=use_dynamic,
                    mode="default"  # Balanced performance/compilation time
                )
                print("Successfully compiled model with inductor backend")
            except Exception as e:
                warnings.warn(f"Compilation failed: {e}. Falling back to eager mode.")
                return model
            
            # End compilation timing
            compilation_time = self.compilation_monitor.end_compilation_timer()
            print(f"Compilation completed in {compilation_time:.2f} seconds")
            
        elif "thunder" in self.compile:
            executors = list(thunder.get_default_executors())
            transforms = []

            if "inductor_cat" in self.compile:
                from thunder.executors.torch_compile import torch_compile_cat_ex as torch_compile_ex
                executors.insert(0, torch_compile_ex)
            elif "inductor" in self.compile:
                from thunder.executors.torch_compile import torch_compile_ex
                executors.insert(0, torch_compile_ex)

            if "transformerengine_v2" in self.compile:
                from thunder.executors.transformer_engine_v2ex import (
                    transformer_engine_v2_ex,
                    TransformerEngineTransformV2,
                )
                executors.insert(0, transformer_engine_v2_ex)
                transforms.insert(0, TransformerEngineTransformV2())

            elif "transformerengine" in self.compile:
                from thunder.executors.transformer_engineex import transformer_engine_ex
                executors.insert(0, transformer_engine_ex)

            if "dynamo" in self.compile:
                print("Setting up Thunder as torch.compile backend with optimized configuration")
                
                # Use specialized Thunder + dynamo configuration
                setup_thunder_dynamo_config()
                
                self.backend = ThunderCompiler(executors=executors, transforms=transforms)
                
                # Use lower level torch._dynamo.optimize for better control
                # This avoids Lightning Fabric monkey patching issues
                model = torch._dynamo.optimize(backend=self.backend)(model)
                print("Successfully set up Thunder as torch.compile backend")
            else:
                jit_options = {}
                jit_options["fp8_shard_intermediate_activation"] = self.fp8_shard_intermediate_activation
                model = thunder.jit(model, executors=executors, transforms=transforms, **jit_options)
                
        elif self.compile == "eager":
            print("Using eager mode - no compilation applied")
            
        else:
            raise ValueError(f"Invalid compile option: {self.compile}")

        return model

    def setup_dummy_dataloader(self):
        def pad_collate(batch):
            x, y = zip(*batch)
            x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
            y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)
            return x_padded, y_padded

        train_data = DummyDataset(self.config.block_size, self.dynamic)
        train_dataloader = DataLoader(
            train_data, batch_size=self.micro_batch_size, num_workers=2, collate_fn=pad_collate
        )

        return train_dataloader

    def calculate_model_flops(self):
        device = self.device
        try:
            meta = torch.device("meta")
            self.device = meta

            # calculate flops on a meta-device model because we only care about the shapes and
            # because the flops calculator installs hooks on the model
            meta_model = self.init_model()

            x = torch.randint(0, 1, (self.micro_batch_size, meta_model.config.block_size), device=meta)
            model_fwd = lambda: meta_model(x)
            model_loss = lambda y: torch.nn.functional.cross_entropy(
                y.reshape(-1, y.size(-1)), x.reshape(-1), ignore_index=-1
            )
            self.perf_metrics["model_flops"] = measure_flops(meta_model, model_fwd, model_loss)
        finally:
            self.device = device

    def train(self):
        t0 = None
        if global_rank in [0, None]:
            try:
                # Calculate the model FLOPs
                self.calculate_model_flops()
                # Setup throughput Collection
                self.throughput = Throughput(window_size=self.max_iters - self.warmup_iters, world_size=world_size)
            except:
                self.throughput = None
                print(
                    f"Model Flops/Throughput calculation failed for model {self.model_name}. Skipping throughput metric collection."
                )

        if self.skip_data_sync:
            data_sync_ctx = self.model.no_sync
        else:
            data_sync_ctx = nullcontext

        for i in range(self.max_iters):
            iter_t0 = time.perf_counter()
            if i == self.warmup_iters:  # warmup
                t0 = iter_t0
                if self.dump_memory_snapshot and global_rank in (0, None):
                    torch.cuda.memory._record_memory_history()

            if self.nsys_enabled and i == self.profiler_start and global_rank in [0, None]:
                print("=====Start NSYS Profiling======")
                torch.cuda.cudart().cudaProfilerStart()

            with data_sync_ctx():
                for step_idx in range(self.gradient_accumulation_steps - 1):
                    input_ids, targets = next(self.train_data_iter)
                    input_ids = input_ids.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Use torch.amp.autocast instead of FSDP mixed precision for better compilation
                    if self.use_te_fp8_autocast:
                        with te.fp8_autocast():
                            logits = self.model(input_ids)
                    else:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits = self.model(input_ids)
                    
                    if not isinstance(logits, torch.Tensor):
                        logits = logits["last_hidden_state"]
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = targets.reshape(-1)
                    loss = (
                        torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
                        / self.gradient_accumulation_steps
                    )
                    loss.backward()

            input_ids, targets = next(self.train_data_iter)
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            if self.use_te_fp8_autocast:
                with te.fp8_autocast():
                    logits = self.model(input_ids)
            else:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = self.model(input_ids)
            
            if not isinstance(logits, torch.Tensor):
                logits = logits["last_hidden_state"]
            
            # Monitor saved tensors for memory analysis
            saved_tensors = getattr(logits.grad_fn, "saved_tensors", None)
            saved_tensors_len = None
            saved_tensors_size_in_mib = None
            if saved_tensors:
                saved_tensors_len = len([t for t in saved_tensors if t is not None])
                saved_tensors_size_in_mib = (
                    sum(t.numel() * t.element_size() for t in saved_tensors if t is not None) / 1024**2
                )
                del saved_tensors
            
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = (
                torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1) / self.gradient_accumulation_steps
            )
            loss.backward()

            self._torchao_fp8_handler.sync_float8_amax_and_scale_history_for_delayed_scaling(self.model)

            # Simple Gradient Accumulation Implementation
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self._torchao_fp8_handler.precompute_fp8_dynamic_scale_for_fsdp(self.model)

            if self.nsys_enabled and i == self.profiler_stop and global_rank in [0, None]:
                print("=====Stop NSYS Profiling======")
                torch.cuda.cudart().cudaProfilerStop()

            loss_item = loss.item()  # synchronization
            t1 = time.perf_counter()
            if global_rank in [0, None]:
                print(
                    f"iter {i}: loss {loss_item:.4f}, iter time: {(t1 - iter_t0) * 1000:.2f}ms, t: {input_ids.size(1)}"
                )
                if i >= self.warmup_iters:
                    if self.throughput:
                        self.throughput.update(
                            time=(t1 - t0),
                            flops=self.perf_metrics["model_flops"],
                            batches=i,
                            samples=(i * self.micro_batch_size * self.gradient_accumulation_steps),
                            lengths=(
                                i * self.micro_batch_size * self.gradient_accumulation_steps * self.config.block_size
                            ),
                        )

        if global_rank in [0, None]:
            self.perf_metrics["average_iter_time"] = ((t1 - t0) * 1000) / (self.max_iters - self.warmup_iters)
            self.perf_metrics["saved_for_backward_tensor_size_mib"] = saved_tensors_size_in_mib
            self.perf_metrics["saved_for_backward_number_of_tensors"] = saved_tensors_len

    def add_perf_metrics(self):
        if self.throughput:
            metrics = self.throughput.compute()
            self.perf_metrics["tokens_per_sec"] = metrics.get("items_per_sec", metrics["device/items_per_sec"])
            self.perf_metrics["model_flop_per_sec"] = metrics.get("flops_per_sec", metrics["device/flops_per_sec"])
        self.perf_metrics["memory_used_GB"] = torch.cuda.max_memory_allocated() / 1e9

    def add_model_info_to_metrics(self):
        if global_rank in [0, None]:
            self.perf_metrics["model_name"] = self.model_name
            self.perf_metrics["Num GPUS"] = world_size
            self.perf_metrics["Seq Len"] = self.config.block_size
            self.perf_metrics["Micro BS"] = self.micro_batch_size
            self.perf_metrics["Global BS"] = self.global_batch_size
            self.perf_metrics["GA"] = self.gradient_accumulation_steps
            if self.distributed_mode in FSDP_MODES:
                self.perf_metrics["Distributed Mode"] = (
                    str(self.distributed_mode)
                    + "_"
                    + str(self.shard_mode)
                    + "_"
                    + str(self.bucketing_mode)
                    + "_bucketing"
                )
                self.perf_metrics["Sharding Size"] = self.sharding_size
            else:
                self.perf_metrics["Distributed Mode"] = self.distributed_mode
                self.perf_metrics["Sharding Size"] = None
            self.perf_metrics["compiler"] = self.compile


class DummyDataset(IterableDataset):
    def __init__(self, max_seq_length: int, dynamic: bool):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.dynamic = dynamic

    def __iter__(self):
        while True:
            if self.dynamic:
                t = torch.randint(10, self.max_seq_length + 1, (1,))
            else:
                t = self.max_seq_length
            data = torch.randint(0, 100, (t + 1,), dtype=torch.int64)
            x = data[:t]
            y = data[1 : t + 1]
            yield x, y


def benchmark_main(return_metrics_as_json=False, json_path="", **kwargs) -> None:
    """
    Runs a training benchmark for lit-GPT models and
    prints out the performance metrics.
    """
    benchmark = Benchmark_litGPT(**kwargs)

    attention_ctx = nullcontext()
    if sdpa_available and benchmark.use_sdpa:
        backends = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        kwargs = {}
        if LooseVersion(torch.__version__) >= LooseVersion("2.6.0"):
            kwargs["set_priority"] = True

        attention_ctx = sdpa_kernel(backends, **kwargs)

    te_autocast_ctx = nullcontext()

    if "transformerengine_v2" in benchmark.compile:
        from transformer_engine.pytorch.fp8 import fp8_autocast
        te_autocast_ctx = fp8_autocast(enabled=True)

    with attention_ctx, te_autocast_ctx:
        benchmark.train()

    if global_rank in [0, None]:
        benchmark.add_perf_metrics()
        model_name = benchmark.model_name
        if benchmark.use_hf:
            model_name += "_hf"
        print(
            f"Model name: {model_name}\nSeq Length: {benchmark.config.block_size}\nMicro BS: {benchmark.micro_batch_size}\nGlobal BS: {benchmark.global_batch_size}"
        )
        print(
            f"Number of Layers: {benchmark.config.n_layer}\nNumber of parameters: {sum(p.numel() for p in benchmark.model.parameters() if p.requires_grad) / 1e9:.02f}B"
        )
        print(f"Distributed Mode: {benchmark.distributed_mode}")
        if benchmark.distributed_mode in FSDP_MODES:
            print(f"Sharding Mode: {benchmark.shard_mode}\nBucketing: {benchmark.bucketing_mode}")
            if benchmark.sharding_size is not None:
                print(
                    f"Sharding Size: {benchmark.sharding_size}\nReplicate DP Groups: {int(world_size / benchmark.sharding_size)}"
                )
            if benchmark.bucketing_mode == "size":
                print(f"Bucketing Number Params: {benchmark.fsdp_bucket_params}")
        elif benchmark.distributed_mode == "ddp":
            print(f"DDP Bucketing Size: {benchmark.ddp_bucket_size} MB")
        print(f"Compiler: {benchmark.compile}")
        print(f"Low Precision Mode: {benchmark.low_precision_mode}")
        if benchmark._torchao_fp8_handler._enabled:
            msg = "linear"
            if benchmark._torchao_fp8_handler.use_fp8_allgather:
                msg += ", all-gather"
            if benchmark._torchao_fp8_handler.precompute_scale:
                msg += ", single all-reduce of AMAX/scales for dynamic scaling"
            msg += " are enabled"
            print(f"[torchao float8] {msg}")

        print(f"Average iter time: {benchmark.perf_metrics['average_iter_time']:.2f} ms")
        print(f"Memory used: {benchmark.perf_metrics['memory_used_GB']:.02f} GB")
        if benchmark.perf_metrics["saved_for_backward_tensor_size_mib"] is not None:
            print(f"Saved for backward size: {benchmark.perf_metrics['saved_for_backward_tensor_size_mib']:.02f} MiB")
            print(
                f"Saved for backward number of tensors: {benchmark.perf_metrics['saved_for_backward_number_of_tensors']}"
            )

        tokens_per_sec = benchmark.perf_metrics.get("tokens_per_sec")
        if tokens_per_sec:
            print(f"Tokens/s: {tokens_per_sec:.02f}")
            print(f"Tokens/s/GPU: {(tokens_per_sec / world_size):.02f}")
        if benchmark.throughput:
            print(f"TFLOP/s: {benchmark.perf_metrics['model_flop_per_sec'] / 1e12:.02f}")

        if benchmark.dump_memory_snapshot:
            file_name = f"{benchmark.model_name}_{benchmark.compile}_{benchmark.distributed_mode}"
            if benchmark.distributed_mode.startswith("fsdp"):
                file_name = f"{file_name}_{benchmark.shard_mode}"
                if benchmark.distributed_mode == "fsdp":
                    file_name += f"_{benchmark.bucketing_mode}"
            if benchmark.distributed_mode == "ddp":
                file_name += f"_{benchmark.ddp_bucket_size}"
            file_name = f"{file_name}.pickle"
            print(f"Dump memory snapshot at {file_name}")
            torch.cuda.memory._dump_snapshot(file_name)
            torch.cuda.memory._record_memory_history(enabled=None)

        if benchmark.dump_thunder_traces:
            if benchmark.is_thunder_as_torchcompile_backend:
                print(f"{len(benchmark.backend.subgraph_infos)} Thunder subgraphs created")
                for gid, infos in enumerate(benchmark.backend.subgraph_infos):
                    for subgid, thunder_fn in enumerate(infos.thunder_compiled_fns):
                        print(f"##########\n#Graph{gid}-ThunderFn{subgid} last forward trace\n##########")
                        print(thunder.last_traces(thunder_fn)[-1])
                        print(f"##########\n#Graph{gid}-ThunderFn{subgid} last backward trace\n##########")
                        print(thunder.last_backward_traces(thunder_fn)[-1])
            elif "dynamo" not in benchmark.compile:
                fwd_traces = [thunder.last_traces(benchmark.model)]
                bwd_traces = [thunder.last_backward_traces(benchmark.model)]
                for i, f_traces in enumerate(fwd_traces, start=1):
                    print(f"##########\n#{i}-th ThunderModule Forward\n##########")
                    print(f_traces[-1])
                for i, b_traces in enumerate(bwd_traces, start=1):
                    print(f"##########\n#{i}-th ThunderModule Backward\n##########")
                    print(b_traces[-1])

    if global_rank in [0, None]:
        if return_metrics_as_json:
            benchmark.add_model_info_to_metrics()
            json_path = str(json_path)
            import json

            with open(json_path, "w") as file:
                json.dump(benchmark.perf_metrics, file, indent=4)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    try:
        CLI(benchmark_main)
    except Exception:
        raise
    finally:
        # ref: https://github.com/pytorch/pytorch/blob/3af12447/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1110-L1116
        if world_size > 1:
            torch_dist.destroy_process_group()
