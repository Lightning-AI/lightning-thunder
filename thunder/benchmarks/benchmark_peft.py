# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import time
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model
from torch.distributed import DeviceMesh, init_process_group
from torch.distributed._composable.fsdp import fully_shard
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Set up distributed training variables
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

# Initialize distributed environment
if WORLD_SIZE > 1:
    torch.cuda.set_device(LOCAL_RANK)

# With this setting, Dynamo Graphs inline all the modules
torch._dynamo.config.inline_inbuilt_nn_modules = True

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # Increase timeout to 60 seconds
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"  # Increase timeout to 60 seconds


# Configure loguru to only log on LOCAL_RANK 0
def rank_filter(record):
    return LOCAL_RANK == 0


logger.remove()  # Remove default handler
logger.add(lambda msg: print(msg) if LOCAL_RANK == 0 else None, filter=rank_filter)


def collate_fn(batch):
    """Custom collate function to handle tensor batching."""
    input_ids = torch.stack([torch.tensor(item["input_ids"], dtype=torch.long) for item in batch])
    labels = torch.stack([torch.tensor(item["labels"], dtype=torch.long) for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def make_dummy_dataset(
    tokenizer, seq_len: int, mbs: int, gbs: int, n: int = 100, seq_lengths: list[int] | None = None
) -> Dataset:
    """Create a dummy dataset for training."""
    data = {
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nThe sky is usually clear above the desert and the sunshine duration is extremely high everywhere in the Sahara. Most of the desert enjoys more than 3,600 h of bright sunshine annually or over 82% of the time and a wide area in the eastern part experiences in excess of 4,000 h of bright sunshine a year or over 91% of the time, and the highest values are very close to the theoretical maximum value. A value of 4,300 h or 98% of the time would be recorded in Upper Egypt (Aswan, Luxor) and in the Nubian Desert (Wadi Halfa). The annual average direct solar irradiation is around 2,800 kWh/(m2 year) in the Great Desert. The Sahara has a huge potential for solar energy production. The constantly high position of the sun, the extremely low relative humidity, the lack of vegetation and rainfall make the Great Desert the hottest continuously large area worldwide and certainly the hottest place on Earth during summertime in some spots. The average high temperature exceeds 38 °C (100.4 °F) - 40 °C (104 °F) during the hottest month nearly everywhere in the desert except at very high mountainous areas. The highest officially recorded average high temperature was 47 °C (116.6 °F) in a remote desert town in the Algerian Desert called Bou Bernous with an elevation of 378 meters above sea level. It's the world's highest recorded average high temperature and only Death Valley, California rivals it. Other hot spots in Algeria such as Adrar, Timimoun, In Salah, Ouallene, Aoulef, Reggane with an elevation between 200 and 400 meters above sea level get slightly lower summer average highs around 46 °C (114.8 °F) during the hottest months of the year. Salah, well known in Algeria for its extreme heat, has an average high temperature of 43.8 °C (110.8 °F), 46.4 °C (115.5 °F), 45.5 (113.9 °F). Furthermore, 41.9 °C (107.4 °F) in June, July, August and September. In fact, there are even hotter spots in the Sahara, but they are located in extremely remote areas, especially in the Azalai, lying in northern Mali. The major part of the desert experiences around 3 – 5 months when the average high strictly exceeds 40 °C (104 °F). The southern central part of the desert experiences up to 6 – 7 months when the average high temperature strictly exceeds 40 °C (104 °F) which shows the constancy and the length of the really hot season in the Sahara. Some examples of this are Bilma, Niger and Faya-Largeau, Chad. The annual average daily temperature exceeds 20 °C (68 °F) everywhere and can approach 30 °C (86 °F) in the hottest regions year-round. However, most of the desert has a value in excess of 25 °C (77 °F). The sand and ground temperatures are even more extreme. During daytime, the sand temperature is extremely high as it can easily reach 80 °C (176 °F) or more. A sand temperature of 83.5 °C (182.3 °F) has been recorded in Port Sudan. Ground temperatures of 72 °C (161.6 °F) have been recorded in the Adrar of Mauritania and a value of 75 °C (167 °F) has been measured in Borkou, northern Chad. Due to lack of cloud cover and very low humidity, the desert usually features high diurnal temperature variations between days and nights. However, it's a myth that the nights are cold after extremely hot days in the Sahara. The average diurnal temperature range is typically between 13 °C (55.4 °F) and 20 °C (68 °F). The lowest values are found along the coastal regions due to high humidity and are often even lower than 10 °C (50 °F), while the highest values are found in inland desert areas where the humidity is the lowest, mainly in the southern Sahara. Still, it's true that winter nights can be cold as it can drop to the freezing point and even below, especially in high-elevation areas.\n\n### Input:\nWhat percent of time is the sun generally over most of the desert?\n\n### Response:\n82% of the time"
        * 50
    }

    def fmt(example):
        ans = tokenizer(example["text"], truncation=True, max_length=seq_len)
        tokens = ans["input_ids"]
        return {
            "input_ids": tokens,
            "labels": tokens[1:] + [tokens[-1]],
        }

    class CachedFormatterWithVariableSequence:
        def __init__(self, fn, seq_lengths=None):
            self.cache = None
            self.fn = fn
            self.seq_lengths = seq_lengths

        def __call__(self, *args, **kwargs):
            if self.cache is None:
                self.cache = self.fn(*args, **kwargs)
            if self.seq_lengths is None:
                return self.cache
            else:
                seq_len = self.seq_lengths.pop(0)
                return {k: v[:seq_len] for k, v in self.cache.items()}

    dataset = Dataset.from_dict({"text": [data["text"] for _ in range(n)]})
    dataset = dataset.map(
        CachedFormatterWithVariableSequence(fmt, seq_lengths),
        batched=False,
        batch_size=1,
        remove_columns=["text"],
    )
    return dataset


def load_model(
    model_name: str,
    attn_implementation: str = "sdpa",
    fixed_num_hidden_layers: int | None = None,
    trust_remote_code: bool = False,
    verbose: bool = False,
) -> torch.nn.Module:
    """Load and configure a HuggingFace model."""
    if verbose:
        logger.info("Starting model configuration")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if verbose:
        logger.info(f"Config loaded: {config}")

    if fixed_num_hidden_layers is not None:
        if verbose:
            logger.info(f"Setting num_hidden_layers to {fixed_num_hidden_layers}")
        config.num_hidden_layers = fixed_num_hidden_layers

    try:
        if verbose:
            logger.info(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            logger.info("Creating model on meta device")

        with torch.device("meta"):
            try:
                model = AutoModelForCausalLM.from_config(
                    config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation,
                )
            except ValueError as e:
                logger.warning(f"Could not create model as CausalLM, trying AutoModel: {str(e)}")
                try:
                    model = transformers.AutoModel.from_config(
                        config,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=trust_remote_code,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not create model as AutoModel, trying LlavaForConditionalGeneration: {str(e)}"
                    )
                    from transformers import LlavaForConditionalGeneration

                    model = LlavaForConditionalGeneration._from_config(
                        config,
                        torch_dtype=torch.bfloat16,
                    )

        if model is None:
            raise RuntimeError("Failed to create model on meta device")

        if verbose:
            logger.info("Model created on meta device")

        return model

    except Exception as e:
        logger.error(f"Error in model configuration: {str(e)}")
        logger.error(f"GPU memory at error: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        raise


def setup_lora(model: torch.nn.Module, verbose: bool = False) -> torch.nn.Module:
    """Apply LoRA to the model."""
    if verbose:
        logger.info("Applying LoRA to model")

    lora_config = LoraConfig(
        r=16,  # rank
        target_modules="all-linear",  # See: https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.target_modules
        lora_alpha=32,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Freeze all parameters except LoRA
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True
            if verbose:
                logger.info(f"Setting gradient for LoRA parameter: {name}")

    if verbose:
        logger.info("LoRA applied to model")

    return model


def setup_fsdp2(model: torch.nn.Module, devices: int, verbose: bool = False) -> torch.nn.Module:
    """Apply FSDP2 to the model with ZeRO-3 style sharding."""
    if devices <= 1:
        return model

    if verbose:
        logger.info("Applying FSDP2 to model with ZeRO-3 style sharding")

    mesh = DeviceMesh("cuda", torch.arange(devices), mesh_dim_names=("dp",))

    # Configure model for static shapes before FSDP2
    if hasattr(model, "config"):
        model.config.use_cache = True
        model.config.max_position_embeddings = args.seq_length
        if verbose:
            logger.info(f"Configured model for static shapes with sequence length: {args.seq_length}")

    # Apply FSDP2 with ZeRO-3 style sharding
    # First wrap individual layers
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            if verbose:
                logger.info(f"Wrapping layer {name} with FSDP2")
            fully_shard(module, mesh=mesh)

    # Then wrap the entire model
    model = fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
    )
    if verbose:
        logger.info("FSDP2 applied to model")

    return model


def setup_compilation(model: torch.nn.Module, backend: str, verbose: bool = False) -> torch.nn.Module:
    """Apply compilation settings to the model."""
    if backend == "torchjit":
        logger.info("Compiling model with torch.compile")
        dist_print("Resetting cache size for torch.compile")
        import torch._dynamo.config as dynamo_config

        # Fixes recompilation issues with inductor
        dynamo_config.cache_size_limit = 64
        model = torch.compile(model)
    elif backend == "thunder":
        import thunder
        import thunder.dynamo
        import torch._dynamo.config as dynamo_config
        from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform
        from thunder.executors.transformer_engineex import transformer_engine_ex

        # Fixes recompilation issues with Thunder
        dynamo_config.cache_size_limit = 64
        # Disable gradient checkpointing for Thunder
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_disable()
            logger.info("Disabled gradient checkpointing for Thunder compilation")

        # Set static shapes for key-value cache
        if hasattr(model, "config"):
            model.config.use_cache = True
            model.config.max_position_embeddings = args.seq_length
            logger.info(f"Set static cache size to sequence length: {args.seq_length}")

        executors = thunder.get_default_executors()
        xforms: list = [NvtxProfileTransform()]
        logger.info(f"Thunder used executors: {executors}")
        logger.info(f"Applying Thunder compilation with {len(executors)} executors")
        be = thunder.dynamo.ThunderCompiler(transforms=xforms, executors=executors)
        model = torch._dynamo.optimize(backend=be)(model)

    return model


def dist_print(*args, **kwargs):
    """Print only on LOCAL_RANK 0."""
    if LOCAL_RANK == 0:
        print(*args, **kwargs)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--strategy", type=str, default="auto", choices=["auto", "ddp", "fsdp2"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--skip-iters", type=int, default=2, help="Number of warmup iterations to skip in average calculation"
    )
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before performing a backward pass.",
    )
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument(
        "--var-seq-length",
        action="store_true",
        help="If true will use different sequence length for each batch; In this case --seq-length will be used as maximum sequence length allowed.",
    )
    parser.add_argument(
        "--jit-backend",
        default="eager",
        type=str.lower,
        choices=["eager", "torchjit", "thunder"],
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output including model wrapping details")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--fixed-num-hidden-layers",
        type=int,
        default=None,
        help="Number of hidden layers to fix the config to.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Attention implementation to use",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing. Disabled by default due to potential compatibility issues with compilers.",
    )

    args = parser.parse_args()

    # Validate skip-iters
    if args.skip_iters >= args.max_steps:
        raise ValueError(f"skip-iters ({args.skip_iters}) must be less than max-steps ({args.max_steps})")

    return args


def get_tokenizer(model_name: str, trust_remote_code: bool, fallback_model: str = "gpt2") -> Any:
    """Get tokenizer for the model."""
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except ValueError:
        print(f"Warning: Could not load tokenizer for {model_name}, using fallback tokenizer {fallback_model}")
        return AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=trust_remote_code)


@contextmanager
def sdpa_context():
    """Context manager for SDPA attention with optimal backend selection."""
    backends = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    kwargs = {}
    if LooseVersion(torch.__version__) >= LooseVersion("2.6.0"):
        kwargs["set_priority"] = True

    ctx = sdpa_kernel(backends, **kwargs)
    try:
        yield ctx
    finally:
        pass


def main(args: argparse.Namespace):
    """Main training function."""
    dist_print(args)

    # Initialize distributed environment
    if WORLD_SIZE > 1:
        init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)
        if args.verbose:
            logger.info(f"Initialized process group: rank {RANK}, local rank {LOCAL_RANK}, world size {WORLD_SIZE}")

    # Set memory efficient settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Calculate global batch size
    gbs = args.mbs * args.grad_acc_steps * WORLD_SIZE
    logger.info(
        f"Global batch size: {gbs} (mbs: {args.mbs}, grad_acc_steps: {args.grad_acc_steps}, world_size: {WORLD_SIZE})"
    )

    # Initialize batch tracking
    batches_processed = 0
    total_tokens_processed = 0

    seq_lengths = None
    if args.var_seq_length:
        random.seed(42)
        seq_lengths = [random.randint(10, args.seq_length) for _ in range(gbs * args.max_steps)]
        if args.verbose:
            dist_print("variable-seq-lengths= " + str(seq_lengths))

    start_ts = time.time()

    # Initialize metrics tracking
    iteration_times = []
    max_allocated_memory = 0
    total_tokens = 0
    t0 = None  # For tracking throughput after warmup

    logger.info("Loading tokenizer")
    tokenizer = get_tokenizer(args.model, args.trust_remote_code)

    # Load base model on meta device
    logger.info(f"Loading base model on meta device...")
    model = load_model(
        args.model,
        attn_implementation=args.attn_implementation,
        fixed_num_hidden_layers=args.fixed_num_hidden_layers,
        trust_remote_code=args.trust_remote_code,
        verbose=args.verbose,
    )
    logger.info(f"Base model loaded on meta device")

    # Configure model for static shapes
    if hasattr(model, "config"):
        model.config.use_cache = True
        model.config.max_position_embeddings = args.seq_length
        logger.info(f"Configured model for static shapes with sequence length: {args.seq_length}")

    # Materialize the model on CUDA
    model = model.to_empty(device=f"cuda:{LOCAL_RANK}")
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    # Handle gradient checkpointing based on user preference
    if hasattr(model, "gradient_checkpointing_enable"):
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled")
    else:
        logger.info("Model does not support gradient checkpointing")

    # Apply LoRA
    logger.info(f"Applying LoRA to model")
    model = setup_lora(model, verbose=args.verbose)
    logger.info(f"LoRA applied to model")

    # Ensure model is in training mode and verify gradients
    model.train()
    logger.info("Verifying gradient setup...")
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            if not param.requires_grad:
                if args.verbose:
                    logger.warning(f"LoRA parameter {name} does not require grad!")
            else:
                if args.verbose:
                    logger.info(f"LoRA parameter {name} requires grad")

    # Apply FSDP2 if needed
    if args.strategy == "fsdp2":
        logger.info(f"Applying FSDP2 to model with {args.devices} devices")
        model = setup_fsdp2(model, args.devices, verbose=args.verbose)
        logger.info(f"FSDP2 applied to model")

        # After FSDP2, verify and fix gradients
        logger.info("Verifying gradients after FSDP2...")
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                if not param.requires_grad:
                    if args.verbose:
                        logger.warning(f"LoRA parameter {name} lost grad requirement after FSDP2!")
                    param.requires_grad = True
                else:
                    if args.verbose:
                        logger.info(f"LoRA parameter {name} still requires grad")

    # Apply compilation if needed
    if args.jit_backend != "eager":
        logger.info(f"Applying compilation: {args.jit_backend} to model")
        model = setup_compilation(model, args.jit_backend, verbose=args.verbose)
        logger.info(f"Compilation applied to model")

    # Verify only LoRA parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameter ratio: {trainable_params/total_params*100:.2f}%")

    # Create optimizer with memory-efficient settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,  # Disable foreach to reduce memory usage
        fused=True,  # Use fused implementation if available
    )

    # Create dataset
    num_samples = gbs * args.max_steps
    dataset = make_dummy_dataset(
        tokenizer,
        args.seq_length,
        mbs=args.mbs,
        gbs=gbs,
        n=num_samples,
        seq_lengths=seq_lengths,
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.mbs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Training loop
    logger.info("Starting training...")

    # Create progress bar only on LOCAL_RANK 0
    pbar = tqdm(
        range(args.max_steps),
        desc="Training",
        disable=LOCAL_RANK != 0,
        unit="step",
        dynamic_ncols=True,
    )

    for step in pbar:
        iter_t0 = time.perf_counter()

        if step == args.skip_iters:  # warmup
            pass

        batch = next(iter(dataloader))

        # Track batches and tokens processed
        batches_processed += 1
        total_tokens_processed += batch["input_ids"].numel()

        # Move batch to GPU
        batch = {k: v.cuda() for k, v in batch.items()}

        # Track total tokens processed
        total_tokens += batch["input_ids"].numel()

        # Ensure sequence length is within model limits
        max_seq_len = model.config.max_position_embeddings
        seq_len = batch["input_ids"].size(1)  # Get sequence length after tensor conversion
        if seq_len > max_seq_len:
            logger.warning(f"Truncating sequence length from {seq_len} to {max_seq_len}")
            batch = {k: v[:, :max_seq_len] for k, v in batch.items()}

        # Forward pass with SDPA context
        with sdpa_context():
            outputs = model(**batch)
            loss = outputs.loss

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # More memory efficient

        # Track iteration time
        t1 = time.perf_counter()
        iteration_time = t1 - iter_t0
        iteration_times.append(iteration_time)

        # Track maximum memory usage
        current_memory = torch.cuda.max_memory_allocated()
        max_allocated_memory = max(max_allocated_memory, current_memory)

        # Update progress bar with current metrics
        if LOCAL_RANK == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.detach().item():.4f}",  # Use detach() to avoid warnings
                    "mem": f"{current_memory/1024**3:.1f}GB",
                    "iter_time": f"{iteration_time*1000:.1f}ms",
                }
            )

    # Close progress bar
    pbar.close()

    # Print training summary
    total_time = time.time() - start_ts
    print_training_summary(
        args,
        total_time,
        iteration_times,
        max_allocated_memory,
        total_tokens,
        gbs,
        batches_processed,
        total_tokens_processed,
        WORLD_SIZE,
    )

    # Clean up distributed environment if needed
    if WORLD_SIZE > 1:
        torch.distributed.destroy_process_group()


def print_training_summary(
    args: argparse.Namespace,
    total_time: float,
    iteration_times: list[float],
    max_allocated_memory: float,
    total_tokens: int,
    gbs: int,
    batches_processed: int,
    total_tokens_processed: int,
    WORLD_SIZE: int,
) -> None:
    """Print a comprehensive summary of the training run.

    This function prints various metrics and statistics about the training run,
    including model configuration, performance metrics, and verification results.
    """
    # Calculate average iteration time skipping warmup iterations
    if len(iteration_times) > args.skip_iters:
        avg_iteration_time = sum(iteration_times[args.skip_iters :]) / len(iteration_times[args.skip_iters :])
        # Calculate throughput after warmup using global batch size
        post_warmup_time = total_time - sum(iteration_times[: args.skip_iters])
        post_warmup_samples = (args.max_steps - args.skip_iters) * gbs
        post_warmup_tokens = post_warmup_samples * args.seq_length
        tokens_per_second = post_warmup_tokens / post_warmup_time
        samples_per_second = post_warmup_samples / post_warmup_time
    else:
        avg_iteration_time = sum(iteration_times) / len(iteration_times)
        tokens_per_second = total_tokens / total_time
        samples_per_second = args.max_steps * gbs / total_time

    dist_print("\nTraining Summary:")
    dist_print(f"Model: {args.model}")
    dist_print(f"Compiler: {args.jit_backend}")
    dist_print(f"Strategy: {args.strategy}")
    dist_print(f"Devices: {args.devices}")
    dist_print(f"Sequence length: {args.seq_length}")
    dist_print(f"Micro batch size: {args.mbs}")
    dist_print(f"Global batch size: {gbs}")
    dist_print(f"Gradient accumulation steps: {args.grad_acc_steps}")
    dist_print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    dist_print(f"Total training time: {total_time:.2f} seconds")
    dist_print(f"Average iteration time (after {args.skip_iters} warmup): {avg_iteration_time*1000:.2f} ms")
    dist_print(f"Throughput (after warmup): {tokens_per_second:.2f} tokens/second")
    dist_print(f"Throughput (after warmup): {samples_per_second:.2f} samples/second")
    dist_print(f"Maximum allocated memory: {max_allocated_memory/1024**3:.2f} GB")
    dist_print(f"Total tokens processed: {total_tokens:,}")
    dist_print(f"Total iterations: {args.max_steps}")

    # Verify batch processing across all ranks
    if WORLD_SIZE > 1:
        # Gather batch counts from all ranks
        batch_counts = torch.tensor([batches_processed], device="cuda")
        torch.distributed.all_reduce(batch_counts, op=torch.distributed.ReduceOp.SUM)
        total_batches_processed = batch_counts.item()

        # Gather token counts from all ranks
        token_counts = torch.tensor([total_tokens_processed], device="cuda")
        torch.distributed.all_reduce(token_counts, op=torch.distributed.ReduceOp.SUM)
        total_tokens_processed_all = token_counts.item()

        # Calculate expected values
        expected_batches = args.max_steps * WORLD_SIZE
        expected_tokens = expected_batches * args.seq_length * args.mbs

        # Log verification results
        dist_print(f"\nVerification:")
        dist_print(f"Total batches processed across all ranks: {total_batches_processed}")
        dist_print(f"Expected batches: {expected_batches}")
        dist_print(f"Total tokens processed across all ranks: {total_tokens_processed_all}")
        dist_print(f"Expected tokens: {expected_tokens}")
        dist_print(f"All GPUs working: {'✓' if total_batches_processed == expected_batches else '✗'}")

        # Assert that we processed the expected number of batches and tokens
        assert (
            total_batches_processed == expected_batches
        ), f"Expected {expected_batches} batches, but processed {total_batches_processed}"
        assert (
            total_tokens_processed_all == expected_tokens
        ), f"Expected {expected_tokens} tokens, but processed {total_tokens_processed_all}"
    else:
        # Single GPU verification
        expected_batches = args.max_steps
        expected_tokens = expected_batches * args.seq_length * args.mbs

        dist_print(f"\nVerification:")
        dist_print(f"Batches processed: {batches_processed}")
        dist_print(f"Expected batches: {expected_batches}")
        dist_print(f"Tokens processed: {total_tokens_processed}")
        dist_print(f"Expected tokens: {expected_tokens}")
        dist_print(f"Single GPU working: {'✓' if batches_processed == expected_batches else '✗'}")

        assert (
            batches_processed == expected_batches
        ), f"Expected {expected_batches} batches, but processed {batches_processed}"
        assert (
            total_tokens_processed == expected_tokens
        ), f"Expected {expected_tokens} tokens, but processed {total_tokens_processed}"


if __name__ == "__main__":
    args = parse_args()
    main(args)
