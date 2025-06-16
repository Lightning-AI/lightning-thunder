import importlib
import warnings
import os
from collections.abc import Callable
from enum import auto, Enum
from collections.abc import Sequence
from contextlib import nullcontext

import pytest
import torch

from lightning_utilities.core.imports import package_available

from litgpt.config import configs

import thunder

from thunder.benchmarks import (
    OptimBenchmark,
    BatchNormBenchmark,
    Benchmark,
    LitGPTBenchmark,
    LitGPTCausalSelfAttentionBenchmark,
    LitGPTSDPABenchmark,
    LitGPTSwigluBenchmark,
    LitGPTMLPBenchmark,
    NanoGPTBenchmark,
    NanoGPTCrossEntropyBenchmark,
    LitGPTGeluBenchmark,
    NanoGPTLayerNormBenchmark,
    ResNet50Benchmark,
    TorchbenchBenchmark,
    HFBenchmark,
    LinearLoRABenchmark,
    DeepSeekSGLangMoEBenchmark,
    thunder_apex_executor,
    thunder_apex_nvfuser_executor,
    thunder_cudnn_executor,
    thunder_cudnn_nvfuser_executor,
    thunder_executor,
    thunderfx_executor,
    thunder_sdpa_torch_compile_nvfuser_executor,
    torch_compile_executor,
    torch_executor,
    thunder_transformerengine_executor,
    record_peak_allocated_memory,
)
from thunder.core.interpreter import interpret

from thunder.tests.litgpt_model import Config as LitGPTConfig
from thunder.benchmarks.utils import backward_only

LIGER_FUSED_SWIGLU_AVAILABLE: bool = package_available("liger_kernel.ops.swiglu")
APEX_FUSED_ROPE_AVAILABLE: bool = package_available("fused_rotary_positional_embedding")
IMPORTANT_CONFIGS = [
    "Llama-2-13b-hf",
    "Llama-2-70b-hf",
    "Llama-2-7b-hf",
    "Llama-3-70B",
    "Llama-3-8B",
    "Mistral-7B-v0.1",
    "phi-2",
]
RUN_ALL_CONFIGS = os.environ.get("THUNDER_BENCH_RUN_ALL_CONFIGS", "0") == "1"


class ComputeType(Enum):
    INFERENCE = auto()
    TRAINING_FORWARD = auto()
    TRAINING_BACKWARD = auto()


def is_requires_grad(type: ComputeType):
    return type == ComputeType.TRAINING_FORWARD or type == ComputeType.TRAINING_BACKWARD


parametrize_compute_type = pytest.mark.parametrize(
    "compute_type,",
    (ComputeType.INFERENCE, ComputeType.TRAINING_FORWARD, ComputeType.TRAINING_BACKWARD),
    ids=("inference", "forward", "backward"),
)
parametrize_compute_type_only_training = pytest.mark.parametrize(
    "compute_type,",
    (ComputeType.TRAINING_FORWARD, ComputeType.TRAINING_BACKWARD),
    ids=("forward", "backward"),
)
parametrize_compute_type_only_inference = pytest.mark.parametrize(
    "compute_type,",
    (ComputeType.INFERENCE,),
    ids=("inference",),
)


def benchmark_for_compute_type(compute_type: ComputeType, benchmark, fn: Callable, args, kwargs, has_cuda: bool = True):
    context = record_peak_allocated_memory(benchmark) if has_cuda else nullcontext()
    with context:
        match compute_type:
            case ComputeType.INFERENCE | ComputeType.TRAINING_FORWARD:
                benchmark(fn, *args, **kwargs)
            case ComputeType.TRAINING_BACKWARD:
                backward_fn, backward_setup = backward_only(fn, *args, **kwargs)
                backward_args = backward_setup()
                benchmark(backward_fn, *backward_args)


def interpreter_fwd(module: Callable):
    fn_ = torch_executor(module)
    fn_ = interpret(fn_)
    return fn_


executors = (torch_executor, torch_compile_executor, thunder_executor)
executors_ids = (
    "torch",
    "torch.compile",
    "thunder",
)

torchbench_executors = (*executors, thunderfx_executor)
torchbench_executors_ids = (
    *executors_ids,
    "thunderfx",
)

apex_executors = (thunder_apex_executor, thunder_apex_nvfuser_executor)
apex_executors_ids = ("thunder+apex-grad", "thunder+apex+nvfuser-grad")

cudnn_executors = (thunder_cudnn_executor, thunder_cudnn_nvfuser_executor)
cudnn_executors_ids = ("thunder+cudnn", "thunder+cudnn+nvfuser")

cudnn_layernorm_executors = (thunder_cudnn_executor, thunder_cudnn_nvfuser_executor)
cudnn_layernorm_executors_ids = (
    "thunder+cudnn_layernorm",
    "thunder+cudnn_layernorm+nvfuser",
)

transformer_engine_executors = (thunder_transformerengine_executor,)
transformer_engine_execuors_ids = ("thunder+transformerengine",)


def get_unique_configs(config_options: Sequence[str]):
    """
    Get the unique configurations based on the given config options.

    Args:
        config_options: The sequence of configuration options that uniquely identify a LitGPT configuration.
    """
    config_names = list(sorted(c["name"] for c in configs)) if RUN_ALL_CONFIGS else IMPORTANT_CONFIGS
    unique_config_names = {}
    for config_name in config_names:
        config = LitGPTConfig.from_name(config_name)
        key = tuple(getattr(config, k) for k in config_options)
        if config_name in IMPORTANT_CONFIGS:
            unique_config_names[key] = config_name
        unique_config_names.setdefault(key, config_name)

    config_names = list(sorted(unique_config_names.values()))
    return config_names


# There are many configurations but only the following parameters affect the gelu benchmark:
# - gelu_approximate
# - intermediate_size
# - block_size
# Let's select only the configurations that differ in these parameters
def get_configs_for_gelu():
    return get_unique_configs(("gelu_approximate", "intermediate_size", "block_size"))


# Sample command to run this benchmark:
# pytest thunder/benchmarks/targets.py -k "test_litgpt_gelu" --benchmark-group-by='param:config,param:bs'
@pytest.mark.parametrize(
    "executor,",
    executors,
    ids=executors_ids,
)
@pytest.mark.parametrize(
    "bs,",
    (
        1,
        2,
    ),
    ids=("bs1", "bs2"),
)
@parametrize_compute_type
@pytest.mark.parametrize(
    "config,",
    get_configs_for_gelu(),
)
def test_litgpt_gelu(benchmark, executor: Callable, bs: int, compute_type: ComputeType, config: str):
    gelu_bench: Benchmark = LitGPTGeluBenchmark(
        config=config,
        batchdims=(bs,),
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
    )

    args, kwargs = gelu_bench.make_batch()
    fn = executor(gelu_bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


# There are many configurations but only the following parameters affect the swiglu benchmark:
# - intermediate_size
# - block_size
# Let's select only the configurations that differ in these parameters
def get_configs_for_swiglu():
    return get_unique_configs(("intermediate_size", "block_size"))


swiglu_executors = (
    (torch_executor, False),
    (torch_compile_executor, False),
    (thunder_executor, False),
    (torch_executor, True),
)
swiglu_executors_ids = (
    "torch",
    "torch.compile",
    "thunder",
    "liger",
)


# Sample command to run this benchmark:
# pytest thunder/benchmarks/targets.py -k "test_litgpt_swiglu" --benchmark-group-by='param:config,param:bs,param:compute_type'
@pytest.mark.parametrize(
    "executor,use_liger,",
    swiglu_executors,
    ids=swiglu_executors_ids,
)
# bs = batch size
# It's typically small for LLMs
@pytest.mark.parametrize(
    "bs,",
    (2**i for i in range(0, 2)),
    ids=(f"bs{2**i}" for i in range(0, 2)),
)
@parametrize_compute_type_only_training
@pytest.mark.parametrize(
    "config,",
    get_configs_for_swiglu(),
)
def test_litgpt_swiglu(benchmark, executor: Callable, use_liger: bool, bs: int, compute_type: ComputeType, config: str):
    if use_liger and not LIGER_FUSED_SWIGLU_AVAILABLE:
        pytest.skip("Liger fused swiglu is unavailable")

    bench: Benchmark = LitGPTSwigluBenchmark(
        config=config,
        batchdims=(bs,),
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
        use_liger=use_liger,
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


@pytest.mark.parametrize(
    "executor,",
    executors,
    ids=executors_ids,
)
@parametrize_compute_type
def test_batch_norm(benchmark, executor: Callable, compute_type: ComputeType):
    bn_bench: Benchmark = BatchNormBenchmark(
        (16, 128, 768), device="cuda:0", dtype=thunder.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = bn_bench.make_batch()
    fn = executor(bn_bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


# TODO Improve cross entropy's fwd+bwd perf when using the PyTorch executor
#   See "torch.cross_entropy implementation has incorrect dtype metadata + bwd
#        is very slow"
@pytest.mark.parametrize(
    "executor,",
    (executors + apex_executors),
    ids=(executors_ids + apex_executors),
)
@parametrize_compute_type
def test_nanogpt_cross_entropy(benchmark, executor: None | Callable, compute_type: ComputeType):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTCrossEntropyBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


# TODO: Upgrade this benchmark to use LitGPT and config, batch size parametrization
# https://github.com/Lightning-AI/lightning-thunder/issues/740
@pytest.mark.parametrize(
    "executor,",
    (executors + cudnn_layernorm_executors),
    ids=(executors_ids + cudnn_layernorm_executors_ids),
)
@parametrize_compute_type
def test_nanogpt_layer_norm(benchmark, executor: None | Callable, compute_type: ComputeType):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTLayerNormBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


def get_configs_for_rmsnorm():
    config_names = list(sorted(c["name"] for c in configs)) if RUN_ALL_CONFIGS else IMPORTANT_CONFIGS
    unique_config_names = {}
    for config_name in config_names:
        config = LitGPTConfig.from_name(config_name)
        key = tuple(
            getattr(config, k)
            for k in (
                "n_embd",
                "norm_eps",
            )
        )
        if config_name in IMPORTANT_CONFIGS:
            unique_config_names[key] = config_name
        unique_config_names.setdefault(key, config_name)

    config_names = list(sorted(unique_config_names.values()))
    return config_names


rmsnorm_executors = (
    (thunder_executor, False),
    (torch_compile_executor, False),
    (torch_executor, False),
    (torch_executor, True),
    (torch_compile_executor, True),
)
rms_executors_ids = (
    "thunderfx",
    "torch_compile",
    "torch_eager",
    "torch+apex",
    "torch_compile+apex",
)


@pytest.mark.parametrize(
    "executor,use_apex,",
    rmsnorm_executors,
    ids=rms_executors_ids,
)
@pytest.mark.parametrize(
    "bs,",
    (2**i for i in range(0, 2)),
    ids=(f"bs{2**i}" for i in range(0, 2)),
)
@parametrize_compute_type
@pytest.mark.parametrize(
    "config,",
    get_configs_for_rmsnorm(),
)
def test_litgpt_rmsnorm(benchmark, executor: Callable, use_apex: bool, bs: int, compute_type: ComputeType, config: str):
    from thunder.benchmarks import LitGPTRMSNormBenchmark

    bench: Benchmark = LitGPTRMSNormBenchmark(
        config=config,
        batchdims=(bs,),
        device="cuda:0",
        dtype=torch.bfloat16,
        requires_grad=is_requires_grad(compute_type),
        use_apex=use_apex,
    )

    jfn = executor(bench.fn())
    args, kwargs = bench.make_batch()

    benchmark_for_compute_type(compute_type, benchmark, jfn, args, kwargs)


sdpa_executors = (
    torch_executor,
    torch_compile_executor,
    thunder_executor,
    *((thunder_cudnn_nvfuser_executor,) if thunder_cudnn_nvfuser_executor is not None else ()),
)
sdpa_executors_ids = (
    "torch",
    "torch.compile",
    "thunder",
    *(("thunder+cudnn",) if thunder_cudnn_nvfuser_executor is not None else ()),
)


# Sample command to run this benchmark:
# pytest thunder/benchmarks/targets.py -k "test_litgpt_sdpa" --benchmark-group-by='param:config,param:bs,param:compute_type'
@pytest.mark.parametrize(
    "executor,",
    sdpa_executors,
    ids=sdpa_executors_ids,
)
@pytest.mark.parametrize(
    "bs,",
    (
        1,
        2,
    ),
    ids=("bs1", "bs2"),
)
@parametrize_compute_type
@pytest.mark.parametrize(
    "config,",
    IMPORTANT_CONFIGS,
)
def test_litgpt_sdpa(benchmark, executor: Callable, bs, compute_type, config):
    bench: Benchmark = LitGPTSDPABenchmark(
        config=config,
        batchdims=(bs,),
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


# TODO Fix torch.compiles bfloat16 atomic add issue with this benchmark -- why does thunder trigger it but regular torch.compile does not
@pytest.mark.parametrize(
    "executor,",
    executors,
    ids=executors_ids,
)
@parametrize_compute_type
def test_nanogpt_gpt2(benchmark, executor: Callable, compute_type: ComputeType):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


@pytest.mark.parametrize(
    "executor,",
    executors,
    ids=executors_ids,
)
@parametrize_compute_type
def test_nanogpt_gpt2xl(benchmark, executor: Callable, compute_type: ComputeType):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


#
# llama benchmarks
#


@pytest.mark.parametrize(
    "executor,",
    (transformer_engine_executors),
    ids=(transformer_engine_execuors_ids),
)
@parametrize_compute_type
@pytest.mark.parametrize("config,", (get_unique_configs(("name",))))
def test_transformer_engine_executor_regression(benchmark, executor: Callable, compute_type: ComputeType, config: str):
    # NOTE - This benchmark is present to track performance and memory usage regressions.
    #        So, we don't need to run the complete model (running a chunk should give us this signal).
    cfg: LitGPTConfig = LitGPTConfig.from_name(config)

    # Setting this to default leads to OOM.
    cfg.n_layer = 5  # This is/should be sufficient to get signals on perf regression and memory usage regression.
    b = LitGPTBenchmark(
        cfg, batchdims=(1,), device="cuda:0", dtype=torch.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    if compute_type in (ComputeType.INFERENCE, ComputeType.TRAINING_FORWARD):
        benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)
    else:
        # We use `setup_graph_on_each_invocation` here to mitigate - https://github.com/Lightning-AI/lightning-thunder/issues/701
        # Once #701 is fixed, we should remove this `if-else` and use `benchmark_for_compute_type` directly.
        with record_peak_allocated_memory(benchmark):
            backward_fn, backward_setup = backward_only(fn, *args, setup_graph_on_each_invocation=True, **kwargs)
            benchmark.pedantic(
                backward_fn, setup=lambda: (backward_setup(), {}), warmup_rounds=2, iterations=1, rounds=5
            )


@pytest.mark.parametrize("executor,", (executors + cudnn_executors), ids=(executors_ids + cudnn_executors_ids))
@parametrize_compute_type
def test_llama_2_7b_hf(benchmark, executor: Callable, compute_type: ComputeType):
    cfg: LitGPTConfig = LitGPTConfig.from_name("Llama-2-7b-hf")
    b = LitGPTBenchmark(
        cfg, batchdims=(2,), device="cuda:0", dtype=torch.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


def get_configs_for_llamamlp():
    config_names = list(sorted(c["name"] for c in configs)) if RUN_ALL_CONFIGS else IMPORTANT_CONFIGS
    unique_config_names = {}
    for config_name in config_names:
        config = LitGPTConfig.from_name(config_name)
        key = tuple(
            getattr(config, k)
            for k in (
                "n_layer",
                "n_head",
                "n_embd",
                "n_query_groups",
                "intermediate_size",
            )
        )
        if config_name in IMPORTANT_CONFIGS:
            unique_config_names[key] = config_name
        unique_config_names.setdefault(key, config_name)

    config_names = list(sorted(unique_config_names.values()))
    return config_names


@pytest.mark.parametrize(
    "executor,",
    executors,
    ids=executors_ids,
)
@pytest.mark.parametrize(
    "bs,",
    (2**i for i in range(0, 2)),
    ids=(f"bs{2**i}" for i in range(0, 2)),
)
@parametrize_compute_type
@pytest.mark.parametrize(
    "config,",
    get_configs_for_llamamlp(),
)
@pytest.mark.parametrize(
    "name",
    ["GemmaMLP", "GptNeoxMLP", "LLaMAMLP", "LLaMAMoE"],
    ids=["GemmaMLP", "GptNeoxMLP", "LLaMAMLP", "LLaMAMoE"],
)
def test_litgpt_mlp(benchmark, executor: Callable, bs: int, compute_type: ComputeType, config: str, name: str):
    bench: Benchmark = LitGPTMLPBenchmark(
        config=config,
        batchdims=(bs,),
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
        name=name,
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


# TODO: Upgrade this benchmark to use LitGPT and config, batch size parametrization
# https://github.com/Lightning-AI/lightning-thunder/issues/743
@pytest.mark.parametrize(
    "executor,",
    executors,
    ids=executors_ids,
)
@parametrize_compute_type
def test_llama2_causal_self_attention_7b(benchmark, executor: Callable, compute_type: ComputeType):
    bench: Benchmark = LitGPTCausalSelfAttentionBenchmark(
        config="Llama-2-7b-hf",
        batchdims=(2,),
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


@pytest.mark.parametrize(
    "executor,",
    executors,
    ids=executors_ids,
)
@parametrize_compute_type
def test_llama2_7b_rmsnorm_grad(benchmark, executor: Callable, compute_type: ComputeType):
    from thunder.benchmarks import LlamaRMSNormBenchmark

    bench: Benchmark = LlamaRMSNormBenchmark(
        n_embd=4096, device="cuda:0", dtype=thunder.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


# There are many configurations but only the following parameters affect the QKV split+RoPE benchmark:
# - head_size
# - n_head
# - n_query_groups
# - rope_n_elem
# - block_size
# Let's select only the configurations that differ in these parameters
def get_configs_for_qkv_split_rope():
    config_names = list(sorted(c["name"] for c in configs)) if RUN_ALL_CONFIGS else IMPORTANT_CONFIGS
    unique_config_names = {}
    for config_name in config_names:
        config = LitGPTConfig.from_name(config_name)
        key = tuple(
            getattr(config, k)
            for k in (
                "head_size",
                "n_head",
                "n_query_groups",
                "rope_n_elem",
                "block_size",
            )
        )
        if config_name in IMPORTANT_CONFIGS:
            unique_config_names[key] = config_name
        unique_config_names.setdefault(key, config_name)

    config_names = list(sorted(unique_config_names.values()))
    return config_names


qkv_split_rope_executors = (
    (torch_executor, False),
    (torch_compile_executor, False),
    (thunder_executor, False),
    (thunder_sdpa_torch_compile_nvfuser_executor, False),
    (torch_executor, True),
    (torch_compile_executor, True),
)
qkv_split_rope_executors_ids = (
    "torch",
    "torch.compile",
    "thunder",
    "thunder+nvfuser+torch.compile",
    "torch+apex",
    "torch.compile+apex",
)


# Sample command to run this benchmark:
# pytest thunder/benchmarks/targets.py -k "test_litgpt_qkv_split_rope" --benchmark-group-by='param:config,param:bs,param:compute_type'
@pytest.mark.parametrize(
    "executor,use_apex,",
    qkv_split_rope_executors,
    ids=qkv_split_rope_executors_ids,
)
# bs = batch size
# It's typically small for LLMs
@pytest.mark.parametrize(
    "bs,",
    (2**i for i in range(0, 2)),
    ids=(f"bs{2**i}" for i in range(0, 2)),
)
@parametrize_compute_type
@pytest.mark.parametrize(
    "config,",
    get_configs_for_qkv_split_rope(),
)
def test_litgpt_qkv_split_rope(
    benchmark, executor: Callable, use_apex: bool, bs: int, compute_type: ComputeType, config: str
):
    from thunder.benchmarks import LlamaQKVSplitRopeBenchmark

    if use_apex and not APEX_FUSED_ROPE_AVAILABLE:
        pytest.skip("Apex fused rotary positional embedding is unavailable")

    bench: Benchmark = LlamaQKVSplitRopeBenchmark(
        config=config,
        batchdims=(bs,),
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
        use_apex=use_apex,
    )

    jfn = executor(bench.fn())
    args, kwargs = bench.make_batch()

    benchmark_for_compute_type(compute_type, benchmark, jfn, args, kwargs)


# Thunder executor: RuntimeError: Advanced indexing currently only supports zero or one-dimensional integer tensors,
# but found a tensor with dtype thunder.dtypes.int64 and 2 dimensions
# https://github.com/Lightning-AI/lightning-thunder/issues/764
moe_executors = (torch_executor, torch_compile_executor, thunderfx_executor)
moe_executors_ids = (
    "torch",
    "torch.compile",
    "thunderfx",
)


@pytest.mark.parametrize(
    "bs,",
    (2**i for i in range(0, 6)),
    ids=(f"bs{2**i}" for i in range(0, 6)),
)
@pytest.mark.parametrize(
    "executor,",
    moe_executors,
    ids=moe_executors_ids,
)
@pytest.mark.parametrize(
    "compute_type,",
    (ComputeType.INFERENCE,),
    ids=("inference",),
)
def test_deepseek_sglang_moe(benchmark, bs, executor: Callable, compute_type: ComputeType):
    bench: Benchmark = DeepSeekSGLangMoEBenchmark(
        model="deepseek-ai/DeepSeek-R1", tp_size=8, batch_size=bs, use_fp8=False
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


#
# interpreter benchmarks
#


@pytest.mark.parametrize("executor,", (torch_executor, interpreter_fwd), ids=("python", "interpreter"))
def test_interpreter_nanogpt_gpt2_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=False,
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


#
# vision benchmarks
#


# Sample command to run this benchmark:
# pytest thunder/benchmarks/targets.py -k "test_resnet50" --benchmark-group-by='param:compute_type'
@pytest.mark.parametrize(
    "executor,",
    executors,
    ids=executors_ids,
)
@parametrize_compute_type
def test_resnet50(benchmark, executor: Callable, compute_type: ComputeType):
    b = ResNet50Benchmark(
        64, (3, 224, 224), device="cuda:0", dtype=torch.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


#
# Torchbench benchmarks
#
# To setup torchbenchmark please follow the instructions to
# install it as a libraty here https://github.com/pytorch/benchmark .
# To install canary models make sure to add `--canary` to `python install.py`.

torchbench_models = []
torchbench_canary_models = []
if importlib.util.find_spec("torchbenchmark"):
    from torchbenchmark import _list_canary_model_paths, _list_model_paths

    torchbench_models = [os.path.basename(x) for x in _list_model_paths()]
    torchbench_canary_models = [os.path.basename(x) for x in _list_canary_model_paths()]


@pytest.mark.skipif(not torchbench_models, reason="requires torchbenchmark to be installed")
@pytest.mark.parametrize(
    "module_name,",
    torchbench_models,
    ids=torchbench_models,
)
@pytest.mark.parametrize(
    "executor,",
    torchbench_executors,
    ids=torchbench_executors_ids,
)
@parametrize_compute_type
def test_torchbench(benchmark, module_name, executor, compute_type: ComputeType):
    if not importlib.util.find_spec("torchbenchmark.models." + module_name):
        pytest.skip(f"model {module_name} not installed")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        b = TorchbenchBenchmark(module_name, device="cuda", requires_grad=is_requires_grad(compute_type))

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


@pytest.mark.skipif(not torchbench_canary_models, reason="requires torchbenchmark to be installed with flag --canary")
@pytest.mark.parametrize(
    "module_name,",
    torchbench_canary_models,
    ids=torchbench_canary_models,
)
@pytest.mark.parametrize(
    "executor,",
    torchbench_executors,
    ids=torchbench_executors_ids,
)
@parametrize_compute_type
def test_torchbench_canary(benchmark, module_name, executor, compute_type: ComputeType):
    if not importlib.util.find_spec("torchbenchmark.models." + module_name):
        pytest.skip(f"model {module_name} not installed")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        b = TorchbenchBenchmark(module_name, device="cuda", requires_grad=is_requires_grad(compute_type))

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


hf_model_ids = [
    "bigcode/starcoder2-7b",
    "google/gemma-7b",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-4",
    "mistralai/Mistral-Nemo-Base-2407",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2.5-7B-Instruct",
]
hf_seq_lengths = [4096 * 2**i for i in range(0, 6)]


@pytest.mark.parametrize(
    "executor,",
    [
        thunderfx_executor,
        torch_compile_executor,
        torch_executor,
    ],
    ids=["thunderfx", "inductor", "eager"],
)
@parametrize_compute_type
@pytest.mark.parametrize("peft", [True, False], ids=["PEFT", "FT"])
@pytest.mark.parametrize(
    "seq_length",
    hf_seq_lengths,
)
@pytest.mark.parametrize("batch_size", range(1, 5), ids=[f"BS{i}" for i in range(1, 5)])
@pytest.mark.parametrize("model_id", hf_model_ids, ids=hf_model_ids)
def test_hf_transformers(
    benchmark, model_id: str, seq_length: int, batch_size: int, peft: bool, executor, compute_type
):
    if not importlib.util.find_spec("transformers"):
        pytest.skip("HF transformers not available.")

    b = HFBenchmark(
        model_id,
        seq_length,
        batch_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=is_requires_grad(compute_type),
        enable_peft=peft,
    )

    if seq_length > b.config.max_position_embeddings:
        pytest.skip("Sequence length larger than maximum for the model.")

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    if compute_type == ComputeType.TRAINING_BACKWARD:
        return_fn = lambda *args, **kwargs: fn(*args, **kwargs).loss
    else:
        kwargs["labels"] = None
        return_fn = fn

    benchmark_for_compute_type(compute_type, benchmark, return_fn, args, kwargs)


@pytest.mark.parametrize(
    "executor,",
    [
        thunderfx_executor,
        torch_compile_executor,
        torch_executor,
    ],
    ids=["thunderfx", "inductor", "eager"],
)
@parametrize_compute_type
@pytest.mark.parametrize("implementation,", ["HF", "NeMo"])
def test_lora_linear(benchmark, executor, compute_type, implementation):
    match implementation:
        case "HF":
            if not importlib.util.find_spec("transformers"):
                pytest.skip("HF transformers not available.")
            if not importlib.util.find_spec("peft"):
                pytest.skip("HF peft not available.")
        case "NeMo":
            if not importlib.util.find_spec("nemo.collections.llm"):
                pytest.skip("NeMo not available.")

    b = LinearLoRABenchmark(
        implementation=implementation,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=is_requires_grad(compute_type),
    )

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


#
# optimizer benchmarks
#


@pytest.mark.parametrize(
    "executor,",
    [
        thunderfx_executor,
        torch_compile_executor,
        torch_executor,
    ],
    ids=["thunderfx", "inductor", "eager"],
)
@parametrize_compute_type_only_inference
@pytest.mark.parametrize(
    "params,",
    [(64, 64), (128, 64)],
    ids=["64x64", "128x64"],
)
@pytest.mark.parametrize(
    "config,",
    [
        ("single_tensor", False, False),
        ("multi_tensor", True, False),
        ("fused", False, True),
    ],
    ids=["single_tensor", "multi_tensor", "fused"],
)
@pytest.mark.parametrize(
    "optimizer_name",
    ["adam", "adamax", "adadelta", "adagrad", "adamw", "asgd", "nadam", "radam", "rmsprop", "rprop", "sgd"],
    ids=["adam", "adamax", "adadelta", "adagrad", "adamw", "asgd", "nadam", "radam", "rmsprop", "rprop", "sgd"],
)
def test_optim_functional(
    benchmark,
    executor: None | Callable,
    config: tuple[str, bool, bool | None],
    params: Sequence[int],
    compute_type: ComputeType,
    optimizer_name: str,
):
    bench: Benchmark = OptimBenchmark(
        config=config,
        params=params,
        device="cuda:0",
        dtype=thunder.float32,
        requires_grad=is_requires_grad(compute_type),
        optimizer_name=optimizer_name,
    )

    fn = executor(bench.fn())
    args, kwargs = bench.make_batch()

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)
