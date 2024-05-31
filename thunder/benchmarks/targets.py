from collections.abc import Callable
from functools import partial, wraps
from collections.abc import Sequence

from lightning_utilities.core.imports import package_available

import pytest
import os
import torch
import thunder
from enum import Enum, auto
from thunder.core.transforms import clear_grads
from thunder.core.interpreter import interpret

from thunder.benchmarks import (
    Benchmark,
    NanoGPTGeLUBenchmark,
    NanoGPTCrossEntropyBenchmark,
    NanoGPTLayerNormBenchmark,
    NanoGPTSDPABenchmark,
    LitGPTSDPABenchmark,
    NanoGPTMLPBenchmark,
    NanoGPTCSABenchmark,
    NanoGPTBlockBenchmark,
    NanoGPTBenchmark,
    LitGPTBenchmark,
    LitGPTCausalSelfAttentionBenchmark,
    LlamaMLPBenchmark,
    torch_executor,
    torch_compile_executor,
    thunder_executor,
    thunder_apex_executor,
    thunder_apex_nvfuser_executor,
    thunder_cudnn_executor,
    thunder_cudnn_nvfuser_executor,
    thunder_sdpa_torch_compile_nvfuser_executor,
    BatchNormBenchmark,
)

from thunder.tests.litgpt_model import Config as LitGPTConfig
from thunder.tests.make_tensor import make_tensor

from litgpt.config import configs


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


def benchmark_for_compute_type(compute_type: ComputeType, benchmark, fn: Callable, args, kwargs):
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


def make_fwd_bwd(fn: Callable, compile_fn: Callable):
    cfn = compile_fn(fn)

    @wraps(cfn)
    def wrapper(*args, **kwargs):
        clear_grads(fn)
        result = cfn(*args, **kwargs)
        backwardable_tensor_result = thunder.core.utils.sequencify(result)
        backwardable_tensor_result = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, backwardable_tensor_result))
        result_grads = [torch.ones_like(x) for x in backwardable_tensor_result]
        torch.autograd.backward(backwardable_tensor_result, result_grads)
        return result

    return wrapper


# To compare with PyTorch and raw torch.compile (i.e. not through thunder). The
# latter can help us isolate whether it's something we need to fix ourselves or
# report upstream.
torch_fwd_bwd = partial(make_fwd_bwd, compile_fn=torch_executor)
torchcompile_fwd_bwd = partial(make_fwd_bwd, compile_fn=torch_compile_executor)

# Default thunder configs
thunder_fwd_bwd = partial(make_fwd_bwd, compile_fn=thunder_executor)

# Executing with just the apex executor
# NOTE apex may or may not be available
thunder_apex_grad: None | Callable = None
if thunder_apex_executor is not None:
    thunder_apex_grad = partial(make_fwd_bwd, compile_fn=thunder_apex_executor)

# Executing with the apex and nvfuser executors
thunder_apex_nvfuser_grad: None | Callable = None
if thunder_apex_nvfuser_executor is not None:
    thunder_apex_nvfuser_grad = partial(make_fwd_bwd, compile_fn=thunder_apex_nvfuser_executor)


fwd_executors = (torch_executor, torch_compile_executor, thunder_executor)
fwd_executor_ids = (
    "torch",
    "torch.compile",
    "thunder",
)

thunder_fwd_bwd_sdpa_torch_compile_nvfuser = partial(
    make_fwd_bwd, compile_fn=thunder_sdpa_torch_compile_nvfuser_executor
)

grad_executors = (
    *(
        (partial(make_fwd_bwd, compile_fn=thunder_cudnn_nvfuser_executor),)
        if thunder_cudnn_nvfuser_executor is not None
        else ()
    ),
    torch_fwd_bwd,
    torchcompile_fwd_bwd,
    thunder_fwd_bwd,
    thunder_fwd_bwd_sdpa_torch_compile_nvfuser,
)
grad_executors_ids = (
    *(("thunder+cudnn",) if thunder_cudnn_nvfuser_executor is not None else ()),
    "torch",
    "torch.compile",
    "thunder",
    "thunder+nvfuser+torch.compile",
)

apex_executors = (thunder_apex_executor, thunder_apex_nvfuser_executor)
apex_executors_ids = ("thunder+apex-grad", "thunder+apex+nvfuser-grad")

cudnn_fwd_executors = (thunder_cudnn_executor, thunder_cudnn_nvfuser_executor)
cudnn_fwd_executors_ids = ("thunder+cudnn", "thunder+cudnn+nvfuser")

cudnn_layernorm_fwd_executors = (thunder_cudnn_executor, thunder_cudnn_nvfuser_executor)
cudnn_layernorm_fwd_executors_ids = (
    "thunder+cudnn_layernorm",
    "thunder+cudnn_layernorm+nvfuser",
)

#
# nanogpt benchmarks
#


@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
@parametrize_compute_type
def test_nanogpt_gelu(benchmark, executor: Callable, compute_type: ComputeType):
    gelu_bench: Benchmark = NanoGPTGeLUBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = gelu_bench.make_batch()
    fn = executor(gelu_bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
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
    (fwd_executors + apex_executors),
    ids=(fwd_executor_ids + apex_executors),
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


@pytest.mark.parametrize(
    "executor,",
    (fwd_executors + cudnn_layernorm_fwd_executors),
    ids=(fwd_executor_ids + cudnn_layernorm_fwd_executors_ids),
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


@pytest.mark.parametrize(
    "executor,", (fwd_executors + cudnn_fwd_executors), ids=(fwd_executor_ids + cudnn_fwd_executors_ids)
)
@parametrize_compute_type
def test_nanogpt_sdpa(benchmark, executor: None | Callable, compute_type: ComputeType):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTSDPABenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
@parametrize_compute_type
def test_llama2_7b_sdpa(benchmark, executor: Callable, compute_type: ComputeType):
    bench: Benchmark = LitGPTSDPABenchmark(
        config="Llama-2-7b-hf", device="cuda:0", dtype=thunder.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


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
# pytest thunder/benchmarks/targets.py -k "test_litgpt_sdpa_grad" --benchmark-group-by='param:config,param:bs' --benchmark-columns='min,max,mean,stddev,median'
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


@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
@parametrize_compute_type
def test_nanogpt_mlp(benchmark, executor: Callable, compute_type: ComputeType):
    bench: Benchmark = NanoGPTMLPBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=is_requires_grad(compute_type)
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


# NOTE The CSA module is linear -> sdpa -> dropout
@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
@parametrize_compute_type
def test_nanogpt_csa(benchmark, executor: Callable, compute_type: ComputeType):
    bench: Benchmark = NanoGPTCSABenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=is_requires_grad(compute_type),
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark_for_compute_type(compute_type, benchmark, fn, args, kwargs)


# NOTE NanoGPT's block module is layernorm -> csa -> layernorm -> mlp
@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
def test_nanogpt_block_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBlockBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


# NOTE NanoGPT's block module is layernorm -> csa -> layernorm -> mlp
@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_nanogpt_block_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBlockBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


# TODO Fix torch.compiles bfloat16 atomic add issue with this benchmark -- why does thunder trigger it but regular torch.compile does not
@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
def test_nanogpt_gpt2_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=False,
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


# TODO Fix torch.compiles bfloat16 atomic add issue with this benchmark and add thunder-grad+torch.compile executor back
@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_nanogpt_gpt2_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=True,
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
def test_nanogpt_gpt2xl_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=False,
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


# TODO Fix torch.compiles bfloat16 atomic add issue with this benchmark and add thunder-grad+torch.compile executor back
@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_nanogpt_gpt2xl_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=True,
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


#
# llama benchmarks
#


@pytest.mark.parametrize(
    "executor,", (fwd_executors + cudnn_fwd_executors), ids=(fwd_executor_ids + cudnn_fwd_executors_ids)
)
def test_open_llama_7b_fwd(benchmark, executor: Callable):
    cfg: LitGPTConfig = LitGPTConfig.from_name("open_llama_7b")
    b = LitGPTBenchmark(cfg, device="cuda:0", dtype=torch.bfloat16, requires_grad=False)

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    benchmark(fn, *args, **kwargs)


@pytest.mark.parametrize(
    "executor,", (fwd_executors + cudnn_fwd_executors), ids=(fwd_executor_ids + cudnn_fwd_executors_ids)
)
def test_llama_2_7b_hf_fwd(benchmark, executor: Callable):
    cfg: LitGPTConfig = LitGPTConfig.from_name("Llama-2-7b-hf")
    b = LitGPTBenchmark(cfg, device="cuda:0", dtype=torch.bfloat16, requires_grad=False)

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    benchmark(fn, *args, **kwargs)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_llama_2_7b_grad(benchmark, executor: Callable):
    cfg: LitGPTConfig = LitGPTConfig.from_name("Llama-2-7b-hf")
    b = LitGPTBenchmark(
        cfg,
        batchdims=(2,),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    args, kwargs = b.make_batch()
    fn = executor(b.fn())

    benchmark(fn, *args, **kwargs)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_llama2_mlp_7b_grad(benchmark, executor: Callable):
    bench: Benchmark = LlamaMLPBenchmark(
        config="Llama-2-7b-hf", batchdims=(16,), device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_llama2_causal_self_attention_7b_grad(benchmark, executor: Callable):
    bench: Benchmark = LitGPTCausalSelfAttentionBenchmark(
        config="Llama-2-7b-hf", batchdims=(16,), device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_llama2_7b_rmsnorm_grad(benchmark, executor: Callable):
    from thunder.benchmarks import LlamaRMSNormBenchmark

    bench: Benchmark = LlamaRMSNormBenchmark(n_embd=4096, device="cuda:0", dtype=thunder.bfloat16, requires_grad=True)

    args, kwargs = bench.make_batch()
    fn = executor(bench.fn())

    benchmark(fn, *args, **kwargs)


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
def test_litgpt_qkv_split_rope(benchmark, executor: Callable, use_apex: bool, bs: int, compute_type: ComputeType, config: str):
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


def backward_only(fn: Callable, *args, **kwargs):
    """
    Returns a function that runs the backward pass of the given function.

    The returned function should be called with the output gradients.

    Args:
        fn: The forward function
        *args: Arguments to the forward function
        **kwargs: Keyword arguments to the forward function

    Returns:
        A tuple of the backward function and the setup function
        that returns the arguments for the backward function.
    """
    result = fn(*args, **kwargs)
    result = thunder.core.utils.sequencify(result)

    forward_inputs = thunder.core.pytree.tree_flatten((args, kwargs))[0]
    forward_inputs = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, forward_inputs))
    backwardable_tensor_result = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, result))

    # Capture metadata for backward to avoid keeping the result in memory
    backwardable_result_metadata = [(r.dtype, r.device, r.shape) for r in backwardable_tensor_result]

    def backward_setup():
        output_grads = []
        for dtype, device, shape in backwardable_result_metadata:
            torch_dtype = thunder.torch.to_torch_dtype(dtype)
            torch_device = thunder.core.devices.to_torch_device(device)
            output_grads.append(make_tensor(shape, dtype=torch_dtype, device=torch_device, requires_grad=False))
        return output_grads

    def backward_fn(*output_grads):
        for i in forward_inputs:
            i.grad = None

        torch.autograd.backward(result, output_grads, retain_graph=True)

    return backward_fn, backward_setup


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
