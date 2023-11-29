from collections.abc import Callable
from functools import partial
from collections.abc import Sequence

from lightning_utilities.core.imports import package_available

import pytest
import torch
import thunder
from thunder.core.transforms import grad, grad_v1, clear_grads, populate_grads, get_grad, put_grad, put_grads

from thunder.benchmarks import (
    Benchmark,
    NanoGPTGeLUBenchmark,
    NanoGPTCrossEntropyBenchmark,
    NanoGPTLayerNormBenchmark,
    NanoGPTSDPABenchmark,
    NanoGPTMLPBenchmark,
    NanoGPTCSABenchmark,
    NanoGPTBlockBenchmark,
    NanoGPTBenchmark,
    torch_executor,
    torch_compile_executor,
    thunder_nvfuser_executor,
    thunder_torch_executor,
    thunder_torch_compile_executor,
    thunder_apex_executor,
    thunder_apex_nvfuser_executor,
    thunder_cudnn_executor,
    thunder_cudnn_nvfuser_executor,
    thunder_cudnn_layer_norm_executor,
    thunder_cudnn_layer_norm_nvfuser_executor,
    thunder_sdpa_executor,
    thunder_sdpa_nvfuser_executor,
)


APEX_FUSED_ROPE_AVAILABLE: bool = package_available("fused_rotary_positional_embedding")


def make_setup(b: Benchmark):
    def setup():
        args_and_kwargs = b.make_batch()
        torch.cuda.synchronize()
        return args_and_kwargs

    return setup


def wrap_for_benchmark(fn):
    def fn_(*args, **kwargs):
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        return result

    return fn_


def torch_eager_fwd(b: Benchmark):
    module = b.fn()
    fn_ = torch_executor(module)

    if isinstance(module, torch.nn.Sequential):

        def wrapper(*args):
            result = fn_(args)
            return result

        return wrapper

    def wrapper(*args, **kwargs):
        result = fn_(*args, **kwargs)
        return result

    return wrapper


def torch_compile_fwd(b: Benchmark):
    module = b.fn()
    fn_ = torch_compile_executor(module)

    if isinstance(module, torch.nn.Sequential):

        def wrapper(*args):
            result = fn_(args)
            return result

        return wrapper

    def wrapper(*args, **kwargs):
        result = fn_(*args, **kwargs)
        return result

    return wrapper


# NOTE This is hitting torch.compile errors on at least some of the benchmarks
def torch_compile_compiled_bwd(b: Benchmark):
    module = b.fn()

    def foo(*args, **kwargs):
        result = module(*args, **kwargs)
        result.backward(result)
        return result

    cfoo = torch_compile_executor(foo)

    if isinstance(module, torch.nn.Sequential):

        def wrapper(*args):
            clear_grads(module)
            return cfoo(args)

        return wrapper

    def wrapper(*args, **kwargs):
        clear_grads(module)
        result = cfoo(*args, **kwargs)
        return result

    return wrapper


def thunder_fwd(b: Benchmark, compile_fn: Callable):
    module: torch.nn.Module = b.fn()
    cfn = compile_fn(module)

    if isinstance(module, torch.nn.Sequential):

        def wrapper(*args):
            return cfn(args)

        return wrapper

    def wrapper(*args, **kwargs):
        result = cfn(*args, **kwargs)
        return result

    return wrapper


# Requires the function output actually be computed to compute the grad
def grad_specifier(x) -> None:
    put_grad(x, x)


def thunder_grad_transform(b: Benchmark, compile_fn: Callable):
    module: torch.nn.Module = b.fn()
    cfn = compile_fn(module)
    cfn_grad = grad(cfn, grad_specifier=grad_specifier)

    if isinstance(module, torch.nn.Sequential):

        def wrapper(*args):
            clear_grads(cfn)
            grads = cfn_grad(args)
            populate_grads(grads, cfn, args=args)

        return wrapper

    def wrapper(*args, **kwargs):
        clear_grads(cfn)
        grads = cfn_grad(*args, **kwargs)
        populate_grads(grads, cfn, args=args, kwargs=kwargs)

    return wrapper


def thunder_grad_transform_v1(b: Benchmark, compile_fn: Callable):
    module: torch.nn.Module = b.fn()
    cfn = compile_fn(module)
    cfn_grad = grad_v1(cfn)

    if isinstance(module, torch.nn.Sequential):

        def wrapper(*args):
            clear_grads(cfn)
            grads = cfn_grad(args)
            populate_grads(grads, cfn, args=args)

        return wrapper

    def wrapper(*args, **kwargs):
        clear_grads(cfn)
        grads = cfn_grad(*args, **kwargs)
        populate_grads(grads, cfn, args=args, kwargs=kwargs)

    return wrapper


def thunder_fwd_bwd(b: Benchmark, compile_fn: Callable):
    module: torch.nn.Module = b.fn()
    cfn = compile_fn(module)

    if isinstance(module, torch.nn.Sequential):

        def wrapper(*args):
            clear_grads(module)
            result = cfn(args)
            result.backward(result)
            return result

        return wrapper

    def wrapper(*args, **kwargs):
        clear_grads(module)
        result = cfn(*args, **kwargs)
        if isinstance(result, Sequence):
            torch.autograd.backward(result, [torch.randn_like(x) for x in result])
        else:
            result.backward(torch.randn_like(result))
        return result

    return wrapper


torch_eager_bwd = partial(thunder_fwd_bwd, compile_fn=torch_executor)
torch_compile_torch_bwd = partial(thunder_fwd_bwd, compile_fn=torch_compile_executor)

thunder_fwd_nvfuser = partial(thunder_fwd, compile_fn=thunder_nvfuser_executor)
thunder_fwd_torch_compile = partial(thunder_fwd, compile_fn=thunder_torch_compile_executor)
thunder_fwd_bwd_nvfuser = partial(thunder_fwd_bwd, compile_fn=thunder_nvfuser_executor)
thunder_grad_transform_nvfuser = partial(thunder_grad_transform, compile_fn=thunder_nvfuser_executor)
thunder_grad_transform_torch_compile = partial(thunder_grad_transform, compile_fn=thunder_torch_compile_executor)
thunder_grad_transform_torch = partial(thunder_grad_transform, compile_fn=thunder_torch_executor)
thunder_grad_transform_v1_nvfuser = partial(thunder_grad_transform_v1, compile_fn=thunder_nvfuser_executor)
thunder_grad_transform_v1_torch_compile = partial(thunder_grad_transform_v1, compile_fn=thunder_torch_compile_executor)
thunder_grad_transform_v1_torch = partial(thunder_grad_transform_v1, compile_fn=thunder_torch_executor)

# NOTE apex may or may not be available
thunder_grad_transform_apex: None | Callable = None
if thunder_apex_executor is not None:
    thunder_grad_transform_apex = partial(thunder_grad_transform, compile_fn=thunder_apex_executor)

thunder_grad_transform_apex_nvfuser: None | Callable = None
if thunder_apex_nvfuser_executor is not None:
    thunder_grad_transform_apex_nvfuser = partial(thunder_grad_transform, compile_fn=thunder_apex_nvfuser_executor)

# NOTE cudnn may or may not be available
thunder_fwd_cudnn: None | Callable = None
if thunder_cudnn_executor is not None:
    thunder_fwd_cudnn = partial(thunder_fwd, compile_fn=thunder_cudnn_executor)

thunder_fwd_cudnn_nvfuser: None | Callable = None
if thunder_cudnn_nvfuser_executor is not None:
    thunder_fwd_cudnn_nvfuser = partial(thunder_fwd, compile_fn=thunder_cudnn_nvfuser_executor)

thunder_fwd_cudnn_layer_norm: None | Callable = None
if thunder_cudnn_layer_norm_executor is not None:
    thunder_fwd_cudnn_layer_norm = partial(thunder_fwd, compile_fn=thunder_cudnn_layer_norm_executor)

thunder_fwd_cudnn_layer_norm_nvfuser: None | Callable = None
if thunder_cudnn_layer_norm_nvfuser_executor is not None:
    thunder_fwd_cudnn_layer_norm_nvfuser = partial(thunder_fwd, compile_fn=thunder_cudnn_layer_norm_nvfuser_executor)

thunder_grad_sdpa_executor = partial(thunder_grad_transform, compile_fn=thunder_sdpa_executor)
thunder_grad_sdpa_nvfuser_executor = partial(thunder_grad_transform, compile_fn=thunder_sdpa_nvfuser_executor)
thunder_fwd_bwd_sdpa_nvfuser = partial(thunder_fwd_bwd, compile_fn=thunder_sdpa_nvfuser_executor)
thunder_grad_v1_sdpa_executor = partial(thunder_grad_transform_v1, compile_fn=thunder_sdpa_executor)
thunder_grad_v1_sdpa_nvfuser_executor = partial(thunder_grad_transform_v1, compile_fn=thunder_sdpa_nvfuser_executor)


@pytest.mark.parametrize(
    "executor,",
    (torch_eager_fwd, torch_compile_fwd, thunder_fwd_torch_compile, thunder_fwd_nvfuser),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
    ),
)
def test_gelu_fwd(benchmark, executor: Callable):
    gelu_bench: Benchmark = NanoGPTGeLUBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(gelu_bench)
    fn = executor(gelu_bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=0)


@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_grad_transform_torch_compile,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_torch_compile,
        thunder_grad_transform_v1_nvfuser,
        thunder_fwd_bwd_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile+torch-bwd",
        "thunder-grad+torch.compile",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+torch.compile",
        "thunder-grad_v1+nvfuser",
        "thunder-fwd-bwd+nvfuser",
    ),
)
def test_gelu_grad(benchmark, executor: Callable):
    gelu_bench: Benchmark = NanoGPTGeLUBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(gelu_bench)
    fn = executor(gelu_bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=0)


# TODO Improve cross entropy's fwd+bwd perf when using the PyTorch executor
#   See https://github.com/Lightning-AI/lightning-thunder/issues/1319
@pytest.mark.parametrize(
    "executor,",
    (torch_eager_fwd, torch_compile_fwd, thunder_fwd_torch_compile, thunder_fwd_nvfuser),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
    ),
)
def test_cross_entropy_fwd(benchmark, executor: None | Callable):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTCrossEntropyBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


# TODO Improve cross entropy's fwd+bwd perf when using the PyTorch executor
#   See https://github.com/Lightning-AI/lightning-thunder/issues/1319
@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_grad_transform_torch_compile,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_torch_compile,
        thunder_grad_transform_v1_nvfuser,
        thunder_grad_transform_apex,
        thunder_grad_transform_apex_nvfuser,
        thunder_fwd_bwd_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile+torch-bwd",
        "thunder-grad+torch.compile",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+torch.compile",
        "thunder-grad_v1+nvfuser",
        "thunder-grad+apex",
        "thunder-grad+apex+nvfuser",
        "thunder-fwd-bwd+nvfuser",
    ),
)
def test_cross_entropy_grad(benchmark, executor: None | Callable):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTCrossEntropyBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


# TODO Improve cross entropy's fwd+bwd perf when using the PyTorch executor
#   See https://github.com/Lightning-AI/lightning-thunder/issues/1319
@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_fwd,
        torch_compile_fwd,
        thunder_fwd_torch_compile,
        thunder_fwd_nvfuser,
        thunder_fwd_cudnn_layer_norm,
        thunder_fwd_cudnn_layer_norm_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
        "thunder+cudnn_layernorm",
        "thunder+cudnn_layernorm+nvfuser",
    ),
)
def test_layer_norm_fwd(benchmark, executor: None | Callable):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTLayerNormBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_fwd,
        torch_compile_fwd,
        thunder_fwd_torch_compile,
        thunder_fwd_nvfuser,
        thunder_fwd_cudnn,
        thunder_fwd_cudnn_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
        "thunder+cudnn",
        "thunder+cudnn+nvfuser",
    ),
)
def test_sdpa_fwd(benchmark, executor: None | Callable):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTSDPABenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


# TODO Fix thunder-fwd-bwd+nvfuser
@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_grad_transform_torch_compile,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_torch_compile,
        thunder_grad_transform_v1_nvfuser,
        thunder_grad_sdpa_executor,
        thunder_grad_sdpa_nvfuser_executor,
        thunder_fwd_bwd_sdpa_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile+torch-bwd",
        "thunder-grad+torch.compile",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+torch.compile",
        "thunder-grad_v1+nvfuser",
        "thunder-grad+sdpa",
        "thunder-grad+sdpa+nvfuser",
        "thunder-fwd-bwd+sdpa+nvfuser",
    ),
)
def test_sdpa_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTSDPABenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


@pytest.mark.parametrize(
    "executor,",
    (torch_eager_fwd, torch_compile_fwd, thunder_fwd_torch_compile, thunder_fwd_nvfuser),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
    ),
)
def test_mlp_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTMLPBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=0)


@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_grad_transform_torch_compile,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_torch_compile,
        thunder_grad_transform_v1_nvfuser,
        thunder_fwd_bwd_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile+torch-bwd",
        "thunder-grad+torch.compile",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+torch.compile",
        "thunder-grad_v1+nvfuser",
        "thunder-fwd-bwd+nvfuser",
    ),
)
def test_mlp_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTMLPBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=0)


@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_fwd_bwd_nvfuser,
        thunder_grad_transform_torch_compile,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_torch_compile,
        thunder_grad_transform_v1_nvfuser,
    ),
    ids=(
        # "torch-bwd" here means that PyTorch's .backward() is used
        "torch-eager+torch-bwd",
        "torch.compile+torch-bwd",
        "thunder+nvfuser+torch-bwd",
        # The following "executors" return only the gradient wrt the input so
        # they might remove dead code paths from the joint fwd+bwd call
        "thunder-grad+torch.compile",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+torch.compile",
        "thunder-grad_v1+nvfuser",
    ),
)
def test_llama2_mlp_7b_requires_grad(benchmark, executor: Callable):
    from thunder.benchmarks import LlamaMLPBenchmark

    bench: Benchmark = LlamaMLPBenchmark(
        config="Llama-2-7b-hf", batchdims=(16,), device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_fwd_bwd_sdpa_nvfuser,
    ),
    ids=(
        # "torch-bwd" here means that PyTorch's .backward() is used
        "torch-eager+torch-bwd",
        "torch.compile+torch-bwd",
        "thunder+nvfuser+sdpa+torch-bwd",
    ),
)
def test_llama2_causal_self_attention_7b_train(benchmark, executor: Callable):
    from thunder.benchmarks import LlamaCausalSelfAttentionBenchmark

    bench: Benchmark = LlamaCausalSelfAttentionBenchmark(
        config="Llama-2-7b-hf", batchdims=(16,), device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,use_apex,",
    (
        (torch_eager_bwd, False),
        (torch_compile_torch_bwd, False),
        (thunder_fwd_bwd_nvfuser, False),
        (torch_eager_bwd, True),
        (torch_compile_torch_bwd, True),
    ),
    ids=(
        "torch-eager+torch-bwd",
        "torch.compile+torch-bwd",
        "thunder+nvfuser+torch-bwd",
        "torch-eager+torch-bwd+apex",
        "torch.compile+torch-bwd+apex",
    ),
)
def test_llama2_qkv_split_rope_7b_train(benchmark, executor: Callable, use_apex: bool):
    from thunder.benchmarks import LlamaQKVSplitRopeBenchmark

    if use_apex and not APEX_FUSED_ROPE_AVAILABLE:
        pytest.skip("Apex fused rotary positional embedding is unavailable")

    bench: Benchmark = LlamaQKVSplitRopeBenchmark(
        config="Llama-2-7b-hf",
        batchdims=(32,),
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=True,
        use_apex=use_apex,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


# NOTE The CSA module is linear -> sdpa -> dropout
@pytest.mark.parametrize(
    "executor,",
    (torch_eager_fwd, torch_compile_fwd, thunder_fwd_torch_compile, thunder_fwd_nvfuser),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
    ),
)
def test_csa_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTCSABenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=False,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


# NOTE The CSA module is linear -> sdpa -> dropout
@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_grad_transform_torch_compile,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_torch_compile,
        thunder_grad_transform_v1_nvfuser,
        thunder_fwd_bwd_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile+torch-bwd",
        "thunder-grad+torch.compile",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+torch.compile",
        "thunder-grad_v1+nvfuser",
        "thunder-fwd-bwd+nvfuser",
    ),
)
def test_csa_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTCSABenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=True,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


# NOTE NanoGPT's block module is layernorm -> csa -> layernorm -> mlp
@pytest.mark.parametrize(
    "executor,",
    (torch_eager_fwd, torch_compile_fwd, thunder_fwd_torch_compile, thunder_fwd_nvfuser),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
    ),
)
def test_block_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBlockBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


# NOTE NanoGPT's block module is layernorm -> csa -> layernorm -> mlp
@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_grad_transform_torch_compile,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_torch_compile,
        thunder_grad_transform_v1_nvfuser,
        thunder_fwd_bwd_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile+torch-bwd",
        "thunder-grad+torch.compile",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+torch.compile",
        "thunder-grad_v1+nvfuser",
        "thunder-fwd-bwd+nvfuser",
    ),
)
def test_block_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBlockBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=0)


# TODO Fix torch.compiles bfloat16 atomic add issue with this benchmark -- why does thunder trigger it but regular torch.compile does not
@pytest.mark.parametrize(
    "executor,",
    (torch_eager_fwd, torch_compile_fwd, thunder_fwd_torch_compile, thunder_fwd_nvfuser),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
    ),
)
def test_gpt2_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=False,
        only_return_loss=False,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=0)


# TODO Fix torch.compiles bfloat16 atomic add issue with this benchmark and add thunder-grad+torch.compile executor back
@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_nvfuser,
        thunder_fwd_bwd_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile+torch-bwd",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+nvfuser",
        "thunder-fwd-bwd+nvfuser",
    ),
)
def test_gpt2_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=True,
        only_return_loss=True,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=0)


@pytest.mark.parametrize(
    "executor,",
    (torch_eager_fwd, torch_compile_fwd, thunder_fwd_torch_compile, thunder_fwd_nvfuser),
    ids=(
        "torch-eager",
        "torch.compile",
        "thunder+torch.compile",
        "thunder+nvfuser",
    ),
)
def test_gpt2xl_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=False,
        only_return_loss=False,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=0)


# TODO Fix torch.compiles bfloat16 atomic add issue with this benchmark and add thunder-grad+torch.compile executor back
@pytest.mark.parametrize(
    "executor,",
    (
        torch_eager_bwd,
        torch_compile_torch_bwd,
        thunder_grad_transform_nvfuser,
        thunder_grad_transform_v1_nvfuser,
        thunder_fwd_bwd_nvfuser,
    ),
    ids=(
        "torch-eager",
        "torch.compile+torch-bwd",
        "thunder-grad+nvfuser",
        "thunder-grad_v1+nvfuser",
        "thunder-fwd-bwd+nvfuser",
    ),
)
def test_gpt2xl_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTBenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=True,
        only_return_loss=True,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=0)
