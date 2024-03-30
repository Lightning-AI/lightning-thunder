from collections.abc import Callable
from functools import partial, wraps
from collections.abc import Sequence

from lightning_utilities.core.imports import package_available

import pytest
import torch
import thunder
from thunder.core.transforms import grad, grad_v1, clear_grads, populate_grads, get_grad, put_grad, put_grads
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
    thunder_torch_executor,
    thunder_torch_compile_executor,
    thunder_apex_executor,
    thunder_apex_nvfuser_executor,
    thunder_cudnn_executor,
    thunder_cudnn_nvfuser_executor,
    thunder_cudnn_layer_norm_executor,
    thunder_cudnn_layer_norm_nvfuser_executor,
    thunder_sdpa_executor,
    thunder_sdpa_torch_compile_nvfuser_executor,
)

from thunder.tests.litgpt_model import Config as LitGPTConfig


APEX_FUSED_ROPE_AVAILABLE: bool = package_available("fused_rotary_positional_embedding")


def make_setup(b: Benchmark):
    def setup():
        args_and_kwargs = b.make_batch()
        torch.cuda.synchronize()
        return args_and_kwargs

    return setup


def wrap_for_benchmark(fn):
    @wraps(fn)
    def fn_(*args, **kwargs):
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        return result

    return fn_


def torch_fwd(b: Benchmark):
    module = b.fn()
    fn_ = torch_executor(module)

    if isinstance(module, torch.nn.Sequential):

        @wraps(fn_)
        def wrapper(*args):
            result = fn_(args)
            return result

        return wrapper

    @wraps(fn_)
    def wrapper(*args, **kwargs):
        result = fn_(*args, **kwargs)
        return result

    return wrapper


def interpreter_fwd(b: Benchmark):
    module = b.fn()
    fn_ = torch_executor(module)
    fn_ = interpret(fn_)

    if isinstance(module, torch.nn.Sequential):

        @wraps(fn_)
        def wrapper(*args):
            result = fn_(args)
            return result

        return wrapper

    @wraps(fn_)
    def wrapper(*args, **kwargs):
        result = fn_(*args, **kwargs)
        return result

    return wrapper


def torch_compile_fwd(b: Benchmark):
    module = b.fn()
    fn_ = torch_compile_executor(module)

    if isinstance(module, torch.nn.Sequential):

        @wraps(fn_)
        def wrapper(*args):
            result = fn_(args)
            return result

        return wrapper

    @wraps(fn_)
    def wrapper(*args, **kwargs):
        result = fn_(*args, **kwargs)
        return result

    return wrapper


# NOTE This is hitting torch.compile errors on at least some of the benchmarks
def torch_compile_compiled_bwd(b: Benchmark):
    module = b.fn()

    def foo(*args, **kwargs):
        result = module(*args, **kwargs)
        result.backward(torch.ones_like(result))
        return result

    cfoo = torch_compile_executor(foo)

    if isinstance(module, torch.nn.Sequential):

        @wraps(cfoo)
        def wrapper(*args):
            clear_grads(module)
            return cfoo(args)

        return wrapper

    @wraps(cfoo)
    def wrapper(*args, **kwargs):
        clear_grads(module)
        result = cfoo(*args, **kwargs)
        return result

    return wrapper


def thunder_fwd(b: Benchmark, compile_fn: Callable):
    module: torch.nn.Module = b.fn()
    cfn = compile_fn(module)

    if isinstance(module, torch.nn.Sequential):

        @wraps(cfn)
        def wrapper(*args):
            return cfn(args)

        return wrapper

    @wraps(cfn)
    def wrapper(*args, **kwargs):
        result = cfn(*args, **kwargs)
        return result

    return wrapper


# TODO Actually return the fwd, currently just requires computation
# by making the fwd equal to the grad
def thunder_value_and_grad_transform(b: Benchmark, compile_fn: Callable):
    module: torch.nn.Module = b.fn()
    cfn = compile_fn(module)

    # Note on grad_specifier:
    # requires the function output actually be computed to compute the grad
    def grad_specifier(outs):
        if not isinstance(outs, Sequence):
            outs = (outs,)

        for out in outs:
            put_grad(out, out)

    cfn_grad = grad(cfn, grad_specifier=grad_specifier)

    if isinstance(module, torch.nn.Sequential):

        @wraps(cfn_grad)
        def wrapper(*args):
            clear_grads(cfn)
            grads = cfn_grad(args)
            populate_grads(grads, cfn, args=args)

        return wrapper

    @wraps(cfn_grad)
    def wrapper(*args, **kwargs):
        clear_grads(cfn)
        grads = cfn_grad(*args, **kwargs)
        populate_grads(grads, cfn, args=args, kwargs=kwargs)

    return wrapper


def thunder_grad_transform(b: Benchmark, compile_fn: Callable):
    module: torch.nn.Module = b.fn()
    cfn = compile_fn(module)
    cfn_grad = grad(cfn)

    if isinstance(module, torch.nn.Sequential):

        @wraps(cfn_grad)
        def wrapper(*args):
            clear_grads(cfn)
            grads = cfn_grad(args)
            populate_grads(grads, cfn, args=args)

        return wrapper

    @wraps(cfn_grad)
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

        @wraps(cfn_grad)
        def wrapper(*args):
            clear_grads(cfn)
            grads = cfn_grad(args)
            populate_grads(grads, cfn, args=args)

        return wrapper

    @wraps(cfn_grad)
    def wrapper(*args, **kwargs):
        clear_grads(cfn)
        grads = cfn_grad(*args, **kwargs)
        populate_grads(grads, cfn, args=args, kwargs=kwargs)

    return wrapper


def thunder_fwd_bwd(b: Benchmark, compile_fn: Callable):
    module: torch.nn.Module = b.fn()
    cfn = compile_fn(module)

    if isinstance(module, torch.nn.Sequential):

        @wraps(cfn)
        def wrapper(*args):
            clear_grads(module)
            result = cfn(args)
            result.backward(torch.ones_like(result))
            return result

        return wrapper

    @wraps(cfn)
    def wrapper(*args, **kwargs):
        clear_grads(module)
        result = cfn(*args, **kwargs)
        if isinstance(result, Sequence):
            torch.autograd.backward(result, [torch.ones_like(x) for x in result])
        else:
            result.backward(torch.ones_like(result))
        return result

    return wrapper


# To compare with PyTorch and raw torch.compile (i.e. not through thunder). The
# latter can help us isolate whether it's something we need to fix ourself or
# report upstream.
torch_fwd_bwd = partial(thunder_fwd_bwd, compile_fn=torch_executor)
torchcompile_fwd_bwd = partial(thunder_fwd_bwd, compile_fn=torch_compile_executor)

# Executing with just PyTorch
thunder_torch_grad = partial(thunder_grad_transform, compile_fn=thunder_torch_executor)
thunder_torch_gradv1 = partial(thunder_grad_transform_v1, compile_fn=thunder_torch_executor)
thunder_torch_value_and_grad = partial(thunder_value_and_grad_transform, compile_fn=thunder_torch_executor)

# Default thunder configs
thunder_fwd = partial(thunder_fwd, compile_fn=thunder_executor)
thunder_fwd_bwd = partial(thunder_fwd_bwd, compile_fn=thunder_executor)
thunder_grad = partial(thunder_grad_transform, compile_fn=thunder_executor)
thunder_gradv1 = partial(thunder_grad_transform_v1, compile_fn=thunder_executor)
thunder_value_and_grad = partial(thunder_value_and_grad_transform, compile_fn=thunder_executor)

# Executing with torchcompile
thunder_torchcompile_fwd = partial(thunder_fwd, compile_fn=thunder_torch_compile_executor)
thunder_torchcompile_grad = partial(thunder_grad_transform, compile_fn=thunder_torch_compile_executor)
thunder_torchcompile_gradv1 = partial(thunder_grad_transform_v1, compile_fn=thunder_torch_compile_executor)
thunder_torchcompile_value_and_grad = partial(
    thunder_value_and_grad_transform, compile_fn=thunder_torch_compile_executor
)

# Executing with just the sdpa executor
thunder_sdpa_grad = partial(thunder_grad_transform, compile_fn=thunder_sdpa_executor)
thunder_sdpa_gradv1 = partial(thunder_grad_transform_v1, compile_fn=thunder_sdpa_executor)

# Executing with just the apex executor
# NOTE apex may or may not be available
thunder_apex_grad: None | Callable = None
if thunder_apex_executor is not None:
    thunder_apex_grad = partial(thunder_grad_transform, compile_fn=thunder_apex_executor)

# Executing with the apex and nvfuser executors
thunder_apex_nvfuser_grad: None | Callable = None
if thunder_apex_nvfuser_executor is not None:
    thunder_apex_nvfuser_grad = partial(thunder_grad_transform, compile_fn=thunder_apex_nvfuser_executor)

# Executing with just the cuDNN executor
# NOTE cudnn may or may not be available
thunder_cudnn_fwd: None | Callable = None
if thunder_cudnn_executor is not None:
    thunder_cudnn_fwd = partial(thunder_fwd, compile_fn=thunder_cudnn_executor)

# Executing with cuDNN + nvFuser
thunder_cudnn_nvfuser_fwd: None | Callable = None
if thunder_cudnn_nvfuser_executor is not None:
    thunder_cudnn_nvfuser_fwd = partial(thunder_fwd, compile_fn=thunder_cudnn_nvfuser_executor)

# Executing with just the cuDNN layer norm executor
thunder_cudnn_layer_norm_fwd: None | Callable = None
if thunder_cudnn_layer_norm_executor is not None:
    thunder_cudnn_layer_norm_fwd = partial(thunder_fwd, compile_fn=thunder_cudnn_layer_norm_executor)

# Executing with the cuDNN layer norm executor and nvFuser
thunder_cudnn_layer_norm_nvfuser_fwd: None | Callable = None
if thunder_cudnn_layer_norm_nvfuser_executor is not None:
    thunder_cudnn_layer_norm_nvfuser_fwd = partial(thunder_fwd, compile_fn=thunder_cudnn_layer_norm_nvfuser_executor)


fwd_executors = (torch_fwd, torch_compile_fwd, thunder_fwd)
fwd_executor_ids = (
    "torch",
    "torch.compile",
    "thunder",
)

thunder_fwd_bwd_sdpa_torch_compile_nvfuser = partial(
    thunder_fwd_bwd, compile_fn=thunder_sdpa_torch_compile_nvfuser_executor
)

grad_executors = (
    *(
        (partial(thunder_fwd_bwd, compile_fn=thunder_cudnn_nvfuser_executor),)
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

apex_grad_executors = (thunder_apex_grad, thunder_apex_nvfuser_grad)
apex_grad_executors_ids = ("thunder+apex-grad", "thunder+apex+nvfuser-grad")

cudnn_fwd_executors = (thunder_cudnn_fwd, thunder_cudnn_nvfuser_fwd)
cudnn_fwd_executors_ids = ("thunder+cudnn", "thunder+cudnn+nvfuser")

cudnn_layernorm_fwd_executors = (thunder_cudnn_fwd, thunder_cudnn_nvfuser_fwd)
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
def test_nanogpt_gelu_fwd(benchmark, executor: Callable):
    gelu_bench: Benchmark = NanoGPTGeLUBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(gelu_bench)
    fn = executor(gelu_bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_nanogpt_gelu_grad(benchmark, executor: Callable):
    gelu_bench: Benchmark = NanoGPTGeLUBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(gelu_bench)
    fn = executor(gelu_bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


# TODO Improve cross entropy's fwd+bwd perf when using the PyTorch executor
#   See "torch.cross_entropy implementation has incorrect dtype metadata + bwd
#        is very slow"
@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
def test_nanogpt_cross_entropy_fwd(benchmark, executor: None | Callable):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTCrossEntropyBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


# TODO Improve cross entropy's fwd+bwd perf when using the PyTorch executor
#   See "torch.cross_entropy implementation has incorrect dtype metadata + bwd
#        is very slow"
@pytest.mark.parametrize(
    "executor,",
    (grad_executors + apex_grad_executors),
    ids=(grad_executors_ids + apex_grad_executors_ids),
)
def test_nanogpt_cross_entropy_grad(benchmark, executor: None | Callable):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTCrossEntropyBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


# TODO Improve cross entropy's fwd+bwd perf when using the PyTorch executor
#   See "torch.cross_entropy implementation has incorrect dtype metadata + bwd
#        is very slow"
@pytest.mark.parametrize(
    "executor,",
    (fwd_executors + cudnn_layernorm_fwd_executors),
    ids=(fwd_executor_ids + cudnn_layernorm_fwd_executors_ids),
)
def test_nanogpt_layer_norm_fwd(benchmark, executor: None | Callable):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTLayerNormBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,", (fwd_executors + cudnn_fwd_executors), ids=(fwd_executor_ids + cudnn_fwd_executors_ids)
)
def test_nanogpt_sdpa_fwd(benchmark, executor: None | Callable):
    if executor is None:
        pytest.skip("Executor is unavailable")

    bench: Benchmark = NanoGPTSDPABenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


# TODO Fix thunder-fwd-bwd+nvfuser
@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_nanogpt_sdpa_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTSDPABenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_llama2_7b_sdpa_grad(benchmark, executor: Callable):
    bench: Benchmark = LitGPTSDPABenchmark(
        config="Llama-2-7b-hf", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
def test_nanogpt_mlp_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTMLPBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=False
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_nanogpt_mlp_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTMLPBenchmark(
        config="gpt2-xl", device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


# NOTE The CSA module is linear -> sdpa -> dropout
@pytest.mark.parametrize(
    "executor,",
    fwd_executors,
    ids=fwd_executor_ids,
)
def test_nanogpt_csa_fwd(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTCSABenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=False,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


# NOTE The CSA module is linear -> sdpa -> dropout
@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_nanogpt_csa_grad(benchmark, executor: Callable):
    bench: Benchmark = NanoGPTCSABenchmark(
        config="gpt2-xl",
        device="cuda:0",
        dtype=thunder.bfloat16,
        requires_grad=True,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


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

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


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

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


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
        only_return_loss=False,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=1)


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
        only_return_loss=True,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=1)


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
        only_return_loss=False,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=1)


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
        only_return_loss=True,
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=1)


#
# llama benchmarks
#


@pytest.mark.parametrize(
    "executor,", (fwd_executors + cudnn_fwd_executors), ids=(fwd_executor_ids + cudnn_fwd_executors_ids)
)
def test_open_llama_7b_fwd(benchmark, executor: Callable):
    cfg: LitGPTConfig = LitGPTConfig.from_name("open_llama_7b")
    b = LitGPTBenchmark(cfg, device="cuda:0", dtype=torch.bfloat16, requires_grad=False)

    setup = make_setup(b)
    fn = executor(b)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,", (fwd_executors + cudnn_fwd_executors), ids=(fwd_executor_ids + cudnn_fwd_executors_ids)
)
def test_llama_2_7b_hf_fwd(benchmark, executor: Callable):
    cfg: LitGPTConfig = LitGPTConfig.from_name("Llama-2-7b-hf")
    b = LitGPTBenchmark(cfg, device="cuda:0", dtype=torch.bfloat16, requires_grad=False)

    setup = make_setup(b)
    fn = executor(b)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=1)


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

    setup = make_setup(b)
    fn = executor(b)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=20, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_llama2_mlp_7b_grad(benchmark, executor: Callable):
    bench: Benchmark = LlamaMLPBenchmark(
        config="Llama-2-7b-hf", batchdims=(16,), device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_llama2_causal_self_attention_7b_grad(benchmark, executor: Callable):
    bench: Benchmark = LitGPTCausalSelfAttentionBenchmark(
        config="Llama-2-7b-hf", batchdims=(16,), device="cuda:0", dtype=thunder.bfloat16, requires_grad=True
    )

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,",
    grad_executors,
    ids=grad_executors_ids,
)
def test_llama2_7b_rmsnorm_grad(benchmark, executor: Callable):
    from thunder.benchmarks import LlamaRMSNormBenchmark

    bench: Benchmark = LlamaRMSNormBenchmark(n_embd=4096, device="cuda:0", dtype=thunder.bfloat16, requires_grad=True)

    setup = make_setup(bench)
    fn = executor(bench)
    fn = wrap_for_benchmark(fn)

    benchmark.pedantic(fn, setup=setup, rounds=40, warmup_rounds=1)


@pytest.mark.parametrize(
    "executor,use_apex,",
    (
        (torch_fwd_bwd, False),
        (torchcompile_fwd_bwd, False),
        (thunder_fwd_bwd, False),
        (thunder_fwd_bwd_sdpa_torch_compile_nvfuser, False),
        (torch_fwd_bwd, True),
        (torchcompile_fwd_bwd, True),
    ),
    ids=(
        "torch",
        "torch.compile",
        "thunder",
        "thunder+nvfuser+torch.compile",
        "torch+apex",
        "torch.compile+apex",
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


#
# interpreter benchmarks
#


@pytest.mark.parametrize("executor,", (torch_fwd, interpreter_fwd), ids=("python", "interpreter"))
def test_interpreter_nanogpt_gpt2_fwd(benchmark, executor: Callable):
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

    benchmark.pedantic(fn, setup=setup, rounds=5, warmup_rounds=1)
