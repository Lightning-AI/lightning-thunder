import torch

from thunder.benchmarks import (
    run_benchmark,
    NanoGPTConfig,
    NanoGPTBenchmark,
    default_torch_executor,
    default_torch_compile_executor,
    default_thunder_static_caching_executor,
    default_thunder_static_caching_executor_no_grad,
    default_thunder_apex_executor,
    default_thunder_triton_executor,
)

# nanoGPT fwd->bwd benchmark
# gpt2-xl config
config = NanoGPTConfig(n_layer=48, n_head=25, n_embd=1600)
b = NanoGPTBenchmark(config, dtype=torch.bfloat16)

print("torch")
run_benchmark(b, default_torch_executor)

print("torch.compile")
run_benchmark(b, default_torch_compile_executor)

print("thunder")
run_benchmark(b, default_thunder_static_caching_executor)

# nanoGPT fwd-only benchmark
b = NanoGPTBenchmark(config, dtype=torch.bfloat16, requires_grad=False)

print("torch")
run_benchmark(b, default_torch_executor)

print("torch.compile")
run_benchmark(b, default_torch_compile_executor)

print("thunder")
run_benchmark(b, default_thunder_static_caching_executor_no_grad)

# NOTE The apex and triton executors only support fwd at the moment
print("thunder+apex")
run_benchmark(b, default_thunder_apex_executor)

print("thunder+triton")
run_benchmark(b, default_thunder_triton_executor)
