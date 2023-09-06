import torch

from thunder.benchmarks import (
    run_benchmark,
    run_multiprocess_benchmark,
    NanoGPTConfig,
    NanoGPTBenchmark,
    default_torch_executor,
    default_torch_compile_executor,
    default_torch_ddp_executor,
    default_torch_compile_ddp_executor,
    default_thunder_static_caching_executor,
    default_thunder_static_caching_executor_no_grad,
    default_thunder_ddp_static_caching_executor,
    default_thunder_apex_executor,
    default_thunder_triton_executor,
)


if __name__ == "__main__":
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

    if torch.distributed.is_available() and torch.cuda.device_count() > 1:
        # you can set the num of devices via `CUDA_VISIBLE_DEVICES` env var.
        world_size = torch.cuda.device_count()

        print("torch - DistributedDataParallel")
        run_multiprocess_benchmark(b, default_torch_ddp_executor, world_size=2)

        print("torch.compile - DistributedDataParallel")
        run_multiprocess_benchmark(b, default_torch_compile_ddp_executor, world_size=2)

        print("thunder - ddp")
        run_multiprocess_benchmark(b, default_thunder_ddp_static_caching_executor, world_size=2)

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
