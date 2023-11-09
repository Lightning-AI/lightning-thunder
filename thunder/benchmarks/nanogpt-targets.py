import torch

from thunder.benchmarks import (
    run_benchmark,
    run_multiprocess_benchmark,
    NanoGPTConfig,
    NanoGPTBenchmark,
    torch_executor,
    torch_compile_executor,
    thunder_nvfuser_executor,
    default_torch_ddp_executor,
    default_torch_compile_ddp_executor,
    default_thunder_dynamic_strides_executor_no_grad,
    default_thunder_ddp_dynamic_strides_executor,
    default_thunder_apex_executor,
    default_thunder_triton_executor,
)

# TODO Port these benchmarks to pytest (and targets.py)
#   See https://github.com/Lightning-AI/lightning-thunder/issues/1404
if __name__ == "__main__":
    # gpt2-xl config
    config = NanoGPTConfig(n_layer=48, n_head=25, n_embd=1600)
    # fwd->bwd benchmark
    b = NanoGPTBenchmark(config, dtype=torch.bfloat16)

    if torch.distributed.is_available() and torch.cuda.device_count() > 1:
        # you can set the num of devices via `CUDA_VISIBLE_DEVICES` env var.
        world_size = torch.cuda.device_count()

        print("torch - DistributedDataParallel")
        run_multiprocess_benchmark(b, default_torch_ddp_executor, world_size=2)

        print("torch.compile - DistributedDataParallel")
        run_multiprocess_benchmark(b, default_torch_compile_ddp_executor, world_size=2)

        print("thunder - ddp")
        run_multiprocess_benchmark(b, default_thunder_ddp_dynamic_strides_executor, world_size=2)
