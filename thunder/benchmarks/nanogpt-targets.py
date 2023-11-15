import argparse

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
    get_default_thunder_ddp_dynamic_strides_executor,
    default_thunder_apex_executor,
    default_thunder_triton_executor,
)

# TODO Port these benchmarks to pytest (and targets.py)
#   See https://github.com/Lightning-AI/lightning-thunder/issues/1404
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--bucket-sizes", nargs="+", type=float, default=[0, 25])
    args = parser.parse_args()
    # gpt2-xl config
    config = NanoGPTConfig(n_layer=48, n_head=25, n_embd=1600, seq_len=args.seq_len)
    # fwd->bwd benchmark
    b = NanoGPTBenchmark(config, dtype=torch.bfloat16)

    if torch.distributed.is_available() and torch.cuda.device_count() > 1:
        # you can set the num of devices via `CUDA_VISIBLE_DEVICES` env var.
        world_size = torch.cuda.device_count()

        if not args.skip_torch:
            print("torch - DistributedDataParallel")
            run_multiprocess_benchmark(b, default_torch_ddp_executor, world_size=world_size)

            print("torch.compile - DistributedDataParallel")
            run_multiprocess_benchmark(b, default_torch_compile_ddp_executor, world_size=world_size)

        print(f"# bucket sizes: {args.bucket_sizes=}")
        for bucket_size_in_mb in args.bucket_sizes:
            print(f"thunder - ddp - {bucket_size_in_mb=}")
            run_multiprocess_benchmark(
                b,
                get_default_thunder_ddp_dynamic_strides_executor(bucket_size_in_mb),
                world_size=world_size,
            )
