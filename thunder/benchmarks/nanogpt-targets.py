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


gpt2_configs = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}


# TODO Port these benchmarks to pytest (and targets.py)
#   See https://github.com/Lightning-AI/lightning-thunder/issues/1404
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-xl",
        choices=tuple(gpt2_configs.keys()),
        help="parameter to specify model config",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="sequence length. usually 128 or 512")
    parser.add_argument("--skip-torch", action="store_true", help="set this to skip PyTorch things")
    parser.add_argument(
        "--bucket-sizes",
        nargs="+",
        type=float,
        default=[0, 25],
        help="parameters of `bucket_size_in_mb` for thunder.distributed.ddp",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="world size. This script will compare the number against available device count",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    args = parser.parse_args()
    # gpt2-xl config
    config = {"seq_len": args.seq_len}
    config.update(**gpt2_configs[args.model])
    config = NanoGPTConfig(**config)
    # fwd->bwd benchmark
    b = NanoGPTBenchmark(config, dtype=getattr(torch, args.dtype))

    if torch.distributed.is_available() and torch.cuda.device_count() > 1:
        # you can set the num of devices via `CUDA_VISIBLE_DEVICES` env var.
        world_size = args.world_size
        if world_size > torch.cuda.device_count():
            msg = f"{args.world_size=} must be <= {torch.cuda.device_count()=}"
            raise ValueError(msg)

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
