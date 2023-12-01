from functools import partial
from typing import Optional
from collections.abc import Callable
from collections.abc import Sequence

import torch

import thunder
import thunder.torch as ltorch
from thunder.benchmarks import (
    run_benchmark,
    torch_executor,
    torch_compile_executor,
    default_thunder_dynamic_strides_executor_no_grad,
    default_thunder_cudnn_executor,
    LitGPTBenchmark,
)

from thunder.tests.lit_gpt_model import Config

from lightning_utilities.core.imports import package_available

CUDNN_AVAILABLE = package_available("cudnn")

if __name__ == "__main__":
    for name in ("open_llama_7b", "Llama-2-7b-hf"):
        config = Config.from_name(name)
        # fwd-only benchmark
        b = LitGPTBenchmark(config, dtype=torch.bfloat16, requires_grad=False)

        print(f"torch {name}")
        run_benchmark(b, torch_executor)

        print(f"torch.compile {name}")
        run_benchmark(b, torch_compile_executor)

        print(f"thunder {name}")
        run_benchmark(b, default_thunder_dynamic_strides_executor_no_grad)

        if CUDNN_AVAILABLE:
            print(f"thunder + cudnn {name}")
            run_benchmark(b, default_thunder_cudnn_executor)
