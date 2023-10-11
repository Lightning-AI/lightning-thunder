from functools import partial
from typing import Callable, Optional
from collections.abc import Sequence

import torch

import thunder
import thunder.torch as ltorch
from thunder.benchmarks import (
    run_benchmark,
    default_torch_executor,
    default_torch_compile_executor,
    default_thunder_dynamic_strides_executor_no_grad,
    default_thunder_cudnn_executor,
    Benchmark,
    UserFacingBenchmarkMeta,
    BenchmarkArg,
)
from thunder.core import dtypes
from thunder.tests.lit_gpt_model import Config
from thunder.tests.lit_gpt_model import GPT
from thunder.tests.make_tensor import make_tensor, make_tensor_like

from lightning_utilities.core.imports import package_available

CUDNN_AVAILABLE = package_available("cudnn")


class LitGPTBenchmark(Benchmark, metaclass=UserFacingBenchmarkMeta):
    _args = (
        BenchmarkArg(
            name="batchdims",
            description="The shape (Sequence[int]) of input batch dimensions. The input will have innermost dimensions of (config.seq_len,). Default is (16,).",
        ),
        BenchmarkArg(
            name="indices_dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the input and targets. Default is thunder.int64.",
        ),
        BenchmarkArg(
            name="device",
            description="A device (str) to run on. Default is 'cuda'.",
        ),
        BenchmarkArg(
            name="dtype",
            description="The dtype (thunder.dtypes.dtype, torch.dtype, or str) of the model. Default is thunder.float32.",
        ),
        BenchmarkArg(
            name="requires_grad",
            description="Whether the model parameters require grad. Default is True.",
        ),
    )

    @classmethod
    @property
    def name(cls) -> str:
        return "litgpt"

    @classmethod
    @property
    def description(cls) -> str:
        return "LitGPT."

    @classmethod
    @property
    def args(cls) -> tuple[BenchmarkArg, ...]:
        return cls._args

    def __init__(
        self,
        config: Config,
        batchdims: Sequence[int] = (8,),
        indices_dtype: dtypes.dtype = thunder.int64,
        device: str = "cuda",
        dtype: dtypes.dtype = thunder.float32,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()

        self.config = config
        self.batchdims = batchdims
        self.indices_dtype = indices_dtype
        self.device = device
        self.dtype = dtype
        self.requires_grad: bool = requires_grad

        # Performs torch dtype conversions
        self.indices_tdtype: torch.dtype = ltorch.to_torch_dtype(self.indices_dtype)
        self.model_tdtype: torch.dtype = ltorch.to_torch_dtype(self.dtype)

        # Sets required benchmark parameters
        self.devices: list[str] = [device]

    def make_batch(self) -> tuple[list, dict]:
        make = partial(make_tensor, low=0, high=255, device=self.device, dtype=self.indices_tdtype, requires_grad=False)
        shape = self.batchdims + (self.config.block_size,)

        x = make(shape)
        return (x,), {}

    def fn(self) -> Callable:
        gpt = GPT(self.config).to(device=self.device, dtype=self.model_tdtype).requires_grad_(self.requires_grad)
        return gpt

    def postprocess_for_backward(self, output: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.requires_grad:
            return
        logits = output
        targets = make_tensor_like(logits)  # fake targets
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss


if __name__ == "__main__":
    for name in ("open_llama_7b", "Llama-2-7b-hf"):
        config = Config.from_name(name)
        # fwd-only benchmark
        b = LitGPTBenchmark(config, dtype=torch.bfloat16, requires_grad=False)

        print(f"torch {name}")
        run_benchmark(b, default_torch_executor)

        print(f"torch.compile {name}")
        run_benchmark(b, default_torch_compile_executor)

        print(f"thunder {name}")
        run_benchmark(b, default_thunder_dynamic_strides_executor_no_grad)

        if CUDNN_AVAILABLE:
            print(f"thunder + cudnn {name}")
            run_benchmark(b, default_thunder_cudnn_executor)
