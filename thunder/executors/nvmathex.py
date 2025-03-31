import logging
import weakref
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable

import nvmath

import torch
from nvmath.linalg.advanced import Matmul, MatmulOptions
from torch.utils.benchmark import Timer

from thunder.core import prims
from thunder.core.proxies import TensorProxy
from thunder.dynamo.benchmark_utils import TorchProfileTimer
from thunder.extend import OperatorExecutor, register_executor

logger = logging.getLogger(__name__)

nvmath_matmul_ex: OperatorExecutor = OperatorExecutor("nvmath_matmul")
register_executor(nvmath_matmul_ex)


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    """
    A dataclass to store the shape, stride, dtype, and device index of a tensor for caching purposes.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device_index: int


def execute_nvmath_matmul(mm: Matmul, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Executes a matrix multiplication operation using nvmath for the given operands and matmul executor.

    Args:
        mm (Matmul): The matmul executor.
        a (torch.Tensor): The first operand.
        b (torch.Tensor): The second operand.

    Returns:
        torch.Tensor: The result of the matrix multiplication operation.
    """
    mm.reset_operands(a, b)  # This function has about 10 µs overhead
    return mm.execute()  # This function has about 75 µs overhead


def nvmath_or_pytorch_matmul(mm: Matmul) -> Callable:
    """
    Determines whether to use nvmath or PyTorch for matrix multiplication operations based on the benchmark results."

    Args:
        mm (Matmul): The nvmath matmul executor.

    Returns:
        Callable: The function to use for matrix multiplication operations.
    """
    mm.logger.info("= PYTORCH VS NVMATH AUTOTUNING PHASE =")  # Similar format as used in internal nvmath loggers
    a, b = (x.tensor for x in mm.operands)
    inner_timer = TorchProfileTimer()
    torch_timer = Timer(
        stmt="torch_matmul(a, b)", timer=inner_timer, globals={"torch_matmul": torch.matmul, "a": a, "b": b}
    )
    nvmath_timer = Timer(
        stmt="nvmath_matmul(a, b)",
        timer=inner_timer,
        globals={"nvmath_matmul": partial(execute_nvmath_matmul, mm), "a": a, "b": b},
    )
    torch_measurement = torch_timer.adaptive_autorange()
    nvmath_measurement = nvmath_timer.adaptive_autorange()
    mm.logger.info(
        f"Median speedup of nvmath over PyTorch: {torch_measurement.median / nvmath_measurement.median:.2f}x (PyTorch: {torch_measurement.median:.3e} s, nvmath: {nvmath_measurement.median:.3e} s)"
    )
    mm.logger.info(
        "Thunder will use PyTorch for matmul operations"
        if torch_measurement.median < nvmath_measurement.median
        else "Thunder will use nvmath for matmul operations"
        f"for the current session with the given operands ({a.shape=}, {a.stride()=}, {b.shape=}, {b.stride()=})"
    )
    return torch.matmul if torch_measurement.median < nvmath_measurement.median else partial(execute_nvmath_matmul, mm)


@lru_cache
def get_matmul_executor(a: TensorDescriptor, b: TensorDescriptor) -> Matmul | Callable:
    """
    Gets a matmul executor for the given operands.

    This function is cached to avoid creating multiple matmul executors for the same operands.

    Args:
        a (TensorDescriptor): The first operand.
        b (TensorDescriptor): The second operand.

    Returns:
        Matmul | Callable: The matmul executor or the function to use for matrix multiplication operations.
    """
    a = torch.empty_strided(a.shape, a.stride, dtype=a.dtype, device=f"cuda:{a.device_index}")
    b = torch.empty_strided(b.shape, b.stride, dtype=b.dtype, device=f"cuda:{b.device_index}")
    options = MatmulOptions(logger=logger)
    mm = Matmul(a, b, options=options)
    weakref.finalize(mm, mm.free)
    preferences = nvmath.linalg.advanced.MatmulPlanPreferences()
    mm.plan(preferences=preferences)
    mm.autotune(iterations=10)
    return nvmath_or_pytorch_matmul(mm)


def matmul_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Executes a matrix multiplication operation using nvmath or PyTorch based on the benchmark results for the given operands.

    This function is used directly in the execution traces and added to their python_ctx() dictionaries.

    Args:
        a (torch.Tensor): The first operand.
        b (torch.Tensor): The second operand.

    Returns:
        torch.Tensor: The result of the matrix multiplication operation.
    """
    # On a cache hit, this function has 1-2 µs overhead
    mm = get_matmul_executor(
        TensorDescriptor(a.shape, a.stride(), a.dtype, a.device.index),
        TensorDescriptor(b.shape, b.stride(), b.dtype, b.device.index),
    )
    return mm(a, b)


def matmul_checker(a: TensorProxy, b: TensorProxy) -> bool:
    """
    Checks if the given tensors are compatible for a matrix multiplication operation with nvmath.

    Args:
        a (TensorProxy): The first tensor.
        b (TensorProxy): The second tensor.

    Returns:
        bool: True if the tensors are compatible, False otherwise.
    """
    return a.device == b.device and a.device.type == "cuda"


def linear_checker(x: TensorProxy, w: TensorProxy, bias: TensorProxy | None) -> bool:
    """ "
    Checks if the given tensors are compatible for a linear operation with nvmath.

    Args:
        x (TensorProxy): The input tensor.
        w (TensorProxy): The weight tensor.
        bias (TensorProxy | None): The bias tensor.

    Returns:
        bool: True if the tensors are compatible, False otherwise.
    """
    if bias is not None:
        return False
    return matmul_checker(x, w)


def linear_impl(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """
    Executes a linear operation using nvmath for the given operands.

    Args:
        x (torch.Tensor): The input tensor.
        w (torch.Tensor): The weight tensor.
        bias (torch.Tensor | None): The bias tensor.

    Returns:
        torch.Tensor: The result of the linear operation.
    """
    # Transposing the weight tensor is necessary for the linear operation as
    # defined in PyTorch. Usually, it's a view operation, so it should not
    # result in a copy kernel.
    # The checker function ensures that the bias tensor is None.
    return matmul_impl(x, w.mT)


nvmath_matmul = nvmath_matmul_ex.register_operator("nvmath_matmul", like=prims.matmul, fn=matmul_impl)
nvmath_linear = nvmath_matmul_ex.register_operator("nvmath_linear", like=prims.linear, fn=linear_impl)
nvmath_matmul_ex.register_implementation(
    prims.matmul,
    nvmath_matmul,
    checker=matmul_checker,
)
nvmath_matmul_ex.register_implementation(
    prims.linear,
    nvmath_linear,
    checker=linear_checker,
)
