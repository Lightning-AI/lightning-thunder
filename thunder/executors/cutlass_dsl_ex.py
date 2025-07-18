from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec
import warnings
from typing import TYPE_CHECKING

from looseversion import LooseVersion
import torch

from thunder.core.transforms import get_grad, put_grad
from thunder.extend import register_executor, OperatorExecutor
from thunder.core.proxies import TensorProxy
import thunder.torch as ltorch

if TYPE_CHECKING:
    from thunder.core.dtypes import dtype as thunder_dtype


__all__ = [
    "cutlass_dsl_available",
    "cutlass_dsl_version",
    "cutlass_dsl_ex",
]


def cutlass_dsl_version() -> LooseVersion | None:
    """Returns ``cutlass`` version if available, otherwise, :obj:`None`"""

    if not torch.cuda.is_available():
        return None

    if find_spec("cutlass") is None:
        return None

    # First, check if it's cutlass>=4.0.0 which has the distribution name of nvidia-cutlass-dsl
    # ref: https://pypi.org/project/nvidia-cutlass-dsl/
    cutlass_python_version: LooseVersion
    nvidia_cutlass_dsl_version: str | None = None
    nvidia_cutlass_version: str | None = None
    try:
        nvidia_cutlass_dsl_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        try:
            # Then check if it's <4 which has the name of nvidia-cutlass
            # ref: https://pypi.org/project/nvidia-cutlass/
            nvidia_cutlass_version = version("nvidia-cutlass")
        except PackageNotFoundError:
            return None
        else:
            cutlass_python_version = LooseVersion(nvidia_cutlass_version)
    else:
        cutlass_python_version = LooseVersion(nvidia_cutlass_dsl_version)

    return cutlass_python_version


def required_cutlass_dsl_version() -> LooseVersion:
    return LooseVersion("4.0.0")


def cutlass_dsl_available() -> bool:
    ver = cutlass_dsl_version()

    if ver is None:
        return False

    if ver < required_cutlass_dsl_version():
        msg = f"Available cutlass version is out of date. Thunder requires 4.0.0, but found {ver}"
        warnings.warn(msg)
        return False

    return True


cutlass_dsl_ex = OperatorExecutor("cutlass_dsl", version=cutlass_dsl_version())
register_executor(cutlass_dsl_ex)


# Register [`quack`](https://github.com/Dao-AILab/quack) ops
if find_spec("quack") is not None:
    from quack.softmax import _softmax_fwd, _softmax_backward

    def quack_softmax_impl(a: torch.Tensor) -> torch.Tensor:
        return _softmax_fwd(a)

    def quack_softmax_meta(a: TensorProxy) -> TensorProxy:
        return TensorProxy(like=a)

    quack_softmax = cutlass_dsl_ex.register_operator(
        "cutlass_quack_softmax_forward",
        meta=quack_softmax_meta,
        fn=quack_softmax_impl,
    )

    def quack_softmax_backward(g: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return _softmax_backward(g, a)

    def quack_softmax_backward_meta(g: TensorProxy, a: TensorProxy) -> TensorProxy:
        return TensorProxy(like=g)

    quack_softmax_backward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_softmax_backward",
        meta=quack_softmax_backward_meta,
        fn=quack_softmax_backward,
    )

    def quack_softmax_checker(
        a: TensorProxy,
        /,
        dim: int,
        *,
        dtype: thunder_dtype | None = None,
    ) -> bool:
        last_dims = {-1, a.ndim - 1}
        allowed_dtypes = {None, a.dtype}
        return dim in last_dims and dtype in allowed_dtypes and torch.cuda.get_device_capability() in ((9, 0), (10, 0))

    def quack_softmax_transform(
        a: TensorProxy,
        /,
        dim: int,
        *,
        dtype: thunder_dtype | None = None,
    ) -> TensorProxy:
        return quack_softmax(a)

    def quack_softmax_grad(
        a: TensorProxy,
        /,
        dim: int,
        *,
        dtype: thunder_dtype | None = None,
    ) -> TensorProxy:
        fwd = quack_softmax(a)
        g = get_grad(fwd)
        a_grad = quack_softmax_backward(g, fwd)
        put_grad(a, a_grad)

        return fwd

    for ltorch_softmax in (ltorch._softmax, ltorch.softmax):
        cutlass_dsl_ex.register_implementation(
            ltorch_softmax,
            checker=quack_softmax_checker,
            execution_transform=quack_softmax_transform,
            grad_transform=quack_softmax_grad,
        )
