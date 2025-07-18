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
    from typing import Any
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


def is_device_quack_compat() -> bool:
    return torch.cuda.get_device_capability() in ((9, 0), (10, 0))


# Register [`quack`](https://github.com/Dao-AILab/quack) ops
if find_spec("quack") is not None:
    # softmax
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
        return dim in last_dims and dtype in allowed_dtypes and is_device_quack_compat()

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

    # crossentropy
    from quack.cross_entropy import _cross_entropy, _cross_entropy_backward

    def quack_cross_entropy_forward_impl(
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return _cross_entropy(x, target, return_lse=False)

    def quack_cross_entropy_forward_meta(x: TensorProxy, target: TensorProxy) -> TensorProxy:
        return TensorProxy(like=x, shape=(x.shape[0],))

    quack_cross_entropy_forward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_cross_entropy_forward",
        meta=quack_cross_entropy_forward_meta,
        fn=quack_cross_entropy_forward_impl,
    )

    def quack_cross_entropy_backward_impl(
        x: torch.Tensor,
        target: torch.Tensor,
        grad: torch.Tensor,
        lse: torch.Tensor,
    ) -> torch.Tensor:
        return _cross_entropy_backward(x, target, grad, lse, False)

    def quack_cross_entropy_backward_meta(
        x: TensorProxy,
        target: TensorProxy,
        grad: TensorProxy,
        lse: TensorProxy,
    ) -> TensorProxy:
        return TensorProxy(like=grad)

    quack_cross_entropy_backward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_cross_entropy_backward",
        meta=quack_softmax_backward_meta,
        fn=quack_cross_entropy_backward_impl,
    )

    def quack_cross_entropy_checker(
        a: TensorProxy,
        /,
        target: TensorProxy,
        weight: TensorProxy | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> bool:
        if not is_device_quack_compat():
            return False
        if weight is not None:
            return False

        # Assert deprecated flags are not used
        for boolean_flag in (size_average, reduce):
            if boolean_flag is not None:
                return False

        if reduction != "none":
            return False

        if label_smoothing != 0.0:
            return False

        return True

    def quack_cross_entropy_transform(
        a: TensorProxy,
        /,
        target: TensorProxy,
        weight: TensorProxy | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> TensorProxy:
        return quack_cross_entropy_forward(a, target)

    def quack_cross_entropy_aug_forward_impl(
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _cross_entropy(x, target, return_lse=True)

    def quack_cross_entropy_aug_forward_meta(a: TensorProxy, target: TensorProxy) -> tuple[TensorProxy, TensorProxy]:
        return (TensorProxy(like=a, shape=(a.shape[0],)), TensorProxy(like=a, shape=(a.shape[0],)))

    quack_cross_entropy_aug_forward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_cross_entropy_aug_forward",
        meta=quack_cross_entropy_aug_forward_meta,
        fn=quack_cross_entropy_aug_forward_impl,
    )

    def quack_cross_entropy_grad(
        a: TensorProxy,
        /,
        target: TensorProxy,
        weight: TensorProxy | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> TensorProxy:
        fwd, lse = quack_cross_entropy_aug_forward(a, target)
        g = get_grad(fwd)
        a_grad = quack_cross_entropy_backward(a, target, g, lse)
        put_grad(a, a_grad)

        return fwd

    cutlass_dsl_ex.register_implementation(
        ltorch.cross_entropy,
        checker=quack_cross_entropy_checker,
        execution_transform=quack_cross_entropy_transform,
        grad_transform=quack_cross_entropy_grad,
    )
