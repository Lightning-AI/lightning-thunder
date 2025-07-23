from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec
import warnings
from typing import TYPE_CHECKING

from looseversion import LooseVersion
import torch

from thunder.core import dtypes
from thunder.core.transforms import get_grad, put_grad
from thunder.extend import register_executor, OperatorExecutor
from thunder.core.proxies import TensorProxy
import thunder.torch as ltorch

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number
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


# NOTE: This constraint comes from https://github.com/Dao-AILab/quack/blob/59631e98/quack/reduction_base.py#L35-L38
def is_last_dim_divisible(dtype: dtypes.dtype, last_dim_size: int) -> bool:
    return last_dim_size % (128 // 8 // dtype.bytes) == 0


# Register [`quack`](https://github.com/Dao-AILab/quack) ops
if find_spec("quack") is not None:
    # softmax
    from quack.softmax import _softmax_fwd, _softmax_backward

    def quack_softmax_impl(a: torch.Tensor) -> torch.Tensor:
        original_shape = a.shape
        if requires_reshpae := a.ndim > 2:
            a = a.view(-1, original_shape[-1])
        ret = _softmax_fwd(a)
        if requires_reshpae:
            ret = ret.view(original_shape)
        return ret

    def quack_softmax_meta(a: TensorProxy) -> TensorProxy:
        return TensorProxy(like=a)

    quack_softmax = cutlass_dsl_ex.register_operator(
        "cutlass_quack_softmax_forward",
        meta=quack_softmax_meta,
        fn=quack_softmax_impl,
    )

    def quack_softmax_backward(g: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        original_shape = g.shape
        if requires_reshape := g.ndim > 2:
            g = g.view(-1, original_shape[-1])
            a = a.view(-1, original_shape[-1])
        ret = _softmax_backward(g, a)
        if requires_reshape:
            ret = ret.view(original_shape)
        return ret

    def quack_softmax_backward_meta(g: TensorProxy, a: TensorProxy) -> TensorProxy:
        return TensorProxy(like=g)

    quack_softmax_backward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_softmax_backward",
        meta=quack_softmax_backward_meta,
        fn=quack_softmax_backward,
    )

    # Ref: https://github.com/Dao-AILab/quack/blob/3ce89a24/quack/softmax.py#L189-L198
    def quack_softmax_checker(
        a: TensorProxy,
        /,
        dim: int,
        *,
        dtype: thunder_dtype | None = None,
    ) -> bool:
        last_dims = {-1, a.ndim - 1}
        allowed_dtypes = {None, a.dtype}
        return (
            dim in last_dims
            and dtype in allowed_dtypes
            and a.dtype in {dtypes.float16, dtypes.bfloat16, dtypes.float32}
            and is_device_quack_compat()
            and is_last_dim_divisible(a.dtype, a.shape[-1])
        )

    def quack_softmax_transform(
        a: TensorProxy,
        /,
        dim: int,
        *,
        dtype: thunder_dtype | None = None,
    ) -> TensorProxy:
        return quack_softmax(a)

    # NOTE: Softmax backward doesn't look functioning as follows:
    #     def _engine_run_backward(
    #         t_outputs: Sequence[Union[torch.Tensor, GradientEdge]],
    #         *args: Any,
    #         **kwargs: Any,
    #     ) -> tuple[torch.Tensor, ...]:
    #         attach_logging_hooks = log.getEffectiveLevel() <= logging.DEBUG
    #         if attach_logging_hooks:
    #             unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    #         try:
    # >           return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    #                 t_outputs, *args, **kwargs
    #             )  # Calls into the C++ engine to run the backward pass
    # E           RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    #
    # /pytorch/torch/autograd/graph.py:829: RuntimeError
    # def quack_softmax_grad(
    #     a: TensorProxy,
    #     /,
    #     dim: int,
    #     *,
    #     dtype: thunder_dtype | None = None,
    # ) -> TensorProxy:
    #     fwd = quack_softmax(a)
    #     g = get_grad(fwd)
    #     a_grad = quack_softmax_backward(g, fwd)
    #     put_grad(a, a_grad)

    #     return fwd

    for ltorch_softmax in (ltorch._softmax, ltorch.softmax):
        cutlass_dsl_ex.register_implementation(
            ltorch_softmax,
            checker=quack_softmax_checker,
            execution_transform=quack_softmax_transform,
            # grad_transform=quack_softmax_grad,
        )

    # crossentropy
    from quack.cross_entropy import _cross_entropy, _cross_entropy_backward

    def quack_cross_entropy_forward_impl(
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return _cross_entropy(x, target, return_lse=False)

    def quack_cross_entropy_forward_meta(x: TensorProxy, target: TensorProxy) -> TensorProxy:
        return TensorProxy(like=x, shape=(x.shape[0],), dtype=dtypes.float32)

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
        return TensorProxy(like=x)

    quack_cross_entropy_backward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_cross_entropy_backward",
        meta=quack_cross_entropy_backward_meta,
        fn=quack_cross_entropy_backward_impl,
    )

    # Ref: https://github.com/Dao-AILab/quack/blob/3ce89a24/quack/cross_entropy.py#L216-L239
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

        if (
            a.ndim != 2
            or a.dtype not in {dtypes.float16, dtypes.bfloat16, dtypes.float32}
            and target.ndim == 1
            and target.dytpe in {dtypes.int32, dtypes.int64}
        ):
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
        return (
            TensorProxy(like=a, shape=(a.shape[0],), dtype=dtypes.float32),
            TensorProxy(like=a, shape=(a.shape[0],), dtype=dtypes.float32),
        )

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

    # layernorm (only forward as of https://github.com/Dao-AILab/quack/commit/3ce89a24)
    from quack.layernorm import layernorm

    def quack_layer_norm_forward_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        return_rstd: bool,
        return_mean: bool,
    ) -> torch.Tensor:
        original_shape = x.shape
        if requires_reshape := x.ndim > 2:
            x = x.view(-1, original_shape[-1])
        ret = layernorm(x, weight, eps, return_rstd=return_rstd, return_mean=return_mean)
        if requires_reshape:
            ret = ret.view(original_shape)
        return ret

    def quack_layer_norm_forward_meta(
        x: TensorProxy,
        weight: TensorProxy,
        eps: float,
        return_rstd: bool,
        return_mean: bool,
    ) -> TensorProxy:
        return TensorProxy(like=x)

    quack_layer_norm_forward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_layer_norm_forward",
        meta=quack_layer_norm_forward_meta,
        fn=quack_layer_norm_forward_impl,
    )

    # Ref: https://github.com/Dao-AILab/quack/blob/3ce89a24/quack/layernorm.py#L252-L278
    def quack_layer_norm_checker(
        a: TensorProxy,
        /,
        normalized_shape: Sequence[int],
        weight: TensorProxy | None = None,
        bias: TensorProxy | None = None,
        eps: Number = 1e-5,
    ) -> bool:
        if (
            a.dtype not in {dtypes.float16, dtypes.bfloat16, dtypes.float32}
            or weight.ndim != 1
            or a.shape[-1] != weight.shape[0]
            or weight.dtype not in {dtypes.float32}
        ):
            return False
        return is_device_quack_compat()

    def quack_layer_norm_transform(
        a: TensorProxy,
        /,
        normalized_shape: Sequence[int],
        weight: TensorProxy | None = None,
        bias: TensorProxy | None = None,
        eps: Number = 1e-5,
    ) -> TensorProxy:
        return quack_layer_norm_forward(a, weight, eps, return_rstd=False, return_mean=False)

    cutlass_dsl_ex.register_implementation(
        ltorch.layer_norm,
        checker=quack_layer_norm_checker,
        execution_transform=quack_layer_norm_transform,
    )

    # rmsnorm
    from quack.rmsnorm import _rmsnorm_fwd, _rmsnorm_backward

    def quack_rms_norm_forward_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        original_shape = x.shape
        if requires_reshape := x.ndim > 2:
            x = x.view(-1, original_shape[-1])
        ret = _rmsnorm_fwd(x, weight, eps, return_rstd=False)
        if requires_reshape:
            ret = ret.view(original_shape)
        return ret

    def quack_rms_norm_forward_meta(
        x: TensorProxy,
        weight: TensorProxy,
        eps: float = 1e-6,
    ) -> TensorProxy:
        return TensorProxy(like=x)

    quack_rms_norm_forward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_rms_norm_forward",
        meta=quack_rms_norm_forward_meta,
        fn=quack_rms_norm_forward_impl,
    )

    def quack_rms_norm_backward_impl(
        grad: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        rstd: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = grad.shape
        if requires_reshape := grad.ndim > 2:
            grad = grad.view(-1, original_shape[-1])
            x = x.view(-1, original_shape[-1])
        ret = _rmsnorm_backward(x, weight, grad, rstd)
        if requires_reshape:
            ret = ret.view(original_shape)
        return ret

    def quack_rms_norm_backward_meta(
        grad: TensorProxy,
        x: TensorProxy,
        weight: TensorProxy,
        rstd: TensorProxy,
    ) -> TensorProxy:
        return TensorProxy(like=grad)

    quack_rms_norm_backward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_rms_norm_backward",
        meta=quack_rms_norm_backward_meta,
        fn=quack_rms_norm_backward_impl,
    )

    # Ref: https://github.com/Dao-AILab/quack/blob/3ce89a24/quack/rmsnorm.py#L231-L261
    def quack_rms_norm_checker(
        a: TensorProxy,
        /,
        normalized_shape: Sequence[int],
        weight: TensorProxy | None = None,
        eps: float | None = None,
    ) -> bool:
        if (
            weight.ndim != 1
            or a.shape[-1] != weight.shape[0]
            or a.dtype not in {dtypes.float16, dtypes.bfloat16, dtypes.float32}
            or weight.dtype not in {dtypes.float16, dtypes.bfloat16, dtypes.float32}
        ):
            return False
        return weight is not None and is_device_quack_compat() and is_last_dim_divisible(a.dtype, a.shape[-1])

    def quack_rms_norm_aug_forward_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        if requires_reshape := x.ndim > 2:
            x = x.view(-1, original_shape[-1])
        fwd, rstd = _rmsnorm_fwd(x, weight, eps, return_rstd=True)
        if requires_reshape:
            fwd = fwd.view(original_shape)
        return fwd, rstd

    def quack_rms_norm_aug_forward_meta(
        x: TensorProxy,
        weight: TensorProxy,
        eps: float = 1e-6,
    ) -> tuple[TensorProxy, TensorProxy]:
        return (TensorProxy(like=x), TensorProxy(like=x, shape=(x.shape[0],), dtype=dtypes.float32))

    quack_rms_norm_aug_forward = cutlass_dsl_ex.register_operator(
        "cutlass_quack_rms_norm_aug_forward",
        meta=quack_rms_norm_aug_forward_meta,
        fn=quack_rms_norm_aug_forward_impl,
    )

    def quack_rms_norm_transform(
        a: TensorProxy,
        /,
        normalized_shape: Sequence[int],
        weight: TensorProxy | None = None,
        eps: float | None = None,
    ) -> TensorProxy:
        if eps is None:
            eps = 1e-6
        return quack_rms_norm_aug_forward(a, weight, eps)[0]

    # NOTE: The backward looks not functioning:
    #     def _engine_run_backward(
    #         t_outputs: Sequence[Union[torch.Tensor, GradientEdge]],
    #         *args: Any,
    #         **kwargs: Any,
    #     ) -> tuple[torch.Tensor, ...]:
    #         attach_logging_hooks = log.getEffectiveLevel() <= logging.DEBUG
    #         if attach_logging_hooks:
    #             unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    #         try:
    # >           return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    #                 t_outputs, *args, **kwargs
    #             )  # Calls into the C++ engine to run the backward pass
    # E           RuntimeError: One of the differentiated Tensors does not require grad
    #
    # /pytorch/torch/autograd/graph.py:829: RuntimeError

    def quack_rms_norm_grad(
        a: TensorProxy,
        /,
        normalized_shape: Sequence[int],
        weight: TensorProxy | None = None,
        eps: float | None = None,
    ) -> TensorProxy:
        if eps is None:
            eps = 1e-6
        fwd, rstd = quack_rms_norm_aug_forward(a, weight, eps)

        grad = get_grad(fwd)
        a_grad = quack_rms_norm_backward(grad, a, weight, rstd)
        put_grad(a, a_grad)
        return fwd

    cutlass_dsl_ex.register_implementation(
        ltorch.rms_norm,
        checker=quack_rms_norm_checker,
        execution_transform=quack_rms_norm_transform,
        # grad_transform=quack_rms_norm_grad,
    )
