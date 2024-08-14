from collections.abc import Sequence
import math

import torch

import thunder
from thunder.core.proxies import TensorProxy, AnyProxy
from thunder.core.transforms import get_grad, put_grads
from thunder.executors.utils import Context, set_saved_tensors
from thunder.torch import TensorLike
from thunder.core.compile_data import get_compile_option
from thunder.executors.apexex import apex_ex


FUSED_NORMS_AVAILABLE = True
try:
    import fused_layer_norm_cuda
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction
except:
    FUSED_NORMS_AVAILABLE = False


def meta_fn(input: TensorLike, weight: TensorLike, normalized_shape: Sequence[int], eps: float, memory_efficient: bool):
    return TensorProxy(like=input)


# Symbol which will be used by lookaside.
fused_rms_norm = apex_ex.register_operator("fused_rms_norm", meta=meta_fn)


def meta_impl_fn(
    input: TensorLike, weight: TensorLike, normalized_shape: Sequence[int], eps: float, memory_efficient: bool
):
    output_or_input = TensorProxy(like=input)
    weight = TensorProxy(like=input, shape=normalized_shape)
    unnormalized_dims = len(input.shape) - len(normalized_shape)
    invvar = TensorProxy(like=input, shape=(math.prod(input.shape[:unnormalized_dims]),))
    return TensorProxy(like=input), (output_or_input, weight, invvar), AnyProxy(object())


def fused_rms_norm_impl(
    input: TensorLike, weight: TensorLike, normalized_shape: Sequence[int], eps: float, memory_efficient: bool
):
    ctx = Context()
    output = FusedRMSNormAffineMixedDtypesFunction.forward(ctx, input, weight, normalized_shape, eps, memory_efficient)
    return output, ctx.pop_saved_tensors(), ctx


fused_rms_norm_fwd = apex_ex.register_operator("fused_rms_norm_fwd", meta=meta_impl_fn, fn=fused_rms_norm_impl)


def fused_rms_norm_backward_meta(saved_tensors: Sequence[torch.Tensor], ctx: Context, g: TensorLike):
    # saved_tensors[0] - input or output
    # saved_tensors[1] - weight
    return TensorProxy(like=saved_tensors[0]), TensorProxy(like=saved_tensors[1])


def fused_rms_norm_backward_impl(saved_tensors: Sequence[torch.Tensor], ctx: Context, g: TensorLike):
    with set_saved_tensors(ctx, saved_tensors):
        return FusedRMSNormAffineMixedDtypesFunction.backward(ctx, g)[:2]


fused_rms_norm_backward = apex_ex.register_operator(
    "fused_rms_norm_backward", meta=fused_rms_norm_backward_meta, fn=fused_rms_norm_backward_impl
)


def fused_rms_norm_grad_rule(
    input: TensorLike, weight: TensorLike, normalized_shape: Sequence[int], eps: float, memory_efficient: bool
):
    output, saved_tensors, saved_meta = fused_rms_norm_fwd(input, weight, normalized_shape, eps, memory_efficient)
    g = get_grad(output)
    grad_input, grad_weight = fused_rms_norm_backward(saved_tensors, saved_meta, g)
    put_grads((input, weight), (grad_input, grad_weight))
    return output


def execution_tfms(
    input: TensorLike, weight: TensorLike, normalized_shape: Sequence[int], eps: float, memory_efficient: bool
):
    output, _, _ = fused_rms_norm_fwd(input, weight, normalized_shape, eps, memory_efficient)
    return output


def _fused_rms_norm_checker(
    input: TensorLike, weight: TensorLike, normalized_shape: Sequence[int], eps: float, memory_efficient: bool
):
    use_apex_fused_rms_norm = get_compile_option(
        "use_apex_fused_rms_norm", "Whether to enable `fused_rms_norm` from `apex_ex`. Defaults to `True`."
    )
    # We explicitly check for `False` as if the value is unspecified by user, `get_compile_option` returns `None` and `not None` is equal to True.
    if use_apex_fused_rms_norm == False:  # User explicitly disabled this.
        return False

    # use_apex_fused_rms_norm is `None` or `True`.
    return True


apex_ex.register_implementation(
    fused_rms_norm,
    execution_transform=execution_tfms,
    grad_transform=fused_rms_norm_grad_rule,
    checker=_fused_rms_norm_checker,
)
apex_ex.register_implementation(fused_rms_norm_backward, fused_rms_norm_backward)


# Register the lookaside.
def register_apex_fused_rms_norm() -> None:
    @thunder.core.jit_ext.register_general_jit_lookaside(FusedRMSNormAffineMixedDtypesFunction.forward)
    @thunder.core.jit_ext.interpreter_needs_wrap
    def rms_forward_lookaside(ctx, input, weight, normalized_shape, eps, memory_efficient=False):
        # This is the symbol we created.
        # NOTE - We don't use the `ctx` passed by PyTorch but instead use our Context to track saved_tensors and metadata.
        return fused_rms_norm(input, weight, normalized_shape, eps, memory_efficient)

    return None


if FUSED_NORMS_AVAILABLE:
    register_apex_fused_rms_norm()
