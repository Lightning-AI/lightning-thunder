from collections.abc import Sequence

import torch

import thunder
from thunder.core.proxies import TensorProxy, AnyProxy
from thunder.core.transforms import get_grad, put_grads
from thunder.executors.utils import Context, set_saved_tensors


FUSED_NORMS_AVAILABLE = True
try:
    import fused_layer_norm_cuda
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction
except:
    FUSED_NORMS_AVAILABLE = False

import math
from thunder.extend import OperatorExecutor, register_executor

_apex_fused_rms_norm_ex = OperatorExecutor("apex_fused_rms_norm")
register_executor(_apex_fused_rms_norm_ex)


def meta_fn(input, weight, normalized_shape, eps, memory_efficient):
    return TensorProxy(like=input)


# Symbol which will be used by lookaside.
fused_rms_norm = _apex_fused_rms_norm_ex.register_operator("fused_rms_norm", meta=meta_fn)


def meta_impl_fn(input, weight, normalized_shape, eps, memory_efficient):
    output_or_input = TensorProxy(like=input)
    weight = TensorProxy(like=input, shape=normalized_shape)
    unnormalized_dims = len(input.shape) - len(normalized_shape)
    invvar = TensorProxy(like=input, shape=(math.prod(input.shape[:unnormalized_dims]),))
    return TensorProxy(like=input), (output_or_input, weight, invvar), AnyProxy(object())


def fused_rms_norm_impl(input, weight, normalized_shape, eps, memory_efficient=False):
    ctx = Context()
    output = FusedRMSNormAffineMixedDtypesFunction.forward(ctx, input, weight, normalized_shape, eps, memory_efficient)
    return output, ctx.pop_saved_tensors(), ctx


fused_rms_norm_fwd = _apex_fused_rms_norm_ex.register_operator(
    "fused_rms_norm_fwd", meta=meta_impl_fn, fn=fused_rms_norm_impl
)


def fused_rms_norm_backward_meta(saved_tensors: Sequence[torch.Tensor], ctx: Context, g: torch.Tensor):
    # saved_tensors[0] - input or output
    # saved_tensors[1] - weight
    return TensorProxy(like=saved_tensors[0]), TensorProxy(like=saved_tensors[1])


def fused_rms_norm_backward_impl(saved_tensors: Sequence[torch.Tensor], ctx: Context, g: torch.Tensor):
    with set_saved_tensors(ctx, saved_tensors):
        return FusedRMSNormAffineMixedDtypesFunction.backward(ctx, g)[:2]


fused_rms_norm_backward = _apex_fused_rms_norm_ex.register_operator(
    "fused_rms_norm_backward", meta=fused_rms_norm_backward_meta, fn=fused_rms_norm_backward_impl
)


def fused_rms_norm_grad_rule(input, weight, normalized_shape, eps, memory_efficient=False):
    output, saved_tensors, saved_meta = fused_rms_norm_fwd(input, weight, normalized_shape, eps, memory_efficient)
    g = get_grad(output)
    grad_input, grad_weight = fused_rms_norm_backward(saved_tensors, saved_meta, g)
    put_grads((input, weight), (grad_input, grad_weight))
    return output


def execution_tfms(input, weight, normalized_shape, eps, memory_efficient):
    output, _, _ = fused_rms_norm_fwd(input, weight, normalized_shape, eps, memory_efficient)
    return output


_apex_fused_rms_norm_ex.register_implementation(
    fused_rms_norm, execution_transform=execution_tfms, grad_transform=fused_rms_norm_grad_rule
)
_apex_fused_rms_norm_ex.register_implementation(fused_rms_norm_backward, fused_rms_norm_backward)


# Register the lookaside.
def register_apex_fused_rms_norm():
    @thunder.core.jit_ext.register_general_jit_lookaside(FusedRMSNormAffineMixedDtypesFunction.forward)
    @thunder.core.jit_ext.interpreter_needs_wrap
    def rms_forward_lookaside(ctx, input, weight, normalized_shape, eps, memory_efficient=False):
        # This is the symbol we created.
        return fused_rms_norm(input, weight, normalized_shape, eps, memory_efficient)

    return None


def get_apex_fused_rms_norm_ex() -> OperatorExecutor | None:
    if FUSED_NORMS_AVAILABLE:
        register_apex_fused_rms_norm()
        return _apex_fused_rms_norm_ex
    # If the relevant module is unavailable for import
    # then don't return the executor.
    return None
