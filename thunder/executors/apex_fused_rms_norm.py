import thunder
from thunder.core.proxies import TensorProxy, CollectionProxy, AnyProxy
from thunder.core.transforms import register_grad, get_grad, put_grads
from collections.abc import Sequence
import torch
from contextlib import contextmanager

from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction
import math
from thunder.extend import OperatorExecutor, register_executor

apex_fused_rms_norm_ex = OperatorExecutor("apex_fused_rms_norm")
register_executor(apex_fused_rms_norm_ex)


class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def pop_saved_tensors(self):
        try:
            return self.saved_tensors
        finally:
            del self.saved_tensors


@contextmanager
def set_saved_tensors(ctx, saved_tensors):
    ctx.saved_tensors = saved_tensors
    try:
        yield
    finally:
        del ctx.saved_tensors


collection_cnt = 0


def meta_fn(input, weight, normalized_shape, eps, memory_efficient=False):
    output_or_input = TensorProxy(like=input)
    weight = TensorProxy(like=input, shape=normalized_shape)
    unnormalized_dims = len(input.shape) - len(normalized_shape)
    invvar = TensorProxy(like=input, shape=(math.prod(input.shape[:unnormalized_dims]),))
    if any(map(lambda x: x.requires_grad, (input, weight))):
        return (
            TensorProxy(like=input),
            (output_or_input, weight, invvar),
            AnyProxy(object(), name=f"fused_rms_norm_{collection_cnt}"),
        )
    return TensorProxy(like=input)


def fused_rms_norm_impl(input, weight, normalized_shape, eps, memory_efficient=False):
    ctx = Context()
    output = FusedRMSNormAffineMixedDtypesFunction.forward(ctx, input, weight, normalized_shape, eps, memory_efficient)
    if any(map(lambda x: x.requires_grad, (input, weight))):
        return output, ctx.pop_saved_tensors(), ctx
    return output


# fused_rms_norm = thunder.core.symbol.Symbol("fused_rms_norm_impl", id="fused_rms_norm", meta=meta_fn, is_prim=True, python_impl=fused_rms_norm_impl)
fused_rms_norm = apex_fused_rms_norm_ex.register_operator("fused_rms_norm", meta=meta_fn, fn=fused_rms_norm_impl)


def fused_rms_norm_backward_meta(saved_tensors: Sequence[torch.Tensor], ctx: Context, g: torch.Tensor):
    return TensorProxy(like=saved_tensors[0]), TensorProxy(like=saved_tensors[1])


def fused_rms_norm_backward_impl(saved_tensors: Sequence[torch.Tensor], ctx: Context, g: torch.Tensor):
    with set_saved_tensors(ctx, saved_tensors):
        return FusedRMSNormAffineMixedDtypesFunction.backward(ctx, g)[:2]


# fused_rms_norm_backward = thunder.core.symbol.Symbol("fused_rms_norm_backward_impl", id="fused_rms_norm_backward", meta=fused_rms_norm_backward_meta, is_prim=True, python_impl=fused_rms_norm_backward_impl)
fused_rms_norm_backward = apex_fused_rms_norm_ex.register_operator(
    "fused_rms_norm_backward", meta=fused_rms_norm_backward_meta, fn=fused_rms_norm_backward_impl
)


def fused_rms_norm_grad_rule(input, weight, normalized_shape, eps, memory_efficient=False):
    output, saved_tensors, ctx = fused_rms_norm(input, weight, normalized_shape, eps, memory_efficient)
    # saved_tensors = (input_output, weight, invvar)
    g = get_grad(output)
    grad_input, grad_weight = fused_rms_norm_backward(saved_tensors, ctx, g)
    put_grads((input, weight), (grad_input, grad_weight))
    return output


# register_grad(fused_rms_norm, fused_rms_norm_grad_rule)

apex_fused_rms_norm_ex.register_implementation(fused_rms_norm, fused_rms_norm, grad_transform=fused_rms_norm_grad_rule)


# Register the lookaside.
def register_apex_fused_rms_norm():

    @thunder.core.jit_ext.register_general_jit_lookaside(FusedRMSNormAffineMixedDtypesFunction.forward)
    @thunder.core.jit_ext.interpreter_needs_wrap
    def rms_forward_lookaside(ctx, input, weight, normalized_shape, eps, memory_efficient=False):
        # This is the symbol we created.
        return fused_rms_norm(input, weight, normalized_shape, eps, memory_efficient=memory_efficient)

    return None


register_apex_fused_rms_norm()
