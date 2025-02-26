import transformer_engine.pytorch as te
from transformer_engine.pytorch.ops import BasicLinear

from thunder.core.proxies import TensorProxy
from thunder.core.transforms import get_grad, put_grads

from thunder.extend import OperatorExecutor, register_executor

import thunder.torch as ltorch

functional_te_ex = OperatorExecutor("functional_te")
register_executor(functional_te_ex)

recipe = te.fp8.get_default_fp8_recipe()
forward_recipe_state = te.fp8.RecipeState.create(recipe, mode=("forward"), num_quantizers=2,)
input_quantizer, weight_quantizer = forward_recipe_state.make_quantizers()
backward_recipe_state = te.fp8.RecipeState.create(recipe, mode=("backward"), num_quantizers=1,)
grad_output_quantizer, = backward_recipe_state.make_quantizers()

def _functional_te_checker(a, w, bias):
    # BasicLinear._functional API not support bias atm
    if bias:
        return False

    return True


def _linear_fwd_meta(a, w, bias):
    # if not requires_grad -> return return TensorProxy(like=a), None, None
    return TensorProxy(like=a), TensorProxy(like=a), TensorProxy(like=a)


def _linear_fwd_impl(a, w, bias):
    out, gemm_a, gemm_w = BasicLinear._functional_forward(
            input=a,
            weight=w,
            dtype=w.dtype,
            with_quantized_compute=True,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer
        )
    return out, gemm_a, gemm_w


_te_linear_fwd = functional_te_ex.register_operator("te_functional_fwd", meta=_linear_fwd_meta, fn=_linear_fwd_impl)


def _te_linear_fwd_wrapper(a, w, bias):
    out, _, _ = _te_linear_fwd(a, w, bias)
    return out


def _linear_bwd_meta(a, w, b):
    return TensorProxy(like=a), TensorProxy(like=w)

def _te_linear_bwd_wrapper(a, w, bias):
    primal, gemm_a, gemm_w = _te_linear_fwd(a, w, bias)
    grad_a, grad_w = _te_linear_bwd(get_grad(primal), gemm_a, gemm_w)
    put_grads((a, w), (grad_a, grad_w))
    return primal

def _linear_bwd_impl(grad_o, a, w):
    grad_input, grad_weight = BasicLinear._functional_backward(
        grad_output=grad_o,
        input=a,
        weight=w,
        dtype=w.dtype,
        with_quantized_compute=True,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
        grad_output_quantizer=grad_output_quantizer,
    )
    return grad_input, grad_weight

_te_linear_bwd = functional_te_ex.register_operator("te_functional_bwd", meta=_linear_bwd_meta, fn=_linear_bwd_impl)

functional_te_ex.register_implementation(
    ltorch.linear,
    checker=_functional_te_checker,
    execution_transform=_te_linear_fwd_wrapper,
    grad_transform=_te_linear_bwd_wrapper,
)
