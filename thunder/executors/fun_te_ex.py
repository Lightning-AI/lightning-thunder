import transformer_engine.pytorch as te
from transformer_engine.pytorch.ops import BasicLinear
from transformer_engine.pytorch.fp8 import get_fp8_max, _amax_and_scale_update

from thunder.core.proxies import AnyProxy, TensorProxy
from thunder.core.transforms import get_grad, put_grads

from thunder.extend import OperatorExecutor, register_executor

import thunder.torch as ltorch

functional_te_ex = OperatorExecutor("functional_te")
register_executor(functional_te_ex)

recipe = te.fp8.get_default_fp8_recipe()


def _functional_te_checker(a, w, bias):
    # BasicLinear._functional API not support bias atm
    if bias:
        return False

    return True


def _linear_fwd_meta(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer):
    # TODO if not requires_grad -> return return TensorProxy(like=a), None, None
    # Placeholder object
    o = object()
    return TensorProxy(like=a), TensorProxy(like=a), TensorProxy(like=a)


def _linear_fwd_impl(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer):
    out, gemm_a, gemm_w = BasicLinear._functional_forward(
        input=a,
        weight=w,
        dtype=w.dtype,
        with_quantized_compute=True,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
    )
    # Needs to be in the impl because otherwise it could be ran out of order.
    _amax_and_scale_update(
        forward_recipe_state.amax_history, forward_recipe_state.scale, get_fp8_max(recipe, "forward"), recipe
    )
    return out, gemm_a, gemm_w


_te_linear_fwd = functional_te_ex.register_operator("te_functional_fwd", meta=_linear_fwd_meta, fn=_linear_fwd_impl)


def _te_linear_fwd_wrapper(a, w, bias):
    forward_recipe_state, input_quantizer, weight_quantizer = _te_fp8_state(recipe, ("forward",), 2)
    out, _, _ = _te_linear_fwd(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer)
    return out


def _linear_bwd_meta(a, w, b, backward_recipe_state, input_quantizer, weight_quantizer, grad_output_quantizer):
    return TensorProxy(like=a), TensorProxy(like=w)


def _linear_bwd_impl(grad_o, a, w, input_quantizer, weight_quantizer, backward_recipe_state, grad_output_quantizer):
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
    _amax_and_scale_update(
        backward_recipe_state.amax_history, backward_recipe_state.scale, get_fp8_max(recipe, "backward"), recipe
    )
    return grad_input, grad_weight


_te_linear_bwd = functional_te_ex.register_operator("te_functional_bwd", meta=_linear_bwd_meta, fn=_linear_bwd_impl)


def _te_fp8_state_meta(recipe, mode, num_quantizers):
    o = object()
    return AnyProxy(o), *(AnyProxy(o) for _ in range(num_quantizers))


def _te_fp8_state_impl(recipe, mode, num_quantizers):
    recipe_state = te.fp8.RecipeState.create(
        recipe,
        mode=mode,
        num_quantizers=num_quantizers,
    )
    return recipe_state, *recipe_state.make_quantizers()


_te_fp8_state = functional_te_ex.register_operator("te_fp8_state", meta=_te_fp8_state_meta, fn=_te_fp8_state_impl)


def _te_linear_bwd_wrapper(a, w, bias):
    forward_recipe_state, input_quantizer, weight_quantizer = _te_fp8_state(recipe, ("forward",), 2)

    primal, gemm_a, gemm_w = _te_linear_fwd(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer)

    backward_recipe_state, grad_output_quantizer = _te_fp8_state(recipe, ("backward",), 1)

    grad_a, grad_w = _te_linear_bwd(
        get_grad(primal),
        gemm_a,
        gemm_w,
        input_quantizer,
        weight_quantizer,
        backward_recipe_state,
        grad_output_quantizer,
    )
    put_grads((a, w), (grad_a, grad_w))
    return primal


functional_te_ex.register_implementation(
    ltorch.linear,
    checker=_functional_te_checker,
    execution_transform=_te_linear_fwd_wrapper,
    grad_transform=_te_linear_bwd_wrapper,
)
