from dataclasses import dataclass, field

from thunder.core.prims import OpTags
from thunder.core.proxies import AnyProxy, TensorProxy
from thunder.core.transforms import get_grad, put_grads
from thunder.extend import OperatorExecutor, register_executor
import thunder.torch as ltorch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.ops import BasicLinear
from transformer_engine.pytorch.fp8 import RecipeState, get_fp8_max, _amax_and_scale_update

functional_te_ex = OperatorExecutor("functional_te")
register_executor(functional_te_ex)


@dataclass
class FP8States:
    recipe = None
    forward_state_buffer: list[RecipeState] = field(default_factory=list)
    backward_state_buffer: list[RecipeState] = field(default_factory=list)


fun_te_states = FP8States()


def _functional_te_checker(a, w, bias):
    # BasicLinear._functional API not support bias atm
    if bias:
        return False

    return True


def _te_fp8_recipe_meta(recipe_name: str):
    return AnyProxy(object())


def _te_fp8_recipe_impl(recipe_name: str):
    if not fun_te_states.recipe:
        fun_te_states.recipe = te.fp8.get_default_fp8_recipe()

    return fun_te_states.recipe


_te_fp8_recipe = functional_te_ex.register_operator("te_fp8_recipe", meta=_te_fp8_recipe_meta, fn=_te_fp8_recipe_impl)


def _linear_fwd_meta(a: TensorProxy, w, bias, forward_recipe_state, input_quantizer, weight_quantizer):
    # TODO if not requires_grad -> return return TensorProxy(like=a), None, None
    if a.requires_grad:
        return TensorProxy(like=a), TensorProxy(like=a), TensorProxy(like=a)
    return TensorProxy(like=a), None, None


def _linear_fwd_impl(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer):
    out, gemm_a, gemm_w = BasicLinear._functional_forward(
        input=a,
        weight=w,
        dtype=w.dtype,
        with_quantized_compute=True,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
    )

    return out, gemm_a, gemm_w


_te_linear_fwd = functional_te_ex.register_operator("te_functional_fwd", meta=_linear_fwd_meta, fn=_linear_fwd_impl)


def _te_linear_fwd_wrapper(a, w, bias):
    recipe = _te_fp8_recipe("delayed")

    forward_recipe_state, input_quantizer, weight_quantizer = _te_fp8_state(recipe, "forward", 2)

    out, _, _ = _te_linear_fwd(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer)

    _te_fp8_syncronization(recipe, True)

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

    return grad_input, grad_weight


_te_linear_bwd = functional_te_ex.register_operator("te_functional_bwd", meta=_linear_bwd_meta, fn=_linear_bwd_impl)


def _te_fp8_state_meta(recipe, mode, num_quantizers):
    # Placeholder o
    o = object()
    return AnyProxy(o), *(AnyProxy(o) for _ in range(num_quantizers))


def _te_fp8_state_impl(recipe, mode, num_quantizers):
    recipe_state = te.fp8.RecipeState.create(
        recipe,
        mode=mode,
        num_quantizers=num_quantizers,
    )

    if mode == "forward":
        fun_te_states.forward_state_buffer += [recipe_state]
    else:
        fun_te_states.backward_state_buffer += [recipe_state]

    return recipe_state, *recipe_state.make_quantizers()


_te_fp8_state = functional_te_ex.register_operator("te_fp8_state", meta=_te_fp8_state_meta, fn=_te_fp8_state_impl)


# create every time and then resort to cse to eliminate all of them but the last one.
def _te_fp8_sync_meta(recipe: RecipeState, forward: bool):
    return None


def _te_fp8_sync_impl(recipe: RecipeState, forward: bool):
    state_buffer = fun_te_states.forward_state_buffer if forward else fun_te_states.backward_state_buffer
    for state in state_buffer:
        _amax_and_scale_update(state.amax_history, state.scale, get_fp8_max(recipe, forward), fun_te_states.recipe)
    return None


_te_fp8_syncronization = functional_te_ex.register_operator(
    "te_fp8_sync", meta=_te_fp8_sync_meta, fn=_te_fp8_sync_impl, tags=(OpTags.DONT_DCE, OpTags.CSE_KEEP_LAST)
)


def _te_linear_bwd_wrapper(a, w, bias):
    recipe = _te_fp8_recipe("delayed")

    forward_recipe_state, input_quantizer, weight_quantizer = _te_fp8_state(recipe, "forward", 2)

    primal, gemm_a, gemm_w = _te_linear_fwd(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer)

    _te_fp8_syncronization(recipe, True)

    backward_recipe_state, grad_output_quantizer = _te_fp8_state(recipe, "backward", 1)

    grad_a, grad_w = _te_linear_bwd(
        get_grad(primal),
        gemm_a,
        gemm_w,
        input_quantizer,
        weight_quantizer,
        backward_recipe_state,
        grad_output_quantizer,
    )

    _te_fp8_syncronization(recipe, False)

    put_grads((a, w), (grad_a, grad_w))
    return primal


functional_te_ex.register_implementation(
    ltorch.linear,
    checker=_functional_te_checker,
    execution_transform=_te_linear_fwd_wrapper,
    grad_transform=_te_linear_bwd_wrapper,
)
