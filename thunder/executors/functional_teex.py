from thunder.core.prims import OpTags
from thunder.core.proxies import AnyProxy, TensorProxy
from thunder.core.prims import get_grad
from thunder.core.transforms import put_grads
from thunder.extend import OperatorExecutor, register_executor
import thunder.torch as ltorch
from thunder.executors.transformer_engineex import _linear_checker

import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.ops import BasicLinear
from transformer_engine.pytorch.fp8 import (
    _amax_and_scale_update,
    get_fp8_max,
    get_default_fp8_recipe,
    Recipe,
    RecipeState,
)

functional_te_ex = OperatorExecutor("functional_te")
register_executor(functional_te_ex)


class FP8ExecutorState:
    layer_counter = 0
    recipe: Recipe = None
    states: dict[str, RecipeState] = {}
    quantizers: dict[int, list[Float8Quantizer]] = {}

    @classmethod
    def make_layer_id(cls):
        # NOTE: does not need to be a counter here, just return an unique number, can use uuids.
        layer_id = cls.layer_counter
        cls.layer_counter += 1
        return layer_id

    @classmethod
    def get_current_recipe(cls, name):
        if not cls.recipe:
            cls.recipe = get_default_fp8_recipe()
        return cls.recipe

    # TODO needs layer index + recipe + mode + process(ddp)
    @classmethod
    def get_key(cls, recipe: Recipe, layer_id: int, mode: str) -> str:
        return f"{layer_id}:{mode}:{str(recipe)}"

    @classmethod
    def get_state(cls, key: str) -> RecipeState | None:
        return cls.states.get(key, None)

    @classmethod
    def get_quantizers(cls, key: str) -> list[Float8Quantizer] | None:
        return cls.quantizers.get(key, None)

# TODO
def _functional_te_checker(*args):
    return _linear_checker(*args)


def _te_fp8_recipe_meta(recipe_name: str):
    return AnyProxy(object(), prefix="r")


def _te_fp8_recipe_impl(recipe_name: str):
    # can a trace include two different recipes? (shouldn't? design constraint consideration here)
    return FP8ExecutorState.get_current_recipe(recipe_name)


_te_fp8_recipe = functional_te_ex.register_operator("te_fp8_recipe", meta=_te_fp8_recipe_meta, fn=_te_fp8_recipe_impl)


def _te_fp8_quantizers_meta(layer_id: int, recipe_state: RecipeState, num_quantizers: int):
    return (*(AnyProxy(object(), prefix="q") for _ in range(num_quantizers)),)


def _te_fp8_quantizers_impl(layer_id: int, recipe_state: RecipeState, num_quantizers: int):
    # TODO verify that the recipe and mode are actually set
    key = FP8ExecutorState.get_key(recipe_state.recipe, layer_id, recipe_state.mode)

    quantizers = FP8ExecutorState.get_quantizers(key)

    if quantizers:
        return quantizers

    quantizers = recipe_state.make_quantizers()
    FP8ExecutorState.quantizers[key] = quantizers

    return quantizers


_te_fp8_quantizers = functional_te_ex.register_operator(
    "te_fp8_quantizers", meta=_te_fp8_quantizers_meta, fn=_te_fp8_quantizers_impl
)


def _te_fp8_state_meta(state_idx: int, recipe, mode: str, num_quantizers: int, /):
    # Placeholder o
    o = object()
    return AnyProxy(o, prefix="s")


def _te_fp8_state_impl(layer_id, recipe, mode, num_quantizers):
    key = FP8ExecutorState.get_key(recipe, layer_id, mode)
    state = FP8ExecutorState.get_state(key)
    if state:
        return state

    # mode is needed to get the correct dtypes inside for computation(setup quantizers)
    recipe_state = te.fp8.RecipeState.create(
        recipe,
        mode=mode,
        num_quantizers=num_quantizers,
    )

    FP8ExecutorState.states[key] = recipe_state

    return recipe_state


_te_fp8_state = functional_te_ex.register_operator("te_fp8_state", meta=_te_fp8_state_meta, fn=_te_fp8_state_impl)


def _linear_fwd_meta(
    a: TensorProxy,
    w: TensorProxy,
    bias: TensorProxy | None,
    forward_recipe_state: RecipeState,
    input_quantizer: Float8Quantizer,
    weight_quantizer: Float8Quantizer,
):
    return TensorProxy(like=a), TensorProxy(like=a), TensorProxy(like=w)


def _linear_fwd_impl(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer):
    out, gemm_a, gemm_w = BasicLinear._functional_forward(
        input=a,
        weight=w,
        with_quantized_compute=True,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
        output_quantizer=None,  # return out in original dtype (w.dtype)
    )
    return out, gemm_a, gemm_w


_te_linear_fwd = functional_te_ex.register_operator("te_functional_fwd", meta=_linear_fwd_meta, fn=_linear_fwd_impl)


def _te_linear_execution_transform(a, w, bias):
    recipe = _te_fp8_recipe("delayed")

    # can get rid of this by using a uniquie number generated here and then saved as constant(automatically?)
    layer_id = FP8ExecutorState.make_layer_id()

    forward_recipe_state = _te_fp8_state(layer_id, recipe, "forward", 2)

    input_quantizer, weight_quantizer = _te_fp8_quantizers(layer_id, forward_recipe_state, 2)

    out, _, _ = _te_linear_fwd(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer)

    return out


def _linear_bwd_meta(grad_o, a, w, input_quantizer, weight_quantizer, grad_output_quantizer):
    return TensorProxy(like=a), TensorProxy(like=w)


def _linear_bwd_impl(grad_o, a, w, input_quantizer, weight_quantizer, grad_output_quantizer):
    grad_input, grad_weight = BasicLinear._functional_backward(
        grad_output=grad_o,
        input=a,
        weight=w,
        with_quantized_compute=True,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
        grad_output_quantizer=grad_output_quantizer,
    )

    return grad_input, grad_weight


_te_linear_bwd = functional_te_ex.register_operator("te_functional_bwd", meta=_linear_bwd_meta, fn=_linear_bwd_impl)


def _te_linear_grad_transform(a, w, bias):
    recipe = _te_fp8_recipe("delayed")

    layer_id = FP8ExecutorState.make_layer_id()

    forward_recipe_state = _te_fp8_state(layer_id, recipe, "forward", 2)

    input_quantizer, weight_quantizer = _te_fp8_quantizers(layer_id, forward_recipe_state, 2)

    primal, gemm_a, gemm_w = _te_linear_fwd(a, w, bias, forward_recipe_state, input_quantizer, weight_quantizer)

    backward_recipe_state = _te_fp8_state(layer_id, recipe, "backward", 1)

    (grad_output_quantizer,) = _te_fp8_quantizers(layer_id, backward_recipe_state, 1)

    grad_out = get_grad(primal)

    grad_a, grad_w = _te_linear_bwd(
        grad_out,
        gemm_a,
        gemm_w,
        input_quantizer,
        weight_quantizer,
        grad_output_quantizer,
    )

    put_grads((a, w), (grad_a, grad_w))

    return primal


functional_te_ex.register_implementation(
    ltorch.linear,
    checker=_functional_te_checker,
    execution_transform=_te_linear_execution_transform,
    grad_transform=_te_linear_grad_transform,
)


def _te_fp8_sync_meta(recipe, *states, forward: bool):
    return None


def _te_fp8_sync_impl(recipe, *states, forward: bool):
    # TODO use the key here so that it's forward compatible with an amax history buffer in the future.
    for state in states:
        _amax_and_scale_update(state.amax_history, state.scale, get_fp8_max(recipe, forward), recipe)
    return None


_te_fp8_syncronization = functional_te_ex.register_operator(
    "te_fp8_sync",
    meta=_te_fp8_sync_meta,
    fn=_te_fp8_sync_impl,
    tags=(OpTags.DONT_DCE,),
)
