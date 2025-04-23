from enum import auto, Enum
import importlib
import warnings

from thunder.core.symbol import Symbol

from thunder.core.prims import get_grad, put_grad
from thunder.core.proxies import AnyProxy, TensorProxy, IntegerProxy
from thunder.core.vjp_utils import disable_caching_split_forward_and_backward

from thunder.extend import OperatorExecutor, register_executor
from thunder.executors.transformer_engineex import _linear_checker

import thunder.torch as ltorch

if importlib.util.find_spec("transformer_engine"):
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.ops import BasicLinear
    from transformer_engine.pytorch.fp8 import (
        _amax_and_scale_update,
        get_fp8_max,
        Recipe,
        RecipeState,
        FP8GlobalStateManager,
    )
else:
    warnings.warn("transformer_engine module not found!")


class StatefulExecutor(OperatorExecutor):
    def __init__(self, name, *, version=None):
        super().__init__(name, version=version)
        self.state_count: int = 0

    def register_stateful_operator(self, base_name: str, state_class, *, meta):
        def register_state(*args, **kwargs):
            state_id = self.state_count
            name = f"{base_name}_{state_id}"

            def bind_state(bsym):
                bsym._call_ctx = {name: state_class()}

            sym = self.register_operator(name, meta=meta, bind_postprocess=bind_state)

            self.state_count += 1
            return sym(*args, *kwargs)

        return register_state

    def get_grad_transform(self, sym: Symbol):
        grad_transform = super().get_grad_transform(sym)
        # Always disable cache for stateful grad transform
        return disable_caching_split_forward_and_backward(grad_transform)


functional_te_ex = StatefulExecutor("functional_te")
register_executor(functional_te_ex)


def _functional_te_checker(a, w, /, bias):
    return _linear_checker(a, w, bias)


def _te_fp8_recipe_meta() -> AnyProxy:
    return AnyProxy(None, prefix="r")


# TODO add new recipe types.
class TE_RECIPE_TYPE(Enum):
    DELAYED = auto()
    MXFP8 = auto()


def _te_get_recipe_type_meta(recipe: AnyProxy):
    return IntegerProxy()


def _te_get_recipe_type_impl(recipe: Recipe):
    if recipe.delayed():
        return TE_RECIPE_TYPE.DELAYED
    if recipe.mxfp8():
        return TE_RECIPE_TYPE.MXFP8


_te_get_recipe_type = functional_te_ex.register_operator(
    "te_get_recipe_type", meta=_te_get_recipe_type_meta, fn=_te_get_recipe_type_impl
)


class TERecipe:
    def __init__(self):
        self.fp8_recipe = None

    def __call__(self) -> list[Recipe]:
        # If the Thunder trace is not ran inside an fp8_autocast region then
        # use the default recipe.
        if not FP8GlobalStateManager.FP8_RECIPE:
            if not self.fp8_recipe:
                self.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
            return self.fp8_recipe
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        return recipe


_get_te_fp8_recipe = functional_te_ex.register_stateful_operator(
    "get_te_fp8_recipe", meta=_te_fp8_recipe_meta, state_class=TERecipe
)


def _get_te_fp8_quantizers_meta(recipe_state: RecipeState, num_quantizers: int):
    # num_quantizers can be taken from recipe state, but not at trace time.
    # Therefore it's needed here to inform the meta.
    return (*(AnyProxy(None, prefix="q") for _ in range(num_quantizers)),)


class TEQuantizerState:
    def __init__(self):
        self.quantizers = None

    def __call__(self, recipe_state: RecipeState, num_quantizers: int) -> list[Float8Quantizer]:
        if self.quantizers:
            return self.quantizers
        quantizers = recipe_state.make_quantizers()

        self.quantizers = quantizers

        return quantizers


_te_fp8_quantizers = functional_te_ex.register_stateful_operator(
    "get_te_fp8_quantizers", TEQuantizerState, meta=_get_te_fp8_quantizers_meta
)


def _te_fp8_state_meta(recipe: AnyProxy, mode: str, num_quantizers: int, /):
    return AnyProxy(None, prefix="s")


class TERecipeState:
    def __init__(self):
        self.state = None

    def __call__(self, recipe: Recipe, mode: str, num_quantizers: int) -> RecipeState:
        if self.state:
            return self.state

        # mode is needed to get the correct dtypes inside for computation(setup quantizers)
        recipe_state = te.fp8.RecipeState.create(
            recipe,
            mode=mode,
            num_quantizers=num_quantizers,
        )

        self.state = recipe_state

        return recipe_state


_te_fp8_state = functional_te_ex.register_stateful_operator("te_fp8_state", TERecipeState, meta=_te_fp8_state_meta)


def _linear_fwd_meta(
    a: TensorProxy,
    w: TensorProxy,
    bias: TensorProxy | None,
    input_quantizer: AnyProxy,
    weight_quantizer: AnyProxy,
):
    out_shape = (*a.shape[:-1], w.shape[0])
    return TensorProxy(like=a, shape=out_shape), TensorProxy(like=a), TensorProxy(like=w)


def _linear_fwd_impl(a, w, bias, input_quantizer: Float8Quantizer, weight_quantizer: Float8Quantizer):
    out, quantized_a, quantized_w = BasicLinear._functional_forward(
        input=a,
        weight=w,
        with_quantized_compute=True,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
        output_quantizer=None,  # return out in original dtype (w.dtype)
    )
    return out, quantized_a, quantized_w


_te_linear_fwd = functional_te_ex.register_operator(
    "te_functional_linear_fwd", meta=_linear_fwd_meta, fn=_linear_fwd_impl
)


def _te_linear_execution_transform(a, w, /, bias):
    recipe = _get_te_fp8_recipe()

    forward_recipe_state = _te_fp8_state(recipe, "forward", 2)

    input_quantizer, weight_quantizer = _te_fp8_quantizers(forward_recipe_state, 2)

    out, _, _ = _te_linear_fwd(a, w, bias, input_quantizer, weight_quantizer)

    (out,) = _te_fp8_amax_and_scale_update(recipe, states=(forward_recipe_state,), tokens=(out,))

    return out


def _linear_bwd_meta(
    grad_o: TensorProxy,
    a: TensorProxy,
    w: TensorProxy,
    input_quantizer: AnyProxy,
    weight_quantizer: AnyProxy,
    grad_output_quantizer: AnyProxy,
):
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


_te_linear_bwd = functional_te_ex.register_operator(
    "te_functional_linear_bwd", meta=_linear_bwd_meta, fn=_linear_bwd_impl
)


def _te_linear_grad_transform(a, w, bias):
    recipe = _get_te_fp8_recipe()

    forward_recipe_state = _te_fp8_state(recipe, "forward", 2)

    input_quantizer, weight_quantizer = _te_fp8_quantizers(forward_recipe_state, 2)

    # TODO rename gemm_a/w
    primal, _, _ = _te_linear_fwd(a, w, bias, input_quantizer, weight_quantizer)

    (primal,) = _te_fp8_amax_and_scale_update(
        recipe,
        states=(forward_recipe_state,),
        tokens=(primal,),
    )

    grad_out = get_grad(primal)

    backward_recipe_state = _te_fp8_state(recipe, "backward", 1)

    (grad_output_quantizer,) = _te_fp8_quantizers(backward_recipe_state, 1)

    grad_a, grad_w = _te_linear_bwd(
        grad_out,
        a,
        w,
        input_quantizer,
        weight_quantizer,
        grad_output_quantizer,
    )

    grad_a, grad_w = _te_fp8_amax_and_scale_update(
        recipe,
        states=(backward_recipe_state,),
        tokens=(grad_a, grad_w),
    )

    put_grad(a, grad_a)
    put_grad(w, grad_w)

    if bias:
        if primal.ndim > 1:
            grad_bias = ltorch.sum(primal, tuple(range(primal.ndim - 1)))
        else:
            grad_bias = primal
        put_grad(bias, grad_bias)

    return primal


functional_te_ex.register_implementation(
    ltorch.linear,
    checker=_functional_te_checker,
    execution_transform=_te_linear_execution_transform,
    grad_transform=_te_linear_grad_transform,
)


def _te_fp8_amax_and_scale_update_meta(recipe: AnyProxy, *, states: tuple[AnyProxy], tokens: tuple[TensorProxy] | None):
    if tokens:
        return (*(TensorProxy(like=t) for t in tokens),)
    return tuple()


# TODO can gather and scatter be made explicit here for ddp
def _te_fp8_amax_and_scale_update_impl(recipe: Recipe, states: tuple[RecipeState], tokens: tuple[TensorProxy] | None):
    if recipe.mxfp8():
        return (*tokens,)

    for state in states:
        # print("UPDATING", id(state), "WITH RECIPE", id(recipe))
        _amax_and_scale_update(state.amax_history, state.scale, get_fp8_max(recipe, state.mode), recipe)
    return (*tokens,)


_te_fp8_amax_and_scale_update = functional_te_ex.register_operator(
    "te_fp8_amax_and_scale_update",
    meta=_te_fp8_amax_and_scale_update_meta,
    fn=_te_fp8_amax_and_scale_update_impl,
)
