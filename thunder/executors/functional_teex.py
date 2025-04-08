import importlib
import warnings

from thunder.core.prims import OpTags
from thunder.core.proxies import AnyProxy, TensorProxy
from thunder.core.prims import get_grad
from thunder.core.transforms import put_grads
from thunder.extend import OperatorExecutor, register_executor

from thunder.core import prims

from thunder.executors.transformer_engineex import _linear_checker

if importlib.util.find_spec("transformer_engine"):
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
else:
    warnings.warn("transformer_engine module not found!")


class StatefulExecutor(OperatorExecutor):
    def __init__(self, name, *, version=None):
        super().__init__(name, version=version)
        self.op_counter: dict[str, int] = {}

    def register_stateful_operator(self, base_name: str, state_class, *, meta):
        def register_state(*args, **kwargs):

            state_id = self.op_counter.setdefault(base_name, 0)
            name = f"{base_name}_{state_id}"
            self.op_counter[base_name] +=1

            def bind_state(bsym):
                bsym._call_ctx = {name: state_class()}

            sym = self.register_operator(name, meta=meta, bind_postprocess=bind_state)
            return sym(*args, *kwargs)

        return register_state


functional_te_ex = StatefulExecutor("functional_te")
register_executor(functional_te_ex)


# TODO
def _functional_te_checker(a, w, /, bias):
    # TE 2.0 functional API does not yet support bias=True
    # https://github.com/NVIDIA/TransformerEngine/blob/1321b9b5dc96d67d20d6682c52116a3657f293d3/transformer_engine/pytorch/ops/basic/basic_linear.py#L46-L47
    # https://github.com/NVIDIA/TransformerEngine/blob/1321b9b5dc96d67d20d6682c52116a3657f293d3/transformer_engine/pytorch/ops/basic/basic_linear.py#L505
    if bias is not None:
        return False
    return _linear_checker(a, w, bias)


# Assuming one recipe per machine since it's based on hw capabilities.
functional_te_recipe = get_default_fp8_recipe()


def _te_fp8_recipe_meta(recipe_name: str):
    return AnyProxy(recipe_name, prefix="r")


def _te_fp8_recipe_impl(recipe_name: str):
    return functional_te_recipe


_te_fp8_recipe = functional_te_ex.register_operator("te_fp8_recipe", meta=_te_fp8_recipe_meta, fn=_te_fp8_recipe_impl)


def _te_fp8_quantizers_meta(recipe_state: RecipeState, num_quantizers: int):
    # giving num quantizers so there is no need to allocate extra things
    return (*(AnyProxy(num_quantizers, prefix="q") for _ in range(num_quantizers)),)


class TEQuantizerState:
    def __init__(self):
        self.quantizers = None

    def __call__(self, recipe_state: RecipeState, num_quantizers: int):
        if self.quantizers:
            return self.quantizers
        quantizers = recipe_state.make_quantizers()

        self.quantizers = quantizers

        return quantizers


_te_fp8_quantizers = functional_te_ex.register_stateful_operator(
    "te_fp8_quantizers", TEQuantizerState, meta=_te_fp8_quantizers_meta
)


def _te_fp8_state_meta(recipe, mode: str, num_quantizers: int, /):
    # mode just to give the proxy something wihout allocating extra stuff
    return AnyProxy(mode, prefix="s")


class TERecipeState:
    def __init__(self):
        self.state = None

    def __call__(self, recipe: Recipe, mode, num_quantizers):
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
    input_quantizer: Float8Quantizer,
    weight_quantizer: Float8Quantizer,
):
    return TensorProxy(like=a), TensorProxy(like=a), TensorProxy(like=w)


def _linear_fwd_impl(a, w, bias, input_quantizer, weight_quantizer):
    out, gemm_a, gemm_w = BasicLinear._functional_forward(
        input=a,
        weight=w,
        with_quantized_compute=True,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
        output_quantizer=None,  # return out in original dtype (w.dtype)
    )
    return out, gemm_a, gemm_w


_te_linear_fwd = functional_te_ex.register_operator(
    "te_functional_linear_fwd", meta=_linear_fwd_meta, fn=_linear_fwd_impl
)


def _te_linear_execution_transform(a, w, /, bias):
    recipe = _te_fp8_recipe("delayed")

    forward_recipe_state = _te_fp8_state(recipe, "forward", 2)

    input_quantizer, weight_quantizer = _te_fp8_quantizers(forward_recipe_state, 2)

    out, _, _ = _te_linear_fwd(a, w, bias, input_quantizer, weight_quantizer)

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


_te_linear_bwd = functional_te_ex.register_operator(
    "te_functional_linear_bwd", meta=_linear_bwd_meta, fn=_linear_bwd_impl
)


def _te_linear_grad_transform(a, w, bias):
    recipe = _te_fp8_recipe("delayed")

    forward_recipe_state = _te_fp8_state(recipe, "forward", 2)

    input_quantizer, weight_quantizer = _te_fp8_quantizers(forward_recipe_state, 2)

    primal, gemm_a, gemm_w = _te_linear_fwd(a, w, bias, input_quantizer, weight_quantizer)

    backward_recipe_state = _te_fp8_state(recipe, "backward", 1)

    (grad_output_quantizer,) = _te_fp8_quantizers(backward_recipe_state, 1)

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
    prims.linear,
    checker=_functional_te_checker,
    execution_transform=_te_linear_execution_transform,
    grad_transform=_te_linear_grad_transform,
)


def _te_fp8_sync_meta(recipe, *states, forward: bool):
    return None


def _te_fp8_sync_impl(recipe, *states, forward: bool):
    for state in states:
        _amax_and_scale_update(state.amax_history, state.scale, get_fp8_max(recipe, forward), recipe)
    return None


_te_fp8_synchronization = functional_te_ex.register_operator(
    "te_fp8_sync",
    meta=_te_fp8_sync_meta,
    fn=_te_fp8_sync_impl,
    tags=(OpTags.DONT_DCE,),
)
