import time
from typing import TYPE_CHECKING
import warnings
from collections import defaultdict

import torch
import torch.distributed as torch_dist

from thunder.core.prims import linear as linear_prim
from thunder.core.prims import get_grad, put_grad
from thunder.core.proxies import AnyProxy, TensorProxy
from thunder.extend import StatefulExecutor, register_executor
import thunder.torch as ltorch
from thunder import Transform
from thunder.core import prims
import thunder.core.devices as devices
from thunder.core.proxies import Proxy, Variable, unvariableify, variableify
from thunder.core.trace import from_trace, TraceProvenance, TraceTag, TraceCtx
from thunder.core.transforms import (
    _update_forward_with_new_saved_for_backward,
    _update_backward_with_new_saved_for_backward,
)
from thunder.core.transform_common import cse_single_bsym
from thunder.executors.passes import del_last_used
import thunder.core.utils as utils

if TYPE_CHECKING:
    from thunder.core.trace import VariableInterface
    from thunder.core.symbol import BoundSymbolRHS, BoundSymbol
    from thunder.core.proxies import TensorProxy

import transformer_engine.pytorch as te
import transformer_engine.common.recipe as te_recipe
from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.fp8 import (
    _amax_and_scale_update,
    get_fp8_max,
    Recipe,
    RecipeState,
    FP8GlobalStateManager,
)
from transformer_engine.pytorch.ops import BasicLinear
from transformer_engine.pytorch.tensor import Quantizer
from transformer_engine.pytorch.utils import check_dim_for_fp8_exec
from thunder.dev_utils.export_stateful_ex_transform import (
    ExportStatefulExecutorsTransform as _ExportSETransform,
)


transformer_engine_ex = StatefulExecutor("transformer_engine")
register_executor(transformer_engine_ex)


def _te_fp8_recipe_meta() -> AnyProxy:
    return AnyProxy(None, prefix="r")


class TERecipe:
    def __init__(self):
        self.fp8_recipe = None

    def __call__(self) -> Recipe:
        # Since we want to mimic TransformerEngine default behaviour as much as possible, we rely on FP8GlobalStateManager.get_fp8_recipe() to provide the correct TE recipe.
        # If the Thunder function is not ran under an `fp8_autocast`, `get_fp8_recipe` will return the default recipe for the platform.
        # https://github.com/NVIDIA/TransformerEngine/blob/0e45e138c08af8f3b38e46eea58e2e9dbe628d42/transformer_engine/pytorch/fp8.py#L318-L322
        te_fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()

        if not self.fp8_recipe or self.fp8_recipe is not te_fp8_recipe:
            self.fp8_recipe = te_fp8_recipe

        return self.fp8_recipe


_get_te_fp8_recipe = transformer_engine_ex.register_stateful_operator(
    "get_te_fp8_recipe", meta=_te_fp8_recipe_meta, state_class=TERecipe
)


def _get_te_fp8_quantizers_meta(recipe_state: RecipeState, num_quantizers: int):
    # num_quantizers can be taken from recipe state, but not at trace time.
    # Therefore it's needed here to inform the meta.
    return (*(AnyProxy(None, prefix="q") for _ in range(num_quantizers)),)


class TEQuantizerState:
    def __init__(self):
        self.quantizers: None | list[Quantizer] = None
        self.parent_recipe_state: None | RecipeState = None

    def __call__(self, recipe_state: RecipeState, num_quantizers: int) -> list[Quantizer]:
        if self.quantizers and self.parent_recipe_state is recipe_state:
            return self.quantizers
        quantizers = recipe_state.make_quantizers()

        self.quantizers = quantizers
        self.parent_recipe_state = recipe_state

        return quantizers


_get_te_fp8_quantizers = transformer_engine_ex.register_stateful_operator(
    "get_te_fp8_quantizers", TEQuantizerState, meta=_get_te_fp8_quantizers_meta
)


def _get_te_fp8_state_meta(recipe: AnyProxy, mode: str, num_quantizers: int, /):
    return AnyProxy(None, prefix="s")


class TERecipeState:
    def __init__(self):
        self.parent_recipe: None | Recipe = None
        self.state = None

    def __call__(self, recipe: Recipe, mode: str, num_quantizers: int) -> RecipeState:
        # If the recipe changes, then a new state is needed.
        if self.state and self.parent_recipe is recipe:
            return self.state

        # mode is needed to get the correct dtypes inside for computation(setup quantizers)
        recipe_state = te.fp8.RecipeState.create(
            recipe,
            mode=mode,
            num_quantizers=num_quantizers,
        )

        self.state = recipe_state
        self.parent_recipe = recipe

        return recipe_state


_get_te_fp8_state = transformer_engine_ex.register_stateful_operator(
    "get_te_fp8_state", TERecipeState, meta=_get_te_fp8_state_meta
)


def _linear_fwd_meta(
    a: TensorProxy,
    w: TensorProxy,
    bias: TensorProxy | None,
    input_quantizer: AnyProxy,
    weight_quantizer: AnyProxy,
):
    out_shape = (*a.shape[:-1], w.shape[0])
    return TensorProxy(like=a, shape=out_shape), TensorProxy(like=a), TensorProxy(like=w)


def _linear_fwd_impl(a, w, bias, input_quantizer: Quantizer, weight_quantizer: Quantizer):
    out, quantized_a, quantized_w = BasicLinear._functional_forward(
        input=a,
        weight=w,
        bias=bias,
        dtype=w.dtype,  # WAR for TE library issue https://github.com/NVIDIA/TransformerEngine/issues/2011
        with_quantized_compute=True,
        input_quantizer=input_quantizer,
        weight_quantizer=weight_quantizer,
        output_quantizer=None,  # return out in original dtype (w.dtype)
    )
    return out, quantized_a, quantized_w


_te_linear_fwd = transformer_engine_ex.register_operator(
    "te_functional_linear_fwd", meta=_linear_fwd_meta, fn=_linear_fwd_impl
)


def _te_linear_execution_transform(a, w, /, bias):
    recipe = _get_te_fp8_recipe()

    forward_recipe_state = _get_te_fp8_state(recipe, "forward", 2)

    input_quantizer, weight_quantizer = _get_te_fp8_quantizers(forward_recipe_state, 2)

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


def _linear_bwd_impl(
    grad_o, a, w, input_quantizer: Quantizer, weight_quantizer: Quantizer, grad_output_quantizer: Quantizer
):
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


_te_linear_bwd = transformer_engine_ex.register_operator(
    "te_functional_linear_bwd", meta=_linear_bwd_meta, fn=_linear_bwd_impl
)


def _te_linear_grad_transform(a, w, bias):
    recipe = _get_te_fp8_recipe()

    forward_recipe_state = _get_te_fp8_state(recipe, "forward", 2)

    input_quantizer, weight_quantizer = _get_te_fp8_quantizers(forward_recipe_state, 2)

    primal, quantized_a, quantized_w = _te_linear_fwd(a, w, bias, input_quantizer, weight_quantizer)

    (primal, quantized_a, quantized_w) = _te_fp8_amax_and_scale_update(
        recipe,
        states=(forward_recipe_state,),
        tokens=(primal, quantized_a, quantized_w),
    )

    grad_out = get_grad(primal)

    backward_recipe_state = _get_te_fp8_state(recipe, "backward", 1)

    (grad_output_quantizer,) = _get_te_fp8_quantizers(backward_recipe_state, 1)

    grad_a, grad_w = _te_linear_bwd(
        grad_out,
        quantized_a,
        quantized_w,
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

    if bias is not None:
        if grad_out.ndim > 1:
            grad_bias = ltorch.sum(grad_out, tuple(range(grad_out.ndim - 1)))
        else:
            grad_bias = grad_out
        put_grad(bias, grad_bias)

    return primal


def _linear_checker(
    a: TensorProxy,
    w: TensorProxy,
    bias: None | TensorProxy,
) -> bool:
    def is_cuda(t):
        return t.device.devicetype == devices.DeviceType.CUDA

    inputs = (a, w)
    if bias is not None:
        inputs = inputs + (bias,)

    # Helper function as input shape can be (*, Hin)
    def _view_input_as_2d(x):
        shape = x.shape
        return x.view((-1, shape[-1]))

    fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()

    supported_recipes = (te_recipe.DelayedScaling, te_recipe.MXFP8BlockScaling)
    if hasattr(te_recipe, "NVFP4BlockScaling"):
        supported_recipes = (*supported_recipes, te_recipe.NVFP4BlockScaling)

    if not isinstance(fp8_recipe, supported_recipes):
        warnings.warn(f"{type(fp8_recipe)} is not supported by TE executor, TE wont be used.")
        return False

    def check_valid_fp8_shapes(a):
        # Each recipe type has different shape requirements.
        if fp8_recipe.delayed():
            return check_dim_for_fp8_exec(a)

        shape = a.shape

        if fp8_recipe.mxfp8():
            return shape[0] % MXFP8_BLOCK_SCALING_SIZE == 0 and shape[1] % MXFP8_BLOCK_SCALING_SIZE == 0

        if hasattr(fp8_recipe, "nvfp4") and fp8_recipe.nvfp4():
            from transformer_engine.pytorch.constants import NVFP4_BLOCK_SCALING_SIZE

            # Check inherited from TE https://github.com/ksivaman/TransformerEngine-1/blob/1af7dd88aae5afb45e82148089038e1d1de9675d/transformer_engine/pytorch/tensor/nvfp4_tensor.py#L176-L184
            return (
                len(shape) >= 2
                and shape[0] % NVFP4_BLOCK_SCALING_SIZE == 0
                and shape[1] % NVFP4_BLOCK_SCALING_SIZE == 0
            )

        return False

    # Inputs must be on CUDA and
    # input sizes must satisfy size constraints based on the recipe.
    return all(map(is_cuda, inputs)) and check_valid_fp8_shapes(_view_input_as_2d(a)) and check_valid_fp8_shapes(w)


transformer_engine_ex.register_implementation(
    linear_prim,
    checker=_linear_checker,
    execution_transform=_te_linear_execution_transform,
    grad_transform=_te_linear_grad_transform,
)


def _te_fp8_amax_and_scale_update_meta(recipe: AnyProxy, *, states: tuple[AnyProxy], tokens: tuple[TensorProxy]):
    utils.check(tokens is not None, lambda: "tokens cannot be None to create a valid dataflow", RuntimeError)
    return (*(TensorProxy(like=t) for t in tokens),)


# TODO can gather and scatter be made explicit here for ddp
def _te_fp8_amax_and_scale_update_impl(recipe: Recipe, states: tuple[RecipeState], tokens: tuple[TensorProxy]):
    for state in states:
        if getattr(recipe, "reduce_amax", False) and torch_dist.is_available() and torch_dist.is_initialized():
            from torch.distributed import distributed_c10d

            pg = distributed_c10d._get_default_group()
            if torch_dist.get_world_size(group=pg) > 1:
                torch_dist.all_reduce(
                    state.amax_history,
                    op=torch_dist.ReduceOp.MAX,
                    group=pg,
                    async_op=False,
                )

        if recipe.delayed():
            _amax_and_scale_update(
                state.amax_history, state.scale, get_fp8_max(recipe, state.mode == "forward"), recipe
            )

    return (*tokens,)


_te_fp8_amax_and_scale_update = transformer_engine_ex.register_operator(
    "te_fp8_amax_and_scale_update",
    meta=_te_fp8_amax_and_scale_update_meta,
    fn=_te_fp8_amax_and_scale_update_impl,
)


class TransformerEngineTransform(Transform):
    """
    A transform to pair up with the functional TransformerEngine executor.

    With the assumption of one recipe per trace, this transform removes recipe duplicates from the trace and updates all the symbols.

    TODO: this transform can also be used to gather all the amax and scale update calls for delayed scaling recipe.
    """

    def __init__(self):
        self.fp8_recipe = None
        self.swap_map: dict[VariableInterface, TensorProxy] = {}
        self.rhs_to_bsym_map: dict[BoundSymbolRHS, BoundSymbol] = {}
        self.redundant_map: dict[Variable, Proxy] = {}
        self.new_saved_for_backward = None

    def reset(self):
        self.fp8_recipe = None
        self.swap_map = {}
        self.rhs_to_bsym_map = {}
        self.redundant_map = {}
        self.new_saved_for_backward = None

    @staticmethod
    def export_state(computation_trace, tm) -> None:
        """
        Extracts and exports the FP8 amax/scale state information from TransformerEngine (TE) holders
        present in the Python context of a computation trace.

        This method is intended to be called after a TE-enabled computation has executed, in order to
        serialize and record the relevant FP8 state (such as amax and scale tensors) and quantizer
        information for later inspection, debugging, or export.

        Args:
            computation_trace: The Thunder computation trace object containing the Python context
                with TE state and quantizer holders.
            tm: The ThunderModule object.

        Returns:
            None.
        """
        # Extract FP8 amax/scale information from TE holders available in python context
        python_ctx = computation_trace.python_ctx()

        # Helper: serialize small tensors; skip oversized payloads
        def _to_list_limited(t, max_numel: int = 8192):
            if not isinstance(t, torch.Tensor):
                return None
            try:
                n = min(t.numel(), max_numel)
                if t.numel() > max_numel:
                    warnings.warn(
                        f"TE Stateful Executor: Exporting only first {max_numel} elements of tensor with {t.numel()} elements",
                        UserWarning,
                    )
                flat = t.detach().float().cpu().view(-1)[:n].tolist()
                return flat
            except Exception:
                return None

        # Infer context mode from available TE functional symbols
        te_mode = None
        if "te_functional_linear_fwd" in python_ctx:
            te_mode = "forward"
        elif "te_functional_linear_bwd" in python_ctx:
            te_mode = "backward"

        delayed_entries: list[dict] = []
        block_entries: list[dict] = []

        # Gather state and quantizer holders from context
        state_holders = [v for k, v in python_ctx.items() if isinstance(k, str) and k.startswith("get_te_fp8_state")]
        quantizer_holders = [
            v for k, v in python_ctx.items() if isinstance(k, str) and k.startswith("get_te_fp8_quantizers")
        ]

        # Map RecipeState -> quantizers (if materialized)
        state_to_quantizers: dict[int, list] = {}
        for qh in quantizer_holders:
            prs = getattr(qh, "parent_recipe_state", None)
            qs = getattr(qh, "quantizers", None)
            if prs is not None and qs:
                state_to_quantizers.setdefault(id(prs), []).extend(qs)

        for sh in state_holders:
            recipe = getattr(sh, "parent_recipe", None)
            state = getattr(sh, "state", None)
            if recipe is None:
                continue

            # Determine recipe family
            is_delayed = bool(recipe.delayed())
            is_mxfp8_or_block = bool(recipe.mxfp8())

            # DelayedScaling: values live on state.scale and state.amax_history
            if is_delayed and state is not None:
                scale_vals = _to_list_limited(getattr(state, "scale", None))
                amax_hist = getattr(state, "amax_history", None)
                amax_vals = None
                if isinstance(amax_hist, torch.Tensor) and amax_hist.numel() > 0:
                    amax_slice = amax_hist[-1] if amax_hist.dim() >= 1 else amax_hist
                    amax_vals = _to_list_limited(amax_slice)
                delayed_entries.append(
                    {
                        "scale_shape": getattr(getattr(state, "scale", None), "shape", None),
                        "scale": scale_vals,
                        "amax_shape": getattr(getattr(state, "amax_history", None), "shape", None),
                        "amax": amax_vals,
                    }
                )

            # MXFP8/Float8 block scaling: values live on quantizers
            elif is_mxfp8_or_block and state is not None:
                qs = state_to_quantizers.get(id(state), [])
                for q in qs:
                    rowwise_usage = getattr(q, "rowwise_usage", None)
                    columnwise_usage = getattr(q, "columnwise_usage", None)
                    block_entries.append(
                        {
                            "cls": q.__class__.__name__,
                            "rowwise_usage": rowwise_usage,
                            "columnwise_usage": columnwise_usage,
                            "dtype": str(getattr(q, "dtype", None)),
                        }
                    )

        entry = defaultdict(list)
        if delayed_entries:
            entry["delayed"] = delayed_entries
        if block_entries:
            entry["mxfp8_or_block"] = block_entries

        collected = getattr(tm, "te_fp8_stats", None)
        if collected is None:
            tm.te_fp8_stats = {"forward": [], "backward": []}
        if entry["delayed"] or entry["mxfp8_or_block"]:
            tm.te_fp8_stats[te_mode].append(entry)

    def transform_trace_post_optimization(self, computation_trace, **kwargs):
        """
        Finds and replaces TE executor recipe calls and replaces them with one.

        This function may be called twice, once with the forward trace and once with the backward trace.
        It will save the first occurance of a recipe from the trace and use it to replce all the others.

        Args:
            computation_trace: Trace to perform the replacement on.
        """

        if "transformer_engine" not in map(lambda x: x.name, kwargs["executors_list"]):
            return computation_trace

        start_time_ns = time.perf_counter_ns()

        new_trace = from_trace(computation_trace)
        new_bsyms = []

        for bsym in computation_trace.bound_symbols:
            # Remove all the delete since they will be outdated after the proxy update.
            if bsym.sym.id == prims.PrimIDs.DEL:
                continue

            # Save the first occurrence of a recipe symbol and map any later ones in the redundant_map
            if "get_te_fp8_recipe" in bsym.sym.name:
                # Store the first occurrence
                if not self.fp8_recipe:
                    self.fp8_recipe = bsym
                else:
                    vsrc = variableify(bsym.output)
                    self.redundant_map[vsrc] = self.fp8_recipe.output
                    continue

            if bsym.sym.is_fusion:
                new_bsym = bsym.from_bsym_swap_proxies(self.redundant_map)
            else:
                new_bsym = cse_single_bsym(self.redundant_map, self.rhs_to_bsym_map, bsym)

            # cse_single_bsym might return None if the input bsym is a duplicate.
            if new_bsym:
                new_bsyms.append(new_bsym)

        # Couldn't find any TE recipe in the trace
        if not self.fp8_recipe:
            self.reset()
            return computation_trace

        new_trace.bound_symbols = new_bsyms

        if self.new_saved_for_backward:
            _update_backward_with_new_saved_for_backward(new_trace, self.new_saved_for_backward)
            # Reset transform after going through forward and backward
            self.reset()

        # If the trace has been generated by Thunder autograd then we also need to remove extra recipies from the return statement
        if TraceTag.AUGMENTED_FORWARD in computation_trace.tags:
            return_bsym = new_trace.bound_symbols[-1]
            assert return_bsym.sym.id == prims.PrimIDs.RETURN
            _, (saved_for_backward, env) = return_bsym.args
            unique_env = list(dict.fromkeys(Variable(x) for x in env))
            self.new_saved_for_backward = (
                *saved_for_backward,
                *(unvariableify(x) for x in unique_env),
                self.fp8_recipe.output,
            )

            _update_forward_with_new_saved_for_backward(new_trace, self.new_saved_for_backward)

        sync_trace = del_last_used(new_trace)

        end_time_ns = time.perf_counter_ns()
        elapsed_time_ns = end_time_ns - start_time_ns
        elapsed_time_millis = elapsed_time_ns // 1000000

        sync_trace.set_provenance(
            TraceProvenance(f"TransformerEngine Synchronization transform (took {elapsed_time_millis} milliseconds)")
        )

        return sync_trace


def _te_activation_checkpointing_transform(joint_trace: TraceCtx) -> TraceCtx:
    """
    Optimizes FP8 state management in activation checkpointing by removing
    redundant amax/scale updates, keeping only the final update for each state.

    Args:
        joint_trace: Joint trace after recomputation has been inserted.
    """

    swapmap: dict[Variable, Proxy] = {}
    state_swapmap: dict[str, Proxy] = {}
    new_bsyms = []

    # Find the first call for every FP8 state and reuse it throughout the trace.
    # Since get_te_fp8_state symbol names are unique, we use them to identify duplicates.
    for bsym in joint_trace.bound_symbols:
        if "get_te_fp8_state" in bsym.sym.name:
            out_proxy = state_swapmap.get(bsym.sym.name, None)
            if out_proxy is None:
                # First occurrence - save it for reuse
                state_swapmap[bsym.sym.name] = bsym.output
            else:
                swapmap[variableify(bsym.output)] = out_proxy
                # Remove excessive duplicate calls to avoid redundant state creation
                continue

        new_bsyms.append(bsym.from_bsym_swap_proxies(swapmap))

    new_trace = from_trace(joint_trace)
    seen_updated_states = set()
    reversed_bsyms = []

    # Optimize amax/scale updates by keeping only the final update for each state.
    # We iterate in reverse order so the "first" occurrence we keep is actually the last one,
    # which ensures we maintain the most recent scaling information.
    for bsym in reversed(new_bsyms):
        if bsym.sym.name == "te_fp8_amax_and_scale_update":
            # TODO: This takes into account multiple states and does remove the update call only
            # in the case that all the states have been updated already. When update grouping will be added this needs to be revised.
            if all(variableify(state) in seen_updated_states for state in bsym.kwargs["states"]):
                # All states have already been updated - redirect outputs to original tokens
                tokens_to_swap = bsym.kwargs["tokens"]
                for t, o in zip(tokens_to_swap, bsym.output):
                    if o is not None:
                        swapmap[variableify(o)] = t
                continue
            else:
                # Mark these states as having been updated
                for state in bsym.kwargs["states"]:
                    seen_updated_states.add(variableify(state))

        reversed_bsyms.append(bsym)

    # Reverse the trace back to original order and apply proxy updates from removed amax calls
    new_trace.bound_symbols = [bsym.from_bsym_swap_proxies(swapmap) for bsym in reversed(reversed_bsyms)]

    return new_trace


# Register TE export callback with the singleton export transform
try:
    _ExportSETransform.register_export_callback("transformer_engine", TransformerEngineTransform.export_state)
except Exception:
    pass
