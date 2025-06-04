from functools import partial
from itertools import chain
from typing import Any
from collections.abc import Sequence
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from collections import deque
from importlib.metadata import version
from looseversion import LooseVersion
import warnings

import torch

from lightning_utilities.core.imports import package_available

from thunder.core.proxies import TensorProxy
from thunder.core.trace import get_tracectx
from thunder.core.symbol import Symbol, BoundSymbol
import thunder.core.devices as devices
import thunder.core.prims as prims
from thunder.core.proxies import TensorProxy, AnyProxy
from thunder.core.symbol import Symbol
from thunder.core.vjp_utils import disable_caching_split_forward_and_backward
from thunder.extend import OperatorExecutor, register_executor
from thunder.core.compile_data import get_compile_option, get_compile_data
from thunder.distributed import FSDPType
from thunder.executors.utils import Context, set_saved_tensors
from thunder.core.trace import from_trace, TraceCtx, TraceProvenance, tracectx


__all__ = [
    "transformer_engine_ex",
]

TE_AVAILABLE: bool = package_available("transformer_engine")

# We rely on internal details of TransformerEngine like `_Linear` autograd.Function.
# As these details are not public, they can change
# Ex. addition of a positional argument for cpu_offloading (not as the last argument)
# between version 1.2 and 1.3.
# Hence, we have these guards based on version.

te: None | Any = None
if TE_AVAILABLE:
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        from transformer_engine.common.recipe import MXFP8BlockScaling, DelayedScaling
        from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
        from transformer_engine.pytorch.module.linear import _Linear
        from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, get_default_fp8_recipe
        from transformer_engine.pytorch.utils import check_dim_for_fp8_exec
        from transformer_engine.pytorch.cpu_offload import CPUOffloadEnabled
        import transformer_engine_torch as tex
    except Exception as ex:
        warnings.warn(f"transformer_engine failed to import with exception {ex}")
        TE_AVAILABLE = False

    TE_VERSION_2_0_PLUS = LooseVersion(version("transformer_engine")) > LooseVersion("2.0")
    if not TE_VERSION_2_0_PLUS:
        msg = f"Installed version of transformer_engine {version('transformer_engine')} is not supported, please upgrade to version 2.0 from https://github.com/NVIDIA/TransformerEngine/tree/release_v2.0. `transformer_engine_ex` will not be used."
        warnings.warn(msg)
        TE_AVAILABLE = False


if not TE_AVAILABLE:
    TransformerEngineBaseModule = object

# [NOTE] IMPLEMENTATION DETAILS
#
# We try to re-use TransformerEngine implementation of `Linear` and `_Linear` as much as possible.
# As `thunder` expects operator to be passed all of its inputs, we have `TELinear` module which doesn't
# register any parameters and takes all `Tensor` arguments as input (It based on `Linear` from TE)
# FP8 tensors require extra meta-data per Tensor. Similar to TE, this meta-data is saved in module `TELinear`.
# NOTE: Implementation supports a limited set of input sizes where dim0 is divisible by 8 and dim1 is divisible by 16.
#
# Ref to `_Linear`: https://github.com/NVIDIA/TransformerEngine/blob/b957aa475bcbcf22405381d18bd7fefe4fb6b171/transformer_engine/pytorch/module/linear.py#L52
# Ref to `Linear`: https://github.com/NVIDIA/TransformerEngine/blob/b957aa475bcbcf22405381d18bd7fefe4fb6b171/transformer_engine/pytorch/module/linear.py#L543
# Stateful Operator:
# This means that every call to this `linear` requires a corresponding `TELinear` instance for
# backing the required FP8 state. This is done by creating a new `BoundSymbol` with corresponding instance
# when replacing calls to `prims.linear` (see `_create_fp8_linear_bound_symbol`).
# Eg.
# Original Program:
#
# def func(a, b, d):
#   out = torch.nn.functional.linear(a, b)
#   out = torch.nn.functional.linear(out, d)
#   return out
#
# Traced Program:
#
# @torch.no_grad()
# @no_autocast
# @transformer_engine.fp8_autocast(fp8_recipe=te_fp8_recipe)
# def func(a, b, d):
#   # a: "cuda:0 bf16[16, 32]"
#   # b: "cuda:0 bf16[64, 32]"
#   # d: "cuda:0 bf16[32, 64]"
#   (t0, _) = te_linear_0(a, b, None, is_grad_enabled=False)  # Backed by it's own instance of TELinear
#   del a, b
#   (t1, _) = te_linear_1(t0, d, None, is_grad_enabled=False)  # Backed by it's own instance of TELinear
#   del t0, d
#   return t1
#
# Managing Residuals for Backward:
# As we re-use `_Linear` which is a `torch.autograd.Function`, it requires a `ctx` Context object to
# save required objects for backward. We have our own `Context` class for the same.
# `_Linear` saves a lot of objects in `ctx` some of which is generated during the first call to `forward`.
#
# [NOTE] Enable grad within context
# To correctly compute the gradients, `_Linear` expects `requires_grad` to be
# set on the `input`, `weight` and `bias` tensor.
# But when applying `vjp`, the input tensor may not have requires_grad
# (as the rules take care relevant transformation). Thus we use `enable_grad` decorator
# when applying the forward and backward rule.
#
# Reference to points where TE looks at `requires_grad`:
# Ref: https://github.com/NVIDIA/TransformerEngine/blob/b957aa475bcbcf22405381d18bd7fefe4fb6b171/transformer_engine/pytorch/module/linear.py#L264
# Ref: https://github.com/NVIDIA/TransformerEngine/blob/b957aa475bcbcf22405381d18bd7fefe4fb6b171/transformer_engine/pytorch/module/linear.py#L434


# Eagerly apply map without
# storing the output.
def eager_map(*args):
    return deque(map(*args), maxlen=0)


# Set requires_grad to True for passed tensors
# in this context.
@contextmanager
def enable_grad(*tensors):
    original_requires_grad = tuple(map(lambda t: t.requires_grad, tensors))
    eager_map(lambda t: t.requires_grad_(True), tensors)
    try:
        yield
    finally:
        eager_map(lambda t, org_r_grad: t.requires_grad_(org_r_grad), tensors, original_requires_grad)


FP8_SHARD_INTERMEDIATE_ACTIVATIONS = "fp8_shard_intermediate_activation"


def _should_shard_intermediate() -> bool:
    compile_data = get_compile_data()

    should_shard_intermediate_options: bool | None = get_compile_option(
        FP8_SHARD_INTERMEDIATE_ACTIVATIONS,
        "transformer_engine_ex: Whether the intermediate activations should be sharded or not. Only applicable with FSDP Zero3, ignored otherwise.",
    )

    if getattr(compile_data.fn, "use_fsdp", False):
        if getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO3 and should_shard_intermediate_options:
            return True

        if should_shard_intermediate_options:  # user passed `True` but FSDPType was not Zero3
            warnings.warn(
                f"transformer_engine_ex: {FP8_SHARD_INTERMEDIATE_ACTIVATIONS} is only applicable for FSDP Zero3"
            )

    return False


def _get_num_saved_tensors(fp8_recipe, w_requires_grad):
    MIN_DIM = MXFP8_BLOCK_SCALING_SIZE
    te_linear = te.Linear(MIN_DIM, MIN_DIM)
    te_linear.weight.requires_grad_(w_requires_grad)

    x = torch.randn(MIN_DIM, MIN_DIM, device="cuda")
    with te.fp8_autocast(fp8_recipe=fp8_recipe):
        o = te_linear(x)
    return len(o.grad_fn.saved_tensors)


class TELinear(TransformerEngineBaseModule):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Used by `get_fp8_weights_scratchpad`
        self.primary_weights_in_fp8 = False

        if FP8GlobalStateManager.with_fp8_parameters():
            raise RuntimeError("Primary weights in FP8 is not supported under `thunder.jit`.")

        # NOTE - This is available only v1.8 onwards
        if _should_shard_intermediate():
            self.pg = get_compile_data().process_group_for_ddp
        else:
            self.pg = None

    def forward(self, inp, weight, bias, *, input_requires_grad, weight_requires_grad, bias_requires_grad):
        # NOTE: Backward FP8 metadata sync
        # TransformerEngine v1.6 onwards, we control the sync and update of FP8 metadata for FP8 tensors
        # tied to backward pass (i.e. the gradient tensors)
        # Also, note that the forward tensor metadata sync occurs at the exit of `fp8_autocast` context manager
        # which is not controlled by us.
        #
        # We consume the `is_first_fp8_module` so that the automatic sync for FP8 metadata is disabled.
        FP8GlobalStateManager.is_first_fp8_module()  # Consume first module token.

        enable_grad_inputs = ()

        # For PEFT scenarios, weights maybe frozen i.e. weight_requires_grad=False.
        if input_requires_grad:
            enable_grad_inputs += (inp,)
        if weight_requires_grad:
            enable_grad_inputs += (weight,)
        if bias_requires_grad:
            enable_grad_inputs += (bias,)

        is_grad_enabled = input_requires_grad or weight_requires_grad or bias_requires_grad

        # See [NOTE] Enable grad within context
        # TE backward depends on `requires_grad` to compute grads but thunder wraps it's execution trace with `no_grad`,
        # so under grad mode we enable grad for input tensors
        # Ref: https://github.com/NVIDIA/TransformerEngine/blob/b957aa475bcbcf22405381d18bd7fefe4fb6b171/transformer_engine/pytorch/module/linear.py#L264
        grad_ctx = enable_grad(*enable_grad_inputs) if is_grad_enabled else nullcontext()
        with grad_ctx, self.prepare_forward(inp) as inp:
            assert (
                self.fp8 or not self.primary_weights_in_fp8
            ), "Need to run inside fp8_autocast region when weights are stored in FP8."

            (
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_output_quantizer,
                grad_input_quantizer,
            ) = self._get_quantizers(is_grad_enabled)

            ctx = Context() if is_grad_enabled else None

            import inspect

            params = inspect.signature(_Linear.forward).parameters

            # Currently we do not support `tp` meaning tensor model parallel case.
            # We hard-code the arguments related to distributed for now.
            use_bias = bias is not None

            kwargs = {
                "ctx": ctx,
                "weight": weight,
                "inp": inp,
                "bias": torch.tensor([]) if not use_bias else bias,
                "is_first_microbatch": None,
                "fp8": self.fp8,
                "fp8_calibration": self.fp8_calibration,
                "input_quantizer": input_quantizer,
                "weight_quantizer": weight_quantizer,
                "output_quantizer": output_quantizer,
                "grad_output_quantizer": grad_output_quantizer,
                "grad_input_quantizer": grad_input_quantizer,
                "fuse_wgrad_accumulation": False,
                "cpu_offloading": CPUOffloadEnabled,
                "tp_group": None,
                "tp_size": 1,
                "sequence_parallel": False,
                "tensor_parallel": False,
                "activation_dtype": inp.dtype,
                "parallel_mode": None,
                "is_grad_enabled": is_grad_enabled,
                "ub_overlap_rs": False,
                "ub_overlap_ag": False,
                "ub_name": None,
                "fp8_output": False,
                "fsdp_group": self.pg,
                "module": self,
                "skip_fp8_weight_update": None,
            }

            # Optimistic key value insertion for the sake of compatibility with main branch
            for param_name in params:
                if param_name not in kwargs:
                    param = params[param_name]
                    if param.default is not param.empty:
                        kwargs[param_name] = param.default
                    else:
                        kwargs[param_name] = None

            # Remove kwargs if they are not used in the current version.
            unused_kwargs = set(kwargs.keys()) - set(params)
            for unused_kwarg in unused_kwargs:
                kwargs.pop(unused_kwarg)

            out = _Linear.forward(**kwargs)
            ctx = ctx if is_grad_enabled else None
            saved_tensors = ctx.pop_saved_tensors() if is_grad_enabled else None
            return out, saved_tensors, ctx

    def _get_quantizers(self, is_grad_enabled):
        # NOTE: Currently, we disallow changing these settings.
        fp8_output = False
        fp8_grad = False

        if not self.fp8:
            return [None] * 5
        grad_input_quantizer = None
        grad_output_quantizer = None
        output_quantizer = None
        input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
        input_quantizer.internal = True
        weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        if fp8_output:
            output_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
        if is_grad_enabled:
            grad_output_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
            grad_output_quantizer.internal = True
            if fp8_grad:
                grad_input_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT1]
        return (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_output_quantizer,
            grad_input_quantizer,
        )


# # # # # # # # #
# Make Executor for TE
# # # # # # # # #
transformer_engine_ex = OperatorExecutor("transformer_engine")
register_executor(transformer_engine_ex)


def make_te_linear_meta(i_requires_grad, w_requires_grad):
    def _te_functional_linear_meta(
        a: TensorProxy,
        w: TensorProxy,
        bias: None | TensorProxy,
        *,
        input_requires_grad: bool,
        weight_requires_grad: bool,
        bias_requires_grad: bool,
    ) -> tuple[TensorProxy, AnyProxy | None]:
        from thunder.core.dtypes import float8_e4m3fn, uint8

        # Input Shape : (*, Hin)
        # Output Shape : (*, Hout) where * is any number of dims including None.
        output_shape = list(a.shape)
        output_shape[-1] = w.shape[0]
        if i_requires_grad or w_requires_grad:
            global LINEAR_CALLS_COUNTER
            ctx_dict = AnyProxy(object(), prefix=f"ctx_te_{LINEAR_CALLS_COUNTER}")

            # It's not critical to model the exact shape and dtype of
            # saved_tensors since they are not used in Thunder's meta functions.
            saved_tensors = tuple(
                TensorProxy(like=a, shape=a.shape)
                for _ in range(_get_num_saved_tensors(get_recipe_from_options_or_default_recipe(), w_requires_grad))
            )

            return TensorProxy(like=a, shape=output_shape), saved_tensors, ctx_dict
        return TensorProxy(like=a, shape=output_shape), None, None

    return _te_functional_linear_meta


#
# Registers the backward function
#
def _te_functional_linear_backward_impl(
    a_shape: tuple,
    w_shape: tuple,
    b_shape: tuple | None,
    ctx: Context,
    saved_tensors: Sequence[torch.Tensor],
    g: torch.Tensor,
    *,
    input_requires_grad: bool,
    weight_requires_grad: bool,
    bias_requires_grad: bool,
) -> [torch.Tensor, torch.Tensor, None | torch.Tensor]:
    with set_saved_tensors(ctx, saved_tensors):
        grads = _Linear.backward(ctx, g)

    grad_inputs = (grads[1], grads[0], grads[2])
    return grad_inputs


def _te_functional_linear_backward_meta(
    a_shape: tuple,
    w_shape: tuple,
    b_shape: tuple | None,
    ctx: Context,
    saved_tensors: Sequence[TensorProxy],
    g: TensorProxy,
    *,
    input_requires_grad: bool,
    weight_requires_grad: bool,
    bias_requires_grad: bool,
) -> [TensorProxy, TensorProxy, None | TensorProxy]:
    return (
        TensorProxy(like=g, shape=a_shape) if input_requires_grad else None,
        TensorProxy(like=g, shape=w_shape) if weight_requires_grad else None,
        TensorProxy(like=g, shape=b_shape) if bias_requires_grad else None,
    )


te_functional_linear_backward = transformer_engine_ex.register_operator(
    "te_functional_linear_backward", meta=_te_functional_linear_backward_meta, fn=_te_functional_linear_backward_impl
)

LINEAR_CALLS_COUNTER = 0

if TE_AVAILABLE:
    # Recipe is chosen based on hardware platform
    # For H100 or lower, it returns DelayedScaling recipe.
    # For B200, it returns MXFP8BlockScaling recipe.
    _DEFAULT_RECIPE = get_default_fp8_recipe()

IMPORT_CTX_TE_KEY = "transformer_engine"
FP8_RECIPE_KEY = "te_fp8_recipe"


def get_recipe_from_options_or_default_recipe():
    desc = "transformer_engine_ex: Optional fp8_recipe for `fp8_autocast` context manager."
    if (fp8_recipe := get_compile_option(FP8_RECIPE_KEY, desc)) is None:
        fp8_recipe = _DEFAULT_RECIPE

    return fp8_recipe


def _create_fp8_linear_bound_symbol(
    a: TensorProxy, w: TensorProxy, b: TensorProxy, input_requires_grad=True, weight_requires_grad=True
) -> tuple[torch.Tensor, AnyProxy | None]:
    linear_fn = partial(
        TELinear(w.shape[1], w.shape[0]),
    )
    global LINEAR_CALLS_COUNTER
    name = f"te_linear_{LINEAR_CALLS_COUNTER}"

    fp8_recipe = get_recipe_from_options_or_default_recipe()

    def bind_postprocess(bsym: BoundSymbol) -> None:
        # This dict is then used by trace.python_ctx() to resolve the
        # BoundSymbol to the actual function.
        bsym._call_ctx: dict[str, Callable] = {name: linear_fn}
        bsym._import_ctx: dict[str, Any] = {IMPORT_CTX_TE_KEY: te}
        bsym._object_ctx: dict[str, Any] = {FP8_RECIPE_KEY: fp8_recipe}

    meta_fn = make_te_linear_meta(input_requires_grad, weight_requires_grad)
    sym = Symbol(
        name=name,
        meta=meta_fn,
        is_prim=True,
        executor=transformer_engine_ex,
        _bind_postprocess=bind_postprocess,
        tags=(prims.OpTags.DONT_RECOMPUTE_IN_BACKWARD,),
    )
    LINEAR_CALLS_COUNTER += 1
    return sym, meta_fn


# Creates a new stateful operator for each invocation of `linear`.
def _insert_fp8_linear_bound_symbol_in_current_trace(
    a: TensorProxy, w: TensorProxy, b: TensorProxy, input_requires_grad=True, weight_requires_grad=True
) -> tuple[torch.Tensor, AnyProxy | None]:
    is_grad_enabled = input_requires_grad or weight_requires_grad
    sym, meta_fn = _create_fp8_linear_bound_symbol(a, w, b, input_requires_grad, weight_requires_grad)

    # Value for `input_requires_grad`, `weight_requires_grad` and `bias_requires_grad` are default.
    # The correct values are set by looking at the backward trace.
    # See NOTE: Implementation of _transformer_engine_set_requires_grad
    bsym = sym.bind(
        a,
        w,
        b,
        output=meta_fn(
            a,
            w,
            b,
            input_requires_grad=input_requires_grad,
            weight_requires_grad=weight_requires_grad,
            bias_requires_grad=True if b is not None else False,
        ),
        input_requires_grad=input_requires_grad,
        weight_requires_grad=weight_requires_grad,
        bias_requires_grad=True if b is not None else False,
    )

    # Now we need to append the BoundSymbol to the current trace.
    trace = get_tracectx()
    trace.scopes[-1].append(bsym)
    for p in chain(bsym.flat_proxy_outs, bsym.flat_proxy_args):
        trace.names.add(p.name)

    # Used in augmented forward rule.
    # Returns are `result, saved_tensors, ctx`.
    if is_grad_enabled:
        return bsym.output

    return bsym.output[0]


#
# Registers transformer_engine_ex as an executor for torch.nn.functional.linear
#


def _linear_checker(
    a: TensorProxy,
    w: TensorProxy,
    bias: None | TensorProxy,
) -> bool:
    # Make sure that we don't claim an operator
    # if `TransformerEngine` is not available (not installed or version requirements not met)
    # and it is passed as an executor to `thunder.jit()`
    if not TE_AVAILABLE:
        return False

    def is_cuda(t):
        return t.device.devicetype == devices.DeviceType.CUDA

    inputs = (a, w)
    if bias is not None:
        inputs = inputs + (bias,)

    # Helper function as input shape can be (*, Hin)
    def _view_input_as_2d(x):
        shape = x.shape
        return x.view((-1, shape[-1]))

    fp8_recipe = get_recipe_from_options_or_default_recipe()

    def check_valid_fp8_shapes(a):
        # DelayedScaling and MXFP8BlockScaling have different shape requirements.
        if isinstance(fp8_recipe, DelayedScaling):
            return check_dim_for_fp8_exec(a)

        assert isinstance(fp8_recipe, MXFP8BlockScaling)
        shape = a.shape
        return shape[0] % MXFP8_BLOCK_SCALING_SIZE == 0 and shape[1] % MXFP8_BLOCK_SCALING_SIZE == 0

    # Inputs must be on CUDA and
    # input sizes must satisfy -> dim0 is divisible by 8 and dim1 is divisible by 16.
    return all(map(is_cuda, inputs)) and check_valid_fp8_shapes(_view_input_as_2d(a)) and check_valid_fp8_shapes(w)


def linear_forward_rule(a, w, bias):
    out, saved_tensors, ctx = _insert_fp8_linear_bound_symbol_in_current_trace(a, w, bias)
    primal = out
    saved_for_backward = (
        a.shape,
        w.shape,
        bias.shape if bias is not None else None,
        ctx,
        saved_tensors,
        True,  # Dummy default for input_requires_grad updated in `_transformer_engine_set_requires_grad`
        True,  # Dummy default for weight_requires_grad in `_transformer_engine_set_requires_grad`
        True if bias is not None else False,
    )
    return primal, saved_for_backward


# Translate calls from torch.nn.functional.linear to te.Linear (when the checker above returns True)
def _linear_transform(a: TensorProxy, w: TensorProxy, b: TensorProxy) -> torch.Tensor:
    return _insert_fp8_linear_bound_symbol_in_current_trace(
        a, w, b, input_requires_grad=False, weight_requires_grad=False
    )


@disable_caching_split_forward_and_backward
def _linear_grad(a: TensorProxy, w: TensorProxy, b: TensorProxy) -> TensorProxy:
    out, saved_for_backward = linear_forward_rule(a, w, b)
    input_requires_grad, weight_requires_grad, bias_requires_grad = saved_for_backward[-3:]
    g = prims.get_grad(out)
    ga, gw, gb = te_functional_linear_backward(
        *saved_for_backward[:-3],
        g,
        input_requires_grad=input_requires_grad,
        weight_requires_grad=weight_requires_grad,
        bias_requires_grad=bias_requires_grad,
    )
    if input_requires_grad:
        prims.put_grad(a, ga)
    if weight_requires_grad:
        prims.put_grad(w, gw)
    if bias_requires_grad:
        prims.put_grad(b, gb)
    return out


# Registers the implementation for torch.nn.functional.linear
transformer_engine_ex.register_implementation(
    prims.linear,
    checker=_linear_checker,
    execution_transform=_linear_transform,
    grad_transform=_linear_grad,
)


def _is_te_linear_enabled(import_ctx, object_ctx):
    # These keys are present in `import_ctx` and `object_ctx` only if
    # we actually replaced a linear call with a new TE operator.
    is_te_exec_enabled = IMPORT_CTX_TE_KEY in import_ctx and FP8_RECIPE_KEY in object_ctx
    return is_te_exec_enabled


TE_CTX_STR = f"@{IMPORT_CTX_TE_KEY}.fp8_autocast(fp8_recipe={FP8_RECIPE_KEY})"


def _get_te_wrapper_string():
    return TE_CTX_STR


def te_sync_fp8_meta_bwd_meta():
    pass


def te_sync_fp8_meta_bwd_impl():
    FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)


te_sync_fp8_meta_bwd = transformer_engine_ex.register_operator(
    "te_sync_fp8_meta_bwd", meta=te_sync_fp8_meta_bwd_meta, fn=te_sync_fp8_meta_bwd_impl
)


def _transformer_engine_set_requires_grad(fw_extrace: TraceCtx, bw_extrace: TraceCtx) -> tuple[TraceCtx, TraceCtx]:
    # NOTE: Implementation of _transformer_engine_set_requires_grad
    # This function determines the requires_grad for input, weight and bias based on whether the
    # gradients are returned from the backward trace.
    # 1. First, it analyzes the backward trace to determine which gradients (input, weight and bias)
    #    are actually needed. This information is stored in a mapping from
    #    context names to (input_requires_grad, weight_requires_grad, bias_requires_grad) tuples.
    # 2. Then, it updates the forward trace's corresponding bound symbols to pass the correct requires_grad flags.
    #    It also updates the relevant BoundSymbol's output to pass the correct saved_tensors.
    # 3. Finally, it updates the backward trace to pass the correct saved_tensors and update the `te_functional_linear_backward` to
    #    consume the correct saved_tensors.
    # This is required in `PEFT` settings to avoiding unncessary input, weight and bias gradients computation.

    # Step 1: Analyze the backward trace to determine which gradients (input, weight and bias) are actually needed.
    updated_fw_extrace = from_trace(fw_extrace)
    updated_bw_extrace = from_trace(bw_extrace)
    ctx_to_requires_grad = {}
    CTX_ARG_IDX = 3
    SAVED_TENSORS_ARG_IDX = 4
    UNPACK_SEQUENCE_BSYM_IDX = 4
    for bsym in bw_extrace.bound_symbols:
        if bsym.sym.name == "te_functional_linear_backward":
            # Check if `i_requires_grad`, `w_requires_grad` and `b_requires_grad` are used or not.
            # If they are unused, they would show up as `None` in the output of the BoundSymbol.
            dgrad, wgrad, bgrad = bsym.output
            i_requires_grad = True if dgrad is not None else False
            w_requires_grad = True if wgrad is not None else False
            b_requires_grad = True if bgrad is not None else False

            # Update the BoundSymbol to take the correct flags as input.
            ctx_to_requires_grad[bsym.args[CTX_ARG_IDX].name] = (i_requires_grad, w_requires_grad, b_requires_grad)
            args = list(bsym.args)
            bsym.kwargs["input_requires_grad"] = i_requires_grad
            bsym.kwargs["weight_requires_grad"] = w_requires_grad
            bsym.kwargs["bias_requires_grad"] = b_requires_grad
            bsym.args = tuple(args)
        updated_bw_extrace.bound_symbols.append(bsym)

    # Step 2: Update the forward trace to pass the correct requires_grad flags to the `TELinear` module.
    #         Also, it updates the BoundSymbol's output to pass the correct number of saved_tensors.
    tensors_to_remove = []
    ctx_to_new_saved_tensor_len = {}
    for bsym in fw_extrace.bound_symbols:
        if "te_linear" in bsym.sym.name:
            # Update the BoundSymbol to correctly pass the requires_grad flags to the `TELinear` module.
            i_requires_grad, w_requires_grad, b_requires_grad = ctx_to_requires_grad[bsym.output[-1].name]
            args = list(bsym.args)
            org_i_requires_grad = bsym.kwargs["input_requires_grad"]
            org_w_requires_grad = bsym.kwargs["weight_requires_grad"]
            org_b_requires_grad = bsym.kwargs["bias_requires_grad"]
            bsym.kwargs["input_requires_grad"] = i_requires_grad
            bsym.kwargs["weight_requires_grad"] = w_requires_grad
            bsym.kwargs["bias_requires_grad"] = b_requires_grad
            args = tuple(args)
            bsym.args = tuple(args)

            # NOTE: Changing the requires_grad on bias doesn't change the number of saved_tensors.
            if (org_i_requires_grad, org_w_requires_grad, org_b_requires_grad) != (
                i_requires_grad,
                w_requires_grad,
                b_requires_grad,
            ):
                with tracectx(updated_fw_extrace):
                    a, w, b = args[:3]
                    new_args = (a, w, b, i_requires_grad, w_requires_grad)
                    _, meta_fn = _create_fp8_linear_bound_symbol(*new_args)
                    output = meta_fn(
                        a,
                        w,
                        b,
                        input_requires_grad=i_requires_grad,
                        weight_requires_grad=w_requires_grad,
                        bias_requires_grad=b_requires_grad,
                    )

                o, saved_tensors, ctx = bsym.output
                _, new_saved_tensors, _ = output
                # Update the BoundSymbol's output to pass the correct number of saved_tensors
                # computed based on the requires_grad flags.
                bsym.output = (o, saved_tensors[: len(new_saved_tensors)], ctx)
                ctx_to_new_saved_tensor_len[ctx.name] = len(new_saved_tensors)
                for t in saved_tensors[len(new_saved_tensors) :]:
                    tensors_to_remove.append(t.name)

        updated_fw_extrace.bound_symbols.append(bsym)

    from thunder.core.vjp_utils import get_saved_for_backward_tensors, set_saved_for_backward_tensors

    # Update the forward trace's return BoundSymbol to pass the correct number of saved_tensors.
    saved_tensors = get_saved_for_backward_tensors(updated_fw_extrace)
    new_saved_tensors = [tensor for tensor in saved_tensors if tensor.name not in tensors_to_remove]
    set_saved_for_backward_tensors(updated_fw_extrace, new_saved_tensors)

    # Update the backward trace's unpack_sequence for saved tensors to unpack the new sequence of saved tensors.
    assert updated_bw_extrace.bound_symbols[UNPACK_SEQUENCE_BSYM_IDX].sym.id == prims.PrimIDs.UNPACK_SEQUENCE
    assert updated_bw_extrace.bound_symbols[UNPACK_SEQUENCE_BSYM_IDX].args[0].name == "C0"
    updated_bw_extrace.bound_symbols[UNPACK_SEQUENCE_BSYM_IDX].args = (
        updated_bw_extrace.bound_symbols[UNPACK_SEQUENCE_BSYM_IDX].args[0],
        len(new_saved_tensors),
    )
    updated_bw_extrace.bound_symbols[UNPACK_SEQUENCE_BSYM_IDX].output = new_saved_tensors

    # Update the backward trace's BoundSymbol to pass the correct number of saved_tensors to the `te_functional_linear_backward` operator.
    for bsym in updated_bw_extrace.bound_symbols:
        if "te_functional_linear" in bsym.sym.name and bsym.args[CTX_ARG_IDX].name in ctx_to_new_saved_tensor_len:
            new_saved_tensor_len = ctx_to_new_saved_tensor_len[bsym.args[CTX_ARG_IDX].name]
            args = list(bsym.args)
            args[SAVED_TENSORS_ARG_IDX] = args[SAVED_TENSORS_ARG_IDX][:new_saved_tensor_len]
            bsym.args = tuple(args)

    updated_fw_extrace.set_provenance(TraceProvenance(f"TransformerEngine update weight and bias requires_grad pass"))
    updated_bw_extrace.set_provenance(TraceProvenance(f"TransformerEngine update weight and bias requires_grad pass"))
    return updated_fw_extrace, updated_bw_extrace


def _transformer_engine_bwd_fp8_meta_sync(fw_extrace, bw_extrace) -> tuple[TraceCtx, TraceCtx]:
    updated_fw_extrace, updated_bw_extrace = _transformer_engine_set_requires_grad(fw_extrace, bw_extrace)

    # See doc of `_insert_bwd_fp8_meta_sync` for more details.
    # `bw_extrace` is mutated in place.
    _insert_bwd_fp8_meta_sync(updated_bw_extrace)
    return updated_fw_extrace, updated_bw_extrace


def _insert_bwd_fp8_meta_sync(bw_extrace):
    # This functions insert the symbol `te_sync_fp8_meta_bwd` to the end of the backward
    # trace which takes care of syncing and updating the FP8 metadata for backward tensors.
    # See NOTE: Backward FP8 metadata sync
    bwd_idx = len(bw_extrace.bound_symbols) - 1
    bw_extrace.bound_symbols.insert(bwd_idx, te_sync_fp8_meta_bwd.bind(output=None))
