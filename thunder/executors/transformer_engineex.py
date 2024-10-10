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


__all__ = [
    "transformer_engine_ex",
]

TE_AVAILABLE: bool = package_available("transformer_engine")

# We rely on internal details of TransformerEngine like `_Linear` autograd.Function.
# As these details are not public, they can change
# Ex. addition of a positional argument for cpu_offloading (not as the last argument)
# between version 1.2 and 1.3.
# Hence, we have these guards based on version.
TE_VERSION_1_11_PLUS: bool = False

te: None | Any = None
if TE_AVAILABLE:
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        from transformer_engine.pytorch.module.linear import _Linear
        from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
        from transformer_engine.pytorch.utils import check_dim_for_fp8_exec
        from transformer_engine.pytorch.cpu_offload import CPUOffloadEnabled
        import transformer_engine_torch as tex
    except Exception as ex:
        warnings.warn(f"transformer_engine failed to import with exception {ex}")
        TE_AVAILABLE = False

    TE_VERSION_1_9_PLUS = LooseVersion(version("transformer_engine")) > LooseVersion("1.9")
    TE_VERSION_1_11_PLUS = LooseVersion(version("transformer_engine")) > LooseVersion("1.11")
    if not TE_VERSION_1_9_PLUS:
        warnings.warn(
            f"Installed version of transformer_engine {version('transformer_engine')} is not supported, please upgrade. `transformer_engine_ex` will not be used."
        )
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

    def forward(self, inp, weight, bias, is_first_microbatch: bool | None = None, is_grad_enabled: bool = False):
        # NOTE: Backward FP8 metadata sync
        # TransformerEngine v1.6 onwards, we control the sync and update of FP8 metadata for FP8 tensors
        # tied to backward pass (i.e. the gradient tensors)
        # Also, note that the forward tensor metadata sync occurs at the exit of `fp8_autocast` context manager
        # which is not controlled by us.
        #
        # We consume the `is_first_fp8_module` so that the automatic sync for FP8 metadata is disabled.
        FP8GlobalStateManager.is_first_fp8_module()  # Consume first module token.

        tensor_inputs = tuple(filter(lambda t: isinstance(t, torch.Tensor), (inp, weight, bias)))
        # See [NOTE] Enable grad within context
        # TE backward depends on `requires_grad` to compute grads.
        # so under grad mode we enable grad for input tensors
        # Ref: https://github.com/NVIDIA/TransformerEngine/blob/b957aa475bcbcf22405381d18bd7fefe4fb6b171/transformer_engine/pytorch/module/linear.py#L264
        grad_ctx = enable_grad(*tensor_inputs) if is_grad_enabled else nullcontext()
        with grad_ctx, self.prepare_forward(inp, is_first_microbatch) as inp:
            assert (
                self.fp8 or not self.primary_weights_in_fp8
            ), "Need to run inside fp8_autocast region when weights are stored in FP8."

            weight_fp8, weight_t_fp8 = self.get_fp8_weight_version_compat(
                weight=weight, is_first_microbatch=is_first_microbatch, is_grad_enabled=is_grad_enabled
            )

            ctx = Context() if is_grad_enabled else None

            import inspect

            params = inspect.signature(_Linear.forward).parameters

            # Currently we do not support `tp` meaning tensor model parallel case.
            # We hard-code the arguments related to distributed for now.
            use_bias = bias is not None
            kwargs = {
                "ctx": ctx,
                "weight": weight,
                "weight_fp8": weight_fp8,
                "inp": inp,
                "bias": torch.tensor([]) if not use_bias else bias,
                "use_bias": bias is not None,
                "is_first_microbatch": None,
                "fp8": self.fp8,
                "fp8_calibration": self.fp8_calibration,
                "fp8_meta": self.fp8_meta,
                "fuse_wgrad_accumulation": False,
                "tp_group": None,
                "tp_size": 1,
                "sequence_parallel": self.sequence_parallel,
                "tensor_parallel": False,
                "activation_dtype": inp.dtype,
                "parallel_mode": None,
                "is_grad_enabled": is_grad_enabled,
                "ub_name": None,
                "cpu_offloading": CPUOffloadEnabled,
                # ref: https://github.com/NVIDIA/TransformerEngine/blame/7c1828f80edc1405d4ef1a7780c9e0046beab5c7/transformer_engine/pytorch/module/linear.py#L84-L85
                "ub_overlap_rs": False,
                "ub_overlap_ag": False,
                "fsdp_group": self.pg,
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

    def get_fp8_weight_version_compat(self, weight, is_first_microbatch, is_grad_enabled):
        weight_t_fp8: torch.Tensor = None
        weight_fp8: torch.Tensor = None
        # Fetch the fp8 weights placeholders (for linear/gemm)
        # Initialize FP8 weights workspace if needed
        # FP8 cast to workspace buffer
        update_workspace = is_first_microbatch is None or is_first_microbatch
        skip_fp8_weight_update = None

        weight_fp8 = self.get_fp8_workspace(
            tensor=weight,
            fp8_meta_forward=True,
            fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
            cache_name=(None if is_first_microbatch is None else "weight"),
            update_workspace=update_workspace,
            skip_update_flag=skip_fp8_weight_update,
            **{"with_transpose": is_grad_enabled} if not TE_VERSION_1_11_PLUS else {},
        )

        return weight_fp8, weight_t_fp8

    # This method is used for supporting TE v1.6 and v1.7.
    # v1.8 onwards the implementation of this has moved to `TransformerEngineBaseModule`
    # See `get_fp8_workspace`: https://github.com/NVIDIA/TransformerEngine/blob/8b210490b3f46cd409df0ba6a8f4b14273f2975c/transformer_engine/pytorch/module/base.py#L753-L754
    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: bool | None,
    ) -> list["Float8Tensor"]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`) or return empty fp8 weight
        tensors (if `is_first_microbatch is None`)
        """
        if not self.fp8 or self.primary_weights_in_fp8:
            return [None, None]

        if is_first_microbatch is None:
            # Return empty weight placeholders for each fwd/bwd pass
            fp8_weight_tensors = self.get_fp8_weights_empty_tensors(is_first_microbatch)
        else:
            # These persistent weight placeholders should've been created in
            # `set_fp8_weights` method
            fp8_weight_tensors = [self.weight1_fp8, self.weight1_t_fp8]

        return fp8_weight_tensors


# # # # # # # # #
# Make Executor for TE
# # # # # # # # #
transformer_engine_ex = OperatorExecutor("transformer_engine")
register_executor(transformer_engine_ex)


def make_te_linear_meta(is_grad_enabled: bool = False):
    def _te_functional_linear_meta(
        a: TensorProxy, w: TensorProxy, bias: None | TensorProxy
    ) -> tuple[TensorProxy, AnyProxy | None]:
        from thunder.core.dtypes import float8_e4m3fn

        # Input Shape : (*, Hin)
        # Output Shape : (*, Hout) where * is any number of dims including None.
        output_shape = list(a.shape)
        output_shape[-1] = w.shape[0]
        if is_grad_enabled:
            global LINEAR_CALLS_COUNTER
            ctx_dict = AnyProxy(object(), name=f"ctx_te_{LINEAR_CALLS_COUNTER}")

            # It's not critical to model the exact shape and dtype of
            # saved_tensors since they are not used in Thunder's meta functions.
            if TE_VERSION_1_11_PLUS:
                # NOTE - Position of scaling tensor denoted by `scaling_fwd` is different (compared to `else`).
                # https://github.com/NVIDIA/TransformerEngine/blob/7fc50f489b8184fbd93efd4e48140ad0264e362b/transformer_engine/pytorch/module/linear.py#L330C13-L337C14
                saved_tensors = (
                    TensorProxy(like=a, shape=a.shape),  # saved_inputmat
                    TensorProxy(like=a, shape=(a.shape[:-2] + (a.shape[-1], a.shape[-2]))),  # saved_inputmat_t
                    TensorProxy(like=a, shape=(1,)),  # scaling_fwd
                    TensorProxy(like=w, shape=w.shape),  # weight
                    TensorProxy(like=w, shape=(w.shape[1], w.shape[0]), dtype=float8_e4m3fn),  # weight_fp8
                    # fuse_wgrad_accumulation is False
                    # https://github.com/Lightning-AI/lightning-thunder/blob/40da5bd5fabc30e99883d74b70c6a7d7fd61a828/thunder/executors/transformer_engineex.py#L224
                    None,  # weight.main_grad if cpu_offloading and fuse_wgrad_accumulation else None,
                )
            else:
                # https://github.com/NVIDIA/TransformerEngine/blob/37280ecd5e9c6087d18fbe2e668f2ec7761ada3d/transformer_engine/pytorch/module/linear.py#L323-L330
                saved_tensors = (
                    TensorProxy(like=a, shape=a.shape),  # saved_inputmat
                    TensorProxy(like=a, shape=(a.shape[:-2] + (a.shape[-1], a.shape[-2]))),  # saved_inputmat_t
                    TensorProxy(like=w, shape=w.shape),  # weight
                    TensorProxy(like=w, shape=(w.shape[1], w.shape[0]), dtype=float8_e4m3fn),  # weight_fp8
                    # fuse_wgrad_accumulation is False
                    # https://github.com/Lightning-AI/lightning-thunder/blob/40da5bd5fabc30e99883d74b70c6a7d7fd61a828/thunder/executors/transformer_engineex.py#L224
                    None,  # weight.main_grad if cpu_offloading and fuse_wgrad_accumulation else None,
                    TensorProxy(like=a, shape=(1,)),  # scaling_fwd
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
) -> [torch.Tensor, torch.Tensor, None | torch.Tensor]:
    # See [NOTE] Enable grad within context
    # _Linear.backward depends on requires grad of `weight/ctx.saved_tensors[2]`.
    # NOTE - weight is ctx.saved_tensors[3] from TE v1.11 onwards
    # Hence we enable requires_grad for computation.
    # https://github.com/NVIDIA/TransformerEngine/blob/b957aa475bcbcf22405381d18bd7fefe4fb6b171/transformer_engine/pytorch/module/linear.py#L434
    weight_t = saved_tensors[3] if TE_VERSION_1_11_PLUS else saved_tensors[2]
    with set_saved_tensors(ctx, saved_tensors), enable_grad(weight_t):
        grads = _Linear.backward(ctx, g)

    # Due to different in `_Linear.forward` API, position of
    # returned grad has changed.
    grad_inputs = (grads[2], grads[0], grads[3])
    return grad_inputs


def _te_functional_linear_backward_meta(
    a_shape: tuple,
    w_shape: tuple,
    b_shape: tuple | None,
    ctx: Context,
    saved_tensors: Sequence[TensorProxy],
    g: TensorProxy,
) -> [TensorProxy, TensorProxy, None | TensorProxy]:
    return (
        TensorProxy(like=g, shape=a_shape),
        TensorProxy(like=g, shape=w_shape),
        TensorProxy(like=g, shape=b_shape) if b_shape else None,
    )


te_functional_linear_backward = transformer_engine_ex.register_operator(
    "te_functional_linear_backward", meta=_te_functional_linear_backward_meta, fn=_te_functional_linear_backward_impl
)

LINEAR_CALLS_COUNTER = 0

if TE_AVAILABLE:
    _DEFAULT_RECIPE = recipe.DelayedScaling()

IMPORT_CTX_TE_KEY = "transformer_engine"
FP8_RECIPE_KEY = "te_fp8_recipe"


# Creates a new stateful operator for each invocation of `linear`.
def _create_fp8_linear_bound_symbol(
    a: TensorProxy, w: TensorProxy, b: TensorProxy, is_grad_enabled=False
) -> tuple[torch.Tensor, AnyProxy | None]:
    linear_fn = partial(TELinear(w.shape[1], w.shape[0]), is_grad_enabled=is_grad_enabled)
    global LINEAR_CALLS_COUNTER
    name = f"te_linear_{LINEAR_CALLS_COUNTER}"

    desc = "transformer_engine_ex: Optional fp8_recipe for `fp8_autocast` context manager."
    if (fp8_recipe := get_compile_option(FP8_RECIPE_KEY, desc)) is None:
        fp8_recipe = _DEFAULT_RECIPE

    def bind_postprocess(bsym: BoundSymbol) -> None:
        # This dict is then used by trace.python_ctx() to resolve the
        # BoundSymbol to the actual function.
        bsym._call_ctx: dict[str, Callable] = {name: linear_fn}
        bsym._import_ctx: dict[str, Any] = {IMPORT_CTX_TE_KEY: te}
        bsym._object_ctx: dict[str, Any] = {FP8_RECIPE_KEY: fp8_recipe}

    meta_fn = make_te_linear_meta(is_grad_enabled=is_grad_enabled)
    sym = Symbol(
        name=name, meta=meta_fn, is_prim=True, executor=transformer_engine_ex, _bind_postprocess=bind_postprocess
    )
    bsym = sym.bind(a, w, b, output=meta_fn(a, w, b))

    # Now we need to append the BoundSymbol to the current trace.
    trace = get_tracectx()
    trace.scopes[-1].append(bsym)
    for p in chain(bsym.flat_proxy_outs, bsym.flat_proxy_args):
        trace.names.add(p.name)

    LINEAR_CALLS_COUNTER += 1

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

    # Inputs must be on CUDA and
    # input sizes must satisfy -> dim0 is divisible by 8 and dim1 is divisible by 16.
    return all(map(is_cuda, inputs)) and check_dim_for_fp8_exec(_view_input_as_2d(a)) and check_dim_for_fp8_exec(w)


def linear_forwad_rule(a, w, bias):
    out, saved_tensors, ctx = _create_fp8_linear_bound_symbol(a, w, bias, is_grad_enabled=True)
    primal = out
    saved_for_backward = (a.shape, w.shape, bias.shape if bias is not None else None, ctx, saved_tensors)
    return primal, saved_for_backward


# Translate calls from torch.nn.functional.linear to te.Linear (when the checker above returns True)
def _linear_transform(a: TensorProxy, w: TensorProxy, b: TensorProxy) -> torch.Tensor:
    return _create_fp8_linear_bound_symbol(a, w, b, is_grad_enabled=False)


@disable_caching_split_forward_and_backward
def _linear_grad(a: TensorProxy, w: TensorProxy, b: TensorProxy) -> TensorProxy:
    out, saved_for_backward = linear_forwad_rule(a, w, b)
    g = prims.get_grad(out)
    ga, gw, gb = te_functional_linear_backward(*saved_for_backward, g)
    prims.put_grad(a, ga)
    prims.put_grad(w, gw)
    if b is not None:
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


def _transformer_engine_bwd_fp8_meta_sync(fw_extrace, bw_extrace):
    # See doc of `_insert_bwd_fp8_meta_sync` for more details.
    _insert_bwd_fp8_meta_sync(bw_extrace)


def _insert_bwd_fp8_meta_sync(bw_extrace):
    # This functions insert the symbol `te_sync_fp8_meta_bwd` to the end of the backward
    # trace which takes care of syncing and updating the FP8 metadata for backward tensors.
    # See NOTE: Backward FP8 metadata sync
    bwd_idx = len(bw_extrace.bound_symbols) - 1
    bw_extrace.bound_symbols.insert(bwd_idx, te_sync_fp8_meta_bwd.bind(output=None))
