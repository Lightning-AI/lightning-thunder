from functools import partial
from itertools import chain
from typing import Any
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from collections import deque
import warnings

import torch

from lightning_utilities.core.imports import package_available

from thunder.core.proxies import TensorProxy
from thunder.core.trace import get_tracectx
from thunder.core.symbol import Symbol, BoundSymbol
import thunder.core.devices as devices
import thunder.torch as ltorch
from thunder.core.proxies import TensorProxy, CollectionProxy
from thunder.core.symbol import Symbol
from thunder.core.transforms import (
    register_augmented_forward_with_checker,
    register_backward,
)
from thunder.extend import OperatorExecutor, register_executor

__all__ = [
    "transformer_engine_ex",
]

TE_AVAILABLE: bool = package_available("transformer_engine")

te: None | Any = None
if TE_AVAILABLE:
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        import transformer_engine_extensions as tex
        from transformer_engine.pytorch.constants import TE_DType
        from transformer_engine.pytorch.module.linear import _Linear
        from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
        from transformer_engine.pytorch.utils import check_dim_for_fp8_exec
    except Exception as ex:
        warnings.warn(f"transformer_engine failed to import with exception {ex}")
        TE_AVAILABLE = False


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
# when replacing calls to `ltorch.linear` (see `_create_fp8_linear_bound_symbol`).
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
# @no_autocast()
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
# This `ctx` is converted to dictionary to be passed as a residual. See `to_dict` and `from_dict`.
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


class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def to_dict(self):
        return {
            "saved_tensors": self.saved_tensors,
            "activation_dtype": self.activation_dtype,
            "fp8": self.fp8,
            "fp8_meta": self.fp8_meta,
            "fuse_wgrad_accumulation": self.fuse_wgrad_accumulation,
            "is_first_microbatch": self.is_first_microbatch,
            "use_bias": self.use_bias,
            "sequence_parallel": self.sequence_parallel,
            "tensor_parallel": self.tensor_parallel,
            "inp_shape": self.inp_shape,
            "parallel_mode": self.parallel_mode,
            "tp_group": self.tp_group,
            "ub_split_ag": self.ub_split_ag,
            "ub_atomic_gemm_ag": self.ub_atomic_gemm_ag,
            "ub_name": self.ub_name,
            "tp_size": self.tp_size,
            "requires_dgrad": self.requires_dgrad,
        }

    @staticmethod
    def from_dict(d):
        ctx = Context()
        ctx.saved_tensors = d["saved_tensors"]
        ctx.activation_dtype = d["activation_dtype"]
        ctx.fp8 = d["fp8"]
        ctx.fp8_meta = d["fp8_meta"]
        ctx.fuse_wgrad_accumulation = d["fuse_wgrad_accumulation"]
        ctx.is_first_microbatch = d["is_first_microbatch"]
        ctx.use_bias = d["use_bias"]
        ctx.sequence_parallel = d["sequence_parallel"]
        ctx.tensor_parallel = d["tensor_parallel"]
        ctx.inp_shape = d["inp_shape"]
        ctx.parallel_mode = d["parallel_mode"]
        ctx.tp_group = d["tp_group"]
        ctx.ub_split_ag = d["ub_split_ag"]
        ctx.ub_atomic_gemm_ag = d["ub_atomic_gemm_ag"]
        ctx.ub_name = d["ub_name"]
        ctx.tp_size = d["tp_size"]
        ctx.requires_dgrad = d["requires_dgrad"]
        return ctx


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


class TELinear(TransformerEngineBaseModule):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Used by `get_fp8_weights_scratchpad`
        self.primary_weights_in_fp8 = False

        if FP8GlobalStateManager.with_fp8_parameters():
            raise RuntimeError("Primary weights in FP8 is not supported under `thunder.compile`.")

        # Required by `get_fp8_weights_scratchpad`
        self.fp8_weight_shapes.append(torch.Size((self.out_features, self.in_features)))

    def forward(self, inp, weight, bias, is_first_microbatch: bool | None = None, is_grad_enabled: bool = False):
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
            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_t_fp8 = self.get_fp8_weights_scratchpad(is_first_microbatch)

            ctx = Context() if is_grad_enabled else None

            # Currently we support only non-distributed case.
            # We hard-code the arguments related to distributed for now.
            args = (
                ctx,
                weight,
                weight1_fp8,
                weight1_t_fp8,
                inp,
                torch.Tensor() if bias is None else bias,  # bias_tensor
                bias is not None,
                None,  # is_first_microbatch
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                False,  # fuse_wgrad_accumulation
                None,  # tp_group
                1,  # tp_size
                self.sequence_parallel,
                False,  # tp_size > 1
                inp.dtype,
                None,  # parallel_mode
                is_grad_enabled,
                False,  # primary_weights_in_fp8
                False,  # ub_split_rs
                False,  # ub_split_ag
                False,  # ub_atomic_gemm_rs
                False,  # ub_atomic_gemm_ag
                None,  # ub_name
            )

            out = _Linear.forward(*args)
            ctx_dict = ctx.to_dict() if is_grad_enabled else None
            return out, ctx_dict

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
    ) -> tuple[TensorProxy, CollectionProxy | None]:
        # Input Shape : (*, Hin)
        # Output Shape : (*, Hout) where * is any number of dims including None.
        output_shape = list(a.shape)
        output_shape[-1] = w.shape[0]
        if is_grad_enabled:
            global LINEAR_CALLS_COUNTER
            ctx_dict = CollectionProxy({}, name=f"ctx_te_{LINEAR_CALLS_COUNTER}")

            return TensorProxy(like=a, shape=output_shape), ctx_dict
        return TensorProxy(like=a, shape=output_shape), None

    return _te_functional_linear_meta


#
# Registers the backward function
#
def _te_functional_linear_backward_impl(
    g: torch.Tensor, a_shape: tuple, w_shape: tuple, b_shape: tuple | None, ctx: dict
) -> [torch.Tensor, torch.Tensor, None | torch.Tensor]:
    ctx = Context.from_dict(ctx)
    # See [NOTE] Enable grad within context
    # _Linear.backward depends on requires grad of `weight/ctx.saved_tensors[2]`.
    # Hence we enable requires_grad for computation.
    # https://github.com/NVIDIA/TransformerEngine/blob/b957aa475bcbcf22405381d18bd7fefe4fb6b171/transformer_engine/pytorch/module/linear.py#L434
    with enable_grad(ctx.saved_tensors[2]):
        grads = _Linear.backward(ctx, g)
    grad_inputs = (grads[3], grads[0], grads[4])
    return grad_inputs


def _te_functional_linear_backward_meta(
    g: TensorProxy, a_shape: tuple, w_shape: tuple, b_shape: tuple, ctx_idx: Context
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


# Creates a new stateful operator for each invocation of `linear`.
def _create_fp8_linear_bound_symbol(
    a: TensorProxy, w: TensorProxy, b: TensorProxy, is_grad_enabled=False
) -> tuple[torch.Tensor, CollectionProxy | None]:
    linear_fn = partial(TELinear(w.shape[1], w.shape[0]), is_grad_enabled=is_grad_enabled)
    global LINEAR_CALLS_COUNTER
    name = f"te_linear_{LINEAR_CALLS_COUNTER}"

    def bind_postprocess(bsym: BoundSymbol) -> None:
        # This dict is then used by trace.python_ctx() to resolve the
        # BoundSymbol to the actual function.
        bsym._call_ctx: dict[str, Callable] = {name: linear_fn}

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
    # Returns are `result, ctx_id`.
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
    out, ctx_idx = _create_fp8_linear_bound_symbol(a, w, bias, is_grad_enabled=True)
    primal = out
    saved_for_backward = (a.shape, w.shape, bias.shape if bias is not None else None, ctx_idx)
    return primal, saved_for_backward


def linear_forward_rule_checker(a: TensorProxy, w: TensorProxy, bias: None | TensorProxy) -> bool:
    from thunder.core.compile_data import get_compile_data

    cd = get_compile_data()
    if transformer_engine_ex in cd.executors_list:
        return _linear_checker(a, w, bias)
    return False


register_augmented_forward_with_checker(
    transformer_engine_ex,
    ltorch.linear.id,
    linear_forward_rule_checker,
    linear_forwad_rule,
)


@register_backward((transformer_engine_ex, ltorch.linear.id))
def linear_backward_rule(a_shape, w_shape, b_shape, ctx_idx, grad):
    return te_functional_linear_backward(grad, a_shape, w_shape, b_shape, ctx_idx)


# Translate calls from torch.nn.functional.linear to te.Linear (when the checker above returns True)
def _linear_transform(a: TensorProxy, w: TensorProxy, b: TensorProxy) -> torch.Tensor:
    return _create_fp8_linear_bound_symbol(a, w, b, is_grad_enabled=False)


# Registers the implementation for torch.nn.functional.linear
transformer_engine_ex.register_implementation(
    ltorch.linear,
    checker=_linear_checker,
    execution_transform=_linear_transform,
)
