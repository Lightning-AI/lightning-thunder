from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torchao.prototype.mx_formats.nvfp4_tensor as nvfp4_tensor

import thunder
from thunder.core import prims
from thunder.core.trace import tracectx
from thunder.transforms.utils import get_checks, trace_with_replaced_proxy_metadata, add_trace_output

if TYPE_CHECKING:
    from thunder.core.proxies import TensorProxy
    from thunder.core.symbols import BoundSymbol


nvfp4_executor = thunder.extend.OperatorExecutor("nvfp4_executor", version=0.1)

BLOCK_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def compute_per_tensor_scale(t: torch.Tensor) -> torch.Tensor:
    return ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / t.float().abs().amax()).to(torch.float32)


def quantize_fn(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        per_tensor_scale = compute_per_tensor_scale(t)
        qs, qw = nvfp4_tensor.nvfp4_quantize(t, per_tensor_scale=per_tensor_scale)

        # Swizzle the scales
        M, K = t.shape[0], t.shape[1]
        scale_shape = (M, K // BLOCK_SIZE)
        qs = nvfp4_tensor.to_blocked(qs.view(scale_shape)).flatten()

    return qw, qs, per_tensor_scale


# https://github.com/pytorch/ao/blob/4dffb40280ea7b0e1732c580d08df58d0134c543/torchao/prototype/mx_formats/nvfp4_tensor.py#L567-L568
def _nvfp4_linear(
    quantized_a,
    quantized_b,
    a_per_tensor_scale,
    b_per_tensor_scale,
    a_block_scales,
    b_block_scales,
    out_dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    quantized_b = quantized_b.t()

    assert quantized_a.is_contiguous()
    assert quantized_b.t().is_contiguous()
    # assert a._block_size == 16, f"NVFP4 requires block_size=16, got {a._block_size}"
    # assert b._block_size == 16, f"NVFP4 requires block_size=16, got {b._block_size}"
    assert bias is None

    # M, K = quantized_a.shape[0], quantized_a.shape[1]
    # N = quantized_b.shape[1]

    a_scale_blocked = a_block_scales  # Already swizzled
    b_scale_blocked = b_block_scales  # Already swizzled

    # Merge double quant scales into 1 scale for Scale_In^D
    scale_result = a_per_tensor_scale * b_per_tensor_scale

    # THIS IS A WORKAROUND:
    # RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling
    # When we have per-tensor scaling, we need to apply it before bias
    # since bias is not quantized
    should_add_bias_separately = (scale_result is not None) and (bias is not None)
    # should_add_bias_separately = bias is not None

    result = torch._scaled_mm(
        quantized_a.view(torch.float4_e2m1fn_x2),
        quantized_b.view(torch.float4_e2m1fn_x2),
        a_scale_blocked.view(torch.float8_e4m3fn),
        b_scale_blocked.view(torch.float8_e4m3fn),
        bias=None if should_add_bias_separately else bias,
        out_dtype=out_dtype,
        # scale_result=scale_result,  # Not supported yet
    )

    if scale_result is not None:
        result = result * scale_result.to(out_dtype)

    # Add bias after scaling if needed
    if should_add_bias_separately:
        result = result + bias

    return result


def nvfp4_linear_meta(
    a,
    quantized_b,
    b_per_tensor_scale,
    b_block_scales,
    out_dtype,
    bias: torch.Tensor | None = None,
    a_per_tensor_scale=None,
    a_block_scale=None,
):
    return thunder.TensorProxy(like=a, shape=(*a.shape[:-1], quantized_b.shape[0]))


def nvfp4_linear_impl(
    a,
    quantized_b,
    b_per_tensor_scale,
    b_block_scales,
    out_dtype,
    bias: torch.Tensor | None = None,
    a_per_tensor_scale=None,
    a_block_scales=None,
):
    if a_per_tensor_scale is not None and a_block_scales is not None:
        quantized_a = a
    else:
        quantized_a, a_block_scales, a_per_tensor_scale = quantize_fn(a)
    return _nvfp4_linear(
        quantized_a,
        quantized_b,
        a_per_tensor_scale,
        b_per_tensor_scale,
        a_block_scales,
        b_block_scales,
        out_dtype,
        bias,
    )


nvfp4_linear = nvfp4_executor.register_operator("nvfp4_linear", meta=nvfp4_linear_meta, fn=nvfp4_linear_impl)


def nvfp4_quantize_meta(a: TensorProxy) -> tuple[TensorProxy, TensorProxy, TensorProxy]:
    quantized_shape = list(a.shape)
    quantized_shape[-1] //= 2
    block_scale_shape = list(a.shape)
    block_scale_shape[-1] //= 16
    return (
        thunder.TensorProxy(like=a, shape=quantized_shape),
        thunder.TensorProxy(like=a, shape=block_scale_shape),
        thunder.TensorProxy(like=a, shape=()),
    )


def nvfp4_quantize_impl(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return quantize_fn(a)


def nvfp4_quantize_bind_postprocess(bsym: BoundSymbol) -> None:
    bsym.header = "Output tuple has (quantized, block scale, global scale)"


nvfp4_quantize = nvfp4_executor.register_operator(
    "nvfp4_quantize", meta=nvfp4_quantize_meta, fn=nvfp4_quantize_impl, bind_postprocess=nvfp4_quantize_bind_postprocess
)


class QuantizedLinearTransform(thunder.Transform):
    def __init__(self):
        self.quant_states = {}
        self.quantized_submodule_names = set()

    def transform_module(self, model: thunder.ThunderModule):
        self.thunder_module = model
        processed_names = set()

        def convert_linear_submodule(tm, name):
            self.quantized_submodule_names.add(name)
            weight_name = f"{name}.weight"
            w = tm.get_parameter(weight_name)

            qw, qs, per_tensor_scale = quantize_fn(w)

            tm._overrides_parameters[weight_name] = qw.to(w.device)
            tm._overrides_parameters[f"{weight_name}.per_tensor_scale"] = per_tensor_scale.to(w.device)
            tm._overrides_parameters[f"{weight_name}.block_scales"] = qs.to(w.device)
            self.quant_states[weight_name] = {
                "dtype": qw.dtype,
                "per_tensor_scale": per_tensor_scale,
                "per_tensor_scale.shape": tuple(per_tensor_scale.shape),
                "per_tensor_scale.dtype": per_tensor_scale.dtype,
                "block_scales": qs,
                "block_scales.shape": tuple(qs.shape),
                "block_scales.dtype": qs.dtype,
                "quantized_weight": qw,
                "shape": qw.shape,
                "out_dtype": w.dtype,
            }
            processed_names.add(weight_name)

        for n, submodule in model._model.named_modules():
            if isinstance(submodule, torch.nn.Linear):
                convert_linear_submodule(model, n)

    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        tm = self.thunder_module

        checks = get_checks(prologue_trace)

        prologue_proxy_map = {
            get_param_bsym.output.name: dict(
                shape=self.quant_states[model_weight_name]["shape"],
                dtype=thunder.dtypes.to_dtype(self.quant_states[model_weight_name]["dtype"]),
            )
            for model_weight_name, (check_bsym, get_param_bsym) in checks.items()
            if model_weight_name in self.quant_states
        }

        # here we switch the prologue_trace to a copy with new metadata
        prologue_trace = trace_with_replaced_proxy_metadata(prologue_trace, prologue_proxy_map)

        checks = get_checks(prologue_trace)
        proglogue_to_compute_outputs = prologue_trace.output[0]

        output_idxes = {o.name: i for i, o in enumerate(proglogue_to_compute_outputs)}

        quantized_proxies: dict[str, str] = {}  # proxy_name -> param_name
        additional_proxies: dict[str, thunder.Proxy] = {}  # param_name.(absmax|code) -> Proxy
        new_bsyms = []
        new_compute_inputs = []
        for n, qs in self.quant_states.items():
            # Update prologue to return `per_tensor_scale` and `block_scales`
            tm.get_parameter(n)
            n_per_tensor_scale = f"{n}.per_tensor_scale"
            n_block_scales = f"{n}.block_scales"
            tm.get_parameter(n_per_tensor_scale)
            tm.get_parameter(n_block_scales)
            check, get_param = checks[n]
            quantized_proxies[get_param.output.name] = n
            # check has args: tensor, shape, device, dtype, requires_grad
            proxy, _, device, _, requires_grad = check.args
            check.args = (proxy, tuple(qs["shape"]), device, qs["dtype"], False)

            output_idx = output_idxes.get(get_param.output.name)

            if output_idx is not None:
                with tracectx(prologue_trace):
                    # better way
                    proxy_per_tensor_scale = thunder.TensorProxy(
                        name=f"{get_param.output.name}_per_tensor_scale",
                        shape=qs["per_tensor_scale.shape"],
                        dtype=thunder.dtypes.to_dtype(qs["per_tensor_scale.dtype"]),
                        device=thunder.devices.to_device(device),
                        requires_grad=False,
                        tags={thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION},
                    )
                    proxy_block_scales = thunder.TensorProxy(
                        name=f"{get_param.output.name}_block_scales",
                        shape=qs["block_scales.shape"],
                        dtype=thunder.dtypes.to_dtype(qs["block_scales.dtype"]),
                        device=thunder.devices.to_device(device),
                        requires_grad=False,
                        tags={thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION},
                    )
                    # get_param.sym = unpack_buffer/parameter as needed
                    new_bsyms.append(
                        get_param.sym.bind(get_param.args[0], n_per_tensor_scale, output=proxy_per_tensor_scale)
                    )
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_block_scales, output=proxy_block_scales))
                    add_trace_output(prologue_trace, proxy_per_tensor_scale, subindex=0)
                    add_trace_output(prologue_trace, proxy_block_scales, subindex=0)
                    new_compute_inputs.append(proxy_per_tensor_scale)
                    new_compute_inputs.append(proxy_block_scales)
                    # add checks
                    new_bsyms.append(
                        prims.check_tensor_shape_and_metadata.bind(
                            proxy_per_tensor_scale,
                            qs["per_tensor_scale.shape"],
                            device,
                            qs["per_tensor_scale.dtype"],
                            False,
                            output=None,
                        )
                    )
                    new_bsyms.append(
                        prims.check_tensor_shape_and_metadata.bind(
                            proxy_block_scales,
                            qs["block_scales.shape"],
                            device,
                            qs["block_scales.dtype"],
                            False,
                            output=None,
                        )
                    )
                    # this is not good, because we will have several traces...
                    additional_proxies[n_per_tensor_scale] = proxy_per_tensor_scale
                    additional_proxies[n_block_scales] = proxy_block_scales

        prologue_trace.bound_symbols[-1:-1] = new_bsyms

        # Update computation to use `per_tensor_scale` and `block_scales`
        computation_proxy_map = {
            csym.name: dict(
                shape=psym.shape,
                dtype=psym.dtype,
            )
            for psym, csym in zip(prologue_trace.bound_symbols[-1].args[0][0], computation_trace.args)
            if psym.shape != csym.shape or psym.dtype != csym.dtype
        }

        new_computation_trace = trace_with_replaced_proxy_metadata(computation_trace, computation_proxy_map)
        bound_symbols = new_computation_trace.bound_symbols
        new_computation_trace.bound_symbols = []

        new_computation_trace.args = (*new_computation_trace.args, *new_compute_inputs)
        new_computation_trace.names.update(i.name for i in new_compute_inputs)
        new_computation_trace._siginfo.args = [(a.name, None) for a in new_computation_trace.args]

        with tracectx(new_computation_trace):
            new_bindings = [
                thunder.core.prims.unpack_trivial.bind(i, output=i, name=i.name) for i in new_compute_inputs
            ]

        for idx, bsym in enumerate(bound_symbols):
            if bsym.sym != prims.unpack_trivial:
                break
            new_computation_trace.bound_symbols.append(bsym.from_bsym())
        new_computation_trace.bound_symbols += new_bindings

        for bsym in bound_symbols[idx:]:
            if bsym.sym == thunder.torch.linear and bsym.args[1].name in quantized_proxies:
                assert len(bsym.args) == 3  # torch.linear(input, weight, bias)
                activation = bsym.args[0]
                quantized_weight = bsym.args[1]
                with tracectx(new_computation_trace):
                    quantized, block_scale, global_scale = nvfp4_quantize(activation)
                n = quantized_proxies[bsym.args[1].name]
                qs = self.quant_states[n]
                # signature of the new symbol:
                # nvfp4_linear_impl(a, quantized_b, b_per_tensor_scale, b_block_scales, out_dtype, bias: Optional[torch.Tensor] = None)
                new_args = (
                    quantized,
                    quantized_weight,
                    additional_proxies[f"{n}.per_tensor_scale"],
                    additional_proxies[f"{n}.block_scales"],
                    qs["out_dtype"],
                    bsym.args[2],
                    global_scale,
                    block_scale,
                )
                linear_bsym = bsym.from_bsym(
                    sym=nvfp4_linear,
                    subsymbols=[],
                    args=new_args,
                )

                new_computation_trace.bound_symbols.append(linear_bsym)
            else:
                new_computation_trace.bound_symbols.append(bsym.from_bsym())

        return prologue_trace, new_computation_trace, epilogue_trace


tfms = QuantizedLinearTransform()

linear = torch.nn.Sequential(torch.nn.Linear(64, 256, dtype=torch.bfloat16, bias=False, device="cuda"), torch.nn.ReLU())
compiled_linear = thunder.jit(linear, transforms=[tfms], executors=[nvfp4_executor], disable_atograd=True)

# Transformed Computation Trace:
# Constructed by Unwrap the actual return value
# import torch
# from thunder.executors.torchex import no_autocast
#
# @torch.no_grad()
# @no_autocast
# def computation(input, t_0_weight, t_0_weight_per_tensor_scale, t_0_weight_block_scales):
#   # input: "cuda:0 bf16[128, 64]"
#   # t_0_weight: "cuda:0 ui8[256, 32]"
#   # t_0_weight_per_tensor_scale: "cuda:0 f32[]"
#   # t_0_weight_block_scales: "cuda:0 f8_e4m3fn[1024]"
#   #   # Output tuple has (quantized, block scale, global scale)
#   (t0, t1, t2) = nvfp4_quantize(input)
#
#   # /usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py:134:             return F.linear(input, self.weight, self.bias)
#   t3 = nvfp4_linear(t0, t_0_weight, t_0_weight_per_tensor_scale, t_0_weight_block_scales, torch.bfloat16, None, t2, t1)  # t3: "cuda:0 bf16[128, 256]"
#   del t0, t_0_weight_per_tensor_scale, t_0_weight_block_scales, t2, t1
#   return (t3,)
# Getting the following error on RTX 6000 Ada as nvFP4 shouldn't be supported anyways
# RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasLtMatmulAlgoGetHeuristic( ltHandle, computeDesc.descriptor(), Adesc.descriptor(), Bdesc.descriptor(), Cdesc.descriptor(), Ddesc.descriptor(), preference.descriptor(), 1, &heuristicResult, &returnedResult)`
# But it works fine on B200
try:
    compiled_linear(torch.randn(128, 64, device="cuda", dtype=torch.bfloat16))
except Exception:
    raise
finally:
    print(thunder.last_traces(compiled_linear)[-1])
