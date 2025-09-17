from __future__ import annotations
from typing import TYPE_CHECKING
import argparse
import warnings

import torch
import torch.nn as nn
import torchao.prototype.mx_formats.nvfp4_tensor as nvfp4_tensor
from torchao.prototype.mx_formats.inference_workflow import NVFP4InferenceConfig
from torchao.quantization import quantize_

import thunder
from thunder.core import prims
from thunder.core.trace import tracectx
from thunder.transforms.utils import get_checks, trace_with_replaced_proxy_metadata, add_trace_output

if TYPE_CHECKING:
    from collections.abc import Callable
    from thunder.core.proxies import TensorProxy
    from thunder.core.symbol import BoundSymbol


nvfp4_executor = thunder.extend.OperatorExecutor("nvfp4_executor", version=0.1)

BLOCK_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def compute_per_tensor_scale(t: torch.Tensor) -> torch.Tensor:
    return ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / t.float().abs().amax()).to(torch.float32)


def _view_input_as_2d(x):
    shape = x.shape
    return x.view((-1, shape[-1]))


def quantize_fn(
    t: torch.Tensor, no_per_tensor_scale: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    with torch.no_grad():
        if no_per_tensor_scale:
            per_tensor_scale = None
        else:
            per_tensor_scale = compute_per_tensor_scale(t)
        qs, qw = nvfp4_tensor.nvfp4_quantize(t, per_tensor_scale=per_tensor_scale)

        if t.ndim == 2:
            # Swizzle the scales
            M, K = t.shape[0], t.shape[1]
        else:
            assert t.ndim == 3
            M, K = _view_input_as_2d(t).shape

        scale_shape = (M, K // BLOCK_SIZE)
        qs = nvfp4_tensor.to_blocked(qs.view(scale_shape)).flatten()

    return qw, qs, per_tensor_scale


# https://github.com/pytorch/ao/blob/4dffb40280ea7b0e1732c580d08df58d0134c543/torchao/prototype/mx_formats/nvfp4_tensor.py#L567-L568
def _nvfp4_linear(
    quantized_a: torch.Tensor,
    quantized_b: torch.Tensor,
    a_per_tensor_scale: torch.Tensor | None,
    b_per_tensor_scale: torch.Tensor | None,
    a_block_scales: torch.Tensor,
    b_block_scales: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    quantized_b = quantized_b.t()

    # assert quantized_a.is_contiguous()
    # assert quantized_b.t().is_contiguous()
    # assert a._block_size == 16, f"NVFP4 requires block_size=16, got {a._block_size}"
    # assert b._block_size == 16, f"NVFP4 requires block_size=16, got {b._block_size}"
    # assert bias is None

    # M, K = quantized_a.shape[0], quantized_a.shape[1]
    # N = quantized_b.shape[1]

    a_scale_blocked = a_block_scales  # Already swizzled
    b_scale_blocked = b_block_scales  # Already swizzled

    # Merge double quant scales into 1 scale for Scale_In^D
    scale_result: torch.Tensor | None
    if a_per_tensor_scale is not None and b_per_tensor_scale is not None:
        scale_result = a_per_tensor_scale * b_per_tensor_scale
    else:
        scale_result = None

    # THIS IS A WORKAROUND:
    # RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling
    # When we have per-tensor scaling, we need to apply it before bias
    # since bias is not quantized
    should_add_bias_separately = (scale_result is not None) and (bias is not None)
    # should_add_bias_separately = bias is not None

    inp_reshaped = False
    if quantized_a.ndim == 3:
        B, _, _ = quantized_a.shape
        quantized_a = _view_input_as_2d(quantized_a)
        inp_reshaped = True

    result = torch._scaled_mm(
        quantized_a.view(torch.float4_e2m1fn_x2),
        quantized_b.view(torch.float4_e2m1fn_x2),
        a_scale_blocked.view(torch.float8_e4m3fn),
        b_scale_blocked.view(torch.float8_e4m3fn),
        bias=None if should_add_bias_separately else bias,
        out_dtype=out_dtype,
        # scale_result=scale_result,  # Not supported yet
    )
    if inp_reshaped:
        M, W = result.shape
        result = result.view(B, M // B, W)

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
    a: torch.Tensor,
    quantized_b: torch.Tensor,
    b_per_tensor_scale: torch.Tensor,
    b_block_scales: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    a_per_tensor_scale: torch.Tensor | None = None,
    a_block_scales: torch.Tensor | None = None,
):
    if a_block_scales is not None:
        quantized_a = a
    else:
        quantized_a, a_block_scales, _ = quantize_fn(a, no_per_tensor_scale=True)
    return _nvfp4_linear(
        quantized_a,
        quantized_b,
        None,
        b_per_tensor_scale,
        a_block_scales,
        b_block_scales,
        out_dtype,
        bias,
    )


nvfp4_linear = nvfp4_executor.register_operator("nvfp4_linear", meta=nvfp4_linear_meta, fn=nvfp4_linear_impl)


def nvfp4_quantize_meta(a: TensorProxy) -> tuple[TensorProxy, TensorProxy]:
    quantized_shape = list(a.shape)
    quantized_shape[-1] //= 2
    block_scale_shape = list(a.shape)
    block_scale_shape[-1] //= 16
    return (
        thunder.TensorProxy(like=a, shape=quantized_shape),
        thunder.TensorProxy(like=a, shape=block_scale_shape),
    )


def nvfp4_quantize_impl(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fn(a, no_per_tensor_scale=True)[:-1]


def nvfp4_quantize_bind_postprocess(bsym: BoundSymbol) -> None:
    bsym.header = "Output tuple has (quantized, block scale)"


nvfp4_quantize = nvfp4_executor.register_operator(
    "nvfp4_quantize", meta=nvfp4_quantize_meta, fn=nvfp4_quantize_impl, bind_postprocess=nvfp4_quantize_bind_postprocess
)


def _default_filter(name: str, layer: nn.Module) -> bool:
    return isinstance(layer, nn.Linear)


class QuantizedLinearTransform(thunder.Transform):
    """Transform model into NVFP4 for inference.

    Args:
        filter_fn: Takes name and :class:`~torch.nn.Module` and returns ``True`` if the layer is to quantize.
            Default is to quantize all linear layers in the model.
        separate_quantization: If ``True``, the transformed trace have a :class:`thunder.core.symbol.BoundSymbol`
            for nvfp4 quantization and one for nvfp4 linear.
        use_per_tensor_scale: If ``True``, linear weights are quantized with their global scale.
            Otherwise, not (similar to TorchAO :func:`torchao.quantization.quantize_` with
            :class:`torcha.prototype.mx_formats.inference_workflow.NVFP4Inferenceconfig`.)
    """

    def __init__(
        self,
        *,
        filter_fn: Callable[[str, nn.Module], bool] = _default_filter,
        separate_quantization: bool = False,
        use_per_tensor_scale: bool = False,
    ):
        self.filter_fn = filter_fn
        self.separate_quantization = separate_quantization
        self.use_per_tensor_scale = use_per_tensor_scale
        self.quant_states = {}
        self.quantized_submodule_names = set()

    def transform_module(self, model: thunder.ThunderModule):
        self.thunder_module = model
        processed_names = set()

        def convert_linear_submodule(tm, name):
            self.quantized_submodule_names.add(name)
            weight_name = f"{name}.weight"
            w = tm.get_parameter(weight_name)

            qw, qs, per_tensor_scale = quantize_fn(w, not self.use_per_tensor_scale)

            tm._overrides_parameters[weight_name] = qw.to(w.device)
            if self.use_per_tensor_scale:
                tm._overrides_parameters[f"{weight_name}.per_tensor_scale"] = per_tensor_scale.to(w.device)
            else:
                tm._overrides_parameters[f"{weight_name}.per_tensor_scale"] = None
            tm._overrides_parameters[f"{weight_name}.block_scales"] = qs.to(w.device)
            self.quant_states[weight_name] = {
                "dtype": qw.dtype,
                "per_tensor_scale": per_tensor_scale,
                "per_tensor_scale.shape": tuple(per_tensor_scale.shape) if self.use_per_tensor_scale else (),
                # NOTE: per_tensor_scale/per_tensor_scale are always torch.float32, so we might want to remove this.
                "per_tensor_scale.dtype": torch.float32,
                "block_scales": qs,
                "block_scales.shape": tuple(qs.shape),
                "block_scales.dtype": qs.dtype,
                "quantized_weight": qw,
                "shape": qw.shape,
                "out_dtype": w.dtype,
            }
            processed_names.add(weight_name)

        for n, submodule in model._model.named_modules():
            # if isinstance(submodule, nn.Linear):
            if self.filter_fn(n, submodule):
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
            n_block_scales = f"{n}.block_scales"
            tm.get_parameter(n_block_scales)
            if self.use_per_tensor_scale:
                n_per_tensor_scale = f"{n}.per_tensor_scale"
                tm.get_parameter(n_per_tensor_scale)
            check, get_param = checks[n]
            quantized_proxies[get_param.output.name] = n
            # check has args: tensor, shape, device, dtype, requires_grad
            proxy, _, device, _, requires_grad = check.args
            check.args = (proxy, tuple(qs["shape"]), device, qs["dtype"], False)

            output_idx = output_idxes.get(get_param.output.name)

            if output_idx is not None:
                with tracectx(prologue_trace):
                    # better way
                    if self.use_per_tensor_scale:
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
                    if self.use_per_tensor_scale:
                        new_bsyms.append(
                            get_param.sym.bind(get_param.args[0], n_per_tensor_scale, output=proxy_per_tensor_scale)
                        )
                        add_trace_output(prologue_trace, proxy_per_tensor_scale, subindex=0)
                        new_compute_inputs.append(proxy_per_tensor_scale)
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_block_scales, output=proxy_block_scales))
                    add_trace_output(prologue_trace, proxy_block_scales, subindex=0)
                    new_compute_inputs.append(proxy_block_scales)
                    # add checks
                    if self.use_per_tensor_scale:
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
                    if self.use_per_tensor_scale:
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
                if self.separate_quantization:
                    with tracectx(new_computation_trace):
                        quantized, block_scale = nvfp4_quantize(activation)
                n = quantized_proxies[bsym.args[1].name]
                qs = self.quant_states[n]
                # signature of the new symbol:
                # nvfp4_linear_impl(a, quantized_b, b_per_tensor_scale, b_block_scales, out_dtype, bias: Optional[torch.Tensor] = None)
                new_args = (
                    quantized if self.separate_quantization else activation,
                    quantized_weight,
                    additional_proxies[f"{n}.per_tensor_scale"] if self.use_per_tensor_scale else None,
                    additional_proxies[f"{n}.block_scales"],
                    qs["out_dtype"],
                    bsym.args[2],
                    None,
                    block_scale if self.separate_quantization else None,
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


class Module(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 256, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return torch.relu(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate QuantizedLinearTransform that nvfp4 quantized nn.Linears in a model. By default this script does not compute per-tensor at all to reproduce TorchAO behavior. This script compares the output with TorchAO by using `quantize_(model, NVFP4InferenceConfig())`",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=250916, help="seed value")
    parser.add_argument("--skip-trace", action="store_true", help="skip printing execution trace")
    parser.add_argument("--skip-test", action="store_true", help="skip numeric test against torchao")
    parser.add_argument("--separate-quant", action="store_true", help="do quantization separately")
    # TODO: check if this option is correctly applied to activation quantization in the future
    parser.add_argument("--per-tensor-scale", action="store_true", help="use per-tensor scale.")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model = Module().to(device="cuda", dtype=torch.bfloat16)
    ref_model: nn.Module | None = None
    if not args.skip_test:
        ref_model = Module().to(device="cuda", dtype=torch.bfloat16)
        ref_model.load_state_dict(model.state_dict())
        quantize_(ref_model, NVFP4InferenceConfig(use_triton_kernel=False))

    tfms = QuantizedLinearTransform(
        separate_quantization=args.separate_quant,
        use_per_tensor_scale=args.per_tensor_scale,
    )
    compiled_linear: thunder.ThunderModule = thunder.jit(
        model,
        transforms=[tfms],
        executors=[nvfp4_executor],
        disable_atograd=True,
        debug_options=thunder.DebugOptions(check_traces=True),
    )

    x = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    # Getting the following error on RTX 6000 Ada as nvFP4 shouldn't be supported anyways
    # RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasLtMatmulAlgoGetHeuristic( ltHandle, computeDesc.descriptor(), Adesc.descriptor(), Bdesc.descriptor(), Cdesc.descriptor(), Ddesc.descriptor(), preference.descriptor(), 1, &heuristicResult, &returnedResult)`
    # But it works fine on B200
    out = compiled_linear(x)
    if not args.skip_trace:
        # The trace should look like what follows:
        # Constructed by Unwrap the actual return value
        # def computation(x, t_linear_weight, t_linear_weight_block_scales):
        #   # x: "cuda:0 bf16[128, 64]"
        #   # t_linear_weight: "cuda:0 ui8[256, 32]"
        #   # t_linear_weight_block_scales: "cuda:0 f8_e4m3fn[1024]"

        #   # /usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py:134:             return F.linear(input, self.weight, self.bias)
        #   out = nvfp4_linear(x, t_linear_weight, None, t_linear_weight_block_scales, torch.bfloat16, None, None, None)  # out: "cuda:0 bf16[128, 256]"
        #   del t_linear_weight_block_scales

        #   # /opt/pytorch/lightning-thunder/./quantization_transform.py:417:             return torch.relu(out)
        #   t11 = torch.nn.functional.relu(out, False)  # t11: "cuda:0 bf16[128, 256]"
        #     # t11 = ltorch.relu(out, False)  # t11: "cuda:0 bf16[128, 256]"
        #       # t10 = ltorch.gt(out, 0)  # t10: "cuda:0 b8[128, 256]"
        #         # t10 = prims.gt(out, 0.0)  # t10: "cuda:0 b8[128, 256]"
        #       # t11 = ltorch.where(t10, out, 0)  # t11: "cuda:0 bf16[128, 256]"
        #         # t11 = prims.where(t10, out, 0.0)  # t11: "cuda:0 bf16[128, 256]"
        #   del out
        #   return (t11,)
        print(thunder.last_traces(compiled_linear)[-1])
    if not args.skip_test:
        if args.per_tensor_scale:
            warnings.warn("The results would be different with high probability")
        for name, ref_param in ref_model.named_parameters():
            if isinstance(ref_param, nvfp4_tensor.NVFP4Tensor):
                param = compiled_linear.get_parameter(name)
                torch.testing.assert_close(param, ref_param._data)
                torch.testing.assert_close(compiled_linear.get_parameter(f"{name}.block_scales"), ref_param._scale_e4m3)
                torch.testing.assert_close(
                    compiled_linear.get_parameter(f"{name}.per_tensor_scale"), ref_param._per_tensor_scale
                )
        x_ref = x.clone().detach()
        torch.testing.assert_close(x, x_ref)
        ref = ref_model(x_ref)
        torch.testing.assert_close(out, ref)
