from thunder.core.trace_interpreter import TraceSubstitutionProcessor

import thunder
from thunder.core.transform_common import Transform
from thunder.core import prims
import torch

from .utils import (
    get_checks,
    add_trace_output,
    trace_with_replaced_proxy_metadata,
)

te_inference_executor = None


def get_te_inference_executor():
    global transformer_engine
    global transformer_engine_torch
    global te_inference_executor
    global te_linear_fp8
    global te_groupedmm_fp8

    if te_inference_executor is None:
        import transformer_engine
        import transformer_engine_torch

        te_inference_executor = thunder.extend.OperatorExecutor("quant_te", version=0.1)

        def te_linear_fp8_meta(x, qweight, bias, absmax, scale):
            # assert isinstance(shape, Sequence) and len(shape) == 2
            assert len(qweight.shape) == 2
            assert x.shape[-1] == qweight.shape[1], f"{x.shape=}, rhs {qweight.shape=}"
            return thunder.TensorProxy(like=x, shape=(*x.shape[:-1], qweight[0]))

        def te_linear_fp8_impl(x, qweight, bias, absmax, scale):
            wq = transformer_engine.pytorch.Float8Quantizer(
                scale=scale,
                amax=absmax,
                fp8_dtype=transformer_engine_torch.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
            )

            w = wq.create_tensor_from_data(qweight, fake_dtype=x.dtype, requires_grad=False)

            minmax = x.aminmax()
            xmax = torch.maximum(minmax.min.abs(), minmax.max.abs()).to(torch.float32)
            xq = transformer_engine.pytorch.Float8Quantizer(
                scale=1.0 / xmax,  # this needs to 1 (or even somewhat smaller for accumulation?)
                amax=xmax,
                fp8_dtype=transformer_engine_torch.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
            )

            out, *_ = transformer_engine.pytorch.ops.BasicLinear._functional_forward(
                x,
                w,
                input_quantizer=xq,
                with_quantized_compute=False,
                weight_requires_grad=False,
                input_requires_grad=False,
            )

            if bias is not None:
                out = out + bias

            return out

        te_linear_fp8 = te_inference_executor.register_operator(
            "te_linear_fp8", meta=te_linear_fp8_meta, fn=te_linear_fp8_impl
        )

        def te_groupedmm_fp8_meta(a, b, offs=None, bias=None, dtype=None, *, b_absmax, b_scale):
            assert (len(a.shape) == 2 or len(a.shape) == 3) and (len(b.shape) == 2 or len(b.shape) == 3)
            assert a.shape[-1] == b.shape[-2], "contraction dims have to match"
            if len(a.shape) == 2:
                if len(b.shape) == 2:
                    out_shape = (offs.shape[0], a.shape[0], b.shape[1])
                else:
                    assert offs.size[0] == b.size[0], "matrix batch sizes have to match"
                    out_shape = (a.shape[0], b.shape[-1])
            else:  # a.shape == 3
                if len(b.shape) == 2:
                    assert offs.shape[0] == a.shape[0], "matrix batch sizes have to match"
                    out_shape = (a.shape[1], b.shape[-1])
                else:
                    assert a.shape[0] == b.shape[0], "batched dimensions have to match"
                    out_shape = (a.shape[0], a.shape[1], b.shape[-1])
            if dtype is None:
                dtype = a.dtype

            return thunder.TensorProxy(like=a, shape=out_shape, dtype=dtype)

        def te_groupedmm_fp8_impl(a, b, offs=None, bias=None, dtype=None, *, b_absmax, b_scale):
            bq = transformer_engine.pytorch.Float8Quantizer(
                scale=b_scale,
                amax=b_absmax,
                fp8_dtype=transformer_engine_torch.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
            )

            b = bq.create_tensor_from_data(b, fake_dtype=a.dtype, requires_grad=False)
            b = b.to(a.dtype)
            return torch._grouped_mm(a, b, offs=offs, bias=bias, out_dtype=dtype)

        te_groupedmm_fp8 = te_inference_executor.register_operator(
            "te_groupedmm_fp8", meta=te_groupedmm_fp8_meta, fn=te_groupedmm_fp8_impl
        )

    return te_inference_executor


class TEInference8BitTransform(Transform):
    def __init__(self):
        self.quant_states = {}
        self.quantized_submodule_names = set()
        get_te_inference_executor()

    def quantize_weight(self, w):
        minmax = w.aminmax()
        amax = torch.maximum(minmax.min.abs(), minmax.max.abs()).to(torch.float32)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = fp8_max / amax
        if w.device.type != "cuda":  # also if no TE present?
            # this is off by 1 sometimes, probably rounding
            qw = (w * scale).clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            quantizer = transformer_engine.pytorch.Float8Quantizer(
                scale=scale,
                amax=amax,
                fp8_dtype=transformer_engine_torch.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
            )
            qw = quantizer.quantize(w)._data
        return qw, amax, scale

    def transform_module(self, model: thunder.ThunderModule):
        self.thunder_module = model
        shared_names = model._get_shared_names()
        processed_names = set()

        def convert_linear_submodule(tm, name, *, is_grouped):
            self.quantized_submodule_names.add(name)
            weight_name = f"{name}.weight"
            processed_copies = shared_names[weight_name] & processed_names
            if processed_copies:
                copy_name = next(iter(processed_copies))
                self.quant_states[weight_name] = self.quant_states[copy_name]
                tm._overrides_parameters[weight_name] = tm._overrides_parameters[copy_name]
                tm._overrides_parameters[f"{weight_name}.absmax"] = tm._overrides_parameters[f"{copy_name}.absmax"]
                tm._overrides_parameters[f"{weight_name}.scale"] = tm._overrides_parameters[f"{copy_name}.scale"]
                processed_names.add(weight_name)
                return

            w = tm.get_parameter(weight_name)
            if is_grouped:
                w = w.transpose(-2, -1).contiguous()
            qw, absmax, scale = self.quantize_weight(w)
            tm._overrides_parameters[weight_name] = qw
            tm._overrides_parameters[f"{weight_name}.absmax"] = absmax
            tm._overrides_parameters[f"{weight_name}.scale"] = scale
            self.quant_states[weight_name] = {
                "needs_transpose": is_grouped,
                "dtype": w.dtype,
                "shape": w.shape,
                "qweight.dtype": qw.dtype,
                "qweight.shape": tuple(qw.shape),
                "absmax.dtype": absmax.dtype,
                "absmax.shape": tuple(absmax.shape),
                "scale.dtype": scale.dtype,
                "scale.shape": tuple(scale.shape),
                "device": getattr(w, "_thunder_device", w.device),
            }
            processed_names.add(weight_name)

        for n, submodule in model._model.named_modules():
            if isinstance(submodule, torch.nn.Linear) or submodule.__class__.__name__ == "GroupedLinear":
                convert_linear_submodule(model, n, is_grouped=(submodule.__class__.__name__ == "GroupedLinear"))

    def get_executor(self):
        return get_te_inference_executor()

    def transform_state_dict_for_submodule(self, model: thunder.ThunderModule, submodule_name: str, state_dict: dict):
        # note that state dict entries do not include the submodule name as prefix
        if submodule_name not in self.quantized_submodule_names:
            return state_dict
        weight_name_full = f"{submodule_name}.weight"
        qs_dict = self.quant_states[weight_name_full]
        w = state_dict["weight"]
        assert w.dtype == qs_dict["dtype"]
        assert w.shape == qs_dict["shape"]

        qw, absmax, scale = self.quantize_weight(w.to("cuda"))

        state_dict = state_dict.copy()
        state_dict["weight"] = qw
        state_dict["weight.absmax"] = absmax
        state_dict["weight.scale"] = scale

        return state_dict

    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        tm = self.thunder_module
        from thunder.core.trace import tracectx

        checks = get_checks(prologue_trace)

        prologue_proxy_map = {
            get_param_bsym.output.name: dict(
                shape=self.quant_states[model_weight_name]["qweight.shape"],
                dtype=thunder.dtypes.to_dtype(self.quant_states[model_weight_name]["qweight.dtype"]),
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
        additional_proxies: dict[str, thunder.Proxy] = {}  # param_name.(absmax|scale) -> Proxy

        new_bsyms = []
        new_compute_inputs = []
        for n, qs in self.quant_states.items():
            tm.get_parameter(n)
            n_absmax = f"{n}.absmax"
            n_scale = f"{n}.scale"
            tm.get_parameter(n_absmax)
            tm.get_parameter(n_scale)
            check, get_param = checks[n]
            quantized_proxies[get_param.output.name] = n
            # check has args: tensor, shape, device, dtype, requires_grad
            proxy, _, device, _, requires_grad = check.args
            check.args = (proxy, qs["qweight.shape"], device, qs["qweight.dtype"], False)

            output_idx = output_idxes.get(get_param.output.name)
            if output_idx is not None:
                with tracectx(prologue_trace):
                    # better way
                    proxy_absmax = thunder.TensorProxy(
                        name=f"{get_param.output.name}_absmax",
                        shape=qs["absmax.shape"],
                        dtype=thunder.dtypes.to_dtype(qs["absmax.dtype"]),
                        device=thunder.devices.to_device(device),
                        requires_grad=False,
                        tags={thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION},
                    )
                    proxy_scale = thunder.TensorProxy(
                        name=f"{get_param.output.name}_scale",
                        shape=qs["scale.shape"],
                        dtype=thunder.dtypes.to_dtype(qs["scale.dtype"]),
                        device=thunder.devices.to_device(device),
                        requires_grad=False,
                        tags={thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION},
                    )
                    # get_param.sym = unpack_buffer/parameter as needed
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_absmax, output=proxy_absmax))
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_scale, output=proxy_scale))
                    add_trace_output(prologue_trace, proxy_absmax, subindex=0)
                    add_trace_output(prologue_trace, proxy_scale, subindex=0)
                    new_compute_inputs.append(proxy_absmax)
                    new_compute_inputs.append(proxy_scale)
                    # add checks
                    new_bsyms.append(
                        prims.check_tensor_shape_and_metadata.bind(
                            proxy_absmax, qs["absmax.shape"], device, qs["absmax.dtype"], False, output=None
                        )
                    )
                    new_bsyms.append(
                        prims.check_tensor_shape_and_metadata.bind(
                            proxy_scale, qs["scale.shape"], device, qs["scale.dtype"], False, output=None
                        )
                    )
                    # this is not good, because we will have several traces...
                    additional_proxies[n_absmax] = proxy_absmax
                    additional_proxies[n_scale] = proxy_scale

        prologue_trace.bound_symbols[-1:-1] = new_bsyms

        computation_proxy_map = {
            csym.name: dict(
                shape=psym.shape,
                dtype=psym.dtype,
            )
            for psym, csym in zip(prologue_trace.bound_symbols[-1].args[0][0], computation_trace.args)
            if psym.shape != csym.shape or psym.dtype != csym.dtype
        }

        # Add new compute inputs to the trace args before processing
        computation_trace.args = (*computation_trace.args, *new_compute_inputs)
        computation_trace.names.update(i.name for i in new_compute_inputs)
        computation_trace._siginfo.args = [(a.name, None) for a in computation_trace.args]

        # Add unpack_trivial bindings for new inputs in the correct position
        with tracectx(computation_trace):
            new_bindings = [
                thunder.core.prims.unpack_trivial.bind(i, output=i, name=i.name) for i in new_compute_inputs
            ]

        # Insert the new bindings after the existing unpack_trivial bindings to maintain arg order
        # Find the last unpack_trivial binding and insert after it
        insert_idx = len(computation_trace.bound_symbols)
        for i, bsym in enumerate(computation_trace.bound_symbols):
            if bsym.sym.id == prims.PrimIDs.UNPACK_TRIVIAL:
                insert_idx = i + 1

        computation_trace.bound_symbols[insert_idx:insert_idx] = new_bindings

        # Now update metadata for the complete trace
        new_computation_trace = trace_with_replaced_proxy_metadata(computation_trace, computation_proxy_map)

        class QuantizationProcessor(TraceSubstitutionProcessor):
            def __init__(self, trace, quantized_proxies, additional_proxies, quant_states, new_compute_inputs):
                super().__init__(trace)
                self.quantized_proxies = quantized_proxies
                self.additional_proxies = additional_proxies
                self.quant_states = quant_states
                self.new_compute_inputs = new_compute_inputs
                self.orig_producers = thunder.core.utils.producers(trace)

            def process_bsym(self, bsym):
                def handle_grouped_mm(bsmym):
                    if bsym.args[1].name in self.quantized_proxies:
                        return bsym.args[1]
                    pbsym = self.orig_producers[bsym.args[1]]
                    if pbsym.sym.name == "transpose" and pbsym.args[0].name in self.quantized_proxies:
                        return pbsym.args[0]
                    return None

                if bsym.sym == thunder.torch.linear and bsym.args[1].name in self.quantized_proxies:
                    assert len(bsym.args) == 3  # torch.linear(input, weight, bias)
                    n = self.quantized_proxies[bsym.args[1].name]
                    # signature of the new symbol:
                    # te_linear_fp8(x, qweight, bias, absmax, scale)
                    new_args = (
                        *bsym.args[:3],
                        self.additional_proxies[f"{n}.absmax"],
                        self.additional_proxies[f"{n}.scale"],
                    )
                    mm_bsym = bsym.from_bsym(
                        sym=te_linear_fp8,
                        subsymbols=[],
                        args=new_args,
                    )
                    self.add_processed_bsyms([mm_bsym])
                    self.set_result(bsym.output)
                elif bsym.sym == thunder.torch._grouped_mm and (quantized_proxy := handle_grouped_mm(bsym)) is not None:
                    # assert len(bsym.args) == 3  # torch.linear(input, weight, bias)
                    n = self.quantized_proxies[quantized_proxy.name]
                    # signature of the new symbol:
                    # te_linear_fp8(x, qweight, bias, absmax, scale)
                    new_args = (bsym.args[0], quantized_proxy, *bsym.args[2:])
                    new_kwargs = {
                        **bsym.kwargs,
                        "b_absmax": self.additional_proxies[f"{n}.absmax"],
                        "b_scale": self.additional_proxies[f"{n}.scale"],
                    }
                    mm_bsym = bsym.from_bsym(
                        sym=te_groupedmm_fp8,
                        subsymbols=[],
                        args=new_args,
                        kwargs=new_kwargs,
                    )
                    self.add_processed_bsyms([mm_bsym])
                    self.set_result(bsym.output)
                elif bsym.sym == prims.python_return:
                    assert len(bsym.args) == 1 and isinstance(bsym.args[0], dict)
                    new_return_dict = bsym.args[0].copy()
                    new_return_dict["flat_args"] = list(self.new_trace.args)  # we know that the args are flat
                    self.add_processed_bsyms([bsym.from_bsym(args=(new_return_dict,))])
                    self.set_result(bsym.output)
                else:
                    # Keep the original symbol
                    self.add_processed_bsyms([bsym.from_bsym()])
                    self.set_result(bsym.output)

        # Process the trace using the QuantizationProcessor
        processor = QuantizationProcessor(
            new_computation_trace, quantized_proxies, additional_proxies, self.quant_states, new_compute_inputs
        )

        # Now process the trace
        new_computation_trace, _ = processor()
        new_computation_trace.set_provenance(thunder.core.trace.TraceProvenance("te fp8 quant pass"))
        return prologue_trace, new_computation_trace, epilogue_trace
