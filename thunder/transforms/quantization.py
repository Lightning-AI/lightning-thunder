from collections.abc import Sequence
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


bitsandbytes_executor = None


def get_bitsandbytes_executor():
    global bitsandbytes
    global bitsandbytes_executor
    global bnb_matmul_nf4

    if bitsandbytes_executor is None:
        import bitsandbytes

        bitsandbytes_executor = thunder.extend.OperatorExecutor("quant_bnb", version=0.1)

        def bnb_matmul_nf4_meta(x, qweight, bias, absmax, quant_map, blocksize, dtype, shape):
            assert isinstance(shape, Sequence) and len(shape) == 2
            assert x.shape[-1] == shape[1], f"{x.shape=}, rhs {shape=}"
            return thunder.TensorProxy(like=x, shape=(*x.shape[:-1], shape[0]))

        def bnb_matmul_nf4_impl(x, qweight, bias, absmax, quant_map, blocksize, dtype, shape):
            qs = bitsandbytes.functional.QuantState(
                absmax, shape=shape, blocksize=blocksize, code=quant_map, quant_type="nf4", dtype=dtype
            )

            return bitsandbytes.matmul_4bit(x, qweight.t(), bias=bias, quant_state=qs)

        bnb_matmul_nf4 = bitsandbytes_executor.register_operator(
            "bnb_matmul_nf4", meta=bnb_matmul_nf4_meta, fn=bnb_matmul_nf4_impl
        )
    return bitsandbytes_executor


class BitsAndBytesLinearQuant4bit(Transform):
    def __init__(self):
        self.quant_states = {}
        self.quantized_submodule_names = set()
        get_bitsandbytes_executor()

    def quantize_weight(self, w):
        # todo: revisit staying on CPU when bnb supports it
        if w.device.type == "meta":
            w_work = torch.zeros_like(w, device="cuda")
        elif w.device.type != "cuda":
            with torch.no_grad():
                w_work = w.to("cuda")
        else:
            w_work = w

        return bitsandbytes.functional.quantize_4bit(w_work, quant_type="nf4")

    def transform_module(self, model: thunder.ThunderModule):
        self.thunder_module = model
        shared_names = model._get_shared_names()
        processed_names = set()

        def convert_linear_submodule(tm, name):
            self.quantized_submodule_names.add(name)
            weight_name = f"{name}.weight"
            processed_copies = shared_names[weight_name] & processed_names
            if processed_copies:
                copy_name = next(iter(processed_copies))
                self.quant_states[weight_name] = self.quant_states[copy_name]
                tm._overrides_parameters[weight_name] = tm._overrides_parameters[copy_name]
                tm._overrides_parameters[f"{weight_name}.absmax"] = tm._overrides_parameters[f"{copy_name}.absmax"]
                tm._overrides_parameters[f"{weight_name}.code"] = tm._overrides_parameters[f"{copy_name}.code"]
                processed_names.add(weight_name)
                return

            w = tm.get_parameter(weight_name)
            # TODO: double quant support

            qw, qs = self.quantize_weight(w)
            tm._overrides_parameters[weight_name] = qw.to(w.device)
            tm._overrides_parameters[f"{weight_name}.absmax"] = qs.absmax.to(w.device)
            tm._overrides_parameters[f"{weight_name}.code"] = qs.code.to(w.device)
            self.quant_states[weight_name] = {
                "dtype": qs.dtype,
                "shape": tuple(qs.shape),
                "blocksize": qs.blocksize,
                "qweight.dtype": qw.dtype,
                "qweight.shape": tuple(qw.shape),
                "absmax.dtype": qs.absmax.dtype,
                "absmax.shape": tuple(qs.absmax.shape),
                "code.dtype": qs.code.dtype,
                "code.shape": tuple(qs.code.shape),
                "device": getattr(w, "_thunder_device", w.device),
            }
            processed_names.add(weight_name)

        for n, submodule in model._model.named_modules():
            if isinstance(submodule, torch.nn.Linear):
                convert_linear_submodule(model, n)

    def transform_state_dict_for_submodule(self, model: thunder.ThunderModule, submodule_name: str, state_dict: dict):
        # note that state dict entries do not include the submodule name as prefix
        if submodule_name not in self.quantized_submodule_names:
            return state_dict
        weight_name_full = f"{submodule_name}.weight"
        qs_dict = self.quant_states[weight_name_full]
        w = state_dict["weight"]
        assert w.dtype == qs_dict["dtype"]
        assert w.shape == qs_dict["shape"]

        qw, qs = self.quantize_weight(w)

        # double quant support...
        state_dict = state_dict.copy()
        state_dict["weight"] = qw.to(w.device)
        state_dict["weight.absmax"] = qs.absmax.to(w.device)
        state_dict["weight.code"] = qs.code.to(w.device)

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
        additional_proxies: dict[str, thunder.Proxy] = {}  # param_name.(absmax|code) -> Proxy

        new_bsyms = []
        new_compute_inputs = []
        for n, qs in self.quant_states.items():
            tm.get_parameter(n)
            n_absmax = f"{n}.absmax"
            n_code = f"{n}.code"
            tm.get_parameter(n_absmax)
            tm.get_parameter(n_code)
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
                    proxy_code = thunder.TensorProxy(
                        name=f"{get_param.output.name}_code",
                        shape=qs["code.shape"],
                        dtype=thunder.dtypes.to_dtype(qs["code.dtype"]),
                        device=thunder.devices.to_device(device),
                        requires_grad=False,
                        tags={thunder.core.proxies.ProxyTag.STATIC_MEMORY_LOCATION},
                    )
                    # get_param.sym = unpack_buffer/parameter as needed
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_absmax, output=proxy_absmax))
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_code, output=proxy_code))
                    add_trace_output(prologue_trace, proxy_absmax, subindex=0)
                    add_trace_output(prologue_trace, proxy_code, subindex=0)
                    new_compute_inputs.append(proxy_absmax)
                    new_compute_inputs.append(proxy_code)
                    # add checks
                    new_bsyms.append(
                        prims.check_tensor_shape_and_metadata.bind(
                            proxy_absmax, qs["absmax.shape"], device, qs["absmax.dtype"], False, output=None
                        )
                    )
                    new_bsyms.append(
                        prims.check_tensor_shape_and_metadata.bind(
                            proxy_code, qs["code.shape"], device, qs["code.dtype"], False, output=None
                        )
                    )
                    # this is not good, because we will have several traces...
                    additional_proxies[n_absmax] = proxy_absmax
                    additional_proxies[n_code] = proxy_code

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

            def process_bsym(self, bsym):
                if bsym.sym == thunder.torch.linear and bsym.args[1].name in self.quantized_proxies:
                    assert len(bsym.args) == 3  # torch.linear(input, weight, bias)
                    n = self.quantized_proxies[bsym.args[1].name]
                    qs = self.quant_states[n]
                    # signature of the new symbol:
                    # bnb_matmul_nf4(x, qweight, bias, absmax, quant_map, blocksize, dtype, shape)
                    new_args = (
                        *bsym.args[:3],
                        self.additional_proxies[f"{n}.absmax"],
                        self.additional_proxies[f"{n}.code"],
                        qs["blocksize"],
                        qs["dtype"],
                        qs["shape"],
                    )
                    mm_bsym = bsym.from_bsym(
                        sym=bnb_matmul_nf4,
                        subsymbols=[],
                        args=new_args,
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
        new_computation_trace.set_provenance(thunder.core.trace.TraceProvenance("quant pass"))
        return prologue_trace, new_computation_trace, epilogue_trace
