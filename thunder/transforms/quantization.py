from collections.abc import Sequence

import thunder
from thunder.core.transform_common import Transform
from thunder.core import utils
from thunder.core import prims
import torch

from .utils import (
    get_orig_and_thunder_module_proxies_from_prologue,
    get_checks,
    add_trace_output,
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

    def transform_module(self, model: thunder.ThunderModule):
        self.thunder_module = model

        def convert_linear_submodule(tm, name):
            self.quantized_submodule_names.add(name)
            weight_name = f"{name}.weight"
            w = tm.get_parameter(weight_name)
            # TODO: double quant support

            if w.device.type == "meta":
                w_work = torch.zeros_like(w, device="cuda")
            elif w.device.type != "cuda":
                with torch.no_grad():
                    w_work = w.to("cuda")
            else:
                w_work = w

            qw, qs = bitsandbytes.functional.quantize_4bit(w_work, quant_type="nf4")
            tm._overrides_parameters[weight_name] = qw.to(w.device)
            tm._overrides_parameters[f"{weight_name}.absmax"] = qs.absmax.to(w.device)
            tm._overrides_parameters[f"{weight_name}.code"] = qs.code.to(w.device)
            self.quant_states[weight_name] = {"dtype": qs.dtype, "shape": qs.shape, "blocksize": qs.blocksize}

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

        if w.device.type == "meta":
            w_work = torch.zeros_like(w, device="cuda")
        elif w.device.type != "cuda":
            with torch.no_grad():
                w_work = w.to("cuda")
        else:
            w_work = w

        qw, qs = bitsandbytes.functional.quantize_4bit(w_work, blocksize=qs_dict["blocksize"], quant_type="nf4")

        # double quant support...
        state_dict = state_dict.copy()
        state_dict["weight"] = qw.to(w.device)
        state_dict["weight.absmax"] = qs.absmax.to(w.device)
        state_dict["weight.code"] = qs.code.to(w.device)

        return state_dict

    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, *, backward=False):
        if backward:
            return super().transform_traces_pre_prologue(prologue_trace, computation_trace, epilogue_trace)
        tm = self.thunder_module
        from thunder.core.trace import tracectx

        checks = get_checks(prologue_trace)

        compute_producers, compute_consumers = utils.producers_and_consumers(computation_trace)

        proglogue_to_compute_outputs = prologue_trace.output[0]

        output_idxes = {id(o): i for i, o in enumerate(proglogue_to_compute_outputs)}

        computation_trace.push_scope([])
        quantized_proxies: dict[int, str] = {}  # id -> name

        new_bsyms = []
        new_compute_inputs = []
        for n, qs in self.quant_states.items():
            param = tm.get_parameter(n)
            n_absmax = f"{n}.absmax"
            n_code = f"{n}.code"
            param_absmax = tm.get_parameter(n_absmax)
            param_code = tm.get_parameter(n_code)
            check, get_param = checks[n]
            quantized_proxies[id(get_param.output)] = n
            # check has args: tensor, shape, device, dtype, requires_grad
            proxy, _, _, _, requires_grad = check.args
            thunder_device = thunder.devices.to_device(param.device)
            thunder_device_str = thunder_device.device_str()
            check.args = (proxy, (*param.shape,), thunder_device_str, param.dtype, False)

            output_idx = output_idxes.get(id(get_param.output))
            if output_idx is not None:
                with tracectx(prologue_trace):
                    # better way
                    proxy_absmax = thunder.TensorProxy(
                        name=f"{get_param.output.name}_absmax",
                        shape=param_absmax.shape,
                        dtype=thunder.dtypes.to_dtype(param_absmax.dtype),
                        device=thunder.devices.to_device(param_absmax.device),
                        requires_grad=False,
                    )
                    proxy_code = thunder.TensorProxy(
                        name=f"{get_param.output.name}_code",
                        shape=param_code.shape,
                        dtype=thunder.dtypes.to_dtype(param_code.dtype),
                        device=thunder.devices.to_device(param_code.device),
                        requires_grad=False,
                    )
                    # get_param.sym = unpack_buffer/parameter as needed
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_absmax, output=proxy_absmax))
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_code, output=proxy_code))
                    add_trace_output(prologue_trace, proxy_absmax, subindex=0)
                    add_trace_output(prologue_trace, proxy_code, subindex=0)
                    new_compute_inputs.append(proxy_absmax)
                    new_compute_inputs.append(proxy_code)
                    qs["proxy_absmax"] = proxy_absmax
                    qs["proxy_code"] = proxy_code
                compute_input = computation_trace.args[output_idx]

        prologue_trace.bound_symbols[-1:-1] = new_bsyms

        with tracectx(computation_trace):
            new_bindings = [thunder.core.prims.unpack_trivial.bind(i, output=i) for i in new_compute_inputs]

        new_computation_trace = thunder.core.trace.from_trace(computation_trace)
        new_computation_trace.args = (*new_computation_trace.args, *new_compute_inputs)
        new_computation_trace._siginfo.args = [(a.name, None) for a in new_computation_trace.args]
        for idx, bsym in enumerate(computation_trace.bound_symbols):
            if bsym.sym != prims.unpack_trivial:
                break
            new_computation_trace.bound_symbols.append(bsym.from_bsym())
        new_computation_trace.bound_symbols += new_bindings
        proxies_to_replace = {}
        for bsym in computation_trace.bound_symbols[idx:]:
            if bsym.sym == thunder.torch.linear and id(bsym.args[1]) in quantized_proxies:
                assert len(bsym.args) == 3  # torch.linear(input, weight, bias)
                n = quantized_proxies[id(bsym.args[1])]
                qs = self.quant_states[n]
                # signature of the new symbol:
                # bnb_matmul_nf4(x, qweight, bias, absmax, quant_map, blocksize, dtype, shape)
                new_args = (
                    *bsym.args[:3],
                    qs["proxy_absmax"],
                    qs["proxy_code"],
                    qs["blocksize"],
                    qs["dtype"],
                    qs["shape"],
                )
                mm_bsym = bsym.from_bsym(
                    sym=bnb_matmul_nf4,
                    subsymbols=[],
                    args=new_args,
                )

                new_computation_trace.bound_symbols.append(mm_bsym)
                # we need the postprocess to set the internal state (call_ctx) because we do not bind / execute the new symbol to
                # preserve the "meta"-info like source location, header, etc.
                # TODO: switch to a better solution when it is there
                bnb_matmul_nf4._bind_postprocess(mm_bsym)
            else:
                new_computation_trace.bound_symbols.append(bsym.from_bsym())

        new_computation_trace.set_provenance(thunder.core.trace.TraceProvenance("quant pass"))
        return prologue_trace, new_computation_trace, epilogue_trace
