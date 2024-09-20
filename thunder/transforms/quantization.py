from collections.abc import Sequence

import thunder
from thunder.core.proxies import TensorProxy, variableify
from thunder.core.transform_common import Transform
from thunder.core.trace import TraceCtx
from thunder.core.pytree import tree_map
from thunder.core import utils
from thunder.core import prims
import torch
import math

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


def trace_with_replaced_proxy_metadata(trace: TraceCtx, proxy_replacement_metadata) -> TraceCtx:
    t = TraceCtx(trace.fn, prologue=trace.prologue)

    proxymap: dict[str, thunder.Proxy] = {}

    def map_proxy(p):
        if isinstance(p, thunder.Proxy):
            return proxymap[p.name]
        return p

    def create_proxy(p):
        if isinstance(p, thunder.Proxy):
            if p.name in proxymap:  # happens with subsymbols
                return p
            with thunder.core.trace.tracectx(t):
                np = p.replace(**proxy_replacement_metadata.get(p.name, {}))
                proxymap[p.name] = np
                return np
        return p

    def process_bound_symbols(src_bound_symbols, target_bound_symbols):
        for bsym in src_bound_symbols:
            new_args = tree_map(map_proxy, bsym.args)
            new_kwargs = tree_map(map_proxy, bsym.kwargs)
            new_output = tree_map(create_proxy, bsym.output)
            new_bsym = bsym.from_bsym(output=new_output, args=new_args, kwargs=new_kwargs, subsymbols=[])
            target_bound_symbols.append(new_bsym)
            if len(bsym.subsymbols) > 0:
                process_bound_symbols(bsym.subsymbols, new_bsym.subsymbols)

    process_bound_symbols(trace.bound_symbols, t.bound_symbols)

    t.args = tree_map(map_proxy, trace.args)
    t.kwargs = tree_map(map_proxy, trace.kwargs)
    t._siginfo = trace._siginfo
    return t


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
            param = tm.get_parameter(n)
            n_absmax = f"{n}.absmax"
            n_code = f"{n}.code"
            param_absmax = tm.get_parameter(n_absmax)
            param_code = tm.get_parameter(n_code)
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
                    )
                    proxy_code = thunder.TensorProxy(
                        name=f"{get_param.output.name}_code",
                        shape=qs["code.shape"],
                        dtype=thunder.dtypes.to_dtype(qs["code.dtype"]),
                        device=thunder.devices.to_device(device),
                        requires_grad=False,
                    )
                    # get_param.sym = unpack_buffer/parameter as needed
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_absmax, output=proxy_absmax))
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_code, output=proxy_code))
                    add_trace_output(prologue_trace, proxy_absmax, subindex=0)
                    add_trace_output(prologue_trace, proxy_code, subindex=0)
                    new_compute_inputs.append(proxy_absmax)
                    new_compute_inputs.append(proxy_code)
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

        new_computation_trace = trace_with_replaced_proxy_metadata(computation_trace, computation_proxy_map)
        bound_symbols = new_computation_trace.bound_symbols
        new_computation_trace.bound_symbols = []

        new_computation_trace.args = (*new_computation_trace.args, *new_compute_inputs)
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
                n = quantized_proxies[bsym.args[1].name]
                qs = self.quant_states[n]
                # signature of the new symbol:
                # bnb_matmul_nf4(x, qweight, bias, absmax, quant_map, blocksize, dtype, shape)
                new_args = (
                    *bsym.args[:3],
                    additional_proxies[f"{n}.absmax"],
                    additional_proxies[f"{n}.code"],
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


class LORATransform(Transform):
    def __init__(
        self,
        *,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        weights: list[str] = [],
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_linear_names = set()
        self.lora_linear_states = {}

    def init_lora_linear(self, lora_a, lora_b):
        torch.nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        torch.nn.init.zeros_(lora_b)

    def check_proxy(self, x: TensorProxy, shape, device, dtype):
        assert x.shape == shape
        assert x.device == device
        assert x.dtype == dtype

    def transform_module(self, model: thunder.ThunderModule):
        self.thunder_module = model
        shared_names = model._get_shared_names()
        processed_names = set()

        def convert_linear_submodule(tm, name, r):
            self.lora_linear_names.add(name)
            weight_name = f"{name}.weight"
            processed_copies = shared_names[weight_name] & processed_names
            if processed_copies:
                copy_name = next(iter(processed_copies))
                self.lora_linear_states[weight_name] = self.lora_linear_states[copy_name]
                tm._overrides_parameters[weight_name] = tm._overrides_parameters[copy_name]
                tm._overrides_parameters[f"{weight_name}.lora_a"] = tm._overrides_parameters[f"{copy_name}.lora_a"]
                tm._overrides_parameters[f"{weight_name}.lora_b"] = tm._overrides_parameters[f"{copy_name}.lora_b"]
                processed_names.add(weight_name)
                return

            w = tm.get_parameter(weight_name)
            w.requires_grad_(False)
            in_features, out_features = w.shape[0], w.shape[1]

            lora_a = torch.nn.Parameter(w.new_empty((r, in_features)), requires_grad=False)
            lora_b = torch.nn.Parameter(w.new_empty((out_features, r)), requires_grad=False)
            self.init_lora_linear(lora_a, lora_b)

            tm._overrides_parameters[weight_name] = w
            tm._overrides_parameters[f"{weight_name}.lora_a"] = lora_a
            tm._overrides_parameters[f"{weight_name}.lora_b"] = lora_b

            self.lora_linear_states[weight_name] = {
                "dtype": w.dtype,
                "shape": tuple(w.shape),
                "lora_a.dtype": lora_a.dtype,
                "lora_a.shape": tuple(lora_a.shape),
                "lora_b.dtype": lora_b.dtype,
                "lora_b.shape": tuple(lora_b.shape),
                "device": getattr(w, "_thunder_device", w.device),
            }
            processed_names.add(name)

        for n, submodule in model._model.named_modules():
            if isinstance(submodule, torch.nn.Linear):
                convert_linear_submodule(model, n, self.r)

    def transform_state_dict_for_submodule(
        self, model: thunder.ThunderModule, submodule_name: str, state_dict: dict
    ) -> dict:
        if submodule_name not in self.lora_linear_names:
            return state_dict

        w = state_dict["weight"]
        in_features, out_features = w.shape[0], w.shape[1]
        lora_a = torch.nn.Parameter(w.new_empty((self.r, in_features)))
        lora_b = torch.nn.Parameter(w.new_empty((out_features, self.r)))
        self.init_lora_linear(lora_a, lora_b)

        state_dict = state_dict.copy()
        state_dict[f"weight.lora_a"] = lora_a
        state_dict[f"weight.lora_b"] = lora_b

        return state_dict

    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        tm = self.thunder_module
        from thunder.core.trace import tracectx

        checks = get_checks(prologue_trace)

        prologue_trace = trace_with_replaced_proxy_metadata(prologue_trace, {})

        checks = get_checks(prologue_trace)
        proglogue_to_compute_outputs = prologue_trace.output[0]
        output_idxes = {o.name: i for i, o in enumerate(proglogue_to_compute_outputs)}

        lora_linear_proxies: dict[str, str] = {}  # proxy_name -> param_name
        additional_proxies: dict[str, thunder.Proxy] = {}  # param_name.(absmax|code) -> Proxy

        new_bsyms = []
        new_compute_inputs = []
        for n, qs in self.lora_linear_states.items():
            param = tm.get_parameter(n)
            n_lora_a = f"{n}.lora_a"
            n_lora_b = f"{n}.lora_b"
            n_linear = f"{n}"
            check, get_param = checks[n]
            lora_linear_proxies[get_param.output.name] = n
            # check has args: tensor, shape, device, dtype, requires_grad
            proxy, _, device, _, requires_grad = check.args
            check.args = (proxy, qs["shape"], device, qs["dtype"], False)

            output_idx = output_idxes.get(get_param.output.name)
            if output_idx is not None:
                with tracectx(prologue_trace):
                    # better way
                    proxy_lora_a = thunder.TensorProxy(
                        name=f"{get_param.output.name}_lora_a",
                        shape=qs["lora_a.shape"],
                        dtype=thunder.dtypes.to_dtype(qs["lora_a.dtype"]),
                        device=thunder.devices.to_device(device),
                        requires_grad=False,
                    )
                    proxy_lora_b = thunder.TensorProxy(
                        name=f"{get_param.output.name}_lora_b",
                        shape=qs["lora_b.shape"],
                        dtype=thunder.dtypes.to_dtype(qs["lora_b.dtype"]),
                        device=thunder.devices.to_device(device),
                        requires_grad=False,
                    )
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_lora_a, output=proxy_lora_a))
                    new_bsyms.append(get_param.sym.bind(get_param.args[0], n_lora_b, output=proxy_lora_b))

                    add_trace_output(prologue_trace, proxy_lora_a, subindex=0)
                    add_trace_output(prologue_trace, proxy_lora_b, subindex=0)

                    new_compute_inputs.append(proxy_lora_a)
                    new_compute_inputs.append(proxy_lora_b)
                    new_bsyms.append(
                        prims.check_tensor_shape_and_metadata.bind(
                            proxy_lora_a,
                            qs["lora_a.shape"],
                            device,
                            qs["lora_a.dtype"],
                            False,
                            output=None,
                        )
                    )
                    new_bsyms.append(
                        prims.check_tensor_shape_and_metadata.bind(
                            proxy_lora_b,
                            qs["lora_b.shape"],
                            device,
                            qs["lora_b.dtype"],
                            False,
                            output=None,
                        )
                    )
                    # this is not good, because we will have several traces...
                    additional_proxies[n_lora_a] = proxy_lora_a
                    additional_proxies[n_lora_b] = proxy_lora_b

        prologue_trace.bound_symbols[-1:-1] = new_bsyms
        prologue_trace.set_provenance(thunder.core.trace.TraceProvenance("lora linear pass"))

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

        swap_map = dict()

        for bsym in bound_symbols[idx:]:
            if bsym.sym == thunder.torch.linear and bsym.args[1].name in lora_linear_proxies:
                n = lora_linear_proxies[bsym.args[1].name]
                with tracectx(new_computation_trace):
                    new_computation_trace.push_scope(new_computation_trace.bound_symbols)
                    lora_a_bsym = thunder.torch.transpose(additional_proxies[f"{n}.lora_a"], 1, 0)
                    lora_b_bsym = thunder.torch.transpose(additional_proxies[f"{n}.lora_b"], 1, 0)
                    lora_bsym_1 = thunder.torch.matmul(bsym.args[0], lora_a_bsym)
                    lora_bsym = thunder.torch.matmul(lora_bsym_1, lora_b_bsym)
                    original_weight = thunder.torch.linear(*bsym.args[:3])
                    lora_sum = thunder.torch.add(original_weight, lora_bsym)
                    new_scope = new_computation_trace.pop_scope()
                    var_original_bsym_output = variableify(bsym.flat_proxy_outs[0])
                    swap_map[var_original_bsym_output] = lora_sum
            else:
                new_computation_trace.bound_symbols.append(bsym.from_bsym_swap_proxies(swap_map))

        new_computation_trace.set_provenance(thunder.core.trace.TraceProvenance("lora linear pass"))
        return prologue_trace, new_computation_trace, epilogue_trace
