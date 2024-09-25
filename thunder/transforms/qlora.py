import thunder
from thunder.core.proxies import TensorProxy
from thunder.core.transform_common import Transform
from thunder.core import prims
from .quantization import trace_with_replaced_proxy_metadata
import torch
import math

from .utils import (
    get_checks,
    add_trace_output,
    trace_with_replaced_proxy_metadata,
)


class LORATransform(Transform):
    def __init__(
        self,
        *,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        weights: list[str] = [],
        merged: bool = False,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.scaling = self.lora_alpha / self.r
        self.lora_linear_names = set()
        self.lora_linear_states = {}
        self.merged = merged

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
            if bsym.sym == thunder.torch.linear and bsym.args[1].name in lora_linear_proxies:
                n = lora_linear_proxies[bsym.args[1].name]
                with tracectx(new_computation_trace):
                    new_computation_trace.push_scope(new_computation_trace.bound_symbols)
                    dropout_output = thunder.torch.dropout(bsym.args[0], p=self.lora_dropout)
                    lora_a_transpose = thunder.torch.transpose(additional_proxies[f"{n}.lora_a"], 1, 0)
                    lora_b_transpose = thunder.torch.transpose(additional_proxies[f"{n}.lora_b"], 1, 0)
                    lora_matmul1 = thunder.torch.matmul(dropout_output, lora_a_transpose)
                    lora_matmul2 = thunder.torch.matmul(lora_matmul1, lora_b_transpose)
                    lora_scaled = thunder.torch.mul(lora_matmul2, self.scaling)
                    original_weight = thunder.torch.linear(*bsym.args[:3])
                    original_proxy_output = bsym.flat_proxy_outs[0]
                    new_computation_trace.bound_symbols.append(
                        prims.add.bind(original_weight, lora_scaled, output=original_proxy_output)
                    )
                    new_scope = new_computation_trace.pop_scope()
            else:
                new_computation_trace.bound_symbols.append(bsym.from_bsym())
        new_computation_trace.set_provenance(thunder.core.trace.TraceProvenance("lora linear pass"))
        return prologue_trace, new_computation_trace, epilogue_trace
