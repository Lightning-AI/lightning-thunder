"""Early transform for `fsdp(jit(model))` to convert a trace into fsdp."""

from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from thunder.core import devices
from thunder.core import prims
from thunder.core import utils
from thunder.core.proxies import DistParallelType
from thunder.core.proxies import TensorProxy
from thunder.core.proxies import variableify, unvariableify
from thunder.core.trace import from_trace
from thunder.core.trace import tracectx
from thunder.core.trace import TraceProvenance
from thunder.core.transforms import VISIT_TYPE
from thunder.core.transforms import visitor_transform
from thunder.core.transform_common import Transform

if TYPE_CHECKING:
    from typing import Any
    from torch import Tensor
    from torch.nn import Parameter
    from torch.distributed import ProcessGroup
    from thunder.core.symbol import BoundSymbol
    from thunder.core.trace import VariableInterface


__all__ = [
    "FSDPTraceTransform",
]


@dataclass
class FSDPParamUnpaddingVisitor:
    prod_bsym_to_unsharded_and_padding: dict[BoundSymbol, tuple[TensorProxy, int]]
    swap_map: dict[VariableInterface, TensorProxy] = field(init=False, default_factory=dict)

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        import thunder.torch as ltorch
        from thunder.core.trace import get_tracectx

        if bsym.sym.id in {prims.PrimIDs.UNPACK_TRIVIAL, prims.PrimIDs.UNPACK_SEQUENCE}:
            return VISIT_TYPE.NO_OP

        if bsym in self.prod_bsym_to_unsharded_and_padding:
            padded_tensor, padding_size = self.prod_bsym_to_unsharded_and_padding[bsym]
            out = bsym.flat_proxy_outs[0]
            utils.check(out.name == padded_tensor.name, lambda: f"{out.name=}, {padded_tensor.name=}")
            unpadded_tensor = ltorch.getitem(padded_tensor, slice(None, -padding_size, None))
            self.swap_map[variableify(padded_tensor)] = unpadded_tensor
            return VISIT_TYPE.INSERT_AFTER

        if not any(variableify(a) in self.swap_map for a in bsym.flat_args):
            return VISIT_TYPE.NO_OP
        updated_bsym = bsym.from_bsym_swap_proxies(self.swap_map)
        get_tracectx().scopes[-1].append(updated_bsym)
        return VISIT_TYPE.REPLACE


@dataclass(frozen=True)
class FSDPTraceTransform(Transform):
    sharded_params: dict[str, Any]
    process_group: ProcessGroup
    shared_params_name: dict[str, str]
    disable_cpu_offloading: bool = False

    def transform_traces_pre_prologue(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        from thunder.distributed import prims as dist_prims

        prologue_producers, prologue_consumers = utils.producers_and_consumers(prologue_trace)

        modules_and_thunder_modules = [
            (bsym.args[0], bsym.output)
            for bsym in prologue_trace.bound_symbols
            if bsym.sym is prims.unpack_thunder_module
        ]

        if len(modules_and_thunder_modules) != 1:
            raise NotImplementedError("cannot deal with modules other than the compiled module")

        ((orig_module_proxy, thunder_module_proxy),) = modules_and_thunder_modules
        if prologue_producers[orig_module_proxy].sym is not prims.unpack_function_obj:
            raise NotImplementedError("original module does not match the compiled module")

        computation_trace.push_scope([])

        unsharded_to_padding: dict[VariableInterface, TensorProxy] = {}

        synchronized_parameters = []
        param_name_to_comp_trc_proxy = {}  # Track param_name to it's corresponding proxy in computation_trc.
        # todo: deal with epilogue output
        for pro_out_p, comp_inp_p in zip(prologue_trace.output[0], computation_trace.args):
            bsym = prologue_producers[pro_out_p]
            if bsym.sym == prims.unpack_parameter:
                param_thunder_module, param_name = bsym.args
                assert param_thunder_module is thunder_module_proxy
                if param_name in self.sharded_params:
                    param_name_to_comp_trc_proxy[param_name] = comp_inp_p
                    old_shape, new_shape, new_torch_device = self.sharded_params[param_name]
                    thunder_device = devices.to_device(new_torch_device)
                    thunder_device_str = thunder_device.device_str()

                    pro_out_p._distparallel_type = DistParallelType.FULLY_SHARDED
                    pro_out_p._shape = tuple(new_shape)
                    pro_out_p._device = thunder_device
                    if comp_inp_p is not pro_out_p:
                        comp_inp_p._distparallel_type = DistParallelType.FULLY_SHARDED
                        comp_inp_p._shape = tuple(new_shape)
                        comp_inp_p._device = thunder_device
                    with tracectx(computation_trace):
                        synchronized_parameters.append(dist_prims.synchronize(comp_inp_p, self.process_group))
                    if (padding_size := new_shape[0] * self.process_group.size() - old_shape[0]) > 0:
                        unsharded_to_padding[variableify(synchronized_parameters[-1])] = padding_size

                    for c in prologue_consumers[pro_out_p]:
                        if c.sym is prims.check_tensor_shape_and_metadata:
                            # TODO have a more principled way to update this?
                            a0, _, _, *a2pp = c.args
                            c.args = (a0, tuple(new_shape), thunder_device_str, *a2pp)

        new_scope = computation_trace.pop_scope()

        for bsym in prologue_trace.bound_symbols:
            if bsym.sym is prims.check_tensor_shape_and_metadata and prologue_producers[bsym.args[0]].sym in (
                prims.unpack_parameter,
                prims.unpack_buffer,
            ):
                param_thunder_module, name = prologue_producers[bsym.args[0]].args
                assert param_thunder_module is thunder_module_proxy
                if name not in self.sharded_params:
                    a0, shape, _, *a2pp = bsym.args
                    bsym.args = (a0, shape, thunder_device_str, *a2pp)

        proxies_to_replace = {id(bsym.args[0]): bsym.output for bsym in new_scope}

        # See NOTE: Shared Parameters in Trace
        for param_name, base_param in self.shared_params_name.items():
            param_proxy = param_name_to_comp_trc_proxy[param_name]
            base_param_proxy = param_name_to_comp_trc_proxy[base_param]
            allgather_base_param_proxy = proxies_to_replace[id(base_param_proxy)]
            # Update `proxies_to_replace` so we replace all usage of `param_proxy`
            # with the output of `AllGather` on `base_param_proxy`.
            proxies_to_replace[id(param_proxy)] = allgather_base_param_proxy

        new_computation_trace = from_trace(computation_trace)
        for idx, bsym in enumerate(computation_trace.bound_symbols):
            if bsym.sym != prims.unpack_trivial:
                break
            new_computation_trace.bound_symbols.append(bsym.from_bsym())
        new_computation_trace.bound_symbols += new_scope
        for bsym in computation_trace.bound_symbols[idx:]:
            new_args = tuple(proxies_to_replace.get(id(a), a) for a in bsym.args)
            new_computation_trace.bound_symbols.append(bsym.from_bsym(args=new_args))

        new_computation_trace.set_provenance(TraceProvenance("fsdp pass"))
        if unsharded_to_padding:
            producer_map, consumer_map = utils.producers_and_consumers(new_computation_trace)
            prod_bsym_to_unsharded_and_padding: dict[BoundSymbol, tuple[TensorProxy, int]] = {}
            for var_padded_tensor, padding_size in unsharded_to_padding.items():
                padded_tensor = unvariableify(var_padded_tensor)
                prod_bsym = producer_map[padded_tensor]
                prod_bsym_to_unsharded_and_padding[prod_bsym] = (padded_tensor, padding_size)
            visit = FSDPParamUnpaddingVisitor(prod_bsym_to_unsharded_and_padding)
            new_computation_trace = visitor_transform(
                trace_from=new_computation_trace,
                visit=visit,
                provenance="fsdp pass with unpadding",
            )
        return prologue_trace, new_computation_trace, epilogue_trace

    def transform_state_dict(
        self,
        state_dict: dict[str, Parameter | Tensor],
    ) -> dict[str, Parameter | Tensor]:
        from thunder.executors.torchex import _all_gather_prim_impl
        import torch

        for name, param_or_buffer in state_dict.items():
            if not torch.is_tensor(param_or_buffer):
                continue
            unsharded_param_or_buffer = _all_gather_prim_impl(
                param_or_buffer, group=self.process_group, do_async=False, dim=0
            )
            if (padding := getattr(param_or_buffer, "_thunder_fsdp_padding_size", None)) is not None:
                unsharded_param_or_buffer = unsharded_param_or_buffer.narrow(
                    0, 0, unsharded_param_or_buffer.size(0) - padding
                )
                if not self.disable_cpu_offloading:
                    state_dict[name] = unsharded_param_or_buffer.cpu()
                else:
                    state_dict[name] = unsharded_param_or_buffer
        return state_dict
