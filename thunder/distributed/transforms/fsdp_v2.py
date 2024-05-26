"""Early transform for `fsdp(jit(model))` to convert a trace into fsdp."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from thunder.core import devices
from thunder.core import prims
from thunder.core import utils
from thunder.core.proxies import DistParallelType
from thunder.core.trace import from_trace
from thunder.core.trace import tracectx
from thunder.core.trace import TraceProvenance

if TYPE_CHECKING:
    from typing import Any
    from torch.distributed import ProcessGroup


__all__ = [
    "FSDPTraceTransform",
]


@dataclass(frozen=True)
class FSDPTraceTransform:
    sharded_params: dict[str, Any]
    process_group: ProcessGroup

    def __call__(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
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

        synchronized_parameters = []
        # todo: deal with epilogue output
        for pro_out_p, comp_inp_p in zip(prologue_trace.output, computation_trace.args):
            bsym = prologue_producers[pro_out_p]
            if bsym.sym == prims.unpack_parameter:
                param_thunder_module, param_name = bsym.args
                assert param_thunder_module is thunder_module_proxy
                if param_name in self.sharded_params:
                    old_shape, new_shape, new_torch_device = self.sharded_params[param_name]
                    thunder_device = devices.to_device(new_torch_device)
                    thunder_device_str = str(thunder_device)

                    pro_out_p._distparallel_type = DistParallelType.FULLY_SHARDED
                    pro_out_p._shape = tuple(new_shape)
                    pro_out_p._device = thunder_device
                    if comp_inp_p is not pro_out_p:
                        comp_inp_p._distparallel_type = DistParallelType.FULLY_SHARDED
                        comp_inp_p._shape = tuple(new_shape)
                        comp_inp_p._device = thunder_device
                    with tracectx(computation_trace):
                        synchronized_parameters.append(dist_prims.synchronize(comp_inp_p, self.process_group))

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

        return prologue_trace, new_computation_trace, epilogue_trace
