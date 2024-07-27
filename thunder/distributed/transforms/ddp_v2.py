
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
from thunder.core.transform_common import Transform

if TYPE_CHECKING:
    from typing import Any
    from torch.distributed import ProcessGroup
    from thunder.core.trace import TraceCtx


@dataclass(frozen=True)
class DDPTraceTransform(Transform):
    process_group: ProcessGroup

    def transform_traces(self, prologue_trace: TraceCtx, computation_trace: TraceCtx, epilogue_trace: TraceCtx, **kwargs):
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
        param_name_to_comp_trc_proxy = {}  # Track param_name to it's corresponding proxy in computation_trc.
        for pro_out_p, comp_inp_p in zip(prologue_trace.output[0], computation_trace.args):
            bsym = prologue_producers[pro_out_p]
            # if the bsym is an unpack_parameter prim, we need to mark it as REPLICATED (ddp)
            # and insert a sync (then, backward pass will be handled automatically)
            if bsym.sym == prims.unpack_parameter:
                param_thunder_module, param_name = bsym.args
                assert param_thunder_module is thunder_module_proxy
                if True: # param_name in self.sharded_params:
                    param_name_to_comp_trc_proxy[param_name] = comp_inp_p
                    #thunder_device = devices.to_device(new_torch_device)
                    #thunder_device_str = thunder_device.device_str()

                    pro_out_p._distparallel_type = DistParallelType.REPLICATED
                    #pro_out_p._shape = tuple(new_shape)
                    #pro_out_p._device = thunder_device
                    if comp_inp_p is not pro_out_p:
                        #comp_inp_p._distparallel_type = DistParallelType.FULLY_SHARDED
                        #comp_inp_p._shape = tuple(new_shape)
                        #comp_inp_p._device = thunder_device
                        raise NotImplementedError("Expected comp_inp_p to be pro_out_p")
                    with tracectx(computation_trace):
                        # we will produce a new trace with syncs before using the weights
                        # then, the backward sync will be automatically handled by thunder (inserting all_reduce for the gradients)
                        # then, the augmented forward pass will remove the synchronizes from the forward (as expected)

                        synchronized_parameters.append(dist_prims.synchronize(comp_inp_p, self.process_group))
                    
                    # not needed for ddp, params are not sharded
                    """for c in prologue_consumers[pro_out_p]:
                        if c.sym is prims.check_tensor_shape_and_metadata:
                            # TODO have a more principled way to update this?
                            a0, _, _, *a2pp = c.args
                            c.args = (a0, tuple(new_shape), thunder_device_str, *a2pp)"""
        

        new_scope = computation_trace.pop_scope()
        # new_scope contains the new sync prims
        # map of param -> synced param
        proxies_to_replace = {id(bsym.args[0]): bsym.output for bsym in new_scope}

        # See NOTE: Shared Parameters in Trace
        # TODO: handle shared params
        """for param_name, base_param in self.shared_params_name.items():
            param_proxy = param_name_to_comp_trc_proxy[param_name]
            base_param_proxy = param_name_to_comp_trc_proxy[base_param]
            allgather_base_param_proxy = proxies_to_replace[id(base_param_proxy)]
            # Update `proxies_to_replace` so we replace all usage of `param_proxy`
            # with the output of `AllGather` on `base_param_proxy`.
            proxies_to_replace[id(param_proxy)] = allgather_base_param_proxy"""

        new_computation_trace = from_trace(computation_trace)
        for idx, bsym in enumerate(computation_trace.bound_symbols):
            if bsym.sym != prims.unpack_trivial:
                break
            new_computation_trace.bound_symbols.append(bsym.from_bsym())
        # insert the new scope ops after unpack_trivial (why?)
        new_computation_trace.bound_symbols += new_scope
        for bsym in computation_trace.bound_symbols[idx:]:
            # replace param by synced_param
            new_args = tuple(proxies_to_replace.get(id(a), a) for a in bsym.args)
            new_computation_trace.bound_symbols.append(bsym.from_bsym(args=new_args))

        new_computation_trace.set_provenance(TraceProvenance("ddp pass"))
        
        return prologue_trace, new_computation_trace, epilogue_trace
        
