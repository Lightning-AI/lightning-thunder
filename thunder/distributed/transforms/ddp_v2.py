from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from thunder.core import devices
from thunder.core import prims
from thunder.core import utils
from thunder.core.proxies import DistParallelType, TensorProxy, variableify
from thunder.core.trace import from_trace
from thunder.core.trace import tracectx
from thunder.core.trace import TraceProvenance
from thunder.core.transform_common import Transform
from thunder.core.module import ThunderModule
import torch
from torch.utils.weak import WeakTensorKeyDictionary
import torch.distributed as tdist
import copy

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from thunder.core.trace import TraceCtx


@dataclass
class DDPTransform(Transform):
    process_group: ProcessGroup
    bucket_size_in_mb: float
    broadcast_from: int | None

    replicated_params: dict[str, torch.nn.Parameter] | None = None
    shared_params_name: dict[str, str] | None = None

    def transform_module(self, model: ThunderModule):
        """Transforms the ThunderModule. This is executed once on application of the transform"""
        from thunder import compile_data as get_compile_data
        from thunder.core.module import ThunderModule

        process_group = self.process_group
        cd = get_compile_data(model)
        cd.use_ddp = True
        cd.process_group_for_ddp = process_group
        orig_module: torch.nn.Module = cd.fn
        utils.check(
            isinstance(orig_module, torch.nn.Module) and not isinstance(orig_module, ThunderModule),
            lambda: f"CompileData.fn expected to be `nn.Module` but {type(orig_module)}",
        )
        orig_module.use_ddp = True
        orig_module.process_group_for_ddp = process_group
        orig_module.bucket_size_in_mb = self.bucket_size_in_mb

        replicated_params = {}
        # We use `shared_params` dictionary to track the shared parameters.
        # Key to this dictionary is the original parameter from the user's Module.
        # Values are the copied and sharded parameter for the thunder module and meta-data related to sharding.
        shared_params = WeakTensorKeyDictionary()

        # NOTE: Shared Parameters in Trace
        # Shared parameters in PyTorch eager are parameters of module which have different name but share the underlying tensor.
        # For shared parameter, we replace all occurence shared parameter with it's corresponding `base` parameter.
        # In our implementation `base` parameter is the parameter and corresponding name which we see the first time while
        # iterating our parameters (see below). We track subsequent parameter which share the underlying Tensor with this `base` parameter
        # in `shared_params_name` dictionary.
        # Then while, transforming the trace - `see DDPTraceTransform.transform_traces` - we replace all the proxy of shared parameter
        # with the corresponding proxy of base parameter in the computation trace.

        # This is used to track the shared parameters when the transform is applied.
        # key - parameter name, value - `base` parameter name.
        shared_params_name: dict[str, str] = {}
        for module_name, _ in model._model.named_modules():
            submodule = model.get_submodule(module_name)
            # Since we are doing no sharding, we do not need to materialize the params

            # Broadcast parameters if requested
            if self.broadcast_from is not None:
                for pn, _ in submodule.named_parameters(recurse=False, prefix=module_name):
                    tdist.broadcast(
                        model.get_parameter(pn), group_src=self.broadcast_from, group=process_group, async_op=False
                    )
                for pn, _ in submodule.named_buffers(recurse=False, prefix=module_name):
                    tdist.broadcast(
                        model.get_buffer(pn), group_src=self.broadcast_from, group=process_group, async_op=False
                    )

            # note: we use model.get_parameter rather than the submodule's named parameters becasue we want the
            #       ThunderModule's parameter overrides for composability.
            for pn, _ in submodule.named_parameters(recurse=False, prefix=module_name):
                # If there are shared params in the original user Module, we reuse the sharded copy created from the original parameter below.
                # This way we re-create parameter sharing in thunder's copy of the Module.
                p = model.get_parameter(pn)
                if p in shared_params:
                    # Shared param names : current param - base param
                    shared_params_name[pn] = shared_params[p]["param_name"]
                    # Re-use the previous copy of this parameter.
                    model._overrides_parameters[pn] = shared_params[p]["param_copy"]
                    replicated_params[pn] = shared_params[p]["param_meta"]
                    continue

                model._overrides_parameters[pn] = copy.copy(p)
                # we collect shapes and devices because we do not know if other transforms also change it...
                shape = model._overrides_parameters[pn].shape
                replicated_params[pn] = (shape, model._overrides_parameters[pn].device)

                # Track param information
                shared_params[p] = {
                    "param_copy": model._overrides_parameters[pn],
                    "param_meta": replicated_params[pn],
                    "param_name": pn,
                }
        self.shared_params_name = shared_params_name
        self.replicated_params = replicated_params

    def transform_traces_pre_prologue(
        self, prologue_trace: TraceCtx, computation_trace: TraceCtx, epilogue_trace: TraceCtx, **kwargs
    ):
        assert self.replicated_params is not None and self.shared_params_name is not None, (
            "expected transform_module to have run"
        )

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
                if param_name in self.replicated_params:
                    param_name_to_comp_trc_proxy[param_name] = comp_inp_p
                    shape, torch_device = self.replicated_params[param_name]
                    thunder_device = devices.to_device(torch_device)
                    pro_out_p._distparallel_type = DistParallelType.REPLICATED
                    pro_out_p._device = thunder_device
                    if comp_inp_p is not pro_out_p:
                        comp_inp_p._distparallel_type = DistParallelType.REPLICATED
                        comp_inp_p._device = thunder_device
                    with tracectx(computation_trace):
                        # we will produce a new trace with syncs before using the weights
                        # then, the backward sync will be automatically handled by thunder (inserting all_reduce for the gradients)
                        # then, syncs will be removed from the forward pass (as expected, since they are not needed)
                        synchronized_parameters.append(
                            dist_prims.synchronize(comp_inp_p, self.process_group, DistParallelType.REPLICATED)
                        )
        new_scope = computation_trace.pop_scope()
        # map of param -> synced param
        proxies_to_replace = {variableify(bsym.args[0]): bsym.output for bsym in new_scope}

        # See NOTE: Shared Parameters in Trace
        for param_name, base_param in self.shared_params_name.items():
            param_proxy = param_name_to_comp_trc_proxy[param_name]
            base_param_proxy = param_name_to_comp_trc_proxy[base_param]
            synced_base_param_proxy = proxies_to_replace[variableify(base_param_proxy)]
            # Update `proxies_to_replace` so we replace all usage of `param_proxy`
            # with the output of the synced param on `base_param_proxy`.
            proxies_to_replace[variableify(param_proxy)] = synced_base_param_proxy

        new_computation_trace = from_trace(computation_trace)
        for idx, bsym in enumerate(computation_trace.bound_symbols):
            if bsym.sym != prims.unpack_trivial:
                break
            new_computation_trace.bound_symbols.append(bsym.from_bsym(args=bsym.args))

        new_computation_trace.bound_symbols += new_scope
        for bsym in computation_trace.bound_symbols[idx:]:
            if bsym.sym == prims.python_return:
                # we need to preserve flat_args
                # skipping the swapping this assumes we don't return sharded params, but that should be OK
                assert not any(
                    (variableify(o) in proxies_to_replace) for o in bsym.args[0]["output"] if isinstance(o, TensorProxy)
                )
                new_computation_trace.bound_symbols.append(bsym.from_bsym())
                continue
            # replace param by synced_param
            new_computation_trace.bound_symbols.append(bsym.from_bsym_swap_proxies(proxies_to_replace))

        new_computation_trace.set_provenance(TraceProvenance("ddp pass"))

        return prologue_trace, new_computation_trace, epilogue_trace
