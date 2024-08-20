"""Transform for `fsdp(jit(model))` to convert a trace into fsdp."""

from __future__ import annotations
import copy
from dataclasses import dataclass
from dataclasses import field
from itertools import chain
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as tdist
from torch.utils.weak import WeakTensorKeyDictionary

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
from thunder.distributed import copy_default_process_group, FSDPType, FSDPBucketingStrategy, _shard_param, _materialize

if TYPE_CHECKING:
    from typing import Any
    from torch.distributed import ProcessGroup
    from thunder.core.symbol import BoundSymbol
    from thunder.core.trace import VariableInterface


__all__ = [
    "FSDPTransform",
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


# When the user calls fsdp(jitted_module), this function does the following
# - It transforms the ThunderModule jitted_module, materializing and sharding the parameters as `overrides`
#   in the ThunderModule.
# - While doing that, it leaves the original user module alone.
# - It then registers an transform (callback that runs before prologue is executed) that transforms the
#   prologue and compute trace.
#
# Note that for doing so, there are a few constraints / caveats:
# - We do not have prologues/compute traces when we transform the module.
# - We need to record the info from the module transformations because a later transform might modify the module further.


class FSDPTransform(Transform):
    sharded_params: dict[str, Any]
    process_group: ProcessGroup
    shared_params_name: dict[str, str]

    def __init__(
        self,
        device: torch.device | None = None,
        broadcast_from: int | None = None,
        sharding_strategy: FSDPType = FSDPType.ZERO2,
        bucketing_strategy: FSDPBucketingStrategy = FSDPBucketingStrategy.NONE,
        release_original_parameters: bool = False,
    ):
        self.device = device
        self.broadcast_from = broadcast_from
        self.sharding_strategy = sharding_strategy
        self.bucketing_strategy = bucketing_strategy
        self.release_original_parameters = release_original_parameters
        self.sharded_params: dict[str, Any] = {}
        self.process_group: ProcessGroup | None = None
        self.shared_params_name: dict[str, str] = {}

    def transform_module(
        self,
        thunder_model: ThunderModule,
    ):
        from thunder import compile_data as get_compile_data
        from thunder.core.transforms import add_transform
        from thunder.core.module import ThunderModule

        self.process_group = copy_default_process_group()
        utils.check(self.process_group is not None, lambda: "The default process group is None")
        global_rank = tdist.get_rank(group=self.process_group)
        world_size = tdist.get_world_size(group=self.process_group)
        if self.device is None:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device("cuda", local_rank)

        cd = get_compile_data(thunder_model)
        # TODO: promote use_fsdp and use_ddp to public members of CompileData
        cd.use_fsdp = True
        orig_module: torch.nn.Module = cd.fn
        utils.check(
            isinstance(orig_module, torch.nn.Module) and not isinstance(orig_module, ThunderModule),
            lambda: f"CompileData.fn expected to be `nn.Module` but {type(orig_module)}",
        )
        orig_module.use_fsdp = True
        orig_module.process_group_for_ddp = self.process_group
        orig_module.bucketing_strategy = self.bucketing_strategy
        orig_module.sharding_strategy = self.sharding_strategy

        # modify module
        self.sharded_params = {}
        device_adjustments = {}
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
        # Then while, transforming the trace - `see FSDPTraceTransform.transform_traces` - we replace all the proxy of shared parameter
        # with the corresponding proxy of base parameter in the computation trace.

        # This is used to track the shared parameters when the transform is applied.
        # key - parameter name, value - `base` parameter name.
        self.shared_params_name: dict[str, str] = {}
        for module_name, _ in thunder_model._model.named_modules():
            submodule = thunder_model.get_submodule(module_name)

            # we use a copy to let the user's module alone (TODO: does this fully work?)
            module_copy = copy.copy(submodule)
            # TODO: we should probably populate the module copy with parameters that consider overrides

            # Materialize meta-parameters on-device if necessary.
            # This is done before sharding in case the materialization logic depends on the tensor shape.
            # The tradeoff is that all of a module's direct parameters need to fit in device.
            # Each module only initializes its own parameters and not those of its children (recurse=False)
            if any(t.is_meta for t in chain(module_copy.parameters(recurse=False), module_copy.buffers(recurse=False))):
                # TODO: we could also support calling a "param_init_fn" argument like PyTorch
                _materialize(module_copy, self.device)
                for n, p in module_copy.named_parameters(recurse=False, prefix=module_name):
                    thunder_model._overrides_parameters[n] = p
                    device_adjustments[n] = self.device
                for n, b in module_copy.named_buffers(recurse=False, prefix=module_name):
                    thunder_model._overrides_buffers[n] = b
                    device_adjustments[n] = self.device
            else:
                # Move leftover params and buffers to device. This is at least required to broadcast.
                # Cannot `submodule.to(device)` because we don't want it to recurse
                for n, p in module_copy.named_parameters(recurse=False, prefix=module_name):
                    if p.device != self.device:
                        thunder_model._overrides_parameters[n] = torch.nn.Parameter(
                            p.to(device=self.device), requires_grad=p.requires_grad
                        )
                        device_adjustments[n] = self.device
                for n, b in module_copy.named_buffers(recurse=False, prefix=module_name):
                    if b.device != self.device:
                        thunder_model._overrides_buffers[n] = b.to(device=self.device)
                        device_adjustments[n] = self.device

            # Broadcast parameters if requested
            if self.broadcast_from is not None:
                for pn, _ in submodule.named_parameters(recurse=False, prefix=module_name):
                    tdist.broadcast(
                        thunder_model.get_parameter(pn),
                        src=self.broadcast_from,
                        group=self.process_group,
                        async_op=False,
                    )
                for pn, _ in submodule.named_buffers(recurse=False, prefix=module_name):
                    tdist.broadcast(
                        thunder_model.get_buffer(pn), src=self.broadcast_from, group=self.process_group, async_op=False
                    )

            for pn, p in submodule.named_parameters(recurse=False, prefix=module_name):
                # If there are shared params in the original user Module, we reuse the sharded copy created from the original parameter below.
                # This way we re-create parameter sharing in thunder's copy of the Module.
                if p in shared_params:
                    # Shared param names : current param - base param
                    self.shared_params_name[pn] = shared_params[p]["param_name"]
                    # Re-use the previous copy of this parameter.
                    thunder_model._overrides_parameters[pn] = shared_params[p]["param_copy"]
                    self.sharded_params[pn] = shared_params[p]["param_shard_meta"]
                    continue

                # if we don't have an override or it is just the original, do create a copy
                if thunder_model._overrides_parameters.get(pn, p) is p:
                    thunder_model._overrides_parameters[pn] = copy.copy(p)
                # we collect shapes and devices because we do not know if other transforms also change it...
                old_shape = thunder_model._overrides_parameters[pn].shape
                _shard_param(
                    thunder_model._overrides_parameters[pn], global_rank, world_size, pn, allow_padding_for_fsdp=True
                )
                new_shape = thunder_model._overrides_parameters[pn].shape
                self.sharded_params[pn] = (old_shape, new_shape, thunder_model._overrides_parameters[pn].device)
                if self.release_original_parameters:
                    base_pn = pn.rsplit(".", 1)[-1]
                    p_orig = getattr(submodule, base_pn)
                    if p_orig.device.type != "meta":
                        p_meta = torch.nn.Parameter(p.to(device="meta"), requires_grad=p.requires_grad)
                        p_meta._thunder_device = p_orig.device
                        submodule.register_parameter(base_pn, p_meta)
                    else:
                        p_orig._thunder_device = self.device

                # Track the original param and it's corresponding copied shard and metadata.
                shared_params[p] = {
                    "param_copy": thunder_model._overrides_parameters[pn],
                    "param_shard_meta": self.sharded_params[pn],
                    "param_name": pn,
                }

    def transform_state_dict_for_submodule(
        self, model: thunder.ThunderModule, submodule_name: str, state_dict: dict
    ) -> dict:
        raise NotImplementedError("cannot transform state dict yet")

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
