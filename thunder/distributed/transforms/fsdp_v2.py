"""Transform for `fsdp(jit(model))` to convert a model to use fsdp."""

from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from itertools import chain
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as tdist

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
from thunder.distributed import (
    copy_default_process_group,
    FSDPType,
    FSDPBucketingStrategy,
    _shard_tensor,
)

if TYPE_CHECKING:
    from typing import Any
    from torch.distributed import ProcessGroup
    from thunder.core.module import ThunderModule
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

        if not any(variableify(a) in self.swap_map for a in bsym.flat_proxy_args if isinstance(a, TensorProxy)):
            return VISIT_TYPE.NO_OP
        updated_bsym = bsym.from_bsym_swap_proxies(self.swap_map)
        get_tracectx().scopes[-1].append(updated_bsym)
        return VISIT_TYPE.REPLACE


# When the user calls fsdp(jitted_module), or applies this Transform directly, it does the following
# - It transforms the ThunderModule jitted_module, sharding the parameters as `overrides`
#   in the ThunderModule.
# - While doing that, it leaves the original user module alone, except when
#   releasing the original tensors is requested (for memory consumption).
# - When needed, a submodule state dict from the unsharded submodule can be transformed into a one of the sharded
#   submodule. This is used by MaterializationTransform and thunder_model.load_original_state_dict.
# - The prologue and compute trace are transformed, inserting communication and reflecting the weight shape changes.
#
# Note that for doing so, there are a few constraints / caveats:
# - We do not have prologues/compute traces when we transform the module.
# - We need to record the info from the module transformations because a later transform might modify the module further.
#
# The thunder.distributed.fsdp function calls FSDPTransform followed by MaterializationTransform, the latter does
# the materialization of submodules previously on the meta device.
class FSDPTransform(Transform):
    def __init__(
        self,
        device: torch.device | None = None,
        broadcast_from: int | None = None,
        sharding_strategy: FSDPType = FSDPType.ZERO2,
        bucketing_strategy: FSDPBucketingStrategy = FSDPBucketingStrategy.NONE,
        release_original_parameters: bool = False,
        move_state_dict_to_cpu: bool = False,
        *,
        process_group: ProcessGroup | None = None,
    ) -> None:
        self.device = device
        self.broadcast_from = broadcast_from
        self.sharding_strategy = sharding_strategy
        self.bucketing_strategy = bucketing_strategy
        self.release_original_parameters = release_original_parameters
        self.sharded_params: dict[str, Any] = {}
        self.process_group: ProcessGroup | None = process_group
        self.shared_params_name: dict[str, str] = {}
        self.move_state_dict_to_cpu = move_state_dict_to_cpu
        if self.device is None:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device("cuda", local_rank)

    def transform_module(
        self,
        thunder_model: ThunderModule,
    ):
        from thunder import compile_data as get_compile_data
        from thunder.core.module import ThunderModule

        if self.process_group is None:
            self.process_group = copy_default_process_group()
        utils.check(self.process_group is not None, lambda: "The default process group is None")
        global_rank = tdist.get_rank(group=self.process_group)
        world_size = tdist.get_world_size(group=self.process_group)
        self.global_rank = global_rank
        self.world_size = world_size

        cd = get_compile_data(thunder_model)
        # TODO: promote use_fsdp and use_ddp to public members of CompileData
        cd.use_fsdp = True
        cd.process_group_for_ddp = self.process_group
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
        # Note that .named_parameters / .named_buffers used below only return a duplicated parameter once.

        shared_names = thunder_model._get_shared_names()
        self.shared_params_name = {}

        # For materialized parameters and buffers, we move them to the target device as necessary
        # for un-materialized parameters and buffers, we set the ._thunder_device

        is_fully_materialized = True
        for n, p in thunder_model.named_parameters():
            for n2 in shared_names[n]:
                if n2 != n:
                    self.shared_params_name[n2] = n
            try:
                orig_p = thunder_model._model.get_parameter(n)
            except AttributeError:
                orig_p = None
            if p.is_meta:
                is_fully_materialized = False
                p._thunder_device = self.device
                if orig_p is not None:
                    orig_p._thunder_device = self.device
                # TODO: check if device_adjustments are still needed
                for n2 in shared_names[n]:
                    device_adjustments[n2] = self.device
            elif p.device != self.device:
                with torch.no_grad():
                    new_p = torch.nn.Parameter(p.to(device=self.device), requires_grad=p.requires_grad)
                for n2 in shared_names[n]:
                    thunder_model._overrides_parameters[n2] = new_p
                    device_adjustments[n2] = self.device

        for n, b in thunder_model.named_buffers():
            try:
                orig_b = thunder_model._model.get_buffer(n)
            except AttributeError:
                orig_b = None
            if b.is_meta:
                is_fully_materialized = False
                b._thunder_device = self.device
                if orig_b is not None:
                    orig_b._thunder_device = self.device
                # TODO: check if this is still needed
                device_adjustments[n] = self.device
            elif b.device != self.device:
                new_b = b.to(device=self.device)
                for n2 in shared_names[n]:
                    thunder_model._overrides_buffers[n2] = new_b
                    device_adjustments[n2] = self.device

        # Broadcast parameters if requested
        # (todos shared with thunder/distributed/_init__.py)
        # TODO Make these broadcast asyncs
        # TODO Perform up to two broadcasts at a time
        # See issue "Update ddp to use async broadcasts"
        # TODO "Bucket" small tensors together before broadcasting
        if self.broadcast_from is not None:
            if not is_fully_materialized:
                # Note: we could move broadcasting into its own transform coming
                #       after materialization (in thunder.distributed.fsdp) to
                #       support this, if it is useful.
                raise RuntimeError("cannot broadcast from non-materialized model")
            with torch.no_grad():
                for pn, p in chain(thunder_model.named_parameters(), thunder_model.named_buffers()):
                    tdist.broadcast(
                        p,
                        src=self.broadcast_from,
                        group=self.process_group,
                        async_op=False,
                    )

        # do the actual sharding. Note that meta tensors will give sharded meta tensors.
        for pn, p in list(thunder_model.named_parameters()):
            # we collect shapes and devices because we do not know if other transforms also change it.
            old_shape = p.shape
            p_new, _ = _shard_tensor(p, global_rank, world_size, pn, allow_padding_for_fsdp=True)
            p_new = torch.nn.Parameter(p_new.clone(), requires_grad=p.requires_grad)
            new_shape = p_new.shape
            for n2 in shared_names[pn]:
                thunder_model._overrides_parameters[n2] = p_new
                self.sharded_params[n2] = (old_shape, new_shape, getattr(p, "_thunder_device", p.device))
            if self.release_original_parameters:
                p_orig = thunder_model._model.get_parameter(pn)
                if p_orig.device.type != "meta":
                    p_meta = torch.nn.Parameter(p_orig.to(device="meta"), requires_grad=p_orig.requires_grad)
                    p_meta._thunder_device = p_orig.device
                    for n2 in shared_names[pn]:
                        submodule_name, _, base_pn = n2.rpartition(".")
                        submodule = thunder_model._model.get_submodule(submodule_name)
                        submodule.register_parameter(base_pn, p_meta)
                else:
                    p_orig._thunder_device = self.device

    def transform_state_dict_for_submodule(
        self,
        model: ThunderModule,
        submodule_name: str,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        prefix = ""
        if submodule_name:
            prefix = f"{submodule_name}."
        new_state_dict = {}
        for k, v in state_dict.items():
            full_k = prefix + k
            if full_k in self.sharded_params:
                v, _ = _shard_tensor(v, self.global_rank, self.world_size, full_k, allow_padding_for_fsdp=True)
            new_state_dict[k] = v
        return new_state_dict

    def reverse_transform_state_dict_for_submodule(
        self,
        model: ThunderModule,
        submodule_name: str,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        from thunder.executors.torchex import _all_gather_prim_impl

        for name, tensor in state_dict.items():
            fqn: str
            if submodule_name:
                fqn = f"{submodule_name}.{name}"
            else:
                fqn = name

            if fqn not in self.sharded_params:
                continue

            old_shape, *_ = self.sharded_params[fqn]
            unsharded_tensor = _all_gather_prim_impl(
                tensor,
                group=self.process_group,
                do_async=False,
                dim=None,
            ).narrow(0, 0, old_shape[0])
            if self.move_state_dict_to_cpu:
                unsharded_tensor = unsharded_tensor.cpu()
            state_dict[name] = unsharded_tensor
        return state_dict

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
                        synchronized_parameters.append(
                            dist_prims.synchronize(comp_inp_p, self.process_group, DistParallelType.FULLY_SHARDED)
                        )
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

        proxies_to_replace = {variableify(bsym.args[0]): bsym.output for bsym in new_scope}

        # See NOTE: Shared Parameters in Trace
        for param_name, base_param in self.shared_params_name.items():
            param_proxy = param_name_to_comp_trc_proxy[param_name]
            base_param_proxy = param_name_to_comp_trc_proxy[base_param]
            allgather_base_param_proxy = proxies_to_replace[variableify(base_param_proxy)]
            # Update `proxies_to_replace` so we replace all usage of `param_proxy`
            # with the output of `AllGather` on `base_param_proxy`.
            proxies_to_replace[variableify(param_proxy)] = allgather_base_param_proxy

        new_computation_trace = from_trace(computation_trace)
        for idx, bsym in enumerate(computation_trace.bound_symbols):
            if bsym.sym != prims.unpack_trivial:
                break
            new_computation_trace.bound_symbols.append(bsym.from_bsym())
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
