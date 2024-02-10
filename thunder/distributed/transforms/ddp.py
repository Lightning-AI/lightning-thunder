from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from thunder.core import dtypes
from thunder.core import devices
from thunder.core import prims
from thunder.core.prims import PrimIDs
from thunder.core.proxies import FutureTensorProxy
from thunder.core.proxies import TensorProxy
from thunder.core.pytree import tree_flatten
from thunder.core.pytree import tree_unflatten
from thunder.core.transforms import visitor_transform
from thunder.core.transforms import VISIT_TYPE
from thunder.core import utils
from thunder.distributed import get_skip_data_parallel_grad_sync
from thunder.distributed.bucketing import GradBuckets
from thunder.distributed import prims as dist_prims

if TYPE_CHECKING:
    from typing import Any
    from torch.distributed import ProcessGroup
    from thunder.core.symbol import Symbol
    from thunder.core.symbol import BoundSymbol
    from thunder.core.trace import TraceCtx
    from thunder.common import CompileData


__all__ = [
    "optimize_allreduce_in_ddp_backward",
]


def get_bsym_of_allreduce_of_grad(tensor: TensorProxy, producers: utils.ProxyDict) -> BoundSymbol:
    prod_bsym = producers[tensor]
    if prod_bsym.sym.id == dist_prims.PrimIDs.ALL_REDUCE:
        return prod_bsym
    t = prod_bsym.flat_proxy_args[0]
    utils.check_type(t, (TensorProxy, FutureTensorProxy))
    return get_bsym_of_allreduce_of_grad(t, producers)


def get_bsym_of_wait(tensor: TensorProxy, producers: utils.ProxyDict) -> BoundSymbol:
    prod_bsym = producers[tensor]
    if prod_bsym.sym.id == dist_prims.PrimIDs.WAIT:
        return prod_bsym
    t = prod_bsym.flat_proxy_args[0]
    utils.check_type(t, (TensorProxy, FutureTensorProxy))
    return get_bsym_of_wait(t, producers)


@dataclass
class MakeAllReduceInplace:
    allreduce_bsyms: dict[BoundSymbol, FutureTensorProxy]
    # note: This would be error prone, so it might make sense to turn all_reduce's
    # op, group, do_async, and skip_clone into keyword-only args

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        from dataclasses import replace
        from thunder.core.trace import get_tracectx

        if bsym in self.allreduce_bsyms:
            new_args = bsym.args[:-1] + (True,)
            new_bsym = replace(bsym, args=new_args)
            trace = get_tracectx()
            trace.scopes[-1].append(new_bsym)
            return VISIT_TYPE.REPLACE
        return VISIT_TYPE.NO_OP


def make_allreduce_of_vanilla_ddp_inplace(
    backward_trace: TraceCtx,
    producers: utils.ProxyDict,
) -> TraceCtx:
    allreduce_inputs = utils.ProxyDict()
    allreduce_bsyms = {}
    for synced_grad in filter(lambda t: isinstance(t, TensorProxy), backward_trace.bound_symbols[-1].flat_proxy_args):
        wait_bsym = get_bsym_of_wait(synced_grad, producers)
        future_tensor_proxy = wait_bsym.flat_proxy_args[0]
        utils.check_type(future_tensor_proxy, FutureTensorProxy)
        allreduce_bsym = get_bsym_of_allreduce_of_grad(future_tensor_proxy, producers)
        utils.check(allreduce_bsym not in allreduce_bsyms, lambda: f"{allreduce_bsym} is used")
        allreduce_bsyms[allreduce_bsym] = future_tensor_proxy

        orig_grad = allreduce_bsym.flat_proxy_args[0]
        utils.check(
            orig_grad not in allreduce_inputs,
            lambda: f"{orig_grad} is consumed by {allreduce_inputs[orig_grad]}",
        )
        allreduce_inputs[orig_grad] = allreduce_bsym

    visitor = MakeAllReduceInplace(allreduce_bsyms)
    bwd_trace_with_inplace_allreduce = visitor_transform(
        backward_trace, visitor, provenance="Making AllReduce in-place"
    )
    return bwd_trace_with_inplace_allreduce


@dataclass
class BatchAllReduceVisitor:
    process_group: ProcessGroup
    flat_backward_trace_output: list[Any]
    backward_trace_output_spec: Any
    gradient_buckets: GradBuckets
    prims_to_filter: set[prims.PrimIDs, dist_prims.PrimIDs]
    has_replaced_return: bool = False

    def __call__(self, bsym: BoundSymbol) -> None:
        sym: Symbol = bsym.sym

        if sym.id in self.prims_to_filter:
            return VISIT_TYPE.REPLACE

        if sym.id == prims.PrimIDs.RETURN:
            allreduced_grads = self.gradient_buckets.retrieve_allreduced_grads(self.process_group)
            new_return = tree_unflatten(
                [allreduced_grads.get(i, t) for i, t in enumerate(self.flat_backward_trace_output)],
                self.backward_trace_output_spec,
            )
            prims.python_return(new_return)
            self.has_replaced_return = True
            return VISIT_TYPE.REPLACE

        # `grads` here are pre-averaged gradients. Thus we might want to make sure sym.id is prims.PrimIDs.DIV
        grads_of_bsym = tuple(filter(lambda p: p in self.gradient_buckets.grad_to_bucket, bsym.flat_proxy_outs))
        if grads_of_bsym:
            utils.check(
                bsym.sym.id in {PrimIDs.DIV, "torch.true_divide"},
                lambda: f"This bsym's sym.id is expected to be {PrimIDs.DIV=} or 'torch.true_divide' but {bsym.sym.id=}",
            )
            utils.check(len(grads_of_bsym) == 1, lambda: f"{len(grads_of_bsym)=} is expected to be 1")
            self.gradient_buckets.tell(grads_of_bsym[0], self.process_group)
        return VISIT_TYPE.INSERT_AFTER


def optimize_allreduce_in_ddp_backward(
    backward_trace: TraceCtx,
    compile_data: CompileData,
) -> TraceCtx:
    """Reduce all_reduce of the given ``backward_trace`` with gradient bucketing.

    This function collects pre-averaged gradient tensors, replace all existing ``dist_prims.all_reduce``
    and ``dist_prims.wait`` with ``dist_prims.update_bucket_view``, ``dist_prims.pack``,
    ``dist_prims.all_reduce``, ` dist_prims.wait` , and ` dist_prims.unpack` .

    In the first iteration, `dist_prims.pack` creates buckets of greater than or equal to
    ``bucket_size_in_mb`` that are bunching up one or more gradient tensors.
    ``dist_prims.unpack`` writes out allreduce'd gradients to original gradient tensors.
    ``dist_prims.update_bucket_view`` copies pre-averaged gradient tensors to the corresponding
    view of bucket.

    If ``bucket_size_in_mb`` is 0, then this function replaces the existing allreduce's with in-place allreduce's.

    Args:
        backward_trace:
        compile_data:

    Returns:
        :class:`TraceCtx`
    """

    if get_skip_data_parallel_grad_sync():
        return backward_trace

    # Map from preaveraged grad to index in trace.output
    producers = utils.producers(backward_trace)
    preaveraged_grads: list[TensorProxy] = []
    preaveraged_to_index = utils.ProxyDict()
    for i, t in enumerate(tree_flatten(backward_trace.output)[0]):
        if isinstance(t, TensorProxy):
            bsym_of_allreduce: BoundSymbol = get_bsym_of_allreduce_of_grad(t, producers)
            preaveraged_grad_tensor_proxy: TensorProxy = bsym_of_allreduce.flat_proxy_args[0]
            preaveraged_to_index[preaveraged_grad_tensor_proxy] = i
            preaveraged_grads.append(preaveraged_grad_tensor_proxy)

    if (bucket_size_in_mb := getattr(compile_data.fn, "bucket_size_in_mb", 25)) <= 0:
        return make_allreduce_of_vanilla_ddp_inplace(backward_trace, producers)

    gradients_of_same_dtype_and_device: dict[tuple[dtypes.dtype, devices.Device], list[TensorProxy]] = defaultdict(list)
    for grad in preaveraged_grads:
        key = (grad.dtype, grad.device)
        gradients_of_same_dtype_and_device[key].append(grad)
    gradient_buckets = GradBuckets.build(
        gradients_of_same_dtype_and_device=gradients_of_same_dtype_and_device,
        gradient_to_index=preaveraged_to_index,
        bucket_cap_in_mb=bucket_size_in_mb,
    )

    flat_backward_trace_output, backward_trace_output_spec = tree_flatten(backward_trace.output)
    visit_callable = BatchAllReduceVisitor(
        process_group=compile_data.process_group_for_ddp,
        flat_backward_trace_output=flat_backward_trace_output,
        backward_trace_output_spec=backward_trace_output_spec,
        gradient_buckets=gradient_buckets,
        prims_to_filter={dist_prims.PrimIDs.ALL_REDUCE, dist_prims.PrimIDs.WAIT},
    )
    updated_bwd_trace = visitor_transform(
        backward_trace,
        visit_callable,
        provenance="Batching all_reduce calls",
    )
    utils.check(visit_callable.has_replaced_return, lambda: "")
    return updated_bwd_trace
