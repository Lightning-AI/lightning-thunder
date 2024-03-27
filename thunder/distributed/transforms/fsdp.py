from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from torch.distributed import ProcessGroup

from thunder.core import prims
from thunder.core import utils
from thunder.core.prims import PrimIDs
from thunder.core.proxies import variableify
from thunder.core.proxies import Proxy
from thunder.core.proxies import TensorProxy
from thunder.core.proxies import FutureTensorProxy
from thunder.core.pytree import tree_flatten
from thunder.core.pytree import tree_unflatten
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import VariableInterface
from thunder.core.trace import from_trace
from thunder.core.transforms import VISIT_TYPE
from thunder.core.transforms import visitor_transform
from thunder.distributed import FSDPBucketingStrategy
from thunder.distributed import FSDPType
from thunder.distributed import get_extract_bucket_name_from_tensor_proxy
from thunder.distributed.bucketing import FSDPBackwardBucket
from thunder.distributed.bucketing import FSDPForwardBucket
from thunder.distributed import prims as dist_prims
from thunder.executors.torchex import all_gather_prim_impl
from thunder.executors.torchex import pack_prim_impl
from thunder.executors.torchex import pack_for_fsdp_prim_impl
from thunder.executors.torchex import reduce_scatter_prim_impl
from thunder.executors.torchex import unpack_prim_impl
from thunder.executors.torchex import unpack_for_fsdp_prim_impl
from thunder.executors.torchex import wait_prim_impl

if TYPE_CHECKING:
    from thunder import CompileData
    from thunder.core.trace import TraceCtx
    from thunder.distributed.bucketing import Bucket


__all__ = [
    "FSDPCommBucketing",
]


_DENY_LIST: set[dist_prims.PrimIDs | str] = {
    dist_prims.PrimIDs.PACK,
    dist_prims.PrimIDs.UNPACK,
    dist_prims.PrimIDs.PACK_FOR_FSDP,
    dist_prims.PrimIDs.UNPACK_FOR_FSDP,
    pack_prim_impl.id,
    pack_for_fsdp_prim_impl.id,
    unpack_prim_impl.id,
    unpack_for_fsdp_prim_impl.id,
}

_ALL_GATHER_SYM_IDS: set[dist_prims.PrimIDs | str] = {dist_prims.PrimIDs.ALL_GATHER, all_gather_prim_impl.id}
_REDUCE_SCATTER_SYM_IDS: set[dist_prims.PrimIDs | str] = {
    dist_prims.PrimIDs.REDUCE_SCATTER,
    reduce_scatter_prim_impl.id,
}


def create_map_of_before_after_comm(
    trace: TraceCtx,
    *,
    comm_ids: set[prims.PrimIDs | dist_prims.PrimIDs | str],
    return_proxy_lists: bool,
) -> tuple[utils.ProxyDict, tuple[list[TensorProxy], list[TensorProxy]]] | utils.ProxyDict:
    """Create a map from sharded parameter to unsharded parameter from a trace without bucketing.

    Args:
        trace:
        comm_ids:

    Returns:
        utils.ProxyDict:

    """
    params_before_comm: list[TensorProxy] = []
    params_after_comm: list[TensorProxy] = []

    producers, consumers = utils.producers_and_consumers(trace)
    bsym: BoundSymbol
    for bsym in trace.bound_symbols:
        if bsym.sym.id in comm_ids:
            sharded_param = bsym.flat_proxy_args[0]
            if sharded_param in producers:
                bsym_of_param_prod = producers[sharded_param]
                utils.check(
                    bsym_of_param_prod.sym.id not in _DENY_LIST,
                    lambda: f"{bsym.sym} found in the given trace. {_DENY_LIST} are not expected",
                )

            params_before_comm.append(sharded_param)

            future = bsym.flat_proxy_outs[0]
            wait_bsym: BoundSymbol = consumers[future][0]
            utils.check(
                isinstance(future, FutureTensorProxy)
                and wait_bsym.sym.id in {dist_prims.PrimIDs.WAIT, wait_prim_impl.id},
                lambda: f"Unexpected bsym of {wait_bsym.sym}",
            )
            unsharded_param = wait_bsym.flat_proxy_outs[0]
            params_after_comm.append(unsharded_param)
    map_from_sharded_to_unsharded = utils.ProxyDict()
    for s, u in zip(params_before_comm, params_after_comm):
        map_from_sharded_to_unsharded[s] = u
    if return_proxy_lists:
        return map_from_sharded_to_unsharded, (params_before_comm, params_after_comm)
    else:
        return map_from_sharded_to_unsharded


@dataclass
class FSDPCommBucketingTransformVisitor:
    group: ProcessGroup
    original_params: Sequence[TensorProxy]
    param_to_bucket: utils.ProxyDict
    param_to_unsharded: utils.ProxyDict
    comm_to_bucket: dist_prims.PrimIDs

    def __post_init__(self) -> None:
        self.params = utils.ProxyDict()
        for p in self.original_params:
            self.params[p] = None
        self.param_before_after = utils.ProxyDict()
        self.future_before_after = utils.ProxyDict()
        self.future_to_orig = utils.ProxyDict()
        self.bucket_to_future: dict[FSDPForwardBucket, FutureTensorProxy] = {}
        self.future_to_bucket = utils.ProxyDict()
        self.future_to_unpacked = utils.ProxyDict()
        self.future_of_wait_to_ignore = utils.ProxyDict()

        self.swap_map: dict[VariableInterface, TensorProxy] = {}

        match self.comm_to_bucket:
            case dist_prims.PrimIDs.ALL_GATHER:
                self.comm_ids: set[dist_prims.PrimIDs | str] = _ALL_GATHER_SYM_IDS
                self.pack_unpack_mode = "gather"
                self.comm = partial(dist_prims.all_gather, group=self.group, do_async=True)
            case dist_prims.PrimIDs.REDUCE_SCATTER:
                self.comm_ids: set[dist_prims.PrimIDs | str] = _REDUCE_SCATTER_SYM_IDS
                self.pack_unpack_mode = "scatter"
                self.comm = partial(
                    dist_prims.reduce_scatter, op=dist_prims.DistributedReduceOps.SUM, group=self.group, do_async=True
                )
            case _:
                utils.check(
                    False,
                    lambda: f"Invalid {self.comm_to_bucket}, {(dist_prims.PrimIDs.ALL_GATHER, dist_prims.PrimIDs.REDUCE_SCATTER)} are supported",
                )

    @property
    def world_size(self) -> int:
        return self.group.size()

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        if bsym.sym.id in {
            PrimIDs.UNPACK_TRIVIAL,
            PrimIDs.UNPACK_SEQUENCE,
            PrimIDs.UNPACK_KEY,
            PrimIDs.UNPACK_EMPTY_DICT,
        }:
            return VISIT_TYPE.NO_OP

        def maybe_swap_proxies_of_bsym_and_update_swap_map(bsym: BoundSymbol) -> bool:
            if not any(variableify(p) in self.swap_map for p in bsym.flat_proxy_args):
                return False
            new_bsym = bsym.from_bsym_swap_proxies(self.swap_map)
            updated_output = new_bsym.sym(*new_bsym.args, **new_bsym.kwargs)
            if updated_output is not None:
                updated_output, _ = tree_flatten(updated_output)
                orig_outs = bsym.flat_outs
                orig_proxy_outs, orig_proxy_indices = [], []
                for i, o in enumerate(orig_outs):
                    if not isinstance(o, Proxy):
                        continue
                    orig_proxy_outs.append(o)
                    orig_proxy_indices.append(i)
                updated_proxy_output = tuple(updated_output[i] for i in orig_proxy_indices)
                utils.check(
                    len(orig_proxy_outs) == len(updated_proxy_output),
                    lambda: f"{orig_proxy_outs=}, {updated_proxy_output=}, {bsym=}, {new_bsym}",
                )
                for orig_o, o in zip(orig_proxy_outs, updated_proxy_output):
                    if orig_o.name != o.name:
                        self.swap_map[variableify(orig_o)] = o
            return True

        if bsym.sym.id in self.comm_ids:
            param = bsym.flat_proxy_args[0]
            utils.check(
                param in self.params,
                lambda: f"{variableify(param)} not found in param set: {(variableify(p) for p in self.original_params)}",
            )
            if param not in self.param_to_bucket:
                # This path is highly likely to be backward reduce-scatter bucketing:
                # when a param does not require grad, a trace could still have reduce-scatter
                # and wait in its trace while the grad in the return statement is already
                # replaced with `None`.
                return VISIT_TYPE.NO_OP
            bucket = self.param_to_bucket[param]
            # TODO(crcrpar): Think of remove match-case below.
            match self.comm_to_bucket:
                case dist_prims.PrimIDs.ALL_GATHER:
                    if bucket not in self.bucket_to_future:
                        buffer = bucket.pack(world_size=self.world_size)
                        future = self.comm(buffer)
                        bucket.set_future(future, param)
                        self.future_to_bucket[future] = bucket
                        self.bucket_to_future[bucket] = future

                    future = bucket.future
                    self.future_before_after[bsym.flat_proxy_outs[0]] = future
                case dist_prims.PrimIDs.REDUCE_SCATTER:
                    bucket.tell(param)
                    orig_future = bsym.flat_proxy_outs[0]
                    if bucket.recieved_all_tensors():
                        buffer = bucket.pack(world_size=self.world_size)
                        future = self.comm(buffer)
                        bucket.set_future(future, param)
                        self.future_to_bucket[future] = bucket
                        self.bucket_to_future[bucket] = future

                        future = bucket.future
                        self.future_before_after[orig_future] = future
                    else:
                        self.future_of_wait_to_ignore[orig_future] = None
        elif bsym.sym.id in {dist_prims.PrimIDs.WAIT, wait_prim_impl.id}:
            orig_future = bsym.flat_proxy_args[0]
            # backward trace of FSDPType.ZERO3 has waits for ReduceScatter's.
            if orig_future in self.future_before_after:
                future = self.future_before_after[orig_future]
                if future not in self.future_to_unpacked:
                    bucket: FSDPForwardBucket = self.future_to_bucket[future]
                    unsharded_buffer = dist_prims.wait(future)
                    sharded_params = bucket.orig_tensors
                    unsharded_params = dist_prims.unpack_for_fsdp(
                        unsharded_buffer,
                        sharded_params,
                        self.world_size,
                        self.pack_unpack_mode,
                    )
                    self.future_to_unpacked[future] = unsharded_params
                    orig_unsharded_params = [self.param_to_unsharded[p] for p in sharded_params]
                    for before, after in zip(orig_unsharded_params, unsharded_params):
                        self.swap_map[variableify(before)] = after
            else:
                if orig_future in self.future_of_wait_to_ignore:
                    pass
                else:
                    bsym_updated = maybe_swap_proxies_of_bsym_and_update_swap_map(bsym)
                    if not bsym_updated:
                        return VISIT_TYPE.NO_OP
        else:
            if not (bsym_updated := maybe_swap_proxies_of_bsym_and_update_swap_map(bsym)):
                return VISIT_TYPE.NO_OP

        return VISIT_TYPE.REPLACE


def get_bsym_of_reduce_scatter(
    sharded_grad_tensor_proxy: TensorProxy,
    producers: utils.ProxyDict,
) -> BoundSymbol:
    bsym_of_wait = producers[sharded_grad_tensor_proxy]
    utils.check(
        bsym_of_wait.sym.id == dist_prims.PrimIDs.WAIT,
        lambda: f"Expected {dist_prims.PrimIDs.WAIT}, but {bsym_of_wait.sym.id=}",
    )

    future_tensor_proxy = bsym_of_wait.flat_proxy_args[0]
    utils.check_type(future_tensor_proxy, (FutureTensorProxy,))

    bsym_of_reduce_scatter = producers[future_tensor_proxy]
    utils.check(
        bsym_of_reduce_scatter.sym.id == dist_prims.PrimIDs.REDUCE_SCATTER,
        lambda: (
            f"{sharded_grad_tensor_proxy}'s gradparent producer bsym is expected to have "
            f"{dist_prims.PrimIDs.REDUCE_SCATTER=} as its sym.id but {bsym_of_reduce_scatter.sym.id=}"
        ),
    )
    return bsym_of_reduce_scatter


def create_map_from_unsharded_grad_to_bucket(
    index_in_flat_args_to_param_and_bucket: dict[int, tuple[TensorProxy, Bucket]],
    sharded_grad_tensor_proxies: Sequence[TensorProxy],
    producers: utils.ProxyDict,
    group: ProcessGroup,
) -> utils.ProxyDict:
    clusters: dict[str, tuple[list[TensorProxy], list[int]]] = {}
    for index, sharded_grad_tensor_proxy in enumerate(sharded_grad_tensor_proxies):
        arg_of_index: TensorProxy | None = index_in_flat_args_to_param_and_bucket.get(index, (None, None))[0]
        if not isinstance(sharded_grad_tensor_proxy, TensorProxy):
            utils.check(
                arg_of_index is None or not arg_of_index.requires_grad,
                lambda: (
                    f"{index}-th argument of forward is {arg_of_index = } but "
                    f"{index}-th output of backward trace is {sharded_grad_tensor_proxy = }"
                ),
            )
            continue
        utils.check(
            index in index_in_flat_args_to_param_and_bucket,
            lambda: (
                f"Invalid {index=}, {sharded_grad_tensor_proxies}, "
                f"{tuple((k, v[0].name) for k, v in index_in_flat_args_to_param_and_bucket.items())=}"
            ),
        )
        if not arg_of_index.requires_grad:
            continue
        _, bucket = index_in_flat_args_to_param_and_bucket[index]
        new_bucket_name = f"bwd-{bucket.name}"
        bsym_of_reduce_scatter = get_bsym_of_reduce_scatter(sharded_grad_tensor_proxy, producers)
        unsharded_grad: TensorProxy = bsym_of_reduce_scatter.flat_proxy_args[0]
        utils.check(
            unsharded_grad.numel == sharded_grad_tensor_proxy.numel * group.size(),
            lambda: f"{unsharded_grad.shape=}, {sharded_grad_tensor_proxy.shape=}",
        )
        if new_bucket_name not in clusters:
            clusters[new_bucket_name] = ([unsharded_grad], [index])
        else:
            clusters[new_bucket_name][0].append(unsharded_grad)
            clusters[new_bucket_name][1].append(index)

    n_buckets = 0
    unsharded_grad_to_bucket = utils.ProxyDict()
    for bucket_name, (unsharded_grads, indices) in clusters.items():
        bucket = FSDPBackwardBucket(n_buckets, unsharded_grads, indices, bucket_name)
        n_buckets += 1
        for g in unsharded_grads:
            unsharded_grad_to_bucket[g] = bucket
    return unsharded_grad_to_bucket


def check_num_comm_and_wait(
    trace: TraceCtx,
    comm_ids: set[dist_prims.PrimIDs | str],
) -> None:
    wait_ids = (dist_prims.PrimIDs.WAIT, wait_prim_impl.id)

    bsym: BoundSymbol
    comm_bsyms: list[BoundSymbol] = []
    wait_bsyms: list[BoundSymbol] = []

    for bsym in trace.bound_symbols:
        if bsym.sym.id in comm_ids:
            comm_bsyms.append(bsym)
        elif bsym.sym.id in wait_ids:
            wait_bsyms.append(bsym)
        else:
            pass

    if len(comm_bsyms) != len(wait_bsyms):
        produced_futures: set[VariableInterface] = {variableify(bsym.flat_proxy_outs[0]) for bsym in comm_bsyms}
        consumed_futures: set[VariableInterface] = {variableify(bsym.flat_proxy_outs[0]) for bsym in wait_bsyms}
        msg = f"There are {len(produced_futures)} comm bsyms but {len(consumed_futures)} wait bsyms. {produced_futures & consumed_futures} are okay."
        if len(produced_futures) > len(consumed_futures):
            msg += f" {produced_futures - consumed_futures} do not have wait"
        else:
            msg += f" {consumed_futures - produced_futures} do not have comm"
        utils.check(len(produced_futures) == len(consumed_futures), lambda: msg)


class FSDPCommBucketing:
    """Apply communication bucketing to FSDP traces.

    This class is in charge of introducing bucketing into the FSDP traces.
    A given forward trace will be updated so that it has fewer ``AllGather``'s by
    concatenating sharded parameters beforehand and slicing and reshaping unsharded concatenated parameters afterward.
    The backward trace, the counterpart of the forward, will be updated so that it has fewer ``ReduceScatter``'s.
    ``AllGather``s are also updated if ``sharding_strategy`` is ``FSDPType.ZERO3``.

    """

    def __init__(
        self,
        compile_data: CompileData,
    ) -> None:
        self.compile_data = compile_data
        utils.check(
            hasattr(compile_data.fn, "process_group_for_ddp")
            and hasattr(compile_data.fn, "bucketing_strategy")
            and hasattr(compile_data.fn, "sharding_strategy"),
            lambda: f"Given module does not seem to have all the attributes of `process_group_for_ddp`, `bucketing_strategy`, and `sharding_strategy`, {hasattr(compile_data.fn, 'bucketing_strategy')=}, {hasattr(compile_data.fn, 'sharding_strategy')=}",
        )
        self.bucketing_strategy: FSDPBucketingStrategy = compile_data.fn.bucketing_strategy
        self.apply_bucketing = self.bucketing_strategy != FSDPBucketingStrategy.NONE
        self.bucket_naming_func = get_extract_bucket_name_from_tensor_proxy(self.bucketing_strategy)
        self.group: ProcessGroup = compile_data.fn.process_group_for_ddp

        self.requires_bwd_bucketing_allgather = compile_data.fn.sharding_strategy == FSDPType.ZERO3

    def update_name_set(self, backward_trace: TraceCtx) -> TraceCtx:
        if not self.apply_bucketing:
            return
        utils.check(
            hasattr(self, "fsdp_fwd_trace"),
            lambda: "This method must be called after :func:`FSDPCommsOptimizer.apply_bucketing_to_forward_trace`",
        )
        backward_trace.names.update(self.fsdp_fwd_trace.names)

    def _collect_sharded_parameters(self, fwd_trace: TraceCtx) -> list[TensorProxy]:
        fwd_trace_flat_args, _ = tree_flatten((fwd_trace.args, fwd_trace.kwargs))
        return fwd_trace_flat_args

    def apply_bucketing_to_forward_trace(self, fwd_trace: TraceCtx, bwd_trace_names: set[str]) -> TraceCtx:
        """Optimize collective comms in fsdp with bucketing.

        This function is no-op if you pass :obj:`BucketingStrategy.NONE` as kwarg of ``sharding_strategy`` to :func:`thunder.distributed.fsdp`.
        With :obj:`BucketingStrategy.LAYER`, buckets will be created per :class:`torch.nn.Module` such as
        :class:`torch.nn.Linear`, and :class:`torch.nn.LayerNorm`.
        Use :obj:`BucketingStrategy.BLOCK` to assign a bucket to one :class:`torch.nn.Transformer`.

        This function (assuming bucketing strategy is not ``NONE``) uses :func:`thunder.core.transforms.visitor_transform` to
            - insert :class:`~thunder.core.symbol.BoundSymbol`s of :func:`~thunder.distributed.prims.pack`.
            - replace existing :func:`~thunder.distributed.prims.all_gather`s with new ones whose first argument is a bucket produced by inserted :func:`~thunder.distributed.prims.pack`.
            - replace existing :func:`~thunder.distributed.prims.wait` with new ones whose first argument is :class:`~thunder.core.proxies.FutureTensorProxy` of a bucket.
            - modify args and kwargs of consumer :class:`~thunder.core.symbol.BoundSymbol`s of :class:`~thunder.core.proxies.TensorProxy`s representing unsharded parameters.

        Args:
            fwdp_fwd_trace:
            compile_data

        Returns:
            - :class:`TraceCtx`
            - dict[int, tuple[:class:`TensorProxy`, :class:`Bucket`]]: This is for the bucketing in backward.
        """
        if not self.apply_bucketing:
            return fwd_trace
        fsdp_fwd_trace = from_trace(fwd_trace)
        fsdp_fwd_trace.bound_symbols = fwd_trace.bound_symbols
        fsdp_fwd_trace.names.update(bwd_trace_names)
        trace_flat_args = self._collect_sharded_parameters(fsdp_fwd_trace)
        arg_to_index_in_flat_args = utils.ProxyDict()
        index_in_flat_args_to_param_and_bucket: dict[int, tuple[TensorProxy, Bucket]] = {}
        for i, a in enumerate(trace_flat_args):
            if isinstance(a, TensorProxy):
                arg_to_index_in_flat_args[a] = i

        collective_comm_bsyms: tuple[BoundSymbol, ...] = tuple(
            filter(
                lambda bsym: bsym.sym.id == dist_prims.PrimIDs.ALL_GATHER
                and any(arg in arg_to_index_in_flat_args for arg in bsym.flat_proxy_args),
                fsdp_fwd_trace.bound_symbols,
            )
        )
        sharded_parameters: tuple[TensorProxy, ...] = tuple(bsym.flat_proxy_args[0] for bsym in collective_comm_bsyms)

        bucket_name_to_sharded_params: dict[str, list[TensorProxy]] = defaultdict(list)
        for param in sharded_parameters:
            bucket_name = self.bucket_naming_func(param)
            bucket_name_to_sharded_params[bucket_name].append(param)
        n_buckets = 0
        original_params = []
        param_to_bucket = utils.ProxyDict()  # sharded param -> FSDPForwardBucket
        for bucket_name in bucket_name_to_sharded_params:
            sharded_params = tuple(bucket_name_to_sharded_params[bucket_name])
            bucket = FSDPForwardBucket(n_buckets, sharded_params, [], bucket_name)
            for p in sharded_params:
                param_to_bucket[p] = bucket
                index_in_flat_args_to_param_and_bucket[arg_to_index_in_flat_args[p]] = (p, bucket)
                original_params.append(p)

        self.visit_for_fsdp_forward = FSDPCommBucketingTransformVisitor(
            group=self.group,
            original_params=original_params,
            param_to_bucket=param_to_bucket,
            param_to_unsharded=create_map_of_before_after_comm(
                fsdp_fwd_trace,
                comm_ids=_ALL_GATHER_SYM_IDS,
                return_proxy_lists=False,
            ),
            comm_to_bucket=dist_prims.PrimIDs.ALL_GATHER,
        )

        new_trace = visitor_transform(
            fsdp_fwd_trace,
            self.visit_for_fsdp_forward,
            provenance="Merge Collective Comms",
        )
        check_num_comm_and_wait(new_trace, _ALL_GATHER_SYM_IDS)
        self.index_in_flat_args_to_param_and_bucket = index_in_flat_args_to_param_and_bucket

        self.orig_fsdp_fwd_trace = fsdp_fwd_trace
        self.fsdp_fwd_trace = new_trace

        return new_trace

    def _apply_bucketing_to_backward_reduce_scatter(self, fsdp_bwd_trace: TraceCtx) -> TraceCtx:
        producers = utils.producers(fsdp_bwd_trace)

        unsharded_grad_to_bucket = create_map_from_unsharded_grad_to_bucket(
            self.index_in_flat_args_to_param_and_bucket,
            tree_flatten(fsdp_bwd_trace.output)[0],
            producers,
            self.group,
        )

        unsharded_grad_to_sharded, (unsharded_grads, sharded_grads) = create_map_of_before_after_comm(
            fsdp_bwd_trace,
            comm_ids=_REDUCE_SCATTER_SYM_IDS,
            return_proxy_lists=True,
        )

        visitor = FSDPCommBucketingTransformVisitor(
            group=self.group,
            original_params=unsharded_grads,
            param_to_bucket=unsharded_grad_to_bucket,
            param_to_unsharded=unsharded_grad_to_sharded,
            comm_to_bucket=dist_prims.PrimIDs.REDUCE_SCATTER,
        )

        bwd_trace = visitor_transform(fsdp_bwd_trace, visitor, provenance="Merge Collective Comms")
        check_num_comm_and_wait(bwd_trace, _ALL_GATHER_SYM_IDS | _REDUCE_SCATTER_SYM_IDS)
        return bwd_trace

    def _apply_bucketing_to_backward_all_gather(self, fsdp_bwd_trace: TraceCtx) -> TraceCtx:
        fwd_trace_flat_args = self._collect_sharded_parameters(self.fsdp_fwd_trace)
        arg_to_index_in_flat_args = utils.ProxyDict()
        for i, a in enumerate(fwd_trace_flat_args):
            if isinstance(a, TensorProxy):
                arg_to_index_in_flat_args[a] = i

        target_allgather_bsyms: tuple[BoundSymbol, ...] = tuple(
            filter(
                lambda bsym: bsym.sym.id == dist_prims.PrimIDs.ALL_GATHER,
                fsdp_bwd_trace.bound_symbols,
            )
        )
        target_wait_bsyms: set[BoundSymbol] = set()

        _, consumers = utils.producers_and_consumers(fsdp_bwd_trace)

        orig_sharded_params: list[TensorProxy] = []
        orig_futures: list[FutureTensorProxy] = []
        orig_future_to_unsharded_param = utils.ProxyDict()
        orig_sharded_to_unsharded = utils.ProxyDict()
        param_unshard_bsyms: list[BoundSymbol] = []
        for bsym in target_allgather_bsyms:
            t = bsym.flat_proxy_args[0]
            utils.check_type(t, TensorProxy)
            # make sure to avoid non parameter tensors being bucketed.
            if t not in arg_to_index_in_flat_args:
                continue
            f = bsym.flat_proxy_outs[0]
            utils.check_type(f, FutureTensorProxy)
            orig_sharded_params.append(t)
            orig_futures.append(f)
            param_unshard_bsyms.append(bsym)

            consumer_bsyms_of_future: BoundSymbol = consumers[f]
            wait_bsym: BoundSymbol
            for bsym in consumer_bsyms_of_future:
                if bsym.sym.id == dist_prims.PrimIDs.WAIT:
                    wait_bsym = bsym
                    break
            target_wait_bsyms.add(wait_bsym)
            unsharded_param = wait_bsym.flat_proxy_outs[0]
            utils.check_type(unsharded_param, TensorProxy)
            orig_future_to_unsharded_param[f] = unsharded_param
            orig_sharded_to_unsharded[t] = unsharded_param

        utils.check(
            len(orig_sharded_params) < len(arg_to_index_in_flat_args._dict),
            lambda: f"{orig_sharded_params = }, {tuple(arg_to_index_in_flat_args._dict.keys()) = }",
        )

        bucket_name_to_sharded_params: dict[str, list[TensorProxy]] = defaultdict(list)
        for param in orig_sharded_params:
            bucket_name = self.bucket_naming_func(param)
            bucket_name_to_sharded_params[bucket_name].append(param)

        n_buckets = 0
        param_to_bucket = utils.ProxyDict()  # sharded param -> FSDPForwardBucket
        for bucket_name in bucket_name_to_sharded_params:
            sharded_params = tuple(bucket_name_to_sharded_params[bucket_name])
            bucket = FSDPForwardBucket(n_buckets, sharded_params, [], bucket_name)
            for p in sharded_params:
                param_to_bucket[p] = bucket
        visit = FSDPCommBucketingTransformVisitor(
            group=self.group,
            original_params=orig_sharded_params,
            param_to_bucket=param_to_bucket,
            param_to_unsharded=create_map_of_before_after_comm(
                trace=fsdp_bwd_trace,
                comm_ids=_ALL_GATHER_SYM_IDS,
                return_proxy_lists=False,
            ),
            comm_to_bucket=dist_prims.PrimIDs.ALL_GATHER,
        )
        updated_bwd_trace = visitor_transform(
            fsdp_bwd_trace,
            visit,
            provenance="AllGather Bucketing",
        )
        check_num_comm_and_wait(updated_bwd_trace, _ALL_GATHER_SYM_IDS | _REDUCE_SCATTER_SYM_IDS)
        return updated_bwd_trace

    def apply_bucketing_to_backward_trace(self, fsdp_bwd_trace: TraceCtx) -> TraceCtx:
        """Apply bucketing to reduce_scatter in fsdp bwd trace.

        1. Collect unsharded gradient tensor proxies and create buckets for them based on forward's buckets' name.
        2. Copy unsharded gradients into their buckets
        3. Call async redunce-scatter on buckets as they get ready
        4. Call wait on future tensors of buckets

        If ``sharding_strategy`` is ``FSDPType.ZERO3``, this also applies bucketing to AllGathers.

        Args:
            fsdp_bwd_trace:
            compile_data:
            index_in_flat_args_to_bucket:

        Returns:
            - :class:`TraceCtx`
        """

        if not self.apply_bucketing:
            return fsdp_bwd_trace

        # Apply bucketing to parameter unsharding (= AllGather)
        if self.requires_bwd_bucketing_allgather:
            fsdp_bwd_trace = self._apply_bucketing_to_backward_all_gather(fsdp_bwd_trace)

        # Apply bucketing to gradient sharding (= ReduceScatter)
        return self._apply_bucketing_to_backward_reduce_scatter(fsdp_bwd_trace)
