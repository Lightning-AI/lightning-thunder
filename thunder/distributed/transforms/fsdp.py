from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch.distributed import ProcessGroup

from thunder.core import prims
from thunder.core import utils
from thunder.core.baseutils import ProxyInterface
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
from thunder.distributed import get_extract_bucket_name_from_tensor_proxy
from thunder.distributed.bucketing import FSDPBackwardBucket
from thunder.distributed.bucketing import FSDPForwardBucket
from thunder.distributed import prims as dist_prims

if TYPE_CHECKING:
    from thunder import CompileData
    from thunder.core.trace import TraceCtx
    from thunder.core.symbol import BoundSymbol
    from thunder.core.symbol import BoundSymbolRHS
    from thunder.distributed.bucketing import Bucket


__all__ = [
    "FSDPCommBucketing",
]


def _get_bsyms_of_wait_and_math(
    param: TensorProxy,
    consumers: utils.ProxyDict,
) -> tuple[BoundSymbol, BoundSymbol]:
    params_consumers = consumers[param]
    utils.check(
        [bsym for bsym in params_consumers if bsym.sym.id == dist_prims.PrimIDs.ALL_GATHER],
        lambda: f"{params_consumers=}",
    )
    orig_allgather_bsym: BoundSymbol = params_consumers[0]
    orig_future = orig_allgather_bsym.flat_proxy_outs[0]
    utils.check_type(orig_future, FutureTensorProxy)
    orig_wait_bsym: BoundSymbol = consumers[orig_future][0]
    utils.check(orig_wait_bsym.sym.id == dist_prims.PrimIDs.WAIT, lambda: f"{orig_wait_bsym=}")
    orig_unsharded_param = orig_wait_bsym.flat_proxy_outs[0]
    orig_math_bsym: BoundSymbol = consumers[orig_unsharded_param][0]
    return (orig_wait_bsym, orig_math_bsym)


@dataclass
class FSDPFwdTraceVisitor:
    group: ProcessGroup
    collective_comm_bsyms: tuple[BoundSymbol, ...]
    producers: utils.ProxyDict
    consumers: utils.ProxyDict
    param_to_bucket: utils.ProxyDict

    def __post_init__(self) -> None:
        self.bucket_to_sharded_param_and_consumers: dict[FSDPForwardBucket, utils.ProxyDict] = defaultdict(
            utils.ProxyDict
        )
        self.bucket_to_unsharded_param: dict[FSDPForwardBucket, TensorProxy] = {}
        self.bsyms_to_check: set[BoundSymbolRHS] = set()
        self.future_to_bucket = utils.ProxyDict()
        self.bucket_to_unsharded_params: dict[FSDPForwardBucket, tuple[TensorProxy, ...]] = {}
        # The following two variables of ``original_to_updated`` and ``swap_map`` is for trace-to-trace transformation.
        # original here means that proxies in pre-transform trace.
        self.original_to_updated = utils.ProxyDict()
        self.swap_map: dict[VariableInterface, TensorProxy] = {}

    def __call__(self, bsym: BoundSymbol) -> VISIT_TYPE:
        if bsym.sym.id in {
            PrimIDs.UNPACK_TRIVIAL,
            PrimIDs.UNPACK_SEQUENCE,
            PrimIDs.UNPACK_KEY,
            PrimIDs.UNPACK_EMPTY_DICT,
        }:
            return VISIT_TYPE.NO_OP

        if bsym in self.collective_comm_bsyms:
            param = bsym.flat_proxy_args[0]
            utils.check(param in self.param_to_bucket, lambda: f"{param=} not found in `param_to_name_and_bucket`")
            bucket: FSDPForwardBucket = self.param_to_bucket[param]
            if not bucket.has_future():
                bucket_storage = bucket.pack(world_size=self.group.size())
                bucket.set_future(dist_prims.all_gather(bucket_storage, self.group, True), param)
                future = bucket.future
                self.future_to_bucket[future] = bucket

            self.bucket_to_sharded_param_and_consumers[bucket][param] = _get_bsyms_of_wait_and_math(
                param, self.consumers
            )
            self.bsyms_to_check.update(
                {bsym.rhs() for bsym in self.bucket_to_sharded_param_and_consumers[bucket][param]}
            )

            self.original_to_updated[param] = bucket.storage
            self.original_to_updated[bsym.flat_proxy_outs[0]] = bucket.future

            return VISIT_TYPE.REPLACE

        if bsym.sym.id == dist_prims.PrimIDs.WAIT:
            orig_future = bsym.flat_proxy_args[0]
            future = self.original_to_updated[orig_future]
            bucket = self.future_to_bucket[future]
            if bucket not in self.bucket_to_unsharded_param:
                unsharded = dist_prims.wait(bucket.future)
                self.bucket_to_unsharded_param[bucket] = unsharded

                unsharded_params = dist_prims.unpack_for_fsdp(
                    unsharded,
                    bucket.orig_tensors,
                    self.group.size(),
                    "gather",
                )
                self.bucket_to_unsharded_params[bucket] = tuple(unsharded_params)
            unsharded = self.bucket_to_unsharded_param[bucket]
            unsharded_params = self.bucket_to_unsharded_params[bucket]

            orig_unsharded_param = bsym.flat_proxy_outs[0]
            orig_allgather_bsym = self.producers[orig_future]
            utils.check(orig_allgather_bsym.sym.id == dist_prims.PrimIDs.ALL_GATHER, lambda: f"{orig_allgather_bsym=}")
            orig_sharded_param = orig_allgather_bsym.flat_proxy_args[0]
            utils.check(
                orig_sharded_param.name in bucket.tensor_names,
                lambda: f"Queried {orig_sharded_param.name=}, but {bucket.tensor_names=}",
            )
            unsharded_param = unsharded_params[bucket.tensor_names.index(orig_sharded_param.name)]
            utils.check_same_device(orig_unsharded_param, unsharded_param)
            utils.check_same_dtype(orig_unsharded_param, unsharded_param)
            utils.check_same_shape(orig_unsharded_param, unsharded_param)
            self.original_to_updated[orig_unsharded_param] = unsharded_param
            self.swap_map[variableify(orig_unsharded_param)] = unsharded_param
            return VISIT_TYPE.REPLACE

        if bsym.rhs() in self.bsyms_to_check:
            utils.check(
                any(arg in self.original_to_updated for arg in bsym.flat_proxy_args),
                lambda: f"None of {bsym.flat_proxy_args=} has been observed: {tuple(self.original_to_updated._dict.keys())}",
            )
        new_bsym = bsym.from_bsym_swap_proxies(self.swap_map)
        updated_output = new_bsym.sym(*new_bsym.args, **new_bsym.kwargs)
        if updated_output is not None:
            if isinstance(updated_output, ProxyInterface):
                updated_output = (updated_output,)
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
        if not isinstance(sharded_grad_tensor_proxy, TensorProxy):
            continue
        utils.check(
            index in index_in_flat_args_to_param_and_bucket,
            lambda: (
                f"Invalid {index=}, {sharded_grad_tensor_proxies}, "
                f"{tuple((k, v[0].name) for k, v in index_in_flat_args_to_param_and_bucket.items())=}"
            ),
        )
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


class FSDPCommBucketing:
    def __init__(
        self,
        compile_data: CompileData,
    ) -> None:
        from torch.distributed.distributed_c10d import _get_default_group

        self.compile_data = compile_data
        self.bucketing_strategy: FSDPBucketingStrategy = getattr(
            compile_data.fn, "bucketing_strategy", FSDPBucketingStrategy.NONE
        )
        self.apply_bucketing = self.bucketing_strategy != FSDPBucketingStrategy.NONE
        self.bucket_naming_func = get_extract_bucket_name_from_tensor_proxy(self.bucketing_strategy)
        self.group: ProcessGroup = getattr(compile_data.fn, "process_group_for_ddp", _get_default_group())

    def update_name_set(self, backward_trace: TraceCtx) -> TraceCtx:
        if not self.apply_bucketing:
            return
        utils.check(
            hasattr(self, "fsdp_fwd_trace"),
            lambda: "This method must be called after :func:`FSDPCommsOptimizer.apply_bucketing_to_forward_trace`",
        )
        backward_trace.names.update(self.fsdp_fwd_trace.names)

    def apply_bucketing_to_forward_trace(self, fwd_trace: TraceCtx, bwd_trace_names: set[str]) -> TraceCtx:
        """Optimize collective comms in fsdp with bucketing.

        This function is no-op if you pass :obj:`BucketingStrategy.NONE` as kwarg to :func:`thunder.distributed.fsdp`.
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
        trace_flat_args, _ = tree_flatten((fsdp_fwd_trace.args, fsdp_fwd_trace.kwargs))
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
        param_to_bucket = utils.ProxyDict()  # sharded param -> FSDPForwardBucket
        for bucket_name in bucket_name_to_sharded_params:
            sharded_params = tuple(bucket_name_to_sharded_params[bucket_name])
            bucket = FSDPForwardBucket(n_buckets, sharded_params, [], bucket_name)
            for p in sharded_params:
                param_to_bucket[p] = bucket
                index_in_flat_args_to_param_and_bucket[arg_to_index_in_flat_args[p]] = (p, bucket)

        producers, consumers = utils.producers_and_consumers(fsdp_fwd_trace)
        self.visit_for_fsdp_forward = FSDPFwdTraceVisitor(
            self.group,
            collective_comm_bsyms,
            producers,
            consumers,
            param_to_bucket,
        )

        new_trace = visitor_transform(
            fsdp_fwd_trace,
            self.visit_for_fsdp_forward,
            provenance="Merge Collective Comms",
        )
        self.index_in_flat_args_to_param_and_bucket = index_in_flat_args_to_param_and_bucket

        self.orig_fsdp_fwd_trace = fsdp_fwd_trace
        self.fsdp_fwd_trace = new_trace

        return new_trace

    def apply_bucketing_to_backward_trace(self, fsdp_bwd_trace: TraceCtx) -> TraceCtx:
        """Apply bucketing to reduce_scatter in fsdp bwd trace.

        1. Collect unsharded gradient tensor proxies and create buckets for them based on forward's buckets' name.
        2. Copy unsharded gradients into their buckets
        3. Call async redunce-scatter on buckets as they get ready
        4. Call wait on future tensors of buckets

        Args:
            fsdp_bwd_trace:
            compile_data:
            index_in_flat_args_to_bucket:

        Returns:
            - :class:`TraceCtx`
        """

        if not self.apply_bucketing:
            return fsdp_bwd_trace
        producers = utils.producers(fsdp_bwd_trace)

        unsharded_grad_to_bucket = create_map_from_unsharded_grad_to_bucket(
            self.index_in_flat_args_to_param_and_bucket,
            tree_flatten(fsdp_bwd_trace.output)[0],
            producers,
            self.group,
        )

        def visit(bsym: BoundSymbol) -> VISIT_TYPE:
            if bsym.sym.id in {
                PrimIDs.UNPACK_TRIVIAL,
                PrimIDs.UNPACK_SEQUENCE,
                PrimIDs.UNPACK_KEY,
                PrimIDs.UNPACK_EMPTY_DICT,
            }:
                return VISIT_TYPE.NO_OP

            if bsym.sym.id in {dist_prims.PrimIDs.REDUCE_SCATTER, dist_prims.PrimIDs.WAIT}:
                return VISIT_TYPE.REPLACE

            if bsym.sym.id == PrimIDs.RETURN:
                id_to_tensor_proxy = {}
                for bucket in unsharded_grad_to_bucket._dict.values():
                    sharded_bucket = dist_prims.wait(bucket.future)
                    sharded_grads = dist_prims.unpack_for_fsdp(
                        sharded_bucket,
                        bucket.orig_tensors,
                        self.group.size(),
                        "scatter",
                    )
                    id_to_tensor_proxy.update(dict(zip(bucket.tensor_indices, sharded_grads)))
                flat_outs, spec = tree_flatten(fsdp_bwd_trace.output)
                new_flat_outs = [id_to_tensor_proxy.get(i, o) for i, o in enumerate(flat_outs)]
                new_outs = tree_unflatten(new_flat_outs, spec)
                prims.python_return(new_outs)
                return VISIT_TYPE.REPLACE

            for out in bsym.flat_proxy_outs:
                if isinstance(out, TensorProxy) and out in unsharded_grad_to_bucket:
                    bucket: FSDPBackwardBucket = unsharded_grad_to_bucket[out]
                    bucket.tell(out)
                    if bucket.recieved_all_tensors():
                        buffer = bucket.pack(world_size=self.group.size())
                        future = dist_prims.reduce_scatter(
                            buffer,
                            dist_prims.DistributedReduceOps.SUM,
                            group=self.group,
                            do_async=True,
                        )
                        bucket.set_future(future, out)
                return VISIT_TYPE.INSERT_AFTER

        return visitor_transform(fsdp_bwd_trace, visit, provenance="Merge Collective Comms")
