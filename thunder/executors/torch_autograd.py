from dataclasses import replace
from typing import TYPE_CHECKING

import torch

import thunder.core.utils as utils
from thunder.core.prims import PrimIDs
from thunder.core.proxies import TensorProxy, variableify
from thunder.core.pytree import tree_flatten
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import TraceCtx, from_trace, set_tracectx, reset_tracectx
from thunder.core.transform_common import replace_redundant_inputs
from thunder.core.vjp_utils import get_saved_for_backward_tensors, set_saved_for_backward_tensors

if TYPE_CHECKING:
    from thunder.core.trace import VariableInterface


def rename_bwd_trace_outputs(bwd_trace: TraceCtx, fwd_trace: TraceCtx) -> TraceCtx:
    """Have backward trace output tensor proxy names follow `grad_for_<param>` format.

    Since ``i``-th tensor proxy of backward trace's outputs is grad of ``i``-th tensor proxy of forward trace's inputs,
    this method looks up to forward trace's inputs to get the param name for each grad.

    Args:
        bwd_trace:
        fwd_trace:

    Returns:
        :class:`thunder.core.trace.TraceCtx`
    """

    # [note: why setting trace ctx?]
    # [`TensorProxy.replace_name`](https://github.com/Lightning-AI/lightning-thunder/blob/561b699/thunder/core/proxies.py#L1221-L1223) calls
    # [`tensorproxy`](https://github.com/Lightning-AI/lightning-thunder/blob/561b699/thunder/core/proxies.py#L1506-L1520)
    # which then calls `TensorProxy.__init__`. `TensorProxy.__init__` of course calls
    # [` Proxy.__init__`](https://github.com/Lightning-AI/lightning-thunder/blob/561b699/thunder/core/proxies.py#L81-L86).
    # `Proxy`'s dunder init calls [`make_proxy_name`](https://github.com/Lightning-AI/lightning-thunder/blob/561b699/thunder/core/proxies.py#L81-L86)
    # which depends on a tracectx.
    trace_tok = set_tracectx(bwd_trace)

    swap_map: dict[VariableInterface, TensorProxy] = {}
    bwd_outputs, _ = tree_flatten(bwd_trace.output)
    fwd_inputs, _ = tree_flatten((fwd_trace.args, fwd_trace.kwargs))

    utils.check(len(bwd_outputs) == len(fwd_inputs), lambda: f"{len(bwd_outputs)=}, {len(fwd_inputs)=}")

    for fwd_arg, bwd_out in zip(fwd_inputs, bwd_outputs):
        if isinstance(bwd_out, TensorProxy):
            swap_map[variableify(bwd_out)] = bwd_out.replace_name(f"grad_for_{fwd_arg.name}")
    reset_tracectx(trace_tok)

    renamed_bwd_trace = from_trace(bwd_trace)
    renamed_bwd_trace.bound_symbols = []

    bsym: BoundSymbol
    for bsym in bwd_trace.bound_symbols:
        renamed_bwd_trace.bound_symbols.append(bsym.from_bsym_swap_proxies(swap_map=swap_map))

    return renamed_bwd_trace


class ThunderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, return_none_instead_of_grads, compiled_backward, saved_tensors, saved_other, flat_output, *flat_args
    ):
        # Here we just propagate the tensors through the autograd graph
        ctx.return_none_instead_of_grads = return_none_instead_of_grads
        ctx.saved_other = saved_other
        ctx.compiled_backward = compiled_backward

        # NOTE [Saved view of output of torch.autograd.Function leaks]
        # We detach here to avoid a bug in PyTorch where
        # it leaks memory if view of the output of torch.autograd.Function
        # is saved for backward.
        # See - https://github.com/pytorch/pytorch/issues/94990#issuecomment-1435181804
        # NOTE - Detaching here would lead to problem with higher order differentiation but
        #        this is ok for now because ThunderFunction is only `once_differentiable`.
        def detach_if_tensor(t):
            # Some operations may claim to return Tensor (as per their meta function)
            # but may return None at Runtime (eg. noticed this for sdpa)
            if isinstance(t, torch.Tensor):
                return t.detach()
            return t

        saved_tensors = tuple(map(detach_if_tensor, saved_tensors))

        # We must save tensors using ctx.save_for_backward
        ctx.save_for_backward(*saved_tensors)
        return flat_output

    # NOTE: If `torch.autograd.function.once_differentiable` is to be removed,
    # one must take care of correctly removing the `detach_if_tensor` above.
    # For more context, see NOTE [Saved view of output of torch.autograd.Function leaks] above.
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *args):
        # ctx.saved_tensors is a tuple of tensors saved in forward. Our compiled
        # backward is a really long function that takes all the tensors saved in
        # forward and gradually uses them to compute the gradients of the
        # inputs. Unfortunately, Python holds a reference to all arguments of a
        # function until the function returns, even if we delete the variable
        # "saved_tensors" inside the function, the tensors will still be held in
        # memory until the function returns. Fortunately, Python passes mutable
        # objects by reference, so we can just replace the saved_tensors with an
        # empty list and the memory will be freed immediately. We must also
        # delete the reference to the saved_tensors in the context, otherwise
        # the memory will be freed only when the context is deleted.
        saved_tensors_list = list(ctx.saved_tensors)  # Make a copy as we will mutate it

        # This is an undocumented API, but it's the only way to clear the
        # reference to the saved tensors in the context
        ctx.maybe_clear_saved_tensors()  # Delete the reference to all saved tensors in the context
        grads = ctx.compiled_backward([saved_tensors_list, ctx.saved_other], args)

        # Inside the compiled backward we must clear the saved_tensors_list
        assert not saved_tensors_list, "saved_tensors_list must be empty after calling compiled_backward"
        # TODO(crcrpar): Remove if-else once `dist_prims.stash_grad_for_fsdp` starts to return `None`
        # NOTE(crcrpar): In fsdp no-sync, unsharded gradients are attached and accumulated to their parameters as the attr of `_thunder_fsdp_unsharded_grad` in order to avoid shape mismatch of a param and its grad. When exiting the no_sync context, the accumulated, unsharded gradients are reduce-scattered into the attr of `grad` and `_thunder_fsdp_unsharded_grad` is removed.
        if not ctx.return_none_instead_of_grads:
            return (None, None, None, None, None, *grads)
        else:
            n_grads = len(grads)
            del grads
            return (None, None, None, None, None, *([None] * n_grads))


def split_forward_backward(computation_trc: TraceCtx, compile_data, compile_stats, /, *flat_args):
    from thunder.core.rematerialization import rematerialize_all_gather, rematerialize_forward_and_backward
    from thunder.core.transforms import forward_and_backward_from_trace
    from thunder.distributed.transforms import FSDPCommBucketing
    from thunder.distributed.utils import sort_data_parallel_syncs, sort_waits, sort_communication_ops
    from thunder.executors.passes import del_last_used, transform_for_execution

    utils.check(compile_data is not None, lambda: "`compile_data` is required")
    # NOTE: This function is rather slow, so it's intended to be used
    # behind a cache.
    tensor_cls = (torch.Tensor, TensorProxy)
    requires_grad_mask = tuple(isinstance(arg, tensor_cls) and arg.requires_grad for arg in flat_args)
    # If none of the inputs require gradients, raise an error
    if not any(requires_grad_mask):
        raise RuntimeError("PyTorch's Autograd interface requires at least one tensor input with requires_grad=True")

    primal_trace = computation_trc
    primal_trace = sort_data_parallel_syncs(primal_trace)

    if compile_stats is not None:
        compile_stats.last_traces.append(primal_trace)

    # torch.autograd.Function doesn't support non-flat outputs, the
    # grads wouldn't be propagated and backward receives None for each
    # non-flat non-tensor output. The output must also be a flat tuple,
    # not any other container type. So we need to flatten the outputs of
    # the forward trace and inputs of the backward trace.
    fw_trace, bw_trace = forward_and_backward_from_trace(primal_trace, torch_autograd=True)

    fw_traces = [fw_trace]
    bw_traces = [bw_trace]

    from thunder.distributed import FSDPType

    # only enable rematerialize_params_in_backward when using FSDP ZeRO3
    _rematerialize_params_in_backward = (
        getattr(compile_data.fn, "use_fsdp", False) and getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO3
    )
    if _rematerialize_params_in_backward:
        fw_trace, bw_trace = rematerialize_all_gather(fw_trace, bw_trace)

    # evil, but we really, really don't want to have the same name for different things
    fw_trace.names.update(bw_trace.names)
    bw_trace.names = fw_trace.names

    # Update the backward trace to only compute gradients for the
    # inputs that require gradients
    assert bw_trace.bound_symbols[-1].sym.id == PrimIDs.RETURN
    filtered_grads = tuple(
        (arg_grad if requires_grad else None)
        for arg_grad, requires_grad in utils.safe_zip(bw_trace.bound_symbols[-1].args[0], requires_grad_mask)
    )

    # autograd.Function.backward expects a flat tuple of gradients
    bw_trace.bound_symbols[-1] = replace(bw_trace.bound_symbols[-1], args=(filtered_grads,))

    _fsdp_comm_bucketing: FSDPCommBucketing | None = None
    if getattr(compile_data.fn, "use_fsdp", False):
        _fsdp_comm_bucketing = FSDPCommBucketing(compile_data, computation_trc)
        fw_trace = _fsdp_comm_bucketing.apply_bucketing_to_forward_trace(fw_trace)

    # Now we can run the optimization passes on the forward trace
    # TODO Restore request for no rematerialization
    fw_extrace = transform_for_execution(
        fw_trace,
        executors_list=compile_data.executors_list,
    )
    fw_traces.append(fw_extrace)

    # Some of the optimization passes change proxies in the trace and
    # any change in the forward trace must be reflected in the backward
    # trace.
    original_bw_saved_tensors_for_backward = bw_trace.args[0][0]
    new_fw_saved_tensors_for_backward = get_saved_for_backward_tensors(fw_extrace)

    # saved meta data (this could also contain proxies)
    original_bw_saved_meta_for_backward = bw_trace.args[0][1]
    new_fw_saved_meta_for_backward = fw_extrace.output[1][1]

    saved_tensors_swap_map = {
        variableify(x): y
        for x, y in zip(original_bw_saved_tensors_for_backward, new_fw_saved_tensors_for_backward)
        if variableify(x) != variableify(y)
    }

    saved_metadata_swap_map = {
        variableify(x): y
        for x, y in zip(original_bw_saved_meta_for_backward, new_fw_saved_meta_for_backward)
        if variableify(x) != variableify(y)
    }
    swap_map = saved_tensors_swap_map | saved_metadata_swap_map

    new_bsyms = replace_redundant_inputs(swap_map, bw_trace.bound_symbols)
    # replace_redundant_inputs doesn't replace the output of
    # UNPACK_SEQUENCE so we do it manually. Here we have certain
    # assumptions about the structure of the backward trace.
    assert bw_trace.bound_symbols[0].sym.id == PrimIDs.UNPACK_TRIVIAL
    assert bw_trace.bound_symbols[0].kwargs["name"] == "saved_for_backward"
    assert bw_trace.bound_symbols[4].sym.id == PrimIDs.UNPACK_SEQUENCE
    assert bw_trace.bound_symbols[4].args[0].name == "C0"
    assert bw_trace.bound_symbols[5].sym.id == PrimIDs.UNPACK_SEQUENCE
    assert bw_trace.bound_symbols[5].args[0].name == "C1"
    new_bsyms[4] = new_bsyms[4].from_bsym_swap_proxies(
        swap_map,
        skip_inputs=False,
        skip_output=False,
        skip_subsymbols=False,
    )
    new_bsyms[5] = new_bsyms[5].from_bsym_swap_proxies(
        swap_map,
        skip_inputs=False,
        skip_output=False,
        skip_subsymbols=False,
    )

    # remove duplicates
    # The NVFuser (and possibly others) fusion pass applied on the forward during has a
    # CSE pass that may lead to duplicate symbols saved for backward. This causes trouble
    # because we see duplicates in the unpacking. But the passes are unaware of the backward,
    # so they cannot handle it themselves, so we clean this up here.
    seen = set()
    new_fw_out = []
    new_bw_inp = []
    for p_fw, p_bw in zip(get_saved_for_backward_tensors(fw_extrace), new_bsyms[4].output, strict=True):
        if p_fw.name not in seen:
            seen.add(p_fw.name)
            new_fw_out.append(p_fw)
            new_bw_inp.append(p_bw)
    new_bsyms[4] = new_bsyms[4].from_bsym(output=tuple(new_bw_inp))
    set_saved_for_backward_tensors(fw_extrace, new_fw_out)

    bw_trace.bound_symbols = new_bsyms

    if getattr(compile_data.fn, "use_fsdp", False):
        bw_trace = _fsdp_comm_bucketing.apply_bucketing_to_backward_trace(bw_trace)

    # Now we can run the optimization passes on the backward trace
    # TODO Restore request for no rematerialization
    bw_extrace = transform_for_execution(
        bw_trace,
        executors_list=compile_data.executors_list,
    )
    bw_traces.append(bw_extrace)

    fw_extrace, bw_extrace = rematerialize_forward_and_backward(fw_extrace, bw_extrace)
    fw_traces.append(fw_extrace)
    bw_traces.append(bw_extrace)

    # We need to sort the waits in forward and backward trace to overlap
    # computation with communication
    # For performance we need the wait_prim_impl nodes in the execution trace to be as far from the
    # communication ops as possible. But it causes the all_gather_prim_impl nodes gathered at the start of
    # backward trace and increases the peak allocated memory
    use_fsdp: bool = getattr(compile_data.fn, "use_fsdp", False)
    if use_fsdp:
        assert hasattr(compile_data.fn, "sharding_strategy")
        if getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO3:
            from thunder.distributed import FSDPBucketingStrategy
            from thunder.distributed.utils import limit_in_flight_allgathers

            fw_extrace = sort_communication_ops(fw_extrace)
            fw_extrace = limit_in_flight_allgathers(
                fw_extrace,
                3,
                compile_data.fn.bucketing_strategy != FSDPBucketingStrategy.NONE,
            )
            bw_extrace = sort_communication_ops(bw_extrace)
            bw_extrace = limit_in_flight_allgathers(
                bw_extrace,
                3,
                compile_data.fn.bucketing_strategy != FSDPBucketingStrategy.NONE,
            )
        if getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO2:
            from thunder.distributed import FSDPBucketingStrategy
            from thunder.distributed.utils import limit_in_flight_allgathers
            from sys import maxsize as INT_MAX

            # sort the allgather+wait as consumer order just before consumer
            fw_extrace = sort_communication_ops(fw_extrace)
            # unlimited number of allgathers, i.e. allgathers are listed at the beginning of the trace in consumer order and wait stays just before wait
            fw_extrace = limit_in_flight_allgathers(
                fw_extrace,
                INT_MAX,
                compile_data.fn.bucketing_strategy != FSDPBucketingStrategy.NONE,
            )
            bw_extrace = sort_waits(bw_extrace)
    use_ddp: bool = getattr(compile_data.fn, "use_ddp", False)
    if use_ddp:
        bw_extrace = sort_waits(bw_extrace)
    if (not use_ddp) and (not use_fsdp):
        from thunder.distributed.utils import maybe_sort_waits

        _, fw_extrace = maybe_sort_waits(fw_extrace)
        _, bw_extrace = maybe_sort_waits(bw_extrace)

    # Importing here to avoid cyclical dependencies in future.
    from thunder.executors.transformer_engineex import _transformer_engine_bwd_fp8_meta_sync, transformer_engine_ex

    if transformer_engine_ex in compile_data.executors_list:
        # NOTE: `_transformer_engine_bwd_fp8_meta_sync` may mutate `fw_extrace` or `bw_extrace`.
        _transformer_engine_bwd_fp8_meta_sync(fw_extrace, bw_extrace)

    fw_extrace = del_last_used(fw_extrace)
    fw_traces.append(fw_extrace)

    bw_extrace = del_last_used(bw_extrace, clear_mutable_collections=True)
    bw_traces.append(bw_extrace)

    bw_trace = rename_bwd_trace_outputs(bw_extrace, fw_extrace)

    if compile_stats is not None:
        compile_stats.last_traces += fw_traces
        compile_stats.last_backward_traces += bw_traces

    # Enable wrapping with `te.fp8_autocast`.
    fw_extrace._include_te_fp8_autocast = True
    # We only want the forward function to be called with `te.fp8_autocast` manager.
    bw_extrace._include_te_fp8_autocast = False

    return fw_extrace, bw_extrace
