from dataclasses import dataclass, replace
from functools import wraps, partial
from inspect import signature
from typing import Any, TYPE_CHECKING

import torch

from thunder.core.proxies import TensorProxy, FutureTensorProxy, variableify
from thunder.core.prims import PrimIDs
import thunder.core.utils as utils
from thunder.core.pytree import tree_flatten, tree_unflatten
from thunder.core.transform_common import replace_redundant_inputs
from thunder.core.trace import TraceCtx
from thunder.core.symbol import Symbol, BoundSymbol, BoundSymbolRHS
import thunder.distributed.prims as dist_prims
import thunder.torch as ltorch
import thunder


class ThunderFunction(torch.autograd.Function):
    @staticmethod
    def get_forward_backward_splitter(func, compile_data, compile_stats):
        from thunder import trace
        from thunder.executors.passes import transform_for_execution
        from thunder.executors.passes import del_last_used
        from thunder.core.rematerialization import rematerialize_forward_and_backward, rematerialize_all_gather
        from thunder.core.transforms import forward_and_backward_from_trace
        from thunder.cudagraphs import CUDAGraphExecutor
        from thunder.distributed.utils import sort_waits, sort_data_parallel_syncs, sort_waits_for_zero3
        from thunder.distributed.transforms import FSDPCommBucketing

        utils.check(compile_data is not None, lambda: "`compile_data` is required")

        def make_trace(func):
            return partial(
                trace(compile_data=compile_data, inline_trace=False, insert_ddp_syncs=not compile_data.using_jit), func
            )

        def split_forward_backward_compat(*args, **kwargs):
            fw_extrace, bw_extrace = split_forward_backward(func, compile_data, compile_stats, *args, **kwargs)
            return fw_extrace.python_callable(), bw_extrace.python_callable()

        return split_forward_backward_compat

    @staticmethod
    def forward(ctx, compiled_backward, saved_tensors, saved_other, flat_output, *flat_args):
        # Here we just propagate the tensors through the autograd graph
        ctx.saved_other = saved_other
        ctx.compiled_backward = compiled_backward

        # We must save tensors using ctx.save_for_backward
        ctx.save_for_backward(*saved_tensors)
        return flat_output

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
        return (None, None, None, None, *grads)


# TODO: RC1 Remove this
def thunder_backward(*, compile_data, compile_stats=None):
    """Decorator to wrap a Thunder function for use with PyTorch autograd.

    Args:
        thunder_func: A Thunder function.

    Returns:
        A wrapped function that can be used with PyTorch autograd.

    Example:
    >>> import torch
    >>> import thunder.clang as clang
    >>> from thunder.executors.torchex import thunder_backward
    >>> @thunder_backward()
    ... def func(a, b):
    ...     c = a + b
    ...     d = c * b
    ...     e = clang.sin(d) + clang.cos(c)
    ...     return e
    >>> a = torch.randn(3, device="cuda", requires_grad=True)
    >>> b = torch.randn(3, device="cuda", requires_grad=True)
    >>> c = func(a, b)
    >>> print(c)
    >>> sum(c).sum().backward()
    >>> print(f"a.grad: {a.grad}")
    >>> print(f"b.grad: {b.grad}")
    """

    def decorator(thunder_func):
        from thunder import compile

        # Compile's caching only works for many calls to the same compiled function
        # It does not work if the same function is compiled many times, so we must
        # decorate the augmented forward pass once with compile once and reuse it
        split_fw_bw = ThunderFunction.get_forward_backward_splitter(thunder_func, compile_data, compile_stats)
        compile_config = {
            "langctx": compile_data.langctx,
            "executors_list": compile_data.executors_list,
            "only_execute_prims": compile_data.only_execute_prims,
            "cache_option": compile_data.cache_option,
            "use_rematerialization": compile_data.use_rematerialization,
            "use_cudagraphs": compile_data.use_cudagraphs,
            **compile_data.compile_options,
            "disable_preprocessing": True,
            "disable_torch_autograd_support": True,
        }

        compiled_split_fw_bw = compile(
            split_fw_bw,
            **compile_config,
        )
        sig = signature(thunder_func)

        @wraps(thunder_func)
        def wrapper(*args, **kwargs):
            # Fetch the compiled forward and backward functions using the
            # compiled function cache
            compiled_forward, compiled_backward = compiled_split_fw_bw(*args, **kwargs)

            # Compiled forward function currently doesn't support positional
            # arguments passed as kwargs, so we must bind them here
            ba = sig.bind(*args, **kwargs)
            args, kwargs = ba.args, ba.kwargs

            # Run the compiled forward function
            data_for_autograd, (saved_tensors, saved_other) = compiled_forward(*args, **kwargs)

            # Connect produced tensors with PyTorch's autograd graph
            ThunderFunction.apply(
                compiled_backward,
                saved_tensors,
                saved_other,
                data_for_autograd["flat_output"],
                *data_for_autograd["flat_args"],
            )
            return data_for_autograd["output"]

        return wrapper

    return decorator


def split_forward_backward(computation_trc, compile_data, compile_stats, /, *args, **kwargs):
    from thunder import trace
    from thunder.executors.passes import transform_for_execution
    from thunder.executors.passes import del_last_used
    from thunder.core.rematerialization import rematerialize_forward_and_backward, rematerialize_all_gather
    from thunder.core.transforms import forward_and_backward_from_trace
    from thunder.cudagraphs import CUDAGraphExecutor
    from thunder.distributed.utils import sort_waits, sort_data_parallel_syncs, sort_waits_for_zero3
    from thunder.distributed.transforms import FSDPCommBucketing
    from thunder.core.transforms import eval_trace

    # TODO: the trace->func->trace could likely be simplified (and look nicer)
    #       we cannot use python_callable() here, see the old repos 2458
    if not isinstance(computation_trc, TraceCtx):
        # for the legacy codepath
        func = computation_trc
    else:

        def func(*args):
            return eval_trace(computation_trc, *args)

    utils.check(compile_data is not None, lambda: "`compile_data` is required")

    def make_trace(func):
        return partial(
            trace(compile_data=compile_data, inline_trace=False, insert_ddp_syncs=not compile_data.using_jit), func
        )

    computation_trc.kwargs = {}
    # NOTE: This function is rather slow, so it's intended to be used
    # behind a cache.
    ba = signature(func).bind(*args, **kwargs)
    ba.apply_defaults()
    args, kwargs = ba.args, ba.kwargs
    flat_args, _ = tree_flatten((args, kwargs))
    tensor_cls = (torch.Tensor, TensorProxy)
    requires_grad_mask = tuple(isinstance(arg, tensor_cls) and arg.requires_grad for arg in flat_args)
    # If none of the inputs require gradients, raise an error
    if not any(requires_grad_mask):
        raise RuntimeError("PyTorch's Autograd interface requires at least one tensor input with requires_grad=True")

    primal_trace = make_trace(func)(*args, **kwargs) if not compile_data.using_jit else computation_trc
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
        _fsdp_comm_bucketing = FSDPCommBucketing(compile_data)
        fw_trace = _fsdp_comm_bucketing.apply_bucketing_to_forward_trace(fw_trace, bw_trace.names)
        _fsdp_comm_bucketing.update_name_set(bw_trace)

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
    new_fw_saved_tensors_for_backward = fw_extrace.output[1][0]
    swap_map = {
        variableify(x): y
        for x, y in zip(original_bw_saved_tensors_for_backward, new_fw_saved_tensors_for_backward)
        if variableify(x) != variableify(y)
    }
    new_bsyms = replace_redundant_inputs(swap_map, bw_trace.bound_symbols)
    # replace_redundant_inputs doesn't replace the output of
    # UNPACK_SEQUENCE so we do it manually. Here we have certain
    # assumptions about the structure of the backward trace.
    assert bw_trace.bound_symbols[0].sym.id == PrimIDs.UNPACK_TRIVIAL
    assert bw_trace.bound_symbols[0].kwargs["name"] == "saved_for_backward"
    assert bw_trace.bound_symbols[4].sym.id == PrimIDs.UNPACK_SEQUENCE
    assert bw_trace.bound_symbols[4].args[0].name == "C0"
    new_bsyms[4] = new_bsyms[4].from_bsym_swap_proxies(
        swap_map,
        skip_inputs=False,
        skip_output=False,
        skip_subsymbols=False,
    )
    bw_trace.bound_symbols = new_bsyms
    if getattr(compile_data.fn, "use_ddp", False):
        from thunder.distributed.transforms import optimize_allreduce_in_ddp_backward

        bw_trace = optimize_allreduce_in_ddp_backward(bw_trace, compile_data)
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
    if getattr(compile_data.fn, "use_fsdp", False):
        assert hasattr(compile_data.fn, "sharding_strategy")
        if getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO3:
            from thunder.distributed.utils import limit_in_flight_allgathers
            from thunder.distributed import FSDPBucketingStrategy

            fw_extrace = sort_waits_for_zero3(fw_extrace)
            fw_extrace = limit_in_flight_allgathers(
                fw_extrace,
                3,
                compile_data.fn.bucketing_strategy != FSDPBucketingStrategy.NONE,
            )
            bw_extrace = sort_waits_for_zero3(bw_extrace)
            bw_extrace = limit_in_flight_allgathers(
                bw_extrace,
                3,
                compile_data.fn.bucketing_strategy != FSDPBucketingStrategy.NONE,
            )
        if getattr(compile_data.fn, "sharding_strategy") == FSDPType.ZERO2:
            fw_extrace = sort_waits(fw_extrace)
            bw_extrace = sort_waits(bw_extrace)
    if getattr(compile_data.fn, "use_ddp", False):
        bw_extrace = sort_waits(bw_extrace)

    # Importing here to avoid cyclical dependencies in future.
    from thunder.executors.transformer_engineex import transformer_engine_ex, _rearrange_transformer_engine_linear

    if transformer_engine_ex in compile_data.executors_list:
        # NOTE: `_rearrange_transformer_engine_linear` mutates `fw_extrace`.
        _rearrange_transformer_engine_linear(fw_extrace, bw_extrace)

    fw_extrace = del_last_used(fw_extrace)
    fw_traces.append(fw_extrace)

    bw_extrace = del_last_used(bw_extrace, clear_collections=True)
    bw_traces.append(bw_extrace)

    if compile_stats is not None:
        compile_stats.last_traces += fw_traces
        compile_stats.last_backward_traces += bw_traces

    return fw_extrace, bw_extrace
