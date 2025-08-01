from collections.abc import Sequence

import torch

import thunder.core.utils as utils
from .utils import is_cudagraph_capturing


# NOTE: Split autograd.Function
# We split the autograd.Function into two parts because this allows
# the args to the ThunderOutputFunction.backward to go out of scope
# and the tensors (the grad_outs matching the flattened output) to be
# deallocated when they have been processed by the compiled backward function.
# For the correspondence between the functions hidden from autograd, we use
# a side channel (an empt dict) passed as an argument. To link the two
# functions in autograd, we use a dummy tensor on the meta device.
class ThunderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        return_none_instead_of_grads,
        compiled_backward,
        side_channel,
        saved_tensors,
        saved_other,
        is_differentiable_outputs,
        flat_output,
        *flat_args,
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
            if isinstance(t, torch.Tensor) and t._base is not None:
                # Only detach if the Tensor is a view.
                # This is needed because TransformerEngine can create (non-view) tensors that have different
                # metadata on the `t.detach()` output than on `t`. (Ideally, this shouldn't be the case)
                # See https://github.com/Lightning-AI/lightning-thunder/pull/1600 for details.
                return t.detach()
            return t

        saved_tensors = tuple(map(detach_if_tensor, saved_tensors))

        ctx.side_channel = side_channel
        if side_channel is not None:
            assert is_differentiable_outputs is None, (
                "is_differentiable_outputs is not supported when side_channel is not None"
            )
            assert not side_channel
            ctx.side_channel["fw"] = flat_output
            # We must save tensors using ctx.save_for_backward but
            # we want to save the tensors in the function returning the outputs to avoid memory leaks
            # (basically ref-cycles via output.grad_fn.next_functions[0, 0].saved_tensors[0] == output
            # PyTorch autograd handles this gracefully for output.grad_fn.saved_tensors)
            ctx.side_channel["tensors_to_save"] = saved_tensors
            return torch.randn(1, device="meta", requires_grad=True)
        else:
            if is_differentiable_outputs is None:
                # Default to original behavior of marking all outputs as differentiable.
                is_differentiable_outputs = tuple(True for _ in flat_output)

            ctx.save_for_backward(*saved_tensors)

            assert len(flat_output) == len(is_differentiable_outputs)
            filter_non_differentiable = [
                o for o, is_differentiable in zip(flat_output, is_differentiable_outputs) if not is_differentiable
            ]
            ctx.mark_non_differentiable(*filter_non_differentiable)

            return flat_output

    # NOTE: If `torch.autograd.function.once_differentiable` is to be removed,
    # one must take care of correctly removing the `detach_if_tensor` above.
    # For more context, see NOTE [Saved view of output of torch.autograd.Function leaks] above.
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *raw_args):
        if ctx.side_channel is not None:
            args = ctx.side_channel.pop("bw")
            saved_tensors_list = ctx.side_channel.pop("saved_tensors")
            assert not ctx.side_channel
        else:
            args = list(raw_args)
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

        assert not args
        # Inside the compiled backward we must clear the saved_tensors_list
        assert not saved_tensors_list, "saved_tensors_list must be empty after calling compiled_backward"
        # TODO(crcrpar): Remove if-else once `dist_prims.stash_grad_for_fsdp` starts to return `None`
        # NOTE(crcrpar): In fsdp no-sync, unsharded gradients are attached and accumulated to their parameters as the attr of `_thunder_fsdp_unsharded_grad` in order to avoid shape mismatch of a param and its grad. When exiting the no_sync context, the accumulated, unsharded gradients are reduce-scattered into the attr of `grad` and `_thunder_fsdp_unsharded_grad` is removed.
        if not ctx.return_none_instead_of_grads:
            return (None, None, None, None, None, None, None, *grads)
        else:
            n_grads = len(grads)
            del grads
            return (None, None, None, None, None, None, None, *([None] * n_grads))


class ThunderOutputFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy, side_channel, *args):
        ctx.side_channel = side_channel
        ctx.num_args = len(args)
        res = ctx.side_channel.pop("fw")
        ctx.save_for_backward(*ctx.side_channel.pop("tensors_to_save"))
        assert not ctx.side_channel
        return res

    @staticmethod
    def backward(ctx, *args):
        assert not ctx.side_channel
        ctx.side_channel["bw"] = list(args)
        ctx.side_channel["saved_tensors"] = list(ctx.saved_tensors)  # see above
        ctx.maybe_clear_saved_tensors()  # Delete the reference to all saved tensors in the context
        return torch.randn(1, device="meta"), None, *([None] * ctx.num_args)


def connect_to_autograd(
    *,
    backward_fn,
    flat_args,
    flat_output,
    saved_tensors,
    saved_other,
    return_none_instead_of_grads,
    disable_split_autograd,
    is_differentiable_outputs: Sequence[bool] | None,
):
    # PyTorch seems to not like our side channel trick when capturing graphs
    # through dynamo and using cuda graphs.
    # Of course, the real trick is to use the CUDAGraphTransform instead
    # of having something else apply it while introducing funny additional
    # conditions for success.
    if not is_cudagraph_capturing() and not disable_split_autograd:
        side_channel = {}
    else:
        side_channel = None

    if is_differentiable_outputs is not None:
        utils.check(
            disable_split_autograd, lambda: "is_differentiable_outputs is not supported when split_autograd is enabled"
        )

    dummy_res = ThunderFunction.apply(
        return_none_instead_of_grads,
        backward_fn,
        side_channel,
        saved_tensors,
        saved_other,
        is_differentiable_outputs,
        flat_output,
        *flat_args,
    )
    if side_channel is not None:
        # we need to pass the inputs to avoid "leave has moved inside the graph"
        # if the function returns an argument as is
        ThunderOutputFunction.apply(dummy_res, side_channel, *flat_args)
