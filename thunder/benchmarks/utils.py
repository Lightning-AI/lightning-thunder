from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import thunder
from thunder.tests.make_tensor import make_tensor

if TYPE_CHECKING:
    from collections.abc import Callable


def backward_only(fn: Callable, *args, setup_graph_on_each_invocation=False, **kwargs):
    """
    Returns a function that runs the backward pass of the given function.

    The returned function should be called with the output of setup function.

    Args:
        fn: The forward function
        setup_graph_on_each_invocation: Should the forward graph be setup on each invocation.
                                        Defaults to False.
        *args: Arguments to the forward function
        **kwargs: Keyword arguments to the forward function

    Returns:
        A tuple of the backward function and the setup function
        that returns the arguments for the backward function.
    """
    if setup_graph_on_each_invocation:
        return backward_only_setup_graph_on_each_invocation(fn, *args, **kwargs)

    return backward_only_setup_graph_once(fn, *args, **kwargs)


def backward_only_setup_graph_once(fn: Callable, *args, **kwargs):
    """
    Returns a function that runs the backward pass of the given function.

    The returned function should be called with the output gradients.

    Args:
        fn: The forward function
        *args: Arguments to the forward function
        **kwargs: Keyword arguments to the forward function

    Returns:
        A tuple of the backward function and the setup function
        that returns the arguments for the backward function.
    """
    result = fn(*args, **kwargs)
    result = thunder.core.utils.sequencify(result)

    forward_inputs = thunder.core.pytree.tree_flatten((args, kwargs))[0]
    forward_inputs = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, forward_inputs))
    backwardable_tensor_result = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, result))

    # Capture metadata for backward to avoid keeping the result in memory
    backwardable_result_metadata = [(r.dtype, r.device, r.shape) for r in backwardable_tensor_result]

    def backward_setup():
        output_grads = []
        for dtype, device, shape in backwardable_result_metadata:
            torch_dtype = thunder.torch.to_torch_dtype(dtype)
            torch_device = thunder.core.devices.to_torch_device(device)
            output_grads.append(make_tensor(shape, dtype=torch_dtype, device=torch_device, requires_grad=False))
        return output_grads

    def backward_fn(*output_grads):
        for i in forward_inputs:
            i.grad = None

        torch.autograd.backward(backwardable_tensor_result, output_grads, retain_graph=True)

    return backward_fn, backward_setup


def backward_only_setup_graph_on_each_invocation(fn: Callable, *args, **kwargs):
    """
    Returns a function that runs the backward pass of the given function.

    The returned function should be called with the output of setup function.

    NOTE: The forward graph will be setup on each invocation.

    Args:
        fn: The forward function
        *args: Arguments to the forward function
        **kwargs: Keyword arguments to the forward function

    Returns:
        A tuple of the backward function and the setup function
        that returns the arguments for the backward function.
    """

    # backward setup takes care of running the forward, saving the relevant context for backward
    # and returning the `grads` for output.
    def backward_setup():
        result = fn(*args, **kwargs)
        result = thunder.core.utils.sequencify(result)

        forward_inputs = thunder.core.pytree.tree_flatten((args, kwargs))[0]
        forward_inputs = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, forward_inputs))
        backwardable_tensor_result = list(filter(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, result))

        # Capture metadata for backward to avoid keeping the result in memory
        backwardable_result_metadata = [(r.dtype, r.device, r.shape) for r in backwardable_tensor_result]
        output_grads = []
        for dtype, device, shape in backwardable_result_metadata:
            torch_dtype = thunder.torch.to_torch_dtype(dtype)
            torch_device = thunder.core.devices.to_torch_device(device)
            output_grads.append(make_tensor(shape, dtype=torch_dtype, device=torch_device, requires_grad=False))
        return result, forward_inputs, output_grads

    # Actually do the backward pass.
    def backward_fn(result, forward_inputs, output_grads):
        for i in forward_inputs:
            i.grad = None

        torch.autograd.backward(result, output_grads)

    return backward_fn, backward_setup
