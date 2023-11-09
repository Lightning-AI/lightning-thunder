from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Tuple, Union
from collections.abc import Callable
from collections.abc import Sequence

import torch
from torch.cuda.graphs import CUDAGraph

from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten

from thunder.core.utils import check, default_dataclass_params, flatten_func, safe_zip


def static_input(x: torch.Tensor | torch.nn.Parameter | Any):
    """Returns a static input of the same shape as x.

    This is useful for creating static inputs for a CUDA graph.

    Args:
        x (Union[torch.Tensor, torch.nn.Parameter, Any]): The input to create a static version of.

    Returns:
        Union[torch.Tensor, Any]: A static version of x.
    """
    if isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Parameter):
        return torch.empty_like(x)
    return x


@dataclass(**default_dataclass_params)
class ArgsDescriptor:
    dtypes: tuple
    sizes: tuple
    strides: tuple
    non_tensor_arg: tuple
    args: list = field(hash=False, repr=False, compare=False)


def to_args_descriptor(*args):
    def extract(a):
        if isinstance(a, torch.Tensor) or isinstance(a, torch.nn.Parameter):
            return a.dtype, a.size(), a.stride(), None
        else:
            return type(a), None, None, a

    dtypes, sizes, strides, non_tensor_arg = zip(*map(extract, args))
    return ArgsDescriptor(dtypes, sizes, strides, non_tensor_arg, args)


@lru_cache
def make_cuda_graph(
    flat_fn: Callable, args_descr: ArgsDescriptor, static_args_mask: tuple[bool, ...]
) -> tuple[CUDAGraph, Sequence[torch.Tensor | Any], Sequence[torch.Tensor | Any]]:
    """Creates a CUDA graph from a flattened function and its arguments.

    Args:
        flat_fn (Callable): The flattened function to create a CUDA graph from.
        args_descr (ArgsDescriptor): The arguments descriptor of the flattened function.
        static_args_mask (Tuple[bool, ...]): A mask indicating which arguments are static.

    Returns:
        Tuple[CUDAGraph, Sequence[torch.Tensor, Any], Sequence[torch.Tensor, Any]]:
            A tuple containing the CUDA graph, the static inputs, and the static outputs.
    """
    args = args_descr.args

    # Warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        static_inputs = tuple(static_input(a) if not m else a for a, m in zip(args, static_args_mask))
        for _ in range(3):
            flat_fn(*static_inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # Record

    graph = torch.cuda.CUDAGraph()
    # NOTE: We are using default private pool here, but it is possibly better to
    # use a custom pool for better memory management. See CUDA Graphs Tree in
    # PyTorch's Inductor: torch/_inductor/cudagraph_trees.py
    # Design doc: https://docs.google.com/document/d/1ZrxLGWz7T45MSX6gPsL6Ln4t0eZCSfWewtJ_qLd_D0E/view
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = flat_fn(*static_inputs)

    return graph, static_inputs, static_outputs


class CUDAGraphExecutor:
    def __init__(self, fn: Callable, copy_outputs: bool = False, num_constant_args: int | None = None):
        """Creates a CUDA graph executor.

        Args:
            fn (Callable): The function to create a CUDA graph from.
            copy_outputs (bool, optional): Whether to copy outputs to host. Defaults to False.
            num_constant_args (int, optional): The number of constant arguments. Defaults to None.
                It is assumed that the first num_constant_args arguments are constant.

        Raises:
            RuntimeError: If the function is not a Python callable or nn.Module.
        """
        check(
            isinstance(fn, torch.nn.Module) or callable(fn),
            lambda f: f"Expected a Python callable or nn.Module, but got {type(fn)}.",
        )
        self.fn = fn
        self.copy_outputs = copy_outputs
        self.num_constant_args = num_constant_args

        self.flat_fn = None

    def __call__(self, *args, **kwargs):
        if self.flat_fn is None:
            self.flat_fn, flat_args, _ = flatten_func(self.fn, args, kwargs)
        else:
            flat_args, _ = tree_flatten((args, kwargs))

        args_descr = to_args_descriptor(*flat_args)
        if self.num_constant_args is not None:
            static_args_mask = (True,) * self.num_constant_args + (False,) * (len(flat_args) - self.num_constant_args)
        else:
            static_args_mask = tuple(isinstance(a, torch.nn.Parameter) for a in flat_args)
        cuda_graph, static_inputs, static_outputs = make_cuda_graph(self.flat_fn, args_descr, static_args_mask)

        # Update static inputs
        for static_input, new_arg, mask in safe_zip(static_inputs, flat_args, static_args_mask):
            if not mask and isinstance(new_arg, torch.Tensor):
                static_input.copy_(new_arg)

        cuda_graph.replay()
        if self.copy_outputs:
            return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, static_outputs)
        return static_outputs
