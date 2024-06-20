from functools import partial

import optree
import torch
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
from thunder.core.baseutils import ProxyInterface
from types import EllipsisType, NoneType

# We need torch.Size to be treated the same way as a list or tuple
# In PyTorch this is registered here:
# https://github.com/pytorch/pytorch/blob/8bc04f46fe8e69188fa46f1611b46788a7d4824d/torch/fx/experimental/proxy_tensor.py#L51
optree.register_pytree_node(
    torch.Size,
    lambda size: (list(size), None, None),
    lambda _, children: tuple(children),
    namespace=optree.registry.__GLOBAL_NAMESPACE,
)


def tree_flatten(args):
    if type(args) not in {
        dict,
        list,
        str,
        int,
        bool,
        tuple,
        torch.dtype,
        float,
        dtypes.floating,
        dtypes.boolean_dtypes,
        devices.Device,
        torch.memory_format,
        NoneType,
        slice,
        complex,
        type,
        EllipsisType,
    } and not isinstance(args, ProxyInterface):
        raise TypeError(f"tree_flatten of type {type(args)} is not supported.")
    return optree.tree_flatten(args, none_is_leaf=True)


tree_map = partial(optree.tree_map, none_is_leaf=True)


def tree_unflatten(values, spec):
    return optree.tree_unflatten(spec, values)


__all__ = ["tree_flatten", "tree_unflatten", "tree_map"]
