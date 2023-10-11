from functools import partial

import optree
import torch

# We need torch.Size to be treated the same way as a list or tuple
# In PyTorch this is registered here:
# https://github.com/pytorch/pytorch/blob/8bc04f46fe8e69188fa46f1611b46788a7d4824d/torch/fx/experimental/proxy_tensor.py#L51
optree.register_pytree_node(
    torch.Size,
    lambda size: (list(size), None, None),
    lambda _, children: tuple(children),
    namespace=optree.registry.__GLOBAL_NAMESPACE,
)

tree_flatten = partial(optree.tree_flatten, none_is_leaf=True)
tree_map = partial(optree.tree_map, none_is_leaf=True)


def tree_unflatten(values, spec):
    return optree.tree_unflatten(spec, values)


__all__ = ["tree_flatten", "tree_unflatten", "tree_map"]
