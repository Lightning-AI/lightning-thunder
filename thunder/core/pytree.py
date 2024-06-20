from functools import partial

import dataclasses
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


DATACLASS_OPTREE_NAMESPACE = "thunder_dataclass"
_registered_dataclasses = set()


def register_pytree_node_dataclass(cls):
    _flatten = lambda obj: optree.tree_flatten(dataclasses.asdict(obj))
    _unflatten = lambda spec, children: cls(**spec.unflatten(children))
    optree.register_pytree_node(cls, _flatten, _unflatten, DATACLASS_OPTREE_NAMESPACE)
    return cls


def _maybe_register_dataclass(t):
    if dataclasses.is_dataclass(t) and t.__class__ not in _registered_dataclasses:
        return True
    return False


# `tree_flatten_with_dataclass` iterates over the tree and registers functions to flatten dataclass objects present in the `tree`.
# This is to facilitate peeking into the dataclass object to correctly get proxies when inspecting the BoundSymbols in the trace.
def tree_flatten_with_dataclass(tree):
    def dataclass_registry(t):
        if _maybe_register_dataclass(t):
            cls = t.__class__
            register_pytree_node_dataclass(cls)
            _registered_dataclasses.add(cls)
        return t

    # Register unseen dataclass instance, so that we can
    # flatten them to gather any proxies from them.
    tree = tree_map(dataclass_registry, tree)

    return tree_flatten(tree, namespace=DATACLASS_OPTREE_NAMESPACE)


def tree_unflatten(values, spec):
    return optree.tree_unflatten(spec, values)


__all__ = ["tree_flatten", "tree_unflatten", "tree_map", "tree_flatten_with_dataclass"]
