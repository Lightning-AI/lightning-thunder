from functools import partial
from types import FunctionType
import dataclasses

import optree
import torch
import thunder.core.dtypes as dtypes
import thunder.core.devices as devices
from thunder.core.baseutils import ProxyInterface, is_likely_from_collections_namedtuple
from types import FunctionType

OPTREE_NAMESPACE = "thunder"

# We need torch.Size to be treated the same way as a list or tuple
# In PyTorch this is registered here:
# https://github.com/pytorch/pytorch/blob/8bc04f46fe8e69188fa46f1611b46788a7d4824d/torch/fx/experimental/proxy_tensor.py#L51
optree.register_pytree_node(
    torch.Size,
    lambda size: (list(size), None, None),
    lambda _, children: tuple(children),
    namespace=OPTREE_NAMESPACE,
)


optree.register_pytree_node(
    slice,
    lambda s: ([s.start, s.stop, s.step], None, None),
    lambda _, children: slice(*children),
    namespace=OPTREE_NAMESPACE,
)


def tree_flatten(args, namespace=OPTREE_NAMESPACE):
    if (
        type(args)
        not in {
            FunctionType,
            dict,
            list,
            str,
            int,
            bool,
            tuple,
            torch.dtype,
            float,
            dtypes.floating,
            dtypes.bool_,
            devices.Device,
            torch.memory_format,
            type(None),
            slice,
            complex,
            type,
            type(Ellipsis),
            torch.Size,
            torch.finfo,
            dtypes.signedinteger,
            # FakeTensor type is used for automatic registration of torch ops
            torch._subclasses.fake_tensor.FakeTensor,
            torch.device,
            torch.autograd.function.FunctionCtx,
        }
        and not isinstance(args, (ProxyInterface))
        and not is_likely_from_collections_namedtuple(args)
        and not dataclasses.is_dataclass(args)
        and not type(args).__module__.startswith("torch.return_types")
    ):
        raise TypeError(f"tree_flatten of type {type(args)} is not supported.")
    return optree.tree_flatten(args, none_is_leaf=True, namespace=namespace)


# This is required in the `torch_autograd` part of the code where we split forward and backward fn.
# We want to be able to inspect `dataclass` containers to see if they contain proxy
# while generating the split functions.
tree_map = partial(optree.tree_map, none_is_leaf=True, namespace=OPTREE_NAMESPACE)

tree_iter = partial(optree.tree_iter, none_is_leaf=True, namespace=OPTREE_NAMESPACE)


def tree_unflatten(values, spec):
    return optree.tree_unflatten(spec, values)


_registered_dataclasses = set()


def register_pytree_node_dataclass(cls):
    # We don't use `dataclasses.asdict` as it recursively flattens all data classes (including
    # thunder internal ones like `VJPDual` (and also it is relatively slower as it calls copy.deepcopy()).
    def unpack(cls) -> dict:
        return {field.name: getattr(cls, field.name) for field in dataclasses.fields(cls)}

    _flatten = lambda obj: tree_flatten(unpack(obj), namespace=OPTREE_NAMESPACE)
    _unflatten = lambda spec, children: cls(**spec.unflatten(children))
    optree.register_pytree_node(cls, _flatten, _unflatten, namespace=OPTREE_NAMESPACE)
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

    return tree_flatten(tree, namespace=OPTREE_NAMESPACE)


__all__ = ["tree_flatten", "tree_unflatten", "tree_map", "tree_flatten_with_dataclass"]
