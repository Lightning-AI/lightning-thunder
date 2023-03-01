from typing import List, Dict, Tuple, Set, Deque
from numbers import Number
from collections import deque
from collections.abc import Mapping, Sequence, Iterable
import inspect
from inspect import Parameter
import string

import thunder.core.utils as utils


# TODO: support more collections -- like set
# NOTE: doesn't test if something is a sequence, because strings are
#   sequences in Python but we don't want to treat them as collections
def is_collection(x):
    return isinstance(x, (List, Tuple, Dict, CollectionInfo))


# TODO: review what acceptable keys are better
# NOTE: what's scary is that someone could pass a dictionary with
#   tensors as keys, and we would keep those tensors alive by holding
#   references to them in our cache just so we could check if they
#   were passed back
def is_valid_key(x):
    return isinstance(x, (Number, str))


class LeafInfo:
    def __init__(self, idx, *, name=None):
        self.name = name
        self.idx = idx


class CollectionInfo:
    def __init__(self, coll, *, name=None):
        self.coll = coll
        self.name = name


def unpack_collection(leaves, keys, x, *, name=None, name_generator=None):
    if isinstance(x, List):
        return unpack_list(leaves, keys, x, name=name, name_generator=name_generator)
    if isinstance(x, Tuple):
        return unpack_tuple(leaves, keys, x, name=name, name_generator=name_generator)
    if isinstance(x, dict):
        return unpack_dict(leaves, keys, x, name=name, name_generator=name_generator)

    raise ValueError(f"Unknown collection type {type(x)}")


def _unpack(leaves, keys, x, *, name=None, name_generator=None, is_key=False):
    if is_collection(x):
        return unpack_collection(leaves, keys, x, name=name, name_generator=name_generator)
    if is_key:
        return unpack_leaf(keys, x, name=name, name_generator=name_generator)
    return unpack_leaf(leaves, x, name=name, name_generator=name_generator)


def unpack(x, *, name=None, name_generator=None):
    leaves = []
    keys = []
    packinfo = _unpack(leaves, keys, x, name=name, name_generator=name_generator)

    return leaves, keys, packinfo


def pack_collection(leaves, packinfo):
    if isinstance(packinfo.coll, List):
        return pack_list(leaves, packinfo.coll)
    if isinstance(packinfo.coll, Tuple):
        return pack_tuple(leaves, packinfo.coll)
    if isinstance(packinfo.coll, dict):
        return pack_dict(leaves, packinfo.coll)

    raise ValueError(f"Unknown collection type {type(packinfo)}")


def pack(leaves, packinfo):
    if is_collection(packinfo):
        return pack_collection(leaves, packinfo)
    return pack_leaf(leaves, packinfo)


def unpack_code_collection(packinfo):
    if isinstance(packinfo.coll, List):
        return unpack_code_list(packinfo)
    if isinstance(packinfo.coll, Tuple):
        return unpack_code_tuple(packinfo)
    if isinstance(packinfo.coll, dict):
        return unpack_code_dict(packinfo)

    raise ValueError(f"Unknown collection type {type(packinfo)}")


def unpack_code(packinfo):
    utils.check(
        isinstance(packinfo, CollectionInfo), lambda: "Expected to receive a collection to generate the unpack code for"
    )
    utils.check(packinfo.name is not None, lambda: "Expected a named collection to generate unpack code")
    return unpack_code_collection(packinfo)


# leaf functions
def unpack_leaf(leaves, x, *, name=None, name_generator=None):
    name = name if name is not None else name_generator() if name_generator is not None else None
    if name is None:
        leaves.append(x)
    else:
        leaves.append((name, x))
    li = LeafInfo(len(leaves) - 1, name=name)
    return li


def pack_leaf(leaves, packinfo):
    return leaves[packinfo.idx]


# list functions
def unpack_list(leaves, keys, l, *, name=None, name_generator=None):
    packinfo = []

    for x in l:
        y = _unpack(leaves, keys, x, name_generator=name_generator)
        packinfo.append(y)

    name = name if name is not None else name_generator() if name_generator is not None else None
    return CollectionInfo(packinfo, name=name)


def pack_list(leaves, l):
    return [pack(leaves, x) for x in l]


def unpack_code_list(cinfo):
    names = []
    children = []

    l = cinfo.coll

    # Short-circuits if there's nothing to extract
    if len(l) == 0:
        return []

    for x in l:
        utils.check(x.name is not None, lambda: f"Found an unnamed item {x} while trying to build an extraction string")
        names.append(x.name)

        if is_collection(x):
            children.append(x)

    code = [f"{name}, \\" for name in names[:-1]]
    code.append(f"{names[-1]}, = {cinfo.name}")

    for child in children:
        code.extend(unpack_code_collection(child))

    return code


# tuple functions
def unpack_tuple(leaves, keys, tup, *, name=None, name_generator=None):
    packinfo = []

    for x in tup:
        y = _unpack(leaves, keys, x, name_generator=name_generator)
        packinfo.append(y)

    name = name if name is not None else name_generator() if name_generator is not None else None
    return CollectionInfo(tuple(packinfo), name=name)


def pack_tuple(leaves, tup):
    packed = [pack(leaves, x) for x in tup]
    return tuple(packed)


unpack_code_tuple = unpack_code_list


# dict functions
def unpack_dict(leaves, keys, d, *, name=None, name_generator=None):
    packinfo = {}

    for k, v in d.items():
        y = _unpack(leaves, keys, v, name_generator=name_generator)
        packinfo[k] = y
        keys.append(k)

    name = name if name is not None else name_generator() if name_generator is not None else None
    return CollectionInfo(packinfo, name=name)


def pack_dict(leaves, d):
    return {k: pack(leaves, v) for k, v in d.items()}


def unpack_code_dict(cinfo):
    names = []
    children = []

    d = cinfo.coll

    # Short-circuits if there's nothing to extract
    if len(d) == 0:
        return []

    for x in d.values():
        utils.check(x.name is not None, lambda: f"Found an unnamed item {x} while trying to build an extraction string")
        names.append(x.name)

        if is_collection(x):
            children.append(x)

    code = [f"{name}, \\" for name in names[:-1]]
    code.append(f"{names[-1]}, = {cinfo.name}.values()")

    for child in children:
        code.extend(unpack_code_collection(child))

    return code


class SigInfo:
    def __init__(self):
        self.args = []
        self.varargs = None
        self.kwargs = {}
        self.varkwargs = None

    def __repr__(self):
        return f"[SigInfo args={self.args}, varargs={self.varargs}, kwargs={self.kwargs}, varkwargs={self.varkwargs}]"


# Creates a SigInfo object from a function and the inputs to it
# The SigInfo object contains name and value information for the args, varargs, kwargs, and varkwargs
#   given to a function.
# To call a function foo from its SigInfo, you can do the following:
#
# arg_values = tuple(x[1] for x in si.args)
# if si.varargs is not None:
#     arg_values = arg_values + si.varargs[1]
# kwarg_values = si.kwargs
# if si.varkwargs is not None:
#     kwarg_values.update(si.varkwargs[1])
# foo(*arg_values, **kwarg_values)
#
# This removes the name information and combines the args and varargs into arg_values,
#   and the kwargs and varkwargs into kwarg_values
def get_siginfo(fn, args, kwargs):
    # Binds args and kwargs to signature
    sig = inspect.signature(fn)
    ba = sig.bind(*args, **kwargs)

    # Augments arguments with default values
    # NOTE: for example, alpha=1., if alpha is not specified
    #   explicitly then ba above will not contain it
    args_dict = {k: v.default for k, v in sig.parameters.items() if v.default is not Parameter.empty}
    args_dict.update(ba.arguments)

    # Augments the parameters with positional information
    params_with_indices = {k: (v, idx) for idx, (k, v) in enumerate(sig.parameters.items())}

    # Constructs signature information
    si = SigInfo()
    for name, x in args_dict.items():
        p, idx = params_with_indices[name]
        pkind = p.kind

        if pkind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            si.args.append((x, idx, name))
        elif pkind is Parameter.VAR_POSITIONAL:
            si.varargs = (name, x)
        elif pkind is Parameter.KEYWORD_ONLY:
            si.kwargs[name] = x
        elif pkind is Parameter.VAR_KEYWORD:
            si.varkwargs = (name, x)
        else:
            raise ValueError(f"Unexpected parameter kind {pkind}")

    si.args = sorted(si.args, key=lambda x: x[1])
    si.args = tuple((x[2], x[0]) for x in si.args)

    return si
