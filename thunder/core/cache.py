import inspect
from inspect import Parameter
import string
from collections import deque
from collections.abc import Mapping, Sequence
from typing import List, Dict, Set
from numbers import Number

import thunder.core.utils as utils

__all__ = [
    "build_cache",
]

# TODO: merge this logic with the flatten/unflatten logic of the generated fusions to avoid
#   paying that cost

# NOTE: tabs in the generated program will be huge without this
tab = "  "


# TODO: support more collections
# NOTE: doesn't test if something is a sequence, because strings are
#   sequences in Python but we don't want to treat them as collections
def is_collection(x):
    return isinstance(x, (list, dict, set))


# TODO: review what acceptable keys are better
# NOTE: what's scary is that someone could pass a dictionary with
#   tensors as keys, and we would keep those tensors alive by holding
#   references to them in our cache just so we could check if they
#   were passed back
def is_valid_key(x):
    return isinstance(x, (Number, str))


def extract_collection(q, ex, name, x, name_generator):
    if isinstance(x, Mapping):
        for k, v in x.items():
            utils.check(is_valid_key(k), lambda: f"Found invalid key={k}; keys must be numbers or strings")
            item_name = name_generator()
            # TODO: escape any ' characters in the string k (if it's a string)
            lookup = f"'{k}'" if isinstance(k, str) else k
            ex.append(f"{tab}{item_name} = {name}[{lookup}]")
            q.append((item_name, v))
    elif isinstance(x, Sequence):
        for idx, v in enumerate(x):
            item_name = name_generator()
            ex.append(f"{tab}{item_name} = {name}[{idx}]")
            q.append((item_name, v))
    else:
        raise NotImplementedError


# TODO: generate additional metadata checks beyond type
# TODO: don't check type by string, by compare directly (requires saving the type)
def generate_comparisons(lhs, rhs, name, x):
    lhs.append(f"str(type({name}))")
    rhs.append(f'"{str(type(x))}"')


def build_cache(fn, args, kwargs, fusion):
    chars = tuple(string.ascii_lowercase)
    global_ctr = 0

    def _gen_name():
        global global_ctr
        ctr = global_ctr
        place = 0
        s = ""
        while ctr >= 0:
            if place > 0:
                ctr = ctr // (place * len(chars))
            idx = ctr % (len(chars))
            c = chars[idx]
            s = c + s
            ctr = ctr - (idx + 1 + place * len(chars))
            place += 1

        # NOTE: adds "__" to avoid collision with keywords
        # TODO: improve naming to avoid conflicts
        global_ctr += 1
        return "__" + s

    sig = inspect.signature(fn)
    ba = sig.bind(*args, **kwargs)

    # Creates args augmented with default values
    args_dict = {k: v.default for k, v in sig.parameters.items()}
    args_dict.update(ba.arguments)

    q = deque()
    leaves = deque()
    comparisons = deque()
    extractions = deque()

    for name, x in args_dict.items():
        if is_collection(x):
            q.append((name, x))
        else:
            leaves.append((name, x))

    # If the current item is a collection, adds its items to the q
    # If the item is a leaf (not a collection), adds it to leaves
    while len(q) > 0:
        name, x = q.popleft()
        if is_collection(x):
            extract_collection(q, extractions, name, x, name_generator=_gen_name)
        else:
            leaves.append((name, x))

    lhs = []
    rhs = []
    for name, x in leaves:
        generate_comparisons(lhs, rhs, name, x)

    extract_str = f"\n".join(extractions)

    # TODO: make indentation easier to work with
    lhs_str = f",\n{tab * 2}".join(lhs)
    rhs_str = f",\n{tab * 2}".join(rhs)
    compare_str = f"{tab}if (\n{tab * 2}{lhs_str}\n{tab}) == (\n{tab * 2}{rhs_str}):"
    compare_str += f"\n{tab * 2}return fusion"
    compare_str += f"\n{tab}else:"
    compare_str += f"\n{tab * 2}return None"

    # generates the signature
    params = []
    for k, v in sig.parameters.items():
        # *args
        if v.kind == Parameter.VAR_POSITIONAL:
            params.append(f"*{k}")
        elif v.kind == Parameter.VAR_KEYWORD:
            params.append(f"**{k}")
        elif v.default is not Parameter.empty:
            params.append(f"{k}={v.default}")
        else:
            params.append(k)

    param_str = ", ".join(params)
    csig = f"def fn({param_str}):"

    if len(extract_str) > 0:
        cstr = f"{csig}\n{tab}try:\n{extract_str}\n{tab}except:\n{tab * 2}return None\n{compare_str}"
    else:
        cstr = f"{csig}\n{compare_str}"

    ctx = {
        "fusion": fusion,
    }
    code = compile(cstr, "thunder.gen", mode="exec")
    exec(code, ctx)
    fn = ctx["fn"]

    return fn
