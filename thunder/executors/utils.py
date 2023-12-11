from __future__ import annotations

from enum import Enum, auto
from typing import List, Set, Dict, Optional
from collections.abc import Callable
from itertools import chain
from collections.abc import Sequence

import torch
from looseversion import LooseVersion

import thunder.core.utils as utils
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.proxies import Variable, variableify, Proxy, unvariableify
from thunder.core.prims import PrimIDs

# TODO Make these tags
comment_symbols = {
    PrimIDs.COMMENT,
    PrimIDs.UNPACK_TRIVIAL,
    PrimIDs.UNPACK_EMPTY_DICT,
}


# TODO Document this better
# TODO Review non-proxy inputs as being consumed -- currently only proxies can be inputs and outputs of these regions
class Region:
    def __init__(self, producers, consumers, bound_symbols: list[BoundSymbol]):
        # Stores input data
        self.bound_symbols = bound_symbols

        # Identifies inputs and outputs
        # NOTE Inputs and outputs are "variableified" sets
        consumes = set()
        produces = set()

        for bsym in self.bound_symbols:
            flatouts = bsym.flat_outs

            produces.update(
                variableify(x) for x in flatouts if isinstance(x, Proxy) and producers[x] in self.bound_symbols
            )

            # Short-circuits if the symbol is a comment, because comments don't consume anything
            #   Note that comments may produce things
            if bsym.sym.id in comment_symbols:
                continue

            # Updates what this region consumes, skipping symbols that never consume anything
            consumes.update(variableify(x) for x in bsym.flat_args if isinstance(x, Proxy))

        self.inputs = set()
        self.outputs = set()

        # Inputs are things which this consumes which are produced before it
        for x in consumes:
            x = unvariableify(x)

            if producers[x] not in self.bound_symbols:
                self.inputs.add(variableify(x))

        # Outputs are things this produces that are consumed after it
        for x in produces:
            x = unvariableify(x)
            consumed_by = consumers.get(x, ())
            for bsym in consumed_by:
                if bsym not in self.bound_symbols:
                    self.outputs.add(variableify(x))
                    break

    def __repr__(self) -> str:
        s = f"[Region:"

        for bsym in self.bound_symbols:
            s += f"\n{str(bsym)}"

        s += "]"

        return s
