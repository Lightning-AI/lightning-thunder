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
            flatouts = bsym._flat_outs

            produces.update(
                variableify(x) for x in flatouts if isinstance(x, Proxy) and producers[x] in self.bound_symbols
            )

            # Short-circuits if the symbol is a comment, because comments don't consume anything
            #   Note that comments may produce things
            if bsym.sym.id in comment_symbols:
                continue

            # Updates what this region consumes, skipping symbols that never consume anything
            consumes.update(variableify(x) for x in bsym._flat_args if isinstance(x, Proxy))
            consumes.update(variableify(x) for x in bsym._flat_kwargs if isinstance(x, Proxy))

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

        # Set of shape operations to support determining if the region is composed of only shape operations
        # TODO This should have more operations, and should handle symbols that decompose to shape primitives
        # TODO Shape primitives should be programmatically queryable from prims
        self._shape_ops = {PrimIDs.SLICE, PrimIDs.TRANSPOSE, PrimIDs.RESHAPE}

    def __repr__(self) -> str:
        s = f"[Region:"

        for bsym in self.bound_symbols:
            s += f"\n{str(bsym)}"

        s += "]"

        return s

    def only_shape_operations(self) -> bool:
        for bsym in self.bound_symbols:
            if bsym.sym.id not in self._shape_ops:
                return False

        return True


# # Group bookend meta operations into separate regions
# # This function returns a List[Region] which changes the executor of meta regions to torchex
# #
# # NOTE this function assumes bound_symbols in region is toposorted
# def group_bookend_meta_ops(region: Region, producers, consumers) -> list[Region]:
#     # use TorchEx as meta_executor for meta regions
#     meta_executor = TorchEx

#     front_meta_cluster = list()
#     middle_cluster = list()
#     rear_meta_cluster = list()
#     region_inputs = copy(region.inputs)

#     # bsym can be moved to the front if all their inputs are direct region inputs
#     def can_move_to_front(bsym: BoundSymbol) -> bool:
#         # non proxy don't need to be checked here.
#         for x in chain(bsym._flat_args, bsym._flat_kwargs):
#             if not isinstance(x, Proxy):
#                 continue

#             if variableify(x) not in region_inputs:
#                 return False

#         return True

#     # when bsym has no consumer in current region, it can be safely moved to the rear
#     def can_move_to_rear(bsym: BoundSymbol) -> bool:
#         # check no existing bsym in region depends on current bsym
#         for out in bsym._flat_outs:
#             if not isinstance(out, Proxy):
#                 continue

#             consumed_by = consumers.get(out, list())
#             for consumer in consumed_by:
#                 # TODO: switch query to set for faster query
#                 if consumer in middle_cluster:
#                     return False
#         return True

#     # traversing all bound_symbols in topo order
#     for bsym in region.bound_symbols:
#         # we look at meta operations that can be moved to the front
#         if bsym.sym.id in region._shape_ops and can_move_to_front(bsym):
#             # when we remove a node, we add all the bsym's _flat_outs to region_inputs
#             front_meta_cluster.append(bsym)
#             for out in bsym._flat_outs:
#                 if isinstance(out, Proxy):
#                     region_inputs.add(variableify(out))
#         else:
#             # otherwise we just keep the bound_symbol in the middle_cluster
#             middle_cluster.append(bsym)

#     # traversing all bound_symbols in reverse topo order
#     for bsym in reversed(copy(middle_cluster)):
#         if bsym.sym.id in region._shape_ops and can_move_to_rear(bsym):
#             middle_cluster.remove(bsym)
#             # NOTE that rear_meta_cluster is in reverse topo order
#             rear_meta_cluster.append(bsym)

#     ret = list()
#     # check and construct each region
#     if len(front_meta_cluster) > 0:
#         ret.append(Region(producers, consumers, front_meta_cluster, meta_executor, -1))
#     if len(middle_cluster) > 0:
#         ret.append(Region(producers, consumers, middle_cluster, region.executor, -1))
#     if len(rear_meta_cluster) > 0:
#         ret.append(Region(producers, consumers, list(reversed(rear_meta_cluster)), meta_executor, -1))

#     return ret
