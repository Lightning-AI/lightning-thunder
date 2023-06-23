from __future__ import annotations

from enum import Enum, auto
from typing import List, Set, Dict, Callable, Optional
from collections.abc import Sequence

import torch
from looseversion import LooseVersion

import thunder.core.utils as utils
from thunder.core.symbol import BoundSymbol
from thunder.core.trace import TraceCtx, from_trace, TraceProvenance
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.proxies import Variable, variableify, Proxy, unvariableify
from thunder.core.prims import PrimIDs

# TODO Consider renaming this file to common.py?


class Executor(Enum):
    NVFUSER = auto()
    TORCH = auto()
    PYTHON = auto()


# NOTE This is here because we can only import the nvFuser executor conditional on its being available
def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def nvfuser_version() -> Optional[LooseVersion]:
    # Short-circuits if CUDA isn't available
    if not is_cuda_available():
        return None

    try:
        import nvfuser

        if hasattr(nvfuser, "version"):
            return LooseVersion(nvfuser.version())

        # NOTE: This import of nvFuser may or may not have version info
        return LooseVersion("0.0.0")
    except ImportError:
        pass

    try:
        # NOTE This import of nvFuser is so old it didn't have version info
        import torch._C._nvfuser as nvfuser

        return LooseVersion("0.0.0")
    except ImportError:
        pass

    # NOTE This occurs when both attempts at importing nvFuser failed
    return None


def required_nvfuser_version() -> LooseVersion:
    return LooseVersion("0.0.1")


# NOTE We require nvFuser version 0.0.1 or greater
def nvfuser_available() -> bool:
    v = nvfuser_version()
    return v is not None and v >= required_nvfuser_version()


comment_symbols = {
    PrimIDs.COMMENT,
    PrimIDs.UNPACK_TRIVIAL,
    PrimIDs.UNPACK_EMPTY_DICT,
}


# TODO Document this better
# TODO Review non-proxy inputs as being consumed -- currently only proxies can be inputs and outputs of these regions
class Region:
    def __init__(self, trace: TraceCtx, producers, consumers, bound_symbols: List[BoundSymbol], executor, counter: int):
        # Stores input data
        self.trace = trace
        self.bound_symbols = bound_symbols
        self.executor = executor
        self.counter = counter

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
            consumed_by = consumers[x]
            for bsym in consumed_by:
                if bsym not in self.bound_symbols:
                    self.outputs.add(variableify(x))
                    break

        # Set of shape operations to support determining if the region is composed of only shape operations
        # TODO This should have more operations, and should handle symbols that decompose to shape primitives
        # TODO Shape primitives should be programmatically queryable from prims
        self._shape_ops = {PrimIDs.SLICE, PrimIDs.TRANSPOSE, PrimIDs.RESHAPE}

    def __repr__(self) -> str:
        s = f"[Region executor={self.executor}, bound symbols:"

        for bsym in self.bound_symbols:
            s += f"\n{str(bsym)}"

        s += "]"

        return s

    def only_shape_operations(self) -> bool:
        for bsym in self.bound_symbols:
            if bsym.sym.id not in self._shape_ops:
                return False

        return True


# A container for region nodes that supports multiple parentless
#   "root" nodes from which a topological sort could start
class Graph:
    def __init__(self):
        self.roots: List[Node] = []

        self.producers: dict[Variable, Node] = {}
        self.consumers: dict[Variable, List[Node]] = {}


# Represents a region and its parents (regions it consumes the output of) and
#   children (regions that consume its output)
#   Essentially creates a directional graph of regions showing their
#   producer-consumer relationships.
# TODO These classes could be refactored to be region-independent in their logic
class Node:
    def __init__(self, region):
        self.region = region
        self.parents: List[Node] = []
        self.children: List[Node] = []


# Constructs a graph of regions (regions and edges representing their
#   producer-consumer relationships)
# NOTE It is assumed that the list of regions is provided in a
#   valid topological sort
def graph_from_regions(regions: List[Region]) -> Graph:
    # Constructs the graph, producers, and consumers map
    # NOTE Graph construction is facilitated by creating a mapping from
    #   proxies (wrapped as Variables) to the region (wrapped as Nodes)
    #    producing them
    producers: dict[Variable, Node] = {}
    consumers: dict[Variable, List[Node]] = {}
    g = Graph()
    for region in regions:
        node = Node(region)

        # Identifies this region's parents and establishes parent-child
        #   relationships
        # NOTE Doing this here relies on the assumption that the list of
        #   regions provided is a valid topological sort of the graph
        #   If this assumption can't be made then the identification
        #   of parents would need to happen after this initial iteration
        #   through the regions
        for inp in region.inputs:
            parent = producers[inp]
            node.parents.append(parent)
            parent.children.append(node)

            # Updates consumers mapping
            inp_consumers = consumers.get(inp, [])
            inp_consumers.append(node)
            consumers[inp] = inp_consumers

        # Adds nodes without parents as roots
        # NOTE In practice today every region will have the initial
        #   "unpacking" region as an ancestor, and there will only be
        #   one root for each graph
        if len(node.parents) == 0:
            g.roots.append(node)

        # Updates the producer mapping with this region's outputs
        #   (which may be consumed by subsequence regions)
        for output in region.outputs:
            producers[output] = node

    g.producers = producers
    g.consumers = consumers

    return g


# A very simple linked list node for use in linearizations
class LLNode:
    def __init__(self, node: Node, next: Optional[LLNode]):
        self.node = node
        self._next = next


# TODO Document this
class Linearization:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.root_node: Optional[LLNode] = None
        self.cur_node: Optional[LLNode] = None
        self.tail_node: Optional[LLNode] = None

    def append(self, Node):
        llnode = LLNode(Node, None)

        if self.root_node is None:
            self.root_node = llnode
            self.tail_node = llnode
        else:
            self.tail_node._next = llnode
            self.tail_node = llnode

    # TODO Update this to use a proper iter pattern
    def reset(self) -> None:
        self.cur_node = self.root_node
        return self.cur_node

    def next(self) -> Optional[Node]:
        if self.cur_node is None:
            return None

        self.cur_node = self.cur_node._next
        return self.cur_node

    def peek(self) -> Optional[Node]:
        return self.cur_node._next

    # Merges two regions
    # NOTE This assumes that b is topologically weakly "after"
    #   a (that is, a may be an ancestor of b, but
    #   b may not be an ancestor of a)
    # NOTE This mutates the graph and the regions, destroying
    #   b
    def merge(self, a: LLNode, b: LLNode):
        utils.check(a._next is b, lambda: f"Can only merge nodes that are adjacent to each other in the linearization")
        utils.check(
            self.cur_node is a, lambda: f"Can only merge nodes when the current iteration is pointed to the first node"
        )

        # Updates the linked list
        a._next = b._next

        ar = a.node.region
        br = b.node.region

        utils.check(
            ar.executor == br.executor,
            lambda: f"Expected {ar.executor=} to be the same as {br.executor=} when merging regions",
            exception_type=AssertionError,
        )

        ar.bound_symbols.extend(br.bound_symbols)

        # Updates inputs
        #   (b no longer needs inputs that are produced by a)
        # NOTE This relies on the assumption that only a
        #   may be an ancestor of b, and b may not
        #   be an ancestor of a
        for inp in br.inputs:
            self.graph.consumers[inp].remove(b.node)
            if inp not in ar.outputs:
                ar.inputs.add(inp)
                self.graph.consumers[inp].append(a.node)

        # Updates outputs
        #   (a no longer needs to output objects that were only
        #   consumed by b)
        for out in br.outputs:
            self.graph.producers[out] = a.node

        for out in ar.outputs:
            consumers = self.graph.consumers[out]
            if len(consumers) != 0:
                br.outputs.add(out)

        ar.outputs = br.outputs


# Topologically sorts the given graph using Kahn's algorithm
#   (see https://en.wikipedia.org/wiki/Topological_sorting)
#   with the "selector" parameter determining how nodes are removed
#   from the set of next candidates
#   NOTE selector should have the signature (last_added: Optional[Node], nodes: list[Node]) -> Node,
#       where it returns a Node from the list
#   NOTE Kahn's algorithm simply does not specify how nodes are removed
#       ("remove a node n from S"), so any selector will produce a valid
#       topo sort, but the selector can be used to create a sort with
#       a desired property.
#   NOTE Other methods of choosing a topological sort could be considered.
#       Greedily selecting which node should be next could lead to
#       suboptimal optimizations, especially if there are cases where
#       three nvFuser regions A B and C can be paired as
#       AB or BC, and the A BC sorting would be preferred.
#   NOTE This DESTROYS the given graph -- it could be changed
#       to not to do so
def toposort(graph: Graph, selector: Callable) -> Linearization:
    candidates: List[Node] = graph.roots

    l = Linearization(graph)

    last_added: Optional[Node] = None
    while len(candidates) > 0:
        n: Node = selector(last_added, candidates)

        candidates.remove(n)

        # Updates n's children
        for child in n.children:
            child.parents.remove(n)

            if len(child.parents) == 0:
                candidates.append(child)

        l.append(n)
        last_added = n

    return l
