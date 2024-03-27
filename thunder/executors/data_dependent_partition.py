from __future__ import annotations

from typing import List, Set
from collections.abc import Callable
from copy import copy
from itertools import chain

import thunder.core.utils as utils
from thunder.core.trace import TraceCtx
from thunder.core.symbol import BoundSymbol
from thunder.core.proxies import variableify, Proxy
from thunder.core.prims import PrimIDs


# Represents a region and its parents (regions it consumes the output of) and
#   children (regions that consume its output)
#   Essentially creates a directional graph of regions showing their
#   producer-consumer relationships.
# TODO These classes could be refactored to be region-independent in their logic
class Node:
    def __init__(self, ID: int, group_bsyms: list[BoundSymbol], group_indices: list[int], start: int, stop: int):
        self.ID = ID
        self.start = start
        self.stop = stop
        self.group_bsyms = group_bsyms
        self.group_indices = group_indices
        self.parents: set[Node] = set()
        self.children: set[Node] = set()

    def __repr__(self) -> str:
        s = f"node ID {self.ID} : "
        s += self.group_bsyms.__repr__()
        s += f"\n\tparents ids: "
        for parent in self.parents:
            s += f" {parent.ID}, "
        s += f"\n\tchildren ids: "
        for child in self.children:
            s += f" {child.ID}, "
        s += f"\n"
        return s

    def __hash__(self) -> int:
        return self.ID

    def __eq__(self, other) -> bool:
        return self.ID == other.ID

    @staticmethod
    def merge(a: Node, b: Node):
        merged_bsyms = []
        merged_indices = []

        def push_node(n: Node, index: int):
            nonlocal merged_bsyms
            nonlocal merged_indices
            merged_bsyms.append(n.group_bsyms[index])
            merged_indices.append(n.group_indices[index])

        a_index = 0
        b_index = 0
        while a_index < len(a.group_bsyms):
            if b_index < len(b.group_bsyms) and b.group_indices[b_index] < a.group_indices[a_index]:
                push_node(b, b_index)
                b_index += 1
            else:
                push_node(a, a_index)
                a_index += 1

        while b_index < len(b.group_bsyms):
            push_node(b, b_index)
            b_index += 1

        return merged_bsyms, merged_indices


# assumes bound_symbol comes in as a DAG and in valid topo order
# NOTE: consolidate graph implementations, we have several almost identical
# implementations already
class Graph:
    def __init__(self, trace: TraceCtx):
        self.roots: list[Node] = []
        self.return_node: None | Node = None
        self.counter = len(trace.bound_symbols)

        producers = utils.producers(trace, _map_to_numbers=True)
        consumers = utils.consumers(trace, _map_to_numbers=True)

        # Note, even though BoundSymbolInterface is hashable, it's hash is very slow
        # as it appears to be far off from being universal.
        # We use indices as hash values instead.
        bsym_id_to_node_map: list[int] = []
        for bsym_id, bsym in enumerate(trace.bound_symbols):
            node = Node(bsym_id, [bsym], [bsym_id], bsym_id, bsym_id)
            bsym_id_to_node_map.append(node)

            if bsym.sym.id is PrimIDs.RETURN:
                utils.check(
                    self.return_node is None,
                    lambda: f"Found multiple RETURN nodes while converting a list of bound symbols to a dag",
                )
                self.return_node = node

        for bsym_id, node in enumerate(bsym_id_to_node_map):
            has_parents: bool = False

            bsym = node.group_bsyms[0]
            for inp in bsym.flat_args:
                if not isinstance(inp, Proxy):
                    continue

                producer_id = producers[inp]
                parent = bsym_id_to_node_map[producer_id]
                node.parents.add(parent)
                has_parents = True

            if not has_parents:
                self.roots.append(node)

            for out in bsym.flat_outs:
                if not isinstance(out, Proxy):
                    continue

                # Checks that the output is actually produced by this function, and not an input to it
                if variableify(out) in (variableify(x) for x in bsym.flat_args):
                    continue

                children_ids = consumers.get(out, [])
                for child_id in children_ids:
                    child_node = bsym_id_to_node_map[child_id]
                    node.children.add(child_node)

    def __repr__(self) -> str:
        s = f"graph roots:"
        for root in self.roots:
            s += f" {root.ID},"
        s += "\ntraversal nodes:\n"
        visit_stack = list(self.roots)
        visited = set()
        while visit_stack:
            cur = visit_stack.pop(0)
            if cur in visited:
                continue
            s += cur.__repr__()
            visited.add(cur)
            visit_stack.extend(cur.children)
            for child in cur.children:
                assert cur in child.parents
        return s

    # merge consumer `b` into producer `a`
    def merge(self, a: Node, b: Node) -> bool:
        ##############################
        # step0: cyclic check
        ##############################
        max_depth = max(a.stop, b.stop)

        if len(b.parents) != 1 or not a in b.parents:
            visit_stack = list()
            visit_stack.extend([x for x in a.children if x != b])
            visit_stack.extend([x for x in b.children if x != a])

            visited = set()
            while visit_stack:
                cur = visit_stack.pop(0)
                if cur in visited:
                    continue
                if cur in [a, b]:
                    # cycle detected, do nothing and return False
                    return False
                visited.add(cur)
                if cur.start <= max_depth:
                    visit_stack.extend(cur.children)

        ##############################
        # step1: merge the two nodes together
        ##############################

        # create a new_node as the merged node with combined bsyms from a and b

        min_start = min(a.start, b.start)
        merged_bsyms, merged_indices = Node.merge(a, b)
        new_node = Node(self.counter, merged_bsyms, merged_indices, min_start, max_depth)
        self.counter = self.counter + 1

        # TODO: this part is slow! we might want to refactor this section and do one merge with a group, instead of do rewiring for every single pair
        for parent in a.parents.union(b.parents):
            if parent is a or parent is b:
                continue
            parent.children.discard(a)
            parent.children.discard(b)
            parent.children.add(new_node)
            new_node.parents.add(parent)

        for child in a.children.union(b.children):
            if child is a or child is b:
                continue
            child.parents.discard(a)
            child.parents.discard(b)
            child.parents.add(new_node)
            new_node.children.add(child)

        if a in self.roots:
            # we want to put new_node at the same spot, i.e. # args: "Collection" would want to stay at where it was before the merge
            if a.parents.union(b.parents).issubset({a, b}):
                self.roots[self.roots.index(a)] = new_node
            else:
                self.roots.remove(a)

        return new_node


# NOTE this function modifies `graph` and merges node in place
def dataflow_merge(graph, merge_func: Callable):
    # do_while loop flag
    do_while = True
    while do_while:
        do_while = False

        visit_stack = list()
        visit_stack.extend(graph.roots)

        visited = set()

        while len(visit_stack) > 0:
            cur_node = visit_stack.pop(0)

            if cur_node in visited:
                continue

            merge_children = True
            while merge_children:
                merge_children = False

                children = copy(cur_node.children)
                for child in children:
                    # merge_func checks if we do intend to merge the two nodes
                    # Graph.merge only returns true if merged node is still a DAG, otherwise it's a no-op
                    if merge_func(cur_node, child):
                        n = graph.merge(cur_node, child)
                        if n:
                            # add child to visited. since we don't want to go remove it from visit_stack
                            visited.add(child)
                            visited.add(cur_node)
                            do_while = True
                            merge_children = True
                            cur_node = n

            visited.add(cur_node)
            visit_stack.extend(cur_node.children)


def horizontal_merge(graph, merge_func: Callable):
    # TODO: might as well handle toposort and horizontal merge here
    visit_stack = list(graph.roots)
    # book keeping for remaining dependencies
    candidate_map = dict()
    topo_order_groups = list()

    while visit_stack:
        # takes the first node in the stack.
        cur = visit_stack.pop(0)
        bsyms_in_group = []

        def update_candidate(schedule_op):
            nonlocal bsyms_in_group
            nonlocal candidate_map
            nonlocal visit_stack
            bsyms_in_group += schedule_op.group_bsyms
            # iterate through candidate_map and update predicate for schedule_op.children
            for candidate in schedule_op.children:
                remaining_dependencies = candidate_map.setdefault(candidate, len(candidate.parents))
                if remaining_dependencies == 1:
                    candidate_map.pop(candidate)
                    visit_stack.append(candidate)
                else:
                    candidate_map[candidate] = remaining_dependencies - 1

        update_candidate(cur)
        index = 0
        while index < len(visit_stack):
            if merge_func(cur, visit_stack[index]):
                # NOTE: we are not mutating the graph
                update_candidate(visit_stack.pop(index))
            else:
                index += 1

        topo_order_groups.append(bsyms_in_group)

    return topo_order_groups


def fuse_bound_symbols(trace: TraceCtx, merge_func: Callable):
    graph = Graph(trace)
    dataflow_merge(graph, merge_func)
    ret = horizontal_merge(graph, merge_func)
    return ret
