from __future__ import annotations

from collections.abc import Iterable
import itertools
import textwrap
from typing import Generic, ParamSpec, TypeVar, cast

import networkx as nx
from typing_extensions import Self

from thunder.core.utils import OrderedSet

__all__ = ("flatten_map", "sort_adjacent", "compute_condense_map")
P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# == nx.(Di)Graph, but with more safety =======================================
# =============================================================================
class TypedGraph(nx.Graph, Generic[T]):  # type: ignore[misc]
    def __init__(self, edgelist: Iterable[tuple[T, T]] = ()) -> None:
        super().__init__()
        self.add_edges_from(edgelist)

    @property
    def nodes(self) -> Iterable[T]:
        return cast(Iterable[T], super().nodes)

    @property
    def edges(self) -> Iterable[tuple[T, T]]:
        return cast(Iterable[tuple[T, T]], super().edges)

    @property
    def connected_components(self) -> Iterable[set[T]]:
        return cast(Iterable[set[T]], nx.connected_components(self))

    def subgraph(self, nodes: Iterable[T]) -> Self:
        return cast(Self, super().subgraph(nodes))

    def to_undirected_class(self) -> type:
        return TypedGraph[T]

    def to_directed_class(self) -> type:
        return TypedDiGraph[T]


class TypedDiGraph(TypedGraph[T], nx.DiGraph):  # type: ignore[misc]
    def assert_directed_acyclic(self) -> None:
        if not nx.is_directed_acyclic_graph(self):
            cycle = "\n".join(f"{node}" for node, _ in nx.find_cycle(self))
            raise AssertionError(f"Cycle detected:\n{textwrap.indent(cycle, ' ' * 4)}")

    def to_undirected(self, *args: P.args, **kwargs: P.kwargs) -> TypedGraph[T]:
        G = super().to_undirected(*args, **kwargs)
        assert isinstance(G, TypedGraph)
        return G

    def predecessors(self, n: T) -> Iterable[T]:
        return cast(Iterable[T], super().predecessors(n))


# =============================================================================
# == Graph algorithms =========================================================
# =============================================================================
def flatten_map(mapping: dict[T, T]) -> Iterable[tuple[T, T]]:
    """If one interprets the items as edges of a tree (or forest), return items for a tree of at most depth two.

    For example, `{1: 2, 2: 3, 4: 3, 5: 6}` flattens to `{1: 3, 2: 3, 4: 3, 5: 6}`.
    """
    G = TypedDiGraph[T](((j, i) for i, j in mapping.items() if i != j))
    assert nx.is_directed_acyclic_graph(G)
    for cluster in nx.connected_components(G.to_undirected()):
        (root,) = (i for i in cluster if not G.in_degree(i))
        yield from ((i, root) for i in cluster if i != root)


def _extract_paths(G: TypedDiGraph[T]) -> Iterable[tuple[T, ...]]:
    assert nx.is_connected(G.to_undirected())
    subgraph = TypedDiGraph[T]()
    subgraph.add_nodes_from(G.nodes)
    subgraph.add_edges_from(edge for edge, adjacent in nx.get_edge_attributes(G, "adjacent").items() if adjacent)
    assert len(subgraph) == len(G)
    subgraph.assert_directed_acyclic()

    for nodes in subgraph.to_undirected().connected_components:
        path = subgraph.subgraph(nodes)
        sorted_nodes = tuple(nx.topological_sort(path))
        assert len(sorted_nodes) == 1 or nx.is_simple_path(path, sorted_nodes)
        yield sorted_nodes


def sort_adjacent(G: TypedDiGraph[T]) -> Iterable[T]:
    """Sort nodes, respecting strong adjacency requirements and trying to sort return blocks to the end.

    If edges are annotated with `is_return` the annotation will be used; otherwise
    terminal nodes will be inferred.
    """
    (root,) = (node for node in G.nodes if not G.in_degree(node))
    is_return = nx.get_node_attributes(G, "is_return") or {node: not G.out_degree(node) for node in G.nodes}
    assert len(is_return) == len(G) and any(is_return.values())

    sort_map = {}
    for primary_key, sorted_nodes in enumerate(_extract_paths(G)):
        if sorted_nodes[0] is root:
            primary_key = -1
        elif is_return[sorted_nodes[-1]]:
            primary_key += len(G)

        sort_map.update({node: (primary_key, idx) for idx, node in enumerate(sorted_nodes)})

    assert len(sort_map) == len(G)
    yield from sorted(sort_map, key=lambda node: sort_map[node])


def sort_adjacent_dfs(G: TypedDiGraph[T]) -> Iterable[T]:
    """Alternate sorting formulation. Prioritizes program order over moving returns to the end.

    Unlike `sort_adjacent`, this order guarantees that at least one dependency will have
    appeared before the current block. (`undo_ssa` seems to depend on this invariant.)
    """
    paths = {sorted_nodes[0]: sorted_nodes for sorted_nodes in _extract_paths(G)}

    condensed = {}
    for path_root, path in paths.items():
        condensed.update({node: path_root for node in path})

    G_traverse = TypedDiGraph[T]((condensed[source], condensed[sink]) for source, sink in G.edges)
    G_traverse.add_nodes_from(paths)
    for i in nx.dfs_preorder_nodes(G_traverse):
        yield from paths.pop(i)

    assert not paths


def compute_condense_map(edges: Iterable[tuple[T, T]]) -> dict[T, OrderedSet[T]]:
    """Given a graph of identity relations (including unions and cycles), determine a minumum basis.

    A common construct that emerges from program loops is the statement "A is either A or B". However
    if we eliminate the vacuous "A is A" component we reach the much more useful "A is B", which
    allows us to replace a thorny union with a simple value. Similarly, we can eliminate chains of
    equality expressions. ("C is B, B is A" becomes "C is A, B is A")

    At first this seems as simple as finding the roots of the graph, but consider the following:
    "B is A, C is either B or D". B can be replaced with A, but C is the union of A and D. Critically,
    B is NOT D, so simply assigning all non-roots the union of the roots is incorrect.

    This function uses an iterative method to distil the graph. Note that there is significant
    simplification; the input can be an arbitrary directed **cyclic** graph (as long as at least one
    node is not part of a cycle), but the output constituents are trees of at most depth two.
    """
    G = TypedDiGraph(edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    condense_map: dict[T, OrderedSet[T]] = {node: OrderedSet() for node in G}
    for subgraph_nodes in G.to_undirected().connected_components:
        subgraph = cast(TypedDiGraph[T], G.subgraph(subgraph_nodes))
        roots = OrderedSet(node for node in subgraph_nodes if not subgraph.in_degree(node))
        assert roots, subgraph.edges

        equality_edges = OrderedSet((node, node) for node in subgraph.nodes)
        while True:
            # Condense pairs in `equality_edges`. For example, given the
            # following graph and `equality_edges`:
            #   0 → 1 → 2 → 3 → 4 → 5
            #               ↑┄──┘
            #
            #   equality_edges = {(0, 1), (3, 4)}
            #
            # After grouping we're left with:
            #   {0, 1} → 2 → {3, 4} → 5
            clusters: dict[T, T] = {}
            for cluster in TypedGraph(equality_edges).connected_components:
                # The choice of "canonical" value is arbitrary as long as it is consistent.
                canonical = next(iter(cluster))
                clusters.update((i, canonical) for i in cluster)

            assert len(clusters) == len(subgraph)
            reduced_edges = ((clusters[i], clusters[j]) for i, j in subgraph.edges)
            reduced_subgraph = cast(TypedDiGraph[T], TypedDiGraph[T](reduced_edges))  # MyPy can't figure this out...
            reduced_subgraph.remove_edges_from(nx.selfloop_edges(reduced_subgraph))
            num_equality_edges = len(equality_edges)

            # Condense chains.
            equality_edges.update(reduced_subgraph.edges)

            # Condense loops.
            for cycle in nx.simple_cycles(reduced_subgraph):
                equality_edges.update(zip(cycle, itertools.chain(cycle[1:], cycle[:1])))

            if len(equality_edges) == num_equality_edges:
                # No progress has been made, exit loop.
                break

        for root in roots:
            for reachable in itertools.chain([root], *nx.dfs_successors(subgraph, root).values()):
                condense_map[reachable].add(root)

    return condense_map
