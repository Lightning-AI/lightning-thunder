import networkx as nx
import pytest

from thunder.core.script.algorithms import (
    compute_condense_map,
    flatten_map,
    sort_adjacent,
    sort_adjacent_dfs,
    TypedDiGraph,
)


@pytest.mark.parametrize(
    "mapping, expected",
    (
        ({1: 2, 2: 3}, {1: 3, 2: 3}),
        ({1: 2, 3: 2, 2: 4}, {1: 4, 2: 4, 3: 4}),
        ({"a": "b", "d": "e", "c": "d"}, {"a": "b", "c": "e", "d": "e"}),
        ({i: i + 1 for i in range(10)}, {i: 10 for i in range(10)}),
    ),
)
def test_flatten_map(mapping, expected):
    original = mapping.copy()
    result = dict(flatten_map(mapping))
    assert mapping == original
    assert result == expected


@pytest.mark.parametrize(
    "mapping",
    (
        {0: 1, 1: 0},
        {0: 1, 1: 2, 2: 3, 3: 1},
    ),
)
def test_flatten_map_detects_cycles(mapping):
    with pytest.raises(AssertionError):
        _ = dict(flatten_map(mapping))


def _str_to_edges(spec: str, delimiters: tuple[str, ...] = ("-", ".")):
    assert " " not in delimiters
    for delimiter in delimiters:
        spec = spec.replace(delimiter, f"{delimiter} ")

    for source, sink in zip(elements := spec.split(), elements[1:]):
        if source[-1] in delimiters:
            sink = sink[:-1] if sink[-1] in delimiters else sink
            yield source[:-1], source[-1], sink


def _test_sort_adjacent(raw_edges, expected, should_end_with_return, sort_fn):
    next_map = {}
    G = TypedDiGraph[str]()
    for source, edge, sink in _str_to_edges(raw_edges):
        G.add_edge(source, sink, adjacent=(edge == "-"))
        if edge == "-":
            assert source not in next_map
            next_map[source] = sink

    order = tuple(sort_fn(G))
    assert " ".join(order) == expected.replace("-", " ")

    # Root is first (and singular), and last is a return.
    assert not G.in_degree(order[0])
    assert all(G.in_degree(i) for i in order[1:])
    if should_end_with_return:
        assert not G.out_degree(order[-1])

    # Nojump edges are sorted together.
    for idx, node in enumerate(order):
        if node in next_map:
            assert idx + 1 < len(order)
            assert next_map[node] == order[idx + 1], (node, next_map[node], order[idx + 1])


@pytest.mark.parametrize(
    "spec",
    (
        # `-` is an adjacent edge, `.` is a jump edge. Spaces separate chains.
        # Dash in `Expected` is purely for readability.
        # The definitions become cumbersome if we don't use this compact format.
        #
        # Test name             |   Graph definition        |   Expected        |   Expected (DFS)
        "Generic_0              |   0-3-4 0.1-2.4           |   0-3-4 1-2       |   ...",
        "Generic_1              |   0-3.4 0.1-2-4           |   0-3 1-2-4       |   ...",
        "Sort_stability_0       |   0.1.2.3.4               |   0 1 2 3 4       |   ...",
        "Sort_stability_1       |   0.2.3.4.1               |   0 2 3 4 1       |   ...",
        "Slightly_entangled     |   a-1-2-3-4-5 a.2.5       |   a-1-2-3-4-5     |   ...",
        "Multiple_returns_0     |   a-1.2.4 1.3 2-5.1       |   a-1 2-5 4 3     |   ...",
        "Multiple_returns_1     |   a-1.2.4 1-3 2-5.1       |   a-1-3 2-5 4     |   ...",
        "Nested_cycles_0        |   0-1-2 2.1 3.1 2.4 2.3-5 |   0-1-2 3-5 4     |   0-1-2 4 3-5",
        "Nested_cycles_1        |   0-1-2 2.1 2.3-5 2.4 3.1 |   0-1-2 3-5 4     |   ...",
        "Overlapping_cycles_0   |   0-1-2.4.1 1.3-5.2 5-6   |   0-1-2 4 3-5-6   |   0-1-2 3-5-6 4",
        "Overlapping_cycles_1   |   0-1.2 1.3.5-2 2.4.1 5.6 |   0-1 5-2 3 4 6   |   0-1 5-2 4 6 3",
        "Overlapping_cycles_2   |   0-1.3.5-2.4.1 5.6 1.2   |   0-1 3 5-2 4 6   |   0-1 3 5-2 6 4",
    ),
    ids=lambda spec: spec.split("|")[0].strip(),
)
def test_sort_adjacent(spec):
    name, raw_edges, expected, expected_dfs = [i.strip() for i in spec.split("|")]
    _test_sort_adjacent(
        raw_edges,
        expected,
        should_end_with_return=name not in ("Generic_0",),
        sort_fn=sort_adjacent,
    )

    _test_sort_adjacent(
        raw_edges,
        expected_dfs if expected_dfs != "..." else expected,
        should_end_with_return=not (name == "Generic_0" or name.startswith("Overlapping_cycles_")),
        sort_fn=sort_adjacent_dfs,
    )


@pytest.mark.parametrize(
    "raw_edges, E",
    (
        ("0-1 0-2", AssertionError),  #                 Unsortable
        ("0-1 2.1", (ValueError, AssertionError)),  #   Two roots
        ("0-1-2-0", (ValueError, AssertionError)),  #   Cycle
        ("0-1 2-3", (ValueError, AssertionError)),  #   Disconnected
    ),
)
@pytest.mark.parametrize("sort_fn", (sort_adjacent, sort_adjacent_dfs))
def test_sort_adjacent_raises(raw_edges, E, sort_fn):
    with pytest.raises(E):
        _test_sort_adjacent(raw_edges, "", True, sort_fn)


@pytest.mark.parametrize(
    "spec",
    (
        "Empty          |                               ",
        "Trivial        |   1-3 2-3                     1*, 3->(1, 2), 2*",
        "Simple_cycle   |   1-2-3-4-2                   1*, 2->(1), 3->(1), 4->(1)",
        "Branch_cycle   |   1-2-3-4-2 5-3               1*, 2->(1, 5), 3->(1, 5), 4->(1, 5), 5*",
        "Downstream     |   1-2-3-2-4 5-4               1*, 2->(1), 3->(1), 4->(1, 5), 5*",
        "Nested_cycle_0 |   1-2-3-2-4-2                 1*, 2->(1), 3->(1), 4->(1)",
        "Nested_cycle_1 |   1-2-3-2-4-2-5-6-2           1*, 2->(1), 3->(1), 4->(1), 5->(1), 6->(1)",
        "Long_cycle     |   1-2-3-4-5-6-7-8-3-10        1*, 2->(1), 3->(1), 4->(1), 5->(1), 6->(1), 7->(1), 8->(1), 10->(1)",
        "Disjoint       |   1-2-3 4-5-6                 1*, 2->(1), 3->(1), 4*, 5->(4), 6->(4)",
        "Disjoint_cycle |   1-2-3 4-5-6-5 7-8 9-8       1*, 2->(1), 3->(1), 4*, 5->(4), 6->(4), 7*, 8->(7, 9), 9*",
        "Adjacent_cycle |   a-b-c-d-e-b 1-b-c-b c-2     a*, b->(1, a), c->(1, a), d->(1, a), e->(1, a), 1*, 2->(1, a)",
        "Self_edge      |   a-b-b-b-c                   a*, b->(a), c->(a)",
        "Parallel_cycle |   a-b-c-b-d-b-e-b-f-h-i-b     a*, b->(a), c->(a), d->(a), e->(a), f->(a), h->(a), i->(a)",
    ),
    ids=lambda spec: spec.split("|")[0].strip(),
)
def test_compute_condense_map(spec):
    edge_spec, *_, expected = [i.strip() for i in spec.split("|")[1].split("    ")]
    condense_map = compute_condense_map((source, sink) for source, _, sink in _str_to_edges(edge_spec))
    basis = {k for k, v in condense_map.items() if len(v) == 1 and k in v}

    # Check that the result forms a shallow forest.
    assert basis or not edge_spec
    assert all(k in basis or all(vi in basis for vi in v) for k, v in condense_map.items())

    result = ", ".join(
        (f"{k}*" if k in basis else "".join((k, "->(", ", ".join(sorted(v)), ")")) for k, v in condense_map.items())
    )
    assert result == expected


@pytest.mark.parametrize(
    "spec, E",
    (
        ("No_root_0     |   0-1-2-3-0", AssertionError),
        ("No_root_1     |   0-1-2-3-0 4-3-0-4", AssertionError),
    ),
    ids=lambda spec: spec.split("|")[0].strip() if isinstance(spec, str) else spec,
)
def test_compute_condense_map_raises(spec, E):
    _, edge_spec = spec.split("|")
    with pytest.raises(E):
        compute_condense_map((source, sink) for source, _, sink in _str_to_edges(edge_spec.strip()))
