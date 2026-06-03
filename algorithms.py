from __future__ import annotations

import random
from collections import deque
from typing import Callable

import networkx as nx


Coloring = dict[int, int]


def color_count(coloring: Coloring) -> int:
    return len(set(coloring.values())) if coloring else 0


def count_conflicts(graph: nx.Graph, coloring: Coloring) -> int:
    return sum(1 for u, v in graph.edges() if coloring.get(u) == coloring.get(v))


def is_valid_coloring(graph: nx.Graph, coloring: Coloring) -> bool:
    return count_conflicts(graph, coloring) == 0


def greedy_welsh_powell(graph: nx.Graph) -> Coloring:
    nodes = sorted(graph.nodes(), key=lambda node: graph.degree[node], reverse=True)
    coloring: Coloring = {}
    for node in nodes:
        used = {coloring[nbr] for nbr in graph.neighbors(node) if nbr in coloring}
        color = 0
        while color in used:
            color += 1
        coloring[node] = color
    return coloring


def dsatur(graph: nx.Graph) -> Coloring:
    coloring: Coloring = {}
    uncolored = set(graph.nodes())

    while uncolored:
        def priority(node: int) -> tuple[int, int]:
            neighbor_colors = {
                coloring[nbr] for nbr in graph.neighbors(node) if nbr in coloring
            }
            return (len(neighbor_colors), graph.degree[node])

        node = max(uncolored, key=priority)
        used = {coloring[nbr] for nbr in graph.neighbors(node) if nbr in coloring}
        color = 0
        while color in used:
            color += 1
        coloring[node] = color
        uncolored.remove(node)

    return coloring


def tabu_search(
    graph: nx.Graph,
    initial_coloring_fn: Callable[[nx.Graph], Coloring] = dsatur,
    max_iterations: int = 2000,
    tabu_tenure: int = 7,
    seed: int | None = None,
) -> Coloring:
    rng = random.Random(seed)
    initial = initial_coloring_fn(graph)
    best_valid = dict(initial)
    best_color_count = color_count(best_valid)

    for target_k in range(best_color_count - 1, 0, -1):
        candidate = _search_k_coloring(
            graph=graph,
            k=target_k,
            max_iterations=max_iterations,
            tabu_tenure=tabu_tenure,
            rng=rng,
        )
        if candidate is None:
            break
        best_valid = candidate
        best_color_count = target_k

    return best_valid


def structure_aware_hybrid(
    graph: nx.Graph,
    seed: int | None = None,
    max_iterations: int = 2000,
    tabu_tenure: int = 7,
    density_threshold: float = 0.18,
    size_threshold: int = 220,
) -> Coloring:
    """Cost-aware hybrid: use DSATUR, then invoke Tabu only on compact dense graphs."""
    density = nx.density(graph)
    if graph.number_of_nodes() <= size_threshold and density >= density_threshold:
        return tabu_search(
            graph,
            initial_coloring_fn=dsatur,
            max_iterations=max_iterations,
            tabu_tenure=tabu_tenure,
            seed=seed,
        )
    return dsatur(graph)


def _search_k_coloring(
    graph: nx.Graph,
    k: int,
    max_iterations: int,
    tabu_tenure: int,
    rng: random.Random,
) -> Coloring | None:
    nodes = list(graph.nodes())
    coloring: Coloring = {node: rng.randrange(k) for node in nodes}
    tabu: deque[tuple[int, int]] = deque(maxlen=tabu_tenure)
    current_conflicts = count_conflicts(graph, coloring)
    best_conflicts = current_conflicts

    if best_conflicts == 0:
        return coloring

    for _ in range(max_iterations):
        conflicted_nodes = _conflicted_nodes(graph, coloring)
        if not conflicted_nodes:
            return coloring

        best_move = None
        best_move_conflicts = None

        rng.shuffle(conflicted_nodes)
        for node in conflicted_nodes:
            current_color = coloring[node]
            colors = list(range(k))
            rng.shuffle(colors)
            for new_color in colors:
                if new_color == current_color:
                    continue
                move = (node, new_color)
                if move in tabu:
                    continue
                conflicts = current_conflicts + _conflict_delta(
                    graph, coloring, node, current_color, new_color
                )
                if best_move_conflicts is None or conflicts < best_move_conflicts:
                    best_move = move
                    best_move_conflicts = conflicts

        if best_move is None:
            continue

        node, new_color = best_move
        old_color = coloring[node]
        coloring[node] = new_color
        current_conflicts = best_move_conflicts if best_move_conflicts is not None else current_conflicts
        tabu.append((node, old_color))

        if current_conflicts < best_conflicts:
            best_conflicts = current_conflicts

        if current_conflicts == 0:
            return dict(coloring)

    return None


def _conflicted_nodes(graph: nx.Graph, coloring: Coloring) -> list[int]:
    nodes: set[int] = set()
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            nodes.add(u)
            nodes.add(v)
    return list(nodes)


def _conflict_delta(
    graph: nx.Graph,
    coloring: Coloring,
    node: int,
    old_color: int,
    new_color: int,
) -> int:
    delta = 0
    for neighbor in graph.neighbors(node):
        neighbor_color = coloring[neighbor]
        if neighbor_color == old_color:
            delta -= 1
        if neighbor_color == new_color:
            delta += 1
    return delta
