from __future__ import annotations

from pathlib import Path

import networkx as nx


def generate_erdos_renyi_graph(n: int, p: float, seed: int) -> nx.Graph:
    graph = nx.gnp_random_graph(n=n, p=p, seed=seed)
    return nx.convert_node_labels_to_integers(graph)


def generate_barabasi_albert_graph(n: int, m: int, seed: int) -> nx.Graph:
    graph = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return nx.convert_node_labels_to_integers(graph)


def generate_watts_strogatz_graph(n: int, k: int, beta: float, seed: int) -> nx.Graph:
    graph = nx.watts_strogatz_graph(n=n, k=k, p=beta, seed=seed)
    return nx.convert_node_labels_to_integers(graph)


def generate_random_regular_graph(n: int, degree: int, seed: int) -> nx.Graph:
    if degree >= n:
        raise ValueError("degree must be smaller than n")
    if (n * degree) % 2:
        degree -= 1
    graph = nx.random_regular_graph(d=degree, n=n, seed=seed)
    return nx.convert_node_labels_to_integers(graph)


def load_dimacs_col(path: Path) -> nx.Graph:
    graph = nx.Graph()
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p" and len(parts) >= 4:
                node_count = int(parts[2])
                graph.add_nodes_from(range(1, node_count + 1))
            elif parts[0] == "e" and len(parts) >= 3:
                graph.add_edge(int(parts[1]), int(parts[2]))
    return graph


def iter_dimacs_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.col"))
