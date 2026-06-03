from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable

import networkx as nx

from .algorithms import (
    Coloring,
    color_count,
    count_conflicts,
    dsatur,
    greedy_welsh_powell,
    is_valid_coloring,
    structure_aware_hybrid,
    tabu_search,
)
from .datasets import (
    generate_barabasi_albert_graph,
    generate_erdos_renyi_graph,
    generate_random_regular_graph,
    generate_watts_strogatz_graph,
    iter_dimacs_files,
    load_dimacs_col,
)


Algorithm = Callable[[nx.Graph], Coloring]


def run_algorithm(graph: nx.Graph, name: str, algorithm: Algorithm) -> dict[str, object]:
    start = time.perf_counter()
    coloring = algorithm(graph)
    elapsed = time.perf_counter() - start
    return {
        "algorithm": name,
        "colors": color_count(coloring),
        "time_seconds": elapsed,
        "conflicts": count_conflicts(graph, coloring),
        "valid": is_valid_coloring(graph, coloring),
    }


def algorithms(seed: int, tabu_iterations: int, tabu_tenure: int) -> list[tuple[str, Algorithm]]:
    return [
        ("greedy_welsh_powell", greedy_welsh_powell),
        ("dsatur", dsatur),
        (
            "tabu_search",
            lambda graph: tabu_search(
                graph,
                max_iterations=tabu_iterations,
                tabu_tenure=tabu_tenure,
                seed=seed,
            ),
        ),
        (
            "structure_aware_hybrid",
            lambda graph: structure_aware_hybrid(
                graph,
                max_iterations=tabu_iterations,
                tabu_tenure=tabu_tenure,
                seed=seed,
            ),
        ),
    ]


def graph_features(graph: nx.Graph) -> dict[str, object]:
    degrees = [degree for _, degree in graph.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0
    degeneracy = max(nx.core_number(graph).values()) if graph.number_of_nodes() else 0
    clustering = nx.average_clustering(graph) if graph.number_of_nodes() else 0.0
    return {
        "density": nx.density(graph),
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "degeneracy": degeneracy,
        "clustering": clustering,
    }


def append_algorithm_rows(
    rows: list[dict[str, object]],
    graph: nx.Graph,
    dataset_type: str,
    family: str,
    instance: str,
    repeat: int | str,
    seed: int,
    tabu_iterations: int,
    tabu_tenure: int,
    parameters: str,
) -> None:
    features = graph_features(graph)
    for name, algorithm in algorithms(seed, tabu_iterations, tabu_tenure):
        result = run_algorithm(graph, name, algorithm)
        rows.append(
            {
                "dataset_type": dataset_type,
                "family": family,
                "instance": instance,
                "repeat": repeat,
                "seed": seed,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "parameters": parameters,
                **features,
                **result,
            }
        )


def random_graph_experiments(
    node_counts: list[int],
    probabilities: list[float],
    repeats: int,
    tabu_iterations: int,
    tabu_tenure: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for n in node_counts:
        for p in probabilities:
            for repeat in range(repeats):
                seed = int(n * 10000 + p * 1000 + repeat)
                graph = generate_erdos_renyi_graph(n=n, p=p, seed=seed)
                append_algorithm_rows(
                    rows=rows,
                    graph=graph,
                    dataset_type="synthetic",
                    family="erdos_renyi",
                    instance=f"ER_n{n}_p{p}_r{repeat}",
                    repeat=repeat,
                    seed=seed,
                    tabu_iterations=tabu_iterations,
                    tabu_tenure=tabu_tenure,
                    parameters=f"n={n};p={p}",
                )
    return rows


def structured_graph_experiments(
    node_counts: list[int],
    repeats: int,
    tabu_iterations: int,
    tabu_tenure: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in node_counts:
        specs = [
            ("barabasi_albert", "m=2", lambda seed: generate_barabasi_albert_graph(n, 2, seed)),
            ("barabasi_albert", "m=4", lambda seed: generate_barabasi_albert_graph(n, 4, seed)),
            ("watts_strogatz", "k=6;beta=0.1", lambda seed: generate_watts_strogatz_graph(n, 6, 0.1, seed)),
            ("watts_strogatz", "k=10;beta=0.3", lambda seed: generate_watts_strogatz_graph(n, 10, 0.3, seed)),
            ("random_regular", "d=6", lambda seed: generate_random_regular_graph(n, 6, seed)),
            ("random_regular", "d=10", lambda seed: generate_random_regular_graph(n, 10, seed)),
        ]
        for family, parameters, factory in specs:
            for repeat in range(repeats):
                seed = int(n * 20000 + repeat * 17 + len(family) * 101 + len(parameters))
                graph = factory(seed)
                append_algorithm_rows(
                    rows=rows,
                    graph=graph,
                    dataset_type="synthetic",
                    family=family,
                    instance=f"{family}_n{n}_{parameters.replace(';', '_')}_r{repeat}",
                    repeat=repeat,
                    seed=seed,
                    tabu_iterations=tabu_iterations,
                    tabu_tenure=tabu_tenure,
                    parameters=f"n={n};{parameters}",
                )
    return rows


def dimacs_experiments(
    directory: Path,
    tabu_iterations: int,
    tabu_tenure: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in iter_dimacs_files(directory):
        graph = load_dimacs_col(path)
        seed = abs(hash(path.name)) % 100000
        append_algorithm_rows(
            rows=rows,
            graph=graph,
            dataset_type="benchmark",
            family="dimacs_small",
            instance=path.stem,
            repeat="",
            seed=seed,
            tabu_iterations=tabu_iterations,
            tabu_tenure=tabu_tenure,
            parameters="public DIMACS .col instance",
        )
    return rows


def write_results(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_type",
        "family",
        "instance",
        "repeat",
        "seed",
        "nodes",
        "edges",
        "parameters",
        "density",
        "avg_degree",
        "max_degree",
        "degeneracy",
        "clustering",
        "algorithm",
        "colors",
        "time_seconds",
        "conflicts",
        "valid",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--include-dimacs", action="store_true")
    parser.add_argument("--dimacs-dir", default="data/dimacs")
    parser.add_argument("--output", default="results/experiment_results.csv")
    parser.add_argument("--node-counts", default="80,160,320")
    parser.add_argument("--probabilities", default="0.05,0.1,0.2,0.4,0.6")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tabu-iterations", type=int, default=800)
    parser.add_argument("--tabu-tenure", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_counts = [int(value) for value in args.node_counts.split(",") if value]
    probabilities = [float(value) for value in args.probabilities.split(",") if value]
    rows = random_graph_experiments(
        node_counts=node_counts,
        probabilities=probabilities,
        repeats=args.repeats,
        tabu_iterations=args.tabu_iterations,
        tabu_tenure=args.tabu_tenure,
    )

    if not args.random_only:
        rows.extend(
            structured_graph_experiments(
                node_counts=node_counts,
                repeats=args.repeats,
                tabu_iterations=args.tabu_iterations,
                tabu_tenure=args.tabu_tenure,
            )
        )

    if args.include_dimacs and not args.synthetic_only:
        rows.extend(
            dimacs_experiments(
                Path(args.dimacs_dir),
                tabu_iterations=args.tabu_iterations,
                tabu_tenure=args.tabu_tenure,
            )
        )

    write_results(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
