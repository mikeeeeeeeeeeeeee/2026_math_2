from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd


ALGORITHM_ORDER = [
    "greedy_welsh_powell",
    "dsatur",
    "tabu_search",
    "structure_aware_hybrid",
]


def _paired_best(data: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset_type", "family", "instance"]
    best_colors = data.groupby(key_cols)["colors"].min().rename("best_colors")
    joined = data.join(best_colors, on=key_cols)
    joined["color_gap"] = joined["colors"] - joined["best_colors"]
    return joined


def _geomean(values: pd.Series) -> float:
    clean = values[(values > 0) & values.notna()]
    if clean.empty:
        return float("nan")
    return math.exp(clean.map(math.log).mean())


def _bootstrap_mean_ci(values: pd.Series, seed: int = 20260603, samples: int = 5000) -> tuple[float, float]:
    clean = [float(value) for value in values.dropna()]
    if not clean:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return low, high


def _two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _selector_from_threshold(
    color_pivot: pd.DataFrame,
    time_pivot: pd.DataFrame,
    features: pd.DataFrame,
    density_threshold: float,
    size_threshold: int = 220,
) -> tuple[pd.Series, pd.Series]:
    use_tabu = (features["nodes"] <= size_threshold) & (features["density"] >= density_threshold)
    selected_colors = color_pivot["dsatur"].where(~use_tabu, color_pivot["tabu_search"])
    selected_time = time_pivot["dsatur"].where(~use_tabu, time_pivot["tabu_search"])
    return selected_colors, selected_time


def summarize(input_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(input_path)
    data = _paired_best(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    family_summary = (
        data.groupby(["dataset_type", "family", "algorithm"], as_index=False)
        .agg(
            instances=("instance", "nunique"),
            mean_nodes=("nodes", "mean"),
            mean_density=("density", "mean"),
            mean_degeneracy=("degeneracy", "mean"),
            mean_colors=("colors", "mean"),
            std_colors=("colors", "std"),
            mean_color_gap=("color_gap", "mean"),
            mean_time_seconds=("time_seconds", "mean"),
            median_time_seconds=("time_seconds", "median"),
            valid_rate=("valid", "mean"),
        )
        .sort_values(["dataset_type", "family", "algorithm"])
    )
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)

    key_cols = ["dataset_type", "family", "instance"]
    color_pivot = data.pivot_table(index=key_cols, columns="algorithm", values="colors")
    time_pivot = data.pivot_table(index=key_cols, columns="algorithm", values="time_seconds")
    rows = []
    for algorithm in ALGORITHM_ORDER:
        if algorithm not in color_pivot:
            continue
        wins = (color_pivot[algorithm] == color_pivot.min(axis=1)).sum()
        strict_wins = (color_pivot[algorithm] < color_pivot.drop(columns=[algorithm]).min(axis=1)).sum()
        losses = (color_pivot[algorithm] > color_pivot.min(axis=1)).sum()
        rows.append(
            {
                "algorithm": algorithm,
                "instances": len(color_pivot),
                "best_or_tied": int(wins),
                "strict_best": int(strict_wins),
                "not_best": int(losses),
                "mean_color_gap": float((color_pivot[algorithm] - color_pivot.min(axis=1)).mean()),
                "median_time_seconds": float(time_pivot[algorithm].median()),
                "geomean_time_vs_dsatur": _geomean(time_pivot[algorithm] / time_pivot["dsatur"]),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "algorithm_ranking.csv", index=False)

    hybrid = color_pivot["structure_aware_hybrid"]
    dsatur = color_pivot["dsatur"]
    tabu = color_pivot["tabu_search"]
    pd.DataFrame(
        [
            {
                "comparison": "hybrid_vs_dsatur",
                "wins": int((hybrid < dsatur).sum()),
                "ties": int((hybrid == dsatur).sum()),
                "losses": int((hybrid > dsatur).sum()),
                "mean_color_delta": float((hybrid - dsatur).mean()),
                "geomean_time_ratio": _geomean(time_pivot["structure_aware_hybrid"] / time_pivot["dsatur"]),
            },
            {
                "comparison": "hybrid_vs_tabu",
                "wins": int((hybrid < tabu).sum()),
                "ties": int((hybrid == tabu).sum()),
                "losses": int((hybrid > tabu).sum()),
                "mean_color_delta": float((hybrid - tabu).mean()),
                "geomean_time_ratio": _geomean(time_pivot["structure_aware_hybrid"] / time_pivot["tabu_search"]),
            },
        ]
    ).to_csv(output_dir / "hybrid_comparison.csv", index=False)

    stat_rows = []
    for comparison, left, right in [
        ("hybrid_minus_dsatur", hybrid, dsatur),
        ("hybrid_minus_tabu", hybrid, tabu),
        ("tabu_minus_dsatur", tabu, dsatur),
    ]:
        delta = left - right
        wins = int((delta < 0).sum())
        ties = int((delta == 0).sum())
        losses = int((delta > 0).sum())
        ci_low, ci_high = _bootstrap_mean_ci(delta)
        stat_rows.append(
            {
                "comparison": comparison,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_color_delta": float(delta.mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "two_sided_sign_p_excluding_ties": _two_sided_sign_p(wins, losses),
            }
        )
    pd.DataFrame(stat_rows).to_csv(output_dir / "statistical_checks.csv", index=False)

    feature_index = (
        data.groupby(key_cols, as_index=True)
        .first()[["nodes", "density"]]
        .reindex(color_pivot.index)
    )
    threshold_rows = []
    for threshold in [0.10, 0.15, 0.18, 0.20, 0.25, 0.30]:
        selected_colors, selected_time = _selector_from_threshold(
            color_pivot=color_pivot,
            time_pivot=time_pivot,
            features=feature_index,
            density_threshold=threshold,
        )
        best_colors = color_pivot.min(axis=1)
        delta_dsatur = selected_colors - color_pivot["dsatur"]
        delta_tabu = selected_colors - color_pivot["tabu_search"]
        threshold_rows.append(
            {
                "density_threshold": threshold,
                "tabu_calls": int(((feature_index["nodes"] <= 220) & (feature_index["density"] >= threshold)).sum()),
                "best_or_tied": int((selected_colors == best_colors).sum()),
                "mean_color_gap": float((selected_colors - best_colors).mean()),
                "wins_vs_dsatur": int((delta_dsatur < 0).sum()),
                "ties_vs_dsatur": int((delta_dsatur == 0).sum()),
                "losses_vs_dsatur": int((delta_dsatur > 0).sum()),
                "mean_delta_vs_dsatur": float(delta_dsatur.mean()),
                "geomean_time_vs_dsatur": _geomean(selected_time / time_pivot["dsatur"]),
                "geomean_time_vs_tabu": _geomean(selected_time / time_pivot["tabu_search"]),
                "mean_delta_vs_tabu": float(delta_tabu.mean()),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(output_dir / "threshold_sensitivity.csv", index=False)

    feature_rows = (
        data[data["dataset_type"] == "synthetic"]
        .groupby(["family", "instance"], as_index=False)
        .first()[["family", "instance", "density", "degeneracy", "clustering"]]
    )
    best_algorithms = (
        data[data["dataset_type"] == "synthetic"]
        .sort_values(["colors", "time_seconds"])
        .groupby(["family", "instance"], as_index=False)
        .first()[["family", "instance", "algorithm", "colors"]]
        .rename(columns={"algorithm": "best_algorithm", "colors": "best_colors"})
    )
    feature_best = feature_rows.merge(best_algorithms, on=["family", "instance"])
    feature_best["density_bin"] = pd.cut(
        feature_best["density"],
        bins=[0, 0.05, 0.15, 0.30, 1.0],
        labels=["very sparse", "sparse", "medium", "dense"],
        include_lowest=True,
    )
    (
        feature_best.groupby(["density_bin", "best_algorithm"], observed=True)
        .size()
        .reset_index(name="count")
        .to_csv(output_dir / "best_by_density.csv", index=False)
    )

    dimacs = data[data["dataset_type"] == "benchmark"].copy()
    if not dimacs.empty:
        dimacs.pivot_table(
            index=["instance", "nodes", "edges", "density", "degeneracy"],
            columns="algorithm",
            values="colors",
            aggfunc="first",
        ).reset_index().to_csv(output_dir / "dimacs_summary.csv", index=False)

    print(f"Wrote summaries to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/expanded_results.csv")
    parser.add_argument("--output-dir", default="results/expanded_summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
