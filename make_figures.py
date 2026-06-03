from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PALETTE = {
    "greedy_welsh_powell": "#0072B2",
    "dsatur": "#009E73",
    "tabu_search": "#D55E00",
    "structure_aware_hybrid": "#CC79A7",
}

LABELS = {
    "greedy_welsh_powell": "Greedy/WP",
    "dsatur": "DSATUR",
    "tabu_search": "Tabu",
    "structure_aware_hybrid": "Hybrid",
}


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_workflow(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    ax.axis("off")
    labels = [
        ("Graph families", "ER, BA, WS,\nregular, DIMACS"),
        ("Features", "density, degree,\ndegeneracy, clustering"),
        ("Algorithms", "Greedy/WP,\nDSATUR, Tabu, Hybrid"),
        ("Evaluation", "colors, validity,\ntime, gaps, wins"),
    ]
    x_positions = [0.08, 0.34, 0.60, 0.86]
    for idx, ((title, body), x) in enumerate(zip(labels, x_positions)):
        ax.text(
            x,
            0.55,
            f"{title}\n{body}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9FA", ec="#9AA7B1", lw=0.8),
        )
        if idx < len(x_positions) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.10, 0.55),
                xytext=(x + 0.10, 0.55),
                arrowprops=dict(arrowstyle="->", color="#5C6770", lw=1.0),
            )
    fig.savefig(output_dir / "fig_workflow.png")
    plt.close(fig)


def plot_family_gap(family_summary: pd.DataFrame, output_dir: Path) -> None:
    data = family_summary[family_summary["dataset_type"] == "synthetic"].copy()
    families = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "random_regular"]
    algorithms = ["greedy_welsh_powell", "dsatur", "structure_aware_hybrid", "tabu_search"]
    fig, ax = plt.subplots(figsize=(6.6, 2.55))
    width = 0.18
    x = range(len(families))
    for i, algorithm in enumerate(algorithms):
        values = []
        for family in families:
            row = data[(data["family"] == family) & (data["algorithm"] == algorithm)]
            values.append(float(row["mean_color_gap"].iloc[0]))
        offsets = [value + (i - 1.5) * width for value in x]
        ax.bar(offsets, values, width=width, color=PALETTE[algorithm], label=LABELS[algorithm])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["ER", "BA", "WS", "Regular"])
    ax.set_ylabel("Mean color gap to best")
    ax.set_xlabel("Synthetic graph family")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(output_dir / "fig_family_color_gap.png")
    plt.close(fig)


def plot_tradeoff(algorithm_ranking: pd.DataFrame, output_dir: Path) -> None:
    data = algorithm_ranking.copy()
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for _, row in data.iterrows():
        algorithm = row["algorithm"]
        ax.scatter(
            row["geomean_time_vs_dsatur"],
            row["mean_color_gap"],
            s=85,
            color=PALETTE[algorithm],
            edgecolor="white",
            linewidth=0.8,
            label=LABELS[algorithm],
        )
        ax.text(
            row["geomean_time_vs_dsatur"] * 1.06,
            row["mean_color_gap"] + 0.015,
            LABELS[algorithm],
            va="center",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Geometric mean time ratio vs DSATUR")
    ax.set_ylabel("Mean color gap to best")
    ax.axvline(1.0, color="#666666", lw=0.8, ls="--")
    ax.axhline(0.0, color="#666666", lw=0.8, ls="--")
    fig.savefig(output_dir / "fig_quality_time_tradeoff.png")
    plt.close(fig)


def plot_best_by_density(best_by_density: pd.DataFrame, output_dir: Path) -> None:
    pivot = best_by_density.pivot(index="density_bin", columns="best_algorithm", values="count").fillna(0)
    order = ["very sparse", "sparse", "medium", "dense"]
    pivot = pivot.reindex(order)
    algorithms = [algorithm for algorithm in LABELS if algorithm in pivot.columns]
    fig, ax = plt.subplots(figsize=(5.2, 2.55))
    bottom = [0] * len(pivot)
    x = range(len(pivot))
    for algorithm in algorithms:
        values = pivot[algorithm].tolist()
        ax.bar(x, values, bottom=bottom, color=PALETTE[algorithm], label=LABELS[algorithm])
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(value) for value in pivot.index], rotation=15, ha="right")
    ax.set_ylabel("Best/tied-best count")
    ax.set_xlabel("Density bin")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.23))
    fig.savefig(output_dir / "fig_best_by_density.png")
    plt.close(fig)


def plot_dimacs(dimacs_summary: pd.DataFrame, output_dir: Path) -> None:
    algorithms = ["greedy_welsh_powell", "dsatur", "structure_aware_hybrid", "tabu_search"]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = range(len(dimacs_summary))
    width = 0.18
    for i, algorithm in enumerate(algorithms):
        offsets = [value + (i - 1.5) * width for value in x]
        ax.bar(offsets, dimacs_summary[algorithm], width=width, color=PALETTE[algorithm], label=LABELS[algorithm])
    ax.set_xticks(list(x))
    ax.set_xticklabels(dimacs_summary["instance"], rotation=25, ha="right")
    ax.set_ylabel("Number of colors")
    ax.set_xlabel("DIMACS instance")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    fig.savefig(output_dir / "fig_dimacs_sanity.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default="results/expanded_summary")
    parser.add_argument("--output-dir", default="MDPI_template_ACS/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_workflow(output_dir)
    plot_family_gap(pd.read_csv(summary_dir / "family_summary.csv"), output_dir)
    plot_tradeoff(pd.read_csv(summary_dir / "algorithm_ranking.csv"), output_dir)
    plot_best_by_density(pd.read_csv(summary_dir / "best_by_density.csv"), output_dir)
    plot_dimacs(pd.read_csv(summary_dir / "dimacs_summary.csv"), output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
