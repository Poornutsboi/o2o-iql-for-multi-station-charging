"""Plot metric summary comparisons as vertically stacked line charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("exps/results")
OUTPUT_PATH = RESULTS_DIR / "metric_comparison_lines.png"

METRIC_FILES = [
    ("avg_wait_summary.csv", "Average waiting time"),
    ("p95_wait_summary.csv", "P95 waiting time"),
    ("cv_li_summary.csv", "Load imbalance"),
]

EXCLUDED_METHODS = {"greedy-split"}
METHOD_ORDER = ["o2o-iql", "ppo", "eoi", "all-no-split", "station-assignment"]
METHOD_LABELS = {
    "o2o-iql": "O2O-IQL",
    "ppo": "PPO",
    "eoi": "EOI",
    "all-no-split": "NS",
    "station-assignment": "SA",
}
METHOD_COLORS = {
    "o2o-iql": "#0072B2",
    "ppo": "#009E73",
    "eoi": "#D55E00",
    "all-no-split": "#E69F00",
    "station-assignment": "#CC79A7",
}
LEGEND_LABEL_ORDER = ["NS", "SA", "PPO", "EOI", "O2O-IQL"]
SCENARIO_LABELS = {
    "normal": "Normal",
    "bias": "Concentrated",
    "extreme": "Extreme",
}


def load_metric_table(csv_path: Path) -> pd.DataFrame:
    """Load a summary CSV using the first column as the method index."""
    table = pd.read_csv(csv_path, index_col=0)
    table.index.name = "method"
    return table


def ordered_methods(table: pd.DataFrame) -> list[str]:
    known = [method for method in METHOD_ORDER if method in table.index]
    extra = sorted(
        method for method in table.index if method not in METHOD_ORDER and method not in EXCLUDED_METHODS
    )
    return known + extra


def _scenario_boundaries(columns: list[str]) -> tuple[list[int], list[tuple[str, float]]]:
    boundaries: list[int] = []
    centers: list[tuple[str, float]] = []
    start = 0
    current = columns[0].split("_", 1)[0]

    for idx, column in enumerate(columns + ["_end"]):
        scenario = column.split("_", 1)[0]
        if scenario != current:
            end = idx - 1
            centers.append((current, (start + end) / 2))
            if idx < len(columns):
                boundaries.append(idx - 1)
            start = idx
            current = scenario

    return boundaries, centers


def plot_metric_comparison(results_dir: Path = RESULTS_DIR, output_path: Path = OUTPUT_PATH) -> Path:
    fig, axes = plt.subplots(len(METRIC_FILES), 1, figsize=(15, 10.5), sharex=True)

    for axis_idx, (ax, (filename, ylabel)) in enumerate(zip(axes, METRIC_FILES)):
        table = load_metric_table(results_dir / filename)
        columns = table.columns.tolist()
        x_values = list(range(1, len(columns) + 1))

        for method in ordered_methods(table):
            values = table.loc[method].astype(float).to_numpy()
            color = METHOD_COLORS.get(method)
            ax.plot(
                x_values,
                values,
                label=METHOD_LABELS.get(method, method),
                color=color,
                linewidth=1.5 if method == "o2o-iql" else 1.2,
            )

        boundaries, centers = _scenario_boundaries(columns)
        for boundary in boundaries:
            ax.axvline(boundary + 1.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.65)
        if axis_idx == 0:
            for scenario, center in centers:
                ax.text(
                    center + 1,
                    1.04,
                    SCENARIO_LABELS.get(scenario, scenario.capitalize()),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )

        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, axis="y", linestyle="--", color="#b8b8b8", alpha=0.55)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=12)

    axes[0].set_title("Comparison of Algorithms on Test Episodes", fontsize=18, fontweight="bold", pad=34)
    axes[-1].set_xlabel("Test Episode", fontsize=14)

    total_points = len(load_metric_table(results_dir / METRIC_FILES[0][0]).columns)
    tick_step = max(1, total_points // 12)
    ticks = list(range(1, total_points + 1, tick_step))
    if ticks[-1] != total_points:
        if total_points - ticks[-1] <= max(1, tick_step // 2):
            ticks.pop()
        ticks.append(total_points)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([str(tick) for tick in ticks])

    handles, labels = axes[0].get_legend_handles_labels()
    legend_items = sorted(
        zip(handles, labels),
        key=lambda item: LEGEND_LABEL_ORDER.index(item[1])
        if item[1] in LEGEND_LABEL_ORDER
        else len(LEGEND_LABEL_ORDER),
    )
    handles, labels = zip(*legend_items)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, bbox_to_anchor=(0.5, 0.035))
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), h_pad=1.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    print(plot_metric_comparison())


if __name__ == "__main__":
    main()
