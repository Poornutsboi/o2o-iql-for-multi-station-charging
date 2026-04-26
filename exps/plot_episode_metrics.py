"""Plot per-episode metric comparisons from experiment result CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("exps/results/episode_comparison")
OUTPUT_DIR = RESULT_DIR / "plots"

METRICS = [
    ("mean_waiting_time", "Average waiting time"),
    ("p95_waiting_time", "P95 waiting time"),
    ("cv_load_imbalance", "Load imbalance"),
]

METHOD_STYLES = {
    "eoi": {"linestyle": "-", "linewidth": 2.4, "markersize": 4.6, "zorder": 10},
}


def episode_index(name: str) -> int:
    match = re.search(r"(\d+)", str(name))
    return int(match.group(1)) if match else 0


def discover_methods(columns: list[str]) -> list[str]:
    methods: set[str] = set()
    metric_suffixes = [f"_{metric}_mean" for metric, _ in METRICS]
    for column in columns:
        for suffix in metric_suffixes:
            if column.endswith(suffix):
                methods.add(column[: -len(suffix)])
    preferred_order = [
        "o2o-iql",
        "ppo",
        "all-no-split",
        "greedy-split",
        "station-assignment",
        "eoi",
    ]
    return [m for m in preferred_order if m in methods] + sorted(methods - set(preferred_order))


def plot_file(csv_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    df = df.assign(episode=df["episode_name"].map(episode_index)).sort_values("episode")
    methods = discover_methods(list(df.columns))
    scenario = csv_path.name.replace("_episode_comparison.csv", "")

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(13, 10), sharex=True)
    colors = plt.get_cmap("tab10").colors

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        for idx, method in enumerate(methods):
            mean_col = f"{method}_{metric_key}_mean"
            std_col = f"{method}_{metric_key}_std"
            if mean_col not in df:
                continue

            x = df["episode"].to_numpy()
            y = df[mean_col].astype(float).to_numpy()
            color = colors[idx % len(colors)]
            style = METHOD_STYLES.get(method, {})
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=style.get("linewidth", 1.8),
                markersize=style.get("markersize", 3.5),
                linestyle=style.get("linestyle", "-"),
                label=method,
                color=color,
                markerfacecolor="white" if method == "eoi" else color,
                markeredgewidth=1.4 if method == "eoi" else 1.0,
                zorder=style.get("zorder", 3 + idx),
            )

            if std_col in df:
                std = df[std_col].fillna(0).astype(float).to_numpy()
                ax.fill_between(
                    x,
                    y - std,
                    y + std,
                    color=color,
                    alpha=0.08 if method == "eoi" else 0.12,
                    linewidth=0,
                    zorder=1,
                )

        ax.set_ylabel(metric_label)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title(f"{scenario.capitalize()} per-episode method comparison")
    axes[-1].set_xlabel("Episode")
    axes[-1].set_xticks(df["episode"].to_numpy())
    axes[-1].set_xticklabels([str(v) for v in df["episode"].to_numpy()], rotation=45, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{scenario}_episode_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    csv_files = sorted(RESULT_DIR.glob("*_episode_comparison.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No episode comparison CSV files found in {RESULT_DIR}")

    for csv_path in csv_files:
        print(plot_file(csv_path))


if __name__ == "__main__":
    main()
