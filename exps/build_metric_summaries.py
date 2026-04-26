"""Build cross-scenario per-episode metric summary tables and heatmaps."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("exps/results")
SCENARIOS = ["normal", "bias", "extreme"]
METHOD_ORDER = ["o2o-iql", "ppo", "eoi", "all-no-split", "greedy-split", "station-assignment"]
BIAS_EPISODE_LIMIT = 30
METRICS = {
    "avg_wait": "average waiting time",
    "p95_wait": "p95 waiting time",
    "cv_li": "load imbalance",
}
METRIC_OUTPUT_STEMS = {
    "avg_wait": ["avg_wait_summary"],
    "p95_wait": ["p95_wait_summary", "p956_wait_summary"],
    "cv_li": ["cv_li_summary", "cv_cli_summary"],
}


def _episode_id(episode_name: str) -> str:
    match = re.search(r"(\d+)", str(episode_name))
    return match.group(1) if match else str(episode_name)


def _best_metrics_path(method: str, scenario: str) -> Path:
    if method in {"all-no-split", "greedy-split", "station-assignment"}:
        return RESULTS_DIR / "baselines_test" / "summary" / f"{method}_{scenario}_best_episode_metrics.csv"
    return RESULTS_DIR / method / "summary" / f"{scenario}_best_episode_metrics.csv"


def _scenario_episode_columns(scenario: str) -> list[str]:
    sample_path = _best_metrics_path(METHOD_ORDER[0], scenario)
    df = pd.read_csv(sample_path)
    return [column for column in df.columns if column != "metric"]


def _build_metric_table(metric_key: str, scenario_episode_ids: dict[str, set[str]] | None = None) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {method: {} for method in METHOD_ORDER}
    metric_label = METRICS[metric_key]

    for scenario in SCENARIOS:
        allowed_episode_ids = scenario_episode_ids.get(scenario) if scenario_episode_ids else None
        for method in METHOD_ORDER:
            csv_path = _best_metrics_path(method, scenario)
            df = pd.read_csv(csv_path)
            metric_rows = df[df["metric"] == metric_label]
            if metric_rows.empty:
                raise ValueError(f"Metric '{metric_label}' not found in {csv_path}")
            row = metric_rows.iloc[0]
            for episode_column in [column for column in df.columns if column != "metric"]:
                episode_id = _episode_id(episode_column)
                if allowed_episode_ids is not None and episode_id not in allowed_episode_ids:
                    continue
                rows[method][f"{scenario}_{episode_id}"] = float(row[episode_column])

    return pd.DataFrame.from_dict(rows, orient="index")


def _rank_high_avg_wait(avg_wait_table: pd.DataFrame, episode_columns: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    methods = [method for method in METHOD_ORDER if method in avg_wait_table.index]

    for episode_column in episode_columns:
        values = avg_wait_table.loc[methods, episode_column].astype(float)
        scores[episode_column] = float(values.mean())

    return sorted(
        episode_columns,
        key=lambda column: (scores[column], -int(_episode_id(column))),
        reverse=True,
    )


def _select_bias_episode_ids(limit: int = BIAS_EPISODE_LIMIT) -> set[str]:
    avg_wait_table = _build_metric_table("avg_wait")
    bias_columns = [column for column in avg_wait_table.columns if column.startswith("bias_")]
    ranked_columns = _rank_high_avg_wait(avg_wait_table, bias_columns)
    selected_columns = ranked_columns[:limit]
    return {_episode_id(column) for column in selected_columns}


def _summary_episode_filter() -> dict[str, set[str]]:
    return {"bias": _select_bias_episode_ids()}


def _scenario_columns(columns: list[str], scenario: str) -> list[str]:
    return [column for column in columns if column.startswith(f"{scenario}_")]


def _ordered_summary_columns(metric_tables: dict[str, pd.DataFrame]) -> list[str]:
    reference_columns = metric_tables["avg_wait"].columns.tolist()
    normal_columns = sorted(
        _scenario_columns(reference_columns, "normal"),
        key=lambda column: (
            float(metric_tables["avg_wait"].loc["o2o-iql", column]),
            int(_episode_id(column)),
        ),
    )
    bias_columns = sorted(
        _scenario_columns(reference_columns, "bias"),
        key=lambda column: (
            float(metric_tables["p95_wait"].loc["o2o-iql", column]),
            int(_episode_id(column)),
        ),
    )
    extreme_columns = _scenario_columns(reference_columns, "extreme")
    return normal_columns + bias_columns + extreme_columns


def _plot_heatmap(table: pd.DataFrame, metric_key: str, output_stem: str) -> None:
    fig_width = max(14.0, table.shape[1] * 0.28)
    fig_height = max(4.5, table.shape[0] * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="viridis")

    ax.set_title(f"{metric_key} summary")
    ax.set_xlabel("Episode id")
    ax.set_ylabel("Model")
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index.tolist())
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns.tolist(), rotation=90, fontsize=8)

    boundaries = []
    offset = 0
    for scenario in SCENARIOS[:-1]:
        count = sum(1 for column in table.columns if column.startswith(f"{scenario}_"))
        offset += count
        boundaries.append(offset - 0.5)
    for boundary in boundaries:
        ax.axvline(boundary, color="white", linewidth=1.5)

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel(metric_key)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{output_stem}.png", dpi=200)
    plt.close(fig)


def _write_metric_outputs(table: pd.DataFrame, metric_key: str) -> None:
    for output_stem in METRIC_OUTPUT_STEMS[metric_key]:
        csv_path = RESULTS_DIR / f"{output_stem}.csv"
        png_path = RESULTS_DIR / f"{output_stem}.png"
        table.to_csv(csv_path, encoding="utf-8")
        _plot_heatmap(table, metric_key, output_stem)
        print(csv_path)
        print(png_path)


def main() -> None:
    scenario_episode_ids = _summary_episode_filter()
    metric_tables = {
        metric_key: _build_metric_table(metric_key, scenario_episode_ids)
        for metric_key in METRICS
    }
    ordered_columns = _ordered_summary_columns(metric_tables)
    for metric_key, table in metric_tables.items():
        _write_metric_outputs(table.reindex(columns=ordered_columns), metric_key)


if __name__ == "__main__":
    main()
