from pathlib import Path

import pandas as pd

from exps.plot_metric_summary_lines import (
    METRIC_FILES,
    load_metric_table,
    ordered_methods,
    plot_metric_comparison,
)


def test_load_metric_table_reads_methods_and_episode_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "metric.csv"
    pd.DataFrame(
        {
            "method": ["o2o-iql", "ppo"],
            "normal_0000": [1.0, 2.0],
            "bias_0000": [3.0, 4.0],
        }
    ).to_csv(csv_path, index=False)

    table = load_metric_table(csv_path)

    assert table.index.tolist() == ["o2o-iql", "ppo"]
    assert table.columns.tolist() == ["normal_0000", "bias_0000"]


def test_ordered_methods_excludes_greedy_split(tmp_path: Path) -> None:
    csv_path = tmp_path / "metric.csv"
    pd.DataFrame(
        {
            "method": ["o2o-iql", "greedy-split", "ppo"],
            "normal_0000": [1.0, 9.0, 2.0],
        }
    ).to_csv(csv_path, index=False)

    table = load_metric_table(csv_path)

    assert ordered_methods(table) == ["o2o-iql", "ppo"]


def test_plot_metric_comparison_creates_three_vertical_subplots(tmp_path: Path) -> None:
    for filename, _ in METRIC_FILES:
        pd.DataFrame(
            {
                "method": ["o2o-iql", "ppo"],
                "normal_0000": [1.0, 2.0],
                "normal_0001": [1.5, 2.5],
                "bias_0000": [3.0, 4.0],
            }
        ).to_csv(tmp_path / filename, index=False)

    output_path = plot_metric_comparison(tmp_path, tmp_path / "comparison.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
