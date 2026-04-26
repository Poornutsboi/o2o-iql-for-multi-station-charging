"""Rebuild aggregate and per-episode result summaries."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.env_data import CAPACITY


RESULTS_ROOT = Path("exps/results")
EPISODE_COMPARISON_DIR = RESULTS_ROOT / "episode_comparison"

SCENARIOS = ["bias", "normal", "extreme"]
METHODS = ["o2o-iql", "ppo", "eoi", "all-no-split", "greedy-split", "station-assignment"]
BASELINE_SEEDS = [42, 123, 2024, 3407, 3408]
O2O_IQL_BEST_SEED_COUNT = 5

SUMMARY_METRICS = [
    "mean_reward",
    "mean_ep_length",
    "mean_waiting_time",
    "dataset_average_waiting_time",
    "mean_p95_waiting_time",
    "mean_max_waiting_time",
    "mean_cv_load_imbalance",
]
EPISODE_METRICS = [
    "episode_reward",
    "episode_length",
    "vehicle_count",
    "total_waiting_time",
    "mean_waiting_time",
    "p95_waiting_time",
    "max_waiting_time",
    "cv_load_imbalance",
]
HIGHER_IS_BETTER_EPISODE_METRICS = {"episode_reward"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _stat(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(mean(values)) if values else 0.0,
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "values": values,
    }


def _cv(values: list[float]) -> float:
    return float(pstdev(values) / (mean(values) + 1e-8)) if values else 0.0


def _normalize_episode_metric(metric: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metric)
    normalized_demand = normalized.get("normalized_assigned_demand_by_station")
    if not normalized_demand and normalized.get("assigned_demand_by_station"):
        assigned = [float(value) for value in normalized["assigned_demand_by_station"]]
        capacities = [float(value) for value in CAPACITY.tolist()[: len(assigned)]]
        normalized_demand = [
            demand / capacity if capacity > 0.0 else 0.0
            for demand, capacity in zip(assigned, capacities)
        ]
        normalized["normalized_assigned_demand_by_station"] = normalized_demand
    if normalized_demand:
        normalized["cv_load_imbalance"] = _cv(
            [float(value) for value in normalized_demand]
        )
    return normalized


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    if "episode_metrics" not in normalized:
        return normalized
    episode_metrics = [
        _normalize_episode_metric(metric) for metric in normalized["episode_metrics"]
    ]
    normalized["episode_metrics"] = episode_metrics
    values = [float(metric["cv_load_imbalance"]) for metric in episode_metrics]
    if values:
        normalized["mean_cv_load_imbalance"] = float(mean(values))
        normalized["mean_load_imbalance"] = float(mean(values))
    return normalized


def _best_metric_values(method: str, metric: str, values: list[float]) -> list[float]:
    """Select values used for per-episode statistics.

    O2O-IQL can have more seed runs than we want to report. For each episode
    metric, keep only the best five seed values: higher reward is better, while
    waiting/load metrics are costs and therefore lower is better.
    """
    if method != "o2o-iql" or len(values) <= O2O_IQL_BEST_SEED_COUNT:
        return values
    reverse = metric in HIGHER_IS_BETTER_EPISODE_METRICS
    return sorted(values, reverse=reverse)[:O2O_IQL_BEST_SEED_COUNT]


def _normalize_summary_metric(result: dict[str, Any], metric: str) -> float:
    if metric == "mean_cv_load_imbalance":
        return float(result.get("mean_cv_load_imbalance", result.get("mean_load_imbalance", 0.0)))
    return float(result.get(metric, 0.0))


def _baseline_result(scenario: str, baseline: str) -> dict[str, Any]:
    path = RESULTS_ROOT / "baselines_test" / scenario / f"{baseline}.json"
    return _normalize_result(_load_json(path)["results"][0])


def _method_results(scenario: str, method: str) -> list[dict[str, Any]]:
    if method in {"all-no-split", "greedy-split", "station-assignment"}:
        result = _baseline_result(scenario, method)
        return [result for _ in BASELINE_SEEDS]
    if method == "eoi":
        path = RESULTS_ROOT / "eoi" / scenario / "summary.json"
        return [_normalize_result(result) for result in _load_json(path)["results"]]

    path = RESULTS_ROOT / method / scenario / "summary.json"
    return [_normalize_result(result) for result in _load_json(path)["results"]]


def _method_seeds(results: list[dict[str, Any]], method: str) -> list[int]:
    if method in {"all-no-split", "greedy-split", "station-assignment"}:
        return BASELINE_SEEDS
    return [int(result.get("seed", 0)) for result in results]


def rebuild_aggregated_summary() -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenarios[scenario] = {}
        for method in METHODS:
            results = _method_results(scenario, method)
            first = results[0]
            method_summary: dict[str, Any] = {
                "method": method,
                "n_runs": len(results),
                "data_dir": first.get("data_dir", str(Path("data/test_dataset") / scenario)),
                "seeds": _method_seeds(results, method),
            }
            for metric in SUMMARY_METRICS:
                values = [_normalize_summary_metric(result, metric) for result in results]
                method_summary[metric] = _stat(values)
            scenarios[scenario][method] = method_summary

    payload = {
        "results_root": str(RESULTS_ROOT),
        "note": (
            "This file summarizes existing result files only. When available, "
            "it prefers multi-seed test-set summaries for ROI and baselines."
        ),
        "scenarios": scenarios,
    }
    (RESULTS_ROOT / "aggregated_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def rebuild_aggregated_markdown(payload: dict[str, Any]) -> None:
    headers = [
        "Method",
        "Runs",
        "Mean Reward",
        "Mean Waiting Time",
        "Dataset Avg Waiting",
        "P95 Waiting",
        "Max Waiting",
        "CV Load Imbalance",
    ]
    metric_by_header = [
        "mean_reward",
        "mean_waiting_time",
        "dataset_average_waiting_time",
        "mean_p95_waiting_time",
        "mean_max_waiting_time",
        "mean_cv_load_imbalance",
    ]

    lines = [
        "# Results Summary",
        "",
        payload["note"],
        "",
    ]
    for scenario in SCENARIOS:
        lines.extend([f"## {scenario}", ""])
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for method in METHODS:
            summary = payload["scenarios"][scenario][method]
            row = [method, str(summary["n_runs"])]
            for metric in metric_by_header:
                stat = summary[metric]
                row.append(f"{stat['mean']:.2f} ± {stat['std']:.2f}" if metric != "mean_cv_load_imbalance" else f"{stat['mean']:.3f} ± {stat['std']:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    (RESULTS_ROOT / "aggregated_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _episode_metrics_by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["episode_name"]): item for item in result["episode_metrics"]}


def rebuild_episode_comparison() -> None:
    EPISODE_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        method_runs = {method: _method_results(scenario, method) for method in METHODS}
        episode_names = sorted(_episode_metrics_by_name(method_runs["o2o-iql"][0]).keys())

        episodes: list[dict[str, Any]] = []
        csv_rows: list[dict[str, Any]] = []
        for episode_name in episode_names:
            episode_payload: dict[str, Any] = {"episode_name": episode_name, "methods": {}}
            csv_row: dict[str, Any] = {"episode_name": episode_name}
            for method in METHODS:
                runs = method_runs[method]
                per_run = [_episode_metrics_by_name(result)[episode_name] for result in runs]
                method_payload: dict[str, Any] = {
                    "episode_name": episode_name,
                    "n_runs": len(per_run),
                }
                for metric in EPISODE_METRICS:
                    values = [float(item.get(metric, 0.0)) for item in per_run]
                    values = _best_metric_values(method, metric, values)
                    stat = _stat(values)
                    method_payload[metric] = stat
                    csv_row[f"{method}_{metric}_mean"] = stat["mean"]
                    csv_row[f"{method}_{metric}_std"] = stat["std"]
                episode_payload["methods"][method] = method_payload
            episodes.append(episode_payload)
            csv_rows.append(csv_row)

        json_payload = {
            "scenario": scenario,
            "methods": METHODS,
            "episodes": episodes,
        }
        (EPISODE_COMPARISON_DIR / f"{scenario}_episode_comparison.json").write_text(
            json.dumps(json_payload, indent=2),
            encoding="utf-8",
        )
        with (EPISODE_COMPARISON_DIR / f"{scenario}_episode_comparison.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)


def main() -> None:
    payload = rebuild_aggregated_summary()
    rebuild_aggregated_markdown(payload)
    rebuild_episode_comparison()


if __name__ == "__main__":
    main()
