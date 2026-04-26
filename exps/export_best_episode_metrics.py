"""Export per-episode best metric tables for all evaluated methods."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exps.rebuild_result_summaries import _normalize_result


RESULTS_ROOT = Path("exps/results")
SCENARIOS = ["normal", "bias", "extreme"]
METRICS = [
    ("average waiting time", "mean_waiting_time"),
    ("p95 waiting time", "p95_waiting_time"),
    ("load imbalance", "cv_load_imbalance"),
]
METHOD_SOURCES = {
    "o2o-iql": RESULTS_ROOT / "o2o-iql",
    "ppo": RESULTS_ROOT / "ppo",
    "eoi": RESULTS_ROOT / "eoi",
}
BASELINES = ["all-no-split", "greedy-split", "station-assignment"]


def episode_id(name: str) -> str:
    match = re.search(r"(\d+)", str(name))
    return match.group(1) if match else str(name)


def load_summary(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return [_normalize_result(result) for result in payload["results"]]


def export_table(results: list[dict], out_path: Path) -> None:
    per_episode: dict[str, dict[str, list[float]]] = {}
    for result in results:
        for item in result["episode_metrics"]:
            eid = episode_id(item["episode_name"])
            per_episode.setdefault(eid, {key: [] for _, key in METRICS})
            for _label, key in METRICS:
                per_episode[eid][key].append(float(item[key]))

    episode_ids = sorted(
        per_episode,
        key=lambda value: int(value) if value.isdigit() else value,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *episode_ids])
        for label, key in METRICS:
            writer.writerow([label, *[min(per_episode[eid][key]) for eid in episode_ids]])


def main() -> None:
    written: list[Path] = []
    for method, method_root in METHOD_SOURCES.items():
        for scenario in SCENARIOS:
            results = load_summary(method_root / scenario / "summary.json")
            out_path = method_root / "summary" / f"{scenario}_best_episode_metrics.csv"
            export_table(results, out_path)
            written.append(out_path)

    baseline_summary_root = RESULTS_ROOT / "baselines_test" / "summary"
    for baseline in BASELINES:
        for scenario in SCENARIOS:
            results = load_summary(
                RESULTS_ROOT / "baselines_test" / scenario / f"{baseline}.json"
            )
            out_path = baseline_summary_root / f"{baseline}_{scenario}_best_episode_metrics.csv"
            export_table(results, out_path)
            written.append(out_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
