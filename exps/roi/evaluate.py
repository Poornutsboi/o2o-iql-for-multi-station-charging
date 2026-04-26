"""CLI: run the ROI benchmark on each split and emit baseline-format JSON.

Example:
    python -m exps.roi.evaluate \
        --data_dir data/train_dataset \
        --cache_dir exps/roi/cache \
        --output_root exps/results/baselines \
        --label roi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.env_data import CAPACITY, MIN_SEG, TRAVEL_MATRIX
from envs.charging_env import EpisodeBankChargingEnv, travel_time_fn_from_matrix
from exps.roi.estimator import estimate_split_params, save_split_params
from exps.roi.policy import RoiPolicy
from exps.roi.wait_lookup import (
    SplitLookup,
    build_split_lookup,
    load_split_lookup,
    save_split_lookup,
)
from simulator.orchestrator import load_demand_vehicles_from_csv


def _load_episode_paths(split_dir: Path) -> list[Path]:
    paths = sorted(split_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV episode files in '{split_dir}'.")
    return paths


def _make_env_for_episode(episode_path: Path, n_bins: int) -> EpisodeBankChargingEnv:
    episode_bank = [list(load_demand_vehicles_from_csv(str(episode_path)))]
    return EpisodeBankChargingEnv(
        episode_bank=episode_bank,
        station_capacities=CAPACITY.tolist(),
        travel_time_fn=travel_time_fn_from_matrix(TRAVEL_MATRIX),
        min_first_charge=float(MIN_SEG),
        min_second_charge=float(MIN_SEG),
        n_bins=int(n_bins),
    )


def _ensure_split_lookup(
    *,
    split_name: str,
    split_dir: Path,
    cache_dir: Path,
    n_snapshots: int,
    q_max: int,
    delta_max: float,
    delta_step: float,
    seed: int,
) -> SplitLookup:
    lookups_dir = cache_dir / "lookups"
    lookup_path = lookups_dir / f"{split_name}.json"
    if lookup_path.exists():
        print(f"[ROI] Loading cached lookup for split '{split_name}' from {lookup_path}")
        return load_split_lookup(lookup_path)

    print(f"[ROI] No cached lookup found for split '{split_name}'. Building now...")
    params = estimate_split_params(split_dir)
    params_path = cache_dir / "params" / f"{split_name}.json"
    save_split_params(params, params_path)

    delta_grid: list[float] = []
    value = 0.0
    while value <= delta_max + 1e-9:
        delta_grid.append(float(value))
        value += float(delta_step)

    lookup = build_split_lookup(
        params=params,
        q_max=int(q_max),
        delta_grid=delta_grid,
        n_snapshots=int(n_snapshots),
        seed=int(seed),
    )
    save_split_lookup(lookup, lookup_path)
    print(f"[ROI] Saved lookup -> {lookup_path}")
    return lookup


def _run_one_episode(
    episode_path: Path,
    policy: RoiPolicy,
    n_bins: int,
    seed: int,
) -> dict:
    env = _make_env_for_episode(episode_path, n_bins=n_bins)
    try:
        env.reset(seed=int(seed))
        episode_reward = 0.0
        episode_length = 0
        invalid_actions = 0
        final_info: dict = {}
        while env.pending_vehicle is not None:
            action = int(policy.select_action(env))
            _, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            invalid_actions += int(bool(step_info.get("invalid_action", False)))
            final_info = dict(step_info)
            if terminated or truncated:
                break
        assigned_demand = [0.0 for _ in range(env.num_stations)]
        for request in getattr(env, "_submitted_requests", []):
            station_id = int(request.station_id)
            if 0 <= station_id < env.num_stations:
                assigned_demand[station_id] += float(request.charge_duration)
        normalized_demand = [
            float(demand / float(CAPACITY[idx]))
            if idx < len(CAPACITY) and float(CAPACITY[idx]) > 0.0
            else 0.0
            for idx, demand in enumerate(assigned_demand)
        ]
        return {
            "episode_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "episode_name": episode_path.name,
            "vehicle_count": int(len(getattr(env, "_vehicle_total_wait", {}))),
            "total_waiting_time": float(final_info.get("total_wait", 0.0)),
            "mean_waiting_time": float(final_info.get("mean_waiting_time", 0.0)),
            "p95_waiting_time": float(final_info.get("p95_waiting_time", 0.0)),
            "max_waiting_time": float(final_info.get("max_waiting_time", 0.0)),
            "load_imbalance": float(final_info.get("load_imbalance", 0.0)),
            "assigned_demand_by_station": assigned_demand,
            "normalized_assigned_demand_by_station": normalized_demand,
            "invalid_action_count": int(invalid_actions),
            "decision_step_count": int(episode_length),
        }
    finally:
        env.close()


def _summarize(
    records: list[dict],
    label: str,
    scenario: str,
    seed: int,
    data_dir: str,
) -> dict:
    rewards = np.asarray([r["episode_reward"] for r in records], dtype=np.float64)
    lengths = np.asarray([r["episode_length"] for r in records], dtype=np.int64)
    mean_wait = np.asarray([r["mean_waiting_time"] for r in records], dtype=np.float64)
    p95_wait = np.asarray([r["p95_waiting_time"] for r in records], dtype=np.float64)
    max_wait = np.asarray([r["max_waiting_time"] for r in records], dtype=np.float64)
    imbalance = np.asarray([r["load_imbalance"] for r in records], dtype=np.float64)
    total_waiting_time = float(sum(float(r["total_waiting_time"]) for r in records))
    vehicle_count = int(sum(int(r["vehicle_count"]) for r in records))
    invalid_total = int(sum(r["invalid_action_count"] for r in records))
    decision_total = int(sum(r["decision_step_count"] for r in records))
    episode_metrics = [
        {
            "episode_name": str(r["episode_name"]),
            "episode_reward": float(r["episode_reward"]),
            "episode_length": int(r["episode_length"]),
            "vehicle_count": int(r["vehicle_count"]),
            "total_waiting_time": float(r["total_waiting_time"]),
            "mean_waiting_time": float(r["mean_waiting_time"]),
            "p95_waiting_time": float(r["p95_waiting_time"]),
            "max_waiting_time": float(r["max_waiting_time"]),
            "cv_load_imbalance": float(r["load_imbalance"]),
            "assigned_demand_by_station": [
                float(value) for value in r["assigned_demand_by_station"]
            ],
            "normalized_assigned_demand_by_station": [
                float(value) for value in r["normalized_assigned_demand_by_station"]
            ],
        }
        for r in records
    ]
    return {
        "label": label,
        "baseline_name": label,
        "scenario": scenario,
        "seed": int(seed),
        "data_dir": str(data_dir),
        "n_eval_episodes": int(len(records)),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=0)),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "mean_ep_length": float(lengths.mean()),
        "std_ep_length": float(lengths.std(ddof=0)),
        "mean_waiting_time": float(mean_wait.mean()),
        "std_waiting_time": float(mean_wait.std(ddof=0)),
        "mean_p95_waiting_time": float(p95_wait.mean()),
        "mean_max_waiting_time": float(max_wait.mean()),
        "mean_load_imbalance": float(imbalance.mean()),
        "dataset_total_waiting_time": total_waiting_time,
        "dataset_vehicle_count": vehicle_count,
        "dataset_average_waiting_time": (
            float(total_waiting_time / vehicle_count) if vehicle_count > 0 else 0.0
        ),
        "mean_cv_load_imbalance": float(imbalance.mean()),
        "invalid_action_rate": float(invalid_total / decision_total) if decision_total > 0 else 0.0,
        "invalid_action_count": invalid_total,
        "decision_step_count": decision_total,
        "episode_rewards": rewards.tolist(),
        "episode_lengths": lengths.astype(int).tolist(),
        "episode_names": [r["episode_name"] for r in records],
        "episode_metrics": episode_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ROI benchmark policy.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="exps/roi/cache")
    parser.add_argument("--output_root", type=str, default="exps/results/baselines")
    parser.add_argument("--label", type=str, default="roi")
    parser.add_argument("--n_eval_episodes", type=int, default=0,
                        help="0 means evaluate all episodes in the split.")
    parser.add_argument("--n_bins", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--n_snapshots", type=int, default=10000)
    parser.add_argument("--q_max", type=int, default=50)
    parser.add_argument("--delta_max", type=float, default=200.0)
    parser.add_argument("--delta_step", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output_root = Path(args.output_root)

    direct_csvs = sorted(data_root.glob("*.csv"))
    if direct_csvs:
        split_dirs = [data_root]
    else:
        split_dirs = sorted(child for child in data_root.iterdir() if child.is_dir())
    if not split_dirs:
        raise FileNotFoundError(f"No splits or CSV files under '{data_root}'.")

    for split_dir in split_dirs:
        split_name = split_dir.name
        episode_paths = _load_episode_paths(split_dir)
        if args.n_eval_episodes > 0:
            episode_paths = episode_paths[: int(args.n_eval_episodes)]
        print(
            f"\n=== ROI on split '{split_name}': {len(episode_paths)} episode(s) ==="
        )

        lookup = _ensure_split_lookup(
            split_name=split_name,
            split_dir=split_dir,
            cache_dir=cache_dir,
            n_snapshots=int(args.n_snapshots),
            q_max=int(args.q_max),
            delta_max=float(args.delta_max),
            delta_step=float(args.delta_step),
            seed=int(args.seed),
        )
        policy = RoiPolicy(split_lookup=lookup)

        output_dir = output_root / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        split_data_dir = str(data_root / split_name) if direct_csvs == [] else str(data_root)
        seed_values = args.seeds if args.seeds else [int(args.seed)]
        summaries: list[dict] = []
        for seed in seed_values:
            label = f"{args.label}-seed{int(seed)}"
            records: list[dict] = []
            print(f"  -- seed {int(seed)}")
            for episode_idx, episode_path in enumerate(episode_paths):
                record = _run_one_episode(
                    episode_path=episode_path,
                    policy=policy,
                    n_bins=int(args.n_bins),
                    seed=int(seed) + episode_idx,
                )
                records.append(record)
                if (episode_idx + 1) % 25 == 0 or episode_idx == len(episode_paths) - 1:
                    print(
                        f"  [{episode_idx + 1:>4}/{len(episode_paths)}] "
                        f"reward={record['episode_reward']:.1f}  "
                        f"mean_wait={record['mean_waiting_time']:.2f}"
                    )

            summary = _summarize(
                records,
                label=label,
                scenario=split_name,
                seed=int(seed),
                data_dir=split_data_dir,
            )
            summaries.append(summary)
            output_path = output_dir / f"seed{int(seed)}.json"
            output_path.write_text(
                json.dumps(
                    {
                        "data_dir": split_data_dir,
                        "split_name": split_name,
                        "results": [summary],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"Saved ROI seed summary -> {output_path}\n"
                f"  mean_reward={summary['mean_reward']:.2f}  "
                f"mean_waiting_time={summary['mean_waiting_time']:.2f}  "
                f"mean_p95_waiting_time={summary['mean_p95_waiting_time']:.2f}"
            )

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "data_dir": split_data_dir,
                    "split_name": split_name,
                    "results": summaries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved ROI split summary -> {summary_path}")


if __name__ == "__main__":
    main()
