"""Estimate per-station M/M/C parameters from a train_dataset split.

For each station j we estimate
    lambda_j  : mean arrival rate (vehicles per minute)
    mu_j      : per-charger service rate (1 / mean charge minutes)
    C_j       : number of chargers (taken from data.env_data.CAPACITY)

Arrival counting follows the no-split convention used by the baseline
policies: a vehicle is counted as arriving at its first route station,
because that is where the simulator forces it to charge unless a split
is chosen. The estimator is intentionally policy-agnostic; the resulting
parameters describe the background charging load and are used only as a
prior for the wait-time lookup.
"""
from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from data.env_data import CAPACITY


@dataclass(frozen=True)
class StationParams:
    station_id: int
    capacity: int
    lambda_per_min: float
    mu_per_min: float
    n_arrivals: int
    mean_duration: float
    rho: float


@dataclass(frozen=True)
class SplitParams:
    split_name: str
    horizon_minutes: float
    n_episodes: int
    stations: tuple[StationParams, ...]


def _parse_route(value: str) -> list[int]:
    return [int(station_id) for station_id in ast.literal_eval(value)]


def _iter_episode_csvs(split_dir: Path) -> list[Path]:
    csvs = sorted(split_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV episode files found in '{split_dir}'.")
    return csvs


def estimate_split_params(
    split_dir: str | Path,
    *,
    capacities: Iterable[int] | None = None,
    eps: float = 1e-9,
) -> SplitParams:
    """Estimate per-station queue parameters for a single split directory."""
    split_path = Path(split_dir)
    capacity_list = list(int(value) for value in (capacities if capacities is not None else CAPACITY.tolist()))
    num_stations = len(capacity_list)

    arrival_counts = [0 for _ in range(num_stations)]
    duration_sum = [0.0 for _ in range(num_stations)]
    horizon_total = 0.0
    n_episodes = 0

    for csv_path in _iter_episode_csvs(split_path):
        n_episodes += 1
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            episode_horizon = 0.0
            for record in reader:
                route = _parse_route(record["Route"])
                station_id = int(route[0])
                duration = float(record["Duration"])
                if 0 <= station_id < num_stations:
                    arrival_counts[station_id] += 1
                    duration_sum[station_id] += duration
                episode_horizon = max(
                    episode_horizon,
                    float(record.get("episode_horizon_minutes", 0.0) or 0.0),
                )
            horizon_total += episode_horizon

    if horizon_total <= 0.0:
        raise ValueError(f"Total horizon for '{split_path}' is zero; cannot estimate lambda.")

    stations: list[StationParams] = []
    for station_id, capacity in enumerate(capacity_list):
        n_arrivals = int(arrival_counts[station_id])
        mean_duration = float(duration_sum[station_id] / n_arrivals) if n_arrivals > 0 else 0.0
        lambda_per_min = float(n_arrivals / horizon_total)
        mu_per_min = float(1.0 / mean_duration) if mean_duration > eps else 0.0
        rho = float(lambda_per_min / (capacity * mu_per_min)) if mu_per_min > eps else 0.0
        stations.append(
            StationParams(
                station_id=station_id,
                capacity=int(capacity),
                lambda_per_min=lambda_per_min,
                mu_per_min=mu_per_min,
                n_arrivals=n_arrivals,
                mean_duration=mean_duration,
                rho=rho,
            )
        )

    return SplitParams(
        split_name=split_path.name,
        horizon_minutes=horizon_total,
        n_episodes=n_episodes,
        stations=tuple(stations),
    )


def save_split_params(params: SplitParams, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_name": params.split_name,
        "horizon_minutes": params.horizon_minutes,
        "n_episodes": params.n_episodes,
        "stations": [asdict(station) for station in params.stations],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_params(input_path: str | Path) -> SplitParams:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    stations = tuple(StationParams(**entry) for entry in payload["stations"])
    return SplitParams(
        split_name=str(payload["split_name"]),
        horizon_minutes=float(payload["horizon_minutes"]),
        n_episodes=int(payload["n_episodes"]),
        stations=stations,
    )
