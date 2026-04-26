"""Build M/M/C/inf wait-time lookup table per station (paper Sec 4.1.1).

The paper's W(q, g, Delta) function returns the expected wait time of a
probe vehicle that arrives Delta minutes from now, given that the station
currently has queue length q and g free chargers. The original paper used
M/M/1/kappa; this project's stations have multiple chargers so we use
M/M/C/inf instead.

For each station the lookup table is precomputed by running one long
M/M/C simulation, sampling (q_obs, g_obs) snapshots, then for each
(snapshot, Delta) pair branching an independent forward simulation of
length Delta and probing the wait that an arrival would experience at
the end of that branch.
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from exps.roi.estimator import SplitParams, StationParams, load_split_params


@dataclass(frozen=True)
class StationLookup:
    station_id: int
    capacity: int
    q_max: int
    delta_grid: tuple[float, ...]
    table: np.ndarray  # shape (q_max + 1, capacity + 1, len(delta_grid))
    sample_count: np.ndarray  # shape (q_max + 1, capacity + 1, len(delta_grid))


@dataclass(frozen=True)
class SplitLookup:
    split_name: str
    q_max: int
    delta_grid: tuple[float, ...]
    stations: tuple[StationLookup, ...]
    config: dict


class _MMCSim:
    """Tiny event-driven M/M/C/inf simulator with FIFO queue."""

    def __init__(
        self,
        lam: float,
        mu: float,
        capacity: int,
        rng: np.random.Generator,
        *,
        clock: float = 0.0,
        chargers: Sequence[float] | None = None,
        queue: Sequence[float] | None = None,
        next_arrival: float | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.lam = float(lam)
        self.mu = float(mu)
        self.capacity = int(capacity)
        self.rng = rng
        self.t = float(clock)
        self.chargers: list[float] = (
            [float(value) for value in chargers]
            if chargers is not None
            else [self.t for _ in range(self.capacity)]
        )
        if len(self.chargers) != self.capacity:
            raise ValueError("chargers length must equal capacity.")
        heapq.heapify(self.chargers)
        self.queue: list[float] = (
            [float(value) for value in queue] if queue is not None else []
        )
        if next_arrival is not None:
            self.next_arrival = float(next_arrival)
        elif self.lam > 0.0:
            self.next_arrival = self.t + float(self.rng.exponential(1.0 / self.lam))
        else:
            self.next_arrival = float("inf")

    def _draw_service(self) -> float:
        return float(self.rng.exponential(1.0 / self.mu))

    def _earliest_release(self) -> float:
        return self.chargers[0]

    def _dispatch_ready(self, now: float) -> None:
        while self.queue and self._earliest_release() <= now:
            free_at = heapq.heappop(self.chargers)
            self.queue.pop(0)
            start = max(free_at, now)
            service = self._draw_service()
            heapq.heappush(self.chargers, start + service)

    def _on_arrival(self, arrival_time: float) -> None:
        self._dispatch_ready(arrival_time)
        if self._earliest_release() <= arrival_time:
            free_at = heapq.heappop(self.chargers)
            service = self._draw_service()
            heapq.heappush(self.chargers, max(free_at, arrival_time) + service)
        else:
            self.queue.append(arrival_time)
        if self.lam > 0.0:
            self.next_arrival = arrival_time + float(self.rng.exponential(1.0 / self.lam))
        else:
            self.next_arrival = float("inf")

    def advance_to(self, target_time: float) -> None:
        target_time = float(target_time)
        if target_time < self.t:
            raise ValueError("cannot advance backwards in time.")
        while True:
            next_release = self._earliest_release() if self.queue else float("inf")
            next_event = min(self.next_arrival, next_release)
            if next_event >= target_time:
                self.t = target_time
                self._dispatch_ready(target_time)
                return
            self.t = next_event
            if next_event == self.next_arrival:
                self._on_arrival(next_event)
            else:
                self._dispatch_ready(next_event)

    def observe(self, query_time: float) -> tuple[int, int]:
        self.advance_to(query_time)
        free = sum(1 for release in self.chargers if release <= query_time)
        return len(self.queue), free

    def snapshot_state(self) -> dict:
        return {
            "t": float(self.t),
            "chargers": list(self.chargers),
            "queue": list(self.queue),
            "next_arrival": float(self.next_arrival),
        }

    def probe_wait_at(self, probe_time: float) -> float:
        """Inject a probe arrival at probe_time and return the wait, no mutation."""
        self.advance_to(probe_time)
        release_heap = list(self.chargers)
        heapq.heapify(release_heap)
        for _ in self.queue:
            free_at = heapq.heappop(release_heap)
            start = max(free_at, probe_time)
            service = self._draw_service()
            heapq.heappush(release_heap, start + service)
        free_at = heapq.heappop(release_heap)
        return max(0.0, free_at - probe_time)


def _build_station_lookup(
    station: StationParams,
    *,
    q_max: int,
    delta_grid: Sequence[float],
    n_snapshots: int,
    rng: np.random.Generator,
) -> StationLookup:
    capacity = int(station.capacity)
    table = np.zeros((q_max + 1, capacity + 1, len(delta_grid)), dtype=np.float64)
    count = np.zeros((q_max + 1, capacity + 1, len(delta_grid)), dtype=np.int64)

    if station.lambda_per_min <= 0.0 or station.mu_per_min <= 0.0:
        return StationLookup(
            station_id=int(station.station_id),
            capacity=capacity,
            q_max=int(q_max),
            delta_grid=tuple(float(value) for value in delta_grid),
            table=table,
            sample_count=count,
        )

    mean_inter = 1.0 / station.lambda_per_min
    burn_in = max(mean_inter * 100.0, 100.0)
    spacing = max(mean_inter * 4.0, 0.5)

    sim = _MMCSim(
        lam=station.lambda_per_min,
        mu=station.mu_per_min,
        capacity=capacity,
        rng=rng,
    )

    for snapshot_idx in range(int(n_snapshots)):
        t_snap = burn_in + float(snapshot_idx) * spacing
        q_obs, g_obs = sim.observe(t_snap)
        q_idx = int(min(q_obs, q_max))
        g_idx = int(g_obs)
        saved = sim.snapshot_state()
        for delta_idx, delta in enumerate(delta_grid):
            branch_seed = int(rng.integers(0, 2**31 - 1))
            branch_rng = np.random.default_rng(branch_seed)
            branch = _MMCSim(
                lam=station.lambda_per_min,
                mu=station.mu_per_min,
                capacity=capacity,
                rng=branch_rng,
                clock=saved["t"],
                chargers=saved["chargers"],
                queue=saved["queue"],
                next_arrival=saved["next_arrival"],
            )
            wait = branch.probe_wait_at(t_snap + float(delta))
            table[q_idx, g_idx, delta_idx] += wait
            count[q_idx, g_idx, delta_idx] += 1

    safe_count = np.maximum(count, 1)
    averaged = np.where(count > 0, table / safe_count, 0.0)
    return StationLookup(
        station_id=int(station.station_id),
        capacity=capacity,
        q_max=int(q_max),
        delta_grid=tuple(float(value) for value in delta_grid),
        table=averaged,
        sample_count=count,
    )


def build_split_lookup(
    params: SplitParams,
    *,
    q_max: int = 50,
    delta_grid: Iterable[float] | None = None,
    n_snapshots: int = 10000,
    seed: int = 0,
) -> SplitLookup:
    delta_grid_tuple: tuple[float, ...] = tuple(
        float(value)
        for value in (delta_grid if delta_grid is not None else range(0, 205, 5))
    )
    master_rng = np.random.default_rng(seed)
    station_lookups: list[StationLookup] = []
    for station in params.stations:
        station_seed = int(master_rng.integers(0, 2**31 - 1))
        station_rng = np.random.default_rng(station_seed)
        station_lookups.append(
            _build_station_lookup(
                station=station,
                q_max=int(q_max),
                delta_grid=delta_grid_tuple,
                n_snapshots=int(n_snapshots),
                rng=station_rng,
            )
        )
    config = {
        "q_max": int(q_max),
        "delta_grid": list(delta_grid_tuple),
        "n_snapshots": int(n_snapshots),
        "seed": int(seed),
    }
    return SplitLookup(
        split_name=str(params.split_name),
        q_max=int(q_max),
        delta_grid=delta_grid_tuple,
        stations=tuple(station_lookups),
        config=config,
    )


def save_split_lookup(lookup: SplitLookup, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_name": lookup.split_name,
        "q_max": int(lookup.q_max),
        "delta_grid": list(lookup.delta_grid),
        "config": dict(lookup.config),
        "stations": [
            {
                "station_id": int(station.station_id),
                "capacity": int(station.capacity),
                "table": station.table.tolist(),
                "sample_count": station.sample_count.tolist(),
            }
            for station in lookup.stations
        ],
    }
    output.write_text(json.dumps(payload), encoding="utf-8")


def load_split_lookup(input_path: str | Path) -> SplitLookup:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    delta_grid = tuple(float(value) for value in payload["delta_grid"])
    q_max = int(payload["q_max"])
    stations = tuple(
        StationLookup(
            station_id=int(entry["station_id"]),
            capacity=int(entry["capacity"]),
            q_max=q_max,
            delta_grid=delta_grid,
            table=np.asarray(entry["table"], dtype=np.float64),
            sample_count=np.asarray(entry["sample_count"], dtype=np.int64),
        )
        for entry in payload["stations"]
    )
    return SplitLookup(
        split_name=str(payload["split_name"]),
        q_max=q_max,
        delta_grid=delta_grid,
        stations=stations,
        config=dict(payload.get("config", {})),
    )


def lookup_wait(
    station_lookup: StationLookup,
    *,
    q: int,
    g: int,
    delta: float,
) -> float:
    """Bilinear lookup along the Delta axis; nearest-neighbor on (q, g)."""
    q_idx = int(np.clip(q, 0, station_lookup.q_max))
    g_idx = int(np.clip(g, 0, station_lookup.capacity))
    grid = np.asarray(station_lookup.delta_grid, dtype=np.float64)
    delta = float(max(0.0, delta))
    if delta <= grid[0]:
        return float(station_lookup.table[q_idx, g_idx, 0])
    if delta >= grid[-1]:
        return float(station_lookup.table[q_idx, g_idx, -1])
    upper = int(np.searchsorted(grid, delta, side="left"))
    lower = max(0, upper - 1)
    if grid[upper] == grid[lower]:
        return float(station_lookup.table[q_idx, g_idx, upper])
    weight = (delta - grid[lower]) / (grid[upper] - grid[lower])
    lower_value = float(station_lookup.table[q_idx, g_idx, lower])
    upper_value = float(station_lookup.table[q_idx, g_idx, upper])
    return float((1.0 - weight) * lower_value + weight * upper_value)
