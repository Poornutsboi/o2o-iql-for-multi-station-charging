"""ROI policy: pick the action with the minimum system wait pressure.

Implements the action-selection step of Algorithm 'ROI' from
Dastpak et al. 2024. Because the vehicle route is fixed in this project,
we enumerate split-charging actions on the route and score them with a
duration-weighted wait-pressure objective backed by the precomputed
M/M/C/inf wait lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from envs.charging_env import EpisodeBankChargingEnv
from envs.maskable_actions import (
    decode_maskable_action,
    frac_from_bin,
    iter_valid_maskable_actions,
    no_split_action_int,
)
from exps.roi.arrival_bound import StationOccupancy, evaluate_decision
from exps.roi.wait_lookup import SplitLookup, StationLookup, lookup_wait


def _occupancy_from_station_payload(
    station_id: int, payload: Mapping[str, object]
) -> StationOccupancy:
    charger_status = payload["charger_status"]
    queue_waiting_time = payload["queue_waiting_time"]
    free_chargers = sum(1 for value in charger_status if float(value) <= 0.0)
    queue_length = len(queue_waiting_time)
    return StationOccupancy(
        station_id=int(station_id),
        q_observed=int(queue_length),
        g_observed=int(free_chargers),
    )


@dataclass(frozen=True)
class RoiScoredAction:
    action_int: int
    second_choice: int
    frac_bin: int
    total_wait: float
    total_arrival_time: float


class RoiPolicy:
    """ROI policy that scores each masked action and picks the cheapest.

    A pure route-tail arrival objective degenerates in this environment:
    split and no-split have the same total charging duration and route travel,
    while split adds a second non-negative wait. To make ROI act as a system
    heuristic, the primary score is wait pressure:

        first_wait * first_charge_duration
        + second_wait * second_charge_duration

    This approximates how much charging demand is assigned to congested
    stations. The forward arrival lower bound remains a tie-breaker.
    """

    def __init__(self, split_lookup: SplitLookup) -> None:
        self._lookup_by_station: dict[int, StationLookup] = {
            int(station.station_id): station for station in split_lookup.stations
        }
        self.split_lookup = split_lookup

    def _wait_estimator(self, station_id: int, q: int, g: int, delta: float) -> float:
        station_lookup = self._lookup_by_station.get(int(station_id))
        if station_lookup is None:
            return 0.0
        return lookup_wait(
            station_lookup=station_lookup,
            q=int(q),
            g=int(g),
            delta=float(delta),
        )

    def _build_occupancy(self, env: EpisodeBankChargingEnv) -> dict[int, StationOccupancy]:
        sim_state = env._sim.get_state(query_time=float(env.clock))  # noqa: SLF001
        return {
            int(station_id): _occupancy_from_station_payload(int(station_id), payload)
            for station_id, payload in sim_state["stations"].items()
        }

    def select_action(self, env: EpisodeBankChargingEnv) -> int:
        if env.pending_vehicle is None:
            return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)

        vehicle = env.pending_vehicle
        occupancy = self._build_occupancy(env)
        travel_matrix = env._orchestrator._build_travel_time_matrix()  # noqa: SLF001

        def _travel_time(station_a: int, station_b: int) -> float:
            return float(travel_matrix[int(station_a)][int(station_b)])

        valid_actions = iter_valid_maskable_actions(
            route=vehicle.route,
            n_bins=env.n_bins,
            total_duration=float(vehicle.duration),
            t_first_min=float(env.min_first_charge),
            t_second_min=float(env.min_second_charge),
            num_stations=env.num_stations,
        )

        canonical_no_split = no_split_action_int(
            n_bins=env.n_bins, num_stations=env.num_stations
        )
        best_action: int = canonical_no_split
        best_score = float("inf")
        best_tiebreak = float("inf")

        for action_int in valid_actions:
            second_choice, frac_bin = decode_maskable_action(
                action_int=int(action_int),
                n_bins=env.n_bins,
                num_stations=env.num_stations,
            )
            if action_int == canonical_no_split or second_choice == env.num_stations:
                first_duration = float(vehicle.duration)
                second_station_id: int | None = None
                second_duration = 0.0
            else:
                fraction = frac_from_bin(frac_bin, env.n_bins)
                first_duration = fraction * float(vehicle.duration)
                second_duration = float(vehicle.duration) - first_duration
                second_station_id = int(second_choice)

            evaluation = evaluate_decision(
                now=float(env.clock),
                route=vehicle.route,
                occupancy=occupancy,
                travel_time=_travel_time,
                first_charge_duration=first_duration,
                second_station_id=second_station_id,
                second_charge_duration=second_duration,
                wait_estimator=self._wait_estimator,
            )
            score = (
                float(evaluation.first_wait) * float(first_duration)
                + float(evaluation.second_wait) * float(second_duration)
            )
            tiebreak = float(evaluation.total_arrival_time)
            if score + 1e-9 < best_score or (
                abs(score - best_score) <= 1e-9 and tiebreak < best_tiebreak
            ):
                best_score = score
                best_tiebreak = tiebreak
                best_action = int(action_int)

        return int(best_action)


def make_roi_baseline(split_lookup: SplitLookup):
    """Adapter so RoiPolicy plugs into the existing BaselineFn signature."""
    policy = RoiPolicy(split_lookup=split_lookup)

    def _select(env: EpisodeBankChargingEnv) -> int:
        return policy.select_action(env)

    return _select
