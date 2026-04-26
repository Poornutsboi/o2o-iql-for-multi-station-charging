"""Forward arrival lower bound (paper Sec 4.1.2 / Algorithm 1).

The paper's Algorithm 1 propagates a lower bound on arrival time along a
candidate path by summing travel times, charge times, and wait estimates
W(q, g, Delta) at every charging station encountered. In this project the
vehicle route is fixed, so the recursion collapses to: charge at most
twice (first station and optional second station), with deterministic
travel between them.

This module exposes a single :func:`evaluate_decision` that returns the
expected arrival time at the route tail for a given split-charging
decision, given a wait-estimator callback. The returned cost is used by
the ROI policy to score and rank actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

WaitEstimator = Callable[[int, int, int, float], float]
"""(station_id, q_observed, g_observed, delta_minutes) -> expected wait minutes."""


@dataclass(frozen=True)
class StationOccupancy:
    station_id: int
    q_observed: int
    g_observed: int


@dataclass(frozen=True)
class DecisionEvaluation:
    total_wait: float
    total_arrival_time: float
    first_wait: float
    second_wait: float
    second_delta: float


def evaluate_decision(
    *,
    now: float,
    route: Sequence[int],
    occupancy: dict[int, StationOccupancy],
    travel_time: Callable[[int, int], float],
    first_charge_duration: float,
    second_station_id: int | None,
    second_charge_duration: float,
    wait_estimator: WaitEstimator,
) -> DecisionEvaluation:
    """Forward-propagate arrival times along the route for one decision."""
    if not route:
        raise ValueError("route must be non-empty.")

    first_station = int(route[0])
    first_state = occupancy[first_station]
    first_wait = float(
        wait_estimator(
            first_station,
            int(first_state.q_observed),
            int(first_state.g_observed),
            0.0,
        )
    )
    t_finish_first = float(now) + first_wait + float(first_charge_duration)

    if second_station_id is None or float(second_charge_duration) <= 0.0:
        second_wait = 0.0
        second_delta = 0.0
        tail_arrival = t_finish_first
        for station_a, station_b in zip(route[:-1], route[1:]):
            tail_arrival += float(travel_time(int(station_a), int(station_b)))
        return DecisionEvaluation(
            total_wait=first_wait,
            total_arrival_time=tail_arrival,
            first_wait=first_wait,
            second_wait=0.0,
            second_delta=0.0,
        )

    second_idx = list(route).index(int(second_station_id))
    if second_idx <= 0:
        raise ValueError("second_station_id must appear strictly downstream of route[0].")

    t_arrive_second = t_finish_first
    for station_a, station_b in zip(route[: second_idx], route[1 : second_idx + 1]):
        t_arrive_second += float(travel_time(int(station_a), int(station_b)))
    second_delta = float(t_arrive_second - float(now))
    second_state = occupancy[int(second_station_id)]
    second_wait = float(
        wait_estimator(
            int(second_station_id),
            int(second_state.q_observed),
            int(second_state.g_observed),
            second_delta,
        )
    )
    t_finish_second = t_arrive_second + second_wait + float(second_charge_duration)

    tail_arrival = t_finish_second
    for station_a, station_b in zip(route[second_idx:-1], route[second_idx + 1 :]):
        tail_arrival += float(travel_time(int(station_a), int(station_b)))

    return DecisionEvaluation(
        total_wait=first_wait + second_wait,
        total_arrival_time=tail_arrival,
        first_wait=first_wait,
        second_wait=second_wait,
        second_delta=second_delta,
    )
