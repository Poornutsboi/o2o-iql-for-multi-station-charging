from __future__ import annotations

import math
from typing import Iterable


DEFAULT_NUM_STATIONS = 7


def _validate_n_bins(n_bins: int) -> int:
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    return n_bins


def _validate_num_stations(num_stations: int) -> int:
    num_stations = int(num_stations)
    if num_stations <= 0:
        raise ValueError("num_stations must be positive.")
    return num_stations


def _no_split_choice(num_stations: int) -> int:
    return _validate_num_stations(num_stations)


def frac_from_bin(frac_bin: int, n_bins: int) -> float:
    n_bins = _validate_n_bins(n_bins)
    frac_bin = int(frac_bin)
    if frac_bin < 0 or frac_bin >= n_bins:
        raise ValueError(f"frac_bin must be in [0, {n_bins - 1}].")
    return float(frac_bin / (n_bins - 1))


def encode_maskable_action(
    second_choice: int,
    frac_bin: int,
    n_bins: int,
    num_stations: int = DEFAULT_NUM_STATIONS,
) -> int:
    n_bins = _validate_n_bins(n_bins)
    num_stations = _validate_num_stations(num_stations)
    second_choice = int(second_choice)
    frac_bin = int(frac_bin)

    if second_choice == _no_split_choice(num_stations):
        frac_bin = n_bins - 1
    elif second_choice < 0 or second_choice >= num_stations:
        raise ValueError(
            f"second_choice must be in [0, {num_stations - 1}] or equal to {num_stations}."
        )

    if frac_bin < 0 or frac_bin >= n_bins:
        raise ValueError(f"frac_bin must be in [0, {n_bins - 1}].")

    return int((second_choice * n_bins) + frac_bin)


def decode_maskable_action(
    action_int: int,
    n_bins: int,
    num_stations: int = DEFAULT_NUM_STATIONS,
) -> tuple[int, int]:
    n_bins = _validate_n_bins(n_bins)
    num_stations = _validate_num_stations(num_stations)
    action_int = int(action_int)
    max_action = ((num_stations + 1) * n_bins) - 1
    if action_int < 0 or action_int > max_action:
        raise ValueError(f"action_int must be in [0, {max_action}].")

    second_choice = int(action_int // n_bins)
    frac_bin = int(action_int % n_bins)
    return second_choice, frac_bin


def no_split_action_int(
    n_bins: int,
    num_stations: int = DEFAULT_NUM_STATIONS,
) -> int:
    return encode_maskable_action(
        second_choice=_no_split_choice(num_stations),
        frac_bin=_validate_n_bins(n_bins) - 1,
        n_bins=n_bins,
        num_stations=num_stations,
    )


def compute_split_bin_bounds(
    total_duration: float,
    n_bins: int,
    t_first_min: float = 0.0,
    t_second_min: float = 0.0,
    tol: float = 1e-9,
) -> tuple[int, int] | None:
    n_bins = _validate_n_bins(n_bins)
    total_duration = float(total_duration)
    t_first_min = max(0.0, float(t_first_min))
    t_second_min = max(0.0, float(t_second_min))

    if total_duration <= tol:
        return None
    if total_duration + tol < (t_first_min + t_second_min):
        return None

    scale = n_bins - 1
    min_bin = int(math.ceil(((t_first_min / total_duration) * scale) - tol))
    max_bin = int(math.floor(((1.0 - (t_second_min / total_duration)) * scale) + tol))
    min_bin = max(0, min_bin)
    max_bin = min(n_bins - 2, max_bin)
    if min_bin > max_bin:
        return None
    return min_bin, max_bin


def iter_valid_maskable_actions(
    route: Iterable[int],
    n_bins: int,
    total_duration: float | None = None,
    t_first_min: float = 0.0,
    t_second_min: float = 0.0,
    num_stations: int = DEFAULT_NUM_STATIONS,
) -> list[int]:
    n_bins = _validate_n_bins(n_bins)
    num_stations = _validate_num_stations(num_stations)

    actions = [no_split_action_int(n_bins=n_bins, num_stations=num_stations)]
    downstream_stations: list[int] = []
    seen: set[int] = set()
    route_list = [int(station_id) for station_id in route]
    for station_id in route_list[1:]:
        if station_id in seen:
            continue
        if station_id < 0 or station_id >= num_stations:
            continue
        seen.add(station_id)
        downstream_stations.append(station_id)

    if not downstream_stations:
        return actions

    if total_duration is None:
        split_bounds = (0, n_bins - 2)
    else:
        split_bounds = compute_split_bin_bounds(
            total_duration=total_duration,
            n_bins=n_bins,
            t_first_min=t_first_min,
            t_second_min=t_second_min,
        )
        if split_bounds is None:
            return actions

    min_bin, max_bin = split_bounds
    for station_id in downstream_stations:
        for frac_bin in range(min_bin, max_bin + 1):
            actions.append(
                encode_maskable_action(
                    second_choice=station_id,
                    frac_bin=frac_bin,
                    n_bins=n_bins,
                    num_stations=num_stations,
                )
            )
    return actions
