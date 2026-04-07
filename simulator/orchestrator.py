from __future__ import annotations

import ast
import csv
import math
from dataclasses import asdict
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from simulator.commitment import Commitment, CommitmentStore
from simulator.history import ChargingHistoryLog
from simulator.models import ChargingAssignment, ChargingRequest
from simulator.planner import ChargingDecision, DecisionVehicle, SplitPlanner
from simulator.simulator import SimulatorCore

if TYPE_CHECKING:
    from envs.charging_env import Vehicle


def _parse_demand_route(route_value: Any) -> list[int]:
    if isinstance(route_value, str):
        parsed_value = ast.literal_eval(route_value)
    else:
        parsed_value = route_value

    if not isinstance(parsed_value, (list, tuple)):
        raise ValueError("Demand record Route must be a list or tuple of station ids.")

    route = [int(station_id) for station_id in parsed_value]
    if not route:
        raise ValueError("Demand record Route must contain at least one station.")
    return route


def demand_record_to_vehicle(record: Mapping[str, Any]) -> Vehicle:
    from envs.charging_env import Vehicle

    return Vehicle(
        vid=int(record["Vehicle_ID"]),
        arrival_time=float(record["Arrival_time"]),
        route=_parse_demand_route(record["Route"]),
        duration=float(record["Duration"]),
    )


def demand_records_to_vehicles(records: Iterable[Mapping[str, Any]]) -> list[Vehicle]:
    return [demand_record_to_vehicle(record) for record in records]


def load_demand_vehicles_from_csv(csv_path: str | PathLike[str]) -> list[Vehicle]:
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        return demand_records_to_vehicles(csv.DictReader(csv_file))


class DemandForecaster:
    def __init__(self, station_ids: list[int]) -> None:
        self._station_ids = sorted(int(station_id) for station_id in station_ids)
        self._metric_size = 1 + max(self._station_ids)

    def predict(
        self,
        method: str,
        now: float,
        history_log: ChargingHistoryLog,
        params: dict[str, float] | None = None,
    ) -> list[float]:
        if method == "exponential-decay":
            return self._predict_exponential_decay(
                now=now,
                history_log=history_log,
                params=params,
            )
        raise ValueError(f"Unsupported demand prediction method: {method}")

    def _predict_exponential_decay(
        self,
        now: float,
        history_log: ChargingHistoryLog,
        params : dict[str, float] | None = None,
    ) -> list[float]:
        config = params or {}
        horizon = float(config.get("horizon", 15.0))
        decay_tau = float(config.get("decay_tau", 15.0))
        counts = [0.0 for _ in range(self._metric_size)]
        kernel_multiplier = 1.0 - math.exp(-horizon / decay_tau)

        for record in history_log.records():
            arrival_time = float(record.arrival_time)
            if arrival_time > float(now):
                continue
            elapsed = float(now) - arrival_time
            counts[int(record.station_id)] += math.exp(-elapsed / decay_tau) * kernel_multiplier

        return counts


class SplitChargingOrchestrator:
    def __init__(
        self,
        simulator: SimulatorCore,
        travel_time_estimator: Callable[[int, int], float] | None = None,
        planner: SplitPlanner | None = None,
        demand_prediction_method: str = "exponential-decay",
        demand_forecaster: DemandForecaster | None = None,
    ) -> None:
        self.simulator = simulator
        self.travel_time_estimator = travel_time_estimator or (lambda _a, _b: 0.0)
        self.planner = planner or SplitPlanner()
        self.commitment_store = CommitmentStore(station_ids=self.simulator.station_ids)
        self.demand_prediction_method = str(demand_prediction_method)
        self.demand_forecaster = demand_forecaster or DemandForecaster(
            station_ids=self.simulator.station_ids,
        )

    def build_observation(
        self,
        current_ev: DecisionVehicle,
        now: float,
        vehicle_info: bool = False,
    ) -> dict:
        return {
            "sim_state": self.simulator.get_state(query_time=now, vehicle_info=vehicle_info),
            "commitment_features": self.commitment_store.summary(now=now),
            "current_ev": asdict(current_ev),
            "future_demand": self.demand_forecaster.predict(
                method=self.demand_prediction_method,
                now=now,
                history_log=self.simulator.history_log,
            ),
            "travel_time_matrix": self._build_travel_time_matrix(),
        }

    def apply_decision(
        self,
        current_ev: DecisionVehicle,
        decision: ChargingDecision,
    ) -> dict:
        first_request, second_leg_plan = self.planner.translate(
            current_ev=current_ev,
            decision=decision,
        )
        first_assignment = self.simulator.submit_arrival(first_request)

        commitment = None
        if second_leg_plan is not None:
            expected_arrival_time = float(first_assignment.end_time) + float(
                self.travel_time_estimator(
                    int(current_ev.station_id),
                    int(second_leg_plan.target_station_id),
                )
            )
            commitment = Commitment(
                vehicle_id=int(second_leg_plan.vehicle_id),
                target_station_id=int(second_leg_plan.target_station_id),
                expected_arrival_time=float(expected_arrival_time),
                planned_second_charge_duration=float(second_leg_plan.charge_duration),
                created_at=float(current_ev.arrival_time),
            )
            self.commitment_store.add(commitment)

        return {
            "first_request": first_request,
            "first_assignment": first_assignment,
            "commitment": commitment,
        }

    def submit_second_leg_arrival(
        self,
        vehicle_id: int,
        actual_arrival_time: float,
    ) -> ChargingAssignment:
        commitment = self.commitment_store.pop(vehicle_id)
        second_request = ChargingRequest(
            vehicle_id=int(commitment.vehicle_id),
            station_id=int(commitment.target_station_id),
            charge_duration=float(commitment.planned_second_charge_duration),
            arrival_time=float(actual_arrival_time),
        )
        return self.simulator.submit_arrival(second_request)

    def _build_travel_time_matrix(self) -> list[list[float]]:
        station_ids = self.simulator.station_ids
        matrix_size = 1 + max(station_ids)
        matrix = [
            [0.0 for _ in range(matrix_size)]
            for _ in range(matrix_size)
        ]

        for from_station in station_ids:
            for to_station in station_ids:
                if from_station == to_station:
                    continue
                matrix[int(from_station)][int(to_station)] = float(
                    self.travel_time_estimator(int(from_station), int(to_station))
                )

        return matrix
