from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import (
    ChargingRequest,
    ChargingDecision,
    DecisionVehicle,
    SimulatorCore,
    SplitChargingOrchestrator,
    StationSpec,
)


def _to_payload(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_payload(item) for item in value]
    return value


def run_observation_format_demo() -> dict[str, Any]:
    simulator = SimulatorCore(
        station_specs=[
            StationSpec(station_id=0, charge_capacity=1),
            StationSpec(station_id=1, charge_capacity=2),
            StationSpec(station_id=2, charge_capacity=1),
        ]
    )
    orchestrator = SplitChargingOrchestrator(
        simulator=simulator,
        travel_time_estimator=lambda from_station, to_station: {
            (0, 1): 3.0,
            (0, 2): 5.0,
            (1, 0): 4.0,
            (1, 2): 2.0,
            (2, 0): 6.0,
            (2, 1): 2.5,
        }.get((from_station, to_station), 0.0),
    )

    current_ev = DecisionVehicle(
        vehicle_id=301,
        station_id=0,
        arrival_time=12.0,
        total_charge_demand=9.0,
        downstream_stations=(1, 2),
    )
    decision = ChargingDecision(
        first_charge_duration=4.0,
        second_station_id=2,
        second_charge_duration=5.0,
    )

    # Seed a small amount of recent history so future_demand is visible in the output.
    orchestrator.simulator.submit_arrival(
        request=ChargingRequest(
            vehicle_id=201,
            station_id=0,
            charge_duration=4.0,
            arrival_time=2.0,
        )
    )
    orchestrator.simulator.submit_arrival(
        request=ChargingRequest(
            vehicle_id=202,
            station_id=2,
            charge_duration=3.0,
            arrival_time=7.0,
        )
    )

    observation_now = 12.0
    observation = orchestrator.build_observation(
        current_ev=current_ev,
        now=observation_now,
        vehicle_info=True,
    )
    decision_result = orchestrator.apply_decision(
        current_ev=current_ev,
        decision=decision,
    )

    return {
        "observation_input": {
            "current_ev": _to_payload(current_ev),
            "now": float(observation_now),
            "vehicle_info": True,
        },
        "observation_output": _to_payload(observation),
        "decision_input": {
            "current_ev": _to_payload(current_ev),
            "decision": _to_payload(decision),
        },
        "decision_output": _to_payload(decision_result),
    }


def main() -> None:
    print(json.dumps(run_observation_format_demo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
