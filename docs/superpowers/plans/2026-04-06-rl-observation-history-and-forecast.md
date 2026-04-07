# RL Observation History And Forecast Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an internal charging-history log and expose `future_demand` plus `travel_time_matrix` in the RL env observation without changing `SystemState` semantics.

**Architecture:** Keep simulator state and RL observation separate. `SimulatorCore` records every successful assignment into a dedicated `ChargingHistoryLog`, while `SplitChargingOrchestrator.build_observation()` reads `sim_state`, the history log, and the travel-time estimator to append derived RL-only features. Add a `DemandForecaster` class inside `simulator.orchestrator` to dispatch forecasting methods by string; the initial method is `exponential-decay`.

**Tech Stack:** Python 3.12, standard library `unittest`, dataclasses

---

## File Map

- Create: `docs/superpowers/plans/2026-04-06-rl-observation-history-and-forecast.md`
- Create: `simulator/history.py`
- Create: `tests/test_history.py`
- Create: `tests/test_simulator_history_integration.py`
- Create: `tests/test_orchestrator_observation.py`
- Modify: `simulator/models.py`
- Modify: `simulator/simulator.py`
- Modify: `simulator/orchestrator.py`
- Modify: `simulator/__init__.py`
- Modify: `simulator/demo_workflow.py`

## Forecast To Implement

Use a deterministic exponential-decay forecast:

- horizon = `15.0`
- method string = `exponential-decay`
- for each station, sum exponentially decayed contributions from historical arrivals
- expose that per-station floating-point vector as the prediction for the next 15 minutes

This is deterministic, testable, and extensible to future forecasting methods.

## Chunk 1: History Model And Log

### Task 1: Add failing tests for history-record storage

**Files:**
- Create: `tests/test_history.py`
- Modify: `simulator/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from simulator.history import ChargingHistoryLog
from simulator.models import ChargingHistoryRecord


class ChargingHistoryLogTests(unittest.TestCase):
    def test_records_are_appended_in_submission_order(self) -> None:
        history_log = ChargingHistoryLog()

        history_log.append(
            ChargingHistoryRecord(
                vehicle_id=1,
                station_id=0,
                charger_id=0,
                arrival_time=0.0,
                start_time=0.0,
                end_time=4.0,
                wait_time=0.0,
            )
        )
        history_log.append(
            ChargingHistoryRecord(
                vehicle_id=1,
                station_id=2,
                charger_id=0,
                arrival_time=10.0,
                start_time=10.0,
                end_time=16.0,
                wait_time=0.0,
            )
        )

        self.assertEqual(
            [(item.vehicle_id, item.station_id, item.arrival_time) for item in history_log.records()],
            [(1, 0, 0.0), (1, 2, 10.0)],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_history -v`
Expected: `ModuleNotFoundError` or `ImportError` for `simulator.history` / `ChargingHistoryRecord`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ChargingHistoryRecord:
    vehicle_id: int
    station_id: int
    charger_id: int
    arrival_time: float
    start_time: float
    end_time: float
    wait_time: float


class ChargingHistoryLog:
    def __init__(self) -> None:
        self._records: list[ChargingHistoryRecord] = []

    def append(self, record: ChargingHistoryRecord) -> None:
        self._records.append(record)

    def records(self) -> list[ChargingHistoryRecord]:
        return list(self._records)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_history -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/test_history.py simulator/history.py simulator/models.py simulator/__init__.py
git commit -m "test: add charging history log model"
```

## Chunk 2: Simulator Integration

### Task 2: Add failing tests for history accumulation without changing `SystemState`

**Files:**
- Create: `tests/test_simulator_history_integration.py`
- Modify: `simulator/simulator.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from simulator import ChargingRequest, SimulatorCore, StationSpec


class SimulatorHistoryIntegrationTests(unittest.TestCase):
    def test_history_log_keeps_all_requests_while_vehicles_keeps_latest_snapshot(self) -> None:
        simulator = SimulatorCore(
            station_specs=[
                StationSpec(station_id=0, charge_capacity=1),
                StationSpec(station_id=1, charge_capacity=1),
            ]
        )

        simulator.submit_arrival(
            ChargingRequest(vehicle_id=1, station_id=0, charge_duration=5.0, arrival_time=0.0)
        )
        simulator.submit_arrival(
            ChargingRequest(vehicle_id=1, station_id=1, charge_duration=4.0, arrival_time=6.0)
        )

        state = simulator.get_state(query_time=6.0, vehicle_info=True)

        self.assertEqual(len(simulator.history_log.records()), 2)
        self.assertEqual(state["vehicles"][1]["station_id"], 1)
        self.assertNotIn("history_log", state)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_simulator_history_integration -v`
Expected: failure because `history_log` is missing from `SimulatorCore`

- [ ] **Step 3: Write minimal implementation**

```python
from simulator.history import ChargingHistoryLog
from simulator.models import ChargingHistoryRecord


class SimulatorCore:
    def __init__(self, station_specs):
        ...
        self.history_log = ChargingHistoryLog()

    def submit_arrival(self, request):
        ...
        assignment = ChargingAssignment(...)
        self.history_log.append(
            ChargingHistoryRecord(
                vehicle_id=assignment.vehicle_id,
                station_id=assignment.station_id,
                charger_id=assignment.charger_id,
                arrival_time=assignment.arrival_time,
                start_time=assignment.start_time,
                end_time=assignment.end_time,
                wait_time=assignment.wait_time,
            )
        )
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_simulator_history_integration -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/test_simulator_history_integration.py simulator/simulator.py
git commit -m "feat: track charging assignment history"
```

## Chunk 3: RL Observation Enrichment

### Task 3: Add failing tests for `travel_time_matrix`

**Files:**
- Create: `tests/test_orchestrator_observation.py`
- Modify: `simulator/orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from simulator import DecisionVehicle, SimulatorCore, SplitChargingOrchestrator, StationSpec


class OrchestratorObservationTests(unittest.TestCase):
    def test_build_observation_exposes_full_travel_time_matrix(self) -> None:
        orchestrator = SplitChargingOrchestrator(
            simulator=SimulatorCore(
                station_specs=[
                    StationSpec(station_id=0, charge_capacity=1),
                    StationSpec(station_id=1, charge_capacity=1),
                    StationSpec(station_id=2, charge_capacity=1),
                ]
            ),
            travel_time_estimator=lambda from_station, to_station: {
                (0, 1): 3.0,
                (1, 0): 4.0,
                (0, 2): 5.0,
                (2, 0): 6.0,
                (1, 2): 7.0,
                (2, 1): 8.0,
            }.get((from_station, to_station), 0.0),
        )

        observation = orchestrator.build_observation(
            current_ev=DecisionVehicle(
                vehicle_id=1,
                station_id=0,
                arrival_time=0.0,
                total_charge_demand=4.0,
                downstream_stations=(1, 2),
            ),
            now=0.0,
        )

        self.assertEqual(
            observation["travel_time_matrix"],
            [
                [0.0, 3.0, 5.0],
                [4.0, 0.0, 7.0],
                [6.0, 8.0, 0.0],
            ],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_orchestrator_observation -v`
Expected: `KeyError` for `travel_time_matrix`

- [ ] **Step 3: Write minimal implementation**

```python
class SplitChargingOrchestrator:
    def _build_travel_time_matrix(self) -> list[list[float]]:
        station_ids = self.simulator.station_ids
        return [
            [
                0.0 if from_station == to_station else float(self.travel_time_estimator(from_station, to_station))
                for to_station in station_ids
            ]
            for from_station in station_ids
        ]
```

Add the result to `build_observation()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_orchestrator_observation.OrchestratorObservationTests.test_build_observation_exposes_full_travel_time_matrix -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/test_orchestrator_observation.py simulator/orchestrator.py
git commit -m "feat: expose travel time matrix in rl observation"
```

### Task 4: Add failing tests for `future_demand`

**Files:**
- Modify: `tests/test_orchestrator_observation.py`
- Modify: `simulator/orchestrator.py`
- Modify: `simulator/demo_workflow.py`

- [ ] **Step 1: Write the failing test**

```python
from simulator import ChargingRequest

    def test_build_observation_exposes_future_demand_from_recent_history(self) -> None:
        orchestrator = SplitChargingOrchestrator(
            simulator=SimulatorCore(
                station_specs=[
                    StationSpec(station_id=0, charge_capacity=1),
                    StationSpec(station_id=1, charge_capacity=1),
                    StationSpec(station_id=2, charge_capacity=1),
                ]
            )
        )

        orchestrator.simulator.submit_arrival(
            ChargingRequest(vehicle_id=1, station_id=0, charge_duration=4.0, arrival_time=1.0)
        )
        orchestrator.simulator.submit_arrival(
            ChargingRequest(vehicle_id=2, station_id=2, charge_duration=4.0, arrival_time=6.0)
        )
        orchestrator.simulator.submit_arrival(
            ChargingRequest(vehicle_id=3, station_id=2, charge_duration=4.0, arrival_time=14.0)
        )

        observation = orchestrator.build_observation(
            current_ev=DecisionVehicle(
                vehicle_id=99,
                station_id=1,
                arrival_time=14.0,
                total_charge_demand=4.0,
                downstream_stations=(2,),
            ),
            now=16.0,
        )

        self.assertEqual(observation["future_demand"], [1, 0, 2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_orchestrator_observation.OrchestratorObservationTests.test_build_observation_exposes_future_demand_from_recent_history -v`
Expected: `KeyError` for `future_demand`

- [ ] **Step 3: Write minimal implementation**

```python
class SplitChargingOrchestrator:
    def _predict_future_demand(self, now: float, horizon: float = 15.0) -> list[int]:
        counts = [0 for _ in range(1 + max(self.simulator.station_ids))]
        window_start = float(now) - float(horizon)
        for record in self.simulator.history_log.records():
            if float(record.arrival_time) < window_start or float(record.arrival_time) > float(now):
                continue
            counts[int(record.station_id)] += 1
        return counts
```

Add the result to `build_observation()`, and update `demo_workflow.py` to print the new observation fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_orchestrator_observation.OrchestratorObservationTests.test_build_observation_exposes_future_demand_from_recent_history -v`
Expected: `OK`

- [ ] **Step 5: Run the focused test suite**

Run: `python -m unittest tests.test_history tests.test_simulator_history_integration tests.test_orchestrator_observation -v`
Expected: all tests `OK`

- [ ] **Step 6: Commit**

```bash
git add tests/test_orchestrator_observation.py simulator/orchestrator.py simulator/demo_workflow.py
git commit -m "feat: add rl observation demand forecast"
```

## Chunk 4: Final Verification

### Task 5: Run end-to-end verification and sanity-check the demo output

**Files:**
- Modify: none, unless verification reveals a defect

- [ ] **Step 1: Run the full test suite**

Run: `python -m unittest discover -s tests -v`
Expected: all tests `OK`

- [ ] **Step 2: Run the demo**

Run: `python -m simulator.demo_workflow`
Expected: JSON output includes:
- `sim_state`
- `commitment_features`
- `current_ev`
- `future_demand`
- `travel_time_matrix`

- [ ] **Step 3: Review exports**

Check that `simulator/__init__.py` exports any new public types that callers should import directly.

- [ ] **Step 4: Commit**

```bash
git add simulator/__init__.py
git commit -m "chore: finalize rl observation history integration"
```
