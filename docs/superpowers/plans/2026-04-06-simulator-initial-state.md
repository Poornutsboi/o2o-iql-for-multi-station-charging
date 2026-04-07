# Simulator Initial State Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SimulatorCore(initial_state=...)` support so the simulator can start from non-empty charger occupancy and anonymous queue load, while upgrading `StationState` to expose `queue_waiting_time` and `queue_demand`.

**Architecture:** Keep `StationState` as the public station snapshot shape, but make its queue fields explicit by renaming `queue` to `queue_waiting_time` and adding `queue_demand`. Put anonymous startup-load execution semantics inside `StationRuntime`, then let `SimulatorCore` validate and apply per-station initial state without polluting vehicle-level state, history, or metrics.

**Tech Stack:** Python 3.12, standard library `unittest`, dataclasses

---

## File Map

- Create: `docs/superpowers/plans/2026-04-06-simulator-initial-state.md`
- Create: `tests/test_simulator_initial_state.py`
- Modify: `simulator/models.py`
- Modify: `simulator/station.py`
- Modify: `simulator/simulator.py`
- Modify: `simulator/__init__.py`
- Modify: `simulator/demo_workflow.py`
- Modify: `simulator/demo_observation_format.py`
- Modify: `tests/test_demo_observation_format.py`

## Chunk 1: Public Station-State Schema

### Task 1: Rename the public queue field and add queue demand

**Files:**
- Modify: `simulator/models.py`
- Modify: `simulator/station.py`
- Modify: `simulator/__init__.py`

- [ ] **Step 1: Write the failing test**

Add a new test to `tests/test_simulator_initial_state.py`:

```python
import unittest

from simulator import SimulatorCore, StationSpec


class SimulatorInitialStateTests(unittest.TestCase):
    def test_empty_station_snapshot_uses_queue_waiting_time_and_queue_demand_fields(self) -> None:
        simulator = SimulatorCore(
            station_specs=[
                StationSpec(station_id=0, charge_capacity=2),
            ]
        )

        state = simulator.get_state(query_time=0.0)

        self.assertEqual(
            state["stations"][0],
            {
                "station_id": 0,
                "charge_capacity": 2,
                "charger_status": [0.0, 0.0],
                "available_info": [True, True],
                "queue_waiting_time": [],
                "queue_demand": [],
            },
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_simulator_initial_state.SimulatorInitialStateTests.test_empty_station_snapshot_uses_queue_waiting_time_and_queue_demand_fields -v`
Expected: failure because `queue` is still present and the new keys are missing

- [ ] **Step 3: Write minimal implementation**

Update the public station snapshot model and serializer:

```python
@dataclass(frozen=True)
class StationState:
    station_id: int
    charge_capacity: int
    charger_status: list[float]
    available_info: list[bool]
    queue_waiting_time: list[float]
    queue_demand: list[float]
```

Update `StationRuntime.to_state(...)` to populate the renamed field and the new `queue_demand` field.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_simulator_initial_state.SimulatorInitialStateTests.test_empty_station_snapshot_uses_queue_waiting_time_and_queue_demand_fields -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/test_simulator_initial_state.py simulator/models.py simulator/station.py simulator/__init__.py
git commit -m "feat: expose queue waiting time and demand in station state"
```

## Chunk 2: Anonymous Initial Load Bootstrap

### Task 2: Add failing tests for charger occupancy restored from `initial_state`

**Files:**
- Modify: `tests/test_simulator_initial_state.py`
- Modify: `simulator/simulator.py`
- Modify: `simulator/station.py`

- [ ] **Step 1: Write the failing test**

Append this test:

```python
    def test_initial_charger_status_delays_first_real_arrival(self) -> None:
        simulator = SimulatorCore(
            station_specs=[StationSpec(station_id=0, charge_capacity=1)],
            initial_state={
                "stations": {
                    0: {
                        "charger_status": [5.0],
                        "queue_waiting_time": [],
                        "queue_demand": [],
                    }
                }
            },
        )

        assignment = simulator.submit_arrival(
            ChargingRequest(
                vehicle_id=1,
                station_id=0,
                charge_duration=4.0,
                arrival_time=0.0,
            )
        )

        self.assertEqual(assignment.start_time, 5.0)
        self.assertEqual(assignment.wait_time, 5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_simulator_initial_state.SimulatorInitialStateTests.test_initial_charger_status_delays_first_real_arrival -v`
Expected: failure because `SimulatorCore.__init__()` does not yet accept `initial_state`

- [ ] **Step 3: Write minimal implementation**

Extend `SimulatorCore.__init__()` to accept `initial_state`, validate the payload, and call a new station bootstrap helper:

```python
class SimulatorCore:
    def __init__(self, station_specs, initial_state=None) -> None:
        ...
        self._apply_initial_state(initial_state or {"stations": {}})
```

Add a `StationRuntime.bootstrap(charger_status, queue_waiting_time, queue_demand)` method that restores charger release times from `charger_status`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_simulator_initial_state.SimulatorInitialStateTests.test_initial_charger_status_delays_first_real_arrival -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/test_simulator_initial_state.py simulator/simulator.py simulator/station.py
git commit -m "feat: restore charger occupancy from simulator initial state"
```

### Task 3: Add failing tests for anonymous initial queue demand affecting future scheduling

**Files:**
- Modify: `tests/test_simulator_initial_state.py`
- Modify: `simulator/station.py`
- Modify: `simulator/simulator.py`

- [ ] **Step 1: Write the failing test**

Append these tests:

```python
    def test_initial_queue_demand_delays_later_real_arrivals_in_fcfs_order(self) -> None:
        simulator = SimulatorCore(
            station_specs=[StationSpec(station_id=0, charge_capacity=1)],
            initial_state={
                "stations": {
                    0: {
                        "charger_status": [5.0],
                        "queue_waiting_time": [0.0, 0.0],
                        "queue_demand": [6.0, 4.0],
                    }
                }
            },
        )

        assignment = simulator.submit_arrival(
            ChargingRequest(
                vehicle_id=1,
                station_id=0,
                charge_duration=3.0,
                arrival_time=1.0,
            )
        )

        self.assertEqual(assignment.start_time, 15.0)
        self.assertEqual(assignment.wait_time, 14.0)

    def test_initial_queue_items_appear_in_station_snapshot_but_not_vehicle_views_or_metrics(self) -> None:
        simulator = SimulatorCore(
            station_specs=[StationSpec(station_id=0, charge_capacity=1)],
            initial_state={
                "stations": {
                    0: {
                        "charger_status": [5.0],
                        "queue_waiting_time": [2.0, 0.0],
                        "queue_demand": [6.0, 4.0],
                    }
                }
            },
        )

        state = simulator.get_state(query_time=0.0, vehicle_info=True)

        self.assertEqual(state["stations"][0]["queue_waiting_time"], [2.0, 0.0])
        self.assertEqual(state["stations"][0]["queue_demand"], [6.0, 4.0])
        self.assertEqual(state["vehicles"], {})
        self.assertEqual(simulator.history_log.records(), [])
        self.assertEqual(simulator.get_metrics(query_time=0.0).ev_served[0], 0)
        self.assertEqual(simulator.get_metrics(query_time=0.0).queue_time[0], 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_simulator_initial_state.SimulatorInitialStateTests.test_initial_queue_demand_delays_later_real_arrivals_in_fcfs_order tests.test_simulator_initial_state.SimulatorInitialStateTests.test_initial_queue_items_appear_in_station_snapshot_but_not_vehicle_views_or_metrics -v`
Expected: failing assertions because anonymous queue demand is not yet compiled into station scheduling / station snapshots

- [ ] **Step 3: Write minimal implementation**

Inside `StationRuntime.bootstrap(...)`:

```python
for waiting_time, charge_duration in zip(queue_waiting_time, queue_demand):
    arrival_time = 0.0 - float(waiting_time)
    charger_id, start_time, end_time, wait_time = self.reserve(
        arrival_time=arrival_time,
        charge_duration=float(charge_duration),
    )
    self._anonymous_queue_records.append(
        {
            "arrival_time": arrival_time,
            "start_time": start_time,
            "end_time": end_time,
            "wait_time": wait_time,
            "charge_duration": float(charge_duration),
        }
    )
```

Then teach `to_state(query_time, ...)` to merge still-waiting anonymous queue items into the returned `queue_waiting_time` / `queue_demand` lists before adding real queue items.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_simulator_initial_state.SimulatorInitialStateTests.test_initial_queue_demand_delays_later_real_arrivals_in_fcfs_order tests.test_simulator_initial_state.SimulatorInitialStateTests.test_initial_queue_items_appear_in_station_snapshot_but_not_vehicle_views_or_metrics -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/test_simulator_initial_state.py simulator/station.py simulator/simulator.py
git commit -m "feat: schedule anonymous initial queue demand"
```

### Task 4: Add failing tests for malformed `initial_state`

**Files:**
- Modify: `tests/test_simulator_initial_state.py`
- Modify: `simulator/simulator.py`

- [ ] **Step 1: Write the failing test**

Append this test:

```python
    def test_initial_state_rejects_invalid_queue_lengths(self) -> None:
        with self.assertRaises(ValueError):
            SimulatorCore(
                station_specs=[StationSpec(station_id=0, charge_capacity=1)],
                initial_state={
                    "stations": {
                        0: {
                            "charger_status": [0.0],
                            "queue_waiting_time": [0.0],
                            "queue_demand": [],
                        }
                    }
                },
            )
```

Also add one more validation test for `charger_status` length mismatch or negative values if coverage is still thin.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_simulator_initial_state.SimulatorInitialStateTests.test_initial_state_rejects_invalid_queue_lengths -v`
Expected: failure because invalid initial-state payload is accepted

- [ ] **Step 3: Write minimal implementation**

Add `_apply_initial_state(...)` / `_validate_station_initial_state(...)` helpers in `SimulatorCore` that enforce:

- station id exists
- `len(charger_status) == charge_capacity`
- `len(queue_waiting_time) == len(queue_demand)`
- all values are `>= 0`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_simulator_initial_state.SimulatorInitialStateTests.test_initial_state_rejects_invalid_queue_lengths -v`
Expected: `OK`

- [ ] **Step 5: Run the focused initial-state suite**

Run: `python -m unittest tests.test_simulator_initial_state -v`
Expected: all tests `OK`

- [ ] **Step 6: Commit**

```bash
git add tests/test_simulator_initial_state.py simulator/simulator.py
git commit -m "feat: validate simulator initial state payloads"
```

## Chunk 3: Compatibility Updates And Final Verification

### Task 5: Update demos and compatibility tests to the new station-state shape

**Files:**
- Modify: `simulator/demo_workflow.py`
- Modify: `simulator/demo_observation_format.py`
- Modify: `tests/test_demo_observation_format.py`

- [ ] **Step 1: Write the failing test**

Add assertions that the observation payload uses the new station fields:

```python
        station_payload = payload["observation_output"]["sim_state"]["stations"][0]
        self.assertIn("queue_waiting_time", station_payload)
        self.assertIn("queue_demand", station_payload)
        self.assertNotIn("queue", station_payload)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_demo_observation_format -v`
Expected: failure because the demo output still exposes the old `queue` field

- [ ] **Step 3: Write minimal implementation**

Update any demo serialization assumptions and helper payloads so they reflect the renamed station fields without changing unrelated observation behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_demo_observation_format -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add simulator/demo_workflow.py simulator/demo_observation_format.py tests/test_demo_observation_format.py
git commit -m "test: update demos for station queue schema"
```

### Task 6: Run full verification and inspect for regression risk

**Files:**
- Modify: none, unless verification reveals a defect

- [ ] **Step 1: Run the focused compatibility suites**

Run: `python -m unittest tests.test_simulator_initial_state tests.test_simulator_history_integration tests.test_orchestrator_observation tests.test_demo_observation_format -v`
Expected: all tests `OK`

- [ ] **Step 2: Run the full test suite**

Run: `python -m unittest discover -s tests -v`
Expected: all tests `OK`

- [ ] **Step 3: Run the demos**

Run: `python -m simulator.demo_workflow`
Expected: JSON output with station snapshots using `queue_waiting_time` and `queue_demand`

Run: `python -m simulator.demo_observation_format`
Expected: JSON output with the same station-field names inside `sim_state`

- [ ] **Step 4: Review public exports**

Check `simulator/__init__.py` and confirm that any newly public types or helpers are exported only if callers need them.

- [ ] **Step 5: Commit**

```bash
git add simulator/__init__.py
git commit -m "chore: finalize simulator initial state support"
```
