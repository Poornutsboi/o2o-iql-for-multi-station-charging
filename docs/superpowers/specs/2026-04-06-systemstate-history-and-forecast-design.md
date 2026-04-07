# RL Observation History And Forecast Design

**Date:** 2026-04-06

**Status:** Approved in chat, documented for implementation planning

## Goal

Extend the RL environment observation surface so callers can inspect both:

- current simulator state snapshots, as they work today
- derived forecast features for downstream decision-making

The new information should support future-demand estimation without changing the meaning of the existing `SystemState` structure and without polluting simulator state with raw request history.

## Current Behavior

### `vehicles`

`SimulatorCore` currently keeps one latest `VehicleRecord` per `vehicle_id` in `_latest_record_by_vehicle`.

`get_state(query_time, vehicle_info=True)` serializes that mapping into `SystemState["vehicles"]`.

Implications:

- `vehicles` is keyed by `vehicle_id`
- each vehicle contributes at most one record
- a later request from the same vehicle replaces the previous one
- completed assignments remain visible as `status="complete"`
- this field is a current-state snapshot, not a request history log

### `commitment_features`

`SplitChargingOrchestrator.build_observation()` currently returns:

- `sim_state`
- `commitment_features`
- `current_ev`

`commitment_features` currently includes:

- `commitment_count`
- `commitment_charge_demand`
- `earliest_expected_arrival_eta`

This design keeps the commitment model unchanged.

## Approved Design

### 1. Keep `SystemState` and `vehicles` unchanged

`SystemState["vehicles"]` remains the latest per-vehicle snapshot.

We do not change `SystemState` shape or semantics in this work.

### 2. Add a dedicated history-log class

Introduce a dedicated history-log class that records every submitted charging assignment in append-only order.

Each history item contains exactly these fields:

- `vehicle_id`
- `station_id`
- `charger_id`
- `arrival_time`
- `start_time`
- `end_time`
- `wait_time`

This log is sourced from completed assignment creation inside `SimulatorCore.submit_arrival()`.

It represents assignment history since simulator startup.

This log is intentionally not part of `SystemState`. It should be maintained as a separate object so raw history does not change the meaning or payload shape of simulator state snapshots.

### 3. Add estimated travel-time matrix to RL env state

The RL env state returned by `SplitChargingOrchestrator.build_observation()` should expose a complete station-to-station estimated travel-time matrix.

Shape:

`travel_time_matrix[from_station_id][to_station_id]`

Rules:

- matrix covers all simulator station ids
- diagonal entries are `0.0`
- off-diagonal entries come from `travel_time_estimator(from_station_id, to_station_id)`

### 4. Add `future_demand`

The RL env state returned by `SplitChargingOrchestrator.build_observation()` should expose a `future_demand` prediction vector for the next 15 minutes.

This feature is intentionally predictive. It should not be derived from `commitment`.

Instead, it should be estimated from the dedicated history-log object, while preserving the current `vehicles` snapshot behavior.

Prediction logic should live in a dedicated `DemandForecaster` class inside `simulator.orchestrator`.

`DemandForecaster` should dispatch by prediction method string. The first supported method is:

- `exponential-decay`

This method should produce one floating-point demand estimate per station for the next 15 minutes by applying an exponential decay kernel to historical arrivals.

The design should allow additional methods to be added later without changing the observation field name.

The forecasting implementation should:

- uses simulator-observable history/state only
- produces one predicted value per station
- is deterministic and testable
- can be refined later without changing the output field name

The prediction logic lives at the orchestrator layer. The orchestrator can read the history-log object and expose the resulting `future_demand` vector through RL observation output.

### 5. Output placement

The current request clarified that "state" means RL env state, not `SystemState`.

To match that requirement while keeping raw logs out of simulator state:

- `SystemState` should remain focused on simulator state
- RL env state should expose `future_demand`
- RL env state should expose `travel_time_matrix`
- raw `history_log` should remain outside `SystemState`

Because `SplitChargingOrchestrator.build_observation()` already returns the RL env observation, the cleanest path is:

- keep `sim_state` as the serialized `SystemState`
- let the orchestrator assemble `future_demand` and `travel_time_matrix` before returning the observation
- expose those derived fields at the observation layer, not inside `SystemState`
- keep `history_log` internal to simulator/orchestrator infrastructure

## Data Model Changes

### `simulator.models`

Add a new dataclass for history records, for example:

- `ChargingHistoryRecord`

Do not add `history_log`, `future_demand`, or `travel_time_matrix` to `SystemState`.

### Dedicated history-log module

Add a dedicated class, for example:

- `ChargingHistoryLog`

Responsibilities:

- store assignment records in append-only order
- provide read access for demand-prediction logic
- stay separate from `SystemState`

### `simulator.simulator`

Attach and update the dedicated history-log object from successful `submit_arrival()` calls.

`get_state()` should continue to serialize station, metric, and vehicle snapshot information, but it should not serialize raw history-log contents or RL-only forecast features.

### `simulator.orchestrator`

Add a `DemandForecaster` class responsible for future-demand prediction methods.

Use the history-log object plus current simulator state to build:

- `future_demand`
- `travel_time_matrix`

Expose those values through returned RL env observation output.

## Testing Requirements

Implementation must follow TDD.

Required test coverage:

1. A test that confirms the dedicated history-log class records every submitted request, including repeated requests from the same vehicle.
2. A test that confirms `vehicles` keeps latest-per-vehicle semantics even while the history log accumulates all requests.
3. A test that confirms the RL env observation exposes a full travel-time matrix with `0.0` on the diagonal and estimator-driven values off the diagonal.
4. A test that confirms the RL env observation exposes a `future_demand` vector and that it follows the chosen heuristic deterministically.

## Risks And Constraints

- The repository has no commits yet and has unrelated staged/deleted paths under `simulator_core/`; implementation should avoid disturbing those files.
- `get_state(query_time)` forbids querying earlier than the current simulator clock, so demand prediction should not rely on retroactive state reconstruction unless explicitly added.
- `future_demand` semantics must be documented in code comments or tests because it is predictive rather than directly observed.
- The distinction between simulator `SystemState` and RL env state must remain explicit in naming and tests.

## Out Of Scope

- Changing `Commitment` or `CommitmentStore`
- Building a full vehicle trajectory engine
- Replacing `vehicles` with a history structure
- Exposing raw `history_log` inside `SystemState`
- Exposing `future_demand` or `travel_time_matrix` inside `SystemState`
- Adding stochastic forecasting or model training
