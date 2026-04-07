# Simulator Initial State Design

**Date:** 2026-04-06

**Status:** Approved in chat, documented for review

## Goal

Allow `SimulatorCore` to start from a non-empty station snapshot so the simulator can model a realistic system state at `t=0`.

The initial state must support both:

- chargers that are already occupied when the simulation starts
- anonymous queue items that are already waiting when the simulation starts

Those anonymous initial vehicles are not part of the research target and must not be promoted into vehicle-level simulator entities.

## Current Behavior

`SimulatorCore` currently assumes every charger is free at initialization time.

`StationRuntime` starts with:

- `_release_times = [0.0] * charge_capacity`
- a heap of `(0.0, charger_id)` entries

As a result:

- only post-start `submit_arrival()` calls can create charging load
- `get_state(query_time=0.0)` always reflects an empty system unless real arrivals have already been submitted
- there is no way to express a queue that already exists at simulator startup

## Approved Design

### 1. `initial_state` is station-scoped and `StationState`-shaped

`SimulatorCore` should accept an optional `initial_state` argument.

The approved shape is:

```python
initial_state = {
    "stations": {
        0: {
            "charger_status": [8.0, 3.0],
            "queue_waiting_time": [0.0, 0.0],
            "queue_demand": [6.0, 4.0],
        },
        1: {
            "charger_status": [0.0],
            "queue_waiting_time": [],
            "queue_demand": [],
        },
    }
}
```

This format intentionally mirrors `SystemState["stations"]` as closely as possible.

Per-station fields mean:

- `charger_status`: remaining occupied time for each charger at `t=0`
- `queue_waiting_time`: elapsed waiting time for each anonymous queue item at `t=0`
- `queue_demand`: charging duration each anonymous queue item will require once it reaches a charger

This design updates `StationState` itself so the same station payload can be used both for observations and for simulator initialization.

### 2. Anonymous initial queue items must remain anonymous

Initial queue items and initial charging load affect future station availability, but they must not create simulator-managed vehicle entities.

Therefore they must not be added to:

- `SimulatorCore._latest_record_by_vehicle`
- `SimulatorCore.history_log`
- `SystemState["vehicles"]`
- metrics such as `ev_served` or accumulated historical `queue_time`

These anonymous initial items only exist to shape station runtime behavior and station-level snapshots.

### 3. `StationState` remains an observation view

`StationState` should keep its role as the read model returned by `get_state()`, but its queue fields should be made more explicit.

The approved field change is:

- rename `queue` to `queue_waiting_time`
- add `queue_demand`

Their meanings are:

- `queue_waiting_time[i]`: how long anonymous queue item `i` has already waited
- `queue_demand[i]`: how long anonymous queue item `i` will occupy a charger once service begins

As a result:

- `initial_state` can directly reuse station-state structure
- runtime state queries expose richer queue information without a separate initialization-only schema
- queue display data and queue execution data stay aligned by index

### 4. Station bootstrap logic lives in `StationRuntime`

The cleanest place to apply anonymous initial load is `StationRuntime`.

Add a bootstrap/restore-style API that:

1. restores the charger heap from `charger_status`
2. converts each anonymous queued item into an internal reservation in FCFS order
3. preserves enough station-local information for future `reserve()` calls and `to_state()` queries

This keeps execution semantics close to the station scheduling logic rather than spreading them across `SimulatorCore`.

### 5. Queue waiting and queue demand are both first-class station state

The discussion clarified an important distinction:

- waiting-time information is useful for describing the queue at `t=0`
- demand information is required to simulate future resource occupancy from that queue

To achieve realistic post-start queue evolution, the simulator must also know how long each anonymous queued item will occupy a charger after reaching service.

That is why `queue_demand` becomes part of `StationState` itself.

## Runtime Semantics

At simulator initialization:

1. each station starts from its declared `charger_status`
2. each station processes anonymous queued items in the order given by `queue_waiting_time` and `queue_demand`
3. later real `submit_arrival()` calls are scheduled behind any already-bootstrapped anonymous workload

At `get_state(query_time)`:

- station snapshots should reflect remaining charger occupancy as usual
- station snapshots should also reflect anonymous queue items that are still waiting at `query_time`
- `vehicle_info=True` still reports only real simulator vehicles created by `submit_arrival()`

## Validation Rules

The simulator should reject malformed initial state input.

Required validation:

- every referenced station id must exist in `station_specs`
- `len(charger_status)` must equal the station's `charge_capacity`
- `len(queue_waiting_time)` must equal `len(queue_demand)`
- every time value must be `>= 0`
- omitted stations default to an empty initial state

## Implementation Notes

### `simulator.simulator`

- extend `SimulatorCore.__init__()` to accept `initial_state`
- validate and apply per-station bootstrap data after constructing `StationRuntime` objects

### `simulator.station`

- add station-local bootstrap support for anonymous initial charging and queue load
- keep anonymous initial workload internal to the station runtime

### `simulator.models`

- rename `StationState.queue` to `queue_waiting_time`
- add `StationState.queue_demand`
- keep time-like fields as `float`-typed for consistency with the rest of the simulator

Typed helpers are optional because the approved external contract is now the same station-shaped payload used in state output.

## Testing Requirements

Implementation must follow TDD.

Required coverage:

1. a simulator initialized with no `initial_state` keeps existing behavior unchanged
2. initial `charger_status` delays the first real arrival as expected
3. initial anonymous queue items delay later real arrivals according to FCFS order and `queue_demand`
4. anonymous initial items appear in station snapshots but do not appear in `vehicles`, `history_log`, or service metrics
5. serialized station snapshots expose `queue_waiting_time` and `queue_demand` instead of the old `queue`
6. malformed `initial_state` payloads are rejected with clear errors

## Out Of Scope

- adding vehicle ids for initial anonymous items
- exposing anonymous bootstrap items through `vehicles`
- recording anonymous bootstrap items in `history_log`
