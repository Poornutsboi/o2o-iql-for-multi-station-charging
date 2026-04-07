from __future__ import annotations

import heapq

from simulator.models import StationRuntimeSnapshot, StationSpec, StationState


class StationRuntime:
    def __init__(self, spec: StationSpec) -> None:
        if int(spec.charge_capacity) <= 0:
            raise ValueError("charge_capacity must be > 0.")
        self.spec = spec
        self._release_times = [0.0 for _ in range(int(spec.charge_capacity))]
        self._heap = [
            (0.0, charger_id) for charger_id in range(int(spec.charge_capacity))
        ]
        heapq.heapify(self._heap)
        self._anonymous_queue_records: list[dict[str, float]] = []

    def bootstrap(
        self,
        charger_status: list[float],
        queue_waiting_time: list[float],
        queue_demand: list[float],
    ) -> None:
        self._anonymous_queue_records = []
        self._release_times = [float(value) for value in charger_status]
        self._heap = [
            (float(release_time), charger_id)
            for charger_id, release_time in enumerate(self._release_times)
        ]
        heapq.heapify(self._heap)

        for waiting_time, charge_duration in zip(queue_waiting_time, queue_demand):
            arrival_time = 0.0 - float(waiting_time)
            _charger_id, start_time, end_time, _wait_time = self.reserve(
                arrival_time=arrival_time,
                charge_duration=float(charge_duration),
            )
            self._anonymous_queue_records.append(
                {
                    "arrival_time": float(arrival_time),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "charge_duration": float(charge_duration),
                }
            )

    def reserve(self, arrival_time: float, charge_duration: float) -> tuple[int, float, float, float]:
        next_free_time, charger_id = heapq.heappop(self._heap)
        start_time = max(float(arrival_time), float(next_free_time))
        end_time = float(start_time + float(charge_duration))
        wait_time = float(start_time - float(arrival_time))
        self._release_times[int(charger_id)] = float(end_time)
        heapq.heappush(self._heap, (float(end_time), int(charger_id)))
        return int(charger_id), float(start_time), float(end_time), float(wait_time)

    def anonymous_queue_state(self, query_time: float) -> tuple[list[float], list[float]]:
        waiting_time: list[float] = []
        demand: list[float] = []

        for record in self._anonymous_queue_records:
            if float(query_time) >= float(record["start_time"]):
                continue
            waiting_time.append(max(0.0, float(query_time) - float(record["arrival_time"])))
            demand.append(float(record["charge_duration"]))

        return waiting_time, demand

    def to_state(
        self,
        query_time: float,
        queue_waiting_time: list[float],
        queue_demand: list[float] | None = None,
    ) -> StationState:
        charger_status = [
            max(0.0, float(release_time) - float(query_time))
            for release_time in self._release_times
        ]
        anonymous_waiting_time, anonymous_demand = self.anonymous_queue_state(query_time)
        return StationState(
            station_id=int(self.spec.station_id),
            charge_capacity=int(self.spec.charge_capacity),
            charger_status=charger_status,
            available_info=[bool(status == 0.0) for status in charger_status],
            queue_waiting_time=anonymous_waiting_time + list(queue_waiting_time),
            queue_demand=anonymous_demand + list(queue_demand or []),
        )

    def snapshot(self) -> StationRuntimeSnapshot:
        return StationRuntimeSnapshot(
            station_id=int(self.spec.station_id),
            release_times=list(self._release_times),
        )

    def restore(self, snapshot: StationRuntimeSnapshot) -> None:
        self._release_times = list(snapshot.release_times)
        self._heap = [
            (float(release_time), charger_id)
            for charger_id, release_time in enumerate(self._release_times)
        ]
        heapq.heapify(self._heap)
