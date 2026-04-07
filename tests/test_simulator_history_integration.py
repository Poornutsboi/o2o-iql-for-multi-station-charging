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
