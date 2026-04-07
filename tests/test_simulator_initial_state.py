import unittest

from simulator import ChargingRequest, SimulatorCore, StationSpec


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

    def test_initial_queue_items_appear_in_station_snapshot_but_not_vehicle_views_or_metrics(
        self,
    ) -> None:
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

    def test_initial_state_rejects_invalid_charger_status_length(self) -> None:
        with self.assertRaises(ValueError):
            SimulatorCore(
                station_specs=[StationSpec(station_id=0, charge_capacity=1)],
                initial_state={
                    "stations": {
                        0: {
                            "charger_status": [0.0, 1.0],
                            "queue_waiting_time": [],
                            "queue_demand": [],
                        }
                    }
                },
            )
