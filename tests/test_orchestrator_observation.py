import math
import tempfile
import unittest
from pathlib import Path

from simulator import ChargingRequest, DecisionVehicle, SimulatorCore, SplitChargingOrchestrator, StationSpec
from simulator.orchestrator import (
    DemandForecaster,
    demand_record_to_vehicle,
    demand_records_to_vehicles,
    load_demand_vehicles_from_csv,
)


class OrchestratorObservationTests(unittest.TestCase):
    def test_demand_record_to_vehicle_maps_dataset_fields_to_env_vehicle(self) -> None:
        vehicle = demand_record_to_vehicle(
            {
                "Vehicle_ID": "7",
                "Route": "[2, 3, 4]",
                "Duration": "49",
                "Arrival_time": "12",
                "episode_id": "999",
            }
        )

        self.assertEqual(vehicle.vid, 7)
        self.assertEqual(vehicle.route, [2, 3, 4])
        self.assertEqual(vehicle.duration, 49.0)
        self.assertEqual(vehicle.arrival_time, 12.0)

    def test_demand_records_to_vehicles_preserves_dataset_order(self) -> None:
        vehicles = demand_records_to_vehicles(
            [
                {
                    "Vehicle_ID": "1",
                    "Route": "[0, 5]",
                    "Duration": "23",
                    "Arrival_time": "0",
                },
                {
                    "Vehicle_ID": "2",
                    "Route": "[1]",
                    "Duration": "34",
                    "Arrival_time": "1",
                },
            ]
        )

        self.assertEqual([vehicle.vid for vehicle in vehicles], [1, 2])
        self.assertEqual([vehicle.route for vehicle in vehicles], [[0, 5], [1]])

    def test_load_demand_vehicles_from_csv_reads_bc_dataset_shape(self) -> None:
        csv_content = "\n".join(
            [
                "Vehicle_ID,Route,Duration,Arrival_time,episode_id",
                '0,"[2, 3, 4]",49,0,12',
                '1,"[2, 3, 4, 6]",40,1,12',
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "episode.csv"
            csv_path.write_text(csv_content, encoding="utf-8")

            vehicles = load_demand_vehicles_from_csv(csv_path)

        self.assertEqual(len(vehicles), 2)
        self.assertEqual(vehicles[0].vid, 0)
        self.assertEqual(vehicles[0].route, [2, 3, 4])
        self.assertEqual(vehicles[0].duration, 49.0)
        self.assertEqual(vehicles[1].arrival_time, 1.0)

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

    def test_build_observation_exposes_exponential_decay_future_demand(self) -> None:
        orchestrator = SplitChargingOrchestrator(
            simulator=SimulatorCore(
                station_specs=[
                    StationSpec(station_id=0, charge_capacity=1),
                    StationSpec(station_id=1, charge_capacity=1),
                    StationSpec(station_id=2, charge_capacity=1),
                ]
            ),
            demand_prediction_method="exponential-decay",
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

        horizon = 15.0
        tau = 15.0
        kernel_multiplier = 1.0 - math.exp(-horizon / tau)

        expected_station_0 = math.exp(-(16.0 - 1.0) / tau) * kernel_multiplier
        expected_station_2 = (
            math.exp(-(16.0 - 6.0) / tau) * kernel_multiplier
            + math.exp(-(16.0 - 14.0) / tau) * kernel_multiplier
        )

        self.assertAlmostEqual(observation["future_demand"][0], expected_station_0, places=6)
        self.assertAlmostEqual(observation["future_demand"][1], 0.0, places=6)
        self.assertAlmostEqual(observation["future_demand"][2], expected_station_2, places=6)

    def test_demand_forecaster_rejects_unknown_method(self) -> None:
        forecaster = DemandForecaster(station_ids=[0, 1])

        with self.assertRaises(ValueError):
            forecaster.predict(
                method="unknown",
                now=10.0,
                history_log=SimulatorCore(
                    station_specs=[
                        StationSpec(station_id=0, charge_capacity=1),
                        StationSpec(station_id=1, charge_capacity=1),
                    ]
                ).history_log,
            )

    def test_demand_forecaster_accepts_algorithm_params_via_predict(self) -> None:
        simulator = SimulatorCore(
            station_specs=[
                StationSpec(station_id=0, charge_capacity=1),
                StationSpec(station_id=1, charge_capacity=1),
            ]
        )
        simulator.submit_arrival(
            ChargingRequest(vehicle_id=1, station_id=0, charge_duration=4.0, arrival_time=8.0)
        )

        forecaster = DemandForecaster(station_ids=[0, 1])
        prediction = forecaster.predict(
            method="exponential-decay",
            now=10.0,
            history_log=simulator.history_log,
            params={
                "horizon": 10.0,
                "decay_tau": 5.0,
            },
        )

        expected_station_0 = math.exp(-(10.0 - 8.0) / 5.0) * (1.0 - math.exp(-10.0 / 5.0))
        self.assertAlmostEqual(prediction[0], expected_station_0, places=6)
        self.assertAlmostEqual(prediction[1], 0.0, places=6)
