import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from train.evalution import (
    STATION_ASSIGNMENT_NEXT_QUEUE_ERROR_MEAN,
    STATION_ASSIGNMENT_NEXT_QUEUE_ERROR_STD,
    _station_assignment_target_station,
    evaluate_baseline,
)


class StationAssignmentBaselineTests(unittest.TestCase):
    def test_station_assignment_uses_local_route_view(self) -> None:
        env = SimpleNamespace(
            pending_vehicle=SimpleNamespace(route=[0, 1, 2]),
            num_stations=3,
            _current_observation=lambda: {
                "sim_state": {
                    "stations": [
                        {"queue_waiting_time": [10.0, 10.0]},
                        {"queue_waiting_time": [30.0, 30.0, 30.0]},
                        {"queue_waiting_time": []},
                    ]
                }
            },
        )

        self.assertEqual(_station_assignment_target_station(env), 0)

    def test_station_assignment_stays_current_when_queue_advantage_is_small(self) -> None:
        env = SimpleNamespace(
            pending_vehicle=SimpleNamespace(route=[0, 1, 2]),
            num_stations=3,
            _current_observation=lambda: {
                "sim_state": {
                    "stations": [
                        {"queue_waiting_time": [10.0, 10.0]},
                        {"queue_waiting_time": []},
                        {"queue_waiting_time": [40.0]},
                    ]
                }
            },
        )

        self.assertEqual(_station_assignment_target_station(env), 0)

    def test_station_assignment_adds_gaussian_error_to_next_queue(self) -> None:
        rng = Mock()
        rng.normal.return_value = STATION_ASSIGNMENT_NEXT_QUEUE_ERROR_MEAN
        env = SimpleNamespace(
            pending_vehicle=SimpleNamespace(route=[0, 1, 2]),
            num_stations=3,
            np_random=rng,
            _current_observation=lambda: {
                "sim_state": {
                    "stations": [
                        {"queue_waiting_time": list(range(21))},
                        {"queue_waiting_time": []},
                        {"queue_waiting_time": []},
                    ]
                }
            },
        )

        self.assertEqual(_station_assignment_target_station(env), 1)
        rng.normal.assert_called_once_with(
            loc=10.0,
            scale=float(STATION_ASSIGNMENT_NEXT_QUEUE_ERROR_STD),
        )


class BaselineSummaryTests(unittest.TestCase):
    def test_evaluate_baseline_preserves_episode_metric_schema(self) -> None:
        records = [
            {
                "episode_reward": -10.0,
                "episode_length": 2,
                "episode_name": "episode_0000.csv",
                "mean_waiting_time": 1.0,
                "p95_waiting_time": 2.0,
                "max_waiting_time": 3.0,
                "load_imbalance": 0.4,
                "vehicle_count": 4,
                "total_waiting_time": 4.0,
                "invalid_action_count": 0,
                "decision_step_count": 2,
            },
            {
                "episode_reward": -20.0,
                "episode_length": 3,
                "episode_name": "episode_0001.csv",
                "mean_waiting_time": 2.0,
                "p95_waiting_time": 4.0,
                "max_waiting_time": 6.0,
                "load_imbalance": 0.6,
                "vehicle_count": 5,
                "total_waiting_time": 10.0,
                "invalid_action_count": 1,
                "decision_step_count": 3,
            },
        ]

        with patch("train.evalution._evaluate_single_episode", side_effect=records):
            summary = evaluate_baseline(
                baseline_name="station-assignment",
                baseline_fn=lambda _env: 0,
                episode_paths=[Path("episode_0000.csv"), Path("episode_0001.csv")],
                n_bins=21,
                n_eval_episodes=0,
                seed=42,
                num_workers=1,
            )

        self.assertEqual(summary["mean_cv_load_imbalance"], 0.5)
        self.assertEqual(summary["dataset_total_waiting_time"], 14.0)
        self.assertEqual(summary["dataset_vehicle_count"], 9)
        self.assertAlmostEqual(summary["dataset_average_waiting_time"], 14.0 / 9.0)
        self.assertEqual(summary["episode_metrics"][0]["cv_load_imbalance"], 0.4)
        self.assertEqual(summary["episode_metrics"][1]["total_waiting_time"], 10.0)


if __name__ == "__main__":
    unittest.main()
