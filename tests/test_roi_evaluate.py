import unittest

from exps.roi.evaluate import _summarize


class RoiEvaluateSummaryTests(unittest.TestCase):
    def test_summarize_preserves_common_result_schema(self) -> None:
        records = [
            {
                "episode_reward": -10.0,
                "episode_length": 2,
                "episode_name": "episode_0000.csv",
                "vehicle_count": 4,
                "total_waiting_time": 8.0,
                "mean_waiting_time": 2.0,
                "p95_waiting_time": 3.0,
                "max_waiting_time": 4.0,
                "load_imbalance": 0.5,
                "assigned_demand_by_station": [10.0, 20.0],
                "normalized_assigned_demand_by_station": [1.0, 2.0],
                "invalid_action_count": 0,
                "decision_step_count": 2,
            }
        ]

        summary = _summarize(
            records,
            label="roi-seed42",
            scenario="normal",
            seed=42,
            data_dir="data/test_dataset/normal",
        )

        self.assertEqual(summary["scenario"], "normal")
        self.assertEqual(summary["seed"], 42)
        self.assertEqual(summary["dataset_total_waiting_time"], 8.0)
        self.assertEqual(summary["dataset_vehicle_count"], 4)
        self.assertEqual(summary["dataset_average_waiting_time"], 2.0)
        self.assertEqual(summary["mean_cv_load_imbalance"], 0.5)
        self.assertEqual(summary["episode_metrics"][0]["assigned_demand_by_station"], [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
