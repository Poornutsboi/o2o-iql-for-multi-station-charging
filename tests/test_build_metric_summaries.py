import unittest
from unittest.mock import patch

import pandas as pd

from exps import build_metric_summaries
from exps.build_metric_summaries import (
    _ordered_summary_columns,
    _rank_high_avg_wait,
    _summary_episode_filter,
)


class BuildMetricSummariesTests(unittest.TestCase):
    def test_requested_summary_output_names_are_available(self) -> None:
        output_stems = getattr(build_metric_summaries, "METRIC_OUTPUT_STEMS", {})

        self.assertIn("avg_wait", output_stems)
        self.assertIn("p95_wait", output_stems)
        self.assertIn("cv_li", output_stems)
        self.assertEqual(output_stems["avg_wait"], ["avg_wait_summary"])
        self.assertIn("p956_wait_summary", output_stems["p95_wait"])
        self.assertIn("cv_cli_summary", output_stems["cv_li"])

    def test_summary_episode_filter_preserves_original_bias_subset(self) -> None:
        with patch(
            "exps.build_metric_summaries._select_bias_episode_ids",
            return_value={"0001", "0003"},
        ):
            self.assertEqual(_summary_episode_filter(), {"bias": {"0001", "0003"}})

    def test_ordered_summary_columns_uses_one_order_for_all_metrics(self) -> None:
        metric_tables = {
            "avg_wait": pd.DataFrame(
                {
                    "normal_0000": {"o2o-iql": 3.0},
                    "normal_0001": {"o2o-iql": 1.0},
                    "bias_0000": {"o2o-iql": 0.0},
                    "bias_0001": {"o2o-iql": 0.0},
                    "extreme_0000": {"o2o-iql": 2.0},
                    "extreme_0001": {"o2o-iql": 1.0},
                }
            ),
            "p95_wait": pd.DataFrame(
                {
                    "normal_0000": {"o2o-iql": 30.0},
                    "normal_0001": {"o2o-iql": 10.0},
                    "bias_0000": {"o2o-iql": 40.0},
                    "bias_0001": {"o2o-iql": 90.0},
                    "extreme_0000": {"o2o-iql": 20.0},
                    "extreme_0001": {"o2o-iql": 10.0},
                }
            ),
        }

        self.assertEqual(
            _ordered_summary_columns(metric_tables),
            [
                "normal_0001",
                "normal_0000",
                "bias_0000",
                "bias_0001",
                "extreme_0000",
                "extreme_0001",
            ],
        )

    def test_rank_high_avg_wait_prefers_large_nonzero_waiting_time(self) -> None:
        avg_wait_table = pd.DataFrame(
            {
                "bias_0000": {"o2o-iql": 0.0, "ppo": 1.0, "eoi": 0.0},
                "bias_0001": {"o2o-iql": 10.0, "ppo": 12.0, "eoi": 8.0},
                "bias_0002": {"o2o-iql": 6.0, "ppo": 6.0, "eoi": 6.0},
            }
        )

        self.assertEqual(
            _rank_high_avg_wait(avg_wait_table, ["bias_0000", "bias_0001", "bias_0002"]),
            ["bias_0001", "bias_0002", "bias_0000"],
        )


if __name__ == "__main__":
    unittest.main()
