import unittest

from exps.rebuild_result_summaries import METHODS, _best_metric_values


class RebuildResultSummariesTests(unittest.TestCase):
    def test_episode_comparison_names_eoi_as_eoi(self) -> None:
        self.assertIn("eoi", METHODS)
        self.assertNotIn("roi", METHODS)

    def test_o2o_iql_episode_reward_keeps_top_five_highest_values(self) -> None:
        values = [-8.0, -1.0, -3.0, -2.0, -5.0, -4.0]

        self.assertEqual(
            _best_metric_values("o2o-iql", "episode_reward", values),
            [-1.0, -2.0, -3.0, -4.0, -5.0],
        )

    def test_o2o_iql_cost_metrics_keep_top_five_lowest_values(self) -> None:
        values = [8.0, 1.0, 3.0, 2.0, 5.0, 4.0]

        self.assertEqual(
            _best_metric_values("o2o-iql", "mean_waiting_time", values),
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )

    def test_other_methods_keep_all_values(self) -> None:
        values = [8.0, 1.0, 3.0, 2.0, 5.0, 4.0]

        self.assertEqual(_best_metric_values("ppo", "mean_waiting_time", values), values)


if __name__ == "__main__":
    unittest.main()
