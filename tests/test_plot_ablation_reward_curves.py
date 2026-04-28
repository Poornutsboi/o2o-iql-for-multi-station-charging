import json
import tempfile
import unittest
from pathlib import Path


class PlotAblationRewardCurvesTests(unittest.TestCase):
    def test_collect_episode_curves_preserves_dense_episode_samples_as_mean_reward(self) -> None:
        from exps.ablations.plot_reward_curves import collect_episode_curves

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "runs" / "demo_variant" / "seed42" / "logs"
            log_dir.mkdir(parents=True)
            records = [
                {"step": 10, "episode_return": -5.0, "episode_length": 5},
                {"step": 25, "episode_return": -2.5, "episode_length": 10},
                {"step": 40, "episode_return": -1.0, "episode_length": 20},
            ]
            with (log_dir / "online_episode.jsonl").open("w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")

            curves = collect_episode_curves(
                runs_root=root / "runs",
                variants=("demo_variant",),
                seeds=(42,),
            )

        self.assertEqual(len(curves), 3)
        self.assertEqual([row["step"] for row in curves], [10, 25, 40])
        self.assertEqual([row["reward"] for row in curves], [-1.0, -0.25, -0.05])
        self.assertEqual(
            [round(row["recent_reward_mean"], 6) for row in curves],
            [-1.0, -0.625, -0.433333],
        )
        self.assertEqual([row["episode_length"] for row in curves], [5, 10, 20])
        self.assertEqual({row["variant"] for row in curves}, {"demo_variant"})
        self.assertEqual({row["seed"] for row in curves}, {42})

    def test_build_step_summary_uses_all_available_seed_points_at_each_step(self) -> None:
        from exps.ablations.plot_reward_curves import build_step_summary

        rows = [
            {"variant": "a", "label": "A", "seed": 1, "step": 10, "reward": -4.0},
            {"variant": "a", "label": "A", "seed": 2, "step": 10, "reward": -2.0},
            {"variant": "a", "label": "A", "seed": 1, "step": 20, "reward": -1.0},
        ]

        summary = build_step_summary(rows)

        self.assertEqual(
            summary,
            [
                {
                    "variant": "a",
                    "label": "A",
                    "step": 10,
                    "mean_reward": -3.0,
                    "std_reward_across_seeds": 1.0,
                    "n_seeds": 2,
                    "n_points": 2,
                },
                {
                    "variant": "a",
                    "label": "A",
                    "step": 20,
                    "mean_reward": -1.0,
                    "std_reward_across_seeds": 0.0,
                    "n_seeds": 1,
                    "n_points": 1,
                },
            ],
        )

    def test_group_specs_separate_requested_ablation_questions(self) -> None:
        from exps.ablations.plot_reward_curves import GROUP_SPECS

        self.assertEqual(
            GROUP_SPECS["replay_buffer"]["variants"],
            ("full_o2o_iql", "no_dual_buffer", "no_density_priority"),
        )
        self.assertEqual(
            GROUP_SPECS["offline_online"]["variants"],
            ("full_o2o_iql", "offline_only", "no_offline"),
        )
        self.assertEqual(
            GROUP_SPECS["ucb_anneal"]["variants"],
            ("full_o2o_iql", "no_ucb", "no_anneal"),
        )


if __name__ == "__main__":
    unittest.main()
