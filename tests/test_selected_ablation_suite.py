import sys
import unittest

from exps.ablations import configs
from exps.ablations import run_selected_ablations


class SelectedAblationSuiteTests(unittest.TestCase):
    def test_selected_suite_contains_only_requested_variants_in_order(self) -> None:
        self.assertEqual(
            run_selected_ablations.SELECTED_ABLATIONS,
            (
                "offline_only",
                "no_offline",
                "no_dual_buffer",
                "no_density_priority",
                "no_anneal",
                "no_ucb",
            ),
        )
        for variant in run_selected_ablations.SELECTED_ABLATIONS:
            self.assertIn(variant, configs.ABLATIONS)

    def test_offline_only_variant_reuses_shared_pretrain_without_online_steps(self) -> None:
        offline_only = configs.ABLATIONS["offline_only"]

        self.assertTrue(offline_only["use_pretrained"])
        self.assertIsNone(offline_only["patch"])
        self.assertEqual(offline_only["cli"], {"online_steps": 0})

    def test_training_commands_cover_selected_variants_and_seeds(self) -> None:
        commands = run_selected_ablations.build_training_commands(
            variants=("offline_only", "no_ucb"),
            seeds=(42, 123),
            python=sys.executable,
        )

        self.assertEqual(
            commands,
            [
                [
                    sys.executable,
                    "-m",
                    "exps.ablations.run_variant",
                    "--variant",
                    "offline_only",
                    "--seed",
                    "42",
                ],
                [
                    sys.executable,
                    "-m",
                    "exps.ablations.run_variant",
                    "--variant",
                    "offline_only",
                    "--seed",
                    "123",
                ],
                [
                    sys.executable,
                    "-m",
                    "exps.ablations.run_variant",
                    "--variant",
                    "no_ucb",
                    "--seed",
                    "42",
                ],
                [
                    sys.executable,
                    "-m",
                    "exps.ablations.run_variant",
                    "--variant",
                    "no_ucb",
                    "--seed",
                    "123",
                ],
            ],
        )

    def test_evaluation_command_uses_selected_variants(self) -> None:
        command = run_selected_ablations.build_evaluation_command(
            variants=("offline_only", "no_ucb"),
            seeds=(42,),
            splits=("normal", "bias"),
            n_eval_episodes=20,
            python=sys.executable,
        )

        self.assertEqual(
            command,
            [
                sys.executable,
                "-m",
                "exps.ablations.evaluate_ablation",
                "--variants",
                "offline_only",
                "no_ucb",
                "--seeds",
                "42",
                "--splits",
                "normal",
                "bias",
                "--n_eval_episodes",
                "20",
            ],
        )


if __name__ == "__main__":
    unittest.main()
