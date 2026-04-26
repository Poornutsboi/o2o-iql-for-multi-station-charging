import unittest

from exps.ablations import configs
from exps.ablations import run_variant


class AblationDataPathTests(unittest.TestCase):
    def test_config_points_to_existing_offline_expert_data(self) -> None:
        self.assertEqual(configs.OFFLINE_DEMAND_DIR, "data/offline_dataset/demand")
        self.assertEqual(configs.OFFLINE_SOLUTION_DIR, "data/offline_dataset/solutions")

    def test_pretrain_argv_passes_offline_expert_data_paths(self) -> None:
        argv = run_variant._pretrain_argv()

        self.assertIn("--demand_dir", argv)
        self.assertIn("--solution_dir", argv)
        self.assertEqual(
            argv[argv.index("--demand_dir") + 1],
            configs.OFFLINE_DEMAND_DIR,
        )
        self.assertEqual(
            argv[argv.index("--solution_dir") + 1],
            configs.OFFLINE_SOLUTION_DIR,
        )

    def test_variant_argv_passes_offline_expert_data_paths(self) -> None:
        argv, _ = run_variant._variant_argv("no_offline", seed=42)

        self.assertIn("--demand_dir", argv)
        self.assertIn("--solution_dir", argv)
        self.assertEqual(
            argv[argv.index("--demand_dir") + 1],
            configs.OFFLINE_DEMAND_DIR,
        )
        self.assertEqual(
            argv[argv.index("--solution_dir") + 1],
            configs.OFFLINE_SOLUTION_DIR,
        )


if __name__ == "__main__":
    unittest.main()
