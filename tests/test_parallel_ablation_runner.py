import sys
import unittest
from pathlib import Path

from exps.ablations import configs
from exps.ablations import run_parallel_ablations


class ParallelAblationRunnerTests(unittest.TestCase):
    def test_training_jobs_cover_variants_and_seeds_with_log_paths(self) -> None:
        jobs = run_parallel_ablations.build_training_jobs(
            variants=("offline_only", "no_ucb"),
            seeds=(42, 123),
            python=sys.executable,
            log_root=Path("logs"),
        )

        self.assertEqual([job.variant for job in jobs], ["offline_only", "offline_only", "no_ucb", "no_ucb"])
        self.assertEqual([job.seed for job in jobs], [42, 123, 42, 123])
        self.assertEqual(
            [list(job.command) for job in jobs],
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
        self.assertEqual(
            [job.log_path for job in jobs],
            [
                Path("logs/offline_only_seed42.log"),
                Path("logs/offline_only_seed123.log"),
                Path("logs/no_ucb_seed42.log"),
                Path("logs/no_ucb_seed123.log"),
            ],
        )

    def test_requires_shared_checkpoint_only_for_pretrained_variants(self) -> None:
        self.assertFalse(run_parallel_ablations.requires_shared_checkpoint(("no_offline",)))
        self.assertTrue(run_parallel_ablations.requires_shared_checkpoint(("no_offline", "no_ucb")))

    def test_default_variants_follow_ablation_config_order(self) -> None:
        self.assertEqual(
            run_parallel_ablations.normalize_variants(None),
            tuple(configs.ABLATIONS),
        )

    def test_max_workers_is_bounded_by_job_count(self) -> None:
        self.assertEqual(run_parallel_ablations.resolve_max_workers(8, total_jobs=3), 3)
        self.assertEqual(run_parallel_ablations.resolve_max_workers(0, total_jobs=3), 1)


if __name__ == "__main__":
    unittest.main()
