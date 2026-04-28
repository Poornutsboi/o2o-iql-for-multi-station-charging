"""Run O2O-IQL ablation training jobs in parallel subprocesses.

The shared offline pretraining pass is still run at most once before the
parallel stage. Each (variant, seed) training run is isolated in its own Python
process and writes stdout/stderr to a per-run log file.

Usage:
    python -m exps.ablations.run_parallel_ablations --max_workers 2
    python -m exps.ablations.run_parallel_ablations --variants no_offline no_ucb --seeds 42
    python -m exps.ablations.run_parallel_ablations --skip_pretrain --max_workers 3
    python -m exps.ablations.run_parallel_ablations --evaluate
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from exps.ablations import configs


@dataclass(frozen=True)
class TrainingJob:
    variant: str
    seed: int
    command: tuple[str, ...]
    log_path: Path


@dataclass(frozen=True)
class JobResult:
    job: TrainingJob
    returncode: int
    elapsed_seconds: float


def normalize_variants(variants: Iterable[str] | None) -> tuple[str, ...]:
    if variants is None:
        return tuple(configs.ABLATIONS)

    normalized = tuple(str(variant) for variant in variants)
    unknown = sorted(set(normalized) - set(configs.ABLATIONS))
    if unknown:
        raise ValueError("Unknown ablation variant(s): " + ", ".join(unknown))
    return normalized


def normalize_seeds(seeds: Iterable[int] | None) -> tuple[int, ...]:
    if seeds is None:
        return tuple(int(seed) for seed in configs.SEEDS)
    return tuple(int(seed) for seed in seeds)


def build_pretrain_command(python: str = sys.executable) -> list[str]:
    return [
        python,
        "-m",
        "exps.ablations.run_variant",
        "--pretrain_only",
    ]


def build_training_jobs(
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    python: str = sys.executable,
    log_root: Path | str = Path(configs.ROOT) / "parallel_logs",
) -> list[TrainingJob]:
    selected_variants = normalize_variants(variants)
    selected_seeds = normalize_seeds(seeds)
    root = Path(log_root)

    jobs: list[TrainingJob] = []
    for variant in selected_variants:
        for seed in selected_seeds:
            jobs.append(
                TrainingJob(
                    variant=variant,
                    seed=int(seed),
                    command=(
                        python,
                        "-m",
                        "exps.ablations.run_variant",
                        "--variant",
                        variant,
                        "--seed",
                        str(int(seed)),
                    ),
                    log_path=root / f"{variant}_seed{int(seed)}.log",
                )
            )
    return jobs


def build_evaluation_command(
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    splits: Iterable[str] | None = None,
    n_eval_episodes: int = 0,
    python: str = sys.executable,
) -> list[str]:
    selected_variants = normalize_variants(variants)
    selected_seeds = normalize_seeds(seeds)
    selected_splits = tuple(str(split) for split in (splits or configs.SPLITS))

    return [
        python,
        "-m",
        "exps.ablations.evaluate_ablation",
        "--variants",
        *selected_variants,
        "--seeds",
        *(str(int(seed)) for seed in selected_seeds),
        "--splits",
        *selected_splits,
        "--n_eval_episodes",
        str(int(n_eval_episodes)),
    ]


def requires_shared_checkpoint(variants: Sequence[str]) -> bool:
    return any(bool(configs.ABLATIONS[variant].get("use_pretrained")) for variant in variants)


def resolve_max_workers(max_workers: int, total_jobs: int) -> int:
    if total_jobs <= 0:
        return 0
    if int(max_workers) <= 0:
        return 1
    return min(int(max_workers), int(total_jobs))


def _format_command(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _run(command: list[str], dry_run: bool = False) -> int:
    print(f"\n$ {_format_command(command)}\n", flush=True)
    if dry_run:
        return 0
    start = time.time()
    proc = subprocess.run(command)
    elapsed = time.time() - start
    print(f"  -> exit={proc.returncode}  elapsed={elapsed / 60:.1f} min", flush=True)
    return int(proc.returncode)


def _run_training_job(job: TrainingJob, dry_run: bool = False) -> JobResult:
    print(f"[start] {job.variant}/seed{job.seed} -> {job.log_path}", flush=True)
    if dry_run:
        print(f"        $ {_format_command(job.command)}", flush=True)
        return JobResult(job=job, returncode=0, elapsed_seconds=0.0)

    start = time.time()
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {_format_command(job.command)}\n\n")
        log_file.flush()
        proc = subprocess.run(
            list(job.command),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - start
    print(
        f"[done ] {job.variant}/seed{job.seed} "
        f"exit={proc.returncode} elapsed={elapsed / 60:.1f} min",
        flush=True,
    )
    return JobResult(job=job, returncode=int(proc.returncode), elapsed_seconds=elapsed)


def run_training_jobs(
    jobs: Sequence[TrainingJob],
    max_workers: int,
    dry_run: bool = False,
) -> list[JobResult]:
    worker_count = resolve_max_workers(max_workers=max_workers, total_jobs=len(jobs))
    if worker_count == 0:
        return []

    print(f"=== Training {len(jobs)} runs with max_workers={worker_count} ===", flush=True)
    if dry_run:
        return [_run_training_job(job, dry_run=True) for job in jobs]

    results: list[JobResult] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {
            executor.submit(_run_training_job, job, False): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            results.append(future.result())
    return results


def _print_summary(results: Sequence[JobResult]) -> None:
    failures = [result for result in results if result.returncode != 0]
    print("\n=== Parallel sweep summary ===")
    print(f"  total runs  : {len(results)}")
    print(f"  successful  : {len(results) - len(failures)}")
    print(f"  failed      : {len(failures)}")
    for result in failures:
        job = result.job
        print(
            f"    - {job.variant}/seed{job.seed} "
            f"exit={result.returncode} log={job.log_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run O2O-IQL ablations in parallel.")
    parser.add_argument("--variants", nargs="*", default=None, choices=list(configs.ABLATIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=None, choices=configs.SPLITS)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Maximum concurrent training subprocesses. Use 1 for serial execution.",
    )
    parser.add_argument(
        "--log_root",
        type=Path,
        default=Path(configs.ROOT) / "parallel_logs",
        help="Directory for per-run stdout/stderr logs.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=0,
        help="0 = all episodes in each evaluation split.",
    )
    parser.add_argument(
        "--skip_pretrain",
        action="store_true",
        help="Do not create the shared offline checkpoint.",
    )
    parser.add_argument(
        "--force_pretrain",
        action="store_true",
        help="Rebuild the shared offline checkpoint even if it already exists.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training commands and optionally only evaluate existing checkpoints.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate checkpoints after all training jobs finish.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    variants = normalize_variants(args.variants)
    seeds = normalize_seeds(args.seeds)
    shared_ckpt = Path(configs.SHARED_CKPT)

    print("=== Parallel O2O-IQL ablation suite ===")
    print(f"  variants: {', '.join(variants)}")
    print(f"  seeds   : {', '.join(str(seed) for seed in seeds)}")
    print(f"  logs    : {args.log_root}")

    if not args.skip_pretrain and requires_shared_checkpoint(variants):
        if args.force_pretrain or not shared_ckpt.exists():
            rc = _run(build_pretrain_command(), dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
        else:
            print(f"  Shared offline checkpoint exists -> {shared_ckpt}")

    if (
        not args.dry_run
        and args.skip_pretrain
        and requires_shared_checkpoint(variants)
        and not shared_ckpt.exists()
    ):
        raise SystemExit(
            f"Shared offline checkpoint missing at '{shared_ckpt}'. "
            "Run without --skip_pretrain or run "
            "`python -m exps.ablations.run_variant --pretrain_only` first."
        )

    results: list[JobResult] = []
    if not args.skip_train:
        jobs = build_training_jobs(
            variants=variants,
            seeds=seeds,
            log_root=args.log_root,
        )
        results = run_training_jobs(
            jobs=jobs,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
        )
        _print_summary(results)
        if any(result.returncode != 0 for result in results):
            raise SystemExit(1)

    if args.evaluate:
        command = build_evaluation_command(
            variants=variants,
            seeds=seeds,
            splits=args.splits,
            n_eval_episodes=args.n_eval_episodes,
        )
        rc = _run(command, dry_run=args.dry_run)
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
