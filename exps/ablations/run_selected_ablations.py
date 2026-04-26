"""Run the requested O2O-IQL ablation suite.

The generic sweep runner executes every variant in configs.ABLATIONS. This file
keeps the paper ablation suite fixed to the seven variants requested for the
O2O-IQL experiment section.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from exps.ablations import configs


SELECTED_ABLATIONS = (
    "offline_only",
    "no_offline",
    "no_dual_buffer",
    "no_density_priority",
    "no_anneal",
    "no_ucb",
)


def _normalize_variants(variants: Iterable[str] | None) -> tuple[str, ...]:
    if variants is None:
        return SELECTED_ABLATIONS
    normalized = tuple(str(variant) for variant in variants)
    unknown = sorted(set(normalized) - set(SELECTED_ABLATIONS))
    if unknown:
        raise ValueError(
            "Unsupported selected-suite variant(s): " + ", ".join(unknown)
        )
    return normalized


def _normalize_seeds(seeds: Iterable[int] | None) -> tuple[int, ...]:
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


def build_training_commands(
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    python: str = sys.executable,
) -> list[list[str]]:
    selected_variants = _normalize_variants(variants)
    selected_seeds = _normalize_seeds(seeds)
    commands: list[list[str]] = []
    for variant in selected_variants:
        for seed in selected_seeds:
            commands.append(
                [
                    python,
                    "-m",
                    "exps.ablations.run_variant",
                    "--variant",
                    variant,
                    "--seed",
                    str(int(seed)),
                ]
            )
    return commands


def build_evaluation_command(
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    splits: Iterable[str] | None = None,
    n_eval_episodes: int = 0,
    python: str = sys.executable,
) -> list[str]:
    selected_variants = _normalize_variants(variants)
    selected_seeds = _normalize_seeds(seeds)
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


def _requires_shared_checkpoint(variants: Sequence[str]) -> bool:
    return any(bool(configs.ABLATIONS[variant].get("use_pretrained")) for variant in variants)


def _run(command: list[str], dry_run: bool = False) -> int:
    print(f"\n$ {' '.join(command)}\n", flush=True)
    if dry_run:
        return 0
    start = time.time()
    proc = subprocess.run(command)
    elapsed = time.time() - start
    print(f"  -> exit={proc.returncode}  elapsed={elapsed / 60:.1f} min", flush=True)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the seven requested O2O-IQL ablation experiments."
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        choices=SELECTED_ABLATIONS,
        help="Subset of the requested ablation variants to run.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=None, choices=configs.SPLITS)
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=0,
        help="0 = all episodes in each split.",
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
        help="Evaluate checkpoints after training finishes.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    variants = _normalize_variants(args.variants)
    seeds = _normalize_seeds(args.seeds)
    shared_ckpt = Path(configs.SHARED_CKPT)

    print("=== Selected O2O-IQL ablation suite ===")
    print(f"  variants: {', '.join(variants)}")
    print(f"  seeds   : {', '.join(str(seed) for seed in seeds)}")

    if not args.skip_pretrain:
        if args.force_pretrain or not shared_ckpt.exists():
            rc = _run(build_pretrain_command(), dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
        else:
            print(f"  Shared offline checkpoint exists -> {shared_ckpt}")

    if (
        not args.dry_run
        and args.skip_pretrain
        and _requires_shared_checkpoint(variants)
        and not shared_ckpt.exists()
    ):
        raise SystemExit(
            f"Shared offline checkpoint missing at '{shared_ckpt}'. "
            "Run without --skip_pretrain or run "
            "`python -m exps.ablations.run_variant --pretrain_only` first."
        )

    if not args.skip_train:
        commands = build_training_commands(variants=variants, seeds=seeds)
        print(f"=== Training {len(commands)} runs ===")
        failures: list[tuple[list[str], int]] = []
        for command in commands:
            rc = _run(command, dry_run=args.dry_run)
            if rc != 0:
                failures.append((command, rc))

        if failures:
            print("\n=== Failed training commands ===")
            for command, rc in failures:
                print(f"  exit={rc}: {' '.join(command)}")
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
