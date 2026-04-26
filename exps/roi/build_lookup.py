"""CLI: estimate per-split queue parameters and build the M/M/C/inf wait lookup.

Example:
    python -m exps.roi.build_lookup \
        --data_dir data/train_dataset \
        --output_dir exps/roi/cache \
        --n_snapshots 10000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from exps.roi.estimator import estimate_split_params, save_split_params
from exps.roi.wait_lookup import build_split_lookup, save_split_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ROI per-station M/M/C wait lookup tables for each split.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory whose subdirectories are evaluation splits with episode CSVs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exps/roi/cache",
        help="Directory under which params/<split>.json and lookups/<split>.json are written.",
    )
    parser.add_argument("--n_snapshots", type=int, default=10000)
    parser.add_argument("--q_max", type=int, default=50)
    parser.add_argument(
        "--delta_max", type=float, default=200.0,
        help="Maximum Delta minutes covered by the lookup grid.",
    )
    parser.add_argument(
        "--delta_step", type=float, default=5.0,
        help="Spacing in minutes between Delta grid points.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _delta_grid(delta_max: float, delta_step: float) -> list[float]:
    if delta_step <= 0.0:
        raise ValueError("delta_step must be positive.")
    grid: list[float] = []
    value = 0.0
    while value <= delta_max + 1e-9:
        grid.append(float(value))
        value += float(delta_step)
    return grid


def main() -> None:
    args = parse_args()
    root = Path(args.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir '{root}' not found.")

    split_dirs = sorted(child for child in root.iterdir() if child.is_dir())
    if not split_dirs:
        split_dirs = [root]

    output_dir = Path(args.output_dir)
    params_dir = output_dir / "params"
    lookups_dir = output_dir / "lookups"
    params_dir.mkdir(parents=True, exist_ok=True)
    lookups_dir.mkdir(parents=True, exist_ok=True)

    delta_grid = _delta_grid(float(args.delta_max), float(args.delta_step))

    for split_dir in split_dirs:
        split_name = split_dir.name
        print(f"[ROI] Estimating queue parameters for split '{split_name}'...")
        params = estimate_split_params(split_dir)
        params_path = params_dir / f"{split_name}.json"
        save_split_params(params, params_path)
        print(f"  saved params -> {params_path}")
        for station in params.stations:
            print(
                f"  station {station.station_id:>2}: lambda={station.lambda_per_min:.4f}/min, "
                f"mean_dur={station.mean_duration:.2f}min, mu={station.mu_per_min:.4f}/min, "
                f"C={station.capacity}, rho={station.rho:.3f}"
            )
        print(f"[ROI] Building wait lookup for split '{split_name}' (n_snapshots={args.n_snapshots})...")
        lookup = build_split_lookup(
            params=params,
            q_max=int(args.q_max),
            delta_grid=delta_grid,
            n_snapshots=int(args.n_snapshots),
            seed=int(args.seed),
        )
        lookup_path = lookups_dir / f"{split_name}.json"
        save_split_lookup(lookup, lookup_path)
        print(f"  saved lookup -> {lookup_path}")


if __name__ == "__main__":
    main()
