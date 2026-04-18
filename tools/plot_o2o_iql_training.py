"""Plot training curves for O2O IQL runs from existing metrics logs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_seed_metrics(metrics_path: Path) -> dict[str, list[dict[str, Any]]]:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        by_stage[record["stage"]].append(record)
    return dict(by_stage)


def _aggregate_series(
    seed_records: dict[str, dict[str, list[dict[str, Any]]]],
    stage: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_by_step: dict[int, list[float]] = defaultdict(list)
    for records in seed_records.values():
        for row in records.get(stage, []):
            if metric not in row:
                continue
            values_by_step[int(row["step"])].append(float(row[metric]))

    if not values_by_step:
        return np.asarray([]), np.asarray([]), np.asarray([])

    steps = np.asarray(sorted(values_by_step), dtype=np.int64)
    means = np.asarray(
        [np.mean(values_by_step[int(step)]) for step in steps],
        dtype=np.float64,
    )
    stds = np.asarray(
        [np.std(values_by_step[int(step)], ddof=0) for step in steps],
        dtype=np.float64,
    )
    return steps, means, stds


def _plot_seed_lines(
    ax: plt.Axes,
    seed_records: dict[str, dict[str, list[dict[str, Any]]]],
    stage: str,
    metric: str,
    *,
    alpha: float = 0.28,
    linewidth: float = 1.5,
) -> None:
    for seed, records in sorted(seed_records.items(), key=lambda item: int(item[0])):
        rows = records.get(stage, [])
        if not rows:
            continue
        xs = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
        ys = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        ax.plot(xs, ys, alpha=alpha, linewidth=linewidth, label=f"seed{seed}")


def _format_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)


def plot_training_process(run_root: Path, scenario: str, output_path: Path) -> None:
    scenario_dir = run_root / scenario
    seed_dirs = sorted(child for child in scenario_dir.glob("seed*") if child.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found under {scenario_dir}")

    seed_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "logs" / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        seed = seed_dir.name.replace("seed", "")
        seed_records[seed] = _load_seed_metrics(metrics_path)

    if not seed_records:
        raise FileNotFoundError(f"No metrics.jsonl files found under {scenario_dir}")

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    online_stage = "online_train"
    if not any(records.get(online_stage) for records in seed_records.values()):
        online_stage = "online"

    # Panel 1: online training return
    ax = axes[0, 0]
    _plot_seed_lines(ax, seed_records, online_stage, "recent_return_mean")
    steps, mean_vals, std_vals = _aggregate_series(seed_records, online_stage, "recent_return_mean")
    ax.plot(steps, mean_vals, color="black", linewidth=2.6, label="seed mean")
    ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color="black", alpha=0.12)
    _format_axis(ax, f"{scenario} online training return", "Recent return mean")
    ax.legend(ncol=2, fontsize=9)

    # Panel 2: evaluation reward
    ax = axes[0, 1]
    _plot_seed_lines(ax, seed_records, "eval", "mean_reward")
    steps, mean_vals, std_vals = _aggregate_series(seed_records, "eval", "mean_reward")
    ax.plot(steps, mean_vals, color="black", linewidth=2.6, label="seed mean")
    ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color="black", alpha=0.12)

    offline_steps, offline_mean, offline_std = _aggregate_series(seed_records, "offline_eval", "mean_reward")
    if offline_steps.size > 0:
        ax.errorbar(
            offline_steps,
            offline_mean,
            yerr=offline_std,
            fmt="o",
            markersize=8,
            color="tab:red",
            capsize=4,
            label="offline init",
        )
    _format_axis(ax, f"{scenario} evaluation reward", "Mean eval reward")
    ax.legend(fontsize=9)

    # Panel 3: actor/value loss
    ax = axes[1, 0]
    for metric, color, label in (
        ("actor_loss", "tab:blue", "actor_loss"),
        ("value_loss", "tab:green", "value_loss"),
    ):
        steps, mean_vals, std_vals = _aggregate_series(seed_records, online_stage, metric)
        ax.plot(steps, mean_vals, color=color, linewidth=2.0, label=label)
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.12)
    _format_axis(ax, f"{scenario} optimization losses", "Loss")
    ax.legend(fontsize=9)

    # Panel 4: critic loss and replay refresh signal
    ax = axes[1, 1]
    critic_steps, critic_mean, critic_std = _aggregate_series(seed_records, online_stage, "critic_loss")
    ax.plot(critic_steps, critic_mean, color="tab:orange", linewidth=2.0, label="critic_loss")
    ax.fill_between(
        critic_steps,
        critic_mean - critic_std,
        critic_mean + critic_std,
        color="tab:orange",
        alpha=0.12,
    )
    _format_axis(ax, f"{scenario} critic / priority dynamics", "Critic loss")

    ax2 = ax.twinx()
    pr_steps, pr_mean, _ = _aggregate_series(seed_records, "priority_refresh", "priority_effective_sample_size")
    if pr_steps.size > 0:
        ax2.plot(
            pr_steps,
            pr_mean,
            color="tab:purple",
            linewidth=1.8,
            linestyle="--",
            label="priority_ess",
        )
        ax2.set_ylabel("Priority ESS")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    fig.suptitle(
        f"O2O IQL training process: {scenario} ({len(seed_records)} seeds)",
        fontsize=14,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot O2O IQL training curves from existing logs.")
    parser.add_argument("--run_root", type=str, default="runs/o2o_iql")
    parser.add_argument("--scenario", type=str, default="bias")
    parser.add_argument(
        "--output",
        type=str,
        default="runs/o2o_iql/bias/training_process.png",
    )
    args = parser.parse_args()

    plot_training_process(
        run_root=Path(args.run_root),
        scenario=args.scenario,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
