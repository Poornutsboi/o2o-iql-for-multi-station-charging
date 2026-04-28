"""Plot dense reward curves for the O2O-IQL ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exps.ablations import configs


RUNS_ROOT = Path(configs.ROOT) / "runs"
RESULTS_ROOT = Path(configs.ROOT) / "results"

VARIANT_ORDER = (
    "full_o2o_iql",
    "no_offline",
    "no_dual_buffer",
    "no_density_priority",
    "no_anneal",
    "no_ucb",
    "offline_only",
)
VARIANT_LABELS = {
    "full_o2o_iql": "Full O2O-IQL",
    "no_offline": "No offline",
    "no_dual_buffer": "No dual buffer",
    "no_density_priority": "No density priority",
    "no_anneal": "No anneal",
    "no_ucb": "No UCB",
    "offline_only": "Offline only",
}
VARIANT_COLORS = {
    "full_o2o_iql": "#0072B2",
    "no_offline": "#D55E00",
    "no_dual_buffer": "#009E73",
    "no_density_priority": "#CC79A7",
    "no_anneal": "#E69F00",
    "no_ucb": "#56B4E9",
    "offline_only": "#666666",
}
ROLLING_REWARD_WINDOW = 20
GROUP_SPECS = {
    "replay_buffer": {
        "title": "Replay Buffer Ablations on Concentrated Demand",
        "variants": ("full_o2o_iql", "no_dual_buffer", "no_density_priority"),
        "output_stem": "ablation_replay_buffer_mean_reward_curve",
    },
    "offline_online": {
        "title": "Offline/Online Ablations on Concentrated Demand",
        "variants": ("full_o2o_iql", "offline_only", "no_offline"),
        "output_stem": "ablation_offline_online_mean_reward_curve",
    },
    "ucb_anneal": {
        "title": "UCB and Annealing Ablations on Concentrated Demand",
        "variants": ("full_o2o_iql", "no_ucb", "no_anneal"),
        "output_stem": "ablation_ucb_anneal_mean_reward_curve",
    },
}


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant.replace("_", " "))


def collect_episode_curves(
    runs_root: Path = RUNS_ROOT,
    variants: Iterable[str] = VARIANT_ORDER,
    seeds: Iterable[int] = configs.SEEDS,
) -> list[dict]:
    """Collect one dense per-vehicle mean-reward sample per completed episode."""
    rows: list[dict] = []
    for variant in variants:
        for seed in seeds:
            log_path = runs_root / variant / f"seed{int(seed)}" / "logs" / "online_episode.jsonl"
            seed_rewards: list[float] = []
            for record in _read_jsonl(log_path):
                if "step" not in record or "episode_return" not in record or "episode_length" not in record:
                    continue
                episode_length = int(record["episode_length"])
                if episode_length <= 0:
                    continue
                reward = float(record["episode_return"]) / float(episode_length)
                seed_rewards.append(reward)
                recent_rewards = seed_rewards[-ROLLING_REWARD_WINDOW:]
                rows.append(
                    {
                        "variant": variant,
                        "label": _variant_label(variant),
                        "seed": int(seed),
                        "episode_idx": int(record.get("episode_idx", len(rows) + 1)),
                        "step": int(record["step"]),
                        "episode_return": float(record["episode_return"]),
                        "episode_length": int(episode_length),
                        "reward": reward,
                        "recent_reward_mean": float(np.mean(recent_rewards)),
                    }
                )
    return sorted(rows, key=lambda row: (VARIANT_ORDER.index(row["variant"]) if row["variant"] in VARIANT_ORDER else 999, row["seed"], row["step"]))


def build_step_summary(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["label"]), int(row["step"]))].append(row)

    summary: list[dict] = []
    for (variant, label, step), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        rewards = np.asarray([float(row["reward"]) for row in group], dtype=np.float64)
        seeds = {int(row["seed"]) for row in group}
        summary.append(
            {
                "variant": variant,
                "label": label,
                "step": int(step),
                "mean_reward": float(rewards.mean()),
                "std_reward_across_seeds": float(rewards.std(ddof=0)),
                "n_seeds": int(len(seeds)),
                "n_points": int(len(group)),
            }
        )
    return summary


def build_interpolated_summary(rows: Iterable[dict], value_key: str = "recent_reward_mean") -> list[dict]:
    by_variant_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_variant_seed[(str(row["variant"]), int(row["seed"]))].append(row)

    by_variant: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for (variant, seed), group in by_variant_seed.items():
        ordered = sorted(group, key=lambda row: int(row["step"]))
        steps = np.asarray([int(row["step"]) for row in ordered], dtype=np.float64)
        values = np.asarray([float(row[value_key]) for row in ordered], dtype=np.float64)
        if len(steps) > 0:
            by_variant[variant].append((seed, steps, values))

    summary: list[dict] = []
    for variant in sorted(by_variant, key=lambda name: VARIANT_ORDER.index(name) if name in VARIANT_ORDER else 999):
        grid = np.unique(np.concatenate([steps for _, steps, _ in by_variant[variant]])).astype(np.float64)
        for step in grid:
            values: list[float] = []
            seeds: list[int] = []
            for seed, seed_steps, seed_values in by_variant[variant]:
                if step < seed_steps[0] or step > seed_steps[-1]:
                    continue
                values.append(float(np.interp(step, seed_steps, seed_values)))
                seeds.append(seed)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            summary.append(
                {
                    "variant": variant,
                    "label": _variant_label(variant),
                    "step": int(step),
                    "mean_reward": float(arr.mean()),
                    "std_reward_across_seeds": float(arr.std(ddof=0)),
                    "n_seeds": int(len(set(seeds))),
                    "n_points": int(len(values)),
                }
            )
    return summary


def build_analysis_table(rows: Iterable[dict], curve: Iterable[dict]) -> list[dict]:
    raw_by_variant: dict[str, list[dict]] = defaultdict(list)
    curve_by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        raw_by_variant[str(row["variant"])].append(row)
    for row in curve:
        curve_by_variant[str(row["variant"])].append(row)

    analysis: list[dict] = []
    for variant in sorted(raw_by_variant, key=lambda name: VARIANT_ORDER.index(name) if name in VARIANT_ORDER else 999):
        raw = raw_by_variant[variant]
        dense = sorted(curve_by_variant.get(variant, []), key=lambda row: int(row["step"]))
        max_step = max(int(row["step"]) for row in raw)
        tail_start = int(max_step * 0.8)
        tail_values = [float(row["mean_reward"]) for row in dense if int(row["step"]) >= tail_start]
        final_row = dense[-1] if dense else None
        analysis.append(
            {
                "variant": variant,
                "label": _variant_label(variant),
                "n_raw_episode_points": int(len(raw)),
                "n_curve_points": int(len(dense)),
                "n_seeds_with_online_curve": int(len({int(row["seed"]) for row in raw})),
                "max_step": int(max_step),
                "final_curve_reward": float(final_row["mean_reward"]) if final_row else "",
                "tail_20pct_curve_reward_mean": float(np.mean(tail_values)) if tail_values else "",
            }
        )
    return analysis


def _offline_only_reward(results_root: Path) -> float | None:
    path = results_root / "offline_only" / "bias" / "summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    reward = aggregate.get("mean_reward_mean")
    episode_length = aggregate.get("mean_ep_length_mean")
    if reward is None or episode_length in (None, 0):
        return None
    return float(reward) / float(episode_length)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_dense_curve(
    curve: list[dict],
    output_path: Path,
    results_root: Path = RESULTS_ROOT,
    variants: Iterable[str] = VARIANT_ORDER,
    title: str = "Ablation Mean Reward Curves on Concentrated Demand",
) -> Path:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in curve:
        by_variant[str(row["variant"])].append(row)

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    max_step = 0
    variants = tuple(variants)
    for variant in variants:
        if variant == "offline_only":
            continue
        rows = sorted(by_variant.get(variant, []), key=lambda row: int(row["step"]))
        if not rows:
            continue
        steps = np.asarray([int(row["step"]) for row in rows], dtype=np.float64)
        rewards = np.asarray([float(row["mean_reward"]) for row in rows], dtype=np.float64)
        stds = np.asarray([float(row["std_reward_across_seeds"]) for row in rows], dtype=np.float64)
        color = VARIANT_COLORS.get(variant)
        linewidth = 2.2 if variant == "full_o2o_iql" else 1.6
        ax.plot(steps, rewards, label=_variant_label(variant), color=color, linewidth=linewidth)
        ax.fill_between(steps, rewards - stds, rewards + stds, color=color, alpha=0.10, linewidth=0)
        max_step = max(max_step, int(steps[-1]))

    offline_reward = _offline_only_reward(results_root) if "offline_only" in variants else None
    if offline_reward is not None and max_step > 0:
        ax.hlines(
            offline_reward,
            xmin=0,
            xmax=max_step,
            colors=VARIANT_COLORS["offline_only"],
            linestyles="--",
            linewidth=1.8,
            label=f"{_variant_label('offline_only')} eval",
        )

    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.set_xlabel("Online environment step", fontsize=13)
    ax.set_ylabel("Mean reward per vehicle", fontsize=13)
    ax.grid(True, axis="both", linestyle="--", color="#b8b8b8", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)
    return output_path


def plot_group_curves(curve: list[dict], results_root: Path = RESULTS_ROOT) -> list[Path]:
    outputs: list[Path] = []
    for spec in GROUP_SPECS.values():
        for suffix in ("png", "svg"):
            output_path = results_root / f"{spec['output_stem']}.{suffix}"
            outputs.append(
                plot_dense_curve(
                    curve=curve,
                    output_path=output_path,
                    results_root=results_root,
                    variants=spec["variants"],
                    title=str(spec["title"]),
                )
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dense O2O-IQL ablation reward curves.")
    parser.add_argument("--runs_root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--results_root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--variants", nargs="*", default=list(VARIANT_ORDER))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(configs.SEEDS))
    args = parser.parse_args()

    rows = collect_episode_curves(args.runs_root, args.variants, args.seeds)
    curve = build_interpolated_summary(rows, value_key="recent_reward_mean")
    exact = build_step_summary(rows)
    analysis = build_analysis_table(rows, curve)

    raw_csv = args.results_root / "ablation_dense_mean_episode_rewards.csv"
    curve_csv = args.results_root / "ablation_dense_mean_reward_curve.csv"
    exact_csv = args.results_root / "ablation_dense_mean_reward_curve_exact_steps.csv"
    analysis_csv = args.results_root / "ablation_dense_mean_reward_analysis.csv"

    write_csv(raw_csv, rows)
    write_csv(curve_csv, curve)
    write_csv(exact_csv, exact)
    write_csv(analysis_csv, analysis)
    plot_paths = plot_group_curves(curve, results_root=args.results_root)

    print(f"raw mean-reward samples: {len(rows)} -> {raw_csv}")
    print(f"interpolated dense curve samples: {len(curve)} -> {curve_csv}")
    print(f"analysis table -> {analysis_csv}")
    for plot_path in plot_paths:
        print(f"plot -> {plot_path}")


if __name__ == "__main__":
    main()
