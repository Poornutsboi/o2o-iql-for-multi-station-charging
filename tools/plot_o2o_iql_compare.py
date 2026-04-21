"""Plot training curves comparing o2o_iql and o2o_iql_2 on the bias scenario."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['o2o_iql', 'o2o_iql_2']
SEEDS = [42, 123, 2024]
COLORS = {42: '#1f77b4', 123: '#ff7f0e', 2024: '#2ca02c'}


def load_eval(path: Path):
    steps, rewards, stds = [], [], []
    off_step, off_reward = None, None
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            st = d.get('stage', '')
            if st == 'offline_eval':
                off_step, off_reward = d.get('step', 0), d.get('mean_reward')
            elif st == 'eval':
                steps.append(d.get('step', 0))
                rewards.append(d.get('mean_reward'))
                stds.append(d.get('std_reward'))
    return off_step, off_reward, np.asarray(steps), np.asarray(rewards), np.asarray(stds)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # --- per-run raw curves ---
    per_run_mean = {}
    for ax, run in zip(axes[:2], RUNS):
        agg = []
        for seed in SEEDS:
            p = ROOT / 'runs' / run / 'bias' / f'seed{seed}' / 'logs' / 'metrics.jsonl'
            off_s, off_r, steps, rewards, stds = load_eval(p)
            ax.plot(steps, rewards, color=COLORS[seed], alpha=0.9, lw=1.5,
                    label=f'seed{seed}')
            ax.fill_between(steps, rewards - stds, rewards + stds,
                            color=COLORS[seed], alpha=0.08)
            ax.scatter([off_s], [off_r], color=COLORS[seed], marker='x', s=60, zorder=5)
            agg.append(rewards)
        # mean curve across seeds (only if step grids match)
        lens = {len(a) for a in agg}
        if len(lens) == 1:
            stack = np.stack(agg)
            mean_r = stack.mean(axis=0)
            ax.plot(steps, mean_r, color='black', lw=2.2, label='mean')
            per_run_mean[run] = (steps, mean_r, off_r)
        ax.axhline(off_r, color='grey', ls='--', lw=0.8, alpha=0.6,
                   label=f'offline baseline ({off_r:.0f})')
        ax.set_title(f'{run}  (bias)')
        ax.set_xlabel('env step')
        ax.set_ylabel('eval mean_reward')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)

    # --- normalized improvement subplot ---
    ax = axes[2]
    for run, ls in zip(RUNS, ['--', '-']):
        if run not in per_run_mean:
            continue
        steps, mean_r, off_r = per_run_mean[run]
        # improvement over offline baseline (higher = better)
        rel = (off_r - mean_r) / abs(off_r) * 100.0
        ax.plot(steps, rel, lw=2.2, ls=ls, label=run)
    ax.axhline(0, color='grey', lw=0.8, alpha=0.6)
    ax.set_title('Relative improvement over offline baseline\n(mean across 3 seeds)')
    ax.set_xlabel('env step')
    ax.set_ylabel('(offline - online) / |offline|  [%]')
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle('o2o_iql vs o2o_iql_2 — bias scenario, first 3 seeds', y=1.02,
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    out_dir = ROOT / 'runs'
    out_png = out_dir / 'o2o_iql_compare_bias.png'
    out_svg = out_dir / 'o2o_iql_compare_bias.svg'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    print(f'saved: {out_png}')
    print(f'saved: {out_svg}')


if __name__ == '__main__':
    main()
