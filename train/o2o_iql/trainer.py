"""Balanced dual-buffer offline-to-online IQL trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from train.iql.agent import DiscreteIQLAgent, IQLUpdateMetrics
from train.iql.data import (
    TransitionDataset,
    build_episode_bank_env,
    load_episode_bank_from_dir,
    load_offline_dataset,
)
from train.iql.iql_trainer import evaluate_agent
from train.o2o_iql.replay import BalancedReplayManager, PriorityRefreshStats


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path: str, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _mean_metrics(window: list[IQLUpdateMetrics]) -> dict[str, float]:
    if not window:
        return {}
    return {
        "actor_loss": float(np.mean([item.actor_loss for item in window])),
        "critic_loss": float(np.mean([item.critic_loss for item in window])),
        "value_loss": float(np.mean([item.value_loss for item in window])),
        "actor_grad_norm": float(np.mean([item.actor_grad_norm for item in window])),
        "critic_grad_norm": float(np.mean([item.critic_grad_norm for item in window])),
        "value_grad_norm": float(np.mean([item.value_grad_norm for item in window])),
        "mean_q": float(np.mean([item.mean_q for item in window])),
        "std_q": float(np.mean([item.std_q for item in window])),
        "mean_target_q": float(np.mean([item.mean_target_q for item in window])),
        "mean_v": float(np.mean([item.mean_v for item in window])),
        "td_error_mean": float(np.mean([item.td_error_mean for item in window])),
        "td_error_abs_mean": float(np.mean([item.td_error_abs_mean for item in window])),
        "td_error_abs_p90": float(np.mean([item.td_error_abs_p90 for item in window])),
        "policy_entropy": float(np.mean([item.policy_entropy for item in window])),
        "action_std": float(np.mean([item.action_std for item in window])),
        "log_prob_mean": float(np.mean([item.log_prob_mean for item in window])),
        "mean_advantage": float(np.mean([item.mean_advantage for item in window])),
        "advantage_mean": float(np.mean([item.mean_advantage for item in window])),
        "advantage_std": float(np.mean([item.std_advantage for item in window])),
        "advantage_p90": float(np.mean([item.advantage_p90 for item in window])),
        "mean_weight": float(np.mean([item.mean_weight for item in window])),
        "weight_mean": float(np.mean([item.mean_weight for item in window])),
        "weight_std": float(np.mean([item.std_weight for item in window])),
        "weight_max": float(np.max([item.max_weight for item in window])),
        "positive_adv_ratio": float(np.mean([item.positive_adv_ratio for item in window])),
    }


def _default_run_id(args: argparse.Namespace) -> str:
    scenario = Path(args.train_data_dir).name or "run"
    return f"{args.algo}_{scenario}_seed{args.seed}"


class StructuredRunLogger:
    def __init__(self, log_dir: str, metrics_path: str) -> None:
        self.log_dir = Path(log_dir)
        self.metrics_path = Path(metrics_path)
        self.stage_paths = {
            "offline_eval": self.log_dir / "offline_eval.jsonl",
            "online_train": self.log_dir / "online_train.jsonl",
            "online_episode": self.log_dir / "online_episode.jsonl",
            "priority_refresh": self.log_dir / "priority_refresh.jsonl",
            "eval": self.log_dir / "eval.jsonl",
        }
        self.run_config_path = self.log_dir / "run_config.json"

    def initialize(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for path in [self.metrics_path, *self.stage_paths.values()]:
            path.write_text("", encoding="utf-8")

    def write_run_config(self, payload: dict[str, object]) -> None:
        self.run_config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def log(self, payload: dict[str, object]) -> None:
        _append_jsonl(str(self.metrics_path), payload)
        stage = str(payload.get("stage", ""))
        stage_path = self.stage_paths.get(stage)
        if stage_path is not None:
            _append_jsonl(str(stage_path), payload)


def _build_run_config_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "run_id": str(args.run_id),
        "seed": int(args.seed),
        "algo": str(args.algo),
        "env_name": str(args.env_name),
        "total_env_steps": int(args.online_steps),
        "batch_size": int(args.batch_size),
        "actor_lr": float(args.learning_rate),
        "critic_lr": float(args.learning_rate),
        "value_lr": float(args.learning_rate),
        "discount": float(args.discount),
        "tau": float(args.target_update_rate),
        "expectile": float(args.expectile),
        "temperature": float(args.temperature),
        "eval_interval": int(args.eval_freq),
        "priority_refresh_interval": int(args.priority_refresh_freq),
        "n_bins": int(args.n_bins),
        "max_queue_len": int(args.max_queue_len),
        "online_buffer_size": int(args.online_buffer_size),
        "updates_per_step": int(args.updates_per_step),
        "start_training_after": int(args.start_training_after),
    }


def _build_eval_payload(
    *,
    stage: str,
    run_id: str,
    seed: int,
    env_step: int,
    evaluation: dict[str, float],
) -> dict[str, object]:
    return {
        "stage": stage,
        "run_id": run_id,
        "seed": int(seed),
        "env_step": int(env_step),
        "step": int(env_step),
        "progress": int(env_step),
        "eval_return_mean": float(evaluation["eval_return_mean"]),
        "eval_return_std": float(evaluation["eval_return_std"]),
        "eval_return_p50": float(evaluation["eval_return_p50"]),
        "eval_return_min": float(evaluation["eval_return_min"]),
        "eval_return_max": float(evaluation["eval_return_max"]),
        "eval_episode_length_mean": float(evaluation["eval_episode_length_mean"]),
        "n_eval_episodes": int(evaluation["n_eval_episodes"]),
        "mean_reward": float(evaluation["mean_reward"]),
        "std_reward": float(evaluation["std_reward"]),
        "mean_length": float(evaluation["mean_length"]),
    }


def run_offline_pretraining(
    agent: DiscreteIQLAgent,
    offline_dataset: TransitionDataset,
    args: argparse.Namespace,
    rng: np.random.Generator,
    logger: StructuredRunLogger,
) -> None:
    updates_per_epoch = max(math.ceil(len(offline_dataset) / args.batch_size), 1)
    total_updates = int(args.offline_epochs) * updates_per_epoch
    history: list[IQLUpdateMetrics] = []

    print(
        f"  Offline dataset: {len(offline_dataset)} transitions"
        f"  obs_dim={offline_dataset.obs_dim}  act_dim={offline_dataset.act_dim}"
    )
    print(
        f"  Offline pretraining: {args.offline_epochs} epochs"
        f"  ({total_updates} gradient updates, batch_size={args.batch_size})"
    )

    for update_idx in range(1, total_updates + 1):
        batch = offline_dataset.sample(batch_size=args.batch_size, rng=rng)
        history.append(agent.update(batch))

        if update_idx % updates_per_epoch == 0:
            epoch = update_idx // updates_per_epoch
            averaged = _mean_metrics(history[-updates_per_epoch:])
            averaged["stage"] = "offline"
            averaged["run_id"] = str(args.run_id)
            averaged["seed"] = int(args.seed)
            averaged["progress"] = int(update_idx)
            averaged["epoch"] = int(epoch)
            averaged["update"] = int(update_idx)
            print(
                f"  Offline epoch {epoch:3d}/{args.offline_epochs}"
                f"  actor={averaged['actor_loss']:.4f}"
                f"  critic={averaged['critic_loss']:.4f}"
                f"  value={averaged['value_loss']:.4f}"
                f"  adv={averaged['mean_advantage']:.4f}"
            )
            logger.log(averaged)


def _priority_stats_to_dict(
    *,
    stats: PriorityRefreshStats,
    run_id: str,
    seed: int,
    update_step: int,
) -> dict[str, float | int | str]:
    return {
        "stage": "priority_refresh",
        "run_id": run_id,
        "seed": int(seed),
        "progress": int(stats.step),
        "step": int(stats.step),
        "env_step": int(stats.step),
        "update_step": int(update_step),
        "classifier_loss": float(stats.classifier_loss),
        "priority_entropy": float(stats.entropy),
        "priority_effective_sample_size": float(stats.effective_sample_size),
        "priority_mean": float(stats.mean_priority),
        "priority_std": float(stats.std_priority),
        "priority_max": float(stats.max_priority),
        "priority_p90": float(stats.p90_priority),
        "priority_top_1pct_mass": float(stats.top_1pct_mass),
        "priority_top": float(stats.top_priority),
        "online_buffer_size": int(stats.online_buffer_size),
    }


def run_online_finetuning(
    agent: DiscreteIQLAgent,
    offline_dataset: TransitionDataset,
    train_episode_bank: list[list],
    eval_episode_bank: list[list],
    args: argparse.Namespace,
    rng: np.random.Generator,
    logger: StructuredRunLogger,
    save_path: str,
) -> None:
    """Balanced dual-buffer online fine-tuning with prioritized offline replay."""
    env = build_episode_bank_env(
        episode_bank=train_episode_bank,
        n_bins=args.n_bins,
        max_queue_len=args.max_queue_len,
        invalid_action_penalty=args.invalid_action_penalty,
    )
    replay = BalancedReplayManager(
        offline_dataset=offline_dataset,
        obs_dim=offline_dataset.obs_dim,
        act_dim=offline_dataset.act_dim,
        online_buffer_size=args.online_buffer_size,
        online_sample_prob=args.online_sample_prob,
        min_online_samples=args.min_online_samples,
        priority_refresh_freq=args.priority_refresh_freq,
        priority_model_steps=args.priority_model_steps,
        priority_batch_size=args.priority_batch_size,
        priority_model_lr=args.priority_model_lr,
        priority_uniform_floor=args.priority_uniform_floor,
        priority_temperature=args.priority_temperature,
        priority_max_ratio=args.priority_max_ratio,
        device=args.device,
    )

    # Annealing schedules: relax conservatism during online training.
    offline_expectile = agent.expectile
    online_expectile = getattr(args, "online_expectile", 0.5)
    offline_temperature = agent.temperature
    online_temperature = getattr(args, "online_temperature", 1.0)
    raw_anneal = int(getattr(args, "anneal_steps", 0))
    anneal_steps = raw_anneal if raw_anneal > 0 else args.online_steps
    ucb_coef = float(getattr(args, "ucb_coef", 1.0))

    def _anneal(start: float, end: float, step: int) -> float:
        if anneal_steps <= 0:
            return end
        t = min(step / anneal_steps, 1.0)
        t = 0.5 * (1.0 - math.cos(math.pi * t))
        return start + (end - start) * t

    recent_returns: deque[float] = deque(maxlen=20)
    best_eval_reward = float("-inf")
    metric_window: list[IQLUpdateMetrics] = []
    source_counter: Counter[str] = Counter()
    episode_idx = 0

    try:
        obs, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_length = 0

        for step in range(1, args.online_steps + 1):
            agent.set_expectile(_anneal(offline_expectile, online_expectile, step))
            agent.set_temperature(_anneal(offline_temperature, online_temperature, step))

            action_mask = env.action_masks().astype(np.uint8)
            action = agent.act_ucb(obs, action_mask, ucb_coef=ucb_coef)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            replay.add_online_transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
                action_mask=action_mask,
            )

            refresh_stats = replay.maybe_refresh_priorities(step=step, rng=rng)
            if refresh_stats is not None:
                payload = _priority_stats_to_dict(
                    stats=refresh_stats,
                    run_id=str(args.run_id),
                    seed=int(args.seed),
                    update_step=int(agent.update_count),
                )
                print(
                    f"  Priority refresh step={refresh_stats.step:7d}"
                    f"  clf_loss={refresh_stats.classifier_loss:.4f}"
                    f"  ess={refresh_stats.effective_sample_size:.1f}"
                    f"  top_p={refresh_stats.top_priority:.6f}"
                )
                logger.log(payload)

            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if step >= args.start_training_after:
                for _ in range(args.updates_per_step):
                    batch, source = replay.sample(batch_size=args.batch_size, rng=rng)
                    metric_window.append(agent.update(batch))
                    source_counter[source] += 1

            if done:
                episode_idx += 1
                recent_returns.append(episode_return)
                episode_payload = {
                    "stage": "online_episode",
                    "run_id": str(args.run_id),
                    "seed": int(args.seed),
                    "episode_idx": int(episode_idx),
                    "progress": int(step),
                    "step": int(step),
                    "env_step_end": int(step),
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                    "online_buffer_size": int(len(replay.online_buffer)),
                    "recent_return_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
                    "recent_return_std": float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0,
                    "online_return_recent_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
                    "online_return_recent_std": float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0,
                    "expectile": float(agent.expectile),
                    "temperature": float(agent.temperature),
                    "success": bool(terminated and not truncated),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                logger.log(episode_payload)
                print(
                    f"  Online episode finished"
                    f"  step={step:7d}/{args.online_steps}"
                    f"  return={episode_return:10.2f}"
                    f"  len={episode_length:4d}"
                    f"  online_buf={len(replay.online_buffer):6d}"
                )
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            if step % args.log_interval == 0 and metric_window:
                window_size = min(len(metric_window), args.log_interval * args.updates_per_step)
                averaged = _mean_metrics(metric_window[-window_size:])
                offline_updates = int(source_counter.get("offline", 0))
                online_updates = int(source_counter.get("online", 0))
                total_updates = max(offline_updates + online_updates, 1)
                averaged.update(
                    {
                        "stage": "online_train",
                        "run_id": str(args.run_id),
                        "seed": int(args.seed),
                        "progress": int(step),
                        "step": int(step),
                        "env_step": int(step),
                        "update_step": int(agent.update_count),
                        "episode_idx": int(episode_idx + 1),
                        "online_buffer_size": int(len(replay.online_buffer)),
                        "recent_return_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
                        "recent_return_std": float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0,
                        "online_return_recent_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
                        "online_return_recent_std": float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0,
                        "offline_updates": offline_updates,
                        "online_updates": online_updates,
                        "offline_sample_ratio": float(offline_updates / total_updates),
                        "online_sample_ratio": float(online_updates / total_updates),
                        "replay_size_online": int(len(replay.online_buffer)),
                        "expectile": float(agent.expectile),
                        "temperature": float(agent.temperature),
                    }
                )
                print(
                    f"  Online train step {step:7d}/{args.online_steps}"
                    f"  actor={averaged['actor_loss']:.4f}"
                    f"  critic={averaged['critic_loss']:.4f}"
                    f"  value={averaged['value_loss']:.4f}"
                    f"  recent_return={averaged['recent_return_mean']:.2f}"
                    f"  expectile={agent.expectile:.3f}"
                    f"  temperature={agent.temperature:.2f}"
                    f"  src=off:{averaged['offline_updates']} on:{averaged['online_updates']}"
                )
                logger.log(averaged)
                source_counter.clear()

            if step % args.eval_freq == 0:
                evaluation = evaluate_agent(
                    agent=agent,
                    episode_bank=eval_episode_bank,
                    n_eval_episodes=args.n_eval_episodes,
                    n_bins=args.n_bins,
                    max_queue_len=args.max_queue_len,
                    seed=args.seed + 10_000 + step,
                )
                evaluation_payload = _build_eval_payload(
                    stage="eval",
                    run_id=str(args.run_id),
                    seed=int(args.seed),
                    env_step=int(step),
                    evaluation=evaluation,
                )
                print(
                    f"  Eval step={step:7d}"
                    f"  mean_reward={evaluation_payload['eval_return_mean']:.2f}"
                    f"  std={evaluation_payload['eval_return_std']:.2f}"
                )
                logger.log(evaluation_payload)
                if float(evaluation_payload["eval_return_mean"]) > best_eval_reward:
                    best_eval_reward = float(evaluation_payload["eval_return_mean"])
                    best_path = os.path.join(save_path, "o2o_iql_best.pt")
                    agent.save(best_path, extra_state={"best_eval": evaluation_payload, "step": step})
                    print(f"  Saved new best checkpoint -> {best_path}")

            if step % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(save_path, f"o2o_iql_step_{step}.pt")
                agent.save(checkpoint_path, extra_state={"step": step})
                print(f"  Saved checkpoint -> {checkpoint_path}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced dual-buffer offline-to-online IQL trainer for EV charging"
    )

    parser.add_argument("--demand_dir", type=str, default="data/bc_dataset/demand")
    parser.add_argument("--solution_dir", type=str, default="data/bc_dataset/solutions")
    parser.add_argument("--train_data_dir", type=str, default="data/train_dataset/bias")
    parser.add_argument("--eval_data_dir", type=str, default="")

    parser.add_argument("--n_bins", type=int, default=21)
    parser.add_argument("--max_queue_len", type=int, default=10)
    parser.add_argument("--invalid_action_penalty", type=float, default=0.0)

    parser.add_argument("--offline_epochs", type=int, default=100)
    parser.add_argument("--online_steps", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--target_update_rate", type=float, default=5e-3)
    parser.add_argument("--exp_adv_max", type=float, default=100.0)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--online_buffer_size", type=int, default=100_000)
    parser.add_argument("--online_sample_prob", type=float, default=0.6)
    parser.add_argument("--min_online_samples", type=int, default=2_000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--start_training_after", type=int, default=1)
    parser.add_argument(
        "--exploration_epsilon",
        type=float,
        default=0.05,
        help="(deprecated, kept for compat) epsilon-greedy not used with UCB",
    )
    parser.add_argument(
        "--ucb_coef",
        type=float,
        default=1.0,
        help="UCB exploration coefficient: a = argmax [Q_mean + coef * Q_std]",
    )
    parser.add_argument(
        "--online_expectile",
        type=float,
        default=0.6,
        help="Target expectile after annealing (0.5 = no conservatism)",
    )
    parser.add_argument(
        "--online_temperature",
        type=float,
        default=1.0,
        help="Target AWR temperature after annealing",
    )
    parser.add_argument(
        "--anneal_steps",
        type=int,
        default=0,
        help="Steps to anneal expectile/temperature (0 = 30%% of online_steps)",
    )

    parser.add_argument("--priority_refresh_freq", type=int, default=5_000)
    parser.add_argument("--priority_model_steps", type=int, default=100)
    parser.add_argument("--priority_batch_size", type=int, default=512)
    parser.add_argument("--priority_model_lr", type=float, default=1e-3)
    parser.add_argument("--priority_uniform_floor", type=float, default=0.05)
    parser.add_argument("--priority_temperature", type=float, default=1.0)
    parser.add_argument("--priority_max_ratio", type=float, default=50.0)

    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--checkpoint_freq", type=int, default=50_000)
    parser.add_argument("--log_interval", type=int, default=10_000)

    parser.add_argument("--offline_limit_episodes", type=int, default=0)
    parser.add_argument("--train_limit_episodes", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--offline_dataset_cache", type=str, default="")
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--save_path", type=str, default="train/o2o_iql/checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/o2o_iql")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--algo", type=str, default="full_o2o_iql")
    parser.add_argument("--env_name", type=str, default="multi_station_ev_charging")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = _resolve_device(args.device)
    if not args.run_id:
        args.run_id = _default_run_id(args)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _ensure_dir(args.save_path)
    _ensure_dir(args.log_dir)
    metrics_path = os.path.join(args.log_dir, "metrics.jsonl")
    logger = StructuredRunLogger(log_dir=args.log_dir, metrics_path=metrics_path)
    logger.initialize()
    logger.write_run_config(_build_run_config_payload(args))

    print("=== O2O IQL with Balanced Dual Buffers ===")
    print(f"  device={args.device}")
    print(f"  run_id={args.run_id}")
    print(f"  save_path={args.save_path}")
    print(f"  log_dir={args.log_dir}")
    print(f"  eval_freq={args.eval_freq}")

    if args.offline_dataset_cache and Path(args.offline_dataset_cache).exists():
        print(f"  Loading cached offline dataset from '{args.offline_dataset_cache}'")
        offline_dataset = TransitionDataset.load(args.offline_dataset_cache)
    else:
        offline_dataset = load_offline_dataset(
            demand_dir=args.demand_dir,
            solution_dir=args.solution_dir,
            n_bins=args.n_bins,
            max_queue_len=args.max_queue_len,
            seed=args.seed,
            limit_episodes=args.offline_limit_episodes,
        )
        if args.offline_dataset_cache:
            cache_path = Path(args.offline_dataset_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            offline_dataset.save(cache_path)
            print(f"  Cached offline dataset -> {cache_path}")

    if args.pretrained_checkpoint:
        print(f"  Loading pretrained O2O-IQL checkpoint from '{args.pretrained_checkpoint}'")
        agent = DiscreteIQLAgent.load(args.pretrained_checkpoint, device=args.device)
    else:
        agent = DiscreteIQLAgent(
            obs_dim=offline_dataset.obs_dim,
            act_dim=offline_dataset.act_dim,
            hidden_dims=(args.hidden_dim, args.hidden_dim),
            learning_rate=args.learning_rate,
            discount=args.discount,
            expectile=args.expectile,
            temperature=args.temperature,
            target_update_rate=args.target_update_rate,
            exp_adv_max=args.exp_adv_max,
            device=args.device,
        )
        run_offline_pretraining(
            agent=agent,
            offline_dataset=offline_dataset,
            args=args,
            rng=rng,
            logger=logger,
        )
        offline_ckpt = os.path.join(args.save_path, "offline_iql.pt")
        agent.save(offline_ckpt, extra_state={"stage": "offline"})
        print(f"  Saved offline checkpoint -> {offline_ckpt}")

    eval_dir = args.eval_data_dir or args.train_data_dir
    eval_episode_bank = load_episode_bank_from_dir(
        eval_dir,
        limit=max(args.n_eval_episodes, 1),
    )
    offline_eval = evaluate_agent(
        agent=agent,
        episode_bank=eval_episode_bank,
        n_eval_episodes=args.n_eval_episodes,
        n_bins=args.n_bins,
        max_queue_len=args.max_queue_len,
        seed=args.seed + 5_000,
    )
    offline_eval_payload = _build_eval_payload(
        stage="offline_eval",
        run_id=str(args.run_id),
        seed=int(args.seed),
        env_step=0,
        evaluation=offline_eval,
    )
    print(
        f"  Offline eval mean_reward={offline_eval_payload['eval_return_mean']:.2f}"
        f"  std={offline_eval_payload['eval_return_std']:.2f}"
    )
    logger.log(offline_eval_payload)

    if args.online_steps > 0:
        train_episode_bank = load_episode_bank_from_dir(
            args.train_data_dir,
            limit=args.train_limit_episodes,
        )
        run_online_finetuning(
            agent=agent,
            offline_dataset=offline_dataset,
            train_episode_bank=train_episode_bank,
            eval_episode_bank=eval_episode_bank,
            args=args,
            rng=rng,
            logger=logger,
            save_path=args.save_path,
        )

    final_path = os.path.join(
        args.save_path,
        "o2o_iql_final.pt" if args.online_steps > 0 else "offline_iql_final.pt",
    )
    agent.save(
        final_path,
        extra_state={
            "stage": "online" if args.online_steps > 0 else "offline",
            "offline_eval": offline_eval_payload,
        },
    )
    print(f"  Saved final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
