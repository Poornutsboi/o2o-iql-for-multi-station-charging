"""Offline transition collection helpers for discrete masked IQL."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.env_data import CAPACITY, MIN_SEG, TRAVEL_MATRIX
from envs.charging_env import EpisodeBankChargingEnv, travel_time_fn_from_matrix
from envs.maskable_actions import no_split_action_int
from simulator.orchestrator import load_demand_vehicles_from_csv
from train.finetune.ppo_trainer import FlatObsWrapper
from train.imitation.bc_trainer import load_paired_dataset, solution_to_action


@dataclass(slots=True)
class TransitionDataset:
    """Numpy-backed transition dataset used by offline and mixed replay updates."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    action_masks: np.ndarray

    def __len__(self) -> int:
        return int(len(self.observations))

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def act_dim(self) -> int:
        return int(self.action_masks.shape[1])

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty transition dataset.")
        idx = rng.integers(0, len(self), size=int(batch_size))
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones": self.dones[idx],
            "action_masks": self.action_masks[idx],
        }

    def compute_obs_stats(self, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
        """Per-dim mean/std of observations; std<eps is clamped to 1.0.

        Used to install frozen z-score buffers on the IQL agent so the network
        sees inputs with comparable scale across the heterogeneous obs vector.
        Stats are computed once on the offline transition buffer and reused
        unchanged through the entire offline + online phase.
        """
        if len(self) == 0:
            return (
                np.zeros(self.obs_dim, dtype=np.float32),
                np.ones(self.obs_dim, dtype=np.float32),
            )
        mean = self.observations.mean(axis=0).astype(np.float32)
        std = self.observations.std(axis=0).astype(np.float32)
        std = np.where(std < float(eps), np.float32(1.0), std)
        return mean, std

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            next_observations=self.next_observations,
            dones=self.dones,
            action_masks=self.action_masks,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransitionDataset":
        payload = np.load(path)
        return cls(
            observations=payload["observations"].astype(np.float32),
            actions=payload["actions"].astype(np.int64),
            rewards=payload["rewards"].astype(np.float32),
            next_observations=payload["next_observations"].astype(np.float32),
            dones=payload["dones"].astype(np.float32),
            action_masks=payload["action_masks"].astype(np.uint8),
        )


def _build_wrapped_episode_env(
    episode_bank: list[list],
    n_bins: int,
    max_queue_len: int,
    invalid_action_penalty: float = 0.0,
    reward_normalize_by: str = "none",
) -> FlatObsWrapper:
    """Create a flat-observation env without SB3 wrappers for direct control."""
    env = EpisodeBankChargingEnv(
        episode_bank=episode_bank,
        station_capacities=CAPACITY.tolist(),
        travel_time_fn=travel_time_fn_from_matrix(TRAVEL_MATRIX),
        min_first_charge=float(MIN_SEG),
        min_second_charge=float(MIN_SEG),
        n_bins=int(n_bins),
        invalid_action_penalty=float(invalid_action_penalty),
        reward_normalize_by=str(reward_normalize_by),
    )
    return FlatObsWrapper(env, max_queue_len=max_queue_len)


def build_single_episode_env(
    vehicles: list,
    n_bins: int = 21,
    max_queue_len: int = 10,
    invalid_action_penalty: float = 0.0,
    reward_normalize_by: str = "none",
) -> FlatObsWrapper:
    """Create a deterministic one-episode env for fixed evaluation or rollout."""
    return _build_wrapped_episode_env(
        episode_bank=[vehicles],
        n_bins=n_bins,
        max_queue_len=max_queue_len,
        invalid_action_penalty=invalid_action_penalty,
        reward_normalize_by=reward_normalize_by,
    )


def build_episode_bank_env(
    episode_bank: list[list],
    n_bins: int = 21,
    max_queue_len: int = 10,
    invalid_action_penalty: float = 0.0,
    reward_normalize_by: str = "none",
) -> FlatObsWrapper:
    """Create a stochastic env that samples episodes from the supplied bank."""
    return _build_wrapped_episode_env(
        episode_bank=episode_bank,
        n_bins=n_bins,
        max_queue_len=max_queue_len,
        invalid_action_penalty=invalid_action_penalty,
        reward_normalize_by=reward_normalize_by,
    )


def load_episode_bank_from_dir(data_dir: str, limit: int = 0) -> list[list]:
    """Load a sorted episode bank from a directory of CSV demand files."""
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if limit > 0:
        csv_files = csv_files[: int(limit)]
    if not csv_files:
        raise FileNotFoundError(f"No CSV episode files found in '{data_dir}'")
    episodes = [load_demand_vehicles_from_csv(str(path)) for path in csv_files]
    print(f"  Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


def collect_expert_transitions(
    paired_data: list[tuple[list, dict[int, dict]]],
    n_bins: int = 21,
    max_queue_len: int = 10,
    seed: int = 0,
    reward_normalize_by: str = "none",
) -> TransitionDataset:
    """Roll out expert actions episode by episode and collect full transitions.

    Each pair is rolled out inside its own one-episode env to keep the solution
    and demand trajectory aligned exactly.
    """
    num_stations = int(len(CAPACITY))
    no_split = no_split_action_int(n_bins=n_bins, num_stations=num_stations)

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[float] = []
    action_masks: list[np.ndarray] = []
    invalid_fallbacks = 0

    for episode_idx, (vehicles, solution) in enumerate(paired_data):
        env = build_single_episode_env(
            vehicles=vehicles,
            n_bins=n_bins,
            max_queue_len=max_queue_len,
            reward_normalize_by=reward_normalize_by,
        )
        try:
            obs, _ = env.reset(seed=seed + episode_idx)
            base_env = env.env

            while True:
                pending = base_env.pending_vehicle
                if pending is None:
                    break

                mask = env.action_masks().astype(np.uint8)
                action = solution_to_action(
                    vehicle_id=int(pending.vid),
                    route=list(pending.route),
                    required_charge_time=float(pending.duration),
                    solution=solution,
                    n_bins=n_bins,
                    num_stations=num_stations,
                )
                if mask[action] == 0:
                    action = no_split
                    invalid_fallbacks += 1

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)

                observations.append(obs.copy())
                actions.append(int(action))
                rewards.append(float(reward))
                next_observations.append(next_obs.copy())
                dones.append(float(done))
                action_masks.append(mask.copy())

                obs = next_obs
                if done:
                    break
        finally:
            env.close()

    dataset = TransitionDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        action_masks=np.asarray(action_masks, dtype=np.uint8),
    )
    split_rate = float((dataset.actions != no_split).mean()) if len(dataset) > 0 else 0.0
    print(
        f"  Collected {len(dataset)} expert transitions"
        f"  (split-action rate={split_rate:.1%}, invalid fallbacks={invalid_fallbacks})"
    )
    return dataset


def load_offline_dataset(
    demand_dir: str,
    solution_dir: str,
    n_bins: int = 21,
    max_queue_len: int = 10,
    seed: int = 0,
    limit_episodes: int = 0,
    reward_normalize_by: str = "none",
) -> TransitionDataset:
    """Load paired expert data and turn it into a transition dataset."""
    paired_data = load_paired_dataset(
        demand_dir=demand_dir,
        solution_dir=solution_dir,
    )
    if limit_episodes > 0:
        paired_data = paired_data[: int(limit_episodes)]
        print(f"  Using first {len(paired_data)} paired offline episodes")
    return collect_expert_transitions(
        paired_data=paired_data,
        n_bins=n_bins,
        max_queue_len=max_queue_len,
        seed=seed,
        reward_normalize_by=reward_normalize_by,
    )
