"""Model dimensions, search defaults and training configuration."""

from __future__ import annotations

from dataclasses import dataclass

from spectra_scheduler.rl import RewardConfig

MAX_BANDS = 8
GRU_HIDDEN = 128
STEP_FEATURE_DIM = MAX_BANDS + 5 + 2 * MAX_BANDS
DEFAULT_MCTS_SIMS = 50
DEFAULT_GAMMA = 0.97
DEFAULT_COVERAGE_LIMIT = 20
MODEL_VERSION = 3
VERSION = 3
REWARD = RewardConfig(coverage=0.2)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for world-model pre-training."""

    demo_episodes_per_policy: int = 200
    demo_policies: tuple[str, ...] = (
        "adaptive-dwell",
        "track-aware",
        "transition-band",
        "bayesian-band",
        "period-aware",
    )
    epochs: int = 50
    batch_size: int = 8  # episodes per gradient step
    lr: float = 3e-4
    gamma: float = DEFAULT_GAMMA
    unroll_steps: int = 5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    dynamics_loss_weight: float = 0.5
    reward_loss_weight: float = 0.5
    device: str = "cpu"
    seed: int = 0


@dataclass(frozen=True)
class Config:
    iterations: int = 200
    episodes: int = 24
    workers: int = 6
    batch_size: int = 32
    updates: int = 100
    replay_capacity: int = 1000
    simulations: int = 32
    depth: int = 5
    unroll: int = 5
    td_steps: int = 10
    gamma: float = 0.97
    lr: float = 3e-4
    ema: float = 0.99
    seed: int = 0
    validation_episodes: int = 8
    validation_every: int = 5
    threads: int = 2
    device: str = "cpu"
    exploration_hold: int = 4
    exploration_decay_iterations: int = 50
    final_temperature: float = 0.25
    normalize_search: bool = True
    observation_loss_weight: float = 1.0

    def __post_init__(self):
        for name in (
            "iterations",
            "episodes",
            "workers",
            "batch_size",
            "updates",
            "replay_capacity",
            "simulations",
            "depth",
            "unroll",
            "td_steps",
            "validation_episodes",
            "validation_every",
            "threads",
            "exploration_hold",
            "exploration_decay_iterations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.gamma < 1 or not 0 <= self.ema < 1 or not 0 < self.lr < 1:
            raise ValueError("invalid discount, EMA or learning rate")
        if self.seed < 0 or self.device not in ("cpu", "cuda"):
            raise ValueError("invalid seed or device")
        if not 0 < self.final_temperature <= 1 or not 0 <= self.observation_loss_weight <= 10:
            raise ValueError("invalid exploration temperature or observation loss weight")


def saved_config(settings):
    """Old inference artifacts retain their original search/exploration settings."""
    defaults = dict(
        exploration_hold=1,
        exploration_decay_iterations=50,
        final_temperature=1.0,
        normalize_search=False,
        observation_loss_weight=0.0,
    )
    return Config(**{**defaults, **settings})
