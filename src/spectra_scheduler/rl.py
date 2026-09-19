"""Shared-action neural Double Q-learning using NumPy and observation-only reward.

Every band remains selectable on every step. No policy fallback, minimum dwell,
forced sweep, or overdue-band override is used by this agent.
"""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from spectra_scheduler.learned_scheduler import FEATURE_NAMES, HISTORY_STEPS, ObservationHistory
from spectra_scheduler.rl_scenarios import GENERATOR_VERSION, procedural_scenario

RL_FEATURES = (
    *FEATURE_NAMES,
    "previous_step_listening",
    *(f"recent_{i}_{part}" for i in range(4) for part in ("hit", "age")),
)


class Context:
    def __init__(self, num_bands):
        self.history = ObservationHistory(num_bands)
        self.listening = True

    def encode(self, step):
        rows = []
        for band in range(self.history.num_bands):
            recent = list(self.history.recent[band])[-4:][::-1]
            temporal = []
            for i in range(4):
                temporal.extend(
                    (recent[i][1], min(step - recent[i][0], 24) / 24)
                    if i < len(recent)
                    else (0.0, 1.0)
                )
            rows.append((*self.history.features(step, band), float(self.listening), *temporal))
        return np.asarray(rows, dtype=np.float32)

    def observe(self, observation):
        self.history.update(observation)
        self.listening = observation.listening

    def coverage_debt(self):
        t = self.history.last_time
        return float(
            np.mean(
                [
                    min(t - last if last >= 0 else t + 1, 24) / 24
                    for last in self.history.last_listen
                ]
            )
        )


@dataclass(frozen=True)
class RewardConfig:
    hit: float = 1.0
    retuning: float = 0.05
    coverage: float = 0.05

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("reward weights must be finite and nonnegative")

    def compute(self, observation, context):
        return (
            self.hit * float(observation.hit)
            - self.retuning * float(not observation.listening)
            - self.coverage * context.coverage_debt()
        )


class QNetwork:
    """One ReLU hidden layer, shared weights score each candidate band's context."""

    def __init__(self, seed=0, hidden=64):
        if type(hidden) is not int or not 1 <= hidden <= 1024:
            raise ValueError("hidden width must be in 1..1024")
        rng = np.random.default_rng(seed)
        self.parameters = [
            rng.normal(0, np.sqrt(2 / len(RL_FEATURES)), (len(RL_FEATURES), hidden)),
            np.zeros(hidden),
            rng.normal(0, 0.01, (hidden, 1)),
            np.zeros(1),
        ]

    def predict(self, x):
        w, b, v, c = self.parameters
        return (np.maximum(x @ w + b, 0) @ v + c)[..., 0]

    def loss_gradients(self, x, targets):
        w, b, v, c = self.parameters
        pre = x @ w + b
        hidden = np.maximum(pre, 0)
        residual = (hidden @ v + c)[:, 0] - targets
        loss = np.where(abs(residual) <= 1, 0.5 * residual**2, abs(residual) - 0.5).mean()
        grad = np.clip(residual, -1, 1)[:, None] / len(x)
        back = (grad @ v.T) * (pre > 0)
        return float(loss), [x.T @ back, back.sum(axis=0), hidden.T @ grad, grad.sum(axis=0)]

    def copy(self):
        result = QNetwork(hidden=self.parameters[0].shape[1])
        result.parameters = [a.copy() for a in self.parameters]
        return result


class Adam:
    def __init__(self, network, rate):
        self.network, self.rate, self.step = network, rate, 0
        self.m = [np.zeros_like(a) for a in network.parameters]
        self.v = [np.zeros_like(a) for a in network.parameters]

    def update(self, gradients):
        self.step += 1
        norm = np.sqrt(sum(np.sum(g * g) for g in gradients))
        for p, m, v, g in zip(self.network.parameters, self.m, self.v, gradients, strict=True):
            g = g * min(1.0, 10 / max(float(norm), 1e-12))
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= (
                self.rate
                * (m / (1 - 0.9**self.step))
                / (np.sqrt(v / (1 - 0.999**self.step)) + 1e-8)
            )


class Replay:
    def __init__(self, capacity, bands):
        if capacity < 1 or bands < 1:
            raise ValueError("invalid replay dimensions")
        self.states = np.empty((capacity, bands, len(RL_FEATURES)), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=bool)
        self.count = 0

    def add(self, state, action, reward, next_state, done):
        index = self.count % len(self.actions)
        self.states[index], self.next_states[index] = state, next_state
        self.actions[index], self.rewards[index], self.done[index] = action, reward, done
        self.count += 1

    def sample(self, size, rng):
        indices = rng.integers(min(self.count, len(self.actions)), size=size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.done[indices],
        )


def double_q_targets(online, target, next_states, rewards, done, gamma):
    choices = online.predict(next_states).argmax(axis=1)
    values = target.predict(next_states)[np.arange(len(choices)), choices]
    return rewards + gamma * (~done) * values


class RLScheduler:
    def __init__(self, network, reward=None):
        self.network = network
        self.reward = reward or RewardConfig()

    def reset(self, num_bands):
        self.context = Context(num_bands)
        self.total_reward = 0.0
        self.pending = None

    def choose_band(self, time_step):
        if self.pending is not None:
            raise ValueError("previous action has no observation")
        self.state = self.context.encode(time_step)
        self.action = int(self.network.predict(self.state).argmax())
        self.pending = time_step, self.action
        return self.action

    def observe(self, observation):
        if self.pending != (observation.time_step, observation.band):
            raise ValueError("observation does not match action")
        self.context.observe(observation)
        self.last_reward = self.reward.compute(observation, self.context)
        self.total_reward += self.last_reward
        self.pending = None


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 1500
    seed: int = 0
    hidden: int = 64
    replay_capacity: int = 10000
    batch_size: int = 64
    warmup: int = 512
    update_every: int = 4
    target_every: int = 250
    gamma: float = 0.97
    learning_rate: float = 0.001
    epsilon_final: float = 0.08

    def __post_init__(self):
        for key in (
            "episodes",
            "hidden",
            "replay_capacity",
            "batch_size",
            "warmup",
            "update_every",
            "target_every",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.replay_capacity < max(self.batch_size, self.warmup):
            raise ValueError("replay capacity must cover batch size and warmup")
        if not 0 <= self.gamma < 1 or not 0 <= self.epsilon_final <= 1:
            raise ValueError("invalid discount or epsilon")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


class TrainingScheduler(RLScheduler):
    def __init__(self, network, target, replay, optimizer, rng, config, reward):
        super().__init__(network, reward)
        self.target, self.replay, self.optimizer = target, replay, optimizer
        self.rng, self.config = rng, config
        self.steps, self.losses = 0, []
        self.epsilon, self.duration = 1.0, 120

    def choose_band(self, time_step):
        action = super().choose_band(time_step)
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(len(self.state)))
        self.action, self.pending = action, (time_step, action)
        return action

    def observe(self, observation):
        super().observe(observation)
        done = observation.time_step == self.duration - 1
        self.replay.add(
            self.state,
            self.action,
            self.last_reward,
            self.context.encode(observation.time_step + 1),
            done,
        )
        self.steps += 1
        cfg = self.config
        if self.steps >= cfg.warmup and self.steps % cfg.update_every == 0:
            states, actions, rewards, next_states, terminals = self.replay.sample(
                cfg.batch_size, self.rng
            )
            targets = double_q_targets(
                self.network, self.target, next_states, rewards, terminals, cfg.gamma
            )
            loss, gradients = self.network.loss_gradients(
                states[np.arange(len(actions)), actions], targets
            )
            self.optimizer.update(gradients)
            self.losses.append(loss)
        if self.steps % cfg.target_every == 0:
            self.target.parameters = [a.copy() for a in self.network.parameters]


@dataclass
class RLModel:
    network: QNetwork
    training: dict
    reward: RewardConfig

    def to_dict(self):
        return {
            "schema_version": 1,
            "algorithm": "shared-action-double-dqn",
            "features": list(RL_FEATURES),
            "history_steps": HISTORY_STEPS,
            "weights": [a.tolist() for a in self.network.parameters],
            "training": self.training,
            "reward": asdict(self.reward),
        }

    @property
    def fingerprint(self):
        return sha256(
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @classmethod
    def load(cls, path):
        try:
            data = json.loads(Path(path).read_text())
            if (
                data["schema_version"] != 1
                or data["algorithm"] != "shared-action-double-dqn"
                or data["features"] != list(RL_FEATURES)
                or data["history_steps"] != HISTORY_STEPS
            ):
                raise ValueError("unsupported RL model schema")
            config = TrainConfig(**data["training"]["config"])
            if data["training"]["generator_version"] != GENERATOR_VERSION:
                raise ValueError("unsupported training world generator")
            if data["training"]["split"] != "train" or data["training"]["world_seeds"] != [
                0,
                config.episodes - 1,
            ]:
                raise ValueError("invalid training split provenance")
            network = QNetwork(config.seed, config.hidden)
            weights = [np.asarray(a, dtype=float) for a in data["weights"]]
            if len(weights) != 4 or any(
                a.shape != b.shape or not np.isfinite(a).all()
                for a, b in zip(weights, network.parameters, strict=True)
            ):
                raise ValueError("invalid network weights")
            network.parameters = weights
            return cls(network, data["training"], RewardConfig(**data["reward"]))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid RL artifact: {error}") from error


def train(config=None, reward=None, progress=None):
    config = config or TrainConfig()
    reward = reward or RewardConfig()
    network = QNetwork(config.seed, config.hidden)
    agent = TrainingScheduler(
        network,
        network.copy(),
        Replay(config.replay_capacity, 6),
        Adam(network, config.learning_rate),
        np.random.default_rng(config.seed + 1),
        config,
        reward,
    )
    curve = []
    returns = []
    for episode in range(config.episodes):
        simulation = procedural_scenario(episode, "train")
        agent.duration = simulation.duration
        agent.epsilon = max(
            config.epsilon_final,
            1 - (1 - config.epsilon_final) * episode / max(1, config.episodes * 0.7),
        )
        simulation.run(agent)
        returns.append(agent.total_reward / simulation.duration)
        if (episode + 1) % 100 == 0 or episode + 1 == config.episodes:
            row = {
                "episode": episode + 1,
                "environment_steps": agent.steps,
                "epsilon": agent.epsilon,
                "mean_training_reward_per_step": float(np.mean(returns)),
                "mean_td_loss": float(np.mean(agent.losses)) if agent.losses else None,
            }
            curve.append(row)
            returns.clear()
            agent.losses.clear()
            if progress:
                progress(row)
    if not all(np.isfinite(a).all() for a in network.parameters):
        raise ValueError("training diverged")
    return RLModel(
        network,
        {
            "config": asdict(config),
            "generator_version": GENERATOR_VERSION,
            "split": "train",
            "world_seeds": [0, config.episodes - 1],
            "numpy_version": np.__version__,
            "curve": curve,
            "checkpoint_selection": "fixed final episode; no validation selection",
        },
        reward,
    )
