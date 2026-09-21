"""Bounded causal sequence batches for future replay learners; no ML dependency."""

from dataclasses import dataclass

import numpy as np

from .evaluation_contract import Decision
from .policy_benchmark import validate_plan
from .replay_env import InterfaceConfig, ReplayEnv
from .replay_features import FEATURE_NAMES


@dataclass(frozen=True)
class BatchConfig:
    batch_size: int = 64
    history_steps: int = 8
    time_bins: int = 16

    def __post_init__(self):
        if any(
            type(v) is not int or v < 1
            for v in (self.batch_size, self.history_steps, self.time_bins)
        ):
            raise ValueError("batch size, history and time bins must be positive integers")


class HistoryWindow:
    """Reusable fixed history; same zero-padded context at collection and inference."""

    def __init__(self, length, features):
        self.values = np.zeros((length, features), dtype=np.float32)

    def append(self, observation):
        if observation.shape != self.values.shape[1:] or not np.isfinite(observation).all():
            raise ValueError("invalid observation shape or values")
        self.values[:-1] = self.values[1:]
        self.values[-1] = observation
        return self.values


class UniformActionPolicy:
    """Exploratory collector covering every band and dwell choice."""

    def reset(self, specification, seed):
        self.actions = specification["action_count"]
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return int(self.rng.integers(self.actions))


def training_batches(root, plan, behavior_factory, receiver, interface=None, config=None):
    """Yield borrowed batches; consume or copy before requesting the next batch.

    Only a validated training manifest is accepted. Truth labels are constructed
    AFTER actions and never sent to behavior policies. Input history excludes the
    result of its labeled action. Memory is O(batch_size * history * features).
    The caller must close this generator if stopping early (contextlib.closing).
    """
    from dataclasses import replace

    if plan.get("split") != "train":
        raise ValueError("training requires a train-only manifest")
    paths = validate_plan(root, plan)
    cfg, interface = config or BatchConfig(), interface or InterfaceConfig()
    features = interface.bands * len(FEATURE_NAMES) + 2
    arrays = {
        "history": np.empty((cfg.batch_size, cfg.history_steps, features), np.float32),
        "action": np.empty(cfg.batch_size, np.int64),
        "time_class": np.empty(cfg.batch_size, np.int64),
        "ratio": np.empty(cfg.batch_size, np.float32),
        "ratio_valid": np.empty(cfg.batch_size, np.bool_),
        "reward": np.empty(cfg.batch_size, np.float32),
        "discount": np.empty(cfg.batch_size, np.float32),
        "elapsed_seconds": np.empty(cfg.batch_size, np.float32),
        "terminated": np.empty(cfg.batch_size, np.bool_),
    }
    size = 0
    for path in paths:
        for seed in plan["seeds"]:
            with ReplayEnv(path, replace(receiver, seed=seed), interface) as env:
                observation = env.reset()
                behavior = behavior_factory()
                behavior.reset(env.specification(), seed)
                history = HistoryWindow(cfg.history_steps, features)
                while True:
                    arrays["history"][size] = history.append(observation)
                    decision = behavior.act(observation)
                    action = decision.action if isinstance(decision, Decision) else decision
                    transition = env.step(action)
                    outcome = env.evaluation_outcome()
                    delay, elapsed = outcome.first_intercept_seconds, outcome.elapsed_seconds
                    arrays["action"][size] = action
                    arrays["time_class"][size] = (
                        cfg.time_bins
                        if delay is None
                        else min(cfg.time_bins - 1, int(delay / elapsed * cfg.time_bins))
                    )
                    arrays["ratio"][size] = outcome.captured_count / max(1, outcome.truth_count)
                    arrays["ratio_valid"][size] = outcome.truth_count > 0
                    arrays["reward"][size] = transition.reward
                    arrays["discount"][size] = transition.discount
                    arrays["elapsed_seconds"][size] = elapsed
                    arrays["terminated"][size] = transition.terminated
                    size += 1
                    observation = transition.observation
                    if size == cfg.batch_size:
                        yield {
                            **{key: value[:size] for key, value in arrays.items()},
                            "time_bins": cfg.time_bins,
                        }
                        size = 0
                    if transition.terminated:
                        break
    if size:
        yield {**{key: value[:size] for key, value in arrays.items()}, "time_bins": cfg.time_bins}
    validate_plan(root, plan)
