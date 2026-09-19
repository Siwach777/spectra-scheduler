"""Gymnasium adapter using exactly the same receiver engine as baseline runs."""

import gymnasium as gym
import numpy as np

from spectra_scheduler.rl import Context, RewardConfig
from spectra_scheduler.rl_scenarios import procedural_scenario
from spectra_scheduler.simulation import SimulationEpisode

MAX_BANDS = 8
OBSERVATION_VERSION = 1


def encode_context(context, step):
    # Includes a validity mask; no emitter truth or simulator clock as an input.
    state = context.encode(step)
    if len(state) > MAX_BANDS:
        raise ValueError("recurrent policy supports at most eight bands")
    padded = np.zeros((MAX_BANDS, state.shape[1]), dtype=np.float32)
    padded[: len(state)] = state
    mask = np.zeros(MAX_BANDS, dtype=np.float32)
    mask[: len(state)] = 1
    return np.concatenate((padded.ravel(), mask))


class SpectrumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, split="train", worker=0, workers=1, start_episode=0, reward=None, shifted=False
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError("invalid split")
        self.split, self.worker, self.workers = split, worker, workers
        self.episode_index = start_episode
        self.start_episode = start_episode
        self.reward = reward or RewardConfig(coverage=0.2)
        self.shifted = shifted
        self.action_space = gym.spaces.Discrete(MAX_BANDS)
        self.observation_space = gym.spaces.Box(
            0, 1, shape=encode_context(Context(8), 0).shape, dtype=np.float32
        )
        self.episode = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.episode_index = self.start_episode + seed
        world_seed = self.episode_index * self.workers + self.worker
        self.episode_index += 1
        simulation = procedural_scenario(world_seed, self.split, self.shifted, num_bands=8)
        self.episode = SimulationEpisode(simulation)
        self.context = Context(8)
        return encode_context(self.context, 0), {}

    def step(self, action):
        if self.episode is None:
            raise RuntimeError("reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError("invalid band action")
        observation = self.episode.step(int(action))
        self.context.observe(observation)
        reward = self.reward.compute(observation, self.context)
        ended = self.episode.time_step == self.episode.simulation.duration
        # Finite episode is a task boundary: never bootstrap into a new random world.
        return encode_context(self.context, self.episode.time_step), reward, ended, False, {}


class RecurrentScheduler:
    """Frozen recurrent inference with per-episode memory reset, no scan overrides."""

    def __init__(self, model):
        self.model = model

    def reset(self, num_bands):
        if not 1 <= num_bands <= MAX_BANDS:
            raise ValueError("recurrent policy supports one to eight bands")
        import torch

        self.context = Context(num_bands)
        shape = self.model.policy.lstm_hidden_state_shape
        self.state = (
            torch.zeros(shape, device=self.model.device),
            torch.zeros(shape, device=self.model.device),
        )
        self.episode_start = True
        self.pending = None

    def choose_band(self, time_step):
        import torch

        if self.pending is not None:
            raise ValueError("previous action has no observation")
        observation = encode_context(self.context, time_step)
        with torch.inference_mode():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            distribution, self.state = self.model.policy.get_distribution(
                tensor,
                self.state,
                torch.tensor([self.episode_start], device=self.model.device, dtype=torch.float32),
            )
            # Only nonexistent physical bands are masked on smaller legacy worlds.
            # Every real band is always available, even during retuning.
            logits = distribution.distribution.logits[0, : self.context.history.num_bands]
            action = int(logits.argmax().item())
        self.episode_start = False
        self.pending = time_step, action
        return action

    def observe(self, observation):
        if self.pending != (observation.time_step, observation.band):
            raise ValueError("observation does not match action")
        self.context.observe(observation)
        self.pending = None
