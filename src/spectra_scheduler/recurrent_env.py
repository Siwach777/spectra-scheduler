"""Gymnasium adapter using exactly the same receiver engine as baseline runs."""

import gymnasium as gym
import numpy as np

from spectra_scheduler.recurrent_context import make_context
from spectra_scheduler.rl import RewardConfig
from spectra_scheduler.rl_scenarios import physical_scenario, procedural_scenario
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
        self,
        split="train",
        worker=0,
        workers=1,
        start_episode=0,
        reward=None,
        shifted=False,
        mixed_receivers=False,
        dwell_steps=(1,),
        gamma=0.99,
        physical_contract=False,
        observation_version=OBSERVATION_VERSION,
        coverage_steps=512,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError("invalid split")
        self.split, self.worker, self.workers = split, worker, workers
        self.episode_index = start_episode
        self.start_episode = start_episode
        self.reward = reward or RewardConfig(coverage=0.2)
        self.shifted = shifted
        self.mixed_receivers = mixed_receivers
        if (
            not dwell_steps
            or any(type(d) is not int or d < 1 for d in dwell_steps)
            or len(set(dwell_steps)) != len(dwell_steps)
            or not 0 < gamma <= 1
        ):
            raise ValueError("invalid dwell steps or physical-time discount")
        self.dwell_steps, self.gamma = tuple(dwell_steps), gamma
        self.physical_contract = physical_contract
        if observation_version == 2 and not physical_contract:
            raise ValueError("multiscale context requires the physical action contract")
        self.observation_version = observation_version
        self.coverage_steps = coverage_steps
        self.action_space = gym.spaces.Discrete(MAX_BANDS * len(self.dwell_steps))
        self.observation_space = gym.spaces.Box(
            0, 1,
            shape=encode_context(make_context(8, observation_version, coverage_steps), 0).shape,
            dtype=np.float32,
        )
        self.episode = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.episode_index = self.start_episode + seed
        world_seed = self.episode_index * self.workers + self.worker
        self.episode_index += 1
        shifted = self.shifted or (self.mixed_receivers and world_seed % 4 == 0)
        simulation = (
            physical_scenario(world_seed, self.split, shifted)
            if self.physical_contract
            else procedural_scenario(world_seed, self.split, shifted, num_bands=8)
        )
        self.episode = SimulationEpisode(simulation)
        self.context = make_context(8, self.observation_version, self.coverage_steps)
        return encode_context(self.context, 0), {}

    def step(self, action):
        if self.episode is None:
            raise RuntimeError("reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError("invalid band action")
        band, dwell = divmod(int(action), len(self.dwell_steps))
        reward, elapsed, listening = 0.0, 0, 0
        while (listening if self.physical_contract else elapsed) < self.dwell_steps[dwell]:
            observation = self.episode.step(band)
            self.context.observe(observation)
            reward += self.gamma**elapsed * self.reward.compute(observation, self.context)
            elapsed += 1
            listening += int(observation.listening)
            ended = self.episode.time_step == self.episode.simulation.duration
            if ended:
                break
        # Finite episode is a task boundary: never bootstrap into a new random world.
        info = (
            {"elapsed_steps": elapsed} if self.dwell_steps != (1,) or self.physical_contract else {}
        )
        return encode_context(self.context, self.episode.time_step), reward, ended, False, info


class RecurrentScheduler:
    """Frozen recurrent inference with per-episode memory reset, no scan overrides."""

    def __init__(self, model):
        self.model = model

    def reset(self, num_bands):
        if not 1 <= num_bands <= MAX_BANDS:
            raise ValueError("recurrent policy supports one to eight bands")
        import torch

        self.context = make_context(
            num_bands, getattr(self.model, "observation_version", OBSERVATION_VERSION),
            getattr(self.model, "coverage_steps", 512),
        )
        shape = self.model.policy.lstm_hidden_state_shape
        self.state = (
            torch.zeros(shape, device=self.model.device),
            torch.zeros(shape, device=self.model.device),
        )
        self.episode_start = True
        self.pending = None
        self.remaining = 0
        self.dwell_steps = getattr(self.model, "dwell_steps", (1,))
        self.physical_contract = getattr(self.model, "physical_contract", False)

    def choose_band(self, time_step):
        import torch

        if self.pending is not None:
            raise ValueError("previous action has no observation")
        if self.remaining:
            if not self.physical_contract:
                self.remaining -= 1
            self.pending = time_step, self.held_band
            return self.held_band
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
            logits = distribution.distribution.logits[
                0, : self.context.history.num_bands * len(self.dwell_steps)
            ]
            band, dwell = divmod(int(logits.argmax().item()), len(self.dwell_steps))
            self.remaining = (
                self.dwell_steps[dwell] if self.physical_contract else self.dwell_steps[dwell] - 1
            )
            self.held_band = band
        self.episode_start = False
        self.pending = time_step, band
        return band

    def observe(self, observation):
        if self.pending != (observation.time_step, observation.band):
            raise ValueError("observation does not match action")
        self.context.observe(observation)
        if self.physical_contract and observation.listening:
            self.remaining -= 1
        self.pending = None
