"""Recurrent PPO for variable-duration actions with physical-time GAE.

Within-action rewards are discounted in the environment. The continuation and
trace discounts here are gamma**elapsed and (gamma*lambda)**elapsed. Episode
boundaries mask both; action duration must never be confused with one time tick.
"""

import numpy as np
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback


class DurationRolloutBuffer(RecurrentRolloutBuffer):
    def reset(self):
        super().reset()
        self.elapsed_steps = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.detach().cpu().numpy().flatten()
        advantage = np.zeros(self.n_envs, dtype=np.float32)
        discounts = self.gamma**self.elapsed_steps
        traces = (self.gamma * self.gae_lambda) ** self.elapsed_steps
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                continuing = 1.0 - dones.astype(np.float32)
                following = last_values
            else:
                continuing = 1.0 - self.episode_starts[step + 1]
                following = self.values[step + 1]
            residual = self.rewards[step] + discounts[step] * following * continuing
            residual -= self.values[step]
            advantage = residual + traces[step] * continuing * advantage
            self.advantages[step] = advantage
        self.returns = self.advantages + self.values


class _RecordDurations(BaseCallback):
    def _on_step(self):
        infos = self.locals["infos"]
        if any(info.get("TimeLimit.truncated", False) for info in infos):
            raise ValueError(
                "duration PPO requires finite task termination, not time-limit truncation"
            )
        durations = np.asarray([info["elapsed_steps"] for info in infos], dtype=np.float32)
        if (
            not np.isfinite(durations).all()
            or (durations < 1).any()
            or (durations != np.floor(durations)).any()
        ):
            raise ValueError("invalid physical action duration")
        buffer = self.model.rollout_buffer
        buffer.elapsed_steps[buffer.pos] = durations
        self.model.physical_steps += int(durations.sum())
        return True


class DurationRecurrentPPO(RecurrentPPO):
    def __init__(self, *args, **kwargs):
        self.physical_steps = 0
        self.dwell_steps = (1, 4, 8)
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        if self.action_space.n != 8 * len(self.dwell_steps):
            raise ValueError("duration PPO requires eight bands and matching dwell actions")
        shape = self.rollout_buffer.hidden_state_shape
        self.rollout_buffer = DurationRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def _init_callback(self, callback, progress_bar=False):
        callbacks = (
            callback if isinstance(callback, list) else ([] if callback is None else [callback])
        )
        callbacks = [
            item if isinstance(item, BaseCallback) else ConvertCallback(item) for item in callbacks
        ]
        return super()._init_callback(CallbackList([_RecordDurations(), *callbacks]), progress_bar)
