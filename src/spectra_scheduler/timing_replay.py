"""Transfer frozen timing forecasts to label-free, whole-tick PDW receiver replay."""

import numpy as np
import torch

from .models import Observation
from .simulation import SyntheticAction
from .timing_belief import BeliefPolicyConfig
from .timing_ensemble import load_predictor
from .timing_planner import (
    CalibratedTimingPlannerPolicy,
    ForecastPlannerWorkspace,
    first_actions,
)


class TimingReplayPolicy:
    """Use only delivered PDWs, listening masks and public receiver timing.

    Ticks are 1 ms. Retuning and dwell must be whole ticks. Dataset amplitudes
    are not dBm, so the pretrained power channel is zeroed rather than calibrated
    without evidence. Counts retain the model's saturation at four per tick.
    Synthetic forecast calibration is not claimed for these external recordings.
    """

    tick_us = 1000

    def __init__(self, model, config, maximum_batch=20):
        self.model, self.config = model, config
        self.maximum_batch = maximum_batch
        self._host = self._device = self._workspace = None

    @classmethod
    def from_checkpoint(cls, path, maximum_batch=20):
        model, metadata = load_predictor(path)
        settings = dict(metadata["policy"])
        settings["dwells"] = tuple(settings["dwells"])
        return cls(model, BeliefPolicyConfig(**settings), maximum_batch)

    def reset(self, specification, seed):
        receiver, interface = specification["receiver"], specification["interface"]
        for value in (receiver["start_us"], receiver["stop_us"], receiver["retune_us"]):
            if value % self.tick_us:
                raise ValueError("timing replay requires whole 1-ms receiver ticks")
        if receiver["slew_mhz_per_us"] is not None:
            raise ValueError("fractional slew timing is outside the frozen model interface")
        self.dwells = tuple(interface["dwell_us"])
        if self.dwells != tuple(d * self.tick_us for d in self.config.dwells):
            raise ValueError("replay dwell menu differs from the frozen model")
        self.bands, self.start_us = interface["bands"], receiver["start_us"]
        horizon = int((receiver["stop_us"] - self.start_us) / self.tick_us)
        self.scheduler = CalibratedTimingPlannerPolicy(self.model, self.config)
        retune = np.full((self.bands, self.bands), receiver["retune_us"] / self.tick_us)
        np.fill_diagonal(retune, 0)
        self.scheduler.set_retune_table(retune)
        self.scheduler.set_episode_horizon(horizon)
        self.scheduler.set_detection_probability(receiver["detection_probability"])
        self.scheduler.reset(self.bands)
        self._selected = None

    def act(self, observation):
        action = self.scheduler.choose_action(self.scheduler.history.time)
        self._selected = action.band
        return action.band * len(self.dwells) + self.config.dwells.index(action.dwell_steps)

    @torch.inference_mode()
    def act_batch(self, policies, observations):
        if not policies or len(policies) > self.maximum_batch:
            raise ValueError("replay inference exceeds its bounded batch")
        bands = policies[0].bands
        if any(p.bands != bands or p.config != self.config for p in policies):
            raise ValueError("replay inference requires matching band and policy configurations")
        if self._host is None:
            shape = (self.maximum_batch, bands, 3, self.model.config.history)
            self._host = torch.empty(shape, pin_memory=True)
            self._device = torch.empty(shape, device="cuda")
        for index, policy in enumerate(policies):
            self._host.numpy()[index] = policy.scheduler.history.encode()
        size = len(policies)
        self._device[:size].copy_(self._host[:size], non_blocking=True)
        predictions = self.model(self._device[:size]).cpu().numpy()
        if self._workspace is None or self._workspace.batch_size != size:
            self._workspace = ForecastPlannerWorkspace(
                size, bands, self.model.config.future, self.config.dwells
            )
        chosen, dwells = first_actions(
            predictions, [p.scheduler.history.current_band for p in policies],
            np.stack([p.scheduler.retune for p in policies]),
            [p.scheduler.horizon - p.scheduler.history.time for p in policies],
            self.config.dwells, self._workspace,
        )
        actions = []
        for index, policy in enumerate(policies):
            scheduler = policy.scheduler
            step = scheduler.history.time
            forced = scheduler.coverage_action(step)
            action = forced or SyntheticAction(int(chosen[index]), int(dwells[index]))
            scheduler.accept_action(step, predictions[index], action)
            policy._selected = action.band
            actions.append(action.band * len(self.config.dwells)
                           + self.config.dwells.index(action.dwell_steps))
        return actions

    def observe_pulses(self, observation):
        if observation is None or self._selected is None:
            raise ValueError("replay feedback requires an executed receiver action")
        relative = np.array([
            observation.start_us, observation.listening_start_us, observation.end_us
        ]) - self.start_us
        if np.any(relative % self.tick_us):
            raise ValueError("receiver observation is not aligned to 1-ms ticks")
        start, listening, end = (relative / self.tick_us).astype(np.int64)
        if start != self.scheduler.history.time:
            raise ValueError("PDW feedback is not contiguous with the causal history")
        offsets = ((observation.pulses[:, 0] - observation.start_us) / self.tick_us).astype(int)
        counts = np.bincount(offsets, minlength=end - start)
        for tick in range(int(start), int(end)):
            self.scheduler.observe(Observation(
                tick, self._selected, int(counts[tick - start]), tick >= listening
            ))
        self._selected = None
