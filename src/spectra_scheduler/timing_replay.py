"""Transfer frozen timing forecasts to label-free, whole-tick PDW receiver replay."""

import numpy as np
import torch

from .models import Observation
from .simulation import SyntheticAction
from .experiments.storage import load_torch, save_torch
from .timing_belief import BeliefConfig, BeliefPolicyConfig, TimingBeliefNetwork, TimingHistory
from .timing_ensemble import load_predictor
from .timing_planner import (
    CalibratedTimingPlannerPolicy,
    ForecastPlannerWorkspace,
    first_actions,
)


class PowerBlindTimingPredictor(torch.nn.Module):
    """Keep delivered counts when calibrated measurement power is unavailable."""

    def __init__(self, model):
        super().__init__()
        self.model, self.config = model, model.config

    def forward(self, history, *, rate_prior=None):
        return self.model(history, power_available=False, rate_prior=rate_prior)


def load_pdw_predictor(path):
    """PDW models use a distinct version so synthetic CLI loaders reject them."""
    payload = load_torch(path)
    if payload.get("kind") != "pdw_timing_belief" or payload.get("version") != 2:
        raise ValueError("unsupported PDW timing checkpoint")
    model = TimingBeliefNetwork(BeliefConfig(**payload["config"]))
    model.load_state_dict(payload["weights"])
    if not torch.cuda.is_available():
        raise RuntimeError("PDW timing inference requires CUDA")
    return PowerBlindTimingPredictor(model.to("cuda").eval()).eval(), payload["metadata"]


def save_pdw_predictor(path, predictor, metadata):
    from dataclasses import asdict

    model = predictor.model
    save_torch(path, {"kind": "pdw_timing_belief", "version": 2,
        "config": asdict(model.config), "metadata": metadata,
        "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()}})


def record_pdw_history(history, observation, band, *, start_us=0, observe=None):
    """Reconstruct a causal ring from delivered PDWs; also usable in CPU preparation."""
    tick_us = TimingReplayPolicy.tick_us
    relative = np.array([
        observation.start_us, observation.listening_start_us, observation.end_us
    ]) - start_us
    if np.any(relative % tick_us):
        raise ValueError("receiver observation is not aligned to 1-ms ticks")
    start, listening, end = (relative / tick_us).astype(np.int64)
    if start != history.time:
        raise ValueError("PDW feedback is not contiguous with the causal history")
    toa = observation.pulses[:, 0]
    if np.any(~np.isfinite(toa)) or np.any(toa < observation.listening_start_us) or np.any(
        toa >= observation.end_us
    ):
        raise ValueError("delivered PDWs must lie inside the executed listening window")
    offsets = ((toa - observation.start_us) / tick_us).astype(np.int64)
    counts = np.bincount(offsets, minlength=end - start)
    feed = history.observe if observe is None else observe
    for tick in range(int(start), int(end)):
        feed(Observation(tick, band, int(counts[tick - start]), tick >= listening))
    return int(counts.sum())


class TimingReplayPolicy:
    """Use only delivered PDWs, listening masks and public receiver timing.

    Ticks are 1 ms. Retuning and dwell must be whole ticks. Dataset amplitudes
    are not dBm. PDW checkpoints bypass the measured-power gate and retain raw
    counts. Synthetic checkpoints use the corrected adapter when loaded through
    from_checkpoint(); adapter="legacy" reproduces the historical transfer.
    Synthetic forecast calibration is not claimed for external recordings.
    """

    tick_us = 1000

    def __init__(self, model, config, maximum_batch=20, *, count_limit=4,
                 persistent_rates=False):
        if maximum_batch < 1:
            raise ValueError("maximum replay batch must be positive")
        self.model, self.config = model, config
        self.count_limit = count_limit
        self.persistent_rates = persistent_rates
        self.maximum_batch = maximum_batch
        self._host = self._device = self._workspace = None
        self._prior_host = self._prior_device = None

    @classmethod
    def from_checkpoint(cls, path, maximum_batch=20, *, adapter="missing-power"):
        if adapter not in ("legacy", "missing-power"):
            raise ValueError("unknown timing replay adapter")
        pdw = load_torch(path).get("kind") == "pdw_timing_belief"
        model, metadata = load_pdw_predictor(path) if pdw else load_predictor(path)
        corrected = pdw or adapter == "missing-power"
        if corrected and not pdw:
            if not isinstance(model, TimingBeliefNetwork):
                raise ValueError("missing-power transfer requires a single timing forecaster")
            model = PowerBlindTimingPredictor(model)
        settings = dict(metadata["policy"])
        settings["dwells"] = tuple(settings["dwells"])
        policy = cls(model, BeliefPolicyConfig(**settings), maximum_batch,
                     count_limit=None if corrected else 4)
        policy.metadata = metadata
        policy.adapter = "pdw-trained" if pdw else adapter
        return policy

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
        self.scheduler.history = TimingHistory(
            self.bands, self.model.config.history, count_limit=self.count_limit
        )
        self.delivered_counts = np.zeros(self.bands, np.float64)
        self._selected = None

    def causal_rates(self):
        history = self.scheduler.history
        return np.where(history.visits > 0,
                        (self.delivered_counts + 0.2) / (history.visits + 4), -1)

    @torch.inference_mode()
    def act(self, observation):
        scheduler = self.scheduler
        if self.persistent_rates:
            history = torch.from_numpy(scheduler.history.encode()[None]).to("cuda")
            prior = torch.as_tensor(self.causal_rates()[None], device="cuda", dtype=torch.float32)
            predicted = self.model(history, rate_prior=prior)[0].cpu().numpy()
            action = scheduler.select(scheduler.history.time, predicted)
        else:
            action = scheduler.choose_action(scheduler.history.time)
        self._selected = action.band
        return action.band * len(self.dwells) + self.config.dwells.index(action.dwell_steps)

    @torch.inference_mode()
    def act_batch(self, policies, observations):
        if not policies or len(policies) > self.maximum_batch:
            raise ValueError("replay inference exceeds its bounded batch")
        bands = policies[0].bands
        if any(p.bands != bands or p.config != self.config
               or p.persistent_rates != self.persistent_rates
               or p.count_limit != self.count_limit for p in policies):
            raise ValueError("replay inference requires matching band and policy configurations")
        if self._host is None or self._host.shape[1] != bands:
            shape = (self.maximum_batch, bands, 3, self.model.config.history)
            self._host = torch.empty(shape, pin_memory=True)
            self._device = torch.empty(shape, device="cuda")
            if self.persistent_rates:
                self._prior_host = torch.empty((self.maximum_batch, bands), pin_memory=True)
                self._prior_device = torch.empty((self.maximum_batch, bands), device="cuda")
        for index, policy in enumerate(policies):
            self._host.numpy()[index] = policy.scheduler.history.encode()
            if self.persistent_rates:
                self._prior_host.numpy()[index] = policy.causal_rates()
        size = len(policies)
        self._device[:size].copy_(self._host[:size], non_blocking=True)
        if self.persistent_rates:
            self._prior_device[:size].copy_(self._prior_host[:size], non_blocking=True)
            predictions = self.model(self._device[:size],
                                     rate_prior=self._prior_device[:size]).cpu().numpy()
        else:
            predictions = self.model(self._device[:size]).cpu().numpy()
        if (self._workspace is None or self._workspace.batch_size != size
                or self._workspace.bands != bands):
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
        self.delivered_counts[self._selected] += record_pdw_history(
            self.scheduler.history, observation, self._selected, start_us=self.start_us,
            observe=self.scheduler.observe,
        )
        self._selected = None
