"""Learned causal timing beliefs for passive synthetic band-and-dwell scheduling.

Missing observations have an explicit mask. A shared temporal encoder learns to
weight periodic phase hypotheses from observed hits, misses and measurement power.
Future truth is used only by the experiment collector, never by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .evaluation_contract import Forecast
from .experiments.storage import load_torch, save_torch
from .simulation import SyntheticAction


@dataclass(frozen=True)
class BeliefConfig:
    history: int = 288
    future: int = 80
    max_period: int = 144
    width: int = 32

    def __post_init__(self):
        if min(self.history, self.future, self.max_period, self.width) < 2:
            raise ValueError("belief dimensions must be at least two")
        if self.history < self.max_period:
            raise ValueError("history must cover the longest timing hypothesis")


class TimingHistory:
    """Fixed ring of receiver-visible ticks: listen mask, count and measured power."""

    def __init__(self, bands, steps, *, count_limit=4):
        if count_limit is not None and (type(count_limit) is not int or count_limit < 1):
            raise ValueError("count limit must be a positive integer or None")
        self.count_limit = count_limit
        self.values = np.zeros((bands, 3, steps), np.float32)
        self.steps = steps
        self.time = 0
        self.current_band = -1
        self.last_listen = np.full(bands, -1, np.int64)
        self.last_hit = np.full(bands, -1, np.int64)
        self.visits = np.zeros(bands, np.int64)
        self.hits = np.zeros(bands, np.float64)
        self.output = np.empty_like(self.values)

    def observe(self, observation):
        if observation.time_step != self.time:
            raise ValueError("timing history requires contiguous causal observations")
        slot = self.time % self.steps
        self.values[:, :, slot] = 0
        band = observation.band
        self.current_band = band
        if observation.listening:
            self.values[band, 0, slot] = 1
            self.values[band, 1, slot] = (
                observation.detections if self.count_limit is None
                else min(observation.detections, self.count_limit)
            )
            if observation.measurements:
                power = max(m.power_dbm for m in observation.measurements)
                self.values[band, 2, slot] = np.clip((power + 100) / 40, 0, 2)
            self.last_listen[band] = self.time
            self.visits[band] += 1
            self.hits[band] += observation.hit
            if observation.hit:
                self.last_hit[band] = self.time
        self.time += 1

    def encode(self):
        slot = self.time % self.steps
        tail = self.steps - slot
        self.output[:, :, :tail] = self.values[:, :, slot:]
        self.output[:, :, tail:] = self.values[:, :, :slot]
        return self.output


class _TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return x + self.mix(F.silu(self.conv(F.pad(x, (self.padding, 0)))))


class TimingBeliefNetwork(nn.Module):
    """Temporal convolution plus learned mixture of periodic phase posteriors.

    Every band shares parameters and has no absolute band identifier. Periods
    are hypotheses, not supplied emitter parameters. An aperiodic expert retains
    a causal rate estimate; the learned gate can reject spurious periodic fits.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or BeliefConfig()
        self.config = config
        width = config.width
        self.temporal = nn.Sequential(
            nn.Conv1d(3, width, 1),
            nn.SiLU(),
            *(_TemporalBlock(width, d) for d in (1, 2, 4, 8, 16, 32, 64)),
        )
        self.context = nn.Sequential(nn.Linear(width * 2 + 4, 64), nn.SiLU())
        self.gate = nn.Sequential(
            nn.Linear(64 + 8, 96), nn.SiLU(), nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1)
        )
        self.rate_head = nn.Linear(64, 1)
        self.quality_threshold = nn.Parameter(torch.tensor(0.35))
        self.quality_gain = nn.Parameter(torch.tensor(2.5))
        self.log_prior_strength = nn.Parameter(torch.tensor(-1.5))
        self.evidence_scale = nn.Parameter(torch.tensor(0.3))
        periods = torch.arange(2, config.max_period + 1)
        past = torch.arange(-config.history, 0)
        future = torch.arange(config.future)
        self.register_buffer("periods", periods, persistent=False)
        self.register_buffer("phase_index", past[None] % periods[:, None], persistent=False)
        self.register_buffer("future_phase", future[None] % periods[:, None], persistent=False)
        self.register_buffer(
            "phase_valid", torch.arange(config.max_period)[None] < periods[:, None],
            persistent=False,
        )

    def forward(self, history, *, power_available=True, rate_prior=None):
        if history.ndim != 4 or tuple(history.shape[2:]) != (3, self.config.history):
            raise ValueError("expected batch, bands, three channels, history ticks")
        batch, bands = history.shape[:2]
        x = history.flatten(0, 1).float()
        mask, observed, power = x.unbind(1)
        quality = (
            ((power - self.quality_threshold) * self.quality_gain.exp()).sigmoid()
            if power_available else torch.ones_like(observed)
        )
        hits = observed * quality
        exposure = mask.sum(-1)
        count = hits.sum(-1)
        rate = (count + 0.2) / (exposure + 4)
        global_rate = rate.reshape(batch, bands).mean(-1, keepdim=True).expand(-1, bands).flatten()
        unseen = exposure == 0
        backoff = global_rate.clamp_min(0.025)
        if rate_prior is not None:
            if tuple(rate_prior.shape) != (batch, bands):
                raise ValueError("causal rate prior must match batch and bands")
            prior_rate = rate_prior.flatten()
            backoff = torch.where(prior_rate >= 0, prior_rate, backoff)
        rate = torch.where(unseen, backoff, rate)
        encoded = self.temporal(torch.stack((mask, hits, power * (observed > 0)), 1))
        summary = torch.stack(
            (torch.log1p(exposure) / 6, torch.log1p(count) / 5, rate, global_rate), -1
        )
        context = self.context(torch.cat((encoded[:, :, -1], encoded.mean(-1), summary), -1))
        n, p, phases = len(x), len(self.periods), self.config.max_period
        index = self.phase_index.expand(n, -1, -1)
        phase_exposure = x.new_zeros((n, p, phases))
        phase_hits = x.new_zeros((n, p, phases))
        phase_exposure.scatter_add_(2, index, mask[:, None].expand(-1, p, -1))
        phase_hits.scatter_add_(2, index, hits[:, None].expand(-1, p, -1))
        strength = self.log_prior_strength.exp().clamp(0.02, 5)
        prior = rate[:, None, None]
        # Shrink unseen phases toward the causal band rate, rather than zero.
        phase_rate = (phase_hits + strength * prior) / (phase_exposure + strength)
        probability = phase_rate.clamp(1e-5, 1 - 1e-5)
        positive = phase_hits.clamp(max=phase_exposure)
        negative = (phase_exposure - positive).clamp_min(0)
        log_likelihood = positive * probability.log() + negative * torch.log1p(-probability)
        base_probability = prior.clamp(1e-5, 1 - 1e-5)
        base_ll = positive * base_probability.log() + negative * torch.log1p(-base_probability)
        improvement = (log_likelihood - base_ll).sum(-1) / exposure[:, None].clamp_min(1)
        supported = (phase_exposure >= 2) & self.phase_valid[None]
        confident = (positive >= 1) & supported
        conflict = (positive * negative / phase_exposure.clamp_min(1)).sum(-1)
        features = torch.stack(
            (
                improvement,
                supported.float().sum(-1) / self.periods,
                confident.float().sum(-1) / self.periods,
                conflict / exposure[:, None].clamp_min(1),
                self.periods.float().log().expand(n, -1) / 6,
                (exposure[:, None] / self.periods).clamp(max=10) / 10,
                (count[:, None] / self.periods).clamp(max=5) / 5,
                rate[:, None].expand(-1, p),
            ), -1,
        )
        logits = self.gate(torch.cat((context[:, None].expand(-1, p, -1), features), -1))[..., 0]
        logits = logits + self.evidence_scale * improvement * exposure[:, None].sqrt()
        base_features = features.new_zeros((n, 1, 8))
        base_features[:, 0, -1] = rate
        base_logit = self.gate(torch.cat((context[:, None], base_features), -1))[..., 0]
        weights = torch.cat((logits, base_logit), -1).softmax(-1)
        projected = phase_rate.gather(2, self.future_phase.expand(n, -1, -1))
        predicted = (weights[:, :-1, None] * projected).sum(1)
        predicted += weights[:, -1, None] * rate[:, None]
        correction = (self.rate_head(context).tanh() * 1.5).exp()
        return (predicted * correction).reshape(batch, bands, self.config.future)


def belief_loss(model, history, future, valid):
    """All-band future count supervision, including observed and unobserved bands."""
    prediction = model(history).clamp_min(1e-6)
    # Poisson score estimates a conditional mean without binary class weighting
    # that would systematically destroy probability calibration.
    loss = prediction - future * prediction.log()
    weights = valid[:, None].expand_as(loss)
    count_loss = (loss * weights).sum() / weights.sum().clamp_min(1)
    # Also fit integrated capture, so dense windows matter to action ranking.
    horizons = tuple(h for h in (4, 8, 16, 32, 64) if h <= prediction.shape[-1])
    if not horizons:
        horizons = (prediction.shape[-1],)
    integrated = prediction.cumsum(-1)
    target = future.cumsum(-1)
    ranking = prediction.new_zeros(())
    for horizon in horizons:
        fitted = integrated[..., horizon - 1]
        actual = target[..., horizon - 1]
        eligible = valid[:, horizon - 1, None]
        ranking += (F.smooth_l1_loss(torch.log1p(fitted), torch.log1p(actual), reduction="none")
                    * eligible).sum() / (eligible.sum() * prediction.shape[1]).clamp_min(1)
    return count_loss + 0.3 * ranking / len(horizons)


def save_belief(path, model, metadata):
    save_torch(path, {
        "version": 1, "config": asdict(model.config), "metadata": metadata,
        "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    })


def load_belief(path, device="cuda"):
    if device != "cuda":
        raise RuntimeError("neural belief inference requires CUDA")
    payload = load_torch(path)
    if payload.get("version") != 1:
        raise ValueError("unsupported timing belief checkpoint")
    model = TimingBeliefNetwork(BeliefConfig(**payload["config"]))
    model.load_state_dict(payload["weights"])
    model.to(device).eval()
    return model, payload["metadata"]


@dataclass(frozen=True)
class BeliefPolicyConfig:
    dwells: tuple[int, ...] = (1, 4, 8, 16, 32)
    revisit: int = 96
    probe: int = 8
    exploration: float = 0.05
    retune_cost: float = 0.03
    switch_margin: float = 0.04


def validated_retune_table(table, future, dwells):
    """Reject receiver timings that cannot be scored inside the forecast horizon."""
    values = np.asarray(table, dtype=np.float64)
    if (
        values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0
        or not np.isfinite(values).all() or (values < 0).any()
        or (values != np.floor(values)).any()
    ):
        raise ValueError("retune table must be square, finite, nonnegative integer ticks")
    if values.max() + max(dwells) > future:
        raise ValueError("forecast horizon does not cover public retuning and available dwells")
    return values.astype(np.int32)


class TimingBeliefPolicy:
    """Schedule by expected captured pulses per elapsed tick from causal forecasts."""

    def __init__(self, model, config=None, *, empirical=False):
        self.model = model
        self.config = config or BeliefPolicyConfig()
        config = self.config
        self.empirical = empirical
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError("neural belief policy requires CUDA")
        if max(config.dwells) + 8 > model.config.future:
            raise ValueError("forecast horizon does not cover dwell and retuning")

    def set_retune_table(self, table):
        self.retune = validated_retune_table(table, self.model.config.future, self.config.dwells)

    def set_episode_horizon(self, horizon):
        self.horizon = horizon

    def reset(self, bands):
        if self.retune.shape != (bands, bands):
            raise ValueError("public retune table differs from the receiver band count")
        self.history = TimingHistory(bands, self.model.config.history)
        self.bands = bands
        self.pending = None
        self.decisions = self.probes = 0

    @torch.inference_mode()
    def choose_action(self, time_step):
        if time_step != self.history.time:
            raise ValueError("policy clock differs from causal history")
        if self.empirical:
            rate = (self.history.hits + 0.2) / (self.history.visits + 4)
            predicted = np.repeat(rate[:, None], self.model.config.future, axis=1)
        else:
            x = torch.from_numpy(self.history.encode()).unsqueeze(0).to(self.device)
            predicted = self.model(x)[0].cpu().numpy()
        return self.select(time_step, predicted)

    def select(self, time_step, predicted):
        cfg = self.config
        previous = self.history.current_band
        delay = self.retune[previous] if previous >= 0 else np.zeros(self.bands, np.int32)
        remaining = self.horizon - time_step
        dwells = np.asarray(cfg.dwells)
        elapsed = np.minimum(delay[:, None] + dwells, remaining)
        cumulative = np.pad(predicted.cumsum(-1), ((0, 0), (1, 0)))
        counts = np.take_along_axis(cumulative, elapsed, 1) - np.take_along_axis(
            cumulative, np.broadcast_to(np.minimum(delay, remaining)[:, None], elapsed.shape), 1
        )
        ages = np.where(self.history.last_listen >= 0,
                        time_step - self.history.last_listen, time_step + cfg.revisit)
        score = counts / np.maximum(elapsed, 1)
        score += cfg.exploration * np.minimum(ages[:, None] / cfg.revisit, 1)
        score -= cfg.retune_cost * delay[:, None] / np.maximum(elapsed, 1)
        if previous >= 0:
            score[previous] += cfg.switch_margin * score.max()
        overdue = np.flatnonzero(ages >= cfg.revisit)
        self.decisions += 1
        if len(overdue):
            band = int(overdue[np.argmax(ages[overdue])])
            dwell_index = int(np.argmin(abs(dwells - cfg.probe)))
            self.probes += 1
        else:
            best = np.argwhere(score == score.max())
            band, dwell_index = map(int, best[np.argmax(dwells[best[:, 1]])])
        dwell = int(dwells[dwell_index])
        start, end = min(int(delay[band]), remaining), int(elapsed[band, dwell_index])
        rates = np.clip(predicted[band, start:end], 0, 1 - 1e-7)
        survival = np.concatenate(([1.0], np.cumprod(1 - rates)))
        mass = survival[:-1] * rates
        probability = float(1 - survival[-1])
        conditional_time = float(mass @ (np.arange(start, end) * 0.001) / max(probability, 1e-12))
        denominator = predicted[:, :end].sum()
        ratio = float(np.clip(counts[band, dwell_index] / max(denominator, 1e-12), 0, 1))
        self.pending = (time_step, band, dwell, Forecast(
            probability, conditional_time if probability >= 0.5 else None, ratio
        ))
        return SyntheticAction(band, dwell)

    def forecast(self, time_step, action):
        start, band, dwell, forecast = self.pending
        if (time_step, action.band, action.dwell_steps) != (start, band, dwell):
            raise ValueError("forecast requested for a different action")
        return None if self.empirical else forecast

    def observe(self, observation):
        self.history.observe(observation)
