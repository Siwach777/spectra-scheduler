"""Trajectory-trained action policy over a frozen causal timing belief.

The action network starts as the measured, trained timing scheduler. Only its
residual action preferences are optimized by the grouped trajectory experiment.
Emitter parameters, identities, truth and scenario labels never enter features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .evaluation_contract import Forecast
from .experiments.calibrated_timing import _CalibratedRatio
from .experiments.storage import fingerprint, load_torch
from .simulation import SyntheticAction
from .timing_belief import BeliefPolicyConfig, TimingBeliefPolicy, load_belief


class ActionResidual(nn.Module):
    def __init__(self, features=100, hidden=256):
        super().__init__()
        self.features, self.hidden = features, hidden
        self.layers = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, prior, legal):
        residual = 4 * (self.layers(features.float())[..., 0] / 4).tanh()
        return (prior + residual).masked_fill(~legal, -1e9)


def leave_one_out(returns):
    """Other independent trajectories of the same world form the baseline."""
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError("at least two returns per world required")
    return returns - (returns.sum(1, keepdims=True) - returns) / (returns.shape[1] - 1)


class GroupedTimingPolicy(_CalibratedRatio, TimingBeliefPolicy):
    def __init__(self, model, actor, config=None):
        super().__init__(model, config or BeliefPolicyConfig(revisit=256, exploration=0.02))
        self.actor = actor
        if next(actor.parameters()).device.type != "cuda":
            raise RuntimeError("grouped policy inference requires CUDA")

    def action_features(self, step, predicted):
        cfg = self.config
        dwells = np.asarray(cfg.dwells)
        previous = self.history.current_band
        delay = self.retune[previous] if previous >= 0 else np.zeros(self.bands, np.int32)
        remaining = self.horizon - step
        elapsed = np.minimum(delay[:, None] + dwells, remaining)
        cumulative = np.pad(predicted.cumsum(-1), ((0, 0), (1, 0)))
        captured = np.take_along_axis(cumulative, elapsed, 1) - np.take_along_axis(
            cumulative, np.broadcast_to(np.minimum(delay, remaining)[:, None], elapsed.shape), 1
        )
        age = np.where(
            self.history.last_listen >= 0, step - self.history.last_listen, step + cfg.revisit
        )
        hit_age = np.where(self.history.last_hit >= 0, step - self.history.last_hit, step + 1)
        score = captured / np.maximum(elapsed, 1)
        score += cfg.exploration * np.minimum(age[:, None] / cfg.revisit, 1)
        score -= cfg.retune_cost * delay[:, None] / np.maximum(elapsed, 1)
        if previous >= 0:
            score[previous] += cfg.switch_margin * score.max()
        history = self.history.encode()
        exposure, hits = history[:, 0].sum(-1), history[:, 1].sum(-1)
        band_rate = predicted.mean(-1)
        shape = score.shape

        def band(x):
            return np.broadcast_to(x[:, None], shape)

        def constant(x):
            return np.full(shape, x)

        statistics = np.stack(
            (
                np.log1p(captured),
                captured / np.maximum(elapsed, 1),
                band(delay / 8),
                elapsed / 64,
                np.broadcast_to(dwells[None] / 32, shape),
                band((np.arange(self.bands) == previous).astype(float)),
                constant(step / self.horizon),
                constant(remaining / self.horizon),
                band(age / (age + 64)),
                band(age / (age + 256)),
                band(hit_age / (hit_age + 256)),
                band((self.history.hits + 0.2) / (self.history.visits + 4)),
                band(exposure / self.model.config.history),
                band(hits / np.maximum(exposure, 1)),
                band(history[:, 2].sum(-1) / np.maximum(hits, 1)),
                band(predicted.std(-1)),
                constant(band_rate.mean()),
                constant(band_rate.max()),
                band(band_rate / max(float(band_rate.sum()), 1e-6)),
                score,
            ),
            -1,
        ).astype(np.float32)
        future = np.broadcast_to(predicted[:, None], (*shape, predicted.shape[-1]))
        features = np.concatenate((future, statistics), -1).reshape(-1, self.actor.features)
        # The tiny dwell term only resolves equal-scoring actions toward longer holds.
        prior = (score * 80 + np.broadcast_to(dwells[None], shape) * 1e-7).ravel()
        legal = np.ones(prior.shape, dtype=bool)
        overdue = np.flatnonzero(age >= cfg.revisit)
        if len(overdue):
            chosen_band = int(overdue[np.argmax(age[overdue])])
            chosen_dwell = int(np.argmin(abs(dwells - cfg.probe)))
            legal.fill(False)
            legal[chosen_band * len(dwells) + chosen_dwell] = True
        return features.astype(np.float16), prior.astype(np.float32), legal

    def accept_action(self, step, predicted, index):
        band, dwell_index = divmod(int(index), len(self.config.dwells))
        dwell = int(self.config.dwells[dwell_index])
        previous = self.history.current_band
        delay = int(self.retune[previous, band]) if previous >= 0 else 0
        remaining = self.horizon - step
        start, end = min(delay, remaining), min(delay + dwell, remaining)
        rates = np.clip(predicted[band, start:end], 0, 1 - 1e-7)
        survival = np.r_[1.0, np.cumprod(1 - rates)]
        mass = survival[:-1] * rates
        probability = float(1 - survival[-1])
        conditional = float(mass @ (np.arange(start, end) * 0.001) / max(probability, 1e-12))
        ratio = float(
            np.clip(predicted[band, start:end].sum() / max(predicted[:, :end].sum(), 1e-12), 0, 1)
        )
        self.pending = (
            step,
            band,
            dwell,
            Forecast(probability, conditional if probability >= 0.5 else None, ratio),
        )
        return SyntheticAction(band, dwell)

    @torch.inference_mode()
    def choose_action(self, step):
        if step != self.history.time:
            raise ValueError("policy clock differs from causal history")
        tensor = torch.from_numpy(self.history.encode()).unsqueeze(0).to("cuda")
        predicted = self.model(tensor)[0].cpu().numpy()
        features, prior, legal = self.action_features(step, predicted)
        logits = self.actor(
            torch.from_numpy(features).to("cuda"),
            torch.from_numpy(prior).to("cuda"),
            torch.from_numpy(legal).to("cuda"),
        )
        return self.accept_action(step, predicted, logits.argmax().item())


def load_grouped(path):
    payload = load_torch(path)
    if payload.get("version") != 1:
        raise ValueError("unsupported grouped policy checkpoint")
    forecaster = path.parent.parent / "forecaster.pt"
    if fingerprint(forecaster) != payload["metadata"]["semantic"]["initial_forecaster_sha256"]:
        raise ValueError("grouped policy's frozen forecaster changed")
    model, _ = load_belief(forecaster)
    actor = ActionResidual(payload["features"], payload["hidden"]).cuda().eval()
    actor.load_state_dict(payload["actor"])
    config = dict(payload["policy"])
    config["dwells"] = tuple(config["dwells"])
    return model, actor, BeliefPolicyConfig(**config), payload["metadata"]
