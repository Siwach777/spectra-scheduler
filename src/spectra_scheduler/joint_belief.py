"""Causal joint timing hypotheses with a frozen independent-band anchor.

Joint period evidence is shared across bands. A competing-emitter expert uses
hits on another band as evidence against simultaneous activity; an independent
expert and the learned anchor gate retain support for overlapping emitters.
No emitter parameters, world identifiers or future truth enter inference.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from .experiments.storage import load_torch, save_torch
from .timing_belief import BeliefConfig, TimingBeliefNetwork, _TemporalBlock


class JointTimingNetwork(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.config = base.config
        width = self.config.width
        self.temporal = nn.Sequential(
            nn.Conv1d(3, width, 1),
            nn.SiLU(),
            *(_TemporalBlock(width, d) for d in (1, 2, 4, 8, 16, 32, 64)),
        )
        self.temporal.load_state_dict(base.temporal.state_dict())
        self.context = nn.Sequential(nn.Linear(2 * width + 4, 64), nn.SiLU())
        self.period_gate = nn.Sequential(
            nn.Linear(64 + 10, 96), nn.SiLU(), nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1)
        )
        self.band_gate = nn.Sequential(nn.Linear(64 + 8, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.band_gate[-1].weight)
        with torch.no_grad():
            self.band_gate[-1].bias.copy_(torch.tensor([-2.0, 0.0]))
        self.quality_threshold = nn.Parameter(base.quality_threshold.detach().clone())
        self.quality_gain = nn.Parameter(base.quality_gain.detach().clone())
        self.log_strength = nn.Parameter(torch.tensor(-1.5))
        self.exclusive_logit = nn.Parameter(torch.tensor(0.0))
        self.evidence_scale = nn.Parameter(torch.tensor(1.0))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, history, base_prediction=None):
        if history.ndim != 4 or tuple(history.shape[2:]) != (3, self.config.history):
            raise ValueError("expected batch, bands, three channels, history ticks")
        batch, bands = history.shape[:2]
        if base_prediction is None:
            with torch.no_grad():
                base_prediction = self.base(history)
        mask, observed, power = history.float().unbind(2)
        quality = ((power - self.quality_threshold) * self.quality_gain.exp()).sigmoid()
        hits = observed * quality
        exposure, count = mask.sum(-1), hits.sum(-1)
        rate = (count + 0.2) / (exposure + 4)
        rate = torch.where(exposure > 0, rate, rate.mean(1, keepdim=True).clamp_min(0.025))
        global_input = torch.stack((mask.sum(1), hits.sum(1), (power * (observed > 0)).sum(1)), 1)
        encoded = self.temporal(global_input)
        summary = torch.stack(
            (
                torch.log1p(exposure.sum(1)) / 7,
                torch.log1p(count.sum(1)) / 6,
                rate.mean(1),
                rate.max(1).values,
            ),
            -1,
        )
        context = self.context(torch.cat((encoded[:, :, -1], encoded.mean(-1), summary), -1))
        periods, phases = len(self.base.periods), self.config.max_period
        index = self.base.phase_index[None, None].expand(batch, bands, -1, -1)
        phase_exposure = history.new_zeros((batch, bands, periods, phases), dtype=torch.float32)
        phase_hits = torch.zeros_like(phase_exposure)
        phase_exposure.scatter_add_(3, index, mask[:, :, None].expand(-1, -1, periods, -1))
        phase_hits.scatter_add_(3, index, hits[:, :, None].expand(-1, -1, periods, -1))
        strength = self.log_strength.exp().clamp(0.02, 5)
        prior = rate[:, :, None, None]
        independent = (phase_hits + strength * prior) / (phase_exposure + strength)
        other_hits = (phase_hits.sum(1, keepdim=True) - phase_hits).clamp_min(0)
        competing = (phase_hits + strength * prior) / (phase_exposure + other_hits + strength)
        mixture = self.exclusive_logit.sigmoid()
        phase_rate = (1 - mixture) * independent + mixture * competing
        probability = phase_rate.clamp(1e-5, 1 - 1e-5)
        positive = torch.minimum(phase_hits, phase_exposure)
        negative = (phase_exposure - positive).clamp_min(0)
        ll = positive * probability.log() + negative * torch.log1p(-probability)
        bp = prior.clamp(1e-5, 1 - 1e-5)
        baseline_ll = positive * bp.log() + negative * torch.log1p(-bp)
        total_exposure = exposure.sum(1).clamp_min(1)
        improvement = (ll - baseline_ll).sum((1, 3)) / total_exposure[:, None]
        valid = self.base.phase_valid[None, None]
        supported = (phase_exposure >= 2) & valid
        confident = (positive >= 1) & supported
        conflict = (positive * negative / phase_exposure.clamp_min(1)).sum((1, 3))
        mass = phase_hits.sum(1)
        collision = (mass.square() - phase_hits.square().sum(1)).sum(-1)
        features = torch.stack(
            (
                improvement,
                supported.float().sum((1, 3)) / (self.base.periods * bands),
                confident.float().sum((1, 3)) / (self.base.periods * bands),
                conflict / total_exposure[:, None],
                self.base.periods.float().log().expand(batch, -1) / 6,
                (total_exposure[:, None] / self.base.periods).clamp(max=20) / 20,
                (count.sum(1)[:, None] / self.base.periods).clamp(max=10) / 10,
                rate.mean(1)[:, None].expand(-1, periods),
                collision / mass.sum(-1).square().clamp_min(1),
                (mass > 0.5).float().sum(-1) / self.base.periods,
            ),
            -1,
        )
        logits = self.period_gate(
            torch.cat((context[:, None].expand(-1, periods, -1), features), -1)
        )[..., 0]
        logits = logits + self.evidence_scale * improvement * total_exposure[:, None].sqrt()
        base_features = features.new_zeros((batch, 1, 10))
        base_features[:, 0, 7] = rate.mean(1)
        base_logit = self.period_gate(torch.cat((context[:, None], base_features), -1))[..., 0]
        weights = torch.cat((logits, base_logit), -1).softmax(-1)
        future_index = self.base.future_phase[None, None].expand(batch, bands, -1, -1)
        projected = phase_rate.gather(3, future_index)
        predicted = (weights[:, None, :-1, None] * projected).sum(2)
        predicted = predicted + weights[:, None, -1, None] * rate[:, :, None]
        band_stats = torch.stack(
            (
                torch.log1p(exposure) / 6,
                torch.log1p(count) / 5,
                rate,
                base_prediction.mean(-1),
                base_prediction.std(-1),
                mask[..., -32:].mean(-1),
                hits[..., -32:].mean(-1),
                (observed > 0).float().sum(-1) / exposure.clamp_min(1),
            ),
            -1,
        )
        blend, correction = (
            self.band_gate(torch.cat((context[:, None].expand(-1, bands, -1), band_stats), -1))
            .float()
            .unbind(-1)
        )
        predicted = predicted * (1.5 * correction.tanh()).exp()[:, :, None]
        blend = blend.sigmoid()[:, :, None]
        return (1 - blend) * base_prediction + blend * predicted


def save_joint(path, model, metadata):
    save_torch(
        path,
        {
            "kind": "joint-timing",
            "version": 1,
            "config": asdict(model.config),
            "metadata": metadata,
            "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
    )


def load_joint(path, device="cuda"):
    if device != "cuda":
        raise RuntimeError("neural joint timing inference requires CUDA")
    payload = load_torch(path)
    if payload.get("kind") != "joint-timing" or payload.get("version") != 1:
        raise ValueError("unsupported joint timing checkpoint")
    model = JointTimingNetwork(TimingBeliefNetwork(BeliefConfig(**payload["config"])))
    model.load_state_dict(payload["weights"])
    return model.to(device).eval(), payload["metadata"]
