"""Causal conditional correction of measured rates by a frozen dense model."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DenseRateGate(nn.Module):
    """Shared band gate; base checkpoint weights remain frozen."""

    def __init__(self, base, specification, hidden=32):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.config = base.config
        self.specification = specification
        self.hidden = hidden
        self.gate = nn.Sequential(nn.Linear(25, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(0.25 / 0.75))
        self.register_buffer(
            "dwell_seconds", torch.tensor(specification["interface"]["dwell_us"]) * 1e-6
        )
        self.register_buffer("band_indices", torch.arange(self.config.bands), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def rates_and_gate(self, history):
        with torch.no_grad():
            counts = self.base(history).reshape(-1, self.config.bands, self.config.dwells)
            model_log_rate = torch.log1p(counts / self.dwell_seconds)
        local = history[..., :-2].reshape(-1, self.config.history_steps, self.config.bands, 9)
        latest = local[:, -1]
        observed = latest[..., 3] * 20
        series = local[..., 3] * 20
        refresh = (local[:, 1:, :, 1] < local[:, :-1, :, 1]).float().mean(1)
        # Existing history repeats stale measurements, so variation and refresh
        # are support proxies rather than independent pulse samples.
        long_model = model_log_rate[..., -1]
        model_global = long_model.mean(-1, keepdim=True).expand_as(observed)
        observed_global = observed.mean(-1, keepdim=True).expand_as(observed)
        observed_max = observed.amax(-1, keepdim=True).expand_as(observed)
        left = F.pad(observed.unsqueeze(1), (1, 0), mode="replicate")[:, 0, :-1]
        right = F.pad(observed.unsqueeze(1), (0, 1), mode="replicate")[:, 0, 1:]
        progress = history[:, -1, -2, None].expand_as(observed)
        tuned = history[:, -1, -1, None] * max(1, self.config.bands - 1)
        distance = (tuned - self.band_indices).abs() / max(1, self.config.bands - 1)
        extras = torch.stack(
            (
                latest[..., 1] * 20,
                refresh,
                series.std(1) / 20,
                series.amin(1) / 20,
                series.amax(1) / 20,
                long_model / 20,
                (long_model - observed) / 20,
                model_global / 20,
                observed_global / 20,
                observed_max / 20,
                left / 20,
                right / 20,
                progress,
                distance,
                (distance < 1e-3).float(),
                latest[..., 0] * latest[..., 2],
            ),
            dim=-1,
        )
        gate = self.gate(torch.cat((latest, extras), dim=-1)).sigmoid()
        blended = gate * model_log_rate + (1 - gate) * observed.unsqueeze(-1)
        return blended, gate

    def forward(self, history):
        log_rate, _ = self.rates_and_gate(history)
        return torch.expm1(log_rate.clamp_max(30)) * self.dwell_seconds


def gate_loss(model, batch, opportunity_power=0.75):
    history = batch["history"]
    captured = batch["captured"].float().reshape(-1, model.config.bands, model.config.dwells)
    elapsed = batch["elapsed_us"].reshape_as(captured)
    log_rate, gate = model.rates_and_gate(history)
    expected = torch.expm1(log_rate.clamp_max(30)) * model.dwell_seconds
    latest = history[:, -1, :-2].reshape(-1, model.config.bands, 9)
    # Train the gate on states where the actual policy exploits. Forced probes
    # remain controlled by the coverage rule and have no learned gate decision.
    eligible = latest[..., 0].all(-1) & (latest[..., 1].amax(-1) < 0.05)
    truth_rate = captured[..., -1] / elapsed[..., -1]
    best = truth_rate.amax(-1, keepdim=True)
    active = eligible & (best[:, 0] > 0)
    opportunity = (1 + captured[..., -1].amax(-1)).pow(opportunity_power)
    weights = active * opportunity
    probabilities = (
        (log_rate[..., -1] + model.dwell_seconds[-1].log() - elapsed[..., -1].log()) / 0.25
    ).softmax(-1)
    scaled_truth = truth_rate / best.clamp_min(1e-12)
    regret = (probabilities * (1 - scaled_truth)).sum(-1)
    ranking = (regret * weights).sum() / weights.sum().clamp_min(1)
    count_per_row = F.smooth_l1_loss(
        torch.log1p(expected), torch.log1p(captured), reduction="none"
    ).mean((1, 2))
    count = (count_per_row * weights).sum() / weights.sum().clamp_min(1)
    prior = (gate - 0.25).square().mean()
    loss = ranking + 0.2 * count + 0.01 * prior
    return loss, torch.stack(
        (ranking.detach(), count.detach(), gate.mean().detach(), active.float().mean())
    )
