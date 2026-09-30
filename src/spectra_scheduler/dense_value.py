"""Dense counterfactual capture model for passive band-and-dwell scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DenseValueConfig:
    bands: int = 8
    dwells: int = 3
    history_steps: int = 16
    hidden: int = 128
    encoder: str = "mlp"

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.bands, self.dwells, self.history_steps, self.hidden)
        ):
            raise ValueError("dense model dimensions must be positive integers")
        if self.encoder not in ("mlp", "spectral_mlp", "gru"):
            raise ValueError("dense encoder must be 'mlp', 'spectral_mlp' or 'gru'")

    @property
    def features(self) -> int:
        return self.bands * 9 + 2

    @property
    def actions(self) -> int:
        return self.bands * self.dwells


class DenseValueNetwork(nn.Module):
    """Predict nonnegative, dwell-monotone captured-pulse counts for all actions.

    Every band uses one shared encoder. The MLP consumes the *entire* causal
    history rather than the final feature frame. The GRU is an alternative
    temporal encoder under the same counterfactual training objective.
    """

    def __init__(self, config: DenseValueConfig) -> None:
        super().__init__()
        self.config = config
        if config.encoder in ("mlp", "spectral_mlp"):
            step_features = 57 if config.encoder == "spectral_mlp" else 20
            self.encoder = nn.Sequential(
                nn.Linear(config.history_steps * step_features, 2 * config.hidden),
                nn.SiLU(),
                nn.Linear(2 * config.hidden, config.hidden),
                nn.SiLU(),
            )
        else:
            self.encoder = nn.GRU(20, config.hidden, batch_first=True)
        self.count_head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.dwells),
        )
        self.register_buffer("band_indices", torch.arange(config.bands), persistent=False)

    def forward(self, history: Tensor) -> Tensor:
        cfg = self.config
        if history.ndim != 3 or tuple(history.shape[1:]) != (
            cfg.history_steps,
            cfg.features,
        ):
            raise ValueError("history shape differs from dense model contract")
        batch = history.shape[0]
        local = history[..., :-2].reshape(batch, cfg.history_steps, cfg.bands, 9)
        context = local.mean(2, keepdim=True).expand_as(local)
        progress = history[..., -2, None, None].expand(-1, -1, cfg.bands, -1)
        current = history[..., -1, None] * max(1, cfg.bands - 1)
        observed = local[..., 0].sum(-1, keepdim=True) > 0
        tuned = ((current - self.band_indices).abs() < 0.1) & observed
        if cfg.encoder == "spectral_mlp":
            # Overlapping passbands have local spectral structure. Preserve
            # each neighbor's measurements and age at every causal frame;
            # a global mean alone cannot distinguish left from right.
            spectrum = local.permute(0, 1, 3, 2).reshape(batch * cfg.history_steps, 9, cfg.bands)
            windows = F.pad(spectrum, (2, 2), mode="replicate").unfold(-1, 5, 1)
            neighborhood = windows.permute(0, 2, 3, 1).reshape(
                batch, cfg.history_steps, cfg.bands, 45
            )
            distance = ((current - self.band_indices) / max(1, cfg.bands - 1)).unsqueeze(-1)
            sequence = torch.cat(
                (neighborhood, context, progress, tuned.unsqueeze(-1), distance), dim=-1
            )
            step_features = 57
        else:
            sequence = torch.cat((local, context, progress, tuned.unsqueeze(-1)), dim=-1)
            step_features = 20
        sequence = sequence.transpose(1, 2).reshape(
            batch * cfg.bands, cfg.history_steps, step_features
        )
        if cfg.encoder in ("mlp", "spectral_mlp"):
            state = self.encoder(sequence.flatten(1))
        else:
            _, hidden = self.encoder(sequence)
            state = hidden[-1]
        increments = F.softplus(self.count_head(state))
        # A longer dwell contains every pulse captured by its shorter sibling.
        return increments.cumsum(-1).reshape(batch, cfg.actions)

    @staticmethod
    def action_scores(expected_counts: Tensor, elapsed_us: Tensor) -> Tensor:
        """Expected captured pulses per physical second, including retuning."""
        if expected_counts.shape != elapsed_us.shape or expected_counts.ndim != 2:
            raise ValueError("count and elapsed action grids must have the same shape")
        if not torch.isfinite(elapsed_us).all() or (elapsed_us <= 0).any():
            raise ValueError("each action must have positive finite elapsed time")
        return expected_counts / (elapsed_us * 1e-6)


def dense_value_loss(
    model: DenseValueNetwork,
    batch: dict,
    *,
    device: torch.device | str | None = None,
    ranking_weight: float = 1.0,
    ranking_temperature: float = 0.5,
    opportunity_power: float = 0.0,
    validated: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Robust count fit plus differentiable regret over all legal actions.

    Counterfactual targets are training-only. The count term fits all actions.
    The regret term ranks long-dwell bands, matching the exploitation decision
    made by DenseValuePolicy. Log predicted rates make this ranking invariant
    to a common rescaling of counts, so regret gradients cannot improve solely
    by inflating every count estimate. Zero-capture rows train only the count
    objective.
    """
    if ranking_weight < 0 or ranking_temperature <= 0 or not 0 <= opportunity_power <= 1:
        raise ValueError("invalid ranking loss settings")
    device = device or next(model.parameters()).device
    history = torch.as_tensor(batch["history"], device=device, dtype=torch.float32)
    captured = torch.as_tensor(batch["captured"], device=device, dtype=torch.float32)
    elapsed = torch.as_tensor(batch["elapsed_us"], device=device, dtype=torch.float32)
    if captured.shape != elapsed.shape or captured.shape != (len(history), model.config.actions):
        raise ValueError("dense counterfactual target dimensions differ")
    if not validated and (
        not torch.isfinite(history).all()
        or not torch.isfinite(captured).all()
        or not torch.isfinite(elapsed).all()
        or (captured < 0).any()
        or (elapsed <= 0).any()
    ):
        raise ValueError("dense training batch has invalid values")
    expected = model(history)
    # Pulse counts can be strongly overdispersed. A log-count Huber target
    # preserves zeros and limits domination by the busiest recording.
    count_loss = F.smooth_l1_loss(torch.log1p(expected), torch.log1p(captured))
    long_actions = torch.arange(model.config.bands, device=expected.device)
    long_actions = long_actions * model.config.dwells + model.config.dwells - 1
    long_captured = captured.index_select(1, long_actions)
    long_expected = expected.index_select(1, long_actions)
    long_elapsed = elapsed.index_select(1, long_actions)
    target_rate = long_captured / long_elapsed
    predicted_rate = long_expected / long_elapsed
    best_rate = target_rate.amax(-1, keepdim=True)
    active = best_rate[:, 0] > 0
    scaled_truth = target_rate / best_rate.clamp_min(1e-12)
    log_rate = torch.log(long_expected.clamp_min(1e-6)) - torch.log(long_elapsed)
    selection = (log_rate / ranking_temperature).softmax(-1)
    regret = (selection * (1 - scaled_truth)).sum(-1)
    # A decision at a busy recording can lose far more pulses than a decision
    # at a quiet one. Sublinear weighting gives it more influence without
    # letting a handful of high-count windows dominate the whole epoch.
    opportunity = (1 + long_captured.amax(-1)).pow(opportunity_power)
    weights = active * opportunity
    ranking_loss = (regret * weights).sum() / weights.sum().clamp_min(1)
    greedy = predicted_rate.argmax(-1, keepdim=True)
    greedy_regret = 1 - scaled_truth.gather(-1, greedy).squeeze(-1)
    greedy_regret = (greedy_regret * active).sum() / active.sum().clamp_min(1)
    loss = count_loss + ranking_weight * ranking_loss
    return loss, {
        "count": count_loss.detach(),
        "ranking_regret": ranking_loss.detach(),
        "greedy_regret": greedy_regret.detach(),
        "active_fraction": active.float().mean().detach(),
    }
