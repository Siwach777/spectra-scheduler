"""Optional Torch reference predictor and adapters; no automatic training runs."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .evaluation_contract import Decision, Forecast
from .experiments.storage import load_torch, save_torch
from .pulse_replay import ReplayConfig
from .replay_env import InterfaceConfig, tune_centers, validate_specification
from .replay_training import HistoryWindow


@dataclass(frozen=True)
class ModelConfig:
    features: int = 74
    actions: int = 24
    history_steps: int = 8
    hidden: int = 64
    time_bins: int = 16
    bands: int = 0  # Zero retains compatibility with existing global-head artifacts.
    encoder: str = "gru"

    def __post_init__(self):
        if any(
            type(v) is not int or v < 1
            for k, v in asdict(self).items()
            if k not in ("bands", "encoder")
        ):
            raise ValueError("model dimensions must be positive integers")
        if type(self.bands) is not int or self.bands < 0:
            raise ValueError("bands must be a nonnegative integer")
        if self.bands and (self.features != self.bands * 9 + 2 or self.actions % self.bands):
            raise ValueError("band-shared model dimensions differ from replay features")
        if self.encoder not in ("gru", "mlp", "tcn"):
            raise ValueError("unknown sequence encoder")


class TemporalBlock(nn.Module):
    """Residual dilated causal convolution; only left padding is used."""

    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.first = nn.Conv1d(width, width, 3, dilation=dilation)
        self.second = nn.Conv1d(width, width, 3, dilation=dilation)

    def forward(self, value):
        hidden = F.silu(self.first(F.pad(value, (self.padding, 0))))
        return F.silu(value + self.second(F.pad(hidden, (self.padding, 0))))


class ForecastNetwork(nn.Module):
    """One batched recurrent encoder, all-action timing and ratio heads.

    time_bins event categories plus a no-intercept category model censoring
    explicitly. Hit probability is derived from this distribution, so its
    probability and timing heads cannot disagree about event mass.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        inputs = 20 if config.bands else config.features
        actions = config.actions // config.bands if config.bands else config.actions
        if config.encoder == "gru":
            self.encoder = nn.GRU(inputs, config.hidden, batch_first=True)
        elif config.encoder == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(inputs, config.hidden),
                nn.SiLU(),
                nn.Linear(config.hidden, config.hidden),
                nn.SiLU(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(inputs, config.hidden, 1),
                TemporalBlock(config.hidden, 1),
                TemporalBlock(config.hidden, 2),
                TemporalBlock(config.hidden, 4),
            )
        self.time_head = nn.Linear(config.hidden, actions * (config.time_bins + 1))
        self.ratio_head = nn.Linear(config.hidden, actions)
        self.register_buffer("band_indices", torch.arange(config.bands), persistent=False)

    def forward(self, history):
        if history.ndim != 3 or tuple(history.shape[1:]) != (
            self.config.history_steps,
            self.config.features,
        ):
            raise ValueError("history shape differs from model contract")
        batch = len(history)
        if self.config.bands:
            # Shared weights and symmetric context prohibit absolute-frequency priors.
            # All bands are encoded in one batched GRU call, not a Python loop.
            bands = self.config.bands
            local = history[..., :-2].reshape(batch, self.config.history_steps, bands, 9)
            context = local.mean(2, keepdim=True).expand_as(local)
            progress = history[..., -2, None, None].expand(-1, -1, bands, -1)
            current = history[..., -1, None] * max(1, bands - 1)
            valid_history = local[..., 0].sum(-1, keepdim=True) > 0
            tuned = ((current - self.band_indices).abs().lt(0.1) & valid_history).unsqueeze(-1)
            history = torch.cat((local, context, progress, tuned), -1)
            history = history.transpose(1, 2).reshape(batch * bands, self.config.history_steps, 20)
        if self.config.encoder == "gru":
            _, hidden = self.encoder(history)
            state = hidden[-1]
        elif self.config.encoder == "mlp":
            state = self.encoder(history[:, -1])
        else:
            state = self.encoder(history.transpose(1, 2))[:, :, -1]
        timing = self.time_head(state).reshape(
            batch, self.config.actions, self.config.time_bins + 1
        )
        return timing, self.ratio_head(state).sigmoid().reshape(batch, self.config.actions)


def prediction_loss(model, batch, device="cpu", *, validated=False):
    """Supervised selected-action loss; censoring retained and empty ratios masked.

    Reward, elapsed-time discounts and termination are preserved by the collector
    for future RL adapters; they are not used in this supervised objective.
    """
    if batch["time_bins"] != model.config.time_bins:
        raise ValueError("batch time bins differ from model contract")
    # Validate on the host before transfer: avoid repeated CUDA synchronizations.
    history = torch.as_tensor(batch["history"])
    actions = torch.as_tensor(batch["action"])
    classes = torch.as_tensor(batch["time_class"])
    ratios = torch.as_tensor(batch["ratio"])
    valid = torch.as_tensor(batch["ratio_valid"])
    if not validated and (
        not len(history) or not torch.isfinite(history).all() or not torch.isfinite(ratios).all()
    ):
        raise ValueError("empty or nonfinite training batch")
    if any(tuple(v.shape) != (len(history),) for v in (actions, classes, ratios, valid)):
        raise ValueError("target batch dimensions differ")
    if actions.dtype != torch.int64 or classes.dtype != torch.int64 or valid.dtype != torch.bool:
        raise ValueError("invalid training target dtypes")
    if not validated and ((actions < 0) | (actions >= model.config.actions)).any():
        raise ValueError("training action outside model action space")
    if not validated and ((classes < 0) | (classes > model.config.time_bins)).any():
        raise ValueError("training time bin outside model contract")
    if not validated and ((ratios < 0) | (ratios > 1)).any():
        raise ValueError("ratio target outside [0,1]")
    history, actions, classes, ratios, valid = (
        value.to(device) for value in (history, actions, classes, ratios, valid)
    )
    timing, predicted_ratios = model(history)
    rows = torch.arange(len(history), device=device)
    time_loss = F.cross_entropy(timing[rows, actions], classes)
    errors = (predicted_ratios[rows, actions] - ratios).square()
    ratio_loss = (errors * valid).sum() / valid.sum().clamp_min(1)
    return time_loss + ratio_loss, {"timing": time_loss.detach(), "ratio": ratio_loss.detach()}


def optimization_step(model, optimizer, batch, *, validated=False):
    """Explicit learner hook; callers own sampling, validation and training budgets."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, components = prediction_loss(
        model, batch, next(model.parameters()).device, validated=validated
    )
    torch._assert_async(torch.isfinite(loss), "nonfinite training loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    torch._assert_async(torch.isfinite(norm), "nonfinite gradient norm")
    optimizer.step()
    return components


def save_predictor(path, model, specification, training_hashes=()):
    """Atomic weights-only artifact. No pickle-defined model classes or optimizers."""
    if (
        model.config.features != specification["observation_size"]
        or model.config.actions != specification["action_count"]
    ):
        raise ValueError("model dimensions differ from replay specification")
    payload = {
        "schema_version": 1,
        "config": asdict(model.config),
        "specification": specification,
        "training_hashes": list(training_hashes),
        "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    save_torch(path, payload)


def load_predictor(path, device="cpu"):
    payload = load_torch(path)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported predictor artifact")
    model = ForecastNetwork(ModelConfig(**payload["config"]))
    spec = payload["specification"]
    if (model.config.features, model.config.actions) != (
        spec["observation_size"],
        spec["action_count"],
    ):
        raise ValueError("artifact model/specification mismatch")
    model.load_state_dict(payload["weights"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite predictor weights")
    model.to(device).eval()
    return model, spec, tuple(payload["training_hashes"])


class PredictorPolicy:
    """Reference policy scoring all actions in one network call.

    Coverage bonus and retune cost are declared policy choices, not learned claims.
    The public clock and previous tuning are reconstructed from executed actions;
    no simulator truth enters the model or action selection.
    """

    def __init__(
        self,
        path,
        device="cpu",
        coverage_weight=0.05,
        retune_weight=0.05,
        threads=1,
        expected_sha256=None,
        revisit_us=500_000,
        constant_predictions=False,
        observed_rate=False,
    ):
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)
        if expected_sha256 is not None:
            from .policy_benchmark import fingerprint

            if fingerprint(path) != expected_sha256:
                raise ValueError("predictor checkpoint content changed")
        if (
            not np.isfinite([coverage_weight, retune_weight]).all()
            or min(coverage_weight, retune_weight) < 0
        ):
            raise ValueError("policy weights must be nonnegative and finite")
        self.model, self.saved_spec, self.training_hashes = load_predictor(path, device)
        if expected_sha256 is not None and fingerprint(path) != expected_sha256:
            raise ValueError("predictor checkpoint content changed during load")
        self.device = torch.device(device)
        self.coverage_weight, self.retune_weight = coverage_weight, retune_weight
        if not np.isfinite(revisit_us) or revisit_us <= 0:
            raise ValueError("revisit interval must be positive and finite")
        self.revisit_us = revisit_us
        self.constant_predictions = constant_predictions
        self.observed_rate = observed_rate

    def reset(self, specification, seed):
        validate_specification(self.saved_spec, specification)
        self.receiver = specification["receiver"]
        self.interface = specification["interface"]
        self.time = self.receiver["start_us"]
        self.current_band = None
        self.history = HistoryWindow(self.model.config.history_steps, self.model.config.features)
        dwells = self.interface["dwell_us"]
        self.bands = np.repeat(np.arange(self.interface["bands"]), len(dwells))
        self.tune_centers = tune_centers(
            ReplayConfig(**self.receiver), InterfaceConfig(**self.interface)
        )
        self.dwells = np.tile(dwells, self.interface["bands"])
        self.centers = (np.arange(self.model.config.time_bins) + 0.5) / self.model.config.time_bins
        self.input = torch.empty((1, *self.history.values.shape), device=self.device)

    @torch.inference_mode()
    def act(self, observation):
        if self.time >= self.receiver["stop_us"]:
            raise RuntimeError("predictor called after episode end")
        if self.constant_predictions or self.observed_rate:
            # Matched scheduling-rule ablation; no claim of calibrated forecasts.
            predictions = np.full(
                (len(self.bands), self.model.config.time_bins + 2),
                1 / (self.model.config.time_bins + 1),
            )
            predictions[:, -1] = 0.5
        else:
            self.input[0].copy_(torch.from_numpy(self.history.append(observation)))
            timing, ratios = self.model(self.input)
            # One device transfer for all prediction heads; no per-action inference.
            predictions = torch.cat((timing[0].softmax(-1), ratios[0, :, None]), -1).cpu().numpy()
        return self._select(observation, predictions)

    @staticmethod
    @torch.inference_mode()
    def act_batch(policies, observations):
        """One CUDA call for independent episodes sharing immutable model weights."""
        first = policies[0]
        if first.constant_predictions or first.observed_rate:
            return [p.act(o) for p, o in zip(policies, observations, strict=True)]
        count = len(policies)
        if not hasattr(first.model, "_inference_host") or len(first.model._inference_host) < count:
            shape = (count, *first.history.values.shape)
            first.model._inference_host = torch.empty(shape, pin_memory=first.device.type == "cuda")
            first.model._inference_device = torch.empty(shape, device=first.device)
        host = first.model._inference_host[:count]
        for row, (policy, observation) in enumerate(zip(policies, observations, strict=True)):
            if policy.model is not first.model:
                raise ValueError("batched policies must share model weights")
            host[row].copy_(torch.from_numpy(policy.history.append(observation)))
        device = first.model._inference_device[:count]
        device.copy_(host, non_blocking=True)
        timing, ratios = first.model(device)
        # Completion of this transfer also makes host staging safe to reuse next call.
        predictions = torch.cat((timing.softmax(-1), ratios[..., None]), -1).cpu().numpy()
        return [
            p._select(o, v) for p, o, v in zip(policies, observations, predictions, strict=True)
        ]

    def _select(self, observation, predictions):
        if self.time >= self.receiver["stop_us"]:
            raise RuntimeError("predictor called after episode end")
        probabilities, ratio = predictions[:, :-1], predictions[:, -1]
        delay = np.zeros(len(self.bands))
        if self.current_band is not None:
            distance = abs(self.tune_centers[self.bands] - self.tune_centers[self.current_band])
            delay[:] = self.receiver["retune_us"]
            slew = self.receiver["slew_mhz_per_us"]
            if slew is not None:
                np.maximum(delay, distance / slew, out=delay)
            delay[self.bands == self.current_band] = 0
        remaining = self.receiver["stop_us"] - self.time
        elapsed = np.minimum(delay + self.dwells, remaining)
        features = observation[:-2].reshape(self.interface["bands"], -1)
        if self.observed_rate:
            rates = np.expm1(features[:, 3] * 20)
            visited = features[:, 0] > 0
            rates[~visited] = rates[visited].mean() if visited.any() else 1
            ratio = rates[self.bands] / max(float(rates.sum()), 1e-12)
        ages = features[:, 1] * (self.receiver["stop_us"] - self.receiver["start_us"])
        score = ratio + self.coverage_weight * ages[self.bands] / self.revisit_us
        score -= self.retune_weight * np.minimum(delay, remaining) / elapsed
        # Search unseen bands first; then prioritize the oldest overdue band.
        # This is a scheduling constraint, not evidence of learned exploration.
        unseen = features[:, 0] == 0
        if unseen.any() or ages.max() >= self.revisit_us:
            priority = np.where(unseen, ages, -np.inf) if unseen.any() else ages
            band = int(np.argmax(priority))
            score[self.bands != band] = -np.inf
        # Exact ties prefer the longest dwell, avoiding gratuitous short actions.
        best = np.flatnonzero(score == score.max())
        action = int(best[np.argmax(self.dwells[best])])
        p_hit = float(1 - probabilities[action, -1])
        event_mass = probabilities[action, :-1]
        seconds = elapsed[action] / 1e6
        predicted_delay = float(event_mass @ self.centers / max(p_hit, 1e-12) * seconds)
        self.time += elapsed[action]
        self.current_band = int(self.bands[action])
        return Decision(
            action,
            None
            if self.constant_predictions or self.observed_rate
            else Forecast(p_hit, predicted_delay if p_hit >= 0.5 else None, float(ratio[action])),
        )


def predictor_spec(path, name="predictor", **kwargs):
    """Register a checkpoint with the generic benchmark and training provenance."""
    from functools import partial

    from .policy_benchmark import PolicySpec, fingerprint

    digest = fingerprint(path)
    _, _, trained_on = load_predictor(path)
    if digest != fingerprint(path):
        raise ValueError("predictor checkpoint content changed during registration")
    return PolicySpec(
        name,
        partial(PredictorPolicy, str(path), expected_sha256=digest, **kwargs),
        f"sha256:{digest};policy:{sorted(kwargs.items())}",
        trained_on,
    )
