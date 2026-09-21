"""Optional Torch reference predictor and adapters; no automatic training runs."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .evaluation_contract import Decision, Forecast
from .experiments.storage import load_torch, save_torch
from .replay_env import validate_specification
from .replay_training import HistoryWindow


@dataclass(frozen=True)
class ModelConfig:
    features: int = 74
    actions: int = 24
    history_steps: int = 8
    hidden: int = 64
    time_bins: int = 16

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("model dimensions must be positive integers")


class ForecastNetwork(nn.Module):
    """One batched recurrent encoder, all-action timing and ratio heads.

    time_bins event categories plus a no-intercept category model censoring
    explicitly. Hit probability is derived from this distribution, so its
    probability and timing heads cannot disagree about event mass.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.GRU(config.features, config.hidden, batch_first=True)
        self.time_head = nn.Linear(config.hidden, config.actions * (config.time_bins + 1))
        self.ratio_head = nn.Linear(config.hidden, config.actions)

    def forward(self, history):
        if history.ndim != 3 or tuple(history.shape[1:]) != (
            self.config.history_steps,
            self.config.features,
        ):
            raise ValueError("history shape differs from model contract")
        _, hidden = self.encoder(history)
        state = hidden[-1]
        timing = self.time_head(state).reshape(-1, self.config.actions, self.config.time_bins + 1)
        return timing, self.ratio_head(state).sigmoid()


def prediction_loss(model, batch, device="cpu"):
    """Supervised selected-action loss; censoring retained and empty ratios masked.

    Reward, elapsed-time discounts and termination are preserved by the collector
    for future RL adapters; they are not used in this supervised objective.
    """
    if batch["time_bins"] != model.config.time_bins:
        raise ValueError("batch time bins differ from model contract")
    history = torch.as_tensor(batch["history"], device=device)
    actions = torch.as_tensor(batch["action"], device=device)
    classes = torch.as_tensor(batch["time_class"], device=device)
    ratios = torch.as_tensor(batch["ratio"], device=device)
    valid = torch.as_tensor(batch["ratio_valid"], device=device)
    if not len(history) or not torch.isfinite(history).all() or not torch.isfinite(ratios).all():
        raise ValueError("empty or nonfinite training batch")
    if any(tuple(v.shape) != (len(history),) for v in (actions, classes, ratios, valid)):
        raise ValueError("target batch dimensions differ")
    if actions.dtype != torch.int64 or classes.dtype != torch.int64 or valid.dtype != torch.bool:
        raise ValueError("invalid training target dtypes")
    if ((actions < 0) | (actions >= model.config.actions)).any():
        raise ValueError("training action outside model action space")
    if ((classes < 0) | (classes > model.config.time_bins)).any():
        raise ValueError("training time bin outside model contract")
    if ((ratios < 0) | (ratios > 1)).any():
        raise ValueError("ratio target outside [0,1]")
    timing, predicted_ratios = model(history)
    rows = torch.arange(len(history), device=device)
    time_loss = F.cross_entropy(timing[rows, actions], classes)
    errors = (predicted_ratios[rows, actions] - ratios).square()
    ratio_loss = (errors * valid).sum() / valid.sum().clamp_min(1)
    return time_loss + ratio_loss, {"timing": time_loss.detach(), "ratio": ratio_loss.detach()}


def optimization_step(model, optimizer, batch):
    """Explicit learner hook; callers own sampling, validation and training budgets."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, components = prediction_loss(model, batch, next(model.parameters()).device)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
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

    def reset(self, specification, seed):
        validate_specification(self.saved_spec, specification)
        self.receiver = specification["receiver"]
        self.interface = specification["interface"]
        self.time = self.receiver["start_us"]
        self.current_band = None
        self.history = HistoryWindow(self.model.config.history_steps, self.model.config.features)
        dwells = self.interface["dwell_us"]
        self.bands = np.repeat(np.arange(self.interface["bands"]), len(dwells))
        self.dwells = np.tile(dwells, self.interface["bands"])
        self.centers = (np.arange(self.model.config.time_bins) + 0.5) / self.model.config.time_bins
        self.input = torch.empty((1, *self.history.values.shape), device=self.device)

    @torch.inference_mode()
    def act(self, observation):
        if self.time >= self.receiver["stop_us"]:
            raise RuntimeError("predictor called after episode end")
        self.input[0].copy_(torch.from_numpy(self.history.append(observation)))
        timing, ratios = self.model(self.input)
        # One device transfer for all prediction heads; no per-action inference.
        predictions = torch.cat((timing[0].softmax(-1), ratios[0, :, None]), -1).cpu().numpy()
        probabilities, ratio = predictions[:, :-1], predictions[:, -1]
        delay = np.zeros(len(self.bands))
        if self.current_band is not None:
            distance = abs(self.bands - self.current_band) * self.receiver["bandwidth_mhz"]
            delay[:] = self.receiver["retune_us"]
            slew = self.receiver["slew_mhz_per_us"]
            if slew is not None:
                np.maximum(delay, distance / slew, out=delay)
            delay[self.bands == self.current_band] = 0
        remaining = self.receiver["stop_us"] - self.time
        elapsed = np.minimum(delay + self.dwells, remaining)
        ages = observation[:-2].reshape(self.interface["bands"], -1)[:, 1]
        score = ratio + self.coverage_weight * ages[self.bands]
        score -= self.retune_weight * np.minimum(delay, remaining) / elapsed
        action = int(np.argmax(score))
        p_hit = float(1 - probabilities[action, -1])
        event_mass = probabilities[action, :-1]
        seconds = elapsed[action] / 1e6
        predicted_delay = float(event_mass @ self.centers / max(p_hit, 1e-12) * seconds)
        self.time += elapsed[action]
        self.current_band = int(self.bands[action])
        return Decision(
            action, Forecast(p_hit, predicted_delay if p_hit >= 0.5 else None, float(ratio[action]))
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
