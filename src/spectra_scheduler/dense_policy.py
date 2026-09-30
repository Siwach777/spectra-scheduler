"""Passive scheduling from a frozen dense action-value model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .dense_value import DenseValueConfig, DenseValueNetwork
from .experiments.storage import load_torch, save_torch
from .pulse_replay import ReplayConfig
from .replay_env import InterfaceConfig, tune_centers, validate_specification
from .replay_training import HistoryWindow


def save_dense_model(path, model, specification, training_hashes):
    if (model.config.features, model.config.actions) != (
        specification["observation_size"],
        specification["action_count"],
    ):
        raise ValueError("dense model and replay interface differ")
    save_torch(
        path,
        {
            "version": 1,
            "config": asdict(model.config),
            "specification": specification,
            "training_hashes": list(training_hashes),
            "weights": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
    )


def load_dense_model(path, device="cuda"):
    payload = load_torch(path)
    if payload.get("version") != 1:
        raise ValueError("unsupported dense model artifact")
    model = DenseValueNetwork(DenseValueConfig(**payload["config"]))
    spec = payload["specification"]
    if (model.config.features, model.config.actions) != (
        spec["observation_size"],
        spec["action_count"],
    ):
        raise ValueError("dense artifact and interface differ")
    model.load_state_dict(payload["weights"], strict=True)
    model.to(device).eval()
    return model, spec, tuple(payload["training_hashes"])


class DenseValuePolicy:
    """Short causal coverage probes; learned long-dwell band ranking.

    The same probe/exploit allocation is used by the rate-probe control. Only
    the exploitation ranking comes from model predictions. This isolates the
    learned contribution in paired evaluation.
    """

    def __init__(self, model, specification, revisit_us=500_000, model_weight=1.0):
        if revisit_us <= 0 or not np.isfinite(revisit_us):
            raise ValueError("revisit interval must be positive and finite")
        if not np.isfinite(model_weight) or not 0 <= model_weight <= 1:
            raise ValueError("model weight must lie in [0, 1]")
        self.model = model.eval()
        self.saved_spec = specification
        self.revisit_us = revisit_us
        self.model_weight = model_weight
        self.device = next(model.parameters()).device

    def reset(self, specification, seed):
        validate_specification(self.saved_spec, specification)
        self.receiver = specification["receiver"]
        self.interface = specification["interface"]
        self.time = self.receiver["start_us"]
        self.current_band = None
        self.history = HistoryWindow(self.model.config.history_steps, self.model.config.features)
        self.short = int(np.argmin(self.interface["dwell_us"]))
        self.long = int(np.argmax(self.interface["dwell_us"]))
        self.count = len(self.interface["dwell_us"])
        self.bands = self.interface["bands"]
        self.tune_centers = tune_centers(
            ReplayConfig(**self.receiver), InterfaceConfig(**self.interface)
        )

    @torch.inference_mode()
    def act(self, observation):
        if self.time >= self.receiver["stop_us"]:
            raise RuntimeError("dense policy called after episode end")
        history = self.history.append(observation)
        inputs = torch.from_numpy(history).unsqueeze(0).to(self.device)
        counts = self.model(inputs)[0].cpu().numpy()
        return self._select(observation, counts)

    @staticmethod
    @torch.inference_mode()
    def act_batch(policies, observations):
        first = policies[0]
        if any(policy.model is not first.model for policy in policies):
            raise ValueError("batched policies must share one model")
        count = len(policies)
        if not hasattr(first.model, "_policy_host") or len(first.model._policy_host) < count:
            shape = (count, *first.history.values.shape)
            first.model._policy_host = torch.empty(shape, pin_memory=first.device.type == "cuda")
            first.model._policy_device = torch.empty(shape, device=first.device)
        host = first.model._policy_host[:count]
        for row, (policy, observation) in enumerate(zip(policies, observations, strict=True)):
            host[row].copy_(torch.from_numpy(policy.history.append(observation)))
        device = first.model._policy_device[:count]
        device.copy_(host, non_blocking=first.device.type == "cuda")
        counts = first.model(device).cpu().numpy()
        return [
            policy._select(observation, predicted)
            for policy, observation, predicted in zip(policies, observations, counts, strict=True)
        ]

    def _select(self, observation, counts):
        features = observation[:-2].reshape(self.bands, -1)
        duration = self.receiver["stop_us"] - self.receiver["start_us"]
        ages = features[:, 1] * duration
        unseen = features[:, 0] == 0
        if unseen.any():
            band = int(np.argmax(np.where(unseen, ages, -np.inf)))
            dwell = self.short
        elif ages.max() >= self.revisit_us:
            band, dwell = int(ages.argmax()), self.short
        else:
            long_counts = counts.reshape(self.bands, self.count)[:, self.long]
            if self.model_weight < 1:
                # The receiver's latest measured pulse rate is a causal,
                # calibrated control. Blend in log-rate space so predictions
                # can correct it without replacing it wholesale.
                observed_log_rate = features[:, 3] * 20.0
                model_log_rate = np.log1p(
                    np.maximum(long_counts, 0)
                    / (self.interface["dwell_us"][self.long] * 1e-6)
                )
                log_rate = (
                    self.model_weight * model_log_rate
                    + (1 - self.model_weight) * observed_log_rate
                )
                long_counts = np.expm1(np.minimum(log_rate, 30.0)) * (
                    self.interface["dwell_us"][self.long] * 1e-6
                )
            delay = np.full(self.bands, self.receiver["retune_us"], dtype=np.float64)
            if self.current_band is None:
                delay.fill(0)
            else:
                slew = self.receiver["slew_mhz_per_us"]
                if slew is not None:
                    distance = abs(self.tune_centers - self.tune_centers[self.current_band])
                    delay = np.maximum(delay, distance / slew)
                delay[self.current_band] = 0
            remaining = self.receiver["stop_us"] - self.time
            elapsed = np.minimum(delay + self.interface["dwell_us"][self.long], remaining)
            band = int(np.argmax(long_counts / np.maximum(elapsed, 1)))
            dwell = self.long
        action = band * self.count + dwell
        delay = 0.0
        if self.current_band is not None and band != self.current_band:
            delay = self.receiver["retune_us"]
            slew = self.receiver["slew_mhz_per_us"]
            if slew is not None:
                delay = max(
                    delay,
                    abs(self.tune_centers[band] - self.tune_centers[self.current_band]) / slew,
                )
        self.time += min(
            delay + self.interface["dwell_us"][dwell], self.receiver["stop_us"] - self.time
        )
        self.current_band = band
        return action


def dense_spec(path, name="dense-value", **kwargs):
    from functools import partial

    from .policy_benchmark import PolicySpec, fingerprint

    path = Path(path)
    digest = fingerprint(path)
    model, specification, trained_on = load_dense_model(path, "cuda")
    if fingerprint(path) != digest:
        raise ValueError("dense model artifact changed during load")
    return PolicySpec(
        name,
        partial(DenseValuePolicy, model, specification, **kwargs),
        f"sha256:{digest};policy:{sorted(kwargs.items())}",
        trained_on,
    )
