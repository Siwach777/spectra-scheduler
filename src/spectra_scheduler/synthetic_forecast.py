"""Causal GRU forecasts for synthetic band-and-dwell experiments."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .evaluation_contract import Forecast
from .forecast_model import ForecastNetwork, ModelConfig, prediction_loss
from .scenarios import (
    REQUIREMENT_SCENARIO_VERSION,
    REQUIREMENT_SCENARIOS,
    build_requirement_scenario,
)
from .simulation import SimulationEpisode, SyntheticAction

DWELLS = (1, 8, 32)
HISTORY = 8
TIME_BINS = 8
FEATURES_PER_BAND = 5
MODEL_VERSION = 1


class CausalFeatures:
    def __init__(self, bands):
        self.bands = bands
        self.visits = np.zeros(bands, np.float32)
        self.hits = np.zeros(bands, np.float32)
        self.last_listen = np.full(bands, -1, np.int32)
        self.last_hit = np.full(bands, -1, np.int32)
        self.recent = [deque(maxlen=8) for _ in range(bands)]
        self.history = np.zeros((HISTORY, bands * FEATURES_PER_BAND + 2), np.float32)
        self.previous_band = -1

    def frame(self, step, horizon):
        frame = np.zeros(self.history.shape[1], np.float32)
        for band in range(self.bands):
            offset = band * FEATURES_PER_BAND
            seen = self.last_listen[band] >= 0
            frame[offset] = float(seen)
            frame[offset + 1] = min(step - self.last_listen[band], 128) / 128 if seen else 1
            frame[offset + 2] = min(self.visits[band], 32) / 32
            frame[offset + 3] = sum(self.recent[band]) / max(len(self.recent[band]), 1)
            hit_seen = self.last_hit[band] >= 0
            frame[offset + 4] = min(step - self.last_hit[band], 128) / 128 if hit_seen else 1
        frame[-2] = step / horizon
        frame[-1] = self.previous_band / max(self.bands - 1, 1)
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame
        return self.history

    def observe(self, observation):
        self.previous_band = observation.band
        if observation.listening:
            band = observation.band
            self.visits[band] += 1
            self.hits[band] += int(observation.hit)
            self.last_listen[band] = observation.time_step
            self.recent[band].append(int(observation.hit))
            if observation.hit:
                self.last_hit[band] = observation.time_step


class SyntheticGRUPolicy:
    """Evaluate one causal model call per macro decision."""

    def __init__(
        self,
        checkpoint,
        *,
        device="cuda",
        timing_weight=0.5,
        ratio_weight=0.25,
        coverage_weight=0.15,
        retune_weight=0.05,
        empirical_weight=0.5,
        coverage_limit=128,
        ablation=None,
    ):
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("synthetic neural inference requires CUDA")
        if ablation not in (None, "constant", "observed-rate", "no-timing"):
            raise ValueError("unknown policy ablation")
        weights = (timing_weight, ratio_weight, coverage_weight, retune_weight, empirical_weight)
        if not np.isfinite(weights).all() or min(weights) < 0 or empirical_weight > 1:
            raise ValueError("invalid policy weights")
        if type(coverage_limit) is not int or coverage_limit < 1:
            raise ValueError("invalid coverage limit")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload["version"] != MODEL_VERSION
            or tuple(payload["dwells"]) != DWELLS
            or payload.get("requirement_scenario_version") != REQUIREMENT_SCENARIO_VERSION
        ):
            raise ValueError("incompatible synthetic forecast checkpoint")
        self.model = ForecastNetwork(ModelConfig(**payload["config"])).to(device).eval()
        self.model.load_state_dict(payload["weights"])
        if self.model.config.encoder != "gru":
            raise ValueError("synthetic policy requires a GRU checkpoint")
        self.device = device
        self.weights = weights
        self.coverage_limit = coverage_limit
        self.ablation = ablation

    def set_retune_table(self, table):
        self.retune = np.asarray(table, np.int32)

    def set_episode_horizon(self, steps):
        self.horizon = steps

    def reset(self, num_bands):
        if self.model.config.features != num_bands * FEATURES_PER_BAND + 2:
            raise ValueError("checkpoint feature width differs from receiver")
        if self.model.config.actions != num_bands * len(DWELLS):
            raise ValueError("checkpoint action count differs from receiver")
        if self.retune.shape != (num_bands, num_bands):
            raise ValueError("receiver retune schedule is missing")
        self.features = CausalFeatures(num_bands)
        self.num_bands = num_bands
        self.pending = None

    @torch.inference_mode()
    def choose_action(self, time_step):
        history = self.features.frame(time_step, self.horizon)
        count = self.num_bands * len(DWELLS)
        if self.ablation in ("constant", "observed-rate"):
            probabilities = np.zeros((count, TIME_BINS + 1))
            probabilities[:, -1] = 0.5
            probabilities[:, :-1] = 0.5 / TIME_BINS
            ratios = np.full(count, 0.5)
            if self.ablation == "observed-rate":
                rates = (self.features.hits + 1) / (self.features.visits + 2)
                probabilities[:, -1] = np.repeat(1 - rates, len(DWELLS))
                probabilities[:, :-1] = np.repeat(rates / TIME_BINS, len(DWELLS))[:, None]
                ratios = np.repeat(rates, len(DWELLS))
        else:
            x = torch.as_tensor(history.copy(), device=self.device).unsqueeze(0)
            timing, ratio = self.model(x)
            probabilities = timing[0].softmax(-1).cpu().numpy()
            ratios = ratio[0].cpu().numpy()
        bands = np.repeat(np.arange(self.num_bands), len(DWELLS))
        dwells = np.tile(DWELLS, self.num_bands)
        previous = self.features.previous_band
        retune = self.retune[previous, bands] if previous >= 0 else np.zeros(count)
        elapsed = np.minimum(retune + dwells, self.horizon - time_step)
        p_hit = 1 - probabilities[:, -1]
        deadline = probabilities[:, : max(1, TIME_BINS // 4)].sum(-1)
        age = np.where(
            self.features.last_listen >= 0,
            time_step - self.features.last_listen,
            128,
        )
        timing_weight, ratio_weight, coverage_weight, retune_weight, empirical_weight = self.weights
        if self.ablation == "no-timing":
            timing_weight = 0
        rates = (self.features.hits + 1) / (self.features.visits + 2)
        score = (
            (1 - empirical_weight) * p_hit
            + empirical_weight * np.repeat(rates, len(DWELLS))
            + timing_weight * deadline
            + ratio_weight * ratios
            + coverage_weight * np.minimum(age[bands], 128) / 128
            - retune_weight * retune / np.maximum(elapsed, 1)
        )
        overdue = np.flatnonzero(age >= self.coverage_limit)
        if len(overdue):
            target = int(overdue[np.argmax(age[overdue])])
            index = target * len(DWELLS) + 1
        else:
            index = int(np.argmax(score))
        band, dwell = divmod(index, len(DWELLS))
        self.pending = (
            time_step,
            index,
            probabilities[index].copy(),
            float(ratios[index]),
            float(elapsed[index]),
        )
        return SyntheticAction(band, DWELLS[dwell])

    def forecast(self, time_step, action):
        start, index, probabilities, ratio, elapsed = self.pending
        expected = action.band * len(DWELLS) + DWELLS.index(action.dwell_steps)
        if start != time_step or index != expected:
            raise ValueError("forecast requested for a different action")
        if self.ablation in ("constant", "observed-rate"):
            return None
        p_hit = float(1 - probabilities[-1])
        centers = (np.arange(TIME_BINS) + 0.5) / TIME_BINS
        delay = float(probabilities[:-1] @ centers / max(p_hit, 1e-12) * elapsed * 0.001)
        return Forecast(p_hit, delay if p_hit >= 0.5 else None, ratio)

    def observe(self, observation):
        self.features.observe(observation)


def collect_examples(worlds, seed, limit=50_000):
    """Bounded examples with post-action truth labels and pre-action features."""
    rng = np.random.default_rng(seed)
    histories, actions, classes, ratios, valid = [], [], [], [], []
    for scenario in REQUIREMENT_SCENARIOS:
        for world_index in range(worlds):
            world = build_requirement_scenario(scenario, seed * 1_000_000 + world_index)
            episode = SimulationEpisode(world)
            features = CausalFeatures(world.num_bands)
            while episode.time_step < world.duration:
                start = episode.time_step
                history = features.frame(start, world.duration).copy()
                band = int(rng.integers(world.num_bands))
                dwell_index = int(rng.integers(len(DWELLS)))
                observations = episode.step_action(SyntheticAction(band, DWELLS[dwell_index]))
                records = episode.records[-len(observations) :]
                first = next(
                    (r.observation.time_step for r in records if r.detected_emitters), None
                )
                truth = sum(
                    len(episode.events.get((t, b), ()))
                    for t in range(start, episode.time_step)
                    for b in range(world.num_bands)
                )
                captures = sum(len(r.detected_emitters) for r in records)
                elapsed = len(observations)
                histories.append(history)
                actions.append(band * len(DWELLS) + dwell_index)
                classes.append(
                    TIME_BINS
                    if first is None
                    else min(TIME_BINS - 1, (first - start) * TIME_BINS // elapsed)
                )
                ratios.append(captures / max(truth, 1))
                valid.append(truth > 0)
                for observation in observations:
                    features.observe(observation)
                if len(histories) >= limit:
                    break
            if len(histories) >= limit:
                break
        if len(histories) >= limit:
            break
    return {
        "history": np.stack(histories),
        "action": np.asarray(actions, np.int64),
        "time_class": np.asarray(classes, np.int64),
        "ratio": np.asarray(ratios, np.float32),
        "ratio_valid": np.asarray(valid, np.bool_),
        "time_bins": TIME_BINS,
    }


def train(output, *, seed=0, worlds=80, epochs=12, batch_size=512):
    if not torch.cuda.is_available():
        raise RuntimeError("synthetic GRU training requires CUDA")
    if min(worlds, epochs, batch_size) < 1:
        raise ValueError("training budgets must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    examples = collect_examples(worlds, seed)
    config = ModelConfig(
        features=8 * FEATURES_PER_BAND + 2,
        actions=8 * len(DWELLS),
        history_steps=HISTORY,
        time_bins=TIME_BINS,
        hidden=64,
        encoder="gru",
    )
    model = ForecastNetwork(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, fused=True)
    rng = np.random.default_rng(seed)
    count = len(examples["action"])
    for epoch in range(epochs):
        losses = []
        permutation = rng.permutation(count)
        for start in range(0, count, batch_size):
            indices = permutation[start : start + batch_size]
            batch = {key: value[indices] for key, value in examples.items() if key != "time_bins"}
            batch["time_bins"] = TIME_BINS
            optimizer.zero_grad(set_to_none=True)
            loss, _ = prediction_loss(model, batch, "cuda", validated=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch} loss={np.mean(losses):.5f} samples={count}", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": MODEL_VERSION,
            "requirement_scenario_version": REQUIREMENT_SCENARIO_VERSION,
            "config": asdict(config),
            "dwells": DWELLS,
            "seed": seed,
            "worlds_per_scenario": worlds,
            "epochs": epochs,
            "samples": count,
            "weights": {key: value.cpu() for key, value in model.state_dict().items()},
        },
        output,
    )
    return output


def synthetic_gru_spec(checkpoint, name, **kwargs):
    from .experiments.storage import fingerprint
    from .policy_benchmark import PolicySpec

    digest = fingerprint(checkpoint)
    return PolicySpec(
        name,
        partial(SyntheticGRUPolicy, str(checkpoint), **kwargs),
        f"synthetic-gru-v{MODEL_VERSION}:sha256:{digest};settings:{sorted(kwargs.items())}",
    )


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worlds", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(arguments)
    train(
        args.output,
        seed=args.seed,
        worlds=args.worlds,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
