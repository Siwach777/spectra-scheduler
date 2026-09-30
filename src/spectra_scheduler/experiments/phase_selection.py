"""Selection-only diagnostics for structural phase beliefs and policy settings.

The beta phase control has no trained weights. It is an attribution control for
the timing neural model, never evidence of an offline learned improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np

from ..policy_benchmark import PolicySpec, benchmark_synthetic
from ..timing_belief import (
    BeliefConfig, BeliefPolicyConfig, TimingBeliefPolicy, TimingHistory, validated_retune_table,
)
from .storage import fingerprint, write_json
from .timing_study import ObservedRateScheduler, controls


class BetaPhaseScheduler(TimingBeliefPolicy):
    """Causal beta-binomial phase marginalization, with an aperiodic expert.

    Independent phase parameters incur the exact integrated likelihood penalty.
    Periods are generic hypotheses; emitter timing and scenario labels are absent.
    """

    def __init__(self, config=None, *, alpha=0.1, beta=1.0, temperature=1.0,
                 base_mass=0.5, quality=0.4):
        self.config = config or BeliefPolicyConfig()
        self.belief = BeliefConfig()
        self.alpha, self.beta, self.temperature = alpha, beta, temperature
        self.base_mass, self.quality = base_mass, quality
        self.periods = np.arange(2, self.belief.max_period + 1)
        self.phase_count = self.belief.max_period
        self.flat_index = ((np.arange(-self.belief.history, 0)[None] % self.periods[:, None])
                           + np.arange(len(self.periods))[:, None] * self.phase_count).ravel()
        self.future_index = np.arange(self.belief.future)[None] % self.periods[:, None]
        self.valid_phases = np.arange(self.phase_count)[None] < self.periods[:, None]
        steps = range(self.belief.history + 1)
        self.log_gamma_alpha = np.asarray([math.lgamma(k + alpha) for k in steps])
        self.log_gamma_beta = np.asarray([math.lgamma(k + beta) for k in steps])
        self.log_gamma_sum = np.asarray([math.lgamma(k + alpha + beta) for k in steps])
        self.log_beta_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        prior = 1 / self.periods
        prior = prior / prior.sum() * (1 - base_mass)
        self.log_prior = np.log(np.r_[prior, base_mass])

    def set_retune_table(self, table):
        self.retune = validated_retune_table(table, self.belief.future, self.config.dwells)

    def reset(self, bands):
        if self.retune.shape != (bands, bands):
            raise ValueError("public retune table differs from the receiver band count")
        self.history = TimingHistory(bands, self.belief.history)
        self.bands = bands
        self.pending = None
        self.decisions = self.probes = 0

    def _evidence(self, count, exposure):
        return (self.log_gamma_alpha[count] + self.log_gamma_beta[exposure - count]
                - self.log_gamma_sum[exposure] - self.log_beta_prior)

    def predict(self):
        history = self.history.encode()
        masks = history[:, 0].astype(np.int64)
        hits = ((history[:, 1] > 0) & (history[:, 2] >= self.quality)).astype(np.int64)
        outputs = np.empty((self.bands, self.belief.future))
        shape = (len(self.periods), self.phase_count)
        for band in range(self.bands):
            exposure = np.bincount(self.flat_index, weights=np.tile(masks[band], len(self.periods)),
                                   minlength=np.prod(shape)).astype(np.int64).reshape(shape)
            count = np.bincount(self.flat_index, weights=np.tile(hits[band], len(self.periods)),
                                minlength=np.prod(shape)).astype(np.int64).reshape(shape)
            rates = (count + self.alpha) / (exposure + self.alpha + self.beta)
            evidence = (self._evidence(count, exposure) * self.valid_phases).sum(-1)
            band_hits, band_exposure = int(hits[band].sum()), int(masks[band].sum())
            base_evidence = self._evidence(band_hits, band_exposure)
            logits = np.r_[evidence, base_evidence] / self.temperature + self.log_prior
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
            projected = np.take_along_axis(rates, self.future_index, axis=1)
            base_rate = (band_hits + self.alpha) / (band_exposure + self.alpha + self.beta)
            outputs[band] = weights[:-1] @ projected + weights[-1] * base_rate
        return outputs

    def choose_action(self, time_step):
        if time_step != self.history.time:
            raise ValueError("policy clock differs from causal history")
        return self.select(time_step, self.predict())

    def forecast(self, time_step, action):
        start, band, dwell, forecast = self.pending
        if (time_step, action.band, action.dwell_steps) != (start, band, dwell):
            raise ValueError("forecast requested for a different action")
        return forecast


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--grid", choices=("evidence", "coverage"), default="evidence")
    args = parser.parse_args(arguments)
    config = BeliefPolicyConfig(exploration=0.02)
    source = fingerprint(__file__)
    candidates = []
    grid = (
        ("beta-default", config, {}),
        ("beta-strong", config, {"alpha": 0.5, "beta": 4.5}),
        ("beta-cool", config, {"temperature": 0.5}),
        ("beta-warm", config, {"temperature": 2.0}),
        ("beta-base-low", config, {"base_mass": 0.05}),
        ("beta-coverage192", replace(config, revisit=192), {}),
        ("beta-long", replace(config, dwells=(8, 16, 32)), {}),
        ("beta-probe16", replace(config, probe=16), {}),
    ) if args.grid == "evidence" else (
        ("beta-coverage128", replace(config, revisit=128), {}),
        ("beta-coverage192", replace(config, revisit=192), {}),
        ("beta-coverage256", replace(config, revisit=256), {}),
        ("beta-coverage384", replace(config, revisit=384), {}),
        ("beta-192-cool", replace(config, revisit=192), {"temperature": 0.5}),
        ("beta-192-warm", replace(config, revisit=192), {"temperature": 2.0}),
        ("beta-192-probe4", replace(config, revisit=192, probe=4), {}),
        ("beta-192-probe12", replace(config, revisit=192, probe=12), {}),
    )
    for name, policy, kwargs in grid:
        candidates.append(PolicySpec(name, partial(BetaPhaseScheduler, policy, **kwargs),
            json.dumps({"policy": asdict(policy), "structural": kwargs, "source": source,
                        "learned": False}, sort_keys=True)))
    if args.grid == "coverage":
        for revisit in (128, 192, 256, 384):
            matched = replace(config, revisit=revisit)
            candidates.append(PolicySpec(f"observed-rate-{revisit}",
                partial(ObservedRateScheduler, matched), str(asdict(matched))))
    started = time.perf_counter()
    report = benchmark_synthetic(candidates + controls(config), baseline="observed-rate",
        seeds=tuple(range(2000, 2000 + args.runs)), split="val", workers=args.workers)
    report["scope"] = "selection-only development worlds; reporting and test worlds unused"
    report["evaluation_seconds"] = time.perf_counter() - started
    write_json(args.output, report)
    print(json.dumps({s: {p: m["interception_ratio"]["mean"] for p, m in v.items()}
                      for s, v in report["by_scenario"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
