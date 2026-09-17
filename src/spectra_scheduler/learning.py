"""Bounded feedback collection, supervised fitting and seed-held-out evaluation."""

from dataclasses import asdict
from math import log, sqrt
from random import Random
from statistics import fmean, stdev

import numpy as np

from spectra_scheduler.comparison import scheduler_factories
from spectra_scheduler.learned_scheduler import (
    FEATURE_NAMES,
    HitModel,
    LearnedScheduler,
    ObservationHistory,
)
from spectra_scheduler.metrics import calculate_metrics
from spectra_scheduler.scenarios import SCENARIO_NAMES, build_scenario
from spectra_scheduler.schedulers import AdaptiveDwellScheduler, DwellSweepScheduler

TRAIN_SCENARIOS = SCENARIO_NAMES[:-1]


class FeedbackReservoir:
    """Uniform Algorithm-R sample with memory independent of simulated duration."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 2:
            raise ValueError("max_examples must be at least two")
        self.x = np.empty((capacity, len(FEATURE_NAMES)), dtype=np.float64)
        self.y = np.empty(capacity, dtype=np.int8)
        self.seen = 0
        self.random = Random(seed)

    def add(self, features, target):
        index = self.seen if self.seen < len(self.y) else self.random.randrange(self.seen + 1)
        self.seen += 1
        if index < len(self.y):
            self.x[index], self.y[index] = features, target

    def arrays(self):
        size = min(self.seen, len(self.y))
        return self.x[:size], self.y[:size]


class FeedbackCollector:
    def __init__(self, policy, reservoir):
        self.policy, self.reservoir = policy, reservoir

    def reset(self, num_bands):
        self.policy.reset(num_bands)
        self.history = ObservationHistory(num_bands)

    def choose_band(self, time_step):
        band = self.policy.choose_band(time_step)
        self.pending = self.history.features(time_step, band)
        return band

    def observe(self, observation):
        if observation.listening:
            self.reservoir.add(self.pending, int(observation.hit))
        self.history.update(observation)
        self.policy.observe(observation)


def validate_experiment(runs, scenarios):
    if runs < 1 or not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("use positive runs and distinct scenarios")
    if any(name not in SCENARIO_NAMES for name in scenarios):
        raise ValueError("unknown scenario")


def train_model(runs=100, seed=0, scenarios=TRAIN_SCENARIOS, max_examples=50000):
    from sklearn import __version__ as sklearn_version
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    validate_experiment(runs, scenarios)
    reservoir = FeedbackReservoir(max_examples, seed)
    for name in scenarios:
        for current_seed in range(seed, seed + runs):
            simulation = build_scenario(name, current_seed)
            truth = simulation.generate_truth()
            start = current_seed % simulation.num_bands
            policies = (
                AdaptiveDwellScheduler(start_band=start),
                DwellSweepScheduler(dwell_steps=4, start_band=start),
            )
            for policy in policies:
                simulation.run(FeedbackCollector(policy, reservoir), truth=truth)
    x, y = reservoir.arrays()
    if len(np.unique(y)) != 2:
        raise ValueError("training requires both observed hits and misses")
    with threadpool_limits(limits=1):
        estimator = LogisticRegression(C=1.0, max_iter=500).fit(x, y)
    if int(estimator.n_iter_[0]) >= 500:
        raise ValueError("model did not converge; no artifact produced")
    return HitModel(
        tuple(float(v) for v in estimator.coef_[0]),
        float(estimator.intercept_[0]),
        {
            "seed": seed,
            "runs": runs,
            "scenarios": list(scenarios),
            "collection_policies": ["adaptive-dwell", "four-step-dwell"],
            "examples_seen": reservoir.seen,
            "examples_used": len(y),
            "max_examples": max_examples,
            "observed_hit_prior": float(y.mean()),
            "estimator": "LogisticRegression",
            "C": 1.0,
            "max_iter": 500,
            "sklearn_version": sklearn_version,
            "numpy_version": np.__version__,
            "target": "observed hit, including false alarms; listening steps only",
        },
    )


def evaluate_model(model, runs=50, seed=10000, scenarios=SCENARIO_NAMES):
    validate_experiment(runs, scenarios)
    provenance = model.training
    try:
        train_seed, train_runs = provenance["seed"], provenance["runs"]
        prior = provenance["observed_hit_prior"]
        train_scenarios = provenance["scenarios"]
        if (
            type(train_seed) is not int
            or type(train_runs) is not int
            or train_runs < 1
            or not 0 < prior < 1
            or not isinstance(train_scenarios, list)
        ):
            raise ValueError("invalid training provenance")
    except (KeyError, TypeError) as error:
        raise ValueError("missing or invalid training provenance") from error
    if max(seed, train_seed) < min(seed + runs, train_seed + train_runs):
        raise ValueError("evaluation seeds overlap training seeds")
    constant = HitModel(
        (0.0,) * len(FEATURE_NAMES), log(prior / (1 - prior)), provenance, model.policy
    )
    report = {
        "schema_version": 1,
        "model_sha256": model.fingerprint,
        "training": provenance,
        "seed": seed,
        "runs": runs,
        "scenarios": {},
    }
    for name in scenarios:
        results = {}
        errors, constant_errors = [], []
        for current_seed in range(seed, seed + runs):
            simulation = build_scenario(name, current_seed)
            truth = simulation.generate_truth()
            factories = scheduler_factories(current_seed)
            factories.update(
                {
                    "learned": lambda: LearnedScheduler(model),
                    "constant-model": lambda: LearnedScheduler(constant),
                }
            )
            for strategy, factory in factories.items():
                policy = factory()
                metrics = asdict(calculate_metrics(simulation.run(policy, truth=truth)))
                results.setdefault(strategy, []).append(metrics)
                if strategy == "learned":
                    errors.extend(
                        (p - y) ** 2
                        for p, y in zip(policy.predictions, policy.targets, strict=True)
                    )
                    constant_errors.extend((prior - y) ** 2 for y in policy.targets)
        means = {
            strategy: {key: fmean(row[key] for row in rows) for key in rows[0]}
            for strategy, rows in results.items()
        }
        paired = {}
        for strategy, rows in results.items():
            if strategy == "learned":
                continue
            differences = [
                a["interception_ratio"] - b["interception_ratio"]
                for a, b in zip(results["learned"], rows, strict=True)
            ]
            mean = fmean(differences)
            margin = 1.96 * stdev(differences) / sqrt(runs) if runs > 1 else None
            paired[strategy] = {
                "mean": mean,
                "approximate_95pct_ci": [mean - margin, mean + margin]
                if margin is not None
                else None,
            }
        report["scenarios"][name] = {
            "unseen_scenario": name not in train_scenarios,
            "mean_metrics": means,
            "paired_interception_delta": paired,
            "prediction": {
                "listening_examples": len(errors),
                "brier": fmean(errors) if errors else None,
                "constant_prior_brier_same_observations": fmean(constant_errors)
                if constant_errors
                else None,
            },
        }
    return report
