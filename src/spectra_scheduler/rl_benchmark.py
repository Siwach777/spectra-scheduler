"""Paired benchmark: structural holdouts, receiver shift, and legacy scenarios."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter

from spectra_scheduler.comparison import scheduler_factories
from spectra_scheduler.emitters import (
    BurstEmitter,
    FrequencyHoppingEmitter,
    ModeSwitchingEmitter,
    PeriodicEmitter,
    ScanningEmitter,
    SpatialScanningEmitter,
    WindowedEmitter,
)
from spectra_scheduler.learned_scheduler import HitModel, LearnedScheduler
from spectra_scheduler.metrics import calculate_metrics
from spectra_scheduler.rl import Context, QNetwork, RewardConfig, RLModel, RLScheduler
from spectra_scheduler.rl_scenarios import (
    GENERATOR_VERSION,
    PERIODIC_VISIBILITY_GENERATOR_VERSION,
    PHYSICAL_GENERATOR_VERSION,
    physical_periodic_visibility_scenario,
    physical_scenario,
    procedural_scenario,
    scenario_fingerprint,
)
from spectra_scheduler.scenarios import SCENARIO_NAMES, build_scenario
from spectra_scheduler.synthetic_evaluation import evaluate_completed_episode

SUITES = ("randomized", "receiver-shift", *SCENARIO_NAMES)


def emitter_family_counts(simulation, result, *, focused_periodic=False):
    """Evaluator-only pulse counts for each physical emitter family."""
    families = {}
    for emitter in simulation.emitters:
        core = emitter.emitter if isinstance(emitter, WindowedEmitter) else emitter
        family = (
            ("periodic-scan" if focused_periodic else "spatial-scan")
            if isinstance(core, SpatialScanningEmitter)
            else "frequency-agile"
            if isinstance(core, FrequencyHoppingEmitter)
            else "frequency-scan"
            if isinstance(core, ScanningEmitter)
            else "periodic"
            if isinstance(core, PeriodicEmitter)
            else "burst"
            if isinstance(core, BurstEmitter)
            else "change"
            if isinstance(core, ModeSwitchingEmitter)
            else type(core).__name__
        )
        families[emitter.emitter_id] = family
    truth = Counter(families[event.emitter_id] for event in result.transmissions)
    detected = Counter(
        families[emitter_id]
        for record in result.detection_records
        for emitter_id in record.detected_emitters
    )
    return {
        family: {"detected": detected[family], "truth": count} for family, count in truth.items()
    }


class Monitor:
    def __init__(self, policy, reward, profile=False):
        self.policy, self.reward, self.profile = policy, reward, profile
        self.decision_seconds = []

    def reset(self, num_bands):
        self.policy.reset(num_bands)
        self.context = Context(num_bands)
        self.total_reward = 0.0
        self.decision_seconds.clear()

    def set_retune_table(self, table):
        if hasattr(self.policy, "set_retune_table"):
            self.policy.set_retune_table(table)

    def set_episode_horizon(self, steps):
        if hasattr(self.policy, "set_episode_horizon"):
            self.policy.set_episode_horizon(steps)

    def choose_band(self, step):
        start = perf_counter() if self.profile else 0
        band = self.policy.choose_band(step)
        if self.profile:
            self.decision_seconds.append(perf_counter() - start)
        return band

    def observe(self, observation):
        self.policy.observe(observation)
        self.context.observe(observation)
        self.total_reward += self.reward.compute(observation, self.context)


def paired_delta(candidate, reference):
    differences = [a - b for a, b in zip(candidate, reference, strict=True)]
    mean = fmean(differences)
    margin = 1.96 * stdev(differences) / sqrt(len(differences)) if len(differences) > 1 else None
    return {
        "mean": mean,
        "approximate_95pct_ci": [mean - margin, mean + margin] if margin is not None else None,
    }


def benchmark(
    models: list[RLModel],
    runs=30,
    seed=10000,
    split="validation",
    suites=SUITES,
    hit_model: HitModel | None = None,
    profile=False,
    progress=None,
    extra_policies=None,
    extra_metadata=None,
    reward=None,
    num_bands=None,
    physical_worlds=False,
):
    extra_policies = extra_policies or {}
    reward = reward or (models[0].reward if models else RewardConfig())
    if (
        not (models or extra_policies)
        or type(runs) is not int
        or runs < 1
        or type(seed) is not int
        or not 0 <= seed < 1_000_000
        or seed + runs > 1_000_000
        or split not in ("validation", "test")
    ):
        raise ValueError("use models, positive runs, nonnegative seed and validation/test split")
    if not suites or len(set(suites)) != len(suites) or set(suites) - set(SUITES):
        raise ValueError("invalid or duplicate benchmark suites")
    physical_suites = {"randomized", "receiver-shift", "periodic-scan"}
    if physical_worlds and (num_bands != 8 or set(suites) - physical_suites):
        raise ValueError("physical-world benchmark requires eight bands and physical suites")
    if len({m.fingerprint for m in models}) != len(models):
        raise ValueError("duplicate model artifacts")
    if len({m.training["config"]["seed"] for m in models}) != len(models):
        raise ValueError("use distinct training RNG seeds for variability estimates")
    if models and any(m.reward != models[0].reward for m in models):
        raise ValueError("models must use the same reward definition")
    if hit_model is not None:
        start = hit_model.training.get("seed")
        count = hit_model.training.get("runs")
        if type(start) is not int or type(count) is not int or count < 1:
            raise ValueError("hit model requires training-seed provenance")
        legacy_seed = seed + (1_000_000 if split == "test" else 0)
        if max(legacy_seed, start) < min(legacy_seed + runs, start + count):
            raise ValueError("benchmark seeds overlap hit-model training")
    report = {
        "schema_version": 1,
        "benchmark_version": 2,
        "generator_version": PHYSICAL_GENERATOR_VERSION if physical_worlds else GENERATOR_VERSION,
        "generator": "physical" if physical_worlds else "legacy-procedural",
        "family_label_version": 2,
        "focused_periodic_generator_version": (
            PERIODIC_VISIBILITY_GENERATOR_VERSION
            if physical_worlds and "periodic-scan" in suites
            else None
        ),
        "split": split,
        "runs": runs,
        "seed": seed,
        "reward": asdict(reward),
        "extra_models": extra_metadata or {},
        "procedural_num_bands": num_bands,
        "models": [
            {"name": f"rl-{i}", "sha256": m.fingerprint, "training": m.training}
            for i, m in enumerate(models)
        ],
        "hit_model_sha256": hit_model.fingerprint if hit_model else None,
        "confidence_interval": (
            "paired scenario seeds, normal approximation, no multiplicity correction"
        ),
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "rl.py",
                "rl_scenarios.py",
                "rl_benchmark.py",
                "simulation.py",
                "receiver.py",
                "schedulers.py",
                "metrics.py",
                "scenarios.py",
                "learned_scheduler.py",
            )
        },
        "suites": {},
    }
    for suite in suites:
        rows = []
        for current_seed in range(seed, seed + runs):
            scenario_seed = current_seed + (
                1_000_000 if split == "test" and suite in SCENARIO_NAMES else 0
            )
            if suite in ("randomized", "receiver-shift"):
                simulation = (
                    physical_scenario(current_seed, split, suite == "receiver-shift")
                    if physical_worlds
                    else procedural_scenario(
                        current_seed, split, suite == "receiver-shift", num_bands
                    )
                )
            elif physical_worlds and suite == "periodic-scan":
                simulation = physical_periodic_visibility_scenario(current_seed, split)
            else:
                simulation = build_scenario(suite, scenario_seed)
            truth = simulation.generate_truth()
            factories = scheduler_factories(current_seed)
            if set(factories) & set(extra_policies):
                raise ValueError("extra policies cannot replace baseline names")
            factories.update(extra_policies)
            for i, model in enumerate(models):
                factories[f"rl-{i}"] = lambda m=model: RLScheduler(m.network, m.reward)
                initial = QNetwork(
                    model.training["config"]["seed"], model.training["config"]["hidden"]
                )
                factories[f"untrained-{i}"] = lambda n=initial: RLScheduler(n, reward)
            if hit_model is not None:
                factories["supervised-hit"] = lambda: LearnedScheduler(hit_model)
            scores = {}
            family_scores = {}
            evaluations = {}
            for name, factory in factories.items():
                scheduler = factory()
                policy = Monitor(scheduler, reward, profile)
                outcome = simulation.run(policy, truth=truth)
                metrics = asdict(calculate_metrics(outcome))
                if physical_worlds:
                    family_scores[name] = emitter_family_counts(
                        simulation, outcome, focused_periodic=suite == "periodic-scan"
                    )
                    evaluations[name] = evaluate_completed_episode(
                        simulation,
                        outcome,
                        step_seconds=0.001,
                        reward_sum=policy.total_reward,
                        reward_description=(
                            "hit * observed_hit - retuning * retune_tick "
                            "- coverage * public_coverage_debt"
                        ),
                    )
                metrics["reward_per_step"] = policy.total_reward / simulation.duration
                if profile:
                    metrics["mean_decision_seconds"] = fmean(policy.decision_seconds)
                scores[name] = metrics
            episode = {
                "seed": current_seed,
                "scenario_seed": scenario_seed,
                "scenario_sha256": scenario_fingerprint(simulation),
                "num_bands": simulation.num_bands,
                "duration": simulation.duration,
                "metrics": scores,
            }
            if physical_worlds:
                episode["family_metrics"] = family_scores
                episode["evaluation"] = evaluations
            rows.append(episode)
        names = list(rows[0]["metrics"])
        means = {
            name: {
                key: fmean(row["metrics"][name][key] for row in rows)
                for key in rows[0]["metrics"][name]
            }
            for name in names
        }
        paired = {}
        for name in [f"rl-{i}" for i in range(len(models))] + list(extra_policies):
            paired[name] = {
                other: paired_delta(
                    [row["metrics"][name]["interception_ratio"] for row in rows],
                    [row["metrics"][other]["interception_ratio"] for row in rows],
                )
                for other in names
                if other != name
            }
        rl_means = [means[f"rl-{i}"]["interception_ratio"] for i in range(len(models))]
        report["suites"][suite] = {
            "scope": (
                "unseen procedural layouts"
                if suite == "randomized"
                else "unseen layouts plus eight bands, longer horizon, higher noise/retuning"
                if suite == "receiver-shift"
                else "focused physical periodic visibility with varied beam and pulse clocks"
                if physical_worlds and suite == "periodic-scan"
                else "legacy fixed layout, new phase/noise seeds"
            ),
            "mean_metrics": means,
            "paired_interception_delta": paired,
            "training_seed_variability": {
                "models": len(models),
                "mean_interception": fmean(rl_means) if rl_means else None,
                "stddev": stdev(rl_means) if len(models) > 1 else None,
            },
            "episodes": rows,
        }
        if physical_worlds:
            by_policy = {}
            for name in names:
                totals = {}
                for row in rows:
                    for family, counts in row["family_metrics"][name].items():
                        aggregate = totals.setdefault(family, {"detected": 0, "truth": 0})
                        aggregate["detected"] += counts["detected"]
                        aggregate["truth"] += counts["truth"]
                by_policy[name] = {
                    family: {**counts, "capture": counts["detected"] / counts["truth"]}
                    for family, counts in totals.items()
                }
            report["suites"][suite]["family_capture"] = by_policy
        if progress:
            progress(suite, means)
    return report
