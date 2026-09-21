"""Frozen benchmark plans, arbitrary policy factories and paired grouped inference."""

import argparse
import hashlib
import json
import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from .pulse_replay import ReplayConfig
from .replay_env import InterfaceConfig
from .replay_evaluation import ReferencePolicy, evaluate_policy
from .scenarios import REQUIREMENT_SCENARIOS, build_scenario
from .synthetic_evaluation import evaluate_scheduler


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Atomic publication; never leave a partial benchmark or manifest."""
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_plan(root, split="val", max_files=10, seeds=(0, 1, 2), selection_seed=0):
    """Freeze a seeded recording sample, content hashes and receiver/policy seeds."""
    from .dataset_io import discover_files

    if split not in ("train", "val", "test") or type(max_files) is not int or max_files < 1:
        raise ValueError("invalid split or file limit")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("seeds must be unique uint32 integers")
    root = Path(root).resolve()
    files = discover_files(root, "stare", split)
    if len(files) < max_files:
        raise ValueError(f"requested {max_files} files but only {len(files)} are available")
    selected = np.random.default_rng(selection_seed).choice(len(files), max_files, replace=False)
    return {
        "version": 1,
        "split": split,
        "seeds": list(seeds),
        "selection_seed": selection_seed,
        "recordings": [
            {"path": str(files[i].resolve().relative_to(root)), "sha256": fingerprint(files[i])}
            for i in sorted(selected)
        ],
    }


def validate_plan(root, plan):
    from .dataset_io import discover_files

    if plan.get("version") != 1 or plan.get("split") not in ("train", "val", "test"):
        raise ValueError("unsupported benchmark plan")
    seeds = plan.get("seeds", [])
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("invalid plan seeds")
    root = Path(root).resolve()
    allowed = set(p.resolve() for p in discover_files(root, "stare", plan["split"]))
    paths = []
    content_hashes = set()
    for recording in plan["recordings"]:
        path = (root / recording["path"]).resolve()
        if path not in allowed or path in paths:
            raise ValueError("recording outside selected split or duplicated")
        if fingerprint(path) != recording["sha256"]:
            raise ValueError(f"recording content changed: {path}")
        if recording["sha256"] in content_hashes:
            raise ValueError("duplicate recording content is not an independent benchmark group")
        content_hashes.add(recording["sha256"])
        paths.append(path)
    if not paths:
        raise ValueError("empty recording plan")
    return paths


@dataclass(frozen=True)
class PolicySpec:
    """Factory must return a fresh policy and be spawn-pickleable for workers > 1.

    provenance should identify configuration/checkpoint content, not just a label.
    trained_on contains SHA-256 recording hashes to reject evaluation leakage.
    """

    name: str
    factory: object
    provenance: str
    trained_on: tuple[str, ...] = ()


def _validate_policies(policies, baseline):
    names = [p.name for p in policies]
    if not names or len(set(names)) != len(names) or baseline not in names:
        raise ValueError("unique policy names and a registered baseline are required")
    if any(not p.name or not p.provenance or not callable(p.factory) for p in policies):
        raise ValueError("policy name, factory and provenance are required")


def _replay_job(job):
    index, path, seed, policies, receiver, interface = job
    return {
        "group": str(index),
        "seed": seed,
        "policies": {
            p.name: evaluate_policy(path, p.factory(), replace(receiver, seed=seed), interface)
            for p in policies
        },
    }


def _observable_reward(observation):
    return float(observation.hit) - 0.05 * (not observation.listening)


def _synthetic_job(job):
    scenario, seed, split, policies, step_seconds = job
    # Separate deterministic world namespaces prevent train/val/test seed aliasing.
    key = f"synthetic-v1:{split}:{scenario}:{seed}".encode()
    world_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    simulation = build_scenario(scenario, world_seed)
    truth = simulation.generate_truth()
    return {
        "group": f"{scenario}:{seed}",
        "scenario": scenario,
        "seed": seed,
        "world_seed": world_seed,
        "policies": {
            p.name: {
                "evaluation": evaluate_scheduler(
                    simulation,
                    p.factory(),
                    step_seconds=step_seconds,
                    reward=_observable_reward,
                    reward_description="observed_hit - 0.05 * retuning",
                    truth=truth,
                )
            }
            for p in policies
        },
    }


def _run_jobs(function, jobs, workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1:
        return [function(job) for job in jobs]
    # Bound pending jobs and preserve order regardless of completion timing.
    variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in variables}
    try:
        os.environ.update(dict.fromkeys(variables, "1"))
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            iterator, pending, results = iter(jobs), deque(), []
            for _ in range(2 * workers):
                job = next(iterator, None)
                if job is None:
                    break
                pending.append(pool.submit(function, job))
            while pending:
                results.append(pending.popleft().result())
                job = next(iterator, None)
                if job is not None:
                    pending.append(pool.submit(function, job))
            return results
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


METRICS = (
    "probability_of_detection",
    "probability_of_false_alarm",
    "sensitivity_loss_fraction",
    "average_intercept_rate_per_second",
    "interception_ratio",
    "reward_per_second",
    "average_reward_per_action",
    "prediction.percentage_correct",
    "prediction.brier_score",
    "prediction.interception_ratio_mae",
    "prediction.average_intercept_time_error_seconds",
    "prediction.restricted_intercept_time_mae_seconds",
    "prediction.coverage",
    "prediction.timing_event_coverage",
)


def summarize(results, policies, baseline, bootstrap_seed=0):
    """Average seeds within recordings before bootstrapping independent groups."""
    output = {}
    for policy in policies:
        metrics = {}
        for metric in METRICS:
            groups, paired = {}, {}
            for result in results:

                def get(name, result=result, metric=metric):
                    value = result["policies"][name]["evaluation"]
                    for part in metric.split("."):
                        value = value[part]
                    return value

                value, reference = get(policy.name), get(baseline)
                group = result["group"]
                if value is not None:
                    groups.setdefault(group, []).append(value)
                if value is not None and reference is not None:
                    paired.setdefault(group, []).append(value - reference)
            values = [float(np.mean(v)) for v in groups.values()]
            deltas = np.array([np.mean(v) for v in paired.values()])
            interval = None
            if len(deltas) > 1:
                rng = np.random.default_rng(bootstrap_seed)
                samples = np.empty(2000)
                batch_size = max(1, min(128, 65536 // len(deltas)))
                for i in range(0, len(samples), batch_size):
                    count = min(batch_size, len(samples) - i)
                    samples[i : i + count] = rng.choice(
                        deltas, (count, len(deltas)), replace=True
                    ).mean(axis=1)
                interval = np.quantile(samples, [0.025, 0.975]).tolist()
            metrics[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "groups": len(values),
                "paired_groups": len(deltas),
                "paired_mean_difference": float(deltas.mean()) if len(deltas) else None,
                "paired_bootstrap_95_interval": interval,
            }
        output[policy.name] = metrics
    return output


def benchmark_policies(root, plan, policies, *, baseline, workers=1, receiver=None, interface=None):
    _validate_policies(policies, baseline)
    paths = validate_plan(root, plan)
    hashes = {r["sha256"] for r in plan["recordings"]}
    if plan["split"] != "train" and any(hashes.intersection(p.trained_on) for p in policies):
        raise ValueError("evaluation recordings overlap model training provenance")
    receiver, interface = receiver or ReplayConfig(), interface or InterfaceConfig()
    jobs = (
        (i, path, seed, policies, receiver, interface)
        for i, path in enumerate(paths)
        for seed in plan["seeds"]
    )
    results = _run_jobs(_replay_job, jobs, workers)
    # Detect mutations during the complete comparison, not only a single episode.
    validate_plan(root, plan)
    return {
        "schema_version": 1,
        "backend": "stare",
        "plan": plan,
        "receiver": asdict(receiver),
        "interface": asdict(interface),
        "baseline": baseline,
        "policies": [
            {"name": p.name, "provenance": p.provenance, "trained_on": list(p.trained_on)}
            for p in policies
        ],
        "summary": summarize(results, policies, baseline),
        "results": results,
    }


def benchmark_synthetic(
    policies,
    *,
    baseline,
    split="val",
    seeds=(0, 1, 2),
    scenarios=REQUIREMENT_SCENARIOS,
    step_seconds=0.001,
    workers=1,
):
    _validate_policies(policies, baseline)
    if (
        split not in ("train", "val", "test")
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("invalid split or seeds")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be nonempty and unique")
    jobs = (
        (scenario, seed, split, policies, step_seconds) for scenario in scenarios for seed in seeds
    )
    results = _run_jobs(_synthetic_job, jobs, workers)
    return {
        "schema_version": 1,
        "backend": "synthetic",
        "split": split,
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "step_seconds": step_seconds,
        "baseline": baseline,
        "policies": [{"name": p.name, "provenance": p.provenance} for p in policies],
        "summary": summarize(results, policies, baseline),
        "by_scenario": {
            s: summarize([r for r in results if r["scenario"] == s], policies, baseline)
            for s in scenarios
        },
        "results": results,
    }


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--backend", choices=("stare", "synthetic"), default="stare")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--files", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    if args.backend == "stare":
        if args.plan and args.plan.exists():
            plan = json.loads(args.plan.read_text())
            if plan["split"] != args.split or plan["seeds"] != args.seeds:
                raise ValueError("CLI split/seeds differ from frozen plan")
        else:
            plan = make_plan(args.root, args.split, args.files, tuple(args.seeds))
            if args.plan:
                write_json(args.plan, plan)
        policies = [
            PolicySpec(name, partial(ReferencePolicy, name), f"{name}:dwell-index=1")
            for name in ("sweep", "random")
        ]
        if args.model:
            from .forecast_model import predictor_spec

            policies.extend(
                predictor_spec(path, name=f"predictor-{i}") for i, path in enumerate(args.model)
            )
        report = benchmark_policies(
            args.root, plan, policies, baseline="sweep", workers=args.workers
        )
    else:
        if args.model or args.plan:
            raise ValueError("replay checkpoints and recording plans require the stare backend")
        from .schedulers import PeriodAwareScheduler, RoundRobinScheduler, ShuffledSweepScheduler

        policies = [
            PolicySpec("sweep", RoundRobinScheduler, "round-robin"),
            PolicySpec("shuffled", ShuffledSweepScheduler, "shuffled-sweep:seed=0"),
            PolicySpec("period-aware", PeriodAwareScheduler, "period-aware:defaults"),
        ]
        report = benchmark_synthetic(
            policies,
            baseline="sweep",
            split=args.split,
            seeds=tuple(args.seeds),
            workers=args.workers,
        )
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
