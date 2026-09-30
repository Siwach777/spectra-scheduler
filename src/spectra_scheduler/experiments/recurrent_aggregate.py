"""Aggregate paired recurrent-policy assessments across independent training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .storage import fingerprint, write_json

POLICIES = ("ppo-greedy", "ppo-sampled")
MEASURES = ("interception_ratio", "emitter_discovery_ratio")
BOOTSTRAP_DRAWS = 2_000


def _source(report: dict, path: Path) -> dict:
    trained = report["extra_models"]["trained"]
    config = trained["config"]
    seed = config["seed"]
    if type(seed) is not int or seed < 0 or not trained.get("sha256"):
        raise ValueError(f"missing trained checkpoint provenance: {path}")
    if report["split"] != "validation" or report["procedural_num_bands"] != 8:
        raise ValueError(f"assessment is not an eight-band validation report: {path}")
    if not trained.get("environment_steps") or not trained.get("physical_steps"):
        raise ValueError(f"missing training exposure counts: {path}")
    return {
        "path": str(path),
        "report_sha256": fingerprint(path),
        "training_seed": seed,
        "checkpoint_sha256": trained["sha256"],
        "decision_steps": trained["environment_steps"],
        "physical_steps": trained["physical_steps"],
        "implementation_sha256": config["implementation"],
    }


def _aligned(reports: list[dict]) -> tuple[str, ...]:
    first = reports[0]
    for report in reports[1:]:
        for key in (
            "schema_version",
            "benchmark_version",
            "generator_version",
            "split",
            "runs",
            "seed",
            "reward",
            "procedural_num_bands",
            "assessment_sha256",
        ):
            if report[key] != first[key]:
                raise ValueError(f"assessment settings differ: {key}")
        # The generic benchmark gained an optional episode-horizon hook for
        # MPC between these reports. Recurrent schedulers do not expose that
        # hook; exact paired control outcomes are checked below.
        left_benchmark = first["implementation_sha256"]
        right_benchmark = report["implementation_sha256"]
        for source in set(left_benchmark) | set(right_benchmark):
            if source != "rl_benchmark.py" and left_benchmark.get(source) != right_benchmark.get(
                source
            ):
                raise ValueError(f"benchmark implementation differs: {source}")
        if set(report["suites"]) != set(first["suites"]):
            raise ValueError("assessment suite sets differ")
        left = dict(first["extra_models"]["trained"]["config"])
        right = dict(report["extra_models"]["trained"]["config"])
        left.pop("seed", None)
        right.pop("seed", None)
        left_sources = left.pop("implementation")
        right_sources = right.pop("implementation")
        if left != right:
            raise ValueError("training configurations differ beyond the seed")
        # The seed-zero CLI hash changed when assessment/metadata handling was
        # improved. Core environment, policy, reward and SMDP code must match.
        for key in set(left_sources) | set(right_sources):
            if key != "recurrent_cli.py" and left_sources.get(key) != right_sources.get(key):
                raise ValueError(f"training implementation differs: {key}")
        for suite_name, first_suite in first["suites"].items():
            suite = report["suites"][suite_name]
            if set(suite["mean_metrics"]) != set(first_suite["mean_metrics"]):
                raise ValueError(f"policy set differs in {suite_name}")
            if len(suite["episodes"]) != len(first_suite["episodes"]):
                raise ValueError(f"world count differs in {suite_name}")
            for left_world, right_world in zip(
                first_suite["episodes"], suite["episodes"], strict=True
            ):
                for key in ("seed", "scenario_seed", "scenario_sha256", "num_bands", "duration"):
                    if left_world[key] != right_world[key]:
                        raise ValueError(f"paired worlds differ in {suite_name}: {key}")
                for policy in first_suite["mean_metrics"]:
                    if policy in POLICIES or policy.startswith("untrained-"):
                        continue
                    if left_world["metrics"][policy] != right_world["metrics"][policy]:
                        raise ValueError(f"control results differ in {suite_name}: {policy}")
    return tuple(first["suites"])


def _matrix(reports: list[dict], suite: str, policy: str, measure: str) -> np.ndarray:
    matrix = np.asarray(
        [
            [world["metrics"][policy][measure] for world in report["suites"][suite]["episodes"]]
            for report in reports
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError(f"undefined or nonfinite {measure} in {suite}/{policy}")
    return matrix


def _interval(matrix: np.ndarray, key: str) -> list[float]:
    """Resample model seeds and shared worlds independently, retaining pairing."""
    digest = hashlib.sha256(key.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    seeds = rng.integers(matrix.shape[0], size=(BOOTSTRAP_DRAWS, matrix.shape[0]))
    worlds = rng.integers(matrix.shape[1], size=(BOOTSTRAP_DRAWS, matrix.shape[1]))
    samples = matrix[seeds[:, :, None], worlds[:, None, :]].mean(axis=(1, 2))
    return np.quantile(samples, (0.025, 0.975)).tolist()


def aggregate(paths: list[Path]) -> dict:
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("provide at least two distinct comparison reports")
    reports = [json.loads(path.read_text()) for path in paths]
    sources = [_source(report, path) for report, path in zip(reports, paths, strict=True)]
    seeds = [source["training_seed"] for source in sources]
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be distinct")
    if len({source["checkpoint_sha256"] for source in sources}) != len(sources):
        raise ValueError("trained checkpoints must be distinct")
    suites = _aligned(reports)
    result = {
        "schema_version": 1,
        "scope": (
            "development validation across independent training seeds and paired worlds; "
            "fixed-layout suites vary phase/noise, not layout; no test-set claim"
        ),
        "bootstrap": (
            "2000 independent resamples of training seeds and paired world indices; "
            "percentile 95% intervals; no multiplicity correction"
        ),
        "sources": sources,
        "suites": {},
    }
    for suite in suites:
        names = tuple(reports[0]["suites"][suite]["mean_metrics"])
        summary = {
            "scope": reports[0]["suites"][suite]["scope"],
            "worlds_per_training_seed": len(reports[0]["suites"][suite]["episodes"]),
            "policies": {},
            "paired_capture_delta": {},
        }
        capture = {}
        for name in names:
            capture[name] = _matrix(reports, suite, name, "interception_ratio")
            row = {}
            for measure in MEASURES:
                values = (
                    capture[name]
                    if measure == "interception_ratio"
                    else _matrix(reports, suite, name, measure)
                )
                per_seed = values.mean(axis=1)
                row[measure] = {
                    "mean": float(values.mean()),
                    "per_training_seed": {
                        str(seed): float(value) for seed, value in zip(seeds, per_seed, strict=True)
                    },
                    "training_seed_sd": float(per_seed.std(ddof=1)),
                    "bootstrap_95_interval": _interval(values, f"{suite}:{name}:{measure}"),
                }
            summary["policies"][name] = row
        for name in POLICIES:
            if name not in capture:
                raise ValueError(f"missing trained policy in {suite}: {name}")
            summary["paired_capture_delta"][name] = {}
            for reference in names:
                if name == reference:
                    continue
                delta = capture[name] - capture[reference]
                summary["paired_capture_delta"][name][reference] = {
                    "mean": float(delta.mean()),
                    "per_training_seed": {
                        str(seed): float(value)
                        for seed, value in zip(seeds, delta.mean(axis=1), strict=True)
                    },
                    "bootstrap_95_interval": _interval(delta, f"{suite}:{name}:{reference}"),
                }
        result["suites"][suite] = summary
    return result


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    paths = [path / "comparison.json" for path in args.run_dir]
    write_json(args.output, aggregate(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
