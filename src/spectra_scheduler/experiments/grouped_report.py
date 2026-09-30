"""Frozen grouped-policy reporting against native MPC and round-robin."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..grouped_policy import GroupedTimingPolicy, load_grouped
from ..policy_benchmark import PolicySpec, _run_jobs, benchmark_synthetic, summarize
from ..scenarios import REQUIREMENT_SCENARIOS
from ..schedulers import DwellSweepScheduler
from ..synthetic_evaluation import evaluate_scheduler
from ..timing_belief import BeliefPolicyConfig, load_belief
from .calibrated_timing import configure_public_detection
from .grouped_study import trajectories
from .storage import fingerprint, run_lock, write_json
from .timing_mpc_compare import batched_mpc, load_mpc
from .timing_report import _control_job, batched_neural_results, reporting_world, reward


def verify_grouped(path):
    """Compare batched collection with the shared serial CUDA evaluation contract."""
    model, actor, config, _ = load_grouped(path)
    worlds = [reporting_world(scenario, 2000)[0] for scenario in REQUIREMENT_SCENARIOS]
    _, _, batched, _ = trajectories(model, actor, config, worlds)

    def compare(actual, expected):
        if isinstance(actual, dict):
            if actual.keys() != expected.keys():
                raise ValueError("grouped evaluation schema differs")
            for key in actual:
                compare(actual[key], expected[key])
        elif isinstance(actual, (float, int)) and actual is not None:
            if not np.isclose(actual, expected, rtol=1e-5, atol=1e-7):
                raise ValueError(f"grouped batched evaluation differs: {actual} versus {expected}")
        elif actual != expected:
            raise ValueError("grouped batched evaluation differs")

    for world, expected in zip(worlds, batched, strict=True):
        scheduler = GroupedTimingPolicy(model, actor, config)
        configure_public_detection(world, scheduler)
        actual = evaluate_scheduler(
            world, scheduler, step_seconds=0.001, reward=reward,
            reward_description="observed_hit - 0.05 * retuning",
        )
        compare(actual, expected)
    return {"checkpoint": str(path), "selection_worlds": len(worlds), "serial_cuda_parity": True}


def pooled_metrics(results, names, reference):
    rng = np.random.default_rng(0)
    truth = np.array([r["policies"][reference]["evaluation"]["counts"]["truth"] for r in results])
    draws = rng.integers(len(results), size=(2000, len(results)))
    denominators = truth[draws].sum(-1)
    baseline = np.array(
        [r["policies"][reference]["evaluation"]["counts"]["captured"] for r in results]
    )
    values = {}
    for name in names:
        captured = np.array(
            [r["policies"][name]["evaluation"]["counts"]["captured"] for r in results]
        )
        differences = (captured[draws] - baseline[draws]).sum(-1) / denominators
        values[name] = {
            "capture": float(captured.sum() / truth.sum()),
            "captured": int(captured.sum()),
            "truth": int(truth.sum()),
            "paired_difference_95_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        }
    return values


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mpc", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--inference-batch-size", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(arguments)
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    configuration = json.loads((args.run_dir / "config.json").read_text())
    seeds = tuple(configuration["reporting_seeds"])
    if set(seeds) & set(configuration["selection_seeds"]):
        raise ValueError("reporting overlaps checkpoint selection")
    training_seeds = configuration["arguments"]["seeds"]
    if args.verify_only:
        checks = [
            verify_grouped(args.run_dir / f"seed-{seed}" / "best.pt")
            for seed in training_seeds
        ]
        print(json.dumps(checks, indent=2))
        return 0
    frozen = {
        "training_configuration_sha256": fingerprint(args.run_dir / "config.json"),
        "source_sha256": fingerprint(__file__),
        "seeds": list(seeds),
        "split": "val",
        "mpc_sha256": {str(p): fingerprint(p) for p in args.mpc},
        "checkpoints": {},
        "policy": configuration["policy"],
        "step_seconds": 0.001,
        "physical_horizon_ticks": 512,
        "mpc_dwell_steps": {
            f"mpc-{i}": list(
                torch.load(p, map_location="cpu", weights_only=True)["config"]["dwell_steps"]
            )
            for i, p in enumerate(args.mpc)
        },
    }
    started = time.perf_counter()
    with run_lock(args.output_dir):
        if (args.output_dir / "frozen.json").exists():
            raise ValueError("reporting directory already frozen")
        shutil.copyfile(args.run_dir / "forecaster.pt", args.output_dir / "forecaster.pt")
        for seed in training_seeds:
            source = args.run_dir / f"seed-{seed}" / "best.pt"
            selection = json.loads((source.parent / "frozen.json").read_text())
            if fingerprint(source) != selection["checkpoint_sha256"]:
                raise ValueError("selected grouped checkpoint changed")
            directory = args.output_dir / f"seed-{seed}"
            directory.mkdir()
            shutil.copyfile(source, directory / "best.pt")
            frozen["checkpoints"][str(seed)] = selection
        write_json(args.output_dir / "frozen.json", frozen)
        policy = dict(configuration["policy"])
        policy["dwells"] = tuple(policy["dwells"])
        config = BeliefPolicyConfig(**policy)
        phase_config = BeliefPolicyConfig(
            dwells=config.dwells, revisit=256, probe=4, exploration=0.02
        )
        jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in seeds]
        results = _run_jobs(
            _control_job, ((s, seed, config, phase_config) for s, seed in jobs), args.workers
        )
        rows = {r["group"]: r for r in results}
        model, _ = load_belief(args.output_dir / "forecaster.pt")
        values, profile = batched_neural_results(model, config, jobs, args.inference_batch_size)
        profiles = {"timing-before-rl": profile}
        for row in values:
            rows[row["group"]]["policies"]["timing-before-rl"] = row["value"]
        del model
        for seed in training_seeds:
            model, actor, policy, metadata = load_grouped(
                args.output_dir / f"seed-{seed}" / "best.pt"
            )
            name = f"grouped-{seed}"
            for offset in range(0, len(jobs), args.inference_batch_size):
                chunk = jobs[offset : offset + args.inference_batch_size]
                worlds = [reporting_world(s, world_seed)[0] for s, world_seed in chunk]
                _, _, evaluations, profile = trajectories(model, actor, policy, worlds)
                for (scenario, world_seed), evaluation in zip(chunk, evaluations, strict=True):
                    rows[f"{scenario}:{world_seed}"]["policies"][name] = {
                        "evaluation": evaluation,
                        "discovery_fraction": evaluation["discovery"]["emitter_discovery_ratio"],
                    }
            profiles[name] = {"selected_iteration": metadata["iteration"]}
            del model, actor
            print(json.dumps({"policy": name, **profiles[name], "episodes": len(jobs)}), flush=True)
        for index, path in enumerate(args.mpc):
            name = f"mpc-{index}"
            model, search_config = load_mpc(path)
            values, profiles[name] = batched_mpc(
                model, search_config, jobs, args.inference_batch_size
            )
            for value in values:
                rows[value["group"]]["policies"][name] = value["value"]
            del model
            print(json.dumps({"policy": name, "episodes": len(jobs)}), flush=True)
        sweeps = [
            PolicySpec(
                f"round-robin-{d}", partial(DwellSweepScheduler, dwell_steps=d), f"dwell:{d}"
            )
            for d in (10, 50)
        ]
        sweep_report = benchmark_synthetic(
            sweeps,
            baseline="round-robin-50",
            seeds=seeds,
            scenarios=REQUIREMENT_SCENARIOS,
            workers=args.workers,
        )
        for row in sweep_report["results"]:
            rows[row["group"]]["policies"].update(row["policies"])
        names = list(results[0]["policies"])
        for row in results:
            evaluations = [row["policies"][name]["evaluation"] for name in names]
            if len({v["counts"]["truth"] for v in evaluations}) != 1:
                raise ValueError("paired policies received different truth denominators")
            if any(abs(v["elapsed_seconds"] - 0.512) > 1e-9 for v in evaluations):
                raise ValueError("paired policies received different physical time budgets")
        policies = [PolicySpec(name, DwellSweepScheduler, name) for name in names]
        references = [
            *(f"mpc-{i}" for i in range(len(args.mpc))),
            "round-robin-50",
            "timing-before-rl",
        ]
        report = {
            "frozen": frozen,
            "results": results,
            "by_reference": {
                baseline: {
                    scenario: summarize(
                        [r for r in results if r["scenario"] == scenario], policies, baseline
                    )
                    for scenario in REQUIREMENT_SCENARIOS
                }
                for baseline in references
            },
            "pooled": {
                scenario: pooled_metrics(
                    [r for r in results if r["scenario"] == scenario], names, "mpc-0"
                )
                for scenario in REQUIREMENT_SCENARIOS
            },
            "resources": {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "profiles": profiles,
            },
            "scope": (
                "fresh development reporting; two action-policy seeds "
                "share one frozen forecaster; test unused"
            ),
        }
        write_json(args.output_dir / "comparison.json", report)
        print(
            json.dumps(
                {
                    s: {p: m["interception_ratio"]["mean"] for p, m in v.items()}
                    for s, v in report["by_reference"]["mpc-0"].items()
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
