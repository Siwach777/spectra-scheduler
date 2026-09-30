"""Frozen joint timing comparison with previous best, native MPC and round-robin."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..grouped_policy import load_grouped
from ..joint_belief import load_joint
from ..policy_benchmark import PolicySpec, _run_jobs, benchmark_synthetic, summarize
from ..scenarios import REQUIREMENT_SCENARIOS
from ..schedulers import DwellSweepScheduler
from ..synthetic_evaluation import evaluate_scheduler
from ..timing_belief import BeliefPolicyConfig, load_belief
from .calibrated_timing import CalibratedBeliefPolicy, configure_public_detection
from .grouped_report import pooled_metrics
from .grouped_study import trajectories
from .storage import fingerprint, run_lock, write_json
from .timing_mpc_compare import batched_mpc, load_mpc
from .timing_report import _control_job, batched_neural_results, reporting_world, reward


def verify_model(model, config):
    jobs = [(s, 2000) for s in REQUIREMENT_SCENARIOS]
    rows, _ = batched_neural_results(model, config, jobs, len(jobs))
    by_group = {row["group"]: row for row in rows}

    def compare(a, b):
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError("joint report schemas differ")
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (float, int)):
            if not np.isclose(a, b, rtol=5e-5, atol=1e-5):
                raise ValueError(f"joint serial/batched parity failure: {a} vs {b}")
        elif a != b:
            raise ValueError("joint serial/batched parity failure")

    for scenario, seed in jobs:
        row = by_group[f"{scenario}:{seed}"]
        world, _ = reporting_world(scenario, seed)
        policy = CalibratedBeliefPolicy(model, config)
        configure_public_detection(world, policy)
        actual = evaluate_scheduler(
            world,
            policy,
            step_seconds=0.001,
            reward=reward,
            reward_description="observed_hit - 0.05 * retuning",
        )
        compare(row["value"]["evaluation"], actual)
    return {
        "serial_cuda_parity": True,
        "selection_worlds": len(jobs),
        "continuous_metric_tolerance": {"relative": 5e-5, "absolute": 1e-5},
    }


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-best", type=Path, required=True)
    parser.add_argument("--mpc", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--inference-batch-size", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(arguments)
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    semantic = json.loads((args.run_dir / "config.json").read_text())
    seeds = semantic["reporting_seeds"]
    if set(seeds) & set(semantic["selection_seeds"]):
        raise ValueError("selection/reporting overlap")
    policy = dict(semantic["policy"])
    policy["dwells"] = tuple(policy["dwells"])
    config = BeliefPolicyConfig(**policy)
    checkpoints = [
        args.run_dir / f"seed-{seed}" / "best.pt" for seed in semantic["arguments"]["seeds"]
    ]
    if args.verify_only:
        checks = {}
        for path in checkpoints:
            model, _ = load_joint(path)
            checks[str(path)] = verify_model(model, config)
        print(json.dumps(checks, indent=2))
        return 0
    started = time.perf_counter()
    with run_lock(args.output_dir):
        if (args.output_dir / "frozen.json").exists():
            raise ValueError("report directory already frozen")
        frozen = {
            "configuration_sha256": fingerprint(args.run_dir / "config.json"),
            "seeds": seeds,
            "policy": asdict(config),
            "sources": {"joint_report": fingerprint(__file__)},
            "checkpoints": {str(p): fingerprint(p) for p in checkpoints},
            "previous_best_sha256": fingerprint(args.previous_best),
            "mpc_sha256": {str(p): fingerprint(p) for p in args.mpc},
            "variants": "joint forecaster alone and with the same frozen previous-best action head",
            "scope": "fresh development reporting; final test unused",
        }
        for path in checkpoints:
            selection = json.loads((path.parent / "frozen.json").read_text())
            if selection["checkpoint_sha256"] != fingerprint(path):
                raise ValueError("selected joint checkpoint changed")
            shutil.copyfile(path, args.output_dir / f"joint-{path.parent.name}.pt")
        write_json(args.output_dir / "frozen.json", frozen)
        jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in seeds]
        phase = BeliefPolicyConfig(dwells=config.dwells, revisit=256, probe=4, exploration=0.02)
        results = _run_jobs(
            _control_job, ((s, seed, config, phase) for s, seed in jobs), args.workers
        )
        rows = {r["group"]: r for r in results}
        profiles = {}
        base, actor, previous_config, _ = load_grouped(args.previous_best)
        if previous_config != config:
            raise ValueError("previous-best and joint action configurations differ")
        for offset in range(0, len(jobs), args.inference_batch_size):
            chunk = jobs[offset : offset + args.inference_batch_size]
            worlds = [reporting_world(s, seed)[0] for s, seed in chunk]
            _, _, evaluations, profile = trajectories(base, actor, config, worlds)
            for (s, seed), evaluation in zip(chunk, evaluations, strict=True):
                rows[f"{s}:{seed}"]["policies"]["previous-best"] = {
                    "evaluation": evaluation,
                    "discovery_fraction": evaluation["discovery"]["emitter_discovery_ratio"],
                }
        profiles["previous-best"] = profile
        del base
        base, _ = load_belief(Path(semantic["arguments"]["checkpoint"]))
        values, profiles["previous-forecaster"] = batched_neural_results(
            base, config, jobs, args.inference_batch_size
        )
        for row in values:
            rows[row["group"]]["policies"]["previous-forecaster"] = row["value"]
        del base
        for path in checkpoints:
            name = f"joint-{path.parent.name}"
            model, _ = load_joint(args.output_dir / f"{name}.pt")
            values, profiles[name] = batched_neural_results(
                model, config, jobs, args.inference_batch_size
            )
            for row in values:
                rows[row["group"]]["policies"][name] = row["value"]
            transfer_name = f"{name}-transfer-head"
            for offset in range(0, len(jobs), args.inference_batch_size):
                chunk = jobs[offset : offset + args.inference_batch_size]
                worlds = [reporting_world(s, seed)[0] for s, seed in chunk]
                _, _, evaluations, profile = trajectories(model, actor, config, worlds)
                for (s, seed), evaluation in zip(chunk, evaluations, strict=True):
                    rows[f"{s}:{seed}"]["policies"][transfer_name] = {
                        "evaluation": evaluation,
                        "discovery_fraction": evaluation["discovery"]["emitter_discovery_ratio"],
                    }
            profiles[transfer_name] = profile
            del model
            print(json.dumps({"policy": name, "episodes": len(jobs)}), flush=True)
        del actor
        for i, path in enumerate(args.mpc):
            model, search = load_mpc(path)
            values, profiles[f"mpc-{i}"] = batched_mpc(
                model, search, jobs, args.inference_batch_size
            )
            for row in values:
                rows[row["group"]]["policies"][f"mpc-{i}"] = row["value"]
            del model
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
            evaluations = [p["evaluation"] for p in row["policies"].values()]
            if len({v["counts"]["truth"] for v in evaluations}) != 1:
                raise ValueError("truth denominators differ")
            if any(abs(v["elapsed_seconds"] - 0.512) > 1e-9 for v in evaluations):
                raise ValueError("physical budgets differ")
        policies = [PolicySpec(name, DwellSweepScheduler, name) for name in names]
        references = [
            "previous-best",
            "previous-forecaster",
            "round-robin-50",
            *(f"mpc-{i}" for i in range(len(args.mpc))),
        ]
        report = {
            "frozen": frozen,
            "results": results,
            "by_reference": {
                ref: {
                    s: summarize([r for r in results if r["scenario"] == s], policies, ref)
                    for s in REQUIREMENT_SCENARIOS
                }
                for ref in references
            },
            "pooled": {
                s: pooled_metrics([r for r in results if r["scenario"] == s], names, "mpc-0")
                for s in REQUIREMENT_SCENARIOS
            },
            "resources": {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "profiles": profiles,
            },
        }
        write_json(args.output_dir / "comparison.json", report)
        print(
            json.dumps(
                {
                    s: {p: m["interception_ratio"]["mean"] for p, m in v.items()}
                    for s, v in report["by_reference"]["previous-best"].items()
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
