"""Select coverage on development worlds; keep capture-retention constraints explicit."""

import argparse
import json
import os
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..scenarios import REQUIREMENT_SCENARIOS
from ..synthetic_evaluation import evaluate_scheduler
from ..timing_belief import BeliefPolicyConfig
from ..timing_coverage import RecoveryConfig, RecoveryTimingPlanner
from ..timing_ensemble import load_predictor
from ..timing_planner import CalibratedTimingPlannerPolicy
from .planner_study import batched_planned_results
from .storage import fingerprint, run_lock, write_json
from .timing_refine_report import compare_reports
from .timing_report import reporting_world, reward


def selection_means(rows):
    output = {}
    for scenario in REQUIREMENT_SCENARIOS:
        values = [r["value"] for r in rows if r["scenario"] == scenario]
        capture = [v["evaluation"]["interception_ratio"] for v in values
                   if v["evaluation"]["interception_ratio"] is not None]
        output[scenario] = {
            "capture": float(np.mean(capture)), "capture_worlds": len(capture),
            "discovery": float(np.mean([v["discovery_fraction"] for v in values])),
            "acquisition_ms": float(np.mean([
                v["evaluation"]["discovery"]["mean_first_detection_delay_seconds"] * 1000
                for v in values])),
        }
    return output


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(arguments)
    if min(args.runs, args.batch_size) < 1:
        parser.error("positive episode counts and inference batch required")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("coverage validation requires CUDA")
    model, metadata = load_predictor(args.checkpoint)
    raw = dict(metadata["policy"])
    raw["dwells"] = tuple(raw["dwells"])
    config = BeliefPolicyConfig(**raw)
    checkpoint_digest = fingerprint(args.checkpoint)
    coverage_digest = fingerprint(Path(__file__).parent.parent / "timing_coverage.py")
    if args.verify_only:
        for dwell in (10, 50):
            factory = partial(RecoveryTimingPlanner, recovery=RecoveryConfig(96, dwell))
            jobs = [(s, 2000) for s in REQUIREMENT_SCENARIOS]
            rows, _ = batched_planned_results(model, config, jobs, 2,
                                               scheduler_factory=factory)
            indexed = {r["group"]: r["value"]["evaluation"] for r in rows}
            for scenario, seed in jobs:
                world, _ = reporting_world(scenario, seed)
                scheduler = factory(model, config)
                scheduler.set_detection_probability(world.receiver.detection_probability)
                serial = evaluate_scheduler(world, scheduler, step_seconds=0.001, reward=reward,
                    reward_description="observed_hit - 0.05 * retuning")
                compare_reports(serial, indexed[f"{scenario}:{seed}"])
        print("Serial and batched CUDA coverage reports agree in six worlds.", flush=True)
        return 0
    candidates = {"incumbent": (config, None)}
    for revisit in (128, 192, 256):
        candidates[f"uniform-{revisit}"] = (replace(config, revisit=revisit), None)
    for revisit in (96, 192):
        for dwell in (10, 50):
            for fraction in (0.15, 0.25):
                recovery = RecoveryConfig(revisit, dwell, fraction)
                candidates[f"recovery-{revisit}-{dwell}-{fraction}"] = (config, recovery)
    jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in range(2000, 2000 + args.runs)]
    with run_lock(args.run_dir):
        if (args.run_dir / "selection.json").exists():
            raise ValueError("selection already exists")
        results = {}
        for name, (policy, recovery) in candidates.items():
            factory = (CalibratedTimingPlannerPolicy if recovery is None else
                       partial(RecoveryTimingPlanner, recovery=recovery))
            rows, profile = batched_planned_results(model, policy, jobs, args.batch_size,
                                                    scheduler_factory=factory)
            results[name] = {"policy": asdict(policy), "recovery": (
                asdict(recovery) if recovery is not None else None),
                "means": selection_means(rows), "profile": profile, "results": rows}
            write_json(args.run_dir / "progress.json", results)
            print(json.dumps({"candidate": name, "means": results[name]["means"]}), flush=True)
        incumbent = results["incumbent"]["means"]
        eligible = [n for n, row in results.items() if all(
            row["means"][s]["capture"] >= 0.90 * incumbent[s]["capture"]
            and row["means"][s]["discovery"] >= incumbent[s]["discovery"]
            for s in REQUIREMENT_SCENARIOS)]
        selected = max(eligible, key=lambda n: (
            np.mean([v["discovery"] for v in results[n]["means"].values()]),
            np.mean([v["capture"] for v in results[n]["means"].values()])))
        if fingerprint(args.checkpoint) != checkpoint_digest or fingerprint(
            Path(__file__).parent.parent / "timing_coverage.py"
        ) != coverage_digest:
            raise ValueError("model or coverage implementation changed during selection")
        report = {"checkpoint_sha256": checkpoint_digest,
            "selection_seeds": list(range(2000, 2000 + args.runs)),
            "rule": "retain >=90% capture, no discovery loss per scenario; maximize discovery",
            "selected": selected, "eligible": eligible, "candidates": results,
            "scope": "development selection only; no guaranteed discovery",
            "source_sha256": coverage_digest}
        write_json(args.run_dir / "selection.json", report)
        print(json.dumps({"selected": selected, "eligible": eligible}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
