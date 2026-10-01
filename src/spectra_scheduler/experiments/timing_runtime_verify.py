"""Verify accelerated CUDA decisions against a frozen paired-world benchmark."""

import argparse
import json
import os
from pathlib import Path

import torch

from ..timing_belief import BeliefPolicyConfig
from ..timing_ensemble import load_predictor
from ..timing_planner import CalibratedTimingPlannerPolicy
from ..timing_runtime import CapturedTimingPredictor
from .calibrated_timing import configure_public_detection
from .planner_study import batched_planned_results
from .storage import fingerprint, write_json
from .timing_refine_report import compare_reports
from .timing_report import reporting_world, reward
from ..synthetic_evaluation import evaluate_scheduler


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--planner-library", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    if args.batch_size < 1:
        parser.error("inference batch must be positive")
    if args.output.resolve() in {p.resolve() for p in (
        args.checkpoint, args.reference_report, args.planner_library
    )}:
        parser.error("verification cannot overwrite an input")
    if not torch.cuda.is_available():
        raise RuntimeError("timing runtime verification requires CUDA")
    torch.set_num_threads(1)
    os.environ["SPECTRA_PLANNER_LIBRARY"] = str(args.planner_library.resolve())
    snapshot = {str(p): fingerprint(p) for p in (
        args.checkpoint, args.reference_report, args.planner_library
    )}
    reference = json.loads(args.reference_report.read_text())
    if fingerprint(args.checkpoint) != reference["frozen"]["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the frozen benchmark")
    settings = dict(reference["frozen"]["policy"])
    settings["dwells"] = tuple(settings["dwells"])
    config = BeliefPolicyConfig(**settings)
    jobs = [(r["scenario"], r["seed"]) for r in reference["results"]]
    expected = {r["group"]: r["policies"]["timing-trained"]["evaluation"]
                for r in reference["results"]}
    if len(jobs) != len(expected):
        raise ValueError("reference contains duplicate worlds")
    model, _ = load_predictor(args.checkpoint)
    captured = CapturedTimingPredictor(model, args.batch_size)
    rows, profile = batched_planned_results(captured, config, jobs, args.batch_size)
    for row in rows:
        compare_reports(row["value"]["evaluation"], expected[row["group"]])
    del captured
    captured = CapturedTimingPredictor(model)
    serial_worlds = []
    for scenario in reference["frozen"]["scenarios"]:
        for seed in reference["frozen"]["reporting_seeds"][:2]:
            world, _ = reporting_world(scenario, seed)
            scheduler = CalibratedTimingPlannerPolicy(captured, config)
            configure_public_detection(world, scheduler)
            actual = evaluate_scheduler(world, scheduler, step_seconds=0.001, reward=reward,
                reward_description="observed_hit - 0.05 * retuning")
            compare_reports(actual, expected[f"{scenario}:{seed}"])
            serial_worlds.append(f"{scenario}:{seed}")
    if any(fingerprint(Path(path)) != digest for path, digest in snapshot.items()):
        raise ValueError("verification inputs changed during evaluation")
    source = Path(__file__).parent.parent
    result = {"schema_version": 1, "inputs_sha256": snapshot,
        "source_sha256": {name: fingerprint(source / name) for name in (
            "timing_runtime.py", "planner_native.py", "_planner.cpp", "timing_planner.py")},
        "batched_worlds": len(rows), "serial_worlds": serial_worlds,
        "exact_counts_and_report_parity": True, "continuous_rtol": 5e-5,
        "continuous_atol": 1e-5, "profile": profile,
        "scope": "same frozen checkpoint and policy; runtime changes only"}
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
