"""Fresh-world coverage reporting against frozen adaptive, phase and MPC controls."""

import argparse
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from ..policy_benchmark import PolicySpec, _run_jobs, summarize
from ..scenarios import REQUIREMENT_SCENARIOS
from ..timing_belief import BeliefPolicyConfig
from ..timing_coverage import (
    AcquisitionConfig,
    AcquisitionTimingPlanner,
    RecoveryConfig,
    RecoveryTimingPlanner,
)
from ..timing_ensemble import load_predictor
from ..timing_planner import CalibratedTimingPlannerPolicy
from .planner_study import batched_planned_results
from .scan_strategy_study import control_job, means, validate_pairs, value
from .storage import fingerprint, run_lock, write_json
from .timing_mpc_compare import batched_mpc, load_mpc


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coverage-selection", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mpc", type=Path, action="append", default=[])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=54000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args(arguments)
    if min(args.runs, args.workers, args.batch_size) < 1:
        parser.error("positive episode, worker and batch settings required")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("coverage reporting requires CUDA")
    selection = json.loads(args.coverage_selection.read_text())
    control_selection = json.loads(args.control_selection.read_text())
    digest = fingerprint(args.checkpoint)
    source = Path(__file__).parent.parent / "timing_coverage.py"
    if selection["checkpoint_sha256"] != digest or selection["source_sha256"] != fingerprint(source):
        raise ValueError("coverage selection checkpoint or implementation changed")
    if control_selection["frozen"]["checkpoint_sha256"] != digest:
        raise ValueError("controls were selected against a different timing checkpoint")
    seeds = list(range(args.seed, args.seed + args.runs))
    used = (set(selection["selection_seeds"]) | set(control_selection["frozen"]["selection_seeds"])
            | set(control_selection["frozen"]["reporting_seeds"]))
    if used.intersection(seeds):
        raise ValueError("fresh reporting seeds overlap selection or previous reporting")
    mode = selection.get("mode", "recovery")
    choice = selection["candidates"][selection["selected"]]
    settings = dict(choice["policy"])
    settings["dwells"] = tuple(settings["dwells"])
    selected_config = BeliefPolicyConfig(**settings)
    intervention = choice[mode]
    factory = (CalibratedTimingPlannerPolicy if intervention is None else
               partial(AcquisitionTimingPlanner, acquisition=AcquisitionConfig(**intervention))
               if mode == "acquisition" else
               partial(RecoveryTimingPlanner, recovery=RecoveryConfig(**intervention)))
    model, metadata = load_predictor(args.checkpoint)
    settings = dict(metadata["policy"])
    settings["dwells"] = tuple(settings["dwells"])
    incumbent_config = BeliefPolicyConfig(**settings)
    definitions = control_selection["frozen"]["statistical_candidates"]
    controls = {name: definitions[name] for name in control_selection["report_controls"]}
    frozen = {"checkpoint_sha256": digest, "coverage_source_sha256": fingerprint(source),
        "coverage_selection_sha256": fingerprint(args.coverage_selection),
        "control_selection_sha256": fingerprint(args.control_selection),
        "seeds": seeds, "incumbent_policy": asdict(incumbent_config),
        "selected_policy": asdict(selected_config), "mode": mode, "intervention": intervention,
        "controls": controls, "mpc_sha256": {str(p): fingerprint(p) for p in args.mpc}}
    for path, sha in frozen["mpc_sha256"].items():
        if control_selection["frozen"]["mpc_sha256"].get(path) != sha:
            raise ValueError("MPC model differs from the frozen control selection")
    jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in seeds]
    started = perf_counter()
    with run_lock(args.run_dir):
        if (args.run_dir / "frozen.json").exists():
            raise ValueError("coverage reporting directory is already frozen")
        write_json(args.run_dir / "frozen.json", frozen)
        rows = _run_jobs(control_job, ((s, seed, controls) for s, seed in jobs), args.workers)
        indexed = {r["group"]: r for r in rows}
        profiles = {}
        for name, config, scheduler in (
            ("timing-incumbent", incumbent_config, CalibratedTimingPlannerPolicy),
            ("timing-acquisition", selected_config, factory),
        ):
            reports, profiles[name] = batched_planned_results(model, config, jobs, args.batch_size,
                                                             scheduler_factory=scheduler)
            for row in reports:
                indexed[row["group"]]["policies"][name] = value(row["value"]["evaluation"])
            print(json.dumps({"policy": name, "means": means(rows, name)}), flush=True)
        for index, path in enumerate(args.mpc):
            name = f"mpc-{index}"
            mpc, config = load_mpc(path)
            reports, profiles[name] = batched_mpc(mpc, config, jobs, args.batch_size)
            for row in reports:
                indexed[row["group"]]["policies"][name] = value(row["value"]["evaluation"])
            del mpc
        validate_pairs(rows)
        if fingerprint(args.checkpoint) != digest or fingerprint(source) != frozen[
            "coverage_source_sha256"]:
            raise ValueError("checkpoint or coverage implementation changed during reporting")
        names = list(rows[0]["policies"])
        specs = [PolicySpec(name, lambda: None, name) for name in names]
        comparisons = {baseline: {s: summarize([r for r in rows if r["scenario"] == s],
                        specs, baseline) for s in REQUIREMENT_SCENARIOS}
                       for baseline in ("round-robin-50", "phase-planner", "timing-incumbent")}
        write_json(args.run_dir / "comparison.json", {"schema_version": 1,
            "frozen": frozen, "means": {name: means(rows, name) for name in names},
            "comparisons": comparisons, "results": rows,
            "profiles": profiles, "elapsed_seconds": perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "scope": "fresh synthetic development worlds; selection unchanged; not hardware"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
