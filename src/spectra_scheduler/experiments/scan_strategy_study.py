"""Select Whittle and scan handover strategies, then compare on fresh worlds."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..policy_benchmark import PolicySpec, _run_jobs, summarize
from ..scan_handover import PhasedTimingPlanner
from ..scenarios import REQUIREMENT_SCENARIOS
from ..synthetic_evaluation import evaluate_scheduler
from ..timing_belief import BeliefPolicyConfig
from ..timing_cli import ListeningRoundRobin, PhasePlannerControl
from ..timing_ensemble import load_predictor
from ..timing_planner import CalibratedTimingPlannerPolicy
from ..whittle_policy import ScanStrategyConfig, WhittleScanScheduler
from .calibrated_timing import configure_public_detection
from .planner_study import batched_planned_results
from .storage import fingerprint, run_lock, write_json
from .timing_mpc_compare import batched_mpc, load_mpc
from .timing_refine_report import compare_reports
from .timing_report import reporting_world, reward


def candidates(policy):
    statistical = {}
    for dwell in (1, 10, 50):
        statistical[f"round-robin-{dwell}"] = {"kind": "round-robin", "dwell": dwell}
        statistical[f"golden-{dwell}"] = {
            "kind": "scan-strategy", "config": asdict(
                ScanStrategyConfig(strategy="golden", dwell=dwell))}
        for weight in (0.0, 1.0):
            statistical[f"whittle-{dwell}-age{weight:g}"] = {
                "kind": "scan-strategy", "config": asdict(
                    ScanStrategyConfig(dwell=dwell, coverage_weight=weight))}
            statistical[f"markov-whittle-{dwell}-age{weight:g}"] = {
                "kind": "scan-strategy", "config": asdict(ScanStrategyConfig(
                    dwell=dwell, coverage_weight=weight, belief_mode="markov"))}
    for dwell in (1, 10):
        for patience in (64, 128):
            statistical[f"phased-whittle-{dwell}-{patience}"] = {
                "kind": "scan-strategy", "config": asdict(
                    ScanStrategyConfig(strategy="phased", dwell=dwell, patience=patience))}
        statistical[f"adaptive-whittle-{dwell}"] = {
            "kind": "scan-strategy", "config": asdict(
                ScanStrategyConfig(strategy="adaptive", dwell=dwell))}
    statistical["phase-planner"] = {"kind": "phase", "config": asdict(policy)}
    neural = {"timing-trained": None}
    for dwell in (1, 10):
        neural[f"timing-phased-{dwell}-64"] = asdict(
            ScanStrategyConfig(strategy="phased", dwell=dwell, patience=64))
        neural[f"timing-adaptive-{dwell}"] = asdict(
            ScanStrategyConfig(strategy="adaptive", dwell=dwell))
    neural["timing-phased-10-128"] = asdict(
        ScanStrategyConfig(strategy="phased", dwell=10, patience=128))
    return statistical, neural


def factory(definition):
    if definition["kind"] == "scan-strategy":
        return partial(WhittleScanScheduler, ScanStrategyConfig(**definition["config"]))
    if definition["kind"] == "round-robin":
        return partial(ListeningRoundRobin, definition["dwell"])
    settings = dict(definition["config"])
    settings["dwells"] = tuple(settings["dwells"])
    return partial(PhasePlannerControl, BeliefPolicyConfig(**settings))


def control_job(job):
    scenario, seed, definitions = job
    world, world_seed = reporting_world(scenario, seed)
    truth = world.generate_truth()
    policies = {}
    for name, definition in definitions.items():
        scheduler = factory(definition)()
        if hasattr(scheduler, "set_observation_probabilities"):
            scheduler.set_observation_probabilities(world.receiver.detection_probability,
                                                    world.receiver.false_alarm_probability)
        if hasattr(scheduler, "set_detection_probability"):
            configure_public_detection(world, scheduler)
        started = time.perf_counter()
        evaluation = evaluate_scheduler(world, scheduler, step_seconds=0.001, reward=reward,
            reward_description="observed_hit - 0.05 * retuning", truth=truth)
        policies[name] = value(evaluation)
        policies[name]["elapsed_seconds"] = time.perf_counter() - started
        if isinstance(scheduler, WhittleScanScheduler):
            policies[name]["strategy_diagnostics"] = {
                "handover_tick": scheduler.switch.switched_at,
                "declared_hit_bands": int(scheduler.switch.seen.sum()),
                "numerical_indexability_violation_bands": sorted(scheduler.indexability_violations),
            }
    return {"group": f"{scenario}:{seed}", "scenario": scenario, "seed": seed,
            "world_seed": world_seed, "policies": policies}


def value(evaluation):
    discovery = evaluation["discovery"]
    return {"evaluation": evaluation, "discovery_fraction": discovery["emitter_discovery_ratio"],
            "mean_discovery_delay_us": discovery["mean_first_detection_delay_seconds"] * 1e6}


def neural_factory(acquisition):
    return (CalibratedTimingPlannerPolicy if acquisition is None else
            partial(PhasedTimingPlanner, acquisition=ScanStrategyConfig(**acquisition)))


def means(results, name):
    output = {}
    for scenario in REQUIREMENT_SCENARIOS:
        rows = [r["policies"][name] for r in results if r["scenario"] == scenario]
        output[scenario] = {
            "capture": float(np.mean([r["evaluation"]["interception_ratio"] for r in rows
                                      if r["evaluation"]["interception_ratio"] is not None])),
            "capture_worlds": sum(r["evaluation"]["interception_ratio"] is not None
                                  for r in rows),
            "discovery": float(np.mean([r["discovery_fraction"] for r in rows])),
            "acquisition_ms": float(np.mean([r["mean_discovery_delay_us"] / 1000 for r in rows])),
        }
    return output


def score(metrics):
    return float(np.mean([v["capture"] + 0.15 * v["discovery"] for v in metrics.values()]))


def validate_pairs(results):
    for row in results:
        evaluations = [v["evaluation"] for v in row["policies"].values()]
        if (len({v["counts"]["truth"] for v in evaluations}) != 1
                or len({round(v["elapsed_seconds"], 9) for v in evaluations}) != 1):
            raise ValueError("paired policies received different truth counts or time budgets")


def verify_neural(model, policy, acquisition):
    jobs = [(scenario, 2000) for scenario in REQUIREMENT_SCENARIOS]
    # Two batches deliberately cover the prior band-count shadowing regression.
    rows, _ = batched_planned_results(model, policy, jobs, 2,
                                     scheduler_factory=neural_factory(acquisition))
    indexed = {r["group"]: r["value"]["evaluation"] for r in rows}
    for scenario, seed in jobs:
        world, _ = reporting_world(scenario, seed)
        scheduler = neural_factory(acquisition)(model, policy)
        configure_public_detection(world, scheduler)
        serial = evaluate_scheduler(world, scheduler, step_seconds=0.001, reward=reward,
                                    reward_description="observed_hit - 0.05 * retuning")
        compare_reports(serial, indexed[f"{scenario}:{seed}"])


def sources():
    package = Path(__file__).resolve().parent.parent
    names = ("whittle_policy.py", "scan_handover.py", "timing_planner.py", "timing_belief.py",
             "scenarios.py", "simulation.py", "receiver.py", "emitters.py", "timing_cli.py",
             "experiments/scan_strategy_study.py", "experiments/planner_study.py",
             "experiments/timing_report.py", "experiments/timing_mpc_compare.py")
    return {name: fingerprint(package / name) for name in names}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("select", "report"), default="select")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--selection-runs", type=int, default=32)
    parser.add_argument("--report-runs", type=int, default=100)
    parser.add_argument("--report-seed", type=int, default=26000)
    parser.add_argument("--mpc", type=Path, action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(arguments)
    if min(args.workers, args.batch_size, args.selection_runs, args.report_runs) < 1:
        parser.error("positive workers, inference batch and episode counts required")
    if args.stage == "report" and args.selection is None:
        parser.error("reporting requires a frozen selection document")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    model, metadata = load_predictor(args.checkpoint)
    settings = dict(metadata["policy"])
    settings["dwells"] = tuple(settings["dwells"])
    policy = BeliefPolicyConfig(**settings)
    statistical, neural = candidates(policy)
    if args.verify_only:
        for name, acquisition in neural.items():
            verify_neural(model, policy, acquisition)
            print(json.dumps({"policy": name, "serial_cuda_parity": True,
                              "multi_batch_worlds": 3}), flush=True)
        return 0
    selection_seeds = list(range(2000, 2000 + args.selection_runs))
    report_seeds = list(range(args.report_seed, args.report_seed + args.report_runs))
    if set(selection_seeds) & set(report_seeds):
        raise ValueError("selection and reporting seeds overlap")
    policy_settings = asdict(policy)
    policy_settings["dwells"] = list(policy.dwells)
    frozen = {"checkpoint_sha256": fingerprint(args.checkpoint), "policy": policy_settings,
              "source_sha256": sources(), "selection_seeds": selection_seeds,
              "reporting_seeds": report_seeds, "scenarios": list(REQUIREMENT_SCENARIOS),
              "adaptation": "single-band native listening dwell after retune; online noisy hits",
              "neural_candidates": neural, "statistical_candidates": statistical,
              "selection_rule": (
                  "neural mean capture with no scenario discovery loss; fallback incumbent"),
              "control_selection_rule": "capture + 0.15 discovery, equal scenario weights",
              "mpc_sha256": {str(p): fingerprint(p) for p in args.mpc}}
    if args.stage == "report":
        selection = json.loads(args.selection.read_text())
        for key in ("checkpoint_sha256", "policy", "source_sha256", "mpc_sha256"):
            if selection["frozen"][key] != frozen[key]:
                raise ValueError(f"frozen selection changed: {key}")
        frozen = selection["frozen"]
        report_seeds = frozen["reporting_seeds"]
        statistical = {name: statistical[name] for name in selection["report_controls"]}
        neural = {name: neural[name] for name in selection["report_neural"]}
    seeds = selection_seeds if args.stage == "select" else report_seeds
    jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in seeds]
    started = time.perf_counter()
    with run_lock(args.run_dir):
        if (args.run_dir / "frozen.json").exists():
            raise ValueError("run directory already frozen; choose a fresh directory")
        write_json(args.run_dir / "frozen.json", frozen)
        results = _run_jobs(control_job, ((s, seed, statistical) for s, seed in jobs), args.workers)
        by_group = {r["group"]: r for r in results}
        print(json.dumps({"phase": "statistical", "worlds": len(results),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
        profiles = {}
        for name, acquisition in neural.items():
            rows, profiles[name] = batched_planned_results(
                model, policy, jobs, args.batch_size,
                scheduler_factory=neural_factory(acquisition))
            for row in rows:
                by_group[row["group"]]["policies"][name] = value(row["value"]["evaluation"])
            print(json.dumps({"policy": name, "means": means(results, name)}), flush=True)
        if args.stage == "report":
            for index, path in enumerate(args.mpc):
                name = f"mpc-{index}"
                mpc, config = load_mpc(path)
                rows, profiles[name] = batched_mpc(mpc, config, jobs, args.batch_size)
                for row in rows:
                    by_group[row["group"]]["policies"][name] = value(row["value"]["evaluation"])
                profiles[name]["saved_settings"] = {
                    "simulations": config.simulations, "depth": config.depth,
                    "search_method": config.search_method, "gamma": config.gamma}
                del mpc
                print(json.dumps({"policy": name, "means": means(results, name)}), flush=True)
        validate_pairs(results)
        if fingerprint(args.checkpoint) != frozen["checkpoint_sha256"] or sources() != frozen[
            "source_sha256"
        ]:
            raise ValueError("model or source changed during comparison")
        names = list(results[0]["policies"])
        specs = [PolicySpec(name, ListeningRoundRobin, name) for name in names]
        report = {"frozen": frozen, "results": results,
                  "means": {name: means(results, name) for name in names},
                  "by_scenario": {s: summarize([r for r in results if r["scenario"] == s],
                      specs, "round-robin-50") for s in REQUIREMENT_SCENARIOS},
                  "paired_to_incumbent": {s: summarize(
                      [r for r in results if r["scenario"] == s], specs, "timing-trained")
                      for s in REQUIREMENT_SCENARIOS},
                  "resources": {"elapsed_seconds": time.perf_counter() - started,
                      "workers": args.workers, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                      "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "profiles": profiles},
                  "scope": "selection worlds" if args.stage == "select" else
                           "fresh synthetic development holdout; frozen choices; not hardware"}
        if args.stage == "select":
            incumbent = report["means"]["timing-trained"]
            eligible = [name for name in neural if all(
                report["means"][name][s]["discovery"] >= incumbent[s]["discovery"]
                for s in REQUIREMENT_SCENARIOS)]
            selected = max(eligible, key=lambda name: np.mean(
                [r["capture"] for r in report["means"][name].values()]))
            best_hybrid = max((n for n in neural if n != "timing-trained"),
                              key=lambda n: score(report["means"][n]))
            controls = ["round-robin-1", "round-robin-10", "round-robin-50", "phase-planner"]
            for family in ("whittle-", "markov-whittle-", "golden-", "phased-whittle-",
                           "adaptive-whittle-"):
                controls.append(max((n for n in statistical if n.startswith(family)),
                                    key=lambda n: score(report["means"][n])))
            report.update(selected_neural=selected, discovery_eligible=eligible,
                          report_controls=controls,
                          report_neural=list(dict.fromkeys(
                              ["timing-trained", selected, best_hybrid])))
            write_json(args.run_dir / "selection.json", report)
        else:
            write_json(args.run_dir / "comparison.json", report)
        print(json.dumps({"stage": args.stage, "means": report["means"],
                          "selected_neural": report.get("selected_neural")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
