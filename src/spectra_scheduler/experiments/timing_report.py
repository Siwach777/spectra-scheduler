"""Frozen paired reporting with bounded CUDA batches and parallel CPU controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import time
from collections import Counter
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from ..evaluation_contract import EvaluationAccumulator, TruthOutcome
from ..metrics import calculate_metrics
from ..policy_benchmark import PolicySpec, _run_jobs, summarize
from ..scenarios import REQUIREMENT_SCENARIO_VERSION, REQUIREMENT_SCENARIOS, build_scenario
from ..simulation import SimulationEpisode, configure_scheduler
from ..synthetic_evaluation import evaluate_scheduler
from ..timing_belief import BeliefPolicyConfig, load_belief
from .calibrated_timing import (
    CalibratedBeliefPolicy,
    CalibratedPhasePolicy,
    configure_public_detection,
)
from .storage import fingerprint, run_lock, write_json
from .timing_study import ObservedRateScheduler, controls


def reporting_world(scenario, seed):
    key = f"synthetic-v{REQUIREMENT_SCENARIO_VERSION}:val:{scenario}:{seed}".encode()
    world_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return build_scenario(scenario, world_seed), world_seed


def reward(observation):
    return float(observation.hit) - 0.05 * (not observation.listening)


class BatchedEpisode:
    """Score supplied causal macro actions with the common evaluation contract."""

    def __init__(self, simulation, scheduler):
        self.simulation, self.scheduler = simulation, scheduler
        self.episode = SimulationEpisode(simulation)
        self.totals = Counter(event.time_step for event in self.episode.transmissions)
        self.evaluation = EvaluationAccumulator()
        configure_scheduler(simulation, scheduler)
        if hasattr(scheduler, "set_detection_probability"):
            configure_public_detection(simulation, scheduler)

    def advance(self, action, forecast):
        start = self.episode.time_step
        observations = self.episode.step_action(action)
        records = self.episode.records[-len(observations) :]
        first = next((r.observation.time_step for r in records if r.detected_emitters), None)
        outcome = TruthOutcome(
            elapsed_seconds=len(observations) * 0.001,
            truth_count=sum(self.totals[t] for t in range(start, self.episode.time_step)),
            eligible_count=sum(
                len(self.episode.events.get((r.observation.time_step, action.band), ()))
                for r in records
                if r.observation.listening
            ),
            detectable_count=sum(len(r.detectable_emitters) for r in records),
            captured_count=sum(len(r.detected_emitters) for r in records),
            first_intercept_seconds=(first - start) * 0.001 if first is not None else None,
            negative_opportunities=sum(
                r.observation.listening and not r.detectable_emitters for r in records
            ),
            false_alarms=sum(r.false_alarm for r in records),
        )
        total_reward = 0.0
        for observation in observations:
            total_reward += reward(observation)
            self.scheduler.observe(observation)
        self.evaluation.add(outcome, total_reward, forecast)

    def report(self):
        report = self.evaluation.report()
        support = report["false_alarm_support"]
        scan = calculate_metrics(self.episode.result())
        report.update(
            contract_version=2,
            target="synthetic_selected_band_listening_dwell_including_retune",
            action_contract="SyntheticAction(band, dwell_steps): listening ticks after retune",
            backend="synthetic_discrete",
            step_seconds=0.001,
            reward_description="observed_hit - 0.05 * retuning",
            false_alarm_status="modelled",
            false_alarm_support={
                "modelled_action_windows": support["modelled_windows"],
                "negative_listening_ticks": support["negative_windows"],
                "false_alarm_ticks": support["false_positive_windows"],
            },
            sensitivity={
                "threshold": self.simulation.receiver.sensitivity_dbm,
                "unit": "simulated_dbm",
                "status": "configured",
                "calibrated_dbm": False,
            },
            discovery={
                "emitter_discovery_ratio": scan.emitter_discovery_ratio,
                "mean_first_detection_delay_seconds": scan.mean_first_detection_delay * 0.001,
                "reacquisition_ratio": scan.reacquisition_ratio,
                "mean_reacquisition_delay_seconds": scan.mean_reacquisition_delay * 0.001,
                "total_emitter_changes": scan.total_emitter_changes,
                "reacquired_changes": scan.reacquired_changes,
            },
        )
        return report


@torch.inference_mode()
def batched_neural_results(model, config, jobs, batch_size):
    """At most batch_size episodes; one transfer and forward per decision round."""
    bands = 8
    shape = (batch_size, bands, 3, model.config.history)
    host = torch.empty(shape, pin_memory=True)
    host_array = host.numpy()
    device = torch.empty(shape, device="cuda")
    output = []
    latency_seconds, decisions = 0.0, 0
    for offset in range(0, len(jobs), batch_size):
        active = []
        for scenario, seed in jobs[offset : offset + batch_size]:
            world, world_seed = reporting_world(scenario, seed)
            if world.num_bands != bands:
                raise ValueError("reporting worlds must share the configured band count")
            scheduler = CalibratedBeliefPolicy(model, config)
            active.append((scenario, seed, world_seed, BatchedEpisode(world, scheduler)))
        while active:
            started = time.perf_counter()
            for i, (_, _, _, state) in enumerate(active):
                host_array[i] = state.scheduler.history.encode()
            device[: len(active)].copy_(host[: len(active)], non_blocking=True)
            predictions = model(device[: len(active)]).cpu().numpy()
            latency_seconds += time.perf_counter() - started
            decisions += len(active)
            remaining = []
            for (_, _, _, state), predicted in zip(active, predictions, strict=True):
                step = state.episode.time_step
                action = state.scheduler.select(step, predicted)
                state.advance(action, state.scheduler.forecast(step, action))
            for scenario, seed, world_seed, state in active:
                if state.episode.time_step < state.simulation.duration:
                    remaining.append((scenario, seed, world_seed, state))
                else:
                    evaluation = state.report()
                    output.append(
                        {
                            "group": f"{scenario}:{seed}",
                            "scenario": scenario,
                            "seed": seed,
                            "world_seed": world_seed,
                            "value": {
                                "evaluation": evaluation,
                                "discovery_fraction": evaluation["discovery"][
                                    "emitter_discovery_ratio"
                                ],
                            },
                        }
                    )
            active = remaining
    return output, {
        "batch_size": batch_size,
        "decision_count": decisions,
        "neural_seconds": latency_seconds,
        "amortized_neural_seconds_per_decision": latency_seconds / max(decisions, 1),
    }


def _control_job(job):
    scenario, seed, policy_config, phase_config = job
    simulation, world_seed = reporting_world(scenario, seed)
    policies = controls(policy_config)
    policies.append(
        PolicySpec(
            "beta-phase",
            partial(CalibratedPhasePolicy, phase_config),
            "independently selected nonlearned phase control",
        )
    )
    truth = simulation.generate_truth()
    results = {}
    for specification in policies:
        scheduler = specification.factory()
        if hasattr(scheduler, "set_detection_probability"):
            configure_public_detection(simulation, scheduler)
        evaluation = evaluate_scheduler(
            simulation,
            scheduler,
            step_seconds=0.001,
            reward=reward,
            reward_description="observed_hit - 0.05 * retuning",
            truth=truth,
        )
        results[specification.name] = {
            "evaluation": evaluation,
            "discovery_fraction": evaluation["discovery"]["emitter_discovery_ratio"],
        }
    return {
        "group": f"{scenario}:{seed}",
        "scenario": scenario,
        "seed": seed,
        "world_seed": world_seed,
        "policies": results,
    }


def verify_batching(checkpoint, config, batch_size):
    model, _ = load_belief(checkpoint)
    jobs = [(scenario, seed) for scenario in REQUIREMENT_SCENARIOS for seed in (2000, 2001)]
    batched, profiling = batched_neural_results(model, config, jobs, batch_size)
    values = {row["group"]: row["value"]["evaluation"] for row in batched}
    for scenario, seed in jobs:
        simulation, _ = reporting_world(scenario, seed)
        scheduler = CalibratedBeliefPolicy(model, config)
        configure_public_detection(simulation, scheduler)
        serial = evaluate_scheduler(
            simulation,
            scheduler,
            step_seconds=0.001,
            reward=reward,
            reward_description="observed_hit - 0.05 * retuning",
        )
        result = values[f"{scenario}:{seed}"]
        for key in ("counts", "windows", "false_alarm_support", "discovery"):
            if result[key] != serial[key]:
                raise ValueError(f"batched evaluation changed {key} for {scenario}:{seed}")
        for key, expected in serial["prediction"].items():
            actual = result["prediction"][key]
            equal = (
                math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7)
                if isinstance(expected, float)
                else actual == expected
            )
            if not equal:
                raise ValueError(f"batched evaluation changed prediction.{key}")
    return {"selection_worlds": len(jobs), "matches_serial_cuda": True, "inference": profiling}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--untrained", type=Path, required=True)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--inference-batch-size", type=int, default=20)
    parser.add_argument("--revisit", type=int, default=256)
    parser.add_argument("--probe", type=int, default=8)
    parser.add_argument("--phase-revisit", type=int, default=256)
    parser.add_argument("--phase-probe", type=int, default=4)
    parser.add_argument("--verify-batching", action="store_true")
    args = parser.parse_args(arguments)
    if min(args.runs, args.seed, args.workers, args.inference_batch_size) < 1:
        parser.error("positive sizes and reporting seed required")
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    seeds = tuple(range(args.seed, args.seed + args.runs))
    selection_documents = [json.loads(p.read_text()) for p in args.selection]
    if any(set(seeds) & set(d["seeds"]) for d in selection_documents):
        raise ValueError("reporting seeds overlap selection")
    config = BeliefPolicyConfig(revisit=args.revisit, probe=args.probe, exploration=0.02)
    phase_config = BeliefPolicyConfig(
        revisit=args.phase_revisit, probe=args.phase_probe, exploration=0.02
    )
    if args.verify_batching:
        print(json.dumps(verify_batching(args.checkpoint, config, args.inference_batch_size)))
        return 0
    package = Path(__file__).resolve().parent.parent
    frozen = {
        "version": 1,
        "split": "val",
        "reporting_seeds": list(seeds),
        "scenarios": list(REQUIREMENT_SCENARIOS),
        "trained_policy": asdict(config),
        "untrained_policy": asdict(config),
        "phase_policy": asdict(phase_config),
        "selection_sources": {str(p): fingerprint(p) for p in args.selection},
        "selection_objective": "equal-scenario mean capture + 0.05 * mean discovery",
        "ratio_calibration": "public detection probability; assumes above-sensitivity signals",
        "source_sha256": {
            n: fingerprint(package / n)
            for n in (
                "timing_belief.py",
                "experiments/timing_report.py",
                "experiments/calibrated_timing.py",
                "experiments/phase_selection.py",
                "simulation.py",
                "synthetic_evaluation.py",
                "evaluation_contract.py",
                "scenarios.py",
                "receiver.py",
                "emitters.py",
            )
        },
    }
    started = time.perf_counter()
    with run_lock(args.run_dir):
        if (args.run_dir / "frozen.json").exists():
            raise ValueError("reporting directory already frozen; refuse to overwrite")
        for name, source in (("trained", args.checkpoint), ("untrained", args.untrained)):
            target = args.run_dir / f"{name}.pt"
            shutil.copyfile(source, target)
            frozen[name + "_sha256"] = fingerprint(target)
        write_json(args.run_dir / "frozen.json", frozen)
        jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in seeds]
        results = _run_jobs(
            _control_job, ((s, seed, config, phase_config) for s, seed in jobs), args.workers
        )
        rows = {r["group"]: r for r in results}
        inference = {}
        metadata = {}
        for name in ("trained", "untrained"):
            model, metadata[name] = load_belief(args.run_dir / f"{name}.pt")
            values, inference[name] = batched_neural_results(
                model, config, jobs, args.inference_batch_size
            )
            for value in values:
                rows[value["group"]]["policies"][name] = value["value"]
            del model
            print(
                json.dumps({"phase": name, "episodes": len(values), "inference": inference[name]}),
                flush=True,
            )
        policies = controls(config) + [
            PolicySpec(
                "beta-phase",
                partial(CalibratedPhasePolicy, phase_config),
                json.dumps(asdict(phase_config)),
            ),
            PolicySpec("trained", ObservedRateScheduler, frozen["trained_sha256"]),
            PolicySpec("untrained", ObservedRateScheduler, frozen["untrained_sha256"]),
        ]
        report = {
            "schema_version": 1,
            "backend": "synthetic",
            "split": "val",
            "seeds": list(seeds),
            "requirement_scenario_version": REQUIREMENT_SCENARIO_VERSION,
            "scenarios": list(REQUIREMENT_SCENARIOS),
            "baseline": "beta-phase",
            "policies": [{"name": p.name, "provenance": p.provenance} for p in policies],
            "summary": summarize(results, policies, "beta-phase"),
            "by_scenario": {
                s: summarize([r for r in results if r["scenario"] == s], policies, "beta-phase")
                for s in REQUIREMENT_SCENARIOS
            },
            "attribution": {
                b: {
                    s: summarize([r for r in results if r["scenario"] == s], policies, b)
                    for s in REQUIREMENT_SCENARIOS
                }
                for b in ("untrained", "observed-rate")
            },
            "results": results,
            "frozen": frozen,
            "model_metadata": metadata,
            "scope": "fresh development reporting worlds; one learned seed; test worlds unused",
            "resources": {
                "elapsed_seconds": time.perf_counter() - started,
                "workers": args.workers,
                "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "inference": inference,
            },
        }
        for name in ("trained", "untrained"):
            if fingerprint(args.run_dir / f"{name}.pt") != frozen[name + "_sha256"]:
                raise RuntimeError("frozen checkpoint changed during reporting")
        write_json(args.run_dir / "comparison.json", report)
        print(
            json.dumps(
                {
                    s: {p: m["interception_ratio"]["mean"] for p, m in v.items()}
                    for s, v in report["by_scenario"].items()
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
