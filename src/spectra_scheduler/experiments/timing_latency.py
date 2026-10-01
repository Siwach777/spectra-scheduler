"""Serial observation ingestion, CUDA forecast, planning and forecast-export latency."""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..scenarios import REQUIREMENT_SCENARIOS
from ..synthetic_evaluation import evaluate_scheduler
from ..timing_belief import BeliefPolicyConfig
from ..timing_cli import PhasePlannerControl
from ..timing_ensemble import load_predictor
from ..timing_planner import CalibratedTimingPlannerPolicy
from ..timing_runtime import CapturedTimingPredictor
from .storage import fingerprint, write_json
from .timing_report import reporting_world, reward


class MeasuredPath:
    """Include feedback processing since the last decision; exclude simulator truth."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def set_retune_table(self, table):
        self.scheduler.set_retune_table(table)

    def set_episode_horizon(self, horizon):
        self.scheduler.set_episode_horizon(horizon)

    def reset(self, bands):
        self.scheduler.reset(bands)
        self.samples, self.compute_samples = [], []
        self.feedback_seconds = 0.0

    def observe(self, observation):
        start = perf_counter()
        self.scheduler.observe(observation)
        self.feedback_seconds += perf_counter() - start

    def choose_action(self, step):
        start = perf_counter()
        action = self.scheduler.choose_action(step)
        self._forecast = self.scheduler.forecast(step, action)
        compute = perf_counter() - start
        self.compute_samples.append(compute)
        self.samples.append(compute + self.feedback_seconds)
        self.feedback_seconds = 0.0
        return action

    def forecast(self, step, action):
        return self._forecast


def distribution(samples, deadline_ms):
    values = np.asarray(samples) * 1000
    return {"decisions": len(values), "p50_ms": float(np.quantile(values, 0.5)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "p99_ms": float(np.quantile(values, 0.99)), "maximum_ms": float(values.max()),
            "fraction_over_budget": float(np.mean(values > deadline_ms))}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--deadline-ms", type=float, default=1.0)
    parser.add_argument("--planner-library", type=Path)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args(arguments)
    if args.runs < 1 or not np.isfinite(args.deadline_ms) or args.deadline_ms <= 0:
        parser.error("positive episode count and latency budget required")
    if args.output.resolve() in {p.resolve() for p in (
        args.checkpoint, args.planner_library
    ) if p is not None}:
        parser.error("latency report cannot overwrite an input")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    if args.planner_library:
        os.environ["SPECTRA_PLANNER_LIBRARY"] = str(args.planner_library.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("neural serial latency validation requires CUDA")
    digest = fingerprint(args.checkpoint)
    model, metadata = load_predictor(args.checkpoint)
    if args.cuda_graph:
        model = CapturedTimingPredictor(model)
    settings = dict(metadata["policy"])
    settings["dwells"] = tuple(settings["dwells"])
    config = BeliefPolicyConfig(**settings)
    results = {}
    for name, factory in (
        ("cuda-timing", lambda: CalibratedTimingPlannerPolicy(model, config)),
        ("cpu-phase", lambda: PhasePlannerControl(config)),
    ):
        scenarios = {}
        for scenario in REQUIREMENT_SCENARIOS:
            samples, compute, cold = [], [], None
            for seed in [2000, *range(52000, 52000 + args.runs)]:
                world, _ = reporting_world(scenario, seed)
                scheduler = factory()
                scheduler.set_detection_probability(world.receiver.detection_probability)
                path = MeasuredPath(scheduler)
                evaluate_scheduler(world, path, step_seconds=0.001, reward=reward,
                                   reward_description="observed_hit - 0.05 * retuning")
                if seed == 2000:
                    cold = path.samples[0] * 1000
                else:
                    samples.extend(path.samples)
                    compute.extend(path.compute_samples)
            scenarios[scenario] = {
                "full_path": distribution(samples, args.deadline_ms),
                "decision_compute": distribution(compute, args.deadline_ms),
                "first_warmup_decision_ms": cold,
            }
        results[name] = scenarios
    if fingerprint(args.checkpoint) != digest:
        raise ValueError("checkpoint changed during latency measurement")
    report = {"schema_version": 1, "checkpoint_sha256": digest,
        "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
        "torch_threads": 1, "inference_batch": 1,
        "cuda_graph": args.cuda_graph,
        "budget_ms": args.deadline_ms, "minimum_listening_dwell_ms": min(config.dwells),
        "planner_library_sha256": (fingerprint(args.planner_library)
                                   if args.planner_library else None),
        "seeds": list(range(52000, 52000 + args.runs)), "results": results,
        "scope": "host serial measurement; declared software budget, not hardware certification",
        "path": "feedback ingestion + history encoding + inference + planning + forecast creation",
        "excluded": "checkpoint loading, CUDA graph capture, warmup worlds, simulator truth generation and receiver I/O"}
    write_json(args.output, report)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
