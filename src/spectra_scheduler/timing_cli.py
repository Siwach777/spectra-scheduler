"""Main-console comparison of trained timing policies with paired controls."""

from __future__ import annotations

import csv
import resource
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from .comparison import scheduler_factories
from .experiments.calibrated_timing import CalibratedPhasePolicy, configure_public_detection
from .experiments.planner_study import batched_planned_results
from .experiments.storage import fingerprint, write_json
from .experiments.timing_mpc_compare import batched_mpc, load_mpc
from .experiments.timing_report import reporting_world, reward
from .policy_benchmark import PolicySpec, _run_jobs, summarize
from .planner_native import runtime_details
from .scenario_io import build_scenario_from_definition, load_scenario_definition, scenario_name
from .scenarios import REQUIREMENT_SCENARIOS
from .simulation import SyntheticAction
from .synthetic_evaluation import evaluate_scheduler
from .timing_belief import BeliefPolicyConfig
from .timing_ensemble import load_predictor
from .timing_planner import ForecastPlannerWorkspace, TimingPlannerPolicy


class ListeningRoundRobin:
    """Sweep with complete listening dwells after retuning, matching ML actions."""

    def __init__(self, dwell):
        self.dwell = dwell

    def reset(self, bands):
        self.bands, self.next_band = bands, 0

    def choose_action(self, step):
        action = SyntheticAction(self.next_band, self.dwell)
        self.next_band = (self.next_band + 1) % self.bands
        return action

    def observe(self, observation):
        pass


class PhasePlannerControl(CalibratedPhasePolicy):
    """Non-neural phase forecasts with the same planning and coverage as ML."""

    def __init__(self, config):
        super().__init__(config)
        self.model = SimpleNamespace(config=self.belief)
        self.coverage = True

    def reset(self, bands):
        super().reset(bands)
        self.workspace = ForecastPlannerWorkspace(1, bands, self.belief.future, self.config.dwells)

    select = TimingPlannerPolicy.select
    coverage_action = TimingPlannerPolicy.coverage_action
    accept_action = TimingPlannerPolicy.accept_action


def console_world(scenario, seed, definition=None):
    if definition is None:
        return reporting_world(scenario, seed)
    return build_scenario_from_definition(definition, seed), seed


def control_job(job):
    scenario, seed, definition, config = job
    world, world_seed = console_world(scenario, seed, definition)
    factories = scheduler_factories(seed)
    factories.update({f"round-robin-listen-{d}": partial(ListeningRoundRobin, d)
                      for d in config.dwells})
    factories["phase-planner"] = partial(PhasePlannerControl, config)
    truth, policies = world.generate_truth(), {}
    for name, factory in factories.items():
        scheduler = factory()
        if hasattr(scheduler, "set_detection_probability"):
            configure_public_detection(world, scheduler)
        evaluation = evaluate_scheduler(
            world, scheduler, step_seconds=0.001, reward=reward,
            reward_description="observed_hit - 0.05 * retuning", truth=truth)
        policies[name] = {"evaluation": evaluation,
                          "discovery_fraction": evaluation["discovery"]["emitter_discovery_ratio"]}
    return {"group": f"{scenario}:{seed}", "scenario": scenario, "seed": seed,
            "world_seed": world_seed, "policies": policies}


def write_timing_report(report, path, output_format=None):
    path = Path(path)
    kind = output_format or path.suffix.lstrip(".").lower()
    if kind not in ("json", "csv"):
        raise ValueError("timing report must be JSON or CSV")
    if kind == "json":
        write_json(path, report)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["policy", "metric", "mean", "groups",
                "paired_groups", "paired_mean_difference", "paired_bootstrap_95_interval"])
            writer.writeheader()
            for policy, metrics in report["summary"].items():
                for metric, values in metrics.items():
                    writer.writerow({"policy": policy, "metric": metric, **values})
    return path


def run_timing_comparison(args):
    definition = load_scenario_definition(args.scenario_file) if args.scenario_file else None
    scenario = (scenario_name(definition, Path(args.scenario_file).stem)
                if definition is not None else args.scenario)
    if args.mpc_checkpoints and (definition is not None or scenario not in REQUIREMENT_SCENARIOS):
        raise ValueError("MPC comparison requires a named eight-band requirement scenario")
    paths = [args.timing_checkpoint, *args.mpc_checkpoints]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"checkpoint does not exist: {path}")
        if args.output and Path(args.output).resolve() == path.resolve():
            raise ValueError("report cannot overwrite an input checkpoint")
    if args.planner_library:
        if not args.planner_library.is_file():
            raise ValueError(f"planner library does not exist: {args.planner_library}")
        if args.output and Path(args.output).resolve() == args.planner_library.resolve():
            raise ValueError("report cannot overwrite the planner library")
    if args.output and (args.output_format or Path(args.output).suffix.lstrip(".").lower()) not in (
        "json", "csv"
    ):
        raise ValueError("timing report must be JSON or CSV")
    if not 0 <= args.seed < 2**32 or args.seed + args.runs > 2**32:
        raise ValueError("seed range must fit unsigned 32-bit integers")
    torch.set_num_threads(1)
    try:
        torch.empty(1, device="cuda")
    except RuntimeError as error:
        raise RuntimeError("trained timing scheduler requires CUDA") from error
    torch.cuda.reset_peak_memory_stats()
    digests = {str(path): fingerprint(path) for path in paths}
    model, metadata = load_predictor(args.timing_checkpoint)
    if args.cuda_graph:
        from .timing_runtime import CapturedTimingPredictor
        first_world, _ = console_world(scenario, args.seed, definition)
        model = CapturedTimingPredictor(model, args.inference_batch_size, first_world.num_bands)
    settings = dict(metadata.get("policy") or metadata.get("semantic", {}).get("policy") or {})
    if not settings:
        raise ValueError("timing checkpoint does not include its scheduling policy")
    settings["dwells"] = tuple(settings["dwells"])
    config = BeliefPolicyConfig(**settings)
    jobs = [(scenario, seed) for seed in range(args.seed, args.seed + args.runs)]
    started = time.perf_counter()
    results = _run_jobs(control_job,
        ((scenario, seed, definition, config) for _, seed in jobs), min(args.workers, args.runs))
    by_group = {row["group"]: row for row in results}
    values, profile = batched_planned_results(
        model, config, jobs, min(args.inference_batch_size, args.runs),
        world_factory=partial(console_world, definition=definition))
    for row in values:
        by_group[row["group"]]["policies"]["timing-planner"] = row["value"]
    del model
    profiles = {"timing-planner": profile}
    for index, path in enumerate(args.mpc_checkpoints):
        model, mpc_config = load_mpc(path)
        name = f"mpc-{index}-{mpc_config.search_method}"
        values, profiles[name] = batched_mpc(
            model, mpc_config, jobs, min(args.inference_batch_size, args.runs))
        for row in values:
            by_group[row["group"]]["policies"][name] = row["value"]
        del model
    for row in results:
        evaluations = [v["evaluation"] for v in row["policies"].values()]
        if (len({v["counts"]["truth"] for v in evaluations}) != 1
                or len({round(v["elapsed_seconds"], 9) for v in evaluations}) != 1):
            raise ValueError("paired policies received different worlds or time budgets")
    if any(fingerprint(path) != digests[str(path)] for path in paths):
        raise ValueError("checkpoint changed during comparison")
    names = list(results[0]["policies"])
    baseline = f"round-robin-listen-{max(config.dwells)}"
    policies = [PolicySpec(name, ListeningRoundRobin, name) for name in names]
    report = {
        "schema_version": 2, "backend": "synthetic_timing", "scenario": scenario,
        "scenario_definition": definition, "start_seed": args.seed, "runs": args.runs,
        "checkpoint_sha256": digests, "policy": asdict(config), "baseline": baseline,
        "inference_runtime": {"cuda_graph": args.cuda_graph, **runtime_details()},
        "summary": summarize(results, policies, baseline), "results": results,
        "resources": {"elapsed_seconds": time.perf_counter() - started, "profiles": profiles,
                      "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                      "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "scope": "paired synthetic development comparison; no training or policy selection",
    }
    print(f"Spectra Scheduler - trained timing comparison ({scenario}, {args.runs} runs)")
    print(f"{'policy':<25} {'capture':>9} {'discovery':>10} {'P(detect)':>10} {'P(false)':>10}")
    for name, metrics in report["summary"].items():
        values = [metrics[m]["mean"] for m in ("interception_ratio", "discovery_fraction",
                  "probability_of_detection", "probability_of_false_alarm")]
        columns = [f"{v:>9.1%}" if v is not None else f"{'n/a':>9}" for v in values]
        print(f"{name:<25} " + " ".join(columns))
    if args.output:
        path = write_timing_report(report, args.output, args.output_format)
        print(f"Report written to {path}")
    return 0
