"""Fresh paired reporting of planning and controlled timing refinement."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..grouped_policy import load_grouped
from ..policy_benchmark import PolicySpec, _run_jobs, benchmark_synthetic, summarize
from ..scenarios import REQUIREMENT_SCENARIOS
from ..schedulers import DwellSweepScheduler
from ..synthetic_evaluation import evaluate_scheduler
from ..timing_belief import BeliefPolicyConfig
from ..timing_ensemble import load_predictor
from ..timing_planner import CalibratedTimingPlannerPolicy
from .calibrated_timing import configure_public_detection
from .grouped_report import pooled_metrics
from .grouped_study import trajectories
from .planner_study import batched_planned_results
from .storage import fingerprint, run_lock, write_json
from .timing_mpc_compare import batched_mpc, load_mpc
from .timing_report import _control_job, reporting_world, reward


def compare_reports(actual, expected):
    """Counts must match exactly; tolerate only continuous GPU rounding."""
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError("serial/batched report schema differs")
        for key in actual:
            compare_reports(actual[key], expected[key])
    elif isinstance(actual, int):
        if actual != expected:
            raise ValueError(f"serial/batched counts differ: {actual} versus {expected}")
    elif isinstance(actual, float):
        if expected is None or not np.isclose(actual, expected, rtol=5e-5, atol=1e-5):
            raise ValueError(f"serial/batched metric differs: {actual} versus {expected}")
    elif actual != expected:
        raise ValueError("serial/batched report differs")


def verify_planner(path, config):
    model, _ = load_predictor(path)
    jobs = [(scenario, 2000) for scenario in REQUIREMENT_SCENARIOS]
    rows, _ = batched_planned_results(model, config, jobs, len(jobs))
    by_group = {r["group"]: r["value"]["evaluation"] for r in rows}
    for scenario, seed in jobs:
        world, _ = reporting_world(scenario, seed)
        scheduler = CalibratedTimingPlannerPolicy(model, config)
        configure_public_detection(world, scheduler)
        actual = evaluate_scheduler(
            world, scheduler, step_seconds=0.001, reward=reward,
            reward_description="observed_hit - 0.05 * retuning")
        compare_reports(actual, by_group[f"{scenario}:{seed}"])
    return {"selection_worlds": len(jobs), "serial_cuda_parity": True}


def learning_curves(run_dir, output_dir, training_seeds):
    """Selection curves only; fresh reporting never chooses epochs or seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {str(seed): json.loads((run_dir / f"seed-{seed}" / "progress.json").read_text())
              for seed in training_seeds}
    write_json(output_dir / "learning-curves.json", curves)
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for seed, rows in curves.items():
        trained = [r for r in rows if r["loss"] is not None]
        selected = [r for r in rows if "selection" in r]
        axes[0, 0].plot([r["epoch"] for r in trained], [r["loss"] for r in trained],
                        label=f"seed {seed}")
        axes[0, 1].plot([r["epoch"] for r in selected], [r["score"] for r in selected],
                        marker="o", label=f"seed {seed}")
        for index, scenario in enumerate(REQUIREMENT_SCENARIOS):
            ax = axes.flat[index + 2]
            ax.plot([r["epoch"] for r in selected],
                    [100 * r["selection"][scenario]["capture"] for r in selected],
                    marker="o", label=f"seed {seed}")
            ax.set_title(f"{scenario}: selection capture (%)")
        axes[1, 2].plot([r["epoch"] for r in trained],
                        [r["learning_rate"] for r in trained], label=f"seed {seed}")
    axes[0, 0].set_title("Training loss")
    axes[0, 1].set_title("Selection capture + 0.15 discovery")
    axes[1, 2].set_title("Learning rate")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend()
    figure.savefig(output_dir / "learning-curves.png", dpi=180)
    plt.close(figure)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-best", type=Path, required=True)
    parser.add_argument("--blend-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--mpc", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--inference-batch-size", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-initial-only", action="store_true")
    args = parser.parse_args(arguments)
    if min(args.workers, args.inference_batch_size) < 1:
        parser.error("positive worker and inference batch counts required")
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    configuration = json.loads((args.run_dir / "config.json").read_text())
    reporting, selection = configuration["reporting_seeds"], configuration["selection_seeds"]
    if set(reporting) & set(selection):
        raise ValueError("fresh reporting overlaps selection")
    policy = dict(configuration["policy"])
    policy["dwells"] = tuple(policy["dwells"])
    config = BeliefPolicyConfig(**policy)
    sources = {"planner-frozen": args.run_dir / "initial.pt"}
    if fingerprint(sources["planner-frozen"]) != configuration["initial_sha256"]:
        raise ValueError("initial checkpoint changed")
    if args.verify_initial_only:
        print(json.dumps(verify_planner(sources["planner-frozen"], config), indent=2))
        return 0
    training_seeds = args.seeds or configuration["arguments"]["seeds"]
    if (len(set(training_seeds)) != len(training_seeds)
            or not set(training_seeds) <= set(configuration["arguments"]["seeds"])):
        raise ValueError("reporting seeds must identify completed training seeds")
    selected = {}
    for seed in training_seeds:
        path = args.run_dir / f"seed-{seed}" / "best.pt"
        selected[str(seed)] = json.loads((path.parent / "frozen.json").read_text())
        if fingerprint(path) != selected[str(seed)]["checkpoint_sha256"]:
            raise ValueError("selected refinement checkpoint changed")
        progress = json.loads((path.parent / "progress.json").read_text())
        if progress[-1]["epoch"] != configuration["arguments"]["epochs"]:
            raise ValueError("training budget is incomplete")
        sources[f"refined-{seed}"] = path
    blend_selection = None
    if args.blend_dir:
        blend_selection = json.loads((args.blend_dir / "selection.json").read_text())
        blend_config = blend_selection["configuration"]
        if (blend_config["policy"] != configuration["policy"]
                or blend_config["selection_seeds"] != selection
                or blend_config["checkpoints"] != {str(p): fingerprint(p)
                                                    for p in sources.values()}):
            raise ValueError("blend selection provenance differs")
        path = args.blend_dir / "best.pt"
        if fingerprint(path) != blend_selection["checkpoint_sha256"]:
            raise ValueError("selected blend changed")
        sources["selected-blend"] = path
    checks = {name: verify_planner(path, config) for name, path in sources.items()}
    if args.verify_only:
        print(json.dumps(checks, indent=2))
        return 0
    frozen = {
        "training_configuration_sha256": fingerprint(args.run_dir / "config.json"),
        "source_sha256": fingerprint(__file__), "policy": configuration["policy"],
        "selection_seeds": selection, "reporting_seeds": reporting,
        "checkpoints": {name: fingerprint(path) for name, path in sources.items()},
        "previous_best_sha256": fingerprint(args.previous_best),
        "mpc_sha256": {str(p): fingerprint(p) for p in args.mpc},
        "selected_seed": max(training_seeds, key=lambda seed: selected[str(seed)]["score"]),
        "selection": selected, "verification": checks,
        "blend_selection": blend_selection,
        "best_policy": "selected-blend" if blend_selection else "selected_seed",
        "step_seconds": 0.001, "physical_horizon_ticks": 512,
    }
    started = time.perf_counter()
    with run_lock(args.output_dir):
        if (args.output_dir / "frozen.json").exists():
            raise ValueError("reporting directory already frozen")
        write_json(args.output_dir / "frozen.json", frozen)
        for name, path in sources.items():
            shutil.copyfile(path, args.output_dir / f"{name}.pt")
        winner = ("selected-blend" if blend_selection else f"refined-{frozen['selected_seed']}")
        shutil.copyfile(args.output_dir / f"{winner}.pt", args.output_dir / "best.pt")
        learning_curves(args.run_dir, args.output_dir, training_seeds)
        phase_config = BeliefPolicyConfig(
            dwells=config.dwells, revisit=256, probe=4, exploration=0.02)
        jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in reporting]
        results = _run_jobs(
            _control_job, ((s, seed, config, phase_config) for s, seed in jobs), args.workers)
        rows = {r["group"]: r for r in results}
        profiles = {}
        for name in sources:
            model, metadata = load_predictor(args.output_dir / f"{name}.pt")
            values, profiles[name] = batched_planned_results(
                model, config, jobs, args.inference_batch_size)
            profiles[name]["selected_epoch"] = metadata.get("epoch")
            for row in values:
                rows[row["group"]]["policies"][name] = row["value"]
            del model
            print(json.dumps({"policy": name, "episodes": len(jobs), **profiles[name]}), flush=True)
        model, actor, old_config, metadata = load_grouped(args.previous_best)
        for offset in range(0, len(jobs), args.inference_batch_size):
            chunk = jobs[offset:offset + args.inference_batch_size]
            worlds = [reporting_world(s, world_seed)[0] for s, world_seed in chunk]
            _, _, evaluations, _ = trajectories(model, actor, old_config, worlds)
            for (scenario, world_seed), evaluation in zip(chunk, evaluations, strict=True):
                rows[f"{scenario}:{world_seed}"]["policies"]["previous-best"] = {
                    "evaluation": evaluation,
                    "discovery_fraction": evaluation["discovery"]["emitter_discovery_ratio"]}
        profiles["previous-best"] = {"selected_iteration": metadata["iteration"]}
        del model, actor
        for index, path in enumerate(args.mpc):
            name = f"mpc-{index}"
            model, search_config = load_mpc(path)
            values, profiles[name] = batched_mpc(
                model, search_config, jobs, args.inference_batch_size)
            for row in values:
                rows[row["group"]]["policies"][name] = row["value"]
            del model
            print(json.dumps({"policy": name, "episodes": len(jobs)}), flush=True)
        sweeps = [PolicySpec(f"round-robin-{d}", partial(DwellSweepScheduler, dwell_steps=d),
                             f"dwell:{d}") for d in (10, 50)]
        sweep_report = benchmark_synthetic(
            sweeps, baseline="round-robin-50", seeds=reporting,
            scenarios=REQUIREMENT_SCENARIOS, workers=args.workers)
        for row in sweep_report["results"]:
            rows[row["group"]]["policies"].update(row["policies"])
        names = list(results[0]["policies"])
        for row in results:
            evaluations = [row["policies"][name]["evaluation"] for name in names]
            if len({v["counts"]["truth"] for v in evaluations}) != 1:
                raise ValueError("paired policies received different truth denominators")
            if any(abs(v["elapsed_seconds"] - 0.512) > 1e-9 for v in evaluations):
                raise ValueError("paired policies received different physical budgets")
        policies = [PolicySpec(name, DwellSweepScheduler, name) for name in names]
        references = [*(f"mpc-{i}" for i in range(len(args.mpc))), "round-robin-50",
                      "previous-best", "planner-frozen"]
        report = {
            "frozen": frozen, "results": results,
            "by_reference": {baseline: {
                scenario: summarize([r for r in results if r["scenario"] == scenario],
                                    policies, baseline) for scenario in REQUIREMENT_SCENARIOS
            } for baseline in references},
            "pooled": {scenario: pooled_metrics(
                [r for r in results if r["scenario"] == scenario], names, "mpc-0")
                for scenario in REQUIREMENT_SCENARIOS},
            "resources": {"elapsed_seconds": time.perf_counter() - started,
                          "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                          "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "profiles": profiles},
            "scope": "fresh synthetic development reporting; fixed menu and truth; test unused",
        }
        write_json(args.output_dir / "comparison.json", report)
        print(json.dumps({s: {p: m["interception_ratio"]["mean"] for p, m in v.items()}
                          for s, v in report["by_reference"]["mpc-0"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
