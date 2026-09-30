"""Frozen MOPD students against their old head, native MPC and round-robin."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch

from ..grouped_policy import load_grouped
from ..mopd_policy import load_mopd
from ..policy_benchmark import PolicySpec, benchmark_synthetic, summarize
from ..scenarios import REQUIREMENT_SCENARIOS
from ..schedulers import DwellSweepScheduler
from .grouped_report import pooled_metrics
from .grouped_study import trajectories
from .storage import fingerprint, run_lock, write_json
from .timing_mpc_compare import batched_mpc, load_mpc
from .timing_report import reporting_world


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mpc", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--inference-batch-size", type=int, default=20)
    args = parser.parse_args(arguments)
    if not torch.cuda.is_available():
        raise RuntimeError("MOPD and MPC neural reporting require CUDA")
    if min(args.workers, args.inference_batch_size) < 1:
        parser.error("positive workers and inference batch size required")
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    configuration = json.loads((args.run_dir / "config.json").read_text())
    seeds = configuration["reporting_seeds"]
    if set(seeds) & set(configuration["selection_seeds"]):
        raise ValueError("reporting overlaps selection")
    initial = Path(configuration["arguments"]["initial_actor"])
    if fingerprint(initial) != configuration["initial_actor_sha256"]:
        raise ValueError("previous-best actor changed")
    students = [
        args.run_dir / f"seed-{seed}" / "best.pt" for seed in configuration["arguments"]["seeds"]
    ]
    frozen = {
        "training_configuration_sha256": fingerprint(args.run_dir / "config.json"),
        "sources": {
            str(path): fingerprint(path)
            for path in (
                Path(__file__),
                Path(__file__).parent / "mopd_study.py",
                Path(__file__).parent.parent / "mopd_policy.py",
            )
        },
        "initial_actor_sha256": fingerprint(initial),
        "forecaster_sha256": fingerprint(args.run_dir / "forecaster.pt"),
        "students": {},
        "mpc_sha256": {str(path): fingerprint(path) for path in args.mpc},
        "reporting_seeds": seeds,
        "physical_horizon_ticks": 512,
        "step_seconds": 0.001,
        "policy": configuration["policy"],
    }
    for path in students:
        selected = json.loads((path.parent / "frozen.json").read_text())
        if (
            fingerprint(path) != selected["checkpoint_sha256"]
            or frozen["forecaster_sha256"] != selected["forecaster_sha256"]
        ):
            raise ValueError("frozen student or forecaster changed")
        for teacher in selected["teachers"].values():
            if fingerprint(teacher["path"]) != teacher["sha256"]:
                raise ValueError("specialist teacher changed after integration")
        frozen["students"][str(path)] = selected
    began = time.perf_counter()
    with run_lock(args.output_dir):
        if (args.output_dir / "frozen.json").exists():
            raise ValueError("use a fresh reporting directory")
        write_json(args.output_dir / "frozen.json", frozen)
        sweeps = [
            PolicySpec(
                f"round-robin-{d}", partial(DwellSweepScheduler, dwell_steps=d), f"dwell:{d}"
            )
            for d in (10, 50)
        ]
        comparison = benchmark_synthetic(
            sweeps,
            baseline="round-robin-50",
            seeds=seeds,
            scenarios=REQUIREMENT_SCENARIOS,
            workers=args.workers,
        )
        results = comparison["results"]
        rows = {row["group"]: row for row in results}
        jobs = [(scenario, seed) for scenario in REQUIREMENT_SCENARIOS for seed in seeds]
        profiles = {}
        checkpoints = [("previous-best-fixed-head", initial, load_grouped)]
        checkpoints += [
            (f"mopd-{seed}", path, load_mopd)
            for seed, path in zip(configuration["arguments"]["seeds"], students, strict=True)
        ]
        for name, path, loader in checkpoints:
            model, actor, policy, metadata = loader(path)
            for start in range(0, len(jobs), args.inference_batch_size):
                chunk = jobs[start : start + args.inference_batch_size]
                worlds = [reporting_world(scenario, seed)[0] for scenario, seed in chunk]
                _, _, evaluations, _ = trajectories(model, actor, policy, worlds)
                for (scenario, seed), evaluation in zip(chunk, evaluations, strict=True):
                    rows[f"{scenario}:{seed}"]["policies"][name] = {
                        "evaluation": evaluation,
                        "discovery_fraction": evaluation["discovery"]["emitter_discovery_ratio"],
                    }
            profiles[name] = {
                "selected_iteration": metadata["iteration"],
                "checkpoint_sha256": fingerprint(path),
            }
            del model, actor
            print(json.dumps({"policy": name, "episodes": len(jobs)}), flush=True)
        for index, path in enumerate(args.mpc):
            name = f"mpc-{index}"
            model, config = load_mpc(path)
            values, profiles[name] = batched_mpc(model, config, jobs, args.inference_batch_size)
            for value in values:
                rows[value["group"]]["policies"][name] = value["value"]
            del model
            print(json.dumps({"policy": name, "episodes": len(jobs)}), flush=True)
        names = list(results[0]["policies"])
        for row in results:
            metrics = [row["policies"][name]["evaluation"] for name in names]
            if len({metric["counts"]["truth"] for metric in metrics}) != 1:
                raise ValueError("paired policies received different truth denominators")
            if any(abs(metric["elapsed_seconds"] - 0.512) > 1e-9 for metric in metrics):
                raise ValueError("paired policies received different physical budgets")
        policies = [PolicySpec(name, DwellSweepScheduler, name) for name in names]
        references = [
            "previous-best-fixed-head",
            "round-robin-50",
            *(f"mpc-{index}" for index in range(len(args.mpc))),
        ]
        report = {
            "frozen": frozen,
            "results": results,
            "profiles": profiles,
            "by_reference": {
                reference: {
                    scenario: summarize(
                        [row for row in results if row["scenario"] == scenario], policies, reference
                    )
                    for scenario in REQUIREMENT_SCENARIOS
                }
                for reference in references
            },
            "pooled": {
                reference: {
                    scenario: pooled_metrics(
                        [row for row in results if row["scenario"] == scenario], names, reference
                    )
                    for scenario in REQUIREMENT_SCENARIOS
                }
                for reference in references
            },
            "reporting_seconds": time.perf_counter() - began,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "scope": (
                "Two student training seeds on fresh preregistered validation worlds; "
                "no teacher dispatch or scenario identity supplied to runtime actors."
            ),
        }
        write_json(args.output_dir / "comparison.json", report)
        print(
            json.dumps(
                {
                    scenario: {
                        name: value["interception_ratio"]["mean"] for name, value in metrics.items()
                    }
                    for scenario, metrics in report["by_reference"][
                        "previous-best-fixed-head"
                    ].items()
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
