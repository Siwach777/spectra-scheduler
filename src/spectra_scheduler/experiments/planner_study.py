"""CUDA selection of a causal receding planner using learned timing forecasts."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ..joint_belief import load_joint
from ..scenarios import REQUIREMENT_SCENARIOS
from ..simulation import SyntheticAction
from ..timing_belief import BeliefPolicyConfig, load_belief
from ..timing_planner import (
    CalibratedTimingPlannerPolicy,
    ForecastPlannerWorkspace,
    first_actions,
)
from .storage import fingerprint, run_lock, write_json
from .timing_report import BatchedEpisode, reporting_world


@torch.inference_mode()
def batched_planned_results(model, config, jobs, batch_size, *, coverage=True,
                            world_factory=reporting_world,
                            scheduler_factory=CalibratedTimingPlannerPolicy):
    if not jobs or batch_size < 1:
        raise ValueError("nonempty jobs and a positive inference batch are required")
    first_world, first_seed = world_factory(*jobs[0])
    bands = first_world.num_bands
    shape = (batch_size, bands, 3, model.config.history)
    host = torch.empty(shape, pin_memory=True)
    host_array = host.numpy()
    device = torch.empty(shape, device="cuda")
    output = []
    decisions, neural_seconds, planning_seconds = 0, 0.0, 0.0
    started = time.perf_counter()
    for offset in range(0, len(jobs), batch_size):
        active = []
        for index, (scenario, seed) in enumerate(jobs[offset : offset + batch_size]):
            world, world_seed = ((first_world, first_seed) if offset + index == 0
                                 else world_factory(scenario, seed))
            if world.num_bands != bands:
                raise ValueError("batched timing worlds must share their band count")
            scheduler = scheduler_factory(model, config, coverage=coverage)
            active.append((scenario, seed, world_seed, BatchedEpisode(world, scheduler)))
        workspace = None
        while active:
            before = time.perf_counter()
            for i, (_, _, _, state) in enumerate(active):
                host_array[i] = state.scheduler.history.encode()
            device[: len(active)].copy_(host[: len(active)], non_blocking=True)
            prediction = model(device[: len(active)]).cpu().numpy()
            neural_seconds += time.perf_counter() - before
            before = time.perf_counter()
            current = np.array([state.scheduler.history.current_band for _, _, _, state in active])
            retune = np.stack([state.scheduler.retune for _, _, _, state in active])
            remaining = np.array(
                [state.simulation.duration - state.episode.time_step for _, _, _, state in active]
            )
            if workspace is None or workspace.batch_size != len(active):
                workspace = ForecastPlannerWorkspace(
                    len(active), prediction.shape[1], prediction.shape[2], config.dwells
                )
            chosen_bands, dwells = first_actions(
                prediction, current, retune, remaining, dwells=config.dwells, workspace=workspace
            )
            for i, ((_, _, _, state), predicted) in enumerate(zip(active, prediction, strict=True)):
                step = state.episode.time_step
                if coverage:
                    forced = state.scheduler.coverage_action(step)
                    if forced is not None:
                        chosen_bands[i], dwells[i] = forced.band, forced.dwell_steps
                action = SyntheticAction(int(chosen_bands[i]), int(dwells[i]))
                state.scheduler.accept_action(step, predicted, action)
                state.advance(action, state.scheduler.forecast(step, action))
            planning_seconds += time.perf_counter() - before
            decisions += len(active)
            remaining_states = []
            for scenario, seed, world_seed, state in active:
                if state.episode.time_step < state.simulation.duration:
                    remaining_states.append((scenario, seed, world_seed, state))
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
            active = remaining_states
    return output, {
        "batch_size": batch_size,
        "decision_count": decisions,
        "neural_seconds": neural_seconds,
        "planning_and_simulation_seconds": planning_seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=("timing", "joint"), default="timing")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--revisits", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--probes", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--allow-no-coverage", action="store_true")
    args = parser.parse_args(arguments)
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    model, _ = (load_joint if args.kind == "joint" else load_belief)(args.checkpoint)
    base = BeliefPolicyConfig(
        dwells=(1, 10, 50),
        revisit=256,
        probe=10,
        exploration=0.0,
        retune_cost=0.0,
        switch_margin=0.0,
    )
    jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in range(2000, 2000 + args.runs)]
    candidates = [
        (replace(base, revisit=r, probe=p), True) for r in args.revisits for p in args.probes
    ]
    if args.allow_no_coverage:
        candidates.append((base, False))
    with run_lock(args.run_dir):
        if (args.run_dir / "selection.json").exists():
            raise ValueError("planner selection already exists")
        semantic = {
            "checkpoint_sha256": fingerprint(args.checkpoint),
            "kind": args.kind,
            "seeds": list(range(2000, 2000 + args.runs)),
            "objective": "equal-scenario capture + 0.15 * discovery",
            "candidates": [{"policy": asdict(c), "coverage": cov} for c, cov in candidates],
            "sources": {
                "planner": fingerprint(Path(__file__).parent.parent / "timing_planner.py"),
                "study": fingerprint(__file__),
            },
        }
        write_json(args.run_dir / "config.json", semantic)
        rows = []
        for config, coverage in candidates:
            values, profile = batched_planned_results(
                model, config, jobs, args.workers, coverage=coverage
            )
            means = {}
            for scenario in REQUIREMENT_SCENARIOS:
                group = [r["value"] for r in values if r["scenario"] == scenario]
                capture = [
                    r["evaluation"]["interception_ratio"]
                    for r in group
                    if r["evaluation"]["interception_ratio"] is not None
                ]
                means[scenario] = {
                    "capture": float(np.mean(capture)),
                    "discovery": float(np.mean([r["discovery_fraction"] for r in group])),
                }
            score = float(np.mean([r["capture"] + 0.15 * r["discovery"] for r in means.values()]))
            row = {
                "policy": asdict(config),
                "coverage": coverage,
                "means": means,
                "score": score,
                "profile": profile,
            }
            rows.append(row)
            write_json(args.run_dir / "progress.json", rows)
            print(json.dumps(row), flush=True)
        best = max(rows, key=lambda r: r["score"])
        write_json(args.run_dir / "selection.json", {**semantic, "selected": best, "results": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
