"""Explain missed-emitter acquisition after causal CUDA scheduling has finished."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..scenarios import REQUIREMENT_SCENARIOS
from ..timing_belief import BeliefPolicyConfig
from ..timing_ensemble import load_predictor
from ..timing_planner import CalibratedTimingPlannerPolicy
from .storage import fingerprint, run_lock, write_json
from .timing_report import BatchedEpisode, reporting_world


def describe(state, actions):
    """Truth joins are evaluator-only and run after the completed receiver episode."""
    episode = state.episode
    found = {emitter for record in episode.records for emitter in record.detected_emitters}
    rows = []
    for emitter in sorted({event.emitter_id for event in episode.transmissions}):
        events = [event for event in episode.transmissions if event.emitter_id == emitter]
        bands = sorted({event.band for event in events})
        records = [record for record in episode.records
                   if record.observation.listening and record.observation.band in bands]
        if emitter in found:
            continue
        selected = [action for action in actions if action["band"] in bands]
        rows.append({
            "emitter_id_evaluator_only": emitter, "truth_bands_evaluator_only": bands,
            "truth_pulses": len(events), "listening_ticks": len(records),
            "eligible_pulses": sum(
                episode.records[event.time_step].observation.listening
                and episode.records[event.time_step].observation.band == event.band
                for event in events
            ),
            "detectable_ticks": sum(emitter in record.detectable_emitters for record in records),
            "false_alarm_ticks": sum(record.false_alarm for record in records),
            "observed_hit_ticks": sum(record.observation.hit for record in records),
            "actions": selected,
        })
    return rows


@torch.inference_mode()
def diagnose(model, config, jobs, batch_size):
    host = torch.empty((batch_size, 8, 3, model.config.history), pin_memory=True)
    device = torch.empty_like(host, device="cuda")
    output = []
    for offset in range(0, len(jobs), batch_size):
        active = []
        for scenario, seed in jobs[offset:offset + batch_size]:
            world, world_seed = reporting_world(scenario, seed)
            scheduler = CalibratedTimingPlannerPolicy(model, config)
            active.append((scenario, seed, world_seed, BatchedEpisode(world, scheduler), []))
        while active:
            for index, (_, _, _, state, _) in enumerate(active):
                host.numpy()[index] = state.scheduler.history.encode()
            device[:len(active)].copy_(host[:len(active)], non_blocking=True)
            forecasts = model(device[:len(active)]).cpu().numpy()
            remaining = []
            for (scenario, seed, world_seed, state, actions), predicted in zip(
                active, forecasts, strict=True
            ):
                step, scheduler = state.episode.time_step, state.scheduler
                forced = scheduler.coverage_action(step) is not None
                action = scheduler.select(step, predicted)
                actions.append({"time_ms": step, "band": action.band,
                    "dwell_ms": action.dwell_steps, "coverage": forced,
                    "prior_hit_ticks": float(scheduler.history.hits[action.band]),
                    "forecast_rate": float(predicted[action.band].mean())})
                state.advance(action, scheduler.forecast(step, action))
                if state.episode.time_step < state.simulation.duration:
                    remaining.append((scenario, seed, world_seed, state, actions))
                else:
                    report = state.report()
                    output.append({"scenario": scenario, "seed": seed, "world_seed": world_seed,
                        "capture": report["interception_ratio"],
                        "discovery": report["discovery"]["emitter_discovery_ratio"],
                        "missed_emitters": describe(state, actions)})
            active = remaining
    return output


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args(arguments)
    if min(args.runs, args.batch_size) < 1:
        parser.error("positive episode count and batch size required")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("neural discovery diagnostics require CUDA")
    model, metadata = load_predictor(args.checkpoint)
    raw = dict(metadata["policy"])
    raw["dwells"] = tuple(raw["dwells"])
    config = BeliefPolicyConfig(**raw)
    digest = fingerprint(args.checkpoint)
    jobs = [(scenario, seed) for scenario in REQUIREMENT_SCENARIOS
            for seed in range(args.seed, args.seed + args.runs)]
    with run_lock(args.run_dir):
        if (args.run_dir / "diagnosis.json").exists():
            raise ValueError("discovery diagnosis already exists")
        rows = diagnose(model, config, jobs, args.batch_size)
        if fingerprint(args.checkpoint) != digest:
            raise ValueError("checkpoint changed during diagnosis")
        missed = [emitter for row in rows for emitter in row["missed_emitters"]]
        summary = {"worlds": len(rows), "missed_emitters": len(missed),
            "missed_without_any_eligible_pulse": sum(e["eligible_pulses"] == 0 for e in missed),
            "missed_with_false_alarms": sum(e["false_alarm_ticks"] > 0 for e in missed),
            "mean_listening_ms_on_missed_bands": (
                float(np.mean([e["listening_ticks"] for e in missed])) if missed else None),
            "mean_visits_on_missed_bands": (
                float(np.mean([len(e["actions"]) for e in missed])) if missed else None)}
        write_json(args.run_dir / "diagnosis.json", {
            "schema_version": 1, "checkpoint_sha256": digest, "summary": summary,
            "scope": "post-episode evaluator-only truth joins; development worlds",
            "results": rows,
        })
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
