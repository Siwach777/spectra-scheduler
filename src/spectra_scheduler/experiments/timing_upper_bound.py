"""Evaluation-only native-action capture bounds on selection worlds.

Every oracle in this module sees future truth. None is a deployable scheduler,
training label generator, or ML result. Realized-capture bounds additionally see
counterfactual future detector outcomes and are labeled separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..policy_benchmark import _run_jobs
from ..scenarios import (
    REQUIREMENT_SCENARIO_VERSION,
    REQUIREMENT_SCENARIOS,
    build_requirement_scenario,
)
from ..simulation import SimulationEpisode, SyntheticAction, configure_scheduler
from ..timing_belief import (
    BeliefPolicyConfig,
    TimingBeliefPolicy,
    TimingHistory,
    validated_retune_table,
)
from ..timing_planner import ForecastPlannerWorkspace, first_actions
from .storage import fingerprint, write_json


def optimal_schedule(rewards, retune, dwells, previous=None):
    """Exact finite-horizon DP over (physical tick, previously tuned band).

    The extra previous-band state represents initial reception without retuning.
    A macro action listens for its dwell after public retuning, clipped at horizon.
    """
    bands, horizon = rewards.shape
    cumulative = np.pad(rewards.cumsum(1), ((0, 0), (1, 0)))
    delays = np.vstack((retune, np.zeros((1, bands), np.int64)))
    band_index = np.arange(bands)[None, :, None]
    values = np.zeros((horizon + 1, bands + 1), np.float64)
    actions = np.empty((horizon, bands + 1, 2), np.int16)
    dwells = np.asarray(dwells)
    for tick in range(horizon - 1, -1, -1):
        starts = np.minimum(tick + delays[:, :, None], horizon)
        ends = np.minimum(starts + dwells[None, None], horizon)
        capture = cumulative[band_index, ends] - cumulative[band_index, starts]
        score = capture + values[ends, band_index]
        choice = score.reshape(bands + 1, -1).argmax(1)
        actions[tick, :, 0] = choice // len(dwells)
        actions[tick, :, 1] = dwells[choice % len(dwells)]
        values[tick] = score.reshape(bands + 1, -1).max(1)
    initial = bands if previous is None else previous
    tick, previous, schedule = 0, initial, []
    while tick < horizon:
        band, dwell = map(int, actions[tick, previous])
        schedule.append((tick, band, dwell))
        tick = min(horizon, tick + int(delays[previous, band]) + dwell)
        previous = band
    return float(values[0, initial]), schedule


class _EvaluationTruthScorer(TimingBeliefPolicy):
    """Oracle scorer solely inside this offline diagnostic; never a runtime input."""

    def __init__(self, rewards, config):
        self.rewards = np.pad(rewards, ((0, 0), (0, 80)))
        self.config = config

    def set_retune_table(self, table):
        self.retune = validated_retune_table(table, 80, self.config.dwells)

    def reset(self, bands):
        self.history = TimingHistory(bands, 1)
        self.bands = bands
        self.decisions = self.probes = 0

    def choose_action(self, tick):
        return self.select(tick, self.rewards[:, tick : tick + 80])


def replay_schedule(world, schedule):
    episode = SimulationEpisode(world)
    for tick, band, dwell in schedule:
        if tick != episode.time_step:
            raise ValueError("DP plan and shared receiver clock disagree")
        episode.step_action(SyntheticAction(band, dwell))
    captured = sum(len(record.detected_emitters) for record in episode.records)
    found = {emitter for record in episode.records for emitter in record.detected_emitters}
    return {
        "captured": captured,
        "interception_ratio": captured / len(episode.transmissions),
        "discovery_fraction": len(found) / len(world.emitters),
        "decisions": len(schedule),
    }


def replay_scorer(world, rewards, config):
    scheduler = _EvaluationTruthScorer(rewards, config)
    configure_scheduler(world, scheduler)
    episode = SimulationEpisode(world)
    schedule = []
    while episode.time_step < world.duration:
        tick = episode.time_step
        action = scheduler.choose_action(tick)
        schedule.append((tick, action.band, action.dwell_steps))
        for observation in episode.step_action(action):
            scheduler.observe(observation)
    result = replay_schedule(world, schedule)
    result["coverage_probes"] = scheduler.probes
    return result


def replay_receding_dp(world, rewards, retune, config, *, coverage=False, horizon=80):
    """Oracle finite-horizon planning using only the next forecast-sized window."""
    episode = SimulationEpisode(world)
    previous, schedule = None, []
    last_listen = np.full(world.num_bands, -1, np.int64)
    probes = 0
    padded = np.pad(rewards, ((0, 0), (0, horizon)))
    workspace = ForecastPlannerWorkspace(1, world.num_bands, horizon, config.dwells)
    while episode.time_step < world.duration:
        tick = episode.time_step
        age = np.where(last_listen >= 0, tick - last_listen, tick + config.revisit)
        if coverage and age.max() >= config.revisit:
            band = int(age.argmax())
            dwell = min(config.dwells, key=lambda value: abs(value - config.probe))
            probes += 1
        else:
            band_values, dwell_values = first_actions(
                padded[None, :, tick : tick + horizon],
                [-1 if previous is None else previous],
                retune[None],
                [world.duration - tick],
                config.dwells,
                workspace,
            )
            band, dwell = int(band_values[0]), int(dwell_values[0])
        schedule.append((tick, band, dwell))
        for observation in episode.step_action(SyntheticAction(band, dwell)):
            if observation.listening:
                last_listen[band] = observation.time_step
        previous = band
    result = replay_schedule(world, schedule)
    result["coverage_probes"] = probes
    return result


def _world_job(job):
    scenario, seed, config = job
    key = f"synthetic-v{REQUIREMENT_SCENARIO_VERSION}:val:{scenario}:{seed}".encode()
    world_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    world = build_requirement_scenario(scenario, world_seed)
    episode = SimulationEpisode(world)
    arrivals = np.zeros((world.num_bands, world.duration), np.float64)
    realized = np.zeros_like(arrivals)
    for (tick, band), events in episode.events.items():
        arrivals[band, tick] = len(events)
        realized[band, tick] = len(world.receiver.listen(tick, band, events).detected_emitters)
    retune = np.asarray(
        [
            [world.receiver.retune_duration(a, b) for b in range(world.num_bands)]
            for a in range(world.num_bands)
        ],
        np.int64,
    )
    expected = arrivals * world.receiver.detection_probability
    realized_value, realized_plan = optimal_schedule(realized, retune, config.dwells)
    expected_value, expected_plan = optimal_schedule(expected, retune, config.dwells)
    realized_result = replay_schedule(world, realized_plan)
    if not np.isclose(realized_value, realized_result["captured"]):
        raise ValueError("DP capture bound differs from the shared receiver replay")
    expected_result = replay_schedule(world, expected_plan)
    expected_result["expected_capture_objective"] = expected_value
    expected_result["expected_interception_ratio"] = expected_value / len(episode.transmissions)
    no_coverage = replace(
        config, revisit=world.duration + 1, exploration=0, retune_cost=0, switch_margin=0
    )
    # Unknown bands still force initial probes in the shared selector. Explicitly
    # mark all as initially seen for a pure perfect-forecast scoring diagnostic.
    scorer = _EvaluationTruthScorer(expected, no_coverage)
    configure_scheduler(world, scorer)
    scorer.history.last_listen[:] = 0
    scorer_episode = SimulationEpisode(world)
    score_plan = []
    while scorer_episode.time_step < world.duration:
        tick = scorer_episode.time_step
        action = scorer.choose_action(tick)
        score_plan.append((tick, action.band, action.dwell_steps))
        for observation in scorer_episode.step_action(action):
            scorer.observe(observation)
    perfect_myopic = replay_schedule(world, score_plan)
    configured_myopic = replay_scorer(world, expected, config)
    receding_dp = replay_receding_dp(world, expected, retune, config)
    receding_dp_coverage = replay_receding_dp(world, expected, retune, config, coverage=True)
    return {
        "scenario": scenario,
        "seed": seed,
        "world_seed": world_seed,
        "truth_arrivals": len(episode.transmissions),
        "realized_clairvoyant_dp": realized_result,
        "expected_clairvoyant_dp": expected_result,
        "perfect_forecast_myopic": perfect_myopic,
        "perfect_forecast_current_coverage": configured_myopic,
        "perfect_forecast_receding_dp80": receding_dp,
        "perfect_forecast_receding_dp80_coverage": receding_dp_coverage,
        "instant_tuning_realized_ratio": realized.max(0).sum() / len(episode.transmissions),
        "retune_max_ticks": int(retune.max()),
    }


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(arguments)
    config = BeliefPolicyConfig(dwells=(1, 10, 50), revisit=256, probe=10, exploration=0.02)
    started = time.perf_counter()
    rows = _run_jobs(
        _world_job,
        [
            (scenario, seed, config)
            for scenario in REQUIREMENT_SCENARIOS
            for seed in range(2000, 2000 + args.runs)
        ],
        args.workers,
    )
    names = (
        "realized_clairvoyant_dp",
        "expected_clairvoyant_dp",
        "perfect_forecast_myopic",
        "perfect_forecast_current_coverage",
        "perfect_forecast_receding_dp80",
        "perfect_forecast_receding_dp80_coverage",
    )
    summary = {
        s: {
            name: {
                metric: float(np.mean([r[name][metric] for r in rows if r["scenario"] == s]))
                for metric in ("interception_ratio", "discovery_fraction", "decisions")
            }
            for name in names
        }
        for s in REQUIREMENT_SCENARIOS
    }
    package = Path(__file__).resolve().parent.parent
    report = {
        "scope": (
            "evaluation-only clairvoyant bounds; selection worlds only; no runtime truth input"
        ),
        "policy_config": asdict(config),
        "selection_seeds": list(range(2000, 2000 + args.runs)),
        "sources": {
            name: fingerprint(package / name)
            for name in (
                "experiments/timing_upper_bound.py",
                "timing_belief.py",
                "simulation.py",
                "receiver.py",
                "scenarios.py",
                "emitters.py",
            )
        },
        "expected_detector_assumption": (
            "all required-scenario signals are above sensitivity; expected counts use public pdet"
        ),
        "realized_bound_information": (
            "future arrivals and deterministic counterfactual detector outcomes"
        ),
        "expected_plan_information": (
            "future arrivals and public detector probability; no realized detector outcomes"
        ),
        "diagnostic_seconds": time.perf_counter() - started,
        "summary": summary,
        "results": rows,
    }
    write_json(args.output, report)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
