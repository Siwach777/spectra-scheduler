"""Pair frozen timing results with native MPC search and round-robin dwells."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..mpc.checkpoints import implementation_hashes, load_model
from ..mpc.config import STEP_FEATURE_DIM, saved_config
from ..mpc.coverage import coverage_probe_action
from ..mpc.macro import MacroObservationEncoder
from ..mpc.scheduler import NeuralMPCScheduler
from ..mpc.search import search_batch
from ..policy_benchmark import PolicySpec, benchmark_synthetic, summarize
from ..schedulers import DwellSweepScheduler
from ..simulation import SyntheticAction
from ..synthetic_evaluation import evaluate_scheduler
from .storage import fingerprint, write_json
from .timing_report import BatchedEpisode, reporting_world, reward


class MPCObserver:
    """Same causal macro inputs as the native MPC scheduler, without a CUDA copy per tick."""

    def __init__(self, step_dim):
        self.step_dim = step_dim

    def set_retune_table(self, table):
        self.retune = np.asarray(table, dtype=np.int64)

    def set_episode_horizon(self, horizon):
        self.horizon = horizon

    def reset(self, bands):
        self.encoder = MacroObservationEncoder(
            bands, include_elapsed=self.step_dim > STEP_FEATURE_DIM
        )

    def observe(self, observation):
        self.encoder.observe(observation)


class RootGenerators:
    """Keep native seed-zero Gumbel RNG state independent for every world."""

    def __init__(self, generators):
        self.generators, self.index = generators, 0

    def gumbel(self, size):
        output = self.generators[self.index].gumbel(size=size)
        self.index += 1
        return output


def load_mpc(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = load_model(path, torch.device("cuda"))
    settings = dict(payload["metadata"]["config"])
    settings.update(device="cuda")
    config = saved_config(settings)
    if not model.physical_contract:
        raise ValueError("comparison requires physical MPC checkpoints")
    return model, config


@torch.inference_mode()
def batched_mpc(model, config, jobs, batch_size):
    output = []
    decisions, search_seconds = 0, 0.0
    for start in range(0, len(jobs), batch_size):
        current_jobs = jobs[start : start + batch_size]
        states = []
        for scenario, seed in current_jobs:
            simulation, world_seed = reporting_world(scenario, seed)
            states.append(
                (
                    scenario,
                    seed,
                    world_seed,
                    BatchedEpisode(simulation, MPCObserver(model.step_dim)),
                )
            )
        hidden = model.representation.initial_state(len(states))
        generators = [np.random.default_rng(0) for _ in states]
        host = torch.empty((len(states), 1, model.step_dim), pin_memory=True)
        features = host.numpy()[:, 0]
        staging = torch.empty_like(host, device="cuda")
        while True:
            active = [
                i
                for i, (_, _, _, state) in enumerate(states)
                if state.episode.time_step < state.simulation.duration
            ]
            if not active:
                break
            remaining = np.array(
                [states[i][3].simulation.duration - states[i][3].episode.time_step for i in active]
            )
            previous = np.array(
                [
                    -1
                    if states[i][3].episode.previous_band is None
                    else states[i][3].episode.previous_band
                    for i in active
                ]
            )
            retune = [states[i][3].scheduler.retune for i in active]
            forced = [
                coverage_probe_action(
                    states[i][3].scheduler.encoder.encoder.last_visit_step,
                    states[i][3].episode.time_step,
                    int(previous[row]),
                    retune[row],
                    model.dwell_steps,
                    int(remaining[row]),
                    config.coverage_probe_limit,
                )
                for row, i in enumerate(active)
            ]
            free = [row for row, action in enumerate(forced) if action is None]
            actions = np.array([action if action is not None else -1 for action in forced])
            active_device = torch.as_tensor(active, device="cuda")
            if free:
                started = time.perf_counter()
                ids = torch.as_tensor([active[row] for row in free], device="cuda")
                policies, _, planned = search_batch(
                    model,
                    hidden[0].index_select(0, ids),
                    remaining[free],
                    config,
                    RootGenerators([generators[active[row]] for row in free]),
                    current_bands=previous[free],
                    retune_tables=[retune[row] for row in free],
                    return_actions=True,
                )
                actions[free] = planned if config.search_method == "gumbel" else policies.argmax(-1)
                search_seconds += time.perf_counter() - started
                decisions += len(free)
            for row, i in enumerate(active):
                state = states[i][3]
                band, dwell = divmod(int(actions[row]), len(model.dwell_steps))
                state.advance(SyntheticAction(band, int(model.dwell_steps[dwell])), None)
                features[row] = state.scheduler.encoder.finish()
            staging[: len(active)].copy_(host[: len(active)], non_blocking=True)
            _, next_hidden = model.representation(
                staging[: len(active)], hidden.index_select(1, active_device)
            )
            hidden.index_copy_(1, active_device, next_hidden)
        for scenario, seed, world_seed, state in states:
            evaluation = state.report()
            output.append(
                {
                    "group": f"{scenario}:{seed}",
                    "scenario": scenario,
                    "seed": seed,
                    "world_seed": world_seed,
                    "value": {
                        "evaluation": evaluation,
                        "discovery_fraction": evaluation["discovery"]["emitter_discovery_ratio"],
                    },
                }
            )
    return output, {
        "batch_size": batch_size,
        "search_decisions": decisions,
        "search_seconds": search_seconds,
        "amortized_search_seconds_per_decision": search_seconds / max(decisions, 1),
    }


def verify_native(path, batch_size):
    model, config = load_mpc(path)
    jobs = [(s, 2000) for s in ("frequency-agile", "spatial-scan", "periodic-scan")]
    batched, _ = batched_mpc(model, config, jobs, batch_size)
    values = {r["group"]: r["value"]["evaluation"] for r in batched}
    for scenario, seed in jobs:
        simulation, _ = reporting_world(scenario, seed)
        native = NeuralMPCScheduler(
            model=model,
            device="cuda",
            num_simulations=config.simulations,
            gamma=config.gamma,
            depth=config.depth,
            normalize_search=config.normalize_search,
            coverage_probe_limit=config.coverage_probe_limit,
            search_method=config.search_method,
            gumbel_candidates=config.gumbel_candidates,
            gumbel_q_scale=config.gumbel_q_scale,
        )
        result = evaluate_scheduler(
            simulation,
            native,
            step_seconds=0.001,
            reward=reward,
            reward_description="observed_hit - 0.05 * retuning",
        )
        for key in ("counts", "discovery", "reward_sum", "probability_of_false_alarm"):
            a, b = result[key], values[f"{scenario}:{seed}"][key]
            equal = np.isclose(a, b, rtol=1e-7, atol=1e-7) if isinstance(a, float) else a == b
            if not equal:
                raise ValueError(f"batched MPC differs from native {key} on {scenario}")
    return {"checkpoint": str(path), "selection_worlds": len(jobs), "native_parity": True}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mpc", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(arguments)
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    if args.verify_only:
        for path in args.mpc:
            print(json.dumps(verify_native(path, args.batch_size)), flush=True)
        return 0
    if args.output.exists():
        raise ValueError("refuse to overwrite completed comparison")
    reference = json.loads(args.reference.read_text())
    frozen = {
        "reference_sha256": fingerprint(args.reference),
        "mpc_sha256": {str(p): fingerprint(p) for p in args.mpc},
        "source_sha256": fingerprint(__file__),
        "mpc_sources": implementation_hashes(),
        "protocol": "native saved MPC checkpoints and settings; same worlds and receiver draws",
    }
    write_json(args.output.with_suffix(".frozen.json"), frozen)
    jobs = [(r["scenario"], r["seed"]) for r in reference["results"]]
    rows = {r["group"]: r for r in reference["results"]}
    profiling, settings = {}, {}
    started = time.perf_counter()
    for index, path in enumerate(args.mpc):
        name = f"mpc-{index}"
        model, config = load_mpc(path)
        values, profiling[name] = batched_mpc(model, config, jobs, args.batch_size)
        settings[name] = {
            "checkpoint": str(path),
            "gamma": config.gamma,
            "simulations": config.simulations,
            "depth": config.depth,
            "search_method": config.search_method,
            "coverage_probe_limit": config.coverage_probe_limit,
            "dwell_steps": list(model.dwell_steps),
        }
        for value in values:
            rows[value["group"]]["policies"][name] = value["value"]
        del model
        print(
            json.dumps({"policy": name, "episodes": len(values), "profile": profiling[name]}),
            flush=True,
        )
    from functools import partial

    extra = [
        PolicySpec(f"round-robin-{d}", partial(DwellSweepScheduler, dwell_steps=d), f"dwell:{d}")
        for d in (10, 50)
    ]
    sweeps = benchmark_synthetic(
        extra,
        baseline="round-robin-50",
        split="val",
        seeds=tuple(reference["seeds"]),
        scenarios=tuple(reference["scenarios"]),
        workers=20,
    )
    for row in sweeps["results"]:
        rows[row["group"]]["policies"].update(row["policies"])
    policies = [
        PolicySpec(name, DwellSweepScheduler, name)
        for name in rows[jobs[0][0] + ":" + str(jobs[0][1])]["policies"]
    ]
    results = list(rows.values())
    report = {
        "frozen": frozen,
        "settings": settings,
        "scenarios": reference["scenarios"],
        "seeds": reference["seeds"],
        "results": results,
        "by_reference": {
            baseline: {
                s: summarize([r for r in results if r["scenario"] == s], policies, baseline)
                for s in reference["scenarios"]
            }
            for baseline in [*(f"mpc-{i}" for i in range(len(args.mpc))), "round-robin-50"]
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "profiling": profiling,
        },
        "scope": "same frozen development worlds; reporting results do not select model settings",
    }
    for path in args.mpc:
        if fingerprint(path) != frozen["mpc_sha256"][str(path)]:
            raise ValueError("MPC checkpoint changed")
    write_json(args.output, report)
    print(
        json.dumps(
            {
                s: {name: metrics["interception_ratio"]["mean"] for name, metrics in values.items()}
                for s, values in report["by_reference"]["mpc-0"].items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
