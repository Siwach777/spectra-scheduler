"""Frozen cross-distribution model comparison on MPC's physical-world suites."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from ..grouped_policy import GroupedTimingPolicy, load_grouped
from ..mpc.config import REWARD
from ..mpc.scheduler import NeuralMPCScheduler
from ..recurrent_cli import load_checked
from ..recurrent_env import RecurrentScheduler
from ..rl_benchmark import benchmark
from ..schedulers import DwellSweepScheduler
from ..timing_belief import BeliefPolicyConfig, TimingBeliefPolicy, load_belief
from ..timing_ensemble import load_predictor
from ..timing_planner import TimingPlannerPolicy
from .mpc_assess import _bootstrap_intervals
from .storage import fingerprint, write_json
from .timing_mpc_compare import load_mpc


class TickMacroAdapter:
    """Preserve listening dwell through retuning in the existing tick benchmark."""

    def __init__(self, policy):
        self.policy = policy

    def set_retune_table(self, table):
        self.policy.set_retune_table(table)

    def set_episode_horizon(self, horizon):
        self.policy.set_episode_horizon(horizon)

    def reset(self, bands):
        self.policy.reset(bands)
        self.remaining = 0

    def choose_band(self, step):
        if not self.remaining:
            action = self.policy.choose_action(step)
            self.band, self.remaining = action.band, action.dwell_steps
        return self.band

    def observe(self, observation):
        self.policy.observe(observation)
        if observation.listening:
            self.remaining -= 1


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-run", type=Path, required=True)
    parser.add_argument("--mpc", type=Path, action="append", required=True)
    parser.add_argument("--ppo-run", type=Path)
    parser.add_argument("--refine-run", type=Path)
    parser.add_argument("--blend-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=40000)
    args = parser.parse_args(arguments)
    if args.blend_dir and not args.refine_run:
        parser.error("blend requires its refinement run for the public policy configuration")
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    if args.output.exists():
        raise ValueError("refuse to overwrite finished comparison")
    semantic = json.loads((args.grouped_run / "config.json").read_text())
    models, factories, digests = [], {}, {}
    policy_settings = dict(semantic["policy"])
    policy_settings["dwells"] = tuple(policy_settings["dwells"])
    config = BeliefPolicyConfig(**policy_settings)
    model, _ = load_belief(args.grouped_run / "forecaster.pt")
    models.append(model)
    factories["timing-before-rl"] = lambda b=model: TickMacroAdapter(TimingBeliefPolicy(b, config))
    digests["forecaster"] = fingerprint(args.grouped_run / "forecaster.pt")
    for seed in semantic["arguments"]["seeds"]:
        path = args.grouped_run / f"seed-{seed}" / "best.pt"
        frozen = json.loads((path.parent / "frozen.json").read_text())
        if fingerprint(path) != frozen["checkpoint_sha256"]:
            raise ValueError("selected grouped checkpoint changed")
        belief, actor, policy, _ = load_grouped(path)
        models.extend((belief, actor))
        factories[f"grouped-{seed}"] = lambda b=belief, a=actor, p=policy: TickMacroAdapter(
            GroupedTimingPolicy(b, a, p)
        )
        digests[f"grouped-{seed}"] = fingerprint(path)
    if args.refine_run:
        refinement = json.loads((args.refine_run / "config.json").read_text())
        settings = dict(refinement["policy"])
        settings["dwells"] = tuple(settings["dwells"])
        planned_config = BeliefPolicyConfig(**settings)
        paths = {"planner-frozen": args.refine_run / "initial.pt"}
        if fingerprint(paths["planner-frozen"]) != refinement["initial_sha256"]:
            raise ValueError("initial planning checkpoint changed")
        for seed in refinement["arguments"]["seeds"]:
            path = args.refine_run / f"seed-{seed}" / "best.pt"
            frozen = json.loads((path.parent / "frozen.json").read_text())
            if fingerprint(path) != frozen["checkpoint_sha256"]:
                raise ValueError("selected refinement checkpoint changed")
            paths[f"refined-{seed}"] = path
        if args.blend_dir:
            selected = json.loads((args.blend_dir / "selection.json").read_text())
            path = args.blend_dir / "best.pt"
            if fingerprint(path) != selected["checkpoint_sha256"]:
                raise ValueError("selected forecast blend changed")
            if selected["configuration"]["policy"] != refinement["policy"]:
                raise ValueError("selected blend policy differs")
            paths["selected-blend"] = path
        for name, path in paths.items():
            belief, _ = load_predictor(path)
            models.append(belief)
            factories[name] = lambda b=belief, p=planned_config: TickMacroAdapter(
                TimingPlannerPolicy(b, p))
            digests[name] = fingerprint(path)
    for index, path in enumerate(args.mpc):
        mpc, settings = load_mpc(path)
        models.append(mpc)
        factories[f"mpc-{index}"] = partial(
            NeuralMPCScheduler,
            model=mpc,
            device="cuda",
            num_simulations=settings.simulations,
            gamma=settings.gamma,
            depth=settings.depth,
            normalize_search=settings.normalize_search,
            coverage_probe_limit=settings.coverage_probe_limit,
            search_method=settings.search_method,
            gumbel_candidates=settings.gumbel_candidates,
            gumbel_q_scale=settings.gumbel_q_scale,
        )
        digests[f"mpc-{index}"] = fingerprint(path)
    if args.ppo_run:
        ppo, _ = load_checked(args.ppo_run / "best.zip", "cuda")
        models.append(ppo)
        factories["recurrent-ppo"] = partial(RecurrentScheduler, ppo)
        digests["recurrent-ppo"] = fingerprint(args.ppo_run / "best.zip")
    factories["round-robin-50"] = partial(DwellSweepScheduler, dwell_steps=50)
    frozen = {
        "checkpoint_sha256": digests,
        "source_sha256": fingerprint(__file__),
        "seeds": list(range(args.seed, args.seed + args.runs)),
        "suites": ["randomized", "receiver-shift", "periodic-scan"],
        "scope": "separate procedural holdout on MPC's own physical evaluation distribution",
    }
    write_json(args.output.with_suffix(".frozen.json"), frozen)
    started = perf_counter()
    report = benchmark(
        [],
        runs=args.runs,
        seed=args.seed,
        split="validation",
        suites=tuple(frozen["suites"]),
        num_bands=8,
        physical_worlds=True,
        profile=True,
        reward=REWARD,
        extra_policies=factories,
        extra_metadata=frozen,
        progress=lambda suite, means: print(
            json.dumps(
                {
                    "suite": suite,
                    "capture": {name: means[name]["interception_ratio"] for name in factories},
                }
            ),
            flush=True,
        ),
    )
    _bootstrap_intervals(report, tuple(name for name in factories
                                     if name.startswith(("timing-", "grouped-", "planner-",
                                                         "refined-", "selected-blend"))))
    report["runtime"] = {
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "latency": "serial per-tick choose_band mean; includes held dwells",
    }
    report["scope"] = (
        "fresh procedural development holdout; required-world training transferred without tuning; "
        "forecast scores are provided separately on the macro-action reporting contract"
    )
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
