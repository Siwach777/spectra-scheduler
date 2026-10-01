"""Paired replay input ablations with a frozen timing checkpoint and recording plan."""

import argparse
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from ..policy_benchmark import (
    PolicySpec,
    _replay_job,
    _run_jobs,
    make_plan,
    summarize,
    validate_plan,
)
from ..pulse_replay import ReplayConfig
from ..replay_baselines import RateProbePolicy
from ..replay_env import InterfaceConfig
from ..replay_evaluation import ReferencePolicy, evaluate_policy_batch
from ..timing_belief import BeliefPolicyConfig, load_belief
from ..timing_replay import PowerBlindTimingPredictor, TimingReplayPolicy
from .storage import fingerprint, run_lock, write_json


VARIANTS = {
    "timing-legacy": (True, 4, False),
    "timing-missing-power": (False, 4, False),
    "timing-raw-counts": (True, None, False),
    "timing-missing-power-raw-counts": (False, None, False),
    "timing-count-memory": (False, None, True),
}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--selection-seed", type=int, default=71)
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--duration-us", type=int, default=10_000_000)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                        default=list(VARIANTS))
    args = parser.parse_args(arguments)
    if min(args.max_files, args.workers, args.batch_size, args.duration_us) < 1:
        parser.error("positive resource and duration settings required")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("timing replay transfer evaluation requires CUDA")
    started = perf_counter()
    digest = fingerprint(args.checkpoint)
    plan = (json.loads(args.plan.read_text()) if args.plan else make_plan(
        args.root, split=args.split, max_files=args.max_files,
        seeds=(0,), selection_seed=args.selection_seed,
    ))
    if plan["split"] not in ("train", "val") or plan["seeds"] != [0]:
        raise ValueError("transfer ablations require a train/val plan and receiver seed zero")
    receiver = ReplayConfig(stop_us=args.duration_us, retune_us=2000,
                            detection_probability=0.9)
    interface = InterfaceConfig(dwell_us=(1000, 10000, 50000))
    model, metadata = load_belief(args.checkpoint)
    settings = dict(metadata["policy"])
    settings["dwells"] = tuple(settings["dwells"])
    config = BeliefPolicyConfig(**settings)
    controls = [
        PolicySpec("sweep-50", partial(ReferencePolicy, "sweep", 2), "fixed sweep; 50 ms"),
        PolicySpec("rate-probe", RateProbePolicy, "causal pulse rate; revisit 500 ms"),
    ]
    with run_lock(args.run_dir):
        if (args.run_dir / "comparison.json").exists():
            raise ValueError("transfer comparison already exists")
        write_json(args.run_dir / "plan.json", plan)
        paths = validate_plan(args.root, plan)
        jobs = [(i, path, 0, controls, receiver, interface) for i, path in enumerate(paths)]
        results = _run_jobs(_replay_job, jobs, args.workers)
        write_json(args.run_dir / "controls.json", results)
        policies = controls.copy()
        for name in args.variants:
            power_available, count_limit, persistent_rates = VARIANTS[name]
            predictor = model if power_available else PowerBlindTimingPredictor(model)
            template = TimingReplayPolicy(
                predictor, config, args.batch_size, count_limit=count_limit,
                persistent_rates=persistent_rates,
            )
            policy = PolicySpec(name, lambda template=template: template,
                f"frozen {digest}; power_available={power_available}; count_limit={count_limit}; "
                f"persistent_rates={persistent_rates}")
            policies.append(policy)
            for offset in range(0, len(paths), args.batch_size):
                reports = evaluate_policy_batch(
                    [(p, receiver) for p in paths[offset:offset + args.batch_size]],
                    policy.factory, interface,
                )
                for row, report in zip(results[offset:offset + len(reports)], reports, strict=True):
                    row["policies"][name] = report
            write_json(args.run_dir / f"{name}.json", results)
            summary = summarize(results, [policy, controls[0]], "sweep-50")
            print(json.dumps({"policy": name, "capture": summary[name]["interception_ratio"],
                              "discovery": summary[name]["discovery_fraction"]}), flush=True)
        for row in results:
            values = list(row["policies"].values())
            if len({v["truth_pulses"] for v in values}) != 1 or len({
                v["simulated_us"] for v in values
            }) != 1:
                raise ValueError("paired policies received different truth or time budgets")
        validate_plan(args.root, plan)
        if fingerprint(args.checkpoint) != digest:
            raise ValueError("checkpoint changed during replay")
        write_json(args.run_dir / "comparison.json", {
            "schema_version": 1, "backend": "external_synthetic_stare", "plan": plan,
            "receiver": asdict(receiver), "interface": asdict(interface),
            "checkpoint_sha256": digest,
            "zero_power_detection_weight": float(torch.sigmoid(
                -model.quality_threshold * model.quality_gain.exp())),
            "summary": summarize(results, policies, "sweep-50"),
            "versus_rate_probe": summarize(results, policies, "rate-probe"),
            "results": results, "elapsed_seconds": perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "scope": "frozen checkpoint input ablations; synthetic external PDWs; not real RF",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
