"""Frozen timing checkpoint versus shared receiver controls on external stare PDWs."""

import argparse
import json
import os
from dataclasses import asdict, replace
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
from ..replay_evaluation import ReferencePolicy, evaluate_policy, evaluate_policy_batch
from ..timing_replay import TimingReplayPolicy
from .storage import fingerprint, run_lock, write_json


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--selection-seed", type=int, default=53)
    parser.add_argument("--adapter", choices=("legacy", "missing-power"), default="missing-power")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--duration-us", type=int, default=10_000_000)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(arguments)
    if min(args.max_files, args.workers, args.batch_size, args.duration_us) < 1:
        parser.error("positive resource and duration settings required")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("timing replay validation requires CUDA")
    started = perf_counter()
    checkpoint_digest = fingerprint(args.checkpoint)
    plan = (json.loads(args.plan.read_text()) if args.plan else make_plan(
        args.root, split="val", max_files=args.max_files,
        seeds=(0,), selection_seed=args.selection_seed,
    ))
    if plan["split"] != "val" or plan["seeds"] != [0]:
        raise ValueError("external validation requires val recordings and receiver seed zero")
    receiver = ReplayConfig(stop_us=args.duration_us, retune_us=2000,
                            detection_probability=0.9)
    interface = InterfaceConfig(dwell_us=(1000, 10000, 50000))
    template = TimingReplayPolicy.from_checkpoint(args.checkpoint, args.batch_size,
                                                   adapter=args.adapter)
    models = [("timing-trained", args.checkpoint, checkpoint_digest, template)]
    for name, path, adapter in (
        ("timing-reference", args.reference_checkpoint, "missing-power"),
        ("timing-legacy", args.legacy_checkpoint, "legacy"),
    ):
        if path is not None:
            models.append((name, path, fingerprint(path), TimingReplayPolicy.from_checkpoint(
                path, args.batch_size, adapter=adapter)))
    for _, _, _, policy in models:
        seen = set(policy.metadata.get("trained_on", ())) | set(
            policy.metadata.get("development_on", ()))
        if seen.intersection(r["sha256"] for r in plan["recordings"]):
            raise ValueError("validation recording overlaps model fitting or checkpoint selection")
    if args.verify_only:
        paths = validate_plan(args.root, plan)[:2]
        short = replace(receiver, stop_us=512000)
        batched = evaluate_policy_batch([(p, short) for p in paths], lambda: template, interface)
        for path, expected in zip(paths, batched, strict=True):
            actual = evaluate_policy(path, template, short, interface)
            for key in ("truth_pulses", "intercepted_pulses", "delivered_pulses",
                        "emitters_discovered", "simulated_us"):
                if actual[key] != expected[key]:
                    raise ValueError(f"serial/batched replay mismatch: {key}")
        print("Serial and batched CUDA replay agree on two external recording windows.")
        return 0
    controls = [PolicySpec(f"sweep-{d}", partial(ReferencePolicy, "sweep", index),
                           f"complete listening dwell {d} ms")
                for index, d in enumerate((1, 10, 50))]
    controls.append(PolicySpec("rate-probe", RateProbePolicy, "causal rate; revisit 500 ms"))
    with run_lock(args.run_dir):
        if (args.run_dir / "comparison.json").exists():
            raise ValueError("replay comparison already exists")
        write_json(args.run_dir / "plan.json", plan)
        paths = validate_plan(args.root, plan)
        jobs = [(i, path, 0, controls, receiver, interface) for i, path in enumerate(paths)]
        results = _run_jobs(_replay_job, jobs, args.workers)
        write_json(args.run_dir / "controls.json", results)
        print(json.dumps({"phase": "controls", "recordings": len(results)}), flush=True)
        learned_specs = []
        for name, _, digest, template in models:
            learned = PolicySpec(
                name, lambda template=template: template,
                f"frozen CUDA timing checkpoint {digest}; adapter={template.adapter}",
            )
            learned_specs.append(learned)
            for offset in range(0, len(paths), args.batch_size):
                reports = evaluate_policy_batch(
                    [(p, receiver) for p in paths[offset:offset + args.batch_size]],
                    learned.factory, interface,
                )
                for row, report in zip(results[offset:offset + len(reports)], reports, strict=True):
                    row["policies"][learned.name] = report
            write_json(args.run_dir / f"{name}.json", results)
            print(json.dumps({"phase": "model", "policy": name, "recordings": len(paths)}),
                  flush=True)
        for row in results:
            values = list(row["policies"].values())
            if len({v["truth_pulses"] for v in values}) != 1 or len({
                v["simulated_us"] for v in values
            }) != 1:
                raise ValueError("paired replay policies received different truth or time budgets")
        validate_plan(args.root, plan)
        if any(fingerprint(path) != digest for _, path, digest, _ in models):
            raise ValueError("checkpoint changed during replay")
        report = {"schema_version": 1, "backend": "external_synthetic_stare",
            "plan": plan, "receiver": asdict(receiver), "interface": asdict(interface),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoints": {name: {"sha256": digest, "path": str(path),
                "adapter": policy.adapter, "count_limit": policy.count_limit}
                for name, path, digest, policy in models},
            "summary": summarize(results, [*controls, *learned_specs], "sweep-50"),
            "versus_rate_probe": summarize(results, [*controls, *learned_specs], "rate-probe"),
            "results": results, "elapsed_seconds": perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "scope": "validation recordings; frozen checkpoint; synthetic external data; not real RF",
            "transfer": {"adapter": models[0][3].adapter, "tick_us": template.tick_us,
                "count_limit": models[0][3].count_limit, "amplitude_dbm": "unavailable"}}
        write_json(args.run_dir / "comparison.json", report)
        print(json.dumps({name: {key: metrics[key] for key in (
            "interception_ratio", "discovery_fraction"
        )} for name, metrics in report["summary"].items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
