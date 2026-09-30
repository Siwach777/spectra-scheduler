"""Frozen CUDA checkpoint and structural attribution on selection worlds only."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from ..policy_benchmark import PolicySpec, benchmark_synthetic
from ..timing_belief import BeliefPolicyConfig, TimingBeliefPolicy, load_belief
from .phase_selection import BetaPhaseScheduler
from .storage import fingerprint, write_json
from .timing_study import ObservedRateScheduler


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--revisits", type=int, nargs="+", default=[96, 192])
    parser.add_argument("--probes", type=int, nargs="+", default=[8])
    args = parser.parse_args(arguments)
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    policies, metadata = [], {}
    for index, path in enumerate(args.checkpoint):
        digest = fingerprint(path)
        model, model_metadata = load_belief(path)
        metadata[str(index)] = {"path": str(path), "sha256": digest, "metadata": model_metadata}
        for revisit in args.revisits:
            for probe in args.probes:
                config = BeliefPolicyConfig(revisit=revisit, probe=probe, exploration=0.02)
                suffix = f"revisit{revisit}" + (f"-probe{probe}" if args.probes != [8] else "")
                name = f"checkpoint-{index}-{suffix}"
                policies.append(PolicySpec(name, partial(TimingBeliefPolicy, model, config),
                    json.dumps({"checkpoint_sha256": digest, "policy": asdict(config),
                                "source_sha256": fingerprint(__file__)}, sort_keys=True)))
    configs = [(revisit, probe) for revisit in args.revisits for probe in args.probes]
    baseline = None
    for revisit, probe in configs:
        config = BeliefPolicyConfig(revisit=revisit, probe=probe, exploration=0.02)
        suffix = str(revisit) + (f"-probe{probe}" if args.probes != [8] else "")
        baseline = baseline or f"observed-rate-{suffix}"
        policies.extend([
            PolicySpec(f"observed-rate-{suffix}", partial(ObservedRateScheduler, config),
                       str(asdict(config))),
            PolicySpec(f"beta-phase-{suffix}", partial(BetaPhaseScheduler, config),
                       json.dumps({"learned": False, "policy": asdict(config)}, sort_keys=True)),
        ])
    started = time.perf_counter()
    report = benchmark_synthetic(policies, baseline=baseline,
        split="val", seeds=tuple(range(2000, 2000 + args.runs)), workers=1)
    report["scope"] = "selection-only development worlds; reporting and test worlds unused"
    report["checkpoints"] = metadata
    report["evaluation_seconds"] = time.perf_counter() - started
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    for index, path in enumerate(args.checkpoint):
        if fingerprint(path) != metadata[str(index)]["sha256"]:
            raise RuntimeError("checkpoint changed during evaluation; use a frozen copy")
    write_json(args.output, report)
    print(json.dumps({s: {p: m["interception_ratio"]["mean"] for p, m in v.items()}
                      for s, v in report["by_scenario"].items()}, indent=2))
    print(json.dumps({"evaluation_seconds": report["evaluation_seconds"],
                      "peak_cuda_bytes": report["peak_cuda_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
