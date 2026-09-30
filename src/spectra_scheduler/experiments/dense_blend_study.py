"""Freeze an observed-rate/MLP blend on selection data, then report paired capture.

Existing model artifacts remain immutable. One shared weight is selected with
seed zero; separate reporting evaluates every supplied training seed, with both
the original rate-probe and a retune-aware zero-model-weight control.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..dense_policy import DenseValuePolicy, load_dense_model
from ..policy_benchmark import PolicySpec, benchmark_policies, summarize, validate_plan
from ..pulse_replay import ReplayConfig
from ..replay_baselines import RateProbePolicy
from ..replay_env import InterfaceConfig
from .storage import fingerprint, run_lock, write_json


def pooled_capture(report, name):
    rows = [row["policies"][name] for row in report["results"]]
    truth = sum(row["truth_pulses"] for row in rows)
    return sum(row["intercepted_pulses"] for row in rows) / truth if truth else None


def pooled_difference_interval(report, name, baseline, samples=2000, seed=0):
    """Bootstrap independent recording groups, preserving paired pulse weights."""
    groups = {}
    for result in report["results"]:
        candidate, control = (result["policies"][key] for key in (name, baseline))
        groups.setdefault(result["group"], []).append(
            [
                candidate["intercepted_pulses"],
                control["intercepted_pulses"],
                candidate["truth_pulses"],
            ]
        )
        if candidate["truth_pulses"] != control["truth_pulses"]:
            raise ValueError("paired policies have different truth denominators")
    counts = np.asarray([np.mean(rows, axis=0) for rows in groups.values()])
    if len(counts) < 2 or counts[:, 2].sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    differences = np.empty(samples)
    for start in range(0, samples, 128):
        stop = min(start + 128, samples)
        indices = rng.integers(len(counts), size=(stop - start, len(counts)))
        totals = counts[indices].sum(axis=1)
        differences[start:stop] = (totals[:, 0] - totals[:, 1]) / totals[:, 2]
    return np.quantile(differences, [0.025, 0.975]).tolist()


def annotate(report, policies):
    report["pooled_capture"] = {p.name: pooled_capture(report, p.name) for p in policies}
    report["pooled_paired_bootstrap_95_interval"] = {
        p.name: pooled_difference_interval(report, p.name, report["baseline"]) for p in policies
    }
    return report


def _spec(model, specification, trained_on, digest, name, weight):
    return PolicySpec(
        name,
        partial(DenseValuePolicy, model, specification, model_weight=weight),
        f"sha256:{digest};model_weight:{weight};revisit_us:500000",
        trained_on,
    )


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--plans", type=Path, default=Path("artifacts/predictor-study/plans.json"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/dense-blend-study"))
    parser.add_argument("--weights", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(arguments)
    if args.selection_only and args.report_only:
        parser.error("selection-only and report-only are mutually exclusive")
    if not torch.cuda.is_available():
        raise RuntimeError("dense blend neural evaluation requires CUDA")
    if args.inference_batch_size < 2 or any(
        not np.isfinite(w) or not 0 <= w <= 1 for w in args.weights
    ):
        parser.error("batched inference requires batch size >=2 and finite weights in [0,1]")
    if len(args.weights) != len(set(args.weights)):
        parser.error("candidate weights must be unique")
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    plans = json.loads(args.plans.read_text())
    hashes = {}
    for key, split in (("train", "train"), ("selection", "val"), ("report", "val")):
        if plans[key]["split"] != split:
            raise ValueError("blend study requires disjoint train/validation plans")
        validate_plan(args.root, plans[key])
        hashes[key] = {row["sha256"] for row in plans[key]["recordings"]}
    if any(
        hashes[a] & hashes[b]
        for a, b in (("train", "selection"), ("train", "report"), ("selection", "report"))
    ):
        raise ValueError("blend study content splits overlap")
    digests = [fingerprint(path) for path in args.checkpoints]
    loaded = [load_dense_model(path, "cuda") for path in args.checkpoints]
    specification = loaded[0][1]
    if any(spec != specification for _, spec, _ in loaded):
        raise ValueError("model receiver interfaces differ")
    if any((hashes["selection"] | hashes["report"]) & set(trained) for _, _, trained in loaded):
        raise ValueError("model training overlaps selection or reporting")
    receiver, interface = (
        ReplayConfig(**specification["receiver"]),
        InterfaceConfig(**specification["interface"]),
    )
    source_root = Path(__file__).resolve().parent.parent
    semantic = {
        "version": 1,
        "objective": "maximum mean recording capture; ties choose smaller model weight",
        "weight_selection_training_seed": 0,
        "weights": args.weights,
        "checkpoints": [
            {"path": str(path), "sha256": digest}
            for path, digest in zip(args.checkpoints, digests, strict=True)
        ],
        "plans_sha256": fingerprint(args.plans),
        "selection_plan": plans["selection"],
        "report_plan": plans["report"],
        "source_sha256": {
            name: fingerprint(source_root / name)
            for name in (
                "dense_policy.py",
                "dense_value.py",
                "policy_benchmark.py",
                "replay_evaluation.py",
                "experiments/dense_blend_study.py",
            )
        },
    }
    baseline = PolicySpec("rate-probe", RateProbePolicy, "short-probe/long-exploit", ())
    with run_lock(args.run_dir):
        frozen_path = args.run_dir / "frozen.json"
        if not args.report_only:
            model, spec, trained = loaded[0]
            policies = [
                _spec(model, spec, trained, digests[0], f"blend-{w:g}", w) for w in args.weights
            ] + [baseline]
            selection = annotate(
                benchmark_policies(
                    args.root,
                    plans["selection"],
                    policies,
                    baseline="rate-probe",
                    receiver=receiver,
                    interface=interface,
                    inference_batch_size=args.inference_batch_size,
                ),
                policies,
            )
            selection["semantic"] = semantic
            write_json(args.run_dir / "selection.json", selection)
            chosen = max(
                args.weights,
                key=lambda w: (
                    selection["summary"][f"blend-{w:g}"]["interception_ratio"]["mean"],
                    -w,
                ),
            )
            write_json(
                frozen_path,
                {
                    "semantic": semantic,
                    "selected_weight": chosen,
                    "selection_sha256": fingerprint(args.run_dir / "selection.json"),
                },
            )
            print(
                json.dumps(
                    {
                        "phase": "selection",
                        "chosen_weight": chosen,
                        "pooled_capture": selection["pooled_capture"],
                        "mean_capture": {
                            name: row["interception_ratio"]["mean"]
                            for name, row in selection["summary"].items()
                        },
                    }
                ),
                flush=True,
            )
        frozen = json.loads(frozen_path.read_text())
        if frozen["semantic"] != semantic or frozen["selection_sha256"] != fingerprint(
            args.run_dir / "selection.json"
        ):
            raise ValueError("frozen selection inputs changed")
        if args.selection_only:
            return 0
        chosen = frozen["selected_weight"]
        policies = [
            _spec(model, spec, trained, digest, f"blend-seed-{i}", chosen)
            for i, ((model, spec, trained), digest) in enumerate(zip(loaded, digests, strict=True))
        ]
        model, spec, trained = loaded[0]
        policies += [_spec(model, spec, trained, digests[0], "rate-retune-aware", 0), baseline]
        report = annotate(
            benchmark_policies(
                args.root,
                plans["report"],
                policies,
                baseline="rate-probe",
                receiver=receiver,
                interface=interface,
                inference_batch_size=args.inference_batch_size,
            ),
            policies,
        )
        report["retune_aware_comparison"] = summarize(
            report["results"], policies, "rate-retune-aware"
        )
        report["pooled_vs_retune_aware_95_interval"] = {
            p.name: pooled_difference_interval(report, p.name, "rate-retune-aware")
            for p in policies
        }
        report["frozen_selection"] = frozen
        report["scope"] = (
            "Shared blend selected on seed-zero selection recordings; all training seeds "
            "evaluated on separate reporting validation recordings; test split unused."
        )
        report["resources"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "inference_batch_size": args.inference_batch_size,
        }
        if [fingerprint(path) for path in args.checkpoints] != digests:
            raise ValueError("checkpoint changed during study")
        write_json(args.run_dir / "comparison.json", report)
        print(
            json.dumps(
                {
                    "phase": "report",
                    "pooled_capture": report["pooled_capture"],
                    "resources": report["resources"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
