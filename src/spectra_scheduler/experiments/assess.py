"""Assess frozen study checkpoints and ensembles against adaptive probing."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ..forecast_ensemble import ensemble_spec
from ..policy_benchmark import PolicySpec, benchmark_policies, summarize
from ..replay_baselines import RateProbePolicy
from .storage import checkpoint_path, fingerprint, verify_artifacts, write_json


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(arguments)
    if not torch.cuda.is_available():
        raise ValueError("neural assessment requires CUDA")
    torch.set_num_threads(1)
    settings = json.loads((args.study / "study.json").read_text())
    plans = json.loads((args.study / "plans.json").read_text())
    prior = json.loads((args.study / "comparison.json").read_text())
    if prior["plan"] != plans["report"]:
        raise ValueError("reporting plan changed")
    policies = [
        PolicySpec(
            "rate-probe",
            RateProbePolicy,
            "observed-rate;short-probe;long-exploitation;500ms-revisit",
        )
    ]
    for encoder in settings["encoders"]:
        paths = []
        for seed in settings["seeds"]:
            run = args.study / f"{encoder}-{seed}"
            state = json.loads((run / "state.json").read_text())
            verify_artifacts(run, state["best"])
            paths.append(checkpoint_path(run, state["best"], "policy.bin"))
            metadata = next(p for p in prior["policies"] if p["name"] == f"{encoder}-{seed}")
            if not metadata["provenance"].startswith(f"sha256:{fingerprint(paths[-1])};"):
                raise ValueError("individual report differs from frozen ensemble checkpoints")
        policies.append(ensemble_spec(paths, name=f"{encoder}-ensemble", device="cuda"))
    report = benchmark_policies(
        settings["root"], plans["report"], policies, baseline="rate-probe", inference_batch_size=32
    )
    # Retain each frozen individual model's exact prior observations and forecasts.
    # No repeated GPU execution just to change the reference for paired statistics.
    for current, previous in zip(report["results"], prior["results"], strict=True):
        if (current["group"], current["seed"]) != (previous["group"], previous["seed"]):
            raise ValueError("recording/seed pairing changed")
        current["policies"].update(previous["policies"])
    registered = [*policies]
    for item in prior["policies"]:
        registered.append(SimpleNamespace(name=item["name"]))
    report["summary"] = summarize(report["results"], registered, "rate-probe")
    report["policies"].extend(prior["policies"])
    report["source_comparison_sha256"] = fingerprint(args.study / "comparison.json")
    report["scope"] = "development validation; no test data; ensembles use all three seeds"
    write_json(args.study / "assessment.json", report)
    print(
        json.dumps(
            {
                name: {
                    metric: values[metric]
                    for metric in ("interception_ratio", "discovery_fraction")
                }
                for name, values in report["summary"].items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
