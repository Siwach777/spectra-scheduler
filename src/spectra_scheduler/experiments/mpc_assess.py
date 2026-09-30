"""Paired CUDA assessment of frozen MPC checkpoints on development worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..mpc.checkpoints import implementation_hashes, load_model
from ..mpc.config import REWARD
from ..mpc.scheduler import NeuralMPCScheduler
from ..rl_benchmark import SUITES, benchmark
from ..schedulers import AdaptiveDwellScheduler, DwellSweepScheduler
from .storage import fingerprint, write_json


def _checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"missing MPC checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    return load_model(path, torch.device("cuda")), metadata, fingerprint(path)


def _bootstrap_intervals(report: dict, candidates: tuple[str, ...]) -> None:
    """Resample paired development worlds; retain their shared scenario seeds."""
    for suite in report["suites"].values():
        rows = suite["episodes"]
        for candidate in candidates:
            for reference in suite["mean_metrics"]:
                if candidate == reference:
                    continue
                deltas = np.fromiter(
                    (
                        row["metrics"][candidate]["interception_ratio"]
                        - row["metrics"][reference]["interception_ratio"]
                        for row in rows
                    ),
                    dtype=np.float64,
                    count=len(rows),
                )
                # Independent seed per pair keeps intervals stable if the policy
                # registration order changes while preserving paired resampling.
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"{candidate}:{reference}:{rows[0]['scenario_sha256']}".encode()
                    ).digest()[:8],
                    "little",
                )
                rng = np.random.default_rng(seed)
                draws = rng.integers(len(deltas), size=(2000, len(deltas)))
                means = deltas[draws].mean(axis=1)
                suite["paired_interception_delta"][candidate][reference][
                    "bootstrap_95_interval"
                ] = np.quantile(means, [0.025, 0.975]).tolist()


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--suite", action="append", choices=SUITES)
    args = parser.parse_args(arguments)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA inference required for MPC assessment")
    torch.set_num_threads(1)

    trained, metadata, trained_digest = _checkpoint(args.run_dir / "best.pt")
    untrained, initial_metadata, initial_digest = _checkpoint(args.run_dir / "untrained.pt")
    if initial_metadata.get("iteration") != 0:
        raise ValueError("untrained checkpoint must be from iteration zero")
    if metadata.get("config") != initial_metadata.get("config"):
        raise ValueError("trained and untrained checkpoints have different run configurations")
    if trained.max_bands != untrained.max_bands or trained.max_bands != 8:
        raise ValueError("assessment requires matching eight-band MPC checkpoints")

    saved = metadata.get("config", {})
    physical_worlds = bool(saved.get("physical_contract", False))
    simulations = args.simulations if args.simulations is not None else saved.get("simulations", 32)
    if type(simulations) is not int or simulations < 1:
        raise ValueError("search simulations must be a positive integer")
    gamma = float(saved.get("gamma", 0.97))
    if not 0 < gamma < 1:
        raise ValueError("checkpoint contains an invalid physical-time discount")
    depth = saved.get("depth", 5)
    if type(depth) is not int or depth < 1:
        raise ValueError("checkpoint contains an invalid search depth")
    normalize_search = saved.get("normalize_search", True)
    if type(normalize_search) is not bool:
        raise ValueError("checkpoint contains an invalid search normalization setting")
    search_settings = {
        "gamma": gamma,
        "depth": depth,
        "normalize_search": normalize_search,
        "device": "cuda",
        "coverage_probe_limit": int(saved.get("coverage_probe_limit", 0)),
        "search_method": saved.get("search_method", "puct"),
        "gumbel_candidates": int(saved.get("gumbel_candidates", 8)),
        "gumbel_q_scale": float(saved.get("gumbel_q_scale", 2.0)),
    }
    policies = {
        "mpc-search": partial(
            NeuralMPCScheduler,
            model=trained,
            num_simulations=simulations,
            **search_settings,
        ),
        "mpc-policy": partial(
            NeuralMPCScheduler, model=trained, num_simulations=0, **search_settings
        ),
        "untrained-search": partial(
            NeuralMPCScheduler,
            model=untrained,
            num_simulations=simulations,
            **search_settings,
        ),
        "untrained-policy": partial(
            NeuralMPCScheduler, model=untrained, num_simulations=0, **search_settings
        ),
        "dwell-4": partial(DwellSweepScheduler, dwell_steps=4),
        "dwell-8": partial(DwellSweepScheduler, dwell_steps=8),
        "adaptive-long": partial(
            AdaptiveDwellScheduler,
            minimum_dwell_steps=4,
            hit_extension_steps=4,
            maximum_dwell_steps=16,
        ),
    }
    report = benchmark(
        [],
        runs=args.runs,
        seed=args.seed,
        split="validation",
        suites=tuple(args.suite)
        if args.suite
        else (("randomized", "receiver-shift", "periodic-scan") if physical_worlds else SUITES),
        num_bands=8,
        physical_worlds=physical_worlds,
        reward=REWARD,
        extra_policies=policies,
        extra_metadata={
            "trained": {"sha256": trained_digest, "metadata": metadata},
            "untrained": {"sha256": initial_digest, "metadata": initial_metadata},
            "search_simulations": simulations,
            "search_settings": search_settings,
            "mpc_implementation_sha256": implementation_hashes(),
        },
        progress=lambda suite, means: print(
            json.dumps(
                {
                    "suite": suite,
                    "capture": {
                        name: values["interception_ratio"] for name, values in means.items()
                    },
                }
            ),
            flush=True,
        ),
    )
    _bootstrap_intervals(
        report,
        ("mpc-search", "mpc-policy", "untrained-search", "untrained-policy"),
    )
    report["scope"] = (
        "development validation; paired scenario worlds; one trained seed; "
        "no test-set or operational-performance claim"
    )
    report["assessment_sha256"] = fingerprint(__file__)
    write_json(args.run_dir / "mpc-assessment.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
