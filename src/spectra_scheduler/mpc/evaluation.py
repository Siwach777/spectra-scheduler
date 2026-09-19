"""Held-out metrics, fixed prediction probes and paired baseline evaluation."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spectra_scheduler.rl import RewardConfig

from .checkpoints import atomic_json, implementation_hashes, load_model
from .config import MAX_BANDS, Config
from .data import collect
from .learning import batch_loss
from .scheduler import NeuralMPCScheduler


def validate(model, cfg, pool, split="validation"):
    results = {}
    for shifted in (False, True):
        for policy_only in (False, True):
            name = ("shift" if shifted else "randomized") + (
                "/policy" if policy_only else "/search"
            )
            episodes = collect(
                model,
                cfg,
                list(range(10000, 10000 + cfg.validation_episodes)),
                split,
                pool,
                shifted=shifted,
                policy_only=policy_only,
            )
            results[name] = {
                "reward": float(np.mean([float(e["rewards"].mean()) for e in episodes])),
                **{
                    key: float(np.mean([e["metrics"][key] for e in episodes]))
                    for key in ("interception_ratio", "emitter_discovery_ratio", "max_band_gap")
                },
            }
            with torch.no_grad():
                _, errors = batch_loss(
                    model, model, episodes, cfg, np.random.default_rng(0), cfg.device
                )
            results[name]["prediction_losses"] = errors
    return results


@torch.no_grad()
def probe_reward_error(model, episodes, device):
    """Fixed held-out trajectories: comparable one-step reward MSE across iterations."""
    x = torch.stack([e["features"] for e in episodes]).to(device)
    hidden = model.representation.initial_state(len(episodes)).to(device)
    latent, _ = model.representation(x, hidden)
    before = torch.cat([hidden.transpose(0, 1), latent[:, :-1]], dim=1)
    actions = torch.stack([e["actions"] for e in episodes]).to(device)
    actual = torch.stack([e["rewards"] for e in episodes]).to(device)
    _, predicted = model.dynamics(before.flatten(0, 1), actions.flatten())
    return float(F.mse_loss(predicted, actual.flatten()))


def evaluate_run(directory, split="test", episodes=30, workers=1):
    """Evaluate best checkpoint with its saved search settings, without learning."""
    directory = Path(directory)
    checkpoint = directory / "best.pt"
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)["metadata"]
    settings = dict(metadata["config"])
    settings.update(device="cpu", workers=workers, validation_episodes=episodes)
    cfg = Config(**settings)
    model = load_model(checkpoint)
    torch.set_num_threads(1)
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) if workers > 1 else None
    try:
        metrics = validate(model, cfg, pool, split)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    report = {
        "split": split,
        "seeds": list(range(10000, 10000 + episodes)),
        "config": asdict(cfg),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_iteration": metadata["iteration"],
        "metrics": metrics,
    }
    atomic_json(directory / f"evaluation-{split}.json", report)
    return report


def _benchmark(
    model_path: str | Path,
    runs: int = 20,
    start_seed: int = 10000,
    device: str = "cpu",
    simulations: int = 50,
    suites: tuple = ("randomized",),
    output: str = "reports/generated/neural-mpc.json",
) -> None:
    """Benchmark the Neural-MPC scheduler against baselines."""
    from spectra_scheduler.learning_cli import write_json
    from spectra_scheduler.rl_benchmark import benchmark

    model = load_model(model_path, torch.device(device))
    report = benchmark(
        [],
        runs=runs,
        seed=start_seed,
        suites=suites,
        num_bands=MAX_BANDS,
        reward=RewardConfig(coverage=0.2),
        profile=True,
        extra_policies={
            "neural-mpc": lambda: NeuralMPCScheduler(
                model=model, num_simulations=simulations, device=device
            ),
            "policy-only": lambda: NeuralMPCScheduler(
                model=model, num_simulations=0, device=device
            ),
        },
        extra_metadata={
            "sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            "simulations": simulations,
            "depth": 5,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "implementation_sha256": implementation_hashes(),
        },
    )
    write_json(Path(output), report)
    print(f"Paired benchmark saved to {output}")
    return
