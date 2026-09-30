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
from .config import MAX_BANDS, saved_config
from .data import collect
from .learning import batch_loss
from .scheduler import NeuralMPCScheduler


def episode_mean(episode, key, column=None):
    values = episode[key] if column is None else episode[key][:, column]
    if "elapsed" not in episode:
        return float(values.mean())
    if key == "rewards":
        return float(episode["raw_rewards"].sum() / episode["elapsed"].sum())
    return float((values * episode["elapsed"]).sum() / episode["elapsed"].sum())


def selection_score(results):
    """Capture/discovery balance across both development distributions."""
    scores = []
    for name, row in results.items():
        if name.endswith("/search"):
            capture, discovery = row["interception_ratio"], row["emitter_discovery_ratio"]
            scores.append(2 * capture * discovery / max(capture + discovery, 1e-12))
    return float(np.mean(scores))


def validate(model, cfg, pool, split="validation", progress=None, prefix="Validation"):
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
                progress=progress.callback(f"{prefix}: {name}") if progress else None,
            )
            results[name] = {
                "reward": float(np.mean([episode_mean(e, "rewards") for e in episodes])),
                **{
                    key: float(np.mean([e["metrics"][key] for e in episodes]))
                    for key in ("interception_ratio", "emitter_discovery_ratio", "max_band_gap")
                },
            }
            actions = torch.cat(
                [
                    torch.repeat_interleave(e["actions"] // len(model.dwell_steps), e["elapsed"])
                    if "elapsed" in e
                    else e["actions"]
                    for e in episodes
                ]
            )
            results[name]["dominant_band_fraction"] = float(
                torch.bincount(actions, minlength=MAX_BANDS).max() / len(actions)
            )
            results[name]["bands_visited_mean"] = float(
                np.mean([len((e["actions"] // len(model.dwell_steps)).unique()) for e in episodes])
            )
            results[name]["listening_fraction"] = float(
                np.mean([episode_mean(e, "features", 9) for e in episodes])
            )
            with torch.no_grad():
                _, errors = batch_loss(
                    model, model, episodes, cfg, np.random.default_rng(0), cfg.device
                )
            results[name]["prediction_losses"] = errors
            if progress:
                progress.message(
                    f"  {name}: reward={results[name]['reward']:.4f}, "
                    f"interception={results[name]['interception_ratio']:.2%}"
                )
                if results[name]["dominant_band_fraction"] > 0.95:
                    progress.message("  Warning: more than 95% of actions choose one band.")
    return results


@torch.no_grad()
def probe_reward_error(model, episodes, device):
    """Fixed held-out trajectories: comparable one-step reward MSE across iterations."""
    from torch.nn.utils.rnn import pad_sequence

    lengths = torch.tensor([len(e["actions"]) for e in episodes], device=device)
    x = pad_sequence([e["features"] for e in episodes], batch_first=True).to(device)
    hidden = model.representation.initial_state(len(episodes)).to(device)
    latent, _ = model.representation(x, hidden)
    before = torch.cat([hidden.transpose(0, 1), latent[:, :-1]], dim=1)
    actions = pad_sequence([e["actions"] for e in episodes], batch_first=True).to(device)
    actual = pad_sequence([e["rewards"] for e in episodes], batch_first=True).to(device)
    _, predicted = model.dynamics(before.flatten(0, 1), actions.flatten())
    valid = (torch.arange(x.shape[1], device=device)[None] < lengths[:, None]).flatten()
    return float(F.mse_loss(predicted[valid], actual.flatten()[valid]))


def evaluate_run(directory, split="test", episodes=30, workers=1):
    """Evaluate best checkpoint with its saved search settings, without learning."""
    directory = Path(directory)
    checkpoint = directory / "best.pt"
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)["metadata"]
    settings = dict(metadata["config"])
    if not torch.cuda.is_available():
        raise RuntimeError("MPC neural evaluation requires CUDA")
    settings.update(device="cuda", workers=workers, validation_episodes=episodes)
    cfg = saved_config(settings)
    model = load_model(checkpoint, torch.device("cuda"))
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
