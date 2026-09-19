"""Resumable actor/learner orchestration and validation-based checkpoint selection."""

from __future__ import annotations

import copy
import fcntl
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from spectra_scheduler.rl_scenarios import GENERATOR_VERSION

from .checkpoints import atomic_json, implementation_hashes, load_model, save_model, save_training
from .config import VERSION
from .data import Replay, collect
from .evaluation import probe_reward_error, validate
from .learning import batch_loss
from .model import NeuralMPCModel
from .progress import LiveProgress


def run(cfg, directory, resume=False, initial=None):
    """Synchronous actor/learner iterations, restartable at committed boundaries."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(cfg, directory, resume, initial)


def _run_locked(cfg, directory, resume, initial):
    progress = LiveProgress()
    progress.message(
        f"Training in foreground | device={cfg.device} | workers={cfg.workers} | "
        f"iterations={cfg.iterations} | run={directory}"
    )
    checkpoint = directory / "training.pt"
    if resume and initial:
        raise ValueError("resume and initialization are mutually exclusive")
    if not resume and (checkpoint.exists() or (directory / "config.json").exists()):
        raise FileExistsError("existing run: use --resume or a new directory")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or expose the GPU")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = (load_model(initial) if initial else NeuralMPCModel()).to(cfg.device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    replay = Replay(cfg.replay_capacity)
    iteration, best, history = 0, -float("inf"), []
    probes = None
    if resume:
        progress.message("Loading model, optimizer and replay checkpoint...")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["version"] != VERSION or saved["generator"] != GENERATOR_VERSION:
            raise ValueError("incompatible training checkpoint")
        old, new = dict(saved["config"]), asdict(cfg)
        for key in ("iterations", "device", "threads", "workers"):
            old.pop(key)
            new.pop(key)
        if old != new:
            raise ValueError("resume configuration differs from saved training semantics")
        model.load_state_dict(saved["model"])
        target.load_state_dict(saved["target"])
        optimizer.load_state_dict(saved["optimizer"])
        replay.extend(saved["replay"])
        iteration, best, history = saved["iteration"], saved["best"], saved["history"]
        probes = saved["probes"]
        if cfg.iterations < iteration:
            raise ValueError("requested iterations precede saved checkpoint")
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        if cfg.device == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        progress.message(f"Resumed after iteration {iteration}/{cfg.iterations}.")
    atomic_json(directory / "config.json", asdict(cfg))
    atomic_json(
        directory / "environment.json",
        {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": cfg.device,
            "generator_version": GENERATOR_VERSION,
            "source_sha256": implementation_hashes(),
        },
    )
    pool = (
        ProcessPoolExecutor(cfg.workers, mp_context=mp.get_context("spawn"))
        if cfg.workers > 1
        else None
    )
    try:
        if not history:
            probes = collect(
                model,
                cfg,
                list(range(20000, 20000 + cfg.validation_episodes)),
                "validation",
                pool,
                policy_only=True,
                progress=progress.callback("Initial diagnostic collection"),
            )
            initial_validation = validate(
                model, cfg, pool, progress=progress, prefix="Initial validation"
            )
            best = np.mean(
                [v["reward"] for k, v in initial_validation.items() if k.endswith("/search")]
            )
            history.append(
                {
                    "iteration": 0,
                    "validation": initial_validation,
                    "probe_reward_mse": probe_reward_error(model, probes, cfg.device),
                }
            )
            save_model(
                model,
                directory / "best.pt",
                {"iteration": 0, "validation": initial_validation, "config": asdict(cfg)},
            )
            progress.message(f"Initial checkpoint saved | validation reward={best:.4f}")
        for index in range(iteration, cfg.iterations):
            label = f"Iteration {index + 1}/{cfg.iterations}"
            progress.message(f"\n{label} | episodes seen={index * cfg.episodes}")
            start = perf_counter()
            model.eval()
            seeds = list(range(index * cfg.episodes, (index + 1) * cfg.episodes))
            episodes = collect(
                model,
                cfg,
                seeds,
                "train",
                pool,
                explore=True,
                progress=progress.callback(f"{label}: collection steps"),
            )
            replay.extend(episodes)
            collection_seconds = perf_counter() - start
            model.train()
            torch.set_num_threads(cfg.threads)
            losses = []
            progress.update(f"{label}: learner updates", 0, cfg.updates)
            for update in range(cfg.updates):
                batch = replay.sample(cfg.batch_size, rng)
                loss, terms = batch_loss(model, target, batch, cfg, rng, cfg.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for slow, fast in zip(target.parameters(), model.parameters(), strict=True):
                        slow.lerp_(fast, 1 - cfg.ema)
                losses.append(terms)
                progress.update(
                    f"{label}: learner updates",
                    update + 1,
                    cfg.updates,
                    " ".join(f"{key}={value:.4f}" for key, value in terms.items()),
                )
            row = {
                "iteration": index + 1,
                "episodes_seen": (index + 1) * cfg.episodes,
                "replay_episodes": len(replay.episodes),
                "collection_seconds": collection_seconds,
                "loss": {key: float(np.mean([x[key] for x in losses])) for key in losses[0]},
                "collection_reward": float(np.mean([float(e["rewards"].mean()) for e in episodes])),
            }
            if (index + 1) % cfg.validation_every == 0 or index + 1 == cfg.iterations:
                model.eval()
                row["validation"] = validate(
                    model, cfg, pool, progress=progress, prefix=f"{label}: validation"
                )
                score = np.mean(
                    [v["reward"] for k, v in row["validation"].items() if k.endswith("/search")]
                )
                if score > best:
                    best = float(score)
                    save_model(
                        model,
                        directory / "best.pt",
                        {
                            "iteration": index + 1,
                            "validation": row["validation"],
                            "config": asdict(cfg),
                        },
                    )
                    progress.message(f"New best checkpoint saved | validation reward={best:.4f}")
            row["iteration_seconds"] = perf_counter() - start
            row["probe_reward_mse"] = probe_reward_error(model, probes, cfg.device)
            history.append(row)
            progress.message(f"{label}: saving model, replay and optimizer...")
            save_model(
                model, directory / "latest.pt", {"iteration": index + 1, "config": asdict(cfg)}
            )
            save_training(
                checkpoint,
                {
                    "version": VERSION,
                    "generator": GENERATOR_VERSION,
                    "config": asdict(cfg),
                    "model": model.state_dict(),
                    "target": target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "replay": replay.episodes,
                    "probes": probes,
                    "iteration": index + 1,
                    "best": float(best),
                    "history": history,
                    "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else [],
                },
            )
            atomic_json(
                directory / "progress.json",
                {"history": history, "best_validation_reward": float(best)},
            )
            progress.message(
                f"{label} complete | replay={len(replay.episodes)} episodes | "
                f"reward={row['collection_reward']:.4f} | "
                f"probe MSE={row['probe_reward_mse']:.4f} | checkpoint saved"
            )
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    progress.message(f"Training complete. Checkpoints and metrics: {directory}")
    return history
