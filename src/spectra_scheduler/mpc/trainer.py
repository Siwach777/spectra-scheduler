"""Resumable actor/learner orchestration and validation-based checkpoint selection."""

from __future__ import annotations

import copy
import fcntl
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from spectra_scheduler.rl_scenarios import GENERATOR_VERSION, PHYSICAL_GENERATOR_VERSION

from .checkpoints import atomic_json, implementation_hashes, load_model, save_model, save_training
from .config import STEP_FEATURE_DIM, VERSION
from .data import Replay, collect
from .evaluation import episode_mean, probe_reward_error, selection_score, validate
from .learning import BatchWorkspace, batch_loss
from .model import NeuralMPCModel
from .progress import LiveProgress
from .reanalyse import refresh_replay_targets


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
    source_hashes = implementation_hashes()
    if resume and initial:
        raise ValueError("resume and initialization are mutually exclusive")
    if not resume and (checkpoint.exists() or (directory / "config.json").exists()):
        raise FileExistsError("existing run: use --resume or a new directory")
    if cfg.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MPC training requires CUDA; expose the host GPU")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = NeuralMPCModel(
        step_dim=STEP_FEATURE_DIM + int(cfg.physical_elapsed_feature),
        dwell_steps=cfg.dwell_steps,
        physical_contract=cfg.physical_contract,
    )
    if initial:
        source = load_model(initial)
        incompatible = model.load_state_dict(source.state_dict(), strict=False)
        if incompatible.unexpected_keys or any(
            not k.startswith("_observation.") for k in incompatible.missing_keys
        ):
            raise ValueError("initial model architecture is incompatible")
    model = model.to(cfg.device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4, fused=True)
    replay = Replay(cfg.replay_capacity)
    iteration, best, history = 0, -float("inf"), []
    probes = None
    if resume:
        progress.message("Loading model, optimizer and replay checkpoint...")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        recorded = saved.get("source_sha256")
        if recorded is None:
            environment = directory / "environment.json"
            if not environment.is_file():
                raise ValueError("resume requires original source provenance")
            recorded = json.loads(environment.read_text()).get("source_sha256")
        if recorded != source_hashes:
            raise ValueError("MPC implementation changed since checkpoint; start a new run")
        if saved["version"] != VERSION or saved["generator"] != GENERATOR_VERSION:
            raise ValueError(
                "incompatible training checkpoint: use a new run directory; "
                "--initial best.pt can warm-start shared model weights"
            )
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
    model.representation.gru.flatten_parameters()
    target.representation.gru.flatten_parameters()
    workspace = BatchWorkspace()
    atomic_json(directory / "config.json", asdict(cfg))
    atomic_json(
        directory / "environment.json",
        {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": cfg.device,
            "generator_version": (
                PHYSICAL_GENERATOR_VERSION if cfg.physical_contract else GENERATOR_VERSION
            ),
            "generator": "physical" if cfg.physical_contract else "legacy-procedural",
            "source_sha256": source_hashes,
        },
    )
    pool = (
        ProcessPoolExecutor(cfg.workers, mp_context=mp.get_context("spawn"))
        if cfg.workers > 1 and len(cfg.dwell_steps) == 1
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
            best = selection_score(initial_validation)
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
            save_model(
                model,
                directory / "untrained.pt",
                {"iteration": 0, "validation": initial_validation, "config": asdict(cfg)},
            )
            progress.message(f"Initial checkpoint saved | capture/discovery score={best:.4f}")
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
                shifted=cfg.mixed_receivers and index % 4 == 3,
                progress=progress.callback(f"{label}: collection steps"),
            )
            replay.extend(episodes)
            refreshed = refresh_replay_targets(target, replay, cfg, rng, cfg.reanalyse_episodes)
            collection_seconds = perf_counter() - start
            model.train()
            torch.set_num_threads(cfg.threads)
            losses = []
            progress.update(f"{label}: learner updates", 0, cfg.updates)
            for update in range(cfg.updates):
                batch = replay.sample(cfg.batch_size, rng)
                loss, terms = batch_loss(model, target, batch, cfg, rng, cfg.device, workspace)
                torch._assert_async(torch.isfinite(loss), "nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, foreach=True)
                torch._assert_async(torch.isfinite(norm), "nonfinite gradient norm")
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
                "collection_reward": float(np.mean([episode_mean(e, "rewards") for e in episodes])),
                "listening_fraction": float(
                    np.mean([episode_mean(e, "features", 9) for e in episodes])
                ),
                "hit_fraction": float(np.mean([episode_mean(e, "features", 8) for e in episodes])),
                "physical_steps": sum(
                    int(e["elapsed"].sum()) if "elapsed" in e else len(e["actions"])
                    for e in episodes
                ),
                "decisions": sum(len(e["actions"]) for e in episodes),
                "reanalysis": refreshed,
            }
            if (index + 1) % cfg.validation_every == 0 or index + 1 == cfg.iterations:
                model.eval()
                row["validation"] = validate(
                    model, cfg, pool, progress=progress, prefix=f"{label}: validation"
                )
                score = selection_score(row["validation"])
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
                    progress.message(
                        f"New best checkpoint saved | capture/discovery score={best:.4f}"
                    )
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
                    "generator": (
                        PHYSICAL_GENERATOR_VERSION if cfg.physical_contract else GENERATOR_VERSION
                    ),
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
                    "source_sha256": source_hashes,
                },
            )
            atomic_json(
                directory / "progress.json",
                {"history": history, "best_validation_score": float(best)},
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
