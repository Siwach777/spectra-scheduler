"""Joint timing training with dense supervision on student-generated scan states."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..joint_belief import JointTimingNetwork, load_joint, save_joint
from ..scenarios import REQUIREMENT_SCENARIOS, build_scenario
from ..simulation import SimulationEpisode, SyntheticAction
from ..timing_belief import BeliefPolicyConfig, belief_loss, load_belief
from .calibrated_timing import CalibratedBeliefPolicy
from .storage import fingerprint, load_torch, run_lock, save_torch, write_json
from .timing_report import BatchedEpisode, batched_neural_results
from .timing_study import CacheStream


def world_and_targets(job):
    scenario, seed, future = job
    key = f"joint-timing-v1:train:{scenario}:{seed}".encode()
    world_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    world = build_scenario(scenario, world_seed)
    episode = SimulationEpisode(world)
    target = np.zeros((world.num_bands, world.duration + future), np.uint8)
    for (step, band), events in episode.events.items():
        target[band, step] = len(world.receiver.listen(step, band, events).detected_emitters)
    return world, target


def copy_valid_rows(destination, source, count, block_size=1024):
    """Trim preallocated collection storage without reading its unused tail."""
    if count != len(destination) or count > len(source) or block_size < 1:
        raise ValueError("invalid collection copy extent")
    for offset in range(0, count, block_size):
        stop = min(count, offset + block_size)
        destination[offset:stop] = source[offset:stop]


@torch.inference_mode()
def collect_student(model, policy, directory, worlds, pool, batch_size=20, *,
                    scheduler_factory=CalibratedBeliefPolicy, seed_offset=0):
    """Bounded world batches, CUDA acting, pre-action inputs and receiver labels."""
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(s, seed, model.config.future) for seed in range(seed_offset, seed_offset + worlds)
            for s in REQUIREMENT_SCENARIOS]
    capacity = len(jobs) * 128
    shapes = {
        "history": (capacity, 8, 3, model.config.history),
        "future": (capacity, 8, model.config.future),
        "valid": (capacity, model.config.future),
    }
    dtypes = {"history": np.float16, "future": np.uint8, "valid": np.bool_}
    # Partial arrays are replaced with their exact-size final memmaps after collection.
    arrays = {
        k: np.lib.format.open_memmap(
            directory / f"{k}.partial.npy", mode="w+", dtype=dtypes[k], shape=shape
        )
        for k, shape in shapes.items()
    }
    host = torch.empty((batch_size, 8, 3, model.config.history), pin_memory=True)
    device = torch.empty_like(host, device="cuda")
    host_array = host.numpy()
    cursor = 0
    started = time.perf_counter()
    for offset in range(0, len(jobs), batch_size):
        chunk = list(pool.map(world_and_targets, jobs[offset : offset + batch_size], chunksize=1))
        active = [
            (BatchedEpisode(world, scheduler_factory(model, policy)), target, -4)
            for world, target in chunk
        ]
        workspace = None
        while active:
            for i, (state, target, last_sample) in enumerate(active):
                step = state.episode.time_step
                host_array[i] = state.scheduler.history.encode()
                if step - last_sample >= 4:
                    if cursor >= capacity:
                        raise MemoryError("student cache sample bound exceeded")
                    arrays["history"][cursor] = host_array[i]
                    arrays["future"][cursor] = target[:, step : step + model.config.future]
                    arrays["valid"][cursor] = (
                        np.arange(model.config.future) < state.simulation.duration - step
                    )
                    cursor += 1
                    active[i] = (state, target, step)
            device[: len(active)].copy_(host[: len(active)], non_blocking=True)
            predictions = model(device[: len(active)]).cpu().numpy()
            planned = None
            if hasattr(active[0][0].scheduler, "coverage_action"):
                from ..timing_planner import ForecastPlannerWorkspace, first_actions
                if workspace is None or workspace.batch_size != len(active):
                    workspace = ForecastPlannerWorkspace(
                        len(active), predictions.shape[1], predictions.shape[2], policy.dwells
                    )
                bands, dwells = first_actions(
                    predictions,
                    [s.scheduler.history.current_band for s, _, _ in active],
                    np.stack([s.scheduler.retune for s, _, _ in active]),
                    [s.simulation.duration - s.episode.time_step for s, _, _ in active],
                    dwells=policy.dwells,
                    workspace=workspace,
                )
                planned = [SyntheticAction(int(b), int(d))
                           for b, d in zip(bands, dwells, strict=True)]
            for i, ((state, target, _), predicted) in enumerate(
                zip(active, predictions, strict=True)
            ):
                start = state.episode.time_step
                if planned is None:
                    action = state.scheduler.select(start, predicted)
                else:
                    action = state.scheduler.coverage_action(start) or planned[i]
                    state.scheduler.accept_action(start, predicted, action)
                state.advance(action, state.scheduler.forecast(start, action))
                for record in state.episode.records[start : state.episode.time_step]:
                    obs = record.observation
                    if obs.listening and target[obs.band, obs.time_step] != len(
                        record.detected_emitters
                    ):
                        raise ValueError("student supervision differs from the actual receiver")
            active = [
                (state, target, last)
                for state, target, last in active
                if state.episode.time_step < state.simulation.duration
            ]
        if offset % 120 == 0:
            print(
                json.dumps(
                    {
                        "collection_worlds": offset + len(chunk),
                        "samples": cursor,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    for name, array in arrays.items():
        final = np.lib.format.open_memmap(
            directory / f"{name}.npy",
            mode="w+",
            dtype=dtypes[name],
            shape=(cursor, *array.shape[1:]),
        )
        copy_valid_rows(final, array, cursor)
        final.flush()
        del final
    arrays.clear()
    for name in shapes:
        (directory / f"{name}.partial.npy").unlink()
    manifest = {
        "samples": cursor,
        "worlds": len(jobs),
        "namespace": "joint-timing-v1:train",
        "seed_offset": seed_offset,
        "scheduler": scheduler_factory.__name__,
        "policy": asdict(policy),
        "collection_seconds": time.perf_counter() - started,
        "arrays": {k: fingerprint(directory / f"{k}.npy") for k in shapes},
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


@torch.inference_mode()
def cache_anchor(base, directory, destination, batch_size):
    """Reusable frozen-base predictions: bounded CUDA batches and disk output."""
    source = np.load(directory / "history.npy", mmap_mode="r")
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float16,
        shape=(len(source), source.shape[1], base.config.future),
    )
    host = torch.empty((batch_size, *source.shape[1:]), pin_memory=True)
    for offset in range(0, len(source), batch_size):
        size = min(batch_size, len(source) - offset)
        host[:size].numpy()[:] = source[offset : offset + size]
        array[offset : offset + size] = (
            base(host[:size].to("cuda", non_blocking=True)).cpu().numpy()
        )
    array.flush()


class JointStream(CacheStream):
    def __init__(self, directory, batch_size, anchor):
        super().__init__(directory, batch_size)
        self.arrays["anchor"] = np.load(anchor, mmap_mode="r")
        if len(self.arrays["anchor"]) != self.total:
            raise ValueError("anchor cache length mismatch")
        for buffer in self.buffers:
            buffer["anchor"] = torch.empty(
                (batch_size, *self.arrays["anchor"].shape[1:]), dtype=torch.float32, pin_memory=True
            )


class _CachedPrediction:
    def __init__(self, model, anchor):
        self.model, self.anchor = model, anchor

    def __call__(self, history):
        return self.model(history, self.anchor)


def selection(model, config, seeds, batch_size):
    jobs = [(s, seed) for s in REQUIREMENT_SCENARIOS for seed in seeds]
    rows, _ = batched_neural_results(model, config, jobs, batch_size)
    means = {}
    for scenario in REQUIREMENT_SCENARIOS:
        values = [r["value"] for r in rows if r["scenario"] == scenario]
        capture = [v["evaluation"]["interception_ratio"] for v in values]
        # Unsupported empty-truth worlds are excluded from capture means consistently.
        capture = [v for v in capture if v is not None]
        means[scenario] = {
            "capture": float(np.mean(capture)),
            "discovery": float(np.mean([v["discovery_fraction"] for v in values])),
        }
    score = np.mean([m["capture"] + 0.15 * m["discovery"] for m in means.values()])
    return float(score), means


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--student-epochs", type=int, default=4)
    parser.add_argument("--student-worlds", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--selection-runs", type=int, default=12)
    parser.add_argument("--report-seed", type=int, default=18000)
    parser.add_argument("--report-runs", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(arguments)
    if (
        args.batch_size < 256
        or min(args.workers, args.epochs, args.student_epochs, args.student_worlds) < 1
    ):
        parser.error("batch >= 256 and positive worker, epoch and world counts required")
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    config = BeliefPolicyConfig(dwells=(1, 10, 50), revisit=256, probe=10, exploration=0.02)
    seeds = list(range(2000, 2000 + args.selection_runs))
    reporting = list(range(args.report_seed, args.report_seed + args.report_runs))
    if set(seeds) & set(reporting):
        raise ValueError("selection and reporting overlap")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    # Retain original manifest. Check world/receiver implementations and array content;
    # timing_belief.py gained validation guards after this historical cache was made.
    package = Path(__file__).resolve().parent.parent
    for name in ("simulation.py", "receiver.py", "scenarios.py", "emitters.py"):
        if fingerprint(package / name) != manifest["semantic"]["sources"][name]:
            raise ValueError(f"historical data source changed: {name}")
    for name, digest in manifest["arrays"].items():
        if fingerprint(args.cache_dir / f"{name}.npy") != digest:
            raise ValueError("historical data changed")
    with run_lock(args.run_dir):
        if (args.run_dir / "config.json").exists() and not args.resume:
            raise ValueError("use a fresh joint training directory")
        semantic = {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "policy": asdict(config),
            "selection_seeds": seeds,
            "reporting_seeds": reporting,
            "selection_objective": "equal-scenario capture + 0.15 * discovery",
            "base_sha256": fingerprint(args.checkpoint),
            "historical_manifest_sha256": fingerprint(args.cache_dir / "manifest.json"),
            "sources": {
                name: fingerprint(package / name)
                for name in ("joint_belief.py", "experiments/joint_study.py")
            },
        }
        if args.resume:
            previous = json.loads((args.run_dir / "config.json").read_text())
            for key in (
                "policy",
                "selection_seeds",
                "reporting_seeds",
                "base_sha256",
                "historical_manifest_sha256",
            ):
                if previous[key] != json.loads(json.dumps(semantic[key])):
                    raise ValueError(f"resume semantics changed: {key}")
            for key, value in previous["arguments"].items():
                if key != "resume" and semantic["arguments"][key] != value:
                    raise ValueError(f"resume arguments changed: {key}")
            write_json(args.run_dir / "resume-implementation.json", semantic["sources"])
            semantic = previous
        else:
            write_json(args.run_dir / "config.json", semantic)
        base, _ = load_belief(args.checkpoint)
        anchor_path = args.run_dir / "anchor-prediction.npy"
        started = time.perf_counter()
        if not args.resume:
            cache_anchor(base, args.cache_dir, anchor_path, args.batch_size)
        print(
            json.dumps(
                {
                    "anchor_cache_seconds": time.perf_counter() - started,
                    "samples": manifest["samples"],
                }
            ),
            flush=True,
        )
        anchor_stream = JointStream(args.cache_dir, args.batch_size, anchor_path)
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for seed in args.seeds:
                torch.manual_seed(seed)
                model = JointTimingNetwork(base).cuda()
                optimizer = torch.optim.AdamW(
                    (p for p in model.parameters() if p.requires_grad),
                    lr=0.0003,
                    weight_decay=0.0001,
                    fused=True,
                )
                run = args.run_dir / f"seed-{seed}"
                run.mkdir(exist_ok=args.resume)
                best, history = -np.inf, []
                student_stream = None
                start_epoch = 0
                if args.resume and (run / "progress.json").exists():
                    history = json.loads((run / "progress.json").read_text())
                    last_epoch = history[-1]["epoch"]
                    if last_epoch == args.epochs + args.student_epochs:
                        continue
                    model, latest = load_joint(run / "latest.pt")
                    if latest["epoch"] != last_epoch:
                        raise ValueError("resume checkpoint/progress disagree")
                    optimizer = torch.optim.AdamW(
                        (p for p in model.parameters() if p.requires_grad),
                        lr=0.00015 if last_epoch > args.epochs else 0.0003,
                        weight_decay=0.0001,
                        fused=True,
                    )
                    if (run / "optimizer.pt").exists():
                        state = load_torch(run / "optimizer.pt")
                        if state["epoch"] != last_epoch:
                            raise ValueError("resume optimizer/progress disagree")
                        optimizer.load_state_dict(state["optimizer"])
                    elif last_epoch != args.epochs:
                        raise ValueError("optimizer missing outside stage boundary")
                    best = max(row["score"] for row in history)
                    start_epoch = last_epoch + 1
                    if last_epoch > args.epochs:
                        student_stream = JointStream(
                            run / "student-cache", args.batch_size, run / "student-anchor.npy"
                        )
                for epoch in range(start_epoch, args.epochs + args.student_epochs + 1):
                    started = time.perf_counter()
                    total, samples = torch.zeros((), device="cuda"), 0
                    if epoch == args.epochs + 1:
                        model, _ = load_joint(run / "best.pt")
                        optimizer = torch.optim.AdamW(
                            (p for p in model.parameters() if p.requires_grad),
                            lr=0.00015,
                            weight_decay=0.0001,
                            fused=True,
                        )
                        collection_source = fingerprint(run / "best.pt")
                        student_directory = run / "student-cache"
                        student_manifest = collect_student(
                            model,
                            config,
                            student_directory,
                            args.student_worlds,
                            pool,
                            args.workers,
                        )
                        student_manifest["collection_checkpoint_sha256"] = collection_source
                        write_json(student_directory / "manifest.json", student_manifest)
                        student_anchor = run / "student-anchor.npy"
                        cache_anchor(base, student_directory, student_anchor, args.batch_size)
                        student_stream = JointStream(
                            student_directory, args.batch_size, student_anchor
                        )
                    if epoch:
                        model.train()
                        # Anchor states remain present after aggregation to prevent forgetting.
                        streams = (
                            [anchor_stream]
                            if student_stream is None
                            else [anchor_stream, student_stream]
                        )
                        for stream in streams:
                            for update, batch in enumerate(stream.batches(seed, epoch)):
                                optimizer.zero_grad(set_to_none=True)
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    loss = belief_loss(
                                        _CachedPrediction(model, batch["anchor"]),
                                        batch["history"],
                                        batch["future"],
                                        batch["valid"],
                                    )
                                torch._assert_async(torch.isfinite(loss), "nonfinite joint loss")
                                loss.backward()
                                norm = torch.nn.utils.clip_grad_norm_(
                                    (p for p in model.parameters() if p.requires_grad), 2
                                )
                                torch._assert_async(
                                    torch.isfinite(norm), "nonfinite joint gradient"
                                )
                                optimizer.step()
                                total += loss.detach() * len(batch["history"])
                                samples += len(batch["history"])
                                if update and update % 256 == 0:
                                    print(
                                        json.dumps(
                                            {
                                                "seed": seed,
                                                "epoch": epoch,
                                                "samples": samples,
                                                "seconds": time.perf_counter() - started,
                                            }
                                        ),
                                        flush=True,
                                    )
                    training_seconds = time.perf_counter() - started
                    model.eval()
                    score, means = selection(model, config, seeds, args.workers)
                    row = {
                        "seed": seed,
                        "epoch": epoch,
                        "phase": "student" if student_stream else "anchor",
                        "loss": float(total / samples) if samples else None,
                        "samples": samples,
                        "training_seconds": training_seconds,
                        "score": score,
                        "selection": means,
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    }
                    history.append(row)
                    metadata = {**semantic, "seed": seed, "epoch": epoch, "score": score}
                    if epoch == 0:
                        save_joint(run / "untrained.pt", model, metadata)
                    if score > best:
                        best = score
                        save_joint(run / "best.pt", model, metadata)
                    save_joint(run / "latest.pt", model, metadata)
                    save_torch(
                        run / "optimizer.pt", {"epoch": epoch, "optimizer": optimizer.state_dict()}
                    )
                    write_json(run / "progress.json", history)
                    print(json.dumps(row), flush=True)
                write_json(
                    run / "frozen.json",
                    {"checkpoint_sha256": fingerprint(run / "best.pt"), "selection_score": best},
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
