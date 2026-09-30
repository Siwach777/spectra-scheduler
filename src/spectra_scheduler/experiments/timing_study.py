"""CUDA timing-belief training and paired synthetic scheduling assessment."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..policy_benchmark import PolicySpec, _run_jobs, benchmark_synthetic
from ..scenarios import REQUIREMENT_SCENARIOS, build_requirement_scenario
from ..schedulers import (
    AdaptiveDwellScheduler,
    DwellSweepScheduler,
    PeriodAwareScheduler,
    RoundRobinScheduler,
    ShuffledSweepScheduler,
)
from ..simulation import SimulationEpisode, SyntheticAction, configure_scheduler
from ..timing_belief import (
    BeliefConfig,
    BeliefPolicyConfig,
    TimingBeliefNetwork,
    TimingBeliefPolicy,
    TimingHistory,
    belief_loss,
    load_belief,
    save_belief,
)
from .storage import atomic_file, fingerprint, load_torch, run_lock, save_torch, write_json

VERSION = 1


def source_hashes():
    package = Path(__file__).resolve().parent.parent
    names = (
        "timing_belief.py", "experiments/timing_study.py", "simulation.py", "receiver.py",
        "scenarios.py", "emitters.py", "synthetic_evaluation.py", "evaluation_contract.py",
    )
    return {name: fingerprint(package / name) for name in names}


class ObservedRateScheduler:
    """Causal count-rate control with the same variable dwells and coverage rule."""

    def __init__(self, config=None, *, random=False, seed=0):
        self.config, self.random, self.seed = config or BeliefPolicyConfig(), random, seed

    def set_retune_table(self, table):
        self.retune = np.asarray(table, np.int32)

    def reset(self, bands):
        self.history = TimingHistory(bands, 1)
        self.rng = np.random.default_rng(self.seed)
        self.bands = bands

    def choose_action(self, step):
        cfg = self.config
        if self.random:
            return SyntheticAction(int(self.rng.integers(self.bands)),
                                   int(self.rng.choice((4, 8, 16, 32))))
        age = np.where(self.history.last_listen >= 0,
                       step - self.history.last_listen, step + cfg.revisit)
        if age.max() >= cfg.revisit:
            return SyntheticAction(int(age.argmax()), cfg.probe)
        rate = (self.history.hits + 0.2) / (self.history.visits + 4)
        previous = self.history.current_band
        delay = self.retune[previous] if previous >= 0 else np.zeros(self.bands)
        score = rate - cfg.retune_cost * delay / (delay + max(cfg.dwells))
        if previous >= 0:
            score[previous] += cfg.switch_margin * score.max()
        return SyntheticAction(int(score.argmax()), max(cfg.dwells))

    def observe(self, observation):
        self.history.observe(observation)


def _collect_world(job):
    index, scenario, seed, config, directory, checkpoint = job
    world_seed = int.from_bytes(hashlib.sha256(
        f"timing-world-v{VERSION}:train:{scenario}:{seed}".encode()
    ).digest()[:4], "little")
    world = build_requirement_scenario(scenario, world_seed)
    episode = SimulationEpisode(world)
    # Training-only counterfactual true captures for every band and physical tick.
    future_truth = np.zeros((world.num_bands, world.duration + config.future), np.uint8)
    for (step, band), events in episode.events.items():
        future_truth[band, step] = len(world.receiver.listen(step, band, events).detected_emitters)
    if checkpoint is not None:
        raise ValueError("GPU on-policy collection is performed by collect_on_policy")
    kind = index % 5
    if kind == 0:
        behavior = ObservedRateScheduler(random=True, seed=world_seed)
    elif kind == 1:
        behavior = ObservedRateScheduler(seed=world_seed)
    elif kind == 2:
        behavior = DwellSweepScheduler(dwell_steps=8)
    elif kind == 3:
        behavior = DwellSweepScheduler(dwell_steps=32)
    else:
        behavior = AdaptiveDwellScheduler(
            minimum_dwell_steps=4, hit_extension_steps=4, maximum_dwell_steps=32
        )
    configure_scheduler(world, behavior)
    history = TimingHistory(world.num_bands, config.history)
    # At most one sample per four ticks bounds per-world retention independently
    # of whether the behavior exposes tick or macro actions.
    capacity = world.duration // 4 + 2
    inputs = np.empty((capacity, world.num_bands, 3, config.history), np.float16)
    targets = np.empty((capacity, world.num_bands, config.future), np.uint8)
    validity = np.empty((capacity, config.future), np.bool_)
    count, last_sample = 0, -4
    while episode.time_step < world.duration:
        step = episode.time_step
        if step - last_sample >= 4:
            inputs[count] = history.encode()
            targets[count] = future_truth[:, step:step + config.future]
            validity[count] = np.arange(config.future) < world.duration - step
            count += 1
            last_sample = step
        if hasattr(behavior, "choose_action"):
            observations = episode.step_action(behavior.choose_action(step))
        else:
            observations = (episode.step(behavior.choose_band(step)),)
        for observation in observations:
            # A chosen-action label must match the actual shared receiver engine.
            record = episode.records[observation.time_step]
            selected_label = future_truth[observation.band, observation.time_step]
            if observation.listening and selected_label != len(record.detected_emitters):
                raise ValueError("counterfactual capture label differs from receiver")
            behavior.observe(observation)
            history.observe(observation)
    path = directory / f"world-{index:05d}.npz"
    with atomic_file(path) as stream:
        np.savez_compressed(stream, history=inputs[:count], future=targets[:count],
                            valid=validity[:count])
    return {"path": str(path), "samples": count, "scenario": scenario,
            "seed": seed, "world_seed": world_seed, "behavior": type(behavior).__name__,
            "sha256": fingerprint(path)}


def collect_cache(directory, worlds, config, workers):
    directory.mkdir(parents=True, exist_ok=True)
    semantic = {"version": VERSION, "worlds_per_scenario": worlds,
                "config": asdict(config), "sources": source_hashes()}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        previous = manifest["semantic"]
        if any(previous[k] != semantic[k] for k in ("version", "worlds_per_scenario", "config")):
            raise ValueError("timing cache configuration changed")
        # Collection is independent of optimizer and policy-selection changes.
        # History/schema and receiver/world implementation must still match.
        for name in ("timing_belief.py", "simulation.py", "receiver.py", "scenarios.py",
                     "emitters.py"):
            if previous["sources"][name] != semantic["sources"][name]:
                raise ValueError(f"timing collection implementation changed: {name}")
        for name, digest in manifest["arrays"].items():
            if fingerprint(directory / f"{name}.npy") != digest:
                raise ValueError("timing cache contents changed")
        return manifest
    jobs = [(i * len(REQUIREMENT_SCENARIOS) + j, scenario, i, config, directory, None)
            for i in range(worlds) for j, scenario in enumerate(REQUIREMENT_SCENARIOS)]
    started = time.perf_counter()
    shards = _run_jobs(_collect_world, jobs, workers)
    total = sum(s["samples"] for s in shards)
    shapes = {"history": (total, 8, 3, config.history),
              "future": (total, 8, config.future), "valid": (total, config.future)}
    types = {"history": np.float16, "future": np.uint8, "valid": np.bool_}
    arrays = {k: np.lib.format.open_memmap(directory / f"{k}.npy", mode="w+",
                                         dtype=types[k], shape=v) for k, v in shapes.items()}
    cursor = 0
    for shard in shards:
        with np.load(shard["path"]) as chunk:
            stop = cursor + shard["samples"]
            for name, array in arrays.items():
                array[cursor:stop] = chunk[name]
            cursor = stop
    for array in arrays.values():
        array.flush()
    del arrays
    manifest = {"semantic": semantic, "samples": total, "shards": shards,
                "collection_seconds": time.perf_counter() - started,
                "arrays": {k: fingerprint(directory / f"{k}.npy") for k in shapes}}
    write_json(manifest_path, manifest)
    return manifest


class CacheStream:
    """Two bounded pinned buffers overlap memmap gathering and CUDA transfers."""

    def __init__(self, directory, batch_size):
        self.arrays = {k: np.load(directory / f"{k}.npy", mmap_mode="r")
                       for k in ("history", "future", "valid")}
        self.total = len(self.arrays["history"])
        self.batch_size = batch_size
        self.buffers = [{k: torch.empty((batch_size, *v.shape[1:]),
                                       dtype=torch.float32, pin_memory=True) for k, v in
                         self.arrays.items()} for _ in range(2)]
        self.events = [torch.cuda.Event(), torch.cuda.Event()]

    def batches(self, seed, epoch):
        order = np.random.default_rng(seed * 10000 + epoch).permutation(self.total)
        def fill(slot, start):
            stop = min(start + self.batch_size, self.total)
            for key, array in self.arrays.items():
                self.buffers[slot][key][:stop-start].numpy()[:] = array[order[start:stop]]
            return stop - start
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(fill, 0, 0)
            for index, start in enumerate(range(0, self.total, self.batch_size)):
                slot = index % 2
                size = pending.result()
                batch = {k: v[:size].to("cuda", non_blocking=True)
                         for k, v in self.buffers[slot].items()}
                self.events[slot].record()
                if start + self.batch_size < self.total:
                    other = (index + 1) % 2
                    if index > 0:
                        self.events[other].synchronize()
                    pending = pool.submit(fill, other, start + self.batch_size)
                yield batch


def controls(config):
    return [
        PolicySpec("observed-rate", partial(ObservedRateScheduler, config), str(asdict(config))),
        PolicySpec("sweep", RoundRobinScheduler, "round-robin"),
        PolicySpec("shuffled", ShuffledSweepScheduler, "shuffled-sweep"),
        PolicySpec("period-aware", PeriodAwareScheduler, "period-aware"),
        *[PolicySpec(f"dwell-{d}", partial(DwellSweepScheduler, dwell_steps=d), f"dwell:{d}")
          for d in (4, 8, 16, 32)],
        PolicySpec("adaptive-long", partial(AdaptiveDwellScheduler,
                   minimum_dwell_steps=4, hit_extension_steps=4, maximum_dwell_steps=32),
                   "adaptive:4:4:32"),
    ]


def assess(model, config, seeds, *, include_controls=False, source=None):
    policies = [PolicySpec("timing-belief", partial(TimingBeliefPolicy, model, config),
                           json.dumps({"config": asdict(config), "sources": source_hashes(),
                                       "checkpoint": source}, sort_keys=True))]
    if include_controls:
        policies += controls(config)
        baseline = "observed-rate"
    else:
        baseline = "timing-belief"
    started = time.perf_counter()
    report = benchmark_synthetic(policies, baseline=baseline, split="val", seeds=seeds, workers=1)
    report["evaluation_seconds"] = time.perf_counter() - started
    return report


def score(report):
    rows = [v["timing-belief"]["interception_ratio"]["mean"]
            for v in report["by_scenario"].values()]
    discovery = [v["timing-belief"]["discovery_fraction"]["mean"]
                 for v in report["by_scenario"].values()]
    return float(np.mean(rows) + 0.05 * np.mean(discovery))


def train_seed(args, manifest, seed, config, policy_config):
    torch.manual_seed(seed)
    run = args.run_dir / f"seed-{seed}"
    run.mkdir(parents=True, exist_ok=True)
    model = TimingBeliefNetwork(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4,
                                  fused=True)
    semantic = {"config": asdict(config), "policy": asdict(policy_config), "seed": seed,
                "manifest_sha256": fingerprint(args.cache_dir / "manifest.json"),
                "training_worlds": len(manifest["shards"]), "samples": manifest["samples"],
                "epochs": args.epochs, "sources": source_hashes(),
                "selection_seeds": list(range(2000, 2000 + args.selection_runs))}
    semantic["batch_size"] = args.batch_size
    semantic["workers"] = args.workers
    semantic["mixed_precision"] = "bfloat16 neural layers; float32 phase statistics and loss"
    start_epoch = 0
    rows, best = [], -np.inf
    if args.resume:
        model, old_metadata = load_belief(run / "latest.pt")
        state = load_torch(run / "optimizer.pt")
        for key in ("config", "policy", "seed", "manifest_sha256", "samples", "selection_seeds"):
            if json.loads(json.dumps(old_metadata[key])) != json.loads(json.dumps(semantic[key])):
                raise ValueError(f"resume semantics changed: {key}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=1e-4, fused=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        rows = state["history"]
        best = max((r.get("score", -np.inf) for r in rows), default=-np.inf)
        semantic["resumed_from"] = {
            "epoch": state["epoch"], "source_sha256": old_metadata["sources"],
            "checkpoint_sha256": fingerprint(run / "latest.pt"),
        }
    elif (run / "latest.pt").exists():
        raise ValueError("training run exists; use --resume or a new run directory")
    else:
        save_belief(run / "untrained.pt", model, semantic)
    stream = CacheStream(args.cache_dir, args.batch_size)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        if epoch:
            model.train()
            total = torch.zeros((), device="cuda")
            samples = 0
            for update, batch in enumerate(stream.batches(seed, epoch)):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = belief_loss(model, batch["history"], batch["future"], batch["valid"])
                torch._assert_async(torch.isfinite(loss), "nonfinite belief loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2, foreach=True)
                torch._assert_async(torch.isfinite(norm), "nonfinite belief gradient")
                optimizer.step()
                total += loss.detach() * len(batch["history"])
                samples += len(batch["history"])
                if update and update % 256 == 0:
                    print(json.dumps({"seed": seed, "epoch": epoch, "updates": update,
                                      "samples": samples,
                                      "samples_per_second": samples / (
                                          time.perf_counter() - started)}), flush=True)
            mean_loss = float(total / samples)
        else:
            mean_loss = None
        train_seconds = time.perf_counter() - started
        row = {"epoch": epoch, "loss": mean_loss, "training_seconds": train_seconds,
               "samples_per_second": manifest["samples"] / train_seconds if epoch else None,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
               "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if epoch % args.evaluate_every == 0 or epoch == args.epochs:
            model.eval()
            selection = assess(model, policy_config, tuple(semantic["selection_seeds"]))
            row["selection"] = {s: {k: v["timing-belief"][k]["mean"]
                                       for k in ("interception_ratio", "discovery_fraction")}
                                for s, v in selection["by_scenario"].items()}
            value = score(selection)
            row["score"] = value
            if value > best:
                best = value
                save_belief(run / "best.pt", model, {**semantic, "epoch": epoch, "score": best})
                write_json(run / "selection.json", selection)
        rows.append(row)
        save_belief(run / "latest.pt", model, {**semantic, "epoch": epoch})
        save_torch(run / "optimizer.pt", {"optimizer": optimizer.state_dict(), "epoch": epoch,
                                          "semantic": semantic, "history": rows})
        write_json(run / "progress.json", {"semantic": semantic, "best_score": best,
                                           "history": rows})
        print(json.dumps({"seed": seed, **row}), flush=True)
    return run / "best.pt"


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=800, help="training worlds per scenario")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--selection-runs", type=int, default=12)
    parser.add_argument("--report-runs", type=int, default=40)
    parser.add_argument("--evaluate-every", type=int, default=2)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revisit", type=int, default=96)
    parser.add_argument("--exploration", type=float, default=0.02)
    args = parser.parse_args(arguments)
    if min(args.worlds, args.workers, args.epochs, args.batch_size,
           args.selection_runs, args.report_runs, args.evaluate_every, args.revisit) < 1:
        parser.error("budgets must be positive")
    # Actual CUDA allocation fails explicitly; there is no CPU training fallback.
    torch.set_num_threads(2)
    torch.empty(1, device="cuda")
    config = BeliefConfig()
    policy_config = BeliefPolicyConfig(revisit=args.revisit, exploration=args.exploration)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.run_dir):
        if args.evaluate_only:
            paths = [args.run_dir / f"seed-{s}" / "best.pt" for s in args.seeds]
        else:
            manifest = collect_cache(args.cache_dir, args.worlds, config, args.workers)
            print(json.dumps({"collected_samples": manifest["samples"],
                              "worlds": len(manifest["shards"])}), flush=True)
            paths = [train_seed(args, manifest, seed, config, policy_config) for seed in args.seeds]
        for seed, path in zip(args.seeds, paths, strict=True):
            model, metadata = load_belief(path)
            report = assess(model, policy_config, tuple(range(10000, 10000 + args.report_runs)),
                            include_controls=True, source=fingerprint(path))
            report["model_metadata"] = metadata
            report["scope"] = "held-out development worlds; final test unused"
            write_json(args.run_dir / f"comparison-{seed}.json", report)
            print(json.dumps({"seed": seed, "report": {s: {p: m["interception_ratio"]["mean"]
                  for p, m in v.items()} for s, v in report["by_scenario"].items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
