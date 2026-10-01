"""Warm-start the timing forecaster on causal, training-only external PDW histories."""

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..dataset_io import iter_pulses
from ..policy_benchmark import PolicySpec, _replay_job, _run_jobs, make_plan, summarize, validate_plan
from ..pulse_replay import ReplayConfig
from ..replay_baselines import RateProbePolicy
from ..replay_env import InterfaceConfig, ReplayEnv
from ..replay_evaluation import ReferencePolicy, evaluate_policy_batch
from ..timing_belief import BeliefPolicyConfig, TimingHistory, belief_loss, load_belief
from ..timing_replay import (
    PowerBlindTimingPredictor,
    TimingReplayPolicy,
    load_pdw_predictor,
    record_pdw_history,
    save_pdw_predictor,
)
from .storage import fingerprint, run_lock, write_json


def partition_plans(root, fit_files, development_files, seed, exclude_plan=None):
    excluded = json.loads(exclude_plan.read_text())["recordings"] if exclude_plan else []
    excluded_hashes = {r["sha256"] for r in excluded}
    plan = make_plan(root, "train", fit_files + development_files + len(excluded), (0,), seed)
    recordings = [r for r in plan["recordings"] if r["sha256"] not in excluded_hashes]
    if len({r["sha256"] for r in recordings}) != len(recordings):
        raise ValueError("duplicate recording content cannot cross training partitions")
    rng = np.random.default_rng(seed)
    rng.shuffle(recordings)
    if len(recordings) < fit_files + development_files:
        raise ValueError("not enough disjoint training recordings")
    return tuple({**plan, "partition": name, "recordings": recordings[start:stop]}
                 for name, start, stop in (
                     ("fit", 0, fit_files),
                     ("development", fit_files, fit_files + development_files)))


def arrival_counts(path, receiver, bands):
    """Evaluator/collector-only supervision; no labels or transmitter metadata used."""
    ticks = int((receiver.stop_us - receiver.start_us) / 1000)
    result = np.zeros((bands, ticks), np.float32)
    width = (receiver.max_frequency_mhz - receiver.min_frequency_mhz) / bands
    for batch in iter_pulses(path, receiver.batch_rows):
        toa, frequency = batch.features[:, 0], batch.features[:, 1]
        valid = ((toa >= receiver.start_us) & (toa < receiver.stop_us)
                 & (frequency >= receiver.min_frequency_mhz)
                 & (frequency < receiver.max_frequency_mhz))
        time = ((toa[valid] - receiver.start_us) / 1000).astype(np.int64)
        band = ((frequency[valid] - receiver.min_frequency_mhz) / width).astype(np.int64)
        result += np.bincount(band * ticks + time, minlength=bands * ticks).reshape(bands, ticks)
    return result * receiver.detection_probability


def collect_recording(job):
    index, path, output, config, receiver, samples, seed = job
    torch.set_num_threads(1)
    interface = InterfaceConfig(dwell_us=(1000, 10000, 50000))
    rng = np.random.default_rng(seed)
    history = TimingHistory(interface.bands, config.history, count_limit=None)
    histories = np.empty((samples, interface.bands, 3, config.history), np.float32)
    times = np.empty(samples, np.int64)
    teacher = (RateProbePolicy() if index % 2 == 0 else
               ReferencePolicy("sweep", 1) if index % 4 == 1 else
               ReferencePolicy("random", 2))
    seen = 0
    # Store pre-action histories first. Future targets are joined only after replay.
    with ReplayEnv(path, receiver, interface) as env:
        observation = env.reset()
        teacher.reset(env.specification(), receiver.seed)
        while True:
            slot = seen if seen < samples else int(rng.integers(seen + 1))
            if slot < samples:
                histories[slot] = history.encode()
                times[slot] = history.time
            seen += 1
            action = teacher.act(observation)
            transition = env.step(action)
            record_pdw_history(history, transition.receiver_observation,
                               action // len(interface.dwell_us), start_us=receiver.start_us)
            observation = transition.observation
            if transition.terminated:
                break
    count = min(samples, seen)
    truth = arrival_counts(path, receiver, interface.bands)
    padded = np.pad(truth, ((0, 0), (0, config.future)))
    offsets = times[:count, None] + np.arange(config.future)[None]
    future = padded[:, offsets].transpose(1, 0, 2)
    valid = (offsets < truth.shape[1]).astype(np.float32)
    prefix = output / f"recording-{index}"
    arrays = {"history": histories[:count], "future": future, "valid": valid}
    artifacts = {}
    for name, value in arrays.items():
        target = prefix.with_name(f"{prefix.name}-{name}.npy")
        np.save(target, value)
        artifacts[name] = {"path": str(target), "sha256": fingerprint(target)}
    return {"samples": count, "source_sha256": fingerprint(path), "arrays": artifacts,
            "teacher": type(teacher).__name__, "receiver_decisions": seen}


class ShardBatches:
    """Read memory-mapped shards into two bounded, reusable pinned staging buffers."""

    def __init__(self, shards, batch_size):
        self.maps = []
        for shard in shards:
            arrays = {}
            for name, item in shard["arrays"].items():
                if fingerprint(Path(item["path"])) != item["sha256"]:
                    raise ValueError("training shard changed")
                arrays[name] = np.load(item["path"], mmap_mode="r")
            self.maps.append(arrays)
        self.ends = np.cumsum([s["samples"] for s in shards])
        self.starts = np.r_[0, self.ends[:-1]]
        self.samples, self.batch_size = int(self.ends[-1]), batch_size
        example = self.maps[0]
        self.host = [{name: torch.empty((batch_size, *value.shape[1:]), pin_memory=True)
                      for name, value in example.items()} for _ in range(2)]
        self.device = {name: torch.empty_like(value, device="cuda")
                       for name, value in self.host[0].items()}
        self.events = [torch.cuda.Event(), torch.cuda.Event()]
        self.recorded = [False, False]
        self.iteration = 0

    def batch(self, indices):
        slot = self.iteration % 2
        if self.recorded[slot]:
            self.events[slot].synchronize()
        groups = np.searchsorted(self.ends, indices, side="right")
        staging = self.host[slot]
        for group in np.unique(groups):
            selected = np.flatnonzero(groups == group)
            rows = indices[selected] - self.starts[group]
            for name in staging:
                staging[name].numpy()[selected] = self.maps[group][name][rows]
        size = len(indices)
        for name, value in staging.items():
            self.device[name][:size].copy_(value[:size], non_blocking=True)
        self.events[slot].record()
        self.recorded[slot] = True
        self.iteration += 1
        return tuple(self.device[name][:size] for name in ("history", "future", "valid"))


@torch.inference_mode()
def assess(predictor, config, paths, receiver, batch_size, controls):
    predictor.eval()
    rows = [{**row, "policies": dict(row["policies"])} for row in controls]
    template = TimingReplayPolicy(predictor, config, batch_size, count_limit=None)
    learned = PolicySpec("timing-pdw", lambda: template, "PDW counts; missing-power gate bypass")
    interface = InterfaceConfig(dwell_us=(1000, 10000, 50000))
    for offset in range(0, len(paths), batch_size):
        reports = evaluate_policy_batch([(p, receiver) for p in paths[offset:offset + batch_size]],
                                        learned.factory, interface)
        for row, report in zip(rows[offset:offset + len(reports)], reports, strict=True):
            row["policies"][learned.name] = report
    specs = [PolicySpec(name, lambda: None, name) for name in ("sweep-50", "rate-probe")]
    return {"summary": summarize(rows, [*specs, learned], "rate-probe"), "results": rows}


def select_replay_settings(root, run_dir, batch_size):
    """Select cheaper dense-PDW probes on the existing train-only development files."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    training = json.loads((run_dir / "training.json").read_text())
    if (run_dir / "policy-selection.json").exists():
        raise ValueError("replay policy selection already exists")
    predictor, metadata = load_pdw_predictor(run_dir / "best.pt")
    if fingerprint(run_dir / "best.pt") != training["best_checkpoint_sha256"]:
        raise ValueError("selected training checkpoint changed")
    raw = dict(metadata["policy"])
    raw["dwells"] = tuple(raw["dwells"])
    config = BeliefPolicyConfig(**raw)
    paths = validate_plan(root, manifest["development"])
    controls = json.loads((run_dir / "controls.json").read_text())
    receiver = ReplayConfig(**manifest["receiver"])
    incumbent = json.loads((run_dir / f"evaluation-{metadata['epoch']}.json").read_text())
    candidates = {"incumbent": {"policy": asdict(config), "assessment": incumbent}}
    for revisit in (512, 1024):
        candidate = replace(config, probe=1, revisit=revisit)
        assessment = assess(predictor, candidate, paths, receiver, batch_size, controls)
        name = f"probe-1-revisit-{revisit}"
        candidates[name] = {"policy": asdict(candidate), "assessment": assessment}
        summary = assessment["summary"]["timing-pdw"]
        print(json.dumps({"candidate": name, "capture": summary["interception_ratio"]["mean"],
                          "discovery": summary["discovery_fraction"]["mean"]}), flush=True)
    eligible = [name for name, row in candidates.items() if row["assessment"]["summary"][
        "timing-pdw"]["discovery_fraction"]["mean"] >= training["discovery_floor"]]
    selected = max(eligible, key=lambda name: candidates[name]["assessment"]["summary"][
        "timing-pdw"]["interception_ratio"]["mean"])
    save_pdw_predictor(run_dir / "deployment.pt", predictor,
                       {**metadata, "policy": candidates[selected]["policy"]})
    validate_plan(root, manifest["development"])
    if fingerprint(run_dir / "best.pt") != training["best_checkpoint_sha256"]:
        raise ValueError("selected weights changed during policy selection")
    write_json(run_dir / "policy-selection.json", {"schema_version": 1,
        "selected": selected, "eligible": eligible,
        "rule": "maximize train-only development capture; preserve training discovery floor",
        "discovery_floor": training["discovery_floor"], "candidates": candidates,
        "deployment_checkpoint_sha256": fingerprint(run_dir / "deployment.pt"),
        "scope": "same disjoint train-split development recordings; no validation files"})
    print(json.dumps({"selected": selected, "eligible": eligible}), flush=True)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--exclude-plan", type=Path)
    parser.add_argument("--training-files", type=int, default=64)
    parser.add_argument("--development-files", type=int, default=16)
    parser.add_argument("--samples-per-file", type=int, default=256)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--evaluate-every", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--select-policy-only", action="store_true")
    args = parser.parse_args(arguments)
    if min(args.training_files, args.development_files, args.samples_per_file, args.workers,
           args.batch_size, args.evaluation_batch, args.epochs, args.evaluate_every) < 1:
        parser.error("positive sample, worker, batch and epoch settings required")
    if args.batch_size < 256 or not 0 < args.learning_rate < 1:
        parser.error("training batch must be at least 256 and learning rate in (0, 1)")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    if not args.prepare_only and not torch.cuda.is_available():
        raise RuntimeError("PDW forecaster training and neural validation require CUDA")
    started = perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    receiver = ReplayConfig(stop_us=10_000_000, retune_us=2000, detection_probability=0.9)
    interface = InterfaceConfig(dwell_us=(1000, 10000, 50000))
    with run_lock(args.run_dir):
        if args.select_policy_only:
            if args.prepare_only:
                parser.error("preparation and policy selection are separate operations")
            select_replay_settings(args.root, args.run_dir, args.evaluation_batch)
            return 0
        manifest_path = args.run_dir / "manifest.json"
        if (args.run_dir / "training.json").exists():
            raise ValueError("completed training run already exists")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["source_checkpoint_sha256"] != fingerprint(args.checkpoint):
                raise ValueError("prepared data belongs to a different source checkpoint")
        else:
            fit, development = partition_plans(args.root, args.training_files,
                args.development_files, args.seed, args.exclude_plan)
            from ..experiments.storage import load_torch
            from ..timing_belief import BeliefConfig

            model_config = BeliefConfig(**load_torch(args.checkpoint)["config"])
            cache = args.run_dir / "cache"
            cache.mkdir(parents=True, exist_ok=True)
            paths = validate_plan(args.root, fit)
            jobs = [(i, p, cache, model_config, receiver, args.samples_per_file, args.seed + i)
                    for i, p in enumerate(paths)]
            shards = _run_jobs(collect_recording, jobs, args.workers)
            validate_plan(args.root, fit)
            if [s["source_sha256"] for s in shards] != [r["sha256"] for r in fit["recordings"]]:
                raise ValueError("source recording changed during preparation")
            manifest = {"source_checkpoint_sha256": fingerprint(args.checkpoint),
                "fit": fit, "development": development, "shards": shards,
                "receiver": asdict(receiver), "interface": asdict(interface),
                "target": "all-band future arrival counts times public detection probability",
                "history": "delivered PDWs only; raw counts; explicit retune/listen mask; no dBm",
                "seed": args.seed}
            write_json(manifest_path, manifest)
        fit_hashes = {r["sha256"] for r in manifest["fit"]["recordings"]}
        development_hashes = {r["sha256"] for r in manifest["development"]["recordings"]}
        if fit_hashes & development_hashes:
            raise ValueError("fitting and development recording content overlaps")
        print(json.dumps({"phase": "prepared", "samples": sum(
            s["samples"] for s in manifest["shards"]), "elapsed_seconds": perf_counter() - started}),
            flush=True)
        if args.prepare_only:
            return 0
        validate_plan(args.root, manifest["fit"])
        paths = validate_plan(args.root, manifest["development"])
        model, metadata = load_belief(args.checkpoint)
        predictor = PowerBlindTimingPredictor(model)
        settings = dict(metadata["policy"])
        settings["dwells"] = tuple(settings["dwells"])
        config = BeliefPolicyConfig(**settings)
        controls = [PolicySpec("sweep-50", partial(ReferencePolicy, "sweep", 2), "fixed 50-ms sweep"),
                    PolicySpec("rate-probe", RateProbePolicy, "rate probe; revisit 500 ms")]
        controls = _run_jobs(_replay_job, [(i, p, 0, controls, receiver, interface)
                                        for i, p in enumerate(paths)], args.workers)
        write_json(args.run_dir / "controls.json", controls)
        batches = ShardBatches(manifest["shards"], args.batch_size)
        encoder = list(model.temporal.parameters())
        encoder_ids = {id(p) for p in encoder}
        head = [p for p in model.parameters() if id(p) not in encoder_ids]
        optimizer = torch.optim.AdamW([
            {"params": encoder, "lr": args.learning_rate * 0.2, "scale": 0.2},
            {"params": head, "lr": args.learning_rate, "scale": 1.0},
        ], weight_decay=1e-4)
        rng = np.random.default_rng(args.seed)
        steps_per_epoch = math.ceil(batches.samples / args.batch_size)
        total_steps = args.epochs * steps_per_epoch
        curves, best_capture, discovery_floor = [], -1.0, None
        base_metadata = {"policy": asdict(config),
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "trained_on": [r["sha256"] for r in manifest["fit"]["recordings"]],
            "development_on": [r["sha256"] for r in manifest["development"]["recordings"]],
            "modality": "uncalibrated PDWs; raw counts; no measured-power gate", "seed": args.seed}
        for epoch in range(args.epochs + 1):
            loss_sum, samples = torch.zeros((), device="cuda"), 0
            if epoch:
                predictor.train()
                permutation = rng.permutation(batches.samples)
                for offset in range(0, batches.samples, args.batch_size):
                    step = (epoch - 1) * steps_per_epoch + offset // args.batch_size
                    progress = step / max(total_steps - 1, 1)
                    schedule = min((step + 1) / 32, 1) * (0.25 + 0.75 * (1 + math.cos(
                        math.pi * progress)) / 2)
                    for group in optimizer.param_groups:
                        group["lr"] = args.learning_rate * group["scale"] * schedule
                    indices = permutation[offset:offset + args.batch_size]
                    history, future, valid = batches.batch(indices)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = belief_loss(predictor, history, future, valid)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    loss_sum += loss.detach() * len(indices)
                    samples += len(indices)
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("non-finite PDW training loss")
                print(json.dumps({"epoch": epoch, "loss": float(loss_sum) / samples,
                    "elapsed_seconds": perf_counter() - started,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated()}), flush=True)
            if epoch == 0 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
                assessment = assess(predictor, config, paths, receiver, args.evaluation_batch, controls)
                summary = assessment["summary"]["timing-pdw"]
                capture = summary["interception_ratio"]["mean"]
                discovery = summary["discovery_fraction"]["mean"]
                if discovery_floor is None:
                    discovery_floor = discovery - 0.01
                eligible = discovery >= discovery_floor
                selected = eligible and capture > best_capture
                checkpoint_metadata = {**base_metadata, "epoch": epoch}
                save_pdw_predictor(args.run_dir / f"epoch-{epoch}.pt", predictor, checkpoint_metadata)
                if selected:
                    best_capture = capture
                    save_pdw_predictor(args.run_dir / "best.pt", predictor, checkpoint_metadata)
                curves.append({"epoch": epoch, "loss": float(loss_sum) / samples if samples else None,
                    "capture": capture, "discovery": discovery, "eligible": eligible,
                    "selected": selected, "summary": assessment["summary"]})
                write_json(args.run_dir / f"evaluation-{epoch}.json", assessment)
                write_json(args.run_dir / "learning-curve.json", curves)
                print(json.dumps({"epoch": epoch, "capture": capture, "discovery": discovery,
                                  "selected": selected}), flush=True)
        validate_plan(args.root, manifest["fit"])
        validate_plan(args.root, manifest["development"])
        if fingerprint(args.checkpoint) != manifest["source_checkpoint_sha256"]:
            raise ValueError("source checkpoint changed during training")
        write_json(args.run_dir / "training.json", {
            "schema_version": 1, "manifest_sha256": fingerprint(manifest_path),
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "best_checkpoint_sha256": fingerprint(args.run_dir / "best.pt"),
            "settings": {"epochs": args.epochs, "batch_size": args.batch_size,
                "workers": args.workers, "learning_rate": args.learning_rate,
                "encoder_lr_factor": 0.2, "seed": args.seed},
            "selection_rule": "maximize development capture; discovery no more than 1 pp below epoch 0",
            "discovery_floor": discovery_floor, "learning_curve": curves,
            "elapsed_seconds": perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "scope": "fit and checkpoint selection use disjoint train-split recordings only",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
