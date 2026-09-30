"""Warm-start timing training on histories from causal receding-horizon planning."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import resource
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..scenarios import REQUIREMENT_SCENARIOS
from ..timing_belief import BeliefPolicyConfig, belief_loss, load_belief, save_belief
from ..timing_planner import CalibratedTimingPlannerPolicy
from .joint_study import JointStream, cache_anchor, collect_student
from .planner_study import batched_planned_results
from .storage import fingerprint, load_torch, run_lock, save_torch, write_json


class _Prediction:
    def __init__(self, prediction):
        self.prediction = prediction

    def __call__(self, history):
        return self.prediction


def refinement_loss(prediction, batch, window_weight=0.15, anchor_weight=0.03):
    """Proper count loss plus the native listening windows and frozen count anchor."""
    prediction = prediction.float().clamp_min(1e-6)
    loss = belief_loss(_Prediction(prediction), batch["history"], batch["future"], batch["valid"])
    pc = F.pad(prediction.cumsum(-1), (1, 0))
    tc = F.pad(batch["future"].cumsum(-1), (1, 0))
    windows = prediction.new_zeros(())
    count = 0
    for offset in (0, 1, 2, 4, 8):
        for dwell in (1, 10, 50):
            end = offset + dwell
            if end > prediction.shape[-1]:
                continue
            eligible = batch["valid"][:, end - 1, None]
            actual = tc[..., end] - tc[..., offset]
            fitted = pc[..., end] - pc[..., offset]
            item = F.smooth_l1_loss(torch.log1p(fitted), torch.log1p(actual), reduction="none")
            windows += (item * eligible).sum() / (eligible.sum() * prediction.shape[1]).clamp_min(1)
            count += 1
    anchor = batch["anchor"].float().clamp_min(1e-6)
    kl = anchor * (anchor.log() - prediction.log()) + prediction - anchor
    valid = batch["valid"][:, None].expand_as(kl)
    kl = (kl * valid).sum() / valid.sum().clamp_min(1)
    return loss + window_weight * windows / max(count, 1) + anchor_weight * kl


def assess(model, config, seeds, batch_size):
    jobs = [(scenario, seed) for scenario in REQUIREMENT_SCENARIOS for seed in seeds]
    rows, profile = batched_planned_results(model, config, jobs, batch_size)
    means = {}
    for scenario in REQUIREMENT_SCENARIOS:
        values = [r["value"] for r in rows if r["scenario"] == scenario]
        capture = [v["evaluation"]["interception_ratio"] for v in values
                   if v["evaluation"]["interception_ratio"] is not None]
        means[scenario] = {"capture": float(np.mean(capture)),
                           "discovery": float(np.mean([v["discovery_fraction"] for v in values]))}
    score = float(np.mean([v["capture"] + 0.15 * v["discovery"] for v in means.values()]))
    return score, means, profile


def optimizer_for(model, rate, encoder_factor):
    temporal = list(model.temporal.parameters())
    identities = {id(p) for p in temporal}
    other = [p for p in model.parameters() if id(p) not in identities]
    return torch.optim.AdamW([
        {"params": temporal, "lr": rate * encoder_factor, "factor": encoder_factor},
        {"params": other, "lr": rate, "factor": 1.0},
    ], weight_decay=0.0001, fused=True)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--historical-cache", type=Path, required=True)
    parser.add_argument("--planner-selection", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps-per-epoch", type=int, default=256)
    parser.add_argument("--worlds", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--encoder-factor", type=float, default=0.25)
    parser.add_argument("--selection-runs", type=int, default=32)
    parser.add_argument("--report-seed", type=int, default=24000)
    parser.add_argument("--report-runs", type=int, default=100)
    parser.add_argument("--evaluate-every", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(arguments)
    if args.batch_size < 256 or min(args.epochs, args.worlds, args.workers,
                                    args.steps_per_epoch, args.evaluate_every) < 1:
        parser.error("batch >= 256 and positive budgets required")
    if (not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)
            or not 0 < args.encoder_factor <= 1 or args.learning_rate <= 0
            or min(args.selection_runs, args.report_runs) < 1):
        parser.error("unique model seeds and an encoder factor in (0,1] required")
    torch.set_num_threads(1)
    torch.empty(1, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    selected = json.loads(args.planner_selection.read_text())
    if selected["checkpoint_sha256"] != fingerprint(args.checkpoint):
        raise ValueError("planner was selected using a different checkpoint")
    policy = selected["selected"]["policy"]
    policy["dwells"] = tuple(policy["dwells"])
    config = BeliefPolicyConfig(**policy)
    if not selected["selected"]["coverage"]:
        raise ValueError("this refinement study requires continued causal coverage")
    selection_seeds = list(range(2000, 2000 + args.selection_runs))
    reporting_seeds = list(range(args.report_seed, args.report_seed + args.report_runs))
    if set(selection_seeds) & set(reporting_seeds):
        raise ValueError("reporting overlaps selection")
    with run_lock(args.run_dir):
        historical = json.loads((args.historical_cache / "manifest.json").read_text())
        package = Path(__file__).resolve().parent.parent
        for name in ("simulation.py", "receiver.py", "scenarios.py", "emitters.py",
                     "synthetic_evaluation.py", "evaluation_contract.py"):
            if historical["semantic"]["sources"][name] != fingerprint(package / name):
                raise ValueError(f"historical receiver/world implementation changed: {name}")
        for name, digest in historical["arrays"].items():
            if fingerprint(args.historical_cache / f"{name}.npy") != digest:
                raise ValueError(f"historical cache contents changed: {name}")
        semantic = {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "policy": asdict(config), "selection_seeds": selection_seeds,
            "reporting_seeds": reporting_seeds, "initial_sha256": fingerprint(args.checkpoint),
            "planner_selection_sha256": fingerprint(args.planner_selection),
            "historical_manifest_sha256": fingerprint(args.historical_cache / "manifest.json"),
            "objective": "equal-scenario capture + 0.15 * discovery",
            "batch_mix": "three planner-state batches per historical-anchor batch",
            "loss": "Poisson + integrated count + native dwell windows + 0.03 anchor count KL",
            "sources": {str(p): fingerprint(p) for p in (
                Path(__file__), Path(__file__).parent.parent / "timing_belief.py",
                Path(__file__).parent.parent / "timing_planner.py",
                Path(__file__).parent / "joint_study.py")},
        }
        config_path = args.run_dir / "config.json"
        if args.resume:
            previous = json.loads(config_path.read_text())
            for key in ("policy", "selection_seeds", "reporting_seeds", "initial_sha256",
                        "planner_selection_sha256", "historical_manifest_sha256", "sources"):
                if previous[key] != json.loads(json.dumps(semantic[key])):
                    raise ValueError(f"resume semantics changed: {key}")
            for key, value in previous["arguments"].items():
                if key != "resume" and semantic["arguments"][key] != value:
                    raise ValueError(f"resume argument changed: {key}")
            semantic = previous
        elif config_path.exists():
            raise ValueError("use a fresh directory or resume")
        else:
            write_json(config_path, semantic)
            shutil.copyfile(args.checkpoint, args.run_dir / "initial.pt")
        base, _ = load_belief(args.run_dir / "initial.pt")
        if historical["semantic"]["config"] != asdict(base.config):
            raise ValueError("historical input and forecast schema differs")
        student_dir = args.run_dir / "planner-cache"
        anchor_path = args.run_dir / "planner-anchor.npy"
        historical_anchor = args.run_dir / "historical-anchor.npy"
        if not args.resume:
            with ProcessPoolExecutor(args.workers,
                    mp_context=multiprocessing.get_context("spawn")) as pool:
                manifest = collect_student(base, config, student_dir, args.worlds, pool,
                    args.workers, scheduler_factory=CalibratedTimingPlannerPolicy,
                    seed_offset=100000)
            write_json(args.run_dir / "collection.json", manifest)
            cache_anchor(base, student_dir, anchor_path, args.batch_size)
            cache_anchor(base, args.historical_cache, historical_anchor, args.batch_size)
            write_json(args.run_dir / "cache-integrity.json", {
                "planner_manifest_sha256": fingerprint(student_dir / "manifest.json"),
                "planner_anchor_sha256": fingerprint(anchor_path),
                "historical_anchor_sha256": fingerprint(historical_anchor),
            })
        else:
            integrity = json.loads((args.run_dir / "cache-integrity.json").read_text())
            for key, path in (("planner_manifest_sha256", student_dir / "manifest.json"),
                              ("planner_anchor_sha256", anchor_path),
                              ("historical_anchor_sha256", historical_anchor)):
                if integrity[key] != fingerprint(path):
                    raise ValueError(f"resume cache changed: {key}")
            manifest = json.loads((student_dir / "manifest.json").read_text())
            for name, digest in manifest["arrays"].items():
                if fingerprint(student_dir / f"{name}.npy") != digest:
                    raise ValueError(f"planner cache contents changed: {name}")
        student_stream = JointStream(student_dir, args.batch_size, anchor_path)
        old_stream = JointStream(args.historical_cache, args.batch_size, historical_anchor)
        for seed in args.seeds:
            torch.manual_seed(seed)
            run = args.run_dir / f"seed-{seed}"
            run.mkdir(exist_ok=args.resume)
            model, _ = load_belief(args.run_dir / "initial.pt")
            optimizer = optimizer_for(model, args.learning_rate, args.encoder_factor)
            rows, best, start_epoch = [], -np.inf, 0
            if args.resume and (run / "optimizer.pt").exists():
                model, latest = load_belief(run / "latest.pt")
                optimizer = optimizer_for(model, args.learning_rate, args.encoder_factor)
                state = load_torch(run / "optimizer.pt")
                optimizer.load_state_dict(state["optimizer"])
                rows, best, start_epoch = state["rows"], state["best"], state["epoch"] + 1
                if latest["epoch"] != state["epoch"]:
                    raise ValueError("resume optimizer/model epoch differs")
                if start_epoch > args.epochs:
                    write_json(run / "frozen.json", {
                        "checkpoint_sha256": fingerprint(run / "best.pt"), "score": best})
                    continue
            for epoch in range(start_epoch, args.epochs + 1):
                started = time.perf_counter()
                loss_sum, samples = torch.zeros((), device="cuda"), 0
                if epoch:
                    model.train()
                    streams = [iter(student_stream.batches(seed, epoch)),
                               iter(old_stream.batches(seed, epoch))]
                    for step in range(args.steps_per_epoch):
                        slot = int(step % 4 == 3)
                        try:
                            batch = next(streams[slot])
                        except StopIteration:
                            stream = old_stream if slot else student_stream
                            streams[slot] = iter(stream.batches(seed, epoch + step + 1000))
                            batch = next(streams[slot])
                        progress = ((epoch - 1) + step / args.steps_per_epoch) / args.epochs
                        rate = args.learning_rate * (
                            0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
                        for group in optimizer.param_groups:
                            group["lr"] = rate * group["factor"]
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prediction = model(batch["history"])
                            loss = refinement_loss(prediction, batch)
                        torch._assert_async(torch.isfinite(loss), "nonfinite refinement loss")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        torch._assert_async(torch.isfinite(norm), "nonfinite refinement gradient")
                        optimizer.step()
                        loss_sum += loss.detach() * len(batch["history"])
                        samples += len(batch["history"])
                    for stream in streams:
                        stream.close()
                row = {"seed": seed, "epoch": epoch, "samples": samples,
                       "loss": float(loss_sum / samples) if samples else None,
                       "training_seconds": time.perf_counter() - started,
                       "learning_rate": optimizer.param_groups[-1]["lr"],
                       "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                       "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                if epoch % args.evaluate_every == 0 or epoch == args.epochs:
                    model.eval()
                    value, means, profile = assess(model, config, selection_seeds, args.workers)
                    row.update(score=value, selection=means, selection_profile=profile)
                    save_belief(run / f"epoch-{epoch}.pt", model, {**semantic, **row})
                    if value > best:
                        best = value
                        save_belief(run / "best.pt", model, {**semantic, **row})
                rows.append(row)
                save_belief(run / "latest.pt", model, {**semantic, **row})
                save_torch(run / "optimizer.pt", {
                    "epoch": epoch, "best": best, "rows": rows,
                    "optimizer": optimizer.state_dict()})
                write_json(run / "progress.json", rows)
                print(json.dumps(row), flush=True)
            write_json(run / "frozen.json", {"checkpoint_sha256": fingerprint(run / "best.pt"),
                                              "score": best})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
