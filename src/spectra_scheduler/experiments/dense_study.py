"""CUDA study of decision-focused MLPs on train-only all-action replay labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..dense_policy import DenseValuePolicy, dense_spec, load_dense_model, save_dense_model
from ..dense_value import DenseValueConfig, DenseValueNetwork, dense_value_loss
from ..policy_benchmark import PolicySpec, benchmark_policies, validate_plan
from ..pulse_replay import ReplayConfig
from ..replay_baselines import RateProbePolicy
from ..replay_env import InterfaceConfig, ReplayEnv
from .dense_cache import build_dense_cache, cache_configuration, open_dense_cache
from .dense_stream import DenseBlockStream
from .storage import fingerprint, load_torch, run_lock, save_torch, write_json


def _harmonic(report, name):
    metrics = report["summary"][name]
    capture = metrics["interception_ratio"]["mean"]
    discovery = metrics["discovery_fraction"]["mean"]
    if capture is None or discovery is None:
        return None
    return 2 * capture * discovery / max(capture + discovery, 1e-12)


def _sources():
    root = Path(__file__).resolve().parent.parent
    names = (
        "dense_value.py",
        "dense_policy.py",
        "experiments/dense_study.py",
        "experiments/dense_replay.py",
        "experiments/dense_cache.py",
        "experiments/dense_stream.py",
        "pulse_replay.py",
        "replay_features.py",
        "replay_env.py",
        "dataset_io.py",
        "policy_benchmark.py",
        "replay_evaluation.py",
    )
    return {name: fingerprint(root / name) for name in names}


def _rate_probe():
    return PolicySpec("rate-probe", RateProbePolicy, "short-probe/long-exploit", ())


def _working_policy(model, specification, trained_on, name):
    return PolicySpec(
        name,
        partial(DenseValuePolicy, model, specification),
        "frozen-inference-current-epoch",
        trained_on,
    )


def _pooled_capture(comparison, name):
    rows = (row["policies"][name] for row in comparison["results"])
    intercepted, truth = 0, 0
    for row in rows:
        intercepted += row["intercepted_pulses"]
        truth += row["truth_pulses"]
    return intercepted / truth if truth else None


def _selection(root, plan, model, specification, trained_on, receiver, interface, name, metric):
    comparison = benchmark_policies(
        root,
        plan,
        [_working_policy(model, specification, trained_on, name), _rate_probe()],
        baseline="rate-probe",
        receiver=receiver,
        interface=interface,
        inference_batch_size=16,
    )
    mean_score = _harmonic(comparison, name)
    pooled_capture = _pooled_capture(comparison, name)
    discovery = comparison["summary"][name]["discovery_fraction"]["mean"]
    pooled_score = (
        2 * pooled_capture * discovery / max(pooled_capture + discovery, 1e-12)
        if pooled_capture is not None and discovery is not None
        else None
    )
    score = (
        mean_score if metric == "mean" or pooled_score is None or mean_score is None
        else (mean_score + pooled_score) / 2
    )
    return {
        "score": score,
        "pooled_capture": pooled_capture,
        "rate_probe_pooled_capture": _pooled_capture(comparison, "rate-probe"),
        "capture": comparison["summary"][name]["interception_ratio"]["mean"],
        "discovery": discovery,
        "rate_probe_capture": comparison["summary"]["rate-probe"]["interception_ratio"]["mean"],
        "rate_probe_discovery": comparison["summary"]["rate-probe"]["discovery_fraction"]["mean"],
    }


def _train_one(
    run,
    seed,
    root,
    plans,
    specification,
    cache,
    tensors,
    *,
    epochs,
    batch_size,
    learning_rate,
    receiver,
    interface,
    encoder,
    training_hashes,
    opportunity_power,
    selection_metric,
    initial_model,
):
    run.mkdir(parents=True, exist_ok=True)
    trained_on = tuple(sorted(training_hashes))
    cfg = DenseValueConfig(
        bands=interface.bands,
        dwells=len(interface.dwell_us),
        history_steps=16,
        hidden=128,
        encoder=encoder,
    )
    semantic = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "opportunity_power": opportunity_power,
        "selection_metric": selection_metric,
        "initial_model_sha256": fingerprint(initial_model) if initial_model else None,
        "model": asdict(cfg),
        "cache_sha256": [fingerprint(path / "manifest.json") for path in cache],
        "selection_plan": plans["selection"],
        "source_sha256": _sources(),
    }
    torch.manual_seed(seed)
    model = DenseValueNetwork(cfg).to("cuda")
    if initial_model is not None:
        pretrained, saved_spec, _ = load_dense_model(initial_model, device="cpu")
        if pretrained.config != cfg or saved_spec != specification:
            raise ValueError("initial dense checkpoint architecture or interface differs")
        model.load_state_dict(pretrained.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)
    latest = run / "latest.pt"
    best_score, best_epoch, start, rows = -1.0, 0, 0, []
    if latest.exists():
        state = load_torch(latest)
        if state["semantic"] != json.loads(json.dumps(semantic)):
            raise ValueError("dense training resume configuration or source changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_score = state["best_score"]
        best_epoch = state["best_epoch"]
        start, rows = state["epoch"], state["history"]
    else:
        model.eval()
        initial = _selection(
            root,
            plans["selection"],
            model,
            specification,
            trained_on,
            receiver,
            interface,
            f"dense-mlp-{seed}",
            selection_metric,
        )
        rows.append({"epoch": 0, "selection": initial})
        best_score = initial["score"] if initial["score"] is not None else -1.0
        save_dense_model(run / "best.pt", model, specification, trained_on)
        write_json(
            run / "progress.json",
            {
                "semantic": semantic,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": rows,
            },
        )
    total = len(tensors) if isinstance(tensors, DenseBlockStream) else len(tensors["history"])
    for epoch in range(start, epochs):
        model.train()
        if isinstance(tensors, DenseBlockStream):
            batches_iter = tensors.batches(batch_size, seed, epoch)
        else:
            indices = torch.randperm(
                total,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(seed * 10_000 + epoch),
            )
            batches_iter = (
                {
                    key: value.index_select(0, indices[offset : offset + batch_size])
                    for key, value in tensors.items()
                }
                for offset in range(0, total, batch_size)
            )
        sums = torch.zeros(4, device="cuda")
        batches = 0
        for batch in batches_iter:
            optimizer.zero_grad(set_to_none=True)
            loss, terms = dense_value_loss(
                model, batch, opportunity_power=opportunity_power, validated=True
            )
            torch._assert_async(torch.isfinite(loss), "nonfinite dense model loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            torch._assert_async(torch.isfinite(norm), "nonfinite dense model gradient")
            optimizer.step()
            sums += torch.stack(
                (
                    terms["count"],
                    terms["ranking_regret"],
                    terms["greedy_regret"],
                    terms["active_fraction"],
                )
            )
            batches += 1
        row = {
            "epoch": epoch + 1,
            "loss": dict(
                zip(
                    ("count", "ranking_regret", "greedy_regret", "active_fraction"),
                    (sums / batches).cpu().tolist(),
                    strict=True,
                )
            ),
            "samples": total,
            "updates": batches,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        if (epoch + 1) % 2 == 0 or epoch + 1 == epochs:
            model.eval()
            row["selection"] = _selection(
                root,
                plans["selection"],
                model,
                specification,
                trained_on,
                receiver,
                interface,
                f"dense-mlp-{seed}",
                selection_metric,
            )
            score = row["selection"]["score"]
            if score is not None and score > best_score:
                best_score = score
                best_epoch = epoch + 1
                save_dense_model(run / "best.pt", model, specification, trained_on)
        rows.append(row)
        save_torch(
            latest,
            {
                "semantic": semantic,
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": rows,
            },
        )
        write_json(
            run / "progress.json",
            {
                "semantic": semantic,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": rows,
            },
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "loss": row["loss"],
                    "selection": row.get("selection"),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                }
            ),
            flush=True,
        )
    return run / "best.pt"


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--plans", type=Path, default=Path("artifacts/predictor-study/plans.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/dense-value"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--extra-train-plan", type=Path)
    parser.add_argument("--extra-cache-dir", type=Path)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--opportunity-power", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=("mean", "balanced"), default="mean")
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--encoder", choices=("mlp", "spectral_mlp"), default="mlp")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args(arguments)
    if args.report_only and args.selection_only:
        parser.error("report-only and selection-only are mutually exclusive")
    if not torch.cuda.is_available():
        raise RuntimeError("dense model training and neural evaluation require CUDA")
    if min(
        args.epochs, args.batch_size, args.workers, args.bands, args.num_seeds, args.block_size
    ) < 1:
        raise ValueError("positive training budget and workers required")
    if (args.extra_train_plan is None) != (args.extra_cache_dir is None):
        parser.error("extra-train-plan and extra-cache-dir must be supplied together")
    if not 0 <= args.opportunity_power <= 1:
        parser.error("opportunity-power must lie in [0, 1]")
    if args.initial_model is not None and args.num_seeds != 1:
        parser.error("warm-start training requires one seed")
    torch.set_num_threads(2)
    plans = json.loads(args.plans.read_text())
    if any(
        plans[key]["split"] != split
        for key, split in (("train", "train"), ("selection", "val"), ("report", "val"))
    ):
        raise ValueError("dense study requires train/validation plans")
    for plan in plans.values():
        validate_plan(args.root, plan)
    hashes = [
        {record["sha256"] for record in plans[name]["recordings"]}
        for name in ("train", "selection", "report")
    ]
    if any(hashes[i] & hashes[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("dense study plans overlap by content")
    extra_plan = None
    if args.extra_train_plan is not None:
        extra_plan = json.loads(args.extra_train_plan.read_text())
        if extra_plan.get("split") != "train":
            raise ValueError("additional dense recordings must be training-only")
        validate_plan(args.root, extra_plan)
        additional = {record["sha256"] for record in extra_plan["recordings"]}
        if any(additional & group for group in hashes):
            raise ValueError("additional training recordings overlap another study split")
    receiver, interface = ReplayConfig(), InterfaceConfig(bands=args.bands)
    with run_lock(args.run_dir):
        specification = ReplayEnv(
            args.root / plans["train"]["recordings"][0]["path"], receiver, interface
        ).specification()
        cache_dir = args.cache_dir or args.run_dir / "cache"
        cache_paths = [cache_dir]
        training_hashes = hashes[0].copy()
        if extra_plan is not None:
            cache_paths.append(args.extra_cache_dir)
            training_hashes.update(additional)
        best = [args.run_dir / f"mlp-{seed}" / "best.pt" for seed in range(args.num_seeds)]
        if not args.report_only:
            build_dense_cache(
                args.root,
                plans["train"],
                cache_dir,
                workers=args.workers,
                receiver=receiver,
                interface=interface,
            )
            _, arrays = open_dense_cache(
                cache_dir, cache_configuration(args.root, plans["train"], receiver, interface)
            )
            if extra_plan is None:
                tensors = {
                    name: torch.from_numpy(np.array(value, copy=True)).to("cuda")
                    for name, value in arrays.items()
                }
            else:
                build_dense_cache(
                    args.root,
                    extra_plan,
                    args.extra_cache_dir,
                    workers=args.workers,
                    receiver=receiver,
                    interface=interface,
                )
                _, additional_arrays = open_dense_cache(
                    args.extra_cache_dir,
                    cache_configuration(args.root, extra_plan, receiver, interface),
                )
                tensors = DenseBlockStream([arrays, additional_arrays], args.block_size)
            for seed in range(args.num_seeds):
                _train_one(
                    args.run_dir / f"mlp-{seed}",
                    seed,
                    args.root,
                    plans,
                    specification,
                    cache_paths,
                    tensors,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    receiver=receiver,
                    interface=interface,
                    encoder=args.encoder,
                    training_hashes=training_hashes,
                    opportunity_power=args.opportunity_power,
                    selection_metric=args.selection_metric,
                    initial_model=args.initial_model,
                )
        for path in best:
            if not path.is_file():
                raise FileNotFoundError(f"missing trained dense checkpoint: {path}")
        if args.selection_only:
            return 0
        policies = [dense_spec(path, f"dense-mlp-{i}") for i, path in enumerate(best)]
        policies.append(_rate_probe())
        comparison = benchmark_policies(
            args.root,
            plans["report"],
            policies,
            baseline="rate-probe",
            receiver=receiver,
            interface=interface,
            inference_batch_size=32,
        )
        comparison["scope"] = (
            f"{len(training_hashes)} train recordings; "
            f"{len(plans['selection']['recordings'])} checkpoint-selection validation recordings; "
            f"{len(plans['report']['recordings'])} separate reporting validation recordings; "
            "test split unused"
        )
        comparison["models"] = [
            {
                "path": str(path),
                "sha256": fingerprint(path),
                "selected_epoch": json.loads((path.parent / "progress.json").read_text())[
                    "best_epoch"
                ],
            }
            for path in best
        ]
        comparison["source_sha256"] = _sources()
        comparison["pooled_capture"] = {
            policy.name: _pooled_capture(comparison, policy.name) for policy in policies
        }
        write_json(args.run_dir / "comparison.json", comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
