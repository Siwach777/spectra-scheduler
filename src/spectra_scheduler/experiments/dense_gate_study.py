"""CUDA-only train-cache conditional gate, frozen selection and paired reporting."""

import argparse
import json
import resource
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..dense_gate import DenseRateGate, gate_loss
from ..dense_policy import DenseValuePolicy, load_dense_model
from ..policy_benchmark import PolicySpec, benchmark_policies, summarize, validate_plan
from ..pulse_replay import ReplayConfig
from ..replay_baselines import RateProbePolicy
from ..replay_env import InterfaceConfig
from .dense_blend_study import annotate, pooled_difference_interval
from .dense_cache import cache_configuration, open_dense_cache
from .storage import fingerprint, load_torch, run_lock, save_torch, write_json


def exploitation_tensors(arrays, bands, max_bytes=512 * 1024**2):
    """Select legal exploitation states with bounded CPU chunks, then stage CUDA."""
    indices = []
    for start in range(0, len(arrays["history"]), 4096):
        latest = arrays["history"][start : start + 4096, -1, :-2].reshape(-1, bands, 9)
        eligible = latest[..., 0].all(-1) & (latest[..., 1].max(-1) < 0.05)
        indices.append(np.flatnonzero(eligible) + start)
    indices = np.concatenate(indices)
    required = sum(len(indices) * np.prod(a.shape[1:]) * a.dtype.itemsize for a in arrays.values())
    if not len(indices) or required > max_bytes:
        raise MemoryError(
            f"eligible CUDA training cache requires {required} bytes; limit {max_bytes}"
        )
    tensors = {}
    for name, array in arrays.items():
        dtype = torch.from_numpy(np.empty((), dtype=array.dtype)).dtype
        tensor = torch.empty((len(indices), *array.shape[1:]), dtype=dtype, device="cuda")
        for start in range(0, len(indices), 2048):
            chosen = indices[start : start + 2048]
            tensor[start : start + len(chosen)].copy_(torch.from_numpy(array[chosen]))
        tensors[name] = tensor
    return tensors


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--plans", type=Path, default=Path("artifacts/predictor-study/plans.json"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/dense-value-24/mlp-0/best.pt")
    )
    parser.add_argument("--cache", type=Path, default=Path("artifacts/dense-value-24/cache"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/dense-gate-study"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args(arguments)
    if not torch.cuda.is_available():
        raise RuntimeError("conditional gate training and neural evaluation require CUDA")
    if args.epochs < 1 or args.batch_size < 256 or args.seed < 0:
        parser.error("positive epochs, batch size >=256 and nonnegative seed required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    plans = json.loads(args.plans.read_text())
    groups = {}
    for key, split in (("train", "train"), ("selection", "val"), ("report", "val")):
        if plans[key]["split"] != split:
            raise ValueError("train/validation split required")
        validate_plan(args.root, plans[key])
        groups[key] = {r["sha256"] for r in plans[key]["recordings"]}
    if any(
        groups[a] & groups[b]
        for a, b in (("train", "selection"), ("train", "report"), ("selection", "report"))
    ):
        raise ValueError("study splits overlap")
    base_hash = fingerprint(args.checkpoint)
    base, specification, trained = load_dense_model(args.checkpoint, "cuda")
    if (groups["selection"] | groups["report"]) & set(trained):
        raise ValueError("base training overlaps evaluation")
    trained = tuple(sorted(set(trained) | groups["train"]))
    receiver = ReplayConfig(**specification["receiver"])
    interface = InterfaceConfig(**specification["interface"])
    manifest, arrays = open_dense_cache(
        args.cache, cache_configuration(args.root, plans["train"], receiver, interface)
    )
    tensors = exploitation_tensors(arrays, interface.bands)
    model = DenseRateGate(base, specification).to("cuda")
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=args.learning_rate, fused=True)
    baseline = PolicySpec("rate-probe", RateProbePolicy, "short-probe/long-exploit", ())
    constant = PolicySpec(
        "constant-blend",
        partial(DenseValuePolicy, base, specification, model_weight=0.25),
        f"sha256:{base_hash};weight:.25",
        trained,
    )
    candidate = PolicySpec(
        "conditional-gate",
        partial(DenseValuePolicy, model, specification),
        f"base-sha256:{base_hash};gate-seed:{args.seed}",
        trained,
    )
    policies = [candidate, constant, baseline]
    source_root = Path(__file__).resolve().parent.parent
    semantic = {
        "version": 1,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "opportunity_power": 0.75,
        "count_loss_weight": 0.2,
        "ranking_temperature": 0.25,
        "gate_prior_weight": 0.01,
        "initial_weight": 0.25,
        "base_checkpoint": str(args.checkpoint),
        "base_sha256": base_hash,
        "cache_manifest_sha256": fingerprint(args.cache / "manifest.json"),
        "training_exploitation_rows": len(tensors["history"]),
        "plans_sha256": fingerprint(args.plans),
        "selection_objective": "mean of recording-mean and pulse-pooled capture",
        "source_sha256": {
            name: fingerprint(source_root / name)
            for name in (
                "dense_gate.py",
                "dense_policy.py",
                "dense_value.py",
                "experiments/dense_gate_study.py",
            )
        },
    }
    best_score, best_epoch, history = -1.0, 0, []
    with run_lock(args.run_dir):
        for epoch in range(args.epochs + 1):
            if epoch:
                model.train()
                terms = torch.zeros(4, device="cuda")
                count = 0
                permutation = torch.randperm(
                    len(tensors["history"]),
                    device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(args.seed * 10000 + epoch),
                )
                for start in range(0, len(permutation), args.batch_size):
                    chosen = permutation[start : start + args.batch_size]
                    batch = {
                        name: tensor.index_select(0, chosen) for name, tensor in tensors.items()
                    }
                    optimizer.zero_grad(set_to_none=True)
                    loss, current = gate_loss(model, batch)
                    torch._assert_async(torch.isfinite(loss), "nonfinite gate loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.0)
                    torch._assert_async(torch.isfinite(norm), "nonfinite gate gradient")
                    optimizer.step()
                    terms += current
                    count += 1
                loss_terms = (terms / count).cpu().tolist()
            else:
                loss_terms = None
            model.eval()
            comparison = annotate(
                benchmark_policies(
                    args.root,
                    plans["selection"],
                    policies,
                    baseline="rate-probe",
                    receiver=receiver,
                    interface=interface,
                    inference_batch_size=16,
                ),
                policies,
            )
            mean = comparison["summary"][candidate.name]["interception_ratio"]["mean"]
            pooled = comparison["pooled_capture"][candidate.name]
            score = (mean + pooled) / 2
            if score > best_score:
                best_score, best_epoch = score, epoch
                save_torch(
                    args.run_dir / "best.pt",
                    {
                        "version": 1,
                        "semantic": semantic,
                        "specification": specification,
                        "training_hashes": list(trained),
                        "gate_hidden": model.hidden,
                        "gate_weights": {
                            k: v.detach().cpu() for k, v in model.gate.state_dict().items()
                        },
                        "selected_epoch": epoch,
                    },
                )
                write_json(args.run_dir / "best-selection.json", comparison)
            row = {
                "epoch": epoch,
                "loss_terms": loss_terms,
                "mean_capture": mean,
                "pooled_capture": pooled,
                "score": score,
                "best_epoch": best_epoch,
                "constant_mean": comparison["summary"][constant.name]["interception_ratio"]["mean"],
                "constant_pooled": comparison["pooled_capture"][constant.name],
            }
            history.append(row)
            write_json(
                args.run_dir / "progress.json",
                {"semantic": semantic, "history": history, "best_epoch": best_epoch},
            )
            print(json.dumps(row), flush=True)
        checkpoint = args.run_dir / "best.pt"
        frozen = {
            "semantic": semantic,
            "selected_epoch": best_epoch,
            "checkpoint_sha256": fingerprint(checkpoint),
            "selection_sha256": fingerprint(args.run_dir / "best-selection.json"),
        }
        write_json(args.run_dir / "frozen.json", frozen)
        if args.selection_only:
            return 0
        model.gate.load_state_dict(load_torch(checkpoint)["gate_weights"])
        candidate = PolicySpec(
            candidate.name,
            candidate.factory,
            f"sha256:{fingerprint(checkpoint)};base-sha256:{base_hash}",
            trained,
        )
        policies = [candidate, constant, baseline]
        report = annotate(
            benchmark_policies(
                args.root,
                plans["report"],
                policies,
                baseline="rate-probe",
                receiver=receiver,
                interface=interface,
                inference_batch_size=16,
            ),
            policies,
        )
        report["versus_constant_blend"] = summarize(report["results"], policies, constant.name)
        report["pooled_vs_constant_95_interval"] = pooled_difference_interval(
            report, candidate.name, constant.name
        )
        report["frozen"] = frozen
        report["resources"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        if (
            fingerprint(args.checkpoint) != base_hash
            or fingerprint(checkpoint) != frozen["checkpoint_sha256"]
        ):
            raise ValueError("model artifact changed during reporting")
        write_json(args.run_dir / "comparison.json", report)
        print(
            json.dumps(
                {
                    "phase": "report",
                    "pooled_capture": report["pooled_capture"],
                    "resources": report["resources"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
