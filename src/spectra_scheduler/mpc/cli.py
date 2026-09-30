"""Command-line entry points for pretraining, search learning and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoints import save_model
from .config import MAX_BANDS, Config, TrainConfig
from .data import collect_demonstrations
from .evaluation import _benchmark, evaluate_run
from .learning import train_model
from .trainer import run


def demonstration_main(argv: list[str] | None = None) -> None:
    """Command-line interface for training and benchmarking Neural-MPC."""
    parser = argparse.ArgumentParser(
        prog="spectra_scheduler.neural_mpc",
        description="Neural-MPC: MuZero-style cognitive receiver scheduler",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train ---
    p_train = sub.add_parser("train", help="Pre-train the world model")
    p_train.add_argument(
        "--episodes", type=int, default=200, help="Demo episodes per baseline policy (default: 200)"
    )
    p_train.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--threads", type=int, default=2)
    p_train.add_argument("--device", default="cuda", choices=["cuda"])
    p_train.add_argument("--output", required=True, help="Output path for .pt checkpoint")

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Benchmark against baselines")
    p_bench.add_argument("--model", required=True, help="Path to trained .pt checkpoint")
    p_bench.add_argument("--runs", type=int, default=20)
    p_bench.add_argument("--seed", type=int, default=10000)
    p_bench.add_argument("--simulations", type=int, default=50)
    p_bench.add_argument(
        "--suites", nargs="+", default=["randomized", "receiver-shift", "tracking"]
    )
    p_bench.add_argument("--output", default="reports/generated/neural-mpc.json")
    p_bench.add_argument("--device", default="cuda", choices=["cuda"])

    args = parser.parse_args(argv)
    torch.set_num_threads(getattr(args, "threads", 1))

    if args.command == "train":
        cfg = TrainConfig(
            demo_episodes_per_policy=args.episodes,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )

        print(
            f"[1/3] Collecting demonstrations "
            f"({len(cfg.demo_policies)} policies × {cfg.demo_episodes_per_policy} seeds)..."
        )

        def demo_progress(policy: str, seed: int, total: int) -> None:
            if seed % 50 == 0:
                print(f"  {policy}: seed {seed} ({total} episodes collected)")

        t0 = time.time()
        episodes = collect_demonstrations(cfg, progress=demo_progress)
        t1 = time.time()
        print(f"  Collected {len(episodes)} episodes in {t1 - t0:.1f}s")

        print(f"\n[2/3] Training world model ({cfg.epochs} epochs)...")

        def train_progress(epoch: int, loss: float, n: int) -> None:
            if epoch % 5 == 0 or epoch == cfg.epochs - 1:
                print(f"  epoch {epoch:3d}/{cfg.epochs}  loss={loss:.4f}  episodes={n}")

        model = train_model(episodes, cfg, progress=train_progress)
        t2 = time.time()
        print(f"  Training complete in {t2 - t1:.1f}s")

        print(f"\n[3/3] Saving checkpoint to {args.output}")
        save_model(
            model,
            args.output,
            metadata={
                "demo_episodes": len(episodes),
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "policies": list(cfg.demo_policies),
                "seed": cfg.seed,
                "gamma": cfg.gamma,
                "training_split": "train",
                "num_bands": MAX_BANDS,
                "reward": {"hit": 1.0, "retuning": 0.05, "coverage": 0.2},
            },
        )
        print("  Done.")

    elif args.command == "benchmark":
        print(f"Benchmarking {args.model} ({args.runs} validation seeds)...\n")
        _benchmark(
            args.model,
            runs=args.runs,
            start_seed=args.seed,
            device=args.device,
            simulations=args.simulations,
            suites=tuple(args.suites),
            output=args.output,
        )

    else:
        parser.print_help()
        sys.exit(1)


def training_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initial", help="Optional demonstration-pretrained .pt model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate best.pt; no training")
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    for name, value in asdict(Config()).items():
        if isinstance(value, tuple):
            parser.add_argument(
                "--" + name.replace("_", "-"), nargs="+", type=int, default=argparse.SUPPRESS
            )
        elif isinstance(value, bool):
            parser.add_argument(
                "--" + name.replace("_", "-"),
                default=argparse.SUPPRESS,
                action=argparse.BooleanOptionalAction,
            )
        else:
            parser.add_argument(
                "--" + name.replace("_", "-"), default=argparse.SUPPRESS, type=type(value)
            )
    args = vars(parser.parse_args())
    if "dwell_steps" in args:
        args["dwell_steps"] = tuple(args["dwell_steps"])
    directory, resume, initial = args.pop("run_dir"), args.pop("resume"), args.pop("initial")
    evaluate, split = args.pop("evaluate"), args.pop("evaluation_split")
    if evaluate:
        if resume or initial:
            parser.error("evaluation cannot resume or initialize training")
        print(
            json.dumps(
                evaluate_run(
                    directory, split, args.get("validation_episodes", 30), args.get("workers", 1)
                )
            )
        )
        return
    if resume:
        saved = torch.load(Path(directory) / "training.pt", map_location="cpu", weights_only=True)
        args = {**saved["config"], **args}
    run(Config(**args), directory, resume, initial)
