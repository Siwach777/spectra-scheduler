"""Run the reference learner using the shared experiment lifecycle."""

import argparse
import json
import signal
from pathlib import Path

from .runner import RunConfig, run_experiment


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--epochs", type=int, default=1, help="total epoch budget, including resumed epochs"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda",),
        help="training requires CUDA; no CPU fallback",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--validation-workers", type=int, default=1)
    args = parser.parse_args(arguments)
    # Optional Torch imports only after --help and argument parsing.
    from ..pulse_replay import ReplayConfig
    from ..replay_env import InterfaceConfig
    from .predictor import PredictorConfig, PredictorLearner

    raw = json.loads(args.config.read_text())
    allowed = {
        "root",
        "train_plan",
        "validation_plan",
        "predictor",
        "receiver",
        "interface",
        "selection",
        "maximize",
        "min_delta",
        "cache",
    }
    if set(raw) - allowed:
        raise ValueError(f"unknown experiment fields: {sorted(set(raw) - allowed)}")
    base = args.config.resolve().parent
    train = json.loads((base / raw["train_plan"]).read_text())
    validation = json.loads((base / raw["validation_plan"]).read_text())
    learner = PredictorLearner(
        base / raw["root"],
        train,
        validation,
        PredictorConfig(**raw.get("predictor", {})),
        ReplayConfig(**raw.get("receiver", {})),
        InterfaceConfig(**raw.get("interface", {})),
        device=args.device,
        threads=args.threads,
        validation_workers=args.validation_workers,
        cache=base / raw["cache"] if raw.get("cache") else None,
    )
    config = RunConfig(
        epochs=args.epochs,
        selection=tuple(raw.get("selection", RunConfig().selection)),
        maximize=raw.get("maximize", True),
        min_delta=raw.get("min_delta", 0.0),
    )

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt("termination requested; resume the last committed epoch")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        run_experiment(args.run_dir, learner, config, resume=args.resume)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
