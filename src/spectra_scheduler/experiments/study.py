"""CUDA multi-seed predictor comparison with frozen selection/reporting recordings."""

import argparse
import gc
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ..forecast_model import load_predictor, predictor_spec
from ..policy_benchmark import PolicySpec, benchmark_policies, make_plan
from ..pulse_replay import ReplayConfig
from ..replay_env import InterfaceConfig
from ..replay_evaluation import ReferencePolicy
from .cache import build_cache
from .predictor import PredictorConfig, PredictorLearner
from .runner import RunConfig, run_experiment
from .storage import checkpoint_path, run_lock, verify_artifacts, write_json


def aggregate(report, encoders, seeds):
    """Resample independent recording groups and trained seeds, keeping pairs intact."""
    output = {}
    groups = sorted({row["group"] for row in report["results"]})
    names = [f"{encoder}-{seed}" for encoder in encoders for seed in seeds] + ["observed-rate"]
    eligible = [
        group
        for group in groups
        if all(
            row["policies"][name]["evaluation"]["interception_ratio"] is not None
            for row in report["results"]
            if row["group"] == group
            for name in names
        )
    ]
    excluded = len(groups) - len(eligible)
    groups = eligible
    if not groups:
        raise ValueError("no recordings with defined paired interception ratios")
    for encoder in encoders:
        values = np.array(
            [
                [
                    np.mean(
                        [
                            row["policies"][f"{encoder}-{seed}"]["evaluation"]["interception_ratio"]
                            for row in report["results"]
                            if row["group"] == group
                        ]
                    )
                    for group in groups
                ]
                for seed in seeds
            ]
        )
        reference = np.array(
            [
                np.mean(
                    [
                        row["policies"]["observed-rate"]["evaluation"]["interception_ratio"]
                        for row in report["results"]
                        if row["group"] == group
                    ]
                )
                for group in groups
            ]
        )
        rng = np.random.default_rng(0)
        bootstrap = np.empty(2000)
        for i in range(len(bootstrap)):
            sampled_seeds = rng.integers(len(seeds), size=len(seeds))
            sampled_groups = rng.integers(len(groups), size=len(groups))
            bootstrap[i] = (
                values[np.ix_(sampled_seeds, sampled_groups)] - reference[sampled_groups]
            ).mean()
        ordered = np.sort(values.ravel())
        trim = len(ordered) // 4
        output[encoder] = {
            "training_seeds": list(seeds),
            "independent_recordings": len(groups),
            "excluded_undefined_recordings": excluded,
            "mean_interception": float(values.mean()),
            "interquartile_mean_interception": float(ordered[trim : len(ordered) - trim].mean()),
            "mean_interception_by_training_seed": values.mean(1).tolist(),
            "paired_gain_over_observed_rate": float((values - reference).mean()),
            "hierarchical_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    return output


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--train-files", type=int, default=128)
    parser.add_argument("--selection-files", type=int, default=16)
    parser.add_argument("--report-files", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--encoders", nargs="+", choices=("mlp", "gru", "tcn"), default=["mlp", "gru", "tcn"]
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(arguments)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; no CPU training fallback")
    if (
        len(set(args.seeds)) != len(args.seeds)
        or len(set(args.encoders)) != len(args.encoders)
        or min(
            args.epochs,
            args.train_files,
            args.selection_files,
            args.report_files,
            args.workers,
            args.batch_size,
            args.hidden,
            args.bands,
        )
        < 1
    ):
        raise ValueError("positive budgets and distinct seeds/encoders required")
    for seed in args.seeds:
        PredictorConfig(seed=seed)
    torch.set_num_threads(1)
    root, directory = args.root.resolve(), args.directory.resolve()
    receiver, interface = ReplayConfig(), InterfaceConfig(bands=args.bands)
    settings = {
        **vars(args),
        "root": str(root),
        "directory": str(directory),
        "receiver": asdict(receiver),
        "interface": asdict(interface),
    }
    settings.pop("epochs")
    settings.pop("evaluate_only")
    settings = json.loads(json.dumps(settings))
    with run_lock(directory):
        settings_path = directory / "study.json"
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != settings:
                raise ValueError("study settings changed; use a fresh directory")
            plans = json.loads((directory / "plans.json").read_text())
        else:
            if any(p.name != "run.lock" for p in directory.iterdir()):
                raise ValueError("new study requires an empty directory")
            train = make_plan(root, "train", args.train_files, seeds=(0, 1), selection_seed=17)
            validation = make_plan(
                root, "val", args.selection_files + args.report_files, seeds=(0,), selection_seed=43
            )
            # Randomly split the frozen sample, not lexicographic early/late file names.
            np.random.default_rng(91).shuffle(validation["recordings"])
            plans = {
                "train": train,
                "selection": {
                    **validation,
                    "recordings": validation["recordings"][: args.selection_files],
                },
                "report": {
                    **validation,
                    "recordings": validation["recordings"][args.selection_files :],
                },
            }
            write_json(directory / "plans.json", plans)
            write_json(settings_path, settings)
        selection_hashes = {r["sha256"] for r in plans["selection"]["recordings"]}
        if selection_hashes.intersection(r["sha256"] for r in plans["report"]["recordings"]):
            raise ValueError("selection and reporting content overlap")
        if not args.evaluate_only:
            print("Collecting/verifying complete training cache", flush=True)
            cached = build_cache(
                directory / "cache", root, plans["train"], receiver, interface, workers=args.workers
            )
            print(f"Training cache: {cached['samples']:,} causal examples", flush=True)
        policies, runs = [], []
        for encoder in args.encoders:
            for seed in args.seeds:
                name = f"{encoder}-{seed}"
                run = directory / name
                if args.evaluate_only:
                    state = json.loads((run / "state.json").read_text())
                    if state["latest"]["epoch"] < args.epochs:
                        raise ValueError("requested study training has not completed")
                    verify_artifacts(run, state["best"])
                    verify_artifacts(run, state["latest"])
                    path = checkpoint_path(run, state["best"], "policy.bin")
                    model, _, _ = load_predictor(path)
                    training = json.loads(
                        checkpoint_path(run, state["latest"], "training.json").read_text()
                    )
                    policies.append(predictor_spec(path, name=name, device="cuda"))
                    runs.append(
                        {
                            "name": name,
                            "best_epoch": state["best"]["epoch"],
                            "best_selection_score": state["best"]["score"],
                            "training_samples": training["total_samples"],
                            "parameters": sum(p.numel() for p in model.parameters()),
                        }
                    )
                    del model
                    continue
                print(f"Training {name}: {args.epochs} full cache passes on CUDA", flush=True)
                torch.cuda.reset_peak_memory_stats()
                learner = PredictorLearner(
                    root,
                    plans["train"],
                    plans["selection"],
                    PredictorConfig(
                        seed=seed,
                        batch_size=args.batch_size,
                        history_steps=16,
                        hidden=args.hidden,
                        encoder=encoder,
                    ),
                    receiver,
                    interface,
                    device="cuda",
                    cache=directory / "cache",
                )
                state = run_experiment(
                    run,
                    learner,
                    RunConfig(
                        epochs=args.epochs, selection=("selection", "capture_discovery_harmonic")
                    ),
                    resume=(run / "state.json").exists(),
                )
                verify_artifacts(run, state["best"])
                path = checkpoint_path(run, state["best"], "policy.bin")
                policies.append(predictor_spec(path, name=name, device="cuda"))
                runs.append(
                    {
                        "name": name,
                        "best_epoch": state["best"]["epoch"],
                        "best_selection_score": state["best"]["score"],
                        "training_samples": learner.samples,
                        "parameters": sum(p.numel() for p in learner.model.parameters()),
                    }
                )
                del learner
                gc.collect()
                torch.cuda.empty_cache()
        checkpoint = checkpoint_path(
            directory / runs[0]["name"],
            json.loads((directory / runs[0]["name"] / "state.json").read_text())["best"],
            "policy.bin",
        )
        policies.extend(
            [
                predictor_spec(
                    checkpoint, name="constant-coverage", device="cuda", constant_predictions=True
                ),
                predictor_spec(checkpoint, name="observed-rate", device="cuda", observed_rate=True),
                PolicySpec("sweep-10ms", partial(ReferencePolicy, "sweep", 1), "dwell=10ms"),
                PolicySpec("sweep-50ms", partial(ReferencePolicy, "sweep", 2), "dwell=50ms"),
                PolicySpec("random-50ms", partial(ReferencePolicy, "random", 2), "dwell=50ms"),
            ]
        )
        print("Paired evaluation on reporting recordings excluded from selection", flush=True)
        report = benchmark_policies(
            root,
            plans["report"],
            policies,
            baseline="observed-rate",
            receiver=receiver,
            interface=interface,
            inference_batch_size=32,
        )
        report["study_runs"] = runs
        # Commit expensive episode results before secondary statistical aggregation.
        write_json(directory / "comparison.json", report)
        report["across_training_seeds"] = aggregate(report, args.encoders, args.seeds)
        report["scope"] = "development validation; test split remains untouched"
        write_json(directory / "comparison.json", report)
        print(json.dumps(report["across_training_seeds"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
