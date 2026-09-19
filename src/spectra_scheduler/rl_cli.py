"""Train an RL policy or benchmark frozen models against shared-world baselines."""

import argparse
import sys
from pathlib import Path

from spectra_scheduler.learned_scheduler import HitModel
from spectra_scheduler.learning_cli import write_json
from spectra_scheduler.rl import RewardConfig, RLModel, TrainConfig, train
from spectra_scheduler.rl_benchmark import SUITES, benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--episodes", type=int, default=1500)
    training.add_argument(
        "--seed", type=int, default=0, help="network, exploration and replay seed"
    )
    training.add_argument("--coverage-penalty", type=float, default=0.05)
    training.add_argument("--retune-penalty", type=float, default=0.05)
    training.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("benchmark")
    evaluation.add_argument("--models", type=Path, nargs="+", required=True)
    evaluation.add_argument("--hit-model", type=Path)
    evaluation.add_argument("--runs", type=int, default=30)
    evaluation.add_argument("--seed", type=int, default=10000)
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluation.add_argument("--suites", choices=SUITES, nargs="+", default=SUITES)
    evaluation.add_argument("--profile", action="store_true")
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix != ".json":
            raise ValueError("output must be a .json file")
        if args.command == "train":
            config = TrainConfig(episodes=args.episodes, seed=args.seed)
            reward = RewardConfig(retuning=args.retune_penalty, coverage=args.coverage_penalty)

            def progress(row):
                print(
                    f"episode {row['episode']}/{config.episodes}; "
                    f"steps {row['environment_steps']}; "
                    f"reward/step {row['mean_training_reward_per_step']:.4f}; "
                    f"epsilon {row['epsilon']:.3f}",
                    flush=True,
                )

            payload = train(config, reward, progress).to_dict()
        else:
            inputs = args.models + ([args.hit_model] if args.hit_model else [])
            if args.output.resolve() in [path.resolve() for path in inputs]:
                raise ValueError("report must not overwrite an input model")

            def progress(suite, means):
                value = means["rl-0"]["interception_ratio"]
                print(f"{suite}: rl-0 interception {value:.2%}", flush=True)

            payload = benchmark(
                [RLModel.load(p) for p in args.models],
                args.runs,
                args.seed,
                args.split,
                args.suites,
                HitModel.load(args.hit_model) if args.hit_model else None,
                args.profile,
                progress,
            )
        write_json(args.output, payload)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
