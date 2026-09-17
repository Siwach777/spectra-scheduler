"""Train and evaluate portable observation-only scheduling models."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from spectra_scheduler.learned_scheduler import HitModel
from spectra_scheduler.learning import TRAIN_SCENARIOS, evaluate_model, train_model
from spectra_scheduler.scenarios import SCENARIO_NAMES


def write_json(path: Path, payload: dict):
    if path.suffix != ".json":
        raise ValueError("output must have a .json suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        child = subcommands.add_parser(command)
        child.add_argument("--runs", type=int, default=100 if command == "train" else 50)
        child.add_argument("--seed", type=int, default=0 if command == "train" else 10000)
        child.add_argument(
            "--scenarios",
            nargs="+",
            choices=SCENARIO_NAMES,
            default=TRAIN_SCENARIOS if command == "train" else SCENARIO_NAMES,
        )
        child.add_argument("--output", type=Path, required=True)
        if command == "train":
            child.add_argument("--max-examples", type=int, default=50000)
        else:
            child.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix != ".json":
            raise ValueError("output must have a .json suffix")
        if args.command == "train":
            model = train_model(args.runs, args.seed, args.scenarios, args.max_examples)
            payload = model.to_dict()
            summary = f"Fitted model using {model.training['examples_used']} listening examples"
        else:
            if args.model.resolve() == args.output.resolve():
                raise ValueError("report output must not overwrite the model")
            payload = evaluate_model(
                HitModel.load(args.model), args.runs, args.seed, args.scenarios
            )
            summary = f"Evaluated {args.runs} held-out seeds per scenario"
        write_json(args.output, payload)
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"{summary}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
