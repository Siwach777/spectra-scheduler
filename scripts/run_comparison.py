import argparse

from spectra_scheduler.comparison import (
    print_comparison,
    print_repeated_comparison,
    run_comparison,
    run_repeated_comparison,
)
from spectra_scheduler.scenarios import SCENARIO_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scan strategies on a generated scenario")
    parser.add_argument("--runs", type=int, default=1, help="number of seeded runs")
    parser.add_argument("--seed", type=int, default=0, help="first random seed")
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_NAMES,
        default="mixed",
        help="generated evaluation scenario",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel worker processes for repeated runs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.runs == 1:
        print_comparison(run_comparison(args.seed, args.scenario))
    else:
        print_repeated_comparison(
            run_repeated_comparison(
                args.runs,
                args.seed,
                args.workers,
                args.scenario,
            )
        )
