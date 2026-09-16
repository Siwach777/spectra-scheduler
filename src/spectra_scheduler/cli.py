import argparse
from collections.abc import Sequence

from spectra_scheduler.comparison import (
    print_comparison,
    print_repeated_comparison,
    print_track_evaluation,
    run_comparison,
    run_repeated_comparison,
    run_repeated_track_evaluation,
)
from spectra_scheduler.reports import (
    build_experiment_report,
    write_experiment_report,
)
from spectra_scheduler.scenarios import SCENARIO_NAMES


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare scan strategies on a generated scenario"
    )
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
    parser.add_argument(
        "--association",
        action="store_true",
        help="also summarize track association quality",
    )
    parser.add_argument(
        "--output",
        help="write a complete experiment report to this JSON or CSV path",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        dest="output_format",
        help="report format; otherwise inferred from the output path",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.output:
        report = build_experiment_report(
            runs=args.runs,
            start_seed=args.seed,
            workers=args.workers,
            scenario=args.scenario,
        )
        print_repeated_comparison(report.strategy_results)
        print_track_evaluation(report.track_association)
        path = write_experiment_report(
            report,
            args.output,
            args.output_format,
        )
        print(f"Report written to {path}")
        return 0

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
    if args.association:
        print_track_evaluation(
            run_repeated_track_evaluation(
                args.runs,
                args.seed,
                args.workers,
                args.scenario,
            )
        )
    return 0
