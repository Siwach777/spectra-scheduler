import argparse
from collections.abc import Sequence
from pathlib import Path

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
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--scenario",
        choices=SCENARIO_NAMES,
        default="mixed",
        help="generated evaluation scenario",
    )
    scenario_group.add_argument(
        "--scenario-file",
        help="JSON definition for a custom generated scenario",
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
    parser.add_argument(
        "--timing-model", "--timing-checkpoint", dest="timing_checkpoint", type=Path,
        help="include a trained timing planner checkpoint (CUDA required)",
    )
    parser.add_argument(
        "--mpc-model", dest="mpc_checkpoints", type=Path, action="append", default=[],
        help="include a saved physical MPC checkpoint in the timing comparison; repeatable",
    )
    parser.add_argument(
        "--inference-batch-size", type=int, default=20,
        help="maximum simultaneous timing-model episodes",
    )
    args = parser.parse_args(arguments)
    if min(args.runs, args.workers, args.inference_batch_size) < 1:
        parser.error("runs, workers and inference batch size must be positive")
    if args.mpc_checkpoints and not args.timing_checkpoint:
        parser.error("--mpc-model requires --timing-model")
    if args.timing_checkpoint and args.association:
        parser.error("--association is available through the comparison without --timing-model")
    return args


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.timing_checkpoint:
        import os
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
        from spectra_scheduler.timing_cli import run_timing_comparison
        return run_timing_comparison(args)
    if args.output:
        report = build_experiment_report(
            runs=args.runs,
            start_seed=args.seed,
            workers=args.workers,
            scenario=args.scenario,
            scenario_file=args.scenario_file,
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
        print_comparison(
            run_comparison(args.seed, args.scenario, args.scenario_file)
        )
    else:
        print_repeated_comparison(
            run_repeated_comparison(
                args.runs,
                args.seed,
                args.workers,
                args.scenario,
                args.scenario_file,
            )
        )
    if args.association:
        print_track_evaluation(
            run_repeated_track_evaluation(
                args.runs,
                args.seed,
                args.workers,
                args.scenario,
                args.scenario_file,
            )
        )
    return 0
