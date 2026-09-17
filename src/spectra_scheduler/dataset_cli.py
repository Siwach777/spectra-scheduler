"""Offline dataset commands, independent of the simulator command interface."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main(arguments=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect TSRD files or benchmark pulse clustering")
    parser.add_argument("command", choices=("inspect", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("data/tsrd"))
    parser.add_argument("--mode", choices=("scan", "stare"), default="scan")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-files", type=int, default=10, help="0 selects all completed files")
    parser.add_argument("--batch-rows", type=int, default=65536)
    parser.add_argument(
        "--sample-rows", type=int, default=20000, help="maximum clustered pulses per file"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", choices=("raw", "signature"), default="signature")
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--profile", action="store_true", help="include runtime and process peak RSS"
    )
    parser.add_argument("--output", type=Path, help="JSON report (stdout otherwise)")
    args = parser.parse_args(arguments)
    if args.output and args.output.suffix.lower() != ".json":
        parser.error("--output must be a .json report, not an input pulse file")
    try:
        from spectra_scheduler.dataset_evaluation import DatasetConfig, run_dataset
    except ImportError:
        parser.error("dataset dependencies missing; install with: uv sync --extra dataset")
    try:
        config = DatasetConfig(
            **{name: getattr(args, name) for name in DatasetConfig.__dataclass_fields__}
        )
        report = run_dataset(
            args.root,
            config,
            evaluate=args.command == "evaluate",
            workers=args.workers,
            profile=args.profile,
        )
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=args.output.parent,
                    prefix=".dataset-report-",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                os.replace(temporary, args.output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            print(payload, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except ImportError:
        parser.error("dataset dependencies missing; install with: uv sync --extra dataset")
    summary = report["summary"]
    print(
        f"Validated {summary['valid_files']}/{summary['selected_files']} files; "
        f"{summary['validated_pulses']} pulses; scored {summary['scored_files']} files; "
        f"errors {summary['error_files']}",
        file=sys.stderr,
    )
    if summary["macro_metrics"] is not None:
        metrics = summary["macro_metrics"]
        print(
            f"File-macro V-measure {metrics['v_measure']:.4f}; "
            f"pairwise F1 {metrics['pairwise_f1']:.4f} "
            "(sampled pulses; noise treated as one cluster)",
            file=sys.stderr,
        )
    return (
        1
        if summary["error_files"] or (args.command == "evaluate" and not summary["scored_files"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
