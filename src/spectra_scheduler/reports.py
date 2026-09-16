import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from spectra_scheduler.comparison import (
    ComparisonStats,
    TrackStats,
    run_repeated_comparison,
    run_repeated_track_evaluation,
)

REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentReport:
    schema_version: int
    scenario: str
    start_seed: int
    runs: int
    strategy_results: dict[str, ComparisonStats]
    track_association: TrackStats


def build_experiment_report(
    runs: int,
    start_seed: int = 0,
    workers: int = 1,
    scenario: str = "mixed",
) -> ExperimentReport:
    strategy_results = run_repeated_comparison(
        runs=runs,
        start_seed=start_seed,
        workers=workers,
        scenario=scenario,
    )
    track_association = run_repeated_track_evaluation(
        runs=runs,
        start_seed=start_seed,
        workers=workers,
        scenario=scenario,
    )
    return ExperimentReport(
        schema_version=REPORT_SCHEMA_VERSION,
        scenario=scenario,
        start_seed=start_seed,
        runs=runs,
        strategy_results=strategy_results,
        track_association=track_association,
    )


def write_experiment_report(
    report: ExperimentReport,
    output_path: str | Path,
    output_format: str | None = None,
) -> Path:
    path = Path(output_path)
    report_format = _resolve_format(path, output_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    if report_format == "json":
        _write_json(report, path)
    else:
        _write_csv(report, path)
    return path


def _resolve_format(path: Path, output_format: str | None) -> str:
    if output_format is not None:
        normalized_format = output_format.lower()
        if normalized_format not in {"json", "csv"}:
            raise ValueError("output format must be 'json' or 'csv'")
        return normalized_format

    suffix = path.suffix.lower().lstrip(".")
    if suffix not in {"json", "csv"}:
        raise ValueError("report path must end in .json or .csv")
    return suffix


def _write_json(report: ExperimentReport, path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(asdict(report), output, indent=2, sort_keys=True)
        output.write("\n")


def _write_csv(report: ExperimentReport, path: Path) -> None:
    strategy_fields = [
        field.name for field in fields(ComparisonStats) if field.name != "runs"
    ]
    track_fields = [field.name for field in fields(TrackStats) if field.name != "runs"]
    fieldnames = [
        "schema_version",
        "scenario",
        "start_seed",
        "runs",
        "record_type",
        "name",
        *strategy_fields,
        *track_fields,
    ]
    base_row = {
        "schema_version": report.schema_version,
        "scenario": report.scenario,
        "start_seed": report.start_seed,
        "runs": report.runs,
    }
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in report.strategy_results.items():
            values = asdict(stats)
            values.pop("runs")
            writer.writerow(
                {
                    **base_row,
                    "record_type": "strategy",
                    "name": name,
                    **values,
                }
            )
        track_values = asdict(report.track_association)
        track_values.pop("runs")
        writer.writerow(
            {
                **base_row,
                "record_type": "track-association",
                "name": "track-aware",
                **track_values,
            }
        )
