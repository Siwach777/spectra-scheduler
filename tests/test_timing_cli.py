"""CLI report and listening-dwell checks, without neural model validation."""

import csv
import json

from spectra_scheduler.cli import parse_args
from spectra_scheduler.experiments.timing_report import reporting_world
from spectra_scheduler.simulation import SimulationEpisode, configure_scheduler
from spectra_scheduler.timing_cli import (
    ListeningRoundRobin,
    run_timing_comparison,
    write_timing_report,
)


def test_round_robin_completes_listening_dwell_after_retune():
    world, _ = reporting_world("periodic-scan", 24000)
    scheduler = ListeningRoundRobin(10)
    configure_scheduler(world, scheduler)
    episode = SimulationEpisode(world)
    first = episode.step_action(scheduler.choose_action(0))
    second = episode.step_action(scheduler.choose_action(episode.time_step))
    assert sum(o.listening for o in first) == 10
    assert sum(o.listening for o in second) == 10
    assert len(second) > 10
    assert {o.band for o in second} == {1}


def test_reports_preserve_undefined_metrics_and_paired_intervals(tmp_path):
    metric = {"mean": None, "groups": 0, "paired_groups": 0,
              "paired_mean_difference": None, "paired_bootstrap_95_interval": None}
    report = {"summary": {"round-robin": {"interception_ratio": metric}}}
    json_path, csv_path = tmp_path / "run.json", tmp_path / "run.csv"
    write_timing_report(report, json_path)
    write_timing_report(report, csv_path)
    assert json.loads(json_path.read_text()) == report
    with csv_path.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["mean"] == ""
    assert row["paired_mean_difference"] == ""


def test_missing_checkpoint_is_rejected_before_cuda_initialization(tmp_path):
    import pytest
    args = parse_args(["--timing-model", str(tmp_path / "missing.pt")])
    with pytest.raises(ValueError, match="checkpoint does not exist"):
        run_timing_comparison(args)


def test_output_cannot_overwrite_checkpoint(tmp_path):
    import pytest
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"preserve")
    args = parse_args(["--timing-model", str(checkpoint), "--output", str(checkpoint),
                       "--format", "json"])
    with pytest.raises(ValueError, match="overwrite"):
        run_timing_comparison(args)
    assert checkpoint.read_bytes() == b"preserve"
