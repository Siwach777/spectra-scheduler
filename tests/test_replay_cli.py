"""CLI safeguards and fixed-sweep compatibility; neural integration runs on CUDA."""

import json

import h5py
import numpy as np
import pytest

from spectra_scheduler.replay_cli import main


def recording(root):
    path = root / "stare/train_stare/config_0.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle["data"] = np.array([[100, 1125, 1, 0, -20], [2100, 3375, 1, 0, -20]])
        handle["labels"] = np.array([0, 1])
        handle["metadata/feature_names"] = np.array(
            ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S"
        )
    return path


def test_fixed_sweep_still_runs_without_a_model(tmp_path, capsys):
    root = tmp_path / "data"
    recording(root)
    output = tmp_path / "report.json"
    assert main(["--root", str(root), "--stop-us", "3000", "--dwell-us", "1000",
                 "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text())
    assert report["policy"] == "fixed-sweep"
    assert report["truth_pulses"] == 2
    assert report["simulated_us"] == 3000


@pytest.mark.parametrize("target", ["dataset", "checkpoint"])
def test_output_cannot_overwrite_inputs(tmp_path, capsys, target):
    root = tmp_path / "data"
    source = recording(root)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"preserve checkpoint")
    output = source if target == "dataset" else checkpoint
    before = output.read_bytes()
    with pytest.raises(SystemExit, match="1"):
        main(["--root", str(root), "--timing-model", str(checkpoint),
              "--retune-us", "2000", "--output", str(output)])
    assert "must not overwrite" in capsys.readouterr().err
    assert output.read_bytes() == before


def test_fractional_timing_is_rejected_before_loading_model(capsys):
    with pytest.raises(SystemExit, match="1"):
        main(["--timing-model", "missing.pt"])
    assert "whole 1-ms ticks" in capsys.readouterr().err


def test_comparison_requires_a_trained_policy(capsys):
    with pytest.raises(SystemExit, match="1"):
        main(["--compare-controls"])
    assert "requires timing-model" in capsys.readouterr().err
