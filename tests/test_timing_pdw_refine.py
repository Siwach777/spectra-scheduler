"""CPU checks for the causal external-data collector and checkpoint domain boundary."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from spectra_scheduler.experiments.timing_pdw_refine import collect_recording  # noqa: E402
from spectra_scheduler.pulse_replay import ReplayConfig  # noqa: E402
from spectra_scheduler.timing_belief import BeliefConfig, load_belief  # noqa: E402


def write_recording(path, label):
    with h5py.File(path, "w") as handle:
        handle["data"] = np.array([[toa, 100, 1, 0, -20] for toa in range(100, 12000, 100)],
                                  dtype=float)
        handle["metadata/feature_names"] = np.array(
            ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S")
        handle["labels"] = np.full(119, label, dtype=np.int64)


def test_pdw_collector_snapshots_are_pre_action_and_ignore_emitter_labels(tmp_path):
    receiver = ReplayConfig(stop_us=12000, retune_us=2000, detection_probability=1)
    config = BeliefConfig(history=16, future=8, max_period=8, width=4)
    path = tmp_path / "recording.h5"
    write_recording(path, 1)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = collect_recording((0, path, first_dir, config, receiver, 4, 0))
    write_recording(path, 9999)
    second = collect_recording((0, path, second_dir, config, receiver, 4, 0))
    arrays = {name: np.load(item["path"]) for name, item in first["arrays"].items()}
    for name, item in second["arrays"].items():
        np.testing.assert_array_equal(arrays[name], np.load(item["path"]))
    assert not arrays["history"][0].any()
    assert arrays["future"][0, 0, 0] == 9
    assert arrays["history"][1, 0, 1, -1] == 9
    assert not arrays["history"][:, :, 2].any()
    assert (arrays["history"][:, :, 1] * (1 - arrays["history"][:, :, 0]) == 0).all()
    assert not arrays["valid"][-1].all()


def test_external_count_checkpoint_cannot_load_as_synthetic_model(tmp_path):
    path = tmp_path / "pdw.pt"
    torch.save({"kind": "pdw_timing_belief", "version": 2}, path)
    with pytest.raises(ValueError, match="unsupported timing belief checkpoint"):
        load_belief(path)
