# ruff: noqa: E402
import io
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from spectra_scheduler.experiments import RunConfig, run_experiment
from spectra_scheduler.experiments.predictor import PredictorConfig, PredictorLearner
from spectra_scheduler.experiments.storage import checkpoint_path, load_torch
from spectra_scheduler.policy_benchmark import make_plan
from spectra_scheduler.pulse_replay import ReplayConfig
from spectra_scheduler.replay_env import InterfaceConfig


@pytest.fixture
def plans(tmp_path):
    for split, offset in (("train", 0), ("val", 1), ("test", 2)):
        directory = tmp_path / "data" / "stare" / f"{split}_stare"
        directory.mkdir(parents=True)
        with h5py.File(directory / "tiny.h5", "w") as f:
            f["data"] = np.array(
                [[i, 5 if i % 2 else 15, 0.1, offset, -20] for i in range(12)], dtype=float
            )
            f["metadata/feature_names"] = np.array(
                ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S"
            )
    root = tmp_path / "data"
    return root, {s: make_plan(root, s, max_files=1, seeds=(0,)) for s in ("train", "val", "test")}


def learner(plans):
    root, plans = plans
    return PredictorLearner(
        root,
        plans["train"],
        plans["val"],
        PredictorConfig(
            batch_size=2, history_steps=2, hidden=4, time_bins=3, max_batches_per_epoch=2
        ),
        ReplayConfig(stop_us=12, max_frequency_mhz=20, bandwidth_mhz=10, retune_us=1),
        InterfaceConfig(bands=2, dwell_us=(2, 4)),
        device="cpu",  # Small deterministic lifecycle unit check, not an experiment.
    )


def test_resume_equals_uninterrupted_optimizer_and_model(plans, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    a = run_experiment(full, learner(plans), RunConfig(2), stream=io.StringIO())
    run_experiment(resumed, learner(plans), RunConfig(1), stream=io.StringIO())
    b = run_experiment(resumed, learner(plans), RunConfig(2), resume=True, stream=io.StringIO())
    x = load_torch(checkpoint_path(full, a["latest"], "training.bin"))
    y = load_torch(checkpoint_path(resumed, b["latest"], "training.bin"))
    for key in x["model"]:
        torch.testing.assert_close(x["model"][key], y["model"][key], rtol=0, atol=0)
    for param in x["optimizer"]["state"]:
        for key in x["optimizer"]["state"][param]:
            torch.testing.assert_close(
                x["optimizer"]["state"][param][key],
                y["optimizer"]["state"][param][key],
                rtol=0,
                atol=0,
            )
    assert x["updates"] == y["updates"] == 4
    assert a["best"]["score"] == b["best"]["score"]


def test_no_test_selection_or_overlap(plans):
    root, p = plans
    with pytest.raises(ValueError, match="test is forbidden"):
        PredictorLearner(root, p["train"], p["test"])
    # Copying a training recording into validation is caught by content provenance.
    import shutil

    shutil.copyfile(
        root / p["train"]["recordings"][0]["path"], root / p["val"]["recordings"][0]["path"]
    )
    val = make_plan(root, "val", max_files=1, seeds=(0,))
    with pytest.raises(ValueError, match="overlap"):
        PredictorLearner(root, p["train"], val)


def test_cli_end_to_end_on_fixture(plans, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("training CLI requires CUDA")
    from spectra_scheduler.experiments.__main__ import main

    root, p = plans
    (tmp_path / "train.json").write_text(json.dumps(p["train"]))
    (tmp_path / "val.json").write_text(json.dumps(p["val"]))
    settings = {
        "root": str(root),
        "train_plan": "train.json",
        "validation_plan": "val.json",
        "predictor": {
            "batch_size": 2,
            "history_steps": 2,
            "hidden": 4,
            "time_bins": 3,
            "max_batches_per_epoch": 1,
        },
        "receiver": {"stop_us": 12, "max_frequency_mhz": 20, "bandwidth_mhz": 10},
        "interface": {"bands": 2, "dwell_us": [2, 4]},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings))
    args = ["--config", str(path), "--run-dir", str(tmp_path / "cli")]
    assert main(args) == 0
    assert main([*args, "--resume", "--epochs", "2"]) == 0
