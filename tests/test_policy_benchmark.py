import json
from functools import partial

import numpy as np
import pytest

from spectra_scheduler.policy_benchmark import (
    PolicySpec,
    benchmark_policies,
    benchmark_synthetic,
    make_plan,
    summarize,
    validate_plan,
)
from spectra_scheduler.pulse_replay import ReplayConfig
from spectra_scheduler.replay_env import InterfaceConfig
from spectra_scheduler.replay_evaluation import ReferencePolicy
from spectra_scheduler.schedulers import RoundRobinScheduler, ShuffledSweepScheduler


@pytest.fixture
def recordings(tmp_path):
    h5py = pytest.importorskip("h5py")
    for split in ("train", "val", "test"):
        directory = tmp_path / "stare" / f"{split}_stare"
        directory.mkdir(parents=True)
        for index in range(2):
            with h5py.File(directory / f"config_{index}.h5", "w") as f:
                f["data"] = np.array(
                    [[i, 5 + 10 * (i % 2), 0.1, index, -20] for i in range(20)], dtype=float
                )
                f["metadata/feature_names"] = np.array(
                    ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S"
                )
    return tmp_path


def settings():
    return dict(
        receiver=ReplayConfig(stop_us=20, max_frequency_mhz=20, bandwidth_mhz=10, retune_us=1),
        interface=InterfaceConfig(bands=2, dwell_us=(2, 4)),
    )


def policies():
    return [PolicySpec(k, partial(ReferencePolicy, k), k) for k in ("sweep", "random")]


def test_frozen_manifest_and_parallel_pairing(recordings):
    plan = make_plan(recordings, max_files=2, seeds=(0, 1))
    assert plan == make_plan(recordings, max_files=2, seeds=(0, 1))
    serial = benchmark_policies(recordings, plan, policies(), baseline="sweep", **settings())
    parallel = benchmark_policies(
        recordings, plan, policies(), baseline="sweep", workers=2, **settings()
    )
    assert serial["summary"] == parallel["summary"]
    assert len(serial["results"]) == 4
    assert serial["summary"]["random"]["interception_ratio"]["groups"] == 2
    json.dumps(serial, allow_nan=False)


def test_plan_rejects_wrong_split_changed_file_and_training_overlap(recordings):
    plan = make_plan(recordings, max_files=1)
    bad = json.loads(json.dumps(plan))
    bad["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        validate_plan(recordings, bad)
    trained = policies()
    trained[0] = PolicySpec("sweep", ReferencePolicy, "trained", (plan["recordings"][0]["sha256"],))
    with pytest.raises(ValueError, match="overlap"):
        benchmark_policies(recordings, plan, trained, baseline="sweep", **settings())
    plan["recordings"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="content changed"):
        validate_plan(recordings, plan)


def test_synthetic_parallel_and_split_isolation():
    specs = [
        PolicySpec("sweep", RoundRobinScheduler, "default"),
        PolicySpec("shuffled", ShuffledSweepScheduler, "default"),
    ]
    a = benchmark_synthetic(specs, baseline="sweep", seeds=(0, 1))
    b = benchmark_synthetic(specs, baseline="sweep", seeds=(0, 1), workers=2)
    assert a == b
    training = benchmark_synthetic(specs, baseline="sweep", seeds=(0, 1), split="train")
    assert {r["world_seed"] for r in a["results"]}.isdisjoint(
        {r["world_seed"] for r in training["results"]}
    )
    assert len(a["by_scenario"]) == 3


def test_bootstrap_groups_repeated_seeds_not_independent_samples():
    from spectra_scheduler.evaluation_contract import EvaluationAccumulator, TruthOutcome

    def metrics(captured):
        acc = EvaluationAccumulator()
        acc.add(TruthOutcome(1, 1, 1, 1, captured, 0 if captured else None), 0)
        return {"evaluation": acc.report()}

    rows = [
        {"group": "same-recording", "policies": {"sweep": metrics(0), "random": metrics(1)}}
        for _ in range(10)
    ]
    summary = summarize(rows, policies(), "sweep")["random"]["interception_ratio"]
    assert summary["paired_mean_difference"] == 1
    assert summary["paired_groups"] == 1
    assert summary["paired_bootstrap_95_interval"] is None


def test_training_batches_causal_masked_and_bounded(recordings):
    from spectra_scheduler.replay_training import BatchConfig, training_batches

    plan = make_plan(recordings, split="train", max_files=1, seeds=(0,))
    batches = training_batches(
        recordings,
        plan,
        ReferencePolicy,
        config=BatchConfig(batch_size=2, history_steps=3, time_bins=4),
        **settings(),
    )
    first = next(batches)
    assert first["history"].shape == (2, 3, 20)
    assert not first["history"][0, :, :-2].any()
    assert first["history"][0, -1, -1] == -1  # No previous band before first action.
    assert first["time_class"].tolist() == [0, 0]
    assert first["ratio"].tolist() == pytest.approx([0.5, 0.4])
    assert first["discount"][0] == pytest.approx(0.99 ** (4 / 10000))
    storage = first["history"].base
    second = next(batches)
    assert second["history"].base is storage
    last = list(batches)[-1]
    assert last["terminated"][-1]
    assert last["discount"][-1] == 0
    bad = make_plan(recordings, split="val", max_files=1)
    with pytest.raises(ValueError, match="train-only"):
        next(training_batches(recordings, bad, ReferencePolicy, **settings()))
