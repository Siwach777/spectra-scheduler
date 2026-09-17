import json

import pytest

from spectra_scheduler.learned_scheduler import (
    FEATURE_NAMES,
    HitModel,
    LearnedScheduler,
    ObservationHistory,
)
from spectra_scheduler.learning import (
    FeedbackCollector,
    FeedbackReservoir,
    evaluate_model,
    train_model,
)
from spectra_scheduler.learning_cli import main, write_json
from spectra_scheduler.models import Observation
from spectra_scheduler.schedulers import RoundRobinScheduler


def test_context_uses_only_past_listening_feedback():
    history = ObservationHistory(2)
    before = history.features(0, 0)
    history.update(Observation(0, 0, listening=False))
    assert history.visits == [0, 0]
    assert history.features(1, 0)[0] == before[0]
    history.update(Observation(1, 0, detections=1))
    assert history.features(2, 0)[0] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        history.features(1, 0)


def test_collector_excludes_retunes_and_current_target():
    reservoir = FeedbackReservoir(10, 0)
    collector = FeedbackCollector(RoundRobinScheduler(), reservoir)
    collector.reset(2)
    band = collector.choose_band(0)
    collector.observe(Observation(0, band, listening=False))
    assert reservoir.seen == 0
    band = collector.choose_band(1)
    collector.observe(Observation(1, band, detections=1))
    x, y = reservoir.arrays()
    assert x[0, 0] == 0.5
    assert y.tolist() == [1]


def test_reservoir_is_bounded_and_repeatable():
    a, b = FeedbackReservoir(5, 9), FeedbackReservoir(5, 9)
    for i in range(100):
        for reservoir in (a, b):
            reservoir.add((i,) * len(FEATURE_NAMES), i % 2)
    assert a.seen == 100
    assert a.arrays()[0].shape == (5, len(FEATURE_NAMES))
    assert (a.x == b.x).all()


def test_scheduler_waits_through_retune_and_minimum_dwell():
    scheduler = LearnedScheduler(HitModel((0.0,) * len(FEATURE_NAMES), 0.0, {}))
    scheduler.reset(2)
    for step, listening in enumerate((False, False, True, True)):
        assert scheduler.choose_band(step) == 0
        scheduler.observe(Observation(step, 0, listening=listening))
    assert scheduler.choose_band(4) == 1
    assert len(scheduler.targets) == 2


def test_single_band_scheduler():
    scheduler = LearnedScheduler(HitModel((0.0,) * len(FEATURE_NAMES), -1000.0, {}))
    scheduler.reset(1)
    for step in range(30):
        assert scheduler.choose_band(step) == 0
        scheduler.observe(Observation(step, 0))
    assert scheduler.predictions == [0.0] * 30


@pytest.fixture
def model():
    pytest.importorskip("sklearn")
    return train_model(runs=3, max_examples=100)


def test_artifact_round_trip_and_schema_validation(model, tmp_path):
    path = tmp_path / "model.json"
    write_json(path, model.to_dict())
    assert HitModel.load(path).fingerprint == model.fingerprint
    payload = model.to_dict()
    payload["feature_names"] = []
    write_json(path, payload)
    with pytest.raises(ValueError, match="schema"):
        HitModel.load(path)
    payload = model.to_dict()
    payload["coefficients"][0] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite"):
        HitModel.load(path)


def test_training_repeatable_and_evaluation_disjoint(model):
    assert train_model(runs=3, max_examples=100).fingerprint == model.fingerprint
    with pytest.raises(ValueError, match="overlap"):
        evaluate_model(model, runs=2, seed=2)
    report = evaluate_model(model, runs=2, scenarios=("crowded",))
    assert report == evaluate_model(model, runs=2, scenarios=("crowded",))
    result = report["scenarios"]["crowded"]
    assert result["unseen_scenario"]
    assert len(result["mean_metrics"]) == 15
    assert result["prediction"]["listening_examples"] > 0
    assert result["paired_interception_delta"]["constant-model"]["approximate_95pct_ci"]


def test_cli_protects_model_and_handles_invalid_arguments(model, tmp_path):
    path = tmp_path / "model.json"
    write_json(path, model.to_dict())
    assert main(["evaluate", "--model", str(path), "--output", str(path)]) == 1
    assert HitModel.load(path).fingerprint == model.fingerprint
    assert main(["train", "--runs", "0", "--output", str(path)]) == 1
    assert main(["train", "--output", str(tmp_path / "bad.h5")]) == 1
