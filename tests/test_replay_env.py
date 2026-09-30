# Dataset dependency must be checked before importing the replay modules.
# ruff: noqa: E402
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from spectra_scheduler.pulse_replay import ReplayConfig
from spectra_scheduler.replay_env import (
    InterfaceConfig,
    ReplayEnv,
    tune_centers,
    validate_specification,
)
from spectra_scheduler.replay_evaluation import ReferencePolicy, benchmark, evaluate_policy, main


@pytest.fixture
def recording(tmp_path):
    directory = tmp_path / "stare" / "val_stare"
    directory.mkdir(parents=True)
    path = directory / "example.h5"
    with h5py.File(path, "w") as f:
        f["data"] = np.array(
            [[i, 5 if i % 2 else 15, 0.1, 359, -20] for i in range(30)], dtype=float
        )
        f["metadata/feature_names"] = np.array(
            ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"], dtype="S"
        )
        f["labels"] = np.arange(30) % 2
    return path


def settings():
    return (
        ReplayConfig(stop_us=30, max_frequency_mhz=20, bandwidth_mhz=10, retune_us=5),
        InterfaceConfig(bands=2, dwell_us=(5, 10), reference_us=10),
    )


def test_reset_reward_time_discount_and_terminal(recording):
    receiver, interface = settings()
    with ReplayEnv(recording, receiver, interface) as env:
        initial = env.reset()
        first = env.step(1)
        assert first.reward == pytest.approx(5 / 1000)
        assert first.discount == pytest.approx(0.99)
        assert first.elapsed_us == 10
        second = env.step(3)
        assert second.elapsed_us == 15
        assert second.discount == pytest.approx(0.99**1.5)
        assert second.reward == pytest.approx(5 / 1000 - 0.05)
        terminal = env.step(3)
        assert terminal.terminated and terminal.discount == 0
        assert np.isfinite(terminal.observation).all()
        assert initial.dtype == np.float32
        assert initial.shape == (env.observation_size,)
        np.testing.assert_array_equal(initial, env.reset())
        np.testing.assert_array_equal(first.observation, env.step(1).observation)


def test_overlapping_centers_match_dense_labels_and_slew(recording):
    from spectra_scheduler.experiments.dense_replay import DenseReplayIndex

    receiver = ReplayConfig(
        stop_us=30,
        max_frequency_mhz=20,
        bandwidth_mhz=10,
        retune_us=0,
        slew_mhz_per_us=3,
    )
    interface = InterfaceConfig(bands=3, dwell_us=(10,), reference_us=10)
    np.testing.assert_array_equal(tune_centers(receiver, interface), [5, 10, 15])
    index = DenseReplayIndex(recording, receiver, interface)
    labels = index.outcomes(0, None)
    with ReplayEnv(recording, receiver, interface) as env:
        env.reset()
        env.step(1)
        assert env.metrics()["intercepted_pulses"] == labels["captured"][1] == 5
        after = env.step(2)
        expected = index.outcomes(10, 1)
        assert after.elapsed_us == expected["elapsed_us"][2] == pytest.approx(11 + 2 / 3)
        assert env.evaluation_outcome().captured_count == expected["captured"][2]


def test_labels_do_not_change_features_or_rewards(recording):
    receiver, interface = settings()
    with ReplayEnv(recording, receiver, interface) as env:
        env.reset()
        before = env.step(1)
    with h5py.File(recording, "r+") as f:
        f["labels"][:] = 900
    with ReplayEnv(recording, receiver, interface) as env:
        env.reset()
        after = env.step(1)
    np.testing.assert_array_equal(before.observation, after.observation)
    assert before.reward == after.reward


@pytest.mark.parametrize("action", [-1, 4, 1.2, True, float("nan")])
def test_invalid_actions(recording, action):
    with ReplayEnv(recording, *settings()) as env:
        env.reset()
        with pytest.raises(ValueError):
            env.step(action)


@pytest.mark.parametrize(
    "kwargs",
    [{"bands": 0}, {"dwell_us": ()}, {"dwell_us": (1, 1)}, {"gamma": 0}, {"reference_us": -1}],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        InterfaceConfig(**kwargs)


def test_inference_runner_and_paired_worker_parity(recording, tmp_path):
    receiver, interface = settings()
    report = evaluate_policy(recording, ReferencePolicy(), receiver, interface)
    assert report["complete"]
    assert report["truth_pulses"] == 30
    assert report["specification"]["action_count"] == 4
    serial = benchmark(tmp_path, max_files=1, receiver=receiver, interface=interface)
    parallel = benchmark(tmp_path, max_files=1, workers=2, receiver=receiver, interface=interface)
    for name in ("sweep", "random"):
        a, b = serial["results"][0][name], parallel["results"][0][name]
        for key in ("reward_sum", "delivered_pulses", "retuning_us", "discovery_fraction"):
            assert a[key] == b[key]
    assert serial["bootstrap_95_interval"] is None


def test_cli_report(recording, tmp_path):
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "--max-files",
                "1",
                "--stop-us",
                "30",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["split"] == "val"
    assert report["results"][0]["sweep"]["complete"]


def test_checkpoint_specification_compatibility(recording):
    with ReplayEnv(recording, *settings()) as env:
        spec = env.specification()
        saved = json.loads(json.dumps(spec))
        saved["receiver"]["seed"] = 999
        validate_specification(saved, spec)
        saved["feature_version"] = -1
        with pytest.raises(ValueError, match="incompatible"):
            validate_specification(saved, spec)


def test_evaluation_closes_stream_on_policy_failure(recording, monkeypatch):
    from spectra_scheduler.pulse_replay import PulseReplay

    closed = []
    original = PulseReplay.close

    def close(replay):
        original(replay)
        closed.append(replay._closed)

    class BrokenPolicy:
        def reset(self, specification, seed):
            pass

        def act(self, observation):
            raise RuntimeError("inference failed")

    monkeypatch.setattr(PulseReplay, "close", close)
    with pytest.raises(RuntimeError, match="inference failed"):
        evaluate_policy(recording, BrokenPolicy(), *settings())
    assert closed == [True]


def test_forecasts_are_evaluated_for_selected_action_before_feedback(recording):
    from spectra_scheduler.evaluation_contract import Decision, Forecast

    class ForecastPolicy(ReferencePolicy):
        def act(self, observation):
            # First decision receives no future pulse information.
            if self.step == 0:
                assert not observation[:-2].any()
            return Decision(super().act(observation), Forecast(1, 0, 0.5))

    report = evaluate_policy(recording, ForecastPolicy(), *settings())
    metrics = report["evaluation"]
    assert metrics["windows"] == 3
    # Third action reaches episode end during retuning and has no listening window.
    assert metrics["counts"] == dict(truth=30, eligible=10, detectable=10, captured=10)
    assert metrics["probability_of_detection"] == 1
    assert metrics["probability_of_false_alarm"] is None
    assert metrics["false_alarm_status"] == "not_modelled"
    assert metrics["prediction"]["coverage"] == 1
    assert metrics["prediction"]["percentage_correct"] == pytest.approx(200 / 3)
    assert metrics["average_intercept_rate_per_second"] == pytest.approx(10 / 0.00003)
    assert metrics["prediction"]["average_intercept_time_error_seconds"] == pytest.approx(
        (1 + 6) / 2 / 1e6
    )
    json.dumps(report, allow_nan=False)


def test_legacy_policy_reports_missing_forecasts(recording):
    report = evaluate_policy(recording, ReferencePolicy(), *settings())
    p = report["evaluation"]["prediction"]
    assert p["coverage"] == 0
    assert p["percentage_correct"] is None
    assert p["average_intercept_time_error_seconds"] is None


def test_truth_is_not_added_to_transition_and_reset_clears_last_outcome(recording):
    with ReplayEnv(recording, *settings()) as env:
        env.reset()
        with pytest.raises(RuntimeError, match="no completed"):
            env.evaluation_outcome()
        transition = env.step(1)
        assert not hasattr(transition, "truth_count")
        assert env.evaluation_outcome().truth_count == 10
        env.reset()
        with pytest.raises(RuntimeError, match="no completed"):
            env.evaluation_outcome()
