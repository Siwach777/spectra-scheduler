import json

import pytest

from spectra_scheduler.evaluation_contract import EvaluationAccumulator, Forecast, TruthOutcome


def test_hand_calculated_metrics_and_censored_forecasts():
    metrics = EvaluationAccumulator()
    metrics.add(TruthOutcome(2, 10, 5, 4, 2, 0.5, False, False), 3, Forecast(0.8, 0.75, 0.3))
    metrics.add(TruthOutcome(1, 2, 1, 0, 0, None, True, True), -1, Forecast(0.2, None, 0.1))
    report = metrics.report()
    assert report["probability_of_detection"] == 0.5
    assert report["probability_of_false_alarm"] == 1
    assert report["sensitivity_loss_fraction"] == pytest.approx(2 / 6)
    assert report["average_intercept_rate_per_second"] == pytest.approx(2 / 3)
    assert report["interception_ratio"] == pytest.approx(2 / 12)
    assert report["average_reward_per_action"] == 1
    p = report["prediction"]
    assert p["percentage_correct"] == 100
    assert p["brier_score"] == pytest.approx(0.04)
    assert p["interception_ratio_mae"] == pytest.approx(0.1)
    assert p["average_intercept_time_error_seconds"] == 0.25
    assert p["restricted_intercept_time_mae_seconds"] == 0.125
    assert p["censored_windows"] == 1
    json.dumps(report, allow_nan=False)


def test_abstentions_cannot_hide_timing_failures_and_empty_truth_is_undefined():
    metrics = EvaluationAccumulator()
    metrics.add(TruthOutcome(2, 1, 1, 1, 1, 0.5), 0, Forecast(0.1, None, 0))
    metrics.add(TruthOutcome(2, 0, 0, 0, 0, None), 0, Forecast(0.8, 0.2, 0))
    metrics.add(TruthOutcome(2, 0, 0, 0, 0, None), 0)
    p = metrics.report()["prediction"]
    assert p["coverage"] == pytest.approx(2 / 3)
    assert p["confusion"] == dict(tp=0, tn=0, fp=1, fn=1)
    assert p["ratio_windows"] == 1
    assert p["average_intercept_time_error_seconds"] is None
    assert p["timing_event_coverage"] == 0
    assert p["restricted_intercept_time_mae_seconds"] == pytest.approx(1.65)
    assert metrics.report()["probability_of_false_alarm"] is None


def test_empty_report_and_missing_predictions_are_not_zero_error():
    report = EvaluationAccumulator().report()
    for field in (
        "probability_of_detection",
        "probability_of_false_alarm",
        "average_intercept_rate_per_second",
        "average_reward_per_action",
    ):
        assert report[field] is None
    assert report["prediction"]["percentage_correct"] is None
    assert report["prediction"]["average_intercept_time_error_seconds"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "values",
    [(float("nan"), 1, 0), (0.5, -1, 0), (0.5, float("inf"), 0), (0.5, 1, 1.1), (True, 1, 0)],
)
def test_invalid_forecasts(values):
    with pytest.raises(ValueError):
        Forecast(*values)


@pytest.mark.parametrize(
    "changes",
    [
        dict(elapsed_seconds=0),
        dict(truth_count=0),
        dict(eligible_count=0),
        dict(captured_count=2),
        dict(first_intercept_seconds=None),
        dict(first_intercept_seconds=1),
        dict(false_alarm=True),
        dict(truth_count=True),
    ],
)
def test_invalid_truth(changes):
    values = dict(
        elapsed_seconds=1,
        truth_count=1,
        eligible_count=1,
        detectable_count=1,
        captured_count=1,
        first_intercept_seconds=0,
    )
    with pytest.raises(ValueError):
        TruthOutcome(**(values | changes))


def test_validation_is_atomic_and_probability_threshold_is_inclusive():
    metrics = EvaluationAccumulator()
    outcome = TruthOutcome(1, 1, 1, 1, 1, 0)
    with pytest.raises(ValueError):
        metrics.add(outcome, float("nan"))
    assert metrics.report()["windows"] == 0
    metrics.add(outcome, 1, Forecast(0.5, 4, 1))
    p = metrics.report()["prediction"]
    assert p["percentage_correct"] == 100
    assert p["average_intercept_time_error_seconds"] == 4
    assert p["restricted_intercept_time_mae_seconds"] == 1


def test_synthetic_adapter_uses_pre_action_prediction_and_observable_reward():
    from spectra_scheduler.emitters import PeriodicEmitter
    from spectra_scheduler.receiver import Receiver
    from spectra_scheduler.simulation import Simulation
    from spectra_scheduler.synthetic_evaluation import evaluate_scheduler

    class Policy:
        def reset(self, num_bands):
            self.seen = 0

        def choose_band(self, time_step):
            return 0

        def forecast(self, time_step, band):
            assert self.seen == time_step
            return Forecast(1, 0, 1)

        def observe(self, observation):
            assert not hasattr(observation, "detected_emitters")
            self.seen += 1

    simulation = Simulation(
        1, 4, (PeriodicEmitter("a", 0, 2),), Receiver(false_alarm_probability=1)
    )
    report = evaluate_scheduler(
        simulation,
        Policy(),
        step_seconds=0.5,
        reward=lambda obs: float(obs.hit),
        reward_description="hit",
    )
    assert report["probability_of_detection"] == 1
    assert report["probability_of_false_alarm"] == 1
    assert report["average_intercept_rate_per_second"] == 1
    assert report["prediction"]["percentage_correct"] == 50
    assert report["average_reward_per_action"] == 1  # Includes false alarms.
    assert report["sensitivity"]["unit"] == "simulated_dbm"
