"""CPU contract checks for batched scoring and public receiver calibration."""

from dataclasses import dataclass

import pytest

pytest.importorskip("torch")

from spectra_scheduler.evaluation_contract import Forecast  # noqa: E402
from spectra_scheduler.experiments.calibrated_timing import _CalibratedRatio  # noqa: E402
from spectra_scheduler.experiments.timing_report import (  # noqa: E402
    BatchedEpisode,
    reporting_world,
    reward,
)
from spectra_scheduler.experiments.timing_study import ObservedRateScheduler  # noqa: E402
from spectra_scheduler.synthetic_evaluation import evaluate_scheduler  # noqa: E402
from spectra_scheduler.timing_belief import BeliefPolicyConfig  # noqa: E402


@dataclass
class ConstantForecast:
    def forecast(self, time_step, action):
        return Forecast(0.6, 0.003, 0.4)


class CalibratedConstant(_CalibratedRatio, ConstantForecast):
    pass


def test_ratio_uses_only_public_detection_response():
    scheduler = CalibratedConstant()
    with pytest.raises(RuntimeError, match="configure"):
        scheduler.forecast(0, None)
    for probability in (0, -1, float("nan"), 1.1):
        with pytest.raises(ValueError):
            scheduler.set_detection_probability(probability)
    scheduler.set_detection_probability(0.9)
    prediction = scheduler.forecast(0, None)
    assert prediction.interception_ratio == pytest.approx(0.36)
    assert prediction.hit_probability == 0.6
    assert prediction.intercept_delay_seconds == 0.003


def test_batched_macro_scoring_matches_shared_evaluator_with_retuning():
    simulation, _ = reporting_world("periodic-scan", 2001)
    config = BeliefPolicyConfig(revisit=128, probe=4)
    expected = evaluate_scheduler(
        simulation,
        ObservedRateScheduler(config),
        step_seconds=0.001,
        reward=reward,
        reward_description="observed_hit - 0.05 * retuning",
    )
    state = BatchedEpisode(simulation, ObservedRateScheduler(config))
    while state.episode.time_step < simulation.duration:
        state.advance(state.scheduler.choose_action(state.episode.time_step), None)
    assert state.report() == expected
    assert any(not record.observation.listening for record in state.episode.records)
