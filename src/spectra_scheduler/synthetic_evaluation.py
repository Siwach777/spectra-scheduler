"""Apply the common metric contract to the existing discrete-time simulator."""

from collections import Counter
from collections.abc import Callable
from math import isfinite

from .evaluation_contract import EvaluationAccumulator, Forecast, TruthOutcome
from .models import Observation
from .simulation import Simulation, SimulationEpisode


def evaluate_scheduler(
    simulation: Simulation,
    scheduler,
    *,
    step_seconds: float,
    reward: Callable[[Observation], float],
    reward_description: str,
    truth=None,
):
    """Run a legacy scheduler, optionally implementing forecast(time_step, band).

    forecast() runs after choose_band() and before receiver feedback. It returns a
    Forecast or None and only has access to scheduler-owned observation history.
    Rewards consume Observation, never truth. Explicit step duration avoids silently
    converting arbitrary simulator steps to seconds. Instantiate stateful rewards
    afresh for each episode.
    """
    if not isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be positive and finite")
    if not isinstance(reward_description, str) or not reward_description.strip():
        raise ValueError("a reward description is required")
    episode = SimulationEpisode(simulation, truth=truth)
    totals = Counter(event.time_step for event in episode.transmissions)
    evaluation = EvaluationAccumulator()
    scheduler.reset(simulation.num_bands)
    predict = getattr(scheduler, "forecast", None)
    for time_step in range(simulation.duration):
        band = scheduler.choose_band(time_step)
        forecast = predict(time_step, band) if predict is not None else None
        if forecast is not None and not isinstance(forecast, Forecast):
            raise ValueError("forecast() must return Forecast or None")
        observation = episode.step(band)
        record = episode.records[-1]
        captured = len(record.detected_emitters)
        outcome = TruthOutcome(
            elapsed_seconds=step_seconds,
            truth_count=totals[time_step],
            eligible_count=(
                len(episode.events.get((time_step, band), ())) if observation.listening else 0
            ),
            detectable_count=len(record.detectable_emitters),
            captured_count=captured,
            first_intercept_seconds=0.0 if captured else None,
            negative_opportunity=observation.listening and not record.detectable_emitters,
            false_alarm=record.false_alarm,
        )
        evaluation.add(outcome, reward(observation), forecast)
        scheduler.observe(observation)
    report = evaluation.report()
    report.update(
        backend="synthetic_discrete",
        step_seconds=step_seconds,
        reward_description=reward_description,
        sensitivity={
            "threshold": simulation.receiver.sensitivity_dbm,
            "unit": "simulated_dbm",
            "status": "configured",
            "calibrated_dbm": False,
        },
        false_alarm_status="modelled",
    )
    return report
