"""Apply the common metric contract to the existing discrete-time simulator."""

from collections import Counter
from collections.abc import Callable
from math import isfinite

from .evaluation_contract import EvaluationAccumulator, Forecast, TruthOutcome
from .models import Observation
from .simulation import Simulation, SimulationEpisode, SyntheticAction, configure_scheduler


def evaluate_completed_episode(
    simulation: Simulation,
    result,
    *,
    step_seconds: float,
    reward_sum: float,
    reward_description: str,
):
    """Score an existing tick-policy run without executing the receiver again."""
    if not isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be positive and finite")
    if not isinstance(reward_description, str) or not reward_description.strip():
        raise ValueError("a reward description is required")
    if not isfinite(reward_sum):
        raise ValueError("reward_sum must be finite")
    if result.duration != simulation.duration or len(result.detection_records) != result.duration:
        raise ValueError("result does not cover the simulation horizon")
    totals = Counter(event.time_step for event in result.transmissions)
    cells = Counter((event.time_step, event.band) for event in result.transmissions)
    evaluation = EvaluationAccumulator()
    for index, record in enumerate(result.detection_records):
        observation = record.observation
        if observation.time_step != index:
            raise ValueError("result observations are not in time order")
        captured = len(record.detected_emitters)
        evaluation.add(
            TruthOutcome(
                elapsed_seconds=step_seconds,
                truth_count=totals[index],
                eligible_count=cells[index, observation.band] if observation.listening else 0,
                detectable_count=len(record.detectable_emitters),
                captured_count=captured,
                first_intercept_seconds=0.0 if captured else None,
                negative_opportunity=observation.listening and not record.detectable_emitters,
                false_alarm=record.false_alarm,
            ),
            reward_sum if index == 0 else 0.0,
        )
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


def evaluate_scheduler(
    simulation: Simulation,
    scheduler,
    *,
    step_seconds: float,
    reward: Callable[[Observation], float],
    reward_description: str,
    truth=None,
):
    """Evaluate tick or macro actions with causal, pre-action forecasts.

    Tick policies use choose_band()/forecast(time_step, band). Macro policies use
    choose_action()/forecast(time_step, action), where action is SyntheticAction.
    Forecasts run before receiver feedback and return Forecast or None.
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
    configure_scheduler(simulation, scheduler)
    predict = getattr(scheduler, "forecast", None)
    macro = hasattr(scheduler, "choose_action")
    while episode.time_step < simulation.duration:
        time_step = episode.time_step
        if macro:
            action = scheduler.choose_action(time_step)
            if not isinstance(action, SyntheticAction):
                raise ValueError("choose_action() must return SyntheticAction")
            band = action.band
            if not 0 <= band < simulation.num_bands:
                raise ValueError("action band outside receiver range")
            forecast = predict(time_step, action) if predict is not None else None
        else:
            band = scheduler.choose_band(time_step)
            forecast = predict(time_step, band) if predict is not None else None
        if forecast is not None and not isinstance(forecast, Forecast):
            raise ValueError("forecast() must return Forecast or None")
        observations = episode.step_action(action) if macro else (episode.step(band),)
        records = episode.records[-len(observations) :]
        captured = sum(len(record.detected_emitters) for record in records)
        negative_ticks = sum(
            record.observation.listening and not record.detectable_emitters for record in records
        )
        false_alarms = sum(record.false_alarm for record in records)
        first = next(
            (record.observation.time_step for record in records if record.detected_emitters), None
        )
        elapsed = len(observations)
        outcome = TruthOutcome(
            elapsed_seconds=step_seconds * elapsed,
            truth_count=sum(totals[t] for t in range(time_step, episode.time_step)),
            eligible_count=sum(
                len(episode.events.get((record.observation.time_step, band), ()))
                for record in records
                if record.observation.listening
            ),
            detectable_count=sum(len(record.detectable_emitters) for record in records),
            captured_count=captured,
            first_intercept_seconds=(
                (first - time_step) * step_seconds if first is not None else None
            ),
            **(
                {"negative_opportunities": negative_ticks, "false_alarms": false_alarms}
                if macro
                else {
                    "negative_opportunity": bool(negative_ticks),
                    "false_alarm": bool(false_alarms),
                }
            ),
        )
        window_reward = 0.0
        for observation in observations:
            window_reward += reward(observation)
            scheduler.observe(observation)
        evaluation.add(outcome, window_reward, forecast)
    report = evaluation.report()
    if macro:
        report.update(
            contract_version=2,
            target="synthetic_selected_band_listening_dwell_including_retune",
            action_contract="SyntheticAction(band, dwell_steps): listening ticks after retune",
        )
        support = report["false_alarm_support"]
        report["false_alarm_support"] = {
            "modelled_action_windows": support["modelled_windows"],
            "negative_listening_ticks": support["negative_windows"],
            "false_alarm_ticks": support["false_positive_windows"],
        }
    from .metrics import calculate_metrics

    scan = calculate_metrics(episode.result())
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
        discovery={
            "emitter_discovery_ratio": scan.emitter_discovery_ratio,
            "mean_first_detection_delay_seconds": scan.mean_first_detection_delay * step_seconds,
            "reacquisition_ratio": scan.reacquisition_ratio,
            "mean_reacquisition_delay_seconds": scan.mean_reacquisition_delay * step_seconds,
            "total_emitter_changes": scan.total_emitter_changes,
            "reacquired_changes": scan.reacquired_changes,
        },
    )
    return report
