from dataclasses import dataclass

from spectra_scheduler.models import SimulationResult


@dataclass(frozen=True)
class ScanMetrics:
    total_transmissions: int
    eligible_transmissions: int
    detected_transmissions: int
    probability_of_detection: float
    interception_ratio: float
    false_alarms: int
    false_alarm_rate: float
    hit_rate: float
    detected_emitters: int
    total_emitters: int
    mean_first_detection_delay: float


def calculate_metrics(result: SimulationResult) -> ScanMetrics:
    detected_transmissions = sum(
        len(observation.detected_emitters) for observation in result.observations
    )
    hit_steps = sum(observation.hit for observation in result.observations)
    false_alarms = sum(observation.false_alarm for observation in result.observations)

    tuned_band_by_time = {
        observation.time_step: observation.band for observation in result.observations
    }
    eligible_transmissions = sum(
        tuned_band_by_time.get(event.time_step) == event.band for event in result.transmissions
    )
    active_observation_slots = {
        (event.time_step, event.band) for event in result.transmissions
    }
    inactive_observations = sum(
        (observation.time_step, observation.band) not in active_observation_slots
        for observation in result.observations
    )

    first_transmission: dict[str, int] = {}
    for transmission in result.transmissions:
        first_transmission.setdefault(transmission.emitter_id, transmission.time_step)

    first_detection: dict[str, int] = {}
    for observation in result.observations:
        for emitter_id in observation.detected_emitters:
            first_detection.setdefault(emitter_id, observation.time_step)

    delays = [
        first_detection.get(emitter_id, result.duration) - first_time
        for emitter_id, first_time in first_transmission.items()
    ]

    total_transmissions = len(result.transmissions)
    return ScanMetrics(
        total_transmissions=total_transmissions,
        eligible_transmissions=eligible_transmissions,
        detected_transmissions=detected_transmissions,
        probability_of_detection=(
            detected_transmissions / eligible_transmissions if eligible_transmissions else 0.0
        ),
        interception_ratio=(
            detected_transmissions / total_transmissions if total_transmissions else 0.0
        ),
        false_alarms=false_alarms,
        false_alarm_rate=(false_alarms / inactive_observations if inactive_observations else 0.0),
        hit_rate=hit_steps / len(result.observations) if result.observations else 0.0,
        detected_emitters=len(first_detection),
        total_emitters=len(first_transmission),
        mean_first_detection_delay=sum(delays) / len(delays) if delays else 0.0,
    )
