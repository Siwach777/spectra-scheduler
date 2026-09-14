from dataclasses import dataclass

from spectra_scheduler.models import SimulationResult


@dataclass(frozen=True)
class ScanMetrics:
    total_transmissions: int
    eligible_transmissions: int
    detectable_transmissions: int
    sensitivity_misses: int
    sensitivity_loss_rate: float
    detected_transmissions: int
    probability_of_detection: float
    interception_ratio: float
    false_alarms: int
    false_alarm_rate: float
    hit_rate: float
    detected_emitters: int
    total_emitters: int
    emitter_discovery_ratio: float
    mean_first_detection_delay: float
    total_emitter_changes: int
    reacquired_changes: int
    reacquisition_ratio: float
    mean_reacquisition_delay: float
    max_band_gap: int


def calculate_metrics(result: SimulationResult) -> ScanMetrics:
    detected_transmissions = sum(
        len(record.detected_emitters) for record in result.detection_records
    )
    hit_steps = sum(observation.hit for observation in result.observations)
    false_alarms = sum(record.false_alarm for record in result.detection_records)
    detectable_transmissions = sum(
        len(record.detectable_emitters) for record in result.detection_records
    )

    tuned_band_by_time = {
        observation.time_step: observation.band for observation in result.observations
    }
    eligible_transmissions = sum(
        tuned_band_by_time.get(event.time_step) == event.band for event in result.transmissions
    )
    inactive_observations = sum(
        not record.detectable_emitters for record in result.detection_records
    )

    first_transmission: dict[str, int] = {}
    for transmission in result.transmissions:
        first_transmission.setdefault(transmission.emitter_id, transmission.time_step)

    first_detection: dict[str, int] = {}
    for record in result.detection_records:
        for emitter_id in record.detected_emitters:
            first_detection.setdefault(emitter_id, record.observation.time_step)

    delays = [
        first_detection.get(emitter_id, result.duration) - first_time
        for emitter_id, first_time in first_transmission.items()
    ]

    reacquisition_delays: list[int] = []
    reacquired_changes = 0
    for change in result.emitter_changes:
        detection_time = next(
            (
                record.observation.time_step
                for record in result.detection_records
                if record.observation.time_step >= change.time_step
                and change.emitter_id in record.detected_emitters
            ),
            None,
        )
        if detection_time is None:
            reacquisition_delays.append(result.duration - change.time_step)
        else:
            reacquired_changes += 1
            reacquisition_delays.append(detection_time - change.time_step)

    total_transmissions = len(result.transmissions)
    total_emitters = len(first_transmission)
    sensitivity_misses = eligible_transmissions - detectable_transmissions
    max_band_gap = 0
    for band in range(result.num_bands):
        last_visit = -1
        for observation in result.observations:
            if observation.band == band:
                max_band_gap = max(max_band_gap, observation.time_step - last_visit - 1)
                last_visit = observation.time_step
        max_band_gap = max(max_band_gap, result.duration - last_visit - 1)

    return ScanMetrics(
        total_transmissions=total_transmissions,
        eligible_transmissions=eligible_transmissions,
        detectable_transmissions=detectable_transmissions,
        sensitivity_misses=sensitivity_misses,
        sensitivity_loss_rate=(
            sensitivity_misses / eligible_transmissions if eligible_transmissions else 0.0
        ),
        detected_transmissions=detected_transmissions,
        probability_of_detection=(
            detected_transmissions / detectable_transmissions
            if detectable_transmissions
            else 0.0
        ),
        interception_ratio=(
            detected_transmissions / total_transmissions if total_transmissions else 0.0
        ),
        false_alarms=false_alarms,
        false_alarm_rate=(false_alarms / inactive_observations if inactive_observations else 0.0),
        hit_rate=hit_steps / len(result.observations) if result.observations else 0.0,
        detected_emitters=len(first_detection),
        total_emitters=total_emitters,
        emitter_discovery_ratio=(len(first_detection) / total_emitters if total_emitters else 0.0),
        mean_first_detection_delay=sum(delays) / len(delays) if delays else 0.0,
        total_emitter_changes=len(result.emitter_changes),
        reacquired_changes=reacquired_changes,
        reacquisition_ratio=(
            reacquired_changes / len(result.emitter_changes) if result.emitter_changes else 0.0
        ),
        mean_reacquisition_delay=(
            sum(reacquisition_delays) / len(reacquisition_delays)
            if reacquisition_delays
            else 0.0
        ),
        max_band_gap=max_band_gap,
    )
