from collections import Counter, defaultdict
from dataclasses import dataclass

from spectra_scheduler.models import SimulationResult
from spectra_scheduler.tracking import SignalTracker


@dataclass(frozen=True)
class ScanMetrics:
    total_transmissions: int
    listening_steps: int
    retuning_steps: int
    retuning_fraction: float
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


@dataclass(frozen=True)
class TrackMetrics:
    assigned_measurements: int
    confirmed_tracks: int
    mixed_tracks: int
    association_purity: float
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    detected_emitters: int
    mean_tracks_per_emitter: float


def calculate_metrics(result: SimulationResult) -> ScanMetrics:
    listening_steps = sum(observation.listening for observation in result.observations)
    retuning_steps = len(result.observations) - listening_steps
    detected_transmissions = sum(
        len(record.detected_emitters) for record in result.detection_records
    )
    hit_steps = sum(observation.hit for observation in result.observations)
    false_alarms = sum(record.false_alarm for record in result.detection_records)
    detectable_transmissions = sum(
        len(record.detectable_emitters) for record in result.detection_records
    )

    tuned_band_by_time = {
        observation.time_step: observation.band
        for observation in result.observations
        if observation.listening
    }
    eligible_transmissions = sum(
        tuned_band_by_time.get(event.time_step) == event.band for event in result.transmissions
    )
    inactive_observations = sum(
        record.observation.listening and not record.detectable_emitters
        for record in result.detection_records
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
            if observation.listening and observation.band == band:
                max_band_gap = max(max_band_gap, observation.time_step - last_visit - 1)
                last_visit = observation.time_step
        max_band_gap = max(max_band_gap, result.duration - last_visit - 1)

    return ScanMetrics(
        total_transmissions=total_transmissions,
        listening_steps=listening_steps,
        retuning_steps=retuning_steps,
        retuning_fraction=(
            retuning_steps / len(result.observations) if result.observations else 0.0
        ),
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
        hit_rate=hit_steps / listening_steps if listening_steps else 0.0,
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


def calculate_track_metrics(
    result: SimulationResult,
    tracker: SignalTracker | None = None,
) -> TrackMetrics:
    tracker = SignalTracker() if tracker is None else tracker
    tracker.reset()
    labels_by_track: dict[int, list[tuple[str, str]]] = defaultdict(list)
    track_ids_by_emitter: dict[str, set[int]] = defaultdict(set)

    for record in result.detection_records:
        tracker.update(record.observation)
        labels = [("emitter", emitter_id) for emitter_id in record.detected_emitters]
        if record.false_alarm:
            false_alarm_id = f"{record.observation.time_step}:{record.observation.band}"
            labels.append(("false-alarm", false_alarm_id))
        if len(labels) != len(tracker.last_assignments):
            raise ValueError("detection truth and track assignments do not align")

        for assignment in tracker.last_assignments:
            label = labels[assignment.measurement_index]
            labels_by_track[assignment.track_id].append(label)
            if label[0] == "emitter":
                track_ids_by_emitter[label[1]].add(assignment.track_id)

    assigned_measurements = sum(len(labels) for labels in labels_by_track.values())
    correctly_grouped = sum(
        max(Counter(labels).values(), default=0) for labels in labels_by_track.values()
    )
    true_label_counts = Counter(
        label for labels in labels_by_track.values() for label in labels
    )
    correctly_joined_pairs = sum(
        _pair_count(label_count)
        for labels in labels_by_track.values()
        for label_count in Counter(labels).values()
    )
    joined_pairs = sum(
        _pair_count(len(labels)) for labels in labels_by_track.values()
    )
    true_pairs = sum(_pair_count(count) for count in true_label_counts.values())
    pairwise_precision = (
        correctly_joined_pairs / joined_pairs if joined_pairs else 0.0
    )
    pairwise_recall = correctly_joined_pairs / true_pairs if true_pairs else 0.0
    pairwise_f1 = (
        2.0
        * pairwise_precision
        * pairwise_recall
        / (pairwise_precision + pairwise_recall)
        if pairwise_precision + pairwise_recall
        else 0.0
    )
    confirmed_tracks = sum(len(labels) >= 2 for labels in labels_by_track.values())
    mixed_tracks = sum(
        len(set(labels)) > 1 for labels in labels_by_track.values() if len(labels) >= 2
    )
    tracks_per_emitter = [len(track_ids) for track_ids in track_ids_by_emitter.values()]
    return TrackMetrics(
        assigned_measurements=assigned_measurements,
        confirmed_tracks=confirmed_tracks,
        mixed_tracks=mixed_tracks,
        association_purity=(
            correctly_grouped / assigned_measurements if assigned_measurements else 0.0
        ),
        pairwise_precision=pairwise_precision,
        pairwise_recall=pairwise_recall,
        pairwise_f1=pairwise_f1,
        detected_emitters=len(track_ids_by_emitter),
        mean_tracks_per_emitter=(
            sum(tracks_per_emitter) / len(tracks_per_emitter)
            if tracks_per_emitter
            else 0.0
        ),
    )


def _pair_count(item_count: int) -> int:
    return item_count * (item_count - 1) // 2
