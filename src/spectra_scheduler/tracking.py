from dataclasses import dataclass, field

from spectra_scheduler.models import Observation, SignalMeasurement


@dataclass
class SignalTrack:
    """A receiver-derived signal track with no simulator identity."""

    track_id: int
    mean_power_dbm: float
    mean_pulse_width_us: float
    last_band: int
    last_time: int
    observation_count: int = 1
    previous_band: int | None = None
    previous_time: int | None = None

    def update(
        self,
        time_step: int,
        band: int,
        measurement: SignalMeasurement,
    ) -> None:
        self.previous_band = self.last_band
        self.previous_time = self.last_time
        self.last_band = band
        self.last_time = time_step
        self.observation_count += 1
        weight = 1.0 / self.observation_count
        self.mean_power_dbm += weight * (
            measurement.power_dbm - self.mean_power_dbm
        )
        self.mean_pulse_width_us += weight * (
            measurement.pulse_width_us - self.mean_pulse_width_us
        )

    def predicted_band(self, time_step: int, num_bands: int) -> int:
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.previous_band is None or self.previous_time is None:
            return self.last_band

        elapsed = self.last_time - self.previous_time
        if elapsed <= 0:
            return self.last_band
        velocity = (self.last_band - self.previous_band) / elapsed
        prediction_steps = max(0, time_step - self.last_time)
        predicted = round(self.last_band + velocity * prediction_steps)
        return min(max(predicted, 0), num_bands - 1)


@dataclass
class SignalTracker:
    """Associate detections using measured power and pulse width."""

    pulse_width_tolerance_us: float = 0.2
    power_tolerance_db: float = 8.0
    max_age_steps: int = 10
    _tracks: list[SignalTrack] = field(init=False, default_factory=list)
    _next_track_id: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.pulse_width_tolerance_us <= 0:
            raise ValueError("pulse_width_tolerance_us must be positive")
        if self.power_tolerance_db <= 0:
            raise ValueError("power_tolerance_db must be positive")
        if self.max_age_steps <= 0:
            raise ValueError("max_age_steps must be positive")

    @property
    def tracks(self) -> tuple[SignalTrack, ...]:
        return tuple(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 0

    def update(self, observation: Observation) -> tuple[SignalTrack, ...]:
        self._tracks = [
            track
            for track in self._tracks
            if observation.time_step - track.last_time <= self.max_age_steps
        ]
        if not observation.listening:
            return self.tracks

        available_track_ids = {track.track_id for track in self._tracks}
        for measurement in observation.measurements:
            track = self._best_match(measurement, available_track_ids)
            if track is None:
                track = self._new_track(observation, measurement)
            else:
                track.update(observation.time_step, observation.band, measurement)
                available_track_ids.remove(track.track_id)
        return self.tracks

    def _best_match(
        self,
        measurement: SignalMeasurement,
        available_track_ids: set[int],
    ) -> SignalTrack | None:
        candidates = []
        for track in self._tracks:
            if track.track_id not in available_track_ids:
                continue
            width_difference = abs(
                measurement.pulse_width_us - track.mean_pulse_width_us
            )
            power_difference = abs(measurement.power_dbm - track.mean_power_dbm)
            if (
                width_difference <= self.pulse_width_tolerance_us
                and power_difference <= self.power_tolerance_db
            ):
                score = (
                    width_difference / self.pulse_width_tolerance_us
                    + power_difference / self.power_tolerance_db
                )
                candidates.append((score, track.track_id, track))
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: candidate[:2])[2]

    def _new_track(
        self,
        observation: Observation,
        measurement: SignalMeasurement,
    ) -> SignalTrack:
        track = SignalTrack(
            track_id=self._next_track_id,
            mean_power_dbm=measurement.power_dbm,
            mean_pulse_width_us=measurement.pulse_width_us,
            last_band=observation.band,
            last_time=observation.time_step,
        )
        self._next_track_id += 1
        self._tracks.append(track)
        return track
